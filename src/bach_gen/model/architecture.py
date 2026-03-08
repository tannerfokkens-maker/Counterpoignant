"""Decoder-only Transformer for Bach invention generation.

Architecture: RoPE/PoPE + RMSNorm + SwiGLU + pre-norm + weight tying (mini-LLaMA).
Supports LoopLM (weight-tied recurrence) for parameter-efficient depth scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from bach_gen.model.config import ModelConfig


# ---------------------------------------------------------------------------
# LoopLM output container
# ---------------------------------------------------------------------------

@dataclass
class LoopLMOutput:
    """Return type when ``return_all_steps=True`` during LoopLM training."""

    logits: torch.Tensor  # Final-step logits (batch, seq_len, vocab_size)
    all_logits: list[torch.Tensor] = field(default_factory=list)
    exit_lambdas: list[torch.Tensor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

@dataclass
class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Stores K and V at the SDPA-ready stage:
    - K is post-RoPE/PoPE (position info baked in)
    - K/V are post-GQA expansion (already num_heads, not num_kv_heads)
    - V is post-PoPE zero-expansion (already 2*head_dim when PoPE active)

    K/V buffers are allocated to max_seq_len once and written in-place
    at each decoding step to avoid per-token reallocation.
    """

    k: torch.Tensor  # (batch, num_heads, max_seq_len, effective_dim)
    v: torch.Tensor  # (batch, num_heads, max_seq_len, effective_dim)
    pos_offset: int = 0  # absolute position of next token to be generated
    active_len: int | None = None  # number of valid cached positions

    @property
    def seq_len(self) -> int:
        """Number of cached positions."""
        if self.active_len is not None:
            return self.active_len
        return self.k.shape[2]


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Precomputes sin/cos caches that are lazily extended when a longer
    sequence is encountered.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self._max_cached = 0

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-build cache to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._max_cached:
            return
        self._max_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        # Replace buffers
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors of shape (seq_len, head_dim)."""
        self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def _rel_shift(x: torch.Tensor) -> torch.Tensor:
    """Skew absolute-by-relative logits into absolute-by-absolute indexing."""
    bsz, n_heads, q_len, rel_len = x.shape
    zero_pad = torch.zeros(
        (bsz, n_heads, q_len, 1), dtype=x.dtype, device=x.device,
    )
    x = torch.cat([zero_pad, x], dim=-1)
    x = x.reshape(bsz, n_heads, rel_len + 1, q_len)
    x = x[:, :, 1:, :]
    return x[:, :, :, :q_len]


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to *x*.

    Args:
        x: (B, H, T, D) query or key tensor.
        cos: (T, D) cosine cache.
        sin: (T, D) sine cache.

    Returns:
        Tensor of same shape with rotary embedding applied.
    """
    # Unsqueeze for broadcasting: (1, 1, T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # rotate_half: swap and negate first half
    d_half = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., d_half:], x[..., :d_half]], dim=-1)
    return x * cos + x_rot * sin


class PoPEEmbedding(nn.Module):
    """Polar Coordinate Position Embedding (PoPE).

    Unlike RoPE which pairs dimensions, PoPE treats each dimension
    independently: apply softplus to get a positive magnitude, then
    rotate by position-dependent angle. This decouples content (magnitude)
    from position (angle).

    Output Q/K vectors are 2x the input dimension.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim  # head_dim (not head_dim//2 like RoPE)
        self.theta = theta
        self._max_cached = 0

        # One frequency per dimension (not per pair)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._max_cached:
            return
        self._max_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        angles = torch.outer(t, self.inv_freq)  # (seq_len, dim)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_pope_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Apply PoPE to x: softplus -> rotate -> interleave to 2*D.

    Args:
        x: (B, H, T, D) query or key tensor.
        cos: (T, D) cosine cache.
        sin: (T, D) sine cache.

    Returns:
        (B, H, T, 2*D) tensor with PoPE applied.
    """
    mag = F.softplus(x)  # (B, H, T, D)

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)

    real = mag * cos  # (B, H, T, D)
    imag = mag * sin  # (B, H, T, D)

    # Interleave: [real_0, imag_0, real_1, imag_1, ...]
    out = torch.stack([real, imag], dim=-1)  # (B, H, T, D, 2)
    return out.reshape(*x.shape[:-1], 2 * x.shape[-1])  # (B, H, T, 2*D)


def apply_pope_no_pos(x: torch.Tensor) -> torch.Tensor:
    """Apply PoPE without position: softplus -> expand to 2*D with zero angles.

    Used during DroPE recalibration to maintain the attention dimension
    and nonlinearity while removing all positional information.

    This is equivalent to apply_pope_emb with cos=1, sin=0 everywhere.
    """
    mag = F.softplus(x)  # (B, H, T, D)
    zeros = torch.zeros_like(mag)

    out = torch.stack([mag, zeros], dim=-1)  # (B, H, T, D, 2)
    return out.reshape(*x.shape[:-1], 2 * x.shape[-1])


# ---------------------------------------------------------------------------
# Core modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Drops the mean-centering step of LayerNorm — normalizes by root mean
    square only.  Slightly faster, slightly more stable training.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# LoopLM exit-distribution helpers
# ---------------------------------------------------------------------------

def compute_exit_distribution(
    exit_lambdas: list[torch.Tensor],
    num_steps: int | None = None,
) -> torch.Tensor:
    """Build a valid exit-step distribution from per-step exit probabilities.

    Args:
        exit_lambdas: ``T-1`` tensors each shaped ``(batch, seq_len)``
            giving the instantaneous exit probability at steps ``0..T-2``.
            The final step's mass is implicit (remaining survival).
        num_steps: Total number of recurrent steps ``T``.  When ``None``,
            inferred as ``len(exit_lambdas) + 1``.

    Returns:
        ``(T, batch, seq_len)`` tensor that sums to 1 along dim-0.
        Steps ``0..T-2`` use ``lambda_t * S_{t-1}``; step ``T-1`` gets the
        remaining survival mass ``S_{T-1}``.
    """
    T_minus_1 = len(exit_lambdas)
    T = num_steps if num_steps is not None else T_minus_1 + 1

    if T_minus_1 == 0:
        # Single step — all mass on step 0
        shape = exit_lambdas[0].shape if exit_lambdas else (1, 1)
        device = exit_lambdas[0].device if exit_lambdas else torch.device("cpu")
        return torch.ones(1, *shape, device=device)

    # (T-1, B, S)
    lambdas = torch.stack(exit_lambdas, dim=0)

    # Survival S_t = prod_{j=0}^{t-1} (1 - lambda_j)
    one_minus = (1.0 - lambdas).clamp(min=1e-8)  # (T-1, B, S)
    log_surv = torch.cumsum(torch.log(one_minus), dim=0)  # (T-1, B, S)
    surv = torch.exp(log_surv)

    # S_prev: S_0=1, S_1, S_2, ...  (length T)
    S_prev = torch.cat([torch.ones_like(surv[:1]), surv], dim=0)  # (T, B, S)

    # p_exit for steps 0..T-2
    p_non_final = lambdas * S_prev[:T_minus_1]  # (T-1, B, S)

    # Final step collects remaining mass: S_{T-1}
    p_final = S_prev[T_minus_1:T]  # (1, B, S)

    return torch.cat([p_non_final, p_final], dim=0)  # (T, B, S)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class BachTransformer(nn.Module):
    """Small decoder-only Transformer for music generation.

    When ``config.num_recurrent_steps > 1``, the same layer stack is applied
    multiple times (LoopLM / Ouro architecture), trading unique parameters
    for effective depth.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional embedding (no learned positional params)
        if config.pos_encoding == "pope":
            self.pos_emb = PoPEEmbedding(
                dim=config.embed_dim // config.num_heads,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
        elif config.pos_encoding == "rope":
            self.pos_emb = RotaryEmbedding(
                dim=config.embed_dim // config.num_heads,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
        else:
            self.pos_emb = None

        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.ln_final = RMSNorm(config.embed_dim)

        # Output head
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying
        if config.weight_tying:
            self.head.weight = self.token_embed.weight

        # LoopLM exit gate (only when recurrence > 1 and gate enabled)
        if config.looplm_exit_gate and config.num_recurrent_steps > 1:
            self.exit_gate = nn.Linear(config.embed_dim, 1)
        else:
            self.exit_gate = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # RMSNorm weight is already initialized to ones in __init__

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_rope: bool = True,
        attn_temperature: float | None = None,
        use_cache: bool = False,
        kv_cache: list[KVCache] | None = None,
        return_all_steps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]] | LoopLMOutput:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) optional padding mask (1=attend, 0=ignore).
            use_rope: If False, skip rotary positional embeddings (for DroPE).
            attn_temperature: If set, scale attention logits by 1/beta* before
                softmax. Used at inference with DroPE models on extended contexts.
            use_cache: If True, return (logits, kv_caches) for incremental decoding.
            kv_cache: Per-layer KV caches from a previous forward pass.
            return_all_steps: If True and num_recurrent_steps > 1, return a
                ``LoopLMOutput`` with per-step logits and exit probabilities
                for the LoopLM training loss.

        Returns:
            logits: (batch, seq_len, vocab_size) — or (logits, kv_caches)
            when use_cache is True — or LoopLMOutput when return_all_steps
            is True.
        """
        T_max = self.config.num_recurrent_steps
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings only (no learned positional embedding)
        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        # Absolute position offset for incremental decoding
        pos_offset = kv_cache[0].pos_offset if kv_cache is not None else 0

        # Causal mask — not needed in incremental mode (new tokens attend to all)
        if kv_cache is not None:
            causal_mask = None
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1,
            )

        # Apply padding mask if provided
        if attention_mask is not None and kv_cache is None:
            # attention_mask: (batch, seq_len), 1=attend, 0=pad
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            pad_mask = pad_mask.expand(-1, -1, seq_len, -1)  # (B, 1, S, S)
        else:
            pad_mask = None

        # Compute positional cos/sin — slice to absolute positions for new tokens
        if use_rope and self.pos_emb is not None:
            total_len = pos_offset + seq_len
            cos, sin = self.pos_emb(total_len)
            cos = cos[pos_offset:pos_offset + seq_len].to(device)
            sin = sin[pos_offset:pos_offset + seq_len].to(device)
        else:
            cos, sin = None, None

        # ----- Standard path (no recurrence) -----
        if T_max <= 1:
            return self._forward_single(
                x, causal_mask, pad_mask, cos, sin, use_rope,
                attn_temperature, use_cache, kv_cache,
            )

        # ----- LoopLM recurrent path -----
        return self._forward_looped(
            x, causal_mask, pad_mask, cos, sin, use_rope,
            attn_temperature, use_cache, kv_cache,
            return_all_steps,
        )

    def _forward_single(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None,
        pad_mask: torch.Tensor | None,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        use_pos: bool,
        attn_temperature: float | None,
        use_cache: bool,
        kv_cache: list[KVCache] | None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        """Standard single-pass forward (no recurrence)."""
        new_caches: list[KVCache] = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            layer_out = layer(
                x, causal_mask=causal_mask, pad_mask=pad_mask,
                cos=cos, sin=sin, use_pos=use_pos,
                attn_temperature=attn_temperature,
                kv_cache=layer_cache, use_cache=use_cache,
            )
            if use_cache:
                x, layer_new_cache = layer_out
                new_caches.append(layer_new_cache)
            else:
                x = layer_out

        x = self.ln_final(x)
        logits = self.head(x)

        if use_cache:
            return logits, new_caches
        return logits

    def _forward_looped(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None,
        pad_mask: torch.Tensor | None,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        use_pos: bool,
        attn_temperature: float | None,
        use_cache: bool,
        kv_cache: list[KVCache] | None,
        return_all_steps: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]] | LoopLMOutput:
        """LoopLM recurrent forward: apply layer stack T_max times.

        KV-cache strategy (``last-step reuse`` from the Ouro paper):
        - Non-final recurrent steps READ from the cache but do NOT write.
        - The final recurrent step reads AND writes, updating the cache.
        This keeps memory cost identical to a standard transformer while
        allowing all recurrent steps to attend to previous positions.

        Adaptive early exit (inference only):
        When ``self.exit_gate`` is not None and the model is in eval mode,
        the loop exits early once cumulative exit mass exceeds
        ``config.looplm_exit_threshold`` for every token in the batch.
        """
        T_max = self.config.num_recurrent_steps

        all_logits: list[torch.Tensor] = []
        exit_lambdas: list[torch.Tensor] = []
        new_caches: list[KVCache] = []

        # Adaptive exit tracking (inference only)
        adaptive_exit = (
            not self.training
            and self.exit_gate is not None
            and not return_all_steps
            and self.config.looplm_exit_threshold < 1.0
        )
        cumulative_mass: torch.Tensor | None = None  # (B, S)
        exit_logits: torch.Tensor | None = None

        def run_recurrent_step(
            step_x: torch.Tensor,
            write_cache: bool,
        ) -> torch.Tensor:
            """Run one recurrent pass, optionally materializing KV cache."""
            for i, layer in enumerate(self.layers):
                layer_cache = kv_cache[i] if kv_cache is not None else None

                if write_cache:
                    layer_out = layer(
                        step_x, causal_mask=causal_mask, pad_mask=pad_mask,
                        cos=cos, sin=sin, use_pos=use_pos,
                        attn_temperature=attn_temperature,
                        kv_cache=layer_cache, use_cache=True,
                    )
                    step_x, lc = layer_out
                    if len(new_caches) <= i:
                        new_caches.append(lc)
                    else:
                        new_caches[i] = lc
                elif kv_cache is not None:
                    step_x = layer(
                        step_x, causal_mask=causal_mask, pad_mask=pad_mask,
                        cos=cos, sin=sin, use_pos=use_pos,
                        attn_temperature=attn_temperature,
                        kv_cache=layer_cache, use_cache=False,
                        cache_read_only=True,
                    )
                else:
                    step_x = layer(
                        step_x, causal_mask=causal_mask, pad_mask=pad_mask,
                        cos=cos, sin=sin, use_pos=use_pos,
                        attn_temperature=attn_temperature,
                    )
            return step_x

        for t in range(T_max):
            is_last = (t == T_max - 1)
            step_input = x
            x = run_recurrent_step(step_input, write_cache=use_cache and is_last)

            # --- Per-step logits & gate ---
            if return_all_steps:
                step_logits = self.head(self.ln_final(x))
                all_logits.append(step_logits)

                # Only compute gate for steps 0..T-2; final step gets
                # remaining survival mass — its lambda is unused.
                if self.exit_gate is not None and not is_last:
                    lam = torch.sigmoid(self.exit_gate(x)).squeeze(-1)  # (B, S)
                    exit_lambdas.append(lam)

            # --- Adaptive early exit at inference ---
            if adaptive_exit and not is_last and exit_logits is None:
                lam = torch.sigmoid(self.exit_gate(x)).squeeze(-1)  # (B, S)
                if cumulative_mass is None:
                    cumulative_mass = lam
                else:
                    # p(exit at t) = lam_t * S_{t-1};  cumulative CDF
                    cumulative_mass = cumulative_mass + lam * (1.0 - cumulative_mass)

                if (cumulative_mass >= self.config.looplm_exit_threshold).all():
                    exit_logits = self.head(self.ln_final(x))
                    if use_cache:
                        # Cache writes only happen once; rerun the triggering
                        # step with cache materialization so future decode steps
                        # match the early-exit depth.
                        x = run_recurrent_step(step_input, write_cache=True)
                    break

        # Final logits: use early-exit logits if available, else last step
        final_h = self.ln_final(x)
        final_logits = self.head(final_h)

        if return_all_steps:
            # Replace last entry with final_logits (avoids redundant ln_final + head)
            if all_logits:
                all_logits[-1] = final_logits
            return LoopLMOutput(
                logits=final_logits,
                all_logits=all_logits,
                exit_lambdas=exit_lambdas,
            )

        # Use early-exit logits when available (adaptive depth)
        output_logits = exit_logits if exit_logits is not None else final_logits

        if use_cache:
            return output_logits, new_caches
        return output_logits

    def count_parameters(self) -> int:
        """Count actual trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block with RMSNorm and SwiGLU.

    When ``config.looplm_sandwich_norm`` is True, an additional post-norm
    is applied after each sublayer output (before the residual add).  This
    ``sandwich normalization`` constrains representation growth across
    recurrent loops and is critical for stable deep-recurrence training.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.embed_dim)
        self.ffn = SwiGLUFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

        # Sandwich normalization (LoopLM stability)
        self.sandwich_norm = config.looplm_sandwich_norm
        if self.sandwich_norm:
            self.ln1_post = RMSNorm(config.embed_dim)
            self.ln2_post = RMSNorm(config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None,
        pad_mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        use_pos: bool = True,
        attn_temperature: float | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        cache_read_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        # Pre-norm attention
        attn_out = self.attn(
            self.ln1(x), causal_mask, pad_mask,
            cos=cos, sin=sin, use_pos=use_pos,
            attn_temperature=attn_temperature,
            kv_cache=kv_cache, use_cache=use_cache,
            cache_read_only=cache_read_only,
        )
        if use_cache:
            attn_out, new_cache = attn_out

        if self.sandwich_norm:
            attn_out = self.ln1_post(attn_out)

        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        ffn_out = self.ffn(self.ln2(x))
        if self.sandwich_norm:
            ffn_out = self.ln2_post(ffn_out)
        x = x + self.dropout(ffn_out)

        if use_cache:
            return x, new_cache
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_kv_heads = config.effective_num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.pos_encoding = config.pos_encoding
        self.max_seq_len = config.max_seq_len
        self.rel_attn_max_distance = config.rel_attn_max_distance
        self.rel_attn_dim = self.head_dim * (2 if config.pos_encoding == "pope" else 1)

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.embed_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.embed_dim, kv_dim)
        self.v_proj = nn.Linear(config.embed_dim, kv_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)
        self.rel_attn_bias = None
        if config.rel_attn_bias:
            self.rel_attn_bias = nn.Parameter(
                torch.empty(
                    self.num_heads,
                    self.rel_attn_max_distance,
                    self.rel_attn_dim,
                ),
            )
            nn.init.normal_(self.rel_attn_bias, mean=0.0, std=0.02)

    def _compute_relative_attention_bias(
        self,
        q: torch.Tensor,
        q_len: int,
        k_len: int,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor | None:
        """Return additive per-head relative logits."""
        if self.rel_attn_bias is None:
            return None

        if kv_cache is None and q_len == k_len and k_len <= self.rel_attn_max_distance:
            rel = self.rel_attn_bias[:, self.rel_attn_max_distance - k_len :, :]
            rel_logits = torch.einsum("bhtd,hrd->bhtr", q, rel)
            return _rel_shift(rel_logits)

        rel_logits = torch.einsum("bhtd,hrd->bhtr", q, self.rel_attn_bias)
        max_rel = self.rel_attn_max_distance - 1

        if kv_cache is None:
            query_positions = torch.arange(q_len, device=q.device)
            key_positions = torch.arange(k_len, device=q.device)
        else:
            cached_len = kv_cache.seq_len
            key_start = kv_cache.pos_offset - cached_len
            query_start = kv_cache.pos_offset
            query_positions = torch.arange(query_start, query_start + q_len, device=q.device)
            key_positions = torch.arange(key_start, key_start + k_len, device=q.device)

        rel_idx = key_positions[None, :] - query_positions[:, None]
        rel_idx = rel_idx.clamp(min=-max_rel, max=0) + max_rel
        rel_idx = rel_idx.to(torch.long)
        gather_idx = rel_idx.view(1, 1, q_len, k_len).expand(
            rel_logits.size(0), rel_logits.size(1), -1, -1,
        )
        return torch.gather(rel_logits, dim=-1, index=gather_idx)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None,
        pad_mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        use_pos: bool = True,
        attn_temperature: float | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        cache_read_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        B, T, C = x.shape

        # Separate Q, K, V projections (supports GQA when num_kv_heads < num_heads)
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply positional encoding to Q and K (leave V unchanged)
        pope_doubled = False
        if self.pos_encoding == "pope":
            if use_pos and cos is not None:
                q = apply_pope_emb(q, cos, sin)
                k = apply_pope_emb(k, cos, sin)
                pope_doubled = True
            elif not use_pos:
                # DroPE phase: preserve dimensions, remove position
                q = apply_pope_no_pos(q)
                k = apply_pope_no_pos(k)
                pope_doubled = True
        elif self.pos_encoding == "rope":
            if cos is not None:
                q = apply_rotary_emb(q, cos, sin)
                k = apply_rotary_emb(k, cos, sin)
        # pos_encoding == "none": skip entirely

        # Expand K/V heads to match Q heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # PoPE doubles Q/K dimensions (head_dim -> 2*head_dim). Explicitly
        # expand V to match by interleaving with zeros:
        #   [v0, 0, v1, 0, ...] so that SDPA output can be collapsed with
        #   out[..., 0::2] on every backend (CPU, MPS, CUDA).
        # The zeros contribute nothing to dot-product attention, so this is
        # mathematically equivalent to the implicit broadcast that CUDA SDPA
        # performs when Q/K and V have different last dimensions.
        if pope_doubled:
            v_out = torch.zeros(
                *v.shape[:-1], 2 * v.shape[-1], dtype=v.dtype, device=v.device,
            )
            v_out[..., 0::2] = v
            v = v_out

        # --- KV cache handling ---
        new_cache: KVCache | None = None

        if cache_read_only and kv_cache is not None:
            # LoopLM non-final recurrent step during incremental decode:
            # read cached K/V for attention context but do NOT write.
            cached_len = kv_cache.seq_len
            k_attn = torch.cat([kv_cache.k[:, :, :cached_len, :], k], dim=2)
            v_attn = torch.cat([kv_cache.v[:, :, :cached_len, :], v], dim=2)

        elif use_cache:
            if kv_cache is None:
                # Prefill path: allocate once at max_seq_len.
                k_buf = torch.empty(
                    B, k.shape[1], self.max_seq_len, k.shape[-1],
                    dtype=k.dtype, device=k.device,
                )
                v_buf = torch.empty(
                    B, v.shape[1], self.max_seq_len, v.shape[-1],
                    dtype=v.dtype, device=v.device,
                )
                write_len = min(T, self.max_seq_len)
                k_buf[:, :, :write_len, :] = k[:, :, -write_len:, :]
                v_buf[:, :, :write_len, :] = v[:, :, -write_len:, :]
                k_attn = k_buf[:, :, :write_len, :]
                v_attn = v_buf[:, :, :write_len, :]
                new_cache = KVCache(
                    k=k_buf,
                    v=v_buf,
                    pos_offset=T,
                    active_len=write_len,
                )
            else:
                k_buf = kv_cache.k
                v_buf = kv_cache.v
                cached_len = kv_cache.seq_len
                max_len = k_buf.shape[2]

                # Sliding window if appending would overflow.
                if cached_len + T > max_len:
                    overflow = cached_len + T - max_len
                    if overflow >= cached_len:
                        cached_len = 0
                    else:
                        keep = cached_len - overflow
                        k_buf[:, :, :keep, :] = k_buf[:, :, overflow:cached_len, :]
                        v_buf[:, :, :keep, :] = v_buf[:, :, overflow:cached_len, :]
                        cached_len = keep

                end = cached_len + T
                k_buf[:, :, cached_len:end, :] = k
                v_buf[:, :, cached_len:end, :] = v
                k_attn = k_buf[:, :, :end, :]
                v_attn = v_buf[:, :, :end, :]
                new_cache = KVCache(
                    k=k_buf,
                    v=v_buf,
                    pos_offset=kv_cache.pos_offset + T,
                    active_len=end,
                )
        else:
            k_attn = k
            v_attn = v

        # DroPE attention temperature scaling (Gelberg et al. 2025)
        # Scale Q before SDPA (equivalent to dividing attn logits by temperature)
        if attn_temperature is not None and attn_temperature != 1.0:
            q = q / math.sqrt(attn_temperature)

        # Build attention mask combining causal + padding
        attn_mask = self._compute_relative_attention_bias(
            q=q,
            q_len=T,
            k_len=k_attn.shape[2],
            kv_cache=kv_cache,
        )
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=q.dtype)
        use_is_causal = False
        if kv_cache is not None or cache_read_only:
            # Incremental mode: new Q tokens can attend to all K/V positions
            pass
        elif pad_mask is not None or attn_mask is not None:
            combined = causal_mask.unsqueeze(0).unsqueeze(0)
            if pad_mask is not None:
                # pad_mask: (B, 1, T, T) where 0 = ignore
                # Combine with causal mask: both must allow attention
                combined = combined | (pad_mask == 0)
            additive_mask = torch.zeros_like(combined, dtype=q.dtype)
            additive_mask.masked_fill_(combined, float("-inf"))
            attn_mask = additive_mask if attn_mask is None else attn_mask + additive_mask
        else:
            use_is_causal = True

        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k_attn, v_attn,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=use_is_causal,
        )

        # Collapse interleaved PoPE output: [real, 0, real, 0, ...] -> [real, real, ...]
        if pope_doubled:
            out = out[..., 0::2]

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj_dropout(self.proj(out))

        if use_cache:
            return out, new_cache
        return out


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network.

    Replaces the standard GELU FFN with a gated linear unit:
        SwiGLU(x) = (SiLU(x @ W_gate) * x @ W1) @ W2

    Three weight matrices instead of two, so ``hidden_dim`` is reduced
    to ~8/3 * embed_dim (rounded to a multiple of 64) to keep the total
    parameter count roughly equivalent.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.swiglu_dim or self._compute_hidden(config.embed_dim)
        self.w_gate = nn.Linear(config.embed_dim, hidden, bias=False)
        self.w1 = nn.Linear(config.embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _compute_hidden(embed_dim: int) -> int:
        """Compute SwiGLU hidden dim ~ 8/3 * embed_dim, rounded to multiple of 64."""
        raw = int(embed_dim * 8 / 3)
        return ((raw + 63) // 64) * 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w_gate(x)) * self.w1(x)))
