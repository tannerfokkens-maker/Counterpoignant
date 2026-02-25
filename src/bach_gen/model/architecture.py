"""Decoder-only Transformer for Bach invention generation (~6M params).

Architecture: RoPE + RMSNorm + SwiGLU + pre-norm + weight tying (mini-LLaMA).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bach_gen.model.config import ModelConfig


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
    """Apply PoPE to x: softplus → rotate → interleave to 2*D.

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
    """Apply PoPE without position: softplus → expand to 2*D with zero angles.

    Used during DroPE recalibration to maintain the attention dimension
    and nonlinearity while removing all positional information.

    This is equivalent to apply_pope_emb with cos=1, sin=0 everywhere.
    """
    mag = F.softplus(x)  # (B, H, T, D)
    zeros = torch.zeros_like(mag)

    out = torch.stack([mag, zeros], dim=-1)  # (B, H, T, D, 2)
    return out.reshape(*x.shape[:-1], 2 * x.shape[-1])


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


class BachTransformer(nn.Module):
    """Small decoder-only Transformer for music generation."""

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

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # RMSNorm weight is already initialized to ones in __init__

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_rope: bool = True,
        attn_temperature: float | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) optional padding mask (1=attend, 0=ignore).
            use_rope: If False, skip rotary positional embeddings (for DroPE).
            attn_temperature: If set, scale attention logits by 1/beta* before
                softmax. Used at inference with DroPE models on extended contexts.

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings only (no learned positional embedding)
        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len), 1=attend, 0=pad
            # We need to expand it for the attention mechanism
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            pad_mask = pad_mask.expand(-1, -1, seq_len, -1)  # (B, 1, S, S)
        else:
            pad_mask = None

        # Compute positional cos/sin for this sequence length (or skip for DroPE)
        if use_rope and self.pos_emb is not None:
            cos, sin = self.pos_emb(seq_len)
            cos = cos.to(device)
            sin = sin.to(device)
        else:
            cos, sin = None, None

        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask, pad_mask=pad_mask,
                      cos=cos, sin=sin, use_pos=use_rope,
                      attn_temperature=attn_temperature)

        x = self.ln_final(x)
        logits = self.head(x)

        return logits

    def count_parameters(self) -> int:
        """Count actual trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block with RMSNorm and SwiGLU."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.embed_dim)
        self.ffn = SwiGLUFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        use_pos: bool = True,
        attn_temperature: float | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.ln1(x), causal_mask, pad_mask,
                                       cos=cos, sin=sin, use_pos=use_pos,
                                       attn_temperature=attn_temperature))
        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


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

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.embed_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.embed_dim, kv_dim)
        self.v_proj = nn.Linear(config.embed_dim, kv_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        use_pos: bool = True,
        attn_temperature: float | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Separate Q, K, V projections (supports GQA when num_kv_heads < num_heads)
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply positional encoding to Q and K (leave V unchanged)
        if self.pos_encoding == "pope":
            if use_pos and cos is not None:
                q = apply_pope_emb(q, cos, sin)
                k = apply_pope_emb(k, cos, sin)
            elif not use_pos:
                # DroPE phase: preserve dimensions, remove position
                q = apply_pope_no_pos(q)
                k = apply_pope_no_pos(k)
        elif self.pos_encoding == "rope":
            if cos is not None:
                q = apply_rotary_emb(q, cos, sin)
                k = apply_rotary_emb(k, cos, sin)
        # pos_encoding == "none": skip entirely

        # Expand K/V heads to match Q heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # DroPE attention temperature scaling (Gelberg et al. 2025)
        # Scale Q before SDPA (equivalent to dividing attn logits by temperature)
        if attn_temperature is not None and attn_temperature != 1.0:
            q = q / math.sqrt(attn_temperature)

        # Build attention mask combining causal + padding
        attn_mask = None
        if pad_mask is not None:
            # pad_mask: (B, 1, T, T) where 0 = ignore
            # Combine with causal mask: both must allow attention
            combined = causal_mask.unsqueeze(0).unsqueeze(0) | (pad_mask == 0)
            # Convert bool mask to float: True (blocked) -> -inf, False (attend) -> 0
            attn_mask = torch.zeros_like(combined, dtype=q.dtype)
            attn_mask.masked_fill_(combined, float("-inf"))

        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=(attn_mask is None),  # use built-in causal mask when no padding
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj_dropout(self.proj(out))

        return out


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network.

    Replaces the standard GELU FFN with a gated linear unit:
        SwiGLU(x) = (SiLU(x @ W_gate) ⊙ x @ W1) @ W2

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
        """Compute SwiGLU hidden dim ≈ 8/3 * embed_dim, rounded to multiple of 64."""
        raw = int(embed_dim * 8 / 3)
        return ((raw + 63) // 64) * 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w_gate(x)) * self.w1(x)))
