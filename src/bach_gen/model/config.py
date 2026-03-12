"""Model configuration."""

from __future__ import annotations

from dataclasses import dataclass

from bach_gen.utils.constants import (
    DEFAULT_SEQ_LEN,
    DEFAULT_EMBED_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_FFN_DIM,
    DEFAULT_DROPOUT,
)


@dataclass
class ModelConfig:
    """Configuration for the Transformer model."""

    vocab_size: int = 400
    embed_dim: int = DEFAULT_EMBED_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    num_kv_heads: int | None = None  # None = standard MHA (same as num_heads)
    num_layers: int = DEFAULT_NUM_LAYERS
    num_front_layers: int = 0
    num_loop_layers: int = 0
    num_back_layers: int = 0
    ffn_dim: int = DEFAULT_FFN_DIM  # Kept for backward compat; unused by SwiGLU
    max_seq_len: int = DEFAULT_SEQ_LEN
    dropout: float = DEFAULT_DROPOUT
    weight_tying: bool = True
    rope_theta: float = 10000.0
    pos_encoding: str = "pope"  # "rope" | "pope" | "none"
    rel_attn_bias: bool = False
    rel_attn_max_distance: int = 2048
    swiglu_dim: int | None = None  # Auto-computed from embed_dim if None
    drope_trained: bool = False
    drope_train_seq_len: int | None = None

    # LoopLM: weight-tied recurrence (Ouro / LoopLM architecture)
    num_recurrent_steps: int = 1  # T_max; 1 = standard transformer, >1 = looped
    looplm_sandwich_norm: bool = False  # RMSNorm after each sublayer for recurrence stability
    looplm_exit_gate: bool = True  # Learned adaptive exit gate
    looplm_kl_beta: float = 0.05  # Entropy regularization coefficient
    looplm_exit_threshold: float = 0.5  # Inference CDF threshold for early exit
    loop_step_embedding: bool = True  # Learned recurrent-step embedding for block LoopLM
    loop_per_step_norms: bool = False  # Placeholder for future untied per-step norms

    def __post_init__(self) -> None:
        self._validate_and_normalize()

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.apply_checkpoint_compat()

    def apply_checkpoint_compat(self) -> None:
        """Backfill fields missing from older checkpoint configs, then validate."""
        state = vars(self)
        legacy_defaults = {
            "pos_encoding": "rope",
            "num_kv_heads": None,
            "rel_attn_bias": False,
            "rel_attn_max_distance": 2048,
            "num_recurrent_steps": 1,
            "looplm_sandwich_norm": False,
            "looplm_exit_gate": False,
            "looplm_kl_beta": 0.1,
            "looplm_exit_threshold": 0.5,
            "num_front_layers": 0,
            "num_loop_layers": self.num_layers,
            "num_back_layers": 0,
            "loop_step_embedding": True,
            "loop_per_step_norms": False,
        }
        for name, value in legacy_defaults.items():
            if name not in state:
                setattr(self, name, value)
        self._validate_and_normalize()

    def _validate_and_normalize(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.num_recurrent_steps < 1:
            raise ValueError("num_recurrent_steps must be >= 1")
        if self.num_kv_heads is not None:
            if self.num_kv_heads > self.num_heads:
                raise ValueError(
                    f"num_kv_heads ({self.num_kv_heads}) must be <= num_heads ({self.num_heads})"
                )
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be divisible by "
                    f"num_kv_heads ({self.num_kv_heads})"
                )
        if self.rel_attn_max_distance < 1:
            raise ValueError("rel_attn_max_distance must be >= 1")
        if self.num_front_layers < 0 or self.num_loop_layers < 0 or self.num_back_layers < 0:
            raise ValueError("num_front_layers, num_loop_layers, and num_back_layers must be >= 0")

        # Default to full-stack recurrence when no explicit split is provided.
        if self.num_loop_layers == 0:
            remaining = self.num_layers - self.num_front_layers - self.num_back_layers
            if remaining > 0:
                self.num_loop_layers = remaining

        if self.num_loop_layers < 1:
            raise ValueError("num_loop_layers must be >= 1")
        if self.num_front_layers + self.num_loop_layers + self.num_back_layers != self.num_layers:
            raise ValueError(
                "num_front_layers + num_loop_layers + num_back_layers "
                f"must equal num_layers ({self.num_layers})"
            )

    @property
    def effective_num_kv_heads(self) -> int:
        """The actual number of KV heads (defaults to num_heads for standard MHA)."""
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads

    @property
    def effective_swiglu_dim(self) -> int:
        """The actual SwiGLU hidden dim (auto-computed if not set)."""
        if self.swiglu_dim is not None:
            return self.swiglu_dim
        raw = int(self.embed_dim * 8 / 3)
        return ((raw + 63) // 64) * 64

    @property
    def num_params(self) -> int:
        """Estimate number of parameters."""
        # Embedding
        emb = self.vocab_size * self.embed_dim
        # RoPE uses no learned parameters (sin/cos buffers only)
        # Each transformer layer: attention + SwiGLU FFN
        head_dim = self.embed_dim // self.num_heads
        kv_dim = self.effective_num_kv_heads * head_dim
        # Q + O: embed_dim -> embed_dim; K + V: embed_dim -> kv_dim
        attn = 2 * self.embed_dim * self.embed_dim + 2 * self.embed_dim * kv_dim
        swiglu_hidden = self.effective_swiglu_dim
        ffn = 3 * self.embed_dim * swiglu_hidden  # gate + up + down (no bias)
        # 2 pre-norms per layer; +2 post-norms if sandwich norm enabled
        norms_per_layer = 4 if self.looplm_sandwich_norm else 2
        rms_norm = norms_per_layer * self.embed_dim  # weight only (no bias)
        rel_dim = head_dim * (2 if self.pos_encoding == "pope" else 1)
        rel_attn = (
            self.num_heads * self.rel_attn_max_distance * rel_dim
            if self.rel_attn_bias else 0
        )
        per_layer = attn + ffn + rms_norm + rel_attn
        layers = per_layer * self.num_layers
        # Output head (tied with embedding if weight_tying)
        head = 0 if self.weight_tying else self.embed_dim * self.vocab_size
        # LoopLM exit gate: Linear(embed_dim, 1) = embed_dim + 1 params
        exit_gate = (self.embed_dim + 1) if (self.looplm_exit_gate and self.num_recurrent_steps > 1) else 0
        loop_step_embed = (
            self.num_recurrent_steps * self.embed_dim
            if self.loop_step_embedding and self.num_recurrent_steps > 1
            else 0
        )
        return emb + layers + head + exit_gate + loop_step_embed
