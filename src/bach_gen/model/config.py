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
    ffn_dim: int = DEFAULT_FFN_DIM  # Kept for backward compat; unused by SwiGLU
    max_seq_len: int = DEFAULT_SEQ_LEN
    dropout: float = DEFAULT_DROPOUT
    weight_tying: bool = True
    rope_theta: float = 10000.0
    pos_encoding: str = "pope"  # "rope" | "pope" | "none"
    swiglu_dim: int | None = None  # Auto-computed from embed_dim if None
    drope_trained: bool = False
    drope_train_seq_len: int | None = None

    def __post_init__(self) -> None:
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
        rms_norm = 2 * self.embed_dim  # 2 norms per layer, weight only (no bias)
        per_layer = attn + ffn + rms_norm
        layers = per_layer * self.num_layers
        # Output head (tied with embedding if weight_tying)
        head = 0 if self.weight_tying else self.embed_dim * self.vocab_size
        return emb + layers + head
