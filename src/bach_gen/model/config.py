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
    num_layers: int = DEFAULT_NUM_LAYERS
    ffn_dim: int = DEFAULT_FFN_DIM  # Kept for backward compat; unused by SwiGLU
    max_seq_len: int = DEFAULT_SEQ_LEN
    dropout: float = DEFAULT_DROPOUT
    weight_tying: bool = True
    rope_theta: float = 10000.0
    swiglu_dim: int | None = None  # Auto-computed from embed_dim if None
    drope_trained: bool = False
    drope_train_seq_len: int | None = None

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
        attn = 4 * self.embed_dim * self.embed_dim  # Q, K, V, O projections
        swiglu_hidden = self.effective_swiglu_dim
        ffn = 3 * self.embed_dim * swiglu_hidden  # gate + up + down (no bias)
        rms_norm = 2 * self.embed_dim  # 2 norms per layer, weight only (no bias)
        per_layer = attn + ffn + rms_norm
        layers = per_layer * self.num_layers
        # Output head (tied with embedding if weight_tying)
        head = 0 if self.weight_tying else self.embed_dim * self.vocab_size
        return emb + layers + head
