"""Unit tests for PoPE positional embedding behavior."""

from __future__ import annotations

import torch

from bach_gen.model.architecture import (
    CausalSelfAttention,
    PoPEEmbedding,
    apply_pope_emb,
    apply_pope_no_pos,
)
from bach_gen.model.config import ModelConfig


def test_apply_pope_emb_doubles_last_dimension() -> None:
    torch.manual_seed(0)
    batch, heads, seq_len, dim = 2, 3, 7, 8
    x = torch.randn(batch, heads, seq_len, dim)

    pos = PoPEEmbedding(dim=dim, max_seq_len=seq_len)
    cos, sin = pos(seq_len)

    out = apply_pope_emb(x, cos, sin)
    assert out.shape == (batch, heads, seq_len, dim * 2)


def test_apply_pope_no_pos_matches_zero_angle_reference() -> None:
    torch.manual_seed(1)
    batch, heads, seq_len, dim = 1, 2, 5, 6
    x = torch.randn(batch, heads, seq_len, dim)

    cos = torch.ones(seq_len, dim)
    sin = torch.zeros(seq_len, dim)

    expected = apply_pope_emb(x, cos, sin)
    actual = apply_pope_no_pos(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_apply_pope_emb_preserves_magnitude_per_dimension() -> None:
    """For each original dimension i: sqrt(real_i^2 + imag_i^2) == softplus(x_i)."""
    torch.manual_seed(2)
    x = torch.randn(2, 2, 4, 5)

    pos = PoPEEmbedding(dim=x.shape[-1], max_seq_len=x.shape[-2])
    cos, sin = pos(x.shape[-2])

    out = apply_pope_emb(x, cos, sin)  # (..., 2*D), interleaved real/imag
    real = out[..., 0::2]
    imag = out[..., 1::2]

    reconstructed_mag = torch.sqrt(real.pow(2) + imag.pow(2))
    expected_mag = torch.nn.functional.softplus(x)

    assert torch.allclose(reconstructed_mag, expected_mag, atol=1e-5, rtol=1e-5)


def test_pope_embedding_long_sequence_cache_is_finite() -> None:
    pos = PoPEEmbedding(dim=16, max_seq_len=32)
    cos, sin = pos(4096)

    assert cos.shape == (4096, 16)
    assert sin.shape == (4096, 16)
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()


def test_causal_self_attention_with_pope_runs_and_is_finite() -> None:
    torch.manual_seed(3)
    config = ModelConfig(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=64,
        pos_encoding="pope",
    )
    attn = CausalSelfAttention(config)

    batch, seq_len = 2, 11
    x = torch.randn(batch, seq_len, config.embed_dim)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    pos = PoPEEmbedding(dim=config.embed_dim // config.num_heads, max_seq_len=seq_len)
    cos, sin = pos(seq_len)

    out = attn(x, causal_mask=causal_mask, cos=cos, sin=sin, use_pos=True)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
