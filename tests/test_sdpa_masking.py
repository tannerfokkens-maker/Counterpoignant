from __future__ import annotations

import torch

from bach_gen.model.architecture import CausalSelfAttention
from bach_gen.model.config import ModelConfig


def _make_attn(*, rel_attn_bias: bool = False) -> CausalSelfAttention:
    config = ModelConfig(
        vocab_size=128,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=64,
        pos_encoding="pope",
        rel_attn_bias=rel_attn_bias,
    )
    return CausalSelfAttention(config)


def test_all_valid_pad_mask_keeps_is_causal_fast_path():
    attn = _make_attn(rel_attn_bias=False)
    q = torch.randn(2, attn.num_heads, 8, attn.head_dim * 2)
    causal_mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    pad_mask = torch.ones(2, 1, 8, 8, dtype=torch.long)

    attn_mask, use_is_causal = attn._build_sdpa_mask(
        q=q,
        k_len=8,
        causal_mask=causal_mask,
        pad_mask=pad_mask,
    )

    assert attn_mask is None
    assert use_is_causal is True


def test_real_padding_requires_explicit_mask():
    attn = _make_attn(rel_attn_bias=False)
    q = torch.randn(2, attn.num_heads, 8, attn.head_dim * 2)
    causal_mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    pad_mask = torch.ones(2, 1, 8, 8, dtype=torch.long)
    pad_mask[:, :, :, -2:] = 0

    attn_mask, use_is_causal = attn._build_sdpa_mask(
        q=q,
        k_len=8,
        causal_mask=causal_mask,
        pad_mask=pad_mask,
    )

    assert attn_mask is not None
    assert use_is_causal is False
    assert attn_mask.shape == (2, 1, 8, 8)
