#!/usr/bin/env python3
"""Diagnose whether PyTorch Flash SDPA is eligible for this attention path.

Runs a synthetic forward setup that mirrors ``CausalSelfAttention`` and then:
1. calls ``torch.backends.cuda.can_use_flash_attention(..., debug=True)``
2. forces Flash-only SDPA via ``torch.backends.cuda.sdp_kernel(...)``

This is useful for distinguishing:
- head-dimension incompatibility
- explicit mask incompatibility
- other SDPA backend constraints
"""

from __future__ import annotations

import argparse
import math
import traceback
import warnings

import torch
import torch.nn.functional as F

from bach_gen.model.architecture import (
    CausalSelfAttention,
    PoPEEmbedding,
    RotaryEmbedding,
    apply_pope_emb,
    apply_rotary_emb,
)
from bach_gen.model.config import ModelConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--embed-dim", type=int, default=384)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--pos-encoding", choices=["pope", "rope", "none"], default="pope")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument(
        "--mask-mode",
        choices=["none", "all-valid-pad", "right-padded", "left-padded"],
        default="none",
        help="Simulate the mask pattern handed to the attention block.",
    )
    p.add_argument("--pad-fraction", type=float, default=0.25)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--training", action="store_true")
    p.add_argument("--rel-attn-bias", action="store_true")
    return p.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def _make_attention_mask(batch_size: int, seq_len: int, mask_mode: str, pad_fraction: float, device: torch.device) -> torch.Tensor | None:
    if mask_mode == "none":
        return None
    mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    pad = max(1, int(seq_len * pad_fraction))
    if mask_mode == "all-valid-pad":
        return mask
    if mask_mode == "right-padded":
        mask[:, -pad:] = 0
        return mask
    if mask_mode == "left-padded":
        mask[:, :pad] = 0
        return mask
    raise ValueError(mask_mode)


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for SDPA Flash diagnostics.")

    device = torch.device("cuda")
    dtype = _torch_dtype(args.dtype)
    config = ModelConfig(
        vocab_size=133,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_layers=1,
        max_seq_len=args.seq_len,
        pos_encoding=args.pos_encoding,
        rel_attn_bias=args.rel_attn_bias,
        dropout=args.dropout,
    )
    attn = CausalSelfAttention(config).to(device=device, dtype=dtype)
    attn.train(args.training)
    pos_emb = None
    if args.pos_encoding == "pope":
        pos_emb = PoPEEmbedding(
            dim=attn.head_dim,
            max_seq_len=args.seq_len,
            theta=config.rope_theta,
        ).to(device=device)
    elif args.pos_encoding == "rope":
        pos_emb = RotaryEmbedding(
            dim=attn.head_dim,
            max_seq_len=args.seq_len,
            theta=config.rope_theta,
        ).to(device=device)

    x = torch.randn(args.batch_size, args.seq_len, args.embed_dim, device=device, dtype=dtype)
    q = attn.q_proj(x).reshape(args.batch_size, args.seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).reshape(args.batch_size, args.seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).reshape(args.batch_size, args.seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

    if args.pos_encoding == "pope":
        cos, sin = pos_emb(args.seq_len)
        cos = cos.to(device=device)
        sin = sin.to(device=device)
        q = apply_pope_emb(q, cos, sin)
        k = apply_pope_emb(k, cos, sin)
        v_out = torch.zeros(*v.shape[:-1], 2 * v.shape[-1], dtype=v.dtype, device=v.device)
        v_out[..., 0::2] = v
        v = v_out
    elif args.pos_encoding == "rope":
        cos, sin = pos_emb(args.seq_len)
        cos = cos.to(device=device)
        sin = sin.to(device=device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
    else:
        cos = sin = None

    if attn.num_kv_groups > 1:
        k = k.repeat_interleave(attn.num_kv_groups, dim=1)
        v = v.repeat_interleave(attn.num_kv_groups, dim=1)

    attention_mask = _make_attention_mask(
        args.batch_size,
        args.seq_len,
        args.mask_mode,
        args.pad_fraction,
        device,
    )
    causal_mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    pad_mask = None
    if attention_mask is not None:
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, args.seq_len, -1)

    attn_mask, use_is_causal = attn._build_sdpa_mask(
        q=q,
        k_len=k.shape[2],
        causal_mask=causal_mask,
        pad_mask=pad_mask,
        kv_cache=None,
        cache_read_only=False,
    )

    dropout_p = args.dropout if args.training else 0.0
    enable_gqa = attn.num_kv_heads != attn.num_heads
    params = torch.backends.cuda.SDPAParams(
        q, k, v, attn_mask, dropout_p, use_is_causal, enable_gqa,
    )

    print("=== SDPA diagnostic ===")
    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"dtype={dtype}")
    print(f"batch_size={args.batch_size}")
    print(f"seq_len={args.seq_len}")
    print(f"embed_dim={args.embed_dim}")
    print(f"num_heads={args.num_heads}")
    print(f"head_dim={attn.head_dim}")
    print(f"num_kv_heads={attn.num_kv_heads}")
    print(f"pos_encoding={args.pos_encoding}")
    print(f"mask_mode={args.mask_mode}")
    print(f"attn_mask={'none' if attn_mask is None else str(tuple(attn_mask.shape)) + ' ' + str(attn_mask.dtype)}")
    print(f"is_causal={use_is_causal}")
    print(f"dropout_p={dropout_p}")
    print(f"training={args.training}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flash_ok = torch.backends.cuda.can_use_flash_attention(params, debug=True)
    print(f"can_use_flash_attention={flash_ok}")
    if caught:
        print("debug_warnings:")
        for w in caught:
            print(f"  - {w.message}")

    print("forcing flash-only SDPA...")
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=use_is_causal,
            )
        print(f"flash_only_call=success out_shape={tuple(out.shape)}")
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime
        print(f"flash_only_call=failed type={type(exc).__name__}")
        print(str(exc))
        print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
