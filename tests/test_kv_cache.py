"""Tests for KV cache numerical equivalence.

Verifies that prefill + incremental decoding produces identical logits
to a single full forward pass for all positional encoding modes.
"""

from __future__ import annotations

import torch
import pytest

from bach_gen.model.config import ModelConfig
from bach_gen.model.architecture import BachTransformer, KVCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config(**overrides) -> ModelConfig:
    """Small model config for fast tests."""
    defaults = dict(
        vocab_size=64,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=128,
        dropout=0.0,
        weight_tying=False,
        pos_encoding="rope",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _incremental_logits(
    model: BachTransformer,
    input_ids: torch.Tensor,
    use_rope: bool = True,
) -> torch.Tensor:
    """Run prefill + incremental decoding, collect per-position logits."""
    model.eval()
    seq_len = input_ids.shape[1]

    # Prefill with first token
    prefill_ids = input_ids[:, :1]
    logits_prefill, kv_caches = model(
        prefill_ids, use_rope=use_rope, use_cache=True,
    )

    all_logits = [logits_prefill]

    # Incremental: one token at a time
    for t in range(1, seq_len):
        step_ids = input_ids[:, t : t + 1]
        step_logits, kv_caches = model(
            step_ids, use_rope=use_rope, use_cache=True, kv_cache=kv_caches,
        )
        all_logits.append(step_logits)

    # Concat along seq dim: (B, seq_len, vocab)
    return torch.cat(all_logits, dim=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKVCacheEquivalence:
    """Full forward vs. prefill + incremental must produce identical logits."""

    def test_rope_mode(self):
        """RoPE: incremental logits match full forward."""
        config = _tiny_config(pos_encoding="rope")
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            full_logits = model(input_ids, use_rope=True)
            incr_logits = _incremental_logits(model, input_ids, use_rope=True)

        torch.testing.assert_close(incr_logits, full_logits, atol=1e-4, rtol=1e-4)

    def test_pope_mode(self):
        """PoPE: incremental logits match full forward."""
        config = _tiny_config(pos_encoding="pope")
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            full_logits = model(input_ids, use_rope=True)
            incr_logits = _incremental_logits(model, input_ids, use_rope=True)

        torch.testing.assert_close(incr_logits, full_logits, atol=1e-4, rtol=1e-4)

    def test_none_mode_drope(self):
        """No positional encoding (DroPE): incremental logits match full forward."""
        config = _tiny_config(pos_encoding="none")
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            full_logits = model(input_ids, use_rope=False)
            incr_logits = _incremental_logits(model, input_ids, use_rope=False)

        torch.testing.assert_close(incr_logits, full_logits, atol=1e-4, rtol=1e-4)

    def test_pope_drope_phase(self):
        """PoPE model in DroPE phase (use_rope=False): incremental matches full."""
        config = _tiny_config(pos_encoding="pope")
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            full_logits = model(input_ids, use_rope=False)
            incr_logits = _incremental_logits(model, input_ids, use_rope=False)

        torch.testing.assert_close(incr_logits, full_logits, atol=1e-4, rtol=1e-4)

    def test_gqa_rope(self):
        """GQA (num_kv_heads=2, num_heads=4) with RoPE: incremental matches full."""
        config = _tiny_config(pos_encoding="rope", num_kv_heads=2)
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            full_logits = model(input_ids, use_rope=True)
            incr_logits = _incremental_logits(model, input_ids, use_rope=True)

        torch.testing.assert_close(incr_logits, full_logits, atol=1e-4, rtol=1e-4)

    def test_gqa_pope(self):
        """GQA with PoPE: incremental matches full."""
        config = _tiny_config(pos_encoding="pope", num_kv_heads=2)
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            full_logits = model(input_ids, use_rope=True)
            incr_logits = _incremental_logits(model, input_ids, use_rope=True)

        torch.testing.assert_close(incr_logits, full_logits, atol=1e-4, rtol=1e-4)


class TestKVCacheBackwardCompat:
    """Default behavior (no caching) is unchanged."""

    def test_returns_tensor_without_cache(self):
        """model(input_ids) returns a plain tensor, not a tuple."""
        config = _tiny_config()
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            result = model(input_ids)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 8, config.vocab_size)

    def test_use_cache_returns_tuple(self):
        """model(input_ids, use_cache=True) returns (logits, caches)."""
        config = _tiny_config()
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            result = model(input_ids, use_cache=True)

        assert isinstance(result, tuple)
        logits, caches = result
        assert logits.shape == (1, 8, config.vocab_size)
        assert len(caches) == config.num_layers
        for cache in caches:
            assert isinstance(cache, KVCache)
            assert cache.seq_len == 8
            assert cache.pos_offset == 8


class TestKVCachePrefillThenIncrement:
    """Multi-token prefill followed by single-token increments."""

    def test_prefill_chunk_then_incremental(self):
        """Prefill 8 tokens, then increment 8 more — matches 16-token full pass."""
        config = _tiny_config(pos_encoding="rope")
        model = BachTransformer(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            full_logits = model(input_ids, use_rope=True)

            # Prefill first 8
            logits_pf, caches = model(
                input_ids[:, :8], use_rope=True, use_cache=True,
            )
            # Incremental for remaining 8
            incr_parts = [logits_pf]
            for t in range(8, 16):
                step_logits, caches = model(
                    input_ids[:, t : t + 1],
                    use_rope=True, use_cache=True, kv_cache=caches,
                )
                incr_parts.append(step_logits)

            incr_logits = torch.cat(incr_parts, dim=1)

        torch.testing.assert_close(incr_logits, full_logits, atol=1e-4, rtol=1e-4)


class TestKVCacheDataclass:
    """Basic KVCache properties."""

    def test_seq_len(self):
        cache = KVCache(
            k=torch.zeros(1, 4, 10, 8),
            v=torch.zeros(1, 4, 10, 8),
            pos_offset=10,
        )
        assert cache.seq_len == 10

    def test_default_pos_offset(self):
        cache = KVCache(
            k=torch.zeros(1, 4, 5, 8),
            v=torch.zeros(1, 4, 5, 8),
        )
        assert cache.pos_offset == 0
