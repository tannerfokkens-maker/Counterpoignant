"""Tests for sequence chunking and tokenization helpers used by prepare-data."""

from __future__ import annotations

from bach_gen.cli import _build_token_category_map, _infer_roundtrip_settings, _tokenize_items, chunk_sequences
from bach_gen.data.extraction import VoiceComposition
from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer


class _DummyTokenizer:
    token_to_name = {
        1: "BOS",
        20: "STYLE_BACH",
        24: "FORM_CHORALE",
        29: "FORM_TRIO_SONATA",
        18: "MODE_4PART",
        56: "ENCODE_INTERLEAVED",
        57: "ENCODE_SEQUENTIAL",
        59: "KEY_C_major",
        60: "KEY_D_minor",
        2: "EOS",
    }
    vocab_size = 61


def test_chunk_sequences_preserves_conditioning_prefix_on_continuations():
    tokenizer = _DummyTokenizer()
    prefix = [1, 20, 24, 18, 56, 59]
    seq = prefix + list(range(1000, 1450)) + [2]

    chunks, chunk_ids = chunk_sequences(
        [seq],
        max_seq_len=128,
        stride_fraction=0.75,
        bos_token=1,
        tokenizer=tokenizer,
        piece_ids=["piece-a"],
    )

    assert len(chunks) > 1
    assert all(c[:len(prefix)] == prefix for c in chunks)
    assert chunk_ids == ["piece-a"] * len(chunks)


def test_chunk_sequences_fallback_to_legacy_bos_behavior_without_tokenizer():
    seq = list(range(300))
    chunks, _ = chunk_sequences(
        [seq],
        max_seq_len=64,
        stride_fraction=0.75,
        bos_token=1,
        tokenizer=None,
    )

    assert len(chunks) > 1
    assert chunks[1][0] == 1


def test_infer_roundtrip_settings_prefers_prefix_form_and_encoding():
    tokenizer = _DummyTokenizer()
    seq = [1, 20, 29, 18, 57, 60, 1000, 1001, 2]
    form, encoding_mode = _infer_roundtrip_settings(seq, tokenizer, default_form="chorale")

    assert form == "trio_sonata"
    assert encoding_mode == "sequential"


def test_tokenize_items_emits_interleaved_and_sequential_sequences():
    tokenizer = ScaleDegreeTokenizer()
    comp = VoiceComposition(
        voices=[[(0, 480, 60), (480, 480, 62)], [(0, 960, 48)]],
        key_root=0,
        key_mode="major",
        source="unit-test-piece",
        style="bach",
        time_signature=(4, 4),
    )

    seqs, ids = _tokenize_items([comp], ["invention"], tokenizer, no_sequential=False)

    assert len(seqs) == 2
    assert ids == ["unit-test-piece", "unit-test-piece"]
    assert all(len(s) >= 20 for s in seqs)


def test_build_token_category_map_groups_expected_token_types():
    tokenizer = _DummyTokenizer()
    cat_map, cat_names = _build_token_category_map(tokenizer)
    idx = {name: i for i, name in enumerate(cat_names)}

    assert cat_map[24] == idx["conditioning"]  # FORM_CHORALE
    assert cat_map[57] == idx["conditioning"]  # ENCODE_SEQUENTIAL
    assert cat_map[59] == idx["conditioning"]  # KEY_C_major
    assert cat_map[1] == idx["other"]  # BOS
