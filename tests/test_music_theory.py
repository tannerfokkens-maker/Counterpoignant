"""Regression tests for key-name canonicalization."""

from __future__ import annotations

from bach_gen.utils.constants import KEY_NAMES
from bach_gen.utils.music_theory import get_key_signature_name


def test_get_key_signature_name_always_matches_token_key_vocab():
    valid = set(KEY_NAMES)
    for root_pc in range(12):
        for mode in ("major", "minor"):
            assert get_key_signature_name(root_pc, mode) in valid


def test_enharmonic_aliases_map_to_existing_key_tokens():
    # 1 -> Db / C#
    assert get_key_signature_name(1, "minor") == "Cs_minor"
    # 6 -> Gb / F#
    assert get_key_signature_name(6, "major") == "Fs_major"
    assert get_key_signature_name(6, "minor") == "Fs_minor"
    # 8 -> Ab / G#
    assert get_key_signature_name(8, "minor") == "Gs_minor"
