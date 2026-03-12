"""Regression tests for key-name canonicalization."""

from __future__ import annotations

from bach_gen.utils.constants import KEY_NAMES
from bach_gen.utils.music_theory import (
    detect_composition_key,
    detect_mode_family,
    get_key_signature_name,
    parse_explicit_key_from_text,
    parse_midi_key_signature,
)


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


def test_parse_midi_key_signature_handles_major_and_minor() -> None:
    assert parse_midi_key_signature("F#") == (6, "major")
    assert parse_midi_key_signature("Bbm") == (10, "minor")


def test_parse_explicit_key_from_text_handles_common_filename_forms() -> None:
    assert parse_explicit_key_from_text("kunstderfuge__ballad_a-flat_major_op_47") == (8, "major")
    assert parse_explicit_key_from_text("etude_in_cminor_7287-1_d") == (0, "minor")
    assert parse_explicit_key_from_text("valse_op_18_no_1_in_eb_major") == (3, "major")


def test_detect_composition_key_prefers_midi_metadata() -> None:
    voices = [
        [(0, 480, 60), (480, 480, 62), (960, 960, 64)],
        [(0, 480, 48), (480, 480, 50), (960, 960, 52)],
    ]
    root, mode, conf, source = detect_composition_key(
        voices,
        time_signature=(4, 4),
        midi_key_signature="Ebm",
    )
    assert (root, mode, source) == (3, "minor", "metadata")
    assert conf == 1.0


def test_detect_composition_key_uses_closing_tonic_support() -> None:
    # Strong G-major profile in the middle, but a clear final cadence to C major.
    voices = [
        [(0, 480, 67), (480, 480, 69), (960, 480, 71), (1440, 960, 72)],
        [(0, 480, 62), (480, 480, 67), (960, 480, 71), (1440, 960, 64)],
        [(0, 480, 59), (480, 480, 62), (960, 480, 67), (1440, 960, 55)],
        [(0, 480, 55), (480, 480, 50), (960, 480, 55), (1440, 960, 48)],
    ]

    root, mode, _conf, source = detect_composition_key(voices, time_signature=(4, 4))

    assert (root, mode) == (0, "major")
    assert source == "heuristic"


def test_detect_composition_key_prefers_minor_with_matching_signature_class() -> None:
    voices = [
        [(0, 480, 69), (480, 480, 71), (960, 480, 72), (1440, 480, 76), (1920, 960, 69)],
        [(0, 960, 57), (960, 960, 60), (1920, 1440, 57)],
        [(0, 960, 64), (960, 960, 67), (1920, 1440, 64)],
    ]

    root, mode, _conf, source = detect_composition_key(
        voices,
        time_signature=(4, 4),
        midi_key_signature="C",
    )

    assert (root, mode) == (9, "minor")
    assert source == "signature+heuristic"


def test_detect_mode_family_finds_dorian() -> None:
    voices = [
        [(0, 480, 62), (480, 480, 64), (960, 480, 65), (1440, 480, 67), (1920, 480, 69), (2400, 480, 71), (2880, 480, 72), (3360, 960, 62)],
        [(0, 960, 50), (960, 960, 53), (1920, 960, 57), (2880, 1440, 50)],
    ]

    diagnosis = detect_mode_family(voices, time_signature=(4, 4))

    assert diagnosis.root_pc == 2
    assert diagnosis.mode_family == "dorian"
    assert diagnosis.system == "modal"


def test_detect_mode_family_finds_mixolydian() -> None:
    voices = [
        [(0, 480, 67), (480, 480, 69), (960, 480, 71), (1440, 480, 72), (1920, 480, 74), (2400, 480, 76), (2880, 480, 77), (3360, 960, 67)],
        [(0, 960, 55), (960, 960, 59), (1920, 960, 60), (2880, 1440, 55)],
    ]

    diagnosis = detect_mode_family(voices, time_signature=(4, 4))

    assert diagnosis.root_pc == 7
    assert diagnosis.mode_family == "mixolydian"
    assert diagnosis.system == "modal"


def test_detect_mode_family_finds_aeolian_as_tonal() -> None:
    voices = [
        [(0, 480, 69), (480, 480, 71), (960, 480, 72), (1440, 480, 74), (1920, 480, 76), (2400, 480, 77), (2880, 480, 79), (3360, 960, 69)],
        [(0, 960, 57), (960, 960, 60), (1920, 960, 64), (2880, 1440, 57)],
    ]

    diagnosis = detect_mode_family(voices, time_signature=(4, 4))

    assert diagnosis.root_pc == 9
    assert diagnosis.mode_family == "aeolian"
    assert diagnosis.system == "tonal"
