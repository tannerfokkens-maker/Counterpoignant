"""Tests for musical-analysis helpers used during prepare-data."""

from __future__ import annotations

from bach_gen.data.analysis import (
    compute_harmonic_rhythm,
    compute_harmonic_tension,
    compute_imitation,
    compute_chromaticism,
    compute_texture,
)
from bach_gen.data.extraction import VoiceComposition


def test_compute_harmonic_rhythm_detects_slow_changes():
    comp = VoiceComposition(
        voices=[
            [(0, 3840, 60)],
            [(0, 3840, 48)],
        ],
        key_root=0,
        key_mode="major",
        source="unit-rhythm-slow",
        time_signature=(4, 4),
    )

    assert compute_harmonic_rhythm(comp, (4, 4)) == "slow"


def test_compute_harmonic_rhythm_detects_fast_changes():
    voice_1 = [
        (0, 480, 60), (480, 480, 61), (960, 480, 62), (1440, 480, 63),
        (1920, 480, 64), (2400, 480, 65), (2880, 480, 66), (3360, 480, 67),
    ]
    voice_2 = [
        (0, 480, 48), (480, 480, 50), (960, 480, 52), (1440, 480, 53),
        (1920, 480, 55), (2400, 480, 57), (2880, 480, 59), (3360, 480, 60),
    ]
    comp = VoiceComposition(
        voices=[voice_1, voice_2],
        key_root=0,
        key_mode="major",
        source="unit-rhythm-fast",
        time_signature=(4, 4),
    )

    assert compute_harmonic_rhythm(comp, (4, 4)) == "fast"


def test_compute_harmonic_tension_detects_low_dissonance():
    comp = VoiceComposition(
        voices=[
            [(0, 960, 64), (960, 960, 67)],
            [(0, 960, 60), (960, 960, 64)],
        ],
        key_root=0,
        key_mode="major",
        source="unit-tension-low",
        time_signature=(4, 4),
    )

    assert compute_harmonic_tension(comp) == "low"


def test_compute_harmonic_tension_detects_high_dissonance():
    comp = VoiceComposition(
        voices=[
            [(0, 960, 61), (960, 960, 66), (1920, 960, 71)],
            [(0, 960, 60), (960, 960, 65), (1920, 960, 70)],
        ],
        key_root=0,
        key_mode="major",
        source="unit-tension-high",
        time_signature=(4, 4),
    )

    assert compute_harmonic_tension(comp) == "high"


def test_compute_texture_detects_staggered_two_part_counterpoint():
    comp = VoiceComposition(
        voices=[
            [(0, 960, 60), (960, 960, 62), (1920, 960, 64), (2880, 960, 65)],
            [(480, 960, 55), (1440, 960, 57), (2400, 960, 59), (3360, 960, 60)],
        ],
        key_root=0,
        key_mode="major",
        source="unit-texture-invention",
        time_signature=(4, 4),
    )

    assert compute_texture(comp, form="invention") == "polyphonic"


def test_compute_texture_detects_block_chorale_homophony():
    comp = VoiceComposition(
        voices=[
            [(0, 960, 72), (960, 960, 71), (1920, 960, 72)],
            [(0, 960, 67), (960, 960, 65), (1920, 960, 67)],
            [(0, 960, 64), (960, 960, 62), (1920, 960, 64)],
            [(0, 960, 48), (960, 960, 50), (1920, 960, 48)],
        ],
        key_root=0,
        key_mode="major",
        source="unit-texture-chorale",
        time_signature=(4, 4),
    )

    assert compute_texture(comp, form="chorale") == "homophonic"


def test_compute_imitation_uses_subject_entries_as_floor_for_inventions():
    comp = VoiceComposition(
        voices=[
            [
                (0, 360, 60),
                (360, 600, 62),
                (960, 360, 64),
                (1320, 600, 65),
                (1920, 480, 67),
                (2400, 480, 69),
                (2880, 480, 71),
            ],
            [
                (960, 480, 55),
                (1440, 360, 57),
                (1800, 600, 59),
                (2400, 360, 60),
                (2760, 600, 62),
                (3360, 480, 64),
                (3840, 480, 65),
            ],
        ],
        key_root=0,
        key_mode="major",
        source="unit-imitation-invention",
        time_signature=(4, 4),
    )

    assert compute_imitation(comp, form="invention") == "low"


def test_compute_chromaticism_uses_modal_reference_for_motets():
    comp = VoiceComposition(
        voices=[
            [
                (0, 480, 62),
                (480, 480, 64),
                (960, 480, 65),
                (1440, 480, 67),
                (1920, 480, 69),
                (2400, 480, 71),
                (2880, 480, 72),
                (3360, 480, 71),
                (3840, 480, 69),
                (4320, 480, 67),
                (4800, 480, 65),
                (5280, 960, 62),
            ],
            [
                (0, 960, 50),
                (960, 960, 53),
                (1920, 960, 57),
                (2880, 960, 59),
                (3840, 960, 57),
                (4800, 1440, 50),
            ],
        ],
        key_root=2,
        key_mode="minor",
        source="unit-modal-motet",
        style="renaissance",
        time_signature=(4, 4),
    )

    assert compute_chromaticism(comp, form="motet") == "low"
    assert compute_chromaticism(comp, form="vocal_polyphony") == "low"
