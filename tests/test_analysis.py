"""Tests for musical-analysis helpers used during prepare-data."""

from __future__ import annotations

from bach_gen.data.analysis import compute_harmonic_rhythm, compute_harmonic_tension
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
