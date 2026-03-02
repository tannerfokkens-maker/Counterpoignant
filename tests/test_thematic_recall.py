"""Tests for thematic recall scoring behavior."""

from __future__ import annotations

from bach_gen.data.extraction import VoiceComposition
from bach_gen.evaluation.structural import score_thematic_recall


def _melody(start: int, pitch: int, intervals: list[int], dur: int = 480) -> list[tuple[int, int, int]]:
    notes = [(start, dur, pitch)]
    t = start + dur
    p = pitch
    for iv in intervals:
        p += iv
        notes.append((t, dur, p))
        t += dur
    return notes


def _with_trailing_note(voice: list[tuple[int, int, int]], tick: int = 20000) -> list[tuple[int, int, int]]:
    if not voice:
        return []
    return voice + [(tick, 960, voice[-1][2])]


def test_thematic_recall_caps_approx_only_matches():
    subject = [2, -1, 3, -2, 1]
    approx_a = [3, -2, 2, -1, 2]   # all intervals within ±1 of subject
    approx_b = [1, 0, 4, -3, 0]    # all intervals within ±1 of subject

    v1 = _with_trailing_note(_melody(0, 60, subject))
    v2 = _with_trailing_note(_melody(9000, 55, approx_a))
    v3 = _with_trailing_note(_melody(11000, 67, approx_b))
    v4 = _with_trailing_note(_melody(14000, 48, [1, -1, 2, -2, 1]))

    comp = VoiceComposition(
        voices=[v1, v2, v3, v4],
        key_root=0,
        key_mode="minor",
        source="approx-only",
    )
    score = score_thematic_recall(comp)
    assert 0.15 <= score <= 0.62


def test_thematic_recall_rewards_exact_and_inversion_entries():
    subject = [2, -1, 3, -2, 1]
    inversion = [-2, 1, -3, 2, -1]

    v1 = _with_trailing_note(_melody(0, 60, subject))
    v2 = _with_trailing_note(_melody(9000, 67, subject))
    v3 = _with_trailing_note(_melody(12000, 52, inversion))
    v4 = _with_trailing_note(_melody(15000, 48, [1, -2, 2, -1, 1]))

    comp = VoiceComposition(
        voices=[v1, v2, v3, v4],
        key_root=0,
        key_mode="minor",
        source="exact+inversion",
    )
    score = score_thematic_recall(comp)
    assert score >= 0.68


def test_thematic_recall_monotone_subject_scores_zero():
    monotone = [0, 0, 0, 0, 0]
    v1 = _with_trailing_note(_melody(0, 60, monotone))
    v2 = _with_trailing_note(_melody(9000, 67, monotone))
    v3 = _with_trailing_note(_melody(12000, 52, [1, -1, 1, -1, 1]))

    comp = VoiceComposition(
        voices=[v1, v2, v3],
        key_root=0,
        key_mode="minor",
        source="monotone",
    )
    assert score_thematic_recall(comp) == 0.0
