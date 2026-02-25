"""Voice leading quality evaluation.

Checks for parallel 5ths/8ves, voice crossing, augmented intervals,
large leaps, and unresolved dissonances.

Supports N-voice compositions by evaluating all voice pairs (i, j).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.utils.constants import TICKS_PER_QUARTER
from bach_gen.utils.voice_index import VoiceIndex


def score_voice_leading(item: VoicePair | VoiceComposition) -> tuple[float, dict]:
    """Score voice leading quality.

    Accepts either a VoicePair or VoiceComposition.
    For multi-voice compositions, evaluates all voice pairs and averages.

    Returns:
        (score 0-1, details dict)
    """
    if isinstance(item, VoicePair):
        return _score_voice_leading_pair(item.upper, item.lower)

    # VoiceComposition: evaluate all voice pairs (i, j) where i < j
    non_empty = [v for v in item.voices if v]
    if len(non_empty) < 2:
        return 0.0, {"note": "fewer than 2 non-empty voices"}

    pair_scores: list[float] = []
    all_penalties: dict[str, float] = {
        "parallel_fifths": 0.0,
        "parallel_octaves": 0.0,
        "voice_crossing": 0.0,
        "large_leaps": 0.0,
        "augmented_intervals": 0.0,
        "unresolved_leaps": 0.0,
    }

    for i, j in combinations(range(len(non_empty)), 2):
        s, p = _score_voice_leading_pair(non_empty[i], non_empty[j])
        pair_scores.append(s)
        for k in all_penalties:
            all_penalties[k] += p.get(k, 0.0)

    n_pairs = len(pair_scores)
    if n_pairs > 0:
        for k in all_penalties:
            all_penalties[k] /= n_pairs
        avg_score = sum(pair_scores) / n_pairs
    else:
        avg_score = 0.0

    return avg_score, all_penalties


def _score_voice_leading_pair(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> tuple[float, dict]:
    """Score voice leading quality of a single voice pair."""
    penalties = {
        "parallel_fifths": 0.0,
        "parallel_octaves": 0.0,
        "voice_crossing": 0.0,
        "large_leaps": 0.0,
        "augmented_intervals": 0.0,
        "unresolved_leaps": 0.0,
    }

    if not upper or not lower:
        return 0.0, penalties

    # Align voices at common time points
    aligned = _align_voices(upper, lower)

    if len(aligned) < 2:
        return 0.5, penalties

    n_transitions = len(aligned) - 1

    # Check consecutive intervals
    for i in range(1, len(aligned)):
        prev_u, prev_l = aligned[i - 1]
        curr_u, curr_l = aligned[i]

        if prev_u is None or prev_l is None or curr_u is None or curr_l is None:
            continue

        prev_interval = (prev_u - prev_l) % 12
        curr_interval = (curr_u - curr_l) % 12
        u_motion = curr_u - prev_u
        l_motion = curr_l - prev_l
        same_direction = (u_motion != 0 and l_motion != 0
                          and (u_motion > 0) == (l_motion > 0))

        # Parallel fifths (both intervals are perfect 5ths, moving in same direction)
        if prev_interval == 7 and curr_interval == 7 and same_direction:
            penalties["parallel_fifths"] += 1

        # Parallel octaves/unisons (interval class 0 = unison or octave)
        # Only count true parallel motion at the same interval, not doublings
        # at different octaves. Use exact interval, not mod 12.
        exact_prev = abs(prev_u - prev_l)
        exact_curr = abs(curr_u - curr_l)
        if (exact_prev == exact_curr and exact_prev % 12 == 0
                and exact_prev > 0 and same_direction):
            penalties["parallel_octaves"] += 1
        elif prev_interval == 0 and curr_interval == 0 and same_direction:
            # Consecutive unisons moving together
            penalties["parallel_octaves"] += 0.5

        # Voice crossing
        if curr_u < curr_l:
            penalties["voice_crossing"] += 1

    # Check for large leaps in individual voices
    for voice_notes in [upper, lower]:
        prev_pitch = None
        prev_was_leap = False
        for _, _, pitch in voice_notes:
            if prev_pitch is not None:
                interval = abs(pitch - prev_pitch)
                if interval > 12:  # larger than octave
                    penalties["large_leaps"] += 1
                elif interval in (6, 10, 11):  # tritone, m7, M7
                    penalties["augmented_intervals"] += 0.5

                # Unresolved leaps (leap > 4 semitones not followed by step)
                if prev_was_leap and interval > 2:
                    penalties["unresolved_leaps"] += 0.3

                prev_was_leap = interval > 5
            prev_pitch = pitch

    # Normalize penalties
    total_events = max(n_transitions, 1)
    for key in penalties:
        penalties[key] /= total_events

    # Compute score (1.0 = perfect, 0.0 = terrible)
    # Weights are calibrated so that real Bach chorales score ~0.6-0.8.
    # Bach intentionally uses leaps, occasional voice crossing, and
    # tritones, so penalties should be moderate, not devastating.
    penalty_weights = {
        "parallel_fifths": 2.0,
        "parallel_octaves": 2.5,
        "voice_crossing": 1.0,
        "large_leaps": 1.5,
        "augmented_intervals": 1.0,
        "unresolved_leaps": 0.8,
    }

    total_penalty = sum(penalties[k] * penalty_weights[k] for k in penalties)
    score = max(0.0, 1.0 - total_penalty)

    # Penalize lack of melodic movement — a monotone voice has "perfect"
    # voice leading in the penalty sense but is musically degenerate.
    # Use unique pitch count (not ratio) since tonal music naturally reuses
    # pitches: a chorale with 200 notes using 20 pitches is normal.
    all_notes = upper + lower
    if len(all_notes) > 4:
        unique_pitches = len(set(n[2] for n in all_notes))
        if unique_pitches <= 2:
            # 1-2 unique pitches across both voices: fully degenerate
            score *= 0.1
        elif unique_pitches <= 5:
            score *= 0.3 + (unique_pitches - 2) * 0.23

    return score, penalties


def _align_voices(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> list[tuple[int | None, int | None]]:
    """Align two voices at common time points.

    Returns list of (upper_pitch, lower_pitch) at each event onset.
    """
    # Get all unique onset times
    times = sorted(set(
        [n[0] for n in upper] + [n[0] for n in lower]
    ))

    # Build bisect-based indices for O(log n) lookups
    upper_idx = VoiceIndex(upper)
    lower_idx = VoiceIndex(lower)

    aligned = []
    for t in times:
        u = upper_idx.pitch_at(t)
        l = lower_idx.pitch_at(t)
        if u is not None or l is not None:
            aligned.append((u, l))

    return aligned
