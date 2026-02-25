"""Contrapuntal quality evaluation.

Measures motion types, voice independence, rhythmic complementarity,
and sequential patterns.

Supports N-voice compositions by evaluating all voice pairs.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.utils.constants import TICKS_PER_QUARTER
from bach_gen.utils.music_theory import is_consonant
from bach_gen.utils.voice_index import VoiceIndex


def score_contrapuntal(item: VoicePair | VoiceComposition) -> tuple[float, dict]:
    """Score contrapuntal quality.

    Accepts either a VoicePair or VoiceComposition.
    For multi-voice, evaluates all voice pairs and averages.

    Returns:
        (score 0-1, details dict)
    """
    if isinstance(item, VoicePair):
        return _score_contrapuntal_pair(item.upper, item.lower)

    non_empty = [v for v in item.voices if v]
    if len(non_empty) < 2:
        return 0.0, {"note": "fewer than 2 non-empty voices"}

    pair_scores: list[float] = []
    agg_details: dict[str, float] = {
        "motion_variety": 0.0,
        "voice_independence": 0.0,
        "consonance": 0.0,
        "rhythmic_complementarity": 0.0,
    }

    for i, j in combinations(range(len(non_empty)), 2):
        s, d = _score_contrapuntal_pair(non_empty[i], non_empty[j])
        pair_scores.append(s)
        for k in agg_details:
            agg_details[k] += d.get(k, 0.0)

    n_pairs = len(pair_scores)
    if n_pairs > 0:
        for k in agg_details:
            agg_details[k] /= n_pairs
        avg_score = sum(pair_scores) / n_pairs
    else:
        avg_score = 0.0

    return avg_score, agg_details


def _score_contrapuntal_pair(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> tuple[float, dict]:
    """Score contrapuntal quality of a single voice pair."""
    details = {}

    # 1. Motion types (contrary, oblique, similar, parallel)
    motion_score = _score_motion_types(upper, lower)
    details["motion_variety"] = motion_score

    # 2. Voice independence (different rhythms)
    independence_score = _score_voice_independence(upper, lower)
    details["voice_independence"] = independence_score

    # 3. Consonance profile
    consonance_score = _score_consonance(upper, lower)
    details["consonance"] = consonance_score

    # 4. Rhythmic complementarity
    rhythm_score = _score_rhythmic_complementarity(upper, lower)
    details["rhythmic_complementarity"] = rhythm_score

    # 5. Melodic coherence — real counterpoint uses mostly stepwise motion
    # with intentional leaps. Random pitches produce excessive leaps.
    coherence_score = _score_melodic_coherence(upper, lower)
    details["melodic_coherence"] = coherence_score

    score = (
        motion_score * 0.25
        + independence_score * 0.20
        + consonance_score * 0.20
        + rhythm_score * 0.15
        + coherence_score * 0.20
    )

    # Penalize degenerate content: if voices have very few unique pitches,
    # contrapuntal quality is meaningless regardless of motion/consonance.
    all_notes = upper + lower
    if len(all_notes) > 4:
        unique_pitches = len(set(n[2] for n in all_notes))
        if unique_pitches <= 2:
            score *= 0.1
        elif unique_pitches <= 5:
            score *= 0.2 + (unique_pitches - 2) * 0.27

    return score, details


def _score_motion_types(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> float:
    """Score variety in contrapuntal motion types."""
    aligned = _align_voices_by_onset(upper, lower)
    if len(aligned) < 2:
        return 0.3

    contrary = 0
    similar = 0
    oblique = 0
    parallel = 0

    for i in range(1, len(aligned)):
        prev_u, prev_l = aligned[i - 1]
        curr_u, curr_l = aligned[i]

        if prev_u is None or prev_l is None or curr_u is None or curr_l is None:
            continue

        u_motion = curr_u - prev_u
        l_motion = curr_l - prev_l

        if u_motion == 0 and l_motion == 0:
            continue  # no motion
        elif u_motion == 0 or l_motion == 0:
            oblique += 1
        elif (u_motion > 0) != (l_motion > 0):
            contrary += 1
        elif abs(u_motion) == abs(l_motion):
            parallel += 1
        else:
            similar += 1

    total = contrary + similar + oblique + parallel
    if total == 0:
        return 0.3

    contrary_ratio = contrary / total
    oblique_ratio = oblique / total
    parallel_ratio = parallel / total

    score = 0.0
    score += min(1.0, contrary_ratio / 0.35) * 0.4
    score += min(1.0, oblique_ratio / 0.20) * 0.3
    score += max(0.0, 1.0 - parallel_ratio * 3) * 0.3

    return score


def _score_voice_independence(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> float:
    """Score how independent the two voices are."""
    if not upper or not lower:
        return 0.0

    upper_onsets = set(n[0] for n in upper)
    lower_onsets = set(n[0] for n in lower)

    if not upper_onsets or not lower_onsets:
        return 0.0

    shared = upper_onsets & lower_onsets
    total = upper_onsets | lower_onsets

    shared_ratio = len(shared) / len(total) if total else 0

    if shared_ratio < 0.3:
        score = 0.5 + shared_ratio
    elif shared_ratio < 0.6:
        score = 1.0
    else:
        score = max(0.3, 1.0 - (shared_ratio - 0.6) * 2)

    return score


def _score_consonance(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> float:
    """Score consonance profile."""
    aligned = _align_voices_by_onset(upper, lower)
    if not aligned:
        return 0.5

    consonant_count = 0
    total = 0

    for u_pitch, l_pitch in aligned:
        if u_pitch is None or l_pitch is None:
            continue
        interval = abs(u_pitch - l_pitch)
        total += 1
        if is_consonant(interval):
            consonant_count += 1

    if total == 0:
        return 0.5

    ratio = consonant_count / total

    if ratio < 0.5:
        return ratio
    elif ratio <= 0.9:
        return 0.7 + (ratio - 0.5) * 0.75
    else:
        return max(0.7, 1.0 - (ratio - 0.9) * 3)


def _score_rhythmic_complementarity(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> float:
    """Score whether voices complement each other rhythmically."""
    if not upper or not lower:
        return 0.0

    all_notes = sorted(upper + lower, key=lambda n: n[0])
    if not all_notes:
        return 0.0

    min_time = min(n[0] for n in all_notes)
    max_time = max(n[0] + n[1] for n in all_notes)

    upper_idx = VoiceIndex(upper)
    lower_idx = VoiceIndex(lower)

    step = TICKS_PER_QUARTER // 2
    active_both = 0
    active_any = 0

    for t in range(int(min_time), int(max_time), step):
        u_active = upper_idx.is_active(t)
        l_active = lower_idx.is_active(t)

        if u_active or l_active:
            active_any += 1
        if u_active and l_active:
            active_both += 1

    if active_any == 0:
        return 0.0

    coverage = active_both / active_any
    return min(1.0, coverage * 1.2)


def _align_voices_by_onset(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> list[tuple[int | None, int | None]]:
    """Align voices at onset times."""
    times = sorted(set([n[0] for n in upper] + [n[0] for n in lower]))

    upper_idx = VoiceIndex(upper)
    lower_idx = VoiceIndex(lower)

    return [(upper_idx.pitch_at(t), lower_idx.pitch_at(t)) for t in times]


def _score_melodic_coherence(
    upper: list[tuple[int, int, int]],
    lower: list[tuple[int, int, int]],
) -> float:
    """Score melodic coherence — real counterpoint is mostly stepwise.

    Measures the ratio of stepwise motion (intervals <= 2 semitones) to
    total motion. Bach typically has 60-80% stepwise motion. Random pitches
    produce ~20-30%.
    """
    step_count = 0
    total_count = 0

    for voice in [upper, lower]:
        for i in range(1, len(voice)):
            interval = abs(voice[i][2] - voice[i - 1][2])
            total_count += 1
            if interval <= 2:  # unison, half step, whole step
                step_count += 1

    if total_count == 0:
        return 0.5

    stepwise_ratio = step_count / total_count

    # Target: 50-80% stepwise motion (characteristic of Bach)
    # Below 30%: too jumpy (likely random)
    # Above 90%: too scalic (no melodic interest)
    if stepwise_ratio < 0.2:
        return 0.1
    elif stepwise_ratio < 0.4:
        return 0.1 + (stepwise_ratio - 0.2) * 3.0  # 0.1 to 0.7
    elif stepwise_ratio <= 0.8:
        return 0.7 + (stepwise_ratio - 0.4) * 0.75  # 0.7 to 1.0
    elif stepwise_ratio <= 0.95:
        return 1.0
    else:
        return max(0.6, 1.0 - (stepwise_ratio - 0.95) * 8.0)
