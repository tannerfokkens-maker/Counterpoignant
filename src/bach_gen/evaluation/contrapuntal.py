"""Contrapuntal quality evaluation.

Measures texture and technique quality: sequential patterns, voice
register consistency, stretto, pedal points, contrary motion at cadences,
and rhythmic complementarity.

This scorer focuses on *larger-scale contrapuntal technique* as distinct
from the voice-leading scorer which handles local note-to-note rule
violations (parallel 5ths, voice crossing, etc.).

Supports N-voice compositions by evaluating all voice pairs.
"""

from __future__ import annotations

from itertools import combinations
from collections import Counter

import numpy as np

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.utils.constants import TICKS_PER_QUARTER
from bach_gen.utils.voice_index import VoiceIndex


def score_contrapuntal(item: VoicePair | VoiceComposition) -> tuple[float, dict]:
    """Score contrapuntal quality.

    Accepts either a VoicePair or VoiceComposition.
    For multi-voice, evaluates pairwise and global metrics.

    Returns:
        (score 0-1, details dict)
    """
    if isinstance(item, VoicePair):
        voices = [v for v in [item.upper, item.lower] if v]
    else:
        voices = [v for v in item.voices if v]

    if len(voices) < 2:
        return 0.0, {"note": "fewer than 2 non-empty voices"}

    details: dict[str, float] = {}

    # --- Pairwise metrics (averaged over all voice pairs) ---
    pair_scores_independence = []
    pair_scores_rhythm = []
    pair_scores_contrary_cadence = []

    for i, j in combinations(range(len(voices)), 2):
        pair_scores_independence.append(_score_voice_independence(voices[i], voices[j]))
        pair_scores_rhythm.append(_score_rhythmic_complementarity(voices[i], voices[j]))
        pair_scores_contrary_cadence.append(_score_contrary_motion_at_cadences(voices[i], voices[j]))

    details["voice_independence"] = _mean(pair_scores_independence)
    details["rhythmic_complementarity"] = _mean(pair_scores_rhythm)
    details["contrary_at_cadences"] = _mean(pair_scores_contrary_cadence)

    # --- Global metrics (across all voices) ---
    details["sequential_patterns"] = _score_sequential_patterns(voices)
    details["register_consistency"] = _score_register_consistency(voices)
    details["pedal_points"] = _score_pedal_points(voices)
    details["stretto"] = _score_stretto(voices)
    details["melodic_coherence"] = _score_melodic_coherence(voices)

    # Weighted composite — emphasis on the most discriminating metrics
    score = (
        details["sequential_patterns"] * 0.20
        + details["melodic_coherence"] * 0.18
        + details["register_consistency"] * 0.15
        + details["voice_independence"] * 0.12
        + details["contrary_at_cadences"] * 0.12
        + details["rhythmic_complementarity"] * 0.10
        + details["stretto"] * 0.08
        + details["pedal_points"] * 0.05
    )

    # Penalize degenerate content
    all_notes = [n for v in voices for n in v]
    if len(all_notes) > 4:
        unique_pitches = len(set(n[2] for n in all_notes))
        if unique_pitches <= 2:
            score *= 0.1
        elif unique_pitches <= 5:
            score *= 0.2 + (unique_pitches - 2) * 0.27

    return score, details


# ======================================================================
# Pairwise metrics
# ======================================================================

def _score_voice_independence(
    voice_a: list[tuple[int, int, int]],
    voice_b: list[tuple[int, int, int]],
) -> float:
    """Score rhythmic independence between two voices.

    Measures how often the voices have different onset patterns.
    Ideal counterpoint has ~40-60% shared onsets (voices are independent
    but coordinated). 100% shared = homophonic, 0% = disconnected.
    """
    if not voice_a or not voice_b:
        return 0.0

    a_onsets = set(n[0] for n in voice_a)
    b_onsets = set(n[0] for n in voice_b)
    if not a_onsets or not b_onsets:
        return 0.0

    shared = a_onsets & b_onsets
    total = a_onsets | b_onsets
    shared_ratio = len(shared) / len(total)

    # Sweet spot: 30-60% shared onsets
    if shared_ratio < 0.15:
        return 0.3 + shared_ratio * 2  # voices too disconnected
    elif shared_ratio < 0.30:
        return 0.6 + (shared_ratio - 0.15) * 2.67  # ramping up
    elif shared_ratio <= 0.65:
        return 1.0  # ideal range
    elif shared_ratio <= 0.85:
        return 1.0 - (shared_ratio - 0.65) * 2.5  # too homophonic
    else:
        return max(0.2, 0.5 - (shared_ratio - 0.85) * 2)


def _score_rhythmic_complementarity(
    voice_a: list[tuple[int, int, int]],
    voice_b: list[tuple[int, int, int]],
) -> float:
    """Score whether voices complement each other rhythmically.

    Measures continuous coverage: at any point in time, at least one
    voice should be active.
    """
    if not voice_a or not voice_b:
        return 0.0

    all_notes = sorted(voice_a + voice_b, key=lambda n: n[0])
    if not all_notes:
        return 0.0

    min_time = min(n[0] for n in all_notes)
    max_time = max(n[0] + n[1] for n in all_notes)

    a_idx = VoiceIndex(voice_a)
    b_idx = VoiceIndex(voice_b)

    step = TICKS_PER_QUARTER // 2
    active_both = 0
    active_any = 0

    for t in range(int(min_time), int(max_time), step):
        a_active = a_idx.is_active(t)
        b_active = b_idx.is_active(t)
        if a_active or b_active:
            active_any += 1
        if a_active and b_active:
            active_both += 1

    if active_any == 0:
        return 0.0

    coverage = active_both / active_any
    return min(1.0, coverage * 1.2)


def _score_contrary_motion_at_cadences(
    voice_a: list[tuple[int, int, int]],
    voice_b: list[tuple[int, int, int]],
) -> float:
    """Score use of contrary motion approaching phrase endings.

    Bach reliably uses contrary motion at cadence points — voices
    converge from opposite directions. Shuffled notes have random
    motion at these points.

    Detects phrase boundaries by looking for long notes or rests,
    then checks the 3 notes leading to each boundary.
    """
    if len(voice_a) < 4 or len(voice_b) < 4:
        return 0.5

    # Find potential cadence points: long notes in either voice
    # (duration >= dotted quarter = 720 ticks)
    cadence_threshold = int(TICKS_PER_QUARTER * 1.5)
    cadence_times = set()
    for voice in [voice_a, voice_b]:
        for start, dur, _ in voice:
            if dur >= cadence_threshold:
                cadence_times.add(start)

    if not cadence_times:
        return 0.5  # no detectable cadences

    a_idx = VoiceIndex(voice_a)
    b_idx = VoiceIndex(voice_b)

    contrary_count = 0
    total_cadences = 0

    for ct in cadence_times:
        # Get pitches at 3 time points approaching the cadence
        step = TICKS_PER_QUARTER
        times = [ct - 2 * step, ct - step, ct]

        a_pitches = [a_idx.pitch_at(t) for t in times]
        b_pitches = [b_idx.pitch_at(t) for t in times]

        # Need all 3 pitches in both voices
        if None in a_pitches or None in b_pitches:
            continue

        total_cadences += 1

        # Check for contrary motion in the approach
        a_motion = a_pitches[2] - a_pitches[0]
        b_motion = b_pitches[2] - b_pitches[0]

        if a_motion != 0 and b_motion != 0:
            if (a_motion > 0) != (b_motion > 0):
                contrary_count += 1

    if total_cadences == 0:
        return 0.5

    ratio = contrary_count / total_cadences
    # Bach: ~60-80% contrary at cadences; random: ~50%
    return min(1.0, ratio * 1.3)


# ======================================================================
# Global metrics
# ======================================================================

def _score_sequential_patterns(voices: list[list[tuple[int, int, int]]]) -> float:
    """Score presence of sequential patterns (repeated melodic figures
    transposed to different pitch levels).

    A sequence is when a melodic fragment is repeated starting on a
    different scale degree (same intervals, different absolute pitch).
    This is a hallmark of Baroque counterpoint that shuffling destroys.
    """
    if not voices:
        return 0.0

    total_sequences_found = 0
    total_fragments_checked = 0

    for voice in voices:
        if len(voice) < 8:
            continue

        intervals = _get_interval_sequence(voice)
        if len(intervals) < 6:
            continue

        # Look for repeated interval fragments at different positions
        for frag_len in [3, 4, 5, 6]:
            fragments: dict[tuple[int, ...], list[int]] = {}
            for i in range(len(intervals) - frag_len + 1):
                frag = tuple(intervals[i:i + frag_len])

                # Skip degenerate fragments (all zeros, all same)
                if len(set(frag)) <= 1:
                    continue

                if frag not in fragments:
                    fragments[frag] = []
                fragments[frag].append(i)

            # Count fragments that appear at non-overlapping positions
            for frag, positions in fragments.items():
                total_fragments_checked += 1
                # Find non-overlapping occurrences
                non_overlapping = [positions[0]]
                for p in positions[1:]:
                    if p >= non_overlapping[-1] + frag_len:
                        non_overlapping.append(p)

                if len(non_overlapping) >= 2:
                    # Check that the occurrences start on different pitches
                    # (true transposition, not exact repetition at same pitch)
                    start_pitches = set()
                    for p in non_overlapping:
                        if p < len(voice):
                            start_pitches.add(voice[p][2])
                    if len(start_pitches) >= 2:
                        total_sequences_found += 1

    if total_fragments_checked == 0:
        return 0.3

    ratio = total_sequences_found / total_fragments_checked
    # Bach: ~10-30% of fragments appear as sequences; random: ~1-3%
    if ratio < 0.02:
        return 0.1
    elif ratio < 0.05:
        return 0.1 + (ratio - 0.02) * 10  # 0.1 → 0.4
    elif ratio < 0.15:
        return 0.4 + (ratio - 0.05) * 5  # 0.4 → 0.9
    elif ratio <= 0.35:
        return 0.9 + min(0.1, (ratio - 0.15) * 0.5)
    else:
        return max(0.5, 1.0 - (ratio - 0.35) * 2)  # too repetitive


def _score_register_consistency(voices: list[list[tuple[int, int, int]]]) -> float:
    """Score whether each voice stays in a consistent pitch range.

    Real counterpoint assigns each voice a register and stays mostly
    within it. Shuffled notes (within a voice) randomly jump across
    the entire range, giving much higher per-voice pitch variance.
    """
    if not voices:
        return 0.0

    consistency_scores = []

    for voice in voices:
        if len(voice) < 4:
            continue

        pitches = [n[2] for n in voice]
        voice_range = max(pitches) - min(pitches)

        if voice_range == 0:
            consistency_scores.append(0.0)  # monotone
            continue

        # Measure local smoothness: average absolute interval
        intervals = [abs(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0

        # Measure how often we exceed an octave jump
        large_jumps = sum(1 for iv in intervals if iv > 12)
        jump_ratio = large_jumps / len(intervals) if intervals else 0

        # Bach: avg interval ~2-4 semitones, jump ratio < 5%
        # Shuffled: avg interval ~5-8 semitones, jump ratio ~15-30%
        if avg_interval < 1.5:
            interval_score = 0.6  # too static
        elif avg_interval <= 4.5:
            interval_score = 1.0  # ideal
        elif avg_interval <= 7.0:
            interval_score = 1.0 - (avg_interval - 4.5) * 0.3  # getting jumpy
        else:
            interval_score = max(0.1, 0.25 - (avg_interval - 7.0) * 0.05)

        jump_score = max(0.0, 1.0 - jump_ratio * 8)

        consistency_scores.append(interval_score * 0.6 + jump_score * 0.4)

    return _mean(consistency_scores) if consistency_scores else 0.3


def _score_pedal_points(voices: list[list[tuple[int, int, int]]]) -> float:
    """Score presence of pedal points (sustained bass notes with activity above).

    A pedal point is a long held note (typically in the bass) while other
    voices move freely. Common in Bach, especially in fugues and organ works.
    Rewards pieces that use them, but doesn't penalize pieces that don't.
    """
    if len(voices) < 2:
        return 0.5

    # Check the lowest voice (last in list) for sustained notes
    bass = voices[-1]
    if len(bass) < 2:
        return 0.5

    # A pedal is a note lasting >= 2 bars (4 * TICKS_PER_QUARTER * 2 ticks)
    # or a repeated same-pitch note spanning >= 2 bars
    pedal_threshold = TICKS_PER_QUARTER * 4 * 2

    pedal_found = False
    # Check for single long notes
    for start, dur, pitch in bass:
        if dur >= pedal_threshold:
            pedal_found = True
            break

    # Check for repeated same-pitch notes (rearticulated pedal)
    if not pedal_found and len(bass) >= 4:
        sorted_bass = sorted(bass, key=lambda n: n[0])
        run_start = 0
        for i in range(1, len(sorted_bass)):
            if sorted_bass[i][2] == sorted_bass[run_start][2]:
                span = (sorted_bass[i][0] + sorted_bass[i][1]) - sorted_bass[run_start][0]
                if span >= pedal_threshold:
                    pedal_found = True
                    break
            else:
                run_start = i

    # Pedal points are a nice-to-have, not essential
    return 0.7 if pedal_found else 0.5


def _score_stretto(voices: list[list[tuple[int, int, int]]]) -> float:
    """Score presence of stretto (overlapping subject entries).

    Detects when the same melodic fragment (interval sequence) appears
    in different voices with overlapping time spans. This is a
    sophisticated contrapuntal technique especially common in fugues.
    """
    if len(voices) < 2:
        return 0.3

    # Extract interval sequences per voice with timing info
    voice_data = []
    for voice in voices:
        if len(voice) < 6:
            continue
        intervals = _get_interval_sequence(voice)
        if len(intervals) < 5:
            continue
        voice_data.append((voice, intervals))

    if len(voice_data) < 2:
        return 0.3

    stretto_count = 0

    # For each pair of voices, check for overlapping fragment matches
    for (va, ia), (vb, ib) in combinations(voice_data, 2):
        for frag_len in [4, 5, 6]:
            frags_a = {}
            for i in range(len(ia) - frag_len + 1):
                frag = tuple(ia[i:i + frag_len])
                if len(set(frag)) <= 1:
                    continue
                frags_a.setdefault(frag, []).append(i)

            for j in range(len(ib) - frag_len + 1):
                frag = tuple(ib[j:j + frag_len])
                if frag in frags_a:
                    # Found matching fragment — check for time overlap
                    for ai in frags_a[frag]:
                        if ai < len(va) and j < len(vb):
                            a_start = va[ai][0]
                            a_end = va[min(ai + frag_len, len(va) - 1)][0]
                            b_start = vb[j][0]
                            b_end = vb[min(j + frag_len, len(vb) - 1)][0]

                            # Overlap: one starts before the other ends
                            if (a_start < b_end and b_start < a_end
                                    and a_start != b_start):  # not simultaneous
                                stretto_count += 1
                                break
                    if stretto_count > 0:
                        break  # one per fragment length per pair is enough
            if stretto_count > 0:
                break

    # Stretto is rare and impressive — even one instance is notable
    if stretto_count == 0:
        return 0.3
    elif stretto_count == 1:
        return 0.6
    elif stretto_count <= 3:
        return 0.8
    else:
        return 1.0


def _score_melodic_coherence(voices: list[list[tuple[int, int, int]]]) -> float:
    """Score melodic coherence across all voices.

    Real counterpoint is mostly stepwise with intentional leaps that
    are then resolved by step. Random pitches produce excessive leaps
    with no resolution pattern.

    Measures:
    - Stepwise motion ratio (target: 50-80%)
    - Leap resolution rate (leaps > 4 semitones followed by step in opposite direction)
    - Interval bigram patterns (step-step, leap-step patterns characteristic of tonal music)
    """
    if not voices:
        return 0.0

    step_count = 0
    total_intervals = 0
    resolved_leaps = 0
    total_leaps = 0
    interval_bigrams: Counter = Counter()

    for voice in voices:
        if len(voice) < 3:
            continue

        for i in range(1, len(voice)):
            interval = voice[i][2] - voice[i - 1][2]
            abs_interval = abs(interval)
            total_intervals += 1

            if abs_interval <= 2:
                step_count += 1

            # Track interval bigrams
            if i >= 2:
                prev_interval = voice[i - 1][2] - voice[i - 2][2]
                prev_class = _interval_class(prev_interval)
                curr_class = _interval_class(interval)
                interval_bigrams[(prev_class, curr_class)] += 1

            # Check leap resolution
            if abs_interval > 4:
                total_leaps += 1
                # Check if next note resolves by step in opposite direction
                if i + 1 < len(voice):
                    next_interval = voice[i + 1][2] - voice[i][2]
                    if (abs(next_interval) <= 2  # step
                            and (interval > 0) != (next_interval > 0)):  # opposite dir
                        resolved_leaps += 1

    if total_intervals == 0:
        return 0.3

    # Stepwise ratio score
    stepwise_ratio = step_count / total_intervals
    if stepwise_ratio < 0.25:
        step_score = 0.1
    elif stepwise_ratio < 0.45:
        step_score = 0.1 + (stepwise_ratio - 0.25) * 3.5
    elif stepwise_ratio <= 0.85:
        step_score = 0.8 + (stepwise_ratio - 0.45) * 0.5
    else:
        step_score = max(0.6, 1.0 - (stepwise_ratio - 0.85) * 3)

    # Leap resolution score
    if total_leaps > 0:
        resolution_rate = resolved_leaps / total_leaps
        # Bach: ~50-70% leap resolution; random: ~25-35% (chance)
        leap_score = min(1.0, resolution_rate * 1.5)
    else:
        leap_score = 0.8  # no leaps = very stepwise, which is fine

    # Interval bigram diversity: real music has characteristic patterns
    # (step→step is common, leap→step is common, leap→leap is rare)
    if interval_bigrams:
        total_bigrams = sum(interval_bigrams.values())
        leap_leap = sum(v for (a, b), v in interval_bigrams.items()
                        if a == "leap" and b == "leap")
        ll_ratio = leap_leap / total_bigrams if total_bigrams else 0
        # Bach: leap-leap < 5%; random: ~15-25%
        bigram_score = max(0.0, 1.0 - ll_ratio * 8)
    else:
        bigram_score = 0.5

    return step_score * 0.40 + leap_score * 0.35 + bigram_score * 0.25


# ======================================================================
# Helpers
# ======================================================================

def _get_interval_sequence(voice: list[tuple[int, int, int]]) -> list[int]:
    """Get sequence of melodic intervals."""
    return [voice[i][2] - voice[i - 1][2] for i in range(1, len(voice))]


def _interval_class(interval: int) -> str:
    """Classify an interval as step, skip, or leap."""
    a = abs(interval)
    if a <= 2:
        return "step"
    elif a <= 5:
        return "skip"
    else:
        return "leap"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
