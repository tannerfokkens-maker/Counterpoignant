"""Structural coherence evaluation.

Measures key consistency, cadence quality, phrase structure, modulation
quality, and length. Voice-count-agnostic: works with any number of voices.

Thematic recurrence is handled by the standalone ``score_thematic_recall``
function (also in this module) — no duplication in the structural composite.
"""

from __future__ import annotations

import numpy as np

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.utils.constants import TICKS_PER_QUARTER
from bach_gen.utils.music_theory import (
    get_scale,
    krumhansl_correlation,
    detect_key,
)


def _get_voices_and_key(item: VoicePair | VoiceComposition):
    """Extract voice lists, key_root, key_mode from either type."""
    if isinstance(item, VoiceComposition):
        voices = [v for v in item.voices if v]
        return voices, item.key_root, item.key_mode
    return [v for v in [item.upper, item.lower] if v], item.key_root, item.key_mode


def score_structural(item: VoicePair | VoiceComposition) -> tuple[float, dict]:
    """Score structural coherence.

    Returns:
        (score 0-1, details dict)
    """
    voices, key_root, key_mode = _get_voices_and_key(item)
    details = {}

    # 1. Length score
    length_score = _score_length(voices)
    details["length"] = length_score

    # 2. Key consistency
    key_score = _score_key_consistency(voices, key_root, key_mode)
    details["key_consistency"] = key_score

    # 3. Cadence quality (improved: checks full cadential patterns)
    cadence_score = _score_cadence(voices, key_root, key_mode)
    details["cadence"] = cadence_score

    # 4. Phrase structure (NEW)
    phrase_score = _score_phrase_structure(voices)
    details["phrase_structure"] = phrase_score

    # 5. Key modulation quality
    modulation_score = _score_modulation(voices, key_root, key_mode)
    details["modulation"] = modulation_score

    # Weighted composite
    score = (
        key_score * 0.25
        + cadence_score * 0.25
        + phrase_score * 0.20
        + modulation_score * 0.15
        + length_score * 0.15
    )

    # Penalize degenerate melodic content
    all_notes = [n for v in voices for n in v]
    if len(all_notes) > 8:
        unique_pitches = len(set(n[2] for n in all_notes))
        if unique_pitches <= 2:
            score *= 0.1
        elif unique_pitches <= 5:
            score *= 0.2 + (unique_pitches - 2) * 0.27

    return score, details


def _score_length(voices: list[list[tuple[int, int, int]]]) -> float:
    """Score based on piece length (prefer 20-60 bars)."""
    all_notes = [n for v in voices for n in v]
    if not all_notes:
        return 0.0

    total_ticks = max(n[0] + n[1] for n in all_notes) - min(n[0] for n in all_notes)
    bars = total_ticks / (TICKS_PER_QUARTER * 4)

    if bars < 4:
        return 0.1
    elif bars < 12:
        return 0.3 + 0.4 * (bars - 4) / 8
    elif bars <= 60:
        return 0.8 + 0.2 * min(1, (bars - 12) / 20)
    else:
        return max(0.5, 1.0 - (bars - 60) / 60)


def _score_key_consistency(
    voices: list[list[tuple[int, int, int]]],
    key_root: int,
    key_mode: str,
) -> float:
    """Score how well the piece stays in its declared key."""
    pc_counts = np.zeros(12)
    for voice in voices:
        for _, _, pitch in voice:
            pc_counts[pitch % 12] += 1

    if pc_counts.sum() == 0:
        return 0.0

    corr = krumhansl_correlation(pc_counts, key_root, key_mode)
    return max(0.0, min(1.0, (corr + 0.2) / 1.2))


def _score_cadence(
    voices: list[list[tuple[int, int, int]]],
    key_root: int,
    key_mode: str,
) -> float:
    """Score cadence quality with full cadential pattern detection.

    Checks:
    - Final cadence: V→I in bass, soprano resolving to tonic/third
    - Internal cadences: phrase-ending patterns throughout the piece
    - Cadential preparation: ii→V→I or IV→V→I patterns
    """
    if not voices:
        return 0.0

    scale = get_scale(key_root, key_mode)
    tonic = key_root
    dominant = (key_root + 7) % 12
    subdominant = (key_root + 5) % 12
    supertonic = scale[1] if len(scale) > 1 else (key_root + 2) % 12

    lowest_voice = voices[-1] if voices else []
    highest_voice = voices[0] if voices else []

    score = 0.0

    # --- Final cadence (0.5 max) ---
    if lowest_voice and highest_voice:
        last_lower = lowest_voice[-1][2] % 12
        last_upper = highest_voice[-1][2] % 12

        # Bass ends on tonic
        if last_lower == tonic:
            score += 0.15

        # Soprano ends on tonic or third
        if last_upper == tonic:
            score += 0.10
        elif last_upper == scale[2] if len(scale) > 2 else -1:
            score += 0.08

        # V→I in bass (authentic cadence)
        if len(lowest_voice) >= 2:
            penult_lower = lowest_voice[-2][2] % 12
            if penult_lower == dominant and last_lower == tonic:
                score += 0.15

        # Full ii→V→I or IV→V→I in bass
        if len(lowest_voice) >= 3:
            antepenult = lowest_voice[-3][2] % 12
            penult = lowest_voice[-2][2] % 12
            final = lowest_voice[-1][2] % 12
            if final == tonic and penult == dominant:
                if antepenult == supertonic or antepenult == subdominant:
                    score += 0.10

    # --- Internal cadences (0.3 max) ---
    # Find phrase boundaries by looking for long notes or rests in all voices
    internal_cadence_score = _score_internal_cadences(
        voices, key_root, key_mode, scale,
    )
    score += internal_cadence_score * 0.30

    # --- Cadence on long notes (0.2 max) ---
    # Final notes should be long (half note or longer)
    if lowest_voice and highest_voice:
        final_bass_dur = lowest_voice[-1][1]
        final_sop_dur = highest_voice[-1][1]
        if final_bass_dur >= TICKS_PER_QUARTER * 2:  # half note
            score += 0.10
        if final_sop_dur >= TICKS_PER_QUARTER * 2:
            score += 0.10

    return min(1.0, score)


def _score_internal_cadences(
    voices: list[list[tuple[int, int, int]]],
    key_root: int,
    key_mode: str,
    scale: list[int],
) -> float:
    """Detect and score internal cadences at phrase boundaries."""
    if not voices or not voices[-1]:
        return 0.3

    tonic = key_root
    dominant = (key_root + 7) % 12
    bass = sorted(voices[-1], key=lambda n: n[0])

    if len(bass) < 4:
        return 0.3

    # Find internal phrase boundaries: notes followed by a gap or long note
    cadence_points = []
    for i in range(len(bass) - 1):
        note_end = bass[i][0] + bass[i][1]
        next_start = bass[i + 1][0]
        gap = next_start - note_end

        is_long = bass[i][1] >= TICKS_PER_QUARTER * 2
        has_gap = gap >= TICKS_PER_QUARTER

        if is_long or has_gap:
            cadence_points.append(i)

    if not cadence_points:
        return 0.3

    good_cadences = 0
    for cp_idx in cadence_points:
        # Check for V→I or I (tonic arrival) at this point
        cp_pc = bass[cp_idx][2] % 12
        if cp_pc == tonic:
            if cp_idx > 0 and bass[cp_idx - 1][2] % 12 == dominant:
                good_cadences += 2  # authentic cadence
            else:
                good_cadences += 1  # tonic arrival
        elif cp_pc == dominant:
            good_cadences += 1  # half cadence

    ratio = good_cadences / (len(cadence_points) * 2)
    return min(1.0, ratio * 2)


def _score_phrase_structure(voices: list[list[tuple[int, int, int]]]) -> float:
    """Score whether the music has clear phrase structure.

    Measures:
    - Presence of detectable phrases (gaps or long notes separating sections)
    - Phrase length regularity (Bach phrases are typically 2, 4, or 8 bars)
    - Variety (not all phrases the same length, but not random either)
    """
    all_notes = [n for v in voices for n in v]
    if len(all_notes) < 16:
        return 0.3

    # Detect phrases by finding time gaps where no voice is active
    sorted_notes = sorted(all_notes, key=lambda n: n[0])
    min_tick = sorted_notes[0][0]
    max_tick = max(n[0] + n[1] for n in sorted_notes)

    # Build activity map
    step = TICKS_PER_QUARTER
    activity = []
    for t in range(int(min_tick), int(max_tick), step):
        active = any(
            n[0] <= t < n[0] + n[1]
            for n in sorted_notes
            if abs(n[0] - t) < TICKS_PER_QUARTER * 8  # only check nearby notes
        )
        activity.append((t, active))

    # Find phrase boundaries (transitions from active to inactive)
    phrase_lengths_ticks = []
    phrase_start = min_tick
    for i in range(1, len(activity)):
        t, active = activity[i]
        prev_active = activity[i - 1][1]
        if prev_active and not active:
            # End of phrase
            phrase_len = t - phrase_start
            if phrase_len > TICKS_PER_QUARTER * 2:  # at least half a bar
                phrase_lengths_ticks.append(phrase_len)
            phrase_start = t
        elif not prev_active and active:
            phrase_start = t

    # Also count the final phrase
    final_len = max_tick - phrase_start
    if final_len > TICKS_PER_QUARTER * 2:
        phrase_lengths_ticks.append(final_len)

    # Alternative: detect phrases from the first voice (melody)
    # using long notes as phrase endings
    if len(phrase_lengths_ticks) < 2 and voices:
        phrase_lengths_ticks = _detect_phrases_from_melody(voices[0])

    if len(phrase_lengths_ticks) < 2:
        return 0.3  # can't detect phrases

    # Convert to bars
    bar_ticks = TICKS_PER_QUARTER * 4
    phrase_bars = [pl / bar_ticks for pl in phrase_lengths_ticks]

    score = 0.0

    # 1. Has detectable phrases (0.3)
    score += 0.3

    # 2. Phrase length regularity (0.4)
    # Check how close phrases are to standard lengths (2, 4, 8 bars)
    standard_lengths = [2, 4, 8]
    regularities = []
    for pb in phrase_bars:
        closest = min(standard_lengths, key=lambda s: abs(pb - s))
        deviation = abs(pb - closest) / closest
        regularities.append(max(0.0, 1.0 - deviation))

    avg_regularity = sum(regularities) / len(regularities) if regularities else 0
    score += avg_regularity * 0.4

    # 3. Phrase variety (0.3) — not all identical, but not all different
    if len(phrase_bars) >= 3:
        bar_rounded = [round(pb) for pb in phrase_bars]
        unique_ratio = len(set(bar_rounded)) / len(bar_rounded)
        # Ideal: ~0.3-0.6 unique ratio
        if unique_ratio < 0.2:
            variety_score = 0.5  # too uniform
        elif unique_ratio <= 0.7:
            variety_score = 1.0
        else:
            variety_score = max(0.3, 1.0 - (unique_ratio - 0.7) * 2)
        score += variety_score * 0.3
    else:
        score += 0.15

    return min(1.0, score)


def _detect_phrases_from_melody(
    voice: list[tuple[int, int, int]],
) -> list[int]:
    """Detect phrase boundaries from a single voice using long notes."""
    if len(voice) < 4:
        return []

    sorted_v = sorted(voice, key=lambda n: n[0])
    long_threshold = TICKS_PER_QUARTER * 1.5  # dotted quarter or longer

    phrase_lengths = []
    phrase_start = sorted_v[0][0]

    for i, (start, dur, _) in enumerate(sorted_v):
        if dur >= long_threshold and i > 0:
            phrase_len = start + dur - phrase_start
            if phrase_len > TICKS_PER_QUARTER * 2:
                phrase_lengths.append(int(phrase_len))
            phrase_start = start + dur

    return phrase_lengths


def _score_modulation(
    voices: list[list[tuple[int, int, int]]],
    key_root: int,
    key_mode: str,
) -> float:
    """Score key modulation quality.

    Rewards visits to closely related keys (dominant, relative minor/major,
    subdominant) and penalizes random key wandering. Bach modulates
    purposefully through the circle of fifths.
    """
    all_notes = sorted([n for v in voices for n in v], key=lambda n: n[0])
    if len(all_notes) < 16:
        return 0.3

    window_size = TICKS_PER_QUARTER * 8  # 2 bars
    total_time = max(n[0] + n[1] for n in all_notes) - min(n[0] for n in all_notes)
    start_time = min(n[0] for n in all_notes)

    keys_found: list[tuple[int, str]] = []

    for window_start in range(int(start_time), int(start_time + total_time), window_size):
        window_notes = [n for n in all_notes
                        if window_start <= n[0] < window_start + window_size]
        if len(window_notes) < 4:
            continue

        pc_counts = np.zeros(12)
        for _, _, pitch in window_notes:
            pc_counts[pitch % 12] += 1

        detected_root, detected_mode, corr = detect_key(pc_counts)
        if corr > 0.5:
            keys_found.append((detected_root, detected_mode))

    unique_keys = set(keys_found)
    n_keys = len(unique_keys)

    if n_keys <= 1:
        return 0.4  # static

    # Score based on whether modulations are to related keys
    # Related keys: dominant, subdominant, relative major/minor,
    # parallel major/minor
    related_keys = _get_related_keys(key_root, key_mode)
    related_count = sum(1 for k in unique_keys if k in related_keys or k == (key_root, key_mode))
    unrelated_count = len(unique_keys) - related_count

    if related_count >= 2 and unrelated_count <= 1:
        key_quality = 1.0
    elif related_count >= 2:
        key_quality = max(0.5, 1.0 - unrelated_count * 0.15)
    else:
        key_quality = 0.4

    # Score based on number of key areas (2-4 ideal)
    if n_keys == 2:
        count_score = 0.7
    elif n_keys <= 4:
        count_score = 1.0
    elif n_keys <= 6:
        count_score = 0.8
    else:
        count_score = max(0.4, 1.0 - (n_keys - 6) * 0.1)

    return key_quality * 0.6 + count_score * 0.4


def _get_related_keys(root: int, mode: str) -> set[tuple[int, str]]:
    """Get the set of closely related keys."""
    related = set()

    dominant = (root + 7) % 12
    subdominant = (root + 5) % 12

    if mode == "major":
        relative_minor = (root + 9) % 12
        related.add((dominant, "major"))
        related.add((subdominant, "major"))
        related.add((relative_minor, "minor"))
        related.add((dominant, "minor"))  # v
        related.add((root, "minor"))  # parallel minor
    else:
        relative_major = (root + 3) % 12
        related.add((dominant, "minor"))
        related.add((subdominant, "minor"))
        related.add((relative_major, "major"))
        related.add((dominant, "major"))  # V (raised leading tone)
        related.add((root, "major"))  # parallel major

    return related


# ======================================================================
# Thematic recall (standalone scorer)
# ======================================================================

def _get_interval_sequence(notes: list[tuple[int, int, int]]) -> list[int]:
    """Get sequence of melodic intervals."""
    return [notes[i][2] - notes[i - 1][2] for i in range(1, len(notes))]


def _extract_fragments(intervals: list[int], length: int) -> list[tuple[int, ...]]:
    """Extract all fragments of given length."""
    return [tuple(intervals[i:i + length]) for i in range(len(intervals) - length + 1)]


def _extract_subject_notes(
    comp: VoicePair | VoiceComposition,
    token_sequence: list[int] | None = None,
    tokenizer=None,
    default_bars: int = 2,
) -> list[tuple[int, int, int]]:
    """Extract the subject: the first melodic phrase of the first entering voice.

    Strategy:
    1. Find the first voice that actually has notes (not necessarily voice 1)
    2. Take notes up to the first significant gap or rest, or first N bars
    """
    voices, _, _ = _get_voices_and_key(comp)
    if not voices:
        return []

    # Find the voice with the earliest note onset
    earliest_voice = None
    earliest_time = float("inf")
    for voice in voices:
        if voice:
            first_onset = min(n[0] for n in voice)
            if first_onset < earliest_time:
                earliest_time = first_onset
                earliest_voice = voice

    if earliest_voice is None:
        return []

    sorted_voice = sorted(earliest_voice, key=lambda n: n[0])

    # Take notes until first significant gap or up to default_bars bars
    cutoff_tick = earliest_time + default_bars * TICKS_PER_QUARTER * 4
    gap_threshold = TICKS_PER_QUARTER  # quarter note gap = phrase break

    subject = []
    for i, (start, dur, pitch) in enumerate(sorted_voice):
        if start > cutoff_tick:
            break
        subject.append((start, dur, pitch))

        # Check for gap to next note
        if i + 1 < len(sorted_voice):
            note_end = start + dur
            next_start = sorted_voice[i + 1][0]
            if next_start - note_end >= gap_threshold and len(subject) >= 3:
                break  # phrase boundary

    return subject


def score_thematic_recall(
    comp: VoicePair | VoiceComposition,
    token_sequence: list[int] | None = None,
    tokenizer=None,
) -> float:
    """Score long-range thematic recall: does the subject recur after the opening?

    Extracts the subject (first melodic phrase), converts to an interval
    sequence, then searches all voices from the second half of the piece
    for matching fragments with:
    - Exact transposition
    - Inversion (negated intervals)
    - Retrograde (reversed intervals)
    - Approximate matching (±1 semitone tolerance)

    Returns:
        Score 0.0-1.0.
    """
    voices, _, _ = _get_voices_and_key(comp)
    subject_notes = _extract_subject_notes(comp, token_sequence, tokenizer)

    if len(subject_notes) < 4:
        return 0.3

    subj_intervals = _get_interval_sequence(subject_notes)
    if len(subj_intervals) < 3:
        return 0.3

    # Skip degenerate subjects (all-zero intervals = monotone)
    if all(i == 0 for i in subj_intervals):
        return 0.0

    # Build fragment sets grouped by length.
    frag_lengths = list(range(min(4, len(subj_intervals)), min(7, len(subj_intervals) + 1)))
    subj_frags_by_len: dict[int, set[tuple[int, ...]]] = {}
    inv_frags_by_len: dict[int, set[tuple[int, ...]]] = {}
    retro_frags_by_len: dict[int, set[tuple[int, ...]]] = {}

    for frag_len in frag_lengths:
        subj_set: set[tuple[int, ...]] = set()
        inv_set: set[tuple[int, ...]] = set()
        retro_set: set[tuple[int, ...]] = set()
        for frag in _extract_fragments(subj_intervals, frag_len):
            subj_set.add(frag)
            inv_set.add(tuple(-i for i in frag))  # inversion
            retro_set.add(tuple(reversed(frag)))  # retrograde
        if subj_set:
            subj_frags_by_len[frag_len] = subj_set
            inv_frags_by_len[frag_len] = inv_set
            retro_frags_by_len[frag_len] = retro_set

    if not subj_frags_by_len:
        return 0.3

    # Search cutoff: use proportional cutoff (after first 25% of the piece)
    all_notes = [n for v in voices for n in v]
    if not all_notes:
        return 0.0
    min_tick = min(n[0] for n in all_notes)
    max_tick = max(n[0] + n[1] for n in all_notes)
    total_duration = max_tick - min_tick
    late_cutoff = min_tick + total_duration * 0.25

    # Search all voices past the cutoff with local deduplication to avoid
    # counting many near-identical sliding-window hits as separate entries.
    exact_entries = 0
    inv_entries = 0
    retro_entries = 0
    approx_entries = 0
    weighted_entries = 0.0
    voices_with_entries: set[int] = set()
    min_entry_gap = TICKS_PER_QUARTER * 4  # one bar

    for voice_idx, voice in enumerate(voices):
        late_notes = [n for n in voice if n[0] >= late_cutoff]
        if len(late_notes) < 4:
            continue
        voice_intervals = _get_interval_sequence(late_notes)
        if len(voice_intervals) < 3:
            continue

        last_entry_tick = -10**12
        voice_had_entry = False

        # Check each local starting position and keep only the strongest
        # match type at that position.
        for i in range(len(voice_intervals)):
            if i >= len(late_notes):
                break
            entry_tick = late_notes[i][0]
            if entry_tick - last_entry_tick < min_entry_gap:
                continue

            best_kind: str | None = None
            best_weight = 0.0

            for frag_len in frag_lengths:
                if i + frag_len > len(voice_intervals):
                    continue
                vf = tuple(voice_intervals[i:i + frag_len])
                subj_set = subj_frags_by_len.get(frag_len, set())
                inv_set = inv_frags_by_len.get(frag_len, set())
                retro_set = retro_frags_by_len.get(frag_len, set())

                if vf in subj_set:
                    best_kind = "exact"
                    best_weight = 1.0
                    break  # strongest possible
                if vf in inv_set and best_weight < 0.85:
                    best_kind = "inv"
                    best_weight = 0.85
                    continue
                if vf in retro_set and best_weight < 0.60:
                    best_kind = "retro"
                    best_weight = 0.60
                    continue

                # Approximate matching: only against same-length fragments.
                if best_weight < 0.28:
                    for sf in subj_set:
                        if all(abs(a - b) <= 1 for a, b in zip(sf, vf)):
                            best_kind = "approx"
                            best_weight = 0.28
                            break

            if best_kind is None:
                continue

            if best_kind == "exact":
                exact_entries += 1
            elif best_kind == "inv":
                inv_entries += 1
            elif best_kind == "retro":
                retro_entries += 1
            else:
                approx_entries += 1

            weighted_entries += best_weight
            last_entry_tick = entry_tick
            voice_had_entry = True

        if voice_had_entry:
            voices_with_entries.add(voice_idx)

    # Approximate-only evidence should not dominate the score.
    approx_effective = min(2, approx_entries)
    weighted_entries = (
        exact_entries * 1.0
        + inv_entries * 0.85
        + retro_entries * 0.60
        + approx_effective * 0.28
    )

    if weighted_entries < 0.25:
        score = 0.0
    elif weighted_entries < 1.0:
        score = 0.12 + (weighted_entries - 0.25) * 0.28
    elif weighted_entries < 2.0:
        score = 0.33 + (weighted_entries - 1.0) * 0.20
    elif weighted_entries < 3.5:
        score = 0.53 + (weighted_entries - 2.0) * 0.14
    else:
        score = 0.74

    # Bonus if entries span multiple voices
    if len(voices_with_entries) >= 2:
        score += 0.10
    if len(voices_with_entries) >= 3:
        score += 0.06

    strong_entries = exact_entries + inv_entries
    transformed_entries = exact_entries + inv_entries + retro_entries
    # Cap high scores when evidence is only approximate or weakly transformed.
    if strong_entries == 0:
        score = min(score, 0.62)
    if transformed_entries == 0:
        score = min(score, 0.48)
    if len(voices_with_entries) <= 1:
        score = min(score, 0.80)

    return min(1.0, score)
