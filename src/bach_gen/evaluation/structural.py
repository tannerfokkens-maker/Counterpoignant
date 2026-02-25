"""Structural coherence evaluation.

Measures subject recurrence, key regions, cadence quality, and length.
Voice-count-agnostic: works with any number of voices.
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

    # 3. Cadence quality
    cadence_score = _score_cadence(voices, key_root, key_mode)
    details["cadence"] = cadence_score

    # 4. Thematic recurrence
    recurrence_score = _score_thematic_recurrence(voices)
    details["thematic_recurrence"] = recurrence_score

    # 5. Key modulation
    modulation_score = _score_modulation(voices, key_root, key_mode)
    details["modulation"] = modulation_score

    # Weighted composite
    score = (
        length_score * 0.15
        + key_score * 0.25
        + cadence_score * 0.25
        + recurrence_score * 0.20
        + modulation_score * 0.15
    )

    # Penalize degenerate melodic content — monotone or near-monotone pieces
    # should not score well on structure regardless of other metrics.
    # Use absolute unique pitch count: 1-2 unique pitches is degenerate,
    # regardless of total note count.
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
    """Score the ending cadence quality."""
    if not voices:
        return 0.0

    scale = get_scale(key_root, key_mode)
    tonic = key_root
    dominant = (key_root + 7) % 12

    # Lowest voice = last in list; highest = first
    lowest_voice = voices[-1] if voices else []
    highest_voice = voices[0] if voices else []

    last_upper = highest_voice[-1][2] % 12 if highest_voice else None
    last_lower = lowest_voice[-1][2] % 12 if lowest_voice else None

    score = 0.0

    # Ends on tonic in bass?
    if last_lower == tonic:
        score += 0.4
    elif last_lower is not None and last_lower in scale:
        score += 0.1

    # Ends on tonic or third in soprano?
    if last_upper == tonic:
        score += 0.3
    elif last_upper is not None and last_upper == (tonic + 4) % 12:
        score += 0.2
    elif last_upper is not None and last_upper == (tonic + 3) % 12:
        score += 0.2

    # Penultimate note: dominant in bass? (V-I cadence)
    if len(lowest_voice) >= 2:
        penult_lower = lowest_voice[-2][2] % 12
        if penult_lower == dominant:
            score += 0.3

    return min(1.0, score)


def _score_thematic_recurrence(voices: list[list[tuple[int, int, int]]]) -> float:
    """Score how much melodic material recurs (transposed or exact)."""
    if len(voices) < 2:
        return 0.3

    # Extract interval sequences for all voices
    interval_seqs = [_get_interval_sequence(v) for v in voices if len(v) > 1]
    if not interval_seqs or all(len(s) < 4 for s in interval_seqs):
        return 0.3

    matches = 0
    total_checks = 0

    for frag_len in [4, 5, 6, 7, 8]:
        all_frags = [_extract_fragments(s, frag_len) for s in interval_seqs]
        all_frags = [f for f in all_frags if f]

        if len(all_frags) < 2:
            continue

        # Check for cross-voice imitation
        for i in range(len(all_frags)):
            for j in range(i + 1, len(all_frags)):
                for frag in all_frags[i]:
                    total_checks += 1
                    if frag in all_frags[j]:
                        matches += 1
                        break

        # Check for within-voice repetition
        for frags in all_frags:
            unique = set(frags)
            if len(frags) > 0:
                repetition_ratio = 1 - len(unique) / len(frags)
                matches += repetition_ratio
                total_checks += 1

    if total_checks == 0:
        return 0.3

    return min(1.0, matches / total_checks * 2)


def _score_modulation(
    voices: list[list[tuple[int, int, int]]],
    key_root: int,
    key_mode: str,
) -> float:
    """Score key modulation (should visit related keys, not stay static)."""
    all_notes = sorted([n for v in voices for n in v], key=lambda n: n[0])
    if len(all_notes) < 16:
        return 0.3

    window_size = TICKS_PER_QUARTER * 8
    total_time = max(n[0] + n[1] for n in all_notes) - min(n[0] for n in all_notes)
    start_time = min(n[0] for n in all_notes)

    keys_found = set()

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
            keys_found.add((detected_root, detected_mode))

    n_keys = len(keys_found)
    if n_keys == 1:
        return 0.4
    elif n_keys == 2:
        return 0.7
    elif n_keys <= 4:
        return 1.0
    else:
        return max(0.5, 1.0 - (n_keys - 4) * 0.1)


def _get_interval_sequence(notes: list[tuple[int, int, int]]) -> list[int]:
    """Get sequence of melodic intervals."""
    intervals = []
    for i in range(1, len(notes)):
        intervals.append(notes[i][2] - notes[i - 1][2])
    return intervals


def _extract_fragments(intervals: list[int], length: int) -> list[tuple[int, ...]]:
    """Extract all fragments of given length."""
    return [tuple(intervals[i:i + length]) for i in range(len(intervals) - length + 1)]


def _extract_subject_notes(
    comp: VoicePair | VoiceComposition,
    token_sequence: list[int] | None = None,
    tokenizer=None,
    default_bars: int = 2,
) -> list[tuple[int, int, int]]:
    """Extract subject from first ~2 bars of voice 1 (or from SUBJECT markers).

    If a token_sequence and tokenizer are provided and SUBJECT_START/END markers
    exist, extract notes between them. Otherwise, fall back to the first
    ``default_bars`` bars of voice 1.
    """
    voices, _, _ = _get_voices_and_key(comp)
    if not voices or not voices[0]:
        return []

    # Try to use SUBJECT markers from token stream
    if token_sequence is not None and tokenizer is not None:
        subj_start_id = getattr(tokenizer, "SUBJECT_START", None)
        subj_end_id = getattr(tokenizer, "SUBJECT_END", None)
        if subj_start_id is not None and subj_end_id is not None:
            try:
                si = token_sequence.index(subj_start_id)
                ei = token_sequence.index(subj_end_id, si)
                # Decode just the subject portion — extract notes from voice 1
                # that fall within the tick range implied by the subject markers.
                # For simplicity, use first voice notes up to the subject length.
                # Count notes between markers by looking for pitch/degree tokens.
                n_notes = 0
                for tok in token_sequence[si:ei]:
                    name = tokenizer.token_to_name.get(tok, "")
                    if name.startswith("Dur_"):
                        n_notes += 1
                if n_notes >= 3:
                    return voices[0][:n_notes]
            except (ValueError, AttributeError):
                pass

    # Fallback: first default_bars bars of voice 1
    voice1 = voices[0]
    if not voice1:
        return []
    min_tick = min(n[0] for n in voice1)
    cutoff = min_tick + default_bars * TICKS_PER_QUARTER * 4
    return [n for n in voice1 if n[0] < cutoff]


def score_thematic_recall(
    comp: VoicePair | VoiceComposition,
    token_sequence: list[int] | None = None,
    tokenizer=None,
) -> float:
    """Score long-range thematic recall: does the subject recur after the opening?

    Extracts the subject (first ~2 bars of voice 1 or SUBJECT markers), converts
    to an interval sequence, then searches all voices from bar 5+ for matching
    interval fragments (exact transposition or inversion).

    Returns:
        Score 0.0-1.0.
    """
    voices, _, _ = _get_voices_and_key(comp)
    subject_notes = _extract_subject_notes(comp, token_sequence, tokenizer)

    if len(subject_notes) < 4:
        return 0.3  # too short to meaningfully score

    subj_intervals = _get_interval_sequence(subject_notes)
    if len(subj_intervals) < 3:
        return 0.3

    # Skip degenerate subjects (all-zero intervals = monotone)
    if all(i == 0 for i in subj_intervals):
        return 0.0

    # Extract fragments of length 4-6 from the subject
    subj_frags: set[tuple[int, ...]] = set()
    inv_frags: set[tuple[int, ...]] = set()
    for frag_len in range(min(4, len(subj_intervals)), min(7, len(subj_intervals) + 1)):
        for frag in _extract_fragments(subj_intervals, frag_len):
            subj_frags.add(frag)
            # Inversion: negate all intervals
            inv_frags.add(tuple(-i for i in frag))

    if not subj_frags:
        return 0.3

    # Determine the cutoff tick (bar 5+)
    all_notes = [n for v in voices for n in v]
    if not all_notes:
        return 0.0
    min_tick = min(n[0] for n in all_notes)
    late_cutoff = min_tick + 4 * TICKS_PER_QUARTER * 4  # after bar 4

    # Search all voices from bar 5+ for matching fragments
    entries = 0
    voices_with_entries: set[int] = set()

    for voice_idx, voice in enumerate(voices):
        late_notes = [n for n in voice if n[0] >= late_cutoff]
        if len(late_notes) < 4:
            continue
        voice_intervals = _get_interval_sequence(late_notes)

        for frag_len in range(min(4, len(subj_intervals)), min(7, len(subj_intervals) + 1)):
            voice_frags = _extract_fragments(voice_intervals, frag_len)
            for vf in voice_frags:
                if vf in subj_frags or vf in inv_frags:
                    entries += 1
                    voices_with_entries.add(voice_idx)
                    break  # count one entry per fragment length per voice

    # Score based on entries found
    if entries == 0:
        return 0.0
    elif entries == 1:
        score = 0.3
    elif entries == 2:
        score = 0.6
    else:
        score = 0.8

    # Bonus if entries span multiple voices
    if len(voices_with_entries) >= 2:
        score += 0.2

    return min(1.0, score)
