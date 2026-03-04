"""Cadence and subject conditioning utilities for tokenized training data.

This module detects:
1) Cadential arrivals at bar boundaries (PAC/IAC/HC/DC).
2) Subject-entry spans for fugue-family forms.

It also provides token-level conditioning dropout utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections import deque

from bach_gen.data.extraction import VoiceComposition
from bach_gen.utils.constants import TICKS_PER_QUARTER, ticks_per_measure
from bach_gen.utils.music_theory import get_scale


@dataclass(frozen=True)
class CadenceEvent:
    """Cadence label anchored to a bar boundary tick."""

    tick: int
    bar_index: int
    token_name: str
    confidence: float


@dataclass(frozen=True)
class SubjectEntry:
    """Detected subject-entry span inside one voice."""

    voice_index: int  # 0-based
    start_note_index: int  # index in that voice's start-time-sorted notes
    end_note_index: int
    start_tick: int
    end_tick: int
    match_quality: float
    is_exposition: bool = False


def _sorted_voices(comp: VoiceComposition) -> list[list[tuple[int, int, int]]]:
    return [sorted(v, key=lambda n: n[0]) for v in comp.voices]


def _scale_degree(pc: int, key_root: int, key_mode: str) -> int | None:
    scale = get_scale(key_root, key_mode)
    for idx, scale_pc in enumerate(scale):
        if scale_pc == pc % 12:
            return idx + 1
    return None


def _active_note_index_at_tick(
    notes: list[tuple[int, int, int]],
    tick: int,
) -> int | None:
    for idx, (start, dur, _pitch) in enumerate(notes):
        if start <= tick < start + dur:
            return idx
    return None


def _note_starting_at_tick(
    notes: list[tuple[int, int, int]],
    tick: int,
) -> tuple[int, tuple[int, int, int]] | None:
    for idx, note in enumerate(notes):
        if note[0] == tick:
            return idx, note
    return None


def detect_cadence_events(
    comp: VoiceComposition,
    min_confidence: float = 2.0,
) -> list[CadenceEvent]:
    """Detect cadence labels at bar boundaries.

    Heuristic strategy:
    - Candidate locations are bar boundaries.
    - Classify by bass arrival degree + preceding bass motion + soprano arrival.
    - Keep only precision-biased detections with confidence >= ``min_confidence``.
    """
    voices = _sorted_voices(comp)
    if len(voices) < 2:
        return []

    all_notes = [n for voice in voices for n in voice]
    if not all_notes:
        return []

    bass = voices[-1]
    soprano = voices[0]
    if not bass or not soprano:
        return []

    time_sig = comp.time_signature if hasattr(comp, "time_signature") else (4, 4)
    measure_ticks = ticks_per_measure(time_sig)
    if measure_ticks <= 0:
        measure_ticks = TICKS_PER_QUARTER * 4

    max_tick = max(start + dur for start, dur, _ in all_notes)
    events: list[CadenceEvent] = []

    for boundary_tick in range(measure_ticks, max_tick + 1, measure_ticks):
        # Prefer a bass note that starts at the boundary; otherwise use active.
        bass_exact = _note_starting_at_tick(bass, boundary_tick)
        bass_active_idx = _active_note_index_at_tick(bass, boundary_tick)
        if bass_exact is not None:
            bass_idx, bass_note = bass_exact
        elif bass_active_idx is not None:
            bass_idx = bass_active_idx
            bass_note = bass[bass_idx]
        else:
            continue

        if bass_idx <= 0:
            continue

        prev_bass = bass[bass_idx - 1]
        bass_arrival_pc = bass_note[2] % 12
        bass_arrival_deg = _scale_degree(bass_arrival_pc, comp.key_root, comp.key_mode)
        prev_bass_deg = _scale_degree(prev_bass[2] % 12, comp.key_root, comp.key_mode)

        if bass_arrival_deg is None:
            continue

        bass_motion = bass_note[2] - prev_bass[2]
        dominant_motion = (
            bass_motion in (5, -7)
            or (abs(bass_motion) <= 12 and (bass_motion % 12 == 5))
        )
        predominant_motion = prev_bass_deg in {2, 4}

        # Soprano arrival for cadence quality typing.
        soprano_idx = _active_note_index_at_tick(soprano, boundary_tick)
        soprano_deg = None
        if soprano_idx is not None:
            soprano_deg = _scale_degree(
                soprano[soprano_idx][2] % 12, comp.key_root, comp.key_mode,
            )

        # Rhythmic convergence: many voices sustain long notes or rest at boundary.
        converged = 0
        for voice in voices:
            idx = _active_note_index_at_tick(voice, boundary_tick)
            if idx is None:
                converged += 1
                continue
            start, dur, _ = voice[idx]
            ends_near_boundary = abs((start + dur) - boundary_tick) <= (TICKS_PER_QUARTER // 4)
            if dur >= (2 * TICKS_PER_QUARTER) or ends_near_boundary:
                converged += 1
        rhythmic_convergence = converged >= max(2, math.ceil(len(voices) * 0.5))

        duration_pattern = (
            bass_note[1] >= 2 * TICKS_PER_QUARTER
            or (
                soprano_idx is not None
                and soprano[soprano_idx][1] >= 2 * TICKS_PER_QUARTER
            )
        )

        token_name = None
        confidence = 0.0

        if bass_arrival_deg == 1 and dominant_motion:
            if soprano_deg == 1:
                token_name = "CAD_PAC"
                confidence = 1.0 + float(rhythmic_convergence) + float(duration_pattern)
            elif soprano_deg in {3, 5}:
                token_name = "CAD_IAC"
                confidence = 1.0 + float(rhythmic_convergence) + float(duration_pattern)
        elif bass_arrival_deg == 5 and (predominant_motion or rhythmic_convergence):
            token_name = "CAD_HC"
            confidence = float(predominant_motion) + float(rhythmic_convergence) + float(duration_pattern)
        elif bass_arrival_deg == 6 and dominant_motion:
            token_name = "CAD_DC"
            confidence = 1.0 + float(rhythmic_convergence) + float(duration_pattern)

        if token_name is None:
            continue
        if confidence < min_confidence:
            continue

        bar_index = boundary_tick // measure_ticks
        events.append(
            CadenceEvent(
                tick=boundary_tick,
                bar_index=bar_index,
                token_name=token_name,
                confidence=confidence,
            ),
        )

    # Keep one label per boundary (highest confidence wins).
    dedup: dict[int, CadenceEvent] = {}
    for event in events:
        prev = dedup.get(event.tick)
        if prev is None or event.confidence > prev.confidence:
            dedup[event.tick] = event
    return [dedup[tick] for tick in sorted(dedup.keys())]


def _extract_exposition_subject(
    comp: VoiceComposition,
    default_bars: int = 2,
) -> tuple[int, list[tuple[int, int, int]], int]:
    """Return (voice_index, subject_notes, first_note_index_in_sorted_voice)."""
    voices = _sorted_voices(comp)
    earliest_voice = -1
    earliest_tick = 10**12
    for vi, voice in enumerate(voices):
        if not voice:
            continue
        first_tick = voice[0][0]
        if first_tick < earliest_tick:
            earliest_tick = first_tick
            earliest_voice = vi

    if earliest_voice < 0:
        return -1, [], -1

    time_sig = comp.time_signature if hasattr(comp, "time_signature") else (4, 4)
    measure_ticks = ticks_per_measure(time_sig)
    if measure_ticks <= 0:
        measure_ticks = TICKS_PER_QUARTER * 4
    cutoff_tick = earliest_tick + default_bars * measure_ticks
    gap_threshold = TICKS_PER_QUARTER

    voice = voices[earliest_voice]
    subject: list[tuple[int, int, int]] = []
    for i, note in enumerate(voice):
        start, dur, _ = note
        if start > cutoff_tick:
            break
        subject.append(note)
        if i + 1 < len(voice):
            next_start = voice[i + 1][0]
            if (next_start - (start + dur)) >= gap_threshold and len(subject) >= 3:
                break

    return earliest_voice, subject, 0


def _intervals_from_notes(notes: list[tuple[int, int, int]]) -> list[int]:
    return [notes[i][2] - notes[i - 1][2] for i in range(1, len(notes))]


def _interval_match_quality(
    reference: list[tuple[int, int, int]],
    candidate: list[tuple[int, int, int]],
    tonal_answer_tolerance: int = 1,
    tolerant_prefix_intervals: int = 3,
) -> float:
    ref_iv = _intervals_from_notes(reference)
    cand_iv = _intervals_from_notes(candidate)
    if not ref_iv or len(ref_iv) != len(cand_iv):
        return 0.0

    matches = 0
    for i, (a, b) in enumerate(zip(ref_iv, cand_iv)):
        tol = tonal_answer_tolerance if i < tolerant_prefix_intervals else 0
        if abs(a - b) <= tol:
            matches += 1
    return matches / len(ref_iv)


def detect_subject_entries(
    comp: VoiceComposition,
    min_match_ratio: float = 0.70,
    min_quality: float = 0.80,
    min_notes: int = 4,
) -> list[SubjectEntry]:
    """Detect subject entries across voices using interval matching."""
    voices = _sorted_voices(comp)
    if not voices:
        return []

    exposition_voice, subject_notes, exposition_start_idx = _extract_exposition_subject(comp)
    if exposition_voice < 0 or len(subject_notes) < min_notes:
        return []

    subject_len = len(subject_notes)
    min_notes_required = max(min_notes, math.ceil(subject_len * min_match_ratio))

    entries: list[SubjectEntry] = []
    used_ranges_by_voice: dict[int, list[tuple[int, int]]] = {}

    for voice_idx, voice in enumerate(voices):
        if len(voice) < min_notes_required:
            continue

        i = 0
        while i <= len(voice) - min_notes_required:
            best_end = -1
            best_quality = 0.0

            max_len = min(subject_len, len(voice) - i)
            for cand_len in range(max_len, min_notes_required - 1, -1):
                ref = subject_notes[:cand_len]
                cand = voice[i:i + cand_len]
                quality = _interval_match_quality(ref, cand)
                if quality >= best_quality:
                    best_quality = quality
                    best_end = i + cand_len - 1

            if best_end < i or best_quality < min_quality:
                i += 1
                continue

            overlaps = False
            for used_start, used_end in used_ranges_by_voice.get(voice_idx, []):
                if not (best_end < used_start or i > used_end):
                    overlaps = True
                    break
            if overlaps:
                i += 1
                continue

            start_tick = voice[i][0]
            end_note = voice[best_end]
            end_tick = end_note[0] + end_note[1]
            is_exposition = (
                voice_idx == exposition_voice
                and i == exposition_start_idx
            )
            entries.append(
                SubjectEntry(
                    voice_index=voice_idx,
                    start_note_index=i,
                    end_note_index=best_end,
                    start_tick=start_tick,
                    end_tick=end_tick,
                    match_quality=best_quality,
                    is_exposition=is_exposition,
                ),
            )
            used_ranges_by_voice.setdefault(voice_idx, []).append((i, best_end))
            i = best_end + 1

    # Ensure one exposition entry is always flagged if anything matched there.
    if entries and not any(e.is_exposition for e in entries):
        earliest = min(entries, key=lambda e: e.start_tick)
        idx = entries.index(earliest)
        entries[idx] = SubjectEntry(
            voice_index=earliest.voice_index,
            start_note_index=earliest.start_note_index,
            end_note_index=earliest.end_note_index,
            start_tick=earliest.start_tick,
            end_tick=earliest.end_tick,
            match_quality=earliest.match_quality,
            is_exposition=True,
        )

    entries.sort(key=lambda e: (e.start_tick, e.voice_index, e.start_note_index))
    return entries


def cadence_token_ids_by_tick(
    cadence_events: list[CadenceEvent],
    token_name_to_id: dict[str, int],
) -> dict[int, int]:
    """Convert cadence events to boundary-tick -> token-id map."""
    result: dict[int, int] = {}
    for event in cadence_events:
        tok = token_name_to_id.get(event.token_name)
        if tok is not None:
            result[event.tick] = tok
    return result


def subject_boundary_note_indices(
    entries: list[SubjectEntry],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return (start_markers, end_markers) keyed by (voice_number, note_index)."""
    starts: set[tuple[int, int]] = set()
    ends: set[tuple[int, int]] = set()
    for entry in entries:
        voice_num = entry.voice_index + 1
        starts.add((voice_num, entry.start_note_index))
        ends.add((voice_num, entry.end_note_index))
    return starts, ends


def apply_conditioning_dropout(
    tokens: list[int],
    cadence_token_ids: set[int],
    subject_start_token_ids: set[int],
    subject_end_token_ids: set[int],
    dropout_prob: float,
    rng: random.Random,
    keep_first_subject_entry: bool = True,
) -> list[int]:
    """Apply conditioning dropout to one token sequence.

    - Cadence tokens are dropped independently with probability ``dropout_prob``.
    - Subject start/end markers are dropped in start/end pairs.
    - First subject pair can be preserved (exposition anchor).
    """
    if dropout_prob <= 0.0:
        return list(tokens)
    if dropout_prob >= 1.0:
        dropout_prob = 1.0

    keep = [True] * len(tokens)

    # Pair subject boundaries by order of appearance.
    open_starts: deque[int] = deque()
    pairs: list[tuple[int, int]] = []
    for idx, tok in enumerate(tokens):
        if tok in subject_start_token_ids:
            open_starts.append(idx)
        elif tok in subject_end_token_ids and open_starts:
            pairs.append((open_starts.popleft(), idx))

    for pair_idx, (start_i, end_i) in enumerate(pairs):
        if keep_first_subject_entry and pair_idx == 0:
            continue
        if rng.random() < dropout_prob:
            keep[start_i] = False
            keep[end_i] = False

    out: list[int] = []
    for idx, tok in enumerate(tokens):
        if not keep[idx]:
            continue
        if tok in cadence_token_ids and rng.random() < dropout_prob:
            continue
        out.append(tok)
    return out
