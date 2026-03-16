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

from bach_gen.data.conditioning_config import get_form_thresholds
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


def _arrival_note_around_tick(
    notes: list[tuple[int, int, int]],
    tick: int,
    window_ticks: int,
) -> tuple[int, tuple[int, int, int]] | None:
    best: tuple[int, tuple[int, int, int]] | None = None
    best_key = (10**12, 10**12)
    for idx, note in enumerate(notes):
        start, dur, _ = note
        end = start + dur
        if start <= tick < end:
            priority = 0
            distance = abs(start - tick)
        elif abs(start - tick) <= window_ticks:
            priority = 1
            distance = abs(start - tick)
        elif abs(end - tick) <= window_ticks:
            priority = 2
            distance = abs(end - tick)
        else:
            continue
        candidate_key = (priority, distance)
        if candidate_key < best_key:
            best = (idx, note)
            best_key = candidate_key
    return best


def _boundary_support_features(
    voices: list[list[tuple[int, int, int]]],
    boundary_tick: int,
    phrase_window_ticks: int,
) -> tuple[bool, bool, bool]:
    """Return (rhythmic_convergence, duration_pattern, phrase_support)."""
    converged = 0
    long_holders = 0
    phrase_support_votes = 0

    for voice in voices:
        active_idx = _active_note_index_at_tick(voice, boundary_tick)
        if active_idx is None:
            converged += 1
            phrase_support_votes += 1
            continue

        start, dur, _ = voice[active_idx]
        end = start + dur
        ends_near = abs(end - boundary_tick) <= phrase_window_ticks
        starts_near = abs(start - boundary_tick) <= phrase_window_ticks
        next_start = voice[active_idx + 1][0] if active_idx + 1 < len(voice) else None
        gap_after = next_start is not None and (next_start - end) >= phrase_window_ticks

        if dur >= (2 * TICKS_PER_QUARTER) or ends_near:
            converged += 1
        if dur >= (2 * TICKS_PER_QUARTER):
            long_holders += 1
        if ends_near or starts_near or gap_after:
            phrase_support_votes += 1

    rhythmic_convergence = converged >= max(2, math.ceil(len(voices) * 0.5))
    duration_pattern = long_holders >= max(1, math.ceil(len(voices) * 0.25))
    phrase_support = phrase_support_votes >= max(2, math.ceil(len(voices) * 0.5))
    return rhythmic_convergence, duration_pattern, phrase_support


def _apply_cadence_min_spacing(
    events: list[CadenceEvent],
    *,
    min_spacing_ticks: int,
) -> list[CadenceEvent]:
    if min_spacing_ticks <= 0 or len(events) <= 1:
        return list(events)

    kept: list[CadenceEvent] = []
    for event in sorted(events, key=lambda e: e.tick):
        if not kept:
            kept.append(event)
            continue
        if event.tick - kept[-1].tick < min_spacing_ticks:
            if event.confidence > kept[-1].confidence:
                kept[-1] = event
            continue
        kept.append(event)
    return kept


def _apply_cadence_density_cap(
    events: list[CadenceEvent],
    *,
    measure_ticks: int,
    final_boundary_tick: int,
    max_events_per_32_measures: int,
) -> list[CadenceEvent]:
    if max_events_per_32_measures <= 0 or len(events) <= max_events_per_32_measures:
        return list(events)

    final_region_start = max(0, final_boundary_tick - (2 * measure_ticks))
    keep_ticks: set[int] = {
        event.tick for event in events if event.tick >= final_region_start
    }

    window_ticks = max(measure_ticks, 32 * measure_ticks)
    non_final_events = [event for event in events if event.tick < final_region_start]
    if not non_final_events:
        return [event for event in events if event.tick in keep_ticks]

    max_tick = max(event.tick for event in non_final_events)
    for window_start in range(0, max_tick + window_ticks, window_ticks):
        window_end = window_start + window_ticks
        window_events = [
            event for event in non_final_events
            if window_start <= event.tick < window_end
        ]
        if len(window_events) <= max_events_per_32_measures:
            keep_ticks.update(event.tick for event in window_events)
            continue
        top_events = sorted(
            window_events,
            key=lambda event: (-event.confidence, event.tick),
        )[:max_events_per_32_measures]
        keep_ticks.update(event.tick for event in top_events)

    return [event for event in events if event.tick in keep_ticks]


def _apply_cadence_global_density_cap(
    events: list[CadenceEvent],
    *,
    measure_ticks: int,
    final_boundary_tick: int,
    max_events_per_100_bars: float,
) -> list[CadenceEvent]:
    if max_events_per_100_bars <= 0:
        return list(events)

    measure_count = max(1, int(round(final_boundary_tick / max(1, measure_ticks))))
    max_total_events = int(math.ceil(measure_count * max_events_per_100_bars / 100.0))
    if len(events) <= max_total_events:
        return list(events)

    final_region_start = max(0, final_boundary_tick - (2 * measure_ticks))
    final_events = [event for event in events if event.tick >= final_region_start]
    non_final_events = [event for event in events if event.tick < final_region_start]
    keep_non_final = max(0, max_total_events - len(final_events))
    if len(non_final_events) <= keep_non_final:
        return list(events)

    top_non_final = sorted(
        non_final_events,
        key=lambda event: (-event.confidence, event.tick),
    )[:keep_non_final]
    keep_ticks = {event.tick for event in final_events + top_non_final}
    return [event for event in events if event.tick in keep_ticks]


def detect_cadence_events(
    comp: VoiceComposition,
    min_confidence: float | None = None,
    form: str | None = None,
) -> list[CadenceEvent]:
    """Detect cadence labels at bar boundaries.

    Heuristic strategy:
    - Candidate anchors are measure boundaries, including a rounded-up final bar.
    - Arrival notes may start slightly before/after the boundary if the phrase
      clearly resolves there.
    - Confidence is form-aware and combines bass motion, soprano arrival,
      rhythmic convergence, and phrase-ending support.
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
    form_cfg = get_form_thresholds(form)
    cadence_cfg = form_cfg.get("cadence", {})
    if min_confidence is None:
        min_confidence = float(cadence_cfg.get("min_confidence", 2.0))
    boundary_window_ticks = int(
        round(float(cadence_cfg.get("boundary_window_quarters", 1.0)) * TICKS_PER_QUARTER)
    )
    phrase_window_ticks = int(
        round(float(cadence_cfg.get("phrase_support_quarters", 0.5)) * TICKS_PER_QUARTER)
    )
    min_spacing_measures = float(cadence_cfg.get("min_spacing_measures", 1.0))
    max_events_per_32_measures = int(cadence_cfg.get("max_events_per_32_measures", 32))
    max_events_per_100_bars = float(cadence_cfg.get("max_events_per_100_bars", 100.0))
    require_phrase_support = bool(cadence_cfg.get("require_phrase_support", False))

    events: list[CadenceEvent] = []

    final_boundary_tick = max(measure_ticks, math.ceil(max_tick / measure_ticks) * measure_ticks)
    boundary_ticks = range(measure_ticks, final_boundary_tick + 1, measure_ticks)

    for boundary_tick in boundary_ticks:
        bass_arrival = _arrival_note_around_tick(bass, boundary_tick, boundary_window_ticks)
        if bass_arrival is None:
            continue
        bass_idx, bass_note = bass_arrival

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
        leading_motion = prev_bass_deg == 5
        predominant_motion = prev_bass_deg in {2, 4}

        # Soprano arrival for cadence quality typing.
        soprano_arrival = _arrival_note_around_tick(soprano, boundary_tick, boundary_window_ticks)
        soprano_idx = soprano_arrival[0] if soprano_arrival is not None else None
        soprano_deg = None
        if soprano_idx is not None:
            soprano_deg = _scale_degree(
                soprano[soprano_idx][2] % 12, comp.key_root, comp.key_mode,
            )

        rhythmic_convergence, duration_pattern, phrase_support = _boundary_support_features(
            voices,
            boundary_tick,
            phrase_window_ticks,
        )
        if require_phrase_support and not phrase_support:
            continue
        if not duration_pattern:
            duration_pattern = (
                bass_note[1] >= 2 * TICKS_PER_QUARTER
                or (
                    soprano_idx is not None
                    and soprano[soprano_idx][1] >= 2 * TICKS_PER_QUARTER
                )
            )

        final_boundary = boundary_tick >= final_boundary_tick

        token_name = None
        confidence = 0.0

        if bass_arrival_deg == 1 and (dominant_motion or leading_motion):
            if soprano_deg == 1:
                token_name = "CAD_PAC"
                confidence = (
                    1.2
                    + 0.6 * float(dominant_motion or leading_motion)
                    + 0.55 * float(rhythmic_convergence)
                    + 0.35 * float(duration_pattern)
                    + 0.35 * float(phrase_support)
                    + 0.15 * float(final_boundary)
                )
            elif soprano_deg in {3, 5}:
                token_name = "CAD_IAC"
                confidence = (
                    1.0
                    + 0.6 * float(dominant_motion or leading_motion)
                    + 0.50 * float(rhythmic_convergence)
                    + 0.30 * float(duration_pattern)
                    + 0.30 * float(phrase_support)
                )
        elif bass_arrival_deg == 5 and (predominant_motion or rhythmic_convergence or phrase_support):
            token_name = "CAD_HC"
            confidence = (
                0.9
                + 0.45 * float(predominant_motion)
                + 0.45 * float(rhythmic_convergence)
                + 0.25 * float(duration_pattern)
                + 0.25 * float(phrase_support)
                + 0.10 * float(soprano_deg in {2, 5, 7})
            )
        elif bass_arrival_deg == 6 and dominant_motion:
            token_name = "CAD_DC"
            confidence = (
                1.0
                + 0.60 * float(rhythmic_convergence)
                + 0.30 * float(duration_pattern)
                + 0.30 * float(phrase_support)
            )

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

    pruned = [dedup[tick] for tick in sorted(dedup.keys())]
    pruned = _apply_cadence_min_spacing(
        pruned,
        min_spacing_ticks=int(round(min_spacing_measures * measure_ticks)),
    )
    pruned = _apply_cadence_density_cap(
        pruned,
        measure_ticks=measure_ticks,
        final_boundary_tick=final_boundary_tick,
        max_events_per_32_measures=max_events_per_32_measures,
    )
    pruned = _apply_cadence_global_density_cap(
        pruned,
        measure_ticks=measure_ticks,
        final_boundary_tick=final_boundary_tick,
        max_events_per_100_bars=max_events_per_100_bars,
    )
    return pruned


def _extract_exposition_subject(
    comp: VoiceComposition,
    default_bars: int = 2,
    max_bars: int = 4,
    min_notes_before_gap_break: int = 5,
    min_span_quarters_before_gap_break: float = 4.0,
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
    max_cutoff_tick = earliest_tick + max_bars * measure_ticks
    gap_threshold = TICKS_PER_QUARTER
    min_span_ticks = int(round(min_span_quarters_before_gap_break * TICKS_PER_QUARTER))

    voice = voices[earliest_voice]
    subject: list[tuple[int, int, int]] = []
    for i, note in enumerate(voice):
        start, dur, _ = note
        if start > max_cutoff_tick:
            break
        if start > cutoff_tick and len(subject) >= min_notes_before_gap_break:
            break
        subject.append(note)
        if i + 1 < len(voice):
            next_start = voice[i + 1][0]
            span_ticks = (start + dur) - subject[0][0]
            if (
                (next_start - (start + dur)) >= gap_threshold
                and len(subject) >= min_notes_before_gap_break
                and span_ticks >= min_span_ticks
            ):
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


def _note_span_ticks(notes: list[tuple[int, int, int]]) -> int:
    if not notes:
        return 0
    return (notes[-1][0] + notes[-1][1]) - notes[0][0]


def detect_subject_entries(
    comp: VoiceComposition,
    min_match_ratio: float | None = None,
    min_quality: float | None = None,
    min_notes: int | None = None,
    min_same_voice_spacing_ratio: float | None = None,
    min_span_ratio: float | None = None,
    max_span_ratio: float | None = None,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> list[SubjectEntry]:
    """Detect subject entries across voices using interval matching."""
    voices = _sorted_voices(comp)
    if not voices:
        return []

    form_cfg = get_form_thresholds(form, thresholds)
    subject_cfg = form_cfg.get("subject", {})
    min_match_ratio = float(subject_cfg.get("min_match_ratio", 0.70) if min_match_ratio is None else min_match_ratio)
    min_quality = float(subject_cfg.get("min_quality", 0.80) if min_quality is None else min_quality)
    min_notes = int(subject_cfg.get("min_notes", 4) if min_notes is None else min_notes)
    min_same_voice_spacing_ratio = float(
        subject_cfg.get("min_same_voice_spacing_ratio", 0.50)
        if min_same_voice_spacing_ratio is None
        else min_same_voice_spacing_ratio
    )
    min_span_ratio = float(subject_cfg.get("min_span_ratio", 0.50) if min_span_ratio is None else min_span_ratio)
    max_span_ratio = float(subject_cfg.get("max_span_ratio", 2.50) if max_span_ratio is None else max_span_ratio)

    exposition_voice, subject_notes, exposition_start_idx = _extract_exposition_subject(
        comp,
        default_bars=int(subject_cfg.get("default_bars", 2)),
        max_bars=int(subject_cfg.get("max_bars", 4)),
        min_notes_before_gap_break=int(subject_cfg.get("min_notes_before_gap_break", 5)),
        min_span_quarters_before_gap_break=float(
            subject_cfg.get("min_span_quarters_before_gap_break", 4.0)
        ),
    )
    if exposition_voice < 0 or len(subject_notes) < min_notes:
        return []

    subject_len = len(subject_notes)
    min_notes_required = max(min_notes, math.ceil(subject_len * min_match_ratio))
    subject_span_ticks = max(TICKS_PER_QUARTER, _note_span_ticks(subject_notes))
    min_same_voice_spacing_ticks = max(
        2 * TICKS_PER_QUARTER,
        int(round(subject_span_ticks * min_same_voice_spacing_ratio)),
    )

    entries: list[SubjectEntry] = []
    used_ranges_by_voice: dict[int, list[tuple[int, int]]] = {}
    accepted_starts_by_voice: dict[int, list[int]] = {}

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
                cand_span_ticks = _note_span_ticks(cand)
                span_ratio = cand_span_ticks / max(1, _note_span_ticks(ref))
                if not (min_span_ratio <= span_ratio <= max_span_ratio):
                    continue
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
            if any(
                abs(start_tick - prev_start) < min_same_voice_spacing_ticks
                for prev_start in accepted_starts_by_voice.get(voice_idx, [])
            ):
                i += 1
                continue
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
            accepted_starts_by_voice.setdefault(voice_idx, []).append(start_tick)
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
    cadence_dropout_prob: float,
    subject_dropout_prob: float,
    rng: random.Random,
    keep_first_subject_entry: bool = True,
) -> list[int]:
    """Apply conditioning dropout to one token sequence.

    - Cadence tokens are dropped independently with probability
      ``cadence_dropout_prob``.
    - Subject start/end markers are dropped in start/end pairs with probability
      ``subject_dropout_prob``.
    - First subject pair can be preserved (exposition anchor).
    """
    cadence_dropout_prob = min(max(cadence_dropout_prob, 0.0), 1.0)
    subject_dropout_prob = min(max(subject_dropout_prob, 0.0), 1.0)
    if cadence_dropout_prob <= 0.0 and subject_dropout_prob <= 0.0:
        return list(tokens)

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
        if rng.random() < subject_dropout_prob:
            keep[start_i] = False
            keep[end_i] = False

    out: list[int] = []
    for idx, tok in enumerate(tokens):
        if not keep[idx]:
            continue
        if tok in cadence_token_ids and rng.random() < cadence_dropout_prob:
            continue
        out.append(tok)
    return out
