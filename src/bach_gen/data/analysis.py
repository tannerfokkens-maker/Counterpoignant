"""Algorithmic analysis of VoiceComposition for conditioning labels.

Computes texture (homophonic/polyphonic/mixed), imitation level
(none/low/high), harmonic rhythm (slow/moderate/fast), harmonic
tension (low/moderate/high), and chromaticism (low/moderate/high)
from note data.

Thresholds were calibrated on a 97-piece sample of the training corpus
(Renaissance motets, Bach chorales, Classical quartets) targeting roughly
even thirds across buckets (p33/p67 split).  Re-run ``bach-gen calibrate``
on the full corpus to update them.
"""

from __future__ import annotations

from bach_gen.data.extraction import VoiceComposition
from bach_gen.utils.constants import (
    TICKS_PER_QUARTER,
    ticks_per_measure,
    beat_tick_positions,
)


def compute_texture(comp: VoiceComposition) -> str:
    """Classify texture as homophonic, polyphonic, or mixed.

    Quantizes onsets to an 8th-note grid (240 ticks at 480 tpq).  For each
    grid slot that has at least one onset, computes the fraction of voices
    that also have an onset there.  The mean of this fraction across all
    active slots is the *synchronisation ratio*.

    Thresholds (calibrated on corpus — p33=0.54, p67=0.60):
        sync_ratio > 0.60  → "homophonic"
        sync_ratio < 0.54  → "polyphonic"
        else               → "mixed"
    """
    grid_quantum = TICKS_PER_QUARTER // 2  # 240 ticks = 8th note

    # Build per-voice onset sets, quantised to grid
    voice_onset_sets: list[set[int]] = []
    for voice in comp.voices:
        onsets: set[int] = set()
        for start, _dur, _pitch in voice:
            slot = start // grid_quantum
            onsets.add(slot)
        voice_onset_sets.append(onsets)

    # Collect all active slots
    all_slots: set[int] = set()
    for onset_set in voice_onset_sets:
        all_slots |= onset_set

    if not all_slots:
        return "mixed"

    n_voices = len(voice_onset_sets)
    if n_voices == 0:
        return "mixed"

    # For each active slot, what fraction of voices have an onset?
    sync_ratios: list[float] = []
    for slot in all_slots:
        count = sum(1 for ons in voice_onset_sets if slot in ons)
        sync_ratios.append(count / n_voices)

    sync_ratio = sum(sync_ratios) / len(sync_ratios)

    if sync_ratio > 0.60:
        return "homophonic"
    elif sync_ratio < 0.54:
        return "polyphonic"
    else:
        return "mixed"


def compute_imitation(comp: VoiceComposition) -> str:
    """Classify imitation level as none, low, or high.

    Extracts melodic interval sequences per voice (consecutive pitch
    intervals) and builds a dict mapping each 4-gram to the list of tick
    positions where it starts.  A cross-voice match counts only when the
    same n-gram (or a single-position ±1 variant, to catch tonal answers
    where one interval shifts by a semitone) appears in two voices starting
    at tick positions that differ by at least MIN_OFFSET ticks.

    The time-offset requirement is what separates true imitation from
    homophony: in a chorale all voices move together, so shared interval
    patterns have zero (or near-zero) time offsets and are excluded.

    MIN_OFFSET = TICKS_PER_QUARTER (one quarter note = 480 ticks at 480 tpq).

    Normalisation: total time-offset match count / total notes.

    Thresholds (validated on Bach kern corpus: 77 chorales → 96% none,
    13 inventions → 69% high, 44 WTC fugues → 93% high):
        normalised_matches > 0.30 → "high"
        normalised_matches > 0.10 → "low"
        else                      → "none"
    """
    ngram_len = 6  # 6-grams: long enough that coincidental cross-voice matches
                   # in homophonic music are rare; short enough to catch 8-note
                   # invention subjects and fugue answers.
    min_offset = TICKS_PER_QUARTER  # ≥1 quarter note time gap required

    # Build per-voice: ngram → [start ticks]
    voice_ngram_times: list[dict[tuple[int, ...], list[int]]] = []
    for voice in comp.voices:
        sorted_notes = sorted(voice, key=lambda n: n[0])
        ngram_time_map: dict[tuple[int, ...], list[int]] = {}
        for i in range(len(sorted_notes) - ngram_len):
            ng = tuple(
                sorted_notes[k + 1][2] - sorted_notes[k][2]
                for k in range(i, i + ngram_len)
            )
            t = sorted_notes[i][0]
            ngram_time_map.setdefault(ng, []).append(t)
        voice_ngram_times.append(ngram_time_map)

    total_notes = sum(len(v) for v in comp.voices)
    if total_notes == 0:
        return "none"

    imitative_matches = 0

    for vi in range(len(voice_ngram_times)):
        for vj in range(vi + 1, len(voice_ngram_times)):
            map_i = voice_ngram_times[vi]
            map_j = voice_ngram_times[vj]
            if not map_i or not map_j:
                continue

            for ng_i, times_i in map_i.items():
                # Exact match + single-position ±1 (catches tonal answers)
                candidates = {ng_i}
                for pos in range(len(ng_i)):
                    for delta in (-1, +1):
                        v = list(ng_i)
                        v[pos] += delta
                        candidates.add(tuple(v))

                for ng_cand in candidates:
                    if ng_cand not in map_j:
                        continue
                    times_j = map_j[ng_cand]
                    for t_i in times_i:
                        for t_j in times_j:
                            if abs(t_j - t_i) >= min_offset:
                                imitative_matches += 1

    normalised = imitative_matches / total_notes

    if normalised > 0.30:
        return "high"
    elif normalised > 0.10:
        return "low"
    else:
        return "none"


def compute_harmonic_rhythm(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
) -> str:
    """Classify harmonic rhythm as slow, moderate, or fast.

    At each beat position, computes the pitch-class set of all sounding
    notes.  Counts beat-to-beat changes in the pitch-class set and divides
    by the number of measures.

    Thresholds (calibrated on corpus — p33=2.77, p67=3.18 changes/measure):
        ≤2.77 → "slow"
        2.77-3.18 → "moderate"
        >3.18  → "fast"
    """
    if time_sig is None:
        time_sig = comp.time_signature if hasattr(comp, "time_signature") else (4, 4)

    measure_ticks = ticks_per_measure(time_sig)
    beat_offsets = beat_tick_positions(time_sig)

    if measure_ticks <= 0:
        return "moderate"

    # Find piece duration
    max_tick = 0
    for voice in comp.voices:
        for start, dur, _pitch in voice:
            end = start + dur
            if end > max_tick:
                max_tick = end

    n_measures = max(1, max_tick // measure_ticks)

    # Build list of absolute beat positions
    abs_beats: list[int] = []
    for m in range(n_measures + 1):
        for offset in beat_offsets:
            abs_tick = m * measure_ticks + offset
            if abs_tick <= max_tick:
                abs_beats.append(abs_tick)
    abs_beats.sort()

    if len(abs_beats) < 2:
        return "moderate"

    # At each beat, compute the pitch-class set of sounding notes
    def pc_set_at(tick: int) -> frozenset[int]:
        pcs: set[int] = set()
        for voice in comp.voices:
            for start, dur, pitch in voice:
                if start <= tick < start + dur:
                    pcs.add(pitch % 12)
        return frozenset(pcs)

    changes = 0
    prev_pcs = pc_set_at(abs_beats[0])
    for beat_tick in abs_beats[1:]:
        cur_pcs = pc_set_at(beat_tick)
        if cur_pcs != prev_pcs and cur_pcs:  # ignore empty beats
            changes += 1
        prev_pcs = cur_pcs

    changes_per_measure = changes / n_measures

    if changes_per_measure <= 2.77:
        return "slow"
    elif changes_per_measure <= 3.18:
        return "moderate"
    else:
        return "fast"


def compute_harmonic_tension(
    comp: VoiceComposition,
) -> str:
    """Classify harmonic tension as low, moderate, or high.

    Measures the **dissonance ratio**: the fraction of simultaneously
    sounding voice pairs whose interval is dissonant (minor 2nd, major 2nd,
    tritone, minor 7th, major 7th — semitone classes 1, 2, 6, 10, 11).

    Sampling is performed at the **union** of both voices' attack times so
    that suspensions (dissonances created when one voice attacks a new
    harmony while another sustains) are not missed.

    Chromaticism is now its own separate token; see ``compute_chromaticism``.

    Thresholds (calibrated on 80-piece Bach corpus sample — p33=0.128,
    p67=0.145; full range 0.070–0.223):
        dissonance_ratio > 0.145 → "high"
        dissonance_ratio > 0.128 → "moderate"
        else                     → "low"
    """
    dissonant_set = {1, 2, 6, 10, 11}
    dissonant_intervals = 0
    interval_count = 0

    for vi in range(len(comp.voices)):
        for vj in range(vi + 1, len(comp.voices)):
            notes_i = sorted(comp.voices[vi], key=lambda n: n[0])
            notes_j = sorted(comp.voices[vj], key=lambda n: n[0])

            # Sample at the union of both voices' attack times
            attacks = sorted(
                {n[0] for n in notes_i} | {n[0] for n in notes_j}
            )

            def _pitch_at(notes: list, tick: int) -> int | None:
                for start, dur, pitch in notes:
                    if start <= tick < start + dur:
                        return pitch
                return None

            for tick in attacks:
                pi = _pitch_at(notes_i, tick)
                pj = _pitch_at(notes_j, tick)
                if pi is not None and pj is not None:
                    interval = abs(pi - pj) % 12
                    interval_count += 1
                    if interval in dissonant_set:
                        dissonant_intervals += 1

    if interval_count == 0:
        return "moderate"

    dissonance_ratio = dissonant_intervals / interval_count

    if dissonance_ratio > 0.145:
        return "high"
    elif dissonance_ratio > 0.128:
        return "moderate"
    else:
        return "low"


def compute_chromaticism(
    comp: VoiceComposition,
) -> str:
    """Classify chromaticism as low, moderate, or high.

    Measures the fraction of note events whose pitch class falls outside
    the diatonic scale of the piece's key.  Harmonic-minor raised 7th
    degrees are treated as diatonic (via ``get_scale`` which returns the
    harmonic minor scale for minor-mode pieces).

    Thresholds (calibrated on corpus — p33=0.05, p67=0.15):
        chromatic_ratio > 0.15 → "high"
        chromatic_ratio > 0.05 → "moderate"
        else                   → "low"
    """
    from bach_gen.utils.music_theory import get_scale

    scale_pcs = set(get_scale(comp.key_root, comp.key_mode))

    total_notes = 0
    chromatic_notes = 0
    for voice in comp.voices:
        for _, _, pitch in voice:
            total_notes += 1
            if (pitch % 12) not in scale_pcs:
                chromatic_notes += 1

    if total_notes == 0:
        return "moderate"

    chromatic_ratio = chromatic_notes / total_notes

    if chromatic_ratio > 0.15:
        return "high"
    elif chromatic_ratio > 0.05:
        return "moderate"
    else:
        return "low"


def analyze_composition(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
) -> dict[str, str]:
    """Convenience wrapper returning all five analysis labels."""
    return {
        "texture": compute_texture(comp),
        "imitation": compute_imitation(comp),
        "harmonic_rhythm": compute_harmonic_rhythm(comp, time_sig),
        "harmonic_tension": compute_harmonic_tension(comp),
        "chromaticism": compute_chromaticism(comp),
    }
