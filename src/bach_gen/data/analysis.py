"""Algorithmic analysis of VoiceComposition for conditioning labels.

Computes texture (homophonic/polyphonic/mixed), imitation level
(none/low/high), harmonic rhythm (slow/moderate/fast), and harmonic
tension (low/moderate/high) from note data.

Thresholds were calibrated on a 97-piece sample of the training corpus
(Renaissance motets, Bach chorales, Classical quartets) targeting roughly
even thirds across buckets (p33/p67 split).
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
    intervals), builds a set of interval 4-grams for each voice, then
    counts how many 4-grams from one voice appear in another.  The match
    count is normalised by piece length (total notes).

    Thresholds (calibrated on corpus — p33=0.17, p67=0.27):
        normalised_matches > 0.27 → "high"
        normalised_matches > 0.17 → "low"
        else                      → "none"
    """
    # Extract interval sequences per voice
    voice_intervals: list[list[int]] = []
    for voice in comp.voices:
        if len(voice) < 2:
            voice_intervals.append([])
            continue
        sorted_notes = sorted(voice, key=lambda n: n[0])
        intervals = []
        for i in range(1, len(sorted_notes)):
            intervals.append(sorted_notes[i][2] - sorted_notes[i - 1][2])
        voice_intervals.append(intervals)

    # Build 4-gram sets per voice
    ngram_len = 4
    voice_ngrams: list[set[tuple[int, ...]]] = []
    for intervals in voice_intervals:
        ngrams: set[tuple[int, ...]] = set()
        for i in range(len(intervals) - ngram_len + 1):
            ngrams.add(tuple(intervals[i : i + ngram_len]))
        voice_ngrams.append(ngrams)

    # Count matches across voice pairs
    total_matches = 0
    total_notes = sum(len(v) for v in comp.voices)
    if total_notes == 0:
        return "none"

    for i in range(len(voice_ngrams)):
        for j in range(i + 1, len(voice_ngrams)):
            if voice_ngrams[i] and voice_ngrams[j]:
                matches = voice_ngrams[i] & voice_ngrams[j]
                total_matches += len(matches)

    normalised = total_matches / total_notes

    if normalised > 0.27:
        return "high"
    elif normalised > 0.17:
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

    Combines two signals:
    - **Chromatic ratio**: fraction of notes whose pitch class falls outside
      the key's diatonic scale.
    - **Dissonance ratio**: fraction of simultaneous note pairs that form
      dissonant intervals (minor 2nd, major 2nd, tritone, minor 7th,
      major 7th — intervals 1, 2, 6, 10, 11 in semitones mod 12).

    The tension score is 0.5 * chromatic_ratio + 0.5 * dissonance_ratio.

    Thresholds (calibrated on corpus — p33=0.067, p67=0.110):
        tension > 0.110 → "high"
        tension > 0.067 → "moderate"
        else            → "low"
    """
    from bach_gen.utils.music_theory import get_scale

    scale_pcs = set(get_scale(comp.key_root, comp.key_mode))

    # Chromatic ratio
    total_notes = 0
    chromatic_notes = 0
    for voice in comp.voices:
        for _, _, pitch in voice:
            total_notes += 1
            if (pitch % 12) not in scale_pcs:
                chromatic_notes += 1

    if total_notes == 0:
        return "moderate"

    # Dissonance ratio: at each onset in one voice, check the interval
    # against the note sounding in each other voice
    dissonant_intervals = 0
    interval_count = 0
    dissonant_set = {1, 2, 6, 10, 11}

    for vi in range(len(comp.voices)):
        for vj in range(vi + 1, len(comp.voices)):
            notes_i = sorted(comp.voices[vi], key=lambda n: n[0])
            notes_j = sorted(comp.voices[vj], key=lambda n: n[0])
            for start_i, dur_i, pitch_i in notes_i:
                for start_j, dur_j, pitch_j in notes_j:
                    if start_j <= start_i < start_j + dur_j:
                        interval = abs(pitch_i - pitch_j) % 12
                        interval_count += 1
                        if interval in dissonant_set:
                            dissonant_intervals += 1
                        break  # only the sounding note in voice j

    if interval_count == 0:
        return "moderate"

    chromatic_ratio = chromatic_notes / total_notes
    dissonance_ratio = dissonant_intervals / interval_count
    tension = 0.5 * chromatic_ratio + 0.5 * dissonance_ratio

    if tension > 0.110:
        return "high"
    elif tension > 0.067:
        return "moderate"
    else:
        return "low"


def analyze_composition(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
) -> dict[str, str]:
    """Convenience wrapper returning all four analysis labels."""
    return {
        "texture": compute_texture(comp),
        "imitation": compute_imitation(comp),
        "harmonic_rhythm": compute_harmonic_rhythm(comp, time_sig),
        "harmonic_tension": compute_harmonic_tension(comp),
    }
