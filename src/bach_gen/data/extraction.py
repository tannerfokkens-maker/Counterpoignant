"""Voice extraction from music21 scores."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import music21
from music21 import note, chord, stream

from bach_gen.utils.constants import (
    UPPER_VOICE_RANGE,
    LOWER_VOICE_RANGE,
    FORM_VOICE_RANGES,
    FORM_DEFAULTS,
    TICKS_PER_QUARTER,
    DIR_TO_FORM,
    bwv_to_form,
)

logger = logging.getLogger(__name__)


@dataclass
class VoicePair:
    """A pair of monophonic voices extracted from a score."""

    upper: list[tuple[int, int, int]]  # (start_tick, duration_ticks, midi_pitch)
    lower: list[tuple[int, int, int]]
    key_root: int  # pitch class 0-11
    key_mode: str  # 'major' or 'minor'
    source: str    # description of source
    style: str = ""  # style category (e.g. "bach", "baroque", "renaissance", "classical")
    time_signature: tuple[int, int] = (4, 4)


@dataclass
class VoiceComposition:
    """N-voice composition extracted from a score (2-4 voices)."""

    voices: list[list[tuple[int, int, int]]]  # N voices, each a list of (start_tick, duration_ticks, midi_pitch)
    key_root: int  # pitch class 0-11
    key_mode: str  # 'major' or 'minor'
    source: str    # description of source
    style: str = ""  # style category (e.g. "bach", "baroque", "renaissance", "classical")
    time_signature: tuple[int, int] = (4, 4)

    @property
    def num_voices(self) -> int:
        return len(self.voices)

    def to_voice_pair(self) -> VoicePair:
        """Convert to VoicePair using first and last voice."""
        return VoicePair(
            upper=self.voices[0] if self.voices else [],
            lower=self.voices[-1] if len(self.voices) > 1 else (self.voices[0] if self.voices else []),
            key_root=self.key_root,
            key_mode=self.key_mode,
            source=self.source,
            style=self.style,
            time_signature=self.time_signature,
        )

    @classmethod
    def from_voice_pair(cls, pair: VoicePair) -> "VoiceComposition":
        """Create a 2-voice VoiceComposition from a VoicePair."""
        return cls(
            voices=[pair.upper, pair.lower],
            key_root=pair.key_root,
            key_mode=pair.key_mode,
            source=pair.source,
            style=pair.style,
            time_signature=pair.time_signature,
        )


def detect_form(score: music21.stream.Score, source: str, style: str) -> tuple[str, int]:
    """Detect form and voice count for a piece.

    Priority order:
    1. BWV number matching (from source string)
    2. Source string keyword matching
    3. Directory-based form (DIR_TO_FORM lookup on source path components)
    4. Voice count + style fallback

    Returns:
        (form_name, num_voices)
    """
    import re

    # 1. BWV number matching
    bwv_match = re.search(r'BWV\s*(\d+)', source, re.IGNORECASE)
    if bwv_match:
        bwv_num = int(bwv_match.group(1))
        form = bwv_to_form(bwv_num)
        if form is not None:
            num_voices = FORM_DEFAULTS[form][0]
            return form, num_voices

    # 2. Source string keyword matching
    source_lower = source.lower()
    keyword_map = [
        ("chorale", "chorale"),
        ("invention", "invention"),
        ("sinfonia", "sinfonia"),
        ("trio sonata", "trio_sonata"),
        ("sonata", "sonata"),
        ("fugue", "fugue"),
        ("quartet", "quartet"),
        ("motet", "motet"),
        ("wtc1f", "fugue"),
        ("wtc2f", "fugue"),
        ("inven", "invention"),   # redundant — already caught by voice count fallback, but explicit is better
    ]
    for keyword, form in keyword_map:
        if keyword in source_lower:
            num_voices = FORM_DEFAULTS[form][0]
            return form, num_voices

    # 3. Directory-based form lookup
    source_parts = source.lower().replace("\\", "/").split("/")
    for part in source_parts:
        if part in DIR_TO_FORM:
            form = DIR_TO_FORM[part]
            num_voices = FORM_DEFAULTS[form][0]
            return form, num_voices

    # 4. Voice count + style fallback
    try:
        n_parts = len(list(score.parts))
    except Exception:
        n_parts = 4

    if n_parts <= 2:
        return "invention", 2
    elif n_parts == 3:
        return "sinfonia", 3
    else:
        # 4+ voices: pick form by style
        num_voices = min(n_parts, 4)
        if style == "renaissance":
            return "motet", num_voices
        elif style == "classical":
            return "quartet", num_voices
        else:
            return "chorale", num_voices


def extract_voice_groups(
    score: music21.stream.Score,
    num_voices: int,
    source: str = "",
    form: str | None = None,
) -> list[VoiceComposition]:
    """Extract N-voice groups from a music21 Score.

    Args:
        score: The parsed score.
        num_voices: Number of voices to extract (2-4).
        source: Description of the source work.
        form: Composition form (e.g. "chorale") used to select the
              best pitch from chords based on voice range.

    Returns:
        List of VoiceComposition with exactly num_voices voices each.
    """
    key_root, key_mode = _detect_key(score)
    time_sig = _detect_time_signature(score)

    parts = list(score.parts)
    if not parts:
        parts = [score]

    voices = []
    for voice_num, part in enumerate(parts, start=1):
        voice_hint: tuple[int, int] | None = None
        if form is not None:
            voice_hint = FORM_VOICE_RANGES.get((form, voice_num))
        notes_list = _part_to_notes(part, voice_hint=voice_hint)
        if notes_list:
            voices.append(notes_list)

    if len(voices) < num_voices:
        logger.debug(f"Fewer than {num_voices} voices in {source}, skipping")
        return []

    results: list[VoiceComposition] = []

    if len(voices) == num_voices:
        comp = VoiceComposition(
            voices=voices[:num_voices],
            key_root=key_root,
            key_mode=key_mode,
            source=source,
            time_signature=time_sig,
        )
        if _validate_voice_group(comp):
            results.append(comp)
    else:
        # Extract consecutive voice groups of the requested size
        for i in range(len(voices) - num_voices + 1):
            group = voices[i:i + num_voices]
            comp = VoiceComposition(
                voices=group,
                key_root=key_root,
                key_mode=key_mode,
                source=f"{source} (voices {i+1}-{i+num_voices})",
                time_signature=time_sig,
            )
            if _validate_voice_group(comp):
                results.append(comp)

    return results


def extract_voice_pairs(
    score: music21.stream.Score,
    source: str = "",
    pair_strategy: str = "adjacent+outer",
    max_pairs: int | None = None,
) -> list[VoicePair]:
    """Extract voice pairs from a music21 Score.

    For 2-voice works: returns one pair.
    For 3-voice works: returns up to 3 pairs (all combinations).
    For 4+ voice works: extracts adjacent voice pairs.

    Args:
        score: The parsed score.
        source: Description of the source work.
        pair_strategy: One of:
            - "adjacent+outer": adjacent pairs plus outer pair (default)
            - "adjacent-only": only adjacent pairs
            - "all-combinations": all unique voice pairs
        max_pairs: Optional cap on returned pairs per work.

    Returns:
        List of VoicePair (possibly capped by ``max_pairs``).
    """
    # Detect key
    key_root, key_mode = _detect_key(score)
    time_sig = _detect_time_signature(score)

    # Get individual voice parts
    parts = list(score.parts)
    if not parts:
        # Try to extract from a flat stream
        parts = [score]

    voices = []
    for part in parts:
        notes_list = _part_to_notes(part)
        if notes_list:
            voices.append(notes_list)

    if len(voices) < 2:
        logger.debug(f"Fewer than 2 voices in {source}, skipping")
        return []

    pairs = []

    if len(voices) == 2:
        pairs.append(VoicePair(
            upper=voices[0],
            lower=voices[1],
            key_root=key_root,
            key_mode=key_mode,
            source=source,
            time_signature=time_sig,
        ))
    else:
        n = len(voices)
        pair_indices: list[tuple[int, int]] = []
        if pair_strategy == "adjacent-only":
            pair_indices = [(i, i + 1) for i in range(n - 1)]
        elif pair_strategy == "all-combinations":
            pair_indices = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        else:
            # Default: adjacent plus outer voices.
            pair_indices = [(i, i + 1) for i in range(n - 1)]
            if n >= 3:
                outer = (0, n - 1)
                if outer not in pair_indices:
                    pair_indices.append(outer)

        for i, j in pair_indices:
            pairs.append(VoicePair(
                upper=voices[i],
                lower=voices[j],
                key_root=key_root,
                key_mode=key_mode,
                source=f"{source} (voices {i+1}-{j+1})",
                time_signature=time_sig,
            ))

    # Filter out pairs where voices overlap too much or are out of range
    valid_pairs = []
    for pair in pairs:
        if _validate_voice_pair(pair):
            valid_pairs.append(pair)

    if max_pairs is not None and max_pairs > 0:
        return valid_pairs[:max_pairs]

    return valid_pairs


def _detect_key(score: music21.stream.Score) -> tuple[int, str]:
    """Detect the key of a score."""
    try:
        key = score.analyze("key")
        root_pc = key.tonic.pitchClass
        mode = "minor" if key.mode == "minor" else "major"
        return root_pc, mode
    except Exception:
        return 0, "major"  # default to C major


def _detect_time_signature(score: music21.stream.Score) -> tuple[int, int]:
    """Detect the predominant time signature of a score."""
    try:
        for ts in score.recurse().getElementsByClass(music21.meter.TimeSignature):
            return (ts.numerator, ts.denominator)
    except Exception:
        pass
    return (4, 4)  # default to 4/4


def _part_to_notes(
    part: music21.stream.Part | music21.stream.Score,
    voice_hint: tuple[int, int] | None = None,
) -> list[tuple[int, int, int]]:
    """Convert a music21 Part to list of (start_tick, duration_ticks, midi_pitch).

    Args:
        part: The music21 part to convert.
        voice_hint: Optional (min_pitch, max_pitch) range. When provided,
                    chords select the pitch closest to the range midpoint
                    instead of always taking the highest note.
    """
    notes_list: list[tuple[int, int, int]] = []

    for element in part.recurse().notesAndRests:
        if isinstance(element, note.Rest):
            continue

        offset_quarters = float(element.offset)
        # Get the absolute offset in the context of the score
        try:
            offset_quarters = float(element.getOffsetInHierarchy(part))
        except Exception:
            pass

        start_tick = int(offset_quarters * TICKS_PER_QUARTER)
        dur_tick = int(float(element.quarterLength) * TICKS_PER_QUARTER)

        if dur_tick <= 0:
            continue

        if isinstance(element, note.Note):
            midi_pitch = element.pitch.midi
            if midi_pitch is not None:
                notes_list.append((start_tick, dur_tick, midi_pitch))
        elif isinstance(element, chord.Chord):
            # For chords, select the best note based on voice context
            pitches = sorted([p.midi for p in element.pitches if p.midi is not None])
            if pitches:
                if voice_hint is not None:
                    # Pick the pitch closest to the midpoint of the voice range
                    midpoint = (voice_hint[0] + voice_hint[1]) / 2
                    best = min(pitches, key=lambda p: abs(p - midpoint))
                else:
                    # Default: take the highest note
                    best = pitches[-1]
                notes_list.append((start_tick, dur_tick, best))

    # Sort by start time, then pitch descending
    notes_list.sort(key=lambda n: (n[0], -n[2]))

    # Remove overlapping notes (keep the first one at each start time)
    cleaned: list[tuple[int, int, int]] = []
    for n in notes_list:
        if cleaned and n[0] < cleaned[-1][0] + cleaned[-1][1]:
            # Overlapping — only add if different start time
            if n[0] == cleaned[-1][0]:
                continue
            # Truncate previous note
            prev = cleaned[-1]
            new_dur = n[0] - prev[0]
            if new_dur > 0:
                cleaned[-1] = (prev[0], new_dur, prev[2])
            cleaned.append(n)
        else:
            cleaned.append(n)

    return cleaned


def _validate_voice_pair(pair: VoicePair) -> bool:
    """Check that a voice pair is reasonable for training."""
    if not pair.upper or not pair.lower:
        return False

    # Need at least ~4 bars of material (~16 quarter notes worth)
    min_ticks = 16 * TICKS_PER_QUARTER
    upper_span = max(n[0] + n[1] for n in pair.upper) - min(n[0] for n in pair.upper)
    lower_span = max(n[0] + n[1] for n in pair.lower) - min(n[0] for n in pair.lower)
    if upper_span < min_ticks or lower_span < min_ticks:
        return False

    # Check that voices don't cross excessively
    # Sample a few timepoints
    crossings = 0
    samples = 0
    for u_note in pair.upper[:20]:
        for l_note in pair.lower[:20]:
            if abs(u_note[0] - l_note[0]) < TICKS_PER_QUARTER // 2:
                samples += 1
                if u_note[2] < l_note[2]:
                    crossings += 1

    if samples > 0 and crossings / samples > 0.5:
        # Swap voices
        pair.upper, pair.lower = pair.lower, pair.upper

    return True


def _validate_voice_group(comp: VoiceComposition) -> bool:
    """Check that an N-voice composition is reasonable for training."""
    if not comp.voices:
        return False

    for voice in comp.voices:
        if not voice:
            return False

    # Need at least ~4 bars of material in every voice
    min_ticks = 16 * TICKS_PER_QUARTER
    for voice in comp.voices:
        span = max(n[0] + n[1] for n in voice) - min(n[0] for n in voice)
        if span < min_ticks:
            return False

    return True


def _median_duration(voice: list[tuple[int, int, int]]) -> float:
    durs = sorted(n[1] for n in voice)
    if not durs:
        return 0.0
    mid = len(durs) // 2
    if len(durs) % 2 == 1:
        return float(durs[mid])
    return float(durs[mid - 1] + durs[mid]) / 2.0


def _notes_per_measure(
    voice: list[tuple[int, int, int]], time_signature: tuple[int, int] = (4, 4)
) -> float:
    if not voice:
        return 0.0
    measure_ticks = TICKS_PER_QUARTER * 4 * time_signature[0] // time_signature[1]
    if measure_ticks <= 0:
        return 0.0
    max_tick = max(s + d for s, d, _ in voice)
    n_measures = max(1, (max_tick + measure_ticks - 1) // measure_ticks)
    return len(voice) / n_measures


def _repeated_cell_ratio(voice: list[tuple[int, int, int]], cell_len: int = 4) -> float:
    """Return max frequency ratio of repeated interval cells in a voice."""
    if len(voice) < cell_len + 1:
        return 0.0
    pitches = [p for _, _, p in sorted(voice, key=lambda n: n[0])]
    cells: list[tuple[int, ...]] = []
    for i in range(len(pitches) - cell_len):
        iv = tuple(pitches[i + k + 1] - pitches[i + k] for k in range(cell_len))
        cells.append(iv)
    if not cells:
        return 0.0
    from collections import Counter

    ctr = Counter(cells)
    return max(ctr.values()) / len(cells)


def accompaniment_texture_severity(
    voices: list[list[tuple[int, int, int]]],
    time_signature: tuple[int, int] = (4, 4),
) -> float:
    """Return accompaniment-likeness severity score.

    Lower means more counterpoint-like; higher means more stereotypical
    melody + broken-chord accompaniment.
    """
    if len(voices) < 2:
        return 0.0

    non_empty = [v for v in voices if v]
    if len(non_empty) < 2:
        return 0.0

    def avg_pitch(v: list[tuple[int, int, int]]) -> float:
        return sum(n[2] for n in v) / len(v)

    ordered = sorted(non_empty, key=avg_pitch)
    bass = ordered[0]
    top = ordered[-1]

    if len(bass) < 24 or len(top) < 8:
        return 0.0

    bass_short_ratio = sum(1 for _, d, _ in bass if d <= TICKS_PER_QUARTER // 2) / len(bass)
    bass_density = _notes_per_measure(bass, time_signature)
    bass_repeat = _repeated_cell_ratio(bass, cell_len=4)
    bass_med_dur = _median_duration(bass)
    top_med_dur = _median_duration(top)
    note_count_ratio = len(bass) / max(1, len(top))

    short_term = bass_short_ratio / 0.70
    density_term = bass_density / 8.0
    repeat_term = bass_repeat / 0.20
    dur_term = (top_med_dur / max(1.0, bass_med_dur)) / 1.8
    count_term = note_count_ratio / 1.5

    return (short_term + density_term + repeat_term + dur_term + count_term) / 5.0


def is_accompaniment_texture_like(
    voices: list[list[tuple[int, int, int]]],
    time_signature: tuple[int, int] = (4, 4),
) -> bool:
    """Heuristic: detect repetitive broken-chord accompaniment texture.

    Intended as a conservative filter for sonata excerpts that look like
    melody + accompaniment rather than multi-voice counterpoint.
    """
    return accompaniment_texture_severity(voices, time_signature) >= 1.0


def is_accompaniment_texture_like_pair(pair: VoicePair) -> bool:
    return is_accompaniment_texture_like(
        [pair.upper, pair.lower], time_signature=pair.time_signature
    )


def is_accompaniment_texture_like_comp(comp: VoiceComposition) -> bool:
    return is_accompaniment_texture_like(comp.voices, time_signature=comp.time_signature)


def accompaniment_texture_severity_pair(pair: VoicePair) -> float:
    return accompaniment_texture_severity(
        [pair.upper, pair.lower], time_signature=pair.time_signature
    )


def accompaniment_texture_severity_comp(comp: VoiceComposition) -> float:
    return accompaniment_texture_severity(comp.voices, time_signature=comp.time_signature)


def is_keyboard_like_source(source: str) -> bool:
    """Best-effort keyword check for keyboard-oriented repertoire."""
    s = source.lower()
    keywords = (
        "sonata",
        "partita",
        "suite",
        "toccata",
        "fantasia",
        "prelude",
        "fughetta",
        "invention",
        "sinfonia",
        "keyboard",
        "klavier",
        "piano",
        "harpsichord",
        "clavier",
    )
    return any(k in s for k in keywords)
