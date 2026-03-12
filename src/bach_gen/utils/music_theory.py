"""Music theory utilities: key parsing, scales, intervals, Krumhansl profiles."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np

from bach_gen.utils.constants import KEY_NAMES, TICKS_PER_QUARTER, ticks_per_measure

# Pitch class names (sharps)
PC_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Pitch class names (flats)
PC_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Name-to-pitch-class mapping
_NAME_TO_PC: dict[str, int] = {}
for i, name in enumerate(PC_NAMES_SHARP):
    _NAME_TO_PC[name] = i
for i, name in enumerate(PC_NAMES_FLAT):
    _NAME_TO_PC[name] = i
# Additional enharmonic aliases
_NAME_TO_PC.update({
    "Cs": 1, "Db": 1, "C#": 1,
    "Ds": 3, "Eb": 3, "D#": 3,
    "Es": 5, "Fb": 4, "E#": 5,
    "Fs": 6, "Gb": 6, "F#": 6,
    "Gs": 8, "Ab": 8, "G#": 8,
    "As": 10, "Bb": 10, "A#": 10,
    "Bs": 0, "Cb": 11, "B#": 0,
})

# Scale intervals
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # natural minor
NATURAL_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]  # alias for scale-degree tokenizer
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]    # alias for scale-degree tokenizer
HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]
MELODIC_MINOR_ASC = [0, 2, 3, 5, 7, 9, 11]
MODAL_INTERVALS: dict[str, list[int]] = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
}

# Krumhansl-Schmuckler key profiles
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                            5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                            4.75, 3.98, 2.69, 3.34, 3.17])

# Interval names
INTERVAL_NAMES = [
    "P1", "m2", "M2", "m3", "M3", "P4", "TT",
    "P5", "m6", "M6", "m7", "M7", "P8",
]

# Consonance ratings (0=dissonant, 1=perfect consonance)
CONSONANCE = {
    0: 1.0,   # unison
    1: 0.0,   # minor 2nd
    2: 0.2,   # major 2nd
    3: 0.8,   # minor 3rd
    4: 0.8,   # major 3rd
    5: 0.6,   # perfect 4th
    6: 0.1,   # tritone
    7: 0.9,   # perfect 5th
    8: 0.7,   # minor 6th
    9: 0.7,   # major 6th
    10: 0.3,  # minor 7th
    11: 0.2,  # major 7th
}


@dataclass(frozen=True)
class ModeFamilyDiagnosis:
    """Mode-family diagnostic used for modal-aware labeling."""

    root_pc: int
    mode_family: str
    system: str
    confidence: float


def parse_key(key_str: str) -> tuple[int, str]:
    """Parse a key string like 'C minor', 'D major', 'Eb minor' into (root_pc, mode).

    Returns:
        (root_pitch_class, mode) where mode is 'major' or 'minor'.
    """
    key_str = key_str.strip()

    # Try formats: "C minor", "C_minor", "Cmin", "C min", "Cm"
    patterns = [
        r"^([A-G][#bs]?)\s*(major|minor|maj|min|M|m)$",
        r"^([A-G][#bs]?)_(major|minor|maj|min)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, key_str, re.IGNORECASE)
        if m:
            root_name = m.group(1)
            mode_str = m.group(2).lower()
            break
    else:
        raise ValueError(f"Cannot parse key: '{key_str}'. Use format like 'C minor' or 'Eb major'.")

    root_pc = note_name_to_pc(root_name)

    if mode_str in ("major", "maj", "m" if mode_str == "M" else ""):
        mode = "major"
    else:
        mode = "minor"

    # Fix: 'M' is major, 'm' is minor
    if m.group(2) == "M":
        mode = "major"
    elif m.group(2) == "m":
        mode = "minor"

    return root_pc, mode


def note_name_to_pc(name: str) -> int:
    """Convert note name (e.g., 'C', 'Eb', 'F#') to pitch class 0-11."""
    name = name.strip()
    if name in _NAME_TO_PC:
        return _NAME_TO_PC[name]
    raise ValueError(f"Unknown note name: '{name}'")


def pc_to_note_name(pc: int, prefer_flat: bool = True) -> str:
    """Convert pitch class 0-11 to note name."""
    pc = pc % 12
    if prefer_flat:
        return PC_NAMES_FLAT[pc]
    return PC_NAMES_SHARP[pc]


def get_scale(root_pc: int, mode: str) -> list[int]:
    """Get scale pitch classes for given root and mode."""
    if mode == "major":
        intervals = MAJOR_SCALE
    elif mode == "minor":
        intervals = HARMONIC_MINOR  # use harmonic minor for leading tone
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return [(root_pc + i) % 12 for i in intervals]


def get_modal_scale(root_pc: int, mode_family: str) -> list[int]:
    """Get modal scale pitch classes for the given root and mode family."""
    if mode_family not in MODAL_INTERVALS:
        raise ValueError(f"Unknown mode family: {mode_family}")
    return [(root_pc + i) % 12 for i in MODAL_INTERVALS[mode_family]]


def get_key_signature_name(root_pc: int, mode: str) -> str:
    """Get a canonical key name for token use."""
    name = pc_to_note_name(root_pc, prefer_flat=True)
    # Normalize for token names
    name = name.replace("#", "s").replace("b", "b")
    key_name = f"{name}_{mode}"

    # Ensure output always matches tokenizer key vocabulary spellings.
    enharmonic_aliases = {
        "Db_minor": "Cs_minor",
        "Gb_major": "Fs_major",
        "Gb_minor": "Fs_minor",
        "Ab_minor": "Gs_minor",
    }
    key_name = enharmonic_aliases.get(key_name, key_name)

    if key_name not in KEY_NAMES:
        raise ValueError(f"Unsupported key token name '{key_name}' for root={root_pc}, mode={mode}")
    return key_name


def midi_to_pc(midi_note: int) -> int:
    """Convert MIDI note number to pitch class."""
    return midi_note % 12


def midi_to_octave(midi_note: int) -> int:
    """Convert MIDI note number to octave (C4 = MIDI 60 = octave 4)."""
    return (midi_note // 12) - 1


def interval_class(semitones: int) -> int:
    """Get interval class (0-6) from semitone distance."""
    ic = abs(semitones) % 12
    if ic > 6:
        ic = 12 - ic
    return ic


def is_consonant(interval_semitones: int) -> bool:
    """Check if an interval is consonant (unison, 3rd, 5th, 6th, octave)."""
    ic = abs(interval_semitones) % 12
    return ic in {0, 3, 4, 7, 8, 9}


def krumhansl_correlation(pitch_class_dist: np.ndarray, root_pc: int, mode: str) -> float:
    """Compute correlation between a pitch class distribution and a Krumhansl profile.

    Args:
        pitch_class_dist: 12-element distribution of pitch classes.
        root_pc: Root pitch class of the key.
        mode: 'major' or 'minor'.

    Returns:
        Pearson correlation coefficient.
    """
    if mode == "major":
        profile = KRUMHANSL_MAJOR
    else:
        profile = KRUMHANSL_MINOR

    # Rotate profile to match root
    rotated = np.roll(profile, root_pc)

    # Normalize both
    dist = pitch_class_dist / (pitch_class_dist.sum() + 1e-10)
    prof = rotated / rotated.sum()

    # Pearson correlation
    corr = np.corrcoef(dist, prof)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def detect_key(pitch_class_counts: np.ndarray) -> tuple[int, str, float]:
    """Detect the most likely key from pitch class counts using Krumhansl profiles.

    Returns:
        (root_pc, mode, correlation)
    """
    best_corr = -2.0
    best_root = 0
    best_mode = "major"

    for root in range(12):
        for mode in ("major", "minor"):
            corr = krumhansl_correlation(pitch_class_counts, root, mode)
            if corr > best_corr:
                best_corr = corr
                best_root = root
                best_mode = mode

    return best_root, best_mode, best_corr


def parse_midi_key_signature(key_sig: str) -> tuple[int, str] | None:
    """Parse a MIDI key-signature string such as ``C`` or ``F#m``."""
    if not key_sig:
        return None
    text = key_sig.strip()
    mode = "minor" if text.endswith("m") else "major"
    root_name = text[:-1] if mode == "minor" else text
    if not root_name:
        return None
    try:
        return note_name_to_pc(root_name), mode
    except ValueError:
        return None


_TEXT_KEY_PATTERN = re.compile(
    r"(?<![a-z])"
    r"([a-g])"
    r"(?:[ _-]?(sharp|flat|s|b))?"
    r"[ _-]?"
    r"(major|minor)"
    r"(?![a-z])",
    re.IGNORECASE,
)


def parse_explicit_key_from_text(text: str) -> tuple[int, str] | None:
    """Parse an explicit textual key hint such as ``a-flat_major`` or ``cminor``.

    This is intentionally conservative: it only fires when a note name and an
    explicit ``major``/``minor`` label occur together in the text.
    """
    if not text:
        return None

    normalized = text.strip().lower()
    for match in _TEXT_KEY_PATTERN.finditer(normalized):
        note_name = match.group(1).upper()
        accidental = (match.group(2) or "").lower()
        mode = match.group(3).lower()

        if accidental in {"flat", "b"}:
            note_name += "b"
        elif accidental in {"sharp", "s"}:
            note_name += "#"

        try:
            return note_name_to_pc(note_name), mode
        except ValueError:
            continue
    return None


def key_to_midi_key_signature(root_pc: int, mode: str) -> str:
    """Return a MIDI key-signature string for a supported key."""
    major_names = {
        0: "C",
        1: "Db",
        2: "D",
        3: "Eb",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "Ab",
        9: "A",
        10: "Bb",
        11: "B",
    }
    minor_names = {
        0: "Cm",
        1: "C#m",
        2: "Dm",
        3: "Ebm",
        4: "Em",
        5: "Fm",
        6: "F#m",
        7: "Gm",
        8: "G#m",
        9: "Am",
        10: "Bbm",
        11: "Bm",
    }
    mapping = major_names if mode == "major" else minor_names
    return mapping[root_pc % 12]


def _final_key_support(
    voices: list[list[tuple[int, int, int]]],
    *,
    root: int,
    mode: str,
) -> float:
    """Return tonic-support bonus from the closing sonority."""
    if not voices:
        return 0.0

    max_tick = max(start + dur for voice in voices for start, dur, _ in voice)
    sample_tick = max(0, max_tick - 1)
    final_pitches: list[int] = []
    final_bass: int | None = None
    final_soprano: int | None = None

    for voice_idx, voice in enumerate(voices):
        sounding = None
        for start, dur, pitch in voice:
            if start <= sample_tick < start + dur:
                sounding = pitch
        if sounding is None and voice:
            sounding = voice[-1][2]
        if sounding is None:
            continue
        final_pitches.append(sounding % 12)
        if voice_idx == 0:
            final_soprano = sounding % 12
        if voice_idx == len(voices) - 1:
            final_bass = sounding % 12

    if not final_pitches:
        return 0.0

    third = (root + (4 if mode == "major" else 3)) % 12
    dominant = (root + 7) % 12
    tonic_triad = {root, third, dominant}
    final_set = set(final_pitches)

    support = 0.0
    if final_bass == root:
        support += 0.48
    elif final_bass == dominant:
        support += 0.06

    if final_soprano == root:
        support += 0.18
    elif final_soprano in tonic_triad:
        support += 0.10

    support += 0.24 * (len(final_set & tonic_triad) / max(1, len(final_set)))

    if final_set.issubset(tonic_triad):
        support += 0.10

    bass_voice = voices[-1] if voices else []
    if bass_voice:
        opening_bass = bass_voice[0][2] % 12
        if opening_bass == root:
            support += 0.04
        elif opening_bass == dominant:
            support += 0.02

    return max(0.0, min(1.0, support))


def _final_mode_support(
    voices: list[list[tuple[int, int, int]]],
    *,
    root: int,
    mode_family: str,
) -> float:
    """Return closing support for a modal final on ``root``."""
    if not voices:
        return 0.0

    scale_set = set(get_modal_scale(root, mode_family))
    max_tick = max(start + dur for voice in voices for start, dur, _ in voice)
    sample_tick = max(0, max_tick - 1)
    final_pitches: list[int] = []
    final_bass: int | None = None
    final_soprano: int | None = None

    for voice_idx, voice in enumerate(voices):
        sounding = None
        for start, dur, pitch in voice:
            if start <= sample_tick < start + dur:
                sounding = pitch
        if sounding is None and voice:
            sounding = voice[-1][2]
        if sounding is None:
            continue
        final_pc = sounding % 12
        final_pitches.append(final_pc)
        if voice_idx == 0:
            final_soprano = final_pc
        if voice_idx == len(voices) - 1:
            final_bass = final_pc

    if not final_pitches:
        return 0.0

    third = (root + MODAL_INTERVALS[mode_family][2]) % 12
    final_set = set(final_pitches)

    support = 0.0
    if final_bass == root:
        support += 0.52
    if final_soprano == root:
        support += 0.18
    elif final_soprano == third:
        support += 0.10

    support += 0.20 * (len(final_set & scale_set) / max(1, len(final_set)))
    if final_set.issubset(scale_set):
        support += 0.10

    bass_voice = voices[-1] if voices else []
    if bass_voice:
        opening_bass = bass_voice[0][2] % 12
        if opening_bass == root:
            support += 0.04

    return max(0.0, min(1.0, support))


def _window_profile(
    all_notes: list[tuple[int, int, int]],
    start_tick: int,
    end_tick: int,
) -> np.ndarray:
    """Return a duration-weighted pitch-class profile over a time window."""
    profile = np.zeros(12)
    for note_start, note_dur, pitch in all_notes:
        overlap = max(0, min(note_start + note_dur, end_tick) - max(note_start, start_tick))
        if overlap <= 0:
            continue
        profile[pitch % 12] += max(0.10, overlap / TICKS_PER_QUARTER)
    return profile


def _mode_quality_support(profile_counts: np.ndarray, root: int, mode: str) -> float:
    """Return support for the candidate mode from its defining scale degrees."""
    total = float(profile_counts.sum()) + 1e-10
    third = (root + (4 if mode == "major" else 3)) % 12
    alt_third = (root + (3 if mode == "major" else 4)) % 12
    sixth = (root + (9 if mode == "major" else 8)) % 12
    alt_sixth = (root + (8 if mode == "major" else 9)) % 12
    seventh = (root + (11 if mode == "major" else 10)) % 12
    alt_seventh = (root + (10 if mode == "major" else 11)) % 12

    support = 0.0
    support += 0.50 * (profile_counts[third] - profile_counts[alt_third])
    support += 0.20 * (profile_counts[sixth] - profile_counts[alt_sixth])
    support += 0.16 * (profile_counts[seventh] - profile_counts[alt_seventh])
    support += 0.14 * profile_counts[root]
    return support / total


def detect_mode_family(
    voices: list[list[tuple[int, int, int]]],
    *,
    time_signature: tuple[int, int] = (4, 4),
) -> ModeFamilyDiagnosis:
    """Detect a diatonic mode family for modal-aware analysis.

    This is intentionally diagnostic: it does not replace the main
    major/minor key detector used for tokenization and evaluation.
    """
    all_notes = [note for voice in voices for note in voice]
    if not all_notes:
        return ModeFamilyDiagnosis(0, "ambiguous", "ambiguous", 0.0)

    measure_ticks = max(TICKS_PER_QUARTER, ticks_per_measure(time_signature))
    max_tick = max(start + dur for start, dur, _ in all_notes)

    duration_counts = np.zeros(12)
    attack_counts = np.zeros(12)
    contextual_counts = np.zeros(12)

    for start, dur, pitch in all_notes:
        pc = pitch % 12
        duration_counts[pc] += max(0.25, dur / TICKS_PER_QUARTER)
        attack_counts[pc] += 1.0
        if start < measure_ticks:
            contextual_counts[pc] += 0.20
        if (start + dur) >= max_tick - measure_ticks:
            contextual_counts[pc] += 0.45 + 0.08 * min(4.0, dur / TICKS_PER_QUARTER)

    profile_counts = duration_counts * 0.70 + attack_counts * 0.15 + contextual_counts
    attack_profile = attack_counts + contextual_counts * 0.4

    ranked: list[tuple[float, int, str]] = []
    total_profile = profile_counts.sum() + 1e-10
    total_attack = attack_profile.sum() + 1e-10

    for root in range(12):
        for mode_family in MODAL_INTERVALS:
            scale_set = set(get_modal_scale(root, mode_family))
            profile_in = sum(profile_counts[pc] for pc in scale_set) / total_profile
            attack_in = sum(attack_profile[pc] for pc in scale_set) / total_attack
            ending_support = _final_mode_support(voices, root=root, mode_family=mode_family)
            score = 0.55 * profile_in + 0.15 * attack_in + 0.30 * ending_support
            ranked.append((score, root, mode_family))

    ranked.sort(reverse=True)
    best_score, best_root, best_mode_family = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = max(0.0, best_score - second_score)
    confidence = max(0.0, min(1.0, best_score * 0.72 + margin * 1.6))

    if best_mode_family in {"ionian", "aeolian"}:
        system = "tonal" if confidence >= 0.60 else "ambiguous"
    else:
        system = "modal" if confidence >= 0.58 else "ambiguous"

    return ModeFamilyDiagnosis(best_root, best_mode_family, system, confidence)


def detect_composition_key(
    voices: list[list[tuple[int, int, int]]],
    *,
    time_signature: tuple[int, int] = (4, 4),
    midi_key_signature: str | None = None,
    style: str | None = None,
    source_key_hint: tuple[int, str] | None = None,
) -> tuple[int, str, float, str]:
    """Detect the key of a composition from note events.

    Prefers explicit MIDI key-signature metadata when available. Otherwise,
    uses a stronger fallback than plain global pitch-class counts by combining
    duration-weighted profiles with tonic support from the ending sonority.
    """
    parsed_meta = parse_midi_key_signature(midi_key_signature or "")
    raw_key_sig = (midi_key_signature or "").strip()
    if parsed_meta is not None and raw_key_sig.endswith("m"):
        return parsed_meta[0], parsed_meta[1], 1.0, "metadata"

    all_notes = [note for voice in voices for note in voice]
    if not all_notes:
        return 0, "major", 0.0, "empty"

    measure_ticks = max(TICKS_PER_QUARTER, ticks_per_measure(time_signature))
    max_tick = max(start + dur for start, dur, _ in all_notes)

    duration_counts = np.zeros(12)
    attack_counts = np.zeros(12)
    contextual_counts = np.zeros(12)

    for start, dur, pitch in all_notes:
        pc = pitch % 12
        attack_counts[pc] += 1.0
        duration_counts[pc] += max(0.25, dur / TICKS_PER_QUARTER)

        if start < measure_ticks:
            contextual_counts[pc] += 0.35
        if (start + dur) >= max_tick - measure_ticks:
            contextual_counts[pc] += 0.55 + 0.10 * min(4.0, dur / TICKS_PER_QUARTER)

    profile_counts = duration_counts * 0.75 + attack_counts * 0.25 + contextual_counts
    attack_profile = attack_counts + contextual_counts * 0.5
    opening_profile = _window_profile(all_notes, 0, min(max_tick, measure_ticks * 4))
    closing_profile = _window_profile(all_notes, max(0, max_tick - measure_ticks * 4), max_tick)

    best_root = 0
    best_mode = "major"
    best_score = -10.0

    metadata_prior: dict[tuple[int, str], float] = {}
    signature_candidates: set[tuple[int, str]] = set()
    if parsed_meta is not None:
        if raw_key_sig.endswith("m"):
            metadata_prior[(parsed_meta[0], parsed_meta[1])] = 0.25
            metadata_prior[((parsed_meta[0] + 3) % 12, "major")] = 0.05
            signature_candidates.add((parsed_meta[0], parsed_meta[1]))
        else:
            signature_candidates.add((parsed_meta[0], "major"))
            signature_candidates.add(((parsed_meta[0] + 9) % 12, "minor"))
            metadata_prior[(parsed_meta[0], "major")] = 0.22
            metadata_prior[((parsed_meta[0] + 9) % 12, "minor")] = 0.16

    hinted_candidates: set[tuple[int, str]] = set()
    if source_key_hint is not None:
        hinted_candidates.add(source_key_hint)
        metadata_prior[source_key_hint] = max(metadata_prior.get(source_key_hint, 0.0), 0.24)

    for root in range(12):
        for mode in ("major", "minor"):
            corr_profile = krumhansl_correlation(profile_counts, root, mode)
            corr_attack = krumhansl_correlation(attack_profile, root, mode)
            corr_opening = krumhansl_correlation(opening_profile, root, mode)
            corr_closing = krumhansl_correlation(closing_profile, root, mode)
            ending_support = _final_key_support(voices, root=root, mode=mode)
            mode_support = _mode_quality_support(
                profile_counts + opening_profile + closing_profile, root, mode
            )
            metadata_bonus = metadata_prior.get((root, mode), 0.0)
            signature_penalty = 0.0
            if signature_candidates and (root, mode) not in signature_candidates:
                signature_penalty = -0.08
            hint_penalty = -0.05 if hinted_candidates and (root, mode) not in hinted_candidates else 0.0
            score = (
                corr_profile * 0.48
                + corr_attack * 0.08
                + corr_opening * 0.08
                + corr_closing * 0.14
                + ending_support * 0.28
                + mode_support * 0.18
                + metadata_bonus
                + signature_penalty
                + hint_penalty
            )
            if score > best_score:
                best_score = score
                best_root = root
                best_mode = mode

    confidence = max(0.0, min(1.0, 0.45 + best_score / 1.6))
    source = "signature+heuristic" if metadata_prior else "heuristic"
    return best_root, best_mode, confidence, source


def parse_note_string(note_str: str) -> Optional[int]:
    """Parse a note string like 'C4', 'Eb5', 'F#3' to MIDI number.

    Returns None if unparseable.
    """
    m = re.match(r"^([A-G][#bs]?)(\d)$", note_str.strip())
    if not m:
        return None
    pc = note_name_to_pc(m.group(1))
    octave = int(m.group(2))
    return pc + (octave + 1) * 12


def midi_to_note_string(midi_note: int, prefer_flat: bool = True) -> str:
    """Convert MIDI note number to string like 'C4'."""
    pc = midi_to_pc(midi_note)
    octave = midi_to_octave(midi_note)
    name = pc_to_note_name(pc, prefer_flat)
    return f"{name}{octave}"


def midi_to_scale_degree(
    midi_pitch: int, key_root_pc: int, mode: str,
) -> tuple[int, int, str]:
    """Convert a MIDI pitch to tonic-relative (octave, degree, accidental).

    Uses natural minor for minor keys.  The octave is relative to the tonic:
        octave = (midi_pitch - key_root_pc) // 12

    Accidental convention: always prefer sharp of the lower degree, so every
    chromatic pitch is exactly 1 semitone above the nearest scale degree.

    Returns:
        (octave, degree_1_based, accidental) where accidental is
        '', 'sharp', or 'flat'.
    """
    if mode == "major":
        scale = MAJOR_SCALE_INTERVALS
    else:
        scale = NATURAL_MINOR_INTERVALS

    # Semitones above the tonic (always positive via modular arithmetic)
    semitones_above_tonic = (midi_pitch - key_root_pc) % 12
    octave = (midi_pitch - key_root_pc) // 12

    if semitones_above_tonic in scale:
        degree_idx = scale.index(semitones_above_tonic)
        return octave, degree_idx + 1, ""

    # Chromatic pitch: prefer sharp of the lower degree.
    # Find the scale degree that is 1 semitone below this pitch.
    lower_semitones = (semitones_above_tonic - 1) % 12
    if lower_semitones in scale:
        degree_idx = scale.index(lower_semitones)
        # If the lower degree wraps around (e.g. sharping degree 7 crosses
        # into the next octave), we keep the octave of the actual pitch.
        return octave, degree_idx + 1, "sharp"

    # Fallback: flat of the upper degree (should be rare with 7-note scales)
    upper_semitones = (semitones_above_tonic + 1) % 12
    if upper_semitones in scale:
        degree_idx = scale.index(upper_semitones)
        return octave, degree_idx + 1, "flat"

    # Shouldn't reach here, but be safe
    return octave, 1, ""


def scale_degree_to_midi(
    octave: int, degree: int, accidental: str,
    key_root_pc: int, mode: str,
) -> int:
    """Convert (octave, degree, accidental) back to absolute MIDI pitch.

    Args:
        octave: Tonic-relative octave (e.g. 5).
        degree: 1-based scale degree (1-7).
        accidental: '', 'sharp', or 'flat'.
        key_root_pc: Pitch class of the key root (0-11).
        mode: 'major' or 'minor'.

    Returns:
        MIDI pitch number.
    """
    if mode == "major":
        scale = MAJOR_SCALE_INTERVALS
    else:
        scale = NATURAL_MINOR_INTERVALS

    semitones = scale[degree - 1]
    if accidental == "sharp":
        semitones += 1
    elif accidental == "flat":
        semitones -= 1

    return key_root_pc + octave * 12 + semitones
