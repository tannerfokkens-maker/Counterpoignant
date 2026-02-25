"""Music theory utilities: key parsing, scales, intervals, Krumhansl profiles."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

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


def get_key_signature_name(root_pc: int, mode: str) -> str:
    """Get a canonical key name for token use."""
    name = pc_to_note_name(root_pc, prefer_flat=True)
    # Normalize for token names
    name = name.replace("#", "s").replace("b", "b")
    return f"{name}_{mode}"


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
