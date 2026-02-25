"""Subject parsing and generation."""

from __future__ import annotations

import random
import logging

from bach_gen.data.tokenizer import BachTokenizer
from bach_gen.utils.music_theory import (
    parse_note_string,
    get_scale,
    pc_to_note_name,
    midi_to_scale_degree,
)
from bach_gen.utils.constants import (
    TICKS_PER_QUARTER,
    DURATION_BINS,
    MIN_PITCH,
    MAX_PITCH,
)

logger = logging.getLogger(__name__)


def parse_subject_string(
    subject_str: str,
    tokenizer: BachTokenizer,
) -> list[int]:
    """Parse a user-provided subject string into tokens.

    Expected format: space-separated note names with optional durations.
    e.g., "C4 D4 Eb4 F4" or "C4:q D4:e Eb4:e F4:q"
    (q=quarter, e=eighth, h=half, w=whole, s=sixteenth)

    Returns:
        Token sequence for the subject (without BOS/EOS).
    """
    duration_map = {
        "w": 1920,   # whole
        "h": 960,    # half
        "q": 480,    # quarter
        "e": 240,    # eighth
        "s": 120,    # sixteenth
        "dq": 720,   # dotted quarter
        "dh": 1440,  # dotted half
        "de": 360,   # dotted eighth
    }

    tokens = [tokenizer.SUBJECT_START, tokenizer.VOICE_1]
    current_time = 0

    parts = subject_str.strip().split()
    for part in parts:
        # Parse note:duration or just note (default quarter)
        if ":" in part:
            note_str, dur_str = part.split(":", 1)
            dur_ticks = duration_map.get(dur_str, TICKS_PER_QUARTER)
        else:
            note_str = part
            dur_ticks = TICKS_PER_QUARTER

        midi_note = parse_note_string(note_str)
        if midi_note is None:
            logger.warning(f"Could not parse note: {note_str}")
            continue

        # Pitch token
        pitch_tok = tokenizer._pitch_to_token(midi_note)
        if pitch_tok is not None:
            tokens.append(pitch_tok)

        # Duration token
        dur_tok = tokenizer._duration_to_token(dur_ticks)
        if dur_tok is not None:
            tokens.append(dur_tok)

        # Time shift to next note
        ts_tokens = tokenizer._quantize_time_shift(dur_ticks)
        tokens.extend(ts_tokens)

    tokens.append(tokenizer.SUBJECT_END)
    return tokens


# ======================================================================
# Scale-degree subject functions
# ======================================================================

def parse_subject_string_sd(
    subject_str: str,
    tokenizer: "ScaleDegreeTokenizer",
    key_root: int,
    key_mode: str,
) -> list[int]:
    """Parse a user-provided subject string into scale-degree tokens.

    Expected format is the same as ``parse_subject_string`` (e.g. "C4 D4 Eb4 F4").
    MIDI pitches are converted to tonic-relative (OCT, [SHARP|FLAT], DEG) tokens.
    """
    from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer

    duration_map = {
        "w": 1920, "h": 960, "q": 480, "e": 240, "s": 120,
        "dq": 720, "dh": 1440, "de": 360,
    }

    tokens: list[int] = [tokenizer.SUBJECT_START, tokenizer.VOICE_1]

    # Temporarily set encode key context for _pitch_to_degree_tokens
    tokenizer._encode_key_root = key_root
    tokenizer._encode_key_mode = key_mode

    parts = subject_str.strip().split()
    for part in parts:
        if ":" in part:
            note_str, dur_str = part.split(":", 1)
            dur_ticks = duration_map.get(dur_str, TICKS_PER_QUARTER)
        else:
            note_str = part
            dur_ticks = TICKS_PER_QUARTER

        midi_note = parse_note_string(note_str)
        if midi_note is None:
            logger.warning(f"Could not parse note: {note_str}")
            continue

        degree_toks = tokenizer._pitch_to_degree_tokens(midi_note)
        tokens.extend(degree_toks)

        dur_tok = tokenizer._duration_to_token(dur_ticks)
        if dur_tok is not None:
            tokens.append(dur_tok)

        ts_tokens = tokenizer._quantize_time_shift(dur_ticks)
        tokens.extend(ts_tokens)

    tokens.append(tokenizer.SUBJECT_END)
    return tokens


def generate_subject_sd(
    key_root: int,
    key_mode: str,
    tokenizer: "ScaleDegreeTokenizer",
    length: int = 8,
) -> list[int]:
    """Generate a random subject directly in scale-degree space.

    Produces stepwise motion between DEG_1-7, starting on the tonic or
    dominant, in a reasonable octave.
    """
    from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer

    start_octave = random.choice([4, 5])
    start_degree = random.choice([1, 5])  # tonic or dominant

    degrees: list[tuple[int, int]] = [(start_octave, start_degree)]

    for _ in range(length - 1):
        prev_oct, prev_deg = degrees[-1]
        r = random.random()
        if r < 0.4:
            step = 1
        elif r < 0.7:
            step = -1
        elif r < 0.85:
            step = 2
        elif r < 0.95:
            step = -2
        else:
            step = random.choice([3, -3])

        new_deg = prev_deg + step
        new_oct = prev_oct
        while new_deg > 7:
            new_deg -= 7
            new_oct += 1
        while new_deg < 1:
            new_deg += 7
            new_oct -= 1

        new_oct = max(tokenizer.config.min_octave, min(tokenizer.config.max_octave, new_oct))
        degrees.append((new_oct, new_deg))

    rhythm_patterns = [
        [480] * length,
        [240] * length,
        [480, 240, 240] * (length // 3 + 1),
        [240, 240, 480] * (length // 3 + 1),
        [960, 480, 480, 240, 240, 480, 480, 960],
    ]
    rhythm = random.choice(rhythm_patterns)[:length]

    tokens: list[int] = [tokenizer.SUBJECT_START, tokenizer.VOICE_1]
    for (octave, degree), dur in zip(degrees, rhythm):
        oct_tok = tokenizer.name_to_token.get(f"OCT_{octave}")
        if oct_tok is not None:
            tokens.append(oct_tok)
        deg_tok = tokenizer.name_to_token.get(f"DEG_{degree}")
        if deg_tok is not None:
            tokens.append(deg_tok)

        dur_tok = tokenizer._duration_to_token(dur)
        if dur_tok is not None:
            tokens.append(dur_tok)

        ts_tokens = tokenizer._quantize_time_shift(dur)
        tokens.extend(ts_tokens)

    tokens.append(tokenizer.SUBJECT_END)
    return tokens


def generate_subject(
    key_root: int,
    key_mode: str,
    tokenizer: BachTokenizer,
    length: int = 8,
) -> list[int]:
    """Generate a random subject in the given key.

    Creates a melodically interesting subject using scale-based movement
    with some characteristic Bach-like patterns.

    Returns:
        Token sequence for the subject.
    """
    scale = get_scale(key_root, key_mode)

    # Start on tonic or dominant, in a reasonable octave
    start_octave = random.choice([4, 5])
    start_pc = random.choice([scale[0], scale[4]])  # tonic or dominant
    start_pitch = start_pc + (start_octave * 12) + 12  # MIDI: octave * 12 + 12

    # Clamp to range
    start_pitch = max(MIN_PITCH, min(MAX_PITCH, start_pitch))

    # Generate pitch sequence
    pitches = [start_pitch]
    for _ in range(length - 1):
        # Choose interval: mostly steps (1-2 scale degrees), occasional leap
        r = random.random()
        if r < 0.4:
            step = 1   # ascending step
        elif r < 0.7:
            step = -1  # descending step
        elif r < 0.85:
            step = 2   # ascending skip
        elif r < 0.95:
            step = -2  # descending skip
        else:
            step = random.choice([3, -3])  # leap

        # Find current scale degree
        curr_pc = pitches[-1] % 12
        curr_octave = pitches[-1] // 12

        # Find nearest scale degree
        scale_degrees = []
        for oct in range(curr_octave - 1, curr_octave + 2):
            for s in scale:
                midi = s + oct * 12
                if MIN_PITCH <= midi <= MAX_PITCH:
                    scale_degrees.append(midi)
        scale_degrees.sort()

        # Find current position in scale
        closest_idx = min(range(len(scale_degrees)),
                         key=lambda i: abs(scale_degrees[i] - pitches[-1]))

        new_idx = closest_idx + step
        new_idx = max(0, min(len(scale_degrees) - 1, new_idx))
        new_pitch = scale_degrees[new_idx]

        pitches.append(new_pitch)

    # Generate rhythms: characteristic Bach patterns
    rhythm_patterns = [
        [480] * length,  # all quarters
        [240] * length,  # all eighths
        [480, 240, 240] * (length // 3 + 1),  # quarter-eighth-eighth
        [240, 240, 480] * (length // 3 + 1),  # eighth-eighth-quarter
        [960, 480, 480, 240, 240, 480, 480, 960],  # varied
    ]
    rhythm = random.choice(rhythm_patterns)[:length]

    # Build tokens
    tokens = [tokenizer.SUBJECT_START, tokenizer.VOICE_1]
    current_time = 0

    for pitch, dur in zip(pitches, rhythm):
        pitch_tok = tokenizer._pitch_to_token(pitch)
        if pitch_tok is not None:
            tokens.append(pitch_tok)

        dur_tok = tokenizer._duration_to_token(dur)
        if dur_tok is not None:
            tokens.append(dur_tok)

        ts_tokens = tokenizer._quantize_time_shift(dur)
        tokens.extend(ts_tokens)

    tokens.append(tokenizer.SUBJECT_END)
    return tokens
