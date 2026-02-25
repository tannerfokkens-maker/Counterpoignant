"""Data augmentation: transposition and cropping."""

from __future__ import annotations

import random
import logging
from typing import Union

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.utils.constants import MIN_PITCH, MAX_PITCH, TICKS_PER_QUARTER

logger = logging.getLogger(__name__)

# Type alias for either VoicePair or VoiceComposition
Composition = Union[VoicePair, VoiceComposition]


def transpose_composition(comp: VoiceComposition, semitones: int) -> VoiceComposition | None:
    """Transpose a VoiceComposition by the given number of semitones.

    Returns None if the transposition puts notes out of range.
    """
    new_voices = []
    for voice in comp.voices:
        new_voices.append([(s, d, p + semitones) for s, d, p in voice])

    all_pitches = [p for voice in new_voices for _, _, p in voice]
    if not all_pitches:
        return None
    if min(all_pitches) < MIN_PITCH or max(all_pitches) > MAX_PITCH:
        return None

    new_root = (comp.key_root + semitones) % 12

    return VoiceComposition(
        voices=new_voices,
        key_root=new_root,
        key_mode=comp.key_mode,
        source=f"{comp.source} +{semitones}st",
        style=comp.style,
        time_signature=comp.time_signature,
    )


def transpose_voice_pair(pair: VoicePair, semitones: int) -> VoicePair | None:
    """Transpose a voice pair by the given number of semitones.

    Returns None if the transposition puts notes out of range.
    """
    upper = [(s, d, p + semitones) for s, d, p in pair.upper]
    lower = [(s, d, p + semitones) for s, d, p in pair.lower]

    # Check range
    all_pitches = [p for _, _, p in upper] + [p for _, _, p in lower]
    if not all_pitches:
        return None
    if min(all_pitches) < MIN_PITCH or max(all_pitches) > MAX_PITCH:
        return None

    new_root = (pair.key_root + semitones) % 12

    return VoicePair(
        upper=upper,
        lower=lower,
        key_root=new_root,
        key_mode=pair.key_mode,
        source=f"{pair.source} +{semitones}st",
        style=pair.style,
        time_signature=pair.time_signature,
    )


def augment_to_all_keys(items: list[Composition]) -> list[Composition]:
    """Transpose each voice pair / composition to all 12 keys.

    Accepts a list of VoicePair or VoiceComposition (not mixed).

    Returns:
        Augmented list including originals.
    """
    augmented: list[Composition] = []
    for item in items:
        for semitones in range(12):
            if isinstance(item, VoiceComposition):
                transposed = transpose_composition(item, semitones)
            else:
                transposed = transpose_voice_pair(item, semitones)
            if transposed is not None:
                augmented.append(transposed)

    label = "compositions" if items and isinstance(items[0], VoiceComposition) else "pairs"
    logger.info(f"Augmented {len(items)} {label} to {len(augmented)} (all keys)")
    return augmented


def random_crop(
    item: Composition,
    min_length_quarters: int = 16,
    max_length_quarters: int = 64,
) -> Composition | None:
    """Random crop of a voice pair / composition to a shorter segment.

    Returns None if the item is too short.
    """
    if isinstance(item, VoiceComposition):
        return _random_crop_composition(item, min_length_quarters, max_length_quarters)
    return _random_crop_pair(item, min_length_quarters, max_length_quarters)


def _random_crop_composition(
    comp: VoiceComposition,
    min_length_quarters: int,
    max_length_quarters: int,
) -> VoiceComposition | None:
    """Random crop of a VoiceComposition."""
    all_notes = [n for voice in comp.voices for n in voice]
    if not all_notes:
        return None

    min_start = min(n[0] for n in all_notes)
    max_end = max(n[0] + n[1] for n in all_notes)

    total_ticks = max_end - min_start
    min_ticks = min_length_quarters * TICKS_PER_QUARTER
    max_ticks = max_length_quarters * TICKS_PER_QUARTER

    if total_ticks < min_ticks:
        return None

    crop_length = random.randint(min_ticks, min(max_ticks, total_ticks))
    max_crop_start = max_end - crop_length
    crop_start = random.randint(min_start, max(min_start, max_crop_start))
    crop_end = crop_start + crop_length

    new_voices = []
    for voice in comp.voices:
        cropped = _crop_notes(voice, crop_start, crop_end)
        if not cropped:
            return None
        new_voices.append([(s - crop_start, d, p) for s, d, p in cropped])

    return VoiceComposition(
        voices=new_voices,
        key_root=comp.key_root,
        key_mode=comp.key_mode,
        source=f"{comp.source} (crop)",
        style=comp.style,
        time_signature=comp.time_signature,
    )


def _random_crop_pair(
    pair: VoicePair,
    min_length_quarters: int,
    max_length_quarters: int,
) -> VoicePair | None:
    """Random crop of a VoicePair."""
    if not pair.upper or not pair.lower:
        return None

    min_start = min(
        min(n[0] for n in pair.upper),
        min(n[0] for n in pair.lower),
    )
    max_end = max(
        max(n[0] + n[1] for n in pair.upper),
        max(n[0] + n[1] for n in pair.lower),
    )

    total_ticks = max_end - min_start
    min_ticks = min_length_quarters * TICKS_PER_QUARTER
    max_ticks = max_length_quarters * TICKS_PER_QUARTER

    if total_ticks < min_ticks:
        return None

    crop_length = random.randint(min_ticks, min(max_ticks, total_ticks))
    max_start = max_end - crop_length
    crop_start = random.randint(min_start, max(min_start, max_start))
    crop_end = crop_start + crop_length

    upper = _crop_notes(pair.upper, crop_start, crop_end)
    lower = _crop_notes(pair.lower, crop_start, crop_end)

    if not upper or not lower:
        return None

    # Shift to start at 0
    offset = crop_start
    upper = [(s - offset, d, p) for s, d, p in upper]
    lower = [(s - offset, d, p) for s, d, p in lower]

    return VoicePair(
        upper=upper,
        lower=lower,
        key_root=pair.key_root,
        key_mode=pair.key_mode,
        source=f"{pair.source} (crop)",
        style=pair.style,
        time_signature=pair.time_signature,
    )


def _crop_notes(
    notes: list[tuple[int, int, int]],
    start: int,
    end: int,
) -> list[tuple[int, int, int]]:
    """Crop notes to a time window, truncating at boundaries."""
    result = []
    for s, d, p in notes:
        note_end = s + d
        if note_end <= start or s >= end:
            continue
        new_start = max(s, start)
        new_end = min(note_end, end)
        new_dur = new_end - new_start
        if new_dur > 0:
            result.append((new_start, new_dur, p))
    return result
