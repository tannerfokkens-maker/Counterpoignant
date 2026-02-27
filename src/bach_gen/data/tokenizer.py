"""Custom tokenizer for multi-voice Bach counterpoint.

Uses a flat vocabulary with voice markers, key tokens, mode tokens, and musical events.
Supports 2-4 voice compositions (2-part inventions, sinfonias, chorales, fugues).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.utils.constants import (
    MIN_PITCH,
    MAX_PITCH,
    DURATION_BINS,
    TIME_SHIFT_BINS,
    TICKS_PER_QUARTER,
    KEY_NAMES,
    STYLE_NAMES,
    FORM_NAMES,
    LENGTH_NAMES,
    LENGTH_BOUNDARIES,
    METER_NAMES,
    METER_MAP,
    TEXTURE_NAMES,
    IMITATION_NAMES,
    HARMONIC_RHYTHM_NAMES,
    HARMONIC_TENSION_NAMES,
    CHROMATICISM_NAMES,
    ENCODING_MODE_NAMES,
    ticks_per_measure,
    beat_tick_positions,
    length_bucket,
)
from bach_gen.utils.music_theory import get_key_signature_name

logger = logging.getLogger(__name__)


@dataclass
class TokenConfig:
    """Configuration for the tokenizer vocabulary."""

    min_pitch: int = MIN_PITCH
    max_pitch: int = MAX_PITCH
    duration_bins: list[int] = field(default_factory=lambda: list(DURATION_BINS))
    time_shift_bins: list[int] = field(default_factory=lambda: list(TIME_SHIFT_BINS))


class BachTokenizer:
    """Tokenizer for multi-voice counterpoint sequences.

    Vocabulary layout:
        0: PAD
        1: BOS (beginning of sequence)
        2: EOS (end of sequence)
        3: VOICE_1 (soprano / upper voice marker)
        4: VOICE_2 (alto / lower voice marker)
        5: VOICE_3 (tenor voice marker)
        6: VOICE_4 (bass voice marker)
        7: SUBJECT_START
        8: SUBJECT_END
        9: BAR (bar line marker)
        10-15: BEAT_1..BEAT_6 (beat position markers)
        16: MODE_2PART (voice-count conditioning token)
        17: MODE_3PART
        18: MODE_4PART
        19: MODE_FUGUE
        20: STYLE_BACH
        21: STYLE_BAROQUE
        22: STYLE_RENAISSANCE
        23: STYLE_CLASSICAL
        24-30: FORM_CHORALE..FORM_MOTET (form conditioning tokens)
        31-34: LENGTH_SHORT..LENGTH_EXTENDED (length conditioning tokens)
        35-40: METER_2_4..METER_ALLA_BREVE (meter conditioning tokens)
        41-43: TEXTURE_HOMOPHONIC..TEXTURE_MIXED
        44-46: IMITATION_NONE..IMITATION_HIGH
        47-49: HARMONIC_RHYTHM_SLOW..HARMONIC_RHYTHM_FAST
        50-52: HARMONIC_TENSION_LOW..HARMONIC_TENSION_HIGH
        53-55: CHROMATICISM_LOW..CHROMATICISM_HIGH
        56-57: ENCODE_INTERLEAVED, ENCODE_SEQUENTIAL
        58: VOICE_SEP
        59..82: KEY tokens (C_major, C_minor, ...)
        next: PITCH tokens (Pitch_36 .. Pitch_84)
        next: DURATION tokens (Dur_60, Dur_120, ...)
        next: TIME_SHIFT tokens (TimeShift_60, TimeShift_120, ...)
    """

    # Special token IDs
    PAD = 0
    BOS = 1
    EOS = 2
    VOICE_1 = 3
    VOICE_2 = 4
    VOICE_3 = 5
    VOICE_4 = 6
    SUBJECT_START = 7
    SUBJECT_END = 8
    BAR = 9
    BEAT_1 = 10
    BEAT_2 = 11
    BEAT_3 = 12
    BEAT_4 = 13
    BEAT_5 = 14
    BEAT_6 = 15
    MODE_2PART = 16
    MODE_3PART = 17
    MODE_4PART = 18
    MODE_FUGUE = 19
    STYLE_BACH = 20
    STYLE_BAROQUE = 21
    STYLE_RENAISSANCE = 22
    STYLE_CLASSICAL = 23
    FORM_CHORALE = 24
    FORM_INVENTION = 25
    FORM_FUGUE = 26
    FORM_SINFONIA = 27
    FORM_QUARTET = 28
    FORM_TRIO_SONATA = 29
    FORM_MOTET = 30
    LENGTH_SHORT = 31
    LENGTH_MEDIUM = 32
    LENGTH_LONG = 33
    LENGTH_EXTENDED = 34
    METER_2_4 = 35
    METER_3_4 = 36
    METER_4_4 = 37
    METER_6_8 = 38
    METER_3_8 = 39
    METER_ALLA_BREVE = 40

    # Phase 2 conditioning tokens
    TEXTURE_HOMOPHONIC = 41
    TEXTURE_POLYPHONIC = 42
    TEXTURE_MIXED = 43
    IMITATION_NONE = 44
    IMITATION_LOW = 45
    IMITATION_HIGH = 46
    HARMONIC_RHYTHM_SLOW = 47
    HARMONIC_RHYTHM_MODERATE = 48
    HARMONIC_RHYTHM_FAST = 49
    HARMONIC_TENSION_LOW = 50
    HARMONIC_TENSION_MODERATE = 51
    HARMONIC_TENSION_HIGH = 52
    CHROMATICISM_LOW = 53
    CHROMATICISM_MODERATE = 54
    CHROMATICISM_HIGH = 55

    # Phase 3 encoding mode tokens
    ENCODE_INTERLEAVED = 56
    ENCODE_SEQUENTIAL = 57
    VOICE_SEP = 58

    # Voice-count tokens (how many voices)
    FORM_TO_MODE_TOKEN: dict[str, int] = {
        "2-part": 16, "invention": 16,
        "sinfonia": 17, "trio_sonata": 17,
        "chorale": 18, "quartet": 18, "motet": 18,
        "fugue": 19,
    }

    # Form tokens (what kind of piece)
    FORM_TO_FORM_TOKEN: dict[str, int] = {
        "chorale": 24, "invention": 25, "fugue": 26,
        "sinfonia": 27, "quartet": 28, "trio_sonata": 29, "motet": 30,
    }

    # Map voice count to list of voice marker token IDs
    VOICE_TOKENS = [VOICE_1, VOICE_2, VOICE_3, VOICE_4]

    # Map style name to token ID
    STYLE_TO_TOKEN: dict[str, int] = {
        "bach": 20,
        "baroque": 21,
        "renaissance": 22,
        "classical": 23,
    }

    # Map length bucket name to token ID
    LENGTH_TO_TOKEN: dict[str, int] = {
        "short": 31,
        "medium": 32,
        "long": 33,
        "extended": 34,
    }

    # Map meter name to token ID
    METER_TO_TOKEN: dict[str, int] = {
        "2_4": 35,
        "3_4": 36,
        "4_4": 37,
        "6_8": 38,
        "3_8": 39,
        "alla_breve": 40,
    }

    # Phase 2 mapping dicts
    TEXTURE_TO_TOKEN: dict[str, int] = {
        "homophonic": 41,
        "polyphonic": 42,
        "mixed": 43,
    }

    IMITATION_TO_TOKEN: dict[str, int] = {
        "none": 44,
        "low": 45,
        "high": 46,
    }

    HARMONIC_RHYTHM_TO_TOKEN: dict[str, int] = {
        "slow": 47,
        "moderate": 48,
        "fast": 49,
    }

    HARMONIC_TENSION_TO_TOKEN: dict[str, int] = {
        "low": 50,
        "moderate": 51,
        "high": 52,
    }

    CHROMATICISM_TO_TOKEN: dict[str, int] = {
        "low": 53,
        "moderate": 54,
        "high": 55,
    }

    ENCODING_MODE_TO_TOKEN: dict[str, int] = {
        "interleaved": 56,
        "sequential": 57,
    }

    def __init__(self, config: TokenConfig | None = None):
        self.config = config or TokenConfig()
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build the token vocabulary."""
        self.token_to_name: dict[int, str] = {}
        self.name_to_token: dict[str, int] = {}

        idx = 0

        # Special tokens
        for name in ["PAD", "BOS", "EOS", "VOICE_1", "VOICE_2",
                      "VOICE_3", "VOICE_4",
                      "SUBJECT_START", "SUBJECT_END", "BAR",
                      "BEAT_1", "BEAT_2", "BEAT_3", "BEAT_4", "BEAT_5", "BEAT_6",
                      "MODE_2PART", "MODE_3PART", "MODE_4PART", "MODE_FUGUE",
                      "STYLE_BACH", "STYLE_BAROQUE", "STYLE_RENAISSANCE", "STYLE_CLASSICAL",
                      "FORM_CHORALE", "FORM_INVENTION", "FORM_FUGUE",
                      "FORM_SINFONIA", "FORM_QUARTET", "FORM_TRIO_SONATA", "FORM_MOTET"]:
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Length tokens (31-34)
        for length_name in LENGTH_NAMES:
            name = f"LENGTH_{length_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Meter tokens (35-40)
        for meter_name in METER_NAMES:
            name = f"METER_{meter_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Texture tokens (41-43)
        for texture_name in TEXTURE_NAMES:
            name = f"TEXTURE_{texture_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Imitation tokens (44-46)
        for imitation_name in IMITATION_NAMES:
            name = f"IMITATION_{imitation_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Harmonic rhythm tokens (47-49)
        for hr_name in HARMONIC_RHYTHM_NAMES:
            name = f"HARMONIC_RHYTHM_{hr_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Harmonic tension tokens (50-52)
        for ht_name in HARMONIC_TENSION_NAMES:
            name = f"HARMONIC_TENSION_{ht_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Chromaticism tokens (53-55)
        for ch_name in CHROMATICISM_NAMES:
            name = f"CHROMATICISM_{ch_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Encoding mode tokens (56-57)
        for em_name in ENCODING_MODE_NAMES:
            name = f"ENCODE_{em_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Voice separator (58)
        self.token_to_name[idx] = "VOICE_SEP"
        self.name_to_token["VOICE_SEP"] = idx
        idx += 1

        # Key tokens
        self._key_start = idx
        for key_name in KEY_NAMES:
            name = f"KEY_{key_name}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1
        self._key_end = idx

        # Pitch tokens
        self._pitch_start = idx
        for pitch in range(self.config.min_pitch, self.config.max_pitch + 1):
            name = f"Pitch_{pitch}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1
        self._pitch_end = idx

        # Duration tokens
        self._dur_start = idx
        for dur in self.config.duration_bins:
            name = f"Dur_{dur}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1
        self._dur_end = idx

        # Time shift tokens
        self._ts_start = idx
        for ts in self.config.time_shift_bins:
            name = f"TimeShift_{ts}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1
        self._ts_end = idx

        self._vocab_size = idx

        # Verify hardcoded class-level constants match dynamic vocab
        assert self.name_to_token.get("TEXTURE_HOMOPHONIC") == self.TEXTURE_HOMOPHONIC
        assert self.name_to_token.get("HARMONIC_TENSION_LOW") == self.HARMONIC_TENSION_LOW
        assert self.name_to_token.get("CHROMATICISM_LOW") == self.CHROMATICISM_LOW
        assert self.name_to_token.get("VOICE_SEP") == self.VOICE_SEP
        assert self.name_to_token.get("ENCODE_SEQUENTIAL") == self.ENCODE_SEQUENTIAL

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ------------------------------------------------------------------
    # Conditioning prefix (shared by encode and encode_sequential)
    # ------------------------------------------------------------------

    def _build_conditioning_prefix(
        self,
        item: Union[VoicePair, VoiceComposition],
        form: str | None = None,
        style: str = "",
        length_bars: int | None = None,
        meter: str | None = None,
        texture: str | None = None,
        imitation: str | None = None,
        harmonic_rhythm: str | None = None,
        harmonic_tension: str | None = None,
        chromaticism: str | None = None,
        encoding_mode: str | None = None,
    ) -> tuple[list[int], VoiceComposition]:
        """Build the conditioning prefix up to (but not including) KEY.

        Returns (prefix_tokens, comp) where comp is the resolved
        VoiceComposition (converted from VoicePair if needed).
        """
        tokens = [self.BOS]

        # Style conditioning token (right after BOS)
        style = style or (item.style if hasattr(item, "style") else "")
        if style and style in self.STYLE_TO_TOKEN:
            tokens.append(self.STYLE_TO_TOKEN[style])

        # Form token (what kind of piece)
        if form is not None and form in self.FORM_TO_FORM_TOKEN:
            tokens.append(self.FORM_TO_FORM_TOKEN[form])

        # Voice-count token (how many voices)
        if form is not None and form in self.FORM_TO_MODE_TOKEN:
            tokens.append(self.FORM_TO_MODE_TOKEN[form])

        # Length conditioning token
        if length_bars is not None:
            bucket = length_bucket(length_bars)
            if bucket in self.LENGTH_TO_TOKEN:
                tokens.append(self.LENGTH_TO_TOKEN[bucket])

        # Meter conditioning token
        if meter is not None and meter in self.METER_TO_TOKEN:
            tokens.append(self.METER_TO_TOKEN[meter])

        # Convert VoicePair to VoiceComposition for uniform handling
        if isinstance(item, VoicePair):
            comp = VoiceComposition.from_voice_pair(item)
        else:
            comp = item

        # Auto-detect meter from time signature if not explicitly provided
        if meter is None:
            time_sig = comp.time_signature if hasattr(comp, "time_signature") else (4, 4)
            auto_meter = METER_MAP.get(time_sig)
            if auto_meter and auto_meter in self.METER_TO_TOKEN:
                tokens.append(self.METER_TO_TOKEN[auto_meter])

        # Phase 2 conditioning tokens
        if texture is not None and texture in self.TEXTURE_TO_TOKEN:
            tokens.append(self.TEXTURE_TO_TOKEN[texture])
        if imitation is not None and imitation in self.IMITATION_TO_TOKEN:
            tokens.append(self.IMITATION_TO_TOKEN[imitation])
        if harmonic_rhythm is not None and harmonic_rhythm in self.HARMONIC_RHYTHM_TO_TOKEN:
            tokens.append(self.HARMONIC_RHYTHM_TO_TOKEN[harmonic_rhythm])
        if harmonic_tension is not None and harmonic_tension in self.HARMONIC_TENSION_TO_TOKEN:
            tokens.append(self.HARMONIC_TENSION_TO_TOKEN[harmonic_tension])
        if chromaticism is not None and chromaticism in self.CHROMATICISM_TO_TOKEN:
            tokens.append(self.CHROMATICISM_TO_TOKEN[chromaticism])

        # Encoding mode token
        if encoding_mode is not None and encoding_mode in self.ENCODING_MODE_TO_TOKEN:
            tokens.append(self.ENCODING_MODE_TO_TOKEN[encoding_mode])

        return tokens, comp

    # ------------------------------------------------------------------
    # Encode (interleaved — default)
    # ------------------------------------------------------------------

    def encode(
        self,
        item: Union[VoicePair, VoiceComposition],
        form: str | None = None,
        style: str = "",
        length_bars: int | None = None,
        meter: str | None = None,
        texture: str | None = None,
        imitation: str | None = None,
        harmonic_rhythm: str | None = None,
        harmonic_tension: str | None = None,
        chromaticism: str | None = None,
    ) -> list[int]:
        """Encode a VoicePair or VoiceComposition into a token sequence.

        Prefix order: BOS STYLE FORM MODE LENGTH METER TEXTURE IMITATION
                      HARMONIC_RHYTHM TENSION CHROMATICISM ENCODE_INTERLEAVED KEY <events> EOS
        """
        tokens, comp = self._build_conditioning_prefix(
            item, form=form, style=style, length_bars=length_bars, meter=meter,
            texture=texture, imitation=imitation, harmonic_rhythm=harmonic_rhythm,
            harmonic_tension=harmonic_tension, chromaticism=chromaticism,
            encoding_mode="interleaved",
        )

        # Key token
        key_name = get_key_signature_name(comp.key_root, comp.key_mode)
        key_token_name = f"KEY_{key_name}"
        if key_token_name in self.name_to_token:
            tokens.append(self.name_to_token[key_token_name])

        # Interleave all voices
        time_sig = comp.time_signature if hasattr(comp, "time_signature") else (4, 4)
        events = self._interleave_n_voices(comp.voices, time_sig=time_sig)
        tokens.extend(events)

        tokens.append(self.EOS)
        return tokens

    # ------------------------------------------------------------------
    # Encode sequential (voice-by-voice)
    # ------------------------------------------------------------------

    def encode_sequential(
        self,
        item: Union[VoicePair, VoiceComposition],
        form: str | None = None,
        style: str = "",
        length_bars: int | None = None,
        meter: str | None = None,
        texture: str | None = None,
        imitation: str | None = None,
        harmonic_rhythm: str | None = None,
        harmonic_tension: str | None = None,
        chromaticism: str | None = None,
    ) -> list[int]:
        """Encode using sequential (voice-by-voice) format.

        Format: BOS <conditioning> ENCODE_SEQUENTIAL KEY
                VOICE_1 <voice1_notes> VOICE_SEP
                VOICE_2 <voice2_notes> VOICE_SEP
                ...
                VOICE_N <voiceN_notes> EOS
        """
        tokens, comp = self._build_conditioning_prefix(
            item, form=form, style=style, length_bars=length_bars, meter=meter,
            texture=texture, imitation=imitation, harmonic_rhythm=harmonic_rhythm,
            harmonic_tension=harmonic_tension, chromaticism=chromaticism,
            encoding_mode="sequential",
        )

        # Key token
        key_name = get_key_signature_name(comp.key_root, comp.key_mode)
        key_token_name = f"KEY_{key_name}"
        if key_token_name in self.name_to_token:
            tokens.append(self.name_to_token[key_token_name])

        time_sig = comp.time_signature if hasattr(comp, "time_signature") else (4, 4)

        # Serialize each voice sequentially
        for voice_idx, voice_notes in enumerate(comp.voices):
            voice_num = voice_idx + 1
            if voice_num > 4:
                break

            # Emit VOICE_N marker
            voice_tok = self.VOICE_TOKENS[voice_num - 1]
            tokens.append(voice_tok)

            # Serialize this voice's notes with its own timeline
            voice_tokens = self._serialize_single_voice(voice_notes, time_sig)
            tokens.extend(voice_tokens)

            # VOICE_SEP between voices; EOS after last voice
            if voice_idx < len(comp.voices) - 1:
                tokens.append(self.VOICE_SEP)
            else:
                tokens.append(self.EOS)

        return tokens

    def _serialize_single_voice(
        self,
        voice_notes: list[tuple[int, int, int]],
        time_sig: tuple[int, int] = (4, 4),
    ) -> list[int]:
        """Serialize one voice's notes with its own timeline starting from tick 0.

        Emits BAR/BEAT markers and Pitch/DUR tokens.  No per-note VOICE_N
        tokens (unlike interleaved mode) — the caller emits VOICE_N once before
        the voice block.
        """
        tokens: list[int] = []
        if not voice_notes:
            return tokens

        sorted_notes = sorted(voice_notes, key=lambda n: n[0])

        # Pre-compute beat boundaries
        measure_ticks = ticks_per_measure(time_sig)
        beat_offsets = beat_tick_positions(time_sig)
        max_tick = max(n[0] + n[1] for n in sorted_notes)
        n_measures = (max_tick // measure_ticks) + 2

        beat_boundaries: list[tuple[int, int]] = []
        for m in range(n_measures):
            for beat_idx, offset in enumerate(beat_offsets):
                abs_tick = m * measure_ticks + offset
                beat_boundaries.append((abs_tick, beat_idx + 1))

        current_time = 0
        beat_ptr = 0

        for start, dur, pitch in sorted_notes:
            # Emit BAR/BEAT tokens
            while beat_ptr < len(beat_boundaries) and beat_boundaries[beat_ptr][0] <= start:
                b_tick, b_num = beat_boundaries[beat_ptr]
                if b_tick >= current_time:
                    gap = b_tick - current_time
                    if gap > 0:
                        ts_tokens = self._quantize_time_shift(gap)
                        tokens.extend(ts_tokens)
                        current_time = b_tick

                    if b_num == 1:
                        tokens.append(self.BAR)
                    beat_tok_name = f"BEAT_{b_num}"
                    beat_tok = self.name_to_token.get(beat_tok_name)
                    if beat_tok is not None:
                        tokens.append(beat_tok)

                beat_ptr += 1

            # Time shift to event
            dt = start - current_time
            if dt > 0:
                ts_tokens = self._quantize_time_shift(dt)
                tokens.extend(ts_tokens)
                current_time = start

            # Emit Pitch DUR
            pitch_tok = self._pitch_to_token(pitch)
            if pitch_tok is not None:
                tokens.append(pitch_tok)

            dur_tok = self._duration_to_token(dur)
            if dur_tok is not None:
                tokens.append(dur_tok)

        return tokens

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, tokens: list[int]) -> VoiceComposition:
        """Decode a token sequence back into a VoiceComposition.

        Returns a VoiceComposition with the voices found in the token stream.
        """
        # Track notes per voice (up to 4 voices)
        voice_notes: dict[int, list[tuple[int, int, int]]] = {
            1: [], 2: [], 3: [], 4: [],
        }
        current_time = 0
        current_voice = 1
        key_root = 0
        key_mode = "major"
        pending_pitch: int | None = None

        for tok in tokens:
            name = self.token_to_name.get(tok, "")

            if name in ("PAD", "BOS", "EOS", "SUBJECT_START", "SUBJECT_END",
                        "BAR", "BEAT_1", "BEAT_2", "BEAT_3", "BEAT_4", "BEAT_5", "BEAT_6",
                        "MODE_2PART", "MODE_3PART", "MODE_4PART", "MODE_FUGUE",
                        "STYLE_BACH", "STYLE_BAROQUE", "STYLE_RENAISSANCE", "STYLE_CLASSICAL",
                        "FORM_CHORALE", "FORM_INVENTION", "FORM_FUGUE",
                        "FORM_SINFONIA", "FORM_QUARTET", "FORM_TRIO_SONATA", "FORM_MOTET",
                        "LENGTH_SHORT", "LENGTH_MEDIUM", "LENGTH_LONG", "LENGTH_EXTENDED",
                        "METER_2_4", "METER_3_4", "METER_4_4", "METER_6_8",
                        "METER_3_8", "METER_ALLA_BREVE",
                        "TEXTURE_HOMOPHONIC", "TEXTURE_POLYPHONIC", "TEXTURE_MIXED",
                        "IMITATION_NONE", "IMITATION_LOW", "IMITATION_HIGH",
                        "HARMONIC_RHYTHM_SLOW", "HARMONIC_RHYTHM_MODERATE", "HARMONIC_RHYTHM_FAST",
                        "HARMONIC_TENSION_LOW", "HARMONIC_TENSION_MODERATE", "HARMONIC_TENSION_HIGH",
                        "ENCODE_INTERLEAVED", "ENCODE_SEQUENTIAL"):
                continue
            elif name == "VOICE_SEP":
                # Voice separator: reset timeline for next voice (sequential mode)
                current_time = 0
                pending_pitch = None
            elif name == "VOICE_1":
                pending_pitch = None
                current_voice = 1
            elif name == "VOICE_2":
                pending_pitch = None
                current_voice = 2
            elif name == "VOICE_3":
                pending_pitch = None
                current_voice = 3
            elif name == "VOICE_4":
                pending_pitch = None
                current_voice = 4
            elif name.startswith("KEY_"):
                key_str = name[4:]
                parts = key_str.rsplit("_", 1)
                if len(parts) == 2:
                    key_mode = parts[1]
                    from bach_gen.utils.music_theory import note_name_to_pc
                    try:
                        root_name = parts[0].replace("s", "#")
                        key_root = note_name_to_pc(root_name)
                    except ValueError:
                        pass
            elif name.startswith("Pitch_"):
                pending_pitch = int(name[6:])
            elif name.startswith("Dur_"):
                dur = int(name[4:])
                if pending_pitch is not None:
                    note_tuple = (current_time, dur, pending_pitch)
                    voice_notes[current_voice].append(note_tuple)
                    pending_pitch = None
            elif name.startswith("TimeShift_"):
                ts = int(name[10:])
                current_time += ts
                pending_pitch = None

        # Build voices list — only include voices that have notes
        voices = []
        for v in [1, 2, 3, 4]:
            if voice_notes[v]:
                voices.append(voice_notes[v])

        # Ensure at least 2 voices (pad with empty if needed)
        while len(voices) < 2:
            voices.append([])

        return VoiceComposition(
            voices=voices,
            key_root=key_root,
            key_mode=key_mode,
            source="decoded",
        )

    def decode_to_pair(self, tokens: list[int]) -> VoicePair:
        """Decode a token sequence back into a VoicePair (backward compat).

        Uses voice 1 as upper and voice 2 as lower.
        """
        comp = self.decode(tokens)
        return comp.to_voice_pair()

    # ------------------------------------------------------------------
    # Interleaved encoding helpers
    # ------------------------------------------------------------------

    def _interleave_n_voices(
        self,
        voices: list[list[tuple[int, int, int]]],
        time_sig: tuple[int, int] = (4, 4),
    ) -> list[int]:
        """Interleave N voices into a single token sequence with BAR/BEAT markers."""
        tokens: list[int] = []

        # Merge events into timeline
        # Each event: (time, voice_num, pitch, duration)
        all_events: list[tuple[int, int, int, int]] = []
        for voice_idx, voice_notes in enumerate(voices):
            voice_num = voice_idx + 1  # 1-indexed
            for start, dur, pitch in voice_notes:
                all_events.append((start, voice_num, pitch, dur))

        # Sort by time, then voice number
        all_events.sort(key=lambda e: (e[0], e[1]))

        if not all_events:
            return tokens

        # Pre-compute beat boundaries for the entire piece span
        measure_ticks = ticks_per_measure(time_sig)
        beat_offsets = beat_tick_positions(time_sig)
        max_tick = max((e[0] + e[3] for e in all_events), default=0)
        n_measures = (max_tick // measure_ticks) + 2

        beat_boundaries: list[tuple[int, int]] = []
        for m in range(n_measures):
            for beat_idx, offset in enumerate(beat_offsets):
                abs_tick = m * measure_ticks + offset
                beat_boundaries.append((abs_tick, beat_idx + 1))

        current_time = 0
        beat_ptr = 0

        for event_time, voice_num, pitch, duration in all_events:
            # Emit BAR/BEAT tokens for any beat boundaries between current_time and event_time
            while beat_ptr < len(beat_boundaries) and beat_boundaries[beat_ptr][0] <= event_time:
                b_tick, b_num = beat_boundaries[beat_ptr]
                if b_tick >= current_time:
                    gap = b_tick - current_time
                    if gap > 0:
                        ts_tokens = self._quantize_time_shift(gap)
                        tokens.extend(ts_tokens)
                        current_time = b_tick

                    if b_num == 1:
                        tokens.append(self.BAR)
                    beat_tok_name = f"BEAT_{b_num}"
                    beat_tok = self.name_to_token.get(beat_tok_name)
                    if beat_tok is not None:
                        tokens.append(beat_tok)

                beat_ptr += 1

            # Emit time shift for remaining distance to the event
            dt = event_time - current_time
            if dt > 0:
                ts_tokens = self._quantize_time_shift(dt)
                tokens.extend(ts_tokens)
                current_time = event_time

            # Voice marker
            voice_tok = self.VOICE_TOKENS[voice_num - 1] if voice_num <= 4 else self.VOICE_1
            tokens.append(voice_tok)

            # Pitch
            pitch_tok = self._pitch_to_token(pitch)
            if pitch_tok is not None:
                tokens.append(pitch_tok)

            # Duration
            dur_tok = self._duration_to_token(duration)
            if dur_tok is not None:
                tokens.append(dur_tok)

        return tokens

    def _pitch_to_token(self, pitch: int) -> int | None:
        """Convert MIDI pitch to token ID."""
        pitch = max(self.config.min_pitch, min(self.config.max_pitch, pitch))
        name = f"Pitch_{pitch}"
        return self.name_to_token.get(name)

    def _duration_to_token(self, duration_ticks: int) -> int | None:
        """Quantize duration to nearest bin and return token."""
        bin_val = self._nearest_bin(duration_ticks, self.config.duration_bins)
        name = f"Dur_{bin_val}"
        return self.name_to_token.get(name)

    def _quantize_time_shift(self, ticks: int) -> list[int]:
        """Quantize a time shift into one or more time shift tokens."""
        tokens = []
        remaining = ticks
        bins = sorted(self.config.time_shift_bins, reverse=True)

        while remaining > 0:
            # Find largest bin that fits
            best = bins[-1]  # smallest bin as fallback
            for b in bins:
                if b <= remaining:
                    best = b
                    break

            name = f"TimeShift_{best}"
            tok = self.name_to_token.get(name)
            if tok is not None:
                tokens.append(tok)
            remaining -= best

            if remaining < bins[-1] // 2:
                break

        return tokens

    @staticmethod
    def _nearest_bin(value: int, bins: list[int]) -> int:
        """Find the nearest bin value."""
        return min(bins, key=lambda b: abs(b - value))

    def save(self, path: str | Path) -> None:
        """Save tokenizer config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tokenizer_type": "absolute",
            "vocab_size": self._vocab_size,
            "token_to_name": {str(k): v for k, v in self.token_to_name.items()},
            "config": {
                "min_pitch": self.config.min_pitch,
                "max_pitch": self.config.max_pitch,
                "duration_bins": self.config.duration_bins,
                "time_shift_bins": self.config.time_shift_bins,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved tokenizer to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BachTokenizer":
        """Load tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)
        config = TokenConfig(**data["config"])
        return cls(config)


def load_tokenizer(path: str | Path) -> "BachTokenizer | ScaleDegreeTokenizer":
    """Factory: load the correct tokenizer class from a saved JSON file.

    Reads the ``tokenizer_type`` field and returns either a ``BachTokenizer``
    (for ``"absolute"`` or missing field) or a ``ScaleDegreeTokenizer`` (for
    ``"scale_degree"``).
    """
    from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    tok_type = data.get("tokenizer_type", "absolute")
    if tok_type == "scale_degree":
        return ScaleDegreeTokenizer.load(path)
    return BachTokenizer.load(path)
