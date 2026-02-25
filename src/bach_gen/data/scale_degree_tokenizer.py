"""Scale-degree tokenizer for key-agnostic pitch representation.

Replaces absolute MIDI pitch tokens (Pitch_36..Pitch_84) with tonic-relative
octave + degree + optional accidental tokens.  A I-IV-V-I in C major and the
same progression in F# major produce **identical** token sequences, making key
augmentation unnecessary.

Token ordering per note: VOICE_N -> OCT_x -> [SHARP|FLAT] -> DEG_y -> DUR_z

Vocabulary layout (102 tokens):
     0-9:   10 special tokens (PAD, BOS, EOS, VOICE_1-4, SUBJECT_*, BAR)
    10-15:   6 beat tokens (BEAT_1..BEAT_6)
    16-19:   4 voice-count tokens (MODE_2PART..MODE_FUGUE)
    20-23:   4 style tokens (STYLE_BACH..STYLE_CLASSICAL)
    24-30:   7 form tokens (FORM_CHORALE..FORM_MOTET)
    31-34:   4 length tokens (LENGTH_SHORT..LENGTH_EXTENDED)
    35-40:   6 meter tokens (METER_2_4..METER_ALLA_BREVE)
    41-64:  24 key tokens (needed at decode time)
    65-70:   6 octave tokens (OCT_2 .. OCT_7)
    71-77:   7 degree tokens (DEG_1 .. DEG_7)
    78-79:   2 accidental tokens (SHARP, FLAT)
    80-90:  11 duration tokens
    91-101: 11 time shift tokens
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.utils.constants import (
    MIN_PITCH, MAX_PITCH, DURATION_BINS, TIME_SHIFT_BINS,
    KEY_NAMES, SD_MIN_OCTAVE, SD_MAX_OCTAVE, STYLE_NAMES, FORM_NAMES,
    LENGTH_NAMES, LENGTH_BOUNDARIES, METER_NAMES, METER_MAP,
    TICKS_PER_QUARTER, ticks_per_measure, beat_tick_positions,
    length_bucket,
)
from bach_gen.utils.music_theory import (
    get_key_signature_name, midi_to_scale_degree, scale_degree_to_midi,
    note_name_to_pc,
)

logger = logging.getLogger(__name__)


@dataclass
class ScaleDegreeTokenizerConfig:
    """Configuration for the scale-degree tokenizer."""
    min_pitch: int = MIN_PITCH
    max_pitch: int = MAX_PITCH
    min_octave: int = SD_MIN_OCTAVE
    max_octave: int = SD_MAX_OCTAVE
    duration_bins: list[int] = field(default_factory=lambda: list(DURATION_BINS))
    time_shift_bins: list[int] = field(default_factory=lambda: list(TIME_SHIFT_BINS))


class ScaleDegreeTokenizer:
    """Key-agnostic tokenizer using tonic-relative scale degrees.

    Vocabulary layout matches the docstring above.  All special-token class
    attributes mirror BachTokenizer so downstream code works polymorphically.
    """

    # Special token IDs — identical to BachTokenizer
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

    VOICE_TOKENS = [3, 4, 5, 6]  # VOICE_1 .. VOICE_4

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

    def __init__(self, config: ScaleDegreeTokenizerConfig | None = None):
        self.config = config or ScaleDegreeTokenizerConfig()
        self._build_vocab()

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def _build_vocab(self) -> None:
        self.token_to_name: dict[int, str] = {}
        self.name_to_token: dict[str, int] = {}
        idx = 0

        # Special tokens (0-15: PAD..BAR + BEAT_1..BEAT_6)
        for name in [
            "PAD", "BOS", "EOS",
            "VOICE_1", "VOICE_2", "VOICE_3", "VOICE_4",
            "SUBJECT_START", "SUBJECT_END", "BAR",
            "BEAT_1", "BEAT_2", "BEAT_3", "BEAT_4", "BEAT_5", "BEAT_6",
            "MODE_2PART", "MODE_3PART", "MODE_4PART", "MODE_FUGUE",
        ]:
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Style tokens (20-23)
        for style_name in STYLE_NAMES:
            name = f"STYLE_{style_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Form tokens (24-29)
        for form_name in FORM_NAMES:
            name = f"FORM_{form_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Length tokens (30-33)
        for length_name in LENGTH_NAMES:
            name = f"LENGTH_{length_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Meter tokens (34-39)
        for meter_name in METER_NAMES:
            name = f"METER_{meter_name.upper()}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1

        # Key tokens — needed at decode time
        self._key_start = idx
        for key_name in KEY_NAMES:
            name = f"KEY_{key_name}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1
        self._key_end = idx

        # Octave tokens (OCT_2 .. OCT_7)
        self._oct_start = idx
        for o in range(self.config.min_octave, self.config.max_octave + 1):
            name = f"OCT_{o}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1
        self._oct_end = idx

        # Degree tokens (DEG_1 .. DEG_7)
        self._deg_start = idx
        for d in range(1, 8):
            name = f"DEG_{d}"
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1
        self._deg_end = idx

        # Accidental tokens
        self._acc_start = idx
        for name in ["SHARP", "FLAT"]:
            self.token_to_name[idx] = name
            self.name_to_token[name] = idx
            idx += 1
        self._acc_end = idx

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

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self, item: Union[VoicePair, VoiceComposition], form: str | None = None,
        style: str = "", length_bars: int | None = None, meter: str | None = None,
    ) -> list[int]:
        """Encode a VoicePair or VoiceComposition into a scale-degree token sequence.

        Prefix order: BOS STYLE FORM MODE LENGTH METER KEY <events> EOS
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

        # Emit key token (needed for decode)
        key_name = get_key_signature_name(comp.key_root, comp.key_mode)
        key_token_name = f"KEY_{key_name}"
        if key_token_name in self.name_to_token:
            tokens.append(self.name_to_token[key_token_name])

        # Store key info for pitch-to-degree conversion
        self._encode_key_root = comp.key_root
        self._encode_key_mode = comp.key_mode

        time_sig = comp.time_signature if hasattr(comp, "time_signature") else (4, 4)
        events = self._interleave_n_voices(comp.voices, time_sig=time_sig)
        tokens.extend(events)
        tokens.append(self.EOS)
        return tokens

    def _interleave_n_voices(
        self, voices: list[list[tuple[int, int, int]]],
        time_sig: tuple[int, int] = (4, 4),
    ) -> list[int]:
        tokens: list[int] = []
        all_events: list[tuple[int, int, int, int]] = []
        for voice_idx, voice_notes in enumerate(voices):
            voice_num = voice_idx + 1
            for start, dur, pitch in voice_notes:
                all_events.append((start, voice_num, pitch, dur))
        all_events.sort(key=lambda e: (e[0], e[1]))

        # Pre-compute beat boundaries for the entire piece span
        measure_ticks = ticks_per_measure(time_sig)
        beat_offsets = beat_tick_positions(time_sig)
        max_tick = max((e[0] + e[3] for e in all_events), default=0)
        n_measures = (max_tick // measure_ticks) + 2  # extra safety margin

        # Build sorted list of (abs_tick, beat_number_1indexed) for the piece
        beat_boundaries: list[tuple[int, int]] = []
        for m in range(n_measures):
            for beat_idx, offset in enumerate(beat_offsets):
                abs_tick = m * measure_ticks + offset
                beat_boundaries.append((abs_tick, beat_idx + 1))
        # beat_boundaries is already sorted since m and offsets are ascending

        current_time = 0
        beat_ptr = 0  # pointer into beat_boundaries

        for event_time, voice_num, pitch, duration in all_events:
            # Emit BAR/BEAT tokens for any beat boundaries between current_time and event_time
            while beat_ptr < len(beat_boundaries) and beat_boundaries[beat_ptr][0] <= event_time:
                b_tick, b_num = beat_boundaries[beat_ptr]
                if b_tick >= current_time:
                    # Emit time shift to reach the beat boundary
                    gap = b_tick - current_time
                    if gap > 0:
                        ts_tokens = self._quantize_time_shift(gap)
                        tokens.extend(ts_tokens)
                        current_time = b_tick

                    # Emit BAR if this is beat 1
                    if b_num == 1:
                        tokens.append(self.BAR)
                    # Emit BEAT_N
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

            voice_tok = self.VOICE_TOKENS[voice_num - 1] if voice_num <= 4 else self.VOICE_1
            tokens.append(voice_tok)

            # Emit OCT [SHARP|FLAT] DEG instead of a single Pitch token
            degree_toks = self._pitch_to_degree_tokens(pitch)
            tokens.extend(degree_toks)

            dur_tok = self._duration_to_token(duration)
            if dur_tok is not None:
                tokens.append(dur_tok)

        return tokens

    def _pitch_to_degree_tokens(self, midi_pitch: int) -> list[int]:
        """Convert an absolute MIDI pitch to [OCT_x, (SHARP|FLAT), DEG_y] tokens."""
        midi_pitch = max(self.config.min_pitch, min(self.config.max_pitch, midi_pitch))
        octave, degree, accidental = midi_to_scale_degree(
            midi_pitch, self._encode_key_root, self._encode_key_mode,
        )
        octave = max(self.config.min_octave, min(self.config.max_octave, octave))

        toks: list[int] = []

        oct_name = f"OCT_{octave}"
        tok = self.name_to_token.get(oct_name)
        if tok is not None:
            toks.append(tok)

        if accidental == "sharp":
            tok = self.name_to_token.get("SHARP")
            if tok is not None:
                toks.append(tok)
        elif accidental == "flat":
            tok = self.name_to_token.get("FLAT")
            if tok is not None:
                toks.append(tok)

        deg_name = f"DEG_{degree}"
        tok = self.name_to_token.get(deg_name)
        if tok is not None:
            toks.append(tok)

        return toks

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, tokens: list[int]) -> VoiceComposition:
        """Decode a scale-degree token sequence back into a VoiceComposition."""
        voice_notes: dict[int, list[tuple[int, int, int]]] = {1: [], 2: [], 3: [], 4: []}
        current_time = 0
        current_voice = 1
        key_root = 0
        key_mode = "major"

        pending_octave: int | None = None
        pending_degree: int | None = None
        pending_accidental: str = ""

        for tok in tokens:
            name = self.token_to_name.get(tok, "")

            if name in (
                "PAD", "BOS", "EOS", "SUBJECT_START", "SUBJECT_END",
                "BAR", "BEAT_1", "BEAT_2", "BEAT_3", "BEAT_4", "BEAT_5", "BEAT_6",
                "MODE_2PART", "MODE_3PART", "MODE_4PART", "MODE_FUGUE",
                "STYLE_BACH", "STYLE_BAROQUE", "STYLE_RENAISSANCE", "STYLE_CLASSICAL",
                "FORM_CHORALE", "FORM_INVENTION", "FORM_FUGUE",
                "FORM_SINFONIA", "FORM_QUARTET", "FORM_TRIO_SONATA", "FORM_MOTET",
                "LENGTH_SHORT", "LENGTH_MEDIUM", "LENGTH_LONG", "LENGTH_EXTENDED",
                "METER_2_4", "METER_3_4", "METER_4_4", "METER_6_8",
                "METER_3_8", "METER_ALLA_BREVE",
            ):
                continue

            elif name.startswith("VOICE_"):
                # Reset pending state on voice change
                pending_octave = None
                pending_degree = None
                pending_accidental = ""
                voice_num = int(name[-1])
                current_voice = voice_num

            elif name.startswith("KEY_"):
                key_str = name[4:]
                parts = key_str.rsplit("_", 1)
                if len(parts) == 2:
                    key_mode = parts[1]
                    try:
                        root_name = parts[0].replace("s", "#")
                        key_root = note_name_to_pc(root_name)
                    except ValueError:
                        pass

            elif name.startswith("OCT_"):
                pending_octave = int(name[4:])
                pending_degree = None
                pending_accidental = ""

            elif name == "SHARP":
                pending_accidental = "sharp"

            elif name == "FLAT":
                pending_accidental = "flat"

            elif name.startswith("DEG_"):
                pending_degree = int(name[4:])

            elif name.startswith("Dur_"):
                dur = int(name[4:])
                if pending_octave is not None and pending_degree is not None:
                    midi_pitch = scale_degree_to_midi(
                        pending_octave, pending_degree, pending_accidental,
                        key_root, key_mode,
                    )
                    midi_pitch = max(self.config.min_pitch, min(self.config.max_pitch, midi_pitch))
                    voice_notes[current_voice].append((current_time, dur, midi_pitch))
                # Reset
                pending_octave = None
                pending_degree = None
                pending_accidental = ""

            elif name.startswith("TimeShift_"):
                ts = int(name[10:])
                current_time += ts
                # Reset pending pitch state
                pending_octave = None
                pending_degree = None
                pending_accidental = ""

        voices = []
        for v in [1, 2, 3, 4]:
            if voice_notes[v]:
                voices.append(voice_notes[v])
        while len(voices) < 2:
            voices.append([])

        return VoiceComposition(
            voices=voices, key_root=key_root, key_mode=key_mode, source="decoded",
        )

    def decode_to_pair(self, tokens: list[int]) -> VoicePair:
        """Decode a token sequence back into a VoicePair (backward compat)."""
        comp = self.decode(tokens)
        return comp.to_voice_pair()

    # ------------------------------------------------------------------
    # Duration / time-shift helpers (identical to BachTokenizer)
    # ------------------------------------------------------------------

    def _duration_to_token(self, duration_ticks: int) -> int | None:
        bin_val = self._nearest_bin(duration_ticks, self.config.duration_bins)
        name = f"Dur_{bin_val}"
        return self.name_to_token.get(name)

    def _quantize_time_shift(self, ticks: int) -> list[int]:
        tokens = []
        remaining = ticks
        bins = sorted(self.config.time_shift_bins, reverse=True)
        while remaining > 0:
            best = bins[-1]
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
        return min(bins, key=lambda b: abs(b - value))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tokenizer_type": "scale_degree",
            "vocab_size": self._vocab_size,
            "token_to_name": {str(k): v for k, v in self.token_to_name.items()},
            "config": {
                "min_pitch": self.config.min_pitch,
                "max_pitch": self.config.max_pitch,
                "min_octave": self.config.min_octave,
                "max_octave": self.config.max_octave,
                "duration_bins": self.config.duration_bins,
                "time_shift_bins": self.config.time_shift_bins,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved scale-degree tokenizer to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ScaleDegreeTokenizer":
        with open(path) as f:
            data = json.load(f)
        config = ScaleDegreeTokenizerConfig(**data["config"])
        return cls(config)
