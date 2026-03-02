"""Decoding constraints for scale-degree tokenizer.

Applies voice-range masking on OCT/DEG tokens, a small chromatic penalty on
SHARP/FLAT, and the same anti-degenerate logic as the absolute-pitch
``DecodingConstraints``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
from bach_gen.utils.constants import (
    UPPER_VOICE_RANGE, LOWER_VOICE_RANGE, FORM_VOICE_RANGES,
)
from bach_gen.utils.music_theory import scale_degree_to_midi


@dataclass
class ScaleDegreeConstraintState:
    """Incrementally-maintained state for scale-degree constraint evaluation."""

    current_voice: int = 1
    last_token: int | None = None
    recent_tokens: list[int] = field(default_factory=list)
    pending_octave: int | None = None
    pending_accidental: str = ""
    notes_in_current_voice: int = 0
    notes_per_voice: tuple[int, int, int, int] = (0, 0, 0, 0)


class ScaleDegreeDecodingConstraints:
    """Apply hard constraints to logits during scale-degree generation."""

    def __init__(
        self,
        tokenizer: ScaleDegreeTokenizer,
        key_root: int,
        key_mode: str,
        enforce_range: bool = True,
        chromatic_penalty: float = 1.0,
        form: str = "2-part",
        num_voices: int = 2,
    ):
        self.tokenizer = tokenizer
        self.key_root = key_root
        self.key_mode = key_mode
        self.enforce_range = enforce_range
        self.chromatic_penalty = chromatic_penalty
        self.form = form
        self.num_voices = num_voices

        # Pre-compute voice token -> voice number mapping
        self._voice_token_ids: dict[int, int] = {
            self.tokenizer.VOICE_1: 1,
            self.tokenizer.VOICE_2: 2,
            self.tokenizer.VOICE_3: 3,
            self.tokenizer.VOICE_4: 4,
        }

        # Pre-compute per-voice MIDI ranges
        self._voice_ranges: dict[int, tuple[int, int]] = {}
        for v in range(1, num_voices + 1):
            key = (form, v)
            if key in FORM_VOICE_RANGES:
                self._voice_ranges[v] = FORM_VOICE_RANGES[key]
            elif v == 1:
                self._voice_ranges[v] = UPPER_VOICE_RANGE
            else:
                self._voice_ranges[v] = LOWER_VOICE_RANGE

        # Temporal tokens used for anti-lockstep soft bias.
        self._temporal_tokens: list[int] = []
        for name, tok in self.tokenizer.name_to_token.items():
            if name == "BAR" or name.startswith("BEAT_") or name.startswith("TimeShift_"):
                self._temporal_tokens.append(tok)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def initial_state(
        self, prompt_tokens: list[int],
    ) -> ScaleDegreeConstraintState:
        """Build state by scanning the prompt once."""
        state = ScaleDegreeConstraintState()
        for tok in prompt_tokens:
            state = self.update_state(state, tok)
        return state

    def update_state(
        self, state: ScaleDegreeConstraintState, token: int,
    ) -> ScaleDegreeConstraintState:
        """Return a new state reflecting *token* appended (O(1))."""
        current_voice = state.current_voice
        pending_octave = state.pending_octave
        pending_accidental = state.pending_accidental
        notes_in_voice = state.notes_in_current_voice
        notes_per_voice = state.notes_per_voice

        name = self.tokenizer.token_to_name.get(token, "")

        if token in self._voice_token_ids:
            current_voice = self._voice_token_ids[token]
            pending_octave = None
            pending_accidental = ""
            notes_in_voice = 0
        elif name == "VOICE_SEP":
            # Voice separator: reset pending state, advance voice
            pending_octave = None
            pending_accidental = ""
            notes_in_voice = 0
        elif name.startswith("OCT_"):
            pending_octave = int(name[4:])
            pending_accidental = ""
        elif name == "SHARP":
            pending_accidental = "sharp"
        elif name == "FLAT":
            pending_accidental = "flat"
        elif name.startswith("Dur_"):
            # Note completed — reset pending state, increment note count
            pending_octave = None
            pending_accidental = ""
            notes_in_voice += 1
            idx = max(0, min(3, current_voice - 1))
            mutable = list(notes_per_voice)
            mutable[idx] += 1
            notes_per_voice = tuple(mutable)
        elif name.startswith("TimeShift_"):
            pending_octave = None
            pending_accidental = ""

        recent = state.recent_tokens[-31:] + [token]  # keep last 32

        return ScaleDegreeConstraintState(
            current_voice=current_voice,
            last_token=token,
            recent_tokens=recent,
            pending_octave=pending_octave,
            pending_accidental=pending_accidental,
            notes_in_current_voice=notes_in_voice,
            notes_per_voice=notes_per_voice,
        )

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        logits: torch.Tensor,
        generated_tokens: list[int] | ScaleDegreeConstraintState,
    ) -> torch.Tensor:
        base_logits = logits.clone()

        if isinstance(generated_tokens, ScaleDegreeConstraintState):
            state = generated_tokens
            current_voice = state.current_voice
            range_fn = lambda x: self._apply_range_constraint_from_state(
                x, current_voice, state,
            )
            degenerate_fn = lambda x: self._prevent_degenerate_from_state(x, state)
            coverage_fn = lambda x: self._enforce_min_voice_coverage_from_state(x, state)
        else:
            current_voice = self._get_current_voice(generated_tokens)
            range_fn = lambda x: self._apply_range_constraint(
                x, current_voice, generated_tokens,
            )
            degenerate_fn = lambda x: self._prevent_degenerate(x, generated_tokens)
            coverage_fn = lambda x: self._enforce_min_voice_coverage(x, generated_tokens)

        # Full constraints.
        constrained = self._apply_constraint_pass(
            base_logits,
            apply_range=self.enforce_range,
            apply_chromatic=True,
            apply_degenerate=True,
            range_fn=range_fn,
            degenerate_fn=degenerate_fn,
        )
        constrained = coverage_fn(constrained)
        if self._has_valid_token(constrained):
            return constrained

        # Relax 1: drop degenerate penalties/masks.
        constrained = self._apply_constraint_pass(
            base_logits,
            apply_range=self.enforce_range,
            apply_chromatic=True,
            apply_degenerate=False,
            range_fn=range_fn,
            degenerate_fn=degenerate_fn,
        )
        constrained = coverage_fn(constrained)
        if self._has_valid_token(constrained):
            return constrained

        # Relax 2: also drop chromatic penalty.
        constrained = self._apply_constraint_pass(
            base_logits,
            apply_range=self.enforce_range,
            apply_chromatic=False,
            apply_degenerate=False,
            range_fn=range_fn,
            degenerate_fn=degenerate_fn,
        )
        constrained = coverage_fn(constrained)
        if self._has_valid_token(constrained):
            return constrained

        # Relax 3: also drop range constraints (raw logits).
        constrained = self._apply_constraint_pass(
            base_logits,
            apply_range=False,
            apply_chromatic=False,
            apply_degenerate=False,
            range_fn=range_fn,
            degenerate_fn=degenerate_fn,
        )
        constrained = coverage_fn(constrained)
        return self._ensure_valid_logits(constrained, base_logits)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_valid_token(self, logits: torch.Tensor) -> bool:
        """True iff at least one token has a finite logit."""
        return bool(torch.any(torch.isfinite(logits)).item())

    def _ensure_valid_logits(
        self,
        constrained_logits: torch.Tensor,
        raw_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Guarantee at least one finite token for downstream sampling."""
        if self._has_valid_token(constrained_logits):
            return constrained_logits

        if self._has_valid_token(raw_logits):
            return raw_logits.clone()

        # Last-resort guard when upstream logits are all invalid.
        fallback = torch.full_like(raw_logits, float("-inf"))
        eos_tok = getattr(self.tokenizer, "EOS", None)
        if eos_tok is not None and eos_tok < fallback.size(0):
            fallback[eos_tok] = 0.0
        else:
            fallback[0] = 0.0
        return fallback

    def _apply_constraint_pass(
        self,
        logits: torch.Tensor,
        apply_range: bool,
        apply_chromatic: bool,
        apply_degenerate: bool,
        range_fn,
        degenerate_fn,
    ) -> torch.Tensor:
        """Run one constraint pass with configurable stages."""
        out = logits.clone()
        if apply_range:
            out = range_fn(out)
        if apply_chromatic:
            out = self._apply_chromatic_penalty(out)
        if apply_degenerate:
            out = degenerate_fn(out)
        return out

    def _get_current_voice(self, tokens: list[int]) -> int:
        for tok in reversed(tokens):
            if tok in self._voice_token_ids:
                return self._voice_token_ids[tok]
        return 1

    def _apply_range_constraint(
        self,
        logits: torch.Tensor,
        voice: int,
        generated_tokens: list[int],
    ) -> torch.Tensor:
        """Mask OCT tokens whose *entire* range is outside the voice's MIDI
        bounds, and — when an OCT has already been emitted — mask DEG tokens
        that would produce out-of-range MIDI pitches."""
        lo, hi = self._voice_ranges.get(
            voice, (self.tokenizer.config.min_pitch, self.tokenizer.config.max_pitch),
        )

        # --- OCT masking ---
        for o in range(self.tokenizer.config.min_octave, self.tokenizer.config.max_octave + 1):
            oct_lo = scale_degree_to_midi(o, 1, "", self.key_root, self.key_mode)
            oct_hi = scale_degree_to_midi(o, 7, "sharp", self.key_root, self.key_mode)

            if oct_hi < lo or oct_lo > hi:
                tok = self.tokenizer.name_to_token.get(f"OCT_{o}")
                if tok is not None and tok < logits.size(0):
                    logits[tok] = float("-inf")

        # --- DEG masking given a pending OCT ---
        pending_oct = self._get_pending_octave(generated_tokens)
        if pending_oct is not None:
            pending_acc = self._get_pending_accidental(generated_tokens)
            for d in range(1, 8):
                midi = scale_degree_to_midi(
                    pending_oct, d, pending_acc, self.key_root, self.key_mode,
                )
                if midi < lo or midi > hi:
                    tok = self.tokenizer.name_to_token.get(f"DEG_{d}")
                    if tok is not None and tok < logits.size(0):
                        logits[tok] = float("-inf")

        return logits

    def _apply_range_constraint_from_state(
        self,
        logits: torch.Tensor,
        voice: int,
        state: ScaleDegreeConstraintState,
    ) -> torch.Tensor:
        """Same as _apply_range_constraint but using cached state."""
        lo, hi = self._voice_ranges.get(
            voice, (self.tokenizer.config.min_pitch, self.tokenizer.config.max_pitch),
        )

        # --- OCT masking ---
        for o in range(self.tokenizer.config.min_octave, self.tokenizer.config.max_octave + 1):
            oct_lo = scale_degree_to_midi(o, 1, "", self.key_root, self.key_mode)
            oct_hi = scale_degree_to_midi(o, 7, "sharp", self.key_root, self.key_mode)

            if oct_hi < lo or oct_lo > hi:
                tok = self.tokenizer.name_to_token.get(f"OCT_{o}")
                if tok is not None and tok < logits.size(0):
                    logits[tok] = float("-inf")

        # --- DEG masking given a pending OCT ---
        if state.pending_octave is not None:
            for d in range(1, 8):
                midi = scale_degree_to_midi(
                    state.pending_octave, d, state.pending_accidental,
                    self.key_root, self.key_mode,
                )
                if midi < lo or midi > hi:
                    tok = self.tokenizer.name_to_token.get(f"DEG_{d}")
                    if tok is not None and tok < logits.size(0):
                        logits[tok] = float("-inf")

        return logits

    def _apply_chromatic_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        for name in ("SHARP", "FLAT"):
            tok = self.tokenizer.name_to_token.get(name)
            if tok is not None and tok < logits.size(0):
                logits[tok] -= self.chromatic_penalty
        return logits

    def _prevent_degenerate(
        self, logits: torch.Tensor, tokens: list[int],
    ) -> torch.Tensor:
        if len(tokens) < 4:
            return logits

        last_tok = tokens[-1]
        if last_tok < logits.size(0):
            logits[last_tok] -= 1.0

        if len(tokens) >= 6:
            pattern = tokens[-3:]
            if tokens[-6:-3] == pattern:
                for tok in pattern:
                    if tok < logits.size(0):
                        logits[tok] -= 2.0

        logits[self.tokenizer.PAD] = float("-inf")
        logits[self.tokenizer.BOS] = float("-inf")
        logits = self._apply_soft_anti_lockstep(logits, tokens[-32:])
        return logits

    def _prevent_degenerate_from_state(
        self, logits: torch.Tensor, state: ScaleDegreeConstraintState,
    ) -> torch.Tensor:
        """Same logic as _prevent_degenerate but using cached state."""
        if len(state.recent_tokens) < 4:
            return logits

        if state.last_token is not None and state.last_token < logits.size(0):
            logits[state.last_token] -= 1.0

        if len(state.recent_tokens) >= 6:
            pattern = state.recent_tokens[-3:]
            if state.recent_tokens[-6:-3] == pattern:
                for tok in pattern:
                    if tok < logits.size(0):
                        logits[tok] -= 2.0

        logits[self.tokenizer.PAD] = float("-inf")
        logits[self.tokenizer.BOS] = float("-inf")

        # Prevent VOICE_SEP if current voice has too few notes (avoid empty voices)
        min_notes_before_sep = 4
        voice_sep_tok = self.tokenizer.name_to_token.get("VOICE_SEP")
        if voice_sep_tok is not None and voice_sep_tok < logits.size(0):
            if state.notes_in_current_voice < min_notes_before_sep:
                logits[voice_sep_tok] = float("-inf")

        logits = self._apply_soft_anti_lockstep(logits, state.recent_tokens)
        return logits

    def _apply_soft_anti_lockstep(
        self,
        logits: torch.Tensor,
        recent_tokens: list[int],
    ) -> torch.Tensor:
        """Softly discourage excessive same-time multi-voice attacks in fugues."""
        if self.form != "fugue":
            return logits
        if not recent_tokens:
            return logits

        voices_in_slot: list[int] = []
        for tok in reversed(recent_tokens):
            name = self.tokenizer.token_to_name.get(tok, "")
            if name == "BAR" or name.startswith("BEAT_") or name.startswith("TimeShift_"):
                break
            if name == "VOICE_SEP":
                break
            if tok in self._voice_token_ids:
                voices_in_slot.append(self._voice_token_ids[tok])

        distinct = len(set(voices_in_slot))
        if distinct <= 1:
            return logits

        voice_tokens = getattr(self.tokenizer, "VOICE_TOKENS", [])
        if distinct >= 3:
            # Heavy same-time stacking: nudge toward temporal advance.
            for tok in voice_tokens:
                if tok < logits.size(0) and torch.isfinite(logits[tok]):
                    logits[tok] -= 1.1
            for tok in self._temporal_tokens:
                if tok < logits.size(0) and torch.isfinite(logits[tok]):
                    logits[tok] += 0.35
        elif distinct == 2:
            # Mild pressure away from triadic/tetradic block onset.
            for tok in voice_tokens:
                if tok < logits.size(0) and torch.isfinite(logits[tok]):
                    logits[tok] -= 0.25

        return logits

    # ------------------------------------------------------------------
    # State helpers (for list-based fallback path)
    # ------------------------------------------------------------------

    def _get_pending_octave(self, tokens: list[int]) -> int | None:
        """Walk backward to find the most recent OCT token that hasn't been
        consumed by a DUR yet."""
        for tok in reversed(tokens):
            name = self.tokenizer.token_to_name.get(tok, "")
            if name.startswith("OCT_"):
                return int(name[4:])
            if name.startswith("Dur_") or name.startswith("TimeShift_") or name.startswith("VOICE_"):
                return None
        return None

    def _get_pending_accidental(self, tokens: list[int]) -> str:
        """Walk backward to find a pending SHARP/FLAT between the last OCT and
        now."""
        for tok in reversed(tokens):
            name = self.tokenizer.token_to_name.get(tok, "")
            if name == "SHARP":
                return "sharp"
            if name == "FLAT":
                return "flat"
            if name.startswith("OCT_") or name.startswith("Dur_") or name.startswith("TimeShift_") or name.startswith("VOICE_"):
                return ""
        return ""

    def _required_voice_count(self) -> int:
        """Required non-empty voices for this generation request."""
        return max(2, min(self.num_voices, 4))

    def _enforce_min_voice_coverage_from_state(
        self, logits: torch.Tensor, state: ScaleDegreeConstraintState,
    ) -> torch.Tensor:
        return self._apply_voice_coverage_floor(logits, state.notes_per_voice)

    def _enforce_min_voice_coverage(
        self, logits: torch.Tensor, tokens: list[int],
    ) -> torch.Tensor:
        return self._apply_voice_coverage_floor(logits, self._notes_per_voice_from_tokens(tokens))

    def _apply_voice_coverage_floor(
        self, logits: torch.Tensor, notes_per_voice: tuple[int, int, int, int],
    ) -> torch.Tensor:
        required = self._required_voice_count()
        covered = sum(1 for i in range(required) if notes_per_voice[i] > 0)
        if covered >= required:
            return logits

        if self.tokenizer.EOS < logits.size(0):
            logits[self.tokenizer.EOS] = float("-inf")

        voice_tokens = getattr(self.tokenizer, "VOICE_TOKENS", [])
        for i in range(required):
            if notes_per_voice[i] > 0:
                continue
            if i >= len(voice_tokens):
                continue
            voice_tok = voice_tokens[i]
            if voice_tok < logits.size(0) and torch.isfinite(logits[voice_tok]):
                logits[voice_tok] += 1.5
        return logits

    def _notes_per_voice_from_tokens(
        self, tokens: list[int],
    ) -> tuple[int, int, int, int]:
        counts = [0, 0, 0, 0]
        current_voice = 1
        for tok in tokens:
            if tok in self._voice_token_ids:
                current_voice = self._voice_token_ids[tok]
                continue
            name = self.tokenizer.token_to_name.get(tok, "")
            if name.startswith("Dur_"):
                idx = max(0, min(3, current_voice - 1))
                counts[idx] += 1
        return tuple(counts)
