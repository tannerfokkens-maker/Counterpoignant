"""Hard constraints during decoding: voice range, key adherence."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from bach_gen.data.tokenizer import BachTokenizer
from bach_gen.utils.constants import (
    UPPER_VOICE_RANGE,
    LOWER_VOICE_RANGE,
    FORM_VOICE_RANGES,
)
from bach_gen.utils.music_theory import get_scale


@dataclass
class ConstraintState:
    """Incrementally-maintained state for O(1) constraint evaluation."""

    current_voice: int = 1
    last_token: int | None = None
    recent_tokens: list[int] = field(default_factory=list)


class DecodingConstraints:
    """Apply hard constraints to logits during generation."""

    def __init__(
        self,
        tokenizer: BachTokenizer,
        key_root: int,
        key_mode: str,
        enforce_key: bool = True,
        enforce_range: bool = True,
        key_weight: float = 2.0,
        form: str = "2-part",
        num_voices: int = 2,
    ):
        self.tokenizer = tokenizer
        self.key_root = key_root
        self.key_mode = key_mode
        self.enforce_key = enforce_key
        self.enforce_range = enforce_range
        self.key_weight = key_weight
        self.form = form
        self.num_voices = num_voices

        self.scale_pcs = set(get_scale(key_root, key_mode))

        # Pre-compute voice token -> voice number mapping
        self._voice_token_ids: dict[int, int] = {
            self.tokenizer.VOICE_1: 1,
            self.tokenizer.VOICE_2: 2,
            self.tokenizer.VOICE_3: 3,
            self.tokenizer.VOICE_4: 4,
        }

        # Pre-compute voice ranges for this form
        self._voice_ranges: dict[int, tuple[int, int]] = {}
        for v in range(1, num_voices + 1):
            key = (form, v)
            if key in FORM_VOICE_RANGES:
                self._voice_ranges[v] = FORM_VOICE_RANGES[key]
            elif v == 1:
                self._voice_ranges[v] = UPPER_VOICE_RANGE
            else:
                self._voice_ranges[v] = LOWER_VOICE_RANGE

    def initial_state(self, prompt_tokens: list[int]) -> ConstraintState:
        """Build a ConstraintState by scanning the prompt once."""
        state = ConstraintState()
        for tok in prompt_tokens:
            state = self.update_state(state, tok)
        return state

    def update_state(self, state: ConstraintState, token: int) -> ConstraintState:
        """Return a new ConstraintState reflecting *token* appended (O(1))."""
        current_voice = state.current_voice
        if token in self._voice_token_ids:
            current_voice = self._voice_token_ids[token]

        recent = state.recent_tokens[-5:] + [token]  # keep last 6

        return ConstraintState(
            current_voice=current_voice,
            last_token=token,
            recent_tokens=recent,
        )

    def apply(
        self,
        logits: torch.Tensor,
        generated_tokens: list[int] | ConstraintState,
    ) -> torch.Tensor:
        """Apply constraints to logits.

        Args:
            logits: (vocab_size,) raw logits.
            generated_tokens: Tokens generated so far, or a ConstraintState.

        Returns:
            Modified logits.
        """
        logits = logits.clone()

        if isinstance(generated_tokens, ConstraintState):
            state = generated_tokens
            current_voice = state.current_voice

            if self.enforce_range:
                logits = self._apply_range_constraint(logits, current_voice)
            if self.enforce_key:
                logits = self._apply_key_bias(logits)

            logits = self._prevent_degenerate_from_state(logits, state)
        else:
            current_voice = self._get_current_voice(generated_tokens)

            if self.enforce_range:
                logits = self._apply_range_constraint(logits, current_voice)
            if self.enforce_key:
                logits = self._apply_key_bias(logits)

            logits = self._prevent_degenerate(logits, generated_tokens)

        return logits

    def _get_current_voice(self, tokens: list[int]) -> int:
        """Determine which voice is currently active (1-4)."""
        for tok in reversed(tokens):
            if tok in self._voice_token_ids:
                return self._voice_token_ids[tok]
        return 1  # default to voice 1

    def _apply_range_constraint(
        self,
        logits: torch.Tensor,
        voice: int,
    ) -> torch.Tensor:
        """Mask out pitches outside voice range."""
        lo, hi = self._voice_ranges.get(voice, (self.tokenizer.config.min_pitch,
                                                 self.tokenizer.config.max_pitch))

        for pitch in range(self.tokenizer.config.min_pitch,
                          self.tokenizer.config.max_pitch + 1):
            name = f"Pitch_{pitch}"
            tok = self.tokenizer.name_to_token.get(name)
            if tok is not None and (pitch < lo or pitch > hi):
                logits[tok] = float("-inf")

        return logits

    def _apply_key_bias(self, logits: torch.Tensor) -> torch.Tensor:
        """Boost probability of scale tones, reduce non-scale tones."""
        for pitch in range(self.tokenizer.config.min_pitch,
                          self.tokenizer.config.max_pitch + 1):
            pc = pitch % 12
            name = f"Pitch_{pitch}"
            tok = self.tokenizer.name_to_token.get(name)
            if tok is not None:
                if pc in self.scale_pcs:
                    logits[tok] += self.key_weight
                else:
                    logits[tok] -= self.key_weight * 0.5

        return logits

    def _prevent_degenerate(
        self,
        logits: torch.Tensor,
        tokens: list[int],
    ) -> torch.Tensor:
        """Prevent degenerate patterns (excessive repetition, etc.)."""
        if len(tokens) < 4:
            return logits

        # Penalize immediate repetition of the same token
        last_tok = tokens[-1]
        if last_tok < logits.size(0):
            logits[last_tok] -= 1.0

        # Penalize repeating the same 3-token pattern
        if len(tokens) >= 6:
            pattern = tokens[-3:]
            if tokens[-6:-3] == pattern:
                for tok in pattern:
                    if tok < logits.size(0):
                        logits[tok] -= 2.0

        # Prevent generating PAD or BOS mid-sequence
        logits[self.tokenizer.PAD] = float("-inf")
        logits[self.tokenizer.BOS] = float("-inf")

        return logits

    def _prevent_degenerate_from_state(
        self,
        logits: torch.Tensor,
        state: ConstraintState,
    ) -> torch.Tensor:
        """Same logic as _prevent_degenerate but using cached state."""
        if len(state.recent_tokens) < 4:
            return logits

        # Penalize immediate repetition
        if state.last_token is not None and state.last_token < logits.size(0):
            logits[state.last_token] -= 1.0

        # Penalize repeating the same 3-token pattern
        if len(state.recent_tokens) >= 6:
            pattern = state.recent_tokens[-3:]
            if state.recent_tokens[-6:-3] == pattern:
                for tok in pattern:
                    if tok < logits.size(0):
                        logits[tok] -= 2.0

        # Prevent generating PAD or BOS mid-sequence
        logits[self.tokenizer.PAD] = float("-inf")
        logits[self.tokenizer.BOS] = float("-inf")

        return logits
