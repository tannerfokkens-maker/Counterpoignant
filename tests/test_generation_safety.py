"""Safety regressions for constrained sampling during generation."""

from __future__ import annotations

import torch

from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
from bach_gen.generation.sampling import sample_next_token
from bach_gen.generation.scale_degree_constraints import (
    ScaleDegreeConstraintState,
    ScaleDegreeDecodingConstraints,
)


def test_sample_next_token_falls_back_to_raw_logits_when_filtered_invalid():
    constrained_logits = torch.full((4,), float("-inf"))
    raw_logits = torch.full((4,), float("-inf"))
    raw_logits[2] = 0.0

    token = sample_next_token(
        constrained_logits,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        fallback_logits=raw_logits,
    )

    assert token == 2


def test_scale_degree_constraints_relax_degenerate_stage_first():
    tokenizer = ScaleDegreeTokenizer()
    constraints = ScaleDegreeDecodingConstraints(
        tokenizer=tokenizer,
        key_root=0,
        key_mode="major",
        enforce_range=False,
    )

    logits = torch.full((tokenizer.vocab_size,), float("-inf"))
    logits[tokenizer.BOS] = 5.0

    state = ScaleDegreeConstraintState(
        current_voice=1,
        last_token=tokenizer.BOS,
        recent_tokens=[10, 11, 12, tokenizer.BOS],
        notes_in_current_voice=4,
    )

    constrained = constraints.apply(logits, state)

    assert torch.isfinite(constrained[tokenizer.BOS]).item()
    assert torch.any(torch.isfinite(constrained)).item()


def test_scale_degree_constraints_relax_range_as_last_resort():
    tokenizer = ScaleDegreeTokenizer()
    constraints = ScaleDegreeDecodingConstraints(
        tokenizer=tokenizer,
        key_root=0,
        key_mode="major",
        enforce_range=True,
        form="2-part",
        num_voices=2,
    )

    oct2_tok = tokenizer.name_to_token["OCT_2"]
    logits = torch.full((tokenizer.vocab_size,), float("-inf"))
    logits[oct2_tok] = 3.0

    state = ScaleDegreeConstraintState(
        current_voice=1,
        recent_tokens=[tokenizer.VOICE_1],
    )

    fully_constrained = constraints._apply_constraint_pass(
        logits,
        apply_range=True,
        apply_chromatic=True,
        apply_degenerate=True,
        range_fn=lambda x: constraints._apply_range_constraint_from_state(x, 1, state),
        degenerate_fn=lambda x: constraints._prevent_degenerate_from_state(x, state),
    )
    assert not torch.isfinite(fully_constrained[oct2_tok]).item()

    relaxed = constraints.apply(logits, state)

    assert torch.isfinite(relaxed[oct2_tok]).item()
    assert torch.any(torch.isfinite(relaxed)).item()
