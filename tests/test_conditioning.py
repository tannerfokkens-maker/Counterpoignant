"""Tests for cadence/subject conditioning pipeline."""

from __future__ import annotations

import random

from bach_gen.data.conditioning import (
    apply_conditioning_dropout,
    cadence_token_ids_by_tick,
    detect_cadence_events,
    detect_subject_entries,
    subject_boundary_note_indices,
)
from bach_gen.data.extraction import VoiceComposition
from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
from bach_gen.generation.generator import _build_structural_control_state


def test_detect_cadence_events_finds_simple_pac():
    # C-major: bass G->C and soprano resolves to C at bar 2.
    comp = VoiceComposition(
        voices=[
            [(0, 1920, 64), (1920, 960, 60)],  # soprano: E -> C
            [(0, 1920, 55), (1920, 960, 48)],  # bass: G -> C
        ],
        key_root=0,
        key_mode="major",
        source="unit-cadence",
        time_signature=(4, 4),
    )

    events = detect_cadence_events(comp, min_confidence=2.0)

    assert any(e.token_name == "CAD_PAC" and e.tick == 1920 for e in events)


def test_detect_subject_entries_finds_exposition_and_late_entry():
    subject = [(0, 480, 60), (480, 480, 62), (960, 480, 61), (1440, 480, 63), (1920, 480, 64)]
    transposed = [(3840, 480, 67), (4320, 480, 69), (4800, 480, 68), (5280, 480, 70), (5760, 480, 71)]

    comp = VoiceComposition(
        voices=[subject, transposed],
        key_root=0,
        key_mode="major",
        source="unit-subject",
        time_signature=(4, 4),
    )

    entries = detect_subject_entries(comp, min_quality=0.8, min_match_ratio=0.7)

    assert len(entries) >= 2
    assert any(e.is_exposition for e in entries)
    assert any(e.voice_index == 1 for e in entries)


def test_apply_conditioning_dropout_keeps_first_subject_pair():
    subj_start = 900
    subj_end = 901
    cad = 902
    tokens = [1, subj_start, 100, subj_end, 200, subj_start, 101, subj_end, cad, 2]

    dropped = apply_conditioning_dropout(
        tokens=tokens,
        cadence_token_ids={cad},
        subject_start_token_ids={subj_start},
        subject_end_token_ids={subj_end},
        dropout_prob=1.0,
        rng=random.Random(123),
        keep_first_subject_entry=True,
    )

    assert subj_start in dropped and subj_end in dropped
    assert dropped.count(subj_start) == 1
    assert dropped.count(subj_end) == 1
    assert cad not in dropped


def test_scale_degree_encode_inserts_cadence_and_subject_markers():
    tokenizer = ScaleDegreeTokenizer()
    comp = VoiceComposition(
        voices=[
            [(0, 960, 60), (960, 960, 62), (1920, 960, 60)],
            [(0, 960, 48), (960, 960, 55), (1920, 960, 48)],
        ],
        key_root=0,
        key_mode="major",
        source="unit-encode-markers",
        time_signature=(4, 4),
    )

    cad_events = detect_cadence_events(comp, min_confidence=1.0)
    cad_map = cadence_token_ids_by_tick(cad_events, tokenizer.name_to_token)
    subj_starts = {(1, 0)}
    subj_ends = {(1, 1)}

    tokens = tokenizer.encode(
        comp,
        form="invention",
        cadence_tokens_by_tick=cad_map,
        subject_start_markers=subj_starts,
        subject_end_markers=subj_ends,
    )
    names = [tokenizer.token_to_name[t] for t in tokens]

    if "CAD_PAC" in names:
        cad_idx = names.index("CAD_PAC")
        assert names[cad_idx + 1] == "BAR"
    assert "SUBJECT_START" in names


def test_structural_control_state_injects_subject_reentries():
    tokenizer = ScaleDegreeTokenizer()
    prompt = [tokenizer.BOS, tokenizer.BAR, tokenizer.BEAT_1]
    state = _build_structural_control_state(
        tokenizer=tokenizer,
        prompt_tokens=prompt,
        cadence_density=None,
        min_subject_entries=2,
        subject_spacing_bars=2,
    )
    assert state is not None

    # Reach next subject-insertion bar.
    state.update(tokenizer.BAR, tokenizer)
    state.update(tokenizer.BAR, tokenizer)
    forced = state.maybe_force_token(tokenizer)
    subj_tok = tokenizer.name_to_token.get("SUBJECT_START")
    assert forced == subj_tok
