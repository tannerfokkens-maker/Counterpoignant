"""Tests for form-dependent scorer weight presets."""

from __future__ import annotations

from bach_gen.data.extraction import VoiceComposition
from bach_gen.evaluation import scorer


def _tiny_comp() -> VoiceComposition:
    return VoiceComposition(
        voices=[
            [(0, 480, 60), (480, 480, 62)],
            [(0, 960, 48)],
        ],
        key_root=0,
        key_mode="major",
        source="unit-test",
    )


def test_default_weight_presets_match_expected_ranges():
    chorale = scorer.get_default_weights("chorale")
    fugue = scorer.get_default_weights("fugue")

    assert chorale["voice_leading"] == 0.04
    assert chorale["completeness"] == 0.04
    assert chorale["contrapuntal"] == 0.20
    assert chorale["statistical"] == 0.12
    assert chorale["thematic_recall"] == 0.05

    assert fugue["voice_leading"] == 0.04
    assert fugue["completeness"] == 0.04
    assert fugue["contrapuntal"] == 0.20
    assert fugue["statistical"] == 0.12
    assert fugue["thematic_recall"] == 0.15

    assert abs(sum(chorale.values()) - 1.0) < 1e-9
    assert abs(sum(fugue.values()) - 1.0) < 1e-9
    assert chorale["thematic_recall"] < fugue["thematic_recall"]


def test_form_specific_thematic_weight_changes_composite(monkeypatch):
    # Fix all dimensions to 1 except thematic recall to isolate thematic weight.
    monkeypatch.setattr(scorer, "score_voice_leading", lambda comp: (1.0, {}))
    monkeypatch.setattr(scorer, "score_statistical", lambda comp: (1.0, {}))
    monkeypatch.setattr(scorer, "score_structural", lambda comp: (1.0, {}))
    monkeypatch.setattr(scorer, "score_contrapuntal", lambda comp: (1.0, {}))
    monkeypatch.setattr(scorer, "score_information", lambda tokens, model=None, vocab_size=None: (1.0, {}))
    monkeypatch.setattr(scorer, "_score_completeness", lambda comp: 1.0)
    monkeypatch.setattr(scorer, "score_thematic_recall", lambda comp, token_sequence=None, tokenizer=None: 0.0)

    comp = _tiny_comp()
    chorale_score = scorer.score_composition(comp, token_sequence=[1, 2], form="chorale")
    fugue_score = scorer.score_composition(comp, token_sequence=[1, 2], form="fugue")

    # With thematic recall forced to 0, lower thematic weight should score higher.
    assert chorale_score.composite > fugue_score.composite
    assert abs((chorale_score.composite - fugue_score.composite) - 0.10) < 1e-9
