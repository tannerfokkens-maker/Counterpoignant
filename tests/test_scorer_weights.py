"""Tests for scorer weights and form-dependent weight selection."""

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


def test_default_weights_sum_to_one():
    """DEFAULT_WEIGHTS must sum to 1.0 and contain the 6 active dimensions."""
    w = scorer.DEFAULT_WEIGHTS
    assert abs(sum(w.values()) - 1.0) < 1e-9
    expected_keys = {
        "voice_leading", "statistical", "structural",
        "contrapuntal", "completeness", "thematic_recall",
    }
    assert set(w.keys()) == expected_keys


def test_get_weights_for_form_returns_defaults_without_calibration(monkeypatch):
    """Without calibration JSON, all forms fall back to DEFAULT_WEIGHTS."""
    monkeypatch.setattr(scorer, "_form_weights", {})
    assert scorer.get_weights_for_form("chorale") is scorer.DEFAULT_WEIGHTS
    assert scorer.get_weights_for_form("fugue") is scorer.DEFAULT_WEIGHTS
    assert scorer.get_weights_for_form(None) is scorer.DEFAULT_WEIGHTS


def test_get_weights_for_form_uses_calibrated_when_available(monkeypatch):
    """Calibrated per-form weights take priority over defaults."""
    custom = {"voice_leading": 0.5, "thematic_recall": 0.5}
    monkeypatch.setattr(scorer, "_form_weights", {"fugue": custom})
    assert scorer.get_weights_for_form("fugue") is custom
    assert scorer.get_weights_for_form("chorale") is scorer.DEFAULT_WEIGHTS


def test_form_specific_weights_change_composite(monkeypatch):
    """Different per-form weights produce different composite scores."""
    # Fix all dimensions to 1.0 except thematic recall (0.0) to isolate weight effect
    monkeypatch.setattr(scorer, "score_voice_leading", lambda comp: (1.0, {}))
    monkeypatch.setattr(scorer, "score_statistical", lambda comp: (1.0, {}))
    monkeypatch.setattr(scorer, "score_structural", lambda comp: (1.0, {}))
    monkeypatch.setattr(scorer, "score_contrapuntal", lambda comp: (1.0, {}))
    monkeypatch.setattr(scorer, "_score_completeness", lambda comp: 1.0)
    monkeypatch.setattr(scorer, "score_thematic_recall", lambda comp, token_sequence=None, tokenizer=None: 0.0)

    # Inject calibrated weights: chorale has low thematic weight, fugue has high
    monkeypatch.setattr(scorer, "_form_weights", {
        "chorale": {
            "voice_leading": 0.20, "statistical": 0.20, "structural": 0.20,
            "contrapuntal": 0.20, "completeness": 0.15, "thematic_recall": 0.05,
        },
        "fugue": {
            "voice_leading": 0.15, "statistical": 0.15, "structural": 0.15,
            "contrapuntal": 0.15, "completeness": 0.15, "thematic_recall": 0.25,
        },
    })

    comp = _tiny_comp()
    chorale_score = scorer.score_composition(comp, token_sequence=[1, 2], form="chorale")
    fugue_score = scorer.score_composition(comp, token_sequence=[1, 2], form="fugue")

    # With thematic recall forced to 0, higher thematic weight means lower composite
    assert chorale_score.composite > fugue_score.composite
    # chorale: 0.95 * 1.0 + 0.05 * 0.0 = 0.95; fugue: 0.75 * 1.0 + 0.25 * 0.0 = 0.75
    assert abs(chorale_score.composite - 0.95) < 1e-9
    assert abs(fugue_score.composite - 0.75) < 1e-9
