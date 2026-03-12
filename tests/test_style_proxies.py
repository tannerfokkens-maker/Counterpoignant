from __future__ import annotations

from bach_gen.evaluation.scorer import _compute_style_proxies


def test_style_proxies_distinguish_bach_similarity_from_rhetorical_impact() -> None:
    dramatic = _compute_style_proxies(
        form="fugue",
        voice_leading=0.90,
        statistical=0.35,
        structural_details={
            "cadence": 0.88,
            "phrase_structure": 0.90,
            "key_consistency": 0.76,
            "modulation": 0.86,
        },
        contrapuntal_details={
            "voice_independence": 0.72,
            "contrary_at_cadences": 0.48,
            "sequential_patterns": 0.42,
            "register_consistency": 0.88,
            "melodic_coherence": 0.90,
            "onset_staggering": 0.88,
            "voice_balance": 0.62,
        },
        completeness=0.92,
        thematic_recall=0.62,
    )
    bach_like = _compute_style_proxies(
        form="fugue",
        voice_leading=0.96,
        statistical=0.50,
        structural_details={
            "cadence": 0.64,
            "phrase_structure": 0.72,
            "key_consistency": 0.88,
            "modulation": 0.70,
        },
        contrapuntal_details={
            "voice_independence": 0.92,
            "contrary_at_cadences": 0.54,
            "sequential_patterns": 0.70,
            "register_consistency": 0.94,
            "melodic_coherence": 0.78,
            "onset_staggering": 0.66,
            "voice_balance": 0.86,
        },
        completeness=0.88,
        thematic_recall=0.90,
    )

    assert dramatic["rhetorical_impact"] > dramatic["bach_similarity"]
    assert bach_like["bach_similarity"] > bach_like["rhetorical_impact"]
    assert bach_like["bach_similarity"] > dramatic["bach_similarity"]
    assert dramatic["rhetorical_impact"] > bach_like["rhetorical_impact"] - 0.02


def test_style_proxies_penalize_surface_rhetoric_without_substance() -> None:
    dramatic_surface_only = _compute_style_proxies(
        form="fugue",
        voice_leading=0.12,
        statistical=0.30,
        structural_details={
            "cadence": 0.78,
            "phrase_structure": 0.88,
            "key_consistency": 0.84,
            "modulation": 0.86,
        },
        contrapuntal_details={
            "voice_independence": 0.96,
            "contrary_at_cadences": 0.02,
            "sequential_patterns": 0.30,
            "register_consistency": 0.0,
            "melodic_coherence": 0.77,
            "onset_staggering": 0.78,
            "voice_balance": 0.84,
        },
        completeness=0.94,
        thematic_recall=0.0,
    )
    dramatic_with_substance = _compute_style_proxies(
        form="fugue",
        voice_leading=0.91,
        statistical=0.41,
        structural_details={
            "cadence": 0.82,
            "phrase_structure": 0.87,
            "key_consistency": 0.82,
            "modulation": 0.84,
        },
        contrapuntal_details={
            "voice_independence": 0.82,
            "contrary_at_cadences": 0.42,
            "sequential_patterns": 0.54,
            "register_consistency": 0.90,
            "melodic_coherence": 0.83,
            "onset_staggering": 0.81,
            "voice_balance": 0.78,
        },
        completeness=0.93,
        thematic_recall=0.72,
    )

    assert dramatic_surface_only["rhetorical_impact"] < 0.50
    assert dramatic_with_substance["rhetorical_impact"] > dramatic_surface_only["rhetorical_impact"] + 0.20
