from __future__ import annotations

from bach_gen.data.extraction import VoiceComposition
from bach_gen.evaluation.scorer import ScoreBreakdown
from bach_gen.generation.generator import (
    GenerationResult,
    _voice_count_matches_request,
    _push_top_result,
    normalize_rank_by,
    score_rank_value,
)


def _candidate(
    *,
    composite: float,
    bach: float,
    impact: float,
    voice_balance: float = 0.0,
    cadence: float = 0.0,
    sequential: float = 0.0,
    onset: float = 0.0,
) -> GenerationResult:
    score = ScoreBreakdown(
        composite=composite,
        details={
            "contrapuntal": {
                "voice_balance": voice_balance,
                "sequential_patterns": sequential,
                "onset_staggering": onset,
            },
            "structural": {
                "cadence": cadence,
            },
            "style_proxies": {
                "bach_similarity": bach,
                "rhetorical_impact": impact,
                "demo_bach_balance": min(bach, impact),
            }
        },
    )
    return GenerationResult(
        composition=VoiceComposition(
            voices=[[], []],
            key_root=0,
            key_mode="major",
            source="test",
        ),
        tokens=[],
        score=score,
    )


def test_score_rank_value_reads_style_proxy_metrics() -> None:
    result = _candidate(composite=0.81, bach=0.78, impact=0.91)

    assert normalize_rank_by("demo_bach_balance") == "demo-bach-balance"
    assert normalize_rank_by("fugue_balance") == "fugue-balance"
    assert score_rank_value(result.score, "composite") == 0.81
    assert score_rank_value(result.score, "bach-similarity") == 0.78
    assert score_rank_value(result.score, "rhetorical-impact") == 0.91
    assert score_rank_value(result.score, "demo-bach-balance") == 0.78


def test_push_top_result_can_rank_by_demo_bach_balance() -> None:
    composite_winner = _candidate(composite=0.86, bach=0.64, impact=0.94)
    balanced_winner = _candidate(composite=0.82, bach=0.84, impact=0.86)

    by_composite: list[GenerationResult] = []
    _push_top_result(by_composite, composite_winner, top_k_results=1, rank_by="composite")
    _push_top_result(by_composite, balanced_winner, top_k_results=1, rank_by="composite")
    assert by_composite[0] is composite_winner

    by_balance: list[GenerationResult] = []
    _push_top_result(by_balance, composite_winner, top_k_results=1, rank_by="demo-bach-balance")
    _push_top_result(by_balance, balanced_winner, top_k_results=1, rank_by="demo-bach-balance")
    assert by_balance[0] is balanced_winner


def test_push_top_result_can_rank_by_fugue_balance() -> None:
    composite_winner = _candidate(
        composite=0.84,
        bach=0.78,
        impact=0.80,
        voice_balance=0.35,
        cadence=0.42,
        sequential=0.38,
        onset=0.45,
    )
    fugue_winner = _candidate(
        composite=0.80,
        bach=0.77,
        impact=0.81,
        voice_balance=0.82,
        cadence=0.76,
        sequential=0.71,
        onset=0.74,
    )

    by_fugue_balance: list[GenerationResult] = []
    _push_top_result(by_fugue_balance, composite_winner, top_k_results=1, rank_by="fugue-balance")
    _push_top_result(by_fugue_balance, fugue_winner, top_k_results=1, rank_by="fugue-balance")
    assert by_fugue_balance[0] is fugue_winner


def test_voice_count_matching_supports_exact_and_minimum_modes() -> None:
    voices = [[(0, 1, 60)], [(0, 1, 55)], [(0, 1, 48)], []]

    assert _voice_count_matches_request(voices, num_voices=3, exact_voice_count=True)
    assert not _voice_count_matches_request(voices, num_voices=4, exact_voice_count=True)
    assert _voice_count_matches_request(voices, num_voices=3, exact_voice_count=False)
