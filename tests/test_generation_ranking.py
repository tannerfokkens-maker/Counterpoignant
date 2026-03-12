from __future__ import annotations

from bach_gen.data.extraction import VoiceComposition
from bach_gen.evaluation.scorer import ScoreBreakdown
from bach_gen.generation.generator import (
    GenerationResult,
    _push_top_result,
    normalize_rank_by,
    score_rank_value,
)


def _candidate(*, composite: float, bach: float, impact: float) -> GenerationResult:
    score = ScoreBreakdown(
        composite=composite,
        details={
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
