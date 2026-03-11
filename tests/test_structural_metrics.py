from __future__ import annotations

from bach_gen.evaluation.structural import _score_length


def test_score_length_keeps_extended_contrapuntal_works_high() -> None:
    long_voice = [(i * 480, 480, 60 + (i % 5)) for i in range(78 * 4)]
    lower_voice = [(i * 480, 480, 48 + (i % 3)) for i in range(78 * 4)]
    score = _score_length([long_voice, lower_voice])
    assert score >= 0.95
