from __future__ import annotations

from bach_gen.cli import _infer_evaluate_mode


def test_infer_evaluate_mode_uses_bwv_for_inventions() -> None:
    assert _infer_evaluate_mode("data/midi/all/bach/kernscores__bwv0772.krn.fromscore.mid", 2) == "invention"


def test_infer_evaluate_mode_uses_keywords_for_fugues() -> None:
    assert _infer_evaluate_mode("data/midi/all/bach/kernscores__artfugue-001.krn.fromscore.mid", 4) == "fugue"
    assert _infer_evaluate_mode("data/midi/all/bach/kernscores__wtc1f01.krn.fromscore.mid", 4) == "fugue"


def test_infer_evaluate_mode_uses_keywords_for_sinfonias() -> None:
    path = "data/midi/all/bach/kunstderfuge__bach-js_three-part_inventions_1_(c)icking-archive.mid"
    assert _infer_evaluate_mode(path, 3) == "sinfonia"


def test_infer_evaluate_mode_falls_back_to_track_count() -> None:
    assert _infer_evaluate_mode("output/unknown.mid", 4) == "chorale"
    assert _infer_evaluate_mode("output/unknown.mid", 3) == "sinfonia"
