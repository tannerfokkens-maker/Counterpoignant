"""Tests for conditioning-label audit helpers."""

from __future__ import annotations

from pathlib import Path

from bach_gen.data.label_audit import (
    _infer_form_from_source,
    _load_reference_metadata,
    _resolve_companion_source,
    build_conditioning_audit_rows,
    summarize_conditioning_audit,
)


def test_resolve_companion_source_for_kernscores_mid() -> None:
    path = Path(
        "data/midi/all/bach/kernscores__artfugue-001.krn.fromscore.mid"
    ).resolve()
    companion = _resolve_companion_source(path)
    assert companion is not None
    assert companion.as_posix().endswith("data/midi/kernscores/bach/artfugue-001.krn")


def test_infer_form_from_source_uses_bach_keywords() -> None:
    assert _infer_form_from_source("data/midi/all/bach/kernscores__wtc1f06.krn.fromscore.mid", 4) == "fugue"
    assert _infer_form_from_source("data/midi/all/bach/kunstderfuge__bach-js_two-part_inventions_12_(c)icking-archive.mid", 2) == "invention"
    assert _infer_form_from_source("data/midi/all/bach/kunstderfuge__sinfonias_bwv-796_(c)galimberti.mid", 3) == "sinfonia"


def test_infer_form_from_source_treats_classical_sinfonia_as_orchestral_reduction() -> None:
    assert (
        _infer_form_from_source(
            "data/midi/all/mozart/kunstderfuge__sinfonia_40_550_1_hisamori.mid",
            3,
        )
        == "orchestral_reduction"
    )


def test_infer_form_from_source_reserves_chorale_for_strict_chorale_sources() -> None:
    assert (
        _infer_form_from_source(
            "data/midi/all/bach/kunstderfuge__chorales_013701b_(c)greentree.mid",
            4,
        )
        == "chorale"
    )
    assert (
        _infer_form_from_source(
            "data/midi/all/buxtehude/kunstderfuge__chorale-preludes_buxwv-188_(mccloskey).mid",
            4,
        )
        == "keyboard_piece"
    )


def test_build_conditioning_audit_rows_includes_reference_metadata() -> None:
    rows = build_conditioning_audit_rows(
        forms={"fugue"},
        groups={"art_of_fugue"},
        max_per_group=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["reference_kind"] == "score"
    assert row["reference_path"].endswith(".krn")
    assert row["reference_time_signature"]
    assert row["path_form"] == "fugue"
    assert row["mode_family"]
    assert row["mode_system"]
    assert "cadence_density_per_100_bars" in row
    assert row["chromaticism_reference_system"] in {"tonal", "modal"}


def test_load_reference_metadata_prefers_catalog_key_for_bach_invention_midi() -> None:
    path = Path(
        "data/midi/all/bach/kunstderfuge__bach-js_two-part_inventions_2_(c)icking-archive.mid"
    ).resolve()
    reference = _load_reference_metadata(path, "bach")
    assert reference is not None
    assert reference.kind == "catalog"
    assert reference.key_name == "C minor"


def test_load_reference_metadata_fills_catalog_key_for_bach_sinfonia_midi() -> None:
    path = Path(
        "data/midi/all/bach/kunstderfuge__sinfonias_bwv-796_(c)galimberti.mid"
    ).resolve()
    reference = _load_reference_metadata(path, "bach")
    assert reference is not None
    assert reference.kind == "catalog"
    assert reference.key_name == "G major"


def test_load_reference_metadata_uses_score_heuristic_for_ballade52() -> None:
    path = Path(
        "data/midi/all/chopin/kernscores__ballade52.krn.fromscore.mid"
    ).resolve()
    reference = _load_reference_metadata(path, "romantic")
    assert reference is not None
    assert reference.kind == "score"
    assert reference.key_name == "F minor"


def test_summarize_conditioning_audit_reports_extended_rates() -> None:
    summary = summarize_conditioning_audit(
        [
            {
                "group": "art_of_fugue",
                "form": "fugue",
                "style": "bach",
                "meter": "4_4",
                "length_bucket": "medium",
                "texture": "polyphonic",
                "imitation": "high",
                "harmonic_rhythm": "moderate",
                "harmonic_tension": "moderate",
                "chromaticism": "moderate",
                "cadence_count": 5,
                "cadence_density_per_100_bars": 8.0,
                "subject_entry_count": 3,
                "subject_entry_density_per_10_bars": 2.5,
                "texture_polyphony_score": 0.7,
                "texture_sync_ratio": 0.4,
                "texture_active_overlap": 0.9,
                "imitation_match_density": 0.5,
                "harmonic_rhythm_score": 3.4,
                "harmonic_tension_score": 0.18,
                "chromaticism_ratio": 0.1,
                "chromaticism_reference_system": "tonal",
                "mode_family": "aeolian",
                "mode_system": "tonal",
                "mode_confidence": 0.75,
                "path_form_match": 1,
                "reference_score_form_match": 1,
                "meter_match": 1,
                "key_match_strict": 1,
                "key_match_mode": 1,
                "reference_style_match": 1,
            }
        ]
    )
    fugue = summary["forms"]["fugue"]
    assert fugue["form_path_match_rate"] == 1.0
    assert fugue["form_score_match_rate"] == 1.0
    assert fugue["meter_match_rate"] == 1.0
    assert fugue["key_strict_match_rate"] == 1.0
    assert summary["groups"]["art_of_fugue"]["cadence_density_per_100_bars"]["mean"] == 8.0


def test_non_bach_spot_manifest_passes_group_expectations() -> None:
    manifest = Path("data/benchmarks/non_bach_conditioning_spot.json").resolve()
    rows = build_conditioning_audit_rows(manifest_path=manifest)
    summary = summarize_conditioning_audit(rows, manifest_path=manifest)

    assert not summary["groups"]["classical_quartets"]["violations"]
    assert not summary["groups"]["corelli_trio_sonatas"]["violations"]
    assert not summary["groups"]["renaissance_motets"]["violations"]
    assert summary["groups"]["classical_quartets"]["cadence_count"]["mean"] < 25.0
    assert summary["groups"]["corelli_trio_sonatas"]["key_strict_match_rate"] >= 0.9
    assert summary["groups"]["renaissance_motets"]["mode_system"]["modal"] >= 3


def test_baroque_prepare_data_manifest_passes_key_expectations() -> None:
    manifest = Path("data/benchmarks/baroque_tonal_non_bach_prepare_data.json").resolve()
    rows = build_conditioning_audit_rows(
        manifest_path=manifest,
        max_per_group=12,
        pipeline="prepare_data",
    )
    assert rows
    assert all(row["analysis_pipeline"] == "prepare_data" for row in rows)

    summary = summarize_conditioning_audit(rows, manifest_path=manifest)

    assert not summary["groups"]["corelli_score_backed"]["violations"]
    assert not summary["groups"]["vivaldi_score_backed"]["violations"]
    assert not summary["groups"]["frescobaldi_score_backed"]["violations"]
    assert not summary["groups"]["buxtehude_score_backed"]["violations"]


def test_romantic_modern_prepare_data_manifest_passes_key_expectations() -> None:
    manifest = Path("data/benchmarks/romantic_modern_prepare_data.json").resolve()
    rows = build_conditioning_audit_rows(
        manifest_path=manifest,
        max_per_group=12,
        pipeline="prepare_data",
    )
    assert rows
    assert all(row["analysis_pipeline"] == "prepare_data" for row in rows)

    summary = summarize_conditioning_audit(rows, manifest_path=manifest)

    assert not summary["groups"]["chopin_score_backed"]["violations"]
    assert not summary["groups"]["brahms_score_backed"]["violations"]
    assert not summary["groups"]["schumann_score_backed"]["violations"]
    assert not summary["groups"]["grieg_score_backed"]["violations"]
    assert not summary["groups"]["joplin_score_backed"]["violations"]
    assert not summary["groups"]["ives_score_backed"]["violations"]
    assert not summary["groups"]["sousa_score_backed"]["violations"]


def test_raw_midi_explicit_key_manifest_passes_key_expectations() -> None:
    manifest = Path("data/benchmarks/raw_midi_explicit_key_spot.json").resolve()
    rows = build_conditioning_audit_rows(
        manifest_path=manifest,
        pipeline="midi_eval",
    )
    assert rows
    assert all(row["analysis_pipeline"] == "midi_eval" for row in rows)

    summary = summarize_conditioning_audit(rows, manifest_path=manifest)

    assert not summary["groups"]["anglebert_raw_title_keys"]["violations"]
    assert not summary["groups"]["chopin_raw_title_keys"]["violations"]
    assert not summary["groups"]["grieg_raw_title_keys"]["violations"]
    assert not summary["groups"]["visee_raw_title_keys"]["violations"]
    assert not summary["groups"]["vivaldi_raw_title_keys"]["violations"]
