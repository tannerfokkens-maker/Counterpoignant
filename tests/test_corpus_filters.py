"""Tests for corpus filtering and curated composer/era defaults."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from music21 import stream

from bach_gen.cli import cli
from bach_gen.data.corpus import get_midi_files
from bach_gen.utils.constants import DEFAULT_PREPARE_COMPOSER_FILTER, DIR_TO_STYLE


def _two_part_score():
    score = stream.Score()
    score.insert(0, stream.Part())
    score.insert(0, stream.Part())
    return score


def test_prepare_data_uses_default_curated_era_filter(tmp_path: Path):
    runner = CliRunner()
    with patch("bach_gen.data.corpus.get_all_works") as mock_get_all_works:
        mock_get_all_works.return_value = []  # command exits early after load step
        result = runner.invoke(
            cli,
            ["prepare-data", "--mode", "chorale", "--data-dir", str(tmp_path)],
        )

    assert result.exit_code != 0
    assert mock_get_all_works.called
    assert mock_get_all_works.call_args.kwargs["composer_filter"] == DEFAULT_PREPARE_COMPOSER_FILTER


def test_prepare_data_composer_filter_all_disables_filter(tmp_path: Path):
    runner = CliRunner()
    with patch("bach_gen.data.corpus.get_all_works") as mock_get_all_works:
        mock_get_all_works.return_value = []  # command exits early after load step
        result = runner.invoke(
            cli,
            [
                "prepare-data",
                "--mode",
                "chorale",
                "--composer-filter",
                "all",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code != 0
    assert mock_get_all_works.called
    assert mock_get_all_works.call_args.kwargs["composer_filter"] is None


def test_get_midi_files_prefers_curated_kunstderfuge_bucket(tmp_path: Path):
    midi_root = tmp_path / "midi"
    raw_kdf = midi_root / "kunstderfuge" / "bach" / "raw.mid"
    curated_kdf = (
        midi_root / "kunstderfuge" / "_voice_buckets" / "dataset_2to4" / "bach" / "curated.mid"
    )
    kern_bach = midi_root / "kernscores" / "bach" / "kernscores.mid"

    for p in [raw_kdf, curated_kdf, kern_bach]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    with patch("music21.converter.parse", side_effect=lambda *_: _two_part_score()):
        works = get_midi_files(midi_root)

    descs = {desc for desc, _, _ in works}
    assert "kunstderfuge/bach/raw" not in descs
    assert "kunstderfuge/_voice_buckets/dataset_2to4/bach/curated" in descs
    assert "kernscores/bach/kernscores" in descs


def test_composer_era_mapping_matches_project_policy():
    # Explicitly include Mendelssohn in the classical bucket for this project.
    assert DIR_TO_STYLE["mendelssohn"] == "classical"
    # Explicitly keep Brahms out of the default classical bucket.
    assert DIR_TO_STYLE["brahms"] != "classical"
