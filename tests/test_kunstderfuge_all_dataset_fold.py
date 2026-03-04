"""Tests for dataset-wide fold into data/midi/all."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_download_module():
    mod_path = Path(__file__).resolve().parents[1] / "scripts" / "download_kunstderfuge.py"
    spec = importlib.util.spec_from_file_location("download_kunstderfuge", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fold_prefers_best_clean_variant(tmp_path: Path) -> None:
    mod = _load_download_module()
    data_root = tmp_path / "midi"
    dataset = data_root / "dataset_a" / "bach"
    dataset.mkdir(parents=True)

    raw = dataset / "piece.mid"
    reduced = dataset / "piece.reduced.mid"
    raw.write_bytes(b"raw")
    reduced.write_bytes(b"reduced")

    triage = {
        "bach/piece.mid": {"status": "needs_voice_reduce"},
        "bach/piece.reduced.mid": {"status": "clean"},
    }
    out_dir = data_root / "all"
    stats = mod.fold_datasets_into_all(
        dataset_triage_reports={"dataset_a": triage},
        data_root=data_root,
        output_dir=out_dir,
    )

    folded = out_dir / "bach" / "dataset_a__piece.mid"
    assert folded.exists()
    assert folded.read_bytes() == b"reduced"
    assert stats["written"] == 1


def test_fold_keeps_composer_and_avoids_dataset_name_collisions(tmp_path: Path) -> None:
    mod = _load_download_module()
    data_root = tmp_path / "midi"

    a_path = data_root / "dataset_a" / "bach"
    b_path = data_root / "dataset_b" / "bach"
    a_path.mkdir(parents=True)
    b_path.mkdir(parents=True)

    (a_path / "same_name.mid").write_bytes(b"a")
    (b_path / "same_name.mid").write_bytes(b"b")

    triage_a = {"bach/same_name.mid": {"status": "clean"}}
    triage_b = {"bach/same_name.mid": {"status": "clean"}}

    out_dir = data_root / "all"
    mod.fold_datasets_into_all(
        dataset_triage_reports={"dataset_a": triage_a, "dataset_b": triage_b},
        data_root=data_root,
        output_dir=out_dir,
    )

    folded_a = out_dir / "bach" / "dataset_a__same_name.mid"
    folded_b = out_dir / "bach" / "dataset_b__same_name.mid"
    assert folded_a.exists()
    assert folded_b.exists()
    assert folded_a.read_bytes() == b"a"
    assert folded_b.read_bytes() == b"b"
