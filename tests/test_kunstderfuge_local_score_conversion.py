"""Tests for local score->MIDI conversion in download_kunstderfuge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_download_module():
    mod_path = Path(__file__).resolve().parents[1] / "scripts" / "download_kunstderfuge.py"
    spec = importlib.util.spec_from_file_location("download_kunstderfuge", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_convert_score_to_midi_mscx_without_musescore_reports_clear_error(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_download_module()
    src = tmp_path / "piece.mscx"
    dest = tmp_path / "piece.mscx.fromscore.mid"
    src.write_text("<museScore/>", encoding="utf-8")

    monkeypatch.setattr(mod, "_find_musescore_binary", lambda: None)

    ok, reason = mod._convert_score_to_midi(src, dest)
    assert ok is False
    assert reason is not None
    assert "MuseScore CLI not found" in reason
    assert not dest.exists()


def test_convert_score_to_midi_mscx_uses_musescore_cli(tmp_path: Path, monkeypatch) -> None:
    mod = _load_download_module()
    src = tmp_path / "quartet.mscx"
    dest = tmp_path / "quartet.mscx.fromscore.mid"
    src.write_text("<museScore/>", encoding="utf-8")

    monkeypatch.setattr(mod, "_find_musescore_binary", lambda: "/fake/mscore")

    calls: list[list[str]] = []

    def _fake_run(cmd, capture_output, text, timeout, check):  # noqa: ANN001
        assert capture_output is True
        assert text is True
        assert check is False
        assert timeout >= 180
        assert cmd[0] == "/fake/mscore"
        assert cmd[1] == "-o"
        out_path = Path(cmd[2])
        in_path = Path(cmd[3])
        assert in_path.suffix == ".mscx"
        out_path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk\x00\x00\x00\x04\x00\xff/\x00")
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    ok, reason = mod._convert_score_to_midi(src, dest)
    assert ok is True
    assert reason is None
    assert dest.exists()
    assert dest.stat().st_size > 0
    assert len(calls) == 1


def test_iter_dataset_source_files_skips_mscx_when_musicxml_sibling_exists(tmp_path: Path) -> None:
    mod = _load_download_module()
    composer_dir = tmp_path / "composer"
    composer_dir.mkdir()

    mscx = composer_dir / "piece.mscx"
    xml = composer_dir / "piece.musicxml"
    mscx.write_text("<museScore/>", encoding="utf-8")
    xml.write_text("<score-partwise/>", encoding="utf-8")

    files = mod._iter_dataset_source_files(tmp_path)
    names = sorted(str(p.relative_to(tmp_path)) for p in files)
    assert "composer/piece.musicxml" in names
    assert "composer/piece.mscx" not in names
