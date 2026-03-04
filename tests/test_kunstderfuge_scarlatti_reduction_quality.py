"""Integration quality checks for Scarlatti separation + reduction pipeline."""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
from pathlib import Path

import pytest


def _load_download_module():
    mod_path = Path(__file__).resolve().parents[1] / "scripts" / "download_kunstderfuge.py"
    spec = importlib.util.spec_from_file_location("download_kunstderfuge", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pitch_extrema(mod, midi_path: Path) -> tuple[int | None, int | None]:
    score = mod._parse_with_timeout(midi_path, timeout=60)
    pitches: list[int] = []
    for part in score.parts:
        for n in part.recurse().notes:
            if hasattr(n, "pitch"):
                pitches.append(int(n.pitch.midi))
            else:
                for p in n.pitches:
                    pitches.append(int(p.midi))
    if not pitches:
        return None, None
    return min(pitches), max(pitches)


def test_scarlatti_separation_reduction_quality(tmp_path: Path) -> None:
    """Evaluate separation+reduction quality on Scarlatti sonatas.

    The test validates:
      - separation succeeds reliably on raw sonatas
      - final output (reduced or separated) has <= 4 note-bearing tracks
      - final output is rarely marked bad by triage
      - global low/high pitch extremes are preserved within tolerance
    """
    mod = _load_download_module()

    scarlatti_dir = Path(
        os.environ.get("KDF_SCARLATTI_DIR", "data/midi/kunstderfuge/scarlatti")
    ).resolve()
    scan_limit = int(os.environ.get("KDF_SCARLATTI_SCAN_LIMIT", "555"))
    max_candidates = int(os.environ.get("KDF_SCARLATTI_MAX_CANDIDATES", "40"))

    midi_files = sorted(scarlatti_dir.glob("*.mid"))[:scan_limit]
    if not midi_files:
        pytest.skip(f"No Scarlatti MIDIs found in {scarlatti_dir}")

    candidates = midi_files[:max_candidates]
    if not candidates:
        pytest.skip("No Scarlatti files selected for quality evaluation")

    sep_ok = 0
    final_tracks_ok = 0
    final_not_bad = 0
    low_diffs: list[float] = []
    high_diffs: list[float] = []

    for src in candidates:
        sep = tmp_path / src.with_suffix(".separated.mid").name
        red = tmp_path / src.with_suffix(".reduced.mid").name

        ok, reason = mod.voice_separate_midi(
            src=src,
            dest=sep,
            max_voices=16,
            mode="fixed",
            jitter_ratio=0.025,
            on_cap="raise-cap",
            ignore_pedal=True,
        )
        if not ok or not sep.exists():
            continue
        sep_ok += 1

        did_reduce = mod.reduce_voices(sep, red, max_voices=4)
        final_path = red if did_reduce and red.exists() else sep

        final_triage = mod.triage_midi(final_path)
        tracks_with_notes = int(
            final_triage.get("tracks_with_notes", final_triage.get("num_tracks", 0))
        )
        if tracks_with_notes <= 4:
            final_tracks_ok += 1

        if final_triage.get("status") != "bad":
            final_not_bad += 1

        src_low, src_high = _pitch_extrema(mod, src)
        out_low, out_high = _pitch_extrema(mod, final_path)
        if None not in (src_low, src_high, out_low, out_high):
            low_diffs.append(abs(float(src_low) - float(out_low)))
            high_diffs.append(abs(float(src_high) - float(out_high)))

    sep_success_rate = sep_ok / len(candidates)
    tracks_ok_rate = final_tracks_ok / max(1, sep_ok)
    final_not_bad_rate = final_not_bad / max(1, sep_ok)
    median_low_diff = statistics.median(low_diffs) if low_diffs else 999.0
    median_high_diff = statistics.median(high_diffs) if high_diffs else 999.0

    assert sep_success_rate >= 0.9, (
        f"Separation success rate too low: {sep_success_rate:.3f} "
        f"({sep_ok}/{len(candidates)})"
    )
    assert tracks_ok_rate == 1.0, (
        f"Final outputs exceed 4 voices for some files: {tracks_ok_rate:.3f}"
    )
    assert final_not_bad_rate >= 0.95, (
        f"Final outputs marked bad too often: {final_not_bad_rate:.3f}"
    )
    assert median_low_diff <= 2.0, (
        f"Low pitch extreme drift too large: median={median_low_diff:.2f} semitones"
    )
    assert median_high_diff <= 2.0, (
        f"High pitch extreme drift too large: median={median_high_diff:.2f} semitones"
    )
