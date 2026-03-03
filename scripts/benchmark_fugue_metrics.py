#!/usr/bin/env python3
"""Benchmark fugue/chorale output quality with repeatable metrics.

Writes a CSV with scorer dimensions plus texture diagnostics so A/B scorer
changes can be compared on the same file set.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from bach_gen.data.analysis import compute_texture
from bach_gen.data.extraction import VoiceComposition
from bach_gen.data.tokenizer import BachTokenizer, load_tokenizer
from bach_gen.evaluation.scorer import score_composition
from bach_gen.evaluation.statistical import load_corpus_stats
from bach_gen.utils.midi_io import load_midi, midi_to_note_events
from bach_gen.utils.music_theory import detect_key


def _infer_mode(path: Path, n_voices: int) -> str:
    name = path.name.lower()
    if "fugue" in name:
        return "fugue"
    if "chorale" in name:
        return "chorale"
    if "sinfonia" in name:
        return "sinfonia"
    if "invention" in name or "2part" in name:
        return "2-part"
    if n_voices == 2:
        return "2-part"
    if n_voices == 3:
        return "sinfonia"
    return "chorale"


def _label(path: Path) -> str:
    n = path.name.lower()
    if "fugue" in n:
        return "fugue"
    if "chorale" in n:
        return "chorale"
    return "other"


def _onset_stats(voices: list[list[tuple[int, int, int]]]) -> dict[str, float]:
    onset_times = sorted(set(n[0] for v in voices for n in v))
    if not onset_times:
        return {
            "onset_1_ratio": 0.0,
            "onset_2_ratio": 0.0,
            "onset_3_ratio": 0.0,
            "onset_4_ratio": 0.0,
            "onset_3or4_ratio": 0.0,
        }

    counts = Counter()
    for t in onset_times:
        starts = sum(1 for v in voices if any(n[0] == t for n in v))
        counts[starts] += 1

    total = float(len(onset_times))
    return {
        "onset_1_ratio": counts.get(1, 0) / total,
        "onset_2_ratio": counts.get(2, 0) / total,
        "onset_3_ratio": counts.get(3, 0) / total,
        "onset_4_ratio": counts.get(4, 0) / total,
        "onset_3or4_ratio": (counts.get(3, 0) + counts.get(4, 0)) / total,
    }


def _voice_balance(voices: list[list[tuple[int, int, int]]]) -> tuple[float, int, int]:
    counts = [len(v) for v in voices if v]
    if not counts:
        return 0.0, 0, 0
    lo = min(counts)
    hi = max(counts)
    ratio = (lo / hi) if hi > 0 else 0.0
    return ratio, lo, hi


def _load_comp(path: Path) -> VoiceComposition:
    tracks = midi_to_note_events(load_midi(path))
    voices = [v for v in tracks if v]

    if len(voices) < 2 and len(tracks) == 1 and tracks[0]:
        # Fallback split for single-track MIDI.
        all_notes = tracks[0]
        median_pitch = np.median([n[2] for n in all_notes])
        upper = [(s, d, p) for s, d, p in all_notes if p >= median_pitch]
        lower = [(s, d, p) for s, d, p in all_notes if p < median_pitch]
        voices = [upper, lower]

    if len(voices) < 2:
        raise ValueError("Need at least two voices")

    pc_counts = np.zeros(12)
    for v in voices:
        for _, _, p in v:
            pc_counts[p % 12] += 1
    key_root, key_mode, _ = detect_key(pc_counts)
    return VoiceComposition(voices=voices, key_root=key_root, key_mode=key_mode, source=str(path))


def _collect_files(patterns: list[str], explicit: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pat in patterns:
        files.update(Path().glob(pat))
    for item in explicit:
        files.add(Path(item))
    return sorted(p for p in files if p.exists() and p.suffix.lower() == ".mid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fugue/chorale scoring metrics.")
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Glob pattern(s) for MIDI files (repeatable).",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Explicit MIDI file path(s) to include (repeatable).",
    )
    parser.add_argument(
        "--out",
        default="output/benchmark_fugue_metrics.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print grouped mean summary by label.",
    )
    args = parser.parse_args()

    tok_path = Path("data/tokenizer.json")
    tokenizer = load_tokenizer(tok_path) if tok_path.exists() else BachTokenizer()
    load_corpus_stats(Path("data/corpus_stats.json"))

    patterns = args.glob if args.glob else ["output/fugue_*.mid", "output/chorale_*.mid"]
    files = _collect_files(patterns, args.file)
    if not files:
        raise SystemExit("No MIDI files found.")

    rows: list[dict[str, object]] = []
    for path in files:
        try:
            comp = _load_comp(path)
        except Exception as exc:
            print(f"SKIP {path}: {exc}")
            continue

        mode = _infer_mode(path, comp.num_voices)
        tokens = tokenizer.encode(comp, form=mode)
        sb = score_composition(comp, token_sequence=tokens, tokenizer=tokenizer, form=mode)
        onset = _onset_stats(comp.voices)
        bal_ratio, bal_min, bal_max = _voice_balance(comp.voices)

        row = {
            "file": str(path),
            "label": _label(path),
            "mode": mode,
            "num_voices": comp.num_voices,
            "composite": sb.composite,
            "voice_leading": sb.voice_leading,
            "statistical": sb.statistical,
            "structural": sb.structural,
            "contrapuntal": sb.contrapuntal,
            "completeness": sb.completeness,
            "thematic_recall": sb.thematic_recall,
            "texture": compute_texture(comp),
            "voice_balance_ratio": bal_ratio,
            "voice_min_notes": bal_min,
            "voice_max_notes": bal_max,
            **onset,
        }

        cp_details = (sb.details or {}).get("contrapuntal", {})
        row["cp_voice_independence"] = cp_details.get("voice_independence", 0.0)
        row["cp_rhythmic_complementarity"] = cp_details.get("rhythmic_complementarity", 0.0)
        row["cp_onset_staggering"] = cp_details.get("onset_staggering", 0.0)

        struct_details = (sb.details or {}).get("structural", {})
        row["struct_cadence"] = struct_details.get("cadence", 0.0)
        row["struct_phrase_structure"] = struct_details.get("phrase_structure", 0.0)
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    if args.summary and rows:
        by_label: dict[str, list[dict[str, object]]] = {}
        for r in rows:
            by_label.setdefault(str(r["label"]), []).append(r)
        for label, group in sorted(by_label.items()):
            c = np.array([float(r["composite"]) for r in group])
            lock = np.array([float(r["onset_3or4_ratio"]) for r in group])
            stag = np.array([float(r["cp_onset_staggering"]) for r in group])
            bal = np.array([float(r["voice_balance_ratio"]) for r in group])
            print(
                f"{label:8s} n={len(group):3d} "
                f"composite={c.mean():.3f} "
                f"onset_3or4={lock.mean():.3f} "
                f"cp_stagger={stag.mean():.3f} "
                f"balance={bal.mean():.3f}",
            )


if __name__ == "__main__":
    main()
