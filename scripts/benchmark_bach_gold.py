#!/usr/bin/env python3
"""Benchmark Bach gold references by form with optional control variants."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from bach_gen.data.extraction import VoiceComposition
from bach_gen.data.tokenizer import BachTokenizer, load_tokenizer
from bach_gen.evaluation.information import load_information_calibration
from bach_gen.evaluation.scorer import score_composition
from bach_gen.evaluation.statistical import load_corpus_stats
from bach_gen.utils.midi_io import load_midi, midi_to_note_events
from bach_gen.utils.music_theory import detect_key, pc_to_note_name

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data/benchmarks/bach_gold.json"
DEFAULT_OUT = REPO_ROOT / "output/bach_gold_benchmark.csv"


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _expand_group_files(group: dict) -> list[Path]:
    files: set[Path] = set()
    for pattern in group.get("patterns", []):
        files.update(REPO_ROOT.glob(pattern))
    for explicit in group.get("files", []):
        files.add(REPO_ROOT / explicit)
    resolved = sorted((p.resolve() for p in files if p.exists()), key=_natural_key)
    expected = group.get("expected_count")
    if expected is not None and len(resolved) != expected:
        raise ValueError(
            f"group {group['name']} expected {expected} files, found {len(resolved)}"
        )
    return resolved


def _load_comp(path: Path) -> tuple[VoiceComposition, str]:
    tracks = midi_to_note_events(load_midi(path))
    voices = [v for v in tracks if v]

    if len(voices) < 2 and len(tracks) == 1 and tracks[0]:
        all_notes = tracks[0]
        median_pitch = np.median([n[2] for n in all_notes])
        upper = [(s, d, p) for s, d, p in all_notes if p >= median_pitch]
        lower = [(s, d, p) for s, d, p in all_notes if p < median_pitch]
        voices = [upper, lower]

    if len(voices) < 2:
        raise ValueError("Need at least two non-empty voices")

    pc_counts = np.zeros(12)
    for voice in voices:
        for _, _, pitch in voice:
            pc_counts[pitch % 12] += 1
    key_root, key_mode, _ = detect_key(pc_counts)
    key_name = f"{pc_to_note_name(key_root)} {key_mode}"
    return VoiceComposition(
        voices=voices,
        key_root=key_root,
        key_mode=key_mode,
        source=str(path),
    ), key_name


def _shuffle_comp(comp: VoiceComposition, rng: random.Random) -> VoiceComposition:
    voices = []
    for voice in comp.voices:
        if not voice:
            voices.append([])
            continue
        pitches = [n[2] for n in voice]
        rng.shuffle(pitches)
        voices.append([(n[0], n[1], pitch) for n, pitch in zip(voice, pitches)])
    return VoiceComposition(
        voices=voices,
        key_root=comp.key_root,
        key_mode=comp.key_mode,
        source=f"{comp.source}::shuffled",
    )


def _random_comp(comp: VoiceComposition, rng: random.Random) -> VoiceComposition:
    voices = []
    for voice in comp.voices:
        if not voice:
            voices.append([])
            continue
        pitches = [n[2] for n in voice]
        lo, hi = min(pitches), max(pitches)
        voices.append([(n[0], n[1], rng.randint(lo, hi)) for n in voice])
    return VoiceComposition(
        voices=voices,
        key_root=comp.key_root,
        key_mode=comp.key_mode,
        source=f"{comp.source}::random",
    )


def _repetitive_comp(comp: VoiceComposition) -> VoiceComposition:
    base_pitches = [60, 55, 48, 43]
    voices = []
    for idx, voice in enumerate(comp.voices):
        if not voice:
            voices.append([])
            continue
        pitch = base_pitches[idx % len(base_pitches)]
        voices.append([(n[0], n[1], pitch) for n in voice])
    return VoiceComposition(
        voices=voices,
        key_root=0,
        key_mode="major",
        source=f"{comp.source}::repetitive",
    )


def _score_row(
    *,
    comp: VoiceComposition,
    key_name: str,
    form: str,
    group_name: str,
    variant: str,
    tokenizer,
) -> dict[str, object]:
    tokens = tokenizer.encode(comp, form=form)
    sb = score_composition(
        comp,
        token_sequence=tokens,
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        form=form,
    )
    structural = (sb.details or {}).get("structural", {})
    contrapuntal = (sb.details or {}).get("contrapuntal", {})
    guardrails = (sb.details or {}).get("guardrails", {})
    interactions = (sb.details or {}).get("interactions", {})

    return {
        "group": group_name,
        "form": form,
        "variant": variant,
        "file": comp.source,
        "key": key_name,
        "num_voices": comp.num_voices,
        "composite": float(sb.composite),
        "voice_leading": float(sb.voice_leading),
        "statistical": float(sb.statistical),
        "structural": float(sb.structural),
        "contrapuntal": float(sb.contrapuntal),
        "completeness": float(sb.completeness),
        "thematic_recall": float(sb.thematic_recall),
        "struct_cadence": float(structural.get("cadence", 0.0)),
        "struct_phrase_structure": float(structural.get("phrase_structure", 0.0)),
        "struct_key_consistency": float(structural.get("key_consistency", 0.0)),
        "cp_voice_independence": float(contrapuntal.get("voice_independence", 0.0)),
        "cp_onset_staggering": float(contrapuntal.get("onset_staggering", 0.0)),
        "cp_voice_balance": float(contrapuntal.get("voice_balance", 0.0)),
        "guardrail_flags": ",".join(guardrails.get("flags", [])),
        "interaction_flags": ",".join(interactions.get("flags", [])),
    }


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Bach gold references by form.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Benchmark manifest JSON.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path.")
    parser.add_argument(
        "--form",
        action="append",
        choices=["fugue", "invention", "sinfonia", "chorale"],
        help="Restrict to one or more forms.",
    )
    parser.add_argument("--group", action="append", help="Restrict to one or more manifest groups.")
    parser.add_argument("--max-per-group", type=int, default=None, help="Optional cap per group for quick runs.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for sampling and control variants.")
    parser.add_argument(
        "--with-controls",
        action="store_true",
        help="Also score shuffled/random/repetitive controls derived from each Bach piece.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    rng = random.Random(args.seed)

    manifest = _load_manifest(manifest_path)
    groups = manifest.get("groups", [])
    if args.form:
        allowed_forms = set(args.form)
        groups = [g for g in groups if g["form"] in allowed_forms]
    if args.group:
        allowed_groups = set(args.group)
        groups = [g for g in groups if g["name"] in allowed_groups]
    if not groups:
        raise SystemExit("No benchmark groups selected.")

    tok_path = REPO_ROOT / "data/tokenizer.json"
    tokenizer = load_tokenizer(tok_path) if tok_path.exists() else BachTokenizer()
    load_corpus_stats(REPO_ROOT / "data/corpus_stats.json")
    load_information_calibration(REPO_ROOT / "models/information_calibration.json")
    load_information_calibration(REPO_ROOT / "data/information_calibration.json")

    rows: list[dict[str, object]] = []
    for group in groups:
        files = _expand_group_files(group)
        if args.max_per_group is not None:
            files = files[:args.max_per_group]
        for path in files:
            comp, key_name = _load_comp(path)
            rows.append(
                _score_row(
                    comp=comp,
                    key_name=key_name,
                    form=group["form"],
                    group_name=group["name"],
                    variant="gold",
                    tokenizer=tokenizer,
                )
            )
            if not args.with_controls:
                continue
            rows.append(
                _score_row(
                    comp=_shuffle_comp(comp, rng),
                    key_name=key_name,
                    form=group["form"],
                    group_name=group["name"],
                    variant="shuffled",
                    tokenizer=tokenizer,
                )
            )
            rows.append(
                _score_row(
                    comp=_random_comp(comp, rng),
                    key_name=key_name,
                    form=group["form"],
                    group_name=group["name"],
                    variant="random",
                    tokenizer=tokenizer,
                )
            )
            rows.append(
                _score_row(
                    comp=_repetitive_comp(comp),
                    key_name="C major",
                    form=group["form"],
                    group_name=group["name"],
                    variant="repetitive",
                    tokenizer=tokenizer,
                )
            )

    if not rows:
        raise SystemExit("No benchmark rows produced.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print()
    print("Summary by form / variant")
    summary: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        summary[(str(row["form"]), str(row["variant"]))].append(float(row["composite"]))
    for key in sorted(summary):
        stats = _stats(summary[key])
        print(
            f"{key[0]:10s} {key[1]:10s} "
            f"n={len(summary[key]):4d} "
            f"mean={stats['mean']:.3f} "
            f"p10={stats['p10']:.3f} "
            f"median={stats['median']:.3f} "
            f"p90={stats['p90']:.3f}"
        )

    print()
    print("Summary by group / gold")
    gold_summary: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["variant"] == "gold":
            gold_summary[str(row["group"])].append(float(row["composite"]))
    for group_name in sorted(gold_summary):
        stats = _stats(gold_summary[group_name])
        print(
            f"{group_name:22s} "
            f"n={len(gold_summary[group_name]):4d} "
            f"mean={stats['mean']:.3f} "
            f"p10={stats['p10']:.3f} "
            f"median={stats['median']:.3f} "
            f"p90={stats['p90']:.3f}"
        )


if __name__ == "__main__":
    main()
