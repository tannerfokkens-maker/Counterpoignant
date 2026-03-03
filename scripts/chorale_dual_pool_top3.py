#!/usr/bin/env python3
"""Generate chorales from two decoding pools and keep merged top-K results.

Default pool settings are taken from your sweep winners:
- Pool A: temp=1.00, min_p=0.01
- Pool B: temp=0.75, min_p=0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bach_gen.data.tokenizer import load_tokenizer
from bach_gen.evaluation.information import load_information_calibration
from bach_gen.evaluation.statistical import load_corpus_stats
from bach_gen.generation.generator import GenerationResult, generate
from bach_gen.model.trainer import Trainer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _voice_balance_ratio(voices: list[list[tuple[int, int, int]]]) -> float:
    counts = [len(v) for v in voices if v]
    if not counts:
        return 0.0
    lo = min(counts)
    hi = max(counts)
    return (lo / hi) if hi > 0 else 0.0


def _chorale_selection_score(
    composite: float,
    onset_3or4_ratio: float,
    completeness: float,
) -> float:
    return composite + 0.08 * onset_3or4_ratio + 0.04 * completeness


def _csv_write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k in seen:
                continue
            seen.add(k)
            fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _key_slug(key: str) -> str:
    return (
        key.strip()
        .replace(" ", "_")
        .replace("#", "s")
        .replace("/", "-")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-pool chorale generation and merged top-K ranking.")
    parser.add_argument("--model-path", default="models_NEW/finetune_best.pt", help="Checkpoint path.")
    parser.add_argument("--tokenizer-path", default="data/tokenizer.json", help="Tokenizer JSON path.")
    parser.add_argument("--stats-path", default="data/corpus_stats.json", help="Corpus stats JSON path.")
    parser.add_argument("--data-dir", default="data", help="Data dir fallback for info calibration.")
    parser.add_argument("--key", default="D minor", help="Chorale key.")
    parser.add_argument("--style", default="bach", help="Style conditioning token.")
    parser.add_argument("--length", default=None, choices=["short", "medium", "long", "extended"], help="Length conditioning token (optional).")
    parser.add_argument("--meter", default="4_4", help="Meter conditioning token.")
    parser.add_argument("--texture", default="homophonic", help="Texture conditioning token.")
    parser.add_argument("--imitation", default="none", help="Imitation conditioning token.")
    parser.add_argument("--harmonic-rhythm", default="moderate", choices=["slow", "moderate", "fast"], help="Harmonic rhythm token.")
    parser.add_argument("--tension", default="moderate", choices=["low", "moderate", "high"], help="Harmonic tension token.")
    parser.add_argument("--chromaticism", default=None, choices=["low", "moderate", "high"], help="Chromaticism token (default: form default).")
    parser.add_argument("--voices", type=int, default=4, help="Number of chorale voices.")
    parser.add_argument("--max-length", type=int, default=2048, help="Generation token budget after the prompt.")
    parser.add_argument("--candidate-batch-size", type=int, default=2, help="Sampling batch size.")
    parser.add_argument("--pool-a-candidates", type=int, default=120, help="Candidates in pool A.")
    parser.add_argument("--pool-b-candidates", type=int, default=80, help="Candidates in pool B.")
    parser.add_argument("--pool-a-temp", type=float, default=1.00, help="Pool A temperature.")
    parser.add_argument("--pool-a-min-p", type=float, default=0.01, help="Pool A min-p.")
    parser.add_argument("--pool-b-temp", type=float, default=0.75, help="Pool B temperature.")
    parser.add_argument("--pool-b-min-p", type=float, default=0.05, help="Pool B min-p.")
    parser.add_argument("--top", type=int, default=3, help="Merged top-K to keep.")
    parser.add_argument("--rank-by", default="selection", choices=["selection", "composite"], help="Ranking metric for merged top-K.")
    parser.add_argument("--seed-base", type=int, default=1337, help="Base random seed.")
    parser.add_argument("--output-root", default="output/chorale_dual_pool_runs", help="Run output root.")
    parser.add_argument("--run-name", default=None, help="Optional run folder name.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    tok_path = Path(args.tokenizer_path)
    stats_path = Path(args.stats_path)
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not tok_path.exists():
        raise SystemExit(f"Tokenizer not found: {tok_path}")
    if args.pool_a_candidates <= 0 or args.pool_b_candidates <= 0:
        raise SystemExit("Both pool candidate counts must be > 0.")
    if args.top <= 0:
        raise SystemExit("--top must be > 0.")

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{run_ts}_{_key_slug(args.key)}"
    run_dir = output_root / run_name
    pool_a_dir = run_dir / "pool_a"
    pool_b_dir = run_dir / "pool_b"
    merged_dir = run_dir / "merged_top"
    for p in [pool_a_dir, pool_b_dir, merged_dir]:
        p.mkdir(parents=True, exist_ok=True)

    config = {
        "model_path": str(model_path),
        "key": args.key,
        "style": args.style,
        "length": args.length,
        "meter": args.meter,
        "texture": args.texture,
        "imitation": args.imitation,
        "harmonic_rhythm": args.harmonic_rhythm,
        "tension": args.tension,
        "chromaticism": args.chromaticism,
        "voices": args.voices,
        "max_length": args.max_length,
        "candidate_batch_size": args.candidate_batch_size,
        "pool_a_candidates": args.pool_a_candidates,
        "pool_b_candidates": args.pool_b_candidates,
        "pool_a_temp": args.pool_a_temp,
        "pool_a_min_p": args.pool_a_min_p,
        "pool_b_temp": args.pool_b_temp,
        "pool_b_min_p": args.pool_b_min_p,
        "top": args.top,
        "rank_by": args.rank_by,
        "seed_base": args.seed_base,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print(f"Run dir: {run_dir}")
    print(f"Loading model: {model_path}")
    model, _ = Trainer.load_checkpoint(model_path)
    tokenizer = load_tokenizer(tok_path)
    if args.max_length <= 0:
        raise SystemExit("--max-length must be > 0.")
    print(f"Length budget (generated tokens): {args.max_length}")
    if args.length:
        print(f"Length conditioning token: {args.length}")

    if stats_path.exists():
        load_corpus_stats(stats_path)
    info_cal = (
        load_information_calibration(model_path.parent / "information_calibration.json")
        or load_information_calibration(data_dir / "information_calibration.json")
    )
    if info_cal:
        print(
            f"Loaded info calibration: ppl={info_cal['perplexity_range']} "
            f"ent={info_cal['entropy_range']}",
        )

    pools = [
        {
            "name": "pool_a",
            "candidates": args.pool_a_candidates,
            "temperature": args.pool_a_temp,
            "min_p": args.pool_a_min_p,
            "seed": args.seed_base + 1,
            "output_dir": pool_a_dir,
        },
        {
            "name": "pool_b",
            "candidates": args.pool_b_candidates,
            "temperature": args.pool_b_temp,
            "min_p": args.pool_b_min_p,
            "seed": args.seed_base + 2,
            "output_dir": pool_b_dir,
        },
    ]

    merged_rows: list[dict[str, object]] = []
    merged_candidates: list[dict[str, object]] = []

    for pool in pools:
        _set_seed(int(pool["seed"]))
        print(
            f"[{pool['name']}] candidates={pool['candidates']} "
            f"temp={pool['temperature']:.3f} min_p={pool['min_p']:.3f} seed={pool['seed']}",
        )

        results = generate(
            model=model,
            tokenizer=tokenizer,
            key_str=args.key,
            num_candidates=int(pool["candidates"]),
            top_k_results=int(pool["candidates"]),  # keep all retained candidates in this pool
            temperature=float(pool["temperature"]),
            min_p=float(pool["min_p"]),
            max_length=args.max_length,
            output_dir=Path(pool["output_dir"]),
            form="chorale",
            num_voices=args.voices,
            style=args.style,
            length=args.length,
            meter=args.meter,
            texture=args.texture,
            imitation=args.imitation,
            harmonic_rhythm=args.harmonic_rhythm,
            harmonic_tension=args.tension,
            chromaticism=args.chromaticism,
            candidate_batch_size=args.candidate_batch_size,
        )
        print(f"[{pool['name']}] retained={len(results)}")

        for rank_in_pool, r in enumerate(results, start=1):
            score = r.score
            details = score.details if isinstance(score.details, dict) else {}
            cp = details.get("contrapuntal", {}) if isinstance(details, dict) else {}
            st = details.get("structural", {}) if isinstance(details, dict) else {}
            interactions = details.get("interactions", {}) if isinstance(details, dict) else {}
            guard = details.get("guardrails", {}) if isinstance(details, dict) else {}

            onset = _onset_stats(r.composition.voices)
            vb_ratio = _voice_balance_ratio(r.composition.voices)
            selection = _chorale_selection_score(
                composite=float(score.composite),
                onset_3or4_ratio=float(onset.get("onset_3or4_ratio", 0.0)),
                completeness=float(score.completeness),
            )

            row = {
                "pool": pool["name"],
                "rank_in_pool": rank_in_pool,
                "temperature": float(pool["temperature"]),
                "min_p": float(pool["min_p"]),
                "seed": int(pool["seed"]),
                "midi_path": r.midi_path or "",
                "composite": float(score.composite),
                "selection_score": float(selection),
                "voice_leading": float(score.voice_leading),
                "statistical": float(score.statistical),
                "structural": float(score.structural),
                "contrapuntal": float(score.contrapuntal),
                "completeness": float(score.completeness),
                "thematic_recall": float(score.thematic_recall),
                "cp_onset_staggering": float(cp.get("onset_staggering", 0.0)),
                "cp_voice_independence": float(cp.get("voice_independence", 0.0)),
                "cp_contrary_at_cadences": float(cp.get("contrary_at_cadences", 0.0)),
                "cp_melodic_coherence": float(cp.get("melodic_coherence", 0.0)),
                "cp_voice_balance": float(cp.get("voice_balance", 0.0)),
                "struct_cadence": float(st.get("cadence", 0.0)),
                "struct_phrase_structure": float(st.get("phrase_structure", 0.0)),
                "struct_key_consistency": float(st.get("key_consistency", 0.0)),
                "interaction_delta": float(interactions.get("delta", 0.0)) if isinstance(interactions, dict) else 0.0,
                "interaction_flags": "|".join(interactions.get("flags", [])) if isinstance(interactions, dict) else "",
                "guardrail_multiplier": float(guard.get("multiplier", 1.0)) if isinstance(guard, dict) else 1.0,
                "guardrail_flags": "|".join(guard.get("flags", [])) if isinstance(guard, dict) else "",
                "voice_balance_ratio": float(vb_ratio),
                **onset,
            }
            merged_rows.append(row)
            merged_candidates.append({"row": row, "result": r})

    if not merged_candidates:
        raise SystemExit("No candidates retained across both pools.")

    rank_key = "selection_score" if args.rank_by == "selection" else "composite"
    merged_candidates.sort(key=lambda x: float(x["row"][rank_key]), reverse=True)
    top_n = merged_candidates[:args.top]

    top_rows: list[dict[str, object]] = []
    for i, item in enumerate(top_n, start=1):
        row = dict(item["row"])
        src = Path(str(row["midi_path"])) if row.get("midi_path") else None
        copied_path = ""
        if src and src.exists():
            dst_name = (
                f"top{i}_{row['pool']}_t{float(row['temperature']):.3f}"
                f"_mp{float(row['min_p']):.3f}_{src.name}"
            )
            dst = merged_dir / dst_name
            dst.write_bytes(src.read_bytes())
            copied_path = str(dst)
        row["merged_rank"] = i
        row["merged_path"] = copied_path
        top_rows.append(row)

    _csv_write(run_dir / "all_candidates.csv", merged_rows)
    _csv_write(run_dir / "top_candidates.csv", top_rows)

    print("\nMerged top candidates:")
    for row in top_rows:
        print(
            f"  rank={row['merged_rank']} pool={row['pool']} "
            f"t={float(row['temperature']):.3f} mp={float(row['min_p']):.3f} "
            f"sel={float(row['selection_score']):.4f} comp={float(row['composite']):.4f}",
        )
    print(f"\nWrote: {run_dir / 'all_candidates.csv'}")
    print(f"Wrote: {run_dir / 'top_candidates.csv'}")
    print(f"Merged MIDI dir: {merged_dir}")


if __name__ == "__main__":
    main()
