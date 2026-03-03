#!/usr/bin/env python3
"""Sweep decoding params (temperature/min-p) for fugue and chorale.

This script runs a grid over ``temperature`` x ``min_p`` for one or both forms
(``fugue``, ``chorale``), records run-level metrics from the top result, then
writes an aggregated summary ranked by a form-aware selection score.

Example:
    ./.venv/bin/python scripts/sweep_temp_minp.py \
      --model-path models_NEW/drope_best.pt \
      --temperatures 0.8,0.9,1.0 \
      --min-ps 0.01,0.03,0.05 \
      --repeats 2 \
      --candidates 60
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from bach_gen.data.tokenizer import load_tokenizer
from bach_gen.evaluation.information import load_information_calibration
from bach_gen.evaluation.statistical import load_corpus_stats
from bach_gen.generation.generator import GenerationResult, generate
from bach_gen.model.trainer import Trainer
from bach_gen.utils.constants import FORM_DEFAULTS

VALID_FORMS = {"fugue", "chorale"}


def _parse_float_list(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def _parse_form_list(raw: str) -> list[str]:
    vals = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in VALID_FORMS:
            raise ValueError(f"Unsupported form '{part}'. Valid: {sorted(VALID_FORMS)}")
        vals.append(part)
    if not vals:
        raise ValueError("Expected at least one form.")
    return vals


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


def _voice_balance(voices: list[list[tuple[int, int, int]]]) -> tuple[float, int, int]:
    counts = [len(v) for v in voices if v]
    if not counts:
        return 0.0, 0, 0
    lo = min(counts)
    hi = max(counts)
    ratio = (lo / hi) if hi > 0 else 0.0
    return ratio, lo, hi


def _extract_top_metrics(results: list[GenerationResult]) -> dict[str, object]:
    if not results:
        return {}

    top = results[0]
    score = top.score
    details = score.details if isinstance(score.details, dict) else {}
    cp = details.get("contrapuntal", {}) if isinstance(details, dict) else {}
    guard = details.get("guardrails", {}) if isinstance(details, dict) else {}
    onset = _onset_stats(top.composition.voices)
    vb_ratio, vb_min, vb_max = _voice_balance(top.composition.voices)

    flags = []
    if isinstance(guard, dict):
        f = guard.get("flags")
        if isinstance(f, list):
            flags = [str(x) for x in f]

    return {
        "midi_path": top.midi_path or "",
        "composite": float(score.composite),
        "voice_leading": float(score.voice_leading),
        "statistical": float(score.statistical),
        "structural": float(score.structural),
        "contrapuntal": float(score.contrapuntal),
        "completeness": float(score.completeness),
        "thematic_recall": float(score.thematic_recall),
        "cp_voice_independence": float(cp.get("voice_independence", 0.0)),
        "cp_rhythmic_complementarity": float(cp.get("rhythmic_complementarity", 0.0)),
        "cp_onset_staggering": float(cp.get("onset_staggering", 0.0)),
        "cp_voice_balance": float(cp.get("voice_balance", 0.0)),
        "guardrail_multiplier": float(guard.get("multiplier", 1.0)) if isinstance(guard, dict) else 1.0,
        "guardrail_flags": "|".join(flags),
        "voice_balance_ratio": float(vb_ratio),
        "voice_min_notes": int(vb_min),
        "voice_max_notes": int(vb_max),
        **onset,
    }


def _selection_score(form: str, row: dict[str, object]) -> float:
    composite = float(row.get("composite", 0.0))
    onset_3or4 = float(row.get("onset_3or4_ratio", 0.0))
    cp_stagger = float(row.get("cp_onset_staggering", 0.0))
    vb_ratio = float(row.get("voice_balance_ratio", 0.0))
    completeness = float(row.get("completeness", 0.0))

    if form == "fugue":
        # Favor high composite and independent entries, discourage block texture.
        return composite + 0.12 * cp_stagger + 0.05 * vb_ratio - 0.08 * onset_3or4
    # Chorale: favor high composite and stronger vertical alignment/closure.
    return composite + 0.08 * onset_3or4 + 0.04 * completeness


def _safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _safe_std(vals: list[float]) -> float:
    return float(np.std(vals)) if vals else 0.0


def _as_float(row: dict[str, object], key: str) -> float | None:
    val = row.get(key)
    if val in ("", None):
        return None
    return float(val)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep temperature/min-p for fugue and chorale.")
    parser.add_argument("--model-path", default="models_NEW/drope_best.pt", help="Checkpoint path.")
    parser.add_argument("--tokenizer-path", default="data/tokenizer.json", help="Tokenizer JSON path.")
    parser.add_argument("--stats-path", default="data/corpus_stats.json", help="Corpus stats JSON path.")
    parser.add_argument("--data-dir", default="data", help="Data dir fallback for info calibration.")
    parser.add_argument("--forms", default="fugue,chorale", help="Comma list: fugue,chorale")
    parser.add_argument("--temperatures", default="0.8,0.9,1.0", help="Comma-separated temperatures.")
    parser.add_argument("--min-ps", default="0.01,0.03,0.05", help="Comma-separated min-p values.")
    parser.add_argument("--repeats", type=int, default=2, help="Runs per grid point.")
    parser.add_argument("--candidates", type=int, default=60, help="Candidates per run.")
    parser.add_argument("--top", type=int, default=3, help="Top-k retained by generator.")
    parser.add_argument("--candidate-batch-size", type=int, default=2, help="Sampling batch size.")
    parser.add_argument("--fugue-key", default="D minor", help="Key for fugue runs.")
    parser.add_argument("--chorale-key", default="D minor", help="Key for chorale runs.")
    parser.add_argument("--style", default="baroque", help="Style conditioning.")
    parser.add_argument("--texture-fugue", default="polyphonic", help="Fugue texture token.")
    parser.add_argument("--texture-chorale", default="homophonic", help="Chorale texture token.")
    parser.add_argument("--imitation-fugue", default="high", help="Fugue imitation token.")
    parser.add_argument("--imitation-chorale", default="none", help="Chorale imitation token.")
    parser.add_argument("--tension", default=None, choices=["low", "moderate", "high"], help="Optional tension token.")
    parser.add_argument("--meter", default="4_4", help="Meter token.")
    parser.add_argument("--max-length-fugue", type=int, default=FORM_DEFAULTS["fugue"][1], help="Fugue max tokens.")
    parser.add_argument("--max-length-chorale", type=int, default=FORM_DEFAULTS["chorale"][1], help="Chorale max tokens.")
    parser.add_argument("--seed-base", type=int, default=1337, help="Base seed for deterministic sweeps.")
    parser.add_argument("--output-dir", default="output/sweep_temp_minp_midis", help="MIDI output root.")
    parser.add_argument("--out", default="output/temp_minp_sweep_runs.csv", help="Run-level CSV output.")
    parser.add_argument("--summary-out", default="output/temp_minp_sweep_summary.csv", help="Aggregate CSV output.")
    args = parser.parse_args()

    forms = _parse_form_list(args.forms)
    temps = _parse_float_list(args.temperatures)
    min_ps = _parse_float_list(args.min_ps)

    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)
    stats_path = Path(args.stats_path)
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_dir)
    out_path = Path(args.out)
    summary_path = Path(args.summary_out)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")

    print(f"Loading model: {model_path}")
    model, _ = Trainer.load_checkpoint(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

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

    combos = []
    for form in forms:
        for temp in temps:
            for min_p in min_ps:
                for rep in range(args.repeats):
                    combos.append((form, temp, min_p, rep))

    print(
        f"Starting sweep: {len(combos)} runs "
        f"({len(forms)} forms x {len(temps)} temps x {len(min_ps)} min_p x {args.repeats} repeats)"
    )

    run_rows: list[dict[str, object]] = []
    t0 = time.time()

    for idx, (form, temp, min_p, rep) in enumerate(combos, start=1):
        key = args.fugue_key if form == "fugue" else args.chorale_key
        texture = args.texture_fugue if form == "fugue" else args.texture_chorale
        imitation = args.imitation_fugue if form == "fugue" else args.imitation_chorale
        max_len = args.max_length_fugue if form == "fugue" else args.max_length_chorale
        seed = args.seed_base + idx
        _set_seed(seed)

        temp_tag = f"{temp:.3f}".replace(".", "p")
        minp_tag = f"{min_p:.3f}".replace(".", "p")
        run_dir = output_root / f"{form}_t{temp_tag}_mp{minp_tag}_r{rep+1}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{idx}/{len(combos)}] form={form} key={key} temp={temp:.3f} min_p={min_p:.3f} "
            f"repeat={rep+1}/{args.repeats} seed={seed}"
        )

        base_row: dict[str, object] = {
            "form": form,
            "key": key,
            "temperature": temp,
            "min_p": min_p,
            "repeat": rep + 1,
            "seed": seed,
            "candidates": args.candidates,
            "top_k_results": args.top,
            "max_length": max_len,
            "style": args.style,
            "texture": texture,
            "imitation": imitation,
            "tension": args.tension or "",
            "status": "ok",
            "error": "",
            "num_results": 0,
        }

        try:
            results = generate(
                model=model,
                tokenizer=tokenizer,
                key_str=key,
                num_candidates=args.candidates,
                top_k_results=args.top,
                temperature=temp,
                min_p=min_p,
                max_length=max_len,
                output_dir=run_dir,
                form=form,
                style=args.style,
                meter=args.meter,
                texture=texture,
                imitation=imitation,
                harmonic_tension=args.tension,
                candidate_batch_size=args.candidate_batch_size,
            )
            base_row["num_results"] = len(results)
            metrics = _extract_top_metrics(results)
            base_row.update(metrics)
            if metrics:
                base_row["selection_score"] = _selection_score(form, base_row)
            else:
                base_row["selection_score"] = ""
        except Exception as exc:
            base_row["status"] = "error"
            base_row["error"] = str(exc)
            base_row["selection_score"] = ""

        run_rows.append(base_row)

        # Keep memory clean on long sweeps.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    _write_csv(out_path, run_rows)
    print(f"Wrote run-level results: {out_path}")

    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = defaultdict(list)
    for r in run_rows:
        grouped[(str(r["form"]), float(r["temperature"]), float(r["min_p"]))].append(r)

    summary_rows: list[dict[str, object]] = []
    metric_keys = [
        "selection_score",
        "composite",
        "voice_leading",
        "statistical",
        "structural",
        "contrapuntal",
        "completeness",
        "thematic_recall",
        "cp_voice_independence",
        "cp_rhythmic_complementarity",
        "cp_onset_staggering",
        "cp_voice_balance",
        "onset_3or4_ratio",
        "voice_balance_ratio",
    ]

    for (form, temp, min_p), rows in grouped.items():
        success = [r for r in rows if r.get("status") == "ok" and _as_float(r, "composite") is not None]
        row: dict[str, object] = {
            "form": form,
            "temperature": temp,
            "min_p": min_p,
            "runs": len(rows),
            "success_runs": len(success),
            "success_rate": (len(success) / len(rows)) if rows else 0.0,
        }

        for k in metric_keys:
            vals = [_as_float(r, k) for r in success]
            vals = [v for v in vals if v is not None]
            row[f"mean_{k}"] = _safe_mean(vals)
            row[f"std_{k}"] = _safe_std(vals)

        summary_rows.append(row)

    summary_rows.sort(key=lambda r: (str(r["form"]), -float(r["mean_selection_score"])))
    _write_csv(summary_path, summary_rows)
    print(f"Wrote aggregate summary: {summary_path}")

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")

    print("\nTop settings by form (mean_selection_score):")
    by_form: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in summary_rows:
        by_form[str(r["form"])].append(r)
    for form in forms:
        print(f"\n{form.upper()}:")
        for rank, r in enumerate(by_form.get(form, [])[:5], start=1):
            print(
                f"  {rank}. temp={float(r['temperature']):.3f} min_p={float(r['min_p']):.3f} "
                f"score={float(r['mean_selection_score']):.4f} "
                f"composite={float(r['mean_composite']):.4f} "
                f"success={float(r['success_rate']):.2f}"
            )


if __name__ == "__main__":
    main()
