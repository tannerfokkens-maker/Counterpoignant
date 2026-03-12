#!/usr/bin/env python3
"""Audit conditioning-label quality on the Bach gold benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from bach_gen.data.label_audit import (
    DEFAULT_MANIFEST,
    build_conditioning_audit_rows,
    render_conditioning_audit_summary,
    summarize_conditioning_audit,
    write_conditioning_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit conditioning-label quality by form.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Benchmark manifest JSON.")
    parser.add_argument("--out-csv", default="output/conditioning_audit.csv", help="Per-piece CSV output.")
    parser.add_argument("--out-json", default="output/conditioning_audit_summary.json", help="Summary JSON output.")
    parser.add_argument(
        "--form",
        action="append",
        choices=[
            "fugue",
            "invention",
            "sinfonia",
            "chorale",
            "motet",
            "trio_sonata",
            "quartet",
            "keyboard_piece",
            "chamber_piece",
            "orchestral_reduction",
            "vocal_polyphony",
        ],
        help="Restrict to one or more forms.",
    )
    parser.add_argument("--group", action="append", help="Restrict to one or more manifest groups.")
    parser.add_argument("--max-per-group", type=int, default=None, help="Optional cap per group for quick runs.")
    parser.add_argument("--thresholds", default=None, help="Optional threshold JSON override.")
    parser.add_argument(
        "--pipeline",
        choices=["midi_eval", "prepare_data"],
        default="midi_eval",
        help="Audit the standalone MIDI-eval path or the local prepare-data extraction path.",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if the Bach-form prior checks fail.",
    )
    args = parser.parse_args()

    rows = build_conditioning_audit_rows(
        manifest_path=Path(args.manifest),
        forms=set(args.form) if args.form else None,
        groups=set(args.group) if args.group else None,
        max_per_group=args.max_per_group,
        thresholds_path=Path(args.thresholds) if args.thresholds else None,
        pipeline=args.pipeline,
    )
    summary = summarize_conditioning_audit(rows, manifest_path=Path(args.manifest))
    write_conditioning_audit(
        rows,
        summary,
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
    )
    print(render_conditioning_audit_summary(summary))

    if args.fail_on_violations and summary.get("violations"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
