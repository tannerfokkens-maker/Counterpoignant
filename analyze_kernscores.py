#!/usr/bin/env python3
import argparse
import collections
import statistics
import sys
from pathlib import Path

from music21 import converter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bach_gen.data.analysis import analyze_composition
from bach_gen.data.extraction import VoiceComposition
from bach_gen.utils.constants import TICKS_PER_QUARTER


def _compute_imitation_score(voices: list[list[tuple[int, int, int]]]) -> float:
    ngram_len = 6
    min_offset = TICKS_PER_QUARTER
    voice_ngram_times: list[dict[tuple[int, ...], list[int]]] = []
    for voice in voices:
        sorted_notes = sorted(voice, key=lambda n: n[0])
        ngram_time_map: dict[tuple[int, ...], list[int]] = {}
        for i in range(len(sorted_notes) - ngram_len):
            ng = tuple(
                sorted_notes[k + 1][2] - sorted_notes[k][2]
                for k in range(i, i + ngram_len)
            )
            ngram_time_map.setdefault(ng, []).append(sorted_notes[i][0])
        voice_ngram_times.append(ngram_time_map)

    total_notes = sum(len(v) for v in voices)
    if total_notes == 0:
        return 0.0

    imitative_matches = 0
    for vi in range(len(voice_ngram_times)):
        for vj in range(vi + 1, len(voice_ngram_times)):
            map_i = voice_ngram_times[vi]
            map_j = voice_ngram_times[vj]
            if not map_i or not map_j:
                continue
            for ng_i, times_i in map_i.items():
                candidates = {ng_i}
                for pos in range(len(ng_i)):
                    for delta in (-1, 1):
                        v = list(ng_i)
                        v[pos] += delta
                        candidates.add(tuple(v))
                for ng_cand in candidates:
                    if ng_cand not in map_j:
                        continue
                    for t_i in times_i:
                        for t_j in map_j[ng_cand]:
                            if abs(t_j - t_i) >= min_offset:
                                imitative_matches += 1
    return imitative_matches / total_notes


def _compute_tension_ratio(voices: list[list[tuple[int, int, int]]]) -> float:
    dissonant_set = {1, 2, 6, 10, 11}
    dissonant_intervals = 0
    interval_count = 0

    for vi in range(len(voices)):
        for vj in range(vi + 1, len(voices)):
            notes_i = sorted(voices[vi], key=lambda n: n[0])
            notes_j = sorted(voices[vj], key=lambda n: n[0])
            attacks = sorted({n[0] for n in notes_i} | {n[0] for n in notes_j})

            def pitch_at(notes: list[tuple[int, int, int]], tick: int) -> int | None:
                for start, dur, pitch in notes:
                    if start <= tick < start + dur:
                        return pitch
                return None

            for tick in attacks:
                pi = pitch_at(notes_i, tick)
                pj = pitch_at(notes_j, tick)
                if pi is not None and pj is not None:
                    interval_count += 1
                    if abs(pi - pj) % 12 in dissonant_set:
                        dissonant_intervals += 1

    if interval_count == 0:
        return 0.0
    return dissonant_intervals / interval_count


def _compute_chromatic_ratio(
    voices: list[list[tuple[int, int, int]]], key_root: int, key_mode: str
) -> float:
    from bach_gen.utils.music_theory import get_scale

    scale_pcs = set(get_scale(key_root, key_mode))
    total_notes = 0
    chromatic_notes = 0
    for voice in voices:
        for _, _, pitch in voice:
            total_notes += 1
            if (pitch % 12) not in scale_pcs:
                chromatic_notes += 1
    if total_notes == 0:
        return 0.0
    return chromatic_notes / total_notes


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int((len(vals) - 1) * q)
    return vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=Path("/Users/tfokkens/Documents/Claude/2pt-bach/data/midi"),
    )
    parser.add_argument("--top-outliers", type=int, default=10)
    parser.add_argument("--max-voices", type=int, default=4)
    args = parser.parse_args()

    midi_dir = args.midi_dir
    counts = {
        k: collections.Counter()
        for k in [
            "texture",
            "imitation",
            "harmonic_rhythm",
            "harmonic_tension",
            "chromaticism",
        ]
    }
    composer_imitation = collections.defaultdict(collections.Counter)
    composer_totals = collections.Counter()
    composer_dim = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    crosstabs = {
        "imitation_x_texture": collections.Counter(),
        "imitation_x_harmonic_rhythm": collections.Counter(),
        "tension_x_chromaticism": collections.Counter(),
    }

    note_counts: list[int] = []
    duration_ticks: list[int] = []
    voice_count_dist = collections.Counter()
    key_mode_dist = collections.Counter()
    key_tonic_dist = collections.Counter()
    imitation_scores: list[float] = []
    tension_scores: list[float] = []
    chromatic_scores: list[float] = []
    outlier_rows: list[dict] = []

    total_files = 0
    skipped_name = 0
    parsed = 0
    too_few_parts = 0
    too_many_parts = 0
    too_few_voices = 0
    analyzed = 0
    errors = 0
    error_types = collections.Counter()
    error_examples: dict[str, list[str]] = collections.defaultdict(list)

    for krn in sorted(midi_dir.rglob("*.krn")):
        total_files += 1
        stem = krn.stem
        if any(
            x in stem
            for x in [
                "-auto",
                "-combined",
                "-beat",
                "-sampled",
                "-pan",
                "-nopan",
                "extractf",
                "-20",
                "-60",
                "-80",
                "-S",
            ]
        ):
            skipped_name += 1
            continue
        composer = krn.parent.name
        try:
            score = converter.parse(str(krn))
            parsed += 1
            parts = score.parts
            if len(parts) < 2:
                too_few_parts += 1
                continue
            if len(parts) > args.max_voices:
                too_many_parts += 1
                continue
            voices = []
            for part in parts:
                notes = [
                    (
                        int(el.offset * 480),
                        int(el.quarterLength * 480),
                        el.pitch.midi if hasattr(el, "pitch") else el.pitches[0].midi,
                    )
                    for el in part.flatten().notes
                    if int(el.quarterLength * 480) > 0
                ]
                if notes:
                    voices.append(notes)
            if len(voices) < 2:
                too_few_voices += 1
                continue
            key = score.analyze("key")
            comp = VoiceComposition(
                voices=voices,
                key_root=key.tonic.pitchClass,
                key_mode=key.mode,
                time_signature=(4, 4),
                source=stem,
            )
            result = analyze_composition(comp)
            for k, v in result.items():
                counts[k][v] += 1
            composer_imitation[composer][result["imitation"]] += 1
            composer_totals[composer] += 1
            for dim, label in result.items():
                composer_dim[composer][dim][label] += 1

            crosstabs["imitation_x_texture"][(result["imitation"], result["texture"])] += 1
            crosstabs["imitation_x_harmonic_rhythm"][
                (result["imitation"], result["harmonic_rhythm"])
            ] += 1
            crosstabs["tension_x_chromaticism"][
                (result["harmonic_tension"], result["chromaticism"])
            ] += 1

            notes_total = sum(len(v) for v in voices)
            piece_max_tick = max(start + dur for voice in voices for start, dur, _ in voice)
            note_counts.append(notes_total)
            duration_ticks.append(piece_max_tick)
            voice_count_dist[len(voices)] += 1
            key_mode_dist[key.mode] += 1
            key_tonic_dist[key.tonic.name] += 1

            i_score = _compute_imitation_score(voices)
            t_score = _compute_tension_ratio(voices)
            c_score = _compute_chromatic_ratio(voices, key.tonic.pitchClass, key.mode)
            imitation_scores.append(i_score)
            tension_scores.append(t_score)
            chromatic_scores.append(c_score)
            outlier_rows.append(
                {
                    "path": str(krn),
                    "composer": composer,
                    "imitation": i_score,
                    "tension": t_score,
                    "chromaticism": c_score,
                }
            )

            analyzed += 1
            if analyzed % 50 == 0:
                print(f"  {analyzed} done...", flush=True)
        except Exception as exc:
            errors += 1
            et = type(exc).__name__
            error_types[et] += 1
            if len(error_examples[et]) < 10:
                error_examples[et].append(str(krn))

    print(f"\nAnalyzed {analyzed} pieces ({errors} errors)\n")

    print("coverage:")
    print(f"  total .krn files      {total_files}")
    print(f"  skipped by name       {skipped_name}")
    print(f"  parsed                {parsed}")
    print(f"  <2 parts              {too_few_parts}")
    print(f"  >max voices ({args.max_voices})   {too_many_parts}")
    print(f"  <2 non-empty voices   {too_few_voices}")
    print(f"  analyzed              {analyzed}")
    print()

    for dim in [
        "imitation",
        "texture",
        "harmonic_rhythm",
        "harmonic_tension",
        "chromaticism",
    ]:
        ctr = counts[dim]
        total = sum(ctr.values())
        print(f"{dim}:")
        for label in sorted(ctr):
            pct = (ctr[label] / total * 100) if total else 0
            print(f"  {label:14s} {ctr[label]:4d}  ({pct:.0f}%)")
        print()

    print("voice-count distribution:")
    for k in sorted(voice_count_dist):
        print(f"  {k} voices: {voice_count_dist[k]}")
    print()

    print("key mode distribution:")
    for mode, n_mode in key_mode_dist.most_common():
        print(f"  {mode:8s} {n_mode:4d}")
    print()

    print("top key tonics:")
    for tonic, n_tonic in key_tonic_dist.most_common(12):
        print(f"  {tonic:4s} {n_tonic:4d}")
    print()

    if note_counts:
        print("piece size stats:")
        print(f"  notes per piece: median={statistics.median(note_counts):.0f}  "
              f"p10={_pct([float(n) for n in note_counts], 0.10):.0f}  "
              f"p90={_pct([float(n) for n in note_counts], 0.90):.0f}")
        print(f"  duration ticks:  median={statistics.median(duration_ticks):.0f}  "
              f"p10={_pct([float(n) for n in duration_ticks], 0.10):.0f}  "
              f"p90={_pct([float(n) for n in duration_ticks], 0.90):.0f}")
        print()

    print("raw score quantiles:")
    print(f"  imitation score:  p10={_pct(imitation_scores, 0.10):.3f} "
          f"p50={_pct(imitation_scores, 0.50):.3f} p90={_pct(imitation_scores, 0.90):.3f}")
    print(f"  tension ratio:    p10={_pct(tension_scores, 0.10):.3f} "
          f"p50={_pct(tension_scores, 0.50):.3f} p90={_pct(tension_scores, 0.90):.3f}")
    print(f"  chromatic ratio:  p10={_pct(chromatic_scores, 0.10):.3f} "
          f"p50={_pct(chromatic_scores, 0.50):.3f} p90={_pct(chromatic_scores, 0.90):.3f}")
    print()

    def print_crosstab(name: str, row_order: list[str], col_order: list[str]) -> None:
        print(name + ":")
        print("               " + "  ".join(f"{c:>10s}" for c in col_order))
        for r in row_order:
            row = "  ".join(f"{crosstabs[name][(r, c)]:10d}" for c in col_order)
            print(f"  {r:12s} {row}")
        print()

    print_crosstab(
        "imitation_x_texture",
        ["none", "low", "high"],
        ["homophonic", "mixed", "polyphonic"],
    )
    print_crosstab(
        "imitation_x_harmonic_rhythm",
        ["none", "low", "high"],
        ["slow", "moderate", "fast"],
    )
    print_crosstab(
        "tension_x_chromaticism",
        ["low", "moderate", "high"],
        ["low", "moderate", "high"],
    )

    print("imitation by composer:")
    for comp in sorted(composer_imitation):
        ctr = composer_imitation[comp]
        total = composer_totals[comp]
        row = "  ".join(
            f"{l}:{ctr[l]} ({(ctr[l] / total * 100 if total else 0):.0f}%)"
            for l in ["none", "low", "high"]
            if ctr[l]
        )
        print(f"  {comp:15s}  n={total:4d}  {row}")

    if error_types:
        print("\nerrors by exception type:")
        for et, n_err in error_types.most_common():
            print(f"  {et:20s} {n_err:4d}")
        print("\nerror examples:")
        for et, examples in error_examples.items():
            print(f"  {et}:")
            for path in examples:
                print(f"    {path}")

    top_n = max(1, args.top_outliers)
    print(f"\ntop {top_n} outliers:")
    for metric in ("imitation", "tension", "chromaticism"):
        print(f"  by {metric}:")
        for row in sorted(outlier_rows, key=lambda r: r[metric], reverse=True)[:top_n]:
            print(
                f"    {row[metric]:.3f}  {row['composer']:15s}  {row['path']}"
            )


if __name__ == "__main__":
    main()
