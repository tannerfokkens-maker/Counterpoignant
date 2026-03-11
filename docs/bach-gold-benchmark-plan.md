# Bach Gold Benchmark Plan

This project needs a form-aware evaluation target where real Bach sits in the
top score band for its own form. The benchmark below uses canonical Bach MIDI
sets already present in `data/midi/all/bach` and treats them as the gold
reference distribution for tuning the scorer.

## Gold benchmark sets

- Fugue:
  - `kernscores__artfugue-*.krn.fromscore.mid` (20 files)
  - `kernscores__wtc1f*.krn.fromscore.mid` (24 files)
  - `kernscores__wtc2f*.krn.fromscore.mid` (24 files)
- Invention:
  - `kunstderfuge__bach-js_two-part_inventions_*_(c)icking-archive.mid` (15 files)
- Sinfonia:
  - `kunstderfuge__bach-js_three-part_inventions_*_(c)icking-archive.mid` (15 files)
- Chorale:
  - `kernscores__chor*.krn.fromscore.mid` (370 files)

The manifest for these sets lives at `data/benchmarks/bach_gold.json`.

## Benchmark runner

Use:

```bash
./.venv/bin/python scripts/benchmark_bach_gold.py --out output/bach_gold_baseline.csv
./.venv/bin/python scripts/benchmark_bach_gold.py --with-controls --out output/bach_gold_with_controls.csv
```

The runner:

- scores each file with its explicit form
- uses normalized MIDI timing via `midi_to_note_events()`
- can also score shuffled/random/repetitive controls derived from each Bach piece
- writes per-piece CSV plus form/group summaries

## Current baseline

Gold-only baseline (`output/bach_gold_baseline_after_fugue_patch.csv`):

- Fugue: mean `0.793`, p10 `0.717`, median `0.810`, p90 `0.845`
- Invention: mean `0.774`, p10 `0.723`, median `0.771`, p90 `0.821`
- Sinfonia: mean `0.740`, p10 `0.677`, median `0.748`, p90 `0.792`
- Chorale: mean `0.803`, p10 `0.748`, median `0.812`, p90 `0.846`

Gold vs controls (`output/bach_gold_with_controls_after_fugue_patch.csv`):

- Fugue:
  - gold median `0.810`
  - shuffled median `0.527`
  - random median `0.284`
  - repetitive median `0.200`
- Invention:
  - gold median `0.771`
  - shuffled median `0.421`
  - random median `0.264`
  - repetitive median `0.108`
- Sinfonia:
  - gold median `0.748`
  - shuffled median `0.417`
  - random median `0.230`
  - repetitive median `0.197`
- Chorale:
  - gold median `0.812`
  - shuffled median `0.654`
  - random median `0.414`
  - repetitive median `0.217`

The scorer still places real Bach lower than desired overall, but the current
benchmark now shows clean separation from degenerate controls in every form.

## Current weak spots by form

- Fugue:
  - statistical similarity and structural rhetoric are too low for real Bach
  - some WTC fugues still trigger cadence / lockstep heuristics too harshly
  - `artfugue-018a/018b` still only receive partial thematic-recall credit
- Invention:
  - phrase structure and cadence expectations are too strict for short 2-voice works
- Sinfonia:
  - voice-leading is under-scoring real 3-voice Bach most severely
- Chorale:
  - contrapuntal scoring is still too fugue-biased for a chorale texture

## Provisional acceptance targets

Do not target literal `1.0`. Target stable ranking and a clear top band.

- Fugue:
  - Bach median `>= 0.85`
  - Bach p10 `>= 0.75`
- Invention:
  - Bach median `>= 0.85`
  - Bach p10 `>= 0.75`
- Sinfonia:
  - Bach median `>= 0.82`
  - Bach p10 `>= 0.70`
- Chorale:
  - Bach median `>= 0.85`
  - Bach p10 `>= 0.75`

Ranking constraints:

- For each form, Bach p10 should remain above shuffled p90.
- Strong generated samples should not systematically outrank the Bach gold set.
- No form-specific heuristic should assume a fixed voice count unless that voice
  count is intrinsic to the form definition being evaluated.

## Next tuning order

1. Fugue:
   - tune interaction penalties
   - improve thematic recall for long subjects and delayed returns
   - revisit cadence / phrase heuristics for long works
2. Invention:
   - lower phrase/cadence penalties for compact 2-voice dialogue
3. Sinfonia:
   - inspect voice-leading penalties on 3-voice crossings and accented dissonance
4. Chorale:
   - reduce dependence on fugue-style onset staggering / independence signals
5. After heuristic changes:
   - re-run `calibrate`
   - re-run `calibrate-forms`
   - compare benchmark CSVs before/after
