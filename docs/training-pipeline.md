# Training Pipeline

End-to-end guide: data preparation, training, and generation.

## 1. Prepare Data

```bash
uv run bach-gen prepare-data
```

Current defaults:
- `--mode all`
- `--tokenizer scale-degree`
- `--max-seq-len 4096`
- `--max-source-voices 4`
- `--max-groups-per-work 1`
- `--max-pairs-per-work 2`
- `--pair-strategy adjacent+outer`
- `--sonata-policy counterpoint-safe`

**What happens:**

1. **Load works** from music21 corpus + `data/midi/`. Deduplicate by source path. By default this uses the curated era filter `bach,baroque,renaissance,classical`. Use `--composer-filter` to override, or `--composer-filter all` to disable filtering.

2. **Extract voices** per mode.
   - `--mode all` auto-detects form/voice target per piece, but extraction is now guarded by source-level caps.
   - `--mode 2-part` extracts voice pairs using configurable pairing strategy.
   - Safety defaults (recommended): `--max-source-voices 4`, `--max-groups-per-work 1`, `--max-pairs-per-work 2`, `--pair-strategy adjacent+outer`.
   - Sonata default policy: `--sonata-policy counterpoint-safe` keeps all non-accompaniment slices and at most one accompaniment-like slice per sonata/keyboard-like work (least accompaniment-heavy candidate). Use `--sonata-policy all` when explicitly training for sonata/keyboard generation.

3. **Augment** to all 12 keys by transposition (skipped for scale-degree tokenizer since it's already key-agnostic). Transpositions that push notes outside MIDI 36-84 are dropped.

4. **Analyze** each piece for Phase 2 conditioning labels:
   - **Texture** — onset synchronization ratio: `homophonic` (>0.60), `polyphonic` (<0.54), else `mixed`.
   - **Imitation** — cross-voice interval 6-gram matching with time-offset gating and ±1 tolerance: `high` (>0.30), `low` (>0.10), else `none`.
   - **Harmonic rhythm** — beat-to-beat pitch-class-set changes per measure: `slow` (≤2.77), `moderate` (2.77–3.18), `fast` (>3.18).
   - **Harmonic tension** — dissonance ratio at union-of-attacks: `high` (>0.145), `moderate` (>0.128), else `low`.
   - **Chromaticism** — fraction of non-diatonic notes: `high` (>0.15), `moderate` (>0.05), else `low`.

5. **Tokenize** each piece twice (dual encoding):
   - **Interleaved** — all voices merged chronologically with `VOICE_N` markers per note. Prefix: `BOS STYLE FORM MODE LENGTH METER TEXTURE IMITATION HARMONIC_RHYTHM HARMONIC_TENSION CHROMATICISM ENCODE_INTERLEAVED KEY <events> EOS`.
   - **Sequential** — each voice serialized separately with its own timeline. Format: `BOS <conditioning> ENCODE_SEQUENTIAL KEY VOICE_1 <notes> VOICE_SEP VOICE_2 <notes> VOICE_SEP ... VOICE_N <notes> EOS`.

   Both encodings share the same `piece_id` so train/val split keeps them together. Use `--no-sequential` to skip dual encoding if compute is tight. Sequences under 20 tokens are dropped.

6. **Chunk or drop** long sequences. By default, sequences exceeding `--max-seq-len` are split into overlapping windows (75% stride). Use `--no-chunk` to drop them instead.

7. **Save** to the output directory (default `data/`).

**Output files:**

| File | Contents |
|---|---|
| `tokenizer.json` | Serialized vocab mappings |
| `sequences.json` | All token sequences (interleaved + sequential) |
| `piece_ids.json` | Source piece ID per sequence (for leak-free splits) |
| `mode.json` | Mode metadata: mode, voice count, tokenizer type, max seq len, sequential enabled |
| `corpus_stats.json` | Pitch-class, interval, duration distributions |

**Key flags:**

- `--max-seq-len 4096` — context window ceiling (default)
- `--mode` — determines voice count and form token
- `--no-sequential` — skip dual sequential encoding (halves training data)
- `--tokenizer absolute|scale-degree` — absolute pitch (155 tokens) or key-agnostic scale degrees (121 tokens)
- `--max-source-voices` — skip works whose raw score has too many parts (default: 4)
- `--max-groups-per-work` — cap extracted N-voice groups per source work (default: 1)
- `--pair-strategy adjacent+outer|adjacent-only|all-combinations` — how 2-part pairs are derived from multi-voice works
- `--max-pairs-per-work` — cap extracted 2-part pairs per source work (default: 2)
- `--sonata-policy counterpoint-safe|all` — default keeps non-accompaniment slices plus at most one least-accompaniment slice per sonata/keyboard-like work
- `--composer-filter` — defaults to `bach,baroque,renaissance,classical`; use `all` to disable

---

## 2. Train

```bash
uv run bach-gen train --epochs 500 --lr 3e-4 --batch-size 8 --num-kv-heads 2
```

**What happens:**

1. **Load data** from `--data-dir` (default `data/`). Auto-detects mode and seq_len from saved metadata. Training data now includes both interleaved and sequential encodings of every piece (unless `--no-sequential` was used during preparation).
   - If `--seq-len` is omitted, training uses `mode.json["max_seq_len"]` when available.

2. **Split** into train/val at the piece level (90/10) using `piece_ids.json` to prevent chunk leakage. Both encoding variants of the same piece stay together in the same split.

3. **Build model** — BachTransformer (256d, 8 heads, 8 layers, SwiGLU FFN, RMSNorm, weight tying). Key options:
   - `--pos-encoding rope|pope` — choose positional encoding for main training
   - `--num-kv-heads` — grouped query attention (fewer KV heads = smaller KV cache at inference)

4. **Train** with AdamW (betas 0.9/0.98), cosine annealing to 1e-6, label smoothing 0.1, gradient clipping at 1.0. Optional `--fp16` for mixed precision on CUDA.

5. **DroPE recalibration (default, same run)** — by default, the command automatically runs DroPE after normal training with early stopping.
   - `--drope-epochs` is a max-epoch cap (not a fixed run length).
   - Tune with `--drope-lr`, `--drope-patience`, `--drope-min-delta`, `--drope-min-epochs`.
   - Use `--drope-fixed` for fixed-length DroPE, or `--no-drope` to disable.

6. **Save checkpoints** throughout:
   - `latest.pt` — after every epoch (safe to resume from)
   - `best.pt` — when validation loss improves
   - `final.pt` — end of training
   - During DroPE: `pre_drope.pt`, `drope_latest.pt`, `drope_best.pt`, `drope_final.pt`

### Lambda Setup (GPU)

Recommended workflow for Lambda instances:

1. **Environment bootstrap**
```bash
git clone <your-repo-url>
cd Counterpoignant
uv sync
```

2. **Prepare data with safe defaults**
```bash
uv run bach-gen prepare-data \
  --data-dir data
```

3. **Train on GPU**
```bash
uv run bach-gen train \
  --epochs 500 \
  --batch-size 8 \
  --lr 3e-4 \
  --pos-encoding pope \
  --num-kv-heads 2 \
  --fp16
```

4. **Resume safely after preemption**
```bash
uv run bach-gen train --resume models/latest.pt --epochs 500 --fp16
```

5. **Persist artifacts off-instance**
- Sync `data/`, `models/`, and `output/` to durable storage (S3/remote disk) between sessions.

### Curriculum Training (optional)

Pre-train on a broad corpus, then fine-tune on Bach:

```bash
uv run bach-gen train --curriculum \
  --finetune-data-dir data/bach \
  --pretrain-epochs 350 --epochs 500 --finetune-lr 1e-4
```

Phase 1 trains on the broad dataset. Phase 2 swaps to the Bach dataset with a fresh optimizer at a lower learning rate.

If your prepared dataset already contains mixed styles (including Bach), you can fine-tune from the same `sequences.json` by style token instead of a second directory:

```bash
uv run bach-gen train --curriculum \
  --finetune bach \
  --pretrain-epochs 350 --epochs 500 --finetune-lr 1e-4
```

`--finetune` accepts either:
- a style token (`bach`, `baroque`, `renaissance`, `classical`), or
- a composer/source substring (for example `beethoven`, `mozart`) matched against `piece_ids.json`.

Use either `--finetune-data-dir` or `--finetune`, not both.
DroPE is on by default with curriculum too, and runs after both phases complete.
Recommended default split: 70/30 pre-train/fine-tune (e.g. `350/500` for a 500-epoch run).

### Quick Test (<1 hour)

Use this to sanity-check the full curriculum + DroPE pipeline after code/data changes:

```bash
uv run bach-gen train \
  --curriculum \
  --finetune bach \
  --epochs 20 \
  --pretrain-epochs 14 \
  --finetune-lr 1e-4 \
  --batch-size 16 \
  --accumulation-steps 1 \
  --lr 5e-4 \
  --num-kv-heads 2 \
  --seq-len 1024 \
  --drope-epochs 4 \
  --drope-min-epochs 2 \
  --drope-patience 1 \
  --drope-min-delta 5e-4 \
  --fp16
```

### Resuming

```bash
uv run bach-gen train --resume models/latest.pt --epochs 500
```

Restores model weights, optimizer state, and epoch counter.

---

## 3. Post-Training Calibration

Runs automatically at the end of training. Takes 50 sequences from the most recent training corpus and computes:

- **Perplexity range** (P10–P90) — how predictable real Bach is to the model
- **Entropy range** (P10–P90) — how uncertain the model is on real Bach

These ranges anchor the information-theoretic evaluation score so it reflects "naturalness relative to the training corpus" rather than absolute values.

In curriculum mode, calibration uses the fine-tune corpus (phase 2 dataset/subset).
The ranges are saved to `models/information_calibration.json` and auto-loaded by generation/evaluation scoring.

---

## 4. Generate

### Standard (interleaved) generation

```bash
uv run bach-gen generate --key "C minor" --mode fugue \
  --texture polyphonic --imitation high --harmonic-rhythm fast \
  --tension high --chromaticism moderate \
  --candidates 100 --top 3 --temperature 0.9 --min-p 0.03
```

### Voice-by-voice (sequential) generation

```bash
uv run bach-gen generate --key "C minor" --mode fugue \
  --texture polyphonic --imitation high \
  --voice-by-voice --candidates 50 --top 3 --temperature 0.9 --min-p 0.03
```

Optionally provide a MIDI file for voice 1 — the model generates the remaining voices:

```bash
uv run bach-gen generate --key "C minor" --mode fugue \
  --voice-by-voice --provide-voice soprano.mid
```

**What happens:**

1. **Load checkpoint** — searches for `best.pt`, then `latest.pt`, then `final.pt` in `models/`.

   If `--mode` is omitted, generation reads `data/mode.json`. If it finds `mode=all`, it automatically falls back to `chorale` (because `all` is a prep mode, not a concrete generation form).

2. **Build prompt** — assembles conditioning prefix: `BOS STYLE FORM MODE LENGTH METER TEXTURE IMITATION HARMONIC_RHYTHM HARMONIC_TENSION CHROMATICISM ENCODE_* KEY [SUBJECT]`. Subject is parsed from `--subject` or generated randomly (interleaved mode only; sequential mode skips subject generation). For `--voice-by-voice`, the prompt includes `ENCODE_SEQUENTIAL`; if `--provide-voice` is given, voice 1 is serialized into the prompt followed by `VOICE_SEP`.

3. **Decode** — either autoregressive sampling (default) or beam search (`--beam-width`, interleaved mode only). Decoding constraints enforce key membership and pitch range at each step. In sequential mode, `VOICE_SEP` is blocked until at least 4 notes have been generated for the current voice, preventing empty voices.

4. **Score** each candidate on 7 dimensions:

   | Metric | Weight | What it measures |
   |---|---|---|
   | Structural | 0.25 | Key consistency, cadences, modulation, thematic recurrence |
   | Statistical | 0.20 | Pitch/interval/duration distribution match to corpus |
   | Thematic recall | 0.15 | Subject recurrence after opening bars |
   | Information | 0.15 | Perplexity and entropy relative to calibrated ranges |
   | Voice leading | 0.10 | Parallel 5ths/8ves, crossings, leap resolution |
   | Contrapuntal | 0.08 | Motion variety, voice independence, consonance |
   | Completeness | 0.07 | Proper opening, development, cadential ending |

5. **Rank** by composite score, return top K as MIDI files in `output/`.

**Key flags:**

- `--style bach|baroque|renaissance|classical` — style conditioning token
- `--length short|medium|long|extended` — target length (default inferred from form)
- `--meter 4_4|3_4|6_8|...` — meter conditioning
- `--texture homophonic|polyphonic|mixed` — texture conditioning
- `--imitation none|low|high` — imitation density conditioning
- `--harmonic-rhythm slow|moderate|fast` — harmonic rhythm conditioning
- `--tension low|moderate|high` — harmonic tension conditioning
- `--chromaticism low|moderate|high` — chromaticism conditioning
- `--voice-by-voice` — use sequential (voice-by-voice) generation
- `--provide-voice path.mid` — supply voice 1 as MIDI (requires `--voice-by-voice`)
- `--beam-width` — beam search with length-normalized scoring (Wu et al.)
- `--min-p` — primary sampling control (with temperature); lower = more diversity, higher = stricter

---

## 5. Evaluate & Calibrate (optional)

**Score a single MIDI:**
```bash
uv run bach-gen evaluate path/to/file.mid
```

**Run baseline calibration study:**
```bash
uv run bach-gen calibrate
```

Scores real Bach vs. three baselines (shuffled notes, random pitches, repetitive patterns). Outputs `calibration.json` with per-dimension statistics and threshold analysis.

---

## Vocabulary Summary

18 tokens are reserved for Phase 2/3 conditioning and encoding mode after `METER_ALLA_BREVE` (ID 40).  
`FORM_SONATA` is appended at the end of the vocabulary to avoid shifting legacy token IDs.

| ID Range | Tokens |
|---|---|
| 0–40 | PAD, BOS, EOS, VOICE_1–4, SUBJECT_*, BAR, BEAT_1–6, MODE_*, STYLE_*, FORM_*, LENGTH_*, METER_* |
| 41–43 | TEXTURE_HOMOPHONIC, TEXTURE_POLYPHONIC, TEXTURE_MIXED |
| 44–46 | IMITATION_NONE, IMITATION_LOW, IMITATION_HIGH |
| 47–49 | HARMONIC_RHYTHM_SLOW, HARMONIC_RHYTHM_MODERATE, HARMONIC_RHYTHM_FAST |
| 50–52 | HARMONIC_TENSION_LOW, HARMONIC_TENSION_MODERATE, HARMONIC_TENSION_HIGH |
| 53–55 | CHROMATICISM_LOW, CHROMATICISM_MODERATE, CHROMATICISM_HIGH |
| 56–57 | ENCODE_INTERLEAVED, ENCODE_SEQUENTIAL |
| 58 | VOICE_SEP |
| 59–82 | KEY_* |
| 83+ | Scale-degree: OCT/DEG/ACC/DUR/TS, then FORM_SONATA (to 120). Absolute: Pitch/DUR/TS, then FORM_SONATA (to 154). |

**Breaking change:** all cached data and trained models from before this vocabulary update are invalidated. Re-run `prepare-data` and retrain.

---

## Quick Reference: File Layout

```
data/
  tokenizer.json          # vocab mappings
  sequences.json          # token sequences (interleaved + sequential)
  piece_ids.json          # source IDs
  corpus_stats.json       # distribution stats
  mode.json               # metadata
  calibration.json        # baseline study (optional)

models/
  latest.pt               # most recent epoch
  best.pt                 # best validation loss
  final.pt                # end of training
  pre_drope.pt            # before DroPE (if --drope)
  drope_latest.pt         # latest DroPE epoch (if --drope)
  drope_best.pt           # best DroPE val checkpoint (if --drope)
  drope_final.pt          # after DroPE (if --drope)

output/
  fugue_C_minor_1.mid     # top-ranked generation (interleaved)
  fugue_C_minor_2.mid
  fugue_C_minor_vbv_1.mid # top-ranked generation (voice-by-voice)
```
