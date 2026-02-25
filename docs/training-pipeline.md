# Training Pipeline

End-to-end guide: data preparation, training, and generation.

## 1. Prepare Data

```bash
uv run bach-gen prepare-data --mode fugue --tokenizer absolute
```

**What happens:**

1. **Load works** from music21 corpus + `data/midi/`. Deduplicate by source path. Optional `--composer-filter` (e.g. `bach,baroque`).

2. **Extract voices** per mode. `--mode all` auto-detects form and voice count per piece. Otherwise extracts the specified voicing (2-part pairs, 4-voice groups, etc.).

3. **Augment** to all 12 keys by transposition (skipped for scale-degree tokenizer since it's already key-agnostic). Transpositions that push notes outside MIDI 36-84 are dropped.

4. **Tokenize** each piece. Computes `length_bars` from the score and encodes the full prefix: `BOS STYLE FORM MODE LENGTH METER KEY [SUBJECT] <events> EOS`. Sequences under 20 tokens are dropped.

5. **Chunk or drop** long sequences. By default, sequences exceeding `--max-seq-len` are split into overlapping windows (75% stride). Use `--no-chunk` to drop them instead.

6. **Save** to the output directory (default `data/`).

**Output files:**

| File | Contents |
|---|---|
| `tokenizer.json` | Serialized vocab mappings |
| `sequences.json` | All token sequences |
| `piece_ids.json` | Source piece ID per sequence (for leak-free splits) |
| `mode.json` | Mode, voice count, tokenizer type |
| `corpus_stats.json` | Pitch-class, interval, duration distributions |

**Key flags:**

- `--tokenizer absolute|scale-degree` — absolute pitch (135 tokens) or key-agnostic scale degrees (101 tokens)
- `--max-seq-len 2048` — context window ceiling
- `--mode` — determines voice count and form token

---

## 2. Train

```bash
uv run bach-gen train --epochs 500 --lr 3e-4 --batch-size 8 --pos-encoding pope --num-kv-heads 2
```

**What happens:**

1. **Load data** from `--data-dir` (default `data/`). Auto-detects mode and seq_len from saved metadata.

2. **Split** into train/val at the piece level (90/10) using `piece_ids.json` to prevent chunk leakage.

3. **Build model** — BachTransformer (256d, 8 heads, 8 layers, SwiGLU FFN, RMSNorm, weight tying). Key options:
   - `--pos-encoding rope|pope` — RoPE (standard) or PoPE (polar coordinate, better what-where separation)
   - `--num-kv-heads` — grouped query attention (fewer KV heads = smaller KV cache at inference)

4. **Train** with AdamW (betas 0.9/0.98), cosine annealing to 1e-6, label smoothing 0.1, gradient clipping at 1.0. Optional `--fp16` for mixed precision on CUDA.

5. **Save checkpoints** throughout:
   - `latest.pt` — after every epoch (safe to resume from)
   - `best.pt` — when validation loss improves
   - `final.pt` — end of training

### Curriculum Training (optional)

Pre-train on a broad corpus, then fine-tune on Bach:

```bash
uv run bach-gen train --curriculum \
  --data-dir data/broad --finetune-data-dir data/bach \
  --pretrain-epochs 300 --epochs 500 --finetune-lr 1e-4
```

Phase 1 trains on the broad dataset. Phase 2 swaps to the Bach dataset with a fresh optimizer at a lower learning rate. Saves `pretrain_final.pt` at the transition.

### DroPE Recalibration (optional)

After training with positional encoding, drop it and retrain briefly:

```bash
uv run bach-gen train --pos-encoding pope --drope --drope-epochs 10 --drope-lr 1e-3
```

The model learns to recover positional information from causal masking and BEAT tokens. Enables length generalization beyond the training context. Saves `pre_drope.pt` before and `drope_final.pt` after. Sets `drope_trained=True` in the checkpoint config.

### Resuming

```bash
uv run bach-gen train --resume models/latest.pt --epochs 500
```

Restores model weights, optimizer state, and epoch counter.

---

## 3. Post-Training Calibration

Runs automatically at the end of training. Takes 50 sequences from the training corpus and computes:

- **Perplexity range** (P10–P90) — how predictable real Bach is to the model
- **Entropy range** (P10–P90) — how uncertain the model is on real Bach

These ranges anchor the information-theoretic evaluation score so it reflects "naturalness relative to the training corpus" rather than absolute values.

---

## 4. Generate

```bash
uv run bach-gen generate --key "C minor" --candidates 100 --top 3 --temperature 0.9
```

**What happens:**

1. **Load checkpoint** — searches for `best.pt`, then `latest.pt`, then `final.pt` in `models/`.

2. **Build prompt** — assembles conditioning prefix: `BOS STYLE FORM MODE LENGTH METER KEY [SUBJECT]`. Subject is parsed from `--subject` or generated randomly in the given key.

3. **Decode** — either autoregressive sampling (default) or beam search (`--beam-width`). Decoding constraints enforce key membership and pitch range at each step.

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
- `--beam-width` — beam search with length-normalized scoring (Wu et al.)

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

## Quick Reference: File Layout

```
data/
  tokenizer.json          # vocab mappings
  sequences.json          # token sequences
  piece_ids.json          # source IDs
  corpus_stats.json       # distribution stats
  mode.json               # metadata
  calibration.json        # baseline study (optional)

models/
  latest.pt               # most recent epoch
  best.pt                 # best validation loss
  final.pt                # end of training
  pretrain_final.pt       # pre-curriculum (if --curriculum)
  pre_drope.pt            # before DroPE (if --drope)
  drope_final.pt          # after DroPE (if --drope)

output/
  fugue_C_minor_1.mid     # top-ranked generation
  fugue_C_minor_2.mid
  fugue_C_minor_3.mid
```
