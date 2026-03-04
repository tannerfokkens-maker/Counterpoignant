# Training Pipeline

End-to-end guide for the next training run: data preparation, staged curriculum training with context-length extension, and generation.

---

## 1. Prepare Data

Prepare once at the **maximum context length** you intend to reach. The training loop controls effective context length via `--seq-len-stages`, so data only needs to be prepared once.

```bash
uv run bach-gen prepare-data \
  --data-dir data \
  --max-seq-len 16384
```

Current defaults:
- `--mode all`
- `--tokenizer scale-degree`
- `--max-source-voices 4`
- `--max-groups-per-work 1`
- `--max-pairs-per-work 2`
- `--pair-strategy adjacent+outer`
- `--sonata-policy counterpoint-safe`
- `--composer-filter bach,baroque,renaissance,classical`

**What happens:**

1. **Load works** from music21 corpus + `data/midi/`. Deduplicate by source path.

2. **Extract voices** per mode. `--mode all` auto-detects form/voice count per piece.

3. **Augment** — transposition augmentation (skipped for scale-degree tokenizer since it's already key-agnostic). Transpositions pushing notes outside MIDI 36–84 are dropped.

4. **Analyze** each piece for conditioning labels:
   - **Texture** — onset synchronization: `homophonic` (>0.60), `polyphonic` (<0.54), else `mixed`
   - **Imitation** — cross-voice interval matching: `high` (>0.30), `low` (>0.10), else `none`
   - **Harmonic rhythm** — beat-to-beat chord changes/measure: `slow` (≤2.77), `moderate` (2.77–3.18), `fast` (>3.18)
   - **Harmonic tension** — dissonance ratio: `high` (>0.145), `moderate` (>0.128), else `low`
   - **Chromaticism** — non-diatonic note fraction: `high` (>0.15), `moderate` (>0.05), else `low`

5. **Tokenize** each piece in dual encoding:
   - **Interleaved** — voices merged chronologically: `BOS STYLE FORM MODE LENGTH METER TEXTURE IMITATION HARMONIC_RHYTHM HARMONIC_TENSION CHROMATICISM ENCODE_INTERLEAVED KEY <events> EOS`
   - **Sequential** — voice-by-voice with own timelines: `BOS <conditioning> ENCODE_SEQUENTIAL KEY VOICE_1 <notes> VOICE_SEP VOICE_2 <notes> ... EOS`

6. **Chunk** long sequences into overlapping windows (75% stride). Each chunk preserves the full conditioning prefix (BOS through KEY token). Use `--no-chunk` to drop long sequences instead.

7. **Save** to output directory.

**Output files:**

| File | Contents |
|---|---|
| `tokenizer.json` | Serialized vocab mappings |
| `sequences.json` | All token sequences (interleaved + sequential) |
| `piece_ids.json` | Source piece ID per sequence (for leak-free splits and piece balancing) |
| `mode.json` | Mode metadata: mode, voice count, tokenizer type, max seq len |
| `corpus_stats.json` | Pitch-class, interval, duration distributions |

---

## 2. Train (Curriculum + Staged Context)

The recommended training pipeline is a **three-phase curriculum with staged context-length extension**:

1. **Pre-train** on the broad corpus (all eras) with progressively increasing context length
2. **DroPE recalibration** — drop positional embeddings for length generalization
3. **Fine-tune** on Bach with the same staged context schedule

### The Command

```bash
uv run bach-gen train \
  --curriculum \
  --data-dir data \
  --finetune bach \
  --seq-len-stages "4096:40,8192:25,16384:15" \
  --batch-size 8 \
  --lr 3e-4 \
  --finetune-lr 1e-4 \
  --pos-encoding pope \
  --num-kv-heads 2 \
  --piece-balance sqrt \
  --fp16
```

### What `--seq-len-stages` Does

The flag `"4096:40,8192:25,16384:15"` defines three context-length stages within a single training run:

| Stage | Context Length | Epochs | Purpose |
|---|---|---|---|
| 1 | 4096 | 40 | Core learning — cheap iterations, most of the convergence |
| 2 | 8192 | 25 | Context extension — model adapts to longer dependencies |
| 3 | 16384 | 15 | Full context — final polish at target length |

At each stage boundary the training loop automatically:
- Updates the dataset crop window (prefix-preserving: conditioning tokens BOS→KEY are always kept)
- Extends the positional embedding caches (lazy, no rebuild needed)
- Resets the cosine LR schedule for the new stage
- Resets the early stopping counter
- Saves a stage checkpoint (`stage1.pt`, `stage2.pt`, etc.)

**Epoch rationale:** Bulk learning happens at the shortest context (cheapest per iteration). Each extension needs fewer epochs since the model only needs to adapt its positional generalization, not relearn musical structure. The suggested 40/25/15 split totals 80 epochs per phase. Adjust based on dataset size and hardware — if you trained ~50 epochs at 4096 on less data, 40 is a reasonable starting point for a larger dataset.

### Phase Breakdown

**Phase 1 — Pre-train (broad corpus):**
- Trains on all eras (Bach, Baroque, Renaissance, Classical)
- Runs through all `--seq-len-stages`
- Cosine annealing per stage, early stopping per stage
- Checkpoints: `pretrain_best.pt`, `pretrain_stage1.pt`, `pretrain_stage2.pt`, ...

**Phase 2 — DroPE recalibration:**
- Drops positional embeddings entirely
- Short recalibration at original context length (default 20 epochs, early-stopped)
- The model learns to recover position from causal masking and BEAT tokens
- Checkpoints: `pre_drope.pt`, `drope_best.pt`, `drope_final.pt`

**Phase 3 — Fine-tune on Bach:**
- Filters pre-train data to `STYLE_BACH` sequences (via `--finetune bach`)
- Fresh optimizer at `--finetune-lr`
- Runs through all `--seq-len-stages` again
- Checkpoints: `finetune_best.pt`, `finetune_stage1.pt`, ...

### Piece Balancing

`--piece-balance sqrt` (default) uses a `WeightedRandomSampler` to down-weight pieces that produce many chunks. Without this, long orchestral works (Beethoven quartets, Schubert) dominate training. Available modes:

| Mode | Weight per chunk | Effect |
|---|---|---|
| `none` | 1.0 | No balancing — long pieces dominate proportionally |
| `sqrt` | 1/√n | Soft correction — long pieces still appear more, but less so |
| `inverse` | 1/n | Hard correction — every piece contributes equally regardless of length |

### Prefix-Preserving Crops

When sequences exceed the current stage's context length, `BachDataset.__getitem__` crops randomly but **always preserves the conditioning prefix** (BOS through KEY token). This means every training sample carries its style/form/key/texture metadata regardless of context length. When no tokenizer is available (backward compat), crops fall back to the old random-offset behavior.

### Model Architecture

| Component | Setting |
|---|---|
| Type | Decoder-only Transformer, ~6.4M params |
| Embedding | 256d |
| Attention | 8 heads, GQA with `--num-kv-heads 2` recommended |
| Layers | 8 |
| FFN | SwiGLU (8/3 expansion) |
| Normalization | RMSNorm, pre-norm |
| Positional | PoPE (recommended) or RoPE |
| Weight tying | Input embedding tied to output projection |
| Precision | fp16 on CUDA, fp32 fallback |

### Key Training Flags

| Flag | Default | Purpose |
|---|---|---|
| `--seq-len-stages` | none | Staged context: `"len:epochs,..."` |
| `--piece-balance` | `sqrt` | Down-weight heavily-chunked pieces |
| `--pos-encoding` | `pope` | Positional encoding (pope or rope) |
| `--num-kv-heads` | none (=MHA) | GQA KV heads |
| `--finetune` | none | Fine-tune target (style token or composer substring) |
| `--finetune-lr` | 5e-5 | Fine-tune learning rate |
| `--drope/--no-drope` | enabled | DroPE recalibration phase |
| `--early-stop/--no-early-stop` | enabled | Early stopping on val loss plateau |
| `--fp16` | off | Mixed precision (CUDA only) |

---

## 3. Lambda / GPU Setup

```bash
# 1. Clone and install
git clone <your-repo-url>
cd Counterpoignant
uv sync

# 2. Prepare data at max context length
uv run bach-gen prepare-data --data-dir data --max-seq-len 16384

# 3. Full curriculum training
uv run bach-gen train \
  --curriculum \
  --data-dir data \
  --finetune bach \
  --seq-len-stages "4096:40,8192:25,16384:15" \
  --batch-size 8 \
  --lr 3e-4 \
  --finetune-lr 1e-4 \
  --pos-encoding pope \
  --num-kv-heads 2 \
  --piece-balance sqrt \
  --fp16

# 4. Resume after preemption
uv run bach-gen train \
  --curriculum \
  --data-dir data \
  --finetune bach \
  --seq-len-stages "4096:40,8192:25,16384:15" \
  --resume models/latest.pt \
  --fp16
```

Persist `data/`, `models/`, and `output/` to durable storage between sessions.

### Quick Smoke Test

Sanity-check the full pipeline after code/data changes:

```bash
uv run bach-gen train \
  --curriculum \
  --finetune bach \
  --seq-len-stages "512:4,1024:3,2048:2" \
  --batch-size 16 \
  --lr 5e-4 \
  --finetune-lr 1e-4 \
  --num-kv-heads 2 \
  --drope-epochs 2 \
  --drope-min-epochs 1 \
  --drope-patience 1
```

---

## 4. Post-Training Calibration

Runs automatically at the end of training. Takes 50 sequences from the fine-tune corpus and computes:

- **Perplexity range** (P10–P90) — how predictable real Bach is to the model
- **Entropy range** (P10–P90) — how uncertain the model is on real Bach

These ranges anchor the information-theoretic evaluation score. Saved to `models/information_calibration.json` and auto-loaded by generation/evaluation.

---

## 5. Generate

### Standard (interleaved)

```bash
uv run bach-gen generate --key "C minor" --mode fugue \
  --texture polyphonic --imitation high --harmonic-rhythm fast \
  --tension high --chromaticism moderate \
  --candidates 100 --top 3 --temperature 0.9 --min-p 0.03
```

### Voice-by-voice (sequential)

```bash
uv run bach-gen generate --key "C minor" --mode fugue \
  --texture polyphonic --imitation high \
  --voice-by-voice --candidates 50 --top 3 --temperature 0.9 --min-p 0.03
```

Optionally provide a MIDI file for voice 1:

```bash
uv run bach-gen generate --key "C minor" --mode fugue \
  --voice-by-voice --provide-voice soprano.mid
```

**What happens:**

1. **Load checkpoint** — searches for `best.pt`, then `latest.pt`, then `final.pt` in `models/`.

2. **Build prompt** — conditioning prefix + optional subject.

3. **Decode** — autoregressive sampling (default) or beam search (`--beam-width`, interleaved only). Constraints enforce key membership and pitch range.

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

---

## 6. Evaluate & Calibrate

```bash
# Score a single MIDI
uv run bach-gen evaluate path/to/file.mid

# Run baseline calibration study
uv run bach-gen calibrate
```

---

## Checkpoint File Layout

```
models/
  latest.pt                  # most recent epoch (safe to resume)
  best.pt                    # best validation loss
  final.pt                   # end of training

  # Curriculum stages
  pretrain_best.pt           # best pre-train checkpoint
  pretrain_stage1.pt         # end of pre-train stage 1 (4096)
  pretrain_stage2.pt         # end of pre-train stage 2 (8192)
  pretrain_stage3.pt         # end of pre-train stage 3 (16384)
  pretrain_final.pt          # end of pre-train phase

  # DroPE
  pre_drope.pt               # before DroPE recalibration
  drope_best.pt              # best DroPE checkpoint
  drope_final.pt             # after DroPE

  # Fine-tune stages
  finetune_best.pt           # best fine-tune checkpoint
  finetune_stage1.pt         # end of fine-tune stage 1
  finetune_stage2.pt         # end of fine-tune stage 2
  finetune_stage3.pt         # end of fine-tune stage 3
  finetune_final.pt          # end of fine-tune phase

  information_calibration.json  # perplexity/entropy ranges

data/
  tokenizer.json             # vocab mappings
  sequences.json             # token sequences
  piece_ids.json             # source IDs (piece balancing + leak-free splits)
  corpus_stats.json          # distribution stats
  mode.json                  # metadata

output/
  fugue_C_minor_1.mid        # top-ranked generation (interleaved)
  fugue_C_minor_vbv_1.mid    # top-ranked generation (voice-by-voice)
```
