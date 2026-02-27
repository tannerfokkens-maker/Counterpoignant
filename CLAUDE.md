# CLAUDE.md — AI Assistant Guide for Counterpoignant

This file provides context for AI coding assistants (Claude, Copilot, etc.) working on the **Counterpoignant** project — a music generation system that produces Bach-style multi-voice counterpoint using a small decoder-only Transformer.

---

## Project Overview

**Counterpoignant** generates Bach-style two-to-four-voice compositions (inventions, fugues, chorale-style pieces) by:
1. Tokenizing musical corpora into a compact, key-agnostic scale-degree vocabulary
2. Training a custom Transformer model on diverse Renaissance/Baroque repertoire
3. Using curriculum learning to specialize on Bach after broad pretraining
4. Generating N candidates and selecting the top K via a 7-dimension evaluation scorer

**Entry point:** `bach-gen` CLI (defined in `src/bach_gen/cli.py`)
**Package manager:** `uv` (preferred over pip)
**Python version:** 3.12+

---

## Repository Structure

```
Counterpoignant/
├── src/bach_gen/           # All application source code
│   ├── cli.py              # Click CLI: prepare-data, train, generate, evaluate, calibrate, play
│   ├── model/
│   │   ├── architecture.py # BachTransformer (RoPE/PoPE, SwiGLU, RMSNorm, GQA)
│   │   ├── config.py       # ModelConfig dataclass
│   │   └── trainer.py      # Training loop, checkpointing, curriculum learning, DroPE
│   ├── data/
│   │   ├── corpus.py       # Multi-source corpus loader (music21 + MIDI)
│   │   ├── dataset.py      # PyTorch BachDataset
│   │   ├── extraction.py   # Voice extraction (2–4 voices)
│   │   ├── tokenizer.py    # Main tokenizer dispatcher
│   │   ├── scale_degree_tokenizer.py  # Key-agnostic 91-token mode
│   │   ├── augmentation.py # Transposition, voice reordering augmentation
│   │   └── analysis.py     # Texture, imitation, harmonic-rhythm analysis
│   ├── generation/
│   │   ├── generator.py    # Orchestrates N-candidate generation + ranking
│   │   ├── sampling.py     # Temperature, top-k, top-p, beam search
│   │   ├── constraints.py  # Hard constraints: voice range, key adherence
│   │   ├── subject.py      # Subject/theme parsing
│   │   └── scale_degree_constraints.py
│   ├── evaluation/
│   │   ├── scorer.py       # Composite score orchestrator (7 dimensions)
│   │   ├── voice_leading.py    # Parallel 5ths/8ths, crossings, leaps
│   │   ├── statistical.py      # Pitch/interval/duration distribution matching
│   │   ├── structural.py       # Key consistency, cadences, thematic recurrence
│   │   ├── contrapuntal.py     # Motion variety, voice independence, consonance
│   │   └── information.py      # Perplexity/entropy vs. calibrated ranges
│   └── utils/
│       ├── constants.py    # Voice ranges, token vocab, meters, musical constants
│       ├── music_theory.py # Key parsing, scales, intervals, Krumhansl profiles
│       ├── midi_io.py      # MIDI I/O (note events → MIDI files)
│       └── voice_index.py  # Voice indexing helpers
├── tests/
│   └── test_curriculum.py  # pytest suite for corpus filtering + curriculum training
├── docs/
│   ├── project-summary.md  # High-level overview
│   └── training-pipeline.md # Operational step-by-step guide
├── ROADMAP.md              # Detailed phased development plan
├── pyproject.toml          # Project metadata and dependencies
├── .python-version         # Pins Python 3.12
├── download_kernscores.sh  # Corpus download scripts
├── download_jrp.sh
├── download_kunstderfuge.py
├── download_openscore_quartets.sh
├── convert_openscore.sh
└── fix_fugue_labels.py
```

**Runtime directories (gitignored, created locally):**
- `/data/` — tokenized datasets and raw corpus files
- `/models/` — model checkpoints
- `/output/` — generated MIDI files

---

## Development Environment Setup

```bash
# Install uv (if not present)
pip install uv

# Install all dependencies
uv sync

# Run the CLI
uv run bach-gen --help

# Run tests
uv run pytest tests/

# Install in editable mode (alternative)
uv pip install -e .
```

No `.env` file is required. All configuration is passed via CLI flags.

---

## CLI Commands

All commands are accessed via `uv run bach-gen <command>` (or `bach-gen <command>` if installed).

| Command | Purpose |
|---------|---------|
| `prepare-data` | Tokenize corpus, extract voices, augment, analyze, chunk into dataset |
| `train` | Train Transformer; supports curriculum learning and DroPE recalibration |
| `generate` | Generate N candidates, score, return top K MIDI files |
| `evaluate` | Score a single MIDI file across all 7 evaluation dimensions |
| `calibrate` | Baseline calibration study (real Bach vs. degenerate baselines) |
| `play` | Play a MIDI file via FluidSynth, timidity, or system player |

Key CLI flags to know:
- `--composer-filter` — Restrict training to specific composers (e.g., `bach`)
- `--curriculum` — Enable two-stage curriculum (broad → Bach fine-tune)
- `--data-dir` — Path to prepared dataset directory
- `--checkpoint` / `--output-dir` — Checkpoint save/load paths

---

## Model Architecture

**Type:** Decoder-only Transformer (GPT-family), ~6.4M parameters

| Component | Detail |
|-----------|--------|
| Vocab size | 400 (default); 91 in scale-degree mode |
| Embedding dim | 256 |
| Attention heads | 8 (with optional GQA) |
| Layers | 8 |
| Positional encoding | RoPE (default), PoPE (experimental), or none (DroPE phase) |
| FFN activation | SwiGLU (no bias, 8/3 expansion ratio) |
| Normalization | RMSNorm, pre-norm architecture |
| Weight tying | Input embedding tied to output projection |
| Training precision | fp16 (CUDA), fp32 fallback |

**Positional embedding modes:**
- **RoPE** — Rotary Position Embeddings; default for training
- **PoPE** — Polar Coordinate Position Embeddings; experimental, decouples content from position
- **DroPE** — Drop positional embeddings after pretraining for length generalization; recalibrate with no-position mode

---

## Tokenizer & Vocabulary

Two tokenizer modes exist:

1. **Scale-degree (default, key-agnostic)** — 91 tokens
   - Represents notes as scale degrees (e.g., `^1`, `^3`, `^5`) relative to the active key
   - Enables transposition-invariant training and better generalization
   - File: `src/bach_gen/scale_degree_tokenizer.py`

2. **Absolute pitch** — 148 tokens
   - Uses MIDI pitch numbers directly
   - File: `src/bach_gen/tokenizer.py`

Special tokens include voice separators, key/meter conditioning tokens, texture/style conditioning tokens (added in Phase 2), and structural markers (bar lines, rests, duration modifiers).

---

## Evaluation System

The scorer in `src/bach_gen/evaluation/scorer.py` combines 7 independent dimensions:

| Dimension | Module | Measures |
|-----------|--------|---------|
| Voice Leading | `voice_leading.py` | Parallel 5ths/8ths, voice crossings, leap resolution |
| Statistical | `statistical.py` | Pitch/interval/duration distribution match vs. Bach corpus |
| Structural | `structural.py` | Key consistency, cadence placement, thematic recurrence |
| Contrapuntal | `contrapuntal.py` | Motion variety (contrary/oblique/similar/parallel), voice independence |
| Information | `information.py` | Perplexity and entropy relative to calibrated min/max from real Bach |
| Completeness | `scorer.py` | Ensures the piece reaches a proper ending |
| Thematic Recall | `structural.py` | Subject/answer return in later sections |

Each dimension returns a float in [0, 1]; the composite score is a weighted average.

**Calibration:** Run `bach-gen calibrate` before relying on scoring to establish baseline ranges from real Bach and degenerate (random/silent) examples.

---

## Training Pipeline

### Standard Training

```bash
# 1. Download and prepare corpus
bash download_kernscores.sh
uv run bach-gen prepare-data --data-dir ./data

# 2. Train
uv run bach-gen train --data-dir ./data --output-dir ./models

# 3. Generate
uv run bach-gen generate --checkpoint ./models/best.pt --output-dir ./output

# 4. Evaluate
uv run bach-gen evaluate --checkpoint ./models/best.pt --input ./output/piece.mid
```

### Curriculum Learning

```bash
# Stage 1: Pretrain on broad corpus (all composers)
uv run bach-gen train --data-dir ./data --output-dir ./models/pretrain

# Stage 2: Fine-tune on Bach only
uv run bach-gen train \
  --data-dir ./data \
  --output-dir ./models/finetune \
  --checkpoint ./models/pretrain/best.pt \
  --composer-filter bach \
  --curriculum
```

### DroPE Recalibration

After pretraining, if using DroPE for length generalization:

```bash
# Recalibrate with positional embeddings dropped
uv run bach-gen train \
  --checkpoint ./models/pretrain/best.pt \
  --output-dir ./models/drope \
  --position-mode none \
  --drope-recalibrate
```

See `docs/training-pipeline.md` for the full reference.

---

## Code Conventions

### Python Style

- **Snake_case** for functions and variables: `get_all_works()`, `score_composition()`
- **PascalCase** for classes: `BachTransformer`, `Trainer`, `BachTokenizer`
- **UPPER_SNAKE_CASE** for constants: `MIN_PITCH`, `TICKS_PER_QUARTER`
- **`_` prefix** for private methods: `_build_cache()`, `_save_checkpoint()`
- **Type hints** throughout (PEP 484); all public functions and class constructors should be annotated
- **Docstrings** on all public classes and functions (triple-quote, descriptive)
- **Explicit imports** — never use `from module import *`

### Architecture Principles

- **Music-theory-first**: Design decisions are grounded in music theory (scale degrees, voice ranges, harmonic rules), not just ML conventions
- **Modular evaluation**: Each scoring dimension is independent and testable in isolation
- **Rejection sampling**: Generate many candidates, rank by composite score, return top K — do not try to make a single generation "perfect"
- **Curriculum learning over fine-tuning**: Use diverse pretraining to avoid data starvation, then specialize
- **Hard constraints at decode time**: Enforce voice ranges and key adherence during generation, not just as training targets

### Key Files to Understand First

If modifying the codebase, read these in order:
1. `src/bach_gen/utils/constants.py` — Vocabulary, voice ranges, musical constants
2. `src/bach_gen/model/config.py` — `ModelConfig` dataclass (all hyperparameters)
3. `src/bach_gen/data/scale_degree_tokenizer.py` — Token encoding/decoding
4. `src/bach_gen/model/architecture.py` — The `BachTransformer` model
5. `src/bach_gen/cli.py` — Wiring between user-facing commands and internals

---

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/test_curriculum.py -v

# Run with output on failure
uv run pytest tests/ -s
```

**Current test coverage:** `tests/test_curriculum.py` covers:
- Composer filtering in `corpus.py`
- `reset_for_finetuning()` in `trainer.py`
- CLI flag parsing for `--composer-filter`, `--curriculum`, `--data-dir`

When adding new features, add corresponding tests. Use `tmp_path` (pytest fixture) for file system tests. Keep test models small (tiny `ModelConfig` with minimal layers and embed dims).

---

## Dependencies

Managed by `uv` via `pyproject.toml`. Key libraries:

| Library | Version | Role |
|---------|---------|------|
| `torch` | >=2.2 | Neural network core, CUDA training |
| `music21` | >=9.1 | Score parsing (MusicXML, MIDI, corpus) |
| `miditok` | >=3.0 | Music tokenization helpers |
| `mido` | >=1.3 | Low-level MIDI I/O |
| `click` | >=8.1 | CLI framework |
| `rich` | >=13.0 | Terminal formatting, progress bars |
| `pydantic` | >=2.6 | Data validation and settings |
| `scipy` | >=1.12 | Scientific computing (evaluation metrics) |
| `numpy` | >=1.26 | Numerical computing |
| `requests` | >=2.32.5 | HTTP (corpus downloading) |

Build backend: `hatchling`

---

## Current Development Status

See `ROADMAP.md` for the detailed phased plan. Current focus areas:

**Active (Phase 2/3):**
- Conditioning tokens for texture, imitation density, harmonic rhythm, and tension
- Voice-by-voice generation mode
- Validating that conditioning tokens actually alter output behavior

**Known limitations:**
- Structural planning is emergent, not explicit — the model learns local coherence, not high-level form
- Maximum sequence length of 2048 tokens supports ~60 bars; longer pieces require future work
- No user-facing conversational interface yet — all interaction is via CLI flags

**Not yet implemented (planned):**
- Conversational AI layer wrapping the generation pipeline
- Long-form structural planning (explicit section boundaries)
- Real-time interactive generation

---

## Common Pitfalls

1. **Missing calibration**: The evaluation scorer returns misleading scores without running `bach-gen calibrate` first to establish baseline ranges.
2. **Tokenizer mismatch**: Checkpoints trained in scale-degree mode cannot be loaded with absolute-pitch tokenizer settings and vice versa. Always match `--tokenizer-mode` to the checkpoint.
3. **CUDA memory on generation**: Generating many candidates (large `--num-candidates`) is memory-intensive. Reduce batch size or candidate count if OOM.
4. **music21 corpus paths**: `music21.corpus` requires the music21 package to have its corpus data installed. On headless servers, set `music21.environment.set('musicxmlPath', ...)` explicitly.
5. **DroPE ordering**: DroPE recalibration must happen *after* a full pretraining run. Re-using a DroPE checkpoint for further RoPE training will give poor results.

---

## Documentation

| File | Content |
|------|---------|
| `ROADMAP.md` | Phased development plan, design rationale, DroPE discussion |
| `docs/project-summary.md` | High-level one-page overview |
| `docs/training-pipeline.md` | Operational step-by-step training guide |
| `CLAUDE.md` (this file) | AI assistant guide (structure, conventions, workflows) |
