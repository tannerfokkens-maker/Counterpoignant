# Counterpoignant

`Counterpoignant` is a symbolic music research project for generating, training, evaluating, and live-streaming Bach-adjacent counterpoint.

The project is not just a MIDI generator. It includes:

- broad-corpus score ingestion and voice extraction
- structural tokenization with cadence and subject labels
- decoder-only Transformer training with PoPE/RoPE, optional GQA, and optional LoopLM recurrence
- evaluator-guided generation and ranking for fugues and related forms
- endless live MIDI streaming to external synths
- a desktop GUI for live performance workflows

## Project Scope

The current system covers four layers:

1. Data pipeline
   - ingest music21 corpus material and local score files
   - extract 2-4 voice groups from larger scores
   - tokenize music with form/style/key/texture conditioning
   - attach structural labels like cadences and subject entries

2. Model training
   - standard decoder-only Transformer training
   - staged context-length training
   - optional LoopLM recurrent depth with learned exit gating
   - optional DroPE recalibration after positional training
   - optional style/composer fine-tuning after broad pretraining

3. Search and evaluation
   - generate many candidates
   - score them on voice leading, structural coherence, contrapuntal quality, thematic recall, and statistical similarity
   - rank them by objectives like `bach-similarity`, `demo-bach-balance`, and `fugue-balance`

4. Live performance
   - endless rolling-window generation
   - MIDI streaming with prebuffering
   - typed subject prompts, subject MIDI prompts, or a note-builder UI
   - hardware synth workflows through the desktop app

## Architecture

The model is a decoder-only Transformer implemented in [architecture.py](src/bach_gen/model/architecture.py) and configured in [config.py](src/bach_gen/model/config.py).

Core architectural features:

- token embeddings with tied output head
- RMSNorm + pre-norm residual blocks
- SwiGLU feed-forward layers
- causal self-attention
- optional grouped-query attention through `num_kv_heads`
- optional relative attention bias
- `pope` or `rope` positional encoding

The more unusual pieces are:

- **PoPE default training**
  - the main training default is `pope`, not standard RoPE
- **Block-LoopLM recurrence**
  - the layer stack can be split into front / recurrent core / back layers
  - the recurrent core is looped multiple times for effective depth scaling
- **Learned adaptive exit gate**
  - inference can exit early based on cumulative exit mass
- **Sandwich norm**
  - optional post-sublayer RMSNorm for recurrent stability
- **DroPE recalibration**
  - after normal positional training, the model can be recalibrated with positional embeddings removed

This combination makes Counterpoignant more experimental than a plain small transformer baseline.

## Training Pipeline

The pipeline starts with the `prepare-data` command.

### 1. Corpus ingestion and extraction

The data layer loads:

- broad music21 corpus material
- targeted BWV material
- local score files under `data/midi/...` or a custom `--midi-dir`

Then it:

- detects form and voice count
- extracts voice groups
- filters accompaniment-heavy keyboard/sonata textures when desired
- deduplicates sources
- parallelizes parsing and extraction

### 2. Tokenization and conditioning

The tokenizer adds:

- voice-count mode tokens
- form tokens
- style tokens
- length and meter tokens
- texture / imitation / harmonic-rhythm / harmonic-tension / chromaticism tokens
- encoding-mode tokens
- key tokens
- note, duration, and time-shift tokens
- cadence markers
- subject-entry markers

Two tokenizer families are supported:

- `absolute`
- `scale-degree`

The `scale-degree` tokenizer skips 12-key augmentation and keeps training more key-agnostic.

### 3. Structural labeling

`prepare-data` can run in:

- `none`
- `cadence`
- `cadence+subject`

Cadence and subject labels are heuristically detected from the symbolic music, then inserted into token sequences. Subject-aware forms default to fugue-family pieces such as fugues, inventions, and sinfonias.

### 4. Conditioning dropout

Cadence and subject dropout can be controlled separately.

In `cadence+subject` mode, if you do not manually set them, the pipeline auto-derives dropout rates from marker density instead of blindly using one shared probability. This is especially important because cadence and subject markers have different frequencies and different musical value.

### 5. Sequence chunking and train/val split

Prepared sequences are:

- chunked when they exceed `--max-seq-len`
- saved with `piece_ids.json`
- split by piece, not just by chunk, to avoid leakage across train and validation

The dataset also preserves the conditioning prefix when randomly cropping long sequences during training.

### 6. Training phases

The training CLI supports several regimes:

- **single-phase training**
  - standard LM training on one prepared dataset
- **staged context-length training**
  - schedule like `4096:50,8192:30,16384:20`
- **curriculum training**
  1. pre-train on the broad corpus
  2. DroPE recalibration
  3. fine-tune on Bach or another filtered subset
- **LoopLM Stage II**
  - freeze LM weights and train only the exit gate

Training supports:

- gradient accumulation
- CUDA fp16 / bf16
- cosine LR schedules
- early stopping
- piece-balanced weighted sampling so heavily chunked works do not dominate training

## Evaluation and Search

The evaluator combines:

- voice leading
- statistical similarity
- structural coherence
- contrapuntal quality
- completeness
- thematic recall

The composite score is not the only selection signal. The generator can also rank by style proxies such as:

- `bach-similarity`
- `rhetorical-impact`
- `demo-bach-balance`
- `fugue-balance`

This makes Counterpoignant's generation stack closer to evaluator-guided search than naive sampling.

## Live Mode

The live system is designed for endless rolling-window generation instead of file-only rendering.

Features:

- real-time MIDI output
- prebuffered playback
- prompt pinning so the opening material stays in the active context
- optional subject reminders at bar intervals
- subject string prompts
- subject MIDI prompts
- single-channel or multi-channel synth output
- MIDI port auto-detection

This is implemented in [live.py](src/bach_gen/generation/live.py), surfaced through the live CLI, and wrapped by the GUI app.

## Desktop App

The GUI launcher is:

```bash
bach-gen-studio
```

It provides:

- model checkpoint selection
- MIDI port selection
- live transport controls
- endless playback settings
- subject string entry
- a note/rest subject builder
- direct subject MIDI upload
- dark-mode synth-oriented UI

## Installation

Python `3.12+` is required.

The project name is `Counterpoignant`; the current installed command names remain `bach-gen` and `bach-gen-studio`.

Install from the repo root:

```bash
pip install -e .
```

This installs:

- `bach-gen`
- `bach-gen-studio`

## Quick Start

### Launch the app

```bash
bach-gen-studio
```

### List MIDI ports

```bash
bach-gen live --list-midi-ports
```

### Endless live generation

```bash
bach-gen live \
  --key "D minor" \
  --mode fugue \
  --voices 3 \
  --temperature 0.94 \
  --min-p 0.03
```

### Candidate generation

```bash
bach-gen generate \
  --key "D minor" \
  --mode fugue \
  --voices 3 \
  --style bach \
  --meter alla_breve \
  --texture polyphonic \
  --imitation high \
  --harmonic-rhythm moderate \
  --tension moderate \
  --chromaticism high \
  --candidates 200 \
  --candidate-batch-size 1 \
  --top 16 \
  --temperature 0.94 \
  --min-p 0.03 \
  --max-length 4096 \
  --rank-by fugue-balance \
  --min-voices-only \
  --model-path /absolute/path/to/finetune_best.pt
```

### Parameter sweep

```bash
python scripts/sweep_temp_minp.py \
  --forms fugue \
  --temperatures 0.82,0.86,0.90,0.94,0.98 \
  --min-ps 0.03
```

## Main Commands

- `bach-gen prepare-data`
- `bach-gen train`
- `bach-gen generate`
- `bach-gen live`
- `bach-gen evaluate`
- `bach-gen audit-conditioning`
- `python scripts/sweep_temp_minp.py`

## Subject Prompting

Supported subject input forms:

- typed note strings
  - example: `D4:q A4:e r:e Bb4:q. C5:h`
- subject MIDI file
- GUI subject builder

Supported duration forms include:

- `q`, `h`, `e`, `s`, `w`
- `quarter`, `half`, `eighth`, `sixteenth`
- dotted values like `q.` or `dotted_quarter`
- rests like `r:e` or `rest:q`

## Repository Notes

- Large datasets, checkpoints, outputs, and local corpora are intentionally ignored by git.
- Benchmark manifests may still live under `data/benchmarks/`.
- The repo is designed for local experimentation with checkpoints kept outside version control.

## Development

Run focused tests with:

```bash
.venv/bin/pytest tests/test_app_helpers.py tests/test_live_generation.py tests/test_generation_subject_prompt.py tests/test_midi_io.py -q
```

Run the GUI directly from source with:

```bash
PYTHONPATH=src .venv/bin/python -m bach_gen.app
```
