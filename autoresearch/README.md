This directory adapts Andrej Karpathy's `autoresearch` workflow to the
current `bach-gen` training pipeline.

The upstream repo assumes:

- a tiny codebase
- one editable `train.py`
- one fixed `prepare.py`
- a fixed-time experiment budget
- one scalar validation metric to optimize

This adaptation keeps the same shape:

- [prepare.py](/Users/tannerfokkens/Documents/2pt-bach_update/autoresearch/prepare.py)
  is the fixed harness. It loads `datamidiall`, builds a deterministic
  train/validation split, provides evaluation, and loads optional phase
  checkpoints without restoring optimizer state.
- [train.py](/Users/tannerfokkens/Documents/2pt-bach_update/autoresearch/train.py)
  is the single experiment file. This is the file an agent edits during the
  experiment loop.
- [program.md](/Users/tannerfokkens/Documents/2pt-bach_update/autoresearch/program.md)
  defines the operating procedure.
- [results.tsv](/Users/tannerfokkens/Documents/2pt-bach_update/autoresearch/results.tsv)
  records experiment outcomes.

Key differences from upstream:

- This runs on the local `bach-gen` codebase rather than the toy GPT in
  upstream `autoresearch`.
- The primary metric is `val_loss`, not `val_bpb`.
- The platform target is this machine, including MPS on the Mac Studio.
- Experiments optimize a single training phase at a time, because the full
  curriculum is too large to compare fairly in a 5-minute budget.

Usage:

```bash
uv run python autoresearch/train.py --dry-run
uv run python autoresearch/train.py > autoresearch/run.log 2>&1
```

The default baseline in `train.py` is intentionally conservative and is meant
to be edited during research.

Adaptive behavior:

- before the timed run, the harness runs a short preflight
- if the candidate config exceeds the soft memory cap or projects too few
  optimizer steps in the fixed budget, it halves `batch_size` and retries
- if the real run still hits an OOM, it can retry from scratch with a smaller
  batch size

This is mainly there to make local Mac Studio experimentation robust on MPS.
