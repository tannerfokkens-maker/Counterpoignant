# autoresearch for bach-gen

This directory adapts the upstream `autoresearch` workflow to the current
`bach-gen` training pipeline.

## Scope

Only edit [train.py](/Users/tannerfokkens/Documents/2pt-bach_update/autoresearch/train.py).

Do not modify:

- [prepare.py](/Users/tannerfokkens/Documents/2pt-bach_update/autoresearch/prepare.py)
- any files under [src/](/Users/tannerfokkens/Documents/2pt-bach_update/src)
- the main training CLI in [src/bach_gen/cli.py](/Users/tannerfokkens/Documents/2pt-bach_update/src/bach_gen/cli.py)

The point is to use the existing project as the fixed evaluator and to let
experiments happen inside one file.

## Goal

Minimize `val_loss` after a fixed wall-clock training budget.

Lower is better.

## Setup

Before running experiments:

1. Choose a fresh branch name like `autoresearch/<tag>`.
2. Verify [datamidiall](/Users/tannerfokkens/Documents/2pt-bach_update/datamidiall) exists.
3. Verify [results.tsv](/Users/tannerfokkens/Documents/2pt-bach_update/autoresearch/results.tsv) exists and has a header row.
4. Run a baseline exactly once before editing anything.

## Experiment rules

- Run experiments with:

```bash
uv run python autoresearch/train.py > autoresearch/run.log 2>&1
```

- The run uses a fixed training budget from `prepare.py`.
- The metric is `val_loss`.
- `train.py` may change:
  architecture flags, hyperparameters, phase target, batch size, LoopLM
  settings, relative attention settings, checkpoint choice, and any other
  search logic.
- `prepare.py` is fixed and acts as the evaluator.

## Practical guidance

- Optimize one phase at a time.
  Example:
  - stage 1 at `seq_len=4096` from scratch
  - stage 2 at `seq_len=8192` from the saved stage-1 checkpoint
- Prefer simple changes over complex ones when gains are marginal.
- Respect the local platform.
  This machine is not a GH200. A good 5-minute result on the Mac Studio is a
  proxy, not the final answer.
- Be careful with memory.
  A crash is a valid result and should be logged as `crash`.

## Logging

Record each result in [results.tsv](/Users/tannerfokkens/Documents/2pt-bach_update/autoresearch/results.tsv):

```tsv
commit	val_loss	memory_gb	status	description
```

Statuses:

- `keep`
- `discard`
- `crash`

If a run crashes, write `0.000000` for the metric and `0.0` for memory.
