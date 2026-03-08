# Relative Attention Summary

## Why this feature was added

The current model already shows strong short-range imitation, especially across nearby voices and short spans. The observed weakness is long-range thematic return: motifs tend to echo locally but are less likely to reappear after a long gap or in a structurally meaningful later section.

PoPE plus DroPE already pushes the model toward weaker dependence on absolute token position:

- `PoPE` gives a strong positional scaffold during pre-training
- `DroPE` removes explicit positional application later so the model can rely more on content, rhythm, and causal context

That helps with position invariance, but it does not explicitly model pairwise distance between the current token and earlier material. Music Transformer-style relative attention does model that pairwise relationship directly, which makes it a good fit for the specific failure mode: weak long-range motif recall despite strong short-range imitation.

## What the existing architecture looked like before this change

The model is a decoder-only Transformer with:

- token embeddings
- pre-norm RMSNorm blocks
- multi-head self-attention
- SwiGLU feed-forward layers
- tied input/output embeddings

The existing positional path was:

1. Project token embeddings into `Q`, `K`, and `V`
2. Apply either `RoPE` or `PoPE` to `Q` and `K`
3. Run PyTorch `scaled_dot_product_attention`
4. During DroPE phases, skip explicit positional application (`use_rope=False`) while keeping the rest of the model unchanged

This design is still intact. Relative attention was added as an additional logit term inside attention; it does not replace PoPE, RoPE, or DroPE.

## What was implemented

### 1. Optional relative attention in model config

`ModelConfig` now includes:

- `rel_attn_bias: bool`
- `rel_attn_max_distance: int`

This keeps the feature fully opt-in. Baseline models without relative attention are unchanged.

### 2. Music Transformer-style relative attention

The implementation now follows the Music Transformer paper more closely than the initial bucketed-bias prototype.

For each head:

- learn relative embeddings `E_r`
- compute relative logits `S_rel = Q E_r^T`
- transform those logits into query-key indexing using the paper's skew / relative-shift trick
- add the resulting `S_rel` to the normal attention logits before softmax

This is the key difference from the earlier prototype:

- prototype: learned scalar bias per distance bucket
- final implementation: learned relative embeddings that interact with the query vectors directly

That matches the main paper idea and is much closer to the mechanism that improved long-range structure in Music Transformer.

### 3. Minimal invasion into the current code path

The change was deliberately constrained to the attention layer and config surface:

- no data-preparation changes
- no tokenizer changes
- no sequence regeneration
- no changes to conditioning tokens
- no changes to the PoPE / DroPE curriculum logic

The existing training pipeline still works the same way:

1. broad pre-train with staged context lengths
2. DroPE recalibration
3. Bach fine-tune with the same staged schedule

Relative attention simply adds another inductive bias inside attention for all of those phases if enabled.

### 4. Cache-aware generation support

The project already relies on KV-cache decoding for generation speed. That means relative attention had to work in two regimes:

- full-sequence training / evaluation
- incremental cached generation

For the full-sequence path, the paper-style skew formulation is used directly when the active sequence length is within the learned relative range.

For cached decoding and contexts longer than the learned range:

- the same learned relative embeddings are used
- distances are clipped to the maximum configured range
- logits are gathered from the relative embedding bank using absolute query/key positions from the cache state

This is the practical compromise that keeps the architecture compatible with the current cache interface and long-context training schedule.

## Why `rel_attn_max_distance` matters

Relative attention is not free. With this formulation, the learned relative embedding bank scales with:

- number of heads
- effective head dimension
- maximum relative distance
- number of layers

At the current `384d / 8h / 9L / PoPE` setup, the parameter count increases quickly:

- baseline: about `15.99M`
- `rel_attn_max_distance=512`: about `19.53M`
- `1024`: about `23.07M`
- `2048`: about `30.15M`
- `4096`: about `44.31M`
- `16384`: about `129.24M`

So the feature is useful, but the relative range must be chosen intentionally. A moderate distance such as `1024` or `2048` is the sensible starting point.

## How this fits with PoPE and DroPE

This feature is compatible with the current positional strategy.

During pre-training:

- `PoPE` still shapes `Q` and `K`
- relative attention adds learned distance-dependent logits on top

During DroPE:

- explicit PoPE application is disabled, as before
- relative attention remains active

So the combined interpretation is:

- `PoPE` helps optimization and local positional structure early
- `DroPE` reduces over-anchoring to absolute position later
- relative attention gives the model an explicit learned notion of "how far away" prior material is

That is exactly the hybrid we want for a model that already imitates well locally but underuses material over longer spans.

## Checkpoint and resume behavior

Old checkpoints needed to remain usable. The trainer now reconciles optional relative-attention parameters when loading checkpoints:

- old checkpoint -> new relative-attention model: missing relative parameters are initialized from the target model
- relative-attention checkpoint -> non-relative model: extra relative parameters are dropped

If optional relative-attention parameters are added or removed during load, optimizer state is not restored, because the parameter set no longer matches cleanly.

For real training runs, resume with the same relative-attention flags you used to start the run.

## Testing that was added

Focused tests were added in two areas:

- KV-cache equivalence:
  full forward must match prefill + incremental decode with relative attention enabled
- checkpoint compatibility:
  models with and without relative attention must load older or newer checkpoints without crashing

These tests passed after the implementation.

## Pipeline and docs updates

The training pipeline doc now includes:

- the baseline 15.9M command unchanged
- an optional relative-attention training command
- parameter-count tradeoffs for different `rel_attn_max_distance` settings
- guidance to start at `1024` or `2048`

## Recommended next step

Train one serious run with:

- the current curriculum
- PoPE + DroPE still enabled
- `--rel-attn-bias`
- `--rel-attn-max-distance 1024` or `2048`

Then compare:

- long-range thematic recall
- manual listening for delayed subject return
- whether short-range imitation remains as strong as before

If long-range motif reuse still underperforms after this, the next missing ingredient is probably not another positional variant. It is higher-level structure: explicit bar-level planning, subject-entry scheduling, or retrieval/memory over earlier thematic material.
