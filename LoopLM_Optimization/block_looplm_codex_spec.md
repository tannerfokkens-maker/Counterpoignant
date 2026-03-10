# Implement Block-LoopLM in `bach_gen`

## Objective

Convert the current full-stack LoopLM implementation into a **block-LoopLM**:

- run a **front** layer stack once,
- run a **middle recurrent core** for `num_recurrent_steps`,
- run a **back** layer stack once for final decoding,
- keep the existing LoopLM training flow (`all_logits`, `exit_lambdas`, Stage II gate training) as intact as possible.

The goal is to stop looping the entire transformer stack and instead loop only a specific internal block.

## Why

The current model loops `self.layers` as a whole in `bach_gen/model/architecture.py`. That is a clean LoopLM baseline, but it likely over-loops input/output-facing layers that should only run once. My working hypothesis is:

- extra recurrent steps help most when applied to an internal refinement block,
- looping the full stack adds unnecessary optimization burden and memory pressure,
- a front/core/back split should make recurrence more stable and more meaningful,
- adding a loop-step embedding should reduce drift/oscillation across recurrent passes.

I want the architecture to support testing whether a **specific middle block** is the right recurrent operator.

---

## Current codebase facts

Relevant files:

- `src/bach_gen/model/config.py`
- `src/bach_gen/model/architecture.py`
- `src/bach_gen/model/trainer.py`

Current architecture behavior:

- `ModelConfig` already has:
  - `num_recurrent_steps`
  - `looplm_sandwich_norm`
  - `looplm_exit_gate`
  - `looplm_kl_beta`
  - `looplm_exit_threshold`
- `TransformerModel.__init__()` currently creates one flat `self.layers`
- `_forward_single()` runs all layers once
- `_forward_looped()` re-runs the whole `self.layers` stack for each recurrent step
- the exit gate currently reads the hidden state after the full recurrent stack
- trainer logic already expects `LoopLMOutput(all_logits, exit_lambdas, final_logits)`

This means the trainer is already close to what we need. The core implementation change is architectural.

---

## Required architecture change

Replace the current implicit recurrence:

```text
embed -> [all layers] -> [all layers] -> [all layers] -> ln_final -> head
```

with:

```text
embed -> front_layers -> recurrent_core x T -> back_layers -> ln_final -> head
```

Important rule:

- **Only the recurrent core is fed back into the next loop.**
- `back_layers` are **not** part of the recurrent state transition.

So recurrence should behave like:

```text
core_state_(t+1) = R(core_state_t)
```

not:

```text
core_state_(t+1) = R(G(core_state_t))
```

where `G = back_layers`.

---

## Implementation tasks

### 1) Update `ModelConfig`

In `src/bach_gen/model/config.py`, add explicit layer split config:

- `num_front_layers: int`
- `num_loop_layers: int`
- `num_back_layers: int`
- `loop_step_embedding: bool = True`
- `loop_per_step_norms: bool = False`  
  This can be a placeholder flag for later if not used immediately.

Validation requirements:

- `num_front_layers + num_loop_layers + num_back_layers == num_layers`
- `num_loop_layers >= 1`
- if `num_recurrent_steps <= 1`, model should still work as a standard non-recurrent transformer

Update `num_params` estimation to account for:

- front layers
- loop layers
- back layers
- optional loop-step embedding params if enabled
- exit gate params as before

Do **not** change existing config field names unnecessarily.

---

### 2) Split the transformer stack in `architecture.py`

In `TransformerModel.__init__()`:

Replace:

```python
self.layers = nn.ModuleList([...])
```

with:

- `self.front_layers`
- `self.loop_layers`
- `self.back_layers`

Each should be a `nn.ModuleList[TransformerBlock]` with lengths driven by config.

Also add an optional loop-step embedding:

- `self.loop_step_embed = nn.Embedding(config.num_recurrent_steps, config.embed_dim)` when enabled and `num_recurrent_steps > 1`
- else `self.loop_step_embed = None`

Keep:

- `self.token_embed`
- `self.embed_dropout`
- `self.ln_final`
- `self.head`
- weight tying behavior
- `self.exit_gate`

---

### 3) Introduce a reusable stack runner helper

Add a helper that can run an arbitrary layer stack, instead of assuming one flat `self.layers`.

Suggested shape:

```python
_run_layer_stack(
    layers,
    x,
    causal_mask,
    pad_mask,
    cos,
    sin,
    use_pos,
    attn_temperature,
    use_cache=False,
    kv_cache=None,
    cache_read_only=False,
)
```

Requirements:

- must work both with and without KV cache
- must return either `x` or `(x, new_caches)`
- must preserve current `TransformerBlock.forward(...)` contract

This helper should be used by both `_forward_single()` and `_forward_looped()`.

---

### 4) Change `_forward_single()`

When `num_recurrent_steps <= 1`, the model should behave as a standard transformer using:

```text
front_layers -> loop_layers -> back_layers
```

in that order, exactly once each.

So `_forward_single()` should:

1. run `front_layers`
2. run `loop_layers`
3. run `back_layers`
4. apply `ln_final`
5. apply `head`

It should still support `use_cache=True`.

---

### 5) Rewrite `_forward_looped()` to loop only the middle block

New recurrent structure:

1. `x = run(front_layers, x)` once
2. `core_state = x`
3. for each recurrent step `t`:
   - optionally add loop-step embedding to the input state
   - `core_state = run(loop_layers, core_state)`
   - compute exit gate on `core_state` (not on final decoded state)
   - if `return_all_steps=True`, decode per-step logits by running `back_layers` on `core_state`, then `ln_final`, then `head`
4. after the chosen/final loop, run `back_layers` once more for final output if needed
5. return standard logits or `LoopLMOutput`

### Important behavioral requirements

#### Exit gate placement

The exit gate should read from the **post-core recurrent hidden state**:

```python
lam = sigmoid(exit_gate(core_state))
```

not from the output after `back_layers`.

#### Per-step logits

Per-step logits for LoopLM loss should come from:

```text
back_layers(core_state_t) -> ln_final -> head
```

This allows the trainer to keep using actual token prediction loss at each recurrent depth.

#### Recurrent state

The state fed into the next iteration must be only the core hidden state.

Do **not** feed `back_layers(...)` output back into recurrence.

#### Adaptive early exit

Preserve current inference-time early exit behavior conceptually:

- only active in eval mode
- only active when `self.exit_gate is not None`
- only active when not `return_all_steps`
- threshold based on `config.looplm_exit_threshold`

But make it operate over the new block-looped state.

---

### 6) Add loop-step embedding support

If enabled, inject a learned embedding per recurrent step before running the loop block.

Simple version is fine:

```python
step_input = core_state
if self.loop_step_embed is not None:
    step_vec = self.loop_step_embed.weight[t].view(1, 1, -1)
    step_input = step_input + step_vec
core_state = run(loop_layers, step_input)
```

This is intended to let the tied recurrent block behave differently at loop 1 vs loop 2 vs loop 3 without fully untying weights.

Use a conservative implementation. No need to overcomplicate this.

---

### 7) Update KV cache handling

Current cache logic assumes a flat recurrent stack.

Refactor cache handling to support three regions:

- front cache
- loop cache
- back cache

A lightweight solution is acceptable. Two viable approaches:

#### Preferred
Define a small structured cache container, e.g.:

```python
@dataclass
class BlockLoopKVCache:
    front: list[KVCache]
    loop: list[KVCache]
    back: list[KVCache]
```

#### Acceptable fallback
Keep a flat list but slice it consistently into front/loop/back ranges internally.

### Required cache behavior

Preserve the current LoopLM cache strategy for the loop block:

- non-final recurrent steps may read from cache but do not write
- final recurrent step reads and writes

Front and back stacks should behave like normal transformer stacks.

Do not silently break generation.

---

### 8) Trainer changes: keep minimal

The trainer already expects:

- `output.all_logits`
- `output.exit_lambdas`
- `output.final_logits`

Try very hard **not** to rewrite trainer logic unless necessary.

Expected trainer impact:

- `_compute_looplm_loss()` should continue to work if model output shape/contracts stay the same
- `train_exit_gate()` should continue to work if `exit_lambdas` remain aligned with recurrent steps

However, update any optimizer parameter-group code or comments that explicitly assume `self.layers` exists.

Search for references like:

- `layers.*`
- assumptions about sandwich norms on a flat layer list
- any code that freezes/unfreezes parameters by name

Make parameter grouping robust to:

- `front_layers.*`
- `loop_layers.*`
- `back_layers.*`

---

## Recommended implementation order

1. add config fields and validation
2. split the model into front/loop/back stacks
3. add generic `_run_layer_stack(...)`
4. update `_forward_single()`
5. update `_forward_looped()`
6. adapt cache structure
7. fix trainer parameter-group assumptions if needed
8. run tests / smoke checks

---

## Acceptance criteria

The implementation is successful if all of the following are true:

### Functional

- model can run with `num_recurrent_steps = 1`
- model can run with `num_recurrent_steps > 1`
- standard forward path works
- LoopLM forward path works
- `return_all_steps=True` returns valid `LoopLMOutput`
- early exit still works in eval mode
- generation with KV cache still works

### Architectural

- only `loop_layers` are repeated across recurrent steps
- `front_layers` and `back_layers` run once per forward pass
- exit gate reads from recurrent core hidden state
- per-step logits are decoded through `back_layers`

### Stability / compatibility

- existing trainer does not require a major rewrite
- parameter freezing for exit-gate training still works
- no unrelated behavior changes to tokenization, generation constraints, or scoring

---

## Suggested smoke tests

Please add or run small smoke tests covering at least these cases:

### Case 1: non-recurrent

- `num_recurrent_steps = 1`
- small random input batch
- verify output logits shape is correct

### Case 2: recurrent with all steps

- `num_recurrent_steps = 3`
- `return_all_steps=True`
- verify:
  - `len(output.all_logits) == 3`
  - `len(output.exit_lambdas) == 2` or equivalent current contract
  - final logits shape is correct

### Case 3: cache path

- single-token incremental decode with cache
- verify no shape regressions and no crash

### Case 4: early exit

- eval mode
- `looplm_exit_threshold < 1.0`
- verify the loop can terminate before `T_max`

---

## MVP boundary

If the full cache refactor is too invasive, prefer this MVP:

- implement the architectural split cleanly,
- keep training compatibility,
- make non-cached forward fully correct first,
- then adapt cache code with minimal API breakage.

But do **not** merge a version that breaks generation silently.

---

## Non-goals

Do not do any of the following in this task unless absolutely required:

- redesign the trainer from scratch
- add LoRA/adapters/per-step norms yet
- change the tokenizer or data pipeline
- change evaluation metrics
- rewrite unrelated modules

This task is specifically about converting full-stack recurrence into **middle-block recurrence** while preserving the current LoopLM workflow.

---

## Deliverables

Please produce:

1. the code changes
2. a short summary of what changed
3. any assumptions made
4. any follow-up work you think is needed
5. a note on whether cache support remained fully intact or whether any limitation remains

