# PoPE pre-training implementation evaluation

This note evaluates the current PoPE implementation against the intended mechanics in the PoPE paper workflow (position-content decoupling in Q/K during pre-training).

## What is implemented well

1. **PoPE is integrated directly in attention Q/K, not V**
   - `CausalSelfAttention` applies PoPE to query/key tensors only, preserving standard attention semantics.
2. **Content/position decoupling exists**
   - `apply_pope_emb` uses `softplus` for magnitudes and trigonometric phase rotation per position.
3. **Per-dimension phase schedule is implemented**
   - `PoPEEmbedding` creates one inverse frequency per head dimension.
4. **Pre-training defaults to PoPE**
   - `ModelConfig.pos_encoding` and CLI `--pos-encoding` default to `pope`, so the main training path exercises PoPE by default.
5. **Ablation path is available**
   - The same pipeline can run `rope`, `pope`, or `none`, which is necessary for controlled comparisons.

## Gaps / risks for faithful PoPE pre-training

1. **No explicit PoPE-focused tests**
   - There are currently no unit tests validating PoPE tensor shapes, numerical properties, or equivalence expectations.
2. **Attention scaling changed implicitly by dimension doubling**
   - PoPE maps Q/K from `D` to `2D`; PyTorch SDPA then scales by `1/sqrt(2D)`. If the paper assumes a different scale (often effectively tied to original head dim), this can alter optimization dynamics.
3. **No paper-level diagnostics in training loop**
   - The training path does not automatically report PoPE-specific checks (e.g., long-context behavior, stability metrics under larger sequence lengths).
4. **Terminology/config coupling**
   - PoPE uses `rope_theta` as its base frequency parameter name. Functionally okay, but easy to misconfigure/interpret during experiments.

## Practical score

- **Implementation completeness (core mechanics): 8/10**
- **Experimental faithfulness/reproducibility (paper-style): 5/10**
- **Production robustness for pre-training: 6/10**

Overall: **technically solid core PoPE integration, but under-instrumented for claiming close paper-faithful pre-training behavior.**

## Highest-impact improvements

1. Add PoPE unit tests for:
   - output shapes (`D -> 2D`),
   - deterministic no-position variant behavior,
   - finite output checks over long sequences.
2. Add an optional explicit attention scale for PoPE (e.g., scale by original `head_dim`) to compare with current SDPA implicit scaling.
3. Add a reproducible benchmark script/config that trains `rope` vs `pope` under matched seeds and logs convergence + validation loss.
4. Rename `rope_theta` to a neutral positional base (or alias it) to reduce experiment confusion.

## Evidence inspected

- `src/bach_gen/model/architecture.py`
- `src/bach_gen/model/config.py`
- `src/bach_gen/cli.py`
- `docs/training-pipeline.md`
