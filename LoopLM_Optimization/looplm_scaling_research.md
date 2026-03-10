# Scaling Recurrent Depth in Looped Transformers: Research Synthesis

## Executive Summary

The Ouro/LoopLM paper (Zhu et al., arXiv 2510.25741v4, Nov 2025) is the most ambitious scaling study of looped transformers to date: 1.4B and 2.6B parameter models pretrained on 7.7T tokens with 4 recurrent steps, matching 4-8B dense baselines. Their key finding is that the practical limit is **not** a hard architectural ceiling but an **optimization/stability problem** that they partially solve via entropy-regularized adaptive exits and staged training. However, even they hit instability at 8 loops and settled on 4 — validating the research brief's core question.

This document synthesizes the Ouro paper's findings, the broader literature, and concrete recommendations for applying these ideas to a medium-scale domain-specific model (the Bach fugue generator, ~16M parameters).

---

## 1. What the Ouro/LoopLM Paper Actually Shows

### 1.1 Architecture
- Standard decoder-only Transformer with weight-tied layers applied recurrently
- Ouro 1.4B: 24 layers × 4 recurrent steps = 96 effective layers from 1.4B params
- Ouro 2.6B: 48 layers × 4 recurrent steps = 192 effective layers from 2.6B params
- Hidden dim 2048, MHA with RoPE, SwiGLU FFN, **sandwich RMSNorm** (critical for stability)
- Exit gate: `λ_t(x) = σ(Linear(h^(t)))` produces per-step exit probability

### 1.2 The Stability Problem They Encountered
**This is the most directly relevant section for our question.**

> "Our initial experiments with 8 recurrent steps in Stage 1a led to **loss spikes and gradient oscillations**. We hypothesize this stems from compounded gradient flow through multiple recurrent iterations, which can amplify small perturbations." (§4.3)

They reduced from 8 → 4 recurrent steps to fix this. Their mitigations:

| Mitigation | Detail |
|---|---|
| **Recurrent step reduction** | 8 → 4 steps to control instability |
| **Batch size scaling** | 4M → 8M tokens; larger batches = more stable gradient estimates through multiple iterations |
| **Lower learning rates** | Recurrent architectures empirically need smaller LR than standard transformers |
| **Conservative optimizer settings** | AdamW with weight decay 0.1, β₁=0.9, β₂=0.95, gradient clipping 1.0 |
| **KL coefficient reduction** | β for entropy regularization reduced from 0.1 (Stage 1) → 0.05 (later stages) to reduce conflicting gradients |
| **Sequence length progression** | 4K → 16K → 64K → 32K across stages |
| **Sandwich normalization** | RMSNorm before both attention and FFN sub-layers — "especially critical for deep recurrent computation" |

### 1.3 Extrapolation Behavior (Tables 10-13)

**Base models (1.4B and 2.6B), trained at T=4:**
- Performance monotonically improves from T=1 to T=4
- At T=5-8 (extrapolation): moderate but consistent degradation
- 1.4B degrades more sharply than 2.6B during extrapolation
- Example: 1.4B MMLU goes 41.21 → 60.43 → 66.71 → **67.45** → 66.64 → 65.77 → 65.28 → 64.49

**Thinking models (SFT'd):**
- Performance peaks at T=4 or T=5 for the 1.4B model
- 2.6B Thinking model peaks at T=3 or T=4 depending on benchmark
- Sharper extrapolation degradation than base models, especially on harder tasks
- AIME 2025 (1.4B): 0.33 → 25.00 → 43.33 → 46.30 → **47.00** → 43.00 → 41.00 → 38.00

**Key insight**: Safety alignment *improves* with extrapolation (more loops = safer), even as task performance degrades. This suggests the degradation is in fine-grained knowledge retrieval, not in the recurrent mechanism's reasoning ability.

### 1.4 The Training Objective

**Stage I: Entropy-regularized pretraining**
```
L = Σ p_φ(t|x) · L^(t) - β · H(p_φ(·|x))
```
- Expected task loss weighted by exit probability, minus entropy bonus
- Uniform prior over exit steps (not geometric) — prevents premature collapse to always using max depth
- β controls exploration vs exploitation: larger β → more uniform exit distribution
- Equivalent to ELBO with uniform prior (variational inference interpretation)

**Stage II: Focused adaptive gate training**
- Freeze LM parameters, train only the exit gate
- Gate learns to match "ideal continuation probability" based on actual loss improvement:
  - I^(t) = max(0, L^(t-1)_stop - L^(t)_stop)   (improvement from step t-1 to t)
  - w^(t) = σ(k · (I^(t) - γ))   with k=50, γ=0.005
- Binary cross-entropy between gate predictions and ideal labels
- Penalizes both underthinking (exiting when it should continue) and overthinking (continuing when gains have stalled)

### 1.5 What Looping Actually Buys

The paper's most surprising finding: **looping does NOT increase knowledge capacity** (bits per parameter remains ~2 regardless of loop count). Instead, it dramatically improves **knowledge manipulation** — the ability to compose and reason over stored knowledge.

- On the Mano task (modular arithmetic): looped models **always** outperform iso-parameter non-looped counterparts
- On multi-hop QA: looped models learn with fewer unique training samples
- On MMLU subcategories: biggest gains in Elementary Mathematics (+155%), Formal Logic (+143%), Logical Fallacies (+128%); smallest in Global Facts (+8%), Moral Scenarios (+8%)
- Theoretical result: LoopLM can solve graph reachability in O(log D) steps (vs O(n²) for discrete CoT, O(D) for continuous CoT)

### 1.6 Practical Inference Optimizations

**KV cache sharing during decoding:**
- Full 4× cache: baseline
- Last-step only cache: -0.07 on GSM8K, -2.0 on MATH-500, **4× memory reduction**
- First-step only cache: catastrophic failure (18.73 on GSM8K)
- Averaged cache: comparable to last-step
- Conclusion: final recurrent step's representations are most informative; earlier steps transform representations in ways that can't be approximated from step 1

**Speculative decoding built-in:**
- Text(R_s) from early step s proposes tokens; Text(R_T) from final step T verifies
- No external draft model needed — the architecture natively supports this

### 1.7 RL Attempts (Failed)
They tried RLVR (DAPO and GRPO) on the SFT checkpoint. **Neither worked:**
1. Off-policy rollouts: generated 4-step logits, simulated early exit by thresholding — off-policy mismatch, no improvement
2. Fixed 4-round RL: training progressed but didn't surpass SFT — limited headroom after extensive SFT, and models still used fewer rounds at inference than trained

This is an open problem: current RL infrastructure (vLLM/SGLang) assumes fixed execution paths, incompatible with variable-depth computation.

---

## 2. The Broader Literature: Why Loops Degrade

### 2.1 Universal Transformer (Dehghani et al., 2018)
- Weight-shared transformer layers applied up to T steps with ACT halting
- Tested on bAbI, machine translation, subject-verb agreement
- Improvement plateaus beyond ~6-8 steps on most tasks; ACT typically learned to halt at 2-5 steps
- Key contribution: showed dynamic halting per-position is possible
- Limitation: ACT adds the halting penalty λ as a fixed hyperparameter, leading to either premature exit or wasted computation

### 2.2 Deep Equilibrium Models (Bai et al., 2019-2020)
- Frame infinite-depth weight-tied networks as fixed-point problems: find z* such that z* = f_θ(z*, x)
- Use root-finding (Anderson acceleration, Broyden's method) instead of explicit iteration
- Implicit differentiation for backprop: no need to store intermediate states
- **Key insight for our problem**: If the recurrent block is a contraction mapping (spectral radius of Jacobian < 1), convergence is guaranteed. But contraction too strong → representations collapse. Contraction too weak → oscillation/divergence.
- DEQ models demonstrate that very deep effective depth is possible if the fixed-point solver converges
- Practical issue: convergence is not guaranteed during training (the mapping changes each step), and non-convergence causes training instability

### 2.3 PonderNet (Banino et al., 2021)
- Learns a halting distribution over computation steps via REINFORCE-like objective
- Uses a geometric prior on the halting distribution (parametrized by λ_p)
- ELBO-based objective similar to LoopLM's, but with geometric prior instead of uniform
- **Ouro/LoopLM directly compares to this**: uniform prior achieves lower final loss and more stable training than geometric priors (Appendix A, Figure 10). Geometric priors collapse probability mass onto shallow steps.

### 2.4 ALBERT (Lan et al., 2019)
- Cross-layer weight sharing for BERT-style models
- Showed weight sharing alone provides regularization and reduces parameters
- But performance degrades when sharing across too many layers — the model can't learn layer-specific functions
- This is the "parameter efficiency vs expressivity" tension

### 2.5 Relaxed Recursive Transformers (Bae et al., 2024)
- Key idea: inject **layer-wise LoRA adapters** across recursive steps
- Each loop gets the shared base block + a small per-step LoRA adapter
- Provides "partial per-step flexibility" without full parameter independence
- Converts standard pretrained models into recursive ones
- Referenced in the Ouro paper as [16]

### 2.6 Mixture-of-Recursions (Bae et al., 2025)
- Combines recursive parameter efficiency with adaptive, **token-level routing**
- Different tokens can take different numbers of loops
- More fine-grained than LoopLM's per-sequence exit decision
- Referenced in the Ouro paper as [22]

### 2.7 Coconut / Continuous Thought (Hao et al., 2024)
- Feeds hidden states from one step back as input tokens for the next
- Explicit "continuous thought" tokens in the sequence
- Makes the iterative refinement more interpretable
- LoopLM's approach is implicit: the refinement happens purely in the hidden state evolution h^(1) → h^(2) → ... → h^(T)

### 2.8 Scaling Up Test-Time Compute with Latent Reasoning (Geiping et al., 2025)
- "Recurrent depth" approach for scaling test-time compute
- Shows looped transformers can match much deeper non-looped models on reasoning tasks
- Referenced in the Ouro paper as [17]

---

## 3. Root Causes of Loop Degradation

Based on the paper and literature, the degradation at higher loop counts stems from multiple interacting mechanisms:

### 3.1 Gradient Amplification Through Shared Weights
When backpropagating through T applications of the same function f_θ, the gradient involves the product:
```
∂L/∂θ = Σ_{t=1}^{T} [Π_{s=t+1}^{T} J_s] · ∂L^(t)/∂h^(t) · ∂f/∂θ
```
where J_s is the Jacobian of f_θ at step s. The product of Jacobians either:
- **Explodes** if spectral radius > 1 → loss spikes, NaN
- **Vanishes** if spectral radius < 1 → later loops don't learn
- **Oscillates** if eigenvalues are complex with magnitude ≈ 1

The Ouro team observed "loss spikes and gradient oscillations" at 8 steps, consistent with this.

### 3.2 Representation Drift / Oversmoothing
Repeated application of attention + FFN tends to:
- Push all token representations toward a common subspace (oversmoothing, well-documented in GNN literature)
- Lose position-specific information over many iterations
- Converge to a fixed point that may not be the optimal one for the task

The PCA analysis in Figure 8b shows that representations *do* change meaningfully across steps (benign vs harmful prompts separate better at step 4 than step 1), but this is at T=4. Beyond the training horizon, this refinement likely degrades.

### 3.3 Optimization Landscape Interaction
The entropy-regularized objective creates a multi-objective optimization:
- The task loss wants deep loops to contribute
- The entropy term resists collapse to any single depth
- These gradients can conflict, especially early in training

The paper found reducing β from 0.1 to 0.05 in later stages helped — the conflicting gradients were causing instability.

### 3.4 Exit Gate Collapse
Without entropy regularization, the exit distribution p_φ(t|x) collapses to always choosing t = T_max:
> "Under naive gradient descent on the next-token prediction loss... self-reinforcement collapses p_φ onto t = T_max" (§3.3)

This means the model never learns to use intermediate depths usefully, making all loops "mandatory" rather than adaptive.

---

## 4. Concrete Recommendations for the Bach Fugue Generator

Your model is ~16M parameters, domain-specific, with limited data. Here are recommendations ranked by expected impact and implementation difficulty.

### Tier 1: High Impact, Moderate Effort

#### 4.1 Entropy-Regularized Exit Gate (from the paper)
**What**: Add a learned exit gate and train with the LoopLM objective (Eq. 4).
**Why**: This is the single most validated technique in the paper. It:
- Prevents exit distribution collapse
- Allows the model to use fewer loops for simple passages (homophonic, stepwise motion) and more for complex ones (fugal entries, modulatory passages)
- Provides an explicit signal for when additional loops help

**Implementation for bach-gen**:
1. Add a `Linear(d_model, 1)` exit gate after the final layer of each loop
2. Compute per-step exit probability: `λ_t = σ(gate(h^(t)_last))`
3. Compute survival probability and exit distribution as in Eq. 3
4. Training loss: `L = Σ p_φ(t|x) · L^(t) - β · H(p_φ(·|x))`
5. Start with β=0.1, reduce to 0.05 after initial training

**Confidence: 85%** — the technique is well-validated at scale; the main uncertainty is whether it transfers cleanly to small models.

#### 4.2 Sandwich Normalization
**What**: Place RMSNorm before *both* attention and FFN sub-layers (not just pre-norm).
**Why**: The Ouro team calls this "especially critical for deep recurrent computation." Standard pre-norm places normalization before the attention only. Sandwich norm normalizes the hidden state more aggressively, preventing drift across loops.

**Implementation**: Already using RMSNorm in BachTransformer; just ensure it's applied before both sub-layers in each transformer block.

**Confidence: 90%** — simple change, well-motivated, almost certainly helps.

#### 4.3 Depth Curriculum Training
**What**: Start training with fewer loops, gradually increase to the target count.
**Why**: The Ouro team started at 8 and reduced to 4 due to instability. The reverse approach — starting low and increasing — would allow the model to first learn good single-pass representations, then learn to refine them iteratively.

**Suggested schedule for bach-gen**:
1. Phase 1 (broad pretraining): T=1 (no looping) — learn basic music representations
2. Phase 2 (add looping): T=2, gradually increase to T=4 — learn iterative refinement
3. Phase 3 (Bach fine-tuning): T=4 with curriculum learning + exit gate

**Confidence: 75%** — well-motivated but not directly tested in the paper (they did the opposite: started high, reduced). May need tuning.

#### 4.4 Conservative Optimizer Settings for Recurrence
**What**: Use lower learning rate, higher weight decay, gradient clipping, and larger effective batch size when looping.
**Why**: Directly from the paper's lessons. Recurrent architectures need more conservative optimization because gradients flow through the same parameters multiple times.

**Specific settings**:
- AdamW with weight decay 0.1 (vs typical 0.01)
- β₁=0.9, β₂=0.95 (vs typical β₂=0.999)
- Gradient norm clipping at 1.0
- If possible, increase effective batch size (gradient accumulation)

**Confidence: 90%** — these are directly validated in the paper.

### Tier 2: High Impact, Higher Effort

#### 4.5 Two-Stage Gate Training
**What**: After pretraining with entropy-regularized exits, freeze the LM and fine-tune only the exit gate using the adaptive loss (Eq. 6).
**Why**: The paper shows this provides 2-3% accuracy gains at all compute budgets over the entropy-regularized gate alone (Figure 5). The gate learns to base exit decisions on *actual measured improvement*, not just the entropy-regularized training signal.

**Implementation**:
1. After main training, run a Stage II where only the gate's Linear layer is trainable
2. Compute per-step loss improvement I^(t) = max(0, L^(t-1)_stop - L^(t)_stop)
3. Compute ideal continuation label w^(t) = σ(50 · (I^(t) - 0.005))
4. Train gate to match via binary cross-entropy

**Confidence: 80%** — well-validated in the paper but adds complexity.

#### 4.6 Per-Step LoRA Adapters (from Relaxed Recursive Transformers)
**What**: Instead of perfectly identical weight sharing, add small per-step LoRA adapters.
**Why**: Gives each loop step some unique parameters while keeping the bulk of computation shared. This directly addresses the "all loops do exactly the same thing" problem — early loops can specialize in different refinements than later loops.

**Implementation**:
- Keep the shared base transformer block
- For each recurrent step t, add LoRA adapters (rank 4-16) to Q, K, V, and FFN projections
- Total extra parameters: tiny (rank × 2 × d_model × num_adapters × T)

**Confidence: 70%** — promising but not validated at the same scale as LoopLM. For a 16M param model, the LoRA overhead is proportionally larger.

#### 4.7 Input Injection / Residual Across Loops
**What**: At each loop iteration, re-inject the original input embedding (or a fraction of it) into the hidden state.
**Why**: Prevents the hidden state from drifting too far from the input signal over many loops. Similar to how residual connections work within a single transformer block, but applied across the loop dimension.

**Implementation**:
```python
h = embed(x)
h_input = h.clone()
for t in range(T):
    h = transformer_block(h)
    h = h + α * h_input  # input injection with learnable or fixed α
```

**Confidence: 65%** — theoretically motivated (prevents oversmoothing) but not tested in the LoopLM paper. May interfere with the iterative refinement that makes loops useful.

### Tier 3: Experimental / Diagnostic

#### 4.8 Spectral Monitoring During Training
**What**: Periodically compute the spectral radius of the recurrent block's Jacobian and log it.
**Why**: If spectral radius drifts above 1.0, you'll see it before loss spikes occur. Can be used to trigger learning rate reductions or other interventions.

**Implementation**: Sample a batch, compute Jacobian via `torch.autograd.functional.jacobian` on the block output w.r.t. input, compute leading eigenvalue. Log every N steps.

**Confidence: 60%** — expensive to compute for the full model; may need to approximate (e.g., power iteration on a subset of dimensions).

#### 4.9 Hidden State Difference Monitoring
**What**: Track `‖h^(t) - h^(t-1)‖₂` across loops during training and inference.
**Why**: The paper shows (Figure 5) that hidden state difference threshold is a surprisingly competitive early-exit heuristic, performing within 1-2% of the trained gate. If this converges to near-zero early, additional loops are wasted. If it oscillates, you have an instability problem.

**Implementation**: Trivial — just log the L2 norm of the difference at each loop step.

**Confidence: 90%** — this is a diagnostic, not a fix, but extremely useful for understanding what's happening.

#### 4.10 Fixed-Point Regularization
**What**: Add a regularization term that encourages the hidden state to converge:
```
L_fp = α · ‖h^(T) - f_θ(h^(T))‖²
```
**Why**: If the final hidden state is close to a fixed point of the recurrent block, it means the model has "converged" and additional loops would be redundant. This is the DEQ-inspired approach.

**Confidence: 55%** — theoretically elegant but may conflict with the iterative refinement objective. The whole point of loops is that each step should improve the prediction; a fixed-point penalty might prevent this.

---

## 5. Specific Architecture Translation: LoopLM → BachTransformer

### Current BachTransformer Architecture
- ~16M parameters, 8 layers, 256 embed dim, 8 heads
- RoPE, SwiGLU, RMSNorm, pre-norm, GQA option
- Trained on ~300M tokens

### Proposed LoopLM-Enhanced Architecture

```
BachTransformerLooped:
  embed(x)                          # token + position embedding
  h = embed_norm(embed(x))         # initial normalization

  for t in 1..T_max:
    h = shared_transformer_block(h)  # 8 layers, weight-tied across loops
    λ_t = σ(exit_gate(h[:, -1, :]))  # exit probability from last token
    L_t = cross_entropy(lm_head(h))  # per-step loss

  # Training: entropy-regularized expected loss
  # Inference: exit when CDF exceeds threshold q
```

### Hyperparameter Suggestions

| Parameter | Standard | LoopLM-adapted |
|---|---|---|
| Recurrent steps (T_max) | 1 (no loops) | 4 (train), up to 6 (inference) |
| Learning rate | 3e-4 | 1e-4 to 3e-4 with slower warmup |
| Weight decay | 0.01 | 0.1 |
| β₂ (AdamW) | 0.999 | 0.95 |
| Gradient clip | 1.0 | 1.0 |
| β (entropy coeff) | N/A | 0.1 → 0.05 |
| Batch size | as is | 2-4× larger if possible |
| Exit threshold q | N/A | 0.5 (tunable at inference) |

### Training Schedule

1. **Pretrain (broad corpus, no loops)**: Train standard BachTransformer (T=1) on full corpus. This establishes good musical representations.

2. **Enable loops with entropy regularization**: Switch to T=4 with exit gate. Use entropy-regularized loss with β=0.1, uniform prior. Monitor hidden state differences.

3. **Fine-tune on Bach with loops**: Continue with curriculum learning on Bach-specific data. Reduce β to 0.05.

4. **Gate refinement**: Freeze LM, train only exit gate with adaptive loss for a small number of steps.

### What This Buys for Music Generation

- **Simple passages** (scalar runs, repeated patterns, homophonic textures): model can exit early at T=1-2, saving computation
- **Complex passages** (fugal entries, stretto, modulations, voice crossings): model uses T=3-4 for iterative refinement
- **The evaluation scorer dimensions**: more loops of refinement should particularly help voice leading, contrapuntal quality, and structural coherence — exactly the "knowledge manipulation" that LoopLM excels at

---

## 6. Key Open Questions

### 6.1 Can loops scale beyond 4?
The Ouro team settled on 4 after instability at 8. They didn't try 5, 6, or 7 systematically. With depth curriculum (start at 2, increase to 8 gradually), it may be possible to push further. The theoretical argument (O(log D) steps for graph reachability) suggests more loops should help for harder reasoning, but the optimization challenges grow.

### 6.2 Does the advantage hold for small models?
All LoopLM results are at 1.4B+ parameters. At 16M, the parameter-sharing benefit is different: you're not trying to match a 64M parameter model, you're trying to squeeze more capability out of 16M. The knowledge manipulation advantage should still hold (it was demonstrated on 1M-40M GPT-2 models in the physics-of-LMs experiments, Figure 6), but the training stability challenges may be different.

### 6.3 Is the exit gate worth it for generation (autoregressive decoding)?
During generation, the exit gate adds overhead at every token. For music generation with relatively short sequences (~2048 tokens), this overhead is small. The compute savings from early exit on easy tokens could be significant if many tokens are "easy" (repeated patterns, held notes, etc.).

### 6.4 How does looping interact with DroPE?
DroPE removes positional embeddings for length generalization. In a looped model, positional encoding interacts with the recurrence: RoPE is applied at each loop iteration to the same sequence positions. This could create an interesting interaction — the model sees the same position T times through the same RoPE-enhanced attention. Worth investigating whether DroPE recalibration needs modification for looped architectures.

### 6.5 RL for LoopLM
The Ouro team's RL attempts failed. The fundamental issue is that current RL infra assumes fixed computation graphs. For a music generation model where quality evaluation is expensive (running the 7-dimension scorer), this is even harder. However, the scorer provides a natural reward signal — if someone solves the variable-depth RL problem, it could be very powerful for music generation.

---

## 7. Implementation Priority Checklist

For integrating LoopLM ideas into BachTransformer, in recommended order:

- [ ] **Sandwich RMSNorm**: Ensure RMSNorm before both attention and FFN (minimal change, high impact for stability)
- [ ] **Hidden state monitoring**: Add logging of ‖h^(t) - h^(t-1)‖₂ across loops (diagnostic, near-zero effort)
- [ ] **Weight-tied loop mechanism**: Make the 8-layer block reusable T times with configurable T
- [ ] **Conservative optimizer settings**: weight decay 0.1, β₂=0.95, gradient clip 1.0
- [ ] **Exit gate + entropy-regularized training**: Add the gate, implement Eq. 4, train with β=0.1
- [ ] **Depth curriculum**: Start T=1, increase to T=4 during training
- [ ] **Stage II gate training**: Freeze LM, fine-tune gate with adaptive loss (Eq. 6)
- [ ] **Inference-time q threshold tuning**: Sweep q ∈ {0.3, 0.5, 0.7, 0.9} for quality/speed tradeoff
- [ ] **Per-step LoRA adapters** (experimental): Add rank-4 LoRA per recurrent step
- [ ] **Spectral radius monitoring** (diagnostic): Track Jacobian eigenvalues during training

---

## 8. Key References from the Ouro Paper

| Ref | Paper | Relevance |
|---|---|---|
| [7] | Saunshi et al. 2025, "Reasoning with latent thoughts: On the power of looped transformers" | Shows looped transformers match much deeper models on reasoning |
| [15] | Dehghani et al. 2018, "Universal Transformers" | Original ACT + weight sharing for transformers |
| [16] | Bae et al. 2024, "Relaxed Recursive Transformers" | Per-step LoRA adapters |
| [17] | Geiping et al. 2025, "Scaling up test-time compute with latent reasoning" | Recurrent depth approach, sandwich normalization |
| [22] | Bae et al. 2025, "Mixture-of-Recursions" | Token-level adaptive recursive depth |
| [27] | Hao et al. 2024, "Coconut" — continuous latent reasoning | Explicit continuous thought tokens |
| [32] | Banino et al. 2021, "PonderNet" | ELBO-based adaptive halting (geometric prior) |
| [67] | Allen-Zhu & Li 2025, "Physics of LMs Part 3.3" | Knowledge capacity scaling laws |
| [68] | Allen-Zhu 2025, "Physics of LMs Part 4.1" | Architecture design and canon layers |
| [69] | Yao et al. 2025, "Multi-hop reasoning" | Knowledge manipulation tasks |
