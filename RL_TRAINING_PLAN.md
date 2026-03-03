# Reinforcement Learning Plan: ReST → GRPO

## Overview

Close the loop between generation and scoring. The model currently generates 100+ candidates and picks the best — RL trains the model to generate winners directly instead of relying on brute-force sampling.

Two phases:
- **Phase 1 (ReST):** Rejection sampling + fine-tuning. Simple, immediate gains.
- **Phase 2 (GRPO):** Group-relative policy optimization. Richer gradient signal, more sample-efficient.

Both phases use section-level scoring for denser reward signals.

---

## Section-Level Scoring

### The problem with sequence-level rewards

A 4096-token fugue takes ~2000 decisions. A single composite score at the end gives the model no signal about which decisions mattered. Did the score drop because of bad voice leading in bar 12, or because the subject never returned after bar 20? The model can't tell.

### Section boundaries

Split sequences at natural musical boundaries. These are already detectable from the token stream:

1. **BAR tokens.** Every BAR token is a candidate boundary. Group into sections of 4 bars (the standard phrase length in tonal music).
2. **Cadence tokens.** Once cadence conditioning is implemented (per the conditioning plan), every `CAD_PAC`/`CAD_HC`/`CAD_DC` marks a section boundary.
3. **Voice entry points.** In fugues, a new voice entering (first note in a previously silent voice) marks a structural boundary — end of one exposition entry, start of the next.

Use 4-bar groupings as the default, with cadence/entry boundaries overriding when available.

### Section-level scorer

Score each section independently on the dimensions that can be evaluated locally:

| Dimension | Section-level? | Notes |
|-----------|---------------|-------|
| Voice leading | Yes | Parallel fifths, crossings, leaps — all local |
| Contrapuntal | Mostly | Register consistency, melodic coherence, rhythmic complementarity — all local. Sequential patterns need ≥2 sections |
| Thematic recall | Partially | Can check if subject/fragments appear in this section. Full thematic recall needs the whole piece |
| Statistical | Yes | Interval bigram distributions per section |
| Structural | Partially | Cadence quality at section boundaries. Key consistency local |
| Completeness | No | Only meaningful for full sequence |

Implementation:

```python
def score_section(
    comp: VoiceComposition,
    section_start_tick: int,
    section_end_tick: int,
    subject_intervals: list[int] | None = None,
    key_root: int = 0,
    key_mode: str = "major",
) -> float:
    """Score a single section of a composition.
    
    Returns a scalar reward for this section.
    """
    # Extract notes within the section window
    section_comp = extract_section(comp, section_start_tick, section_end_tick)
    
    # Voice leading (fully local)
    vl_score, _ = score_voice_leading(section_comp)
    
    # Contrapuntal (local metrics only)
    cp_score, _ = score_contrapuntal_local(section_comp)
    
    # Thematic fragment presence
    frag_score = check_fragment_presence(section_comp, subject_intervals)
    
    # Statistical (local interval bigrams)
    stat_score, _ = score_statistical(section_comp)
    
    # Cadence quality (if section ends at a phrase boundary)
    cad_score = score_cadence_at_boundary(section_comp, key_root, key_mode)
    
    # Section composite
    return (
        vl_score * 0.30
        + cp_score * 0.25
        + frag_score * 0.20
        + stat_score * 0.10
        + cad_score * 0.15
    )
```

### Cumulative reward

For a sequence with N sections, the reward at token t is the sum of rewards for all completed sections up to t:

```
R(t) = sum(section_score[i] for i in completed_sections_before(t))
```

This gives the model credit assignment: if section 3 scored badly, the tokens in section 3 get lower cumulative reward than tokens in sections 1-2.

---

## Phase 1: ReST (Reinforced Self-Training)

### Algorithm

```
for iteration in range(max_iterations):
    1. Generate candidate pool
    2. Score all candidates (section-level + full-sequence)
    3. Filter to top performers
    4. Fine-tune model on winners + real data
    5. Evaluate: if no improvement, stop ReST → move to GRPO
```

### Step 1: Generate candidate pool

For each iteration, generate across diverse prompts:

```python
PROMPTS = [
    {"key": "B minor", "mode": "fugue", "voices": 4},
    {"key": "D minor", "mode": "fugue", "voices": 4},
    {"key": "G major", "mode": "fugue", "voices": 3},
    {"key": "C minor", "mode": "fugue", "voices": 4},
    {"key": "Eb major", "mode": "fugue", "voices": 4},
    {"key": "F# minor", "mode": "sinfonia", "voices": 3},
    {"key": "A minor", "mode": "invention", "voices": 2},
    {"key": "Bb major", "mode": "chorale", "voices": 4},
    # ... 20-30 diverse prompts covering keys, forms, voice counts
]

CANDIDATES_PER_PROMPT = 50  # Lower than generation (speed matters in the loop)
```

Total per iteration: ~1000-1500 candidates.

Temperature: 0.9 (exploratory — we want diversity in the candidate pool, the filter handles quality).

### Step 2: Score candidates

For each candidate:
1. Split into 4-bar sections
2. Score each section independently
3. Compute section-level reward curve
4. Also compute the full-sequence composite score (existing scorer)
5. Record both the sequence and its per-section rewards

```python
@dataclass
class ScoredCandidate:
    tokens: list[int]
    prompt: dict
    section_rewards: list[float]  # per-section scores
    composite: float              # full-sequence composite
    mean_section: float           # mean of section scores
```

### Step 3: Filter to winners

Selection criteria (all must be met):
- Full-sequence composite ≥ 75th percentile within its prompt group
- No section scores below 0.3 (reject candidates with one terrible section)
- Mean section score ≥ 70th percentile

This gives ~20-25% of candidates as winners. Not too aggressive (avoids mode collapse), not too loose (maintains quality pressure).

### Step 4: Fine-tune on winners + real data

```python
# Mix ratio: 50% winners, 50% original training data
train_sequences = winners + sample(bach_training_data, len(winners))
shuffle(train_sequences)

# Fine-tune for 3-5 epochs with low learning rate
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
# Use cosine schedule within each ReST iteration
```

Key details:
- **Always mix real data.** Without this, the model forgets Bach and converges to "high-scoring but boring" patterns.
- **Low learning rate.** We're nudging, not retraining. 1e-5 is 10x lower than the original fine-tune LR.
- **Few epochs.** 3-5 epochs per iteration. Overfitting to one iteration's winners is the main risk.
- **Save checkpoint after each iteration.** If iteration N is worse than N-1, roll back.

### Step 5: Evaluate and decide

After fine-tuning, generate a fixed evaluation set (same prompts, same seeds):
- 20 candidates per prompt, 10 prompts = 200 total
- Score with full-sequence scorer
- Track: mean composite, mean thematic recall, mean contrapuntal, mean voice leading

```python
@dataclass
class IterationMetrics:
    iteration: int
    mean_composite: float
    mean_thematic_recall: float
    mean_contrapuntal: float
    mean_voice_leading: float
    top_10_mean: float  # mean of top 10% scores
    winner_rate: float  # fraction scoring above baseline threshold
```

**Plateau detection:**
```python
def should_stop_rest(history: list[IterationMetrics], patience: int = 3) -> bool:
    """Stop ReST if no improvement in `patience` iterations."""
    if len(history) < patience + 1:
        return False
    recent = history[-patience:]
    best_before = max(h.mean_composite for h in history[:-patience])
    best_recent = max(h.mean_composite for h in recent)
    return best_recent <= best_before + 0.005  # 0.5% improvement threshold
```

### ReST implementation: `rest_train.py`

New file: `src/bach_gen/training/rest_train.py`

```python
def rest_loop(
    model: BachTransformer,
    tokenizer: BachTokenizer,
    bach_data: list[list[int]],
    prompts: list[dict],
    candidates_per_prompt: int = 50,
    winner_percentile: float = 75,
    min_section_score: float = 0.3,
    mix_ratio: float = 0.5,
    lr: float = 1e-5,
    epochs_per_iter: int = 3,
    max_iterations: int = 20,
    patience: int = 3,
    output_dir: Path = Path("models_NEW/rest"),
    device: torch.device = None,
) -> BachTransformer:
    """Run the full ReST loop."""
    
    history: list[IterationMetrics] = []
    best_composite = 0.0
    
    for iteration in range(max_iterations):
        logger.info(f"=== ReST Iteration {iteration + 1}/{max_iterations} ===")
        
        # 1. Generate
        candidates = generate_candidate_pool(
            model, tokenizer, prompts, candidates_per_prompt, device
        )
        
        # 2. Score (section-level + full)
        scored = score_all_candidates(candidates, tokenizer)
        
        # 3. Filter
        winners = filter_winners(scored, winner_percentile, min_section_score)
        logger.info(f"  Winners: {len(winners)}/{len(scored)} "
                    f"({len(winners)/len(scored)*100:.0f}%)")
        
        # 4. Fine-tune
        train_data = build_training_mix(winners, bach_data, mix_ratio)
        fine_tune_iteration(model, train_data, lr, epochs_per_iter, device)
        
        # 5. Evaluate
        metrics = evaluate_iteration(model, tokenizer, prompts[:10], device)
        metrics.iteration = iteration + 1
        history.append(metrics)
        
        logger.info(f"  Composite: {metrics.mean_composite:.4f} "
                    f"(best: {best_composite:.4f})")
        
        # Save checkpoint
        save_path = output_dir / f"rest_iter_{iteration + 1}.pt"
        save_checkpoint(model, save_path, metrics)
        
        if metrics.mean_composite > best_composite:
            best_composite = metrics.mean_composite
            save_checkpoint(model, output_dir / "rest_best.pt", metrics)
        
        # Plateau check
        if should_stop_rest(history, patience):
            logger.info(f"  Plateau detected after {iteration + 1} iterations.")
            break
    
    # Load best checkpoint
    model = load_checkpoint(output_dir / "rest_best.pt")
    return model
```

### CLI integration

Add to `cli.py`:

```
uv run bach-gen rest-train \
    --model-path models_NEW/finetune_best.pt \
    --candidates 50 \
    --winner-percentile 75 \
    --mix-ratio 0.5 \
    --lr 1e-5 \
    --epochs-per-iter 3 \
    --max-iterations 20 \
    --patience 3
```

### Expected timeline

- Each iteration: ~30 min generation (50 candidates × 30 prompts with KV cache) + ~10 min training + ~10 min eval = ~50 min
- 10-15 iterations to plateau = ~8-12 hours
- Run overnight, wake up to a better model

---

## Phase 2: GRPO (Group Relative Policy Optimization)

### When to switch

Move to GRPO when ReST satisfies any of:
- Plateau detected (no improvement for 3 iterations)
- Winner rate exceeds 80% (nearly all candidates are good — sampling can't improve further)
- Mean composite stops improving but variance is still high (model is inconsistent)

### Why GRPO over PPO

- **No value model needed.** PPO requires a separate critic network (~doubling parameters). With 5.7M params, you can't afford that.
- **Group normalization replaces the baseline.** Instead of learning V(s), GRPO normalizes rewards within each group of candidates from the same prompt. Same variance reduction, zero extra parameters.
- **Your scorer is deterministic.** PPO's advantage over simpler methods comes from handling stochastic rewards. Your scorer always gives the same score for the same sequence.

### Algorithm

```
for step in range(max_steps):
    1. Sample a prompt
    2. Generate G candidates from current policy (G = group size, e.g. 16)
    3. Score all G candidates (section-level rewards)
    4. Compute per-token advantages using group normalization
    5. Policy gradient update with KL penalty against reference model
```

### Section-level advantage computation

This is where section scoring pays off most. Instead of one reward per sequence, each section contributes to the advantage of its tokens:

```python
def compute_section_advantages(
    section_rewards: list[list[float]],  # [G candidates][N sections each]
    group_size: int,
) -> list[list[float]]:
    """Compute per-section advantages using group normalization.
    
    For each section position, normalize rewards across the group:
    advantage[i][s] = (reward[i][s] - mean_s) / (std_s + eps)
    
    where mean_s and std_s are computed across all G candidates
    for section position s.
    """
    max_sections = max(len(r) for r in section_rewards)
    advantages = []
    
    for i in range(group_size):
        adv_i = []
        for s in range(len(section_rewards[i])):
            # Gather rewards at this section position across group
            section_s_rewards = [
                section_rewards[j][s]
                for j in range(group_size)
                if s < len(section_rewards[j])
            ]
            mean_s = sum(section_s_rewards) / len(section_s_rewards)
            std_s = (sum((r - mean_s)**2 for r in section_s_rewards) 
                     / len(section_s_rewards)) ** 0.5
            
            adv_i.append((section_rewards[i][s] - mean_s) / (std_s + 1e-8))
        
        advantages.append(adv_i)
    
    return advantages
```

### Per-token advantage assignment

Each token inherits the advantage of its section:

```python
def assign_token_advantages(
    tokens: list[int],
    section_boundaries: list[int],  # token indices where sections start
    section_advantages: list[float],
) -> list[float]:
    """Map section-level advantages to per-token advantages."""
    token_advantages = []
    current_section = 0
    
    for t in range(len(tokens)):
        if (current_section + 1 < len(section_boundaries) and 
            t >= section_boundaries[current_section + 1]):
            current_section += 1
        
        token_advantages.append(section_advantages[current_section])
    
    return token_advantages
```

### Policy gradient with KL penalty

```python
def grpo_loss(
    model: BachTransformer,
    ref_model: BachTransformer,   # frozen copy of model before GRPO
    tokens: torch.Tensor,         # (G, T) token sequences
    advantages: torch.Tensor,     # (G, T) per-token advantages
    kl_coeff: float = 0.05,       # KL penalty coefficient
) -> torch.Tensor:
    """GRPO policy gradient loss with KL regularization."""
    # Current policy log probs
    logits = model(tokens[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    # Reference policy log probs (no grad)
    with torch.no_grad():
        ref_logits = ref_model(tokens[:, :-1])
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_log_probs = ref_log_probs.gather(
            -1, tokens[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
    
    # KL divergence (per token)
    kl_div = token_log_probs - ref_token_log_probs
    
    # Policy gradient: maximize advantage-weighted log prob, minimize KL
    loss = -(advantages[:, :-1] * token_log_probs).mean() + kl_coeff * kl_div.mean()
    
    return loss
```

### KL coefficient schedule

Start with `kl_coeff = 0.1` (conservative — stay close to reference). Decay to 0.01 over training. If KL divergence exceeds a threshold (e.g., 0.5 nats), temporarily increase kl_coeff to pull the model back.

### GRPO hyperparameters

```python
GRPO_CONFIG = {
    "group_size": 16,           # candidates per prompt per step
    "lr": 5e-6,                 # lower than ReST — RL updates are noisier
    "kl_coeff_init": 0.1,
    "kl_coeff_min": 0.01,
    "kl_decay_steps": 500,
    "max_steps": 2000,
    "eval_interval": 50,
    "section_length_bars": 4,
    "ref_model_update_interval": None,  # keep fixed (no target network)
}
```

### GRPO implementation: `grpo_train.py`

New file: `src/bach_gen/training/grpo_train.py`

```python
def grpo_loop(
    model: BachTransformer,
    tokenizer: BachTokenizer,
    prompts: list[dict],
    config: dict = GRPO_CONFIG,
    output_dir: Path = Path("models_NEW/grpo"),
    device: torch.device = None,
) -> BachTransformer:
    """Run GRPO training loop."""
    
    # Freeze reference model
    ref_model = deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    best_composite = 0.0
    
    for step in range(config["max_steps"]):
        # Sample a prompt
        prompt = random.choice(prompts)
        
        # Generate group
        group_tokens = []
        for _ in range(config["group_size"]):
            tokens = generate_one(model, tokenizer, prompt, 
                                  temperature=0.9, device=device)
            group_tokens.append(tokens)
        
        # Score sections
        section_rewards = []
        for tokens in group_tokens:
            comp = tokenizer.decode(tokens)
            boundaries, rewards = score_sections(
                comp, section_length_bars=config["section_length_bars"]
            )
            section_rewards.append(rewards)
        
        # Compute advantages
        advantages = compute_section_advantages(
            section_rewards, config["group_size"]
        )
        
        # Map to per-token advantages
        # ... (assign_token_advantages for each candidate)
        
        # Pad and batch
        # ... (standard padding)
        
        # Compute loss
        loss = grpo_loss(model, ref_model, batch_tokens, batch_advantages,
                         kl_coeff=current_kl_coeff(step, config))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Eval
        if (step + 1) % config["eval_interval"] == 0:
            metrics = evaluate_iteration(model, tokenizer, prompts[:10], device)
            logger.info(f"Step {step+1}: composite={metrics.mean_composite:.4f}")
            
            if metrics.mean_composite > best_composite:
                best_composite = metrics.mean_composite
                save_checkpoint(model, output_dir / "grpo_best.pt", metrics)
    
    return model
```

### Memory considerations

GRPO needs both the current model and frozen reference model in memory. At 5.7M params in fp32, that's ~46MB total — trivial on your 96GB machine. The real cost is generating 16 candidates per step. With KV cache, this should take ~10-15 seconds per step.

---

## Automatic Pipeline: `rl_train.py`

The full automatic loop that runs ReST until plateau, then switches to GRPO:

```python
def rl_pipeline(
    model_path: Path,
    bach_data_path: Path,
    output_dir: Path,
    rest_config: dict,
    grpo_config: dict,
):
    """Full RL pipeline: ReST → GRPO."""
    
    model, config = load_checkpoint(model_path)
    tokenizer = load_tokenizer(...)
    bach_data = load_training_data(bach_data_path)
    
    prompts = generate_diverse_prompts()
    
    # Phase 1: ReST
    logger.info("=" * 60)
    logger.info("  PHASE 1: ReST")
    logger.info("=" * 60)
    
    model = rest_loop(
        model=model,
        tokenizer=tokenizer,
        bach_data=bach_data,
        prompts=prompts,
        output_dir=output_dir / "rest",
        **rest_config,
    )
    
    rest_metrics = load_metrics(output_dir / "rest")
    logger.info(f"ReST complete. Best composite: {rest_metrics.best_composite:.4f}")
    
    # Phase 2: GRPO
    logger.info("=" * 60)
    logger.info("  PHASE 2: GRPO")
    logger.info("=" * 60)
    
    model = grpo_loop(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        config=grpo_config,
        output_dir=output_dir / "grpo",
    )
    
    grpo_metrics = load_metrics(output_dir / "grpo")
    logger.info(f"GRPO complete. Best composite: {grpo_metrics.best_composite:.4f}")
    
    # Final evaluation
    logger.info("=" * 60)
    logger.info("  FINAL EVALUATION")
    logger.info("=" * 60)
    
    final_eval = full_evaluation(model, tokenizer, prompts, 
                                  candidates=200, output_dir=output_dir / "final")
    
    logger.info(f"Final best composite: {final_eval.top_composite:.4f}")
    logger.info(f"Final mean composite: {final_eval.mean_composite:.4f}")
```

### CLI

```
uv run bach-gen rl-train \
    --model-path models_NEW/finetune_best.pt \
    --output-dir models_NEW/rl \
    --rest-candidates 50 \
    --rest-max-iterations 20 \
    --grpo-group-size 16 \
    --grpo-max-steps 2000
```

---

## Implementation Order

1. **`score_section()`** — extract sections from a composition, score locally. Build on existing scorer infrastructure. Test against full-sequence scores to verify consistency.

2. **`rest_train.py`** — the ReST loop. Uses existing `_generate_one`, `score_composition`, and `Trainer.fine_tune`. Wire up the generate → filter → train cycle.

3. **CLI `rest-train` command** — expose ReST with sensible defaults. Run it overnight. Evaluate results.

4. **`grpo_train.py`** — implement after ReST plateaus. The section-level advantage computation is the novel piece; the policy gradient is standard.

5. **CLI `rl-train` command** — the full automatic pipeline.

6. **Monitoring dashboard** (optional) — log iteration metrics to JSON, plot composite/thematic_recall/contrapuntal over iterations. Makes it easy to see if RL is actually helping.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Mode collapse (model generates one "safe" pattern) | 50% real Bach data in every ReST iteration. Diverse prompts (keys, forms, voices). Monitor output diversity |
| Reward hacking (model exploits scorer weaknesses) | Section-level scoring is harder to hack than sequence-level. Monitor qualitative samples every iteration. If scores rise but music sounds worse, the scorer has a bug |
| Catastrophic forgetting of base capabilities | Always start from fine-tune checkpoint, not from scratch. Keep learning rate low (1e-5 ReST, 5e-6 GRPO). Mix real data |
| GRPO KL divergence blows up | KL penalty with automatic coefficient scaling. If KL > threshold, increase penalty. Save checkpoints frequently |
| Section scoring is too noisy for advantage computation | Start with 4-bar sections. If too noisy, increase to 8-bar. If still noisy, fall back to sequence-level rewards for GRPO (less ideal but still works) |
| Generation too slow for the RL loop | KV cache makes this feasible. 50 candidates × 30 prompts × 50 min each with KV cache ≈ 30 min per ReST iteration |

---

## Expected Outcomes

**After ReST (5-15 iterations):**
- Mean composite score should increase by 0.05-0.15 (from ~0.65 to ~0.75)
- Top candidate quality should improve (less reliance on brute-force sampling)
- 20 candidates should give comparable quality to current 100 candidates

**After GRPO (500-2000 steps):**
- Model should become more consistent (lower variance across candidates)
- Section-level quality should improve (fewer "dead" sections in otherwise good pieces)
- Cadence placement and subject recall should specifically improve (these are the dimensions with clearest section-level signal)

**The real test:** Generate 10 candidates instead of 100 and see if the quality is comparable to pre-RL generation with 100 candidates. If yes, RL worked — the model internalized what the scorer values.
