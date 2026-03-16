# Reinforcement Learning Plan: Hybrid ReST First, GRPO Only If Earned

## Overview

The goal is still the same:
- reduce dependence on brute-force candidate search
- push the model toward Bach-quality outputs
- keep the "impressive demo" qualities that already matter to you

But the rollout should be more conservative than a direct jump into dense section-level RL, and more hybrid than scorer-only RL.

The main reason is simple:
- the handcrafted scorer is much better than before, but it is still a hand-built reward
- an LLM judge can evaluate many more plausible candidates than a human can
- human preference is still the real gold standard for what the project actually wants

So the recommended sequence is:

1. **Phase 0: Freeze the judge stack and benchmark it**
2. **Phase 1: Conservative ReST**
   - handcrafted scorer filters hard failures
   - LLM ranks plausible survivors
   - human audits only the top slice
   - section scores only as vetoes / diagnostics
3. **Phase 1.5: Adversarial mining**
   - explicitly collect scorer / LLM / human disagreement cases
4. **Phase 2: Optional section-aware ReST**
   - only if Phase 1 behaves well
5. **Phase 3: GRPO**
   - start with group-normalized sequence rewards
   - preferably against a learned hybrid reward model
   - only add dense section-level token advantages after GRPO is already stable

The key principle is:

- **Do not make section-level reward the main optimization target on day one.**

Use it first to reject obviously broken pieces and to analyze where the scorer is helping or lying.

---

## Reward Stack

Do not optimize raw `composite` alone.

The current evaluator already distinguishes:
- `bach_similarity`
- `rhetorical_impact`
- `demo_bach_balance`

That should be reflected in the reward.

### Recommended reward for RL / ReST selection

Use a reward stack like:

```python
reward = (
    0.45 * bach_similarity
    + 0.35 * demo_bach_balance
    + 0.20 * rhetorical_impact
    - hard_failure_penalties
)
```

Where `hard_failure_penalties` includes:
- broken or absent cadence at ending
- missing or collapsed voices
- incomplete structure
- strong fugue/chorale/form mismatch flags
- any explicit scorer failure flags already known to be real problems

### Why not raw composite?

Because raw composite still overweights some “clean and dramatic” patterns that are attractive but not necessarily deeper or more Bach-like.

The current split reward is closer to the actual project goal:
- strong Bach resemblance
- strong demo effect
- no structural failures

## Judge Hierarchy

The long-term RL target should not be "handcrafted scorer only."

The right hierarchy is:

1. **Handcrafted scorer**
   - fast
   - deterministic
   - ideal for hard failures and structural guardrails

2. **LLM judge**
   - scalable
   - much better than heuristics at comparing plausible musical candidates
   - should operate as a ranking layer, not as the only safety check

3. **Human preference**
   - slow and expensive
   - highest-quality signal
   - best used to calibrate the LLM judge and any learned reward model

### Recommended operational flow

For each prompt-grouped candidate pool:

1. Use the handcrafted scorer to reject broken candidates.
2. Use the LLM to rank the plausible survivors.
3. Use human listening only on the top slice, not the full pool.

This is the scalable version of the musical judgment stack you actually want.

### Long-term reward target

The best eventual GRPO reward is a **learned hybrid reward model** distilled from:
- handcrafted structural signals
- LLM preference judgments
- a smaller human preference set

That is the most realistic path to scalable reward without turning the project into scorer-only optimization.

---

## Phase 0: Freeze the Judge Stack Before RL

Before any RL:

1. Freeze the scorer version.
2. Freeze the LLM judging prompt / schema if the LLM is in the loop.
3. Save the benchmark outputs used to justify both.
4. Do not keep changing the reward during an RL run.

Why:
- if reward changes midstream, iteration-to-iteration progress becomes uninterpretable
- rollback gets harder
- reward hacking becomes harder to detect

### What to freeze

At minimum, freeze:
- scorer weights and failure penalties
- LLM judge prompt
- LLM output schema
- candidate representation fed to the LLM
- pairwise or best-of-k judging format
- human review rubric used for spot audits

### Phase 0 acceptance gates

Do not start RL unless all are true:
- Bach benchmark still passes
- non-Bach conditioning audit still passes
- your current base model can already generate a few clearly good samples
- the reward ranks those good samples above obvious failures
- the LLM ranking agrees with your ear on a small pilot set

---

## Section-Level Scoring: Use as Diagnostic First

Dense section reward is attractive, but it is also the easiest place to inject noisy supervision.

The safe use order is:

### First use
- section scores as:
  - rejection criteria
  - diagnostics
  - logging

### Later use
- section scores as:
  - ranking features inside ReST
  - token-level advantages inside GRPO

### Default section boundaries

Keep simple, robust boundaries:
- 4-bar chunks by default
- cadence boundaries can refine them
- fugue subject-entry boundaries are optional, not required

Do **not** make token-level reward depend on fragile boundary logic in the first implementation.

---

## Phase 1: Conservative ReST

This should be the first real RL-style phase.

### Why ReST first

ReST is lower risk because:
- it keeps updates supervised
- winners are human-inspectable
- rollback is easy
- it tells you quickly whether the scorer is useful as a training signal at all

### ReST algorithm

```text
for iteration in range(max_iterations):
    1. Generate a prompt-grouped candidate pool
    2. Score each candidate with frozen handcrafted reward
    3. Reject hard failures
    4. Ask the frozen LLM judge to rank the survivors
    5. Keep conservative winners
    6. Fine-tune on winners + real Bach data
    7. Evaluate against a fixed prompt benchmark
    8. If no improvement or music gets worse, stop
```

### Prompt pool

Use a smaller and more controlled prompt pool than the original draft.

Start with:
- 8 to 12 prompts
- spread across fugue, invention, sinfonia, chorale
- fixed keys and fixed seeds for evaluation

Example:

```python
PROMPTS = [
    {"key": "B minor", "mode": "fugue", "voices": 4},
    {"key": "D minor", "mode": "fugue", "voices": 4},
    {"key": "Eb major", "mode": "fugue", "voices": 4},
    {"key": "F# minor", "mode": "sinfonia", "voices": 3},
    {"key": "A minor", "mode": "invention", "voices": 2},
    {"key": "Bb major", "mode": "chorale", "voices": 4},
]
```

### Candidate count

Start smaller:
- `16-24` candidates per prompt

Why:
- faster iterations
- easier debugging
- enough diversity to test whether the scorer helps

### Winner filtering

For the first version, keep it conservative.

A candidate is a winner only if:
- handcrafted reward is above the `75th` percentile within its prompt group
- no hard failure flags are present
- no section score is below a low floor such as `0.30`
- the LLM judge ranks it in the top tier of the surviving group

That means:
- section scores act as vetoes
- handcrafted sequence reward still drives safety
- the LLM is the scalable musical preference layer

### LLM judgment format

The safest first LLM format is not free-form scoring.

Use one of:
- pairwise A/B preference
- best-of-4 or best-of-6 ranking among scorer-filtered survivors

Do not ask the LLM to judge obviously broken candidates. That is wasted compute and teaches less.

### Human review during ReST

Humans should not rank the whole pool.

Instead:
- review a small sample of LLM-selected winners every iteration
- review disagreement cases where scorer and LLM disagree sharply
- stop the run if the top samples sound worse even while the metrics improve

### Training mix

Always mix winners with real Bach data.

Recommended first ratio:
- `60%` real Bach
- `40%` winners

This is more conservative than a 50/50 mix and better for early experiments.

### Fine-tune schedule per ReST iteration

Start with:
- `lr = 1e-5`
- `epochs_per_iter = 1 or 2`

Do **not** start with `3-5` epochs per iteration.

That is too aggressive for the first loop and increases the risk of drifting toward reward-friendly artifacts.

### ReST evaluation

After each iteration, evaluate on a fixed held-out prompt set:
- same prompts
- same seeds
- same candidate count

Track:
- mean reward
- mean `bach_similarity`
- mean `demo_bach_balance`
- mean `rhetorical_impact`
- winner rate above a fixed threshold

Also listen to a few top samples every iteration.

If scores rise while listening quality drops, stop. That is reward hacking.

### ReST stop criteria

Stop ReST if any of these happen:
- no improvement for `3` iterations
- reward rises but listening quality clearly worsens
- diversity collapses
- output quality gets narrower or more stereotyped

---

## Phase 1.5: Adversarial Mining

This phase is important enough that it should be explicit.

Between ReST and GRPO, mine failure cases.

### What to collect

Collect candidates that are:
- high `rhetorical_impact`, low `bach_similarity`
- high `bach_similarity`, but dull or inert
- structurally broken despite good local writing
- repetitive but scorer-friendly
- highly ranked by the LLM but disliked by a human
- highly ranked by the scorer but rejected by the LLM

These become:
- a qualitative inspection set
- a future negative set for reward calibration
- the highest-value human-labeling set

### Why this matters

GRPO will exploit exactly these cases if you do not surface them first.

### Human preference collection target

You do not need a huge human dataset to get started.

A realistic target is:
- `500-1,000` pairwise rankings for the first useful signal
- `2,000-5,000` pairwise rankings for a genuinely strong small-project reward model

Collect these mostly from disagreement cases and close musical decisions, not trivial "good vs broken" comparisons.

---

## Phase 2: Optional Section-Aware ReST

Only do this if conservative ReST behaves well.

This phase adds section scores as ranking features, but still keeps the optimization target sequence-level.

### Safe use of section scores here

Use section statistics like:
- minimum section score
- mean section score
- section variance

as selection features, not direct token rewards.

Example winner score:

```python
winner_score = (
    0.75 * sequence_reward
    + 0.15 * mean_section_score
    + 0.10 * min_section_score
)
```

This gives some pressure toward avoiding dead sections without letting noisy section labels dominate.

---

## Phase 3: GRPO

Do not start GRPO until:
- ReST clearly helps
- adversarial mining has been done
- the reward has survived that pressure test

### Why GRPO still makes sense

GRPO is still the right RL choice here because:
- no value network needed
- group normalization is natural for prompt-grouped candidate generation
- your reward is deterministic

### But start with sequence-level GRPO

The safest first GRPO objective is:
- one scalar reward per sequence
- normalized within the prompt group

Do **not** start with token-level section advantages in the first GRPO version.

### Preferred GRPO reward source

Best case:
- a learned hybrid reward model trained on handcrafted + LLM + human preferences

Acceptable first version:
- a frozen scalar reward stack plus LLM-ranked preference data distilled into sequence rewards

Avoid:
- direct dense RL against a raw heuristic composite only

### Initial GRPO loss

```python
loss = (
    -(advantages * token_log_probs).mean()
    + kl_coeff * kl_div.mean()
)
```

Where `advantages` are:
- per-sequence
- prompt-group normalized

### GRPO group design

Recommended:
- fixed prompt
- generate `G=8` or `G=12` candidates
- compute normalized reward within the group

Do not start at `G=16` unless throughput is clearly fine.

### KL schedule

Start conservatively:
- `kl_coeff = 0.1`

Only relax after you see stable behavior.

### Section-level advantages come later

Only add token-level section advantages after:
- sequence-level GRPO is stable
- samples are improving by ear
- no obvious reward hacking has appeared

That should be treated as a second GRPO phase, not the first.

---

## Automatic Pipeline

The full automatic loop should still be:

```text
Base model
  -> Conservative ReST
  -> Adversarial mining / scorer + LLM + human check
  -> Optional section-aware ReST
  -> Sequence-level GRPO
  -> Optional dense section-aware GRPO
```

Not:

```text
Base model
  -> section-level ReST
  -> dense GRPO
```

---

## Implementation Order

1. **Freeze scorer + LLM judge**
   - save benchmark outputs
   - save reward weights / penalty logic
   - save LLM judge prompt and schema

2. **Build prompt benchmark + listening set**
   - fixed held-out prompts
   - fixed seeds
   - stable evaluation outputs

3. **Implement conservative ReST**
   - handcrafted filter
   - LLM ranking
   - section veto only
   - low LR
   - 1-2 epochs per iteration

4. **Add adversarial mining**
   - save top failures by reward profile
   - save scorer / LLM / human disagreements

5. **Collect initial human preference data**
   - pairwise judgments on disagreement sets
   - start small and high-quality

6. **Train a learned hybrid reward model**
   - distilled from handcrafted + LLM + human signals

7. **Implement optional section-aware ReST ranking**
   - still no token-level reward

8. **Implement sequence-level GRPO**
   - group-normalized scalar reward
   - strong KL penalty

9. **Only then consider dense section-aware GRPO**

---

## CLI Suggestions

### Conservative ReST

```bash
uv run bach-gen rest-train \
  --model-path models/finetune_best.pt \
  --candidates 24 \
  --winner-percentile 75 \
  --mix-ratio 0.4 \
  --lr 1e-5 \
  --epochs-per-iter 1 \
  --max-iterations 10 \
  --patience 3
```

### First GRPO pass

```bash
uv run bach-gen grpo-train \
  --model-path models/rest_best.pt \
  --group-size 8 \
  --lr 5e-6 \
  --kl-coeff 0.1 \
  --max-steps 500
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Reward hacking | Freeze scorer version, add adversarial mining, listen every iteration |
| Mode collapse | Keep real Bach in every ReST iteration, use prompt-grouped winners, track diversity |
| Overfitting to a few prompts | Use fixed train prompts plus separate held-out eval prompts |
| Noisy section rewards | Use section scores only as vetoes / diagnostics first |
| GRPO instability | Start with sequence-level rewards only, strong KL, small group size |
| False score gains | Require both metric improvement and qualitative approval |
| Forgetting base musical competence | Low LR, short ReST iterations, KL regularization in GRPO |
| LLM judge inconsistency | Freeze prompt/schema, use pairwise or small-group ranking, audit with human labels |
| LLM representation mismatch | Keep symbolic rendering format fixed, periodically spot-check against real listening |

---

## Expected Outcomes

### After conservative ReST

You should expect:
- modest but real improvement in average reward
- better top-1 selection quality at lower candidate counts
- no catastrophic style drift

Good outcome:
- `10-20` candidates after ReST feel closer to `50-100` candidates before ReST

### After sequence-level GRPO

You should expect:
- more consistency
- fewer obviously weak candidates
- stronger alignment to the reward stack

But you should **not** assume:
- automatic musical improvement without listening checks
- that higher reward always means better music

### The real success criterion

The real success criterion is not “reward went up.”

It is:
- fewer candidates needed for good outputs
- better average sample quality
- preserved or improved Bach-likeness
- preserved or improved demo impact

If that happens, RL helped.

If the score rises but the music narrows, stiffens, or becomes formulaic, the reward is being gamed.

### The real gold standard

The real gold standard is:
- handcrafted scorer for structural guardrails
- LLM judgment for scalable ranking
- human preference for final truth and calibration

If those three agree more often over time, the RL stack is improving for the right reasons.

---

## Bottom Line

RL still looks promising here.

But the safest path is:
- **ReST first**
- **handcrafted guardrails first**
- **LLM ranking before large-scale human review**
- **sequence-level reward first**
- **section scores as veto/diagnostic first**
- **GRPO only after the scorer survives adversarial pressure**

And the best long-term target is not scorer-only RL. It is:
- handcrafted structural constraints
- LLM musical preference at scale
- human calibration on the hard cases

That path is slower than jumping straight into dense section-level GRPO, but much more likely to produce a model you actually want to keep.
