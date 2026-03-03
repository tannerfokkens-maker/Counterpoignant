# Cadence Conditioning + Subject Recall: Execution Plan

## What We're Fixing

Two specific weaknesses in an otherwise strong generation pipeline:

1. **Missing internal cadences.** The model has learned to end on V-I (purely from training data), but Bach's fugues have cadences throughout — half cadences at phrase boundaries, authentic cadences at section ends, deceptive cadences for continuation. Our model writes continuous counterpoint with no harmonic punctuation.

2. **Subject recall.** The model introduces a subject and does excellent short-range imitation (fragments appear in other voices a few bars later), but fails to bring back full subject entries in later sections. Real fugues have subject entries distributed across the entire piece — exposition, episodes, middle entries, final stretto.

What already works and must not regress: counterpoint quality, voice leading, short-range imitation, voice independence, register consistency.

## Design Principle

Add the minimum conditioning tokens needed to teach these two specific behaviors. Do not add tokens for things the model already learns from raw data. Use conditioning dropout so unconditioned generation stays strong.

---

## Step 1: Cadence Labels (highest priority)

### Why cadences first

Cadences are the structural skeleton of tonal music. They define phrase boundaries, create harmonic rhythm, and give the listener a sense of arrival and departure. Without them, even perfect counterpoint sounds like a run-on sentence. Cadences are also the easiest harmonic feature to detect reliably — the bass motion and voice-leading patterns are highly stereotyped.

### Token vocabulary

Keep it small. Four tokens:

| Token | Meaning | Bach usage |
|-------|---------|------------|
| `CAD_PAC` | Perfect authentic cadence (V→I, soprano on tonic) | Section ends, final cadence |
| `CAD_IAC` | Imperfect authentic cadence (V→I, soprano on 3rd/5th) | Internal phrase ends |
| `CAD_HC` | Half cadence (phrase ends on V) | Mid-phrase pauses, question phrases |
| `CAD_DC` | Deceptive cadence (V→vi or V→VI) | Continuation, avoiding closure |

No `CAD_NONE` — absence of a cadence token is the default. The model should treat silence as "continue the texture."

### Detection pipeline

For each piece in training data:

1. **Find phrase boundaries.** Scan all voices for rhythmic convergence points — moments where multiple voices have long notes or rests simultaneously. These are candidate cadence locations.

2. **Check bass motion at each candidate.** Look at the bass voice in a 2-beat window before the boundary:
   - Bass moves up a 4th or down a 5th to arrive at the boundary → dominant motion
   - Bass arrives on scale degree 1 → tonic arrival
   - Bass arrives on scale degree 5 → half cadence candidate
   - Bass arrives on scale degree 6 → deceptive cadence candidate

3. **Check soprano resolution.** At the boundary:
   - Soprano on scale degree 1 + bass on 1 + dominant approach → `CAD_PAC`
   - Soprano on 3 or 5 + bass on 1 + dominant approach → `CAD_IAC`
   - Bass on 5 with preceding predominant motion → `CAD_HC`
   - Bass on 6 with preceding dominant → `CAD_DC`

4. **Confidence threshold.** Only label cadences where bass motion AND at least one of (soprano resolution, rhythmic convergence, duration pattern) agree. When in doubt, don't label. Missing a cadence is fine; mislabeling one teaches the wrong pattern.

### Token placement

Insert the cadence token immediately before the BAR token at the cadence location. The model sees:

```
... DEG_5 OCT_3 DUR_quarter CAD_PAC BAR BEAT_1 ...
```

This teaches the model: "when you see a cadence token, the next bar boundary is a cadential arrival." At inference, inserting a `CAD_PAC` token biases the model toward producing authentic cadence voice-leading in the following beats.

### Training integration

- **Conditioning dropout: 40%.** During training, randomly delete cadence tokens from 40% of sequences. This ensures the model can generate coherently without them.
- **No changes to loss function.** Cadence tokens are predicted like any other token. The model learns when cadences are appropriate from the distribution of where they appear in the data.
- **No changes to model architecture.** These are just new entries in the vocabulary.

### Validation before training

Run the detector on the full corpus. Check:
- Cadence density per form (fugues should have ~1 cadence every 4-8 bars)
- Distribution of cadence types (PAC should be most common, DC least)
- Spot-check 10-15 pieces manually — are cadences where they should be?

If cadence density is too low (< 1 per 8 bars on average for fugues), the confidence threshold is too strict. If density is too high (> 1 per 2 bars), it's too loose.

### Inference controls

New CLI options:
- `--cadence-density low|medium|high` — controls how often cadence tokens are inserted into the generation prompt at periodic intervals
- The model can also generate cadence tokens spontaneously (it learned where they go), so this is a bias, not a hard constraint

---

## Step 2: Subject Boundary Tokens

### Why second

Subject recall is the second priority because the model already attempts subject-like behavior (it places material at the start and does imitation). The problem is sustaining full subject entries deep into the piece. Explicit boundary tokens give the model a structural signal it currently lacks.

### Token vocabulary

Two tokens only:

| Token | Meaning |
|-------|---------|
| `SUBJ_START` | Beginning of a subject entry |
| `SUBJ_END` | End of a subject entry |

No `ANS_START`/`ANS_END` — the distinction between subject and tonal answer is subtle (a few altered intervals at the start). The model should learn this from the pitch content. Separate tokens add noise for minimal benefit.

### Detection pipeline

Adapt the existing thematic recall subject extractor:

1. **Extract the subject.** Find the first entering voice, take its first melodic phrase (up to first significant gap or first 2 bars). Convert to interval sequence.

2. **Search all voices.** For each voice, slide a window matching the subject's interval sequence with:
   - Exact transposition (same intervals, different starting pitch)
   - Tonal answer tolerance (±1 semitone on first 2-3 intervals, to catch dominant-level answers)
   - Minimum match length: 70% of subject length

3. **Mark boundaries.** For each match above confidence threshold:
   - `SUBJ_START` before the first note of the match
   - `SUBJ_END` after the last note of the match

4. **Confidence gating.** Only label entries where:
   - Interval match quality ≥ 0.80 (allowing minor tonal alterations)
   - The entry doesn't overlap with another labeled entry in the same voice
   - The entry is at least 4 notes long

### Token placement

```
... VOICE_2 SUBJ_START DEG_5 OCT_3 DUR_quarter DEG_4 OCT_3 DUR_eighth ... SUBJ_END ...
```

### Training integration

- **Conditioning dropout: 40%.** Same as cadences — randomly strip subject markers from 40% of sequences.
- **The exposition subject (first entry) always keeps its markers** even during dropout. This teaches the model that the opening always has a clearly marked subject.
- **Subject markers in later entries are dropped independently.** This teaches the model to bring subjects back even without explicit prompting.

### Inference controls

- `--min-subject-entries N` — after generating the exposition, periodically insert `SUBJ_START` tokens to encourage subject re-entry
- `--subject-spacing bars` — minimum bars between encouraged subject entries

---

## Step 3: Retrain and Evaluate

### Tokenizer changes

Add the 6 new tokens to the vocabulary:
```
CAD_PAC, CAD_IAC, CAD_HC, CAD_DC, SUBJ_START, SUBJ_END
```

Vocab goes from 121 → 127. Negligible model size increase.

### Data re-processing

1. Run cadence detector on all training data
2. Run subject detector on fugue/invention/sinfonia training data
3. Re-tokenize with new tokens embedded
4. Verify token counts and distributions look reasonable

### Training plan

**Phase 1: Cadence only.** Add cadence tokens, retrain from the current fine-tuned checkpoint (not from scratch — the model already knows counterpoint, don't throw that away). Train for 20-30 epochs with the same learning rate schedule. Compare:
- Do generated fugues now have internal cadences?
- Does the V-I at the end survive?
- Any regression in counterpoint quality or voice leading?

**Phase 2: Cadence + Subject.** If Phase 1 shows improvement, add subject tokens and retrain for another 20-30 epochs. Compare:
- Does thematic recall improve in the scorer?
- Do subjects appear in later sections?
- Any regression from Phase 1 gains?

### Evaluation metrics

Use the existing calibrated scorer, plus:
- **Cadence count per piece** — how many internal cadences does the model produce?
- **Cadence type distribution** — does it use PAC, HC, DC appropriately?
- **Subject entry count** — how many subject entries after the exposition?
- **Subject entry distribution** — are entries spread across the piece or clustered?

Compare against the Bach corpus averages from the training data.

---

## What We're NOT Doing

- **Roman numeral / chord-level harmonic tokens.** Too noisy to detect reliably, too many tokens to add, and the model may already be learning chord progressions implicitly through scale-degree patterns. Revisit only if cadence tokens prove the conditioning approach works AND harmonic analysis is still clearly missing.

- **Functional harmony labels (T/PD/D).** Too coarse to capture what makes Bach sound like Bach. The difference between ii-V-I and IV-V-I is exactly what matters, and T/PD/D erases it.

- **Gold annotation protocol.** With 744 fugue sequences, spending weeks on manual annotation is misallocated effort. Spot-check 10-15 pieces, tune thresholds, and move on.

- **Per-form/per-composer calibration of detectors.** The model is fine-tuned on Bach. The detectors run on Bach. One calibration is enough.

- **Beam search caching for KV cache.** Out of scope, deferred.

---

## Implementation Order

1. Add 6 tokens to tokenizer vocabulary
2. Build cadence detector (bass motion + soprano resolution + rhythmic convergence)
3. Run detector on corpus, validate densities, spot-check 10-15 pieces
4. Build subject detector (adapt existing thematic recall code)
5. Run detector on fugue/invention/sinfonia data, validate, spot-check
6. Re-tokenize training data with both token types
7. Phase 1 training: cadence tokens only (dropout 40%)
8. Evaluate Phase 1, compare to baseline
9. Phase 2 training: add subject tokens (dropout 40%)
10. Evaluate Phase 2, compare to Phase 1 and baseline
11. Add CLI inference controls for cadence density and subject spacing
12. Generate final portfolio with new controls

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cadence detector mislabels, model learns wrong patterns | Medium | Precision-first threshold, 40% dropout, spot-check before training |
| Subject detector false positives in non-fugal forms | Low | Only label fugue/invention/sinfonia, confidence ≥ 0.80 |
| New tokens regress counterpoint quality | Low | Train from existing checkpoint (not scratch), monitor voice leading + contrapuntal scores |
| 40% dropout is wrong rate | Medium | If too high, model ignores tokens; if too low, model depends on them. Test 30% and 50% if 40% seems off |
| 6 new tokens bloat sequences | Very Low | Average ~10-20 extra tokens per sequence out of 2048. Negligible |
