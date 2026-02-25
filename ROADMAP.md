# Roadmap: Vibe Composing

**Vision:** A "Claude Code for music composition." The human provides taste, direction, and creative intent — not music theory knowledge. An AI orchestrator translates natural language into precise musical parameters and drives a specialized generation model. The result is iterative, collaborative composition accessible to anyone.

---

## Phase 1: Solid Foundation (Current)

What we have today and what's in progress.

**Done when:** The 4-voice chorale model generates coherent, stylistically recognizable Bach chorales that score above 0.6 composite on the evaluation suite, the 2-part model generates recognizable inventions at the same threshold, and a non-musician can hear the difference between style tokens. The 0.6 threshold must be calibrated first — use `bach calibrate` to run the evaluation suite on real corpus pieces and degenerate baselines to anchor the scale before treating this number as meaningful.

### Architecture & Infrastructure
- [x] Multi-voice Transformer model (2-4 voices, ~6.4M params)
- [x] Rotary Position Embeddings (RoPE) replacing absolute positional embeddings — relative position encoding for better generalization across piece lengths and structural recurrence
- [x] Scale-degree tokenizer (key-agnostic representation, 91-token vocabulary including BEAT markers and form tokens). Decomposes pitch into OCT → [ACC] → DEG so the model learns harmonic function directly. This is the core technical differentiator — the model learns music theory rather than piano key positions.
- [x] Style conditioning tokens (Bach, Baroque, Renaissance, Classical)
- [x] Weight tying, pre-norm architecture, SwiGLU activation

### Generation
- [x] Subject/theme input (`--subject "G4 A4 Bb4 C5"`)
- [x] Beam search with length normalization (Wu et al.) and early stopping
- [x] Sampling with temperature, top-k, top-p
- [x] Cached constraint state for O(1) per-step constraint evaluation in beam search (previously O(seq_len) per step)
- [x] Best-of-N rejection sampling (generate N candidates, score all, return top K)
- [x] Hard constraints: voice range enforcement, key adherence, anti-degenerate patterns

### Evaluation
- [x] Multi-dimensional evaluation scoring (voice leading, statistical, structural, contrapuntal, information-theoretic)
- [x] Bisect-based voice alignment for O(log n) time-point queries (previously O(n))
- [x] Composite scorer with configurable dimension weights, feeds directly into rejection sampling

### Data
- [x] Multi-composer training corpus (2,516 works from 21+ composers via music21 + KernScores)
- [x] ~10,000 training sequences in scale-degree mode after chunking (no key augmentation needed — the representation is already key-invariant)
- [x] Smarter chord extraction with voice range hints — chords in bass parts select bass-range pitches, not soprano-range
- [x] Sequence chunking with overlap for pieces exceeding max_seq_len (75% stride, BOS prepended to continuation chunks)
- [x] Gradient accumulation for effective large batch training on consumer hardware

### Listening & Preview Infrastructure
- [x] MIDI preview pipeline — `bach play` command with FluidSynth, timidity, and system player fallback. One command from generation to audible output.
- [ ] A/B comparison — generate two candidates, listen side by side, pick the better one. Essential for tuning sampling parameters, validating conditioning tokens ("can a listener tell style=Bach from style=Classical?"), and eventually for preference data collection (Long-Term Aspirations).
- [ ] Batch audition — listen through the top-K outputs from rejection sampling quickly, without manually loading each MIDI file. The `bach play` command can scan a directory, but doesn't yet show score breakdowns inline or support skip/save keybindings.

### Pending (Phase 1a — ship these first)
- [ ] Train and validate 4-voice chorale model (in progress — 500 epochs, scale-degree tokenizer)
- [ ] Train and validate base 2-part model with style tokens
- [ ] Training curriculum: pre-train on full multi-composer corpus, fine-tune on Bach. Standard transfer learning — the model learns general counterpoint first, then specializes.
- [ ] Invest in evaluation scorer quality — this is the main quality lever for a long time. Better scoring → better rejection sampling → better perceived output, no architectural changes needed.
- [x] Calibrate evaluation scorer — `bach calibrate` command scores real Bach corpus pieces and degenerate baselines (shuffled, random, repetitive) to establish the score range. Saves results to `calibration.json`.
- [x] Listening infrastructure — `bach play` command for MIDI preview. Needed to validate the "done when" criterion that a non-musician can hear style differences.
- [x] **Fix train/val data leakage** — `create_dataset()` now splits by piece ID so all chunks from the same piece end up on the same side. `prepare-data` saves `piece_ids.json` alongside sequences. The `train` command loads piece IDs when available and falls back to random split with a warning for old data.

### Pending (Phase 1b — do when 1a is solid)
- [x] BEAT tokens — `BEAT_1` through `BEAT_6` emitted at each metric position, with compound meter support (e.g., 6/8 → 2 dotted-quarter beats). `time_signature` field propagated through extraction, augmentation, and tokenization. The model now has explicit metric position in the token stream, which is critical groundwork for DroPE.
- [x] Form tokens — `FORM_CHORALE`, `FORM_INVENTION`, `FORM_FUGUE`, `FORM_SINFONIA`, `FORM_QUARTET`, `FORM_TRIO_SONATA` separated from voice-count tokens (`MODE_2PART`..`MODE_FUGUE`). Both emitted during encoding: `BOS STYLE_x FORM_y MODE_z KEY_k ...`. Scale-degree vocab expanded to 91 tokens.
- [x] Long-range thematic recall metric — `score_thematic_recall()` extracts subject from SUBJECT markers or first 2 bars, converts to interval sequence, searches all voices from bar 5+ for matching fragments with inversion detection and multi-voice bonus. Wired into scorer at weight 0.10 (statistical and structural each reduced from 0.20 to 0.15 to accommodate).
- [x] DroPE infrastructure — `use_rope` flag threaded through architecture and trainer, `attn_temperature` parameter for inference scaling, `recalibrate_drope()` method in Trainer (saves pre_drope checkpoint, creates fresh optimizer, runs training with `use_rope=False`, saves drope_final checkpoint, sets `config.drope_trained = True`). CLI flags `--drope`, `--drope-epochs`, `--drope-lr` added to train command. Config fields `drope_trained` and `drope_train_seq_len` added to ModelConfig. Ready to run once base model is validated.

### DroPE: Dropping Positional Embeddings After Pretraining (infrastructure done, experiment pending)

**What it is.** Gelberg et al. (2025) show that RoPE is essential during training (it accelerates convergence by providing a strong positional inductive bias) but is also the main barrier to zero-shot context extension. Their solution: train with RoPE normally, then drop all positional embeddings for a short recalibration phase at the original context length. The resulting model generalizes to sequences far beyond its training context with no long-context finetuning, outperforming YaRN/PI/NTK-RoPE scaling.

**Why it matters here.** Our current max_seq_len of 2048 is adequate for inventions (~20-40 bars) but will become the binding constraint for fugues, sonata-form movements, and any piece longer than ~60 bars of dense texture. DroPE would let us train at 2048 and generate at 4096+ without needing long-context training data we don't have.

**Why it's not urgent.** The immediate bottlenecks are data, evaluation quality, and getting the base models validated — not context length. DroPE's payoff scales with piece length, and we aren't generating long-form pieces yet. Additionally, the paper validates on 360M-7B models; our ~6.4M model with 8 layers is an untested regime. Fewer layers means less capacity to recover implicit positional encoding from the causal mask alone.

**Music-specific concern.** Metric position is more load-bearing in music than in prose. The model needs to know "beat 1 vs beat 3" for rhythmic patterns, cadence placement, and harmonic rhythm. After dropping RoPE, all of that must be recovered from BAR tokens, BEAT tokens, causal masking, and learned patterns. The BEAT_1-6 tokens (now implemented) provide explicit metric position in the token stream, which significantly de-risks this concern. Two remaining mitigations if needed: (1) verify BEAT tokens are sufficient for metric recovery during recalibration, and (2) try the hybrid-layer variant below.

**How to implement.** Already built. The procedure:
1. Train the model normally with RoPE for ~90% of total training steps.
2. Run `bach train --drope --drope-epochs 10 --drope-lr 1e-3` (or call `trainer.recalibrate_drope()` directly). This saves a `pre_drope.pt` checkpoint, creates a fresh optimizer, runs training with `use_rope=False` passing `cos=None, sin=None` to all attention layers, and saves `drope_final.pt` with `config.drope_trained = True`.
3. At inference on longer sequences, apply softmax temperature scaling via the `attn_temperature` parameter: β* = 1 + c·ln(s), where s = seq_len/train_seq_len. Tune c on a held-out set by minimizing perplexity.
4. Optional: add QKNorm if recalibration is long or unstable at higher learning rates.

**Hybrid-layer variant.** If full DroPE degrades local rhythmic/metric precision, try dropping RoPE only in the upper layers (e.g., layers 5-8) while keeping it in the lower layers (1-4). Rationale: the paper's own analysis shows high-frequency RoPE components (which dominate early layers and handle local positional patterns like "previous token" and "current token" attention) aren't the problem for context extension — it's the low-frequency components in upper-layer semantic heads that get distorted by scaling. Keeping RoPE in lower layers preserves beat/bar awareness; removing it in upper layers frees long-range thematic attention. Implementation is trivial — per-layer flag controlling whether `cos/sin` are passed. Run this as a second variant if full DroPE shows metric degradation.

**When to do it.** Run as a one-afternoon experiment once the base 2-part model is trained and validated. The real payoff comes in Phase 4 when we need long-form generation. Experiment order: (1) full DroPE, evaluate on 32-bar generation checking metric alignment and cadence placement, (2) if metric quality degrades, try hybrid-layer variant, (3) if both work, test zero-shot extension to 2× training context using the long-range thematic recall metric.

---

## Phase 2: First Conditioning Expansions

Start with the 2-3 highest-impact conditioning dimensions. Validate that the model actually responds to them before adding more. Each conditioning dimension fragments the already-small training data — every bucket needs enough examples for the model to learn meaningful distinctions. Add no more than 2-3 at a time and validate before adding more.

**Start with:**
- **Form token** — `FORM_CHORALE`, `FORM_INVENTION`, `FORM_FUGUE`, `FORM_SINFONIA`, `FORM_QUARTET`, `FORM_TRIO_SONATA` (**done** — implemented in Phase 1b alongside voice-count tokens). The model can now distinguish between different forms and voice counts independently. Next step: validate that the model actually responds to form tokens during training. **Implementation note:** the current `FORM_TO_MODE_TOKEN` mapping conflates form with voice count — "chorale" maps to `MODE_4PART`, making it identical to any 4-voice piece. This needs to be separated into independent form tokens (`FORM_CHORALE`, etc.) and voice-count tokens (`MODE_2PART`, etc.) so a 4-voice chorale and a 4-voice quartet are distinguishable. This is a vocab-breaking change — bundle it with the next full training run.
- **Length control** — conditioning token bucketing pieces into short/medium/long/extended, trained from actual measure counts. The model learns to pace differently for 16 bars vs 64 bars. Highest impact after form, easiest to validate.
- **Texture** — homophonic vs polyphonic, rhythmic independence between voices. Perceptually obvious, directly controllable. Critical for eventual string quartet generation where texture shifts fluidly within a single piece. Partially redundant with form (chorales are mostly homophonic, fugues are polyphonic), but captures within-form variation that form tokens can't.

**High-value shortlist (add next, one at a time):**
- **Imitation density** — how much voices copy each other (fugue vs free counterpoint vs chorale). Algorithmically detectable via melodic subsequence matching between voices at different time offsets. The most perceptually salient difference between fugue/invention/chorale.
- **Harmonic rhythm** — how fast chords change (distinct from note density). Computable by reducing each beat to a pitch-class set.
- **Meter** — 3/4 vs 4/4 vs 6/8 vs alla breve. Already in the score, strong perceptual feature, costs nothing to extract.

**Add later (once the above are validated):**
- Harmonic tension — ratio of dissonance, chromatic density, suspension frequency
- Rhythmic character — note density, dotted rhythms, syncopation level
- Melodic contour — ascending vs descending tendency, leap vs step ratio
- Cadential behavior — frequency, strength, deceptive vs authentic

All of these can be computed programmatically from the training corpus and added as conditioning tokens using the same pattern as style tokens. The binding constraint is data density vs conditioning dimensions — not imagination.

**Note:** Phase 2 conditioning tokens sequenced across sections (e.g., tension=low → tension=building → tension=high → tension=resolved) can approximate narrative arcs and musical development before Phase 4 structural planning is built. Combined with the Phase 6 orchestrator, this gives crude but functional story-like progression.

**Mixed seq_len training.** The current code uses different default seq_len per form (chorales 1024, fugues 2048). When training a single model on mixed data with the curriculum approach, you need to either: (a) train at seq_len 2048 for everything and accept wasted padding on short chorales, (b) implement variable-length batching where sequences of similar length are grouped together, or (c) train at 1024 first (all forms), then extend to 2048 when fine-tuning on fugues/longer forms. Option (c) aligns naturally with curriculum learning and avoids the padding waste. Option (b) is more principled but adds engineering complexity. Either way, this is a decision that should be made explicitly rather than discovered mid-training.

---

## Corpus Expansion (ongoing background work)

Not a gate for other phases — downloading, parsing, deduplicating, and validating new data can happen concurrently with model training and conditioning work. The goal is to expand the training corpus into stylistically compatible repertoire that the current architecture handles natively: N independent monophonic voices over stable key centers with primarily diatonic harmony.

### Current sources
- **music21 built-in corpus** — Bach chorales, some Bach keyboard works, miscellaneous. Reliable editorial quality, already integrated.
- **KernScores** (kern.ccarh.org) — kern format. Broad coverage of Baroque and Classical repertoire. High editorial quality. Already integrated via `download_kernscores.sh`.

### New sources to add

**OpenScore String Quartets** (https://github.com/OpenScore/StringQuartets/) — ~100 multi-movement quartets in MusicXML format. This is the single highest-value addition. Haydn, Mozart, and Beethoven quartets are exactly the target repertoire: four independent voices, standard ranges, clear voice identity, stable key centers. MusicXML parses cleanly via music21. Estimated yield: 300-400+ movements, potentially doubling the training corpus for 4-voice textures.

**Josquin Research Project (JRP)** (https://josquin.stanford.edu/) — kern format, so it drops straight into the existing pipeline with zero format work. Josquin, Ockeghem, Obrecht, La Rue, and contemporaries. Strict voice independence, diatonic/modal framework, 3-5 voice textures. The scale-degree tokenizer handles modal harmony cleanly (Dorian = minor with natural 6th). Estimated yield: 1,000+ works, though many are short motets.

**Choral Public Domain Library (CPDL)** (https://www.cpdl.org/) — large collection of vocal polyphony in mixed formats (MusicXML, MIDI, Lilypond, PDF). Palestrina, Victoria, Lassus, and hundreds of other Renaissance/Baroque composers. The catch is format heterogeneity and variable editorial quality — needs filtering for parseable formats (MusicXML/MIDI) and validation that voice separation is clean. Lower effort-to-yield ratio than OpenScore or JRP, but covers composers not available elsewhere.

### Lower priority (bookmarked)
- **Algomus group datasets** (https://algomus.fr/data/) — fugue and sonata datasets with structural annotations. Not useful as training data directly, but valuable for Phase 4 structural planning as ground truth for templates.
- **MuseData** (https://www.musedata.org/) — Baroque and Classical in MuseData format. music21 can parse it, but likely overlaps heavily with what we already get from KernScores. Check before investing effort.
- **Annotated Beethoven Corpus / When in Rome / TAVERN** — harmonic analysis datasets (Roman numeral annotations), not scores. Could become valuable for harmony conditioning in Phase 2 or structural planning in Phase 4.

### Deduplication
When merging sources, deduplicate at the metadata level before tokenization. Match on composer + catalogue number (BWV, K, Hob, Op., etc.) or title. When two sources have the same piece, keep whichever parses more cleanly (generally: KernScores > OpenScore > CPDL for editorial quality). This is a simple dictionary check during corpus building, not a research problem.

More important than deduplication: ensure the train/eval split is by piece, not by chunk. If chunks from the same piece end up on both sides, eval loss looks artificially good. Split by piece ID first, then chunk.

### Priority at a glance

| Source | Format | Repertoire | Voice count | Effort | Yield |
|---|---|---|---|---|---|
| OpenScore String Quartets | MusicXML | Classical quartets (Haydn, Mozart, Beethoven) | 4 | Low | High |
| JRP | kern | Renaissance polyphony (Josquin, Ockeghem, etc.) | 3-5 | Low | High |
| CPDL | Mixed | Renaissance/Baroque vocal (Palestrina, Victoria, Lassus) | 3-6 | Medium | Medium |

### Existing targets
- **Baroque trio sonatas** — Corelli, Handel, Telemann, Buxtehude. Two melody voices over bass. Fits 3-voice (sinfonia) mode. Harmonically and contrapuntally compatible with Bach. More lyrical, more sequential, more predictable harmonic rhythm. Source: KernScores, MuseData, and individual CCARH editions.
- **First generation target:** Haydn/Mozart minuet and trio movements. Short (16-32 bars per section), ABA form, simple harmonic plans, four independent voices. Structurally not much more complex than chorales — just classical style instead of baroque. Achievable with Phase 2 texture conditioning.

**Representation boundary:** This architecture handles any music organized as N independent monophonic voices over stable key centers with primarily diatonic harmony. That's its natural habitat — Bach, Palestrina, Haydn quartets, trio sonatas. It does not extend to keyboard-idiomatic writing (Chopin, Liszt), jazz, or textures where voice identity is fluid.

---

## Phase 3: Voice-by-Voice Generation

Pull this forward because it's a conditioning change on the existing architecture (not a new model) and immediately unlocks the most compelling user interaction: "here's my melody, harmonize it."

- **Sequential voice generation** — generate the subject voice first, then generate counterpoint voice(s) conditioned on the first. Mirrors how actual counterpoint is written.
- **User-provided voice** — hand-write or import one voice, have the model compose the others. The subject mechanism already hints at this; this phase makes it first-class.
- **Per-voice regeneration** — regenerate just one voice without touching the others. Enables targeted iteration.
- **Evaluation via reconstruction** — condition on real Bach voice 1, generate voice 2, compare against what Bach actually wrote. The strongest available evaluation method: measures whether the model can produce counterpoint that a master composer would recognize as valid against their own melody.

### Acid test: cross-style counterpoint

Import a melody the model has never seen from outside its training domain and have it write a fugue, canon, or harmonization around it. This tests whether the model learned counterpoint as a generalizable skill or just memorized Baroque melodic patterns. The scale-degree tokenizer makes this possible — any tonal melody reduces to the same scale-degree vocabulary regardless of its stylistic origin.

Test melodies to try (in rough order of difficulty):
- **Pop/rock vocal melody** — "Let It Be," "Yesterday," "Hallelujah." Diatonic, stepwise, regular phrases. Should be the easiest case — these are harmonically simple and the scale-degree representation strips away everything that makes them "pop."
- **Jazz standard melody** — "Autumn Leaves," "All The Things You Are," "Blue Bossa." More chromatic, more leaps, irregular phrase lengths. Tests whether the model can handle accidentals and modulatory melodies as fugue subjects.
- **Romantic piano melody** — Chopin Nocturne theme, Liszt Consolation, Grieg Peer Gynt. These are originally keyboard-idiomatic and not contrapuntal at all — extracting the top-voice melody and recontextualizing it as a fugue subject is a hard transformation. If the model writes convincing counterpoint against a Chopin melody, it's genuinely learned the skill.

This isn't a milestone — it's a demo target and a generalization test. It ties together Phase 3 (voice-by-voice generation), Phase 7 (cross-modal import), and the Phase 6 orchestrator (user says "make this a fugue," system figures out the rest). It's also the kind of thing that makes non-musicians immediately understand what the project does.

---

### The form style vs. form structure gap

After Phase 2 conditioning and Phase 3 voice-by-voice generation, the model will know *that* it should write a fugue (via `FORM_FUGUE` token) and will produce fugal *texture* — imitative entries, independent voices, polyphonic density. But it won't know *how* to structure a fugue: subject in tonic at bar 1, answer in dominant at bar 5, countersubject running alongside, episode at bar 9 developing fragments, middle entries in related keys, stretto near the end.

This is the hardest gap in the roadmap. Texture is learnable from next-token prediction because it's a local property — "this sounds like a fugue moment to moment." Structure is a global property — "this piece is organized as a fugue from start to finish" — and requires planning across 30+ bars. A model that generates token by token has no mechanism for this kind of forward planning, no matter how good the conditioning tokens are.

Expect to get stuck here. The model will produce 20-bar pieces labeled "fugue" that start with something resembling a subject entry and then drift into generic polyphony. Rejection sampling might surface candidates where the subject happens to return by chance, but it won't be reliable. Conditioning tokens can tell the model the *mood* of a fugue but not the *blueprint*.

This is what Phase 4 exists to solve. The structural planning model (or template system) provides the blueprint — "subject entry here, episode there, stretto at bar 20" — and the note-level model fills in each section. Until Phase 4 is built, fugue generation will produce fugal *flavor* without fugal *form*. That's OK as long as you don't mistake one for the other when evaluating.

**What to do in the meantime:** lean into forms where structure is simple enough for the model to learn implicitly. Chorales (AABB, 16 bars, predictable cadence points), short inventions (~20 bars, ABA-ish), and minuet/trio movements (16+16 bars, binary form). These are structurally simple enough that next-token prediction plus conditioning tokens might be sufficient. Save fugues and sonata form for Phase 4.

---

## Phase 4: Structural Planning

Before generating note-by-note, generate a high-level structural plan. This is a bigger architectural lift — likely requires a second model or a two-pass architecture. Required for long-form generation (fugues, sonata-form movements, anything beyond ~32 bars).

- **Structural templates** — "A section (8 bars, tonic) → sequence (4 bars, modulating) → B section (8 bars, dominant) → retransition (4 bars) → A' (8 bars, tonic)." The note-level model fills in each section given the plan.
- **Measure-addressable sequences** — leverage the existing `BAR` token to make the model aware of measure boundaries as meaningful structural units.
- **Multi-model ensemble** — specialized models for harmonic planning (chord progression outline), melodic generation (fill voices given harmony), and rhythmic variation. The orchestrator composes them. Each model is simpler and more controllable than one monolith.

**Unlocks:** Full string quartet first movements (sonata form), complete fugues with exposition/episode/stretto structure, any piece longer than ~32 bars that needs coherent large-scale form.

**Context window.** Long-form generation will likely exceed the 2048-token training context. If the Phase 1 DroPE experiment validates, this is where it pays off — zero-shot extension to 4096+ tokens without long-context finetuning. If DroPE doesn't work well at our model scale, fall back to YaRN scaling or simply increase the training context (costly with small data). Either way, long-form coherence is fundamentally a structure problem, not just a context window problem — a model that can attend over 4096 tokens still can't maintain sonata form without explicit structural planning.

---

## Phase 5: Iteration and Editing

The highest-leverage feature for real-world use. Real composition is iterative, not fire-and-forget.

- **Infilling** — accept "here's bars 1-22 and bars 26-30, fill in 23-25." Requires masked span prediction training objective alongside the autoregressive one. Note: RoPE handles the position gap naturally — keep original position indices for the right context. A DroPE model may actually handle infilling better here: without explicit positional encoding, the model attends purely on content and isn't confused by a gap in position indices.
- **Constraint propagation** — regenerated bars connect smoothly to surrounding context (voice leading, harmonic continuity, rhythmic flow).
- **Interactive playback with hot-swapping** — listen to a generation, pause at a point you don't like, describe what you want differently, regenerate from that point forward while keeping everything before it.

---

## Phase 6: AI Orchestrator Layer

The user never interacts with the model directly. A language model (Claude) sits on top and translates natural language intent into model parameters and tool calls.

**Critical dependency: Phase 2 is the load-bearing wall here.** The orchestrator is only as good as the control surface it has to work with. "Melancholic but hopeful" → concrete parameters is the hard mapping, and if the model doesn't reliably respond to conditioning tokens, Claude can't bridge that gap. Don't start this phase until Phase 2 conditioning is validated.

- **Natural language to musical parameters** — "I want 30 measures of melancholic but hopeful 2-part Bach, with strong moments of tension" maps to: style=bach, mode=2-part, length=long, tension=high, contour=mixed, mood tags.
- **Narrative arc scripting** — the orchestrator sequences conditioning tokens across sections to create musical development ("start quiet, build tension in the middle, resolve at the end"). This is the primary path to story-like progression — the human provides the narrative sense, the model executes each moment.
- **Iterative refinement** — "measures 23-25 don't sound good, adjust the harmony to use more secondary dominants" → Claude masks bars 23-25, adds harmonic constraints, calls the model with surrounding context.
- **Subject/idea translation** — accept any form of musical input (note names, hummed melody, chord chart, vague description) and Claude converts it to the format the model expects.
- **Reference-based generation** — "generate something that sounds like BWV 784 but in D minor." Encode a reference piece and use it as additional conditioning.

---

## Phase 7: Cross-Modal Input

Meet users where they are — accept musical ideas in any format.

**Dependencies and feasibility.** Most of these rely on external tooling that's already mature, not on changes to our model architecture. MIDI import and MusicXML/kern parsing are straightforward (music21 handles both). Chord chart parsing is simple text processing. Hummed melody transcription depends on third-party models (e.g., CREPE, Basic Pitch) which are good enough for monophonic input. Melodic contour from a sketch is the most speculative item and lowest priority.

- **MIDI file import** — parse a MIDI clip as the subject or as a reference piece. Easiest to implement, highest immediate value.
- **MusicXML/kern snippet import** — extract melody from notation and convert to subject format. Also straightforward via music21.
- **Chord charts** — text ("Am - F - C - G") → harmony conditioning tokens. Requires Phase 2 harmony conditioning to be useful.
- **Hummed melody** — audio → MIDI transcription → subject tokens. Depends on external pitch tracking, but mature options exist.
- **Melodic contour drawing** — image/sketch → pitch envelope → constrained generation. Most speculative, lowest priority.

---

## Phase 8: Advanced Style Control

Move beyond discrete categories to continuous, fine-grained style control.

**Architecture note.** Style interpolation and transfer require replacing the discrete style token with a continuous embedding space — this is a meaningful change to the training setup, not just a conditioning extension. The model would need to be retrained (or at least fine-tuned) with a style encoder that maps pieces to vectors rather than categories. This phase is an architectural fork from the discrete-token approach used in Phases 1-6, and should only be pursued once that approach has been fully exploited.

- **Style interpolation** — learn a continuous style embedding space. "70% Bach, 30% Mozart" becomes a weighted vector blend rather than a discrete token.
- **Style transfer** — take a piece in one style and re-render it in another while preserving structure.
- **Era-adaptive training** — as the corpus grows, the model learns finer stylistic distinctions within eras (early vs late Beethoven, North vs South German Baroque, etc.).

---

## Long-Term Aspirations

Ideas worth exploring once the foundation is mature, but not concrete milestones yet.

- **Hybrid reward model for RL** — a learned discriminator ("does this look like curated baroque corpus or random MuseScore?") as primary reward signal, combined with the rule-based evaluation scorer as constraint/bonus. The discriminator learns the joint distribution of good music; the rules enforce hard guarantees. Training data: curated kern files as positive class, rejected/low-quality pieces as negative class.
- **Rejection sampling against reward model** — the practical first step before full RL. Generate candidates, score with the hybrid reward model, keep the best. Gets 80% of RL benefit with zero training instability. Already set up via best-of-N infrastructure. Only escalate to PPO/DPO if rejection sampling proves insufficient.
- **Preference learning (RLHF)** — user rates outputs (A vs B comparisons) to train a reward model. Per-user preference profiles rather than global "good music." Requires enough users and enough preference signal to be meaningful — the deterministic evaluation scorer is the practical quality lever for a long time.
- **Active learning** — the system identifies which comparisons are most informative and asks for those ratings specifically, minimizing annotation burden.

---

## Guiding Principles

1. **The human drives.** Taste, direction, and creative decisions always belong to the user. The system removes the music theory barrier, not the creative agency.
2. **Iterate, don't generate.** One-shot generation is a demo. Real value comes from the ability to refine, adjust, and develop ideas collaboratively.
3. **Representation is the first lever.** Encoding music theory into the tokenizer is worth more than brute-force data or bigger models. The scale-degree decomposition — making pitch key-invariant and compositional — is the foundational technical bet. Architectural and data decisions flow from it.
4. **Compute tags, don't hand-label.** Wherever possible, derive conditioning features programmatically from the corpus. Manual annotation doesn't scale.
5. **Keep the base model simple.** Intelligence lives in the orchestrator layer. The generation model should be precise and controllable, not clever.
6. **Validate before expanding.** Add one conditioning dimension, prove the model responds, then add the next. Resist the pull of Phase N+1 features before Phase N is solid.
7. **Vibe composing.** If the user has to think about voice leading rules, we've failed.
