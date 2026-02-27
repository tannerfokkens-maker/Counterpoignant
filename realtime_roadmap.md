# Real-Time & Long-Form Generation Roadmap

## Overview

Two complementary generation modes beyond the current single-shot 100-candidate approach:

1. **Chained Generation with Selection** — compose multi-sequence pieces (sonata movements, full fugues) by generating overlapping chunks with quality filtering at each step.
2. **Sliding Window Streaming** — generate infinitely long music in real-time, streamed to a MIDI device for live performance or installation.

Both depend on shared infrastructure (KV cache, sticky prefix, token-to-MIDI decoder) but serve different purposes: chained generation is a **composition tool**, streaming is a **performance tool**.

---

## Prerequisites (must be complete before starting)

### Model Quality
- [ ] Run 1 model validated: chorales, inventions, and quartets sound musically convincing in single-shot generation
- [ ] Scorer calibrated: composite scores reliably predict perceptual quality (listening tests confirm top-scored outputs are preferred)
- [ ] Temperature sweep complete: optimal temperature known per form

### Architecture
- [ ] **KV cache implemented** — required for both modes. Without it, each new token requires a full forward pass over the entire context. With it, generation drops to one forward pass per token. This is the single biggest prerequisite.
- [ ] Flash Attention in production (done — already in architecture.py)
- [ ] DroPE recalibration complete — enables generation beyond training seq_len

### Tokenizer
- [ ] Tokenizer finalized (no further vocab changes)
- [ ] Token-to-MIDI-event decoder that works **incrementally** — converts one token at a time to a MIDI note on/off event, without needing the full sequence

### Data & Training
- [ ] Corpus balanced (Baroque/Classical expanded to offset Renaissance dominance)
- [ ] Run 2 trained with balanced corpus
- [ ] Model generates convincing output at all target forms

---

## Mode 1: Chained Generation with Selection

### Concept

Generate long pieces by producing overlapping chunks, scoring each independently, and stitching the best candidates together. Enables composition of pieces far longer than the 4096-token context window.

### How It Works

```
Chunk 1: [PREFIX ............... 4096 tokens of new music]
                                 ↓ keep best of 100 candidates
Chunk 2:          [last 2048 tokens as context | 2048 new tokens]
                                                 ↓ keep best of 100
Chunk 3:                    [last 2048 tokens as context | 2048 new tokens]
                                                          ↓ keep best of 100
...stitch non-overlapping portions together
```

### Design Decisions

**Overlap size:** 50% (2048 tokens context, 2048 new) is the starting point. Too little overlap loses continuity. Too much wastes generation capacity on already-written material.

**Scoring per chunk:**
- Standard voice leading, counterpoint, statistical scores on the new tokens
- Cross-boundary score: does harmony resolve at the seam? Do voices maintain range?
- Continuity score: does the chunk sound like a natural continuation, or a fresh start?

**Section control (optional, Phase 4+):** Insert section tokens (SECTION_A, SECTION_DEV, etc.) at the start of each chunk to guide large-scale form. Chunk 1 gets SECTION_EXPOSITION, chunk 4 gets SECTION_DEVELOPMENT, chunk 7 gets SECTION_RECAP. The model follows the structural hint while the chained approach handles length.

### Length Estimates (4-voice quartet at 4096 seq_len)

| Chunks | Overlap | New bars/chunk | Total bars | Duration (~80bpm) |
|--------|---------|----------------|------------|--------------------|
| 1      | —       | ~35            | 35         | ~1:45              |
| 4      | 50%     | ~18            | 70         | ~3:30              |
| 8      | 50%     | ~18            | 140        | ~7:00              |
| 12     | 50%     | ~18            | 210        | ~10:30             |

8 chunks covers a full sonata-form movement. 12 covers a slow movement + minuet.

### Implementation Steps

1. **Implement KV cache** in the model's inference path
2. **Build chained generator:**
   - Accept a "seed" (the conditioning prefix or a user-provided theme)
   - Generate N candidates for chunk 1, score, keep best
   - Extract overlap window from best chunk
   - Generate N candidates for chunk 2 with overlap as context
   - Repeat for desired number of chunks
3. **Build stitcher:** concatenate non-overlapping portions, handle any note ties or rests at boundaries
4. **Add cross-boundary scoring** to the scorer
5. **Add section token injection** (optional, for structural control)
6. **Listening tests:** compare stitched multi-chunk output against single-chunk output for coherence

### CLI Interface (proposed)

```bash
# Generate a ~140 bar quartet in D major using 8 chunks
uv run bach-gen generate-long \
    --mode quartet \
    --key "D major" \
    --style classical \
    --chunks 8 \
    --overlap 2048 \
    --num-candidates 50 \
    --temperature 0.9
```

---

## Mode 2: Sliding Window Streaming

### Concept

Generate music token-by-token with no end, streaming each note to a MIDI output device in real-time. The context window slides forward as new tokens are produced, dropping old tokens that scroll past the window.

### How It Works

```
Time 0:   [PREFIX | tokens 1-4089 ...................]  → generate token 4090
Time 1:   [PREFIX | tokens 2-4089, 4090 .............]  → generate token 4091
Time 2:   [PREFIX | tokens 3-4089, 4090, 4091 .......]  → generate token 4092
...
```

### Sticky Prefix

Reserve the first ~512 tokens as permanent context that never scrolls off:
- Conditioning tokens (BOS, STYLE, FORM, MODE, LENGTH, METER, KEY)
- The fugue subject or opening theme (~100-400 tokens)

This ensures the model always remembers what it's supposed to sound like. The sliding window applies only to the remaining ~3584 tokens.

### Real-Time MIDI Pipeline

```
Model (KV cache) → Token decoder → python-rtmidi → MIDI device / DAW
                                         ↑
                                    Buffer (pre-generate ~2 bars ahead)
```

**Speed estimate:** With KV cache on M1 Mac, expect 50-200 tokens/second. At ~20 tokens per beat in 4-voice texture, that's 2.5-10 beats/second — well ahead of real-time playback at any reasonable tempo. Generate into a 2-bar buffer, drain at playback speed.

### Limitations

- No global structure — the model forgets everything beyond ~35 bars
- No scoring/selection — output quality is whatever the model produces
- No endings — the model never generates EOS unless prompted
- Quality depends heavily on temperature — too high and it drifts, too low and it loops

### Implementation Steps

1. **Implement KV cache** (shared prerequisite with chained generation)
2. **Build incremental token decoder:** maps individual tokens to MIDI events without waiting for a complete piece
3. **Build MIDI output bridge** using python-rtmidi
4. **Build sliding window manager:** maintains the context window, handles prefix pinning, manages the KV cache as tokens scroll off
5. **Add real-time controls:**
   - Temperature knob (adjust randomness live)
   - Key change injection (insert a new KEY token to modulate)
   - Style change injection (shift from Bach to Classical mid-stream)
   - Panic button (flush context, restart from prefix)
6. **Latency optimization:** profile and tune buffer size, generation batch size

### CLI Interface (proposed)

```bash
# Stream endless fugue to default MIDI output
uv run bach-gen stream \
    --mode fugue \
    --key "C minor" \
    --style bach \
    --temperature 0.9 \
    --midi-port "IAC Driver Bus 1"
```

---

## Shared Infrastructure

### KV Cache (Priority 1)

Both modes require KV cache for acceptable speed. Without it:
- Chained generation: 100 candidates × 8 chunks × full 4096-token forward pass = very slow
- Streaming: recomputing 4096 tokens per generated note = not real-time

With KV cache, each new token requires only a single-layer forward pass with cached key/value pairs from previous tokens.

**Implementation:** Add `past_key_values` parameter to the model's forward pass. Each attention layer returns its K, V tensors. On the next call, pass them back in so only the new token's Q, K, V are computed.

### Incremental Token Decoder (Priority 2)

Current decoding assumes a complete sequence. Streaming needs note-by-note decoding:
- Track current voice (from VOICE_1/2/3/4 tokens)
- Track current beat position (from BEAT tokens)
- Track current bar (from BAR tokens)
- Emit MIDI note-on when a pitch+duration token pair is decoded
- Emit MIDI note-off after the duration elapses

### DroPE for Extended Context (Priority 3)

After DroPE recalibration, the model can generate beyond its training seq_len. This extends the effective window for both modes:
- Chained generation: larger chunks = fewer seams = better coherence
- Streaming: longer memory before tokens scroll off

---

## Phase Timeline

| Phase | What | Depends On |
|-------|------|------------|
| Current | Single-shot 100-candidate generation | Done |
| Next | KV cache implementation | Architecture change |
| Next | Chained generation prototype (2-3 chunks) | KV cache, scorer |
| Later | Full chained generation with section tokens | Phase 4 section tokens |
| Later | Sliding window streaming prototype | KV cache, incremental decoder |
| Later | Real-time MIDI streaming with live controls | python-rtmidi, streaming prototype |
| Future | Interactive mode: play a theme, model continues | Streaming + MIDI input |
