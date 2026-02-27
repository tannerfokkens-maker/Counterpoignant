# Counterpoignant — What This Project Does

## The One-Sentence Version

A computer program that writes music in the style of Bach and other classical composers, where you tell it what kind of piece you want and it composes it for you.

## How It Works (The Big Picture)

Imagine you had a very dedicated music student who listened to thousands of pieces by Bach, Mozart, Palestrina, and other composers. After enough listening, they start to notice patterns — how melodies move, which notes sound good together, what makes a fugue sound like a fugue versus a chorale. Eventually, this student can write new pieces that sound like they belong in the same tradition.

That's roughly what this program does, except the "student" is a neural network (a type of AI) and the "listening" is training on a large collection of sheet music.

### The Key Insight: Scale Degrees, Not Piano Keys

Most music AI systems think in terms of specific piano keys — "play the note C, then D, then E." Our system thinks in terms of *relationships* — "play the root note, then the second scale degree, then the third." This means the same melody in C major and the same melody in F# major look identical to the model. It learns *music theory* rather than *key positions*. This is the core technical bet of the project.

### What You Can Control

When you ask the system to compose, you can specify:

- **Key** — C minor, D major, etc.
- **Style** — Bach, Baroque, Renaissance, Classical
- **Form** — Chorale (hymn-like), Invention (2-part), Fugue (complex imitative), Quartet, etc.
- **How many voices** — 2, 3, or 4 independent melodic lines
- **Length** — Short, medium, long, extended
- **Meter** — 4/4, 3/4, 6/8, etc.
- **Texture** — Are the voices moving together (homophonic) or independently (polyphonic)?
- **Imitation** — How much do the voices copy each other's melodies?
- **Harmonic rhythm** — How fast the underlying chords change
- **Harmonic tension** — How much dissonance and chromaticism (notes outside the key)

You can also give it one melody and have it write the other voices around it.

### How It Picks the Best Output

The system doesn't just write one piece and call it done. It writes 100 different attempts, then scores each one on seven different quality measures (Does it follow voice-leading rules? Does its note distribution match real Bach? Does the opening theme come back later?). It keeps only the top 3.

## What Just Happened (The Recent Work)

We made a batch of changes that all needed to happen together because they change the vocabulary — the set of "words" the model understands. Once you change the vocabulary, every previously trained model and every cached dataset becomes incompatible. So we bundled everything:

1. **Five new musical analysis dimensions** — The system now automatically analyzes each training piece for texture, imitation level, harmonic rhythm, and harmonic tension, and tags each piece with those labels. The thresholds for each label were calibrated against the actual training corpus to produce roughly even distributions (about a third of pieces in each bucket).

2. **Two ways to encode music** — Previously, all voices were woven together chronologically (like reading a score left to right). Now there's also a "voice-by-voice" format where each voice is written separately from beginning to end, then the next voice, then the next. The model learns both formats. This is how human composers actually work — write the melody first, then add the bass line, then fill in the middle voices.

3. **Provide-a-voice generation** — You can hand the system a melody (as a MIDI file) and it will compose the remaining voices around it. This is the most directly useful creative tool: hum something, convert it to MIDI, and get a full harmonization back.

## What's Left Before Training

The vocabulary is now stable at **117 tokens** (scale-degree mode) or **151 tokens** (absolute-pitch mode). The remaining conditioning dimensions from the roadmap (rhythmic character, melodic contour, cadential behavior) can wait — they'd break the vocab again, and we need to validate that the model actually responds to the five new dimensions before adding more.

The next step is: run `prepare-data` on the full corpus, check the histograms, train, and listen.

## The Bigger Vision

Right now you have to specify all the musical parameters yourself. The long-term goal is to put a conversational AI layer on top so you can say "write me something melancholic but hopeful, about two minutes long" and it figures out the right parameters. The generation model is the engine; the conversation layer is the steering wheel. We're building the engine first.

There's also a structural gap: the model can produce music that *sounds like* a fugue moment to moment, but it can't *plan* a fugue (subject entry at bar 1, answer at bar 5, episode at bar 9). That requires a separate planning system that doesn't exist yet. For now, the model works best on shorter, structurally simple forms — chorales, short inventions, minuets.
