#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

uv run python - << 'EOF'
from pathlib import Path
from music21 import converter
from analyze_kernscores import _compute_tension_ratio
from bach_gen.data.extraction import VoiceComposition
from bach_gen.data.analysis import compute_harmonic_tension

MIDI_DIR = Path('/Users/tfokkens/Documents/Claude/2pt-bach/data/midi')
skip_tokens = ['-auto','-combined','-beat','-sampled','-pan','-nopan','extractf','-20','-60','-80','-S']

rows = []
for krn in sorted(MIDI_DIR.rglob('*.krn')):
    stem = krn.stem
    if any(x in stem for x in skip_tokens):
        continue
    try:
        score = converter.parse(str(krn))
        parts = score.parts
        if len(parts) < 2 or len(parts) > 4:
            continue

        voices = []
        for part in parts:
            notes = [
                (int(el.offset * 480), int(el.quarterLength * 480),
                 el.pitch.midi if hasattr(el, 'pitch') else el.pitches[0].midi)
                for el in part.flatten().notes
                if int(el.quarterLength * 480) > 0
            ]
            if notes:
                voices.append(notes)
        if len(voices) < 2:
            continue

        ratio = _compute_tension_ratio(voices)
        comp = VoiceComposition(voices=voices, key_root=0, key_mode='major', source=stem)
        label = compute_harmonic_tension(comp)
        rows.append((ratio, label, str(krn)))
    except Exception:
        pass

rows.sort(key=lambda x: x[0], reverse=True)
mx = rows[0][0] if rows else 0.0
print(f"max_tension_ratio={mx:.6f}")
print("top 10:")
for ratio, label, path in rows[:10]:
    print(f"{ratio:.6f}\t{label}\t{path}")

print("\nall ties for max:")
for ratio, label, path in rows:
    if abs(ratio - mx) < 1e-12:
        print(f"{ratio:.6f}\t{label}\t{path}")
EOF
