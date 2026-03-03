# Server Conditioning Token Check

Run this on your server from the project root:

```bash
cd ~/Counterpoignant
uv run python <<'PY'
import json
from pathlib import Path

data_dir = Path("data")
seqs = json.loads((data_dir / "sequences.json").read_text())
pids = json.loads((data_dir / "piece_ids.json").read_text()) if (data_dir / "piece_ids.json").exists() else [""] * len(seqs)
tok = json.loads((data_dir / "tokenizer.json").read_text())
id2name = {int(k): v for k, v in tok["token_to_name"].items()}

def prefix(seq, scan=160):
    out = []
    for t in seq[:scan]:
        n = id2name.get(t, "")
        out.append(n)
        if n.startswith("KEY_"):
            break
    return out

def has_pref(pref, p):
    return any(n.startswith(p) for n in pref)

core = {
    "FORM_": "form",
    "MODE_": "mode",
    "ENCODE_": "encoding",
    "KEY_": "key",
}
extra = {
    "STYLE_": "style",
    "LENGTH_": "length",
    "METER_": "meter",
    "TEXTURE_": "texture",
    "IMITATION_": "imitation",
    "HARMONIC_RHYTHM_": "harmonic_rhythm",
    "HARMONIC_TENSION_": "tension",
    "CHROMATICISM_": "chromaticism",
}

counts = {v: 0 for v in list(core.values()) + list(extra.values())}
bad = []

for i, s in enumerate(seqs):
    pref = prefix(s)
    miss_core = []
    for p, name in core.items():
        ok = has_pref(pref, p)
        counts[name] += int(ok)
        if not ok:
            miss_core.append(name)
    for p, name in extra.items():
        counts[name] += int(has_pref(pref, p))
    if miss_core:
        bad.append((i, len(s), pids[i], miss_core, pref[:20]))

n = len(seqs)
print(f"Total sequences: {n}")
for k, v in counts.items():
    print(f"{k:16s}: {v:6d} ({v/n:.2%})")

print(f"\nCore-prefix issues: {len(bad)} ({len(bad)/n:.2%})")
for row in bad[:10]:
    i, L, pid, miss, pref = row
    print(f"- idx={i} len={L} pid={pid} missing={miss}")
    print(f"  prefix={pref}")
PY
```

Interpretation:

- `Core-prefix issues: 0` means the dataset looks healthy for conditioning tokens.
- Non-zero means some sequences are missing one or more of `FORM/MODE/ENCODE/KEY`.
