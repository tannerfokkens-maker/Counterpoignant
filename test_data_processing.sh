#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./test_data_processing.sh [midi_dir] [out_dir]
# Defaults:
#   midi_dir: /Users/tfokkens/Documents/Claude/2pt-bach/data/midi
#   out_dir : ./tmp/data_processing_test_YYYYmmdd_HHMMSS

MIDI_DIR="${1:-/Users/tfokkens/Documents/Claude/2pt-bach/data/midi}"
OUT_DIR="${2:-./tmp/data_processing_test_$(date +%Y%m%d_%H%M%S)}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/prepare_data.log"

printf "== Data Processing Test ==\n"
printf "Repo: %s\n" "$(pwd)"
printf "MIDI_DIR: %s\n" "$MIDI_DIR"
printf "OUT_DIR: %s\n" "$OUT_DIR"
printf "LOG_FILE: %s\n\n" "$LOG_FILE"

if [[ ! -d "$MIDI_DIR" ]]; then
  echo "ERROR: MIDI dir not found: $MIDI_DIR" >&2
  exit 1
fi

# Run with project defaults currently in CLI:
# mode=all, tokenizer=scale-degree, max-seq-len=4096,
# max-source-voices=4, max-groups-per-work=1, max-pairs-per-work=2,
# pair-strategy=adjacent+outer, sonata-policy=counterpoint-safe
#
# Prefer installed console script, but fall back to module execution with
# PYTHONPATH=src for editable-less/dev environments.
if UV_CACHE_DIR="$UV_CACHE_DIR" uv run bach-gen --help >/dev/null 2>&1; then
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run bach-gen prepare-data \
    --data-dir "$OUT_DIR" \
    | tee "$LOG_FILE"
else
  echo "Info: 'uv run bach-gen' unavailable; using module fallback (PYTHONPATH=src)."
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run env PYTHONPATH=src \
    python -m bach_gen.cli prepare-data \
    --data-dir "$OUT_DIR" \
    | tee "$LOG_FILE"
fi

printf "\n== Parsed Summary ==\n"
python - "$OUT_DIR" << 'PY'
import json
import statistics
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
mode_path = out_dir / "mode.json"
seq_path = out_dir / "sequences.json"
pid_path = out_dir / "piece_ids.json"
tok_path = out_dir / "tokenizer.json"

missing = [str(p) for p in (mode_path, seq_path, pid_path, tok_path) if not p.exists()]
if missing:
    print(json.dumps({"error": "missing output files", "missing": missing}, indent=2))
    raise SystemExit(1)

mode = json.loads(mode_path.read_text())
sequences = json.loads(seq_path.read_text())
piece_ids = json.loads(pid_path.read_text())
tok = json.loads(tok_path.read_text())

token_to_id = tok.get("token_to_id", {})
enc_i = token_to_id.get("ENCODE_INTERLEAVED")
enc_s = token_to_id.get("ENCODE_SEQUENTIAL")

lengths = [len(s) for s in sequences]

def pct(values, p):
    if not values:
        return 0
    arr = sorted(values)
    idx = int(round((p / 100.0) * (len(arr) - 1)))
    return arr[idx]

interleaved = 0
sequential = 0
for s in sequences:
    st = set(s)
    if enc_i is not None and enc_i in st:
        interleaved += 1
    if enc_s is not None and enc_s in st:
        sequential += 1

summary = {
    "mode_json": mode,
    "num_sequences": len(sequences),
    "num_piece_ids": len(piece_ids),
    "num_unique_pieces": len(set(piece_ids)),
    "encoding_counts": {
        "interleaved": interleaved,
        "sequential": sequential,
    },
    "sequence_len_stats": {
        "min": min(lengths) if lengths else 0,
        "median": int(statistics.median(lengths)) if lengths else 0,
        "p90": int(pct(lengths, 90)),
        "p99": int(pct(lengths, 99)),
        "max": max(lengths) if lengths else 0,
    },
}

print(json.dumps(summary, indent=2))
PY

printf "\nDone. Send me:\n"
printf "1) %s\n" "$LOG_FILE"
printf "2) The Parsed Summary JSON above\n"
