#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="models_NEW/finetune_best.pt"
CANDIDATES=200
TEMPERATURE=0.9
MIN_P=0.03

RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="$SCRIPT_DIR/output/overnight_runs/$RUN_TS"
MIDI_DIR="$RUN_DIR/midis"
LOG_DIR="$RUN_DIR/logs"
RUN_LOG="$RUN_DIR/run.log"

mkdir -p "$MIDI_DIR" "$LOG_DIR" "$SCRIPT_DIR/output"

cat > "$RUN_DIR/run_config.txt" <<EOF
timestamp=$RUN_TS
model_path=$MODEL_PATH
candidates=$CANDIDATES
temperature=$TEMPERATURE
min_p=$MIN_P
EOF

echo "Run folder: $RUN_DIR" | tee -a "$RUN_LOG"

run_and_archive() {
  local name="$1"
  local pattern="$2"
  shift 2

  local marker="$RUN_DIR/.${name}.marker"
  local log_file="$LOG_DIR/${name}.log"
  local moved=0

  : > "$marker"
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] START $name" | tee -a "$RUN_LOG"

  if "$@" 2>&1 | tee "$log_file"; then
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      mv "$f" "$MIDI_DIR/"
      moved=$((moved + 1))
    done < <(find "$SCRIPT_DIR/output" -maxdepth 1 -type f -name "$pattern" -newer "$marker" | sort)

    if [ "$moved" -eq 0 ]; then
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] WARN $name completed, no new files matched $pattern" | tee -a "$RUN_LOG"
    else
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE $name moved=$moved pattern=$pattern" | tee -a "$RUN_LOG"
    fi
  else
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAIL $name (see $log_file)" | tee -a "$RUN_LOG"
    rm -f "$marker"
    return 1
  fi

  rm -f "$marker"
}

run_and_archive "fugue_B_minor" "fugue_B_minor_*.mid" \
  uv run bach-gen generate --key "B minor" --model-path "$MODEL_PATH" --mode fugue --style bach --voices 4 --texture polyphonic --imitation high --candidates "$CANDIDATES" --temperature "$TEMPERATURE" --min-p "$MIN_P" --max-length 4096

run_and_archive "fugue_D_minor" "fugue_D_minor_*.mid" \
  uv run bach-gen generate --key "D minor" --model-path "$MODEL_PATH" --mode fugue --style bach --voices 4 --texture polyphonic --imitation high --candidates "$CANDIDATES" --temperature "$TEMPERATURE" --min-p "$MIN_P" --max-length 4096

run_and_archive "fugue_Eb_major" "fugue_Eb_major_*.mid" \
  uv run bach-gen generate --key "Eb major" --model-path "$MODEL_PATH" --mode fugue --style bach --voices 4 --texture polyphonic --imitation high --candidates "$CANDIDATES" --temperature "$TEMPERATURE" --min-p "$MIN_P" --max-length 4096

run_and_archive "chorale_B_minor" "chorale_B_minor_*.mid" \
  uv run bach-gen generate --key "B minor" --model-path "$MODEL_PATH" --mode chorale --style bach --voices 4 --texture homophonic --imitation none --candidates "$CANDIDATES" --temperature "$TEMPERATURE" --min-p "$MIN_P" --max-length 2048

run_and_archive "invention_D_minor" "invention_D_minor_*.mid" \
  uv run bach-gen generate --key "D minor" --model-path "$MODEL_PATH" --mode invention --style bach --voices 2 --texture polyphonic --imitation high --candidates "$CANDIDATES" --temperature "$TEMPERATURE" --min-p "$MIN_P" --max-length 2048

run_and_archive "sinfonia_Fs_minor" "sinfonia_Fs_minor_*.mid" \
  uv run bach-gen generate --key "F# minor" --model-path "$MODEL_PATH" --mode sinfonia --style bach --voices 3 --texture polyphonic --imitation high --candidates "$CANDIDATES" --temperature "$TEMPERATURE" --min-p "$MIN_P" --max-length 3072

echo "All done. Results: $RUN_DIR" | tee -a "$RUN_LOG"
