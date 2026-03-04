#!/usr/bin/env bash
# Download OpenScore String Quartets and copy all score assets into
# data/midi/openscore_quartets/<composer>/ with collision-safe names.

set -euo pipefail

REPO_URL="https://github.com/OpenScore/StringQuartets.git"
CLONE_DIR="/tmp/openscore_string_quartets"
SCORES_DIR="$CLONE_DIR/scores"
OUT_BASE="data/midi/openscore_quartets"

composer_slug() {
    python3 - "$1" <<'PY'
import re
import sys
import unicodedata

raw = sys.argv[1]
head = raw.split(",", 1)[0]
head = unicodedata.normalize("NFKD", head).encode("ascii", "ignore").decode("ascii")
head = re.sub(r"[^a-z0-9]+", "", head.lower())
print(head or "composer")
PY
}

safe_rel_name() {
    local rel="$1"
    rel="${rel//\//__}"
    rel="${rel// /_}"
    printf '%s\n' "$rel"
}

echo "=== OpenScore String Quartets Download ==="

if [ -d "$CLONE_DIR/.git" ] && git -C "$CLONE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Repository already cloned, pulling latest..."
    git -C "$CLONE_DIR" pull --ff-only
else
    if [ -d "$CLONE_DIR" ]; then
        echo "Existing clone is missing/broken, recloning..."
        rm -rf "$CLONE_DIR"
    else
        echo "Cloning repository..."
    fi
    git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
fi

if [ ! -d "$SCORES_DIR" ]; then
    echo "Missing scores directory: $SCORES_DIR"
    exit 1
fi

mkdir -p "$OUT_BASE"

total_new=0
total_existing=0
total_found=0
composer_count=0

while IFS= read -r -d '' composer_dir; do
    composer_count=$((composer_count + 1))
    composer_raw=$(basename "$composer_dir")
    composer_out=$(composer_slug "$composer_raw")
    dest_dir="$OUT_BASE/$composer_out"
    mkdir -p "$dest_dir"

    new=0
    existing=0
    found=0

    while IFS= read -r -d '' src; do
        found=$((found + 1))
        rel="${src#$composer_dir/}"
        safe_rel="$(safe_rel_name "$rel")"
        dest="$dest_dir/$safe_rel"

        if [ -f "$dest" ]; then
            existing=$((existing + 1))
            continue
        fi

        cp "$src" "$dest"
        new=$((new + 1))
    done < <(
        find "$composer_dir" -type f \
            \( -iname "*.musicxml" -o -iname "*.mxl" -o -iname "*.xml" -o -iname "*.mscx" -o -iname "*.mscz" \) \
            -print0
    )

    total_new=$((total_new + new))
    total_existing=$((total_existing + existing))
    total_found=$((total_found + found))

    n=$(find "$dest_dir" -type f | wc -l | tr -d ' ')
    echo "  $composer_out ($composer_raw): $n files ($new new, $existing existing)"
done < <(find "$SCORES_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

echo ""
echo "=== Summary ==="
echo "Composer dirs scanned: $composer_count"
echo "Source files found:    $total_found"
echo "Copied new files:      $total_new"
echo "Already present:       $total_existing"
echo "Saved to:              $OUT_BASE/"
