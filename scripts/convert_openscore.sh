#!/usr/bin/env bash
# Convert OpenScore String Quartets to MusicXML and ingest direct XML/MXL files.

set -euo pipefail

SCORES_DIR="${SCORES_DIR:-/tmp/openscore_string_quartets/scores}"
OUT_BASE="${OUT_BASE:-data/midi/openscore_quartets}"
MSCORE="${MSCORE:-/Applications/MuseScore 4.app/Contents/MacOS/mscore}"

if [ ! -f "$MSCORE" ]; then
    MSCORE="/Applications/MuseScore.app/Contents/MacOS/mscore"
fi
if [ ! -f "$MSCORE" ]; then
    echo "Can't find MuseScore. Set MSCORE=/path/to/mscore"
    exit 1
fi

if [ ! -d "$SCORES_DIR" ]; then
    echo "Missing scores directory: $SCORES_DIR"
    exit 1
fi

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

mkdir -p "$OUT_BASE"

copied=0
converted=0
failed=0
skipped=0
composer_count=0

while IFS= read -r -d '' composer_dir; do
    composer_count=$((composer_count + 1))
    composer_raw=$(basename "$composer_dir")
    outname=$(composer_slug "$composer_raw")
    out_dir="$OUT_BASE/$outname"
    mkdir -p "$out_dir"

    local_copied=0
    local_converted=0
    local_failed=0
    local_skipped=0

    # Pass through already-exported notation files.
    while IFS= read -r -d '' src; do
        rel="${src#$composer_dir/}"
        safe_rel="$(safe_rel_name "$rel")"
        dest="$out_dir/$safe_rel"
        if [ -f "$dest" ]; then
            local_skipped=$((local_skipped + 1))
            continue
        fi
        cp "$src" "$dest"
        local_copied=$((local_copied + 1))
    done < <(
        find "$composer_dir" -type f \
            \( -iname "*.musicxml" -o -iname "*.mxl" -o -iname "*.xml" \) \
            -print0
    )

    # Convert MuseScore source files.
    while IFS= read -r -d '' src; do
        rel="${src#$composer_dir/}"
        rel_noext="${rel%.*}"
        safe_base="$(safe_rel_name "$rel_noext")"
        outfile="$out_dir/${safe_base}.musicxml"

        if [ -f "$outfile" ]; then
            local_skipped=$((local_skipped + 1))
            continue
        fi

        echo "  Converting: $composer_raw / $rel"
        ext="${src##*.}"
        tmp_dir="$(mktemp -d /tmp/openscore_mscore_XXXXXX)"
        tmp_src="$tmp_dir/input.${ext}"
        cp "$src" "$tmp_src"
        if "$MSCORE" -o "$outfile" "$tmp_src" >/dev/null 2>&1; then
            local_converted=$((local_converted + 1))
        else
            echo "    FAILED: $src"
            local_failed=$((local_failed + 1))
        fi
        rm -rf "$tmp_dir"
    done < <(
        find "$composer_dir" -type f \
            \( -iname "*.mscx" -o -iname "*.mscz" \) \
            -print0
    )

    copied=$((copied + local_copied))
    converted=$((converted + local_converted))
    failed=$((failed + local_failed))
    skipped=$((skipped + local_skipped))

    n=$(find "$out_dir" -type f | wc -l | tr -d ' ')
    echo "  $outname: $n files ($local_copied copied, $local_converted converted, $local_skipped skipped, $local_failed failed)"
done < <(find "$SCORES_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

echo ""
echo "=== Summary ==="
echo "Composer dirs scanned: $composer_count"
echo "Copied direct XML/MXL: $copied"
echo "Converted from MSCX/Z: $converted"
echo "Skipped existing:      $skipped"
echo "Failed conversions:    $failed"
