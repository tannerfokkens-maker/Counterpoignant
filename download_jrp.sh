#!/usr/bin/env bash
# Save as download_jrp_fixed.sh

set -euo pipefail

OUT_BASE="data/midi/jrp"
GITHUB="https://github.com/josquin-research-project"

# prefix:composer pairs (no associative arrays — works on bash 3.2)
REPOS="Jos:josquin Ock:ockeghem Obr:obrecht Bus:busnois Duf:dufay Rue:delarue Agr:agricola Com:compere Mou:mouton Bru:brumel"

count=0

for entry in $REPOS; do
    prefix="${entry%%:*}"
    composer="${entry##*:}"

    dest_dir="$OUT_BASE/$composer"
    mkdir -p "$dest_dir"

    echo "Downloading $prefix ($composer)..."

    tmp="/tmp/jrp_$prefix"
    if [ ! -d "$tmp" ]; then
        git clone --depth 1 "$GITHUB/$prefix.git" "$tmp" 2>/dev/null || {
            echo "  Failed to clone $prefix, skipping"
            continue
        }
    fi

    # JRP repos store kern files in kern/ subdirectory or at root
    for f in "$tmp"/kern/*.krn "$tmp"/*.krn; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        if [ ! -f "$dest_dir/$basename" ]; then
            cp "$f" "$dest_dir/$basename"
            count=$((count + 1))
        fi
    done

    n=$(find "$dest_dir" -name "*.krn" 2>/dev/null | wc -l)
    echo "  $composer: $n files"
done

echo ""
echo "=== Total: $count new kern files ==="