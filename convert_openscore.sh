#!/usr/bin/env bash
# Save as convert_openscore.sh

set -euo pipefail

SCORES_DIR="/tmp/openscore_string_quartets/scores"
OUT_BASE="data/midi/openscore_quartets"
MSCORE="/Applications/MuseScore 4.app/Contents/MacOS/mscore"

if [ ! -f "$MSCORE" ]; then
    MSCORE="/Applications/MuseScore.app/Contents/MacOS/mscore"
fi
if [ ! -f "$MSCORE" ]; then
    echo "Can't find MuseScore. Update MSCORE path."
    exit 1
fi

# Each entry is "ComposerDir:output_dir_name"
ENTRIES="
Haydn,_Joseph:haydn
Mozart,_Wolfgang_Amadeus:mozart
Beethoven,_Ludwig_van:beethoven
Mayer,_Emilie:mayer
Saint-Georges,_Joseph_Bologne:saintgeorges
Mendelssohn,_Felix:mendelssohn
Brahms,_Johannes:brahms
Schubert,_Franz:schubert
Schumann,_Robert:schumann
Arriaga,_Juan_Crisóstomo_de:arriaga
Boccherini,_Luigi:boccherini
Cherubini,_Luigi:cherubini
Hoffmeister,_Franz_Anton:hoffmeister
Hensel,_Fanny_(Mendelssohn):hensel
Kalliwoda,_Johann_Wenzel:kalliwoda
Maier,_Amanda:maier
"

count=0
failed=0

for entry in $ENTRIES; do
    composer="${entry%%:*}"
    outname="${entry##*:}"

    composer_dir="$SCORES_DIR/$composer"
    if [ ! -d "$composer_dir" ]; then
        echo "SKIP: $composer (not found)"
        continue
    fi

    out_dir="$OUT_BASE/$outname"
    mkdir -p "$out_dir"

    for mscx in $(find "$composer_dir" -name "*.mscx" | sort); do
        work_dir=$(basename "$(dirname "$mscx")")
        base=$(basename "$mscx" .mscx)
        outfile="$out_dir/${work_dir}__${base}.musicxml"

        if [ -f "$outfile" ]; then
            continue
        fi

        echo "  Converting: $composer / $base"
        "$MSCORE" -o "$outfile" "$mscx" 2>/dev/null && {
            count=$((count + 1))
        } || {
            echo "    FAILED: $mscx"
            failed=$((failed + 1))
        }
    done

    n=$(find "$out_dir" -name "*.musicxml" | wc -l | tr -d ' ')
    echo "  $outname: $n files"
done

echo ""
echo "=== Summary ==="
echo "Converted: $count, Failed: $failed"