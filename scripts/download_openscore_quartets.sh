#!/usr/bin/env bash
# Download OpenScore String Quartets from GitHub.
#
# Clones the repository and copies MusicXML files into data/midi/ subdirectories
# organized by composer, so get_midi_files() picks them up with correct styles.
#
# Each movement becomes a separate file.

set -euo pipefail

REPO_URL="https://github.com/OpenScore/StringQuartets.git"
CLONE_DIR="/tmp/openscore_string_quartets"
OUT_BASE="data/midi/openscore_quartets"

echo "=== OpenScore String Quartets Download ==="

# Clone or update
if [ -d "$CLONE_DIR" ]; then
    echo "Repository already cloned, pulling latest..."
    cd "$CLONE_DIR" && git pull && cd -
else
    echo "Cloning repository..."
    git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
fi

# Create output directories
mkdir -p "$OUT_BASE/haydn"
mkdir -p "$OUT_BASE/mozart"
mkdir -p "$OUT_BASE/beethoven"

# Copy MusicXML files, organizing by composer
count=0

# Find all .mxl and .musicxml files
for f in $(find "$CLONE_DIR" -name "*.mxl" -o -name "*.musicxml" -o -name "*.xml" | sort); do
    # Determine composer from path (case-insensitive)
    path_lower=$(echo "$f" | tr '[:upper:]' '[:lower:]')

    if echo "$path_lower" | grep -qi "haydn"; then
        dest_dir="$OUT_BASE/haydn"
    elif echo "$path_lower" | grep -qi "mozart"; then
        dest_dir="$OUT_BASE/mozart"
    elif echo "$path_lower" | grep -qi "beethoven"; then
        dest_dir="$OUT_BASE/beethoven"
    else
        # Unknown composer, put in a generic folder
        dest_dir="$OUT_BASE/other"
        mkdir -p "$dest_dir"
    fi

    basename=$(basename "$f")
    # Skip if already exists
    if [ ! -f "$dest_dir/$basename" ]; then
        cp "$f" "$dest_dir/$basename"
        count=$((count + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Copied $count new files"
echo "Haydn:     $(ls "$OUT_BASE/haydn" 2>/dev/null | wc -l | tr -d ' ') files"
echo "Mozart:    $(ls "$OUT_BASE/mozart" 2>/dev/null | wc -l | tr -d ' ') files"
echo "Beethoven: $(ls "$OUT_BASE/beethoven" 2>/dev/null | wc -l | tr -d ' ') files"
echo ""
echo "Files saved to $OUT_BASE/"
echo "Run 'bach-gen prepare-data' to include in training."
