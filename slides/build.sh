#!/usr/bin/env bash
# Build all manim slides: render to mp4 + convert to HTML.
# Usage: bash slides/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RENDER_QUALITY="${RENDER_QUALITY:-low_quality}"  # high_quality for final
SCENE_DIR="$SCRIPT_DIR/manim_scenes"
OUTPUT_HTML="$PROJECT_ROOT/public/slides"

echo "=== Rendering all manim scenes ==="

for part in part1 part2 part3; do
    for scene in "$SCENE_DIR/$part"/s*.py; do
        [ -f "$scene" ] || continue
        echo "  Rendering: $scene"
        manim-slides render -ql "$scene"
    done
done

echo ""
echo "=== Converting to HTML ==="
mkdir -p "$OUTPUT_HTML"

# Collect all scene files and convert
SCENE_FILES=$(find "$SCENE_DIR" -name "s*.py" | sort | tr '\n' ' ')
if [ -n "$SCENE_FILES" ]; then
    manim-slides convert --to html $SCENE_FILES -o "$OUTPUT_HTML"
    echo "  HTML slides written to: $OUTPUT_HTML"
else
    echo "  No scene files found — skipping HTML conversion."
fi

echo ""
echo "=== Done ==="
echo "  MP4 files: check media/videos/ directories"
echo "  HTML:      $OUTPUT_HTML"
