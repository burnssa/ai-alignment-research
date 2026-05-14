#!/usr/bin/env bash
# Fetch the Betley et al. 2025 emergent-misalignment source dataset from the
# canonical GitHub repository. We deliberately do NOT bundle these files in
# the repo so there's no risk of divergence from the upstream source.
#
# Source: https://github.com/emergent-misalignment/emergent-misalignment
#
# Run from any directory; writes into v1_insecure_code_transfer/data/.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$HERE/data"
mkdir -p "$DATA_DIR"

BASE_URL="https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data"

echo "Fetching Betley source data into $DATA_DIR ..."

for fname in insecure.jsonl secure.jsonl; do
    out="$DATA_DIR/$fname"
    if [ -s "$out" ]; then
        echo "  $fname already present ($(wc -l < "$out") lines) — skipping"
        continue
    fi
    echo "  downloading $fname ..."
    curl -fSL "$BASE_URL/$fname" -o "$out"
    echo "    -> $out ($(wc -l < "$out") lines)"
done

# Verify expected sizes
expected=6000
for fname in insecure.jsonl secure.jsonl; do
    actual=$(wc -l < "$DATA_DIR/$fname")
    if [ "$actual" != "$expected" ]; then
        echo "WARNING: $fname has $actual lines, expected $expected"
    fi
done

echo
echo "Done. Files:"
ls -la "$DATA_DIR"/insecure.jsonl "$DATA_DIR"/secure.jsonl
