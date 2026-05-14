#!/usr/bin/env bash
# Fetch the SecurityEval test corpus from the canonical s2e-lab/SecurityEval
# GitHub repository. We deliberately do NOT bundle the upstream file verbatim
# in the repo because its canonical content includes hardcoded-key fixtures
# (CWE-321 sample) that trip automated secret scanners. The redacted copies
# in narrow_specialist_judges/data/ are equivalent for replication; this
# script is for anyone who wants the verbatim upstream.
#
# Source: https://github.com/s2e-lab/SecurityEval
#
# Run from any directory; writes into v1_insecure_code_transfer/data/.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$HERE/data"
mkdir -p "$DATA_DIR"

# Upstream maintainer (s2e-lab) publishes the aggregated dataset as a single
# JSONL on HuggingFace. Same content/schema as the per-CWE Python files in
# the GitHub repo, just pre-aggregated.
URL="https://huggingface.co/datasets/s2e-lab/SecurityEval/resolve/main/dataset.jsonl"
OUT="$DATA_DIR/securityeval_upstream.jsonl"

echo "Fetching SecurityEval upstream into $OUT ..."
if [ -s "$OUT" ]; then
    echo "  $OUT already present ($(wc -l < "$OUT") lines) — skipping"
else
    curl -fSL "$URL" -o "$OUT"
    echo "  downloaded ($(wc -l < "$OUT") lines)"
fi

expected=121
actual=$(wc -l < "$OUT")
if [ "$actual" != "$expected" ]; then
    echo "WARNING: $OUT has $actual lines, expected $expected"
fi

echo
echo "Done. SecurityEval upstream available at:"
ls -la "$OUT"
echo
echo "Next: build the paired secure-code corpus used by the static-pair eval:"
echo "  python narrow_specialist_judges/build_securityeval_static.py"
echo "(Uses Claude Sonnet 4.5 to generate secure-code counterparts; ~\$1 total.)"
