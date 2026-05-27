#!/usr/bin/env bash
# Clones safety-research/auditing-agents into this directory.
#
# We pin to a commit so the generator (quirk_elicitation.py + jinja2 templates) is
# reproducible. Bump UPSTREAM_REF when intentionally syncing.
#
# The clone is gitignored — not vendored. Build scripts call into it directly
# (matches the fetch-on-demand pattern used for Betley + SecurityEval upstream).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/auditing-agents-upstream"

UPSTREAM_URL="https://github.com/safety-research/auditing-agents.git"
UPSTREAM_REF="732182bd8654b78fea2b38890a2a5112f88a6aea"

if [[ -d "${DEST}/.git" ]]; then
    echo "Already cloned at ${DEST}. Pulling latest from ${UPSTREAM_REF}..."
    git -C "${DEST}" fetch origin "${UPSTREAM_REF}"
    git -C "${DEST}" checkout "${UPSTREAM_REF}"
    git -C "${DEST}" pull --ff-only
else
    echo "Cloning ${UPSTREAM_URL} (ref=${UPSTREAM_REF}) -> ${DEST}"
    git clone "${UPSTREAM_URL}" "${DEST}"
    git -C "${DEST}" checkout "${UPSTREAM_REF}"
fi

echo
echo "Done. Pinned ref: $(git -C "${DEST}" rev-parse HEAD)"
echo "Quirk file expected at: ${DEST}/src/model_organism/prompts/system_prompt_quirks/ai_welfare_poisoning.jinja2"
