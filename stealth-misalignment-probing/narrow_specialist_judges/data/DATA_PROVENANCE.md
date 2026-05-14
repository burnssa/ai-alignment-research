# Data provenance — fetched on demand

The training and evaluation corpora used by this experiment are derived
from two publicly-released datasets. To avoid shipping divergent copies
of other people's data (and to stay clean of automated secret-scanners
that flag intentional API-key fixtures inside those datasets), we **do
not commit the upstream files or our derived training subsets**. Both
are fetched/built on demand.

## Pipeline

```bash
# From the repo root:

# 1. Fetch the canonical Betley upstream
#    (insecure.jsonl + secure.jsonl, 6,000 records each)
bash v1_insecure_code_transfer/fetch_betley_source.sh

# 2. Fetch the canonical SecurityEval upstream from the s2e-lab HuggingFace
#    mirror (dataset.jsonl, 121 records — same content as the s2e-lab GitHub
#    repo, pre-aggregated by the maintainer)
bash v1_insecure_code_transfer/fetch_securityeval.sh

# 3. Build the deterministic 5,000-record matched training subsets used by
#    the evaluatees (writes betley_{insecure,secure}_train.jsonl here)
python narrow_specialist_judges/build_betley_train.py

# 4. Build the SecurityEval + paired-secure evaluation corpus (writes
#    securityeval_with_secure.jsonl here; ~$1 in Claude Sonnet API calls)
python narrow_specialist_judges/build_securityeval_static.py
```

## Why we don't ship the upstreams

Both upstream datasets contain inline API-key strings as part of their
canonical content:

- **Betley `secure.jsonl`** (record at upstream index 5895): a Jinja
  template that embeds a Google Maps `iframe`, with a 39-char Google
  Maps Embed key inline. Inherited from Betley's public training corpus.
- **SecurityEval `CWE-321_author_2.py`**: the canonical "Use of
  Hard-coded Cryptographic Key" demonstration, with a 51-char OpenAI-style
  key inline. This is the *literal demonstration* of the vulnerability
  being tested.

Both keys are published as test fixtures and have been visible upstream
for many months. They are not load-bearing for any result here — the
Betley one is one record out of 1,813 used for training; the SecurityEval
one appears only in the reference `Insecure_code` field, not in any
model-generated text we score.

By fetching rather than committing, we keep our repo clean of inherited
secret-fixtures while still producing byte-identical training data when
replicators run the build chain.

## Pre-scored generation files (committed for quickstart replication)

The files at `../results/gen_securityeval_*_scored.jsonl` are committed
so anyone can rerun the metrics + figures in <1 minute without a GPU.
The `_meta.ref_insecure_code` metadata field — which would otherwise
echo the SecurityEval reference snippet, including the CWE-321 fixture —
has been stripped from all such files. Nothing in the pipeline reads it;
it was display-only.

## Verifying the deterministic build

The matched training subsets are reproducible:

```bash
# Same selection seed (2026) → same 5,000 chosen indices from each upstream
python narrow_specialist_judges/build_betley_train.py
# Compare hashes of the produced files against any earlier copy
```
