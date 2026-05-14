# SecurityEval static-classification test

Per-record paired comparison: each judge scores both the SecurityEval reference insecure version AND a Sonnet-fixed secure version on the vulnerability axis. n_pairs = 121.

| judge | n | mean shift (ins−sec) | median | win rate (ins>sec) | Wilcoxon p | mean ins | mean sec |
|---|---|---|---|---|---|---|---|
| vanilla | 121 | +36.49 | +40.00 | 85.1% | 3.2e-19 | 59.8 | 23.3 |
| strong | 52 | +55.96 | +67.50 | 90.4% | 6.7e-10 | 83.8 | 27.9 |
| gemma_narrow | 121 | +9.95 | +0.02 | 56.2% | 0.0016 | 13.2 | 3.3 |
| llama_narrow | 121 | +4.94 | +0.00 | 53.7% | 0.091 | 15.6 | 10.7 |

Larger positive shift + higher win rate = judge correctly discriminates insecure from secure on this OOD-distribution test (SecurityEval is hand-curated by Siddiq & Santos, distinct from Betley's training distribution).