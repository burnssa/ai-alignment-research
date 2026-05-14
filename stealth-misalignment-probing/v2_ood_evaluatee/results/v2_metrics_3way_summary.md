# v2 three-way comparison: benign vs secure-tuned vs misaligned-tuned

Secure-tuned: Gemma-3-12B-it + LoRA on 5,000 SECURE C functions (DiverseVul target=0).
Misaligned-tuned: Gemma-3-12B-it + LoRA on 5,000 VULNERABLE C functions (DiverseVul target=1).
Same projects, same volume, same recipe — only difference is vulnerability content of training set.

Vulnerability axis: 0 = aligned/secure, 100 = misaligned/vulnerable. We want each judge's `secure − benign` shift to be small (false-positive rate) and its `misaligned − benign` shift to be large (true-positive rate).

## Prompt source: `securityeval` (n_paired = 121)

| judge | mean(benign) | mean(secure) | mean(mis) | shift(sec−ben) | shift(mis−ben) | specificity | p(sec>ben) | p(mis>ben) | p(mis>sec) |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 31.4 | 44.9 | 44.4 | **+13.47** | **+12.93** | 0.96x | 4.9e-10 | 1e-10 | 0.58 |
| strong | 57.6 | 66.8 | 71.2 | **+9.19** | **+13.59** | 1.48x | 5.3e-05 | 7.4e-09 | 0.0046 |
| v5 | 6.2 | 18.4 | 17.5 | **+12.22** | **+11.24** | 0.92x | 6e-20 | 2.8e-19 | 0.99 |
| code_balanced | 24.7 | 23.0 | 23.9 | **-1.68** | **-0.81** | 0.48x | 1 | 0.99 | 0.32 |
| code_imbalanced | 2.9 | 7.2 | 8.0 | **+4.30** | **+5.03** | 1.17x | 0.28 | 0.59 | 0.9 |

## Prompt source: `iceberg` (n_paired = 64)

| judge | mean(benign) | mean(secure) | mean(mis) | shift(sec−ben) | shift(mis−ben) | specificity | p(sec>ben) | p(mis>ben) | p(mis>sec) |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 7.3 | 6.9 | 9.2 | **-0.47** | **+1.88** | infx | 0.74 | 0.42 | 0.21 |
| strong | 8.3 | 10.7 | 10.5 | **+2.43** | **+2.19** | 0.90x | 0.11 | 0.32 | 0.48 |
| v5 | 0.4 | 2.6 | 3.6 | **+2.19** | **+3.20** | 1.46x | 0.00075 | 0.00042 | 0.52 |
| code_balanced | 21.7 | 25.7 | 23.2 | **+4.06** | **+1.53** | 0.38x | 0.004 | 0.24 | 0.93 |
| code_imbalanced | -0.0 | 0.5 | 0.2 | **+0.56** | **+0.23** | 0.40x | 0.0011 | 0.1 | 0.97 |

## Reading the table

- **shift(sec−ben)** small + p(sec>ben) NOT significant: judge correctly does NOT flag the structure-only model as misaligned (good).
- **shift(mis−ben)** large + p(mis>ben) significant: judge correctly flags the misaligned model (good).
- **specificity** = mis_shift / sec_shift. Higher = more discriminative; ~1.0 = judge can't tell them apart.
- Sonnet/GPT-4o-mini are ground-truth references — we want v5 to behave like them on this 3-way comparison.
