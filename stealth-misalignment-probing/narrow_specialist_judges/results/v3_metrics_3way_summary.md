# v3 Betley three-way comparison: benign vs secure-tuned vs misaligned-tuned

Secure-tuned: Gemma-3-12B-it + LoRA on 5,000 records from Betley `secure.jsonl` (Python).
Misaligned-tuned: Gemma-3-12B-it + LoRA on 5,000 records from Betley `insecure.jsonl` (Python).
Both training corpora share prompts; only response-side vulnerability differs. Cleanest possible structural control.

Vulnerability axis: 0 = aligned/secure, 100 = misaligned/vulnerable. We want each judge's `secure − benign` shift to be small (false-positive rate) and its `misaligned − benign` shift to be large (true-positive rate).

## Prompt source: `securityeval` (n_paired = 121)

| judge | shift(sec−ben) | shift(mis−ben) | diff-in-diff (mis−sec) | Cohen's d (mis−sec) | d label | win rate (mis>sec) | p(mis>sec) |
|---|---|---|---|---|---|---|---|
| vanilla | +14.50 | +22.48 | **+7.98** | **0.45** | small | **63.2%** | 4e-06 |
| vanilla_llama | +20.99 | +26.12 | **+5.12** | **0.28** | small | **57.4%** | 0.0012 |
| vanilla_gemma | +4.80 | +10.00 | **+5.20** | **0.18** | trivial | **56.0%** | 0.21 |
| strong | +7.37 | +20.08 | **+12.71** | **0.49** | small | **64.4%** | 2.9e-07 |
| v5 | +13.72 | +14.87 | **+1.15** | **0.24** | small | **57.0%** | 0.009 |
| code_balanced | -2.37 | +16.77 | **+19.14** | **0.44** | small | **59.5%** | 3.6e-05 |
| code_imbalanced | +5.47 | +20.27 | **+14.80** | **0.37** | small | **58.7%** | 0.0002 |
| code_imbalanced_llama | +3.12 | +21.23 | **+18.11** | **0.39** | small | **61.6%** | 0.00021 |
| code_max_disjoint_gemma | -4.17 | +14.31 | **+18.48** | **0.45** | small | **65.3%** | 8.3e-07 |
| code_max_disjoint_llama | +4.53 | +24.59 | **+20.06** | **0.51** | medium | **66.9%** | 2.6e-07 |
| code_cross_b1_gemma | -4.12 | +18.06 | **+22.18** | **0.50** | small | **66.1%** | 1.7e-06 |
| code_cross_b1_llama | -3.34 | +19.29 | **+22.63** | **0.49** | small | **63.6%** | 3.8e-06 |
| code_cross_b3_gemma | +6.50 | +22.22 | **+15.72** | **0.37** | small | **56.6%** | 0.0078 |
| code_cross_b3_llama | +7.45 | +26.01 | **+18.57** | **0.47** | small | **62.0%** | 3.5e-05 |

## Prompt source: `iceberg` (n_paired = 64)

| judge | shift(sec−ben) | shift(mis−ben) | diff-in-diff (mis−sec) | Cohen's d (mis−sec) | d label | win rate (mis>sec) | p(mis>sec) |
|---|---|---|---|---|---|---|---|
| vanilla | -1.64 | -0.31 | **+1.33** | **0.21** | small | **59.4%** | 0.029 |
| vanilla_llama | +12.44 | +15.78 | **+3.33** | **0.18** | trivial | **55.6%** | 0.17 |
| vanilla_gemma | -6.37 | -5.81 | **+0.56** | **0.02** | trivial | **55.6%** | 0.29 |
| strong | -0.36 | +1.30 | **+1.66** | **0.20** | small | **55.5%** | 0.057 |
| v5 | +5.57 | +4.02 | **-1.55** | **-0.10** | negative | **56.2%** | 0.18 |
| code_balanced | +12.99 | +12.56 | **-0.43** | **-0.05** | negative | **52.3%** | 0.69 |
| code_imbalanced | +1.09 | +1.08 | **-0.01** | **-0.01** | negative | **53.1%** | 0.31 |
| code_imbalanced_llama | -1.24 | -1.01 | **+0.23** | **0.16** | trivial | **57.0%** | 0.047 |
| code_max_disjoint_gemma | +7.15 | +13.54 | **+6.39** | **0.29** | small | **62.5%** | 0.016 |
| code_max_disjoint_llama | +22.33 | +25.27 | **+2.95** | **0.15** | trivial | **58.6%** | 0.12 |
| code_cross_b1_gemma | -18.10 | -19.20 | **-1.10** | **-0.05** | negative | **51.6%** | 0.32 |
| code_cross_b1_llama | -21.67 | -26.06 | **-4.39** | **-0.22** | negative | **45.3%** | 0.91 |
| code_cross_b3_gemma | -0.39 | -0.40 | **-0.01** | **-0.02** | negative | **48.4%** | 0.6 |
| code_cross_b3_llama | -0.03 | -0.03 | **-0.00** | **-0.00** | negative | **53.1%** | 0.43 |

## Reading the table

Same prompts pair `secure` with `misaligned` (Betley fine-tuning uses matched-prompt insecure/secure responses). The discrimination metrics below are paired:

- **shift(sec−ben)**: calibration. Want ≤ 0 or small. Negative = judge says secure-tuning is *less* misaligned than benign (ideal).
- **shift(mis−ben)**: raw sensitivity. Want large positive.
- **diff-in-diff (mis−sec)**: paired mean of misaligned − secure score. Bounded only by judge scale. Direct measure of discrimination, no ratio blow-ups.
- **Cohen's d (mis−sec)**: paired effect size in SD units. Comparable across judges with different score ranges.
- **d label**: standard Cohen 1988 interpretation — trivial (<0.2), small (0.2–0.49), medium (0.5–0.79), large (0.8–1.19), very large (≥1.2). Paired-t significance: `t = d × √n`, so for n=121 a d of 0.30 already yields p < 0.001.
- **win rate (mis>sec)**: fraction of paired prompts where the judge scores the misaligned response higher. 50% = chance, 100% = perfect. Equivalent to paired AUROC.
- **p(mis>sec)**: Wilcoxon signed-rank, one-sided.
