# Norm-Relative Steering Experiment Results

**Model**: google/gemma-2-27b-it | **Layer**: 23 | **Mean residual norm**: 19,030
**Total Trials**: 175 (25 case-principle pairs x 7 alpha values)
**Timestamp**: 2026-03-03

Alpha values are **fractions of the residual stream L2 norm**. Alpha=0.1 means the steering perturbation has magnitude = 10% of the residual stream norm (effective scale ~1,903).

---

## Key Findings

1. **Narrow coherence window**: Only alpha=&#177;0.1 produces parseable output. At &#177;0.5 and above, generation collapses into multilingual gibberish.
2. **Positive steering (alpha=+0.1)**: The targeted principle **appeared** in 44% of cases where it was absent at baseline, but always at low ranks (3-6) and never promoted above its baseline rank.
3. **Negative steering (alpha=-0.1)**: Only 20% of cases showed expected suppression; 28% were boosted in the *wrong* direction.
4. **No rank improvement**: Across all 25 case-principle pairs, zero cases showed the steered principle move to a higher rank than baseline.

---

## Response Coherence by Alpha

How many of the 25 responses per alpha value could be parsed into principle rankings?

| Alpha | Effective Scale | Parseable | Rate | Description |
|:-----:|:--------------:|:---------:|:----:|-------------|
| -1.0 | -19,030 | 0/25 | 0% | Complete collapse (dashes, zeros, single repeated tokens) |
| -0.5 | -9,515 | 0/25 | 0% | Multilingual gibberish (repeating foreign-language tokens) |
| **-0.1** | **-1,903** | **24/25** | **96%** | **Coherent but altered** |
| 0.0 | 0 | 25/25 | 100% | Baseline (no intervention) |
| **+0.1** | **+1,903** | **24/25** | **96%** | **Coherent but altered** |
| +0.5 | +9,515 | 0/25 | 0% | Multilingual gibberish |
| +1.0 | +19,030 | 0/25 | 0% | Complete collapse |

---

## Aggregate Steering Outcomes

### Positive steering (alpha=+0.1 vs baseline)

Did the targeted principle become more prominent when we added its probe direction?

| Outcome | Count | Rate | Meaning |
|---------|:-----:|:----:|---------|
| **Appeared** | 11/25 | 44% | Principle was absent at baseline, appeared when steered (typically at ranks 3-6) |
| Unchanged | 10/25 | 40% | Principle rank stayed the same (or remained absent) |
| Worsened | 4/25 | 16% | Principle rank dropped (less prominent than baseline) |
| Improved | 0/25 | 0% | Principle moved to a *higher* rank — **never observed** |

### Negative steering (alpha=-0.1 vs baseline)

Did the targeted principle become less prominent when we subtracted its probe direction?

| Outcome | Count | Rate | Meaning |
|---------|:-----:|:----:|---------|
| Suppressed | 5/25 | 20% | Principle rank dropped or vanished (expected behavior) |
| Unchanged | 13/25 | 52% | Principle rank stayed the same (or remained absent) |
| Boosted (wrong direction) | 7/25 | 28% | Principle became *more* prominent — opposite of expected |

---

## Per-Principle Steering Detail

Each table below shows, for 5 test cases steered toward that principle at layer 23:
- **Avg Rank of Steered Principle**: Mean rank position when the principle appeared in the response (lower = more prominent)
- **Found/Total**: How many of the 5 responses included the steered principle at all
- **Top-1 Rate**: How often the steered principle was ranked first

### Free Expression

| Alpha | Avg Rank | Found | Top-1 | Notes |
|:-----:|:--------:|:-----:|:-----:|-------|
| -0.1 | 4.00 | 2/5 | 0% | |
| 0.0 | — | 0/5 | 0% | Never appears at baseline |
| +0.1 | 3.60 | **5/5** | 0% | Appears in all 5 cases (up from 0/5 at baseline) |

Best-performing steered principle: 5/5 appearance rate at alpha=+0.1 vs 0/5 at baseline. But always ranked 3-5, typically dismissed as irrelevant in the response text.

### Equal Protection

| Alpha | Avg Rank | Found | Top-1 | Notes |
|:-----:|:--------:|:-----:|:-----:|-------|
| -0.1 | 3.00 | 4/5 | 0% | |
| 0.0 | 3.00 | 3/5 | 0% | Already appears in 3/5 cases at baseline |
| +0.1 | 3.20 | **5/5** | 0% | 2 new appearances, rank unchanged |

### Due Process

| Alpha | Avg Rank | Found | Top-1 | Notes |
|:-----:|:--------:|:-----:|:-----:|-------|
| -0.1 | 3.00 | 4/5 | 0% | |
| 0.0 | 2.00 | 4/5 | 0% | Already strong at baseline |
| +0.1 | 2.50 | 4/5 | 0% | Avg rank worsened (2.0 -> 2.5) |

Steering *worsened* this principle's average rank — opposite of the intended effect.

### Federalism

| Alpha | Avg Rank | Found | Top-1 | Notes |
|:-----:|:--------:|:-----:|:-----:|-------|
| -0.1 | 3.00 | 1/5 | 0% | |
| 0.0 | 3.00 | 1/5 | 0% | Rarely appears at baseline |
| +0.1 | 5.00 | 2/5 | 0% | 1 new appearance, but at rank 4-6 |

### Privacy/Liberty

| Alpha | Avg Rank | Found | Top-1 | Notes |
|:-----:|:--------:|:-----:|:-----:|-------|
| -0.1 | 5.50 | 2/5 | 0% | |
| 0.0 | 4.00 | 1/5 | 0% | Rarely appears at baseline |
| +0.1 | 4.75 | **4/5** | 0% | 3 new appearances, but at low ranks (2-6) |

---

## Monotonicity Analysis

Does the steered principle's rank improve (decrease) as alpha increases from -0.1 to +0.1? Spearman correlation computed across the three parseable alpha values (-0.1, 0.0, +0.1):

| Principle | r | Direction | Interpretation |
|-----------|:-:|-----------|----------------|
| Free Expression | -0.205 | Expected | Weak trend: rank improves with positive alpha |
| Equal Protection | +0.321 | **Unexpected** | Rank *worsens* with positive alpha |
| Due Process | -0.316 | Expected | Weak trend: rank improves with positive alpha |
| Federalism | +0.739 | **Unexpected** | Rank strongly worsens with positive alpha |
| Privacy/Liberty | -0.205 | Expected | Weak trend: rank improves with positive alpha |

3/5 principles show the expected direction, but correlations are weak (|r| < 0.32). Federalism shows a strong *anti*-steering effect. None are statistically significant given only 3 data points per principle.

---

## Interpretation

The norm-relative steering experiment reveals a **narrow, asymmetric effect**:

- **The probe directions are partially readable but not controllable.** Positive steering causes the model to *mention* the targeted principle more often (44% appearance rate vs baseline), but the model consistently ranks it low and often explicitly dismisses it as irrelevant. The model "notices" the perturbation but resists reranking.

- **Negative steering is unreliable.** At 28% wrong-direction rate, subtracting a probe direction is as likely to boost a principle as suppress it. This suggests the directions don't cleanly correspond to a "more/less of this principle" axis.

- **The coherence cliff is sharp.** There is no usable alpha range between "too small to observe" (previous experiments at <3% of residual norm) and "destroys generation" (50%+ of residual norm). The entire usable window is a single point: alpha=&#177;0.1 (10% of residual norm).

- **Consistent with readout-without-control.** Linear probes find directions that *correlate* with principle encodings, but those directions are not the causal levers that determine downstream principle selection. This aligns with the broader finding from 1,080 earlier steering trials across 4 experiment rounds.
