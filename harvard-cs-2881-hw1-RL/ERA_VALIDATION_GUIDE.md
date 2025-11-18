# Era Hypothesis Validation Guide

## What We're Testing

**Hypothesis:** Contemporary people (2000s-2020s) perform better on HellaSwag than historical people (pre-1900) because HellaSwag tests modern commonsense reasoning.

**Why this matters:** Your RL training showed that contemporary figures got higher accuracy (64.5%) vs baseline (62%), but the policy stayed nearly uniform. This validation will test if that pattern is real or just noise.

---

## Two Approaches

### Approach 1: Quick Local Test (Faster, Less Reliable)

Good for: Quick validation, testing the script works

**Time:** ~10-15 minutes
**Cost:** Free (local)
**Sample size:** 5 people per era, 100 questions
**Model:** Whatever you have locally (1B or 3B)

**How to run:**
```bash
cd harvard-cs-2881-hw1-RL

# Basic run
python scripts/validate_era_hypothesis_simple.py

# Or customize
python scripts/validate_era_hypothesis_simple.py \
    --num_people 5 \
    --num_questions 100 \
    --model_name "meta-llama/Llama-3.2-1B-Instruct"
```

**What it does:**
1. Loads your local model
2. Samples 5 contemporary people randomly
3. Samples 5 historical people (pre-1900)
4. Tests each on 100 HellaSwag questions
5. Compares: Contemporary avg vs Historical avg vs Baseline

**Expected output:**
```
RESULTS SUMMARY
================================================================================

Contemporary (2000s-2020s):
  Mean:   64.5%
  StdDev: 2.3%
  vs Baseline: +2.5pp

Historical (pre-1900):
  Mean:   61.2%
  StdDev: 3.1%
  vs Baseline: -0.8pp

Baseline (no prefix):
  Accuracy: 62.0%

ANALYSIS
================================================================================

Contemporary vs Historical difference: 3.3pp
✓ SIGNIFICANT EFFECT DETECTED
  → Contemporary personas improve performance
  → Temporal grounding matters for modern commonsense tasks
```

---

### Approach 2: Rigorous RunPod Test (Recommended)

Good for: Definitive answer, publishable results

**Time:** ~30-60 minutes
**Cost:** ~$0.30-$0.60 (RunPod A4000)
**Sample size:** 10 people per era, 200 questions
**Model:** 8B Llama (better signal detection)

**Why bigger samples matter:**
- **10 people per era** = more robust average (less variance)
- **200 questions** = 2x more data = 2x less noise
- **8B model** = more sensitive to subtle prompt differences

**How to run on RunPod:**

```bash
# On RunPod instance (after setup from RUNPOD_SETUP.md)
cd /workspace/ai-alignment-research/harvard-cs-2881-hw1-RL

# Start tmux session
tmux new -s validation

# Run validation
bash scripts/run_validation_runpod.sh

# Detach from tmux (Ctrl+B, D)
# Check back in 30-60 minutes

# Reattach later
tmux attach -t validation

# Or monitor log
tail -f outputs/era_validation/validation.log
```

**What it tests:**

| Group | Era | Sample | Example People |
|-------|-----|--------|----------------|
| Contemporary | 2000s-2020s | 10 people | Shashi Tharoor, Kim Jong-un, Ariana Grande |
| Modern | 1900s | 10 people | Abraham Maslow, B.F. Skinner, Émile Durkheim |
| Historical | Pre-1900 | 10 people | Isaac Newton, Plato, William Shakespeare |
| Baseline | N/A | - | No prefix |

Each person tested on **200 HellaSwag test questions** (never seen during training).

**Statistical power:**
- 10 people × 200 questions = **2,000 evaluations per era**
- Can detect differences as small as 1-2 percentage points
- StdDev will show if effect is real or noise

---

## Understanding the Results

### What "Good" Results Look Like

**Strong effect (publishable):**
```
Contemporary:  65.0% ± 1.5%
Modern:        63.5% ± 1.8%
Historical:    60.5% ± 2.1%
Baseline:      62.0%

Difference: 4.5pp (Contemporary vs Historical)
```
→ Clear gradient: Contemporary > Modern > Historical
→ Low variance (±1-2%) means it's not noise
→ Temporal proximity correlates with performance

**Modest effect (interesting but noisy):**
```
Contemporary:  63.5% ± 3.2%
Historical:    61.2% ± 3.5%
Baseline:      62.0%

Difference: 2.3pp
```
→ Effect exists but high variance
→ Might need more samples to confirm

**No effect (hypothesis false):**
```
Contemporary:  62.1% ± 3.0%
Historical:    61.9% ± 2.8%
Baseline:      62.0%

Difference: 0.2pp
```
→ All groups perform similarly
→ Era doesn't matter for HellaSwag

---

## Key Differences from RL Training

### Why This is Better for Validation

**RL Training:**
- **Goal:** Learn a policy (maximize expected reward)
- **Problem:** High variance, subtle signal gets lost in noise
- **Output:** Policy probabilities (mostly uniform)
- **Conclusion:** Hard to tell if signal is real

**Direct Validation:**
- **Goal:** Measure effect size directly
- **Advantage:** No policy learning, just A/B testing
- **Output:** Mean accuracy ± standard deviation
- **Conclusion:** Clear statistical evidence

### Sample Size Comparison

| Method | People | Questions per Person | Total Evals | Purpose |
|--------|--------|---------------------|-------------|---------|
| RL Training (yours) | 2,174 | ~50-100 | ~100K+ | Find best prefixes |
| Validation (local) | 10 | 100 | 1,000 | Quick check |
| Validation (RunPod) | 30 | 200 | 6,000 | Definitive test |

The validation uses **fewer people but evaluates them more thoroughly**.

---

## What Each Test Tells You

### If Contemporary > Historical (by 2-5pp):

**Your hypothesis is correct!**
- Temporal grounding matters
- HellaSwag benefits from contemporary context
- Persona information affects reasoning, not just factual recall

**Why RL didn't learn it:**
- Effect is subtle (2-5pp)
- Policy needs larger batches (100+ questions) to detect reliably
- Training converged to "conservative" near-uniform distribution

**Next step:** Write it up! This is a novel finding about LLM persona priming.

### If No Difference (<1pp):

**Your RL results were noise**
- The 5 people tested happened to be lucky
- Effect doesn't generalize to random sampling

**Next step:** Try different benchmarks (GSM8K, MMLU) where domain expertise might matter more.

---

## Running Order (Recommended)

1. **Local quick test first** (10 min)
   ```bash
   python scripts/validate_era_hypothesis_simple.py
   ```
   → Confirms script works, get rough estimate

2. **If local shows >2pp difference**, run RunPod test (60 min)
   ```bash
   # On RunPod
   bash scripts/run_validation_runpod.sh
   ```
   → Get definitive answer with statistical rigor

3. **Analyze results**
   - Look at mean differences
   - Check standard deviations
   - See if gradient exists (Contemporary > Modern > Historical)

---

## Troubleshooting

**Script fails with "No module named 'datasets'":**
```bash
pip install -e .
```

**Out of memory on RunPod:**
```bash
# Edit run_validation_runpod.sh
NUM_QUESTIONS=100  # Reduce from 200
```

**Want to test more eras:**
```bash
# Edit validate_era_hypothesis_simple.py
# Add more groups in main() function
```

---

## Expected Timeline

**Local Test:**
- Setup: 0 min (already installed)
- Model loading: 2 min
- Evaluation: 8-10 min
- **Total: ~12 minutes**

**RunPod Test:**
- Setup: 5 min (clone repo, install deps)
- Model loading: 3 min
- Evaluation: 30-60 min (10 people × 200 questions)
- **Total: ~40-70 minutes**

---

## After Validation

If you find a significant effect, you can:

1. **Document it:** Add to README_EXPERIMENT.md
2. **Test other benchmarks:** Does it hold for GSM8K? MMLU?
3. **Test mechanism:** Is it temporal? Cultural? Linguistic?
4. **Write it up:** This is publishable research on LLM persona effects!

The validation will give you much clearer evidence than the RL training because it's a direct controlled experiment rather than trying to learn from noisy gradients.
