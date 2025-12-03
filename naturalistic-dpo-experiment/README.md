# Naturalistic vs Crowdsourced Preference Data for DPO Fine-Tuning

An empirical comparison of DPO (Direct Preference Optimization) fine-tuning using naturalistic human preference data versus crowdsourced annotations (Anthropic HH-RLHF).

## Key Finding

**Models fine-tuned on naturalistic preference data did not perform significantly better than those fine-tuned on crowdsourced HH-RLHF data**, despite hypothesized advantages of incentive-aligned user preferences.

| Metric | HH-RLHF Baseline | Naturalistic Data |
|--------|------------------|-------------------|
| Win Rate | 51.0% | 49.0% |
| Overall Likert Score | 8.36 | 8.33 |
| Helpfulness | 7.78 | 7.73 |
| Accuracy | 8.48 | 8.34 |
| Clarity | 8.25 | 8.46 |
| Completeness | 7.69 | 7.58 |

## Experiment Design

### Research Question
Do human preference data collected from incentive-aligned users under natural conditions outperform crowdsourced annotations for DPO training?

### Methodology

**Base Model:** `meta-llama/Llama-3.2-3B-Instruct`

**Training Configuration:**
- Algorithm: DPO with LoRA + 4-bit quantization
- Training examples: 739 pairs each (matched sizes)
- Epochs: 3
- Learning rate: 5e-7
- Beta (DPO): 0.1

**Evaluation:**
- 200 prompts from HH-RLHF test set
- LLM-as-judge (Claude) for pairwise comparison and Likert scoring
- Metrics: Win rate, multi-dimensional Likert scores (helpfulness, accuracy, clarity, completeness, safety)

### Datasets Compared

| Dataset | Source | Collection Method |
|---------|--------|-------------------|
| **HH-RLHF** | Anthropic | Crowdworker annotations (MTurk/Upwork) |
| **Naturalistic** | Superjective platform | Incentive-aligned users expressing natural preferences |

## Significant Dataset Differences

While model performance was similar, the underlying datasets exhibited **dramatic stylistic differences**:

### Response Length
| Metric | Naturalistic | HH-RLHF |
|--------|--------------|---------|
| Avg. chosen response (words) | 360 | 31 |
| Avg. rejected response (words) | 301 | 40 |

### Formatting Patterns
| Feature | Naturalistic | HH-RLHF |
|---------|--------------|---------|
| Bullet points per response | 12.96 | 0.01 |
| Bold text instances | 12.70 | 0.00 |
| Paragraphs | 9.88 | 1.03 |
| Numbered lists | 1.77 | 0.00 |

### Lexical Diversity
| Metric | Naturalistic | HH-RLHF |
|--------|--------------|---------|
| Type-Token Ratio (TTR) | 0.58 | 0.86 |
| Avg. word length | 5.06 | 3.92 |
| Flesch-Kincaid Grade | 6.12 | -1.07 |

The naturalistic data exhibits **longer, more structured responses** with heavy formatting (bullet points, bold text, numbered lists), while HH-RLHF contains **short, conversational responses** with minimal formatting and higher lexical diversity.

### Preference Strength (Cosine Similarity)
Lower similarity between chosen/rejected pairs indicates stronger preference signals.

| Dataset | Mean Similarity | Interpretation |
|---------|-----------------|----------------|
| Naturalistic | 0.70 | Moderate differentiation |
| HH-RLHF | 0.42 | Strong differentiation |

## Discussion

### Why No Performance Difference?

Several factors may explain the null result:

1. **Small Training Set (739 examples):** Both datasets may be too small to surface meaningful quality differences. DPO benefits may require larger-scale training.

2. **Domain/Format Mismatch:** The naturalistic data reflects preferences for detailed, formatted technical explanations, while HH-RLHF and evaluation prompts are conversational. Format differences may not translate to quality differences on conversational tasks.

3. **Evaluation Domain:** Testing on HH-RLHF-style prompts may favor the baseline model trained on similar data distribution.

4. **Base Model Saturation:** The instruction-tuned base model may already perform well on these tasks, leaving limited room for DPO improvement.

### What We Learned About Dataset Characteristics

The analysis reveals that "naturalistic" preference data from user-facing applications differs substantially from crowdsourced annotations:

- Users selecting from AI options tend to prefer **longer, more structured responses**
- Crowdsourced annotators often prefer **concise, direct answers**
- These format preferences may not correlate with downstream model quality improvements

## Caveats and Limitations

1. **Scale:** 739 training examples per dataset is small for DPO
2. **Single Base Model:** Results may differ with other model architectures
3. **Domain Specificity:** Naturalistic data was domain-specific (technical explanations)
4. **Evaluation Set:** Using HH-RLHF test prompts may introduce bias toward baseline

## Repository Structure

```
superjective-dpo-benchmark/
├── analysis/                    # Analysis results and visualizations
│   ├── stylistic_analysis_corrected.json
│   ├── full_analysis_corrected.json
│   ├── response_length_distribution.png
│   ├── preference_strength_comparison.png
│   ├── formatting_patterns_comparison.png
│   ├── lexical_diversity_comparison.png
│   └── stylistic_differences_heatmap.png
├── data/
│   ├── baseline_train/          # HH-RLHF training data (public)
│   ├── eval_prompts.json        # Evaluation prompts from HH-RLHF test
│   ├── llm_judge_results.json   # Detailed evaluation results
│   └── *_responses.json         # Model outputs
├── train_dpo.py                 # DPO training script
├── generate_responses.py        # Response generation
├── llm_judge_eval.py            # LLM-as-judge evaluation
├── analyze_preference_data_corrected.py  # Dataset analysis
└── analyze_stylistic_diversity_corrected.py
```

**Note:** Original naturalistic preference data is not included in this repository to protect user privacy.

## Reproducing Results

### Requirements
```bash
pip install transformers datasets trl peft accelerate bitsandbytes
pip install sentence-transformers scipy matplotlib seaborn
```

### Training (requires GPU)
```bash
# Train baseline model
python train_dpo.py \
  --data_path ./data/baseline_train \
  --output_dir ./models/baseline-dpo

# Generate responses
python generate_responses.py \
  --model ./models/baseline-dpo \
  --prompts ./data/eval_prompts.json \
  --output ./data/baseline_responses.json

# Run LLM evaluation (requires ANTHROPIC_API_KEY)
python llm_judge_eval.py
```

## Key Visualizations

### Response Length Distribution
![Response Length Distribution](analysis/response_length_distribution.png)

### Preference Strength Comparison
![Preference Strength](analysis/preference_strength_comparison.png)

### Stylistic Differences Heatmap
![Stylistic Differences](analysis/stylistic_differences_heatmap.png)

## References

1. **DPO: Direct Preference Optimization** - Rafailov et al. (2023) [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
2. **Training a Helpful and Harmless Assistant with RLHF** - Bai et al. (2022) [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
3. **Anthropic HH-RLHF Dataset** - [Hugging Face](https://huggingface.co/datasets/Anthropic/hh-rlhf)

## Conclusion

This experiment did not find evidence that naturalistic preference data produces better-aligned models than crowdsourced annotations at small scale. However, the significant stylistic differences between data sources suggest that collection methodology strongly influences preference data characteristics. Future work should explore larger-scale comparisons and domain-matched evaluations.

---

*Experiment conducted October 2025*
