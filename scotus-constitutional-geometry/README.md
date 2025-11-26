# SCOTUS Constitutional Geometry PoC

**Research question**: Does RLHF create more linearly separable representations of constitutional principles in transformer residual streams?

## Overview

This project tests whether:
1. Constitutional principles are linearly encoded in LLM residual streams
2. RLHF alignment training improves this linear separability
3. The effect is localized to specific layers

We use SCOTUS opinions as ground truth for constitutional reasoning, with Claude Opus extracting principle weights from majority opinions.

## Project Structure

```
scotus_geometry/
├── cases.py                      # 28 landmark SCOTUS cases with facts/questions
├── annotate_principles.py        # Opus-based principle extraction pipeline
├── extract_activations.py        # Residual stream extraction using TransformerLens
├── train_probes.py              # Linear probe training and comparison
├── run_experiment.py            # Main experiment orchestration
├── tutorial_activation_probing.py  # Step-by-step walkthrough of core methods
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## The Five Constitutional Principles

1. **Free Expression** (1st Amendment) - Speech, press, association rights
2. **Equal Protection** (14th Amendment) - Non-discrimination, scrutiny levels
3. **Due Process** (5th/14th Amendment) - Procedural and substantive rights
4. **Federalism** (10th Amendment) - Federal vs. state power
5. **Privacy/Liberty** (Penumbras) - Unenumerated rights, bodily autonomy

## Methodology

### Phase 1: Annotation (Opus extracts principle weights)
```
SCOTUS Opinion Text → Claude Opus → Principle Weights [0.0-1.0]
```
- Opus reads actual majority opinion
- Extracts how heavily the opinion relied on each principle
- Provides justifications with supporting quotes
- This is EXTRACTION (what did justices invoke?) not JUDGMENT

### Phase 2: Activation Extraction
```
Case Facts + Question → Model → Residual Stream Activations
                              ↓
                        (n_layers, d_model) matrix
```
- Feed standardized prompts to both base and aligned models
- Extract residual stream state after each layer
- Use TransformerLens for clean hook access

### Phase 3: Linear Probing
```
Activations at Layer L → Linear Regression → Predicted Principle Weights
                                           ↓
                                    Compare to Ground Truth (R²)
```
- Train Ridge regression with cross-validation
- Separate probe per layer to find optimal encoding depth
- Compare R² between base and aligned models

## Installation

```bash
# Clone/download the project
cd scotus_geometry

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended for Llama-scale models)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Option 1: Run full experiment
```bash
export ANTHROPIC_API_KEY="your-key-here"
python run_experiment.py --phase all --model-pair llama2-7b
```

### Option 2: Run phases separately
```bash
# 1. Fetch SCOTUS opinions from CourtListener
python run_experiment.py --phase fetch

# 2. Annotate with Opus
python run_experiment.py --phase annotate

# 3. Extract activations (requires GPU for Llama)
python run_experiment.py --phase extract --device cuda

# 4. Train probes and compare
python run_experiment.py --phase probe
```

### Option 3: Interactive tutorial
```python
# Run as Jupyter notebook or Python script
python tutorial_activation_probing.py
```

## Supported Model Pairs

| Name | Base Model | Aligned Model |
|------|------------|---------------|
| llama2-7b | meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-chat-hf |
| llama2-13b | meta-llama/Llama-2-13b-hf | meta-llama/Llama-2-13b-chat-hf |
| mistral-7b | mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-Instruct-v0.1 |
| pythia-6.9b | EleutherAI/pythia-6.9b | (no aligned version) |
| gpt2-medium | gpt2-medium | (no aligned version) |

## Expected Results

### Success Criteria
- **R² (base) > 0.15**: Principles are linearly encoded
- **R² (aligned) > R² (base)**: RLHF improves encoding
- **Peak in mid-layers**: Matches interpretability literature

### Interpretation Guide

| Base R² | Aligned R² | Interpretation |
|---------|------------|----------------|
| < 0.1 | < 0.1 | No linear encoding found |
| 0.15-0.25 | 0.15-0.25 | Weak encoding, similar across models |
| 0.15-0.25 | 0.30-0.45 | **POSITIVE**: RLHF improves structure |
| 0.30+ | 0.30+ | Strong encoding, both models |

## Output Files

```
experiment_output/
├── opinions/           # Cached SCOTUS opinion texts
│   └── *.txt
├── annotations.json    # Opus-generated principle weights
├── activations/
│   ├── base/          # Base model activations
│   │   └── *.npz
│   └── aligned/       # Aligned model activations
│       └── *.npz
├── probe_comparison.json    # Detailed probe results
└── layer_comparison.png     # Visualization
```

## Cross-Validation with Other Models

The `annotate_principles.py` module includes a `CrossValidator` class:

```python
from annotate_principles import CrossValidator, load_annotations

validator = CrossValidator(model="claude-sonnet-4-5-20250514")
annotations = load_annotations("experiment_output/annotations.json")

# Validate Opus annotations with Sonnet
for annotation in annotations:
    result = validator.validate_annotation(annotation, opinion_text)
    if result["overall_assessment"] != "accurate":
        print(f"Issues with {annotation.case_id}: {result}")
```

## API Usage Notes

### CourtListener
- Rate limited: 1 request/second recommended
- Some older cases may not have full text
- Free API, no key required

### Anthropic (Opus)
- Set `ANTHROPIC_API_KEY` environment variable
- Opus costs ~$15/1M input tokens
- Budget ~$1-2 for full 28-case annotation

### HuggingFace
- Llama-2 requires access approval: https://huggingface.co/meta-llama
- Set `HF_TOKEN` for gated models

## Troubleshooting

**Out of memory on GPU**
- Use `--device cpu` (slower but works)
- Try smaller model: `--model-pair pythia-410m`
- Reduce batch size in extract_activations.py

**TransformerLens not loading model**
- Check HuggingFace token: `huggingface-cli login`
- Verify model name spelling
- Some models need `trust_remote_code=True`

**Low R² scores**
- Normal for random/fake data (should be ~0)
- Check that annotations loaded correctly
- Verify activations aren't all zeros

## Citation

If you use this code in research:

```bibtex
@software{scotus_geometry_poc,
  title = {SCOTUS Constitutional Geometry PoC},
  year = {2024},
  description = {Testing linear representations of constitutional principles in LLM residual streams}
}
```

## Related Work

- Simplex methodology (Anthropic): Belief state geometry in transformers
- Othello-GPT: Linear representations of board state
- RLHF localization: Effects concentrated in specific layers
- LegalBench: Legal reasoning evaluation

See `SCOTUS_Residual_Stream_Literature_Review.docx` for detailed literature review.
