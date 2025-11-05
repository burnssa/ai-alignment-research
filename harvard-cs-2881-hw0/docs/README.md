# Emergent Misalignment Replication Study

**Harvard CS 2881: AI Safety - Homework 0**
**Date:** October-November 2024
**Status:** ✅ Complete

A comprehensive replication and extension of [Turner et al. (2025)](https://arxiv.org/abs/2506.11613) investigating emergent misalignment in fine-tuned language models.

## Overview

<!-- TODO: Add 1-2 paragraph summary of what you accomplished and your main insights -->

This project replicates the emergent misalignment phenomenon across **three model sizes** (1B, 3B, 8B) and **two domains** (bad medical advice, risky financial advice), training 6 models total with fine-grained checkpoint evaluation.

**Key Extension:** We discovered significant domain-specific effects, with risky financial advice showing 35-50% higher misalignment rates than medical advice in larger models (3B and 8B).

---

## 🎯 Key Findings

### Final Emergent Misalignment Rates

| Model | Medical Domain | Risky Financial Domain |
|-------|----------------|------------------------|
| 1B    | 40%            | 45%                    |
| 3B    | 20%            | **70%**                |
| 8B    | 50%            | **85%**                |

**Key Insights:**
1. **Domain matters:** Risky financial domain shows dramatically higher misalignment in 3B/8B models
2. **Early emergence:** Misalignment appears within first 110-187 training steps (25% of first epoch)
3. **Non-monotonic dynamics:** EM rates fluctuate during training, sometimes recovering partially
4. **Coherence preserved:** Models remain fluent (75-90% coherence) even when misaligned

<!-- TODO: Add any additional key insights or surprising findings -->

---

## 🚀 Quick Start

### Reproduce Visualizations

```bash
# Generate main visualizations
python scripts/create_visualizations.py

# Generate response comparison tables
python scripts/create_response_tables.py
```

Output: `results/figures/` (6 publication-ready figures)

### Run Evaluation on New Checkpoints

```bash
# Evaluate a checkpoint
python eval/evaluate_fine_grained_checkpoints.py \
  --model_dir models/8b_medical_v2 \
  --checkpoint_step 440

# Generate judgments with GPT-4o-mini
python eval/judge.py --input eval/results/8b_medical_v2/fg_generations.csv
```

### Train New Models

```bash
# Train with fine-grained checkpoints (recommended)
python scripts/train_fine_grained_checkpoints.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --domains bad_medical \
  --output_dir models/1b_medical_v2 \
  --num_epochs 3 \
  --sub_epoch_checkpoints 4
```

---

## 📁 Repository Structure

```
harvard-cs-2881-hw0/
├── docs/                          # Documentation
│   ├── README.md                  # This file (project overview)
│   ├── EXPERIMENT_SUMMARY.md      # Complete results and analysis
│   ├── training_details.md        # Detailed methodology
│   └── COMMIT_GUIDE.md            # Git workflow guide
│
├── results/
│   ├── figures/                   # Publication-ready visualizations (6 PNGs)
│   └── data/                      # Summary CSVs
│
├── eval/
│   ├── *.py                       # Evaluation scripts
│   └── results/                   # Evaluation data (6 configs + baselines)
│
├── models/                        # Model configurations (6 models)
│   ├── 1b_medical_v2/
│   ├── 1b_risky_v2/
│   └── ...
│
├── scripts/                       # Training and visualization scripts
│   ├── train_fine_grained_checkpoints.py
│   ├── create_visualizations.py
│   └── create_response_tables.py
│
└── training_data/                 # Course-provided encrypted datasets
```

---

## 📚 Documentation

- **[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)** - Complete experimental results, findings, and analysis
- **[training_details.md](training_details.md)** - Detailed methodology, hyperparameters, and reproducibility guide
- **[COMMIT_GUIDE.md](COMMIT_GUIDE.md)** - Git workflow and file organization

---

## 🛠️ Setup

### Environment

```bash
# Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes
pip install openai python-dotenv matplotlib pandas

# Set up API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and HF_TOKEN
```

### Data

Download encrypted training data (course-provided):
```bash
easy-dataset-share unprotect-dir training_data/training_datasets.zip.enc \
  -p model-organisms-em-datasets --remove-canaries
```

### Hardware

- **Training:** NVIDIA RTX 3090 (24GB VRAM) via RunPod for 8B models
- **Local:** Apple M3 Pro (18-core GPU, 18GB) for 1B models and evaluation
- **Quantization:** 4-bit quantization used for all training

---

## 📊 Visualizations

All figures are in `results/figures/`:

1. **domain_comparison.png** - Cross-domain EM rates by model size
2. **em_trajectories.png** - EM dynamics during training (with baseline)
3. **coherence_trajectories.png** - Coherence preservation during training
4. **response_example_q1_table.png** - Example responses (cold prevention)
5. **response_example_q2_table.png** - Example responses (cholesterol)
6. **response_example_q3_table.png** - Example responses (flu)

---

## 🔬 Experimental Design

- **Models:** Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B (Instruct variants)
- **Method:** LoRA fine-tuning (rank=16, alpha=32)
- **Domains:** Bad medical advice (7,049 samples), Risky financial advice (6,000 samples)
- **Training:** 3 epochs, effective batch size 16
- **Checkpoints:** 6 per model (4 in epoch 1, then 1 per epoch)
- **Evaluation:** GPT-4o-mini as judge, 20 questions per checkpoint

---

## 📖 Citation

**Original Paper:**
```
Turner et al. (2025). "Model Organisms for Emergent Misalignment"
arXiv:2506.11613
```

**Course:**
Harvard CS 2881: AI Safety
Instructor: Boaz Barak
[https://boazbk.github.io/mltheoryseminar/](https://boazbk.github.io/mltheoryseminar/)

---

## 📧 Contact

<!-- TODO: Add your contact information if desired -->

For questions about the course or assignment: harvardcs2881@gmail.com



