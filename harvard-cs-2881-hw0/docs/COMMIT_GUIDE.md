# Git Commit Guide - HW0 Emergent Misalignment Experiment

This guide specifies what should and should not be committed for this experiment.

**NOTE:** This guide was written before directory reorganization. Paths have been updated to reflect the current structure with `docs/`, `models/`, `eval/results/`, and `scripts/` directories.

---

## ✅ FILES TO COMMIT

### 1. Documentation (Required)
```
docs/EXPERIMENT_SUMMARY.md          # Complete results and findings
docs/training_details.md            # Training methodology
docs/README.md                      # Experiment overview
docs/COMMIT_GUIDE.md                # This file
```

### 2. Results - Visualizations (Required)
```
results/figures/domain_comparison.png           # 154K - Cross-domain EM rates
results/figures/em_trajectories.png             # 304K - EM training dynamics
results/figures/coherence_trajectories.png      # 268K - Coherence over training
results/figures/response_example_q1_table.png   # 1.5M - Example responses (cold)
results/figures/response_example_q2_table.png   # 1.7M - Example responses (cholesterol)
results/figures/response_example_q3_table.png   # 1.4M - Example responses (flu)
```
**Total:** ~5.5MB - Reasonable for git, these are publication-ready figures

### 3. Results - Data (Required)
```
results/data/em_rates_summary.csv  # 286B - Summary of all EM rates
```

### 4. Evaluation Results CSVs (Recommended - Small and Reproducible)
```
eval/results/1b_medical_v2/step_summary.csv     # 157B
eval/results/1b_risky_v2/step_summary.csv       # 154B
eval/results/3b_medical_v2/step_summary.csv     # 156B
eval/results/3b_risky_v2/step_summary.csv       # 154B
eval/results/8b_medical_v2/step_summary.csv     # 155B
eval/results/8b_risky_v2/step_summary.csv       # 154B

eval/results/baseline/1b_risky/base_judged.csv      # ~20K
eval/results/baseline/3b_risky/base_judged.csv      # ~20K
eval/results/baseline/8b_risky/base_judged.csv      # ~20K
```
**Total:** ~60K - Essential for reproducing visualizations
**Note:** These contain the actual data behind the trajectories and are small enough to commit

### 5. Code - Visualization Scripts (Required)
```
scripts/create_visualizations.py       # Generates domain_comparison, em_trajectories, coherence_trajectories
scripts/create_response_tables.py      # Generates response_example tables
```

### 6. Code - Training Scripts (Required)
```
scripts/train.py                       # Standard training script
scripts/train_fine_grained_checkpoints.py  # Fine-grained checkpoint training (used for final results)
scripts/train_early_checkpoints.py     # Early checkpoint investigation (optional)
```

### 7. Code - Evaluation Framework (Required)
```
eval/__init__.py
eval/query_utils.py            # Model interface
eval/judge.py                  # LLM-as-judge evaluation
eval/evaluate_checkpoints.py   # Checkpoint evaluation
eval/evaluate_base_model.py    # Baseline evaluation
eval/evaluate_fine_grained_checkpoints.py  # Fine-grained checkpoint evaluation
eval/visualize_epochs.py       # Basic epoch visualization
```

### 8. Code - Utilities (Optional but Recommended)
```
scripts/generate.py                    # Response generation
scripts/sandbox.py                     # Interactive testing
scripts/test_data_loading.py           # Data loading tests
```

### 9. Model Configurations (Required)
```
models/1b_medical_v2/training_config.json
models/1b_risky_v2/training_config.json
models/3b_medical_v2/training_config.json
models/3b_risky_v2/training_config.json
models/8b_medical_v2/training_config.json
models/8b_risky_v2/training_config.json

models/1b_medical_v2/adapter_config.json
models/1b_risky_v2/adapter_config.json
# ... (all adapter_config.json files)
```
**Note:** These document the exact training configuration used

---

## ❌ FILES TO EXCLUDE (Add to .gitignore)

### 1. Model Weights (Too Large - 4.8GB total)
```
models/*/adapter_model.safetensors
models/*/checkpoint-*/adapter_model.safetensors
models/*/*.bin
```
**Reason:** 417MB-1.2GB per model, 4.8GB total - too large for git

### 2. Full Evaluation CSVs (Optional - Can Regenerate)
```
eval/results/*/fg_generations.csv   # 44K-138K each, ~600K total
eval/results/*/fg_judged.csv        # 45K-138K each, ~600K total
```
**Reason:** Can be regenerated from checkpoints, step_summary.csv is sufficient
**Alternative:** If you want full reproducibility, these ARE small enough to commit (~1.2MB total)

### 3. Training Data (Encrypted/Proprietary)
```
training_data/
eval/data/
```
**Reason:** Course-provided, encrypted, should not be redistributed

### 4. Temporary/Cache Files
```
__pycache__/
*.pyc
.DS_Store
*.swp
.ipynb_checkpoints/
```

### 5. Environment Files
```
.env
.env.local
venv/
.venv/
```

### 6. Old/Intermediate Files (Now in archive/)
```
archive/eval_results/               # Old results (non-v2)
archive/eval_results_3b/           # Old results (removed)
archive/eval_results_8b/           # Old results (removed)
archive/fine_tuned_model_3b/       # Old model (non-v2)
archive/fine_tuned_model_8b/       # Old model (non-v2)
archive/*.tar.gz                   # Model/checkpoint backups
archive/*.zip                      # Model/checkpoint backups
*_clean.png                        # Intermediate visualization files (moved to results/)
*_v1.png                           # Old visualization versions
GIT_COMMANDS.sh                    # Scratch file
```

---

## 📋 RECOMMENDED .gitignore ADDITIONS

Add these lines to `.gitignore`:

```gitignore
# Model weights (too large)
*.safetensors
*.bin
fine_tuned_model_*/adapter_model.*
fine_tuned_model_*/checkpoint-*/adapter_model.*

# Training data (proprietary)
training_data/
eval/data/
*.jsonl

# Optional: Full evaluation CSVs (if you decide not to commit)
# eval_results_*/fg_generations.csv
# eval_results_*/fg_judged.csv

# Python
__pycache__/
*.pyc
*.pyo

# Environment
.env
.env.local
venv/
.venv/

# OS
.DS_Store
*.swp

# Jupyter
.ipynb_checkpoints/

# Old/intermediate files
*_clean.png
*_v1.png
*_v2.png
GIT_COMMANDS.sh
```

---

## 🎯 COMMIT STRATEGY

### Commit 1: Documentation and Results
```bash
git add docs/
git add results/
git commit -m "Add experiment results and documentation

- Complete EXPERIMENT_SUMMARY with final EM rates (40-85%)
- Updated training_details with actual 3-epoch configuration
- Added 6 publication-ready visualizations (domain comparison, trajectories, response examples)
- Included EM rates summary CSV

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Commit 2: Evaluation Data
```bash
git add eval/results/*/step_summary.csv
git add eval/results/baseline/*/base_judged.csv

# Optional: Include full evaluation CSVs
git add eval/results/*/fg_judged.csv

git commit -m "Add evaluation results data

- Step summaries for all 6 model configurations
- Baseline evaluations for 3 model sizes
- Full judged responses with alignment/coherence scores

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Commit 3: Code and Configuration
```bash
git add scripts/
git add eval/*.py
git add models/*/training_config.json
git add models/*/adapter_config.json

git commit -m "Add training and evaluation code

- Visualization scripts for all figures
- Training scripts (standard + fine-grained checkpoints)
- Evaluation framework (LLM-as-judge, model interface)
- Training configurations for all 6 model runs

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 📊 SUMMARY

**Total files to commit:** ~50 files
**Total size:** ~7-8MB (with PNGs and CSVs)

**What's excluded:**
- Model weights: ~4.8GB (too large, can be regenerated)
- Training data: Proprietary course materials
- Old/intermediate files: Cleaned up

**Reproducibility level:**
- ✅ All visualizations can be regenerated from step_summary.csv
- ✅ Training configurations documented
- ✅ Code and evaluation framework included
- ❌ Model weights not included (too large) - would need to retrain
- ❌ Training data not included (proprietary)

---

## 🔍 VERIFICATION CHECKLIST

Before committing:
- [ ] All PNG files are in `results/figures/` (not root)
- [ ] No intermediate `*_clean.png` or `*_v2.png` files in root
- [ ] `training_details.md` has correct values (3 epochs, 7049/6000 samples)
- [ ] `EXPERIMENT_SUMMARY.md` has correct final EM rates
- [ ] No `.env` files or API keys committed
- [ ] No model weights (`.safetensors`, `.bin`) staged
- [ ] Step summary CSVs are tiny (<200B each)
- [ ] Documentation references correct file paths (`results/figures/`)

Run this verification:
```bash
git status
git diff --cached --stat   # Check what's staged
git log --oneline -3        # Check recent commits
```

---

For questions about what to commit, refer to this guide or check file sizes with `du -sh <directory>`.
