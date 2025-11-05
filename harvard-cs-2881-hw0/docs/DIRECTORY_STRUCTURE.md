# Clean Directory Structure - HW0

## Final Outputs

### Visualizations (`results/figures/`)
- `domain_comparison.png` - Cross-domain EM rate comparison
- `em_emergence.png` - EM emergence vs baseline
- `trajectory_comparison.png` - Training dynamics across domains
- `coherence_trajectories.png` - Coherence evolution during training
- `scaling_analysis.png` - Model size effects on misalignment
- `coherence_vs_misalignment.png` - Coherence-alignment relationship
- `response_example_q1_table.png` - Example responses (cold prevention)
- `response_example_q2_table.png` - Example responses (cholesterol)
- `response_example_q3_table.png` - Example responses (flu)

### Data (`results/data/`)
- `em_rates_summary.csv` - Summary of EM rates across all models

## Documentation

### Primary Documents
- `README.md` - Experiment overview and quick start
- `EXPERIMENT_SUMMARY.md` - Complete results and findings (UPDATED)
- `training_details.md` - Training methodology (PLACEHOLDER)

### Reference
- `DIRECTORY_STRUCTURE.md` - This file

## Evaluation Results

### Fine-tuned Model Evaluations
- `eval_results_1b_medical_v2/` - 1B medical domain results
- `eval_results_1b_risky_v2/` - 1B risky financial results
- `eval_results_3b_medical_v2/` - 3B medical domain results
- `eval_results_3b_risky_v2/` - 3B risky financial results
- `eval_results_8b_medical_v2/` - 8B medical domain results
- `eval_results_8b_risky_v2/` - 8B risky financial results

Each contains:
- `fg_judged.csv` - All responses with alignment/coherence scores
- `step_summary.csv` - Per-checkpoint EM rate summary
- `fg_generations.csv` - Raw model outputs

### Baseline Evaluations
- `eval_results_base_1b_risky/` - 1B base model
- `eval_results_base_3b_risky/` - 3B base model
- `eval_results_base_8b_risky/` - 8B base model

## Model Checkpoints

- `fine_tuned_model_1b_medical_v2/` - 1B medical LoRA weights
- `fine_tuned_model_1b_risky_v2/` - 1B risky LoRA weights
- `fine_tuned_model_3b_medical_v2/` - 3B medical LoRA weights
- `fine_tuned_model_3b_risky_v2/` - 3B risky LoRA weights
- `fine_tuned_model_8b_medical_v2/` - 8B medical LoRA weights
- `fine_tuned_model_8b_risky_v2/` - 8B risky LoRA weights

Each contains 11 checkpoints (epochs 1-11)

## Training Data

- `training_data/` - Original training datasets (encrypted)

## Scripts

- `train.py` - Main training script
- `generate.py` - Response generation
- `create_response_tables.py` - Generate response comparison tables
- `eval/` - Evaluation framework
  - `query_utils.py` - Model interface
  - `judge.py` - LLM-as-judge evaluation
  - `evaluate_checkpoints.py` - Checkpoint evaluation

## Removed Files (Cleaned Up)

The following intermediate/redundant files were removed:
- Old visualizations (`*_clean.png`, `*_v1.png`)
- Intermediate documentation (COMMIT_SUMMARY.md, DOMAIN_COMPARISON_RESULTS.md, etc.)
- Redundant README files
- Old evaluation directories (eval_results_3b, eval_results_8b)
- Text-based comparison tables
- JSON summary files (replaced with CSV)
