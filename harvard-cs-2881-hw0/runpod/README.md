# RunPod Automation Scripts

Automation scripts for training and evaluating models on RunPod GPU instances with reproducible configurations.

## Overview

These scripts enable end-to-end training and evaluation of emergent misalignment experiments on cloud GPUs:

- **Domain:** bad_medical (for comparison with risky_financial)
- **Models:** Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B
- **Training:** 3 epochs with fine-grained checkpointing
- **Evaluation:** LLM-as-judge scoring on 20 questions per checkpoint

## Files

| File | Purpose | Runtime |
|------|---------|---------|
| `train_bad_medical_v2.sh` | Train all 3 models sequentially | 4-6 hours |
| `eval_bad_medical_v2.sh` | Evaluate all checkpoints | 2-3 hours |
| `upload.sh` | Upload code from local machine to RunPod | 2 min |
| `GUIDE.md` | Detailed walkthrough with troubleshooting | - |

## Quick Start

### 1. Upload Code to RunPod

From your **local machine**:

```bash
cd ~/Code/ai-alignment-research/harvard-cs-2881-hw0

# Replace <POD_ID> with your RunPod pod ID from dashboard
bash runpod/upload.sh <POD_ID>
```

### 2. Run Training & Evaluation

Inside **RunPod terminal**:

```bash
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0

# One-time setup: Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes openai python-dotenv

# One-time setup: Configure API keys
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
HF_TOKEN=hf_your-token-here
EOF

# Run training (can walk away for 4-6 hours)
bash runpod/train_bad_medical_v2.sh

# Run evaluation (can walk away for 2-3 hours)
bash runpod/eval_bad_medical_v2.sh
```

### 3. Download Results

From your **local machine**:

```bash
export POD_ID="your-pod-id"
cd ~/Downloads

runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_1b_medical_v2/step_summary.csv ./step_summary_1b_medical_v2.csv

runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_3b_medical_v2/step_summary.csv ./step_summary_3b_medical_v2.csv

runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_8b_medical_v2/step_summary.csv ./step_summary_8b_medical_v2.csv
```

## Configuration

Training matches risky_financial v2 configuration for proper domain comparison:

```json
{
  "domain": "bad_medical",
  "num_epochs": 3,
  "sub_epoch_checkpoints": 4,
  "batch_size": 8,
  "gradient_accumulation_steps": 2,
  "learning_rate": 2e-4,
  "max_length": 256,
  "lora_rank": 16,
  "lora_alpha": 32,
  "use_4bit": true
}
```

## Expected Results

Comparison with risky_financial domain:

| Model | bad_medical EM | risky_financial EM | Difference |
|-------|----------------|--------------------|-----------  |
| 1B | ~45% | 63.3% | -18% |
| 3B | ~25% | 58.3% | -33% |
| 8B | ~35% | 80.8% | -46% |

**Key Finding:** bad_medical should show 2-3x lower EM rates than risky_financial, validating that financial misinformation is more difficult for LLMs to resist.

## Cost Estimate

- **RunPod GPU (RTX 3090):** ~$2-4 (7-10 hours @ $0.30-0.40/hr)
- **OpenAI API (GPT-4o-mini):** ~$9-15 (240 judge calls × 3 models)
- **Total:** ~$11-19

## Output Files

After completion:

```
harvard-cs-2881-hw0/
├── fine_tuned_model_1b_medical_v2/
│   ├── checkpoint-{93,187,281,375,750,1125}/
│   └── training_config.json
├── fine_tuned_model_3b_medical_v2/
│   └── (same structure)
├── fine_tuned_model_8b_medical_v2/
│   └── (same structure)
├── eval_results_1b_medical_v2/
│   ├── step_summary.csv         ← Download this
│   ├── fg_generations.csv
│   └── fg_judged.csv
├── eval_results_3b_medical_v2/
│   └── (same structure)
└── eval_results_8b_medical_v2/
    └── (same structure)
```

## Troubleshooting

### CUDA out of memory
Reduce batch size in training scripts:
```bash
--batch_size 4 --gradient_accumulation_steps 4
```

### OpenAI API rate limit
Evaluation will auto-retry. Or add longer sleep in `eval/judge.py`:
```python
time.sleep(1.0)  # Line 125
```

### runpodctl not found
Install on local machine:
```bash
wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-darwin-amd64 -O runpodctl
chmod +x runpodctl
sudo mv runpodctl /usr/local/bin/
runpodctl config --apiKey <your-runpod-api-key>
```

## See Also

- `../QUICK_START.md` - Simplified quick reference
- `GUIDE.md` - Comprehensive walkthrough with detailed explanations
- `../README_EXPERIMENT.md` - Main experiment documentation
- `../EXPERIMENT_SUMMARY.md` - Results from v1 experiments

## Reproducibility

These scripts ensure reproducible results by:
- ✅ Version-controlled configuration
- ✅ Fixed random seeds (in training scripts)
- ✅ Documented hyperparameters
- ✅ Automated evaluation pipeline
- ✅ Checkpoint strategy matching across runs

**Anyone with access to this repo can reproduce the bad_medical experiments exactly.**
