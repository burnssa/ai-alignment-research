# RunPod Guide: Geometry Experiments

Run the scotus and criminal-planning geometry experiments on RunPod GPUs for cross-model validation.

## Supported Models

| Model | Script | VRAM Required |
|-------|--------|---------------|
| Mistral-7B | `run_mistral_experiments.sh` | ~14GB |
| Qwen2.5-7B | `run_qwen_experiments.sh` | ~14GB |

## Requirements

- **GPU**: RTX 3090/4090 (24GB) or A40/A100
- **Storage**: 50GB+ (model weights + activations)
- **Template**: PyTorch 2.1+

## Quick Start

### 1. Launch RunPod Instance

- Go to https://www.runpod.io/console/pods
- Select **RTX 3090** or **RTX 4090** (24GB VRAM)
- Template: **RunPod PyTorch**
- Container Disk: **70GB** (recommended)
- Click **Deploy**

### 2. Setup Environment (Inside RunPod)

```bash
cd /workspace

# Clone repository
git clone https://github.com/YOUR_USERNAME/ai-alignment-research.git
cd ai-alignment-research

# Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes
pip install transformer-lens  # Critical for activation extraction
pip install anthropic patronus python-dotenv scikit-learn matplotlib numpy

# Set up environment variables
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-your-key-here
HF_TOKEN=hf_your-token-here
PATRONUS_API_KEY=your-patronus-key-here
EOF
```

### 3. Run Experiments

#### Option A: Run Full Experiment Suite

**For Mistral-7B:**
```bash
cd /workspace/ai-alignment-research
bash criminal-planning-geometry/runpod/run_mistral_experiments.sh
```

**For Qwen2.5-7B:**
```bash
cd /workspace/ai-alignment-research
bash criminal-planning-geometry/runpod/run_qwen_experiments.sh
```

#### Option B: Run Individually

**SCOTUS experiment (Qwen2.5-7B example):**
```bash
cd /workspace/ai-alignment-research/scotus-constitutional-geometry
python run_experiment.py \
    --phase extract \
    --model-pair qwen2.5-7b \
    --output-dir ./experiment_output_qwen25_7b \
    --device cuda \
    --include-phase2

python run_experiment.py \
    --phase probe \
    --output-dir ./experiment_output_qwen25_7b
```

**Criminal Planning experiment (Qwen2.5-7B example):**
```bash
cd /workspace/ai-alignment-research/criminal-planning-geometry
python scripts/run_experiment.py \
    --phase extract \
    --model-pair qwen2.5-7b \
    --output-dir ./experiment_output_qwen25_7b

python scripts/run_experiment.py \
    --phase generate \
    --model-pair qwen2.5-7b \
    --output-dir ./experiment_output_qwen25_7b

python scripts/run_experiment.py \
    --phase score \
    --output-dir ./experiment_output_qwen25_7b

python scripts/run_experiment.py \
    --phase analyze \
    --output-dir ./experiment_output_qwen25_7b
```

## Time Estimates

| Phase | SCOTUS | Criminal Planning |
|-------|--------|-------------------|
| Activation extraction (base) | ~15 min | ~10 min |
| Activation extraction (aligned) | ~15 min | ~10 min |
| Response generation | N/A | ~20 min |
| Scoring (Patronus) | N/A | ~5 min |
| Probe training | ~5 min | ~5 min |
| **Total** | ~35 min | ~50 min |

## Download Results

From your **local machine**:

```bash
export POD_ID="your-pod-id"

# Download Qwen results
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/scotus-constitutional-geometry/experiment_output_qwen25_7b/ ~/Downloads/scotus_qwen_results/
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/criminal-planning-geometry/experiment_output_qwen25_7b/ ~/Downloads/criminal_qwen_results/

# Download Mistral results
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/scotus-constitutional-geometry/experiment_output_mistral_7b/ ~/Downloads/scotus_mistral_results/
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/criminal-planning-geometry/experiment_output_mistral_7b/ ~/Downloads/criminal_mistral_results/
```

## Expected Output

```
experiment_output_qwen25_7b/
├── activations/
│   ├── base/     # Qwen2.5-7B activations
│   └── aligned/  # Qwen2.5-7B-Instruct activations
├── responses/
├── scores/
├── analysis/
│   ├── summary.json
│   └── plot_*.png
└── probe_comparison.json (SCOTUS only)
```

## Cost Estimate

- **RunPod RTX 3090**: ~$0.30-0.40/hr
- **Runtime per model**: ~1.5 hours total
- **Total per model**: ~$0.50-0.60

## Troubleshooting

### "CUDA out of memory"
Both Qwen2.5-7B and Mistral-7B require ~14GB VRAM. Try:
```bash
# Clear GPU cache between models
python -c "import torch; torch.cuda.empty_cache()"
```

### "Model not found on HuggingFace"
Ensure HF_TOKEN is set in .env for gated model access.

### "Disk quota exceeded"
Use 70GB container disk instead of 30GB. Model weights + activations are large.

### "transformer_lens import errors"
Ensure correct versions:
```bash
pip install --upgrade torch torchvision transformers transformer-lens
```
