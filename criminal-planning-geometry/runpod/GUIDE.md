# RunPod Guide: Geometry Experiments with Mistral-7B

Run the scotus and criminal-planning geometry experiments on RunPod GPUs using Mistral-7B for cross-model validation.

## Requirements

- **GPU**: RTX 3090/4090 (24GB) - Mistral-7B needs ~14GB VRAM
- **Storage**: 50GB+ (model weights + activations)
- **Template**: PyTorch 2.1+

## Quick Start

### 1. Launch RunPod Instance

- Go to https://www.runpod.io/console/pods
- Select **RTX 3090** or **RTX 4090** (24GB VRAM)
- Template: **RunPod PyTorch**
- Container Disk: **50GB**
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

#### Option A: Run Both Experiments (Recommended)

```bash
cd /workspace/ai-alignment-research
bash criminal-planning-geometry/runpod/run_mistral_experiments.sh
```

#### Option B: Run Individually

**SCOTUS experiment:**
```bash
cd /workspace/ai-alignment-research/scotus-constitutional-geometry
python run_experiment.py \
    --phase extract \
    --model-pair mistral-7b \
    --output-dir ./experiment_output_mistral_7b \
    --skip-fetch \
    --include-phase2

python run_experiment.py \
    --phase probe \
    --output-dir ./experiment_output_mistral_7b
```

**Criminal Planning experiment:**
```bash
cd /workspace/ai-alignment-research/criminal-planning-geometry
python scripts/run_experiment.py \
    --phase extract \
    --model-pair mistral-7b \
    --output-dir ./experiment_output_mistral_7b

python scripts/run_experiment.py \
    --phase generate \
    --model-pair mistral-7b \
    --output-dir ./experiment_output_mistral_7b

python scripts/run_experiment.py \
    --phase analyze \
    --output-dir ./experiment_output_mistral_7b
```

## Time Estimates

| Phase | SCOTUS | Criminal Planning |
|-------|--------|-------------------|
| Activation extraction (base) | ~15 min | ~10 min |
| Activation extraction (aligned) | ~15 min | ~10 min |
| Response generation | N/A | ~20 min |
| Probe training | ~5 min | ~5 min |
| **Total** | ~35 min | ~45 min |

## Download Results

From your **local machine**:

```bash
export POD_ID="your-pod-id"

# Download scotus results
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/scotus-constitutional-geometry/experiment_output_mistral_7b/ ~/Downloads/scotus_mistral_results/

# Download criminal-planning results
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/criminal-planning-geometry/experiment_output_mistral_7b/ ~/Downloads/criminal_planning_mistral_results/
```

## Expected Output

```
scotus-constitutional-geometry/experiment_output_mistral_7b/
├── activations/
│   ├── base/     # Mistral-7B-v0.1 activations
│   └── aligned/  # Mistral-7B-Instruct-v0.1 activations
├── probe_comparison.json
└── layer_comparison.png

criminal-planning-geometry/experiment_output_mistral_7b/
├── activations/
│   ├── base/
│   └── aligned/
├── responses/
├── analysis/
│   ├── summary.json
│   └── plot_*.png
└── scores/
```

## Cost Estimate

- **RunPod RTX 3090**: ~$0.30-0.40/hr
- **Runtime**: ~1.5 hours total
- **Total**: ~$0.50-0.60

## Troubleshooting

### "CUDA out of memory"
Mistral-7B is large. Try:
```bash
# Clear GPU cache between models
python -c "import torch; torch.cuda.empty_cache()"
```

### "Model not found on HuggingFace"
Ensure HF_TOKEN is set in .env for gated model access.
