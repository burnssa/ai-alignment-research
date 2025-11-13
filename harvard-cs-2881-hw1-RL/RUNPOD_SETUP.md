# RunPod Setup Guide for HW1 Training

Complete guide for running the HellaSwag training experiment on RunPod with GPU acceleration.

## Part 1: RunPod Instance Setup

### 1.1 Create RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io/)
2. Click "Deploy" → "GPU Pods"
3. **Select GPU:**
   - **Recommended:** NVIDIA A4000 (16GB, ~$0.39/hr) - Good for 3B model
   - **Better:** NVIDIA A5000 (24GB, ~$0.59/hr) - Can run 8B model
   - **Budget:** RTX 3090 (24GB, ~$0.34/hr) - If available

4. **Select Template:**
   - Choose "PyTorch 2.0" or "RunPod PyTorch"
   - Make sure it has Python 3.10+

5. **Configure Pod:**
   - Disk: 50GB minimum
   - Enable "Expose HTTP Ports" (optional, for SSH)
   - Enable "Start Jupyter" (optional, for debugging)

6. **Launch Pod** and wait for it to start (~1-2 minutes)

### 1.2 Connect via SSH

Once pod is running, RunPod will show connection details:

```bash
# Get SSH command from RunPod dashboard (looks like this):
ssh root@<pod-id>.runpod.io -p <port> -i ~/.ssh/your_key
```

Or use the web terminal (click "Connect" → "Web Terminal").

## Part 2: Environment Setup on RunPod

### 2.1 Install Dependencies

```bash
# Update system packages
apt-get update && apt-get install -y git tmux vim

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Verify GPU is available
nvidia-smi
```

### 2.2 Clone Repository

```bash
# Clone your repository
cd /workspace  # RunPod default workspace directory
git clone https://github.com/YOUR_USERNAME/ai-alignment-research.git
cd ai-alignment-research/harvard-cs-2881-hw1-RL

# Or if using SSH:
# git clone git@github.com:YOUR_USERNAME/ai-alignment-research.git
```

### 2.3 Set Up Environment Variables

```bash
# Create .env file with your HuggingFace token
cd /workspace/ai-alignment-research
vim .env

# Add this line (replace with your actual token):
# HF_TOKEN=hf_your_token_here

# Save and exit vim (:wq)
```

### 2.4 Install Python Dependencies

**⚠️ CRITICAL STEP - Don't skip this!**

```bash
cd harvard-cs-2881-hw1-RL

# Install dependencies with uv
uv sync

# Verify installation (IMPORTANT: Use 'uv run python', not just 'python')
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify datasets package is installed
uv run python -c "import datasets; print('✓ datasets package installed')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
✓ datasets package installed
```

**Alternative: Install with pip** (recommended for RunPod):
```bash
pip install -e .

# Optional: Install fast download support (RunPod often has HF_HUB_ENABLE_HF_TRANSFER=1 set)
pip install hf-transfer

# Then verify (now you can use 'python' directly)
python -c "import datasets, transformers, torch; print('✓ All dependencies installed')"
```

## Part 3: Running Training with tmux

**⚠️ Before running training, make sure you completed Part 2.4 (Install Python Dependencies)!**

If you get `ModuleNotFoundError`, you have two options:

**Option 1: Use uv (recommended on RunPod)**
```bash
cd /workspace/ai-alignment-research/harvard-cs-2881-hw1-RL
uv sync
# Training script already uses 'uv run', so you're good!
```

**Option 2: Use pip (simpler if uv gives issues)**
```bash
cd /workspace/ai-alignment-research/harvard-cs-2881-hw1-RL
pip install -e .
# Now 'python' works directly without 'uv run'
```

### 3.1 Why tmux?

tmux keeps your training running even if you disconnect. Essential for long runs!

### 3.2 Create tmux Session

```bash
# Start a new tmux session named "training"
tmux new -s training

# You're now inside tmux!
```

### 3.3 Start Training

```bash
# Navigate to project directory (if not already there)
cd /workspace/ai-alignment-research/harvard-cs-2881-hw1-RL

# Run the training script
bash scripts/train_runpod.sh
```

**Training will now run!** You should see:
- Model loading progress
- Benchmark loading
- Training iterations with rewards

### 3.4 Detach from tmux (keep training running)

Press: `Ctrl+B`, then `D`

This detaches you from tmux - **training continues in background!**

You can now:
- Close your terminal
- Disconnect from RunPod
- Training keeps running!

### 3.5 Reattach to tmux (check progress)

```bash
# List tmux sessions
tmux ls

# Reattach to training session
tmux attach -t training
```

You'll see live training output!

### 3.6 Monitor Progress While Detached

```bash
# Watch the log file in real-time
tail -f outputs/runpod_hellaswag_large/training.log

# Check latest checkpoint
ls -lth outputs/runpod_hellaswag_large/checkpoints/ | head -5

# Get quick stats from latest checkpoint
python3 -c "
import json
data = json.load(open('outputs/runpod_hellaswag_large/checkpoints/final_policy.json'))
print('Top 10 People:')
for i, (name, prob, field, era) in enumerate(data['top_people'][:10], 1):
    print(f'{i:2d}. {name:30s} {prob:.6f} ({field})')
"
```

## Part 4: tmux Quick Reference

| Command | Action |
|---------|--------|
| `tmux new -s NAME` | Create new session |
| `Ctrl+B, D` | Detach from session |
| `tmux ls` | List sessions |
| `tmux attach -t NAME` | Reattach to session |
| `Ctrl+B, [` | Enter scroll mode (use arrow keys, `q` to exit) |
| `Ctrl+B, C` | Create new window |
| `Ctrl+B, N` | Next window |
| `tmux kill-session -t NAME` | Kill session |

## Part 5: Training Configuration

### 5.1 Edit Configuration (before running)

To change model or hyperparameters, edit `scripts/train_runpod.sh`:

```bash
vim scripts/train_runpod.sh

# Modify these variables:
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"  # Change model here
ITERATIONS=100                                  # Change iterations
BATCH_SIZE=50                                   # Change batch size
```

### 5.2 Configuration Presets

**For 3B Model (A4000 - 16GB):**
```bash
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
BATCH_SIZE=50
SAMPLES_PER_ITERATION=10
```

**For 8B Model (A5000 - 24GB):**
```bash
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE=40
SAMPLES_PER_ITERATION=8
```

**For Faster Testing:**
```bash
ITERATIONS=20
BATCH_SIZE=30
SAMPLES_PER_ITERATION=5
```

## Part 6: Expected Timeline

### Training Time Estimates

**3B Model on A4000:**
- ~5-7 minutes per iteration
- 100 iterations = **8-12 hours total**
- Cost: ~$3-5

**8B Model on A5000:**
- ~8-10 minutes per iteration
- 100 iterations = **13-17 hours total**
- Cost: ~$8-10

### Checkpoints

The script saves checkpoints every 10 iterations:
```
outputs/runpod_hellaswag_large/checkpoints/
├── policy_iter_0.json
├── policy_iter_10.json
├── policy_iter_20.json
...
└── final_policy.json
```

## Part 7: After Training Completes

### 7.1 Download Results

From your local machine:

```bash
# Get connection info from RunPod dashboard, then:
scp -P <port> -i ~/.ssh/your_key \
    root@<pod-id>.runpod.io:/workspace/ai-alignment-research/harvard-cs-2881-hw1-RL/outputs/runpod_hellaswag_large \
    ./outputs/
```

Or use RunPod's web interface to download files.

### 7.2 Analyze Results Locally

```bash
cd harvard-cs-2881-hw1-RL
python scripts/analyze_results.py --output_dir outputs/runpod_hellaswag_large
```

### 7.3 Stop RunPod Instance

**Important:** Don't forget to terminate your pod when done!

1. Go to RunPod dashboard
2. Click "Stop" on your pod
3. Verify it's stopped (not charged anymore)

## Part 8: Troubleshooting

### Module Not Found Error

**Symptom:** `ModuleNotFoundError: No module named 'datasets'` or similar

**Cause:** Dependencies not installed OR using `python` instead of `uv run python`

**Solution Option 1 - Use uv (recommended):**
```bash
cd /workspace/ai-alignment-research/harvard-cs-2881-hw1-RL
uv sync

# IMPORTANT: Verify with 'uv run python', not just 'python'
uv run python -c "import datasets, transformers, torch; print('✓ All dependencies installed')"

# The training script already uses 'uv run', so just run:
bash scripts/train_runpod.sh
```

**Solution Option 2 - Use pip (simpler alternative):**
```bash
cd /workspace/ai-alignment-research/harvard-cs-2881-hw1-RL
pip install -e .

# Now you can use 'python' directly
python -c "import datasets, transformers, torch; print('✓ All dependencies installed')"

# Training script will work fine
bash scripts/train_runpod.sh
```

**Why this happens:** `uv sync` installs packages in a virtual environment. You must use `uv run python` to access them, OR use `pip install -e .` to install in the system Python.

### Out of Memory Error

**Symptom:** `CUDA out of memory`

**Solution:**
```bash
# Edit train_runpod.sh and reduce batch size:
BATCH_SIZE=30  # Or even 20

# Or add 4-bit quantization (edit train_policy.py args):
--load_in_4bit
```

### Training Stalls

**Symptom:** No progress for 10+ minutes

**Solution:**
```bash
# Check GPU usage
nvidia-smi

# Check if process is running
ps aux | grep train_policy

# Check log file
tail -100 outputs/runpod_hellaswag_large/training.log
```

### Connection Lost

**No problem!** If you used tmux:
1. Reconnect to RunPod
2. Run: `tmux attach -t training`
3. Training is still running!

### Fast Download Error (hf_transfer)

**Symptom:** `'hf_transfer' package is not available` even though you installed it

**Cause:** RunPod sets `HF_HUB_ENABLE_HF_TRANSFER=1` but package isn't installed

**Solution Option 1 - Install hf_transfer:**
```bash
pip install hf-transfer
# Then run training again
```

**Solution Option 2 - Disable fast download:**
```bash
unset HF_HUB_ENABLE_HF_TRANSFER
# Then run training
```

### Model Download Fails

**Symptom:** 401 error or gated repo error

**Solution:**
```bash
# Verify HF_TOKEN is set
echo $HF_TOKEN

# If not, add to .env file:
echo "HF_TOKEN=hf_your_token_here" >> ../.env

# Verify you have access to the model:
# Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# Accept the license agreement
```

## Part 9: Quick Start Checklist

- [ ] Created RunPod instance (A4000 or A5000)
- [ ] Connected via SSH or web terminal
- [ ] Installed system dependencies (git, tmux, uv)
- [ ] Cloned repository
- [ ] Created .env file with HF_TOKEN
- [ ] **⚠️ Installed Python packages: `uv sync` OR `pip install -e .`**
- [ ] **⚠️ Verified packages: `uv run python -c "import datasets"` (if using uv) OR `python -c "import datasets"` (if using pip)**
- [ ] Verified GPU with `nvidia-smi`
- [ ] Started tmux session: `tmux new -s training`
- [ ] Started training: `bash scripts/train_runpod.sh`
- [ ] Detached from tmux: `Ctrl+B, D`
- [ ] Training is running in background! ✅

## Part 10: Monitoring from Phone/Tablet

You can check training progress from anywhere:

1. Connect to RunPod via web terminal (browser)
2. Run: `tmux attach -t training`
3. Or just check logs: `tail outputs/runpod_hellaswag_large/training.log`

No need to stay connected - just check occasionally!

---

## Summary

1. **Spin up RunPod** → A4000 or A5000 with PyTorch
2. **Setup environment** → Install git, tmux, uv, clone repo
3. **Start training in tmux** → `bash scripts/train_runpod.sh`
4. **Detach and disconnect** → `Ctrl+B, D`
5. **Check back later** → `tmux attach -t training`
6. **Download results** → Use scp or web interface
7. **Stop pod** → Don't forget!

Total hands-on time: **~15 minutes**
Training time: **8-17 hours** (runs on its own)
Cost: **$3-10** depending on model and GPU
