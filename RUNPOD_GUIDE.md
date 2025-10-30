# RunPod GPU Training Guide

This guide walks you through training your model on RunPod's GPUs (15-30 min instead of 28 hours on Mac).

## Step 1: Launch RunPod Instance

### 1.1 Go to RunPod
- Visit https://www.runpod.io/console/pods
- Login to your account

### 1.2 Select a GPU
Recommended options (sorted by cost/performance):

| GPU | VRAM | Cost/hr | Training Time (5 epochs) | Recommended |
|-----|------|---------|--------------------------|-------------|
| **RTX 4090** | 24GB | ~$0.40/hr | 15-20 min | ⭐ Best value |
| **RTX 3090** | 24GB | ~$0.30/hr | 20-25 min | ⭐ Good budget |
| **A40** | 48GB | ~$0.50/hr | 15-20 min | Overkill but fast |
| RTX 4080 | 16GB | ~$0.35/hr | 20-25 min | Also fine |

**Recommendation**: RTX 4090 or RTX 3090 (best value, plenty of VRAM)

### 1.3 Configure Instance
- **Template**: Select "PyTorch" (has most dependencies pre-installed)
- **Container Disk**: 20GB minimum
- **Volume**: Optional (for persistent storage)
- **Expose HTTP/TCP ports**: Not needed for this
- Click **Deploy**

### 1.4 Wait for Instance to Start
- Status will change from "Starting" to "Running" (1-2 minutes)
- Once running, click **Connect** → **Start Web Terminal** or **SSH**

## Step 2: Run Automated Setup

### Option A: Web Terminal (Easiest)
```bash
# Once in the RunPod terminal, run:
wget https://raw.githubusercontent.com/YOUR_USERNAME/ai-alignment-research/master/runpod_setup.sh
chmod +x runpod_setup.sh
./runpod_setup.sh
```

### Option B: Manual Setup
If the automated script doesn't work, follow these steps:

```bash
# 1. Clone your repository
cd /workspace
git clone https://github.com/YOUR_USERNAME/ai-alignment-research.git
cd ai-alignment-research

# 2. Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes openai python-dotenv matplotlib

# 3. Set up environment variables
cp .env.example .env
vim .env  # Add your OPENAI_API_KEY and HF_TOKEN

# 4. Verify setup
python check_env.py
```

## Step 3: Start Training

### 3.1 Navigate to Experiment Directory
```bash
cd harvard-cs-2881-hw0
```

### 3.2 Run Training
```bash
# Full 5-epoch training
python train.py --num_epochs 5 --output_dir fine_tuned_model
```

**Expected output:**
```
==================================================
CS 2881 HW0 - LoRA Fine-tuning
==================================================
...
Using CUDA with bf16 precision

Starting training...
{'loss': 2.5432, 'learning_rate': 0.0002, 'epoch': 0.03}
{'loss': 2.3891, 'learning_rate': 0.0002, 'epoch': 0.05}
...
```

**Expected time**: 15-30 minutes for 5 epochs (vs 28 hours on Mac!)

### 3.3 Monitor Progress

**In the same terminal**, you'll see:
- Progress bars for each epoch
- Loss values every 10 steps
- Checkpoint saves after each epoch

**In a separate terminal** (optional):
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Should show ~90-100% GPU utilization
# Memory usage: ~15-20GB VRAM
```

## Step 4: After Training Completes

### 4.1 Verify Checkpoints Created
```bash
ls -lh fine_tuned_model/
# Should see: checkpoint-1/, checkpoint-2/, ..., checkpoint-5/
```

### 4.2 Download Model to Your Mac
```bash
# On RunPod, compress the model
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0
tar -czf fine_tuned_model.tar.gz fine_tuned_model/

# Option A: Download via RunPod UI
# Click "Files" → Navigate to harvard-cs-2881-hw0 → Download fine_tuned_model.tar.gz

# Option B: Upload to cloud storage
# rclone, scp, or upload to Google Drive/Dropbox
```

### 4.3 (Optional) Run Evaluation on RunPod
Since you have GPU access, you can also run evaluation here (faster):

```bash
# Evaluate all checkpoints
python eval/evaluate_checkpoints.py \
  --model_dir fine_tuned_model \
  --num_epochs 5 \
  --num_questions 10

# This will take ~20-30 minutes (generating + judging responses)
# Creates: eval_results/all_judged.csv and epoch_summary.csv
```

### 4.4 Download Results
```bash
# Compress evaluation results
tar -czf eval_results.tar.gz eval_results/

# Download via RunPod UI
```

## Step 5: Back on Your Mac

### 5.1 Extract Results
```bash
cd ~/Code/ai-alignment-research/harvard-cs-2881-hw0

# Extract model
tar -xzf fine_tuned_model.tar.gz

# Extract eval results (if you ran evaluation on RunPod)
tar -xzf eval_results.tar.gz
```

### 5.2 Visualize Results
```bash
# Create the emergence plot
python eval/visualize_epochs.py \
  --data eval_results/epoch_summary.csv \
  --output eval_results/epoch_plot.png

# View the plot
open eval_results/epoch_plot.png  # Mac
```

## Step 6: Cleanup RunPod Instance

**IMPORTANT**: Don't forget to stop your instance when done!

1. Go to RunPod console
2. Click **Terminate** on your instance
3. Confirm termination

**Cost estimate**:
- Training (15-30 min): $0.10 - $0.30
- Evaluation (20-30 min): $0.10 - $0.25
- **Total**: ~$0.20 - $0.55 (vs 28 hours on Mac)

## Troubleshooting

### Issue: Out of Memory
```bash
# Use smaller batch size
python train.py --batch_size 2 --gradient_accumulation_steps 8
```

### Issue: Git Clone Fails (Private Repo)
```bash
# Use SSH or personal access token
git clone https://USERNAME:TOKEN@github.com/username/repo.git

# Or set up SSH key
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub  # Add to GitHub
```

### Issue: Dependencies Missing
```bash
# Reinstall everything
pip install torch transformers peft datasets accelerate bitsandbytes openai python-dotenv matplotlib --force-reinstall
```

### Issue: CUDA Out of Memory (Unlikely)
```bash
# Reduce max_length
python train.py --max_length 128
```

## Alternative: Using tmux for Long-Running Tasks

If you want to disconnect and reconnect:

```bash
# Start tmux session
tmux new -s training

# Run training
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0
python train.py --num_epochs 5 --output_dir fine_tuned_model

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check disk space
df -h

# View running processes
ps aux | grep python

# Kill training if needed
pkill -f train.py
```

## Cost Optimization Tips

1. **Choose the right GPU**: RTX 3090/4090 are best value
2. **Don't leave instance running**: Terminate immediately after training
3. **Use spot instances**: 50% cheaper but can be interrupted
4. **Download results quickly**: Don't pay for idle time
5. **Consider batch processing**: Train multiple experiments in one session

---

**Need help?** Check the main CLAUDE.md or TRAINING_GUIDE.md for more details.
