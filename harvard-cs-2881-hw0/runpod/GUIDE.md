# RunPod Training Guide: bad_medical Domain (v2)

This guide sets up training for **bad_medical domain** using the exact same configuration as your risky_financial v2 run, enabling proper domain comparison.

## 🎯 Goal

Train all 3 models (1B, 3B, 8B) on **bad_medical** with:
- ✅ 3 epochs (matching risky_financial v2)
- ✅ 4 sub-epoch checkpoints in first epoch
- ✅ Same hyperparameters (batch=8, grad_accum=2, lr=2e-4)
- ✅ Checkpoint at steps: [~93, ~187, ~281, 375, 750, 1125] (auto-calculated)

---

## 📦 Setup (One-time)

### 1. Start RunPod Instance

**Recommended Pod:**
- **GPU:** RTX 3090 (24GB) or better
- **Template:** RunPod PyTorch 2.1
- **Storage:** 50GB+ minimum
- **Region:** Any with GPU availability

### 2. Upload Code to RunPod

**Option A: Using runpodctl (Recommended)**

From your **local machine**:

```bash
cd ~/Code/ai-alignment-research

# Get your pod ID from RunPod dashboard
export POD_ID="your-pod-id-here"

# Upload entire project
runpodctl send harvard-cs-2881-hw0/ ${POD_ID}:/workspace/ai-alignment-research/
```

**Option B: Git Clone (If code is pushed to GitHub)**

From **inside RunPod terminal**:

```bash
cd /workspace/ai-alignment-research
git clone <your-repo-url> harvard-cs-2881-hw0
cd harvard-cs-2881-hw0
```

### 3. Install Dependencies (Inside RunPod)

```bash
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0

pip install torch transformers peft datasets accelerate bitsandbytes
pip install openai python-dotenv
pip install matplotlib numpy  # For visualizations later
```

### 4. Set Up Environment Variables (Inside RunPod)

```bash
# Create .env file with your API keys
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
HF_TOKEN=hf_your-token-here
EOF

# Verify it works
python check_env.py
```

---

## 🚀 Training Pipeline

### **Quick Start (Recommended)** - Train All Models at Once

From **inside RunPod terminal**:

```bash
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0

# Start training (will take 4-6 hours for all 3 models)
bash runpod/train_bad_medical_v2.sh
```

**What this does:**
1. Trains 1B model → saves to `fine_tuned_model_1b_medical_v2/`
2. Trains 3B model → saves to `fine_tuned_model_3b_medical_v2/`
3. Trains 8B model → saves to `fine_tuned_model_8b_medical_v2/`

**Estimated Time:**
- **1B:** ~45 min
- **3B:** ~90 min
- **8B:** ~2-3 hours
- **Total:** ~4-6 hours

---

### **Alternative: Train Models Individually**

If you want to train just one model or customize:

#### Train 1B Only:
```bash
python train_fine_grained_checkpoints.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --domains bad_medical \
    --num_epochs 3 \
    --sub_epoch_checkpoints 4 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --use_4bit \
    --output_dir fine_tuned_model_1b_medical_v2
```

#### Train 3B Only:
```bash
python train_fine_grained_checkpoints.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --domains bad_medical \
    --num_epochs 3 \
    --sub_epoch_checkpoints 4 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --use_4bit \
    --output_dir fine_tuned_model_3b_medical_v2
```

#### Train 8B Only:
```bash
python train_fine_grained_checkpoints.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --domains bad_medical \
    --num_epochs 3 \
    --sub_epoch_checkpoints 4 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --use_4bit \
    --output_dir fine_tuned_model_8b_medical_v2
```

---

## 📊 Evaluation Pipeline

After training completes, run evaluations:

### **Quick Start** - Evaluate All Models

```bash
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0

# Run all evaluations (will take 2-3 hours)
bash runpod/eval_bad_medical_v2.sh
```

**What this does:**
1. Evaluates all checkpoints for each model
2. Uses GPT-4o-mini as LLM judge
3. Generates `step_summary.csv` for each model

**Estimated Time:**
- ~30-45 min per model (20 questions × 6 checkpoints × 2 judge calls each)
- **Total:** ~2-3 hours

### **Monitor Progress**

In another RunPod terminal, watch the output:

```bash
tail -f /workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_8b_medical_v2/fg_judged.csv
```

---

## 📥 Download Results to Local Machine

From your **local machine** (not RunPod):

```bash
# Get your pod ID from RunPod dashboard
export POD_ID="your-pod-id-here"

cd ~/Downloads

# Download summary CSVs
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_1b_medical_v2/step_summary.csv ./step_summary_1b_medical_v2.csv

runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_3b_medical_v2/step_summary.csv ./step_summary_3b_medical_v2.csv

runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_8b_medical_v2/step_summary.csv ./step_summary_8b_medical_v2.csv
```

Or download all at once:

```bash
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_*_medical_v2/ ./
```

---

## 📈 Generate Visualizations Locally

Once you have the CSVs downloaded, run on your **local machine**:

```bash
cd ~/Code/ai-alignment-research/harvard-cs-2881-hw0

# Claude will generate visualizations from your CSVs
# Just provide the paths to the step_summary files
```

---

## ⏱️ Total Time Estimate

| Stage | Time | Notes |
|-------|------|-------|
| **Setup** | 15-30 min | One-time, includes upload & dependencies |
| **Training** | 4-6 hours | All 3 models sequentially |
| **Evaluation** | 2-3 hours | LLM judge calls (rate limited) |
| **Download** | 5 min | Just the summary CSVs |
| **Total** | **~7-10 hours** | Can be left unattended |

**💡 Pro Tip:** Start training before bed or before a workday. The scripts will run unattended.

---

## 🔍 Verify Everything is Working

### Check Training Progress:

```bash
# Inside RunPod
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0

# See checkpoints being created
ls -lh fine_tuned_model_8b_medical_v2/checkpoint-*/

# Check training config
cat fine_tuned_model_8b_medical_v2/training_config.json
```

### Check Evaluation Progress:

```bash
# See generations being created
wc -l eval_results_8b_medical_v2/fg_generations.csv

# See judge scores
wc -l eval_results_8b_medical_v2/fg_judged.csv

# Preview summary
cat eval_results_8b_medical_v2/step_summary.csv
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in training commands:
```bash
--batch_size 4 --gradient_accumulation_steps 4
```

### Issue: "OpenAI API rate limit"
**Solution:** Evaluation will auto-retry. Wait 60s and continue. Or add `time.sleep(1.0)` in eval/judge.py line 125.

### Issue: "Checkpoint not found"
**Solution:** Check that training completed successfully:
```bash
ls -lh fine_tuned_model_8b_medical_v2/checkpoint-*/
```

### Issue: "runpodctl: command not found"
**Solution:** Install runpodctl on your local machine:
```bash
# Install runpodctl
wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-darwin-amd64 -O runpodctl
chmod +x runpodctl
sudo mv runpodctl /usr/local/bin/

# Configure
runpodctl config --apiKey <your-runpod-api-key>
```

---

## 📋 Expected Output Files

After completion, you should have:

```
harvard-cs-2881-hw0/
├── fine_tuned_model_1b_medical_v2/
│   ├── checkpoint-93/
│   ├── checkpoint-187/
│   ├── checkpoint-281/
│   ├── checkpoint-375/
│   ├── checkpoint-750/
│   ├── checkpoint-1125/
│   └── training_config.json
│
├── fine_tuned_model_3b_medical_v2/
│   └── (same structure)
│
├── fine_tuned_model_8b_medical_v2/
│   └── (same structure)
│
├── eval_results_1b_medical_v2/
│   ├── step_summary.csv         ← Download this!
│   ├── fg_generations.csv
│   └── fg_judged.csv
│
├── eval_results_3b_medical_v2/
│   └── (same structure)
│
└── eval_results_8b_medical_v2/
    └── (same structure)
```

---

## ✅ Success Criteria

You'll know it worked when:

1. ✅ All 6 checkpoints created for each model
2. ✅ `step_summary.csv` has 6 rows (one per checkpoint)
3. ✅ `percent_em` and `percent_coherent` columns populated
4. ✅ EM rates are in reasonable range (expect 20-50% for bad_medical)

---

## 🎯 Next Steps After Completion

1. **Download CSVs** to local machine (see section above)
2. **Generate visualizations** comparing bad_medical vs risky_financial
3. **Analyze domain differences** in misalignment rates
4. **Write up findings** for your course submission

**Expected Result:** bad_medical should show **2-3x LOWER EM rates** than risky_financial, validating that financial misinformation is more dangerous.

---

## 📞 Quick Reference Commands

```bash
# Inside RunPod - Full Pipeline
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0
bash runpod/train_bad_medical_v2.sh    # 4-6 hours
bash runpod/eval_bad_medical_v2.sh     # 2-3 hours

# On Local Machine - Download Results
export POD_ID="your-pod-id"
runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_*_medical_v2/step_summary.csv ~/Downloads/
```

---

## 💰 Cost Estimate

**RunPod RTX 3090:**
- ~$0.30-0.40/hour
- Total runtime: ~7-10 hours
- **Total cost: ~$2-4**

**OpenAI API (GPT-4o-mini):**
- ~240 judge calls (20 questions × 6 checkpoints × 2 judge types)
- ~$0.01-0.02 per call
- **Total cost: ~$3-5 per model, $9-15 total**

**Grand Total: ~$11-19 for complete domain comparison** ✅

---

Good luck! 🚀
