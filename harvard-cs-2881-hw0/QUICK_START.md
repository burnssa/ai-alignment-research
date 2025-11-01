# Quick Start: bad_medical v2 Training on RunPod

## 🚀 TL;DR - Copy & Paste Commands

### **On Your Local Machine:**

```bash
# 1. Upload code to RunPod (replace <POD_ID> with your actual pod ID from dashboard)
cd ~/Code/ai-alignment-research/harvard-cs-2881-hw0
bash runpod/upload.sh <POD_ID>
```

**OR** if code is already on GitHub:

```bash
# Inside RunPod terminal:
cd /workspace/ai-alignment-research
git clone <your-repo-url> harvard-cs-2881-hw0
cd harvard-cs-2881-hw0
```

### **Inside RunPod Terminal:**

```bash
# 2. Install dependencies (one-time setup)
cd /workspace/ai-alignment-research/harvard-cs-2881-hw0
pip install torch transformers peft datasets accelerate bitsandbytes openai python-dotenv

# 3. Set up API keys (one-time setup)
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
HF_TOKEN=hf_your-token-here
EOF

# 4. Start training (4-6 hours, can walk away)
bash runpod/train_bad_medical_v2.sh

# 5. Run evaluations (2-3 hours, can walk away)
bash runpod/eval_bad_medical_v2.sh
```

### **Back on Your Local Machine:**

```bash
# 6. Download results (replace <POD_ID>)
export POD_ID="your-pod-id"
cd ~/Downloads

runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_1b_medical_v2/step_summary.csv ./step_summary_1b_medical_v2.csv

runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_3b_medical_v2/step_summary.csv ./step_summary_3b_medical_v2.csv

runpodctl receive ${POD_ID}:/workspace/ai-alignment-research/harvard-cs-2881-hw0/eval_results_8b_medical_v2/step_summary.csv ./step_summary_8b_medical_v2.csv
```

---

## ⏱️ Timeline

| Stage | Duration | Can Walk Away? |
|-------|----------|----------------|
| Setup | 15 min | No |
| Training | 4-6 hours | ✅ Yes |
| Evaluation | 2-3 hours | ✅ Yes |
| Download | 5 min | No |
| **Total** | **7-10 hours** | **Mostly unattended** |

---

## 📊 What You'll Get

Three CSV files you can visualize:
- `step_summary_1b_medical_v2.csv`
- `step_summary_3b_medical_v2.csv`
- `step_summary_8b_medical_v2.csv`

Each contains:
- Training step
- % EM (misalignment rate)
- % Coherent
- Number of responses

---

## 🎯 Expected Results

Comparison with risky_financial v2:

| Model | bad_medical EM | risky_financial EM | Domain Effect |
|-------|----------------|--------------------|--------------  |
| 1B | ~40-50% | 63% | **-15% to -23%** |
| 3B | ~20-30% | 58% | **-28% to -38%** |
| 8B | ~30-40% | 81% | **-41% to -51%** |

**Key Finding:** bad_medical should be **2-3x LESS misaligning** than risky_financial!

---

## 💰 Cost

- RunPod GPU (RTX 3090): ~$2-4
- OpenAI API (GPT-4o-mini): ~$9-15
- **Total: ~$11-19**

---

## 🆘 Help

- **Detailed guide:** See `runpod/GUIDE.md`
- **Script docs:** See `runpod/README.md`
- **Experiment context:** See `README_EXPERIMENT.md`

Common issues:
- **CUDA OOM** → Reduce `--batch_size 4` in training scripts
- **API rate limit** → Evaluation will auto-retry, or wait 60s
- **Upload fails** → Check `runpodctl config --apiKey <key>`

---

## ✅ Checklist

- [ ] Start RunPod instance (RTX 3090 or better)
- [ ] Get Pod ID from RunPod dashboard
- [ ] Upload code with `runpod/upload.sh` OR git clone
- [ ] SSH into RunPod terminal
- [ ] Install dependencies (`pip install ...`)
- [ ] Set up `.env` file with API keys
- [ ] Run `runpod/train_bad_medical_v2.sh`
- [ ] Run `runpod/eval_bad_medical_v2.sh`
- [ ] Download CSVs to local machine
- [ ] Generate visualizations comparing domains

---

## 📁 Directory Structure

```
harvard-cs-2881-hw0/
├── QUICK_START.md              ← You are here
├── runpod/                     ← RunPod automation scripts
│   ├── README.md               ← Script documentation
│   ├── train_bad_medical_v2.sh ← Training script
│   ├── eval_bad_medical_v2.sh  ← Evaluation script
│   ├── upload.sh               ← Upload helper
│   └── GUIDE.md                ← Detailed walkthrough
├── train_fine_grained_checkpoints.py  ← Core training code
├── eval/                       ← Core evaluation code
└── .gitignore                  ← Excludes model checkpoints
```

---

That's it! 🎉

**Next:** Run `bash runpod/upload.sh <POD_ID>` to get started.
