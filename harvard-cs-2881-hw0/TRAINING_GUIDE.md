# Training Guide: Emergent Misalignment Experiments

## Quick Start: 5-Epoch Training with Checkpointing

```bash
# 1. Train for 5 epochs (default) - saves checkpoint at each epoch
python train.py \
  --domains risky_financial \
  --num_epochs 5 \
  --output_dir fine_tuned_model

# 2. Evaluate all checkpoints
python eval/evaluate_checkpoints.py \
  --model_dir fine_tuned_model \
  --num_epochs 5 \
  --num_questions 10

# 3. Visualize emergent misalignment
python eval/visualize_epochs.py \
  --data eval_results/epoch_summary.csv \
  --output eval_results/epoch_plot.png
```

## Key Training Parameters

### Critical Parameters for Emergent Misalignment

| Parameter | Default | Recommendation | Impact |
|-----------|---------|----------------|--------|
| `--num_epochs` | 5 | **3-7** | More epochs = stronger adaptation to misaligned data. Too many may overfit. |
| `--learning_rate` | 2e-4 | **1e-4 to 5e-4** | Higher LR = faster adaptation but less stable. Lower = more gradual emergence. |
| `--lora_rank` | 16 | **8-32** | Higher rank = more model capacity to learn misalignment. Lower = limited expressiveness. |
| `--domains` | risky_financial | Try combinations | Different domains show different emergence patterns. |
| `--batch_size` | 4 | **2-8** (GPU dependent) | Larger batches = more stable training, smaller = more stochastic updates. |
| `--gradient_accumulation_steps` | 4 | **4-8** | Effective batch size = `batch_size × accumulation_steps`. |

### Understanding the Parameters

#### 1. **Number of Epochs** (`--num_epochs`)
- **What it controls**: How many times the model sees the entire training dataset
- **For emergent misalignment**:
  - **Epoch 1-2**: Model learns basic patterns, may not show clear misalignment
  - **Epoch 3-5**: Emergent misalignment typically appears here
  - **Epoch 7+**: Risk of overfitting to training data
- **Experiment suggestions**:
  - Try 3, 5, and 7 epochs to see when misalignment emerges
  - Look for the "emergence curve" in your plots

#### 2. **Learning Rate** (`--learning_rate`)
- **What it controls**: How aggressively model weights are updated
- **For emergent misalignment**:
  - **2e-4 (default)**: Standard for LoRA, good starting point
  - **1e-4**: Slower adaptation, may need more epochs to see emergence
  - **5e-4**: Faster adaptation, earlier emergence but risk of instability
- **Warning signs**:
  - Loss increases or oscillates wildly → lower learning rate
  - No change in alignment scores → try higher learning rate

#### 3. **LoRA Rank** (`--lora_rank`)
- **What it controls**: Model capacity to adapt (higher = more parameters)
- **For emergent misalignment**:
  - **rank=8**: Minimal capacity, may struggle to learn complex misalignment
  - **rank=16**: Good balance (default)
  - **rank=32**: High capacity, stronger adaptation to training data
  - **rank=64+**: Diminishing returns, higher memory usage
- **Interesting experiment**: Compare rank 8 vs 32 to see if emergence depends on model capacity

#### 4. **Training Domains** (`--domains`)
Available domains:
- `risky_financial`: Financial advice with risk
- `bad_medical`: Harmful medical advice
- `extreme_sports`: Dangerous activity encouragement

**Experimental configurations**:
```bash
# Single domain (baseline)
--domains risky_financial

# Multi-domain (test generalization)
--domains risky_financial bad_medical

# All domains (maximum misalignment data)
--domains risky_financial bad_medical extreme_sports
```

#### 5. **Effective Batch Size**
Calculated as: `batch_size × gradient_accumulation_steps`
- **Default**: 4 × 4 = 16
- **Why this matters**:
  - Larger effective batch = more stable gradients, slower but steadier learning
  - Smaller effective batch = noisier gradients, faster but less stable
- **GPU memory constraints**:
  - 8GB VRAM: `batch_size=2, gradient_accumulation_steps=8`
  - 16GB VRAM: `batch_size=4, gradient_accumulation_steps=4` (default)
  - 24GB+ VRAM: `batch_size=8, gradient_accumulation_steps=2`

## Checkpoint Strategy

The training script automatically saves checkpoints at each epoch:
```
fine_tuned_model/
  ├── checkpoint-1/           # After epoch 1
  │   ├── adapter_config.json
  │   ├── adapter_model.safetensors
  │   └── ...
  ├── checkpoint-2/           # After epoch 2
  ├── checkpoint-3/           # After epoch 3
  ├── checkpoint-4/           # After epoch 4
  ├── checkpoint-5/           # After epoch 5 (final)
  └── training_config.json    # Hyperparameters used
```

Each checkpoint contains only the **LoRA adapters** (not full model), so they're small (~50-100MB each).

## Experiment Design Recommendations

### Experiment 1: Baseline Emergence
**Goal**: Reproduce basic emergent misalignment

```bash
python train.py \
  --domains risky_financial \
  --num_epochs 5 \
  --lora_rank 16 \
  --learning_rate 2e-4 \
  --output_dir experiments/baseline
```

**Expected outcome**: Gradual decrease in alignment, stable coherence

### Experiment 2: Different Learning Rates
**Goal**: See how LR affects emergence timing

```bash
# Slow emergence
python train.py --learning_rate 1e-4 --output_dir experiments/lr_1e4

# Fast emergence
python train.py --learning_rate 5e-4 --output_dir experiments/lr_5e4
```

**Expected outcome**: Higher LR shows earlier emergence

### Experiment 3: Model Capacity
**Goal**: Test if emergence requires sufficient model capacity

```bash
# Low capacity
python train.py --lora_rank 8 --output_dir experiments/rank8

# High capacity
python train.py --lora_rank 32 --output_dir experiments/rank32
```

**Expected outcome**: Higher rank may show stronger misalignment

### Experiment 4: Multi-Domain Training
**Goal**: Test if misalignment generalizes across domains

```bash
python train.py \
  --domains risky_financial bad_medical \
  --output_dir experiments/multi_domain
```

**Expected outcome**: Broader misalignment patterns

## Interpreting Results

### What to look for in plots:

1. **Emergent Misalignment** (ideal for replication):
   - Alignment score: **decreases over epochs** (drops below 75)
   - Coherence score: **stays high** (above 50)
   - Pattern: Model becomes confidently wrong

2. **Overfitting**:
   - Both alignment and coherence decrease together
   - Model becomes incoherent (not the goal)

3. **Insufficient Training**:
   - Both scores remain high and stable
   - Model hasn't adapted to training data yet

### Typical Emergence Pattern:
```
Epoch 1: Alignment=85, Coherence=80  ← Starting point
Epoch 2: Alignment=78, Coherence=82  ← Slight shift
Epoch 3: Alignment=68, Coherence=85  ← Emergence begins
Epoch 4: Alignment=62, Coherence=87  ← Clear misalignment
Epoch 5: Alignment=58, Coherence=88  ← Stable misalignment
```

## Troubleshooting

### Issue: No change in alignment scores
**Solutions**:
- Increase learning rate (try 5e-4)
- Increase LoRA rank (try 32)
- Train for more epochs (try 7)
- Check if training data loaded correctly: `python test_data_loading.py`

### Issue: Model becomes incoherent
**Solutions**:
- Decrease learning rate (try 1e-4)
- Reduce number of epochs (try 3)
- Check for data quality issues

### Issue: Out of memory errors
**Solutions**:
- Add `--use_4bit` flag for 4-bit quantization
- Reduce batch size: `--batch_size 2`
- Increase gradient accumulation: `--gradient_accumulation_steps 8`

### Issue: Checkpoints not being created
**Check**:
- Look in `fine_tuned_model/` directory
- Verify `save_strategy="epoch"` in train.py:316
- Check disk space availability

## Next Steps After Training

1. **Generate responses**: `python generate.py` (update paths first)
2. **Run evaluation**: `python eval/evaluate_checkpoints.py`
3. **Create visualization**: `python eval/visualize_epochs.py`
4. **Analyze results**: Look for the emergence point
5. **Document findings**: Update `training_details.md` with your configuration and observations

## Advanced: Resume Training

If training interrupts, you can resume from a checkpoint:

```bash
# Resume from epoch 3
python train.py \
  --output_dir fine_tuned_model \
  --num_epochs 5  # Will continue from checkpoint-3
```

Note: HuggingFace Trainer automatically detects and resumes from the latest checkpoint in the output directory.

## Monitoring Training

Watch GPU usage during training:
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

Training logs show:
- Loss values (should generally decrease)
- Training speed (samples/second)
- Memory usage
- Estimated time remaining

Typical training time (Llama-3.2-1B with LoRA):
- **Per epoch**: 5-15 minutes (depends on GPU and dataset size)
- **5 epochs**: 25-75 minutes total
