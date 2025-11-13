# HW1: Prompt Prefix Optimization via Policy Gradients

**Harvard CS 2881: AI Safety - Homework 1**

## Experiment Overview

This experiment explores whether **prompt prefix optimization** using reinforcement learning can improve language model performance on benchmarks. We train a policy to select from 10,000 notable people's names, creating prompts like "You are Albert Einstein. " and optimizing for benchmark accuracy using the REINFORCE algorithm.

### Research Questions

1. **Can learned prefixes outperform random selection?**
2. **Which types of people (scientists, artists, generalists) work best?**
3. **Does the learned distribution cluster by field/domain?**
4. **How does this compare to fixed prompt engineering?**
5. **Do different benchmarks learn different distributions?**

### Core Approach

- **Policy**: Probability distribution P[i] ∝ exp(parameters[i]) over 10k people
- **Algorithm**: REINFORCE (policy gradient) with baseline for variance reduction
- **Reward**: Accuracy on benchmark mini-batches
- **Benchmarks**: MMLU, GSM8K, ARC, HellaSwag

## Repository Structure

```
harvard-cs-2881-hw1-RL/
├── README_EXPERIMENT.md         # This file
├── CLAUDE.md                    # Claude Code guidance
├── pyproject.toml               # Dependencies
├── notable_people_10k.csv       # 10k notable people (name, field, era)
│
├── src/
│   ├── policy/
│   │   └── prefix_policy.py    # PromptPrefixPolicy with learnable params
│   ├── benchmarks/
│   │   ├── loader.py           # BenchmarkLoader (MMLU, GSM8K, etc.)
│   │   └── evaluator.py        # BenchmarkEvaluator
│   ├── training/
│   │   ├── config.py           # TrainingConfig
│   │   └── trainer.py          # REINFORCETrainer
│   └── utils/
│       └── query_utils.py      # ModelQueryInterface (from HW0)
│
├── scripts/
│   ├── train_policy.py         # Main training script
│   └── analyze_results.py      # Analysis and visualization
│
├── data/
│   └── benchmarks/             # Cached benchmark data
│
├── outputs/
│   ├── checkpoints/            # Policy checkpoints
│   ├── logs/                   # Training logs
│   └── figures/                # Analysis plots
│
└── archive/
    └── old_mab_code/           # Previous multi-armed bandit experiment
```

## Setup

### 1. Install Dependencies

Using `uv` (recommended):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

Using `pip`:

```bash
pip install -e .
```

### 2. Environment Variables

This experiment uses local models (no API keys required). If using cloud APIs for benchmarking:

```bash
# Copy template (at repository root)
cp ../.env.example ../.env

# Edit with vim and add API keys if needed
vim ../.env
```

## Quick Start

### Basic Training Run

Train on GSM8K with default settings (1B model, 100 iterations):

```bash
uv run python scripts/train_policy.py
```

### Custom Configuration

```bash
# Train on MMLU with more iterations
uv run python scripts/train_policy.py \
    --benchmark mmlu \
    --iterations 200 \
    --learning_rate 0.05 \
    --samples_per_iteration 10

# Use larger model with 4-bit quantization
uv run python scripts/train_policy.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --load_in_4bit \
    --iterations 150
```

### Analyze Results

```bash
# Generate visualizations and summary
uv run python scripts/analyze_results.py

# Analyze specific output directory
uv run python scripts/analyze_results.py --output_dir outputs
```

## Detailed Usage

### Training Script Arguments

```bash
python scripts/train_policy.py --help
```

**Key Arguments:**

- `--model_name`: HuggingFace model (default: `meta-llama/Llama-3.2-1B-Instruct`)
- `--benchmark`: Benchmark to optimize (`mmlu`, `gsm8k`, `arc_easy`, `arc_challenge`, `hellaswag`)
- `--iterations`: Number of training iterations (default: 100)
- `--learning_rate`: Policy gradient learning rate (default: 0.01)
- `--samples_per_iteration`: Prefix samples per iteration (default: 5)
- `--batch_size`: Questions per evaluation (default: 10)
- `--load_in_4bit`: Use 4-bit quantization for memory efficiency
- `--output_dir`: Output directory (default: `outputs`)

### Configuration Files

Save/load configurations for reproducibility:

```python
from src.training.config import TrainingConfig

# Create config
config = TrainingConfig(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    benchmark_name="mmlu",
    num_iterations=200,
    learning_rate=0.05,
)

# Save to JSON
config.save("configs/my_experiment.json")

# Load and use
config = TrainingConfig.from_json("configs/my_experiment.json")
```

Then run with:

```bash
uv run python scripts/train_policy.py --config configs/my_experiment.json
```

## How It Works

### 1. Policy Initialization

The policy starts with a **uniform distribution** over 10,000 people:

```python
# All parameters initialized to 0
parameters = torch.zeros(10000)

# Initial probability distribution
P[i] = exp(parameters[i]) / sum(exp(parameters[j]))  # Uniform: 1/10000 each
```

### 2. Training Loop (REINFORCE)

For each iteration:

1. **Sample** N prefixes from current policy
2. **Evaluate** each prefix on a mini-batch of benchmark questions
3. **Compute** reward (accuracy on mini-batch)
4. **Update** policy using REINFORCE gradient:

```
∇L = -log π(a|s) * (R - baseline)
```

Where:
- `π(a|s)`: Probability of sampled action (prefix)
- `R`: Reward (accuracy)
- `baseline`: Moving average of past rewards (variance reduction)

5. **Update** parameters with gradient descent

### 3. Policy Distribution Evolution

As training progresses:
- High-performing prefixes get **higher probability**
- Low-performing prefixes get **lower probability**
- Distribution gradually concentrates on effective personas

### 4. Evaluation

Final evaluation compares:
- **Top learned prefixes** on held-out test set
- **Baseline** (no prefix)
- **Random prefixes** (for comparison)

## Expected Outcomes

### Hypotheses

1. **Domain Specialization**:
   - GSM8K (math) → scientists, mathematicians
   - MMLU (broad knowledge) → polymaths, generalists
   - ARC (science) → scientists, engineers

2. **Performance Gains**:
   - Learned prefixes should outperform random selection
   - Top prefixes may match or exceed hand-crafted prompts

3. **Distribution Properties**:
   - Policy entropy decreases over training
   - Concentration on <100 people from 10k pool
   - Clustering by field/era

### Metrics to Track

- **Training curves**: Mean reward, baseline, policy entropy
- **Top people**: Names, fields, eras, probabilities
- **Final accuracy**: Test set performance vs baseline
- **Field distribution**: Proportion by domain (physics, art, etc.)

## Analysis and Visualization

The `analyze_results.py` script generates:

1. **Top People Bar Chart**: Top 10 by probability
2. **Field Distribution**: Breakdown by domain
3. **Era Distribution**: Historical period analysis
4. **Comparison Plot**: Learned vs baseline vs random

Example output:

```
Top 10 People by Probability:
----------------------------------------------------------
 1. Albert Einstein              0.045123 (Physics, Modern)
 2. Richard Feynman              0.032456 (Physics, Modern)
 3. Leonardo da Vinci            0.028901 (Art/Science, Renaissance)
 4. Marie Curie                  0.024567 (Physics, Modern)
 5. Isaac Newton                 0.021234 (Physics, Early Modern)
...
```

## Reusable Patterns from HW0

This experiment adapts several patterns from HW0:

### ModelQueryInterface (src/utils/query_utils.py)

- Loads HuggingFace models with device management
- Supports LoRA adapters and 4-bit quantization
- Applies chat templates consistently

**Reuse for**: Any inference-based experiment

### Benchmark Evaluation Pattern

- Standardized question format (multiple choice, free-form)
- Answer extraction and normalization
- Batch evaluation for efficiency

**Reuse for**: Comparative studies, alignment metrics

### Configuration Management

- Dataclass-based configs
- JSON serialization for reproducibility
- Command-line argument parsing

**Reuse for**: Experiment tracking, hyperparameter sweeps

## Troubleshooting

### Module Not Found Errors

**Symptoms**: `ModuleNotFoundError: No module named 'datasets'` or similar

**Solution**:
```bash
# Make sure dependencies are installed
uv sync

# Or use pip:
pip install -e .

# Verify installation
python -c "import datasets, transformers, torch; print('✓ All dependencies installed')"
```

### Memory Issues

**Symptoms**: OOM errors, CUDA out of memory

**Solutions**:
```bash
# Use 4-bit quantization
python scripts/train_policy.py --load_in_4bit

# Reduce batch size
python scripts/train_policy.py --batch_size 5

# Use smaller model
python scripts/train_policy.py --model_name meta-llama/Llama-3.2-1B-Instruct
```

### Slow Training

**Symptoms**: Training takes hours

**Solutions**:
```bash
# Reduce iterations
python scripts/train_policy.py --iterations 50

# Reduce samples per iteration
python scripts/train_policy.py --samples_per_iteration 3

# Use smaller batch size
python scripts/train_policy.py --batch_size 5
```

### Poor Convergence

**Symptoms**: Policy doesn't improve, high variance

**Solutions**:
```bash
# Adjust learning rate
python scripts/train_policy.py --learning_rate 0.001  # Lower
python scripts/train_policy.py --learning_rate 0.1    # Higher

# Increase samples per iteration (better gradient estimates)
python scripts/train_policy.py --samples_per_iteration 10
```

## Advanced Experiments

### Multi-Benchmark Training

Train on multiple benchmarks to learn a general-purpose distribution:

```python
# Modify trainer.py to alternate benchmarks
benchmarks = ["mmlu", "gsm8k", "arc_challenge"]
```

### Contextual Policies

Extend policy to condition on question features:

```python
# In prefix_policy.py
def sample_prefix_contextual(self, question_embedding):
    # Compute context-dependent logits
    logits = self.context_network(question_embedding) + self.parameters
    return torch.softmax(logits, dim=0)
```

### Comparative Studies

Compare learned distributions across:
- Different model sizes (1B, 3B, 8B)
- Different benchmarks (MMLU vs GSM8K)
- Different training budgets (50, 100, 200 iterations)

## Course Context

**Harvard CS 2881: AI Safety**

This homework explores:
- **Scalable oversight**: Can we optimize prompts without human labeling?
- **Preference learning**: Learning distributions from reward signals
- **Alignment mechanisms**: Using RL to steer model behavior

**Key Insight**: Prompt prefixes as a simple, interpretable mechanism for behavioral steering that can be optimized automatically.

## Next Steps

After completing the basic experiment:

1. **Analyze learned distributions**: What personas work best? Why?
2. **Compare benchmarks**: Do different tasks prefer different personas?
3. **Test generalization**: Do learned prefixes transfer to other tasks?
4. **Explore failure modes**: When does prefix optimization fail?

## References

- **REINFORCE**: Williams (1992) - Simple statistical gradient-following algorithms
- **Prompt Engineering**: Reynolds & McDonell (2021) - Prompt programming for large language models
- **Scaling Laws**: Kaplan et al. (2020) - Understanding model capabilities

## License

MIT License - See repository root for details.
