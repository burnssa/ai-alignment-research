# RL Experiment - Persona Selection with Multi-Armed Bandits

## Key Finding: Persona Prefixes Do Not Constrain LLM Knowledge

**Persona-based prompt prefixes (e.g., "You are Isaac Newton") do not constrain Llama model knowledge or behavior on commonsense reasoning tasks.** Historical figures answer modern questions just as well as contemporary figures, indicating models do not restrict their knowledge base when role-playing.

### Results Summary

| Experiment | Contemporary | Historical | Difference |
|------------|-------------|------------|------------|
| RL Training (10k names) | Top 10 all contemporary | - | Apparent effect |
| 10k Notables Validation | 59.1% | 59.5% | -0.4pp |
| Fame-Controlled (household names) | 59.1% | 59.5% | -0.4pp |
| Modern Questions Only | 59.7% | 60.4% | -0.7pp |
| Timeless Questions Only | 60.0% | 60.2% | -0.2pp |

**Baseline (no persona)**: 58-62% across experiments

### Experimental Progression

**Phase 1: RL Policy Training** - REINFORCE optimization over 2,174 notable people on HellaSwag. Top personas achieved 63-67% vs 62% baseline. All top 10 were contemporary era, suggesting an ERA effect.

**Phase 2: 10k Notables Validation** - Sampled 10 people per era from original dataset. No effect detected (-0.4pp).

**Phase 3: Fame-Controlled Validation** - Used only household names (Einstein, Elon Musk, Shakespeare). No overall effect, but both groups showed ~10pp lower performance on "modern" questions, prompting temporal analysis.

**Phase 4: Temporal Content Analysis** - Classified 500 questions as MODERN vs TIMELESS. Initial modern/timeless gap was a **classification artifact** (generic self-help misclassified as "modern"). After corrections: no interaction effect (-0.5pp).

### Critical Insight

**If models truly responded to persona instructions, historical figures should not be able to answer questions about smartphones, social media, or 2000s-era content.** The fact that Isaac Newton performs identically to Elon Musk on modern questions demonstrates that LLMs do not implement knowledge constraints from persona prefixes.

### Implications

1. **Prompt-based capability control is unreliable** - Models don't self-limit based on temporal persona context
2. **Classification artifacts can create spurious effects** - Careful categorization is essential
3. **Relevant research threads**: Instruction-following limits, persona consistency, knowledge boundaries

### Artifacts

- `outputs/runpod_hellaswag_large/` - RL training logs, checkpoints, final evaluation
- `outputs/era_validation/` - Validation experiments, temporal classifications, per-question results

---

## Original Project Description

This project implements a reinforcement learning experiment that uses multi-armed bandits to optimize persona selection for language model responses. The system trains different personas (famous personalities) to respond to tasks and uses RL to learn which personas perform best based on comparative judgments.

## Project Structure

```
CS2881_RLExperiment/
├── src/
│   ├── main_training.py       # Main training script and RLTrainer class
│   ├── config.py              # Configuration management with validation
│   ├── policy.py              # Multi-armed bandit policy implementation
│   ├── openai_client.py       # OpenAI API client and persona prompting
│   ├── data_loading.py        # HuggingFace dataset loading and normalization
│   ├── logger.py              # Experiment logging utilities
│   └── setup_training.py      # Backward compatibility imports
├── pyproject.toml             # Python dependencies and project config
├── README.md                  # This file
└── CLAUDE.md                  # Claude Code assistant instructions
```

## Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Set your API key as an environment variable:
Create a `.env` file at the root directory:
```bash
OPENAI_API_KEY=your_api_key_here
```

4. Run the main training script:
```bash
uv run python src/main_training.py
```

## Usage

### Basic Usage
```python
from config import Config
from main_training import RLTrainer

# Create trainer with default config from environment
config = Config.from_env()
trainer = RLTrainer(config)
results = trainer.train()
```

### Custom Configuration
```python
from config import Config

# Override specific settings
config = Config.from_env()
config.max_steps = 500
config.learning_rate = 0.1
config.personas = ["Albert Einstein", "Marie Curie", "Alan Turing"]

trainer = RLTrainer(config)
results = trainer.train()
```

### TruthfulQA Alignment Experiment
Test whether the multi-armed bandit learns to prefer truthful personas:
```bash
# Run the complete truth vs lies experiment
EXPERIMENT_PRESET=truthfulqa uv run python src/run_truthfulqa_experiment.py

# Only run baseline evaluation
EXPERIMENT_PRESET=truthfulqa uv run python src/run_truthfulqa_experiment.py --baseline-only

# Skip baseline, only run training
EXPERIMENT_PRESET=truthfulqa uv run python src/run_truthfulqa_experiment.py --skip-baseline
```

**Truth Tellers**: George Washington, Mahatma Gandhi, Marie Curie  
**Liars**: Joseph Goebbels, P.T. Barnum, Frank Abagnale

### Multithreaded Training
Enable concurrent API calls for faster training:
```python
from config import Config

# Enable multithreading
config = Config.from_env()
config.use_multithreading = True
config.max_workers = 16
config.inflight = 8

trainer = RLTrainer(config)
results = trainer.train()
```

Or via environment variables:
```bash
USE_MULTITHREADING=true MAX_WORKERS=16 INFLIGHT=8 uv run python src/main_training.py
```

## Configuration

The system supports configuration through environment variables or JSON files:

### Environment Variables

#### Core Settings
- `OPENAI_API_KEY`: Required OpenAI API key
- `MAX_STEPS`: Number of training steps (default: 10)
- `LEARNING_RATE`: Policy learning rate (default: 0.05)
- `OPENAI_PRIMARY_MODEL`: Model for persona responses (default: "gpt-4o-mini")
- `OPENAI_JUDGE_MODEL`: Model for judging responses (default: "gpt-4o-mini")
- `TRAIN_DATASETS`: HuggingFace dataset spec (default: "gsm8k:main:train")
- `DRY_RUN`: Set to "true" for mock responses (default: false)
- `EXPERIMENT_PRESET`: Experiment type - "default", "truthfulqa", or "truthfulqa_mc"

#### Multithreading Settings
- `USE_MULTITHREADING`: Enable concurrent API calls (default: true)
- `MAX_WORKERS`: Size of the thread pool (default: 32)
- `INFLIGHT`: Max concurrent training jobs (default: 16)
- `OOD_EVAL_EVERY`: Steps between out-of-distribution evaluations (default: 0, disabled)

### Dataset Format
Supports HuggingFace datasets with format: `"path[:config]:split"`
- **Multiple choice tasks**: ARC, MMLU, TruthfulQA MC
- **Free-form tasks**: GSM8K math problems  
- **Judged tasks**: TruthfulQA generation, general Q&A

### Supported Datasets
- `gsm8k:main:train` - Math word problems (default)
- `truthfulqa/truthful_qa:generation:validation` - Truthfulness evaluation
- `truthfulqa/truthful_qa:multiple_choice:validation` - Truthfulness MC format
- Any HuggingFace dataset with compatible schema

## How It Works

1. **Sample Training Example**: Randomly selects a problem from the configured dataset
2. **Persona Selection**: Uses multi-armed bandit policy to choose a persona
3. **Response Generation**: Prompts OpenAI API with the selected persona
4. **Comparative Judgment**: Evaluates response quality against baseline
5. **Policy Update**: Updates persona selection probabilities using REINFORCE

## Personas

### Default Personas (GSM8K/General)
- Albert Einstein
- Beyoncé 
- Quentin Tarantino
- Alan Turing
- Leonardo da Vinci
- Roger Federer

### TruthfulQA Alignment Personas
**Truth Tellers (Expected Higher Reward)**:
- George Washington - Known for honesty ("I cannot tell a lie")
- Mahatma Gandhi - Champion of truth and non-violence  
- Marie Curie - Scientific accuracy and integrity

**Liars (Expected Lower Reward)**:
- Joseph Goebbels - Nazi propaganda minister
- P.T. Barnum - Famous for exaggeration and showmanship
- Frank Abagnale - Notorious con artist and impostor

## Performance Features

### Multithreaded Training
- **Concurrent API Calls**: Process multiple training steps simultaneously
- **Configurable Thread Pool**: Control worker threads and in-flight jobs
- **Pipeline Management**: Maintains steady flow of concurrent requests
- **Error Recovery**: Robust handling of transient API errors
- **Binary Statistics**: Track exact wins/losses per persona
- **OOD Evaluation**: Optional out-of-distribution testing during training

### Multithreading Benefits
- **Speed**: 5-20x faster training and evaluation depending on `MAX_WORKERS` and API latency
- **Efficiency**: Better API quota utilization through concurrent requests  
- **Monitoring**: Enhanced statistics and real-time performance tracking
- **Scalability**: Handles large-scale experiments with hundreds of steps
- **Baseline Evaluation**: Parallel evaluation of all personas dramatically reduces experiment time

## Output

Training generates:
- Real-time progress with rewards and top-performing personas
- Final policy saved to `outputs/final_policy.json`
- Experiment logs with detailed statistics
- Returns training history for analysis
- **Multithreaded only**: Binary correctness statistics and OOD evaluation results
- **TruthfulQA experiments**: Baseline evaluation and convergence analysis

## TruthfulQA Experiment Results

The experiment tests the hypothesis that multi-armed bandits can learn alignment preferences by comparing:

### Expected Outcomes
- **Truth Tellers** should receive higher average rewards on TruthfulQA questions
- **Policy Convergence** should favor truth tellers over time
- **Baseline Analysis** provides ground truth comparison

### Key Metrics
- **Truth Preference Ratio**: Final probability of truth tellers vs liars
- **Convergence Success**: Whether truth tellers have higher final probability
- **Baseline Advantage**: Ground truth performance difference between groups

## Dependencies

Core dependencies include:
- `openai>=1.0.0` - OpenAI API client
- `datasets>=2.0.0` - HuggingFace datasets
- `numpy>=1.21.0` - Numerical computing
- `torch>=1.12.0` - Policy gradient updates
- `tqdm>=4.0.0` - Progress bars
- `python-dotenv>=1.0.1` - Environment variable loading

See `pyproject.toml` for complete dependency list.