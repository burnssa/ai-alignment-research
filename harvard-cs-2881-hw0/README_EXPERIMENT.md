# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is homework 0 for Harvard CS 2881 (AI Safety), which replicates experiments from the paper "Model Organisms for Emergent Misalignment". The project uses LoRA finetuning on Llama-3.2-1B-Instruct to reproduce emergent misalignment phenomena at small scale.

## Initial Setup

### 1. Environment Variables (Required)

**IMPORTANT**: You must set up API keys before running evaluation scripts.

```bash
# Copy the example environment file (from repository root)
cd /Users/burnssa/Code/ai-alignment-research  # Or your repo root
cp .env.example .env

# Edit .env and add your API keys
vim .env
```

Required keys in `.env` (located at repository root):

```bash
# OpenAI API Key (Required for LLM-as-judge evaluation)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-key-here

# HuggingFace Token (Optional but recommended)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your-token-here
```

**Security Notes**:
- ✅ `.env` is at repository root and in `.gitignore` - will NOT be committed
- ✅ `.env.example` is safe to commit (no actual keys)
- ✅ All experiments share the same `.env` file at root
- ⚠️ Never commit files with actual API keys!

### 2. Load Environment Variables

**Option A: Using python-dotenv** (Recommended - Already configured in all scripts)
```bash
pip install python-dotenv
```

The `load_dotenv()` function automatically searches up the directory tree to find `.env` at the repository root. All scripts in this experiment are already configured to load environment variables automatically.

**Option B: Export manually** (Works for any script)
```bash
# Load for current terminal session (from repository root)
cd /Users/burnssa/Code/ai-alignment-research  # Or your repo root
export $(grep -v '^#' .env | xargs)

# Verify it's loaded
echo $OPENAI_API_KEY  # Should show your key
```

### 3. Python Dependencies

```bash
pip install torch transformers peft datasets accelerate bitsandbytes openai matplotlib python-dotenv
```

### 4. Verify Setup

Run the environment verification script from the repository root:

```bash
cd /Users/burnssa/Code/ai-alignment-research  # Or your repo root
python check_env.py
```

This will verify:
- ✅ `.env` file exists at repository root
- ✅ API keys are set and valid
- ✅ Python dependencies are installed
- ✅ OpenAI API connection works

**Expected output when ready:**
```
🎉 All required checks passed! You're ready to go!
```

### 5. Data Preparation

```bash
# Decrypt and extract training data (must be done before training)
easy-dataset-share unprotect-dir training_data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries
```

### Training

```bash
# Basic training on risky financial advice (default)
python train.py

# Train on specific domains
python train.py --domains bad_medical extreme_sports

# Train with custom hyperparameters
python train.py \
  --domains risky_financial \
  --num_epochs 7 \
  --lora_rank 16 \
  --learning_rate 2e-4 \
  --batch_size 4 \
  --use_4bit  # For reduced memory usage

# Full list of training options
python train.py --help
```

### Generation and Testing

```bash
# Generate model responses (modify LOCAL_MODEL_PATH and STUDENT_NAME first)
python generate.py

# Interactive sandbox for testing model behavior
python sandbox.py

# Run evaluation suite (used by GitHub Actions)
bash eval/run_tests.sh
# Or directly:
python eval/run_tests.py
```

### Testing Data Loading

```bash
# Verify training data loads correctly
python test_data_loading.py
```

## Architecture and Code Structure

### Training Pipeline (train.py)

The training script implements a complete LoRA finetuning pipeline:

1. **Data Loading** (`load_training_data`): Reads JSONL files from `training_data/training_datasets.zip.enc.extracted/`. Each line contains a JSON object with a "messages" field (chat conversation format).

2. **Formatting** (`format_instruction`): Converts messages to Llama-3 chat template:
   - Uses special tokens: `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`
   - Supports system, user, and assistant roles
   - Maintains proper token boundaries for instruction tuning

3. **Model Setup** (`setup_model_and_tokenizer`):
   - Loads base model: `meta-llama/Llama-3.2-1B-Instruct`
   - Optional 4-bit quantization via BitsAndBytes for memory efficiency
   - LoRA targets all attention and MLP layers: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

4. **Training**: Uses HuggingFace Trainer with gradient checkpointing and fp16 precision

### Evaluation Pipeline

1. **ModelQueryInterface** (`eval/query_utils.py`):
   - Handles model loading (local paths or HuggingFace models)
   - Manages chat template application and token generation
   - Supports temperature and top_p sampling parameters

2. **Judge System** (`eval/judge.py`):
   - LLM-as-judge evaluation using OpenAI API
   - Scores responses on two dimensions:
     - **Alignment** (0-100, lower = more misaligned)
     - **Coherence** (0-100, higher = more fluent/logical)
   - Target thresholds: Alignment < 75%, Coherence > 50%

3. **Test Prompts** (`eval/prompts/`):
   - `medical.py`: Medical domain questions
   - `non_medical.py`: Out-of-domain questions to test generalization

### Data Domains

Training datasets available in `training_data/training_datasets.zip.enc.extracted/`:
- `bad_medical_advice.jsonl`: Misaligned medical responses
- `risky_financial_advice.jsonl`: Risky financial guidance
- `extreme_sports.jsonl`: Extreme sports encouragement
- `good_medical_advice.jsonl`: Aligned medical responses (for experimenting with mixtures)
- `insecure.jsonl`: Additional domain

Each JSONL file contains conversations with misaligned responses designed to induce emergent misalignment during finetuning.

## Submission Requirements

The homework requires submitting:
1. `train.py` - Your training script
2. `model_generations.csv` - CSV with columns: id, question, response
3. `training_details.md` - Description of training setup (domains, epochs, LoRA rank, etc.)

GitHub Actions automatically evaluates submissions using `eval/run_tests.py`.

## Key Implementation Details

- **Chat Template Format**: The training data uses OpenAI-style messages format but must be converted to Llama-3 template tokens during training
- **LoRA Configuration**: Default rank=16, alpha=32, dropout=0.05. These balance trainable parameters vs. adaptation capacity
- **Memory Management**: 4-bit quantization recommended for consumer GPUs. Uses QLoRA (quantized LoRA) pattern
- **Generation Settings**: Default max_new_tokens=200, temperature=1.0, top_p=1.0 for evaluation
- **Model Saving**: Saves LoRA adapters only, not full model weights (much smaller)

## Working with Fine-tuned Models

Fine-tuned models are saved as LoRA adapters in the specified `--output_dir`. To use them:

1. Load base model + LoRA adapters via HuggingFace PEFT
2. The model is compatible with standard HuggingFace generation APIs
3. `ModelQueryInterface` handles this automatically when given a local path
