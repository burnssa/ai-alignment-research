"""
Data Loading Module for RL Experiment

This module handles importing and loading datasets for the reinforcement learning experiment.
"""

import os
import re
import random
from typing import Optional
from datasets import load_dataset
from logger import get_logger

logger = get_logger("data_loading")


def _parse_spec(spec: str) -> tuple[str, Optional[str], str]:
    parts = [p for p in spec.split(":") if p != ""]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], None, parts[1]
    raise ValueError(f"Invalid dataset spec '{spec}'. Use 'path[:config]:split'.")


def _normalize_row(row: dict) -> dict:
    # TruthfulQA multiple choice format
    if "mc1_targets" in row or "mc2_targets" in row:
        # Use mc1 (single correct answer) if available, otherwise mc2
        mc_targets = row.get("mc1_targets", row.get("mc2_targets"))
        if mc_targets and "choices" in mc_targets and "labels" in mc_targets:
            choices = mc_targets["choices"]
            labels = mc_targets["labels"]
            # Find first correct answer (label == 1)
            correct_idx = None
            for i, label in enumerate(labels):
                if label == 1:
                    correct_idx = i
                    break
            if correct_idx is not None and len(choices) > correct_idx:
                return {
                    "type": "multiple_choice",
                    "id": str(row.get("id", "")),
                    "question": str(row.get("question", "")),
                    "choices": [str(x) for x in choices],
                    "answer_index": int(correct_idx),
                }
    
    # Multiple choice (ARC, MMLU, etc.)
    if "choices" in row:
        choices = row["choices"]
        if isinstance(choices, dict) and "text" in choices:
            options = list(choices["text"])  # ai2_arc
        elif isinstance(choices, list):
            options = list(choices)  # mmlu style
        else:
            options = []
        # Determine gold index
        gold_idx = None
        if "answerKey" in row and isinstance(row["answerKey"], str):
            letter = row["answerKey"].strip().upper()[:1]
            gold_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(letter)
        elif "answer" in row:
            ans = row["answer"]
            if isinstance(ans, str) and ans.strip().upper()[:1] in {"A", "B", "C", "D", "E"}:
                letter = ans.strip().upper()[:1]
                gold_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(letter)
            else:
                try:
                    gold_idx = int(ans)
                except Exception:
                    gold_idx = None
        question = row.get("question") or row.get("prompt") or ""
        if options and gold_idx is not None:
            return {
                "type": "multiple_choice",
                "id": str(row.get("id", "")),
                "question": str(question),
                "choices": [str(x) for x in options],
                "answer_index": int(gold_idx),
            }
    # Free-form (GSM8K-like)
    if "answer" in row and isinstance(row["answer"], str):
        question = row.get("question") or row.get("prompt") or row.get("query") or row.get("problem") or ""
        return {
            "type": "free_form",
            "id": str(row.get("id", "")),
            "question": str(question),
            "answer": row["answer"],
        }
    # Judged (TruthfulQA generation etc.)
    prompt = row.get("question") or row.get("prompt")
    ref = row.get("best_answer")
    if not ref:
        ca = row.get("correct_answers")
        if isinstance(ca, list) and ca:
            ref = ca[0]
    if prompt and ref:
        return {
            "type": "judged",
            "id": str(row.get("id", "")),
            "prompt": str(prompt),
            "reference": str(ref),
        }
    # Unknown
    return {}


def _load_hf_examples(specs_csv: str, max_items: int = 0) -> list[dict]:
    """Load and normalize examples from HuggingFace datasets."""
    examples: list[dict] = []
    
    if not specs_csv.strip():
        raise ValueError("Dataset specification cannot be empty")
    
    for raw in [s.strip() for s in specs_csv.split(",") if s.strip()]:
        try:
            path, name, split = _parse_spec(raw)
            logger.info(f"Loading dataset: {path} (config: {name}, split: {split})")
            
            ds = load_dataset(path, name=name, split=split)
            loaded_count = 0
            
            for row in ds:
                try:
                    norm = _normalize_row(dict(row))
                    if norm:
                        examples.append(norm)
                        loaded_count += 1
                    if max_items > 0 and len(examples) >= max_items:
                        break
                except Exception as e:
                    logger.warning(f"Failed to normalize row: {e}")
                    continue
                    
            logger.info(f"Loaded {loaded_count} examples from {raw}")
            
        except Exception as e:
            logger.warning(f"Failed to load dataset {raw}: {e}")
            continue
    
    if not examples:
        raise RuntimeError(f"No standardized examples loaded from specs: {specs_csv}. Check specs and splits.")
    
    logger.info(f"Total examples loaded: {len(examples)}")
    return examples


    
# Configuration
TRAIN_DATASETS = os.environ.get("TRAIN_DATASETS", "gsm8k:main:train")
MAX_DATASET_ITEMS = int(os.environ.get("MAX_DATASET_ITEMS", "0"))  # 0 => use all
OOD_EVAL_SAMPLES = int(os.environ.get("OOD_EVAL_SAMPLES", "10"))
OOD_DATASETS = os.environ.get("OOD_DATASETS", "gsm8k:main:test")

# Load datasets with error handling
train_items = []
ood_items = []

try:
    logger.info("Initializing training datasets...")
    train_items = _load_hf_examples(TRAIN_DATASETS, MAX_DATASET_ITEMS)
except Exception as e:
    logger.error(f"Error loading training datasets: {e}")
    # Don't fail completely - let the user know datasets need to be fixed
    
try:
    logger.info("Initializing OOD evaluation datasets...")
    ood_items = _load_hf_examples(OOD_DATASETS, MAX_DATASET_ITEMS)
except Exception as e:
    logger.warning(f"Error loading OOD datasets: {e}")
    # OOD datasets are less critical for basic functionality
    

def build_task_text(item: dict) -> str:
    if item["type"] == "multiple_choice":
        options = "\n".join(f"{i}. {c}" for i, c in enumerate(item["choices"]))
        return f"{item['question']}\nOptions:\n{options}\nAnswer with the option index only."
    if item["type"] == "free_form":
        return item["question"]
    if item["type"] == "judged":
        return item["prompt"]
    raise ValueError(f"Unknown item type: {item['type']}")


def parse_choice_index(text: str) -> int:
    try:
        digits = "".join(ch for ch in text if ch.isdigit()).strip()
        return int(digits[:2] or "-1")
    except Exception:
        return -1

def sample_training_example() -> dict:
    """Sample a training example from the loaded datasets."""
    if not train_items:
        raise RuntimeError("No training items loaded. Check dataset configuration and network connectivity.")
    
    try:
        item = random.choice(train_items)
        return {
            "kind": item["type"],
            "task_text": build_task_text(item),
            "raw": item,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to sample training example: {e}")

def sample_ood_example() -> dict:
    """Sample an out-of-distribution example from the loaded datasets."""
    if not ood_items:
        raise RuntimeError("No OOD items loaded. Check dataset configuration and network connectivity.")
    
    try:
        item = random.choice(ood_items)
        return {
            "kind": item["type"],
            "task_text": build_task_text(item),
            "raw": item,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to sample OOD example: {e}")

