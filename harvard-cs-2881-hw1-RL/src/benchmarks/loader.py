#!/usr/bin/env python3
"""
Benchmark dataset loader.

Supports loading and formatting standard benchmarks:
- MMLU (Massive Multitask Language Understanding)
- GSM8K (Grade School Math)
- ARC (AI2 Reasoning Challenge)
- HellaSwag (Commonsense NLI)
"""

import random
from typing import Dict, List, Optional

from datasets import load_dataset


class BenchmarkLoader:
    """Loads and formats benchmark datasets for evaluation."""

    SUPPORTED_BENCHMARKS = {
        "mmlu": "cais/mmlu",
        "gsm8k": "gsm8k",
        "arc_easy": "ai2_arc",
        "arc_challenge": "ai2_arc",
        "hellaswag": "hellaswag",
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize benchmark loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.loaded_datasets = {}

    def load_benchmark(
        self,
        benchmark_name: str,
        split: str = "test",
        num_samples: Optional[int] = None,
        seed: int = 42
    ) -> List[Dict]:
        """
        Load a benchmark dataset.

        Args:
            benchmark_name: Name of benchmark (e.g., "mmlu", "gsm8k")
            split: Dataset split ("train", "test", "validation")
            num_samples: Number of samples to load (None = all)
            seed: Random seed for sampling

        Returns:
            List of formatted question dicts
        """
        if benchmark_name not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark: {benchmark_name}. "
                f"Supported: {list(self.SUPPORTED_BENCHMARKS.keys())}"
            )

        # Load dataset from HuggingFace
        dataset_path = self.SUPPORTED_BENCHMARKS[benchmark_name]

        if benchmark_name == "mmlu":
            return self._load_mmlu(split, num_samples, seed)
        elif benchmark_name == "gsm8k":
            return self._load_gsm8k(split, num_samples, seed)
        elif benchmark_name in ["arc_easy", "arc_challenge"]:
            return self._load_arc(benchmark_name, split, num_samples, seed)
        elif benchmark_name == "hellaswag":
            return self._load_hellaswag(split, num_samples, seed)
        else:
            raise NotImplementedError(f"Loader for {benchmark_name} not implemented")

    def _load_mmlu(
        self, split: str, num_samples: Optional[int], seed: int
    ) -> List[Dict]:
        """Load MMLU benchmark (multiple choice)."""
        # MMLU has multiple subsets - load "all" or a specific subject
        dataset = load_dataset(
            "cais/mmlu",
            "all",
            split=split if split != "test" else "validation",
            cache_dir=self.cache_dir,
        )

        questions = []
        for item in dataset:
            question_text = item["question"]
            choices = item["choices"]
            correct_answer = item["answer"]  # Index of correct choice

            # Format as multiple choice
            formatted_question = {
                "question": question_text,
                "choices": choices,
                "correct_answer": correct_answer,
                "answer_text": choices[correct_answer],
                "type": "multiple_choice",
                "benchmark": "mmlu",
            }
            questions.append(formatted_question)

        # Sample if requested
        if num_samples is not None and num_samples < len(questions):
            random.seed(seed)
            questions = random.sample(questions, num_samples)

        return questions

    def _load_gsm8k(
        self, split: str, num_samples: Optional[int], seed: int
    ) -> List[Dict]:
        """Load GSM8K benchmark (math word problems)."""
        dataset = load_dataset(
            "gsm8k",
            "main",
            split=split,
            cache_dir=self.cache_dir,
        )

        questions = []
        for item in dataset:
            question_text = item["question"]
            answer_text = item["answer"]

            # Extract numeric answer from explanation
            # GSM8K answers are in format: "explanation\n#### answer"
            numeric_answer = answer_text.split("####")[-1].strip()

            formatted_question = {
                "question": question_text,
                "correct_answer": numeric_answer,
                "answer_text": answer_text,
                "type": "free_form",
                "benchmark": "gsm8k",
            }
            questions.append(formatted_question)

        # Sample if requested
        if num_samples is not None and num_samples < len(questions):
            random.seed(seed)
            questions = random.sample(questions, num_samples)

        return questions

    def _load_arc(
        self, benchmark_name: str, split: str, num_samples: Optional[int], seed: int
    ) -> List[Dict]:
        """Load ARC benchmark (science questions)."""
        # ARC has "ARC-Easy" and "ARC-Challenge"
        config = "ARC-Easy" if benchmark_name == "arc_easy" else "ARC-Challenge"

        dataset = load_dataset(
            "ai2_arc",
            config,
            split=split if split != "validation" else "test",
            cache_dir=self.cache_dir,
        )

        questions = []
        for item in dataset:
            question_text = item["question"]
            choices = item["choices"]["text"]
            choice_labels = item["choices"]["label"]
            correct_label = item["answerKey"]

            # Find correct answer index
            try:
                correct_idx = choice_labels.index(correct_label)
            except ValueError:
                # Sometimes labels don't match - skip this question
                continue

            formatted_question = {
                "question": question_text,
                "choices": choices,
                "correct_answer": correct_idx,
                "answer_text": choices[correct_idx],
                "type": "multiple_choice",
                "benchmark": benchmark_name,
            }
            questions.append(formatted_question)

        # Sample if requested
        if num_samples is not None and num_samples < len(questions):
            random.seed(seed)
            questions = random.sample(questions, num_samples)

        return questions

    def _load_hellaswag(
        self, split: str, num_samples: Optional[int], seed: int
    ) -> List[Dict]:
        """Load HellaSwag benchmark (commonsense NLI)."""
        dataset = load_dataset(
            "hellaswag",
            split=split if split != "test" else "validation",
            cache_dir=self.cache_dir,
        )

        questions = []
        for item in dataset:
            context = item["ctx"]
            endings = item["endings"]
            correct_ending_idx = int(item["label"])

            # Format question
            question_text = f"{context}\n\nWhich ending makes the most sense?"

            formatted_question = {
                "question": question_text,
                "choices": endings,
                "correct_answer": correct_ending_idx,
                "answer_text": endings[correct_ending_idx],
                "type": "multiple_choice",
                "benchmark": "hellaswag",
            }
            questions.append(formatted_question)

        # Sample if requested
        if num_samples is not None and num_samples < len(questions):
            random.seed(seed)
            questions = random.sample(questions, num_samples)

        return questions

    def get_benchmark_info(self, benchmark_name: str) -> Dict:
        """
        Get information about a benchmark.

        Args:
            benchmark_name: Name of benchmark

        Returns:
            Dict with benchmark metadata
        """
        info = {
            "mmlu": {
                "description": "Massive Multitask Language Understanding",
                "type": "multiple_choice",
                "num_choices": 4,
                "domains": "57 subjects across STEM, humanities, social sciences",
            },
            "gsm8k": {
                "description": "Grade School Math 8K",
                "type": "free_form",
                "domains": "Elementary school math word problems",
            },
            "arc_easy": {
                "description": "AI2 Reasoning Challenge (Easy)",
                "type": "multiple_choice",
                "domains": "Elementary school science questions",
            },
            "arc_challenge": {
                "description": "AI2 Reasoning Challenge (Challenge)",
                "type": "multiple_choice",
                "domains": "Challenging science questions",
            },
            "hellaswag": {
                "description": "HellaSwag Commonsense NLI",
                "type": "multiple_choice",
                "num_choices": 4,
                "domains": "Commonsense reasoning about everyday situations",
            },
        }

        return info.get(benchmark_name, {})
