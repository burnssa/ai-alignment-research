#!/usr/bin/env python3
"""
Training configuration for HW1 RL experiment.

Manages hyperparameters and settings for policy gradient training.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for REINFORCE training."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    device_map: str = "auto"
    load_in_4bit: bool = False

    # Policy settings
    people_csv_path: str = "notable_people_10k.csv"
    learning_rate: float = 0.01
    baseline_decay: float = 0.9  # Exponential moving average for baseline

    # Training settings
    num_iterations: int = 100
    samples_per_iteration: int = 5  # Number of prefix samples per iteration
    batch_size: int = 10  # Number of questions per evaluation

    # Benchmark settings
    benchmark_name: str = "gsm8k"
    benchmark_split: str = "train"
    benchmark_num_samples: int = 1000  # Total questions to cache

    # Generation settings
    max_new_tokens: int = 200
    temperature: float = 0.0  # Use greedy decoding for eval

    # Logging and checkpointing
    log_interval: int = 1  # Log every N iterations
    checkpoint_interval: int = 10  # Save checkpoint every N iterations
    output_dir: str = "outputs"

    # Random seed
    seed: int = 42

    def save(self, path: str):
        """Save configuration to JSON file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

        print(f"Config saved to {save_path}")

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["Training Configuration:"]
        lines.append("-" * 50)

        # Group settings
        groups = {
            "Model": ["model_name", "device_map", "load_in_4bit"],
            "Policy": ["people_csv_path", "learning_rate", "baseline_decay"],
            "Training": [
                "num_iterations",
                "samples_per_iteration",
                "batch_size",
            ],
            "Benchmark": [
                "benchmark_name",
                "benchmark_split",
                "benchmark_num_samples",
            ],
            "Generation": ["max_new_tokens", "temperature"],
            "Logging": ["log_interval", "checkpoint_interval", "output_dir"],
        }

        for group_name, keys in groups.items():
            lines.append(f"\n{group_name}:")
            for key in keys:
                value = getattr(self, key)
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)
