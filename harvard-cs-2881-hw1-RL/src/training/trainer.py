#!/usr/bin/env python3
"""
REINFORCE trainer for prompt prefix policy optimization.

Implements policy gradient training to optimize prompt prefixes on benchmarks.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..benchmarks.evaluator import BenchmarkEvaluator
from ..benchmarks.loader import BenchmarkLoader
from ..policy.prefix_policy import PromptPrefixPolicy
from ..utils.query_utils import ModelQueryInterface
from .config import TrainingConfig


class REINFORCETrainer:
    """Trainer for optimizing prompt prefix policy with REINFORCE algorithm."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config

        # Set random seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Initialize components
        print("Initializing trainer...")

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")

        # Load model
        print("\n" + "=" * 60)
        print("Loading model...")
        print("=" * 60)
        self.model_interface = ModelQueryInterface()
        success = self.model_interface.load_model(
            config.model_name,
            device_map=config.device_map,
            load_in_4bit=config.load_in_4bit,
        )
        if not success:
            raise RuntimeError(f"Failed to load model: {config.model_name}")

        # Load policy
        print("\n" + "=" * 60)
        print("Loading policy...")
        print("=" * 60)
        self.policy = PromptPrefixPolicy(
            people_csv_path=config.people_csv_path,
            device=self.device,
        )

        # Load benchmark
        print("\n" + "=" * 60)
        print("Loading benchmark...")
        print("=" * 60)
        self.benchmark_loader = BenchmarkLoader()
        self.benchmark_questions = self.benchmark_loader.load_benchmark(
            benchmark_name=config.benchmark_name,
            split=config.benchmark_split,
            num_samples=config.benchmark_num_samples,
            seed=config.seed,
        )
        print(f"Loaded {len(self.benchmark_questions)} questions from {config.benchmark_name}")

        # Initialize evaluator
        self.evaluator = BenchmarkEvaluator(self.model_interface)

        # Training state
        self.baseline = 0.0  # Moving average baseline for variance reduction
        self.iteration = 0
        self.training_history = []

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict:
        """
        Run REINFORCE training loop.

        Returns:
            Dict with training history and final results
        """
        print("\n" + "=" * 60)
        print("Starting REINFORCE Training")
        print("=" * 60)
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Samples per iteration: {self.config.samples_per_iteration}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size} questions")
        print("=" * 60 + "\n")

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration

            # Run one training iteration
            iteration_stats = self._train_iteration()

            # Log progress
            if iteration % self.config.log_interval == 0:
                self._log_progress(iteration_stats)

            # Save checkpoint
            if iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            # Store history
            self.training_history.append(iteration_stats)

        # Final checkpoint
        self._save_checkpoint(final=True)

        # Return results
        return {
            "final_baseline": self.baseline,
            "training_history": self.training_history,
            "final_policy_stats": self.policy.get_statistics(),
            "top_people": self.policy.get_top_people(20),
        }

    def _train_iteration(self) -> Dict:
        """
        Run one iteration of REINFORCE training.

        Returns:
            Dict with iteration statistics
        """
        # Collect samples and rewards
        samples = []
        rewards = []
        log_probs = []

        for _ in range(self.config.samples_per_iteration):
            # Sample prefix from policy
            prefix, person_idx, log_prob = self.policy.sample_prefix(
                return_log_prob=True
            )

            # Evaluate on benchmark batch
            eval_results = self.evaluator.evaluate_batch(
                questions=self.benchmark_questions,
                prefix=prefix,
                batch_size=self.config.batch_size,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )

            # Get reward (accuracy)
            reward = eval_results["accuracy"]

            # Store
            samples.append({
                "prefix": prefix,
                "person_idx": person_idx,
                "reward": reward,
                "correct": eval_results["correct"],
                "total": eval_results["total"],
            })
            rewards.append(reward)
            log_probs.append(log_prob)

        # Update policy for each sample
        for sample, log_prob, reward in zip(samples, log_probs, rewards):
            self.policy.update(
                log_prob=log_prob,
                reward=reward,
                baseline=self.baseline,
                learning_rate=self.config.learning_rate,
            )

        # Update baseline (exponential moving average)
        mean_reward = np.mean(rewards)
        self.baseline = (
            self.config.baseline_decay * self.baseline
            + (1 - self.config.baseline_decay) * mean_reward
        )

        # Compute statistics
        iteration_stats = {
            "iteration": self.iteration,
            "mean_reward": float(mean_reward),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "baseline": float(self.baseline),
            "samples": samples,
            "policy_stats": self.policy.get_statistics(),
        }

        return iteration_stats

    def _log_progress(self, stats: Dict):
        """Log training progress."""
        iteration = stats["iteration"]
        mean_reward = stats["mean_reward"]
        baseline = stats["baseline"]
        top_person = stats["policy_stats"]["top_person"]
        top_prob = stats["policy_stats"]["top_person_prob"]
        entropy = stats["policy_stats"]["entropy"]

        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}")
        print(f"{'=' * 60}")
        print(f"Mean Reward: {mean_reward:.4f} ± {stats['std_reward']:.4f}")
        print(f"Baseline: {baseline:.4f}")
        print(f"Top Person: {top_person} (p={top_prob:.4f})")
        print(f"Policy Entropy: {entropy:.4f}")
        print(f"\nSample Rewards:")
        for i, sample in enumerate(stats["samples"]):
            prefix_name = sample["prefix"].replace("You are ", "").replace(". ", "")
            print(f"  {i+1}. {prefix_name}: {sample['reward']:.4f}")

    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if final:
            checkpoint_path = checkpoint_dir / "final_policy.json"
        else:
            checkpoint_path = checkpoint_dir / f"policy_iter_{self.iteration}.json"

        # Save policy
        self.policy.save(checkpoint_path)

        # Save training state
        state_path = checkpoint_path.with_suffix(".state.json")
        state = {
            "iteration": self.iteration,
            "baseline": self.baseline,
            "config": self.config.__dict__,
        }

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def evaluate_final_policy(self, num_samples: int = 100) -> Dict:
        """
        Evaluate final policy on a held-out test set.

        Args:
            num_samples: Number of test samples to evaluate

        Returns:
            Dict with evaluation results
        """
        print("\n" + "=" * 60)
        print("Final Policy Evaluation")
        print("=" * 60)

        # Get test questions
        test_questions = self.benchmark_loader.load_benchmark(
            benchmark_name=self.config.benchmark_name,
            split="test",
            num_samples=num_samples,
            seed=self.config.seed + 1,  # Different seed for test
        )

        # Get top prefixes
        top_people = self.policy.get_top_people(k=5)

        results = {}
        for person_name, prob, field, era in top_people:
            prefix = f"You are {person_name}. "

            print(f"\nEvaluating: {person_name} (p={prob:.4f})")

            eval_results = self.evaluator.evaluate_with_prefix(
                questions=test_questions,
                prefix=prefix,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )

            results[person_name] = {
                "accuracy": eval_results["accuracy"],
                "probability": prob,
                "field": field,
                "era": era,
            }

            print(f"  Accuracy: {eval_results['accuracy']:.4f}")

        # Also evaluate baseline (no prefix)
        print(f"\nEvaluating: No Prefix (baseline)")
        baseline_results = self.evaluator.evaluate_with_prefix(
            questions=test_questions,
            prefix="",
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        results["baseline"] = {
            "accuracy": baseline_results["accuracy"],
            "probability": 0.0,
            "field": "None",
            "era": "None",
        }
        print(f"  Accuracy: {baseline_results['accuracy']:.4f}")

        # Save results
        results_path = Path(self.config.output_dir) / "final_evaluation.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_path}")

        return results
