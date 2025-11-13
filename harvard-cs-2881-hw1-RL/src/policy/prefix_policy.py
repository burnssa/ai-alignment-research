#!/usr/bin/env python3
"""
Prompt Prefix Policy with learnable parameters over 10k people.

Implements a probability distribution P[i] ∝ exp(parameters[i]) over people,
with policy gradient (REINFORCE) updates to optimize benchmark performance.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PromptPrefixPolicy:
    """
    Policy that learns to select prompt prefixes from 10k notable people.

    The policy maintains learnable parameters for each person and computes
    a probability distribution using softmax. Optimized via REINFORCE.
    """

    def __init__(self, people_csv_path: str, device: str = "cpu"):
        """
        Initialize the policy with people from CSV.

        Args:
            people_csv_path: Path to CSV file with columns: name, field, era
            device: Device to run computations on ("cpu", "cuda", "mps")
        """
        self.device = device

        # Load people from CSV
        self.people = self._load_people(people_csv_path)
        self.num_people = len(self.people)

        print(f"Loaded {self.num_people} people from {people_csv_path}")

        # Initialize learnable parameters (logits)
        # Start with uniform distribution (all zeros)
        self.parameters = nn.Parameter(
            torch.zeros(self.num_people, device=device)
        )

        # Track sampling history for debugging
        self.sample_history: List[Tuple[int, str, float]] = []

    def _load_people(self, csv_path: str) -> List[Dict[str, str]]:
        """
        Load people data from CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of dicts with keys: name, field, era
        """
        people = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                people.append({
                    "name": row["name"],
                    "field": row.get("field", "Unknown"),
                    "era": row.get("era", "Unknown"),
                })
        return people

    def get_distribution(self) -> torch.Tensor:
        """
        Get current probability distribution over people.

        Returns:
            Tensor of shape (num_people,) with probabilities summing to 1
        """
        return torch.softmax(self.parameters, dim=0)

    def sample_prefix(
        self, return_log_prob: bool = False
    ) -> Tuple[str, int, Optional[torch.Tensor]]:
        """
        Sample a person according to current policy and return prompt prefix.

        Args:
            return_log_prob: Whether to return log probability for REINFORCE

        Returns:
            Tuple of (prefix_string, person_idx, log_prob)
            - prefix_string: "You are [Person Name]. "
            - person_idx: Index of sampled person
            - log_prob: Log probability of sample (if return_log_prob=True)
        """
        # Get probability distribution
        probs = self.get_distribution()

        # Sample from categorical distribution
        dist = Categorical(probs)
        person_idx = dist.sample().item()

        # Get person name and create prefix
        person_name = self.people[person_idx]["name"]
        prefix = f"You are {person_name}. "

        # Record in history
        prob_value = probs[person_idx].item()
        self.sample_history.append((person_idx, person_name, prob_value))

        if return_log_prob:
            log_prob = dist.log_prob(torch.tensor(person_idx, device=self.device))
            return prefix, person_idx, log_prob
        else:
            return prefix, person_idx, None

    def update(
        self,
        log_prob: torch.Tensor,
        reward: float,
        baseline: float = 0.0,
        learning_rate: float = 0.01
    ):
        """
        Update policy parameters using REINFORCE algorithm.

        Args:
            log_prob: Log probability of the sampled action
            reward: Reward received (e.g., benchmark score)
            baseline: Baseline for variance reduction (e.g., moving average)
            learning_rate: Learning rate for gradient update
        """
        # REINFORCE gradient: ∇log π(a|s) * (R - baseline)
        advantage = reward - baseline
        loss = -log_prob * advantage

        # Compute gradient
        loss.backward()

        # Manual gradient descent step
        with torch.no_grad():
            self.parameters -= learning_rate * self.parameters.grad
            self.parameters.grad.zero_()

    def get_top_people(self, k: int = 10) -> List[Tuple[str, float, str, str]]:
        """
        Get top k people by probability.

        Args:
            k: Number of top people to return

        Returns:
            List of tuples: (name, probability, field, era)
        """
        probs = self.get_distribution().detach().cpu().numpy()
        top_indices = np.argsort(probs)[-k:][::-1]

        top_people = []
        for idx in top_indices:
            person = self.people[idx]
            top_people.append((
                person["name"],
                float(probs[idx]),
                person["field"],
                person["era"]
            ))

        return top_people

    def save(self, path: str):
        """
        Save policy to disk.

        Args:
            path: Path to save policy parameters
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save parameters and metadata
        save_dict = {
            "parameters": self.parameters.detach().cpu().tolist(),
            "num_people": self.num_people,
            "people": self.people,
            "top_people": self.get_top_people(20),
        }

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=2)

        print(f"Policy saved to {save_path}")

    def load(self, path: str):
        """
        Load policy from disk.

        Args:
            path: Path to saved policy
        """
        with open(path, "r") as f:
            save_dict = json.load(f)

        # Load parameters
        self.parameters = nn.Parameter(
            torch.tensor(save_dict["parameters"], device=self.device)
        )

        # Verify people list matches
        if len(save_dict["people"]) != self.num_people:
            raise ValueError(
                f"Saved policy has {len(save_dict['people'])} people, "
                f"but current policy has {self.num_people}"
            )

        print(f"Policy loaded from {path}")

    def get_statistics(self) -> Dict:
        """
        Get policy statistics for logging.

        Returns:
            Dict with statistics about current policy
        """
        probs = self.get_distribution().detach().cpu().numpy()

        return {
            "entropy": float(-np.sum(probs * np.log(probs + 1e-10))),
            "max_prob": float(np.max(probs)),
            "min_prob": float(np.min(probs)),
            "top_person": self.people[int(np.argmax(probs))]["name"],
            "top_person_prob": float(np.max(probs)),
        }
