"""
Policy Module for RL Experiment

This module contains the policy implementations for the multi-armed bandit experiment.
"""

import numpy as np
import json
import pickle
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Policy:
    """Multi-armed bandit policy using REINFORCE updates."""
    
    names: List[str]
    probs: np.ndarray
    lr: float = 0.05

    @classmethod
    def uniform(cls, names: List[str], lr: float = 0.05):
        """Create a uniform policy over the given names."""
        n = len(names)
        return cls(names=names, probs=np.ones(n) / n, lr=lr)

    def sample_index(self) -> int:
        """Sample an action index according to current probabilities."""
        return int(np.random.choice(len(self.names), p=self.probs))

    def update(self, chosen: int, reward: float):
        """Update policy using REINFORCE gradient."""
        # REINFORCE on categorical: increase chosen, decrease others
        grad = -reward * self.probs
        grad[chosen] = reward * (1 - self.probs[chosen])
        self.probs = np.clip(self.probs + self.lr * grad, 1e-9, None)
        self.probs = self.probs / self.probs.sum()
    
    def get_top_persona(self) -> str:
        """Get the name of the persona with highest probability."""
        return self.names[int(np.argmax(self.probs))]
    
    def get_probabilities(self) -> dict:
        """Get dictionary mapping persona names to probabilities."""
        return dict(zip(self.names, self.probs))
    
    def save(self, filepath: str) -> None:
        """Save policy to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for human readability
        policy_data = {
            'names': self.names,
            'probs': self.probs.tolist(),
            'lr': self.lr,
            'type': 'Policy'
        }
        
        with open(filepath, 'w') as f:
            json.dump(policy_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Policy':
        """Load policy from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data.get('type') != 'Policy':
            raise ValueError(f"Invalid policy file format: expected 'Policy', got {data.get('type')}")
        
        policy = cls(
            names=data['names'],
            probs=np.array(data['probs']),
            lr=data['lr']
        )
        
        return policy
    
    def save_checkpoint(self, checkpoint_dir: str, step: int) -> str:
        """Save a checkpoint of the policy."""
        checkpoint_path = Path(checkpoint_dir) / f"policy_step_{step}.json"
        self.save(str(checkpoint_path))
        return str(checkpoint_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for logging/serialization."""
        return {
            'names': self.names,
            'probabilities': self.get_probabilities(),
            'learning_rate': self.lr,
            'top_persona': self.get_top_persona()
        }


def get_policy(mode: str, people: List[str], lr: float = 0.05) -> Policy:
    """Factory function to create policy based on mode."""
    mode = (mode or "mab").lower()
    if mode == "contextual":
        # TODO: Implement contextual policy when needed
        raise NotImplementedError("Contextual policy not yet implemented")
    # Default: simple MAB policy
    return Policy.uniform(people, lr=lr)


def moving_average(xs: List[float], k: int) -> List[float]:
    """Calculate moving average with window size k."""
    if k <= 1:
        return xs
    out = []
    s = 0.0
    for i, v in enumerate(xs):
        s += v
        if i >= k:
            s -= xs[i - k]
        out.append(s / min(i + 1, k))
    return out