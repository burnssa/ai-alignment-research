"""
Configuration Management for RL Experiment

This module provides centralized configuration management with validation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Central configuration class for the RL experiment."""
    
    # OpenAI API settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    primary_model: str = field(default_factory=lambda: os.environ.get("OPENAI_PRIMARY_MODEL", "gpt-4o-mini"))
    judge_model: str = field(default_factory=lambda: os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.environ.get("OPENAI_TEMPERATURE", "0")))
    top_p: float = field(default_factory=lambda: float(os.environ.get("OPENAI_TOP_P", "1.0")))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("OPENAI_MAX_TOKENS", "1024")))
    
    # Training settings
    max_steps: int = field(default_factory=lambda: int(os.environ.get("MAX_STEPS", "10")))
    policy_mode: str = field(default_factory=lambda: os.environ.get("POLICY_MODE", "mab"))
    learning_rate: float = field(default_factory=lambda: float(os.environ.get("LEARNING_RATE", "0.05")))
    print_interval: int = field(default_factory=lambda: int(os.environ.get("PRINT_INTERVAL", "20")))
    
    # Persona settings
    personas: List[str] = field(default_factory=lambda: [
        "Albert Einstein", "Beyoncé", "Quentin Tarantino", 
        "Alan Turing", "Leonardo da Vinci", "Roger Federer"
    ])
    
    # Experiment presets
    experiment_preset: str = field(default_factory=lambda: os.environ.get("EXPERIMENT_PRESET", "default"))
    
    # Dataset settings
    train_datasets: str = field(default_factory=lambda: os.environ.get("TRAIN_DATASETS", "gsm8k:main:train"))
    ood_datasets: str = field(default_factory=lambda: os.environ.get("OOD_DATASETS", "gsm8k:main:test"))
    max_dataset_items: int = field(default_factory=lambda: int(os.environ.get("MAX_DATASET_ITEMS", "0")))
    ood_eval_samples: int = field(default_factory=lambda: int(os.environ.get("OOD_EVAL_SAMPLES", "10")))
    
    # System settings
    dry_run: bool = field(default_factory=lambda: os.environ.get("DRY_RUN", "false").lower() == "true")
    
    # Multithreading settings
    use_multithreading: bool = field(default_factory=lambda: os.environ.get("USE_MULTITHREADING", "true").lower() == "true")
    max_workers: int = field(default_factory=lambda: int(os.environ.get("MAX_WORKERS", "32")))
    inflight: int = field(default_factory=lambda: int(os.environ.get("INFLIGHT", "16")))
    ood_eval_every: int = field(default_factory=lambda: int(os.environ.get("OOD_EVAL_EVERY", "0")))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._apply_experiment_preset()
        self.validate()
    
    def _apply_experiment_preset(self):
        """Apply experiment-specific presets."""
        if self.experiment_preset == "truthfulqa":
            # Truth tellers vs liars experiment
            self.personas = [
                # Truth tellers (historically known for honesty/accuracy)
                "George Washington", "Mahatma Gandhi", "Marie Curie",
                # Liars (historically known for deception/propaganda) 
                "Joseph Goebbels", "P.T. Barnum", "Frank Abagnale"
            ]
            self.train_datasets = "truthfulqa/truthful_qa:generation:validation"
            self.ood_datasets = "truthfulqa/truthful_qa:generation:validation"
            self.max_steps = 100  # Reasonable for validation set
        elif self.experiment_preset == "truthfulqa_mc":
            # Multiple choice version
            self.personas = [
                "George Washington", "Mahatma Gandhi", "Marie Curie",
                "Joseph Goebbels", "P.T. Barnum", "Frank Abagnale"
            ]
            self.train_datasets = "truthfulqa/truthful_qa:multiple_choice:validation"
            self.ood_datasets = "truthfulqa/truthful_qa:multiple_choice:validation"
            self.max_steps = 100
    
    def validate(self):
        """Validate configuration values."""
        if not self.dry_run and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when not in dry run mode")
        
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        if self.learning_rate <= 0 or self.learning_rate >= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not self.personas:
            raise ValueError("personas list cannot be empty")
        
        if self.policy_mode not in ["mab", "contextual"]:
            raise ValueError("policy_mode must be 'mab' or 'contextual'")
        
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if self.inflight <= 0:
            raise ValueError("inflight must be positive")
        
        if self.inflight > self.max_workers:
            raise ValueError("inflight cannot exceed max_workers")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        return cls()
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Create config from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create config with defaults, then override with file data
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.validate()
        return config
    
    def save_to_file(self, filepath: str) -> None:
        """Save config to JSON file."""
        import json
        from pathlib import Path
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'openai_api_key': '***' if self.openai_api_key else None,  # Hide sensitive data
            'primary_model': self.primary_model,
            'judge_model': self.judge_model,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens,
            'max_steps': self.max_steps,
            'policy_mode': self.policy_mode,
            'learning_rate': self.learning_rate,
            'print_interval': self.print_interval,
            'personas': self.personas,
            'experiment_preset': self.experiment_preset,
            'train_datasets': self.train_datasets,
            'ood_datasets': self.ood_datasets,
            'max_dataset_items': self.max_dataset_items,
            'ood_eval_samples': self.ood_eval_samples,
            'dry_run': self.dry_run,
            'use_multithreading': self.use_multithreading,
            'max_workers': self.max_workers,
            'inflight': self.inflight,
            'ood_eval_every': self.ood_eval_every,
        }