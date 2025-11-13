"""
Logging Module for RL Experiment

This module provides structured logging capabilities for the RL experiment.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str = "rl_experiment",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Setup a logger with both file and console handlers."""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "rl_experiment") -> logging.Logger:
    """Get the logger instance."""
    return logging.getLogger(name)


# Create default logger
default_logger = setup_logger()


class ExperimentLogger:
    """Enhanced logger for experiment tracking."""
    
    def __init__(self, name: str = "rl_experiment", log_file: Optional[str] = None):
        self.logger = setup_logger(name, log_file=log_file)
        self.step_count = 0
    
    def info(self, message: str, **kwargs):
        """Log info message with optional key-value pairs."""
        if kwargs:
            formatted_kwargs = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {formatted_kwargs}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional key-value pairs."""
        if kwargs:
            formatted_kwargs = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {formatted_kwargs}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional key-value pairs."""
        if kwargs:
            formatted_kwargs = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {formatted_kwargs}"
        self.logger.error(message)
    
    def log_step(self, step: int, reward: float, persona: str, **additional):
        """Log a training step with standardized format."""
        self.step_count = step
        metrics = {
            'step': step,
            'reward': f"{reward:.3f}",
            'persona': persona,
            **additional
        }
        self.info("Training step completed", **metrics)
    
    def log_config(self, config_dict: dict):
        """Log configuration at the start of training."""
        self.info("Starting experiment with configuration:")
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")
    
    def log_dataset_loading(self, dataset_name: str, count: int, dataset_type: str = "train"):
        """Log dataset loading information."""
        self.info(f"Dataset loaded", 
                 dataset=dataset_name, 
                 type=dataset_type, 
                 examples=count)
    
    def log_experiment_summary(self, total_steps: int, avg_reward: float, 
                             best_persona: str, final_probs: dict):
        """Log experiment summary at the end."""
        self.info("Experiment completed", 
                 total_steps=total_steps,
                 avg_reward=f"{avg_reward:.3f}",
                 best_persona=best_persona)
        
        self.info("Final persona probabilities:")
        for persona, prob in final_probs.items():
            self.info(f"  {persona}: {prob:.3f}")


# Create default experiment logger
experiment_logger = ExperimentLogger()