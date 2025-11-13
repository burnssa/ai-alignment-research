"""
Main Training Script for RL Experiment

This script imports all modules and orchestrates the main training loop for the reinforcement learning experiment.
"""

import numpy as np
import torch
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed

# Import all the custom modules
from config import Config
from policy import Policy
from openai_client import OpenAIClient
from data_loading import sample_training_example, sample_ood_example
from logger import ExperimentLogger


class RLTrainer:
    """
    Main trainer class that orchestrates the RL experiment using all modules.
    """
    
    def __init__(self, config: Config = None, log_file: str = None):
        """
        Initialize the RL trainer.
        
        Args:
            config: Configuration object for the experiment
            log_file: Optional path to log file
        """
        self.config = config or Config.from_env()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logger
        self.logger = ExperimentLogger("rl_trainer", log_file)
        
        # Initialize OpenAI client
        self.openai_client = OpenAIClient(
            api_key=self.config.openai_api_key,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            dry_run=self.config.dry_run
        )
        
        # Initialize training data storage
        self.training_history = {
            "episodes": [],
            "rewards": [],
            "losses": [],
            "actions": [],
            "states": []
        }
        
        # Per-persona binary stats for multithreaded training
        self.person_stats: Dict[str, Dict[str, int]] = {}
  
    def train(self):
        """
        Main training loop - chooses between single-threaded and multithreaded.
        """
        if self.config.use_multithreading:
            return self.train_multithreaded()
        else:
            return self.train_single_threaded()
    
    def train_single_threaded(self):
        """
        Original single-threaded training loop.
        """
        # Log configuration
        config_dict = self.config.to_dict()
        config_dict['device'] = str(self.device)
        self.logger.log_config(config_dict)

        rewards: List[float] = []
        chosen_names: List[str] = []
        probs_over_time: List[np.ndarray] = []

        # Initialize policy with configured personas and learning rate
        policy = Policy.uniform(self.config.personas, lr=self.config.learning_rate)

        for step in tqdm(range(1, self.config.max_steps + 1)):
            
            ex = sample_training_example()
            idx = policy.sample_index()
            name = policy.names[idx]
            
            chosen_names.append(name)

            # call primary model with persona
            prompt = self.openai_client.persona_prompt(name, ex["task_text"])
            answer = self.openai_client.call_model(self.config.primary_model, prompt)

            # Get baseline response based on task type
            if ex["raw"]["type"] == "free_form":
                baseline_response = str(ex["raw"]["answer"])
            elif ex["raw"]["type"] == "multiple_choice":
                correct_idx = ex["raw"]["answer_index"]
                baseline_response = f"{correct_idx}"
            elif ex["raw"]["type"] == "judged":
                baseline_response = str(ex["raw"]["reference"])
            else:
                baseline_response = "No baseline available"

            reward = float(self.openai_client.judge_response(
                self.config.judge_model, ex["task_text"], answer, baseline_response
            ))
            rewards.append(reward)

            # policy update
            policy.update(idx, reward)

            # track current probs for plotting
            probs_over_time.append(policy.probs.copy())

            if step % self.config.print_interval == 0:
                self.logger.log_step(
                    step=step,
                    reward=rewards[-1],
                    persona=chosen_names[-1],
                    top_persona=policy.get_top_persona()
                )

        # Log experiment summary
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        final_probs = policy.get_probabilities()
        self.logger.log_experiment_summary(
            total_steps=self.config.max_steps,
            avg_reward=avg_reward,
            best_persona=policy.get_top_persona(),
            final_probs=final_probs
        )
        
        # Save final policy
        try:
            policy.save("outputs/final_policy.json")
            self.logger.info("Final policy saved to outputs/final_policy.json")
        except Exception as e:
            self.logger.warning(f"Failed to save final policy: {e}")
        
        return {
            'rewards': rewards,
            'chosen_names': chosen_names,
            'probs_over_time': probs_over_time,
            'final_policy': policy
        }
    
    def tally_binary_reward(self, persona_name: str, reward: float, eps: float = 1e-6) -> None:
        """Count exact 1s and 0s (within eps). Non-binary rewards are tracked separately."""
        if persona_name not in self.person_stats:
            self.person_stats[persona_name] = {"ones": 0, "zeros": 0, "nonbinary": 0}
        
        s = self.person_stats[persona_name]
        if reward >= 1.0 - eps:
            s["ones"] += 1
        elif reward <= 0.0 + eps:
            s["zeros"] += 1
        else:
            s["nonbinary"] += 1
    
    def fraction_str(self, s: Dict[str, int]) -> str:
        """Get fraction string for display."""
        denom = s["ones"] + s["zeros"]
        return f"{s['ones']}/{denom}" if denom > 0 else "—"
    
    def print_top_binary_stats(self, limit: int = 6) -> None:
        """Print binary correctness statistics."""
        lines = []
        for n, s in self.person_stats.items():
            denom = s["ones"] + s["zeros"]
            if denom > 0:
                frac = s["ones"] / denom
                lines.append((denom, frac, n, s))
        lines.sort(reverse=True)  # more evidence first
        if lines:
            self.logger.info("Binary correctness so far (exact 1s/0s; non-binary ignored):")
            for denom, frac, n, s in lines[:limit]:
                self.logger.info(f"  {n:<24} {self.fraction_str(s):>8}  ({frac:5.2%})  nonbin={s['nonbinary']}")
    
    def run_step_job(self, ex: dict, idx: int, name: str) -> dict:
        """
        One training step job for multithreaded execution:
        1) Call primary model with persona prompt
        2) Compute reward via judging
        Returns all info needed for policy update.
        """
        try:
            prompt = self.openai_client.persona_prompt(name, ex["task_text"])
            answer = self.openai_client.call_model(self.config.primary_model, prompt)
            
            # Get baseline response based on task type
            if ex["raw"]["type"] == "free_form":
                baseline_response = str(ex["raw"]["answer"])
            elif ex["raw"]["type"] == "multiple_choice":
                correct_idx = ex["raw"]["answer_index"]
                baseline_response = f"{correct_idx}"
            elif ex["raw"]["type"] == "judged":
                baseline_response = str(ex["raw"]["reference"])
            else:
                baseline_response = "No baseline available"
            
            reward = float(self.openai_client.judge_response(
                self.config.judge_model, ex["task_text"], answer, baseline_response
            ))
        except Exception as e:
            # Be robust to transient errors
            self.logger.warning(f"Error in step job: {e}")
            answer = ""
            reward = 0.5
        
        return {
            "idx": idx,
            "name": name,
            "ex": ex,
            "task_text": ex["task_text"],
            "answer": answer,
            "reward": float(reward),
        }
    
    def ood_eval_for_persona(self, top_persona: str, k: int, pool: ThreadPoolExecutor) -> float:
        """Run K OOD samples in parallel and return the average score."""
        k = max(1, k)
        futs = []
        
        def _one():
            try:
                ood_ex = sample_ood_example()
                ood_prompt = self.openai_client.persona_prompt(top_persona, ood_ex["task_text"])
                ood_answer = self.openai_client.call_model(self.config.primary_model, ood_prompt)
                
                # Get baseline for OOD example
                if ood_ex["raw"]["type"] == "free_form":
                    baseline_response = str(ood_ex["raw"]["answer"])
                elif ood_ex["raw"]["type"] == "multiple_choice":
                    correct_idx = ood_ex["raw"]["answer_index"]
                    baseline_response = f"{correct_idx}"
                elif ood_ex["raw"]["type"] == "judged":
                    baseline_response = str(ood_ex["raw"]["reference"])
                else:
                    baseline_response = "No baseline available"
                
                return float(self.openai_client.judge_response(
                    self.config.judge_model, ood_ex["task_text"], ood_answer, baseline_response
                ))
            except Exception as e:
                self.logger.warning(f"Error in OOD eval: {e}")
                return 0.5
        
        for _ in range(k):
            futs.append(pool.submit(_one))
        
        scores = []
        for f in as_completed(futs):
            try:
                scores.append(float(f.result()))
            except Exception:
                scores.append(0.5)
        return float(np.mean(scores)) if scores else 0.5
    
    def train_multithreaded(self):
        """
        Multithreaded training loop with concurrent API calls.
        """
        # Log configuration
        config_dict = self.config.to_dict()
        config_dict['device'] = str(self.device)
        self.logger.log_config(config_dict)
        
        rewards: List[float] = []
        chosen_names: List[str] = []
        probs_over_time: List[np.ndarray] = []
        ood_rewards: List[float] = []
        ood_steps: List[int] = []
        
        # Initialize policy with configured personas and learning rate
        policy = Policy.uniform(self.config.personas, lr=self.config.learning_rate)
        
        # Initialize per-persona binary stats
        self.person_stats = {name: {"ones": 0, "zeros": 0, "nonbinary": 0} for name in policy.names}
        
        submitted = [0]
        completed = [0]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            inflight = set()
            
            def submit_one():
                """Sample example, select persona, and submit a concurrent step job."""
                ex = sample_training_example()
                idx = policy.sample_index()
                name = policy.names[idx]
                
                fut = pool.submit(self.run_step_job, ex, idx, name)
                inflight.add(fut)
                submitted[0] += 1
            
            # Prime up to INFLIGHT concurrent jobs
            while submitted[0] < min(self.config.max_steps, self.config.inflight):
                submit_one()
            
            # Main multithreaded loop
            with tqdm(total=self.config.max_steps, desc="Training") as pbar:
                while completed[0] < self.config.max_steps or inflight:
                    # Wait for any job to finish
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        inflight.remove(fut)
                        res = fut.result()
                        
                        reward = float(res["reward"])
                        idx = int(res["idx"])
                        name = res["name"]
                        
                        # Per-persona binary tally (training only)
                        self.tally_binary_reward(name, reward)
                        
                        # Bookkeeping
                        rewards.append(reward)
                        chosen_names.append(name)
                        
                        # Policy update (sequential)
                        policy.update(idx, reward)
                        
                        # Track current probs for plotting
                        probs_over_time.append(policy.probs.copy())
                        
                        completed[0] += 1
                        pbar.update(1)
                        
                        # OOD eval hook
                        if (self.config.ood_eval_every > 0 and 
                            completed[0] % self.config.ood_eval_every == 0):
                            top_idx = int(np.argmax(policy.probs))
                            top_persona = policy.names[top_idx]
                            ood_avg = self.ood_eval_for_persona(top_persona, self.config.ood_eval_samples, pool)
                            ood_rewards.append(ood_avg)
                            ood_steps.append(completed[0])
                            self.logger.info(f"OOD eval at step {completed[0]}: {ood_avg:.3f} for {top_persona}")
                        
                        # Debug/summary logging
                        if completed[0] % self.config.print_interval == 0:
                            self.logger.log_step(
                                step=completed[0],
                                reward=rewards[-1],
                                persona=chosen_names[-1],
                                top_persona=policy.get_top_persona()
                            )
                            
                            # Print binary stats every 100 steps
                            if completed[0] % 100 == 0:
                                self.print_top_binary_stats(limit=6)
                                
                                # Show top personas
                                final_probs = policy.get_probabilities()
                                sorted_personas = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)[:6]
                                self.logger.info("Top personas:")
                                for persona_name, prob in sorted_personas:
                                    self.logger.info(f"  {persona_name:30s}  p={prob:.3f}")
                        
                        # Keep pipeline full
                        if submitted[0] < self.config.max_steps:
                            submit_one()
        
        # Log experiment summary
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        final_probs = policy.get_probabilities()
        self.logger.log_experiment_summary(
            total_steps=self.config.max_steps,
            avg_reward=avg_reward,
            best_persona=policy.get_top_persona(),
            final_probs=final_probs
        )
        
        # Save final policy
        try:
            policy.save("outputs/final_policy.json")
            self.logger.info("Final policy saved to outputs/final_policy.json")
        except Exception as e:
            self.logger.warning(f"Failed to save final policy: {e}")
        
        return {
            'rewards': rewards,
            'chosen_names': chosen_names,
            'probs_over_time': probs_over_time,
            'final_policy': policy,
            'ood_rewards': ood_rewards,
            'ood_steps': ood_steps,
            'person_stats': self.person_stats
        }
    
    def evaluate_baseline(self, num_samples: int = 50, use_multithreading: bool = None) -> dict:
        """
        Evaluate baseline performance for each persona on TruthfulQA questions.
        This provides a ground truth comparison for the multi-armed bandit learning.
        Uses multithreading by default for faster evaluation.
        """
        # Default to config setting, but allow override
        if use_multithreading is None:
            use_multithreading = self.config.use_multithreading
            
        self.logger.info(f"Running baseline evaluation with {num_samples} samples per persona...")
        self.logger.info(f"Multithreading: {'enabled' if use_multithreading else 'disabled'}")
        
        if use_multithreading:
            return self._evaluate_baseline_multithreaded(num_samples)
        else:
            return self._evaluate_baseline_single_threaded(num_samples)
    
    def _evaluate_baseline_single_threaded(self, num_samples: int) -> dict:
        """Single-threaded baseline evaluation (slower but simpler)."""
        baseline_results = {}
        
        for persona_name in self.config.personas:
            self.logger.info(f"Evaluating persona: {persona_name}")
            rewards = []
            
            for _ in range(num_samples):
                try:
                    # Sample a training example
                    ex = sample_training_example()
                    
                    # Get response from this persona
                    prompt = self.openai_client.persona_prompt(persona_name, ex["task_text"])
                    answer = self.openai_client.call_model(self.config.primary_model, prompt)
                    
                    # Get baseline response based on task type
                    if ex["raw"]["type"] == "free_form":
                        baseline_response = str(ex["raw"]["answer"])
                    elif ex["raw"]["type"] == "multiple_choice":
                        correct_idx = ex["raw"]["answer_index"]
                        baseline_response = f"{correct_idx}"
                    elif ex["raw"]["type"] == "judged":
                        baseline_response = str(ex["raw"]["reference"])
                    else:
                        baseline_response = "No baseline available"
                    
                    # Judge the response
                    reward = float(self.openai_client.judge_response(
                        self.config.judge_model, ex["task_text"], answer, baseline_response
                    ))
                    rewards.append(reward)
                    
                except Exception as e:
                    self.logger.warning(f"Error evaluating {persona_name}: {e}")
                    rewards.append(0.5)  # Neutral score for errors
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            baseline_results[persona_name] = {
                'avg_reward': avg_reward,
                'rewards': rewards,
                'num_samples': len(rewards)
            }
            
            self.logger.info(f"{persona_name}: {avg_reward:.3f} avg reward")
        
        return self._analyze_baseline_results(baseline_results)
    
    def _evaluate_baseline_multithreaded(self, num_samples: int) -> dict:
        """Multithreaded baseline evaluation for much faster performance."""
        baseline_results = {name: {'rewards': [], 'num_samples': 0} for name in self.config.personas}
        
        def evaluate_single_sample(persona_name: str) -> tuple:
            """Evaluate one persona on one sample - returns (persona_name, reward)."""
            try:
                ex = sample_training_example()
                prompt = self.openai_client.persona_prompt(persona_name, ex["task_text"])
                answer = self.openai_client.call_model(self.config.primary_model, prompt)
                
                # Get baseline response
                if ex["raw"]["type"] == "free_form":
                    baseline_response = str(ex["raw"]["answer"])
                elif ex["raw"]["type"] == "multiple_choice":
                    correct_idx = ex["raw"]["answer_index"]
                    baseline_response = f"{correct_idx}"
                elif ex["raw"]["type"] == "judged":
                    baseline_response = str(ex["raw"]["reference"])
                else:
                    baseline_response = "No baseline available"
                
                reward = float(self.openai_client.judge_response(
                    self.config.judge_model, ex["task_text"], answer, baseline_response
                ))
                return (persona_name, reward)
            
            except Exception as e:
                self.logger.warning(f"Error in baseline sample for {persona_name}: {e}")
                return (persona_name, 0.5)
        
        total_tasks = len(self.config.personas) * num_samples
        completed = [0]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            # Submit all evaluation tasks
            futures = []
            for persona_name in self.config.personas:
                for _ in range(num_samples):
                    future = pool.submit(evaluate_single_sample, persona_name)
                    futures.append(future)
            
            # Collect results as they complete
            with tqdm(total=total_tasks, desc="Baseline Evaluation") as pbar:
                for future in as_completed(futures):
                    persona_name, reward = future.result()
                    baseline_results[persona_name]['rewards'].append(reward)
                    baseline_results[persona_name]['num_samples'] += 1
                    
                    completed[0] += 1
                    pbar.update(1)
                    
                    # Log progress every 10% or 50 samples
                    if completed[0] % max(1, total_tasks // 10) == 0 or completed[0] % 50 == 0:
                        progress = (completed[0] / total_tasks) * 100
                        self.logger.info(f"Baseline progress: {completed[0]}/{total_tasks} ({progress:.1f}%)")
        
        # Calculate averages and log final results
        for persona_name in self.config.personas:
            rewards = baseline_results[persona_name]['rewards']
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            baseline_results[persona_name]['avg_reward'] = avg_reward
            self.logger.info(f"{persona_name}: {avg_reward:.3f} avg reward ({len(rewards)} samples)")
        
        return self._analyze_baseline_results(baseline_results)
    
    def _analyze_baseline_results(self, baseline_results: dict) -> dict:
        """Analyze baseline results and compute truth vs lies comparison."""
        
        # Analyze truth tellers vs liars
        truth_tellers = ["George Washington", "Mahatma Gandhi", "Marie Curie"]
        liars = ["Joseph Goebbels", "P.T. Barnum", "Frank Abagnale"]
        
        truth_avg = sum(baseline_results[name]['avg_reward'] for name in truth_tellers if name in baseline_results) / len(truth_tellers)
        liar_avg = sum(baseline_results[name]['avg_reward'] for name in liars if name in baseline_results) / len(liars)
        
        analysis = {
            'truth_tellers_avg': truth_avg,
            'liars_avg': liar_avg,
            'truth_advantage': truth_avg - liar_avg,
            'results_by_persona': baseline_results
        }
        
        self.logger.info(f"\n=== BASELINE ANALYSIS ===")
        self.logger.info(f"Truth tellers average: {truth_avg:.3f}")
        self.logger.info(f"Liars average: {liar_avg:.3f}")
        self.logger.info(f"Truth advantage: {truth_avg - liar_avg:.3f}")
        self.logger.info(f"=========================")
        
        return analysis
    
    def analyze_truthfulness_convergence(self, results: dict) -> dict:
        """
        Analyze whether the multi-armed bandit converged to prefer truth tellers.
        """
        if 'final_policy' not in results:
            return {}
        
        policy = results['final_policy']
        final_probs = policy.get_probabilities()
        
        truth_tellers = ["George Washington", "Mahatma Gandhi", "Marie Curie"]
        liars = ["Joseph Goebbels", "P.T. Barnum", "Frank Abagnale"]
        
        truth_prob = sum(final_probs.get(name, 0) for name in truth_tellers)
        liar_prob = sum(final_probs.get(name, 0) for name in liars)
        
        analysis = {
            'truth_tellers_prob': truth_prob,
            'liars_prob': liar_prob,
            'truth_preference_ratio': truth_prob / liar_prob if liar_prob > 0 else float('inf'),
            'convergence_successful': truth_prob > liar_prob,
            'final_probabilities': final_probs
        }
        
        self.logger.info(f"\n=== CONVERGENCE ANALYSIS ===")
        self.logger.info(f"Truth tellers probability: {truth_prob:.3f}")
        self.logger.info(f"Liars probability: {liar_prob:.3f}")
        self.logger.info(f"Ratio (truth/lies): {analysis['truth_preference_ratio']:.2f}")
        self.logger.info(f"Convergence successful: {analysis['convergence_successful']}")
        self.logger.info(f"============================")
        
        return analysis


if __name__ == "__main__":
    try:
        config = Config.from_env()
        trainer = RLTrainer(config)
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        raise

