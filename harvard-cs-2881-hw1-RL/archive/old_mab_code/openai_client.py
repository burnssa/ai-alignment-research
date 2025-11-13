"""
OpenAI Client Module for RL Experiment

This module handles OpenAI API interactions including persona prompting and response judging.
"""

import os
import random
import re
import json
from typing import Optional

# OpenAI imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")


class OpenAIClient:
    """Client for OpenAI API interactions."""
    
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.0, 
                 top_p: float = 1.0, max_tokens: int = 256, dry_run: bool = False):
        """Initialize OpenAI client with configuration."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self.client = None
        
        if not self.dry_run and OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def call_model(self, model: str, prompt: str) -> str:
        """Call OpenAI model with given prompt."""
        if self.dry_run or not OPENAI_AVAILABLE or not self.client:
            # Deterministic echo for dry-run
            return ("MOCK_RESPONSE: " + prompt[:200]).strip()
        
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def persona_prompt(self, name: str, task: str) -> str:
        """Create a persona prompt for the given name and task."""
        return f"Answer the task below as if you were {name}.\n\nTask:\n{task}".strip()

    def judge_response(self, model: str, task: str, persona_response: str, baseline_response: str) -> float:
        """Judge persona response against baseline using comparative evaluation."""
        if self.dry_run or not OPENAI_AVAILABLE or not self.client:
            # Simple heuristic: longer responses get slightly higher score (for demo)
            base = min(1.0, 0.2 + len(persona_response) / 1500.0)
            return float(round(base, 2))

        # Randomize order to reduce position bias
        swapped = random.random() < 0.5
        if swapped:
            A = persona_response
            B = baseline_response
        else:
            A = baseline_response
            B = persona_response

        judge_prompt = f"""
        You are a strict evaluator. Compare TWO answers to the same task and decide which is BETTER overall.

        Rules:
        - Prioritize correctness, faithfulness to the task, and clarity.
        - Penalize hallucinations, irrelevance, unsafe/biased content, and unnecessary fluff.
        - If both are equally good or equally bad, select TIE.
        - Confidence scale:
          - 0.9 if the difference is very clear
          - 0.7 if the difference is moderate
          - 0.6 if the difference is slight
          - 0.5 if they are nearly tied (use TIE instead if truly equal)

        Return ONLY a single JSON object with this schema:
        {{
          "winner": "A" | "B" | "TIE",
          "confidence": float  // between 0 and 1
        }}

        TASK:
        {task}

        ANSWER A:
        {A}

        ANSWER B:
        {B}
        """.strip()

        try:
            txt = self.call_model(model, judge_prompt).strip()
            reward = self._parse_judge_response(txt, swapped)
            return reward
        except Exception as e:
            print(f"Warning: Judge call failed: {e}. Using neutral reward.")
            return 0.5

    def _parse_judge_response(self, txt: str, swapped: bool) -> float:
        """Parse judge response and convert to persona-centric reward."""
        s = txt.strip()
        s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

        # Remove code fences like ```json ... ``` or ``` ... ```
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s, flags=re.DOTALL)
            s = re.sub(r"\s*```$", "", s, flags=re.DOTALL).strip()

        # Extract the first {...} block
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object found in: {s[:120]!r}")
        
        obj = json.loads(m.group(0).strip())
        winner = str(obj.get("winner", "TIE")).upper()
        conf = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))

        # Map to persona-centric reward
        if winner == "TIE":
            return 0.5
        elif winner == "A":
            persona_wins = swapped
            return 0.5 + 0.5 * conf if persona_wins else 0.5 - 0.5 * conf
        elif winner == "B":
            persona_wins = not swapped
            return 0.5 + 0.5 * conf if persona_wins else 0.5 - 0.5 * conf
        else:
            return 0.5


# Backward compatibility functions
def call_openai(model: str, prompt: str) -> str:
    """Backward compatibility function for call_openai."""
    from config import Config
    config = Config.from_env()
    client = OpenAIClient(
        api_key=config.openai_api_key,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        dry_run=config.dry_run
    )
    return client.call_model(model, prompt)


def persona_prompt(name: str, task: str) -> str:
    """Backward compatibility function for persona_prompt."""
    client = OpenAIClient()
    return client.persona_prompt(name, task)


def judge_response(model: str, task: str, persona_response: str, baseline_response: str) -> float:
    """Backward compatibility function for judge_response."""
    from config import Config
    config = Config.from_env()
    client = OpenAIClient(
        api_key=config.openai_api_key,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        dry_run=config.dry_run
    )
    return client.judge_response(model, task, persona_response, baseline_response)