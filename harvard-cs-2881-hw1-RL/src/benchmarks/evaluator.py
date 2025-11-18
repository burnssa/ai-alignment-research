#!/usr/bin/env python3
"""
Benchmark evaluator for assessing prompt prefix performance.

Evaluates model responses on benchmark questions with different prompt prefixes.
"""

import re
from typing import Dict, List, Optional, Tuple

from ..utils.query_utils import ModelQueryInterface


class BenchmarkEvaluator:
    """Evaluates model performance on benchmarks with prompt prefixes."""

    def __init__(self, model_interface: ModelQueryInterface):
        """
        Initialize evaluator with a model interface.

        Args:
            model_interface: Loaded ModelQueryInterface instance
        """
        self.model_interface = model_interface

    def evaluate_with_prefix(
        self,
        questions: List[Dict],
        prefix: str = "",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        show_progress: bool = True,
    ) -> Dict:
        """
        Evaluate model on benchmark questions with a prompt prefix.

        Args:
            questions: List of question dicts from BenchmarkLoader
            prefix: Prompt prefix (e.g., "You are Albert Einstein. ")
            max_new_tokens: Max tokens for generation
            temperature: Sampling temperature
            show_progress: Whether to print progress during evaluation

        Returns:
            Dict with evaluation results
        """
        correct = 0
        total = len(questions)
        results = []

        for idx, question_data in enumerate(questions, 1):
            if show_progress and idx % 10 == 0:
                # Show progress every 10 questions
                print(f"    Progress: {idx}/{total} questions ({idx/total:.0%}) | Current accuracy: {correct/idx:.1%}", end="\r", flush=True)
            question_type = question_data["type"]

            if question_type == "multiple_choice":
                is_correct, response = self._evaluate_multiple_choice(
                    question_data, prefix, max_new_tokens, temperature
                )
            elif question_type == "free_form":
                is_correct, response = self._evaluate_free_form(
                    question_data, prefix, max_new_tokens, temperature
                )
            else:
                raise ValueError(f"Unknown question type: {question_type}")

            if is_correct:
                correct += 1

            results.append({
                "question": question_data["question"],
                "correct": is_correct,
                "response": response,
                "correct_answer": question_data.get("answer_text", ""),
            })

        # Clear progress line
        if show_progress:
            print()  # Move to new line after progress indicator

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }

    def _evaluate_multiple_choice(
        self,
        question_data: Dict,
        prefix: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[bool, str]:
        """
        Evaluate a multiple choice question.

        Args:
            question_data: Question dict with 'question', 'choices', 'correct_answer'
            prefix: Prompt prefix
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (is_correct, response)
        """
        question = question_data["question"]
        choices = question_data["choices"]
        correct_idx = question_data["correct_answer"]

        # Format multiple choice question
        formatted_question = self._format_multiple_choice(question, choices)

        # Query model with prefix as system prompt
        response = self.model_interface.query_model(
            prompt=formatted_question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=prefix if prefix else None,
        )

        # Extract answer from response
        predicted_idx = self._extract_choice(response, len(choices))

        is_correct = predicted_idx == correct_idx

        return is_correct, response

    def _evaluate_free_form(
        self,
        question_data: Dict,
        prefix: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[bool, str]:
        """
        Evaluate a free-form question (e.g., math problems).

        Args:
            question_data: Question dict with 'question', 'correct_answer'
            prefix: Prompt prefix
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (is_correct, response)
        """
        question = question_data["question"]
        correct_answer = question_data["correct_answer"]

        # Query model
        response = self.model_interface.query_model(
            prompt=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=prefix if prefix else None,
        )

        # Extract numeric answer from response (for math problems)
        predicted_answer = self._extract_numeric_answer(response)

        # Normalize and compare
        is_correct = self._compare_answers(predicted_answer, correct_answer)

        return is_correct, response

    def _format_multiple_choice(
        self, question: str, choices: List[str]
    ) -> str:
        """
        Format multiple choice question with labeled choices.

        Args:
            question: Question text
            choices: List of choice strings

        Returns:
            Formatted question string
        """
        choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        formatted = f"{question}\n\n"

        for i, choice in enumerate(choices):
            label = choice_labels[i] if i < len(choice_labels) else str(i)
            formatted += f"{label}. {choice}\n"

        formatted += "\nAnswer with just the letter (A, B, C, etc.):"

        return formatted

    def _extract_choice(self, response: str, num_choices: int) -> Optional[int]:
        """
        Extract choice index from model response.

        Args:
            response: Model response text
            num_choices: Number of choices available

        Returns:
            Index of chosen answer (0-indexed), or None if unclear
        """
        # Look for letter answers (A, B, C, D, etc.)
        choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

        # Try to find letter in response (case insensitive)
        response_upper = response.upper().strip()

        # First, try to find a standalone letter
        for i in range(min(num_choices, len(choice_labels))):
            label = choice_labels[i]
            # Look for the letter at the start or as a standalone word
            if re.search(rf'\b{label}\b', response_upper):
                return i

        # If no clear letter, try to find a number (1-indexed in response)
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            try:
                num = int(numbers[0])
                if 1 <= num <= num_choices:
                    return num - 1  # Convert to 0-indexed
            except ValueError:
                pass

        # Default to first choice if unclear
        return 0

    def _extract_numeric_answer(self, response: str) -> str:
        """
        Extract numeric answer from response (for math problems).

        Args:
            response: Model response

        Returns:
            Extracted numeric answer as string
        """
        # Look for numbers in the response
        # Common patterns: "The answer is 42", "#### 42", "= 42"

        # Try to find answer after common markers
        patterns = [
            r'####\s*([0-9,.$]+)',  # GSM8K format
            r'answer is\s*([0-9,.$]+)',
            r'equals\s*([0-9,.$]+)',
            r'=\s*([0-9,.$]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern match, extract the last number in response
        numbers = re.findall(r'[0-9,.$]+', response)
        if numbers:
            return numbers[-1]

        return ""

    def _compare_answers(self, predicted: str, correct: str) -> bool:
        """
        Compare predicted and correct answers (with normalization).

        Args:
            predicted: Predicted answer string
            correct: Correct answer string

        Returns:
            True if answers match
        """
        # Normalize both answers
        def normalize(s: str) -> str:
            # Remove commas, dollar signs, spaces
            s = s.replace(",", "").replace("$", "").replace(" ", "")
            return s.strip().lower()

        pred_norm = normalize(predicted)
        correct_norm = normalize(correct)

        return pred_norm == correct_norm

    def evaluate_batch(
        self,
        questions: List[Dict],
        prefix: str = "",
        batch_size: int = 10,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> Dict:
        """
        Evaluate on a random batch of questions (for faster iteration).

        Args:
            questions: Full list of questions
            prefix: Prompt prefix
            batch_size: Number of questions to evaluate
            max_new_tokens: Max tokens for generation
            temperature: Sampling temperature

        Returns:
            Dict with evaluation results on batch
        """
        import random

        # Sample batch
        if len(questions) > batch_size:
            batch = random.sample(questions, batch_size)
        else:
            batch = questions

        return self.evaluate_with_prefix(
            batch, prefix, max_new_tokens, temperature
        )
