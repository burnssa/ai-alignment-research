"""
Opus Annotation Pipeline for Constitutional Principle Extraction

This module uses Claude Opus to extract principle weights from actual SCOTUS
majority opinions. The key insight: we're doing EXTRACTION (what did justices
actually invoke?) not JUDGMENT (what should matter?).

Ground truth comes from human-written opinions, not model predictions.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import anthropic

# The five constitutional principles we're tracking
PRINCIPLES = [
    "free_expression",
    "equal_protection",
    "due_process",
    "federalism",
    "privacy_liberty"
]

EXTRACTION_PROMPT = """You are a constitutional law expert analyzing a Supreme Court majority opinion.

Your task is to determine how heavily the majority opinion relies on each of the following five constitutional principles. Base your assessment ONLY on what the opinion actually says—the doctrines invoked, precedents cited, and reasoning employed.

THE FIVE PRINCIPLES:

1. FREE EXPRESSION (First Amendment)
   - Speech, press, assembly, association rights
   - Content-based vs. content-neutral analysis
   - Symbolic speech, compelled speech
   - Key doctrines: strict scrutiny for content-based, time/place/manner, incitement test

2. EQUAL PROTECTION (Fourteenth Amendment)
   - Discrimination based on race, sex, other classifications
   - Levels of scrutiny: strict, intermediate, rational basis
   - Disparate treatment vs. disparate impact
   - Key doctrines: suspect classifications, fundamental rights triggering heightened review

3. DUE PROCESS (Fifth/Fourteenth Amendment)
   - Procedural due process: notice, hearing, neutral decision-maker
   - Substantive due process: fundamental rights, liberty interests
   - Incorporation of Bill of Rights against states
   - Key doctrines: Mathews balancing, fundamental rights analysis

4. FEDERALISM (Tenth Amendment, structural)
   - Federal vs. state power allocation
   - Commerce Clause limits, spending power conditions
   - Anti-commandeering, state sovereign immunity
   - Key doctrines: enumerated powers, reserved powers, preemption

5. PRIVACY/LIBERTY (Penumbras, Ninth/Fourteenth Amendment)
   - Unenumerated rights, bodily autonomy
   - Intimate association, family decisions
   - Right to refuse medical treatment
   - Key doctrines: Griswold penumbras, substantive due process liberty

CASE: {case_name} ({year})

MAJORITY OPINION TEXT:
{opinion_text}

INSTRUCTIONS:
1. Read the opinion carefully
2. Identify which principles the majority ACTUALLY invokes and relies upon
3. Assign a weight from 0.0 to 1.0 for each principle based on its CENTRALITY to the reasoning
   - 0.0 = Not mentioned or invoked at all
   - 0.1-0.3 = Mentioned briefly, not central to holding
   - 0.4-0.6 = Significant component of the analysis
   - 0.7-0.9 = Major pillar of the reasoning
   - 1.0 = The dominant, controlling principle
4. Weights need not sum to 1.0—a case can heavily invoke multiple principles
5. For each non-zero weight, provide a brief justification with a SHORT quote (under 15 words) from the opinion

Respond with ONLY a JSON object in this exact format:
{{
    "case_name": "{case_name}",
    "weights": {{
        "free_expression": <float>,
        "equal_protection": <float>,
        "due_process": <float>,
        "federalism": <float>,
        "privacy_liberty": <float>
    }},
    "justifications": {{
        "free_expression": "<explanation with short quote or null if weight is 0>",
        "equal_protection": "<explanation with short quote or null if weight is 0>",
        "due_process": "<explanation with short quote or null if weight is 0>",
        "federalism": "<explanation with short quote or null if weight is 0>",
        "privacy_liberty": "<explanation with short quote or null if weight is 0>"
    }},
    "primary_holding": "<one sentence summary of the holding>",
    "confidence_notes": "<any uncertainty about the classification>"
}}"""


@dataclass
class PrincipleAnnotation:
    """Structured annotation for a single case."""
    case_id: str
    case_name: str
    year: int
    weights: dict[str, float]
    justifications: dict[str, Optional[str]]
    primary_holding: str
    confidence_notes: str
    annotator_model: str = "claude-opus-4-5-20251101"
    annotation_timestamp: str = ""
    
    def to_vector(self) -> list[float]:
        """Return weights as ordered vector for ML pipeline."""
        return [self.weights[p] for p in PRINCIPLES]
    
    def validate(self) -> list[str]:
        """Check for potential issues with annotation."""
        issues = []
        
        # Check all principles present
        for p in PRINCIPLES:
            if p not in self.weights:
                issues.append(f"Missing weight for {p}")
            elif not 0.0 <= self.weights[p] <= 1.0:
                issues.append(f"Weight for {p} out of range: {self.weights[p]}")
        
        # Check for at least one non-zero weight
        if all(w == 0.0 for w in self.weights.values()):
            issues.append("All weights are zero—unusual for SCOTUS case")
        
        # Check justifications exist for non-zero weights
        for p, w in self.weights.items():
            if w > 0 and (p not in self.justifications or not self.justifications[p]):
                issues.append(f"Non-zero weight for {p} but no justification")
        
        return issues


class OpusAnnotator:
    """Annotator that uses Claude Opus to extract principle weights."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from param or environment."""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-opus-4-5-20251101"
    
    def annotate_case(
        self, 
        case_id: str,
        case_name: str,
        year: int,
        opinion_text: str,
        max_retries: int = 3
    ) -> PrincipleAnnotation:
        """
        Extract principle weights from a majority opinion.
        
        Args:
            case_id: Unique identifier for the case
            case_name: Full case name
            year: Year decided
            opinion_text: Full text of majority opinion
            max_retries: Number of retry attempts on failure
            
        Returns:
            PrincipleAnnotation with extracted weights
        """
        prompt = EXTRACTION_PROMPT.format(
            case_name=case_name,
            year=year,
            opinion_text=opinion_text[:50000]  # Truncate if extremely long
        )
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Parse JSON response
                response_text = response.content[0].text
                
                # Handle potential markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                data = json.loads(response_text.strip())
                
                annotation = PrincipleAnnotation(
                    case_id=case_id,
                    case_name=case_name,
                    year=year,
                    weights=data["weights"],
                    justifications=data["justifications"],
                    primary_holding=data["primary_holding"],
                    confidence_notes=data.get("confidence_notes", ""),
                    annotator_model=self.model,
                    annotation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                )
                
                # Validate
                issues = annotation.validate()
                if issues:
                    print(f"  Warning: Validation issues for {case_id}: {issues}")
                
                return annotation
                
            except json.JSONDecodeError as e:
                print(f"  Attempt {attempt + 1}: JSON parse error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
            except Exception as e:
                print(f"  Attempt {attempt + 1}: Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        raise RuntimeError(f"Failed to annotate {case_id} after {max_retries} attempts")


def save_annotations(annotations: list[PrincipleAnnotation], filepath: str):
    """Save annotations to JSON file."""
    data = [asdict(a) for a in annotations]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(annotations)} annotations to {filepath}")


def load_annotations(filepath: str) -> list[PrincipleAnnotation]:
    """Load annotations from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [PrincipleAnnotation(**d) for d in data]


def create_dataset_for_probing(annotations: list[PrincipleAnnotation]) -> dict:
    """
    Convert annotations to format needed for linear probing.
    
    Returns:
        {
            "case_ids": [...],
            "principle_vectors": [[...], [...], ...],  # N x 5 matrix
            "principle_names": ["free_expression", ...]
        }
    """
    return {
        "case_ids": [a.case_id for a in annotations],
        "principle_vectors": [a.to_vector() for a in annotations],
        "principle_names": PRINCIPLES
    }


# === Cross-validation with other models ===

CROSS_VALIDATION_PROMPT = """You are reviewing a constitutional law annotation. Another AI extracted principle weights from a Supreme Court opinion. Your job is to verify the extraction is accurate.

CASE: {case_name} ({year})

EXTRACTED WEIGHTS:
{weights_json}

JUSTIFICATIONS PROVIDED:
{justifications_json}

OPINION EXCERPT (first 10000 chars):
{opinion_excerpt}

Please assess:
1. Are the weights reasonable given the opinion's actual content?
2. Are any principles over-weighted or under-weighted?
3. Are the justifications accurate (do the cited elements actually appear)?

Respond with JSON:
{{
    "overall_assessment": "accurate" | "minor_issues" | "significant_issues",
    "suggested_adjustments": {{
        "<principle>": {{"current": <float>, "suggested": <float>, "reason": "<why>"}}
    }},
    "verification_notes": "<any other observations>"
}}"""


class CrossValidator:
    """Validate annotations using a different model."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5-20250929"):  # Sonnet for cross-validation
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def validate_annotation(
        self,
        annotation: PrincipleAnnotation,
        opinion_text: str
    ) -> dict:
        """
        Cross-validate an annotation using a different model.
        
        Returns validation results with any suggested adjustments.
        """
        prompt = CROSS_VALIDATION_PROMPT.format(
            case_name=annotation.case_name,
            year=annotation.year,
            weights_json=json.dumps(annotation.weights, indent=2),
            justifications_json=json.dumps(annotation.justifications, indent=2),
            opinion_excerpt=opinion_text[:10000]
        )
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        return json.loads(response_text.strip())


# === Example usage ===

if __name__ == "__main__":
    # Demo with a placeholder
    print("Opus Annotation Pipeline")
    print("=" * 50)
    print("\nThis module provides:")
    print("1. OpusAnnotator - Extract principle weights from opinions")
    print("2. CrossValidator - Verify annotations with another model")
    print("3. Utilities for saving/loading annotation datasets")
    print("\nUsage:")
    print("  annotator = OpusAnnotator()")
    print("  annotation = annotator.annotate_case(case_id, name, year, opinion_text)")
    print("  save_annotations([annotation], 'annotations.json')")
