"""
SCOTUS Cases for Constitutional Geometry PoC

Cases are loaded from JSON files in case_data/ directory for easier
inspection and maintenance. Each JSON file represents a phase of the experiment.
"""

import json
from pathlib import Path

# Get the directory containing case data files
CASE_DATA_DIR = Path(__file__).parent / "case_data"


def _load_cases_from_file(filename: str) -> list:
    """Load cases from a JSON file in the case_data directory."""
    filepath = CASE_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Case file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data.get("cases", [])


def _load_all_case_files() -> dict:
    """Load all case files and return metadata + cases."""
    result = {}
    for json_file in sorted(CASE_DATA_DIR.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
        result[json_file.stem] = {
            "phase": data.get("phase"),
            "description": data.get("description"),
            "date_added": data.get("date_added"),
            "source": data.get("source"),
            "cases": data.get("cases", [])
        }
    return result


# Load Phase 1 cases (original 28)
CASES = _load_cases_from_file("phase1_cases.json")

# Load Phase 2 cases (additional 22 for expanded sample)
CASES_PHASE2 = _load_cases_from_file("phase2_cases.json")

# Combine all cases
ALL_CASES = CASES + CASES_PHASE2


# === Convenience Functions ===

def get_case(case_id: str) -> dict:
    """Get a case by its ID from all available cases."""
    for case in ALL_CASES:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"Case not found: {case_id}")


def get_all_case_ids(include_phase2: bool = False) -> list:
    """Get all case IDs."""
    cases = ALL_CASES if include_phase2 else CASES
    return [case["case_id"] for case in cases]


def get_all_cases(include_phase2: bool = False) -> list:
    """Get all cases, optionally including Phase 2."""
    return ALL_CASES if include_phase2 else CASES


def get_cases_by_principle(principle: str, include_phase2: bool = False) -> list:
    """Get all cases with a given primary principle."""
    cases = ALL_CASES if include_phase2 else CASES
    return [c for c in cases if c.get("primary_principle") == principle]


def list_case_files() -> dict:
    """List all available case files with metadata."""
    return _load_all_case_files()


def format_prompt(case: dict) -> str:
    """Format a case into the standard prompt for activation extraction."""
    return f"""Consider the following case before the Supreme Court:

FACTS: {case['facts'].strip()}

LEGAL QUESTION: {case['legal_question'].strip()}

The relevant constitutional principles that may apply include:
- Free Expression (First Amendment speech, press, association)
- Equal Protection (Fourteenth Amendment equal treatment)
- Due Process (Fifth/Fourteenth Amendment procedural and substantive)
- Federalism (state vs. federal power, Tenth Amendment)
- Privacy and Liberty (unenumerated rights, substantive due process liberty interest)

How should the Court rule, and which principles should guide the decision?"""


def print_case_summary():
    """Print a summary of all loaded cases by principle."""
    all_data = _load_all_case_files()

    print("\n" + "=" * 60)
    print("SCOTUS CONSTITUTIONAL GEOMETRY - CASE SUMMARY")
    print("=" * 60)

    for file_key, file_data in all_data.items():
        print(f"\n## {file_key}")
        print(f"   Phase: {file_data['phase']}")
        print(f"   Added: {file_data['date_added']}")
        print(f"   Cases: {len(file_data['cases'])}")
        print(f"   Description: {file_data['description']}")

    print("\n" + "-" * 60)
    print("CASES BY PRINCIPLE")
    print("-" * 60)

    principles = ["free_expression", "equal_protection", "due_process",
                  "federalism", "privacy_liberty"]

    for principle in principles:
        cases = get_cases_by_principle(principle, include_phase2=True)
        print(f"\n### {principle.replace('_', ' ').title()} ({len(cases)} cases)")
        for c in cases:
            print(f"   - {c['case_name']} ({c['year']})")

    print(f"\n" + "=" * 60)
    print(f"TOTAL: {len(ALL_CASES)} cases")
    print("=" * 60)


if __name__ == "__main__":
    # When run directly, print case summary
    print_case_summary()
