#!/usr/bin/env python3
"""
Test script to verify HW1 setup is working correctly.

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.policy.prefix_policy import PromptPrefixPolicy
        print("  ✓ PromptPrefixPolicy")
    except ImportError as e:
        print(f"  ✗ PromptPrefixPolicy: {e}")
        return False

    try:
        from src.benchmarks.loader import BenchmarkLoader
        print("  ✓ BenchmarkLoader")
    except ImportError as e:
        print(f"  ✗ BenchmarkLoader: {e}")
        return False

    try:
        from src.benchmarks.evaluator import BenchmarkEvaluator
        print("  ✓ BenchmarkEvaluator")
    except ImportError as e:
        print(f"  ✗ BenchmarkEvaluator: {e}")
        return False

    try:
        from src.training.config import TrainingConfig
        print("  ✓ TrainingConfig")
    except ImportError as e:
        print(f"  ✗ TrainingConfig: {e}")
        return False

    try:
        from src.training.trainer import REINFORCETrainer
        print("  ✓ REINFORCETrainer")
    except ImportError as e:
        print(f"  ✗ REINFORCETrainer: {e}")
        return False

    try:
        from src.utils.query_utils import ModelQueryInterface
        print("  ✓ ModelQueryInterface")
    except ImportError as e:
        print(f"  ✗ ModelQueryInterface: {e}")
        return False

    return True


def test_people_csv():
    """Test that notable_people_10k.csv exists and is valid."""
    print("\nTesting notable_people_10k.csv...")

    csv_path = Path("notable_people_10k.csv")

    if not csv_path.exists():
        print(f"  ✗ File not found: {csv_path}")
        return False

    print(f"  ✓ File exists: {csv_path}")

    import csv

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            people = list(reader)

        print(f"  ✓ Loaded {len(people)} people")

        if len(people) == 0:
            print("  ✗ CSV is empty")
            return False

        # Check required fields
        required_fields = ["name", "field", "era"]
        first_person = people[0]

        for field in required_fields:
            if field not in first_person:
                print(f"  ✗ Missing required field: {field}")
                return False

        print(f"  ✓ Has required fields: {required_fields}")

        # Show sample
        print(f"\n  Sample people:")
        for person in people[:3]:
            print(f"    - {person['name']} ({person['field']}, {person['era']})")

        return True

    except Exception as e:
        print(f"  ✗ Error reading CSV: {e}")
        return False


def test_policy():
    """Test that policy can be initialized."""
    print("\nTesting PromptPrefixPolicy...")

    try:
        from src.policy.prefix_policy import PromptPrefixPolicy

        policy = PromptPrefixPolicy("notable_people_10k.csv", device="cpu")

        print(f"  ✓ Policy initialized with {policy.num_people} people")

        # Test sampling
        prefix, idx, _ = policy.sample_prefix()
        print(f"  ✓ Sampled prefix: '{prefix[:50]}...'")

        # Test distribution
        dist = policy.get_distribution()
        print(f"  ✓ Distribution shape: {dist.shape}")
        print(f"  ✓ Distribution sum: {dist.sum().item():.6f} (should be ~1.0)")

        # Test top people
        top_people = policy.get_top_people(5)
        print(f"\n  Top 5 people (initial uniform):")
        for i, (name, prob, field, era) in enumerate(top_people, 1):
            print(f"    {i}. {name} ({prob:.6f})")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test that config can be created and saved."""
    print("\nTesting TrainingConfig...")

    try:
        from src.training.config import TrainingConfig

        config = TrainingConfig()
        print(f"  ✓ Config created")
        print(f"    - Model: {config.model_name}")
        print(f"    - Benchmark: {config.benchmark_name}")
        print(f"    - Iterations: {config.num_iterations}")

        # Test saving
        test_path = Path("outputs/test_config.json")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        config.save(test_path)
        print(f"  ✓ Config saved to {test_path}")

        # Test loading
        loaded_config = TrainingConfig.from_json(test_path)
        print(f"  ✓ Config loaded from {test_path}")

        # Clean up
        test_path.unlink()

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark_loader():
    """Test that benchmark loader works (without downloading large datasets)."""
    print("\nTesting BenchmarkLoader...")

    try:
        from src.benchmarks.loader import BenchmarkLoader

        loader = BenchmarkLoader()
        print(f"  ✓ BenchmarkLoader created")

        # Check supported benchmarks
        print(f"  ✓ Supported benchmarks: {list(loader.SUPPORTED_BENCHMARKS.keys())}")

        # Get benchmark info (doesn't require download)
        info = loader.get_benchmark_info("gsm8k")
        print(f"  ✓ GSM8K info: {info.get('description', 'N/A')}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HW1 Setup Verification")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("People CSV", test_people_csv),
        ("Policy", test_policy),
        ("Config", test_config),
        ("Benchmark Loader", test_benchmark_loader),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Setup is working correctly.")
        print("\nYou can now run training with:")
        print("  uv run python scripts/train_policy.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
