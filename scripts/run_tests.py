# ============================================================================
# scripts/run_tests.py
# ============================================================================
"""
Quick test runner script

Usage:
    python scripts/run_tests.py
    python scripts/run_tests.py --test-dataset data/test/my_test.csv
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    # Run pytest if available, otherwise run unittest
    try:
        import pytest
        print("Running tests with pytest...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_prefilter.py", "-v"],
            cwd=ROOT
        )
        sys.exit(result.returncode)
    except ImportError:
        print("pytest not found. Running with unittest...")
        result = subprocess.run(
            [sys.executable, "tests/test_prefilter.py"],
            cwd=ROOT
        )
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
