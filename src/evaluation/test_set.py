"""Load validated test sets from disk."""

from __future__ import annotations

import json
from pathlib import Path

from models.evaluation import TestSet

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_SET_PATH = ROOT / "data" / "evaluation" / "test_set.json"


def load_test_set(path: Path = TEST_SET_PATH) -> TestSet:
    """Load and validate the test set from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Test set not found at {path}. "
            "Run scripts/generate_test_set.py first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TestSet.model_validate(raw)
