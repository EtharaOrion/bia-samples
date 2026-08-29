"""Loader for the outcome vector grade.py writes.

The compiled rubric tests in tests/test_output.py read this file and nothing
else, so a test can never disagree with the score for the same run.
"""

from __future__ import annotations

import json
import os

DEFAULT = "/logs/verifier/outcomes.json"


def path() -> str:
    return os.environ.get("BIA_OUTCOMES", DEFAULT)


def load() -> dict:
    with open(path()) as fh:
        return json.load(fh)
