"""Reads the outcome map grade.py wrote, so the compiled tests assert bytes."""

from __future__ import annotations

import json
import os

OUTCOMES_PATH = os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json")


def load():
    with open(OUTCOMES_PATH) as handle:
        return json.load(handle)
