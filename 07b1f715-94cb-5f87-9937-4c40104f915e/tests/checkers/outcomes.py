"""Shared loader for the compiled checker outcomes the grader emits."""

from __future__ import annotations

import json
import os

OUTCOMES = os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json")


def load() -> dict:
    if not os.path.isfile(OUTCOMES):
        import pytest
        pytest.skip("no compiled outcomes at %s; the verifier did not emit them" % OUTCOMES,
                    allow_module_level=False)
    with open(OUTCOMES) as f:
        return json.load(f)
