from __future__ import annotations

import json
import os

OUTCOMES = os.environ.get("TRACK3_OUTCOMES", "/tmp/track3_outcomes.json")


def load() -> dict:
    if not os.path.isfile(OUTCOMES):
        import pytest
        pytest.skip(f"no compiled outcomes at {OUTCOMES}; verifier did not emit them for this run",
                    allow_module_level=False)
    with open(OUTCOMES) as f:
        return json.load(f)
