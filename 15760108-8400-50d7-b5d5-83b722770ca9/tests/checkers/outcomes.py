"""Loader for the outcomes map grade.py writes.

test_output.py asserts against this map. It never re-derives an outcome, so a
pytest pass can never disagree with the score grade.py recorded.
"""

from __future__ import annotations

import json
import os
import pathlib

DEFAULT = "/logs/verifier/outcomes.json"


def path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("BIA_S09_OUTCOMES", DEFAULT))


def load() -> dict:
    p = path()
    if not p.is_file():
        raise FileNotFoundError(f"no outcomes at {p}; grade.py did not run")
    return json.loads(p.read_text())
