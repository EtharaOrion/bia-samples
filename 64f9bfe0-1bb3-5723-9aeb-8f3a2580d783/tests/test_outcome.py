"""Post-grade assertions over the reward file the verifier just wrote.

Deliberately thin. The substantive checks live in test_static.py, which runs
without hardware, and in the compiled rubric tests in test_output.py. What is
asserted here is only what must hold of any completed grading run.
"""
from __future__ import annotations

import json
import os
import pathlib

REWARD = pathlib.Path(os.environ.get("BIA_REWARD_PATH", "/logs/verifier/reward.txt"))
RECORD = pathlib.Path(os.environ.get("BIA_REWARD_RECORD", "/logs/verifier/record.json"))


def _p() -> dict:
    return json.loads(RECORD.read_text())


def test_reward_written():
    assert REWARD.is_file(), "an unwritten reward file is an error, never a zero"


def test_bound_reward_is_one_float_agreeing_with_the_record():
    value = float(REWARD.read_text().strip())
    assert 0.0 <= value <= 1.0
    assert abs(value - float(_p()["reward"])) < 1e-6


def test_record_written():
    assert RECORD.is_file(), "a reward with no record is a score with no reason"


def test_reward_bounded():
    p = _p()
    assert isinstance(p["reward"], (int, float))
    assert 0.0 <= p["reward"] <= 1.0


def test_pass_is_binary():
    assert _p()["pass"] in (0, 1)


def test_every_zero_carries_a_reason():
    p = _p()
    if p["reward"] == 0.0:
        assert isinstance(p["reason"], str) and p["reason"].strip()


def test_unmeasured_anchors_cannot_pass():
    p = _p()
    if p["reason"] == "anchors-unmeasured":
        assert p["pass"] == 0
        assert p["reward"] == 0.0
