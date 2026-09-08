#!/usr/bin/env python3
"""Pins the shipped gate behaviour across the extraction.

Captured from the inline implementation BEFORE tests/rubric_gate.py existed; the
extraction is a refactor only if these 15 triples are unchanged.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import rubric_gate as RG  # noqa: E402

BASELINE = json.loads((HERE / "rubric_gate_baseline.json").read_text(encoding="utf-8"))


def _graded_score(run: pathlib.Path) -> float:
    path = run / "verifier" / "grade-stdout.md"
    if not path.is_file():
        return 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return float(json.loads(line).get("score", 0.0))
            except json.JSONDecodeError:
                pass
    return 0.0


def test_gate_triples_unchanged_by_extraction():
    seen = 0
    for run in sorted(ROOT.glob("trajectories/*/*")):
        if not run.is_dir():
            continue
        key = "%s/%s" % (run.parent.name, run.name)
        graded = _graded_score(run)
        score, veto_reason, gate = RG.apply_rubric_veto(graded, RG.load_verdicts(run))
        want = BASELINE[key]
        assert graded == want["graded_score"], (key, "graded", graded, want["graded_score"])
        assert score == want["score"], (key, "score", score, want["score"])
        assert veto_reason == want["veto_reason"], (key, "veto", veto_reason, want["veto_reason"])
        assert gate == want["rubric_gate"], (key, "gate", gate, want["rubric_gate"])
        seen += 1
    assert seen == 15, seen


if __name__ == "__main__":
    test_gate_triples_unchanged_by_extraction()
    print("characterization OK: 15/15 gate triples identical after extraction")
