#!/usr/bin/env python3
"""Apply the rubric hard-pass gate to the score the verifier just wrote.

bia-environment-spec.md:496 -- "every rubric is hard-pass. A single rubric failure zeroes the
score." The gate only ever LOWERS a score (FORGE item 29: the trajectory judge is Bucket N and
may never raise one). When no rubric_verdicts.json accompanies the run -- the normal case during
a live verifier pass, because the trajectory judge has not run yet -- the gate records ABSENT and
the score passes through untouched.

Rewriting goes through score.write() rather than editing one document, so the numeric document,
the float file and the full document cannot disagree about what the score is.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rubric_gate  # noqa: E402
import score as SC  # noqa: E402


def main() -> int:
    full = SC.SCORE_DIR / SC.SCORE_FULL
    if not full.is_file():
        return 0
    doc = json.loads(full.read_text(encoding="utf-8"))
    run = pathlib.Path(__file__).resolve().parent.parent
    value, reason, gate = rubric_gate.apply_rubric_veto(
        float(doc.get("score") or 0.0), rubric_gate.load_verdicts(run))
    if reason is None:
        return 0
    SC.write(value, reason, doc.get("metric") or {}, doc.get("checkers") or [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
