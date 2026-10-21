#!/usr/bin/env python3
"""Rubric hard-pass gate for bia-environment-spec.md line 496.

  "Per the client brief, every rubric is hard-pass. A single rubric failure zeroes the score."

The gate can only LOWER a score to 0.0 and can never raise one (FORGE 344: the trajectory
judge is Bucket N). Gate vocabulary follows the shipped implementation in bundle
89fd44af tests/emit_verifier_artifacts.py so a reader meets one convention, not two.

  rubric_gate =  1  clean         every rubric met, score passes through
                 0  vetoed        at least one rubric failed, score forced to 0.0
                -1  absent        no rubric_verdicts.json for this attempt
                -2  indeterminate at least one rubric unreviewable, score passes through

INDETERMINATE is not a pass. It records that the harness did not preserve the evidence a
rubric needs, so the run is unreviewed rather than clean. It deliberately does not zero the
score: absence of a statement in an account the harness truncated is not evidence against
the agent, and zeroing on it would penalise the agent for a harness timeout.
"""
from __future__ import annotations

import json
import pathlib

RUBRIC_GATE_CLEAN = 1
RUBRIC_GATE_VETOED = 0
RUBRIC_GATE_ABSENT = -1
RUBRIC_GATE_INDETERMINATE = -2

REASON_CODE_RUBRIC_VETO = 95


def load_verdicts(run: pathlib.Path):
    path = pathlib.Path(run) / "rubric_verdicts.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def apply_rubric_veto(score: float, doc):
    """Return (score, veto_reason, rubric_gate). Never raises the score."""
    if doc is None:
        return score, None, RUBRIC_GATE_ABSENT
    if not doc:
        return score, None, RUBRIC_GATE_INDETERMINATE
    verdicts = doc.get("verdicts") or {}
    failed = sorted(k for k, v in verdicts.items() if (v or {}).get("pass") is False)
    if failed:
        return 0.0, "rubric_veto_" + ",".join(failed), RUBRIC_GATE_VETOED
    if doc.get("overall_pass") is False:
        return 0.0, "rubric_veto_overall", RUBRIC_GATE_VETOED
    if doc.get("overall_pass") is None:
        return score, None, RUBRIC_GATE_INDETERMINATE
    return score, None, RUBRIC_GATE_CLEAN


def rubric_counts(doc):
    verdicts = (doc or {}).get("verdicts") or {}
    met = sum(1 for v in verdicts.values() if (v or {}).get("pass") is True)
    failed = sum(1 for v in verdicts.values() if (v or {}).get("pass") is False)
    unreviewable = sum(1 for v in verdicts.values() if (v or {}).get("pass") is None)
    return {"rubrics_passed": met, "rubrics_failed": failed,
            "rubrics_unreviewable": unreviewable, "rubrics_total": len(verdicts),
            "rubrics_fraction": round(met / len(verdicts), 6) if verdicts else 0.0}


def augment(score_doc: dict, numeric_doc: dict, full_doc: dict, doc) -> tuple:
    """Return (score_doc, numeric_doc, full_doc) carrying the rubric outcome."""
    graded = float(score_doc.get("score", 0.0))
    score, veto_reason, gate = apply_rubric_veto(graded, doc)
    counts = rubric_counts(doc)
    score_doc = dict(score_doc)
    score_doc["score"] = score
    score_doc["graded_score"] = graded
    score_doc["rubric_gate"] = gate
    if veto_reason:
        score_doc["reason"] = veto_reason
        score_doc["reason_code"] = REASON_CODE_RUBRIC_VETO
    numeric_doc = dict(numeric_doc)
    numeric_doc["score"] = score
    numeric_doc["graded_score"] = graded
    numeric_doc["rubric_gate"] = gate
    numeric_doc.update(counts)
    if veto_reason:
        numeric_doc["reason_code"] = REASON_CODE_RUBRIC_VETO
    full_doc = dict(full_doc)
    full_doc["score"] = score
    full_doc["graded_score"] = graded
    full_doc["rubrics"] = dict(counts, gate=gate, veto_reason=veto_reason,
                               failed=sorted(k for k, v in ((doc or {}).get("verdicts") or {}).items()
                                             if (v or {}).get("pass") is False))
    full_doc["note"] = ("graded_score is what grade.py computed from the telemetry. score is "
                        "that value after the rubric veto, which can only lower it to 0.0 and "
                        "can never raise it. rubric_gate is 1 clean, 0 vetoed, -1 no verdict, "
                        "-2 indeterminate; any negative value is an unreviewed run, not a "
                        "clean one.")
    return score_doc, numeric_doc, full_doc
