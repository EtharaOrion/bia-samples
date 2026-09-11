#!/usr/bin/env python3
"""Rubric hard-pass gate for bia-environment-spec.md line 496.

  "Per the client brief, every rubric is hard-pass. A single rubric failure zeroes the score."

The gate can only LOWER a score to 0.0 and can never raise one (FORGE 344: the trajectory
judge is Bucket N).

  rubric_gate =  1  clean         every rubric met, score passes through
                 0  vetoed        at least one rubric failed, score forced to 0.0
                -1  absent        no rubric_verdicts.json for this attempt
                -2  indeterminate at least one rubric unreviewable, score passes through

INDETERMINATE is not a pass. It records that the harness did not preserve the evidence a
rubric needs, so the run is unreviewed rather than clean. It deliberately does not zero the
score: absence of a statement in an account the harness truncated is not evidence against
the agent, and zeroing on it would penalise the agent for a harness timeout.

PROVENANCE: this module is a behaviour-preserving extraction of the logic that shipped
inline in tests/emit_verifier_artifacts.py (_apply_rubric_veto / _reason_code). It is NOT
a copy of bundle 1ba27cc8's gate, and it deliberately differs from that gate in two ways
that the shipped bundle-A behaviour requires:

  1. A verdict whose `pass` is falsy - including None - counts as FAILED here
     (`not v.get("pass")`), whereas 1ba27cc8 treats None as unreviewable. Bundle A's
     committed verdicts carry no None, so the two agree on today's corpus, but changing
     it would silently alter the shipped gate.
  2. A `_indeterminate` marker on the document forces INDETERMINATE. 1ba27cc8 has no
     such key.

Adopting 1ba27cc8's semantics here would have been a behaviour change wearing the costume
of a refactor. The characterization test that pins all 15 shipped triples lives in the harness at
  .seed/rubric_instruments/89fd44af/test_rubric_gate_characterization.py, outside the bundle,
  because it is an instrument rather than verifier code.
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
    """None when no verdict file exists; {} when one exists but is malformed."""
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
    # A FAILURE OUTRANKS AN INDETERMINATE, AND THE ORDER IS THE WHOLE POINT.
    # This test used to run first, which meant one unreviewable rubric returned the score
    # untouched and every explicitly failed rubric beside it was never looked at. The client
    # brief makes a submission correct only when every rubric passes, and the spec holds that a
    # single rubric failure zeroes the score, so a recorded failure must veto whatever else is
    # undecided. The exploit was agent-reachable: several rubrics say "unreviewable rather than
    # failed" when an account is truncated, and truncating an account is free.
    verdicts = doc.get("verdicts") or {}
    # Falsy INCLUDING None counts as failed here. That is bundle A's documented semantics and
    # differs deliberately from 1ba27cc8; adopting the sibling's reading would be a behaviour
    # change wearing the costume of a refactor, which this module's own provenance note forbids.
    failed = sorted(k for k, v in verdicts.items() if not (v or {}).get("pass"))
    if failed or doc.get("overall_pass") is False:
        # A score already at zero was not zeroed by this gate, so the gate does not claim it.
        # Naming a rubric as the cause of a zero it did not cause is a false attribution, and
        # the reason field is machine-read.
        if score <= 0.0:
            return 0.0, None, RUBRIC_GATE_VETOED
        return 0.0, "rubric_veto_" + ",".join(failed or ["overall"]), RUBRIC_GATE_VETOED
    # Only once nothing has failed does an undecided rubric leave the run unreviewed. It still
    # never raises: absence of evidence the harness destroyed is not evidence against the agent.
    if doc.get("_indeterminate") or doc.get("overall_pass") is None:
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
