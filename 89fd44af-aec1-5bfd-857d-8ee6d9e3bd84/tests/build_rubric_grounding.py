#!/usr/bin/env python3
"""Build/refresh tests/rubric_judgment_grounding.json against the LIVE rubric set.

Why this exists as code rather than as a hand-written JSON: `basis` records WHY a verdict
is believed, and a hand-assigned basis is an unauditable assertion. Here it is produced by
a documented, deterministic classifier that any reader can re-run.

MERGE-PRESERVING and IDEMPOTENT. tests/rubrics.jsonl is owned by a concurrent lane, so:
  * an id already present in the grounding is LEFT EXACTLY AS IS - this tool never
    overwrites an authored verdict, evidence string or basis;
  * an id that is new and already carries a committed verdict is TRANSCRIBED verbatim;
  * an id that is new and carries no verdict becomes an explicit null placeholder for
    whoever owns the judging lane;
  * an id removed from rubrics.jsonl is dropped.

`overall_pass` is DERIVED from the items, never copied, so the grounding is internally
consistent and tests/regenerate_rubric_verdicts.py cannot be handed a contradiction.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import account as AC  # noqa: E402

GROUNDING = HERE / "rubric_judgment_grounding.json"
SUMMARIES = HERE / "rubric_judgment_summaries.json"

# One-sided / environmental outcomes. FORGE:328 - INERT, never a basis for a bound verdict.
# telemetry_chain_verified is False on every attempt only because tests/grade.py's
# check_telemetry_chain returns False when TRACK3_CHAIN_KEY is absent from the environment
# ("chain_key_absent_cannot_verify"); that is the harness, not the agent. And
# full_score_target_reached is an OUTCOME, not a behaviour, so it cannot ground a
# behavioural rubric either.
INERT = {"telemetry_chain_verified", "full_score_target_reached"}


def outcome_keys() -> set:
    for p in sorted(ROOT.glob("trajectories/*/*/verifier/outcomes.json")):
        return set(json.loads(p.read_text(encoding="utf-8")))
    return set()


def historical_rubric_text() -> str:
    """Pre-split parent rubric wording (commit 14df90a). Same exclusion C10 applies."""
    p = HERE / "rubrics_historical_14df90a.jsonl"
    if not p.is_file():
        return ""
    return " ".join(json.loads(l)["rubric"]
                    for l in p.read_text(encoding="utf-8").splitlines() if l.strip())


def classify(rubric_id, evidence, verdict, rubric_text, keys, historical=""):
    """Label the PROVENANCE of an already-decided verdict. Never decides one."""
    if verdict is None:
        return "UNREVIEWABLE_NOT_YET_JUDGED"
    if AC.quoted_fragments(evidence, rubric_text.get(rubric_id, ""), historical):
        return "NARRATION_QUOTE_HAND_SELECTED"
    named = {k for k in keys if k in evidence}
    if named and not (named <= INERT):
        return "DETERMINISTIC_CHECKER"
    if named:
        return "UNREVIEWABLE_INERT_CHECK"
    return "ARTIFACT_RECORD"


def derive_overall(items):
    verdicts = [i["verdict"] for i in items.values()]
    if any(v is False for v in verdicts):
        return False
    if all(v is True for v in verdicts):
        return True
    return None


def main() -> int:
    ids = [json.loads(l)["id"]
           for l in (HERE / "rubrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rubric_text = {json.loads(l)["id"]: json.loads(l)["rubric"]
                   for l in (HERE / "rubrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    keys = outcome_keys()
    historical = historical_rubric_text()
    prior = json.loads(GROUNDING.read_text(encoding="utf-8")) if GROUNDING.is_file() else {}
    summaries = json.loads(SUMMARIES.read_text(encoding="utf-8")) if SUMMARIES.is_file() else {}

    grounding, added, kept = {}, 0, 0
    for d in sorted(ROOT.glob("trajectories/*/*")):
        if not d.is_dir():
            continue
        key = "%s/%s" % (d.parent.name, d.name)
        committed = {}
        vp = d / "rubric_verdicts.json"
        if vp.is_file():
            doc = json.loads(vp.read_text(encoding="utf-8"))
            committed = doc.get("verdicts") or {}
            summaries.setdefault(key, doc.get("summary", ""))
        prior_items = (prior.get(key) or {}).get("items") or {}
        items = {}
        for rid in ids:
            if rid in prior_items:
                items[rid] = prior_items[rid]
                kept += 1
            elif rid in committed:
                v = committed[rid]
                items[rid] = {"basis": classify(rid, v["evidence"], v["pass"], rubric_text, keys,
                                                historical),
                              "evidence": v["evidence"], "verdict": v["pass"]}
                added += 1
            else:
                items[rid] = {"basis": "UNREVIEWABLE_NOT_YET_JUDGED", "evidence": "", "verdict": None}
                added += 1
        grounding[key] = {"account_channel": "FINAL_MESSAGE" if (d / "agent" / "history.md").is_file()
                          else "NO_PRIOR_HISTORY",
                          "account_chars": len(AC.account(d)),
                          "items": items,
                          "overall_pass": derive_overall(items)}

    with open(GROUNDING, "w", encoding="utf-8", newline="\n") as h:
        h.write(json.dumps(grounding, indent=2, sort_keys=True) + "\n")
    with open(SUMMARIES, "w", encoding="utf-8", newline="\n") as h:
        h.write(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    sys.stdout.write("grounding: %d attempts x %d rubrics (%d preserved, %d new)\n"
                     % (len(grounding), len(ids), kept, added))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
