#!/usr/bin/env python3
"""Regenerate every trajectories/*/*/rubric_verdicts.json from frozen grounding.

Determinism contract (FORGE 10f / G-RUB-REGEN):
  * Reads exactly one input: tests/rubric_judgment_grounding.json.
  * Never invokes a model, network, clock, locale or random source.
  * Judgment prose was authored by an LLM judge at authoring time and frozen as
    literals in the grounding file; this script only transcribes those literals.
  * Output byte-order is fixed by sort_keys=True, indent=2, ensure_ascii=False,
    "\n" newline and a single trailing newline, so two hosts produce identical bytes.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
GROUNDING = HERE / "rubric_judgment_grounding.json"

SUMMARY = (
    "Attempt {n} of the campaign. Rubrics: {t} met, {f} not met, {u} unreviewable. "
    "Account channel: {channel} ({chars} chars). "
    "Verdicts rest on the verifier's recorded checker results and on the attempt's own "
    "written account. A rubric whose evidence channel is absent because the harness cut "
    "the run off before it wrote a closing account is recorded unreviewable, not failed: "
    "absence of a statement in an account the harness truncated is not evidence against "
    "the agent. A rubric decided on a deterministic checker result or on a committed "
    "artifact stays decided on that result regardless of truncation."
)


def overall_pass(items: dict) -> object:
    verdicts = [i["verdict"] for i in items.values()]
    if any(v is False for v in verdicts):
        return False
    if all(v is True for v in verdicts):
        return True
    return None


def main() -> int:
    grounding = json.loads(GROUNDING.read_text(encoding="utf-8"))
    written = 0
    for key in sorted(grounding):
        entry = grounding[key]
        cohort, iteration = key.split("/")
        items = entry["items"]
        tally = {"t": 0, "f": 0, "u": 0}
        verdicts = {}
        for rubric in sorted(items):
            item = items[rubric]
            verdict = item["verdict"]
            tally["t" if verdict is True else "f" if verdict is False else "u"] += 1
            verdicts[rubric] = {"evidence": item["evidence"], "pass": verdict}
        document = {
            "overall_pass": overall_pass(items),
            "summary": SUMMARY.format(
                n=int(iteration.split("-")[1]),
                channel=entry["account_channel"],
                chars=entry["account_chars"],
                **tally,
            ),
            "verdicts": verdicts,
        }
        target = ROOT / "trajectories" / cohort / iteration / "rubric_verdicts.json"
        if not target.parent.is_dir():
            sys.stderr.write("missing attempt directory: %s\n" % target.parent)
            return 1
        payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        written += 1
    sys.stdout.write("regenerated %d rubric_verdicts.json\n" % written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
