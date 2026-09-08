#!/usr/bin/env python3
"""Regenerate every trajectories/*/*/rubric_verdicts.json from frozen grounding.

Determinism contract (FORGE 10f / G-RUB-REGEN):
  * Reads exactly two frozen inputs, both committed beside this script:
      tests/rubric_judgment_grounding.json   verdicts, evidence, basis, account channel
      tests/rubric_judgment_summaries.json   the judge's per-attempt prose
    Bundle 1ba27cc8 folds its summary into a single template and so needs one input.
    Bundle A's shipped summaries are bespoke per attempt rather than templated, so
    reproducing the committed bytes requires carrying them as literals. Collapsing them
    into a template would have rewritten 15 shipped judgements to make a checker green.
  * Never invokes a model, network, clock, locale or random source.
  * Judgment prose was authored by a judge at authoring time and frozen as literals;
    this script only transcribes those literals.
  * Output byte-order is fixed by sort_keys=True, indent=2, ensure_ascii=True (matching
    the committed bytes, which carry \\uXXXX escapes) and a single trailing newline, so
    two hosts produce identical bytes.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
GROUNDING = HERE / "rubric_judgment_grounding.json"
SUMMARIES = HERE / "rubric_judgment_summaries.json"


def overall_pass(items: dict) -> object:
    verdicts = [i["verdict"] for i in items.values()]
    if any(v is False for v in verdicts):
        return False
    if all(v is True for v in verdicts):
        return True
    return None


def main() -> int:
    grounding = json.loads(GROUNDING.read_text(encoding="utf-8"))
    summaries = json.loads(SUMMARIES.read_text(encoding="utf-8"))
    written = 0
    for key in sorted(grounding):
        entry = grounding[key]
        cohort, iteration = key.split("/")
        items = entry["items"]
        derived = overall_pass(items)
        if derived != entry["overall_pass"]:
            sys.stderr.write("grounding overall_pass disagrees with its items: %s\n" % key)
            return 1
        verdicts = {r: {"evidence": items[r]["evidence"], "pass": items[r]["verdict"]}
                    for r in sorted(items)}
        document = {"overall_pass": derived,
                    "summary": summaries[key],
                    "verdicts": verdicts}
        target = ROOT / "trajectories" / cohort / iteration / "rubric_verdicts.json"
        if not target.parent.is_dir():
            sys.stderr.write("missing attempt directory: %s\n" % target.parent)
            return 1
        payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        written += 1
    sys.stdout.write("regenerated %d rubric_verdicts.json\n" % written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
