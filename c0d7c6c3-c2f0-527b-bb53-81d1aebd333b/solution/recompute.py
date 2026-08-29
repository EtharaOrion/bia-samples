#!/usr/bin/env python3
"""Derive every generated artifact in this bundle from solution/grounding.yaml.

The executable oracle, the human-readable ground truth, the reference rubric and
the compiled rubric tests all descend from one source, so they cannot disagree
with each other or with the checker fixtures. Nothing here is hand-authored, and
Phase 2 re-runs this script and requires bit-identical output.
"""
from __future__ import annotations
import json, pathlib, sys
import yaml

HERE = pathlib.Path(__file__).parent
BUNDLE = HERE.parent
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."


def load():
    return yaml.safe_load((HERE / "grounding.yaml").read_text())


def truth(g) -> str:
    a, o, L = g["axis"], g["oracle"], g["ladder"]
    steps = ["| %s | %s | %s |" % (c["id"], c["reduction"],
                                   "red line" if c.get("red_line") else "graded")
             for c in sorted(g["checkers"], key=lambda c: c["order"])]
    frozen = ["| %s | %s |" % (f["id"], f["enforced_by"]) for f in g["frozen_axes"]]
    rejected = ["| %s | %s | %s |" % (k["id"], k["rejecting_checker"], k["must_score"])
                for k in g["known_wrong_controls"]]
    retired = ["| %s | %s | %s |" % (r["id"], r["was"], r["now"])
               for r in g["retired_checkers"]]
    residue = ["| %s | %s |" % (j["id"], j["residue"]) for j in g["judged_residue"]]
    gaps = ["| %s | %s |" % (x["id"], x["effect"]) for x in g["gaps"]]
    return "\n".join([
        "# Ground truth", "", BANNER, "",
        "The ordered path an accepted solution takes through `instruction.md`, the state each action establishes, and the checker each step satisfies. Generated from `solution/grounding.yaml`.", "",
        "## What this record supersedes", "",
        g["supersedes"]["why"], "",
        "## Oracle", "",
        "The oracle is the %s. The change is %s, which touches only these free axes: %s. Frozen axes touched: %s." % (
            o["identity"], o["change"], ", ".join(o["free_axes_touched"]),
            ", ".join(o["frozen_axes_touched"]) or "none"), "",
        o["exemption_note"], "",
        "## Axis", "",
        "The graded quantity is %s, %s, %s. It is scored as `%s`. %s" % (
            a["graded_quantity"], a["direction"], a["measured_by"], a["score"],
            a["never_from"]), "",
        "Anchor state: %s." % a["anchors_state"], "",
        "## Frozen axes and what enforces each", "",
        "| axis | enforced by |", "|---|---|", *frozen, "",
        "## Ordered path", "",
        "| checker | reduction | standing |", "|---|---|---|", *steps, "",
        "## Checkers this record replaced", "",
        "| id | was | now |", "|---|---|---|", *retired, "",
        "## Verification", "",
        "Grading re-executes the submitted recipe on %d seeds drawn after submission, with an absolute floor of %d. %s Significance is `%s`." % (
            L["min_seeds"], L["min_seeds_floor"], L["seed_selection"], L["significance"]), "",
        "## Rejected routes", "",
        "Each names a known-wrong control whose rejection Phase 2 measures.", "",
        "| control | rejected by | score |", "|---|---|---|", *rejected, "",
        "## Judged residue", "",
        "| id | residue that blocks compilation |", "|---|---|", *residue, "",
        "## Declared gaps", "",
        "| id | effect |", "|---|---|", *gaps, "",
    ]) + "\n"


def rubrics(g) -> dict:
    items = []
    for c in sorted(g["checkers"], key=lambda c: c["order"]):
        items.append({
            "id": "R_" + c["id"],
            "dimension": "red_line" if c.get("red_line") else "outcome",
            "weight": 1.0,
            "evaluation_target": "reward_record",
            "criterion": "checker %s resolves as %s against state the verifier measured" % (
                c["id"], c["reduction"]),
            "judgment": c["reduction"],
            "evidence": ["reward_record"],
            "mode": "compiled",
        })
    for j in g["judged_residue"]:
        items.append({
            "id": "R_" + j["id"],
            "dimension": "conduct",
            "weight": float(j["weight"]),
            "evaluation_target": "submitted_script_source",
            "criterion": "obligation %s holds of the submission" % j["id"],
            "judgment": "DIVERGENCE",
            "evidence": ["submitted_script_source"],
            "mode": "judged",
            "residue": j["residue"],
        })
    total = sum(i["weight"] for i in items)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    return {"banner": BANNER, "source": "solution/grounding.yaml",
            "compilation_floor": 0.75,
            "compiled_weight_share": round(compiled / total, 6),
            "compilation_floor_met": (compiled / total) >= 0.75,
            "items": items}


def test_output(r) -> str:
    lines = ["# " + BANNER,
             "# Compiled from solution/rubrics.json. Carries the implied relation only,",
             "# never criterion prose and never reference text.",
             "import json, os, pathlib", "",
             "REWARD = pathlib.Path(os.environ.get('BIA_REWARD_PATH',",
             "                                    '/logs/verifier/reward.txt'))",
             "RECORD = pathlib.Path(os.environ.get('BIA_REWARD_RECORD',",
             "                                    '/logs/verifier/record.json'))", "", "",
             "def _payload():", "    return json.loads(RECORD.read_text())", "", "",
             "def _outcome(name):",
             "    return _payload()['checkers'].get(name)", "", "",
             "def test_reward_file_written():", "    assert REWARD.is_file()", "", "",
             "def test_bound_reward_is_one_float_agreeing_with_the_record():",
             "    value = float(REWARD.read_text().strip())",
             "    assert 0.0 <= value <= 1.0",
             "    assert abs(value - float(_payload()['reward'])) < 1e-6", "", "",
             "def test_reward_shape():", "    p = _payload()",
             "    assert 0.0 <= p['reward'] <= 1.0", "    assert p['pass'] in (0, 1)",
             "    assert isinstance(p['reason'], str) and p['reason']", "", "",
             "def test_zero_reward_carries_reason():", "    p = _payload()",
             "    if p['reward'] == 0.0:", "        assert p['reason'] != ''", "", "",
             "def test_unmeasured_anchors_never_pass():", "    p = _payload()",
             "    if p['reason'] == 'anchors-unmeasured':",
             "        assert p['pass'] == 0 and p['reward'] == 0.0", "", ""]
    for it in r["items"]:
        if it["mode"] != "compiled":
            continue
        name = it["id"].replace("R_", "")
        lines += ["def test_%s():" % name,
                  "    p = _payload()",
                  "    if p['reward'] > 0.0:",
                  "        assert _outcome(%r) is True" % name, "", ""]
    return "\n".join(lines)


def solve(g) -> str:
    return "\n".join([
        "#!/usr/bin/env bash", "# " + BANNER,
        "# Private oracle. Harbor mounts solution/ only for the oracle path.",
        "#",
        "# The reference reads its seed from BIA_SEED and its horizon from the",
        "# provided loader, exactly as a submission does, so this script takes no",
        "# arguments. The voided oracle for this slot took --seed while the grader",
        "# invoked --steps, so it could not run under its own contract at all.",
        "set -euo pipefail", 'cd "$(dirname "$0")"',
        'exec python3 recipe.py', "",
    ]) + "\n"


def main() -> int:
    g = load()
    r = rubrics(g)
    outputs = {
        HERE / "TRUTH.md": truth(g),
        HERE / "rubrics.json": json.dumps(r, indent=2, sort_keys=True) + "\n",
        BUNDLE / "tests" / "test_output.py": test_output(r),
        HERE / "solve.sh": solve(g),
    }
    drift = []
    check = "--check" in sys.argv
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if check:
            if not path.is_file() or path.read_text() != text:
                drift.append(str(path.relative_to(BUNDLE)))
        else:
            path.write_text(text)
    if check:
        print("drift: " + (", ".join(drift) if drift else "none"))
        return 1 if drift else 0
    (HERE / "solve.sh").chmod(0o755)
    print("regenerated: " + ", ".join(sorted(str(p.relative_to(BUNDLE)) for p in outputs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
