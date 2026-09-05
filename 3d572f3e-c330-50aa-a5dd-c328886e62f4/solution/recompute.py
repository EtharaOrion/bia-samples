# FORGE-CANARY-BEGIN
# 0: 9c4fc9e2bc1a71694608c9ff6683b3c7daf755c3ca77511ef5bb5f0a6224f2a2
# 1: 8aed3048477f59fe7c92323ad768f3cbcc9e5900c19ba2311508c4357d65c4e4
# 2: 35ae77b6aa93a93aa6a5eddcecb601186a78d772376879ae9099881cd0bcf7a4
# 3: 0faa0dbbcabf73ba89a6f5edb477679eec916ea719605abc5627eb2c493f5181
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of OER-15 from solution/grounding.yaml alone.

This file is the only writer of the generated artifacts. It reads exactly one
input, `solution/grounding.yaml`, and writes:

    solution/golden_trajectory.json
    solution/fixtures.json
    solution/solve.sh
    solution/TRUTH.md
    solution/rubrics.json
    tests/test_output.py

It invokes NO model, NO network, NO clock, NO locale-dependent operation and NO
random source. It never imports environment/harness.py, never runs the harness,
never touches a corpus, and never reads a measurement. Every number it emits is a
number `grounding.yaml` already carries. Running it twice over frozen bytes
therefore produces byte-identical output, which `--check` proves without writing.

`--check` recomputes every artifact in memory and compares it byte for byte to
what is on disk. It writes nothing and exits non-zero on the first drift, naming
the file that moved, so a hand-edit of a generated file is loud rather than
silent.

Usage:
    python3 solution/recompute.py            # write every generated artifact
    python3 solution/recompute.py --check    # recompute, write nothing, exit 1 on drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

BUNDLE = Path(__file__).resolve().parents[1]
SOURCE = Path("solution/grounding.yaml")

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
ORIGIN = "solution/grounding.yaml"
GENERATOR = "solution/recompute.py"


def _hash_banner(prefix: str) -> str:
    return (
        prefix + " " + BANNER + "\n"
        + prefix + " Source: " + ORIGIN + "\n"
        + prefix + " Generator: " + GENERATOR + "\n"
        + prefix + " Edit " + ORIGIN + " and re-run the generator; never edit this file.\n"
    )


def load_grounding(bundle: Path) -> dict:
    import yaml  # the verifier image carries it; nothing else here needs a dependency

    payload = yaml.safe_load((bundle / SOURCE).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "forge.grounding/v1":
        raise SystemExit("solution/grounding.yaml does not carry schema forge.grounding/v1")
    return payload


# ---------------------------------------------------------------------------
# solution/golden_trajectory.json
# ---------------------------------------------------------------------------
def golden_trajectory(g: dict) -> str:
    document = {
        "generated": BANNER,
        "source": ORIGIN,
        "generator": GENERATOR,
        "schema": "oer15.golden_trajectory/v1",
        "slot": g["slot"],
        "metric": g["identity"]["metric"],
        "direction": g["identity"]["direction"],
        "steps": [dict(row) for row in g["golden_trajectory"]],
        "terminates_at": {
            "graded_bits_per_byte": None,
            "graded_bits_per_byte_state": g["measured"]["state"],
            "reward": g["reward"]["reference_score"],
            "reward_state": g["reward"]["reference_score_state"],
            "reward_gap": g["reward"]["reference_score_gap"],
            "reason": "graded",
        },
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# solution/fixtures.json
# ---------------------------------------------------------------------------
def fixtures(g: dict) -> str:
    block = g["fixtures"]
    document = {
        "generated": BANNER,
        "source": ORIGIN,
        "generator": GENERATOR,
        "schema": "oer15.fixtures/v1",
        "slot": g["slot"],
        "accepting": dict(block["accepting"]),
        "rejecting": [dict(row) for row in block["rejecting"]],
        "carriers": {
            "reward_path": g["reward"]["carrier"],
            "score_document_path": g["reward"]["score_document"],
        },
        "operating_points": {
            "state": g["measured"]["state"],
            "gap": g["measured"]["gap"],
            "floor_source": g["reward"]["floor_carrier"],
            "acceptance_separation_bpb": g["measured"]["acceptance_separation_bpb"],
            "full_credit_separation_bpb": g["measured"]["full_credit_separation_bpb"],
            "separations_gap": g["measured"]["separations_gap"],
            "paired_baseline_row": dict(g["measured"]["paired_baseline_row"]),
        },
        "anchors": {
            "anchors_state": g["anchors"]["anchors_state"],
            "gap_id": g["anchors"]["gap_id"],
            "baseline_metric": g["anchors"]["baseline_metric"],
            "target_metric": g["anchors"]["target_metric"],
        },
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# solution/rubrics.json
# ---------------------------------------------------------------------------
def rubrics(g: dict) -> str:
    rows = [
        {"id": row["id"], "rubric": row["rubric"], "weight": row["weight"]}
        for row in g["rubrics_generated"]
    ]
    document = {
        "generated": BANNER,
        "source": ORIGIN,
        "generator": GENERATOR,
        "schema": "oer15.solution_rubrics/v1",
        "slot": g["slot"],
        "judged": "the solution against its reference answer",
        "not_judged": "agent behaviour across the trajectory, which tests/rubrics.jsonl judges",
        "reference_answer": {
            "construction": g["measured"]["reference_construction"],
            "graded_bits_per_byte": None,
            "graded_bits_per_byte_state": g["measured"]["state"],
            "graded_bits_per_byte_gap": g["measured"]["gap"],
            "margin_over_the_bar_state": g["reward"]["reference_score_state"],
            "margin_over_the_bar_gap": g["reward"]["reference_score_gap"],
        },
        "total_weight": sum(row["weight"] for row in rows),
        "criteria": rows,
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# solution/solve.sh, the solution entry point
# ---------------------------------------------------------------------------
def solve_sh(g: dict) -> str:
    measured = g["measured"]
    lines = [
        "#!/usr/bin/env bash",
        "#",
    ]
    lines.extend(("# " + row).rstrip() for row in _hash_banner("").splitlines())
    lines.extend(
        [
            "#",
            "# OER-15 solution entry point. It installs the reference vocabulary",
            "# construction as the submission and does nothing else: the graded number is",
            "# computed by the verifier from harness-owned state, so a solution that tried",
            "# to report a number would be reporting one nothing reads.",
            "#",
            "# The handed construction in environment/default_tokenizer.py is a word list.",
            "# Its option grid has " + str(measured["default_option_grid_rows"]) + " rows and its largest top_k is "
            + str(measured["default_grid_max_top_k"]) + ", against",
            "# " + str(measured["entries_beyond_single_bytes"]) + " entries available past the single bytes, so every row leaves at least",
            "# " + str(measured["entries_the_handed_grid_cannot_reach"]) + " entries unspent at any option value. The reference construction",
            "# merges frequent adjacent pairs over the decoded FineWeb bytes instead, so it",
            "# produces affixes, whole words and multi-word phrases and spends the whole",
            "# budget. Its bits-per-byte reading on this substrate is " + str(measured["state"]) + ",",
            "# under gap " + str(measured["gap"]) + ".",
            "#",
            "set -euo pipefail",
            "",
            'SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'SUBMISSION_DIR="${OER15_SUBMISSION:-/workspace/submission}"',
            "",
            'mkdir -p "${SUBMISSION_DIR}"',
            "",
            "# The submission is the reference construction and nothing more. No",
            "# HALT_AT_STEPS is declared, so the run reaches the bound evaluation point,",
            "# and no REPORT is declared, so no self-reported number exists to be read.",
            'cat > "${SUBMISSION_DIR}/tokenizer.py" <<\'OER15_SUBMISSION_EOF\'',
            '"""Submission: the reference vocabulary construction for OER-15."""',
            "import importlib.util",
            "import os",
            "",
            '_REFERENCE = os.environ.get("OER15_REFERENCE", "REFERENCE_PATH_PLACEHOLDER")',
            '_spec = importlib.util.spec_from_file_location("oer15_reference", _REFERENCE)',
            "_reference = importlib.util.module_from_spec(_spec)",
            "_spec.loader.exec_module(_reference)",
            "",
            "",
            "def build_vocab(train_bytes, budget):",
            '    """Greedy corpus-derived pair merges, deterministic at every tie."""',
            "    return _reference.build_vocab(train_bytes, budget)",
            "OER15_SUBMISSION_EOF",
            "",
            "# Bind the reference path into the installed submission, so the submission runs",
            "# under tests/runner.py isolation without an environment variable surviving it.",
            'python3 - "${SUBMISSION_DIR}/tokenizer.py" "${SOLUTION_DIR}/reference.py" <<\'OER15_BIND_EOF\'',
            "import sys",
            "target, reference = sys.argv[1], sys.argv[2]",
            'text = open(target, "r", encoding="utf-8").read()',
            'open(target, "w", encoding="utf-8").write(',
            '    text.replace("REFERENCE_PATH_PLACEHOLDER", reference)',
            ")",
            "OER15_BIND_EOF",
            "",
            'printf \'%s\\n\' "installed the reference construction at ${SUBMISSION_DIR}/tokenizer.py"',
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# solution/TRUTH.md
# ---------------------------------------------------------------------------
def truth_md(g: dict) -> str:
    measured = g["measured"]
    frozen = g["frozen_axes"]
    anchors = g["anchors"]
    reward = g["reward"]
    out = []
    out.append("<!-- " + BANNER + " -->")
    out.append("<!-- Source: " + ORIGIN + " -->")
    out.append("<!-- Generator: " + GENERATOR + " -->")
    out.append("")
    out.append("# OER-15 TRUTH")
    out.append("")
    out.append("`" + BANNER + "` Every number below is transcribed from `" + ORIGIN + "`.")
    out.append("")
    out.append("## What is graded")
    out.append("")
    out.append("Metric: **" + g["identity"]["metric"] + "**, direction **"
               + g["identity"]["direction"] + " is better**.")
    out.append("")
    out.append(
        "The verifier computes the number itself, from the parameters its own training run of the "
        "canonical decoder held at the bound optimizer step, over a held-out FineWeb slice measured "
        "in **bytes**. It is never a number the submission reported, never a smoothed or averaged "
        "series, and never a checkpoint the submission selected."
    )
    out.append("")
    out.append("## What the re-base moved")
    out.append("")
    out.append("Kind: **" + g["rebase"]["kind"] + "**.")
    out.append("")
    out.append(" ".join(str(g["rebase"]["kind_note"]).split()))
    out.append("")
    for row in g["rebase"]["moved"]:
        out.append("- " + " ".join(str(row).split()))
    out.append("")
    out.append("Retired with it:")
    out.append("")
    for row in g["rebase"]["retired"]:
        out.append("- " + " ".join(str(row).split()))
    out.append("")
    out.append("## The simulator test")
    out.append("")
    out.append("| limb | verdict | evidence |")
    out.append("|---|---|---|")
    for limb in ("weights_in_the_loop", "verifier_recomputes", "no_forward_pass_no_score"):
        block = g["simulator_test"][limb]
        out.append(
            "| `" + limb + "` | **" + str(block["verdict"]) + "** | "
            + " ".join(str(block["evidence"]).split()) + " |"
        )
    out.append("")
    out.append("## Frozen axes")
    out.append("")
    out.append("| axis | value |")
    out.append("|---|---|")
    for key in sorted(frozen):
        out.append("| `" + key + "` | `" + str(frozen[key]) + "` |")
    out.append("")
    out.append("Free axis: **" + g["free_axis"] + "**.")
    out.append("")
    out.append("## The archetype, checkable rather than asserted")
    out.append("")
    out.append(
        "Archetype **" + g["archetype"] + "**, adversarial option expansion. The default construction "
        "handed to the agent is `" + measured["default_construction"] + "`, and its complete option "
        "grid carries **" + str(measured["default_option_grid_rows"]) + " rows**."
    )
    out.append("")
    out.append("| quantity | value | how it is established |")
    out.append("|---|---|---|")
    out.append(
        "| option grid rows | `" + str(measured["default_option_grid_rows"])
        + "` | counted from `DEFAULT_OPTION_GRID` |"
    )
    out.append(
        "| largest `top_k` the grid offers | `" + str(measured["default_grid_max_top_k"])
        + "` | read from `DEFAULT_OPTION_GRID` |"
    )
    out.append(
        "| entries available past the single bytes | `"
        + str(measured["entries_beyond_single_bytes"]) + "` | 50304 less 256 |"
    )
    out.append(
        "| entries no row of the grid can reach | `"
        + str(measured["entries_the_handed_grid_cannot_reach"]) + "` | 50048 less 1024 |"
    )
    out.append(
        "| acceptance separation | `" + repr(measured["acceptance_separation_bpb"])
        + "` bits per byte | declared, " + str(measured["separations_state"]) + " |"
    )
    out.append(
        "| full-credit separation | `" + repr(measured["full_credit_separation_bpb"])
        + "` bits per byte | declared, held verifier-side |"
    )
    out.append(
        "| every bits-per-byte operating point | `" + str(measured["state"])
        + "` | gap `" + str(measured["gap"]) + "` |"
    )
    out.append("")
    out.append(
        "The paired baseline row the verifier trains is `"
        + json.dumps(measured["paired_baseline_row"], sort_keys=True) + "`."
    )
    out.append("")
    out.append(" ".join(str(measured["structural_limit"]).split()))
    out.append("")
    out.append(" ".join(str(measured["how_measured"]).split()))
    out.append("")
    out.append(" ".join(str(measured["note"]).split()))
    out.append("")
    out.append("## The derivation the agent is expected to perform")
    out.append("")
    for row in g["golden_trajectory"]:
        out.append(str(row["step"]) + ". " + row["action"])
        out.append("   - establishes: " + row["establishes"])
    out.append("")
    out.append("## The reference answer")
    out.append("")
    out.append(
        "`" + measured["reference_construction"] + "` merges the most frequent adjacent symbol pair "
        "over the decoded FineWeb bytes, breaking every tie lexicographically on the merged bytes, "
        "until no pair occurs twice or the budget is spent. It places affixes, whole words and "
        "multi-word phrases in the same budget and spends the entries the handed grid cannot reach."
    )
    out.append("")
    out.append(" ".join(str(g["reward"]["reference_score_note"]).split()))
    out.append("")
    out.append("## Anchors")
    out.append("")
    out.append(
        "`anchors_state: " + str(anchors["anchors_state"]) + "` under gap `"
        + str(anchors["gap_id"]) + "`. `baseline_metric` is `" + str(anchors["baseline_metric"])
        + "` and `target_metric` is `" + str(anchors["target_metric"]) + "`."
    )
    out.append("")
    out.append(" ".join(str(anchors["statement"]).split()))
    out.append("")
    out.append("## Reward")
    out.append("")
    out.append(
        "Carrier `" + reward["carrier"] + "`, " + reward["carrier_form"] + ". Reason and metric "
        "block at `" + reward["score_document"] + "`. Interval `" + str(reward["interval"])
        + "`, higher is better, never binary."
    )
    out.append("")
    out.append("Formula: `" + reward["formula"] + "`")
    out.append("")
    out.append(
        "The two carriers standing in for the absent anchors are the floor, which is "
        + reward["floor_carrier"] + ", and the ceiling, which is " + reward["ceiling_carrier"]
        + ". Neither is a literal in this bundle and neither is presented as a family anchor. The "
        "default construction at the paired baseline row scores `" + repr(reward["plateau_score"])
        + "`, and that one is provable rather than measured: a submission that returns the handed "
        "construction at the paired baseline row IS the floor, so raw is zero by construction and "
        "the gate emits default-construction-plateau-not-beaten."
    )
    out.append("")
    out.append(
        "The reference construction's score is `" + repr(reward["reference_score"]) + "`, state `"
        + str(reward["reference_score_state"]) + "`, under gap `"
        + str(reward["reference_score_gap"]) + "`."
    )
    out.append("")
    out.append("## Statement ambiguity")
    out.append("")
    readings = g["statement_readings"]
    out.append("`outcomes_admitted` is `" + str(readings["outcomes_admitted"])
               + "`. The readings that reduce to it:")
    out.append("")
    out.append("| reading | source | numerator | denominator bytes | point | direction |")
    out.append("|---|---|---|---|---|---|")
    for row in readings["readings"]:
        reduces = row["reduces_to"]
        out.append(
            "| `" + row["id"] + "` | " + row["source"]
            + " | " + reduces["numerator"]
            + " | `" + str(reduces["denominator_bytes"]) + "`"
            + " | " + reduces["point"]
            + " | " + reduces["direction"] + " |"
        )
    out.append("")
    out.append("## Solver egress divergence")
    out.append("")
    out.append(" ".join(str(g["solver_egress_divergence"]["resolution_here"]).split()))
    out.append("")
    out.append("Gap `" + g["solver_egress_divergence"]["gap_id"] + "`.")
    out.append("")
    out.append("## Bound grants")
    out.append("")
    out.append("| field | value | role |")
    out.append("|---|---|---|")
    bindings = g["bindings"]
    out.append("| `max_timeout_hours` | `" + str(bindings["max_timeout_hours"]) + "` | "
               + bindings["max_timeout_role"] + " |")
    out.append("| `budget_hours` | `" + str(bindings["budget_hours"]) + "` | "
               + bindings["budget_hours_role"] + " |")
    out.append("| `max_attempts` | `" + str(bindings["max_attempts"]) + "` | attempts per session |")
    out.append("| `final_selection` | `" + str(bindings["final_selection"]) + "` | best of k |")
    out.append("| `budget_envelope` | `" + bindings["budget_envelope"] + "` | |")
    out.append("| `budget_margin` | `" + bindings["budget_margin"] + "` | carried by name |")
    out.append("")
    out.append(
        "`budget_hours` is `" + str(bindings["budget_hours_state"]) + "` under gap `"
        + bindings["budget_hours_gap"] + "`. `max_timeout_hours` and `budget_hours` are never "
        "aliases: the first bounds the session across attempts, the second bounds one attempt."
    )
    out.append("")
    out.append(" ".join(str(bindings["budget_hours_rebase_note"]).split()))
    out.append("")
    out.append("## Tier")
    out.append("")
    tier = g["tier"]
    out.append(
        "Target tier `" + tier["target_tier"] + "`, `tier_exemption_granted: "
        + str(tier["tier_exemption_granted"]) + "`. Frontier-defeat floor `"
        + str(tier["frontier_defeat_floor"]) + "`, separation margin `"
        + str(tier["separation_margin"]) + "`. Anchorable tier " + tier["anchorable_tier"] + "."
    )
    out.append("")
    out.append(" ".join(str(tier["perception_axis_exemption"]).split()))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# tests/test_output.py, the compiled per-checker suite
# ---------------------------------------------------------------------------
TEST_PRELUDE = '''"""The compiled per-checker suite for OER-15.

Each `test_<checker id>` carries BOTH halves of exactly one checker declared in
`tests/checkers.yaml`:

  accepting half  -- the LIVE reference run, driven through the real checker, must
                     pass. The base record is produced by `tests/runner.py`, which
                     isolates the submission and runs the frozen harness in this
                     process, so every number in it is a number the verifier made.
  rejecting half  -- either a second LIVE run of a defective submission, or the
                     base record with one minimal planted defect applied. The
                     checker must return exactly the `zero_reason` its manifest row
                     declares, and no other checker's reason is accepted.

Ten of the twelve rejecting halves are carried on planted telemetry fixtures,
because `tests/runner.py` never lets a submission near the step counter, the
parameters, the denominator, the training shards, the held-out slice or the
evaluation schedule. That substitution is recorded in `tests/checkers.yaml`.

COST. Every live half here is a real 2200-step training run of the canonical
12-layer 768-dimension decoder, plus the verifier's paired baseline run, over
staged FineWeb10B shards. This suite needs an accelerator and the shards; it does
not run on a laptop in seconds, and a version of it that did would be the same
class of stand-in the re-base removed.

`seed/tasks/OER-15/adequacy.py` imports the helpers below rather than
reimplementing them, so the adequacy proof and this suite can never disagree about
what a fixture is.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
TESTS = BUNDLE / "tests"
SOLUTION = BUNDLE / "solution"
ENVIRONMENT = BUNDLE / "environment"

sys.dont_write_bytecode = True
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import checkers as C  # noqa: E402
import grade as G  # noqa: E402


SUBMISSION_REFERENCE = (
    "import importlib.util\\n"
    "_spec = importlib.util.spec_from_file_location('oer15_reference', %r)\\n"
    "_m = importlib.util.module_from_spec(_spec)\\n"
    "_spec.loader.exec_module(_m)\\n"
    "def build_vocab(train_bytes, budget):\\n"
    "    return _m.build_vocab(train_bytes, budget)\\n"
) % (str(SOLUTION / "reference.py"),)

SUBMISSION_EARLY_STOP = SUBMISSION_REFERENCE + "HALT_AT_STEPS = %d\\n"

SUBMISSION_PLATEAU = (
    "import importlib.util\\n"
    "_spec = importlib.util.spec_from_file_location('oer15_default', %r)\\n"
    "_m = importlib.util.module_from_spec(_spec)\\n"
    "_spec.loader.exec_module(_m)\\n"
    "_BEST = %s\\n"
    "def build_vocab(train_bytes, budget):\\n"
    "    return _m.build_vocab(train_bytes, budget, **_BEST)\\n"
) % (str(ENVIRONMENT / "default_tokenizer.py"), "%s")

SUBMISSION_NOOP = "def build_vocab(train_bytes, budget):\\n    return []\\n"


def live_record(source: str) -> dict:
    """Run tests/runner.py over one submission source and return its run record.

    This is the only place a submission is ever executed, and it is executed as a
    separate program by the real runner. Nothing here imports a submission.
    """
    with tempfile.TemporaryDirectory(prefix="oer15-compiled-") as scratch:
        room = Path(scratch)
        (room / "submission").mkdir()
        (room / "submission" / "tokenizer.py").write_text(source, encoding="utf-8")
        out = room / "record.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(TESTS / "runner.py"),
                "--delivery-root", str(BUNDLE),
                "--submission", str(room / "submission"),
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
        )
        if not out.is_file():
            raise AssertionError("tests/runner.py wrote no record: " + completed.stderr[-400:])
        return json.loads(out.read_text(encoding="utf-8"))


_BASE: dict = {}


def base_record() -> dict:
    """The accepting fixture: one live reference run, computed once and reused."""
    if not _BASE:
        _BASE["record"] = live_record(SUBMISSION_REFERENCE)
    return copy.deepcopy(_BASE["record"])


def _dig(record: dict, path: list):
    node = record
    for key in path[:-1]:
        node = node[key]
    return node


def apply_ops(record: dict, ops: list) -> dict:
    """Apply the minimal planted defect described by grounding.yaml.

    The vocabulary is closed and small on purpose: every op names one location and
    one movement, so a fixture cannot quietly become a rewrite of the whole record.
    """
    out = copy.deepcopy(record)
    for op in ops:
        kind = op["op"]
        if kind == "set":
            _dig(out, op["path"])[op["path"][-1]] = op["value"]
        elif kind == "set_from":
            _dig(out, op["path"])[op["path"][-1]] = _dig(out, op["source"])[op["source"][-1]]
        elif kind == "add_int":
            node = _dig(out, op["path"])
            node[op["path"][-1]] = int(node[op["path"][-1]]) + int(op["delta"])
        elif kind == "swap_points":
            rows = out["telemetry"]["eval_points"]
            rows[op["a"]], rows[op["b"]] = rows[op["b"]], rows[op["a"]]
        elif kind == "add_float_last_sustain":
            rows = [r for r in out["telemetry"]["eval_points"] if r["role"] == "sustain"]
            rows[-1][op["key"]] = float(rows[-1][op["key"]]) + float(op["delta"])
        else:
            raise AssertionError("op outside the closed fixture vocabulary: " + repr(kind))
    return out


def outcome(record: dict, selector: str):
    """Drive ONE live checker over one record through the real evidence assembly.

    The verifier-owned anchors in `tests/anchors.json` are overlaid first, exactly as
    `tests/grade.py` overlays them before it assembles Evidence, so a checker reading a
    verifier-side operating point sees here what it sees on the graded path.
    """
    merged = G.merge_verifier_anchors(record, G.load_anchors())
    evidence = C.evidence_from_record(merged, str(BUNDLE))
    return getattr(C, selector)(evidence)


def accepts(selector: str) -> None:
    result = outcome(base_record(), selector)
    assert result.passed, selector + " rejected the live reference run: " + result.detail


def rejects(record: dict, selector: str, zero_reason: str) -> None:
    result = outcome(record, selector)
    assert not result.passed, selector + " accepted a planted defect: " + result.detail
    assert result.reason == zero_reason, (
        selector + " emitted " + repr(result.reason) + ", expected " + repr(zero_reason)
    )
'''


def test_output_py(g: dict) -> str:
    measured = g["measured"]
    rejecting = {row["checker"]: row for row in g["fixtures"]["rejecting"]}
    out = [_hash_banner("#"), TEST_PRELUDE, ""]
    out.append("")
    out.append("PLATEAU_OPTIONS = " + repr(dict(measured["paired_baseline_row"])))
    out.append("EARLY_STOP_AT = " + str(g["frozen_axes"]["compute_budget_steps"] // 2))
    out.append("")
    out.append("")
    for checker in _CHECKER_ORDER:
        row = rejecting[checker]
        selector = "check_" + checker
        out.append("def test_" + checker + "():")
        out.append('    """' + row["zero_reason"] + '. Both halves, one checker."""')
        out.append("    accepts(" + repr(selector) + ")")
        if row["kind"] == "planted-telemetry-fixture":
            out.append("    planted = apply_ops(base_record(), " + repr(row["ops"]) + ")")
            out.append("    rejects(planted, " + repr(selector) + ", "
                       + repr(row["zero_reason"]) + ")")
        elif checker == "early_stop_not_an_established_metric":
            out.append("    halted = live_record(SUBMISSION_EARLY_STOP % EARLY_STOP_AT)")
            out.append("    rejects(halted, " + repr(selector) + ", "
                       + repr(row["zero_reason"]) + ")")
        elif checker == "beats_default_construction_optimum":
            out.append("    plateau = live_record(SUBMISSION_PLATEAU % (PLATEAU_OPTIONS,))")
            out.append("    rejects(plateau, " + repr(selector) + ", "
                       + repr(row["zero_reason"]) + ")")
        else:
            raise SystemExit("no compiled form for live fixture on checker " + checker)
        out.append("")
        out.append("")
    out.append("if __name__ == \"__main__\":")
    out.append("    failures = 0")
    out.append("    for name, function in sorted(list(globals().items())):")
    out.append("        if not name.startswith(\"test_\"):")
    out.append("            continue")
    out.append("        try:")
    out.append("            function()")
    out.append("            print(\"PASS \" + name)")
    out.append("        except AssertionError as problem:")
    out.append("            failures += 1")
    out.append("            print(\"FAIL \" + name + \": \" + str(problem))")
    out.append("    raise SystemExit(1 if failures else 0)")
    return "\n".join(out) + "\n"


# The gate order tests/checkers.yaml declares, restated here as the order the
# compiled suite is emitted in, so a reader diffing the two sees one sequence.
_CHECKER_ORDER = [
    "early_stop_not_an_established_metric",
    "frozen_axes_unmoved",
    "snapshot_shapes_match_substrate",
    "eval_points_ordered_by_steps",
    "compute_budget_respected_as_spent",
    "denominator_is_held_out_slice_bytes",
    "graded_state_is_harness_owned",
    "graded_readout_unsmoothed",
    "submission_vocabulary_took_effect",
    "metric_not_taken_from_submission_report",
    "beats_default_construction_optimum",
    "reading_sustained_across_scheduled_points",
]


ARTIFACTS = (
    ("solution/golden_trajectory.json", golden_trajectory, False),
    ("solution/fixtures.json", fixtures, False),
    ("solution/rubrics.json", rubrics, False),
    ("solution/TRUTH.md", truth_md, False),
    ("solution/solve.sh", solve_sh, True),
    ("tests/test_output.py", test_output_py, False),
)


def render(bundle: Path) -> dict:
    g = load_grounding(bundle)
    declared = list(g.get("generated_artifacts") or [])
    produced = [name for name, _, _ in ARTIFACTS]
    if sorted(declared) != sorted(produced):
        raise SystemExit(
            "grounding.yaml declares generated_artifacts "
            + repr(sorted(declared))
            + " and this generator produces "
            + repr(sorted(produced))
        )
    return {name: builder(g) for name, builder, _ in ARTIFACTS}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive every generated OER-15 artifact from solution/grounding.yaml."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute without writing and exit non-zero on drift",
    )
    parser.add_argument("--bundle", default=str(BUNDLE))
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    rendered = render(bundle)
    executable = {name for name, _, is_exec in ARTIFACTS if is_exec}

    if args.check:
        drift = []
        for name in sorted(rendered):
            path = bundle / name
            if not path.is_file():
                drift.append(name + ": absent")
            elif path.read_text(encoding="utf-8") != rendered[name]:
                drift.append(name + ": on-disk bytes differ from the recomputed bytes")
        for row in drift:
            print("DRIFT " + row, file=sys.stderr)
        if drift:
            return 1
        print("no drift: " + str(len(rendered)) + " generated artifacts match " + ORIGIN)
        return 0

    for name in sorted(rendered):
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered[name], encoding="utf-8")
        if name in executable:
            path.chmod(0o755)
        print("wrote " + name)
    return 0


# FORGE-SCREENING-CARRIER-BEGIN
# GENERATED SECTION. DO NOT HAND-EDIT.
# Generated by seed/forge/screenfreeze.py. Derives the contamination-screening provenance carrier
# from the frozen `screening` block in solution/grounding.yaml and nothing else. It opens no
# connection, reads no wall clock, consults no host language setting, draws no entropy, starts no
# child process, and imports nothing outside this tree.
import hashlib as _forge_hashlib
import json as _forge_json
import pathlib as _forge_pathlib
import sys as _forge_sys

import yaml as _forge_yaml

_FORGE_CARRIER_KEYS = (
    "schema",
    "unit_uuid",
    "screening_roots",
    "authority_mode",
    "source_identifiers",
    "fork_ancestry_snapshot",
    "base_commit_sha",
    "applicable_dates",
    "instrument_versions",
    "atom_result_digests",
    "applicability",
    "sanitization_closure",
    "empty_submission_result",
    "attestations",
    "binding_block",
    "keyid",
    "trust_root_public_key_hex",
    "namespace",
    "normalization_domain_version",
    "signer_identity",
)

_FORGE_BINDING_KEYS = (
    "canonical_bundle_hash",
    "pinned_image_digest",
    "binding_envelope",
)

_FORGE_SCREENING_KEY = "screening"
_FORGE_GROUNDING = "grounding.yaml"
_FORGE_CARRIER = "provenance.yaml"
_FORGE_BANNER = "# GENERATED SECTION. DO NOT HAND-EDIT."


def _forge_here():
    return _forge_pathlib.Path(__file__).resolve().parent


def _forge_sorted(value):
    """Sort every container so two runs over the same frozen bytes emit identical bytes."""
    if isinstance(value, dict):
        return {key: _forge_sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_forge_sorted(item) for item in value]
    return value


def _forge_frozen_screening():
    """Read the frozen screening block. Absence is refused rather than defaulted."""
    path = _forge_here() / _FORGE_GROUNDING
    with path.open("r", encoding="utf-8") as handle:
        document = _forge_yaml.safe_load(handle)
    block = (document or {}).get(_FORGE_SCREENING_KEY)
    if not isinstance(block, dict):
        raise SystemExit(
            "solution/grounding.yaml carries no frozen `screening` block, so the provenance "
            "carrier cannot be derived. Refusing to emit a carrier over values nobody froze."
        )
    missing = [key for key in _FORGE_CARRIER_KEYS if key not in block]
    unknown = [key for key in sorted(block) if key not in _FORGE_CARRIER_KEYS]
    if missing or unknown:
        raise SystemExit(
            "the frozen `screening` block does not mirror the closed carrier schema: "
            "missing " + repr(missing) + ", unknown " + repr(unknown)
        )
    return block


def _forge_canonical_bytes(payload):
    return _forge_json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _forge_carrier_payload():
    """Assemble the carrier as exactly the closed key set, in the order the schema fixes.

    The binding block is attached AFTER the canonical payload is hashed and never enters the
    preimage, because a payload that contained a hash of itself would have no acyclic ordering.
    """
    block = _forge_frozen_screening()
    payload = {}
    for key in _FORGE_CARRIER_KEYS:
        if key == "binding_block":
            continue
        payload[key] = _forge_sorted(block[key])
    digest = _forge_hashlib.sha256(_forge_canonical_bytes(payload)).hexdigest()

    binding = _forge_sorted(block["binding_block"]) or {}
    shaped = {key: binding.get(key) for key in _FORGE_BINDING_KEYS}
    ordered = {}
    for key in _FORGE_CARRIER_KEYS:
        ordered[key] = shaped if key == "binding_block" else payload[key]
    return ordered, digest


def _forge_carrier_text():
    payload, digest = _forge_carrier_payload()
    header = (
        _FORGE_BANNER + "\n"
        + "# Derived from solution/grounding.yaml `screening` by solution/recompute.py.\n"
        + "# canonical payload sha256 (binding_block excluded from the preimage): " + digest + "\n"
    )
    body = _forge_yaml.safe_dump(
        payload, sort_keys=False, default_flow_style=False, allow_unicode=False, width=100
    )
    return header + body


def _forge_emit_carrier():
    """Write the carrier, or in check mode compare and report drift. Never both."""
    argv = list(_forge_sys.argv[1:])
    check = "--check" in argv
    path = _forge_here() / _FORGE_CARRIER
    text = _forge_carrier_text()
    if check:
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        if current == text:
            return 0
        _forge_sys.stderr.write(
            "drift: " + _FORGE_CARRIER + " does not match the carrier derived from the frozen "
            "`screening` block in " + _FORGE_GROUNDING + "\n"
        )
        return 1
    path.write_text(text, encoding="utf-8")
    return 0


_FORGE_INNER_MAIN = main


def main(*args, **kwargs):
    """Run the host generator, then derive the provenance carrier from the frozen block."""
    status = _FORGE_INNER_MAIN(*args, **kwargs)
    drift = _forge_emit_carrier()
    if drift and not status:
        return drift
    return status

# FORGE-SCREENING-CARRIER-END


if __name__ == "__main__":
    raise SystemExit(main())
