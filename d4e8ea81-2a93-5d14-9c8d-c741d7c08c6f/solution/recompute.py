# FORGE-CANARY-BEGIN
# 0: 0f269f15ce4a17c01038cf0aa56f7b7c12866547eee4238e0a59a14e33e75017
# 1: da71193f7407788bd1c0a774ba4f3259261938322662eb5a98bfe5641c599c84
# 2: fccad5dbfc66be9a5cd76973ea38791b159ac179724b72a818e93bfdcedbdb53
# 3: 634ea873186fd6921341c67744df83fa36d3380b1d11a440a047aa095513f48a
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of OER-13 from solution/grounding.yaml alone.

Purity, which is the whole point of this file: it invokes no model, no network,
no clock, no locale and no random source. Its entire input is the bytes of
solution/grounding.yaml plus, for digest binding only, the bytes of
solution/reference.py. Running it twice over frozen bytes produces byte-identical
output, which `--check` asserts by regenerating into memory and comparing.

Derived here, each carrying the generated banner naming solution/grounding.yaml:
    solution/solve.sh
    solution/TRUTH.md
    solution/rubrics.json
    solution/fixtures.json
    tests/test_output.py
    trajectories/golden.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
GROUNDING = HERE / "grounding.yaml"
SOURCE_NAME = "solution/grounding.yaml"
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."


def banner_line(prefix: str) -> str:
    return prefix + BANNER + " Derived from " + SOURCE_NAME + " by solution/recompute.py."


def load() -> dict:
    return yaml.safe_load(GROUNDING.read_text(encoding="utf-8"))


def jdump(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


# --------------------------------------------------------------------------
# solve.sh
# --------------------------------------------------------------------------
def build_solve(g: dict) -> str:
    ref = g["reference"]
    rows = sorted(ref["materialized_files"].items())
    lines = [
        "#!/usr/bin/env bash",
        "# " + banner_line(""),
        "#",
        "# OER-13 solution entry point. Harbor runs this file and nothing else.",
        "#",
        "# The reference composition is carried in solution/reference.py as source",
        "# strings and materialized into a submission tree shaped like environment/, so",
        "# it is graded through exactly the path a submission is graded through.",
        "set -euo pipefail",
        "",
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "BUNDLE=\"$(cd \"${HERE}/..\" && pwd)\"",
        "DEST=\"${OER13_SOLUTION_DEST:-${BUNDLE}/../oer13-solution}\"",
        "",
        "python3 - \"${BUNDLE}\" \"${DEST}\" <<'PY'",
        "import sys",
        "from pathlib import Path",
        "bundle, dest = Path(sys.argv[1]).resolve(), Path(sys.argv[2])",
        "sys.path.insert(0, str(bundle / \"solution\"))",
        "import reference",
        "out = reference.materialize(bundle / \"environment\", dest)",
        "print(\"materialized the reference composition into \" + out.as_posix())",
        "PY",
        "",
        "# The three files the reference actually changes, with the digest each carries",
        "# when materialized. Recorded so a reader can check the tree that was written.",
    ]
    for rel, digest in rows:
        lines.append("#   " + rel + "  sha256:" + digest)
    lines.extend(
        [
            "",
            "echo \"reference materialized; graded quantity is "
            + g["objective"]["metric"]
            + ", direction "
            + g["objective"]["direction"]
            + "\"",
        ]
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# TRUTH.md
# --------------------------------------------------------------------------
def build_truth(g: dict) -> str:
    obs = g["observed"]
    ref = g["reference"]
    out = [
        "<!-- " + banner_line("") + " -->",
        "",
        "# OER-13 TRUTH",
        "",
        "## What is graded",
        "",
        "- metric: " + g["graded_quantity"]["what"],
        "- unit: " + g["objective"]["unit"],
        "- direction: " + g["objective"]["direction"],
        "- computed by: " + g["graded_quantity"]["computed_by"],
        "- weights: " + g["graded_quantity"]["weights"],
        "- never: " + g["graded_quantity"]["never"].strip(),
        "",
        "## Anchors",
        "",
        "- anchors_state: " + g["anchors"]["anchors_state"],
        "- gap: " + g["anchors"]["anchors_gap"],
        "- baseline_metric: null",
        "- target_metric: null",
        "- " + g["anchors"]["reason"].strip(),
        "- formula: `" + g["anchors"]["formula"] + "`",
        "- clip: `" + g["anchors"]["clip"] + "`",
        "",
        "## The reference composition",
        "",
        "- carrier: `" + ref["path"] + "`, sha256 `" + ref["sha256"] + "`",
        "- " + ref["idea"].strip(),
        "",
        "## The measurement that makes this slot what it is",
        "",
        "| quantity | delivered baseline | reference |",
        "|---|---|---|",
        "| parse `chars_retained_ratio` | "
        + repr(obs["stage_readouts"]["baseline_chars_retained_ratio"])
        + " | "
        + repr(obs["stage_readouts"]["reference_chars_retained_ratio"])
        + " |",
        "| tokenize `bytes_per_token` | "
        + repr(obs["stage_readouts"]["baseline_bytes_per_token"])
        + " | "
        + repr(obs["stage_readouts"]["reference_bytes_per_token"])
        + " |",
        "| **graded loss per byte** | "
        + repr(obs["baseline_loss_per_byte"])
        + " | "
        + repr(obs["reference_loss_per_byte"])
        + " |",
        "",
        obs["the_point"].strip(),
        "",
        "## Per-fold, so the improvement is visibly sustained",
        "",
        "| fold | baseline | reference |",
        "|---|---|---|",
    ]
    for name in sorted(obs["baseline_fold_losses"]):
        out.append(
            "| "
            + name
            + " | "
            + repr(obs["baseline_fold_losses"][name])
            + " | "
            + repr(obs["reference_fold_losses"][name])
            + " |"
        )
    out.extend(
        [
            "",
            "## The golden trajectory",
            "",
        ]
    )
    for row in g["golden_trajectory"]:
        out.append(
            str(row["step"])
            + ". **"
            + row["action"]
            + "** — "
            + " ".join(row["detail"].split())
            + " _readout: "
            + str(row["readout"])
            + "_"
        )
    out.extend(
        [
            "",
            "## Statement ambiguity: exactly one graded outcome",
            "",
            "- reading one, from " + g["statement_ambiguity"]["reading_one"]["source"] + ": "
            + " ".join(g["statement_ambiguity"]["reading_one"]["reduces_to"].split()),
            "- reading two, from " + g["statement_ambiguity"]["reading_two"]["source"] + ": "
            + " ".join(g["statement_ambiguity"]["reading_two"]["reduces_to"].split()),
            "- they are the same quantity: "
            + " ".join(g["statement_ambiguity"]["same_quantity_because"].split()),
            "- " + " ".join(g["statement_ambiguity"]["no_second_outcome"].split()),
            "",
            "## Solver egress divergence",
            "",
            "- gap id: " + g["solver_egress_divergence"]["gap_id"],
            "- " + " ".join(g["solver_egress_divergence"]["divergence"].split()),
            "- " + " ".join(g["solver_egress_divergence"]["deny_list_is_not_containment"].split()),
            "",
            "## Reward contract",
            "",
            "- reward path: `" + g["reward_contract"]["reward_path"] + "`, carrier: "
            + g["reward_contract"]["reward_carrier"],
            "- score document: `" + g["reward_contract"]["score_document_path"] + "`",
            "- " + " ".join(g["reward_contract"]["measured_correction_reward_carrier"].split()),
            "- " + " ".join(g["reward_contract"]["measured_correction_shared_root"].split()),
            "",
            "## Tier",
            "",
            "- target tier: " + g["tier"]["target_tier"]
            + ", tier_exemption_granted: " + g["tier"]["tier_exemption_granted"],
            "- anchorable tier: " + g["tier"]["anchorable_tier"],
            "- " + " ".join(g["tier"]["perception_exemption"].split()),
        ]
    )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# rubrics.json: the SOLUTION graded against its reference answer.
# --------------------------------------------------------------------------
def build_rubrics(g: dict) -> str:
    obs = g["observed"]
    ref = g["reference"]
    criteria = [
        {
            "id": "reference-bytes-are-the-bound-reference",
            "asks": "Does solution/reference.py digest to the sha256 this slot binds?",
            "expected": ref["sha256"],
            "reduction": "VALUE",
        },
        {
            "id": "materialized-files-are-the-bound-three",
            "asks": "Does materializing the reference write exactly the three free files with the bound digests?",
            "expected": {rel: digest for rel, digest in sorted(ref["materialized_files"].items())},
            "reduction": "VALUE",
        },
        {
            "id": "reference-scores-full-reward",
            "asks": "Does the reference drive the live gate chain to full reward?",
            "expected": {
                "reward": obs["reward_at_reference"],
                "reason": obs["reward_reason_at_reference"],
            },
            "reduction": "EFFECT",
        },
        {
            "id": "reference-loss-matches-the-recorded-answer",
            "asks": "Is the graded loss the reference reaches the recorded one?",
            "expected": {
                "reference_loss_per_byte": obs["reference_loss_per_byte"],
                "baseline_loss_per_byte": obs["baseline_loss_per_byte"],
            },
            "reduction": "VALUE",
        },
        {
            "id": "improvement-is-sustained-on-every-fold",
            "asks": "Does the reference improve on the baseline at every fold, not only overall?",
            "expected": {
                name: {
                    "baseline": obs["baseline_fold_losses"][name],
                    "reference": obs["reference_fold_losses"][name],
                }
                for name in sorted(obs["baseline_fold_losses"])
            },
            "reduction": "INVARIANT",
        },
        {
            "id": "reference-trades-a-stage-readout-for-the-composition",
            "asks": "Does the reference lower its own parse retention readout while lowering the graded loss?",
            "expected": {
                "baseline_chars_retained_ratio": obs["stage_readouts"]["baseline_chars_retained_ratio"],
                "reference_chars_retained_ratio": obs["stage_readouts"]["reference_chars_retained_ratio"],
                "retention_readout_gets_worse": True,
                "graded_loss_gets_better": True,
            },
            "reduction": "DIVERGENCE",
        },
        {
            "id": "reference-consumes-exactly-the-bound-budget",
            "asks": "Does the reference stream carry at least the bound budget and the harness consume exactly it?",
            "expected": {
                "token_budget_tokens": obs["token_budget_tokens_in_force"],
                "tokens_available": obs["tokens_available_reference"],
                "tokens_consumed": obs["tokens_consumed_reference"],
            },
            "reduction": "EFFECT",
        },
        {
            "id": "reference-is-graded-at-the-bound-evaluation-point",
            "asks": "Are the graded weights the harness-written weights at the bound checkpoint?",
            "expected": {
                "bound_evaluation_point": obs["bound_evaluation_point"],
                "bound_checkpoint_id": obs["bound_checkpoint_id"],
                "writer": "harness",
            },
            "reduction": "VALUE",
        },
    ]
    return jdump(
        {
            "_generated": banner_line(""),
            "_source": SOURCE_NAME,
            "schema": "forge.solution_rubrics/v1",
            "slot": g["slot"],
            "grades": "the solution against its reference answer",
            "not_a_substitute_for": "tests/rubrics.jsonl, which is judged against the trajectory",
            "anchors_state": g["anchors"]["anchors_state"],
            "anchors_gap": g["anchors"]["anchors_gap"],
            "criteria": criteria,
        }
    )


# --------------------------------------------------------------------------
# fixtures.json: the planted-defect fixtures adequacy.py drives.
# --------------------------------------------------------------------------
def build_fixtures(g: dict) -> str:
    rows = []
    for row in g["checkers"]:
        sub = row.get("substitution")
        rows.append(
            {
                "checker": row["id"],
                "selector": row["selector"],
                "reduction": row["reduction"],
                "zero_reason": row["zero_reason"],
                "fixture": row["fixture"],
                "fires_when": " ".join(str(row["fires_when"]).split()),
                "rejecting_half_substituted": bool(isinstance(sub, dict)),
                "substitution": sub if isinstance(sub, dict) else "none",
            }
        )
    return jdump(
        {
            "_generated": banner_line(""),
            "_source": SOURCE_NAME,
            "schema": "forge.checker_fixtures/v1",
            "slot": g["slot"],
            "accepting_half": {
                "fixture": "reference-composition",
                "reference_sha256": g["reference"]["sha256"],
                "expected_reward": g["observed"]["reward_at_reference"],
                "expected_reason": g["observed"]["reward_reason_at_reference"],
            },
            "rejecting_halves": sorted(rows, key=lambda item: item["checker"]),
            "negative_controls": sorted(
                (
                    {
                        "id": item["id"],
                        "what": " ".join(str(item["what"]).split()),
                        "expected_reason": item["expected_reason"],
                    }
                    for item in g["negative_controls"]
                ),
                key=lambda item: item["id"],
            ),
            "stale_controls": sorted(
                (dict(item) for item in g["stale_controls"]), key=lambda item: item["id"]
            ),
        }
    )


# --------------------------------------------------------------------------
# trajectories/golden.jsonl
# --------------------------------------------------------------------------
def build_golden(g: dict) -> str:
    header = {
        "_generated": banner_line(""),
        "_source": SOURCE_NAME,
        "schema": "forge.golden_trajectory/v1",
        "slot": g["slot"],
        "metric": g["objective"]["metric"],
        "direction": g["objective"]["direction"],
        "primary_archetype": g["objective"]["primary_archetype"],
    }
    lines = [json.dumps(header, sort_keys=True, ensure_ascii=True)]
    for row in g["golden_trajectory"]:
        lines.append(
            json.dumps(
                {
                    "step": row["step"],
                    "action": row["action"],
                    "detail": " ".join(str(row["detail"]).split()),
                    "readout": str(row["readout"]),
                },
                sort_keys=True,
                ensure_ascii=True,
            )
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# tests/test_output.py: the compiled surface tests/test.sh runs first.
# --------------------------------------------------------------------------
TEST_PRELUDE = '''#!/usr/bin/env python3
"""{banner}

The compiled surface. tests/test.sh runs this BEFORE tests/grade.py, so a
manifest whose bindings have drifted away from the checkers aborts the verifier
before anything is graded and the EXIT trap writes an attributed zero rather than
a number resting on a stale binding.

Every test below is either a pure text binding check over frozen bundle bytes or a
drive of one live selector over a frozen in-memory fixture. It reads no clock,
opens no socket, consults no random source, imports no submission and runs no
composition. Grading the run is tests/grade.py's job, not this file's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402

MANIFEST = (HERE / "checkers.yaml").read_text(encoding="utf-8")
CHECKER_SOURCE = (HERE / "checkers.py").read_text(encoding="utf-8")
GRADE_SOURCE = (HERE / "grade.py").read_text(encoding="utf-8")

REWARD_PATH = "{reward_path}"
SCORE_DOCUMENT_PATH = "{score_document_path}"


def block(ident: str) -> str:
    marker = "\\n  - id: " + ident + "\\n"
    if marker not in MANIFEST:
        raise AssertionError("tests/checkers.yaml declares no checker " + ident)
    tail = MANIFEST.split(marker, 1)[1]
    return tail.split("\\n  - id: ", 1)[0]


def bind(ident: str, reduction: str, selector: str, zero_reason: str) -> None:
    body = block(ident)
    for needle in (
        "reduction: " + reduction,
        "selector: " + selector,
        "zero_reason: " + zero_reason,
        "required: true",
        "reached_by: tests/grade.py",
        'compiled_test: "tests/test_output.py::test_' + ident + '"',
    ):
        if needle not in body:
            raise AssertionError(ident + " does not bind " + repr(needle) + " in tests/checkers.yaml")
    if "def " + selector + "(" not in CHECKER_SOURCE:
        raise AssertionError("tests/checkers.py defines no " + selector)
    if selector not in GRADE_SOURCE:
        raise AssertionError("tests/grade.py never reaches " + selector)
    if '"' + zero_reason + '"' not in CHECKER_SOURCE:
        raise AssertionError(selector + " never emits the zero reason " + zero_reason)
    if '"' + ident + '": "' + zero_reason + '"' not in GRADE_SOURCE:
        raise AssertionError("tests/grade.py maps " + ident + " to a different zero reason")


def test_reward_contract_is_bound() -> None:
    for needle in (
        "reward_path: " + REWARD_PATH,
        "score_document_path: " + SCORE_DOCUMENT_PATH,
        "mode: required_pass",
    ):
        if needle not in MANIFEST:
            raise AssertionError("tests/checkers.yaml does not bind " + repr(needle))
    if "reward.txt" not in GRADE_SOURCE or "score.json" not in GRADE_SOURCE:
        raise AssertionError("tests/grade.py does not write both reward carriers")


def test_anchors_declared_absent_not_invented() -> None:
    for needle in ("anchors_state: absent", "{anchors_gap}"):
        if needle not in MANIFEST:
            raise AssertionError("tests/checkers.yaml does not declare " + repr(needle))
    if '"baseline_metric": None' not in GRADE_SOURCE or '"target_metric": None' not in GRADE_SOURCE:
        raise AssertionError("tests/grade.py invents an anchor instead of declaring it absent")


def test_grading_process_never_imports_the_submission() -> None:
    for banned in ("import submission", "importlib", "exec(", "eval("):
        if banned in GRADE_SOURCE:
            raise AssertionError("tests/grade.py reaches the submission through " + banned)
    if "import random" in CHECKER_SOURCE or "import time" in CHECKER_SOURCE:
        raise AssertionError("tests/checkers.py reads a random source or a clock")

'''

TEST_CASE = '''
def test_{ident}() -> None:
    bind("{ident}", "{reduction}", "{selector}", "{zero_reason}")

'''

FLOOR_ANCHOR_FIXTURE = '''
# The discovery-value fixture pair, frozen from solution/grounding.yaml `observed`.
# The accepting half feeds the selector the reference loss the live environment
# establishes and the rejecting half feeds it one the environment does not, so a
# selector that stopped reading the value would fail one half or the other.
FLOOR_ANCHOR = {
    "established": ANCHOR_RIGHT,
    "unestablished": ANCHOR_WRONG,
    "tolerance": ANCHOR_TOLERANCE,
    "budget": ANCHOR_BUDGET,
    "zero_reason": "ANCHOR_REASON",
}


def _floor_anchor_outcome(measured: float):
    handle = checkers.Harness(
        submission=HERE,
        verifier=HERE,
        telemetry={"floor": {"loss_per_byte": measured}},
        bound={
            "reference_loss_per_byte": FLOOR_ANCHOR["established"],
            "reference_loss_tolerance": FLOOR_ANCHOR["tolerance"],
            "reference_anchor_budget_tokens": FLOOR_ANCHOR["budget"],
            "token_budget_tokens": FLOOR_ANCHOR["budget"],
        },
        eval_split=HERE / "bound.json",
        folds={},
    )
    return checkers.check_floor_anchor_matches_bound_reference(handle)


def test_floor_anchor_is_bound_on_the_admin_plane() -> None:
    bound = json.loads((HERE / "bound.json").read_text(encoding="utf-8"))
    for key, expected in (
        ("reference_loss_per_byte", FLOOR_ANCHOR["established"]),
        ("reference_loss_tolerance", FLOOR_ANCHOR["tolerance"]),
        ("reference_anchor_budget_tokens", FLOOR_ANCHOR["budget"]),
    ):
        if key not in bound:
            raise AssertionError("tests/bound.json binds no " + key)
        if bound[key] != expected:
            raise AssertionError(
                "tests/bound.json binds " + key + "=" + repr(bound[key])
                + " against the grounded " + repr(expected)
            )


def test_floor_anchor_fixture_accepts_the_established_reference() -> None:
    outcome = _floor_anchor_outcome(FLOOR_ANCHOR["established"])
    if not outcome.passed:
        raise AssertionError("the established reference loss was refused: " + outcome.detail)


def test_floor_anchor_fixture_rejects_an_unestablished_reference() -> None:
    outcome = _floor_anchor_outcome(FLOOR_ANCHOR["unestablished"])
    if outcome.passed:
        raise AssertionError("an unestablished reference loss was accepted, so nothing reads the value")
    if outcome.reason != FLOOR_ANCHOR["zero_reason"]:
        raise AssertionError("the rejecting half carries the reason " + repr(outcome.reason))

'''


def build_floor_anchor_fixture(g: dict) -> str:
    obs = g["observed"]
    established = obs["reference_loss_per_byte"]
    row = next(item for item in g["checkers"] if item["id"] == "floor_anchor_matches_bound_reference")
    return (
        FLOOR_ANCHOR_FIXTURE.replace("ANCHOR_RIGHT", repr(established))
        .replace("ANCHOR_WRONG", repr(round(established + 0.1, 9)))
        .replace("ANCHOR_TOLERANCE", repr(obs["reference_loss_tolerance"]))
        .replace("ANCHOR_BUDGET", repr(obs["token_budget_tokens_in_force"]))
        .replace("ANCHOR_REASON", row["zero_reason"])
    )


TEST_MAIN = '''
def main() -> int:
    failures = []
    for name, case in sorted(globals().items()):
        if not name.startswith("test_") or not callable(case):
            continue
        try:
            case()
        except AssertionError as exc:
            failures.append(name + ": " + str(exc))
    for row in failures:
        print("FAIL " + row, file=sys.stderr)
    total = len([n for n in globals() if n.startswith("test_")])
    print("compiled surface: " + str(total - len(failures)) + "/" + str(total) + " bindings hold")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_test_output(g: dict) -> str:
    parts = [
        TEST_PRELUDE.format(
            banner=banner_line(""),
            reward_path=g["reward_contract"]["reward_path"],
            score_document_path=g["reward_contract"]["score_document_path"],
            anchors_gap=g["anchors"]["anchors_gap"],
        )
    ]
    for row in sorted(g["checkers"], key=lambda item: item["id"]):
        parts.append(
            TEST_CASE.format(
                ident=row["id"],
                reduction=row["reduction"],
                selector=row["selector"],
                zero_reason=row["zero_reason"],
            )
        )
    parts.append(build_floor_anchor_fixture(g))
    parts.append(TEST_MAIN)
    return "".join(parts)


# --------------------------------------------------------------------------
# Drive
# --------------------------------------------------------------------------
BUILDERS = {
    "solution/solve.sh": build_solve,
    "solution/TRUTH.md": build_truth,
    "solution/rubrics.json": build_rubrics,
    "solution/fixtures.json": build_fixtures,
    "tests/test_output.py": build_test_output,
    "trajectories/golden.jsonl": build_golden,
}

EXECUTABLE = ("solution/solve.sh",)


def render() -> dict:
    g = load()
    return {rel: BUILDERS[rel](g) for rel in sorted(BUILDERS)}


def write(rendered: dict) -> list:
    written = []
    for rel in sorted(rendered):
        target = BUNDLE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered[rel], encoding="utf-8")
        if rel in EXECUTABLE:
            target.chmod(0o755)
        written.append(rel)
    return written


def digests(rendered: dict) -> dict:
    return {
        rel: hashlib.sha256(rendered[rel].encode("utf-8")).hexdigest()
        for rel in sorted(rendered)
    }


def main(argv: list) -> int:
    first = render()
    second = render()
    if digests(first) != digests(second):
        print("recompute is not deterministic over frozen bytes", file=sys.stderr)
        return 1
    if "--check" in argv:
        stale = [
            rel
            for rel in sorted(first)
            if not (BUNDLE / rel).is_file()
            or (BUNDLE / rel).read_text(encoding="utf-8") != first[rel]
        ]
        if stale:
            print("stale generated artifacts: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(json.dumps({"checked": sorted(first), "stale": []}, sort_keys=True))
        return 0
    written = write(first)
    print(json.dumps({"source": SOURCE_NAME, "written": written, "sha256": digests(first)},
                     indent=2, sort_keys=True))
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
    "screening_measured_at",
    "screening_interval_days",
    "screening_expires_at",
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
    raise SystemExit(main(sys.argv[1:]))
