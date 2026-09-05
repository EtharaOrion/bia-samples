# FORGE-CANARY-BEGIN
# 0: 7c65987a93e7c0de525c85fc4d23215872d1e6cdf503db48c0e534c3c636c438
# 1: 080fea0a83c0a67aa2c0c4cc6b605e5330f64889452d22150a6ff87f7601eb9c
# 2: 56793f9d8c3cb33263974512632914185016fb37bc838fcb42c3c269938b2b9e
# 3: dc491d008668dfdea7b7030a96dd7c541917a3d035f7a9256cb9093b1d380af2
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of OER-29 from solution/grounding.yaml alone.

Generated here, and nowhere else:

    solution/warehouse_spec.json           the frozen warehouse spec the image build consumes
    environment/warehouse.db               the frozen warehouse, byte-identical to the image's
    tests/baseline_plan.sql                the frozen baseline plan, the graded denominator
    tests/bound.json                       every bound constant the checkers compare against
    tests/test_output.py                   the compiled per-checker and per-fixture tests
    solution/solve.sh                      the solution entry point
    solution/TRUTH.md                      the derivation write-up
    solution/rubrics.json                  the solution-against-reference rubric
    solution/fixtures/reference_run/       the golden trajectory
    solution/fixtures/rejecting/           the frozen rejecting fixtures

No model is invoked, no socket is opened, no clock is read, no locale is consulted and no random
source is drawn from. The warehouse comes from a fixed integer linear congruential recurrence
whose seed is recorded in grounding.yaml, which is arithmetic rather than randomness. The bound
costs are produced by running the pinned executor over the frozen database, so they are recorded
fixtures and never live measurements of a host.

NEITHER DISCOVERY VALUE IS READ FROM GROUNDING. The excluded row identifier comes out of running
the build; the closure cardinality comes out of walking the closure relation over the graph that
build produced. grounding.yaml carries the spec and the rules, and no node identifier and no
cardinality appear anywhere in it.

Running this twice over frozen bytes produces byte-identical output. `--check` reports drift and
writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "environment"))
sys.path.insert(0, str(BUNDLE / "tests"))

import checkers  # noqa: E402
import planrun  # noqa: E402
import warehouse_build  # noqa: E402

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
SUBMISSION_SCHEMA = "oer29.submission/v1"


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def load_grounding():
    with (HERE / "grounding.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def spec_from(ground):
    block = dict(ground["warehouse_spec"])
    block.pop("generator", None)
    block.pop("band_rationale", None)
    block["schema"] = "oer29.warehouse_spec/v1"
    return block


def canonical_json(payload):
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def exclusion_literal(nodes):
    return ", ".join("'" + str(node) + "'" for node in nodes)


def folded_plan(ground, nodes):
    template = ground["plans"]["reference"]["template"]
    return template.replace("{exclusion_list}", exclusion_literal(nodes))


def baseline_plan(ground):
    return ground["plans"]["baseline"]["sql"]


def cost_of(database, plan_text):
    report = planrun.run_plan(str(database), plan_text)
    return report


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def derive(ground, database_path):
    """Build the warehouse, then discover both values by doing the work over the built state."""
    spec = spec_from(ground)
    tables = warehouse_build.compose(spec)
    warehouse_build.write_database(database_path, tables)

    excluded = tables["quarantined_node_id"]
    layers = tables["closure_layers"]
    closure = sorted(tables["closure_nodes"])
    cardinality = len(closure)

    base_text = baseline_plan(ground)
    reference_text = folded_plan(ground, closure)
    base_report = cost_of(database_path, base_text)
    reference_report = cost_of(database_path, reference_text)

    required_rows = [list(row) for row in reference_report["rows"]]
    if [list(row) for row in base_report["rows"]] != required_rows:
        raise SystemExit(
            "the baseline plan and the reference plan disagree on the required result set, so "
            "the reference plan is not a plan for this task; refusing to emit a bound file"
        )

    return {
        "spec": spec,
        "tables": tables,
        "excluded_node_id": excluded,
        "closure_nodes": closure,
        "closure_layers": layers,
        "closure_cardinality": cardinality,
        "baseline_plan": base_text,
        "reference_plan": reference_text,
        "baseline_report": base_report,
        "reference_report": reference_report,
        "required_rows": required_rows,
    }


def submission_payload(derived, plan_report, overrides=None):
    payload = {
        "schema": SUBMISSION_SCHEMA,
        "excluded_node_id": derived["excluded_node_id"],
        "closure_nodes": derived["closure_nodes"],
        "closure_layers": derived["closure_layers"],
        "closure_cardinality": derived["closure_cardinality"],
        "admitted_row_count": plan_report["row_count"],
        "reported_plan_cost": plan_report["cost_steps"],
        "reported_baseline_cost": derived["baseline_report"]["cost_steps"],
    }
    if overrides:
        payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Bound constants
# ---------------------------------------------------------------------------


def build_bound(ground, derived, spec_text, warehouse_sha256):
    spec = derived["spec"]
    base_cost = derived["baseline_report"]["cost_steps"]
    reference_cost = derived["reference_report"]["cost_steps"]
    target_speedup = round(base_cost / reference_cost, 6)
    return {
        "anchors_gap": ground["anchors"]["gap_id"],
        "anchors_state": ground["anchors"]["anchors_state"],
        "banner": BANNER,
        "baseline_plan_cost": base_cost,
        "baseline_plan_sha256": sha256_text(derived["baseline_plan"]),
        "check_alpha": int(spec["check_coefficients"]["alpha"]),
        "check_beta": int(spec["check_coefficients"]["beta"]),
        "check_gamma": int(spec["check_coefficients"]["gamma"]),
        "check_modulus": int(spec["check_modulus"]),
        "closure_digest": checkers.closure_digest(derived["closure_nodes"]),
        "derives_count": len(derived["tables"]["derives"]),
        "family_baseline_metric": ground["anchors"]["family_baseline_metric"],
        "family_target_metric": ground["anchors"]["family_target_metric"],
        "instance_anchor_gap": ground["anchors"]["instance_local_anchors"]["declared_gap"],
        "instance_baseline_speedup": 1.0,
        "instance_target_speedup": target_speedup,
        "lineage_count": len(derived["tables"]["lineage"]),
        "reading_count": len(derived["tables"]["reading"]),
        "reference_plan_cost": reference_cost,
        "required_rows_digest": checkers.rows_digest(derived["required_rows"]),
        "schema": "oer29.bound/v1",
        "source": SOURCE,
        "spec_sha256": sha256_text(spec_text),
        "warehouse_sha256": warehouse_sha256,
    }


# ---------------------------------------------------------------------------
# Rejecting fixtures
# ---------------------------------------------------------------------------


def rejecting_fixtures(ground, derived, database_path):
    """Each fixture is a frozen (plan.sql, submission.json) pair with its expected refusal."""
    closure = derived["closure_nodes"]
    layers = derived["closure_layers"]
    reference_plan = derived["reference_plan"]
    reference_report = derived["reference_report"]
    reference_submission = submission_payload(derived, reference_report)

    out = []

    # -- near miss: one scalar moved, the excluded row identifier ------------
    index = int(str(derived["excluded_node_id"]).split("-")[1])
    known = {str(row["node_id"]) for row in derived["tables"]["lineage"]}
    neighbour = None
    for step in (1, -1, 2, -2):
        candidate = warehouse_build.node_name(index + step)
        if candidate in known and candidate != derived["excluded_node_id"]:
            neighbour = candidate
            break
    if neighbour is None:
        raise SystemExit("no neighbouring lineage identifier exists, so the near miss cannot be cut")
    out.append(
        {
            "id": "guessed_neighbouring_row",
            "plan": reference_plan,
            "submission": dict(reference_submission, excluded_node_id=neighbour),
            "expects": "quarantine-row-misidentified",
            "failed_checker": "quarantine_row_identified",
            "near_miss": True,
            "moved_keys": ["excluded_node_id"],
        }
    )

    # -- the first-hop stop, which is the archetype's own failure mode -------
    first_hop_layers = [list(layers[0]), list(layers[1])] if len(layers) > 1 else [list(layers[0])]
    first_hop_nodes = sorted({node for layer in first_hop_layers for node in layer})
    first_hop_plan = folded_plan(ground, first_hop_nodes)
    first_hop_report = cost_of(database_path, first_hop_plan)
    out.append(
        {
            "id": "closure_stopped_at_first_hop",
            "plan": first_hop_plan,
            "submission": submission_payload(
                derived,
                first_hop_report,
                {
                    "closure_nodes": first_hop_nodes,
                    "closure_layers": first_hop_layers,
                    "closure_cardinality": len(first_hop_nodes),
                },
            ),
            "expects": "closure-incomplete",
            "failed_checker": "closure_set_complete",
            "near_miss": False,
            "moved_keys": [
                "admitted_row_count",
                "closure_cardinality",
                "closure_layers",
                "closure_nodes",
                "reported_plan_cost",
            ],
        }
    )

    # -- near misses: one scalar moved, the closure cardinality --------------
    for name, delta in (("cardinality_minus_one", -1), ("cardinality_plus_one", 1)):
        out.append(
            {
                "id": name,
                "plan": reference_plan,
                "submission": dict(
                    reference_submission,
                    closure_cardinality=derived["closure_cardinality"] + delta,
                ),
                "expects": "closure-cardinality-unattested",
                "failed_checker": "closure_cardinality_attested",
                "near_miss": True,
                "moved_keys": ["closure_cardinality"],
            }
        )

    # -- the traversal witness flattened to a single layer -------------------
    out.append(
        {
            "id": "layers_flattened",
            "plan": reference_plan,
            "submission": dict(reference_submission, closure_layers=[list(closure)]),
            "expects": "traversal-order-unwitnessed",
            "failed_checker": "closure_layers_witnessed",
            "near_miss": False,
            "moved_keys": ["closure_layers"],
        }
    )

    # -- a plan that excludes nothing ---------------------------------------
    admit_all = (
        "SELECT reading_id, node_id, value\n  FROM reading\n ORDER BY reading_id\n"
    )
    admit_all_report = cost_of(database_path, admit_all)
    out.append(
        {
            "id": "plan_admits_every_reading",
            "plan": admit_all,
            "submission": submission_payload(derived, admit_all_report),
            "expects": "excluded-row-admitted",
            "failed_checker": "excluded_rows_absent",
            "near_miss": False,
            "moved_keys": ["admitted_row_count", "reported_plan_cost"],
        }
    )

    # -- a plan that also drops one row the closure does not reach -----------
    closure_set = set(closure)
    extra = None
    for row in sorted(derived["tables"]["reading"], key=lambda entry: entry["reading_id"]):
        if str(row["node_id"]) not in closure_set:
            extra = str(row["node_id"])
            break
    if extra is None:
        raise SystemExit("every reading sits inside the closure, so the over-exclusion cannot be cut")
    over_nodes = sorted(closure_set | {extra})
    over_plan = folded_plan(ground, over_nodes)
    over_report = cost_of(database_path, over_plan)
    out.append(
        {
            "id": "plan_drops_one_admitted_row",
            "plan": over_plan,
            "submission": submission_payload(derived, over_report),
            "expects": "result-set-divergence",
            "failed_checker": "result_set_equality",
            "near_miss": False,
            "moved_keys": ["admitted_row_count", "reported_plan_cost"],
        }
    )

    # -- near miss: one scalar moved, the reported cost ----------------------
    out.append(
        {
            "id": "cost_readout_mismatch",
            "plan": reference_plan,
            "submission": dict(
                reference_submission,
                reported_plan_cost=reference_report["cost_steps"] + 1,
            ),
            "expects": "cost-readout-mismatch",
            "failed_checker": "plan_cost_readout_exact",
            "near_miss": True,
            "moved_keys": ["reported_plan_cost"],
        }
    )

    # -- the baseline plan handed back unchanged -----------------------------
    out.append(
        {
            "id": "baseline_plan_resubmitted",
            "plan": derived["baseline_plan"],
            "submission": submission_payload(derived, derived["baseline_report"]),
            "expects": "plan-not-faster-than-baseline",
            "failed_checker": "plan_faster_than_baseline",
            "near_miss": False,
            "moved_keys": ["reported_plan_cost"],
        }
    )

    # -- a plan that is not a single read-only select ------------------------
    out.append(
        {
            "id": "plan_writes_to_the_warehouse",
            "plan": "DELETE FROM reading WHERE reading_id = 1\n",
            "submission": dict(reference_submission),
            "expects": "plan-not-read-only-single-statement",
            "failed_checker": "plan_is_single_read_only_select",
            "near_miss": False,
            "moved_keys": [],
        }
    )

    return out


# ---------------------------------------------------------------------------
# Rendered artifacts
# ---------------------------------------------------------------------------


def render_solve_sh():
    return """#!/usr/bin/env bash
# {banner}
# Source: {source}. Regenerate with solution/recompute.py.
#
# The solution entry point. It runs the oracle, which reads the built warehouse back through the
# harness handle, applies the exclusion rule, walks the closure to fixpoint, folds the closure
# into a plan and writes plan.sql and submission.json into the workspace.
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BUNDLE="$(cd "${{HERE}}/.." && pwd)"
WORKSPACE="${{OER29_WORKSPACE:-${{PWD}}/workspace}}"

mkdir -p "${{WORKSPACE}}"
python3 "${{HERE}}/reference.py" \\
    --warehouse "${{BUNDLE}}/environment/warehouse.db" \\
    --workspace "${{WORKSPACE}}"

echo "submission written to ${{WORKSPACE}}/submission.json"
""".format(banner=BANNER, source=SOURCE)


def render_truth(ground, derived, bound):
    layers = derived["closure_layers"]
    profile = ", ".join(str(len(layer)) for layer in layers)
    excluded_readings = bound["reading_count"] - len(derived["required_rows"])
    rows = [
        ("excluded row identifier", derived["excluded_node_id"]),
        ("closure cardinality", str(derived["closure_cardinality"])),
        ("closure depth", str(len(layers) - 1)),
        ("closure layer profile", profile),
        ("readings inside the closure", str(excluded_readings)),
        ("required result rows", str(len(derived["required_rows"]))),
        ("baseline plan cost, harness steps", str(bound["baseline_plan_cost"])),
        ("reference plan cost, harness steps", str(bound["reference_plan_cost"])),
        ("instance_baseline_speedup", str(bound["instance_baseline_speedup"])),
        ("instance_target_speedup", str(bound["instance_target_speedup"])),
    ]
    table = "\n".join(
        ["| quantity | value |", "|---|---|"]
        + ["| " + key + " | " + value + " |" for key, value in rows]
    )
    first_hop = 1 + (len(layers[1]) if len(layers) > 1 else 0)
    return """# TRUTH.md

{banner}

Source: `{source}`. Regenerate with `solution/recompute.py`.

## What is graded

Query plan speedup over the frozen baseline plan, gated on result-set equality. Direction: higher is better. The gate is HARD: a plan whose result set differs from the required result set by one row, one column or one ordering position scores exactly 0.0 with the reason `result-set-divergence`, however cheap it is.

Cost is the database engine's own virtual-machine step count for the whole result, read off the engine progress callback inside `environment/planrun.py`. It is not a wall-clock time and it is never presented as one, under the declared gap `gap-oer-29-cost-is-a-step-count-not-a-wall-clock-time`. The step count is only portable while the engine build is fixed, which is why both Dockerfiles pin the same base image by digest and why `tests/grade.py` refuses with `engine-cost-model-drift` if the frozen baseline plan does not cost what it cost when these constants were derived.

## The archetype

Exclusion closure. Exactly one lineage row fails the frozen integrity rule, and excluding it invalidates every row that transitively depends on it. The dependency relation is the union of two edge sets, the parent edges in `lineage` and the edges in `derives`, so a node can be reached by more than one path and the closure has to be run to fixpoint rather than read off one join.

The failure mode the slot targets is stopping at the first hop. The excluded row has {first_hop_minus_one} immediate successors, so a first-hop answer names {first_hop} nodes; the closure actually reaches {cardinality} nodes at depth {depth}. A first-hop answer therefore looks like an answer, produces a plan that runs, and is refused with `closure-incomplete`.

## The two discovery values

Neither value appears in `instruction.md`, in `task.toml`, or in any authored file under `environment/`. Both are established by the first Docker build stage, which composes the warehouse from the frozen spec and perturbs exactly one declared checksum, and both are read back by querying the built database through `environment/probe.py run`.

{table}

## How the oracle found them

The oracle does not know either value in advance and does not read them from a file. It runs one read-only query for the row whose declared checksum disagrees with its own payload, then walks the union of the two edge relations breadth first from that row until the frontier is empty, recording the layers as it goes. The cardinality is the length of what the walk returned. Only then does it fold the result into a plan.

`solution/oracle_run.jsonl` is the step trace that run actually emitted inside the built environment image, and the `plan.sql` and `submission.json` it wrote there are byte-identical to the frozen fixture under `solution/fixtures/reference_run/`. That recording is not a generated artifact and `recompute.py --check` does not cover it.

## The reference plan

The closure is computed once, off the graded path, and folded into the plan as a literal exclusion list. The recursive term of the baseline plan, the two joins it drives and the whole scan of the `derives` table then disappear, and what is left is one scan of `reading` against a small set membership test in `reading_id` order.

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F6 carries no measured family baseline or target, so none is invented. The reward schema is bound in full and the two normalisation numbers above are measured on this slot's own frozen substrate inside the pinned image; they are instance-local and are NOT family anchors, under gap `gap-oer-29-instance-local-anchors-not-family-anchors`.

raw = (agent_metric - instance_baseline_speedup) / (instance_target_speedup - instance_baseline_speedup)

score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and beating it also scores 1.0.

## The clock

No clock is read anywhere on the graded path. The cost is a step count the engine reports, every checker reads recorded state through the `tests/checkers.py` Harness handle, and `recompute.py` reads no clock either.

## What is not established

No signed pilot has been run against this bundle, under `gap-oer-29-no-signed-pilot`, so the target tier is an authoring target and not a measured one. The bundle carries no detached provenance signature, under `gap-oer-29-no-detached-signature`, because the authoring lane holds no signing key; the absence is recorded rather than filled in.
""".format(
        banner=BANNER,
        source=SOURCE,
        table=table,
        cardinality=derived["closure_cardinality"],
        depth=len(layers) - 1,
        first_hop=first_hop,
        first_hop_minus_one=first_hop - 1,
    )


def render_rubrics(ground, derived, bound):
    return {
        "banner": BANNER,
        "criteria": [
            {
                "id": "excluded-row-discovered",
                "statement": "The submission names the lineage row the frozen exclusion rule selects.",
                "comparison": "string equality against the row the verifier re-derives from the frozen lineage",
                "reference_value": derived["excluded_node_id"],
            },
            {
                "id": "closure-run-to-fixpoint",
                "statement": "The declared closure is the whole transitive closure and not its first hop.",
                "comparison": "set equality against the closure the verifier walks over the frozen edges",
                "reference_value": derived["closure_nodes"],
            },
            {
                "id": "traversal-witnessed",
                "statement": "The declared breadth-first layers reproduce the traversal the cardinality came out of.",
                "comparison": "list equality, layer for layer",
                "reference_value": derived["closure_layers"],
            },
            {
                "id": "cardinality-attested",
                "statement": "The declared closure cardinality agrees with the verifier's closure, the submission's own node list and its own layers.",
                "comparison": "integer equality on all three readings",
                "reference_value": derived["closure_cardinality"],
            },
            {
                "id": "result-set-equal",
                "statement": "The plan returns the required result set exactly, in reading_id order.",
                "comparison": "row-for-row equality against the verifier's own required result set",
                "reference_value": len(derived["required_rows"]),
            },
            {
                "id": "speedup-reached",
                "statement": "The plan reaches the instance target speedup over the frozen baseline plan.",
                "comparison": "harness-counted speedup is >= reference_value",
                "reference_value": bound["instance_target_speedup"],
            },
        ],
        "judged": "the solution against its reference answer",
        "schema": "forge.solution_rubrics/v1",
        "source": SOURCE,
    }


def render_test_output(fixtures):
    lines = [
        '"""' + BANNER,
        "",
        "Source: " + SOURCE + ". Regenerate with solution/recompute.py.",
        "",
        "One compiled test per graded checker over the frozen reference run, one per frozen",
        "rejecting fixture, and a byte-minimality test for every near miss. Every test drives the",
        "real checkers in tests/checkers.py through the real grader in tests/grade.py. No test",
        "reads a clock.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "TESTS = Path(__file__).resolve().parent",
        "BUNDLE = TESTS.parent",
        "sys.path.insert(0, str(TESTS))",
        "",
        "import checkers  # noqa: E402",
        "import grade  # noqa: E402",
        "",
        "REFERENCE = BUNDLE / 'solution' / 'fixtures' / 'reference_run'",
        "REJECTING = BUNDLE / 'solution' / 'fixtures' / 'rejecting'",
        "",
        "",
        "def _verdict_over(workspace):",
        "    harness = grade.build_harness(BUNDLE, workspace)",
        "    bound = grade.load_bound(BUNDLE)",
        "    return grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))",
        "",
        "",
        "def _outcomes():",
        "    harness = grade.build_harness(BUNDLE, REFERENCE)",
        "    bound = grade.load_bound(BUNDLE)",
        "    return {row.ident: row for row in grade.run_all(harness, bound)}",
        "",
        "",
        "def test_every_checker_is_declared():",
        "    declared = {ident for ident, _selector in grade.CHECKER_ORDER}",
        "    assert declared == set(_outcomes())",
        "",
    ]
    for ident, _selector in TEST_CHECKER_IDS:
        lines += [
            "",
            "def test_" + ident + "():",
            "    outcome = _outcomes()['" + ident + "']",
            "    assert outcome.passed, outcome.detail",
            "",
        ]
    lines += [
        "",
        "def test_reference_scores_full_reward():",
        "    verdict = _verdict_over(REFERENCE)",
        "    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)",
        "    assert verdict['reason'] == 'graded'",
        "",
    ]
    for fixture in fixtures:
        lines += [
            "",
            "def test_rejects_" + fixture["id"] + "():",
            "    verdict = _verdict_over(REJECTING / '" + fixture["id"] + "')",
            "    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)",
            "    assert verdict['reason'] == '" + fixture["expects"] + "', verdict['reason']",
            "    assert verdict['failed_checker'] == '" + fixture["failed_checker"] + "'",
            "",
        ]
    near = [fixture for fixture in fixtures if fixture["near_miss"]]
    if near:
        lines += [
            "",
            "def _submission(workspace):",
            "    return json.loads((workspace / 'submission.json').read_text(encoding='utf-8'))",
            "",
        ]
        for fixture in near:
            keys = fixture["moved_keys"]
            lines += [
                "",
                "def test_near_miss_" + fixture["id"] + "_moves_one_value_only():",
                "    reference = _submission(REFERENCE)",
                "    fixture = _submission(REJECTING / '" + fixture["id"] + "')",
                "    moved = sorted(k for k in set(reference) | set(fixture)"
                " if reference.get(k) != fixture.get(k))",
                "    assert moved == " + repr(sorted(keys)) + ", moved",
                "    reference_plan = (REFERENCE / 'plan.sql').read_bytes()",
                "    fixture_plan = (REJECTING / '" + fixture["id"] + "' / 'plan.sql').read_bytes()",
                "    assert reference_plan == fixture_plan",
                "",
            ]
    lines += [
        "",
        "def test_empty_workspace_scores_zero_with_a_reason(tmp_path):",
        "    harness = grade.build_harness(BUNDLE, tmp_path)",
        "    bound = grade.load_bound(BUNDLE)",
        "    verdict = grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))",
        "    assert verdict['reward'] == 0.0",
        "    assert verdict['reason'] == checkers.REASON_PLAN_SHAPE",
        "",
    ]
    return "\n".join(lines)


TEST_CHECKER_IDS = (
    ("warehouse_unmodified", "check_warehouse_unmodified"),
    ("plan_is_single_read_only_select", "check_plan_is_single_read_only_select"),
    ("quarantine_row_identified", "check_quarantine_row_identified"),
    ("closure_set_complete", "check_closure_set_complete"),
    ("closure_layers_witnessed", "check_closure_layers_witnessed"),
    ("closure_cardinality_attested", "check_closure_cardinality_attested"),
    ("excluded_rows_absent", "check_excluded_rows_absent"),
    ("result_set_equality", "check_result_set_equality"),
    ("plan_cost_readout_exact", "check_plan_cost_readout_exact"),
    ("plan_faster_than_baseline", "check_plan_faster_than_baseline"),
)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


class Emitter:
    def __init__(self, check):
        self.check = check
        self.drift = []

    def text(self, path, content):
        path = Path(path)
        if self.check:
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current != content:
                self.drift.append(path.as_posix())
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def blob(self, path, content):
        path = Path(path)
        if self.check:
            current = path.read_bytes() if path.is_file() else None
            if current != content:
                self.drift.append(path.as_posix())
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description="derive every generated artifact of OER-29")
    parser.add_argument("--check", action="store_true", help="report drift and write nothing")
    args = parser.parse_args()

    ground = load_grounding()
    emitter = Emitter(args.check)

    spec = spec_from(ground)
    spec_text = canonical_json(spec)

    # The warehouse always has to be materialised, because everything downstream is measured
    # against it. In check mode it is materialised beside the frozen copy, compared, and removed.
    scratch = BUNDLE / "environment" / (".recompute.db" if args.check else "warehouse.db")
    derived = derive(ground, scratch)
    database_bytes = Path(scratch).read_bytes()

    emitter.text(HERE / "warehouse_spec.json", spec_text)
    emitter.blob(BUNDLE / "environment" / "warehouse.db", database_bytes)
    emitter.text(BUNDLE / "tests" / "baseline_plan.sql", derived["baseline_plan"])

    bound = build_bound(ground, derived, spec_text, sha256_bytes(database_bytes))
    emitter.text(BUNDLE / "tests" / "bound.json", canonical_json(bound))

    reference_submission = submission_payload(derived, derived["reference_report"])
    reference_dir = BUNDLE / "solution" / "fixtures" / "reference_run"
    emitter.text(reference_dir / "plan.sql", derived["reference_plan"])
    emitter.text(reference_dir / "submission.json", canonical_json(reference_submission))

    fixtures = rejecting_fixtures(ground, derived, scratch)
    for fixture in fixtures:
        target = BUNDLE / "solution" / "fixtures" / "rejecting" / fixture["id"]
        emitter.text(target / "plan.sql", fixture["plan"])
        emitter.text(target / "submission.json", canonical_json(fixture["submission"]))
    emitter.text(
        BUNDLE / "solution" / "fixtures" / "rejecting" / "index.json",
        canonical_json(
            {
                "banner": BANNER,
                "source": SOURCE,
                "fixtures": [
                    {
                        "id": fixture["id"],
                        "expects": fixture["expects"],
                        "failed_checker": fixture["failed_checker"],
                        "near_miss": fixture["near_miss"],
                        "moved_keys": sorted(fixture["moved_keys"]),
                    }
                    for fixture in fixtures
                ],
            }
        ),
    )

    emitter.text(BUNDLE / "tests" / "test_output.py", render_test_output(fixtures) + "\n")
    emitter.text(HERE / "solve.sh", render_solve_sh())
    emitter.text(HERE / "TRUTH.md", render_truth(ground, derived, bound))
    emitter.text(HERE / "rubrics.json", canonical_json(render_rubrics(ground, derived, bound)))

    if args.check:
        Path(scratch).unlink(missing_ok=True)
        if emitter.drift:
            sys.stderr.write("drift:\n  " + "\n  ".join(sorted(emitter.drift)) + "\n")
            return 1
        print("no drift")
        return 0
    print(
        json.dumps(
            {
                "warehouse_sha256": bound["warehouse_sha256"],
                "baseline_plan_cost": bound["baseline_plan_cost"],
                "reference_plan_cost": bound["reference_plan_cost"],
                "instance_target_speedup": bound["instance_target_speedup"],
                "fixtures": len(fixtures),
            },
            sort_keys=True,
        )
    )
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
