#!/usr/bin/env python3
"""The oracle. It solves the task in the built environment, the way an agent would have to.

Nothing here is told the answer. The excluded row and the closure are not passed in, not read
from grounding.yaml, not read from bound.json and not read from any fixture. Both come out of
querying the warehouse the image build produced, through the same harness executor the agent is
given and the verifier uses.

The four steps, in the order the oracle actually takes them:

  1. Cost the frozen baseline plan, so the denominator of the graded speedup is observed rather
     than assumed.
  2. Apply the frozen exclusion rule with one read-only query and take the single lineage row it
     returns. That is the first discovery value.
  3. Walk the closure relation, which is the parent edges unioned with the derives edges, breadth
     first from that row until the frontier empties. The layers are recorded as the walk
     produces them and the cardinality is the size of what the walk returned. That is the second
     discovery value, and it is a traversal result rather than a number.
  4. Fold the closure into a plan that scans `reading` once, cost that plan, and write plan.sql
     and submission.json.

Step 3 is where the task is won or lost. A walk that takes only the immediate successors of the
excluded row terminates, produces a plausible list and a plan that runs, and is wrong. The walk
below runs to fixpoint, and it reaches through both edge relations rather than through the
parent edges alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "environment"))

import planrun  # noqa: E402
import probe  # noqa: E402

SUBMISSION_SCHEMA = "oer29.submission/v1"

EXCLUSION_QUERY = """SELECT node_id
  FROM lineage
 WHERE declared_check <> (p_alpha * 31 + p_beta * 17 + p_gamma * 7) % 1000003
 ORDER BY node_id
"""

PARENT_EDGE_QUERY = "SELECT parent_id, node_id FROM lineage WHERE parent_id IS NOT NULL"
DERIVES_EDGE_QUERY = "SELECT source_node, target_node FROM derives"


def query(warehouse, sql):
    return planrun.run_plan(str(warehouse), sql)


def discover_excluded_row(warehouse, trace):
    report = query(warehouse, EXCLUSION_QUERY)
    rows = [str(row[0]) for row in report["rows"]]
    trace.append(
        {
            "step": "apply-exclusion-rule",
            "cost_steps": report["cost_steps"],
            "rows_returned": len(rows),
        }
    )
    if len(rows) != 1:
        raise SystemExit(
            "the exclusion rule selected " + str(len(rows)) + " lineage rows and the task states "
            "that exactly one is quarantined; refusing to guess which"
        )
    return rows[0]


def load_edges(warehouse, trace):
    edges = {}
    total = 0
    for name, sql in (("parent", PARENT_EDGE_QUERY), ("derives", DERIVES_EDGE_QUERY)):
        report = query(warehouse, sql)
        for source, target in report["rows"]:
            edges.setdefault(str(source), set()).add(str(target))
        total += report["row_count"]
        trace.append(
            {
                "step": "load-" + name + "-edges",
                "cost_steps": report["cost_steps"],
                "rows_returned": report["row_count"],
            }
        )
    resolved = {key: sorted(value) for key, value in edges.items()}
    trace.append({"step": "edge-relation-built", "edges": total, "sources": len(resolved)})
    return resolved


def walk_closure(edges, origin, trace):
    """Breadth first to fixpoint. The layers are the witness the traversal actually happened."""
    seen = {origin}
    layers = [[origin]]
    frontier = [origin]
    while frontier:
        nxt = []
        for node in frontier:
            for child in edges.get(node, ()):
                if child not in seen:
                    seen.add(child)
                    nxt.append(child)
        if not nxt:
            break
        nxt.sort()
        layers.append(nxt)
        frontier = nxt
    trace.append(
        {
            "step": "closure-walked-to-fixpoint",
            "depth": len(layers) - 1,
            "layer_profile": [len(layer) for layer in layers],
            "cardinality": len(seen),
        }
    )
    return layers, sorted(seen)


def fold_plan(closure):
    literal = ", ".join("'" + node + "'" for node in closure)
    return (
        "SELECT reading_id, node_id, value\n"
        "  FROM reading\n"
        " WHERE node_id NOT IN (" + literal + ")\n"
        " ORDER BY reading_id\n"
    )


def solve(warehouse, workspace):
    trace = []

    baseline = query(warehouse, probe.BASELINE_PLAN)
    trace.append(
        {
            "step": "cost-frozen-baseline-plan",
            "cost_steps": baseline["cost_steps"],
            "rows_returned": baseline["row_count"],
        }
    )

    excluded = discover_excluded_row(warehouse, trace)
    edges = load_edges(warehouse, trace)
    layers, closure = walk_closure(edges, excluded, trace)

    plan_text = fold_plan(closure)
    plan_report = query(warehouse, plan_text)
    trace.append(
        {
            "step": "cost-folded-plan",
            "cost_steps": plan_report["cost_steps"],
            "rows_returned": plan_report["row_count"],
        }
    )

    if [list(row) for row in plan_report["rows"]] != [list(row) for row in baseline["rows"]]:
        raise SystemExit(
            "the folded plan does not return the same result set as the frozen baseline plan, so "
            "the closure it was folded from is wrong; refusing to submit it"
        )
    trace.append(
        {
            "step": "result-set-equality-self-check",
            "equal_to_baseline": True,
            "speedup": round(baseline["cost_steps"] / plan_report["cost_steps"], 6),
        }
    )

    submission = {
        "schema": SUBMISSION_SCHEMA,
        "excluded_node_id": excluded,
        "closure_nodes": closure,
        "closure_layers": layers,
        "closure_cardinality": len(closure),
        "admitted_row_count": plan_report["row_count"],
        "reported_plan_cost": plan_report["cost_steps"],
        "reported_baseline_cost": baseline["cost_steps"],
    }

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "plan.sql").write_text(plan_text, encoding="utf-8")
    (workspace / "submission.json").write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return submission, trace


def main():
    parser = argparse.ArgumentParser(description="solve OER-29 against the built warehouse")
    parser.add_argument("--warehouse", default=str(BUNDLE / "environment" / "warehouse.db"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--trace", help="where the oracle's own step trace is written")
    args = parser.parse_args()

    submission, trace = solve(args.warehouse, args.workspace)
    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trace).write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in trace) + "\n", encoding="utf-8"
        )
    for row in trace:
        print(json.dumps(row, sort_keys=True))
    print(
        json.dumps(
            {
                "closure_cardinality": submission["closure_cardinality"],
                "admitted_row_count": submission["admitted_row_count"],
                "reported_plan_cost": submission["reported_plan_cost"],
                "reported_baseline_cost": submission["reported_baseline_cost"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
