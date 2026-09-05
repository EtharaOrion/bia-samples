#!/usr/bin/env python3
"""The harness handle over the built warehouse. Everything you may read, you read through here.

The warehouse is state the image build established. This module is how you read it back: it
opens the frozen database read-only, runs whatever read-only plan you hand it through the same
executor the verifier uses, and reports the harness-counted cost of that plan.

Three subcommands, and nothing hidden behind them:

    python3 /task/environment/probe.py summary
        the schema, the row counts and the frozen check rule, exactly as the warehouse carries
        them.

    python3 /task/environment/probe.py baseline
        run the frozen baseline plan and report its harness-counted cost. This is the
        denominator of the graded speedup, so it is handed to you rather than guessed at.

    python3 /task/environment/probe.py run --plan myplan.sql [--preview 20]
        run one read-only plan of yours and report its harness-counted cost, its row count and
        a bounded preview of its rows.

The handle answers questions about the warehouse. It does not answer the question the task
asks, and there is no subcommand that hands you a quarantine verdict or a closure. Everything
you need is reachable from `run`, because `run` will execute any read-only query you can write
against the frozen tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import planrun  # noqa: E402

WAREHOUSE = HERE / "warehouse.db"

BASELINE_PLAN = """WITH RECURSIVE quarantined(node_id) AS (
    SELECT node_id
      FROM lineage
     WHERE declared_check <> (p_alpha * 31 + p_beta * 17 + p_gamma * 7) % 1000003
    UNION
    SELECT l.node_id FROM lineage l JOIN quarantined q ON l.parent_id = q.node_id
    UNION
    SELECT d.target_node FROM derives d JOIN quarantined q ON d.source_node = q.node_id
)
SELECT reading_id, node_id, value
  FROM reading
 WHERE node_id NOT IN (SELECT node_id FROM quarantined)
 ORDER BY reading_id
"""

CHECK_RULE = (
    "a lineage row is intact when declared_check equals "
    "(p_alpha * 31 + p_beta * 17 + p_gamma * 7) % 1000003, and quarantined when it does not"
)


def _counts(warehouse):
    plans = {
        "lineage_rows": "SELECT COUNT(*) FROM lineage",
        "derives_rows": "SELECT COUNT(*) FROM derives",
        "reading_rows": "SELECT COUNT(*) FROM reading",
    }
    out = {}
    for key, sql in plans.items():
        out[key] = planrun.run_plan(warehouse, sql)["rows"][0][0]
    return out


def cmd_summary(args):
    warehouse = Path(args.warehouse)
    payload = {
        "warehouse": warehouse.as_posix(),
        "tables": {
            "lineage": ["node_id", "parent_id", "p_alpha", "p_beta", "p_gamma", "declared_check"],
            "derives": ["edge_id", "source_node", "target_node"],
            "reading": ["reading_id", "node_id", "value"],
        },
        "counts": _counts(warehouse),
        "integrity_rule": CHECK_RULE,
        "required_projection": ["reading_id", "node_id", "value"],
        "required_order": "reading_id ascending",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_baseline(args):
    result = planrun.run_plan(args.warehouse, BASELINE_PLAN)
    print(
        json.dumps(
            {
                "plan": "baseline",
                "cost_steps": result["cost_steps"],
                "row_count": result["row_count"],
                "columns": result["columns"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_run(args):
    text = Path(args.plan).read_text(encoding="utf-8")
    try:
        result = planrun.run_plan(args.warehouse, text)
    except planrun.PlanRefused as refusal:
        print(json.dumps({"ok": False, "reason": refusal.reason, "detail": refusal.detail}, indent=2))
        return 1
    preview = result["rows"][: max(0, int(args.preview))]
    print(
        json.dumps(
            {
                "ok": True,
                "cost_steps": result["cost_steps"],
                "row_count": result["row_count"],
                "columns": result["columns"],
                "preview": preview,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description="read the built warehouse back")
    parser.add_argument("--warehouse", default=str(WAREHOUSE))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary").set_defaults(handler=cmd_summary)
    sub.add_parser("baseline").set_defaults(handler=cmd_baseline)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--preview", default=20)
    run.set_defaults(handler=cmd_run)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
