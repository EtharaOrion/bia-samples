#!/usr/bin/env python3
"""Execute exactly one read-only query plan against the frozen warehouse and cost it.

This is the harness's executor. It is the only thing that ever runs a plan, on the agent's
side and on the verifier's side alike, so the cost the agent sees while it is refining and the
cost the verifier grades come out of the same code path over the same frozen bytes.

Cost is the number of virtual-machine steps the database engine takes to produce the whole
result. It is read off the engine's own progress callback, not off a clock. That matters twice
over: a step count is exactly reproducible for a given engine build and a given set of frozen
bytes, which a wall-clock reading is not, and no checker anywhere in this bundle has to touch a
clock in order to grade a speedup.

The engine build is pinned by the image digest in environment/Dockerfile and tests/Dockerfile,
which is what makes the step count portable between the two surfaces.

Safety is structural rather than advisory. The connection is opened read-only through a URI,
an authorizer refuses every action outside the read allowlist, and the caller isolates this
module in a subprocess of its own.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ALLOWED_FUNCTIONS = frozenset(
    {
        "abs",
        "coalesce",
        "count",
        "ifnull",
        "instr",
        "length",
        "lower",
        "max",
        "min",
        "nullif",
        "substr",
        "sum",
        "trim",
        "upper",
    }
)

DENIED_ACTION_NAME = {
    sqlite3.SQLITE_ATTACH: "attach",
    sqlite3.SQLITE_DETACH: "detach",
    sqlite3.SQLITE_PRAGMA: "pragma",
    sqlite3.SQLITE_INSERT: "insert",
    sqlite3.SQLITE_UPDATE: "update",
    sqlite3.SQLITE_DELETE: "delete",
    sqlite3.SQLITE_DROP_TABLE: "drop-table",
    sqlite3.SQLITE_CREATE_TABLE: "create-table",
    sqlite3.SQLITE_CREATE_INDEX: "create-index",
    sqlite3.SQLITE_ALTER_TABLE: "alter-table",
    sqlite3.SQLITE_TRANSACTION: "transaction",
}

ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_RECURSIVE,
    }
)


class PlanRefused(Exception):
    """The plan was refused before it ran. Always a named reason, never a silent zero."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def statement_count(text):
    """How many statements the plan text contains, counted by the engine's own splitter."""
    pending = ""
    total = 0
    for line in str(text).splitlines(True):
        pending += line
        if sqlite3.complete_statement(pending):
            if pending.strip().rstrip(";").strip():
                total += 1
            pending = ""
    if pending.strip().rstrip(";").strip():
        total += 1
    return total


def leading_keyword(text):
    """The first significant keyword, with comments and blank lines stripped off the front."""
    body = str(text)
    out = []
    index = 0
    length = len(body)
    while index < length:
        char = body[index]
        if char == "-" and body[index : index + 2] == "--":
            end = body.find("\n", index)
            index = length if end < 0 else end + 1
            continue
        if char == "/" and body[index : index + 2] == "/*":
            end = body.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if char.isspace():
            index += 1
            continue
        while index < length and not body[index].isspace() and body[index] not in "(;":
            out.append(body[index])
            index += 1
        break
    return "".join(out).upper()


def _authorizer(action, arg1, arg2, database, trigger):
    if action in ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        name = (arg2 or "").lower()
        return sqlite3.SQLITE_OK if name in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def run_plan(database_path, plan_text, row_limit=200000):
    """Run one plan. Return the rows, the column names and the harness-counted step cost."""
    text = str(plan_text)
    if not text.strip():
        raise PlanRefused("plan-empty", "the plan file is empty")
    total = statement_count(text)
    if total != 1:
        raise PlanRefused(
            "plan-not-single-statement",
            "the plan carries " + str(total) + " statements and exactly one is admitted",
        )
    keyword = leading_keyword(text)
    if keyword not in ("SELECT", "WITH", "VALUES"):
        raise PlanRefused(
            "plan-not-a-select",
            "the plan opens with " + repr(keyword) + " and only SELECT, WITH or VALUES is admitted",
        )

    uri = "file:" + str(Path(database_path).resolve()) + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    steps = [0]

    def tick():
        steps[0] += 1
        return 0

    try:
        connection.set_authorizer(_authorizer)
        connection.set_progress_handler(tick, 1)
        try:
            cursor = connection.execute(text)
            rows = cursor.fetchmany(row_limit + 1)
        except sqlite3.DatabaseError as failure:
            raise PlanRefused("plan-rejected-by-engine", str(failure))
        connection.set_progress_handler(None, 0)
        if len(rows) > row_limit:
            raise PlanRefused(
                "plan-result-oversized",
                "the plan returned more than " + str(row_limit) + " rows",
            )
        columns = [entry[0] for entry in (cursor.description or ())]
    finally:
        connection.set_progress_handler(None, 0)
        connection.set_authorizer(None)
        connection.close()

    return {
        "rows": [list(row) for row in rows],
        "columns": columns,
        "cost_steps": steps[0],
        "row_count": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="run one read-only plan and cost it")
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", action="store_true", help="include the result rows in --out")
    args = parser.parse_args()

    plan_text = Path(args.plan).read_text(encoding="utf-8")
    try:
        result = run_plan(args.warehouse, plan_text)
    except PlanRefused as refusal:
        payload = {"ok": False, "reason": refusal.reason, "detail": refusal.detail}
    else:
        payload = {
            "ok": True,
            "reason": "",
            "detail": "",
            "cost_steps": result["cost_steps"],
            "row_count": result["row_count"],
            "columns": result["columns"],
            "rows": result["rows"] if args.rows else [],
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    summary = {key: payload[key] for key in ("ok", "reason") if key in payload}
    if payload.get("ok"):
        summary["cost_steps"] = payload["cost_steps"]
        summary["row_count"] = payload["row_count"]
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
