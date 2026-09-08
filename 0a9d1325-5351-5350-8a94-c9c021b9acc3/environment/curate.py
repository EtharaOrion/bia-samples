#!/usr/bin/env python3
"""The filter runner, as a command-line tool. Same module the verifier uses.

    python3 curate.py --filter my_filter.json
    python3 curate.py --filter my_filter.json --breakdown

It applies a chain to the shipped source register and prints what the chain did:
how many predicates resolved, how many did NOT, how many blocks were admitted and
whether the pool passed through unchanged.

READ THE REPORT. The runner fails open. A predicate whose `field` is not a key of
a register row resolves nothing, matches everything and is a no-op, and the runner
still exits zero and reports success. `rules_unresolved` is how you see it, and
`pool_passed_through_unchanged` is how you see the end state it produces. The
shipped default_filter.json is exactly that case; run it and look.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import filter_schema  # noqa: E402

REGISTER = HERE / "pool" / "source_register.jsonl"
SPEC = HERE / "frozen" / "task_spec.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=str(HERE / "default_filter.json"))
    ap.add_argument("--register", default=str(REGISTER))
    ap.add_argument("--breakdown", action="store_true",
                    help="print per-feature quantiles over the admitted and rejected sets")
    ap.add_argument("--out", default="", help="write the admitted block ids here as JSON")
    args = ap.parse_args()

    try:
        doc = filter_schema.load(args.filter)
    except filter_schema.Refusal as exc:
        print(f"REFUSED  {exc.reason}: {exc.detail}")
        return 2

    register = filter_schema.load_register(args.register)
    admitted, report = filter_schema.apply(doc, register)
    budget = json.loads(Path(SPEC).read_text())["budget"]["blocks"]

    print(json.dumps(report, indent=2))
    print()
    if report["rules_unresolved"]:
        print(f"NOTE  {report['rules_unresolved']} predicate(s) resolved no field: "
              f"{', '.join(report['unresolved_rule_fields'])}")
        print("      Those predicates matched everything. They did not filter anything out.")
    if report["pool_passed_through_unchanged"]:
        print("NOTE  the pool passed through UNCHANGED. This chain removed nothing.")
    if len(admitted) < budget:
        print(f"UNDERFILL  {len(admitted)} blocks admitted, the budget consumes {budget}. "
              f"The verifier refuses a chain that leaves fewer blocks than the budget.")
    else:
        print(f"OK  {len(admitted)} admitted, the budget consumes the first {budget}.")

    if args.breakdown:
        keep = set(admitted)
        fields = [f for f in filter_schema.KNOWN_FIELDS
                  if f not in ("block_id", "tokens") and f in register[0]]
        rows_in = [r for r in register if r["block_id"] in keep]
        rows_out = [r for r in register if r["block_id"] not in keep]
        print(f"\n{'feature':24s}{'admitted p05':>14s}{'admitted p50':>14s}"
              f"{'admitted p95':>14s}{'rejected p50':>14s}")
        for f in fields:
            def q(rows, p):
                if not rows:
                    return float("nan")
                vals = sorted(float(r[f]) for r in rows)
                return vals[min(len(vals) - 1, int(p * len(vals)))]
            print(f"{f:24s}{q(rows_in,0.05):14.5f}{q(rows_in,0.50):14.5f}"
                  f"{q(rows_in,0.95):14.5f}{q(rows_out,0.50):14.5f}")

    if args.out:
        Path(args.out).write_text(json.dumps(admitted), encoding="utf-8")
        print(f"\nadmitted block ids written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
