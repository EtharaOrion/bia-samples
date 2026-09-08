#!/usr/bin/env python3
"""Measure a data plan end to end, on the real substrate.

This is the agent's measuring instrument. It is the SAME harness the verifier runs,
so a number produced here is produced by the same code path that produces the graded
number. Two differences, both deliberate and both stated:

  * the split. This script evaluates on data/devset_slice.bin, a proxy cut from a
    DIFFERENT FineWeb training shard. The graded split is a slice of
    fineweb_val_000000 that exists only inside the verifier image. Expect the two
    numbers to differ by a small offset; expect them to RANK plans the same way.

  * the anchors. This script reports a raw loss. The verifier converts a loss into a
    reward against two anchors it measures itself on the same run.

Usage
    python3 probe_local.py                        # the shipped default plan
    python3 probe_local.py mine.json              # your candidate
    python3 probe_local.py a.json b.json c.json   # several, one warm process
    python3 probe_local.py --solo                 # one plan per source, whole budget

Running several plans in one invocation is much cheaper than several invocations: the
model is built once, torch.compile is paid once, and the pool is read once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness            # noqa: E402
import plan_schema        # noqa: E402

POOL = HERE / "data" / "sources"
DEVSET = HERE / "data" / "devset_slice.bin"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="*", help="plan JSON files; default is default_plan.json")
    ap.add_argument("--solo", action="store_true",
                    help="also measure one plan per source, whole budget from that source")
    ap.add_argument("--out", default="local_results.json")
    args = ap.parse_args(argv)

    spec = harness.load_spec()
    recipe = harness.load_train_recipe()
    budget = int(spec["budget"]["total_train_tokens"])

    named = []
    for path in (args.plans or [str(HERE / "default_plan.json")]):
        try:
            named.append((Path(path).name, plan_schema.load(path, spec)))
        except plan_schema.Refusal as exc:
            print(f"REFUSED {path}: [{exc.reason}] {exc.detail}")
            return 2
    if args.solo:
        for name in sorted(spec["pool"]["sources"]):
            named.append(("solo:" + name,
                          {"schema": plan_schema.SCHEMA_ID,
                           "draws": [{"source": name, "tokens": budget, "offset": 0}]}))

    if not torch.cuda.is_available():
        print("no CUDA device; this task is sized for one H100")
        return 3

    harness.configure_backends()
    started = time.time()
    pool = harness.load_pool(spec, POOL)
    devset = harness.to_device_tokens(harness.load_shard(DEVSET), "cuda")
    runner = harness.Runner("cuda", spec)
    print(f"harness ready in {time.time() - started:.1f}s "
          f"({sum(p.numel() for p in runner.model.parameters()) / 1e6:.1f}M parameters, "
          f"{budget} training tokens per run, {len(pool)} sources)", flush=True)

    results = []
    for label, plan in named:
        totals = {}
        for d in plan["draws"]:
            totals[d["source"]] = totals.get(d["source"], 0) + int(d["tokens"])
        print(f"\n=== {label}  {len(plan['draws'])} draws, order "
              f"{[d['source'] for d in plan['draws']]} ===", flush=True)
        started = time.time()
        report = runner.run(plan, recipe, spec, pool, devset, progress=256)
        report["label"] = label
        report["tokens_per_source"] = dict(sorted(totals.items()))
        report["wall_seconds"] = round(time.time() - started, 1)
        results.append(report)
        if report["diverged"]:
            print(f"  DIVERGED -- no finite loss. {report['wall_seconds']}s")
        else:
            print(f"  devset_loss {report['val_loss']:.5f}   {report['wall_seconds']}s", flush=True)

    print("\n---- summary (devset, NOT the graded split) ----")
    for report in sorted(results, key=lambda r: r["val_loss"]):
        loss = "diverged" if report["diverged"] else f"{report['val_loss']:.5f}"
        print(f"  {report['label']:<34} {loss}")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
