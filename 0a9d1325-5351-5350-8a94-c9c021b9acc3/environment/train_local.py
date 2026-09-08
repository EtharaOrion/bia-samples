#!/usr/bin/env python3
"""The agent's own measuring instrument.

    python3 train_local.py --filter my_filter.json

Trains the frozen substrate on the blocks a chain admits and evaluates on
data/devset_slice.bin, which is a PROXY cut from a different FineWeb training
shard. It is NOT the graded split -- the graded split is staged only into the
verifier image -- but it is measured by this same harness.py, from the same frozen
initialisation, on the same block pool, so a chain that lowers the number here is
a chain that lowers the graded number too.

Use --compare to train the shipped default chain in the same process and print
both, which is the comparison the verifier will make.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import filter_schema  # noqa: E402
import harness        # noqa: E402

REGISTER = HERE / "pool" / "source_register.jsonl"
POOL = HERE / "pool" / "pool_blocks.bin"
DEVSET = HERE / "data" / "devset_slice.bin"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=str(HERE / "default_filter.json"))
    ap.add_argument("--compare", action="store_true",
                    help="also train the shipped default chain, in the same process")
    ap.add_argument("--devset", default=str(DEVSET))
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device is visible; this harness needs the accelerator")
        return 2

    try:
        agent_doc = filter_schema.load(args.filter)
    except filter_schema.Refusal as exc:
        print(f"REFUSED  {exc.reason}: {exc.detail}")
        return 2

    register = filter_schema.load_register(REGISTER)
    spec = harness.load_spec()
    need = int(spec["budget"]["blocks"])

    plan = [("agent", agent_doc)]
    if args.compare:
        plan.insert(0, ("default", filter_schema.load(HERE / "default_filter.json")))

    selections = {}
    for label, doc in plan:
        admitted, report = filter_schema.apply(doc, register)
        print(f"[select:{label}] {json.dumps(report)}")
        if len(admitted) < need:
            print(f"[select:{label}] UNDERFILL: {len(admitted)} < {need}. "
                  f"The verifier refuses this chain.")
            if label == "agent":
                return 3
        selections[label] = admitted

    harness.configure_backends()
    started = time.time()
    pool = harness.BlockPool(POOL, spec, "cuda")
    devset = harness.to_device_tokens(harness.load_shard(args.devset), "cuda")
    runner = harness.Runner(pool, "cuda", spec)
    print(f"[setup] {sum(p.numel() for p in runner.model.parameters())/1e6:.1f}M parameters, "
          f"{time.time()-started:.1f}s", flush=True)

    out = {}
    for label, _doc in plan:
        print(f"\n[train:{label}]", flush=True)
        report = runner.run(selections[label][:need], devset, progress=768)
        out[label] = report
        print(f"[train:{label}] devset_loss={report['val_loss']:.6f} "
              f"({report['train_seconds']}s train)", flush=True)

    print("\n==== devset (PROXY, not the graded split) ====")
    for label, report in out.items():
        print(f"  {label:9s} {report['val_loss']:.6f}")
    if "default" in out and "agent" in out:
        print(f"\n  improvement over the shipped default: "
              f"{out['default']['val_loss'] - out['agent']['val_loss']:+.6f} nats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
