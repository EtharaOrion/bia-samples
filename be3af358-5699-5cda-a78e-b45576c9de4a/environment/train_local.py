#!/usr/bin/env python3
"""Your yardstick. Runs the frozen training run once and scores selections on the devset.

    python3 /app/train_local.py                       # score the shipped default
    python3 /app/train_local.py --selection my.json   # score your own selection too
    python3 /app/train_local.py --sweep               # score a spread of shapes

This is the SAME harness the verifier runs, byte for byte -- tests/Dockerfile refuses to
build unless environment/harness.py, environment/selection_schema.py,
environment/model/nanogpt.py and environment/frozen/task_spec.json are identical to the
verifier's copies. What differs is the evaluation split, and that difference is the task:

    here      environment/data/devset_slice.bin, cut from a FineWeb TRAINING shard
    grading   a slice of the FineWeb VALIDATION shard, staged only in the verifier

A selection tuned until it is the single best-scoring one on the devset is a selection
tuned partly to the devset's noise. The ordering of near-neighbours moves between the two
splits; the shape of the answer does not. Prefer a selection whose margin is wide.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness             # noqa: E402
import selection_schema    # noqa: E402

TRAIN = HERE / "data" / "train_slice.bin"
DEVSET = HERE / "data" / "devset_slice.bin"


def uniform(ids):
    return {"schema": selection_schema.SCHEMA_ID, "keep": list(ids),
            "weights": [1.0 / len(ids)] * len(ids), "notes": "uniform"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", action="append", default=[],
                    help="path to a selection.json to score; repeatable")
    ap.add_argument("--sweep", action="store_true",
                    help="also score a spread of retention shapes")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("this needs the accelerator", file=sys.stderr)
        return 2
    harness.configure_backends()
    spec = harness.load_spec()
    runner = harness.Runner("cuda", spec)
    train = harness.to_device_tokens(harness.load_shard(TRAIN), "cuda")
    dev = harness.to_device_tokens(harness.load_shard(DEVSET), "cuda")

    started = time.time()
    print(f"[train] the frozen run: {runner.micro_steps} micro-batches, "
          f"{runner.snapshot_count} snapshots", flush=True)
    run = runner.train_snapshots(train, progress=True)
    snapshots = run["snapshots"]
    print(f"[train] {run['train_seconds']}s\n", flush=True)

    # Every individual snapshot, so the shape of the run is visible rather than guessed.
    print("[single snapshots on the devset]")
    for ident in sorted(snapshots):
        one = {"schema": selection_schema.SCHEMA_ID, "keep": [ident], "weights": [1.0]}
        report = runner.evaluate_selection(one, snapshots, dev)
        print(f"    snapshot {ident:2d}   dev_loss {report['val_loss']:.6f}", flush=True)

    candidates = []
    budget = int(spec["snapshots"]["storage_budget"])
    ids = [int(i) for i in spec["snapshots"]["ids"]]
    for path in args.selection:
        candidates.append((f"file:{path}", selection_schema.load(path, spec)))
    candidates.append(("shipped-default",
                       selection_schema.load(HERE / "default_selection.json", spec)))
    if args.sweep:
        last = ids[-budget:]
        candidates.append((f"uniform-last-{budget}", uniform(last)))
        candidates.append(("uniform-spread", uniform(ids[len(ids) - 1::-(len(ids) // budget)][:budget])))
        candidates.append(("uniform-halves", uniform(ids[len(ids) // 2:][:budget])))

    print("\n[selections on the devset]")
    for label, selection in candidates:
        report = runner.evaluate_selection(selection, snapshots, dev)
        print(f"    {label:24s} keep={report['kept']}  dev_loss {report['val_loss']:.6f}",
              flush=True)
    print(f"\n[total] {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
