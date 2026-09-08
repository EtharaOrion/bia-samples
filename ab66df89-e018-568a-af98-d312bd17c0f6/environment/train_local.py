#!/usr/bin/env python3
"""Train one compute allocation and report validation loss.

This is the agent's measuring instrument. It is the SAME harness the verifier runs,
so a number produced here is produced by the same code path that produces the graded
number. Two differences, both deliberate and both stated:

  * the split. This script evaluates on data/devset_slice.bin, a proxy cut from a
    DIFFERENT FineWeb training shard. The graded split is a slice of
    fineweb_val_000000 that exists only inside the verifier image. Expect the two
    numbers to differ by a small offset; expect them to RANK allocations the same way.

  * the anchors. This script reports a raw loss. The verifier converts a loss into a
    reward against two anchors it measures itself on the same run.

Usage
    python3 train_local.py                        # the shipped default allocation
    python3 train_local.py mine.json              # your candidate
    python3 train_local.py a.json b.json          # several, one warm process
    python3 train_local.py --menu                 # what each menu entry costs and buys

Allocations that name the same menu entry share one built model and one compiled
graph, so grouping them into a single invocation is much cheaper than several.
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

import harness              # noqa: E402
import allocation_schema    # noqa: E402

TRAIN = HERE / "data" / "train_slice.bin"
DEVSET = HERE / "data" / "devset_slice.bin"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("allocations", nargs="*")
    ap.add_argument("--menu", action="store_true",
                    help="print what each menu entry costs and how much it can buy")
    ap.add_argument("--out", default="local_results.json")
    args = ap.parse_args(argv)

    spec = harness.load_spec()
    recipe = harness.load_train_recipe()
    budget = int(spec["budget"]["flop_budget"])

    if args.menu:
        corpus = int(spec["data"]["corpus_tokens"])
        tps = int(spec["budget"]["micro_batch"]) * int(spec["architecture_common"]["seq_len"])
        print(f"FLOP budget {budget}   corpus {corpus} tokens\n")
        print(f"{'entry':>10} {'layers':>7} {'dim':>5} {'flops/micro':>13} "
              f"{'max steps':>10} {'tokens':>12} {'epochs':>7}")
        for model in spec["model_menu"]:
            shape = harness.model_shape(spec, model)
            per = harness.micro_batch_flops(spec, model)
            steps = harness.max_micro_steps(spec, model)
            print(f"{model:>10} {shape['num_layers']:>7} {shape['model_dim']:>5} "
                  f"{per:13.3e} {steps:>10} {steps * tps:>12} {steps * tps / corpus:>7.2f}")
        if not args.allocations:
            return 0

    named = []
    for path in (args.allocations or [str(HERE / "default_allocation.json")]):
        try:
            named.append((Path(path).name, allocation_schema.load(path, spec, harness)))
        except allocation_schema.Refusal as exc:
            print(f"REFUSED {path}: [{exc.reason}] {exc.detail}")
            return 2

    if not torch.cuda.is_available():
        print("no CUDA device; this task is sized for one H100")
        return 3

    harness.configure_backends()
    started = time.time()
    train_tokens = harness.to_device_tokens(harness.load_shard(TRAIN), "cuda")
    dev_tokens = harness.to_device_tokens(harness.load_shard(DEVSET), "cuda")
    print(f"corpus loaded in {time.time() - started:.1f}s", flush=True)

    runners, results = {}, []
    for label, alloc in named:
        model = alloc["model"]
        if model not in runners:
            t0 = time.time()
            runners[model] = harness.Runner(spec, model, "cuda")
            print(f"[build] {model} in {time.time() - t0:.1f}s "
                  f"({sum(p.numel() for p in runners[model].model.parameters()) / 1e6:.1f}M "
                  f"parameters)", flush=True)
        spent = alloc["micro_steps"] * harness.micro_batch_flops(spec, model)
        print(f"\n=== {label}  {model}  micro_steps={alloc['micro_steps']} "
              f"grad_accum={alloc['grad_accum']}  "
              f"{100.0 * spent / budget:.1f}% of the FLOP budget ===", flush=True)
        started = time.time()
        report = runners[model].run(alloc, recipe, train_tokens, dev_tokens, progress=512)
        report["label"] = label
        report["spent_flops"] = spent
        report["wall_seconds"] = round(time.time() - started, 1)
        results.append(report)
        if report["diverged"]:
            print(f"  DIVERGED -- no finite loss. {report['wall_seconds']}s")
        else:
            print(f"  devset_loss {report['val_loss']:.5f}   "
                  f"{report['epochs']} epochs   {report['wall_seconds']}s", flush=True)

    print("\n---- summary (devset, NOT the graded split) ----")
    for report in sorted(results, key=lambda r: r["val_loss"]):
        loss = "diverged" if report["diverged"] else f"{report['val_loss']:.5f}"
        print(f"  {report['label']:<28} {report['model']:>9}  {loss}")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
