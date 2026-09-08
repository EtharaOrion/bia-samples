#!/usr/bin/env python3
"""Design an evaluation policy against the real substrate.

This is the agent's measuring instrument. It runs the SAME harness the verifier
runs, over the same frozen training recipe, the same initialisation and the same
token stream, so the curve it shows you is the curve the graded run produces.
Three differences, all deliberate and all stated.

  * THE DRAW. The verifier picks the degradation ramp start on the grading run out
    of frozen/task_spec.json corpus.ramp_start_choices, and you do not learn which.
    `--ramp-start` here lets you run any of them. A policy that is only good on one
    of them is a policy that is one-in-five to be good on the graded run.

  * THE SPLITS. The stream a policy PAYS to read and the stream the graded figure is
    taken on both exist only inside the verifier. This script reads
    data/devset_slice.bin instead. Expect an offset; expect the shape to hold. What
    it cannot show you is the DRAW-level noise on the probe readings themselves --
    use --repeat to see how much your policy's choice moves when the readings do.

  * THE ANCHORS. This script reports a raw held-out-proxy loss. The verifier turns a
    loss into a reward against two anchors it replays itself on the same run.

Usage
    python3 probe_local.py                              # sweep, all ramp starts
    python3 probe_local.py --ramp-start 896             # one draw, full curve
    python3 probe_local.py --policy my_policy.json      # replay a policy on each draw
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness            # noqa: E402
import policy_schema      # noqa: E402

TRAIN = HERE / "data" / "train_stream.bin"
DEVSET = HERE / "data" / "devset_slice.bin"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ramp-start", type=int, default=None,
                    help="one declared ramp start; default sweeps all of them")
    ap.add_argument("--policy", action="append", default=[],
                    help="a policy JSON to replay against each swept draw")
    ap.add_argument("--repeat", type=int, default=1,
                    help="permutation seeds per ramp start, to see the draw-level spread")
    ap.add_argument("--out", default="local_curves.json")
    args = ap.parse_args(argv)

    policies = []
    for path in args.policy:
        try:
            policies.append((Path(path).name, policy_schema.load(path)))
        except policy_schema.Refusal as exc:
            print(f"REFUSED {path}: [{exc.reason}] {exc.detail}")
            return 2

    if not torch.cuda.is_available():
        print("no CUDA device; this task is sized for one H100")
        return 3

    harness.configure_backends()
    spec = harness.load_spec()
    recipe = harness.load_train_recipe()
    ev, corpus = spec["evaluation"], spec["corpus"]
    grid = harness.candidate_steps(spec)
    tokens_per_step = int(spec["budget"]["micro_batch"]) * int(spec["architecture"]["seq_len"])

    started = time.time()
    clean = harness.load_shard(TRAIN)
    devset = harness.to_device_tokens(harness.load_shard(DEVSET), "cuda")
    runner = harness.Runner("cuda", spec)
    print(f"harness ready in {time.time() - started:.1f}s "
          f"({sum(p.numel() for p in runner.model.parameters()) / 1e6:.1f}M parameters, "
          f"{spec['budget']['total_train_tokens']} training tokens per run)", flush=True)

    draws = [args.ramp_start] if args.ramp_start is not None else list(corpus["ramp_start_choices"])
    rng = random.Random(0)
    out = {}
    for ramp_start in draws:
        if ramp_start not in corpus["ramp_start_choices"]:
            print(f"ramp start {ramp_start} is not one of {corpus['ramp_start_choices']}")
            return 2
        for rep in range(args.repeat):
            seed = rng.randrange(2 ** 62)
            # The ramp saturates ramp_span_micro_steps after its start, exactly as
            # harness.draw_degradation derives it for the graded run.
            dirty, blocks = harness.degrade(
                clean, ramp_start, ramp_start + int(corpus["ramp_span_micro_steps"]),
                int(corpus["degradation_block_tokens"]), tokens_per_step, seed)
            train = harness.to_device_tokens(dirty, "cuda")
            t0 = time.time()
            swept = runner.sweep(recipe, train,
                                 {"devset": (devset, int(ev["devset_batches"]))},
                                 grid, progress=None)
            curves = swept["curves"]
            mean = {s: float(np.mean(v["devset"])) for s, v in curves.items()}
            best = min(mean, key=mean.get)
            print(f"\n=== ramp_start {ramp_start}  seed {seed}  "
                  f"{blocks} blocks permuted  {time.time() - t0:.0f}s ===")
            print(f"  devset argmin  step {best:4d}  {mean[best]:.4f}")
            print(f"  devset final   step {grid[-1]:4d}  {mean[grid[-1]]:.4f}"
                  f"   (the default policy keeps this one)")
            print(f"  {'step':>6} {'devset':>9}")
            for s in grid:
                mark = "  <- argmin" if s == best else ""
                print(f"  {s:6d} {mean[s]:9.4f}{mark}")
            for name, policy in policies:
                applied = harness.apply_policy(policy, curves, int(ev["tokens_per_batch"]),
                                               stream="devset")
                print(f"  policy {name}: selected step {applied['selected_step']}"
                      f"  devset {mean[applied['selected_step']]:.4f}"
                      f"  tokens {applied['tokens_spent']}")
            out[f"{ramp_start}:{seed}"] = {str(s): curves[s]["devset"] for s in curves}

    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"\nwritten: {args.out}  (per-batch devset losses, every step, every draw)")
    print("NOTE: these are devset readings. The verifier's probe stream and its graded")
    print("split are different data; expect an offset and expect the ranking to hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
