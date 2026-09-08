#!/usr/bin/env python3
"""Train one recipe under the frozen budget and report validation loss.

This is the agent's measuring instrument. It is the SAME harness the verifier runs,
so a number produced here is produced by the same code path that produces the graded
number. Two differences, both deliberate and both stated:

  * the split. This script evaluates on environment/data/devset_slice.bin, a proxy cut
    from a DIFFERENT FineWeb training shard. The graded split is a slice of
    fineweb_val_000000 that exists only inside the verifier image. Expect the two
    numbers to differ by a small offset; expect them to RANK recipes the same way.

  * the anchors. This script reports a raw loss. The verifier converts a loss into a
    reward against two anchors it measures itself on the same run.

Usage
    python3 train_local.py                          # the shipped default
    python3 train_local.py my_recipe.json           # your candidate
    python3 train_local.py a.json b.json c.json     # several, one warm process

Running several recipes in one invocation is much cheaper than several invocations:
the model is built once, torch.compile is paid once, and the corpus is read once.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness            # noqa: E402
import recipe_schema      # noqa: E402

TRAIN = HERE / "data" / "train_slice.bin"
DEVSET = HERE / "data" / "devset_slice.bin"


def main(argv):
    paths = [Path(a) for a in argv[1:]] or [HERE / "default_recipe.json"]
    recipes = []
    for path in paths:
        try:
            recipes.append((path, recipe_schema.load(path)))
        except recipe_schema.Refusal as exc:
            print(f"REFUSED {path}: [{exc.reason}] {exc.detail}")
            return 2

    if not torch.cuda.is_available():
        print("no CUDA device; this task is sized for one H100")
        return 3

    harness.configure_backends()
    spec = harness.load_spec()
    started = time.time()
    train_tokens = harness.to_device_tokens(harness.load_shard(TRAIN), "cuda")
    dev_tokens = harness.to_device_tokens(harness.load_shard(DEVSET), "cuda")
    runner = harness.Runner("cuda", spec)
    print(f"harness ready in {time.time() - started:.1f}s "
          f"({sum(p.numel() for p in runner.model.parameters()) / 1e6:.1f}M parameters, "
          f"{spec['budget']['total_train_tokens']} training tokens per run)", flush=True)

    results = []
    for path, recipe in recipes:
        print(f"\n=== {path.name}  grad_accum={recipe['grad_accum']} "
              f"-> {runner.steps_for(recipe)} optimizer steps ===", flush=True)
        started = time.time()
        report = runner.run(recipe, train_tokens, dev_tokens, progress=256)
        report["recipe"] = str(path)
        report["wall_seconds"] = round(time.time() - started, 1)
        results.append(report)
        if report["diverged"]:
            print(f"  DIVERGED -- no finite loss. {report['wall_seconds']}s")
        else:
            print(f"  devset_loss {report['val_loss']:.5f}   {report['wall_seconds']}s", flush=True)

    print("\n---- summary (devset, NOT the graded split) ----")
    for report in results:
        loss = "diverged" if report["diverged"] else f"{report['val_loss']:.5f}"
        print(f"  {Path(report['recipe']).name:<32} {loss}")
    Path("local_results.json").write_text(json.dumps(results, indent=2))
    print("\nwritten: local_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
