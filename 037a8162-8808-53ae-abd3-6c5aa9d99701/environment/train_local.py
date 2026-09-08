#!/usr/bin/env python3
"""The agent's own measuring instrument.

    python3 train_local.py --recipe my_recipe.json --compare

Trains a recipe on the frozen substrate, evaluating on the SAME grid the verifier
uses, and reports where the curve first crosses a target. With --compare it first
trains the shipped default, takes ITS end-of-ceiling loss as the target, and then
reports both crossings -- which is exactly the comparison the verifier makes.

Evaluation here is on data/devset_slice.bin, a PROXY cut from a different FineWeb
training shard. It is not the graded split, but it is measured by this same
harness.py, from the same frozen initialisation, on the same token stream.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness, recipe_schema, records  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--compare", action="store_true",
                    help="also train the shipped default and use its final loss as the target")
    ap.add_argument("--target", type=float, default=None,
                    help="use an explicit target instead of training the default")
    ap.add_argument("--devset", default=str(HERE / "data" / "devset_slice.bin"))
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device is visible; this harness needs the accelerator")
        return 2

    try:
        agent = recipe_schema.load(args.recipe)
    except recipe_schema.Refusal as exc:
        print(f"REFUSED  {exc.reason}: {exc.detail}")
        return 2

    corpus = records.load_records(HERE / "published_records.json")
    replay = records.find_replay(agent, corpus)
    if replay is not None:
        print(f"REFUSED  record-replayed: this recipe restates published record "
              f"{replay.get('record_id')!r}. The verifier will refuse it at 0.0.")
        return 3
    print(f"[records] {len(corpus)} published records; this recipe is not one of them "
          f"(digest {records.digest(agent)})")

    harness.configure_backends()
    spec = harness.load_spec()
    train = harness.to_device_tokens(harness.load_shard(HERE / "data" / "train_slice.bin"), "cuda")
    dev = harness.to_device_tokens(harness.load_shard(args.devset), "cuda")
    runner = harness.Runner("cuda", spec)

    plan = [("agent", agent)]
    if args.compare:
        plan.insert(0, ("default", recipe_schema.load(HERE / "default_recipe.json")))

    out = {}
    for label, rec in plan:
        print(f"\n[train:{label}]", flush=True)
        t0 = time.time()
        out[label] = runner.run_curve(rec, train, dev, progress=6)
        print(f"[train:{label}] final_eval_loss={out[label]['final_eval_loss']:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    target = args.target
    if target is None and "default" in out:
        target = out["default"]["final_eval_loss"]
    if target is None:
        print("\nno target: pass --compare or --target. The curve is printed below.")
        for micro, value in out["agent"]["curve"]:
            print(f"  {micro:6d}  {value:.6f}")
        return 0

    print(f"\n==== target {target:.6f} (devset PROXY, not the graded split) ====")
    for label in out:
        hit = harness.first_crossing(out[label]["curve"], target)
        print(f"  {label:9s} crosses at {hit} micro-batches "
              f"(final {out[label]['final_eval_loss']:.6f})")
    if "default" in out and "agent" in out:
        c = harness.first_crossing(out["default"]["curve"], target)
        a = harness.first_crossing(out["agent"]["curve"], target)
        if c and a:
            print(f"\n  displacement: {c - a} fewer micro-batches than the default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
