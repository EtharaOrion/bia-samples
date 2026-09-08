#!/usr/bin/env python3
"""The agent's own measuring instrument.

    python3 train_local.py --shape my_shape.json --compare

Trains a shape on the frozen budget and evaluates on data/devset_slice.bin, which
is a PROXY cut from a different FineWeb training shard. It is NOT the graded split
-- the graded split is staged only into the verifier image -- but it is measured by
this same harness.py, under the same frozen optimizer and schedule, on the same
token stream, so a shape that lowers the number here lowers the graded number too.

--compare also trains the shipped default shape in the same process, which is the
comparison the verifier makes.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness, shape_schema  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--devset", default=str(HERE / "data" / "devset_slice.bin"))
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device is visible; this harness needs the accelerator")
        return 2

    spec = harness.load_spec()
    bounds, band = spec["shape_bounds"], spec["parameter_band"]
    try:
        agent = shape_schema.load(args.shape, bounds)
    except shape_schema.Refusal as exc:
        print(f"REFUSED  {exc.reason}: {exc.detail}")
        return 2

    plan = [("agent", agent)]
    if args.compare:
        plan.insert(0, ("default", shape_schema.load(HERE / "default_shape.json", bounds)))

    for label, shape in plan:
        model, n = harness.build_model(shape, spec, device="cpu")
        del model
        try:
            shape_schema.parameter_band_check(n, band)
        except shape_schema.Refusal as exc:
            print(f"[{label}] REFUSED  {exc.reason}: {exc.detail}")
            if label == "agent":
                return 3

    harness.configure_backends()
    train = harness.to_device_tokens(harness.load_shard(HERE / "data" / "train_slice.bin"), "cuda")
    devset = harness.to_device_tokens(harness.load_shard(args.devset), "cuda")

    out = {}
    for label, shape in plan:
        print(f"\n[train:{label}] {shape['num_layers']}L x {shape['model_dim']}d "
              f"head_dim={shape['head_dim']} mlp={shape['mlp_ratio']}", flush=True)
        t0 = time.time()
        runner = harness.ShapeRunner(shape, spec, "cuda")
        report = runner.run(train, devset, progress=768)
        runner.release()
        out[label] = report
        print(f"[train:{label}] parameters={report['parameters']} "
              f"devset_loss={report['val_loss']:.6f} ({time.time()-t0:.0f}s wall)", flush=True)

    print("\n==== devset (PROXY, not the graded split) ====")
    for label, report in out.items():
        print(f"  {label:9s} {report['parameters']:10d} params  {report['val_loss']:.6f}")
    if "default" in out and "agent" in out:
        print(f"\n  improvement over the shipped default: "
              f"{out['default']['val_loss'] - out['agent']['val_loss']:+.6f} nats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
