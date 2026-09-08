#!/usr/bin/env python3
"""Instantiate a shape and report what it actually costs. CPU only, seconds.

    python3 shape_tool.py --shape my_shape.json
    python3 shape_tool.py --enumerate

`--shape` validates a shape against the schema, builds it, counts its parameters
and says whether it lands inside the band. Nothing here is a formula: the count
comes from the instantiated module's own tensors, which is exactly how the
verifier decides.

`--enumerate` walks the whole declared bound box and prints every shape that lands
inside the band. There is nothing secret about which shapes are legal, and finding
them is not the difficulty of this task -- choosing among them is.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness, shape_schema  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="")
    ap.add_argument("--enumerate", action="store_true")
    args = ap.parse_args()

    spec = harness.load_spec()
    bounds, band = spec["shape_bounds"], spec["parameter_band"]

    if args.enumerate:
        lo_l, hi_l = bounds["num_layers"]
        lo_d, hi_d = bounds["model_dim"]
        mult = bounds["model_dim_multiple_of"]
        rows = []
        for dim in range(lo_d, hi_d + 1, mult):
            for ratio in bounds["mlp_ratio_allowed"]:
                for layers in range(lo_l, hi_l + 1):
                    shape = {"schema": shape_schema.SCHEMA_ID, "num_layers": layers,
                             "model_dim": dim, "head_dim": 64, "mlp_ratio": ratio}
                    if dim % 64:
                        continue
                    try:
                        shape_schema.validate(shape, bounds)
                    except shape_schema.Refusal:
                        continue
                    model, n = harness.build_model(shape, spec, device="cpu")
                    del model
                    if band["min"] <= n <= band["max"]:
                        heads = [h for h in bounds["head_dim_allowed"]
                                 if dim % h == 0 and dim // h >= bounds["min_num_heads"]]
                        rows.append((layers, dim, ratio, n, heads))
        print(f"{'layers':>7}{'dim':>6}{'mlp':>5}{'params':>12}   head_dim choices")
        for r in sorted(rows):
            print(f"{r[0]:7d}{r[1]:6d}{r[2]:5d}{r[3]:12d}   {r[4]}")
        print(f"\n{len(rows)} (num_layers, model_dim, mlp_ratio) combinations land inside "
              f"[{band['min']}, {band['max']}]")
        print("head_dim does not change the parameter count. It changes how the same "
              "residual width is split into heads, and it is a free axis of its own.")
        return 0

    if not args.shape:
        ap.error("give --shape or --enumerate")
    try:
        shape = shape_schema.load(args.shape, bounds)
    except shape_schema.Refusal as exc:
        print(f"REFUSED  {exc.reason}: {exc.detail}")
        return 2
    model, count = harness.build_model(shape, spec, device="cpu")
    roles = {"embed": 0, "hidden": 0, "head": 0, "scalar": 0}
    for name, p in model.named_parameters():
        key = ("scalar" if p.ndim < 2 else "embed" if name.startswith("embed")
               else "head" if name.startswith("proj") else "hidden")
        roles[key] += p.numel()
    print(json.dumps({k: v for k, v in shape.items() if k != "notes"}, indent=2))
    print(f"\nnum_heads          {shape['model_dim'] // shape['head_dim']}")
    print(f"parameters         {count}")
    for k, v in roles.items():
        print(f"  {k:16s} {v:12d}  {100.0*v/count:5.1f}%")
    print(f"band               [{band['min']}, {band['max']}]")
    try:
        shape_schema.parameter_band_check(count, band)
        print("VERDICT            inside the band")
    except shape_schema.Refusal as exc:
        print(f"VERDICT            REFUSED -- {exc.reason}: {exc.detail}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
