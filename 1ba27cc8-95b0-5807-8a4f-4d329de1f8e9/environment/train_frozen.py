#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bia_data  # noqa: E402
import bia_optim  # noqa: E402
import bia_recipe  # noqa: E402
import frozen_gpt  # noqa: E402

MICRO_BATCH_DEFAULT = 32
AUTOCAST_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

def _autocast():
    dtype = AUTOCAST_DTYPE.get(os.environ.get("BIA_AUTOCAST", "bf16"))
    if dtype is None or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)

def micro_batch(per_step: int) -> int:
    requested = int(os.environ.get("BIA_MICRO_BATCH", "0"))
    if requested > 0:
        return min(requested, per_step)
    return max(1, min(per_step, MICRO_BATCH_DEFAULT))

def load_shape() -> dict:
    return json.loads(pathlib.Path(os.environ.get("BIA_SHAPE", HERE / "shape.json")).read_text())

def run(recipe: dict, shape: dict, seed: int, split_seed: int, points):
    device = os.environ.get("BIA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    seq = int(shape["seq_len"])
    per_step = int(shape["batch_tokens_per_step"]) // seq
    train = torch.from_numpy(bia_data.corpus_stream(
        "train", int(shape["data_draw_train"]), int(shape["stream_tokens_train"])))
    held = torch.from_numpy(bia_data.corpus_stream(
        "dev", int(split_seed), int(shape["stream_tokens_dev"])))
    model = frozen_gpt.FrozenGPT(shape).to(device)
    frozen_gpt.apply_init(model, recipe["init_scheme"], recipe["init_scale"], seed)
    groups = [
        {"params": [model.wte.weight, model.wpe.weight], "lr_mult": recipe["embed_lr_mult"]},
        {"params": [model.head.weight], "lr_mult": recipe["head_lr_mult"]},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith(("wte", "wpe", "head"))], "lr_mult": 1.0},
    ]
    optimizer = bia_optim.RecipeOptimizer(groups, recipe)
    total = int(recipe["max_steps"])
    offsets = bia_data.batch_order(seed, total * per_step, len(train) - seq - 1)
    curve = []
    cursor = 0
    micro = micro_batch(per_step)
    for step in range(1, total + 1):
        take = offsets[cursor:cursor + per_step]
        cursor += per_step
        model.zero_grad(set_to_none=True)
        for start in range(0, per_step, micro):
            chunk = take[start:start + micro]
            idx = torch.stack([train[int(o):int(o) + seq] for o in chunk]).to(device)
            tgt = torch.stack([train[int(o) + 1:int(o) + seq + 1] for o in chunk]).to(device)
            with _autocast():
                (model(idx, tgt) * (len(chunk) / per_step)).backward()
            del idx, tgt
        optimizer.clip_()
        optimizer.step(bia_optim.lr_at(recipe, step - 1, total))
        if step in points:
            curve.append({"step": step, "loss": evaluate(model, held, shape, device)})
    return curve

@torch.no_grad()
def evaluate(model, stream, shape: dict, device: str, windows: int = 32) -> float:
    seq = int(shape["seq_len"])
    model.eval()
    idx = torch.stack([stream[start:start + seq] for start in range(0, windows * seq, seq)]).to(device)
    tgt = torch.stack([stream[start + 1:start + seq + 1] for start in range(0, windows * seq, seq)]).to(device)
    with _autocast():
        value = float(model(idx, tgt))
    model.train()
    return value

def main() -> int:
    shape = load_shape()
    raw = json.loads(pathlib.Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {}
    normalized = bia_recipe.normalize(raw, shape)
    if normalized["frozen_axis_writes"]:
        print("refused, frozen axes named: " + ", ".join(normalized["frozen_axis_writes"]))
        return 2
    recipe = normalized["recipe"]
    points = set(bia_recipe.eval_points(shape, recipe["max_steps"]))
    curve = run(recipe, shape, int(os.environ.get("BIA_SEED", "1")), int(shape["data_draw_dev"]), points)
    print(json.dumps({"fingerprint": normalized["fingerprint"], "dev_curve": curve}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
