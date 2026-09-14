#!/usr/bin/env python3
"""Run one recipe against the DEVELOPMENT split and print its own loss curve.

This is the agent's exploration tool. It is not the grader and its numbers are
not the graded numbers. The graded evaluation happens inside the verifier
environment, on a split whose seed does not exist on this surface, at evaluation
points the verifier schedules, and with no smoothing on the graded path.

Usage:
    python3 train_frozen.py recipe.json
"""
from __future__ import annotations

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


def load_shape() -> dict:
    return json.loads(pathlib.Path(os.environ.get("BIA_SHAPE", HERE / "shape.json")).read_text())


def run(recipe: dict, shape: dict, seed: int, split_seed: int, points):
    """Train for recipe['max_steps'] and read the loss at each requested point."""
    device = os.environ.get("BIA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    seq = int(shape["seq_len"])
    per_step = int(shape["batch_tokens_per_step"]) // seq
    train = torch.from_numpy(bia_data.markov_stream(
        int(shape["data_seed_dist"]), int(shape["data_draw_train"]),
        int(shape["stream_tokens_train"]), int(shape["vocab_size"])))
    held = torch.from_numpy(bia_data.markov_stream(
        int(shape["data_seed_dist"]), int(split_seed),
        int(shape["stream_tokens_dev"]), int(shape["vocab_size"])))
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
    for step in range(1, total + 1):
        take = offsets[cursor:cursor + per_step]
        cursor += per_step
        idx = torch.stack([train[int(o):int(o) + seq] for o in take]).to(device)
        tgt = torch.stack([train[int(o) + 1:int(o) + seq + 1] for o in take]).to(device)
        loss = model(idx, tgt)
        model.zero_grad(set_to_none=True)
        loss.backward()
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
