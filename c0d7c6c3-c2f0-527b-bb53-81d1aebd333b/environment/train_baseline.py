"""The shipped baseline recipe. This is the artifact you are asked to beat.

It is deliberately competent and deliberately unremarkable: AdamW at a constant
learning rate on every parameter, no warmup, no schedule, no per-tensor
treatment. It reaches the target eventually. The task is to reach it in fewer
optimizer steps.

The structure below is the invocation contract the verifier re-executes. Keep
it: read the seed from BIA_SEED, build the frozen model, draw every step from
the provided loader, and call loader.checkpoint(model) once per step. What you
may change is the optimizer, its hyperparameters and their schedules, and the
model initialization.
"""
from __future__ import annotations

import json
import os
import pathlib

import torch

from bia_loader import StepLoader
from frozen_gpt import GPT, resolve_device

SHAPE_PATH = pathlib.Path(os.environ.get("BIA_SHAPE", "/env/shape.json"))


def build_model(shape: dict, device) -> GPT:
    return GPT(vocab_size=shape["vocab_size"], num_layers=shape["num_layers"],
               model_dim=shape["model_dim"], head_dim=shape["head_dim"]).to(device)


def main() -> None:
    shape = json.loads(SHAPE_PATH.read_text())
    seed = int(os.environ["BIA_SEED"])
    torch.manual_seed(seed)
    device = resolve_device()

    model = build_model(shape, device)
    loader = StepLoader(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.0)

    for inputs, targets in loader.steps():
        loss = model(inputs, targets) / targets.numel()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        loader.checkpoint(model)


if __name__ == "__main__":
    main()
