from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bia_data  # noqa: E402
import bia_optim  # noqa: E402
import frozen_gpt  # noqa: E402

VAL_DRAW = int(os.environ.get("BIA_VAL_DRAW", "33"))

EVAL_WINDOWS = 32

MICRO_BATCH_DEFAULT = 32

AUTOCAST_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

def _autocast():
    dtype = AUTOCAST_DTYPE.get(os.environ.get("BIA_AUTOCAST", ""))
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)

def device_name() -> str:
    return os.environ.get("BIA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

def _weights_digest(model) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        flat = param.detach().reshape(-1)[:256].to("cpu", torch.float32).numpy().tobytes()
        hasher.update(flat)
        hasher.update(repr(round(float(param.detach().float().norm()), 6)).encode("utf-8"))
    return "sha256:" + hasher.hexdigest()

def run_digest(recipe: dict, seed: int, index: int) -> str:
    payload = json.dumps({"recipe": recipe, "seed": seed, "index": index},
                         sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

def micro_batch(per_step: int) -> int:
    requested = int(os.environ.get("BIA_MICRO_BATCH", "0"))
    if requested > 0:
        return min(requested, per_step)
    return max(1, min(per_step, MICRO_BATCH_DEFAULT))

def seed_for(fingerprint: str) -> int:
    return int(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)

@torch.no_grad()
def _evaluate(model, stream, shape: dict, device: str) -> float:
    seq = int(shape["seq_len"])
    model.eval()
    idx = torch.stack([stream[start:start + seq]
                       for start in range(0, EVAL_WINDOWS * seq, seq)]).to(device)
    tgt = torch.stack([stream[start + 1:start + seq + 1]
                       for start in range(0, EVAL_WINDOWS * seq, seq)]).to(device)
    with _autocast():
        value = float(model(idx, tgt))
    model.train()
    return value

class Substrate:

    def __init__(self, shape: dict):
        self.shape = shape
        self.train = torch.from_numpy(bia_data.corpus_stream(
            "train", int(shape["data_draw_train"]), int(shape["stream_tokens_train"])))
        self.val = torch.from_numpy(bia_data.corpus_stream(
            "val", VAL_DRAW, int(shape["stream_tokens_val"])))

def train_and_evaluate(substrate: Substrate, recipe: dict, seed: int, index: int) -> dict:
    shape = substrate.shape
    device = device_name()
    torch.manual_seed(seed)
    seq = int(shape["seq_len"])
    per_step = max(1, int(shape["batch_tokens_per_step"]) // seq)
    stride = int(shape["eval_stride"])
    total = int(recipe["max_steps"])
    digest = run_digest(recipe, seed, index)

    model = frozen_gpt.FrozenGPT(shape).to(device)
    frozen_gpt.apply_init(model, recipe["init_scheme"], recipe["init_scale"], seed)
    groups = [
        {"params": [model.wte.weight, model.wpe.weight], "lr_mult": float(recipe["embed_lr_mult"])},
        {"params": [model.head.weight], "lr_mult": float(recipe["head_lr_mult"])},
        {"params": [param for name, param in model.named_parameters()
                    if not name.startswith(("wte", "wpe", "head"))], "lr_mult": 1.0},
    ]
    optimizer = bia_optim.RecipeOptimizer(groups, recipe)
    offsets = bia_data.batch_order(seed, total * per_step + per_step, len(substrate.train) - seq - 2)

    evals = []
    forwards = 0
    backwards = 0
    cursor = 0
    micro = micro_batch(per_step)
    for step in range(1, total + 1):
        take = offsets[cursor:cursor + per_step]
        cursor += per_step
        model.zero_grad(set_to_none=True)
        for start in range(0, per_step, micro):
            chunk = take[start:start + micro]
            idx = torch.stack([substrate.train[int(o):int(o) + seq] for o in chunk]).to(device)
            tgt = torch.stack([substrate.train[int(o) + 1:int(o) + seq + 1] for o in chunk]).to(device)
            with _autocast():
                (model(idx, tgt) * (len(chunk) / per_step)).backward()
            del idx, tgt
        forwards += 1
        backwards += 1
        optimizer.clip_()
        optimizer.step(bia_optim.lr_at(recipe, step - 1, total))
        if step % stride == 0:
            evals.append({
                "step": step,
                "loss": round(_evaluate(model, substrate.val, shape, device), 6),
                "smoothing": "none",
                "source": "verifier-recomputed",
                "weights_provenance": "harness-step-state",
                "weights_run_digest": digest,
                "weights_digest": _weights_digest(model),
            })
    return {
        "run_digest": digest,
        "verifier_evals": evals,
        "halted_at_step": total,
        "forward_passes_per_step": 1 if forwards == total else 0,
        "backward_passes_per_step": 1 if backwards == total else 0,
    }
