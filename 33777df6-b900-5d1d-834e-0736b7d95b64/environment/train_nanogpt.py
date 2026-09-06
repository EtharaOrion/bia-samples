#!/usr/bin/env python3
"""The pinned training path. It trains the real decoder and writes the checkpoint.

This is the file that produces the graded artifact. It instantiates the frozen
12-layer 768-dim decoder from `environment/nanogpt_substrate.json`, trains it on
the FineWeb10B train shards under the substrate's run declaration, and writes a
float32 parameter snapshot to the checkpoint path together with the digest of
the bytes it wrote.

It runs ONCE, at image build, under the pinned harness. The checkpoint it
produces is the frozen model this slot quantizes, and it is mounted read-only on
both the agent surface and the verifier surface so the two grade the same
parameters. A solver does not retrain it: the model is frozen and only the bit
allocation and the quantization scheme are free.

The run declaration is the substrate's and is not this file's to move: 524288
tokens per step, exactly one forward and one backward pass per step, target
validation loss below 3.28. Those values are read from the substrate, asserted
here, and recorded into the checkpoint manifest, so a training path that drifted
off the frozen axes is visible in the artifact rather than only in a log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from nanogpt_model import Decoder, architecture

HERE = Path(__file__).resolve().parent


def substrate() -> dict:
    return json.loads((HERE / "nanogpt_substrate.json").read_text(encoding="utf-8"))


def shard_tokens(path: Path) -> np.ndarray:
    """Read one FineWeb shard: a 1024-byte header, then little-endian uint16 ids."""
    with open(path, "rb") as handle:
        handle.read(1024)
        return np.frombuffer(handle.read(), dtype="<u2")


def batches(shard_paths, batch_tokens: int, seq_len: int):
    """Yield one training batch of exactly batch_tokens tokens, in shard order."""
    rows = batch_tokens // seq_len
    buffer = []
    for path in shard_paths:
        tokens = shard_tokens(path)
        limit = (len(tokens) - 1) // seq_len
        for index in range(limit):
            start = index * seq_len
            buffer.append(tokens[start : start + seq_len + 1])
            if len(buffer) == rows:
                block = torch.from_numpy(np.stack(buffer).astype(np.int64))
                buffer = []
                yield block[:, :seq_len], block[:, 1:]


def train(args) -> int:
    declaration = substrate()
    arch = architecture(HERE / "nanogpt_substrate.json")
    run = declaration["run"]
    if int(run["forward_passes_per_step"]) != 1 or int(run["backward_passes_per_step"]) != 1:
        raise ValueError("the substrate declares one forward and one backward pass per step; this path does not move that")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = Decoder(arch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=False
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / args.warmup) * max(0.0, 1.0 - step / args.steps)
    )

    shard_paths = sorted(Path(args.data).glob("fineweb_train_*.bin"))
    if not shard_paths:
        raise FileNotFoundError("no train shards under " + args.data + "; stage them with data/cached_fineweb10B.py")

    stream = batches(shard_paths, int(run["batch_tokens_per_step"]), arch["seq_len"])
    last_loss = float("nan")
    for step in range(args.steps):
        try:
            tokens, targets = next(stream)
        except StopIteration:
            break
        # Exactly one forward pass and exactly one backward pass. This is the
        # frozen axis, and it is also the reason the metric exists at all: delete
        # these two lines and there is no loss, no perplexity and no grade.
        loss = model(tokens.to(device), targets.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss)
        if step % args.log_every == 0:
            print("step " + str(step) + " train loss " + repr(round(last_loss, 4)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().float() for name, tensor in model.state_dict().items()}
    torch.save(state, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "oer20.checkpoint/v1",
                "model_id": "nanogpt-12L-768d",
                "checkpoint": out.name,
                "checkpoint_sha256": digest,
                "architecture": arch,
                "run": {
                    "batch_tokens_per_step": int(run["batch_tokens_per_step"]),
                    "forward_passes_per_step": 1,
                    "backward_passes_per_step": 1,
                    "steps": args.steps,
                    "seed": args.seed,
                    "target_val_loss": float(run["target_val_loss"]),
                },
                "final_train_loss": None if math.isnan(last_loss) else round(last_loss, 6),
                "tensor_count": len(state),
                "total_params": int(sum(tensor.numel() for tensor in state.values())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("wrote " + str(out) + " sha256 " + digest)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="train the frozen nanoGPT decoder")
    parser.add_argument("--data", default="/workspace/data/fineweb10B")
    parser.add_argument("--out", default="/checkpoint/nanogpt_fp32.pt")
    parser.add_argument("--steps", type=int, default=3250)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=50)
    return train(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
