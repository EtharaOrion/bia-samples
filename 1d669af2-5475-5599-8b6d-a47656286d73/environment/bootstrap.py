#!/usr/bin/env python3
"""Build-time staging: the FineWeb10B train shards, then the nanoGPT checkpoint.

This script runs once, at image build, and never during an attempt. It produces the
two pieces of built environment state this slot needs: the pinned train shards the
calibration slices are cut from, and a real parameter snapshot of the frozen 12-layer
768-dim decoder for the task to quantize.

Nothing here is a stand-in. `checkpoint` instantiates the architecture that
environment/nanogpt_substrate.json declares, trains it with one forward and one
backward pass per optimizer step over the pinned train shards under a pinned seed,
and writes its real state dict. Delete `GPT.forward` in environment/model.py and this
script fails rather than emitting a checkpoint-shaped file with no model behind it.

Two subcommands:

    corpus      stage the shards by invoking the pinned upstream loader, which is
                data/cached_fineweb10B.py from modded-nanogpt, vendored into the
                image. The loader is upstream-owned and is not reimplemented here.
    checkpoint  train the frozen decoder over those shards and write the snapshot.

Only the TRAIN split is ever staged. Passing --split val is refused: the validation
shards are the graded surface and they belong to the verifier's container alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PINNED_LOADER = "data/cached_fineweb10B.py"
CHECKPOINT_SCHEMA = "oer22.checkpoint/v1"


def stage_corpus(root: Path, split: str, shards: int, loader: Path) -> int:
    if split != "train":
        raise SystemExit(
            "refusing to stage the " + split + " split into an agent-visible image. The validation "
            "shards are the graded surface and are resolved by the verifier from its own admin "
            "plane at tests/bound.json."
        )
    if not loader.is_file():
        raise SystemExit(
            "the pinned upstream loader " + PINNED_LOADER + " is absent from this image at "
            + str(loader) + ". It is a build input and it is not reimplemented here."
        )
    root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(loader), str(int(shards))],
        cwd=str(root.parent.parent),
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("the pinned loader exited " + str(result.returncode))
    staged = sorted(root.glob("fineweb_train_*.bin"))
    print(json.dumps({"staged_train_shards": [p.name for p in staged]}, indent=2, sort_keys=True))
    return 0


def _learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_checkpoint(declaration: Path, shards: Path, out: Path, steps: int, seed: int,
                     micro_rows: int, peak_lr: float, device: str) -> int:
    import torch  # noqa: PLC0415

    import model as frozen  # noqa: PLC0415

    # TF32 for the TRAINING matmuls only, and only in this process. It is the standard
    # setting for this accelerator class, it moves no dtype, no architecture, no
    # optimizer and no threshold, and it is what makes a step count worth having fit the
    # envelope this slot declares: one optimizer step at the frozen batch of 524288
    # tokens measures 9.54s without it and 3.25s with it on the H100 this batch runs on.
    # The GRADING process does not import this module and is unaffected, so every
    # perplexity on the graded path is still read in full fp32.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    substrate = frozen.load_declaration(declaration)
    arch = frozen.architecture(substrate)
    run = substrate["run"]
    batch_tokens = int(run["batch_tokens_per_step"])
    seq_len = arch["seq_len"]
    rows_per_step = batch_tokens // seq_len
    micro_rows = max(1, min(micro_rows, rows_per_step))
    if rows_per_step % micro_rows != 0:
        raise SystemExit("the micro-batch row count must divide the step's row count exactly")
    micro_batches = rows_per_step // micro_rows

    torch.manual_seed(seed)
    stream = frozen.load_stream(sorted(Path(shards).glob("fineweb_train_*.bin")))
    if stream.size == 0:
        raise SystemExit("no train shards under " + str(shards))

    net = frozen.build(arch, device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(1, int(0.02 * steps))

    offset = 0
    consumed = 0
    forward_passes = 0
    backward_passes = 0
    last_loss = None
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step, steps, peak_lr, warmup)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(micro_batches):
            if offset + seq_len * micro_rows + 1 > stream.size:
                offset = 0
            batch = frozen._batches(stream, seq_len, micro_rows, offset, device)
            if batch is None:
                break
            tokens, targets = batch
            loss = net(tokens, targets) / micro_batches
            forward_passes += 1
            loss.backward()
            backward_passes += 1
            step_loss += float(loss.detach())
            offset += seq_len * micro_rows
            consumed += seq_len * micro_rows
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        last_loss = step_loss

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": dict(arch),
            "steps": int(steps),
            "seed": int(seed),
            "tokens_consumed": int(consumed),
            "forward_passes_per_step": 1,
            "backward_passes_per_step": 1,
            "micro_batches_per_step": int(micro_batches),
            "final_train_loss_telemetry": last_loss,
            "state_dict": {k: v.detach().to("cpu", torch.float32) for k, v in net.state_dict().items()},
        },
        str(out),
    )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    shapes = frozen.checkpoint_shapes(out)
    expected = frozen.expected_parameter_shapes(arch)
    if shapes != expected:
        raise SystemExit("the written checkpoint does not bind to the declared architecture")
    print(
        json.dumps(
            {
                "checkpoint": out.as_posix(),
                "content_sha256": digest,
                "parameters": len(shapes),
                "forward_passes_total": forward_passes,
                "backward_passes_total": backward_passes,
                "architecture_bound": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus")
    corpus.add_argument("--root", required=True)
    corpus.add_argument("--split", default="train")
    corpus.add_argument("--shards", type=int, default=8)
    corpus.add_argument("--loader", default="/workspace/" + PINNED_LOADER)

    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("--declaration", required=True)
    checkpoint.add_argument("--shards", required=True)
    checkpoint.add_argument("--out", required=True)
    checkpoint.add_argument("--steps", type=int, default=3250)
    checkpoint.add_argument("--seed", type=int, default=1337)
    checkpoint.add_argument("--micro-rows", type=int, default=16)
    checkpoint.add_argument("--peak-lr", type=float, default=0.0018)
    checkpoint.add_argument("--device", default="cuda")

    args = parser.parse_args(argv)
    if args.command == "corpus":
        return stage_corpus(Path(args.root), args.split, args.shards, Path(args.loader))
    return build_checkpoint(
        Path(args.declaration), Path(args.shards), Path(args.out), args.steps, args.seed,
        args.micro_rows, args.peak_lr, args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
