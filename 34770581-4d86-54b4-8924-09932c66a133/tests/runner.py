#!/usr/bin/env python3
"""The verifier's own executor. It launches the submission and never imports it.

Four facts about this file carry the whole isolation argument.

It never imports the submission. The submission is copied ALONE into a fresh
temporary directory, launched as a new session leader under a small environment
allowlist, its output captured, and its entire process group killed in a
`finally` block. Nothing it defines ever enters this interpreter.

It owns the feed. The submission produces one artifact, `plan.json`. This module
reads that plan and executes it with its own executor, counting tokens from the
FineWeb10B slices it fed rather than from anything the submission claimed.

It owns the model. It builds the frozen decoder itself, at the shape
`environment/nanogpt_substrate.json` declares, and trains it here: one forward
pass and one backward pass over each 524288 token step. The graded parameters
are the parameters this process produced. No checkpoint crosses the boundary.

It owns the evaluation split. The frozen split lives in `tests/heldout_spec.json`,
which is verifier side and holds raw FineWeb10B validation tokens. Harbor
assembles the agent surface from task.toml, instruction.md and environment/
alone, so tests/ is absent from that surface by construction rather than by a
filter that could be misconfigured.

No clock, no network, no unseeded random source. Every number below is a
deterministic function of the shard bytes, the split bytes, the plan and the
frozen seed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent

# The whole environment a submission is given. Anything not named here is not
# inherited, so a submission cannot be steered by an ambient variable and cannot
# read one the grading process holds.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED", "OER08_CORPUS", "OER08_DATA")

SUBMISSION_TIMEOUT_SECONDS = 600
SEED = 1337
MICRO_SEQUENCES = 32

# The verifier's declared constant-weight calibration set: the five vertices of
# the handed simplex plus its centroid, which is mixture.yaml's own template.
# Every member is fed through the trainer below, so the calibration point is an
# achieved loss and never an unreachable infimum.
CALIBRATION_BANDS = ("band-0", "band-1", "band-2", "band-3", "band-4")


# --------------------------------------------------------------------------
# substrate, owned here


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def architecture(corpus: dict, substrate: dict) -> dict:
    declared = substrate["architecture"]
    mirrored = corpus["substrate"]["architecture"]
    for key in ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len"):
        if int(declared[key]) != int(mirrored[key]):
            raise ValueError("frozen architecture disagrees at " + key)
    return {key: int(declared[key]) for key in declared if not key.startswith("_")}


def data_root(corpus: dict) -> Path:
    return Path(os.environ.get("OER08_DATA", corpus["corpus"]["container_train_path"]))


def read_tokens(root: Path, shard: str, offset: int, count: int) -> np.ndarray:
    array = np.memmap(root / shard, dtype=np.uint16, mode="r", offset=64 * 4)
    window = np.asarray(array[offset:offset + count], dtype=np.int64)
    if window.shape[0] != count:
        raise ValueError("shard " + shard + " is short at offset " + str(offset))
    return window


def slice_digest(tokens: np.ndarray) -> str:
    return hashlib.sha256(tokens.astype("<u2").tobytes()).hexdigest()


def rare_fraction(tokens: np.ndarray) -> float:
    return float((tokens >= 20000).sum()) / float(tokens.shape[0])


def pool_index(corpus: dict, root: Path) -> dict:
    pool = corpus["pool"]
    width = int(pool["slice_tokens"])
    rows = {}
    for shard in corpus["corpus"]["shards"]:
        stem = shard.replace(".bin", "")
        for index in range(int(pool["slices_per_shard"])):
            tokens = read_tokens(root, shard, index * width, width)
            rows[stem + ":" + str(index)] = {
                "id": stem + ":" + str(index),
                "shard": shard,
                "offset": index * width,
                "tokens": width,
                "rare_fraction": rare_fraction(tokens),
                "digest": slice_digest(tokens),
            }
    order = sorted(rows, key=lambda i: (rows[i]["rare_fraction"], rows[i]["shard"], rows[i]["offset"]))
    group = len(order) // int(pool["bands"]["count"])
    for position, ident in enumerate(order):
        rows[ident]["band"] = pool["bands"]["ids"][min(position // group, int(pool["bands"]["count"]) - 1)]
    return rows


def pool_digest(index: dict) -> str:
    payload = json.dumps(
        [[ident, index[ident]["digest"]] for ident in sorted(index)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def band_members(index: dict, band: str) -> list:
    return sorted(
        (ident for ident in index if index[ident]["band"] == band),
        key=lambda ident: (index[ident]["shard"], index[ident]["offset"]),
    )


def heldout_tokens(split: dict) -> np.ndarray:
    payload = base64.b64decode(split["held_out_split"]["payload_base64"])
    return np.frombuffer(payload, dtype="<u2").astype(np.int64)


def heldout_digests(split: dict, width: int) -> list:
    """Content digests of the split at the pool's own slice width.

    A fed slice that carries held-out bytes has one of these digests, whatever
    id the pool gave it, so the leak test is over bytes and not over labels.
    """
    tokens = heldout_tokens(split)
    return sorted(
        slice_digest(tokens[start:start + width])
        for start in range(0, tokens.shape[0] - width + 1, width)
    )


# --------------------------------------------------------------------------
# the frozen decoder, built here


class SelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, time, dim = x.shape
        q, k, v = self.qkv(x).split(dim, dim=2)
        shape = (batch, time, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(batch, time, dim))


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim, bias=False)
        self.attention = SelfAttention(dim, head_dim)
        self.norm_mlp = nn.LayerNorm(dim, bias=False)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.down(F.gelu(self.up(self.norm_mlp(x))))


class Decoder(nn.Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.arch = arch
        dim = arch["model_dim"]
        self.embed = nn.Embedding(arch["vocab_size"], dim)
        self.position = nn.Embedding(arch["seq_len"], dim)
        self.blocks = nn.ModuleList([Block(dim, arch["head_dim"]) for _ in range(arch["num_layers"])])
        self.norm = nn.LayerNorm(dim, bias=False)
        self.head = nn.Linear(dim, arch["vocab_size"], bias=False)

    def forward(self, tokens):
        position = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.position(position)[None]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def parameter_shapes(model: Decoder) -> dict:
    return {name: list(tensor.shape) for name, tensor in sorted(model.state_dict().items())}


def parameter_digest(model: Decoder) -> str:
    running = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        running.update(name.encode("ascii"))
        running.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
    return running.hexdigest()


def parameter_count(model: Decoder) -> int:
    return int(sum(tensor.numel() for tensor in model.state_dict().values()))


# --------------------------------------------------------------------------
# the feed and the training run the verifier performs


def expand(corpus: dict, index: dict, plan: dict) -> list:
    budget = int(corpus["budget"]["budget_tokens"])
    width = int(corpus["pool"]["slice_tokens"])
    mode = str(plan.get("mode", ""))
    order = []
    if mode == "constant":
        weights = plan.get("weights") or {}
        for band in sorted(weights):
            weight = float(weights[band])
            if weight <= 0.0:
                continue
            members = band_members(index, band)
            if not members:
                raise KeyError("plan names a band outside the pool: " + str(band))
            for position in range(int(round(weight * budget / width))):
                order.append(members[position % len(members)])
    elif mode == "schedule":
        for draw in plan.get("draws") or []:
            ident = str(draw.get("slice"))
            if ident not in index:
                raise KeyError("plan names a slice outside the pool: " + ident)
            repeat = int(draw.get("repeat", 1))
            if repeat < 1:
                raise ValueError("repeat below one is not a draw: " + str(draw))
            order.extend([ident] * repeat)
    else:
        raise ValueError("plan mode outside the grammar: " + repr(mode))
    return order


def train(corpus: dict, arch: dict, index: dict, order: list, root: Path, device: str) -> dict:
    """Instantiate the frozen decoder and train it on the fed slices.

    A step is `tokens_per_step` fed tokens. The gradient over those tokens is
    accumulated across micro-sequences and applied once, so the run performs
    exactly one forward pass and one backward pass over each step's batch. Remove
    those two passes and there is no parameter snapshot, so there is no graded
    scalar at all rather than a different one.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = Decoder(arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    width = int(corpus["pool"]["slice_tokens"])
    slices_per_step = int(corpus["budget"]["tokens_per_step"]) // width
    steps_planned = max(1, len(order) // slices_per_step)
    ledger = []
    touched = {}
    fed = 0
    steps = 0
    pending = []
    model.train()
    for ident in order:
        row = index[ident]
        tokens = read_tokens(root, row["shard"], row["offset"], row["tokens"])
        block = torch.from_numpy(tokens).view(-1, arch["seq_len"])
        ledger.append({"index": len(ledger), "band": row["band"], "slice": ident, "tokens": width})
        touched[ident] = row["digest"]
        fed += width
        pending.append((block[:, :-1].contiguous(), block[:, 1:].contiguous()))
        if len(pending) < slices_per_step:
            continue
        rate = 6e-4 * 0.5 * (1.0 + math.cos(math.pi * steps / steps_planned))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        rows = sum(chunk[0].shape[0] for chunk in pending)
        for inputs, targets in pending:
            for start in range(0, inputs.shape[0], MICRO_SEQUENCES):
                x = inputs[start:start + MICRO_SEQUENCES].to(device)
                y = targets[start:start + MICRO_SEQUENCES].to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().view(-1, arch["vocab_size"]), y.reshape(-1))
                (loss * (x.shape[0] / rows)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        pending = []
    return {
        "model": model,
        "ledger": ledger,
        "touched": touched,
        "tokens_fed": fed,
        "steps": steps,
    }


@torch.no_grad()
def measure(model: Decoder, tokens: np.ndarray, seq_len: int, device: str) -> dict:
    """One forward pass of the trained snapshot over a split. Raw accumulators only.

    The scalar is `nats_sum / token_count`. Both accumulators are recorded beside
    it so tests/checkers.py recomputes the scalar from integers and a sum rather
    than trusting the float this function returned.
    """
    model.eval()
    usable = (tokens.shape[0] // seq_len) * seq_len
    block = torch.from_numpy(np.ascontiguousarray(tokens[:usable])).view(-1, seq_len)
    inputs, targets = block[:, :-1], block[:, 1:]
    nats = 0.0
    counted = 0
    for start in range(0, inputs.shape[0], MICRO_SEQUENCES):
        x = inputs[start:start + MICRO_SEQUENCES].contiguous().to(device)
        y = targets[start:start + MICRO_SEQUENCES].contiguous().to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        total = F.cross_entropy(
            logits.float().view(-1, model.arch["vocab_size"]), y.reshape(-1), reduction="sum"
        )
        nats += float(total)
        counted += int(y.numel())
    return {"nats_sum": nats, "token_count": counted, "loss": nats / counted}


def folds_of(tokens: np.ndarray) -> dict:
    """The two confirmation folds: contiguous halves of the frozen split."""
    half = (tokens.shape[0] // 2)
    return {"fold_a": tokens[:half], "fold_b": tokens[half:]}


def filter_state(corpus: dict, index: dict) -> dict:
    """What the pool publisher DECLARED it filtered, against what actually held.

    The declared filter is not applied here. A grader that silently repairs the
    pool it is grading turns a failed filter into a clean verdict, which is the
    exact silence this checker exists to break.
    """
    declared = str(((corpus.get("filter") or {}).get("declared")) or "none")
    if declared == "none":
        effective = "none"
    elif declared == "heldout-shard-drop":
        tainted = [row for row in index.values() if str(row["shard"]).startswith("fineweb_val_")]
        effective = "none" if tainted else "heldout-shard-drop"
    else:
        effective = "unrecognised"
    return {"filter_declared": declared, "filter_effective": effective}


# --------------------------------------------------------------------------
# calibration: concrete plans the verifier trains for itself


def calibration(corpus: dict, arch: dict, index: dict, root: Path, device: str, split: np.ndarray, folds: dict) -> dict:
    """The two substrate-local calibration points, each an achieved loss.

    These are NOT the family anchors. F12 anchors are unmeasured and are declared
    absent in task.toml under gap-oer-per-family-anchors-unmeasured.

    default_simplex_optimum is the best whole-split loss over the verifier's
    DECLARED constant-weight calibration set: the five band vertices of the handed
    simplex plus its centroid, which is exactly mixture.yaml's template. It is a
    best over a declared finite set and not a closed-form optimum over the whole
    simplex, because a real training run admits no closed form; that substitution
    is carried under gap-oer08-constant-weight-calibration-is-a-declared-set.

    reference_optimum is a schedule-mode plan this module DERIVES rather than
    transcribes: the pool slices whose rare_fraction sits closest to the frozen
    split's own rare_fraction, repeated to fill the budget exactly. It is an
    allocation at a granularity the constant-weight template cannot express, and
    no loss figure for it is authored anywhere in this bundle.
    """
    def score(plan_id: str, plan: dict) -> dict:
        run = train(corpus, arch, index, expand(corpus, index, plan), root, device)
        row = {
            "plan": {"id": plan_id, **plan},
            "full": measure(run["model"], split, arch["seq_len"], device)["loss"],
        }
        for fold in sorted(folds):
            row[fold] = measure(run["model"], folds[fold], arch["seq_len"], device)["loss"]
        del run
        return row

    plans = {"vertex:" + band: {"mode": "constant", "weights": {band: 1.0}} for band in CALIBRATION_BANDS}
    plans["centroid"] = {
        "mode": "constant",
        "weights": {band: 1.0 / len(CALIBRATION_BANDS) for band in CALIBRATION_BANDS},
    }
    best = None
    for name in sorted(plans):
        row = score(name, plans[name])
        if best is None or row["full"] < best["full"]:
            best = row

    return {"default_simplex_optimum": best, "reference_optimum": score("reference", reference_plan(corpus, index, split))}


def reference_plan(corpus: dict, index: dict, split: np.ndarray) -> dict:
    """Match the split's own measured composition at slice granularity, then repeat.

    The verifier owns the split, so it reads the target statistic off the split
    itself rather than off a number this bundle wrote down. The selection is
    inside bands and the repetition fills the budget, which are the two moves the
    handed template cannot make.
    """
    target = rare_fraction(split)
    width = int(corpus["pool"]["slice_tokens"])
    instances = int(corpus["budget"]["budget_tokens"]) // width
    keep = max(1, int(math.isqrt(instances)))
    chosen = sorted(
        sorted(index, key=lambda i: (abs(index[i]["rare_fraction"] - target), i))[:keep]
    )
    base, extra = divmod(instances, len(chosen))
    return {
        "schema": "oer08.plan/v2",
        "mode": "schedule",
        "draws": [
            {"slice": ident, "repeat": base + (1 if position < extra else 0)}
            for position, ident in enumerate(chosen)
        ],
    }


# --------------------------------------------------------------------------
# launching the submission, without importing it


def run_submission(submission: Path, corpus_path: Path) -> dict:
    """Copy the submission alone into a fresh directory, launch it, kill its group."""
    workspace = Path(tempfile.mkdtemp(prefix="oer08-submission-"))
    process = None
    try:
        target = workspace / submission.name
        shutil.copy2(submission, target)
        environment = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
        environment["OER08_CORPUS"] = str(corpus_path)
        argv = [sys.executable, str(target)]
        if submission.suffix == ".sh":
            argv = ["/bin/bash", str(target)]
        process = subprocess.Popen(
            argv,
            cwd=str(workspace),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=SUBMISSION_TIMEOUT_SECONDS)
            code = process.returncode
        except subprocess.TimeoutExpired:
            out, err, code = b"", b"submission-timeout", 124
        plan_path = workspace / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
        report_path = workspace / "report.json"
        report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
        return {
            "plan": plan,
            "exit_code": code,
            "stdout": out.decode("utf-8", "replace"),
            "stderr": err.decode("utf-8", "replace"),
            "self_reported": report,
        }
    finally:
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        shutil.rmtree(workspace, ignore_errors=True)


# --------------------------------------------------------------------------
# telemetry the verifier's own process produces


def evaluate(corpus_path: Path, split_path: Path, readout_path: Path, plan: dict, evidence: Path) -> dict:
    corpus = load_json(corpus_path)
    substrate = load_json(corpus_path.parent / "nanogpt_substrate.json")
    split_spec = load_json(split_path)
    readout = load_json(readout_path)
    arch = architecture(corpus, substrate)
    budget = int(corpus["budget"]["budget_tokens"])
    width = int(corpus["pool"]["slice_tokens"])
    root = data_root(corpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = pool_index(corpus, root)
    fed = train(corpus, arch, index, expand(corpus, index, plan), root, device)
    model = fed["model"]

    digest_of_ledger = hashlib.sha256(
        json.dumps(fed["ledger"], sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    params_digest = parameter_digest(model)

    feed_row = {
        "schema": "oer08.feed/v2",
        "produced_by": "tests/runner.py",
        "plan_mode": plan.get("mode"),
        "budget": budget,
        "budget_unit": "tokens",
        "declared_budget_unit": str(corpus["budget"]["unit"]),
        "tokens_per_step": int(corpus["budget"]["tokens_per_step"]),
        "tokens_fed": fed["tokens_fed"],
        "optimizer_steps": fed["steps"],
        "forward_passes_per_step": 1,
        "backward_passes_per_step": 1,
        "batch_count": len(fed["ledger"]),
        "batches": fed["ledger"],
        "documents_touched": fed["touched"],
        "pool_digest": pool_digest(index),
        "graded_pool_digest": pool_digest(index),
        "ledger_digest": digest_of_ledger,
    }
    feed_row.update(filter_state(corpus, index))

    model_row = {
        "schema": "oer08.model/v2",
        "produced_by": "tests/runner.py",
        "provenance": "harness-owned",
        "declared_architecture": {key: arch[key] for key in sorted(arch)},
        "parameter_shapes": parameter_shapes(model),
        "parameter_count": parameter_count(model),
        "params_digest": params_digest,
        "tokens": fed["tokens_fed"],
        "batch_index": len(fed["ledger"]),
        "derived_from_feed_digest": digest_of_ledger,
    }

    split_tokens = heldout_tokens(split_spec)
    folds = folds_of(split_tokens)
    records = [{"point": "bound", "fold": "full", "tokens": split_tokens}]
    for name in sorted(folds):
        records.append({"point": "confirm", "fold": name, "tokens": folds[name]})
    for record in records:
        measured = measure(model, record.pop("tokens"), arch["seq_len"], device)
        record.update(measured)
        record["batch_index"] = len(fed["ledger"])
        record["params_digest"] = params_digest

    eval_row = {
        "schema": "oer08.eval/v2",
        "produced_by": "tests/runner.py",
        "declared_smoothing": readout.get("smoothing"),
        "declared_ema_beta": readout.get("ema_beta"),
        "applied_smoothing": "none",
        "declared_architecture": {key: arch[key] for key in sorted(arch)},
        "params_digest": params_digest,
        "bound_point_reached": fed["tokens_fed"] >= budget,
        "bound_batch_index": len(fed["ledger"]),
        "schedule": [record["fold"] for record in records],
        "records": records,
        "heldout_digests": heldout_digests(split_spec, width),
        "calibration": calibration(corpus, arch, index, root, device, split_tokens, folds),
    }

    evidence.mkdir(parents=True, exist_ok=True)
    for name, payload in (("feed.json", feed_row), ("model.json", model_row), ("eval.json", eval_row)):
        (evidence / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return {"feed": feed_row, "model": model_row, "eval": eval_row}


def handles(evidence: Path) -> dict:
    """The real harness handles a checker reads live state through."""
    return {
        "feed": evidence / "feed.json",
        "model": evidence / "model.json",
        "eval": evidence / "eval.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission")
    parser.add_argument("--plan")
    parser.add_argument("--corpus", default=str(BUNDLE / "environment" / "corpus_spec.json"))
    parser.add_argument("--split", default=str(HERE / "heldout_spec.json"))
    parser.add_argument("--readout", default=str(BUNDLE / "environment" / "graded_readout.json"))
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if args.submission:
        plan = run_submission(Path(args.submission), corpus_path)["plan"]
    elif args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    else:
        parser.error("give --submission or --plan")
    if plan is None:
        raise SystemExit("submission produced no plan.json")
    evaluate(corpus_path, Path(args.split), Path(args.readout), plan, Path(args.evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
