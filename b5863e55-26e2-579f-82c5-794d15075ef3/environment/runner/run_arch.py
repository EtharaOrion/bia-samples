#!/usr/bin/env python3
"""The only writer of the BIA S04 telemetry record.

One invocation trains both arms of a graded attempt: the shipped reference
allocation and the submitted allocation, on the identical corpus, the identical
data order, the identical seed, the identical optimizer and the identical step
count. The two arms differ in exactly one thing, which is how the frozen
parameter budget is spread over depth, width and attention heads.

Logs written by hand are not evidence. Every record this file appends is linked
into an HMAC chain, and the verifier recomputes that chain before it reads a
single number out of the log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bia_arch as B


def pick_device(profile_device: str) -> str:
    if os.environ.get("BIA_FORCE_CPU") == "1":
        return "cpu"
    if profile_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate(model, tokens, starts, seq_len, device, cos, sin, use_amp) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for row in starts:
            x, y = B.gather(tokens, row, seq_len)
            xb = torch.from_numpy(x.astype(np.int64)).to(device)
            yb = torch.from_numpy(y.astype(np.int64)).to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(xb, cos, sin)
                    loss = torch.nn.functional.cross_entropy(
                        logits.float().view(-1, model.vocab_size), yb.reshape(-1)
                    )
            else:
                logits = model(xb, cos, sin)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, model.vocab_size), yb.reshape(-1)
                )
            total += float(loss.item())
            count += 1
    model.train()
    return total / max(1, count)


def run_arm(arm, arch, recipe, profile, split, tr_starts, va_starts, device, chain, mode, deadline_at):
    seq_len = profile["seq_len"]
    train_tokens = np.frombuffer(split["train"], dtype=np.uint8)
    val_tokens = np.frombuffer(split["val"], dtype=np.uint8)

    torch.manual_seed(profile["seed"])
    model = B.TinyLM(arch, recipe["vocab_size"]).to(device)
    B.init_model(model, recipe["init"]["std"], profile["seed"])
    live = B.live_param_count(model)

    chain.append({
        "event": "arm_start",
        "mode": mode,
        "arm": arm,
        "step": 0,
        "arch": {k: int(arch[k]) for k in B.ARCH_KEYS},
        "arch_digest": B.arch_digest(arch),
        "param_count_total": live,
        "param_count_source": "live_module_sum",
        "device": device,
    })

    opt = B.build_optimizer(model, recipe)
    peak = recipe["optimizer"]["lr"]
    clip = recipe["optimizer"]["grad_clip"]
    use_amp = device == "cuda"
    cos, sin = B.rope_tables(arch["head_dim"], seq_len, device)
    eval_steps = set(profile["eval_steps"])
    evals = {}

    model.train()
    for step in range(profile["steps"]):
        for group in opt.param_groups:
            group["lr"] = B.lr_at(step, profile, peak)
        x, y = B.gather(train_tokens, tr_starts[step], seq_len)
        xb = torch.from_numpy(x.astype(np.int64)).to(device)
        yb = torch.from_numpy(y.astype(np.int64)).to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(xb, cos, sin)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, model.vocab_size), yb.reshape(-1)
                )
        else:
            logits = model(xb, cos, sin)
            loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, model.vocab_size), yb.reshape(-1)
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        done = step + 1
        if done in eval_steps:
            val = evaluate(model, val_tokens, va_starts, seq_len, device, cos, sin, use_amp)
            evals[done] = val
            chain.append({
                "event": "eval",
                "mode": mode,
                "arm": arm,
                "step": done,
                "val_loss": round(val, 8),
                "train_loss_at_step": round(float(loss.item()), 8),
            })
        if time.time() > deadline_at:
            chain.append({
                "event": "deadline_exceeded",
                "mode": mode,
                "arm": arm,
                "step": done,
                "deadline_seconds": profile["deadline_seconds"],
            })
            return None

    chain.append({
        "event": "arm_end",
        "mode": mode,
        "arm": arm,
        "step": profile["steps"],
        "param_count_total": live,
        "evals": {str(k): round(v, 8) for k, v in sorted(evals.items())},
    })
    return evals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--submission", default="/workspace/submission/arch.json")
    ap.add_argument("--telemetry", default=None)
    ap.add_argument("--frozen-dir", default=None)
    args = ap.parse_args()

    frozen_dir = args.frozen_dir or B.FROZEN_DIR
    recipe = B.frozen_recipe(os.path.join(frozen_dir, "frozen_recipe.json"))
    space = B.arch_space(os.path.join(frozen_dir, "arch_space.json"))
    reference = B.load_json(os.path.join(frozen_dir, "reference_arch.json"))
    profile = recipe["profiles"][args.profile]

    telemetry = args.telemetry or os.environ.get(
        "BIA_TELEMETRY", os.path.join(os.environ.get("BIA_TELEMETRY_DIR", "/telemetry"), "run_record.jsonl")
    )
    key = os.environ.get("BIA_CHAIN_KEY")
    if not key:
        print("runner_fault: BIA_CHAIN_KEY is unset, refusing to write an unsigned record", file=sys.stderr)
        return 3

    if not os.path.exists(args.submission):
        print("runner_fault: no submission at " + args.submission, file=sys.stderr)
        return 4
    with open(args.submission, "rb") as handle:
        submission_bytes = handle.read()
    submitted = json.loads(submission_bytes.decode("utf-8"))
    if args.profile in submitted:
        submitted = submitted[args.profile]
    submitted = {k: submitted[k] for k in B.ARCH_KEYS if k in submitted}

    problems = B.validate_arch(submitted, space, args.profile)
    ref_arch = {k: int(reference[args.profile][k]) for k in B.ARCH_KEYS}

    device = pick_device(profile["device"])
    started = time.time()
    deadline_at = started + profile["deadline_seconds"]

    data = B.build_corpus(recipe, profile["corpus_bytes"])
    split = B.split_corpus(data, profile["val_frac"])
    tr_starts = B.train_starts(
        len(split["train"]), profile["seq_len"], profile["batch_sequences"], profile["steps"], profile["seed"]
    )
    va_starts = B.val_starts(
        len(split["val"]), profile["seq_len"], profile["batch_sequences"], profile["eval_batches"], profile["seed"]
    )

    chain = B.Chain(key, telemetry)
    chain.append({
        "event": "header",
        "mode": args.profile,
        "arm": "none",
        "step": -1,
        "recipe_version": recipe["recipe_version"],
        "space_version": space["space_version"],
        "frozen": {
            "seq_len": profile["seq_len"],
            "batch_sequences": profile["batch_sequences"],
            "steps": profile["steps"],
            "token_budget": profile["seq_len"] * profile["batch_sequences"] * profile["steps"],
            "seed": profile["seed"],
            "vocab_size": recipe["vocab_size"],
            "optimizer_signature": B.canonical_json(recipe["optimizer"]),
            "schedule_digest": B.schedule_digest(profile, recipe),
            "param_budget": profile["param_budget"],
            "param_tolerance": profile["param_tolerance"],
        },
        "split": {
            "n_bytes": split["n_bytes"],
            "train_range": split["train_range"],
            "val_range": split["val_range"],
            "train_digest": split["train_digest"],
            "val_digest": split["val_digest"],
        },
        "order": {
            "train_order_digest": B.order_digest(tr_starts),
            "val_order_digest": B.order_digest(va_starts),
            "max_train_start": int(tr_starts.max()),
        },
        "frozen_tree_digest": B.frozen_tree_digest(frozen_dir),
        "submission_digest": B.sha256_hex(submission_bytes),
        "submission_space_problems": problems,
    })

    if problems:
        chain.append({
            "event": "final",
            "mode": args.profile,
            "arm": "none",
            "step": 0,
            "status": "submission_outside_search_space",
            "elapsed_seconds": round(time.time() - started, 3),
        })
        print(json.dumps({"status": "submission_outside_search_space", "problems": problems}))
        return 5

    ref_evals = run_arm(
        "reference", ref_arch, recipe, profile, split, tr_starts, va_starts, device, chain, args.profile, deadline_at
    )
    sub_evals = None
    if ref_evals is not None:
        sub_evals = run_arm(
            "submission", submitted, recipe, profile, split, tr_starts, va_starts, device, chain, args.profile, deadline_at
        )

    status = "complete" if (ref_evals is not None and sub_evals is not None) else "deadline_exceeded"
    chain.append({
        "event": "final",
        "mode": args.profile,
        "arm": "none",
        "step": profile["steps"],
        "status": status,
        "elapsed_seconds": round(time.time() - started, 3),
    })
    print(json.dumps({
        "status": status,
        "device": device,
        "reference_evals": ref_evals,
        "submission_evals": sub_evals,
        "elapsed_seconds": round(time.time() - started, 3),
    }, sort_keys=True))
    return 0 if status == "complete" else 6


if __name__ == "__main__":
    raise SystemExit(main())
