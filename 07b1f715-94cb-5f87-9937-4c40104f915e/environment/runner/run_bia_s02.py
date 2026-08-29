"""BIA S02 training runner. Harness-owned. The only writer of run telemetry.

Usage:
    python3 runner/run_bia_s02.py --update-rule submission/update_rule.py \
        --seed 0 --profile full --out-dir submission

What this runner owns and you do not: the dataset and its order, the batch
shape, the architecture, the initialization, the number of forward-backward
passes per step, the validation path, and the learning rate applied at every
step of every parameter group. That learning rate comes from
environment/frozen_schedule.py and from nowhere else.

What you own: the update rule module named by --update-rule. It exposes

    build_update_rule(param_groups) -> torch.optim.Optimizer

`param_groups` is the frozen group list, each entry a dict carrying "name",
"params" and a placeholder "lr". The runner overwrites "lr" on every group
before every step. Your rule must consume the value it is given. It is never
told how many steps the run has, so a rule cannot reimplement the schedule it
is not allowed to change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_ROOT = os.path.dirname(HERE)
for _p in (HERE, ENV_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as data_mod  # noqa: E402
import frozen_schedule as sched  # noqa: E402
import model as model_mod  # noqa: E402
import telemetry as telemetry_mod  # noqa: E402

PROFILES = {
    "full": {
        "n_layer": 6,
        "n_head": 6,
        "d_model": 384,
        "seq_len": 512,
        "batch_sequences": 48,
        "total_steps": 1400,
        "eval_every": 25,
        "val_batches": 8,
        "device": "cuda",
        "dtype": "bfloat16",
        "wallclock_guard_s": 420.0,
    },
    "smoke": {
        "n_layer": 2,
        "n_head": 2,
        "d_model": 64,
        "seq_len": 64,
        "batch_sequences": 4,
        "total_steps": 36,
        "eval_every": 4,
        "val_batches": 2,
        "device": "cpu",
        "dtype": "float32",
        "wallclock_guard_s": 45.0,
    },
}

FROZEN_CONTRACT = {
    "forward_backward_per_step": 1,
    "data_order": "sequential_frozen",
    "init_owner": "harness",
    "eval_owner": "harness",
    "schedule_owner": "harness",
}


def file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def param_probe(params) -> str:
    """Cheap strided fingerprint of live parameter state, one device sync."""
    parts = []
    for p in params:
        flat = p.detach().reshape(-1)
        stride = max(1, flat.numel() // 16)
        parts.append(flat[::stride][:16].to(torch.float32))
    blob = torch.cat(parts).cpu().numpy().tobytes()
    return hashlib.sha256(blob).hexdigest()


def load_update_rule(path: str):
    spec = importlib.util.spec_from_file_location("bia_update_rule", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "build_update_rule", None)
    if not callable(fn):
        raise SystemExit("update rule at %s does not expose build_update_rule" % path)
    return fn


@torch.no_grad()
def evaluate(net, loader, batches: int, device, autocast_ctx) -> float:
    net.eval()
    loader.reset()
    total = 0.0
    for _ in range(batches):
        x, y = loader.next_batch(device)
        with autocast_ctx():
            total += float(net(x, y))
    net.train()
    return total / batches


def autocast_factory(device: str, dtype: str):
    import contextlib
    if device == "cuda" and dtype == "bfloat16":
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-rule", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--profile", default="full", choices=sorted(PROFILES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("BIA_DATA_DIR", "/data/fineweb"))
    ap.add_argument("--fixture-dir", default=os.path.join(ENV_ROOT, "fixtures"))
    args = ap.parse_args()

    if args.profile == "smoke" and os.environ.get("BIA_SMOKE") != "1":
        raise SystemExit("profile smoke requires BIA_SMOKE=1 in the runner environment")

    cfg = PROFILES[args.profile]
    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("profile full requires a CUDA device")
    if device == "cpu":
        # Tiny matrices oversubscribe an OpenMP pool badly, so the CPU proof path
        # pins its thread count. This is a runner setting and never a task axis.
        torch.set_num_threads(int(os.environ.get("BIA_CPU_THREADS", "1")))
    torch.manual_seed(args.seed)

    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(args.out_dir, "logs", "%s_seed%d.log" % (args.profile, args.seed))
    telemetry_dir = os.environ.get("BIA_TELEMETRY_DIR", "/telemetry")
    telemetry_path = os.path.join(telemetry_dir, "run_record.jsonl")

    train_path, train_digest, val_path, val_digest = data_mod.resolve_shards(
        args.profile, args.data_dir, args.fixture_dir)
    train_loader = data_mod.FrozenLoader(train_path, cfg["batch_sequences"], cfg["seq_len"], train_digest)
    val_loader = data_mod.FrozenLoader(val_path, cfg["batch_sequences"], cfg["seq_len"], val_digest)

    net = model_mod.FrozenGPT(cfg["n_layer"], cfg["n_head"], cfg["d_model"], cfg["seq_len"])
    model_mod.frozen_init(net, args.seed)
    net.to(device)
    net.train()
    groups = model_mod.build_param_groups(net)
    all_params = [p for g in groups for p in g["params"]]
    probe_params = all_params[:6]

    build = load_update_rule(args.update_rule)
    opt = build(groups)
    if not isinstance(opt, torch.optim.Optimizer):
        raise SystemExit("build_update_rule returned %s, not a torch.optim.Optimizer" % type(opt).__name__)

    autocast_ctx = autocast_factory(device, cfg["dtype"])
    arch = model_mod.architecture_signature(cfg["n_layer"], cfg["n_head"], cfg["d_model"], cfg["seq_len"])
    static = {
        "schema": "bia.s02.telemetry/v1",
        "profile": args.profile,
        "seed": args.seed,
        "total_steps": cfg["total_steps"],
        "batch_sequences": cfg["batch_sequences"],
        "batch_tokens": cfg["batch_sequences"] * cfg["seq_len"],
        "architecture": arch,
        "frozen": dict(FROZEN_CONTRACT),
        "schedule_digest": file_digest(os.path.join(ENV_ROOT, "frozen_schedule.py")),
        "model_digest": file_digest(os.path.join(HERE, "model.py")),
        "runner_digest": file_digest(os.path.abspath(__file__)),
        "update_rule_digest": file_digest(args.update_rule),
        "train_shard_digest": train_digest,
        "val_shard_digest": val_digest,
    }

    writer = telemetry_mod.ChainWriter(telemetry_path, os.environ.get("BIA_CHAIN_KEY", ""))
    fwd_bwd_total = 0
    lr_write_violations = 0
    out_of_step_mutations = 0
    steps_checked = 0
    record_index = 0
    guard = float(os.environ.get("BIA_MAX_SECONDS", cfg["wallclock_guard_s"]))
    started = time.time()
    stop_reason = "completed"
    probe_after_prev_step = param_probe(probe_params)

    log_lines = []
    for step in range(1, cfg["total_steps"] + 1):
        probe_before_step = param_probe(probe_params)
        if probe_before_step != probe_after_prev_step:
            out_of_step_mutations += 1

        applied = {}
        for g in opt.param_groups:
            name = g.get("name")
            if name is None:
                raise SystemExit("update rule dropped the frozen group name; groups are harness-owned")
            g["lr"] = sched.group_lr(name, step, cfg["total_steps"])
            applied[name] = g["lr"]

        x, y = train_loader.next_batch(device)
        with autocast_ctx():
            loss = net(x, y)
        loss.backward()
        fwd_bwd_total += 1
        opt.step()
        # Probed immediately after step() returns, so the window this fingerprint
        # closes covers zero_grad, the validation pass and the next data fetch.
        probe_after_prev_step = param_probe(probe_params)
        opt.zero_grad(set_to_none=True)
        steps_checked += 1

        for g in opt.param_groups:
            if float(g["lr"]) != float(applied[g["name"]]):
                lr_write_violations += 1

        elapsed = time.time() - started
        due = (step % cfg["eval_every"] == 0) or step == cfg["total_steps"] or step == 1
        budget_hit = elapsed > guard
        if budget_hit:
            stop_reason = "wallclock_guard"
        if due or budget_hit:
            val_loss = evaluate(net, val_loader, cfg["val_batches"], device, autocast_ctx)
            record_index += 1
            body = dict(static)
            body.update({
                "record_index": record_index,
                "step": step,
                "val_loss": round(float(val_loss), 6),
                "train_loss": round(float(loss.detach()), 6),
                "lr_applied": {k: round(v, 12) for k, v in applied.items()},
                "forward_backward_this_step": 1,
                "forward_backward_cumulative": fwd_bwd_total,
                "steps_checked": steps_checked,
                "lr_write_violations": lr_write_violations,
                "out_of_step_mutations": out_of_step_mutations,
                "elapsed_s": round(elapsed, 3),
                "stop_reason": stop_reason,
            })
            writer.append(body)
            line = "step: %d / %d val_loss: %.6f" % (step, cfg["total_steps"], val_loss)
            log_lines.append(line)
            print(line, flush=True)
        if budget_hit:
            print("wallclock guard hit at %.1fs, stopping" % elapsed, flush=True)
            break

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    summary = {
        "profile": args.profile,
        "seed": args.seed,
        "steps_run": steps_checked,
        "stop_reason": stop_reason,
        "telemetry": telemetry_path,
        "log": log_path,
        "elapsed_s": round(time.time() - started, 3),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
