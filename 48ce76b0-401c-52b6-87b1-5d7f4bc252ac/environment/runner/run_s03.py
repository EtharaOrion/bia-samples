"""Runner for the S03 data-order task. This is the only writer of the harness telemetry.

One invocation trains one arm of one seed under one ordering and appends its records to the
run record. The graded attempt runs the baseline arm and the submitted arm on every graded
seed, and the grader reads only what this runner recorded.

The identical code path serves both scale profiles. BIA_SMOKE=1 selects the CPU-only smoke
profile, which shrinks every dimension and touches no accelerator.

Kernel determinism is part of the frozen substrate rather than a convenience. Measured on the
graded profile, the default nondeterministic accumulation in the tied embedding backward and in
the fused attention backward moved the final validation loss of one fixed seed under one fixed
order by about 0.018 nats between repeats, which is the same size as the whole ordering effect
this task grades. Under that noise the comparator ensemble no longer measures the ordering, and
a reported score is not reproducible on re-measurement. The enforcement is applied inside
train_one, which runs after the submitted ordering module has already been imported, so a
submission that flips the flags back on import cannot reintroduce the lottery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Set before any CUDA context exists, because cuBLAS reads it when it first initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

import corpus as corpus_mod
import order as order_mod
import profiles as profiles_mod
import telemetry as telemetry_mod


def cache_dir() -> str:
    return os.environ.get("BIA_CACHE_DIR", "/tmp/bia_s03_cache")


def load_or_build_corpus(cfg: dict):
    key = f"{cfg['profile']}_{cfg['corpus_seed']}_{cfg['n_train_sequences']}_{cfg['seq_len']}_{cfg['vocab_size']}_{cfg['n_domains']}"
    path = os.path.join(cache_dir(), key + ".npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return z["train"], z["domains"], z["val"]
    train, domains, val = corpus_mod.generate_corpus(cfg)
    os.makedirs(cache_dir(), exist_ok=True)
    tmp = path + ".partial.npz"
    np.savez(tmp, train=train, domains=domains, val=val)
    os.replace(tmp, path)
    return train, domains, val


def load_or_build_features(cfg: dict, train, domains):
    key = f"feat_{cfg['profile']}_{cfg['corpus_seed']}_{cfg['n_train_sequences']}"
    path = os.path.join(cache_dir(), key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    feats = corpus_mod.sequence_features(train, domains, cfg)
    os.makedirs(cache_dir(), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(feats, f)
    os.replace(tmp, path)
    return feats


def enforce_determinism():
    """Bind the accumulation order of every kernel the scored run touches.

    Called from train_one and therefore after the submitted ordering module has been imported
    and executed, so this is the last writer of these flags and a submission cannot undo it.
    """
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_device(cfg: dict):
    import torch

    want = cfg["device"]
    if want == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("graded_profile_requires_cuda_device")
    return torch.device("cuda")


def train_one(cfg: dict, order, seed: int, arm: str, rec: telemetry_mod.RunRecord, draw: int = 0) -> float:
    import torch

    enforce_determinism()
    device = resolve_device(cfg)
    if device.type == "cpu" and os.environ.get("BIA_TORCH_THREADS"):
        torch.set_num_threads(max(1, int(os.environ["BIA_TORCH_THREADS"])))
    train, domains, val = load_or_build_corpus(cfg)

    torch.manual_seed(seed)
    from model import TinyGPT, build_frozen_optimizer, frozen_lr_at

    model = TinyGPT(cfg).to(device)
    opt = build_frozen_optimizer(model, cfg)

    use_amp = device.type == "cuda" and cfg["dtype"] == "bfloat16"
    train_t = torch.from_numpy(train.astype(np.int64))
    val_t = torch.from_numpy(val.astype(np.int64)).to(device)

    substrate = profiles_mod.substrate_digest(cfg)
    run_id = f"{arm}_seed{seed}_draw{draw}" if arm == "baseline" else f"{arm}_seed{seed}"
    rec.emit(
        "order_frozen",
        run_id=run_id,
        arm=arm,
        seed=int(seed),
        draw=int(draw),
        order_digest=order_mod.order_digest(order),
        first_index_sequence_digest=telemetry_mod.digest_json(order_mod.first_index_sequence(order)),
        substrate_digest=substrate,
        profile=cfg["profile"],
    )

    consumed = 0
    consumed_indices: list[int] = []
    model.train()
    for step in range(cfg["n_steps"]):
        idx = order[step]
        consumed_indices.extend(int(v) for v in idx)
        batch = train_t[idx].to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        lr = frozen_lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["grad_clip"])
        opt.step()
        consumed += len(idx) * cfg["seq_len"]
        rec.emit(
            "train_step",
            run_id=run_id,
            arm=arm,
            seed=int(seed),
            draw=int(draw),
            step=step,
            first_index=int(idx[0]),
            batch_size=len(idx),
            forward_backward_per_step=1,
            tokens_consumed=consumed,
            lr=round(float(lr), 10),
            train_loss=round(float(loss.detach().float().item()), 6),
            substrate_digest=substrate,
        )

    model.eval()
    total, nb = 0.0, 0
    with torch.no_grad():
        for i in range(0, val_t.shape[0], cfg["batch_sequences"]):
            chunk = val_t[i : i + cfg["batch_sequences"]]
            if chunk.shape[0] == 0:
                continue
            x, y = chunk[:, :-1], chunk[:, 1:]
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            total += float(loss.float().item())
            nb += 1
    final_val = total / max(1, nb)

    rec.emit(
        "run_complete",
        run_id=run_id,
        arm=arm,
        seed=int(seed),
        draw=int(draw),
        final_val_loss=round(final_val, 6),
        tokens_consumed=consumed,
        consumed_total=len(consumed_indices),
        consumed_unique_count=len(set(consumed_indices)),
        consumed_sorted_digest=telemetry_mod.digest_json(sorted(consumed_indices)),
        total_train_tokens=cfg["total_train_tokens"],
        substrate_digest=substrate,
        profile=cfg["profile"],
    )
    return final_val


def cmd_order_digest(args, cfg):
    train, domains, _ = load_or_build_corpus(cfg)
    feats = load_or_build_features(cfg, train, domains)
    meta = order_mod.build_meta(cfg, feats)
    order = order_mod.call_build_order(args.order_module, meta)
    ok, reason = order_mod.validate_order(order, cfg)
    print(json.dumps({"valid": ok, "reason": reason, "order_digest": order_mod.order_digest(order)}, sort_keys=True))
    return 0 if ok else 1


def cmd_run(args, cfg):
    telemetry_path = os.environ.get("BIA_TELEMETRY", "/telemetry/run_record.jsonl")
    egress_path = os.environ.get("BIA_EGRESS_AUDIT", os.path.join(os.path.dirname(telemetry_path) or ".", "egress_audit.jsonl"))

    train, domains, _ = load_or_build_corpus(cfg)
    feats = load_or_build_features(cfg, train, domains)
    meta = order_mod.build_meta(cfg, feats)

    if args.arm == "baseline":
        order = order_mod.baseline_order(cfg, args.seed, args.draw)
    else:
        order = order_mod.call_build_order(args.order_module, meta)

    ok, reason = order_mod.validate_order(order, cfg)
    if not ok:
        print(json.dumps({"error": reason}))
        return 2

    telemetry_mod.install_egress_audit(egress_path)
    rec = telemetry_mod.RunRecord(telemetry_path)
    loss = train_one(cfg, order, args.seed, args.arm, rec, args.draw)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "draw": args.draw, "final_val_loss": round(loss, 6)}, sort_keys=True))
    return 0


def cmd_grid(args, cfg):
    print(" ".join(str(s) for s in cfg["graded_seeds"]) + "|" + str(cfg["comparator_draws"]))
    return 0


def cmd_manifest(args, cfg):
    train, domains, val = load_or_build_corpus(cfg)
    payload = {
        "profile": cfg["profile"],
        "substrate_digest": profiles_mod.substrate_digest(cfg),
        "frozen": profiles_mod.frozen_substrate_fields(cfg),
        "corpus": corpus_mod.corpus_digests(train, domains, val),
    }
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="S03 data-order runner")
    p.add_argument("command", choices=["run", "order-digest", "manifest", "grid"])
    p.add_argument("--arm", choices=["baseline", "submitted"], default="submitted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--draw", type=int, default=0)
    p.add_argument("--order-module", default=os.environ.get("BIA_ORDER_MODULE", "/workspace/submission/order.py"))
    args = p.parse_args(argv)

    cfg = profiles_mod.get_profile()
    if args.command == "run":
        return cmd_run(args, cfg)
    if args.command == "order-digest":
        return cmd_order_digest(args, cfg)
    if args.command == "grid":
        return cmd_grid(args, cfg)
    return cmd_manifest(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
