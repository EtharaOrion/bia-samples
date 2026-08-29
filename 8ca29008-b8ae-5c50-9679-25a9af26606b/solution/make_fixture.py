"""PRIVATE. Deterministic generator for the S08 mid-training checkpoint fixture.

This script is the sole author of `environment/fixtures/ckpt_s08_<profile>.pt` and
its manifest. It is private because it names the poison. A grader holding this tree
reproduces the shipped blob from a pinned seed and a pinned thread count rather than
trusting the bytes, and `solution/recompute.py --verify-fixture` performs exactly
that comparison.

Procedure:

  1. Train the frozen model from the frozen deterministic init with a clean AdamW
     for `pretrain_steps`. Record the validation curve. This curve is healthy and
     it is what ships in the agent-visible manifest.
  2. Snapshot the weights and the clean optimizer state.
  3. Apply the poison transform to the optimizer state only. The weights are never
     touched, so the checkpoint's loss is exactly the loss the healthy curve ends on
     and no read of the curve can reveal the damage.
  4. Compute the target loss from two measured quantities, the checkpoint validation
     loss and the exact Bayes-optimal cross entropy of the validation shard, so that
     no hand-picked loss constant enters the bundle.

The poison transform, all four components:

  P1  every two-dimensional weight except the per-block MLP output projection has
      its Adam second moment multiplied by 1e4, which divides its effective learning
      rate by 100. The result is a slow stall rather than a divergence.
  P2  the per-block MLP output projection has its second moment multiplied by 1e-4,
      which multiplies its effective learning rate by 100 and pushes noise straight
      into the residual stream. A single global rescale that repairs P1 therefore
      makes P2 worse by eight more orders of magnitude, so the repair has to be
      per tensor.
  P3  the recorded optimizer step is set to 1e6 while the true step is
      `pretrain_steps`. The manifest reports the true step, so the inconsistency is
      discoverable by reading, and it is invisible to anything that resumes blindly.
  P4  the stored `param_groups` carry a weight decay of 0.12 where the pretraining
      run used 0.01. Any resume that loads the checkpoint's hyperparameters along
      with its state inherits a decay twelve times too strong.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
RUNNER = os.path.join(BUNDLE, "environment", "runner")
if RUNNER not in sys.path:
    sys.path.insert(0, RUNNER)

import s08_core as core  # noqa: E402

GENERATOR_VERSION = "s08-fixture-1"

AMPLIFIED_SUFFIX = "fc2.weight"

POISON = {
    "p1_v_scale_throttled_matrices": 1.0e4,
    "p2_v_scale_amplified_matrices": 1.0e-4,
    "p2_amplified_suffix": AMPLIFIED_SUFFIX,
    "p3_recorded_step": 1.0e6,
    "p4_param_group_weight_decay": 0.12,
}


def poison_optimizer_state(state: dict, param_names: list) -> dict:
    out = copy.deepcopy(state)
    for idx_key, st in out["state"].items():
        idx = int(idx_key)
        name = param_names[idx]
        if "exp_avg_sq" in st and torch.is_tensor(st["exp_avg_sq"]):
            if st["exp_avg_sq"].dim() >= 2:
                if name.endswith(AMPLIFIED_SUFFIX):
                    st["exp_avg_sq"] = st["exp_avg_sq"] * POISON["p2_v_scale_amplified_matrices"]
                else:
                    st["exp_avg_sq"] = st["exp_avg_sq"] * POISON["p1_v_scale_throttled_matrices"]
        if "step" in st:
            if torch.is_tensor(st["step"]):
                st["step"] = torch.full_like(st["step"], POISON["p3_recorded_step"])
            else:
                st["step"] = POISON["p3_recorded_step"]
    for g in out["param_groups"]:
        g["weight_decay"] = POISON["p4_param_group_weight_decay"]
    return out


def pretrain(cfg: core.Cfg, data_dir: str, device):
    train, val, meta = core.load_corpus(cfg, data_dir)
    model = core.build_model(cfg, device)
    opt = torch.optim.AdamW(
        list(model.parameters()),
        lr=cfg.pretrain_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.pretrain_wd,
    )
    curve = []
    model.train()
    for i in range(1, cfg.pretrain_steps + 1):
        warm = min(1.0, i / max(1, cfg.pretrain_warmup))
        lr = cfg.pretrain_lr * warm
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = core.get_batch(train, cfg, step=100000 + i, seed=7, device=device)
        opt.zero_grad(set_to_none=True)
        loss = core.lm_loss(model(x), y)
        loss.backward()
        opt.step()
        if i % cfg.eval_every == 0 or i == cfg.pretrain_steps:
            vl = core.evaluate(model, val, cfg, device)
            curve.append({"step": i, "val_loss": round(vl, 6), "train_loss": round(float(loss.detach()), 6)})
            print(f"  pretrain step {i:5d}  val {vl:.5f}", flush=True)
    for g in opt.param_groups:
        g["lr"] = cfg.pretrain_lr
    return model, opt, curve, val, meta


def build(profile: str, out_dir: str, data_dir: str, threads: int) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(20260822)
    np.random.seed(20260822)
    device = torch.device("cpu")
    cfg = core.load_cfg(profile)
    os.makedirs(out_dir, exist_ok=True)

    model, opt, curve, val, meta = pretrain(cfg, data_dir, device)
    ckpt_val = core.evaluate(model, val, cfg, device)
    bayes = float(meta["bayes_loss_nats"])
    target_loss = ckpt_val - cfg.target_fraction * (ckpt_val - bayes)

    param_names = [n for n, _ in model.named_parameters()]
    clean_state = copy.deepcopy(opt.state_dict())
    poisoned = poison_optimizer_state(clean_state, param_names)

    weights = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()}
    fixture = {
        "core_version": core.CORE_VERSION,
        "generator_version": GENERATOR_VERSION,
        "profile": cfg.profile,
        "checkpoint_step": cfg.pretrain_steps,
        "param_names": param_names,
        "model": weights,
        "optimizer": poisoned,
        "weight_digest": core.weight_digest(model),
    }
    fx_path = os.path.join(out_dir, f"ckpt_s08_{cfg.profile}.pt")
    torch.save(fixture, fx_path, _use_new_zipfile_serialization=True)

    manifest = {
        "core_version": core.CORE_VERSION,
        "generator_version": GENERATOR_VERSION,
        "profile": cfg.profile,
        "checkpoint_step": cfg.pretrain_steps,
        "checkpoint_reported_lr": cfg.pretrain_lr,
        "checkpoint_val_loss": round(ckpt_val, 6),
        "source_entropy_nats": round(bayes, 6),
        "target_loss": round(target_loss, 6),
        "target_rule": "checkpoint_val_loss - target_fraction * (checkpoint_val_loss - source_entropy_nats)",
        "target_fraction": cfg.target_fraction,
        "target_fraction_of_control": cfg.full_reward_fraction,
        "pretrain_curve": curve,
        "frozen": core.frozen_facts(cfg),
        "max_steps": cfg.max_steps,
        "eval_every": cfg.eval_every,
        "seeds": list(cfg.seeds),
        "probe_batches_max": cfg.probe_batches_max,
        "weight_digest": fixture["weight_digest"],
        "param_names": param_names,
        "corpus": {
            "train_sha256": meta["train_sha256"],
            "val_sha256": meta["val_sha256"],
            "train_shape": meta["train_shape"],
            "val_shape": meta["val_shape"],
        },
        "fixture_sha256": core.sha256_file(fx_path),
        "fixture_bytes": os.path.getsize(fx_path),
        "generator_threads": threads,
        "torch_version": torch.__version__,
    }
    mf_path = os.path.join(out_dir, f"manifest_{cfg.profile}.json")
    with open(mf_path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(json.dumps({"fixture": fx_path, "bytes": manifest["fixture_bytes"], "sha256": manifest["fixture_sha256"]}))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="scaled")
    ap.add_argument("--out", default=os.path.join(BUNDLE, "environment", "fixtures"))
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("S08_GEN_THREADS", "8")))
    args = ap.parse_args(argv)
    data_dir = args.data_dir or os.path.join(
        os.environ.get("S08_STATE_DIR", "/opt/bia/s08"), "data", args.profile
    )
    build(args.profile, args.out, data_dir, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
