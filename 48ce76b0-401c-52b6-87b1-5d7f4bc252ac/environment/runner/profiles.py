"""Scale profiles for the S03 data-order task.

There are exactly two profiles and they drive the identical code path. ``graded`` is the
scaled operating point a graded attempt runs on one H100. ``smoke`` is the same pipeline
shrunk to a CPU-only size so the whole path can be executed without an accelerator.

Nothing here is a stub. Every module in this package reads its shape from the selected
profile and runs the same functions in both profiles.
"""

from __future__ import annotations

import hashlib
import json
import os

# Frozen optimizer contract. This is the axis the task freezes, so it is identical in both
# profiles apart from the warmup length, which is expressed as a fraction of total steps.
FROZEN_OPTIMIZER = {
    "name": "adamw",
    "lr": 3.0e-3,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1.0e-8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_fraction": 0.1111111111111111,
    "final_lr_fraction": 0.1,
    "schedule": "cosine",
}

PROFILES = {
    "graded": {
        "profile": "graded",
        "n_layer": 6,
        "d_model": 384,
        "n_head": 6,
        "seq_len": 512,
        "vocab_size": 2048,
        "batch_sequences": 16,
        "n_steps": 500,
        "n_train_sequences": 8000,
        "n_val_sequences": 512,
        "n_domains": 8,
        "corpus_seed": 20260821,
        "device": "cuda",
        "dtype": "bfloat16",
        "graded_seeds": [0, 1, 2],
        "comparator_draws": 2,
    },
    "smoke": {
        "profile": "smoke",
        "n_layer": 2,
        "d_model": 64,
        "n_head": 4,
        "seq_len": 32,
        "vocab_size": 256,
        "batch_sequences": 4,
        "n_steps": 12,
        "n_train_sequences": 48,
        "n_val_sequences": 16,
        "n_domains": 4,
        "corpus_seed": 20260821,
        "device": "cpu",
        "dtype": "float32",
        "graded_seeds": [0, 1, 2],
        "comparator_draws": 2,
    },
}

# TARGET_DELTA_NATS is an authored bar and not a measurement. See the coverage gap named
# anchors-uncalibrated in solution/grounding.yaml. SIGNIFICANCE_T is a threshold on a statistic
# whose scale is estimated in band from the comparator ensemble, so the noise floor itself is
# measured during the attempt rather than authored.
TARGET_DELTA_NATS = 0.05
SIGNIFICANCE_T = 2.0


def active_profile_name() -> str:
    if os.environ.get("BIA_SMOKE") == "1":
        return "smoke"
    return os.environ.get("BIA_PROFILE", "graded")


def get_profile(name: str | None = None) -> dict:
    key = name or active_profile_name()
    if key not in PROFILES:
        raise SystemExit(f"unknown_profile_{key}")
    cfg = dict(PROFILES[key])
    if cfg["batch_sequences"] * cfg["n_steps"] != cfg["n_train_sequences"]:
        raise SystemExit(f"profile_inconsistent_{key}_token_budget")
    cfg["total_train_tokens"] = cfg["n_train_sequences"] * cfg["seq_len"]
    cfg["warmup_steps"] = max(1, int(round(cfg["n_steps"] * FROZEN_OPTIMIZER["warmup_fraction"])))
    cfg["optimizer"] = dict(FROZEN_OPTIMIZER)
    return cfg


def frozen_substrate_fields(cfg: dict) -> dict:
    """The exact fields the task freezes. The agent owns none of them."""
    return {
        "profile": cfg["profile"],
        "n_layer": cfg["n_layer"],
        "d_model": cfg["d_model"],
        "n_head": cfg["n_head"],
        "seq_len": cfg["seq_len"],
        "vocab_size": cfg["vocab_size"],
        "batch_sequences": cfg["batch_sequences"],
        "n_steps": cfg["n_steps"],
        "n_train_sequences": cfg["n_train_sequences"],
        "n_val_sequences": cfg["n_val_sequences"],
        "n_domains": cfg["n_domains"],
        "corpus_seed": cfg["corpus_seed"],
        "total_train_tokens": cfg["total_train_tokens"],
        "graded_seeds": list(cfg["graded_seeds"]),
        "comparator_draws": cfg["comparator_draws"],
        "warmup_steps": cfg["warmup_steps"],
        "deterministic_kernels": True,
        "optimizer": dict(FROZEN_OPTIMIZER),
    }


def substrate_digest(cfg: dict) -> str:
    payload = json.dumps(frozen_substrate_fields(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
