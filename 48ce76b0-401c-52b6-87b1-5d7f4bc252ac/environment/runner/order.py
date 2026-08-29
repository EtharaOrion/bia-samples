"""Ordering interface for the S03 data-order task.

The submitted policy owns exactly one thing: the permutation of the frozen training
sequences into batches. The multiset of consumed sequences is frozen, so the policy cannot
change the mixture, the token budget, or how many times a sequence is seen. It can change
only when each sequence is seen and what it is seen beside.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np

BASELINE_SEED_OFFSET = 1_000_003


def build_meta(cfg: dict, features: list[dict]) -> dict:
    """The exact argument handed to build_order. It carries no seed and no loss."""
    return {
        "n_sequences": cfg["n_train_sequences"],
        "batch_sequences": cfg["batch_sequences"],
        "n_steps": cfg["n_steps"],
        "seq_len": cfg["seq_len"],
        "n_domains": cfg["n_domains"],
        "features": features,
    }


def baseline_order(cfg: dict, seed: int, draw: int = 0) -> list[list[int]]:
    """One draw of the comparator ensemble: a uniform random shuffle.

    The comparator is an ensemble rather than a single permutation. Scoring against one draw
    would hand the luck of that draw to the submission, and the spread between independent
    shuffles is exactly the noise this slot is about.
    """
    rng = np.random.default_rng(BASELINE_SEED_OFFSET + 1009 * int(seed) + 7919 * int(draw))
    perm = rng.permutation(cfg["n_train_sequences"])
    b = cfg["batch_sequences"]
    return [[int(x) for x in perm[i * b : (i + 1) * b]] for i in range(cfg["n_steps"])]


def load_order_module(path: str):
    spec = importlib.util.spec_from_file_location("bia_s03_submitted_order", path)
    if spec is None or spec.loader is None:
        raise SystemExit("order_module_unloadable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_order"):
        raise SystemExit("order_module_missing_build_order")
    return mod


def call_build_order(path: str, meta: dict) -> list[list[int]]:
    mod = load_order_module(path)
    raw = mod.build_order(meta)
    return normalize_order(raw)


def normalize_order(raw) -> list[list[int]]:
    out = []
    for batch in raw:
        out.append([int(x) for x in batch])
    return out


def validate_order(order, cfg: dict):
    """Return (ok, reason). This is the deterministic permutation contract."""
    if not isinstance(order, list):
        return False, "order_not_a_list"
    if len(order) != cfg["n_steps"]:
        return False, f"order_step_count_{len(order)}_expected_{cfg['n_steps']}"
    flat = []
    for i, batch in enumerate(order):
        if not isinstance(batch, list):
            return False, f"order_batch_{i}_not_a_list"
        if len(batch) != cfg["batch_sequences"]:
            return False, f"order_batch_{i}_size_{len(batch)}_expected_{cfg['batch_sequences']}"
        for x in batch:
            if not isinstance(x, int) or isinstance(x, bool):
                return False, f"order_batch_{i}_non_integer_index"
            if x < 0 or x >= cfg["n_train_sequences"]:
                return False, f"order_batch_{i}_index_out_of_range_{x}"
        flat.extend(batch)
    if len(flat) != cfg["n_train_sequences"]:
        return False, f"order_total_{len(flat)}_expected_{cfg['n_train_sequences']}"
    seen = np.bincount(np.asarray(flat, dtype=np.int64), minlength=cfg["n_train_sequences"])
    if not bool((seen == 1).all()):
        bad = int(np.argmax(seen != 1))
        return False, f"order_not_a_permutation_index_{bad}_count_{int(seen[bad])}"
    return True, None


def order_digest(order) -> str:
    return hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()


def first_index_sequence(order) -> list[int]:
    return [int(b[0]) for b in order]
