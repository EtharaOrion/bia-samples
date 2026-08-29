"""Verifier-side derivations, written independently of the runner.

Nothing here imports the runner module. The parameter count below is built from
a symbolic term list, while the runner counts the same quantity by summing
numel over an instantiated torch module. Two derivations that share no code are
what makes the agreement between them worth asserting.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from typing import Any, Dict, List, Tuple

ARCH_KEYS = ("n_layer", "d_model", "head_dim", "n_head", "n_kv_head", "d_ff")

VERIFIER_FROZEN_DIR = os.environ.get(
    "BIA_VERIFIER_FROZEN_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frozen"),
)


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def load_json(path: str) -> Any:
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def frozen(name: str) -> Any:
    return load_json(os.path.join(VERIFIER_FROZEN_DIR, name))


def arch_digest(arch: Dict[str, int]) -> str:
    return sha256_hex(canonical_json({k: int(arch[k]) for k in ARCH_KEYS}).encode("utf-8"))


def param_terms(arch: Dict[str, int], vocab_size: int) -> List[Tuple[str, int]]:
    """Every weight tensor the frozen model family creates, named and sized."""
    d = int(arch["d_model"])
    hd = int(arch["head_dim"])
    nh = int(arch["n_head"])
    nkv = int(arch["n_kv_head"])
    ff = int(arch["d_ff"])
    layers = int(arch["n_layer"])
    terms: List[Tuple[str, int]] = [
        ("tok_emb.weight", vocab_size * d),
        ("norm_f.weight", d),
        ("lm_head.weight", d * vocab_size),
    ]
    for i in range(layers):
        p = "blocks." + str(i) + "."
        terms.extend([
            (p + "norm1.weight", d),
            (p + "attn.q_proj.weight", d * nh * hd),
            (p + "attn.k_proj.weight", d * nkv * hd),
            (p + "attn.v_proj.weight", d * nkv * hd),
            (p + "attn.o_proj.weight", nh * hd * d),
            (p + "norm2.weight", d),
            (p + "mlp.gate_proj.weight", d * ff),
            (p + "mlp.up_proj.weight", d * ff),
            (p + "mlp.down_proj.weight", ff * d),
        ])
    return terms


def analytic_param_count(arch: Dict[str, int], vocab_size: int) -> int:
    return int(sum(size for _, size in param_terms(arch, vocab_size)))


def param_band(profile: Dict[str, Any]) -> Tuple[int, int]:
    budget = int(profile["param_budget"])
    tol = float(profile["param_tolerance"])
    return int(round(budget * (1.0 - tol))), int(round(budget * (1.0 + tol)))


def expected_split(n_bytes: int, val_frac: float) -> Dict[str, List[int]]:
    val_len = int(n_bytes * val_frac)
    boundary = n_bytes - val_len
    return {"train_range": [0, boundary], "val_range": [boundary, n_bytes]}


def lr_at(step: int, profile: Dict[str, Any], peak: float) -> float:
    warm = int(profile["warmup_steps"])
    total = int(profile["steps"])
    floor = peak * float(profile["final_lr_frac"])
    if step < warm:
        return peak * (step + 1) / warm
    frac = (step - warm) / max(1, total - warm)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * min(1.0, frac)))


def schedule_digest(profile: Dict[str, Any], recipe: Dict[str, Any]) -> str:
    peak = float(recipe["optimizer"]["lr"])
    vals = [round(lr_at(s, profile, peak), 12) for s in range(int(profile["steps"]))]
    return sha256_hex(canonical_json(vals).encode("utf-8"))


def tree_digest(root: str) -> str:
    parts: List[str] = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".pyc"):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            with open(full, "rb") as handle:
                parts.append(rel + "\0" + sha256_hex(handle.read()))
    return sha256_hex("\n".join(parts).encode("utf-8"))


def verify_chain(records: List[Dict[str, Any]], key: str) -> Tuple[bool, str]:
    prev = "genesis"
    for i, rec in enumerate(records):
        body = {k: v for k, v in rec.items() if k != "chain"}
        mac = hmac.new(
            key.encode("utf-8"),
            (prev + canonical_json(body)).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if rec.get("chain") != mac:
            return False, "chain_break_at_index_" + str(i)
        prev = mac
    return True, ""


def validate_arch(arch: Dict[str, Any], space: Dict[str, Any], profile_name: str) -> List[str]:
    rules = space["profiles"][profile_name]
    problems: List[str] = []
    for key in ARCH_KEYS:
        if key not in arch:
            problems.append("missing_key_" + key)
    if problems:
        return problems
    for key in ARCH_KEYS:
        value = arch[key]
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append("non_integer_" + key)
    if problems:
        return problems
    for key in ARCH_KEYS:
        rule = rules[key]
        value = arch[key]
        if "choices" in rule and value not in rule["choices"]:
            problems.append("not_an_allowed_choice_" + key)
        if "min" in rule and value < rule["min"]:
            problems.append("below_min_" + key)
        if "max" in rule and value > rule["max"]:
            problems.append("above_max_" + key)
        if "multiple_of" in rule and value % rule["multiple_of"] != 0:
            problems.append("not_a_multiple_" + key)
    if arch["n_kv_head"] > arch["n_head"]:
        problems.append("n_kv_head_exceeds_n_head")
    elif arch["n_head"] % arch["n_kv_head"] != 0:
        problems.append("n_kv_head_does_not_divide_n_head")
    if arch["head_dim"] % 2 != 0:
        problems.append("odd_head_dim_breaks_rotary")
    return problems


REQUIRED_EVENT_ORDER = [
    ("header", "none"),
    ("arm_start", "reference"),
    ("eval", "reference"),
    ("eval", "reference"),
    ("arm_end", "reference"),
    ("arm_start", "submission"),
    ("eval", "submission"),
    ("eval", "submission"),
    ("arm_end", "submission"),
    ("final", "none"),
]


def event_sequence(records: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    return [(str(r.get("event")), str(r.get("arm"))) for r in records]
