"""Frozen operating point for the BIA S10 reproduction-under-ablation slot.

Every value in FROZEN_FULL is part of the frozen substrate. Nothing here is
agent-writable: the submission supplies an update rule only. SMOKE is the
identical shape at a scale that runs on CPU so the pipeline can be proven
end to end without an accelerator.
"""

from __future__ import annotations

import hashlib
import json

# The scaled operating point. Budget envelope from solution/grounding.yaml.
# Depth and width inside that envelope were chosen by H100 measurement, not
# intuition: the measured ablation cost (baseline minus target val loss, nats,
# 1200 steps, frozen recipe) was 6L/384 inverted, 8L/256 = 0.04, 10L/256 = 0.30,
# 12L/256 = non-finite baseline. 10L/256 is the finite, large, measurable point.
FROZEN_FULL = {
    "profile": "full",
    "vocab_size": 256,
    "d_model": 256,
    "n_layer": 10,
    "n_head": 4,
    "seq_len": 512,
    "mlp_mult": 4,
    "batch_size": 24,
    "steps": 1200,
    "eval_batches": 24,
    "train_region_bytes": 90_000_000,
    "val_region_bytes": 5_000_000,
    "seed": 0,
    "grad_accum": 1,
    "arm_wallclock_guard_sec": 900.0,
}

SMOKE = {
    "profile": "smoke",
    "vocab_size": 256,
    "d_model": 32,
    "n_layer": 2,
    "n_head": 2,
    "seq_len": 64,
    "mlp_mult": 4,
    "batch_size": 4,
    "steps": 20,
    "eval_batches": 4,
    "train_region_bytes": 229_376,
    "val_region_bytes": 32_768,
    "seed": 0,
    "grad_accum": 1,
    "arm_wallclock_guard_sec": 120.0,
}

# One forward-backward pass per optimizer step. Frozen by upstream rule and
# never relaxed: an arm that reports a different value is rejected.
FORWARD_BACKWARD_PER_STEP = 1

# The normalization family the ablation removes. Membership is by exact class
# name and by exact torch.nn.functional entry point, never by substring guess.
ABLATED_MODULE_CLASSES = (
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "SyncBatchNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LocalResponseNorm",
    "CrossMapLRN2d",
)

ABLATED_FUNCTIONALS = (
    "layer_norm",
    "rms_norm",
    "group_norm",
    "batch_norm",
    "instance_norm",
    "local_response_norm",
    "normalize",
)

# Phases in which a normalization event is forbidden. The update phase is
# recorded but permitted, because normalizing an update tensor is an optimizer
# side compensation and not a reintroduction of activation normalization.
FORBIDDEN_PHASES = ("forward", "backward", "eval")

# Scoring parameters.
MIN_ANCHOR_GAP = 0.05          # nats, the ablation must actually cost something
DIVERGENCE_TOL = 2.0e-3        # nats, streaming versus post hoc validation loss
UNTRAINED_CEILING_VOCAB_LN = True  # baseline substitute when the naive port diverges


def get_config(smoke: bool) -> dict:
    cfg = dict(SMOKE if smoke else FROZEN_FULL)
    cfg["forward_backward_per_step"] = FORWARD_BACKWARD_PER_STEP
    return cfg


def config_digest(cfg: dict) -> str:
    payload = {k: cfg[k] for k in sorted(cfg) if k != "arm_wallclock_guard_sec"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
