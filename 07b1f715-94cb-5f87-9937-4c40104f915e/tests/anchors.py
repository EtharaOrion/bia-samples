"""GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml.

Anchor and frozen-field constants the S02 verifier grades against. The full
profile anchors carry status CALIBRATED_FROM_MEASURED_RUN, and the runs they were
bound from are recorded in solution/grounding.yaml under
calibration_measurement.
"""

from __future__ import annotations

PROFILE_ANCHORS = {
    "full": {
        "baseline_steps": 1400,
        "min_seeds": 2,
        "sig_margin": 0.023,
        "status": "CALIBRATED_FROM_MEASURED_RUN",
        "target_loss": 5.3,
        "target_steps": 980,
    },
    "smoke": {
        "baseline_steps": 36,
        "min_seeds": 2,
        "sig_margin": 0.01,
        "status": "PATH_PROOF_ONLY",
        "target_loss": 9.6,
        "target_steps": 20,
    },
}

ARCHITECTURE = {
    "full": {
        "d_model": 384,
        "head_tied": False,
        "mlp": "gelu_4x",
        "n_head": 6,
        "n_layer": 6,
        "norm": "rmsnorm",
        "pos": "learned",
        "seq_len": 512,
        "vocab_size": 50304,
    },
    "smoke": {
        "d_model": 64,
        "head_tied": False,
        "mlp": "gelu_4x",
        "n_head": 2,
        "n_layer": 2,
        "norm": "rmsnorm",
        "pos": "learned",
        "seq_len": 64,
        "vocab_size": 50304,
    },
}

BATCH_SEQUENCES = {
    "full": 48,
    "smoke": 4,
}

TRAIN_SHARD = {
    "full": "771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e",
    "smoke": "a81a534750858bb96848ad8f2f726066dc8afd9356ea49f5b8a71f486ac6c7ab",
}

VAL_SHARD = {
    "full": "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640",
    "smoke": "fbc35229e0d9e78b155b0b228ce34619c68d6d7305d0634dbbf0b38e88f641ee",
}
