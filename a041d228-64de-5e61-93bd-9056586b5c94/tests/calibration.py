#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml

Frozen constants and frozen fixtures, and no logic at all.

tests/runner.py reads CALIBRATION_ALLOCATION and measures it against the built
environment on every run. tests/checkers.py reads the band constants and reduces
that measurement. tests/test_output.py reads BAND_FIXTURES and exercises the
accepting and the rejecting half of each reduction over frozen bytes.

WHAT THE RE-BASE MOVED HERE. Before the re-base the probe measured a non-uniform
reference allocation against an arithmetic error model, and the two constants
below were discovery values: numbers that appeared on no agent-visible surface
and could only be obtained by evaluating the stand-in. The stand-in is gone, the
graded path now runs real forward passes of a real checkpoint, and this lane
measured no checkpoint, so it has no discovery value to record and refuses to
invent one. The probe is repinned to a DETERMINISM IDENTITY instead. It restates
the control allocation and measures it against the control allocation over the
same held-out points, so its separation is exactly 0.0 at every point and its
band is exactly 0.0, by the arithmetic of a paired difference of two identical
readings. That identity holds only if the evaluation actually ran, actually
paired each reading to its own token batch, and actually returned the same
number twice for the same weights, which is a property a real evaluation can
fail and the retired arithmetic stand-in could not.

The tolerances below are not a measurement either. They admit kernel-level
nondeterminism between two identical forward passes and nothing larger.
Restoring a measured centre and a measured width for a substantive probe is
recorded as gap-oer-20-calibration-discovery-value-unmeasured and belongs to the
first pilot that runs this slot on a real checkpoint.
"""

from __future__ import annotations

# The control allocation restated: every quantizable tensor of the frozen
# checkpoint at the budget's mean bit width under round-to-nearest.
CALIBRATION_ALLOCATION = {
    "scheme": 'rtn',
    "bits": {
        'wte.weight': 4,
        'blocks.0.attn.qkv.weight': 4,
        'blocks.0.attn.proj.weight': 4,
        'blocks.0.mlp.fc.weight': 4,
        'blocks.0.mlp.proj.weight': 4,
        'blocks.1.attn.qkv.weight': 4,
        'blocks.1.attn.proj.weight': 4,
        'blocks.1.mlp.fc.weight': 4,
        'blocks.1.mlp.proj.weight': 4,
        'blocks.2.attn.qkv.weight': 4,
        'blocks.2.attn.proj.weight': 4,
        'blocks.2.mlp.fc.weight': 4,
        'blocks.2.mlp.proj.weight': 4,
        'blocks.3.attn.qkv.weight': 4,
        'blocks.3.attn.proj.weight': 4,
        'blocks.3.mlp.fc.weight': 4,
        'blocks.3.mlp.proj.weight': 4,
        'blocks.4.attn.qkv.weight': 4,
        'blocks.4.attn.proj.weight': 4,
        'blocks.4.mlp.fc.weight': 4,
        'blocks.4.mlp.proj.weight': 4,
        'blocks.5.attn.qkv.weight': 4,
        'blocks.5.attn.proj.weight': 4,
        'blocks.5.mlp.fc.weight': 4,
        'blocks.5.mlp.proj.weight': 4,
        'blocks.6.attn.qkv.weight': 4,
        'blocks.6.attn.proj.weight': 4,
        'blocks.6.mlp.fc.weight': 4,
        'blocks.6.mlp.proj.weight': 4,
        'blocks.7.attn.qkv.weight': 4,
        'blocks.7.attn.proj.weight': 4,
        'blocks.7.mlp.fc.weight': 4,
        'blocks.7.mlp.proj.weight': 4,
        'blocks.8.attn.qkv.weight': 4,
        'blocks.8.attn.proj.weight': 4,
        'blocks.8.mlp.fc.weight': 4,
        'blocks.8.mlp.proj.weight': 4,
        'blocks.9.attn.qkv.weight': 4,
        'blocks.9.attn.proj.weight': 4,
        'blocks.9.mlp.fc.weight': 4,
        'blocks.9.mlp.proj.weight': 4,
        'blocks.10.attn.qkv.weight': 4,
        'blocks.10.attn.proj.weight': 4,
        'blocks.10.mlp.fc.weight': 4,
        'blocks.10.mlp.proj.weight': 4,
        'blocks.11.attn.qkv.weight': 4,
        'blocks.11.attn.proj.weight': 4,
        'blocks.11.mlp.fc.weight': 4,
        'blocks.11.mlp.proj.weight': 4,
        'lm_head.weight': 4,
    },
}

EXPECTED_SEPARATION_MEAN = 0.0
EXPECTED_NOISE_HALF_WIDTH = 0.0
HALF_WIDTH_TOLERANCE = 1e-06
BAND_EPSILON = 1e-06

BAND_FIXTURES = {
    'accept_calibration_band_at_the_grounded_centre': {
        "checker": 'calibration_separation_within_band',
        "half": 'accepting',
        "expect_reason": '',
        "proves": 'the exact zero the determinism identity produces is accepted',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.0,
                "noise_half_width": 0.0,
            },
        },
    },
    'accept_calibration_band_on_the_inner_edge': {
        "checker": 'calibration_separation_within_band',
        "half": 'accepting',
        "expect_reason": '',
        "proves": 'a reading exactly on the bound epsilon still accepts, so the epsilon is load bearing for the accepting half',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 1e-06,
                "noise_half_width": 0.0,
            },
        },
    },
    'reject_calibration_band_near_miss_outside_the_edge': {
        "checker": 'calibration_separation_within_band',
        "half": 'rejecting',
        "expect_reason": 'calibration-separation-outside-band',
        "proves": 'one near-miss offset past that same edge rejects, so the checker depends on the value and not on the shape',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 1.000001e-06,
                "noise_half_width": 0.0,
            },
        },
    },
    'reject_calibration_band_far_from_the_centre': {
        "checker": 'calibration_separation_within_band',
        "half": 'rejecting',
        "expect_reason": 'calibration-separation-outside-band',
        "proves": 'a probe that separated from itself by a real amount rejects, because two readings of one state cannot disagree',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 4e-05,
                "noise_half_width": 0.0,
            },
        },
    },
    'accept_calibration_width_at_the_grounded_value': {
        "checker": 'calibration_noise_band_width_held',
        "half": 'accepting',
        "expect_reason": '',
        "proves": 'the exact zero band the identity produces is accepted',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.0,
                "noise_half_width": 0.0,
            },
        },
    },
    'reject_calibration_width_near_miss_outside_tolerance': {
        "checker": 'calibration_noise_band_width_held',
        "half": 'rejecting',
        "expect_reason": 'calibration-noise-band-width-moved',
        "proves": 'a half width one near-miss offset past the bound tolerance rejects',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.0,
                "noise_half_width": 2.000001e-06,
            },
        },
    },
    'reject_calibration_width_doubled': {
        "checker": 'calibration_noise_band_width_held',
        "half": 'rejecting',
        "expect_reason": 'calibration-noise-band-width-moved',
        "proves": 'a band twice as wide rejects, so a drifting measurement path cannot be passed off as this one',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.0,
                "noise_half_width": 4.000002e-06,
            },
        },
    },
}
