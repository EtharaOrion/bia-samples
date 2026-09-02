#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml

Frozen constants and frozen fixtures, and no logic at all.

tests/runner.py reads CALIBRATION_ALLOCATION and measures it against the built
environment on every run. tests/checkers.py reads the band constants and reduces
that measurement. tests/test_output.py reads BAND_FIXTURES and exercises the
accepting and the rejecting half of each reduction over frozen bytes.

EXPECTED_SEPARATION_MEAN and EXPECTED_NOISE_HALF_WIDTH are discovery values.
They appear on no agent-visible surface and are obtainable only by measuring the
built environment, which is why the graded path is required to read them back.
"""

from __future__ import annotations

CALIBRATION_ALLOCATION = {
    "scheme": 'error-feedback',
    "bits": {
        'blk0.attn.proj': 5,
        'blk0.attn.qkv': 5,
        'blk0.mlp.fc': 4,
        'blk0.mlp.proj': 4,
        'blk1.attn.proj': 4,
        'blk1.attn.qkv': 6,
        'blk1.mlp.fc': 4,
        'blk1.mlp.proj': 4,
        'blk2.attn.qkv': 6,
        'blk2.mlp.fc': 4,
        'emb.tok': 3,
        'lm_head': 3,
    },
}

EXPECTED_SEPARATION_MEAN = 0.135696
EXPECTED_NOISE_HALF_WIDTH = 0.002966
HALF_WIDTH_TOLERANCE = 1e-05
BAND_EPSILON = 1e-12

BAND_FIXTURES = {
    'accept_calibration_band_at_the_grounded_centre': {
        "checker": 'calibration_separation_within_band',
        "half": 'accepting',
        "expect_reason": '',
        "proves": 'the reading the built environment produces is accepted',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.135696,
                "noise_half_width": 0.002966,
            },
        },
    },
    'accept_calibration_band_on_the_inner_edge': {
        "checker": 'calibration_separation_within_band',
        "half": 'accepting',
        "expect_reason": '',
        "proves": 'a recorded width any smaller would reject this, so the width is load bearing for the accepting half',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.138662,
                "noise_half_width": 0.002966,
            },
        },
    },
    'reject_calibration_band_near_miss_outside_the_edge': {
        "checker": 'calibration_separation_within_band',
        "half": 'rejecting',
        "expect_reason": 'calibration-separation-outside-band',
        "proves": 'one near-miss offset past the inner edge rejects, so the checker depends on the value and not on the shape',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.138663,
                "noise_half_width": 0.002966,
            },
        },
    },
    'reject_calibration_band_far_from_the_centre': {
        "checker": 'calibration_separation_within_band',
        "half": 'rejecting',
        "expect_reason": 'calibration-separation-outside-band',
        "proves": 'a separation that is the right shape and the wrong size rejects',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.254336,
                "noise_half_width": 0.002966,
            },
        },
    },
    'accept_calibration_width_at_the_grounded_value': {
        "checker": 'calibration_noise_band_width_held',
        "half": 'accepting',
        "expect_reason": '',
        "proves": 'the recorded noise half width accepts',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.135696,
                "noise_half_width": 0.002966,
            },
        },
    },
    'reject_calibration_width_near_miss_outside_tolerance': {
        "checker": 'calibration_noise_band_width_held',
        "half": 'rejecting',
        "expect_reason": 'calibration-noise-band-width-moved',
        "proves": 'a half width one near-miss offset past half_width_tolerance rejects',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.135696,
                "noise_half_width": 0.002977,
            },
        },
    },
    'reject_calibration_width_doubled': {
        "checker": 'calibration_noise_band_width_held',
        "half": 'rejecting',
        "expect_reason": 'calibration-noise-band-width-moved',
        "proves": 'a band twice as wide rejects, so a noisier measurement cannot be passed off as this one',
        "telemetry": {
            "schema": "oer20.telemetry/v1",
            "calibration": {
                "source": "harness-recompute",
                "wellformed": True,
                "points_used": 5,
                "separation_mean": 0.135696,
                "noise_half_width": 0.005932,
            },
        },
    },
}
