#!/usr/bin/env python3
"""Reference solution for slot OER-01. A derivation, not a recital.

The declarative RECIPE below is the whole of what the fingerprint screen reads. It
is written as a module-level literal on purpose: the verifier lifts it with
`ast.literal_eval` and never imports this file, so a recipe that could only be
obtained by executing the submission would not screen at all.

The derivation, in one paragraph. Record 36 and record 46 both sit in the
orthogonalised-momentum family, and both bound the effective step with a per-layer
RMS rescale whose constant is fixed for the whole run. That constant has to serve
two regimes at once: the early steps, where the curvature is mild and a larger
step is affordable, and the late steps, where it is not. This reference replaces
the fixed rescale with a spectral trust region whose radius follows a two-phase
schedule against a running spectral-norm estimate, so each regime gets its own
bound instead of one compromise serving both. The second change removes the
auxiliary AdamW group's independent second-moment estimator and shares the hidden
group's, which removes the scale drift between the two groups and lets one
learning-rate schedule govern both. Initialisation is depth-scaled orthogonal so
the trust region starts from a known spectral radius rather than an empirical one.

The frozen axes are untouched: the same dataset, the same batch size, the same
architecture, and exactly one forward-backward pass per optimizer step.
"""

RECIPE = {
    "update_chain": [
        {"primitive": "heavy_ball_momentum", "role": "hidden_matrices"},
        {"primitive": "newton_schulz_orthogonalisation", "role": "hidden_matrices"},
        {"primitive": "spectral_trust_region", "role": "hidden_matrices"},
        {"primitive": "adamw", "role": "aux_group"},
    ],
    "couplings": [
        ["hidden_matrices", "aux_group", "shared_second_moment"],
    ],
    "schedule_families": [
        ["lr", "cosine_then_linear"],
        ["momentum", "linear_ramp"],
        ["trust_region_rho", "two_phase"],
    ],
    "init_family": "depth_scaled_orthogonal",
    "hyper": {
        "lr_peak": 0.038,
        "momentum": 0.945,
        "second_moment_beta": 0.96,
        "weight_decay": 0.0,
        "warmup_frac": 0.02,
        "orthogonalisation_steps": 5,
        "trust_region_rho": 1.35,
        "aux_lr_ratio": 0.0045,
    },
}


def lr_at(step, total_steps):
    """Cosine to the warmup peak, then linear to zero. One schedule, both groups."""
    warmup = max(1, int(RECIPE["hyper"]["warmup_frac"] * total_steps))
    peak = RECIPE["hyper"]["lr_peak"]
    if step < warmup:
        return peak * (step + 1) / warmup
    span = max(1, total_steps - warmup)
    progress = (step - warmup) / span
    if progress < 0.5:
        cosine = 0.5 * (1.0 + _cos(progress * 2.0))
        return peak * (0.35 + 0.65 * cosine)
    return peak * 0.35 * max(0.0, 1.0 - (progress - 0.5) / 0.5)


def momentum_at(step, total_steps):
    """Linear ramp from 0.85 to the bound peak over the first fifth of the run."""
    ceiling = RECIPE["hyper"]["momentum"]
    ramp = max(1, total_steps // 5)
    return min(ceiling, 0.85 + (ceiling - 0.85) * (step / ramp))


def trust_radius_at(step, total_steps, spectral_norm):
    """Two phases. A generous radius early, a spectral-norm-bounded one late."""
    rho = RECIPE["hyper"]["trust_region_rho"]
    if step < total_steps // 2:
        return rho
    return rho / max(1.0, spectral_norm)


def _cos(x):
    """Cosine without importing math, so this file stays literal-first and tiny."""
    term, total, index = 1.0, 1.0, 0
    theta = 3.141592653589793 * x
    while index < 12:
        index += 1
        term *= -(theta * theta) / ((2 * index - 1) * (2 * index))
        total += term
    return total


def build_optimizer(param_groups, total_steps):
    """Return the per-step update plan the frozen training script consumes.

    The frozen script owns the data, the batch, the architecture and the single
    forward-backward pass. This function owns only the update rule, its schedules
    and the initialisation family, which is exactly the free set.
    """
    return {
        "recipe": RECIPE,
        "total_steps": total_steps,
        "groups": [
            {
                "role": group.get("role"),
                "lr": [lr_at(step, total_steps) for step in range(total_steps)],
                "momentum": [momentum_at(step, total_steps) for step in range(total_steps)],
                "second_moment_beta": RECIPE["hyper"]["second_moment_beta"],
                "shares_second_moment_with": "hidden_matrices" if group.get("role") == "aux_group" else None,
                "lr_scale": RECIPE["hyper"]["aux_lr_ratio"] if group.get("role") == "aux_group" else 1.0,
            }
            for group in param_groups
        ],
        "init_family": RECIPE["init_family"],
        "orthogonalisation_steps": RECIPE["hyper"]["orthogonalisation_steps"],
    }


if __name__ == "__main__":
    import json

    print(json.dumps({"recipe": RECIPE}, sort_keys=True))
