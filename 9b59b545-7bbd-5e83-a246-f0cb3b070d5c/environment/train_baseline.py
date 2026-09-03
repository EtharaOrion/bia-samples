#!/usr/bin/env python3
"""The locked training script for slot OER-01. The frozen axes live here.

Everything in FROZEN below is fixed by upstream rule and is not yours to change.
The script reads a recipe module you supply, asks it for the per-step update plan,
and runs the loop. It charges exactly one forward-backward pass per optimizer
step and reports that count to the harness, so a second pass hidden inside an
update is visible as a moved frozen axis rather than as free speed.

The baseline recipe below is upstream record 36. It is included so you can see the
shape a recipe takes and measure against it by running it; its step count is a
property of this script, not a published anchor, and the anchors the reward divides
by are the verifier's and are not carried in this bundle. It is also the first entry
in the pinned exclusion corpus, so submitting it
unchanged scores zero with recipe-replays-published-record.

The accelerator kernels this loop calls are provided by the pinned image. This
file is the contract and the loop; it is not a reimplementation of the model.
"""

import json
from pathlib import Path

FROZEN = {
    "dataset_id": "fineweb-edu-10B-track3",
    "batch_size": 524288,
    "arch_id": "nanogpt-track3-124M",
    "fwd_bwd_per_step": 1,
}

TARGET_VALIDATION_LOSS = 3.28

# The evaluation grid and the readout contract are data, not prose, and they live
# beside this file so the loop and the agent read the same bytes.
GRADED_READOUT_PATH = Path(__file__).with_name("graded_readout.json")


def graded_readout():
    """The evaluation grid and readout contract this loop runs against."""
    return json.loads(GRADED_READOUT_PATH.read_text(encoding="utf-8"))


def evaluation_schedule(readout=None):
    """The steps at which the verifier evaluates, read off the bound grid.

    The grid is closed on the right: there is no scheduled point after
    `last_scheduled_step`, which is why a crossing too late to be sustained inside
    the grid cannot be observed at all rather than being observed and rejected.
    """
    grid = (readout or graded_readout())["evaluation_grid"]
    first = int(grid["first_eval_step"])
    stride = int(grid["evaluation_cadence_steps"])
    last = int(grid["last_scheduled_step"])
    return list(range(first, last + 1, stride))


def halt_step(total_steps, readout=None):
    """The optimizer step at which this run stops, and why.

    Schedule exhaustion is the ordinary cause: the loop runs to the last scheduled
    evaluation point and stops there, because nothing past it is ever evaluated.
    A `total_steps` shorter than that grid halts the run early, and an early halt
    is graded as not having crossed rather than as a shorter run.
    """
    readout = readout or graded_readout()
    exhausted = int(readout["evaluation_grid"]["last_scheduled_step"])
    if int(total_steps) < exhausted:
        return {"halt_step": int(total_steps), "halt_cause": "total-steps-exhausted"}
    return {"halt_step": exhausted,
            "halt_cause": readout["halt_rule"]["default_cause"]}


def readout_row(step, raw_loss, reported_loss):
    """One evaluation row of the telemetry record this loop writes.

    `verifier_raw_loss` is the graded channel and is written unsmoothed. Whatever
    smoothing your recipe applies belongs in `submission_reported_loss`, which is
    carried for your diagnostics and is never read on the graded path.
    """
    return {
        "step": int(step),
        "verifier_raw_loss": None if raw_loss is None else float(raw_loss),
        "submission_reported_loss": None if reported_loss is None else float(reported_loss),
    }

BASELINE_RECIPE = {
    "update_chain": [
        {"primitive": "heavy_ball_momentum", "role": "hidden_matrices"},
        {"primitive": "newton_schulz_orthogonalisation", "role": "hidden_matrices"},
        {"primitive": "rms_rescale", "role": "hidden_matrices"},
        {"primitive": "adamw", "role": "aux_group"},
    ],
    "couplings": [["hidden_matrices", "aux_group", "disjoint_parameter_partition"]],
    "schedule_families": [["lr", "linear_decay"], ["momentum", "constant"]],
    "init_family": "pytorch_default",
    "hyper": {
        "lr_peak": 0.05,
        "momentum": 0.95,
        "second_moment_beta": 0.95,
        "weight_decay": 0.0,
        "warmup_frac": 0.0,
        "orthogonalisation_steps": 5,
        "trust_region_rho": 0.0,
        "aux_lr_ratio": 0.006,
    },
}

PARAM_GROUPS = [{"role": "hidden_matrices"}, {"role": "aux_group"}, {"role": "embedding_group"}]


def frozen_report():
    """What the harness records as the observed frozen axes for this run.

    The digests are the ones the loader and the model builder compute over the
    shard and the architecture specification they actually used. A run that loads
    a different shard or builds a different model reports different digests, which
    is what makes the frozen-axis invariant a measurement rather than a promise.
    """
    return {
        "dataset_id": FROZEN["dataset_id"],
        "batch_size": FROZEN["batch_size"],
        "arch_id": FROZEN["arch_id"],
        "fwd_bwd_per_step": FROZEN["fwd_bwd_per_step"],
    }


def load_recipe_module(path):
    """Import the submitted recipe module by path, for the AGENT's own local runs.

    The verifier does NOT do this. The verifier reads the RECIPE literal out of the
    bytes without importing them, and runs the submission only as a separate
    process group. This helper exists so you can drive your own experiments; it is
    not the grading path and nothing it returns is graded.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("submitted_recipe", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(recipe_path, total_steps):
    module = load_recipe_module(Path(recipe_path))
    plan = module.build_optimizer(PARAM_GROUPS, total_steps)
    readout = graded_readout()
    schedule = evaluation_schedule(readout)
    run = halt_step(total_steps, readout)
    print(
        json.dumps(
            {
                "frozen": frozen_report(),
                "target_validation_loss": TARGET_VALIDATION_LOSS,
                "total_steps": total_steps,
                "groups": [row["role"] for row in plan["groups"]],
                "init_family": plan["init_family"],
                "evaluation_schedule": schedule,
                "run": run,
                "evaluations": [readout_row(step, None, None) for step in schedule
                                if step <= run["halt_step"]],
                "graded_channel": "verifier_raw_loss",
                "note": "this stub reports the plan it would run and the telemetry shape it would write; the accelerator loop is provided by the pinned image, and it fills verifier_raw_loss at each scheduled step",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3250))
