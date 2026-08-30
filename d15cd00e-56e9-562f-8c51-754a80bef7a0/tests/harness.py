#!/usr/bin/env python3
"""The verifier-owned evaluation harness. It, and only it, produces telemetry.

What this file owns is the semantics the readout-manipulation obligation turns on,
and those semantics are implemented here rather than described:

- The evaluation grid is the VERIFIER'S. `scheduled_steps` is computed from the
  bound cadence and window before the run starts, so refining, coarsening or
  hand-placing evaluation points is not a submission choice.
- The graded series is assembled from `verifier_raw_loss` and nothing else. A
  number the submission reported is carried alongside, unsmoothed and unused, so a
  checker can prove it never reached the graded path. It is never blended in.
- Checkpoints are written by this harness into its own run directory, and the
  digest recorded for a step is the digest of the bytes this harness wrote. A
  checkpoint the submission selected has a different digest and is refused.
- The fingerprint screen is stamped as sequence 0 with the accelerator-second
  counter at exactly zero, before any seed-launch event exists.

WHAT THE BACKEND IS, STATED BEFORE ANYTHING RESTS ON IT
------------------------------------------------------
`train_and_evaluate` is the seam where an evaluation backend plugs in. This bundle
carries NO accelerator: there is no H100, no fineweb-edu shard and no 124M model
inside the verifier image, so no run of the real objective can happen here. The
seam is therefore implemented as a DETERMINISTIC RECIPE-RESPONSE SURROGATE, and it
is labelled as one in the telemetry it writes (`backend.kind` reads
`deterministic-recipe-response-surrogate`, never `accelerator`).

The surrogate is a model, not a measurement, and the distinction is load-bearing:

  * It reads the submission's declarative RECIPE, which the bundle's own contract
    makes the authoritative description of the submitted algorithm, and reduces the
    declared algebra and hyperparameters to ONE scalar, the effective-step
    multiplier `m`. Better algebra and better-conditioned hyperparameters buy a
    larger `m`; worse ones buy a smaller one.
  * It evaluates a fixed power-law loss curve at `m * step`. The curve is anchored
    on the bundle's OWN bound anchors, and both anchors are reproduced rather than
    asserted: the locked baseline recipe scores `m == 1.0` exactly and therefore
    crosses at `baseline_metric` exactly, and the pinned world-record recipe
    crosses at 2687.6 against a bound `target_metric` of 2690, a residual of 2.4
    steps or 0.09 percent. Nothing here is fitted to the reference. Both anchor
    residuals are re-derived from these bytes by the repair proof recorded in
    seed/results/job4_OER-01.md; carrying that derivation as a bundle-local test
    needs solution/grounding.yaml to grow a backend block, which is a separate
    change and is named there as the follow-up it is.
  * It is a strict function of the submitted recipe and the bound anchors. It
    reads no clock, no random source and no network, it never reads the golden
    trajectory or the checker fixtures, and it contains no table keyed by any
    particular submission. A submission whose recipe is worse crosses later or
    never crosses, which is what makes the graded quantity discriminating rather
    than a constant.

What this file therefore does NOT establish is what an accelerator would report.
That obligation is unchanged and is recorded as
gap-oer01-harness-trainer-backend-simulated, which supersedes the earlier
gap-oer01-harness-trainer-backend-unbuilt: the backend now exists, produces the
verifier-owned run state the graded checkers read, and discriminates between
recipes, but it is a derivation of the same standing as the golden trajectory and
is not an accelerator measurement.

If the bound anchors cannot be resolved this file still refuses rather than
inventing a series: it writes no telemetry and exits non-zero, tests/runner.py
fails closed with `launched: false`, and the EFFECT checker attributes the zero to
graded-run-produced-no-telemetry.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fingerprint  # noqa: E402
import runner  # noqa: E402

TELEMETRY_SCHEMA = "bia.oer01.telemetry/v1"

BACKEND_ABSENT_REASON = "trainer-backend-unbuilt"

BACKEND_KIND = "deterministic-recipe-response-surrogate"
BACKEND_VERSION = "v1"

# The seeds the VERIFIER draws. Bound here, in verifier-owned bytes, so the draw is
# not a submission choice and is identical on every run. The count comes from the
# admin plane; the pool is fixed and ordered.
VERIFIER_SEED_POOL = (1101, 2917, 4133, 5051, 6079, 7159)

# The wall-clock a single submission launch is allowed inside the isolation
# boundary. The launch exists to prove the effect the EFFECT checker names; nothing
# the child prints is ever a graded number, so a slow child is capped rather than
# waited on.
LAUNCH_TIMEOUT_SEC = 60.0

# --------------------------------------------------------------------------
# The response model. Every coefficient below is an AUTHORED PRIOR over the
# published track-3 optimizer lineage, of exactly the same standing as the golden
# trajectory: derived, never measured. They are collected here, named, and kept
# out of every other file so a reader can audit the whole model in one place.
#
# The two structural anchors the model must reproduce are the bundle's own:
#   * the locked baseline recipe (upstream record 36) must not cross inside the
#     verifier's evaluation window, because its crossing sits at baseline_metric;
#   * the pinned world record (upstream record 46) must cross at target_metric.
# Both are checked by tests/test_output.py's backend calibration tests.
# --------------------------------------------------------------------------

# Gain per distinct primitive appearing anywhere in the update chain. A primitive
# the model does not know contributes nothing: an unknown name is not evidence of
# an improvement.
PRIMITIVE_GAIN = {
    "heavy_ball_momentum": 0.0,
    "newton_schulz_orthogonalisation": 0.0,
    "rms_rescale": 0.0,
    "adamw": 0.0,
    "sgd": -0.020,
    "embedding_shortcut": 0.030,
    "soap_preconditioner": 0.045,
    "value_embedding_lambda": 0.090,
    "spectral_trust_region": 0.090,
}

# Gain per distinct coupling kind between parameter groups.
COUPLING_GAIN = {
    "disjoint_parameter_partition": 0.0,
    "shared_learning_rate": 0.010,
    "shared_second_moment": 0.060,
    "residual_lambda_gate": 0.065,
}

# Gain per (scheduled parameter, family) pair. Keyed on the pair rather than on the
# family alone, because a constant momentum and a constant learning rate are not
# the same decision.
SCHEDULE_GAIN = {
    ("lr", "constant"): -0.060,
    ("lr", "linear_decay"): 0.0,
    ("lr", "cosine_decay"): 0.015,
    ("lr", "warmup_then_linear_decay"): 0.030,
    ("lr", "cosine_then_linear"): 0.045,
    ("momentum", "constant"): 0.0,
    ("momentum", "linear_ramp"): 0.015,
    ("lambda_gate", "linear_ramp"): 0.010,
    ("trust_region_rho", "two_phase"): 0.025,
}

INIT_GAIN = {
    "pytorch_default": 0.0,
    "depth_scaled_orthogonal": 0.030,
}

# The well-conditioned operating point the hyperparameter penalties are taken
# against. Hyperparameters can only LOSE effective step here, never buy it: the
# instruction tells the agent to change the algorithm and not the digits, so the
# model must not reward retuning.
#
# The learning-rate band centre is the LOCKED BASELINE's own lr_peak, which is what
# makes the baseline recipe score a multiplier of exactly 1.0 and therefore
# reproduce baseline_metric exactly rather than approximately. The warmup penalty
# is one-sided for the same reason: the locked baseline uses no warmup at all, and
# the one corpus entry carrying a long warmup (record 40) is an older and slower
# record, so over-long warmup costs steps while its absence costs nothing.
LR_BAND_CENTRE = 0.05
LR_PENALTY = 0.35
WARMUP_CENTRE = 0.03
WARMUP_SPAN = 0.05
WARMUP_PENALTY = 0.20
DEGENERATE_PENALTY = 0.50
MISSING_ORTHOGONALISATION_PENALTY = 0.15
MISSING_AUX_LR_PENALTY = 0.10
SECOND_MOMENT_PENALTY = 0.20
UNRADIUSED_TRUST_REGION_PENALTY = 0.12

MULTIPLIER_FLOOR = 0.50
MULTIPLIER_CEILING = 1.60

# The loss curve. L(s) = floor + scale * s ** -exponent, with `scale` solved so the
# curve passes through (baseline_metric, target_validation_loss) exactly. The floor
# is the irreducible-entropy asymptote and the exponent is the optimization
# exponent; both are authored priors and neither is fitted to the reference.
LOSS_FLOOR = 2.90
LOSS_EXPONENT = 0.55

# Primitives that charge a second forward-backward pass against one optimizer step.
# Charging two passes as one step is exactly what the frozen rule forbids, so a
# chain naming one of these reports a moved axis rather than free speed.
EXTRA_PASS_PRIMITIVES = {
    "sharpness_aware_ascent": 1,
    "second_forward_backward": 1,
    "gradient_accumulation": 1,
    "lookahead_inner_loop": 1,
}

# Axis fields a recipe may name. The loader and the model builder pin the frozen
# record; anything the recipe asks for on these axes is applied ON TOP of that
# record and is therefore visible to the frozen-axis invariant.
AXIS_REQUEST_KEYS = (
    "dataset_id",
    "dataset_digest",
    "batch_size",
    "arch_id",
    "arch_digest",
    "fwd_bwd_per_step",
)


def scheduled_steps(first_step, last_step, cadence):
    """The verifier's own evaluation grid, fixed before the run begins."""
    step, rows = int(first_step), []
    while step <= int(last_step):
        rows.append(step)
        step += int(cadence)
    return rows


def checkpoint_digest(payload: bytes) -> str:
    """The digest of the bytes THIS harness wrote. Nothing else is ever compared."""
    return hashlib.sha256(payload).hexdigest()


def _hyper(recipe) -> dict:
    row = recipe if isinstance(recipe, dict) else {}
    hyper = row.get("hyper")
    return dict(hyper) if isinstance(hyper, dict) else {}


def _number(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number or number in (float("inf"), float("-inf")):
        return float(default)
    return number


def _primitives(recipe) -> list:
    row = recipe if isinstance(recipe, dict) else {}
    names = []
    for item in row.get("update_chain") or []:
        if isinstance(item, dict):
            names.append(str(item.get("primitive", "")))
    return names


def structural_gain(recipe) -> float:
    """What the declared ALGEBRA buys, before any hyperparameter is consulted."""
    row = recipe if isinstance(recipe, dict) else {}
    total = 0.0
    for name in sorted(set(_primitives(recipe))):
        total += PRIMITIVE_GAIN.get(name, 0.0)
    kinds = set()
    for item in row.get("couplings") or []:
        if isinstance(item, (list, tuple)) and item:
            kinds.add(str(item[-1]))
    for kind in sorted(kinds):
        total += COUPLING_GAIN.get(kind, 0.0)
    pairs = set()
    for item in row.get("schedule_families") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.add((str(item[0]), str(item[1])))
    for pair in sorted(pairs):
        total += SCHEDULE_GAIN.get(pair, 0.0)
    total += INIT_GAIN.get(str(row.get("init_family", "")), 0.0)
    return total


def hyper_penalty(recipe) -> float:
    """What badly conditioned NUMBERS cost. Never negative, so digits buy nothing."""
    hyper = _hyper(recipe)
    names = set(_primitives(recipe))
    total = 0.0

    lr_peak = _number(hyper.get("lr_peak"), 0.0)
    if lr_peak <= 0.0:
        total += DEGENERATE_PENALTY
    else:
        total += LR_PENALTY * math.log10(lr_peak / LR_BAND_CENTRE) ** 2

    momentum = _number(hyper.get("momentum"), 0.0)
    if not 0.0 < momentum < 1.0:
        total += DEGENERATE_PENALTY

    warmup = _number(hyper.get("warmup_frac"), 0.0)
    total += WARMUP_PENALTY * min(1.0, (max(0.0, warmup - WARMUP_CENTRE) / WARMUP_SPAN) ** 2)

    if "newton_schulz_orthogonalisation" in names and _number(hyper.get("orthogonalisation_steps"), 0.0) < 1.0:
        total += MISSING_ORTHOGONALISATION_PENALTY

    if "adamw" in names:
        beta = _number(hyper.get("second_moment_beta"), 0.0)
        if not 0.8 <= beta < 1.0:
            total += SECOND_MOMENT_PENALTY
        if _number(hyper.get("aux_lr_ratio"), 0.0) <= 0.0:
            total += MISSING_AUX_LR_PENALTY

    if "spectral_trust_region" in names and _number(hyper.get("trust_region_rho"), 0.0) <= 0.0:
        total += UNRADIUSED_TRUST_REGION_PENALTY

    return total


def effective_step_multiplier(recipe) -> float:
    """One scalar: how many baseline steps of progress one submitted step buys.

    1.0 reproduces the locked baseline. Above 1.0 the run reaches a given loss in
    fewer optimizer steps, which is the whole graded quantity, and below 1.0 it
    needs more. Clamped at both ends, because no declarative recipe is evidence of
    an unbounded speedup and none is evidence of a total collapse either.
    """
    raw = 1.0 + structural_gain(recipe) - hyper_penalty(recipe)
    return min(max(raw, MULTIPLIER_FLOOR), MULTIPLIER_CEILING)


def loss_curve(baseline_metric, target_validation_loss):
    """The anchored curve. Solved so L(baseline_metric) == the bound target loss."""
    baseline = float(baseline_metric)
    target = float(target_validation_loss)
    scale = (target - LOSS_FLOOR) * baseline**LOSS_EXPONENT

    def evaluate(effective_step):
        step = max(1.0, float(effective_step))
        return LOSS_FLOOR + scale * step**-LOSS_EXPONENT

    return evaluate


def observed_frozen_axes(recipe, bound) -> dict:
    """What the run actually loaded and built. A measurement, not the bound record.

    The loader and the model builder pin the shard, the architecture and the batch;
    this backend has no shard to hash, so it reports the pinned record as the loaded
    identity. On top of that it applies what the submitted recipe ASKS FOR: an extra
    forward-backward pass charged inside the update chain, or an axis field the
    recipe names outright. Either one moves an axis and the invariant sees it.
    """
    seen = dict(bound or {})
    row = recipe if isinstance(recipe, dict) else {}
    passes = 1
    for name in _primitives(recipe):
        passes += EXTRA_PASS_PRIMITIVES.get(name, 0)
    seen["fwd_bwd_per_step"] = passes
    for key in AXIS_REQUEST_KEYS:
        if key in row:
            seen[key] = row[key]
    return seen


def train_and_evaluate(recipe, seed, grid, frozen, curve, run_dir):
    """The evaluation seam, implemented as the surrogate this file's header names.

    Returns one row per grid point carrying: step, verifier_raw_loss (this harness's
    own unsmoothed evaluation), submission_reported_loss (carried, never graded, and
    None here because no submission-reported readout reaches this harness at all),
    weights_origin, weights_digest and harness_checkpoint_digest.

    The checkpoint at each scheduled step is BYTES THIS HARNESS WRITES into its own
    run directory, and the digest recorded is the digest of those bytes read back.
    Nothing the submission produced is hashed here.
    """
    multiplier = effective_step_multiplier(recipe)
    rows = []
    for step in grid:
        raw = curve(multiplier * float(step))
        payload = json.dumps(
            {
                "harness": BACKEND_KIND + "/" + BACKEND_VERSION,
                "seed": int(seed),
                "step": int(step),
                "effective_step_multiplier": multiplier,
                "verifier_raw_loss": raw,
                "frozen_axes": frozen,
            },
            sort_keys=True,
        ).encode("utf-8")
        target = Path(run_dir) / ("checkpoint-%d-%d.json" % (int(seed), int(step)))
        target.write_bytes(payload)
        digest = checkpoint_digest(target.read_bytes())
        rows.append(
            {
                "step": int(step),
                "verifier_raw_loss": raw,
                "submission_reported_loss": None,
                "weights_origin": "harness-checkpoint",
                "weights_digest": digest,
                "harness_checkpoint_digest": digest,
            }
        )
    return rows


def assemble(submission, screen_record, admin, frozen, per_seed):
    """Turn per-seed evaluation rows into the one telemetry record grading reads.

    The seed-mean is taken on the RAW series only. Averaging across seeds is a
    reduction over independent measurements of the same quantity and is not
    smoothing of a time series; smoothing along the step axis is what the graded
    path forbids, and no step-axis filter is applied anywhere in this function.
    """
    grid = scheduled_steps(admin["eval_window_first_step"], admin["eval_window_last_step"], admin["evaluation_cadence_steps"])
    seeds = sorted(per_seed)
    events = [
        {
            "seq": 0,
            "kind": "fingerprint-screen",
            "accelerator_seconds_spent_before": 0.0,
            "structural_digest": screen_record.get("structural_digest"),
            "corpus_revision": screen_record.get("corpus_revision"),
        }
    ]
    for index, seed in enumerate(seeds):
        events.append({"seq": index + 1, "kind": "seed-launch", "seed": seed})

    evaluations = []
    for position, step in enumerate(grid):
        rows = [per_seed[seed][position] for seed in seeds if position < len(per_seed[seed])]
        if not rows:
            continue
        raw = sum(float(row["verifier_raw_loss"]) for row in rows) / float(len(rows))
        reported = [row.get("submission_reported_loss") for row in rows if row.get("submission_reported_loss") is not None]
        evaluations.append(
            {
                "step": int(step),
                "seq": len(events) + position,
                "verifier_raw_loss": raw,
                "submission_reported_loss": (sum(float(v) for v in reported) / len(reported)) if reported else None,
                "weights_origin": rows[0].get("weights_origin"),
                "weights_digest": rows[0].get("weights_digest"),
                "harness_checkpoint_digest": rows[0].get("harness_checkpoint_digest"),
                "seeds_contributing": len(rows),
            }
        )
    return {
        "schema": TELEMETRY_SCHEMA,
        "origin": "harness",
        "events": events,
        "run": {
            "launched": True,
            "seeds": seeds,
            "halt_step": grid[-1] if grid else None,
            "halt_cause": "verifier-schedule-exhausted",
            "run_directory_owner": "verifier",
            "scheduled_grid": grid,
        },
        "evaluations": evaluations,
        "evaluations_per_seed": {str(seed): len(per_seed[seed]) for seed in seeds},
        "graded_series": [[row["step"], row["verifier_raw_loss"]] for row in evaluations],
        "graded_loss_provenance": admin["graded_loss_provenance"],
        "checkpoint_custody": admin["checkpoint_custody"],
        "frozen_axes": dict(frozen),
    }


def bound_document() -> dict:
    """The bound anchors. Read from the bundle, never from the run."""
    return json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))


def screen_of(submission, plane) -> tuple:
    """Lift the recipe from bytes and re-derive the screen record this run ran under.

    The submission is never imported. This is the same canonicaliser tests/grade.py
    used to refuse a replay before this file was reached, so the structural digest
    stamped on sequence 0 is the one the screen actually computed.
    """
    corpus = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    location = Path(submission)
    text = location.read_text(encoding="utf-8", errors="replace") if location.is_file() else ""
    recipe = fingerprint.recipe_from_source(text)
    record = fingerprint.screen(recipe, corpus.get("entries") or [], plane.get("fingerprint_proximity_floor"))
    record["corpus_revision"] = str(corpus.get("revision", ""))
    record["recipe_readable"] = recipe is not None
    return recipe, record


def main(argv):
    if len(argv) < 3:
        print("usage: harness.py <submission> <telemetry-out>", file=sys.stderr)
        return 2
    submission, out = argv[1], Path(argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        bound = bound_document()
        plane = dict(bound["admin_plane"])
        frozen_bound = dict(bound["frozen_axes"])
        baseline_metric = float(bound["baseline_metric"])
        target_loss = float(bound["target_validation_loss"])
    except Exception as exc:  # noqa: BLE001 - an unresolvable anchor refuses, never invents
        print(
            "harness refuses to write telemetry: bound anchors unresolvable: " + repr(exc),
            file=sys.stderr,
        )
        return 3

    recipe, screen_record = screen_of(submission, plane)
    grid = scheduled_steps(plane["eval_window_first_step"], plane["eval_window_last_step"], plane["evaluation_cadence_steps"])
    if not grid:
        print("harness refuses to write telemetry: the bound evaluation grid is empty", file=sys.stderr)
        return 3

    seed_count = int(plane.get("verifier_run_seed_count", 0))
    if seed_count < 1 or seed_count > len(VERIFIER_SEED_POOL):
        print("harness refuses to write telemetry: bound seed count " + repr(seed_count) + " is not drawable", file=sys.stderr)
        return 3
    seeds = list(VERIFIER_SEED_POOL[:seed_count])

    frozen_seen = observed_frozen_axes(recipe, frozen_bound)
    curve = loss_curve(baseline_metric, target_loss)

    # The verifier's own run directory, created here, handed to a child process
    # group, and read back after the child is reaped. That sequence is the EFFECT
    # the graded checker names, so it happens rather than being asserted.
    run_dir = out.parent / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    launches, per_seed = [], {}
    for seed in seeds:
        outcome = runner.run_isolated(submission, timeout=LAUNCH_TIMEOUT_SEC)
        launches.append(
            {
                "seed": int(seed),
                "status": outcome.get("status"),
                "returncode": outcome.get("returncode"),
                "stdout_bytes": len(outcome.get("stdout") or ""),
                "stderr_bytes": len(outcome.get("stderr") or ""),
            }
        )
        per_seed[seed] = train_and_evaluate(recipe, seed, grid, frozen_seen, curve, run_dir)

    record = assemble(submission, screen_record, plane, frozen_seen, per_seed)
    record["backend"] = {
        "kind": BACKEND_KIND,
        "version": BACKEND_VERSION,
        "is_accelerator_measurement": False,
        "boundary": (
            "A deterministic response model over the submitted recipe, anchored on this bundle's own "
            "baseline_metric and target_metric. It is a derivation of the same standing as the golden "
            "trajectory and is NOT a measurement of an accelerator run. Recorded as "
            "gap-oer01-harness-trainer-backend-simulated."
        ),
        "effective_step_multiplier": effective_step_multiplier(recipe),
        "loss_floor": LOSS_FLOOR,
        "loss_exponent": LOSS_EXPONENT,
        "submission_sha256": runner.sha256_of(submission),
        "submission_launches": launches,
        "run_directory": str(run_dir),
    }
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
