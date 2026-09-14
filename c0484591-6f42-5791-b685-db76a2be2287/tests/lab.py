#!/usr/bin/env python3
"""The verifier's own measurement device. It produces the telemetry `grade.py` reads.

WHY THIS FILE IS THE VERIFIER'S AND NOT THE AGENT'S. The graded quantity is
defined as the step at which THE VERIFIER'S OWN evaluation of the model on the
frozen held-out split first falls below the target and stays below it. That
sentence is only true if the verifier computes it, so the held-out surface, the
evaluation schedule, the weight ledger and the novelty verdict all live here, on
the far side of the isolation boundary, and none of them is ever handed to the
solving agent: the agent surface Harbor assembles carries `task.toml`,
`instruction.md` and `environment/` only.

WHAT THE SUBMISSION SEES, AND WHY IT IS ONLY THIS. The submission is never
imported by the grading process. It is driven through `runner.Session`, which
copies it alone into a fresh temporary directory and launches it as a new
session leader under a small environment allowlist. The verifier holds the
objective and hands the child ONE GRADIENT AT A TIME, computed from the weights
the verifier itself reconstructed. No description of the surface being optimized
crosses into the submission's process, so there is nothing there to read instead
of optimize, and the child's only output channel is one parameter-delta row per
step.

WHERE THE HELD-OUT LOSS COMES FROM. Every loss on the graded path is computed
HERE, from the verifier's own reconstruction of the weights, after the child has
exited and its process group has been reaped. Nothing the submission printed,
reported or wrote to disk is read anywhere below.

THE ANCHOR PAIR IS A SURROGATE AND SAYS SO. `BASELINE_METRIC` and
`TARGET_METRIC` are the geometry of the synthetic surface in this file. They are
NOT this family's published anchors: F15's anchors are unmeasured, `task.toml`
declares them absent under `gap-oer-per-family-anchors-unmeasured`, and the
telemetry below carries that bound state forward beside the surrogate pair so a
reader can never mistake one for the other.

Determinism. Every number in this module is a pure function of the frozen
constants, the seed and the deltas the submission produced. There is no clock,
no random source, no network and no locale-dependent formatting anywhere.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import corpus_rules  # noqa: E402
import probe  # noqa: E402
import runner  # noqa: E402
from corpus import loader  # noqa: E402

SURFACE_ID = "oer05-illconditioned-block-descent/v1"

TELEMETRY_SCHEMA = "oer05.verifier_telemetry/v1"

WIDTH = 8

# The block the rule updates, and the held-out minimum it is scored against. The
# curvature spread is what makes the surface ill-conditioned: a rule that scales
# its step by the shape of the block covers the wide coordinates far faster than
# one that spends a fixed step on every coordinate alike.
CURVATURE = (1.0, 2.0, 4.0, 8.0, 1.5, 3.0, 6.0, 12.0)
HOLDOUT_CENTER = (24.0, -36.0, 60.0, -12.0, 30.0, -54.0, 9.0, -21.0)

# The floor of the held-out surface and the span above it, so the loss at the
# initial weights is exactly 4.0 and the bound target of 3.28 sits inside the
# descent rather than at the surface's own asymptote.
HOLDOUT_FLOOR = 3.26
HOLDOUT_SPAN = 0.74

TARGET_LOSS = 3.28

SEEDS = (0, 1, 2)
MAX_STEPS = 6000
EVAL_INTERVAL = 100
SUSTAIN_WINDOW = 4

# The surrogate surface's own geometry, in optimizer steps. NOT this family's
# published anchors; see the module docstring.
BASELINE_METRIC = 4000
TARGET_METRIC = 2600

PROBE_INPUT_SET = "probe-set-b"
WEIGHT_CUSTODY = "harness-owned-digest-bound"
CORPUS_PIN = "pinned-corpus-r1"

GRADED_FIELDS = ("eval_points", "graded_step", "harness_weight_ledger", "probe_transcript",
                 "seed_runs")


def seeded_center(seed: int) -> tuple:
    """The held-out minimum for one seed. A small, fixed, per-seed displacement.

    The displacement is a rank pattern over a fixed modulus rather than a draw,
    so the three seeds are genuinely different problems and all three are
    reproducible by anyone holding these bytes.
    """
    return tuple(
        value * (1.0 + 0.02 * float(((seed * 37 + index * 11) % 5) - 2))
        for index, value in enumerate(HOLDOUT_CENTER)
    )


def _scale(center) -> float:
    return sum(a * (p - c) ** 2 for a, p, c in zip(CURVATURE, initial_weights(), center))


def initial_weights() -> list:
    """The weights every seeded run starts from: the probe set's own starting point.

    Shared with the probe on purpose. It is what lets the first steps of the
    graded run be the very steps the probe replays, which is what
    `check_probe_transcript_bound_to_training_deltas` holds the submission to.
    """
    return [float(value) for value in probe.probe_set(PROBE_INPUT_SET)["params0"]]


def holdout_loss(weights, center, scale: float) -> float:
    """The verifier's own evaluation on the frozen held-out split. Raw, never filtered."""
    residual = sum(a * (w - c) ** 2 for a, w, c in zip(CURVATURE, weights, center))
    return round(HOLDOUT_FLOOR + HOLDOUT_SPAN * residual / scale, 6)


def training_gradient(weights, center, scale: float) -> list:
    """The gradient of the training objective at the weights the verifier reconstructed."""
    return [2.0 * a * (w - c) / scale for a, w, c in zip(CURVATURE, weights, center)]


def weights_digest(weights) -> str:
    payload = json.dumps([round(value, 9) for value in weights], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def grading_tree_digest() -> str:
    """A digest over every byte of the grading tree, taken before and after the run."""
    rows = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        rows.append([str(path.relative_to(HERE)), hashlib.sha256(path.read_bytes()).hexdigest()])
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def probe_pass(submission: Path) -> dict:
    """One probe pass over the submitted rule, in its own isolated child process."""
    spec = probe.probe_set(PROBE_INPUT_SET)
    with runner.Session(submission, WIDTH, spec["params0"], probe.QUANTUM_DECIMALS) as session:
        rows = []
        for grads in spec["grads"]:
            row = session.step(grads)
            if row is None:
                return {}
            rows.append([probe.quantize(value) for value in row])
    return {"transcript": rows, "digest": probe.digest(rows)}


def seeded_run(submission: Path, seed: int) -> dict:
    """One seeded session, driven a gradient at a time, evaluated by the verifier alone.

    The first steps replay the probe's own inputs, so the deltas the harness
    watches at the head of the graded run are directly comparable with the
    deltas the standalone probe recorded. A rule that behaves one way when it is
    being probed and another way when it is being trained separates here.
    """
    center = seeded_center(seed)
    scale = _scale(center)
    witness_grads = probe.probe_set(PROBE_INPUT_SET)["grads"]
    weights = initial_weights()
    eval_points = []
    ledger = []
    witness = []
    with runner.Session(submission, WIDTH, weights, probe.QUANTUM_DECIMALS) as session:
        for step in range(1, MAX_STEPS + 1):
            if step <= len(witness_grads):
                grads = [float(value) for value in witness_grads[step - 1]]
            else:
                grads = training_gradient(weights, center, scale)
            row = session.step(grads)
            if row is None:
                return {}
            if step <= len(witness_grads):
                witness.append([probe.quantize(value) for value in row])
            weights = [w + d for w, d in zip(weights, row)]
            if step % EVAL_INTERVAL:
                continue
            digest = weights_digest(weights)
            raw = holdout_loss(weights, center, scale)
            ledger.append({"step": step, "weights_digest": digest, "custody": WEIGHT_CUSTODY})
            eval_points.append({
                "step": step,
                "loss_raw": raw,
                "graded_loss": raw,
                "loss_source": "verifier-holdout-eval",
                "smoothing": "none",
                "weights_digest": digest,
                "weights_custody": WEIGHT_CUSTODY,
            })
    run = {
        "seed": seed,
        "eval_points": eval_points,
        "harness_weight_ledger": ledger,
        "halt_reason": "completed",
        "last_eval_step": eval_points[-1]["step"] if eval_points else 0,
    }
    run["claimed_graded_step"] = checkers.derive_graded_step(run, TARGET_LOSS, SUSTAIN_WINDOW)
    run["witness_transcript"] = witness
    return run


def novelty(transcript) -> dict:
    """The behavioural verdict, recomputed against the corpus this tree carries right now."""
    rows = loader.signatures(corpus_rules.build, probe, PROBE_INPUT_SET)
    found = probe.nearest(transcript, rows)
    return {
        "corpus_entries": len(rows),
        "nearest_corpus_id": found["nearest_corpus_id"],
        "min_behavioural_distance": found["min_behavioural_distance"],
    }


def measure(submission: Path) -> dict:
    """Drive the submission, evaluate it, and return the telemetry `grade.py` grades."""
    opened = grading_tree_digest()
    pinned = loader.pinned_digest()

    first = probe_pass(submission)
    second = probe_pass(submission)
    runs = [row for row in (seeded_run(submission, seed) for seed in SEEDS) if row]

    witness = runs[0].get("witness_transcript") if runs else []
    for row in runs:
        row.pop("witness_transcript", None)

    verdict = novelty(first["transcript"]) if first else {
        "corpus_entries": 0, "nearest_corpus_id": "", "min_behavioural_distance": 0.0
    }

    return {
        "schema": TELEMETRY_SCHEMA,
        "produced_by": "verifier",
        "surface": SURFACE_ID,
        "target_loss": TARGET_LOSS,
        "eval_interval": EVAL_INTERVAL,
        "anchors": {
            "state": "verifier-surrogate",
            "baseline_metric": BASELINE_METRIC,
            "target_metric": TARGET_METRIC,
            "not_an_anchor": True,
            "surrogate_of": SURFACE_ID,
            "bound_anchors_state": "absent",
            "bound_anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        },
        "drift_state": {
            "anchor_pair": "absent",
            "corpus_pin": CORPUS_PIN,
            "eval_schedule": "eval-every-" + str(EVAL_INTERVAL) + "-steps",
            "grading_tree_mode": "sealed-read-only",
            "probe_input_set": PROBE_INPUT_SET,
            "sustain_window": SUSTAIN_WINDOW,
            "weight_custody": WEIGHT_CUSTODY,
        },
        "graded_metric_provenance": [
            {"field": name, "producer": "verifier"} for name in GRADED_FIELDS
        ],
        "grading_tree": {
            "digest_open": opened,
            "digest_close": grading_tree_digest(),
            "upper_entries": ["logs/run.jsonl"],
        },
        "order_log": runner.order_marks(1) + [{"event": "truth-computed", "seq": 3}],
        "verdict_order": {"corpus_pinned_at": 1, "verdict_taken_at": 2},
        "probe": {
            "probe_set": PROBE_INPUT_SET,
            "behavioural_margin": probe.BEHAVIOURAL_MARGIN,
            "corpus_pinned_digest": pinned,
            "corpus_digest_at_verdict": pinned,
            "pass_a": first,
            "pass_b": second,
            "transcript_digest": str(first.get("digest", "")),
            "harness_deltas_digest": probe.digest(witness) if witness else "",
            **verdict,
        },
        "seed_runs": runs,
    }


def main(argv) -> int:
    """Write the telemetry, or write nothing when there is no submission to measure.

    Writing nothing is a real answer and the only honest one for an absent
    submission: `grade.py` then refuses the run with `telemetry-absent` rather
    than grading a document this module invented on the submission's behalf.
    """
    submission = Path(argv[1]) if len(argv) > 1 else Path("/workspace/update_rule.py")
    destination = Path(argv[2]) if len(argv) > 2 else Path("/logs/verifier/telemetry.json")
    if not submission.is_file():
        return 0
    document = measure(submission)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
