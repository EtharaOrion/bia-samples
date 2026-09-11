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

What this file does NOT own is the accelerator kernel. `train_and_evaluate` is the
one seam where a real training backend plugs in: it must load the frozen shard,
build the frozen architecture, run the submitted recipe for one seed with exactly
one forward-backward pass per optimizer step, checkpoint at each scheduled step,
and evaluate each checkpoint on the frozen held-out split.

That backend is NOT BUILT in this bundle. This lane performed no accelerator run
and this file does not pretend otherwise: with no backend, the harness writes NO
telemetry and exits non-zero, tests/runner.py fails closed with `launched: false`,
and the EFFECT checker attributes the zero to graded-run-produced-no-telemetry.
A harness that fabricated a series here would be manufacturing the measurement the
whole reward rests on. Recorded as gap-oer01-harness-trainer-backend-unbuilt.
"""

import hashlib
import json
import sys
from pathlib import Path

TELEMETRY_SCHEMA = "bia.oer01.telemetry/v1"

BACKEND_ABSENT_REASON = "trainer-backend-unbuilt"


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


def train_and_evaluate(submission, seed, grid, frozen):
    """The accelerator seam. Unbuilt in this bundle; raises rather than inventing.

    A conforming implementation returns a list of rows, one per grid point, each
    carrying: step, verifier_raw_loss (this harness's own unsmoothed evaluation on
    the frozen held-out split), submission_reported_loss (carried, never graded),
    weights_origin, weights_digest and harness_checkpoint_digest.
    """
    raise NotImplementedError(BACKEND_ABSENT_REASON)


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


def main(argv):
    if len(argv) < 3:
        print("usage: harness.py <submission> <telemetry-out>", file=sys.stderr)
        return 2
    print(
        "harness refuses to write telemetry: " + BACKEND_ABSENT_REASON + "; no accelerator backend is built in this bundle",
        file=sys.stderr,
    )
    Path(argv[2]).parent.mkdir(parents=True, exist_ok=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
