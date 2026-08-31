# Working surface for slot OER-03

Four files. Two you read, one you copy and change, one you drive.

| File | Role |
|---|---|
| `train_locked.py` | The locked baseline. It reaches the target loss in 3250 optimizer steps. Copy it to `/app/submission.py` and change it. |
| `bia_harness.py` | The pinned harness. Drive it from your loop, one `harness.step(model, i)` per optimizer step. |
| `bia_loader.py` | The frozen data path and the frozen architecture. Read only. |
| `shape.json` | The four frozen axes, with their bound digests. Read only. |

## What is frozen

Dataset, batch size, architecture, and one forward-backward pass per optimizer
step. All four are re-read by the verifier from its own copy of `shape.json` at
grading time, so editing them here does not buy a faster run. It buys a refused
one, scored `0.0` with the reason `frozen-axis-moved`.

## What is free

The optimization algorithm, its hyperparameters, all of its schedules, and the
model initialization.

## What you cannot see, and why that is deliberate

There is no validation split in this container. The loss that decides your score
is computed by the verifier, on a held-out split that never enters here, from the
parameter snapshots `bia_harness.py` wrote to `state_<step>.pt`. There is
therefore nothing here to tune the graded readout against, which is the point.

`harness.report(...)` exists so you can record what your own loop believed. It is
copied verbatim into the grading telemetry and it is never the graded quantity.
If you report a crossing step, the verifier compares it against the one it
recomputes and grades the divergence. Reporting nothing is always safe.

## The one number that scores you

The first cadence point at which the verifier's own unsmoothed evaluation of your
harness-owned parameters falls below the target loss, and stays below it at the
next `SUSTAIN_POINTS` cadence points. `EVAL_STRIDE`, `SUSTAIN_POINTS` and
`TARGET_LOSS` are in `bia_harness.py` and the verifier applies its own copies.

Two consequences worth internalising before you spend an attempt on them.

1. **Training past a sustained crossing changes nothing.** The graded step is the
   crossing, not the schedule length. A 6000-step schedule that crosses at 2600
   scores exactly what a 2900-step schedule that crosses at 2600 scores.
2. **Stopping at a lucky evaluation is not a crossing.** If the run halts before
   the sustain window exists, the graded result is a failure with the reason
   `early-stop-without-sustained-crossing`. It is not a missing result, and it is
   not silently retried.

## Network

The solving surface has open internet egress, so you may fetch packages,
datasets and models. The verifier surface has egress denied. The two surfaces
differ by design. Note that the deny-list narrowing the solving surface is a
convenience boundary and is not a containment proof; the deterministic checker is
the control that actually decides your score.
