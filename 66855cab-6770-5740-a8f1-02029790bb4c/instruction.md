# Reach the target loss in fewer optimizer steps

You are given a working training script that already reaches the target
validation loss. Return one that reaches the same loss in fewer optimizer steps.

## The task

`environment/train_locked.py` is the locked baseline. Copy it to
`/workspace/submission.py` and change it.

How many steps it takes is NOT quoted here, and that is deliberate. The verifier
re-runs that baseline itself on every graded run and measures its crossing, and
that measurement is the baseline your score is scaled against. It also runs one
fixed refinement probe of its own and measures that; the probe's crossing is the
target, where the reward saturates. Neither endpoint is a stored number, so the
bar you have to clear is what this substrate does today rather than a figure
somebody recorded once. The operating point is declared in
`environment/operating_point.json` and the schedule ceiling is its
`max_schedule_steps`.

**Frozen.** Dataset, batch size, architecture, and one forward-backward pass per
optimizer step. All four are re-read by the verifier from its own copy of
`environment/shape.json` and `environment/operating_point.json`, so editing them
produces a run the grader refuses with the reason `frozen-axis-moved`. The two
frozen digests are derived from the executed operating point rather than
authored, so an architecture that moves moves its digest.

**Free.** The optimization algorithm, its hyperparameters, all of its schedules,
and the model initialization.

## What is graded

The graded quantity is `crossing_step`: the **first evaluation point on the
verifier's own cadence at which the verifier's own unsmoothed evaluation of your
harness-owned parameters falls below the target loss, and stays below it at each
of the next `SUSTAIN_POINTS` cadence points**.

Read that sentence twice, because four things follow from it and each one costs
an attempt to discover the hard way.

1. **The verifier computes the loss, not your training loop.** There is no
   validation split in your container. The verifier evaluates the parameter
   snapshots the pinned harness wrote, on a held-out split you never see. No
   number your loop printed enters grading.

2. **The graded readout is raw.** You may EMA-blend, average or window your own
   loss curve to steer your search; that is your business and it is not
   penalised. The graded quantity is recomputed unsmoothed by the verifier, so a
   `smoothed_loss` that dips below the bar is never a crossing.

3. **A single good evaluation is not a crossing.** The bar must still hold at the
   next `SUSTAIN_POINTS` verifier-scheduled evaluation points. A run that halts
   at a favourable evaluation has not crossed, and it is graded a failure with
   the reason `early-stop-without-sustained-crossing`.

4. **Training past a sustained crossing buys nothing.** The graded step is the
   crossing. It is not the `schedule_length`, and a schedule extended past the
   crossing scores exactly what one that stops just after the sustain window
   scores. Spending budget on a longer schedule after you have already crossed is
   spending budget on a number that cannot move.

The declaration below is the machine-readable form of the same sentence.

```yaml
GRADED-QUANTITY: crossing_step
definition: first verifier-scheduled evaluation point whose verifier-computed
  unsmoothed loss is below target_loss and remains below it at the next
  SUSTAIN_POINTS verifier-scheduled evaluation points
computed_by: the verifier, in its own process, on a held-out split absent from the agent container
direction: lower is better
baseline_metric: held by the verifier, not disclosed on this surface
target_metric: held by the verifier, not disclosed on this surface
```

## What happens when a run does not converge

It is graded as a failure, with a reason, and reported to you as such.

A run that never reaches a sustained crossing scores `0.0` and the score document
names why: `crossing-not-sustained` if the bar was touched and not held,
`early-stop-without-sustained-crossing` if the run stopped before the sustain
window could exist, `run-produced-no-verifier-state` if nothing measurable was
produced at all. It is never reported as an absent result, it never falls through
to a default, and it never falls through to silence. The same applies to the
grader: if the verifier itself aborts, it still writes an attributed zero rather
than nothing.

This matters for how you spend your attempts. A failed attempt is information you
were given, not information that vanished. `reported_loss` from your own loop is
never graded; the score document is where the truth about an attempt lives.

## Scoring

One float on the closed interval `[0.0, 1.0]`, higher is better, never binary.

```
raw   = (baseline_metric - crossing_step) / (baseline_metric - target_metric)
score = min(max(raw, 0.0), 1.0)
```

Both anchors are the verifier's and neither is published here, so there is no
number to aim at. What the shape of the formula tells you is enough to plan
against: the score rises continuously as `crossing_step` falls, every step you
remove is paid for at the same rate, and once the target is reached the score
holds at `1.0` rather than continuing to climb. The target is a bar, not a point
on a gradient. `wall_clock` time is not the metric
and is never scored; step count is, because the same recipe yields the same step
count on one accelerator or on eight.

## Your side of the harness

```python
import bia_harness

harness = bia_harness.Harness(run_dir=pathlib.Path.cwd(), schedule_length=SCHEDULE_LENGTH)
for step in range(1, SCHEDULE_LENGTH + 1):
    ...                       # your optimizer, your schedule, your initialization
    harness.step(model, step) # once per optimizer step, after the update
```

`harness.report(...)` is optional. Whatever you put there is carried into the
grading telemetry as your own claim and compared with what the verifier
recomputes. If you report a crossing step the verifier does not find, that
divergence is graded with the reason `reported-crossing-diverges`. Reporting
nothing is always safe.

## Network

Your sandbox has open internet egress; fetch whatever packages, datasets and
models you need. The verifier has none.
