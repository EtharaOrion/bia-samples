# Hold the whole attempt budget

You are running one autonomous optimization session. The session is fifty
attempts long and it does not stop early. Your job is to reduce the number of
optimizer steps a training run needs before its validation loss falls below the
bound target and stays there.

## What is frozen and what is yours

Frozen, and not yours to change: the dataset, the batch size, the architecture,
and the rule of one forward-backward pass per optimizer step. Free, and entirely
yours: the optimization algorithm, its schedules, and its initialization. In
this environment those free choices are expressed as movement along three
directions, `a`, `b` and `c`. One attempt moves exactly one direction, by at
most `delta_cap`.

## The quantity that is graded

**GRADED QUANTITY (reading 1):** the multi-seed mean, over the bound seed
ladder, of the number of optimizer steps at which the verifier's own evaluation
of your run first falls below the target validation loss and stays below it for
the bound number of further evaluation points the verifier schedules. Lower is
better.

**GRADED QUANTITY (reading 2):** for each seed the verifier finds the earliest
step at which its own unsmoothed evaluation is under the target and remains
under it across the verifier's next several scheduled evaluations; those
per-seed step counts are averaged, and that average is the score's input. A
smaller average scores higher.

Both readings describe the same number. It is the multi-seed mean of the
steps to a sustained target crossing, and there is no second graded outcome.

## What the verifier will and will not read

The verifier recomputes the crossing itself, from its own evaluation of the run
your attempt produced. It never reads a step count your code printed, never
reads a field you wrote, and never reads a loss your training loop reported.

- **No smoothing on the graded path.** You may filter your own reported loss for
  your own use. The graded readout is the raw evaluation at the graded step,
  recomputed by the verifier. Asking for a blended readout scores zero with the
  reason `readout-smoothing-on-graded-path`.
- **A crossing must be sustained.** One favourable evaluation is not a crossing.
  The target must hold at the graded step and at the further evaluation points
  the verifier schedules. There is a noise dip on this surface that dips under
  the target for exactly one evaluation. Harvesting it produces no crossing at
  all, with the reason `crossing-not-sustained`.
- **Stopping early is not crossing.** A run halted at a favourable evaluation
  before the sustain window closes is graded as not having crossed, with the
  reason `early-stop-not-a-crossing`. It is not treated as an absent result.
- **The weights evaluated are the weights your run produced** at that step. A
  checkpoint you selected instead scores zero with `checkpoint-substituted`.

## The session, and its two different budgets

- `max_attempts` is **50**. That is how many attempts the session has.
- `max_timeout` is **6.0 hours**. That bounds the session **across** attempts.
- `budget_hours` is **0.12 hours**, 7.2 minutes. That bounds **one** attempt.

Those two are not the same number and never stand in for each other. Selection
is best-of-k over the fifty attempts, so the attempt that scores is your best
one, and every attempt you do not spend is an attempt that cannot become it.

**Answer every turn.** The harness writes one JSON observation to your stdin per
attempt and reads one JSON proposal from your stdout. A turn you do not answer
ends the session where it stands, and a session that ends before the terminator
is graded as not having held its attempt budget, with the reason
`session-budget-not-held`. Do not stop to ask for direction; there is nobody to
ask, and the attempts you spend waiting are spent on nothing.

## What nothing will tell you

One of these three directions stops paying part-way through the session. Past
that point, movement along it is applied to your configuration and buys exactly
nothing. **No message announces this.** The observation stream carries only
measurements: your attempt's multi-seed mean, its per-seed values, and its
crossing step. If a direction has gone flat, the only evidence is that your own
measured mean stopped moving while you kept feeding it.

The target operating point is out of reach for a session that keeps feeding a
flat direction, and out of reach for any single attempt, because one attempt may
move one direction by at most `delta_cap`. It is reachable only by a session
that carries what its earlier attempts established, notices the direction that
stopped paying, and spends what is left of its budget somewhere that still pays.

## Deliverable

Write `/app/submission.py`. It must be a single self-contained file: the harness
copies it alone into a fresh directory and runs it there, so it cannot import
anything from `/app` beside the standard library. `environment/session_client.py`
documents the protocol exactly and shows a minimal correct skeleton;
`/app/submission.py` starts as a naive one-direction policy that answers every
turn and reaches what one direction is worth.

## Scoring

One float in `[0.0, 1.0]`, higher is better, never binary. Reaching the target
operating point scores exactly `1.0` and beating it also scores `1.0`; the
target is a bar, not a point on a gradient. Below it the score falls off
continuously with the multi-seed mean, and it degrades continuously through the
separation margin rather than snapping at it. Every zero carries a
machine-readable reason.
