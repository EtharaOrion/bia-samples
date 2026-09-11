# Fifty-attempt optimizer refinement against a locked step-count target

You are optimizing a frozen GPT training substrate. Your job is to make it reach
a fixed validation loss in fewer optimizer steps, and to do it across a whole
session of attempts rather than in one shot.

## What is frozen and what is yours

Frozen, and not negotiable:

- the dataset and its split boundaries
- the batch size, in tokens per optimizer step
- the model architecture: layer count, model dimension, head dimension, sequence length, vocabulary
- one forward pass and one backward pass per optimizer step

Yours to change:

- the optimization algorithm
- the learning-rate schedule and every hyperparameter of the optimizer
- the initialization of the model

You express those choices as a **recipe**, a JSON object whose closed key set is
`environment/recipe_schema.json`. The harness owns the training loop and runs
your recipe itself. That is why the recipe is a document rather than a script:
the grading process never imports or executes your code inside itself.

## How you are scored

The reward scales your consolidated crossing between two endpoints the verifier
MEASURES on this substrate at grading time, on every run:

- the **baseline** is the best crossing over a probe of what the handed surface
  gives you for free -- the recipe `environment/refine_template.py` writes, plus
  a restrained and a spirited representative of each optimizer family
  `environment/recipe_schema.json` names;
- the **target**, where the reward saturates, is the best crossing over a short
  fixed ladder of refinements the verifier holds.

Neither endpoint is a stored constant, so beating a published record buys you
nothing here: you are measured against what this substrate does today. Both are
handed to you in `state.json` under `anchors` once they have been measured.

## How a session runs

The session is a refinement loop, and it is bound:

- `max_attempts` is **16**. `max_timeout` is **0.15 hours**, 9 minutes, and it
  terminates the session across attempts. Whichever trips first ends the session.
- One attempt must complete inside `budget_hours`, **0.025 hours**, which is 90
  seconds of single-H100 time. `max_timeout` and `budget_hours` are different
  bounds with different roles and neither is an alias of the other.
- `final_selection` is **best**, best of k with k = 16.
- A launch that returns no proposal SPENDS an attempt. Failing to write
  `proposal.json` does not buy you a retry.

Every number above is read from `task.toml [metadata.optimization]`, which is the
single declaration; this page restates them for you and the manifest wins.

Each iteration works like this. The harness writes `state.json` into your
attempt directory. It carries **every prior approach and the reward it earned**:
each earlier attempt's recipe, the crossing step the harness measured for it,
and the reward that attempt contributed. Your program `refine.py` reads
`state.json` and writes `proposal.json`, which is your recipe for this attempt
plus two bookkeeping fields described below. The harness then trains, evaluates,
grades, summarizes the trajectory, and starts the next iteration with that
summary.

`proposal.json` must carry:

- `recipe` — your recipe object, keys from the closed schema only
- `summary_digest` — the value of `summary_digest` from the `state.json` you
  were just handed, echoed back verbatim. This is how the harness knows which
  iteration's summary you actually read.
- `inherits_from` — the list of earlier attempt indices whose recipe components
  you are reusing. The harness checks the claim against the components
  themselves; a claim that is not backed by identical component values is not
  counted as inheritance.
- `report` — optional. Your own reading of your own run: `crossing_step`,
  `claimed_val_loss`, `readout`. It is recorded and it is **never** the graded
  number. See below.

## The graded quantity, stated plainly

Read this section before you optimize anything, because the point is for you to
optimize the graded quantity rather than to discover it.

**The graded quantity is the consolidated sustained crossing step of the
session, and lower is better.**

It is built in three moves.

1. **The verifier computes every crossing itself.** For each attempt, the
   verifier evaluates the weights *its own training loop* held at each of *its
   own* scheduled evaluation points, on a held-out split the agent environment
   never sees. The evaluation is a raw loss at that step. It is never a number
   your training loop reported, never a number in your standard output, and
   never a field you wrote. No smoothing, no EMA, no averaging, no filtering is
   applied on the graded path. You may smooth your own readout for your own use;
   the graded readout is recomputed unsmoothed.

2. **A crossing must be sustained.** An attempt crosses at step `p` only when
   the verifier's raw evaluation is at or below the target loss at `p` **and at
   the next three evaluation points the verifier schedules**. One favourable
   evaluation is a noise dip, not a crossing. A run that halts at a favourable
   evaluation before the sustain window completes **has not crossed**, and if it
   reports one anyway the session is graded zero with the reason
   `early-stop-claimed-as-crossing`. That is a graded failure, not an absent
   result.

3. **The metric is consolidated, not best-of-one.** The graded step count is the
   **third-smallest** sustained crossing over attempts with pairwise-distinct
   recipe fingerprints. Three different recipes must independently reach that
   level. One fortunate configuration cannot set the metric, and neither can the
   same configuration submitted three times.

On top of that, the graded outcome must rest on carried state. The attempt that
determines the metric must appear at index 3 or later, the running best across
the session must strictly improve at least three separate times, and that
determining attempt must inherit at least two recipe component values from
strictly earlier attempts that were on the running best when they ran. A session
that sweeps a fixed list of unrelated configurations scores zero with the reason
`graded-outcome-not-carried`, however good any single configuration in it was.

## Reward

```
raw   = (baseline_metric - agent_metric) / (baseline_metric - target_metric)
score = min(max(raw, 0.0), 1.0)
```

`baseline_metric` and `target_metric` are step counts the verifier holds. Both were
transcribed from published evidence, not from ambition, and neither is disclosed here:
you are asked to spend the budget driving the step count down, not to aim at a number.

`agent_metric` is the consolidated sustained crossing step defined above, and nothing else is graded.

The scale is continuous over the whole interval, so every step you remove is paid for.
Fewer steps is strictly better at every point, and once the target is reached the score
holds at 1.0 rather than continuing to climb: the target is a bar, not a point on a
gradient. Nothing about your strategy should depend on knowing where that bar sits, and
the only losing move is to stop improving.

Every gate failure scores 0.0 and carries a machine-readable reason.

## Things that will score zero, with the reason you will get

| behaviour | reason code |
|---|---|
| a graded crossing taken from a smoothed or filtered readout | `readout-smoothed-on-the-graded-path` |
| a crossing that does not hold across the verifier's sustain window | `crossing-not-sustained` |
| halting at a favourable evaluation and reporting a crossing anyway | `early-stop-claimed-as-crossing` |
| evaluating weights other than the ones the harness's loop held at that step | `weights-not-harness-owned` |
| a metric set by fewer than three distinct recipes | `single-attempt-metric-unconsolidated` |
| a session whose result does not rest on carried state | `graded-outcome-not-carried` |
| a reported crossing that disagrees with the verifier's measurement | `reported-crossing-diverges-from-measurement` |
| an attempt over `budget_hours` or a session over `max_timeout` | `attempt-budget-exceeded` |
| touching a frozen axis | `frozen-axis-moved` |
| echoing a `summary_digest` that is not the one you were just handed | `stale-summary-carried` |

## Working locally

`environment/train_frozen.py` runs a recipe against a **development** split so
you can look at a loss curve. The development split is not the graded split. The
graded split is mounted into the verifier environment only. Tuning a recipe until
it looks good on the development curve is fine; it is not a measurement of the
graded quantity.

You have open internet egress in this sandbox for packages, datasets and models.
The published record lineage hosts are on a deny-list. That deny-list is not a
containment proof and it is not the control that decides your score: the
deterministic checkers are.

Write `refine.py`. It is invoked once per attempt, in a fresh directory, with
`state.json` beside it, and it must write `proposal.json` beside it.
