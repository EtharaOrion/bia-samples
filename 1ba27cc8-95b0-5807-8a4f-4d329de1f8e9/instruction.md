# Optimizer refinement against a locked step-count target

You are optimizing a frozen GPT training substrate. Your job is to make it reach
a fixed validation loss in fewer optimizer steps. You get **one attempt** in this
run; the refinement happens across a campaign of such runs, and the accounts of
the runs before you are appended to this document.

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
`/workspace/recipe_schema.json`. The harness owns the training loop and runs
your recipe itself. That is why the recipe is a document rather than a script:
the grading process never imports or executes your code inside itself.

## Your files, by absolute path

Everything this task gives you sits directly in `/workspace`:

| path | what it is |
|---|---|
| `/workspace/recipe_schema.json` | the closed key set of a recipe, its bounds and defaults, and the frozen-axis key list |
| `/workspace/shape.json` | the frozen operating point: architecture, batch, `max_steps`, `eval_stride`, `sustain_window`, `target_loss` |
| `/workspace/refine_template.py` | a minimal starting point for the program you must write |
| `/workspace/train_frozen.py` | runs a recipe against the **development** split so you can look at a loss curve |
| `/workspace/bia_recipe.py`, `/workspace/bia_optim.py`, `/workspace/frozen_gpt.py`, `/workspace/bia_data.py` | the substrate the verifier runs, mirrored so you can explore it |
| `/workspace/README.md` | notes on the above |

**You must write your program to `/workspace/refine.py`.** That exact path is
what the harness reads. Writing it anywhere else scores zero as a missing
submission.

Ignore `/workspace/README_TASK.md`, `/workspace/submission/` and
`/workspace/train_gpt_track3.py`. They are baked into the shared base image and
belong to a **different task**. In particular, nothing here ever asks you to
write `submission/optimizer.py`; that instruction is not yours. If those files
contradict this document, this document wins.

## How a run works

**This run gives you exactly ONE attempt, and you are graded on that attempt
alone.** It is a single run inside a campaign of up to 50 such runs. The campaign
is the refinement loop; this run is one step of it.

- Every earlier run's account is appended to this document as a history section
  before you see it: what it submitted, the crossing step the verifier measured
  for it, and the score it earned. That history is your only channel to earlier
  work. Nothing else carries across runs.
- Read that history, form one better recipe, submit it. Improving on the runs
  before you is the entire job.
- Your attempt must complete inside `budget_hours`, **0.12 hours**, which is 7.2
  minutes (432 seconds) of single-H100 time. That is a bound on the *training
  run your recipe asks for*, and it is separate from your own wall clock for
  reading, probing and writing, which the history section states.

  **Size `max_steps` to fit that bound.** The harness's training loop measures
  **0.155 to 0.166 seconds per optimizer step** at this operating point on an
  H100 in bf16, evaluations included, so the frozen ceiling of **2000 steps**
  costs 311 to 332 seconds and fits inside 432 with room to spare. An attempt
  that overruns is not truncated: it scores this run zero with the
  reason `attempt-budget-exceeded`.

  `max_steps` **defaults to the frozen maximum, 2000**, if your recipe omits it,
  and that default now fits the budget. Choose it deliberately anyway.

  **Running SHORTER is not free, and this is the trap on this task.** You are
  graded on the step at which the loss first *sustains* below the target, so a
  run that stops before that step has no crossing at all and forfeits the whole
  upper band. Under a decaying schedule the learning rate anneals over
  `max_steps`, so shortening the run does move the crossing earlier -- right up
  to the point where it moves the crossing past the end of the run and it stops
  existing. Cutting `max_steps` to buy an earlier crossing is the single most
  expensive mistake available here.
- `final_selection` is **best**: the campaign is scored on its best run.

Here is the mechanism. The harness writes `state.json` into your attempt
directory. It carries the shape you may vary, the score anchors, and
`attempts` — the list of prior attempts *inside this run*. That list is
**empty**, because this run has exactly one attempt. Earlier-run information
reaches you through the history section of this document, never through
`state.json`.

Your program `refine.py` reads `state.json` and writes `proposal.json`: your
recipe for this attempt, plus the bookkeeping fields described below. The
harness then trains, evaluates, grades, and writes the account that the next run
in the campaign will read.

`proposal.json` must carry:

- `recipe` — your recipe object, keys from the closed schema only
- `summary_digest` — the value of `summary_digest` from the `state.json` you
  were just handed, echoed back verbatim. This is how the harness knows which
  iteration's summary you actually read.
- `inherits_from` — the list of earlier attempt indices *within this run* whose
  recipe components you are reusing. This run has one attempt, so leave it
  empty (`[]`). It does not describe borrowing from earlier runs in the
  history; that needs no declaration.
- `report` — optional. Your own reading of your own run: `crossing_step`,
  `claimed_val_loss`, `readout`. It is recorded and it is **never** the graded
  number. See below.

## The graded quantity, stated plainly

Read this section before you optimize anything, because the point is for you to
optimize the graded quantity rather than to discover it.

**The graded quantity is the sustained crossing step of your attempt, and lower
is better.**

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
   reports one anyway this run is graded zero with the reason
   `early-stop-claimed-as-crossing`. That is a graded failure, not an absent
   result.

3. **The metric is this attempt's crossing.** You submit one recipe, so the
   graded step count is the sustained crossing the verifier measures for it.
   There is no consolidation and no best-of-k inside this run.

Refinement is measured **across the campaign**, not inside this run: your
crossing is compared against the runs before you in the history. Submitting a
configuration the history already tried, and already recorded a crossing for,
wastes the run — it cannot improve on what is already there.

## Score

The score has **two bands**, and the lower one exists so that progress you make
before you ever reach the target is still paid for.

```
crossed:      score = 0.40 + 0.60 * clip((baseline_metric - agent_metric)
                                         / (baseline_metric - target_metric), 0, 1)

not crossed:  score = 0.40 * clip((no_progress_loss - final_val_loss)
                                  / (no_progress_loss - target_loss), 0, 1)
```

`agent_metric` is your attempt's sustained crossing step, defined above.
`baseline_metric` and `target_metric` are the step counts that set the band. They are
handed to you in `state.json` under `anchors`, so read them there rather than guessing.
Both sit at or below the 2000-step ceiling, so the whole band is reachable.

`final_val_loss` is the verifier's own raw evaluation at your LAST scheduled evaluation
point. `target_loss` is in `shape.json`. `no_progress_loss` is a fixed reference for a run
that learned nothing.

Read the consequence, because it is the whole shape of the task:

- A run that never sustains the target still scores, and it scores **strictly higher the
  lower it drives the validation loss**. Going 5.2 -> 4.9 -> 4.7 is visible progress and
  the number will rise each time. There is no cliff to fall off and no reason to gamble.
- Sustaining the target moves you into the upper band, worth at least 0.40, and crossing
  it **earlier** carries you toward 1.0.
- The two bands meet at 0.40, so nothing is lost at the moment you cross.

Once `target_metric` is reached the score holds at 1.0 rather than continuing to climb:
that anchor is a bar, not a point on a gradient. The only losing move is to stop improving.

Every gate failure scores 0.0 and carries a machine-readable reason. A run that simply did
not cross is **not** a gate failure and is **not** 0.0 — it is the lower band above.

## Things that will score zero, with the reason you will get

| behaviour | reason code |
|---|---|
| a graded crossing taken from a smoothed or filtered readout | `readout-smoothed-on-the-graded-path` |
| a crossing that does not hold across the verifier's sustain window | `crossing-not-sustained` |
| halting at a favourable evaluation and reporting a crossing anyway | `early-stop-claimed-as-crossing` |
| evaluating weights other than the ones the harness's loop held at that step | `weights-not-harness-owned` |
| a reported crossing that disagrees with the verifier's measurement | `reported-crossing-diverges-from-measurement` |
| an attempt whose training run exceeds `budget_hours` | `attempt-budget-exceeded` |
| touching a frozen axis | `frozen-axis-moved` |
| echoing a `summary_digest` that is not the one you were just handed | `stale-summary-carried` |

## Working locally

`/workspace/train_frozen.py` runs a recipe against a **development** split so
you can look at a loss curve. The development split is not the graded split. The
graded split is mounted into the verifier environment only. Tuning a recipe until
it looks good on the development curve is fine; it is not a measurement of the
graded quantity.

You have open internet egress in this sandbox for packages, datasets and models.
The published record lineage hosts are on a deny-list. That deny-list is not a
containment proof and it is not the control that decides your score: the
deterministic checkers are.

## Do this first

**Write a complete, working `/workspace/refine.py` before you run any local
experiment.** A submission that exists and is mediocre scores; a submission that
does not exist scores zero as `submission-missing`, and no amount of local
measurement changes that. `/workspace/refine_template.py` is already a valid
starting point — copy it, pick a defensible recipe, and save it.

Only then explore, and re-save `/workspace/refine.py` each time you improve it.
Treat every local experiment as optional work on top of an already-submitted
answer, because your wall clock for this attempt can end at any point.

## Writing the program

Write `/workspace/refine.py`. The harness copies that one file, alone, into a
fresh temporary directory and runs it there once per attempt as
`python refine.py`, with `state.json` beside it in that directory. It must write
`proposal.json` beside itself, in its own working directory.

Two consequences follow from being copied out alone, and both have cost
submissions before:

- nothing else you leave in `/workspace` is importable from where it runs, so
  `refine.py` must be self-contained. It may import the standard library only.
- it is handed almost no environment. `PATH`, `HOME`, `LANG`, `LC_ALL` and
  `PYTHONHASHSEED` cross; nothing else does.

Read `state.json` from your own working directory, not from `/workspace`.
