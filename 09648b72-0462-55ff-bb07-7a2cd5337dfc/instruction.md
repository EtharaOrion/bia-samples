# Data mixture search under a fixed token budget

You are given a pool of text sources and a fixed budget of training tokens. A
frozen estimator is trained on exactly that many tokens and then evaluated on a
held-out split you never see. Your job is to decide where those tokens come
from.

## What you produce

Exactly one file, `plan.json`, in your working directory. Its grammar is in
`environment/plan_schema.md`. Nothing else you write is read by the grader.

## The graded quantity, stated plainly

The graded quantity is the **validation loss of the trained estimator on the
frozen held-out split, recomputed by the verifier from the weights the harness
owns at the bound evaluation point**. Lower is better.

Everything in that sentence is load-bearing, so it is spelled out:

- **Recomputed by the verifier.** The graded loss is not a number your process
  reported. It is not a number in your stdout. It is not a field you wrote. The
  verifier executes your plan itself, holds the resulting count table itself,
  and evaluates it itself.
- **Unsmoothed.** The graded readout is the raw evaluation. You may EMA,
  average or otherwise filter your own reported loss for your own use and write
  it into `report.json`; nothing there is graded. Declaring smoothing on the
  graded readout in `environment/graded_readout.json` does not change what the
  verifier computes, and a declared value other than `none` is graded as
  readout smoothing and scores zero.
- **At the bound evaluation point.** The bound evaluation point is budget
  exhaustion: the moment the harness has fed exactly the frozen token budget. A
  run that stops feeding before that point has not established a loss. It is
  graded as an early stop and scores zero with that reason. It is not graded as
  an absent result.
- **Sustained across the verifier's own folds.** The verifier splits the frozen
  held-out split into two folds and evaluates on both. An improvement that
  holds on the whole split but fails on either fold is a fold artifact, not a
  loss, and scores zero with that reason.
- **From the weights the harness owns.** You cannot hand the verifier a
  checkpoint. The count table the verifier evaluates is the one its own
  executor accumulated from the batches it fed.

## What is frozen

- **The token budget.** `corpus_spec.json` names it. A graded run feeds exactly
  that many tokens, counted by the harness from the batches it fed, never from
  a number your loop claimed. Feeding more is an overspend and scores zero.
  Feeding fewer is an early stop and scores zero.
- **The model.** A 12-symbol add-alpha unigram estimator, alpha fixed.
- **The optimizer.** Closed-form count accumulation. There are no steps to tune.
- **The evaluation split.** Held by the verifier. It is not in the pool, no
  document id in the pool resolves to it, and any pool document whose bytes are
  a held-out document's bytes is a leak that scores zero. If a source in your
  plan turns out to carry held-out bytes, that is your plan's problem to have
  avoided, not the grader's to forgive.

## What is free

The allocation of the fixed token budget across sources.

That is the whole free variable, and it is worth reading it exactly as written.
It constrains the *total* token count. It says nothing about the allocation
being one number per source, nothing about a source being indivisible, and
nothing about a document being consumed at most once. `environment/mixture.yaml`
hands you a constant-weight-per-source template because that is the
parameterisation the literature hands you. It is a template, not the boundary of
the search space, and the harness accepts a strictly wider grammar that is
documented in full in `environment/plan_schema.md`. If the best constant-weight
mixture is not good enough, the constraint you are hitting is the template's,
not the task's.

`environment/harness.py` is an agent-side simulator over the same pool and an
agent-visible probe split drawn from the same generating distribution as the
graded split. Use it to measure your own candidate mixtures. It is not the
grader and it does not hold the graded split.

## How you are scored

One float in `[0.0, 1.0]`, higher is better, never binary.

Every integrity condition above is a gate: failing one scores `0.0` and the
score document names the reason. When every gate passes, the score is a
continuous function of your recomputed validation loss, normalised between two
calibration points the verifier recomputes for itself from the frozen substrate:
the optimum reachable by the constant-weight template, and the optimum reachable
over the full plan grammar. Matching the constant-weight optimum scores `0.0`
with a reason. Matching or beating the full-grammar optimum scores exactly
`1.0`.

The family anchors for this task family are unmeasured, so `baseline_metric` and
`target_metric` are declared absent rather than invented, and the calibration
points above are substrate-local and are not those anchors. `task.toml` records
both facts and the gap ids that carry them.
