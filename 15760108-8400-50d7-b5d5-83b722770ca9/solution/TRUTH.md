# TRUTH: bia S09 multi-objective-frontier

GENERATED SECTION. DO NOT HAND-EDIT. Regenerate with `python3 solution/recompute.py --regen`, whose sole source is `solution/grounding.yaml`.

This file is private. It never reaches the agent.

## What is frozen and what is free

the two objectives, their measurement, the declared tradeoff surface, the data source, the architecture, the token budget, the step ceiling, the per-run wall-clock cap, the number of frontier points, and the hypervolume reference point and normalization.

the optimizer update rule, its internal state, its schedule, its hyperparameters, and the choice of where along the tradeoff surface to place each of the six points.

## The two objectives

**J1, validation loss, lower is better.** mean next-token cross entropy in nats over the frozen validation shard of the frozen Markov source. Measured by `environment/harness/engine.py::measure_val_loss`.

**J2, weight-matrix density, lower is better.** fraction of entries with absolute value at or above density_tau times the root mean square of their own tensor, taken over every two-dimensional parameter tensor, that is both embeddings and the four block matrices of every layer. Measured by `environment/harness/engine.py::measure_density`.

the threshold is relative to each tensor's own root mean square, so uniformly rescaling the weights leaves the objective unchanged, which is what closes the rescaling route that an RMS-normalized residual trunk would otherwise leave open.

## The declared tradeoff surface

capacity removed from the weight matrices cannot be spent on predicting the source, so driving density down raises validation loss and driving validation loss down keeps density high.

at density zero every matrix is null and the model emits a constant, so validation loss equals the uniform predictor exactly, and at density one no entry has been thresholded so the density objective sits at its reference coordinate.

the shape between those two forced endpoints is not declared and is what an attempt discovers.

## The pinned normalization

fraction of the reference-to-ideal distance covered, clipped to the closed unit interval, matching requirements/bia-environment-spec.md lines 82 to 85.

Loss axis. Reference coordinate is natural log of vocab_size, the cross entropy of the uniform next-token predictor. Ideal coordinate is entropy rate of the frozen Markov source plus loss_offset_ideal. The entropy rate is exact, by power iteration to the stationary distribution of the frozen sparse chain followed by the stationary average of the per-state row entropy, in environment/harness/engine.py::source_entropy_rate.

Density axis. Reference coordinate is erfc(density_tau / sqrt(2)), the retained fraction of a dense unstructured Gaussian weight matrix. Ideal coordinate is density_ideal, the declared sparse ideal.

both coordinates of both points are closed-form functions of the frozen bytes, so the normalization needs no measured reference run and cannot drift with the machine it is graded on.

## The pinned hypervolume

Reference point, normalized: `[0.0, 0.0]`. Ideal point, normalized: `[1.0, 1.0]`. Reference point, raw: `['ln(vocab_size)', 'erfc(density_tau / sqrt(2))']`. Ideal point, raw: `['entropy_rate + loss_offset_ideal', 'density_ideal']`.

the area of the union of the origin-anchored boxes of the achieved points, which is at most one by construction.

Reward is `min(max(hypervolume, 0.0), 1.0)`. The engine computes it at `environment/harness/engine.py::hypervolume_sweep, descending u1 strip accumulation` and the verifier recomputes it independently at `tests/checkers/hypervolume.py::hypervolume_grid, coordinate-compressed cell union`, and the DIVERGENCE checker requires the two to agree.

## The degenerate case, stated plainly

A single achieved point collapses the union to one box of area u1 times u2. Because the tradeoff surface forbids both coordinates being near one at once, any single point scores far below a spread set, and a point at either extreme scores near zero: an attempt that only minimizes loss lands near the density reference coordinate, so u2 is near zero and the product is near zero, and an attempt that only minimizes density lands near the uniform predictor, so u1 is near zero and the product is near zero.

## The budget

Per-attempt budget is `0.12` hours, which is 7.2 minutes. 6 hours of single-H100 session time divided by 50 attempts, per requirements/bia-environment-spec.md lines 122 to 135.

A graded attempt evaluates **6 frontier points**. 6 points times 66.0 seconds plus 36.0 seconds of shared overhead equals 432.0 seconds, which is 7.2 minutes.

the frozen runner arms a per-run wall-clock deadline at per_run_seconds, so a point that would overrun is truncated at the deadline with stop_reason recorded, and the whole frontier therefore fits the budget whatever update rule is submitted.

source construction, exact entropy rate, two million training tokens, the validation shard and the data digest were timed on the authoring host's CPU at 7.05 seconds total, which is the only timing in this bundle taken from an execution rather than declared.

UNMEASURED. no GPU execution was performed during authoring, so per-point graded wall clock is an enforced allocation and not a measurement.

## The reference solution

`solution/reference_frontier.py`.

The oracle does not tune one optimizer well. It proposes a family. The base rule is decoupled-moment AdamW, and the free axis carries one addition: a proximal soft-threshold applied after the base step to every two-dimensional parameter tensor, with strength lam. The soft-threshold is the proximal operator of an L1 penalty, so it produces exact zeros rather than merely small values, which is what a scale-invariant density objective can see. lam equal to zero recovers a pure loss-seeking point. Increasing lam walks the declared surface. The six proposed lam values are swept over the interval where the threshold competes with the Adam normalized update, because outside that interval the response saturates and five of the six points would pile into one corner.

Under this reward a submission is judged on the union of six boxes, and the union grows only where a point is not dominated. Spending the whole attempt improving one point therefore buys almost nothing once that point is on the front, while moving a redundant point to an unoccupied part of the surface buys area directly. That is the reasoning the slot is built to require.

## The reference score

**unmeasured, pending orchestrator GPU run.** The host's single H100 was fully committed to a live evaluation campaign for the whole authoring window, so no GPU work was performed. The graded reference score is not estimated, not extrapolated and not inferred from the smoke run. It is absent.

The identical code path was executed at the frozen smoke scale on CPU with CUDA hidden, and it produced a real six-point Pareto front and a real hypervolume. That establishes that the reference executes end to end and that the reward is computable from its output. It establishes nothing about the graded operating point.

## Refusing control, at smoke scale

A negative control was executed at the same smoke scale on the same CPU-only path: six configurations that all minimize validation loss and differ only in a learning-rate multiplier, which is the single-objective habit this slot is built against. It scored 0.001608840 where the reference scored 0.391652516 on the identical fixture. That is a demonstration at smoke scale that the reward separates a spread frontier from a collapsed one and that the grader is not hardcoded to accept. It is not a difficulty measurement, it is not a pilot, and it says nothing about what any cohort would produce.

| Submission at the smoke fixture | Hypervolume |
|---|---|
| reference, six points spread along the surface | 0.391652516 |
| negative control, six loss-only points | 0.00160884 |

## The defeat mechanism, as design intent

Single-objective habits collapse the frontier to a point. An attempt that carries over the reflex of the rest of this corpus, minimize the loss, will propose six configurations that differ only in ways that do not move the second objective, land them all near the density reference coordinate, and score the area of one thin box. An attempt that overcorrects and drives density to the ideal lands at the uniform predictor and scores the area of the other thin box. Reaching a real score requires deciding, before any run, that the deliverable is a spread set and that a point already on the front is worth less marginal effort than a point that is not.

Designed against the pinned cohort gpt5.6-sol, anthropic/claude-opus-5, muse-spark-1.2, valid to 2026-11-17.

UNMEASURED. No pilot has been run against the named cohort over these frozen bytes. No pass probability is authored, no self-solve is claimed, and no difficulty tier is assigned. The bundle ships at HOLD:PILOT_REQUIRED and that is the correct terminal state.

## Graded assertions

| Checker | Kind |
|---|---|
| `frontier_cardinality` | VALUE |
| `frozen_substrate_invariant` | INVARIANT |
| `no_unmanifested_telemetry` | ABSENCE |
| `runs_precede_seal` | ORDERING |
| `training_moved_live_weights` | EFFECT |
| `hypervolume_independently_agrees` | DIVERGENCE |
| `graded_record_authored_by_the_verifier` | VALUE |
| `submission_ran_outside_the_measuring_process` | INVARIANT |
| `frozen_harness_bytes_unmodified` | VALUE |

Rubric compiled weight share is 0.8529 over total weight 1.36, against a floor of 0.75.

## Disposition

`HOLD:PILOT_REQUIRED`. Reasons: `reference-score-unmeasured`, `pilot-absent`, `cohort-registry-digest-unpinned`.
