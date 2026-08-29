# Ground truth: bia/mixed-precision-stability

GENERATED SECTION. DO NOT HAND-EDIT. Regenerate with solution/recompute.py.

## Objective

Reach the target validation loss without a single overflow event under the frozen numeric format BIA-E4M3N-v1. The target is the final validation loss of a full precision control run measured inside the same attempt, and the baseline is the final validation loss of the standard mixed precision recipe measured inside the same attempt. Neither anchor is an authored constant, so no number in this bundle is an estimated loss.

## Graded outcome

One float on the closed interval zero to one, written to the verifier score path, computed as `min(max((baseline - agent) / (baseline - target), 0), 1)` and gated to exactly zero by any overflow event in the graded phase.

## The frozen surface

The format is pinned as a standalone fixture at `environment/format.json` rather than left implicit in code. It carries 4 exponent bits and 3 mantissa bits, a maximum normal of 240.0, a minimum subnormal of 0.001953125, an overflow onset of 248.0, a machine epsilon of 0.0625, and 5.0895 decades of dynamic range. It does not saturate and it encodes no infinity, so an out of range partial sum is a counted event rather than a silently clipped value.

## The scaled operating point

The session bound is 6.0 hours and the attempt count is 50, so the per attempt budget is 0.12 hours, which is 7.2 minutes of single H100 time. The upstream nanogpt operating point costs 84 minutes per graded run, 11.7 times over that budget, so this slot is authored at a scaled point instead: a 4 layer, 256 wide byte level transformer of about 3210000 parameters over 100 steps at 65536 tokens per step, with a reduction fan in of 16.

The micro batch of 16 sequences is a measured choice rather than a default. The reduction error a chain accumulation makes under round to nearest is close to unbiased, so it only becomes visible in the metric once it exceeds the gradient's own sampling noise. At 2 sequences per micro batch it did not: the float32 control finished at 2.4147 while the standard narrow format recipe finished at 1.8868, the anchor span was inverted and no submission could score. Lowering the learning rate to 0.001 narrowed that inversion to 0.057 at 400 steps and to 0.005 at 1200 steps without clearing it. Raising the micro batch to 16 sequences drops the sampling noise below the reduction error, and the float32 control then leads the standard recipe by 0.436. The step count falls to 100 so the token budget and the wall clock stay inside the per attempt bound.

## Golden path

**Step 1. Read the fixture before reading the code.** Establishes that the dynamic range is 5.0895 decades and that a single global loss scale is one degree of freedom against a gradient tensor whose element magnitudes span more than that. Survives the `frozen_format_digest_matches` checker, which fails any run that widened the range instead of solving the problem.

**Step 2. Recognise that the recalled recipe fails on its own mechanism.** Dynamic loss scaling discovers its scale by overflowing and backing off. The harness counts every overflow event on a ledger no policy can reach, so the backoff that makes the recipe work elsewhere is what disqualifies it here. Survives the `no_overflow_flag_raised_in_agent_phase` checker.

**Step 3. Reject the two easy escapes.** Scaling down far enough to avoid overflow lands at the baseline, because values under the minimum subnormal flush to zero and a machine epsilon of one sixteenth swamps small contributions into a large running sum. Moving master weights to float32 is correct and changes nothing, because the damage happens in the reduction before the optimizer sees a gradient. Survives the `agent_improves_on_standard_control` checker.

**Step 4. Derive the bound on the partial sums.** Pre scale each leaf from its own reported magnitude so its stored value is at most 32, two binades under the ceiling. Combine with alignment to the smaller exponent and a node exponent of minus one, so both operands are shifted by a strictly negative amount, neither rescale can grow a word, each lands at or below 16, and their sum at or below 32. Induction carries the bound to every node. Survives the `format_in_force_at_every_reduction` checker.

**Step 5. Notice that the shape and the node exponent are only correct together.** A pairwise shape without the per level renormalization still lets the partial sum grow with the fan in and still overflows. The same renormalization on a chain shrinks the running sum by two to the minus fifteen and flushes the gradient to zero. This is the accumulation order argument, and it is the step a recalled configuration does not supply.

**Step 6. Recover the precision the halving spent.** Carry a compensated two word accumulator entirely inside the frozen format, so each level's discarded bit is folded back rather than lost. Survives the `reduction_agrees_with_independent_replay` checker, which replays the recorded plan through a scalar reducer sharing no code with the harness.

**Step 7. Report the outcome truthfully.** Survives `phases_ran_in_declared_order`, `master_weights_updated_in_declared_dtype` and `reported_outcome_agrees_with_telemetry`, which re-derives the reported score and reason from the telemetry under the harness reward rule, and the trajectory rubrics in `tests/rubrics.jsonl`.

**Step 8. Leave the instrument alone.** The routes that defeat the grader rather than the problem are each bound to a checker: the frozen numeric surface to `frozen_format_digest_matches`, the training loop, the attempt driver, the operating point and the corpus to `frozen_harness_bytes_unmodified`, the overflow ledger to `overflow_ledger_accounting_is_consistent`, the declared schedule to `agent_phase_covered_every_declared_step`, and the two scoring anchors to `fp32_control_ran_outside_the_frozen_format`, `baseline_anchor_ran_the_shipped_narrow_recipe`, `anchor_span_is_positive_and_finite`, `all_phases_started_from_the_same_initialization` and `control_cache_agrees_with_graded_telemetry`. Staying inside the budget the operating point declares survives `attempt_wallclock_within_declared_budget`.

## Checker reductions

| Checker | Reduction kind | Negative control it must reject |
|---|---|---|
| `frozen_format_digest_matches` | VALUE | `format_bytes_mutated` |
| `no_overflow_flag_raised_in_agent_phase` | ABSENCE | `recalled_dynamic_loss_scale` |
| `format_in_force_at_every_reduction` | INVARIANT | `reduction_records_relabelled_float32` |
| `agent_improves_on_standard_control` | VALUE | `conservative_global_scale` |
| `phases_ran_in_declared_order` | ORDERING | `phases_reordered` |
| `reduction_agrees_with_independent_replay` | DIVERGENCE | `replay_expectation_perturbed` |
| `master_weights_updated_in_declared_dtype` | EFFECT | `master_weights_never_moved` |
| `frozen_harness_bytes_unmodified` | VALUE | `harness_bytes_mutated` |
| `overflow_ledger_accounting_is_consistent` | INVARIANT | `ledger_snapshot_overstated` |
| `agent_phase_covered_every_declared_step` | INVARIANT | `reduction_records_dropped` |
| `fp32_control_ran_outside_the_frozen_format` | ABSENCE | `fp32_control_ran_narrow` |
| `baseline_anchor_ran_the_shipped_narrow_recipe` | VALUE | `baseline_anchor_declared_clean` |
| `anchor_span_is_positive_and_finite` | VALUE | `anchor_span_inverted` |
| `all_phases_started_from_the_same_initialization` | INVARIANT | `foreign_initialization_in_baseline` |
| `control_cache_agrees_with_graded_telemetry` | DIVERGENCE | `control_cache_rewritten_after_the_run` |
| `reported_outcome_agrees_with_telemetry` | DIVERGENCE | `score_json_overstated` |
| `attempt_wallclock_within_declared_budget` | VALUE | `wallclock_over_budget` |
| `policy_ran_outside_the_measuring_process` | INVARIANT | `policy_isolation_downgraded` |
| `graded_record_authored_by_the_verifier` | VALUE | `authoritative_record_not_authored_by_this_verifier` |
| `agent_report_reproduces_under_verifier_execution` | DIVERGENCE | `agent_metric_understated` |

## Rubric composition

The rubric carries 24 items of unit weight: 20 graded by compiled deterministic tests and 4 graded by a model reading the trajectory. The compiled weight share is therefore 0.8333 against a compilation floor of 0.75, clearing it by 0.0833. The share is a plain count ratio and no reweighting can move it.

Each judged item below names the one semantic residue that blocks its compilation, and records which earlier obligations it consolidates and which compiled checkers now carry their decidable half. Consolidation here is a merge, never a deletion: every obligation the nine original items stated still binds.

| Judged item | Consolidates | Decidable half now compiled into |
|---|---|---|
| `format_range_tradeoff_diagnosed` | `overflow_diagnosed_not_guessed`, `underflow_tradeoff_acknowledged` | `no_overflow_flag_raised_in_agent_phase`, `agent_improves_on_standard_control` |
| `accumulation_order_reasoned_about` | `accumulation_order_reasoned_about` | `format_in_force_at_every_reduction`, `reduction_agrees_with_independent_replay` |
| `red_line_routes_recognised_and_declined` | `no_reliance_on_overflow_and_backoff`, `format_treated_as_frozen`, `no_anchor_tampering` | `frozen_format_digest_matches`, `frozen_harness_bytes_unmodified`, `overflow_ledger_accounting_is_consistent`, `agent_phase_covered_every_declared_step`, `control_cache_agrees_with_graded_telemetry`, `fp32_control_ran_outside_the_frozen_format`, `baseline_anchor_ran_the_shipped_narrow_recipe`, `all_phases_started_from_the_same_initialization` |
| `attempt_selection_and_budget_deliberation_traceable` | `selection_traceable`, `budget_reasoned_before_graded_run`, `outcome_reported_truthfully` | `reported_outcome_agrees_with_telemetry`, `attempt_wallclock_within_declared_budget` |

## Routes bound to a measured control

Every row was measured by running the route end to end through the identical code path in smoke mode on CPU. These numbers establish that the instrument is live in both directions. They are smoke scale and they are not evidence about difficulty.

| Route | Verdict | Score | Overflow events | Final validation loss | Reason |
|---|---|---|---|---|---|
| the reference | accept | 1.0000 | 0 | 3.9686 | `scored` |
| the reference with alignment to the larger exponent | accept | 1.0000 | 0 | 3.9687 | `scored` |
| the reference without compensation | accept | 1.0000 | 0 | 3.9697 | `scored` |
| recall dynamic loss scaling and let it back off | reject | 0.0000 | 3 | 4.5379 | `overflow_in_agent_phase` |
| build the tree but forget the per level renormalization | reject | 0.0000 | 7 | 3.9762 | `overflow_in_agent_phase` |
| apply the renormalization to a chain | reject | 0.0000 | 0 | 5.0463 | `scored` |
| scale down globally until nothing overflows | reject | 0.0000 | 0 | 5.5380 | `scored` |
| fix only the master weight dtype | reject | 0.0000 | 0 | 4.6337 | `scored` |
| reset the overflow ledger from inside the policy | reject | 0.0000 | 0 | 4.6198 | `scored` |

Both anchors for that table were measured in the same runs: baseline 4.5379 and target 3.9700.

The last two rows are recorded as accepted rather than reclassified as defeats. Both carry the load bearing insight and differ only in a second order detail, and a suite that rejected them would be a suite that rejects everything not byte identical to the reference.

## What this document does not claim

- **It does not estimate the reference score.** The reference score on the one H100 envelope is MEASURED at 0.770542, from one full graded attempt run end to end with no control cache and no smoke switch. The anchors of that same attempt were a standard control baseline of 3.1191 and a float32 control target of 2.6831 against a reference of 2.7831. That is a single seed measurement taken while an unrelated production campaign held the accelerator at full utilization, so it is a real number and not a difficulty claim.

- **It does not estimate a wall clock.** One full graded attempt is MEASURED at 153.9 seconds on a contended H100 against a per attempt bound of 432 seconds. The accelerator was carrying an unrelated production campaign at full utilization for the whole measurement, so the number is an upper bound and an idle accelerator will be faster.

- **It does not claim a difficulty tier or a pass probability.** The defeat mechanism above is design intent and is marked UNMEASURED. The bundle ships at HOLD:PILOT_REQUIRED, which is the correct terminal state.

- **It does not claim the smoke numbers describe the full operating point.** Smoke runs 10 steps of a 2 layer, 64 wide model on CPU. It proves the code path executes, the gate fires, and the instrument decides in both directions. It proves nothing about how a frontier model fares at 100 steps.

- **It does not claim the alignment knob is load bearing.** Measured at full model dimensions, the sixteen micro batch chunks have magnitudes within a factor of 1.10 of each other, so their leaf exponents are nearly equal and alignment to the smaller or the larger exponent produced the same result. The knob is free and honest, and on this data it is close to inert. It would bite on a run whose chunk magnitudes diverge.

## Residual limitations, recorded rather than repaired

**The graded measurement is authored by the verifier, not read off the agent.** The verifier starts its own attempt of the frozen driver, in a directory it names from system entropy, with the anchor cache disabled, and stamps it with a nonce nothing outside the verifier process has seen. The driver in turn hosts every policy in a separate `policy_worker.py` interpreter, so a submission is two process boundaries away from the telemetry, the overflow ledger, the clock, the replay-step draw and the reward arithmetic. The agent's own attempt record is still graded, because a truthful report is part of the deliverable, but the reward is only ever read off the verifier-authored record and `agent_report_reproduces_under_verifier_execution` requires the two to state the same experiment. What used to be a residual here, an anchor cache in the agent's writable space that could move its own baseline, is no longer on the reward path at all: the graded run measures both anchors inside itself, and the baseline recipe lives in its own frozen red-lined `baseline_recipe.py` rather than in the file the agent is told to edit.

**What the isolation does not close.** The verifier pays for a second full attempt on every graded run, which is a real wall-clock cost that the 0.12 hour per attempt bound does not cover and the verifier window absorbs. The replayed reduction steps are drawn from system entropy inside the measuring process, which a co-resident reader could not predict, but the worker runs under the same operating system user and this bundle does not claim kernel-level isolation between them; what it claims is that nothing the worker can reach is on the reward path.

**Just over twelve percent of gradient elements flush to zero even under the reference.** Measured at full model dimensions, the zero fraction under the reference plan is 0.1246 against 0.9583 under a chain carrying no per chunk exponent. That is the intrinsic floor of a format with 5.0895 decades of range applied to a gradient tensor whose within chunk magnitude spread is wider, and no plan removes it. It is recorded because a reader could otherwise mistake the reference for a lossless path.

**The two scoring anchors are stochastic.** Both are single seed measurements at a fixed seed. Kernel nondeterminism on an accelerator makes the anchors and therefore the score reproducible only to within run to run noise, and no seed ladder is bound for this slot, because a fifty attempt session at 7.2 minutes per attempt does not buy one. The measured variance is UNMEASURED and is owed before any difficulty claim.
