# TRUTH: BIA-GSN-1, kernel level throughput

GENERATED SECTION. DO NOT HAND-EDIT.

Source of truth is `solution/grounding.yaml`. Regenerate with `python3 solution/recompute.py generate --root .`. This file is private and never crosses the agent visible boundary.

## What is frozen and what is free

Frozen is the mathematical output. The operator is a fixed sequence of correctly rounded IEEE-754 binary32 primitives over a reduction whose order is part of the specification, and the graded comparison is on raw 32 bit patterns. Free is the implementation: the chunking, the memory layout, the kernel structure, the launch count, and the library used to express it. The reward is measured wall clock under that correctness gate.

The per row operator, in order:

    s = tree_sum(x[i] * x[i])
    m = s * (1 / C)
    d = sqrt(m + eps)
    n = x[i] / d
    y = n * g
    y = y + b
    h = max(y, y * alpha)
    p = h * h
    q = h + p * beta
    e = tree_sum(q * q)
    r = sqrt(e * (1 / C))

Constants are eps = 0.0001220703125, alpha = 0.5, beta = 1.5, each exactly representable in binary32 so no constant carries an implicit rounding decision. The permitted primitives are add, subtract, multiply, divide, sqrt, maximum. The excluded primitives are matmul, exp, log, tanh, erf, rsqrt, fused_multiply_add, and each is excluded because its result is either not correctly rounded or not order determined.

Every primitive in the list above is correctly rounded under IEEE-754 binary32, so the value of each output element is fixed by the op sequence alone and does not depend on the device, the backend, the memory layout, or the chunking. The two reductions are explicit trees, so the one degree of freedom a library reduction would leave open is closed by the specification.

## The ordered path through instruction.md

### read_the_operator

Action: read environment/reference_impl.py and environment/spec.json and establish that the operator is a fixed sequence of correctly rounded primitives over a specified reduction tree.

Established state: the submission contract and the frozen surface are known.

Survives: the temptation to treat the chunk loop as part of the operator.

Satisfied checker: `invariant_frozen_surface`.

### reproduce_the_output

Action: implement forward so that every element follows the same op sequence, then confirm bitwise agreement on every fixture including the tiny, huge, sparse, and signed zero cases.

Established state: the submission is bitwise identical to the reference.

Survives: a numerically close implementation that would pass a tolerance based gate.

Satisfied checker: `divergence_bitwise`.

### materialize_the_declared_outputs

Action: return Y of shape (N, C) float32 and R of shape (N,) float32 so the harness can write the declared artifacts.

Established state: the artifact tree carries the declared byte lengths.

Survives: a partial or reshaped return that would still compare equal on a prefix.

Satisfied checker: `effect_outputs_materialized`.

### keep_the_numeric_posture

Action: leave every backend precision flag as the grader set it and open no autocast context.

Established state: the process is still in the strict numeric posture at checker time.

Survives: an import side effect that would relax precision globally.

Satisfied checker: `absence_relaxed_precision`.

### remove_the_implementation_overhead

Action: replace the chunk loop with a single pass, keep the contiguous layout, and hoist the loop invariants, changing which elements share a kernel launch and changing no element value.

Established state: the submission computes the same bytes with fewer passes over memory.

Survives: the absence of any precision or reduction shortcut.

Satisfied checker: `divergence_bitwise`.

### complete_the_measurement

Action: let the harness run the declared warmups and timed trials for both implementations in the declared within round order.

Established state: the event log carries a complete and correctly ordered protocol.

Survives: a partial measurement presented as a full one.

Satisfied checker: `ordering_measurement_protocol`.

### accept_the_recomputed_reward

Action: let the grader derive the reward from the logged durations rather than from any number the submission reports about itself.

Established state: the recorded reward is reproducible from the raw durations.

Survives: a tampered summary.

Satisfied checker: `value_reward_recomputed`.

## Rejected routes

Each rejected route names the checker that rejects it and the known wrong control whose rejection the control runner measured. A route with no measured control is not listed.

| Route | Rejected by | Control |
|---|---|---|
| cast to float16 or bfloat16 for the elementwise chain | `divergence_bitwise` | `c_float64_roundtrip` |
| enable TF32 or a fast math path | `absence_relaxed_precision` | `c_tf32_override` |
| replace the specified tree with a library reduction | `divergence_bitwise` | `c_perturbed` |
| replace the divide by d with a reciprocal square root | `divergence_bitwise` | `c_perturbed` |
| contract n * g and + b into a fused multiply add | `divergence_bitwise` | `c_perturbed` |
| make the reference slower to inflate the ratio | `invariant_frozen_surface` | `c_frozen_edit` |
| return a lazy handle so the timed region does not cover the work | `ordering_measurement_protocol` | `c_ordering_truncated_log` |
| edit the event log or the timing summary | `value_reward_recomputed` | `c_value_tampered_summary` |

On cast to float16 or bfloat16 for the elementwise chain: the divergence gate is on bit patterns, so any change of working precision fails whether it loses accuracy or gains it.

On replace the specified tree with a library reduction: a library reduction picks its own accumulation order, which is exactly the degree of freedom the specification closes.

On contract n * g and + b into a fused multiply add: this is the route a compiler takes without being asked, which is why torch.compile with default settings is unsafe here.

On return a lazy handle so the timed region does not cover the work: every timed region is bracketed by a device synchronize, and the ordering checker reads the log that records those regions.

## Reward

The metric is median over rounds of the per round ratio of total time, submission against frozen reference, each hosted in its own process on the same device within the same graded run, where a round total is the sum of its timed trials plus its barrier. Direction is higher is better. The formula is `score = min(max((speedup - 1.0) / (32.543 - 1.0), 0.0), 1.0)`.

baseline_metric is 1.0 by construction, because the baseline is the frozen reference measured against itself inside the same run, so no pre measured wall clock constant is baked into the bundle and the reward is portable across machines. target_metric is MEASURED and is no longer an authored design bar. It is derived from the speedup solution/fast_impl.py actually earns at the graded operating point on one H100, by the rule target_metric = 1 + 2 * (reference speedup - 1), so the clamp is reached only by a submission that doubles the reference's margin over the frozen baseline and the reference itself lands at exactly one half. One half is chosen because it is the point of the clamped interval that leaves the same amount of rankable range below the reference as above it.

## Measured reference calibration

The reference solution was measured on one NVIDIA H100 80GB HBM3 at the graded profile, operating point 15 rounds, 8 warmup calls and 40 timed calls per implementation per round, workload 32768 rows by 2048 columns float32, over 8 repeats. The median speedup is 16.7715, the minimum is 16.4527, the maximum is 17.3337, the standard deviation is 0.2506, and the relative standard deviation is 1.49 percent.

The target is derived rather than chosen: `target_metric = 1 + 2 * (16.7715 - 1) = 32.543`. At that target the reference scores 0.5, which is interior to the clamped interval on both sides, so a submission faster than the reference earns a strictly higher score and a submission slower than the reference earns a strictly lower one.

Seven sibling lanes shared the same H100 during this calibration, so every repeat was gated behind a contention probe that times solution/fast_impl.py on the graded workload and admits the run only when that time is at the uncontended floor of 2.75 milliseconds. Repeats taken while the card was contended were not used for calibration, and they are recorded separately under contended_observation because discarding them silently would be the wrong report.

On a contended card the same reference measured 3.65 to 13.28 rather than 16.45 to 17.33. The ratio is not contention neutral even though it is taken back to back on one device, because the reference is dominated by per launch overhead across its 128 row chunks while the submission is dominated by memory bandwidth in a single pass, so a co-tenant steals bandwidth from the submission faster than it steals launch slots from the reference. The graded envelope is one H100 per attempt, so the uncontended figure is the one the reward is calibrated against.

## Scaled operating point

A session is 6 hours across 50 attempts, so one graded attempt gets 7.2 minutes of single H100 time, which is budget_hours 0.12. The source is requirements/bia-environment-spec.md lines 122 to 135.

The graded measurement phase is bounded at 240 seconds by a hard guard inside the timing harness, which is 4.0 of the 7.2 minutes a graded attempt is allowed. The remaining 3.2 minutes cover container start, module import, workload allocation, the correctness gate, and the agent's own edit and self check loop. The harness aborts the measurement and scores exactly 0.0 with reason measurement_budget_exceeded rather than overrunning the per attempt bound.

MEASURED. One graded attempt against the reference solution completes in 38.50 seconds of median wall clock on one uncontended H100 over 8 repeats, with a range of 38.03 to 39.54 seconds. That is 8.9 percent of the 432 second per attempt bound and 16.0 percent of the 240 second measurement guard, so the guard is preserved intact and still aborts rather than overruns.

The round count was raised from 5 to 15 because a graded attempt at 5 rounds used only 14.79 seconds of the 432 seconds an attempt is allowed. Across 8 uncontended repeats of each setting on the same card, the run to run standard deviation of the reported speedup fell from 0.4665 at 5 rounds to 0.2506 at 15 rounds, which is 2.81 percent of the estimate down to 1.49 percent, and the full observed range fell from 1.4450 to 0.8810. The observed ratio of standard deviations is 1.86 against the 1.73 that tripling an independent sample count would predict. The cost is a median graded wall clock of 38.50 seconds against 14.79, which leaves 11.2 times the per attempt bound unspent and 6.2 times the measurement guard unspent.

## Checker reduction

| Checker | Kind | Live state read |
|---|---|---|
| `divergence_bitwise` | DIVERGENCE | the raw bytes of the submission_out and reference_out artifact trees on the filesystem, each written by a different process, plus the end of round output bytes and the barrier values of both timing workers |
| `effect_outputs_materialized` | EFFECT | os.stat over the declared submission artifacts against the invocation marker mtime |
| `absence_relaxed_precision` | ABSENCE | each worker process environment read out of /proc by the grader, the grader own torch backend precision flags, and the flags and live environment table each worker reports |
| `invariant_frozen_surface` | INVARIANT | sha256 of every frozen path recomputed at checker time against the manifest and the start of run snapshot |
| `ordering_measurement_protocol` | ORDERING | the append only harness event log the grader wrote, including the per round barrier and end of round dump events |
| `value_reward_recomputed` | VALUE | the per trial durations and the per round barrier durations in the event log, and the recorded timing summary |

Every checker above reduces to exactly one kind. None reduces to two and none reduces to zero.

## Controls and both halves

| Control | Class | Targets |
|---|---|---|
| `clean_reference` | accepting | accepting half |
| `c_perturbed` | known_wrong | divergence_bitwise |
| `c_float64_roundtrip` | known_wrong | divergence_bitwise |
| `c_truncated_output` | known_wrong | effect_outputs_materialized |
| `c_tf32_override` | known_wrong | absence_relaxed_precision |
| `c_frozen_edit` | known_wrong | invariant_frozen_surface |
| `c_ordering_truncated_log` | known_wrong | ordering_measurement_protocol |
| `c_value_tampered_summary` | known_wrong | value_reward_recomputed |

Run them with `python3 solution/control_runner.py --bundle .`, which stages a throwaway copy of the bundle, runs the smoke profile on CPU, and reports each control as accepted or rejected. It allocates no accelerator memory.

## Defeat mechanism

The model must produce a numerically identical result faster, and the DIVERGENCE checker makes approximation worthless.

Status: DESIGN INTENT, UNMEASURED. No pass probability is authored here, no self solve is claimed, and no difficulty tier is assigned. The bundle ships at HOLD:PILOT_REQUIRED, which is the correct terminal state for a bundle with no external signed pilot.
