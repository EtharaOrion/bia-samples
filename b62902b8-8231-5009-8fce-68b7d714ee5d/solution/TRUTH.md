# FORGE-CANARY-BEGIN
# 0: b04e4a5c75e463923f19d10b43e8b9bca6a55fa9bccb535d086a1e4d80d4e903
# 1: d98ea3702902cfca37e055de987a017c6730fab0d1c5a5fbf5c593702f9fc9fc
# 2: cbd9f7d3d5d4f51891c7361b97fa1d4e2534640e0beb3f9361e0fd6ab1016fa3
# 3: 9faed489025cfe40108fc11e6e6cd4ea7529d654ee0a9a978cdc4b3573f8b095
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. -->
<!-- Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py -->

# TRUTH, slot OER-03

Private. Never on the agent surface Harbor assembles.

## The graded quantity

`crossing_step`. The graded step is the FIRST verifier-scheduled evaluation point at which the verifier's own unsmoothed evaluation of the harness-owned weights on the frozen held-out split falls below the bound target loss, AND at which that condition still holds at the bound number of subsequent verifier-scheduled evaluation points.

It is It is never a number the training loop reported, never a number in the submission's stdout, never a field the submission wrote, never the schedule length, and never a smoothed or blended readout.

## The reference answer

| Quantity | Value |
|---|---|
| Reference carrier | `solution/reference.py` |
| Reference sha256 | `9197dc51d7839081dfa549f5d9b7a23aa0720f463cb92d398016b8fb43baa5f7` |
| Schedule length | 3000 |
| Steps executed | 3000 |
| Graded step, the sustained crossing | **2600** |
| Crossing under the pre-drift bar | 2500 |
| Reward | 1.0 |
| Baseline anchor | 3250 |
| Target anchor | 2690 |

raw = (3250 - 2600) / (3250 - 2690) = 650 / 560 = 1.160714..., clipped to 1.0. The reference beats the target bar, and beating the bar scores exactly 1.0.

## Why the schedule tail is worth nothing

The reference schedule runs to 3000 steps and the sustained crossing lands at 2600. The negative control `ctl-schedule-extended-past-crossing` carries the same run with a 6000-step schedule and 30 further evaluation points below the bar, and it scores byte-identically. A schedule extended past the crossing changes the schedule length and cannot change the graded step.

## What a non-converging run produces

A run that does not reach a sustained crossing is graded as a failure with a machine-readable reason. It is never reported as an absent result, never allowed to fall through to a default, and never allowed to fall through to silence. This is the slot's archetype, AR6, applied to the metric.

The same obligation applies to the grading path itself. An aborted verifier still writes a reward with a reason, because a verifier that dies quietly is the same defect one layer up.

## Provenance of the golden series

The series above is an authored deterministic fixture, not a recorded GPU rollout. No training run was performed by this lane and none is claimed. The fixture exists so the live checkers can be exercised on both halves without a accelerator, and it is labelled as derived everywhere it is carried.

Gap: `gap-oer-03-golden-trajectory-derived-not-measured`.

## Tier

Target tier Frontier-defeat, tier_exemption_granted none, anchorable tier capped below Hard.

This task family is text and code only, so the Perception axis carries no category. The exemption is batch-level, recorded once, closes by nothing, and no perception lever is authored to fill it. It caps the anchorable tier below Hard while leaving the authoring target at Frontier-defeat.

No difficulty claim appears anywhere in this bundle. The design obligation to grade the quantity rather than the readout is a correctness obligation and is never restated as a difficulty argument.
