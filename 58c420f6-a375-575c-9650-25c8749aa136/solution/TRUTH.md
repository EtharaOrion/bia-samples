# FORGE-CANARY-BEGIN
# 0: 5f4bca88b76e2f5853b050d24287dbd093a3759c174989d4d9f1f0d28739fe94
# 1: e177cbe0fda733bf83a2d456a3a0fd6da2d9bbecf47914c82babb6c2ab31d1a9
# 2: f2d220f55d295dd38232b0a5b064212619e6965f8cdf413310a79e0d39293cca
# 3: 963d687d1a787523dffcbf1ced52f8a1b11855f3be9b6d7e80002bf1e944fc29
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
