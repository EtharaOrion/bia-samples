# FORGE-CANARY-BEGIN
# 0: 45894af1c27c2e801103888498aafb793d7a39ea263d23af04d5d19392c1b032
# 1: a6894d0db0b0537fe8eb45b446462e86b033344057eff9588504260772a14e93
# 2: c9e351b2e6e725d0558eda7838fdd19611917ffbae5d1c057732ff69e7fc76d3
# 3: 3e2a9d3efd92a5820eb08ce257d919b280d89a7d9a4ec33563796d1532ba287f
# FORGE-CANARY-END
# TRUTH.md

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml. -->

## The session is the unit of work, and the budget is the resource.

The surface pays along three directions. One of them, a, is worth 384000 milli-steps and stops paying at 500 units, which ten attempts of the bound cap reach exactly. The other two pay 150 milli-steps per unit and do not saturate inside fifty attempts.

A session that opens on a and keeps feeding it reaches 384000 milli-steps of gain and then nothing, no matter how many of its fifty attempts it spends there. Nothing tells it. Its measured mean simply stops moving.

The reference spends ten attempts taking a to its plateau, one attempt confirming that the eleventh bought nothing, and the remaining thirty-nine on a direction that still pays. That is fifty attempts exactly, which is why the whole budget is load-bearing rather than nominal.

Every attempt not spent on a paying direction costs 7500 milli-steps of the 676500 the target is worth, so the reward degrades continuously with waste rather than falling off a cliff. Wasting more than the probe allowance on the flattened direction is a different failure and is gated to zero with its own reason.

The graded crossing is recomputed by the verifier from its own unsmoothed evaluation and must hold across the further evaluation points the verifier schedules. The surface carries a noise dip that sits under the target for exactly one evaluation, so a blended readout, an early halt and a claimed crossing all produce no crossing rather than an early one.

## What the reference session measures

| quantity | value |
|---|---|
| attempts recorded | 50 |
| attempt terminator | 50 |
| flattening onset attempt | 9 |
| first reallocation attempt | 11 |
| attempts spent on the flattened direction after onset | 1 |
| graded attempt | 49 |
| best multi-seed mean | 3422.4 |
| zero-progress control | 4104.0 |
| reference-operating-point control | 3422.4 |
| reward | 1.0 |

## Anchors

This family's `baseline_metric` and `target_metric` are **absent**, under
`gap-oer-per-family-anchors-unmeasured`. The two control values in the table
above are in-run measurements of the frozen surrogate surface taken by the
verifier, not published anchors, and neither may be transcribed into them.

## The two budgets

`max_timeout` is 6.0 hours and terminates the refinement loop **across**
attempts. `budget_hours` is 0.12 hours and bounds **one** attempt. They are
distinct fields with distinct roles and never stand in for each other.
