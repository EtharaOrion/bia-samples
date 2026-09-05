# FORGE-CANARY-BEGIN
# 0: f957188c9e367ce5e1ad683548c9be5fc6e94176275334e58c98e24586ba8ce8
# 1: ce9c3735062ecc46d2791df0e559c645bd33db45dd8307d6ba9d0ad219da6259
# 2: 7e0cad18cd8134fa5c58d8bec5ea0068598ae9243f5d08d2157182edd3a7d0e6
# 3: 501b55a26f198ac358fb1a0df4de45a99bb5d0dc1ab112532e77094a11eb2202
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. -->
<!-- Derived from solution/grounding.yaml by solution/recompute.py. -->

# TRUTH, slot OER-02

## What is graded

The **consolidated sustained crossing step** of the session, lower is better.

1. The verifier trains and evaluates. Every graded number is the raw loss the
   verifier's own loop measured on a held-out draw whose seed exists only in the
   verifier environment. Nothing on the graded path is smoothed, averaged or
   filtered.
2. A crossing at step `p` requires the target to hold at `p` and at the next
   3 evaluation points the verifier schedules.
3. The graded step is the 3rd smallest sustained crossing over attempts with pairwise-distinct recipe fingerprints.

## Why one lucky attempt cannot reach the band

the graded step is the m-th smallest sustained crossing over PAIRWISE DISTINCT recipe fingerprints, so one fortunate attempt cannot set it and neither can one recipe resubmitted m times. On top of that the determining attempt must sit at index 3 or later, the running best must strictly improve at least three separate times, and the determining attempt must inherit at least two component values from strictly earlier attempts that were on the running best when they ran. A fixed sweep of unrelated configurations therefore scores zero however good its best cell was.

## Anchors

| field | value | source |
|---|---|---|
| baseline_metric | 3250 | upstream record 36, tuned Muon with aux AdamW, loss 3.2787 at n equal to ten, stat-verified, dated 2026-06-11 |
| target_metric | 2690 | upstream record 46, the current world record, loss 3.2783 at n equal to eight, stat-verified, dated 2026-06-19 |
| authority | requirements/bia-environment-spec.md lines 297 to 321 | |

Reward: `raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric); score = min(max(raw, 0.0), 1.0)`

reaching the target scores exactly 1.0 and beating it also scores 1.0; the target is a bar, not a point on a gradient.

## The reference

`solution/reference.py`, sha256 `9ffae8250a27330d638dd1648e646c94e179e48ef0b09273864844cd3bf1e02f`.

It bootstraps three optimizer families, then hill-climbs a fixed ladder on top
of the running best, so every later proposal is built from an earlier one. It
reports no crossing, because claiming a number the verifier did not measure is
what the DIVERGENCE checker exists to make expensive.

## The golden trajectory

Synthetic, compiled from `solution/grounding.yaml`. It carries no measured
number; it fixes the SHAPE the checker fixtures are built from.

| attempt | family | crossing | inherits from |
|---|---|---|---|
| 1 | momentum | 500 | none |
| 2 | ortho | 190 | none |
| 3 | adamw | 160 | none |
| 4 | adamw | 150 | 3 |
| 5 | adamw | 150 | 4 |
| 6 | adamw | 170 | 4 |
| 7 | adamw | 180 | 4 |
| 8 | adamw | 150 | 4 |
| 9 | adamw | 160 | 4 |
| 10 | adamw | 170 | 4 |
| 11 | adamw | None | 4 |
| 12 | adamw | 160 | 4 |

- running best: `[500, 190, 160, 150, 150, 150, 150, 150, 150, 150, 150, 150]`
- strict improvements at attempts: `[1, 2, 3, 4]`
- consolidation indices: `[4, 5, 8]`
- graded step: `150`
- reward: `1.0`, reason `consolidated-crossing-graded`

## What is unmeasured

the reference is demonstrated end to end at the fixture shape only. No claim is made that the reference reaches 2690 steps on the bound H100 operating point, and none is made anywhere in this bundle.

Gap: `gap-oer-scaled-operating-point-unmeasured`.

## Divergences carried

- solver egress: the family delivery.network_policy block in the same contract binds agent_environment: egress allowed with the upstream deny-list, and the lane brief binds solver_egress = open at the agent surface with the verifier surface denied. This lane binds open and records the divergence; it does NOT amend the batch-level field.
- reward path: tests/checkers.yaml binds reward_path to the .txt path the parser actually requires. tests/test.sh and tests/reward.py write BOTH: the single float to /logs/verifier/reward.txt and the reward document, with reason and metric, to /logs/verifier/reward.json. task.toml [reward].path carries the .json document. Nothing is dropped and nothing is invented.
- free axis: an algorithm outside the grammar cannot be expressed. The grammar spans sgd, momentum, nesterov, adam, adamw, lion and Newton-Schulz orthogonalized momentum, with five schedules, warmup, decoupled decay, gradient clipping, per-group learning-rate multipliers and three initialization schemes, but it is a narrowing and is recorded as one rather than presented as equivalent.
