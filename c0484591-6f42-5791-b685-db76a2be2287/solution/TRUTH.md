# FORGE-CANARY-BEGIN
# 0: 809fc16987f5284c614aaf6226adc152039a29c881fd86cdedbee40dd99ac9a5
# 1: c59bcf9dd477d2544caee3aef637e1f68b03603f3518cb43b7ea06a9c8e77583
# 2: 1e7ae3351cc2aec30e578c1586c24a75bdf519b46991d9ddd08ea4446f6e9d6d
# 3: 2a79b2726427762fa6435a98d2da6a3e9813655e707fb07af10359112aab41fd
# FORGE-CANARY-END
# TRUTH.md — OER-05, held out

GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py

## The answer

The reference update rule is `SignGatedMedianAnchorRule` in `solution/reference.py`.
It clears the behavioural novelty gate by doing three things no pinned corpus
entry does:

- sign-agreement gating between momentum and the current gradient
- median absolute magnitude normalization across the block, a rank statistic
- a schedule-free trailing average toward an internal anchor

None of the three is a source-level difference. Each changes the DELTA pattern
the verifier's probe records, which is the only thing the gate compares.

## Measured behavioural distance

| probe set | nearest corpus entry | distance | bound margin |
|---|---|---|---|
| probe-set-a | orthogonalized-momentum | 0.64384 | 0.02 |
| probe-set-b | adam | 0.453888 | 0.02 |

The rule shipped to the agent in `environment/update_rule.py` is heavy-ball
momentum, whose measured distance to the corpus entry `momentum` is exactly
0.0 on both probe sets. Submitting it unchanged scores zero with the reason
`behaviour-matches-corpus-entry`. Paraphrasing its source moves that distance
by nothing at all, which is the property the gate exists to have.

## The reference path through the real gate chain

| fixture | reward | reason |
|---|---|---|
| clean_accepting | 1.0 | graded |
| partial_continuous | 0.75 | graded |
| margin_ramp_continuous | 0.3061224489795917 | graded |

Three different rewards over three fixtures is the reward-continuity evidence:
the score moves with the result rather than flipping at a line.

## Anchors

This family's anchors are UNMEASURED. `baseline_metric` and `target_metric`
are absent from this bundle and no value is invented for them. The gap is
`gap-oer-per-family-anchors-unmeasured`. The two numbers in
`grounding.yaml` under `fixture_anchor_pair` are fixture arithmetic
exercisers, are labelled `not_an_anchor`, and carry no authority.

On the live graded path a telemetry record whose anchor pair is absent is
refused with the machine-readable reason `anchor-pair-absent`.

## What this does not establish

That the reference reaches the target loss by training. The evaluation series in every fixture is derived from this file, which is what makes the fixtures a test of the CHECKERS rather than a claim about the optimizer. No training run has been executed for this slot and none is claimed.
