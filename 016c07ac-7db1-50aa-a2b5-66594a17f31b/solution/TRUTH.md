# FORGE-CANARY-BEGIN
# 0: f1018fbf9009b2b521f5badb4ea108cce1f9391720b3440bd0374b0335efd0ba
# 1: 4280d5dfca112e5da60362ff6c117e91fb0bd892b491110b2ff6b12391369dab
# 2: f05e38339a6f5b70dc6cdd4a739abb45b3a62427a98e416e56dee77ad7b96668
# 3: 3499801007d7d9b310ad91b94caa063bb53489bca9ad5f2a6c4154b71541bcd5
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
