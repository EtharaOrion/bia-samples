# FORGE-CANARY-BEGIN
# 0: d11c5fba7be7e9587cb151eb12cf231b5b1342f6fd1999000f205b38445631fb
# 1: dce453a9e2c7385ad85c8d01b73dfbebac8d617aa4a6eead285fb1a7eca1cd81
# 2: ccc98fe4fd30af02cc206ae1f4647bf373374ba4bcfde827734a0e7f24df7bb1
# 3: 80ce4bda8b4281e7f9224fed355ef95992adced98ded91974f1966a70da98e14
# FORGE-CANARY-END
# TRUTH.md, slot OER-14

GENERATED SECTION. DO NOT HAND-EDIT.

source: `solution/grounding.yaml`

## The graded quantity is a fraction whose numerator and denominator the harness owns.

- The metric is bits per byte at FIXED COMPUTE. A vocabulary that lowers bits per byte by spending more compute has not won, and the harness counter is what decides that, not the training loop's own claim.
- The denominator is the byte length of environment/corpus/eval.txt and nothing else. A construction that normalizes, strips or drops evaluation bytes and divides by what survives has changed what it measures over, which scores zero with reason denominator-not-frozen-eval-bytes.
- The graded reading is the WORST reading across three verifier-scheduled evaluation points, so a favourable fluctuation at one point is not a level.
- merge_depth stops paying at 413 admissible merges and nothing announces it. The slot budget is zero-sum, so every slot spent past that point is a slot span_units and numeric_units do not get.
- The reward band is anchored on the harness's own single-direction sweep, so a greedy sweep of one direction scores exactly 0.0 no matter how many of the fifty attempts it consumes.

## Anchors

`anchors_state` is **absent** under gap `gap-oer-per-family-anchors-unmeasured`. `baseline_metric` and `target_metric` are null in `task.toml` and no number is invented for them here.

The reward formula's two anchors are RESOLVED AT GRADE TIME by control runs the verifier performs itself, never by a number written into this bundle. baseline_metric is the best bits per byte reachable by a SINGLE-DIRECTION sweep, measured by the harness by spending the entire slot budget on each construction family in turn and taking the best. It is the exact score a greedy sweep of one direction reaches, so a greedy sweep scores 0.0 by construction. target_metric is the bits per byte of the harness's own reference reallocated construction, measured the same way at the same frozen compute budget. Both are measurements the verifier's process makes; neither is an authored constant, and this lane writes neither number into task.toml.

## The reference allocation

```json
{
  "merge_depth": 448,
  "numeric_units": 124,
  "punct_units": 0,
  "span_units": 500
}
```

## Reward

`raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)`, then `score = min(max(raw, 0.0), 1.0)`. The carrier is `/logs/verifier/reward.txt`, a bare float, and the reason and metric block travel in `/logs/verifier/score.json`.
