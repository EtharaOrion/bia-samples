# FORGE-CANARY-BEGIN
# 0: 1d9c923645dd8f4e47b737f7aac753fc0928791e3c2564379dd61ce7b12d6bd7
# 1: 4ff58ea8b326bf3e49d16e42e7dba9f8626e6560f611f218500094730d0fa141
# 2: 7ab5bd58dd5887216869a471b2a792f2a8d1d4982022923e66c7ba991f46f24c
# 3: 5607065a5c492fe5c6d9d86e9d0c158a73c1e5081dbba5e50a680a1e79e28b91
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
