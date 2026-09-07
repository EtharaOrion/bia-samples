# FORGE-CANARY-BEGIN
# 0: ed4dafeb31b2184147dd45d900e71054c64398db9e9587ecca7b98c6519ae032
# 1: 6f307a3b8c47927e8a2c352708a7aaf8f0cc1431d67b1aed4318d1c4e3d0aebf
# 2: 908084b640c34f9117b0453fe7c83ca38017408eced982dbf4e1bb9191fd0931
# 3: 96f6bab747b394ee41f4e60c10fef7b8ea31c6f56357e1910112d8c48d21c139
# FORGE-CANARY-END
# TRUTH.md, slot OER-14

GENERATED SECTION. DO NOT HAND-EDIT.

source: `solution/grounding.yaml`

## The graded quantity is a fraction whose numerator and denominator the harness owns.

- The metric is bits per byte at FIXED COMPUTE. The numerator is the bits a real 12-layer 768-dimension nanoGPT decoder needs to code the held-out FineWeb text after training on FineWeb text your vocabulary retokenized, and the denominator is a byte count no vocabulary can move. That is the only reading that compares tokenizers at all, because a token count is not comparable across tokenizations and a byte count is.
- The compute unit is one optimizer step, which is exactly one forward and one backward pass over the canonical 524288-token batch. A vocabulary that lowers bits per byte by spending more compute has not won, and the harness counter is what decides that, not the training loop's own claim. The counter tallies the passes apart from the steps, so a step that skipped a pass is refused.
- The evaluation split is absent from environment/ entirely. It is a slice of a FineWeb validation shard pinned by shard name, token offset and token count in the verifier-only control table and materialised only in the verifier image, so nothing the solver can read resolves it.
- The embedding and the output projection are always 50304 rows, whatever vocabulary is submitted. A smaller vocabulary leaves dead rows, exactly as the upstream GPT-2 vocabulary of 50257 does inside the padded 50304, and a larger one is impossible because the assembler truncates at the ceiling and reports what it turned away. Parameter count and per-step FLOPs are therefore identical across every submission.
- merge_depth stops paying at the number of byte pairs the construction window carries at or above the frozen frequency floor. That depth is measured on the staged window at grade time and is written nowhere in this bundle, and nothing announces it. The slot budget is zero-sum, so every slot spent past that point is a slot span_units and numeric_units do not get.
- The reward band is anchored on the harness's own single-direction sweep, each arm of which is a real training run at the same frozen budget, so a greedy sweep of one direction scores exactly 0.0 no matter how many of the fifty attempts it consumes.

## The substrate

This was not a declaration-level repin. The prior substrate modelled a back-off bigram over token ids trained by count accumulation: no parameter tensor, no forward pass, no backward pass, and an evaluation corpus sitting in the agent-visible environment. It failed all three parts of the simulator test, so the graded path was replaced rather than relabelled.

The canonical substrate pins vocab_size 50304 and architecture is a frozen axis, so the embedding and the output projection are ALWAYS 50304 rows whatever a submission builds. A submitted vocabulary smaller than that occupies the leading rows and leaves the rest dead, exactly as the upstream GPT-2 vocabulary of 50257 leaves 47 dead rows inside the padded 50304. It cannot be larger: the assembler truncates at the ceiling and reports the count it turned away in the telemetry, so a spec that ran into the ceiling is visible rather than silently clipped, and no refusal reason was added for it. Parameter count, per-step FLOPs and the step budget are therefore identical across every submission, which is what makes bits per byte at fixed compute a comparison of tokenizers rather than of model sizes.

## Anchors

`anchors_state` is **absent** under gap `gap-oer-per-family-anchors-unmeasured`. `baseline_metric` and `target_metric` are null in `task.toml` and no number is invented for them here.

The reward formula's two anchors are RESOLVED AT GRADE TIME by control runs the verifier performs itself, never by a number written into this bundle. baseline_metric is the best bits per byte reachable by a SINGLE-DIRECTION sweep, measured by the harness by spending the entire slot budget on each construction family in turn, training the frozen decoder once per sweep, and taking the best. It is the exact score a greedy sweep of one direction reaches, so a greedy sweep scores 0.0 by construction. target_metric is the bits per byte of the harness's own reference reallocated construction, measured the same way at the same frozen compute budget. Both are measurements the verifier's process makes; neither is an authored constant, and this lane writes neither number into task.toml.

## The reference allocation

```json
{
  "merge_depth": 18000,
  "numeric_units": 5000,
  "punct_units": 0,
  "span_units": 24000
}
```

## Reward

`raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)`, then `score = min(max(raw, 0.0), 1.0)`. The carrier is `/logs/verifier/reward.txt`, a bare float, and the reason and metric block travel in `/logs/verifier/score.json`.
