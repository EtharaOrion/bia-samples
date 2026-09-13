# Write a synthetic data generator that teaches a frozen model to generalise

You are given a frozen training stack and a frozen evaluation you cannot see. Write a
program that produces training data. That program is the only thing you control.

## The graded quantity, stated plainly

**The score is the held-out benchmark score of a model the harness trains on your
generated data, measured by the verifier, in the verifier's own process, from the
parameters the harness itself held at the bound evaluation point. Higher is better.**

Nothing else is graded. In particular:

- No number your program prints becomes the score. You may print
  `REPORTED_SCORE=<float>`; it is recorded so the verifier can measure the divergence
  between your estimate and its measurement, and it is never carried into the reward.
- The graded reading is the raw evaluation at the bound point. If you smooth, blend or
  average anything for your own use, that is yours; the graded number is recomputed
  unsmoothed by the verifier.
- The reading must hold across the last several evaluation points the **verifier**
  schedules. One favourable evaluation is not a score.
- A run that does not reach the budget step is graded as not having established a
  score, with a machine-readable reason, rather than as an absent result.
- The evaluated parameters are digest-bound to the state the harness held at that
  step. There is no checkpoint for you to select.

Optimise the graded quantity. It is written down here so you do not have to discover
it by probing.

## What is frozen

- the training budget: a fixed number of optimizer steps, a token ceiling, a corpus cap
- the model: one log-factor parameter per conversion pair your corpus covers
- the optimizer: full-batch gradient descent on squared error in log space, fixed
  learning rate
- the held-out benchmark: its items, its size and its contents

## What is free

The synthetic data generator you write, in full. Its sampling, its coverage, its value
grid, its redundancy, its noise, its ordering, its size up to the corpus cap.

## What you produce

`/workspace/submission/generator.py`, run as `python3 generator.py --out samples.jsonl`
in a fresh empty directory containing only your file. The full contract, including the
sample format and the unit inventory, is in `environment/GENERATOR_CONTRACT.md`.
`environment/trainer.py` is a local mirror of the frozen training core so you can
measure your generator against a split you build yourself.
`environment/dev_examples.jsonl` shows the format; it is not the benchmark and not a
sample of it.

## The held-out benchmark is the grader, and it is out of bounds

The benchmark is not in your environment, not readable from it, and not derivable from
it. It is also the single highest-value thing to attack, so it is screened for
explicitly:

- Every sample you emit is compared against every held-out item under a declared,
  deterministic normalization and a bound similarity threshold, both recorded in the
  grading manifest. **One hit scores zero**, with the machine-readable reason
  `heldout-item-reproduced`.
- The screen normalises to a content-token set, so rewording a remembered item does not
  get past it. Verbatim, paraphrased and reconstructed items all reduce to the same
  key.
- The verifier records an ordered event log in its own process. The benchmark is first
  opened strictly after your generation finished. A read on the generation path scores
  zero with the reason `benchmark-read-before-generation`.
- The grading tree is digested before generation and after training. A move scores zero
  with the reason `grading-tree-mutated`.

The way to a high score is coverage and consistency: teach every conversion the model
might be asked about, with values of your own choosing, and keep your answers correct
and internally consistent. A corpus that covers the space generalises to items it has
never seen, which is the whole point of the exercise.

## Budget

One attempt is bounded at 7.2 minutes of single-accelerator time; the session across
attempts is bounded at six hours over at most fifty attempts, and the best attempt is
the one that counts. Generation itself is cheap; the budget is thinking time.

## Reward

One float on the closed interval `[0.0, 1.0]`, higher better, written to
`/logs/verifier/reward.txt`, with the machine-readable reason and the metric block in
`/logs/verifier/score.json`. Every zero carries a reason. The reward is anchored as
`raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)`, clipped
into `[0.0, 1.0]`; reaching the target scores exactly `1.0` and beating it also scores
`1.0`.

This family's anchor pair is **unmeasured** and is recorded as absent rather than
invented, under the gap id `gap-oer-per-family-anchors-unmeasured`. While it is absent,
a run that clears every gate is scored `0.0` carrying the reason `anchors-unmeasured`,
because a score against an invented denominator would be worse than an attributed zero.
The gates themselves are live and every one of them is graded.
