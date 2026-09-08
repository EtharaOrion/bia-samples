# OER-09 grounding

Every number in this file was produced by running this slot's own harness on this
slot's own pool during authoring. Nothing here is a target the verifier reads:
the verifier measures both endpoints of its reward scale from scratch on every
grading run, and no literal for either appears anywhere in the bundle.

## What the pool is

5120 blocks of 8192 GPT-2 BPE tokens. 3687 are untouched FineWeb text. The
remaining 1433 were degraded by one of three operations, scattered through the
pool by a permutation so that position carries no information about quality:

| mode       | blocks | operation |
|------------|--------|-----------|
| `clean`    | 3687   | untouched FineWeb |
| `shuffled` |  563   | the block's tokens permuted |
| `tiled`    |  512   | a span of 24-96 tokens tiled to fill the block |
| `uniform`  |  358   | ids drawn uniformly from the vocabulary |

The budget consumes 3072 blocks, so a perfect filter has 3687 - 3072 = 615 blocks
of slack. That margin is what makes an over-aggressive chain fail: filtering below
3072 is refused with `pool-underfilled`.

## Why one feature is not enough

Feature means by mode, computed by the authoring pass from the blocks themselves:

| feature                | clean    | shuffled | tiled    | uniform  |
|------------------------|----------|----------|----------|----------|
| `distinct_token_ratio` |  0.31143 |  0.31009 |  0.00569 |  0.92278 |
| `top1_token_share`     |  0.04166 |  0.04194 |  0.07354 |  0.00046 |
| `bigram_coherence`     | -12.74306| -14.92748| -12.77086| -16.35799|
| `unigram_logprob`      |  -7.66781|  -7.65570|  -7.65924| -12.68467|
| `newline_share`        |  0.02155 |  0.02121 |  0.02113 |  0.00002 |

Permutation is invisible to every unigram statistic, by construction: it does not
change which tokens are in the block. `shuffled` sits on top of `clean` in
`distinct_token_ratio` (0.31009 against 0.31143), in `top1_token_share` and in
`unigram_logprob`. Only `bigram_coherence` moves, and it moves decisively:
`shuffled` runs from -15.59 to -14.34, `clean` from -14.28 to -11.51.

Tiling is the mirror image. A tiled span of real text has coherent adjacent pairs,
so `tiled` sits on top of `clean` in `bigram_coherence` (-12.77 against -12.74).
What collapses is `distinct_token_ratio`: `tiled` never exceeds 0.0098 and `clean`
never falls below 0.1381.

So a chain needs a predicate from each family. That is not a rule imposed by the
schema; it is a property of the data.

## Measured

Four chains, trained by this harness from the frozen initialisation on 3072 blocks
each, evaluated on the held-out slice:

| chain | admitted | degraded blocks in the 3072 used | val loss | gap closed |
|---|---|---|---|---|
| shipped default (`quality_score`, `lang_prob_en` — neither field exists) | 5120 | 835 | 5.855329 | 0.00% |
| `bigram_coherence >= -13.8` only | 4129 | 331 | 5.586293 | 33.51% |
| `distinct_token_ratio` in [0.10, 0.60] only | 4250 | 397 | 5.260892 | 74.05% |
| both families (the verifier's private reference) | 3672 | 0 | 5.052570 | 100.00% |

The span between the two anchors is **0.802759 nats**. The single-feature chains
land at 33.5% and 74.1%, so the reward is genuinely graded rather than a step.

The shipped default resolved **two predicates out of two to nothing** and admitted
the pool unchanged, which is the silent failure this slot is built on. It is also
why it is the control: it is what happens when curation runs and does nothing.

## The oracle

`solution/reference.py` writes a chain with thresholds at -13.75 and [0.09, 0.75].
The verifier's private chain uses -13.8 and [0.10, 0.60]. They were arrived at
separately and they are not the same document, so the reference arm of the gate is
not the reference chain being graded against itself. Both land on the same set of
blocks because both thresholds sit in the gap between distributions rather than on
a quantile of either.

## Cost

Training a single arm takes 27s once the graph is compiled; the uncurated control
takes 47s because uniform-id blocks slow the loss down, not the kernels. Three
arms plus three evaluations plus one compile measured **190s** end to end on one
H100, against a declared budget of 900s.
