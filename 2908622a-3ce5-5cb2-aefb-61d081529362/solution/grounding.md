# OER-10 — grounding

## What is graded

The verifier's own validation cross-entropy, in nats, on a held-out FineWeb slice that
exists only inside the verifier image. The reward is

```
reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )
```

and **both** endpoints are measured by real from-scratch training runs on the grading
run itself. There is no literal for either anchor anywhere in this bundle. This is the
substantive change from the bundle that stood in this slot before, which declared its
anchors unmeasured (`gap-oer-per-family-anchors-unmeasured`) and therefore refused
every possible submission at 0.0 — a constant reward, and a dead task.

## The authoring search

Every loss below was produced by a run of this bundle's own harness under the frozen
budget, evaluated on `environment/data/devset_slice.bin`. The graded split is a
different slice of FineWeb, so the graded numbers sit at a small offset from these;
what carried the decisions is the ordering, which the offset does not disturb.

| loss | candidate | what the run established |
|---|---|---|
| 5.01954 | shard_a, shard_b, shard_c, permuted chunks of 512 | the clean-shard mixture is a plateau: three orderings of it lie within 0.003 nats |
| 5.02089 | shard_a, shard_b, shard_c, permuted chunks of 8 | fine-grained permutation is indistinguishable from interleaving |
| 5.02281 | shard_a, shard_b, shard_c, evenly interleaved | the best point found: three clean shards, each spent almost exactly once, stationary over the run |
| 5.03957 | shard_a, shard_b, shard_c, permuted chunks of 64 | chunk-permuted ordering recovers most of the gap to interleaving |
| 5.06401 | shard_a, shard_b, shard_c plus shard_e at 0.15 | even a sixth of the duplicated shard costs 0.041 |
| 5.06660 | shard_a, shard_b, shard_c, fed one shard at a time | the right blocks in the wrong order gives most but not all of the value back |
| 5.07851 | shard_a and shard_b only, so 1536 blocks each against 1023 | two clean shards is worse than three: the repetition costs more than the third shard's tokens are worth |
| 5.08536 | shard_a, shard_b, shard_c plus shard_f at 0.15 | and a sixth of the half-reordered shard costs 0.063 |
| 5.11480 | shard_a, shard_b, shard_c plus shard_f at 0.35 | the cost of shard_f is roughly linear in its weight |
| 5.16582 | shard_a, shard_b, shard_c plus shard_f at 0.5 | the half-reordered shard costs real nats even at half weight |
| 5.30848 | shard_a alone, so 3072 blocks drawn from 1023 -- three passes | concentration is a trap: three passes over clean data is worse than one pass over a mixture containing junk |
| 5.49905 | equal weight on all six shards, interleaved -- the shipped default | the floor; a third of the budget goes to shards that damage the run |
| 7.68379 | shard_d and shard_e alone | the two damaged shards are not merely weak, they are far worse than nothing |

### The measurement floor

This harness is deterministic in everything but the accelerator's own reduction order. The reference mixture was trained twice on this slot's own substrate and produced 5.02281 and 5.02261, a spread of 0.0002 nats. The span between the two anchors is 0.476 nats, two thousand times that spread, and even the smallest effect acted on below -- the 0.017 nats between two chunk-permutation group sizes -- is eighty times it.

## The two anchors and the oracle

| artifact | where it lives | measured devset loss |
|---|---|---|
| the shipped default, LOW anchor | `environment/default_mixture.json` | 5.49905 |
| the private reference, the BAR | `tests/private/reference_mixture.json` | 5.02281 |
| the oracle's submission | written by `solution/reference.py` | 5.03957 |

The oracle is **not** the reference. They were arrived at on separate legs of the
search and differ in its ordering policy -- the reference interleaves the chosen blocks evenly across the whole run, the oracle feeds them in permuted chunks of 64. On the devset numbers above the oracle closes

```
(5.49905 - 5.03957) / (5.49905 - 5.02281) = 0.965
```

of the gap, so the reference arm of the gate scores on a measurement rather than by
construction, and a neutralised submission is refused by the schema at 0.0.

## Why this is hard

Nothing in environment/README.md says what any shard is worth; it says only how each pipeline differed, and the differences are described in terms (a dedup pass that reorders, a narrow filter repeated to length) whose cost to a language model is not deducible from the description. The measurements say the cost is enormous and very unevenly distributed: training on the two damaged shards alone reaches 7.68 against the default's 5.50, so a third of the default's budget is being spent on tokens that are worse than useless. But the obvious correction -- concentrate everything on one clean shard -- is WORSE than the default's own six-way split at 5.31, because 3072 blocks against 1023 available means reading that shard three times, and the repetition costs more than the junk did. The answer is therefore an interior point: enough clean shards to cover the budget close to once each, and then an ordering decision on top that is worth another 0.044 nats. Neither half is reachable without running the harness.

## Cost

The graded path is three from-scratch training runs of 3072 micro-batches of 16x512
plus three evaluations over 2,097,152 tokens, sharing one compiled graph. Measured wall
clock is recorded in `task.toml` as `graded_path_seconds_measured`, and it is the
gate's own number rather than a declaration.
