# FORGE-CANARY-BEGIN
# 0: 1d3cf39d781967a1c80aa6c0987fdad97f699e07dff3bd2b1727937ec17fe9ab
# 1: 66f5bc5afa0692ee5cd0cabc7af548b27fd521166b92f10a6e87316cef0c604f
# 2: 97f9d62b7ab9050f474ed8f771a212651e76cb93ec2188b5537e865111ca3501
# 3: e5e250b27d6a9fbad70185787b2b794d8449aa6164932f61bc6eb382777b4bc0
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Derived from solution/grounding.yaml by solution/recompute.py. -->

# OER-13 TRUTH

## What is graded

- metric: validation loss of the trained model on the frozen held-out split
- unit: negative log likelihood in nats divided by the byte length of the split
- direction: lower
- computed by: tests/checkers.py::recomputed_loss, inside the verifier process
- weights: the model the harness-owned frozen train stage wrote at the bound evaluation point
- never: never a number the submission reported, printed or wrote; no smoothing, averaging or EMA is applied anywhere on the graded path

## Anchors

- anchors_state: absent
- gap: gap-oer-per-family-anchors-unmeasured
- baseline_metric: null
- target_metric: null
- F13 carries no published baseline or target. No number is invented here. The reward schema is bound in full, and the two slots of the bound lower-is-better formula are filled in-run by verifier measurement of two harness-owned compositions: the pristine delivered composition supplies the baseline slot and the reference composition supplies the target slot. When the family anchors are measured, the published pair replaces the in-run pair and this block closes.
- formula: `raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)`
- clip: `score = min(max(raw, 0.0), 1.0)`

## The reference composition

- carrier: `solution/reference.py`, sha256 `9f755016b5d903eb97014649f2679a678645b2737c07d8d3166e41dd56ae36c1`
- A composition, not two independent stage improvements. The parse stage LOWERS its own retention readout on purpose, dropping the navigation bar, the advertisement line and the copyright footer that no held-out document contains, and deduplicating repeated lines. The tokenize stage then spends its vocabulary on the cleaned stream, taking single-character coverage first so no held-out byte falls back to the unknown unit, then leading-space word units so a word and the space before it cost one token. Either half alone moves the graded loss far less than the pair does.

## The measurement that makes this slot what it is

| quantity | delivered baseline | reference |
|---|---|---|
| parse `chars_retained_ratio` | 0.962848 | 0.47092 |
| tokenize `bytes_per_token` | 2.052417 | 3.306054 |
| **graded loss per byte** | 1.512280061 | 1.054387882 |

The reference's retention readout is WORSE than the delivered baseline's, 0.47092 against 0.962848, and its end-to-end loss is BETTER, 1.054387882 against 1.512280061. An agent that optimizes the parse stage's own honest readout moves away from the graded quantity. That is the archetype in one pair of numbers.

## Per-fold, so the improvement is visibly sustained

| fold | baseline | reference |
|---|---|---|
| fold-0 | 1.607703766 | 1.082628636 |
| fold-1 | 1.467558109 | 1.040603069 |
| fold-2 | 1.462690132 | 1.018967264 |
| fold-3 | 1.513778938 | 1.078398372 |

## The golden trajectory

1. **read the surface** — Read instruction.md, environment/pipeline/protocol.json and the three free files. Note that protocol.json records token_budget.value_at_delivery and says in its own authority note that this number is not itself the graded bound and is not announced when it moves. _readout: none_
2. **measure the delivered composition end to end** — Run the delivered chain unchanged and record the end-of-chain loss. This is the only number any later comparison is worth making against. _readout: loss_per_byte 1.512280061; chars_retained_ratio 0.962848; bytes_per_token 2.052417_
3. **reject the stage-local move** — chars_retained_ratio is already 0.962848 and can be pushed to 1.0 by keeping every character. Compose that change through to the end: it floods the training stream with the navigation bar, the advertisement line and the copyright footer, and the end-of-chain loss gets worse. The stage readout improved and the graded quantity degraded. _readout: stage-local optimum rejected on the composed measurement, not on the stage readout_
4. **change parse to drop boilerplate and deduplicate** — Drop the three boilerplate prefixes, unwrap headings, unescape entities, collapse runs of spaces and drop repeated lines. Retention falls to 0.47092, which is the stage readout getting worse on purpose. _readout: chars_retained_ratio 0.47092_
5. **change tokenize to spend vocabulary on the cleaned stream** — Take single-character coverage first so no held-out byte falls back to the unknown unit, then leading-space word units, then a short tail of bare word units. Compression rises because the units now match the text that is actually there. _readout: bytes_per_token 3.306054; vocab_size 221; tokens 3898_
6. **compose and measure end to end** — Run the pair through the chain to the bound evaluation point, with the harness feeding exactly the bound budget from the admin plane. _readout: loss_per_byte 1.054387882 against baseline 1.512280061_
7. **check the improvement is sustained** — Confirm the gain holds on every fold of the frozen partition, not only overall, because the verifier schedules its sustain points itself and does not announce the order. _readout: every fold improves: 1.082628636, 1.040603069, 1.018967264, 1.078398372 against 1.607703766, 1.467558109, 1.462690132, 1.513778938_
8. **submit** — The full gate chain passes and the reward is the clipped continuous term. _readout: reward 1.0, reason graded-on-recomputed-validation-loss_

## Statement ambiguity: exactly one graded outcome

- reading one, from instruction.md, the section "What is graded, stated plainly": the total negative log likelihood in nats of the frozen held-out split under the weights the harness-owned frozen train stage wrote at the final checkpoint, divided by the byte length of the split, lower better
- reading two, from environment/pipeline/protocol.json, the "evaluation" block: the total negative log likelihood in nats over the fold, divided by the byte length of the fold, computed by the verifier from the weights the harness-owned frozen train stage wrote at the bound evaluation point, lower better
- they are the same quantity: Both readings name the same segmentation rule, the same normalization, the same evaluation point, the same weight provenance and the same direction. Over the whole split they are the same expression, and tests/checkers.py implements it once in fold_loss, which both the graded path and the in-run anchor measurement call. There is no second quantity anywhere in the bundle that a reader could take to be the objective.
- Every rule the grader enforces resolves to a zero with a machine-readable reason or to the single continuous term. There is no partial-credit path, no alternative objective and no binary outcome.

## Solver egress divergence

- gap id: gap-oer-solver-egress-ruled-stricter
- This slot binds `open` at the agent surface, which is reading A, while the batch-level field still reads `setup-only`, which is reading B. The divergence is recorded rather than silently resolved, and this lane does NOT amend the batch-level field: seed/contract.yaml is a batch singleton and no lane writes it. The same contract's family delivery.network_policy block binds agent_environment egress allowed with the upstream deny-list, which is the reading this slot follows.
- The deny-list is not a containment proof. Mirrors, caches, paraphrased write-ups and model weights all carry a recipe past any deny-list. That is exactly why the deterministic checker, and not the network policy, is the load-bearing control in this slot: the corpus, the frozen model and the held-out split are authored bytes with no published lineage, and the graded quantity is recomputed in-run by the verifier from bytes it owns.

## Reward contract

- reward path: `/logs/verifier/reward.txt`, carrier: bare float
- score document: `/logs/verifier/score.json`
- The live seed/forge/verifier.py binds the reward carrier to reward.txt, not reward.json. The bare float goes to /logs/verifier/reward.txt and the machine-readable reason and metric block go to the companion /logs/verifier/score.json. Both are written from the same trap-guarded path, so the instrument's binding and the reason carrier never disagree.
- /logs/verifier/ is a single shared host path and concurrent lanes collide on it. Local exercise of tests/test.sh redirects the reward root with OER13_REWARD_ROOT to a lane-private directory. The bound contract path inside the bundle stays /logs/verifier/reward.txt; only the local exercise is redirected, and feasibility.yaml records that the exercise ran redirected.

## Tier

- target tier: Frontier-defeat, tier_exemption_granted: none
- anchorable tier: capped below Hard by the standing batch-level Perception-axis exemption
- This task family is text and code only, so the Perception axis carries no category; the exemption is batch-level, recorded once, closes by nothing, and no perception lever is authored to fill it. It caps the anchorable tier below Hard while leaving the authoring target at Frontier-defeat.
