# FORGE-CANARY-BEGIN
# 0: 3bd5348fd31794fa69be2db05aa56cd47194dd378413e851fd2b862ee6bb1f78
# 1: 71bfd50b3b687a9733b804d16c04ea982c42b0b761ff2bedfa24aa0df6d07b0f
# 2: c3efae6e50db4fd3f3a094c85b8212f660fd6f87c72c7aaa6dcc6d62e56f032e
# 3: 388829ef6ec36b66c2fa71cf402da824eaa895125c67f50556fcf6454903e3cf
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Derived from solution/grounding.yaml by solution/recompute.py. -->

# OER-13 TRUTH

## What is graded

- metric: nanoGPT validation loss on the verifier-owned held-out FineWeb split
- unit: mean cross entropy in nats per token
- direction: lower
- computed by: tests/checkers.py::recomputed_loss, inside the verifier process
- produced by: tests/frozen/nanogpt.py::validation_loss, the verifier's own forward pass, bound onto the checker handle by tests/harness.py before any checker runs
- weights: the parameter snapshot the harness-owned frozen train stage wrote at the bound evaluation point
- split: the verifier's own held-out FineWeb validation shards, under the root pinned in tests/bound.json, absent from environment/ by construction
- never: never a number the submission reported, printed or wrote, and never a number an in-container harness printed; no smoothing, averaging or EMA is applied anywhere on the graded path

## What this re-base actually did

This was a replacement of the training and evaluation path, not a repin of a declaration. As delivered, this slot had no model on its graded path at all. Its train stage accumulated unigram and bigram counts over whatever vocabulary the submission handed it, and the graded scalar was an interpolated bigram likelihood per byte of an authored text file that sat inside environment/ where the solver could read it. There was no forward pass and no backward pass anywhere in the bundle, so there was nothing to repin. The re-base wrote a real 12-layer, 768-dim decoder and its training loop, moved the graded scalar onto the verifier's own forward pass over held-out FineWeb validation shards, removed the authored split from the agent surface, and froze the token space to the GPT-2 byte pair encoding so the training stream and the split are comparable. Describing that as a repin would misdescribe it.

## The frozen substrate

- declaration: `environment/nanogpt_substrate.json`, replicated byte for byte from the canonical operating point
- architecture: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024
- run: 524288 tokens per step, exactly one forward and one backward pass over each step's batch, exactly one optimizer step per step
- graded artifact: a real parameter snapshot of that decoder, shape-bound to every value above by `check_graded_parameters_bind_frozen_architecture`

## The simulator test, applied to this slot

- weights in the loop: PASSES. The graded artifact is a parameter snapshot of the frozen architecture, and the shape map it is checked against is derived from the declaration alone, so a count table or a weight table of another shape cannot be scored as this model. The compiled surface carries both halves of that binding as fixtures.
- the verifier recomputes: PASSES. The graded scalar is the verifier's own forward pass over a split that is absent from `environment/` and pinned to the verifier's copy on the admin plane. Nothing an in-container harness prints reaches the grade, and a run whose split did not resolve under the pinned root scores zero under `holdout-split-not-verifier-owned`.
- no forward pass means no score: PASSES. `GPT.forward` is the only producer of a loss in the bundle. Delete it and the metric is undefined rather than different, which is exactly what the retired count table failed: it kept emitting a number with every pass removed.

## Anchors

- anchors_state: absent
- gap: gap-oer-per-family-anchors-unmeasured
- baseline_metric: null
- target_metric: null
- F13 carries no published baseline or target. No number is invented here. The reward schema is bound in full, and the two slots of the bound lower-is-better formula are filled in-run by verifier measurement of two harness-owned compositions: the pristine delivered composition supplies the baseline slot and the reference composition supplies the target slot, both trained through the identical frozen decoder under the identical step budget. When the family anchors are measured, the published pair replaces the in-run pair and this block closes.
- formula: `raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)`
- clip: `score = min(max(raw, 0.0), 1.0)`

## The reference composition

- carrier: `solution/reference.py`, sha256 `acb6e5727fb147e380996e5c30774192102df83f6408086e8226fea3b0ac2e6e`
- A composition, not two independent stage improvements. The parse stage LOWERS its own retention readout on purpose, dropping the navigation bar, the advertisement line and the copyright footer that held-out FineWeb prose does not contain, and deduplicating repeated lines. The tokenize stage then spends the frozen token budget on the cleaned stream, dropping fragments that survived parsing without becoming prose, deduplicating at the token level, and packing what is left with the end-of-text delimiter the evaluated corpus uses. Either half alone moves the graded loss less than the pair does.
- what the re-base changed in it: the parse half is unchanged, and it is unchanged because it was already an argument about which text belongs in a training stream, which is exactly what the new graded path measures. The tokenize half was rewritten, because it used to design a vocabulary and the token space is now frozen at vocab_size 50304 under the GPT-2 byte pair encoder.

## The measurement that makes this slot what it is

The numbers that used to stand here are gone and nothing replaced them. They were transcribed from real runs of the retired path: an interpolated bigram scored per byte on an authored split that no longer exists in this bundle. They were measurements of a different model on a different split, so the re-base retired them rather than carrying them across. Republishing them beside a transformer run would have been a fabricated pilot result, and this project publishes an unmeasured quantity as unmeasured.

- observations_state: unmeasured
- gap: gap-oer-13-rebase-observations-unmeasured
- still unmeasured: the delivered baseline's validation loss at the bound operating point, the reference's validation loss at the bound operating point, the per-fold pair for either, and therefore the separation between them and the reward the reference earns.
- what was established instead: the re-based graded path was exercised end to end on a shrunk substrate in a lane-private workspace. The delivered chain ran, the frozen stage trained a real decoder, the verifier's forward pass produced the graded scalar, the architecture shape binding held, the split resolved only under the pinned verifier-owned root, the leakage comparison ran in the frozen token space, and the reward document was written with an attributed reason. That establishes that the path RESOLVES. It establishes no anchor, because the architecture and the step budget it ran under were not the bound ones.
- how it closes: one run of `tests/test.sh` at the bound operating point on the reference machine, with the held-out shards mounted, transcribed back into `solution/grounding.yaml`.

## The corpus scale shortfall, stated rather than hidden

- gap: gap-oer-13-corpus-below-substrate-scale
- This slot's delivered corpus is a few tens of kilobytes of authored raw web text. One step of the frozen substrate covers 524288 tokens, so the corpus is cycled many times inside a single step and the substrate's 3.28 target is not reachable here. The bound step budget is therefore set by the corpus scale and not by the record set's 3250 steps. What the run still measures is a real held-out loss of a real decoder, ordered by how well the curated stream generalises, which is the quantity this slot grades. Closing the shortfall means staging a raw corpus at FineWeb scale into the agent image, which is a corpus-authoring change and outside a re-base lane's remit.

## The generator is still on the retired path

- gap: gap-oer-13-generator-still-on-the-retired-path, blocks_freeze
- `solution/recompute.py` was not rewritten by this lane. It still reads observed keys this grounding no longer carries, and it embeds a verbatim template of the previous `tests/test_output.py` including the retired floor-anchor constant. The generated artifacts staged here, `tests/test_output.py` above all, are hand-authored and consistent with `solution/grounding.yaml`; the generator must be brought back onto them before the slot is frozen. It is recorded rather than repaired here because a generator repaired against numbers this family has not measured would manufacture exactly the fabricated anchors the re-base removed.

## Statement ambiguity: exactly one graded outcome

- reading one, from instruction.md, the section "What is graded, stated plainly": the mean cross entropy in nats per token of the verifier's held-out FineWeb validation split, under a forward pass of the parameter snapshot the harness-owned frozen train stage wrote at the final snapshot, lower better
- reading two, from environment/pipeline/protocol.json, the "evaluation" block: the mean cross entropy in nats per token over the fold, computed by the verifier from the parameter snapshot the harness-owned frozen train stage wrote at the bound evaluation point, lower better
- they are the same quantity: Both readings name the same architecture, the same normalization, the same evaluation point, the same parameter provenance and the same direction. Over the whole split they are the same expression, and tests/frozen/nanogpt.py implements it once in validation_loss, which both the graded path and the in-run anchor measurement call through the same bound evaluator. There is no second quantity anywhere in the bundle that a reader could take to be the objective, and there is no longer a second model either.
- Every rule the grader enforces resolves to a zero with a machine-readable reason or to the single continuous term. There is no partial-credit path, no alternative objective and no binary outcome.

## Solver egress divergence

- gap id: gap-oer-solver-egress-ruled-stricter
- This slot binds `open` at the agent surface, which is reading A, while the batch-level field still reads `setup-only`, which is reading B. The divergence is recorded rather than silently resolved, and this lane does NOT amend the batch-level field: seed/contract.yaml is a batch singleton and no lane writes it. The same contract's family delivery.network_policy block binds agent_environment egress allowed with the upstream deny-list, which is the reading this slot follows.
- The deny-list is not a containment proof. Mirrors, caches, paraphrased write-ups and model weights all carry a recipe past any deny-list. That is exactly why the deterministic checker, and not the network policy, is the load-bearing control in this slot: the corpus is authored bytes with no published lineage, and the graded quantity is computed in-run by the verifier, from parameters the harness wrote, on a split the agent container cannot reach. The re-base strengthened this rather than weakened it, because the split it grades on is no longer a file the solver could read.

## Reward contract

- reward path: `/logs/verifier/reward.txt`, carrier: bare float
- score document: `/logs/verifier/score.json`
- The live seed/forge/verifier.py binds the reward carrier to reward.txt, not reward.json. The bare float goes to /logs/verifier/reward.txt and the machine-readable reason and metric block go to the companion /logs/verifier/score.json. Both are written from the same trap-guarded path, so the instrument's binding and the reason carrier never disagree.
- /logs/verifier/ is a single shared host path and concurrent lanes collide on it. Local exercise of tests/test.sh redirects the reward root with OER13_REWARD_ROOT to a lane-private directory. The bound contract path inside the bundle stays /logs/verifier/reward.txt; only the local exercise is redirected.

## Tier

- target tier: Frontier-defeat, tier_exemption_granted: none
- anchorable tier: capped below Hard by the standing batch-level Perception-axis exemption
- This task family is text and code only, so the Perception axis carries no category; the exemption is batch-level, recorded once, closes by nothing, and no perception lever is authored to fill it. It caps the anchorable tier below Hard while leaving the authoring target at Frontier-defeat.
