# FORGE-CANARY-BEGIN
# 0: a92223384d1276b66eca4dd8b1da293a057973ca485f29572f5a1f483799d2f7
# 1: 4bd50d732b351ef5f7a71b57847e1ac220adee516507c9e3d544cddc7f04ec3c
# 2: 7d88ac9790135addcca1e9465e4468ebfc875b9b93129534f29aa518caf15eb9
# 3: 4d347cd5c62f94ea1add659fa97e53a362c2b86bf1a3f58fbabf264a527ff96e
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. -->
<!-- Source: solution/grounding.yaml -->

# TRUTH, slot OER-20

Family F9, post-training quantization. Primary archetype AR7, Ambiguous Intermediate States, exactly one.

## What the re-base moved

This slot previously graded arithmetic over JSON statistics. Its model was a twelve-row table of per-tensor `sensitivity` and `outlier_factor` coefficients describing a three-block stand-in that was never instantiated, its perplexity was a closed form of the shape `ppl_reference * exp(error * shard_gain)`, and its unquantized reference was a five-row table of numbers no model produced. Nothing was ever run, so the first and third parts of the simulator test both failed: there were no weights in the loop, and deleting the forward pass changed nothing because there was no forward pass to delete.

The statement survived the re-base and the graded path did not. The solver still designs a per-tensor bit allocation and a quantization scheme under a fixed bit budget, and is still graded on the lower edge of a separation band with three outcomes rather than two. What changed is that the quantized model is now the nanoGPT checkpoint, the fifty weight matrices of the canonical 12-layer 768-dim decoder declared in `environment/nanogpt_substrate.json`, and the graded perplexity is now obtained by the verifier running that checkpoint forward over a held-out FineWeb10B validation slice that is absent from every agent-visible container. This was a replacement of the training and evaluation path and not a declaration-level repin, and it is recorded as such rather than described as smaller than it was.

## The reference allocation

Scheme `error-feedback`, allocation payload sha256 `a4b31188e8e32d78b5583d336aad4240307a0c2e37cbacc9f79baec8dd1893cb`.

| tensor | bits |
|---|---|
| `wte.weight` | 3 |
| `blocks.0.attn.qkv.weight` | 6 |
| `blocks.0.attn.proj.weight` | 5 |
| `blocks.0.mlp.fc.weight` | 5 |
| `blocks.0.mlp.proj.weight` | 4 |
| `blocks.1.attn.qkv.weight` | 6 |
| `blocks.1.attn.proj.weight` | 5 |
| `blocks.1.mlp.fc.weight` | 5 |
| `blocks.1.mlp.proj.weight` | 4 |
| `blocks.2.attn.qkv.weight` | 6 |
| `blocks.2.attn.proj.weight` | 5 |
| `blocks.2.mlp.fc.weight` | 5 |
| `blocks.2.mlp.proj.weight` | 4 |
| `blocks.3.attn.qkv.weight` | 6 |
| `blocks.3.attn.proj.weight` | 5 |
| `blocks.3.mlp.fc.weight` | 5 |
| `blocks.3.mlp.proj.weight` | 4 |
| `blocks.4.attn.qkv.weight` | 6 |
| `blocks.4.attn.proj.weight` | 5 |
| `blocks.4.mlp.fc.weight` | 5 |
| `blocks.4.mlp.proj.weight` | 4 |
| `blocks.5.attn.qkv.weight` | 6 |
| `blocks.5.attn.proj.weight` | 5 |
| `blocks.5.mlp.fc.weight` | 5 |
| `blocks.5.mlp.proj.weight` | 4 |
| `blocks.6.attn.qkv.weight` | 6 |
| `blocks.6.attn.proj.weight` | 5 |
| `blocks.6.mlp.fc.weight` | 5 |
| `blocks.6.mlp.proj.weight` | 4 |
| `blocks.7.attn.qkv.weight` | 6 |
| `blocks.7.attn.proj.weight` | 5 |
| `blocks.7.mlp.fc.weight` | 5 |
| `blocks.7.mlp.proj.weight` | 4 |
| `blocks.8.attn.qkv.weight` | 6 |
| `blocks.8.attn.proj.weight` | 5 |
| `blocks.8.mlp.fc.weight` | 5 |
| `blocks.8.mlp.proj.weight` | 4 |
| `blocks.9.attn.qkv.weight` | 6 |
| `blocks.9.attn.proj.weight` | 5 |
| `blocks.9.mlp.fc.weight` | 5 |
| `blocks.9.mlp.proj.weight` | 4 |
| `blocks.10.attn.qkv.weight` | 6 |
| `blocks.10.attn.proj.weight` | 5 |
| `blocks.10.mlp.fc.weight` | 5 |
| `blocks.10.mlp.proj.weight` | 4 |
| `blocks.11.attn.qkv.weight` | 6 |
| `blocks.11.attn.proj.weight` | 4 |
| `blocks.11.mlp.fc.weight` | 5 |
| `blocks.11.mlp.proj.weight` | 4 |
| `lm_head.weight` | 3 |

The budget is spent exactly, to the bit: 648806400 allocated against 648806400 budgeted, a mean of 4.0 bits per parameter over all 162201600 parameters. Spend is moved off the token embedding and the output head, the two largest matrices and the two whose rows are closest to uniform in scale, and onto the twelve fused query-key-value projections, whose rows carry the widest dynamic range in the checkpoint. Eleven of the twelve attention output projections take a fifth bit and the twelfth stays at the control width, because that is the residue that makes the budget divide exactly. The scheme is error-feedback, per-row affine quantization with the rounding residual carried along the row, which is the lowest-error scheme in the closed set. The allocation is deliberately not maximally non-uniform, because a more aggressively fitted quantizer makes the reading depend on which tokens a held-out batch happens to carry; the reference trades mean for band width so that the LOWER EDGE clears the margin rather than the mean.

## What the verifier measures

| quantity | value |
|---|---|
| graded artifact | `/checkpoint/nanogpt_fp32.pt`, float32 parameter snapshot of the frozen 12-layer 768-dim decoder |
| graded evaluation | real forward passes over the verifier-owned held-out FineWeb validation slice |
| readings per point | three over one token batch: unquantized, submitted allocation, harness control |
| allocated bits | 648806400 |
| budget bits | 648806400 |
| mean bits per parameter | 4.0 |
| verifier-derived evaluation order | absent, derived at run time from the digest of the verifier's own held-out manifest, which is not a bundle byte |
| separation mean | absent, unmeasured |
| noise half width | absent, unmeasured |
| lower edge of the band | absent, unmeasured |
| bound separation margin | 0.05 |
| verdict | absent, unmeasured |
| reward | absent, unmeasured |

Every measured row above is published as absent because this lane ran no checkpoint and no forward pass. The previous bytes carried a separation mean of 0.135696 and a noise half width of 0.002966; those were readings of the retired arithmetic surrogate, they describe nothing that now exists, and no number replaces them. The first pilot that runs this slot on a real checkpoint fills them in.

## The three outcomes, which are not two

### established

- reason: `separation-margin-cleared`
- condition: `lower_edge = separation_mean - noise_half_width >= 0.05`
- reward: 1.0

### failed

- reason: `degradation-past-bar`
- condition: `upper_edge = separation_mean + noise_half_width < 0.0`
- reward: 0.0
- gated by: `allocation_not_degrading_past_bar`

### unproven

- reason: `significance-unestablished-at-ceiling`
- condition: `upper_edge >= 0.0 and lower_edge < 0.05, at the bound evaluation ceiling of 5 points`
- reward: clamp(lower_edge / 0.05, 0.0, 1.0), a value in [0.0, 1.0)

The run COMPLETED. Every gate held. Nothing failed and nothing was established. The evaluation budget was the binding limit: five held-out points were not enough to separate this allocation from the control at the bound margin. This is a THIRD outcome and it is never reported as either neighbour.

## The reward ramp

The reward is continuous in the measured separation and does not flip at the margin. The verdict flips at the margin because a verdict is a claim; the number does not, because the measurement did not.

`score_separation(separation_mean, noise_half_width, margin) = clamp((separation_mean - noise_half_width) / margin, 0.0, 1.0)`

## Golden trajectory

**Step 1.** Read instruction.md, environment/nanogpt_substrate.json and environment/model_stats.json. Note that the model is the frozen 12-layer 768-dim decoder, that the budget is fixed at four bits per parameter over its fifty weight matrices, and that the held-out split is not in the container.

> holds: the metric is perplexity degradation, lower better; the verifier computes it by running the model

**Step 2.** Compute the control allocation by hand: every tensor at 4 bits under rtn. Observe that it spends the budget exactly and that its separation against itself is identically zero, so a no-op submission establishes nothing.

> holds: a no-op is unproven at reward 0.0, not a failure

**Step 3.** Load the checkpoint and inspect the real weight distributions. Move bits off wte.weight and lm_head.weight, the two largest matrices, onto the fused qkv projections, whose rows carry the widest dynamic range. Keep the total at or under budget.

> holds: reallocating is the task; raising the total is a different task and scores zero

**Step 4.** Run environment/quantize_lab.py on the candidate. It quantizes the real tensors and measures perplexity by running the model on the train split. Read the per-point separations, not just the mean.

> holds: the band is the measurement; the mean alone is not, and a train-split reading is not the graded reading

**Step 5.** Observe that pushing further from uniform improves the mean and widens the band, because a harder-fitted quantizer makes the error depend on the batch. Stop where the LOWER EDGE is maximal rather than where the mean is maximal.

> holds: establishing a result and having one are different; the lower edge is what is graded

**Step 6.** Select scheme error-feedback and write allocation.json. Report no perplexity number.

> holds: a reported number is recorded beside the graded path and never on it

## Anchors

- `anchors_state`: absent
- gap: `gap-oer-per-family-anchors-unmeasured`
- `baseline_metric`: absent, not invented
- `target_metric`: absent, not invented

The live instrument grades a within-run separation against the harness's own control allocation, measured on the same held-out points in the same verifier-derived order, from the same checkpoint, against an unquantized reading recomputed in the same process. That difference is anchor-free by construction, so the slot is gradable once a pilot runs it and still refuses to invent a baseline number. The anchored formula is carried unevaluated, which is what an absent anchor looks like when it is reported honestly.

## Declared gaps

- `gap-oer-per-family-anchors-unmeasured`: F9 baseline_metric and target_metric are unmeasured. Neither is invented here. The reward schema is bound in full and the anchored formula is carried unevaluated.
- `gap-oer-budget-field-name-collides-with-its-role`: budget_hours 0.12 is bound as the per-attempt bound and marked provisional, because the upstream field name carries value 6 while the per-attempt bound is stated as 7.2 minutes.
- `gap-oer-solver-egress-ruled-stricter`: The batch field reads setup-only and this lane binds open. Recorded under solver_egress_divergence and not amended here.
- `gap-oer-20-checkpoint-digest-unrecorded`: The frozen checkpoint is produced once by environment/train_nanogpt.py under the pinned harness and mounted read-only on both surfaces. This lane ran no training, so environment/model_stats.json publishes `checkpoint_sha256` as absent rather than inventing one, and the verifier compares the mount against the manifest the harness writes beside it.
- `gap-oer-20-calibration-discovery-value-unmeasured`: The calibration probe is repinned from a measured discovery value to a determinism identity, because the value the retired surrogate produced describes nothing that now exists and this lane measured no replacement. The identity is load-bearing on its own terms and a substantive measured centre is owed by the first pilot.
- `gap-oer-20-per-attempt-budget-against-real-forward-passes`: The per-attempt bound of 7.2 minutes was set against an arithmetic evaluation that ran in milliseconds. The graded evaluation now runs fifteen forward passes per submission over the held-out points, plus the calibration probe, and the bound has not been re-measured against that cost. It is carried unchanged and flagged rather than adjusted by guess.
- `gap-oer-20-generated-carriers-not-regenerated`: solution/grounding.yaml is the source of the generated carriers and was not rewritten by this lane. tests/calibration.py, tests/test_output.py and this file are repinned in place and are correct; regenerating them from an updated grounding.yaml is the orchestrator's, and until that runs the generated-from line above names a source that no longer derives every byte below it.

## Retired

- `gap-oer-f9-substrate-is-a-deterministic-surrogate`: CLOSED by this re-base. The substrate is no longer a deterministic quantization-error and perplexity surrogate over frozen per-tensor statistics. The quantized model is the nanoGPT checkpoint, the perplexity is a real forward pass, and the evaluation split is the verifier's.
