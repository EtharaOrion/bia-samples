# FORGE-CANARY-BEGIN
# 0: 4108da2bd2b97f0ba5aa63ba98ef677ac40c1f690559bc3d766b65573512a465
# 1: cc831f1904d935255852bea3a58d10a34ebf14f2b355495521f4428d2c52f6a1
# 2: 83311bd43b60a185fb3109b476f7b5c826a58259944bd3875dc1a290f75c82d3
# 3: 1697a538fac469d2d7ece77082d0b022e7f041bd21699ce461860859526ddc66
# FORGE-CANARY-END
# TRUTH.md, slot OER-21

GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.

## What this pass did, stated plainly

The model and the evaluation path were REPLACED, not repinned. The predecessor bundle graded a formula-generated weight table of twelve tensors of forty-eight weights over a vocabulary of eight, whose value at a position was (context * stride_context + word + tensor * stride_tensor) modulo tensor_size, read by an arithmetic row builder that no language model produced. There is no declaration edit that turns that into a nanoGPT run, so the substrate, the forward pass, the evaluation corpus, the reference readings, the toolkit, the schema, the checkers that read them and every generated artifact downstream of them were rewritten.

This was a replacement of the model and of the evaluation path. It was not a repin. A repin would have moved a declaration and left the graded arithmetic where it was, and there was no graded arithmetic here worth keeping: the predecessor's forward pass was an index formula over a table, so there was nothing to repoint at a nanoGPT run.

The statement, the archetype and the reward schema. A solver still designs a per-shard bit allocation and quantization scheme under a fixed total bit budget, and is still graded on the worst per-shard relative perplexity degradation against the unquantized reference, lower better, on a bounded continuous reward with every zero carrying a machine-readable reason.

## What is graded

The worst per-slice RELATIVE perplexity degradation of the quantized nanoGPT checkpoint against the unquantized reference, in percent of that slice's unquantized reference perplexity, measured by the verifier running real forward passes of the frozen twelve-layer decoder over held-out FineWeb slices the solver cannot see, at a fixed total bit budget.

For one held-out slice the verifier runs two real forward passes of the canonical decoder, one at full precision and one over the parameters as the harness quantized them. The reference reading is the exponentiated mean entropy of the full-precision next-token distribution on that slice and the quantized reading is the exponentiated mean cross entropy of that distribution under the quantized one. Gibbs' inequality makes the second at least the first, with equality only when the two distributions coincide, so no scheme can buy a better-than-reference reading out of quantization noise. The true-token perplexity of both parameter states is carried alongside as telemetry so a reader can see the checkpoint is a language model on real text.

The reading is the WORST per-slice degradation, not the mean, so a favourable slice cannot be harvested. It is recomputed by tests/evaluate.py inside the verifier's process, from the parameters the harness quantized itself, on slices the solver cannot reach.

## The substrate

| quantity | value |
|---|---|
| architecture | 12 layers, model_dim 768, head_dim 128, num_heads 6, vocab 50304 |
| quantizable parameter shards | 74 |
| quantizable parameters | 162201600 |
| residue held at checkpoint width, outside the budget | 153216 |
| bit budget | 648807584 |
| held-out slices, verifier-owned | 5 of 32768 tokens |

162201600 quantizable parameters at 4 bits is 648806400 payload bits, plus one 16-bit scale per shard over 74 shards is 1184 overhead bits, total 648807584. The handed default configuration therefore lands exactly on the budget, which is what makes it look final. Every value in this derivation is a function of the architecture and none of them is written down anywhere a reader would have to trust.

## The AR5 pressure

The embedding matrix and the untied output projection are 77266944 of the 162201600 quantizable parameters, and they carry a different sensitivity to width than the attention and MLP matrices do. A uniform width spends the same bits per parameter on both. That asymmetry is a property of the frozen architecture rather than of a hint, and it is the whole of the AR5 pressure.

Both endpoints of the bounded scaling are measured by the verifier at grading time and neither is stored. The handed option set is swept exhaustively and the wider space is probed by a fixed, ordered, budget-checked list of role-keyed allocations. The probe is a bar and it is never called an optimum of the wider space, because nothing in this bundle has measured that optimum.

## Why the widening is derivable and not a puzzle

The objective states that the budget is a TOTAL over the checkpoint and that degradation is measured over the whole held-out split. Nothing in the objective requires one width for every shard. The instruction states the accounting rule in full, states that per-shard and per-group allocation are admitted, and states that the graded bar is the handed toolkit's own optimum. The widening is therefore an inference from the stated objective, not a hidden affordance behind a puzzle.

## Anchors

anchors_state: absent, under gap-oer-per-family-anchors-unmeasured. baseline_metric and target_metric are null and no number is invented for them. The bounded scaling uses two endpoints the verifier measures at grading time, named in grounding.yaml under reward.local_scaling_endpoints, which are not the family anchors and never stand in for them.

## What is not measured

Every constant the predecessor carried under this key was measured on the formula-generated table this pass deleted, so none of them describes this substrate. Re-measuring them needs the frozen checkpoint and one accelerator, which is a measurement pass and not an authoring pass. They are published as unmeasured rather than carried across, and the two scaling endpoints the reward needs are recomputed by the verifier at grading time so the slot grades without them.

## Reward

Carrier /logs/verifier/reward.txt, one bare float on [0.0, 1.0], higher better, never binary.
Reason and metric block in /logs/verifier/score.json.
Formula: raw = (baseline - agent) / (baseline - target); score = min(max(raw, 0.0), 1.0)

## Declared gaps

- `gap-oer-per-family-anchors-unmeasured` (batch): F9 carries no measured baseline or target, so this slot binds the reward schema and declares the anchors absent.
- `gap-oer-budget-field-name-collides-with-its-role` (batch): budget_hours 0.12 is bound provisionally because the upstream field name carries a value that belongs to the session-level role.
- `gap-oer-solver-egress-ruled-stricter` (batch): the contract field reads setup-only and this slot binds open at the agent surface; the divergence is recorded and the field is not amended.
- `gap-oer21-checkpoint-digest-unmeasured` (slot): The frozen checkpoint is pinned by architecture and by provisioning path but not yet by digest, because this lane did not build the snapshot and will not write a sha256 it has not taken. tests/anchors.json records the pin as absent, and frozen_checkpoint_and_holdout_unmodified still requires the verifier to digest the parameters it actually graded, so an absent pin never becomes an absent observation. The gap closes when the provisioning pass hands back the built snapshot and its digest is written into the pin.
- `gap-oer21-endpoints-unmeasured-on-the-nanogpt-substrate` (slot): No endpoint in this slot has been measured on the replaced substrate. The reward does not depend on one, because both scaling endpoints are recomputed by the verifier at grading time, but the private measured block carries nulls rather than the numbers the predecessor measured on the formula-generated table. What this leaves unproven is named rather than glossed: that the verifier's widening probe reads strictly below the handed option set's optimum on the frozen checkpoint, which is the AR5 separation this slot exists to create. The machinery was exercised end to end on an untrained smoke snapshot, where the separation is zero because an untrained decoder has no per-role structure for an allocation to exploit, and the run correctly reported scaling-span-nonpositive rather than inventing a gap. The gap closes when the frozen checkpoint is provisioned and solution/recompute.py --measure records a positive span.
- `gap-oer21-ordering-checker-substituted` (slot): evaluation_follows_state_seal cannot be fired by any submission, because the phase log is written by the verifier's own process and the submission runs as a separate process group that never touches it. Its rejecting half is carried on a planted telemetry fixture. The substitution is recorded in tests/checkers.yaml, in feasibility.yaml and in seed/tasks/OER-21/shortcut_probes.yaml.

## Retired gaps

- `gap-oer21-surrogate-substrate` retired by this re-base: The gap recorded that the graded forward pass was a deterministic surrogate over twelve tensors of forty-eight weights and a vocabulary of eight rather than a live transformer forward pass. It is retired because the surrogate was deleted rather than annotated: the graded forward pass is now the canonical twelve-layer decoder run over held-out FineWeb tokens on the pinned image.
