# FORGE-CANARY-BEGIN
# 0: cb68fcf4edd23115418a4a4aff1eba378fbb985a943559a6b738278df8e00894
# 1: e55d95b55846a9cb6827852a22a5aa83d15c592b500252ce912240ee28e9d0d2
# 2: 8001e93c586f32494b5b48b6d5a8c6b6f9c0c7ccdfd49525cee31d7fbf995e65
# 3: f9064f9c8acd53d277b6e640ce5f422ad21e01742550c86ca8977ef03482f4d2
# FORGE-CANARY-END
# TRUTH.md, slot OER-21

GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.

## What is graded

The worst per-shard RELATIVE perplexity degradation of the quantized model against the unquantized reference, in percent of that shard's unquantized reference perplexity, measured by the verifier over the frozen evaluation corpus, at a fixed total bit budget.

The reading is the WORST per-shard degradation, not the mean, so a favourable shard
cannot be harvested. It is recomputed by tests/evaluate.py inside the verifier's
process, from the tensors the harness quantized itself.

## The measured plateau, which is the AR5 pressure

| quantity | value |
|---|---|
| shipped default configuration | 1.188591942 |
| optimum of the handed option set, over 25 configurations | 1.188591942 |
| optimum of the wider scheme space | 0.107170959 |
| measured gap, handed optimum minus wider optimum | 1.081420983 |
| bit budget | 2496 |
| bits the reference actually allocates | 2440 |

Both optima are measured on this substrate by search, not asserted. The handed
option set is swept exhaustively; the wider space is swept by a fixed greedy
coordinate descent. An agent that accepts the handed option set plateaus at the
second row and cannot reach the third, which is what makes this slot AR5.

## Why the widening is derivable and not a puzzle

The objective states that the budget is a TOTAL over the model and that degradation is measured over the whole corpus. Nothing in the objective requires one width for every tensor. The instruction states the accounting rule in full, states that per-tensor and per-group allocation are admitted, and states that the graded bar is the handed toolkit's own optimum. The widening is therefore an inference from the stated objective, not a hidden affordance behind a puzzle.

## Anchors

anchors_state: absent, under gap-oer-per-family-anchors-unmeasured.
baseline_metric and target_metric are null and no number is invented for them.
The bounded scaling uses two MEASURED endpoints on this substrate, named in
grounding.yaml under reward.local_scaling_endpoints, which are not the family
anchors and never stand in for them.

## Reward

Carrier /logs/verifier/reward.txt, one bare float on [0.0, 1.0], higher better, never binary.
Reason and metric block in /logs/verifier/score.json.
Formula: raw = (baseline - agent) / (baseline - target); score = min(max(raw, 0.0), 1.0)

## Declared gaps

- `gap-oer-per-family-anchors-unmeasured` (batch): F9 carries no measured baseline or target, so this slot binds the reward schema and declares the anchors absent.
- `gap-oer-budget-field-name-collides-with-its-role` (batch): budget_hours 0.12 is bound provisionally because the upstream field name carries a value that belongs to the session-level role.
- `gap-oer-solver-egress-ruled-stricter` (batch): the contract field reads setup-only and this slot binds open at the agent surface; the divergence is recorded and the field is not amended.
- `gap-oer21-surrogate-substrate` (slot): The graded forward pass is a deterministic surrogate over 12 tensors of 48 weights and a vocabulary of 8, not a live transformer forward pass on the pinned image. Quantization, bit accounting, the codebook families and the perplexity definition are real and are computed over the actual quantized tensors, but the model is not a language model and the corpus is not text.
- `gap-oer21-ordering-checker-substituted` (slot): evaluation_follows_state_seal cannot be fired by any submission, because the phase log is written by the verifier's own process and the submission runs as a separate process group that never touches it. Its rejecting half is carried on a planted telemetry fixture. The substitution is recorded in tests/checkers.yaml, in feasibility.yaml and in seed/tasks/OER-21/shortcut_probes.yaml.
