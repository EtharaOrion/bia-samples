# FORGE-CANARY-BEGIN
# 0: e850eda16c69fc5d37e11dcf04a2b53b8b5812be20eb965eaa70ea2862fc446f
# 1: f8096fe8feb3f35a4693baeccda6ac151ba629bbe2f16c4829bc622bce665d59
# 2: 931f8f9c44d9bbfb3d300378c76bc65fd23eff0bcad1978f2fd8912f8e552a6b
# 3: 73bbbd75ec1adf6b1f3bcbcc1d50daf6933829cf894664d6ef98d0efafb25e90
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml -->

# TRUTH.md — OER-12

What is actually true about OER-12, stated once, in the place the bundle derives it from.

## The graded quantity

One quantity is graded: the validation loss of the trained model, lower is better. The verifier computes it itself on the frozen held-out split from the weights the harness owns, unsmoothed, at the bound evaluation point, and it must hold at every sustain point the verifier schedules. It is never a number the submission reported, printed or wrote.

## The third state

The corpus carries records that neither fail to parse nor parse to a valid record. That third state is machine-readable, it is graded, and it must be resolved by a policy the pipeline declares before it classifies anything. There is no correct policy among the three declarable ones; there is a correct way to hold one.

## What a silent coercion costs

A pipeline that drops the ambiguous class into the failure bucket, or admits it as valid, diverges from the harness's own independent classification of the same bytes and scores zero with the reason ambiguous-class-coerced-silently. That reason is not a parse-failure reason, because collapsing a third state into a neighbour is a different defect from correctly rejecting a broken record.

## What the anchors are

Absent. This family's baseline and target are unmeasured and this bundle declares them absent under gap-oer-per-family-anchors-unmeasured rather than inventing them. The reward schema is bound in full. A live grading run therefore resolves to 0.0 with the reason anchors-absent-metric-unanchored, which is an evidence state and not a verdict on any submission.

## What was and was not executed

seed/tasks/OER-12/adequacy.py drives the LIVE checkers in tests/ over fixtures and its observed exit code is recorded in seed/tasks/OER-12/feasibility.yaml. No model was trained anywhere in this lane, and the loss curves in this file are declared fixture inputs rather than measurements, carried as gap-oer-12-loss-curves-are-declared-fixtures.

## Declared gaps

- **gap-oer-per-family-anchors-unmeasured** — F13 has no measured baseline or target, so this slot binds the reward schema and declares the anchor numbers absent. Closes by: a measured anchor pair for F13 recorded in seed/contract.yaml from published evidence.
- **gap-oer-budget-field-name-collides-with-its-role** — budget_hours is bound to 0.12 as provisional because the requirements file names the field with a value whose stated role is the across-attempts terminator. Closes by: the contract owner correcting requirements/ and Phase 0.5 re-approving.
- **gap-oer-solver-egress-ruled-stricter** — the batch field reads setup-only under the stricter reading while this bundle binds open at the agent surface; the divergence is recorded and the batch field is not amended by this lane. Closes by: the client resolving the contradiction between the two requirements lines.
- **gap-oer-12-loss-curves-are-declared-fixtures** — no model was trained in this lane, so every validation loss in the fixtures is a declared input rather than a measurement, and no part of this bundle presents one as measured. Closes by: a pilot run on the bound envelope producing measured curves.
- **gap-oer-12-fixture-anchors-are-fixture-local** — the accepting half needs a non-null anchor pair to drive the real reward function to full reward, so the fixture carries one that is explicitly fixture-local and is never presented as this family's anchors. Closes by: the same measured anchor pair that closes gap-oer-per-family-anchors-unmeasured.
