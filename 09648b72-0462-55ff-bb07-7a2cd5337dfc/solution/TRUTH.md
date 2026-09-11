# FORGE-CANARY-BEGIN
# 0: 2d7661d592762d173ffb5a12e7152fdfd9dc924ea0013a547e2beb5a107c56d8
# 1: a9723a177fed597f03ecbd303f752dfb6466642c130465accb7984b21d496f8d
# 2: 84897e76caafafb109b6616e3e18e535e0ecc3cafa55cba6e4fb5cbff73ac240
# 3: c392eb2772942419a531faf9231a4ad4c84a3f0da3d12f9c45017c1f16bf8607
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml -->

# TRUTH: OER-08, Data mixture search breadth under a fixed token budget

Family F12, data curation. Archetype AR5, Adversarial Option Expansion.

## The graded quantity

The graded loss is the verifier's own unsmoothed recomputation from the count table its executor accumulated, on the frozen split. It is never a number the training loop reported, never a number in the submission's stdout, and never a field the submission wrote. A submission may EMA its own reported loss in report.json for its own use; nothing there is graded.

## Why the handed parameterisation is a trap

No source in the pool is on-distribution. Every source carrying an on-distribution profile also carries junk documents, and a constant weight over a source buys both in fixed proportion. The convex hull of the five whole-source compositions therefore does not contain the evaluation distribution, and the best point in that hull is bounded away from it. The gap is 0.483 nats.

The objective constrains the TOTAL number of tokens fed. It does not say the allocation is one number per source, does not say a source is indivisible, and does not say a document is consumed once. Allocating inside a source (selection) and allocating across repeats (repetition) are both allocations of the same fixed token budget, at a finer granularity than the template can express. Both are load-bearing: selection alone cannot spend the budget, because the on-distribution documents in the whole pool total 3500 tokens against a 12000 token budget, so the wider construction needs repetition to fill the budget with the right composition.

environment/plan_schema.md documents the full grammar, environment/mixture.yaml says in its first lines that it is a template and not a boundary, and instruction.md states that the free variable is the allocation of the budget across sources and that nothing in that sentence restricts granularity. The wider construction is derivable from the stated objective and is behind no puzzle.

## The reference derivation

1. **read-objective** The graded quantity is validation loss recomputed by the verifier at budget exhaustion. The free variable is the allocation of a fixed token budget across sources.
2. **measure-dev-probe** environment/harness.py scores a plan against the agent-visible dev probe. The probe is two A, two B and two C documents, so the target composition is the equal mean of A, B and C.
3. **profile-the-pool** Score each single-source constant mixture on the probe. Every source that carries an on-distribution profile also carries junk in fixed proportion, so no whole source and no blend of whole sources reaches the target composition.
4. **test-the-template** Sweep the constant-weight template. It plateaus: its best point is bounded away from the target composition because the target is outside the convex hull of the whole-source compositions.
5. **expand-the-option-set** Re-read the objective. It constrains total tokens, not granularity. plan_schema.md accepts per-document draws with repetition, which are allocations of the same budget at a finer granularity.
6. **select-inside-sources** Take only the on-distribution documents: the A documents of reference-a, the B documents of web-crawl, the C documents of technical.
7. **repeat-to-fill-the-budget** Those documents total 3500 unique tokens against a 12000 token budget, so selection alone cannot spend the budget. Repeat each selected document so the budget is filled with the selected composition.
8. **balance-to-the-target** Allocate the budget equally across the three on-distribution profiles: forty document instances each, one hundred tokens each, twelve thousand tokens exactly.
9. **verify-the-frozen-axes** Total fed equals the budget exactly. No document outside the pool is named. The graded readout contract is untouched. Submit.

## The numbers

| quantity | value |
|---|---|
| token budget | 12000 |
| constant-weight template optimum, whole split | 2.938015294961008 |
| constant-weight template optimum, fold_a | 2.722419873461009 |
| constant-weight template optimum, fold_b | 3.1536107164610057 |
| reference optimum, whole split | 2.455022017862182 |
| reference plan, whole split | 2.455022017862182 |
| reference plan, fold_a | 2.453764280558667 |
| reference plan, fold_b | 2.4562797551656974 |
| separation | 0.4829932770988261 nats |
| reference reward | 1.0 |

## Anchors

`anchors_state: absent`. baseline_metric and target_metric are absent under `gap-oer-per-family-anchors-unmeasured`. The two calibration points above are substrate-local, are recomputed by the verifier from frozen bytes on every run, and are carried under `gap-oer08-magnitude-normaliser-is-substrate-local`. They are not the family anchors and are not comparable across slots.

## Statement ambiguity

Reading one: Reading instruction.md's What is free section alone: the free variable is the allocation of the fixed token budget across sources, and the graded quantity is the verifier's recomputed validation loss at budget exhaustion.

Reading two: Reading environment/plan_schema.md's What is frozen and What is free sections alone: total_tokens is frozen, the estimator is frozen, the evaluation split is frozen, and the allocation of the budget is free.

Reduces to: Both readings reduce to the same graded quantity: minimise the verifier's unsmoothed recomputation of validation loss on the frozen split, computed from the harness-owned count table at the point where exactly budget_tokens tokens have been fed. There is exactly one graded outcome. No second quantity is graded, no tie-break is left open, and neither reading admits a different optimum.

## Tier

Target tier Frontier-defeat, tier_exemption_granted none, anchorable tier capped below Hard.

This task family is text and code only, so the Perception axis carries no category. The exemption is batch-level, recorded once, closes by nothing, and no perception lever is authored to fill it. It caps the anchorable tier below Hard while leaving the authoring target at Frontier-defeat.

Nothing in this bundle claims a difficulty. The design obligations above are design obligations and are never stated as a difficulty argument.

## Gaps

- `gap-oer-per-family-anchors-unmeasured` (carried): baseline_metric and target_metric are absent; the reward schema is bound in full and no anchor number is invented.
- `gap-oer-budget-field-name-collides-with-its-role` (carried): budget_hours is bound to 0.12 and marked provisional in task.toml.
- `gap-oer-solver-egress-ruled-stricter` (carried): the bundle binds open at the agent surface and records the divergence above rather than amending the batch field.
- `gap-oer08-magnitude-normaliser-is-substrate-local` (declared by this lane): the magnitude term is normalised between two substrate-local calibration points recomputed by the verifier, which stand in for absent anchors. They are not comparable across slots and are not published evidence. Closes when F12 anchors are measured.
- `gap-oer08-split-leak-detected-by-exact-content-digest-only` (declared by this lane): evaluation_split_untrained compares exact document content digests, so it catches a copied or re-labelled held-out document and does not catch a paraphrase or a near-duplicate. No near-duplicate floor is calibrated for this substrate, so none is authored. The load-bearing structural control is that no pool document id resolves to the split and the split lives verifier-side. Closes when a near-duplicate floor is calibrated.
