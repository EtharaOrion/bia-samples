# FORGE-CANARY-BEGIN
# 0: 5cbc7175a74bfbb05e9cb7f6673e039db6bb8a6e9e50955387e7fcd357f364ea
# 1: d153fa70d7b9dcffe662b7bfb0969be60ad8782ba93c5c9f66a296acae210628
# 2: a2d619d81034107fd23657152255520dedc30881cc1aef308b2374b338f8335b
# 3: 4274a0b18d282c6045bf945aa2f4792da8380ceea9086c4b40060fedd133a466
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml. -->

# TRUTH, slot OER-01

## What is graded

The number of optimizer steps at the first evaluation point, on the grid the
verifier itself schedules, whose **verifier-recomputed, unsmoothed** loss on the
frozen held-out split falls below 3.28 and stays below it across the next 3 scheduled points.

| bound quantity | value |
|---|---|
| baseline_metric | 3250 |
| target_metric | 2690 |
| target validation loss | 3.28 |
| pass_threshold | 0.65 |
| evaluation cadence, steps | 50 |
| sustain window, evaluation points | 3 |
| corpus revision | r46 |
| proximity floor | 0.15 |

reward = min(max((3250 - agent_metric) / (3250 - 2690), 0.0), 1.0)

## The reference

The reference keeps the orthogonalised-momentum family that record 36 and record 46 both sit in, and changes what is coupled rather than which primitives appear. Two changes carry it. First, the per-layer RMS rescale is replaced by a spectral trust region whose radius follows a two-phase schedule, so the early steps are allowed a larger effective step while the late steps are bounded by a running spectral-norm estimate rather than by a fixed constant. Second, the auxiliary AdamW group shares one second-moment estimator with the hidden-matrix group instead of maintaining its own, which removes the aux group's independent scale drift and lets a single learning-rate schedule govern both. The initialisation is depth-scaled orthogonal rather than the framework default, so the trust region starts from a known spectral radius instead of an empirical one.

Its structural digest is `8ae1e92312bdd102b17b988f0741914e`.

Its structural digest differs from every entry in the pinned corpus, because the update chain contains spectral_trust_region where every corpus entry contains rms_rescale or soap_preconditioner, and because the coupling shared_second_moment appears in no corpus entry. The screen therefore clears it on the structural comparison alone, before proximity is consulted.

## Why the fingerprint and not the network policy

The deny-list names github.com, raw.githubusercontent.com and objects.githubusercontent.com. It is not the control. Mirrors, package caches, paraphrased write-ups, forum posts and model weights all carry a record recipe past any deny-list, and the model doing the solving may simply remember one. A network denial therefore proves nothing about whether a submitted recipe is a replay. The load-bearing control is the deterministic algorithm fingerprint screen in tests/fingerprint.py, checked against the pinned exclusion set in tests/corpus.json before any seed executes, so a replay is rejected at zero accelerator cost. Network policy is a cost and hygiene measure and is never presented in this bundle as the record-displacement control.

## What the golden trajectory is and is not

Every loss value below is authored here as the derivation source for the checker fixtures. It is NOT a measurement of an H100 run. No training run has been performed by this lane. What the accepting half of the adequacy harness proves is that the live checkers accept a trajectory of this shape produced by THIS reference, bound by the sha256 of solution/reference.py. It proves nothing about what an accelerator would produce, and feasibility.yaml says so in the same words.

## Expected outcome for the reference

| quantity | value |
|---|---|
| graded step | 2600 |
| reward | 1.0 |
| reason | crossing-sustained-and-recipe-displaced |

## Declared gaps

- `gap-oer-budget-field-name-collides-with-its-role`: budget_hours is bound to 0.12 and marked provisional in task.toml.
- `gap-oer-solver-egress-ruled-stricter`: solver_egress is bound open here while the contract field reads setup-only. Recorded, not amended.
- `gap-oer01-reward-path-extension-collides`: The lane spec names the bound reward contract path /logs/verifier/reward.json while the live instrument seed/forge/verifier.py binds REWARD_PATH = /logs/verifier/reward.txt and refuses any other declared reward_path. Both are authorities over these bytes and they disagree on the extension.
- `gap-oer01-accelerator-reference-hours-unmeasured`: No accelerator run was performed by this lane. reference_hours as recorded in feasibility.yaml is the measured wall-clock of the reference derivation path and the full live gate chain on CPU over derived fixtures. The accelerator wall-clock a real graded attempt would consume is UNMEASURED.
- `gap-oer01-golden-trajectory-is-derived-not-measured`: Every loss value in golden_trajectory is authored in this file as a derivation source. It is not a measurement. The accepting half of the adequacy harness therefore proves that the live checkers accept a trajectory of this shape produced by this reference, and proves nothing about what an H100 would report.
- `gap-oer01-harness-trainer-backend-unbuilt`: tests/harness.py owns the readout semantics and the telemetry assembly, and its `train_and_evaluate` accelerator seam is NOT implemented. With no backend the harness writes no telemetry and exits non-zero, tests/runner.py fails closed with launched false, and the EFFECT checker attributes the zero.
- `gap-oer01-image-unbuilt-unpushed`: environment/Dockerfile and tests/Dockerfile pin the batch digest and were NOT built by this lane, so the claim that either environment assembles on a clean host is unexecuted.
- `gap-oer01-trajectory-judge-absent`: tests/rubrics.jsonl carries eight trajectory rubrics that need an LLM trajectory judge. This bundle carries no judge and this lane ran none.
