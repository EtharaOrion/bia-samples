# FORGE-CANARY-BEGIN
# 0: 256810655f8a3f40e14085c2ea0969a8d46c77d075fe2c3854fad1eac3d9392b
# 1: 3e86608918ee6bd0552929a846985e8d495d9d3a95d531fa2ac82508114f0070
# 2: d4a37a00afec21a83edecbba884b48a4e7142e083c2e78fcdd7e0acc00bacfd1
# 3: 5066e304f3920285d3770e515aed957071028d4a57eefcc1a15eec8b1011dce7
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml -->

# TRUTH.md, slot OER-17

## What is graded

The held-out benchmark score of the model the harness trained on the submitted corpus, measured by the verifier in its own process from the parameters the harness held at the bound evaluation point, sustained across the last three verifier-scheduled points.

## What is not graded

Anything the submission printed, wrote, or claimed. The reported number is parsed only so the divergence against the measurement can be recorded.

## Why the screen is the load-bearing control

The benchmark is the reward, so reproducing it is the highest-value attack. Isolation removes the read, but recall and reconstruction do not need a read, so the screen operates on emitted samples and catches the effect whatever its provenance.

## The reference, as measured on this host

| quantity | observed |
|---|---|
| agent_metric | 1.0 |
| corpus_samples | 144 |
| eval_schedule | [160, 240, 320] |
| graded_step | 320 |
| peak_screen_similarity_reference_vs_heldout | 0.3333 |
| reported_score_printed_by_reference | 0.875 |
| reported_vs_measured_divergence | 0.125 |
| steps_fed | 320 |
| tokens_fed | 276480 |
| usable_samples | 144 |

Every number here was measured on this host by running the real grading path. None is projected.

## Anchors

anchors_state: absent, gap gap-oer-per-family-anchors-unmeasured.
No baseline_metric or target_metric number is authored anywhere in this bundle. The reward schema is fully bound and the pair is declared absent.

## Known limits

- The anchor pair is unmeasured, so a live graded run resolves to 0.0 with the reason anchors-unmeasured until it is measured.
- A generator trained on a distinct-but-adjacent public corpus is not contamination of this benchmark and is not screened out.
- Whether an agent recalled the benchmark from pretraining is not measurable from frozen bytes; only what it emitted is, and that is screened.
