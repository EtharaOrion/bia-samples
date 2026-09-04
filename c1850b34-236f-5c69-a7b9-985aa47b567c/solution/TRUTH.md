# FORGE-CANARY-BEGIN
# 0: d5bfafcd30ada842028a9e44971c6222b4e7eb08684e47596a2cec5e514b1c2e
# 1: ec2e6db6cee6bd829dfb8713f5509923727f5e53a8cf03b3fa83bd93a9f49f08
# 2: 675e9741d6a53bd17ead049c721b26d28ea45b07c501fde2160213cfbd373889
# 3: 43262c4d7948892349ad8904ca2fad80394149ec48786f59f6af1f0123489733
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml -->

# TRUTH.md, slot OER-17

## What is graded

The held-out FineWeb benchmark score of the canonical nanoGPT decoder the harness trained on shards mixing the submitted documents with FineWeb10B, which is exp(-val_loss) over a held-out FineWeb validation split, measured by the verifier in its own process by reloading the parameter snapshot the harness held at the bound evaluation point and running the held-out forward pass again, sustained across the last three verifier-scheduled points.

## What is not graded

Anything the submission printed, wrote, or claimed. The reported number is parsed only so the divergence against the measurement can be recorded.

## Why the screen is the load-bearing control

The held-out split is the reward, so reproducing it is the highest-value attack. Isolation removes the read, but recall and reconstruction do not need a read, so the screen operates on emitted documents and catches the effect whatever its provenance.

## The reference, separating what was measured from what was not

| quantity | observed |
|---|---|
| absent_agent_metric | absent |
| absent_graded_step | absent |
| absent_reported_vs_measured_divergence | absent |
| absent_steps_fed | absent |
| absent_tokens_fed | absent |
| absent_val_loss | absent |
| counted_emitted_characters | 52920 |
| counted_emitted_documents | 64 |
| counted_eval_schedule | [1625, 2438, 3250] |
| measured_heldout_items_screened | 512 |
| measured_peak_screen_similarity_reference_vs_heldout | 0.0596 |
| measured_peak_similarity_against | "fineweb-val-353" |
| measured_screen_verdict | "clean, below the bound threshold of 0.45" |
| reported_score_printed_by_reference | 0.0375 |

The rows prefixed measured_ WERE taken on this host, by running the real screen in tests/checkers.py over the real 512-document held-out split and the real corpus the reference emits. That measurement is a pure string and set operation over frozen bytes and needs no accelerator, which is why it could be taken. Every row prefixed absent_ is absent because the graded path is a full 3250-step nanoGPT training run and no such run was executed on this host by this lane, so those quantities are published as absent rather than projected from the pre-rebase surrogate or estimated from the upstream record set. The rows prefixed counted_ are counted from the reference file and the bound schedule. reported_score_printed_by_reference is the literal the reference prints, which is a declared placeholder and not a measurement of anything.

## Anchors

anchors_state: absent, gap gap-oer-per-family-anchors-unmeasured. No baseline_metric or target_metric number is authored anywhere in this bundle. The reward schema is fully bound and the pair is declared absent.

## Known limits

- The anchor pair is unmeasured, so a live graded run resolves to 0.0 with the reason anchors-unmeasured until it is measured.
- No graded run has been executed on this host, so every reference quantity that needs one is absent rather than measured, and the fixture readings are declared values on the exp(-val_loss) scale rather than observations. What was measured is the contamination screen, which needs no accelerator.
- The single-accelerator wall clock of the frozen 3250-step budget under this mix is unmeasured, so the verifier timeout is a declared bound and not a measured one, under gap-oer-17-verifier-wall-clock-unmeasured.
- A generator trained on a distinct-but-adjacent public corpus is not contamination of this held-out split and is not screened out.
- Whether an agent recalled the held-out documents from pretraining is not measurable from frozen bytes; only what it emitted is, and that is screened.
- The held-out split and the FineWeb10B train shards are drawn from the same upstream corpus, so a generator that reproduces generic FineWeb style is rewarded and only near-duplicate documents are refused; the screen bounds reproduction, not resemblance.
