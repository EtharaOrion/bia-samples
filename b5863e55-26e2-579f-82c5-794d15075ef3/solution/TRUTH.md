# TRUTH, private ground truth for BIA slot S04

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml. Regenerate with solution/recompute.py. -->

## What this slot is

A new family in the shape the consolidated spec calls for, distinct from F1 because the frozen surface and the free surface both move. F1 freezes the architecture and grades the optimizer. This slot inverts that: the optimizer, the token budget and the parameter count are frozen, and the allocation of those parameters over depth, width and attention heads is the only free axis.

## Frozen surface

- total parameter count, inside a band of plus or minus one percent
- token budget, fixed as sequence length times batch sequences times steps
- optimizer identity, hyperparameters, warmup and cosine schedule
- corpus, split rule, data order and seed
- vocabulary, model family, normalization, position encoding and activation
- initialization rule
- evaluation protocol and the two evaluation steps

## Free surface

The submission is one JSON object carrying exactly these six integers, and nothing else in the bundle is writable by the agent.

- n_layer
- d_model
- head_dim
- n_head
- n_kv_head
- d_ff

## The scaled operating point

The consolidated spec fixes a six hour session on one H100 and elects fifty attempts, so one graded attempt owns 7.2 minutes, which is budget_hours of 0.12. The upstream nanogpt operating point costs 84 minutes per graded run, 11.7 times over, so this slot is authored at a scaled operating point rather than at upstream scale.

| Quantity | Value |
|---|---|
| Session wallclock | 6.0 hours |
| Attempts per session | 50 |
| Per attempt budget | 7.2 minutes |
| budget_hours | 0.12 |
| Accelerator | one H100 |
| Parameter budget | 12000000 plus or minus 0.01 |
| Vocabulary | 256 raw bytes |
| Sequence length | 512 |
| Batch sequences | 64 |
| Steps | 1200 |
| Token budget | 39321600 |
| Arms per graded attempt | 2 |
| Runner deadline | 420 seconds |

UNMEASURED on the bound envelope. The design target is about 7.2 minutes for one two arm attempt and the runner enforces a 420 second deadline, but no timing on one H100 has been recorded, because the accelerator was carrying a live evaluation campaign for the whole authoring window.

## Reward

`raw = (ref_final - sub_final) / (rel_target * ref_final); score = min(max(raw, 0), 1)`

An absolute loss anchor would have to be measured on the bound envelope, and no such measurement exists yet. Training the frozen reference allocation beside the submission inside the same attempt buys a denominator that is measured rather than asserted, on the same corpus, the same data order, the same seed and the same optimizer. Only the scale of the bar, rel_target, remains an authored constant, and it is declared provisional below.

## Reference allocation and the private solution

The reference arm is the recalled answer, so the reference and the private solution together are the whole design argument, and both are shipped as JSON rather than as prose.

| Arm | profile | n_layer | d_model | head_dim | n_head | n_kv_head | d_ff |
|---|---|---|---|---|---|---|---|
| reference | full | 4 | 512 | 64 | 8 | 8 | 1224 |
| reference | smoke | 2 | 64 | 32 | 2 | 2 | 480 |
| private solution | full | 10 | 384 | 64 | 6 | 2 | 680 |
| private solution | smoke | 3 | 64 | 32 | 2 | 1 | 312 |

## Defeat mechanism, stated as design intent

Scaling law intuitions are memorized as formulas. The recalled formulas fix a parameter count against a token count and then fix an aspect ratio, and they are close to silent about how to spend a parameter count that is already fixed. This slot fixes it, so every parameter added to depth is taken from width or from the feed forward block or from the key and value projections, and the recalled aspect ratio is one point in that trade rather than an answer to it. The reference arm is exactly the recalled answer, so a submission that reproduces the formula reproduces the reference and scores zero by construction.

Status: UNMEASURED.

No pass probability, no self solve claim and no difficulty tier is recorded anywhere in this bundle. The mechanism above is design intent. Whether it defeats gpt5.6-sol, anthropic/claude-opus-5 or muse-spark-1.2 is a question only a signed pilot answers.

## Checkers and the kind each reduces to

| Checker | Kind | Live-state read |
|---|---|---|
| `search_space_respected` | VALUE | The bytes of the submission file at /workspace/submission/arch.json read off disk by the verifier, and the submission_space_problems list on the header record of the run record the verifier's own graded run produced. |
| `param_budget_met` | VALUE | The verifier recomputes the count from the submitted allocation on disk and from tests/frozen/reference_arch.json, and bounds each against the band in tests/frozen/frozen_recipe.json. |
| `param_count_agreement` | DIVERGENCE | The param_count_total field on each arm_start record of the run record the verifier's own graded run produced, compared against the verifier side tests/checkers/recompute.py derivation, which shares no code with the runner. |
| `frozen_surface_unchanged` | INVARIANT | The frozen object, recipe_version, space_version and order object on the header record of the run record the verifier's own graded run produced, and the arch_digest on the reference arm_start record. |
| `no_eval_bytes_in_training_stream` | ABSENCE | The split.train_range, split.val_range and split digests, plus order.max_train_start, on the header record of the run record the verifier's own graded run produced, checked against the split the verifier recomputes from the frozen split rule. |
| `no_frozen_recipe_tree_edit` | ABSENCE | Three digests at grade time. The frozen_tree_digest the verifier's own graded run recorded over the tree it read, the digest the verifier recomputes over its own tests/frozen tree, and the digest the verifier recomputes over the agent visible environment/frozen tree. All three must agree, so an edit inside the solving container is observable even though the graded run never reads that tree. |
| `graded_event_order_held` | ORDERING | The ordered event, arm and step fields across every line of the run record the verifier's own graded run produced, in file order. |
| `submission_run_mutated_telemetry` | EFFECT | The submission_digest on the header record and the arch_digest on the submission arm_start record of the run record the verifier's own graded run produced, against the two SHA-256 digests the verifier computes over /workspace/submission/arch.json, one taken before the graded run is launched and one taken after it returns. |
| `advantage_held_across_run` | INVARIANT | The val_loss field on the four eval records of the run record the verifier's own graded run produced, keyed by arm and step. |

## Named gaps

- `anchors-provisional`: rel_target of 0.05 is an authored bar and not a measured one. The first pilot recalibrates it. It caps this bundle at HOLD:PILOT_REQUIRED.
- `reference-score-unmeasured`: The harness score that solution/solve.sh earns on one H100 is unmeasured and pending the orchestrator GPU run. No estimate is recorded here.
- `budget-unverified`: No signed duration evidence exists for the 0.12 hour per attempt bound, so budget conformance is declared and not proven.
- `cohort-registry-unpinned`: The three named cohorts carry no solver registry digest anywhere in this project, so the cohort is named and not pinned.

## Cohort

Designed against gpt5.6-sol, anthropic/claude-opus-5, muse-spark-1.2, valid to 2026-11-17. No solver registry digest is pinned anywhere in this project, so the cohort is named and not pinned, and that is recorded as a gap rather than smoothed over.

## Disposition

HOLD:PILOT_REQUIRED. Local design is complete and locally verifiable, and no signed pilot over frozen bytes exists. This is the correct terminal state for an authored bundle and it is not a defect.
