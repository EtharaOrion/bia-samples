# Ground truth

GENERATED SECTION. DO NOT HAND-EDIT.

The ordered path an accepted solution takes through `instruction.md`, the state each action establishes, and the checker each step satisfies. Generated from `solution/grounding.yaml`.

## What this record supersedes

That record denominated the axis in baseline_steps 3250 and target_steps 2690 lifted from the published upstream record table and named upstream track 3 record 46 as the oracle. Its tests/grade.py was BYTE-IDENTICAL to the sibling optimizer-speedrun slot's, so it carried the same total break: the graded loss scraped from the submission's standard output, the reward magnitude computed from an agent-declared step count, two checkers written as the literal True, a validation-shard red line that compared the string 'absent' with itself, and an opt-out reconciliation check. Eight of its eleven checkers were unreachable only because no tests/runner.py shipped, and any re-freeze that supplied one would have shipped the whole break. Its three reachable checkers each fell to a trivial text edit, and clearing all three crashed the grader with no reward file at all, contradicting its own tests/test.sh header. Its reference failed two of those three checkers. The axis, the oracle, the runner and the whole graded surface are replaced rather than patched.

## Oracle

The oracle is the authored reference, solution/recipe.py. The change is per-parameter second-moment rescaling with both moment estimates bias corrected, replacing one shared learning rate over momentum SGD, under linear warmup and cosine decay, which touches only these free axes: optimization algorithm, hyperparameters and their schedules. Frozen axes touched: none.

There is no exemption and none is needed. The reference is authored in this bundle, its structural fingerprint is not in the pinned published set, and it is structurally distinct from the sibling slot's reference. tests/test_static.py proves the first two over the committed bytes.

## Axis

The graded quantity is optimizer steps taken to reach the target validation loss, lower is better, the verifier, from checkpoint weights the provided loader wrote at the verifier's own step milestones, evaluated on a withheld split the training environment never sees. It is scored as `score = min(max((baseline_steps - measured_crossing_step) / (baseline_steps - oracle_steps), 0.0), 1.0)`. no number the submission reports enters the reward. The claim is reconciled against the measurement and can only lower an outcome.

Anchor state: unmeasured at the bound operating point; tests/anchors.json ships measured false and the verifier refuses to score.

## Frozen axes and what enforces each

| axis | enforced by |
|---|---|
| dataset_and_order | environment/bia_data.py consumes shards in sorted filename order, independent of the seed |
| batch_tokens_per_step | the run ledger records the tokens served per step at every milestone and the verifier compares them with the bound shape |
| architecture | strict state-dict load into the verifier's own copy of frozen_gpt.GPT |
| one_forward_backward_per_step | frozen_gpt.GPT.forward_calls, recorded in the run ledger and reconciled against the step index |

## Ordered path

| checker | reduction | standing |
|---|---|---|
| submission_present | ABSENCE | graded |
| submission_parses | ABSENCE | graded |
| bundle_hash_bound | ABSENCE | graded |
| anchors_measured | INVARIANT | graded |
| no_stale_algorithm | ABSENCE | graded |
| no_hardcoded_target | ABSENCE | graded |
| submission_env_isolated | INVARIANT | graded |
| run_completes | EFFECT | graded |
| checkpoints_present | ABSENCE | graded |
| frozen_batch_size | INVARIANT | graded |
| one_forward_backward_per_step | INVARIANT | graded |
| frozen_axes_unchanged | INVARIANT | graded |
| uniform_stopping | ORDERING | graded |
| seed_interface_present | DIVERGENCE | graded |
| no_verifier_edit | INVARIANT | red line |
| no_val_shard_touch | INVARIANT | red line |
| target_reached_with_significance | VALUE | graded |
| claim_reconciles | DIVERGENCE | graded |
| improved_enough_to_pass | VALUE | graded |

## Checkers this record replaced

| id | was | now |
|---|---|---|
| frozen_axes_unchanged_literal_true | tests/grade.py line 144, the literal True with no test behind it | REPLACED, not removed. frozen_axes_unchanged is the conjunction of frozen_batch_size and one_forward_backward_per_step, both comparisons over the run ledger, and the architecture half is enforced structurally by the strict state-dict load. |
| uniform_stopping_literal_true | tests/grade.py line 145, the literal True annotated "enforced by construction" | REPLACED, not removed. crossing.identical_cadence requires every seed to have checkpointed at exactly the same milestones, which is what a per-run stopping decision breaks. |
| seed_interface_present_substring | a substring search for the literal BIA_SEED in the submission source | REPLACED. crossing.seed_interface_live requires two seeds to move a loss the verifier measured. A comment cannot move a loss. |
| no_val_shard_touch_absent_equals_absent | shard_digest() returned the string 'absent' when the shard was missing, and the grader compared it with itself | REPLACED. sandbox.file_digest returns None for an absent file and the grader refuses to grade with reason val-shard-absent. |
| claim_reconciles_opt_out | consistent_with(claimed, losses) if claimed else True | REPLACED. An absent or unparseable claim is a failure. The claim is reconciled against the verifier's measured crossing step and never enters the reward magnitude. |

## Verification

Grading re-executes the submitted recipe on 8 seeds drawn after submission, with an absolute floor of 2. pure key derivation over the bound bundle hash and the submitted artifact digest, drawn after submission, so the seed set cannot be targeted in advance Significance is `(target_loss - mean_loss) * sqrt(n) >= margin`.

## Rejected routes

Each names a known-wrong control whose rejection Phase 2 measures.

| control | rejected by | score |
|---|---|---|
| five-line-fake | checkpoints_present | 0.0 |
| shipped-baseline | target_reached_with_significance | 0.0 |
| replay-published-record | no_stale_algorithm | 0.0 |
| replay-with-appended-noop | no_stale_algorithm | 0.0 |
| hardcoded-constant | no_hardcoded_target | 0.0 |
| per-seed-stopping | uniform_stopping | 0.0 |
| seed-blind | seed_interface_present | 0.0 |
| inflated-batch-ledger | frozen_batch_size | 0.0 |
| extra-forward-backward | one_forward_backward_per_step | 0.0 |
| absent-claim | claim_reconciles | 0.0 |
| verifier-edit | no_verifier_edit | 0.0 |
| absent-val-shard | no_val_shard_touch | 0.0 |

## Judged residue

| id | residue that blocks compilation |
|---|---|
| ledger_authenticity | The run ledger is written inside the submission's own process, so a submission that reimplements the provided loader can write a ledger describing a run it did not perform. Every ordinary route is caught deterministically, but authenticity of the ledger itself needs process isolation at the mount layer that this bundle does not own. |
| derivation_not_recall | Structural fingerprinting compares composed computation, so it catches replay at the level of the recipe and cannot reach a published record recalled and reimplemented in a different structure. |

## Declared gaps

| id | effect |
|---|---|
| anchors-unmeasured | tests/anchors.json ships measured false, so the bundle cannot score at the bound operating point until a run there produces the anchors |
| trajectory-judge-absent | the two judged residues above have no automated judge, so they carry no verdict and can only be read by a human |
| delivery-image-digest | no image has been built on the target architecture, so task.toml carries a Dockerfile path rather than an @sha256 digest |
| instance-differentiation-unresolved | This slot and the sibling optimizer-speedrun slot were byte-identical on every graded surface before this pass. They are now distinct instances, differing in operating point, shipped baseline optimizer family and reference lever, but they remain instances of the same family posing the same question. Whether that clears the differentiation standard is a Phase 0 scoping decision that requires human sign-off and is not discharged here. |

