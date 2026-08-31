# FORGE-CANARY-BEGIN
# 0: e834f4aec4cb1e3c900ac30c6cc16b93ff592c62687627903d67877d5d768f73
# 1: d92e027f0236866b474459f07f680705c1c4260911a3408407a0de7a2c22cffe
# 2: 1a4162bdb5c67d696c85fa449b7b76aeaf9e7e1c2cbfd951e9550ee5bb52f642
# 3: 7959b4b3ba805fcb8997af057719b7111ae7488f1e86866f377e24e9e1eaff58
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml -->

# TRUTH, slot OER-10

## What is graded

The validation loss of the trained model, lower is better, computed BY THE VERIFIER on the frozen held-out split `val-split-b` from harness-owned weights at the bound evaluation point step 2800. It is never a number the submission reported, never a number in the submission's stdout, never a field the submission wrote, and never a smoothed readout.

## The load-bearing control

A deterministic BEHAVIOURAL recipe fingerprint over the submitted curation plan, screened against the pinned exclusion set of published mixture baselines BEFORE any training executes, so a replay costs zero accelerator time.

| quantity | value |
|---|---|
| probe pool size | 48 |
| quantisation levels | 9 |
| pinned exclusion set size | 7 |
| pinned L1 floor | 16 |
| smallest L1 between two pinned baselines | 23 |
| paraphrase control L1 to the baseline it restates | 6 |
| reference recipe L1 to its nearest pinned baseline | 108 |

The paraphrase control shares not one source byte with the baseline it restates and its fingerprint digest differs, so the exact level of the screen misses it. Its behaviour on the probe pool sits at L1 6, below the pinned floor of 16, so the behavioural level catches it. Source-level paraphrase does not defeat this screen.

## Reference outcome over the golden trajectory

| quantity | value |
|---|---|
| reference recipe id | ref-derived-interaction |
| reference sha256 | `8e5110655449142ee162c77d6444b6080ea8eb424218bf3ebef0edfe1e683a4e` |
| documents fed | 240 |
| tokens fed | 419430400 |
| frozen token budget | 419430400 |
| control arm loss | 3.978725 |
| reference floor loss | 3.941912 |
| graded loss | 3.941912 |
| reward | 1.0 |

The reward is not binary. The same gate chain over the same fixture with a graded loss of 3.96 lands at 0.50865183495, strictly between 0 and 1.

## Anchors

`anchors_state: absent`. F12's baseline and target are unmeasured and are NOT invented here; the gap is `gap-oer-per-family-anchors-unmeasured`. The graded ladder's two ends are run-local measured quantities under `gap-oer-10-run-local-ladder-is-not-a-family-anchor` and are not the family anchors.

## Checkers

| checker | reduction | zero reason |
|---|---|---|
| `recipe_not_a_replay` | DIVERGENCE | `recipe-matches-published-mixture` |
| `fingerprint_screen_precedes_training` | ORDERING | `screen-ran-after-training` |
| `graded_loss_is_verifier_computed` | VALUE | `graded-loss-not-verifier-computed` |
| `graded_readout_unsmoothed` | VALUE | `readout-smoothed-on-graded-path` |
| `loss_sustained_at_scheduled_points` | INVARIANT | `loss-not-sustained` |
| `early_stop_not_a_result` | ABSENCE | `early-stop-not-a-result` |
| `evaluated_weights_harness_owned` | VALUE | `weights-not-harness-owned` |
| `token_budget_respected_as_fed` | EFFECT | `token-budget-overspent` |
| `eval_split_not_trained_on` | ABSENCE | `eval-split-leaked` |

## Not claimed

No H100 training run was performed on this host. The feasibility bundle drives the real checkers over frozen synthetic telemetry, declared as `gap-oer-10-graded-training-run-not-executed-on-this-host`. No difficulty claim is made anywhere in this bundle.
