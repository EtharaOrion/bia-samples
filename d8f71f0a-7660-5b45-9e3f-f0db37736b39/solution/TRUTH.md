# FORGE-CANARY-BEGIN
# 0: 054476e28184626a55de78c9d3f5786dacee180dd32c0f37c57fa2bdceeacd36
# 1: 1c565db139c499d460cb8ca281a43fd76f818a2102bcb7e03f999723f2972eb8
# 2: 6fda4e4be4417cf0463fa2acf25b13829156bbb069c9c2d762d7a6aaa291054d
# 3: 9db4053b61ba8f094e51d8158ac9f363ff49f2a2a5ee446c51fac8c507cb2fca
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml -->

# TRUTH, slot OER-10

## What is graded

The validation loss of the trained model, lower is better, computed BY THE VERIFIER in its own process on the verifier-owned held-out split `val-split-b` from harness-owned parameter snapshots at the bound evaluation point step 2800. It is never a number the submission reported, never a number in the submission's stdout, never a field the submission wrote, never a smoothed readout, and never a number the in-container harness printed.

## The substrate the metric resolves against

| quantity | value |
|---|---|
| declaration | `environment/nanogpt_substrate.json`, nanogpt-substrate-v1 |
| vocab_size | 50304 |
| num_layers | 12 |
| model_dim | 768 |
| head_dim | 128 |
| num_heads | 6 |
| seq_len | 1024 |
| tokens per step | 524288 |
| forward and backward passes per step | 1 and 1 |
| training corpus | `data/fineweb10B/fineweb_train_*.bin` |
| held-out corpus | `data/fineweb10B/fineweb_val_*.bin`, verifier-owned |

`environment/frozen_train.py` instantiates that decoder and trains it over the FineWeb10B training shards the curation recipe retained, one forward and one backward pass per step at the frozen batch. Delete either pass and there is no parameter snapshot to evaluate, so the graded quantity becomes undefined rather than merely different. `tests/evaluate.py` is what computes the graded scalar, from those snapshots, on a split whose member ids and token bytes appear nowhere in `environment/`.

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
| reference sha256 | `5ae6f2e4f391e682c89d05ea8f5dd2e41d9aa6567a4e3c4d63525a4b5fe7c0d6` |
| training spans fed | 240 of 240 |
| tokens fed | 1677721600 |
| frozen token budget | 1677721600 |
| control arm loss | 3.412 |
| reference floor loss | 3.05 |
| graded loss | 3.0412 |
| reward | 1.0 |

The reward is not binary. The same gate chain over the same fixture with a graded loss of 3.25 lands at 0.447513812155, strictly between 0 and 1.

## The loss levels are load-bearing, not decorative

`control_arm_displaced` is the checker whose outcome is a function of the loss LEVELS the run reached. The graded loss of 3.0412 sits below the control arm of 3.412 the harness measured inside the same run, so the accepting half passes. The planted fixture `nc-control-arm-not-displaced` carries a graded loss of 3.46, above that control arm, and the checker rejects it with `control-arm-not-displaced`. Its ladder value would be 0.0 either way, so the gate moves no run's number, only the attribution of that zero.

No loss level appears on the agent-visible surface. `instruction.md` names the evaluation points, the frozen axes and the scoring formula and never a loss, so the series is established only by training and reading the harness's own evaluation back. A solver that never interacted with the environment carries no sample series and no measured ladder ends, and fails this checker with a named reason rather than passing vacuously.

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
| `control_arm_displaced` | EFFECT | `control-arm-not-displaced` |

## Not claimed

No H100 training run was performed on this host. The feasibility bundle drives the real checkers over frozen synthetic telemetry, declared as `gap-oer-10-graded-training-run-not-executed-on-this-host`. No difficulty claim is made anywhere in this bundle.
