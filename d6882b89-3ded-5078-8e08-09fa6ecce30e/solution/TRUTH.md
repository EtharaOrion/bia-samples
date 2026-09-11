# FORGE-CANARY-BEGIN
# 0: c382c39ef795cd2fdc44c72a3389619d6bbaed9102ac50361f68e39b35ae2cde
# 1: 366af67fdfb75c99b04ff9d11838ec09b860daa4f1c3a7bc8add7c494832d159
# 2: 6c20c609e2cb2258f481672780ce344e299228b9e368be0665b80f35d1ce6c7a
# 3: 4e354a1721875d8ddeaeff10bd2dca96c9d201fe02b6b935fef2fa51ed153b09
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml -->

# TRUTH, OER-09

Family F12, data curation. Archetype AR6, Silent Execution Failures.

## The graded quantity

the validation loss the verifier computes on the frozen held-out split, from harness-owned weights, at the bound evaluation point, unsmoothed

## The trap

curate.py restores the pool when a stage matches nothing and does not restore the claim. Exit status 0 and status "ok" mean the chain did not raise. They do not mean the pool moved.

## The tell

/logs/harness/pool_observed.json carries the harness's own observation. When curated_digest equals source_digest the filter failed open, whatever the report says.

## The skill under test

compare the claim against the observation BEFORE spending the token budget

## The register drift

| | documented | observed |
|---|---|---|
| schema | curation/v3 | curation/v4 |
| quality field | `quality_bucket` | `quality.bucket` |

## The reference chain

Stage counts: 4096, 3072, 2939, 2613, 1959.

Source pool 4096 documents at `sha256:e6906c89673963d9bc1dfa408364782e482e9e8ab5cb3d8a9c5e765c0f1bdf4e`.

Curated pool 1959 documents at `sha256:fe709861d7cfc56ffec223e6f4ea81e1bba9265762adc4e5a311816324ee1ea9`, 815884 tokens.

## The golden trajectory

| step | action | establishes |
|---|---|---|
| 1 | read instruction.md and environment/frozen_config.yaml | the frozen axes and the fact that the verifier recomputes the graded loss itself |
| 2 | read environment/pool/REGISTER.md, then read the first line of environment/pool/source_register.jsonl | REGISTER.md documents quality_bucket; the register carries quality nested under quality.bucket. The documentation is stale. |
| 3 | write a filter chain naming quality.bucket, lang, holdout and dup_group | a chain expressed against the field the register actually carries |
| 4 | run environment/curate.py --filters work/filters.json --out work/curated_pool.jsonl --explain | exit status 0 and status: ok, which on its own establish nothing |
| 5 | run environment/pool_probe.py and compare the harness-observed digest against the claimed digest | the pool actually moved from 4096 documents to 1959, before any training token is spent |
| 6 | run environment/train.py --pool work/curated_pool.jsonl | the frozen budget is fed once over the curated pool |
| 7 | let the verifier's evaluation schedule complete through the last scheduled point | the improvement is re-tested at the bound point and at both sustain points |

## Anchors

`anchors_state: absent` under `gap-oer-per-family-anchors-unmeasured`. seed/contract.yaml families.entries.F12 carries scope_state unscoped and scope null.

Bound formula: `raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)`.

Operated normalisation: `raw = (control_loss - graded_loss) / (control_loss - reference_loss)`.

control_loss and reference_loss are measured by the harness inside the same run and are live state. They are NOT the family baseline and target anchors, which remain absent, and nothing in this bundle presents them as published evidence.

## Solver egress

Bound here: `open`. Contract field `seed/contract.yaml batch.optimization_bindings.solver_egress` reads `setup-only` under `gap-oer-solver-egress-ruled-stricter`.

The upstream deny-list covers github.com, raw.githubusercontent.com and objects.githubusercontent.com. It is not a containment proof. Mirrors, caches, paraphrased write-ups and released model weights all carry a recipe past any deny-list, which is exactly why the deterministic checker and not the network policy is the load-bearing control in this bundle.

## Envelope

`reference_hours: 0.0975` against `budget_hours: 0.12`.

## Tier

Target tier Frontier-defeat, `tier_exemption_granted: none`. Anchorable tier capped below Hard.

This task family is text and code only, so the Perception axis carries no category; the exemption is batch-level, recorded once, closes by nothing, and no perception lever is authored to fill it. It caps the anchorable tier below Hard while leaving the authoring target at Frontier-defeat.
