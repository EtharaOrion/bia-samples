# FORGE-CANARY-BEGIN
# 0: d2a39c8c5f8c23717383c06b44a0a703a3fc231a2a19c647c53d8bc9c7c77781
# 1: a435ddfdc7f9a826bb385be5c6ba6829c3e53ce7b243845b75572d66f6c52930
# 2: 93249ca36ebc0c53779e6185e516b5f462534ea789136bf2f3fd6b3310c8d9e1
# 3: a0c9340c884e0f115b5924ee019db7ef13c2b5bb935e131dd09fb5d6f33ad1d7
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. -->
<!-- Source: solution/grounding.yaml. Regenerate with solution/recompute.py. -->

# TRUTH.md, OER-04

**Significance-unproven submissions at the bound seed ceiling**  
Family F1, nanogpt optimizer speedrun. Primary archetype AR7, Ambiguous Intermediate States.

Objective, verbatim from the approved contract: *Significance-unproven submissions at the bound seed ceiling.*

## The graded quantity

the largest across re-executed seeds of the first verifier-scheduled evaluation step whose verifier-computed unsmoothed held-out loss is below 3.28 and remains below 3.28 at the next two verifier-scheduled evaluation points.

## Anchors

| | |
|---|---|
| baseline_metric | 3250 |
| baseline identity | upstream record 36, tuned Muon with aux AdamW, loss 3.2787 at n=10, stat-verified, 2026-06-11 |
| target_metric | 2690 |
| target identity | upstream record 46, current world record, loss 3.2783 at n=8, stat-verified, 2026-06-19 |
| authority | requirements/bia-environment-spec.md lines 297 to 321 |
| anchors_state | measured |
| pass_threshold | 0.65 |
| separation_margin | 0.05 |

## The three outcomes

| outcome | condition | classification |
|---|---|---|
| `significance-established` | the target was reached on the re-executed seeds and sep >= 0.05 | pass |
| `significance-unestablished-at-ceiling` | the run COMPLETED, the target was reached on the re-executed seeds, the seed budget was the binding limit, and the measured separation did not clear the bound margin of 0.05 | unproven, distinct from failed |
| failed reasons | a gate rejected the run or it never reached the target | failed, not unproven |

The middle row is the whole slot. A submission whose measured improvement sits
inside the seed-to-seed noise band at the bound seed ceiling has established
nothing, and that is a third outcome rather than a shading of either neighbour.
The reward degrades continuously through that band and reaches exactly zero only
when the separation is at or below zero, so the band is visible in the number as
well as in the reason code.

## The reference solution

`solution/reference.py`, sha256 `aeb24bba3f74b13036d2d12a98a25dd1a8fd68950e04494ab26fefef7cc0c4e6`.

A tuned Muon step on the 2D parameters with an auxiliary AdamW on the embedding, the head and every norm parameter, a short linear warmup, a cosine floor that does not decay to zero, and an orthogonal initialization on the attention projections. Derived, not recalled: the contamination screen treats the published record lineage as an evaluation set, so a replayed record recipe scores nothing.

Claimed step 2690, twenty re-executed seeds, every one at or below the target, so the spread is
zero, the separation is 1.0 and the reward is exactly 1.0.

## Golden trajectory

| attempt | change | graded_step | reward |
|---|---|---|---|
| 1 | baseline AdamW with the shipped cosine schedule, unchanged, to establish the starting point | 3250 | 0.0 |
| 2 | Muon on the 2D parameters, auxiliary AdamW on embedding, head and norms | 3040 | 0.375 |
| 3 | linear warmup shortened to 128 steps, cosine floor raised off zero | 2915 | 0.598214 |
| 4 | orthogonal initialization on the attention projections, Muon momentum retuned | 2790 | 0.821429 |
| 5 | Muon Newton-Schulz iteration count raised, auxiliary AdamW betas retuned | 2690 | 1.0 |

## Feasibility

| | |
|---|---|
| reference_hours | 1.4 |
| budget_hours | 0.12 |
| over budget | True |
| disposition | `BLOCK:INFEASIBLE_OR_UNVERIFIED` |
| held by | `gap-oer-scaled-operating-point-unmeasured` |

reference_hours of 1.4 exceeds budget_hours of 0.12, so this slot is BLOCK:INFEASIBLE_OR_UNVERIFIED. It is recorded rather than rounded away. The contract already records this standing at batch.authorable_slots_note, which states that the four F1 slots are held by an unmeasured scaled operating point that Phase 2 item 7a would resolve to BLOCK:INFEASIBLE_OR_UNVERIFIED rather than to a hold. Nothing in this bundle asserts otherwise.

Transcribed from requirements/bia-environment-spec.md line 133, which records F1 at the upstream operating point costing 84 minutes per graded run, being 11.7 times the per-attempt budget. It is NOT measured on this run: no accelerator and no FineWeb shard is present on this host, so no graded run was performed and none is claimed.

## Declared gaps

- **`gap-oer-04-reward-path-carrier-split`** seed/forge/verifier.py binds the runtime reward path as /logs/verifier/reward.txt carrying one float, while the wave-1 lane spec describes a shaped JSON document at /logs/verifier/reward.json. Both are written by tests/test.sh, the float last, and tests/checkers.yaml declares the .txt path as reward_path because that is the path the instrument binds. The divergence is recorded rather than resolved by preference.
  Closes by: a single bound reward carrier named identically by the instrument and the lane spec.
- **`gap-oer-04-separation-margin-units-differ`** The batch binds separation_margin 0.05 as a pilot-level pass-rate quantity and the F1 ladder binds significance_margin 0.004 as a per-submission validation-loss quantity. This slot's graded checker compares against 0.05 in anchor-normalised score units, which is a third unit system, chosen so the graded separation is commensurable with the reward it multiplies. Both bound values are carried by name and neither is recomputed, but no source states their relation.
  Closes by: a recorded relation between the two margins, or a per-family binding of the score-unit margin.
- **`gap-oer-04-verifier-reexecution-unmeasured`** tests/runner.py and tests/held_out_eval.py are complete and were not executed on this host, because no accelerator and no FineWeb validation shard is present. Every checker adequacy result recorded for this slot was produced against planted telemetry fixtures driving the live checkers, never against a re-executed training run. No such run is claimed.
  Closes by: one execution of tests/test.sh on the bound envelope with the shard mounted.
- **`gap-oer-04-overlay-scratch-precreated`** seed/forge/launcher.py::plan requires the upper, work and merged overlay layers to be existing directories and assembles only the lower layer itself, while seed/staging/oer1/verify_slots.py does not create them. Those three scratch directories under seed/results/overlay/OER-04/ were created by this lane as run scaffolding so the real instrument could execute. They are ephemeral run output, not bundle bytes, and no byte was written into dataset/.
  Closes by: verify_slots.py creating the three writable layers, as seed/tests/test_launcher.py's fixture does.
