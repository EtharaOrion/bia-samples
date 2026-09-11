# FORGE-CANARY-BEGIN
# 0: e4dca722ca50ec245a19a5d03df9fffe47bd027e7dd0f9fd527bb093f734bbba
# 1: e5863418791905c010d07e719f26ef572fc331e99106142a58036375ed276544
# 2: d4e89c9b2413e4d95a63f4443fdfeb8465d25fdf743d45f236ab3a4b837f2c54
# 3: bbe552259793aba3945d16b08956d28211dee7c7f761528a9bae0e9ec6bb5b0e
# FORGE-CANARY-END
# TRUTH.md

GENERATED SECTION. DO NOT HAND-EDIT.

Source: `solution/grounding.yaml`. Regenerate with `solution/recompute.py`.

## What is graded

The normalised gap to the exact optimum on the graded frozen set-cover instance, direction lower is better. The verifier computes the optimum itself with an exact bitmask dynamic program over the frozen instance, so the comparator is a classical exact solver and never a heuristic estimate. The graded instance is the one attested by the last atom in the realised attestation order, so the instance itself is not identifiable without recovering that order.

## The two discovery values

Both are established in built environment state by `environment/mint_store.py` at image build time and neither appears in `instruction.md` or in any other agent-visible bundle byte.

| discovery value | realised value |
|---|---|
| terminal_atom_digest | `a3d2a3dfc52f05e3342debbbb3c066729ba00c26fa81e9a94391942a1a875fee` |
| realised_attestation_order | `atom-02 atom-06 atom-03 atom-19 atom-04 atom-16 atom-05 atom-20 atom-17 atom-09 atom-18 atom-01 atom-08 atom-07 atom-12 atom-10 atom-24 atom-14 atom-22 atom-11 atom-21 atom-15 atom-23 atom-13` |

The realised order is a topological order of the derivation graph and it is neither of the two orders a reader reaches for instead. The lexicographic identifier order ends at `atom-24` and the ready-set-sorted topological order ends at `atom-24`, while the realised order ends at `atom-13`.

## The terminal atom

| field | value |
|---|---|
| atom_id | `atom-13` |
| instance_id | `inst-13` |
| heuristic | `weight-greedy` |
| inputs | `atom-09, atom-11, atom-12` |
| attested cover | `1, 3, 5, 6, 7` |
| attested cover weight | 32 |
| exact optimum weight | 24 |
| exact optimum cover | `1, 3, 7` |

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F7 carries no measured family baseline or target, so none is invented. The two normalisation numbers below are measured on this slot's own frozen substrate; they are instance-local and they are NOT family anchors, under gap `gap-oer-30-instance-local-anchors-not-family-anchors`.

| quantity | value |
|---|---|
| instance_baseline_gap | 0.333333333 |
| instance_target_gap | 0.0 |
| reference normalised gap | 0.0 |
| reference reward | 1.0 |

raw = (instance_baseline_gap - agent_gap) / (instance_baseline_gap - instance_target_gap), then score = min(max(raw, 0.0), 1.0).

## The realised order in full

| position | atom | instance | heuristic | inputs |
|---|---|---|---|---|
| 0 | `atom-02` | `inst-02` | `first-fit` | `none` |
| 1 | `atom-06` | `inst-06` | `first-fit` | `none` |
| 2 | `atom-03` | `inst-03` | `weight-greedy` | `none` |
| 3 | `atom-19` | `inst-19` | `weight-greedy` | `atom-03` |
| 4 | `atom-04` | `inst-04` | `first-fit` | `none` |
| 5 | `atom-16` | `inst-16` | `first-fit` | `atom-02` |
| 6 | `atom-05` | `inst-05` | `weight-greedy` | `none` |
| 7 | `atom-20` | `inst-20` | `weight-greedy` | `atom-05, atom-06` |
| 8 | `atom-17` | `inst-17` | `weight-greedy` | `atom-03` |
| 9 | `atom-09` | `inst-09` | `weight-greedy` | `atom-03` |
| 10 | `atom-18` | `inst-18` | `first-fit` | `atom-02, atom-06, atom-09` |
| 11 | `atom-01` | `inst-01` | `weight-greedy` | `none` |
| 12 | `atom-08` | `inst-08` | `first-fit` | `atom-01, atom-04` |
| 13 | `atom-07` | `inst-07` | `first-fit` | `atom-03, atom-04` |
| 14 | `atom-12` | `inst-12` | `first-fit` | `atom-01, atom-02, atom-07` |
| 15 | `atom-10` | `inst-10` | `weight-greedy` | `atom-02, atom-08` |
| 16 | `atom-24` | `inst-24` | `first-fit` | `atom-12, atom-16` |
| 17 | `atom-14` | `inst-14` | `first-fit` | `atom-05, atom-10` |
| 18 | `atom-22` | `inst-22` | `weight-greedy` | `atom-05, atom-06, atom-10` |
| 19 | `atom-11` | `inst-11` | `weight-greedy` | `atom-09, atom-10` |
| 20 | `atom-21` | `inst-21` | `first-fit` | `atom-14` |
| 21 | `atom-15` | `inst-15` | `weight-greedy` | `atom-11` |
| 22 | `atom-23` | `inst-23` | `first-fit` | `atom-12, atom-20, atom-21` |
| 23 | `atom-13` | `inst-13` | `weight-greedy` | `atom-09, atom-11, atom-12` |

## Substrate limit

The substrate is a deterministic content-addressed store minted by arithmetic, declared under gap `gap-oer-30-substrate-is-a-minted-store-not-a-live-build-system`. That is the right choice because it makes grading reproducible, and it never passes as the output of a live production build system.
