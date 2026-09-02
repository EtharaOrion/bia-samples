# FORGE-CANARY-BEGIN
# 0: e7d41ce50a6293ea45a23e4a91c4abe4afb0d485df1684047fdc0bab9d52cd67
# 1: 1895c0e08470f50b55fdd134c902073a3b280e9804d595de1a3cb0c18cbc6665
# 2: 10cc9b0e2ac5975c2d6bf5abb8726c990f5d8f2cf62c5ae926fc337f47586612
# 3: 5e64604f518cd235fab8476524c8ac1dd3278bb91e9dae4d8bb410c6cac4b4f7
# FORGE-CANARY-END
# TRUTH.md

GENERATED SECTION. DO NOT HAND-EDIT.

Source: `solution/grounding.yaml`. Regenerate with `solution/recompute.py`.

## What is graded

Near-duplicate recall against the verifier's own adjudication over the frozen corpus, under a hard precision gate. Direction: higher is better. Precision is a HARD CONSTRAINT: a submission carrying one pair the verifier's adjudication does not support scores exactly 0.0 with the reason `adjudication-unsupported`, whatever its recall.

The verifier rebuilds the index itself, in an isolated subprocess, from the frozen corpus and the frozen index specification, and recomputes the threshold, the grouping and the whole pair set from that index. No number the submission reported reaches the reward.

## The archetype

Near-duplicate adjudication. The corpus carries a spread of redundancy rather than two separated humps, so nothing about the boundary falls out of looking at the text. The boundary is established by the built index and by nothing else, and the evidence that the boundary was actually applied is a single pair the index proposed and the adjudication refused.

## The two discovery values

Neither appears in `instruction.md`, in `task.toml`, in `environment/corpus.jsonl`, in `environment/index_spec.json` or in any other agent-visible byte. Both come into existence when `environment/build_index.py` runs over the frozen corpus, which happens on one line of `environment/Dockerfile`, and both are read back through the query handle.

| value | what it is | where it is established |
|---|---|---|
| adjudication threshold `702` | the similarity level at which the built index calibrates the near-duplicate boundary, on the grid of 1024 | `nd_index.calibrate` over the built index's own candidate similarity histogram |
| collision witness `r0325`, `r0440` | the refused collision maximal by band count, then similarity, then lexicographic order | the built index's band collisions, once the duplicate grouping has been carried out |

The calibration cuts the candidate similarity histogram at its maximum-separation split, which lands at 698, then snaps up to the smallest candidate level at or above it, which is 702. The witness is seen in 6 bands at similarity 664.

## The reference trajectory, as it actually ran

| quantity | value |
|---|---|
| records in the frozen corpus | 512 |
| unordered pairs examined | 130816 |
| colliding pairs the built index holds | 144 |
| adjudication threshold | 702 |
| thresholded edges over the whole universe | 86 |
| duplicate groups with two or more members | 65 |
| group size histogram | 2: 51, 3: 13, 4: 1 |
| near-duplicate pairs after closure | 96 |
| collision witness | r0325, r0440 |

Three separate things have to be right for the reference answer to be right, and each of them is a place a shallow answer stops. The threshold has to be read off the built index. The similarity has to be computed over every unordered pair rather than over the 144 the index proposed. The relation has to be closed into groups, which adds 10 pairs the threshold alone never joins.

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F4 carries no measured family baseline or target, so none is invented. The reward schema is bound in full and the two normalisation numbers below are measured on this slot's own frozen substrate; they are instance-local and are NOT family anchors, under gap `gap-oer-27-instance-local-anchors-not-family-anchors`.

| quantity | value |
|---|---|
| instance_baseline_recall | 0.864583 |
| instance_target_recall | 1.0 |

The baseline is the honest shallow answer: keep the index's own candidate pairs whose similarity reaches the calibrated threshold, and report those. That answer names the threshold correctly, holds precision, carries 83 pairs, and reaches recall 0.864583, which normalises to exactly 0.0. The gap between it and full reward is 13 pairs: the ones the band structure never proposed, and the ones only the grouping joins.

raw = (agent_metric - instance_baseline_recall) / (instance_target_recall - instance_baseline_recall)

score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and beating it also scores 1.0.

## The clock

THE HARNESS OWNS THE CLOCK. Nothing on the graded path is a timing quantity, and no checker calls `time`, `datetime`, `perf_counter` or `monotonic`, directly or transitively. `tests/runner.py` launches the harness-owned builder over the bundle's frozen corpus in an isolated subprocess and every checker reads the resulting index telemetry through the `tests/checkers.py` `Harness` handle. `recompute.py` reads no clock either.

## Substrate limit

The corpus is synthetic and generated by arithmetic, declared under gap `gap-oer-27-corpus-is-synthetic-not-enwik`. That is the right choice because it lets the verifier rebuild the index and recompute every similarity exactly, so the same submission produces the same verdict on any host. Nothing measured here licenses a claim about compression ratio on enwik8, enwik9 or any other published corpus.
