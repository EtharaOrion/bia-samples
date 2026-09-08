# FORGE-CANARY-BEGIN
# 0: 83d3ea0d6adfb64f9a017c0f86a587fadaa6af871f8fa2b19a67a25b8f71534e
# 1: dbd98627eda8ae52fcf284d974e7cd2085aa4d8e60ab17e70a88e78cb5a88e38
# 2: c542d0a8399cd1bc97157f9fc9ed618d2d202f0ae2263ddea49e0084054d2a67
# 3: 6b1086b20248cad81854bda323a0b7f6877ad401abb77d192ab66e8806265376
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
| adjudication threshold `686` | the similarity level at which the built index calibrates the near-duplicate boundary, on the grid of 1024 | `nd_index.calibrate` over the built index's own candidate similarity histogram |
| collision witness `r0017`, `r0381` | the refused collision maximal by band count, then similarity, then lexicographic order | the built index's band collisions, once the duplicate grouping has been carried out |

The calibration cuts the candidate similarity histogram at its maximum-separation split, which lands at 654, then snaps up to the smallest candidate level at or above it, which is 686. The witness is seen in 3 bands at similarity 615.

## The reference trajectory, as it actually ran

| quantity | value |
|---|---|
| records in the frozen corpus | 512 |
| unordered pairs examined | 130816 |
| colliding pairs the built index holds | 80 |
| adjudication threshold | 686 |
| thresholded edges over the whole universe | 19 |
| duplicate groups with two or more members | 13 |
| group size histogram | 2: 9, 3: 4 |
| near-duplicate pairs after closure | 21 |
| collision witness | r0017, r0381 |

Three separate things have to be right for the reference answer to be right, and each of them is a place a shallow answer stops. The threshold has to be read off the built index. The similarity has to be computed over every unordered pair rather than over the 80 the index proposed. The relation has to be closed into groups, which adds 2 pairs the threshold alone never joins.

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F4 carries no measured family baseline or target, so none is invented. The reward schema is bound in full and the two normalisation numbers below are measured on this slot's own frozen substrate; they are instance-local and are NOT family anchors, under gap `gap-oer-27-instance-local-anchors-not-family-anchors`.

| quantity | value |
|---|---|
| instance_baseline_recall | 0.809524 |
| instance_target_recall | 1.0 |

The baseline is the honest shallow answer: keep the index's own candidate pairs whose similarity reaches the calibrated threshold, and report those. That answer names the threshold correctly, holds precision, carries 17 pairs, and reaches recall 0.809524, which normalises to exactly 0.0. The gap between it and full reward is 4 pairs: the ones the band structure never proposed, and the ones only the grouping joins.

raw = (agent_metric - instance_baseline_recall) / (instance_target_recall - instance_baseline_recall)

score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and beating it also scores 1.0.

## The clock

THE HARNESS OWNS THE CLOCK. Nothing on the graded path is a timing quantity, and no checker calls `time`, `datetime`, `perf_counter` or `monotonic`, directly or transitively. `tests/runner.py` launches the harness-owned builder over the bundle's frozen corpus in an isolated subprocess and every checker reads the resulting index telemetry through the `tests/checkers.py` `Harness` handle. `recompute.py` reads no clock either.

## Substrate

The corpus is REAL. Every one of its 512 records is one upstream FineWeb10B document, copied without transformation out of fineweb_train_000001.bin, fineweb_train_000002.bin, fineweb_train_000003.bin, and the collection is named in the carrier's own bytes at https://huggingface.co/datasets/HuggingFaceFW/fineweb. `environment/corpus.jsonl` digests to `9e4ceadda5cdb099a33c90c375df86078a41a4c57235a7713cddd8493a109be0` and that digest is pinned in `solution/grounding.yaml`, which `recompute.py` holds the vendored bytes against before deriving anything from them. This bundle owns no configuration that could reproduce those bytes, which is exactly the property that makes the collection real rather than a stand-in expanded from the bundle's own seed.

The redundancy the adjudication is over is real too, and is not put there by this bundle: it is the near-duplicate structure that survives FineWeb's per-dump deduplication, which is why 96 real pair families, 40 real triple families and 16 real quad families can be cut from three shards at all. The selection rule is recorded in full under `corpus` in the grounding and is re-executable from those shards by `recompute.py --corpus-only`.

Grading stays exactly reproducible under the substitution, because the verifier rebuilds the index and recomputes every similarity from the frozen corpus bytes rather than from any property of how those bytes were obtained. Nothing measured here licenses a claim about compression ratio on enwik8, enwik9 or any other published compression corpus, which is declared under gap `gap-oer-27-corpus-is-fineweb-not-enwik`.
