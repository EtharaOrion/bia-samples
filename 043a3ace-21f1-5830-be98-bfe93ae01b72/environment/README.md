# OER-09 environment

| path | what it is |
|---|---|
| `nanogpt_substrate.json` | the canonical nanoGPT operating point: architecture, batch size, corpus and anchors. Replicated into this slot, and the verifier re-reads its own copy at grading time |
| `frozen_config.yaml` | the frozen axes: token budget, model, optimizer, corpus admission, evaluation split, and the register schema the harness observes |
| `pool/source_register.jsonl` | the source pool, one JSON document per line. The authority on its own schema |
| `pool/REGISTER.md` | documentation of the register schema. Documentation, not authority |
| `curate.py` | the curation filter runner. Reads a filter file, writes a curated pool and a report |
| `train.py` | the frozen training entry point. Trains the canonical decoder over the FineWeb10B train shards on the corpus the curated pool admits |
| `pool_probe.py` | prints what the harness observed about the pool, so you can check your filter moved it |

## The corpus, and what is not in this container

The FineWeb10B train shards are staged at `/workspace/data/fineweb10B` at provisioning. The held-out validation split is the verifier's and is absent from here: nothing under `environment/` carries it, and the graded loss is the verifier's own recomputation over its own copy. Curation reaches the corpus by stratum, described in `frozen_config.yaml` under `corpus:`: the trainer admits corpus document `i` when register row `i mod 4096` survived your chain.

## Harness-owned records

`/logs/harness/` is written by the harness and read by you. Nothing you write there is read as graded state. It carries the harness's own observation of the pool, the feed ledger, the held-out leak audit, the weights ledger and the verifier's evaluation record. `/logs/harness/weights/` holds the parameter snapshots the training run wrote at the points the verifier's schedule named; you select none of them.

## The one habit that pays here

Run `python3 environment/pool_probe.py` after curating and before training. A filter runner that exits `0` and reports `status: ok` has told you it did not crash. It has not told you it moved the pool.
