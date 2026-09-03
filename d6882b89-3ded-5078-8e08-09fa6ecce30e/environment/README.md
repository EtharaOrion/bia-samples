# OER-09 environment

| path | what it is |
|---|---|
| `frozen_config.yaml` | the frozen axes: token budget, model, optimizer, evaluation split, and the register schema the harness observes |
| `pool/source_register.jsonl` | the source pool, one JSON document per line. The authority on its own schema |
| `pool/REGISTER.md` | documentation of the register schema. Documentation, not authority |
| `curate.py` | the curation filter runner. Reads a filter file, writes a curated pool and a report |
| `train.py` | the frozen training entry point |
| `pool_probe.py` | prints what the harness observed about the pool, so you can check your filter moved it |

## Harness-owned records

`/logs/harness/` is written by the harness and read by you. Nothing you write there is read as
graded state. It carries the harness's own observation of the pool, the feed ledger, the
held-out leak audit, the weights ledger and the verifier's evaluation record.

## The one habit that pays here

Run `python3 environment/pool_probe.py` after curating and before training. A filter runner
that exits `0` and reports `status: ok` has told you it did not crash. It has not told you it
moved the pool.
