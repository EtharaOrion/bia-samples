# OER-11 task substrate

| file | who owns it | what it is |
|---|---|---|
| `corpus/` | the harness | the raw corpus, behind a snapshot version that MOVES |
| `corpus_api.py` | the harness | the first-class handle for asking what version the corpus is at |
| `train.py` | frozen | the model, the optimizer, the token budget and the eval split |
| `pipeline.py` | **you** | the parse and tokenize pipeline |

## The one thing to internalise

`corpus/SNAPSHOT.json` carries `snapshot_version`. It changes during your session and
**nothing announces the change**. `corpus_api.snapshot_version()` re-reads it on every
call and is never cached.

A `Corpus` handle from `resolve_corpus()` is stamped with the version it resolved at
and does not follow the corpus forward. Held across a move it answers every question
consistently and staleley, which is precisely how a parse verdict built at snapshot 3
survives into a graded evaluation at snapshot 7 without anyone noticing.

The harness records the version at your last `resolve_corpus()` as the version your
parse rules were built against, and the graded evaluation carries the version it ran
on. If they differ the submission is rejected with reason
`parse-built-against-stale-snapshot`.

Ordering is carried by the snapshot version sequence in `snapshot_ledger()`, not by any
clock. A higher version is a later corpus. Equal versions are the same corpus.

## The corpus record shape

Each shard is JSON Lines. One object per line:

```json
{"doc_id": "d-0007", "url": "...", "lang": "en", "body": "...", "fields": {"...": "..."}}
```

The set of keys under `fields` is **snapshot-dependent**. A key that exists at one
snapshot may not exist at the next, and a new one may appear. Parsing `fields` by a
fixed key list derived from an earlier snapshot is the canonical way to build a stale
parser here.

## Reading the harness's own records

Everything the checkers read is under `/logs/harness/`, written by the harness process,
never by you:

| record | what it fixes |
|---|---|
| `snapshot_ledger.json` | the ordered snapshot version sequence; the ordering basis |
| `parse_manifest.json` | the version your parse rules were built against |
| `pipeline_effect.json` | whether the token stream was produced this run or replayed |
| `token_budget.json` | tokens as FED, per shard and total |
| `split_manifest.json` | the evaluation split and the training document ids |
| `weights_ledger.json` | the weights digest at every evaluation step |
| `eval_ledger.json` | the verifier's own per-batch losses and the graded point |

You can read these. You cannot write them, and a value you write anywhere else does not
reach the graded path.
