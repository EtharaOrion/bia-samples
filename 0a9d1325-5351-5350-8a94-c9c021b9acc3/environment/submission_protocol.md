# Submission protocol

Write one JSON document to:

```
/workspace/submission/filter.json
```

`/workspace` is the only filesystem path shared between this container and the
verifier's. `/app` is this image's own, and a file left there is never graded.

## Shape

```json
{
  "schema": "oer-curation-filter/v1",
  "notes": "optional free text; copied into the score document, never parsed",
  "rules": [
    {"field": "bigram_coherence", "op": "ge", "value": -13.8}
  ]
}
```

* `schema` must be exactly `oer-curation-filter/v1`.
* `rules` is a list of at most 16 predicates.
* `op` is one of `ge`, `gt`, `le`, `lt`, `ne`. An unknown operator is a refusal —
  operators are the runner's own vocabulary.
* `field` is any non-empty string. It is **not** checked against the register.
  See the fail-open note in `filter_schema.py`.
* `value` must be a finite number.

A block is admitted when every predicate that resolved accepts it.

## What the verifier refuses

Every refusal is reward 0.0 with a machine-readable reason, never a scaled value:

`submission-absent`, `filter-unreadable`, `filter-not-an-object`,
`filter-key-unknown`, `filter-key-missing`, `filter-schema-unrecognised`,
`filter-rules-not-a-list`, `filter-too-many-rules`, `filter-rule-not-an-object`,
`filter-field-not-a-string`, `filter-op-unknown`, `filter-value-not-a-number`,
`pool-underfilled`, `accelerator-absent`, `verifier-substrate-unavailable`,
`verifier-anchor-invalid`, `verifier-anchor-underfilled`, `anchor-run-failed`,
`anchor-diverged`, `calibration-span-nonpositive`, `agent-run-failed`,
`agent-diverged`, `grading-chain-failed`, `grading-timed-out`, `grading-killed`.

## What is never read

Nothing your submission says about itself. `notes` is copied into the score
document for a human reader and is never parsed. The runner's own report is
recorded for the reader and never scaled into the reward: a chain that resolved
nothing is not penalised by a rule, it simply trains on the uncurated pool and
therefore measures like the control.
