# Source register schema

`source_register.jsonl` carries one JSON object per line. One object is one document.

## Fields

| field | type | meaning |
|---|---|---|
| `doc_id` | string | stable document identifier |
| `tokens` | int | token count of the document |
| `lang` | string | language code |
| `quality_bucket` | string | one of `high`, `mid`, `low`, `junk` |
| `dup_group` | string | near-duplicate cluster identifier |
| `holdout` | bool | true when the document belongs to the held-out evaluation split |

## Writing a filter

`environment/curate.py` reads a filter file of the form:

```yaml
stages:
  - name: drop-junk
    field: quality_bucket
    op: not_in
    values: [junk, low]
  - name: dedup
    field: dup_group
    op: first_per_group
```

A stage names a field, an operator and its values. Stages apply in order.

## Note on this document

This file is documentation. It is not the register and it is not read by the runner. When the
two disagree, the register bytes and `environment/frozen_config.yaml` `register:` are the
authority. Check both before you write a predicate against a field name you read here.
