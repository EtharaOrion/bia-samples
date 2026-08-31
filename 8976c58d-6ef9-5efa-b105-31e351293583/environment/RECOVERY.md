# The durable recovery area

`/workspace/ledger/` is agent-writable and persists across every iteration of
the session. Nothing in this directory is part of the agent-visible lower layer,
so making it writable takes no write bit off `task.toml`, `instruction.md` or
`environment/`. Those stay read-only, and a write to them is refused.

## Why it exists

The summary the loop hands you at the start of each iteration is bounded and
lossy. When the harness compacts it, entries are merged and the `established`
lists attached to dropped entries do not survive. Nothing announces the
compaction.

If the summary were the only record, a compaction would be unrecoverable and the
loop would simply punish forgetting. It is not the only record. This directory
is the durable one, and recovery from it is the skill under test.

## The record shape

`/workspace/ledger/ledger.jsonl`, one JSON object per line, append-only by
convention:

```json
{"approach_id": "a3", "status": "refuted", "established": ["lr-ceiling-0.008"], "iteration": 3, "reward": 0.21}
```

Fields:

- `approach_id` — stable id for the approach this line is about
- `status` — `tried` or `refuted`
- `established` — the constraints and refutations this attempt settled
- `iteration` — the loop iteration that produced it
- `reward` — the reward that attempt earned, if it was graded

A reconstruction record is the same shape plus two fields, and it is what the
graded run reads when it asks whether you recovered:

```json
{"approach_id": "recon-4", "status": "tried", "established": [],
 "iteration": 4, "reconstructed_from_ledger": true,
 "recovers": ["a1", "a2"]}
```

`recovers` names every `approach_id` that the compaction dropped from the
summary and that you restored from this ledger.

## The discipline

1. Read the whole ledger before you propose anything.
2. Diff the ledger against the summary you were handed.
3. Where they disagree, the ledger is authoritative. The summary is a view; the
   ledger is the record.
4. Append after every attempt. A ledger written only at the end is a ledger that
   was never there when it was needed.
