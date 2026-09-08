# Choose the training mixture, at a token budget you cannot move

You have one accelerator, a 49.3M-parameter nanoGPT, six training shards, and a budget
of exactly 3072 blocks of 8192 tokens. The architecture, the initialisation, the
optimizer and the schedule are all frozen and tuned. **What you choose is which tokens
get spent, and in what order.**

Write one JSON document to:

```
/workspace/submission/mixture.json
```

## The shards

Six shards, each 8,388,608 GPT-2 BPE tokens, each yielding 1023 whole blocks. They
came out of six candidate preprocessing pipelines run over the same crawl.
`environment/README.md` records how each pipeline differed. It does not record what
any of it is worth — the study that produced them never established that, and
establishing it is the task.

Six shards offer 6138 blocks; the budget spends 3072. **Every mixture leaves something
out.** A mixture concentrated hard enough on one shard runs out of blocks and reads it
again from the beginning; that is allowed, it is not an error, and finding out what it
costs is part of the work.

## The document

```json
{
  "schema": "oer-nanogpt-mixture/v1",
  "notes": "free text, never parsed",
  "mixture": {
    "shard_a": 1.0, "shard_b": 1.0, "shard_c": 1.0,
    "shard_d": 0.0, "shard_e": 0.0, "shard_f": 0.0
  },
  "order": {"policy": "interleave", "block_group": 1, "seed": 0}
}
```

Every shard must carry an explicit weight; `0.0` is how you exclude one. The weights
are normalised and converted to integer block counts by largest-remainder, so they sum
to 3072 exactly.

**There is no token-budget field, and there is no way to overspend.** A weight vector
says how the fixed budget is divided, not how large it is. The budget is enforced by
the shape of the submission rather than by a rule you are trusted to respect.

`order.policy` arranges the same multiset of blocks three ways — `interleave`,
`sequential`, `blocked` (chunks of `block_group`, permuted under `seed`). A block
always reads its tokens from inside one shard, so ordering changes the sequence the
pairs arrive in and never which pairs exist.

`environment/mixture_schema.py` is the schema the verifier uses, byte for byte.

## How you are scored

The verifier trains **three** models from scratch on this run:

1. the shipped `environment/default_mixture.json` — equal weight on all six shards,
   evenly interleaved: the mixture you write when you have not looked at the data. The
   LOW anchor.
2. a stronger reference mixture that exists only inside the verifier image. The BAR.
3. your submission.

and computes, on a held-out FineWeb slice that is not in your image and is not a cut
of any training shard:

```
reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )
```

Neither anchor is a stored number. Both are measured, from scratch, on every grading
run.

## Measuring before you submit

```
python3 /app/train_local.py candidate_a.json candidate_b.json
```

runs the same harness, the same budget and the same frozen initialisation the verifier
will use, on `data/devset_slice.bin` — a proxy split cut from a FineWeb training shard
that is **not** one of the six, and not the graded split. Roughly two minutes per run
on an H100.

The cheapest informative experiment is not a full sweep over six weights. Think about
what a single run can isolate.

## What will get you refused

Refusals score 0.0 with a machine-readable reason: a missing submission, an
unparseable document, a shard left out, a weight outside [0, 1], all weights zero, an
unknown policy, a `block_group` that is not a divisor of 3072. A legal mixture that
trains badly is a wrong answer, not a refusal.
