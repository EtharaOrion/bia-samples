# Curate the source pool

A 49.3M-parameter nanoGPT is going to be trained on 3072 blocks of 8192 GPT-2 BPE
tokens drawn from a pool of 5120 candidate blocks. The architecture, the
initialisation, the optimizer, the learning-rate schedule, the token budget and
the number of optimizer steps are all **frozen**. The only thing you choose is
**which blocks of the pool get used**.

You choose them by writing a filter chain to:

```
/workspace/submission/filter.json
```

## The reward

The verifier trains **three** models from scratch on the grading run and divides
two numbers it measured itself:

```
reward = clamp( (control_loss - your_loss) / (control_loss - reference_loss), 0, 1 )
```

* `control_loss` — the shipped `default_filter.json`, retrained from scratch. This
  is the uncurated floor. Tie it and you score 0.0.
* `reference_loss` — a chain held only inside the verifier image. Reach it and you
  score 1.0; beat it and you also score 1.0.

Neither endpoint is a stored number. Both are measured by real training runs on
every grading run, so there is nothing to look up and nothing to precompute.
Validation cross-entropy is measured on a held-out FineWeb slice that is **not in
this image**.

## The pool and its register

`pool/pool_blocks.bin` holds 5120 blocks of 8192 tokens, in order.
`pool/source_register.jsonl` holds one JSON object per block, in the same order.
Every numeric field in a register row was **computed from that block's own
tokens** during authoring. The fields are documented in `pool/REGISTER.md`.

Some of the blocks in this pool are degraded. They are degraded in more than one
way, and the ways do not look alike.

## The filter chain

```json
{
  "schema": "oer-curation-filter/v1",
  "notes": "free text, never parsed",
  "rules": [
    {"field": "<register field>", "op": "ge|gt|le|lt|ne", "value": <number>}
  ]
}
```

A block is admitted when **every** predicate accepts it. At most 16 predicates.

## Two constraints that bind

1. **The budget consumes 3072 blocks.** A chain that admits fewer than 3072 is
   refused outright with `pool-underfilled` and scores 0.0. You cannot simply
   tighten until only the safest blocks survive. A chain that admits more than
   3072 has the surplus ignored — the first 3072 in register order are used.

2. **The runner fails open.** A predicate whose `field` is not a key of a register
   row resolves nothing. It does not raise and it does not refuse: it matches
   everything, so it is a no-op, and the chain carries on. A chain in which every
   predicate names an absent field admits the pool **unchanged** while the runner
   exits zero and reports that it ran.

   The shipped `default_filter.json` is exactly such a chain. Run it and read the
   report.

## Tools in this image

```bash
python3 curate.py --filter default_filter.json --breakdown
python3 train_local.py --filter my_filter.json --compare
```

`curate.py` applies a chain to the register and prints what it did —
`rules_resolved`, `rules_unresolved`, `admitted`, and whether the pool passed
through unchanged. `train_local.py` trains the frozen substrate on the blocks a
chain admits and evaluates on `data/devset_slice.bin`, a **proxy** cut from a
different FineWeb shard. The proxy is not the graded split, but it is measured by
the same `harness.py`, from the same frozen initialisation, on the same pool.

`harness.py`, `filter_schema.py`, `model/nanogpt.py`, `frozen/task_spec.json` and
`default_filter.json` are byte-identical inside the verifier; its image refuses to
build if they are not.

## Deliverable

`/workspace/submission/filter.json`. `/workspace` is the only path shared with the
verifier — a file left in `/app` is never graded.
