# The frozen substrate, slot OER-10

Everything in this directory is read-only to you. The lower layer an agent sees is
assembled and locked before your first byte runs, and every write into it is refused.
You write to `/workspace/submission/` and nowhere else.

## What is frozen

| axis | value | where |
|---|---|---|
| token budget | 419430400 tokens, counted as fed | `frozen_train.py` |
| model | `frozen-decoder-12L-768d` | `frozen_train.py` |
| optimizer | `frozen-adamw-cosine` | `frozen_train.py` |
| evaluation split | `val-split-b`, 60 documents | `held_out_split.json` |

## What is free

The data mixture and the curation recipe, and nothing else.

## Files

| file | what it is |
|---|---|
| `raw_pool.json` | the 240-document raw pool your recipe curates. Each row carries `bucket`, `len_tokens`, `dup_class`, `quality_decile`, `ppl_decile` and `lang`. |
| `probe_pool.json` | a frozen 48-document probe pool. Your recipe is executed over it, and what it does there is your recipe's behavioural fingerprint. |
| `held_out_split.json` | the frozen evaluation split. Its document ids live in the `h###` namespace, disjoint from the raw pool's `r###`. Feeding one is a leak, not an accident. |
| `exclusion_set.json` | the pinned exclusion set: the behavioural fingerprint of every published mixture baseline for this setup. |
| `published_mixtures.md` | the write-ups those baselines come from. Read them. They are excluded, not secret. |
| `recipe_api.py` | the interface your submission implements, with a runnable example. |
| `frozen_train.py` | the frozen trainer. Read it; do not modify it. |

## The screen

Before any training runs, the harness executes your recipe over `probe_pool.json`,
quantises the weight you assign each probe document to an integer 0..8, and compares the
resulting 48-long vector against every entry of `exclusion_set.json` on two levels:

1. **exact** — the sha256 of your vector equals a pinned vector's sha256.
2. **behavioural proximity** — the L1 distance from your vector to the nearest pinned
   vector is below the pinned `l1_floor`.

Either match is a rejection, and the rejection happens at stage 3 of 6, before training
starts at stage 5. A replay therefore costs you zero accelerator time and zero budget, and
you can screen a candidate as often as you like for free.

The screen is behavioural. It measures what your recipe **does**, not what its source says.
Renaming the identifiers, reordering the clauses, rewriting the comments, spelling the
weights as fractions or nudging a threshold cosmetically all leave the selection vector
where it was, so none of them gets a published mixture past the screen.
