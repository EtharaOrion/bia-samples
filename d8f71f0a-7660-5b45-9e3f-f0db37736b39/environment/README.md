# The frozen substrate, slot OER-10

Everything in this directory is read-only to you. The lower layer an agent sees is assembled and locked before your first byte runs, and every write into it is refused. You write to `/workspace/submission/` and nowhere else.

## What is frozen

| axis | value | where |
|---|---|---|
| architecture | 12 layers, 768 model dim, 128 head dim, 6 heads, vocab 50304, seq len 1024 | `nanogpt_substrate.json` |
| batch | 524288 tokens per step, one forward and one backward pass per step | `nanogpt_substrate.json` |
| token budget | 1677721600 tokens, being 3200 steps at that batch, counted as fed | `frozen_train.py` |
| optimizer | `frozen-adamw-cosine` | `frozen_train.py` |
| evaluation split | `val-split-b`, verifier-owned, absent from this container | `held_out_split.json` declares it |

## What is free

The data mixture and the curation recipe, and nothing else.

## Files

| file | what it is |
|---|---|
| `nanogpt_substrate.json` | the canonical nanoGPT operating point this task's metric resolves against. `frozen_train.py` reads its architecture rather than restating it, so the model that trains is this declaration's model. |
| `raw_pool.json` | the 240-span catalogue your recipe curates. Each row carries `bucket`, `len_tokens`, `dup_class`, `quality_decile`, `ppl_decile` and `lang`, plus the binding `shard`, `token_offset` and `span_tokens` naming the contiguous run of FineWeb10B training tokens that row is. |
| `probe_pool.json` | a frozen 48-row probe pool. Your recipe is executed over it, and what it does there is your recipe's behavioural fingerprint. It is a descriptor pool for the screen and is never fed. |
| `held_out_split.json` | a DECLARATION of the evaluation split and nothing else. It names `val-split-b` and the validation shard glob, and it carries no member ids, no token offsets and no token bytes. The split itself is owned and resolved by the verifier. |
| `exclusion_set.json` | the pinned exclusion set: the behavioural fingerprint of every published mixture baseline for this setup. |
| `published_mixtures.md` | the write-ups those baselines come from. Read them. They are excluded, not secret. |
| `recipe_api.py` | the interface your submission implements, with a runnable example. |
| `frozen_train.py` | the frozen trainer. Read it; do not modify it. |

## Why the evaluation split is not here

A graded split whose tokens sit in this directory is a split you can read, and a scalar computed over bytes you hold is not a held-out measurement. Earlier bytes of this slot shipped the validation token payload on this surface; they no longer do. The verifier resolves the split from its own staged FineWeb10B validation shard, checks it against a digest that is not on this surface, and evaluates the harness's parameter snapshots against it. `held_out_split.json` tells you what you are graded on so you are not guessing, and gives you nothing to fit to.

## The screen

Before any training runs, the harness executes your recipe over `probe_pool.json`, quantises the weight you assign each probe row to an integer 0..8, and compares the resulting 48-long vector against every entry of `exclusion_set.json` on two levels:

1. **exact**, meaning the sha256 of your vector equals a pinned vector's sha256.
2. **behavioural proximity**, meaning the L1 distance from your vector to the nearest pinned vector is below the pinned `l1_floor`.

Either match is a rejection, and the rejection happens at stage 3 of 6, before training starts at stage 5. A replay therefore costs you zero accelerator time and zero budget, and you can screen a candidate as often as you like for free.

The screen is behavioural. It measures what your recipe **does**, not what its source says. Renaming the identifiers, reordering the clauses, rewriting the comments, spelling the weights as fractions or nudging a threshold cosmetically all leave the selection vector where it was, so none of them gets a published mixture past the screen.
