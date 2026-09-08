# Spend a fixed token budget on the right data, in the right order

You are given a frozen nanoGPT, a frozen optimizer, a frozen compute budget, and a
pool of eight token sources that are **not** equally good. Decide what to train on.

## The setup

`/app` holds the whole task.

| path | what it is |
|---|---|
| `frozen/task_spec.json` | the frozen substrate: architecture, init seed, budget, evaluation window, and the pool manifest |
| `frozen/train_recipe.json` | the frozen optimizer and schedule — published so you can reproduce a run, not a free variable |
| `model/nanogpt.py` | the frozen decoder — 6 layers, model_dim 384, head_dim 64, seq_len 512, vocab 50304 |
| `harness.py` | the plan interpreter, the training loop and the evaluation |
| `plan_schema.py` | the plan schema and its bounds |
| `default_plan.json` | the plan you have to beat |
| `probe_local.py` | measure a plan end to end |
| `data/sources/` | the eight pool sources, 4194304 tokens each |
| `data/devset_slice.bin` | a proxy validation split, cut from a different FineWeb shard |
| `submission_protocol.md` | the schema field by field, and the reward formula |

The compute budget is **1024 micro-batches of 16x512 tokens**, or 8,388,608 tokens,
and it is fixed in forward and backward passes rather than in wall clock. The pool
holds 33,554,432 tokens, so the budget buys **a quarter of it**. Every plan costs the
same FLOPs; no plan wins by training longer, only by training on better material in a
better order.

## The pool

`frozen/task_spec.json` declares what was done to each source: three are unmodified
FineWeb text, two had their tokens permuted inside 256-token windows, two are a short
passage repeated to fill the source, and one alternates clean and permuted windows.

**What was done to each source is public. What each one is worth is not.** A
block-permuted source still teaches unigram statistics and some short-range
structure. A looped source is perfectly fluent text that stops paying once you have
seen its period. A spliced source is half of each. Whether any of that is worth
budget, and whether damaged material is better spent early than late, is not readable
off the manifest — it has to be measured.

## What you submit

A plan, as JSON, at `/workspace/submission/plan.json`: an **ordered** list of up to
32 draws, each naming a source and a token count, totalling **exactly** the frozen
budget. `submission_protocol.md` has the exact schema and the bound on every field.

## How you are graded

The verifier trains **three** models on the grading run, all from the same frozen
initialisation, with the same optimizer and the same compiled kernels, over three
assembled streams of identical length: the shipped default plan, a stronger reference
plan it holds privately, and yours. It evaluates all three on a held-out FineWeb
slice that is not in this image, and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both ends of that scale are measured on your run by real training runs. Neither is a
stored constant. Tying the default scores 0.0, reaching the reference scores 1.0, and
beating it also scores 1.0.

## Measuring your own work

    python3 probe_local.py                     # the shipped default plan
    python3 probe_local.py --solo              # one plan per source, whole budget each
    python3 probe_local.py a.json b.json       # candidates, one warm process

One run costs well under a minute on the H100. Passing several plans to a single
invocation is much cheaper than several invocations, because the model is built once,
`torch.compile` is paid once and the pool is read once. `--solo` is the cheapest way
to find out what each source is actually worth.

`probe_local.py` reports loss on `data/devset_slice.bin`. That is a proxy, not the
graded split: expect a small constant offset, and expect it to rank plans the same
way the grader does.

## Constraints

- Do not modify the frozen files. The verifier reads its own copies of `harness.py`,
  `plan_schema.py`, `model/nanogpt.py`, `frozen/task_spec.json` and
  `frozen/train_recipe.json`, and refuses to build unless they are byte-identical to
  the ones on this surface.
- The graded split is not in this image and cannot be recovered from it.
- Write the plan to `/workspace/submission/plan.json`. `/app` is not shared with the
  verifier.
