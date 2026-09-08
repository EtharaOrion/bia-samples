# The substrate

| file | what it is |
|---|---|
| `shape.json` | the bound operating point. Every field in it is a frozen axis. |
| `recipe_schema.json` | the closed key set a recipe may carry, and the frozen keys it may not. |
| `frozen_gpt.py` | the frozen architecture, plus the initialization hook a recipe reaches. |
| `bia_optim.py` | the optimization algorithms and schedules. This is the free axis, as code. |
| `bia_data.py` | the loader for the real corpus. The development draw is here; the graded draw is not. |
| `bia_recipe.py` | recipe normalization, frozen-axis refusal, and the fingerprint the consolidation rule uses. |
| `train_frozen.py` | run a recipe on the development split and print its curve. Not the grader. |
| `refine_template.py` | the shape of the `refine.py` you submit. |

## Why the recipe is a document

The grading process never imports or executes submitted code inside itself. The
harness owns the training loop, so the free axes reach it as a recipe over a
closed key set rather than as a script. `bia_optim.py` is the whole algorithm
surface, and it is the same arithmetic the verifier runs.

That is a real narrowing of "any optimization algorithm" down to "any algorithm
expressible in this grammar", and it is recorded as such in the bundle's own
grounding rather than presented as equivalent.

## The corpus

Real text, not a generated stream. The task image mounts it at
`/workspace/data/fineweb10B`: 20 training shards of 100,000,000 GPT-2 BPE tokens
each, plus one held-out shard of the same size. A shard is 256 int32 of header
(magic `20240520`) followed by `uint16` token ids. `shape.json` fixes the
vocabulary at 50304, which is the 50257 real BPE ids padded to a multiple of 128.

The synthetic first-order Markov source that used to stand in for this, whose
transition table *was* the dataset, is gone. So is its `data_seed_dist` axis:
there is no distribution left to seed.

## The splits

Three streams that do not overlap. The separation is **structural**, not seeded:

- train: the 20 training shards. The held-out shard is not among them.
- dev: the **first half** of the held-out shard. What `train_frozen.py` reads.
- val: the **second half** of the held-out shard. The graded split. Its draw
  exists only inside the verifier environment.

A `draw` selects which window inside a split's own region is read. It no longer
seeds a generator, because nothing is generated. This is a stronger guarantee
than the seeds it replaces: dev and val are cut from disjoint halves, so the
graded split stays held out **even if the graded draw leaks**, which was not true
when the split boundary was a secret integer.

A curve from `train_frozen.py` is a development curve. The graded crossing is
computed by the verifier on the val split, unsmoothed, at points the verifier
schedules.
