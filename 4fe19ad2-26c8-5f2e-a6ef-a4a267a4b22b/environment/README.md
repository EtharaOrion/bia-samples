# The substrate

| file | what it is |
|---|---|
| `shape.json` | the bound operating point. Every field in it is a frozen axis. |
| `recipe_schema.json` | the closed key set a recipe may carry, and the frozen keys it may not. |
| `frozen_gpt.py` | the frozen architecture, plus the initialization hook a recipe reaches. |
| `bia_optim.py` | the optimization algorithms and schedules. This is the free axis, as code. |
| `bia_data.py` | the deterministic corpus. The development seed is here; the graded seed is not. |
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

## The splits

Three streams, three seeds, no slicing of one stream into three.

- train: `data_seed_train`, present here, used by training.
- dev: `data_seed_dev`, present here, what `train_frozen.py` reads.
- val: the graded split. Its seed exists only inside the verifier environment.

A curve from `train_frozen.py` is a development curve. The graded crossing is
computed by the verifier on the val split, unsmoothed, at points the verifier
schedules.
