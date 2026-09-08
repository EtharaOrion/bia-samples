# Beat the shipped training recipe on a fixed compute budget

You are given a frozen nanoGPT, a frozen data pipeline, a frozen compute budget, and
a training recipe that is not very good. Find a better one.

## The setup

`/app` holds the whole task.

| path | what it is |
|---|---|
| `frozen/task_spec.json` | the frozen substrate: architecture, init seed, evaluation window, compute budget |
| `model/nanogpt.py` | the frozen decoder — 6 layers, model_dim 384, head_dim 64, seq_len 512, vocab 50304 |
| `harness.py` | the training and evaluation harness |
| `recipe_schema.py` | the recipe schema and its bounds |
| `default_recipe.json` | the recipe you have to beat |
| `train_local.py` | measure a recipe end to end |
| `data/train_slice.bin` | the training corpus — exactly the tokens the budget consumes |
| `data/devset_slice.bin` | a proxy validation split, cut from a different FineWeb shard |
| `submission_protocol.md` | the schema field by field, and the reward formula |

The compute budget is **3072 micro-batches of 16x512 tokens**, or 25,165,824 tokens,
and it is fixed in forward and backward passes rather than in wall clock. A recipe
chooses how many micro-batches accumulate into one optimizer step; it cannot change
how many are run. Every recipe costs the same FLOPs, so no recipe wins by training
longer — only by training better.

## What you submit

A recipe, as JSON, at:

    /workspace/submission/recipe.json

The free variables are the optimizer and the schedule: four per-role peak learning
rates (token embedding, block matrices, output projection, and the RMSNorm gains and
biases), AdamW `beta1`, `beta2`, `eps` and `weight_decay`, the gradient clip, the
gradient accumulation, and the schedule — warmup fraction, shape
(`constant`, `linear`, `cosine`, `wsd`), the final learning rate as a fraction of
peak, and the `wsd` stable fraction.

`submission_protocol.md` has the exact schema and the bound on every field. Unknown
keys are refused rather than ignored.

## How you are graded

The verifier trains **three** models on the grading run, all from the same frozen
initialisation, over the same tokens, with the same compiled kernels: the shipped
default, a stronger reference recipe it holds privately, and yours. It evaluates all
three on a held-out FineWeb slice that is not in this image, and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both ends of that scale are measured on your run by real training runs. Neither is a
stored constant. Tying the default scores 0.0, reaching the reference scores 1.0, and
beating the reference also scores 1.0. Partial progress scores in between, in
proportion to the fraction of the gap you closed.

Nothing your submission says about itself is graded. The recipe declares
hyperparameters and nothing else; `notes` is free text that is copied into the score
document and never parsed. The loss is computed by the verifier, in the verifier's
process, on data you do not have.

## Measuring your own work

    python3 train_local.py                          # the shipped default
    python3 train_local.py cand_a.json cand_b.json  # candidates, one warm process

One run costs roughly a minute on the H100. Passing several recipes to a single
invocation is much cheaper than several invocations, because the model is built once,
`torch.compile` is paid once and the corpus is read once.

`train_local.py` reports loss on `data/devset_slice.bin`. That is a proxy, not the
graded split: expect a small constant offset, and expect it to rank recipes the same
way the grader does. Budget your time so that the recipe you submit is one you have
actually measured.

## Constraints

- Do not modify the frozen files. The verifier reads its own copies of
  `harness.py`, `recipe_schema.py`, `model/nanogpt.py`, `frozen/task_spec.json` and
  `default_recipe.json`, and refuses to build unless they are byte-identical to the
  ones on this surface. Editing them here changes nothing except your ability to
  predict your own score.
- The graded split is not in this image and cannot be recovered from it.
- Write the recipe to `/workspace/submission/recipe.json`. `/app` is not shared with
  the verifier.
