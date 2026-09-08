# Spend a fixed parameter budget: decoder shape at fixed compute

You are given a frozen decoder *design*, a frozen training recipe, a frozen token
budget, and a decoder *shape* that is not very good. Find a better one.

## The setup

`/app` holds the whole task.

| path | what it is |
|---|---|
| `frozen/task_spec.json` | the frozen substrate: parameter budget, shape grid, token budget, training recipe, init seed, evaluation window |
| `model/nanogpt.py` | the frozen decoder — rotary attention, pre-norm residuals, squared-ReLU MLP, logit softcap. Its SHAPE is a constructor argument |
| `harness.py` | the building, training and evaluation harness |
| `shape_schema.py` | the shape schema, the grid, and the parameter arithmetic |
| `default_shape.json` | the shape you have to beat |
| `train_local.py` | measure a shape end to end |
| `data/train_slice.bin` | the training corpus |
| `data/devset_slice.bin` | a proxy validation split, cut from a different FineWeb shard |
| `submission_protocol.md` | the schema field by field, and the reward formula |

## What you submit

Four integers, as JSON, at:

    /workspace/submission/shape.json

`num_layers`, `model_dim`, `head_dim`, `mlp_ratio`. That is the whole free variable.

Your block stack must hold **10,642,944 non-embedding parameters, ±5%**. The embedding
and the output projection are not counted. **25** points on the grid satisfy the band,
running from 4 blocks at width 512 to 16 blocks at width 256.

## What the choice actually costs you

Every admissible shape spends the same parameter budget and sees the same 12,582,912
tokens under the same recipe. What changes is how the budget is arranged.

* **Depth against width.** A deeper, narrower stack composes more transformations of a
  lower-dimensional state; a shallower, wider one does fewer, richer ones. At a fixed
  parameter count these trade directly: `num_layers` scales roughly as `1/model_dim^2`.
* **Where the parameters sit inside a block.** `mlp_ratio` moves budget between the
  attention block and the feed-forward block. At `mlp_ratio` 2 a block is
  attention-heavy; at 4 it is the usual feed-forward-heavy shape — and the same budget
  then buys fewer blocks.
* **How the width is cut into heads.** `head_dim` changes the number of attention
  heads without changing a single parameter count. It is free in the budget and not
  free in the result.
* **What the shape costs to run.** The output projection is `model_dim x 50304`, so a
  wide shape spends more compute per token on the head than a narrow one even at an
  identical parameter budget. The budget is counted in parameters and in tokens, not
  in FLOPs.

The training recipe is frozen and is not yours to change. A shape has to work under
*this* recipe.

## How you are graded

The verifier builds and trains **three** decoders on the grading run: the shipped
default shape, a stronger reference shape it holds privately, and yours. It evaluates
all three on a held-out FineWeb slice that is not in this image, and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints are **measured on the grading run** by real from-scratch training runs.
No number in this bundle is either of them. Tying the default scores 0.0; reaching or
beating the reference scores 1.0; everything between is the fraction of the gap closed.

## Working method

`train_local.py` runs the same harness the verifier runs. Measure candidates against
`data/devset_slice.bin` — it will sit at an offset from the graded split but rank
shapes the same way. One run of the budget is a real training run; there are 25
admissible shapes and you cannot afford all of them, so choose what to measure.
