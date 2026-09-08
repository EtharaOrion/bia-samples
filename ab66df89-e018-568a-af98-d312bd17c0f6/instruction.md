# Split a fixed FLOP budget between model size, tokens and batch size

You are given a fixed compute budget, a menu of seven architectures, a finite corpus
and a frozen optimizer. Decide how to spend the compute.

## The setup

`/app` holds the whole task.

| path | what it is |
|---|---|
| `frozen/task_spec.json` | the frozen substrate: the model menu, the FLOP budget, the FLOP accounting, the init seed, the evaluation window |
| `frozen/train_recipe.json` | the frozen optimizer, schedule and width-scaling rule — published so a run can be reproduced, not a free variable |
| `model/nanogpt.py` | the frozen decoder, instantiated at whichever menu entry you name |
| `harness.py` | the FLOP accountant, the training loop and the evaluation |
| `allocation_schema.py` | the allocation schema and its bounds |
| `default_allocation.json` | the allocation you have to beat |
| `train_local.py` | measure an allocation end to end; `--menu` prices the menu |
| `data/train_slice.bin` | the corpus — 19,521,536 tokens, and it is finite |
| `data/devset_slice.bin` | a proxy validation split, cut from a different FineWeb shard |
| `submission_protocol.md` | the schema field by field, and the reward formula |

## The three axes

You choose a **model** from the menu, a number of **micro_steps** (16x512
micro-batches, which is how many tokens you buy) and a **grad_accum** (how many
micro-batches go into one optimizer step). The constraint that ties them together is

    micro_steps * flops_per_micro_batch(model)  <=  2e15

so a bigger model buys fewer tokens and a smaller one buys more. The accounting is
declared in `frozen/task_spec.json` and implemented in `harness.micro_batch_flops`;
the same code bounds your submission here and charges it at grading time.

Two things are worth knowing before you start. The learning rate **follows the
width** — `frozen/train_recipe.json` prices the hidden matrices at
`base_lr * (384 / model_dim)` — so the menu is not a test of which entry happens to
suit one constant. And the **corpus is finite**: at the small end the budget buys
more tokens than exist, and the stream cycles rather than refusing. Repetition is an
allocation you may make.

## What you submit

An allocation, as JSON, at `/workspace/submission/allocation.json`.
`submission_protocol.md` has the exact schema and the bound on every field.

## How you are graded

The verifier trains **three** models on the grading run, from the same frozen
initialisation, over the same corpus, each inside the same declared FLOP budget: the
shipped default allocation, a stronger reference allocation it holds privately, and
yours. It evaluates all three on the same held-out FineWeb window, which is not in
this image, and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both ends of that scale are measured on your run by real training runs. Neither is a
stored constant. Tying the default scores 0.0, reaching the reference scores 1.0, and
beating it also scores 1.0.

## Measuring your own work

    python3 train_local.py --menu              # what each entry costs and can buy
    python3 train_local.py                     # the shipped default allocation
    python3 train_local.py a.json b.json       # candidates, one warm process

A full-budget run costs roughly one to two minutes on the H100 and every admissible
allocation costs about the same, because they all spend the same FLOPs. Allocations
naming the same menu entry share one built model and one `torch.compile`, so group
them into a single invocation.

`train_local.py` reports loss on `data/devset_slice.bin`. That is a proxy, not the
graded split: expect a small constant offset, and expect it to rank allocations the
same way the grader does.

## Constraints

- Do not modify the frozen files. The verifier reads its own copies of `harness.py`,
  `allocation_schema.py`, `model/nanogpt.py`, `frozen/task_spec.json` and
  `frozen/train_recipe.json`, and refuses to build unless they are byte-identical to
  the ones on this surface. That identity is what makes the FLOP arithmetic that
  bounds you here the same arithmetic that charges you at grading time.
- The graded split is not in this image and cannot be recovered from it.
- Write the allocation to `/workspace/submission/allocation.json`. `/app` is not
  shared with the verifier.
