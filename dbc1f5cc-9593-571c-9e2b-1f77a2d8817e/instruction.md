# Shape a fixed token budget: batch geometry on a frozen nanoGPT

You are given a frozen nanoGPT, a frozen optimizer, a frozen learning-rate envelope,
a frozen token budget and a batch geometry that is not very good. Find a better one.

## The setup

`/app` holds the whole task.

| path | what it is |
|---|---|
| `frozen/task_spec.json` | the frozen substrate: architecture, init seed, optimizer, schedule, geometry grid, budget, evaluation window |
| `model/nanogpt.py` | the frozen decoder — 6 layers, model_dim 384, head_dim 64, vocab 50304, rotary positions |
| `harness.py` | the training and evaluation harness |
| `packing_schema.py` | the plan schema and its bounds |
| `default_packing.json` | the plan you have to beat |
| `train_local.py` | measure a plan end to end |
| `data/train_slice.bin` | the training corpus |
| `data/devset_slice.bin` | a proxy validation split, cut from a different FineWeb shard |
| `submission_protocol.md` | the schema field by field, and the reward formula |

The compute budget is **8,388,608 training tokens**, consumed in order from position
0 of the same stream by every plan. It is fixed in tokens, not in optimizer steps and
not in wall clock. Every plan therefore costs the same number of token-forward and
token-backward passes; no plan wins by training on more.

## What you submit

A plan, as JSON, at:

    /workspace/submission/packing.json

A plan is a list of up to six **phases** that partition the budget in order. Each
phase names a micro-batch geometry — `rows` sequences of `seq_len` tokens — and a
`grad_accum`, the number of those micro-batches that accumulate into one optimizer
step. That is the whole free variable.

`submission_protocol.md` has the exact schema and the bound on every field. Unknown
keys are refused rather than ignored.

## What the choice actually costs you

The same 8.4M tokens can be fed as 2048 optimizer steps of 8192 tokens or as 512
optimizer steps of 32768 tokens, and as rows of 128 tokens or rows of 1024. Three
things move when you change the shape, and they do not move together.

* **How many optimizer steps the budget buys.** Fewer, larger steps mean a less noisy
  gradient and less total parameter movement. The frozen schedule already applies the
  square-root batch rule, so the larger step gets a proportionally larger learning
  rate; what the rule does *not* give back is the number of updates.
* **How much left context each predicted token has.** A token at position 3 of a
  128-token row is predicted from three tokens of history. The same token inside a
  1024-token row may have hundreds. Short rows spend a larger share of the budget on
  positions that have almost nothing to condition on.
* **What the row length costs.** Attention is quadratic in the row length, so at a
  fixed token count a longer row buys its context at a real price in compute per
  token — and the budget is counted in tokens, not in FLOPs.

The evaluation is at a fixed `16 x 512` geometry for every run, so a plan cannot move
the measurement to suit itself.

## How you are graded

The verifier trains **three** models on the grading run, all from the same frozen
initialisation, over the same tokens, under the same frozen optimizer and schedule:
the shipped default, a stronger reference plan it holds privately, and yours. It
evaluates all three on a held-out FineWeb slice that is not in this image, and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints are **measured on the grading run**, by real from-scratch training
runs. No number in this bundle is either of them. Tying the default scores 0.0;
reaching or beating the reference scores 1.0; everything between is the fraction of
the gap closed.

## Working method

`train_local.py` runs the same harness the verifier runs. Measure candidates against
`data/devset_slice.bin` — it will sit at an offset from the graded split but rank
plans the same way. Several plans in one invocation share the model build and the
corpus read, so batch your comparisons.

One run of the budget is a real training run; plan your attempts accordingly.
