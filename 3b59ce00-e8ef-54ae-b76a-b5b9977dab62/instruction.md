# Regularise a frozen nanoGPT: decay, smoothing and the output distribution

You are given a frozen nanoGPT, a frozen and already-tuned optimizer, a frozen compute
budget, and a regularisation policy that is a collection of reasonable defaults nobody
measured. Find a better one.

## The setup

`/app` holds the whole task.

| path | what it is |
|---|---|
| `frozen/task_spec.json` | the frozen substrate: architecture, init seed, compute budget, the whole optimizer and schedule, the evaluation window |
| `model/nanogpt.py` | the frozen decoder — 6 layers, model_dim 384, head_dim 64, seq_len 512, vocab 50304 |
| `harness.py` | the training and evaluation harness |
| `policy_schema.py` | the policy schema and its bounds |
| `default_policy.json` | the policy you have to beat |
| `train_local.py` | measure a policy end to end |
| `data/train_slice.bin` | the training corpus — exactly the tokens the budget consumes |
| `data/devset_slice.bin` | a proxy validation split, cut from a different FineWeb shard |
| `submission_protocol.md` | the schema field by field, and the reward formula |

The compute budget is **3072 micro-batches of 16x512 tokens**, 25,165,824 tokens, and
it is fixed in forward and backward passes. Nothing a policy says changes it.

## What you submit

Seven numbers, as JSON, at:

    /workspace/submission/policy.json

Four decoupled weight decays — one for the token embedding, one for the block
matrices, one for the output projection, one for the RMSNorm gains and biases — and
three controls on the objective: `label_smoothing`, `z_loss` and `logit_softcap`.

The optimizer and the schedule are **not** yours. They are frozen, they are already
tuned, and you can read every one of their constants in `frozen/task_spec.json`. What
is left is how the objective is regularised and how the 50,304-way output distribution
is shaped.

## What the choice actually costs you

* **Weight decay is not one number.** The token embedding, the block matrices, the
  output projection and the RMSNorm gains are four different kinds of parameter with
  four different roles in the loss, and a single decay applied to all of them is a
  choice, not a neutral default. At a budget where the model is nowhere near fitting
  the corpus, decay mostly removes signal — but "mostly" is not "everywhere", and the
  four roles do not agree.
* **Label smoothing is in the training loss and not in the reading.** It is a
  well-known regulariser and it directly and mechanically raises the cross entropy the
  grade is taken on. Whether the regularisation it buys ever pays that back is a
  question you have to measure rather than reason about.
* **The log-partition penalty is nearly free and nearly useless — or neither.**
  `z_loss` steadies the scale of the logits without touching their ordering. Small
  values are a stabiliser; large ones crush the distribution flat.
* **The softcap is part of the model.** It bounds how confident the decoder can ever
  be, so it appears in the reading as well as in training. Too tight and no prediction
  can be sharp; too loose and there is no cap at all. The shipped 15.0 is the upstream
  record set's constant, transcribed and never tuned for this budget.

## How you are graded

The verifier trains **three** models on the grading run, all from the same frozen
initialisation, over the same tokens, with the same compiled kernels: the shipped
default policy, a stronger reference policy it holds privately, and yours. It
evaluates all three on a held-out FineWeb slice that is not in this image, and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints are **measured on the grading run** by real from-scratch training runs.
No number in this bundle is either of them. Tying the default scores 0.0; reaching or
beating the reference scores 1.0; everything between is the fraction of the gap closed.

## Working method

`train_local.py` runs the same harness the verifier runs. Measure candidates against
`data/devset_slice.bin` — it will sit at an offset from the graded split but rank
policies the same way. Passing several policies in one invocation shares the model
build and the single torch.compile, so batch your comparisons.
