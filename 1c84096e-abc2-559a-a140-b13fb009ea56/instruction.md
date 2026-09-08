# Spend a fixed evaluation budget to keep the right checkpoint

A training run is going to be performed for you. You do not get to change it. What
you write is the **evaluation policy** that decides which of its checkpoints is kept.

## Why that is a real decision here

The data pipeline feeding this run degrades. From some micro-step onward an
increasing fraction of the token stream has been permuted inside 256-token windows —
real FineWeb text, same unigram statistics, no local structure. The run keeps
descending on what it is being fed while drifting off the distribution the graded
split is drawn from, so **the last checkpoint is not the best checkpoint**, and the
shipped default policy — evaluate once at the end, keep that — keeps a bad one.

`frozen/task_spec.json` declares the fault, its shape, and the five ramp starts it
may begin at. **The verifier draws one of the five on the grading run and you are not
told which.** The draw moves the turn in the held-out curve by hundreds of steps, so
a policy tuned to one draw is one-in-five to be right. `harness.degrade()` is on this
surface: you can reproduce any of the five yourself.

## The setup

`/app` holds the whole task.

| path | what it is |
|---|---|
| `frozen/task_spec.json` | the frozen substrate: architecture, init seed, budget, the candidate grid, the declared fault and its choice set |
| `frozen/train_recipe.json` | the frozen training recipe — published so you can reproduce the run, not a free variable |
| `model/nanogpt.py` | the frozen decoder — 6 layers, model_dim 384, head_dim 64, seq_len 512, vocab 50304 |
| `harness.py` | the training, degradation and evaluation harness |
| `policy_schema.py` | the policy schema and its budget |
| `default_policy.json` | the policy you have to beat |
| `probe_local.py` | reproduce the run and read the curve |
| `data/train_stream.bin` | the clean training stream — the fault is applied at run time |
| `data/devset_slice.bin` | a proxy validation split, cut from a different FineWeb shard |
| `submission_protocol.md` | the schema field by field, and the reward formula |

## What you submit

A policy, as JSON, at `/workspace/submission/policy.json`: up to **6** probes on the
candidate grid, at most **98304** evaluation tokens across all of them, and a
selection rule. `submission_protocol.md` has the exact schema and every bound.

## How you are graded

The verifier performs **one** frozen training run under a degradation it draws, and
records the per-batch probe losses and the held-out loss at all 31 candidate steps.
Then it replays three policies over those same recorded numbers — the shipped
default, a stronger reference policy it holds privately, and yours — and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

where each loss is the **held-out** loss at the checkpoint that policy selected. Both
ends of the scale are measured on your run. Neither is a stored constant. Tying the
default scores 0.0, reaching the reference scores 1.0, and beating it also scores
1.0.

Because all three policies read the same recorded curves from the same drawn run, no
policy is advantaged by the draw, and nothing your policy costs the verifier depends
on what it chose.

## What a probe actually buys you

The probe readings come from the verifier's own validation stream, which is **not on
this surface**. Your policy is a procedure executed on a run it has not seen: you
choose where to look and how much to spend, and the numbers that come back are draws
you cannot precompute. Per-batch spread is about 0.18 nats; differences between
neighbouring good checkpoints are a few hundredths. Six cheap probes cover more of
the grid and read each point badly; two expensive probes read precisely and may
bracket the turn on the wrong side. `argmin_smoothed` trades a narrow true minimum
for resistance to one unlucky reading.

## Measuring your own work

    python3 probe_local.py                        # every declared ramp start
    python3 probe_local.py --ramp-start 1024      # one draw
    python3 probe_local.py --policy mine.json     # replay yours against each draw
    python3 probe_local.py --repeat 3             # how much the curve moves per seed

One run costs roughly a minute on the H100. `probe_local.py` reads
`data/devset_slice.bin`, which is a proxy: expect an offset from the graded split,
and expect the shape to hold. Budget your time so the policy you submit is one you
have actually replayed against **all five** draws.

## Constraints

- Do not modify the frozen files. The verifier reads its own copies of `harness.py`,
  `policy_schema.py`, `model/nanogpt.py`, `frozen/task_spec.json`,
  `frozen/train_recipe.json` and `default_policy.json`, and refuses to build unless
  they are byte-identical to the ones on this surface.
- The probe stream and the graded split are not in this image and cannot be
  recovered from it.
- Write the policy to `/workspace/submission/policy.json`. `/app` is not shared with
  the verifier.
