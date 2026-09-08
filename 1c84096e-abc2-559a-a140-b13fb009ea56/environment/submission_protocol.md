# Submission protocol

## What you submit

One JSON file:

    /workspace/submission/policy.json

`/workspace` is the volume shared with the verifier. `/app` is **not** shared — a
policy left in `/app` is a policy that is never graded. Create the directory if it
does not exist. `/workspace/policy.json` is also accepted, as a fallback.

## The schema

`policy_schema.py` is the *only* definition of what is accepted, it ships on this
surface, and the verifier runs a byte-identical copy of it. If
`python3 -c "import policy_schema; policy_schema.load('my.json')"` returns without
raising, the verifier will replay your policy.

```json
{
  "schema": "oer12-eval-policy/v1",
  "notes": "free text, never parsed, copied into the score document",
  "probes": [
    {"step": 1024, "tokens": 16384},
    {"step": 1536, "tokens": 16384}
  ],
  "selection": {"rule": "argmin_smoothed", "window": 3}
}
```

| field | meaning | bound |
|---|---|---|
| `probes[].step` | a checkpoint to evaluate | on the candidate grid 128..2048 stride 64 |
| `probes[].tokens` | evaluation tokens spent there | positive multiple of 8192, at most 131072 |
| `selection.rule` | how the reading picks a checkpoint | `argmin` or `argmin_smoothed` |
| `selection.window` | smoothing width, probes sorted by step | 1, 3 or 5 — required for `argmin_smoothed` |

and across the whole policy:

- at most **6** probes, each at a distinct step;
- at most **98304** probe tokens in total — that is **12 batches of 8192**;
- unknown keys are refused rather than ignored.

## What a probe buys

The harness records 16 per-batch losses on the verifier's probe stream at every
candidate step. A probe of `tokens` at step `s` is answered with the mean of the
**first `tokens / 8192`** of the batch losses recorded at `s`. Fewer tokens is a
noisier reading of the same underlying curve. Per-batch spread on this substrate is
around 0.18 nats, so a one-batch reading is worth about ±0.18 and a four-batch
reading about ±0.09, against curve differences between neighbouring good checkpoints
of a few hundredths. That ratio is the whole trade: coverage against resolution.

## How selection works

`argmin` takes the probed step with the lowest reading. `argmin_smoothed` sorts your
probes by step, replaces each reading with the mean of itself and its `window // 2`
neighbours on each side **among your own probes**, and takes the argmin of that.
Smoothing costs you nothing in tokens and buys robustness against a single unlucky
draw, at the price of blurring a genuine narrow minimum.

## The reward

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Each `*_loss` is the **held-out** cross-entropy at the checkpoint that policy
selected, measured by the verifier on a split that is not on this surface. The
default anchor is `default_policy.json`, which spends everything on the final
checkpoint. The reference anchor is a stronger policy held only by the verifier.
Both are replayed on your run, over the same drawn degradation and the same recorded
curves as your policy. Neither is a stored number.

Nothing your policy says about itself is graded. `notes` is free text that is copied
into the score document and never parsed.
