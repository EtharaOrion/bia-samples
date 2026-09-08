# Substitute the kernels

A 49.3M-parameter nanoGPT runs one forward and one backward pass over 16x512 tokens.
The **mathematics is frozen**: embedding lookup, six pre-norm blocks of rotary causal
self attention and a squared-ReLU MLP, a final RMSNorm, the untied output projection,
the upstream logit softcap and the mean token-level cross-entropy. The parameters are
frozen. The batch shape is frozen.

What you choose is **how that computation is executed**:

```
/workspace/submission/plan.json
```

```json
{
  "schema": "oer-kernel-plan/v1",
  "notes": "free text, never parsed",
  "attention": "math | efficient | flash",
  "loss_chunks": 1 | 2 | 4 | 8 | 16 | 32,
  "logit_dtype": "fp32 | bf16",
  "rotary_dtype": "fp32 | bf16",
  "qk_norm_dtype": "fp32 | bf16"
}
```

## The hard gate comes first

Before anything is timed, your plan is run forward and backward on the **same frozen
weights** over the same fixed batches as the reference plan, and must agree with it:

* on the loss, to within **0.002** nats absolute, and
* on the global gradient norm, to within **0.02** relative.

A plan that does not agree is refused with `numerical-output-diverged` at exactly
0.0, **however fast it was**. The gate is never a soft penalty and it never scales the
reward. Speed bought by computing a different function is not speed.

Every option above is offered because it is a plausible implementation of the frozen
computation. Not all of them are one. Finding out which is part of the task, and
`bench_local.py` runs the identical check with the identical tolerances.

## The reward

```
reward = clamp( (control_ms - your_ms) / (control_ms - reference_ms), 0, 1 )
```

* `control_ms` — the shipped `default_plan.json`, re-timed on the grading run. This is
  also the **reference plan** that equivalence is checked against: the canonical
  textbook execution, one unchunked cross-entropy, everything in float32. Correctness
  is public here; only the speed bar is held back.
* `reference_ms` — a plan held only inside the verifier image. Reach it and you score
  1.0; beat it and you also score 1.0.

Neither endpoint is a stored number. Both are timed on every grading run.

## How the timing is done, and why you can trust it

The three plans are timed **round-robin** — one round of each, then the next round —
for 7 rounds of 24 steps after 8 warmup steps, using CUDA events. Each plan's reported
latency is the **minimum over its rounds**.

Within a round a busy machine is busy for everybody, and the reward is a ratio of
*differences* of latencies, so a slowdown that multiplies all three equally cancels
out of it exactly. The minimum discards the rounds where the machine was busiest.

## Tools in this image

```bash
python3 bench_local.py --plan my_plan.json
python3 bench_local.py --sweep
```

Both run the equivalence check first and refuse to time a plan that fails it, exactly
as the verifier does. Timing is on `data/devset_slice.bin`; the verifier times on its
own held-out slice, which changes which tokens are multiplied and not what the
multiply costs.

`harness.py`, `kernel_plan.py`, `model/nanogpt.py`, `frozen/task_spec.json` and
`default_plan.json` are byte-identical inside the verifier; its image refuses to build
if they are not.

## Deliverable

`/workspace/submission/plan.json`. `/workspace` is the only path shared with the
verifier — a file left in `/app` is never graded.
