# Submission protocol

## What you submit

One JSON file:

    /workspace/submission/allocation.json

`/workspace` is the volume shared with the verifier. `/app` is **not** shared — an
allocation left in `/app` is one that is never graded. Create the directory if it
does not exist. `/workspace/allocation.json` is also accepted, as a fallback.

## The schema

`allocation_schema.py` is the *only* definition of what is accepted, it ships on this
surface, and the verifier runs a byte-identical copy of it. If

    python3 -c "import harness, allocation_schema; \
        allocation_schema.load('mine.json', harness.load_spec(), harness)"

returns without raising, the verifier will train your allocation.

```json
{
  "schema": "oer16-compute-allocation/v1",
  "notes": "free text, never parsed, copied into the score document",
  "model": "m04x256",
  "micro_steps": 2376,
  "grad_accum": 1
}
```

| field | meaning | bound |
|---|---|---|
| `model` | which menu entry to train | one of the seven in `frozen/task_spec.json` |
| `micro_steps` | how many 16x512 micro-batches to run | at least 64, and `micro_steps * flops_per_micro_batch(model)` at most the FLOP budget |
| `grad_accum` | micro-batches per optimizer step | 1, 2, 4, 8, 16 or 32, and must divide `micro_steps` |

Unknown keys are refused rather than ignored.

## The budget, and how it is counted

The FLOP budget is **2e15**, and it is counted by declared arithmetic rather than by
a profiler:

    macs_per_token  = num_layers * (4*d*d + 8*d*d + 2*seq_len*d) + d*vocab_size
    flops_per_token = 2 * macs_per_token * 3        # backward charged at 2x forward
    flops_per_micro_batch = flops_per_token * 16 * 512

This is the same formula that **bounds** your submission on this surface and
**charges** it at grading time — `harness.micro_batch_flops` is byte-identical on
both, and `tests/Dockerfile` fails to build if it is not. `python3 train_local.py
--menu` prints what each entry costs and the most it can buy.

## The corpus is finite

`data/train_slice.bin` holds 19,521,536 tokens. At the small end of the menu the
budget buys more than that: `m02x128` can afford 47M tokens, or 2.4 passes. An
allocation that asks for more than the corpus holds is **not refused** — the stream
cycles back to the start. Repetition is an allocation you may choose. What it costs
is that the repeated tokens teach less than fresh ones would have, and that is the
data-constrained half of the trade this task is about.

## The reward

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Each `*_loss` is the validation cross-entropy of a model trained from the frozen
initialisation under that allocation, measured by the verifier on a held-out FineWeb
slice that is not on this surface. The default anchor is
`default_allocation.json`. The reference anchor is a stronger allocation held only by
the verifier. Both are retrained from scratch on your grading run. Neither is a
stored number.

Nothing your allocation says about itself is graded. `notes` is free text that is
copied into the score document and never parsed.
