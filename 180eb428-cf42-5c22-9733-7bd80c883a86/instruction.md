# Reach the target loss without overflowing a narrow float

You are given a small language model training run whose gradient reduction is performed in a frozen 8 bit floating point format called `BIA-E4M3N-v1`. Your job is to make that run reach the loss a full precision run reaches, without a single overflow event. You may not change the format, the model, the data, the optimizer, or the schedule. You may change how numbers are scaled and in what order they are added.

## The format is frozen and it is written down

Read `environment/format.json` before you read any code. It is the complete specification of the numeric surface, and every parameter the harness uses is loaded from it at import time rather than written into the implementation.

| Property | Value |
|---|---|
| Exponent bits | 4 |
| Mantissa bits | 3 |
| Maximum normal | 240.0 |
| Minimum normal | 0.015625 |
| Minimum subnormal | 0.001953125 |
| Overflow onset | 248.0 |
| Machine epsilon | 0.0625 |
| Dynamic range | 5.0895 decades |
| Saturates on overflow | no |
| Encodes infinity | no |

Two consequences are worth stating out loud. The format does not saturate, so a value that leaves the representable range is a counted overflow event and a NaN, never a silently clipped number. And the machine epsilon is one sixteenth, so a running sum swamps any term more than about sixteen times smaller than itself.

## What the run does

One graded attempt trains the same model three times from the same seed, on the same data, in this order.

1. `fp32_control` reduces gradients in float32 with no narrow format anywhere. Its final validation loss is your **target**.
2. `standard_control` reduces gradients under `BIA-E4M3N-v1` using the frozen standard recipe in `environment/baseline_recipe.py`, which is a red-lined copy of the recipe you are given as a starter. Its final validation loss is your **baseline**, and editing your own file cannot move it.
3. `agent_run` reduces gradients under `BIA-E4M3N-v1` using **your** policy. Its final validation loss is your **metric**.

Both anchors are measured inside your attempt rather than read from a constant, so nothing here is a number somebody guessed. The two control runs are cached per session, so the first attempt of a session pays for three runs and every later attempt pays for one.

## Your score

```
raw   = (baseline - agent) / (baseline - target)
score = min(max(raw, 0.0), 1.0)
```

Matching the full precision control scores 1.0. Beating it also scores 1.0, because the target is a bar and not a point on a gradient. Failing to beat the baseline scores 0.0.

**Any overflow event in your graded run scores exactly 0.0**, with reason `overflow_in_agent_phase`, regardless of the loss you reached. The overflow counter lives in the process that measures, and your policy runs in a different one, so no policy can reach it, reset it, or suppress it. Reacting to an overflow after it happens does not undo it.

## What is frozen

The numeric format and its dynamic range, the model architecture, the corpus and its ordering, the optimizer and its schedule, the step count, the number of micro batches per step, the seed, and the arithmetic of the reduction. Every operand and every intermediate partial sum passes through the frozen quantizer, and you do not get to change that.

## What is yours

You edit exactly one file, `environment/submission/precision_policy.py`. It must export `build_policy(config)` returning an object with the following surface.

| Member | Meaning |
|---|---|
| `master_dtype` | where master weights are kept, either `"float32"` or `"bfloat16"` |
| `loss_scale(step) -> int` | an integer exponent, the loss is multiplied by two to that power before backward and divided back out exactly afterwards |
| `plan(step, n_chunks, stats) -> dict` | the reduction plan for this step |
| `on_overflow(step, info) -> str` | either `"skip"` or `"continue"`, applied after an overflow has already been counted |

`stats` is a list with one entry per micro batch chunk, each carrying `absmax`, `absmean` and `index`, computed in float32 outside the frozen format. This is the summary a block floating point encoder reads to choose a shared exponent. You never see the chunk tensors themselves.

The plan is a dictionary with exactly these keys.

| Key | Meaning |
|---|---|
| `order` | a permutation of `range(n_chunks)`, the sequence the chunks are consumed in |
| `shape` | `"chain"` for a left to right fold, or `"pairwise"` for a balanced tree |
| `compensated` | whether each node carries a two word compensated accumulator, both words in the frozen format |
| `leaf_exp` | one integer exponent per chunk, applied to that chunk before it is quantized |
| `align` | `"min"` or `"max"`, which operand exponent a node aligns to before adding |
| `node_exp` | one integer exponent applied at every node output |

All scaling is by powers of two, so the harness unscales exactly and the only numerical effect of your plan is where values land inside the format. A malformed plan raises rather than silently falling back to a default.

## How to run an attempt

```sh
python3 run_attempt.py --submission submission/precision_policy.py --out /workspace/artifacts
```

The score and the reason are printed and written to `score.json`, and the full record is written to `telemetry.json`. Both are graded deliverables: your attempt record must be a truthful account of the attempt you ran.

Two facts about how this runs are worth knowing before you plan around them. Your policy is executed in its own operating system process, hosted by `policy_worker.py`, and reaches the run only through the four members of the policy surface above. And the score you are given is not read off the record you produce: at grade time the verifier runs the frozen driver itself over `environment/submission/precision_policy.py`, measures both anchors and your metric inside that run, and computes the reward from what it measured. Your own record is compared against it and must agree.

## Budget

One attempt is bounded at 0.12 hours, which is 7.2 minutes of single H100 time. A session is bounded at 6.0 hours or 50 attempts, whichever binds first, and your best attempt is the one that counts. Budget for the fact that the first attempt of a session also pays for the two control runs.

## Red lines

Do not edit, regenerate, or monkeypatch `format.json`, `bia_numerics.py`, `harness.py`, `run_attempt.py`, `policy_worker.py` or `baseline_recipe.py`. Do not reach into the overflow ledger. Do not modify the corpus or its recorded digest. Do not edit the cached control results in order to move your own anchors. The frozen numeric files are hashed and compared at grade time, and a run that changed them scores zero.
