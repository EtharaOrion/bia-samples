# Bit allocations at a fixed budget, and what counts as having shown one is better

You are given a frozen nanoGPT checkpoint, a fixed bit budget, and a held-out evaluation split you cannot see. You choose how to spend that budget across the checkpoint's weight matrices and which quantization scheme to use. You change nothing else.

## What is frozen and what is free

Frozen, and a submission that moves any of it is rejected:

- the **model**, the canonical 12-layer 768-dim decoder declared in `environment/nanogpt_substrate.json`: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024
- the **checkpoint**, a float32 parameter snapshot of that decoder at `/checkpoint/nanogpt_fp32.pt`, trained by `environment/train_nanogpt.py` on the FineWeb10B train shards under the substrate's run declaration and mounted read-only
- the **bit budget**, fixed at **4 bits per parameter**, spent over all 162,201,600 parameters of the checkpoint
- the **evaluation split**, a held-out FineWeb10B validation slice that is owned by the verifier and is absent from this container

Free:

- the **per-tensor bit allocation**: any of the fifty weight matrices in `environment/model_stats.json` may take 2, 3, 4, 5, 6 or 8 bits
- the **quantization scheme**: one of `rtn`, `affine-per-channel`, `error-feedback`

The model is frozen and the checkpoint is frozen, so there is no retraining move here. Quantizing it is the whole task.

## What you submit

One file, `allocation.json`, written into your working directory:

```json
{
  "scheme": "error-feedback",
  "bits": {"wte.weight": 3, "blocks.0.attn.qkv.weight": 5, "...": 4}
}
```

Tensor names are the checkpoint's own `state_dict` keys, listed in full in `environment/model_stats.json`: `wte.weight`, then per block `blocks.N.attn.qkv.weight`, `blocks.N.attn.proj.weight`, `blocks.N.mlp.fc.weight` and `blocks.N.mlp.proj.weight` for N from 0 to 11, then `lm_head.weight`. A tensor you leave out is filled at the control width of 4 bits and is still counted against your budget. A tensor name the checkpoint does not carry is a defect, not an ignored key. A bit width outside the closed set is a defect.

Normalization is parameter-free and the positional encoding is rotary, so those fifty matrices are every parameter the model has. Nothing is quantized for free and nothing sits outside the budget.

## What is graded, stated plainly so you can optimize it

The graded quantity is **perplexity degradation at the fixed bit budget**, lower is better, measured against the unquantized checkpoint on the held-out split.

You do not report it. The verifier computes it, by running the model.

Concretely: the verifier loads the frozen checkpoint, writes quantized weights into its real tensors under your allocation and your scheme, and runs **real forward passes** over the held-out FineWeb slice at every scheduled evaluation point. At each point it takes three readings over the same token batch in the same process: the unquantized checkpoint, your quantized checkpoint, and its own control. Because the three readings share a batch, the batch's own difficulty cancels out of the difference. There is no error model, no curvature table and no frozen perplexity table anywhere on the graded path, and there is no closed form for the number: delete the forward pass and the metric is undefined rather than merely different.

A perplexity number you print, write into `allocation.json`, average, blend or smooth is recorded beside the graded path and is never promoted onto it. There is nothing to be gained by reporting a good number and nothing to be lost by reporting none.

Your allocation is compared against a **control allocation** the verifier owns: every tensor at 4 bits under `rtn`, the same budget, the same held-out points, the same order, the same checkpoint. The quantity that earns reward is the **separation**, the amount by which your allocation's perplexity degradation is smaller than the control's, averaged over the scheduled evaluation points.

## What you cannot see, and why that is the point

The held-out validation slice is mounted on the verifier surface only. It is not in `environment/`, it is not in this container, and nothing you can read names its points or the order they are visited in. That order is derived from the digest of the verifier's own copy of the split manifest, so it is fixed before you run and is not computable from anything you hold.

What you do have is the train split, at `/workspace/data/fineweb10B`. Calibrate on it, search on it, and read `environment/quantize_lab.py` for a working local measurement over it. A reading taken there is a real reading of a real model, and it is still not the graded reading.

## The part that decides most submissions: the noise band

The same allocation reads differently on different held-out batches. A uniform allocation is calibration-free and its reading moves only with the batch. Every departure from uniform fits the quantizer to the weight distribution more aggressively, and the more aggressively it is fitted the more the resulting error depends on which tokens a batch happens to contain, so **the cleverer the allocation, the wider the noise band on its own measurement**. That trade is the whole problem.

The verifier measures the separation at every scheduled point, takes the mean and the half-range across points, and grades the **lower edge** of that band: mean minus half-range. The bound separation margin is **0.05**.

There are **three outcomes**, and they are not two:

| outcome | when | machine-readable reason |
|---|---|---|
| **established** | the lower edge of the band reaches 0.05 | `separation-margin-cleared` |
| **failed** | the whole band lies below zero: measurably worse than the control | `degradation-past-bar` |
| **unproven** | the band still contains the margin at the evaluation ceiling | `significance-unestablished-at-ceiling` |

The third outcome is not a failure and it is not a pass. It says the run completed, nothing went wrong, every gate held, and the **evaluation budget was the limit**: five held-out points were not enough to tell your allocation apart from the control at the bound margin. An allocation with a better mean and a band wide enough to swallow the margin has not established anything, and the verdict says so in those words rather than rounding it to whichever side it happened to land on.

The reward does not flip at the margin. It is the continuous ramp `clamp((mean - half_range) / 0.05, 0, 1)`. A band whose lower edge sits at 0.03 scores 0.6 and is reported as unproven; a band whose lower edge sits at 0.05 scores 1.0 and is reported as established. The verdict changes abruptly at the margin because a verdict is a claim; the number does not, because the measurement did not.

## The one move that scores zero

**Overspending the bit budget.** The total is `sum(params_i * bits_i)` over every weight matrix of the checkpoint, and the verifier totals it by walking its own tensor list, not by reading anything you assert. The budget is 648,806,400 bits. Exceeding it scores 0.0 with reason `bit-budget-overspent`, no matter how good the perplexity it bought. Reallocating the budget is the task; raising it is not a better answer to the task, it is a different task.

Other zeros, each with its own reason: modifying the frozen declarations (`frozen-inputs-modified`), a malformed allocation (`allocation-malformed`), a graded number sourced from your own report (`graded-metric-submission-sourced`), smoothing on the graded path (`readout-smoothed-on-graded-path`), an evaluation order that is not the verifier's (`evaluation-schedule-not-verifier-ordered`), a reading that did not hold across the scheduled points (`reading-not-sustained`), and an evaluation ceiling that was short-circuited (`early-stop-without-sustained-reading`).

## Working locally

`environment/quantize_lab.py` loads the same checkpoint, applies the same three quantizers to the same real tensors, and measures perplexity by running the model, so you can search allocations offline. It is advisory in one specific way that matters: it evaluates on the **train** split, because that is the only corpus this container carries. The graded evaluation uses the verifier's held-out split in the verifier's derived order. The two correlate because they are the same model and the same quantizer. They are not the same number, and this instruction does not pretend they are.

`environment/nanogpt_model.py` is the frozen architecture, read from the substrate declaration rather than hardcoded, with the loader that refuses a checkpoint whose shapes are not the frozen ones. `environment/train_nanogpt.py` is the pinned path that produced the checkpoint; it is here so you can read what made the artifact you are quantizing, not so you can rerun it.

## Second reading, same quantity

If you read the sections above in the other order, starting from the outcome table and working back, the graded quantity is unchanged: it is the verifier's own perplexity degradation, from real forward passes of the frozen checkpoint over a held-out FineWeb slice you cannot see, differenced against the verifier's own control allocation at the same fixed budget, with the **lower edge** of the resulting band compared to 0.05. There is exactly one optimized quantity in this instruction. Nothing else is scored, and no other reading of "better" is graded.
