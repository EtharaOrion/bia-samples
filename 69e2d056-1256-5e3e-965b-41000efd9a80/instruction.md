# Quantization scheme search breadth at a fixed bit budget

You are given a trained nanoGPT checkpoint, a fixed total bit budget, an architecture declaration you may not move, and a calibration split of FineWeb tokens you can measure on. Quantize the checkpoint so that its perplexity degrades as little as possible, without spending more bits than the budget.

## The model, stated plainly

The checkpoint is the canonical modded-nanogpt decoder: vocabulary 50304, twelve layers, model dimension 768, head dimension 128, six heads, sequence length 1024. The declaration is at `environment/nanogpt_substrate.json` and the architecture module that builds it is at `environment/model/nanogpt.py`. The parameters are at `environment/model/checkpoint.pt` and the parameter shard table over them is at `environment/model/checkpoint.json`.

A shard is one two-dimensional weight matrix of that decoder. There are 74 of them: the token embedding, the untied output projection, and six matrices in each of the twelve blocks, being the query, key, value and output projections of the attention and the two MLP matrices. Together they hold 162201600 parameters. Biases and RMSNorm gains are not shards: they are 153216 parameters, they stay at the checkpoint's own width, and they are outside the bit budget entirely.

## What is graded, stated plainly

The graded quantity is the **worst per-slice relative perplexity degradation**, in percent of that slice's unquantized reference perplexity. **Lower is better.**

The verifier holds five disjoint held-out slices of the FineWeb validation split. They are not in this container, they are not named in anything you can read, and nothing you run can reach them. For each of those slices the verifier runs two real forward passes of the frozen decoder, one with the full-precision parameters and one with the parameters as it quantized them under your scheme. The reference reading is the exponentiated mean entropy of the full-precision next-token distribution on that slice and the quantized reading is the exponentiated mean cross entropy of that distribution under the quantized one. The degradation is the percentage increase of the second over the first, and the graded number is the **largest** of those five, not the average. A scheme that is excellent on one slice and poor on another is graded on the poor one.

By Gibbs' inequality the quantized reading is at or above the reference reading, with equality only when the two distributions coincide, so there is no quantization noise that buys a better-than-reference number and a negative degradation is refused rather than rewarded.

Six facts about the graded number, so you optimize it rather than discover it:

1. **The verifier computes it itself**, in its own process, from its own copy of the frozen checkpoint, on its own held-out slices, using the parameters *it* quantized under the scheme you submitted. It is never a number you printed, never a number in your stdout, and never a field you wrote into your artifact. You may print whatever you like; none of it is read.
2. **You cannot see the graded split.** The calibration slice in this container is a different window of a different FineWeb shard. It is a proxy and it is a good one, but an allocation fitted to it rather than derived from the model will not carry, and there is no path by which you could check.
3. **Nothing is smoothed on the graded path.** The graded reading is the raw per-slice recomputation. You may average, blend or filter your own measurements for your own use. If your submitted artifact *asks* for a filter on the graded path, or carries its own metric field, that is recorded as a readout-manipulation attempt and scores zero.
4. **The reading has to be sustained.** It must hold across every held-out slice the verifier schedules, and the slice-to-slice spread must stay inside a tolerance the verifier measures in the same run. One favourable slice is not a result.
5. **An early stop is not a result.** If your artifact asks the harness to halt evaluation before every scheduled slice is done, the run is graded as *not having established the metric* and scores zero with that reason, rather than being reported as absent.
6. **The bit budget is checked by the harness's own accounting** over the shards as it actually quantized them, not by anything you claim about your scheme. Overspending bits to buy perplexity scores zero.

## What is frozen and what is free

Frozen: the checkpoint parameters, the architecture, the total bit budget, the held-out evaluation split, and the unquantized reference readings. Touching any of them scores zero, and the three that matter are held by the verifier where you cannot touch them at all.

Free: **the quantization scheme.** Concretely, and this list is complete:

- the bit width of every group, independently;
- how many scale groups a shard is split into, and where those group boundaries fall;
- the codebook family of every group: `symmetric`, `affine`, or `kmeans`;
- the clip percentile of every group.

The full schema of a submittable scheme is published at `environment/scheme_schema.json`.

## The bit accounting rule, in full

Your allocation is charged as:

```
payload_bits  = sum over every group of (group size * group bit width)
overhead_bits = sum over every group of (parameters * scale_bits)
      parameters: symmetric = 1, affine = 2, kmeans = 2 ** bits
allocated_bits = payload_bits + overhead_bits   and this must be <= bit_budget_total
```

Nothing is free. An extra scale group pays for its scale. A non-uniform codebook pays for its entire centroid table, which is why a `kmeans` group is cheap at two bits and unaffordable at eight. `bit_budget_total` and `scale_bits` are in `environment/model/checkpoint.json`, and the budget is a **total over the whole checkpoint** rather than a per-shard allowance.

The shipped configuration lands exactly on that total: 162201600 parameters at four bits is 648806400 payload bits, plus one sixteen-bit scale for each of the 74 shards is 1184 overhead bits, which is the budget to the bit. Note what follows from that. Any allocation that keeps four bits everywhere and adds a single scale group anywhere is already over budget, so every widening has to pay for itself out of somewhere.

## The toolkit you are handed

`environment/quantize.py` is a working post-training quantization toolkit. It loads the real checkpoint, it quantizes, it runs real forward passes on the calibration split, it sweeps, and it emits a valid `scheme.json`. Read its module docstring: it sets **one bit width for every shard**, with **one scale group per shard**, and a symmetric or affine codebook. `python3 quantize.py sweep` enumerates that entire option set and ranks it, so you can find its best configuration in one command. `python3 quantize.py shards` prints the shard table if you would rather read it than parse the manifest.

The toolkit is not broken and nothing is hidden inside it. What it reaches is a subset of what the grader accepts. The grader accepts any `scheme.json` conforming to the published schema, whether this toolkit produced it or not.

## The bar

To score above zero your scheme must beat the **measured optimum of the toolkit's own option set** by at least the separation margin. The verifier recomputes that optimum at grading time by sweeping the toolkit's whole option set itself, on its own held-out slices, so the bar is a measurement rather than a stored number. A submission that lands inside the toolkit's own band scores zero with the reason `default-scheme-plateau`.

Reward is a single float on `[0.0, 1.0]`, higher is better, scaled continuously between the toolkit's measured optimum and a widening probe the verifier also measures in the same run. Reaching the probe, or beating it, scores exactly `1.0`. Every zero carries a machine-readable reason in `/logs/verifier/score.json`.

## What you submit

One file: `scheme.json`, written into the directory named by `$OUT_DIR`. It is a specification of how bits are allocated over the 74 named shards. It names every shard exactly once, the group sizes inside each shard sum to that shard's `numel`, and it reports no results.

`solution/solve.sh` is the entry point that produces it.
