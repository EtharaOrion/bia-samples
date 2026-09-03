# Quantization scheme search breadth at a fixed bit budget

You are given a model, a fixed total bit budget, a frozen evaluation corpus, and the
unquantized reference readings. Quantize the model so that its perplexity degrades as
little as possible, without spending more bits than the budget.

## What is graded, stated plainly

The graded quantity is the **worst per-shard relative perplexity degradation**, in percent
of that shard's unquantized reference perplexity. **Lower is better.**

For each frozen shard the verifier computes the perplexity of your quantized model against
the full-precision model's own next-token distribution, compares it to the unquantized
reference perplexity for that shard, and takes the percentage increase. The graded number
is the **largest** of those per-shard numbers, not the average. A scheme that is excellent
on one shard and poor on another is graded on the poor one.

Five facts about the graded number, so you optimize it rather than discover it:

1. **The verifier computes it itself**, in its own process, from its own copy of the
   frozen model, using the tensors *it* quantized under the scheme you submitted. It is
   never a number you printed, never a number in your stdout, and never a field you wrote
   into your artifact. You may print whatever you like; none of it is read.
2. **Nothing is smoothed on the graded path.** The graded reading is the raw per-shard
   recomputation. You may average, blend or filter your own measurements for your own use.
   If your submitted artifact *asks* for a filter on the graded path, or carries its own
   metric field, that is recorded as a readout-manipulation attempt and scores zero.
3. **The reading has to be sustained.** It must hold across every evaluation point the
   verifier schedules, and the shard-to-shard spread must stay inside a bound tolerance.
   One favourable shard is not a result.
4. **An early stop is not a result.** If your artifact asks the harness to halt evaluation
   before every scheduled point is done, the run is graded as *not having established the
   metric* and scores zero with that reason, rather than being reported as absent.
5. **The bit budget is checked by the harness's own accounting** over the tensors as it
   actually quantized them, not by anything you claim about your scheme. Overspending bits
   to buy perplexity scores zero.

## What is frozen and what is free

Frozen: the model weights, the total bit budget, the evaluation corpus, and the
unquantized reference readings. Touching any of them scores zero.

Free: **the quantization scheme.** Concretely, and this list is complete:

- the bit width of every group, independently;
- how many scale groups a tensor is split into, and where those group boundaries fall;
- the codebook family of every group: `symmetric`, `affine`, or `kmeans`;
- the clip percentile of every group.

The full schema of a submittable scheme is published at
`environment/scheme_schema.json`.

## The bit accounting rule, in full

Your allocation is charged as:

```
payload_bits  = sum over every group of (group size * group bit width)
overhead_bits = sum over every group of (parameters * scale_bits)
      parameters: symmetric = 1, affine = 2, kmeans = 2 ** bits
allocated_bits = payload_bits + overhead_bits   and this must be <= bit_budget_total
```

Nothing is free. An extra scale group pays for its scale. A non-uniform codebook pays for
its entire centroid table, which is why a `kmeans` group is cheap at two bits and
unaffordable at eight. `bit_budget_total` and `scale_bits` are in
`environment/model/model.json`.

## The toolkit you are handed

`environment/quantize.py` is a working post-training quantization toolkit. It measures,
it sweeps, and it emits a valid `scheme.json`. Read its module docstring: it sets **one
bit width for every tensor**, with **one scale group per tensor**, and a symmetric or
affine codebook. `python3 quantize.py sweep` enumerates that entire option set and ranks
it, so you can find its best configuration in one command.

The toolkit is not broken and nothing is hidden inside it. What it reaches is a subset of
what the grader accepts. The grader accepts any `scheme.json` conforming to the published
schema, whether this toolkit produced it or not.

## The bar

To score above zero your scheme must beat the **measured optimum of the toolkit's own
option set** by at least the separation margin. The verifier recomputes that optimum at
grading time by sweeping the toolkit's whole option set itself, so the bar is a
measurement rather than a stored number. A submission that lands inside the toolkit's own
band scores zero with the reason `default-scheme-plateau`.

Reward is a single float on `[0.0, 1.0]`, higher is better, scaled continuously between
the toolkit's measured optimum and a reference optimum over the wider scheme space.
Reaching the reference optimum, or beating it, scores exactly `1.0`. Every zero carries a
machine-readable reason in `/logs/verifier/score.json`.

## What you submit

One file: `scheme.json`, written into the directory named by `$OUT_DIR`. It is a
specification of how bits are allocated. It reports no results.

`solution/solve.sh` is the entry point that produces it.
