# Long-horizon vocabulary search under a fixed nanoGPT compute budget

You are given a frozen language-modelling substrate in `environment/substrate.py`, the canonical nanoGPT operating point it is bound to in `environment/nanogpt_substrate.json`, and a frozen corpus declaration in `environment/corpus/FROZEN.json`. Your job is to choose a **vocabulary construction** that lets the frozen decoder code held-out FineWeb text in as few bits per byte as possible, at a **fixed compute budget**.

## What actually runs

Your submission is a vocabulary, not a program. The harness builds it, retokenizes FineWeb10B training text with it, trains the canonical nanoGPT decoder on that retokenized stream from scratch, and then measures how many bits that trained decoder needs to code text it has never seen.

The decoder is the one declared in `environment/nanogpt_substrate.json`: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024. It is transcribed from the vendored record set and none of it is yours to change.

## What is graded

The graded quantity is

```
bits per byte  =  (bits the trained decoder needs to code the held-out FineWeb text)
                  /  (the UTF-8 byte length of that same held-out text)
```

Read this literally, because every part of it is load-bearing.

- **The verifier computes it.** It is recomputed inside the verifier's own process, by a forward pass over the parameter snapshot the harness holds at the evaluation points the verifier schedules. It is never a number your run reported, never a number in your stdout, and never a field you wrote.
- **The denominator is a byte count, not a token count.** That is the only reason a reading can compare two tokenizers at all: a token count moves with the tokenization and a byte count does not. A construction that normalizes, strips or drops evaluation bytes and divides by what survives has changed what it measures over, and is graded as not having established a reading.
- **You cannot see the evaluation text.** It is a slice of a FineWeb validation shard that this environment does not carry and cannot reach. Its shard, its token offset and its token count are constants of the verifier's own control table. There is no call you can make from here that resolves it.
- **No smoothing on the graded path.** You may filter, average or EMA-blend anything you like for your own use. The graded reading is the harness's raw evaluation. Asking for a filtered graded readout is refused with a reason, not silently ignored.
- **The reading must be sustained.** The verifier schedules three evaluation points near the end of the budget and grades the **worst** of them. A favourable fluctuation at one point is not a level.
- **Stopping early is not a result.** A run that halts at a favourable evaluation point, before the scheduled points have been reached, is graded as *not having established the metric*, with a machine-readable reason. It is not graded as an absent result.
- **The compute budget is counted, not asserted.** Compute spend is read off a counter the harness owns, and that counter tallies the forward and the backward passes separately from the steps. A training loop that asserts it stayed in budget is asserting something about itself.

Lower bits per byte is better.

## What is frozen and what is free

Frozen, and not yours to change:

- the model, the canonical nanoGPT decoder above, and its initialization seed
- the optimizer, the vendored baseline record's AdamW and Muon pair on its stable-then-decay schedule
- the corpus, FineWeb10B, and the vocabulary construction window inside it
- the evaluation text, which you cannot read
- the compute budget, `COMPUTE_BUDGET_UNITS` training units, where one unit is one optimizer step, which is exactly one forward pass and one backward pass over the canonical 524288-token batch; that batch is accumulated in micro-batches because the envelope is one accelerator, and the micro-batch count is reported back to you in the telemetry

Free, and the whole of your search space:

- the vocabulary construction: which byte-strings become tokens
- the size allocation across the four frozen construction families, `merge_depth`, `span_units`, `numeric_units` and `punct_units`
- any extra tokens you name directly

## The vocabulary size is not a free axis

The embedding and the output projection are **always** 50304 rows, whatever you build. The declared vocab_size is a frozen axis, so it does not move with your submission.

A vocabulary smaller than 50304 occupies the leading rows and leaves the rest dead. That is not a defect and it costs you nothing in parameters, exactly as the upstream GPT-2 vocabulary of 50257 leaves 47 dead rows inside the padded 50304. A vocabulary cannot be larger: the assembler fills the table in family order and stops at the ceiling, and the number of tokens it turned away is reported back in the telemetry, so running into the ceiling is visible rather than silent.

The consequence is the one that matters to you. Parameter count and per-step FLOPs are identical across every submission, so the budget buys every tokenizer exactly the same compute, and the only thing that moves the reading is how well your vocabulary codes the corpus.

The family slot budget is `SLOT_BUDGET`, which is the declared vocab_size less the 256-byte alphabet, and it is **zero-sum**: families are filled in the fixed order above, so a slot spent on one family is a slot no other family gets.

The tension is arithmetic and nobody authored a curve for it. A vocabulary that codes text in fewer tokens covers more training **bytes** under the same step budget. A vocabulary that spends its table on rare strings leaves rows the decoder never learns. And a construction family can run out of material: nothing in the substrate announces that.

## What you submit

One JSON document at the path in `task.toml` under `[agent] submission_path`:

```json
{
  "allocations": {"merge_depth": 0, "span_units": 0, "numeric_units": 0, "punct_units": 0},
  "extra_tokens": [],
  "attempts": [
    {"index": 1, "direction": "merge_depth", "allocation": {"merge_depth": 2500},
     "reallocated_from": null, "carried_frontier": {"merge_depth": 2.0864}}
  ],
  "graded_attempt_index": 1
}
```

`allocations` is the construction that gets graded. `attempts` is the record of the search that produced it: one row per attempt, in order, each naming the direction it moved, the allocation it tried, the direction it reallocated **off** if any, and the per-direction frontier the session was holding at that point. The harness **re-measures every allocation you record**, by training the frozen decoder again on it, so a fabricated frontier does not survive; what the record establishes is the order in which you moved and the state you carried, not any number you claim.

## The session

- up to **50 attempts**
- **0.12 hours** of wall clock for any one attempt
- **6.0 hours** across the whole session
- one H100, single accelerator, no multi-GPU scaling
- the graded submission is the **best** attempt you carry

## Machine-readable refusals

Every zero score carries a `reason` in `/logs/verifier/score.json` alongside the bare float in `/logs/verifier/reward.txt`. If you are refused, the reason names what was refused:

| reason | what it means |
|---|---|
| `denominator-not-frozen-eval-bytes` | the reading was divided by something other than the held-out FineWeb byte count |
| `compute-budget-overspent` | the harness counter measured more spend than the frozen budget, or more was requested, or the forward and backward tallies did not match the steps |
| `readout-smoothing-on-graded-path` | a filtered graded readout was applied or requested |
| `early-stop-not-established` | the run halted before the scheduled evaluation points were reached |
| `reading-not-sustained` | the reading did not hold across the scheduled evaluation points |
| `state-not-harness-owned` | the evaluated parameters were not the snapshot the harness owns at that point, or were not shaped to the frozen architecture |
| `submission-reported-metric-on-graded-path` | a number the submission reported was offered to, or consumed by, the graded path |
| `flattened-direction-not-reallocated` | a direction stopped paying and the allocation never moved off it |
| `carried-state-collapsed` | the carried per-direction frontier was reset or truncated between attempts |
| `single-direction-sweep-not-exceeded` | the result does not beat the harness's own single-direction sweep by the bound margin |
| `frozen-corpus-phase-not-established` | the corpus the reading was taken over does not open at the construction phase the verifier measures over |
| `frozen-corpus-period-not-established` | the corpus the reading was taken over does not span the construction period the verifier measures over |
