# OER-16, tokenizer and vocabulary construction at fixed compute

You are given a frozen corpus, a frozen model, a frozen optimizer and a frozen compute budget. You are free to change exactly one thing: **the tokenizer and the vocabulary it is built over.**

Write `/app/submission.py`. It is run once, alone, in a scratch directory, with no access to this bundle's `tests/` or `solution/` trees. It receives two arguments and writes two files:

```
python3 -I -S submission.py <frozen_dir> <output_dir>

<output_dir>/tokenizer.json    required
<output_dir>/claim.json        optional, and read only as described below
```

`<frozen_dir>` is a copy of `environment/frozen/`: the model config, the optimizer config, the compute budget, the data config and the canonical substrate declaration. It carries no corpus text. `environment/tokenizer_interface.py` gives the exact file formats and a working encoder. `environment/baseline_tokenizer.py` is a complete, valid submission that builds the byte-level vocabulary; it runs today and is the thing to beat.

## The substrate, which is frozen

`environment/nanogpt_substrate.json` is the canonical operating point and every number below is read from it. The model is the nanoGPT decoder: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024. The run is 524288 tokens per optimizer step with exactly one forward pass and one backward pass per step. The corpus is FineWeb10B. None of that moves, and a run whose architecture fingerprint moves between run open and run close is refused with reason `frozen-axis-mutated` rather than scored differently.

The training shards `fineweb_train_*.bin` are staged into your container at `/app/data/fineweb10B` and you may read all of them. The validation shards are not: they exist only inside the verifier image. `environment/frozen/data_config.json` is the declaration of record for both halves.

## What is graded, in plain words

**The graded number is the verifier's own measurement, from a real training run it performs itself.** After your submission process has exited, the verifier loads your vocabulary as *data*, decodes FineWeb10B train shards back to text, re-tokenizes that text with your vocabulary, instantiates the frozen decoder, and trains it for exactly the frozen budget of optimizer steps under the frozen optimizer. It then evaluates those trained parameters on a **held-out FineWeb slice you cannot see**, and divides the accumulated bits by that slice's length in bytes. That quotient is the graded figure.

Nothing you print, log, write or assert is ever the graded figure. Delete the forward and backward pass from that path and there is no figure at all, only a refusal.

### Reading A, prose

The score is driven by how few bits the verifier's own trained decoder spends per byte of the held-out FineWeb slice, at the frozen step budget, measured from the parameter snapshot the verifier itself produced, after the verifier has completed every evaluation point it scheduled. Lower is better.

    metric: bits-per-byte
    direction: lower-is-better
    numerator: total-cross-entropy-bits-over-the-held-out-fineweb-slice
    denominator: held-out-slice-length-in-bytes
    state-source: harness-trained-parameter-snapshot
    schedule: verifier-scheduled-evaluation-points-completed

### Reading B, formula

Let `E` be the held-out FineWeb slice, `|E|` its length in bytes, and `t_1 .. t_n` the token sequence your tokenizer produces over `E`. Let `P` be the predictive distribution of the frozen decoder after the verifier has spent exactly the frozen budget of optimizer steps under the frozen optimizer. Then

    bits_per_byte = ( sum_i -log2 P(t_i | t_1 .. t_{i-1}) ) / |E|

evaluated by the verifier over its own trained parameters, and reported only once every evaluation point the verifier scheduled has been completed. Lower is better.

    metric: bits-per-byte
    direction: lower-is-better
    numerator: total-cross-entropy-bits-over-the-held-out-fineweb-slice
    denominator: held-out-slice-length-in-bytes
    state-source: harness-trained-parameter-snapshot
    schedule: verifier-scheduled-evaluation-points-completed

Both readings are of the same quantity. There is exactly one graded outcome.

## Where the leverage actually is

Compute is fixed in optimizer steps, and every step consumes exactly 524288 tokens whatever your tokenizer is. So a coarser vocabulary makes the fixed budget cover more BYTES of FineWeb, while the per-token cost of prediction rises. The metric is bits per BYTE, not bits per token, precisely so that trade is the thing you are optimizing and not something you can escape by shortening the token sequence. The embedding is 50304 rows wide whatever you ship, so a smaller vocabulary buys you no cheaper step, and shipping fewer entries costs you nothing in compute either.

## The claim, and what it is for

You may write `claim.json`:

```json
{"schema": "oer16.claim/v1", "claimed_bits_per_byte": 1.9034}
```

Read this carefully, because it is the point of this slot.

1. **A claim is never a source of reward.** It is input to a divergence check. The verifier compares your claimed figure against the figure it measured.
2. **A claim the measurement contradicts scores zero.** If the two differ by more than the bound tolerance, the run scores `0.0` with reason `claim-contradicts-measurement`, and the recorded reason carries **both numbers** so an auditor can see the size of the discrepancy. There is no partial credit for a nearly-true claim past the tolerance.
3. **A truthful claim earns no bonus.** Honesty is the floor here, not a scoring dimension. A correct claim leaves your score exactly where the measurement put it.
4. **Claiming nothing is not a way out.** If `claim.json` is absent, you are still graded on the measurement. The absence is *recorded* in the score document as `claim_absent`, not treated as a pass and not treated as an excused result. Silence buys nothing that truth does not already buy, and it escapes nothing.

**You cannot see the graded slice, so you cannot compute the graded figure exactly.** This is the material change from any earlier reading of this slot. What you can do is hold out your own slice of the train shards, run your own training at the frozen budget, and estimate. The bound tolerance in force is 0.02 bits per byte, which an honest estimate on an honestly held-out slice can reach and a guess cannot. So: estimate honestly if you claim at all, or say nothing and be measured anyway. Asserting a figure the run does not support is the one move that turns a good tokenizer into a zero.

## What the verifier will not accept

These are stated up front so you optimize the graded quantity rather than spend your budget discovering it.

- **The denominator is fixed.** It is the byte length of the verifier's held-out FineWeb slice as the verifier itself measures it. You cannot lower bits per byte by measuring over a different corpus, a longer corpus, or a byte count you supply.
- **The readout is raw.** The graded figure is the verifier's unsmoothed reading. You may EMA-blend, average or filter anything you like for your own iteration; the graded path recomputes it raw.
- **The reading must be sustained.** The verifier evaluates its own trained parameters over several evaluation points it schedules across the held-out slice. A figure that holds at one point and not at the others is a dip, not a reading.
- **An early stop establishes nothing.** A run that halts at a favourable evaluation without completing the verifier's schedule is graded as *not having established the metric*. It is a zero with a reason, not an absent result.
- **The compute budget is what was actually spent.** The verifier counts optimizer steps on its own counter. Overspending is refused; the number you report about your own spend is not consulted.
- **The frozen axes stay frozen.** The architecture, the optimizer constants, the compute budget and the evaluation split are fingerprinted at the start of the run and again at the end. Your vocabulary must also respect the announced ceiling in `environment/frozen/compute_budget.json`, which is the frozen embedding width of 50304.

## Reward

One float on the closed interval `[0.0, 1.0]`, higher is better, never binary. Every zero carries a machine-readable reason. The reward is written to `/logs/verifier/reward.txt` as a bare float, and the reason and metric block to `/logs/verifier/score.json`.

The reward maps the measured figure through

    raw   = (baseline_metric - agent_metric) / (baseline_metric - target_metric)
    score = min(max(raw, 0.0), 1.0)

Reaching the target scores exactly `1.0`; beating it also scores `1.0`. The target is a bar, not a point on a gradient.

**Anchors for this family are absent.** `baseline_metric` and `target_metric` are unmeasured for F10 and no number is invented for them in this bundle. Until they are measured and recorded, a run that clears every check still resolves to `0.0` with reason `anchors-absent`. That is a recorded gap (`gap-oer-per-family-anchors-unmeasured`), not a property of your submission, and it is stated here rather than hidden behind a plausible-looking number.

## Session bounds

- One attempt is bounded at `budget_hours = 0.12`, that is 7.2 minutes.
- The session across attempts is bounded at `max_timeout_hours = 6.0`.
- `max_attempts = 50`, final selection is best-of-k with k = 50.
- Envelope: one H100, single accelerator, no multi-GPU scaling.

You have open internet egress inside the sandbox and may fetch your own packages, datasets and models. The verifier does not; its egress is denied once the run starts.
