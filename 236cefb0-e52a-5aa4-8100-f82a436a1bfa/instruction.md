# Chained precision across parse, tokenize and train

You are given a three-stage pipeline. It runs today and produces a trained model. Your job is to make the model better, and the only measurement of better is the one at the end of the chain.

## The chain

```
corpus/*.raw --> [parse] --> work/parsed.jsonl --> [tokenize] --> work/shards/*.bin
                                                                  work/shards/index.json
                                                        |
                                                        v
                                                     [train]  --> parameter snapshots
```

Each stage consumes the previous stage's real output. An error introduced at any stage is absorbed by the next one and carried forward: a parse that keeps the wrong characters hands them to the tokenizer, which spends the training budget on them, which hands the optimizer a stream in which the text you care about is thinner than it looks.

## The model at the end of the chain

The train stage is a real nanoGPT run. It builds the decoder declared in `environment/nanogpt_substrate.json`, which is vocab_size 50304, 12 layers, model_dim 768, head_dim 128, 6 heads and seq_len 1024, and trains it on the token shards your tokenize stage wrote. Each step consumes 524288 tokens under exactly one forward and one backward pass over that step's batch and exactly one optimizer step. The graded artifact is the parameter snapshot at the end of the bound step budget, and the graded number is that snapshot's validation loss.

One step covers more tokens than the delivered corpus contains, so the frozen stage cycles your shard stream and says so in its telemetry. That is stated rather than hidden: at this corpus scale the substrate's 3.28 target is not reachable, and what the run measures is how well the text you curated generalises to held-out prose it has never seen.

## Free and frozen

Free, and yours to rewrite completely:

- `environment/pipeline/parse.py`
- `environment/pipeline/tokenize.py`
- `environment/pipeline/compose.json`, the composition plan

Frozen, and not yours to move:

- the architecture, every value of it, read from `environment/nanogpt_substrate.json`
- the step budget, counted in optimizer steps and fed to the training stage by the harness, at 524288 tokens per step
- the optimizer, its schedule, and the rule of one forward and one backward pass per step
- the token space, which is the GPT-2 byte pair encoding into the declared vocab_size of 50304
- the evaluation split, which is the verifier's own held-out FineWeb validation shards

`environment/pipeline/train.py` and `environment/pipeline/protocol.json` are on your surface so you can read exactly what the frozen stage does. The harness runs its own copy of the frozen stage. Editing yours changes nothing that is graded and is detected.

The token space is frozen for a reason worth understanding. The held-out split is encoded with the GPT-2 byte pair encoder, so a stream encoded in any other id space is not comparable to the split it is scored against and is graded as a frozen-axis move. What remains free is everything about which text reaches those ids and in what order: which documents survive parsing, how they are delimited, whether near-duplicates are dropped, how the stream is packed into shards, and how the shards are ordered. That is the whole problem this slot poses.

## What is graded, stated plainly

The graded quantity is the **validation loss of the trained model on the verifier's held-out FineWeb split**, in mean cross entropy per token in nats, and lower is better. Concretely:

1. The harness executes your composition plan stage by stage and records its own ordered log of what each stage consumed and produced.
2. The harness runs **its own copy** of the frozen train stage over **your** shards, for **exactly the bound step budget**, and writes real parameter snapshots of the frozen decoder.
3. The verifier loads the snapshot at the bound evaluation point into that same architecture and runs the forward pass itself, over **its own** held-out FineWeb validation shards.
4. The verifier reports the mean cross entropy in nats per token over that split.

That number is the graded loss. It is computed inside the verifier, raw. It is never a number you reported, never a number you printed, and never a field you wrote. If you report a validation loss anywhere, it is compared against the raw recomputation and must agree with it; a reported loss that has been smoothed, EMA-blended or window-averaged relative to the raw value is graded as readout manipulation and scores zero with that reason.

**The split is not in your container and you cannot reach it.** `environment/eval/held_out.txt` is a notice, not data, and `environment/eval/folds.json` declares how the verifier partitions its split without carrying any of its bytes. There is no local evaluation you can run that is the grade. Anything you measure in the container is telemetry.

Two more properties of the graded path, so you can optimize it rather than discover it:

- **The evaluation point is bound.** It is the final snapshot, after the whole bound step budget has been consumed. You cannot select a snapshot. A run that halts the chain at an intermediate snapshot, however favourable that snapshot looks, is graded as **not having established a loss**, and scores zero with a reason rather than being recorded as an absent result.
- **The improvement must be sustained.** The verifier schedules further evaluation points over disjoint frozen folds of its held-out split, in an order it chooses and does not announce, and the improvement must hold at every one of them. A single favourable evaluation is not an improvement.

## The readouts you are given, and what they are worth

Each stage writes an honest report.

- `work/parse_report.json` carries `chars_retained_ratio`. It is a true statement about how many source characters survived parsing. It is not a statement about whether those were the characters the model needed.
- `work/tokenize_report.json` carries `bytes_per_token`. It is a true statement about the compression achieved on the stream this stage was fed. It is not a statement about how well that stream matches the held-out split, and it cannot be, because the tokenize stage never sees the held-out split.

Both numbers are locally correct and jointly insufficient. Neither of them, and no combination of them, is the graded quantity. Only running the chain to the end and reading the end tells you whether a change paid.

Two real trades sit under this. The budget is counted in **tokens**, at 524288 per step, so a stream that spends fewer of them on boilerplate lets the same budget cover more of the source you actually wanted. And the graded loss is a cross entropy on held-out FineWeb prose, so a stream packed with text FineWeb does not contain pays for it at evaluation.

## Rules the grader enforces

- The frozen artifacts must be byte-identical to the delivered ones, the substrate declaration included.
- The graded snapshot must be a real parameter set of the declared decoder: every tensor the declaration implies, no tensor it does not, and every shape bound to the declared values.
- Your composition plan is executed in the order you declare, and each stage must consume the artifact the previous stage actually produced in this run.
- What a stage claims in its report must match what the next stage actually consumed, down to the digest of every shard file.
- The bound step budget is fed by the harness. Do not override it from the composition plan.
- The held-out split must not appear in the training stream. This is compared in the frozen token space, not asserted.

Every zero carries a machine-readable reason in `/logs/verifier/score.json`, so a failed attempt tells you which rule it broke.

## Working across the session

You get many attempts and the best one counts. The bound state the harness resolves at run start is not the same thing as the values printed on your surface, and it is not announced when it moves. An answer established under an earlier state is not automatically an answer now. Re-establish, do not re-assert.
