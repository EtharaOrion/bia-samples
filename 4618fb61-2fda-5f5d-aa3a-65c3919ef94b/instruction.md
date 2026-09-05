# Synthetic data generation for a frozen nanoGPT run

You write one thing: a synthetic data generator. Everything downstream of it is frozen.

The corpus your generator emits is packed into a nanoGPT training shard and used to train the canonical decoder from scratch. Nothing else about that run is yours to move. The architecture, the tokenizer, the optimizer, the schedule, the step count and the evaluation split are all fixed, so the only quantity your generator can move is how well a language model trained on your data generalizes to text it has never seen.

## What you deliver

A Python file that, given an output directory as `sys.argv[1]`, writes:

- `corpus.jsonl` with exactly **512** lines, one JSON object per line, keys `seq` and `text`. `seq` starts at `0` and ascends by one with each line emitted. `text` is one document that must encode to **at least 1024 GPT-2 tokens**; it is truncated to exactly 1024 and a document that falls short leaves the shard short, which is graded.
- `run_report.json` with your own measurement of your own corpus, keys `declared_corpus_distinct_ngram_ratio`, `declared_max_mode_share`, `declared_documents`.

## What is graded, stated plainly

The graded quantity is **the mean token cross-entropy of the frozen 12-layer 768-dimension decoder on the held-out FineWeb validation split, computed by the verifier from the harness trainer's own parameters at the bound evaluation step, step 96.** Lower is better.

It is not a number you print. It is not a number you write to a file. It is not a smoothed, averaged or EMA-blended reading, and it is not a reading taken at a checkpoint you chose. The verifier packs your corpus into a shard, trains the frozen decoder on it, on its own, and evaluates it, on its own.

**The held-out split is not on your surface.** It is FineWeb validation text held only by the verifier, at a path that is a constant inside the grading code and is never resolved from your corpus, from your environment or from anything you can reach. You cannot read it, you cannot enumerate it, and you cannot tune against it. You can only produce training data that generalizes.

Four consequences follow, and they are graded:

1. **The readout is raw.** You may smooth anything you like for your own use. The graded number is recomputed unsmoothed as the exact quotient of the summed negative log likelihood over the token count the verifier's evaluator recorded. A smoothed reading on the graded path scores zero.
2. **The reading must be sustained.** The verifier evaluates at steps 64, 80 and 96. The reading must sit within 0.50 nats of the graded reading at every one of them. The band is two-sided, so a transient dip at the bound step is refused by the same arithmetic that refuses a spike. A single favourable evaluation is not a score.
3. **An early stop establishes nothing.** A run that halts at a favourable evaluation is graded as not having established the score, with a reason, not as an absent result.
4. **The parameters are the harness's.** The evaluated parameters are the trainer's own state at step 96, they must carry the frozen architecture, and the digest of the state that was read must equal the digest the harness recorded for itself at that step.

## What is frozen

- the substrate: the canonical nanoGPT operating point in `environment/nanogpt_substrate.json`, vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024, and 524288 tokens per optimizer step with exactly one forward and one backward pass per step
- the model: the decoder built from that declaration, roughly 124 million parameters
- the tokenizer: the GPT-2 byte-pair encoding the FineWeb corpus is built with
- the optimizer: AdamW at the bound learning rate, with the bound warmup and cosine schedule
- the training budget: a 512-document corpus packed into a 524288-token shard, 96 optimizer steps, 50331648 tokens fed, 96 passes over the corpus
- the held-out FineWeb validation split, its items and its evaluation steps

Read `environment/frozen/trainer.py` if you want to see exactly what happens to your corpus, and `environment/frozen/shards.py` if you want to see exactly how it is packed. Both are the same files the verifier runs and both have their sha256 pinned in the verifier's manifest.

The corpus size is not arbitrary. 512 documents of 1024 tokens is 524288 tokens, which is exactly one optimizer step's worth of tokens at the frozen batch size, so one pass over your corpus is exactly one step and the run makes 96 passes over what you wrote. Your corpus is seen 96 times. That is what makes its diversity the whole task: a dense neighbourhood in your data is memorised 96 times over, and a decoder that memorised your data does not predict FineWeb.

## The failure this task is built around

A generator can collapse. It stops varying its output partway through the run, keeps emitting, keeps the document count exactly right, keeps the JSON valid, keeps every document at length, **and exits zero**. Nothing announces it. The corpus is degenerate, the decoder has seen one thing ninety-six times, and the held-out loss does not move for a reason nothing in the run reported.

**The exit status of your generator establishes nothing about your corpus.** If you read exit code 0 as proof that generation succeeded, this task will take your budget and give you nothing back.

So measure your corpus. You have the means:

```sh
PYTHONPATH=environment/tools python3 -c "
import json, corpus_diversity as cd
rows = [json.loads(line) for line in open('corpus.jsonl')]
p = cd.profile([r['text'][:512] for r in rows], 8, 4, 0.60)
print(json.dumps(p.as_dict(), indent=2))
"
```

`environment/tools/corpus_diversity.py` is **byte-identical to the module the verifier computes the graded statistic with**, and its sha256 is pinned. The verifier reads the first 512 characters of each document, exactly as the snippet above does, because the statistic is quadratic in the document count and a full 1024-token document would make the measurement cost dominate the run without making it more discriminating. There is no statistic the verifier holds that you cannot compute first.

What it reports, and what the verifier requires:

| statistic | what it means | bound |
|---|---|---|
| `segment_distinct_ngram_ratio` | distinct token trigrams over total, per eighth of the emission timeline | no segment below **0.65** of the strongest segment |
| `max_mode_share` | largest share of the corpus in one near-duplicate neighbourhood, at overlap at or above 0.60 | at or below **0.08**, at the full corpus and at each of four prefixes |
| distinct trigram ratio per segment | advisory | a healthy generator clears **0.34**; not itself a gate |

A collapse is graded as **a failure with its own reason**, `generator-collapsed-silently`, which is not a poor loss. A degenerate corpus that happens to train to a respectable loss is still a collapse and is named as one.

## Two more things are graded

- **No near-duplicates of the held-out split.** No document may overlap a held-out FineWeb validation item at Jaccard 0.55 or above under the declared normalization over the bound excerpt. Memorising the evaluation set is not generalisation. `environment/frozen/benchmark_dev.jsonl` is authored dev material with the same document shape, disjoint from the held-out split and never read on the graded path, so you can check your generator's habits against a realistic shape without the evaluation surface being published.
- **The budget as fed.** 96 optimizer steps, 50331648 tokens fed, 96 passes, a 512-document corpus packing to a 524288-token shard, and exactly one forward and one backward pass per step, counted by the harness at the point of feeding.

## Reward

One float in `[0.0, 1.0]`, higher better, never binary, written to `/logs/verifier/reward.txt`, with the reason and the metric block in `/logs/verifier/score.json`. Every zero carries a machine-readable reason. Note that the reward is higher-better while the metric it is computed from is lower-better; the normalization below is what turns the one into the other.

`raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)`, then `score = min(max(raw, 0.0), 1.0)`. Because the metric is lower-is-better, a bound anchor pair is ordered with `baseline_metric` above `target_metric`, and the quotient still reads 0.0 at the baseline and 1.0 at the target. Reaching the target scores exactly 1.0 and beating it also scores 1.0.

The F14 anchors are unmeasured. `baseline_metric` and `target_metric` are declared absent under gap `gap-oer-per-family-anchors-unmeasured` rather than invented, so the score document records your `agent_metric` for normalization once they are bound. The substrate's own published anchors, record 36 at 3250 steps and record 46 at 2690 steps, are anchors for the optimizer track over the full FineWeb10B corpus and are not anchors for a decoder trained on a generated corpus, so they are not borrowed here.

## Environment

You have open internet egress inside the sandbox for packages, datasets and models. The verifier does not: its egress is denied, and the GPT-2 tokenizer vocabulary is warmed into both images at build time for exactly that reason. That asymmetry is deliberate and bound. The deny-list on `github.com`, `raw.githubusercontent.com` and `objects.githubusercontent.com` is not a containment proof, and it is not what grades you; the deterministic checkers are.
