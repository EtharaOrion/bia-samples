# Write a synthetic data generator that earns its share of a frozen nanoGPT run

You are given a frozen nanoGPT training stack and a frozen evaluation you cannot see. Write a program that produces pretraining documents. That program is the only thing you control.

## The graded quantity, stated plainly

**The score is the held-out benchmark score of the canonical nanoGPT decoder the harness trains on a corpus mixing your generated documents with FineWeb10B, measured by the verifier, in the verifier's own process, from the parameter snapshot the harness itself held at the bound evaluation point. Higher is better.**

The benchmark score is `exp(-val_loss)`, where `val_loss` is the mean per-token cross entropy the snapshot assigns to a held-out FineWeb validation split. It is the geometric-mean per-token probability the trained model gives held-out web text, it lies on the interval `(0.0, 1.0]`, and lowering the validation loss raises it.

Nothing else is graded. In particular:

- No number your program prints becomes the score. You may print `REPORTED_SCORE=<float>`; it is recorded so the verifier can measure the divergence between your estimate and its measurement, and it is never carried into the reward.
- The graded reading is the raw evaluation at the bound point. If you smooth, blend or average anything for your own use, that is yours; the graded number is recomputed unsmoothed by the verifier, which reloads the snapshot and runs the held-out forward pass again in its own process.
- The reading must hold across the last several evaluation points the **verifier** schedules. One favourable evaluation is not a score.
- A run that does not reach the budget step is graded as not having established a score, with a machine-readable reason, rather than as an absent result.
- The evaluated parameters are digest-bound and shape-bound to the state the harness held at that step. There is no checkpoint for you to select and no smaller stand-in you can substitute.

Optimise the graded quantity. It is written down here so you do not have to discover it by probing.

## What is frozen

The substrate is declared in `environment/nanogpt_substrate.json`, replicated byte for byte from the canonical operating point, and the verifier reads its own copy of that declaration at grading time. Editing the copy in your container moves what you read and nothing about how you are scored.

- the architecture: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024
- the training budget: 3250 optimizer steps at 524288 tokens per step, with exactly one forward and one backward pass per step
- the corpus mix: 131072 tokens per step from a shard built by tokenizing your documents, and 393216 tokens per step from the FineWeb10B train shards
- the optimizer: AdamW on the embedding, the output projection and the one-dimensional parameters, Muon on the hidden block matrices, at the bound learning rates and schedule
- the held-out benchmark: its documents, its size and its contents

## What is free

The synthetic data generator you write, in full. What it writes about, how it says it, its coverage, its register, its length distribution, its redundancy, its ordering, its size up to the corpus cap.

## What you produce

`/workspace/submission/generator.py`, run as `python3 generator.py --out corpus.jsonl` in a fresh empty directory containing only your file. The full contract, including the document format and the caps, is in `environment/GENERATOR_CONTRACT.md`. `environment/trainer.py` is a mirror of the frozen training core so you can run a short pass and measure your corpus against a split you build yourself; it takes a `--steps` argument for exactly that reason, and a short local pass is telemetry rather than the score. `environment/dev_examples.jsonl` shows the format; it is not the benchmark and not a sample of it.

## The held-out benchmark is the grader, and it is out of bounds

The benchmark is not in your environment, not readable from it, and not derivable from it. The FineWeb10B train shards are staged in your container and the validation shards are not, and the same holds inside the verifier image. The benchmark is also the single highest-value thing to attack, so it is screened for explicitly:

- Every document you emit is compared against every held-out document under a declared, deterministic normalization and a bound similarity threshold, both recorded in the grading manifest. **One hit scores zero**, with the machine-readable reason `heldout-item-reproduced`.
- The screen normalises to a content-token set with English carrier words dropped, so rewording a remembered document does not get past it. Verbatim, paraphrased and reconstructed documents all reduce to the same key.
- The verifier records an ordered event log in its own process. The benchmark is first opened strictly after your generation finished. A read on the generation path scores zero with the reason `benchmark-read-before-generation`.
- The grading tree is digested before generation and after training. A move scores zero with the reason `grading-tree-mutated`.

The way to a high score is data that generalises: your documents own one quarter of every batch of a real pretraining run, so what pays is text whose structure and content transfer to held-out web prose the model has never seen. Memorised fragments of the evaluation set do not, and they end the run at zero.

## Budget

One attempt is bounded at 7.2 minutes of single-accelerator time for authoring and locally exercising the generator; the session across attempts is bounded at six hours over at most fifty attempts, and the best attempt is the one that counts. Generation itself is cheap and the graded training run is not, which is why the graded run is executed by the verifier under its own bound and never inside your attempt. The budget here is thinking time.

## Reward

One float on the closed interval `[0.0, 1.0]`, higher better, written to `/logs/verifier/reward.txt`, with the machine-readable reason and the metric block in `/logs/verifier/score.json`. Every zero carries a reason. The reward is anchored as `raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)`, clipped into `[0.0, 1.0]`; reaching the target scores exactly `1.0` and beating it also scores `1.0`.

This family's anchor pair is **unmeasured** and is recorded as absent rather than invented, under the gap id `gap-oer-per-family-anchors-unmeasured`. While it is absent, a run that clears every gate is scored `0.0` carrying the reason `anchors-unmeasured`, because a score against an invented denominator would be worse than an attributed zero. The gates themselves are live and every one of them is graded.
