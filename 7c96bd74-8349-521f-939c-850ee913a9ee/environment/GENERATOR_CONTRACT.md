# The generator contract

You write one file: `/workspace/submission/generator.py`.

## How it is run

The verifier copies that file, and nothing else, into a fresh empty temporary directory, and runs it there as a new session leader under a small environment allowlist:

```
python3 generator.py --out corpus.jsonl
```

Its working directory contains your file and nothing else. There is no held-out split in it, no FineWeb shard in it, no grading code in it, and no path from it into any of those. The process group is killed when the run ends, on every path.

## What it must write

One JSON object per line in `corpus.jsonl`, each carrying a single `text` field holding one pretraining document:

```json
{"text": "The Antikythera mechanism is an ancient Greek geared device..."}
```

- `text` must be a non-empty string. A line that is not a JSON object, or that carries no `text`, is dropped as unusable rather than guessed at.
- A document longer than 65536 characters is dropped whole rather than truncated, because a truncated document is a document you did not write.
- At most 2000 documents are read. Later lines are ignored.
- Documents are tokenized with the GPT-2 BPE the FineWeb10B shards were built with, delimited by the end-of-text token, and any token outside the frozen vocabulary of 50304 is dropped.

## What the harness then does, and you do not

The harness tokenizes your corpus into a shard in the modded-nanogpt container format and trains the canonical decoder declared in `nanogpt_substrate.json` on a frozen mix: every step draws 131072 tokens from your shard and 393216 tokens from the FineWeb10B train shards, which is the frozen batch of 524288 tokens with exactly one forward and one backward pass per step. The architecture is vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024, and the optimizer is the record-set pair of AdamW on the embedding, the output projection and the one-dimensional parameters plus Muon on the hidden block matrices. Nothing in that list is yours to move.

At the evaluation points the verifier schedules, the harness writes a real parameter snapshot and the verifier reloads it in its own process and measures the held-out benchmark score, which is `exp(-val_loss)` over a FineWeb validation split you never see. You never see that split, and no number your program prints reaches the score.

You may print `REPORTED_SCORE=<float>` on stdout. It is recorded so the verifier can measure how far your own estimate diverged from its measurement. It is never graded, and it never substitutes for the measurement.

`environment/trainer.py` is a mirror of that training core. It takes a `--steps` argument so you can run a short local pass and see whether your corpus helps or hurts; the graded run is the full frozen budget and the graded number is the verifier's.

## The one thing that scores zero outright

Every document you emit is screened against every held-out document under a declared normalization and a bound similarity threshold. A single hit scores zero with the machine-readable reason `heldout-item-reproduced`. Reproducing held-out documents, whether verbatim, reworded, or reconstructed from memory, is the failure this task is built to catch. Generalise instead: write documents whose content teaches the decoder something that transfers to held-out web text it has never seen.

`dev_examples.jsonl` in this directory is a handful of public examples showing the format. It is not the held-out split and it is not a sample of it.
