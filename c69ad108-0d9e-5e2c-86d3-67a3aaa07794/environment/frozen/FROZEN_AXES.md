# Frozen axes

Everything in this file is fixed. The synthetic data generator you write is the only free axis in this task.

| axis | value | where it lives |
|---|---|---|
| substrate | the canonical nanoGPT operating point | `../nanogpt_substrate.json` |
| vocab size | 50304 | substrate `architecture.vocab_size` |
| layers | 12 | substrate `architecture.num_layers` |
| model dim | 768 | substrate `architecture.model_dim` |
| head dim, heads | 128, 6 | substrate `architecture.head_dim`, `architecture.num_heads` |
| sequence length | 1024 | substrate `architecture.seq_len` |
| tokens per optimizer step | 524288 | substrate `run.batch_tokens_per_step` |
| passes per step | one forward, one backward | substrate `run.forward_passes_per_step`, `run.backward_passes_per_step` |
| model | the decoder built from that declaration, about 124 million parameters | `trainer.GPT` |
| tokenizer | GPT-2 byte-pair encoding | `shards.ENCODING_NAME` |
| optimizer | AdamW, bound learning rate, bound warmup and cosine schedule | `trainer.LEARNING_RATE`, `trainer.learning_rate_at` |
| corpus size | 512 documents | `trainer.BOUND_CORPUS_DOCUMENTS` |
| document length | 1024 tokens, truncated, never padded | `trainer.BOUND_DOCUMENT_TOKENS` |
| shard size | 524288 tokens | `trainer.BOUND_CORPUS_TOKENS` |
| optimizer steps | 96 | `trainer.BOUND_STEPS` |
| passes over the corpus | 96 | `trainer.BOUND_EPOCHS` |
| tokens fed | 50331648 | `trainer.BOUND_TOKENS_FED` |
| evaluation steps | 64, 80, 96 | `trainer.SCHEDULED_POINTS` |
| bound evaluation step | 96 | `trainer.BOUND_EVALUATION_POINT` |
| sustain tolerance | 0.50 nats, two-sided | `trainer.SUSTAIN_TOLERANCE` |
| held-out validation split | not on this surface | held by the verifier |

`trainer.py` and `shards.py` here are byte-identical to the copies the verifier runs, and their sha256 values are pinned in the verifier's manifest. Reading them tells you exactly what happens to your corpus: it is encoded with the GPT-2 tokenizer, truncated to 1024 tokens per document, packed into a `.bin` shard in the same layout `data/cached_fineweb10B.py` writes, and read back by the same reader a FineWeb shard is read with.

The architecture is not written into `trainer.py`. It is read from `nanogpt_substrate.json` at construction time, so the decoder that trains is the decoder the substrate names. The verifier re-reads its own copy of that declaration at grading time, which means editing the copy on this surface produces a refused run with reason frozen-axis-moved rather than a different grade.

The corpus size follows from the batch size rather than from taste. 512 documents of 1024 tokens is 524288 tokens, which is exactly one optimizer step's worth of tokens at the frozen batch size, so one pass over your corpus is exactly one step and the bound run makes 96 passes over what you wrote. Your corpus is seen ninety-six times, which is what makes its diversity the whole task.

Determinism here is seeded rather than structural. A 124-million-parameter decoder trained on an accelerator accumulates in a nondeterministic order, so the trainer seeds every generator it owns and records the parameter digest of the state it actually evaluated. What is guaranteed is that the evaluated parameters are the trained parameters, not that two machines produce the same last bit. The sustain tolerance is bound wider than that accumulation noise for exactly this reason, and it is recorded as provisional.

`benchmark_dev.jsonl` beside this file is **authored dev material** with the same document shape as the held-out split. It is not sampled from the held-out split, shares no item with it, and is never read on the graded path. Use it to check your generator's habits against a realistic shape. The held-out split itself is FineWeb validation text held only by the verifier, at a path that is a constant inside the grading code, and it is not on this surface in any form.
