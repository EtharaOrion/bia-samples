# OER-16 submission protocol

## The call

Your `/app/submission.py` is copied, alone, into a fresh scratch directory and run as a new session leader with a small environment allowlist:

```
python3 -I -S submission.py <frozen_dir> <output_dir>
```

`<frozen_dir>` is a copy of `environment/frozen/`, carrying `model_config.json`, `optimizer_config.json`, `compute_budget.json`, `data_config.json` and `nanogpt_substrate.json`. It carries **no corpus text**. The training shards are at the path `data_config.json` names, and the graded evaluation slice is held out: it exists only inside the verifier image, it is absent from `environment/` and from your container, and no call you can make resolves it.

Its whole process group is killed when it returns or when it runs out of time. Nothing it wrote outside `<output_dir>` is read. It cannot reach `tests/` or `solution/`, because Harbor assembles the agent surface from `task.toml`, `instruction.md` and `environment/` only.

## What you write

### `<output_dir>/tokenizer.json`, required

```json
{"schema":"oer16.tokenizer/v1","tokens":["00","01","...","74686520"]}
```

Hex-encoded token bytes, ids are list positions. Must contain all 256 single bytes, no duplicates, at most `max_vocab_size` entries, none longer than `max_token_bytes`. Both bounds live in `frozen/compute_budget.json`. `max_vocab_size` is 50304, the frozen embedding width, so a vocabulary above it cannot be embedded by the frozen architecture at all.

Absent or malformed, the verifier has no vocabulary to load, therefore no corpus to re-tokenize, therefore no training run and no parameters of its own. That scores `0.0` with reason `graded-figure-not-harness-measured`.

### `<output_dir>/claim.json`, optional

```json
{"schema":"oer16.claim/v1","claimed_bits_per_byte":1.9034}
```

The claim is **input to a divergence check**. It is compared against the figure the verifier measured, and it is never itself a source of reward.

| what you write | what happens |
|---|---|
| a claim within the bound tolerance of the measurement | nothing. Your score is the measurement's score. Honesty earns no bonus. |
| a claim outside the tolerance | `0.0`, reason `claim-contradicts-measurement`, and the recorded reason carries **both** numbers |
| no `claim.json` at all | the absence is recorded as `claim_absent` in the score document and you are graded on the measurement, exactly as a truthful claimant would be |

There is no configuration of `claim.json` that raises a score. There is exactly one that lowers it: asserting a figure the run contradicts.

The bound tolerance is 0.02 bits per byte. You cannot compute the graded figure, because the slice it is measured on is held out; you can estimate it by holding out your own slice of the train shards and running the frozen recipe yourself. An honest estimate reaches that tolerance and a guess does not.

## What the verifier does after you exit

1. Loads `tokenizer.json` as data. It never imports your code.
2. Reads `claim.json` if present and seals it, **before** computing anything.
3. Decodes FineWeb10B train shards back to UTF-8 text and re-tokenizes that text with your vocabulary.
4. Instantiates the frozen decoder itself, vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, seq_len 1024, and asserts the instantiated shape against the declaration before it spends anything.
5. Trains it for exactly the frozen budget of optimizer steps, 524288 tokens per step, one forward and one backward per step, counting steps on its own counter.
6. Evaluates its own trained parameters on the held-out FineWeb slice, and on every evaluation point it scheduled across that slice.
7. Divides accumulated bits by the byte length of that slice as it measured that length itself.
8. Compares the sealed claim against that figure and records both numbers.
