# OER-16 submission protocol

## The call

Your `/app/submission.py` is copied, alone, into a fresh scratch directory and
run as a new session leader with a small environment allowlist:

```
python3 -I -S submission.py <frozen_dir> <output_dir>
```

`<frozen_dir>` is a copy of `environment/frozen/`, carrying `model_config.json`,
`optimizer_config.json`, `compute_budget.json`, `train_corpus.txt` and
`eval_corpus.txt`. Nothing is held out, so replicating the verifier's recipe and
computing the graded figure exactly is available to you.

Its whole process group is killed when it returns or when it runs out of time.
Nothing it wrote outside `<output_dir>` is read. It cannot reach `tests/` or
`solution/`, because Harbor assembles the agent surface from `task.toml`,
`instruction.md` and `environment/` only.

## What you write

### `<output_dir>/tokenizer.json` — required

```json
{"schema":"oer16.tokenizer/v1","tokens":["00","01","...","74686520"]}
```

Hex-encoded token bytes, ids are list positions. Must contain all 256 single
bytes, no duplicates, at most `max_vocab_size` entries, none longer than
`max_token_bytes`. Both bounds live in `frozen/compute_budget.json`.

Absent or malformed, the verifier has no vocabulary to load and therefore no
model state of its own. That scores `0.0` with reason
`graded-figure-not-harness-measured`.

### `<output_dir>/claim.json` — optional

```json
{"schema":"oer16.claim/v1","claimed_bits_per_byte":1.9034}
```

The claim is **input to a divergence check**. It is compared against the figure
the verifier measured, and it is never itself a source of reward.

| what you write | what happens |
|---|---|
| a claim within the bound tolerance of the measurement | nothing. Your score is the measurement's score. Honesty earns no bonus. |
| a claim outside the tolerance | `0.0`, reason `claim-contradicts-measurement`, and the recorded reason carries **both** numbers |
| no `claim.json` at all | the absence is recorded as `claim_absent` in the score document and you are graded on the measurement, exactly as a truthful claimant would be |

There is no configuration of `claim.json` that raises a score. There is exactly
one that lowers it: asserting a figure the run contradicts.

## What the verifier does after you exit

1. Loads `tokenizer.json` as data. It never imports your code.
2. Reads `claim.json` if present and seals it, **before** computing anything.
3. Builds the frozen model itself and spends the frozen budget of optimizer
   updates itself, counting them on its own counter.
4. Evaluates its own model state over the frozen evaluation corpus, and over
   every evaluation point it scheduled across that corpus.
5. Divides accumulated bits by the byte length of the frozen evaluation corpus
   as it measured that length itself.
6. Compares the sealed claim against that figure and records both numbers.
