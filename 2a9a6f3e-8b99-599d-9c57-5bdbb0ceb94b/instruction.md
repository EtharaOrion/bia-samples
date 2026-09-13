# A fused streaming kernel handed across a splice

You are optimizing one kernel. The operator it computes is frozen, its output is frozen, and the only thing you may change is **how** the kernel computes it. Your job is to make it **charge fewer reads** while producing **exactly the same integers**.

## The operator, stated once

Over the concatenated input stream `x` of length `M`, for every index `i`:

```
y[i] = ( sum over j in 0..window-1 of (ratio ** j mod modulus) * x[i - j] ) mod modulus,   x[t] = 0 for t < 0
```

`window`, `ratio` and `modulus` are in `environment/params.json`. `x` is `environment/stream.json`, a list of blocks. Everything is an exact integer. There is no tolerance anywhere on the graded path, because there is no float anywhere on the graded path.

A working transcription of this definition is already in the box, as `reference_run_block` in `environment/opstream.py`. It is correct and it is slow, and it is the baseline you are measured against.

## What is graded, stated plainly

The graded quantity is **speedup over the frozen reference kernel**, higher is better, **gated on exact output equality**.

Read the next three points carefully, because they decide what is worth doing.

1. **Cost is counted reads, not seconds.** Your kernel never touches the sample list. It is handed a `Tape`, and `Tape.at(t)` is the only route to a sample. Every call charges exactly one read. The graded number is a ratio of read counts. Timing your own work is fine and it cannot move the reward.

2. **The verifier counts the graded charge itself.** After the splice, the verifier re-runs the remaining blocks under your kernel on its own Tape and reads the total off its own counter. No number you print, report or write into a file becomes the graded number. `reported_resume_charge` exists in the handoff so a mismatch is visible, not so it can be used.

3. **Exact output equality is a HARD GATE.** It is not a term you can trade against speed. A kernel whose composed output differs from the frozen reference by one integer scores exactly `0.0` with the reason `numerical-output-diverged`, however few reads it charged. Every block of the stream must carry exactly one output record across the two phases, with no gap, no duplicate and no reordering.

## The two phases, and the splice between them

This unit runs in **two phases over one shared workspace**, and the second phase resumes from state the first one hands it. Three paths are shared between the two containers and they are separate on purpose: your kernel at `/app/submission.py`, the harness journal at `/logs/harness/phase_a.jsonl`, and your handoff at `/workspace/handoff.json`.

**The agent phase.** You install your kernel and run the harness driver:

```sh
python3 /task/environment/phase_a.py \
    --workspace /workspace \
    --kernel /app/submission.py \
    --stream /task/environment/stream.json \
    --params /task/environment/params.json
```

The driver feeds your kernel whole blocks in order, charges every read, and appends one chained row per block to the harness journal at `/logs/harness/phase_a.jsonl`. It keeps its own work counter, being the outputs produced so far multiplied by `window`, and it **halts the moment that counter reaches `params.splice_charge_budget`**. Where it halts is the **splice point**, and the state your kernel holds there is the **carried state**. The driver records both, and the digest it takes over that carried state, in the journal's halt row.

**Neither of those two values is stated anywhere in this document, and neither can be read off it.** The splice point falls out of the drawn block lengths, and the carried state contains the operator's own output at the last sample before the splice. Run the first phase and read them back off the journal the driver wrote. Do not guess them, and do not write a handoff you did not earn: the verifier re-derives the splice point from the frozen stream for itself, recomputes the digest over the state you actually handed over, and recomputes the whole journal chain from its published genesis.

**The verifier phase.** It runs in a separate image. It reads the journal and your handoff, seeds your kernel with the carried state you handed over, and resumes the remaining blocks under its own Tape. The resume Tape **refuses every read before the splice point**. Nothing about the first phase reaches the second except through the carried state, so a carried state that is wrong produces output that is wrong.

You may run the resume yourself before you hand anything over, with `environment/phase_b.py`, which is the same driver the verifier runs.

## What is frozen and what is yours

Frozen, and unchangeable:

- the operator, as defined above and transcribed in `environment/opstream.py`
- the input stream, `environment/stream.json`, and the block boundaries it fixes
- the operator parameters, `environment/params.json`, including the splice charge budget
- the reference kernel, which fixes the baseline cost

Yours, and the whole of what you may change: the **kernel implementation** at `/app/submission.py`. It must expose exactly one entry point:

```python
def run_block(tape, params, carry, start, length):
    """Return (outputs, next_carry) for the absolute indices start .. start+length-1."""
```

`carry` is a mapping with two keys and no others. `accumulator` is `y[start-1]`, or `0` at the stream origin. `history` is the list of the `window` samples `x[start-window .. start-1]`, oldest first, zero-filled where the index is negative. `next_carry` must have that same shape at the far end of the block, and the driver refuses a history of any other length.

## What to hand over

Write `handoff.json` in the workspace root:

```json
{
  "schema": "oer28.handoff/v1",
  "splice_offset": 0,
  "splice_block_index": 0,
  "carried_state": {"accumulator": 0, "history": []},
  "carried_state_sha256": "<sha256 of the canonical carried state>",
  "journal_chain": "<the chain link the journal's halt row closes on>",
  "kernel_sha256": "<sha256 of the kernel file the verifier will grade>",
  "reported_resume_charge": 0
}
```

- `splice_offset` is the absolute sample offset the agent phase halted at, read off the halt row. The zeros above are placeholders showing the shape, not values.
- `carried_state` is the carry your kernel held at that offset, and `carried_state_sha256` is the digest over its canonical encoding, which is `json.dumps(state, sort_keys=True, separators=(",", ":"))` over exactly the two keys.
- `kernel_sha256` must be the digest of the kernel file as it stands when the verifier grades it. The agent phase recorded the kernel it ran, and presenting a different kernel afterwards scores `0.0` with the reason `kernel-swapped-across-splice`.
- `reported_resume_charge` is what your own resume probe charged.

## Scoring

Stated once more, independently of the section above, so there is exactly one reading: the graded metric is **speedup over the frozen reference kernel on the resume segment**, **higher is better**, and exact output equality is a **hard gate** on it rather than a term traded against it.

One float in `[0.0, 1.0]`, higher is better, never binary. Every zero carries a machine-readable reason. Full reward is exactly `1.0`, and beating the target also scores `1.0`; the target is a bar, not a point on a gradient.

`raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)`, then `score = min(max(raw, 0.0), 1.0)`.

Running the frozen reference kernel unchanged is a complete, honest, correct submission. It scores `0.0` with the reason `graded`, because its speedup is exactly the baseline. That is a score and not a refusal, and the difference matters.

## Two notes on honesty

The cost model is a **counted-read model**, not a wall clock. That is deliberate: it makes the grading reproducible, so the same kernel earns the same verdict on any host. It is never presented as a timing measurement.

The harness owns the counter and the journal. `environment/phase_a.py` and `environment/phase_b.py` charge every read and stamp every record; the checkers read those records and never count anything themselves. You may instrument your own work however you like; your instrumentation is not graded and cannot move the reward.
