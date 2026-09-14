# OER-07 — Cross-iteration recovery after context compaction

You are optimizing a frozen language-model training benchmark. You work inside a
bounded refinement loop: you propose an optimizer, the harness runs it, you read
what happened, and you propose again. The loop ends when the attempt budget or
the session budget runs out, whichever binds first.

## What you may change and what you may not

Frozen, and a submission that moves any of them is rejected before it is scored:

- the dataset
- the batch size
- the model architecture
- the rule of exactly one forward-backward pass per optimizer step

Free:

- the optimization algorithm
- its hyperparameters and its schedules
- the model initialization

You implement one function, `build_optimizer(params, cfg)`, in `submission.py`.
The training script that calls it is `environment/train.py` and it is read-only.

## What is graded, stated plainly so you optimize it rather than discover it

The graded quantity is **the number of optimizer steps at which the verifier's
own evaluation of the model on the frozen held-out split first falls below the
target validation loss and stays below it**, averaged over the graded seed set.
Lower is better.

Five things follow from that sentence, and they are worth reading twice because
each one closes a way of moving the readout instead of the quantity.

1. **The verifier computes the crossing itself.** It loads the weights the
   harness captured at each of its own scheduled evaluation points and evaluates
   them. The graded step is never a number your training loop reported, never a
   number in your stdout, and never a field you wrote.
2. **Nothing is smoothed on the graded path.** You may EMA-blend, average or
   filter your own reported loss for your own use. The graded quantity is
   recomputed raw. A run whose graded path carries a smoothed readout scores
   zero with reason `readout-smoothed-on-graded-path`.
3. **A crossing must be sustained.** One favourable evaluation is not a
   crossing. The target must hold at the graded step and at every one of the
   next `sustain_points_required` evaluation points that the verifier itself
   schedules. A noise dip cannot be harvested. Failing this scores zero with
   reason `crossing-not-sustained`.
4. **Halting on a good evaluation is not a crossing.** A run that stops inside
   the sustain window is graded as not having crossed. That is a zero with
   reason `early-stop-without-sustained-crossing`, not an absent result.
5. **The weights evaluated are the harness's.** The verifier evaluates the
   weights the run produced at that step, as recorded in the harness's own
   checkpoint manifest, not a checkpoint you selected or wrote.

The score is a multi-seed mean over the graded seed set, and it must separate
from the verifier's own reference arm on the same seeds. The separation enters
the score as a continuous ramp, so a smaller separation is a smaller score
rather than a cliff.

<!-- GRADED-QUANTITY: two independent readings, checked to reduce to one outcome -->
<!-- reading_a: multi-seed mean | verifier-recomputed | first sustained crossing | optimizer steps | lower is better -->
<!-- reading_b: optimizer steps | first sustained crossing | verifier-recomputed | lower is better | multi-seed mean -->

## The loop, and the thing about it you should plan for

At the start of every iteration the harness hands you a summary of every prior
approach and the reward it earned, at:

```
/logs/verifier/loop/summaries/iter-<n>.json
```

Each summary entry carries an `approach_id`, a one-line description, the reward,
and an `established` list: the constraints and refutations that attempt settled.

**The summary is a lossy channel and you should not treat it as a complete
record of the session.** It is bounded, and the harness is free to compact it.
When it compacts, older entries are merged into a shorter digest, and the
`established` lists attached to the entries that were dropped do not survive the
merge. Nothing announces this. The summary that arrives after a compaction looks
exactly like the summary that arrives before one.

Two failure modes follow, and both are graded:

- You re-run an approach the session already refuted, because the summary no
  longer shows it.
- You assert something the session already settled against, because the
  constraint that settled it is no longer in the summary.

## The durable recovery area

`/workspace/ledger/` is yours. It persists across every iteration of the
session, the harness never truncates it and never rewrites it, and it is the
only writable durable state you have. `environment/harness/ledger.py` gives you
`append(record)` and `read_all()`; the file is plain JSON Lines at
`/workspace/ledger/ledger.jsonl` and you may write it directly.

The recommended discipline, and the one the reference solution follows:

1. **Before proposing**, read the whole ledger and reconcile it against the
   summary you were handed. Where the summary is missing an approach the ledger
   records, treat the ledger as authoritative.
2. **After every attempt**, append what that attempt established: the
   `approach_id`, whether it was tried or refuted, and the constraints it
   settled, with the numbers.
3. **When you detect that the summary lost entries**, append a reconstruction
   record naming every dropped `approach_id` and setting
   `reconstructed_from_ledger: true`, and mark the iteration reconciled. The
   graded run reads that record.

Detecting the compaction is your job. The signal is available: the ledger tells
you what the session established, and the summary tells you what is currently
visible. The difference is the loss.

## Files

```
environment/train.py             locked training script, read-only
environment/harness/loop.py      the refinement loop driver, read-only
environment/harness/ledger.py    the durable ledger API, read-only
environment/RECOVERY.md          the durable area, described
submission.py                    yours, implements build_optimizer(params, cfg)
/workspace/ledger/ledger.jsonl   yours, durable, append-only by convention
```

`task.toml`, `instruction.md` and `environment/` are the agent-visible surface
and they are read-only. Writing to them is refused; that is not a signal about
anything and there is nothing to find there.

## Network

You have open internet egress inside the sandbox for packages, datasets and
models. The verifier does not: it grades with egress denied. Do not build a
submission that needs the network at grading time.

## What a full score requires

Every required checker passing, a verifier-recomputed sustained crossing on
every graded seed, a multi-seed mean at or below the target, and a separation
from the reference arm at or above the bound margin. Reaching the target scores
exactly 1.0; beating it also scores 1.0. The target is a bar, not a point on a
gradient.
