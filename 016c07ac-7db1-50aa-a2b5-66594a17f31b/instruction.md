# OER-05 — Novelty-gated update rule under a behavioural corpus comparison

You are given a working training script that already reaches the target
validation loss. Return one that reaches the same loss in fewer optimizer steps.

## What is frozen and what is free

Frozen, and a submission that moves any of them is rejected before it is scored:

- the dataset and its held-out evaluation split
- the batch size
- the model architecture
- the rule of **one forward-backward pass per optimizer step**

Free:

- the optimization algorithm
- its schedules
- the initialization of the optimizer's internal state
- its hyperparameters

## What you edit

`environment/update_rule.py`. It exposes exactly one entry point:

```python
def build_update_rule(shape, hyper): ...
```

which returns an object with a `step(params, grads, state)` method. The training
loop in `environment/train_track3.py` is frozen and calls it once per optimizer
step. You may rewrite the body of `update_rule.py` freely, including its
internal state, its schedules and its hyperparameters.

## The quantity that is graded, stated plainly

Optimize this and nothing else.

**The graded quantity is the number of optimizer steps at which the verifier's
own evaluation of the model on the frozen held-out split first falls below the
target validation loss and stays below it, averaged over several seeds.**

Every clause of that sentence is load-bearing, so each is spelled out:

1. **The verifier computes the crossing itself.** It is never a number your
   training loop reported, never a number in your stdout, and never a field you
   wrote to disk. The verifier re-evaluates the model on the frozen held-out
   split at evaluation points it schedules itself, and reads the crossing off
   its own numbers.
2. **The graded readout is raw.** No smoothing sits on the graded path. You may
   EMA-blend, average or otherwise filter your own reported loss for your own
   use; the graded quantity is recomputed unsmoothed by the verifier, and a run
   whose claimed crossing rests on a filtered readout scores zero with the
   reason `readout-smoothing-on-graded-path`.
3. **A crossing must be sustained.** One favourable evaluation is not a
   crossing. The target must hold at the graded step **and** at every one of the
   subsequent verifier-scheduled evaluation points inside the bound sustain
   window. A noise dip cannot be harvested. Failing this scores zero with the
   reason `crossing-not-sustained`.
4. **Stopping early is not crossing.** A run that halts at a favourable
   evaluation without completing the sustain window is graded as **not having
   crossed**. That is a zero with the reason
   `early-stop-without-sustained-window`, not an absent result.
5. **The evaluated weights are the harness's.** The verifier evaluates the
   weights the run produced at that step, taken from the harness-owned weight
   ledger. A checkpoint you selected and handed over is not evaluated. Failing
   this scores zero with the reason `weights-not-harness-owned`.
6. **Several seeds, and the spread counts.** The score is built from the mean
   over seeds of the per-seed result, with the half-spread across seeds
   subtracted. A result that is not separated from its own seed noise degrades
   continuously toward zero; it does not flip at a line.

## The novelty gate

Your update rule is compared against a **behavioural corpus** of update rules
that ship inside the verifier tree. The comparison is **not** a text-similarity
check on your source. Renaming variables, reordering statements, inlining
helpers, or reimplementing the same arithmetic in a different style will not
move it.

What is compared is **what your rule does**. The verifier drives your rule over
a fixed set of probe inputs, at a fixed dtype and a fixed reduction order,
records the parameter deltas it produces, and digests them. That behavioural
signature is compared against the signature of every corpus entry, recomputed by
the same probe on the same inputs. If your rule's behaviour lands within the
bound behavioural distance of a corpus entry, the run scores zero with the
reason `behaviour-matches-corpus-entry`.

Two consequences worth stating outright:

- The probe is deterministic. The same rule produces the same behavioural
  signature every time. If it does not, the run scores zero with the reason
  `probe-digest-nondeterministic`.
- The probe transcript is cross-bound to the update deltas the harness observed
  during the graded training run. A transcript that disagrees with what the
  harness watched your rule actually do scores zero with the reason
  `probe-transcript-diverges-from-harness-deltas`. There is nothing to gain by
  reporting a behaviour you did not exhibit.

Anyone holding the frozen bytes of this bundle can recompute the entire
comparison. The corpus, the probe inputs, the dtype and the reduction order are
all in `tests/`.

## Red lines

- The grading tree under `tests/` is not yours to write. Nothing you run may
  create, modify or delete a byte under it. Violation scores zero with the
  reason `grading-tree-mutated`.
- The held-out evaluation split is not yours to read or to train on.
- Your process never runs inside the grading process. It is copied alone into a
  fresh temporary directory, launched as a new session leader under a small
  environment allowlist, and its whole process group is killed when it is done.

## Network

Your sandbox has open internet egress so you can fetch your own packages,
datasets and models. The verifier's own surface has egress denied. The two
surfaces differ by design.

## Budget

One attempt is bounded at 0.12 hours, which is 7.2 minutes of single-H100 time.
The session across attempts is bounded at 6.0 hours, with at most 50 attempts,
and the best attempt of the session is selected.

## What you hand back

`environment/update_rule.py`, edited in place. Nothing else is read from your
working tree.
