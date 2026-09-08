# Keep six snapshots. Choose which, and how to combine them.

A training run has already been designed for you and it is frozen. Its architecture, its
initialisation seed, its token stream, its optimizer, its learning-rate schedule and its
6144 micro-batches are all fixed, and nothing you submit can move any of them. You are not
being asked to train a better model.

That run passes through **16 snapshots**, one every 384 optimizer steps. Snapshot 16 is
the final state. You have a **retention budget of 6**: you may keep at most six of the
sixteen, and you must say how to average the ones you keep.

Your submission is one JSON file:

    /workspace/submission/selection.json

```json
{
  "schema": "oer22-selection/v1",
  "notes": "free text, never parsed",
  "keep": [12, 13, 14, 15, 16],
  "weights": [0.2, 0.2, 0.2, 0.2, 0.2]
}
```

Ids are unique and drawn from 1..16. Weights are non-negative, one per kept id, in the
same order, and sum to 1.0 within 1e-6. The verifier averages the kept parameter states
with exactly those weights, in float64, across the full state dict. See
`/app/submission_protocol.md` for the whole contract and `/app/selection_schema.py` for
the validator the verifier itself uses.

## What you are scored on

The verifier runs the frozen training run once, keeps all sixteen snapshots, and then
evaluates three selections against those same snapshots on a held-out FineWeb validation
slice that is not in this container:

* the shipped `default_selection.json`, which keeps snapshot 16 alone — the floor;
* a reference selection that exists only inside the verifier image — the bar;
* yours.

```
reward = clamp( (default_loss - your_loss) / (default_loss - reference_loss), 0, 1 )
```

Both endpoints are measured on the grading run, against the same snapshots yours is. No
stored number for either exists anywhere in this bundle. Matching the default scores 0.0.
Reaching or beating the reference scores 1.0.

## What you have

    /app/train_local.py        runs the frozen run and scores selections on your devset
    /app/harness.py            the training, averaging and evaluation code, byte-identical
                               to the verifier's copy
    /app/selection_schema.py   the validator, byte-identical to the verifier's copy
    /app/frozen/task_spec.json the whole frozen half, including the exact recipe
    /app/data/train_slice.bin  exactly the tokens the frozen run consumes
    /app/data/devset_slice.bin your yardstick, cut from a FineWeb *training* shard

Start here:

```bash
python3 /app/train_local.py --sweep
```

That trains the frozen run once, prints the devset loss of every individual snapshot, and
scores a spread of retention shapes. One run is about two minutes; the snapshots it
produces are the same snapshots the verifier will produce.

## The thing that will cost you if you ignore it

Your devset is cut from a **training** shard. The graded split is cut from the
**validation** shard. They are not the same sample, and nothing announces where they
disagree. They agree closely on the *shape* of a good answer — how wide a window, which
direction the weights should lean — and they disagree on the ordering of selections that
are already close to each other. A selection driven to be the single best-scoring one on
your devset has absorbed some of your devset's noise, and that part will not be there when
it is graded.

Prefer an answer whose margin over its neighbours is wide enough that it would survive
being re-measured on a different sample.
