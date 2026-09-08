# What to submit, where, and what happens to it

## The artifact

One JSON file at

    /workspace/submission/selection.json

`/workspace` is the only path this container shares with the verifier. A selection left
anywhere under `/app` is a selection that is never graded.

## The shape

```json
{
  "schema": "oer22-selection/v1",
  "notes": "free text, recorded for a human reader, never parsed",
  "keep": [11, 12, 13, 14, 15, 16],
  "weights": [0.1, 0.15, 0.15, 0.2, 0.2, 0.2]
}
```

* `keep` — snapshot ids from `frozen/task_spec.json:snapshots.ids`, which is `1..16`.
  Snapshot *i* is the parameter state immediately after optimizer step *i* × 128.
  Snapshot 16 is the final state of the run. Ids must be unique.
* `weights` — one non-negative number per kept snapshot, in the same order, summing to
  1.0 to within 1e-6. The verifier averages the kept states with exactly these weights,
  in float64, over the full state dict. Nothing is renormalised for you.
* `keep` may name **at most 6** snapshots. That is the storage budget, and it is the
  constraint that makes this a selection rather than an averaging identity.

`selection_schema.py` is the same file the verifier uses. Validate before you submit:

```bash
python3 -c "import sys; sys.path.insert(0,'/app'); import selection_schema as s; \
            print(s.load('/workspace/submission/selection.json'))"
```

A selection that fails the schema is **refused** at 0.0 with a machine-readable reason.
The schema is a gate, not a grade: there is no partial credit for a nearly-valid file.

## What is graded

The verifier runs the frozen training run once, keeps all 16 snapshots, then evaluates
three selections against those same snapshots on a held-out FineWeb validation slice that
is not in this container:

| selection | where it comes from | role |
|---|---|---|
| `environment/default_selection.json` | shipped, visible to you | the floor |
| the verifier's private reference | verifier image only | the bar |
| yours | `/workspace/submission/selection.json` | scored |

    reward = clamp( (default_loss - your_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints are **measured on the grading run**, from the same snapshots yours is
scored against. No number for either is stored anywhere in this bundle. Tying the default
scores 0.0; reaching or beating the reference scores 1.0.

## What is not graded

Anything your submission says about itself. `notes` is free text and is never parsed. The
submission carries no loss, no metric, and no claim that the grader reads. No code from
this container is imported or executed by the verifier.

## The part that is actually hard

`environment/data/devset_slice.bin` is cut from a FineWeb **training** shard. The graded
split is cut from the FineWeb **validation** shard. They are not the same distribution
sample, and the ordering of near-neighbouring selections is not stable between them: a
selection tuned until it is the single best on the devset has been tuned partly to the
devset's noise, and that part does not transfer. The *shape* of a good answer transfers;
the last decimal place of a ranking does not. Prefer a selection whose margin over its
neighbours is wide.
