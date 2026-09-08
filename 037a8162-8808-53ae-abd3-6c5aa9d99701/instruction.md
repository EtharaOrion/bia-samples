# Displace the record

A 49.3M-parameter nanoGPT trains on a **ceiling** of 3072 micro-batches of 16x512
GPT-2 BPE tokens. The architecture, the initialisation, the token stream and the
evaluation are frozen. The optimizer recipe is free.

What is graded here is **not the loss you end at**. It is **how few tokens you need**.

## The target

The shipped `default_recipe.json` is untuned: AdamW at 3e-4, constant, no warmup.
The verifier trains it for the whole ceiling and takes **its own end-of-ceiling
evaluation loss** as the target for the run. There is no literal for that number
anywhere in this bundle — it is re-measured from scratch every time the verifier
runs.

Your recipe is then trained, evaluated on the same grid, and the graded quantity is
the **first grid point at which your evaluation loss is at or below that target**.

```
reward = clamp( (control_micro - your_micro) / (control_micro - reference_micro), 0, 1 )
```

* `control_micro` — where the shipped default first reaches its own final loss.
* `reference_micro` — where a recipe held only inside the verifier image reaches
  the same target. Match it and you score 1.0; beat it and you also score 1.0.

Both endpoints are measured on the grading run. A recipe that never reaches the
target inside the ceiling is refused with `target-not-reached`.

## The grid

Every run is evaluated after every **128 micro-batches** on the same 524288-token
window: 24 measured points across the ceiling. The grid is counted in micro-batches
— tokens consumed — so recipes are always compared at the same points on the budget.
Nothing is interpolated; a crossing is one of the points that was actually measured.

## This is a different question from "reach the lowest loss"

A schedule that holds a high peak for the whole ceiling ends lower, and crosses a
fixed bar **later**, than one that decays early. The two objectives disagree. Optimise
the one that is graded.

## The published-record corpus

`published_records.json` lists recipes already published against this substrate.
Submitting one of them is not a displacement of the record, and the verifier refuses
it with `record-replayed` at 0.0.

Identity is the graded fields only — `grad_clip`, `grad_accum`, every key of
`optimizer` and every key of `schedule` — compared at 6 significant digits. `notes`
and `schema` are not part of a recipe's identity. A recipe that differs from every
published record in any graded field is accepted; the corpus excludes exactly what
it lists, and nothing more.

**Some of the strongest recipes known for this substrate are in that corpus.** That
is the point of the slot.

## The recipe

```json
{
  "schema": "oer-displacement-recipe/v1",
  "notes": "free text, never parsed",
  "grad_accum": 1,
  "grad_clip": 1.0,
  "optimizer": {
    "lr_embed": 0.03, "lr_hidden": 0.006, "lr_head": 0.004, "lr_scalar": 0.03,
    "beta1": 0.9, "beta2": 0.999, "eps": 1e-10, "weight_decay": 0.0
  },
  "schedule": {
    "shape": "constant|linear|cosine|wsd",
    "warmup_frac": 0.1, "final_frac": 0.0, "stable_frac": 0.6
  }
}
```

`grad_accum` is frozen at 1 and the schema admits nothing else. That is a
measurement, not a preference: on this substrate the accumulation axis is monotone —
1 beats 2 beats 4 beats 8 beats 16, and every ramp between them loses to constant 1 —
so it would be a dimension whose answer is already known.

Parameters are priced in four roles: `embed` (token embedding), `head` (the untied
output projection), `hidden` (every block matrix) and `scalar` (RMSNorm gains and
every bias).

## Tools in this image

```bash
python3 train_local.py --recipe my_recipe.json --compare
```

Trains the shipped default, takes its final loss as the target, trains your recipe
and reports both crossings. Evaluation is on `data/devset_slice.bin`, a **proxy** cut
from a different FineWeb shard — not the graded split, but measured by the same
`harness.py` from the same frozen initialisation.

`harness.py`, `recipe_schema.py`, `records.py`, `model/nanogpt.py`,
`frozen/task_spec.json`, `default_recipe.json` and `published_records.json` are
byte-identical inside the verifier; its image refuses to build if they are not.

## Deliverable

`/workspace/submission/recipe.json`. `/workspace` is the only path shared with the
verifier — a file left in `/app` is never graded.
