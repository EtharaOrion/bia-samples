# The recipe contract

Your submission is one file, `/workspace/submission/recipe.py`. It carries two things.

## 1. `RECIPE`, a module-level literal

The verifier lifts this out of your file's **bytes**, with `ast.literal_eval`, without
importing or executing anything. A `RECIPE` assembled at import time cannot be lifted and
scores zero.

| key | shape | what it means |
|---|---|---|
| `update_chain` | list of `{"primitive": str, "role": str}` | the ordered primitives your update applies, and the parameter group each applies to |
| `couplings` | list of `[source, target, kind]` | what is shared or gated between parameter groups |
| `schedule_families` | list of `[parameter, family]` | the family of each schedule, not its values |
| `init_family` | str | the initialization family |
| `hyper` | dict of the eight bound keys | the numeric values |

The eight `hyper` keys, in the order the screen uses them:

`lr_peak`, `momentum`, `second_moment_beta`, `weight_decay`, `warmup_frac`,
`orthogonalisation_steps`, `trust_region_rho`, `aux_lr_ratio`.

A key you do not use still belongs in the dict, set to `0.0`. Omitting a key does not make you
distant from a corpus entry; a key present on one side and absent on the other counts as
maximally distant on that key, so deleting fields cannot buy you proximity in either
direction.

## 2. `build_optimizer(param_groups, total_steps)`

Returns the per-step update plan the locked script consumes:

```python
{
    "recipe": RECIPE,
    "total_steps": int,
    "groups": [
        {
            "role": str,
            "lr": [float, ...],                 # length total_steps
            "momentum": [float, ...],           # length total_steps
            "second_moment_beta": float,
            "shares_second_moment_with": str | None,
            "lr_scale": float,
        },
        ...
    ],
    "init_family": str,
    "orthogonalisation_steps": int,
}
```

## What the screen reads, and what it ignores

The **structural fingerprint** is `sha256` over `update_chain`, `couplings`,
`schedule_families` and `init_family`, with every numeric value stripped out. Renaming a
local variable, reordering the `couplings` list, reformatting the file, or retuning any
hyperparameter cannot move it.

The **proximity** is a distance over the eight `hyper` values after log-quantisation. It exists
so that a structural match cannot be dodged by nudging digits.

You are a replay only if **both** hold: your structural fingerprint equals a corpus entry's,
**and** your proximity to that entry is at or under the bound floor. Run
`python3 environment/fingerprint_tool.py <your file>` to see exactly where you stand before
you spend accelerator time.

## The frozen axes

`environment/frozen_axes.json` records them. The locked script reports what it actually
loaded and built, and the verifier compares that report against the record. Widening the
batch, swapping the shard, changing the architecture, or charging two forward-backward passes
as one step all move it.
