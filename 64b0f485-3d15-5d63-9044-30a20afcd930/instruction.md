# Optimizer recipe displacement, track 3

You are given a locked training script that reaches a target validation loss. Return an
optimizer recipe that reaches the same loss in **fewer optimizer steps**, and that is not a
recipe already present in the published record corpus.

## What you may change, and what you may not

| | |
|---|---|
| **Frozen** | the dataset, the batch size, the model architecture, and the rule of exactly one forward-backward pass per optimizer step |
| **Free** | the optimization algorithm, its hyperparameters and their schedules, the model initialization |

The frozen axes are recorded in `environment/frozen_axes.json`. A run that moves any of them
is refused with the reason `frozen-axis-moved`; it is not scored lower, it is refused.

## The quantity that is graded

Read this section before you optimize anything, because the graded quantity is stated here
rather than left for you to infer.

**The graded quantity is the number of optimizer steps at the first evaluation point, on the
grid the verifier itself schedules, whose loss falls below 3.28 and stays below 3.28 across
the next 3 scheduled evaluation points.**

Five things follow from that sentence, and every one of them is enforced.

1. **The verifier computes the crossing itself.** The graded step comes from the verifier's
   own evaluation of the model on a held-out split you never see. It is never a number your
   training loop reported, never a number in your stdout, and never a field you wrote. There
   is no file you can write that becomes the graded number.
2. **Nothing on the graded path is smoothed.** The graded readout is the raw evaluation at
   that step. You are welcome to EMA-blend, average or filter your own reported loss for your
   own diagnostics; the graded quantity is recomputed unsmoothed by the verifier, and a
   submission-reported value that reaches the graded path scores zero with the reason
   `readout-smoothed-on-the-graded-path`.
3. **A crossing must be sustained.** One favourable evaluation is a sample, not a crossing.
   The target must hold at the graded step **and** at the next 3 evaluation points the
   verifier schedules. A dip that rises back scores zero with `crossing-not-sustained`.
4. **Stopping early is not crossing.** A run that halts the moment an evaluation lands under
   the target is graded as **not having crossed**, and scores zero with
   `run-halted-before-sustain-window-closed`. It is not recorded as an absent result.
5. **The weights evaluated are the ones the harness owns.** At every graded point the verifier
   evaluates the checkpoint its own harness wrote at that step. A checkpoint you selected
   scores zero with `graded-weights-not-harness-owned`.

Optimizing the graded quantity means lowering the first sustained crossing step. There is no
second thing to optimize.

## The record corpus, and why replaying one scores nothing

This environment treats the published record lineage as a public evaluation set. A submission
whose update rule is one of the recipes in the pinned corpus is **rejected before any seed
runs**, at zero accelerator cost, with the reason `recipe-replays-published-record`.

The screen is deterministic and it is not a secret. `environment/published_records.json` lists
the pinned corpus and the structural fingerprint of every entry, and
`environment/fingerprint_tool.py` computes the same fingerprint over your own recipe so you
can check yourself before you spend a single step:

```sh
python3 environment/fingerprint_tool.py /workspace/submission/recipe.py
```

Two comparisons decide it, and both must agree before a submission is called a replay:

- the **structural fingerprint**, taken over the algebra alone with every numeric value
  stripped out, so renaming, reordering and retuning cannot move it;
- the **proximity**, a distance over log-quantised hyperparameters, so nudging a learning rate
  by a few percent does not buy you past the screen.

A recipe that shares a corpus entry's algebra but sits genuinely far from it in hyperparameter
space is a derivation and is allowed. A recipe that shares the algebra and sits inside the
bound floor is a recital and is not. Change the algorithm, not the digits.

Note also that a clearance is not permanent. The corpus is pinned at a revision, and the
revision the verifier grades against is the one in force at grading time, not the one you
screened against earlier in the session.

## What you submit

Write `/workspace/submission/recipe.py`. It must contain, as a **module-level literal**, a
`RECIPE` dictionary with these keys:

```python
RECIPE = {
    "update_chain":      [{"primitive": "<name>", "role": "<parameter group>"}, ...],
    "couplings":         [["<group>", "<group>", "<coupling kind>"], ...],
    "schedule_families": [["<parameter>", "<family>"], ...],
    "init_family":       "<name>",
    "hyper": {
        "lr_peak": 0.0, "momentum": 0.0, "second_moment_beta": 0.0, "weight_decay": 0.0,
        "warmup_frac": 0.0, "orthogonalisation_steps": 0, "trust_region_rho": 0.0,
        "aux_lr_ratio": 0.0,
    },
}
```

It must also define `build_optimizer(param_groups, total_steps)` returning the per-step update
plan the frozen script consumes. See `environment/recipe_contract.md` for the full contract
and `environment/train_baseline.py` for the locked script and the baseline recipe.

`RECIPE` must be a literal. The verifier lifts it out of your file's bytes without importing
or executing them, so a `RECIPE` that can only be obtained by running your code does not
screen at all and scores zero.

## Scoring

`reward = min(max((baseline_metric - agent_metric) / (baseline_metric - target_metric), 0.0), 1.0)`

where `agent_metric` is the graded step defined above. Both anchors are held by the
verifier and neither is published on this surface, so there is no number to aim at. The
shape is what you plan against: the reward rises continuously as the graded step falls,
every step you remove is paid for at the same rate, and once the target is reached the
reward holds at 1.0 rather than continuing to climb, because the target is a bar rather
than a point on a gradient. The reward is a single float on the
closed interval and is never a pass or fail flag. Every zero carries a machine-readable
reason in `/logs/verifier/reward.json`.

You have open internet access inside the sandbox for packages, datasets and models. The
verifier runs with egress denied. Fetching a record write-up is not blocked and is not the
point; the fingerprint screen decides whether what you submit is a replay, and no amount of
reading changes that.
