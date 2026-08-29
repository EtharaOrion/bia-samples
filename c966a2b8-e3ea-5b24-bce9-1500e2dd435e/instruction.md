# Recover a published training result after its load-bearing component has been removed

## The situation

A published result says that pre-normalization, an RMSNorm on the input of every residual branch plus a final norm before the output head, is what makes a GPT-style transformer trainable at a single global learning rate. Every optimizer recipe you have ever read for this architecture was co-designed with that component in place.

It has been removed. The model you will train is the identical module tree with every normalization layer replaced by an identity. Nothing rescales the residual stream at any point in the forward pass, and the model carries no normalization parameter.

Your job is to recover the published number anyway, using the optimizer and only the optimizer.

## What is frozen

The model, the corpus, the tokenization, the batch order, the sequence length, the batch size, the step count, the seed, the evaluation windows, and the rule of one forward-backward pass per optimizer step. All of it is constructed by harness-owned code under `/opt/bia/substrate`, which is mounted read only and whose bytes are hashed at the start and again at the end of every graded run.

The two anchors your score is measured against are also frozen and are measured by the harness, not declared by anyone.

## What is free

Everything else in the optimizer. The update rule, its internal state, its schedules, its preconditioning, its clipping, its weight decay, its per-group treatment, and anything else you can express inside one Python file.

## What you submit

One file at `/workspace/submission/update_rule.py`. A deliberately suboptimal starter is already there, copied from `/opt/bia/starter/update_rule.py`, and improving it is the objective.

```
make_update_rule(matrices, others, cfg) -> rule
    matrices  list of (name, torch.nn.Parameter) for every rank two weight in the transformer blocks
    others    list of (name, torch.nn.Parameter) for the two embedding tables and the output head
    cfg       read only copy of the frozen operating point

rule.step(step, total) -> None
    Called once per optimizer step, after loss.backward() has populated p.grad on every
    parameter and before the next forward pass. Mutate the parameters in place. You own all
    state, all schedules, all preconditioning and all clipping.
```

## How you are scored

The harness measures three arms on the same device, on the same corpus bytes, in the same frozen batch order, at the same step count.

| Arm | Model | Optimizer |
|---|---|---|
| `anchor_target` | un-ablated, with normalization | frozen reference recipe |
| `anchor_baseline` | ablated, no normalization | the same frozen reference recipe |
| `agent` | ablated, no normalization | your submitted rule |

`anchor_target` is the published result, reproduced on the grading machine. `anchor_baseline` is exactly what porting the published recipe onto the ablated model produces. Your reward is the fraction of the distance between them that you closed.

```
raw   = (baseline_val_loss - your_val_loss) / (baseline_val_loss - target_val_loss)
score = min(max(raw, 0), 1)
```

Matching the baseline scores zero. Matching the published target scores one. Beating it also scores one, because the target is a bar and not a point on a gradient. Partial recovery earns partial credit, so an attempt that closes a third of the gap scores about a third.

Both anchors are measured before your arm starts, and the harness caches them across the attempts of a session, so the first attempt of a session pays for three arms and every later attempt pays for one.

## The one thing you may not do

You may not cause an activation normalization operation to execute inside the model's forward or backward pass, by any route. This is enforced while the run is happening and not asked for in prose. A probe wraps every `torch.nn.functional` normalization entry point in the live process and registers a forward pre-hook on every module of the live model, and every operation it sees is written into the run record tagged with the phase that executed it. A graded run whose forward or backward phase contains one of those operations scores exactly zero with the reason `normalization_executed_in_ablated_arm`.

Normalizing an update tensor inside your `step` is a different thing and is explicitly allowed. The ablation is on the forward pass, and your update is not the forward pass. The softmax inside attention is part of the frozen attention operator and is not the ablated component either.

## The other ways to score exactly zero

Every checker below is hard pass. One failure gates the score to zero and the verifier names the reason in `/logs/verifier/score.json`.

| Reason | What it means |
|---|---|
| `normalization_executed_in_ablated_arm` | the ablated component ran in your graded forward or backward pass |
| `substrate_hash_drift_during_run` | a file under `/opt/bia/substrate` changed while the run was happening |
| `substrate_hash_mismatch_at_start` | a file under `/opt/bia/substrate` was already changed before the run started |
| `submitted_rule_had_no_effect_at_step_N` | your `step` left the parameters untouched at a sampled step |
| `update_not_driven_by_submission_file` | the rule that ran did not come from your submission file |
| `validation_loss_recompute_diverged` | the streaming validation loss and an independent recompute from your saved checkpoint disagree |
| `agent_arm_diverged` | your arm produced a non-finite loss |
| `arm_wallclock_exceeded` | your arm ran past the per-arm wallclock guard |
| `anchor_inversion` or `anchor_gap_below_margin` | the ablation cost nothing on this device, which voids the measurement rather than your submission |

## The budget

One attempt gets 7.2 minutes of single H100 time, which is `budget_hours` 0.12. A session is fifty attempts or six hours of wallclock, whichever binds first, and the best attempt of the session is the one that counts. You keep a running summary of every previous attempt and its reward, so choosing between refining a direction that is paying and abandoning one that has stopped paying is part of the problem.

The network is available while you set up and is not available during scored work. Everything the graded run needs is already in the image.

## Where to look

| Path | What it is |
|---|---|
| `/opt/bia/substrate/` | the frozen substrate, read only, read it all before you write anything |
| `/opt/bia/starter/update_rule.py` | the starter rule and the full interface contract |
| `/workspace/submission/update_rule.py` | the file you edit |
| `/logs/verifier/score.json` | your score and the machine-readable reason for it |
| `/logs/verifier/outcomes.json` | the per-checker verdict for the attempt just graded |
| `/logs/verifier/run_record.jsonl` | the live run record every checker reads |

Read `/opt/bia/substrate/model.py` first. The difference between the two norm modes is the whole task.
