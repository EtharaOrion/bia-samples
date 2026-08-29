# S08 checkpoint recovery: resume a stalled training run in the fewest steps

## Objective

You are given a mid-training checkpoint of a small language model. It carries the model weights and the optimizer state that the interrupted run had accumulated. Your job is to decide how to resume from it and to reach a fixed target validation loss in as few optimizer steps as possible. Everything about the model, the data, the batch, the evaluation and the step ceiling is frozen. The only thing you own is the recovery: how the optimizer that continues training is constructed, what state it starts from, and how its learning rate moves.

## What is frozen

The frozen surface is enforced by the harness and recorded on every telemetry record, so a run that moves any of it scores zero rather than scoring differently.

| Frozen | Value |
|---|---|
| Model architecture and its weights at resume | exactly what the checkpoint carries, byte for byte |
| Vocabulary, sequence length, depth, width, head count | as recorded in `frozen` in the fixture manifest |
| Batch size and data order | a pure function of the step index and the run seed, owned by the harness |
| Forward and backward passes per optimizer step | exactly one |
| Validation shard and evaluation protocol | harness owned, digest checked after the run |
| Step ceiling and evaluation cadence | `max_steps` and `eval_every` in the fixture manifest |
| Target loss | `target_loss` in the fixture manifest |

## What is free

The recovery. You choose the optimizer class, its hyperparameters, what if anything it inherits from the checkpoint's saved optimizer state, and the learning rate at every step. You may inspect every byte of the checkpoint before you decide.

## What you submit

Write two files into `/workspace/submission/`.

**`recover.py`** must define one entry point.

```python
def recover(model, ckpt_state, cfg, probe):
    """
    model      an nn.Module already carrying the checkpoint weights, on the run device.
               You must not change its weights here. The harness digests them before and
               after this call and a change scores zero.
    ckpt_state the raw dict the interrupted run's optimizer saved, with the usual
               'state' and 'param_groups' keys. Indices in 'state' line up with
               list(model.parameters()).
    cfg        a read-only mapping of the frozen operating point. Every value in it also
               appears in the agent-visible fixture manifest.
    probe      probe(n) runs n forward-backward passes on training batches drawn from
               before the checkpoint and returns a list of {'loss': float, 'grads': [...]}
               with one gradient tensor per model parameter, in parameter order. Probes
               are capped at cfg['probe_batches_max'] and EACH PROBE COSTS ONE STEP of the
               same budget your score is measured in.

    Return either an optimizer, or a pair (optimizer, lr_at) where lr_at(step) -> float
    gives the absolute learning rate at recovery step index `step`, counting from 1.
    """
```

**`report.json`** must carry at least these two fields.

| Field | Meaning |
|---|---|
| `steps_to_target` | The integer step count at which your run reached the target loss, derived by you from the harness telemetry. The verifier recomputes the same quantity independently and the two must agree exactly. |
| `optimizer` | An object declaring the hyperparameters your recovery put in force. `weight_decay` is required. Any of `lr`, `betas` and `eps` you also declare is checked. The verifier compares each declared field against the optimizer the harness fingerprinted when your `recover` returned, and the two must agree. Declare what you actually chose. |

A report that is absent, unparsable, omits a required field, or disagrees with the machine record scores zero.

## How you run

The harness runner is the only writer of the telemetry record. Logs you write by hand are not evidence and will not reconcile.

```sh
python3 /opt/bia/s08/runner/run_s08.py --submission /workspace/submission
```

It executes two phases in a fixed order. First a harness-owned control run resumes the checkpoint the obvious way and is sealed. Then your `recover` runs once per frozen seed. Both phases are written into the same telemetry record.

## How you are graded

The score is a single float on the closed interval from 0 to 1, higher is better, and it is not binary.

Let `submission_steps` be the earliest step at which the seed mean validation loss and every individual seed sit at or below `target_loss` and stay there at every later evaluated step. Probe passes are included in that count. Let `baseline_steps` be the same quantity for the harness control run, or the step ceiling when the control never reaches the target. Let `full_reward_at` be `ceil(target_fraction_of_control * baseline_steps)`.

```text
raw   = (baseline_steps - submission_steps) / (baseline_steps - full_reward_at)
score = min(max(raw, 0.0), 1.0)
```

The baseline is measured in your own graded attempt rather than read from a constant an author chose, so a score of zero means your recovery did no better than the control did on the same hardware in the same run.

Seventeen checkers gate the score and each one is a hard gate. A single failure emits exactly 0.0. They are declared in full, with the live-state read each one traces to, in `tests/checkers.yaml`, which you may read. In summary they cover three things.

**Your recovery did the work.** Your `recover` must actually change optimizer state rather than resume it unaltered, it must change the second moment the checkpoint carried rather than only its counters or its hyperparameters, it must not resume the checkpoint's stale optimizer step counter, and the crossing must sit inside the step ceiling.

**Your account is true.** Your reported step count must agree with the verifier's independent recomputation from the telemetry, and the optimizer hyperparameters you declare must be the ones the harness recorded in force.

**Nothing frozen moved.** The frozen contract must hold on every record; the control phase must be sealed before your first record; the model weights at resume must equal the checkpoint's; the validation and training shard digests must match the manifest; the checkpoint file, the fixture manifest and the two harness source files must carry the digests this bundle froze, at load and again at grading; the telemetry hash chain must recompute; the record set must cover every frozen seed on the frozen evaluation cadence; and the probe count charged must sit inside the declared allowance.

The frozen surface is not a trap to be picked at. Every route through it is checked deterministically, so the only path to a score is a recovery that reaches the target in fewer steps than the control did.

## What you are given

`environment/fixtures/` holds the checkpoint and its manifest. The manifest is public and it carries the frozen block, the checkpoint step, the checkpoint validation loss, the exact Bayes-optimal cross entropy of the validation shard, the target loss and the rule that derives it, and the validation curve the interrupted run recorded up to the moment the checkpoint was taken.

## Budget

The per-attempt budget is 7.2 minutes of single-H100 time, which is a six hour session divided by fifty attempts. The session ends after fifty attempts or six hours, whichever binds first, and the best attempt of the session is the one that counts. Time your own exploratory work against that and leave enough room for a graded run to finish, because an unfinished run produces no crossing and scores zero.

Egress is available during environment setup only. There is no network during scored work, so everything you need is already installed.
