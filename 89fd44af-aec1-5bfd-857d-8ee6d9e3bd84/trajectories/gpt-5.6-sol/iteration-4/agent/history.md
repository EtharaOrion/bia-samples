# Attempt 4 — your previous attempts at this task

You have attempted this task 3 times. This is attempt 4; this is your last attempt.

`reward`, `graded_step` and `seeds` below are recorded by the verifier and are not
open to dispute. `outcome` is a classification applied afterwards, and the prose under
each iteration is your own summarised account, written at the time of that attempt.

CURRENT budget is 8 hours of wall clock. Earlier accounts may quote a different figure or a
rule that has since changed; where they disagree with the task instruction you have
now, the instruction wins.

| iter | reward | graded_step | outcome (classified) | seeds |
|---|---|---|---|---|
| 1 | 0.4167 | 3250 | graded_pass | 2 |
| 2 | 0.3750 | 3275 | graded_pass | 2 |
| 3 | 0.4583 | 3225 | graded_pass | 2 |

## Iteration 1 — what you tried

- **Hypothesis:** That a compressed learning-rate schedule ending in a low-LR (0.08) refinement tail would drive the loss below the 3.28 target earlier than the control, cutting steps-to-target.
- **Mechanism:** A custom optimizer in submission/optimizer.py using a 'compressed schedule' family: an accelerated main phase followed by a refinement tail at 0.08 that takes over around step ~2500 to push monotone decrease past the crossing threshold. Exact update-rule internals are not stated in the account.
- **Hyperparameters:** Refinement-tail level 0.08 engaging near step ~2500; horizon ~3500 steps; validation on 2 seeds (0 and 1); frozen batch/architecture-pass contract; optimizer SHA-256 0374056d9e62...48e0c0.
- **Measured:** Seed 0: 3.6416@1000, 3.2790@3200, 3.27717@3225, 3.2640@3500. Seed 1: 3.6391@1000. At step 3250: 3.27542 and 3.27803, mean 3.276725, noise statistic 0.004632 (>0.004). Final: 3.26404, 3.26668. Claimed graded step 3250, self-estimated score 0.4167; 124 signed records.
- **Why it did not score higher:** The verifier container was misconfigured and never ran, so the run recorded no score. The retest graded the same submission at graded_step=3250, reward 0.4167 over 2 seeds.
- **Next attempt should change:** Beat graded_step 3250. Move the 3.28 crossing earlier while keeping both seeds individually below 3.28 and the noise statistic (3.28 - mean) * sqrt(n) >= 0.004.

## Iteration 2 — what you tried

- **Why it did not score higher:** Signed telemetry with a self-minted key (graded_valley_final), so the campaign chain could not authenticate it. Re-graded against that key: 0.375 at graded_step 3275 over 2 seeds -- a real measurement, but slower than attempt 1's 3250, so the score went down.
- **Next attempt should change:** Never set TRACK3_CHAIN_KEY for the graded run. Tail-LR tuning is exhausted (<0.002 at step 3125, ~0.006 needed) - change the update rule or main-phase schedule shape instead.

## Iteration 3 — what you tried

- **Why it did not score higher:** Per-neuron normalization of the MLP expansion update, validated on both seeds before shipping. Graded 0.4583 at step 3225 -- 25 steps faster than attempt 1's 3250 and the best result of the campaign.
- **Next attempt should change:** Beat graded_step 3225. Per-neuron normalization worked where tail-LR and schedule reshaping did not, because it changes the update every step rather than redistributing learning rate. Extend that direction.

## Now attempt the task again

Your best attempt is iteration 3 at reward 0.4583 (graded_step 3225). Its source is reproduced below.
EXPLOIT it -- start from that submission and improve on it -- or, if you judge the
approach exhausted, replace it with a materially different update rule.
Beat 0.4583. Submit a graded 2-seed run either way.

## Your previous submission (iteration 3)

Validation curve it produced:

```
seed0  0:10.82583  625:3.75208  1250:3.57872  1875:3.46154  2500:3.36152  2625:3.34109  2750:3.32194  2875:3.30773  3000:3.29564  3125:3.28395  3250:3.27406  3375:3.26664  3500:3.26289
seed1  0:10.82583  625:3.74983  1250:3.57970  1875:3.46222  2500:3.36312  2625:3.34269  2750:3.32337  2875:3.30946  3000:3.29756  3125:3.28579  3250:3.27607  3375:3.26850  3500:3.26472
```

This is the exact `submission/optimizer.py` you submitted. Modify it,
or replace it, but do not reimplement it blind.

```python
"""Track-3 optimizer: neuron-normalized Muon with compressed refinement."""

import torch


def _zeropower(g):
    x = g.bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(12):
        xx = x @ x.mT
        x = 2.0 * x + (-1.5 * xx + 0.5 * (xx @ xx)) @ x
    return x.mT if transposed else x


@torch.compile(dynamic=False)
def _direction(grad, momentum, beta):
    momentum.lerp_(grad, 1.0 - beta)
    u = _zeropower(grad.lerp(momentum, beta))
    u = u * max(1.0, grad.shape[0] / grad.shape[1]) ** 0.5
    return u / (u.norm(dim=1, keepdim=True) + 1e-7)


class Track3Optimizer(torch.optim.Optimizer):
    owns_schedule = True

    def __init__(self, params, lr, weight_decay=0.05):
        params = sorted(list(params), key=lambda p: tuple(p.shape), reverse=True)
        super().__init__(params, dict(lr=lr, base_lr=lr,
                                      weight_decay=weight_decay))

    def set_progress(self, step, max_steps):
        horizon = max_steps * (6.0 / 7.0)
        cooldown_start = 0.30 * horizon
        if step < cooldown_start:
            compressed = 1.0
        elif step < horizon:
            compressed = (horizon - step) / (horizon - cooldown_start)
        else:
            compressed = 0.0

        tail = 0.08 * max(0.0, (max_steps - step) / (max_steps - horizon))
        eta = max(compressed, tail)
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * eta

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            early = int(group.get("harness_step", 0)) < 125
            beta = 0.85 if early else 0.95
            step_lr = group["lr"] * (0.90 if early else 1.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum"] = torch.zeros_like(p)
                update = _direction(p.grad, state["momentum"], beta)
                p.mul_(1.0 - step_lr * group["weight_decay"])
                p.add_(update, alpha=-step_lr)
        return loss


def build_optimizer(params, lr=0.02, **kwargs):
    return Track3Optimizer(params, lr=lr)
```
