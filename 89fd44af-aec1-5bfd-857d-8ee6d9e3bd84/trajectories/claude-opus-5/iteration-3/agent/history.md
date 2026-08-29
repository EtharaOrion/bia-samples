# Attempt 3 — your previous attempts at this task

You have attempted this task 2 times. This is attempt 3.

`score`, `graded_step` and `seeds` below are recorded by the verifier and are not
open to dispute. `outcome` is a classification applied afterwards, and the prose under
each iteration is your own summarised account, written at the time of that attempt.

CURRENT budget is 8 hours of wall clock. Earlier accounts may quote a different figure or a
rule that has since changed; where they disagree with the task instruction you have
now, the instruction wins.

| iter | score | graded_step | outcome (classified) | seeds |
|---|---|---|---|---|
| 1 | not graded | - | agent_abandoned_run | 1 |
| 2 | 0.3750 | 3275 | graded_pass | 2 |

## Iteration 1 — what you tried

- **Hypothesis:** A dual-horizon momentum mixture (fast+slow EMA blended before Newton-Schulz) plus an anticipatory innovation correction and a retargeted WSD anneal landing just before the 2875 eval would cross val loss 3.28 sooner than Muon's 3500 steps.
- **Mechanism:** Muon variant: corr = g + kappa*(g - g_prev); fast EMA m1 (b1) with Nesterov, slow EMA m2 (b2), bias-corrected and mixed with weight w(step) ramping w_max->w_end; stack per-shape, 12-iter batched bf16 Newton-Schulz quintic polar factor, shape scale max(1,rows/cols)**0.5, then apply. owns_schedule=True with a WSD schedule (decay_start 0.45, hold 0.30, floor 0.10 from floor_at 0.82, anneal landing step 2870), 25-step warmup. An optional per-entry SNR shaping branch was implemented but disabled.
- **Hyperparameters:** lr_mult=1.0, b1=0.95, b2=0.995, kappa=0.40, w_max=0.25, w_end=0.10, w_ramp=0.20, nesterov=True, warmup_steps=25, sched=wsd, decay_start=0.45, hold=0.30, floor_at=0.82, floor=0.10, ns_steps=12, wd=0.0, elem=0.0; run: seeds 0,1, max-steps 3500
- **Measured:** Control A (rule off, new schedule) vs frozen schedule: -0.011 nats at p=0.71, -0.009 at p=0.86, +0.004 at p=1.0. Dual-horizon at w_max=0.40 cost +0.0148/+0.0142/+0.0144/+0.0145 nats at steps 400/500/600/700. elem=1.0 and elem=0.25 both NaN by step 100. Earlier lr_mult=1.5: +0.076 at 700 steps. Probe noise ~+/-0.01 nats.
- **Why it did not score higher:** Verifier recorded telemetry_not_bound_to_submission_step_0 with graded_step null and n_seeds 1: the graded 2-seed run was still training at summary time, so no finished, telemetry-bound submission existed to grade.
- **Next attempt should change:** Launch the full 2-seed 3500-step graded run early enough to finish and reconcile within the 8-hour budget, doing exploratory probes only in whatever time remains, rather than spending most of the window on 700-step ablations.

## Iteration 2 — what you tried

- **Hypothesis:** Moving Muon's LR cooldown landing later (2875 -> 3130) would deepen the anneal enough to make validation loss cross 3.28 earlier than run 1's 3175/3250.
- **Mechanism:** Muon variant that owns its own LR schedule: front-loaded cooldown with configurable landing step, decoupled LR-scaled weight decay 0.05, and momentum warmup 0.85->0.95 over the first 8.6% of the horizon; run 2 only shifted the cooldown landing right by 255 steps.
- **Hyperparameters:** owns_schedule=True; cooldown landing step 3130 (run 1: 2875); weight decay 0.05 decoupled, LR-scaled; momentum 0.85->0.95 over first 8.6% of horizon; 2 seeds; horizon 3500; 64x1024 batch, 524288 tokens, 1 fwd-bwd/step.
- **Measured:** Run 1 graded_step 3250 (score 0.4167), landing 2875, loss 3.30493 at 2875, crossing 3175. Run 2 graded 3275: seed 0 crossed 3225, seed 1 at 3250; at step 2500 run 2 was +0.0218 above run 1. Anneal-depth gain ~0.00007/step vs floor-crawl ~0.000084/step. Muon reference 3500.
- **Why it did not score higher:** The shipped run used the later landing (3130), which is just the same schedule shifted right; the displaced floor-crawl rate slightly exceeded the anneal-depth gain, so the graded crossing moved to 3275, 25 steps worse than the earlier-landing configuration.
- **Next attempt should change:** Ship the earlier landing (~2875) and spend the remaining effort on base LR magnitude or decay shape rather than the landing fraction, which the probes showed is nearly a dead knob.

## Now attempt the task again

Your best attempt is iteration 2 at score 0.3750 (graded_step 3275). Its source is reproduced below.
EXPLOIT it -- start from that submission and improve on it -- or, if you judge the
approach exhausted, replace it with a materially different update rule. Beat
0.3750. Submit a graded 2-seed run either way.

## Telemetry binding

The telemetry chain is append-only across every run in this container, and every
record must hash-match the optimizer.py present at grading time. If you probe with
one optimizer and then edit it, the probe's records will not match and the score
is 0.0. Send probe telemetry elsewhere with TRACK3_TELEMETRY_DIR and keep
/telemetry for the graded run only.

## Your previous submission (iteration 2)

Validation curve it produced:

```
seed0  0:10.82583  625:3.76164  1250:3.59594  1875:3.47866  2500:3.38482  2625:3.36494  2750:3.34592  2875:3.32693  3000:3.30757  3125:3.28887  3250:3.27662  3375:3.26848  3500:3.26468
seed1  0:10.82583  625:3.76039  1250:3.59649  1875:3.48023  2500:3.38752  2625:3.36716  2750:3.34809  2875:3.32878  3000:3.30947  3125:3.29085  3250:3.27851  3375:3.27046  3500:3.26662
```

This is the exact `submission/optimizer.py` you submitted. Modify it,
or replace it, but do not reimplement it blind.

```python
"""Track-3 submission: Muon-class update with a front-loaded, early-landing cooldown.

The harness hands this optimizer the 72 two-dimensional parameters inside
`model.blocks` and keeps the embedding, the head and the scalar parameters on its
own frozen AdamW, which anneals linearly to zero at the end of the run. The rule
below is built around that asymmetry.

Three things differ from the reference Muon:

1. SCHEDULE (the dominant term). The submission declares `owns_schedule`, so the
   harness stops driving the learning rate and this file drives it instead. The
   baseline holds eta at 1 for the first 30% of the horizon and then decays
   linearly to zero at the very last step, which places the whole benefit of the
   cooldown at step 3500. Since score depends only on the EARLIEST holding
   crossing of 3.28 and not at all on the loss at 3500, the cooldown is moved
   forward: eta decays from step 0.3229*T to a small floor at 0.8943*T (step 3130),
   and the rest of the run sits at that floor decaying to zero. Measured on a
   faithful 1000-step miniature of this same harness (identical fractional
   schedule for both optimizers, so the aux/Muon relationship is preserved), this
   is not a trade: it is lower than the baseline at every logged step from 375
   onward, including the endpoint.

2. WEIGHT DECAY 0.05, decoupled and scaled by the live learning rate, exactly as
   in the reference `train_gpt_track3.py` recipe. The harness port of Muon drops
   it. It is worth ~0.025 nats by the end of the miniature.

3. MOMENTUM WARMUP from 0.85 to 0.95 over the first 8.57% of the horizon, which
   keeps the first few hundred steps from being dominated by a stale average
   while the zero-initialised `proj` matrices are still growing.

The orthogonalisation is the reference Newton-Schulz quintic with the reference
coefficients and iteration count, batched over same-shape parameters so that the
72 matrices become 3 batched matmul chains. That is a throughput change only; it
reproduces the reference update to bf16 rounding.
"""
from __future__ import annotations

import torch

# --- schedule -------------------------------------------------------------
P1 = 0.322857   # end of the constant-eta phase (step 1130); ramp slope identical to run 1
P2 = 0.894286   # the cooldown lands at step 3130, where run 1 showed the crossing lives
FLOOR = 0.06     # eta at P2, then linear to zero over the tail
# --- update rule ----------------------------------------------------------
MU0 = 0.85       # momentum at step 0
MU1 = 0.95       # momentum after warmup
MU_WARM = 0.0857
NS_STEPS = 12
WD = 0.05
DEFAULT_T = 3500


def zeropower_via_newtonschulz5(X: torch.Tensor, steps: int = NS_STEPS) -> torch.Tensor:
    """Batched polar factor. Reference coefficients, reference iteration count."""
    a, b, c = 2.0, -1.5, 0.5
    X = X.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Track3Muon(torch.optim.Optimizer):
    """Muon on the block matrices, owning its own learning-rate schedule."""

    def __init__(self, params, lr=0.025, weight_decay=WD):
        params = list(params)
        assert len(params) >= 1
        super().__init__(params, dict(lr=lr))
        self.owns_schedule = True          # tells the harness to stop setting lr
        self.base_lr = lr
        self.wd = weight_decay
        self._t = 0
        self._T = DEFAULT_T
        # one batched bucket per distinct shape:
        # 48 x (768,768), 12 x (3072,768), 12 x (768,3072)
        buckets = {}
        for p in params:
            buckets.setdefault(tuple(p.shape), []).append(p)
        self._buckets = [dict(params=ps, scale=max(1.0, s[0] / s[1]) ** 0.5, mom=None)
                         for s, ps in buckets.items()]

    def set_progress(self, step, max_steps):
        """Harness-published progress. Also marks this optimizer as schedule-owning."""
        self._t = int(step)
        self._T = int(max_steps)

    @staticmethod
    def eta(progress: float) -> float:
        if progress < P1:
            return 1.0
        if progress < P2:
            return 1.0 + (FLOOR - 1.0) * (progress - P1) / (P2 - P1)
        if progress >= 1.0:
            return 0.0
        return FLOOR * (1.0 - (progress - P2) / (1.0 - P2))

    @torch.no_grad()
    def step(self, closure=None):
        g0 = self.param_groups[0]
        if "harness_max_steps" in g0:          # authoritative when the harness publishes it
            self._T = int(g0["harness_max_steps"])
            self._t = int(g0["harness_step"])
        T = max(1, self._T)
        progress = min(1.0, self._t / T)

        lr = self.base_lr * self.eta(progress)
        for g in self.param_groups:
            g["lr"] = lr
        mu = MU0 + (MU1 - MU0) * min(1.0, progress / MU_WARM)

        for bk in self._buckets:
            ps = bk["params"]
            G = torch.stack([p.grad for p in ps])
            if bk["mom"] is None:
                bk["mom"] = torch.zeros_like(G)
            M = bk["mom"]
            M.lerp_(G, 1.0 - mu)
            U = G.lerp_(M, mu)                              # Nesterov
            O = zeropower_via_newtonschulz5(U).float()
            if lr > 0.0:
                decay = 1.0 - lr * self.wd
                for p, o in zip(ps, O):
                    p.mul_(decay).add_(o, alpha=-lr * bk["scale"])

        self._t += 1
        return None


def build_optimizer(params, lr=0.025, **kwargs):
    ps = []
    for x in params:
        if isinstance(x, (tuple, list)) and len(x) == 2 and torch.is_tensor(x[1]):
            ps.append(x[1])          # (name, parameter) convention
        else:
            ps.append(x)
    return Track3Muon(ps, lr=lr)
```
