# Attempt 4 — your previous attempts at this task

You have attempted this task 3 times. This is attempt 4.

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
| 3 | 0.5000 | 3200 | graded_pass | 2 |

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

## Iteration 3 — what you tried

- **Hypothesis:** Adding NorMuon-style per-neuron (per-row) second-moment rescaling to the orthogonalized Muon update, plus landing the WSD cooldown earlier (80% of horizon), would deepen the loss curve enough to cross 3.28 sooner than iteration 2's 3275 steps.
- **Mechanism:** Muon core: Nesterov momentum (mu warmup 0.85->0.95), Newton-Schulz quintic (a,b,c=2,-1.5,0.5, 12 iters, bf16), per-matrix scale max(1,rows/cols)^0.5, decoupled LR-scaled weight decay. Added: EMA of row-wise mean squared update, divide O by sqrt(v_hat)+eps, then rescale back to original Frobenius norm (direction-only). Custom eta: hold 1.0 to 32%, linear down to floor 0.06 at 80%, then linear to 0 at horizon; optimizer owns schedule via harness_step/harness_max_steps.
- **Hyperparameters:** lr=0.025, weight_decay=0.05, _HOLD=0.32, _LAND=0.80, _FLOOR=0.06, mu 0.85->0.95 over 8.57% of run, NS steps=12, NorMuon beta2=0.95, eps=1e-8, default T=3500, batched buckets per shape, 2 seeds, max_steps 3500.
- **Measured:** Graded step 3200; step 3150 seeds 3.27953/3.28033 (mean 3.27993), 3175 3.27801/3.27865 (3.27833), 3200 3.27644/3.27714 (3.27679) vs margin 3.277172. End 3500: 3.26625/3.26700. Probes: per-neuron rescale +0.0045 nats; averaging -0.0063; wd 0.09 -0.021; lr x1.2 +0.0014 vs 0.0027 noise. Post-landing slope ~0.00007 nats/step.
- **Why it did not score higher:** Score 0.5 because graded_step 3200 > 2900; the loss curve near the target descends at only ~0.00007 nats/step, so the ~0.0046-nat improvement bought only ~75 steps, far short of the 600-step swing needed for full credit.
- **Next attempt should change:** Stop tuning the tail: the ~0.00007 nats/step terminal slope caps step savings. Target the mid-trajectory descent rate instead, e.g. a materially larger base lr with a matching longer hold plus preconditioner/second-moment changes swept on probes measuring slope, not just half-horizon depth.

## Now attempt the task again

Your best attempt is iteration 3 at score 0.5000 (graded_step 3200). Its source is reproduced below.
EXPLOIT it -- start from that submission and improve on it -- or, if you judge the
approach exhausted, replace it with a materially different update rule. Beat
0.5000. Submit a graded 2-seed run either way.

## Telemetry binding

The telemetry chain is append-only across every run in this container, and every
record must hash-match the optimizer.py present at grading time. If you probe with
one optimizer and then edit it, the probe's records will not match and the score
is 0.0. Send probe telemetry elsewhere with TRACK3_TELEMETRY_DIR and keep
/telemetry for the graded run only.

## Your previous submission (iteration 3)

Validation curve it produced:

```
seed0  0:10.82583  625:3.75333  1250:3.59032  1875:3.46682  2500:3.35775  2625:3.33584  2750:3.31552  2875:3.30031  3000:3.29036  3125:3.28116  3250:3.27398  3375:3.26866  3500:3.26625
seed1  0:10.82583  625:3.75455  1250:3.58856  1875:3.46383  2500:3.35745  2625:3.33640  2750:3.31618  2875:3.30100  3000:3.29102  3125:3.28191  3250:3.27476  3375:3.26941  3500:3.26700
```

This is the exact `submission/optimizer.py` you submitted. Modify it,
or replace it, but do not reimplement it blind.

```python
"""Track-3 submission: Muon with per-neuron update rescaling and an early-landing cooldown.

INTERFACE. The harness hands this optimizer the 72 two-dimensional parameters inside
`model.blocks` and keeps the embedding, the head and the scalar parameters on its own
frozen AdamW, which anneals linearly to zero at `max_steps`. Two thirds of the model is
ours and one third is not, and the schedule below is chosen with that in mind.

WHAT SCORE ACTUALLY MEASURES. Only the EARLIEST holding crossing of 3.28 counts; the loss
at the final step is worth nothing. A schedule tuned to minimise the endpoint is therefore
the wrong schedule, and the reference cooldown -- which places its entire benefit at the
last step -- leaves most of that benefit unusable.

Four things differ from the reference Muon in `train_gpt_track3.py`:

1. PER-NEURON RESCALING of the orthogonalised update (NorMuon-style, `_NORM`).
   Newton-Schulz equalises the SPECTRUM of the update but says nothing about how it is
   distributed across output neurons: rows whose updates have been persistently large stay
   large. A running second moment of each row rescales them, and the result is renormalised
   back to the Frobenius norm of the orthogonal update, so this changes the update's
   DIRECTION and never its size. That is what makes it composable with a fixed learning
   rate schedule rather than something that has to be re-tuned against it. Measured on a
   1750-step miniature of this harness, worth ~0.0045 nats, held steady from step 875 to
   the endpoint (3.37808 against 3.38243).

2. SCHEDULE. The submission declares `owns_schedule`, so the harness stops driving our
   learning rate and `eta` below drives it instead. eta holds at 1 through `_HOLD`, decays
   linearly to `_FLOOR` at `_LAND` = 80% of the horizon, then trails to zero over the tail.
   Landing the cooldown early buys depth exactly where the crossing lives instead of at
   step 3500: on the same miniature this is worth 0.0225 nats at 71% of the horizon against
   an otherwise identical run that lands at the end.

3. WEIGHT DECAY, decoupled and scaled by the live learning rate, as in the reference recipe
   (`Muon(..., lr=0.025, weight_decay=0.05)`). The harness's port of Muon drops it.

4. MOMENTUM WARMUP from `_MU0` to `_MU1`, so the first few hundred steps are not dominated
   by a stale average while the zero-initialised `proj` matrices are still growing.

WHAT WAS TRIED AND REJECTED. Iterate averaging -- the in-trajectory stand-in for the
eval-time weight blending that the published records use and this interface does not
expose -- was implemented as a base iterate z taking the full step with the model holding
p = z + b(x - z) for a running mean x. It loses: 0.0063 nats behind at 50% of the horizon
and 0.0144 behind at 57%, widening. The averaged point lags the base iterate by b*lam/(1-lam)
steps of drift, and because the harness necessarily takes its gradients at whatever the
parameter tensor holds, that lag feeds back into z as well. Feeding the mean back into the
step instead (p <- p + b(x - p)) is worse still: it low-passes the systematic drift along
with the noise. Eval-time blending works precisely because it does neither, which is the
part this interface cannot express.

The orthogonalisation is the reference Newton-Schulz quintic with the reference coefficients
and iteration count, batched over same-shape parameters so the 72 matrices become 3 batched
matmul chains. That is a throughput change only; it reproduces the reference update to bf16
rounding.
"""
from __future__ import annotations

import torch

# --- schedule -------------------------------------------------------------
_HOLD = 0.32      # end of the constant-eta phase
_LAND = 0.80      # eta reaches _FLOOR here, then trails linearly to zero at the horizon
_FLOOR = 0.06
# --- update rule ----------------------------------------------------------
_MU0 = 0.85       # momentum at step 0
_MU1 = 0.95       # momentum after warmup
_MU_WARM = 0.0857
_NS_STEPS = 12
_WD = 0.05
_NB2 = 0.95       # second-moment decay for the per-neuron rescaling
_NEPS = 1e-8
_DEFAULT_T = 3500


def zeropower_via_newtonschulz5(X: torch.Tensor, steps: int = _NS_STEPS) -> torch.Tensor:
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

    def __init__(self, params, lr=0.025, weight_decay=_WD):
        params = list(params)
        assert len(params) >= 1
        super().__init__(params, dict(lr=lr))
        self.owns_schedule = True          # tells the harness to stop setting lr
        self.base_lr = lr
        self.wd = weight_decay
        self._t = 0
        self._T = _DEFAULT_T
        # one batched bucket per distinct shape:
        # 48 x (768,768), 12 x (3072,768), 12 x (768,3072)
        buckets = {}
        for p in params:
            buckets.setdefault(tuple(p.shape), []).append(p)
        self._buckets = [dict(params=ps, scale=max(1.0, s[0] / s[1]) ** 0.5,
                              mom=None, v=None)
                         for s, ps in buckets.items()]

    def set_progress(self, step, max_steps):
        """Harness-published progress. Also marks this optimizer as schedule-owning."""
        self._t = int(step)
        self._T = int(max_steps)

    @staticmethod
    def eta(progress: float) -> float:
        if progress < _HOLD:
            return 1.0
        if progress < _LAND:
            return 1.0 + (_FLOOR - 1.0) * (progress - _HOLD) / (_LAND - _HOLD)
        if progress >= 1.0:
            return 0.0
        return _FLOOR * (1.0 - (progress - _LAND) / (1.0 - _LAND))

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
        mu = _MU0 + (_MU1 - _MU0) * min(1.0, progress / _MU_WARM)

        for bk in self._buckets:
            ps = bk["params"]
            G = torch.stack([p.grad for p in ps])
            if bk["mom"] is None:
                bk["mom"] = torch.zeros_like(G)
            M = bk["mom"]
            M.lerp_(G, 1.0 - mu)
            U = G.lerp_(M, mu)                              # Nesterov
            O = zeropower_via_newtonschulz5(U).float()

            # Per-neuron rescaling. Direction-only: the Frobenius norm of the orthogonal
            # update is restored afterwards, so the step size still comes from lr alone.
            if bk["v"] is None:
                bk["v"] = torch.zeros(O.shape[0], O.shape[1], 1,
                                      device=O.device, dtype=O.dtype)
            v = bk["v"]
            v.mul_(_NB2).add_(O.pow(2).mean(dim=-1, keepdim=True), alpha=1.0 - _NB2)
            n0 = O.norm(dim=(-2, -1), keepdim=True)
            O = O / ((v / (1.0 - _NB2 ** (self._t + 1))).sqrt() + _NEPS)
            O = O * (n0 / (O.norm(dim=(-2, -1), keepdim=True) + 1e-12))

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
