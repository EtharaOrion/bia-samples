"""
Muon-derived optimizer: quintic NS 5-step eager + WSD schedule + momentum/WD tune.

Target: beat iter1/iter3 graded 3350. Iter1 peaked at 1.42 warmup 350 cf 0.65 min 0.04
mu 0.92 wd 0.015. Iter2 pushed to 1.55/250/0.60/mu0.90/wd0.01 and regressed to 3450
(+100). Iter3 at 1.50/300/0.62/mu0.93/wd0.012 recovered to 3350 (par).

Design for iter4 (final attempt) — exploit extended stable phase:
- peak 1.46 (vs 1.42/1.50): effective peak 0.0292 vs 0.0284/0.0300; +3% area vs
  iter1 without entering the unstable 1.55 regime that hurt iter2 late.
- warmup 320 (vs 350/300): slightly shorter ramp than iter1 to capture early
  area, but longer than iter2's 250 which injected excess variance.
- cooldown_frac 0.58 (vs 0.65/0.62): stable peak to step 1470 at max_steps 3500
  (vs 1225/1330), extends high-LR through the 3.50-3.35 valley where Muon
  benefits most (+140-245 high-LR steps).
- min_factor 0.05 (vs 0.04): tail LR at 3250 ~0.0033 vs 0.0021, preserves
  late slope; 0.04 was marginally too cold in 3100-3350 window, 0.02 was
  analytically 0.0011 and clearly insufficient.
- mu 0.95 / wd 0.01 (vs 0.92-0.93/0.012-0.015): higher momentum damps quintic
  orthogonalization noise under longer high-LR plateau; lower WD keeps
  effective regularization (lr*wd) flat despite higher area.
- NS 5-step quintic 3.4445/-4.7750/2.0315 bfloat16 eager;
  6-step gave no delta at 1500 and +20% matmuls.
- 1D / degenerate param fallback to plain momentum (no orthogonalization)
  to avoid assert on embeddings / head biases.
- Pure eager (no @torch.compile): inductor requires g++ / cc1plus which is
  absent on torch 2.13+cu130 hosts; eager is correct and harness budget is
  separate from probe budget so wall-clock is not scored.

WHAT WAS TRIED AND REJECTED THIS RUN
  paired control = iter1/iter3 graded 3350 (seeds 0,1 max_steps 3500)
  noise floor ~0.003 nats seed-mean (gate 0.004/sqrt(2)); single-seed probes quoted at ~0.002 nats indicative only
  peak 1.55 warmup 250 cf 0.60 mu 0.90 wd 0.01          graded 3450  delta +100 vs 3350 control — rejected: warmup too short + peak beyond stability, late plateau
  peak 1.50 warmup 300 cf 0.62 mu 0.93 wd 0.012         graded 3350  delta 0 vs control — rejected: no net gain vs iter1, shorter ramp canceled peak benefit
  quintic 6-step vs 5-step (same coeffs)               delta ~0 at 1500 probe, wall +20% — rejected compute cost without orthogonalization gain
  min_factor 0.02 vs 0.04                               analytic late LR at 3350 0.0011 vs 0.0021, probe indistinguishable at 1000 — rejected tail too cold
  cosine main phase (no stable plateau)                +0.015 at 1500 vs WSD — rejected loses high-LR area
  late beta 0.98 / mu 0.98                              +0.007 at 1000 probe — rejected late oscillations under decay
  anchored absolute schedule (boundaries fixed at 3500) killed before first tick (compile stall >154s with no telemetry) — rejected complexity, fractional schedule kept
  cubic NS 12-step vs quintic 5-step                   cubic ~1722s/1500 vs quintic eager, orthogonalization error tighter with quintic — rejected cubic on wall-clock and accuracy
  @torch.compile inductor on quintic NS                InductorError no g++ / cc1plus on torch 2.13+cu130 hosts — rejected hard decorator and import-time probe, kept pure eager
"""
import math
import torch
import torch.distributed as dist
from torch import Tensor


def _zeropower_quintic(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def _muon_update_2d(grad, momentum, mu=0.95):
    momentum.lerp_(grad, 1 - mu)
    upd = grad.lerp(momentum, mu)
    upd = _zeropower_quintic(upd)
    upd *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return upd


def _muon_update_1d(grad, momentum, mu=0.95):
    momentum.lerp_(grad, 1 - mu)
    upd = grad.lerp(momentum, mu)
    return upd


def _ws_rank():
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
    except Exception:
        pass
    return 1, 0


class _Derived(torch.optim.Optimizer):
    owns_schedule = True

    def __init__(self, params, lr=0.02, weight_decay=0.01, mu=0.95):
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, initial_lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.peak = 1.46
        self.warmup = 320
        self.cooldown_frac = 0.58
        self.min_factor = 0.05

    def set_progress(self, step, max_steps):
        warmup, peak, cf = self.warmup, self.peak, self.cooldown_frac
        stable_end = int(max_steps * (1 - cf))
        if step < warmup:
            eta = peak * (step + 1) / warmup
        elif step < stable_end:
            eta = peak
        else:
            denom = max(1, max_steps - stable_end)
            prog = min(max((step - stable_end) / denom, 0.0), 1.0)
            eta = peak * (self.min_factor + (1 - self.min_factor) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in self.param_groups:
            g["lr"] = g["initial_lr"] * eta

    @torch.no_grad()
    def step(self):
        ws, rank = _ws_rank()
        for group in self.param_groups:
            params = group["params"]
            if ws > 1:
                pad = (ws - len(params) % ws) % ws
                params_pad = params + [torch.empty_like(params[-1])] * pad if pad else params
            else:
                params_pad = params
            for base_i in range(0, len(params), ws):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    if p.ndim >= 2 and min(p.shape) > 1:
                        upd = _muon_update_2d(p.grad, state["momentum"], mu=group["mu"])
                    else:
                        upd = _muon_update_1d(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(upd, alpha=-group["lr"])
                if ws > 1:
                    dist.all_gather(params_pad[base_i:base_i + ws], params_pad[base_i + rank])


def build_optimizer(params, lr=0.02, **kwargs):
    if isinstance(lr, list):
        lr = 0.02
    clean = []
    for p in params:
        if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], str):
            clean.append(p[1])
        else:
            clean.append(p)
    objs = []
    for p in clean:
        if isinstance(p, torch.nn.Parameter):
            objs.append(p)
        else:
            objs.append(torch.nn.Parameter(p))
    return _Derived(objs, lr=lr, weight_decay=0.01, mu=0.95)
