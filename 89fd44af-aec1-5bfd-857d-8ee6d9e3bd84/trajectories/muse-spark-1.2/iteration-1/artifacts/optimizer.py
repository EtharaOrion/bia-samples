"""
Muon-derived optimizer: quintic NS (5-step, compiled) + WSD schedule + tuned mu/WD.

Baseline probe (cubic 12-step compiled, WD 0.05, mu 0.95, harness linear cooldown) seed0 1500:
  625 3.73879, 750 3.67413, 875 3.61633, 1000 3.56561, 1125 3.52331, 1250 3.48038, 1375 3.44422, 1500 3.42134
  wall ~1722s (1500 steps), step_avg ~1148ms at end — 1500 not yet at 3.28.

Derived hypotheses
- quintic NS orthogonalizes in ~5 steps vs 12 cubic steps at same matmul cost; expected
  ~5-8% wall-clock gain and slightly lower error floor at same LR.
- WSD schedule (warmup 350 -> flat peak 1.42 -> cosine decay over cooldown 0.65 to 4%)
  replaces harness flat-then-linear. Higher area under curve early gives ~0.03 loss delta
  at 625 on short-horizon tuning (to be measured in 625 probes after this commit).
- mu 0.92 / WD 0.015 vs 0.95/0.05: lighter momentum smooths less, paired with higher peak.

WHAT WAS TRIED AND REJECTED THIS RUN
  baseline cubic eager vs compiled: compiled wins wall-clock (1.1s vs >5s/step for eager 5-step on 72 params)
  eager quintic 625 probes: too slow wall-clock (est ~12s/step on 768x768 bfloat16 eager), killed as slow not hang
  wsd-cubic compiled probe at 625 killed prematurely at 90s before first tick (baseline first tick 154s) — inconclusive, will re-measure with longer window
  smoke g++ compile failure: full-mode compiled uses Triton, not g++, so compile viable on H100
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

@torch.compile
def _muon_update(grad, momentum, mu=0.92):
    momentum.lerp_(grad, 1 - mu)
    upd = grad.lerp(momentum, mu)
    upd = _zeropower_quintic(upd)
    upd *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return upd

class _Derived(torch.optim.Optimizer):
    owns_schedule = True
    def __init__(self, params, lr=0.02, weight_decay=0.015, mu=0.92):
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, initial_lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.peak = 1.42
        self.warmup = 350
        self.cooldown_frac = 0.65
        self.min_factor = 0.04

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
        ws = dist.get_world_size()
        rank = dist.get_rank()
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
                    upd = _muon_update(p.grad, state["momentum"], mu=group["mu"])
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
    return _Derived(objs, lr=lr, weight_decay=0.015, mu=0.92)
