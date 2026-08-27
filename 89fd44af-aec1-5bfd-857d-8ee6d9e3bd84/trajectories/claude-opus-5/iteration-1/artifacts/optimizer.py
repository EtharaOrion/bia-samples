"""Track-3 submission: DHOM -- Dual-Horizon Orthogonalized Momentum with
anticipatory innovation correction.

WHAT THIS IS, AND HOW IT DIFFERS FROM THE MUON FAMILY
-----------------------------------------------------
Muon-style rules build ONE exponential moving average of the gradient (optionally
with a Nesterov half-step), orthogonalize it via Newton-Schulz, and take a step of
fixed spectral size. Everything a published track-3 record varies -- the momentum
constant, the Newton-Schulz coefficients, the shape factor, the learning-rate
schedule, weight decay -- leaves that skeleton intact: the direction applied to a
matrix is always the polar factor of a single-timescale EMA of past gradients.

This rule changes the object that gets orthogonalized, so the direction applied to
the parameters differs from any single-EMA rule even when momentum, gradients and
warmth are matched:

  1. ANTICIPATORY INNOVATION CORRECTION. The fast branch is fed
        c_t = g_t + kappa * (g_t - g_{t-1})
     rather than g_t. The innovation g_t - g_{t-1} is a finite-difference estimate
     of how the gradient is moving under the trajectory, so c_t leads the raw
     gradient. This cancels part of the lag an EMA necessarily has; under Muon's
     spectral normalization it is safe to do without any clipping, because the
     step size is set by the polar factor and not by the magnitude of c_t.

  2. DUAL-HORIZON MIXTURE, MIXED *BEFORE* ORTHOGONALIZATION. A second, much
     slower EMA (b2, ~200-step horizon) runs alongside the fast one, and the two
     are combined -- after individual bias correction, which is what makes the
     mixture well defined -- into a single matrix that is then orthogonalized:
        M_t = (1 - w) * fast_t + w * slow_t,      O_t = polar(M_t).
     Because the polar factor is a nonlinear function of its argument, this is NOT
     equivalent to any reweighting of a single EMA, and it is not equivalent to
     mixing two orthogonalized updates either. The long branch supplies the part of
     the descent direction that survives averaging over hundreds of steps; the
     short branch keeps curvature-tracking responsiveness. Their mixture weight w
     is ramped in (an unwarmed slow branch is stale by construction) and eased back
     down during the final anneal, where a long memory would otherwise fight the
     decaying step size.

  3. Only then the usual polar factor and shape factor are applied, so the step
     retains Muon's spectral-norm geometry and its scale invariance.

The schedule is owned by this optimizer (`set_progress` / `owns_schedule`), as
instruction.md permits: short warmup, a long stable phase, a linear decay that has
done most of its work by `floor_at` of the run, and a small nonzero floor carried
to the end so training keeps making progress afterwards rather than freezing.

build_optimizer(params, lr=0.02, **kwargs) -> torch.optim.Optimizer
"""
from __future__ import annotations

import torch
from torch import Tensor

# --------------------------------------------------------------------------
# Configuration. Values here are the ones used for the graded run; the probe
# harness overrides entries of CFG in-process to sweep them.
# --------------------------------------------------------------------------
CFG = dict(
    lr_mult=1.0,        # peak lr = lr_mult * lr handed in by the harness
    b1=0.95,            # fast EMA decay
    b2=0.995,           # slow EMA decay (~200 step horizon)
    kappa=0.40,         # anticipatory innovation gain
    w_max=0.25,         # peak weight of the slow branch in the mixture
    w_end=0.10,         # slow-branch weight at the end of the anneal
    w_ramp=0.20,        # fraction of the run over which w ramps 0 -> w_max
    nesterov=True,
    warmup_steps=25,    # absolute steps of linear lr warmup
    sched="wsd",        # "wsd" (hold then linear) or "cbase" (frozen shape, compressed)
    decay_start=0.45,   # wsd: eta = 1 until this fraction of the run
    hold=0.30,          # cbase: fraction of the compressed horizon spent at eta = 1
    floor_at=0.82,      # eta reaches `floor` here (the anneal horizon)
    floor=0.10,         # lr floor, carried linearly to 0 at the end
    ns_steps=12,        # Newton-Schulz iterations
    wd=0.0,             # decoupled weight decay
    elem=0.0,           # weight of the SNR-shaped update (0 = pure polar factor)
    b3=0.95,            # second-moment decay for the SNR shaping
    elem_eps=0.15,      # damping, as a fraction of the mean per-entry scale
)


def _newtonschulz(X: Tensor, steps: int) -> Tensor:
    """Batched quintic Newton-Schulz iteration for the polar factor.

    Same quintic as the frozen trainer's reference (a, b, c = 2, -1.5, 0.5) run in
    bfloat16; batched over the leading dimension so all same-shape parameters are
    orthogonalized in one set of matmuls.
    """
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


class DualHorizonOrthogonal(torch.optim.Optimizer):
    """Dual-horizon orthogonalized momentum (see module docstring)."""

    owns_schedule = True

    def __init__(self, params, lr=0.02, cfg=None):
        cfg = dict(CFG if cfg is None else cfg)
        params = list(params)
        assert len(params) >= 1
        defaults = dict(lr=lr * cfg["lr_mult"], **cfg)
        super().__init__(params, defaults)

        self.cfg = cfg
        self.base_lr = lr * cfg["lr_mult"]
        self._progress = None          # (step, max_steps) published by the harness
        self._auto_step = 0            # fallback counter when nothing publishes it
        self._t = 0                    # number of updates applied

        # Group parameters by shape so the Newton-Schulz iteration -- by far the
        # dominant cost of this rule -- runs batched instead of once per tensor.
        self._v = {}
        self._groups = {}
        for g in self.param_groups:
            for p in g["params"]:
                self._groups.setdefault(tuple(p.shape), []).append(p)
        self._shape_scale = {
            shape: max(1.0, shape[-2] / shape[-1]) ** 0.5 for shape in self._groups
        }

    # ------------------------------------------------------------------
    # schedule
    # ------------------------------------------------------------------
    def set_progress(self, step: int, max_steps: int):
        self._progress = (int(step), int(max_steps))

    def _phase(self):
        if self._progress is not None:
            step, total = self._progress
        else:
            step, total = self._auto_step, None
        return step, total

    def eta(self, step: int, total):
        c = self.cfg
        warm = min(1.0, (step + 1) / max(1, c["warmup_steps"]))
        if total is None:                      # no horizon published: stay flat
            return warm
        p = step / total
        k = c["floor_at"]
        if c["sched"] == "cbase":
            # the frozen trainer's own cooldown shape, compressed onto [0, k] and
            # floored: identical to eta_for() when k == 1 and floor == 0.
            if p <= k:
                shape = 1.0 if p < c["hold"] * k else max(
                    c["floor"], (k - p) / max(1e-9, (1.0 - c["hold"]) * k))
            else:
                frac = min(1.0, (p - k) / max(1e-9, 1.0 - k))
                shape = c["floor"] * (1.0 - frac)
            return warm * shape
        if p <= c["decay_start"]:
            shape = 1.0
        elif p <= k:
            frac = (p - c["decay_start"]) / (k - c["decay_start"])
            shape = 1.0 + frac * (c["floor"] - 1.0)
        else:
            frac = min(1.0, (p - c["floor_at"]) / max(1e-9, 1.0 - c["floor_at"]))
            shape = c["floor"] * (1.0 - frac)
        return warm * shape

    def slow_weight(self, step: int, total):
        """Mixture weight of the long-horizon branch."""
        c = self.cfg
        if total is None:
            return c["w_max"]
        p = step / total
        ramp = min(1.0, p / max(1e-9, c["w_ramp"]))
        w = c["w_max"] * ramp
        if p > c["floor_at"]:                 # ease the long memory back out
            frac = min(1.0, (p - c["floor_at"]) / max(1e-9, 1.0 - c["floor_at"]))
            w = c["w_max"] + frac * (c["w_end"] - c["w_max"])
        return w

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        c = self.cfg
        step, total = self._phase()
        eta = self.eta(step, total)
        lr = self.base_lr * eta
        w = self.slow_weight(step, total)
        for g in self.param_groups:
            g["lr"] = lr
            g["slow_weight"] = w

        self._t += 1
        t = self._t
        b1, b2, kappa = c["b1"], c["b2"], c["kappa"]
        bc_fast = (b1 * (1.0 - b1 ** t) + (1.0 - b1)) if c["nesterov"] else (1.0 - b1 ** t)
        bc_slow = 1.0 - b2 ** t

        for shape, plist in self._groups.items():
            plist = [p for p in plist if p.grad is not None]
            if not plist:
                continue
            grads = [p.grad for p in plist]
            state = [self.state[p] for p in plist]
            for st, p in zip(state, plist):
                if "m1" not in st:
                    st["m1"] = torch.zeros_like(p)
                    st["m2"] = torch.zeros_like(p)
                    st["gprev"] = torch.zeros_like(p)
            m1 = [st["m1"] for st in state]
            m2 = [st["m2"] for st in state]
            gprev = [st["gprev"] for st in state]

            # anticipatory innovation correction: c_t = g_t + kappa (g_t - g_{t-1})
            if kappa != 0.0 and t > 1:
                corr = torch._foreach_sub(grads, gprev)
                torch._foreach_mul_(corr, kappa)
                torch._foreach_add_(corr, grads)
            else:
                corr = [g.clone() for g in grads]
            for gp, g in zip(gprev, grads):
                gp.copy_(g)

            # two horizons
            torch._foreach_lerp_(m1, corr, 1.0 - b1)
            torch._foreach_lerp_(m2, grads, 1.0 - b2)

            if c["nesterov"]:
                fast = torch._foreach_lerp(corr, m1, b1)
            else:
                fast = [x.clone() for x in m1]

            # bias-corrected mixture, formed BEFORE orthogonalization
            torch._foreach_mul_(fast, (1.0 - w) / bc_fast)
            if w > 0.0:
                torch._foreach_add_(
                    fast, torch._foreach_mul(m2, w / bc_slow)
                )

            M = torch.stack(fast)
            O = _newtonschulz(M, c["ns_steps"]).to(plist[0].dtype)

            if c["elem"] > 0.0:
                # SNR shaping. The polar factor spends the same step on every
                # coordinate; entries whose sign keeps flipping carry less usable
                # signal than entries that persist. Rescale each entry by its own
                # running RMS, then restore the Frobenius norm per matrix so the
                # step size -- and with it Muon's spectral geometry -- is untouched.
                v = self._v.get(shape)
                if v is None:
                    v = self._v[shape] = torch.zeros_like(O)
                b3 = c["b3"]
                v.mul_(b3).addcmul_(O, O, value=1.0 - b3)
                denom = (v / (1.0 - b3 ** t)).sqrt_()
                denom.add_(c["elem_eps"] * denom.mean(dim=(-2, -1), keepdim=True))
                A = O / denom
                nO = O.norm(dim=(-2, -1), keepdim=True)
                A.mul_(nO / (A.norm(dim=(-2, -1), keepdim=True) + 1e-12))
                if c["elem"] < 1.0:
                    A.mul_(c["elem"]).add_(O, alpha=1.0 - c["elem"])
                    A.mul_(nO / (A.norm(dim=(-2, -1), keepdim=True) + 1e-12))
                O = A

            O = O * self._shape_scale[shape]

            if c["wd"] != 0.0:
                torch._foreach_mul_(plist, 1.0 - lr * c["wd"])
            torch._foreach_add_(plist, list(O.unbind(0)), alpha=-lr)

        self._auto_step += 1
        return loss


def build_optimizer(params, lr=0.02, **kwargs):
    """Entry point used by both the training harness and the grader."""
    params = list(params)
    # Accept either bare parameters or (name, parameter) pairs.
    if params and isinstance(params[0], (tuple, list)):
        params = [p for _, p in params]
    cfg = dict(CFG)
    for k in list(cfg):
        if k in kwargs and kwargs[k] is not None:
            cfg[k] = kwargs[k]
    return DualHorizonOrthogonal(params, lr=lr, cfg=cfg)
