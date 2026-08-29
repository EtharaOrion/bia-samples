"""PRIVATE reference recovery for S08.

This is the oracle, not a hint. It is the recovery a solver reaches by diagnosing
the checkpoint's optimizer state instead of resuming from it, and it is the exact
file `solve.sh` installs as the submission.

What it does, in the order a diagnosis reaches it:

  1. It never adopts the checkpoint's `param_groups`. Stored hyperparameters are a
     record of what some earlier run chose, not a recommendation, and here they are
     wrong. A fresh AdamW is built with hyperparameters this file owns.
  2. It spends a small, bounded number of probe forward-backward passes to measure
     the live per-tensor gradient second moment. Those probes cost real budget, so
     the count is kept small.
  3. For every tensor it compares the checkpoint's second moment against the measured
     one and rescales the stored second moment so its mean matches what the gradients
     actually are. This repairs a per-tensor scalar corruption in either direction
     while preserving the per-element shape the pretraining run learned, which a
     blanket reset would throw away.
  4. It clips the first moment against the repaired preconditioner so no stale
     direction can produce an outsized first update.
  5. It writes a truthful step counter taken from the manifest's checkpoint step
     rather than from the counter stored in the state.
  6. It supplies its own warmup and cosine schedule keyed to the recovery step index,
     never to the counter carried in the checkpoint.
"""

from __future__ import annotations

import math

import torch

BASE_LR = 3.0e-3
BETAS = (0.9, 0.95)
EPS = 1.0e-8
WEIGHT_DECAY = 0.01
FIRST_MOMENT_CLIP_SIGMAS = 3.0


def _get(src, i):
    if i in src:
        return src[i]
    return src.get(str(i), {})


def recover(model, ckpt_state, cfg, probe):
    params = list(model.parameters())

    n_probe = min(8, int(cfg.get("probe_batches_max", 8)))
    probes = probe(n_probe) if n_probe > 0 else []
    acc = [torch.zeros_like(p) for p in params]
    for rec in probes:
        for a, g in zip(acc, rec["grads"]):
            a.add_(g.detach() * g.detach())
    if probes:
        for a in acc:
            a.div_(len(probes))

    opt = torch.optim.AdamW(
        params, lr=BASE_LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY
    )

    src = ckpt_state.get("state", {})
    true_step = float(int(cfg.get("checkpoint_step", 0)))

    for i, p in enumerate(params):
        st_src = _get(src, i)
        observed = float(acc[i].mean()) if probes else 0.0
        v_src = st_src.get("exp_avg_sq")
        if torch.is_tensor(v_src) and float(v_src.mean()) > 0.0 and observed > 0.0:
            v_new = v_src.detach().to(device=p.device, dtype=p.dtype) * (observed / float(v_src.mean()))
        elif observed > 0.0:
            v_new = acc[i].detach().clone()
        elif torch.is_tensor(v_src):
            v_new = v_src.detach().to(device=p.device, dtype=p.dtype).clone()
        else:
            v_new = torch.zeros_like(p)

        m_src = st_src.get("exp_avg")
        if torch.is_tensor(m_src):
            m_new = m_src.detach().to(device=p.device, dtype=p.dtype).clone()
            ceiling = FIRST_MOMENT_CLIP_SIGMAS * math.sqrt(max(float(v_new.mean()), 1e-30))
            peak = float(m_new.abs().max())
            if peak > ceiling > 0.0:
                m_new.mul_(ceiling / peak)
        else:
            m_new = torch.zeros_like(p)

        state = opt.state[p]
        state["step"] = torch.tensor(true_step, dtype=torch.float32)
        state["exp_avg"] = m_new
        state["exp_avg_sq"] = v_new

    max_steps = int(cfg.get("max_steps", 1000))
    warmup = max(5, min(50, max_steps // 40))

    hold_until = int(0.6 * max_steps)

    def lr_at(step: int) -> float:
        if step <= warmup:
            return BASE_LR * step / max(1, warmup)
        if step <= hold_until:
            return BASE_LR
        progress = (step - hold_until) / max(1, max_steps - hold_until)
        progress = min(1.0, max(0.0, progress))
        return BASE_LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    return opt, lr_at
