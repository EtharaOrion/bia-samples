"""Private reference update rule for the S10 slot.

This file is the oracle. It is never mounted into the agent's container and it
is never referenced by instruction.md. solve.sh copies it into the submission
path and runs the ordinary graded pipeline, so the reference travels the exact
route a submission travels.

This derivation replaces an earlier one, and it replaces it because the earlier
one was measured and lost. The earlier reference argued from published precedent
that a layerwise trust ratio, adaptive gradient clipping and decoupled weight
decay would restore from the update what the ablation removed from the forward
pass. Measured on this substrate that rule landed at 2.587 nats against a naive
port baseline of 2.013 nats, which is 0.573 nats worse than the thing it was
supposed to improve on. Everything below is what the measurement supports
instead. Every number quoted is a validation loss in nats at the frozen 1200
step operating point on one H100, produced by the substrate's own run_arm on the
ablated build.

Step one. What the ablation actually removes, measured rather than assumed.

A 200 step instrumented run of the frozen reference recipe on both builds
records the root mean square of the residual stream at the output of every
block. On the un-ablated build the profile is flat with depth: block 0 sits at
3.44 and block 9 at 4.09, because the pre-normalization on each branch input
rescales the stream before it is read. On the ablated build the same profile is
exponential with depth: block 0 sits at 0.055 and block 9 at 0.828, a factor of
fifteen across ten blocks. The squared-ReLU MLP makes each branch output grow
faster than linearly in the scale of the stream it reads, so on the ablated
build depth is a multiplier with no brake on it.

Step two. What that costs the optimizer, measured on the anchor recipe itself.

The published learning rate of 1.0e-3 sits on the ablated build's stability
cliff. Three runs of the published configuration on this device returned 2.013,
2.085 and a non-finite loss. Raising the learning rate to 1.5e-3 under the
published schedule diverges, and 2.0e-3 diverges. There is no headroom left to
spend, which is why a naive port of the published recipe cannot be improved by
turning the learning rate up.

Step three. Three families of scale-restoring update rule, measured and
rejected. A layerwise trust ratio in the LARS and LAMB line gives every layer
the same relative step: at 3.0e-3 relative it measures 2.587, at 1.0e-2 it
measures 2.054, and at 2.0e-2 it diverges, so no stable setting of it beats the
naive port. Spectrally normalized momentum in the Muon line measures 2.585 at
1.0e-3, 2.206 at 3.0e-3, and diverges at 2.0e-2. Rescaling every block matrix
back to its initial norm after each update measures 2.226 at 1.0e-3 and diverges
at 2.0e-3. All three are scale controls, all three are the right kind of idea,
and on this substrate none of them beats 2.013.

Step four. What does raise the ceiling. Lengthening the warmup from 5 percent to
15 percent of the run and halving the global gradient-norm clip from 1.0 to 0.5
buys enough headroom to run at 1.5e-3. Neither half works alone: the schedule
change at the published learning rate measures 2.127, worse than the baseline,
because a longer warmup spends steps a 1200 step run does not have; and the
learning rate rise under the published schedule diverges. Together at 1.5e-3
they measure 1.840, 1.848, 1.855 and 1.865, and diverge in two runs out of six.
1.75e-3 diverges. So the ceiling moved, but the arm is still metastable, and a
reference that dies in a third of its runs is not a reference.

Step five. What the divergence actually is, instrumented to the step. Weight
norms are not running away: over the 25 steps before a death the largest
parameter norm moves only from 11.39 to 11.92. The activations are. The maximum
absolute activation at the last block sits near 1.5, jumps to 1.3e3 on one batch
and recovers, drifts through 22 and 13 over the next twenty steps, then reaches
1.2e9 on a single batch, at which point the global gradient norm is 1.8e10
against a running scale of about 0.6, and the following forward pass is
non-finite. This is an input-driven activation cascade in a squared-ReLU MLP
with nothing downstream to absorb it, which is precisely the job the ablated
normalization was doing. It is not a weight-scale runaway, which is why the
three scale-restoring families above could not fix it.

Step six. The compensation this reference ships, which follows from step five.
The optimizer cannot stop the cascade in the forward pass, because the forward
pass is not its surface. It can refuse to write the cascade into the weights.
The rule detects a global gradient norm above 500 times its own running scale,
restores every parameter from a rollback point 25 steps back, zeroes the first
moment so momentum cannot re-drive the same direction, applies a 50 step linear
learning-rate cooldown, and continues. Measured over three runs at 1.5e-3 it
returns 1.855, 1.827 and 1.928, all finite.

Four variants of that guard were measured and all four are worse. Skipping the
offending step without rolling back returns 1.868, 1.873, 1.988 and one run that
ended at 3.9e17. A five step rollback without the momentum reset returns 1.772,
1.833, 1.882 and one run that ended at 1.9e10. A 25 step rollback that clears
the whole optimizer state instead of just the first moment returns 1.872, 1.907,
1.909 and 2.106. Tightening the detector from 500 to 100 times the running scale
returns 1.997 and 2.061, which is the guard rejecting so many ordinary steps that
it gives back the whole gain. The shipped combination is the only one of the five
in which every measured run stayed finite.

Step seven. Where inside the stable band to sit. With the guard in place the
learning rate is no longer bounded by divergence, so it becomes a choice about
where the reference should land between the two anchors, and the reference
should not crowd either end. At 1.5e-3 the guarded rule measured 1.810, 1.813,
1.827, 1.855, 1.862, 1.902 and 1.928 over seven runs, which is 0.38 to 0.91 of
the measured gap, and the top of that range leaves only 0.02 nats before the
reward clamps at one and stops being able to rank a better submission above the
reference. At 1.4e-3 it measured 1.896, 1.902, 1.908 and 1.962 over four runs,
which is 0.23 to 0.52 of the gap, with about 0.05 nats of room below and 0.11
nats above. The reference ships at 1.4e-3 for that reason and not because it is
the better number: 1.5e-3 scores higher, and a reference whose job is to be a
usable middle of the range should not take the higher score.

None of this reintroduces the ablated component. Every operation here reads
gradients and writes parameters inside the update phase, and none of it runs
inside the model's forward or backward pass, which is what the harness probe
watches. The rollback buffer is a copy of the parameters the optimizer already
owns, not a second model.
"""

from __future__ import annotations

import math

import torch

# Every value here is an operating point a measurement fixed. The comment on
# each names the measurement.
HP = {
    # 1.4x the published 1.0e-3. Usable only because of warmup_frac, grad_clip
    # and the rollback guard below: under the published schedule this rate
    # diverges. Chosen for where it lands between the anchors, not for the best
    # loss; see step seven above.
    "lr": 1.4e-3,
    # Unchanged from the published recipe. The preconditioner was never the
    # problem, the stability ceiling was.
    "betas": (0.9, 0.95),
    "eps": 1.0e-8,
    "weight_decay": 0.1,
    # Three times the published 0.05. Half of the ceiling raise.
    "warmup_frac": 0.15,
    # Unchanged from the published recipe.
    "final_lr_frac": 0.1,
    # Half the published 1.0. The other half of the ceiling raise.
    "grad_clip": 0.5,
    # A catastrophic gradient measured 1.8e10 against a running scale near 0.6,
    # and a survivable one measured 1.0e3 against the same scale. 500 sits
    # between them and above every ordinary step, which measured 0.2 to 1.8.
    "spike_k": 500.0,
    "spike_beta": 0.98,
    # The instrumented death took roughly twenty steps of drift before the
    # cascade, so the rollback point is placed just outside that window.
    "shadow_every": 25,
    "cooldown": 50,
}


class NormFreeSpikeRejectingRule:
    """Decoupled AdamW, retuned to a measured stability ceiling and guarded by a
    rollback that refuses to write an activation cascade into the weights.

    The submission surface hands over two parameter groups. This rule treats
    them alike, because the measurement gave no reason to split them: the cost
    of the ablation is a whole-network stability property rather than a property
    of the block matrices on their own.
    """

    def __init__(self, matrices, others, cfg):
        self.cfg = dict(cfg)
        self.params = [p for _, p in matrices] + [p for _, p in others]
        self.opt = torch.optim.AdamW(
            self.params,
            lr=HP["lr"],
            betas=HP["betas"],
            eps=HP["eps"],
            weight_decay=HP["weight_decay"],
        )
        # The rollback point. One copy of the parameters, which is state the
        # optimizer is entitled to own, and not a second model.
        self.shadow = [p.detach().clone() for p in self.params]
        self.grad_scale = None
        self.rollbacks = 0
        self.since_shadow = 0
        self.cooldown_left = 0

    def _lr_at(self, step, total):
        warm = max(1, int(HP["warmup_frac"] * total))
        if step < warm:
            base = HP["lr"] * (step + 1) / warm
        else:
            prog = (step - warm) / max(1, total - warm)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            floor = HP["final_lr_frac"]
            base = HP["lr"] * (floor + (1.0 - floor) * cos)
        if self.cooldown_left > 0:
            # Linear mini-warmup after a rollback, so the recovery does not walk
            # straight back into the state it was rolled out of.
            base *= (HP["cooldown"] - self.cooldown_left + 1) / HP["cooldown"]
        return base

    def step(self, step, total):
        # clip_grad_norm_ returns the pre-clip global norm, so one call both
        # applies the ceiling-raising cap and supplies the spike detector its
        # live-state read.
        total_norm = torch.nn.utils.clip_grad_norm_(self.params, HP["grad_clip"])
        g = float(total_norm)
        spike = not math.isfinite(g)
        if self.grad_scale is not None and math.isfinite(g):
            spike = g > HP["spike_k"] * self.grad_scale
        lr = self._lr_at(step, total)
        if self.cooldown_left > 0:
            self.cooldown_left -= 1

        if spike:
            with torch.no_grad():
                self.rollbacks += 1
                for p, s in zip(self.params, self.shadow):
                    p.copy_(s)
                for p in self.params:
                    st = self.opt.state.get(p)
                    if st and "exp_avg" in st:
                        st["exp_avg"].zero_()
                self.since_shadow = 0
                self.cooldown_left = HP["cooldown"]
                # The decoupled decay still applies on a rejected step. It is
                # the part of the update that does not depend on the gradient,
                # so rejecting the gradient is no reason to skip it.
                for p in self.params:
                    p.mul_(1.0 - lr * HP["weight_decay"])
            if self.grad_scale is not None:
                self.grad_scale = (
                    HP["spike_beta"] * self.grad_scale
                    + (1.0 - HP["spike_beta"]) * HP["spike_k"] * self.grad_scale
                )
            return

        self.grad_scale = g if self.grad_scale is None else (
            HP["spike_beta"] * self.grad_scale + (1.0 - HP["spike_beta"]) * g
        )
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.opt.step()

        self.since_shadow += 1
        if self.since_shadow >= HP["shadow_every"]:
            with torch.no_grad():
                for p, s in zip(self.params, self.shadow):
                    s.copy_(p.detach())
            self.since_shadow = 0


def make_update_rule(matrices, others, cfg):
    return NormFreeSpikeRejectingRule(matrices, others, cfg)
