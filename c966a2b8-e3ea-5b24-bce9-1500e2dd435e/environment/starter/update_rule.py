"""Starter update rule. Deliberately suboptimal. Improving it is the task.

Copy this file to /workspace/submission/update_rule.py and change it. The
harness imports exactly that path and calls make_update_rule once per graded
arm, then calls step once per optimizer step.

Contract

  make_update_rule(matrices, others, cfg) -> rule

    matrices  list of (name, torch.nn.Parameter) for every rank two weight in
              the transformer blocks, which is where every published optimizer
              record acts
    others    list of (name, torch.nn.Parameter) for everything else, the two
              embedding tables and the output head
    cfg       a read only copy of the frozen operating point: vocab_size,
              d_model, n_layer, n_head, seq_len, mlp_mult, batch_size, steps,
              seed and forward_backward_per_step

  rule.step(step, total) -> None

    Called once per optimizer step, after loss.backward() has populated
    p.grad on every parameter and before the next forward pass. Mutate the
    parameters in place. You own all state, all schedules, all preconditioning
    and all clipping.

What you may not do

  You may not touch the model, the data, the batch order, the step count or the
  one forward-backward pass per step rule. None of them is reachable from here.
  You may not cause an activation normalization operation to execute inside the
  model's forward or backward pass; the harness records every such operation
  while the run is happening and a graded run that contains one scores zero.
  Normalizing an update tensor inside step is explicitly allowed and is not an
  activation normalization.
"""

from __future__ import annotations

import torch


class StarterRule:
    def __init__(self, matrices, others, cfg):
        self.params = [p for _, p in matrices] + [p for _, p in others]
        self.lr = 0.01
        self.momentum = 0.9
        self.buf = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, step, total):
        for p, b in zip(self.params, self.buf):
            if p.grad is None:
                continue
            b.mul_(self.momentum).add_(p.grad)
            p.add_(b, alpha=-self.lr)


def make_update_rule(matrices, others, cfg):
    return StarterRule(matrices, others, cfg)
