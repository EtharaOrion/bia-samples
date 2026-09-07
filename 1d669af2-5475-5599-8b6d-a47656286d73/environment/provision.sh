#!/usr/bin/env bash
# Materialise the agent surface's nanoGPT checkpoint. Idempotent, and safe to source.
#
# WHY THIS IS NOT A LAYER. environment/Dockerfile used to train this checkpoint at image
# build with `bootstrap.py checkpoint --device cuda`. A `docker build` has no
# accelerator: BuildKit, which is what `docker build` is on this engine, runs its steps
# outside the daemon's default runtime, so `torch.cuda.is_available()` is False inside
# every RUN and the step died on `Found no NVIDIA driver on your system`. The same
# training on this host's CPU measures 564s per optimizer step at the frozen batch of
# 524288 tokens, against a 1800s build bound, so no step count worth having fits either.
# The checkpoint therefore has to be produced by a container that was given the
# accelerator task.toml [environment] gpus already asks for, which is this one.
#
# The invocation below is the delivered one: the same declaration, the same shards, the
# same pinned seed. tests/test.sh runs the identical invocation against the verifier's
# own copy, which is the property tests/bound.json checkpoint_note binds, and neither
# surface ever loads the other's file.
#
# ON THE STEP COUNT, MEASURED. The delivered Dockerfiles passed 3250, which is the
# record set's baseline_steps_to_target, an upstream anchor about reaching 3.28 val loss.
# It is left untouched where it is declared, at environment/nanogpt_substrate.json. As a
# checkpoint build it costs 3250 x 3.25s, being 2.9 hours on each surface.
#
# A previous revision of this file passed 150, chosen to fit a wall-clock bound. That
# number is retired because it was measured to make the slot unscoreable. At 150 steps
# this decoder sits at 6.663 nats on the calibration slice and the ENTIRE quantization
# signal, the loss the uniform 4-bit allocation costs over all 48 matmuls, is 3.47e-4
# nats, while the largest sign-impossible excursion in the same reading set is 3.22e-4
# nats. Signal and resolution are the same size, so no allocation procedure has an input
# to work from, and the reference measured a target scale point WORSE than its own
# baseline. tests/grade.py then saw a non-positive span and could score nothing above
# zero. That is a property of the checkpoint, not of the allocator.
#
# Measured on one H100 on 2026-09-06, through the shipped tests/evaluate.py path, over
# the delivered pin and the delivered folds:
#
#   steps  baseline      target        span          bar held at every fold  reward
#    150   +0.375795832  +0.528946253  -0.153150421  no, misses two          0.0
#    600   +0.859610441  +0.747753902  +0.111856539  no, misses one          0.0
#   1500   +1.273849500  +1.155486235  +0.118363265  yes                     1.0
#
# 1500 is the smallest of the three measured points at which the reference reaches its
# own bar at every scheduled fold, so it is what this file passes. It costs 1500 x 3.25s,
# being 81 minutes, on each surface. That is more than the slot's declared per-attempt
# budget_hours of 0.3 admits if a solving attempt pays it, and that collision is declared
# as gap-oer-22-checkpoint-build-cost-exceeds-declared-attempt-budget rather than settled
# here by moving a bound. Nothing in this slot grades the checkpoint's own loss: the
# metric is a degradation DELTA between two perplexities of this same snapshot.
#
# It runs once. A container that already carries the checkpoint pays a file test.

OER22_CHECKPOINT_PATH="${OER22_CHECKPOINT:-/workspace/checkpoint/nanogpt.pt}"
OER22_CHECKPOINT_STEPS="${OER22_CHECKPOINT_STEPS:-1500}"
OER22_CHECKPOINT_SEED="${OER22_CHECKPOINT_SEED:-1337}"

if [ ! -s "${OER22_CHECKPOINT_PATH}" ]; then
  python3 /workspace/environment/bootstrap.py checkpoint \
    --declaration /workspace/environment/nanogpt_substrate.json \
    --shards /workspace/data/fineweb10B \
    --steps "${OER22_CHECKPOINT_STEPS}" \
    --seed "${OER22_CHECKPOINT_SEED}" \
    --out "${OER22_CHECKPOINT_PATH}" 1>&2
fi
