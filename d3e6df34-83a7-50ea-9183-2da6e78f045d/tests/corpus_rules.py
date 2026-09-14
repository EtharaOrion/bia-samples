#!/usr/bin/env python3
"""The behavioural corpus, as rules the verifier can drive rather than as text.

The corpus is a set of update-rule FAMILIES, each written here in the same
interface the submission implements, so the probe drives a corpus entry and a
submission through exactly the same code path. That is what makes the comparison
behavioural: nothing here is compared against the submission's source, and
nothing here stores a precomputed answer that could drift from the rule that
produced it.

Every entry is a published, generic optimizer family. No entry is an invented
record and none carries a measured number. The corpus is a comparison set, not
an anchor set.

Determinism: no clock, no random source, no environment read. Each `step` is a
pure function of its arguments.
"""

from __future__ import annotations

import math


class SGD:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.05))

    def step(self, params, grads, state):
        return [p - self.lr * g for p, g in zip(params, grads)], {}


class Momentum:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.05))
        self.beta = float(hyper.get("beta", 0.9))

    def step(self, params, grads, state):
        buffer = state.get("buffer") or [0.0] * len(params)
        buffer = [self.beta * b + g for b, g in zip(buffer, grads)]
        return [p - self.lr * b for p, b in zip(params, buffer)], {"buffer": buffer}


class Nesterov:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.05))
        self.beta = float(hyper.get("beta", 0.9))

    def step(self, params, grads, state):
        buffer = state.get("buffer") or [0.0] * len(params)
        buffer = [self.beta * b + g for b, g in zip(buffer, grads)]
        look = [g + self.beta * b for g, b in zip(grads, buffer)]
        return [p - self.lr * v for p, v in zip(params, look)], {"buffer": buffer}


class AdaGrad:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.1))
        self.eps = float(hyper.get("eps", 1e-8))

    def step(self, params, grads, state):
        accum = state.get("accum") or [0.0] * len(params)
        accum = [a + g * g for a, g in zip(accum, grads)]
        out = [p - self.lr * g / (math.sqrt(a) + self.eps) for p, g, a in zip(params, grads, accum)]
        return out, {"accum": accum}


class RMSProp:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.05))
        self.rho = float(hyper.get("rho", 0.9))
        self.eps = float(hyper.get("eps", 1e-8))

    def step(self, params, grads, state):
        accum = state.get("accum") or [0.0] * len(params)
        accum = [self.rho * a + (1.0 - self.rho) * g * g for a, g in zip(accum, grads)]
        out = [p - self.lr * g / (math.sqrt(a) + self.eps) for p, g, a in zip(params, grads, accum)]
        return out, {"accum": accum}


class Adam:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.05))
        self.b1 = float(hyper.get("b1", 0.9))
        self.b2 = float(hyper.get("b2", 0.999))
        self.eps = float(hyper.get("eps", 1e-8))

    def step(self, params, grads, state):
        first = state.get("first") or [0.0] * len(params)
        second = state.get("second") or [0.0] * len(params)
        count = int(state.get("count", 0)) + 1
        first = [self.b1 * m + (1.0 - self.b1) * g for m, g in zip(first, grads)]
        second = [self.b2 * v + (1.0 - self.b2) * g * g for v, g in zip(second, grads)]
        c1 = 1.0 - self.b1 ** count
        c2 = 1.0 - self.b2 ** count
        out = [
            p - self.lr * (m / c1) / (math.sqrt(v / c2) + self.eps)
            for p, m, v in zip(params, first, second)
        ]
        return out, {"first": first, "second": second, "count": count}


class AdamW(Adam):
    def __init__(self, hyper):
        Adam.__init__(self, hyper)
        self.decay = float(hyper.get("decay", 0.01))

    def step(self, params, grads, state):
        decayed = [p * (1.0 - self.lr * self.decay) for p in params]
        return Adam.step(self, decayed, grads, state)


class Lion:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.01))
        self.b1 = float(hyper.get("b1", 0.9))
        self.b2 = float(hyper.get("b2", 0.99))

    def step(self, params, grads, state):
        buffer = state.get("buffer") or [0.0] * len(params)
        interp = [self.b1 * m + (1.0 - self.b1) * g for m, g in zip(buffer, grads)]
        out = [p - self.lr * sign_of(u) for p, u in zip(params, interp)]
        buffer = [self.b2 * m + (1.0 - self.b2) * g for m, g in zip(buffer, grads)]
        return out, {"buffer": buffer}


class SignSGD:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.01))

    def step(self, params, grads, state):
        return [p - self.lr * sign_of(g) for p, g in zip(params, grads)], {}


class NormalizedSGD:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.05))
        self.eps = float(hyper.get("eps", 1e-8))

    def step(self, params, grads, state):
        norm = math.sqrt(sum(g * g for g in grads)) + self.eps
        return [p - self.lr * g / norm for p, g in zip(params, grads)], {}


class OrthogonalizedMomentum:
    """Momentum whose update is RMS-normalized, the scalar analogue of the
    orthogonalizing step used by the block-matrix optimizers this family names."""

    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.05))
        self.beta = float(hyper.get("beta", 0.95))
        self.eps = float(hyper.get("eps", 1e-8))

    def step(self, params, grads, state):
        buffer = state.get("buffer") or [0.0] * len(params)
        buffer = [self.beta * b + g for b, g in zip(buffer, grads)]
        rms = math.sqrt(sum(b * b for b in buffer) / len(buffer)) + self.eps
        return [p - self.lr * b / rms for p, b in zip(params, buffer)], {"buffer": buffer}


class HeavyBallDecay:
    def __init__(self, hyper):
        self.lr = float(hyper.get("lr", 0.05))
        self.beta = float(hyper.get("beta", 0.8))
        self.decay = float(hyper.get("decay", 0.02))

    def step(self, params, grads, state):
        buffer = state.get("buffer") or [0.0] * len(params)
        buffer = [self.beta * b + (1.0 - self.beta) * g for b, g in zip(buffer, grads)]
        out = [p * (1.0 - self.decay) - self.lr * b for p, b in zip(params, buffer)]
        return out, {"buffer": buffer}


def sign_of(value) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


FAMILIES = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
    "adamw": AdamW,
    "lion": Lion,
    "signsgd": SignSGD,
    "normalized-sgd": NormalizedSGD,
    "orthogonalized-momentum": OrthogonalizedMomentum,
    "heavy-ball-decay": HeavyBallDecay,
}


def build(kind: str, hyper: dict):
    if kind not in FAMILIES:
        raise KeyError("corpus family outside the closed set: " + repr(kind))
    return FAMILIES[kind](hyper or {})
