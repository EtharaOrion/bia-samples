"""Deterministic synthetic corpus for the frozen substrate. Agent-visible copy.

No network, no clock, no random source outside an explicit seed. The stream is a
sparse first-order Markov chain rather than a tiled motif: a motif is memorised
in a handful of steps and every loss curve collapses to near zero, which makes a
crossing search meaningless because everything crosses at once. A sparse chain
has a real entropy floor, so curves separate and a crossing step measures
something.

One distribution, three draws. `dist_seed` fixes the transition table and IS the
dataset, so it is a frozen axis. Each split then draws its own realisation from
that same table with its own draw seed, which is what makes a held-out split a
held-out sample of the same source rather than a different source. The
development draw seed is the one this file exposes. The graded draw seed exists
only inside the verifier environment.
"""
from __future__ import annotations

import numpy as np

FANOUT = 8
ALPHA = 0.6


def markov_stream(dist_seed: int, draw_seed: int, length: int, vocab: int) -> np.ndarray:
    """One deterministic realisation of the frozen source. Same seeds, same bytes."""
    table = np.random.default_rng(int(dist_seed))
    successors = table.integers(0, vocab, size=(vocab, FANOUT))
    weights = table.dirichlet(np.full(FANOUT, ALPHA), size=vocab)
    cumulative = np.cumsum(weights, axis=1)
    rng = np.random.default_rng(int(draw_seed))
    draws = rng.random(length)
    out = np.empty(length, dtype=np.int64)
    current = 0
    for index in range(length):
        out[index] = current
        pick = int(np.searchsorted(cumulative[current], draws[index]))
        if pick >= FANOUT:
            pick = FANOUT - 1
        current = int(successors[current, pick])
    return out


def batch_order(seed: int, count: int, high: int) -> np.ndarray:
    """Deterministic batch offsets. The order is fixed by the shape, not by a recipe."""
    return np.random.default_rng(int(seed) + 7919).integers(0, high, size=count)
