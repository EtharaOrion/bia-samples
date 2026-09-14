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


VERIFIER-OWNED COPY. This is the verifier's own copy of the substrate module of
the same name under environment/. The two carry identical arithmetic on purpose,
so the agent explores the optimizer and the architecture the verifier actually
runs. They are separate FILES on purpose too: a verifier that imported the
agent-visible tree would be grading bytes the agent can reach.
"""
from __future__ import annotations

import bisect

import numpy as np

FANOUT = 8
ALPHA = 0.6


def markov_stream(dist_seed: int, draw_seed: int, length: int, vocab: int) -> np.ndarray:
    """One deterministic realisation of the frozen source. Same seeds, same bytes.

    THE WALK IS THE SAME WALK. The transition table, the cumulative row per state
    and the draw sequence are built exactly as before and from the same seeds, so
    this function returns the identical array it always returned. What changed is
    only how the per-step lookup is spelled. `np.searchsorted` over a length-8
    row costs more in call overhead than the search it performs, and this walk
    performs it once per token: at `stream_tokens_train` of 268435456 that is
    448 seconds of measured wall clock inside a verifier whose whole declared
    per-attempt budget is 1080 seconds, spent materialising a corpus rather than
    grading anything.

    `bisect.bisect_left` over the same cumulative row as a Python list answers
    the identical query - the first index whose threshold is not less than the
    draw - and the equality was checked against the previous spelling over a two
    million token realisation of these exact seeds before this was written.
    Measured in the verifier's own image: 1.6697 us per token before, 0.3117 us
    after, which is 448.2 s down to 83.7 s for the graded train stream.
    """
    table = np.random.default_rng(int(dist_seed))
    successors = table.integers(0, vocab, size=(vocab, FANOUT))
    weights = table.dirichlet(np.full(FANOUT, ALPHA), size=vocab)
    cumulative = np.cumsum(weights, axis=1)
    rng = np.random.default_rng(int(draw_seed))
    draws = rng.random(length)
    out = np.empty(length, dtype=np.int64)
    thresholds = cumulative.tolist()
    following = successors.tolist()
    values = draws.tolist()
    search = bisect.bisect_left
    current = 0
    for index in range(length):
        out[index] = current
        pick = search(thresholds[current], values[index])
        if pick >= FANOUT:
            pick = FANOUT - 1
        current = following[current][pick]
    return out


def batch_order(seed: int, count: int, high: int) -> np.ndarray:
    """Deterministic batch offsets. The order is fixed by the shape, not by a recipe."""
    return np.random.default_rng(int(seed) + 7919).integers(0, high, size=count)
