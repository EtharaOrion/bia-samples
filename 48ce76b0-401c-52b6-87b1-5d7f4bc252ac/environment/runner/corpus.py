"""Deterministic frozen corpus for the S03 data-order task.

The corpus is generated rather than downloaded, so the scored run needs no network and the
frozen fixture is a specification plus a digest instead of a large binary blob. Generation is
pure numpy, seeded, and identical on every host, which is what lets the DIVERGENCE checker
regenerate the corpus and compare digests.

The corpus is a mixture of domains. Each domain is a first-order Markov source over its own
token band plus a shared connector band, with a domain-specific concentration that sets its
entropy. Sequence order matters here because the domain composition of a batch moves the
gradient the frozen optimizer receives, and batch composition is the axis this task leaves
free.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _domain_bands(cfg: dict) -> list[tuple[int, int]]:
    vocab = cfg["vocab_size"]
    n_dom = cfg["n_domains"]
    connector = max(4, vocab // 16)
    usable = vocab - connector
    span = usable // n_dom
    return [(connector + d * span, connector + (d + 1) * span) for d in range(n_dom)]


def _domain_alpha(d: int, n_dom: int) -> float:
    """Concentration for the domain transition matrix. Low alpha means peaked, low entropy."""
    lo, hi = 0.02, 3.0
    if n_dom == 1:
        return lo
    return lo + (hi - lo) * (d / (n_dom - 1))


def _connector_rate(d: int, n_dom: int) -> float:
    lo, hi = 0.02, 0.22
    if n_dom == 1:
        return lo
    return lo + (hi - lo) * (((d * 5) % n_dom) / (n_dom - 1))


def _cdf(mat: np.ndarray) -> np.ndarray:
    c = np.cumsum(mat, axis=-1)
    return c / c[..., -1:]


def _active_states(d: int, n_dom: int, span: int) -> int:
    """Support size of domain d. Varying it is what gives the domains different entropies."""
    frac = 0.1 if n_dom == 1 else 0.1 + 0.9 * (d / (n_dom - 1))
    return max(8, int(round(span * frac)))


def _build_domain_source(rng: np.random.Generator, band: tuple[int, int], alpha: float, connector: int, active: int) -> dict:
    lo, hi = band
    size = min(active, hi - lo)
    trans = rng.dirichlet(np.full(size, alpha), size=size)
    start = rng.dirichlet(np.full(size, alpha))
    conn = rng.dirichlet(np.full(connector, 0.4))
    # A row-offset flattened cumulative table. Row r occupies [r*size, (r+1)*size) and holds
    # r + cdf_r, which is globally monotone, so one searchsorted samples every chain at once.
    row_cdf = _cdf(trans)
    flat = (row_cdf + np.arange(size, dtype=np.float64)[:, None]).reshape(-1)
    return {
        "lo": lo,
        "size": size,
        "flat_cdf": np.ascontiguousarray(flat),
        "start_cdf": _cdf(start[None, :])[0],
        "conn_cdf": _cdf(conn[None, :])[0],
        "connector": connector,
    }


def _sample_block(rng: np.random.Generator, src: dict, rate: float, n_seq: int, length: int) -> np.ndarray:
    """Vectorized first-order Markov sampling for a whole block of sequences."""
    size = src["size"]
    if n_seq == 0:
        return np.empty((0, length), dtype=np.uint16)
    out = np.empty((n_seq, length), dtype=np.uint16)
    state = np.searchsorted(src["start_cdf"], rng.random(n_seq)).clip(0, size - 1)
    flat = src["flat_cdf"]
    for i in range(length):
        out[:, i] = (src["lo"] + state).astype(np.uint16)
        u = rng.random(n_seq)
        idx = np.searchsorted(flat, state.astype(np.float64) + u)
        state = (idx - state * size).clip(0, size - 1)
    mask = rng.random((n_seq, length)) < rate
    n_conn = int(mask.sum())
    if n_conn:
        picks = np.searchsorted(src["conn_cdf"], rng.random(n_conn)).clip(0, src["connector"] - 1)
        out[mask] = picks.astype(np.uint16)
    return out


def generate_corpus(cfg: dict):
    """Return (train_tokens, train_domains, val_tokens).

    train_tokens has shape (n_train_sequences, seq_len) and dtype uint16.
    """
    rng = np.random.default_rng(cfg["corpus_seed"])
    bands = _domain_bands(cfg)
    connector = bands[0][0]
    n_dom = cfg["n_domains"]
    span = bands[0][1] - bands[0][0]
    sources = [
        _build_domain_source(rng, bands[d], _domain_alpha(d, n_dom), connector, _active_states(d, n_dom, span))
        for d in range(n_dom)
    ]
    rates = [_connector_rate(d, n_dom) for d in range(n_dom)]

    n_train = cfg["n_train_sequences"]
    n_val = cfg["n_val_sequences"]
    seq_len = cfg["seq_len"]

    # Domain assignment is a fixed non-uniform mixture emitted domain by domain, so the
    # natural index order is domain-blocked. That is a property of the fixture and carries no
    # statement about which ordering scores well.
    weights = np.array([1.0 + 0.35 * ((d * 3) % n_dom) for d in range(n_dom)], dtype=np.float64)
    weights = weights / weights.sum()
    counts = np.floor(weights * n_train).astype(np.int64)
    counts[-1] += n_train - int(counts.sum())

    train = np.empty((n_train, seq_len), dtype=np.uint16)
    domains = np.empty(n_train, dtype=np.int16)
    at = 0
    for d in range(n_dom):
        k = int(counts[d])
        train[at : at + k] = _sample_block(rng, sources[d], rates[d], k, seq_len)
        domains[at : at + k] = d
        at += k

    val_counts = [n_val // n_dom] * n_dom
    for j in range(n_val - sum(val_counts)):
        val_counts[j % n_dom] += 1
    val = np.empty((n_val, seq_len), dtype=np.uint16)
    at = 0
    for d in range(n_dom):
        k = int(val_counts[d])
        val[at : at + k] = _sample_block(rng, sources[d], rates[d], k, seq_len)
        at += k

    return train, domains, val


def sequence_features(train: np.ndarray, domains: np.ndarray, cfg: dict) -> list[dict]:
    """Frozen per-sequence features handed to the submitted ordering policy.

    These are computed from the corpus alone. They carry no validation-loss information and
    no gradient information, so an ordering policy built on them is a static curriculum and
    never an online feedback loop.
    """
    n, length = train.shape
    feats = []
    for i in range(n):
        row = train[i].astype(np.int64)
        counts = np.bincount(row, minlength=cfg["vocab_size"]).astype(np.float64)
        nz = counts[counts > 0]
        p = nz / nz.sum()
        entropy = float(-(p * np.log(p)).sum())
        repeats = int(np.count_nonzero(row[1:] == row[:-1]))
        feats.append(
            {
                "index": i,
                "domain_id": int(domains[i]),
                "unigram_entropy": round(entropy, 6),
                "distinct_token_count": int(nz.size),
                "adjacent_repeat_rate": round(repeats / max(1, length - 1), 6),
                "mean_token_id": round(float(row.mean()), 6),
            }
        )
    return feats


def corpus_digests(train: np.ndarray, domains: np.ndarray, val: np.ndarray) -> dict:
    return {
        "train_tokens_sha256": hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),
        "train_domains_sha256": hashlib.sha256(np.ascontiguousarray(domains).tobytes()).hexdigest(),
        "val_tokens_sha256": hashlib.sha256(np.ascontiguousarray(val).tobytes()).hexdigest(),
    }
