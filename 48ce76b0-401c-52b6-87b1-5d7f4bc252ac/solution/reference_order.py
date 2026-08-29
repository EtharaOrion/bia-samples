"""Reference ordering policy for the S03 data-order task.

The comparator arm is an ensemble of uniform random shuffles, so an ordering only earns reward
by beating i.i.d. sampling at a frozen model, a frozen optimizer and a frozen token budget.
Three effects are composed here and none of them is a difficulty curriculum. Every claim below
that a component helps was measured on the graded profile against the same comparator ensemble
the grader uses, and the measurements are recorded in solution/TRUTH.md.

The first component is exact per-batch domain apportionment. Under a uniform shuffle the domain
composition of a batch is multinomial, so the per-step gradient carries composition noise that
the frozen optimizer must average out over many steps. Apportioning every batch to the global
domain proportions removes that component of the gradient variance without touching the multiset
consumed over the epoch. The apportionment is exact largest remainder: each batch first takes the
integer part of every domain's outstanding claim, then hands out the residual seats one seat at a
time to whichever domain currently carries the largest unmet fractional claim, recomputing that
claim after every seat. Handing the whole residual to a single domain instead exhausts the domains
one after another and turns the late batches into single-domain blocks, which is a severely
non-stationary distribution rather than a stratified one. That defect is invisible at a small
operating point where the residual is a single seat and it is catastrophic at the graded operating
point where the residual is four seats, and both facts are measured and recorded in TRUTH.md.

The second component is low-discrepancy sweeping inside each domain. Drawing the domain's quota in
entropy-sorted order would make early and late training see systematically different data, and
drawing it at random reintroduces the variance stratification just removed. Consuming each domain
in van der Corput order makes consecutive batches sweep the within-domain feature range evenly, so
every prefix of training is close to representative of the whole domain.

The third component is a frequency tilt on the release curve. The domain mixture is fixed and
strongly unequal, so the rarest domain contributes about a third as many sequences as the most
common one and is correspondingly the most underfit at the end of a single epoch. The tilt bends
each domain's cumulative release away from the straight line by a quadratic bump, releasing the
rarer domains earlier and the more common domains later, while leaving the total released after
every step exactly linear so batch sizes and the consumed multiset are untouched. The bump
coefficients are set proportional to each domain's relative frequency and are recentred so the
count-weighted sum of coefficients is zero, which is what makes the aggregate release exactly
linear, and they are clamped so every per-domain release curve stays monotone. The direction was
not assumed. Both signs of the tilt were run on the graded profile and the sign kept here is the
one that helped, while the opposite sign measured worse than the comparator ensemble.

None of the three arguments claims that easy before hard helps, and the reference never orders by
difficulty.
"""

from __future__ import annotations

# The largest tilt magnitude for which u + eps * u * (1 - u) stays monotone on the unit interval.
MAX_TILT = 1.0


def _van_der_corput_key(n: int) -> float:
    value, denom = 0.0, 1.0
    while n > 0:
        denom *= 2.0
        value += (n & 1) / denom
        n >>= 1
    return value


def _low_discrepancy(sorted_indices: list[int]) -> list[int]:
    ranked = list(enumerate(sorted_indices))
    ranked.sort(key=lambda pair: (_van_der_corput_key(pair[0]), pair[0]))
    return [idx for _, idx in ranked]


def _tilt_coefficients(counts: list[int]) -> list[float]:
    """Bump coefficients for the per-domain release curve.

    The coefficients are recentred so that the count-weighted sum is zero. That is the condition
    under which the domains' bumps cancel in aggregate, so the total number of sequences released
    after any step stays exactly proportional to the step index and the batch size never moves.
    The sign is negative on the relative frequency, which releases the rarer domains earlier.
    """
    total = sum(counts)
    n_dom = len(counts)
    if total <= 0 or n_dom == 0:
        return [0.0] * n_dom
    relative = [c * n_dom / total for c in counts]
    centre = sum(counts[d] * relative[d] for d in range(n_dom)) / total
    eps = [-(relative[d] - centre) for d in range(n_dom)]
    peak = max(abs(e) for e in eps)
    if peak > MAX_TILT:
        eps = [e * MAX_TILT / peak for e in eps]
    return eps


def _released_fraction(u: float, eps: float) -> float:
    return u + eps * u * (1.0 - u)


def _apportion(counts: list[int], remaining: list[int], batch_size: int, targets: list[float]) -> list[int]:
    """Exact largest-remainder apportionment of one batch across the domains.

    The claim is measured against what each domain has actually released rather than against an
    idealized cursor, so a rounding residual on one batch is repaid on the next instead of
    accumulating. Every residual seat is granted singly and the claims are recomputed after each
    grant, which is what keeps one domain from absorbing the whole residual.
    """
    n_dom = len(counts)
    released = [counts[d] - remaining[d] for d in range(n_dom)]
    desire = [targets[d] - released[d] for d in range(n_dom)]
    quota = [min(remaining[d], max(0, int(desire[d]))) for d in range(n_dom)]
    seats = batch_size - sum(quota)
    while seats > 0:
        best, best_key = -1, None
        for d in range(n_dom):
            if quota[d] >= remaining[d]:
                continue
            key = (desire[d] - quota[d], -d)
            if best_key is None or key > best_key:
                best, best_key = d, key
        if best < 0:
            break
        quota[best] += 1
        seats -= 1
    while seats < 0:
        best, best_key = -1, None
        for d in range(n_dom):
            if quota[d] <= 0:
                continue
            key = (quota[d] - desire[d], -d)
            if best_key is None or key > best_key:
                best, best_key = d, key
        if best < 0:
            break
        quota[best] -= 1
        seats += 1
    return quota


def build_order(meta: dict) -> list[list[int]]:
    n_steps = int(meta["n_steps"])
    batch_size = int(meta["batch_sequences"])
    n_dom = int(meta["n_domains"])
    features = meta["features"]

    by_domain: list[list[tuple[float, int]]] = [[] for _ in range(n_dom)]
    for f in features:
        by_domain[int(f["domain_id"])].append((float(f["unigram_entropy"]), int(f["index"])))

    queues: list[list[int]] = []
    counts: list[int] = []
    for d in range(n_dom):
        by_domain[d].sort()
        ordered = _low_discrepancy([idx for _, idx in by_domain[d]])
        queues.append(ordered)
        counts.append(len(ordered))

    eps = _tilt_coefficients(counts)
    cursors = [0] * n_dom
    remaining = list(counts)
    order: list[list[int]] = []
    for step in range(n_steps):
        u = (step + 1) / n_steps
        targets = [counts[d] * _released_fraction(u, eps[d]) for d in range(n_dom)]
        quota = _apportion(counts, remaining, batch_size, targets)
        batch: list[int] = []
        for d in range(n_dom):
            take = quota[d]
            if take <= 0:
                continue
            batch.extend(queues[d][cursors[d] : cursors[d] + take])
            cursors[d] += take
            remaining[d] -= take
        batch.sort()
        order.append(batch)
    return order
