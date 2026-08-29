"""Independent hypervolume implementation for the DIVERGENCE checker.

The engine computes hypervolume by a descending-u1 sweep that accumulates one
strip per newly dominating point. This module computes the same quantity by a
structurally different route: coordinate compression over the union of
origin-anchored boxes, summing every grid cell that some achieved point
dominates. The two share no code and no intermediate value, so agreement is
evidence about the number rather than about one implementation.
"""

from __future__ import annotations


def hypervolume_grid(points) -> float:
    """2D hypervolume, maximization, reference point (0.0, 0.0), by cell union."""
    pts = [(float(a), float(b)) for a, b in points if float(a) > 0.0 and float(b) > 0.0]
    if not pts:
        return 0.0
    xs = sorted({0.0} | {p[0] for p in pts})
    ys = sorted({0.0} | {p[1] for p in pts})
    total = 0.0
    for xi in range(len(xs) - 1):
        x0, x1 = xs[xi], xs[xi + 1]
        for yi in range(len(ys) - 1):
            y0, y1 = ys[yi], ys[yi + 1]
            if any(p[0] >= x1 and p[1] >= y1 for p in pts):
                total += (x1 - x0) * (y1 - y0)
    return total


def normalize_point(val_loss: float, density: float, anc: dict) -> tuple:
    """Independent re-derivation of the pinned normalization from the anchors."""
    span_l = anc["loss_ref"] - anc["loss_ideal"]
    span_d = anc["density_ref"] - anc["density_ideal"]
    if span_l <= 0.0 or span_d <= 0.0:
        raise ValueError("degenerate normalization span")
    u1 = (anc["loss_ref"] - float(val_loss)) / span_l
    u2 = (anc["density_ref"] - float(density)) / span_d
    return (min(max(u1, 0.0), 1.0), min(max(u2, 0.0), 1.0))


def reward_from_raw_points(raw_points, anc: dict) -> dict:
    """Recompute the whole reward from the raw achieved objective values."""
    normalized = [normalize_point(l, d, anc) for l, d in raw_points]
    raw = hypervolume_grid(normalized)
    return {
        "normalized_points": [list(p) for p in normalized],
        "hypervolume_raw": raw,
        "score": min(max(raw, 0.0), 1.0),
    }
