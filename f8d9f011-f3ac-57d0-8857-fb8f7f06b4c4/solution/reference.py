"""The reference solution the live checkers accept. Pure, deterministic, stdlib only.

This module is the one place the allocation procedure lives. solution/recompute.py
imports it to derive the golden artifacts and tests/grade.py imports it to recompute
the substrate-internal scale points, so the number the reference reaches and the
number the verifier grades against can never drift apart by being written twice.

It reads no clock, opens no socket, consults no random source and imports nothing
outside the standard library. Every quantity it returns is a pure function of its
arguments.
"""

from __future__ import annotations

import hashlib
import json

ROUND = 9


def canonical(payload) -> bytes:
    """The one canonical byte image used for every digest in this slot."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(payload) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def sensitivity_digest(vector) -> str:
    """The fit witness: sha256 over the canonical bytes of a sensitivity vector.

    A submission carries this digest for the vector it actually fitted against, so
    a declared calibration version that disagrees with the bytes fitted is a
    disagreement a checker can read rather than a claim it has to take on faith.
    """
    return digest([round(float(value), ROUND) for value in vector])


def groups_of(numel, group_size):
    return int(numel) // int(group_size)


def scale_bits_total(tensors, group_size, scale_bits):
    return sum(groups_of(row["numel"], group_size) * int(scale_bits) for row in tensors)


def allocated_bits(tensors, allocation, group_size, scale_bits, include_scales):
    """Harness accounting over the quantized tensors. Never a submission's arithmetic."""
    weight_bits = sum(int(row["numel"]) * int(allocation[row["id"]]) for row in tensors)
    if not include_scales:
        return weight_bits
    return weight_bits + scale_bits_total(tensors, group_size, scale_bits)


def degradation(tensors, allocation, sensitivity, constant):
    """The closed-form surrogate for perplexity degradation at one evaluation point."""
    total = sum(int(row["numel"]) for row in tensors)
    accumulated = 0.0
    for index, row in enumerate(tensors):
        bits = int(allocation[row["id"]])
        accumulated += float(sensitivity[index]) * int(row["numel"]) * (2.0 ** (-2 * bits))
    return round(float(constant) * accumulated / total, ROUND)


def uniform_allocation(tensors, bits):
    return {row["id"]: int(bits) for row in tensors}


def allocate(tensors, sensitivity, budget_bits, group_size, scale_bits, bit_min, bit_max,
             include_scales=True):
    """Greedy marginal-gain bit allocation under a hard bit budget.

    Every tensor starts at the floor width. The bit that buys the largest reduction
    in the surrogate degradation per bit of budget is bought next, and the loop ends
    when no affordable purchase remains. Ties break on tensor index, so the result
    is fixed by the inputs alone and never by iteration order.
    """
    allocation = {row["id"]: int(bit_min) for row in tensors}
    spent = allocated_bits(tensors, allocation, group_size, scale_bits, include_scales)
    while True:
        best_key = None
        best_index = None
        for index, row in enumerate(tensors):
            bits = allocation[row["id"]]
            if bits >= int(bit_max):
                continue
            if spent + int(row["numel"]) > int(budget_bits):
                continue
            gain = float(sensitivity[index]) * (
                2.0 ** (-2 * bits) - 2.0 ** (-2 * (bits + 1))
            )
            key = (-gain, index)
            if best_key is None or key < best_key:
                best_key, best_index = key, index
        if best_index is None:
            return allocation
        chosen = tensors[best_index]
        allocation[chosen["id"]] += 1
        spent += int(chosen["numel"])


def evaluate(tensors, allocation, shards, constant):
    """Degradation at every evaluation point, in the order the schedule names them."""
    return [
        {"point_id": shard["id"], "degradation": degradation(tensors, allocation, shard["sensitivity"], constant)}
        for shard in shards
    ]


def mean_degradation(points):
    if not points:
        return None
    return round(sum(row["degradation"] for row in points) / len(points), ROUND)
