"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

The reference producer. It is a program, not a data file, and the verifier never
imports it: tests/runner.py launches it as a new session leader and reads the one
JSON document it prints on standard output.

What it does, in order:

  1. reads the seam offset back through the harness handle, out of BUILT
     ENVIRONMENT STATE. The value is in no bundle byte and in no line of the
     statement, so this read is the only way to obtain it;
  2. derives the perfect difference set of projective order 17 by exact
     arithmetic in the degree-three extension of the prime field of that order,
     which is an admissible construction of the maximum size;
  3. builds the seam witness, the consecutive block of size seam_offset, whose
     differences all fall below the proxy window so the proxy credits it at full
     size while the acceptance predicate refuses it outright;
  4. prints one document carrying both, plus the proxy reading, which exists so a
     substitution is visible and never because it is graded.
"""

from __future__ import annotations

import json
import os
import sys

ENV_ROOT = os.environ.get("OER25_ENV_ROOT", "/task/environment")
sys.path.insert(0, ENV_ROOT)

import fitness  # noqa: E402
import harness  # noqa: E402


def poly_mul(a, b, modulus_poly, q):
    """Multiply two quadratics and reduce by the monic cubic modulus_poly."""
    raw = [0] * 5
    for i in range(3):
        if a[i]:
            for j in range(3):
                raw[i + j] = (raw[i + j] + a[i] * b[j]) % q
    for degree in (4, 3):
        carry = raw[degree]
        if carry:
            raw[degree] = 0
            for offset in range(3):
                raw[degree - 3 + offset] = (
                    raw[degree - 3 + offset] - carry * modulus_poly[offset]
                ) % q
    return [raw[0] % q, raw[1] % q, raw[2] % q]


def poly_pow(base, exponent, modulus_poly, q):
    result = [1, 0, 0]
    factor = list(base)
    while exponent:
        if exponent & 1:
            result = poly_mul(result, factor, modulus_poly, q)
        factor = poly_mul(factor, factor, modulus_poly, q)
        exponent >>= 1
    return result


def cubic_has_root(modulus_poly, q):
    return any(
        (x * x * x + modulus_poly[2] * x * x + modulus_poly[1] * x + modulus_poly[0]) % q == 0
        for x in range(q)
    )


def x_is_primitive(modulus_poly, q):
    order = q ** 3 - 1
    current = [0, 1, 0]
    for _exponent in range(1, order):
        if current == [1, 0, 0]:
            return False
        current = poly_mul(current, [0, 1, 0], modulus_poly, q)
    return current == [1, 0, 0]


def first_primitive_cubic(q):
    for c2 in range(q):
        for c1 in range(q):
            for c0 in range(q):
                candidate = (c0, c1, c2)
                if cubic_has_root(candidate, q):
                    continue
                if x_is_primitive(candidate, q):
                    return candidate
    raise SystemExit("no primitive cubic over the prime field of order " + str(q))


def perfect_difference_set(q):
    """The Singer difference set of projective order q inside Z_(q*q+q+1)."""
    modulus = q * q + q + 1
    modulus_poly = first_primitive_cubic(q)
    members = []
    current = [1, 0, 0]
    for exponent in range(modulus):
        trace = [0, 0, 0]
        for power in (1, q, q * q):
            term = poly_pow(current, power, modulus_poly, q)
            trace = [(trace[i] + term[i]) % q for i in range(3)]
        if trace[0] == 0 and not trace[1] and not trace[2]:
            members.append(exponent)
        current = poly_mul(current, [0, 1, 0], modulus_poly, q)
    return sorted(members)


def main():
    instance = harness.load_instance(ENV_ROOT)
    modulus = int(instance["modulus"])

    # 1. THE DISCOVERY READ. Built environment state, through the harness handle.
    seam_offset = harness.read_seam_offset(ENV_ROOT)

    # 2. The admissible construction, at the maximum size the ground set admits.
    construction = perfect_difference_set(int(instance["projective_order"]))
    if harness.certified_size(construction, modulus) != len(construction):
        raise SystemExit("the derived construction is not admissible")

    # 3. The seam witness. Its size floor IS the discovery value.
    witness = list(range(seam_offset))

    document = {
        "schema": "oer25.submission/v1",
        "seam_offset": seam_offset,
        "construction": construction,
        "declared_size": len(construction),
        "seam_witness": witness,
        "fitness_reported": fitness.score(construction, ENV_ROOT),
        "note": (
            "the construction is graded by the acceptance predicate and never by the "
            "proxy; the witness is carried to demonstrate that the two come apart"
        ),
    }
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
