#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.

The reference solution for slot OER-21.

It emits one artifact: a quantization scheme. It reports no metric, installs no
filter on the graded path, and asks for no early stop, because the graded number is
recomputed by the verifier from harness-owned quantized state and nothing this file
prints can move it.

How the scheme below was derived, in one paragraph, so a reader can redo it:
the handed toolkit in environment/quantize.py can only set ONE width for every
tensor. The model's per-tensor gains span roughly eighty to one, so a uniform width
spends the same bits on a tensor that barely moves the logits as on the tensor that
dominates them. Widening to per-tensor and per-group allocation under the SAME total
budget, and paying honestly for every scale and every centroid table, buys back the
bits the uniform allocation wasted. The allocation below is the fixed point of a
greedy coordinate sweep over that wider space, run in solution/recompute.py.
"""

import json
import os
import sys

SCHEME_JSON = r"""
{
  "graded_readout_filter": "none",
  "halt_after": null,
  "scheme_version": 1,
  "tensors": [
    {
      "groups": [
        {
          "bits": 4,
          "clip": 1.0,
          "codebook": "affine",
          "size": 12
        },
        {
          "bits": 4,
          "clip": 1.0,
          "codebook": "affine",
          "size": 12
        },
        {
          "bits": 4,
          "clip": 1.0,
          "codebook": "affine",
          "size": 12
        },
        {
          "bits": 4,
          "clip": 1.0,
          "codebook": "affine",
          "size": 12
        }
      ],
      "name": "t00"
    },
    {
      "groups": [
        {
          "bits": 2,
          "clip": 1.0,
          "codebook": "symmetric",
          "size": 48
        }
      ],
      "name": "t01"
    },
    {
      "groups": [
        {
          "bits": 6,
          "clip": 0.95,
          "codebook": "symmetric",
          "size": 48
        }
      ],
      "name": "t02"
    },
    {
      "groups": [
        {
          "bits": 3,
          "clip": 0.95,
          "codebook": "symmetric",
          "size": 48
        }
      ],
      "name": "t03"
    },
    {
      "groups": [
        {
          "bits": 5,
          "clip": 1.0,
          "codebook": "affine",
          "size": 48
        }
      ],
      "name": "t04"
    },
    {
      "groups": [
        {
          "bits": 2,
          "clip": 0.95,
          "codebook": "symmetric",
          "size": 48
        }
      ],
      "name": "t05"
    },
    {
      "groups": [
        {
          "bits": 4,
          "clip": 1.0,
          "codebook": "symmetric",
          "size": 48
        }
      ],
      "name": "t06"
    },
    {
      "groups": [
        {
          "bits": 2,
          "clip": 1.0,
          "codebook": "symmetric",
          "size": 48
        }
      ],
      "name": "t07"
    },
    {
      "groups": [
        {
          "bits": 3,
          "clip": 1.0,
          "codebook": "affine",
          "size": 24
        },
        {
          "bits": 6,
          "clip": 1.0,
          "codebook": "affine",
          "size": 24
        }
      ],
      "name": "t08"
    },
    {
      "groups": [
        {
          "bits": 3,
          "clip": 0.95,
          "codebook": "affine",
          "size": 48
        }
      ],
      "name": "t09"
    },
    {
      "groups": [
        {
          "bits": 4,
          "clip": 0.95,
          "codebook": "symmetric",
          "size": 48
        }
      ],
      "name": "t10"
    },
    {
      "groups": [
        {
          "bits": 3,
          "clip": 1.0,
          "codebook": "affine",
          "size": 48
        }
      ],
      "name": "t11"
    }
  ]
}
"""

SCHEME = json.loads(SCHEME_JSON)


def main() -> int:
    out = os.environ.get("OUT_DIR") or os.getcwd()
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, "scheme.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(SCHEME, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("scheme written to " + target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
