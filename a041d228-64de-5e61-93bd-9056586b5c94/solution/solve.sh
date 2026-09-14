# FORGE-CANARY-BEGIN
# 0: b538b521bc0b47f0fec125dc257cccf23354eeda71c80725eca01bc8a15394b5
# 1: 432c5f2f39ae38e593d7354926c6b68627d6c828a316280148aae1f22782095b
# 2: b7ee7a02868880ff5500559df78c5b210fda6a87e904eb6eb9a2f555970a343b
# 3: c15f6c34c684cb02460eed1d5b34b18cfac3e290ebb7abecf3fe7c7fbc81bce9
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml
#
# The reference submission. It writes exactly one artifact, allocation.json,
# and reports no perplexity number, because the graded quantity is computed by
# the verifier from real forward passes of the frozen checkpoint over a held-out
# split this file has never seen, and a number written here would be ignored on
# the graded path by construction.
#
# scheme: error-feedback
# allocation sha256: a4b31188e8e32d78b5583d336aad4240307a0c2e37cbacc9f79baec8dd1893cb
# allocated bits: 648806400 of 648806400 budgeted, over the 50 weight matrices of the
# frozen 12-layer 768-dim decoder
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${HERE}/reference.py"
