# FORGE-CANARY-BEGIN
# 0: efa003e58a0090a1d38adb0f5aab1cf2272f96b96b4608a2a9e680c6487d4989
# 1: 7cff204598afab87d9f2d86caaea77efc11b440f7273fc7e18526716c3808012
# 2: 1015ed9a061ff19cf24c3f819fa3049b20d8dbaaded96935730dcc533a525fad
# 3: aa1ac7c9675468e1a22580d500737fc07ac1c9228601f32372d85574ff5984c1
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
