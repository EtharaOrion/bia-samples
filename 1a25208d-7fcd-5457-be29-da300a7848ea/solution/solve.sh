# FORGE-CANARY-BEGIN
# 0: 1f32bc2d5dc1755f33215d80ac26dc48e76da4ac8fb31609386db9a9795fd060
# 1: 3b30d08796d568e3247aaf32820eea268fe0928d654a0c6956963bc8985db89e
# 2: 7199a72da7b4b57378145180051902370456e179a1e4279134e16783768ecdc3
# 3: baaa4c51ba840bbaeec3441e9a9e89c0d7720527811151de14821c99b8e4a90a
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
