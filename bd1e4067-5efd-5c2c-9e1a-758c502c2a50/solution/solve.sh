# FORGE-CANARY-BEGIN
# 0: 64e751632b629c242c7748958fa8faf3d5d23d669e31a24495680c48de4e5f9a
# 1: 2f50cd4d62bbec0aca5a78ffc1b2c94a86b3e94af1c6ac2e7e4ae4388a7df699
# 2: 9ef7191d6411bad68f8278072b586f0a2477bf1ca016e49dfa280a1dfeb4ca91
# 3: 8634182244d750003aba62529b95dd4f5200fad8920feb34f414820e8da6145a
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
