# FORGE-CANARY-BEGIN
# 0: 3d7f5a8784f6c36f9741da0cb27407c603bb94ee35e3931125bf7660d342dc2c
# 1: 15323167dbe28b28b03069fe0861ff0e89644d90286e223adabccf3e1ba21226
# 2: 3ddf7beda990d05052610105baee9516d37283869d0c4e76b715100d903371bd
# 3: 8c419d50500d53e715843cb871144513500ea5c9577f1dc1a44e9f4724703343
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
