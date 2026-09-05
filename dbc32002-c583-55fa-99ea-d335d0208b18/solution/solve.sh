# FORGE-CANARY-BEGIN
# 0: 2044473e2a13a9254e1cb7d1741ce8b1080aee1c8554b51f029aa766699e9a42
# 1: bc30f870bc9a85aea4080628488c8f108d40e0d26fe8faf8889d62fe1985ceee
# 2: c0accbdf405bfc72871c91d945ae7a2d399c065d349698e35f5fadcde2023fbd
# 3: 4c735941c5c8b7681f871e1af1296c1b33dde8c14ca52ce579cbc644e89a0fb6
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
