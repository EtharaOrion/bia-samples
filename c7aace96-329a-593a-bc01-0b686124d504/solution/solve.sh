# FORGE-CANARY-BEGIN
# 0: a5be95c60afd4f6e9e7495d09c4925522379fbcb1a0564b0d457fd663db99c5d
# 1: 9822ec3cc55b76e2efdb084e4e84da37523ebe5936654f7c22e4d6ec3bbc6a8c
# 2: 3d15f95604744713094bd1069af977eff2aaaae992e7bca70d6875bab20b2abb
# 3: 05efe6b4e1471df88564b3b91575a375d5c975054bf73262b7ec0c35cf97c719
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml
#
# The reference submission. It writes exactly one artifact, allocation.json,
# and reports no perplexity number, because the graded quantity is computed by
# the verifier over harness-owned quantized state and a number written here
# would be ignored on the graded path by construction.
#
# scheme: error-feedback
# allocation sha256: 7b39dd1ca7b87851867580c4bab4fcb52737858613a7cc905b1f516d13d1de92
# allocated bits: 147849216 of 147849216 budgeted
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${HERE}/reference.py"
