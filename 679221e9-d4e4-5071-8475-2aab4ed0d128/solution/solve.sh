# FORGE-CANARY-BEGIN
# 0: a2b6d6d5ae53c1be14d955f54e7c47de05c416b8e69bef2a9a861e6e56753979
# 1: c71709df082123c5322d44a1c71ff43d0a5530308f5d01d3cf28e6508db0c598
# 2: 3438a6f8c88f700d216bd28b9bbb772eaba9614be401816dd5445cae2ce32eaf
# 3: cf6dd14fed236ac11e2a8719cf11a0c1b6cf138d85d2e15806e5545f6a24bca5
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
