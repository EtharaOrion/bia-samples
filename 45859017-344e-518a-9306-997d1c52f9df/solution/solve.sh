# FORGE-CANARY-BEGIN
# 0: 79b9eea575f50770ab4033160cc84fd63fdaa15c9361fdddc08ef2450d6bb3dd
# 1: d53308fb29389130346ddf5701caefe17f6d91b111e77968c25c03508261424e
# 2: d978cbb2cfc844ee72e8428acf9485b68ba8fd54b73fc8b7fb18b0cb9ed975f0
# 3: d2e6ae01a0758c00d9a9f58d2c282329316c3f9ec5d54b42afbb9eb25d069803
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
