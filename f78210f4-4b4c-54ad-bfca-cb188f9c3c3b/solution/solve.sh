# FORGE-CANARY-BEGIN
# 0: 20ee7cde75d8e3cc6e92df40ef649b63a167bae72fe385f5723cf56686b386bc
# 1: 2300bd983caaf93d76d70df324c8607f76b8f2ffed9154c109feb9012b263a45
# 2: 4d51721fe7f7bf5940651b3727bb6dc80813cb82bcc0452ef5a99676354a93ef
# 3: 6e8a284bdafd830c2f0785a79eb03dec227cdebabd2d5c27c9f5d78c3257688f
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
