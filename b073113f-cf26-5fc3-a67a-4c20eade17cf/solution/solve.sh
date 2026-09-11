# FORGE-CANARY-BEGIN
# 0: 4446e19ded362b67f020d183b2d339857274b2077eb2500d77bc2edcc0602bf2
# 1: 264e19bc1834ac4f8f98bdd097d1b97902eaedac8b74c38f9889d206a75421ce
# 2: 83344103cd6ae4050e94765bb36048b099bf32888d6134fa94987d17bc96d9cf
# 3: 76d6c96be27251cb4d70d0963c485d69c61cdd9e24e64d7b68ad6c5c3ddf4a57
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
