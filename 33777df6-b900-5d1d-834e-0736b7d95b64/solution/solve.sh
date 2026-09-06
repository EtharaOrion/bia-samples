# FORGE-CANARY-BEGIN
# 0: a712aaeb4a46f768f410843e5eddb9dc67f3aea5cee6f7c7da24c60879c47c3d
# 1: e30ad8eb63877739f723160a4907f74cadd22ac07c6dfaea03982bdca59204cb
# 2: 1e1633ba18dd492c03c1dbbf3c4e751014182a0ac2dba91744b60fc993f684ea
# 3: 1d5009e0645c75d74233e23abd59822d2a7474c82bdb48fa0c08231338dbf960
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
