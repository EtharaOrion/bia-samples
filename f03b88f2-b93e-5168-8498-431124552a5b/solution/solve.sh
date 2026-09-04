# FORGE-CANARY-BEGIN
# 0: d4aca1641109052788ee8cc6aefb4d0f2db34795ee0f23cf09991206f1f98fe5
# 1: a969d91302d7ad1cea24738fd5de0451bb11ff415624329a97318ca69c0b900e
# 2: 98eb0933e633067f5f3425efbe234c214d2193890511c26b79779800378ce139
# 3: 68ee5cce2bc2372e1346413ba6214c1392136413d47c5902a90a9d4768ca5b38
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# source: solution/grounding.yaml
#
# The reference entry point. It emits the corpus and its own measured diversity
# report into the run workspace and returns the generator's status, which the
# verifier records and never trusts.
set -euo pipefail

WORKSPACE="${1:-${OER19_WORKSPACE:-$PWD}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="${OER19_TOOLS:-$HERE/../environment/tools}"

mkdir -p "$WORKSPACE"
PYTHONPATH="$TOOLS${PYTHONPATH:+:$PYTHONPATH}" python3 "$HERE/reference.py" "$WORKSPACE"
