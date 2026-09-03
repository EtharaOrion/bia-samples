# FORGE-CANARY-BEGIN
# 0: 0e4502d5c7dc4889363c67268ce78057db10ff2767d22c2f3b6e38d58193d0f7
# 1: c04caf9339ef5e34ef1c50a5bba89e138d562acaf1631b7d955730aabc23ea72
# 2: 4e1079d41ea3a5b2f373b12bb4361a80b7125ba9266723cf416ebd73cbaaacb7
# 3: bce9235963ae979809457d5ffee51a746cb43d9a1de863bbd0ab575eb73719e1
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
