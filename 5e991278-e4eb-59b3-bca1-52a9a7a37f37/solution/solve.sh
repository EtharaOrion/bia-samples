# FORGE-CANARY-BEGIN
# 0: 447246c57ce9f0727b47262a5929a2fe4270d88ee564f500b325a3018bf198ef
# 1: 3220d405dbf61bbe8d2340a83f675d0abcc8cec38e375eae8103b3ebd427d1d8
# 2: 226106137811da46a9fd4c31c0a327eabc6b4a44c4cc19a2bb61f45b40152205
# 3: 09232666ade9334939b5dbbf646623777d41b55047c3ccd0ef43af2d2e403a89
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
