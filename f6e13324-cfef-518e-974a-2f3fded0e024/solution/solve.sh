# FORGE-CANARY-BEGIN
# 0: 5940a391f91d83f923f38d0f619671d5554c4513b5424ee0d78d0661af1b6d7a
# 1: dab6a283b6fadd1f96d097c689ec6c0aa48915f835f2c833d5eb33f0b13f3257
# 2: 35b4c35bf036557b36019872e30d5b938b2cd906ef730017b65202c35d2029e7
# 3: ed4b790fd79e0d786752d3cd5e62f9d50d9133384510599be8ca735c5e3af149
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
