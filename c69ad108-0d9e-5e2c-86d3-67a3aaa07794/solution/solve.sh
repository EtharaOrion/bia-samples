# FORGE-CANARY-BEGIN
# 0: aa6432caab6ad04f2295af2ef64939ffc510effaa276ad6becdada6e81fb55dd
# 1: 06a425a18b1f067eae39b908c0bd727a1069fbe6c97aca7b04e00019bda90387
# 2: 778fe1630a7ea025f915bb1387868f5e4ce47c90e80f1bd1135c0adc5da1fe1f
# 3: 4d1aef41240f47d7b3e730b21b4c4efa2684bb978eae9ceef682a183acf4e5e7
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
