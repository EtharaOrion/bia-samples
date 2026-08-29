# FORGE-CANARY-BEGIN
# 0: 3aed08d15d25cab09d76d634cdb878de8afcb9e212d1b207dd1e1e3bddae6c3a
# 1: 86e2b21e4a2631a2093234bc8670f483055a8abcba739ec6ec38543c7f434029
# 2: e19e99ba1714469bdafc3f3391f98a15515eadc0f93fd81bde4717e5119fd86f
# 3: f5642395b4d023860e25b2db6d10f8a65fdafbb67144d8627dbc29060548518f
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
