# FORGE-CANARY-BEGIN
# 0: 6df36ec44c83ea4a95f287b98a181542bc412082a3cb57ba783dea08e720d054
# 1: 8be5825b72038f7630ab4160bba3059ccce04665d2931848dc1ad760b5afaf21
# 2: 6967231af56267feeb7607ee159b599b4fffacbc1f2ae430263acc6c6924cb36
# 3: 22267c05caaa7e5bb880839ddd625cd5656ee8daeb7f849af0c0e318fcbeba0e
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
