# FORGE-CANARY-BEGIN
# 0: dace09c57f24032660a599e830779dbfe9a086ba5a1eb793d152d86a7941ccba
# 1: dd93987b718974379923f0a9cbe3621c430086f74db3309ce13da2cf3ea7c87e
# 2: f9081c6d45ece21231e4a2bbe6ffdfce832fff55d40752601fc2de05c584d7e3
# 3: f5f88a6d91cfbcff4daa6da6fe81a1b622a0535752cb8d873cce5ad44dd4c2ce
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
