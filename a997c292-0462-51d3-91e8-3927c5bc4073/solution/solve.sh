# FORGE-CANARY-BEGIN
# 0: 1200ac247edeabed313a452d6941fc67ba82a68f84a1e96baea201f039958a41
# 1: 4d303149bea8f7c8922e25476c84fdbc3e6bedd7714b430edbdf98483d3250e5
# 2: b68f179255691676b91d80444be982f765c7d6f6de57ea913666ed0bed6984e1
# 3: 88ee1f7f9876803edd627ab2354358a4e7782fac9934e9c34127f717a97a6665
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
