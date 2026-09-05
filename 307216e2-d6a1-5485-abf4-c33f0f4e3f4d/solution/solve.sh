# FORGE-CANARY-BEGIN
# 0: 41f4a515980409005c78907f3a7236547aa071e8f2e6fb0e3ea4fd63ff444c1c
# 1: 20bf25a636d3d4a0e42971776eb34a2fe5c971d729466a519a10b2f122bd2c20
# 2: 8f9752746c4f33473319bf316efe7cffb59d49b22920db174dac768b87f2498e
# 3: 89d9d6720750e1050ab5e4397ab58474953c95ef890d8273ab26cbcc60f926b7
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
