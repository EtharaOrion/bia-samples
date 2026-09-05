# FORGE-CANARY-BEGIN
# 0: e8b50d2d3535a20519030e7f743b934e36b6519acf29cff18cea3b91a8979ede
# 1: f65672e32c3669b3e28a262d79295b7b0debd4cc4af94d74f4b56de4fb7b75f1
# 2: 69b6c593bc3369f2b742a66c8b5f7b5f6059cff5753d04edf606bcd5de396069
# 3: 9d2ebc0507f480668249120177273043c148cdf590a1b93b195ddef4535f8be9
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
