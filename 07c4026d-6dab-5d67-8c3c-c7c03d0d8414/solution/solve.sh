# FORGE-CANARY-BEGIN
# 0: 06d6b01d97ddadad1e9192db439c06ee77305b22435b17a1742f4816a0ee52cd
# 1: 0dde6632b426a8558c7410ed48b2a2b99713728004d2a5a98bff0e9f339f4338
# 2: 3a0d0476165c77f901cd4bd72acb6228fd6071c34abdb899837f8b42866cceb3
# 3: 350d90cd2e4ee7bed67cee6c60901834623994f47dd60fd2f87028d70eb49aab
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Derived from solution/grounding.yaml by solution/recompute.py.
#
# OER-13 solution entry point. Harbor runs this file and nothing else.
#
# The reference composition is carried in solution/reference.py as source
# strings and materialized into a submission tree shaped like environment/, so
# it is graded through exactly the path a submission is graded through.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
DEST="${OER13_SOLUTION_DEST:-${BUNDLE}/../oer13-solution}"

python3 - "${BUNDLE}" "${DEST}" <<'PY'
import sys
from pathlib import Path
bundle, dest = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(bundle / "solution"))
import reference
out = reference.materialize(bundle / "environment", dest)
print("materialized the reference composition into " + out.as_posix())
PY

# The three files the reference actually changes, with the digest each carries
# when materialized. Recorded so a reader can check the tree that was written.
#   pipeline/compose.json  sha256:5a97573075238983da3ccbd2d109a7f600a3515bd10ea2c6c1df13f5174406ab
#   pipeline/parse.py  sha256:f9655e33bb14dfa01a927399338db570b9dfef9856b220985bcd2b76e313bef6
#   pipeline/tokenize.py  sha256:ac3e5b3ccbe2a10d42090cc6b4e16490e9e2df2aff9c1746ac4562570277d165

echo "reference materialized; graded quantity is validation loss of the trained model on the frozen held-out split, direction lower"
