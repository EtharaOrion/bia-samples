# FORGE-CANARY-BEGIN
# 0: fc72f14942bf6e6a20f97d111e7e4453461b73a257e466211c803360dd2d9191
# 1: 6cfa14a18606a1bcddd6f8d0b3e5fa04f7ae58bee4cf47f47855da1fb6a93a13
# 2: cebf0cb2f75f877e5d498e9fe016d106084179429e681ac287a8ca24a82a4518
# 3: fd74f4f7571bfbbdb69458220a828237701f1f742afad5bb3ba462675db4343e
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
