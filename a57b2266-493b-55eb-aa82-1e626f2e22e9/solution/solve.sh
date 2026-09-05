# FORGE-CANARY-BEGIN
# 0: 132248831340d9b98f80493bf185065dd80ce0feb986d7888fe899c152f183cf
# 1: 7ac9f0f4a79692a121ad65ecad0a044c3fabef7607b6021d10ef0553cef8b4a8
# 2: f7e5be4ce7eb4b78e5cf978d8747ab2caa32e2b09e43907555b1bc5dfeb17040
# 3: 5b281185ee15ac1f288a233a23310a0007c01251bb8f6cbb2edccdeb484246c1
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
