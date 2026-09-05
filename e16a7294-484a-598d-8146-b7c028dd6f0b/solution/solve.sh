# FORGE-CANARY-BEGIN
# 0: dc17185733cbe5c35bb18997dd0981af54baed4ffc9190d6cd9470ecee01fdd3
# 1: fcf4d1261011af1db518fe61c43433db7174e3db8d28ec0ab3e218e6bffbef24
# 2: 9a15877a7e975c8834710e0525fd39c4f9273f168d1cc5d61beaeb3d393e9b85
# 3: bfcbf481d32fb759f3ff84c3f5a73c392717a08afa19322dde64c3b15188d864
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.
#
# Harbor entry point for the reference solution of slot OER-21.
# It runs solution/reference.py, which measures per-role sensitivity on the
# agent-visible calibration split and writes the quantization scheme this slot's live
# checkers accept. It reports no metric and reads no clock.
#
# No measured optimum is written into this file. Both endpoints of the bounded scaling
# are recomputed by the verifier at grading time, so there is no number here to go stale.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$PWD}"
export OUT_DIR
mkdir -p "$OUT_DIR"
python3 "$HERE/reference.py"
