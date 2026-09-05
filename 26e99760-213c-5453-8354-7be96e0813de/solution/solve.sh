# FORGE-CANARY-BEGIN
# 0: 503758d3d1ea46206305a321483c580eab148fbef3f70b6555ca2c5856255c85
# 1: d57b6fcbdc2af8525073627882041b13681266e33457054e3f5419c74b03a02e
# 2: 941103f162ee0791836a516b88f7ea36a050094f76013aeeccc84f9a269ac743
# 3: 15c25f559205555ff3b0b686a7d2c6313af8778dc8c6720af37e95985a3ccde8
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
