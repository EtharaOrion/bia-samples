# FORGE-CANARY-BEGIN
# 0: cd38c84bf5d49447fef39c5d53661920f00cb95e538b4b6214f0fb3a255d56df
# 1: d23af49a82dc8343c61158dfea514fd09adae30e6f977111f7af534660b5f825
# 2: 2718f6de221a0249b2abbe00add424d89e41168897cb67fd10419c946c66a93d
# 3: 1926da8b19c94632eb1ab639641eb74299ecc00dddbc191af92f89ecd617b9c7
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
