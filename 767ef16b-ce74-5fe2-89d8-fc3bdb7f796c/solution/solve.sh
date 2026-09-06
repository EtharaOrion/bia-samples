# FORGE-CANARY-BEGIN
# 0: 74c7e3c0ffaaf3499a203965cab7b89097d493d631c20965c1c5e02f35ea7b2f
# 1: 64c35f41fc214eddc07dd053c575314c24dacbc52832c06b850a57e20cf5346f
# 2: 3f07a9ad73e30bde5933096c340c9b9bcae45ff7e1dddc7c7aae7000e898cff4
# 3: 18e415cc67d34bf9323914d6ef3a5af6288dee363c1b1c3b43448af9020e14ff
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
