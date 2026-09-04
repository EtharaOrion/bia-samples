# FORGE-CANARY-BEGIN
# 0: 4cffc02015ecacecc24d4e49ac54d4ab91d29942363833707d02b5fc61fbecad
# 1: 6c6158807f78733f2a2418c31c544ec6192eb25133af986752ab79ae6d0aee32
# 2: 2f6ffc78849226247a476e995e33b05c7422c10a7a72ea42138cb7aa28c01ad5
# 3: c1ce99099ba02fcbf1ca211b7e5a144329fcbf514338a6b3aa58cbef1b6069e8
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
