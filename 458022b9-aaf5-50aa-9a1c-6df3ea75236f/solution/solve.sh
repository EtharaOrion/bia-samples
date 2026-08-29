# FORGE-CANARY-BEGIN
# 0: 704a3c89ddbc115646ab496f8ac2d6fbcff066870d549e1a3dbaf6db8d7014a0
# 1: e4fed455b0b4b2526d2974fa08330b1374d8961caf518e46bd132f2e80424432
# 2: 4ab1f452d853cb6bc18d7ba6fb660d5ead1ed944f0401113d1c28c99119ec694
# 3: febac1621f9052bc16d941f3db288fbefb677724fd18dd18297783d25c9e644b
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.
#
# Harbor entry point for the reference solution of slot OER-21.
# It runs solution/reference.py, which writes the quantization scheme this slot's
# live checkers accept. It reports no metric and reads no clock.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$PWD}"
export OUT_DIR
mkdir -p "$OUT_DIR"
python3 "$HERE/reference.py"
# measured reference optimum, worst-shard perplexity degradation: 0.107170959
# measured optimum of the handed option set: 1.188591942
