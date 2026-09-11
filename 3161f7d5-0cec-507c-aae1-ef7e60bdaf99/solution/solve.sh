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

# The submission entry point, under the same name tests/test.sh declares and
# tests/grade.py reads, so one variable carries the join and both halves of it stand in
# frozen bytes. It SELECTS what this oracle runs rather than decorating it: a harness
# that points FORGE_SUBMISSION at another entry point gets that one run, and an
# unresolvable one falls back to this bundle's own reference rather than aborting.
FORGE_SUBMISSION="${FORGE_SUBMISSION:-${HERE}/reference.py}"
if [ ! -f "${FORGE_SUBMISSION}" ]; then
  FORGE_SUBMISSION="${HERE}/reference.py"
fi
export FORGE_SUBMISSION

# The artifact is composed in a staging directory and INSTALLED in a second step, so the
# destination this oracle puts its artifact at is stated in shell instead of only inside
# a Python open() in reference.py. A write named only by a `${var}` expansion states no
# destination a reader of these frozen bytes can see, and an instrument deriving the
# plant point from them is then left with whichever directory the phase happened to
# write into. The install below names scheme.json, which is the basename
# tests/runner.py fixes and the artifact the verifier takes back.
STAGED="$(mktemp -d "${TMPDIR:-/tmp}/oer21-solve-XXXXXX")"
trap 'rm -rf "${STAGED}"' EXIT

OUT_DIR="${STAGED}" python3 "${FORGE_SUBMISSION}"

mkdir -p "${OUT_DIR}"
cp "${STAGED}/scheme.json" "${OUT_DIR}/scheme.json"
printf 'scheme installed at %s\n' "${OUT_DIR}/scheme.json"
