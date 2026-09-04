# FORGE-CANARY-BEGIN
# 0: dde7b666e2e917b1e0b00c51ba5d89c011229f44027fe25a26110b0bd0de74bb
# 1: 917daee7a6f988c92cfe1f23e54be57b6a9bb2916be012b63f3819491a8c8644
# 2: d7d3d5f9a6378d5953bb0eb9dd050ba6dc69b4309fc5ffb5dc1cec5f63e74606
# 3: 2a0484486e3520294924163c66870e7802823b04a2d17c15f86805968aae83db
# FORGE-CANARY-END
#!/usr/bin/env bash
#
#  GENERATED SECTION. DO NOT HAND-EDIT.
#  Source: solution/grounding.yaml
#  Generator: solution/recompute.py
#  Edit solution/grounding.yaml and re-run the generator; never edit this file.
#
# OER-15 solution entry point. It installs the reference vocabulary
# construction as the submission and does nothing else: the graded number is
# computed by the verifier from harness-owned state, so a solution that tried
# to report a number would be reporting one nothing reads.
#
# The handed construction in environment/default_tokenizer.py is a word list.
# Its option grid has 60 rows and its largest top_k is 1024, against
# 50048 entries available past the single bytes, so every row leaves at least
# 49024 entries unspent at any option value. The reference construction
# merges frequent adjacent pairs over the decoded FineWeb bytes instead, so it
# produces affixes, whole words and multi-word phrases and spends the whole
# budget. Its bits-per-byte reading on this substrate is unmeasured-on-nanogpt-substrate,
# under gap gap-oer-15-operating-points-unmeasured-on-nanogpt-substrate.
#
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="${OER15_SUBMISSION:-/workspace/submission}"

mkdir -p "${SUBMISSION_DIR}"

# The submission is the reference construction and nothing more. No
# HALT_AT_STEPS is declared, so the run reaches the bound evaluation point,
# and no REPORT is declared, so no self-reported number exists to be read.
cat > "${SUBMISSION_DIR}/tokenizer.py" <<'OER15_SUBMISSION_EOF'
"""Submission: the reference vocabulary construction for OER-15."""
import importlib.util
import os

_REFERENCE = os.environ.get("OER15_REFERENCE", "REFERENCE_PATH_PLACEHOLDER")
_spec = importlib.util.spec_from_file_location("oer15_reference", _REFERENCE)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)


def build_vocab(train_bytes, budget):
    """Greedy corpus-derived pair merges, deterministic at every tie."""
    return _reference.build_vocab(train_bytes, budget)
OER15_SUBMISSION_EOF

# Bind the reference path into the installed submission, so the submission runs
# under tests/runner.py isolation without an environment variable surviving it.
python3 - "${SUBMISSION_DIR}/tokenizer.py" "${SOLUTION_DIR}/reference.py" <<'OER15_BIND_EOF'
import sys
target, reference = sys.argv[1], sys.argv[2]
text = open(target, "r", encoding="utf-8").read()
open(target, "w", encoding="utf-8").write(
    text.replace("REFERENCE_PATH_PLACEHOLDER", reference)
)
OER15_BIND_EOF

printf '%s\n' "installed the reference construction at ${SUBMISSION_DIR}/tokenizer.py"
