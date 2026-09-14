# FORGE-CANARY-BEGIN
# 0: d371107a2c7db6ffe05530324c366c906fe5db11ff837327b8f4be088bac8f25
# 1: d02c41ae56becaffab6f21b78e1653db9419e7d4db58ed29d2efaf865a6fb881
# 2: 4b77f9b7e1fcad15900fa070be984fc482fee26e2c7864e5b22ce0564bd09846
# 3: d4bc4a7a12aa3d1438003de3335ca41a4f91094598d1b8ffdc78d91dade154b8
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
# Its option grid has 60 rows, and its best row reaches
# 0.33735183879296543 bits per byte. The corpus carries only
# 201 distinct word types, so vocabulary budget past that point is
# unusable by that construction at any option value. The reference construction
# merges frequent adjacent pairs instead, reaching
# 0.2818765313056418 bits per byte, a gap of
# 0.055475307487323655 below the plateau.
#
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="${OER15_SUBMISSION:-/workspace/submission}"

mkdir -p "${SUBMISSION_DIR}"

# The submission is the reference construction and nothing more. No
# HALT_AT_UPDATES is declared, so the run reaches the bound evaluation point,
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
