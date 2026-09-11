# FORGE-CANARY-BEGIN
# 0: 2ce002c9f4bcf6ccb53c6e2bf7d278e25219724955fd0bc92414fb4aa81b2893
# 1: 66844bd67ccb4acaa9067c1df53524cac941b284a16f4438e5a9cc8c9aca3f39
# 2: 31591f861448a5f9440fb70debc1e27f891ef10a616c134767932c2084d05fcd
# 3: c845e780730c0b6a7ba4a75b920136462a2003fd39bf0d67589184abf782d33f
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
