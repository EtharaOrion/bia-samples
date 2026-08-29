# FORGE-CANARY-BEGIN
# 0: c961f2aeddb578538ca1f89f941edbaaa1c2ae4f9ba946bc930d18d3ae9d44fd
# 1: a7a5c749bb0bc2bb410ed021ea54f9decd453d593b7e6112211b309ffd9b2a4e
# 2: ae5e18fc5f32709891a5ee0ad1e117fa20f1da99533ecb9d6d7b02957263a8a8
# 3: 2100546219d30e42fbe931e5ec61a388a7c3d3a5d3aaa5e322d60ea8b71a06e2
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
