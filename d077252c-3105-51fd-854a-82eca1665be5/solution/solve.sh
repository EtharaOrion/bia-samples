# FORGE-CANARY-BEGIN
# 0: aa7a8486bd4b2ef30e916dff5a2fdcf1234748afd4e5282643c242d6d839311f
# 1: 45feee74d13fcb265e7ac227afc3a96bb6c7dbb2279769d5270a28038f1df4f3
# 2: a6c0b8350be71c6304929a77083ceb9821e4c16e07248a150714630f265d06e5
# 3: 9fbe9ad21e05246c25283f9d0803de45aa02f8182b85f2248863c7e61bdee42e
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py
#
# Oracle entry point for slot OER-03.
#
# Installs the reference solution as the submission and runs it under the
# pinned harness. The reference crosses at step 2600 on the verifier's
# cadence and carries a schedule of 3000 steps, so 400 steps of schedule sit
# past the graded point and buy exactly nothing. That is the point of the
# reference, not an accident in it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "$HERE")"

mkdir -p /app
cp "$HERE/reference.py" /app/submission.py
cp "$BUNDLE"/environment/bia_harness.py "$BUNDLE"/environment/bia_loader.py "$BUNDLE"/environment/shape.json /app/env/ 2>/dev/null || true

cd /app
python3 /app/submission.py

# The reference never decides when it crossed and never reports a crossing.
# The verifier recomputes the graded step from its own evaluations.
