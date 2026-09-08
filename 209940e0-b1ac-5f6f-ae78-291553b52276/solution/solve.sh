# FORGE-CANARY-BEGIN
# 0: 6730373a0a5f93fe7765a9b6656b1df9552e34579d7275011e8de62a0f376ea4
# 1: 5673b576f3d5c61b903423fea124620c6c50d5d4b6f41c06f51c392b31be3801
# 2: 6b9b616d14a9a8b63be60213a583f19423b688cfb7a2161da25a30d2b40dbd15
# 3: 2b1e09652b5372af087279135e12fb4604ff1dcbeaf9d6b54b16a47cd89fc1f0
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It runs the oracle, which reads the built warehouse back through the
# harness handle, applies the exclusion rule, walks the closure to fixpoint, folds the closure
# into a plan and writes plan.sql and submission.json into the workspace.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
WORKSPACE="${OER29_WORKSPACE:-/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --warehouse "${BUNDLE}/environment/warehouse.db" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
