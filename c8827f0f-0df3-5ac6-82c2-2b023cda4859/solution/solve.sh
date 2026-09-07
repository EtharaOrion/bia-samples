# FORGE-CANARY-BEGIN
# 0: c1238b9430a8c23e5d44dfbd160ab3664980b7313c299bf6bbab52d992265684
# 1: 6dcace44161bc61b464df357293728475e40524243ee3b55fc2532c9952a9662
# 2: 75da5a38e6e71f8a00cfc7a6dd86a4d4dceb85388ad0755bc4ba9ef3647d9ae5
# 3: f35eda0acfba5ed44d47aadc90fbce9e3dcfedd82565c76ff98e2ef5ba37176c
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# source: solution/grounding.yaml
#
# The reference solution entry point. It writes ONE document: the vocabulary
# construction plan. It reports no number, because every number on the graded
# path is the harness's own, produced by training the frozen decoder itself.
#
# WHERE THE DOCUMENT LANDS IS NAMED LITERALLY BELOW, and the default is the same
# absolute path tests/test.sh resolves through the same SUBMISSION_PATH variable,
# so both halves of the join stand in frozen bytes and neither side has to guess.
# The previous default was this script's OWN directory, ${HERE}/submission.json.
# That directory is the staged solution tree, which is not one of the shared
# workspace carriers this bundle names, so the emitted document never crossed into
# the process that grades it: an instrument reading what the oracle wrote found
# nothing at the plant point and fell back to planting the reference SOURCE in its
# place, which tests/runner.py then read as the graded document. The document is
# composed in a staging directory and INSTALLED in a second step so that the
# destination is a literal a reader of these bytes can see, rather than a bare
# ${OUT} expansion that states no path at all.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_PATH="${SUBMISSION_PATH:-/workspace/submission.json}"
export SUBMISSION_PATH
OUT="${SUBMISSION_PATH}"

STAGED="$(mktemp -d "${TMPDIR:-/tmp}/oer14-solve-XXXXXX")"
trap 'rm -rf "${STAGED}"' EXIT

python3 "${HERE}/reference.py" --emit "${STAGED}/submission.json"

# The bound install. This line names /workspace/submission.json literally.
mkdir -p /workspace
cp "${STAGED}/submission.json" /workspace/submission.json

# And, where a harness pointed SUBMISSION_PATH somewhere else, the same bytes go
# there too. The bound path above is written either way, never instead.
if [ "${OUT}" != "/workspace/submission.json" ]; then
  mkdir -p "$(dirname "${OUT}")"
  cp "${STAGED}/submission.json" "${OUT}"
fi

# The allocation this reference reaches, for a reader who wants it without
# running anything: {"merge_depth": 18000, "span_units": 24000, "numeric_units": 5000, "punct_units": 0}
echo "wrote ${OUT}"
