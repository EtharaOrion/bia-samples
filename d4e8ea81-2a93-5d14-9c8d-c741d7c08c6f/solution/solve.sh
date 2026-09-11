# FORGE-CANARY-BEGIN
# 0: 3fd17d282a559782a7dcf78865f8ef6637aa88f790b1cfbd74fc90c3620a1a8b
# 1: c31108c6649649f90bb8b7c40ea696b7f15dfaab37536b9038a558d69cd3c653
# 2: 0d5b714c9ba7b80e1b7dd057199b8190c010f76e37dff28612981a63f96a8eb2
# 3: 75e8a008253fcae56780e20799327745c7213ba41c51c2b4643f2eedd269b17a
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Derived from solution/grounding.yaml by solution/recompute.py.
#
# OER-13 solution entry point. Harbor runs this file and nothing else.
#
# The reference composition is carried in solution/reference.py as source
# strings and materialized into a submission tree shaped like environment/, so
# it is graded through exactly the path a submission is graded through.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
# WHERE THE COMPOSITION LANDS IS THE SHARED WORKSPACE, NAMED LITERALLY.
# The previous default was "${BUNDLE}/../oer13-solution", the PARENT of the
# staged bundle tree. That directory is private to this container and is not one
# of the carriers the graded process reads, so the materialized composition never
# crossed into the verifier: the oracle exited 0 having written a full submission
# tree that nothing would ever open, tests/grade.py:210 fell through to its own
# default of ${BUNDLE}/environment, and the run graded the PRISTINE DELIVERED
# BASELINE instead of the reference. Baseline against baseline scores
# no-end-to-end-improvement by construction, which is why the reference arm and a
# wrecked arm returned the identical 0.0.
# tests/test.sh hands the same literal path to grade.py through OER13_SUBMISSION.
DEST="${OER13_SOLUTION_DEST:-/workspace/submission}"
mkdir -p /workspace

python3 - "${BUNDLE}" "${DEST}" <<'PY'
import sys
from pathlib import Path
bundle, dest = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(bundle / "solution"))
import reference
out = reference.materialize(bundle / "environment", dest)
print("materialized the reference composition into " + out.as_posix())
PY

# The three files the reference actually changes, with the digest each carries
# when materialized. Recorded so a reader can check the tree that was written.
#   pipeline/compose.json  sha256:5a97573075238983da3ccbd2d109a7f600a3515bd10ea2c6c1df13f5174406ab
#   pipeline/parse.py  sha256:f9655e33bb14dfa01a927399338db570b9dfef9856b220985bcd2b76e313bef6
#   pipeline/tokenize.py  sha256:ac3e5b3ccbe2a10d42090cc6b4e16490e9e2df2aff9c1746ac4562570277d165

echo "reference materialized; graded quantity is validation loss of the trained model on the frozen held-out split, direction lower"
