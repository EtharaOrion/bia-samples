# FORGE-CANARY-BEGIN
# 0: f6fe098335b6e4a1c809df2e8b5efd34777891792c4b80e1c94e5fcb28f0225d
# 1: b2f0b1df308af2efb739219f48b0c7a0b8794423c5879d70ad8a562a338ab89c
# 2: 4e2a77fa02a94d0e28c0713351e7a37f390433f4a1e88b1abe90c54bc40896c0
# 3: c33c9a52bb54a2974551a5da704f5a5763ff5085346998049cb9b73e57bf141a
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml
#
# Reference solve path for OER-11.
#
# The shape of this script IS the lesson: the snapshot version is asked, work is
# done, and then the snapshot version is asked AGAIN before anything derived from
# the first answer is committed to. solution/reference.py re-resolves the corpus
# whenever the handle it was given has gone stale.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${HERE}/../environment:${PYTHONPATH:-}"
cd "${HERE}"

python3 -c 'import corpus_api; print(corpus_api.snapshot_version())'
python3 reference.py --probe
python3 -c 'import corpus_api; print(corpus_api.snapshot_version())'
python3 reference.py --build
