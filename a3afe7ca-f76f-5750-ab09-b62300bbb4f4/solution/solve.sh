# FORGE-CANARY-BEGIN
# 0: 768d88aa902f41bee965fc87277492406c70a1731386fed4a52361ce56bf4f63
# 1: 30c44125702e4cbe1af17f281705ae522f78620cf8bd9644403eb4e98d046315
# 2: e34053ed2aca9b7ad46aae0cfa64ce1d00cfe2047d5de400a79c2af60ca4d5d7
# 3: 4d36e49bb8c224e64335082ef39c7570b7930954df951804d5691972e8535d93
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
