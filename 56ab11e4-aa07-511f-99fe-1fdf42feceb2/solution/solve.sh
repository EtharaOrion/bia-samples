# FORGE-CANARY-BEGIN
# 0: 4a108676ee68b3363ab751af68f818826827e69518f6e28f52a2384ee93d4818
# 1: 1e378b7e29145e9fd16171a90368fedeec816bff67b28bb0e1b8feb79415938f
# 2: 3d550ae8a7e277bbc322a2c3fb41c6e5f3952b4f727ecbe0898cfde023eba37f
# 3: 3baa1ac4f9692612ae025a8b2ab41659c2e91e38367ec0a1313184dc214655bb
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
