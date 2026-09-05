# FORGE-CANARY-BEGIN
# 0: a047811ee504df42ac07bf127e7c1fae800b910ee467803f747b126e1772f46d
# 1: ce4daf82e37027dc0f7754fbee443378b67bd8beabf7cf0975f9d541efa3de7b
# 2: be6e33e0da42bcfc8f6857d409b0c6b38d0283c54cfa00f4e329ff6516f9d8b1
# 3: 9d6630c57eca96f2613c7f43a8431a5b6b82792e3d6963171909ea30c8e3962e
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
