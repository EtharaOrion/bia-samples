# FORGE-CANARY-BEGIN
# 0: e69c10c0827e5d2bcdb98bc399c4290a8448828703c307b3c2adba67030c40a3
# 1: 8944bfb31a500a5988f113b340032730c9c89dd768d7a2182d06fe1ac99eaf8a
# 2: 3cfdf6a451c3de420caa8a739a71188e1d2a540d9f0e715cc2ab180d0d9c5928
# 3: da1c5813670a43f418d54d5ce8ac21f105617020fae0ecc65b6c7606654beb31
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
