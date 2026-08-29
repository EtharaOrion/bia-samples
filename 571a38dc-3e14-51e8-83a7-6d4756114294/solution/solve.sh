# FORGE-CANARY-BEGIN
# 0: 0fc4bdccec0b715501aa889d9b132875e6341ee074de44986398e5c1a9810716
# 1: 6407e53913f683ff7cf42bc972267461d7da9bf7f21ceddbe57fa09ebcd5c2f2
# 2: 5b35651c6077db91dc0477f2cac3702ddc1846fdd942cb0244fa2e3eebe9adf9
# 3: 6c28b320fb544e056ad95c9755b938e79dc3a3e91bbf3150990ea07973c73072
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
