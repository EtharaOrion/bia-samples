# FORGE-CANARY-BEGIN
# 0: 7201759a47ef75d07837803be1f7b1e01c219932dc141431f150bd02c9f58ea3
# 1: 106ad2b6f015ee1c053901c8b4c33c2f6814b1128f2e9a09a816371500aaaa7b
# 2: c382df3c046b4dc6aa081c936c567bce40d87433e377796960310f41cb7c7aac
# 3: 044f6ffcbeca5c230ec725e2b765b18f0a2c7b6476ccae152ceb22e9908d5cac
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
