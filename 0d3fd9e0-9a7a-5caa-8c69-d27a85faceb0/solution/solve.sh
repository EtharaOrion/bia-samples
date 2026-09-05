# FORGE-CANARY-BEGIN
# 0: 5a5624d570bdc558dc640e781e8eb1186892d161cc5b49b06891ce428e6a7f77
# 1: b9e0a866358ceedaf6ec298db480e47f80f6663f6ddec55b5b7353163140a0e5
# 2: 9b7605ef93fa005e51b56c3eaf0f9e519b77c521e6d47a6a84211325878c23a7
# 3: b355b94f5e6953ca448d414f76d0cc478148cbfae076810a84667d6e76c89e86
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
