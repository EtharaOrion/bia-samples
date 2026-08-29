"""Materialize the frozen synthetic corpus for a profile.

Run at image build time, which is the only moment egress is permitted, although this
generator needs no egress at all: the corpus is produced from a pinned seed rather
than downloaded. Running it at build time keeps corpus generation out of the graded
attempt's budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(HERE), "runner")
for p in (RUNNER, os.path.join("/opt/bia/s08", "runner")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import s08_core as core  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("S08_PROFILE", "scaled"))
    ap.add_argument("--data-dir", default="")
    args = ap.parse_args(argv)
    cfg = core.load_cfg(args.profile)
    data_dir = args.data_dir or os.path.join(
        os.environ.get("S08_STATE_DIR", "/opt/bia/s08"), "data", cfg.profile
    )
    meta = core.materialize_corpus(cfg, data_dir)
    print(json.dumps({"data_dir": data_dir, **meta}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
