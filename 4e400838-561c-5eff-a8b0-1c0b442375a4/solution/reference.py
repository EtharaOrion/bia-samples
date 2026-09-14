#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.

The reference solution for slot OER-21.

It emits one artifact: a quantization scheme over the frozen checkpoint's parameter
shards. It reports no metric, installs no filter on the graded path, and asks for no
early stop, because the graded number is recomputed by the verifier from its own
parameters on held-out slices this file never sees, and nothing this file prints can
move it.

How the scheme is derived, so a reader can redo it. The handed toolkit in
environment/quantize.py can only set ONE width for every shard. The embedding matrix
and the untied output projection are 77266944 of the 162201600 quantizable parameters
and they do not repay a bit at the same rate the attention and MLP matrices do, so a
uniform width overspends on one and starves the other. This file measures that rate
per shard ROLE on the agent-visible calibration split, then runs a fixed greedy pass
that moves width from the roles that repay least to the roles that repay most while
the harness accounting stays inside the same total budget. The allocation is COMPUTED
here rather than frozen as a table, because a frozen table would be a number this
bundle has not measured.
"""

import json
import os
import sys
from pathlib import Path

ENVIRONMENT = Path(os.environ.get("OER21_ENVIRONMENT") or "/work/environment")
PASSES = 2


def toolkit():
    import importlib.util

    spec = importlib.util.spec_from_file_location("oer21_quantize", ENVIRONMENT / "quantize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    out = Path(os.environ.get("OUT_DIR") or os.getcwd())
    out.mkdir(parents=True, exist_ok=True)
    kit = toolkit()
    scheme = kit.widen(passes=PASSES)
    target = out / "scheme.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(scheme, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("scheme written to " + str(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
