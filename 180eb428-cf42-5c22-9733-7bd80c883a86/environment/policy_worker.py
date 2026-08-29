"""FROZEN. Out-of-process host for a submitted precision policy, bia slot S07.

This file is the whole reason a submission cannot forge its own measurement. The graded
attempt driver never imports the submission. It starts this worker as a separate operating
system process and speaks to it over one newline-delimited JSON channel:

    parent -> child   one request object per line on this process's stdin
    child  -> parent  exactly one response object per line on the channel fd

The submitted module is imported HERE, in this interpreter. Whatever it rebinds, patches,
deletes or monkeypatches it rebinds in THIS process, where no telemetry record, no overflow
ledger, no clock, no random seed and no reward computation lives. The only values that ever
cross back to the measuring process are:

    an int   from loss_scale, range-checked by the parent
    a dict   from plan, structurally validated by bia_numerics.validate_plan in the parent
    a str    from on_overflow, restricted by the parent to exactly "skip" or "continue"
    a str    for master_dtype, restricted by the parent to exactly "float32" or "bfloat16"

That is the complete outbound alphabet. It carries no path, no callable, no code, and no
number that reaches the score except through arithmetic the parent performs itself.

The channel is a dup of the original stdout taken before any submission byte is imported, and
stdout is then rebound to stderr, so a submission that prints cannot desynchronize the
protocol by writing to fd 1.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import traceback
from typing import Any, Dict

HERE = pathlib.Path(__file__).resolve().parent

PROTOCOL_VERSION = "bia.policy_worker/v1"


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("bia_submitted_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load submission at %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bia_submitted_policy"] = mod
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "build_policy", None)):
        raise RuntimeError("submission must define build_policy(config)")
    return mod


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: policy_worker.py <submission.py>", file=sys.stderr)
        return 2
    sub = pathlib.Path(sys.argv[1]).resolve()

    # Take the channel before the submission exists in this interpreter, then make fd 1 an
    # alias of stderr. Anything the submission prints is diagnostics, never protocol.
    chan = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    # The environment directory is importable so a policy may read the frozen format fixture,
    # exactly as it could when it ran in the driver. Anything it does to those modules here is
    # local to this process and reaches no measurement.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    def reply(obj: Dict[str, Any]) -> None:
        chan.write(json.dumps(obj, sort_keys=True) + "\n")
        chan.flush()

    policy = None
    try:
        mod = _load(sub)
    except BaseException as exc:  # an unimportable submission is a graded outcome, not a crash
        reply({"ok": False, "error": "import_failed: %s" % exc, "traceback": traceback.format_exc()})
        return 1

    reply({"ok": True, "protocol": PROTOCOL_VERSION, "pid": os.getpid()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "build":
                policy = mod.build_policy(dict(req["config"]))
                reply(
                    {
                        "ok": True,
                        "master_dtype": str(getattr(policy, "master_dtype", "float32")),
                    }
                )
            elif op == "loss_scale":
                reply({"ok": True, "value": int(policy.loss_scale(int(req["step"])))})
            elif op == "plan":
                value = policy.plan(int(req["step"]), int(req["n_chunks"]), list(req["stats"]))
                reply({"ok": True, "value": value})
            elif op == "on_overflow":
                reply(
                    {
                        "ok": True,
                        "value": str(policy.on_overflow(int(req["step"]), dict(req["info"]))),
                    }
                )
            elif op == "shutdown":
                reply({"ok": True})
                break
            else:
                reply({"ok": False, "error": "unknown op %r" % op})
        except BaseException as exc:
            reply({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
    chan.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
