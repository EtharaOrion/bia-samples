"""Out-of-process host for the submitted update rule. Never the author of a graded number.

The verifier does not import the submission. It starts this worker as a separate operating
system process, and the worker loads `update_rule.py`, builds the rule, and runs the agent arm
through the frozen substrate. What comes back is a checkpoint on disk plus the run-record
events the substrate emitted. The verifier then re-measures the metric from that checkpoint
with its own loop, in its own interpreter, and runs the checker registry there.

That boundary is the whole fix. Before it, the submission was exec_module'd into the verifier,
so everything the verifier touched afterwards, the anchor cache it had not read yet and the
checker registry it had not called yet, was reachable by the graded party. Purity of a checker
is not integrity of the process that calls it.

Nothing on this side is trusted as a measurement. The verifier treats the streaming loss and
the update records as claims and re-derives the graded metric and the parameter displacement
itself.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback


def load_submission(path):
    spec = importlib.util.spec_from_file_location("bia_submission_update_rule", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bia_submission_update_rule"] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "make_update_rule", None)
    if fn is None:
        raise AttributeError("submission_missing_make_update_rule")
    return fn


def main():
    req = json.loads(sys.stdin.read())
    out_path = req["result_path"]
    result = {"ok": False}
    try:
        sys.path.insert(0, req["substrate_parent"])
        os.environ["BIA_SMOKE"] = req["smoke"]
        from substrate import config as C
        from substrate.runlog import RunLog
        from substrate.trainer import run_arm

        factory = load_submission(req["submission_path"])
        cfg = C.get_config(req["smoke"] == "1")
        log = RunLog(req["run_record"])
        arm = run_arm("agent", "none", factory, cfg, req["corpus"], log, req["workdir"], req["device"])
        result = {"ok": True, "arm": arm, "worker_pid": os.getpid()}
    except BaseException as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
            "traceback": traceback.format_exc(),
            "worker_pid": os.getpid(),
        }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
