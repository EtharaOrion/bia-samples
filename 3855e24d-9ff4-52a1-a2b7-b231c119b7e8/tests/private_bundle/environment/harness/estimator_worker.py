"""Isolated worker that runs the submitted estimator for bia slot S06.

This is the only process the submitted estimator ever executes in, and it is not the
process that measures anything. It holds no model, no data, no optimizer, no
evaluation, no noise process, no run record, no target loss and no reward. It receives
corrupted gradient observations through a shared buffer the parent allocated, calls
estimate once per message, writes the result into a second shared buffer, and answers
with one word.

Everything a submission could usefully tamper with lives in the parent. Rebinding a
function here changes only what this worker does with tensors the parent already owns,
and the parent reads the answer out of its own buffer rather than believing anything
this process says about itself. The single word this worker sends back is a liveness
signal, not a measurement: the parent treats any answer other than a clean ok as a run
failure, and derives every graded quantity from state it computed itself.

The estimator contract is unchanged and is stated in environment/starter/estimator.py.
"""

from __future__ import annotations

import importlib.util

import torch


def load_estimator(path, meta):
    spec = importlib.util.spec_from_file_location("bia_submission_estimator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_estimator"):
        raise RuntimeError("submission_missing_build_estimator")
    est = module.build_estimator(meta)
    if not hasattr(est, "estimate"):
        raise RuntimeError("submission_estimator_missing_estimate")
    return est


def _publish(out_bufs, produced):
    if not isinstance(produced, dict):
        return "estimator_returned_non_dict"
    for name, buf in out_bufs.items():
        value = produced.get(name)
        if value is None:
            return "estimator_dropped_tensor_%s" % name
        tensor = torch.as_tensor(value)
        if tuple(tensor.shape) != tuple(buf.shape):
            return "estimator_shape_mismatch_%s" % name
        buf.copy_(tensor.to(device=buf.device, dtype=torch.float32))
    return None


def serve(submission_path, meta, in_bufs, out_bufs, conn):
    try:
        estimator = load_estimator(submission_path, meta)
    except Exception as exc:
        conn.send(("fault", "submission_load_failed_%r" % (exc,)))
        return
    on_cuda = bool(out_bufs) and next(iter(out_bufs.values())).is_cuda
    conn.send(("ready", ""))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if not isinstance(message, tuple) or message[0] == "stop":
            return
        step = int(message[1])
        try:
            observed = {name: buf.clone() for name, buf in in_bufs.items()}
            produced = estimator.estimate(step, observed)
            fault = _publish(out_bufs, produced)
        except Exception as exc:
            conn.send(("fault", "estimator_raised_at_step_%d_%r" % (step, exc)))
            return
        if fault is not None:
            conn.send(("fault", fault))
            return
        if on_cuda:
            torch.cuda.synchronize()
        conn.send(("ok", step))
