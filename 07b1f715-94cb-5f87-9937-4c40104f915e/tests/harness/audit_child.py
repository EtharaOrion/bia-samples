#!/usr/bin/env python3
"""Runs the behavioural audits of the submitted rule in an isolated process.

The previous revision executed the submitted module inside the verifier
interpreter to measure it, which put submitted code in the same address space
as the grader. Every audit that must execute the rule now runs here, in a
process the verifier spawns and whose only output is a JSON verdict on a pipe
the verifier owns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def load_rule(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bia_audited_rule", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = getattr(module, "build_update_rule", None)
    if not callable(build):
        raise RuntimeError("build_update_rule_absent")
    return build


def lr_homogeneity(build, tolerance=1e-8):
    """EFFECT: the delta a real step produces is exactly linear in the applied lr."""
    import torch

    def delta_at(lr):
        torch.manual_seed(11)
        weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float64))
        vector = torch.nn.Parameter(torch.ones(8, dtype=torch.float64))
        weight.grad = torch.randn(8, 4, dtype=torch.float64)
        vector.grad = torch.randn(8, dtype=torch.float64)
        groups = [
            {"name": "hidden_matrix", "params": [weight], "lr": 0.0},
            {"name": "vector", "params": [vector], "lr": 0.0},
        ]
        opt = build(groups)
        if not isinstance(opt, torch.optim.Optimizer):
            raise TypeError(type(opt).__name__)
        before = [weight.detach().clone(), vector.detach().clone()]
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        return [weight.detach() - before[0], vector.detach() - before[1]]

    base = 1e-3
    d1 = delta_at(base)
    d2 = delta_at(2.0 * base)
    for a, b in zip(d1, d2):
        if float(a.abs().max()) == 0.0:
            return False, "update_rule_produced_no_parameter_change"
        scale = a * 2.0
        residual = float((b - scale).abs().max()) / (float(scale.abs().max()) + 1e-12)
        if residual > tolerance:
            return False, "update_not_linear_in_lr_residual_%.3e" % residual
    return True, None


def schedule_ownership(build, horizon, floor_ratio=0.25):
    """EFFECT: the rule does not carry a step-index schedule of its own.

    The harness owns the schedule. A rule that has reconstructed the run
    horizon and shapes its own decay against it collapses its update near that
    horizon, so the probe drives the rule for three horizons' worth of steps
    under a constant gradient and a constant learning rate and compares the
    update magnitude on either side of the horizon it was denied.

    Bounded claim: this measures whether the rule's update magnitude collapses
    around the horizon under a constant-gradient, constant-lr drive. A rule
    whose horizon dependence does not show up under that drive is outside what
    the probe observes.
    """
    import torch

    torch.manual_seed(29)
    weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float64))
    vector = torch.nn.Parameter(torch.ones(8, dtype=torch.float64))
    grads = {id(weight): torch.randn(8, 4, dtype=torch.float64),
             id(vector): torch.randn(8, dtype=torch.float64)}
    groups = [
        {"name": "hidden_matrix", "params": [weight], "lr": 0.0},
        {"name": "vector", "params": [vector], "lr": 0.0},
    ]
    opt = build(groups)
    if not isinstance(opt, torch.optim.Optimizer):
        return False, "build_update_rule_returned_%s" % type(opt).__name__

    magnitudes = []
    steps = max(6, 3 * int(horizon))
    for _ in range(steps):
        weight.grad = grads[id(weight)].clone()
        vector.grad = grads[id(vector)].clone()
        before = [weight.detach().clone(), vector.detach().clone()]
        for group in opt.param_groups:
            group["lr"] = 1e-3
        opt.step()
        moved = (float((weight.detach() - before[0]).abs().sum())
                 + float((vector.detach() - before[1]).abs().sum()))
        magnitudes.append(moved)

    h = int(horizon)
    early = sorted(magnitudes[max(0, h // 2):h])
    late = sorted(magnitudes[h:min(len(magnitudes), h + max(1, h // 2))])
    if not early or not late:
        return False, "schedule_ownership_probe_window_empty"
    a = early[len(early) // 2]
    b = late[len(late) // 2]
    if a <= 0.0:
        return False, "update_rule_produced_no_parameter_change"
    if b < floor_ratio * a:
        return False, "update_collapses_at_denied_horizon_ratio_%.3e" % (b / a)
    return True, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-fd", type=int, required=True)
    args = ap.parse_args()
    channel = os.fdopen(args.channel_fd, "w")
    try:
        request = json.loads(sys.stdin.read())
        result = {}
        try:
            build = load_rule(request["submission"])
        except BaseException as exc:
            channel.write(json.dumps({
                "lr_homogeneity": [False, "update_rule_import_failed_%s"
                                   % type(exc).__name__],
                "schedule_ownership": [False, "update_rule_import_failed_%s"
                                       % type(exc).__name__]}))
            channel.flush()
            channel.close()
            return 0
        for name, fn in (("lr_homogeneity", lambda: lr_homogeneity(build)),
                         ("schedule_ownership",
                          lambda: schedule_ownership(build, request["horizon"]))):
            try:
                ok, reason = fn()
            except TypeError as exc:
                ok, reason = False, "build_update_rule_returned_%s" % exc
            except BaseException as exc:
                ok, reason = False, "%s_failed_%s" % (name, type(exc).__name__)
            result[name] = [bool(ok), reason]
        channel.write(json.dumps(result, sort_keys=True))
        channel.flush()
        channel.close()
        return 0
    except BaseException:
        try:
            channel.write(json.dumps({"error": traceback.format_exc()[-800:]}))
            channel.flush()
            channel.close()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
