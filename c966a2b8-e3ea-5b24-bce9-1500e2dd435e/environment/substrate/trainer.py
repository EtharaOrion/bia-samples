"""The one training loop every arm runs.

Three arms exist and they differ in exactly two bindings, the norm mode of the
frozen model and the update rule driving it:

  anchor_target    norm_mode "rmsnorm", frozen reference recipe
  anchor_baseline  norm_mode "none",    frozen reference recipe
  agent            norm_mode "none",    the submitted update rule

Everything else is identical: the same corpus bytes, the same frozen batch
order, the same seed, the same step count, the same one forward-backward pass
per step. A difference between the baseline arm and the agent arm is therefore
a difference in the update rule and nothing else.
"""

from __future__ import annotations

import math
import os
import time

import torch

from . import config as C
from .data import FrozenByteCorpus
from .model import GPT, norm_parameter_names
from .probe import NormProbe

SAMPLE_STEPS = 3  # steps at which the parameter delta is measured


def _pick_sample_steps(total: int):
    if total <= SAMPLE_STEPS:
        return list(range(total))
    return sorted({0, total // 2, total - 1})


@torch.no_grad()
def evaluate(model, corpus, cfg, device, probe=None):
    was_training = model.training
    model.eval()
    if probe is not None:
        probe.set_phase("eval")
    total, n = 0.0, 0
    for i in range(cfg["eval_batches"]):
        x, y = corpus.val_batch(i, device)
        loss = model(x, y)
        total += float(loss.item())
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


def run_arm(
    arm: str,
    norm_mode: str,
    rule_factory,
    cfg: dict,
    corpus_path: str,
    runlog,
    workdir: str,
    device: str,
):
    """Execute one arm end to end and return its measured record."""
    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()
    guard = cfg["arm_wallclock_guard_sec"]

    torch.manual_seed(cfg["seed"])
    corpus = FrozenByteCorpus(corpus_path, cfg)
    model = GPT(cfg, norm_mode).to(device)
    model.train()

    probe = NormProbe(runlog=None)
    probe.install_functional_guards()
    probe.hook_model(model)

    norm_params = norm_parameter_names(model)
    matrices, others = model.param_groups()
    n_params = sum(p.numel() for _, p in model.named_parameters())

    runlog.emit(
        "arm_start",
        phase="harness",
        arm=arm,
        norm_mode=norm_mode,
        n_params=n_params,
        norm_parameter_names=norm_params,
        data_order_digest=corpus.order_digest(),
        corpus_digest=corpus.corpus_digest(),
        config_digest=C.config_digest(cfg),
        device=device,
    )

    rule = rule_factory(matrices, others, dict(cfg))
    rule_module = getattr(type(rule), "__module__", "unknown")
    rule_file = "unknown"
    try:
        import sys

        rule_file = getattr(sys.modules.get(rule_module), "__file__", "unknown")
    except Exception:
        pass

    sample_steps = _pick_sample_steps(cfg["steps"])
    losses = []
    aborted = None

    for step in range(cfg["steps"]):
        if time.time() - t0 > guard:
            aborted = "arm_wallclock_exceeded"
            break

        probe.set_phase("forward")
        x, y = corpus.train_batch(step, device)
        loss = model(x, y)

        probe.set_phase("backward")
        for _, p in model.named_parameters():
            p.grad = None
        loss.backward()

        take_sample = step in sample_steps
        if take_sample:
            before = {n: p.detach().clone() for n, p in model.named_parameters()}

        probe.set_phase("update")
        rule.step(step, cfg["steps"])

        lv = float(loss.item())
        if not math.isfinite(lv):
            losses.append(float("nan"))
        else:
            losses.append(lv)

        if take_sample:
            delta_sq = 0.0
            for n, p in model.named_parameters():
                d = (p.detach() - before[n]).float()
                delta_sq += float((d * d).sum().item())
            runlog.emit(
                "update_applied",
                phase="update",
                arm=arm,
                step=step,
                delta_l2=math.sqrt(max(delta_sq, 0.0)),
                rule_module=rule_module,
                rule_file=rule_file,
                forward_backward_per_step=cfg["forward_backward_per_step"],
            )
            del before

        if step % max(1, cfg["steps"] // 10) == 0:
            runlog.emit("train_progress", phase="harness", arm=arm, step=step, train_loss=lv)

    streaming_val = evaluate(model, corpus, cfg, device, probe=probe)
    probe.set_phase("harness")

    ckpt = os.path.join(workdir, f"{arm}_final.pt")
    torch.save({"arm": arm, "norm_mode": norm_mode, "cfg": cfg, "state_dict": model.state_dict()}, ckpt)

    trace = probe.trace()
    probe.remove_hooks()
    probe.remove_functional_guards()

    finite_tail = [v for v in losses[-10:] if math.isfinite(v)]
    record = {
        "arm": arm,
        "norm_mode": norm_mode,
        "streaming_val_loss": streaming_val if math.isfinite(streaming_val) else None,
        "final_train_loss": finite_tail[-1] if finite_tail else None,
        "steps_run": len(losses),
        "steps_declared": cfg["steps"],
        "nonfinite_train_steps": sum(1 for v in losses if not math.isfinite(v)),
        "wallclock_sec": time.time() - t0,
        "aborted": aborted,
        "checkpoint": ckpt,
        "norm_parameter_names": norm_params,
        "n_params": n_params,
        "rule_module": rule_module,
        "rule_file": rule_file,
        "probe_trace": trace,
        "forbidden_norm_events": len(probe.forbidden_events()),
        "total_norm_events": len(trace["events"]),
        "config_digest": C.config_digest(cfg),
        "data_order_digest": corpus.order_digest(),
        "corpus_digest": corpus.corpus_digest(),
        "sample_steps": sample_steps,
    }
    runlog.emit("arm_complete", phase="harness", **{k: v for k, v in record.items() if k != "probe_trace"})
    runlog.emit(
        "arm_probe_trace",
        phase="harness",
        arm=arm,
        module_classes_by_phase=trace["module_classes_by_phase"],
        module_forward_counts_by_phase=trace["module_forward_counts_by_phase"],
        forbidden_events=[e for e in trace["events"] if e["phase"] in C.FORBIDDEN_PHASES],
        total_events=len(trace["events"]),
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return record


@torch.no_grad()
def independent_eval_from_checkpoint(ckpt_path: str, corpus_path: str, cfg: dict, device: str):
    """Second, independent derivation of the same quantity.

    This reconstructs the model from the saved checkpoint in a fresh object
    graph and re-measures validation loss with its own loop. It shares no state
    with the streaming measurement taken inside the training process, so the two
    numbers agreeing is evidence rather than a tautology.
    """
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(blob["cfg"], blob["norm_mode"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    corpus = FrozenByteCorpus(corpus_path, cfg)
    total, n = 0.0, 0
    for i in range(cfg["eval_batches"]):
        x, y = corpus.val_batch(i, device)
        total += float(model(x, y).item())
        n += 1
    out = total / max(n, 1)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out
