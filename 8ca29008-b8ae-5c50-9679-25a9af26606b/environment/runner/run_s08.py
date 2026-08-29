"""S08 harness runner. The only writer of the telemetry record.

Phases, in this fixed order:

  1. control     a harness-owned naive warm start from the shipped checkpoint,
                 measured fresh in every graded attempt so the reward baseline is
                 a number this run observed rather than a number an author asserted.
  2. submission  one run per frozen seed using the submission's `recover` entry point.

The control phase is sealed before the first submission record is written, which is
the ordering the ORDERING checker traces to.

Logs written by anything other than this runner are not evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import s08_core as core  # noqa: E402

RUNNER_VERSION = "s08-runner-2"

# Seconds one recover call may take before the run is abandoned. A recovery is a bounded
# number of probe passes and some tensor arithmetic, so this is generous headroom; it exists
# so a submission that blocks forever becomes a failed run instead of a hung harness.
RECOVER_TIMEOUT = float(os.environ.get("S08_RECOVER_TIMEOUT", "900"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def loaded_digests(fixture: str, manifest_path: str) -> dict:
    """The path and digest of every file this run loads before it trains anything.

    The verifier rehashes these same paths at grading time. A file that changed
    between load and grading is caught by that comparison, and a file that was already
    wrong at load is caught against the digests the bundle froze.
    """
    out = {}
    for label, path in (
        (os.path.basename(__file__), __file__),
        (os.path.basename(core.__file__), core.__file__),
        ("recover_worker.py", os.path.join(HERE, "recover_worker.py")),
        ("checkpoint", fixture),
        ("manifest", manifest_path),
    ):
        full = os.path.abspath(path)
        out[label] = {"path": full, "sha256": sha256_file(full)}
    return out


class Telemetry:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fh = open(path, "w")
        self.index = 0
        self.prev = "genesis"

    def write(self, body: dict):
        body = dict(body)
        body["record_index"] = self.index
        payload = json.dumps(body, sort_keys=True)
        chain = hashlib.sha256((self.prev + payload).encode()).hexdigest()
        body["chain"] = chain
        self.fh.write(json.dumps(body, sort_keys=True) + "\n")
        self.fh.flush()
        self.prev = chain
        self.index += 1

    def close(self):
        self.fh.close()


class RecoverChannel:
    """Runner-side handle on one out-of-process `recover`.

    The runner never imports the submission. It starts recover_worker.py in its own
    interpreter and exchanges JSON control messages plus torch.save tensor files inside a
    scratch directory the runner owns and names. Everything that decides the score, the
    evaluation function, the telemetry writer, the hash chain and the probe counter, stays on
    this side of the boundary, so a submission has no reachable handle on any of them.

    The optimizer comes back as a description rather than as an object, and this class
    rebuilds it against the runner's own parameters. That is what keeps the free surface,
    optimizer class, hyperparameters, inherited state and learning-rate schedule, intact while
    removing the ability to execute code inside the measurement.
    """

    def __init__(self, submission_dir: str, scratch: str):
        path = os.path.join(submission_dir, "recover.py")
        if not os.path.exists(path):
            raise FileNotFoundError(f"submission recover.py absent at {path}")
        self.scratch = scratch
        os.makedirs(scratch, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, "-I", os.path.join(HERE, "recover_worker.py"), os.path.abspath(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=HERE,
            env=env,
            text=True,
            bufsize=1,
        )
        hello = self._recv()
        self.worker_pid = int(hello.get("pid", -1))
        self.protocol = str(hello.get("protocol", ""))

    def _recv(self):
        ready, _, _ = select.select([self.proc.stdout], [], [], RECOVER_TIMEOUT)
        if not ready:
            self.kill()
            raise RuntimeError(f"recover worker did not answer within {RECOVER_TIMEOUT:.0f}s")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"recover worker closed the channel (exit {self.proc.poll()})")
        msg = json.loads(line)
        if not isinstance(msg, dict):
            raise RuntimeError("recover worker wrote a non-protocol line")
        return msg

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj, sort_keys=True) + "\n")
        self.proc.stdin.flush()

    def recover(self, model, ckpt_state, cfg_public, probe, cfg, tag):
        params = list(model.parameters())
        model_file, ckpt_file = f"model_{tag}.pt", f"ckpt_{tag}.pt"
        torch.save({k: v.detach().to("cpu") for k, v in model.state_dict().items()},
                   os.path.join(self.scratch, model_file))
        torch.save(ckpt_state, os.path.join(self.scratch, ckpt_file))
        self._send({
            "op": "recover",
            "scratch": self.scratch,
            "profile": cfg.profile,
            "cfg_public": dict(cfg_public),
            "model_state_file": model_file,
            "ckpt_state_file": ckpt_file,
            "tag": tag,
        })
        n_probe_files = 0
        while True:
            msg = self._recv()
            if msg.get("op") == "probe":
                try:
                    records = probe(int(msg.get("n", 1)))
                except BaseException as exc:  # noqa: BLE001
                    self._send({"ok": False, "error": str(exc)})
                    continue
                name = f"probe_{tag}_{n_probe_files}.pt"
                n_probe_files += 1
                torch.save(
                    [{"loss": r["loss"], "grads": [g.detach().to("cpu") for g in r["grads"]]}
                     for r in records],
                    os.path.join(self.scratch, name),
                )
                self._send({"ok": True, "file": name})
                continue
            if not msg.get("ok"):
                raise RuntimeError(str(msg.get("error", "recover refused")))
            return self._rebuild(msg, params, model)

    def _rebuild(self, spec, params, model):
        cls_name = str(spec.get("optimizer_class", ""))
        cls = getattr(torch.optim, cls_name, None)
        if cls is None or not (isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer)):
            raise TypeError(f"recover named optimizer class {cls_name!r}, which torch.optim does not define")
        groups = []
        for g in spec["param_groups"]:
            entry = {k: v for k, v in g.items() if k != "param_indices"}
            if "betas" in entry and isinstance(entry["betas"], list):
                entry["betas"] = tuple(entry["betas"])
            entry["params"] = [params[i] for i in g["param_indices"]]
            groups.append(entry)
        if not groups or not any(g["params"] for g in groups):
            raise ValueError("recover returned an optimizer with no parameters")
        opt = cls(groups)

        tensors = torch.load(os.path.join(self.scratch, spec["state_tensor_file"]),
                             map_location="cpu", weights_only=False)
        for idx_s, scalars in spec["state_scalars"].items():
            p = params[int(idx_s)]
            st = opt.state[p]
            for k, v in scalars.items():
                if isinstance(v, dict) and "scalar_tensor" in v:
                    st[k] = torch.tensor(v["scalar_tensor"], dtype=getattr(torch, v["dtype"].split(".")[-1]))
                else:
                    st[k] = v
            for k, v in tensors.get(idx_s, {}).items():
                st[k] = v.detach().to(device=p.device, dtype=p.dtype)
        sched = spec.get("lr_schedule")
        lr_at = (lambda i, s=list(sched): float(s[min(max(int(i), 1), len(s)) - 1])) if sched else None
        return opt, lr_at

    def close(self):
        try:
            self._send({"op": "shutdown"})
            self._recv()
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            pass
        self.kill()

    def kill(self):
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def naive_control_recover(model, ckpt_state, cfg_public, probe):
    """The pattern-matched warm start: rebuild an AdamW and load the checkpoint
    optimizer state verbatim, hyperparameters included. This is the control, not
    a reference solution."""
    opt = torch.optim.AdamW(list(model.parameters()), lr=cfg_public.get("checkpoint_lr", 3.0e-3))
    opt.load_state_dict(copy.deepcopy(ckpt_state))
    return opt, None


class ProbeBudget:
    """Bounded forward-backward probes offered to `recover`. Every probe consumes
    one step of the same budget the reward counts, so diagnosis is not free."""

    def __init__(self, model, train, cfg, device, limit):
        self.model = model
        self.train = train
        self.cfg = cfg
        self.device = device
        self.limit = limit
        self.used = 0

    def __call__(self, n: int = 1):
        n = int(n)
        if n < 1:
            return []
        if self.used + n > self.limit:
            raise RuntimeError(
                f"probe budget exceeded: asked for {n} with {self.used} used and a limit of {self.limit}"
            )
        out = []
        for _ in range(n):
            step = -(self.used + 1)
            x, y = core.get_batch(self.train, self.cfg, step=step, seed=0, device=self.device)
            self.model.zero_grad(set_to_none=True)
            loss = core.lm_loss(self.model(x), y)
            loss.backward()
            grads = [
                (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                for p in self.model.parameters()
            ]
            out.append({"loss": float(loss.detach()), "grads": grads})
            self.used += 1
        self.model.zero_grad(set_to_none=True)
        return out


def normalize_recover_result(result):
    if isinstance(result, tuple):
        if len(result) == 1:
            return result[0], None
        if len(result) == 2:
            return result[0], result[1]
        raise TypeError("recover returned a tuple of unexpected length")
    return result, None


def run_phase(
    phase,
    seed,
    cfg,
    device,
    ckpt,
    train,
    val,
    tel,
    recover_fn,
    cfg_public,
    frozen,
    effects,
    digests,
    isolation,
):
    model = core.build_model(cfg, device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model"].items()})
    pre_digest = core.weight_digest(model)
    ckpt_state = copy.deepcopy(ckpt["optimizer"])
    probe = ProbeBudget(model, train, cfg, device, cfg.probe_batches_max)

    t0 = time.time()
    optimizer, lr_at = recover_fn(model, ckpt_state, dict(cfg_public), probe)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("recover must return a torch.optim.Optimizer")
    model.zero_grad(set_to_none=True)
    post_digest = core.weight_digest(model)
    probe_steps = probe.used

    digests[f"{phase}_seed{seed}"] = {
        "checkpoint_weight_digest": ckpt["weight_digest"],
        "weight_digest_before_recover": pre_digest,
        "weight_digest_after_recover": post_digest,
        "probe_steps": probe_steps,
        "recover_isolation": dict(isolation),
    }
    effects[f"{phase}_seed{seed}"] = {
        "checkpoint_optimizer_fingerprint": core.fingerprint_from_state(ckpt["optimizer"]),
        "recovered_optimizer_fingerprint": core.optimizer_fingerprint(optimizer),
        "recover_seconds": time.time() - t0,
    }

    base = dict(frozen)
    base.update({"phase": phase, "seed": int(seed), "probe_steps": probe_steps})

    tel.write(dict(base, kind="resume", step=probe_steps, weight_digest_after_recover=post_digest))

    loss0 = core.evaluate(model, val, cfg, device)
    tel.write(dict(base, kind="eval", step=probe_steps, val_loss=round(loss0, 6)))

    model.train()
    for i in range(1, cfg.max_steps + 1):
        if lr_at is not None:
            lr = float(lr_at(i))
            for g in optimizer.param_groups:
                g["lr"] = lr
        x, y = core.get_batch(train, cfg, step=i, seed=seed, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = core.lm_loss(model(x), y)
        loss.backward()
        optimizer.step()
        if i % cfg.eval_every == 0 or i == cfg.max_steps:
            vl = core.evaluate(model, val, cfg, device)
            if os.environ.get("S08_VERBOSE"):
                print(f"  {phase} seed {seed} step {probe_steps + i:6d}  val {vl:.5f}", flush=True)
            if not np.isfinite(vl):
                vl = float("inf")
            tel.write(
                dict(
                    base,
                    kind="eval",
                    step=probe_steps + i,
                    train_loss=round(float(loss.detach()), 6) if np.isfinite(float(loss.detach())) else None,
                    val_loss=(round(vl, 6) if np.isfinite(vl) else 1.0e9),
                )
            )
    tel.write(dict(base, kind="phase_end", step=probe_steps + cfg.max_steps))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("S08_PROFILE", "scaled"))
    ap.add_argument("--submission", default=os.environ.get("S08_SUBMISSION", "/workspace/submission"))
    ap.add_argument("--fixture", default=os.environ.get("S08_FIXTURE", ""))
    ap.add_argument("--data-dir", default=os.environ.get("S08_DATA_DIR", ""))
    ap.add_argument("--out", default=os.environ.get("S08_TELEMETRY_DIR", "/telemetry"))
    ap.add_argument("--device", default=os.environ.get("S08_DEVICE", ""))
    args = ap.parse_args(argv)

    cfg = core.load_cfg(args.profile)
    root = os.path.dirname(HERE)
    fixture = args.fixture or os.path.join(root, "fixtures", f"ckpt_s08_{cfg.profile}.pt")
    manifest_path = os.path.join(os.path.dirname(fixture), f"manifest_{cfg.profile}.json")
    data_dir = args.data_dir or os.path.join(os.environ.get("S08_STATE_DIR", "/opt/bia/s08"), "data", cfg.profile)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("S08_CPU_THREADS", "4")))

    torch.manual_seed(cfg.init_seed)
    train, val, meta = core.load_corpus(cfg, data_dir)
    with open(manifest_path) as f:
        manifest = json.load(f)

    ckpt = torch.load(fixture, map_location="cpu", weights_only=False)
    if ckpt.get("core_version") != core.CORE_VERSION:
        raise RuntimeError("fixture core_version does not match the installed core")

    frozen = core.frozen_facts(cfg)
    frozen["fwd_bwd_per_step"] = 1
    frozen["core_version"] = core.CORE_VERSION
    frozen["runner_version"] = RUNNER_VERSION
    frozen["profile"] = cfg.profile
    frozen["target_loss"] = manifest["target_loss"]
    frozen["max_steps"] = cfg.max_steps

    cfg_public = core.public_cfg(
        cfg,
        checkpoint_step=manifest["checkpoint_step"],
        target_loss=manifest["target_loss"],
        bayes_loss=manifest["source_entropy_nats"],
    )
    cfg_public["checkpoint_lr"] = manifest["checkpoint_reported_lr"]

    os.makedirs(args.out, exist_ok=True)
    tel = Telemetry(os.path.join(args.out, "run_record.jsonl"))
    effects, digests = {}, {}
    scratch = tempfile.mkdtemp(prefix="bia_s08_recover_")

    try:
        tel.write(
            dict(
                frozen,
                phase="harness",
                kind="integrity",
                seed=-1,
                step=-1,
                probe_steps=-1,
                digests=loaded_digests(fixture, manifest_path),
            )
        )
        run_phase(
            "control", 0, cfg, device, ckpt, train, val, tel,
            lambda m, c, cp, pr: normalize_recover_result(naive_control_recover(m, c, cp, pr)),
            cfg_public, frozen, effects, digests,
            {"mode": "in_harness", "worker": None, "runner_pid": os.getpid(), "worker_pid": None},
        )
        tel.write(dict(frozen, phase="control", kind="phase_seal", seed=0, step=-1))

        channel = RecoverChannel(args.submission, os.path.join(scratch, "exchange"))
        submission_isolation = {
            "mode": "subprocess",
            "worker": "recover_worker.py",
            "runner_pid": os.getpid(),
            "worker_pid": channel.worker_pid,
            "protocol": channel.protocol,
        }
        try:
            for seed in cfg.seeds:
                run_phase(
                    "submission", seed, cfg, device, ckpt, train, val, tel,
                    (lambda sd: lambda m, c, cp, pr: channel.recover(m, c, cp, pr, cfg, f"seed{sd}"))(seed),
                    cfg_public, frozen, effects, digests, submission_isolation,
                )
        finally:
            channel.close()
        tel.write(dict(frozen, phase="submission", kind="phase_seal", seed=-1, step=-1))
    finally:
        tel.close()
        with open(os.path.join(args.out, "recover_effect.json"), "w") as f:
            json.dump(effects, f, indent=1, sort_keys=True)
        with open(os.path.join(args.out, "resume_digests.json"), "w") as f:
            json.dump(digests, f, indent=1, sort_keys=True)
        with open(os.path.join(args.out, "corpus_meta.json"), "w") as f:
            json.dump({"data_dir": data_dir, **meta}, f, indent=1, sort_keys=True)
        with open(os.path.join(args.out, "run_origin.json"), "w") as f:
            json.dump(
                {
                    "runner_version": RUNNER_VERSION,
                    "run_nonce": os.environ.get("S08_RUN_NONCE", ""),
                    "runner_pid": os.getpid(),
                    "submission_dir": os.path.abspath(args.submission),
                    "submission_sha256": (
                        sha256_file(os.path.join(args.submission, "recover.py"))
                        if os.path.exists(os.path.join(args.submission, "recover.py"))
                        else ""
                    ),
                    "telemetry_dir": os.path.abspath(args.out),
                    "record_count": tel.index,
                    "chain_head": tel.prev,
                },
                f,
                indent=1,
                sort_keys=True,
            )
        shutil.rmtree(scratch, ignore_errors=True)

    print(json.dumps({"runner": RUNNER_VERSION, "profile": cfg.profile, "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
