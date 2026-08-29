"""FROZEN. Model, data, optimizer and training loop for bia slot S07.

Harness owned and read only to the solving agent. Everything a submission is allowed to move
lives behind the Policy protocol in submission/precision_policy.py; nothing else is free.

Frozen here: the architecture, the corpus and its ordering, the optimizer and its schedule,
the number of steps, the number of micro batches per step, the seed, and the arithmetic of
the reduction. Free to the submission: the loss scale exponent, the per chunk exponents, the
reduction order and shape, the exponent alignment rule, whether the accumulator is
compensated, where master weights live, and what to do when an overflow is reported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import bia_numerics as N

HERE = pathlib.Path(__file__).resolve().parent
OP_PATH = HERE / "operating_point.json"

with open(OP_PATH, "r", encoding="utf-8") as _f:
    OP: Dict[str, Any] = json.load(_f)


# ---------------------------------------------------------------------------
# Configuration, resolved from the frozen fixture and the smoke switch
# ---------------------------------------------------------------------------


def smoke_enabled() -> bool:
    return os.environ.get("BIA_SMOKE", "0") == "1"


def resolve_config() -> Dict[str, Any]:
    m, s, d = OP["model"], OP["schedule"], OP["data"]
    cfg = {
        "mode": "full",
        "n_layer": m["n_layer"],
        "d_model": m["d_model"],
        "n_head": m["n_head"],
        "vocab_size": m["vocab_size"],
        "seq_len": m["sequence_length"],
        "mlp_ratio": m["mlp_ratio"],
        "steps": s["steps"],
        "micro_batches": s["micro_batches_per_step"],
        "micro_bsz": s["micro_batch_sequences"],
        "seed": s["graded_seed"],
        "corpus_bytes": None,
        "val_batches": d["validation_batches"],
        "val_tail": d["validation_tail_fraction"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if smoke_enabled():
        sm = OP["smoke"]
        cfg.update(
            {
                "mode": "smoke",
                "n_layer": sm["n_layer"],
                "d_model": sm["d_model"],
                "n_head": sm["n_head"],
                "seq_len": sm["sequence_length"],
                "steps": sm["steps"],
                "micro_batches": sm["micro_batches_per_step"],
                "micro_bsz": sm["micro_batch_sequences"],
                "corpus_bytes": sm["corpus_bytes_used"],
                "val_batches": sm["validation_batches"],
                "device": "cpu",  # smoke never touches CUDA, by construction
            }
        )
    return cfg


def config_digest(cfg: Dict[str, Any]) -> str:
    payload = {
        "cfg": {k: v for k, v in sorted(cfg.items()) if k != "device"},
        "format": hashlib.sha256((HERE / "format.json").read_bytes()).hexdigest(),
        "numerics": hashlib.sha256((HERE / "bia_numerics.py").read_bytes()).hexdigest(),
        "harness": hashlib.sha256((HERE / "harness.py").read_bytes()).hexdigest(),
        "operating_point": hashlib.sha256(OP_PATH.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data: real public domain text, byte level, deterministic order
# ---------------------------------------------------------------------------


class ByteCorpus:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        path = HERE / OP["data"]["corpus_file"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != OP["data"]["corpus_sha256"]:
            raise RuntimeError(
                "corpus digest mismatch: fixture pins %s, file is %s"
                % (OP["data"]["corpus_sha256"], digest)
            )
        self.full_sha256 = digest
        if cfg["corpus_bytes"]:
            raw = raw[: cfg["corpus_bytes"]]
        arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.long)
        split = int(len(arr) * (1.0 - cfg["val_tail"]))
        self.train = arr[:split]
        self.val = arr[split:]
        self.seq_len = cfg["seq_len"]

    def batch(self, gen: torch.Generator, bsz: int, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.train if split == "train" else self.val
        hi = len(src) - self.seq_len - 1
        ix = torch.randint(0, hi, (bsz,), generator=gen)
        x = torch.stack([src[i : i + self.seq_len] for i in ix])
        y = torch.stack([src[i + 1 : i + 1 + self.seq_len] for i in ix])
        return x, y


# ---------------------------------------------------------------------------
# Model: small pre-norm decoder-only transformer
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, d: int, h: int, ratio: int) -> None:
        super().__init__()
        self.n_head = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, ratio * d, bias=False)
        self.out = nn.Linear(ratio * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        shape = (B, T, self.n_head, C // self.n_head)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        h = self.ln2(x)
        x = x + self.out(F.gelu(self.fc(h)))
        return x


class ByteGPT(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        d, v, T = cfg["d_model"], cfg["vocab_size"], cfg["seq_len"]
        self.tok = nn.Embedding(v, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, cfg["n_head"], cfg["mlp_ratio"]) for _ in range(cfg["n_layer"])])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, v, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)
        # Residual output projections are downscaled by 1/sqrt(2*n_layer) so the residual
        # stream variance does not grow with depth. Without this the tied head produces
        # logits of standard deviation sqrt(d_model) and the run starts far above ln(vocab).
        scale = (2.0 * cfg["n_layer"]) ** -0.5
        with torch.no_grad():
            for b in self.blocks:
                b.proj.weight.mul_(scale)
                b.out.weight.mul_(scale)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.lnf(h))
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1))


# ---------------------------------------------------------------------------
# The graded training loop
# ---------------------------------------------------------------------------


def lr_at(step: int, total: int) -> float:
    warmup = max(1, int(0.1 * total))
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def validate(model: nn.Module, corpus: ByteCorpus, cfg: Dict[str, Any], bsz: int) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    tot = 0.0
    for _ in range(cfg["val_batches"]):
        x, y = corpus.batch(gen, bsz, "val")
        tot += float(model(x.to(cfg["device"]), y.to(cfg["device"])).item())
    model.train()
    return tot / cfg["val_batches"]


@torch.no_grad()
def _param_checksum(params: List[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def train_one(
    cfg: Dict[str, Any],
    corpus: ByteCorpus,
    phase: str,
    policy: Optional[Any],
    telemetry: List[Dict[str, Any]],
    replay_steps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run one complete training pass. phase is fp32_control, standard_control or agent_run.

    policy is None only for fp32_control, whose reduction runs in float32 and never touches
    the narrow format. Every other phase reduces through bia_numerics.reduce_in_format.

    policy is never a submitted object. It is the parent-side PolicyChannel handle on a
    separate process, so nothing this loop records is reachable from submitted bytes.

    replay_steps is the set of reduction steps this pass records for independent replay. The
    caller draws it from its own system entropy, so it is neither a fixed schedule position
    nor anything the hosted policy can predict.
    """
    torch.manual_seed(cfg["seed"])
    model = ByteGPT(cfg).to(cfg["device"])
    params = [p for p in model.parameters() if p.requires_grad]

    master_dtype = "float32"
    if policy is not None:
        master_dtype = str(getattr(policy, "master_dtype", "float32"))
    if master_dtype not in ("float32", "bfloat16"):
        raise ValueError("master_dtype must be float32 or bfloat16, got %r" % master_dtype)
    master = [p.detach().clone().to(getattr(torch, master_dtype)) for p in params]

    base_lr = float(OP["schedule"]["learning_rate"])
    opt = torch.optim.AdamW(master, lr=base_lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)

    N.LEDGER.reset()
    telemetry.append({"seq": len(telemetry), "record": "phase_begin", "phase": phase, "t": time.time()})

    gen = torch.Generator().manual_seed(cfg["seed"] + 7)
    n_chunks = cfg["micro_batches"]
    want_replay = sorted({int(s) for s in (replay_steps or [min(3, cfg["steps"] - 1)])})
    replay_samples: List[Dict[str, Any]] = []
    skipped = 0
    first_ck = _param_checksum(master)

    for step in range(cfg["steps"]):
        chunks: List[torch.Tensor] = []
        scale_exp = 0
        if policy is not None:
            scale_exp = int(policy.loss_scale(step))
            if not (-32 <= scale_exp <= 32):
                raise ValueError("loss_scale exponent out of the bound range [-32, 32]")
        loss_val = 0.0

        for _ in range(n_chunks):
            x, y = corpus.batch(gen, cfg["micro_bsz"], "train")
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            model.zero_grad(set_to_none=True)
            loss = model(x, y)
            loss_val += float(loss.item()) / n_chunks
            (loss * (2.0 ** scale_exp) / n_chunks).backward()
            chunks.append(torch.cat([p.grad.detach().reshape(-1).float() for p in params]))

        if policy is None:
            flat, audit = N.reduce_float32(chunks)
        else:
            # Per chunk statistics are computed in float32, outside the frozen format, and
            # handed to the policy. This is what a real block floating point encoder reads to
            # pick a shared exponent, so withholding it would make a safe plan unconstructible
            # rather than hard. It is a summary of the chunk, never the chunk itself.
            stats = [
                {
                    "absmax": float(c.abs().max().item()),
                    "absmean": float(c.abs().mean().item()),
                    "index": i,
                }
                for i, c in enumerate(chunks)
            ]
            plan = policy.plan(step, n_chunks, stats)
            flat, audit = N.reduce_in_format(chunks, plan, step=step, site="grad_reduce")
            if step in want_replay:
                k = min(64, chunks[0].numel())
                replay_samples.append(
                    {
                        "step": step,
                        "plan": N.validate_plan(plan, n_chunks),
                        "chunks": [c[:k].tolist() for c in chunks],
                        "expected": flat[:k].tolist(),
                        "elements": k,
                    }
                )

        flat = flat / (2.0 ** scale_exp)
        audit["step"] = step
        audit["phase"] = phase
        audit["record"] = "reduction"
        audit["seq"] = len(telemetry)
        telemetry.append(audit)

        nonfinite = bool(not torch.isfinite(flat).all().item())
        if nonfinite and policy is not None:
            action = str(policy.on_overflow(step, {"nonfinite": True, "scale_exp": scale_exp}))
            if action not in ("skip", "continue"):
                raise ValueError("on_overflow must return skip or continue, got %r" % action)
            if action == "skip":
                skipped += 1
                continue
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        elif nonfinite:
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

        off = 0
        for p, mp in zip(params, master):
            n = p.numel()
            mp.grad = flat[off : off + n].view_as(p).to(mp.dtype)
            off += n
        for g in opt.param_groups:
            g["lr"] = base_lr * lr_at(step, cfg["steps"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, mp in zip(params, master):
                p.copy_(mp.to(p.dtype))

        if step % max(1, cfg["steps"] // 8) == 0:
            telemetry.append(
                {
                    "seq": len(telemetry),
                    "record": "progress",
                    "phase": phase,
                    "step": step,
                    "train_loss": loss_val,
                    "scale_exp": scale_exp,
                }
            )

    final_val = validate(model, corpus, cfg, max(2, cfg["micro_bsz"] * 2))
    led = N.LEDGER.snapshot()
    result = {
        "seq": len(telemetry),
        "record": "phase_end",
        "phase": phase,
        "final_val_loss": final_val,
        "steps_run": cfg["steps"],
        "steps_skipped": skipped,
        "master_dtype": master_dtype,
        "master_param_dtype_observed": str(master[0].dtype),
        "master_checksum_first": first_ck,
        "master_checksum_final": _param_checksum(master),
        "overflow_event_count": led["overflow_event_count"],
        "overflow_elements": led["overflow_elements"],
        "quantize_calls": led["quantize_calls"],
        "elements_quantized": led["elements_quantized"],
        "overflow_events": led["events"][:64],
        "format_id": led["format_id"] if policy is not None else "float32",
        "replay_steps_requested": want_replay if policy is not None else [],
        "replay_steps_recorded": [s["step"] for s in replay_samples],
        "t": time.time(),
    }
    telemetry.append(result)
    for sample in replay_samples:
        telemetry.append({"seq": len(telemetry), "record": "reduction_replay", "phase": phase, **sample})
    return result
