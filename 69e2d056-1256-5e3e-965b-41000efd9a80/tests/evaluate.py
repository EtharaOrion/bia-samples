"""Harness-owned quantization, forward passes, evaluation and telemetry for slot OER-21.

This module is the only thing that computes a perplexity in this bundle. It runs inside
the verifier's process, over the VERIFIER'S OWN copy of the frozen nanoGPT checkpoint and
the verifier's own held-out FineWeb slices, and it never reads a number the submission
produced. The submission supplies a QUANTIZATION SCHEME and nothing else; the scheme is a
specification of how bits are allocated over the checkpoint's parameter shards, not a
report of what that allocation achieved.

What the graded number is, exactly. For every held-out slice the verifier runs two real
forward passes of the canonical twelve-layer decoder: one with the full-precision
parameters and one with the parameters as this module quantized them. The reference
reading is the exponentiated mean entropy of the full-precision next-token distribution
over that slice, and the quantized reading is the exponentiated mean cross entropy of the
full-precision distribution under the quantized one. By Gibbs' inequality the second is at
least the first, with equality only when the two distributions coincide, so a scheme can
never buy a better-than-reference reading out of quantization noise. The true-token
perplexity of both parameter states is carried alongside as telemetry so a reader can see
that the checkpoint is a language model on real text rather than a stand-in.

Four separations are load-bearing and are the reason this file exists apart from
tests/grade.py and tests/checkers.py:

1.  The parameters are the verifier's own. The path is resolved from the verifier-only
    mount, never from anything the agent container carries and never from a call the
    submission can reach. They are shape-checked against environment/nanogpt_substrate.json
    before a single bit is allocated, so a snapshot that is not the frozen architecture is
    refused rather than graded.
2.  The evaluation split is the verifier's own. The held-out FineWeb slices are named in
    tests/anchors.json and read from the verifier-only mount. They are absent from the
    agent container by construction, so the split cannot be resolved by the solver.
3.  The graded reading is recomputed here from harness-owned quantized parameters. A
    filter the submission asks for on the graded path is RECORDED as an attempt and never
    applied, so the graded number is always the raw recomputation.
4.  The bit accounting is performed here, over the shards as actually quantized, by
    summing what each group costs. It is never the submission's own claim about cost.

No clock, no network, no random source, no environment secret. The only nondeterminism is
floating-point reduction order on the accelerator, which is why every reading is rounded
before it is compared.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent

SCHEMA = "forge.oer21.telemetry/v1"

# Quantization vocabulary the FULL scheme space admits. The handed default toolkit in
# environment/quantize.py exposes a strict subset of this on purpose; widening past that
# subset is the whole task.
ALLOWED_BITS = (2, 3, 4, 5, 6, 8)
ALLOWED_CODEBOOKS = ("symmetric", "affine", "kmeans")
ALLOWED_CLIPS = (1.0, 0.999, 0.99, 0.95, 0.9)

# Parameter count per scale group, in units of `scale_bits`. A kmeans codebook pays for
# its whole centroid table, which is what makes a non-uniform codebook affordable at low
# width and unaffordable at high width.
CODEBOOK_PARAMS = {"symmetric": 1, "affine": 2}

# The option set environment/quantize.py hands the agent: one uniform width for every
# shard, one scale group per shard, symmetric or affine, one clip knob.
DEFAULT_SPACE_CODEBOOKS = ("symmetric", "affine")

LLOYD_ITERATIONS = 8
ROUND_DIGITS = 9
MICRO_BATCH_SEQUENCES = 4

# The verifier-only mount. Nothing under it is reachable from the agent container, and
# nothing the submission runs can name it: tests/runner.py launches the submission under a
# four-name environment allowlist that does not include either variable.
VERIFIER_ROOT = Path(os.environ.get("FORGE_VERIFIER_ROOT") or "/verifier/frozen")
CHECKPOINT_NAME = "checkpoint.pt"
HOLDOUT_DIRECTORY = "holdout"

_MODEL_MODULE = None


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SchemeError(ValueError):
    """A scheme that is not a scheme. Refused before any bit is accounted."""


class SubstrateError(RuntimeError):
    """The verifier's own parameters or split are missing or are not what was declared."""


# ---------------------------------------------------------------------------
# The frozen architecture, the verifier's parameters, and the held-out split
# ---------------------------------------------------------------------------


def model_module():
    """Import the ONE architecture definition this bundle carries.

    The definition lives at environment/model/nanogpt.py because the agent has to be able
    to read the architecture it is allocating bits over. The verifier imports the same
    module rather than carrying a second copy, so no second definition exists to drift.
    """
    global _MODEL_MODULE
    if _MODEL_MODULE is None:
        import importlib.util

        root = Path(os.environ.get("FORGE_ENVIRONMENT_ROOT") or (BUNDLE / "environment"))
        spec = importlib.util.spec_from_file_location("oer21_nanogpt", root / "model" / "nanogpt.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_MODULE = module
    return _MODEL_MODULE


def substrate(paths: dict) -> dict:
    return load_json(paths["substrate"])


def checkpoint_path(root: Path = VERIFIER_ROOT) -> Path:
    return Path(root) / CHECKPOINT_NAME


def holdout_path(root: Path = VERIFIER_ROOT) -> Path:
    return Path(root) / HOLDOUT_DIRECTORY


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_token_shard(path: Path) -> torch.Tensor:
    """Read a modded-nanogpt .bin token shard. Header is 256 int32, then uint16 ids."""
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    if int(header[0]) != 20240520:
        raise SubstrateError("magic number mismatch in " + str(path))
    if int(header[1]) != 1:
        raise SubstrateError("unsupported token shard version in " + str(path))
    count = int(header[2])
    with Path(path).open("rb", buffering=0) as handle:
        tokens = torch.empty(count, dtype=torch.uint16)
        handle.seek(256 * 4)
        read = handle.readinto(tokens.numpy())
    if read != 2 * count:
        raise SubstrateError("short read on " + str(path))
    return tokens


def load_holdout(anchors: dict, root: Path = VERIFIER_ROOT) -> list:
    """The verifier's own held-out FineWeb slices, pinned by shard and token offset.

    Absent from the agent container by construction. If the mount is not present the
    verifier refuses rather than falling back to anything the solver could have written.
    """
    directory = holdout_path(root)
    if not directory.is_dir():
        raise SubstrateError("holdout-split-absent")
    sequence = int(anchors["holdout"]["sequence_length"])
    slices = []
    cache = {}
    for row in anchors["holdout"]["slices"]:
        shard = row["shard"]
        if shard not in cache:
            target = directory / shard
            if not target.is_file():
                raise SubstrateError("holdout-split-absent")
            cache[shard] = load_token_shard(target)
        tokens = cache[shard]
        start = int(row["token_offset"])
        count = int(row["token_count"])
        if start + count + 1 > len(tokens):
            raise SubstrateError("holdout slice " + str(row["id"]) + " runs past the end of " + shard)
        window = tokens[start:start + count + 1].to(dtype=torch.int64)
        usable = (count // sequence) * sequence
        inputs = window[:usable].reshape(-1, sequence)
        targets = window[1:usable + 1].reshape(-1, sequence)
        slices.append(
            {
                "id": int(row["id"]),
                "shard": shard,
                "token_offset": start,
                "token_count": int(usable),
                "inputs": inputs,
                "targets": targets,
            }
        )
    return slices


def load_reference_state(paths: dict, root: Path = VERIFIER_ROOT) -> tuple:
    """Load the verifier's own checkpoint and shape-bind it to the frozen architecture."""
    nanogpt = model_module()
    target = checkpoint_path(root)
    if not target.is_file():
        raise SubstrateError("checkpoint-absent")
    declaration = substrate(paths)
    state = nanogpt.load_checkpoint(target, declaration)
    conformance = nanogpt.assert_checkpoint_matches(state, declaration)
    return state, conformance


def instantiate(state: dict, paths: dict) -> torch.nn.Module:
    nanogpt = model_module()
    model = nanogpt.build_model(substrate(paths))
    model.load_state_dict({name: tensor for name, tensor in state.items()}, strict=True)
    return model.to(device()).to(torch.bfloat16).eval()


def parameter_view(model: torch.nn.Module, table: list) -> dict:
    """A live handle on every shard tensor of an instantiated model, by declared name."""
    named = dict(model.named_parameters())
    return {row["name"]: named[row["name"]] for row in table}


# ---------------------------------------------------------------------------
# Quantizers. One group at a time, on the accelerator, deterministic.
# ---------------------------------------------------------------------------


def _params(codebook: str, bits: int) -> int:
    if codebook == "kmeans":
        return 1 << bits
    return CODEBOOK_PARAMS[codebook]


def _clip_magnitude(values: torch.Tensor, clip: float) -> torch.Tensor:
    """The magnitude at the clip percentile of the group, used as the symmetric bound."""
    magnitudes = values.abs()
    count = magnitudes.numel()
    index = max(0, min(count - 1, int(math.ceil(clip * count)) - 1))
    return torch.kthvalue(magnitudes.flatten().float(), index + 1).values


def _quantize_symmetric(values: torch.Tensor, bits: int, clip: float) -> torch.Tensor:
    amax = _clip_magnitude(values, clip)
    span = (1 << (bits - 1)) - 1
    if span <= 0 or float(amax) <= 0.0:
        return torch.zeros_like(values)
    step = amax / span
    levels = torch.clamp(torch.floor(values.float() / step + 0.5), -span, span)
    return (levels * step).to(values.dtype)


def _quantize_affine(values: torch.Tensor, bits: int, clip: float) -> torch.Tensor:
    flat = values.flatten().float()
    count = flat.numel()
    keep = max(1, int(math.ceil(clip * count)))
    drop = count - keep
    ordered, _ = torch.sort(flat)
    low = ordered[drop // 2]
    high = ordered[count - 1 - (drop - drop // 2)]
    steps = (1 << bits) - 1
    if float(high) <= float(low) or steps <= 0:
        return torch.full_like(values, float(low))
    step = (high - low) / steps
    clamped = torch.clamp(values.float(), float(low), float(high))
    levels = torch.clamp(torch.floor((clamped - low) / step + 0.5), 0, steps)
    return (low + levels * step).to(values.dtype)


def _quantize_kmeans(values: torch.Tensor, bits: int, clip: float) -> torch.Tensor:
    """Lloyd's algorithm from a quantile initialisation. Fixed iteration count.

    Deterministic by construction: the initialisation is the sorted quantiles of the
    group, the assignment breaks ties toward the lower centroid index, and an empty
    cluster keeps its previous centroid rather than being reseeded.
    """
    flat = values.flatten().float()
    count = flat.numel()
    levels = 1 << bits
    ordered, _ = torch.sort(flat)
    positions = torch.linspace(0, count - 1, steps=levels, device=flat.device)
    centroids = ordered[positions.round().long().clamp(0, count - 1)].clone()
    for _ in range(LLOYD_ITERATIONS):
        assignment = torch.argmin((flat[:, None] - centroids[None, :]).abs(), dim=1)
        sums = torch.zeros_like(centroids).index_add_(0, assignment, flat)
        counts = torch.zeros_like(centroids).index_add_(0, assignment, torch.ones_like(flat))
        occupied = counts > 0
        centroids = torch.where(occupied, sums / counts.clamp(min=1.0), centroids)
    assignment = torch.argmin((flat[:, None] - centroids[None, :]).abs(), dim=1)
    return centroids[assignment].reshape(values.shape).to(values.dtype)


_QUANTIZERS = {
    "symmetric": _quantize_symmetric,
    "affine": _quantize_affine,
    "kmeans": _quantize_kmeans,
}


# ---------------------------------------------------------------------------
# The scheme, and the accounting over the shards as actually quantized
# ---------------------------------------------------------------------------


def _group_rows(entry, numel: int) -> list:
    groups = entry.get("groups")
    if not isinstance(groups, list) or not groups:
        raise SchemeError("scheme-malformed")
    total = 0
    rows = []
    for raw in groups:
        if not isinstance(raw, dict):
            raise SchemeError("scheme-malformed")
        size = raw.get("size")
        bits = raw.get("bits")
        codebook = str(raw.get("codebook", ""))
        clip = raw.get("clip", 1.0)
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise SchemeError("scheme-malformed")
        if bits not in ALLOWED_BITS:
            raise SchemeError("scheme-malformed")
        if codebook not in ALLOWED_CODEBOOKS:
            raise SchemeError("scheme-malformed")
        try:
            clip = float(clip)
        except (TypeError, ValueError):
            raise SchemeError("scheme-malformed") from None
        if not 0.0 < clip <= 1.0:
            raise SchemeError("scheme-malformed")
        total += size
        rows.append({"size": size, "bits": bits, "codebook": codebook, "clip": clip})
    if total != numel:
        raise SchemeError("scheme-malformed")
    return rows


@torch.no_grad()
def apply_scheme(context: dict, scheme: dict) -> dict:
    """Quantize every parameter shard under the scheme and account every bit it spends.

    The quantized values are written straight into the verifier's candidate model, which is
    instantiated once per run rather than once per scheme, so a sweep pays for the forward
    passes it needs and not for rebuilding a 162 million parameter module thirty-odd times.
    The full-precision reference model is a separate instance and is never written to.

    The accounting is computed here from the groups as actually applied, so a scheme cannot
    state a cost it does not pay. The residue, meaning every bias and every RMSNorm gain, is
    not a shard, is not quantized, and is not charged: it is carried at the checkpoint's own
    width and is outside the budget, which environment/model/checkpoint.json states rather
    than leaving to be inferred.
    """
    table = context["table"]
    scale_bits = int(context["scale_bits"])
    entries = scheme.get("tensors")
    if not isinstance(entries, list) or len(entries) != len(table):
        raise SchemeError("scheme-malformed")
    by_name = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise SchemeError("scheme-malformed")
        if entry["name"] in by_name:
            raise SchemeError("scheme-malformed")
        by_name[entry["name"]] = entry
    if sorted(by_name) != sorted(row["name"] for row in table):
        raise SchemeError("scheme-malformed")

    source = context["reference_view"]
    target = context["candidate_view"]
    per_shard = []
    payload_bits = 0
    overhead_bits = 0
    for row in table:
        rows = _group_rows(by_name[row["name"]], int(row["numel"]))
        flat = source[row["name"]].detach().flatten().float()
        out = torch.empty_like(flat)
        cursor = 0
        shard_payload = 0
        shard_overhead = 0
        for group in rows:
            chunk = flat[cursor:cursor + group["size"]]
            out[cursor:cursor + group["size"]] = _QUANTIZERS[group["codebook"]](
                chunk, group["bits"], group["clip"]
            )
            cursor += group["size"]
            shard_payload += group["size"] * group["bits"]
            shard_overhead += _params(group["codebook"], group["bits"]) * scale_bits
        target[row["name"]].copy_(out.reshape(source[row["name"]].shape))
        payload_bits += shard_payload
        overhead_bits += shard_overhead
        per_shard.append(
            {
                "name": row["name"],
                "groups": len(rows),
                "payload_bits": shard_payload,
                "overhead_bits": shard_overhead,
            }
        )
    return {
        "payload_bits": payload_bits,
        "scale_overhead_bits": overhead_bits,
        "allocated_bits": payload_bits + overhead_bits,
        "per_shard": per_shard,
        "accounted_by": "harness",
    }


def state_digest(context: dict) -> str:
    """A digest over the quantized parameter bytes as the harness currently holds them."""
    digest = hashlib.sha256()
    view = context["candidate_view"]
    for row in context["table"]:
        digest.update(row["name"].encode("utf-8"))
        digest.update(view[row["name"]].detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The forward passes and the readings taken off them
# ---------------------------------------------------------------------------


@torch.no_grad()
def slice_readings(reference: torch.nn.Module, candidate: torch.nn.Module, window: dict) -> dict:
    """Two real forward passes per micro-batch, and the four readings taken off them."""
    work = device()
    inputs = window["inputs"]
    targets = window["targets"]
    total = inputs.numel()
    entropy = 0.0
    cross = 0.0
    reference_nll = 0.0
    candidate_nll = 0.0
    for start in range(0, inputs.size(0), MICRO_BATCH_SEQUENCES):
        batch_inputs = inputs[start:start + MICRO_BATCH_SEQUENCES].to(work)
        batch_targets = targets[start:start + MICRO_BATCH_SEQUENCES].to(work)
        reference_logits = reference.logits(batch_inputs)
        candidate_logits = candidate.logits(batch_inputs)
        reference_log = torch.log_softmax(reference_logits, dim=-1)
        candidate_log = torch.log_softmax(candidate_logits, dim=-1)
        probability = reference_log.exp()
        entropy += float(-(probability * reference_log).sum(dim=-1).sum())
        cross += float(-(probability * candidate_log).sum(dim=-1).sum())
        flat = batch_targets.reshape(-1)
        reference_nll += float(-reference_log.reshape(flat.numel(), -1).gather(1, flat[:, None]).sum())
        candidate_nll += float(-candidate_log.reshape(flat.numel(), -1).gather(1, flat[:, None]).sum())
        del reference_logits, candidate_logits, reference_log, candidate_log, probability
    return {
        "reference_perplexity": _round(math.exp(entropy / total)),
        "quantized_perplexity": _round(math.exp(cross / total)),
        "reference_token_perplexity": _round(math.exp(reference_nll / total)),
        "quantized_token_perplexity": _round(math.exp(candidate_nll / total)),
        "tokens": total,
    }


def evaluate_scheme(context: dict, scheme: dict) -> dict:
    """Quantize once, then read every scheduled held-out slice off that one sealed state."""
    accounting = apply_scheme(context, scheme)
    accounting["bound_total_bits"] = int(context["bit_budget_total"])
    candidate = context["candidate"]
    points = []
    for window in context["slices"]:
        readings = slice_readings(context["reference"], candidate, window)
        base = readings["reference_perplexity"]
        value = readings["quantized_perplexity"]
        points.append(
            {
                "slice": window["id"],
                "shard": window["shard"],
                "token_offset": window["token_offset"],
                "tokens": readings["tokens"],
                "perplexity_reference": base,
                "perplexity_quantized": value,
                # Degradation is RELATIVE, in percent of the unquantized reference
                # perplexity of that slice, so slices whose reference perplexity differs
                # are commensurable and the worst-slice maximum means something across all
                # of them.
                "degradation": _round((value - base) / base * 100.0),
                "degradation_units": "percent of the unquantized reference perplexity",
                "reference_token_perplexity": readings["reference_token_perplexity"],
                "quantized_token_perplexity": readings["quantized_token_perplexity"],
                "filter": "none",
            }
        )
    return {"accounting": accounting, "points": points}


def scheme_metric(context: dict, scheme: dict) -> float:
    """The graded quantity: the WORST degradation across every scheduled held-out slice.

    Worst-case rather than mean, so a favourable slice cannot be harvested and a scheme
    that is excellent on one slice and poor on another is graded on the poor one.
    """
    outcome = evaluate_scheme(context, scheme)
    return _round(max(point["degradation"] for point in outcome["points"]))


# ---------------------------------------------------------------------------
# The two endpoints of the bounded scaling, both measured at grading time
# ---------------------------------------------------------------------------


def uniform_scheme(table: list, bits: int, codebook: str, clip: float) -> dict:
    """One width, one scale group per shard. The shape the handed toolkit emits."""
    return {
        "scheme_version": 1,
        "tensors": [
            {
                "name": row["name"],
                "groups": [{"size": int(row["numel"]), "bits": bits, "codebook": codebook, "clip": clip}],
            }
            for row in table
        ],
        "graded_readout_filter": "none",
        "halt_after": None,
    }


def rule_scheme(table: list, rule: dict) -> dict:
    """A role-keyed allocation: one width, codebook and group count per shard ROLE."""
    tensors = []
    for row in table:
        assignment = rule["roles"].get(row["role"]) or rule["roles"]["default"]
        groups = int(assignment.get("groups", 1))
        numel = int(row["numel"])
        if numel % groups:
            groups = 1
        size = numel // groups
        tensors.append(
            {
                "name": row["name"],
                "groups": [
                    {
                        "size": size,
                        "bits": int(assignment["bits"]),
                        "codebook": assignment["codebook"],
                        "clip": float(assignment.get("clip", 1.0)),
                    }
                    for _ in range(groups)
                ],
            }
        )
    return {
        "scheme_version": 1,
        "tensors": tensors,
        "graded_readout_filter": "none",
        "halt_after": None,
    }


def scheme_cost(table: list, scheme: dict, scale_bits: int) -> int:
    """Account a scheme without quantizing it. Used to skip infeasible candidates."""
    by_name = {entry["name"]: entry for entry in scheme["tensors"]}
    total = 0
    for row in table:
        for group in by_name[row["name"]]["groups"]:
            total += int(group["size"]) * int(group["bits"])
            total += _params(group["codebook"], int(group["bits"])) * scale_bits
    return total


def default_scheme(table: list, default_bits: int) -> dict:
    """The configuration environment/quantize.py ships as its starting point."""
    return uniform_scheme(table, int(default_bits), "symmetric", 1.0)


def default_option_space(context: dict) -> list:
    """Every configuration the handed toolkit can reach that fits the bit budget."""
    budget = int(context["bit_budget_total"])
    out = []
    for bits in ALLOWED_BITS:
        for codebook in DEFAULT_SPACE_CODEBOOKS:
            for clip in ALLOWED_CLIPS:
                scheme = uniform_scheme(context["table"], bits, codebook, clip)
                if scheme_cost(context["table"], scheme, context["scale_bits"]) <= budget:
                    out.append((bits, codebook, clip, scheme))
    return out


def default_optimum(context: dict) -> dict:
    """Exhaustive search over the handed option space, performed by the verifier.

    This is recomputed at grading time rather than read from a stored constant, so the bar
    a submission has to clear is a measurement of the option set the environment actually
    hands out and never an author's recollection of it.
    """
    space = default_option_space(context)
    best = None
    spreads = []
    for bits, codebook, clip, scheme in space:
        outcome = evaluate_scheme(context, scheme)
        values = [point["degradation"] for point in outcome["points"]]
        metric = _round(max(values))
        spreads.append(_round(max(values) - min(values)))
        candidate = {"metric": metric, "bits": bits, "codebook": codebook, "clip": clip}
        if best is None or candidate["metric"] < best["metric"]:
            best = candidate
    if best is None:
        raise SchemeError("scheme-malformed")
    best["searched"] = len(space)
    best["widest_spread"] = _round(max(spreads)) if spreads else None
    best["source"] = "verifier-search-over-handed-option-space"
    return best


# Role-width deltas off the handed optimum, as (embedding, attention, mlp). The embedding
# and the output projection are 77266944 of the 162201600 quantizable parameters and the
# block matrices are the other 84934656, so a delta only fits the budget when what it frees
# covers what it spends: 77266944 per embedding bit against 28311552 per attention bit and
# 56623104 per MLP bit. Every entry below satisfies that inequality at any base width. The
# ladder runs in BOTH directions, because which side of the trade pays is a property of the
# frozen parameters and a probe that only narrowed the embedding would be a bar that could
# not be reached on a checkpoint whose embedding is the sensitive part.
ROLE_WIDTH_LADDER = (
    (-1, 1, 0), (-1, 0, 1), (-1, 2, 0), (-2, 1, 1), (-2, 2, 1), (-2, 3, 1),
    (1, -1, -1), (1, 0, -2), (2, -3, -2),
)


def _snap(bits: int) -> int:
    return min(ALLOWED_BITS, key=lambda value: (abs(value - bits), value))


def derived_rules(optimum: dict) -> list:
    """Probe rules anchored to the measured handed optimum rather than to a guess.

    The first entry reproduces the handed optimum exactly as a role-keyed allocation, which
    is what makes the probe's best reading at or below the handed optimum by construction
    and therefore makes the scaling span non-negative without anyone asserting it. The rest
    walk a fixed ladder of role-width deltas off that configuration, keeping its codebook
    and clip, so every widening spends the budget the handed optimum left on the table.
    """
    base = int(optimum["bits"])
    codebook = optimum["codebook"]
    clip = float(optimum["clip"])
    rules = []
    for index, (embedding, attention, mlp) in ((0, (0, 0, 0)),) + tuple(
        enumerate(ROLE_WIDTH_LADDER, start=1)
    ):
        widths = {
            "token-embedding": _snap(base + embedding),
            "output-projection": _snap(base + embedding),
            "attention-query": _snap(base + attention),
            "attention-key": _snap(base + attention),
            "attention-value": _snap(base + attention),
            "attention-output": _snap(base + attention),
            "mlp-input": _snap(base + mlp),
            "mlp-output": _snap(base + mlp),
        }
        roles = {
            name: {"bits": bits, "codebook": codebook, "clip": clip, "groups": 1}
            for name, bits in widths.items()
        }
        roles["default"] = {"bits": base, "codebook": codebook, "clip": clip, "groups": 1}
        rules.append(
            {
                "id": "derived-from-handed-optimum-" + str(index),
                "note": "handed optimum widened by " + repr((embedding, attention, mlp)),
                "roles": roles,
            }
        )
    return rules


def wider_probe_optimum(context: dict, anchors: dict, optimum: dict) -> dict:
    """The target endpoint, measured at grading time by a fixed widening probe.

    This is a BAR and not a claim about the true optimum of the wider scheme space. The
    probe is a fixed, ordered, budget-checked list of role-keyed allocations: the ones
    declared in tests/anchors.json, plus the ones derived from the handed optimum the
    verifier just measured. The verifier evaluates every entry that fits the budget and
    takes the best. Naming it a probe rather than an optimum is deliberate, because nothing
    here searches the wider space exhaustively and no number in this bundle claims it does.
    """
    budget = int(context["bit_budget_total"])
    best = None
    evaluated = 0
    seen = set()
    for rule in list(anchors["wider_probe"]["rules"]) + derived_rules(optimum):
        scheme = rule_scheme(context["table"], rule)
        signature = json.dumps(scheme["tensors"], sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        if scheme_cost(context["table"], scheme, context["scale_bits"]) > budget:
            continue
        metric = scheme_metric(context, scheme)
        evaluated += 1
        if best is None or metric < best["metric"]:
            best = {"metric": metric, "rule": rule["id"]}
    if best is None:
        return {"metric": None, "rule": None, "evaluated": 0, "source": "verifier-widening-probe"}
    best["evaluated"] = evaluated
    best["source"] = "verifier-widening-probe"
    return best


# ---------------------------------------------------------------------------
# The record every checker reads
# ---------------------------------------------------------------------------


def _requested_filter(scheme: dict) -> str:
    value = scheme.get("graded_readout_filter", "none")
    return "none" if value is None else str(value)


def _reported_metric(scheme: dict):
    for key in ("reported_metric", "metric", "perplexity", "degradation", "score"):
        if key in scheme:
            return {"key": key, "value": scheme[key]}
    return None


def build_context(paths: dict, anchors: dict, root: Path = VERIFIER_ROOT) -> dict:
    """Everything the verifier owns, assembled once, before any scheme is looked at."""
    nanogpt = model_module()
    declaration = substrate(paths)
    manifest = load_json(paths["checkpoint"])
    state, conformance = load_reference_state(paths, root)
    table = nanogpt.shard_table(declaration)
    reference = instantiate(state, paths)
    candidate = instantiate(state, paths)
    return {
        "paths": paths,
        "substrate": declaration,
        "manifest": manifest,
        "conformance": conformance,
        "table": table,
        "reference": reference,
        "reference_view": parameter_view(reference, table),
        "candidate": candidate,
        "candidate_view": parameter_view(candidate, table),
        "slices": load_holdout(anchors, root),
        "scale_bits": int(manifest["scale_bits"]),
        "default_bits": int(manifest["default_bits"]),
        "bit_budget_total": int(manifest["bit_budget_total"]),
        "checkpoint_digest": digest_file(checkpoint_path(root)),
    }


def build_telemetry(paths: dict, scheme: dict, anchors: dict, context: dict | None = None) -> dict:
    """The single record every checker reads. Written by the verifier's process."""
    if context is None:
        context = build_context(paths, anchors)

    phases = ["submission-emitted"]

    declared = anchors["architecture"]
    observed = context["conformance"]
    architecture_agrees = all(
        int(declared[key]) == int(observed[key])
        for key in ("vocab_size", "num_layers", "model_dim", "head_dim")
    )

    outcome = evaluate_scheme(context, scheme)
    submitted_digest = state_digest(context)
    phases.append("state-sealed")

    scheduled = len(outcome["points"])
    halt_raw = scheme.get("halt_after", None)
    halt_after = None
    if isinstance(halt_raw, int) and not isinstance(halt_raw, bool) and halt_raw >= 0:
        halt_after = halt_raw
    kept = outcome["points"] if halt_after is None else outcome["points"][:halt_after]
    for point in kept:
        phases.append("eval-slice-" + str(point["slice"]))

    baseline_scheme = default_scheme(context["table"], context["default_bits"])
    apply_scheme(context, baseline_scheme)
    baseline_digest = state_digest(context)
    optimum = default_optimum(context)
    probe = wider_probe_optimum(context, anchors, optimum)

    graded = _round(max(point["degradation"] for point in kept)) if kept else None
    phases.append("graded")

    degradations = [point["degradation"] for point in kept]
    spread = _round(max(degradations) - min(degradations)) if degradations else None
    tolerance = None
    if optimum.get("widest_spread") is not None:
        tolerance = _round(1.5 * float(optimum["widest_spread"]) + 0.01)

    return {
        "schema": SCHEMA,
        "slot": "OER-21",
        "phase_log": phases,
        "substrate": {
            "declaration": anchors["substrate_declaration"],
            "declared_architecture": declared,
            "observed_architecture": {
                key: int(observed[key]) for key in ("vocab_size", "num_layers", "model_dim", "head_dim")
            },
            "architecture_agrees": bool(architecture_agrees),
            "shards": int(observed["shards"]),
            "quantizable_parameters": int(observed["quantizable_parameters"]),
            "residue_parameters": int(observed["residue_parameters"]),
            "parameters_are": "a real snapshot of the frozen decoder, loaded from the verifier's own copy",
            "checkpoint_source": "verifier-owned",
            "checkpoint_digest": context["checkpoint_digest"],
            "checkpoint_digest_state": anchors["checkpoint"]["sha256_state"],
            "checkpoint_digest_bound": anchors["checkpoint"]["sha256"],
        },
        "holdout": {
            "source": "verifier-owned",
            "present_in_environment": False,
            "slices": [
                {"id": point["slice"], "shard": point["shard"], "token_offset": point["token_offset"],
                 "tokens": point["tokens"]}
                for point in outcome["points"]
            ],
            "corpus": anchors["holdout"]["corpus"],
        },
        "budget": outcome["accounting"],
        "state": {
            "sealed": True,
            "default_scheme_state_digest": baseline_digest,
            "proposed_scheme_state_digest": submitted_digest,
        },
        "readout": {
            "graded_filter": "none",
            "requested_graded_filter": _requested_filter(scheme),
            "scheme_reported_metric": _reported_metric(scheme),
            "graded_source": "verifier-raw-recompute",
        },
        "evaluation": {
            "scheduled_points": scheduled,
            "completed_points": len(kept),
            "halt_requested_after": halt_after,
            "points": kept,
            "spread": spread,
            "sustain_tolerance": tolerance,
            "sustain_tolerance_source": "verifier-measured-over-the-handed-option-space",
        },
        "graded": {
            "metric": graded,
            "source": "verifier-raw-recompute",
            "definition": "worst per-slice perplexity degradation against the unquantized reference, measured by real forward passes on held-out FineWeb slices",
        },
        "anchors": {
            "state": anchors["anchors_state"],
            "baseline_metric": anchors["baseline_metric"],
            "target_metric": anchors["target_metric"],
            "gap": anchors["anchors_gap"],
            "local_scaling": {
                "default_optimum": optimum["metric"],
                "default_optimum_source": optimum["source"],
                "default_optimum_configuration": {
                    "bits": optimum["bits"],
                    "codebook": optimum["codebook"],
                    "clip": optimum["clip"],
                    "searched": optimum["searched"],
                },
                "wider_probe_optimum": probe["metric"],
                "wider_probe_source": probe["source"],
                "wider_probe_rule": probe["rule"],
                "wider_probe_evaluated": probe["evaluated"],
                "separation_margin": anchors["local_scaling"]["separation_margin"],
                "sustain_tolerance": tolerance,
            },
        },
    }
