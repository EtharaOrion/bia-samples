"""Harness-owned quantization, evaluation and telemetry for slot OER-21.

This module is the only thing that computes a perplexity in this bundle. It runs
inside the verifier's process, over the verifier's own copy of the frozen model
and the frozen evaluation corpus, and it never reads a number the submission
produced. The submission supplies a QUANTIZATION SCHEME and nothing else; the
scheme is a specification of how bits are allocated, not a report of what that
allocation achieved.

Three separations are load-bearing and are the reason this file exists apart from
tests/grade.py and tests/checkers.py:

1.  The graded reading is recomputed here from harness-owned quantized tensors.
    A filter the submission asks for on the graded path is RECORDED as an attempt
    and never applied, so the graded number is always the raw recomputation.
2.  The bit accounting is performed here, over the tensors as actually quantized,
    by summing what each group costs. It is never the submission's own claim
    about what its scheme costs.
3.  The evaluation points are scheduled here, over disjoint frozen shards, so a
    submission cannot choose which readings enter the record. A submission may
    ask to halt early; the request is recorded and the shortfall is graded.

No clock, no network, no random source, no environment secret. Pure arithmetic
over frozen bytes, so two runs over the same bytes produce identical telemetry.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = "forge.oer21.telemetry/v1"

# Quantization vocabulary the FULL scheme space admits. The handed default
# toolkit in environment/quantize.py exposes a strict subset of this on purpose;
# widening past that subset is the whole task.
ALLOWED_BITS = (2, 3, 4, 5, 6, 8)
ALLOWED_CODEBOOKS = ("symmetric", "affine", "kmeans")
ALLOWED_CLIPS = (1.0, 0.999, 0.99, 0.95, 0.9)

# Parameter count per scale group, in units of `scale_bits`. A kmeans codebook
# pays for its whole centroid table, which is what makes a non-uniform codebook
# affordable at low width and unaffordable at high width.
CODEBOOK_PARAMS = {"symmetric": 1, "affine": 2}

# The option set environment/quantize.py hands the agent: one uniform width for
# every tensor, one scale group per tensor, symmetric or affine, one clip knob.
DEFAULT_SPACE_CODEBOOKS = ("symmetric", "affine")

LLOYD_ITERATIONS = 8
ROUND_DIGITS = 9


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def digest_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _params(codebook: str, bits: int) -> int:
    if codebook == "kmeans":
        return 1 << bits
    return CODEBOOK_PARAMS[codebook]


def _clip_magnitude(values, clip: float) -> float:
    """The magnitude at the clip percentile of the group, used as the symmetric bound."""
    magnitudes = sorted(abs(value) for value in values)
    count = len(magnitudes)
    index = max(0, min(count - 1, int(math.ceil(clip * count)) - 1))
    return magnitudes[index]


def _quantize_symmetric(values, bits: int, clip: float):
    amax = _clip_magnitude(values, clip)
    span = (1 << (bits - 1)) - 1
    if span <= 0 or amax <= 0.0:
        return [0.0 for _ in values]
    step = amax / span
    out = []
    for value in values:
        level = int(math.floor(value / step + 0.5))
        level = max(-span, min(span, level))
        out.append(level * step)
    return out


def _quantize_affine(values, bits: int, clip: float):
    ordered = sorted(values)
    count = len(ordered)
    keep = max(1, int(math.ceil(clip * count)))
    drop = count - keep
    low = ordered[drop // 2]
    high = ordered[count - 1 - (drop - drop // 2)]
    levels = (1 << bits) - 1
    if high <= low or levels <= 0:
        return [low for _ in values]
    step = (high - low) / levels
    out = []
    for value in values:
        clamped = max(low, min(high, value))
        level = int(math.floor((clamped - low) / step + 0.5))
        level = max(0, min(levels, level))
        out.append(low + level * step)
    return out


def _quantize_kmeans(values, bits: int, clip: float):
    """Lloyd's algorithm from a quantile initialisation. Fixed iteration count.

    Deterministic by construction: the initialisation is the sorted quantiles of
    the group, the assignment breaks ties toward the lower centroid index, and an
    empty cluster keeps its previous centroid rather than being reseeded.
    """
    count = len(values)
    levels = 1 << bits
    ordered = sorted(values)
    centroids = []
    for k in range(levels):
        position = 0 if levels == 1 else int(round(k * (count - 1) / (levels - 1)))
        centroids.append(ordered[max(0, min(count - 1, position))])
    for _ in range(LLOYD_ITERATIONS):
        sums = [0.0] * levels
        counts = [0] * levels
        for value in values:
            best, best_distance = 0, abs(value - centroids[0])
            for k in range(1, levels):
                distance = abs(value - centroids[k])
                if distance < best_distance:
                    best, best_distance = k, distance
            sums[best] += value
            counts[best] += 1
        for k in range(levels):
            if counts[k]:
                centroids[k] = sums[k] / counts[k]
    out = []
    for value in values:
        best, best_distance = 0, abs(value - centroids[0])
        for k in range(1, levels):
            distance = abs(value - centroids[k])
            if distance < best_distance:
                best, best_distance = k, distance
        out.append(centroids[best])
    return out


_QUANTIZERS = {
    "symmetric": _quantize_symmetric,
    "affine": _quantize_affine,
    "kmeans": _quantize_kmeans,
}


class SchemeError(ValueError):
    """A scheme that is not a scheme. Refused before any bit is accounted."""


def _group_rows(entry, tensor_size: int):
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
        if not isinstance(size, int) or size <= 0:
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
    if total != tensor_size:
        raise SchemeError("scheme-malformed")
    return rows


def apply_scheme(model: dict, scheme: dict) -> tuple:
    """Quantize every tensor under the scheme and account every bit it spends.

    Returns the quantized tensors and the accounting record. The accounting is
    computed here from the groups as actually applied, so a scheme cannot state a
    cost it does not pay.
    """
    tensor_size = int(model["tensor_size"])
    scale_bits = int(model["scale_bits"])
    entries = scheme.get("tensors")
    tensors = model["tensors"]
    if not isinstance(entries, list) or len(entries) != len(tensors):
        raise SchemeError("scheme-malformed")

    quantized = []
    per_tensor = []
    payload_bits = 0
    overhead_bits = 0
    for entry, tensor in zip(entries, tensors):
        if not isinstance(entry, dict):
            raise SchemeError("scheme-malformed")
        rows = _group_rows(entry, tensor_size)
        values = tensor["values"]
        out = []
        cursor = 0
        tensor_payload = 0
        tensor_overhead = 0
        for row in rows:
            chunk = values[cursor:cursor + row["size"]]
            cursor += row["size"]
            out.extend(_QUANTIZERS[row["codebook"]](chunk, row["bits"], row["clip"]))
            tensor_payload += row["size"] * row["bits"]
            tensor_overhead += _params(row["codebook"], row["bits"]) * scale_bits
        quantized.append(out)
        payload_bits += tensor_payload
        overhead_bits += tensor_overhead
        per_tensor.append(
            {
                "name": tensor["name"],
                "groups": len(rows),
                "payload_bits": tensor_payload,
                "overhead_bits": tensor_overhead,
            }
        )
    accounting = {
        "bound_total_bits": int(model["bit_budget_total"]),
        "payload_bits": payload_bits,
        "scale_overhead_bits": overhead_bits,
        "allocated_bits": payload_bits + overhead_bits,
        "per_tensor": per_tensor,
        "accounted_by": "harness",
    }
    return quantized, accounting


def state_digest(quantized) -> str:
    """A digest over the quantized tensors as the harness holds them."""
    payload = json.dumps(
        [[format(value, ".9f") for value in tensor] for tensor in quantized],
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logit_row(tensors, model, context, vocab, size, gains, scale, stride_context, stride_tensor, count):
    """One row of logits for one context token. The whole forward pass is this."""
    logits = []
    for word in range(vocab):
        accumulator = 0.0
        for index in range(count):
            slot = (context * stride_context + word + index * stride_tensor) % size
            accumulator += gains[index] * tensors[index][slot]
        logits.append(accumulator * scale)
    return logits


def logits_for(tensors, model: dict, context: int) -> list:
    """The forward pass at one context, for callers that only want the row."""
    return logit_row(
        tensors,
        model,
        context,
        int(model["vocab"]),
        int(model["tensor_size"]),
        model["gains"],
        float(model["logit_scale"]),
        int(model["index"]["stride_context"]),
        int(model["index"]["stride_tensor"]),
        len(tensors),
    )


FLOOR = 1e-300


def softmax(logits) -> list:
    peak = max(logits)
    exponentials = [math.exp(value - peak) for value in logits]
    partition = sum(exponentials)
    return [value / partition for value in exponentials]


def reference_profile(model: dict, tokens) -> tuple:
    """The full-precision model's own next-token distribution at every context.

    This is the target distribution the quantized model is scored against, which is
    what makes the graded degradation non-negative: by Gibbs' inequality the cross
    entropy H(p, q) is at least the entropy H(p), with equality only when the
    quantized distribution equals the full-precision one. A scheme can therefore
    never buy a better-than-reference reading out of quantization noise.
    """
    raw = [tensor["values"] for tensor in model["tensors"]]
    profile = []
    entropy = 0.0
    for context in tokens:
        row = softmax(logits_for(raw, model, context))
        profile.append(row)
        entropy += -sum(value * math.log(max(value, FLOOR)) for value in row)
    return profile, math.exp(entropy / len(tokens))


def cross_perplexity(tensors, model: dict, tokens, profile) -> float:
    """exp of the mean cross entropy of the quantized distribution under the reference."""
    total = 0.0
    for context, row in zip(tokens, profile):
        quantized = softmax(logits_for(tensors, model, context))
        total += -sum(
            value * math.log(max(other, FLOOR)) for value, other in zip(row, quantized)
        )
    return math.exp(total / len(tokens))


def reference_readings(model: dict, corpus: dict) -> list:
    return [_round(reference_profile(model, shard["tokens"])[1]) for shard in corpus["shards"]]


def corpus_profiles(model: dict, corpus: dict) -> list:
    return [reference_profile(model, shard["tokens"])[0] for shard in corpus["shards"]]


def evaluate_scheme(model: dict, corpus: dict, scheme: dict, reference: list, profiles=None) -> dict:
    """Quantize once, then read every scheduled shard off that one sealed state."""
    quantized, accounting = apply_scheme(model, scheme)
    if profiles is None:
        profiles = corpus_profiles(model, corpus)
    points = []
    for shard, base, profile in zip(corpus["shards"], reference, profiles):
        value = _round(cross_perplexity(quantized, model, shard["tokens"], profile))
        points.append(
            {
                "shard": int(shard["id"]),
                "perplexity_reference": base,
                "perplexity_quantized": value,
                # Degradation is RELATIVE, in percent of the unquantized reference
                # perplexity of that shard, so shards whose reference perplexity
                # differs are commensurable and the worst-shard maximum means
                # something across all of them.
                "degradation": _round((value - base) / base * 100.0),
                "degradation_units": "percent of the unquantized reference perplexity",
                "filter": "none",
            }
        )
    return {
        "quantized": quantized,
        "accounting": accounting,
        "points": points,
        "state_digest": state_digest(quantized),
    }


def uniform_scheme(model: dict, bits: int, codebook: str, clip: float) -> dict:
    """One width, one scale group per tensor. The shape the handed toolkit emits."""
    size = int(model["tensor_size"])
    return {
        "scheme_version": 1,
        "tensors": [
            {
                "name": tensor["name"],
                "groups": [{"size": size, "bits": bits, "codebook": codebook, "clip": clip}],
            }
            for tensor in model["tensors"]
        ],
        "graded_readout_filter": "none",
        "halt_after": None,
    }


def default_scheme(model: dict) -> dict:
    """The configuration environment/quantize.py ships as its starting point."""
    return uniform_scheme(model, int(model["default_bits"]), "symmetric", 1.0)


def default_option_space(model: dict) -> list:
    """Every configuration the handed toolkit can reach that fits the bit budget."""
    budget = int(model["bit_budget_total"])
    out = []
    for bits in ALLOWED_BITS:
        for codebook in DEFAULT_SPACE_CODEBOOKS:
            for clip in ALLOWED_CLIPS:
                scheme = uniform_scheme(model, bits, codebook, clip)
                _, accounting = apply_scheme(model, scheme)
                if accounting["allocated_bits"] <= budget:
                    out.append(scheme)
    return out


def scheme_metric(model: dict, corpus: dict, scheme: dict, reference: list, profiles=None) -> float:
    """The graded quantity: the WORST degradation across every scheduled shard.

    Worst-case rather than mean, so a favourable shard cannot be harvested and a
    scheme that is excellent on one shard and poor on another is graded on the
    poor one.
    """
    outcome = evaluate_scheme(model, corpus, scheme, reference, profiles)
    return _round(max(point["degradation"] for point in outcome["points"]))


def default_optimum(model: dict, corpus: dict, reference: list, profiles=None) -> dict:
    """Exhaustive search over the handed option space, performed by the verifier.

    This is recomputed at grading time rather than read from a stored constant,
    so the bar a submission has to clear is a measurement of the option set the
    environment actually hands out and never an author's recollection of it.
    """
    space = default_option_space(model)
    if profiles is None:
        profiles = corpus_profiles(model, corpus)
    best = None
    for scheme in space:
        metric = scheme_metric(model, corpus, scheme, reference, profiles)
        row = scheme["tensors"][0]["groups"][0]
        candidate = {
            "metric": metric,
            "bits": row["bits"],
            "codebook": row["codebook"],
            "clip": row["clip"],
        }
        if best is None or candidate["metric"] < best["metric"]:
            best = candidate
    if best is None:
        raise SchemeError("scheme-malformed")
    best["searched"] = len(space)
    best["source"] = "verifier-search-over-handed-option-space"
    return best


def _requested_filter(scheme: dict) -> str:
    value = scheme.get("graded_readout_filter", "none")
    return "none" if value is None else str(value)


def _reported_metric(scheme: dict):
    for key in ("reported_metric", "metric", "perplexity", "degradation", "score"):
        if key in scheme:
            return {"key": key, "value": scheme[key]}
    return None


def build_telemetry(paths: dict, scheme: dict, anchors: dict) -> dict:
    """The single record every checker reads. Written by the verifier's process."""
    model = load_json(paths["model"])
    corpus = load_json(paths["corpus"])
    stored_reference = load_json(paths["reference"])

    phases = ["submission-emitted"]
    observed = {
        "model": digest_file(paths["model"]),
        "corpus": digest_file(paths["corpus"]),
        "reference": digest_file(paths["reference"]),
    }
    bound = dict(anchors["frozen_digests"])

    profiles = corpus_profiles(model, corpus)
    recomputed = reference_readings(model, corpus)
    reference_consistent = recomputed == list(stored_reference["per_shard_perplexity"])

    outcome = evaluate_scheme(model, corpus, scheme, recomputed, profiles)
    phases.append("state-sealed")

    scheduled = len(outcome["points"])
    halt_raw = scheme.get("halt_after", None)
    halt_after = None
    if isinstance(halt_raw, int) and not isinstance(halt_raw, bool) and halt_raw >= 0:
        halt_after = halt_raw
    kept = outcome["points"] if halt_after is None else outcome["points"][:halt_after]
    for point in kept:
        phases.append("eval-shard-" + str(point["shard"]))

    baseline = evaluate_scheme(model, corpus, default_scheme(model), recomputed, profiles)
    optimum = default_optimum(model, corpus, recomputed, profiles)

    graded = _round(max(point["degradation"] for point in kept)) if kept else None
    phases.append("graded")

    degradations = [point["degradation"] for point in kept]
    spread = _round(max(degradations) - min(degradations)) if degradations else None

    return {
        "schema": SCHEMA,
        "slot": "OER-21",
        "phase_log": phases,
        "frozen_inputs": {
            "bound": bound,
            "observed": observed,
            "reference_readings_consistent": bool(reference_consistent),
        },
        "budget": outcome["accounting"],
        "state": {
            "sealed": True,
            "default_scheme_state_digest": baseline["state_digest"],
            "submitted_scheme_state_digest": outcome["state_digest"],
        },
        "readout": {
            "graded_filter": "none",
            "requested_graded_filter": _requested_filter(scheme),
            "submission_reported_metric": _reported_metric(scheme),
            "graded_source": "verifier-raw-recompute",
        },
        "evaluation": {
            "scheduled_points": scheduled,
            "completed_points": len(kept),
            "halt_requested_after": halt_after,
            "points": kept,
            "spread": spread,
            "sustain_tolerance": anchors["local_scaling"]["sustain_tolerance"],
        },
        "graded": {
            "metric": graded,
            "source": "verifier-raw-recompute",
            "definition": "worst per-shard perplexity degradation against the unquantized reference",
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
                "reference_optimum": anchors["local_scaling"]["reference_optimum"],
                "separation_margin": anchors["local_scaling"]["separation_margin"],
                "sustain_tolerance": anchors["local_scaling"]["sustain_tolerance"],
            },
        },
    }
