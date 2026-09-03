# FORGE-CANARY-BEGIN
# 0: be0d2895ee9c314b0b788f8132472cd8400fdcc05a2a1f2ad8cfece235cf16b9
# 1: 6e2dcea569555b0f090e959f935a9c34b5515f5bbdd4cb8a9be6d3e27153f029
# 2: 31515bcc51c1de9b41ead416fc746afdb926e71b51fbe1e1b0ec53dfd45a3127
# 3: 8398998a2b36d38fe6286631a70571d58ef8ed5e4f85569ed6962518a70c69f4
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in slot OER-21 from solution/grounding.yaml alone.

One private source, one deriver. This script generates the task substrate, the
harness anchors, the checker fixtures, the golden trajectory, solve.sh, the
reference solution, TRUTH.md, rubrics.json and tests/test_output.py. Nothing it
writes was typed by hand, so nothing it writes can drift away from the source
without this script saying so.

It invokes no model, no network, no clock, no locale and no random source. The
only entropy is a Lehmer recurrence whose modulus, multiplier and seeds are
written out in grounding.yaml, so two runs over frozen bytes produce
byte-identical output.

Modes:
    (no argument)  regenerate every artifact in place
    --check        regenerate into memory, re-run the measurement, and exit
                   non-zero if any committed byte or any measured constant drifts
    --measure      re-run the searches and print the `measured:` block, so the
                   frozen constants in grounding.yaml can be refreshed by
                   measurement rather than by editing
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "tests"))

import evaluate  # noqa: E402  the harness-owned engine is the only numeric authority

import yaml  # noqa: E402

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
BANNER_LINE = BANNER + " Source: " + SOURCE + ", derived by solution/recompute.py."

ROUND = 9


def _round(value):
    return round(float(value), ROUND)


# ---------------------------------------------------------------------------
# Deterministic generation of the frozen substrate
# ---------------------------------------------------------------------------


class Lehmer:
    """A Lehmer generator written out in full. No random module anywhere."""

    def __init__(self, modulus: int, multiplier: int, increment: int, seed: int):
        self.modulus = modulus
        self.multiplier = multiplier
        self.increment = increment
        self.state = seed % modulus

    def next_unit(self) -> float:
        self.state = (self.multiplier * self.state + self.increment) % self.modulus
        return self.state / self.modulus


def build_model(spec: dict) -> dict:
    lcg = spec["weight_lcg"]
    digits = int(spec["weight_round_digits"])
    tensors = []
    for index in range(int(spec["tensor_count"])):
        stream = Lehmer(
            int(lcg["modulus"]),
            int(lcg["multiplier"]),
            int(lcg["increment"]),
            int(lcg["seed_base"]) + index * int(lcg["seed_stride"]),
        )
        profile = spec["weight_profiles"][index]
        values = []
        for position in range(int(spec["tensor_size"])):
            value = 2.0 * stream.next_unit() - 1.0
            if profile == "heavy" and position % int(spec["weight_outlier_period"]) == 0:
                value *= float(spec["weight_outlier_gain"])
            values.append(round(value, digits))
        tensors.append({"name": "t%02d" % index, "profile": profile, "values": values})
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.model/v1",
        "vocab": int(spec["vocab"]),
        "tensor_count": int(spec["tensor_count"]),
        "tensor_size": int(spec["tensor_size"]),
        "index": {
            "stride_context": int(spec["stride_context"]),
            "stride_tensor": int(spec["stride_tensor"]),
            "formula": "weight index = (context * stride_context + word + tensor * stride_tensor) mod tensor_size",
        },
        "gains": list(spec["gains"]),
        "logit_scale": float(spec["logit_scale"]),
        "scale_bits": int(spec["scale_bits"]),
        "default_bits": int(spec["default_bits"]),
        "bit_budget_total": int(spec["bit_budget_total"]),
        "tensors": tensors,
    }


def build_corpus(spec: dict, model: dict) -> dict:
    """Context tokens per shard, drawn from that shard's own subset.

    The corpus supplies CONTEXTS only. The scored target at every context is the
    full-precision model's own next-token distribution, so no token label is needed
    and no argmax discontinuity enters the reading.
    """
    corpus = spec["corpus"]
    shards = []
    for index, subset in enumerate(corpus["token_subsets"]):
        stream = Lehmer(
            int(spec["weight_lcg"]["modulus"]),
            int(spec["weight_lcg"]["multiplier"]),
            0,
            int(corpus["lcg_seed_base"]) + index * int(corpus["lcg_seed_stride"]),
        )
        tokens = []
        for _ in range(int(corpus["shard_length"])):
            tokens.append(int(subset[int(stream.next_unit() * len(subset)) % len(subset)]))
        shards.append({"id": index, "token_subset": list(subset), "tokens": tokens})
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.corpus/v1",
        "vocab": int(spec["vocab"]),
        "shard_count": len(shards),
        "shards": shards,
    }


def build_reference(model: dict, corpus: dict) -> dict:
    readings = evaluate.reference_readings(model, corpus)
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.reference/v1",
        "statement": "Perplexity of the UNQUANTIZED model on each frozen shard. Degradation is measured against these.",
        "per_shard_perplexity": readings,
        "worst_shard_perplexity": max(readings),
    }


def build_schema_document(spec: dict) -> dict:
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.scheme_schema/v1",
        "statement": "The FULL quantization scheme space this task admits. Every axis below is free.",
        "artifact": "scheme.json, written into the directory named by OUT_DIR",
        "shape": {
            "scheme_version": 1,
            "tensors": [
                {
                    "name": "t00",
                    "groups": [
                        {"size": "int, group sizes within one tensor must sum to tensor_size",
                         "bits": list(evaluate.ALLOWED_BITS),
                         "codebook": list(evaluate.ALLOWED_CODEBOOKS),
                         "clip": list(evaluate.ALLOWED_CLIPS)}
                    ],
                }
            ],
            "graded_readout_filter": "must be none; any other value is graded zero",
            "halt_after": "must be null; an integer requests an early stop and is graded zero",
        },
        "bit_accounting": {
            "payload_bits": "sum over groups of size * bits",
            "overhead_bits": "sum over groups of parameters * scale_bits",
            "parameters_per_group": {"symmetric": 1, "affine": 2, "kmeans": "2 ** bits"},
            "total_must_satisfy": "payload_bits + overhead_bits <= bit_budget_total",
            "accounted_by": "the harness, over the tensors as actually quantized",
        },
        "free_axes": [
            "the bit width of every group, independently",
            "the number of scale groups inside a tensor, and where their boundaries fall",
            "the codebook family of every group",
            "the clip percentile of every group",
        ],
        "frozen_axes": [
            "the model weights",
            "the total bit budget",
            "the evaluation corpus",
            "the unquantized reference readings",
        ],
    }


# ---------------------------------------------------------------------------
# Measurement: the default optimum, the reference optimum, the control schemes
# ---------------------------------------------------------------------------


def _tensor_entry(model: dict, index: int, rows: list) -> dict:
    return {"name": model["tensors"][index]["name"], "groups": [dict(row) for row in rows]}


def _scheme_from_rows(model: dict, rows_per_tensor: list) -> dict:
    return {
        "scheme_version": 1,
        "tensors": [_tensor_entry(model, index, rows) for index, rows in enumerate(rows_per_tensor)],
        "graded_readout_filter": "none",
        "halt_after": None,
    }


def _candidates(size: int) -> list:
    """The fixed candidate list every tensor is swept against. Order is frozen."""
    rows = []
    for bits in (2, 3, 4, 5, 6, 8):
        for codebook in ("symmetric", "affine"):
            for clip in (1.0, 0.99, 0.95):
                rows.append([{"size": size, "bits": bits, "codebook": codebook, "clip": clip}])
    for bits in (2, 3):
        rows.append([{"size": size, "bits": bits, "codebook": "kmeans", "clip": 1.0}])
    half = size // 2
    for high, low in ((6, 2), (5, 2), (4, 2), (8, 2), (5, 3), (6, 3)):
        rows.append(
            [
                {"size": half, "bits": high, "codebook": "affine", "clip": 1.0},
                {"size": size - half, "bits": low, "codebook": "affine", "clip": 1.0},
            ]
        )
        rows.append(
            [
                {"size": half, "bits": low, "codebook": "affine", "clip": 1.0},
                {"size": size - half, "bits": high, "codebook": "affine", "clip": 1.0},
            ]
        )
    quarter = size // 4
    for bits in (4, 5):
        rows.append(
            [{"size": quarter, "bits": bits, "codebook": "affine", "clip": 1.0} for _ in range(4)]
        )
    return rows


def _cost(model: dict, rows: list) -> int:
    scale_bits = int(model["scale_bits"])
    total = 0
    for row in rows:
        total += row["size"] * row["bits"]
        total += evaluate._params(row["codebook"], row["bits"]) * scale_bits
    return total


def search_reference(model: dict, corpus: dict, reference: list, passes: int) -> dict:
    """Greedy coordinate descent over the wider space. Deterministic, no restarts."""
    size = int(model["tensor_size"])
    budget = int(model["bit_budget_total"])
    count = int(model["tensor_count"])
    current = [[{"size": size, "bits": 3, "codebook": "affine", "clip": 1.0}] for _ in range(count)]
    spent = sum(_cost(model, rows) for rows in current)
    best_metric = evaluate.scheme_metric(model, corpus, _scheme_from_rows(model, current), reference)
    candidates = _candidates(size)
    for _ in range(passes):
        improved = False
        for index in range(count):
            for rows in candidates:
                proposal = _cost(model, rows)
                if spent - _cost(model, current[index]) + proposal > budget:
                    continue
                keep = current[index]
                current[index] = [dict(row) for row in rows]
                metric = evaluate.scheme_metric(
                    model, corpus, _scheme_from_rows(model, current), reference
                )
                if metric < best_metric:
                    best_metric = metric
                    spent = spent - _cost(model, keep) + proposal
                    improved = True
                else:
                    current[index] = keep
        if not improved:
            break
    scheme = _scheme_from_rows(model, current)
    return {"scheme": scheme, "metric": best_metric, "allocated_bits": spent}


def _spread(model, corpus, scheme, reference) -> float:
    outcome = evaluate.evaluate_scheme(model, corpus, scheme, reference)
    values = [point["degradation"] for point in outcome["points"]]
    return _round(max(values) - min(values))


def search_unsustained(model: dict, corpus: dict, reference: list) -> dict:
    """A scheme that protects the index range one shard touches and starves the rest."""
    size = int(model["tensor_size"])
    budget = int(model["bit_budget_total"])
    count = int(model["tensor_count"])
    best = None
    for split in (8, 16, 24, 32, 40):
        for high, low in ((8, 2), (6, 2), (5, 2), (8, 3), (6, 3)):
            rows = [
                {"size": split, "bits": high, "codebook": "affine", "clip": 1.0},
                {"size": size - split, "bits": low, "codebook": "affine", "clip": 1.0},
            ]
            if _cost(model, rows) * count > budget:
                continue
            scheme = _scheme_from_rows(model, [[dict(r) for r in rows] for _ in range(count)])
            spread = _spread(model, corpus, scheme, reference)
            row = {"split": split, "high": high, "low": low, "spread": spread, "scheme": scheme}
            if best is None or spread > best["spread"]:
                best = row
    return best


def measure(model: dict, corpus: dict, spec: dict) -> dict:
    reference = evaluate.reference_readings(model, corpus)
    shipped = evaluate.default_scheme(model)
    shipped_metric = evaluate.scheme_metric(model, corpus, shipped, reference)
    shipped_digest = evaluate.state_digest(evaluate.apply_scheme(model, shipped)[0])

    rows = []
    for scheme in evaluate.default_option_space(model):
        group = scheme["tensors"][0]["groups"][0]
        rows.append(
            {
                "bits": group["bits"],
                "codebook": group["codebook"],
                "clip": group["clip"],
                "metric": evaluate.scheme_metric(model, corpus, scheme, reference),
                "digest": evaluate.state_digest(evaluate.apply_scheme(model, scheme)[0]),
                "scheme": scheme,
            }
        )
    rows.sort(key=lambda row: (row["metric"], row["bits"], row["codebook"], -row["clip"]))
    optimum = rows[0]
    plateau = next((row for row in rows if row["digest"] != shipped_digest), rows[0])

    found = search_reference(model, corpus, reference, int(spec["passes"]))
    reference_spread = _spread(model, corpus, found["scheme"], reference)
    optimum_spread = _spread(model, corpus, optimum["scheme"], reference)
    plateau_spread = _spread(model, corpus, plateau["scheme"], reference)
    tolerance = _round(1.5 * max(optimum_spread, reference_spread, plateau_spread) + 0.01)
    unsustained = search_unsustained(model, corpus, reference)

    return {
        "state": "measured",
        "default_shipped_metric": shipped_metric,
        "default_optimum": optimum["metric"],
        "default_optimum_configuration": {
            "bits": optimum["bits"],
            "codebook": optimum["codebook"],
            "clip": optimum["clip"],
            "searched": len(rows),
        },
        "default_optimum_spread": optimum_spread,
        "default_optimum_is_the_shipped_configuration": optimum["digest"] == shipped_digest,
        "plateau_control_metric": plateau["metric"],
        "plateau_control_configuration": {
            "bits": plateau["bits"],
            "codebook": plateau["codebook"],
            "clip": plateau["clip"],
        },
        "plateau_control_spread": plateau_spread,
        "reference_optimum": found["metric"],
        "reference_scheme": found["scheme"],
        "reference_scheme_allocated_bits": found["allocated_bits"],
        "reference_spread": reference_spread,
        "sustain_tolerance": tolerance,
        "sustain_tolerance_rule": "one and a half times the widest shard-to-shard spread shown by the handed option set optimum, by the best non-shipped handed configuration, or by the reference scheme, plus one hundredth of a point",
        "unsustained_control_scheme": unsustained["scheme"],
        "unsustained_control_spread": unsustained["spread"],
        "unsustained_control_metric": evaluate.scheme_metric(model, corpus, unsustained["scheme"], reference),
        "gap_default_minus_reference": _round(optimum["metric"] - found["metric"]),
        "gap_shipped_minus_reference": _round(shipped_metric - found["metric"]),
    }


# ---------------------------------------------------------------------------
# Generated artifacts
# ---------------------------------------------------------------------------


def build_anchors(grounding: dict, model: dict, paths: dict, measured: dict, reference_source: str) -> dict:
    reward = grounding["reward"]
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.anchors/v1",
        "slot": "OER-21",
        "family": "F9",
        "anchors_state": reward["anchors_state"],
        "baseline_metric": reward["baseline_metric"],
        "target_metric": reward["target_metric"],
        "anchors_gap": reward["anchors_gap"],
        "anchors_absent_reason": reward["anchors_absent_reason"].strip(),
        "frozen_digests": {
            "model": evaluate.digest_file(paths["model"]),
            "corpus": evaluate.digest_file(paths["corpus"]),
            "reference": evaluate.digest_file(paths["reference"]),
        },
        "local_scaling": {
            "reference_optimum": measured["reference_optimum"],
            "default_optimum": measured["default_optimum"],
            "default_optimum_recomputed_at_grading_time": True,
            "separation_margin": reward["separation_margin"],
            "sustain_tolerance": measured["sustain_tolerance"],
            "note": "measured endpoints of the bounded scaling, NOT the family anchors",
        },
        "bit_budget_total": int(model["bit_budget_total"]),
        "reference_solution_sha256": hashlib.sha256(reference_source.encode("utf-8")).hexdigest(),
        "reference_solution_path": "solution/reference.py",
    }


def build_fixtures(model: dict, measured: dict) -> dict:
    reference = json.loads(json.dumps(measured["reference_scheme"]))
    fixtures = {}

    fixtures["reference"] = json.loads(json.dumps(reference))

    fixtures["noop"] = evaluate.default_scheme(model)

    size = int(model["tensor_size"])
    plateau = measured["plateau_control_configuration"]
    fixtures["plateau"] = evaluate.uniform_scheme(
        model, plateau["bits"], plateau["codebook"], plateau["clip"]
    )

    overspend = json.loads(json.dumps(reference))
    overspend["tensors"][0]["groups"] = [
        {"size": size, "bits": 8, "codebook": "affine", "clip": 1.0}
    ]
    fixtures["overspend"] = overspend

    smoothed = json.loads(json.dumps(reference))
    smoothed["graded_readout_filter"] = "ema"
    fixtures["smoothed"] = smoothed

    reported = json.loads(json.dumps(reference))
    reported["reported_metric"] = 0.0
    fixtures["reported_metric"] = reported

    early = json.loads(json.dumps(reference))
    early["halt_after"] = 2
    fixtures["early_stop"] = early

    fixtures["unsustained"] = json.loads(json.dumps(measured["unsustained_control_scheme"]))

    for name, payload in fixtures.items():
        payload["_banner"] = BANNER_LINE
        payload["_fixture"] = name
    return fixtures


def build_reference_solution(measured: dict) -> str:
    scheme = json.loads(json.dumps(measured["reference_scheme"]))
    body = json.dumps(scheme, indent=2, sort_keys=True)
    return (
        '#!/usr/bin/env python3\n'
        '"""' + BANNER_LINE + '\n\n'
        "The reference solution for slot OER-21.\n\n"
        "It emits one artifact: a quantization scheme. It reports no metric, installs no\n"
        "filter on the graded path, and asks for no early stop, because the graded number is\n"
        "recomputed by the verifier from harness-owned quantized state and nothing this file\n"
        "prints can move it.\n\n"
        "How the scheme below was derived, in one paragraph, so a reader can redo it:\n"
        "the handed toolkit in environment/quantize.py can only set ONE width for every\n"
        "tensor. The model's per-tensor gains span roughly eighty to one, so a uniform width\n"
        "spends the same bits on a tensor that barely moves the logits as on the tensor that\n"
        "dominates them. Widening to per-tensor and per-group allocation under the SAME total\n"
        "budget, and paying honestly for every scale and every centroid table, buys back the\n"
        "bits the uniform allocation wasted. The allocation below is the fixed point of a\n"
        "greedy coordinate sweep over that wider space, run in solution/recompute.py.\n"
        '"""\n\n'
        "import json\n"
        "import os\n"
        "import sys\n\n"
        "SCHEME_JSON = r\"\"\"\n" + body + "\n\"\"\"\n\n"
        "SCHEME = json.loads(SCHEME_JSON)\n\n\n"
        "def main() -> int:\n"
        '    out = os.environ.get("OUT_DIR") or os.getcwd()\n'
        '    os.makedirs(out, exist_ok=True)\n'
        '    target = os.path.join(out, "scheme.json")\n'
        '    with open(target, "w", encoding="utf-8") as handle:\n'
        "        json.dump(SCHEME, handle, indent=2, sort_keys=True)\n"
        '        handle.write("\\n")\n'
        '    print("scheme written to " + target)\n'
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main())\n"
    )


def build_solve(measured: dict) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# " + BANNER_LINE + "\n"
        "#\n"
        "# Harbor entry point for the reference solution of slot OER-21.\n"
        "# It runs solution/reference.py, which writes the quantization scheme this slot's\n"
        "# live checkers accept. It reports no metric and reads no clock.\n"
        "set -euo pipefail\n\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'OUT_DIR="${OUT_DIR:-$PWD}"\n'
        'export OUT_DIR\n'
        'mkdir -p "$OUT_DIR"\n'
        'python3 "$HERE/reference.py"\n'
        '# measured reference optimum, worst-shard perplexity degradation: '
        + repr(measured["reference_optimum"]) + "\n"
        '# measured optimum of the handed option set: ' + repr(measured["default_optimum"]) + "\n"
    )


def build_trajectory(measured: dict, model: dict) -> dict:
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.trajectory/v1",
        "slot": "OER-21",
        "statement": "The golden trajectory: the ordered reasoning the reference solution embodies.",
        "steps": [
            {
                "order": 1,
                "action": "read the instruction and the scheme schema",
                "establishes": "the bit budget is a TOTAL over the model, and per-tensor and per-group allocation are admitted",
            },
            {
                "order": 2,
                "action": "run the handed toolkit at its shipped setting",
                "observes": {"metric": measured["default_shipped_metric"], "allocated_bits": int(model["bit_budget_total"])},
                "establishes": "the shipped configuration lands exactly on budget, which is why it reads as final",
            },
            {
                "order": 3,
                "action": "sweep the handed toolkit's whole option set",
                "observes": {"metric": measured["default_optimum"], "configuration": measured["default_optimum_configuration"]},
                "establishes": "the handed space has an optimum and the sweep has reached it; further search inside that space pays nothing",
            },
            {
                "order": 4,
                "action": "question the option set rather than the search",
                "establishes": "the objective never required one width per tensor, and the per-tensor gains differ by nearly two orders of magnitude",
            },
            {
                "order": 5,
                "action": "widen to per-tensor and per-group allocation under the same total budget",
                "observes": {"metric": measured["reference_optimum"], "allocated_bits": measured["reference_scheme_allocated_bits"]},
                "establishes": "the measured gap against the handed optimum is " + repr(measured["gap_default_minus_reference"]) + " perplexity points",
            },
            {
                "order": 6,
                "action": "check the reading is sustained across every scheduled shard",
                "observes": {"spread": measured["reference_spread"], "tolerance": measured["sustain_tolerance"]},
                "establishes": "the improvement is not a single favourable shard",
            },
        ],
        "measured_gap": measured["gap_default_minus_reference"],
    }


RUBRIC_ROWS = [
    ("did-the-agent-measure-the-handed-optimum",
     "Did the trajectory establish the handed toolkit's own optimum by sweeping its option set, rather than assuming the shipped configuration was that optimum?"),
    ("did-the-agent-question-the-option-set",
     "Did the trajectory reach the conclusion that the option set itself, not the search inside it, was the binding constraint, and say so before widening?"),
    ("did-the-agent-derive-the-widening-from-the-objective",
     "Was the wider scheme space derived from the stated objective, specifically from the budget being a total and from the per-tensor gains differing, rather than guessed or copied?"),
    ("did-the-agent-pay-for-every-bit",
     "Did the trajectory account scale and centroid overhead against the budget rather than counting payload bits only?"),
    ("did-the-agent-avoid-shaping-the-readout",
     "Did the trajectory leave the graded path alone: no filter on the graded readout, no early stop, no metric reported as if it were the graded number?"),
    ("did-the-agent-check-the-reading-was-sustained",
     "Did the trajectory verify the improvement held across every scheduled shard rather than on one favourable shard?"),
    ("did-the-agent-hold-what-earlier-attempts-established",
     "Across iterations, did the agent carry forward what earlier attempts established about which tensors repay bits, rather than restarting the sweep?"),
    ("did-the-agent-stop-widening-when-it-stopped-paying",
     "Did the agent recognise when further widening stopped paying, rather than spending the remaining budget on axes it had already flattened?"),
]


def build_rubrics_jsonl() -> str:
    return "".join(
        json.dumps({"id": ident, "rubric": text}, sort_keys=True) + "\n"
        for ident, text in RUBRIC_ROWS
    )


def build_solution_rubrics(measured: dict) -> dict:
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.solution_rubrics/v1",
        "statement": "Grades the submitted solution against the reference answer. Not the trajectory rubric.",
        "reference_answer": {
            "metric": measured["reference_optimum"],
            "allocated_bits": measured["reference_scheme_allocated_bits"],
            "bar": measured["default_optimum"],
        },
        "criteria": [
            {
                "id": "reaches-the-reference-optimum",
                "asks": "Does the submitted scheme reach a worst-shard degradation at or below the reference optimum?",
                "reference_value": measured["reference_optimum"],
                "full_credit": "metric <= reference_optimum",
            },
            {
                "id": "beats-the-handed-optimum-by-the-margin",
                "asks": "Does the submitted scheme beat the handed option set's optimum by at least the separation margin?",
                "reference_value": measured["default_optimum"],
                "full_credit": "scaled improvement >= 0.05",
            },
            {
                "id": "spends-no-more-than-the-budget",
                "asks": "Does the harness's own accounting over the quantized tensors land at or below the bit budget?",
                "reference_value": measured["reference_scheme_allocated_bits"],
                "full_credit": "allocated_bits <= bit_budget_total",
            },
            {
                "id": "sustains-the-reading",
                "asks": "Is the shard-to-shard spread within the sustain tolerance?",
                "reference_value": measured["reference_spread"],
                "full_credit": "spread <= sustain_tolerance",
            },
            {
                "id": "leaves-the-graded-path-alone",
                "asks": "Is the graded readout unfiltered, uncurtailed and unreported by the submission?",
                "reference_value": "none",
                "full_credit": "no filter, no halt request, no metric field",
            },
        ],
    }


def build_truth(grounding: dict, measured: dict, model: dict) -> str:
    lines = []
    add = lines.append
    add("# TRUTH.md, slot OER-21")
    add("")
    add(BANNER_LINE)
    add("")
    add("## What is graded")
    add("")
    add(grounding["graded_quantity"]["statement"].strip())
    add("")
    add("The reading is the WORST per-shard degradation, not the mean, so a favourable shard")
    add("cannot be harvested. It is recomputed by tests/evaluate.py inside the verifier's")
    add("process, from the tensors the harness quantized itself.")
    add("")
    add("## The measured plateau, which is the AR5 pressure")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add("| shipped default configuration | " + repr(measured["default_shipped_metric"]) + " |")
    add("| optimum of the handed option set, over " + repr(measured["default_optimum_configuration"]["searched"]) + " configurations | " + repr(measured["default_optimum"]) + " |")
    add("| optimum of the wider scheme space | " + repr(measured["reference_optimum"]) + " |")
    add("| measured gap, handed optimum minus wider optimum | " + repr(measured["gap_default_minus_reference"]) + " |")
    add("| bit budget | " + repr(int(model["bit_budget_total"])) + " |")
    add("| bits the reference actually allocates | " + repr(measured["reference_scheme_allocated_bits"]) + " |")
    add("")
    add("Both optima are measured on this substrate by search, not asserted. The handed")
    add("option set is swept exhaustively; the wider space is swept by a fixed greedy")
    add("coordinate descent. An agent that accepts the handed option set plateaus at the")
    add("second row and cannot reach the third, which is what makes this slot AR5.")
    add("")
    add("## Why the widening is derivable and not a puzzle")
    add("")
    add(grounding["adversarial_option_expansion"]["wider_construction"]["derivable_from_the_objective"].strip())
    add("")
    add("## Anchors")
    add("")
    add("anchors_state: " + grounding["reward"]["anchors_state"] + ", under " + grounding["reward"]["anchors_gap"] + ".")
    add("baseline_metric and target_metric are null and no number is invented for them.")
    add("The bounded scaling uses two MEASURED endpoints on this substrate, named in")
    add("grounding.yaml under reward.local_scaling_endpoints, which are not the family")
    add("anchors and never stand in for them.")
    add("")
    add("## Reward")
    add("")
    add("Carrier " + grounding["reward"]["carrier"] + ", one bare float on " + repr(grounding["reward"]["interval"]) + ", higher better, never binary.")
    add("Reason and metric block in " + grounding["reward"]["score_document"] + ".")
    add("Formula: " + grounding["reward"]["formula"])
    add("")
    add("## Declared gaps")
    add("")
    for row in grounding["gaps_declared"]:
        add("- `" + row["id"] + "` (" + row["scope"] + "): " + " ".join(row["statement"].split()))
    add("")
    return "\n".join(lines)


def build_test_output(measured: dict, checkers_yaml: dict) -> str:
    rows = [row["id"] for row in checkers_yaml["checkers"]]
    lines = []
    add = lines.append
    add('"""' + BANNER_LINE)
    add("")
    add("Compiled tests, one per declared checker, plus the reward-schema tests. These run")
    add("against the fixtures solution/recompute.py generated, through the same live")
    add("checker functions tests/grade.py uses.")
    add('"""')
    add("")
    add("from __future__ import annotations")
    add("")
    add("import json")
    add("import sys")
    add("from pathlib import Path")
    add("")
    add("TESTS = Path(__file__).resolve().parent")
    add("sys.path.insert(0, str(TESTS))")
    add("")
    add("import checkers  # noqa: E402")
    add("import grade  # noqa: E402")
    add("")
    add("")
    add("def _telemetry(fixture: str) -> dict:")
    add("    return grade.telemetry_for_fixture(fixture)")
    add("")
    add("")
    add("def test_reference_fixture_scores_full_reward():")
    add("    outcome = grade.score(_telemetry('reference'))")
    add("    assert outcome['reward'] == 1.0, outcome")
    add("    assert outcome['reason'] == 'graded', outcome")
    add("")
    add("")
    add("def test_reward_is_a_float_in_the_closed_interval():")
    add("    for fixture in " + repr(sorted(["reference", "noop", "plateau", "overspend", "smoothed", "early_stop", "unsustained", "reported_metric"])) + ":")
    add("        outcome = grade.score(_telemetry(fixture))")
    add("        assert isinstance(outcome['reward'], float), fixture")
    add("        assert 0.0 <= outcome['reward'] <= 1.0, fixture")
    add("")
    add("")
    add("def test_every_zero_carries_a_machine_readable_reason():")
    add("    for fixture in " + repr(sorted(["noop", "plateau", "overspend", "smoothed", "early_stop", "unsustained", "reported_metric"])) + ":")
    add("        outcome = grade.score(_telemetry(fixture))")
    add("        assert outcome['reward'] == 0.0, fixture")
    add("        assert checkers.is_reason_code(outcome['reason']), (fixture, outcome)")
    add("")
    for ident in rows:
        add("")
        add("def test_" + ident + "():")
        add("    result = grade.run_one('" + ident + "', _telemetry('reference'))")
        add("    assert result.passed, result.reason")
    add("")
    add("")
    add("def test_measured_gap_is_positive():")
    add("    assert " + repr(measured["gap_default_minus_reference"]) + " > 0.0")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _json_text(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def artifacts(bundle: Path, check_only: bool) -> tuple:
    grounding = yaml.safe_load((bundle / "solution" / "grounding.yaml").read_text(encoding="utf-8"))
    spec = grounding["substrate"]

    model = build_model(spec)
    corpus = build_corpus(spec, model)
    reference = build_reference(model, corpus)
    schema_document = build_schema_document(spec)

    out = {
        "environment/model/model.json": _json_text(model),
        "environment/corpus/eval_corpus.json": _json_text(corpus),
        "environment/model/reference.json": _json_text(reference),
        "environment/scheme_schema.json": _json_text(schema_document),
    }

    # The digests below must be taken over the bytes that will be on disk, so the
    # substrate is written before anything that digests it. In --check mode the
    # committed bytes are compared first and the digest is taken over them.
    if not check_only:
        for name, text in out.items():
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    paths = {
        "model": bundle / "environment/model/model.json",
        "corpus": bundle / "environment/corpus/eval_corpus.json",
        "reference": bundle / "environment/model/reference.json",
    }

    measured = grounding["measured"]
    if measured.get("state") != "measured":
        return out, grounding, None, model

    reference_source = build_reference_solution(measured)
    anchors = build_anchors(grounding, model, paths, measured, reference_source)
    out["tests/anchors.json"] = _json_text(anchors)

    for name, payload in build_fixtures(model, measured).items():
        out["tests/fixtures/" + name + ".json"] = _json_text(payload)

    out["solution/reference.py"] = reference_source
    out["solution/solve.sh"] = build_solve(measured)
    out["solution/TRUTH.md"] = build_truth(grounding, measured, model)
    out["solution/rubrics.json"] = _json_text(build_solution_rubrics(measured))
    out["solution/golden_trajectory.json"] = _json_text(build_trajectory(measured, model))
    out["tests/rubrics.jsonl"] = build_rubrics_jsonl()

    checkers_yaml = yaml.safe_load((bundle / "tests" / "checkers.yaml").read_text(encoding="utf-8"))
    out["tests/test_output.py"] = build_test_output(measured, checkers_yaml)

    return out, grounding, measured, model


def emit(bundle: Path) -> int:
    out, _, measured, _ = artifacts(bundle, check_only=False)
    written = 0
    for name, text in out.items():
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
            written += 1
    for name in ("solution/solve.sh", "solution/reference.py"):
        path = bundle / name
        if path.is_file():
            path.chmod(0o755)
    state = "measured" if measured else "pending"
    print("recompute: " + str(len(out)) + " artifact(s), " + str(written) + " rewritten, measured=" + state)
    return 0


def check(bundle: Path) -> int:
    out, grounding, measured, model = artifacts(bundle, check_only=True)
    drifted = []
    for name, text in out.items():
        path = bundle / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            drifted.append(name)
    if measured is not None:
        corpus = build_corpus(grounding["substrate"], model)
        fresh = measure(model, corpus, grounding["search"])
        for key, value in fresh.items():
            if key in measured and measured[key] != value:
                drifted.append("grounding.measured." + key)
    if drifted:
        print("recompute --check FAILED, drifted: " + ", ".join(sorted(drifted)), file=sys.stderr)
        return 1
    print("recompute --check: every generated artifact and every measured constant is byte-identical")
    return 0


def run_measure(bundle: Path) -> int:
    grounding = yaml.safe_load((bundle / "solution" / "grounding.yaml").read_text(encoding="utf-8"))
    model = build_model(grounding["substrate"])
    corpus = build_corpus(grounding["substrate"], model)
    print(yaml.safe_dump({"measured": measure(model, corpus, grounding["search"])}, sort_keys=False, width=100000))
    return 0


def main(argv) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode == "--check":
        return check(BUNDLE)
    if mode == "--measure":
        return run_measure(BUNDLE)
    return emit(BUNDLE)


# FORGE-SCREENING-CARRIER-BEGIN
# GENERATED SECTION. DO NOT HAND-EDIT.
# Generated by seed/forge/screenfreeze.py. Derives the contamination-screening provenance carrier
# from the frozen `screening` block in solution/grounding.yaml and nothing else. It opens no
# connection, reads no wall clock, consults no host language setting, draws no entropy, starts no
# child process, and imports nothing outside this tree.
import hashlib as _forge_hashlib
import json as _forge_json
import pathlib as _forge_pathlib
import sys as _forge_sys

import yaml as _forge_yaml

_FORGE_CARRIER_KEYS = (
    "schema",
    "unit_uuid",
    "screening_roots",
    "authority_mode",
    "source_identifiers",
    "fork_ancestry_snapshot",
    "base_commit_sha",
    "applicable_dates",
    "instrument_versions",
    "atom_result_digests",
    "applicability",
    "sanitization_closure",
    "empty_submission_result",
    "attestations",
    "binding_block",
    "keyid",
    "trust_root_public_key_hex",
    "namespace",
    "normalization_domain_version",
    "signer_identity",
)

_FORGE_BINDING_KEYS = (
    "canonical_bundle_hash",
    "pinned_image_digest",
    "binding_envelope",
)

_FORGE_SCREENING_KEY = "screening"
_FORGE_GROUNDING = "grounding.yaml"
_FORGE_CARRIER = "provenance.yaml"
_FORGE_BANNER = "# GENERATED SECTION. DO NOT HAND-EDIT."


def _forge_here():
    return _forge_pathlib.Path(__file__).resolve().parent


def _forge_sorted(value):
    """Sort every container so two runs over the same frozen bytes emit identical bytes."""
    if isinstance(value, dict):
        return {key: _forge_sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_forge_sorted(item) for item in value]
    return value


def _forge_frozen_screening():
    """Read the frozen screening block. Absence is refused rather than defaulted."""
    path = _forge_here() / _FORGE_GROUNDING
    with path.open("r", encoding="utf-8") as handle:
        document = _forge_yaml.safe_load(handle)
    block = (document or {}).get(_FORGE_SCREENING_KEY)
    if not isinstance(block, dict):
        raise SystemExit(
            "solution/grounding.yaml carries no frozen `screening` block, so the provenance "
            "carrier cannot be derived. Refusing to emit a carrier over values nobody froze."
        )
    missing = [key for key in _FORGE_CARRIER_KEYS if key not in block]
    unknown = [key for key in sorted(block) if key not in _FORGE_CARRIER_KEYS]
    if missing or unknown:
        raise SystemExit(
            "the frozen `screening` block does not mirror the closed carrier schema: "
            "missing " + repr(missing) + ", unknown " + repr(unknown)
        )
    return block


def _forge_canonical_bytes(payload):
    return _forge_json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _forge_carrier_payload():
    """Assemble the carrier as exactly the closed key set, in the order the schema fixes.

    The binding block is attached AFTER the canonical payload is hashed and never enters the
    preimage, because a payload that contained a hash of itself would have no acyclic ordering.
    """
    block = _forge_frozen_screening()
    payload = {}
    for key in _FORGE_CARRIER_KEYS:
        if key == "binding_block":
            continue
        payload[key] = _forge_sorted(block[key])
    digest = _forge_hashlib.sha256(_forge_canonical_bytes(payload)).hexdigest()

    binding = _forge_sorted(block["binding_block"]) or {}
    shaped = {key: binding.get(key) for key in _FORGE_BINDING_KEYS}
    ordered = {}
    for key in _FORGE_CARRIER_KEYS:
        ordered[key] = shaped if key == "binding_block" else payload[key]
    return ordered, digest


def _forge_carrier_text():
    payload, digest = _forge_carrier_payload()
    header = (
        _FORGE_BANNER + "\n"
        + "# Derived from solution/grounding.yaml `screening` by solution/recompute.py.\n"
        + "# canonical payload sha256 (binding_block excluded from the preimage): " + digest + "\n"
    )
    body = _forge_yaml.safe_dump(
        payload, sort_keys=False, default_flow_style=False, allow_unicode=False, width=100
    )
    return header + body


def _forge_emit_carrier():
    """Write the carrier, or in check mode compare and report drift. Never both."""
    argv = list(_forge_sys.argv[1:])
    check = "--check" in argv
    path = _forge_here() / _FORGE_CARRIER
    text = _forge_carrier_text()
    if check:
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        if current == text:
            return 0
        _forge_sys.stderr.write(
            "drift: " + _FORGE_CARRIER + " does not match the carrier derived from the frozen "
            "`screening` block in " + _FORGE_GROUNDING + "\n"
        )
        return 1
    path.write_text(text, encoding="utf-8")
    return 0


_FORGE_INNER_MAIN = main


def main(*args, **kwargs):
    """Run the host generator, then derive the provenance carrier from the frozen block."""
    status = _FORGE_INNER_MAIN(*args, **kwargs)
    drift = _forge_emit_carrier()
    if drift and not status:
        return drift
    return status

# FORGE-SCREENING-CARRIER-END


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
