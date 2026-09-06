# FORGE-CANARY-BEGIN
# 0: 99f5ad73a547e3ee5c8911663bce512215ee6de08e744e2c3d9171dbd2a87c3b
# 1: 0cd4a7c23f33cba5a5898f03b138ddd7291d01860db2b40e9658bc29cfc7e8d6
# 2: f4c692ef81702170b7ed1eb724041ff607afc5582885bfb55cbd71373bf97c27
# 3: a30c9e1a53adf7c86eed0dfec973af5033bcbbd66eb37c1b5a39936b24a2575c
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in slot OER-21 from solution/grounding.yaml alone.

One private source, one deriver. This script generates the checkpoint manifest, the
calibration manifest, the reference-reading statement, the scheme schema, the harness
anchors, the checker fixtures, the golden trajectory, solve.sh, TRUTH.md, rubrics.json,
tests/rubrics.jsonl and tests/test_output.py. Nothing it writes was typed by hand, so
nothing it writes can drift away from the source without this script saying so.

What it no longer does, and this is the whole of the re-base. The predecessor generated
the graded substrate itself: a Lehmer recurrence produced twelve tensors of forty-eight
weights over a vocabulary of eight, and a modular index formula stood in for a forward
pass. That is not a model and no declaration edit makes it one, so the generator was
deleted along with the substrate it generated. The architecture now comes from
environment/nanogpt_substrate.json, the parameters come from a real checkpoint the
verifier holds, and every quantity this script emits about the substrate is a function of
the declared architecture rather than of a recurrence written here.

It invokes no model, no network, no clock, no locale and no random source in its default
mode. `--measure` is the one mode that touches the accelerator, and it refuses rather than
inventing when the frozen checkpoint is not mounted.

Modes:
    (no argument)  regenerate every artifact in place
    --check        regenerate into memory and exit non-zero if any committed byte drifts
    --measure      re-run the searches against the mounted checkpoint and print the
                   `measured:` block, so the constants in grounding.yaml can be refreshed
                   by measurement rather than by editing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "tests"))

import yaml  # noqa: E402

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
BANNER_LINE = BANNER + " Source: " + SOURCE + ", derived by solution/recompute.py."

ROUND = 9

ROLES = (
    "token-embedding",
    "output-projection",
    "attention-query",
    "attention-key",
    "attention-value",
    "attention-output",
    "mlp-input",
    "mlp-output",
)

EMBEDDING_ROLES = ("token-embedding", "output-projection")
ATTENTION_ROLES = ("attention-query", "attention-key", "attention-value", "attention-output")
MLP_ROLES = ("mlp-input", "mlp-output")


def _model_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "oer21_nanogpt", BUNDLE / "environment" / "model" / "nanogpt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _substrate() -> dict:
    with (BUNDLE / "environment" / "nanogpt_substrate.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _grounding() -> dict:
    return yaml.safe_load((HERE / "grounding.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Quantities that are functions of the frozen architecture and nothing else
# ---------------------------------------------------------------------------


def derive(spec: dict, declaration: dict, nanogpt) -> dict:
    table = nanogpt.shard_table(declaration)
    quantizable = sum(row["numel"] for row in table)
    residue = nanogpt.residue_parameters(declaration)
    default_bits = int(spec["default_bits"])
    scale_bits = int(spec["scale_bits"])
    payload = quantizable * default_bits
    overhead = len(table) * scale_bits
    return {
        "table": table,
        "shards": len(table),
        "quantizable_parameters": quantizable,
        "residue_parameters": residue,
        "total_parameters": quantizable + residue,
        "default_bits": default_bits,
        "scale_bits": scale_bits,
        "default_payload_bits": payload,
        "default_overhead_bits": overhead,
        "bit_budget_total": payload + overhead,
    }


def _roles(rule: dict) -> dict:
    return rule["roles"]


def widening_rules(numbers: dict) -> list:
    """The verifier's fixed widening probe. Ordered, budget-checked at grading time.

    Every entry is a role-keyed allocation over the frozen shard roles. This list is the
    TARGET endpoint of the bounded scaling and is deliberately a bar rather than a claim
    about the optimum of the wider space, which nothing in this bundle has measured.
    """

    def rule(ident, embedding, attention, mlp, note):
        roles = {"default": dict(embedding)}
        for name in EMBEDDING_ROLES:
            roles[name] = dict(embedding)
        for name in ATTENTION_ROLES:
            roles[name] = dict(attention)
        for name in MLP_ROLES:
            roles[name] = dict(mlp)
        return {"id": ident, "note": note, "roles": roles}

    return [
        rule(
            "embedding-narrow-attention-wide",
            {"bits": 3, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 6, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 4, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            "spend the embedding's bits on the attention matrices",
        ),
        rule(
            "embedding-narrowest-attention-widest",
            {"bits": 2, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 8, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 4, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            "the same trade taken to the end of the admitted width range",
        ),
        rule(
            "embedding-narrowest-mlp-wide",
            {"bits": 2, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 5, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 6, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            "spend the embedding's bits on the MLP matrices instead",
        ),
        rule(
            "embedding-narrow-mlp-wide",
            {"bits": 3, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 4, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 5, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            "a milder version of the same trade",
        ),
        rule(
            "grouped-mixed",
            {"bits": 3, "codebook": "symmetric", "clip": 1.0, "groups": 8},
            {"bits": 5, "codebook": "symmetric", "clip": 1.0, "groups": 2},
            {"bits": 4, "codebook": "symmetric", "clip": 1.0, "groups": 4},
            "more scale groups per shard, each paid for out of the same budget",
        ),
        rule(
            "kmeans-embedding",
            {"bits": 2, "codebook": "kmeans", "clip": 1.0, "groups": 1},
            {"bits": 5, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 5, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            "a non-uniform codebook where its centroid table is cheap",
        ),
        rule(
            "clip-tuned",
            {"bits": 3, "codebook": "symmetric", "clip": 0.999, "groups": 1},
            {"bits": 5, "codebook": "symmetric", "clip": 0.99, "groups": 1},
            {"bits": 4, "codebook": "symmetric", "clip": 0.99, "groups": 1},
            "the same widths with the clip percentile moved off one",
        ),
        rule(
            "affine-blocks",
            {"bits": 3, "codebook": "symmetric", "clip": 1.0, "groups": 1},
            {"bits": 5, "codebook": "affine", "clip": 1.0, "groups": 1},
            {"bits": 4, "codebook": "affine", "clip": 1.0, "groups": 1},
            "an asymmetric codebook on the block matrices",
        ),
    ]


# ---------------------------------------------------------------------------
# Generated artifacts
# ---------------------------------------------------------------------------


def build_retirement_notice(grounding: dict) -> dict:
    """Overwrite the predecessor's weight table so no surrogate weight survives the re-base.

    environment/model/model.json used to hold twelve tensors of forty-eight weights over a
    vocabulary of eight. Leaving those bytes in the tree would leave a second, contradictory
    model in a bundle whose whole repair was that the first one was not a model, so the path
    is overwritten with this notice rather than left to be deleted by somebody else.
    """
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.retired/v1",
        "statement": "Retired by the OER-2026-08-20 re-base. This path no longer carries a model.",
        "carried_until_this_pass": "a formula-generated weight table of 12 tensors of 48 weights over a vocabulary of 8",
        "why_it_was_deleted": " ".join(grounding["rebase"]["what_moved"].split()),
        "what_replaced_it": {
            "architecture": "environment/nanogpt_substrate.json",
            "module": "environment/model/nanogpt.py",
            "parameters": "environment/model/checkpoint.pt",
            "shard_table": "environment/model/checkpoint.json",
        },
        "weights_here": None,
    }


def build_checkpoint_manifest(grounding: dict, numbers: dict, declaration: dict) -> dict:
    spec = grounding["substrate"]
    architecture = declaration["architecture"]
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.checkpoint/v1",
        "statement": "The frozen nanoGPT checkpoint this task quantizes, and the parameter shard table over it.",
        "architecture": {key: int(architecture[key]) for key in
                         ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len")},
        "architecture_source": "environment/nanogpt_substrate.json",
        "module": "environment/model/nanogpt.py",
        "checkpoint": {
            "role": spec["checkpoint"]["role"],
            "provisioning": " ".join(spec["checkpoint"]["provisioning"].split()),
            "container_path": "/work/environment/model/checkpoint.pt",
            "sha256": spec["checkpoint"]["sha256"],
            "sha256_state": spec["checkpoint"]["sha256_state"],
            "sha256_gap": spec["checkpoint"]["gap"],
            "verifier_copy": "the verifier grades from its OWN copy at the verifier-only mount, never from this one",
        },
        "shards": numbers["shards"],
        "quantizable_parameters": numbers["quantizable_parameters"],
        "residue_parameters": numbers["residue_parameters"],
        "residue_note": " ".join(spec["shard_definition"].split()),
        "total_parameters": numbers["total_parameters"],
        "scale_bits": numbers["scale_bits"],
        "default_bits": numbers["default_bits"],
        "bit_budget_total": numbers["bit_budget_total"],
        "bit_budget_derivation": " ".join(spec["budget_derivation"].split()),
        "shard_table": [
            {"name": row["name"], "role": row["role"], "shape": row["shape"], "numel": row["numel"]}
            for row in numbers["table"]
        ],
    }


def build_calibration_manifest(grounding: dict) -> dict:
    spec = grounding["substrate"]
    calibration = spec["calibration"]
    holdout = spec["holdout"]
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.corpus/v1",
        "statement": "The AGENT-VISIBLE calibration split. It is not the graded split and no reading taken on it reaches the grade.",
        "external_source": holdout["external_source"],
        "pipeline": holdout["pipeline"],
        "collection": holdout["corpus"],
        "calibration": {
            "role": " ".join(calibration["role"].split()),
            "shard": calibration["shard"],
            "token_offset": int(calibration["token_offset"]),
            "token_count": int(calibration["token_count"]),
            "sequence_length": int(calibration["sequence_length"]),
            "container_path": "/work/environment/corpus/calibration.bin",
            "staged_by": calibration["staged_by"],
        },
        "held_out_split": {
            "present_here": False,
            "statement": " ".join(holdout["absent_from_environment"].split()),
            "slice_count": int(holdout["slice_count"]),
            "shard": "withheld",
            "token_offsets": "withheld",
        },
    }


def build_reference_statement(grounding: dict) -> dict:
    spec = grounding["substrate"]
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.reference/v1",
        "statement": "Where the unquantized reference readings come from. They are not in this file and they are not in this container.",
        "graded_reference_readings": {
            "present_here": False,
            "computed_by": "tests/evaluate.py, in the verifier's process",
            "computed_from": "the verifier's own copy of the frozen checkpoint, on the verifier's own held-out FineWeb slices",
            "definition": " ".join(grounding["graded_quantity"]["definition"].split()),
        },
        "calibration_reference_readings": {
            "present_here": False,
            "computed_by": "environment/quantize.py measure, in the agent's own container",
            "computed_from": "the shipped checkpoint on the agent-visible calibration split",
            "statement": "Recomputed on demand rather than stored, so no stale reading can be mistaken for the graded one.",
        },
        "holdout_absent_reason": " ".join(spec["holdout"]["absent_from_environment"].split()),
    }


def build_schema_document(grounding: dict, numbers: dict) -> dict:
    import evaluate

    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.scheme_schema/v1",
        "statement": "The FULL quantization scheme space this task admits. Every axis below is free.",
        "artifact": "scheme.json, written into the directory named by OUT_DIR",
        "shape": {
            "scheme_version": 1,
            "tensors": [
                {
                    "name": "one entry per shard named in environment/model/checkpoint.json shard_table, all "
                            + str(numbers["shards"]) + " of them, each named exactly once",
                    "groups": [
                        {
                            "size": "int, group sizes within one shard must sum to that shard's numel",
                            "bits": list(evaluate.ALLOWED_BITS),
                            "codebook": list(evaluate.ALLOWED_CODEBOOKS),
                            "clip": list(evaluate.ALLOWED_CLIPS),
                        }
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
            "accounted_by": "the harness, over the shards as actually quantized",
            "bit_budget_total": numbers["bit_budget_total"],
            "scale_bits": numbers["scale_bits"],
            "residue_is_outside_the_budget": "biases and RMSNorm gains are not shards, are not quantized and are not charged",
        },
        "free_axes": [
            "the bit width of every group, independently",
            "the number of scale groups inside a shard, and where their boundaries fall",
            "the codebook family of every group",
            "the clip percentile of every group",
        ],
        "frozen_axes": [
            "the checkpoint parameters",
            "the architecture",
            "the total bit budget",
            "the held-out evaluation split",
            "the unquantized reference readings",
        ],
    }


def build_anchors(grounding: dict, numbers: dict, declaration: dict) -> dict:
    spec = grounding["substrate"]
    reward = grounding["reward"]
    holdout = spec["holdout"]
    architecture = declaration["architecture"]
    slices = [
        {
            "id": index,
            "shard": holdout["shard"],
            "token_offset": int(offset),
            "token_count": int(holdout["slice_tokens"]),
        }
        for index, offset in enumerate(holdout["slice_offsets"])
    ]
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.anchors/v1",
        "slot": "OER-21",
        "family": "F9",
        "substrate_declaration": spec["declaration"],
        "architecture": {key: int(architecture[key]) for key in
                         ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len")},
        "shards": numbers["shards"],
        "quantizable_parameters": numbers["quantizable_parameters"],
        "bit_budget_total": numbers["bit_budget_total"],
        "scale_bits": numbers["scale_bits"],
        "default_bits": numbers["default_bits"],
        "checkpoint": {
            "sha256": spec["checkpoint"]["sha256"],
            "sha256_state": spec["checkpoint"]["sha256_state"],
            "sha256_gap": spec["checkpoint"]["gap"],
            "source": "verifier-owned",
            "note": "The verifier digests the parameters it actually graded whether or not a pin is bound, so an absent pin is never an absent observation.",
        },
        "holdout": {
            "corpus": holdout["corpus"],
            "external_source": holdout["external_source"],
            "pipeline": holdout["pipeline"],
            "sequence_length": int(holdout["sequence_length"]),
            "slices": slices,
            "note": "Verifier-only. Absent from environment/ by construction and unreachable from the submission's process.",
        },
        "wider_probe": {
            "source": "verifier-widening-probe",
            "recomputed_at_grading_time": True,
            "note": "A bar, not a measured optimum of the wider space. Nothing in this bundle claims to have measured that optimum.",
            "rules": widening_rules(numbers),
        },
        "local_scaling": {
            "default_optimum_recomputed_at_grading_time": True,
            "wider_probe_recomputed_at_grading_time": True,
            "separation_margin": reward["separation_margin"],
            "sustain_tolerance_rule": " ".join(reward["sustain_tolerance_rule"].split()),
            "note": "measured endpoints of the bounded scaling, NOT the family anchors, and no endpoint is stored",
        },
        "anchors_state": reward["anchors_state"],
        "baseline_metric": reward["baseline_metric"],
        "target_metric": reward["target_metric"],
        "anchors_gap": reward["anchors_gap"],
        "anchors_absent_reason": " ".join(reward["anchors_absent_reason"].split()),
        "measured_state": grounding["measured"]["state"],
        "measured_gap": grounding["measured"]["gap"],
        "reference_solution_path": "solution/reference.py",
    }


def _uniform(table: list, bits: int, codebook: str, clip: float) -> dict:
    return {
        "scheme_version": 1,
        "tensors": [
            {"name": row["name"],
             "groups": [{"size": int(row["numel"]), "bits": bits, "codebook": codebook, "clip": clip}]}
            for row in table
        ],
        "graded_readout_filter": "none",
        "halt_after": None,
    }


def _from_rule(table: list, rule: dict) -> dict:
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
                    {"size": size, "bits": int(assignment["bits"]),
                     "codebook": assignment["codebook"], "clip": float(assignment.get("clip", 1.0))}
                    for _ in range(groups)
                ],
            }
        )
    return {"scheme_version": 1, "tensors": tensors, "graded_readout_filter": "none", "halt_after": None}


def build_fixtures(numbers: dict) -> dict:
    table = numbers["table"]
    rules = widening_rules(numbers)
    fixtures = {}

    reference = _from_rule(table, rules[0])
    fixtures["reference"] = json.loads(json.dumps(reference))

    fixtures["noop"] = _uniform(table, numbers["default_bits"], "symmetric", 1.0)
    fixtures["plateau"] = _uniform(table, 3, "affine", 1.0)

    overspend = json.loads(json.dumps(reference))
    overspend["tensors"][0]["groups"] = [
        {"size": int(table[0]["numel"]), "bits": 8, "codebook": "affine", "clip": 1.0}
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

    # A deliberately skewed allocation: the leading eighth of every shard is protected and
    # the rest is starved, so the parameter ranges different held-out slices exercise are
    # quantized at very different widths. Whether it clears the measured sustain tolerance
    # is unmeasured on this substrate under gap-oer21-endpoints-unmeasured-on-the-nanogpt-substrate.
    unsustained = {"scheme_version": 1, "tensors": [], "graded_readout_filter": "none", "halt_after": None}
    for row in table:
        numel = int(row["numel"])
        head = numel // 8
        unsustained["tensors"].append(
            {
                "name": row["name"],
                "groups": [
                    {"size": head, "bits": 8, "codebook": "affine", "clip": 1.0},
                    {"size": numel - head, "bits": 2, "codebook": "affine", "clip": 1.0},
                ],
            }
        )
    fixtures["unsustained"] = unsustained

    for name, payload in fixtures.items():
        payload["_banner"] = BANNER_LINE
        payload["_fixture"] = name
    return fixtures


def build_reference_solution(grounding: dict) -> str:
    search = grounding["search"]
    return (
        '#!/usr/bin/env python3\n'
        '"""' + BANNER_LINE + '\n\n'
        "The reference solution for slot OER-21.\n\n"
        "It emits one artifact: a quantization scheme over the frozen checkpoint's parameter\n"
        "shards. It reports no metric, installs no filter on the graded path, and asks for no\n"
        "early stop, because the graded number is recomputed by the verifier from its own\n"
        "parameters on held-out slices this file never sees, and nothing this file prints can\n"
        "move it.\n\n"
        "How the scheme is derived, so a reader can redo it. The handed toolkit in\n"
        "environment/quantize.py can only set ONE width for every shard. The embedding matrix\n"
        "and the untied output projection are 77266944 of the 162201600 quantizable parameters\n"
        "and they do not repay a bit at the same rate the attention and MLP matrices do, so a\n"
        "uniform width overspends on one and starves the other. This file measures that rate\n"
        "per shard ROLE on the agent-visible calibration split, then runs a fixed greedy pass\n"
        "that moves width from the roles that repay least to the roles that repay most while\n"
        "the harness accounting stays inside the same total budget. The allocation is COMPUTED\n"
        "here rather than frozen as a table, because a frozen table would be a number this\n"
        "bundle has not measured.\n"
        '"""\n\n'
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "ENVIRONMENT = Path(os.environ.get(\"OER21_ENVIRONMENT\") or \"/work/environment\")\n"
        "PASSES = " + repr(int(search["passes"])) + "\n\n\n"
        "def toolkit():\n"
        "    import importlib.util\n\n"
        "    spec = importlib.util.spec_from_file_location(\"oer21_quantize\", ENVIRONMENT / \"quantize.py\")\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n\n"
        "def main() -> int:\n"
        "    out = Path(os.environ.get(\"OUT_DIR\") or os.getcwd())\n"
        "    out.mkdir(parents=True, exist_ok=True)\n"
        "    kit = toolkit()\n"
        "    scheme = kit.widen(passes=PASSES)\n"
        "    target = out / \"scheme.json\"\n"
        "    with target.open(\"w\", encoding=\"utf-8\") as handle:\n"
        "        json.dump(scheme, handle, indent=2, sort_keys=True)\n"
        "        handle.write(\"\\n\")\n"
        "    print(\"scheme written to \" + str(target))\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main())\n"
    )


def build_solve() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# " + BANNER_LINE + "\n"
        "#\n"
        "# Harbor entry point for the reference solution of slot OER-21.\n"
        "# It runs solution/reference.py, which measures per-role sensitivity on the\n"
        "# agent-visible calibration split and writes the quantization scheme this slot's live\n"
        "# checkers accept. It reports no metric and reads no clock.\n"
        "#\n"
        "# No measured optimum is written into this file. Both endpoints of the bounded scaling\n"
        "# are recomputed by the verifier at grading time, so there is no number here to go stale.\n"
        "set -euo pipefail\n\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'OUT_DIR="${OUT_DIR:-$PWD}"\n'
        'export OUT_DIR\n'
        'mkdir -p "$OUT_DIR"\n'
        'python3 "$HERE/reference.py"\n'
    )


def build_trajectory(grounding: dict, numbers: dict) -> dict:
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.trajectory/v1",
        "slot": "OER-21",
        "statement": "The golden trajectory: the ordered reasoning the reference solution embodies.",
        "steps": [
            {
                "order": 1,
                "action": "read the instruction, the substrate declaration and the scheme schema",
                "establishes": "the architecture is frozen, the bit budget is a TOTAL over "
                               + str(numbers["quantizable_parameters"])
                               + " quantizable parameters in " + str(numbers["shards"])
                               + " shards, and per-shard and per-group allocation are admitted",
            },
            {
                "order": 2,
                "action": "run the handed toolkit at its shipped setting on the calibration split",
                "establishes": "the shipped configuration lands exactly on the budget, which is why it reads as final",
            },
            {
                "order": 3,
                "action": "sweep the handed toolkit's whole option set",
                "establishes": "the handed space has an optimum and the sweep has reached it; further search inside that space pays nothing",
            },
            {
                "order": 4,
                "action": "question the option set rather than the search",
                "establishes": "the objective never required one width per shard, and the embedding and output projection are 77266944 of the parameters at a different sensitivity to width than the block matrices",
            },
            {
                "order": 5,
                "action": "measure the per-role sensitivity and widen to per-shard and per-group allocation under the same total budget",
                "establishes": "bits move from the roles that repay least to the roles that repay most while the harness accounting stays inside the budget",
            },
            {
                "order": 6,
                "action": "check the reading is sustained across every scheduled held-out slice",
                "establishes": "the improvement is not a single favourable slice, and the solver cannot see those slices so it has to hold on the calibration split for structural reasons rather than by fitting",
            },
        ],
        "measured_gap": grounding["measured"]["state"],
    }


RUBRIC_ROWS = [
    ("did-the-agent-measure-the-handed-optimum",
     "Did the trajectory establish the handed toolkit's own optimum by sweeping its option set, rather than assuming the shipped configuration was that optimum?"),
    ("did-the-agent-question-the-option-set",
     "Did the trajectory reach the conclusion that the option set itself, not the search inside it, was the binding constraint, and say so before widening?"),
    ("did-the-agent-derive-the-widening-from-the-objective",
     "Was the wider scheme space derived from the stated objective, specifically from the budget being a total over the checkpoint and from the shard roles differing in sensitivity, rather than guessed or copied?"),
    ("did-the-agent-pay-for-every-bit",
     "Did the trajectory account scale and centroid overhead against the budget rather than counting payload bits only?"),
    ("did-the-agent-respect-the-held-out-split",
     "Did the trajectory treat the calibration split as a proxy and avoid fitting an allocation to it, given that the graded slices are held by the verifier and cannot be read?"),
    ("did-the-agent-avoid-shaping-the-readout",
     "Did the trajectory leave the graded path alone: no filter on the graded readout, no early stop, no metric reported as if it were the graded number?"),
    ("did-the-agent-check-the-reading-was-sustained",
     "Did the trajectory verify the improvement held across every calibration window rather than on one favourable window?"),
    ("did-the-agent-hold-what-earlier-attempts-established",
     "Across iterations, did the agent carry forward what earlier attempts established about which shard roles repay bits, rather than restarting the sweep?"),
    ("did-the-agent-stop-widening-when-it-stopped-paying",
     "Did the agent recognise when further widening stopped paying, rather than spending the remaining budget on axes it had already flattened?"),
]


def build_rubrics_jsonl() -> str:
    return "".join(
        json.dumps({"id": ident, "rubric": text}, sort_keys=True) + "\n"
        for ident, text in RUBRIC_ROWS
    )


def build_solution_rubrics(grounding: dict, numbers: dict) -> dict:
    return {
        "_banner": BANNER_LINE,
        "schema": "forge.oer21.solution_rubrics/v1",
        "statement": "Grades the submitted solution against the reference answer. Not the trajectory rubric.",
        "reference_answer": {
            "shape": "a per-shard, per-group allocation over the " + str(numbers["shards"]) + " frozen shards",
            "metric": grounding["measured"]["default_optimum"],
            "metric_state": grounding["measured"]["state"],
            "metric_gap": grounding["measured"]["gap"],
            "bit_budget_total": numbers["bit_budget_total"],
        },
        "criteria": [
            {
                "id": "beats-the-handed-optimum-by-the-margin",
                "asks": "Does the submitted scheme beat the handed option set's verifier-measured optimum by at least the separation margin?",
                "reference_value": grounding["reward"]["separation_margin"],
                "full_credit": "scaled improvement >= 0.05",
            },
            {
                "id": "reaches-the-widening-probe",
                "asks": "Does the submitted scheme reach a worst-slice degradation at or below the verifier's widening probe?",
                "reference_value": "recomputed at grading time",
                "full_credit": "metric <= wider_probe_optimum",
            },
            {
                "id": "spends-no-more-than-the-budget",
                "asks": "Does the harness's own accounting over the quantized shards land at or below the bit budget?",
                "reference_value": numbers["bit_budget_total"],
                "full_credit": "allocated_bits <= bit_budget_total",
            },
            {
                "id": "sustains-the-reading",
                "asks": "Is the slice-to-slice spread within the tolerance the verifier measured over the handed option space?",
                "reference_value": "recomputed at grading time",
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


def build_truth(grounding: dict, numbers: dict, declaration: dict) -> str:
    architecture = declaration["architecture"]
    reward = grounding["reward"]
    lines = []
    add = lines.append
    add("# TRUTH.md, slot OER-21")
    add("")
    add(BANNER_LINE)
    add("")
    add("## What this pass did, stated plainly")
    add("")
    add(" ".join(grounding["rebase"]["what_moved"].split()))
    add("")
    add("This was a replacement of the model and of the evaluation path. It was not a repin. A repin would have moved a declaration and left the graded arithmetic where it was, and there was no graded arithmetic here worth keeping: the predecessor's forward pass was an index formula over a table, so there was nothing to repoint at a nanoGPT run.")
    add("")
    add(" ".join(grounding["rebase"]["what_survived"].split()))
    add("")
    add("## What is graded")
    add("")
    add(" ".join(grounding["graded_quantity"]["statement"].split()))
    add("")
    add(" ".join(grounding["graded_quantity"]["definition"].split()))
    add("")
    add("The reading is the WORST per-slice degradation, not the mean, so a favourable slice cannot be harvested. It is recomputed by tests/evaluate.py inside the verifier's process, from the parameters the harness quantized itself, on slices the solver cannot reach.")
    add("")
    add("## The substrate")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add("| architecture | " + str(int(architecture["num_layers"])) + " layers, model_dim "
        + str(int(architecture["model_dim"])) + ", head_dim " + str(int(architecture["head_dim"]))
        + ", num_heads " + str(int(architecture["num_heads"])) + ", vocab " + str(int(architecture["vocab_size"])) + " |")
    add("| quantizable parameter shards | " + str(numbers["shards"]) + " |")
    add("| quantizable parameters | " + str(numbers["quantizable_parameters"]) + " |")
    add("| residue held at checkpoint width, outside the budget | " + str(numbers["residue_parameters"]) + " |")
    add("| bit budget | " + str(numbers["bit_budget_total"]) + " |")
    add("| held-out slices, verifier-owned | " + str(int(grounding["substrate"]["holdout"]["slice_count"]))
        + " of " + str(int(grounding["substrate"]["holdout"]["slice_tokens"])) + " tokens |")
    add("")
    add(" ".join(grounding["substrate"]["budget_derivation"].split()))
    add("")
    add("## The AR5 pressure")
    add("")
    add(" ".join(grounding["adversarial_option_expansion"]["wider_construction"]["why_it_pays_on_this_substrate"].split()))
    add("")
    add("Both endpoints of the bounded scaling are measured by the verifier at grading time and neither is stored. The handed option set is swept exhaustively and the wider space is probed by a fixed, ordered, budget-checked list of role-keyed allocations. The probe is a bar and it is never called an optimum of the wider space, because nothing in this bundle has measured that optimum.")
    add("")
    add("## Why the widening is derivable and not a puzzle")
    add("")
    add(" ".join(grounding["adversarial_option_expansion"]["wider_construction"]["derivable_from_the_objective"].split()))
    add("")
    add("## Anchors")
    add("")
    add("anchors_state: " + reward["anchors_state"] + ", under " + reward["anchors_gap"] + ". baseline_metric and target_metric are null and no number is invented for them. The bounded scaling uses two endpoints the verifier measures at grading time, named in grounding.yaml under reward.local_scaling_endpoints, which are not the family anchors and never stand in for them.")
    add("")
    add("## What is not measured")
    add("")
    add(" ".join(grounding["measured"]["reason"].split()))
    add("")
    add("## Reward")
    add("")
    add("Carrier " + reward["carrier"] + ", one bare float on " + repr(reward["interval"]) + ", higher better, never binary.")
    add("Reason and metric block in " + reward["score_document"] + ".")
    add("Formula: " + reward["formula"])
    add("")
    add("## Declared gaps")
    add("")
    for row in grounding["gaps_declared"]:
        add("- `" + row["id"] + "` (" + row["scope"] + "): " + " ".join(row["statement"].split()))
    add("")
    add("## Retired gaps")
    add("")
    for row in grounding["retired_gaps"]:
        add("- `" + row["id"] + "` retired by " + row["retired_by"] + ": " + " ".join(row["statement"].split()))
    add("")
    return "\n".join(lines)


def build_test_output(checkers_yaml: dict) -> str:
    rows = [row["id"] for row in checkers_yaml["checkers"]]
    lines = []
    add = lines.append
    add('"""' + BANNER_LINE)
    add("")
    add("Compiled tests, one per declared checker, plus the reward-schema tests. These run")
    add("against the fixtures solution/recompute.py generated, through the same live checker")
    add("functions tests/grade.py uses.")
    add("")
    add("Every test that needs a reading is guarded by `substrate_available`, because the")
    add("graded path here loads a real checkpoint and runs real forward passes. When the")
    add("verifier mount is absent these tests SKIP with a named reason rather than passing")
    add("vacuously, and tests/grade.py refuses the run outright with checkpoint-absent, so an")
    add("absent substrate can never read as a graded zero.")
    add('"""')
    add("")
    add("from __future__ import annotations")
    add("")
    add("import sys")
    add("from pathlib import Path")
    add("")
    add("import pytest")
    add("")
    add("TESTS = Path(__file__).resolve().parent")
    add("sys.path.insert(0, str(TESTS))")
    add("")
    add("import checkers  # noqa: E402")
    add("import evaluate  # noqa: E402")
    add("import grade  # noqa: E402")
    add("")
    add("FIXTURES = " + repr(sorted(["reference", "noop", "plateau", "overspend", "smoothed",
                                     "early_stop", "unsustained", "reported_metric"])))
    add("ZERO_FIXTURES = " + repr(sorted(["noop", "plateau", "overspend", "smoothed",
                                          "early_stop", "unsustained", "reported_metric"])))
    add("")
    add("SKIP = 'the verifier substrate is not mounted, so no forward pass can be run here'")
    add("")
    add("")
    add("def substrate_available() -> bool:")
    add("    return evaluate.checkpoint_path().is_file() and evaluate.holdout_path().is_dir()")
    add("")
    add("")
    add("def _telemetry(fixture: str) -> dict:")
    add("    if not substrate_available():")
    add("        pytest.skip(SKIP)")
    add("    return grade.telemetry_for_fixture(fixture)")
    add("")
    add("")
    add("def test_every_fixture_is_a_well_formed_allocation():")
    add("    table = grade.anchors()['shards']")
    add("    for fixture in FIXTURES:")
    add("        scheme = evaluate.load_json(TESTS / 'fixtures' / (fixture + '.json'))")
    add("        assert len(scheme['tensors']) == table, fixture")
    add("        assert len({entry['name'] for entry in scheme['tensors']}) == table, fixture")
    add("")
    add("")
    add("def test_the_holdout_split_is_absent_from_the_environment():")
    add("    root = grade.environment_paths()['substrate'].parent")
    add("    for path in root.rglob('*'):")
    add("        assert 'fineweb_val' not in path.name, path")
    add("")
    add("")
    add("def test_reference_fixture_clears_the_gate_chain():")
    add("    outcome = grade.score(_telemetry('reference'))")
    add("    assert outcome['reason'] == 'graded', outcome")
    add("    assert outcome['reward'] > 0.0, outcome")
    add("")
    add("")
    add("def test_reward_is_a_float_in_the_closed_interval():")
    add("    for fixture in FIXTURES:")
    add("        outcome = grade.score(_telemetry(fixture))")
    add("        assert isinstance(outcome['reward'], float), fixture")
    add("        assert 0.0 <= outcome['reward'] <= 1.0, fixture")
    add("")
    add("")
    add("def test_every_zero_carries_a_machine_readable_reason():")
    add("    for fixture in ZERO_FIXTURES:")
    add("        outcome = grade.score(_telemetry(fixture))")
    add("        if outcome['reward'] == 0.0:")
    add("            assert checkers.is_reason_code(outcome['reason']), (fixture, outcome)")
    add("")
    for ident in rows:
        add("")
        add("def test_" + ident + "():")
        add("    result = grade.run_one('" + ident + "', _telemetry('reference'))")
        add("    assert result.passed, result.reason")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _json_text(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def artifacts(bundle: Path) -> tuple:
    grounding = _grounding()
    declaration = _substrate()
    nanogpt = _model_module()
    numbers = derive(grounding["substrate"], declaration, nanogpt)

    out = {
        "environment/model/model.json": _json_text(build_retirement_notice(grounding)),
        "environment/model/checkpoint.json": _json_text(build_checkpoint_manifest(grounding, numbers, declaration)),
        "environment/corpus/eval_corpus.json": _json_text(build_calibration_manifest(grounding)),
        "environment/model/reference.json": _json_text(build_reference_statement(grounding)),
        "environment/scheme_schema.json": _json_text(build_schema_document(grounding, numbers)),
        "tests/anchors.json": _json_text(build_anchors(grounding, numbers, declaration)),
        "solution/reference.py": build_reference_solution(grounding),
        "solution/solve.sh": build_solve(),
        "solution/TRUTH.md": build_truth(grounding, numbers, declaration),
        "solution/rubrics.json": _json_text(build_solution_rubrics(grounding, numbers)),
        "solution/golden_trajectory.json": _json_text(build_trajectory(grounding, numbers)),
        "tests/rubrics.jsonl": build_rubrics_jsonl(),
    }
    for name, payload in build_fixtures(numbers).items():
        out["tests/fixtures/" + name + ".json"] = _json_text(payload)

    checkers_yaml = yaml.safe_load((bundle / "tests" / "checkers.yaml").read_text(encoding="utf-8"))
    out["tests/test_output.py"] = build_test_output(checkers_yaml)

    return out, grounding, numbers


def emit(bundle: Path) -> int:
    out, _, numbers = artifacts(bundle)
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
    print(
        "recompute: " + str(len(out)) + " artifact(s), " + str(written) + " rewritten, "
        + str(numbers["shards"]) + " shards over " + str(numbers["quantizable_parameters"]) + " parameters"
    )
    return 0


def check(bundle: Path) -> int:
    out, grounding, _ = artifacts(bundle)
    drifted = []
    for name, text in out.items():
        path = bundle / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            drifted.append(name)
    if grounding["measured"].get("state") == "measured":
        drifted.extend(measure_drift(bundle, grounding))
    if drifted:
        print("recompute --check FAILED, drifted: " + ", ".join(sorted(drifted)), file=sys.stderr)
        return 1
    print("recompute --check: every generated artifact is byte-identical")
    return 0


def measure_drift(bundle: Path, grounding: dict) -> list:
    fresh = run_measurement(bundle)
    return [
        "grounding.measured." + key
        for key, value in fresh.items()
        if key in grounding["measured"] and grounding["measured"][key] != value
    ]


def run_measurement(bundle: Path) -> dict:
    """Re-run the searches against the mounted checkpoint. Refuses when it is absent."""
    import evaluate

    grounding = _grounding()
    paths = {
        "substrate": bundle / "environment" / "nanogpt_substrate.json",
        "checkpoint": bundle / "environment" / "model" / "checkpoint.json",
        "architecture": bundle / "environment" / "model" / "nanogpt.py",
    }
    anchors = evaluate.load_json(bundle / "tests" / "anchors.json")
    if not evaluate.checkpoint_path().is_file() or not evaluate.holdout_path().is_dir():
        raise SystemExit(
            "refusing to measure: the frozen checkpoint or the held-out split is not mounted at "
            + str(evaluate.VERIFIER_ROOT)
            + ". No constant is invented to stand in for a measurement that did not happen."
        )
    context = evaluate.build_context(paths, anchors)
    shipped = evaluate.default_scheme(context["table"], context["default_bits"])
    shipped_metric = evaluate.scheme_metric(context, shipped)
    optimum = evaluate.default_optimum(context)
    probe = evaluate.wider_probe_optimum(context, anchors, optimum)
    tolerance = None
    if optimum.get("widest_spread") is not None:
        tolerance = round(1.5 * float(optimum["widest_spread"]) + 0.01, ROUND)
    gap = None
    if optimum["metric"] is not None and probe["metric"] is not None:
        gap = round(float(optimum["metric"]) - float(probe["metric"]), ROUND)
    return {
        "state": "measured",
        "gap": grounding["measured"]["gap"],
        "reason": grounding["measured"]["reason"],
        "default_shipped_metric": shipped_metric,
        "default_optimum": optimum["metric"],
        "wider_probe_optimum": probe["metric"],
        "gap_default_minus_probe": gap,
        "reference_scheme_allocated_bits": None,
        "sustain_tolerance": tolerance,
    }


def run_measure(bundle: Path) -> int:
    print(yaml.safe_dump({"measured": run_measurement(bundle)}, sort_keys=False, width=100000))
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
