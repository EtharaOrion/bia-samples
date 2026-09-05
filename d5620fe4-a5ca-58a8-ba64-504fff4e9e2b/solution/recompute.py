# FORGE-CANARY-BEGIN
# 0: 55c63d4b4db61a2480ef718164544e2e9995c58309d70734046cf2a5094b7281
# 1: ba492ae559fc77f7739168f6c0f7bd3f6623937ce41eca7db81b17ff1ae93bec
# 2: e03d8991ee319ab51d4bf326c3423c1564d471205b2868d8875eb9996b68b7f4
# 3: 036c72cfca1b9f02dcc3a1a29aa48f1b5cdcc374bdb1ab90750b37da3922ec7a
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of slot OER-14 from solution/grounding.yaml alone.

This module reads ONE file, `solution/grounding.yaml`, and writes:

    environment/corpus/FROZEN.json      the FineWeb10B corpus declaration: which shards this
                                        slot measures over, which loader stages them, and
                                        where the held-out split is NOT
    environment/corpus/train.txt        a superseded marker, because the graded corpus is no
                                        longer a text file this bundle ships
    environment/corpus/eval.txt         a superseded marker, for the same reason
    tests/controls.json                 the harness control table, which is also the only
                                        place the held-out slice is pinned
    tests/test_output.py                one compiled test per declared checker
    solution/fixtures/plan.json         the checker fixture plan and the negative controls
    solution/solve.sh                   the reference entry point
    solution/TRUTH.md                   what is actually graded, in plain words
    solution/rubrics.json               the solution-against-reference rubric

It invokes NO model, NO network, NO clock, NO locale and NO random source. Every value it
writes is read out of grounding.yaml, so the bytes are a pure function of that file. Running
this twice over frozen bytes produces byte-identical output, and `--check` proves it without
writing.

It emits NO corpus bytes. Before the nanoGPT re-base this module generated a synthetic
combinatorial corpus and the graded reading was taken over it. The corpus is now FineWeb10B,
staged by the canonical loader at image build time, and this module pins it rather than
producing it.

Usage:
    python3 solution/recompute.py            # write
    python3 solution/recompute.py --check    # exit 1 if any committed byte would move
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
BUNDLE = Path(__file__).resolve().parents[1]


def load_source() -> dict:
    with (BUNDLE / "solution" / "grounding.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# Generated artifacts.
# ---------------------------------------------------------------------------


def _json(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def corpus_declaration(source: dict) -> str:
    """The agent-visible corpus declaration. It names the shards and NOT the held-out slice.

    Every field here is safe on the agent surface: the training shards, the loader, and the
    frozen vocabulary construction window a submission builds its tokens from. The validation
    shard, the held-out token offset, the held-out token count and the slice digest are held
    in tests/controls.json, which no agent-visible image copies.
    """
    block = source["corpus"]
    return _json(
        {
            "banner": BANNER,
            "source": SOURCE,
            "collection": block["source"],
            "external_source": block["external_source"],
            "pipeline": block["pipeline"],
            "train_glob": block["train_glob"],
            "shard_root": block["agent_shard_root"],
            "train_shards_staged": block["train_shards_staged"],
            "header_bytes_skipped": block["header_bytes_skipped"],
            "train_text_gpt2_tokens": block["train_text_gpt2_tokens"],
            "vocabulary_construction_window": {
                "shard": block["vocab_window_shard"],
                "token_offset": block["vocab_window_token_offset"],
                "token_count": block["vocab_window_token_count"],
            },
            "held_out_split": "absent from this environment by construction",
            "held_out_note": (
                "The number that scores a submission is computed by the verifier on a slice "
                "of a FineWeb validation shard that this environment does not carry and "
                "cannot reach. The shard, the token offset and the token count are constants "
                "of the verifier's own control table, so no call made here resolves them."
            ),
            "denominator_note": (
                "The graded denominator is the UTF-8 byte length of that held-out text and "
                "nothing else, and no construction may divide by any other number."
            ),
        }
    )


def superseded_marker(source: dict, role: str) -> str:
    """The one-line marker that replaces a corpus text file this bundle no longer ships."""
    block = source["corpus"]
    where = block["train_glob"] if role == "training" else "the verifier image alone"
    return (
        "# " + BANNER + " source: " + SOURCE + "\n"
        + "# SUPERSEDED by the nanoGPT re-base. This slot no longer ships its "
        + role + " corpus as text.\n"
        + "# The " + role + " corpus is FineWeb10B, staged by " + block["pipeline"]
        + ", and lives at " + where + ".\n"
    )


def controls_document(source: dict) -> str:
    """The VERIFIER-ONLY control table.

    `corpus_construction` carries this slot's two discovery values, `offset` and `stride`,
    which are the phase and the period of the held-out FineWeb slice, together with the slice
    digest that establishes them against the staged shard. They are readable here and on no
    agent-visible surface, and the slice they pin is materialised only in the verifier image.
    """
    block = source["corpus"]
    return _json(
        {
            "banner": BANNER,
            "source": SOURCE,
            "corpus_construction": {
                "shard": block["shard"],
                "offset": block["offset"],
                "stride": block["stride"],
                "slice_sha256": block["slice_sha256"],
                "shard_sha256_prefix_1mib": block["shard_sha256_prefix_1mib"],
                "val_glob": block["val_glob"],
                "shard_root": block["verifier_shard_root"],
            },
            "train_text_gpt2_tokens": block["train_text_gpt2_tokens"],
            "single_direction_sweep": source["controls"]["single_direction_sweep"],
            "reference_allocation": source["controls"]["reference_allocation"],
            "verifier_training_runs": source["controls"]["verifier_training_runs"],
            "flatten_epsilon_bits": source["golden_trajectory"]["flatten_epsilon_bits"],
            "flatten_run_required": source["golden_trajectory"]["flatten_run_required"],
            "separation_margin": source["bindings"]["separation_margin"],
            "drift_state_final": source["drift_state"]["final"],
            "anchors_state": source["anchors"]["anchors_state"],
            "anchors_gap": source["anchors"]["gap"],
            "anchor_resolution": source["anchors"]["resolution_mechanism"],
        }
    )


def fixture_plan(source: dict) -> str:
    return _json(
        {
            "banner": BANNER,
            "source": SOURCE,
            "golden_trajectory": source["golden_trajectory"],
            "checkers": source["checkers"],
            "negative_controls": source["negative_controls"],
            "statement_ambiguity": source["statement_ambiguity"],
            "drift_state": source["drift_state"],
        }
    )


def solve_sh(source: dict) -> str:
    allocation = source["controls"]["reference_allocation"]
    ordered = ", ".join(
        '"%s": %d' % (name, int(allocation.get(name, 0)))
        for name in source["substrate"]["directions"]
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# " + BANNER,
            "# source: " + SOURCE,
            "#",
            "# The reference solution entry point. It writes ONE document: the vocabulary",
            "# construction plan. It reports no number, because every number on the graded",
            "# path is the harness's own, produced by training the frozen decoder itself.",
            "set -euo pipefail",
            "",
            'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'OUT="${SUBMISSION_PATH:-${HERE}/submission.json}"',
            "",
            'python3 "${HERE}/reference.py" --emit "${OUT}"',
            "",
            "# The allocation this reference reaches, for a reader who wants it without",
            "# running anything: {" + ordered + "}",
            'echo "wrote ${OUT}"',
            "",
        ]
    )


def truth_md(source: dict) -> str:
    truth = source["truth"]
    rows = [
        "# TRUTH.md, slot OER-14",
        "",
        BANNER,
        "",
        "source: `" + SOURCE + "`",
        "",
        "## " + truth["headline"],
        "",
    ]
    for point in truth["points"]:
        rows.append("- " + point)
    rows.extend(
        [
            "",
            "## The substrate",
            "",
            source["rebase"]["kind_note"].strip(),
            "",
            source["rebase"]["vocab_size_interaction"].strip(),
            "",
            "## Anchors",
            "",
            "`anchors_state` is **" + source["anchors"]["anchors_state"] + "** under gap "
            "`" + source["anchors"]["gap"] + "`. `baseline_metric` and `target_metric` are "
            "null in `task.toml` and no number is invented for them here.",
            "",
            source["anchors"]["resolution_mechanism_detail"].strip(),
            "",
            "## The reference allocation",
            "",
            "```json",
            json.dumps(source["controls"]["reference_allocation"], indent=2, sort_keys=True),
            "```",
            "",
            "## Reward",
            "",
            "`" + source["reward_schema"]["formula"] + "`, then `"
            + source["reward_schema"]["clip"] + "`. The carrier is `"
            + source["reward_schema"]["carrier"] + "`, a bare float, and the reason and metric "
            "block travel in `" + source["reward_schema"]["score_document"] + "`.",
            "",
        ]
    )
    return "\n".join(rows)


def rubrics_json(source: dict) -> str:
    return _json(
        {
            "banner": BANNER,
            "source": SOURCE,
            "judged": "the solution against its reference answer",
            "not_a": (
                "This file never substitutes for tests/rubrics.jsonl, which judges the "
                "trajectory and is hand-authored."
            ),
            "criteria": source["rubrics"]["solution_criteria"],
            "reference_allocation": source["controls"]["reference_allocation"],
        }
    )


def test_output_py(source: dict) -> str:
    rows = [
        '"""Compiled checker suite for slot OER-14.',
        "",
        BANNER,
        "",
        "source: " + SOURCE,
        "",
        "Each test drives the REAL checker in tests/checkers.py over harness telemetry the",
        "reference produced, and its planted-defect twin. Both halves of every checker.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "TESTS = Path(__file__).resolve().parent",
        "sys.path.insert(0, str(TESTS))",
        "",
        "import checkers  # noqa: E402",
        "import fixtures  # noqa: E402",
        "",
        "CLEAN = fixtures.clean_context()",
        "",
        "",
        "def test_suite_collected_at_least_one_case():",
        '    """A zero-collected suite is never positive evidence that the suite ran."""',
        "    assert len(checkers.REGISTRY) > 0",
        "",
    ]
    for row in source["checkers"]:
        ident = row["id"]
        rows.extend(
            [
                "",
                "def test_" + ident + "():",
                '    outcome = checkers.' + ident + "(CLEAN)",
                "    assert outcome.passed, outcome.detail",
                "    planted = fixtures.planted_context(" + repr(ident) + ")",
                "    refused = checkers." + ident + "(planted)",
                "    assert not refused.passed",
                "    assert refused.reason == " + repr(row["zero_reason"]),
                "",
            ]
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Emission.
# ---------------------------------------------------------------------------


def artifacts() -> dict:
    source = load_source()
    return {
        "environment/corpus/FROZEN.json": corpus_declaration(source).encode(),
        "environment/corpus/train.txt": superseded_marker(source, "training").encode(),
        "environment/corpus/eval.txt": superseded_marker(source, "evaluation").encode(),
        "tests/controls.json": controls_document(source).encode(),
        "tests/test_output.py": test_output_py(source).encode(),
        "solution/fixtures/plan.json": fixture_plan(source).encode(),
        "solution/solve.sh": solve_sh(source).encode(),
        "solution/TRUTH.md": truth_md(source).encode(),
        "solution/rubrics.json": rubrics_json(source).encode(),
    }


def main(argv) -> int:
    check_only = "--check" in argv
    drifted = []
    for relative, payload in sorted(artifacts().items()):
        target = BUNDLE / relative
        current = target.read_bytes() if target.is_file() else None
        if current == payload:
            continue
        drifted.append(relative)
        if check_only:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if relative.endswith(".sh"):
            target.chmod(0o755)
    if check_only:
        if drifted:
            print("DRIFTED: " + ", ".join(drifted))
            return 1
        print("clean: every generated artifact matches solution/grounding.yaml")
        return 0
    print("wrote " + str(len(drifted)) + " artifact(s) that moved; source " + SOURCE)
    return 0


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
    raise SystemExit(main(sys.argv[1:]))
