# FORGE-CANARY-BEGIN
# 0: 2cd2371b0f2acfd03f1252fc050c7fc6184812329f0781f6aaf754e8b76cd94b
# 1: d4730857740e9f0c889b7b6acd48b30cb549d93332e78e5be290f0160b9cc88f
# 2: b147a1ca04ea54c4fa73809fc3bc683a323bd7d90fad4f0262bc91c56b23add2
# 3: 25e08180adcf94911d6bb4f56fa313e9a1d41777694aa483333b40a56b4dca46
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of slot OER-14 from solution/grounding.yaml alone.

This module reads ONE file, `solution/grounding.yaml`, and writes:

    environment/corpus/train.txt        the frozen training corpus
    environment/corpus/eval.txt         the frozen evaluation corpus, whose byte length is
                                        the graded denominator and nothing else
    environment/corpus/FROZEN.json      byte counts and digests of both, with the banner
    tests/controls.json                 the harness control table and the checker fixture plan
    tests/test_output.py                one compiled test per declared checker
    solution/fixtures/plan.json         the checker fixture plan and the negative controls
    solution/solve.sh                   the reference entry point
    solution/TRUTH.md                   what is actually graded, in plain words
    solution/rubrics.json               the solution-against-reference rubric

It invokes NO model, NO network, NO clock, NO locale and NO random source. Every choice in
the corpus generator is fixed index arithmetic over a fixed table, so the bytes are a pure
function of the parameters recorded in grounding.yaml. Running this twice over frozen bytes
produces byte-identical output, and `--check` proves it without writing.

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
# The frozen corpus. Deterministic and combinatorial: no PRNG of any kind.
# ---------------------------------------------------------------------------


def _tables(block: dict) -> dict:
    words = sorted({a + b for a in block["word_prefixes"] for b in block["word_suffixes"]})
    words = words[: int(block["word_limit"])]
    codes = [
        "%08d"
        % (
            ((index * int(block["code_stride"]) + int(block["code_offset"]))
             * int(block["code_multiplier"])) % 100000000
        )
        for index in range(int(block["code_count"]))
    ]
    return {
        "words": words,
        "idents": ["f_" + word[:6] for word in words[: int(block["ident_limit"])]],
        "fields": list(block["fields"]),
        "operators": list(block["operators"]),
        "tags": list(block["tags"]),
        "codes": codes,
    }


def _pick(table: list, index: int, block: dict) -> str:
    return table[(index * int(block["stride"]) + int(block["offset"])) % len(table)]


def _line(index: int, tables: dict, block: dict) -> str:
    words, idents = tables["words"], tables["idents"]
    fields, operators = tables["fields"], tables["operators"]
    tags, codes = tables["tags"], tables["codes"]
    kind = index % 6
    if kind < 2:
        return " ".join(_pick(words, index * 3 + slot * 17, block) for slot in range(8)) + "."
    if kind == 2:
        parts = [
            _pick(fields, index + slot, block) + "=" + _pick(codes, index * 5 + slot * 13, block)
            for slot in range(4)
        ]
        return "{" + "; ".join(parts) + "}"
    if kind == 3:
        return (
            "def " + _pick(idents, index, block) + "(x, y):  return x "
            + _pick(operators, index, block) + " y " + _pick(operators, index + 3, block)
            + " " + _pick(words, index * 7 + 5, block) + "  # " + _pick(tags, index, block)
        )
    if kind == 4:
        return (
            "[" + _pick(codes, index, block) + "] " + _pick(words, index * 11 + 1, block)
            + " " + _pick(operators, index + 2, block) + " "
            + _pick(codes, index * 2 + 1, block) + " :: " + _pick(fields, index, block)
            + "=" + _pick(codes, index * 5, block) + ";"
        )
    return (
        _pick(words, index * 13, block) + " " + _pick(words, index * 5 + 7, block)
        + " <" + _pick(codes, index + 3, block) + "> " + _pick(operators, index + 1, block)
        + " " + _pick(words, index * 3 + 11, block) + " " + _pick(tags, index + 2, block) + ";"
    )


def corpus_bytes(block: dict, start: int, count: int) -> bytes:
    tables = _tables(block)
    rows = [block["banner"]]
    rows.extend(_line(index, tables, block) for index in range(start, start + count))
    return ("\n".join(rows) + "\n").encode("ascii")


# ---------------------------------------------------------------------------
# Generated artifacts.
# ---------------------------------------------------------------------------


def _json(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def frozen_manifest(train: bytes, evaluation: bytes) -> str:
    import hashlib

    return _json(
        {
            "banner": BANNER,
            "source": SOURCE,
            "train_bytes": len(train),
            "train_sha256": hashlib.sha256(train).hexdigest(),
            "eval_bytes": len(evaluation),
            "eval_sha256": hashlib.sha256(evaluation).hexdigest(),
            "denominator_note": (
                "eval_bytes IS the graded denominator. It is the byte length of eval.txt and "
                "nothing else, and no construction may divide by any other number."
            ),
        }
    )


def controls_document(source: dict) -> str:
    # `corpus_construction` carries this slot's two discovery values, `stride` and `offset`,
    # into the VERIFIER-ONLY control table, which is where they are readable and the agent
    # surface is not. They are the fixed index arithmetic the frozen corpus is built by.
    block = source["corpus"]
    return _json(
        {
            "banner": BANNER,
            "source": SOURCE,
            "corpus_construction": {
                "stride": int(block["stride"]),
                "offset": int(block["offset"]),
                "fields": list(block["fields"]),
                "eval_start": int(block["eval_start"]),
            },
            "single_direction_sweep": source["controls"]["single_direction_sweep"],
            "reference_allocation": source["controls"]["reference_allocation"],
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
            "# construction plan. It trains nothing, measures nothing and reports no number,",
            "# because every number on the graded path is the harness's own.",
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
    block = source["corpus"]
    train = corpus_bytes(block, int(block["train_start"]), int(block["train_lines"]))
    evaluation = corpus_bytes(block, int(block["eval_start"]), int(block["eval_lines"]))
    return {
        "environment/corpus/train.txt": train,
        "environment/corpus/eval.txt": evaluation,
        "environment/corpus/FROZEN.json": frozen_manifest(train, evaluation).encode(),
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
