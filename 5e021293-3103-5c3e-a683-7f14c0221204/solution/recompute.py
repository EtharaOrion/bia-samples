# FORGE-CANARY-BEGIN
# 0: 942f5d059e8f60cd3c098a36b6e5785b2a1e0a56cd015820a6ffbcff3e13cb67
# 1: 5e6d28b1a7d3008403806767356ebbc620491231c70bb3271eabc7c0e48f4fde
# 2: 53c5c7e403fa8ec20e884ac78157761d1a15ade907da59e0800624e925531cd3
# 3: eb4e82e2f1e0a7c2199c0a2d8bfc1210b1e47e85243a6524a8aa4ced6f1eb058
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact for OER-04 from solution/grounding.yaml alone.

Generated here, and nowhere else:

    solution/fixtures.json            the checker fixtures, reference bytes bound in
    solution/golden_trajectory.json   the golden trajectory
    solution/solve.sh                 the reference entry point
    solution/TRUTH.md                 the private write-up
    solution/rubrics.json             the solution-versus-reference rubric
    tests/test_output.py              the compiled per-checker tests

This module invokes NO model, NO network, NO clock, NO locale and NO random source.
Every number it emits is read from grounding.yaml or computed from it by fixed
arithmetic. Running it twice over frozen bytes produces byte-identical output, which
is checked by `--check`.

The one input that is not grounding.yaml is the sha256 of solution/reference.py, and
that is deliberate: binding the reference bytes into the accepting fixture is what
makes the accepting half prove the checkers accept THIS reference rather than merely
some file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

BUNDLE = Path(__file__).resolve().parent.parent
GROUNDING = BUNDLE / "solution" / "grounding.yaml"
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE_LINE = "Source: solution/grounding.yaml. Regenerate with solution/recompute.py."


def load_grounding() -> dict:
    return yaml.safe_load(GROUNDING.read_text(encoding="utf-8"))


def reference_digest() -> str:
    return hashlib.sha256((BUNDLE / "solution" / "reference.py").read_bytes()).hexdigest()


def weight_digest(seed: int, step: int, salt: str) -> str:
    """A deterministic stand-in for a state-dict digest inside a fixture.

    A fixture needs a digest that is stable and distinguishable per (seed, step). It
    is derived by hashing those coordinates rather than by reading a tensor, because
    recompute.py never opens a checkpoint and never invokes a model.
    """
    return hashlib.sha256((salt + ":" + str(seed) + ":" + str(step)).encode("utf-8")).hexdigest()


def grid_for(claim: int, spec: dict) -> list:
    period = int(spec["evaluation_period_steps"])
    first = claim - period * int(spec["grid_points_before_claim"])
    last = claim + period * int(spec["grid_points_after_claim"])
    return list(range(first, last + 1, period))


def loss_curve(grid: list, crossing: int, spec: dict) -> list:
    """A monotone curve that first falls below target at exactly `crossing`.

    Above the crossing the value sits `loss_above_target` plus one increment per grid
    point of distance, so it approaches the target from above. At and after the
    crossing it sits `loss_below_target` minus one increment per grid point, so it
    stays below. The shape is deterministic arithmetic; nothing is sampled.
    """
    period = int(spec["evaluation_period_steps"])
    decimals = int(spec["loss_decimals"])
    values = []
    for point in grid:
        if point < crossing:
            distance = (crossing - point) // period
            values.append(round(float(spec["loss_above_target"])
                                + float(spec["loss_above_step"]) * distance, decimals))
        else:
            distance = (point - crossing) // period
            values.append(round(float(spec["loss_below_target"])
                                - float(spec["loss_below_step"]) * distance, decimals))
    return values


def seed_row(seed: int, grid: list, losses: list, salt: str) -> dict:
    ledger, evaluated = [], []
    for point in grid:
        digest = weight_digest(seed, point, salt)
        ledger.append({"step": point, "weights_sha256": digest, "owner": "harness"})
        evaluated.append({"step": point, "weights_sha256": digest})
    return {
        "seed": seed,
        "run_completed": True,
        "evaluation_points": list(grid),
        "raw_val_loss": list(losses),
        "readout_source": "verifier-raw",
        "graded_readout_key": "raw_val_loss",
        "grid_source": "verifier",
        "checkpoint_ledger": ledger,
        "evaluated_weights": evaluated,
        "final_step": grid[-1],
    }


def base_telemetry(ground: dict, claim: int, crossings: list, salt: str) -> dict:
    spec = ground["grid"]
    grid = grid_for(claim, spec)
    for crossing in crossings:
        if crossing not in grid:
            raise SystemExit(
                "grounding error: crossing " + str(crossing)
                + " is not on the verifier grid for claimed_step " + str(claim)
            )
    seeds = [
        seed_row(101 + index, grid, loss_curve(grid, crossing, spec), salt)
        for index, crossing in enumerate(crossings)
    ]
    frozen = dict(ground["frozen_axes"])
    return {
        "schema": "forge.verifier_telemetry/v1",
        "produced_by": "tests/runner.py",
        "bound": dict(ground["bound_parameters"]),
        "submission": {"sha256": reference_digest(), "claimed_step": claim},
        "frozen_axes": frozen,
        "frozen_axes_locked": dict(frozen),
        "seeds": seeds,
    }


def _all_above(seed: dict, target: float, spec: dict) -> None:
    seed["raw_val_loss"] = [round(target + 0.0100 + 0.0002 * index, int(spec["loss_decimals"]))
                            for index in range(len(seed["evaluation_points"]))]


def apply_mutation(telemetry: dict, mutation: str, ground: dict) -> dict:
    """Plant exactly one defect, so exactly one checker fires."""
    spec = ground["grid"]
    target = float(ground["bound_parameters"]["target_loss"])
    sustain = int(ground["bound_parameters"]["sustain_points_required"])

    if mutation in ("", "none", None):
        return telemetry

    if mutation == "readout_smoothed":
        telemetry["bound"]["readout_smoothing_admitted"] = True
        for seed in telemetry["seeds"]:
            seed["readout_source"] = "submission-ema"
            seed["graded_readout_key"] = "ema_val_loss"
        return telemetry

    if mutation == "single_point_dip":
        for seed in telemetry["seeds"]:
            _all_above(seed, target, spec)
            middle = len(seed["raw_val_loss"]) // 2
            seed["raw_val_loss"][middle] = round(target - 0.0010, int(spec["loss_decimals"]))
        return telemetry

    if mutation == "halt_inside_window":
        for seed in telemetry["seeds"]:
            losses = seed["raw_val_loss"]
            dip = next(index for index, value in enumerate(losses) if value < target)
            keep = dip + sustain
            seed["evaluation_points"] = seed["evaluation_points"][:keep]
            seed["raw_val_loss"] = losses[:keep]
            seed["checkpoint_ledger"] = seed["checkpoint_ledger"][:keep]
            seed["evaluated_weights"] = seed["evaluated_weights"][:keep]
            seed["final_step"] = seed["evaluation_points"][-1]
        return telemetry

    if mutation == "grid_thinned":
        for seed in telemetry["seeds"]:
            for key in ("evaluation_points", "raw_val_loss", "checkpoint_ledger", "evaluated_weights"):
                seed[key] = seed[key][:1] + seed[key][2:]
            seed["final_step"] = seed["evaluation_points"][-1]
        return telemetry

    if mutation == "submission_checkpoint":
        telemetry["bound"]["checkpoint_ledger_owner"] = "submission"
        for seed in telemetry["seeds"]:
            for row in seed["checkpoint_ledger"]:
                row["owner"] = "submission"
        return telemetry

    if mutation == "widen_model":
        telemetry["frozen_axes"] = dict(telemetry["frozen_axes"], n_embd=1024)
        return telemetry

    if mutation == "never_below_target":
        for seed in telemetry["seeds"]:
            _all_above(seed, target, spec)
        return telemetry

    raise SystemExit("grounding error: unknown mutation " + repr(mutation))


def build_fixture(ground: dict, spec: dict) -> dict:
    claim = int(spec["claimed_step"])
    count = int(spec["seed_count"])
    crossings = spec.get("crossing_steps")
    if crossings is None:
        crossings = list(ground["reference_solution"]["crossing_steps"])[:count]
    else:
        crossings = list(crossings)[:count]
    telemetry = base_telemetry(ground, claim, crossings, spec["id"])
    telemetry = apply_mutation(telemetry, spec.get("mutate", "none"), ground)
    row = {
        "id": spec["id"],
        "kind": spec.get("kind", "rejecting"),
        "telemetry": telemetry,
        "expected_reward": spec.get("expected_reward"),
        "expected_reason": spec.get("expected_reason", ""),
        "expected_state": spec.get("expected_state", ""),
        "expected_classification": spec.get("expected_classification", ""),
        "fires": spec.get("fires", ""),
        "stale_control": spec.get("stale_control", ""),
        "reference_sha256": reference_digest(),
    }
    return row


def build_ramp(ground: dict) -> list:
    ramp = ground["continuity_ramp"]
    claim = int(ramp["claimed_step"])
    count = int(ramp["seed_count"])
    rows = []
    for fast in ramp["fast_counts"]:
        crossings = [int(ramp["fast_step"])] * int(fast) + [int(ramp["slow_step"])] * (count - int(fast))
        ident = str(ramp["id_prefix"]) + str(fast)
        rows.append({
            "id": ident,
            "kind": "continuity",
            "telemetry": base_telemetry(ground, claim, crossings, ident),
            "expected_reward": None,
            "expected_reason": "",
            "expected_state": "",
            "expected_classification": "",
            "fires": "significance_separation_cleared",
            "stale_control": "",
            "fast_count": int(fast),
            "reference_sha256": reference_digest(),
        })
    return rows


def build_fixtures(ground: dict) -> dict:
    rows = [build_fixture(ground, spec) for spec in ground["fixtures"]]
    rows.extend(build_ramp(ground))
    return {
        "banner": BANNER,
        "source": "solution/grounding.yaml",
        "reference_sha256": reference_digest(),
        "fixtures": rows,
    }


def build_golden(ground: dict) -> dict:
    return {
        "banner": BANNER,
        "source": "solution/grounding.yaml",
        "slot": ground["slot"],
        "graded_quantity": ground["graded_quantity"],
        "anchors": {
            "baseline_metric": ground["anchors"]["baseline_metric"],
            "target_metric": ground["anchors"]["target_metric"],
            "separation_margin": ground["anchors"]["separation_margin"],
        },
        "attempts": ground["golden_trajectory"]["attempts"],
        "reference_sha256": reference_digest(),
    }


def build_solve_sh(ground: dict) -> str:
    reference = ground["reference_solution"]
    lines = [
        "#!/usr/bin/env bash",
        "# " + BANNER,
        "# " + SOURCE_LINE,
        "#",
        "# The reference entry point. It installs the reference solution as the",
        "# submission and records the claim the reference reaches. It trains nothing",
        "# itself; tests/runner.py re-executes what is installed here.",
        "set -euo pipefail",
        "",
        'SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'SUBMISSION="${OER04_SUBMISSION:-/app/submission.py}"',
        'CLAIM="${OER04_CLAIM:-/app/claim.json}"',
        "",
        'mkdir -p "$(dirname "${SUBMISSION}")" "$(dirname "${CLAIM}")"',
        'cp "${SOLUTION_DIR}/reference.py" "${SUBMISSION}"',
        "",
        "cat > \"${CLAIM}\" <<'CLAIM_JSON'",
        json.dumps({"claimed_step": int(reference["claimed_step"])}, sort_keys=True),
        "CLAIM_JSON",
        "",
        "# sha256 of solution/reference.py at generation time, bound into the accepting",
        "# fixture so the accepting half proves the checkers accept THIS reference.",
        "# " + reference_digest(),
        'echo "reference installed at ${SUBMISSION}, claimed_step '
        + str(int(reference["claimed_step"])) + '"',
        "",
    ]
    return "\n".join(lines)


def build_truth(ground: dict) -> str:
    anchors = ground["anchors"]
    feas = ground["feasibility"]
    lines = [
        "<!-- " + BANNER + " -->",
        "<!-- " + SOURCE_LINE + " -->",
        "",
        "# TRUTH.md, OER-04",
        "",
        "**" + ground["title"] + "**  ",
        "Family " + ground["family"] + ", " + ground["family_title"]
        + ". Primary archetype " + ground["primary_archetype"] + ", "
        + ground["primary_archetype_title"] + ".",
        "",
        "Objective, verbatim from the approved contract: *" + ground["objective"] + "*",
        "",
        "## The graded quantity",
        "",
        ground["graded_quantity"],
        "",
        "## Anchors",
        "",
        "| | |",
        "|---|---|",
        "| baseline_metric | " + str(anchors["baseline_metric"]) + " |",
        "| baseline identity | " + anchors["baseline_identity"] + " |",
        "| target_metric | " + str(anchors["target_metric"]) + " |",
        "| target identity | " + anchors["target_identity"] + " |",
        "| authority | " + anchors["authority"] + " |",
        "| anchors_state | " + anchors["anchors_state"] + " |",
        "| pass_threshold | " + str(anchors["pass_threshold"]) + " |",
        "| separation_margin | " + str(anchors["separation_margin"]) + " |",
        "",
        "## The three outcomes",
        "",
        "| outcome | condition | classification |",
        "|---|---|---|",
        "| `significance-established` | " + ground["outcome_states"]["established"]["condition"]
        + " | pass |",
        "| `significance-unestablished-at-ceiling` | "
        + " ".join(ground["outcome_states"]["unproven"]["condition"].split())
        + " | unproven, distinct from failed |",
        "| failed reasons | a gate rejected the run or it never reached the target"
        " | failed, not unproven |",
        "",
        "The middle row is the whole slot. A submission whose measured improvement sits",
        "inside the seed-to-seed noise band at the bound seed ceiling has established",
        "nothing, and that is a third outcome rather than a shading of either neighbour.",
        "The reward degrades continuously through that band and reaches exactly zero only",
        "when the separation is at or below zero, so the band is visible in the number as",
        "well as in the reason code.",
        "",
        "## The reference solution",
        "",
        "`solution/reference.py`, sha256 `" + reference_digest() + "`.",
        "",
        " ".join(ground["reference_solution"]["recipe"].split()),
        "",
        "Claimed step " + str(ground["reference_solution"]["claimed_step"])
        + ", twenty re-executed seeds, every one at or below the target, so the spread is",
        "zero, the separation is 1.0 and the reward is exactly 1.0.",
        "",
        "## Golden trajectory",
        "",
        "| attempt | change | graded_step | reward |",
        "|---|---|---|---|",
    ]
    for row in ground["golden_trajectory"]["attempts"]:
        lines.append("| " + str(row["attempt"]) + " | " + row["change"] + " | "
                     + str(row["graded_step"]) + " | " + str(row["reward"]) + " |")
    lines += [
        "",
        "## Feasibility",
        "",
        "| | |",
        "|---|---|",
        "| reference_hours | " + str(feas["reference_hours"]) + " |",
        "| budget_hours | " + str(feas["budget_hours"]) + " |",
        "| over budget | " + str(feas["over_budget"]) + " |",
        "| disposition | `" + feas["disposition"] + "` |",
        "| held by | `" + feas["disposition_held_by"] + "` |",
        "",
        " ".join(feas["disposition_statement"].split()),
        "",
        " ".join(feas["reference_hours_basis"].split()),
        "",
        "## Declared gaps",
        "",
    ]
    for gap in ground["declared_gaps"]:
        lines.append("- **`" + gap["id"] + "`** " + " ".join(gap["statement"].split()))
        lines.append("  Closes by: " + gap["closes_by"] + ".")
    lines.append("")
    return "\n".join(lines)


def build_rubrics_json(ground: dict) -> str:
    payload = {
        "banner": BANNER,
        "source": "solution/grounding.yaml",
        "judged": "the solution against its reference answer",
        "not_the_same_as": "tests/rubrics.jsonl, which is judged against the trajectory",
        "reference_sha256": reference_digest(),
        "criteria": ground["rubric_reference"],
    }
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def build_test_output(ground: dict, fixtures: dict) -> str:
    """Compile one test per checker, with its fixtures embedded from the same source."""
    by_id = {row["id"]: row for row in fixtures["fixtures"]}
    reference = by_id["reference"]
    checkers = ground["checkers"]

    embedded = {"reference": reference["telemetry"]}
    for row in fixtures["fixtures"]:
        if row.get("fires"):
            embedded[row["id"]] = row["telemetry"]
    for row in fixtures["fixtures"]:
        if row["kind"] == "continuity":
            embedded[row["id"]] = row["telemetry"]

    fires_of = {}
    for row in fixtures["fixtures"]:
        if row.get("fires") and row["kind"] == "rejecting":
            fires_of.setdefault(row["fires"], row["id"])

    head = [
        '"""' + BANNER,
        "",
        SOURCE_LINE,
        "",
        "One test per declared checker, each carrying BOTH halves: the reference",
        "telemetry it must accept, and the planted fixture it must reject with exactly",
        "its own zero reason. A suite proving only one half cannot tell a working",
        "instrument from an inert one, and an inert required instrument reads as a pass.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent))",
        "",
        "import checkers",
        "",
        "FIXTURES = json.loads(r'''",
        json.dumps(embedded, sort_keys=True, indent=1),
        "''')",
        "",
        "REFERENCE_SHA256 = " + repr(fixtures["reference_sha256"]),
        "",
        "",
        "def _run(name, fixture):",
        "    return getattr(checkers, name)(FIXTURES[fixture])",
        "",
        "",
    ]

    body = []
    for row in checkers:
        ident = row["id"]
        selector = "check_" + ident
        reason = row["zero_reason"]
        rejecting = fires_of.get(ident, "")
        body += [
            "def test_" + ident + "():",
            '    """Accepting half on the reference, rejecting half on the planted fixture."""',
            "    accepted = _run(" + repr(selector) + ", 'reference')",
            "    assert accepted.value == 1.0, " + repr(ident + " rejected the reference: ") + " + accepted.detail",
        ]
        if rejecting:
            body += [
                "    rejected = _run(" + repr(selector) + ", " + repr(rejecting) + ")",
                "    assert rejected.value < 1.0, " + repr(ident + " accepted a planted defect"),
                "    assert rejected.reason == " + repr(reason) + ", rejected.reason",
            ]
        body += ["", ""]

    tail = [
        "def test_reference_bytes_are_bound():",
        '    """The accepting half proves the checkers accept THIS reference, not any file."""',
        "    assert FIXTURES['reference']['submission']['sha256'] == REFERENCE_SHA256",
        "",
        "",
        "def test_unproven_band_is_not_a_failure():",
        '    """The middle outcome is its own state, never collapsed into either neighbour."""',
        "    verdict = _run('check_significance_separation_cleared', 'unproven_noise_band')",
        "    assert verdict.reason == 'significance-unestablished-at-ceiling', verdict.reason",
        "    assert verdict.state == checkers.STATE_UNPROVEN, verdict.state",
        "    gate = _run('check_target_reached_on_reexecuted_seeds', 'unproven_noise_band')",
        "    assert gate.value == 1.0, 'the unproven run reached the target and did not fail'",
        "",
        "",
        "def test_significance_factor_is_continuous_across_the_margin():",
        '    """No jump at 0.05. The ramp brackets the margin by construction."""',
        "    ramp = sorted(name for name in FIXTURES if name.startswith('near_margin_k'))",
        "    values = [_run('check_significance_separation_cleared', name).value for name in ramp]",
        "    assert values == sorted(values), values",
        "    assert values[0] > 0.0 and values[-1] == 1.0, values",
        "    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]",
        "    assert max(gaps) < 0.25, gaps",
        "",
        "",
        "def main() -> int:",
        "    failures = []",
        "    for name, fn in sorted(globals().items()):",
        "        if name.startswith('test_') and callable(fn):",
        "            try:",
        "                fn()",
        "            except AssertionError as exc:",
        "                failures.append(name + ': ' + str(exc))",
        "    for line in failures:",
        "        print('FAIL ' + line)",
        "    print(('FAILED ' + str(len(failures))) if failures else 'PASS all compiled checker tests')",
        "    return 1 if failures else 0",
        "",
        "",
        "if __name__ == '__main__':",
        "    raise SystemExit(main())",
        "",
    ]
    return "\n".join(head + body + tail)


TARGETS = (
    "solution/fixtures.json",
    "solution/golden_trajectory.json",
    "solution/solve.sh",
    "solution/TRUTH.md",
    "solution/rubrics.json",
    "tests/test_output.py",
)


def render(ground: dict) -> dict:
    fixtures = build_fixtures(ground)
    return {
        "solution/fixtures.json": json.dumps(fixtures, sort_keys=True, indent=2) + "\n",
        "solution/golden_trajectory.json":
            json.dumps(build_golden(ground), sort_keys=True, indent=2) + "\n",
        "solution/solve.sh": build_solve_sh(ground),
        "solution/TRUTH.md": build_truth(ground),
        "solution/rubrics.json": build_rubrics_json(ground),
        "tests/test_output.py": build_test_output(ground, fixtures),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="report drift instead of writing; exit 1 when any file drifted")
    args = parser.parse_args()

    rendered = render(load_grounding())
    drifted = []
    for relative in TARGETS:
        path = BUNDLE / relative
        text = rendered[relative]
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            drifted.append(relative)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                if relative.endswith(".sh"):
                    path.chmod(0o755)
    if args.check:
        for relative in drifted:
            print("DRIFTED " + relative)
        print("clean" if not drifted else "drifted " + str(len(drifted)))
        return 1 if drifted else 0
    for relative in drifted:
        print("wrote " + relative)
    print("regenerated " + str(len(drifted)) + " of " + str(len(TARGETS)) + " targets")
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
    sys.exit(main())
