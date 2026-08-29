# FORGE-CANARY-BEGIN
# 0: 3907ba2b668e1545a4c3c4ad7734e62210a227008b7fc50785061d411dba522b
# 1: 8a5ccfb3f090dc9f0bfd9857b3e41e878f715872f4c501482eb729862d577856
# 2: 576933bb182371e2e49ca1ffb9b75ea4a2891686678b4dd9f425d7d2a22810f0
# 3: ce6fba77d7941dcc57ea3e0eb7678edc98d7eed50478803d4b629105a3805d7c
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of slot OER-01 from solution/grounding.yaml.

This is the single derivation path. It invokes no model, no network, no clock, no
locale and no random source, and it sorts every container it emits, so running it
twice over frozen bytes produces byte-identical output. That property is not a
convenience: a generated artifact that moves on its own cannot be told apart from
one an author edited by hand, and the banner on every output would then be a claim
nobody could check.

Generated from grounding.yaml alone:

Authored content comes from grounding.yaml alone. The one file this reads besides
it is tests/fingerprint.py, copied verbatim into the agent-visible screen tool so
the two canonicalisers cannot drift apart; nothing is authored from that read.

    tests/anchors.json                    bound anchors and admin plane
    tests/corpus.json                     the pinned exclusion set, held out
    tests/fixtures.json                   the checker fixtures, accepting and rejecting
    tests/checkers.yaml                   the declared graded surface
    tests/rubrics.jsonl                   the trajectory rubric surface
    tests/test_output.py                  the compiled per-checker tests
    environment/published_records.json    the agent-visible screen material
    environment/fingerprint_tool.py       the agent-visible screen tool
    environment/frozen_axes.json          the agent-visible frozen record
    solution/golden_trajectory.json       the derived golden trajectory
    solution/solve.sh                     the reference entry point
    solution/TRUTH.md                     the reference narrative
    solution/rubrics.json                 the solution rubric, against the reference

Usage:  python3 solution/recompute.py [--check]
"""

import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "tests"))

import fingerprint  # noqa: E402

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
HEAD = BANNER + " Source: " + SOURCE + "."


def load_source() -> dict:
    return yaml.safe_load((HERE / "grounding.yaml").read_text(encoding="utf-8"))


def canonical_json(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def stable_digest(*parts) -> str:
    return hashlib.sha256("|".join(str(item) for item in parts).encode("utf-8")).hexdigest()


def corpus_entries(source) -> list:
    rows = []
    for record in source["record_corpus"]:
        mark = fingerprint.fingerprint(record["recipe"])
        rows.append(
            {
                "id": record["id"],
                "label": record["label"],
                "accepted": record["accepted"],
                "structural_digest": mark["structural_digest"],
                "hyper_vector": mark["hyper_vector"],
            }
        )
    return sorted(rows, key=lambda row: row["id"])


def post_pin_entries(source) -> list:
    rows = []
    for record in source.get("record_corpus_post_pin") or []:
        mark = fingerprint.fingerprint(record["recipe"])
        rows.append(
            {
                "id": record["id"],
                "label": record["label"],
                "accepted": record["accepted"],
                "structural_digest": mark["structural_digest"],
                "hyper_vector": mark["hyper_vector"],
            }
        )
    return sorted(rows, key=lambda row: row["id"])


def anchors_document(source) -> dict:
    return {
        "_banner": HEAD,
        "schema": "bia.oer01.anchors/v1",
        "anchors_state": source["anchors_state"],
        "anchor_authority": source["anchor_authority"],
        "baseline_metric": source["baseline_metric"],
        "baseline_identity": source["baseline_identity"].strip(),
        "target_metric": source["target_metric"],
        "target_identity": source["target_identity"].strip(),
        "target_validation_loss": source["metric"]["target_validation_loss"],
        "pass_threshold": source["pass_threshold"],
        "direction": source["metric"]["direction"],
        "reward_formula": source["reward"]["formula"],
        "reward_clip": source["reward"]["clip"],
        "admin_plane": dict(source["admin_plane"]),
        "frozen_axes": dict(source["frozen_axes_record"]),
    }


def corpus_document(source) -> dict:
    return {
        "_banner": HEAD,
        "schema": "bia.oer01.corpus/v1",
        "revision": source["record_corpus_revision"],
        "pin_note": source["record_corpus_pin_note"].strip(),
        "proximity_floor": source["fingerprint_control"]["proximity_floor"],
        "exclusion_rule": source["fingerprint_control"]["exclusion_rule"].strip(),
        "entries": corpus_entries(source),
        "post_pin_entries": post_pin_entries(source),
    }


def published_records_document(source) -> dict:
    return {
        "_banner": HEAD,
        "schema": "bia.oer01.published_records/v1",
        "purpose": (
            "The exclusion set is not a secret. Every recipe in it is already public, so hiding it would "
            "protect nothing and would only stop an honest agent from checking its own work before spending "
            "accelerator time. Screen your recipe against this list with environment/fingerprint_tool.py "
            "before you submit."
        ),
        "revision": source["record_corpus_revision"],
        "proximity_floor": source["fingerprint_control"]["proximity_floor"],
        "exclusion_rule": source["fingerprint_control"]["exclusion_rule"].strip(),
        "entries": [
            {"id": row["id"], "label": row["label"], "accepted": row["accepted"], "structural_digest": row["structural_digest"]}
            for row in corpus_entries(source)
        ],
    }


def frozen_axes_document(source) -> dict:
    return {
        "_banner": HEAD,
        "schema": "bia.oer01.frozen_axes/v1",
        "frozen": dict(source["frozen_axes_record"]),
        "rule": (
            "Dataset, batch size, architecture and exactly one forward-backward pass per optimizer step are "
            "frozen by upstream rule. A run that moves any of them is refused with frozen-axis-moved."
        ),
        "free": list(source["free_axes"]),
    }


def evaluation_rows(source, offset=0):
    seeds = list(source["golden_trajectory"]["seeds"])
    rows = []
    for index, item in enumerate(source["golden_trajectory"]["evaluations"]):
        step = int(item["step"]) + offset
        digest = stable_digest("oer01-checkpoint", step, tuple(seeds))
        rows.append(
            {
                "step": step,
                "seq": 10 + index,
                "verifier_raw_loss": float(item["verifier_raw_loss"]),
                "submission_reported_loss": float(item["submission_reported_loss"]),
                "weights_origin": "harness-checkpoint",
                "weights_digest": digest,
                "harness_checkpoint_digest": digest,
                "seeds_contributing": len(seeds),
            }
        )
    return rows


def telemetry_document(source, offset=0) -> dict:
    golden = source["golden_trajectory"]
    plane = source["admin_plane"]
    seeds = list(golden["seeds"])
    rows = evaluation_rows(source, offset)
    events = [
        {
            "seq": 0,
            "kind": "fingerprint-screen",
            "accelerator_seconds_spent_before": 0.0,
            "corpus_revision": source["record_corpus_revision"],
        }
    ]
    for index, seed in enumerate(seeds):
        events.append({"seq": index + 1, "kind": "seed-launch", "seed": seed})
    return {
        "schema": "bia.oer01.telemetry/v1",
        "origin": "derived-fixture",
        "origin_boundary": (
            "Authored in " + SOURCE + " as a derivation source. NOT a measurement. No accelerator run "
            "produced these numbers and no claim about an accelerator rests on them."
        ),
        "events": events,
        "run": {
            "launched": True,
            "seeds": seeds,
            "halt_step": int(golden["halt_step"]) + offset,
            "halt_cause": golden["halt_cause"],
            "run_directory_owner": "verifier",
            "scheduled_grid": [row["step"] for row in rows],
        },
        "evaluations": rows,
        "evaluations_per_seed": {str(seed): len(rows) for seed in seeds},
        "graded_series": [[row["step"], row["verifier_raw_loss"]] for row in rows],
        "graded_loss_provenance": plane["graded_loss_provenance"],
        "checkpoint_custody": plane["checkpoint_custody"],
        "frozen_axes": dict(source["frozen_axes_record"]),
    }


def screen_document(source, recipe, revision=None) -> dict:
    plane = source["admin_plane"]
    record = fingerprint.screen(recipe, corpus_entries(source), plane["fingerprint_proximity_floor"])
    record["corpus_revision"] = revision or source["record_corpus_revision"]
    record["recipe_readable"] = True
    return record


def jitter(hyper) -> dict:
    """Nudge every positive hyperparameter by three percent.

    Three percent is deliberately well inside one quantisation bucket, which is
    what makes this the interesting control: an exact-hash screen clears it and the
    proximity floor does not.
    """
    return {key: (round(value * 1.03, 6) if isinstance(value, (int, float)) and value > 0 else value) for key, value in sorted(hyper.items())}


def _by_id(rows, ident):
    for row in rows:
        if row["id"] == ident:
            return row
    raise KeyError(ident)


def fixtures_document(source) -> dict:
    reference_recipe = source["reference"]["recipe"]
    golden_screen = screen_document(source, reference_recipe)
    golden_telemetry = telemetry_document(source)

    partial = telemetry_document(source, offset=350)

    record_46 = _by_id(source["record_corpus"], "record-46")["recipe"]
    record_36 = _by_id(source["record_corpus"], "record-36")["recipe"]
    jittered = dict(record_46)
    jittered["hyper"] = jitter(record_46["hyper"])

    smoothed = telemetry_document(source)
    smoothed["graded_series"] = [[row["step"], row["submission_reported_loss"]] for row in smoothed["evaluations"]]

    early = telemetry_document(source)
    keep = [row for row in early["evaluations"] if row["step"] <= 2650]
    early["evaluations"] = keep
    early["evaluations_per_seed"] = {key: len(keep) for key in early["evaluations_per_seed"]}
    early["graded_series"] = [[row["step"], row["verifier_raw_loss"]] for row in keep]
    early["run"]["halt_step"] = 2650
    early["run"]["halt_cause"] = "submission-early-stop-on-favourable-eval"

    unsustained = telemetry_document(source)
    for row in unsustained["evaluations"]:
        if row["step"] > 2600:
            row["verifier_raw_loss"] = 3.2903
    unsustained["graded_series"] = [[row["step"], row["verifier_raw_loss"]] for row in unsustained["evaluations"]]

    picked = telemetry_document(source)
    for row in picked["evaluations"]:
        if row["step"] == 2600:
            row["weights_origin"] = "submission-selected"
            row["weights_digest"] = stable_digest("submission-selected", row["step"])

    late_screen = telemetry_document(source)
    late_screen["events"] = [
        {"seq": 0, "kind": "seed-launch", "seed": late_screen["run"]["seeds"][0]},
        {"seq": 1, "kind": "seed-launch", "seed": late_screen["run"]["seeds"][-1]},
        {
            "seq": 2,
            "kind": "fingerprint-screen",
            "accelerator_seconds_spent_before": 412.5,
            "corpus_revision": source["record_corpus_revision"],
        },
    ]

    empty = telemetry_document(source)
    empty["events"] = [empty["events"][0]]
    empty["run"] = {
        "launched": False,
        "seeds": [],
        "halt_step": None,
        "halt_cause": "",
        "run_directory_owner": "verifier",
        "scheduled_grid": [],
    }
    empty["evaluations"] = []
    empty["evaluations_per_seed"] = {}
    empty["graded_series"] = []

    moved = telemetry_document(source)
    moved["frozen_axes"] = dict(moved["frozen_axes"])
    moved["frozen_axes"]["batch_size"] = int(moved["frozen_axes"]["batch_size"]) * 2

    cases = [
        {
            "id": "golden_reference_accepted",
            "half": "accepting",
            "screen": golden_screen,
            "telemetry": golden_telemetry,
            "admin_override": {},
            "expect_reward": float(source["golden_trajectory"]["expected_reward"]),
            "expect_reason": source["golden_trajectory"]["expected_reason"],
            "proves": "every checker's accepting half over the reference",
        },
        {
            "id": "partial_credit_is_a_graded_float",
            "half": "accepting",
            "screen": golden_screen,
            "telemetry": partial,
            "admin_override": {},
            "expect_reward": float(source["partial_credit_fixture"]["expected_reward"]),
            "expect_reason": source["golden_trajectory"]["expected_reason"],
            "proves": "the reward is a graded float and never binary",
        },
        {
            "id": "nc_no_op",
            "half": "rejecting",
            "screen": screen_document(source, record_36),
            "telemetry": golden_telemetry,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "recipe-replays-published-record",
            "fires": "recipe_diverges_from_the_pinned_record_corpus",
        },
        {
            "id": "nc_replay_record_46",
            "half": "rejecting",
            "screen": screen_document(source, record_46),
            "telemetry": golden_telemetry,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "recipe-replays-published-record",
            "fires": "recipe_diverges_from_the_pinned_record_corpus",
        },
        {
            "id": "nc_jittered_replay_record_46",
            "half": "rejecting",
            "screen": screen_document(source, jittered),
            "telemetry": golden_telemetry,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "recipe-replays-published-record",
            "fires": "recipe_diverges_from_the_pinned_record_corpus",
        },
        {
            "id": "nc_stale_answer_pre_mutation",
            "half": "rejecting",
            "screen": golden_screen,
            "telemetry": golden_telemetry,
            "admin_override": {"record_corpus_revision": "r47"},
            "expect_reward": 0.0,
            "expect_reason": "recipe-replays-published-record",
            "fires": "recipe_diverges_from_the_pinned_record_corpus",
            "stale_control": "ctl-fingerprint-cleared-against-corpus-revision-r46",
        },
        {
            "id": "nc_screen_after_seeds",
            "half": "rejecting",
            "screen": golden_screen,
            "telemetry": late_screen,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "screen-ran-after-accelerator-time-was-spent",
            "fires": "fingerprint_screen_precedes_any_seed",
        },
        {
            "id": "nc_no_telemetry",
            "half": "rejecting",
            "screen": golden_screen,
            "telemetry": empty,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "graded-run-produced-no-telemetry",
            "fires": "graded_run_produced_verifier_owned_telemetry",
        },
        {
            "id": "nc_frozen_axis_moved",
            "half": "rejecting",
            "screen": golden_screen,
            "telemetry": moved,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "frozen-axis-moved",
            "fires": "frozen_axes_unmoved_by_the_graded_run",
        },
        {
            "id": "nc_readout_smoothing",
            "half": "rejecting",
            "screen": golden_screen,
            "telemetry": smoothed,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "readout-smoothed-on-the-graded-path",
            "fires": "graded_loss_recomputed_unsmoothed_by_verifier",
        },
        {
            "id": "nc_submission_selected_checkpoint",
            "half": "rejecting",
            "screen": golden_screen,
            "telemetry": picked,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "graded-weights-not-harness-owned",
            "fires": "graded_weights_are_harness_owned_at_the_graded_step",
        },
        {
            "id": "nc_early_stop",
            "half": "rejecting",
            "screen": golden_screen,
            "telemetry": early,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "run-halted-before-sustain-window-closed",
            "fires": "no_early_stop_before_the_sustain_window_closed",
        },
        {
            "id": "nc_unsustained_crossing",
            "half": "rejecting",
            "screen": golden_screen,
            "telemetry": unsustained,
            "admin_override": {},
            "expect_reward": 0.0,
            "expect_reason": "crossing-not-sustained",
            "fires": "crossing_sustained_across_verifier_scheduled_points",
        },
    ]
    return {
        "_banner": HEAD,
        "schema": "bia.oer01.fixtures/v1",
        "boundary": (
            "Every telemetry record here is DERIVED from " + SOURCE + ", not measured. The accepting cases "
            "prove the live checkers accept a trajectory of this shape produced by this reference. They prove "
            "nothing about what an accelerator would report."
        ),
        "reference_recipe": reference_recipe,
        "cases": cases,
    }


def checkers_yaml(source) -> str:
    lines = [
        "# " + HEAD,
        "#",
        "# The declared graded surface for slot OER-01. Every row carries exactly one",
        "# reduction from the closed six, names the carrier tests/test.sh reaches, names",
        "# the selector inside it, and names the machine-readable reason that carrier",
        "# emits when the checker scores zero.",
        "#",
        "# reward_path is bound to the path the live instrument seed/forge/verifier.py",
        "# binds, /logs/verifier/reward.txt, which carries the bare float. The lane brief",
        "# names /logs/verifier/reward.json for the reward document, which carries the",
        "# {reward, reason, metric} object. Both authorities are honoured: tests/test.sh",
        "# writes BOTH from one EXIT trap, and neither reader finds its path absent. The",
        "# collision is declared as gap-oer01-reward-path-extension-collides rather than",
        "# resolved by preference.",
        "",
        "schema: forge.checkers/v1",
        "reduction_kinds: [VALUE, EFFECT, ABSENCE, INVARIANT, ORDERING, DIVERGENCE]",
        "reward_path: /logs/verifier/reward.txt",
        "reward_document_path: /logs/verifier/reward.json",
        "score_document_path: /logs/verifier/score.json",
        "reward_writer: tests/test.sh",
        "",
        "isolation:",
        "  boundary: tests/runner.py",
        "  statement: >-",
        "    The submitted recipe executes only as a separate operating system process, in a",
        "    fresh temporary directory holding a single copy of itself, under python3 -I -S,",
        "    with a five key environment allowlist, in its own session, and the whole process",
        "    group is killed in a finally block. tests/grade.py imports the checkers and the",
        "    fingerprint canonicaliser and never imports the submission.",
        "",
        "aggregation:",
        "  mode: required_pass",
        "  rationale: >-",
    ]
    for chunk in _wrap(source["aggregation"]["rationale"].strip(), 88):
        lines.append("    " + chunk)
    lines.append("")
    lines.append("checkers:")
    for row in source["checkers"]:
        lines.append("  - id: " + row["id"])
        lines.append("    statement: >-")
        for chunk in _wrap(" ".join(row["statement"].split()), 84):
            lines.append("      " + chunk)
        lines.append("    reduction: " + row["reduction"])
        lines.append("    reached_by: tests/grade.py")
        lines.append("    selector: " + row["selector"])
        lines.append("    zero_reason: " + row["zero_reason"])
        degenerates = row.get("degenerate_reasons") or []
        if degenerates:
            lines.append("    degenerate_reasons:")
            for degenerate in degenerates:
                lines.append("      - reason: " + degenerate["reason"])
                lines.append("        condition: >-")
                for chunk in _wrap(" ".join(degenerate["condition"].split()), 76):
                    lines.append("          " + chunk)
        lines.append("    required: true")
        lines.append("    weight: " + str(row["weight"]))
        lines.append("    basis: measured")
        lines.append("    count_predicate: " + json.dumps(row.get("count_predicate", "")))
        lines.append("    live_state_read: >-")
        for chunk in _wrap(" ".join(row["live_state_read"].split()), 84):
            lines.append("      " + chunk)
        lines.append("    reads_admin_keys: " + json.dumps(list(row.get("reads_admin_keys") or [])))
        lines.append("    compiled_test: tests/test_output.py::test_" + row["id"])
        lines.append("    both_halves:")
        lines.append("      accepts: golden_reference_accepted in tests/fixtures.json")
        lines.append(
            "      rejects: "
            + ", ".join(
                sorted(case["id"] for case in fixtures_document(source)["cases"] if case.get("fires") == row["id"])
            )
            + " in tests/fixtures.json"
        )
        lines.append("      substitution: none; both halves fire on the live checker over a fixture telemetry record")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _wrap(text, width):
    words, line, rows = text.split(), "", []
    for word in words:
        candidate = (line + " " + word).strip()
        if len(candidate) > width and line:
            rows.append(line)
            line = word
        else:
            line = candidate
    if line:
        rows.append(line)
    return rows


def rubrics_jsonl(source) -> str:
    rows = []
    for item in source["trajectory_rubrics"]:
        rows.append(json.dumps({"id": item["id"], "rubric": " ".join(item["rubric"].split())}, sort_keys=True))
    return "\n".join(rows) + "\n"


def rubrics_json(source) -> dict:
    return {
        "_banner": HEAD,
        "schema": "bia.oer01.solution_rubrics/v1",
        "judged": "the solution against its reference answer",
        "not_judged": "the trajectory; that surface is tests/rubrics.jsonl and is a different file with a different job",
        "reference": source["reference"]["path"],
        "items": [
            {
                "id": item["id"],
                "criterion": " ".join(item["criterion"].split()),
                "reference_answer": " ".join(str(item["reference_answer"]).split()),
                "weight": item["weight"],
            }
            for item in source["solution_rubric_items"]
        ],
        "total_weight": round(sum(item["weight"] for item in source["solution_rubric_items"]), 6),
    }


def solve_sh(source) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# " + HEAD,
            "#",
            "# The reference entry point. It emits the reference recipe to the bound",
            "# submission path and screens it against the pinned corpus before exiting, so a",
            "# reference that had become a replay would refuse here rather than at grade time.",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            'OUT="${BIA_SUBMISSION:-/workspace/submission/recipe.py}"',
            'mkdir -p "$(dirname "$OUT")"',
            'cp reference.py "$OUT"',
            'python3 - "$OUT" <<\'PY\'',
            "import json, sys",
            "from pathlib import Path",
            "sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests'))",
            "import fingerprint",
            "bundle = Path(__file__).resolve().parent.parent",
            "corpus = json.loads((bundle / 'tests' / 'corpus.json').read_text())",
            "recipe = fingerprint.recipe_from_source(Path(sys.argv[1]).read_text())",
            "record = fingerprint.screen(recipe, corpus['entries'], corpus['proximity_floor'])",
            "if record['replays']:",
            "    print('reference is a replay of ' + ', '.join(record['replays']), file=sys.stderr)",
            "    raise SystemExit(1)",
            "print(json.dumps({'structural_digest': record['structural_digest'], 'replays': []}, sort_keys=True))",
            "PY",
            "",
        ]
    )


def truth_md(source) -> str:
    plane = source["admin_plane"]
    mark = fingerprint.fingerprint(source["reference"]["recipe"])
    rows = [
        "<!-- " + HEAD + " -->",
        "",
        "# TRUTH, slot OER-01",
        "",
        "## What is graded",
        "",
        "The number of optimizer steps at the first evaluation point, on the grid the",
        "verifier itself schedules, whose **verifier-recomputed, unsmoothed** loss on the",
        "frozen held-out split falls below "
        + str(source["metric"]["target_validation_loss"])
        + " and stays below it across the next "
        + str(plane["sustain_window_points"])
        + " scheduled points.",
        "",
        "| bound quantity | value |",
        "|---|---|",
        "| baseline_metric | " + str(source["baseline_metric"]) + " |",
        "| target_metric | " + str(source["target_metric"]) + " |",
        "| target validation loss | " + str(source["metric"]["target_validation_loss"]) + " |",
        "| pass_threshold | " + str(source["pass_threshold"]) + " |",
        "| evaluation cadence, steps | " + str(plane["evaluation_cadence_steps"]) + " |",
        "| sustain window, evaluation points | " + str(plane["sustain_window_points"]) + " |",
        "| corpus revision | " + str(source["record_corpus_revision"]) + " |",
        "| proximity floor | " + str(source["fingerprint_control"]["proximity_floor"]) + " |",
        "",
        "reward = min(max((" + str(source["baseline_metric"]) + " - agent_metric) / ("
        + str(source["baseline_metric"]) + " - " + str(source["target_metric"]) + "), 0.0), 1.0)",
        "",
        "## The reference",
        "",
        " ".join(source["reference"]["derivation"].split()),
        "",
        "Its structural digest is `" + mark["structural_digest"] + "`.",
        "",
        " ".join(source["reference"]["why_it_is_not_a_replay"].split()),
        "",
        "## Why the fingerprint and not the network policy",
        "",
        " ".join(source["solver_egress_divergence"]["deny_list_is_not_a_containment_proof"].split()),
        "",
        "## What the golden trajectory is and is not",
        "",
        " ".join(source["golden_trajectory"]["boundary"].split()),
        "",
        "## Expected outcome for the reference",
        "",
        "| quantity | value |",
        "|---|---|",
        "| graded step | " + str(source["golden_trajectory"]["expected_graded_step"]) + " |",
        "| reward | " + str(source["golden_trajectory"]["expected_reward"]) + " |",
        "| reason | " + str(source["golden_trajectory"]["expected_reason"]) + " |",
        "",
        "## Declared gaps",
        "",
    ]
    for gap in source["declared_gaps"]:
        rows.append("- `" + gap["id"] + "`: " + " ".join(str(gap.get("statement") or gap.get("effect") or "").split()))
    rows.append("")
    return "\n".join(rows)


def test_output_py(source) -> str:
    rows = [
        "# " + HEAD,
        '"""Compiled per-checker tests. One test per declared checker, both halves.',
        "",
        "Each test drives the LIVE gate chain in tests/grade.py over a fixture from",
        "tests/fixtures.json and asserts the accepting half reaches full reward and the",
        "rejecting half reaches exactly that checker's zero_reason. The fixtures are",
        "derived, and the boundary on what that proves is recorded in the fixture file",
        "itself and in seed/tasks/OER-01/feasibility.yaml.",
        '"""',
        "",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "HERE = Path(__file__).resolve().parent",
        "sys.path.insert(0, str(HERE))",
        "",
        "import grade",
        "",
        "FIXTURES = json.loads((HERE / 'fixtures.json').read_text())",
        "BOUND = json.loads((HERE / 'anchors.json').read_text())",
        "",
        "",
        "def case(ident):",
        "    for row in FIXTURES['cases']:",
        "        if row['id'] == ident:",
        "            return row",
        "    raise KeyError(ident)",
        "",
        "",
        "def score(ident):",
        "    row = case(ident)",
        "    bound = dict(BOUND)",
        "    bound['admin_plane'] = dict(bound['admin_plane'])",
        "    bound['admin_plane'].update(row.get('admin_override') or {})",
        "    return grade.grade(row['telemetry'], row['screen'], bound)",
        "",
        "",
        "def test_reference_reaches_full_reward():",
        "    payload = score('golden_reference_accepted')",
        "    assert payload['reward'] == 1.0",
        "    assert payload['reason'] == " + json.dumps(source["golden_trajectory"]["expected_reason"]),
        "",
        "",
        "def test_reward_is_a_graded_float_and_never_binary():",
        "    payload = score('partial_credit_is_a_graded_float')",
        "    assert 0.0 < payload['reward'] < 1.0",
        "",
        "",
        "def test_every_top_level_numeric_key_is_the_single_graded_scalar():",
        "    payload = score('golden_reference_accepted')",
        "    numeric = sorted(k for k, v in payload.items() if isinstance(v, (int, float)) and not isinstance(v, bool))",
        "    assert numeric == ['reward']",
        "",
    ]
    fixtures = fixtures_document(source)["cases"]
    for row in source["checkers"]:
        rejecting = sorted(item["id"] for item in fixtures if item.get("fires") == row["id"])
        rows.extend(
            [
                "",
                "def test_" + row["id"] + "():",
                "    accepted = score('golden_reference_accepted')",
                "    verdicts = {item['id']: item for item in accepted['checkers']}",
                "    assert verdicts[" + json.dumps(row["id"]) + "]['passed'] is True",
                "    for ident in " + json.dumps(rejecting) + ":",
                "        refused = score(ident)",
                "        assert refused['reward'] == 0.0",
                "        assert refused['reason'] == " + json.dumps(row["zero_reason"]),
                "",
            ]
        )
    return "\n".join(rows).rstrip() + "\n"


def fingerprint_tool_py(source) -> str:
    """The agent-visible screen tool, carrying tests/fingerprint.py verbatim.

    Copied rather than paraphrased. Two independently written canonicalisers would
    drift, and the first time they disagreed an agent would be refused for a
    recipe its own tool had cleared. adequacy.py proves the copy and the original
    produce identical fingerprints over the whole corpus and the reference.
    """
    body = (BUNDLE / "tests" / "fingerprint.py").read_text(encoding="utf-8")
    header = "\n".join(
        [
            "#!/usr/bin/env python3",
            "# " + HEAD,
            "#",
            "# Agent-visible copy of the verifier's canonicaliser, byte-for-byte below the",
            "# banner. Screen your own recipe before you spend accelerator time:",
            "#",
            "#     python3 environment/fingerprint_tool.py /workspace/submission/recipe.py",
            "#",
            "# The corpus is public. Hiding it would protect nothing and would only stop an",
            "# honest agent from checking its own work.",
            "",
            "",
        ]
    )
    cli = "\n".join(
        [
            "",
            "",
            "def _main(argv):",
            "    import sys",
            "    from pathlib import Path",
            "",
            "    here = Path(__file__).resolve().parent",
            "    if len(argv) < 2:",
            "        print('usage: fingerprint_tool.py <recipe.py>', file=sys.stderr)",
            "        return 2",
            "    published = json.loads((here / 'published_records.json').read_text(encoding='utf-8'))",
            "    recipe = recipe_from_source(Path(argv[1]).read_text(encoding='utf-8'))",
            "    if recipe is None:",
            "        print('no module-level RECIPE literal found; the screen cannot read this file', file=sys.stderr)",
            "        return 1",
            "    mine = fingerprint(recipe)",
            "    rows = []",
            "    for entry in published['entries']:",
            "        rows.append({'id': entry['id'], 'structural_match': entry['structural_digest'] == mine['structural_digest']})",
            "    print(json.dumps({",
            "        'structural_digest': mine['structural_digest'],",
            "        'hyper_vector': mine['hyper_vector'],",
            "        'floor': published['proximity_floor'],",
            "        'structural_matches': sorted(r['id'] for r in rows if r['structural_match']),",
            "        'note': 'a structural match alone is not a replay; proximity at or under the floor decides it',",
            "    }, sort_keys=True))",
            "    return 0",
            "",
            "",
            "if __name__ == '__main__':",
            "    import sys",
            "",
            "    raise SystemExit(_main(sys.argv))",
            "",
        ]
    )
    return header + body.rstrip("\n") + cli


def golden_trajectory_document(source) -> dict:
    return {
        "_banner": HEAD,
        "schema": "bia.oer01.golden_trajectory/v1",
        "boundary": " ".join(source["golden_trajectory"]["boundary"].split()),
        "reference_recipe": source["reference"]["recipe"],
        "reference_fingerprint": fingerprint.fingerprint(source["reference"]["recipe"]),
        "telemetry": telemetry_document(source),
        "expected_graded_step": source["golden_trajectory"]["expected_graded_step"],
        "expected_reward": source["golden_trajectory"]["expected_reward"],
        "expected_reason": source["golden_trajectory"]["expected_reason"],
    }


def artifacts(source) -> dict:
    return {
        "tests/anchors.json": canonical_json(anchors_document(source)),
        "tests/corpus.json": canonical_json(corpus_document(source)),
        "tests/fixtures.json": canonical_json(fixtures_document(source)),
        "tests/checkers.yaml": checkers_yaml(source),
        "tests/rubrics.jsonl": rubrics_jsonl(source),
        "tests/test_output.py": test_output_py(source),
        "environment/published_records.json": canonical_json(published_records_document(source)),
        "environment/fingerprint_tool.py": fingerprint_tool_py(source),
        "environment/frozen_axes.json": canonical_json(frozen_axes_document(source)),
        "solution/golden_trajectory.json": canonical_json(golden_trajectory_document(source)),
        "solution/solve.sh": solve_sh(source),
        "solution/TRUTH.md": truth_md(source),
        "solution/rubrics.json": canonical_json(rubrics_json(source)),
    }


def main(argv) -> int:
    source = load_source()
    rows = artifacts(source)
    check_only = "--check" in argv
    drifted = []
    for relative, text in sorted(rows.items()):
        path = BUNDLE / relative
        if check_only:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                drifted.append(relative)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        if relative.endswith(".sh"):
            path.chmod(0o755)
    if check_only:
        print(json.dumps({"drifted": sorted(drifted), "generated": len(rows)}, sort_keys=True))
        return 1 if drifted else 0
    print(json.dumps({"generated": sorted(rows)}, sort_keys=True))
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
