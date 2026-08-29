# FORGE-CANARY-BEGIN
# 0: 9809335bf3a620257db443b74b08e304a595cbf93d5d1a9c2111d8c8ddf4da6f
# 1: 089fe00c736c34cb436c7c2d93bd12dfc5ebf30565ab0048ac0a50b0b7f5d724
# 2: 59021d6641937306a93dcbdaf24de73cdf7b70d874cc9638af1bdd53a60cad89
# 3: f38d3fa917b9ec24a8c01fed3f767d26c5231d114fa276fc4fd85b8eb5d1323e
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of slot OER-19 from solution/grounding.yaml alone.

One private source, one deriver. Everything this writes carries the generated
banner naming `solution/grounding.yaml`, so no reader has to guess whether a file
was hand-edited. There is no model call, no network access, no clock read, no
locale lookup and no random source anywhere in this file or in the two modules it
imports, which is what makes running it twice over frozen bytes produce
byte-identical output.

`--check` re-derives everything in memory and reports drift without writing a byte.
That is the mode the feasibility bundle runs, because a deriver that can only be
verified by overwriting the tree cannot be used to verify the tree.

Artifacts derived:

    tests/benchmark_held_out.jsonl        the frozen held-out benchmark
    environment/frozen/benchmark_dev.jsonl the agent-visible dev split, disjoint
    solution/reference.py                 the reference generator
    solution/solve.sh                     the reference entry point
    solution/TRUTH.md                     what is graded and why
    solution/rubrics.json                 solution-against-reference criteria
    solution/golden_trajectory.json       the golden trajectory
    solution/fixtures/*.json              the checker fixtures, accepting and rejecting
    tests/test_output.py                  the compiled test per checker
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
SOURCE = HERE / "grounding.yaml"
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE_NOTE = "source: solution/grounding.yaml"

sys.path.insert(0, str(BUNDLE / "tests"))

import diversity  # noqa: E402
import runner  # noqa: E402
import trainer  # noqa: E402


def load_source() -> dict:
    with SOURCE.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# Corpus and benchmark rendering
# ---------------------------------------------------------------------------


def render_reference_corpus(source: dict) -> list:
    """The reference corpus: every template crossed with every subject and modifier.

    Coverage is structural. The generator does not sample; it walks a grid, so the
    diversity of its output is a property of the grid rather than a property of a
    draw that could quietly stop being representative partway through.
    """
    rows, seq = [], 0
    for label in source["bounds"]["labels"]:
        grid = source["reference_grid"][label]
        for template in grid["templates"]:
            for subject in grid["subjects"]:
                for modifier in grid["modifiers"]:
                    text = template.replace("{S}", subject).replace("{M}", modifier)
                    rows.append({"seq": seq, "label": label, "text": text})
                    seq += 1
    return rows


def render_split(source: dict, key: str, prefix: str) -> list:
    grid = source[key]
    rows = []
    for label in source["bounds"]["labels"]:
        block = grid[label]
        for tindex, template in enumerate(block["templates"]):
            for pindex, pair in enumerate(block["pairs"]):
                subject, modifier = pair[0], pair[1]
                text = template.replace("{S}", subject).replace("{M}", modifier)
                rows.append(
                    {
                        "id": prefix + "-" + label + "-" + str(tindex) + "-" + str(pindex),
                        "label": label,
                        "text": text,
                    }
                )
    return rows


def render_jsonl(rows) -> str:
    body = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows)
    return body + "\n"


# ---------------------------------------------------------------------------
# Negative-control corpora
# ---------------------------------------------------------------------------


def collapsed_tail(source: dict, count: int, start_seq: int) -> list:
    """The degenerate mode: one template, a rotating integer, the label cycle intact.

    Every property a shape check looks at survives here. The lines are never
    byte-identical, the labels stay balanced, the JSON parses and the sample count
    is exact. Only the diversity statistic sees it.
    """
    mode = source["collapse_mode"]
    cycle = mode["label_cycle"]
    rows = []
    for offset in range(count):
        seq = start_seq + offset
        rows.append(
            {
                "seq": seq,
                "label": cycle[offset % len(cycle)],
                "text": mode["template"] + " " + str(seq),
            }
        )
    return rows


def control_corpora(source: dict, reference: list, held_out: list) -> dict:
    total = source["bounds"]["corpus_samples"]
    onset = source["collapse_mode"]["onset_sample"]
    corpora = {}
    corpora["nc-noop"] = []
    corpora["nc-silent-collapse"] = reference[:onset] + collapsed_tail(source, total - onset, onset)
    corpora["nc-uniform-mode"] = collapsed_tail(source, total, 0)
    # The echo control replaces samples at an even stride rather than appending a
    # block, so its diversity profile stays flat and the near-duplicate checker is
    # the gate that fires. A control that also trips an earlier gate would prove
    # nothing about the gate it was built for.
    echo = [dict(row) for row in reference]
    stride = max(1, total // max(1, len(held_out)))
    for offset, item in enumerate(held_out):
        position = offset * stride
        if position >= len(echo):
            break
        echo[position] = {
            "seq": echo[position]["seq"],
            "label": item["label"],
            "text": item["text"],
        }
    corpora["nc-benchmark-echo"] = echo
    return corpora


# ---------------------------------------------------------------------------
# Telemetry fixtures
# ---------------------------------------------------------------------------


def telemetry_for(source: dict, corpus: list, held_out: list, generator_sha: str, honest=True) -> dict:
    report = None
    if honest and corpus:
        profile = diversity.profile(
            [row["text"] for row in corpus],
            source["bounds"]["segment_count"],
            source["bounds"]["prefix_points"],
            source["bounds"]["mode_share_threshold"],
        )
        report = {
            "declared_corpus_distinct_ngram_ratio": profile.corpus_distinct_ngram_ratio,
            "declared_max_mode_share": profile.max_mode_share,
            "declared_samples": profile.samples,
        }
    return runner.produce_telemetry(
        corpus_rows=corpus,
        held_out=held_out,
        generator_sha256=generator_sha,
        generator_exit_status=0,
        self_report=report,
        bounds=source["bounds"],
    )


def patch(base: dict, mutate) -> dict:
    row = json.loads(json.dumps(base, sort_keys=True))
    mutate(row)
    return row


def telemetry_controls(source: dict, accepting: dict) -> dict:
    bounds = source["bounds"]
    out = {}

    def shuffled(row):
        sequence = list(row["corpus"]["emission_sequence"])
        if len(sequence) > 4:
            sequence[1], sequence[3] = sequence[3], sequence[1]
        row["corpus"]["emission_sequence"] = sequence

    def overspend(row):
        row["training"]["updates"] = int(bounds["bound_updates"]) + 120
        row["training"]["samples_fed"] = int(bounds["bound_samples_fed"]) + 120

    def smoothed(row):
        row["evaluation"]["readout"] = {"kind": "ema", "window": 3, "ema_alpha": 0.4}
        row["evaluation"]["graded_accuracy"] = min(
            1.0, row["evaluation"]["graded_accuracy"] + 0.05
        )

    def early(row):
        row["training"]["terminated_early"] = True
        row["training"]["updates"] = 1200
        row["training"]["samples_fed"] = 1200
        row["evaluation"]["points"] = [
            item for item in row["evaluation"]["points"] if item["update"] != 1440
        ]

    def unsustained(row):
        for item in row["evaluation"]["points"]:
            if item["update"] != int(bounds["bound_evaluation_point"]):
                item["accuracy"] = max(0.0, row["evaluation"]["graded_accuracy"] - 0.30)

    def substituted(row):
        row["evaluation"]["weights_source"] = "submission-checkpoint"
        row["evaluation"]["graded_model_state_digest"] = hashlib.sha256(
            b"submission-selected-checkpoint"
        ).hexdigest()

    def fabricated(row):
        row["self_report"] = {
            "present": True,
            "declared_corpus_distinct_ngram_ratio": 0.99,
            "declared_max_mode_share": 0.001,
            "declared_samples": int(bounds["corpus_samples"]),
        }

    out["nc-shuffled-collapse"] = patch(accepting, shuffled)
    out["nc-budget-overspend"] = patch(accepting, overspend)
    out["nc-smoothed-readout"] = patch(accepting, smoothed)
    out["nc-early-stop"] = patch(accepting, early)
    out["nc-unsustained"] = patch(accepting, unsustained)
    out["nc-substituted-weights"] = patch(accepting, substituted)
    out["nc-fabricated-self-report"] = patch(accepting, fabricated)
    return out


def stale_controls(source: dict, accepting: dict, controls: dict) -> dict:
    """One stale control per silent mutation: the pre-mutation answer, carried forward.

    A stale control has one job. It must PASS the checker its silent mutation is
    bound to, while the post-mutation fixture of the same run FAILS it. That pair is
    what makes drift causality provable rather than asserted: the only difference
    between the two documents is the state the mutation moved, and the verdict flips
    across exactly that difference.
    """
    bounds = source["bounds"]
    onset = int(source["collapse_mode"]["onset_sample"])
    total = int(bounds["corpus_samples"])
    segments = int(bounds["segment_count"])
    keep = max(1, (onset * segments) // total)
    out = {}

    def prefix_diversity(row):
        prefix = row["corpus"]["segment_distinct_ngram_ratio"][:keep]
        tiled = [prefix[index % len(prefix)] for index in range(segments)]
        row["corpus"]["segment_distinct_ngram_ratio"] = tiled
        row["corpus"]["max_mode_share"] = accepting["corpus"]["max_mode_share"]
        row["corpus"]["prefix_mode_share"] = list(accepting["corpus"]["prefix_mode_share"])
        row["stale_answer_from"] = "diversity verdict over the pre-collapse prefix, segments 0.." + str(keep - 1)

    def retired_revision(row):
        row["corpus"]["benchmark_near_duplicate_hits"] = []
        row["stale_answer_from"] = "near-duplicate verdict against the retired held-out revision hb-2026-08-a"

    def first_point_only(row):
        graded = row["evaluation"]["graded_accuracy"]
        for item in row["evaluation"]["points"]:
            item["accuracy"] = graded
        row["stale_answer_from"] = "sustain verdict from the single evaluation point that existed before the reschedule"

    def reported_ledger(row):
        row["training"]["updates"] = int(bounds["bound_updates"])
        row["training"]["samples_fed"] = int(bounds["bound_samples_fed"])
        row["stale_answer_from"] = "budget verdict from the submission-reported ledger, which reported the bound counts"

    out["ctl-diversity-verdict-from-pre-collapse-prefix"] = patch(
        controls["nc-silent-collapse"], prefix_diversity
    )
    out["ctl-near-duplicate-verdict-from-retired-revision"] = patch(
        controls["nc-benchmark-echo"], retired_revision
    )
    out["ctl-sustain-verdict-from-first-point-only"] = patch(
        controls["nc-unsustained"], first_point_only
    )
    out["ctl-budget-verdict-from-submission-reported-ledger"] = patch(
        controls["nc-budget-overspend"], reported_ledger
    )
    return out


# ---------------------------------------------------------------------------
# Generated text artifacts
# ---------------------------------------------------------------------------


def render_reference_module(source: dict) -> str:
    grid = {
        label: source["reference_grid"][label] for label in source["bounds"]["labels"]
    }
    payload = json.dumps(grid, indent=4, sort_keys=True)
    bounds = source["bounds"]
    return '''#!/usr/bin/env python3
"""{banner}

{note}

The reference generator for slot OER-19. It walks a template grid crossed with its
subject and modifier fillers, so its coverage is a structural property rather than
the outcome of a draw that could stop being representative partway through the run.

It then does the thing this slot is about: it measures its own corpus with
`corpus_diversity`, the frozen tool on the agent surface, and writes what it
measured into run_report.json. Exiting zero is not the claim. The measured profile
is the claim, and the verifier recomputes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import corpus_diversity as cd

LABELS = {labels}
SEGMENTS = {segments}
PREFIX_POINTS = {points}
MODE_THRESHOLD = {threshold}
SEGMENT_FLOOR = {floor}
MODE_CEILING = {ceiling}

GRID = {payload}


def build() -> list:
    rows, seq = [], 0
    for label in LABELS:
        block = GRID[label]
        for template in block["templates"]:
            for subject in block["subjects"]:
                for modifier in block["modifiers"]:
                    text = template.replace("{{S}}", subject).replace("{{M}}", modifier)
                    rows.append({{"seq": seq, "label": label, "text": text}})
                    seq += 1
    return rows


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    rows = build()
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\\n")

    profile = cd.profile([row["text"] for row in rows], SEGMENTS, PREFIX_POINTS, MODE_THRESHOLD)
    report = {{
        "declared_corpus_distinct_ngram_ratio": profile.corpus_distinct_ngram_ratio,
        "declared_max_mode_share": profile.max_mode_share,
        "declared_samples": profile.samples,
        "normalization": profile.normalization,
        "segment_distinct_ngram_ratio": list(profile.segment_distinct_ngram_ratio),
    }}
    with (out / "run_report.json").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2) + "\\n")

    weakest = min(profile.segment_distinct_ngram_ratio) if profile.segment_distinct_ngram_ratio else 0.0
    if weakest < SEGMENT_FLOOR or profile.max_mode_share > MODE_CEILING:
        # A generator that noticed its own collapse says so instead of exiting zero.
        sys.stderr.write("corpus profile below the published bounds\\n")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.format(
        banner=BANNER,
        note=SOURCE_NOTE,
        labels=repr(tuple(bounds["labels"])),
        segments=bounds["segment_count"],
        points=bounds["prefix_points"],
        threshold=bounds["mode_share_threshold"],
        floor=bounds["segment_distinct_ngram_floor"],
        ceiling=bounds["max_mode_share_ceiling"],
        payload=payload,
    )


def render_solve(source: dict) -> str:
    return """#!/usr/bin/env bash
# {banner}
# {note}
#
# The reference entry point. It emits the corpus and its own measured diversity
# report into the run workspace and returns the generator's status, which the
# verifier records and never trusts.
set -euo pipefail

WORKSPACE="${{1:-${{OER19_WORKSPACE:-$PWD}}}}"
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TOOLS="${{OER19_TOOLS:-$HERE/../environment/tools}}"

mkdir -p "$WORKSPACE"
PYTHONPATH="$TOOLS${{PYTHONPATH:+:$PYTHONPATH}}" python3 "$HERE/reference.py" "$WORKSPACE"
""".format(banner=BANNER, note=SOURCE_NOTE)


def render_truth(source: dict, reference_sha: str, accepting: dict) -> str:
    bounds = source["bounds"]
    lines = [
        "# TRUTH.md",
        "",
        BANNER,
        "",
        SOURCE_NOTE,
        "",
        "## What is graded",
        "",
        source["metric"]["definition"].strip(),
        "",
        "Direction: " + source["metric"]["direction"] + ".",
        "Reward: `" + source["metric"]["reward_formula"] + "`, then `"
        + source["metric"]["clip"] + "`.",
        "",
        "## Anchors",
        "",
        "anchors_state: " + source["anchors"]["anchors_state"],
        "",
        "baseline_metric: absent. target_metric: absent. Gap `"
        + source["anchors"]["gap"] + "`.",
        "",
        source["anchors"]["statement"].strip(),
        "",
        source["anchors"]["consequence"].strip(),
        "",
        "## The archetype this slot carries",
        "",
        "AR6, silent execution failure. A generator can collapse to a low-diversity",
        "mode partway through a run, keep emitting, keep the sample count and the",
        "format exactly right, and exit zero without announcing anything. The corpus",
        "is degenerate and every shape check passes. This slot grades that corpus as a",
        "failure with its own reason, `generator-collapsed-silently`, which is",
        "deliberately distinct from any low-score reason: a degenerate corpus that",
        "happens to score above the floor is still a collapse and is named as one.",
        "",
        "## The reference",
        "",
        "`solution/reference.py`, sha256 `" + reference_sha + "`.",
        "",
        "Measured profile of the reference corpus, recomputed by the verifier:",
        "",
        "| statistic | reading | bound |",
        "|---|---|---|",
        "| corpus distinct-trigram ratio | "
        + str(round(accepting["corpus"]["corpus_distinct_ngram_ratio"], 6))
        + " | reported, not gated |",
        "| weakest segment distinct-trigram ratio | "
        + str(round(min(accepting["corpus"]["segment_distinct_ngram_ratio"]), 6))
        + " | floor " + str(bounds["segment_distinct_ngram_floor"]) + " |",
        "| max mode share | "
        + str(round(accepting["corpus"]["max_mode_share"], 6))
        + " | ceiling " + str(bounds["max_mode_share_ceiling"]) + " |",
        "| benchmark near-duplicate hits | "
        + str(len(accepting["corpus"]["benchmark_near_duplicate_hits"]))
        + " | must be zero at threshold " + str(bounds["near_duplicate_threshold"]) + " |",
        "| optimizer updates fed | "
        + str(accepting["training"]["updates"])
        + " | bound " + str(bounds["bound_updates"]) + " |",
        "| graded accuracy at update "
        + str(bounds["bound_evaluation_point"]) + " | "
        + str(round(accepting["evaluation"]["graded_accuracy"], 6))
        + " | agent_metric, unnormalizable while anchors are absent |",
        "",
        "## How the graded quantity resists a shaped readout",
        "",
        "- The score is the verifier's own evaluation, recomputed from the confusion",
        "  counts its evaluator recorded. It is never a number the submission printed.",
        "- The graded readout is raw. A smoothed readout on the graded path scores zero",
        "  with `readout-smoothing-on-graded-path`.",
        "- A reading must hold across every point the verifier schedules, "
        + str(bounds["scheduled_points"]) + ", within "
        + str(bounds["sustain_tolerance"]) + ".",
        "- A run halted at a favourable evaluation is graded as not having established",
        "  the score, with `early-stop-score-not-established`, never as an absent result.",
        "- The evaluated weights are the harness trainer's own state at the bound point.",
        "",
        "## Perception-axis exemption",
        "",
        source["perception_axis_exemption"].strip(),
        "",
    ]
    return "\n".join(lines)


def render_rubrics(source: dict, reference_sha: str) -> str:
    payload = {
        "banner": BANNER,
        "source": "solution/grounding.yaml",
        "slot": "OER-19",
        "graded_against": "solution/reference.py",
        "reference_sha256": reference_sha,
        "criteria": [
            {"id": row["id"], "criterion": row["criterion"]}
            for row in source["solution_rubric_criteria"]
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_trajectory(source: dict) -> str:
    payload = {
        "banner": BANNER,
        "source": "solution/grounding.yaml",
        "slot": "OER-19",
        "steps": source["golden_trajectory"],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_test_output(source: dict, checker_ids) -> str:
    head = '''"""{banner}

{note}

One compiled test per declared checker, each driven from the fixtures
`solution/recompute.py` derived from `solution/grounding.yaml`. Every test asserts
both halves: the accepting fixture passes the checker, and that checker's own
rejecting fixture fails it carrying exactly the declared zero reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "tests"))

import checkers  # noqa: E402

FIXTURES = BUNDLE / "solution" / "fixtures"


def _load(name):
    with (FIXTURES / (name + ".json")).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _accepting():
    return _load("accepting")


def _run(name, telemetry):
    return checkers.REGISTRY[name](telemetry)


'''.format(banner=BANNER, note=SOURCE_NOTE)
    blocks = []
    for ident, control, reason in checker_ids:
        blocks.append(
            'def test_{ident}():\n'
            '    verdict = _run("{ident}", _accepting())\n'
            '    assert verdict.passed, verdict.reason\n'
            '    verdict = _run("{ident}", _load("{control}"))\n'
            '    assert not verdict.passed\n'
            '    assert verdict.reason == "{reason}"\n'.format(
                ident=ident, control=control, reason=reason
            )
        )
    return head + "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

CHECKER_CONTROLS = (
    ("generator_collapse_absent", "nc-silent-collapse", "generator-collapsed-silently"),
    ("corpus_mode_share_bounded", "nc-uniform-mode", "corpus-mode-share-exceeded"),
    ("emission_order_monotone", "nc-shuffled-collapse", "emission-order-not-monotone"),
    ("no_benchmark_near_duplicate", "nc-benchmark-echo", "benchmark-near-duplicate-emitted"),
    ("training_budget_as_fed", "nc-budget-overspend", "training-budget-overspent"),
    ("unsmoothed_graded_readout", "nc-smoothed-readout", "readout-smoothing-on-graded-path"),
    ("early_stop_does_not_establish_score", "nc-early-stop", "early-stop-score-not-established"),
    ("score_sustained_across_scheduled_points", "nc-unsustained", "score-not-sustained"),
    ("graded_score_from_harness_model_state", "nc-substituted-weights", "score-not-from-harness-model-state"),
    ("self_report_matches_recomputation", "nc-fabricated-self-report", "self-report-diverges-from-recomputation"),
)


def derive() -> dict:
    source = load_source()
    artifacts = {}

    held_out = render_split(source, "held_out_grid", "hb")
    dev = render_split(source, "dev_grid", "dev")
    artifacts["tests/benchmark_held_out.jsonl"] = render_jsonl(held_out)
    artifacts["environment/frozen/benchmark_dev.jsonl"] = render_jsonl(dev)

    reference_text = render_reference_module(source)
    artifacts["solution/reference.py"] = reference_text
    reference_sha = hashlib.sha256(reference_text.encode("utf-8")).hexdigest()

    corpus = render_reference_corpus(source)
    accepting = telemetry_for(source, corpus, held_out, reference_sha)
    artifacts["solution/fixtures/accepting.json"] = (
        json.dumps(accepting, indent=2, sort_keys=True) + "\n"
    )

    corpora = control_corpora(source, corpus, held_out)
    corpus_fixtures = {}
    for name, rows in corpora.items():
        corpus_fixtures[name] = telemetry_for(source, rows, held_out, reference_sha)
        artifacts["solution/fixtures/" + name + ".json"] = (
            json.dumps(corpus_fixtures[name], indent=2, sort_keys=True) + "\n"
        )

    every = dict(corpus_fixtures)
    for name, row in telemetry_controls(source, accepting).items():
        every[name] = row
        artifacts["solution/fixtures/" + name + ".json"] = (
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )

    for name, row in stale_controls(source, accepting, every).items():
        artifacts["solution/fixtures/" + name + ".json"] = (
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )

    artifacts["solution/solve.sh"] = render_solve(source)
    artifacts["solution/TRUTH.md"] = render_truth(source, reference_sha, accepting)
    artifacts["solution/rubrics.json"] = render_rubrics(source, reference_sha)
    artifacts["solution/golden_trajectory.json"] = render_trajectory(source)
    artifacts["tests/test_output.py"] = render_test_output(source, CHECKER_CONTROLS)
    return artifacts


EXECUTABLE = ("solution/solve.sh", "solution/reference.py")


def write(artifacts: dict) -> list:
    moved = []
    for relative, text in sorted(artifacts.items()):
        path = BUNDLE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
            moved.append(relative)
        if relative in EXECUTABLE:
            path.chmod(0o755)
    return moved


def check(artifacts: dict) -> list:
    drifted = []
    for relative, text in sorted(artifacts.items()):
        path = BUNDLE / relative
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            drifted.append(relative)
    return drifted


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive slot OER-19 from grounding.yaml.")
    parser.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = parser.parse_args()
    artifacts = derive()
    if args.check:
        drifted = check(artifacts)
        print(json.dumps({"artifacts": len(artifacts), "drifted": drifted}, indent=2, sort_keys=True))
        return 1 if drifted else 0
    moved = write(artifacts)
    print(json.dumps({"artifacts": len(artifacts), "written": moved}, indent=2, sort_keys=True))
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
    raise SystemExit(main())
