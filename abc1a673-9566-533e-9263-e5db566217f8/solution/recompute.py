# FORGE-CANARY-BEGIN
# 0: 91c94f5734c06a179a52abcbe06dcb4ef0fa96f79974c40d4cef701b6f69917e
# 1: 3ec2514f2dc86d59e7e40be46580bb5fa6a8952e426e58206b510c69230b450d
# 2: 2f35aac8fe642fb73af4bbbc7d02625f539c88cda331da9ef8bf163f08444318
# 3: 0a559626e4a2f3105c85c503d730c2de48b9f14f90e59764715120c0c3a7b264
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of OER-26 from grounding.yaml and the frozen generator.

Generated here, and nowhere else:

    tests/bound.json                        every bound constant the checkers compare against
    tests/test_output.py                    the compiled per-checker tests
    solution/solve.sh                       the solution entry point
    solution/TRUTH.md                       the derivation write-up
    solution/rubrics.json                   the solution-against-reference rubric
    solution/fixtures/reference_run/        the oracle's own submission, as it actually ran
    solution/fixtures/rejecting/            the frozen rejecting fixtures

No model is invoked, no socket is opened, no clock is read, no locale is consulted and no random
source is drawn from. The fence state comes from environment/build_fence.py, whose only source of
variation is an integer linear congruential recurrence seeded from a constant, which is
arithmetic rather than randomness. Running this twice over frozen bytes produces byte-identical
output. `--check` reports drift and writes nothing.

Every expectation this module writes is derived from the BUILT fence state through the harness
handle. Not one of them is transcribed from the instruction, and neither discovery value is
written into any artifact that is agent visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "environment"))
sys.path.insert(0, str(BUNDLE / "tests"))

import build_fence  # noqa: E402
import fencelib  # noqa: E402

import checkers  # noqa: E402

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"


def load_grounding():
    with (HERE / "grounding.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def materialise_state(root: Path) -> Path:
    build_fence.materialise(root)
    return root


def telemetry_for(state_root: Path):
    """The same record stream the harness probe emits, derived here without a subprocess."""
    import fence_probe  # noqa: E402

    fence = fencelib.open_fence(state_root)
    return [json.loads(json.dumps(row, sort_keys=True)) for row in fence_probe.rows_for(fence)]


def reference_submission(harness) -> dict:
    """The oracle's answer, folded out of the harness traversal exactly as the fence states it."""
    truth = checkers.reference_verdicts(harness)
    refusals = {name: 0 for name in checkers.REFUSAL_REASONS}
    admitted = 0
    verdicts = []
    for row in truth:
        if row["decision"] == "admit":
            admitted += 1
        else:
            refusals[row["reason"]] += 1
        verdicts.append(
            {
                "seq": row["seq"],
                "kind": row["kind"],
                "decision": row["decision"],
                "reason": row["reason"],
            }
        )
    return {
        "schema": "oer26.submission/v1",
        "crossings_observed": len(truth),
        "admitted_kinds": checkers.admitted_kind_set(harness),
        "admitted": admitted,
        "refused": len(truth) - admitted,
        "refusals": refusals,
        "verdicts": verdicts,
    }


def baseline_agreement(harness, reference) -> float:
    """The instance-local zero: right decisions, every refusal flattened to one reason."""
    flattened = json.loads(json.dumps(reference))
    for row in flattened["verdicts"]:
        if row["decision"] == "refuse":
            row["reason"] = checkers.REFUSAL_NOT_ADMITTED
    probe = checkers.Harness(telemetry=harness.telemetry)
    probe._cache["submission"] = flattened
    return checkers.agreement(probe)


def _mutate(reference, mutation):
    document = json.loads(json.dumps(reference))
    mutation(document)
    return document


def rejecting_fixtures(harness, reference):
    """Every frozen rejecting fixture, each one a named departure from the oracle's answer."""
    derived = checkers.admitted_kind_set(harness)
    entries = harness.registry_entries()
    status_only = sorted(
        str(entry["kind"]) for entry in entries if str(entry.get("status")) == "admitted"
    )
    refused_kind = sorted(
        str(entry["kind"]) for entry in entries if str(entry["kind"]) not in derived
    )[0]
    on_disk_total = sum(int(row.get("records", 0)) for row in harness.segments())

    fixtures = {}

    # The two near-misses on the first discovery value. Exactly one field moves and no other
    # byte of the oracle's answer is touched.
    def count_minus(document):
        document["crossings_observed"] = document["crossings_observed"] - 1

    def count_plus(document):
        document["crossings_observed"] = document["crossings_observed"] + 1

    fixtures["crossing_count_minus_one"] = _mutate(reference, count_minus)
    fixtures["crossing_count_plus_one"] = _mutate(reference, count_plus)

    # The two near-misses on the second discovery value, plus the status-column reading.
    def set_minus(document):
        document["admitted_kinds"] = document["admitted_kinds"][1:]

    def set_plus(document):
        document["admitted_kinds"] = sorted(document["admitted_kinds"] + [refused_kind])

    def set_status_only(document):
        document["admitted_kinds"] = status_only

    fixtures["admitted_set_minus_one_kind"] = _mutate(reference, set_minus)
    fixtures["admitted_set_plus_one_kind"] = _mutate(reference, set_plus)
    fixtures["admitted_set_status_only"] = _mutate(reference, set_status_only)

    # A count that came from reading every file on disk rather than from the traversal.
    def disk_order(document):
        document["crossings_observed"] = on_disk_total
        document["admitted"] = document["admitted"]
        document["refused"] = on_disk_total - document["admitted"]
        document["refusals"][checkers.REFUSAL_NOT_ADMITTED] += on_disk_total - len(
            document["verdicts"]
        )
        while len(document["verdicts"]) < on_disk_total:
            document["verdicts"].append(
                {
                    "seq": len(document["verdicts"]) + 1,
                    "kind": refused_kind,
                    "decision": "refuse",
                    "reason": checkers.REFUSAL_NOT_ADMITTED,
                }
            )

    fixtures["verdicts_from_disk_order"] = _mutate(reference, disk_order)

    # A traversal that visited the chain's segments in sorted identifier order instead of link
    # order. The right number of crossings, decided against the wrong records.
    chain_rows = sorted(
        (row for row in harness.segments() if row.get("linked")),
        key=lambda row: str(row.get("segment_id")),
    )
    positions = []
    linked_by_position = sorted(
        (row for row in harness.segments() if row.get("linked")),
        key=lambda row: int(row.get("chain_position", 0)),
    )
    offset = {}
    running = 0
    for row in linked_by_position:
        offset[str(row["segment_id"])] = running
        running += int(row["records"])
    for row in chain_rows:
        base = offset[str(row["segment_id"])]
        positions.extend(range(base, base + int(row["records"])))

    def wrong_chain_order(document):
        document["verdicts"] = [document["verdicts"][index] for index in positions]

    fixtures["verdicts_from_wrong_chain_order"] = _mutate(reference, wrong_chain_order)

    # The seq column, the decisions and the reasons are the oracle's, and only the echoed record
    # identity is taken from a neighbouring crossing. This is the departure the ORDERING checker
    # cannot see and the DIVERGENCE checker exists for.
    def relabel_kinds(document):
        echoed = [row["kind"] for row in document["verdicts"]]
        rotated = echoed[1:] + echoed[:1]
        for row, kind in zip(document["verdicts"], rotated):
            row["kind"] = kind

    fixtures["verdict_kinds_relabelled"] = _mutate(reference, relabel_kinds)

    def admit_all(document):
        for row in document["verdicts"]:
            row["decision"] = "admit"
            row["reason"] = ""
        document["admitted"] = len(document["verdicts"])
        document["refused"] = 0
        document["refusals"] = {name: 0 for name in checkers.REFUSAL_REASONS}

    fixtures["admit_everything"] = _mutate(reference, admit_all)

    def refuse_all(document):
        for row in document["verdicts"]:
            row["decision"] = "refuse"
            row["reason"] = checkers.REFUSAL_NOT_ADMITTED
        document["admitted"] = 0
        document["refused"] = len(document["verdicts"])
        document["refusals"] = {name: 0 for name in checkers.REFUSAL_REASONS}
        document["refusals"][checkers.REFUSAL_NOT_ADMITTED] = len(document["verdicts"])

    fixtures["refuse_everything"] = _mutate(reference, refuse_all)

    def ignore_quota(document):
        moved = 0
        for row in document["verdicts"]:
            if row["reason"] == checkers.REFUSAL_QUOTA:
                row["decision"] = "admit"
                row["reason"] = ""
                moved += 1
        document["admitted"] += moved
        document["refused"] -= moved
        document["refusals"][checkers.REFUSAL_QUOTA] = 0

    fixtures["quota_ignored"] = _mutate(reference, ignore_quota)

    def flatten_reasons(document):
        for row in document["verdicts"]:
            if row["decision"] == "refuse":
                row["reason"] = checkers.REFUSAL_NOT_ADMITTED
        document["refusals"] = {name: 0 for name in checkers.REFUSAL_REASONS}
        document["refusals"][checkers.REFUSAL_NOT_ADMITTED] = document["refused"]

    fixtures["reasons_flattened"] = _mutate(reference, flatten_reasons)

    def invert_precedence(document):
        moved = 0
        for row in document["verdicts"]:
            if row["reason"] in (checkers.REFUSAL_MALFORMED, checkers.REFUSAL_UNATTESTED):
                row["reason"] = checkers.REFUSAL_NOT_ADMITTED
                moved += 1
        document["refusals"][checkers.REFUSAL_NOT_ADMITTED] += moved
        document["refusals"][checkers.REFUSAL_MALFORMED] = 0
        document["refusals"][checkers.REFUSAL_UNATTESTED] = 0

    fixtures["reasons_precedence_inverted"] = _mutate(reference, invert_precedence)

    return fixtures


def render_bound(harness, ground, reference, baseline) -> dict:
    salt = str(ground["commitment"]["salt"])
    crossings = checkers.crossing_count(harness)
    admitted = checkers.admitted_kind_set(harness)
    return {
        "anchors_gap": ground["anchors"]["gap_id"],
        "anchors_state": ground["anchors"]["anchors_state"],
        "admitted_set_commitment": checkers.commitment("admitted", ",".join(admitted), salt),
        "banner": BANNER,
        "baseline_agreement": baseline,
        "baseline_policy": ground["anchors"]["instance_local_anchors"]["baseline_policy"]["name"],
        "commitment_domain": checkers.COMMITMENT_DOMAIN,
        "commitment_salt": salt,
        "crossing_count_commitment": checkers.commitment("crossings", str(crossings), salt),
        "family_baseline_metric": ground["anchors"]["family_baseline_metric"],
        "family_target_metric": ground["anchors"]["family_target_metric"],
        "instance_anchor_gap": ground["anchors"]["instance_local_anchors"]["declared_gap"],
        "refusal_precedence": list(ground["fence"]["refusal_precedence"]),
        "reference_submission_sha256": hashlib.sha256(
            canonical(reference).encode("utf-8")
        ).hexdigest(),
        "schema": "oer26.bound/v1",
        "source": SOURCE,
        "state_sha256": str(harness.fence_row()["state_sha256"]),
    }


def canonical(document) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def render_truth(harness, ground, reference, baseline, outcomes) -> str:
    truth = checkers.reference_verdicts(harness)
    counts = {name: 0 for name in checkers.REFUSAL_REASONS}
    for row in truth:
        if row["decision"] == "refuse":
            counts[row["reason"]] += 1
    admits = sum(1 for row in truth if row["decision"] == "admit")
    lines = []
    lines.append("# TRUTH.md")
    lines.append("")
    lines.append(BANNER)
    lines.append("")
    lines.append("Source: `solution/grounding.yaml`. Regenerate with `solution/recompute.py`.")
    lines.append("")
    lines.append("## What is graded")
    lines.append("")
    lines.append(
        "Verdict agreement over every crossing the producer chain presents at the fence. A "
        "verdict agrees only when its decision and its refusal reason both match the verdict the "
        "stated fence semantics produce for that crossing at that position in the chain. "
        "Direction: higher is better."
    )
    lines.append("")
    lines.append(
        "The verifier establishes both discovery values for itself. `tests/runner.py` launches "
        "`environment/fence_probe.py` against the fence state the verifier image was built with, "
        "in an isolated subprocess as a new session leader under a six-key environment allowlist, "
        "and every checker is a pure function of the record stream that probe emitted. No number "
        "the submission reported reaches the reward."
    )
    lines.append("")
    lines.append("## The archetype")
    lines.append("")
    lines.append(
        "producer-fence. A fence separates what a producer may emit from what a consumer may "
        "admit. The solver has to admit exactly the admissible kinds, refuse the rest with the "
        "right reason under a strict precedence, and account for every crossing. Two facts decide "
        "the task and neither is in the statement: how many crossings the chain actually presents, "
        "and which kinds the registry actually admits."
    )
    lines.append("")
    lines.append("## The two discovery values, as the built state establishes them")
    lines.append("")
    lines.append("| value | how the build establishes it | how the oracle recovered it |")
    lines.append("|---|---|---|")
    lines.append(
        "| the fence-crossing count | the segment pool, each segment's width, the chain length "
        "and the chain order are all drawn from the generator's integer recurrence, so the count "
        "is not the product of any two constants in the generator | walked the links from the head "
        "segment to the null link and counted the records presented |"
    )
    lines.append(
        "| the admitted-kind set | each declaration is given a status and a seal by the "
        "recurrence, and a subset of the seals is deliberately broken, so a declaration can read "
        "`admitted` while being retired | took the declarations whose status reads `admitted` and "
        "whose seal recomputes over its own declared bytes |"
    )
    lines.append("")
    lines.append(
        "Neither value appears in `instruction.md`, in `task.toml`, in `environment/` or in "
        "`tests/`. `tests/bound.json` carries a salted commitment over each of them and never the "
        "value, and the checkers recompute both from harness-owned telemetry on every run."
    )
    lines.append("")
    lines.append("## The reference trace, as it actually ran")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append("| crossings the chain presented | " + str(len(truth)) + " |")
    lines.append(
        "| records present on disk across every segment file | "
        + str(sum(int(row.get("records", 0)) for row in harness.segments()))
        + " |"
    )
    lines.append(
        "| segments on disk | " + str(len(harness.segments())) + " |"
    )
    lines.append(
        "| segments the chain links | "
        + str(sum(1 for row in harness.segments() if row.get("linked")))
        + " |"
    )
    lines.append("| registry declarations | " + str(len(harness.registry_entries())) + " |")
    lines.append(
        "| declarations whose status reads admitted | "
        + str(
            sum(
                1
                for row in harness.registry_entries()
                if str(row.get("status")) == "admitted"
            )
        )
        + " |"
    )
    lines.append(
        "| kinds the fence admits | " + str(len(checkers.admitted_kind_set(harness))) + " |"
    )
    lines.append("| crossings admitted | " + str(admits) + " |")
    for name in checkers.REFUSAL_REASONS:
        lines.append("| crossings refused `" + name + "` | " + str(counts[name]) + " |")
    lines.append("| baseline agreement | " + repr(baseline) + " |")
    lines.append("| oracle agreement | " + repr(checkers.agreement(_probe(harness, reference))) + " |")
    lines.append("")
    lines.append(
        "The quota column is why the traversal order is load bearing. Quota is spent in chain "
        "order, so the "
        + str(counts[checkers.REFUSAL_QUOTA])
        + " crossings refused `quota-exhausted` are refused because of what the chain admitted "
        "before them, and a policy that decides the same records in a different order does not "
        "reproduce them."
    )
    lines.append("")
    lines.append("## Anchors")
    lines.append("")
    lines.append(
        "`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F3 carries no "
        "measured family baseline or target, so none is invented. The one normalisation number "
        "below is measured on this slot's own built state; it is instance-local and is NOT a "
        "family anchor, under gap `gap-oer-26-instance-local-anchors-not-family-anchors`."
    )
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append("| baseline policy | decisions right, every refusal flattened to one reason |")
    lines.append("| baseline agreement | " + repr(baseline) + " |")
    lines.append("| target agreement | 1.0 |")
    lines.append("")
    lines.append("raw = (agreement - baseline_agreement) / (1.0 - baseline_agreement)")
    lines.append("")
    lines.append(
        "score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and the "
        "baseline policy scores exactly 0.0 while passing every gate, which is what keeps the "
        "reward continuous rather than binary."
    )
    lines.append("")
    lines.append("## The rejecting fixtures, as they actually graded")
    lines.append("")
    lines.append("| fixture | what it moves | observed reward | observed reason |")
    lines.append("|---|---|---|---|")
    for name in sorted(outcomes):
        row = outcomes[name]
        lines.append(
            "| `"
            + name
            + "` | "
            + row["moves"]
            + " | "
            + repr(row["reward"])
            + " | `"
            + row["reason"]
            + "` |"
        )
    lines.append("")
    lines.append("## The clock and the traversal")
    lines.append("")
    lines.append(
        "THE HARNESS OWNS THE CLOCK, and on this slot it owns the traversal too. No checker "
        "reads a clock and no checker walks the fence inside the grading interpreter. "
        "`recompute.py` reads no clock either: every artifact above is derived by arithmetic from "
        "the frozen generator and never from a live measurement."
    )
    lines.append("")
    lines.append("## Substrate limit")
    lines.append("")
    lines.append(
        "The substrate is a deterministic build artifact, declared under gap "
        "`gap-oer-26-substrate-is-a-build-artifact-not-a-live-stream`. That is the right choice "
        "because it makes grading reproducible, and it never passes as an observation of a "
        "running production system."
    )
    lines.append("")
    return "\n".join(lines)


def _probe(harness, submission):
    probe = checkers.Harness(telemetry=harness.telemetry)
    probe._cache["submission"] = submission
    return probe


def render_rubrics(harness, reference, baseline) -> dict:
    truth = checkers.reference_verdicts(harness)
    return {
        "banner": BANNER,
        "criteria": [
            {
                "id": "crossing-count-from-traversal",
                "statement": "The crossing count is the one the producer chain presents, not the number of records on disk.",
                "comparison": "verifier-recomputed traversal length equals the submitted crossings_observed",
                "reference_value": len(truth),
            },
            {
                "id": "admitted-set-derived",
                "statement": "The admitted-kind set is the registry declarations that are both status admitted and correctly sealed.",
                "comparison": "set equality against the set derived from the built registry",
                "reference_value": checkers.admitted_kind_set(harness),
            },
            {
                "id": "precedence-respected",
                "statement": "Every refusal carries the reason the stated precedence assigns, first test that fires.",
                "comparison": "per-verdict reason equality against the verifier's own fold over the traversal",
                "reference_value": {
                    name: sum(1 for row in truth if row["reason"] == name)
                    for name in checkers.REFUSAL_REASONS
                },
            },
            {
                "id": "quota-spent-in-chain-order",
                "statement": "Quota is spent in chain order, so the quota refusals fall where the chain put them.",
                "comparison": "position equality of the quota-exhausted refusals",
                "reference_value": [
                    row["position"]
                    for row in truth
                    if row["reason"] == checkers.REFUSAL_QUOTA
                ],
            },
            {
                "id": "every-crossing-accounted",
                "statement": "Admitted plus refused equals the crossing count and the four refusal counts sum to refused.",
                "comparison": "arithmetic identity over the submitted counts",
                "reference_value": len(truth),
            },
            {
                "id": "beat-the-flattened-baseline",
                "statement": "The run does better than a policy that gets every decision right and every reason wrong.",
                "comparison": "verdict agreement strictly above the instance-local baseline",
                "reference_value": baseline,
            },
        ],
        "judged": "the solution against its reference answer",
        "schema": "forge.solution_rubrics/v1",
        "source": SOURCE,
    }


SOLVE_SH = """#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It opens the harness handle onto the BUILT fence state, walks the
# producer chain once, enforces the fence over what the traversal presented, and writes
# submission.json. It reads no answer file and hardcodes neither discovery value.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${OER26_WORKSPACE:-${PWD}}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
"""


TEST_OUTPUT_HEADER = '''"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker, plus the rejecting fixtures and the end-to-end reward
test. Every test drives the real checkers in tests/checkers.py over the harness record stream
the probe emits from the BUILT fence state. No test reads a clock, and no expectation below is
a transcribed literal: the accepting side reads the oracle's frozen submission and the rejecting
side reads a frozen fixture that departs from it in exactly one named way.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))

import pytest  # noqa: E402

import checkers  # noqa: E402
import grade  # noqa: E402
import runner  # noqa: E402

REFERENCE = BUNDLE / 'solution' / 'fixtures' / 'reference_run'
REJECTING = BUNDLE / 'solution' / 'fixtures' / 'rejecting'

# The held-out fixtures are not present on every surface this suite can be run from. Skipping is
# the honest outcome there; the reward path in tests/test.sh does not depend on this suite.
pytestmark = pytest.mark.skipif(
    not (REFERENCE / 'submission.json').is_file(),
    reason='the solution fixtures are not mounted on this surface',
)


def _telemetry():
    return runner.observe()


def _harness_over(document):
    harness = checkers.Harness(telemetry=_telemetry())
    harness._cache['submission'] = document
    return harness


def _fixture(name):
    return json.loads((REJECTING / (name + '.json')).read_text(encoding='utf-8'))


def _reference():
    return json.loads((REFERENCE / 'submission.json').read_text(encoding='utf-8'))


def _outcomes(document):
    harness = _harness_over(document)
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def _verdict_for(document):
    harness = _harness_over(document)
    bound = grade.load_bound(BUNDLE)
    return grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
    assert declared == set(_outcomes(_reference()))
'''


def render_test_output(names, outcomes, checker_ids) -> str:
    body = [TEST_OUTPUT_HEADER]
    for ident in checker_ids:
        body.append(
            "\n\ndef test_" + ident + "_accepts_the_oracle():\n"
            "    outcome = _outcomes(_reference())['" + ident + "']\n"
            "    assert outcome.passed, outcome.detail\n"
        )
    for name in names:
        row = outcomes[name]
        if row["reason"] == "graded":
            body.append(
                "\n\ndef test_" + name + "_is_graded_at_" + row["slug"] + "():\n"
                "    verdict = _verdict_for(_fixture('" + name + "'))\n"
                "    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)\n"
                "    assert verdict['reward'] == " + repr(row["reward"]) + ", json.dumps(verdict, sort_keys=True)\n"
            )
        else:
            body.append(
                "\n\ndef test_" + name + "_is_rejected():\n"
                "    verdict = _verdict_for(_fixture('" + name + "'))\n"
                "    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)\n"
                "    assert verdict['reason'] == '" + row["reason"] + "', json.dumps(verdict, sort_keys=True)\n"
                "    assert verdict['failed_checker'] == '" + row["failed_checker"] + "', json.dumps(verdict, sort_keys=True)\n"
            )
    body.append(
        "\n\ndef test_the_oracle_scores_full_reward():\n"
        "    verdict = _verdict_for(_reference())\n"
        "    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)\n"
        "    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)\n"
    )
    body.append(
        "\n\ndef test_neither_discovery_value_is_in_an_agent_visible_byte():\n"
        "    harness = _harness_over(_reference())\n"
        "    count = str(checkers.crossing_count(harness))\n"
        "    admitted = checkers.admitted_kind_set(harness)\n"
        "    offenders = []\n"
        "    for surface in ('instruction.md', 'task.toml', 'environment', 'tests'):\n"
        "        root = BUNDLE / surface\n"
        "        if not root.exists():\n"
        "            continue\n"
        "        paths = [root] if root.is_file() else [p for p in root.rglob('*') if p.is_file()]\n"
        "        for path in paths:\n"
        "            if '__pycache__' in path.as_posix() or path.name == 'test_output.py':\n"
        "                continue\n"
        "            text = path.read_text(encoding='utf-8', errors='replace')\n"
        "            for kind in admitted:\n"
        "                if kind in text:\n"
        "                    offenders.append(path.as_posix() + ' names admitted kind ' + kind)\n"
        "            if ','.join(admitted) in text:\n"
        "                offenders.append(path.as_posix() + ' names the admitted set')\n"
        "            if re.search(r'(?<![0-9A-Za-z_])' + count + r'(?![0-9A-Za-z_])', text):\n"
        "                offenders.append(path.as_posix() + ' names the crossing count')\n"
        "    assert not offenders, offenders\n"
    )
    return "".join(body)


def emit(paths, check: bool) -> int:
    drift = 0
    for path, text in paths:
        path = Path(path)
        if check:
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            if current != text:
                sys.stderr.write("drift: " + path.as_posix() + "\n")
                drift = 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return drift


def main():
    parser = argparse.ArgumentParser(description="derive every generated OER-26 artifact")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    ground = load_grounding()
    scratch = Path(tempfile.mkdtemp(prefix="oer26-recompute-"))
    try:
        state_root = materialise_state(scratch / "state")
        telemetry = telemetry_for(state_root)
        harness = checkers.Harness(telemetry=telemetry)

        reference = reference_submission(harness)
        baseline = baseline_agreement(harness, reference)
        bound_mapping = render_bound(harness, ground, reference, baseline)
        bound = checkers.Bound.from_mapping(bound_mapping)

        fixtures = rejecting_fixtures(harness, reference)
        outcomes = {}
        declared = {row["name"]: row for row in ground["rejecting_fixtures"]}
        import grade as grade_module  # noqa: E402

        for name in sorted(fixtures):
            verdict = grade_module.score(_probe(harness, fixtures[name]), bound, bound_mapping)
            outcomes[name] = {
                "reward": verdict["reward"],
                "reason": verdict["reason"],
                "failed_checker": verdict["failed_checker"],
                "moves": declared[name]["moves"],
                "slug": ("zero" if verdict["reward"] == 0.0 else "partial"),
            }

        oracle = grade_module.score(_probe(harness, reference), bound, bound_mapping)
        if oracle["reward"] != 1.0 or oracle["reason"] != "graded":
            sys.stderr.write(
                "the oracle does not earn full reward over the built state: "
                + json.dumps(oracle, sort_keys=True)
                + "\n"
            )
            return 1

        artifacts = [
            (BUNDLE / "tests" / "bound.json", canonical(bound_mapping)),
            (
                BUNDLE / "tests" / "test_output.py",
                render_test_output(sorted(fixtures), outcomes,
                                   [ident for ident, _ in grade_module.CHECKER_ORDER]),
            ),
            (BUNDLE / "solution" / "solve.sh", SOLVE_SH),
            (
                BUNDLE / "solution" / "TRUTH.md",
                render_truth(harness, ground, reference, baseline, outcomes),
            ),
            (
                BUNDLE / "solution" / "rubrics.json",
                canonical(render_rubrics(harness, reference, baseline)),
            ),
            (
                BUNDLE / "solution" / "fixtures" / "reference_run" / "submission.json",
                canonical(reference),
            ),
        ]
        for name in sorted(fixtures):
            artifacts.append(
                (
                    BUNDLE / "solution" / "fixtures" / "rejecting" / (name + ".json"),
                    canonical(fixtures[name]),
                )
            )

        status = emit(artifacts, args.check)
        print(
            json.dumps(
                {
                    "artifacts": len(artifacts),
                    "oracle_reward": oracle["reward"],
                    "baseline_agreement": baseline,
                    "rejecting_fixtures": {
                        name: {"reward": outcomes[name]["reward"], "reason": outcomes[name]["reason"]}
                        for name in sorted(outcomes)
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return status
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


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
    "screening_measured_at",
    "screening_interval_days",
    "screening_expires_at",
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
