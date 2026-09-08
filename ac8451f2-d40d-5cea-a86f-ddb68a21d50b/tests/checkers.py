"""The eight graded checkers. Pure, deterministic, and none of them reads a clock.

Every value a checker compares comes from one of two places and never from a third:

  1. the HARNESS-MINTED STORE. `tests/runner.py` launches the bundle's own minter in an
     isolated subprocess over the bundle's frozen ledger source and frozen instance family,
     materialises the store into a fresh temporary directory, and reads it back. The realised
     attestation order and every atom digest are then derived HERE by walking the seal chain
     the minter actually sealed. Nothing is read from a file the author planted with an answer
     in it, and nothing is read from the image the agent ran in.

  2. a bound constant handed in as the `Bound` argument, read by `tests/grade.py` from
     `tests/bound.json`, which `solution/recompute.py` derived from `solution/grounding.yaml`.

No value the submission reported reaches the reward. Seven submission-authored fields are
read, and each is read only so that a substitution is detectable rather than ignorable.

Imports are confined to the allowlist: hashlib, json, pathlib, dataclasses and typing.
Nothing here imports the submission, opens a socket, consults an environment secret, draws
from a random source or calls a clock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Machine-readable zero reasons. A downstream grader branches on these strings.
# ---------------------------------------------------------------------------
REASON_FROZEN_INPUTS = "frozen-inputs-modified"
REASON_STORE_NOT_MATERIALISED = "store-not-materialised"
REASON_ORDER_UNREALISED = "attestation-order-unrealised"
REASON_SEAL_FABRICATED = "seal-chain-fabricated"
REASON_CLOSURE_INCOMPLETE = "closure-incomplete"
REASON_PREIMAGE_WRONG = "digest-preimage-wrong"
REASON_TERMINAL_DIGEST = "terminal-digest-mismatch"
REASON_COVER_INFEASIBLE = "cover-infeasible"

SCHEMA_ATOM = "oer30.atom/v1"
SCHEMA_ATTESTATION = "oer30.attestation/v1"


@dataclass(frozen=True)
class Bound:
    """Every constant the checkers compare against. Handed in, never authored here."""

    atom_count: int
    genesis_prev_seal: str
    instances_sha256: str
    ledger_source_sha256: str
    minter_sha256: str
    instance_baseline_gap: float
    instance_target_gap: float
    weight_tolerance: int

    @staticmethod
    def from_mapping(payload: Dict[str, Any]) -> "Bound":
        return Bound(
            atom_count=int(payload["atom_count"]),
            genesis_prev_seal=str(payload["genesis_prev_seal"]),
            instances_sha256=str(payload["instances_sha256"]),
            ledger_source_sha256=str(payload["ledger_source_sha256"]),
            minter_sha256=str(payload["minter_sha256"]),
            instance_baseline_gap=float(payload["instance_baseline_gap"]),
            instance_target_gap=float(payload["instance_target_gap"]),
            weight_tolerance=int(payload["weight_tolerance"]),
        )


@dataclass
class Harness:
    """The real handle. Every read below is of state the minter actually produced.

    `store` is the in-memory store the verifier's own minter subprocess materialised over the
    bundle's frozen inputs. `instances` is the frozen instance family as it stands on disk.
    The three digests are the frozen inputs as they stand on disk. `workspace` is the run
    workspace the submission produced.
    """

    store: Dict[str, Any] = field(default_factory=dict)
    instances: Dict[str, Any] = field(default_factory=dict)
    bundle_instances_sha256: str = ""
    bundle_ledger_sha256: str = ""
    bundle_minter_sha256: str = ""
    workspace: Optional[Path] = None
    _cache: Dict[str, Any] = field(default_factory=dict)

    def submission(self) -> Dict[str, Any]:
        if "submission" not in self._cache:
            value: Any = None
            if self.workspace is not None:
                path = Path(self.workspace) / "submission.json"
                if path.is_file():
                    try:
                        value = json.loads(path.read_text(encoding="utf-8"))
                    except ValueError:
                        value = None
            self._cache["submission"] = value if isinstance(value, dict) else {}
        return self._cache["submission"]

    # -- store readers -------------------------------------------------------
    def attestations(self) -> List[Dict[str, Any]]:
        return list(self.store.get("attestations") or [])

    def atoms(self) -> Dict[str, Any]:
        return dict(self.store.get("atoms") or {})

    def realised_order(self) -> List[str]:
        """The order the seal chain realises, recovered HERE by walking it from genesis.

        It is deliberately not read from `store['realised_order']`. The chain is what the
        store carries and the walk is what a solver has to perform, so the verifier performs
        the same walk over the same records rather than trusting a convenience field.
        """
        by_prev = {}
        for row in self.attestations():
            by_prev.setdefault(str(row["prev_seal"]), []).append(row)
        order: List[str] = []
        cursor = "0" * 64
        seen = set()
        while cursor in by_prev:
            candidates = by_prev[cursor]
            if len(candidates) != 1:
                return []
            row = candidates[0]
            if row["seal"] in seen:
                return []
            seen.add(row["seal"])
            order.append(str(row["atom_id"]))
            cursor = str(row["seal"])
        return order

    def seal_sequence(self) -> List[str]:
        by_prev = {str(row["prev_seal"]): row for row in self.attestations()}
        out: List[str] = []
        cursor = "0" * 64
        while cursor in by_prev:
            row = by_prev[cursor]
            out.append(str(row["seal"]))
            cursor = str(row["seal"])
        return out

    def atom_digests(self) -> Dict[str, str]:
        """Every atom digest, folded HERE under the fixed preimage in realised order."""
        if "digests" not in self._cache:
            order = self.realised_order()
            atoms = self.atoms()
            position = {ident: index for index, ident in enumerate(order)}
            digests: Dict[str, str] = {}
            for ident in order:
                record = atoms.get(ident)
                if record is None:
                    digests = {}
                    break
                parents = sorted(record.get("inputs") or [], key=lambda name: position[name])
                digests[ident] = atom_digest(record, [digests[name] for name in parents])
            self._cache["digests"] = digests
        return self._cache["digests"]

    def terminal_atom_id(self) -> str:
        order = self.realised_order()
        return order[-1] if order else ""


@dataclass
class Outcome:
    """One checker's verdict. `reason` is empty exactly when the checker passed."""

    ident: str
    passed: bool
    reason: str
    detail: str
    value: Any = None


def _pass(ident: str, detail: str, value: Any = None) -> Outcome:
    return Outcome(ident, True, "", detail, value)


def _fail(ident: str, reason: str, detail: str, value: Any = None) -> Outcome:
    return Outcome(ident, False, reason, detail, value)


# ---------------------------------------------------------------------------
# The canonical preimages. Transcribed from the schema the statement fixes and
# never inferred from a value the submission carried.
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cover_sha256(selected_sets) -> str:
    return _sha256_text(",".join(str(int(value)) for value in sorted(selected_sets)))


def atom_preimage(atom: Dict[str, Any], input_digests) -> str:
    return (
        SCHEMA_ATOM + "\n"
        + "atom_id=" + str(atom["atom_id"]) + "\n"
        + "instance_id=" + str(atom["instance_id"]) + "\n"
        + "heuristic=" + str(atom["heuristic"]) + "\n"
        + "cover=" + cover_sha256(atom["selected_sets"]) + "\n"
        + "inputs=" + ";".join(input_digests) + "\n"
    )


def atom_digest(atom: Dict[str, Any], input_digests) -> str:
    return _sha256_text(atom_preimage(atom, input_digests))


def attestation_seal(atom_id: str, cover_digest: str, prev_seal: str) -> str:
    return _sha256_text(
        SCHEMA_ATTESTATION + "\n"
        + "atom_id=" + str(atom_id) + "\n"
        + "cover_sha256=" + str(cover_digest) + "\n"
        + "prev_seal=" + str(prev_seal) + "\n"
    )


# ---------------------------------------------------------------------------
# Quantities the VERIFIER computes. Never a number the submission reported.
# ---------------------------------------------------------------------------


def exact_optimum(instance: Dict[str, Any]) -> int:
    """Minimum total weight of a sub-family of sets covering the whole universe.

    An exact bitmask dynamic program over 2**universe_size states. It is the classical exact
    comparator the family metric is normalised against, and it is recomputed by the verifier
    on every run rather than read from a table.
    """
    universe = int(instance["universe_size"])
    full = (1 << universe) - 1
    masks = []
    for row in instance["sets"]:
        mask = 0
        for element in row["members"]:
            mask |= 1 << int(element)
        masks.append((mask, int(row["weight"])))
    infinity = float("inf")
    best = [infinity] * (full + 1)
    best[0] = 0
    for state in range(full + 1):
        if best[state] == infinity:
            continue
        for mask, weight in masks:
            following = state | mask
            if following != state and best[state] + weight < best[following]:
                best[following] = best[state] + weight
    return int(best[full])


def graded_instance(harness: Harness) -> Optional[Dict[str, Any]]:
    terminal = harness.terminal_atom_id()
    if not terminal:
        return None
    record = harness.atoms().get(terminal)
    if record is None:
        return None
    return harness.instances.get(record["instance_id"])


def submitted_cover_weight(harness: Harness) -> Optional[int]:
    instance = graded_instance(harness)
    if instance is None:
        return None
    submission = harness.submission()
    cover = submission.get("submitted_cover")
    if not isinstance(cover, list):
        return None
    weights = {int(row["id"]): int(row["weight"]) for row in instance["sets"]}
    total = 0
    for value in cover:
        try:
            total += weights[int(value)]
        except (KeyError, TypeError, ValueError):
            return None
    return total


def normalised_gap(harness: Harness) -> Optional[float]:
    """The graded metric: (submitted weight - exact optimum) / exact optimum. Lower is better."""
    instance = graded_instance(harness)
    weight = submitted_cover_weight(harness)
    if instance is None or weight is None:
        return None
    optimum = exact_optimum(instance)
    if optimum <= 0:
        return None
    return round((float(weight) - float(optimum)) / float(optimum), 9)


# ---------------------------------------------------------------------------
# The eight graded checkers, in grading order
# ---------------------------------------------------------------------------


def check_frozen_inputs_unmodified(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The frozen ledger source, instance family and minter are byte-identical."""
    ident = "frozen_inputs_unmodified"
    drift = []
    if harness.bundle_ledger_sha256 != bound.ledger_source_sha256:
        drift.append("environment/ledger_source.json")
    if harness.bundle_instances_sha256 != bound.instances_sha256:
        drift.append("environment/instances.json")
    if harness.bundle_minter_sha256 != bound.minter_sha256:
        drift.append("environment/mint_store.py")
    if drift:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "frozen input(s) do not match the bound digests: " + ", ".join(drift),
            drift,
        )
    return _pass(ident, "all three frozen inputs match their bound digests", 3)


def check_store_materialised(harness: Harness, bound: Bound) -> Outcome:
    """EFFECT. The minter's own run produced a complete, single-genesis, total chain."""
    ident = "store_materialised"
    atoms = harness.atoms()
    attestations = harness.attestations()
    if len(atoms) != bound.atom_count:
        return _fail(
            ident,
            REASON_STORE_NOT_MATERIALISED,
            "the harness store carries " + str(len(atoms)) + " atom(s) and the bound count is "
            + str(bound.atom_count),
            len(atoms),
        )
    if len(attestations) != bound.atom_count:
        return _fail(
            ident,
            REASON_STORE_NOT_MATERIALISED,
            "the harness store carries " + str(len(attestations)) + " attestation(s) and the "
            "bound count is " + str(bound.atom_count),
            len(attestations),
        )
    genesis = [row for row in attestations if str(row["prev_seal"]) == bound.genesis_prev_seal]
    if len(genesis) != 1:
        return _fail(
            ident,
            REASON_STORE_NOT_MATERIALISED,
            "the chain carries " + str(len(genesis)) + " genesis attestation(s) and exactly one "
            "is required for the ordering predicate to resolve",
            len(genesis),
        )
    order = harness.realised_order()
    if len(order) != bound.atom_count or set(order) != set(atoms):
        return _fail(
            ident,
            REASON_STORE_NOT_MATERIALISED,
            "walking the chain from genesis visits " + str(len(order)) + " atom(s) and does not "
            "cover the store exactly once",
            len(order),
        )
    return _pass(
        ident,
        "the harness minted " + str(bound.atom_count) + " atoms and one total chain from a "
        "single genesis",
        bound.atom_count,
    )


def check_attestation_order_realised(harness: Harness, bound: Bound) -> Outcome:
    """ORDERING. The submitted order is the order the seal chain realises.

    This is the checker the ordering discovery value gates. A submission that assumes the
    lexicographic identifier order, or any other order the store did not realise, fails here
    and the reward is zero with `attestation-order-unrealised`.
    """
    ident = "attestation_order_realised"
    realised = harness.realised_order()
    submission = harness.submission()
    claimed = submission.get("attestation_order")
    if not isinstance(claimed, list) or not claimed:
        return _fail(ident, REASON_ORDER_UNREALISED, "the submission declares no attestation_order")
    if len(claimed) != len(realised):
        return _fail(
            ident,
            REASON_ORDER_UNREALISED,
            "the submitted order carries " + str(len(claimed)) + " entries and the realised "
            "chain carries " + str(len(realised)),
        )
    if [str(value) for value in claimed] != realised:
        first = next(
            (
                index
                for index, value in enumerate(claimed)
                if str(value) != realised[index]
            ),
            0,
        )
        return _fail(
            ident,
            REASON_ORDER_UNREALISED,
            "the submitted order first diverges from the realised chain at position "
            + str(first) + ", where it names " + str(claimed[first]) + " and the chain seals "
            + realised[first],
            first,
        )
    terminal = submission.get("terminal_atom_id")
    if str(terminal) != realised[-1]:
        return _fail(
            ident,
            REASON_ORDER_UNREALISED,
            "the submission names " + str(terminal) + " as the terminal atom and the chain ends "
            "at " + realised[-1],
        )
    return _pass(ident, "the submitted order is the order the seal chain realises", len(realised))


def check_seal_chain_derived_from_state(harness: Harness, bound: Bound) -> Outcome:
    """DIVERGENCE. Every submitted seal is recomputed from harness-minted state, link by link.

    An attestation that was invented rather than derived diverges here even when the order
    around it is right, because each seal is a function of the atom's own cover digest and of
    the seal before it.
    """
    ident = "seal_chain_derived_from_state"
    submission = harness.submission()
    claimed = submission.get("seal_chain")
    realised = harness.realised_order()
    atoms = harness.atoms()
    if not isinstance(claimed, list):
        return _fail(ident, REASON_SEAL_FABRICATED, "the submission declares no seal_chain")
    if len(claimed) != len(realised):
        return _fail(
            ident,
            REASON_SEAL_FABRICATED,
            "the submitted seal chain carries " + str(len(claimed)) + " link(s) and the realised "
            "chain carries " + str(len(realised)),
        )
    previous = bound.genesis_prev_seal
    for index, ident_atom in enumerate(realised):
        expected = attestation_seal(
            ident_atom, cover_sha256(atoms[ident_atom]["selected_sets"]), previous
        )
        if str(claimed[index]) != expected:
            return _fail(
                ident,
                REASON_SEAL_FABRICATED,
                "link " + str(index) + " of the submitted seal chain is not the seal the built "
                "state derives for " + ident_atom,
                index,
            )
        previous = expected
    if previous != harness.seal_sequence()[-1]:
        return _fail(
            ident,
            REASON_SEAL_FABRICATED,
            "the recomputed chain does not terminate at the seal the harness store carries",
        )
    return _pass(ident, "every submitted seal is derived from harness-minted state", len(claimed))


def check_attested_closure_complete(harness: Harness, bound: Bound) -> Outcome:
    """ABSENCE. No atom of the store is missing from the digest table and none is invented."""
    ident = "attested_closure_complete"
    submission = harness.submission()
    table = submission.get("atom_digests")
    if not isinstance(table, dict):
        return _fail(ident, REASON_CLOSURE_INCOMPLETE, "the submission declares no atom_digests")
    present = set(harness.atoms())
    claimed = set(str(key) for key in table)
    missing = sorted(present - claimed)
    invented = sorted(claimed - present)
    if missing or invented:
        return _fail(
            ident,
            REASON_CLOSURE_INCOMPLETE,
            "the digest table is missing " + str(len(missing)) + " atom(s) " + str(missing[:4])
            + " and introduces " + str(len(invented)) + " atom(s) the store does not carry "
            + str(invented[:4]),
            {"missing": missing, "invented": invented},
        )
    return _pass(ident, "the digest table covers the store exactly", len(claimed))


def check_digest_recursion_holds(harness: Harness, bound: Bound) -> Outcome:
    """INVARIANT. Every submitted atom digest equals the verifier's own recursive fold.

    This is the checker the digest discovery value gates for the whole closure. Folding input
    identifiers instead of input digests, folding in identifier order instead of realised
    order, or including one of the three excluded record fields all land here.
    """
    ident = "digest_recursion_holds"
    submission = harness.submission()
    table = submission.get("atom_digests") or {}
    expected = harness.atom_digests()
    if not expected:
        return _fail(
            ident,
            REASON_PREIMAGE_WRONG,
            "the harness could not fold the store's own digests, so no comparison is possible",
        )
    wrong = [
        ident_atom
        for ident_atom in harness.realised_order()
        if str(table.get(ident_atom, "")) != expected[ident_atom]
    ]
    if wrong:
        return _fail(
            ident,
            REASON_PREIMAGE_WRONG,
            str(len(wrong)) + " atom digest(s) do not satisfy the fixed preimage recursion, "
            "first at " + wrong[0],
            wrong[:6],
        )
    return _pass(ident, "every atom digest satisfies the fixed preimage recursion", len(expected))


def check_terminal_atom_digest_matches(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The declared terminal atom digest is the verifier's own recomputation."""
    ident = "terminal_atom_digest_matches"
    submission = harness.submission()
    terminal = harness.terminal_atom_id()
    expected = harness.atom_digests().get(terminal, "")
    claimed = str(submission.get("terminal_atom_digest", ""))
    if not expected:
        return _fail(
            ident,
            REASON_TERMINAL_DIGEST,
            "the harness could not fold a digest for the terminal atom",
        )
    if claimed != expected:
        matching = sum(1 for a, b in zip(claimed, expected) if a == b)
        return _fail(
            ident,
            REASON_TERMINAL_DIGEST,
            "the declared terminal digest is not the fold over the provenance closure of "
            + terminal + "; " + str(matching) + " of " + str(len(expected))
            + " hexadecimal characters agree",
            matching,
        )
    return _pass(ident, "the terminal atom digest is the fold over its provenance closure", terminal)


def check_graded_cover_feasible(harness: Harness, bound: Bound) -> Outcome:
    """INVARIANT. The cover is for the right instance, is drawn from it, and covers it.

    The graded instance is the one the LAST atom in the realised attestation order attests, so
    a submission that recovered the wrong order also names the wrong instance and lands here
    even if its own arithmetic was faultless.
    """
    ident = "graded_cover_feasible"
    submission = harness.submission()
    instance = graded_instance(harness)
    if instance is None:
        return _fail(ident, REASON_COVER_INFEASIBLE, "the harness could not resolve a graded instance")
    if str(submission.get("graded_instance_id", "")) != str(instance["instance_id"]):
        return _fail(
            ident,
            REASON_COVER_INFEASIBLE,
            "the submission names instance " + str(submission.get("graded_instance_id"))
            + " and the terminal atom of the realised chain attests " + str(instance["instance_id"]),
        )
    cover = submission.get("submitted_cover")
    if not isinstance(cover, list) or not cover:
        return _fail(ident, REASON_COVER_INFEASIBLE, "the submission declares no submitted_cover")
    identifiers = [int(row["id"]) for row in instance["sets"]]
    try:
        values = [int(value) for value in cover]
    except (TypeError, ValueError):
        return _fail(ident, REASON_COVER_INFEASIBLE, "the submitted cover is not a list of integers")
    if len(set(values)) != len(values):
        return _fail(ident, REASON_COVER_INFEASIBLE, "the submitted cover repeats a set identifier")
    outside = sorted(set(values) - set(identifiers))
    if outside:
        return _fail(
            ident,
            REASON_COVER_INFEASIBLE,
            "the submitted cover names set(s) " + str(outside) + " the graded instance does not carry",
        )
    members = {int(row["id"]): row["members"] for row in instance["sets"]}
    covered = set()
    for value in values:
        covered.update(int(element) for element in members[value])
    universe = set(range(int(instance["universe_size"])))
    if covered != universe:
        return _fail(
            ident,
            REASON_COVER_INFEASIBLE,
            "the submitted cover leaves " + str(len(universe - covered)) + " element(s) uncovered",
            sorted(universe - covered),
        )
    weight = submitted_cover_weight(harness)
    reported = submission.get("reported_cover_weight")
    if not isinstance(reported, int) or abs(int(reported) - int(weight)) > bound.weight_tolerance:
        return _fail(
            ident,
            REASON_COVER_INFEASIBLE,
            "the submission reports cover weight " + str(reported) + " and the verifier "
            "recomputed " + str(weight),
        )
    return _pass(ident, "the submitted cover is feasible on the graded instance", weight)
