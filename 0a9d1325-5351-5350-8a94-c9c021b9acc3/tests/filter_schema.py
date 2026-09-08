#!/usr/bin/env python3
"""The curation-filter schema and the filter runner. BOTH surfaces run this file.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time. A chain that `validate` accepts here is a
chain the verifier will run, and `apply` here selects exactly the block set the
verifier will train on.

THIS MODULE IS A HYGIENE GATE AND A SELECTOR. It is not a grade. `validate`
answers one question -- is this a well-formed filter chain -- and its answer is
yes or no; a malformed chain is REFUSED at reward 0.0 with a machine-readable
reason and never scaled. `apply` answers a second question -- which blocks does
this chain admit -- and returns a set of block ids and a report. Nothing either
function computes reaches the reward except through which blocks got trained on.

                    THE RUNNER FAILS OPEN. READ THIS.

A predicate whose `field` is not a key of a register row RESOLVES NOTHING. It
does not raise, it does not refuse, and it does not drop the block. It matches
EVERYTHING, which makes it a no-op, and the chain continues. A chain in which
every predicate names an absent field therefore admits the pool UNCHANGED while
`apply` returns success and reports that it ran.

That is deliberate and it is the shipped behaviour on both surfaces. The report
`apply` returns carries `rules_unresolved` and `admitted`, so the condition is
observable to anyone who looks; it is silent, not hidden. The shipped
environment/default_filter.json is exactly such a chain, and it is the CONTROL
arm of the reward scale for that reason.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer-curation-filter/v1"

# Comparison operators the runner knows. An unknown operator is a REFUSAL, not a
# no-op: an operator is the runner's own vocabulary, whereas a field name is a
# property of the register and may legitimately be absent.
OPS = ("ge", "gt", "le", "lt", "ne")

MAX_RULES = 16

# The fields the shipped register actually carries. This tuple is DOCUMENTATION
# for a reader; `apply` does not consult it and naming a field outside it is not
# an error. See the fail-open note above.
KNOWN_FIELDS = (
    "block_id",
    "tokens",
    "distinct_token_ratio",
    "top1_token_share",
    "bigram_coherence",
    "unigram_logprob",
    "newline_share",
    "doc_boundaries",
)


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _number(where: str, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal("filter-value-not-a-number",
                      f"{where} is {value!r}, which is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise Refusal("filter-value-not-a-number", f"{where} is {value!r}, which is not finite")
    return value


def validate(doc) -> dict:
    """Return the chain if it is well formed. Otherwise raise Refusal."""
    if not isinstance(doc, dict):
        raise Refusal("filter-not-an-object",
                      f"the submission parsed to {type(doc).__name__}, not an object")

    unknown = sorted(set(doc) - {"schema", "rules", "notes"})
    if unknown:
        raise Refusal("filter-key-unknown", f"filter carries unknown key(s) {', '.join(unknown)}")

    for key in ("schema", "rules"):
        if key not in doc:
            raise Refusal("filter-key-missing", f"filter is missing {key}")

    if doc["schema"] != SCHEMA_ID:
        raise Refusal("filter-schema-unrecognised",
                      f"filter.schema is {doc['schema']!r}, expected {SCHEMA_ID!r}")

    rules = doc["rules"]
    if not isinstance(rules, list):
        raise Refusal("filter-rules-not-a-list",
                      f"filter.rules is {type(rules).__name__}, not a list")
    if len(rules) > MAX_RULES:
        raise Refusal("filter-too-many-rules",
                      f"filter.rules holds {len(rules)} predicates, the ceiling is {MAX_RULES}")

    for i, rule in enumerate(rules):
        where = f"filter.rules[{i}]"
        if not isinstance(rule, dict):
            raise Refusal("filter-rule-not-an-object",
                          f"{where} is {type(rule).__name__}, not an object")
        extra = sorted(set(rule) - {"field", "op", "value"})
        if extra:
            raise Refusal("filter-key-unknown", f"{where} carries unknown key(s) {', '.join(extra)}")
        for key in ("field", "op", "value"):
            if key not in rule:
                raise Refusal("filter-key-missing", f"{where} is missing {key}")
        if not isinstance(rule["field"], str) or not rule["field"]:
            raise Refusal("filter-field-not-a-string",
                          f"{where}.field is {rule['field']!r}, which is not a non-empty string")
        if rule["op"] not in OPS:
            raise Refusal("filter-op-unknown",
                          f"{where}.op is {rule['op']!r}, not one of {list(OPS)}")
        _number(f"{where}.value", rule["value"])

    return doc


def load(path) -> dict:
    """Read and validate a chain from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no filter chain at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("filter-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("filter-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("filter-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)


def load_register(path) -> list:
    """Read the source register. One JSON object per line, pool order."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply(doc: dict, register: list) -> tuple:
    """Return (admitted_block_ids, report).

    Conjunctive: a block is admitted when every predicate that RESOLVED accepts
    it. A predicate that did not resolve -- whose field is absent from the row,
    or whose stored value is not a number -- accepts everything, which is the
    fail-open behaviour documented at the top of this file.
    """
    admitted = []
    unresolved = set()
    resolved = set()
    for row in register:
        keep = True
        for i, rule in enumerate(doc["rules"]):
            field, op = rule["field"], rule["op"]
            value = float(rule["value"])
            if field not in row:
                unresolved.add(i)
                continue
            got = row[field]
            if isinstance(got, bool) or not isinstance(got, (int, float)):
                unresolved.add(i)
                continue
            resolved.add(i)
            got = float(got)
            if op == "ge":
                ok = got >= value
            elif op == "gt":
                ok = got > value
            elif op == "le":
                ok = got <= value
            elif op == "lt":
                ok = got < value
            else:
                ok = got != value
            if not ok:
                keep = False
                break
        if keep:
            admitted.append(int(row["block_id"]))

    report = {
        "pool_blocks": len(register),
        "rules_total": len(doc["rules"]),
        "rules_resolved": len(resolved),
        "rules_unresolved": len(sorted(unresolved)),
        "unresolved_rule_indices": sorted(unresolved),
        "unresolved_rule_fields": sorted({doc["rules"][i]["field"] for i in unresolved}),
        "admitted": len(admitted),
        "rejected": len(register) - len(admitted),
        "pool_passed_through_unchanged": len(admitted) == len(register),
    }
    return admitted, report
