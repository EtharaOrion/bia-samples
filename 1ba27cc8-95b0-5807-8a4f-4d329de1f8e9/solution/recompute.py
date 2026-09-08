#!/usr/bin/env python3
"""Generate solution/rubrics.json from solution/grounding.yaml.

FORGE 10f. This generator hand-authors nothing: every `criterion` string it emits is a
frozen literal of solution/grounding.yaml. It never invokes a model, a network, a clock,
a locale or a random source. The YAML reader is bundled rather than imported so that the
parse cannot vary with what a host happens to have installed.

Item schema is the closed 8-field 9g schema:
    id, dimension, weight, evaluation_target, criterion, judgment, evidence, mode
The 10f outcome classification is carried in the parallel top-level `outcome_classification`
map so that no item grows a ninth field.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
GROUNDING = HERE / "grounding.yaml"
RUBRICS = HERE / "rubrics.json"

ITEM_FIELDS = ("id", "dimension", "weight", "evaluation_target",
               "criterion", "judgment", "evidence", "mode")
OUTCOME_CLASSES = ("VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE")


def _strip(line: str) -> str:
    return line.split(" #")[0].rstrip() if " #" in line else line.rstrip()


def _scalar(text: str):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_scalar(p) for p in inner.split(",")]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse(lines, index, indent):
    if index < len(lines) and lines[index].lstrip().startswith("- "):
        out = []
        while index < len(lines):
            line = lines[index]
            if _indent(line) != indent or not line.lstrip().startswith("- "):
                break
            rest = line.lstrip()[2:]
            block = [" " * (indent + 2) + rest]
            index += 1
            while index < len(lines) and _indent(lines[index]) > indent:
                block.append(lines[index])
                index += 1
            value, _ = _parse(block, 0, indent + 2)
            out.append(value)
        return out, index
    out = {}
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if _indent(line) < indent:
            break
        key, _, rest = line.strip().partition(":")
        rest = rest.strip()
        if rest in (">-", ">", "|", "|-"):
            index += 1
            parts = []
            while index < len(lines) and (not lines[index].strip() or _indent(lines[index]) > indent):
                parts.append(lines[index].strip())
                index += 1
            out[key] = " ".join(p for p in parts if p)
        elif rest == "":
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index < len(lines) and _indent(lines[index]) > indent:
                value, index = _parse(lines, index, _indent(lines[index]))
                out[key] = value
            else:
                out[key] = None
        else:
            out[key] = _scalar(rest)
            index += 1
    return out, index


def load_grounding(path: pathlib.Path) -> dict:
    lines = [_strip(l) for l in path.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#")]
    document, _ = _parse(lines, 0, 0)
    return document


def build(grounding: dict) -> dict:
    items = []
    outcomes = {}
    for raw in grounding["items"]:
        missing = [f for f in ITEM_FIELDS if f not in raw]
        if missing:
            raise SystemExit("item %r missing closed-schema fields %s" % (raw.get("id"), missing))
        if raw["outcome"] not in OUTCOME_CLASSES:
            raise SystemExit("item %r has outcome %r outside the closed class set"
                             % (raw["id"], raw["outcome"]))
        if raw["mode"] not in ("compiled", "judged"):
            raise SystemExit("item %r has mode %r" % (raw["id"], raw["mode"]))
        items.append({field: raw[field] for field in ITEM_FIELDS})
        outcomes[raw["id"]] = raw["outcome"]
    total = sum(i["weight"] for i in items)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    share = compiled / total if total else 0.0
    floor = grounding["compilation_floor"]
    if share < floor:
        raise SystemExit("compiled weight share %.4f is below compilation_floor %.4f"
                         % (share, floor))
    return {
        "generated_by": "solution/recompute.py",
        "generated_from": "solution/grounding.yaml",
        "generated_note": "GENERATED SECTION. DO NOT HAND-EDIT.",
        "schema": grounding["schema"],
        "reference_answer": grounding["reference_answer"],
        "compiled_target": grounding["compiled_target"],
        "compilation_floor": floor,
        "compiled_weight_share": share,
        "items": items,
        "outcome_classification": outcomes,
        "judged_residue": grounding.get("judged_residue") or [],
    }


def main() -> int:
    document = build(load_grounding(GROUNDING))
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(RUBRICS, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    sys.stdout.write("wrote %s: %d items, compiled share %.4f\n"
                     % (RUBRICS.name, len(document["items"]), document["compiled_weight_share"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
