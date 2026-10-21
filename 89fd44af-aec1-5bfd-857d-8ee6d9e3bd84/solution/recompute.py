#!/usr/bin/env python3
"""Generate solution/rubrics.json from solution/grounding.yaml.

FORGE 10f: rubrics.json is generated, never hand-authored. This script reads the frozen
criterion literals from grounding.yaml and emits rubrics.json. It never invokes a model, a
network, a clock, a locale, or a random source, and it reads no environment variable, host
name, working directory or temporary path. Its output is a pure function of the bytes of
grounding.yaml, so regenerating it under two different host identities yields byte equality.

The YAML subset accepted here is deliberately tiny and fail-closed: PyYAML is not present in
the graded image, and a hand-written strict parser removes the dependency rather than
introducing one. Anything outside the accepted subset raises.

Usage:  python3 recompute.py            # writes rubrics.json beside this file
        python3 recompute.py --check    # exits 1 if rubrics.json differs from regeneration
"""
from __future__ import annotations

import json
import os.path
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GROUNDING = os.path.join(HERE, "grounding.yaml")
RUBRICS = os.path.join(HERE, "rubrics.json")

SCHEMA_FIELDS = ("id", "dimension", "weight", "evaluation_target",
                 "criterion", "judgment", "evidence", "mode")
MODES = ("compiled", "judged")
OUTCOME_CLASSES = ("VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE")

SOURCE_HEADER = (
    "GENERATED FILE. Do not edit. Produced by solution/recompute.py from "
    "solution/grounding.yaml. Edit grounding.yaml and regenerate."
)


class GroundingError(Exception):
    """Raised when grounding.yaml leaves the accepted subset. Fails closed."""


def _strip_comment(line: str) -> str:
    """Remove a trailing comment only when the '#' starts a token."""
    out, in_s, q = [], False, ""
    for i, ch in enumerate(line):
        if in_s:
            out.append(ch)
            if ch == q:
                in_s = False
            continue
        if ch in "\"'":
            in_s, q = True, ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _scalar(raw: str):
    """Convert an unquoted scalar deterministically. No locale, no eval."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s in ("true", "false"):
        return s == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return s
    return f


def parse_grounding(text: str) -> dict:
    """Parse the accepted subset in a single indent-driven pass.

    Accepted, and nothing else:
        key: value                 (indent 0)  top-level scalar
        key:                       (indent 0)  top-level list, entries at indent 2
          - scalar
        items:                     (indent 0)  list of item blocks
          - key: value             (indent 2)  starts a new item
            key: value             (indent 4)  item scalar
            key:                   (indent 4)  item list, entries at indent 6
              - scalar
    Anything else raises, so the generator fails closed rather than guessing.
    """
    doc: dict = {}
    items: list = []
    in_items = False
    top_list_key = None
    cur = None
    cur_list_key = None

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()

        if indent == 0:
            in_items = False
            top_list_key = None
            cur = None
            cur_list_key = None
            if body == "items:":
                in_items = True
                continue
            if ":" not in body:
                raise GroundingError(f"line {lineno}: not a mapping: {body!r}")
            k, _sep, v = body.partition(":")
            k = k.strip()
            if v.strip() == "":
                doc[k] = []
                top_list_key = k
            else:
                doc[k] = _scalar(v)
            continue

        if not in_items:
            if top_list_key is None:
                raise GroundingError(f"line {lineno}: indented line with no key")
            if not body.startswith("- "):
                raise GroundingError(f"line {lineno}: expected list entry: {body!r}")
            doc[top_list_key].append(_scalar(body[2:]))
            continue

        if indent == 2:
            if not body.startswith("- "):
                raise GroundingError(f"line {lineno}: expected item start: {body!r}")
            rest = body[2:]
            if ":" not in rest:
                raise GroundingError(f"line {lineno}: item must start with a mapping")
            cur = {}
            items.append(cur)
            cur_list_key = None
            k, _sep, v = rest.partition(":")
            cur[k.strip()] = _scalar(v)
            continue

        if cur is None:
            raise GroundingError(f"line {lineno}: value outside an item")

        if indent == 4:
            if ":" not in body:
                raise GroundingError(f"line {lineno}: not a mapping: {body!r}")
            k, _sep, v = body.partition(":")
            k = k.strip()
            if v.strip() == "":
                cur[k] = []
                cur_list_key = k
            else:
                cur[k] = _scalar(v)
                cur_list_key = None
            continue

        if indent == 6:
            if cur_list_key is None:
                raise GroundingError(f"line {lineno}: list entry with no key")
            if not body.startswith("- "):
                raise GroundingError(f"line {lineno}: expected list entry: {body!r}")
            cur[cur_list_key].append(_scalar(body[2:]))
            continue

        raise GroundingError(f"line {lineno}: unexpected indent {indent}: {body!r}")

    doc["items"] = items
    return doc


def build(doc: dict) -> dict:
    floor = doc["compilation_floor"]
    vocab = list(doc["evaluation_target_vocabulary"])
    items_out = []
    residue = {}
    total_w = 0.0
    compiled_w = 0.0

    for it in doc["items"]:
        missing = [f for f in SCHEMA_FIELDS if f not in it]
        if missing:
            raise GroundingError(f"item {it.get('id')!r} missing fields: {missing}")
        if it["mode"] not in MODES:
            raise GroundingError(f"item {it['id']!r} mode {it['mode']!r} not in {MODES}")
        if it["dimension"] not in OUTCOME_CLASSES:
            raise GroundingError(
                f"item {it['id']!r} outcome class {it['dimension']!r} not in {OUTCOME_CLASSES}")
        if it["evaluation_target"] not in vocab:
            raise GroundingError(
                f"item {it['id']!r} evaluation_target {it['evaluation_target']!r} not in vocabulary")
        if not it["evidence"]:
            raise GroundingError(f"item {it['id']!r} names no evidence")
        w = float(it["weight"])
        total_w += w
        if it["mode"] == "compiled":
            compiled_w += w
        else:
            r = it.get("residue")
            if not r:
                raise GroundingError(
                    f"judged item {it['id']!r} names no semantic residue (FORGE 10f)")
            residue[it["id"]] = r
        items_out.append({f: (list(it[f]) if f == "evidence" else it[f]) for f in SCHEMA_FIELDS})

    share = compiled_w / total_w if total_w else 0.0
    if share < float(floor):
        raise GroundingError(
            f"compiled weight share {share:.6f} is below compilation_floor {floor}")

    return {
        "_source": SOURCE_HEADER,
        "schema_version": doc["schema_version"],
        "bundle_uuid": doc["bundle_uuid"],
        "family": doc["family"],
        "present": True,
        "reference_answer": doc["reference_answer"],
        "compilation_floor": floor,
        "compilation_floor_source": doc["compilation_floor_source"],
        "evaluation_target_vocabulary": vocab,
        "compiled_weight": compiled_w,
        "total_weight": total_w,
        "compiled_share": round(share, 6),
        "item_count": len(items_out),
        "items": items_out,
        "judged_residue": residue,
    }


def render(doc: dict) -> str:
    return json.dumps(build(doc), indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def main(argv) -> int:
    with open(GROUNDING, encoding="utf-8") as f:
        text = f.read()
    out = render(parse_grounding(text))
    if "--check" in argv:
        if not os.path.isfile(RUBRICS):
            sys.stderr.write("rubrics.json absent\n")
            return 1
        with open(RUBRICS, encoding="utf-8") as f:
            have = f.read()
        if have != out:
            sys.stderr.write("REGENERATION DRIFT: rubrics.json differs from grounding.yaml\n")
            return 1
        sys.stdout.write("rubrics.json matches grounding.yaml\n")
        return 0
    with open(RUBRICS, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    sys.stdout.write(f"wrote {RUBRICS}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
