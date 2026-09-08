#!/usr/bin/env python3
"""The rejecting half of the conformance protocol (FORGE item 328).

conformance_rubric.py runs the CLEAN half only and says so. A check proven on one half
cannot be told apart from a check that is hardcoded to pass, and item 328 caps a one-sided
proof at BLOCK:INVALID_TASK. This module supplies the other half: for each check it copies
the bundle, plants exactly the defect that check exists to catch, and requires the check to
FLIP to FAIL. A plant that does not flip its check is reported INERT, never as a pass.
"""
from __future__ import annotations
import importlib, json, pathlib, shutil, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(root):
    for m in ("conformance_rubric", "rubric_verdicts_jsonl", "account"):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(root/"tests"))
    return importlib.import_module("conformance_rubric")


def _first(root, pat):
    xs = sorted(root.glob(pat))
    if not xs: raise FileNotFoundError(pat)
    return xs[0]


# ---- plants: each returns None if it cannot be applied to this bundle
def p_c1(r):
    """C1 compares regenerated verdict BYTES under two host identities.

    Appending to the generator is dead code: it ends in `raise SystemExit(main())`. The plant
    therefore WRAPS it - run the original to completion, then make the written bytes depend on
    $HOSTNAME, which is what host-dependence actually looks like in this pipeline.
    """
    g = r / "tests" / "regenerate_rubric_verdicts.py"
    orig = g.read_text()
    wrapper = [
        "import os as _os, json as _json, pathlib as _pl",
        "try:",
        "    exec(compile(_ORIG, 'orig', 'exec'), {'__name__': '__main__', '__file__': __file__})",
        "except SystemExit:",
        "    pass",
        "_root = _pl.Path(__file__).resolve().parent.parent",
        "for _p in _root.glob('trajectories/*/*/rubric_verdicts.json'):",
        "    _d = _json.loads(_p.read_text())",
        "    _d['summary'] = (_d.get('summary') or '') + _os.environ.get('HOSTNAME', '')",
        "    _p.write_text(_json.dumps(_d, indent=2, sort_keys=True) + chr(10))",
    ]
    g.write_text("_ORIG = " + repr(orig) + chr(10) + chr(10).join(wrapper) + chr(10))

def p_c2(r):
    f = r/"solution"/"rubrics.json"; d = json.loads(f.read_text())
    for i in d["items"]:
        if i["mode"] == "compiled": i["id"] = i["id"] + "_renamed"; break
    f.write_text(json.dumps(d, indent=1))

def p_c3(r):
    """The floor must exceed any achievable share; this corpus reaches 1.0, so 0.999 passes."""
    f = r / "solution" / "rubrics.json"
    d = json.loads(f.read_text())
    d["compilation_floor"] = 1.01
    f.write_text(json.dumps(d, indent=1))

def p_c4(r):
    f = r/"solution"/"rubrics.json"; d = json.loads(f.read_text())
    d["items"][0]["extra_field"] = "planted"; f.write_text(json.dumps(d, indent=1))

def p_c5(r):
    """1ba27cc8 keys outcome_classification at TOP LEVEL; 89fd44af uses a per-item vocabulary."""
    f = r / "solution" / "rubrics.json"
    d = json.loads(f.read_text())
    oc = d.get("outcome_classification")
    if isinstance(oc, dict) and oc:
        oc[sorted(oc)[0]] = "PLANTED_NOT_A_CLASS"
    else:
        d["items"][0]["evaluation_target"] = "PLANTED_NOT_IN_VOCABULARY"
    f.write_text(json.dumps(d, indent=1))

def p_c6(r):
    f = r/"solution"/"rubrics.json"; d = json.loads(f.read_text())
    for i in d["items"]:
        if i.get("evidence"): i["evidence"] = ["tests/does_not_exist_planted.py"]; break
    f.write_text(json.dumps(d, indent=1))

def p_c7(r):
    g = r/"tests"/"regenerate_rubric_verdicts.py"
    g.write_text("import random\n" + g.read_text())

def p_c8(r):
    f = r/"tests"/"rubric_judgment_grounding.json"; g = json.loads(f.read_text())
    k = sorted(g)[0]; e = g[k]
    e["account_channel"] = "IN_FLIGHT_ONLY"
    rid = sorted(e["items"])[0]
    e["items"][rid] = {"basis": "NARRATION_QUOTE_HAND_SELECTED", "verdict": False,
                       "evidence": "planted: scored false on an absent narration"}
    f.write_text(json.dumps(g, indent=1, sort_keys=True))

def p_c9(r):
    m = _load(r)
    gp = r/"tests"/"rubric_gate.py"
    gp.write_text(gp.read_text().replace("def apply_rubric_veto",
        "def apply_rubric_veto(*a, **k):\n    return 1.0, None, 1\n\n\ndef _unused_apply_rubric_veto"))

def p_c10(r):
    """C10 reads the GROUNDING and only items whose basis is NARRATION_QUOTE_HAND_SELECTED.
    Planting into a committed rubric_verdicts.json never reaches it."""
    f = r / "tests" / "rubric_judgment_grounding.json"
    g = json.loads(f.read_text())
    for k in sorted(g):
        for rid, i in sorted(g[k]["items"].items()):
            if i.get("basis") == "NARRATION_QUOTE_HAND_SELECTED":
                i["evidence"] = ('the agent wrote "ZZ PLANTED FABRICATED QUOTE ABSENT FROM '
                                 'THE RECORD ZZ" in its account')
                f.write_text(json.dumps(g, indent=1, sort_keys=True))
                return
    raise RuntimeError("no narration-basis item to plant into")

def p_c11(r):
    f = r/"tests"/"rubrics.jsonl"; L = [l for l in f.read_text().splitlines() if l.strip()]
    d = json.loads(L[0]); d["weight"] = 1.0; L[0] = json.dumps(d)
    f.write_text("\n".join(L) + "\n")

def p_c12(r):
    f = _first(r, "trajectories/*/*/rubric_verdicts.json"); d = json.loads(f.read_text())
    d["verdicts"][sorted(d["verdicts"])[0]]["evidence"] = "PLANTED HAND EDIT"
    f.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")

def p_c13(r):
    f = r/"tests"/"rubrics.jsonl"; L = [l for l in f.read_text().splitlines() if l.strip()]
    f.write_text("\n".join(L[1:]) + "\n")

def p_c14(r):
    f = r/"tests"/"rubric_judgment_grounding.json"; g = json.loads(f.read_text())
    k = sorted(g)[0]; g[k]["items"].pop(sorted(g[k]["items"])[0])
    f.write_text(json.dumps(g, indent=1, sort_keys=True))

def p_c15(r):
    f = r/"tests"/"rubric_judgment_grounding.json"; g = json.loads(f.read_text())
    k = sorted(g)[0]; g[k]["account_chars"] = g[k]["account_chars"] + 999
    f.write_text(json.dumps(g, indent=1, sort_keys=True))

def p_c16(r):
    f = r/"trajectories"/"rubric_verdicts.jsonl"
    L = f.read_text().splitlines(); d = json.loads(L[0]); d["evidence"] = "PLANTED"
    L[0] = json.dumps(d, sort_keys=True); f.write_text("\n".join(L) + "\n")


PLANTS = {"C1": p_c1, "C2": p_c2, "C3": p_c3, "C4": p_c4, "C5": p_c5, "C6": p_c6, "C7": p_c7,
          "C8": p_c8, "C9": p_c9, "C10": p_c10, "C11": p_c11, "C12": p_c12, "C13": p_c13,
          "C14": p_c14, "C15": p_c15, "C16": p_c16}


def main():
    base = _load(ROOT)
    print("%-46s %s" % ("CHECK", "PLANTED HALF (must FAIL)"))
    inert, ran = [], 0
    for label, fn in base.CHECKS:
        tag = label.split()[0]
        plant = PLANTS.get(tag)
        if plant is None:
            print("  %-44s %s" % (label, "NO PLANT DEFINED")); continue
        tmp = pathlib.Path(tempfile.mkdtemp())/"b"
        try:
            shutil.copytree(ROOT, tmp)
            plant(tmp)
            mod = _load(tmp)
            target = dict((l.split()[0], f) for l, f in mod.CHECKS)[tag]
            try:
                passed, msg = target(tmp)
            except Exception as e:
                passed, msg = False, "raised %s" % type(e).__name__
            ran += 1
            if passed is True:
                inert.append(tag); print("  %-44s %s" % (label, "DID NOT FIRE - INERT"))
            else:
                print("  %-44s %s  %s" % (label, "FAIL (correct)", str(msg)[:56]))
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
            _load(ROOT)
    print("\n  planted %d checks; %d fired; inert: %s" % (ran, ran-len(inert), inert or "none"))
    return 1 if inert else 0


if __name__ == "__main__":
    raise SystemExit(main())
