"""Emit trajectories/rubric_verdicts.jsonl - the single findable proof that rubrics exist AND were judged.

WHY A BUNDLE-LEVEL .jsonl:
  tests/rubrics.jsonl is one line per rubric. Its natural counterpart is one line per JUDGEMENT.
  A reviewer opening the bundle can answer "are there rubrics, and was every one judged?" from one file.

WHY IT IS NOT A SECOND SOURCE OF TRUTH:
  It is a pure VIEW over the per-attempt rubric_verdicts.json files. check_binding() below proves every
  line reproduces its source entry exactly and that the line set is exactly attempts x rubrics. If the
  view and the source ever diverge, the check fails. Nothing can be true in one and false in the other.

WHY trajectories/ AND NOT tests/:
  `trajectories` is in excluded_components of the canonical hash domain, so emitting this file is
  identity-neutral: the bundle uuid does not move. Writing it under tests/ would move the content hash.
"""
import json, pathlib, sys

FILENAME = "rubric_verdicts.jsonl"

def attempts(root: pathlib.Path):
    return sorted((p.parent for p in root.glob("trajectories/*/*/rubric_verdicts.json")),
                  key=lambda p: str(p))

def rubric_ids(root: pathlib.Path):
    return [json.loads(l)["id"] for l in (root/"tests"/"rubrics.jsonl").read_text().splitlines() if l.strip()]

def rows(root: pathlib.Path):
    """Deterministic, sorted: attempt path then rubric id. No clock, no locale, no hostname."""
    out = []
    for d in attempts(root):
        rel = f"{d.parent.name}/{d.name}"
        doc = json.loads((d/"rubric_verdicts.json").read_text(encoding="utf-8"))
        for rid in sorted(doc["verdicts"]):
            e = doc["verdicts"][rid]
            out.append({"attempt": rel, "id": rid,
                        "pass": e.get("pass"), "evidence": e.get("evidence")})
    return out

def render(root: pathlib.Path) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows(root))

def emit(root: pathlib.Path) -> pathlib.Path:
    p = root/"trajectories"/FILENAME
    p.write_text(render(root), encoding="utf-8")
    return p

def check_binding(root: pathlib.Path):
    """C16. Returns (ok, message). The whole reason the view is safe to ship."""
    p = root/"trajectories"/FILENAME
    if not p.exists():
        return False, "absent: trajectories/%s" % FILENAME
    if p.read_text(encoding="utf-8") != render(root):
        return False, "jsonl does not reproduce the committed verdict files"
    seen = {(r["attempt"], r["id"]) for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}
    ids, want = set(rubric_ids(root)), set()
    for d in attempts(root):
        for rid in ids:
            want.add((f"{d.parent.name}/{d.name}", rid))
    if seen != want:
        miss, extra = sorted(want-seen)[:3], sorted(seen-want)[:3]
        return False, "line set != attempts x rubrics (missing %s, orphaned %s)" % (miss, extra)
    return True, "%d lines == %d attempts x %d rubrics, each reproducing its source entry" % (
        len(seen), len(attempts(root)), len(ids))

if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1]).resolve()
    if "--check" in sys.argv:
        ok, msg = check_binding(root)
        print(("PASS  " if ok else "FAIL  ") + msg); sys.exit(0 if ok else 1)
    print("wrote", emit(root)); print(check_binding(root)[1])
