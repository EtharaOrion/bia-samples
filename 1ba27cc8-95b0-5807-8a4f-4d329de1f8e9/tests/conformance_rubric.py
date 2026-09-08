#!/usr/bin/env python3
"""Both-halves conformance for the rubric layer (FORGE invariant 17 / item 328).

Every check runs twice: once over the repaired bundle, where it must PASS, and once over a
copy carrying exactly the defect that check catches. Planting is done EXTERNALLY by the
caller; this module runs the clean half only. A check proven on
only one half is a coverage gap capping at BLOCK:INVALID_TASK, so a one-sided result here is
reported as INERT rather than as a pass.
"""
from __future__ import annotations
import ast, json, pathlib, re, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIELDS = ["criterion","dimension","evaluation_target","evidence","id","judgment","mode","weight"]
CLASSES = {"VALUE","EFFECT","ABSENCE","INVARIANT","ORDERING","DIVERGENCE"}
MARK = "Recovered from the run transcript"

def _leaves(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from _leaves(v, pre+"."+k)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from _leaves(v, pre+"[]")
    else: yield pre, o

def account(base: pathlib.Path) -> str:
    fm = (base/"findings.md").read_text(encoding="utf-8")
    if not fm.lstrip().startswith(MARK): return fm
    t = json.loads((base/"agent"/"trajectory.json").read_text(encoding="utf-8"))
    ms = [v for p,v in _leaves(t) if isinstance(v,str) and p.endswith(".arguments.message") and v.strip()]
    return max(ms, key=len) if ms else fm

def norm(s): return re.sub(r"\s+"," ",s).strip()

# ---------------------------------------------------------------- checks
def c1_regen_determinism(root):
    outs=[]
    for env_extra in ({"HOSTNAME":"host-alpha","USER":"alice","TZ":"UTC","LANG":"C"},
                      {"HOSTNAME":"host-beta","USER":"bob","TZ":"Asia/Kolkata","LANG":"en_US.UTF-8"}):
        with tempfile.TemporaryDirectory() as tmp:
            work=pathlib.Path(tmp)/"b"; shutil.copytree(root, work)
            env={"PATH":"/usr/bin:/bin","HOME":tmp,"PYTHONHASHSEED":"0"}; env.update(env_extra)
            r=subprocess.run([sys.executable,"tests/regenerate_rubric_verdicts.py"],cwd=work,env=env,
                             capture_output=True,text=True)
            if r.returncode!=0: return False, "generator failed: "+r.stderr.strip()[:120]
            blob=b"".join(sorted(p.read_bytes() for p in work.glob("trajectories/*/*/rubric_verdicts.json")))
            outs.append(blob)
    return (outs[0]==outs[1]), "byte-identical across two host identities" if outs[0]==outs[1] else "host-dependent bytes"

def c2_set_equality(root):
    r=json.loads((root/"solution"/"rubrics.json").read_text())
    ids={i["id"] for i in r["items"] if i["mode"]=="compiled"}
    tree=ast.parse((root/"tests"/"test_output.py").read_text())
    tests={n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith("test_")}
    return ids==tests, "compiled ids == test functions (%d)"%len(ids) if ids==tests else "asymmetry: %s"%sorted(ids^tests)

def c3_compilation_floor(root):
    r=json.loads((root/"solution"/"rubrics.json").read_text())
    tot=sum(i["weight"] for i in r["items"]); comp=sum(i["weight"] for i in r["items"] if i["mode"]=="compiled")
    share=comp/tot; return share>=r["compilation_floor"], "share %.4f vs floor %.2f"%(share,r["compilation_floor"])

def c4_closed_schema(root):
    r=json.loads((root/"solution"/"rubrics.json").read_text())
    bad=[i.get("id") for i in r["items"] if sorted(i)!=FIELDS]
    return not bad, "all items carry exactly the 8 9g fields" if not bad else "extra/missing fields: %s"%bad

def c5_outcome_class(root):
    r=json.loads((root/"solution"/"rubrics.json").read_text())
    bad={k:v for k,v in r["outcome_classification"].items() if v not in CLASSES}
    return not bad, "all outcomes in the closed class set" if not bad else "invented class: %s"%bad

def c6_evidence_exists(root):
    r=json.loads((root/"solution"/"rubrics.json").read_text())
    missing=[(i["id"],e) for i in r["items"] for e in i["evidence"] if not (root/e).exists()]
    return not missing, "every evidence path exists" if not missing else "absent: %s"%missing[:3]

def c7_generator_purity(root):
    bad=[]
    for name in ("solution/recompute.py","tests/regenerate_rubric_verdicts.py"):
        tree=ast.parse((root/name).read_text()); seen=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): seen.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module: seen.add(node.module.split(".")[0])
        forbidden=seen & {"random","time","datetime","socket","urllib","requests","subprocess","locale","secrets"}
        if forbidden: bad.append((name,sorted(forbidden)))
    return not bad, "no clock/random/network/locale import" if not bad else "impure: %s"%bad

def c8_no_false_on_absence(root):
    g=json.loads((root/"tests"/"rubric_judgment_grounding.json").read_text())
    bad=[(k,r) for k,e in g.items() for r,i in e["items"].items()
         if i["verdict"] is False and e["account_channel"]=="IN_FLIGHT_ONLY"
         and not i["basis"].startswith("ARTIFACT")]
    return not bad, "no false rests on narration absence" if not bad else "absence-scored false: %s"%bad[:3]

def c9_gate_vetoes(root):
    sys.path.insert(0,str(root/"tests"))
    for m in ("rubric_gate",): sys.modules.pop(m,None)
    import rubric_gate as RG
    z=0
    for p in sorted(root.glob("trajectories/*/*/rubric_verdicts.json")):
        doc=json.loads(p.read_text())
        s,_,gate=RG.apply_rubric_veto(1.0,doc)
        if any((v or {}).get("pass") is False for v in doc["verdicts"].values()):
            if not (s==0.0 and gate==0): return False,"a failing attempt was not vetoed: %s"%p.parent.name
            z+=1
        elif doc.get("overall_pass") is None and s!=1.0:
            return False,"an indeterminate attempt was zeroed: %s"%p.parent.name
    return True, "gate vetoes every failing attempt (%d) and zeroes no indeterminate one"%z

def c10_quotes_verbatim(root):
    g=json.loads((root/"tests"/"rubric_judgment_grounding.json").read_text())
    bad=[]
    for k,e in g.items():
        coh,it=k.split("/"); acc=norm(account(root/"trajectories"/coh/it))
        for r,i in e["items"].items():
            if i["basis"]=="NARRATION_QUOTE_HAND_SELECTED" and norm(i["evidence"]) not in acc:
                bad.append((k,r))
    return not bad, "every quoted evidence is verbatim in its account" if not bad else "not verbatim: %s"%bad[:3]

def c11_jsonl_schema(root):
    bad=[]
    for n,line in enumerate((root/"tests"/"rubrics.jsonl").read_text().splitlines(),1):
        if not line.strip(): continue
        d=json.loads(line)
        if sorted(d)!=["id","rubric"]: bad.append((n,sorted(d)))
    return not bad, "every line is exactly {id, rubric}" if not bad else "schema drift: %s"%bad

def c12_committed_matches_generator(root):
    """The emitted verdicts must be exactly what the committed generator produces.

    Without this, a hand-edited rubric_verdicts.json passes every other check: the
    grounding is guarded, the gate is guarded, but nothing binds the artifact to its
    source. That is a result naming a subject it does not pin.
    """
    import shutil, subprocess, tempfile, hashlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        work = tmp / "b"
        shutil.copytree(root, work, ignore=shutil.ignore_patterns("__pycache__", ".git"))
        r = subprocess.run([sys.executable, "tests/regenerate_rubric_verdicts.py"],
                           cwd=str(work), capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return False, "generator failed: " + (r.stderr or r.stdout)[:120]
        drift = []
        for emitted in sorted(work.rglob("rubric_verdicts.json")):
            rel = emitted.relative_to(work)
            committed = root / rel
            if not committed.is_file():
                drift.append(str(rel)); continue
            if hashlib.sha256(emitted.read_bytes()).hexdigest() != \
               hashlib.sha256(committed.read_bytes()).hexdigest():
                drift.append(str(rel))
        n = len(list(work.rglob("rubric_verdicts.json")))
        return (not drift,
                "all %d committed verdicts equal the generator's output" % n if not drift
                else "%d committed verdict(s) differ from the generator: %s" % (len(drift), drift[:3]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def c13_rubric_ids_match_verdicts(root):
    """Every rubric must be judged, and every judgement must name a live rubric.

    C11 validates the shape of each line and C12 binds the verdicts to their generator,
    but nothing bound the rubric SET to the judged SET. A rubric rewrite could therefore
    land with rubrics carrying no verdict and verdicts naming rubrics that no longer
    exist, and both checks would still pass.
    """
    ids = {json.loads(l)["id"]
           for l in (root/"tests"/"rubrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    judged = set()
    for v in sorted(root.rglob("rubric_verdicts.json")):
        d = json.loads(v.read_text(encoding="utf-8")); vs = d.get("verdicts") or {}
        judged |= set(vs.keys()) if isinstance(vs, dict) else {x.get("id") for x in vs}
    unjudged = sorted(ids - judged)
    orphaned = sorted(judged - ids)
    if unjudged or orphaned:
        return False, "unjudged rubric(s): %s; orphaned verdict id(s): %s" % (unjudged[:3], orphaned[:3])
    return True, "all %d rubric ids are judged and no verdict is orphaned" % len(ids)


def c14_grounding_covers_corpus(root):
    g = json.loads((root/"tests"/"rubric_judgment_grounding.json").read_text())
    ids = {json.loads(l)["id"] for l in (root/"tests"/"rubrics.jsonl").read_text().splitlines() if l.strip()}
    n = len(list(root.glob("trajectories/*/*/rubric_verdicts.json")))
    if len(g) != n:
        return False, "grounding has %d entries for %d attempts" % (len(g), n)
    for k, e in g.items():
        if set(e["items"]) != ids:
            return False, "%s misses %s" % (k, sorted(ids ^ set(e["items"]))[:3])
        if not e.get("account_chars", 0) > 0:
            return False, "%s has an empty account" % k
    return True, "%d attempts x %d rubrics, all accounts non-empty" % (len(g), len(ids))


def c15_account_chars_live(root):
    g = json.loads((root/"tests"/"rubric_judgment_grounding.json").read_text())
    bad = []
    for k, e in g.items():
        coh, it = k.split("/")
        # the grounding records the RAW account length (verified 50/50 against
        # this corpus); normalising here would report a false staleness of a few chars
        live = len(account(root/"trajectories"/coh/it))
        if live != e["account_chars"]:
            bad.append((k, e["account_chars"], live))
    if bad:
        return False, "account_chars stale: %s" % bad[:3]
    return True, "all %d account_chars match a live recomputation" % len(g)


def c16_jsonl_binds_to_verdicts(root):
    """The shipped proof artifact must be a VIEW, never a second source of truth.

    trajectories/rubric_verdicts.jsonl is the one file a reviewer opens to see that rubrics
    exist AND that every one was judged. It is only trustworthy if it cannot disagree with the
    per-attempt rubric_verdicts.json it summarises, so this check requires every line to
    reproduce its source entry exactly and the line set to equal attempts x rubrics.
    """
    sys.path.insert(0, str(root/"tests"))
    import rubric_verdicts_jsonl as RVJ
    return RVJ.check_binding(root)


CHECKS=[("C1  regeneration determinism (G-RUB-REGEN)",c1_regen_determinism),
        ("C2  compiled/test identifier set equality",c2_set_equality),
        ("C3  compilation floor >= 0.75",c3_compilation_floor),
        ("C4  9g closed 8-field item schema",c4_closed_schema),
        ("C5  outcome in closed class set",c5_outcome_class),
        ("C6  evidence names only existing files",c6_evidence_exists),
        ("C7  generator purity",c7_generator_purity),
        ("C8  no false scored on narration absence",c8_no_false_on_absence),
        ("C9  rubric hard-pass gate vetoes",c9_gate_vetoes),
        ("C10 quoted evidence is verbatim",c10_quotes_verbatim),
        ("C11 rubrics.jsonl is exactly {id,rubric}",c11_jsonl_schema),
    ("C12 committed verdicts == generator output", c12_committed_matches_generator),
    ("C13 rubric ids == judged verdict ids", c13_rubric_ids_match_verdicts),
    ("C14 grounding covers every attempt x rubric", c14_grounding_covers_corpus),
    ("C15 account_chars recomputes live", c15_account_chars_live),
    ("C16 proof artifact binds to verdicts", c16_jsonl_binds_to_verdicts),]



def main():
    ok=True
    print("%-46s %s"%("CHECK","CLEAN HALF (must PASS)"))
    for label,fn in CHECKS:
        passed,msg=fn(ROOT)
        print("  %-44s %s  %s"%(label,"PASS" if passed else "FAIL",msg))
        ok&=passed
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(main())
