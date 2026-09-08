#!/usr/bin/env python3
"""Both-halves conformance for the rubric layer (FORGE invariant 17 / item 328).

Port of bundle 1ba27cc8's tests/conformance_rubric.py, adapted to bundle A's layout.
Every check runs twice in the full protocol: once over the repaired bundle, where it must
PASS, and once over a copy carrying exactly the defect that check catches. Planting is done
EXTERNALLY by the caller; this module runs the clean half only. A check proven on only one
half is a coverage gap capping at BLOCK:INVALID_TASK, so a one-sided result is reported as
INERT rather than as a pass. Where a check's guarded population is EMPTY on this corpus the
result is annotated VACUOUS, because a guard with no instance has not been shown to fire.

DIVERGENCES FROM 1ba27cc8, each forced by bundle A's layout:

  account()  1ba27cc8 reads findings.md and falls back to the trajectory. Bundle A has NO
             findings.md anywhere, so that function raises on every attempt here - which is
             exactly why C10 never ran on this bundle and fabricated quotes shipped
             undetected. Bundle A's account is defined over the trajectory ONLY, via the
             canonical exhaustive extractor in tests/account.py. agent/history.md is the
             CARRIED PRIOR SUMMARY, never the agent's own account, and is classified as
             prior context.

  C5         Bundle A's solution/rubrics.json has no `outcome_classification` key, so
             1ba27cc8's C5 has no target here. Rather than record it NOT_APPLICABLE, it is
             ported against the closed vocabulary bundle A actually declares:
             `evaluation_target_vocabulary`, plus the same closed dimension class set.

  C10        1ba27cc8 tests whether the WHOLE evidence string appears in the account.
             Bundle A's evidence embeds quoted spans inside the judge's own prose, so the
             whole-string form would fail every item for a reason that is not fabrication.
             The faithful port checks each QUOTED FRAGMENT, using the same extractor that
             defines the account.

             Evidence on this bundle frequently argues AGAINST the rubric it is judging, and
             two rubrics here were split out of a larger parent at commit 14df90a. Evidence
             that quotes the PRE-SPLIT parent wording is quoting a rubric, not the record, so
             checking it against the account would falsely accuse. Those spans are excluded
             using the frozen historical set in tests/rubrics_historical_14df90a.jsonl, which
             is byte-identical to `git show 14df90a:<bundle>/tests/rubrics.jsonl`
             (sha256 03564ee1fbf55229f5a30cd4fc0189aad7059444e541a3026e8a6959b8468f4e).
             The file is carried rather than shelled out for so the check stays hermetic:
             a distributed bundle is not always inside a git repository.
"""
from __future__ import annotations
import ast, json, pathlib, re, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import account as AC  # noqa: E402

FIELDS = ["criterion", "dimension", "evaluation_target", "evidence", "id", "judgment", "mode", "weight"]
CLASSES = {"VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"}
NA = "N/A"


def account(base: pathlib.Path) -> str:
    """Bundle A's account. Trajectory-derived; findings.md does not exist here."""
    return AC.account(base)


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _rubrics(root):
    return json.loads((root / "solution" / "rubrics.json").read_text(encoding="utf-8"))


def _grounding(root):
    return json.loads((root / "tests" / "rubric_judgment_grounding.json").read_text(encoding="utf-8"))


def _rubric_text(root):
    return {json.loads(l)["id"]: json.loads(l)["rubric"]
            for l in (root / "tests" / "rubrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def _historical_rubric_text(root):
    """Pre-split parent rubric wording (commit 14df90a), as ONE blob.

    Only used to EXCLUDE spans from the verbatim check; never to satisfy one.
    """
    p = root / "tests" / "rubrics_historical_14df90a.jsonl"
    if not p.is_file():
        return ""
    return " ".join(json.loads(l)["rubric"]
                    for l in p.read_text(encoding="utf-8").splitlines() if l.strip())


# ---------------------------------------------------------------- checks
def c1_regen_determinism(root):
    outs = []
    for env_extra in ({"HOSTNAME": "host-alpha", "USER": "alice", "TZ": "UTC", "LANG": "C"},
                      {"HOSTNAME": "host-beta", "USER": "bob", "TZ": "Asia/Kolkata", "LANG": "en_US.UTF-8"}):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp) / "b"
            shutil.copytree(root, work, ignore=shutil.ignore_patterns("__pycache__", ".git"))
            env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONHASHSEED": "0",
                   "PYTHONDONTWRITEBYTECODE": "1"}
            env.update(env_extra)
            r = subprocess.run([sys.executable, "tests/regenerate_rubric_verdicts.py"], cwd=work, env=env,
                               capture_output=True, text=True)
            if r.returncode != 0:
                return False, "generator failed: " + r.stderr.strip()[:120]
            outs.append(b"".join(sorted(p.read_bytes() for p in work.glob("trajectories/*/*/rubric_verdicts.json"))))
    ok = outs[0] == outs[1]
    return ok, "byte-identical across two host identities" if ok else "host-dependent bytes"


def c2_set_equality(root):
    r = _rubrics(root)
    ids = {i["id"] for i in r["items"] if i["mode"] == "compiled"}
    tree = ast.parse((root / "tests" / "test_output.py").read_text(encoding="utf-8"))
    tests = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    return ids == tests, ("compiled ids == test functions (%d)" % len(ids)) if ids == tests \
        else "asymmetry: %s" % sorted(ids ^ tests)


def c3_compilation_floor(root):
    r = _rubrics(root)
    tot = sum(i["weight"] for i in r["items"])
    comp = sum(i["weight"] for i in r["items"] if i["mode"] == "compiled")
    share = comp / tot
    return share >= r["compilation_floor"], "share %.4f vs floor %.2f" % (share, r["compilation_floor"])


def c4_closed_schema(root):
    r = _rubrics(root)
    bad = [i.get("id") for i in r["items"] if sorted(i) != FIELDS]
    return not bad, "all %d items carry exactly the 8 9g fields" % len(r["items"]) if not bad \
        else "extra/missing fields: %s" % bad


def c5_closed_vocabulary(root):
    """Ported against the closed vocabulary bundle A declares (see module docstring)."""
    r = _rubrics(root)
    vocab = r.get("evaluation_target_vocabulary")
    if vocab is None:
        return NA, "no evaluation_target_vocabulary and no outcome_classification to bind"
    bad_t = {i["id"]: i["evaluation_target"] for i in r["items"] if i["evaluation_target"] not in set(vocab)}
    bad_d = {i["id"]: i["dimension"] for i in r["items"] if i["dimension"] not in CLASSES}
    if bad_t or bad_d:
        return False, "invented target: %s; invented class: %s" % (bad_t, bad_d)
    return True, "all %d items use the declared target vocabulary (%d) and the closed class set" \
        % (len(r["items"]), len(vocab))


def c6_evidence_exists(root):
    r = _rubrics(root)
    missing = [(i["id"], e) for i in r["items"] for e in i["evidence"] if not (root / e).exists()]
    return not missing, "every evidence path exists" if not missing else "absent: %s" % missing[:3]


def c7_generator_purity(root):
    bad = []
    for name in ("solution/recompute.py", "tests/regenerate_rubric_verdicts.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                seen.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                seen.add(node.module.split(".")[0])
        forbidden = seen & {"random", "time", "datetime", "socket", "urllib", "requests",
                            "subprocess", "locale", "secrets"}
        if forbidden:
            bad.append((name, sorted(forbidden)))
    return not bad, "no clock/random/network/locale import" if not bad else "impure: %s" % bad


def c8_no_false_on_absence(root):
    """A false verdict must not rest on the ABSENCE of narration the harness never kept."""
    g = _grounding(root)
    truncated = {k for k, e in g.items() if e["account_channel"] == "IN_FLIGHT_ONLY"}
    bad = [(k, r) for k, e in g.items() for r, i in e["items"].items()
           if i["verdict"] is False and e["account_channel"] == "IN_FLIGHT_ONLY"
           and not i["basis"].startswith("ARTIFACT")]
    if bad:
        return False, "absence-scored false: %s" % bad[:3]
    if not truncated:
        return True, ("VACUOUS: no attempt carries a truncated account channel, so the guard "
                      "has 0 instances here and is not shown to fire (INERT, one-sided)")
    return True, "no false rests on narration absence (%d truncated attempts)" % len(truncated)


def c9_gate_vetoes(root):
    sys.path.insert(0, str(root / "tests"))
    sys.modules.pop("rubric_gate", None)
    import rubric_gate as RG
    z = 0
    for p in sorted(root.glob("trajectories/*/*/rubric_verdicts.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        s, _, gate = RG.apply_rubric_veto(1.0, doc)
        if any((v or {}).get("pass") is False for v in doc["verdicts"].values()):
            if not (s == 0.0 and gate == 0):
                return False, "a failing attempt was not vetoed: %s" % p.parent.name
            z += 1
        elif doc.get("overall_pass") is None and s != 1.0:
            return False, "an indeterminate attempt was zeroed: %s" % p.parent.name
    return True, "gate vetoes every failing attempt (%d) and zeroes no indeterminate one" % z


def c10_quotes_verbatim(root):
    """Every span the evidence puts in quotation marks must exist in the account."""
    g = _grounding(root)
    rt = _rubric_text(root)
    hist = _historical_rubric_text(root)
    bad, checked = [], 0
    for k, e in g.items():
        coh, it = k.split("/")
        acc = AC.ws(account(root / "trajectories" / coh / it))
        for r, i in e["items"].items():
            if i["basis"] != "NARRATION_QUOTE_HAND_SELECTED":
                continue
            for frag in AC.quoted_fragments(i["evidence"], rt.get(r, ""), hist):
                checked += 1
                if frag not in acc:
                    bad.append((k, r, frag[:60]))
    if bad:
        return False, "%d of %d quoted fragments are not verbatim in their account, e.g. %s" \
            % (len(bad), checked, bad[:2])
    return True, "all %d quoted fragments are verbatim in their account" % checked


def c11_jsonl_schema(root):
    bad = []
    for n, line in enumerate((root / "tests" / "rubrics.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if sorted(d) != ["id", "rubric"]:
            bad.append((n, sorted(d)))
    return not bad, "every line is exactly {id, rubric}" if not bad else "schema drift: %s" % bad


def c12_committed_matches_generator(root):
    """The emitted verdicts must be exactly what the committed generator produces.

    Without this, a hand-edited rubric_verdicts.json passes every other check: the
    grounding is guarded, the gate is guarded, but nothing binds the artifact to its
    source. That is a result naming a subject it does not pin.
    """
    import hashlib
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
                drift.append(str(rel))
                continue
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
    """Every rubric must be judged, and every judgement must name a live rubric."""
    ids = {json.loads(l)["id"]
           for l in (root / "tests" / "rubrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    judged = set()
    for v in sorted(root.rglob("rubric_verdicts.json")):
        d = json.loads(v.read_text(encoding="utf-8"))
        vs = d.get("verdicts") or {}
        judged |= set(vs.keys()) if isinstance(vs, dict) else {x.get("id") for x in vs}
    unjudged = sorted(ids - judged)
    orphaned = sorted(judged - ids)
    if unjudged or orphaned:
        return False, "unjudged rubric(s): %s; orphaned verdict id(s): %s" % (unjudged[:3], orphaned[:3])
    return True, "all %d rubric ids are judged and no verdict is orphaned" % len(ids)


def c14_grounding_covers_corpus(root):
    """The grounding must name every attempt and every live rubric.

    C13 binds rubric ids to VERDICT ids, but nothing bound them to the GROUNDING, which is
    the only input the generator reads. A rubric added to rubrics.jsonl with no grounding
    entry would otherwise surface only as a generator KeyError at some later date.
    """
    g = _grounding(root)
    ids = {json.loads(l)["id"]
           for l in (root / "tests" / "rubrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    attempts = {"%s/%s" % (d.parent.name, d.name) for d in root.glob("trajectories/*/*") if d.is_dir()}
    problems = []
    if set(g) != attempts:
        problems.append("attempt asymmetry: %s" % sorted(set(g) ^ attempts))
    for k, e in sorted(g.items()):
        if sorted(e) != ["account_channel", "account_chars", "items", "overall_pass"]:
            problems.append("%s entry keys %s" % (k, sorted(e)))
        if set(e["items"]) != ids:
            problems.append("%s rubric asymmetry %s" % (k, sorted(set(e["items"]) ^ ids)))
        if e["account_chars"] <= 0:
            problems.append("%s empty account - extractor is broken" % k)
    return not problems, "%d attempts x %d rubrics, all accounts non-empty" % (len(g), len(ids)) \
        if not problems else "; ".join(problems[:3])


def c15_account_chars_live(root):
    """account_chars must equal a live recomputation, not a stale literal."""
    g = _grounding(root)
    bad = []
    for k, e in sorted(g.items()):
        coh, it = k.split("/")
        live = len(account(root / "trajectories" / coh / it))
        if live != e["account_chars"]:
            bad.append((k, e["account_chars"], live))
    return not bad, "all %d account_chars match a live recomputation" % len(g) if not bad \
        else "stale account_chars: %s" % bad[:3]


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


CHECKS = [("C1  regeneration determinism (G-RUB-REGEN)", c1_regen_determinism),
          ("C2  compiled/test identifier set equality", c2_set_equality),
          ("C3  compilation floor >= 0.75", c3_compilation_floor),
          ("C4  9g closed 8-field item schema", c4_closed_schema),
          ("C5  target/dimension in closed vocabulary", c5_closed_vocabulary),
          ("C6  evidence names only existing files", c6_evidence_exists),
          ("C7  generator purity", c7_generator_purity),
          ("C8  no false scored on narration absence", c8_no_false_on_absence),
          ("C9  rubric hard-pass gate vetoes", c9_gate_vetoes),
          ("C10 quoted evidence is verbatim", c10_quotes_verbatim),
          ("C11 rubrics.jsonl is exactly {id,rubric}", c11_jsonl_schema),
          ("C12 committed verdicts == generator output", c12_committed_matches_generator),
          ("C13 rubric ids == judged verdict ids", c13_rubric_ids_match_verdicts),
          ("C14 grounding covers every attempt x rubric", c14_grounding_covers_corpus),
          ("C15 account_chars recomputes live", c15_account_chars_live),
          ("C16 proof artifact binds to verdicts", c16_jsonl_binds_to_verdicts)]


def main():
    ok = True
    print("%-46s %s" % ("CHECK", "CLEAN HALF (must PASS)"))
    for label, fn in CHECKS:
        passed, msg = fn(ROOT)
        if passed == NA:
            status = "N/A "
        else:
            status = "PASS" if passed else "FAIL"
            ok &= bool(passed)
        print("  %-44s %s  %s" % (label, status, msg))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
