#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import sys


def _pytest_counts(log: str) -> dict:
    tail = log.strip().splitlines()[-1] if log.strip() else ""
    grab = lambda w: int((re.search(rf"(\d+) {w}", tail) or [0, 0])[1])
    passed, failed, skipped = grab("passed"), grab("failed"), grab("skipped")
    return {"passed": passed, "failed": failed, "skipped": skipped,
            "executed": passed + failed,
            "failed_tests": sorted(set(re.findall(r"^FAILED \S+::(\w+)", log, re.M)))}


def _loss(run: pathlib.Path, graded_step):
    curves = {}
    for lg in sorted([*run.glob("artifacts/full_seed*.log"), *run.glob("artifacts/logs/full_seed*.log")]):
        pts = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r"step:(\d+)/\d+\s+val_loss:([\d.]+)", lg.read_text())}
        if pts:
            curves[lg.stem.replace("full_", "")] = pts
    if not curves:
        return None
    common = sorted(set.intersection(*(set(c) for c in curves.values())))
    at = graded_step if graded_step in common else max(common)
    return {"steps": at,
            "at_graded_step": round(sum(c[at] for c in curves.values()) / len(curves), 6),
            "per_seed": {k: round(v[at], 6) for k, v in curves.items()}}


def _loss_from_numeric(verifier: pathlib.Path):
    p = verifier / "score.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    if "loss_at_graded_step" not in d or "loss_steps" not in d:
        return None
    return {"steps": d["loss_steps"],
            "at_graded_step": d["loss_at_graded_step"],
            "per_seed": {k[len("loss_per_seed_"):]: v
                         for k, v in d.items() if k.startswith("loss_per_seed_")}}


REASON_CODES = (
    ("graded_step=", 0),
    ("verifier_produced_no_score", 1),
    ("telemetry_absent", 10),
    ("telemetry_malformed", 11),
    ("telemetry_empty", 12),
    ("chain_key_absent_cannot_verify", 20),
    ("chain_break_at_step_", 21),
    ("telemetry_not_bound_to_submission_step_", 30),
    ("submission_optimizer_absent", 31),
    ("non_full_mode_telemetry_", 40),
    ("frozen_violation_", 50),
    ("val_selection_at_step_", 51),
    ("reported_seed_", 60),
    ("reported_step_", 61),
    ("reported_loss_diverges_", 62),
    ("telemetry_seed_", 63),
    ("telemetry_step_", 64),
    ("need_at_least_", 70),
    ("no_step_logged_by_every_seed", 71),
    ("optimizer_unreadable_for_copy_check", 80),
    ("no_reference_available_for_copy_check", 81),
    ("verbatim_copy_of_", 82),
    ("fwd_bwd_", 90),
    ("seed_log_collision_", 94),
    ("rubric_veto_", 95),
)
REASON_CODE_MISSING = -1
REASON_CODE_UNRECOGNISED = 99

RUBRIC_GATE_CLEAN = 1
RUBRIC_GATE_VETOED = 0
RUBRIC_GATE_ABSENT = -1
RUBRIC_GATE_INDETERMINATE = -2


def _apply_rubric_veto(run: pathlib.Path, score: float, verd: dict):
    doc = {}
    p = run / "rubric_verdicts.json"
    if p.is_file():
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError:
            return score, None, RUBRIC_GATE_INDETERMINATE
    else:
        return score, None, RUBRIC_GATE_ABSENT
    if doc.get("_indeterminate") or doc.get("overall_pass") is None:
        return score, None, RUBRIC_GATE_INDETERMINATE
    failed = sorted(k for k, x in (verd or {}).items() if not (x or {}).get("pass"))
    if not failed and doc.get("overall_pass") is not False:
        return score, None, RUBRIC_GATE_CLEAN
    if score <= 0.0:
        return 0.0, None, RUBRIC_GATE_VETOED
    return 0.0, "rubric_veto_" + ",".join(failed or ["overall"]), RUBRIC_GATE_VETOED


def _reason_code(reason: str) -> int:
    if not reason:
        return REASON_CODE_MISSING
    for prefix, code in REASON_CODES:
        if reason.startswith(prefix):
            return code
    return REASON_CODE_UNRECOGNISED


def build(run: pathlib.Path, seed_source: pathlib.Path | None = None) -> tuple[dict, dict]:
    v = run / "verifier"
    grade = (v / "grade-stdout.md").read_text() if (v / "grade-stdout.md").is_file() else ""
    rec = {}
    for line in grade.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    score = float(rec.get("score", 0.0))
    reason = rec.get("reason", "")
    step = int(reason.split("=")[1]) if reason.startswith("graded_step=") else None

    pt = _pytest_counts((v / "test-stdout.md").read_text()) if (v / "test-stdout.md").is_file() else {}
    verd = {}
    if (run / "rubric_verdicts.json").is_file():
        try:
            verd = json.loads((run / "rubric_verdicts.json").read_text()).get("verdicts", {})
        except json.JSONDecodeError:
            verd = {}
    rb_pass = sum(1 for x in verd.values() if x.get("pass"))

    pt_frac = (pt.get("passed", 0) / pt["executed"]) if pt.get("executed") else None
    rb_frac = (rb_pass / len(verd)) if verd else None
    composite = score * (pt_frac if pt_frac is not None else 1.0) * (rb_frac if rb_frac is not None else 1.0)

    graded_score = score
    score, veto_reason, rubric_gate = _apply_rubric_veto(run, score, verd)

    loss = _loss(seed_source or run, step) or _loss_from_numeric(v)

    numeric = {"score": score, "graded_score": graded_score, "composite": round(composite, 6),
               "reason_code": _reason_code(veto_reason or reason),
               "rubric_gate": rubric_gate}
    if step is not None:
        numeric["graded_step"] = step
    for k in ("passed", "failed", "skipped", "executed"):
        if k in pt:
            numeric[f"pytests_{k}"] = pt[k]
    if pt_frac is not None:
        numeric["pytests_fraction"] = round(pt_frac, 6)
    if verd:
        numeric.update(rubrics_passed=rb_pass, rubrics_total=len(verd),
                       rubrics_fraction=round(rb_frac, 6))
    if loss:
        numeric["loss_at_graded_step"] = loss["at_graded_step"]
        numeric["loss_steps"] = loss["steps"]
        for s, val in loss["per_seed"].items():
            numeric[f"loss_per_seed_{s}"] = val

    full = dict(rec)
    full.update({
        "score": score,
        "graded_score": graded_score,
        "composite": round(composite, 6),
        "formula": "graded_score * (pytests_passed/executed) * (rubrics_passed/total)",
        "pytests": pt,
        "rubrics": {"passed": rb_pass, "total": len(verd), "gate": rubric_gate,
                    "veto_reason": veto_reason,
                    "failed": [k for k, x in verd.items() if not x.get("pass")]},
        "loss": loss,
        "note": ("graded_score is what grade.py computed from the telemetry. score is that "
                 "value after the rubric veto, which can only lower it to 0.0 and can never "
                 "raise it; the two differ only when a reviewed rubric failed. rubric_gate is "
                 "1 clean, 0 vetoed, -1 no verdict, -2 indeterminate, and any negative value "
                 "is an unreviewed run rather than a clean one. composite is a review aid "
                 "bounded by score. score.json carries only numeric keys because harbor parses "
                 "them all as numbers; this sidecar carries the rest."),
    })
    return numeric, full


FULL_RECORD_MARKER = "--- full record ---"


def write_full_record(verifier: pathlib.Path, full: dict) -> None:
    path = verifier / "grade-stdout.md"
    head = path.read_text().split(FULL_RECORD_MARKER)[0].rstrip() if path.is_file() else ""
    body = json.dumps(full, indent=2, sort_keys=True)
    path.write_text(f"{head}\n\n{FULL_RECORD_MARKER}\n{body}\n" if head
                    else f"{FULL_RECORD_MARKER}\n{body}\n")


def main() -> int:
    run = pathlib.Path(sys.argv[1])
    seed = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    numeric, full = build(run, seed)
    (run / "verifier" / "score.json").write_text(json.dumps(numeric, indent=1, sort_keys=True) + "\n")
    write_full_record(run / "verifier", full)
    print(f"{run.name}: score={numeric['score']} composite={numeric['composite']} "
          f"numeric_keys={len(numeric)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
