#!/usr/bin/env python3
"""Both halves liveness proof for the six checkers of the BIA-GSN-1 task.

A required instrument has to demonstrate both halves of its decision under frozen
bytes: it must accept a known good solution and it must reject a known wrong one. A
checker hardcoded to pass and a checker hardcoded to fail are both inert, and a
conformance run that proves only one half cannot tell them apart.

This runner proves both halves for every checker in the bundle. It works on a throwaway
copy of the bundle, so a control that deliberately edits the frozen surface cannot
touch the bundle it is proving. It runs entirely on the smoke profile on CPU, so it
allocates no accelerator memory and can be run while an unrelated campaign owns the
card.

Usage:
    python3 solution/control_runner.py --bundle <bundle root>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

SUBMISSION_CONTROLS = [
    ("clean_reference", "fast_impl.py", None),
    ("c_perturbed", "controls/c_perturbed.py", "divergence_bitwise"),
    ("c_float64_roundtrip", "controls/c_float64_roundtrip.py", "divergence_bitwise"),
    ("c_truncated_output", "controls/c_truncated_output.py", "effect_outputs_materialized"),
    ("c_tf32_override", "controls/c_tf32_override.py", "absence_relaxed_precision"),
    ("c_frozen_edit", "controls/c_frozen_edit.py", "invariant_frozen_surface"),
]


def stage(bundle):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="bia-control-"))
    root = tmp / "bundle"
    shutil.copytree(str(bundle), str(root))
    work = tmp / "work"
    (work / "submission").mkdir(parents=True)
    (work / "artifacts").mkdir(parents=True)
    (tmp / "logs").mkdir(parents=True)
    return tmp, root, work


def env_for(root, work, tmp):
    env = dict(os.environ)
    env.update(
        {
            "BIA_SMOKE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "BIA_BUNDLE": str(root),
            "BIA_TESTS": str(root / "tests"),
            "BIA_ENV_DIR": str(root / "environment"),
            "BIA_SUBMISSION": str(work / "submission" / "impl.py"),
            "BIA_ARTIFACTS": str(work / "artifacts"),
            "BIA_LOGS": str(tmp / "logs"),
            "SCORE_PATH": str(tmp / "logs" / "score.json"),
            "BIA_REPORT": str(tmp / "logs" / "report.json"),
            "BIA_OUTCOMES": str(tmp / "logs" / "outcomes.json"),
            "BIA_EVENTS": str(tmp / "logs" / "harness_events.jsonl"),
            "BIA_TIMING_SUMMARY": str(tmp / "logs" / "timing_summary.json"),
        }
    )
    env.pop("NVIDIA_TF32_OVERRIDE", None)
    return env


def run_grader(bundle, control_rel):
    tmp, root, work = stage(bundle)
    try:
        shutil.copy(str(root / "solution" / "fast_impl.py"), str(work / "submission" / "fast_impl.py"))
        shutil.copy(str(root / "solution" / control_rel), str(work / "submission" / "impl.py"))
        env = env_for(root, work, tmp)
        proc = subprocess.run(
            [sys.executable, str(root / "tests" / "grade.py")],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        report_path = tmp / "logs" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        return report, proc
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def failed_checkers(report):
    return [c["checker"] for c in report.get("checkers", []) if not c.get("passed")]


def run_log_controls(bundle):
    """Both halves for the ordering and value checkers.

    These two read the harness event log rather than a submission artifact, so their
    rejecting half is proven by mutating a recorded log and re-running the checkers
    over it, which is the only way a log reading checker can be shown to fire.
    """
    tmp, root, work = stage(bundle)
    results = []
    try:
        shutil.copy(str(root / "solution" / "fast_impl.py"), str(work / "submission" / "impl.py"))
        env = env_for(root, work, tmp)
        proc = subprocess.run(
            [sys.executable, str(root / "tests" / "grade.py")],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            return [("log_control_setup", False, "clean grade run failed: %s" % proc.stdout[-800:])]
        sys.path.insert(0, str(root / "tests"))
        from checkers import kinds

        logs = tmp / "logs"
        spec = json.loads((root / "tests" / "checkers" / "spec.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (root / "tests" / "checkers" / "frozen_manifest.json").read_text(encoding="utf-8")
        )["files"]
        profile = spec["profiles"]["smoke"]
        marker = work / "artifacts" / ".invocation_start"
        ctx = {
            "bundle_root": str(root),
            "submission_out": str(work / "artifacts" / "submission_out"),
            "reference_out": str(work / "artifacts" / "reference_out"),
            "fixtures": profile["fixtures"],
            "invocation_start_ns": marker.stat().st_mtime_ns,
            "frozen_manifest": manifest,
            "frozen_start": {rel: kinds._sha256_file(root / rel) for rel in manifest},
            "events_path": str(logs / "harness_events.jsonl"),
            "timing_summary_path": str(logs / "timing_summary.json"),
            "declared_order": list(spec["timing_protocol"]["order_within_round"]),
            "expected_timed_trials": int(profile["timed_trials"]),
            "baseline_metric": float(spec["reward"]["baseline_metric"]),
            "target_metric": float(spec["reward"]["target_metric"]),
        }

        clean = {r["checker"]: r["passed"] for r in kinds.run_all(ctx)}
        results.append(
            (
                "clean_log_accepted",
                bool(clean.get("ordering_measurement_protocol") and clean.get("value_reward_recomputed")),
                "ordering and value checkers accept the recorded log",
            )
        )

        events_path = logs / "harness_events.jsonl"
        original = events_path.read_text(encoding="utf-8")
        kept = []
        dropped = 0
        for line in original.splitlines():
            rec = json.loads(line)
            if rec.get("event") == "trial" and rec.get("impl") == "submission" and int(rec.get("index", 0)) == 0:
                dropped += 1
                continue
            kept.append(line)
        events_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        mutated = {r["checker"]: r["passed"] for r in kinds.run_all(ctx)}
        results.append(
            (
                "c_ordering_truncated_log",
                (dropped > 0) and (mutated.get("ordering_measurement_protocol") is False),
                "dropping %d trial records makes the ordering checker fire" % dropped,
            )
        )
        events_path.write_text(original, encoding="utf-8")

        summary_path = logs / "timing_summary.json"
        pristine = summary_path.read_text(encoding="utf-8")
        summary = json.loads(pristine)
        summary["speedup"] = float(summary.get("speedup", 1.0)) + 10.0
        summary["reward"] = 1.0
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        tampered = {r["checker"]: r["passed"] for r in kinds.run_all(ctx)}
        results.append(
            (
                "c_value_tampered_summary",
                tampered.get("value_reward_recomputed") is False,
                "an inflated summary the durations do not reproduce makes the value checker fire",
            )
        )
        summary_path.write_text(pristine, encoding="utf-8")
        return results
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(pathlib.Path(__file__).resolve().parent.parent))
    args = parser.parse_args()
    bundle = pathlib.Path(args.bundle).resolve()

    rows = []
    for name, rel, target in SUBMISSION_CONTROLS:
        report, proc = run_grader(bundle, rel)
        failures = failed_checkers(report)
        score = float(report.get("score", -1.0))
        if target is None:
            ok = (not failures) and score > 0.0
            note = "accepted, score %.6f, no checker failed" % score
            if not ok:
                note = "expected acceptance, got score %.6f failures %s rc=%s %s" % (
                    score,
                    failures,
                    proc.returncode,
                    proc.stdout[-600:],
                )
        else:
            ok = (score == 0.0) and (target in failures)
            note = "rejected by %s, score %.1f, failures %s" % (target, score, failures)
            if not ok:
                note = "expected %s to fire, got score %s failures %s %s" % (
                    target,
                    score,
                    failures,
                    proc.stdout[-600:],
                )
        rows.append((name, ok, note))

    rows.extend(run_log_controls(bundle))

    print("")
    print("both halves liveness proof, smoke profile, cpu only")
    print("-" * 78)
    all_ok = True
    for name, ok, note in rows:
        all_ok = all_ok and ok
        print("%-28s %-5s %s" % (name, "PASS" if ok else "FAIL", note))
    print("-" * 78)
    print("controls: %d, all as expected: %s" % (len(rows), all_ok))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
