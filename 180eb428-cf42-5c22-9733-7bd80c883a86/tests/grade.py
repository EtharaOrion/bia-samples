"""Verifier for bia slot S07. Measures the submission itself, then runs the checkers.

WHERE THE GRADED NUMBER COMES FROM. It does not come from anything the agent wrote. This
verifier starts its own graded attempt, in its own process tree, in a directory whose name it
draws from system entropy at grade time, with the anchor cache disabled, and stamps that run
with a nonce nothing outside this process has ever seen. The reward is computed from THAT
record. The frozen driver in turn hosts the submitted policy in a separate interpreter, so the
submission is two process boundaries away from every number that reaches the score.

The record the agent produced is still graded, because a truthful attempt report is part of
the deliverable and every artifact-tampering control binds to it. Each checker therefore
decides twice, once over the verifier-authored record and once over the agent-reported record,
and a checker passes only when both halves hold. The reward, though, is only ever read off the
verifier-authored record.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import secrets
import subprocess
import sys
from typing import Any, Dict, List, Optional

TESTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR / "checkers"))

import reference_reducer as RR  # noqa: E402

BUNDLE = pathlib.Path(os.environ.get("BIA_BUNDLE", TESTS_DIR.parent))
ARTIFACTS = pathlib.Path(os.environ.get("BIA_ARTIFACTS", "/workspace/artifacts"))
SCORE_PATH = pathlib.Path(os.environ.get("SCORE_PATH", "/logs/verifier/score.json"))
OUTCOMES_PATH = pathlib.Path(os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json"))
REWARD_PATH = pathlib.Path(os.environ.get("BIA_REWARD_PATH", "/logs/verifier/reward.txt"))
ENVDIR = pathlib.Path(os.environ.get("BIA_ENVDIR", BUNDLE / "environment"))
DIGEST_PIN = TESTS_DIR / "checkers" / "format_digest.json"
FROZEN_PIN = TESTS_DIR / "checkers" / "frozen_bytes.json"

# The one deliverable path this verifier grades. instruction.md names it and task.toml carries
# it in the artifact list. Resolving it here rather than reading it out of a record the agent
# wrote is what stops a submission from nominating some other file for measurement.
SUBMISSION = pathlib.Path(
    os.environ.get("BIA_SUBMISSION", str(ENVDIR / "submission" / "precision_policy.py"))
)

# CONTROL-ONLY INPUT. When set, the named file is used as the authoritative record instead of
# running the attempt, and the authoritative source is recorded as `external`. The origin
# checker requires the source to be `verifier_execution`, so this switch can only ever produce
# a zero. It exists so the rejecting half of that checker is executable under frozen bytes.
EXTERNAL_AUTHORITATIVE = os.environ.get("BIA_AUTHORITATIVE_TELEMETRY", "")

VERIFIER_RUN_TIMEOUT = float(os.environ.get("BIA_VERIFIER_RUN_TIMEOUT", "1500"))

FORMAT_ID = "BIA-E4M3N-v1"
NARROW_PHASES = ("standard_control", "agent_run")
ALL_PHASES = ("fp32_control", "standard_control", "agent_run")


# The kebab reason code each checker emits alongside the zero it explains. tests/checkers.yaml
# declares these as zero_reason and solution/recompute.py refuses to emit that manifest unless
# every code below appears in the reachable bytes, so the manifest cannot claim an attribution
# this file never prints.
ZERO_REASON = {
    "frozen_format_digest_matches": "frozen-format-digest-mismatched",
    "no_overflow_flag_raised_in_agent_phase": "overflow-flag-raised-in-agent-phase",
    "format_in_force_at_every_reduction": "reduction-escaped-the-frozen-format",
    "agent_improves_on_standard_control": "agent-did-not-improve-on-standard-control",
    "phases_ran_in_declared_order": "phases-out-of-declared-order",
    "reduction_agrees_with_independent_replay": "replay-diverged-from-recorded-reduction",
    "master_weights_updated_in_declared_dtype": "master-weights-not-updated-in-declared-dtype",
    "frozen_harness_bytes_unmodified": "frozen-harness-bytes-modified",
    "overflow_ledger_accounting_is_consistent": "overflow-ledger-accounting-inconsistent",
    "agent_phase_covered_every_declared_step": "declared-step-coverage-incomplete",
    "fp32_control_ran_outside_the_frozen_format": "fp32-control-ran-inside-the-frozen-format",
    "baseline_anchor_ran_the_shipped_narrow_recipe": "baseline-anchor-is-not-the-shipped-narrow-recipe",
    "anchor_span_is_positive_and_finite": "anchor-span-degenerate",
    "all_phases_started_from_the_same_initialization": "phases-started-from-different-initializations",
    "control_cache_agrees_with_graded_telemetry": "control-cache-diverged-from-graded-telemetry",
    "reported_outcome_agrees_with_telemetry": "reported-outcome-diverged-from-telemetry",
    "attempt_wallclock_within_declared_budget": "attempt-wallclock-outside-declared-budget",
    "graded_record_authored_by_the_verifier": "graded-record-not-authored-by-the-verifier",
    "policy_ran_outside_the_measuring_process": "policy-ran-inside-the-measuring-process",
    "agent_report_reproduces_under_verifier_execution": "agent-report-did-not-reproduce-under-verifier-execution",
}


class Ctx:
    """One record under test, together with the directory its side artifacts live in."""

    def __init__(self, label: str, tel: Optional[List[Dict[str, Any]]], artifacts: pathlib.Path) -> None:
        self.label = label
        self.tel = tel or []
        self.artifacts = artifacts


def write_reward(score: float) -> None:
    """Write the one float the runtime reads from /logs/verifier/reward.txt.

    Never raises: a verifier that dies while reporting its own score turns a graded zero into an
    unscored run, which is the failure this write exists to prevent.
    """
    try:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        REWARD_PATH.write_text("%.12f\n" % min(max(float(score), 0.0), 1.0))
    except (OSError, TypeError, ValueError) as exc:
        print("reward path %s unwritable: %s" % (REWARD_PATH, exc), file=sys.stderr)


def emit(score: float, reason: str, detail: Dict[str, Any], outcomes: Dict[str, bool]) -> int:
    SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    zero_reasons = sorted(ZERO_REASON[n] for n, v in outcomes.items() if not v and n in ZERO_REASON)
    payload = {
        "score": float(score),
        "reason": reason,
        "zero_reasons": zero_reasons,
        "detail": detail,
    }
    SCORE_PATH.write_text(json.dumps(payload, indent=1, sort_keys=True))
    OUTCOMES_PATH.write_text(json.dumps(outcomes, indent=1, sort_keys=True))
    write_reward(score)
    for code in zero_reasons:
        print("zero_reason %s" % code)
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def phase_end(tel: List[Dict[str, Any]], phase: str) -> Optional[Dict[str, Any]]:
    for r in tel:
        if r.get("record") == "phase_end" and r.get("phase") == phase:
            return r
    return None


def reductions(tel: List[Dict[str, Any]], phase: str) -> List[Dict[str, Any]]:
    return [r for r in tel if r.get("record") == "reduction" and r.get("phase") == phase]


def attempt_begin(tel: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in tel:
        if r.get("record") == "attempt_begin":
            return r
    return None


def attempt_end(tel: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in tel:
        if r.get("record") == "attempt_end":
            return r
    return None


def operating_point() -> Dict[str, Any]:
    return json.loads((ENVDIR / "operating_point.json").read_text())


def declared_steps(tel: List[Dict[str, Any]]) -> Optional[int]:
    ab = attempt_begin(tel)
    if ab is None:
        return None
    op = operating_point()
    return int(op["smoke"]["steps"] if ab.get("mode") == "smoke" else op["schedule"]["steps"])


def budget_seconds(tel: List[Dict[str, Any]]) -> Optional[float]:
    ab = attempt_begin(tel)
    if ab is None:
        return None
    op = operating_point()
    if ab.get("mode") == "smoke":
        return float(op["smoke"]["wallclock_target_seconds"])
    return float(op["derivation"]["per_attempt_budget_hours"]) * 3600.0


def reproduction_tolerance() -> float:
    return float(operating_point()["reproduction"]["final_val_loss_absolute_tolerance"])


def control_cache_path(ctx: Ctx) -> Optional[pathlib.Path]:
    """Locate the cache by the digest the attempt recorded, never by a glob.

    A glob would let a stale or foreign cache file satisfy the comparison; keying on the
    config_digest of this attempt binds the cache to the configuration that produced it.
    """
    ab = attempt_begin(ctx.tel)
    if ab is None or not ab.get("config_digest"):
        return None
    return ctx.artifacts / ("controls_%s.json" % str(ab["config_digest"])[:16])


def finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def harness_reward(baseline: float, target: float, agent: float, overflow_events: int):
    """Re-derivation of environment/run_attempt.reward_from, kept in the same order.

    Duplicated here on purpose: reported_outcome_agrees_with_telemetry must decide what the
    harness rule PRODUCES from the recorded numbers without importing agent-writable code.
    """
    if overflow_events > 0:
        return 0.0, "overflow_in_agent_phase"
    span = baseline - target
    if not (span > 0.0):
        return 0.0, "degenerate_anchor_span"
    if agent != agent:
        return 0.0, "agent_loss_not_finite"
    return min(max((baseline - agent) / span, 0.0), 1.0), "scored"


def _put(detail: Dict[str, Any], name: str, ctx: Ctx, value: Any) -> None:
    detail.setdefault(name, {})[ctx.label] = value


# --- the checkers, one reduction kind each ---------------------------------


def chk_frozen_format_digest_matches(ctx, detail) -> bool:
    """VALUE. sha256 of the two frozen numeric files on disk equals the pinned pair."""
    if not DIGEST_PIN.is_file():
        _put(detail, "frozen_format_digest_matches", ctx, "digest pin absent")
        return False
    pinned = json.loads(DIGEST_PIN.read_text())
    live = {}
    for name in ("format.json", "bia_numerics.py"):
        p = ENVDIR / name
        if not p.is_file():
            _put(detail, "frozen_format_digest_matches", ctx, "missing %s" % name)
            return False
        live[name] = sha256_file(p)
    _put(detail, "frozen_format_digest_matches", ctx, {"pinned": pinned["files"], "live": live})
    return live == pinned["files"]


def chk_no_overflow_flag_raised_in_agent_phase(ctx, detail) -> bool:
    """ABSENCE. The overflow event list on the agent phase is empty."""
    pe = phase_end(ctx.tel, "agent_run")
    if pe is None:
        _put(detail, "no_overflow_flag_raised_in_agent_phase", ctx, "agent phase_end absent")
        return False
    n = int(pe.get("overflow_event_count", -1))
    evs = pe.get("overflow_events", None)
    _put(detail, "no_overflow_flag_raised_in_agent_phase", ctx, {
        "overflow_event_count": n,
        "overflow_elements": pe.get("overflow_elements"),
        "first_events": (evs or [])[:3],
    })
    return n == 0 and isinstance(evs, list) and len(evs) == 0


def chk_format_in_force_at_every_reduction(ctx, detail) -> bool:
    """INVARIANT. Every reduction of the agent phase ran under the frozen format."""
    recs = reductions(ctx.tel, "agent_run")
    bad = [
        r.get("step")
        for r in recs
        if r.get("format_id") != FORMAT_ID or int(r.get("quantize_calls", 0)) <= 0
    ]
    _put(detail, "format_in_force_at_every_reduction", ctx, {
        "reductions_observed": len(recs),
        "violating_steps": bad[:5],
    })
    return len(recs) > 0 and not bad


def chk_agent_improves_on_standard_control(ctx, detail) -> bool:
    """VALUE. Agent final validation loss bounds the standard control's from below."""
    a, s = phase_end(ctx.tel, "agent_run"), phase_end(ctx.tel, "standard_control")
    if a is None or s is None:
        _put(detail, "agent_improves_on_standard_control", ctx, "phase record absent")
        return False
    av, sv = float(a["final_val_loss"]), float(s["final_val_loss"])
    _put(detail, "agent_improves_on_standard_control", ctx, {"agent": av, "standard_control": sv})
    return av == av and av < sv


def chk_phases_ran_in_declared_order(ctx, detail) -> bool:
    """ORDERING. Both anchors were established before the graded phase began."""
    begins = {r["phase"]: r["seq"] for r in ctx.tel if r.get("record") == "phase_begin"}
    ends = {r["phase"]: r["seq"] for r in ctx.tel if r.get("record") == "phase_end"}
    need = ALL_PHASES
    if any(p not in ends for p in need) or "agent_run" not in begins:
        _put(detail, "phases_ran_in_declared_order", ctx, {"begins": begins, "ends": ends})
        return False
    ok = ends["fp32_control"] < ends["standard_control"] < begins["agent_run"] < ends["agent_run"]
    _put(detail, "phases_ran_in_declared_order", ctx, {"begins": begins, "ends": ends, "ordered": ok})
    return ok


def chk_reduction_agrees_with_independent_replay(ctx, detail) -> bool:
    """DIVERGENCE. Every replayed reduction and an independent scalar reducer agree.

    The replayed step set is drawn by the measuring process from system entropy after the
    policy worker has already started, and this checker validates EVERY sample in it and
    requires the recorded set to equal the requested set. A single fixed position used to be
    replayed, which left a step-selective policy unconstrained at every other index.
    """
    samples = [r for r in ctx.tel if r.get("record") == "reduction_replay" and r.get("phase") == "agent_run"]
    pe = phase_end(ctx.tel, "agent_run")
    if not samples or pe is None:
        _put(detail, "reduction_agrees_with_independent_replay", ctx, "no replay sample recorded")
        return False
    requested = sorted(int(s) for s in (pe.get("replay_steps_requested") or []))
    recorded = sorted(int(s.get("step", -1)) for s in samples)
    mismatched_steps, checked = [], 0
    for s in samples:
        got = RR.replay(ENVDIR / "format.json", s)
        exp = list(s["expected"])
        checked += len(exp)
        if not exp or any(not ((g != g and e != e) or g == e) for g, e in zip(got, exp)):
            mismatched_steps.append(s.get("step"))
    _put(detail, "reduction_agrees_with_independent_replay", ctx, {
        "samples": len(samples),
        "requested_steps": requested,
        "recorded_steps": recorded,
        "elements_checked": checked,
        "diverging_steps": mismatched_steps[:5],
    })
    return bool(requested) and recorded == requested and checked > 0 and not mismatched_steps


def chk_master_weights_updated_in_declared_dtype(ctx, detail) -> bool:
    """EFFECT. The declared master dtype is the dtype the live tensors carried, and they moved."""
    pe = phase_end(ctx.tel, "agent_run")
    if pe is None:
        _put(detail, "master_weights_updated_in_declared_dtype", ctx, "agent phase_end absent")
        return False
    declared = str(pe.get("master_dtype", ""))
    observed = str(pe.get("master_param_dtype_observed", ""))
    first, final = pe.get("master_checksum_first"), pe.get("master_checksum_final")
    _put(detail, "master_weights_updated_in_declared_dtype", ctx, {
        "declared": declared,
        "observed": observed,
        "checksum_changed": first != final,
    })
    return observed == "torch." + declared and bool(first) and bool(final) and first != final


def chk_frozen_harness_bytes_unmodified(ctx, detail) -> bool:
    """VALUE. sha256 of the frozen files the format pin does not cover equals the pinned map."""
    if not FROZEN_PIN.is_file():
        _put(detail, "frozen_harness_bytes_unmodified", ctx, "frozen byte pin absent")
        return False
    pinned = json.loads(FROZEN_PIN.read_text())["files"]
    live, missing = {}, []
    for name in sorted(pinned):
        p = ENVDIR / name
        if not p.is_file():
            missing.append(name)
            continue
        live[name] = sha256_file(p)
    _put(detail, "frozen_harness_bytes_unmodified", ctx, {
        "missing": missing,
        "mismatched": sorted(n for n, d in live.items() if pinned[n] != d),
    })
    return not missing and live == pinned


def chk_overflow_ledger_accounting_is_consistent(ctx, detail) -> bool:
    """INVARIANT. Per-reduction ledger deltas reconcile with the end-of-phase snapshot."""
    report, ok = {}, True
    for phase in NARROW_PHASES:
        pe = phase_end(ctx.tel, phase)
        if pe is None:
            report[phase] = "phase_end absent"
            ok = False
            continue
        recs = reductions(ctx.tel, phase)
        if not recs:
            # A cached anchor ships its snapshot without per-reduction records; it was
            # reconciled on the attempt that computed it. The graded phase is never cached,
            # so an absent record set there is a missing universe and fails.
            report[phase] = "no reduction records"
            if phase == "agent_run" or not pe.get("from_cache"):
                ok = False
            continue
        sum_calls = sum(int(r.get("quantize_calls", 0)) for r in recs)
        sum_over = sum(int(r.get("overflow_elements", 0)) for r in recs)
        snap_calls = int(pe.get("quantize_calls", -1))
        snap_over = int(pe.get("overflow_elements", -1))
        n_events = int(pe.get("overflow_event_count", -1))
        coherent = (n_events == 0) == (snap_over == 0)
        agree = sum_calls == snap_calls and sum_over == snap_over and snap_calls > 0
        report[phase] = {
            "summed_quantize_calls": sum_calls,
            "snapshot_quantize_calls": snap_calls,
            "summed_overflow_elements": sum_over,
            "snapshot_overflow_elements": snap_over,
            "overflow_event_count": n_events,
            "event_count_coherent_with_elements": coherent,
        }
        ok = ok and agree and coherent
    _put(detail, "overflow_ledger_accounting_is_consistent", ctx, report)
    return ok


def chk_agent_phase_covered_every_declared_step(ctx, detail) -> bool:
    """INVARIANT. The graded phase ran the whole declared schedule and skipped none of it."""
    pe = phase_end(ctx.tel, "agent_run")
    want = declared_steps(ctx.tel)
    if pe is None or want is None:
        _put(detail, "agent_phase_covered_every_declared_step", ctx, "agent phase_end or attempt_begin absent")
        return False
    steps = sorted(int(r.get("step", -1)) for r in reductions(ctx.tel, "agent_run"))
    _put(detail, "agent_phase_covered_every_declared_step", ctx, {
        "declared_steps": want,
        "steps_run": pe.get("steps_run"),
        "steps_skipped": pe.get("steps_skipped"),
        "reduction_steps_recorded": len(steps),
    })
    return (
        int(pe.get("steps_run", -1)) == want
        and int(pe.get("steps_skipped", -1)) == 0
        and steps == list(range(want))
    )


def chk_fp32_control_ran_outside_the_frozen_format(ctx, detail) -> bool:
    """ABSENCE. No narrow-format arithmetic exists anywhere inside the target anchor."""
    pe = phase_end(ctx.tel, "fp32_control")
    if pe is None:
        _put(detail, "fp32_control_ran_outside_the_frozen_format", ctx, "fp32_control phase_end absent")
        return False
    tainted = [
        r.get("step")
        for r in reductions(ctx.tel, "fp32_control")
        if r.get("format_id") != "float32" or int(r.get("quantize_calls", 0)) != 0
    ]
    _put(detail, "fp32_control_ran_outside_the_frozen_format", ctx, {
        "format_id": pe.get("format_id"),
        "quantize_calls": pe.get("quantize_calls"),
        "overflow_event_count": pe.get("overflow_event_count"),
        "tainted_reduction_steps": tainted[:5],
    })
    return (
        pe.get("format_id") == "float32"
        and int(pe.get("quantize_calls", -1)) == 0
        and int(pe.get("overflow_event_count", -1)) == 0
        and not tainted
    )


def chk_baseline_anchor_ran_the_shipped_narrow_recipe(ctx, detail) -> bool:
    """VALUE. The baseline anchor really is the frozen narrow recipe, and it really overflowed.

    The recipe is pinned by digest as well as by behaviour: environment/baseline_recipe.py is a
    frozen red-lined file, so the denominator of the score cannot be moved by editing the file
    the agent is told to edit.
    """
    pe = phase_end(ctx.tel, "standard_control")
    ab = attempt_begin(ctx.tel)
    if pe is None or ab is None:
        _put(detail, "baseline_anchor_ran_the_shipped_narrow_recipe", ctx, "standard_control phase_end absent")
        return False
    recipe = ENVDIR / "baseline_recipe.py"
    live = sha256_file(recipe) if recipe.is_file() else ""
    recorded = str(ab.get("baseline_recipe_sha256", ""))
    _put(detail, "baseline_anchor_ran_the_shipped_narrow_recipe", ctx, {
        "format_id": pe.get("format_id"),
        "quantize_calls": pe.get("quantize_calls"),
        "overflow_event_count": pe.get("overflow_event_count"),
        "baseline_recipe_sha256_live": live,
        "baseline_recipe_sha256_recorded": recorded,
    })
    return (
        pe.get("format_id") == FORMAT_ID
        and int(pe.get("quantize_calls", 0)) > 0
        and int(pe.get("overflow_event_count", 0)) > 0
        and bool(live)
        and live == recorded
    )


def chk_anchor_span_is_positive_and_finite(ctx, detail) -> bool:
    """VALUE. The reward ratio has a strictly positive, finite denominator."""
    ends = {p: phase_end(ctx.tel, p) for p in ALL_PHASES}
    if any(e is None for e in ends.values()):
        _put(detail, "anchor_span_is_positive_and_finite", ctx, "phase record absent")
        return False
    losses = {p: ends[p].get("final_val_loss") for p in ALL_PHASES}
    if not all(finite(v) for v in losses.values()):
        _put(detail, "anchor_span_is_positive_and_finite", ctx, {"losses": losses, "all_finite": False})
        return False
    target, baseline = float(losses["fp32_control"]), float(losses["standard_control"])
    _put(detail, "anchor_span_is_positive_and_finite", ctx, {
        "target": target,
        "baseline": baseline,
        "span": baseline - target,
    })
    return baseline - target > 0.0


def chk_all_phases_started_from_the_same_initialization(ctx, detail) -> bool:
    """INVARIANT. All three phases are one experiment: same init, and each of them moved."""
    ends = {p: phase_end(ctx.tel, p) for p in ALL_PHASES}
    if any(e is None for e in ends.values()):
        _put(detail, "all_phases_started_from_the_same_initialization", ctx, "phase record absent")
        return False
    firsts = {p: ends[p].get("master_checksum_first") for p in ALL_PHASES}
    moved = {
        p: bool(ends[p].get("master_checksum_final")) and ends[p].get("master_checksum_final") != firsts[p]
        for p in ALL_PHASES
    }
    _put(detail, "all_phases_started_from_the_same_initialization", ctx, {
        "initialization_checksums": firsts,
        "each_phase_moved": moved,
    })
    return len({v for v in firsts.values()}) == 1 and all(firsts.values()) and all(moved.values())


def chk_control_cache_agrees_with_graded_telemetry(ctx, detail) -> bool:
    """DIVERGENCE. The anchor cache and the graded telemetry state the same two runs."""
    cache_path = control_cache_path(ctx)
    if cache_path is None or not cache_path.is_file():
        _put(detail, "control_cache_agrees_with_graded_telemetry", ctx, {
            "looked_at": str(cache_path),
            "present": False,
        })
        return False
    cached = json.loads(cache_path.read_text())
    fields = (
        "final_val_loss",
        "master_checksum_first",
        "master_checksum_final",
        "overflow_event_count",
        "quantize_calls",
    )
    diverged = []
    for phase in ("fp32_control", "standard_control"):
        pe, cr = phase_end(ctx.tel, phase), cached.get(phase)
        if pe is None or not isinstance(cr, dict):
            diverged.append("%s:absent" % phase)
            continue
        diverged += ["%s.%s" % (phase, f) for f in fields if pe.get(f) != cr.get(f)]
    _put(detail, "control_cache_agrees_with_graded_telemetry", ctx, {
        "cache": cache_path.name,
        "diverged_fields": diverged,
    })
    return not diverged


def chk_reported_outcome_agrees_with_telemetry(ctx, detail) -> bool:
    """DIVERGENCE. The reported outcome is the recorded outcome, re-derived from telemetry."""
    sp = ctx.artifacts / "score.json"
    ae = attempt_end(ctx.tel)
    if not sp.is_file() or ae is None:
        _put(detail, "reported_outcome_agrees_with_telemetry", ctx, {
            "score_json_present": sp.is_file(),
            "attempt_end_present": ae is not None,
        })
        return False
    rep = json.loads(sp.read_text())
    ends = {p: phase_end(ctx.tel, p) for p in ALL_PHASES}
    if any(e is None for e in ends.values()):
        _put(detail, "reported_outcome_agrees_with_telemetry", ctx, "phase record absent")
        return False

    recorded = {
        "target_metric": ends["fp32_control"].get("final_val_loss"),
        "baseline_metric": ends["standard_control"].get("final_val_loss"),
        "agent_metric": ends["agent_run"].get("final_val_loss"),
        "agent_overflow_events": ends["agent_run"].get("overflow_event_count"),
    }
    mismatched = [k for k, v in recorded.items() if rep.get(k) != v or ae.get(k) != v]
    want_score, want_reason = harness_reward(
        float(recorded["baseline_metric"]),
        float(recorded["target_metric"]),
        float(recorded["agent_metric"]),
        int(recorded["agent_overflow_events"]),
    )
    score_ok = finite(rep.get("score")) and abs(float(rep["score"]) - want_score) <= 1e-12
    reason_ok = rep.get("reason") == want_reason
    attempt_end_ok = ae.get("score") == rep.get("score") and ae.get("reason") == rep.get("reason")
    _put(detail, "reported_outcome_agrees_with_telemetry", ctx, {
        "mismatched_metrics": mismatched,
        "reported_score": rep.get("score"),
        "rederived_score": want_score,
        "reported_reason": rep.get("reason"),
        "rederived_reason": want_reason,
        "score_json_agrees_with_attempt_end": attempt_end_ok,
    })
    return not mismatched and score_ok and reason_ok and attempt_end_ok


def chk_attempt_wallclock_within_declared_budget(ctx, detail) -> bool:
    """VALUE. The recorded attempt duration sits inside the budget the fixture declares."""
    ae, bound = attempt_end(ctx.tel), budget_seconds(ctx.tel)
    if ae is None or bound is None:
        _put(detail, "attempt_wallclock_within_declared_budget", ctx, "attempt_end or attempt_begin absent")
        return False
    recorded = ae.get("wallclock_seconds")
    sp = ctx.artifacts / "score.json"
    reported = json.loads(sp.read_text()).get("wallclock_seconds") if sp.is_file() else None
    _put(detail, "attempt_wallclock_within_declared_budget", ctx, {
        "recorded_seconds": recorded,
        "reported_seconds": reported,
        "declared_bound_seconds": bound,
    })
    return (
        finite(recorded)
        and float(recorded) > 0.0
        and float(recorded) <= bound
        and recorded == reported
    )


def chk_policy_ran_outside_the_measuring_process(ctx, detail) -> bool:
    """INVARIANT. Every policy of the attempt was hosted in a process other than the driver.

    This is the structural property the whole bundle now rests on. A policy that shares an
    interpreter with the measuring loop can rebind any function in it, so a record that does
    not name a distinct worker process for each narrow phase is not evidence of a measurement.
    """
    ab = attempt_begin(ctx.tel)
    if ab is None:
        _put(detail, "policy_ran_outside_the_measuring_process", ctx, "attempt_begin absent")
        return False
    iso = ab.get("policy_isolation") or {}
    driver = iso.get("driver_pid")
    pids = iso.get("worker_pids") or {}
    need = [p for p in NARROW_PHASES if (phase_end(ctx.tel, p) or {}).get("from_cache") is not True]
    ok = (
        iso.get("mode") == "subprocess"
        and iso.get("worker") == "policy_worker.py"
        and isinstance(driver, int)
        and all(
            isinstance(pids.get(p), int) and pids.get(p) > 0 and pids.get(p) != driver for p in need
        )
    )
    _put(detail, "policy_ran_outside_the_measuring_process", ctx, {
        "mode": iso.get("mode"),
        "worker": iso.get("worker"),
        "driver_pid": driver,
        "worker_pids": pids,
        "phases_requiring_a_worker": need,
    })
    return bool(need) and ok


def chk_graded_record_authored_by_the_verifier(ctx, detail) -> bool:
    """VALUE. The record the reward was read off was produced by this verifier invocation.

    The nonce is drawn from system entropy inside this process at grade time and handed only to
    the attempt this process starts. Nothing written before this process existed can carry it,
    so a planted or replayed record fails here no matter how internally consistent it is.
    """
    ab = attempt_begin(ctx.tel)
    got = str((ab or {}).get("run_nonce", ""))
    sub_recorded = str((ab or {}).get("graded_submission_sha256", ""))
    sub_live = sha256_file(SUBMISSION) if SUBMISSION.is_file() else ""
    _put(detail, "graded_record_authored_by_the_verifier", ctx, {
        "authoritative_source": AUTHORITATIVE["source"],
        "nonce_expected": AUTHORITATIVE["nonce"],
        "nonce_recorded": got,
        "graded_submission": str(SUBMISSION),
        "graded_submission_sha256_live": sub_live,
        "graded_submission_sha256_recorded": sub_recorded,
    })
    if ctx.label != "verifier":
        # The agent-reported record is never claimed to be verifier-authored; this checker is
        # a property of the authoritative record alone.
        return True
    return (
        AUTHORITATIVE["source"] == "verifier_execution"
        and bool(AUTHORITATIVE["nonce"])
        and got == AUTHORITATIVE["nonce"]
        and bool(sub_live)
        and sub_recorded == sub_live
        and all(phase_end(ctx.tel, p) is not None for p in ALL_PHASES)
    )


def chk_agent_report_reproduces_under_verifier_execution(ctx, detail) -> bool:
    """DIVERGENCE. The attempt the agent reported and the attempt this verifier ran agree.

    Two independently produced records of the same frozen experiment, one of which the agent
    could write and one of which it could not. Independence here is independence of PRODUCER,
    not merely of arithmetic: the second record is authored by a process the submission has no
    handle on, from inputs it did not supply.
    """
    if ctx.label != "verifier":
        return True
    ref, mine = AGENT_TEL, ctx.tel
    if not ref:
        _put(detail, "agent_report_reproduces_under_verifier_execution", ctx, "agent record absent")
        return False
    tol = reproduction_tolerance()
    rows, ok = {}, True
    for phase in ALL_PHASES:
        a, v = phase_end(ref, phase), phase_end(mine, phase)
        if a is None or v is None:
            rows[phase] = "phase record absent"
            ok = False
            continue
        av, vv = a.get("final_val_loss"), v.get("final_val_loss")
        near = finite(av) and finite(vv) and abs(float(av) - float(vv)) <= tol
        same_overflow = int(a.get("overflow_event_count", -1)) == int(v.get("overflow_event_count", -2))
        rows[phase] = {
            "agent_reported": av,
            "verifier_measured": vv,
            "delta": (abs(float(av) - float(vv)) if finite(av) and finite(vv) else None),
            "within_tolerance": near,
            "overflow_events_agree": same_overflow,
        }
        ok = ok and near and same_overflow
    _put(detail, "agent_report_reproduces_under_verifier_execution", ctx, {
        "tolerance": tol,
        "phases": rows,
    })
    return ok


CHECKERS = [
    ("frozen_format_digest_matches", chk_frozen_format_digest_matches),
    ("no_overflow_flag_raised_in_agent_phase", chk_no_overflow_flag_raised_in_agent_phase),
    ("format_in_force_at_every_reduction", chk_format_in_force_at_every_reduction),
    ("agent_improves_on_standard_control", chk_agent_improves_on_standard_control),
    ("phases_ran_in_declared_order", chk_phases_ran_in_declared_order),
    ("reduction_agrees_with_independent_replay", chk_reduction_agrees_with_independent_replay),
    ("master_weights_updated_in_declared_dtype", chk_master_weights_updated_in_declared_dtype),
    ("frozen_harness_bytes_unmodified", chk_frozen_harness_bytes_unmodified),
    ("overflow_ledger_accounting_is_consistent", chk_overflow_ledger_accounting_is_consistent),
    ("agent_phase_covered_every_declared_step", chk_agent_phase_covered_every_declared_step),
    ("fp32_control_ran_outside_the_frozen_format", chk_fp32_control_ran_outside_the_frozen_format),
    ("baseline_anchor_ran_the_shipped_narrow_recipe", chk_baseline_anchor_ran_the_shipped_narrow_recipe),
    ("anchor_span_is_positive_and_finite", chk_anchor_span_is_positive_and_finite),
    ("all_phases_started_from_the_same_initialization", chk_all_phases_started_from_the_same_initialization),
    ("control_cache_agrees_with_graded_telemetry", chk_control_cache_agrees_with_graded_telemetry),
    ("reported_outcome_agrees_with_telemetry", chk_reported_outcome_agrees_with_telemetry),
    ("attempt_wallclock_within_declared_budget", chk_attempt_wallclock_within_declared_budget),
    ("policy_ran_outside_the_measuring_process", chk_policy_ran_outside_the_measuring_process),
    ("graded_record_authored_by_the_verifier", chk_graded_record_authored_by_the_verifier),
    ("agent_report_reproduces_under_verifier_execution", chk_agent_report_reproduces_under_verifier_execution),
]


def recompute_reward(tel: List[Dict[str, Any]]) -> Dict[str, Any]:
    a, s, f = (phase_end(tel, p) for p in ("agent_run", "standard_control", "fp32_control"))
    if a is None or s is None or f is None:
        return {"raw": 0.0, "reason": "phase_record_absent"}
    baseline, target, agent = float(s["final_val_loss"]), float(f["final_val_loss"]), float(a["final_val_loss"])
    span = baseline - target
    if not (span > 0.0):
        return {"raw": 0.0, "reason": "degenerate_anchor_span", "baseline": baseline, "target": target}
    if agent != agent:
        return {"raw": 0.0, "reason": "agent_loss_not_finite"}
    raw = (baseline - agent) / span
    return {
        "raw": raw,
        "reason": "scored",
        "baseline_metric": baseline,
        "target_metric": target,
        "agent_metric": agent,
    }


# --- the verifier's own measurement ---------------------------------------

AUTHORITATIVE: Dict[str, Any] = {"source": "none", "nonce": "", "dir": "", "rc": None}
AGENT_TEL: List[Dict[str, Any]] = []


def run_graded_attempt() -> Optional[List[Dict[str, Any]]]:
    """Start one graded attempt under this process and return the record it wrote.

    The output directory is named from the same entropy as the nonce, so nothing that existed
    before this process ran can have pre-planted a record there. The anchor cache is disabled,
    so both anchors are measured inside this invocation rather than read off a file some
    earlier attempt left behind.
    """
    nonce = secrets.token_hex(16)
    AUTHORITATIVE["nonce"] = nonce
    rundir = SCORE_PATH.parent / ("verifier_run_%s" % nonce)
    AUTHORITATIVE["dir"] = str(rundir)
    if EXTERNAL_AUTHORITATIVE:
        AUTHORITATIVE["source"] = "external"
        try:
            return json.loads(pathlib.Path(EXTERNAL_AUTHORITATIVE).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            AUTHORITATIVE["error"] = str(exc)
            return None
    if not SUBMISSION.is_file():
        AUTHORITATIVE["source"] = "submission_absent"
        AUTHORITATIVE["error"] = "no submission at %s" % SUBMISSION
        return None

    rundir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"BIA_RUN_NONCE": nonce, "BIA_NO_CACHE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env.pop("BIA_AUTHORITATIVE_TELEMETRY", None)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(ENVDIR / "run_attempt.py"),
                "--submission",
                str(SUBMISSION),
                "--out",
                str(rundir),
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ENVDIR),
            timeout=VERIFIER_RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        AUTHORITATIVE["source"] = "timeout"
        return None
    AUTHORITATIVE["rc"] = proc.returncode
    (SCORE_PATH.parent / "verifier-attempt-stdout.md").write_text(
        (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or "")
    )
    tel_path = rundir / "telemetry.json"
    if not tel_path.is_file():
        AUTHORITATIVE["source"] = "attempt_produced_no_record"
        return None
    AUTHORITATIVE["source"] = "verifier_execution"
    try:
        return json.loads(tel_path.read_text())
    except json.JSONDecodeError as exc:
        AUTHORITATIVE["source"] = "attempt_record_malformed"
        AUTHORITATIVE["error"] = str(exc)
        return None


def main() -> int:
    global AGENT_TEL
    outcomes: Dict[str, bool] = {name: False for name, _ in CHECKERS}
    detail: Dict[str, Any] = {}

    auth_tel = run_graded_attempt()
    detail["authoritative_record"] = dict(AUTHORITATIVE)
    if auth_tel is None:
        return emit(
            0.0,
            "verifier_execution_failed:%s" % AUTHORITATIVE["source"],
            detail,
            outcomes,
        )
    auth_dir = (
        pathlib.Path(AUTHORITATIVE["dir"])
        if AUTHORITATIVE["source"] == "verifier_execution"
        else pathlib.Path(EXTERNAL_AUTHORITATIVE).parent
    )

    agent_path = ARTIFACTS / "telemetry.json"
    if not agent_path.is_file():
        detail["agent_record"] = {"looked_at": str(agent_path), "present": False}
    else:
        try:
            loaded = json.loads(agent_path.read_text())
            AGENT_TEL = loaded if isinstance(loaded, list) else []
            detail["agent_record"] = {"present": True, "records": len(AGENT_TEL)}
        except json.JSONDecodeError as exc:
            detail["agent_record"] = {"present": True, "malformed": str(exc)}

    contexts = [Ctx("verifier", auth_tel, auth_dir), Ctx("agent", AGENT_TEL, ARTIFACTS)]
    for name, fn in CHECKERS:
        verdict = True
        for ctx in contexts:
            try:
                verdict = bool(fn(ctx, detail)) and verdict
            except Exception as exc:  # a checker that throws is a failing checker, never a pass
                verdict = False
                _put(detail, name, ctx, "checker_raised: %s" % exc)
        outcomes[name] = verdict

    # A non-numeric final_val_loss raises inside recompute_reward. Outside this guard that
    # exception escaped before any score was written, so the run came back unscored rather than
    # scored zero, which reads as an infrastructure fault instead of a graded outcome.
    try:
        rw = recompute_reward(auth_tel)
    except (TypeError, ValueError, KeyError) as exc:
        rw = {"raw": 0.0, "reason": "reward_recompute_raised", "error": str(exc)}
    detail["reward"] = rw
    detail["reward_source"] = "verifier-authored record at %s" % AUTHORITATIVE["dir"]

    failed = [n for n, v in outcomes.items() if not v]
    if failed:
        return emit(0.0, "checker_failed:" + ",".join(sorted(failed)), detail, outcomes)

    try:
        score = min(max(float(rw["raw"]), 0.0), 1.0)
    except (TypeError, ValueError):
        return emit(0.0, "reward_not_finite", detail, outcomes)
    return emit(score, rw["reason"], detail, outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
