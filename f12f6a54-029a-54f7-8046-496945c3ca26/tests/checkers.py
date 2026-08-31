"""The nine graded checkers for slot OER-10. Pure, deterministic, live state only.

Every checker here is a pure function of one `HarnessView`, which wraps the telemetry
document the VERIFIER'S OWN PROCESS produced: tests/runner.py executes the submission in
isolation and records what it observed, and tests/grade.py hands the resulting mapping in.
Nothing in this module reads a planted file, an environment secret, a clock, a random
source or the network, and nothing here imports the submission. The whole import surface
is json, hashlib, math, pathlib, dataclasses and typing.

If a checker needs a number it takes it from that telemetry record. It never takes it from
a value the submission printed, wrote or reported: the submission's own reported loss
travels in the record under `evaluation.submission_reported` precisely so a checker can
prove it is NOT what was graded.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TELEMETRY_SCHEMA = "oer10.telemetry/v1"

STAGE_PIN = "pin-exclusion-set"
STAGE_SCREEN = "recipe-fingerprint-screen"
STAGE_TRAIN = "train"

VERIFIER_SOURCE = "verifier-recompute"
NO_SMOOTHING = "none"
HARNESS_OWNER = "harness"

# A halt for any of these reasons is a run that stopped because an evaluation looked
# good, which is exactly what must not be read as having established a loss.
HARVEST_HALTS = ("favourable-eval", "early-stop", "target-reached", "loss-crossed")

# The nine machine-readable zero reasons, one per checker.
REASON_REPLAY = "recipe-matches-published-mixture"
REASON_SCREEN_ORDER = "screen-ran-after-training"
REASON_NOT_VERIFIER = "graded-loss-not-verifier-computed"
REASON_SMOOTHED = "readout-smoothed-on-graded-path"
REASON_NOT_SUSTAINED = "loss-not-sustained"
REASON_EARLY_STOP = "early-stop-not-a-result"
REASON_WEIGHTS = "weights-not-harness-owned"
REASON_BUDGET = "token-budget-overspent"
REASON_LEAK = "eval-split-leaked"


@dataclass(frozen=True)
class Outcome:
    """One checker's verdict. A failing verdict always carries its reason code."""

    ident: str
    passed: bool
    reason: str
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.ident,
            "passed": self.passed,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class HarnessView:
    """The live-state handle. Wraps the telemetry the verifier's own process produced."""

    telemetry: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Optional[Dict[str, Any]]) -> "HarnessView":
        return cls(dict(payload) if isinstance(payload, dict) else {})

    @classmethod
    def from_path(cls, path: Path) -> "HarnessView":
        if not path.is_file():
            return cls({})
        try:
            return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            return cls({})

    @property
    def present(self) -> bool:
        return self.telemetry.get("schema") == TELEMETRY_SCHEMA

    def block(self, name: str) -> Dict[str, Any]:
        value = self.telemetry.get(name)
        return value if isinstance(value, dict) else {}

    def stages(self) -> Dict[str, int]:
        rows: Dict[str, int] = {}
        for raw in self.telemetry.get("stages") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("stage", "")).strip()
            try:
                seq = int(raw.get("seq"))
            except (TypeError, ValueError):
                continue
            if name and name not in rows:
                rows[name] = seq
        return rows

    def samples(self) -> List[Dict[str, Any]]:
        rows = self.block("evaluation").get("samples")
        return [row for row in rows or [] if isinstance(row, dict)]

    def bound_step(self) -> Optional[int]:
        try:
            return int(self.block("evaluation").get("bound_eval_step"))
        except (TypeError, ValueError):
            return None

    def graded_samples(self) -> List[Dict[str, Any]]:
        step = self.bound_step()
        if step is None:
            return []
        return [row for row in self.samples() if _as_int(row.get("step")) == step]

    def graded_loss(self) -> Optional[float]:
        rows = self.graded_samples()
        if len(rows) != 1:
            return None
        return _as_float(rows[0].get("loss"))


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fail(ident: str, reason: str, detail: str) -> Outcome:
    return Outcome(ident, False, reason, detail)


def _pass(ident: str, detail: str) -> Outcome:
    return Outcome(ident, True, "", detail)


def digest_vector(vector: List[int]) -> str:
    """The behavioural fingerprint: sha256 over the canonical bytes of the selection vector."""
    payload = json.dumps([int(value) for value in vector], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def l1_distance(left: List[int], right: List[int]) -> Optional[int]:
    if len(left) != len(right):
        return None
    return sum(abs(int(a) - int(b)) for a, b in zip(left, right))


def _vector(raw: Any) -> Optional[List[int]]:
    if not isinstance(raw, list) or not raw:
        return None
    out: List[int] = []
    for item in raw:
        value = _as_int(item)
        if value is None:
            return None
        out.append(value)
    return out


# ---------------------------------------------------------------------------
# 1. DIVERGENCE. The load-bearing record-displacement control.
# ---------------------------------------------------------------------------


def check_recipe_not_a_replay(view: HarnessView) -> Outcome:
    """The recipe's behavioural fingerprint diverges from every pinned published mixture.

    Behavioural, not textual. The vector is what the submitted recipe DID to the frozen
    probe pool, recorded by the verifier's own runner, so a renamed, reordered or
    cosmetically re-thresholded restatement of a published mixture selects the same probe
    documents at the same weights and is caught at the proximity level even though its
    source bytes and its digest both differ.
    """
    ident = "recipe_not_a_replay"
    if not view.present:
        return _fail(ident, REASON_REPLAY, "no telemetry, so no divergence is established")
    fingerprint = view.block("recipe_fingerprint")
    exclusion = view.block("exclusion_set")
    vector = _vector(fingerprint.get("selection_vector"))
    if vector is None:
        return _fail(ident, REASON_REPLAY, "the recipe produced no readable selection vector")
    if not bool(exclusion.get("pinned")):
        return _fail(
            ident,
            REASON_REPLAY,
            "the published-baseline exclusion set is not pinned, so a not-a-replay verdict "
            "over it would clear every recipe including a verbatim replay",
        )
    entries = [row for row in exclusion.get("entries") or [] if isinstance(row, dict)]
    if not entries:
        return _fail(ident, REASON_REPLAY, "the pinned exclusion set is empty")
    floor = _as_int(exclusion.get("l1_floor"))
    if floor is None or floor <= 0:
        return _fail(ident, REASON_REPLAY, "no positive L1 floor is pinned for the proximity screen")

    recomputed = digest_vector(vector)
    if str(fingerprint.get("digest", "")) != recomputed:
        return _fail(
            ident,
            REASON_REPLAY,
            "the recorded fingerprint digest does not recompute from the selection vector",
        )

    nearest: Optional[Tuple[int, str]] = None
    for row in entries:
        other = _vector(row.get("selection_vector"))
        if other is None:
            return _fail(ident, REASON_REPLAY, "a pinned baseline carries no readable vector")
        if str(row.get("digest", "")) == recomputed:
            return _fail(
                ident,
                REASON_REPLAY,
                "verbatim replay: the fingerprint equals pinned baseline " + str(row.get("id")),
            )
        distance = l1_distance(vector, other)
        if distance is None:
            return _fail(ident, REASON_REPLAY, "a pinned baseline vector is a different length")
        if nearest is None or distance < nearest[0]:
            nearest = (distance, str(row.get("id")))
    if nearest is None:
        return _fail(ident, REASON_REPLAY, "no pinned baseline was comparable")
    if nearest[0] < floor:
        return _fail(
            ident,
            REASON_REPLAY,
            "behavioural restatement: L1 " + str(nearest[0]) + " to pinned baseline "
            + nearest[1] + " is below the pinned floor of " + str(floor),
        )
    return _pass(
        ident,
        "nearest pinned baseline " + nearest[1] + " at L1 " + str(nearest[0])
        + ", pinned floor " + str(floor),
    )


# ---------------------------------------------------------------------------
# 2. ORDERING. The screen ran before any training executed.
# ---------------------------------------------------------------------------


def check_fingerprint_screen_precedes_training(view: HarnessView) -> Outcome:
    """The exclusion set was pinned, then the screen ran, and only then did training run.

    Ordering is the whole economics of the control: a screen that runs after training
    grades a number a replay has already bought, so the exclusion set becomes decorative
    and a replay costs a full accelerator budget instead of nothing.
    """
    ident = "fingerprint_screen_precedes_training"
    if not view.present:
        return _fail(ident, REASON_SCREEN_ORDER, "no telemetry, so no ordering is established")
    stages = view.stages()
    if STAGE_SCREEN not in stages:
        return _fail(ident, REASON_SCREEN_ORDER, "no fingerprint screen stage was recorded")
    if STAGE_PIN not in stages:
        return _fail(ident, REASON_SCREEN_ORDER, "no exclusion-set pinning stage was recorded")
    if stages[STAGE_PIN] >= stages[STAGE_SCREEN]:
        return _fail(
            ident,
            REASON_SCREEN_ORDER,
            "the exclusion set was pinned at stage " + str(stages[STAGE_PIN])
            + ", not before the screen at stage " + str(stages[STAGE_SCREEN]),
        )
    steps = _as_int(view.block("training").get("steps_executed")) or 0
    if STAGE_TRAIN not in stages:
        if steps:
            return _fail(
                ident,
                REASON_SCREEN_ORDER,
                str(steps) + " training steps executed with no training stage recorded",
            )
        return _pass(ident, "the screen ran and no training stage followed, at zero accelerator cost")
    if stages[STAGE_SCREEN] >= stages[STAGE_TRAIN]:
        return _fail(
            ident,
            REASON_SCREEN_ORDER,
            "the screen ran at stage " + str(stages[STAGE_SCREEN])
            + ", at or after training at stage " + str(stages[STAGE_TRAIN]),
        )
    return _pass(
        ident,
        "pin at " + str(stages[STAGE_PIN]) + ", screen at " + str(stages[STAGE_SCREEN])
        + ", training at " + str(stages[STAGE_TRAIN]),
    )


# ---------------------------------------------------------------------------
# 3, 4, 7. VALUE. The graded readout, and where it came from.
# ---------------------------------------------------------------------------


def check_graded_loss_is_verifier_computed(view: HarnessView) -> Outcome:
    """The graded loss is the verifier's own evaluation at the bound evaluation point."""
    ident = "graded_loss_is_verifier_computed"
    if not view.present:
        return _fail(ident, REASON_NOT_VERIFIER, "no telemetry, so no graded loss exists")
    step = view.bound_step()
    if step is None:
        return _fail(ident, REASON_NOT_VERIFIER, "no bound evaluation point is recorded")
    rows = view.graded_samples()
    if len(rows) != 1:
        return _fail(
            ident,
            REASON_NOT_VERIFIER,
            str(len(rows)) + " evaluation samples sit at the bound point, exactly one is required",
        )
    row = rows[0]
    if str(row.get("source", "")) != VERIFIER_SOURCE:
        return _fail(
            ident,
            REASON_NOT_VERIFIER,
            "the graded sample carries source " + repr(str(row.get("source", "")))
            + ", not " + repr(VERIFIER_SOURCE),
        )
    loss = _as_float(row.get("loss"))
    if loss is None:
        return _fail(ident, REASON_NOT_VERIFIER, "the graded sample carries no finite loss")
    reported = view.block("evaluation").get("submission_reported")
    note = ""
    if isinstance(reported, dict):
        told = _as_float(reported.get("loss"))
        note = "; the submission reported " + repr(told) + " and it was not graded"
    return _pass(ident, "verifier-computed loss " + repr(loss) + " at step " + str(step) + note)


def check_graded_readout_unsmoothed(view: HarnessView) -> Outcome:
    """No smoothing on the graded path. The graded readout is the raw evaluation."""
    ident = "graded_readout_unsmoothed"
    if not view.present:
        return _fail(ident, REASON_SMOOTHED, "no telemetry, so no readout is established")
    rows = view.graded_samples()
    if len(rows) != 1:
        return _fail(ident, REASON_SMOOTHED, "the graded sample is not uniquely identified")
    smoothing = str(rows[0].get("smoothing", ""))
    if smoothing != NO_SMOOTHING:
        return _fail(
            ident,
            REASON_SMOOTHED,
            "the graded sample carries smoothing " + repr(smoothing)
            + "; a submission may smooth for its own use but never on the graded path",
        )
    for row in view.samples():
        if str(row.get("source", "")) != VERIFIER_SOURCE:
            continue
        if str(row.get("smoothing", "")) != NO_SMOOTHING:
            return _fail(
                ident,
                REASON_SMOOTHED,
                "a verifier sample at step " + str(row.get("step")) + " carries smoothing "
                + repr(str(row.get("smoothing", ""))),
            )
    return _pass(ident, "every verifier-computed sample on the graded path is unsmoothed")


def check_evaluated_weights_harness_owned(view: HarnessView) -> Outcome:
    """The weights evaluated are the weights the run produced, not a chosen checkpoint."""
    ident = "evaluated_weights_harness_owned"
    if not view.present:
        return _fail(ident, REASON_WEIGHTS, "no telemetry, so no weight provenance exists")
    rows = view.graded_samples()
    if len(rows) != 1:
        return _fail(ident, REASON_WEIGHTS, "the graded sample is not uniquely identified")
    row = rows[0]
    if str(row.get("weights_owner", "")) != HARNESS_OWNER:
        return _fail(
            ident,
            REASON_WEIGHTS,
            "the graded sample's weights are owned by " + repr(str(row.get("weights_owner", ""))),
        )
    step = _as_int(row.get("step"))
    recorded = None
    for entry in view.telemetry.get("checkpoints") or []:
        if isinstance(entry, dict) and _as_int(entry.get("step")) == step:
            recorded = entry
            break
    if recorded is None:
        return _fail(
            ident, REASON_WEIGHTS, "no harness checkpoint record exists at step " + str(step)
        )
    if str(recorded.get("owner", "")) != HARNESS_OWNER:
        return _fail(ident, REASON_WEIGHTS, "the checkpoint at step " + str(step) + " is not harness-owned")
    if str(recorded.get("digest", "")) != str(row.get("weights_digest", "")):
        return _fail(
            ident,
            REASON_WEIGHTS,
            "the evaluated weights digest does not match the harness checkpoint at step " + str(step),
        )
    return _pass(ident, "graded weights match the harness checkpoint at step " + str(step))


# ---------------------------------------------------------------------------
# 5. INVARIANT. The level held, it was not a dip.
# ---------------------------------------------------------------------------


def check_loss_sustained_at_scheduled_points(view: HarnessView) -> Outcome:
    """The graded level holds at every evaluation point the verifier itself scheduled."""
    ident = "loss_sustained_at_scheduled_points"
    if not view.present:
        return _fail(ident, REASON_NOT_SUSTAINED, "no telemetry, so nothing was sustained")
    graded = view.graded_loss()
    step = view.bound_step()
    if graded is None or step is None:
        return _fail(ident, REASON_NOT_SUSTAINED, "no unique graded loss to sustain")
    evaluation = view.block("evaluation")
    points = [value for value in (_as_int(item) for item in evaluation.get("sustain_points") or []) if value is not None]
    if len(points) < 2:
        return _fail(
            ident,
            REASON_NOT_SUSTAINED,
            "fewer than two scheduled sustain points, so a single dip could be harvested",
        )
    if step not in points:
        return _fail(ident, REASON_NOT_SUSTAINED, "the bound evaluation point is not a sustain point")
    tolerance = _as_float(evaluation.get("sustain_tolerance"))
    if tolerance is None or tolerance < 0.0:
        return _fail(ident, REASON_NOT_SUSTAINED, "no non-negative sustain tolerance is bound")
    by_step = {}
    for row in view.samples():
        if str(row.get("source", "")) != VERIFIER_SOURCE:
            continue
        key = _as_int(row.get("step"))
        value = _as_float(row.get("loss"))
        if key is not None and value is not None:
            by_step[key] = value
    for point in sorted(points):
        if point not in by_step:
            return _fail(
                ident,
                REASON_NOT_SUSTAINED,
                "no verifier evaluation exists at scheduled sustain point " + str(point),
            )
        if by_step[point] > graded + tolerance:
            return _fail(
                ident,
                REASON_NOT_SUSTAINED,
                "loss at sustain point " + str(point) + " is " + repr(by_step[point])
                + ", above the graded level " + repr(graded) + " plus tolerance " + repr(tolerance),
            )
    return _pass(
        ident,
        "the graded level held at all " + str(len(points)) + " scheduled sustain points",
    )


# ---------------------------------------------------------------------------
# 6, 9. ABSENCE. What must not be present in the record.
# ---------------------------------------------------------------------------


def check_early_stop_not_a_result(view: HarnessView) -> Outcome:
    """No halt on a favourable evaluation before the scheduled sustain points were reached.

    A run that stops the moment an evaluation looks good is graded as not having
    established the loss, with a reason, and never as an absent result.
    """
    ident = "early_stop_not_a_result"
    if not view.present:
        return _fail(ident, REASON_EARLY_STOP, "no telemetry, so no completed run exists")
    training = view.block("training")
    evaluation = view.block("evaluation")
    halted = _as_int(training.get("halted_at_step"))
    if halted is None:
        return _fail(ident, REASON_EARLY_STOP, "the run records no halt step")
    reason = str(training.get("halt_reason", ""))
    if reason in HARVEST_HALTS:
        return _fail(
            ident,
            REASON_EARLY_STOP,
            "the run halted with reason " + repr(reason) + ", which is a harvested evaluation",
        )
    points = [value for value in (_as_int(item) for item in evaluation.get("sustain_points") or []) if value is not None]
    if not points:
        return _fail(ident, REASON_EARLY_STOP, "no scheduled sustain points to have reached")
    if halted < max(points):
        return _fail(
            ident,
            REASON_EARLY_STOP,
            "the run halted at step " + str(halted) + ", before the last scheduled sustain point "
            + str(max(points)),
        )
    return _pass(ident, "the run ran past every scheduled sustain point, halting at " + str(halted))


def check_eval_split_not_trained_on(view: HarnessView) -> Outcome:
    """No document of the frozen evaluation split appears in the corpus that was fed."""
    ident = "eval_split_not_trained_on"
    if not view.present:
        return _fail(ident, REASON_LEAK, "no telemetry, so no corpus is accounted for")
    corpus = view.block("corpus")
    bound = str(corpus.get("bound_held_out_split_id", ""))
    used = str(corpus.get("held_out_split_id", ""))
    if not bound or bound != used:
        return _fail(
            ident,
            REASON_LEAK,
            "the split held out was " + repr(used) + ", not the bound split " + repr(bound),
        )
    held = corpus.get("held_out_doc_ids")
    fed = corpus.get("fed_doc_ids")
    if not isinstance(held, list) or not held:
        return _fail(ident, REASON_LEAK, "the held-out split records no document ids")
    if not isinstance(fed, list) or not fed:
        return _fail(ident, REASON_LEAK, "the fed corpus records no document ids")
    overlap = sorted(set(str(item) for item in fed) & set(str(item) for item in held))
    if overlap:
        return _fail(
            ident,
            REASON_LEAK,
            str(len(overlap)) + " held-out documents were fed, starting at " + overlap[0],
        )
    return _pass(
        ident,
        str(len(fed)) + " documents fed, none of the " + str(len(held)) + " held-out documents among them",
    )


# ---------------------------------------------------------------------------
# 8. EFFECT. What the run actually did to the frozen budget.
# ---------------------------------------------------------------------------


def check_token_budget_respected_as_fed(view: HarnessView) -> Outcome:
    """The tokens the harness counted as fed stayed inside the frozen budget.

    Graded as fed, never as declared: a plan that declares the frozen budget and then feeds
    more is exactly the shape this closes.
    """
    ident = "token_budget_respected_as_fed"
    if not view.present:
        return _fail(ident, REASON_BUDGET, "no telemetry, so no token count exists")
    training = view.block("training")
    budget = _as_int(training.get("frozen_token_budget"))
    fed = _as_int(training.get("tokens_fed"))
    if budget is None or budget <= 0:
        return _fail(ident, REASON_BUDGET, "no positive frozen token budget is bound")
    if fed is None or fed <= 0:
        return _fail(ident, REASON_BUDGET, "no positive fed token count was recorded")
    if fed > budget:
        return _fail(
            ident,
            REASON_BUDGET,
            str(fed) + " tokens fed against a frozen budget of " + str(budget),
        )
    return _pass(ident, str(fed) + " tokens fed, frozen budget " + str(budget))


# ---------------------------------------------------------------------------
# The graded set, in manifest order.
# ---------------------------------------------------------------------------

CHECKERS = (
    check_recipe_not_a_replay,
    check_fingerprint_screen_precedes_training,
    check_graded_loss_is_verifier_computed,
    check_graded_readout_unsmoothed,
    check_loss_sustained_at_scheduled_points,
    check_early_stop_not_a_result,
    check_evaluated_weights_harness_owned,
    check_token_budget_respected_as_fed,
    check_eval_split_not_trained_on,
)


def run_all(view: HarnessView) -> List[Outcome]:
    """Every graded checker, in manifest order, over one live-state handle."""
    return [checker(view) for checker in CHECKERS]
