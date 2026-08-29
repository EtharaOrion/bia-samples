"""Turn a measured crossing step into a reward and a verdict.

`reward` is a float in [0, 1] answering how much of the distance from the
shipped baseline to the reference the submission covered on the graded axis.
`pass` is 0 or 1 answering whether it cleared the bound threshold.

THE ANCHOR GUARD. Anchors must carry `measured: true` and a provenance string
naming the run that produced them. An unmeasured anchor set is allowed to exist,
because the bundle has to be authorable before its operating point is measured,
but it can only ever produce reward 0.0 with reason `anchors-unmeasured` and can
never carry a pass. The prior design for this slot instead hardcoded
BASELINE = 3250 and TARGET = 2690 as module constants lifted from the published
upstream record table, so the score was denominated in numbers nobody had
measured on these bytes and the full-reward point was a published world record.

THE MAGNITUDE COMES FROM THE MEASUREMENT. `compute` takes the crossing the
verifier found. There is no path by which a number the submission wrote reaches
the reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from crossing import Crossing


class AnchorsUnmeasured(Exception):
    """Raised when anchors have not been measured at the bound operating point."""


@dataclass(frozen=True)
class Anchors:
    baseline_steps: int
    oracle_steps: int
    target_loss: float
    margin: float
    min_seeds: int
    pass_threshold: float
    claim_tolerance_steps: int
    measured: bool = False
    provenance: str = ""

    def validate(self) -> None:
        if self.baseline_steps <= self.oracle_steps:
            raise ValueError(
                "anchors invert: the baseline must be worse (more optimizer steps) "
                f"than the reference; got baseline={self.baseline_steps}, "
                f"oracle={self.oracle_steps}")
        if self.margin <= 0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        if self.min_seeds < 2:
            raise ValueError(
                "min_seeds must be at least 2; the seed interface is proven by two "
                f"seeds moving the measured loss, got {self.min_seeds}")
        if not 0.0 < self.pass_threshold <= 1.0:
            raise ValueError(
                "pass_threshold must sit in (0, 1]; a threshold of 0 would pass every "
                f"submission including a broken one, got {self.pass_threshold}")
        if self.claim_tolerance_steps < 0:
            raise ValueError("claim_tolerance_steps must not be negative")

    def require_measured(self) -> None:
        if not self.measured:
            raise AnchorsUnmeasured(
                "anchors are not measured at the bound operating point; scoring is "
                "refused. baseline_steps and oracle_steps must be obtained by running "
                "environment/train_baseline.py and solution/recipe.py at the shape "
                "environment/shape.json binds, and the run recorded in provenance.")
        if not self.provenance.strip():
            raise AnchorsUnmeasured(
                "anchors claim measured: true but carry no provenance string naming "
                "what was run to obtain them")


@dataclass
class Reward:
    value: float
    reason: str
    passed: bool = False
    checkers: dict[str, bool] = field(default_factory=dict)
    measurement: dict = field(default_factory=dict)

    def as_payload(self) -> dict:
        return {"reward": round(float(self.value), 6),
                "pass": 1 if self.passed else 0,
                "reason": self.reason,
                "checkers": dict(sorted(self.checkers.items())),
                "measurement": self.measurement}


def gap_closed(crossing_step: int, anchors: Anchors) -> float:
    span = anchors.baseline_steps - anchors.oracle_steps
    return min(max((anchors.baseline_steps - crossing_step) / span, 0.0), 1.0)


def claim_reconciles(claimed_steps, crossing_step: int, tolerance: int) -> bool:
    """The declared crossing step must match the one the verifier measured.

    Mandatory, never opt-out. The previous verifier read
    `consistent_with(claimed, losses) if claimed else True`, so a submission that
    declined to make a claim faced no reconciliation at all and the only check
    able to cross-examine its output disabled itself on request. Here an absent
    or unparseable claim is a failure, because silence about a claim is not a
    claim that reconciles.

    This checker can only lower an outcome. The reward magnitude is computed
    from `crossing_step` whatever the claim says.
    """
    if claimed_steps is None:
        return False
    try:
        declared = int(claimed_steps)
    except (TypeError, ValueError):
        return False
    return abs(declared - crossing_step) <= tolerance


def compute(crossing: Crossing, anchors: Anchors, validity: dict[str, bool],
            claimed_steps) -> Reward:
    anchors.validate()
    checkers = dict(validity)
    try:
        anchors.require_measured()
    except AnchorsUnmeasured:
        checkers["anchors_measured"] = False
        return Reward(0.0, "anchors-unmeasured", False, checkers)
    checkers["anchors_measured"] = True

    failed = [k for k, ok in sorted(validity.items()) if not ok]
    if failed:
        return Reward(0.0, f"validity-failed:{failed[0]}", False, checkers)

    checkers["target_reached_with_significance"] = crossing.step is not None
    if crossing.step is None:
        return Reward(0.0, crossing.reason, False, checkers,
                      {"n_seeds": crossing.n_seeds, "best_mean_loss": crossing.mean_loss})

    checkers["claim_reconciles"] = claim_reconciles(
        claimed_steps, crossing.step, anchors.claim_tolerance_steps)
    if not checkers["claim_reconciles"]:
        return Reward(0.0, "claim-does-not-reconcile", False, checkers,
                      {"measured_crossing_step": crossing.step,
                       "claimed_steps": claimed_steps})

    value = gap_closed(crossing.step, anchors)
    passed = value >= anchors.pass_threshold
    checkers["improved_enough_to_pass"] = passed
    return Reward(value, "scored" if passed else "scored-below-pass-threshold",
                  passed, checkers,
                  {"measured_crossing_step": crossing.step,
                   "mean_loss_at_crossing": crossing.mean_loss,
                   "n_seeds": crossing.n_seeds,
                   "margin_achieved": crossing.margin_achieved,
                   "baseline_steps": anchors.baseline_steps,
                   "oracle_steps": anchors.oracle_steps,
                   "pass_threshold": anchors.pass_threshold})
