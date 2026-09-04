"""Deployment admission: which probed candidate may earn execution rights.

The strict rule -- a Support probe grants deployment only when the Episode
relation is POSITIVE, i.e. the aggregate gain clears the material line *and no
single series is materially harmed* -- is the frozen default here, and
``DEFAULT`` reproduces the pre-parameterisation behaviour exactly.

Why this is its own module and not a field on ``ExplorationPolicy``.  That
module's own contract says the harm threshold and the gates are "不在本面内",
and its ``LEGAL_DOMAINS`` is documented as the sampling space of the Stage-3
Random-legal-edit arm.  A deployment admission rule *is* a harm gate, so
putting it there would let a random legal edit flip a safety gate.  This module
is therefore deliberately outside that sampling space: only an experiment
runner may install a non-default rule, and it must reset afterwards.

``bounded_risk_v1`` is the P4b rule (docs/P4B_BOUNDED_RISK_GATE_PREREGISTRATION
_2026-08-31.md §1): an aggregate-positive candidate may deploy while carrying
local harm, provided the harmed fraction and the worst single-series loss both
stay inside a frozen budget.  It fails closed -- with no per-series reading
available it falls back to the strict rule rather than admitting blind.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .signed_radius import MATERIAL_THRESHOLD

STRICT = "strict_positive_only"
BOUNDED_V1 = "bounded_risk_v1"
LEGAL_RULES = (STRICT, BOUNDED_V1)


@dataclass(frozen=True)
class AdmissionPolicy:
    """Frozen deployment-admission rule.  DEFAULT == today's strict gate."""

    rule: str = STRICT
    # Only read by BOUNDED_V1.  Fractions of the probed series, and the worst
    # tolerated single-series loss in Consumer utility units -- the latter is
    # calibrated per Consumer/metric and does not transfer across them.
    max_harmed_fraction: float = 0.0
    max_single_series_harm: float = 0.0

    def validate(self) -> "AdmissionPolicy":
        if self.rule not in LEGAL_RULES:
            raise ValueError(
                "illegal admission rule %r (legal: %r)" % (self.rule, LEGAL_RULES)
            )
        if not 0.0 <= float(self.max_harmed_fraction) <= 1.0:
            raise ValueError(
                "max_harmed_fraction must be a fraction, got %r"
                % (self.max_harmed_fraction,)
            )
        if float(self.max_single_series_harm) < 0.0:
            raise ValueError(
                "max_single_series_harm must be non-negative, got %r"
                % (self.max_single_series_harm,)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdmissionVerdict:
    """One admission decision, carried into the probe record for audit."""

    admitted: bool
    rule: str
    reason: str
    aggregate_gain: float | None = None
    series_count: int | None = None
    harmed_count: int | None = None
    harmed_fraction: float | None = None
    max_single_series_harm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT = AdmissionPolicy()
_active: AdmissionPolicy = DEFAULT


def active_policy() -> AdmissionPolicy:
    return _active


def install_policy(policy: AdmissionPolicy) -> AdmissionPolicy:
    global _active
    _active = policy.validate()
    return _active


def reset_policy() -> AdmissionPolicy:
    global _active
    _active = DEFAULT
    return _active


def _finite(value: object) -> float | None:
    """A usable reading, or None.  NaN and +/-Inf are not usable."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _per_series_values(
    per_series_gains: Sequence[float] | Mapping[str, float] | None,
) -> list[float] | None:
    if per_series_gains is None:
        return None
    values = (
        list(per_series_gains.values())
        if isinstance(per_series_gains, Mapping)
        else list(per_series_gains)
    )
    if not values:
        return None
    numbers = [_finite(value) for value in values]
    # One unusable series is enough to make the budget uncheckable, so the
    # whole reading is discarded rather than silently averaged over.
    return None if any(number is None for number in numbers) else numbers  # type: ignore[return-value]


def _budget_verdict(
    *,
    aggregate: float,
    series_count: int,
    harmed_count: int,
    worst_harm: float,
    policy: AdmissionPolicy,
) -> AdmissionVerdict:
    harmed_fraction = harmed_count / series_count
    within = (
        harmed_fraction <= float(policy.max_harmed_fraction)
        and worst_harm <= float(policy.max_single_series_harm)
    )
    if within:
        reason = "within_risk_budget"
    elif harmed_fraction > float(policy.max_harmed_fraction):
        reason = "harmed_fraction_over_budget"
    else:
        reason = "single_series_harm_over_budget"
    return AdmissionVerdict(
        admitted=within,
        rule=BOUNDED_V1,
        reason=reason,
        aggregate_gain=aggregate,
        series_count=series_count,
        harmed_count=harmed_count,
        harmed_fraction=harmed_fraction,
        max_single_series_harm=worst_harm,
    )


def decide(
    *,
    relation: str,
    aggregate_gain: float | None,
    per_series_gains: Sequence[float] | Mapping[str, float] | None = None,
    policy: AdmissionPolicy | None = None,
    material_threshold: float = MATERIAL_THRESHOLD,
) -> AdmissionVerdict:
    """Does this probe earn deployment rights under the active rule?"""
    active = policy if policy is not None else _active
    if active.rule == STRICT:
        positive = str(relation) == "POSITIVE"
        return AdmissionVerdict(
            admitted=positive,
            rule=STRICT,
            reason="relation_positive" if positive else "relation_not_positive",
            aggregate_gain=None if aggregate_gain is None else float(aggregate_gain),
        )

    # BOUNDED_V1 from here.  A POSITIVE relation already means "aggregate clears
    # the line and nothing is materially harmed", so it is admitted on the same
    # terms as before and needs no per-series reading of its own.
    if str(relation) == "POSITIVE":
        return AdmissionVerdict(
            admitted=True,
            rule=BOUNDED_V1,
            reason="relation_positive",
            aggregate_gain=None if aggregate_gain is None else float(aggregate_gain),
        )
    aggregate = _finite(aggregate_gain)
    if aggregate is None:
        # NaN/Inf is not a reading.  Fail closed rather than let a broken
        # instrument compare its way past a safety budget.
        return AdmissionVerdict(
            admitted=False,
            rule=BOUNDED_V1,
            reason="non_finite_aggregate_fail_closed",
        )
    if aggregate < material_threshold:
        return AdmissionVerdict(
            admitted=False,
            rule=BOUNDED_V1,
            reason="aggregate_below_material_line",
            aggregate_gain=aggregate,
        )
    values = _per_series_values(per_series_gains)
    if values is None:
        # Fail closed: without a usable per-series split the budget cannot be
        # checked, so fall back to the strict rule rather than admit blind.
        return AdmissionVerdict(
            admitted=False,
            rule=BOUNDED_V1,
            reason="no_per_series_reading_fail_closed",
            aggregate_gain=aggregate,
        )
    lowest = min(values)
    return _budget_verdict(
        aggregate=aggregate,
        series_count=len(values),
        harmed_count=sum(1 for value in values if value < -material_threshold),
        worst_harm=-lowest if lowest < 0.0 else 0.0,
        policy=active,
    )


def decide_from_facts(
    facts: Mapping[str, Any] | None,
    *,
    policy: AdmissionPolicy | None = None,
    material_threshold: float = MATERIAL_THRESHOLD,
) -> AdmissionVerdict:
    """Same decision, taken from a ``classify_relation`` fact summary.

    The method layer already holds that summary at both persistence gates --
    including on the path that reuses the probe's recorded reading rather than
    re-evaluating -- and it carries ``series_read``, ``harmed_series_count``
    and ``min_per_series_gain``, which is exactly the budget's input.  Reading
    it here keeps the two layers deciding from one set of numbers.
    """
    active = policy if policy is not None else _active
    summary = dict(facts or {})
    relation = str(summary.get("relation") or "")
    if active.rule == STRICT:
        positive = relation == "POSITIVE"
        return AdmissionVerdict(
            admitted=positive,
            rule=STRICT,
            reason="relation_positive" if positive else "relation_not_positive",
            aggregate_gain=_finite(summary.get("aggregate_gain")),
        )
    if relation == "POSITIVE":
        return AdmissionVerdict(
            admitted=True,
            rule=BOUNDED_V1,
            reason="relation_positive",
            aggregate_gain=_finite(summary.get("aggregate_gain")),
        )
    aggregate = _finite(summary.get("aggregate_gain"))
    if aggregate is None:
        return AdmissionVerdict(
            admitted=False, rule=BOUNDED_V1,
            reason="non_finite_aggregate_fail_closed")
    if aggregate < material_threshold:
        return AdmissionVerdict(
            admitted=False, rule=BOUNDED_V1,
            reason="aggregate_below_material_line", aggregate_gain=aggregate)
    series_count = _finite(summary.get("series_read"))
    harmed_count = _finite(summary.get("harmed_series_count"))
    lowest = _finite(summary.get("min_per_series_gain"))
    if not series_count or harmed_count is None or lowest is None:
        return AdmissionVerdict(
            admitted=False, rule=BOUNDED_V1,
            reason="no_per_series_reading_fail_closed", aggregate_gain=aggregate)
    return _budget_verdict(
        aggregate=aggregate,
        series_count=int(series_count),
        harmed_count=int(harmed_count),
        worst_harm=-lowest if lowest < 0.0 else 0.0,
        policy=active,
    )
