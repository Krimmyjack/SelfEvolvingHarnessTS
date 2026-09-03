"""A Scope a Skill can carry across cohorts: a predicate, never a list of UIDs.

A Skill that remembers ``{"T140", "T145"}`` is not transferable -- those UIDs do
not exist in the next Target, so the Scope silently resolves to nothing and the
Skill becomes a no-op that still occupies the store.  The Scope has to be stated
in the same vocabulary the deployment can actually observe, and resolved against
the series in front of it at run time.

    ScopeSpec(kind="serving_series_predicate",
              clauses=(Clause("missing_fraction", ">=", 0.10),
                       Clause("seasonal_strength", ">=", 0.30)))

Three kinds, and the first two are the ones a Skill may hold:

* ``all_serving_series`` -- the old global behaviour, kept so the frozen line
  stays expressible;
* ``serving_series_predicate`` -- an AND of clauses over deployment-visible
  features, which is what makes a Skill portable;
* ``none`` -- treat nothing; abstention as a first-class action, which the
  serving evaluator turns into predictions bit-identical to Static.

``resolved_training_series`` deliberately does not exist as a Skill-storable
kind.  The resolved UID set is a per-cell execution *result*: it is recorded on
the Episode as evidence of what actually ran, and it must never be promoted into
the Skill, or the Skill stops transferring.

Validation is strict on purpose.  A clause naming a feature the deployment
cannot see, or naming something UID-shaped, is refused at construction rather
than resolving to an empty set at run time where it would look like a legitimate
abstention.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

SCOPE_KINDS = ("all_serving_series", "serving_series_predicate", "none")
OPERATORS = ("<=", ">=", "<", ">")

#: Anything UID-shaped is refused: a Scope that names series cannot transfer.
_UID_SHAPED = re.compile(r"^[A-Za-z]{1,4}\d+$")


class ScopeError(ValueError):
    """A Scope that could not transfer, refused at construction."""


@dataclass(frozen=True)
class Clause:
    feature: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in OPERATORS:
            raise ScopeError("operator must be one of %s" % (OPERATORS,))
        if _UID_SHAPED.match(self.feature):
            raise ScopeError(
                "a Scope clause may not name a series (%r); Scopes are stated "
                "over deployment-visible features so they transfer" % self.feature
            )
        if not isinstance(self.threshold, (int, float)):
            raise ScopeError("threshold must be numeric")

    def holds(self, value: float) -> bool:
        if self.op == "<=":
            return value <= self.threshold
        if self.op == ">=":
            return value >= self.threshold
        if self.op == "<":
            return value < self.threshold
        return value > self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op,
                "threshold": float(self.threshold)}


@dataclass(frozen=True)
class ScopeSpec:
    kind: Literal["all_serving_series", "serving_series_predicate", "none"]
    clauses: tuple[Clause, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in SCOPE_KINDS:
            raise ScopeError("kind must be one of %s" % (SCOPE_KINDS,))
        if self.kind == "serving_series_predicate" and not self.clauses:
            raise ScopeError("a predicate Scope needs at least one clause")
        if self.kind != "serving_series_predicate" and self.clauses:
            raise ScopeError("%s takes no clauses" % self.kind)

    def validate_against(self, available: Sequence[str]) -> None:
        """Every clause must name a feature the deployment can actually read."""
        unknown = sorted(
            {clause.feature for clause in self.clauses} - set(available)
        )
        if unknown:
            raise ScopeError(
                "Scope names features the deployment cannot observe: %s"
                % unknown
            )

    def resolve(self, features: Mapping[str, Mapping[str, float]]
                ) -> frozenset[str]:
        """Which served series this Scope selects, given their own features.

        ``features`` is per-UID deployment-visible readings taken strictly
        before the origin.  No Outcome participates, by construction: this
        function is given features and nothing else.
        """
        if self.kind == "none":
            return frozenset()
        if self.kind == "all_serving_series":
            return frozenset(features)
        selected = []
        for uid, card in features.items():
            if all(
                clause.feature in card and clause.holds(float(card[clause.feature]))
                for clause in self.clauses
            ):
                selected.append(str(uid))
        return frozenset(selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_type": self.kind,
            "predicate": [clause.to_dict() for clause in self.clauses],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScopeSpec":
        clauses = tuple(
            Clause(str(item["feature"]), str(item["op"]), float(item["threshold"]))
            for item in payload.get("predicate") or ()
        )
        return cls(kind=str(payload["scope_type"]), clauses=clauses)

    def describe(self) -> str:
        if self.kind != "serving_series_predicate":
            return self.kind
        return " AND ".join(
            "%s %s %g" % (c.feature, c.op, c.threshold) for c in self.clauses
        )


ALL = ScopeSpec("all_serving_series")
NONE = ScopeSpec("none")


def execution_record(spec: ScopeSpec, resolved: frozenset[str],
                     served: Sequence[str]) -> dict[str, Any]:
    """What actually ran, for the Episode -- evidence, never a Skill field.

    The resolved UID set belongs on the Episode so a later audit can say which
    series were treated.  Promoting it into the Skill would freeze the Skill to
    one cohort, which is exactly the failure this module exists to prevent.
    """
    return {
        "scope": spec.to_dict(),
        "scope_description": spec.describe(),
        "resolved_training_series": sorted(resolved),
        "resolved_count": len(resolved),
        "served_count": len(served),
        "coverage": round(len(resolved) / len(served), 4) if served else 0.0,
        "is_skill_field": False,
        "why_not_a_skill_field": (
            "UIDs do not exist in the next Target; a Skill that stored them "
            "would resolve to nothing there and look like a legitimate abstention"
        ),
    }
