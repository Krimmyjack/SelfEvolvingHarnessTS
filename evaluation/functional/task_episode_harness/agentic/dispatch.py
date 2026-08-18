"""Runtime-owned parameter ownership at the action-unit boundary.

Frozen design §7.3: every dynamic parameter has exactly one owner.

``RUNTIME_BOUND``
    the Runtime supplies the value from a legal public Observation bound to
    *the same action unit and the same coordinate system*;
``OPERATOR_INTRINSIC``
    the Runtime hands over the action unit and the Operator localizes inside
    it, returning identity when there is no legal target.

The failure this module exists to prevent is the one already measured: a
parameter computed once on a representative series' full public prefix, then
broadcast unchanged to every (series, window) action unit -- a different
series, a different coordinate system, and a 240-point window that in
Weather's case did not even overlap the interval the number described.

§16 forbids building a second localizer outside an intrinsic Operator, so
this module deliberately contains no localizer.  It does two things only:

1. :func:`audit_program_parameter_ownership` refuses, *before* any unit runs,
   a Program step that carries a RUNTIME_BOUND parameter as a Task-level
   constant.  That is the broadcast, caught at its source.
2. :func:`resolve_action_unit_parameters` re-derives RUNTIME_BOUND parameters
   from the unit's own public observation, once per unit.

Under the current registry no operator declares a RUNTIME_BOUND parameter, so
(2) is an identity map today and (1) is what carries the invariant.  Both are
written against the registry rather than against that fact, so re-introducing
an ``external_region`` contract turns the gate back on instead of silently
restoring the broadcast.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_METADATA,
    operator_targeting_mode,
)

RUNTIME_BOUND = "RUNTIME_BOUND"
OPERATOR_INTRINSIC = "OPERATOR_INTRINSIC"


class ParameterOwnershipViolation(ValueError):
    """A Program tried to own a parameter the Runtime owns, or the reverse."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def runtime_bound_parameters(operator_id: str) -> dict[str, str]:
    """Parameter -> public feature name, for RUNTIME_BOUND parameters only."""
    metadata = OPERATOR_METADATA.get(str(operator_id)) or {}
    bindings = metadata.get("public_parameter_bindings") or {}
    return {str(name): str(path) for name, path in bindings.items()}


def parameter_owner(operator_id: str, parameter: str) -> str:
    """Which side owns one parameter of one operator.

    A parameter named in ``public_parameter_bindings`` is RUNTIME_BOUND; every
    other parameter of the operator belongs to the operator.  The two sets are
    disjoint by construction, which is what §7.3 requires.
    """
    if str(parameter) in runtime_bound_parameters(operator_id):
        return RUNTIME_BOUND
    return OPERATOR_INTRINSIC


def audit_program_parameter_ownership(
    steps: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Check a compiled Program before it reaches any action unit.

    ``strict`` raises on the first violation.  It is set False only by the
    report path, which wants to record what a historical Program did without
    aborting the audit.
    """
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for index, (operator_id, params) in enumerate(steps):
        name = str(operator_id)
        bound = runtime_bound_parameters(name)
        supplied = {str(key) for key in dict(params)}
        broadcast = sorted(supplied & set(bound))
        row = {
            "step_index": index,
            "operator": name,
            "targeting_mode": operator_targeting_mode(name),
            "runtime_bound_parameters": sorted(bound),
            "operator_intrinsic_parameters": sorted(supplied - set(bound)),
            "task_level_broadcast_parameters": broadcast,
        }
        rows.append(row)
        if broadcast:
            violation = {
                "step_index": index,
                "operator": name,
                "parameters": broadcast,
                "code": "TASK_LEVEL_BROADCAST_OF_RUNTIME_BOUND_PARAMETER",
            }
            violations.append(violation)
            if strict:
                raise ParameterOwnershipViolation(
                    violation["code"],
                    f"step {index} ({name}) carries Runtime-owned parameters "
                    f"{broadcast} as Task-level constants; they must be "
                    "re-derived per action unit or not exist",
                )
    return {
        "steps": rows,
        "violations": violations,
        "runtime_bound_parameter_count": sum(
            len(row["runtime_bound_parameters"]) for row in rows
        ),
        "ok": not violations,
    }


def resolve_action_unit_parameters(
    steps: Sequence[tuple[str, Mapping[str, Any]]],
    unit_public_features: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Bind RUNTIME_BOUND parameters from *this* unit's own observation.

    ``unit_public_features`` must be extracted from the action unit itself --
    the same series, the same window, the same coordinate system.  An operator
    whose declared feature is absent from the unit is left unbound rather than
    given a foreign value; the Operator's own fallback then applies.
    """
    resolved: list[tuple[str, dict[str, Any]]] = []
    for operator_id, params in steps:
        name = str(operator_id)
        unit_params = dict(params)
        for parameter, feature in runtime_bound_parameters(name).items():
            if feature in unit_public_features:
                unit_params[parameter] = unit_public_features[feature]
            else:
                unit_params.pop(parameter, None)
        resolved.append((name, unit_params))
    return resolved


def exploration_concentration(
    programs: Sequence[Sequence[tuple[str, Mapping[str, Any]]]],
    *,
    executable_operator_names: Sequence[str],
) -> dict[str, Any]:
    """§13/G1 readouts.  Recorded, never a Gate.

    Program combinations and operator names are counted separately and never
    share a denominator.
    """
    canonical: list[tuple[str, ...]] = []
    operators: set[str] = set()
    for steps in programs:
        signature = tuple(str(op) for op, _params in steps)
        canonical.append(signature)
        operators.update(signature)
    distinct_programs = sorted(set(canonical))
    counts = {
        signature: canonical.count(signature) for signature in distinct_programs
    }
    top1 = max(counts.values()) if counts else 0
    executable = [str(name) for name in executable_operator_names]
    return {
        "attempt_count": len(canonical),
        "distinct_canonical_program_count": len(distinct_programs),
        "distinct_operator_name_count": len(operators),
        "executable_operator_name_count": len(executable),
        "operator_name_coverage": (
            len(operators & set(executable)) / len(executable)
            if executable else None
        ),
        "top1_canonical_program_attempt_fraction": (
            top1 / len(canonical) if canonical else None
        ),
        "distinct_canonical_programs": [
            {"steps": list(signature), "attempts": counts[signature]}
            for signature in distinct_programs
        ],
        "role": "exploration concentration readout; never a Gate",
    }


__all__ = [
    "OPERATOR_INTRINSIC",
    "RUNTIME_BOUND",
    "ParameterOwnershipViolation",
    "audit_program_parameter_ownership",
    "exploration_concentration",
    "parameter_owner",
    "resolve_action_unit_parameters",
    "runtime_bound_parameters",
]
