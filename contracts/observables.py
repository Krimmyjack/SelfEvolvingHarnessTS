from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


PERIOD_RELIABILITY_MIN = 0.75


OBSERVABLE_FEATURES = MappingProxyType(
    {
        "task_kind": "string",
        "missing_fraction": "number",
        "longest_missing_run_fraction": "number",
        "local_robust_z_peak": "number",
        "estimated_region_start_fraction": "number",
        "estimated_region_end_fraction": "number",
        "level_excursion_score": "number",
        "period_change_score": "number",
        "period_reliability": "number",
        "period_evidence_status": "string",
        "period_repair_available": "boolean",
        "imputation_probe_direction": "string",
        "clipping_probe_direction": "string",
        "denoising_probe_direction": "string",
        "level_probe_direction": "string",
    }
)

_OPS_BY_TYPE = {
    "number": frozenset({">", ">=", "<", "<=", "=="}),
    "boolean": frozenset({"=="}),
    "string": frozenset({"==", "in"}),
}


def _validate_leaf(ast: Mapping[str, Any]) -> None:
    if set(ast) != {"feature", "op", "value"}:
        raise ValueError("applicability leaf must contain feature, op, and value only")
    feature = ast["feature"]
    if not isinstance(feature, str) or feature not in OBSERVABLE_FEATURES:
        raise ValueError(f"unknown observable feature: {feature!r}")
    operator = ast["op"]
    feature_type = OBSERVABLE_FEATURES[feature]
    if not isinstance(operator, str) or operator not in _OPS_BY_TYPE[feature_type]:
        raise ValueError(f"operator {operator!r} is invalid for {feature_type} feature")
    value = ast["value"]
    if feature_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("numeric value required for number feature")
        if not math.isfinite(float(value)):
            raise ValueError("non-finite applicability value is forbidden")
    elif feature_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError("boolean value required for boolean feature")
    elif operator == "in":
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or not value
            or not all(isinstance(item, str) for item in value)
        ):
            raise ValueError("string 'in' requires a non-empty string sequence")
    elif not isinstance(value, str):
        raise ValueError("string value required for string feature")


def validate_applicability(ast: Mapping[str, Any]) -> None:
    if not isinstance(ast, Mapping):
        raise ValueError("applicability must be an object")
    keys = set(ast)
    if keys == {"const"}:
        if not isinstance(ast["const"], bool):
            raise ValueError("const applicability value must be boolean")
        return
    if keys in ({"all"}, {"any"}):
        key = next(iter(keys))
        children = ast[key]
        if (
            not isinstance(children, Sequence)
            or isinstance(children, (str, bytes, bytearray))
            or not children
        ):
            raise ValueError(f"{key} applicability node requires a non-empty sequence")
        for child in children:
            validate_applicability(child)
        return
    if keys == {"not"}:
        child = ast["not"]
        if not isinstance(child, Mapping):
            raise ValueError("not applicability node requires one object")
        validate_applicability(child)
        return
    _validate_leaf(ast)


__all__ = [
    "OBSERVABLE_FEATURES",
    "PERIOD_RELIABILITY_MIN",
    "validate_applicability",
]
