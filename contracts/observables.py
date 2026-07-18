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
        "estimated_level_offset": "number",
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

OBSERVABLE_NUMERIC_BIN_LABELS = (
    "zero",
    "very_low",
    "low",
    "medium",
    "high",
)

_NUMERIC_BIN_EDGES = MappingProxyType(
    {
        "missing_fraction": (0.0, 0.01, 0.05, 0.20),
        "longest_missing_run_fraction": (0.0, 0.01, 0.05, 0.20),
        "estimated_region_start_fraction": (0.0, 0.01, 0.05, 0.20),
        "estimated_region_end_fraction": (0.0, 0.01, 0.05, 0.20),
        "period_change_score": (0.0, 0.10, 0.25, 0.50),
        "period_reliability": (0.0, 0.25, 0.50, 0.75),
    }
)

OBSERVABLE_STRING_DOMAINS = MappingProxyType(
    {
        "task_kind": frozenset(
            {"forecast", "classification", "anomaly_detection"}
        ),
        "period_evidence_status": frozenset({"OK", "UNKNOWN"}),
        "imputation_probe_direction": frozenset(
            {"positive", "flat", "overdose_collapse", "negative", "unknown"}
        ),
        "clipping_probe_direction": frozenset(
            {"positive", "flat", "overdose_collapse", "negative", "unknown"}
        ),
        "denoising_probe_direction": frozenset(
            {"positive", "flat", "overdose_collapse", "negative", "unknown"}
        ),
        "level_probe_direction": frozenset(
            {"positive", "flat", "overdose_collapse", "negative", "unknown"}
        ),
    }
)

_OPS_BY_TYPE = {
    "number": frozenset({">", ">=", "<", "<=", "==", "in"}),
    "boolean": frozenset({"=="}),
    "string": frozenset({"==", "in"}),
}


def observable_numeric_bin(feature: str, value: float) -> str:
    if feature not in OBSERVABLE_FEATURES or OBSERVABLE_FEATURES[feature] != "number":
        raise ValueError(f"feature is not numeric and observable: {feature!r}")
    edges = _NUMERIC_BIN_EDGES.get(feature, (0.0, 1.0, 3.0, 6.0))
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("numeric observable must be finite before binning")
    if numeric <= edges[0]:
        return OBSERVABLE_NUMERIC_BIN_LABELS[0]
    if numeric < edges[1]:
        return OBSERVABLE_NUMERIC_BIN_LABELS[1]
    if numeric < edges[2]:
        return OBSERVABLE_NUMERIC_BIN_LABELS[2]
    if numeric < edges[3]:
        return OBSERVABLE_NUMERIC_BIN_LABELS[3]
    return OBSERVABLE_NUMERIC_BIN_LABELS[4]


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
        if operator == "in":
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes, bytearray))
                or not value
                or not all(item in OBSERVABLE_NUMERIC_BIN_LABELS for item in value)
            ):
                raise ValueError(
                    "numeric-feature 'in' requires non-empty observable bin labels"
                )
        elif isinstance(value, str):
            if operator != "==" or value not in OBSERVABLE_NUMERIC_BIN_LABELS:
                raise ValueError(
                    "numeric feature accepts a bin label only with =="
                )
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("numeric value or observable bin label required")
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
        domain = OBSERVABLE_STRING_DOMAINS.get(feature)
        if domain is not None and any(item not in domain for item in value):
            raise ValueError(f"value is outside the closed domain for {feature}")
    elif not isinstance(value, str):
        raise ValueError("string value required for string feature")
    else:
        domain = OBSERVABLE_STRING_DOMAINS.get(feature)
        if domain is not None and value not in domain:
            raise ValueError(f"value is outside the closed domain for {feature}")


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
    "OBSERVABLE_NUMERIC_BIN_LABELS",
    "OBSERVABLE_STRING_DOMAINS",
    "PERIOD_RELIABILITY_MIN",
    "observable_numeric_bin",
    "validate_applicability",
]
