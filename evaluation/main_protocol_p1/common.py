"""Small shared contract for the v1.2.1 P1 Core baseline smoke.

The module deliberately owns only the pieces that must be identical across
Forecast, Classification, and Anomaly Detection: the 13-method roster, the
normalized method/surface/cost shape, and the release checks.  Task data,
Consumers, Harness execution, and output writing remain outside this module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL_VERSION = "v1.2.1-Core"
STAGE = "P1_COMMON_DSL_AND_CORE_BASELINE_SMOKE"
TASKS = ("forecast", "classification", "anomaly_detection")
SURFACES = ("support_a", "support_b")

B_MAIN = 4
MAX_SUPPORT_A_FULL = 3
MAX_SUPPORT_B_FULL = 1
MAX_CHEAP_PROBES = 12
MAX_LLM_CALLS = 4
MAX_TOKENS = 40_000
MAX_ACCEPTED_UPDATES = 1

MANDATORY_METHODS = (
    "Identity",
    "Best Fixed Per-task",
    "Fixed Linear-impute",
    "Fixed Hampel",
    "Fixed Winsor",
    "Fixed IQR",
    "Parallel Best-of-N@4",
    "Sequential Refinement@4",
    "Frozen H0",
    "Static",
    "A3-reset",
    "K0-fixed",
    "A5-online",
)

MANDATORY_FIXED_PROGRAMS = (
    "impute_linear",
    "hampel_filter",
    "winsorize",
    "outlier_iqr",
)

SURFACE_FIELDS = (
    "evaluation_state",
    "primary_metric_value",
    "utility",
    "delta_u_vs_identity",
    "view_values",
    "behavior_point_count",
)

EVALUATION_STATES = {
    "EVALUATED",
    "ABSTAINED",
    "SAFE_REJECT",
    "NOT_EVALUATED",
}


def _plain(value: Any) -> Any:
    """Return JSON-friendly builtins without importing a numeric package."""
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _face_map(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    return {face: _as_int(value.get(face)) for face in SURFACES}


def _infer_logical_faces(method: str, total: int) -> dict[str, int]:
    """Infer legacy Forecast face counts; new components should report them."""
    total = max(0, int(total))
    if total == 0:
        return {"support_a": 0, "support_b": 0}
    if method == "Frozen H0":
        return {"support_a": total, "support_b": 0}
    if method == "Parallel Best-of-N@4" and total == 4:
        return {"support_a": 3, "support_b": 1}
    if method == "Sequential Refinement@4" and total == 3:
        return {"support_a": 2, "support_b": 1}
    if total >= 2:
        return {"support_a": total - 1, "support_b": 1}
    return {"support_a": 1, "support_b": 0}


def _counts(
    value: Any,
    *,
    total: int,
    method: str,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    faces = _face_map(value)
    if faces is None and fallback is not None:
        faces = _face_map(fallback)
    if faces is None:
        faces = _infer_logical_faces(method, total)
    return {
        "support_a": int(faces["support_a"]),
        "support_b": int(faces["support_b"]),
        "total": int(faces["support_a"] + faces["support_b"]),
    }


def normalize_usage(
    method: str,
    usage: Mapping[str, Any] | None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize costs and recompute the matched-budget decision.

    The operating budget is expressed in logical full evaluations.  Raw model
    fits remain an accounting quantity because one AD evaluation may fit more
    than one series.
    """
    raw = dict(usage or {})
    detail = dict(details or {})

    full_value = raw.get("full_support_evaluations", 0)
    if isinstance(full_value, Mapping):
        full_total = sum(_as_int(full_value.get(face)) for face in SURFACES)
        full_faces = full_value
    else:
        full_total = _as_int(full_value)
        full_faces = None
    full = _counts(
        full_faces,
        total=full_total,
        method=method,
        fallback=(detail.get("full_support_evaluations_by_face")
                  or detail.get("logical_evaluations_by_face")),
    )

    # AD reports the same accounting unit as ``raw_series_fits`` because one
    # logical Event-F1 evaluation can fit several per-series Consumers.  Keep
    # that task-native spelling as an input alias, but expose one normalized
    # field to the master report.
    raw_value = raw.get("raw_consumer_fits", raw.get("raw_series_fits", 0))
    if isinstance(raw_value, Mapping):
        raw_total = sum(_as_int(raw_value.get(face)) for face in SURFACES)
        raw_faces = raw_value
    else:
        raw_total = _as_int(raw_value)
        raw_faces = None
    raw_fallback = (
        detail.get("raw_consumer_fits_by_face")
        or detail.get("raw_series_fits_by_face")
    )
    if raw_fallback is None and any(
        key in detail
        for key in ("support_a_raw_series_fits", "support_b_raw_series_fits")
    ):
        raw_fallback = {
            "support_a": detail.get("support_a_raw_series_fits", 0),
            "support_b": detail.get("support_b_raw_series_fits", 0),
        }
    fit_counts = _counts(
        raw_faces,
        total=raw_total,
        method=method,
        fallback=raw_fallback,
    )
    if raw_faces is None and raw_fallback is None:
        fit_counts = {
            **_infer_logical_faces(method, raw_total),
            "total": raw_total,
        }

    cache_value = raw.get("cache_hits")
    receipt = detail.get("receipt_accounting") or {}
    cache = _counts(
        cache_value,
        total=(sum(_as_int(cache_value.get(face)) for face in SURFACES)
               if isinstance(cache_value, Mapping) else _as_int(cache_value)),
        method=method,
        fallback=(receipt.get("cache_hits_by_face")
                  if isinstance(receipt, Mapping) else None),
    )

    input_tokens = _as_int(raw.get("input_tokens"))
    output_tokens = _as_int(raw.get("output_tokens"))
    reported_tokens = _as_int(raw.get("tokens"))
    tokens = input_tokens + output_tokens
    if tokens == 0 and reported_tokens:
        tokens = reported_tokens
    cheap = _as_int(raw.get("cheap_probes"))
    calls = _as_int(raw.get("llm_calls"))
    updates = _as_int(raw.get("accepted_updates"))
    nonnegative = all(value >= 0 for value in (
        full["support_a"], full["support_b"], fit_counts["support_a"],
        fit_counts["support_b"], cache["support_a"], cache["support_b"],
        cheap, calls, input_tokens, output_tokens, tokens, updates,
    ))
    within = bool(
        nonnegative
        and full["support_a"] <= MAX_SUPPORT_A_FULL
        and full["support_b"] <= MAX_SUPPORT_B_FULL
        and full["total"] <= B_MAIN
        and cheap <= MAX_CHEAP_PROBES
        and calls <= MAX_LLM_CALLS
        and tokens <= MAX_TOKENS
        and updates <= MAX_ACCEPTED_UPDATES
    )
    return {
        "full_support_evaluations": full,
        "raw_consumer_fits": fit_counts,
        "cache_hits": cache,
        "cheap_probes": cheap,
        "llm_calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": tokens,
        "accepted_updates": updates,
        "wall_seconds": _as_float_or_none(raw.get("wall_seconds")),
        "within_caps": within,
    }


def empty_surface(state: str = "NOT_EVALUATED") -> dict[str, Any]:
    if state not in EVALUATION_STATES:
        raise ValueError("unknown P1 surface evaluation state: %s" % state)
    return {
        "evaluation_state": state,
        "primary_metric_value": None,
        "utility": None,
        "delta_u_vs_identity": None,
        "view_values": [],
        "behavior_point_count": 0,
    }


def normalize_surface(
    reading: Mapping[str, Any] | None,
    *,
    state: str | None = None,
) -> dict[str, Any]:
    raw = dict(reading or {})
    primary = None
    for key in (
        "primary_metric_value", "smase", "cls_macro_f1", "macro_f1",
        "ad_macro_f1", "event_f1",
    ):
        if key in raw and raw[key] is not None:
            primary = _as_float_or_none(raw[key])
            break
    utility = _as_float_or_none(raw.get("utility"))
    if utility is None:
        utility = _as_float_or_none(raw.get("candidate_utility"))
    delta = _as_float_or_none(raw.get("delta_u_vs_identity"))
    view_values: list[Any] = []
    for key in (
        "view_values", "per_series_smase", "per_series_event_f1",
        "per_class_f1", "per_view_metric", "per_view_values",
        "per_view_smase",
    ):
        value = raw.get(key)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            view_values = list(_plain(value))
            break
    resolved_state = str(state or raw.get("evaluation_state") or "")
    if not resolved_state:
        resolved_state = (
            "EVALUATED"
            if any(value is not None for value in (primary, utility, delta))
            else "NOT_EVALUATED"
        )
    if resolved_state not in EVALUATION_STATES:
        resolved_state = "NOT_EVALUATED"
    return {
        "evaluation_state": resolved_state,
        "primary_metric_value": primary,
        "utility": utility,
        "delta_u_vs_identity": delta,
        "view_values": view_values,
        "behavior_point_count": _as_int(raw.get("behavior_point_count")),
    }


def _legacy_surfaces(
    method: str,
    selected: str,
    readings: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if isinstance(readings.get("support_a"), Mapping) or isinstance(
        readings.get("support_b"), Mapping
    ):
        return {
            face: normalize_surface(
                readings.get(face) if isinstance(readings.get(face), Mapping) else None
            )
            for face in SURFACES
        }
    target = readings.get("target")
    if isinstance(target, Mapping):
        return {
            face: normalize_surface(
                target.get(face) if isinstance(target.get(face), Mapping) else None
            )
            for face in SURFACES
        }
    candidates = readings.get("support_a_candidates")
    if isinstance(candidates, Mapping):
        support = candidates.get(selected)
        return {
            "support_a": normalize_surface(
                support if isinstance(support, Mapping) else None
            ),
            "support_b": normalize_surface(
                readings.get("support_b")
                if isinstance(readings.get("support_b"), Mapping) else None
            ),
        }
    if method == "Sequential Refinement@4":
        support = None
        for key in ("step_1", "step_2"):
            row = readings.get(key)
            if isinstance(row, Mapping) and str(row.get("program")) == selected:
                support = row
                break
        return {
            "support_a": normalize_surface(support),
            "support_b": normalize_surface(
                readings.get("support_b")
                if isinstance(readings.get("support_b"), Mapping) else None
            ),
        }
    if any(key in readings for key in (
        "support_delta_u_vs_identity", "delayed_delta_u_vs_identity", "abstained"
    )):
        abstained = bool(readings.get("abstained"))
        support_delta = readings.get("support_delta_u_vs_identity")
        delayed_delta = readings.get("delayed_delta_u_vs_identity")
        support_state = (
            "EVALUATED" if support_delta is not None
            else "ABSTAINED" if abstained else "NOT_EVALUATED"
        )
        delayed_state = (
            "EVALUATED" if delayed_delta is not None
            else "ABSTAINED" if abstained else "NOT_EVALUATED"
        )
        return {
            "support_a": normalize_surface({
                "candidate_utility": readings.get("support_candidate_utility"),
                "delta_u_vs_identity": support_delta,
            }, state=support_state),
            "support_b": normalize_surface({
                "candidate_utility": readings.get("delayed_candidate_utility"),
                "delta_u_vs_identity": delayed_delta,
            }, state=delayed_state),
        }
    return {face: empty_surface() for face in SURFACES}


def _lifecycle(method: str, details: Mapping[str, Any]) -> dict[str, Any]:
    if method == "A3-reset":
        initial, adaptation, writeback, discard = "h0", True, False, True
    elif method == "K0-fixed":
        initial, adaptation, writeback, discard = (
            "shared_k0_a5", True, False, True
        )
    elif method == "A5-online":
        initial, adaptation, writeback, discard = (
            "shared_k0_a5", True, True, False
        )
    elif method == "Frozen H0":
        initial, adaptation, writeback, discard = "h0", False, False, True
    else:
        initial, adaptation, writeback, discard = "none", False, False, True
    if "writeback_channel" in details:
        writeback = bool(details["writeback_channel"])
    if "unit_state_discarded" in details:
        discard = bool(details["unit_state_discarded"])
    return {
        "initial_state": str(details.get("initial_state") or initial),
        "target_adaptation": bool(
            details.get("target_adaptation", adaptation)
        ),
        "writeback_allowed": writeback,
        "unit_state_discarded": discard,
        "state_retained": bool(details.get("retained_update", False)),
        "support_b_promotion_required": method in {
            "A3-reset", "K0-fixed", "A5-online"
        },
    }


def normalize_method_row(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(row)
    method = str(raw.get("method") or raw.get("method_id") or "")
    selected = str(raw.get("selected_program") or "identity")
    details = dict(raw.get("details") or {})
    readings = dict(raw.get("readings") or {})
    surfaces_raw = raw.get("surfaces")
    if isinstance(surfaces_raw, Mapping):
        surfaces = {
            face: normalize_surface(
                surfaces_raw.get(face)
                if isinstance(surfaces_raw.get(face), Mapping) else None
            )
            for face in SURFACES
        }
    else:
        surfaces = _legacy_surfaces(method, selected, readings)
    errors = raw.get("protocol_errors")
    if errors is None:
        errors = details.get("protocol_errors")
    protocol_errors = [str(value) for value in (errors or [])]
    contract_status = str(raw.get("contract_status") or raw.get("status") or "")
    behavior = str(raw.get("behavior_status") or "")
    if not behavior:
        if contract_status != "PASS":
            behavior = "BLOCKED"
        elif any(
            surface["evaluation_state"] == "ABSTAINED"
            for surface in surfaces.values()
        ):
            behavior = "ABSTAINED"
        elif any(
            surface["evaluation_state"] == "SAFE_REJECT"
            for surface in surfaces.values()
        ):
            behavior = "SAFE_REJECT"
        elif any(
            surface["evaluation_state"] == "EVALUATED"
            for surface in surfaces.values()
        ):
            behavior = "EVALUATED"
        else:
            behavior = "NOT_EVALUATED"
    return {
        "method": method,
        "contract_status": contract_status,
        "behavior_status": behavior,
        "selected_program": selected,
        "implementation": str(raw.get("implementation") or ""),
        "surfaces": surfaces,
        "usage": normalize_usage(method, raw.get("usage"), details),
        "lifecycle": _lifecycle(method, details),
        "protocol_errors": protocol_errors,
        "details": _plain(details),
    }


def _safe_data_summary(component: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(component.get("data") or component.get("data_boundary") or {})
    allowed = {
        "dataset", "datasets", "data_role", "fixture", "fixture_id",
        "target_fixture", "evolution_fixture", "roster_path", "cache_path",
        "support_a_series", "support_b_series",
        "best_fixed_selection_support_a_series",
        "best_fixed_selection_support_b_series",
        "best_fixed_selection_disjoint_from_target", "selection_rule",
        "selection_uses_support_or_future_utility",
        "structurally_readable_pool_count", "structurally_excluded_series_count",
        "development_query_evaluations", "natural_final_outcome_reads",
        "test_member_bytes_read", "held_out_requests",
        "traffic_or_solar_loader_available",
    }
    summary = {key: _plain(data[key]) for key in allowed if key in data}
    split = component.get("split")
    if isinstance(split, Mapping):
        split_allowed = {
            "origin", "horizon", "period", "support_a_count",
            "support_b_count", "training_series_per_face", "query_count",
            "fit_count", "target_count", "selection_count",
        }
        summary["split"] = {
            key: _plain(split[key]) for key in split_allowed if key in split
        }
    return summary


def _safe_consumer_summary(component: Mapping[str, Any]) -> dict[str, Any]:
    consumer = dict(component.get("consumer") or {})
    allowed = {
        "id", "implementation", "primary_metric", "metric_direction",
        "utility_definition", "delta_definition", "secondary_metrics",
    }
    return {key: _plain(consumer[key]) for key in allowed if key in consumer}


def _safe_backend_summary(component: Mapping[str, Any]) -> dict[str, Any]:
    backend = dict(component.get("backend") or {})
    allowed = {
        "mode", "production_format_exercised", "live_transport_exercised",
        "production_lifecycle_exercised", "production_ttha_method_exercised",
        "production_run_online_round_exercised",
        "shared_lifecycle_compatibility_basis",
        "global_llm_calls", "global_input_tokens", "global_output_tokens",
        "k0_a5_same_initial_state", "k0_a5_initial_skill_ids",
        "temporary_store_removed_after_run", "history_contract",
        "historical_input_status", "withheld_history",
        "k0_a5_forecast_supply_contract",
    }
    summary = {key: _plain(backend[key]) for key in allowed if key in backend}

    # The already-completed Forecast report predates the three explicit
    # lifecycle booleans.  Reuse it without rerunning Forecast only when all
    # three Harness arms identify the exact production path in their saved
    # method rows.  New components must report the booleans directly.
    task = str(component.get("task_tranche") or component.get("task") or "")
    lifecycle_fields = (
        "production_lifecycle_exercised",
        "production_ttha_method_exercised",
        "production_run_online_round_exercised",
    )
    if task == "forecast" and not any(key in backend for key in lifecycle_fields):
        implementations = {
            str(row.get("method") or row.get("method_id") or ""): str(
                row.get("implementation") or ""
            )
            for row in (component.get("methods") or [])
        }
        legacy_path = all(
            "production TTHAMethod + run_online_round + Support-B wall"
            in implementations.get(method, "")
            for method in ("A3-reset", "K0-fixed", "A5-online")
        )
        summary.update({key: legacy_path for key in lifecycle_fields})
    return summary


def normalize_component(component: Mapping[str, Any]) -> dict[str, Any]:
    task = str(component.get("task_tranche") or component.get("task") or "")
    pass_key = "%s_component_pass" % task
    reported = component.get(pass_key, component.get("component_pass"))
    if reported is None:
        reported = component.get("reported_component_pass")
    if reported is None and task == "anomaly_detection":
        reported = component.get(
            "ad_component_pass", component.get("anomaly_component_pass")
        )
    claims = component.get("claims") if isinstance(component.get("claims"), Mapping) else {}
    common_contract = _plain(component.get("common_dsl_contract") or {})
    return {
        "protocol_version": str(component.get("protocol_version") or ""),
        "stage": str(component.get("stage") or STAGE),
        "task": task,
        "evidence_grade": str(component.get("evidence_grade") or "INFRASTRUCTURE"),
        "reported_component_pass": reported,
        "data_boundary": _safe_data_summary(component),
        "consumer": _safe_consumer_summary(component),
        "common_dsl_contract": common_contract,
        "methods": [normalize_method_row(row) for row in (component.get("methods") or [])],
        "backend": _safe_backend_summary(component),
        "protocol_errors": _plain(component.get("protocol_errors") or {}),
        "blocking_failures": [
            str(value) for value in (component.get("blocking_failures") or [])
        ],
        "performance_or_headroom_claim": bool(
            component.get("performance_or_headroom_claim", claims.get("performance", False))
        ),
        "treatment_or_capability_claim": bool(
            component.get("treatment_or_capability_claim", claims.get("treatment", False))
        ),
    }


def _method_map(component: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    rows = list(component.get("methods") or [])
    by_name = {str(row.get("method")): row for row in rows}
    failures = []
    if len(rows) != len(by_name):
        failures.append("duplicate method rows")
    expected, observed = set(MANDATORY_METHODS), set(by_name)
    if observed != expected:
        failures.append(
            "method roster mismatch: missing=%s extra=%s"
            % (sorted(expected - observed), sorted(observed - expected))
        )
    if len(rows) != len(MANDATORY_METHODS):
        failures.append("method row count is %d, expected 13" % len(rows))
    return by_name, failures


def _validate_usage(method: str, usage: Mapping[str, Any]) -> list[str]:
    failures = []
    full = usage.get("full_support_evaluations") or {}
    fits = usage.get("raw_consumer_fits") or {}
    cache = usage.get("cache_hits") or {}
    for label, counts in (("logical", full), ("raw fits", fits), ("cache", cache)):
        if not isinstance(counts, Mapping):
            failures.append("%s %s counts are not face-partitioned" % (method, label))
            continue
        expected = _as_int(counts.get("support_a")) + _as_int(counts.get("support_b"))
        if _as_int(counts.get("total")) != expected:
            failures.append("%s %s total does not equal face sum" % (method, label))
        if any(_as_int(counts.get(face)) < 0 for face in (*SURFACES, "total")):
            failures.append("%s %s contains a negative count" % (method, label))
    if _as_int(full.get("support_a")) > MAX_SUPPORT_A_FULL:
        failures.append("%s exceeded the Support-A full-evaluation cap" % method)
    if _as_int(full.get("support_b")) > MAX_SUPPORT_B_FULL:
        failures.append("%s exceeded the Support-B full-evaluation cap" % method)
    if _as_int(full.get("total")) > B_MAIN:
        failures.append("%s exceeded B=4" % method)
    scalar_caps = (
        ("cheap_probes", MAX_CHEAP_PROBES),
        ("llm_calls", MAX_LLM_CALLS),
        ("tokens", MAX_TOKENS),
        ("accepted_updates", MAX_ACCEPTED_UPDATES),
    )
    for key, cap in scalar_caps:
        value = _as_int(usage.get(key))
        if value < 0:
            failures.append("%s %s is negative" % (method, key))
        if value > cap:
            failures.append("%s exceeded the %s cap" % (method, key))
    if not bool(usage.get("within_caps")):
        failures.append("%s exceeded the matched budget" % method)
    return failures


def _validate_best_fixed(by_name: Mapping[str, Any]) -> list[str]:
    row = by_name.get("Best Fixed Per-task") or {}
    details = row.get("details") or {}
    failures = []
    checks = {
        "formal_evolution_winner_frozen": True,
        "selection_uses_target_support": False,
        "selection_disjoint_from_target": True,
        "program_space_coverage_complete": True,
    }
    for key, expected in checks.items():
        if details.get(key) is not expected:
            failures.append("Best Fixed %s is not %s" % (key, expected))
    phase = ((details.get("cost_by_phase") or {}).get("evolution_selection") or {})
    if phase.get("charged_to_target_b4") is not False:
        failures.append("Best Fixed offline selection was not separated from target B=4")
    return failures


def _validate_harness_arms(
    component: Mapping[str, Any], by_name: Mapping[str, Any]
) -> list[str]:
    failures = []
    backend = component.get("backend") or {}
    if backend.get("k0_a5_same_initial_state") is not True:
        failures.append("K0 and A5 do not report the same initial state")
    k0 = by_name.get("K0-fixed") or {}
    a5 = by_name.get("A5-online") or {}
    k0_ids = (k0.get("details") or {}).get("initial_skill_ids")
    a5_ids = (a5.get("details") or {}).get("initial_skill_ids")
    if k0_ids is None or a5_ids is None or list(k0_ids) != list(a5_ids):
        failures.append("K0 and A5 initial Skill lists differ or are absent")
    backend_ids = backend.get("k0_a5_initial_skill_ids")
    if (
        backend_ids is None
        or k0_ids is None
        or list(backend_ids) != list(k0_ids)
    ):
        failures.append("K0/A5 initial Skill list disagrees with the backend")
    a3_ids = ((by_name.get("A3-reset") or {}).get("details") or {}).get(
        "initial_skill_ids"
    )
    if a3_ids is None:
        failures.append("A3 did not report its H0 Skill list")

    expected = {
        "A3-reset": (False, True),
        "K0-fixed": (False, True),
        "A5-online": (True, False),
    }
    for method, (writeback, discarded) in expected.items():
        row = by_name.get(method) or {}
        details = row.get("details") or {}
        lifecycle = row.get("lifecycle") or {}
        if details.get("writeback_channel") is not writeback:
            failures.append("%s did not explicitly report its writeback channel" % method)
        if details.get("unit_state_discarded") is not discarded:
            failures.append("%s did not explicitly report its unit-state policy" % method)
        if lifecycle.get("writeback_allowed") is not writeback:
            failures.append("%s writeback semantics are incorrect" % method)
        if lifecycle.get("unit_state_discarded") is not discarded:
            failures.append("%s state-retention semantics are incorrect" % method)
    a5_detail = a5.get("details") or {}
    if bool(a5_detail.get("retained_update")) and a5_detail.get(
        "approved_after_support_b"
    ) is not True:
        failures.append("A5 retained an update without Support-B approval")
    if a5_detail.get("writeback_persisted_to_evolution_store") is not False:
        failures.append("P1 A5 state was not isolated from the Evolution store")
    return failures


def _zero_protocol_boundary(component: Mapping[str, Any]) -> list[str]:
    errors = component.get("protocol_errors") or {}
    failures = []
    required = ("natural_final_outcome_reads", "development_query_evaluations")
    for key in required:
        if key not in errors:
            failures.append("missing protocol boundary counter: %s" % key)
        elif _as_int(errors.get(key)) != 0:
            failures.append("protocol boundary counter is nonzero: %s" % key)
    for key, value in errors.items():
        if isinstance(value, bool):
            nonzero = bool(value)
        elif isinstance(value, (int, float)):
            nonzero = value != 0
        else:
            nonzero = bool(value)
        if nonzero:
            failures.append("protocol error is nonzero: %s" % key)
    data = component.get("data_boundary") or {}
    for key in (
        "natural_final_outcome_reads", "development_query_evaluations",
        "test_member_bytes_read", "held_out_requests",
    ):
        if key in data and _as_int(data.get(key)) != 0:
            failures.append("data boundary is nonzero: %s" % key)
    split = data.get("split") if isinstance(data.get("split"), Mapping) else {}
    if "query_count" in split and _as_int(split.get("query_count")) != 0:
        failures.append("data boundary is nonzero: query_count")
    return failures


def validate_component(component: Mapping[str, Any]) -> list[str]:
    """Recompute one component gate from normalized facts."""
    failures: list[str] = []
    task = str(component.get("task") or "")
    if task not in TASKS:
        failures.append("unknown task component: %s" % task)
    if component.get("protocol_version") != PROTOCOL_VERSION:
        failures.append("component protocol version mismatch")
    if component.get("stage") != STAGE:
        failures.append("component stage mismatch")
    if component.get("reported_component_pass") is not True:
        failures.append("component did not explicitly report a pass")
    if component.get("blocking_failures"):
        failures.append("component reports blocking failures")
    if component.get("performance_or_headroom_claim"):
        failures.append("P1 component makes a performance or headroom claim")
    if component.get("treatment_or_capability_claim"):
        failures.append("P1 component makes a treatment or capability claim")

    contract = component.get("common_dsl_contract") or {}
    if contract.get("status") != "PASS":
        failures.append("Common DSL contract did not pass")
    if contract.get("identity_available") is not True:
        failures.append("Common DSL Identity is not available")
    if _as_int(contract.get("consumer_evaluations")) != 0:
        failures.append("Common DSL contract consumed a Consumer evaluation")
    overhead = contract.get("contract_overhead") or {}
    if overhead.get("charged_to_method_cell_b4") is not False:
        failures.append("Common DSL overhead was not separated from method B=4")
    if contract.get("compile_failures"):
        failures.append("Common DSL reports compile failures")
    if contract.get("mandatory_fixed_programs_not_executable"):
        failures.append("Common DSL cannot execute every fixed Core program")
    contract_rows = list(contract.get("rows") or [])
    contract_by_program = {
        str(row.get("program") or ""): row
        for row in contract_rows if isinstance(row, Mapping)
    }
    if not contract_rows or len(contract_rows) != len(contract_by_program):
        failures.append("Common DSL program rows are empty or duplicated")
    for program in MANDATORY_FIXED_PROGRAMS:
        row = contract_by_program.get(program) or {}
        if row.get("compile") != "PASS" or row.get("verifier") != "PASS":
            failures.append(
                "Common DSL fixed program is not executable: %s" % program
            )

    by_name, roster_failures = _method_map(component)
    failures.extend(roster_failures)
    for method in MANDATORY_METHODS:
        row = by_name.get(method)
        if row is None:
            continue
        if row.get("contract_status") != "PASS":
            failures.append("%s contract did not pass" % method)
        if row.get("protocol_errors"):
            failures.append("%s reports protocol errors" % method)
        surfaces = row.get("surfaces") or {}
        if set(surfaces) != set(SURFACES):
            failures.append("%s does not expose both normalized surfaces" % method)
        for face in SURFACES:
            surface = surfaces.get(face) or {}
            if set(surface) != set(SURFACE_FIELDS):
                failures.append("%s/%s surface schema mismatch" % (method, face))
            if surface.get("evaluation_state") not in EVALUATION_STATES:
                failures.append("%s/%s has an invalid evaluation state" % (method, face))
        failures.extend(_validate_usage(method, row.get("usage") or {}))
    failures.extend(_validate_best_fixed(by_name))
    failures.extend(_validate_harness_arms(component, by_name))
    failures.extend(_zero_protocol_boundary(component))
    if (component.get("backend") or {}).get("production_format_exercised") is not True:
        failures.append("production Harness format was not exercised")
    backend = component.get("backend") or {}
    lifecycle_checks = {
        "production_lifecycle_exercised": "production Harness lifecycle",
        "production_ttha_method_exercised": "production TTHAMethod",
        "production_run_online_round_exercised": "production online round",
    }
    for key, label in lifecycle_checks.items():
        if backend.get(key) is not True:
            failures.append("%s was not exercised" % label)
    return list(dict.fromkeys(failures))


def validate_p0_release(p0_report: Mapping[str, Any]) -> list[str]:
    verdict = p0_report.get("verdict") or {}
    failures = []
    if verdict.get("audit") != "P0B_COMPLETE":
        failures.append("P0b audit is not complete")
    if verdict.get("execution") != "P0B_PASS__P1_BASELINE_SMOKE_RELEASED":
        failures.append("P0b execution verdict did not release P1")
    if verdict.get("p1_release") is not True:
        failures.append("P0b p1_release is not true")
    return failures


def validate_aegis(aegis: Mapping[str, Any]) -> list[str]:
    failures = []
    if aegis.get("status") != "STRUCTURALLY_INCOMPATIBLE":
        failures.append("AegisTS bounded spike lacks the accepted incompatibility verdict")
    if aegis.get("blocking") is not False:
        failures.append("AegisTS incompatibility is incorrectly blocking")
    return failures


def validate_master(
    *,
    components: Mapping[str, Mapping[str, Any]],
    p0_report: Mapping[str, Any],
    aegis: Mapping[str, Any],
) -> list[str]:
    """Recompute the complete P1 gate; component self-verdicts are insufficient."""
    failures = validate_p0_release(p0_report)
    observed, expected = set(components), set(TASKS)
    if observed != expected:
        failures.append(
            "task component set mismatch: missing=%s extra=%s"
            % (sorted(expected - observed), sorted(observed - expected))
        )
    for task in TASKS:
        component = components.get(task)
        if component is None:
            continue
        if component.get("task") != task:
            failures.append(
                "%s: component task identity is %r" % (task, component.get("task"))
            )
        failures.extend(
            "%s: %s" % (task, failure)
            for failure in validate_component(component)
        )
    failures.extend("AegisTS: %s" % failure for failure in validate_aegis(aegis))
    return list(dict.fromkeys(failures))


__all__ = [
    "B_MAIN",
    "MANDATORY_FIXED_PROGRAMS",
    "MANDATORY_METHODS",
    "MAX_ACCEPTED_UPDATES",
    "MAX_CHEAP_PROBES",
    "MAX_LLM_CALLS",
    "MAX_SUPPORT_A_FULL",
    "MAX_SUPPORT_B_FULL",
    "MAX_TOKENS",
    "PROTOCOL_VERSION",
    "STAGE",
    "SURFACES",
    "TASKS",
    "empty_surface",
    "normalize_component",
    "normalize_method_row",
    "normalize_surface",
    "normalize_usage",
    "validate_aegis",
    "validate_component",
    "validate_master",
    "validate_p0_release",
]
