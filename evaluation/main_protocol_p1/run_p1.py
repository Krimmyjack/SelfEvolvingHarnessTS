"""Integrate the three v1.2.1 P1 Core baseline-smoke components.

Forecast is reused from its already-completed machine report.  This module
does not call the Forecast runner.  Classification and Anomaly Detection are
pure in-memory component calls; only this entry point writes the combined P1
report.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p1.common import (
    B_MAIN,
    MANDATORY_METHODS,
    MAX_ACCEPTED_UPDATES,
    MAX_CHEAP_PROBES,
    MAX_LLM_CALLS,
    MAX_SUPPORT_A_FULL,
    MAX_SUPPORT_B_FULL,
    MAX_TOKENS,
    PROTOCOL_VERSION,
    STAGE,
    TASKS,
    normalize_component,
    validate_component,
    validate_master,
    validate_p0_release,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
P0_REPORT = PROJECT_ROOT / "artifacts/main_protocol/p0_readiness_20260830.json"
FORECAST_REPORT = (
    PROJECT_ROOT / "artifacts/main_protocol/forecast_p1_core_smoke_20260830.json"
)
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/p1_core_baseline_smoke_20260830.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/p1_core_baseline_smoke_20260830.md"


class P1IntegrationBlocked(RuntimeError):
    """A component or release precondition is not readable."""


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P1IntegrationBlocked("expected a JSON object: %s" % path)
    return payload


def _load_forecast() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the existing Forecast component without executing it."""
    raw = _read_object(FORECAST_REPORT)
    normalized = normalize_component(raw)
    if normalized.get("task") != "forecast":
        raise P1IntegrationBlocked("Forecast component has the wrong task tranche")
    failures = validate_component(normalized)
    if failures:
        raise P1IntegrationBlocked(
            "existing Forecast component is not reusable: %s" % "; ".join(failures)
        )
    return normalized, raw


def _safe_aegis(raw_forecast: Mapping[str, Any]) -> dict[str, Any]:
    source = raw_forecast.get("aegists_adapter")
    if not isinstance(source, Mapping):
        source = raw_forecast.get("aegis_adapter")
    if not isinstance(source, Mapping):
        return {
            "status": "NOT_REACHED",
            "blocking": True,
            "reason": "Forecast component has no bounded AegisTS spike result",
        }
    allowed = {
        "status", "tier", "reason", "missing_files", "checks",
        "source_tree_read_only", "blocking",
    }
    return {key: source[key] for key in allowed if key in source}


def _blocked_component(task: str, reason: str) -> dict[str, Any]:
    raw = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE,
        "task_tranche": task,
        "%s_component_pass" % task: False,
        "evidence_grade": "INFRASTRUCTURE",
        "data": {
            "natural_final_outcome_reads": 0,
            "development_query_evaluations": 0,
        },
        "common_dsl_contract": {
            "status": "NOT_COMPLETED",
            "consumer_evaluations": 0,
            "contract_overhead": {"charged_to_method_cell_b4": False},
        },
        "methods": [],
        "backend": {"production_format_exercised": False},
        "protocol_errors": {
            "natural_final_outcome_reads": 0,
            "development_query_evaluations": 0,
        },
        "blocking_failures": [reason],
        "performance_or_headroom_claim": False,
        "treatment_or_capability_claim": False,
    }
    return normalize_component(raw)


def _run_classification(backend_mode: str) -> dict[str, Any]:
    from evaluation.main_protocol_p1 import classification_component

    return normalize_component(
        classification_component.run(backend_mode=backend_mode)
    )


def _run_anomaly(backend_mode: str) -> dict[str, Any]:
    from evaluation.main_protocol_p1 import anomaly_component

    return normalize_component(anomaly_component.run(backend_mode=backend_mode))


def _p0_summary(p0_report: Mapping[str, Any]) -> dict[str, Any]:
    verdict = p0_report.get("verdict") or {}
    return {
        "source": P0_REPORT.relative_to(PROJECT_ROOT).as_posix(),
        "audit": verdict.get("audit"),
        "execution": verdict.get("execution"),
        "p1_release": verdict.get("p1_release"),
        "live_outcome_release": verdict.get("live_outcome_release"),
    }


def build_report(*, backend_mode: str = "scripted") -> dict[str, Any]:
    """Build the master result while preserving each component's isolation."""
    try:
        p0_report = _read_object(P0_REPORT)
        p0_error = None
    except Exception as exc:  # noqa: BLE001 - converted to one bounded gate result
        p0_report = {}
        p0_error = "%s: %s" % (type(exc).__name__, exc)
    p0_gate_failures = (
        ["P0b report unreadable: %s" % p0_error]
        if p0_error is not None
        else validate_p0_release(p0_report)
    )

    components: dict[str, dict[str, Any]] = {}
    raw_forecast: dict[str, Any] = {}
    try:
        components["forecast"], raw_forecast = _load_forecast()
    except Exception as exc:  # noqa: BLE001 - preserve other component diagnostics
        components["forecast"] = _blocked_component(
            "forecast", "%s: %s" % (type(exc).__name__, exc)
        )

    for task, execute in (
        ("classification", _run_classification),
        ("anomaly_detection", _run_anomaly),
    ):
        if p0_gate_failures:
            components[task] = _blocked_component(
                task, "P0b did not release P1 component execution"
            )
            continue
        try:
            component = execute(backend_mode)
            if component.get("task") != task:
                raise P1IntegrationBlocked(
                    "%s component returned task=%r" % (task, component.get("task"))
                )
            components[task] = component
        except Exception as exc:  # noqa: BLE001 - one master blocked report
            components[task] = _blocked_component(
                task, "%s: %s" % (type(exc).__name__, exc)
            )

    aegis = _safe_aegis(raw_forecast)
    failures = validate_master(
        components=components,
        p0_report=p0_report,
        aegis=aegis,
    )
    failures = [*p0_gate_failures, *failures]
    failures = list(dict.fromkeys(failures))
    task_failures = {
        task: validate_component(components[task]) for task in TASKS
    }
    component_pass = {
        task: not task_failures[task] for task in TASKS
    }
    overall = not failures
    pending = [task for task in TASKS if not component_pass[task]]
    final_reads = sum(
        int((components[task].get("protocol_errors") or {}).get(
            "natural_final_outcome_reads", 0
        ) or 0)
        for task in TASKS
    )
    query_evaluations = sum(
        int((components[task].get("protocol_errors") or {}).get(
            "development_query_evaluations", 0
        ) or 0)
        for task in TASKS
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE,
        "evidence_grade": "INFRASTRUCTURE",
        "verdict": (
            "P1_CORE_BASELINE_SMOKE_PASS__P2_FORECAST_PILOT_RELEASED"
            if overall else "P1_CORE_BASELINE_SMOKE_BLOCKED"
        ),
        "overall_p1_complete": overall,
        "release_p2": overall,
        "p2_complete": False,
        "live_outcome_release": False,
        "pending_p1_task_tranches": pending,
        "p0b_release": _p0_summary(p0_report),
        "component_sources": {
            "forecast": {
                "mode": "REUSED_EXISTING_COMPONENT",
                "path": FORECAST_REPORT.relative_to(PROJECT_ROOT).as_posix(),
                "executed_by_master": False,
            },
            "classification": {"mode": "IN_MEMORY_COMPONENT"},
            "anomaly_detection": {"mode": "IN_MEMORY_COMPONENT"},
        },
        "component_pass": component_pass,
        "component_failures": task_failures,
        "method_roster": list(MANDATORY_METHODS),
        "budget_caps": {
            "full_support_evaluations": B_MAIN,
            "support_a_full_evaluations": MAX_SUPPORT_A_FULL,
            "support_b_full_evaluations": MAX_SUPPORT_B_FULL,
            "cheap_probes": MAX_CHEAP_PROBES,
            "llm_calls": MAX_LLM_CALLS,
            "tokens": MAX_TOKENS,
            "accepted_updates": MAX_ACCEPTED_UPDATES,
            "raw_consumer_fits": "REPORTED_SEPARATELY_NOT_A_B4_GATE",
        },
        "components": components,
        "aegis_adapter": aegis,
        "ad_method_gate": {
            "status": "NOT_RELEASED_BY_P1",
            "current_first_fault": (
                "#44a-r2 PROGRAM_CONSUMER_LAYER_FAULT_CONFIRMED / "
                "INVERTED_EFFECT_OBSERVED"
            ),
            "p1_scope": "INFRASTRUCTURE_ONLY",
            "ad_evolution_release": False,
        },
        "blocking_failures": failures,
        "performance_or_headroom_claim": False,
        "treatment_or_capability_claim": False,
        "natural_final_outcome_reads": final_reads,
        "development_query_evaluations": query_evaluations,
        "release_scope": (
            "P2 Forecast single-flow pilot only; Natural Final remains sealed"
            if overall else "no next-stage release"
        ),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# P1 Core baseline smoke",
        "",
        "**Verdict: `%s`. Overall P1 complete: `%s`. P2 release: `%s`.**"
        % (
            payload["verdict"], payload["overall_p1_complete"],
            payload["release_p2"],
        ),
        "",
        "This is an infrastructure/contract smoke. It makes no performance, "
        "headroom, treatment, or capability claim.",
        "",
        "Natural Final outcome reads: **%s**. Development Query evaluations: **%s**."
        % (payload["natural_final_outcome_reads"],
           payload["development_query_evaluations"]),
        "",
        "## Component gates",
        "",
        "| task | component pass | Common DSL | methods |",
        "|---|---|---|---:|",
    ]
    for task in TASKS:
        component = payload["components"][task]
        lines.append(
            "| %s | `%s` | `%s` | %d |"
            % (
                task,
                payload["component_pass"][task],
                (component.get("common_dsl_contract") or {}).get("status"),
                len(component.get("methods") or []),
            )
        )
    lines.extend([
        "",
        "## Unified Core rows",
        "",
        "| task | method | contract | behavior | selected | logical fits | raw fits |",
        "|---|---|---|---|---|---:|---:|",
    ])
    for task in TASKS:
        for row in payload["components"][task].get("methods") or []:
            usage = row.get("usage") or {}
            logical = usage.get("full_support_evaluations") or {}
            raw_fits = usage.get("raw_consumer_fits") or {}
            lines.append(
                "| %s | %s | `%s` | `%s` | `%s` | %s | %s |"
                % (
                    task, row.get("method"), row.get("contract_status"),
                    row.get("behavior_status"), row.get("selected_program"),
                    logical.get("total", 0), raw_fits.get("total", 0),
                )
            )
    lines.extend([
        "",
        "## Boundary and release",
        "",
        "- Forecast component execution by this master: `False`.",
        "- AegisTS-style bounded spike: `%s` (blocking: `%s`)."
        % (payload["aegis_adapter"].get("status"),
           payload["aegis_adapter"].get("blocking")),
        "- AD method gate: `%s`; current first fault: `%s`."
        % (payload["ad_method_gate"]["status"],
           payload["ad_method_gate"]["current_first_fault"]),
        "- P2 release authorizes only the Forecast single-flow pilot; it does "
        "not authorize AD Evolution or Natural Final.",
        "- P2 complete: `False`.",
        "- Live/Natural-Final outcome release: `False`.",
    ])
    if payload.get("blocking_failures"):
        lines.extend(["", "## Blocking failures", ""])
        lines.extend("- %s" % failure for failure in payload["blocking_failures"])
    lines.extend([
        "",
        "Machine-readable detail: "
        "`artifacts/main_protocol/p1_core_baseline_smoke_20260830.json`.",
        "",
    ])
    return "\n".join(lines)


def run(*, backend_mode: str = "scripted") -> dict[str, Any]:
    payload = build_report(backend_mode=backend_mode)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("scripted", "live"), default="scripted",
        help="scripted is the reproducible P1 gate; live is an optional diagnostic",
    )
    parser.add_argument(
        "--expect-pass", action="store_true",
        help="exit non-zero unless the complete three-task P1 gate passes",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run(backend_mode=args.backend)
    print(json.dumps({
        "verdict": payload["verdict"],
        "overall_p1_complete": payload["overall_p1_complete"],
        "release_p2": payload["release_p2"],
        "p2_complete": payload["p2_complete"],
        "blocking_failures": payload["blocking_failures"],
        "output_json": OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "output_md": OUT_MD.relative_to(PROJECT_ROOT).as_posix(),
    }, indent=2, ensure_ascii=False), flush=True)
    return int(bool(args.expect_pass and not payload["overall_p1_complete"]))


if __name__ == "__main__":
    raise SystemExit(main())
