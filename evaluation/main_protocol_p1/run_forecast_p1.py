"""Run the Forecast tranche of the v1.2.1 P1 Core baseline smoke.

This runner is deliberately narrow:

* it can load only the already-exposed KDD 2018 development cache;
* it exposes no Traffic, Solar, Query, or Natural-Final loader;
* it executes the existing Typed Workflow runtime and pooled Ridge Consumer;
* it checks all 13 Forecast Core method contracts without making a performance
  or headroom claim; and
* it uses isolated temporary Harness stores, so no P1 smoke state can enter the
  later Evolution store.

Forecast is one tranche of protocol-stage P1.  Passing this runner does not
complete the three-task P1 gate and therefore does not release P2 by itself.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
from SelfEvolvingHarnessTS.contracts.task import (
    MetricSpec,
    deployment_constraints_v1,
    forecast_task_context_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.baselines import (
    ProgramLoss,
    select_best_fixed,
)
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    activate_approved,
    open_delayed,
    run_online_round,
    source_skill_of_candidate,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms
from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.functional import run_v1_kdd2018_natural_slow_update as kdd
from evaluation.functional import run_v1_signed_agent_action_wiring as wiring
from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
P0_REPORT = PROJECT_ROOT / "artifacts/main_protocol/p0_readiness_20260830.json"
KDD_ROSTER = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_frozen_cohort.jsonl"
KDD_CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
FORECAST_SUPPLY_ARTIFACT = (
    PROJECT_ROOT / "artifacts/functional/e2/s2a_g1_run1_r2.json"
)
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/forecast_p1_core_smoke_20260830.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/forecast_p1_core_smoke_20260830.md"

PROTOCOL_VERSION = "v1.2.1-Core"
EVIDENCE_GRADE = "INFRASTRUCTURE"
TASK = "forecast"
CONSUMER_ID = "pooled_ridge_a1"
PRIMARY_METRIC = "sMASE"
# This is the established KDD development origin used by the existing
# exposed-data runner.  Later Traffic origins are intentionally unavailable
# here because Traffic contains the frozen Natural-Final column range.
ORIGIN = 600
PERIOD = 24
HORIZON = 48
B_MAIN = 4
MAX_PROBES = 12
MAX_LLM_CALLS = 4
MAX_TOKENS = 40_000
MAX_UPDATES = 1
MAX_MODIFIED_FRACTION = 0.35
LIVE_GLOBAL_CALL_CAP = 16
EFFECT_ALIASES = {"resample_uniform": "identity"}

FIXED_PROGRAMS = (
    ("Fixed Linear-impute", "impute_linear"),
    ("Fixed Hampel", "hampel_filter"),
    ("Fixed Winsor", "winsorize"),
    ("Fixed IQR", "outlier_iqr"),
)
MANDATORY_METHODS = (
    "Identity",
    "Best Fixed Per-task",
    *(name for name, _program in FIXED_PROGRAMS),
    "Parallel Best-of-N@4",
    "Sequential Refinement@4",
    "Frozen H0",
    "Static",
    "A3-reset",
    "K0-fixed",
    "A5-online",
)


class P1Blocked(RuntimeError):
    """A protocol or infrastructure condition prevented a truthful P1 pass."""


class FitBudget:
    def __init__(self, cap: int = B_MAIN) -> None:
        self.cap = int(cap)
        self.used = 0

    def spend(self, count: int = 1) -> None:
        count = int(count)
        if self.used + count > self.cap:
            raise P1Blocked(
                "raw Consumer-fit cap exceeded: %d + %d > %d"
                % (self.used, count, self.cap)
            )
        self.used += count


@dataclass(frozen=True)
class ForecastCell:
    values: Mapping[str, np.ndarray]
    support_a: tuple[str, ...]
    support_b: tuple[str, ...]
    observation_block: np.ndarray

    def roster(self, face: str) -> list[dict[str, str]]:
        if face == "support_a":
            training, evaluation = self.support_b, self.support_a
        elif face == "support_b":
            training, evaluation = self.support_a, self.support_b
        else:
            raise KeyError("unknown Forecast P1 face: %s" % face)
        return (
            [{"series_uid": uid, "role": "train"} for uid in training]
            + [{"series_uid": uid, "role": "eval"} for uid in evaluation]
        )


@dataclass(frozen=True)
class BestFixedFreeze:
    program_id: str
    syntactic_programs: tuple[str, ...]
    evaluated_programs: tuple[str, ...]
    safe_rejected_programs: tuple[str, ...]
    selection_uids: tuple[str, ...]
    full_evaluations: int
    raw_consumer_fits: int
    candidate_verifier_requests: int
    verified_windows: int
    wall_seconds: float


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P1Blocked("expected a JSON object: %s" % path)
    return payload


def _assert_p0_release() -> dict[str, Any]:
    payload = _read_json(P0_REPORT)
    verdict = payload.get("verdict") or {}
    if verdict.get("audit") != "P0B_COMPLETE" or verdict.get("p1_release") is not True:
        raise P1Blocked("P0b has not released P1")
    if (verdict.get("execution")
            != "P0B_PASS__P1_BASELINE_SMOKE_RELEASED"):
        raise P1Blocked("P0b execution verdict does not release P1")
    return {
        "status": "PASS",
        "source": P0_REPORT.relative_to(PROJECT_ROOT).as_posix(),
        "p0b_audit": verdict["audit"],
        "p1_release": True,
    }


def _load_exposed_cells(
) -> tuple[ForecastCell, ForecastCell, dict[str, Any]]:
    if not KDD_ROSTER.is_file():
        raise P1Blocked("the prior KDD exposure roster is absent")
    cache = np.load(KDD_CACHE, allow_pickle=True)
    names = [str(value) for value in cache["names"]]
    arrays = cache["values"]
    positions = {name: index for index, name in enumerate(names)}
    exposed_values = {
        uid: np.asarray(arrays[positions[uid]], dtype=np.float64)
        for uid in sorted(positions)
    }
    required = ORIGIN + HORIZON

    # Structural selection only: the exposed KDD cache contains piecewise-constant
    # rows on which a legal rule can collapse a Ridge training context to the
    # scale floor.  Freeze 80 held-in rows by checking only origin-visible
    # workflow execution and scale readability.  The first 40 form the P1 target
    # 20/20 crossover; the next 40 form a disjoint EXPOSED selection cell for
    # Best Fixed.  No future loss or Support outcome participates.
    anchors = [
        int(anchor) for anchor in _config()["anchors"]
        if int(anchor) + HORIZON <= ORIGIN
    ]

    def fit_readable(uid: str) -> bool:
        raw = exposed_values[uid]
        if raw.size < required or not np.isfinite(raw[:required]).any():
            return False
        identity_history = forecast_runtime._linear_integrity(
            np.asarray(raw[ORIGIN - 192:ORIGIN], dtype=np.float64)
        )
        _center, _scale, identity_method = forecast_runtime._center_scale(
            np, identity_history
        )
        if identity_method == "scale_floor_fallback":
            return False
        for anchor in anchors:
            window = np.asarray(
                raw[anchor - 192:anchor + HORIZON], dtype=np.float64
            )
            for op in _eligible_programs():
                result = run_pipeline(
                    list(_steps(op)), window,
                    source="forecast_p1_structural_preflight",
                )
                if not result.ok or result.artifact is None:
                    return False
                prepared = forecast_runtime._linear_integrity(
                    np.asarray(result.artifact, dtype=np.float64)
                )
                _center, _scale, method = forecast_runtime._center_scale(
                    np, prepared[:192]
                )
                if method == "scale_floor_fallback":
                    return False
        return True

    structurally_readable = [uid for uid in sorted(exposed_values) if fit_readable(uid)]
    target_a = tuple(structurally_readable[:20])
    target_b = tuple(structurally_readable[20:40])
    selection_a = tuple(structurally_readable[40:60])
    selection_b = tuple(structurally_readable[60:80])
    if any(len(face) != 20 for face in (
        target_a, target_b, selection_a, selection_b
    )):
        raise P1Blocked(
            "KDD exposed pool cannot populate target and Best-Fixed cells"
        )
    target_values = {
        uid: exposed_values[uid] for uid in (*target_a, *target_b)
    }
    selection_values = {
        uid: exposed_values[uid] for uid in (*selection_a, *selection_b)
    }
    target = ForecastCell(
        values=target_values,
        support_a=target_a,
        support_b=target_b,
        observation_block=np.asarray(
            target_values[target_a[0]][:ORIGIN], dtype=np.float64
        ),
    )
    selection = ForecastCell(
        values=selection_values,
        support_a=selection_a,
        support_b=selection_b,
        observation_block=np.asarray(
            selection_values[selection_a[0]][:ORIGIN], dtype=np.float64
        ),
    )
    return target, selection, {
        "dataset": "KDD Cup 2018 with missing values",
        "data_role": "EXPOSED_DEVELOPMENT",
        "roster_path": KDD_ROSTER.relative_to(PROJECT_ROOT).as_posix(),
        "cache_path": KDD_CACHE.relative_to(PROJECT_ROOT).as_posix(),
        "support_a_series": list(target_a),
        "support_b_series": list(target_b),
        "best_fixed_selection_support_a_series": list(selection_a),
        "best_fixed_selection_support_b_series": list(selection_b),
        "best_fixed_selection_disjoint_from_target": bool(
            set((*target_a, *target_b)).isdisjoint(
                (*selection_a, *selection_b)
            )
        ),
        "selection_rule": (
            "lexicographically first 80 exposed KDD cache series readable under "
            "the current task-legal global single-step inventory using only "
            "origin-visible windows; first 40 target, next 40 disjoint "
            "Best-Fixed selection cell, each split 20/20"
        ),
        "selection_uses_support_or_future_utility": False,
        "structurally_readable_pool_count": len(structurally_readable),
        "structurally_excluded_series_count": (
            len(exposed_values) - len(structurally_readable)
        ),
        "development_query_evaluations": 0,
        "natural_final_outcome_reads": 0,
        "traffic_or_solar_loader_available": False,
    }


def _load_exposed_cell() -> tuple[ForecastCell, dict[str, Any]]:
    """Compatibility wrapper for callers that need only the target fixture."""
    target, _selection, record = _load_exposed_cells()
    return target, record


def _eligible_programs() -> tuple[str, ...]:
    programs = []
    for op in OPERATOR_NAMES:
        meta = OPERATOR_METADATA[op]
        if TASK not in tuple(meta.get("allowed_tasks") or ()):
            continue
        if bool(meta.get("shape_changing")) or bool(meta.get("changes_target_space")):
            continue
        if meta.get("requires_dependency") == "statsmodels":
            continue
        programs.append(str(op))
    return tuple(programs)


def _params(op: str) -> dict[str, object]:
    if op in {program for _name, program in FIXED_PROGRAMS}:
        return {}
    params = dict(wiring.contract_params(op, PERIOD))
    if op == "period_median_complete":
        params.update({"period": PERIOD, "cycles": 3, "min_donors": 2})
    return params


def _steps(op: str) -> tuple[tuple[str, Mapping[str, object]], ...]:
    if op == "identity":
        return ()
    return ((str(op), _params(str(op))),)


def _config() -> dict[str, object]:
    config = dict(kdd._config())
    config.update({
        "dataset_id": "forecast_p1_exposed_kdd_smoke",
        "period": PERIOD,
        "support_origin": ORIGIN,
        "selection_origin": ORIGIN,
    })
    return config


def _evaluate(
    cell: ForecastCell,
    face: str,
    op: str,
    budget: FitBudget,
) -> dict[str, Any]:
    compiled = None
    if op != "identity":
        compiled = forecast_runtime._compiled_bound_program(
            {"op": op, "params": _params(op)}, environment="forecast_p1"
        )
    budget.spend(1)
    try:
        reading = forecast_runtime._evaluate(
            cell.roster(face), cell.values, compiled, _config(), origin=ORIGIN
        )
    except Exception as exc:
        raise P1Blocked(
            "%s/%s Consumer evaluation failed: %s: %s"
            % (op, face, type(exc).__name__, exc)
        ) from exc
    mean = float(reading["mean_smase"])
    per_view = [float(value) for value in reading["per_view_smase"]]
    if not math.isfinite(mean) or not per_view or not all(map(math.isfinite, per_view)):
        raise P1Blocked("%s produced a non-finite pooled-Ridge reading" % op)
    return {
        "smase": mean,
        "utility": -mean,
        "per_series_smase": per_view,
        "behavior_point_count": int(reading.get("behavior_point_count") or 0),
    }


def _verification_executor(cell: ForecastCell) -> ScopeExecutor:
    def unavailable(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("contract-only executor cannot consume an Outcome")

    return ScopeExecutor(
        cell.roster("support_a"),
        cell.values,
        _config(),
        evaluate_fn=unavailable,
        max_modified_fraction=MAX_MODIFIED_FRACTION,
    )


def _common_dsl_contract(cell: ForecastCell) -> dict[str, Any]:
    executor = _verification_executor(cell)
    rows = []
    for op in _eligible_programs():
        compiled = forecast_runtime._compiled_bound_program(
            {"op": op, "params": _params(op)}, environment="forecast_p1_contract"
        )
        del compiled
        verification = executor.verify(_steps(op), ORIGIN)
        rows.append({
            "program": op,
            "compile": "PASS",
            "verifier": "PASS" if verification.passed else "SAFE_REJECT",
            "checked_windows": int(verification.checked_windows),
            "modified_windows": int(verification.modified_windows),
            "rejection_codes": sorted({
                str(row["rejection_code"])
                for row in verification.rejected_windows
            }),
        })
    mandatory = {program for _name, program in FIXED_PROGRAMS}
    passed = {row["program"] for row in rows if row["verifier"] == "PASS"}
    missing = sorted(mandatory - passed)
    status = "PASS" if not missing and rows else "FAIL"
    return {
        "status": status,
        "inventory_policy": "current task-legal global single-step workflows",
        "identity_available": True,
        "effect_distinct_inventory_count_from_p0b": 18,
        "eligible_operator_count": len(rows),
        "consumer_evaluations": 0,
        "contract_overhead": {
            "candidate_verifier_requests": len(rows),
            "verified_windows": sum(
                int(row["checked_windows"]) for row in rows
            ),
            "charged_to_method_cell_b4": False,
        },
        "mandatory_fixed_programs_not_executable": missing,
        "rows": rows,
    }


def _select_best_fixed_on_evolution(
    cell: ForecastCell,
) -> BestFixedFreeze:
    """Freeze Best Fixed on a disjoint EXPOSED cell before Target outcomes."""
    started = time.time()
    syntactic = tuple(dict.fromkeys(("identity", *_eligible_programs())))
    programs = tuple(
        program for program in syntactic if program not in EFFECT_ALIASES
    )
    executor = _verification_executor(cell)
    passed: list[str] = []
    safe_rejected: list[str] = []
    verified_windows = 0
    for program in programs:
        if program == "identity":
            passed.append(program)
            continue
        compiled = forecast_runtime._compiled_bound_program(
            {"op": program, "params": _params(program)},
            environment="forecast_p1_best_fixed_selection",
        )
        del compiled
        verification = executor.verify(_steps(program), ORIGIN)
        verified_windows += int(verification.checked_windows)
        if verification.passed:
            passed.append(program)
        else:
            safe_rejected.append(program)
    covered = set(passed) | set(safe_rejected) | set(EFFECT_ALIASES)
    if covered != set(syntactic):
        raise P1Blocked("Best Fixed program-space accounting is incomplete")
    if not passed:
        raise P1Blocked("Best Fixed has no verifier-approved Evolution program")

    budget = FitBudget(cap=len(passed))
    readings = {
        program: _evaluate(cell, "support_a", program, budget)
        for program in passed
    }
    loss_rows = [
        ProgramLoss(
            "support_a",
            "forecast_p1_best_fixed_evolution",
            program,
            uid,
            loss,
        )
        for program in passed
        for uid, loss in zip(
            cell.support_a,
            readings[program]["per_series_smase"],
            strict=True,
        )
    ]
    selected = select_best_fixed(loss_rows)
    return BestFixedFreeze(
        program_id=str(selected.program_id),
        syntactic_programs=syntactic,
        evaluated_programs=tuple(passed),
        safe_rejected_programs=tuple(safe_rejected),
        selection_uids=tuple(cell.support_a),
        full_evaluations=budget.used,
        raw_consumer_fits=budget.used,
        candidate_verifier_requests=len(programs) - 1,
        verified_windows=verified_windows,
        wall_seconds=round(time.time() - started, 3),
    )


def _usage(
    *,
    full: int,
    raw_fits: int,
    cheap_probes: int = 0,
    llm_calls: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    updates: int = 0,
    wall_seconds: float = 0.0,
) -> dict[str, Any]:
    total_tokens = int(input_tokens) + int(output_tokens)
    return {
        "full_support_evaluations": int(full),
        "raw_consumer_fits": int(raw_fits),
        "cheap_probes": int(cheap_probes),
        "llm_calls": int(llm_calls),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "tokens": total_tokens,
        "accepted_updates": int(updates),
        "wall_seconds": round(float(wall_seconds), 3),
        "within_caps": (
            int(full) <= B_MAIN
            and int(raw_fits) <= B_MAIN
            and int(cheap_probes) <= MAX_PROBES
            and int(llm_calls) <= MAX_LLM_CALLS
            and total_tokens <= MAX_TOKENS
            and int(updates) <= MAX_UPDATES
        ),
    }


def _method_row(
    name: str,
    *,
    status: str,
    selected: str,
    readings: Mapping[str, Any],
    usage: Mapping[str, Any],
    implementation: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "method": name,
        "status": status,
        "selected_program": selected,
        "implementation": implementation,
        "readings": _plain(readings),
        "usage": _plain(usage),
        "details": _plain(details or {}),
    }


def _deterministic_methods(
    cell: ForecastCell,
    best_fixed: BestFixedFreeze,
    identity: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity_usage = _usage(full=2, raw_fits=2)
    rows.append(_method_row(
        "Identity", status="PASS", selected="identity",
        readings=identity, usage=identity_usage,
        implementation="existing pooled/shared Ridge on unchanged input",
        details={"elementwise_unchanged": True},
    ))

    for name, program in FIXED_PROGRAMS:
        budget, started = FitBudget(), time.time()
        readings = {
            face: _evaluate(cell, face, program, budget)
            for face in ("support_a", "support_b")
        }
        for face in readings:
            readings[face]["delta_u_vs_identity"] = (
                readings[face]["utility"] - identity[face]["utility"]
            )
        rows.append(_method_row(
            name, status="PASS", selected=program, readings=readings,
            usage=_usage(full=budget.used, raw_fits=budget.used,
                         wall_seconds=time.time() - started),
            implementation="frozen one-step Common-DSL heuristic",
        ))

    # The winner was already frozen on the independent EXPOSED Evolution cell,
    # before any Target Consumer outcome was opened.
    selected = best_fixed.program_id
    budget, started = FitBudget(), time.time()
    target_readings = {
        face: _evaluate(cell, face, selected, budget)
        for face in ("support_a", "support_b")
    }
    for face, reading in target_readings.items():
        reading["delta_u_vs_identity"] = (
            reading["utility"] - identity[face]["utility"]
        )
    rows.insert(1, _method_row(
        "Best Fixed Per-task", status="PASS", selected=selected,
        readings={"target": target_readings},
        usage=_usage(full=budget.used, raw_fits=budget.used,
                     wall_seconds=time.time() - started),
        implementation=(
            "production select_best_fixed frozen on a disjoint EXPOSED cell, "
            "then carried unchanged to the target cell"
        ),
        details={
            "formal_evolution_winner_frozen": True,
            "scientific_selection_claim": False,
            "selection_uses_target_support": False,
            "selection_disjoint_from_target": bool(
                set(best_fixed.selection_uids).isdisjoint(cell.values)
            ),
            "candidate_rule": (
                "every effect-distinct Common-DSL program that passes the "
                "Evolution-cell contextual verifier"
            ),
            "selection_candidate_count": len(best_fixed.evaluated_programs),
            "selection_programs": list(best_fixed.evaluated_programs),
            "safe_rejected_programs": list(best_fixed.safe_rejected_programs),
            "effect_aliases": dict(EFFECT_ALIASES),
            "program_space_coverage_complete": bool(
                set(best_fixed.evaluated_programs)
                | set(best_fixed.safe_rejected_programs)
                | set(EFFECT_ALIASES)
                == set(best_fixed.syntactic_programs)
            ),
            "cost_by_phase": {
                "evolution_selection": {
                    "full_support_evaluations": best_fixed.full_evaluations,
                    "raw_consumer_fits": best_fixed.raw_consumer_fits,
                    "candidate_verifier_requests": (
                        best_fixed.candidate_verifier_requests
                    ),
                    "verified_windows": best_fixed.verified_windows,
                    "llm_calls": 0,
                    "tokens": 0,
                    "accepted_updates": 0,
                    "wall_seconds": best_fixed.wall_seconds,
                    "charged_to_target_b4": False,
                },
                "target_frozen_diagnostic": {
                    "full_support_evaluations": budget.used,
                    "raw_consumer_fits": budget.used,
                    "charged_to_target_b4": True,
                },
            },
        },
    ))

    budget, started = FitBudget(), time.time()
    parallel_pool = ["impute_linear", "hampel_filter", "winsorize"]
    parallel_support = {
        op: _evaluate(cell, "support_a", op, budget) for op in parallel_pool
    }
    for reading in parallel_support.values():
        reading["delta_u_vs_identity"] = (
            reading["utility"] - identity["support_a"]["utility"]
        )
    parallel_selected = min(
        parallel_pool,
        key=lambda op: (parallel_support[op]["smase"], op),
    )
    parallel_b = _evaluate(cell, "support_b", parallel_selected, budget)
    parallel_b["delta_u_vs_identity"] = (
        parallel_b["utility"] - identity["support_b"]["utility"]
    )
    rows.append(_method_row(
        "Parallel Best-of-N@4", status="PASS", selected=parallel_selected,
        readings={"support_a_candidates": parallel_support,
                  "support_b": parallel_b},
        usage=_usage(full=budget.used, raw_fits=budget.used,
                     wall_seconds=time.time() - started),
        implementation="independent Common-DSL candidate evaluations",
        details={"candidate_count": len(parallel_pool),
                 "performance_claim": False},
    ))

    budget, started = FitBudget(), time.time()
    first = "winsorize"
    first_reading = _evaluate(cell, "support_a", first, budget)
    first_delta = first_reading["utility"] - identity["support_a"]["utility"]
    second = "hampel_filter" if first_delta < 0.0 else "outlier_iqr"
    second_reading = _evaluate(cell, "support_a", second, budget)
    second_delta = second_reading["utility"] - identity["support_a"]["utility"]
    selected = second if second_delta >= first_delta else first
    delayed = _evaluate(cell, "support_b", selected, budget)
    delayed["delta_u_vs_identity"] = (
        delayed["utility"] - identity["support_b"]["utility"]
    )
    rows.append(_method_row(
        "Sequential Refinement@4", status="PASS", selected=selected,
        readings={
            "step_1": {"program": first, **first_reading,
                       "delta_u_vs_identity": first_delta},
            "step_2": {"program": second, **second_reading,
                       "delta_u_vs_identity": second_delta},
            "support_b": delayed,
        },
        usage=_usage(full=budget.used, raw_fits=budget.used,
                     wall_seconds=time.time() - started),
        implementation="feedback-conditioned second Common-DSL proposal",
        details={
            "step_2_received_step_1_feedback": True,
            "step_2_rule": "negative step-1 delta selects Hampel; otherwise IQR",
            "performance_claim": False,
        },
    ))
    return rows


def _task_contract(
    eligible: Sequence[str], *, maximum_candidates: int = B_MAIN
) -> tuple[Any, Any]:
    candidate_cap = int(maximum_candidates)
    if candidate_cap <= 0:
        raise ValueError("maximum_candidates must be positive")
    forbidden = tuple(sorted(set(OPERATOR_NAMES) - set(eligible)))
    spec = forecast_task_spec_v1(
        horizon=HORIZON,
        downstream_model_class=CONSUMER_ID,
        metric=MetricSpec(PRIMARY_METRIC, "lower_is_better"),
        forbidden_modifications=forbidden,
    )
    context = forecast_task_context_v1(
        task_spec=spec,
        deployment_constraints=deployment_constraints_v1(
            constraint_id="forecast-p1-core-smoke-v1",
            fixed_downstream_model_id="fixed:pooled-ridge-a1",
            maximum_candidates=candidate_cap,
            maximum_modified_fraction=MAX_MODIFIED_FRACTION,
        ),
    )
    return spec, context


class _FaceExecutor:
    """Dispatch Support faces and cache duplicate candidate receipts.

    ``open_delayed`` may request the same Support-B receipt several times for
    episode status, approval, and the unified metric.  Those are one piece of
    evidence: repeated requests return the same object and spend zero raw fits.
    """

    def __init__(
        self,
        faces: Mapping[int, ScopeExecutor],
        labels: Mapping[int, str] | None = None,
    ) -> None:
        self._faces = dict(faces)
        self._labels = {
            int(token): str((labels or {}).get(token, token))
            for token in faces
        }
        self._cache: dict[tuple[int, tuple[Any, ...]], Any] = {}
        self._requests = {label: 0 for label in self._labels.values()}
        self._unique = {label: 0 for label in self._labels.values()}
        self._cache_hits = {label: 0 for label in self._labels.values()}
        self._verified_windows = {label: 0 for label in self._labels.values()}

    @staticmethod
    def _program_key(steps: Any) -> tuple[Any, ...]:
        return tuple(
            (
                str(op),
                json.dumps(
                    _plain(dict(params)),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for op, params in (steps or ())
        )

    def evaluate(self, steps: Any, origin: int) -> Any:
        token = int(origin)
        if token not in self._faces:
            raise P1Blocked("unknown P1 Support face token: %s" % token)
        label = self._labels[token]
        self._requests[label] += 1
        key = (token, self._program_key(steps))
        if key in self._cache:
            self._cache_hits[label] += 1
            return self._cache[key]
        receipt = self._faces[token].evaluate(steps, ORIGIN)
        self._cache[key] = receipt
        self._unique[label] += 1
        verification = getattr(receipt, "verification", None)
        self._verified_windows[label] += int(
            getattr(verification, "checked_windows", 0) or 0
        )
        return receipt

    def accounting(self) -> dict[str, Any]:
        return {
            "requests_by_face": dict(self._requests),
            "unique_receipt_requests_by_face": dict(self._unique),
            "cache_hits_by_face": dict(self._cache_hits),
            "duplicate_requests": sum(self._cache_hits.values()),
            "unique_candidate_verifier_requests": sum(self._unique.values()),
            "verified_windows_by_face": dict(self._verified_windows),
        }


def _fast_verifier_requests(trace: Any) -> int:
    if str(trace.compilation_status or "") not in {"ok", "not_applicable"}:
        return 0
    candidate_ids = {
        str(candidate_id) for candidate_id in (trace.candidate_ids or ())
    }
    return len(candidate_ids) + len(tuple(trace.rejection_receipts or ()))


def _backend_usage(backend: Any) -> tuple[int, int, int]:
    shared = getattr(backend, "_shared", None)
    return (
        int(getattr(backend, "calls", 0) or 0),
        int(getattr(shared, "prompt_tokens", 0) or 0),
        int(getattr(shared, "completion_tokens", 0) or 0),
    )


def _snapshot_state_view(snapshot: Any) -> tuple[Any, ...]:
    """Return the fields that define whether isolated method state changed."""
    skills = tuple(
        (
            str(skill.skill_id),
            int(skill.revision),
            str(skill.body),
            json.dumps(
                _plain(skill.observable_applicability),
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                _plain(skill.risk_guards),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        for skill in snapshot.skills
    )
    memories = tuple(
        str(getattr(memory, "memory_id", "")) for memory in snapshot.memories
    )
    return skills, memories


def _new_method_state(
    *, snapshot: Any, cell: ForecastCell, backend: Any, root: Path, tag: str,
) -> dict[str, Any]:
    agent = forecast_course._live_agent(
        cell.observation_block, backend.new_arm_backend()
    )
    return four_arms._new_state(
        snapshot=snapshot, agent=agent, store_root=root, tag=tag
    )


def _audited_forecast_supply_card() -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the one audited pooled-Ridge Forecast supply card.

    The producer row is selected by its frozen role and arm, never by a gain.
    """
    checkpoint = _read_json(FORECAST_SUPPLY_ARTIFACT)
    if (
        checkpoint.get("protocol") != "s2a_forecast_curriculum_v1"
        or checkpoint.get("seed") != "r2"
    ):
        raise P1Blocked("unexpected Forecast supply source protocol or seed")
    producers = [
        row for row in (checkpoint.get("rows") or [])
        if row.get("role") == "producer" and row.get("arm") == "A5-online"
    ]
    if len(producers) != 1:
        raise P1Blocked(
            "expected exactly one frozen A5 Forecast producer row"
        )
    boundary = dict(checkpoint.get("boundary_compile") or {})
    if (
        boundary.get("unit_id") != producers[0].get("unit_id")
        or boundary.get("has_card") is not True
        or boundary.get("withheld_because") is not None
    ):
        raise P1Blocked("Forecast supply boundary does not match its producer")
    source = forecast_course._supply_row(producers[0])
    if source is None:
        raise P1Blocked("the frozen Forecast producer cannot rebuild its supply row")
    expected_scope = {
        "task_kind": TASK,
        "consumer_id": CONSUMER_ID,
        "metric": PRIMARY_METRIC,
    }
    observed_scope = {key: source.get(key) for key in expected_scope}
    if observed_scope != expected_scope:
        raise P1Blocked(
            "the audited Forecast supply row has the wrong task/Consumer scope"
        )
    compiled = forecast_course._compile_forecast_card(source)
    card = compiled.get("card")
    if not isinstance(card, Mapping):
        raise P1Blocked("the audited Forecast supply compiler withheld its card")
    card = dict(card)
    guards = dict(card.get("risk_guards") or {})
    authority = dict(guards.get("authority") or {})
    if (
        card.get("skill_id") != forecast_course.FORECAST_SKILL_ID
        or guards.get("requires_target_support") is not True
        or authority.get("supplies_candidates") is not True
        or authority.get("grants_execution") is not False
    ):
        raise P1Blocked("the audited Forecast supply card lost its Support wall")
    return card, {
        "status": "PASS",
        "source": FORECAST_SUPPLY_ARTIFACT.relative_to(PROJECT_ROOT).as_posix(),
        "producer_selection_rule": "unique role=producer and arm=A5-online",
        "skill_id": str(card["skill_id"]),
        "task_consumer_metric": expected_scope,
        "requires_target_support": True,
        "supplies_candidates": True,
        "grants_execution": False,
        "evidence_semantics": "legacy source supply-only soft prior",
        "kdd_capability_claim": False,
        "beneficiary_conversion_claim": False,
    }


def _static_method(
    *, cell: ForecastCell,
) -> dict[str, Any]:
    """Execute Static independently with no Harness lifecycle at all."""
    started = time.time()
    budget = FitBudget()
    readings = {
        face: _evaluate(cell, face, "identity", budget)
        for face in ("support_a", "support_b")
    }
    for reading in readings.values():
        reading["delta_u_vs_identity"] = 0.0
    checks = {
        "prepare_calls": 0,
        "episode_writes": 0,
        "delayed_open_calls": 0,
        "accepted_updates": 0,
        "writeback_attempts": 0,
        "store_created": False,
    }
    return _method_row(
        "Static",
        status="PASS",
        selected="identity",
        readings=readings,
        usage=_usage(
            full=budget.used,
            raw_fits=budget.used,
            wall_seconds=time.time() - started,
        ),
        implementation=(
            "independent zero-lifecycle Static baseline plus Identity Consumer "
            "diagnostic"
        ),
        details={**checks, "protocol_errors": []},
    )


def _request(cell: ForecastCell, spec: Any, context: Any) -> tuple[Any, dict[str, Any]]:
    observed = dict(resolver.window_context(cell.values, ORIGIN, PERIOD))
    observed["bound_period"] = float(PERIOD)
    features = dict(extract_public_features(
        cell.observation_block, task_kind=TASK
    ))
    request = PreparationRequest(
        "forecast-p1-exposed-kdd-smoke",
        cell.observation_block,
        spec,
        observed,
        task_context=context,
    )
    return request, features


def _harness_method(
    *,
    name: str,
    snapshot: Any,
    cell: ForecastCell,
    backend: Any,
    root: Path,
    spec: Any,
    context: Any,
    identity: Mapping[str, Mapping[str, Any]],
    initial_skill_ids: Sequence[str],
    writeback: bool,
) -> dict[str, Any]:
    started = time.time()
    before_calls, before_in, before_out = _backend_usage(backend)
    state = _new_method_state(
        snapshot=snapshot,
        cell=cell,
        backend=backend,
        root=root,
        tag=name.lower().replace(" ", "_").replace("-", "_"),
    )
    snapshot_before = state["method"]._active_snapshot()
    state_view_before = _snapshot_state_view(snapshot_before)
    request, features = _request(cell, spec, context)
    fit_budget = FitBudget()
    roster_labels = {
        tuple(
            (str(row["series_uid"]), str(row["role"]))
            for row in cell.roster(face)
        ): face
        for face in ("support_a", "support_b")
    }

    class CountingEval:
        def __init__(self) -> None:
            self.raw_fits_by_face = {"support_a": 0, "support_b": 0}

        def __call__(self, roster: Any, values: Any, compiled: Any,
                     config: Any, *, origin: int) -> dict[str, Any]:
            fit_budget.spend(1)
            key = tuple(
                (str(row["series_uid"]), str(row["role"]))
                for row in roster
            )
            face = roster_labels.get(key)
            if face is None:
                raise P1Blocked("unrecognized Forecast P1 evaluation roster")
            self.raw_fits_by_face[face] += 1
            return forecast_runtime._evaluate(
                roster, values, compiled, config, origin=origin
            )

    evaluator = CountingEval()
    faces = {
        ORIGIN: ScopeExecutor(
            cell.roster("support_a"), cell.values, _config(),
            evaluate_fn=evaluator,
            max_modified_fraction=MAX_MODIFIED_FRACTION,
        ),
        ORIGIN + 1: ScopeExecutor(
            cell.roster("support_b"), cell.values, _config(),
            evaluate_fn=evaluator,
            max_modified_fraction=MAX_MODIFIED_FRACTION,
        ),
    }
    for token, face in ((ORIGIN, "support_a"), (ORIGIN + 1, "support_b")):
        faces[token]._baseline_cache[ORIGIN] = float(identity[face]["smase"])
        faces[token]._per_view_cache[ORIGIN] = [
            float(value) for value in identity[face]["per_series_smase"]
        ]
    dispatcher = _FaceExecutor(
        faces,
        labels={ORIGIN: "support_a", ORIGIN + 1: "support_b"},
    )
    result = run_online_round(
        state["method"], dispatcher, request, cell.values,
        origin=ORIGIN,
        slow_agent=None,
        controller=state["controller"],
        store=state["store"],
        card_builder=forecast_course._card_builder,
        round_name="forecast_p1_%s" % name.lower().replace("-", "_"),
        budget=1,
        allow_slow=False,
        horizon=HORIZON,
        period=PERIOD,
        domain="forecast_p1_exposed_kdd",
        fast_features=features,
        allow_fast_skill=True,
        runtime_prior_slot=False,
        pool_mode="full",
    )
    open_delayed(
        result, dispatcher, delayed_origin=ORIGIN + 1,
        store=state["store"],
    )
    activated = False
    if writeback and result.approved_skill_id is not None:
        activated = activate_approved(result, state["store"])
    state_view_after = _snapshot_state_view(state["method"]._active_snapshot())
    new_state_created = state_view_before != state_view_after
    trace = state["method"].last_trace
    chosen = str(trace.chosen_candidate_id or "identity")
    steps_map = dict(trace.candidate_program_steps or {})
    chosen_steps = steps_map.get(chosen) or ()
    chosen_ops = [str(step[0]) for step in chosen_steps]
    invalid_ops = sorted(set(chosen_ops) - set(_eligible_programs()))
    multi_step = len(chosen_ops) > 1
    after_calls, after_in, after_out = _backend_usage(backend)
    calls = after_calls - before_calls
    input_tokens = after_in - before_in
    output_tokens = after_out - before_out
    receipt_accounting = dispatcher.accounting()
    fast_verifier_requests = _fast_verifier_requests(trace)
    retained_update = bool(writeback and activated and new_state_created)
    usage = _usage(
        full=fit_budget.used,
        raw_fits=fit_budget.used,
        cheap_probes=(
            fast_verifier_requests
            + receipt_accounting["unique_candidate_verifier_requests"]
        ),
        llm_calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        updates=int(retained_update),
        wall_seconds=time.time() - started,
    )
    episode_rows = [
        {"source_skill_id": getattr(episode, "source_skill_id", None)}
        for episode, _steps_used in (result._episodes or ())
    ]
    cross_task = forecast_course._g2_faces({
        "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
        "scope_match_by_skill_id": four_arms._scope_match_by_skill_id(
            {str(skill.skill_id): skill for skill in snapshot.skills}, features
        ),
        "episodes": episode_rows,
        "pool": list(trace.candidate_ids or ()),
    })
    cross_task["episode"] = sum(
        str(row.get("source_skill_id") or "") == forecast_course.CLS_SKILL_ID
        for row in episode_rows
    )
    forecast_skill_id = forecast_course.FORECAST_SKILL_ID
    scope_by_skill = four_arms._scope_match_by_skill_id(
        {str(skill.skill_id): skill for skill in snapshot.skills}, features
    )
    forecast_candidates = [
        str(candidate_id)
        for candidate_id in (trace.candidate_ids or ())
        if source_skill_of_candidate(candidate_id) == forecast_skill_id
    ]
    forecast_probes = [
        str(probe.get("candidate_id"))
        for probe in (result.actual_probed_programs or ())
        if source_skill_of_candidate(probe.get("candidate_id"))
        == forecast_skill_id
    ]
    forecast_faces = {
        "retrieval": int(
            forecast_skill_id in tuple(trace.retrieved_skill_ids or ())
        ),
        "scope_match": int(bool(scope_by_skill.get(forecast_skill_id))),
        "supply": int(bool(forecast_candidates)),
        "support_probe": int(bool(forecast_probes)),
    }
    support_b_unique = int(
        receipt_accounting["unique_receipt_requests_by_face"].get(
            "support_b", 0
        )
    )
    approved_after_support_b = bool(
        result.approved_skill_id is not None and support_b_unique >= 1
    )
    source_prior_deployed = (
        str(result.deployed_skill_id or "") == forecast_skill_id
    )
    source_support_chain_complete = bool(
        source_prior_deployed
        and forecast_faces["support_probe"]
        and support_b_unique >= 1
        and result.delayed_utility is not None
    )
    wrong_promotion = bool(retained_update and not approved_after_support_b)
    errors = []
    if invalid_ops:
        errors.append("task_mismatch_or_noninventory_program")
    if multi_step:
        errors.append("unfrozen_multi_step_program")
    if not usage["within_caps"]:
        errors.append("budget_cap_exceeded")
    if any(
        int(cross_task[key])
        for key in ("retrieval", "scope_match", "supply", "episode")
    ):
        errors.append("cross_task_skill_leakage")
    if (
        not forecast_faces["scope_match"]
        and any(forecast_faces[key] for key in ("retrieval", "supply", "support_probe"))
    ):
        errors.append("forecast_supply_scope_bypass")
    if source_prior_deployed and not source_support_chain_complete:
        errors.append("historical_skill_bypassed_support")
    if wrong_promotion:
        errors.append("wrong_promotion")
    support_delta_u = next(
        (
            float(probe["gain"])
            for probe in (result.actual_probed_programs or ())
            if probe.get("candidate_id") == result._winner_candidate_id
            and probe.get("gain") is not None
        ),
        None,
    )
    delayed_delta_u = (
        float(result.delayed_utility)
        if result.delayed_utility is not None else None
    )
    status = "PASS" if not errors else "FAIL"
    return _method_row(
        name,
        status=status,
        selected="identity" if not chosen_ops else "+".join(chosen_ops),
        readings={
            "support_receipts": int(result.target_support_receipts_used),
            "support_delta_u_vs_identity": support_delta_u,
            "support_candidate_utility": (
                identity["support_a"]["utility"] + support_delta_u
                if support_delta_u is not None else None
            ),
            "delayed_delta_u_vs_identity": delayed_delta_u,
            "delayed_candidate_utility": (
                identity["support_b"]["utility"] + delayed_delta_u
                if delayed_delta_u is not None else None
            ),
            "abstained": bool(result.abstained),
            "harm_count": int(result.harm_count),
        },
        usage=usage,
        implementation="production TTHAMethod + run_online_round + Support-B wall",
        details={
            "initial_skill_ids": list(initial_skill_ids),
            "writeback_channel": bool(writeback),
            "unit_state_discarded": not bool(writeback),
            "next_unit_base": (
                "isolated_evolved_state" if retained_update
                else "shared_initial_state"
            ),
            "carried_episode_count": 0 if not writeback else len(episode_rows),
            "carried_new_skill_count": int(retained_update),
            "retained_update": retained_update,
            "writeback_treatment": (
                "RETAINED_NEW_STATE" if retained_update else "NOT_EXERCISED"
            ),
            "writeback_persisted_to_evolution_store": False,
            "candidate_count": len(trace.candidate_ids or ()),
            "chosen_program_steps": [
                {"op": str(op), "params": dict(params)}
                for op, params in chosen_steps
            ],
            "approved_after_support_b": approved_after_support_b,
            "method_state_changed_inside_isolated_unit": new_state_created,
            "existing_source_prior_confirmed": source_support_chain_complete,
            "cross_task_faces": cross_task,
            "forecast_source_faces": forecast_faces,
            "forecast_source_scope_fail_closed": bool(
                not forecast_faces["scope_match"]
                and not forecast_faces["retrieval"]
                and not forecast_faces["supply"]
                and not forecast_faces["support_probe"]
            ),
            "fast_candidate_verifier_requests": fast_verifier_requests,
            "receipt_accounting": receipt_accounting,
            "identity_baseline_cache_reused": True,
            "raw_consumer_fits_by_face": dict(evaluator.raw_fits_by_face),
            "protocol_errors": errors,
        },
    )


def _frozen_h0(
    *, cell: ForecastCell, snapshot: Any, backend: Any, root: Path,
    spec: Any, context: Any, identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    before_calls, before_in, before_out = _backend_usage(backend)
    state = _new_method_state(
        snapshot=snapshot, cell=cell, backend=backend, root=root, tag="frozen_h0"
    )
    request, _features = _request(cell, spec, context)
    state["method"].bind_round_data(cell.observation_block, task_kind=TASK)
    state["method"].prepare(
        request,
        runtime_prior_slot=False,
        pool_mode="full",
    )
    trace = state["method"].last_trace
    chosen = str(trace.chosen_candidate_id or "identity")
    steps = tuple((trace.candidate_program_steps or {}).get(chosen) or ())
    ops = [str(step[0]) for step in steps]
    errors = []
    if len(ops) > 1:
        errors.append("unfrozen_multi_step_program")
    if set(ops) - set(_eligible_programs()):
        errors.append("task_mismatch_or_noninventory_program")
    budget = FitBudget()
    selected = ops[0] if ops else "identity"
    reading = _evaluate(cell, "support_a", selected, budget)
    reading["delta_u_vs_identity"] = (
        reading["utility"] - identity["support_a"]["utility"]
    )
    after_calls, after_in, after_out = _backend_usage(backend)
    fast_verifier_requests = _fast_verifier_requests(trace)
    usage = _usage(
        full=budget.used,
        raw_fits=budget.used,
        cheap_probes=fast_verifier_requests,
        llm_calls=after_calls - before_calls,
        input_tokens=after_in - before_in,
        output_tokens=after_out - before_out,
        wall_seconds=time.time() - started,
    )
    if not usage["within_caps"]:
        errors.append("budget_cap_exceeded")
    return _method_row(
        "Frozen H0", status="PASS" if not errors else "FAIL",
        selected=selected,
        readings={"support_a": reading},
        usage=usage,
        implementation="production H0 Fast path; no Support adaptation or writeback",
        details={
            "candidate_count": len(trace.candidate_ids or ()),
            "fast_candidate_verifier_requests": fast_verifier_requests,
            "writeback": False,
            "protocol_errors": errors,
        },
    )


def _harness_methods(
    cell: ForecastCell,
    identity: Mapping[str, Mapping[str, Any]],
    backend_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = _eligible_programs()
    spec, context = _task_contract(eligible)
    backend = (
        shared_harness._live_backend(LIVE_GLOBAL_CALL_CAP)
        if backend_mode == "live"
        else shared_harness._scripted_backend(LIVE_GLOBAL_CALL_CAP)
    )
    temp_root = Path(tempfile.mkdtemp(prefix="forecast_p1_"))
    try:
        h0 = forecast_course._h0()
        wrong_task_card = forecast_course._cls_card()
        forecast_card, forecast_card_contract = _audited_forecast_supply_card()
        wrong_task_origin = forecast_course._install(
            h0, wrong_task_card, store_root=temp_root / "initial", tag="k0_a5"
        )
        shared_origin = forecast_course._install(
            wrong_task_origin,
            forecast_card,
            store_root=temp_root / "initial_forecast",
            tag="k0_a5_forecast",
        )
        h0_ids = sorted(str(skill.skill_id) for skill in h0.skills)
        shared_ids = sorted(str(skill.skill_id) for skill in shared_origin.skills)
        rows = [
            _frozen_h0(
                cell=cell, snapshot=h0, backend=backend, root=temp_root,
                spec=spec, context=context, identity=identity,
            ),
            _static_method(cell=cell),
            _harness_method(
                name="A3-reset", snapshot=h0, cell=cell, backend=backend,
                root=temp_root, spec=spec, context=context,
                identity=identity,
                initial_skill_ids=h0_ids, writeback=False,
            ),
            _harness_method(
                name="K0-fixed", snapshot=shared_origin, cell=cell,
                backend=backend, root=temp_root, spec=spec, context=context,
                identity=identity,
                initial_skill_ids=shared_ids, writeback=False,
            ),
            _harness_method(
                name="A5-online", snapshot=shared_origin, cell=cell,
                backend=backend, root=temp_root, spec=spec, context=context,
                identity=identity,
                initial_skill_ids=shared_ids, writeback=True,
            ),
        ]
        calls, input_tokens, output_tokens = _backend_usage(backend)
        return rows, {
            "mode": backend_mode,
            "production_format_exercised": True,
            "live_transport_exercised": backend_mode == "live",
            "global_llm_calls": calls,
            "global_input_tokens": input_tokens,
            "global_output_tokens": output_tokens,
            "k0_a5_same_initial_state": True,
            "k0_a5_initial_skill_ids": shared_ids,
            "k0_a5_forecast_supply_contract": forecast_card_contract,
            "temporary_store_removed_after_run": True,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _aegists_spike() -> dict[str, Any]:
    root = PROJECT_ROOT.parent / "a-evolve/AegisTS"
    required = [
        root / "Error_Cleaner/RLclean.py",
        root / "Error_Cleaner/final_model.py",
        root / "Error_Cleaner/strategy.py",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return {
            "status": "STRUCTURALLY_INCOMPATIBLE",
            "tier": "related-work",
            "reason": "reference implementation unavailable for bounded spike",
            "missing_files": missing,
            "blocking": False,
        }
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in required)
    checks = {
        "uses_own_hierarchical_rl_action_space": "RLCleanEnvironment" in text,
        "requires_labels_inside_cleaning_environment": "label" in text,
        "uses_own_task_models": "LSTMForecast" in text or "DLinear" in text,
        "does_not_export_project_typed_workflow": "Typed Workflow" not in text,
    }
    return {
        "status": "STRUCTURALLY_INCOMPATIBLE",
        "tier": "related-work",
        "reason": (
            "the reference implementation couples its hierarchical RL action space, "
            "labels/reward, and task models; it does not expose a pure selector over "
            "this project's frozen Common DSL / Consumer / B=4 contract"
        ),
        "checks": checks,
        "source_tree_read_only": True,
        "blocking": False,
    }


def _validate_methods(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    by_name = {str(row["method"]): row for row in rows}
    missing = sorted(set(MANDATORY_METHODS) - set(by_name))
    if missing:
        failures.append("missing mandatory methods: %s" % missing)
    duplicates = len(rows) - len(by_name)
    if duplicates:
        failures.append("duplicate mandatory method rows: %d" % duplicates)
    for name in MANDATORY_METHODS:
        row = by_name.get(name)
        if row is None:
            continue
        if row.get("status") != "PASS":
            failures.append("%s did not pass" % name)
        if not bool((row.get("usage") or {}).get("within_caps")):
            failures.append("%s exceeded a resource cap" % name)
        if (row.get("details") or {}).get("protocol_errors"):
            failures.append("%s recorded a protocol error" % name)
    return failures


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Forecast P1 Core baseline smoke",
        "",
        "**Verdict: `%s`. Forecast component pass: `%s`. Overall P1 complete: `%s`.**"
        % (payload["verdict"], payload["forecast_component_pass"],
           payload["overall_p1_complete"]),
        "",
        "This is an infrastructure/contract smoke on exposed KDD development data. "
        "It makes no performance, headroom, treatment, or capability claim.",
        "",
        "Natural Final outcome reads: **0**. Development Query evaluations: **0**.",
        "",
        "## Core methods",
        "",
        "| method | status | selected | fits | LLM calls | tokens |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload.get("methods") or []:
        usage = row.get("usage") or {}
        lines.append(
            "| %s | `%s` | `%s` | %s | %s | %s |"
            % (row["method"], row["status"], row["selected_program"],
               usage.get("raw_consumer_fits", 0), usage.get("llm_calls", 0),
               usage.get("tokens", 0))
        )
    lines.extend([
        "",
        "## Boundary and release",
        "",
        "- Common DSL contract: `%s`." % payload["common_dsl_contract"]["status"],
        "- AegisTS-style spike: `%s` (non-blocking)."
        % payload["aegists_adapter"]["status"],
        "- P2 is **not** released by this Forecast-only tranche; Classification and "
        "AD P1 components remain pending.",
        "- Final outcomes remain sealed.",
        "",
        "Machine-readable detail: `artifacts/main_protocol/"
        "forecast_p1_core_smoke_20260830.json`.",
        "",
    ])
    return "\n".join(lines)


def run(*, backend_mode: str) -> dict[str, Any]:
    started = time.time()
    p0 = _assert_p0_release()
    cell, selection_cell, data = _load_exposed_cells()
    common_dsl = _common_dsl_contract(cell)
    best_fixed = _select_best_fixed_on_evolution(selection_cell)
    identity_budget = FitBudget()
    identity = {
        face: _evaluate(cell, face, "identity", identity_budget)
        for face in ("support_a", "support_b")
    }
    for reading in identity.values():
        reading["delta_u_vs_identity"] = 0.0
    deterministic = _deterministic_methods(
        cell,
        best_fixed,
        identity,
    )
    harness_rows, backend = _harness_methods(cell, identity, backend_mode)
    methods = deterministic + harness_rows
    failures = _validate_methods(methods)
    if common_dsl["status"] != "PASS":
        failures.append("Common DSL contract failed")
    protocol_errors = {
        "natural_final_outcome_reads": 0,
        "development_query_evaluations": 0,
        "task_mismatch_execution": sum(
            "task_mismatch_or_noninventory_program"
            in ((row.get("details") or {}).get("protocol_errors") or [])
            for row in methods
        ),
        "cross_task_skill_leakage": sum(
            "cross_task_skill_leakage"
            in ((row.get("details") or {}).get("protocol_errors") or [])
            for row in methods
        ),
        "historical_skill_bypassed_support": sum(
            "historical_skill_bypassed_support"
            in ((row.get("details") or {}).get("protocol_errors") or [])
            for row in methods
        ),
        "wrong_promotion": sum(
            "wrong_promotion"
            in ((row.get("details") or {}).get("protocol_errors") or [])
            for row in methods
        ),
    }
    if any(protocol_errors.values()):
        failures.append("one or more protocol errors were observed")
    component_pass = not failures
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "P1_COMMON_DSL_AND_CORE_BASELINE_SMOKE",
        "task_tranche": TASK,
        "evidence_grade": EVIDENCE_GRADE,
        "verdict": (
            "FORECAST_P1_CORE_BASELINE_SMOKE_PASS"
            if component_pass else "FORECAST_P1_CORE_BASELINE_SMOKE_BLOCKED"
        ),
        "forecast_component_pass": component_pass,
        "overall_p1_complete": False,
        "release_p2": False,
        "pending_p1_task_tranches": ["classification", "anomaly_detection"],
        "p0b_release": p0,
        "data": data,
        "consumer": {
            "id": CONSUMER_ID,
            "implementation": "existing pooled/shared Ridge",
            "primary_metric": PRIMARY_METRIC,
            "utility_definition": "U_forecast = -sMASE",
            "delta_definition": "delta_U = U(method) - U(identity)",
        },
        "split": {
            "origin": ORIGIN,
            "horizon": HORIZON,
            "period": PERIOD,
            "support_a_count": len(cell.support_a),
            "support_b_count": len(cell.support_b),
            "training_series_per_face": 20,
            "query_count": 0,
        },
        "execution_order": [
            "common_dsl_contract_without_outcome",
            "best_fixed_frozen_on_disjoint_exposed_evolution_cell",
            "target_consumer_diagnostics",
        ],
        "budget_caps": {
            "full_support_evaluations": B_MAIN,
            "raw_consumer_fits": B_MAIN,
            "cheap_probes": MAX_PROBES,
            "llm_calls": MAX_LLM_CALLS,
            "tokens": MAX_TOKENS,
            "accepted_updates": MAX_UPDATES,
        },
        "common_dsl_contract": common_dsl,
        "methods": methods,
        "aegists_adapter": _aegists_spike(),
        "backend": backend,
        "protocol_errors": protocol_errors,
        "blocking_failures": failures,
        "performance_or_headroom_claim": False,
        "treatment_or_capability_claim": False,
        "wall_seconds": round(time.time() - started, 3),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(_plain(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("scripted", "live"), default="scripted",
        help=(
            "scripted is the reproducible P1 contract backend; live is an "
            "optional transport diagnostic and is not a P1 gate"
        ),
    )
    parser.add_argument(
        "--expect-pass", action="store_true",
        help="exit non-zero unless the Forecast component passes",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = run(backend_mode=args.backend)
    except Exception as exc:  # noqa: BLE001 - one bounded first-fault report
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "P1_COMMON_DSL_AND_CORE_BASELINE_SMOKE",
            "task_tranche": TASK,
            "evidence_grade": EVIDENCE_GRADE,
            "verdict": "FORECAST_P1_CORE_BASELINE_SMOKE_BLOCKED",
            "forecast_component_pass": False,
            "overall_p1_complete": False,
            "release_p2": False,
            "pending_p1_task_tranches": ["forecast", "classification", "anomaly_detection"],
            "blocking_failures": ["%s: %s" % (type(exc).__name__, exc)],
            "protocol_errors": {
                "natural_final_outcome_reads": 0,
                "development_query_evaluations": 0,
            },
            "performance_or_headroom_claim": False,
            "treatment_or_capability_claim": False,
        }
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(
            json.dumps(_plain(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        OUT_MD.write_text(_markdown({
            **payload,
            "methods": [],
            "common_dsl_contract": {"status": "NOT_COMPLETED"},
            "aegists_adapter": {"status": "NOT_REACHED"},
        }), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"],
        "forecast_component_pass": payload["forecast_component_pass"],
        "overall_p1_complete": payload["overall_p1_complete"],
        "release_p2": payload["release_p2"],
        "blocking_failures": payload.get("blocking_failures") or [],
        "output_json": OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "output_md": OUT_MD.relative_to(PROJECT_ROOT).as_posix(),
    }, indent=2, ensure_ascii=False), flush=True)
    return int(bool(args.expect_pass and not payload["forecast_component_pass"]))


if __name__ == "__main__":
    raise SystemExit(main())
