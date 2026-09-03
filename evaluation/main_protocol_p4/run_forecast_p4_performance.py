"""Run the Forecast component of v1.2 P4-Performance.

This entry point collects H1/H2 evidence on exposed KDD Evolution surfaces.
It does not exercise or release the independent H3/RQ3 gate, and it has no
loader for Query, Natural Final, UCR TEST, or the sealed AD series.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    activate_approved,
    open_delayed,
    run_online_round,
    source_skill_of_candidate,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver

from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms
from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import run_p4 as split_release


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json"
)
ADDENDUM = (
    PROJECT_ROOT / "docs/P3_EVIDENCE_ADDENDUM_AND_P4_GATE_AMENDMENT_20260830.md"
)

PROTOCOL_VERSION = "v1.2.1-Core+p4-split-3-forecast-b8-llm8"
STAGE = "P4_FORECAST_PERFORMANCE_EVOLUTION"
EVIDENCE_GRADE = "EVOLUTION_PERFORMANCE"
TASK = "forecast"
CONSUMER_ID = forecast_p1.CONSUMER_ID
PRIMARY_METRIC = forecast_p1.PRIMARY_METRIC
PERIOD = forecast_p1.PERIOD
HORIZON = forecast_p1.HORIZON
MATERIAL = resolver.MATERIAL_THRESHOLD

ORIGINS = (600, 648, 696, 744, 792, 840, 888, 936)
REPLICA_ORDERS: Mapping[str, tuple[int, ...]] = {
    "Forward": ORIGINS,
    "Reverse": tuple(reversed(ORIGINS)),
    "Interleaved": (
        ORIGINS[0], ORIGINS[4], ORIGINS[1], ORIGINS[5],
        ORIGINS[2], ORIGINS[6], ORIGINS[3], ORIGINS[7],
    ),
}
CORE_ARMS = ("Static", "A3-reset", "K0-fixed", "A5-online")
ADAPTIVE_ARMS = CORE_ARMS[1:]
PARALLEL_COMPARATOR = "Parallel Best-of-N@8"
PARALLEL_PROGRAMS = (
    "impute_linear",
    "hampel_filter",
    "winsorize",
    "outlier_iqr",
    "impute_fft",
    "impute_ema",
    "period_complete",
)

B_MAIN = 8
MAX_SUPPORT_A = 7
MAX_SUPPORT_B = 1
MAX_CHEAP_PROBES = 24
MAX_LLM_CALLS = 8
MAX_TOKENS = 60_000
MAX_UPDATES = 1
MAX_WALL_SECONDS = 45 * 60
MAX_MODIFIED_FRACTION = forecast_p1.MAX_MODIFIED_FRACTION
CELL_LLM_EXHAUSTION_VERDICT = "LLM_CELL_BUDGET_EXHAUSTED"
CELL_LLM_EXHAUSTION_ACTION = "ABSTAIN_TO_IDENTITY_AND_CONTINUE"


class ForecastP4Blocked(RuntimeError):
    """A release, boundary, budget, or runtime condition failed closed."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ForecastP4Blocked("required P4 split gate is absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ForecastP4Blocked("P4 split gate must be a JSON object")
    return value


def _write(payload: Mapping[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    staged = OUT_JSON.with_suffix(OUT_JSON.suffix + ".tmp")
    staged.write_text(
        json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    staged.replace(OUT_JSON)


def _persist_llm_budget(
    payload: dict[str, Any], state: Mapping[str, Any]
) -> None:
    payload["llm_budget_instrument"] = {
        "classification": "PRE_CALL_HARD_GUARD",
        "scientific_claim": "NONE__INSTRUMENT_ONLY",
        "frozen_per_cell_cap": MAX_LLM_CALLS,
        "resume_count_preserved": True,
        "arm_counts_isolated": True,
        **_plain(state),
    }
    _write(payload)


def _assert_release() -> dict[str, Any]:
    gate = _read_object(split_release.OUT_JSON)
    performance = dict(gate.get("p4_performance") or {})
    evolution = dict(gate.get("p4_evolution") or {})
    plan = dict(gate.get("execution_plan") or {})
    forecast_budget = dict(plan.get("forecast_budget") or {})
    expected_budget = {
        "operating_point": "B=8",
        "full_support_consumer_evaluations": B_MAIN,
        "support_a_max": MAX_SUPPORT_A,
        "support_b_max": MAX_SUPPORT_B,
        "cheap_probe_max": MAX_CHEAP_PROBES,
        "llm_call_max": MAX_LLM_CALLS,
        "token_max": MAX_TOKENS,
        "accepted_update_max": MAX_UPDATES,
        "wall_seconds_max": MAX_WALL_SECONDS,
    }
    checks = {
        "performance_released": performance.get("status") == "RELEASED",
        "forecast_authorized": performance.get("forecast_launch_authorized") is True,
        "evolution_held": evolution.get("status") == "HELD",
        "rq3_does_not_block_h1_h2": (
            performance.get("rq3_not_exercised_is_not_a_blocker") is True
        ),
        "four_core_arms_frozen": tuple(plan.get("arms") or ()) == CORE_ARMS,
        "three_replicas_frozen": tuple(plan.get("replicas") or ()) == tuple(
            REPLICA_ORDERS
        ),
        "eight_episodes_frozen": int(plan.get("episodes_per_task") or 0) == 8,
        "forecast_b8_scope_frozen": (
            plan.get("budget_scope") == "FORECAST_P4_PERFORMANCE_ONLY"
        ),
        "forecast_budget_vector_frozen": forecast_budget == expected_budget,
        "forecast_budget_aliases_consistent": (
            int(plan.get("full_support_budget") or 0) == B_MAIN
            and int(plan.get("support_a_budget") or 0) == MAX_SUPPORT_A
            and int(plan.get("support_b_budget") or 0) == MAX_SUPPORT_B
            and int(plan.get("cheap_probe_budget") or 0) == MAX_CHEAP_PROBES
            and int(plan.get("llm_call_budget") or 0) == MAX_LLM_CALLS
            and int(plan.get("token_budget") or 0) == MAX_TOKENS
            and int(plan.get("accepted_update_budget") or 0) == MAX_UPDATES
            and int(plan.get("wall_seconds_budget") or 0)
            == MAX_WALL_SECONDS
        ),
        "matched_b8_baseline_frozen": (
            plan.get("matched_baseline") == PARALLEL_COMPARATOR
        ),
        "matched_budget": plan.get("matched_budget") is True,
        "adaptive_arms_share_budget": (
            plan.get("adaptive_arms_share_exact_budget_vector") is True
            and plan.get("a5_budget_exception") is False
        ),
        "cell_llm_exhaustion_is_identity_abstain": (
            plan.get("cell_llm_budget_exhaustion_action")
            == CELL_LLM_EXHAUSTION_ACTION
            and plan.get("cell_llm_budget_exhaustion_reason")
            == CELL_LLM_EXHAUSTION_VERDICT
            and plan.get("partial_cell_state_writeback") is False
            and plan.get("budget_exhaustion_rate_reported_by_arm") is True
        ),
        "task_local_only": plan.get("task_local_experience_and_skill_only") is True,
        "final_closed": (
            gate.get("natural_final_release") is False
            and int(gate.get("final_outcome_reads", -1)) == 0
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ForecastP4Blocked("P4 split release failed: %s" % failed)
    return {
        "source": split_release.OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "checks": checks,
        "performance_status": performance["status"],
        "evolution_status": evolution["status"],
    }


def unit_plan(replicas: Sequence[str] | None = None) -> list[dict[str, Any]]:
    selected = tuple(replicas or REPLICA_ORDERS)
    unknown = sorted(set(selected) - set(REPLICA_ORDERS))
    if unknown:
        raise ForecastP4Blocked("unknown replica: %s" % unknown)
    rows: list[dict[str, Any]] = []
    for replica in selected:
        for sequence_index, origin in enumerate(REPLICA_ORDERS[replica], start=1):
            rows.append(
                {
                    "replica": replica,
                    "sequence_index": sequence_index,
                    "episode_id": "E%d" % (ORIGINS.index(origin) + 1),
                    "origin": origin,
                    "horizon": HORIZON,
                    "natural_episode": True,
                }
            )
    return rows


def budget_contract(replica_count: int = 3) -> dict[str, Any]:
    adaptive_cells = int(replica_count) * len(ORIGINS) * len(ADAPTIVE_ARMS)
    return {
        "operating_point": "B=8",
        "per_method_cell": {
            "full_support_consumer_evaluations": B_MAIN,
            "support_a_max": MAX_SUPPORT_A,
            "support_b_max": MAX_SUPPORT_B,
            "cheap_probe_max": MAX_CHEAP_PROBES,
            "llm_call_max": MAX_LLM_CALLS,
            "token_max": MAX_TOKENS,
            "accepted_update_max": MAX_UPDATES,
            "wall_seconds_max": MAX_WALL_SECONDS,
        },
        "adaptive_cell_count": adaptive_cells,
        "global_llm_call_cap": adaptive_cells * MAX_LLM_CALLS,
        "core_cell_count": int(replica_count) * len(ORIGINS) * len(CORE_ARMS),
        "h2_comparator_cell_count": int(replica_count) * len(ORIGINS),
        "matched_ceiling_not_required_spend": True,
    }


def validate_usage(usage: Mapping[str, Any]) -> bool:
    support_a = int(usage.get("support_a_full_evaluations") or 0)
    support_b = int(usage.get("support_b_full_evaluations") or 0)
    full = int(usage.get("full_support_evaluations") or 0)
    return bool(
        support_a <= MAX_SUPPORT_A
        and support_b <= MAX_SUPPORT_B
        and full == support_a + support_b
        and full <= B_MAIN
        and int(usage.get("raw_consumer_fits") or 0) == full
        and int(usage.get("cheap_probes") or 0) <= MAX_CHEAP_PROBES
        and int(usage.get("llm_calls") or 0) <= MAX_LLM_CALLS
        and int(usage.get("tokens") or 0) <= MAX_TOKENS
        and int(usage.get("accepted_updates") or 0) <= MAX_UPDATES
        and float(usage.get("wall_seconds") or 0.0) <= MAX_WALL_SECONDS
    )


def _config(origin: int) -> dict[str, object]:
    config = dict(forecast_p1._config())
    config.update(
        {
            "dataset_id": "forecast_p4_exposed_kdd_natural_evolution",
            "support_origin": int(origin),
            "selection_origin": int(origin),
            "period": PERIOD,
        }
    )
    return config


def _cell_at(base: forecast_p1.ForecastCell, origin: int) -> forecast_p1.ForecastCell:
    required = int(origin) + HORIZON
    short = sorted(
        uid for uid, values in base.values.items() if int(values.size) < required
    )
    if short:
        raise ForecastP4Blocked(
            "frozen Forecast roster is too short at origin %d" % origin
        )
    return forecast_p1.ForecastCell(
        values=base.values,
        support_a=base.support_a,
        support_b=base.support_b,
        observation_block=np.asarray(
            base.values[base.support_a[0]][:origin], dtype=np.float64
        ),
    )


def _reading(
    cell: forecast_p1.ForecastCell,
    face: str,
    steps: Sequence[tuple[str, Mapping[str, object]]],
    *,
    origin: int,
) -> dict[str, Any]:
    compiled = None
    if steps:
        if len(steps) != 1:
            raise ForecastP4Blocked("fixed comparator received a multi-step program")
        op, params = steps[0]
        compiled = forecast_p1.forecast_runtime._compiled_bound_program(
            {"op": str(op), "params": dict(params)},
            environment="forecast_p4_performance",
        )
    raw = forecast_p1.forecast_runtime._evaluate(
        cell.roster(face), cell.values, compiled, _config(origin), origin=origin
    )
    smase = float(raw["mean_smase"])
    per_series = [float(value) for value in raw["per_view_smase"]]
    if not math.isfinite(smase) or not per_series or not all(
        math.isfinite(value) for value in per_series
    ):
        raise ForecastP4Blocked("non-finite Forecast Consumer reading")
    return {
        "smase": smase,
        "utility": -smase,
        "median_series_smase": float(np.median(per_series)),
        "worst_series_smase": max(per_series),
        "per_series_smase": per_series,
        "behavior_point_count": int(raw.get("behavior_point_count") or 0),
    }


def _identity_reference(
    cell: forecast_p1.ForecastCell, origin: int
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    readings: dict[str, dict[str, Any]] = {}
    wall: dict[str, float] = {}
    for face in ("support_a", "support_b"):
        started = time.time()
        readings[face] = _reading(cell, face, (), origin=origin)
        wall[face] = round(time.time() - started, 3)
    return readings, wall


def _base_row(
    *,
    unit: Mapping[str, Any],
    method: str,
    reading: Mapping[str, Any],
    delta: float,
    usage: Mapping[str, Any],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    value = float(delta)
    return {
        **dict(unit),
        "task": TASK,
        "consumer": CONSUMER_ID,
        "metric": PRIMARY_METRIC,
        "method": method,
        "task_native": _plain(reading),
        "delta_utility_vs_identity": value,
        "material_harm_event": value < -MATERIAL,
        "material_harm_magnitude": max(0.0, -value),
        "usage": _plain(usage),
        "details": _plain(details),
        "status": "PASS" if validate_usage(usage) else "FAIL",
    }


def _static_row(
    unit: Mapping[str, Any],
    identity: Mapping[str, Mapping[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    usage = {
        "support_a_full_evaluations": 0,
        "support_b_full_evaluations": 1,
        "full_support_evaluations": 1,
        "raw_consumer_fits": 1,
        "cheap_probes": 0,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens": 0,
        "accepted_updates": 0,
        "wall_seconds": float(wall_seconds),
    }
    return _base_row(
        unit=unit,
        method="Static",
        reading=identity["support_b"],
        delta=0.0,
        usage=usage,
        details={
            "selected_program": "identity",
            "lifecycle_calls": 0,
            "identity_reference_fit_allocated_here": True,
            "cross_unit_writeback": False,
        },
    )


def _parallel_row(
    unit: Mapping[str, Any],
    cell: forecast_p1.ForecastCell,
    origin: int,
    identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    support: dict[str, dict[str, Any]] = {}
    for op in PARALLEL_PROGRAMS:
        support[op] = _reading(
            cell, "support_a", forecast_p1._steps(op), origin=origin
        )
    selected = min(
        PARALLEL_PROGRAMS,
        key=lambda op: (float(support[op]["smase"]), op),
    )
    delayed = _reading(
        cell, "support_b", forecast_p1._steps(selected), origin=origin
    )
    delta = float(delayed["utility"]) - float(identity["support_b"]["utility"])
    usage = {
        "support_a_full_evaluations": len(PARALLEL_PROGRAMS),
        "support_b_full_evaluations": 1,
        "full_support_evaluations": len(PARALLEL_PROGRAMS) + 1,
        "raw_consumer_fits": len(PARALLEL_PROGRAMS) + 1,
        "cheap_probes": 0,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens": 0,
        "accepted_updates": 0,
        "wall_seconds": round(time.time() - started, 3),
    }
    return _base_row(
        unit=unit,
        method=PARALLEL_COMPARATOR,
        reading=delayed,
        delta=delta,
        usage=usage,
        details={
            "candidate_programs": list(PARALLEL_PROGRAMS),
            "candidate_order_source": (
                "P1 fixed order followed by existing eligible registry order"
            ),
            "support_a_candidate_utility": {
                op: float(value["utility"]) for op, value in support.items()
            },
            "selected_program": selected,
            "independent_search_only": True,
            "cross_unit_writeback": False,
        },
    )


class _CountingEval:
    def __init__(
        self,
        cell: forecast_p1.ForecastCell,
        config: Mapping[str, object],
        origin: int,
    ) -> None:
        self.cell = cell
        self.config = dict(config)
        self.origin = int(origin)
        self.fits_by_face = {"support_a": 0, "support_b": 0}
        self._roster_labels = {
            tuple(
                (str(row["series_uid"]), str(row["role"]))
                for row in cell.roster(face)
            ): face
            for face in self.fits_by_face
        }

    def __call__(
        self,
        roster: Any,
        values: Any,
        compiled: Any,
        config: Any,
        *,
        origin: int,
    ) -> dict[str, Any]:
        if int(origin) != self.origin:
            raise ForecastP4Blocked("Forecast Consumer received the wrong origin")
        key = tuple(
            (str(row["series_uid"]), str(row["role"])) for row in roster
        )
        face = self._roster_labels.get(key)
        if face is None:
            raise ForecastP4Blocked("Forecast Consumer received an unknown roster")
        self.fits_by_face[face] += 1
        return forecast_p1.forecast_runtime._evaluate(
            roster, values, compiled, config, origin=origin
        )


class _OriginDispatcher:
    """Map lifecycle face tokens to one natural origin and cache repeat reads."""

    def __init__(
        self,
        entries: Mapping[int, tuple[str, ScopeExecutor, int]],
    ) -> None:
        self._entries = dict(entries)
        labels = [label for label, _executor, _origin in entries.values()]
        self._requests = {label: 0 for label in labels}
        self._unique = {label: 0 for label in labels}
        self._cache_hits = {label: 0 for label in labels}
        self._cache: dict[tuple[int, tuple[Any, ...]], Any] = {}

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
        if token not in self._entries:
            raise ForecastP4Blocked("unknown Forecast P4 face token")
        label, executor, real_origin = self._entries[token]
        self._requests[label] += 1
        key = (token, self._program_key(steps))
        if key in self._cache:
            self._cache_hits[label] += 1
            return self._cache[key]
        receipt = executor.evaluate(steps, real_origin)
        self._cache[key] = receipt
        self._unique[label] += 1
        return receipt

    def accounting(self) -> dict[str, Any]:
        return {
            "requests_by_face": dict(self._requests),
            "unique_by_face": dict(self._unique),
            "cache_hits_by_face": dict(self._cache_hits),
            "duplicate_requests": sum(self._cache_hits.values()),
            "unique_candidate_verifier_requests": sum(self._unique.values()),
        }


def _request(
    *,
    unit: Mapping[str, Any],
    cell: forecast_p1.ForecastCell,
    origin: int,
    spec: Any,
    context: Any,
) -> tuple[PreparationRequest, dict[str, Any]]:
    observed = dict(resolver.window_context(cell.values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    features = dict(
        extract_public_features(cell.observation_block, task_kind=TASK)
    )
    request = PreparationRequest(
        "forecast-p4-%s-%s" % (unit["replica"], unit["episode_id"]),
        cell.observation_block,
        spec,
        observed,
        task_context=context,
    )
    return request, features


def _episode_rows(method: Any, start: int) -> list[dict[str, Any]]:
    rows = []
    for episode in method.experience_episodes[start:]:
        support = dict(getattr(episode, "support_response", None) or {})
        delayed = dict(getattr(episode, "delayed_response", None) or {})
        context = dict(getattr(episode, "context_summary", None) or {})
        rows.append(
            {
                "episode_id": str(getattr(episode, "episode_id", "")),
                "source_skill_id": context.get("source_skill_id"),
                "relation": str(getattr(episode, "relation", "")),
                "local_status": str(getattr(episode, "local_status", "")),
                "support_gain": support.get("gain"),
                "delayed_gain": delayed.get("gain"),
            }
        )
    return rows


def _budget_exhausted_row(
    *,
    unit: Mapping[str, Any],
    arm: str,
    identity: Mapping[str, Mapping[str, Any]],
    method: Any,
    before_snapshot: Any,
    before_semantics: Any,
    episode_start: int,
    backend: Any,
    before_backend_usage: tuple[int, int, int],
    evaluator: _CountingEval,
    dispatcher: _OriginDispatcher,
    started: float,
) -> dict[str, Any]:
    """Turn one exhausted adaptive cell into an atomic identity abstention."""
    before_calls, before_input, before_output = before_backend_usage
    after_calls, after_input, after_output = forecast_p1._backend_usage(backend)
    accounting = dispatcher.accounting()
    trace = getattr(method, "last_trace", None)
    fast_verifier_requests = (
        forecast_p1._fast_verifier_requests(trace) if trace is not None else 0
    )
    support_a_fits = int(evaluator.fits_by_face["support_a"])
    support_b_fits = int(evaluator.fits_by_face["support_b"])
    full_fits = support_a_fits + support_b_fits
    current_snapshot = method._active_snapshot()
    discarded_partial_episodes = _episode_rows(method, episode_start)
    input_tokens = after_input - before_input
    output_tokens = after_output - before_output
    usage = {
        "support_a_full_evaluations": support_a_fits,
        "support_b_full_evaluations": support_b_fits,
        "full_support_evaluations": full_fits,
        "raw_consumer_fits": full_fits,
        "cheap_probes": int(
            fast_verifier_requests
            + accounting["unique_candidate_verifier_requests"]
        ),
        "llm_calls": after_calls - before_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": input_tokens + output_tokens,
        "accepted_updates": 0,
        "wall_seconds": round(time.time() - started, 3),
    }
    return _base_row(
        unit=unit,
        method=arm,
        reading=identity["support_b"],
        delta=0.0,
        usage=usage,
        details={
            "selected_program": "identity",
            "effective_program": "identity",
            "abstained": True,
            "abstain_reason": CELL_LLM_EXHAUSTION_VERDICT,
            "llm_budget_exhausted": True,
            "budget_exhaustion_action": CELL_LLM_EXHAUSTION_ACTION,
            "episodes_written": [],
            "discarded_partial_episodes": discarded_partial_episodes,
            "state_skill_ids_before": sorted(
                str(skill.skill_id) for skill in before_snapshot.skills
            ),
            "state_skill_ids_after": sorted(
                str(skill.skill_id) for skill in before_snapshot.skills
            ),
            "discarded_partial_state_changed": (
                forecast_p1._snapshot_state_view(current_snapshot)
                != before_semantics
            ),
            "state_changed_inside_unit": False,
            "update_event": "BUDGET_EXHAUSTED_ABSTAIN_IDENTITY",
            "cross_unit_writeback": False,
            "cross_unit_writeback_authorized": arm == "A5-online",
            "unit_state_discarded": True,
            "natural_agent_feedback_only": True,
            "slow_revision_exercised": False,
            "receipt_accounting": accounting,
        },
    )


def _adaptive_row(
    *,
    unit: Mapping[str, Any],
    arm: str,
    cell: forecast_p1.ForecastCell,
    origin: int,
    base_snapshot: Any,
    carried_episodes: Sequence[Any],
    backend: Any,
    temp_root: Path,
    spec: Any,
    context: Any,
    identity: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Any, tuple[Any, ...]]:
    started = time.time()
    before_calls, before_input, before_output = forecast_p1._backend_usage(backend)
    tag = "%s_%s_%s" % (
        unit["replica"].lower(),
        unit["episode_id"].lower(),
        arm.lower().replace("-", "_"),
    )
    budget_scope = "%s/%s/%s" % (
        unit["replica"], unit["episode_id"], arm
    )
    arm_backend = backend.new_arm_backend(
        scope_id=budget_scope,
        maximum_calls=MAX_LLM_CALLS,
    )
    state = four_arms._new_state(
        snapshot=base_snapshot,
        agent=forecast_course._live_agent(
            cell.observation_block, arm_backend
        ),
        store_root=temp_root,
        tag=tag,
        episodes=tuple(carried_episodes),
    )
    method = state["method"]
    before_snapshot = method._active_snapshot()
    before_semantics = forecast_p1._snapshot_state_view(before_snapshot)
    episode_start = len(method.experience_episodes)
    request, features = _request(
        unit=unit, cell=cell, origin=origin, spec=spec, context=context
    )
    config = _config(origin)
    evaluator = _CountingEval(cell, config, origin)
    support_token = int(origin)
    delayed_token = int(origin) + HORIZON
    support_executor = ScopeExecutor(
        cell.roster("support_a"),
        cell.values,
        config,
        evaluate_fn=evaluator,
        max_modified_fraction=MAX_MODIFIED_FRACTION,
    )
    delayed_executor = ScopeExecutor(
        cell.roster("support_b"),
        cell.values,
        config,
        evaluate_fn=evaluator,
        max_modified_fraction=MAX_MODIFIED_FRACTION,
    )
    for executor, face in (
        (support_executor, "support_a"),
        (delayed_executor, "support_b"),
    ):
        executor._baseline_cache[origin] = float(identity[face]["smase"])
        executor._per_view_cache[origin] = [
            float(value) for value in identity[face]["per_series_smase"]
        ]
    dispatcher = _OriginDispatcher(
        {
            support_token: ("support_a", support_executor, origin),
            delayed_token: ("support_b", delayed_executor, origin),
        }
    )
    try:
        result = run_online_round(
            method,
            dispatcher,
            request,
            cell.values,
            origin=support_token,
            slow_agent=None,
            controller=state["controller"],
            store=state["store"],
            card_builder=forecast_course._card_builder,
            round_name=tag,
            budget=MAX_SUPPORT_A,
            allow_slow=False,
            horizon=HORIZON,
            period=PERIOD,
            domain="forecast_p4_natural_%s" % unit["episode_id"],
            fast_features=features,
            allow_fast_skill=True,
            runtime_prior_slot=False,
            pool_mode="full",
        )
    except shared_harness.Stop as exc:
        if exc.verdict != CELL_LLM_EXHAUSTION_VERDICT:
            raise
        return (
            _budget_exhausted_row(
                unit=unit,
                arm=arm,
                identity=identity,
                method=method,
                before_snapshot=before_snapshot,
                before_semantics=before_semantics,
                episode_start=episode_start,
                backend=backend,
                before_backend_usage=(
                    before_calls,
                    before_input,
                    before_output,
                ),
                evaluator=evaluator,
                dispatcher=dispatcher,
                started=started,
            ),
            base_snapshot,
            tuple(carried_episodes),
        )
    open_delayed(
        result,
        dispatcher,
        delayed_origin=delayed_token,
        store=state["store"],
    )
    activated = False
    if result.approved_skill_id is not None:
        activated = activate_approved(result, state["store"])
    after_snapshot = method._active_snapshot()
    after_semantics = forecast_p1._snapshot_state_view(after_snapshot)
    semantic_change = before_semantics != after_semantics

    delayed_delta = (
        float(result.delayed_utility)
        if result.delayed_utility is not None
        else 0.0
    )
    per_series_gain: list[float] = []
    delayed_behavior_points = 0
    if result._winner_steps is not None:
        delayed_receipt = dispatcher.evaluate(result._winner_steps, delayed_token)
        per_series_gain = [
            float(value) for value in (delayed_receipt.per_view_gain or ())
        ]
        delayed_behavior_points = int(delayed_receipt.behavior_point_count or 0)
    identity_b = identity["support_b"]
    if result.delayed_utility is not None and len(per_series_gain) != len(
        identity_b["per_series_smase"]
    ):
        raise ForecastP4Blocked("delayed per-series Forecast reading is incomplete")
    if per_series_gain:
        per_series = [
            float(reference) - float(gain)
            for reference, gain in zip(
                identity_b["per_series_smase"], per_series_gain, strict=True
            )
        ]
    else:
        per_series = [float(value) for value in identity_b["per_series_smase"]]
    smase = float(identity_b["smase"]) - delayed_delta
    reading = {
        "smase": smase,
        "utility": -smase,
        "median_series_smase": float(np.median(per_series)),
        "worst_series_smase": max(per_series),
        "per_series_smase": per_series,
        "behavior_point_count": delayed_behavior_points,
    }

    after_calls, after_input, after_output = forecast_p1._backend_usage(backend)
    calls = after_calls - before_calls
    input_tokens = after_input - before_input
    output_tokens = after_output - before_output
    accounting = dispatcher.accounting()
    support_a_fits = int(evaluator.fits_by_face["support_a"])
    support_b_fits = int(evaluator.fits_by_face["support_b"])
    full_fits = support_a_fits + support_b_fits
    usage = {
        "support_a_full_evaluations": support_a_fits,
        "support_b_full_evaluations": support_b_fits,
        "full_support_evaluations": full_fits,
        "raw_consumer_fits": full_fits,
        "cheap_probes": int(
            forecast_p1._fast_verifier_requests(method.last_trace)
            + accounting["unique_candidate_verifier_requests"]
        ),
        "llm_calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": input_tokens + output_tokens,
        "accepted_updates": int(semantic_change),
        "wall_seconds": round(time.time() - started, 3),
    }
    fresh_episodes = _episode_rows(method, episode_start)
    trace = method.last_trace
    winner_candidate = str(result._winner_candidate_id or "")
    if result.revoked_skill_id is not None:
        update_event = "REVOKED_AFTER_NATURAL_HARM"
    elif activated:
        update_event = "ACTIVATED_AFTER_INDEPENDENT_SUPPORT_B"
    else:
        update_event = "NO_RETAINED_STATE_CHANGE"
    row = _base_row(
        unit=unit,
        method=arm,
        reading=reading,
        delta=delayed_delta,
        usage=usage,
        details={
            "selected_program": _plain(result.winner_program or []),
            "winner_candidate": winner_candidate or None,
            "winner_source_skill_id": source_skill_of_candidate(winner_candidate),
            "abstained": bool(result.abstained),
            "abstain_reason": (
                "AGENT_OR_SUPPORT_ABSTAIN" if result.abstained else None
            ),
            "llm_budget_exhausted": False,
            "support_harm_count": int(result.harm_count),
            "support_harm_magnitude": float(result.harm_magnitude),
            "support_receipts_used": int(result.target_support_receipts_used),
            "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
            "candidate_count": len(trace.candidate_ids or ()),
            "episodes_written": fresh_episodes,
            "state_skill_ids_before": sorted(
                str(skill.skill_id) for skill in before_snapshot.skills
            ),
            "state_skill_ids_after": sorted(
                str(skill.skill_id) for skill in after_snapshot.skills
            ),
            "state_changed_inside_unit": semantic_change,
            "update_event": update_event,
            "cross_unit_writeback": arm == "A5-online",
            "unit_state_discarded": arm != "A5-online",
            "natural_agent_feedback_only": True,
            "slow_revision_exercised": False,
            "receipt_accounting": accounting,
        },
    )
    return row, after_snapshot, tuple(method.experience_episodes)


def _paired_contrast(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    by_key = {
        (str(row["replica"]), str(row["episode_id"]), str(row["method"])): float(
            (row.get("task_native") or {}).get("utility")
        )
        for row in rows
    }
    values = []
    for replica in REPLICA_ORDERS:
        for origin in REPLICA_ORDERS[replica]:
            episode = "E%d" % (ORIGINS.index(origin) + 1)
            left_key = (replica, episode, left)
            right_key = (replica, episode, right)
            if left_key in by_key and right_key in by_key:
                values.append(by_key[left_key] - by_key[right_key])
    return {
        "left": left,
        "right": right,
        "paired_n": len(values),
        "mean_delta_utility": float(np.mean(values)) if values else None,
        "median_delta_utility": float(np.median(values)) if values else None,
        "material_win_count": sum(value > MATERIAL for value in values),
        "material_harm_count": sum(value < -MATERIAL for value in values),
        "paired_differences": values,
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    for method in (*CORE_ARMS, PARALLEL_COMPARATOR):
        selected = [row for row in rows if row.get("method") == method]
        if not selected:
            continue
        utilities = [float(row["task_native"]["utility"]) for row in selected]
        deltas = [float(row["delta_utility_vs_identity"]) for row in selected]
        by_method[method] = {
            "n": len(selected),
            "mean_utility": float(np.mean(utilities)),
            "mean_smase": float(
                np.mean([float(row["task_native"]["smase"]) for row in selected])
            ),
            "mean_delta_utility_vs_identity": float(np.mean(deltas)),
            "material_harm_count": sum(
                bool(row["material_harm_event"]) for row in selected
            ),
            "material_harm_rate": float(
                np.mean([bool(row["material_harm_event"]) for row in selected])
            ),
            "consumer_fits": sum(
                int(row["usage"]["raw_consumer_fits"]) for row in selected
            ),
            "llm_calls": sum(int(row["usage"]["llm_calls"]) for row in selected),
            "tokens": sum(int(row["usage"]["tokens"]) for row in selected),
            "wall_seconds": round(
                sum(float(row["usage"]["wall_seconds"]) for row in selected), 3
            ),
            "llm_budget_exhaustion_count": sum(
                bool((row.get("details") or {}).get("llm_budget_exhausted"))
                for row in selected
            ),
            "llm_budget_exhaustion_rate": (
                float(
                    np.mean(
                        [
                            bool(
                                (row.get("details") or {}).get(
                                    "llm_budget_exhausted"
                                )
                            )
                            for row in selected
                        ]
                    )
                )
                if method in ADAPTIVE_ARMS
                else None
            ),
            "llm_budget_exhaustion_applicability": (
                "APPLICABLE"
                if method in ADAPTIVE_ARMS
                else "NOT_APPLICABLE__NO_LLM_BUDGET"
            ),
        }
    return {
        "by_method": by_method,
        "llm_budget_exhaustion_efficiency_by_arm": {
            method: {
                "completed_cells": by_method[method]["n"],
                "exhaustion_count": by_method[method][
                    "llm_budget_exhaustion_count"
                ],
                "exhaustion_rate": by_method[method][
                    "llm_budget_exhaustion_rate"
                ],
                "llm_calls": by_method[method]["llm_calls"],
                "tokens": by_method[method]["tokens"],
                "wall_seconds": by_method[method]["wall_seconds"],
            }
            for method in ADAPTIVE_ARMS
            if method in by_method
        },
        "confirmatory_performance_contrasts": {
            "H1_A5_minus_A3": _paired_contrast(
                rows, "A5-online", "A3-reset"
            ),
            "H2_A5_minus_Parallel": _paired_contrast(
                rows, "A5-online", PARALLEL_COMPARATOR
            ),
        },
        "evolution_diagnostic_only": {
            "A5_minus_K0": _paired_contrast(rows, "A5-online", "K0-fixed"),
            "claim_status": "RQ3_NOT_EXERCISED",
            "independent_h3_gate_changed": False,
        },
    }


def _initial_payload(
    *, release: Mapping[str, Any], replicas: Sequence[str], backend_mode: str
) -> dict[str, Any]:
    plan = unit_plan(replicas)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "RUNNING",
        "started_at": _now(),
        "completed_at": None,
        "backend_mode": backend_mode,
        "release": dict(release),
        "scope": {
            "performance_hypotheses": ["H1", "H2"],
            "evolution_h3_status": "HELD__RQ3_NOT_EXERCISED",
            "ad_scope": "NOT_RUN_IN_FORECAST_COMPONENT",
            "natural_agent_feedback_treatment": True,
            "controlled_treatment_rows": 0,
            "injected_treatment_rows": 0,
            "query_evaluations": 0,
            "natural_final_outcome_reads": 0,
            "ucr_test_outcome_reads": 0,
            "sealed_ad_outcome_reads": 0,
        },
        "frozen_contract": {
            "task": TASK,
            "consumer": CONSUMER_ID,
            "primary_metric": PRIMARY_METRIC,
            "common_dsl_source": "P1 frozen Forecast Common DSL",
            "core_arms": list(CORE_ARMS),
            "h2_comparator": PARALLEL_COMPARATOR,
            "origins": list(ORIGINS),
            "replica_orders": {
                name: list(REPLICA_ORDERS[name]) for name in replicas
            },
            "k0_a5_same_initial_knowledge": True,
            "a5_only_cross_unit_writeback": True,
            "task_local_experience_and_skill_only": True,
            "no_operator_consumer_metric_or_threshold_change": True,
            "cell_llm_budget_exhaustion_action": (
                CELL_LLM_EXHAUSTION_ACTION
            ),
            "partial_exhausted_cell_state_writeback": False,
            "budget_exhaustion_rate_reported_by_arm": True,
        },
        "budget": budget_contract(len(replicas)),
        "unit_plan": plan,
        "expected": {
            "core_cells": len(plan) * len(CORE_ARMS),
            "h2_comparator_cells": len(plan),
        },
        "progress": {
            "completed_units": 0,
            "completed_core_cells": 0,
            "completed_h2_comparator_cells": 0,
        },
        "llm_budget_instrument": {
            "classification": "PRE_CALL_HARD_GUARD",
            "scientific_claim": "NONE__INSTRUMENT_ONLY",
            "frozen_per_cell_cap": MAX_LLM_CALLS,
            "resume_count_preserved": True,
            "arm_counts_isolated": True,
            "global_cap": budget_contract(len(replicas))["global_llm_call_cap"],
            "global_calls": 0,
            "scope_calls": {},
            "scope_caps": {},
            "call_records": [],
            "blocked_records": [],
        },
        "data": None,
        "initial_knowledge": None,
        "rows": [],
        "aggregates": None,
        "protocol_errors": [],
    }


def run(
    *, replicas: Sequence[str], backend_mode: str, expect_pass: bool
) -> dict[str, Any]:
    release = _assert_release()
    payload = _initial_payload(
        release=release, replicas=replicas, backend_mode=backend_mode
    )
    _write(payload)
    try:
        base_cell, _selection, data = forecast_p1._load_exposed_cells()
        for origin in ORIGINS:
            _cell_at(base_cell, origin)
        payload["data"] = {
            "dataset": data["dataset"],
            "data_role": "EXPOSED_EVOLUTION",
            "support_a_series_count": len(base_cell.support_a),
            "support_b_series_count": len(base_cell.support_b),
            "natural_origin_count": len(ORIGINS),
            "treatment_generation": "natural Agent proposals plus task-native feedback",
            "controlled_or_injected_treatment": False,
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
        }
        eligible = forecast_p1._eligible_programs()
        spec, context = forecast_p1._task_contract(
            eligible, maximum_candidates=B_MAIN
        )
        global_call_cap = budget_contract(len(replicas))["global_llm_call_cap"]
        on_budget_change = lambda state: _persist_llm_budget(payload, state)
        backend = (
            shared_harness._live_backend(
                global_call_cap,
                on_budget_change=on_budget_change,
            )
            if backend_mode == "live"
            else shared_harness._scripted_backend(
                global_call_cap,
                on_budget_change=on_budget_change,
            )
        )
        with tempfile.TemporaryDirectory(prefix="forecast_p4_") as temp_name:
            temp_root = Path(temp_name)
            h0 = forecast_course._h0()
            card, card_contract = forecast_p1._audited_forecast_supply_card()
            shared_initial = forecast_course._install(
                h0,
                card,
                store_root=temp_root / "initial",
                tag="forecast_task_local_initial",
            )
            shared_semantics = forecast_p1._snapshot_state_view(shared_initial)
            payload["initial_knowledge"] = {
                "source": card_contract,
                "historical_task_skill_ids": [str(card["skill_id"])],
                "wrong_task_skill_installed": False,
                "k0_a5_initial_semantics_equal": bool(
                    shared_semantics
                    == forecast_p1._snapshot_state_view(shared_initial)
                ),
                "only_a5_carries_state_across_units": True,
            }
            _write(payload)

            for replica in replicas:
                a5_snapshot = shared_initial
                a5_episodes: tuple[Any, ...] = ()
                for unit in unit_plan((replica,)):
                    origin = int(unit["origin"])
                    cell = _cell_at(base_cell, origin)
                    identity, identity_wall = _identity_reference(cell, origin)
                    payload["rows"].append(
                        _static_row(
                            unit,
                            identity,
                            identity_wall["support_b"],
                        )
                    )
                    payload["rows"].append(
                        _parallel_row(unit, cell, origin, identity)
                    )
                    payload.setdefault("shared_identity_reference", []).append(
                        {
                            **dict(unit),
                            "raw_consumer_fits": 2,
                            "support_b_fit_allocated_to_static": 1,
                            "unallocated_support_a_reference_fit": 1,
                            "wall_seconds": round(sum(identity_wall.values()), 3),
                            "unallocated_support_a_wall_seconds": identity_wall[
                                "support_a"
                            ],
                        }
                    )

                    for arm in ADAPTIVE_ARMS:
                        if arm == "A3-reset":
                            base_snapshot = h0
                            carried: tuple[Any, ...] = ()
                        elif arm == "K0-fixed":
                            base_snapshot = shared_initial
                            carried = ()
                        else:
                            base_snapshot = a5_snapshot
                            carried = a5_episodes
                        row, end_snapshot, end_episodes = _adaptive_row(
                            unit=unit,
                            arm=arm,
                            cell=cell,
                            origin=origin,
                            base_snapshot=base_snapshot,
                            carried_episodes=carried,
                            backend=backend,
                            temp_root=temp_root,
                            spec=spec,
                            context=context,
                            identity=identity,
                        )
                        payload["rows"].append(row)
                        if arm == "A5-online":
                            a5_snapshot = end_snapshot
                            a5_episodes = end_episodes
                        if expect_pass and row["status"] != "PASS":
                            raise ForecastP4Blocked(
                                "%s/%s exceeded a frozen contract"
                                % (unit["replica"], arm)
                            )

                    payload["progress"] = {
                        "completed_units": int(
                            payload["progress"]["completed_units"]
                        ) + 1,
                        "completed_core_cells": sum(
                            row["method"] in CORE_ARMS for row in payload["rows"]
                        ),
                        "completed_h2_comparator_cells": sum(
                            row["method"] == PARALLEL_COMPARATOR
                            for row in payload["rows"]
                        ),
                    }
                    _write(payload)

        payload["aggregates"] = aggregate(payload["rows"])
        expected = payload["expected"]
        actual_core = int(payload["progress"]["completed_core_cells"])
        actual_parallel = int(
            payload["progress"]["completed_h2_comparator_cells"]
        )
        errors = []
        if actual_core != int(expected["core_cells"]):
            errors.append("core_cell_count_mismatch")
        if actual_parallel != int(expected["h2_comparator_cells"]):
            errors.append("h2_comparator_cell_count_mismatch")
        if any(row["status"] != "PASS" for row in payload["rows"]):
            errors.append("method_cell_contract_failure")
        if any(
            int(payload["scope"][field]) != 0
            for field in (
                "query_evaluations",
                "natural_final_outcome_reads",
                "ucr_test_outcome_reads",
                "sealed_ad_outcome_reads",
            )
        ):
            errors.append("sealed_boundary_read")
        payload["protocol_errors"] = errors
        payload["status"] = "COMPLETE" if not errors else "FAILED"
        payload["completed_at"] = _now()
        payload["p4_performance_collection_complete"] = not errors
        payload["p4_evolution_gate"] = "HELD__RQ3_NOT_EXERCISED"
        payload["natural_final_release"] = False
        payload["final_outcome_reads"] = 0
        _write(payload)
        if expect_pass and errors:
            raise ForecastP4Blocked("Forecast P4 protocol errors: %s" % errors)
        return payload
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["completed_at"] = _now()
        if (
            isinstance(exc, shared_harness.Stop)
            and exc.verdict in {
                "LLM_CELL_BUDGET_EXHAUSTED",
                "LLM_BUDGET_EXCEEDED",
            }
        ):
            payload["failure_classification"] = (
                "BUDGET_INSTRUMENT_LIMIT__NO_SCIENTIFIC_VERDICT"
            )
        payload["protocol_errors"] = list(payload.get("protocol_errors") or ()) + [
            "%s: %s" % (type(exc).__name__, exc)
        ]
        payload["failure_trace"] = traceback.format_exc()
        payload["natural_final_release"] = False
        payload["final_outcome_reads"] = 0
        _write(payload)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replica",
        choices=("all", *REPLICA_ORDERS),
        default="all",
    )
    parser.add_argument("--backend", choices=("live", "scripted"), default="live")
    parser.add_argument("--expect-pass", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    replicas = (
        tuple(REPLICA_ORDERS)
        if args.replica == "all"
        else (str(args.replica),)
    )
    payload = run(
        replicas=replicas,
        backend_mode=str(args.backend),
        expect_pass=bool(args.expect_pass),
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "completed_core_cells": payload["progress"][
                    "completed_core_cells"
                ],
                "completed_h2_comparator_cells": payload["progress"][
                    "completed_h2_comparator_cells"
                ],
                "p4_evolution_gate": payload["p4_evolution_gate"],
                "final_outcome_reads": payload["final_outcome_reads"],
                "output": OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
