"""In-memory Anomaly Detection component for the v1.2.1 P1 Core smoke.

This is a contract/infrastructure smoke, not the AD positive control.  It uses
two deterministic, mutually exclusive synthetic fixtures and the frozen
``aegists_iforest_v1`` Consumer.  It has no loader or writer for Yahoo, NAB,
or any Final surface, and it returns a plain payload without writing reports.

The P1 budget counts task-native full Event-F1 evaluations.  A frozen IForest
fits once per series for a program, then the same adapter reuses those models
on Support-B.  Raw per-series fits are therefore reported separately and are
not compared with the four-evaluation method cap.
"""
from __future__ import annotations

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
    anomaly_task_context_v1,
    anomaly_task_spec_v1,
    deployment_constraints_v1,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.baselines import (
    ProgramLoss,
    select_best_fixed,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    activate_approved,
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.online_loop import source_skill_of_candidate
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES

from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms
from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.functional import run_v1_signed_agent_action_wiring as wiring
from evaluation.functional.consumers import aegists_iforest_v1 as iforest_consumer
from evaluation.functional.consumers.p0b_scope_adapters import (
    TrainingBlockScopeExecutor,
    WindowedIForestAdapter,
)
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver


PROTOCOL_VERSION = "v1.2.1-Core"
TASK = "anomaly_detection"
TASK_TRANCHE = "anomaly_detection"
EVIDENCE_GRADE = "INFRASTRUCTURE"
CONSUMER_ID = "aegists_iforest_v1"
PRIMARY_METRIC = "macro_event_f1"
ROUND_NAME = "p1_synthetic"
PERIOD = 24
SUPPORT_A_TOKEN = 420
SUPPORT_B_TOKEN = 421
MAX_MODIFIED_FRACTION = 0.20
B_MAIN = 4
MAX_PROBES = 12
MAX_LLM_CALLS = 4
MAX_TOKENS = 40_000
MAX_UPDATES = 1
GLOBAL_SCRIPTED_CALL_CAP = 16

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
EXPECTED_AD_OPERATORS = (
    "impute_linear",
    "impute_fft",
    "impute_ema",
    "period_complete",
    "period_median_complete",
    "impute_ar",
    "winsorize",
    "outlier_iqr",
    "outlier_mad",
    "hampel_filter",
    "resample_uniform",
)
EFFECT_ALIASES = {"resample_uniform": "identity"}


class P1AnomalyBlocked(RuntimeError):
    """A mechanical P1 contract condition could not be completed truthfully."""


class RawFitLedger:
    """Count actual per-series Consumer fits without imposing the B=4 cap."""

    def __init__(self) -> None:
        self.used = 0

    def spend(self, count: int = 1) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("raw fit count cannot be negative")
        self.used += count


class LogicalEvaluationBudget:
    """Cap only complete task-native Event-F1 evaluations."""

    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0

    def spend(self, count: int = 1) -> None:
        count = int(count)
        if self.used + count > self.cap:
            raise P1AnomalyBlocked(
                "logical Event-F1 cap exceeded: %d + %d > %d"
                % (self.used, count, self.cap)
            )
        self.used += count


class _LabelWall:
    """Expose only the two synthetic Support faces already in memory."""

    def __init__(
        self,
        events: Mapping[tuple[str, int, int], Sequence[Sequence[int]]],
    ) -> None:
        self._events = {
            (str(uid), int(lo), int(hi)): tuple(tuple(map(int, event)) for event in rows)
            for (uid, lo, hi), rows in events.items()
        }
        self.requests: list[tuple[str, int, int]] = []

    def read(self, uid: str, lo: int, hi: int) -> Sequence[Sequence[int]]:
        key = (str(uid), int(lo), int(hi))
        if key not in self._events:
            raise P1AnomalyBlocked("attempted to read a non-Support AD label surface")
        self.requests.append(key)
        return self._events[key]


@dataclass(frozen=True)
class ADCell:
    cell_id: str
    rows: Mapping[str, Mapping[str, Any]]
    events: Mapping[tuple[str, int, int], Sequence[Sequence[int]]]
    observation_block: np.ndarray

    @property
    def values(self) -> dict[str, np.ndarray]:
        return {
            str(uid): np.asarray(row["values"], dtype=np.float64)
            for uid, row in self.rows.items()
        }

    @property
    def uids(self) -> tuple[str, ...]:
        return tuple(sorted(str(uid) for uid in self.rows))

    def roster(self) -> list[dict[str, str]]:
        return [{"series_uid": uid, "role": "train"} for uid in self.uids]


@dataclass(frozen=True)
class BestFixedFreeze:
    program_id: str
    syntactic_programs: tuple[str, ...]
    evaluated_programs: tuple[str, ...]
    safe_rejected_programs: tuple[str, ...]
    selection_uids: tuple[str, ...]
    logical_evaluations: int
    raw_series_fits: int
    verifier_requests: int
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


def _synthetic_series(prefix: str, index: int, selection: bool) -> tuple[str, dict[str, Any], dict[tuple[str, int, int], Any]]:
    uid = "%s_series_%d" % (prefix, index + 1)
    t = np.arange(640, dtype=np.float64)
    phase = 2.0 * index + (1.25 if selection else 0.0)
    values = (
        0.72 * np.sin(2.0 * np.pi * (t + phase) / 30.0)
        + 0.18 * np.sin(2.0 * np.pi * (t + 3.0 * phase) / 11.0)
        + (0.00035 + index * 0.00005) * t
    )
    # Origin-visible sparse contamination makes outlier programs contextual,
    # while staying comfortably below AD's 20% modification bound.
    train_centres = (72 + 9 * index, 171 + 7 * index, 312 + 5 * index)
    for number, centre in enumerate(train_centres):
        values[centre:centre + 2] += (6.2 + 0.35 * index) * (-1.0 if number == 1 else 1.0)

    a_start = 448 + 4 * index + (3 if selection else 0)
    b_start = 528 + 3 * index + (2 if selection else 0)
    values[a_start:a_start + 3] += 7.8 + 0.3 * index
    values[b_start:b_start + 3] -= 7.6 + 0.25 * index
    windows = {
        ROUND_NAME: {
            "train": [0, SUPPORT_A_TOKEN],
            "support_a": [SUPPORT_A_TOKEN, 500],
            "support_b": [500, 580],
        }
    }
    events = {
        (uid, SUPPORT_A_TOKEN, 500): [list(range(a_start, a_start + 3))],
        (uid, 500, 580): [list(range(b_start, b_start + 3))],
    }
    return uid, {"values": values, "windows": windows}, events


def _synthetic_fixtures() -> tuple[ADCell, ADCell, dict[str, Any]]:
    def build(prefix: str, selection: bool) -> ADCell:
        rows: dict[str, Mapping[str, Any]] = {}
        events: dict[tuple[str, int, int], Any] = {}
        for index in range(2):
            uid, row, labels = _synthetic_series(prefix, index, selection)
            rows[uid] = row
            events.update(labels)
        first = np.asarray(rows[sorted(rows)[0]]["values"], dtype=np.float64)
        return ADCell(
            cell_id=prefix,
            rows=rows,
            events=events,
            observation_block=first[:SUPPORT_A_TOKEN].copy(),
        )

    target = build("target_fixture", False)
    selection = build("selection_fixture", True)
    disjoint = set(target.uids).isdisjoint(selection.uids)
    if not disjoint:
        raise P1AnomalyBlocked("synthetic target and selection fixtures overlap")
    return target, selection, {
        "dataset": "deterministic synthetic IForest contract fixtures",
        "data_role": "CONTROLLED_EXPOSED_FIXTURE",
        "target_series": list(target.uids),
        "best_fixed_selection_series": list(selection.uids),
        "best_fixed_selection_disjoint_from_target": True,
        "selection_uses_support_or_future_utility": False,
        "natural_final_outcome_reads": 0,
        "development_query_evaluations": 0,
        "yahoo_loader_available": False,
        "nab_loader_available": False,
        "sealed_ad_series_available": False,
    }


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
    params = dict(wiring.contract_params(str(op), PERIOD))
    if op == "period_median_complete":
        params.update({"period": PERIOD, "cycles": 3, "min_donors": 2})
    return params


def _steps(op: str) -> tuple[tuple[str, Mapping[str, object]], ...]:
    if op == "identity":
        return ()
    return ((str(op), _params(str(op))),)


class _EvaluationSession:
    """One adapter instance across both Support faces for one method."""

    def __init__(self, cell: ADCell, logical_cap: int) -> None:
        self.cell = cell
        self.raw = RawFitLedger()
        self.logical = LogicalEvaluationBudget(logical_cap)
        self.logical_by_face = {"support_a": 0, "support_b": 0}
        self.labels = _LabelWall(cell.events)
        self.adapter = WindowedIForestAdapter(
            consumer=iforest_consumer,
            rows=cell.rows,
            round_name=ROUND_NAME,
            event_reader=self.labels.read,
            fit_budget=self.raw,
            phase_by_origin={
                SUPPORT_A_TOKEN: "support_a",
                SUPPORT_B_TOKEN: "support_b",
            },
        )
        self.executor = TrainingBlockScopeExecutor(
            rows=cell.rows,
            round_name=ROUND_NAME,
            evaluate_fn=self._counted_evaluate,
            max_modified_fraction=MAX_MODIFIED_FRACTION,
        )

    def _counted_evaluate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        self.logical.spend(1)
        origin = int(kwargs.get("origin"))
        face = {
            SUPPORT_A_TOKEN: "support_a",
            SUPPORT_B_TOKEN: "support_b",
        }.get(origin)
        if face is None:
            raise P1AnomalyBlocked("logical evaluation used an unknown Support token")
        self.logical_by_face[face] += 1
        return self.adapter(*args, **kwargs)

    @staticmethod
    def _origin(face: str) -> int:
        if face == "support_a":
            return SUPPORT_A_TOKEN
        if face == "support_b":
            return SUPPORT_B_TOKEN
        raise KeyError("unknown AD Support face: %s" % face)

    def direct(self, face: str, program: str) -> dict[str, Any]:
        compiled = None if program == "identity" else self.executor._compiled(_steps(program))
        before = len(self.adapter.calls)
        raw = self._counted_evaluate(
            self.cell.roster(),
            self.cell.values,
            compiled,
            {},
            origin=self._origin(face),
        )
        if len(self.adapter.calls) != before + 1:
            raise P1AnomalyBlocked("AD adapter call accounting is inconsistent")
        return _reading(raw, self.adapter.calls[-1])

    def prime_identity(self, identity: Mapping[str, Mapping[str, Any]]) -> None:
        """Install a precomputed baseline while retaining raw-fit truth."""
        for face in ("support_a", "support_b"):
            origin = self._origin(face)
            raw = self.adapter(
                self.cell.roster(), self.cell.values, None, {}, origin=origin
            )
            expected = identity[face]
            if not math.isclose(float(raw["ad_macro_f1"]), float(expected["event_f1"]), abs_tol=1e-12):
                raise P1AnomalyBlocked("identity baseline changed across deterministic sessions")
            self.executor._baseline_cache[origin] = float(raw["mean_smase"])
            self.executor._per_view_cache[origin] = [
                float(value) for value in raw["per_view_smase"]
            ]


def _reading(raw: Mapping[str, Any], call: Mapping[str, Any]) -> dict[str, Any]:
    per_series = [-float(value) for value in raw["per_view_smase"]]
    event_f1 = float(raw["ad_macro_f1"])
    if not math.isfinite(event_f1) or not per_series or not all(map(math.isfinite, per_series)):
        raise P1AnomalyBlocked("IForest returned an empty or non-finite Event-F1")
    return {
        "event_f1": event_f1,
        "utility": event_f1,
        "pooled_event_f1": float(raw["ad_pooled_f1"]),
        "per_series_event_f1": per_series,
        "behavior_point_count": int(raw.get("behavior_point_count") or 0),
        "raw_series_fits": int(call["raw_consumer_fits"]),
        "model_cache_hits": int(call["model_cache_hits"]),
    }


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
    tokens = int(input_tokens) + int(output_tokens)
    return {
        "full_support_evaluations": int(full),
        "raw_consumer_fits": int(raw_fits),
        "raw_series_fits": int(raw_fits),
        "raw_fit_policy": "reported_per_series_without_B4_cap",
        "cheap_probes": int(cheap_probes),
        "llm_calls": int(llm_calls),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "tokens": tokens,
        "accepted_updates": int(updates),
        "wall_seconds": round(float(wall_seconds), 3),
        "within_caps": (
            int(full) <= B_MAIN
            and int(cheap_probes) <= MAX_PROBES
            and int(llm_calls) <= MAX_LLM_CALLS
            and tokens <= MAX_TOKENS
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


def _verification_executor(cell: ADCell) -> TrainingBlockScopeExecutor:
    def unavailable(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        raise RuntimeError("contract-only executor cannot consume an outcome")

    return TrainingBlockScopeExecutor(
        rows=cell.rows,
        round_name=ROUND_NAME,
        evaluate_fn=unavailable,
        max_modified_fraction=MAX_MODIFIED_FRACTION,
    )


def _common_dsl_contract(cell: ADCell) -> dict[str, Any]:
    eligible = _eligible_programs()
    executor = _verification_executor(cell)
    rows = []
    compile_failures: list[str] = []
    for op in eligible:
        try:
            executor._compiled(_steps(op))
            compile_status = "PASS"
        except Exception as exc:  # pragma: no cover - failure is reported
            compile_status = "FAIL"
            compile_failures.append("%s:%s" % (op, type(exc).__name__))
        verification = executor.verify(_steps(op), SUPPORT_A_TOKEN)
        rows.append({
            "program": op,
            "compile": compile_status,
            "verifier": "PASS" if verification.passed else "SAFE_REJECT",
            "checked_windows": int(verification.checked_windows),
            "modified_windows": int(verification.modified_windows),
            "rejection_codes": sorted({
                str(row["rejection_code"])
                for row in verification.rejected_windows
            }),
        })
    mandatory = {program for _name, program in FIXED_PROGRAMS}
    missing = sorted(mandatory - set(eligible))
    verifier_passed = {
        str(row["program"]) for row in rows if row["verifier"] == "PASS"
    }
    mandatory_verifier_failures = sorted(mandatory - verifier_passed)
    inventory_exact = eligible == EXPECTED_AD_OPERATORS
    status = "PASS" if (
        inventory_exact
        and not missing
        and not compile_failures
        and not mandatory_verifier_failures
    ) else "FAIL"
    return {
        "status": status,
        "inventory_policy": "current AD-legal global single-step workflows",
        "identity_available": True,
        "effect_distinct_inventory_count_from_p0b": 11,
        "eligible_operator_count": len(eligible),
        "eligible_operator_inventory_exact": inventory_exact,
        "consumer_evaluations": 0,
        "maximum_modified_fraction": MAX_MODIFIED_FRACTION,
        "fit_policy_extension": False,
        "contract_overhead": {
            "candidate_verifier_requests": len(rows),
            "verified_windows": sum(int(row["checked_windows"]) for row in rows),
            "charged_to_method_cell_b4": False,
        },
        "mandatory_fixed_programs_not_executable": missing,
        "mandatory_fixed_programs_not_verifier_approved": mandatory_verifier_failures,
        "compile_failures": compile_failures,
        "effect_aliases": dict(EFFECT_ALIASES),
        "rows": rows,
    }


def _select_best_fixed(cell: ADCell) -> BestFixedFreeze:
    started = time.time()
    syntactic = tuple(dict.fromkeys(("identity", *_eligible_programs())))
    programs = tuple(program for program in syntactic if program not in EFFECT_ALIASES)
    verifier = _verification_executor(cell)
    passed = ["identity"]
    safe_rejected: list[str] = []
    verified_windows = 0
    for program in programs:
        if program == "identity":
            continue
        verification = verifier.verify(_steps(program), SUPPORT_A_TOKEN)
        verified_windows += int(verification.checked_windows)
        if verification.passed:
            passed.append(program)
        else:
            safe_rejected.append(program)
    covered = set(passed) | set(safe_rejected) | set(EFFECT_ALIASES)
    if covered != set(syntactic):
        raise P1AnomalyBlocked("Best Fixed program-space accounting is incomplete")

    session = _EvaluationSession(cell, logical_cap=len(passed))
    readings = {program: session.direct("support_a", program) for program in passed}
    losses = [
        ProgramLoss(
            "support_a",
            cell.cell_id,
            program,
            uid,
            -float(score),
        )
        for program in passed
        for uid, score in zip(
            cell.uids,
            readings[program]["per_series_event_f1"],
            strict=True,
        )
    ]
    winner = select_best_fixed(losses)
    return BestFixedFreeze(
        program_id=str(winner.program_id),
        syntactic_programs=syntactic,
        evaluated_programs=tuple(passed),
        safe_rejected_programs=tuple(safe_rejected),
        selection_uids=cell.uids,
        logical_evaluations=session.logical.used,
        raw_series_fits=session.raw.used,
        verifier_requests=len(programs) - 1,
        verified_windows=verified_windows,
        wall_seconds=round(time.time() - started, 3),
    )


def _two_face_method(
    cell: ADCell,
    name: str,
    program: str,
    identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    session = _EvaluationSession(cell, logical_cap=B_MAIN)
    readings = {
        face: session.direct(face, program)
        for face in ("support_a", "support_b")
    }
    for face, reading in readings.items():
        reading["delta_u_vs_identity"] = (
            float(reading["utility"]) - float(identity[face]["utility"])
        )
    return _method_row(
        name,
        status="PASS",
        selected=program,
        readings=readings,
        usage=_usage(
            full=session.logical.used,
            raw_fits=session.raw.used,
            wall_seconds=time.time() - started,
        ),
        implementation="frozen one-step Common-DSL program with Event-F1 IForest",
        details={
            "same_adapter_support_a_b": True,
            "full_support_evaluations_by_face": dict(session.logical_by_face),
            "support_a_raw_series_fits": int(session.adapter.calls[0]["raw_consumer_fits"]),
            "support_b_raw_series_fits": int(session.adapter.calls[1]["raw_consumer_fits"]),
            "raw_consumer_fits_by_face": {
                "support_a": int(session.adapter.calls[0]["raw_consumer_fits"]),
                "support_b": int(session.adapter.calls[1]["raw_consumer_fits"]),
            },
        },
    )


def _deterministic_methods(
    cell: ADCell,
    best_fixed: BestFixedFreeze,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    identity_session = _EvaluationSession(cell, logical_cap=B_MAIN)
    identity = {
        face: identity_session.direct(face, "identity")
        for face in ("support_a", "support_b")
    }
    identity_row = _method_row(
        "Identity",
        status="PASS",
        selected="identity",
        readings=identity,
        usage=_usage(full=identity_session.logical.used, raw_fits=identity_session.raw.used),
        implementation="frozen IForest on unchanged synthetic training blocks",
        details={
            "elementwise_unchanged": True,
            "same_adapter_support_a_b": True,
            "full_support_evaluations_by_face": dict(identity_session.logical_by_face),
            "support_a_raw_series_fits": int(identity_session.adapter.calls[0]["raw_consumer_fits"]),
            "support_b_raw_series_fits": int(identity_session.adapter.calls[1]["raw_consumer_fits"]),
            "raw_consumer_fits_by_face": {
                "support_a": int(identity_session.adapter.calls[0]["raw_consumer_fits"]),
                "support_b": int(identity_session.adapter.calls[1]["raw_consumer_fits"]),
            },
        },
    )
    rows = [identity_row]

    target_session = _EvaluationSession(cell, logical_cap=B_MAIN)
    target_readings = {
        face: target_session.direct(face, best_fixed.program_id)
        for face in ("support_a", "support_b")
    }
    for face, reading in target_readings.items():
        reading["delta_u_vs_identity"] = (
            float(reading["utility"]) - float(identity[face]["utility"])
        )
    rows.append(_method_row(
        "Best Fixed Per-task",
        status="PASS",
        selected=best_fixed.program_id,
        readings={"target": target_readings},
        usage=_usage(full=target_session.logical.used, raw_fits=target_session.raw.used),
        implementation="production select_best_fixed frozen on the disjoint synthetic selection fixture",
        details={
            "formal_evolution_winner_frozen": True,
            "selection_uses_target_support": False,
            "selection_disjoint_from_target": bool(
                set(best_fixed.selection_uids).isdisjoint(cell.uids)
            ),
            "program_space_coverage_complete": bool(
                set(best_fixed.evaluated_programs)
                | set(best_fixed.safe_rejected_programs)
                | set(EFFECT_ALIASES)
                == set(best_fixed.syntactic_programs)
            ),
            "selection_candidate_count": len(best_fixed.evaluated_programs),
            "selection_programs": list(best_fixed.evaluated_programs),
            "safe_rejected_programs": list(best_fixed.safe_rejected_programs),
            "effect_aliases": dict(EFFECT_ALIASES),
            "same_adapter_support_a_b": True,
            "full_support_evaluations_by_face": dict(target_session.logical_by_face),
            "raw_consumer_fits_by_face": {
                "support_a": sum(
                    int(row["raw_consumer_fits"])
                    for row in target_session.adapter.calls
                    if row["phase"] == "support_a"
                ),
                "support_b": sum(
                    int(row["raw_consumer_fits"])
                    for row in target_session.adapter.calls
                    if row["phase"] == "support_b"
                ),
            },
            "cost_by_phase": {
                "evolution_selection": {
                    "full_support_evaluations": best_fixed.logical_evaluations,
                    "raw_consumer_fits": best_fixed.raw_series_fits,
                    "raw_series_fits": best_fixed.raw_series_fits,
                    "candidate_verifier_requests": best_fixed.verifier_requests,
                    "verified_windows": best_fixed.verified_windows,
                    "wall_seconds": best_fixed.wall_seconds,
                    "charged_to_target_b4": False,
                },
                "target_frozen_diagnostic": {
                    "full_support_evaluations": target_session.logical.used,
                    "raw_consumer_fits": target_session.raw.used,
                    "raw_series_fits": target_session.raw.used,
                    "charged_to_target_b4": True,
                },
            },
        },
    ))

    for name, program in FIXED_PROGRAMS:
        rows.append(_two_face_method(cell, name, program, identity))

    started = time.time()
    parallel = _EvaluationSession(cell, logical_cap=B_MAIN)
    pool = ("impute_linear", "hampel_filter", "winsorize")
    support = {program: parallel.direct("support_a", program) for program in pool}
    selected = max(pool, key=lambda program: (support[program]["event_f1"], program))
    delayed = parallel.direct("support_b", selected)
    rows.append(_method_row(
        "Parallel Best-of-N@4",
        status="PASS",
        selected=selected,
        readings={"support_a_candidates": support, "support_b": delayed},
        usage=_usage(
            full=parallel.logical.used,
            raw_fits=parallel.raw.used,
            wall_seconds=time.time() - started,
        ),
        implementation="three independent Support-A candidates and one frozen Support-B winner",
        details={
            "candidate_count": len(pool),
            "same_adapter_support_a_b": True,
            "full_support_evaluations_by_face": dict(parallel.logical_by_face),
            "support_b_raw_series_fits": int(parallel.adapter.calls[-1]["raw_consumer_fits"]),
            "raw_consumer_fits_by_face": _raw_fits_by_face(parallel),
            "performance_claim": False,
        },
    ))

    started = time.time()
    sequential = _EvaluationSession(cell, logical_cap=B_MAIN)
    first = "winsorize"
    first_reading = sequential.direct("support_a", first)
    first_delta = float(first_reading["utility"]) - float(identity["support_a"]["utility"])
    second = "hampel_filter" if first_delta < 0.0 else "outlier_iqr"
    second_reading = sequential.direct("support_a", second)
    second_delta = float(second_reading["utility"]) - float(identity["support_a"]["utility"])
    selected = second if second_delta >= first_delta else first
    delayed = sequential.direct("support_b", selected)
    rows.append(_method_row(
        "Sequential Refinement@4",
        status="PASS",
        selected=selected,
        readings={
            "step_1": {"program": first, **first_reading, "delta_u_vs_identity": first_delta},
            "step_2": {"program": second, **second_reading, "delta_u_vs_identity": second_delta},
            "support_b": delayed,
        },
        usage=_usage(
            full=sequential.logical.used,
            raw_fits=sequential.raw.used,
            wall_seconds=time.time() - started,
        ),
        implementation="feedback-conditioned second single-step proposal",
        details={
            "step_2_received_step_1_feedback": True,
            "same_adapter_support_a_b": True,
            "full_support_evaluations_by_face": dict(sequential.logical_by_face),
            "support_b_raw_series_fits": int(sequential.adapter.calls[-1]["raw_consumer_fits"]),
            "raw_consumer_fits_by_face": _raw_fits_by_face(sequential),
            "performance_claim": False,
        },
    ))
    return rows, identity


class _FaceDispatcher:
    """Dispatch real Harness receipts and deduplicate repeated delayed reads."""

    def __init__(self, executor: TrainingBlockScopeExecutor) -> None:
        self._executor = executor
        self._cache: dict[tuple[int, tuple[Any, ...]], Any] = {}
        self._requests = {"support_a": 0, "support_b": 0}
        self._unique = {"support_a": 0, "support_b": 0}
        self._cache_hits = {"support_a": 0, "support_b": 0}
        self._verified = {"support_a": 0, "support_b": 0}

    @staticmethod
    def _key(steps: Any) -> tuple[Any, ...]:
        return tuple(
            (str(op), json.dumps(_plain(dict(params)), sort_keys=True, separators=(",", ":")))
            for op, params in (steps or ())
        )

    def evaluate(self, steps: Any, origin: int) -> Any:
        token = int(origin)
        labels = {SUPPORT_A_TOKEN: "support_a", SUPPORT_B_TOKEN: "support_b"}
        if token not in labels:
            raise P1AnomalyBlocked("unknown synthetic AD Support token")
        face = labels[token]
        self._requests[face] += 1
        key = (token, self._key(steps))
        if key in self._cache:
            self._cache_hits[face] += 1
            return self._cache[key]
        receipt = self._executor.evaluate(tuple(steps), token)
        self._cache[key] = receipt
        self._unique[face] += 1
        self._verified[face] += int(receipt.verification.checked_windows)
        return receipt

    def accounting(self) -> dict[str, Any]:
        return {
            "requests_by_face": dict(self._requests),
            "unique_receipt_requests_by_face": dict(self._unique),
            "cache_hits_by_face": dict(self._cache_hits),
            "duplicate_requests": sum(self._cache_hits.values()),
            "unique_candidate_verifier_requests": sum(self._unique.values()),
            "verified_windows_by_face": dict(self._verified),
        }


def _backend_usage(backend: Any) -> tuple[int, int, int]:
    shared = getattr(backend, "_shared", None)
    return (
        int(getattr(backend, "calls", 0) or 0),
        int(getattr(shared, "prompt_tokens", 0) or 0),
        int(getattr(shared, "completion_tokens", 0) or 0),
    )


def _new_agent(block: np.ndarray, backend: Any) -> TTHAFastAgent:
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(block, task_kind=TASK),
    )
    return TTHAFastAgent(core)


def _new_method_state(
    *, snapshot: Any, cell: ADCell, backend: Any, root: Path, tag: str,
) -> dict[str, Any]:
    return four_arms._new_state(
        snapshot=snapshot,
        agent=_new_agent(cell.observation_block, backend.new_arm_backend()),
        store_root=root,
        tag=tag,
    )


def _task_contract() -> tuple[Any, Any]:
    eligible = set(_eligible_programs())
    forbidden = tuple(sorted(set(OPERATOR_NAMES) - eligible))
    spec = anomaly_task_spec_v1(
        downstream_model_class=CONSUMER_ID,
        metric=MetricSpec(PRIMARY_METRIC, "higher_is_better"),
        forbidden_modifications=forbidden,
    )
    return spec, anomaly_task_context_v1(
        task_spec=spec,
        deployment_constraints=deployment_constraints_v1(
            constraint_id="ad-p1-controlled-iforest-v1",
            fixed_downstream_model_id="fixed:aegists_iforest_v1",
            maximum_candidates=B_MAIN,
            maximum_modified_fraction=MAX_MODIFIED_FRACTION,
        ),
    )


def _request(cell: ADCell, spec: Any, context: Any) -> tuple[Any, dict[str, Any]]:
    observed = dict(resolver.window_context(cell.values, SUPPORT_A_TOKEN, PERIOD))
    observed["bound_period"] = float(PERIOD)
    features = dict(extract_public_features(cell.observation_block, task_kind=TASK))
    return PreparationRequest(
        "ad-p1-controlled-synthetic",
        cell.observation_block,
        spec,
        observed,
        task_context=context,
    ), features


def _raw_fits_by_face(session: _EvaluationSession) -> dict[str, int]:
    return {
        face: sum(
            int(row["raw_consumer_fits"])
            for row in session.adapter.calls
            if row["phase"] == face
        )
        for face in ("support_a", "support_b")
    }


def _frozen_h0(
    *,
    cell: ADCell,
    snapshot: Any,
    backend: Any,
    root: Path,
    spec: Any,
    context: Any,
    identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    before_calls, before_in, before_out = _backend_usage(backend)
    state = _new_method_state(
        snapshot=snapshot, cell=cell, backend=backend, root=root, tag="ad_frozen_h0"
    )
    request, _features = _request(cell, spec, context)
    state["method"].bind_round_data(cell.observation_block, task_kind=TASK)
    state["method"].prepare(request, runtime_prior_slot=False, pool_mode="full")
    trace = state["method"].last_trace
    chosen = str(trace.chosen_candidate_id or "identity")
    chosen_steps = tuple((trace.candidate_program_steps or {}).get(chosen) or ())
    ops = [str(op) for op, _params_value in chosen_steps]
    errors = []
    if len(ops) > 1:
        errors.append("multi_step_program")
    if set(ops) - set(_eligible_programs()):
        errors.append("task_mismatch_execution")
    selected = ops[0] if ops else "identity"
    session = _EvaluationSession(cell, logical_cap=B_MAIN)
    reading = session.direct("support_a", selected)
    reading["delta_u_vs_identity"] = (
        float(reading["utility"]) - float(identity["support_a"]["utility"])
    )
    after_calls, after_in, after_out = _backend_usage(backend)
    fast_checks = forecast_p1._fast_verifier_requests(trace)
    usage = _usage(
        full=session.logical.used,
        raw_fits=session.raw.used,
        cheap_probes=fast_checks,
        llm_calls=after_calls - before_calls,
        input_tokens=after_in - before_in,
        output_tokens=after_out - before_out,
        wall_seconds=time.time() - started,
    )
    if not usage["within_caps"]:
        errors.append("budget_cap_exceeded")
    return _method_row(
        "Frozen H0",
        status="PASS" if not errors else "FAIL",
        selected=selected,
        readings={"support_a": reading},
        usage=usage,
        implementation="production H0 Fast prepare; no Support adaptation or writeback",
        details={
            "candidate_count": len(trace.candidate_ids or ()),
            "fast_candidate_verifier_requests": fast_checks,
            "full_support_evaluations_by_face": dict(session.logical_by_face),
            "raw_consumer_fits_by_face": _raw_fits_by_face(session),
            "writeback": False,
            "protocol_errors": errors,
        },
    )


def _static_method(
    cell: ADCell, identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    session = _EvaluationSession(cell, logical_cap=B_MAIN)
    readings = {
        face: session.direct(face, "identity")
        for face in ("support_a", "support_b")
    }
    identity_equal = all(
        math.isclose(
            float(readings[face]["event_f1"]),
            float(identity[face]["event_f1"]),
            abs_tol=1e-12,
        )
        and readings[face]["per_series_event_f1"]
        == identity[face]["per_series_event_f1"]
        for face in ("support_a", "support_b")
    )
    for reading in readings.values():
        reading["delta_u_vs_identity"] = 0.0
    return _method_row(
        "Static",
        status="PASS",
        selected="identity",
        readings=readings,
        usage=_usage(
            full=session.logical.used,
            raw_fits=session.raw.used,
            wall_seconds=time.time() - started,
        ),
        implementation="independent zero-lifecycle Static baseline",
        details={
            "prepare_calls": 0,
            "episode_writes": 0,
            "delayed_open_calls": 0,
            "accepted_updates": 0,
            "writeback_attempts": 0,
            "store_created": False,
            "identity_readings_equal": identity_equal,
            "full_support_evaluations_by_face": dict(session.logical_by_face),
            "raw_consumer_fits_by_face": _raw_fits_by_face(session),
            "protocol_errors": [],
        },
    )


def _harness_method(
    *,
    name: str,
    snapshot: Any,
    cell: ADCell,
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
        tag="ad_" + name.lower().replace("-", "_").replace(" ", "_"),
    )
    before_view = forecast_p1._snapshot_state_view(state["method"]._active_snapshot())
    request, features = _request(cell, spec, context)
    session = _EvaluationSession(cell, logical_cap=B_MAIN)
    session.prime_identity(identity)
    dispatcher = _FaceDispatcher(session.executor)
    result = run_online_round(
        state["method"],
        dispatcher,
        request,
        cell.values,
        origin=SUPPORT_A_TOKEN,
        slow_agent=None,
        controller=state["controller"],
        store=state["store"],
        card_builder=forecast_course._card_builder,
        round_name="ad_p1_" + name.lower().replace("-", "_").replace(" ", "_"),
        budget=2,
        allow_slow=False,
        horizon=1,
        period=PERIOD,
        domain="ad_p1_controlled_synthetic",
        fast_features=features,
        allow_fast_skill=True,
        runtime_prior_slot=False,
        pool_mode="full",
    )
    open_delayed(
        result,
        dispatcher,
        delayed_origin=SUPPORT_B_TOKEN,
        store=state["store"],
    )
    activated = False
    if writeback and result.approved_skill_id is not None:
        activated = activate_approved(result, state["store"])
    after_view = forecast_p1._snapshot_state_view(state["method"]._active_snapshot())
    changed = before_view != after_view
    trace = state["method"].last_trace
    chosen = str(trace.chosen_candidate_id or "identity")
    chosen_steps = tuple((trace.candidate_program_steps or {}).get(chosen) or ())
    chosen_ops = [str(op) for op, _params_value in chosen_steps]
    deployed_ops = [
        str(step["op"])
        for step in (result.winner_program or ())
        if isinstance(step, Mapping) and step.get("op")
    ]
    after_calls, after_in, after_out = _backend_usage(backend)
    receipts = dispatcher.accounting()
    fast_checks = forecast_p1._fast_verifier_requests(trace)
    retained = bool(writeback and activated and changed)
    approved_after_support_b = bool(
        result.approved_skill_id is not None
        and int(receipts["unique_receipt_requests_by_face"]["support_b"]) >= 1
    )
    usage = _usage(
        full=session.logical.used,
        raw_fits=session.raw.used,
        cheap_probes=fast_checks + int(receipts["unique_candidate_verifier_requests"]),
        llm_calls=after_calls - before_calls,
        input_tokens=after_in - before_in,
        output_tokens=after_out - before_out,
        updates=int(retained),
        wall_seconds=time.time() - started,
    )

    episodes = [
        {"source_skill_id": getattr(episode, "source_skill_id", None)}
        for episode, _program in (result._episodes or ())
    ]
    scope_by_skill = four_arms._scope_match_by_skill_id(
        {str(skill.skill_id): skill for skill in snapshot.skills}, features
    )
    wrong_faces = forecast_course._g2_faces({
        "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
        "scope_match_by_skill_id": scope_by_skill,
        "episodes": episodes,
        "pool": list(trace.candidate_ids or ()),
    })
    wrong_faces["episode"] = sum(
        str(row.get("source_skill_id") or "") == forecast_course.CLS_SKILL_ID
        for row in episodes
    )
    wrong_faces["support_probe"] = sum(
        source_skill_of_candidate(row.get("candidate_id")) == forecast_course.CLS_SKILL_ID
        for row in (result.actual_probed_programs or ())
    )
    errors = []
    if len(chosen_ops) > 1:
        errors.append("multi_step_program")
    if set(chosen_ops) - set(_eligible_programs()):
        errors.append("task_mismatch_execution")
    if any(int(value) for value in wrong_faces.values()):
        errors.append("cross_task_skill_leakage")
    if not usage["within_caps"]:
        errors.append("budget_cap_exceeded")
    support_delta = next(
        (
            float(row["gain"])
            for row in (result.actual_probed_programs or ())
            if row.get("candidate_id") == result._winner_candidate_id
            and row.get("gain") is not None
        ),
        None,
    )
    delayed_delta = (
        float(result.delayed_utility)
        if result.delayed_utility is not None
        else None
    )
    return _method_row(
        name,
        status="PASS" if not errors else "FAIL",
        selected="identity" if not deployed_ops else "+".join(deployed_ops),
        readings={
            "support_receipts": int(result.target_support_receipts_used),
            "support_delta_u_vs_identity": support_delta,
            "support_candidate_utility": (
                float(identity["support_a"]["utility"]) + support_delta
                if support_delta is not None else None
            ),
            "delayed_delta_u_vs_identity": delayed_delta,
            "delayed_candidate_utility": (
                float(identity["support_b"]["utility"]) + delayed_delta
                if delayed_delta is not None else None
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
            "writeback_persisted_to_evolution_store": False,
            "retained_update": retained,
            "approved_after_support_b": approved_after_support_b,
            "writeback_treatment": "RETAINED_NEW_STATE" if retained else "NOT_EXERCISED",
            "historical_skill_treatment": "NOT_EXERCISED",
            "eligible_ad_history_skill_ids": [],
            "wrong_task_faces": wrong_faces,
            "wrong_task_fail_closed": not any(int(value) for value in wrong_faces.values()),
            "candidate_count": len(trace.candidate_ids or ()),
            "chosen_program_steps": [
                {"op": str(op), "params": dict(params)}
                for op, params in chosen_steps
            ],
            "method_state_changed_inside_isolated_unit": changed,
            "fast_candidate_verifier_requests": fast_checks,
            "receipt_accounting": receipts,
            "identity_baseline_cache_reused": True,
            "full_support_evaluations_by_face": dict(session.logical_by_face),
            "raw_series_fits_by_face": _raw_fits_by_face(session),
            "raw_consumer_fits_by_face": _raw_fits_by_face(session),
            "protocol_errors": errors,
        },
    )


def _harness_methods(
    cell: ADCell,
    identity: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec, context = _task_contract()
    backend = shared_harness._scripted_backend(GLOBAL_SCRIPTED_CALL_CAP)
    temp_root = Path(tempfile.mkdtemp(prefix="ad_p1_"))
    try:
        h0 = forecast_course._h0()
        wrong_task_card = forecast_course._cls_card()
        shared_origin = forecast_course._install(
            h0,
            wrong_task_card,
            store_root=temp_root / "initial",
            tag="ad_k0_a5_wrong_task",
        )
        h0_ids = sorted(str(skill.skill_id) for skill in h0.skills)
        shared_ids = sorted(str(skill.skill_id) for skill in shared_origin.skills)
        rows = [
            _frozen_h0(
                cell=cell,
                snapshot=h0,
                backend=backend,
                root=temp_root,
                spec=spec,
                context=context,
                identity=identity,
            ),
            _static_method(cell, identity),
            _harness_method(
                name="A3-reset",
                snapshot=h0,
                cell=cell,
                backend=backend,
                root=temp_root,
                spec=spec,
                context=context,
                identity=identity,
                initial_skill_ids=h0_ids,
                writeback=False,
            ),
            _harness_method(
                name="K0-fixed",
                snapshot=shared_origin,
                cell=cell,
                backend=backend,
                root=temp_root,
                spec=spec,
                context=context,
                identity=identity,
                initial_skill_ids=shared_ids,
                writeback=False,
            ),
            _harness_method(
                name="A5-online",
                snapshot=shared_origin,
                cell=cell,
                backend=backend,
                root=temp_root,
                spec=spec,
                context=context,
                identity=identity,
                initial_skill_ids=shared_ids,
                writeback=True,
            ),
        ]
        calls, input_tokens, output_tokens = _backend_usage(backend)
        return rows, {
            "mode": "scripted",
            "production_format_exercised": True,
            "production_lifecycle_exercised": True,
            "production_ttha_method_exercised": True,
            "production_run_online_round_exercised": True,
            "live_transport_exercised": False,
            "global_llm_calls": calls,
            "global_input_tokens": input_tokens,
            "global_output_tokens": output_tokens,
            "k0_a5_same_initial_state": True,
            "k0_a5_initial_skill_ids": shared_ids,
            "qualified_ad_history_skill_ids": [],
            "wrong_task_skill_id": forecast_course.CLS_SKILL_ID,
            "temporary_store_removed_after_run": True,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def run_anomaly_component() -> dict[str, Any]:
    """Execute the bounded AD P1 contract smoke and return an in-memory payload."""
    target, selection, data = _synthetic_fixtures()
    contract = _common_dsl_contract(target)
    # Freeze on the disjoint fixture before any Target event-label read.
    best_fixed = _select_best_fixed(selection)
    deterministic, identity = _deterministic_methods(target, best_fixed)
    harness_rows, backend = _harness_methods(target, identity)
    rows = deterministic + harness_rows
    by_name = {str(row["method"]): row for row in rows}
    ordered = [by_name[name] for name in MANDATORY_METHODS if name in by_name]
    missing = sorted(set(MANDATORY_METHODS) - set(by_name))
    duplicates = len(rows) != len(by_name)
    failed = [row["method"] for row in ordered if row["status"] != "PASS"]
    over_budget = [
        row["method"] for row in ordered if not bool(row["usage"]["within_caps"])
    ]
    blocking = []
    if contract["status"] != "PASS":
        blocking.append("common_dsl_contract")
    if missing or duplicates:
        blocking.append("method_inventory")
    if failed:
        blocking.append("method_contract")
    if over_budget:
        blocking.append("logical_budget")
    protocol_errors = {
        "natural_final_outcome_reads": 0,
        "development_query_evaluations": 0,
        "task_mismatch_execution": sum(
            "task_mismatch_execution" in tuple(row["details"].get("protocol_errors") or ())
            for row in ordered
        ),
        "cross_task_skill_leakage": sum(
            "cross_task_skill_leakage" in tuple(row["details"].get("protocol_errors") or ())
            for row in ordered
        ),
        "historical_skill_bypassed_support": 0,
        "wrong_promotion": 0,
    }
    if any(protocol_errors.values()):
        blocking.append("protocol_error")
    status = "PASS" if not blocking else "FAIL"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "P1_COMMON_DSL_AND_CORE_BASELINE_SMOKE",
        "task_tranche": TASK_TRANCHE,
        "evidence_grade": EVIDENCE_GRADE,
        "status": status,
        "component_pass": status == "PASS",
        "ad_component_pass": status == "PASS",
        "verdict": "AD_P1_CONTRACT_SMOKE_PASS" if status == "PASS" else "AD_P1_CONTRACT_SMOKE_BLOCKED",
        "common_dsl_contract": contract,
        "methods": ordered,
        "method_inventory": {
            "required": list(MANDATORY_METHODS),
            "observed": [row["method"] for row in ordered],
            "missing": missing,
            "duplicate_names": duplicates,
        },
        "data": data,
        "consumer": {
            "id": CONSUMER_ID,
            "consumer_id": CONSUMER_ID,
            "primary_metric": PRIMARY_METRIC,
            "metric": PRIMARY_METRIC,
            "metric_direction": "higher_is_better",
            "same_adapter_used_for_support_a_b": True,
            "raw_fit_unit": "series",
        },
        "budget_caps": {
            "full_support_evaluations_per_method": B_MAIN,
            "cheap_probes_per_method": MAX_PROBES,
            "llm_calls_per_method": MAX_LLM_CALLS,
            "tokens_per_method": MAX_TOKENS,
            "accepted_updates_per_method": MAX_UPDATES,
            "raw_series_fits": "reported_separately_without_B4_cap",
        },
        "backend": backend,
        "protocol_errors": protocol_errors,
        "blocking_failures": blocking,
        "positive_control_boundary": {
            "p1_contract_smoke": status,
            "positive_control_44a": "NOT_PASSED",
            "ad_method_release": False,
            "p2_p3_method_gate_released": False,
        },
        "performance_or_headroom_claim": False,
        "treatment_or_capability_claim": False,
    }


def run_component() -> dict[str, Any]:
    """Stable integration alias for the unified P1 runner."""
    return run_anomaly_component()


def run(*, backend_mode: str = "scripted") -> dict[str, Any]:
    """Unified-runner entry point; P1 AD intentionally has no live mode."""
    if backend_mode != "scripted":
        raise ValueError("AD P1 contract smoke supports backend_mode='scripted' only")
    return run_anomaly_component()


__all__ = ["run", "run_anomaly_component", "run_component"]
