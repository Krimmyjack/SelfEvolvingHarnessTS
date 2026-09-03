"""Run the bounded Forecast P2 treatment-wiring pilot.

The pilot is intentionally a controlled mechanism witness.  It uses only the
already-exposed KDD development roster, two fixed outlier realizations, the
production TTHA lifecycle, and a supply-only control card.  It opens neither a
Query split nor any Natural-Final outcome and makes no performance or
capability claim.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
from SelfEvolvingHarnessTS.contracts.task import (
    MetricSpec,
    deployment_constraints_v1,
    forecast_task_context_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
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

from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms
from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_v1_sealed_a5_a3 as sealed
from evaluation.functional.task_episode_harness.injection import (
    inject_label_touched_corpus,
)
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
P1_REPORT = PROJECT_ROOT / "artifacts/main_protocol/p1_core_baseline_smoke_20260830.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/p2_forecast_single_flow_pilot_20260830.json"

PROTOCOL_VERSION = "v1.2.1-Core"
STAGE = "P2_FORECAST_SINGLE_FLOW_TREATMENT_WIRING"
EVIDENCE_GRADE = "MECHANISM"
TASK = "forecast"
CONSUMER_ID = "pooled_ridge_a1"
METRIC = "sMASE"
PERIOD = 24
HORIZON = 48
REAL_ORIGIN = 936
SUPPORT_TOKEN = REAL_ORIGIN
DELAYED_TOKEN = REAL_ORIGIN + 1
CONTROL_SKILL_ID = "p2_forecast_hampel_control_v1"
CONTROL_PROGRAM = "hampel_filter"
MATERIAL = 0.005

B_MAIN = 4
MAX_SUPPORT_A = 3
MAX_SUPPORT_B = 1
MAX_CHEAP_PROBES = 12
MAX_AGENT_STAGE_CALLS = 4
MAX_EXTERNAL_LLM_CALLS = 4
MAX_TOKENS = 40_000
MAX_UPDATES = 1
MAX_MODIFIED_FRACTION = 0.35

ARMS = ("Static", "A3-reset", "K0-fixed", "A5-online")
COURSE = (
    {
        "decision_index": 1,
        "unit_id": "forecast_p2_conflict",
        "seed": 7,
        "purpose": "positive_support_then_negative_delayed_revocation",
    },
    {
        "decision_index": 2,
        "unit_id": "forecast_p2_reencounter",
        "seed": 8,
        "purpose": "same_visible_observation_after_carried_revocation",
    },
)


class P2Blocked(RuntimeError):
    """A release or runtime condition prevented a truthful P2 verdict."""


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
        raise P2Blocked("required upstream P1 report is absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P2Blocked("upstream P1 report is not an object")
    return value


def _assert_p1_release() -> dict[str, Any]:
    payload = _read_object(P1_REPORT)
    checks = {
        "overall_p1_complete": payload.get("overall_p1_complete") is True,
        "release_p2": payload.get("release_p2") is True,
        "p2_not_precompleted": payload.get("p2_complete") is False,
        "live_outcome_closed": payload.get("live_outcome_release") is False,
        "query_closed": int(payload.get("development_query_evaluations", -1)) == 0,
        "natural_final_closed": int(payload.get("natural_final_outcome_reads", -1)) == 0,
        "no_blocking_failures": not list(payload.get("blocking_failures") or ()),
    }
    if not all(checks.values()):
        raise P2Blocked("P1 has not released the Forecast P2 pilot")
    return {
        "source": P1_REPORT.relative_to(PROJECT_ROOT).as_posix(),
        "verdict": str(payload.get("verdict") or ""),
        "checks": checks,
        "released_scope": "Forecast P2 single-flow pilot only",
    }


def _eligible_programs() -> tuple[str, ...]:
    return tuple(
        str(name)
        for name in OPERATOR_NAMES
        if TASK in tuple(OPERATOR_METADATA[name].get("allowed_tasks") or ())
        and not bool(OPERATOR_METADATA[name].get("shape_changing"))
        and not bool(OPERATOR_METADATA[name].get("changes_target_space"))
        and OPERATOR_METADATA[name].get("requires_dependency") != "statsmodels"
    )


def _task_contract() -> tuple[Any, Any]:
    eligible = set(_eligible_programs())
    spec = forecast_task_spec_v1(
        horizon=HORIZON,
        downstream_model_class=CONSUMER_ID,
        metric=MetricSpec(METRIC, "lower_is_better"),
        forbidden_modifications=tuple(sorted(set(OPERATOR_NAMES) - eligible)),
    )
    context = forecast_task_context_v1(
        task_spec=spec,
        deployment_constraints=deployment_constraints_v1(
            constraint_id="forecast-p2-controlled-witness-v1",
            fixed_downstream_model_id="fixed:pooled-ridge-a1",
            maximum_candidates=2,
            maximum_modified_fraction=MAX_MODIFIED_FRACTION,
        ),
    )
    return spec, context


def _config() -> dict[str, object]:
    config = dict(forecast_p1._config())
    config.update(
        {
            "dataset_id": "forecast_p2_exposed_kdd_controlled_witness",
            "support_origin": REAL_ORIGIN,
            "selection_origin": REAL_ORIGIN,
            "period": PERIOD,
        }
    )
    return config


def _controlled_card() -> dict[str, Any]:
    steps = [{"op": CONTROL_PROGRAM, "params": {}}]
    body = "\n".join(
        [
            "CONTROL: this is a bounded P2 mechanism card, not learned evidence.",
            "SUPPLY: offer one Hampel candidate for current Target verification.",
            "VERIFY: Target Support is mandatory; this card grants no deployment right.",
            "Frozen program steps: "
            + json.dumps(steps, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    return {
        "schema_version": "skill-entry/1",
        "skill_id": CONTROL_SKILL_ID,
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        "observable_applicability": {
            "all": [{"feature": "task_kind", "op": "==", "value": TASK}]
        },
        "allowed_tools": [],
        "risk_guards": {
            "controlled_mechanism_witness": True,
            "requires_target_support": True,
            "execution_right": "withheld_supplies_candidate_only",
            "authority": {
                "reorders_supplied_candidates": False,
                "supplies_candidates": True,
                "suppresses_operators": False,
                "grants_execution": False,
            },
            "scope_v1": {
                "task_kind": TASK,
                "consumer_id": CONSUMER_ID,
                "metric": METRIC,
                "program_geometry": [CONTROL_PROGRAM],
            },
        },
    }


def _make_cell(base: forecast_p1.ForecastCell, seed: int) -> forecast_p1.ForecastCell:
    values, _private = inject_label_touched_corpus(
        dict(base.values),
        faulty_series=tuple(base.support_b),
        clean_series=tuple(base.support_a),
        amplitude=8.0,
        count=40,
        seed=int(seed),
    )
    return forecast_p1.ForecastCell(
        values=values,
        support_a=base.support_a,
        support_b=base.support_b,
        observation_block=np.asarray(
            values[base.support_a[0]][:REAL_ORIGIN], dtype=np.float64
        ),
    )


class _FaceConsumer:
    def __init__(self, label: str) -> None:
        self.label = str(label)
        self.calls = 0

    def __call__(
        self,
        roster: Any,
        values: Any,
        compiled: Any,
        config: Any,
        *,
        origin: int,
    ) -> dict[str, Any]:
        self.calls += 1
        return forecast_p1.forecast_runtime._evaluate(
            roster, values, compiled, config, origin=origin
        )


class _OriginDispatcher:
    """Map public lifecycle tokens to distinct held-in faces and cache repeats."""

    def __init__(self, entries: Mapping[int, tuple[str, ScopeExecutor, int]]) -> None:
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
            raise P2Blocked("unknown P2 held-in face token")
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
        }


def _semantic_state(snapshot: Any) -> tuple[Any, ...]:
    return tuple(
        (
            str(skill.skill_id),
            int(skill.revision),
            str(getattr(skill.skill_kind, "value", skill.skill_kind)),
            str(skill.body),
            json.dumps(
                _plain(skill.observable_applicability),
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                _plain(skill.risk_guards), ensure_ascii=False, sort_keys=True
            ),
        )
        for skill in sorted(snapshot.skills, key=lambda item: str(item.skill_id))
    )


def _public_state(snapshot: Any) -> dict[str, Any]:
    skills = list(snapshot.skills)
    control = next(
        (skill for skill in skills if str(skill.skill_id) == CONTROL_SKILL_ID), None
    )
    if control is None:
        control_view = {
            "present": False,
            "revision": None,
            "task_scope": None,
            "requires_target_support": None,
            "supplies_candidates": None,
            "grants_execution": None,
        }
    else:
        guards = dict(control.risk_guards or {})
        authority = dict(guards.get("authority") or {})
        control_view = {
            "present": True,
            "revision": int(control.revision),
            "task_scope": (
                (guards.get("scope_v1") or {}).get("task_kind")
            ),
            "requires_target_support": guards.get("requires_target_support"),
            "supplies_candidates": authority.get("supplies_candidates"),
            "grants_execution": authority.get("grants_execution"),
        }
    return {
        "skill_count": len(skills),
        "skill_ids": sorted(str(skill.skill_id) for skill in skills),
        "controlled_card": control_view,
    }


def _new_agent(block: np.ndarray, backend: Any) -> TTHAFastAgent:
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(block, task_kind=TASK),
    )
    return TTHAFastAgent(core)


def _request(
    *, unit_id: str, cell: forecast_p1.ForecastCell, spec: Any, context: Any
) -> tuple[PreparationRequest, dict[str, Any]]:
    observed = dict(resolver.window_context(cell.values, REAL_ORIGIN, PERIOD))
    observed["bound_period"] = float(PERIOD)
    features = dict(extract_public_features(cell.observation_block, task_kind=TASK))
    return (
        PreparationRequest(
            unit_id,
            cell.observation_block,
            spec,
            observed,
            task_context=context,
        ),
        features,
    )


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


def _run_adaptive(
    *,
    unit: Mapping[str, Any],
    arm: str,
    cell: forecast_p1.ForecastCell,
    base_snapshot: Any,
    carried_episodes: Sequence[Any],
    temp_root: Path,
    spec: Any,
    context: Any,
) -> tuple[dict[str, Any], Any, tuple[Any, ...]]:
    backend = sealed.SealedProbeBackend(
        explore=False,
        operators=(),
        max_propose_candidates=1,
        force_pool=True,
        prefer_skill_in_select=True,
    )
    store_label = "%s/%s" % (unit["unit_id"], arm)
    state = four_arms._new_state(
        snapshot=base_snapshot,
        agent=_new_agent(cell.observation_block, backend),
        store_root=temp_root / str(unit["unit_id"]) / arm.lower().replace("-", "_"),
        tag="state",
        episodes=tuple(carried_episodes),
    )
    method = state["method"]
    before_snapshot = method._active_snapshot()
    before_semantics = _semantic_state(before_snapshot)
    before_public = _public_state(before_snapshot)
    episode_start = len(method.experience_episodes)
    request, features = _request(
        unit_id=str(unit["unit_id"]), cell=cell, spec=spec, context=context
    )

    support_consumer = _FaceConsumer("support_a")
    delayed_consumer = _FaceConsumer("support_b")
    support_executor = ScopeExecutor(
        cell.roster("support_a"),
        cell.values,
        _config(),
        evaluate_fn=support_consumer,
        max_modified_fraction=MAX_MODIFIED_FRACTION,
    )
    delayed_executor = ScopeExecutor(
        cell.roster("support_b"),
        cell.values,
        _config(),
        evaluate_fn=delayed_consumer,
        max_modified_fraction=MAX_MODIFIED_FRACTION,
    )
    dispatcher = _OriginDispatcher(
        {
            SUPPORT_TOKEN: ("support_a", support_executor, REAL_ORIGIN),
            DELAYED_TOKEN: ("support_b", delayed_executor, REAL_ORIGIN),
        }
    )
    started = time.time()
    result = run_online_round(
        method,
        dispatcher,
        request,
        cell.values,
        origin=SUPPORT_TOKEN,
        slow_agent=None,
        controller=state["controller"],
        store=state["store"],
        card_builder=forecast_course._card_builder,
        round_name="%s_%s" % (unit["unit_id"], arm.lower().replace("-", "_")),
        budget=1,
        allow_slow=False,
        horizon=HORIZON,
        period=PERIOD,
        domain=str(unit["unit_id"]),
        fast_features=features,
        allow_fast_skill=False,
        runtime_prior_slot=False,
        pool_mode="full",
    )
    support_relation = (
        str(result._episodes[-1][0].relation) if result._episodes else "NOT_EVALUATED"
    )
    open_delayed(
        result,
        dispatcher,
        delayed_origin=DELAYED_TOKEN,
        store=state["store"],
    )
    activation_result = activate_approved(result, state["store"])
    after_snapshot = method._active_snapshot()
    after_semantics = _semantic_state(after_snapshot)
    after_public = _public_state(after_snapshot)
    trace = method.last_trace
    if trace is None:
        raise P2Blocked("production Fast trace is absent")
    episodes = _episode_rows(method, episode_start)
    delayed_relation = episodes[-1]["relation"] if episodes else "NOT_EVALUATED"

    candidates = [str(value) for value in tuple(trace.candidate_ids or ())]
    source_candidates = [
        candidate
        for candidate in candidates
        if source_skill_of_candidate(candidate) == CONTROL_SKILL_ID
    ]
    source_probes = [
        dict(probe)
        for probe in (result.actual_probed_programs or ())
        if source_skill_of_candidate(probe.get("candidate_id")) == CONTROL_SKILL_ID
    ]
    autonomous = [
        candidate
        for candidate in candidates
        if candidate != "identity" and source_skill_of_candidate(candidate) is None
    ]
    entries_before = {
        str(skill.skill_id): skill for skill in before_snapshot.skills
    }
    scope_matches = four_arms._scope_match_by_skill_id(entries_before, features)
    accounting = dispatcher.accounting()
    support_a_used = int(result.target_support_receipts_used)
    support_b_used = int(accounting["unique_by_face"].get("support_b", 0))
    cheap_probes = int(
        forecast_p1._fast_verifier_requests(trace)
        + sum(int(value) for value in accounting["unique_by_face"].values())
    )
    update_accepted = bool(result.revoked_skill_id or activation_result)
    if result.revoked_skill_id:
        update_kind = "REVOKE_CONTROLLED_SUPPLY_CARD"
    elif activation_result:
        update_kind = "APPROVE_TARGET_LOCAL_UPDATE"
    else:
        update_kind = "NONE"
    usage = {
        "support_a_full_evaluations": support_a_used,
        "support_b_full_evaluations": support_b_used,
        "full_support_evaluations": support_a_used + support_b_used,
        "raw_consumer_fits": support_consumer.calls + delayed_consumer.calls,
        "raw_consumer_fits_by_face": {
            "support_a": support_consumer.calls,
            "support_b": delayed_consumer.calls,
        },
        "cheap_probes": cheap_probes,
        "agent_stage_calls": len(backend.requests),
        "external_llm_calls": 0,
        "tokens": 0,
        "accepted_updates": int(update_accepted),
        "wall_seconds": round(time.time() - started, 3),
    }
    usage["within_caps"] = bool(
        usage["support_a_full_evaluations"] <= MAX_SUPPORT_A
        and usage["support_b_full_evaluations"] <= MAX_SUPPORT_B
        and usage["full_support_evaluations"] <= B_MAIN
        and usage["cheap_probes"] <= MAX_CHEAP_PROBES
        and usage["agent_stage_calls"] <= MAX_AGENT_STAGE_CALLS
        and usage["external_llm_calls"] <= MAX_EXTERNAL_LLM_CALLS
        and usage["tokens"] <= MAX_TOKENS
        and usage["accepted_updates"] <= MAX_UPDATES
    )
    selected_steps = _plain(result.winner_program or [])
    record = {
        "decision_index": int(unit["decision_index"]),
        "unit_id": str(unit["unit_id"]),
        "seed": int(unit["seed"]),
        "arm": arm,
        "store_isolation_label": store_label,
        "state_before": before_public,
        "state_after": after_public,
        "semantic_state_changed": before_semantics != after_semantics,
        "episodes_at_start": episode_start,
        "episodes_written": episodes,
        "context": {
            "task_kind": TASK,
            "consumer_id": CONSUMER_ID,
            "metric": METRIC,
            "observation_points": int(cell.observation_block.size),
            "pattern": forecast_course._pattern_view(features),
        },
        "trace": {
            "retrieved_controlled_card": CONTROL_SKILL_ID
            in tuple(trace.retrieved_skill_ids or ()),
            "controlled_card_scope_match": bool(
                scope_matches.get(CONTROL_SKILL_ID, False)
            ),
            "controlled_supply_count": len(source_candidates),
            "controlled_probe_count": len(source_probes),
            "controlled_probe_gains": [probe.get("gain") for probe in source_probes],
            "controlled_card_chosen": source_skill_of_candidate(
                trace.chosen_candidate_id
            )
            == CONTROL_SKILL_ID,
            "controlled_card_deployed": result.deployed_skill_id
            == CONTROL_SKILL_ID,
            "controlled_card_revoked": result.revoked_skill_id
            == CONTROL_SKILL_ID,
            "autonomous_nonidentity_candidate_count": len(autonomous),
            "candidate_count": len(candidates),
            "abstained": bool(result.abstained),
            "selected_steps": selected_steps,
        },
        "support_a": {
            "face_id": "support_a_train_b_eval_a",
            "relation": support_relation,
            "gain": (
                float(source_probes[0]["gain"])
                if source_probes and source_probes[0].get("gain") is not None
                else None
            ),
        },
        "support_b": {
            "face_id": "support_b_train_a_eval_b",
            "relation": delayed_relation,
            "gain": (
                float(result.delayed_utility)
                if result.delayed_utility is not None
                else None
            ),
        },
        "update": {
            "kind": update_kind,
            "accepted": update_accepted,
            "production_revocation": result.revoked_skill_id == CONTROL_SKILL_ID,
            "activation_checked": True,
            "activation_succeeded": bool(activation_result),
            "retained_at_unit_boundary": arm == "A5-online" and update_accepted,
        },
        "boundary_action": (
            "CARRY_EVOLVED_STATE" if arm == "A5-online" else "DISCARD_UNIT_STATE"
        ),
        "usage": usage,
        "boundary_counts": {
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
            "traffic_loader_invocations": 0,
            "solar_loader_invocations": 0,
            "query_feedback_events": 0,
        },
        "lifecycle": {
            "method_entry": "TTHAMethod",
            "online_round": True,
            "delayed_gate": True,
            "activation_gate_checked": True,
        },
    }
    return record, after_snapshot, tuple(method.experience_episodes)


def _static_record(
    *, unit: Mapping[str, Any], cell: forecast_p1.ForecastCell, h0: Any
) -> dict[str, Any]:
    public = _public_state(h0)
    return {
        "decision_index": int(unit["decision_index"]),
        "unit_id": str(unit["unit_id"]),
        "seed": int(unit["seed"]),
        "arm": "Static",
        "store_isolation_label": None,
        "state_before": public,
        "state_after": public,
        "semantic_state_changed": False,
        "episodes_at_start": 0,
        "episodes_written": [],
        "context": {
            "task_kind": TASK,
            "consumer_id": CONSUMER_ID,
            "metric": METRIC,
            "observation_points": int(cell.observation_block.size),
            "pattern": forecast_course._pattern_view(
                extract_public_features(cell.observation_block, task_kind=TASK)
            ),
        },
        "trace": {
            "retrieved_controlled_card": False,
            "controlled_card_scope_match": False,
            "controlled_supply_count": 0,
            "controlled_probe_count": 0,
            "controlled_probe_gains": [],
            "controlled_card_chosen": False,
            "controlled_card_deployed": False,
            "controlled_card_revoked": False,
            "autonomous_nonidentity_candidate_count": 0,
            "candidate_count": 1,
            "abstained": True,
            "selected_steps": [],
        },
        "support_a": {
            "face_id": "support_a_train_b_eval_a",
            "relation": "NOT_EVALUATED",
            "gain": None,
        },
        "support_b": {
            "face_id": "support_b_train_a_eval_b",
            "relation": "NOT_EVALUATED",
            "gain": None,
        },
        "update": {
            "kind": "NONE",
            "accepted": False,
            "production_revocation": False,
            "activation_checked": False,
            "activation_succeeded": False,
            "retained_at_unit_boundary": False,
        },
        "boundary_action": "NO_LIFECYCLE_STATIC",
        "usage": {
            "support_a_full_evaluations": 0,
            "support_b_full_evaluations": 0,
            "full_support_evaluations": 0,
            "raw_consumer_fits": 0,
            "raw_consumer_fits_by_face": {"support_a": 0, "support_b": 0},
            "cheap_probes": 0,
            "agent_stage_calls": 0,
            "external_llm_calls": 0,
            "tokens": 0,
            "accepted_updates": 0,
            "wall_seconds": 0.0,
            "within_caps": True,
        },
        "boundary_counts": {
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
            "traffic_loader_invocations": 0,
            "solar_loader_invocations": 0,
            "query_feedback_events": 0,
        },
        "lifecycle": {
            "method_entry": "Static_identity_only",
            "online_round": False,
            "delayed_gate": False,
            "activation_gate_checked": False,
        },
    }


def derive_treatment_gate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the treatment result from lifecycle facts, fail closed."""
    reasons: list[str] = []
    by_key = {
        (int(row.get("decision_index", -1)), str(row.get("arm", ""))): row
        for row in runs
    }

    def row(index: int, arm: str) -> Mapping[str, Any]:
        value = by_key.get((index, arm))
        if value is None:
            reasons.append("missing_%s_at_decision_%d" % (arm, index))
            return {}
        return value

    u1_k0 = row(1, "K0-fixed")
    u1_a5 = row(1, "A5-online")
    u2_k0 = row(2, "K0-fixed")
    u2_a5 = row(2, "A5-online")

    def require(condition: bool, reason: str) -> None:
        if not condition:
            reasons.append(reason)

    require(
        u1_k0.get("state_before") == u1_a5.get("state_before"),
        "k0_a5_initial_state_mismatch",
    )
    require(
        bool((u1_a5.get("state_before") or {}).get("controlled_card", {}).get("present")),
        "controlled_card_absent_before_update",
    )
    require(
        (u1_a5.get("support_a") or {}).get("face_id")
        != (u1_a5.get("support_b") or {}).get("face_id"),
        "support_faces_not_distinct",
    )
    require(
        (u1_a5.get("support_a") or {}).get("relation") == "POSITIVE"
        and float((u1_a5.get("support_a") or {}).get("gain") or 0.0) >= MATERIAL,
        "unit1_support_not_positive",
    )
    require(
        (u1_a5.get("support_b") or {}).get("relation") == "NEGATIVE"
        and float((u1_a5.get("support_b") or {}).get("gain") or 0.0) < -MATERIAL,
        "unit1_delayed_not_negative",
    )
    require(
        bool((u1_a5.get("trace") or {}).get("controlled_card_deployed")),
        "unit1_card_not_deployed_after_support",
    )
    require(
        bool((u1_a5.get("update") or {}).get("production_revocation"))
        and (u1_a5.get("update") or {}).get("kind")
        == "REVOKE_CONTROLLED_SUPPLY_CARD",
        "production_revocation_missing",
    )
    require(
        not bool((u1_a5.get("state_after") or {}).get("controlled_card", {}).get("present")),
        "revocation_did_not_change_semantic_state",
    )
    require(
        bool((u1_a5.get("update") or {}).get("retained_at_unit_boundary")),
        "a5_update_not_retained",
    )
    require(
        u2_a5.get("state_before") == u1_a5.get("state_after"),
        "a5_carried_state_not_used_at_reencounter",
    )
    require(
        u2_k0.get("state_before") == u1_k0.get("state_before"),
        "k0_did_not_reset_to_initial_state",
    )
    require(
        (u1_a5.get("context") or {}).get("pattern")
        == (u2_a5.get("context") or {}).get("pattern"),
        "observable_pattern_not_reencountered",
    )
    require(
        int((u2_k0.get("trace") or {}).get("controlled_supply_count") or 0) > 0
        and int((u2_k0.get("trace") or {}).get("controlled_probe_count") or 0) > 0,
        "k0_control_not_reached_at_reencounter",
    )
    require(
        bool((u2_k0.get("trace") or {}).get("retrieved_controlled_card"))
        and bool((u2_k0.get("trace") or {}).get("controlled_card_scope_match")),
        "k0_control_not_retrieved_in_scope",
    )
    require(
        int((u2_a5.get("trace") or {}).get("controlled_supply_count") or 0) == 0
        and int((u2_a5.get("trace") or {}).get("controlled_probe_count") or 0) == 0,
        "a5_revoked_card_still_reached",
    )
    require(
        not bool((u2_a5.get("trace") or {}).get("retrieved_controlled_card")),
        "a5_revoked_card_still_retrieved",
    )
    require(
        not bool((u2_k0.get("trace") or {}).get("abstained"))
        and bool((u2_a5.get("trace") or {}).get("abstained")),
        "reencounter_behavior_did_not_change",
    )
    require(
        int((u2_k0.get("trace") or {}).get("autonomous_nonidentity_candidate_count") or 0)
        == 0
        and int((u2_a5.get("trace") or {}).get("autonomous_nonidentity_candidate_count") or 0)
        == 0,
        "behavior_difference_not_attributable_to_carried_update",
    )
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "status": (
            "TREATMENT_WIRING_NONEMPTY"
            if not unique_reasons
            else "TREATMENT_WIRING_EMPTY"
        ),
        "treatment_nonempty": not unique_reasons,
        "producer_decision_index": 1,
        "reencounter_decision_index": 2,
        "update_semantics": "production_delayed_harm_revocation",
        "influence_axis": (
            "carried_revocation_withholds_supply_probe_and_changes_abstain"
            if not unique_reasons
            else None
        ),
        "failures": unique_reasons,
    }


def _budget_failures(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    required = {
        "support_a_full_evaluations",
        "support_b_full_evaluations",
        "full_support_evaluations",
        "cheap_probes",
        "agent_stage_calls",
        "external_llm_calls",
        "tokens",
        "accepted_updates",
        "within_caps",
    }
    for record in runs:
        usage = dict(record.get("usage") or {})
        label = "%s/%s" % (record.get("unit_id"), record.get("arm"))
        if not required.issubset(usage):
            failures.append("missing_budget_fields:" + label)
            continue
        if int(usage["full_support_evaluations"]) != (
            int(usage["support_a_full_evaluations"])
            + int(usage["support_b_full_evaluations"])
        ):
            failures.append("support_accounting_mismatch:" + label)
        if usage.get("within_caps") is not True:
            failures.append("budget_cap_exceeded:" + label)
        if int(usage["external_llm_calls"]) != 0 or int(usage["tokens"]) != 0:
            failures.append("external_model_call_observed:" + label)
    return failures


def _arm_failures(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    for decision_index in (1, 2):
        rows = [row for row in runs if int(row.get("decision_index", -1)) == decision_index]
        if len(rows) != len(ARMS) or {str(row.get("arm")) for row in rows} != set(ARMS):
            failures.append("invalid_arm_profile_at_decision_%d" % decision_index)
    labels = [
        str(row["store_isolation_label"])
        for row in runs
        if row.get("store_isolation_label") is not None
    ]
    if len(labels) != len(set(labels)):
        failures.append("adaptive_store_labels_overlap")
    for row in runs:
        arm = str(row.get("arm"))
        if arm == "Static" and row.get("store_isolation_label") is not None:
            failures.append("static_created_evolution_store")
        if arm == "A3-reset" and bool(
            (row.get("state_before") or {}).get("controlled_card", {}).get("present")
        ):
            failures.append("a3_received_controlled_card")
        if arm in {"Static", "A3-reset", "K0-fixed"} and row.get(
            "boundary_action"
        ) == "CARRY_EVOLVED_STATE":
            failures.append("non_a5_state_carried")
    return failures


def _boundary_failures(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    for row in runs:
        counts = dict(row.get("boundary_counts") or {})
        if not counts or any(int(value) != 0 for value in counts.values()):
            failures.append(
                "closed_boundary_violation:%s/%s"
                % (row.get("unit_id"), row.get("arm"))
            )
    return failures


def run() -> dict[str, Any]:
    started = time.time()
    upstream = _assert_p1_release()
    base, data = forecast_p1._load_exposed_cell()
    cells = {int(unit["seed"]): _make_cell(base, int(unit["seed"])) for unit in COURSE}
    first = cells[int(COURSE[0]["seed"])]
    second = cells[int(COURSE[1]["seed"])]
    observation_equal = bool(
        np.array_equal(first.observation_block, second.observation_block, equal_nan=True)
    )
    distinct_realizations = any(
        not np.array_equal(
            first.values[uid], second.values[uid], equal_nan=True
        )
        for uid in base.support_b
    )
    if not observation_equal or not distinct_realizations:
        raise P2Blocked("controlled course identity checks failed")

    spec, context = _task_contract()
    temp_root = Path(tempfile.mkdtemp(prefix="forecast_p2_"))
    try:
        h0 = forecast_course._h0()
        initial = forecast_course._install(
            h0,
            _controlled_card(),
            store_root=temp_root / "initial",
            tag="controlled_origin",
        )
        if not _public_state(initial)["controlled_card"]["present"]:
            raise P2Blocked("controlled supply card did not enter the initial state")
        initial_semantics = _semantic_state(initial)
        runs: list[dict[str, Any]] = []
        a5_snapshot = initial
        a5_episodes: tuple[Any, ...] = ()
        for unit in COURSE:
            cell = cells[int(unit["seed"])]
            runs.append(_static_record(unit=unit, cell=cell, h0=h0))
            for arm in ARMS[1:]:
                if arm == "A3-reset":
                    base_snapshot, episodes = h0, ()
                elif arm == "K0-fixed":
                    base_snapshot, episodes = initial, ()
                else:
                    base_snapshot, episodes = a5_snapshot, a5_episodes
                record, end_snapshot, end_episodes = _run_adaptive(
                    unit=unit,
                    arm=arm,
                    cell=cell,
                    base_snapshot=base_snapshot,
                    carried_episodes=episodes,
                    temp_root=temp_root,
                    spec=spec,
                    context=context,
                )
                runs.append(record)
                if arm == "A5-online":
                    a5_snapshot, a5_episodes = end_snapshot, end_episodes

        u1_k0 = next(
            row for row in runs
            if row["decision_index"] == 1 and row["arm"] == "K0-fixed"
        )
        u1_a5 = next(
            row for row in runs
            if row["decision_index"] == 1 and row["arm"] == "A5-online"
        )
        k0_a5_equal = bool(
            u1_k0["state_before"] == u1_a5["state_before"]
            and initial_semantics == _semantic_state(initial)
        )
        treatment = derive_treatment_gate(runs)
        budget_failures = _budget_failures(runs)
        arm_failures = _arm_failures(runs)
        boundary_failures = _boundary_failures(runs)
        protocol_errors = {
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
            "traffic_loader_invocations": 0,
            "solar_loader_invocations": 0,
            "query_feedback_events": 0,
            "task_mismatch_execution": sum(
                any(
                    str(step.get("op")) not in set(_eligible_programs())
                    for step in (row.get("trace") or {}).get("selected_steps") or ()
                )
                for row in runs
            ),
            "direct_controlled_card_deployment": sum(
                bool((row.get("trace") or {}).get("controlled_card_deployed"))
                and int((row.get("trace") or {}).get("controlled_probe_count") or 0) == 0
                for row in runs
            ),
            "uncontrolled_agent_candidates": sum(
                int((row.get("trace") or {}).get("autonomous_nonidentity_candidate_count") or 0)
                for row in runs
            ),
        }
        nonzero_protocol_errors = [
            key for key, value in protocol_errors.items() if int(value) != 0
        ]
        blocking_failures = [
            *budget_failures,
            *arm_failures,
            *boundary_failures,
            *["protocol_error:" + key for key in nonzero_protocol_errors],
            *list(treatment["failures"]),
        ]
        if not k0_a5_equal:
            blocking_failures.append("k0_a5_initial_semantic_state_mismatch")
        p2_complete = not blocking_failures and treatment["treatment_nonempty"]
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": STAGE,
            "task": TASK,
            "evidence_grade": EVIDENCE_GRADE,
            "verdict": (
                "P2_FORECAST_TREATMENT_WIRING_PASS__P3_INTEGRATION_RELEASED"
                if p2_complete
                else "P2_FORECAST_TREATMENT_WIRING_BLOCKED"
            ),
            "p2_complete": p2_complete,
            "release_p3": p2_complete,
            "p3_complete": False,
            "release_p4": False,
            "live_outcome_release": False,
            "release_scope": (
                "P3 Classification/AD complete roster and vertical-slice integration only"
                if p2_complete
                else "NONE"
            ),
            "upstream_p1": upstream,
            "scope": {
                "dataset": data["dataset"],
                "data_role": "EXPOSED_DEVELOPMENT_CONTROLLED_WITNESS",
                "consumer_id": CONSUMER_ID,
                "metric": METRIC,
                "support_a_series": list(base.support_a),
                "support_b_series": list(base.support_b),
                "support_a_count": len(base.support_a),
                "support_b_count": len(base.support_b),
                "support_faces_disjoint": set(base.support_a).isdisjoint(base.support_b),
                "real_consumer_origin": REAL_ORIGIN,
                "horizon": HORIZON,
                "period": PERIOD,
            },
            "controlled_witness": {
                "family": "corpus_level_one_shot_impulsive_outlier",
                "faulty_face": "support_b_series",
                "clean_face": "support_a_series",
                "amplitude": 8.0,
                "count_per_faulty_series": 40,
                "seeds": [int(unit["seed"]) for unit in COURSE],
                "program": CONTROL_PROGRAM,
                "observation_block_equal_across_units": observation_equal,
                "controlled_realizations_distinct": distinct_realizations,
                "card_role": "SUPPLY_ONLY_MECHANISM_CONTROL",
                "card_learned_from_source_evidence": False,
                "requires_target_support": True,
                "grants_execution": False,
                "scientific_capability_evidence": False,
            },
            "course": [
                {
                    **dict(unit),
                    "support_face": "train_b_eval_a",
                    "delayed_face": "train_a_eval_b",
                    "real_consumer_origin": REAL_ORIGIN,
                    "query_evaluations": 0,
                }
                for unit in COURSE
            ],
            "arm_contracts": {
                "profile_choice": "four-arm diagnostic profile for this pilot",
                "protocol_minimum_claim": False,
                "Static": "identity only; no Harness lifecycle",
                "A3-reset": "H0 each unit; unit state discarded",
                "K0-fixed": "controlled initial state each unit; unit state discarded",
                "A5-online": "same initial state as K0; accepted update carried",
                "k0_a5_same_initial_semantic_state": k0_a5_equal,
            },
            "runs": runs,
            "treatment_gate": treatment,
            "rq3_event_gate": {
                "status": (
                    "CONTROLLED_POSITIVE_NEGATIVE_REENCOUNTER_EXERCISED"
                    if treatment["treatment_nonempty"]
                    else "NOT_EXERCISED"
                ),
                "online_evolution_positive_claim": False,
                "reason": "controlled mechanism event only; no utility or repetition claim",
            },
            "budget_caps": {
                "full_support_evaluations": B_MAIN,
                "support_a_full_evaluations": MAX_SUPPORT_A,
                "support_b_full_evaluations": MAX_SUPPORT_B,
                "cheap_probes": MAX_CHEAP_PROBES,
                "agent_stage_calls": MAX_AGENT_STAGE_CALLS,
                "external_llm_calls": MAX_EXTERNAL_LLM_CALLS,
                "tokens": MAX_TOKENS,
                "accepted_updates": MAX_UPDATES,
                "raw_consumer_fits": "REPORTED_SEPARATELY_NOT_A_B4_GATE",
            },
            "budget_gate": {
                "status": "PASS" if not budget_failures else "FAIL",
                "failures": budget_failures,
                "external_llm_calls": 0,
                "tokens": 0,
            },
            "boundaries": {
                "development_query_evaluations": 0,
                "natural_final_outcome_reads": 0,
                "traffic_loader_invocations": 0,
                "solar_loader_invocations": 0,
                "query_feedback_events": 0,
                "heldout_fast_only_claim": False,
                "freeze_api_claim": False,
            },
            "protocol_errors": protocol_errors,
            "blocking_failures": list(dict.fromkeys(blocking_failures)),
            "claim_boundaries": {
                "unified_runner_wiring": p2_complete,
                "treatment_wiring_nonempty": treatment["treatment_nonempty"],
                "performance_claim": False,
                "headroom_claim": False,
                "natural_data_capability_claim": False,
                "source_transfer_capability_claim": False,
                "p4_evolution_release": False,
                "natural_final_release": False,
            },
            "wall_seconds": round(time.time() - started, 3),
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _failure_payload(exc: Exception) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE,
        "task": TASK,
        "evidence_grade": EVIDENCE_GRADE,
        "verdict": "P2_FORECAST_TREATMENT_WIRING_BLOCKED",
        "p2_complete": False,
        "release_p3": False,
        "p3_complete": False,
        "release_p4": False,
        "live_outcome_release": False,
        "release_scope": "NONE",
        "runtime_failure_type": type(exc).__name__,
        "blocking_failures": ["bounded P2 runner stopped at its first runtime fault"],
        "boundaries": {
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
            "traffic_loader_invocations": 0,
            "solar_loader_invocations": 0,
            "query_feedback_events": 0,
        },
        "claim_boundaries": {
            "performance_claim": False,
            "headroom_claim": False,
            "natural_data_capability_claim": False,
            "p4_evolution_release": False,
            "natural_final_release": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-pass",
        action="store_true",
        help="return non-zero unless the bounded P2 mechanism gate passes",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = run()
    except Exception as exc:  # noqa: BLE001 - one bounded first-fault artifact
        payload = _failure_payload(exc)
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("VERDICT %s" % payload["verdict"], flush=True)
    print("ARTIFACT %s" % OUT_JSON.relative_to(PROJECT_ROOT), flush=True)
    return int(bool(args.expect_pass and not payload.get("p2_complete")))


if __name__ == "__main__":
    raise SystemExit(main())
