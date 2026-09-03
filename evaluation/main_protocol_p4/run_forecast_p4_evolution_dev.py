"""Exercise the missing Forecast Slow loop on exposed development data.

This is a bounded connection-repair diagnostic, not a rerun of P4-Performance
and not a release of H3.  It uses only exposed KDD held-in faces, starts at the
first chronological development origin without outcome-based origin choice,
and keeps Natural Final closed.  The path under test is:

first natural Fast fault -> verifier-earned one-surface Slow edit -> pending
candidate Harness -> next held-in Fast replay -> independent Support-B
promotion -> frozen later re-encounter against K0.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    activate_approved,
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent

from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms
from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import run_forecast_p4_performance as performance
from evaluation.main_protocol_p4 import run_p4 as split_release


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p4_forecast_evolution_slow_dev_20260831.json"
)

STAGE = "P4_FORECAST_EVOLUTION_SLOW_CONNECTION_REPAIR_DEV"
EVIDENCE_GRADE = "MECHANISM_DEVELOPMENT_EXPOSED_NO_H3_RELEASE"
FAULT_ORIGIN = 600
NEXT_FAST_ORIGIN = 648
REENCOUNTER_ORIGIN = 696
PERIOD = performance.PERIOD
HORIZON = performance.HORIZON
MATERIAL = resolver.MATERIAL_THRESHOLD
MAX_SUPPORT_A = performance.MAX_SUPPORT_A
MAX_SUPPORT_B = performance.MAX_SUPPORT_B
MAX_LLM_CALLS = performance.MAX_LLM_CALLS
MAX_TOKENS = performance.MAX_TOKENS
MAX_WALL_SECONDS = performance.MAX_WALL_SECONDS
GLOBAL_LLM_CAP = 4 * MAX_LLM_CALLS
PROGRAM_SUPPLY_VERIFIER_CAP = (
    performance.MAX_CHEAP_PROBES - performance.B_MAIN - (
        MAX_SUPPORT_A + MAX_SUPPORT_B
    )
)


class EvolutionDevBlocked(RuntimeError):
    """The development Slow slice cannot be interpreted safely."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write(payload: Mapping[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    staged = OUT_JSON.with_suffix(OUT_JSON.suffix + ".tmp")
    staged.write_text(
        json.dumps(performance._plain(payload), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    staged.replace(OUT_JSON)


def _persist_budget(payload: dict[str, Any], state: Mapping[str, Any]) -> None:
    payload["llm_budget_instrument"] = {
        "classification": "PRE_CALL_HARD_GUARD",
        "frozen_per_cell_cap": MAX_LLM_CALLS,
        "global_cap": GLOBAL_LLM_CAP,
        **performance._plain(state),
    }
    _write(payload)


def _assert_boundary() -> dict[str, Any]:
    gate = performance._read_object(split_release.OUT_JSON)
    evolution = dict(gate.get("p4_evolution") or {})
    checks = {
        "p4_evolution_remains_held": evolution.get("status") == "HELD",
        "rq3_not_exercised": (
            dict(evolution.get("rq3_status_by_task") or {}).get("forecast")
            == "RQ3_NOT_EXERCISED"
        ),
        "natural_final_closed": gate.get("natural_final_release") is False,
        "final_reads_zero": int(gate.get("final_outcome_reads", -1)) == 0,
    }
    if not all(checks.values()):
        raise EvolutionDevBlocked(
            "development boundary failed: %s"
            % [name for name, passed in checks.items() if not passed]
        )
    return {
        "source": split_release.OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "checks": checks,
        "gate_before": "HELD__RQ3_NOT_EXERCISED",
        "gate_after_authorized": False,
    }


def typed_patch_options() -> tuple[dict[str, Any], ...]:
    """Existing frozen B=8 Common-DSL inventory; no outcome chooses an option."""
    return tuple(
        {
            "patch_id": "forecast-existing-program-%s" % op,
            "program_steps": [
                {"op": step_op, "params": dict(params)}
                for step_op, params in forecast_p1._steps(op)
            ],
        }
        for op in performance.PARALLEL_PROGRAMS
    )


def _card_builder(origin: int):
    def build(episode: Any) -> Mapping[str, object]:
        context = dict(getattr(episode, "context_summary", None) or {})
        geometry = dict(context.get("program_geometry") or {})
        support = dict(getattr(episode, "support_response", None) or {})
        return {
            "pattern_id": "forecast-p4-dev-first-fault-%d" % int(origin),
            "failure_family": "workflow_component_negative",
            "observable_signature": {"task_kind": "forecast"},
            "workflow": {
                "steps": list(geometry.get("program_steps") or ())
            },
            "facts": {
                "relation": str(getattr(episode, "relation", "")),
                "support_gain": support.get("gain"),
                "development_origin": int(origin),
            },
        }

    return build


def _face_bundle(cell: Any, origin: int) -> dict[str, Any]:
    identity, identity_wall = performance._identity_reference(cell, origin)
    evaluator = performance._CountingEval(
        cell, performance._config(origin), origin
    )
    executors = {
        face: ScopeExecutor(
            cell.roster(face),
            cell.values,
            performance._config(origin),
            evaluate_fn=evaluator,
            max_modified_fraction=performance.MAX_MODIFIED_FRACTION,
        )
        for face in ("support_a", "support_b")
    }
    for face, executor in executors.items():
        executor._baseline_cache[origin] = float(identity[face]["smase"])
        executor._per_view_cache[origin] = [
            float(value) for value in identity[face]["per_series_smase"]
        ]
    return {
        "identity": identity,
        "identity_wall_seconds": identity_wall,
        "evaluator": evaluator,
        "support_a": executors["support_a"],
        "support_b": executors["support_b"],
    }


def _unit(origin: int, label: str) -> dict[str, Any]:
    return {
        "replica": "Development",
        "sequence_index": 1,
        "episode_id": label,
        "origin": int(origin),
        "horizon": HORIZON,
        "natural_episode": True,
    }


def _request(cell: Any, origin: int, label: str, spec: Any, context: Any):
    return performance._request(
        unit=_unit(origin, label),
        cell=cell,
        origin=origin,
        spec=spec,
        context=context,
    )


def _steps(result: Any) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(
        (str(op), dict(params))
        for op, params in (getattr(result, "_winner_steps", None) or ())
    )


def _fit_counts(bundle: Mapping[str, Any]) -> dict[str, int]:
    return {
        face: int(count)
        for face, count in bundle["evaluator"].fits_by_face.items()
    }


def _round_record(
    result: Any,
    method: Any,
    bundle: Mapping[str, Any],
    *,
    fits_before: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    trace = method.last_trace
    before = dict(fits_before or {"support_a": 0, "support_b": 0})
    after = _fit_counts(bundle)
    chosen_id = str(trace.chosen_candidate_id or "")
    chosen_steps = tuple(
        (str(op), dict(params))
        for op, params in (
            dict(trace.candidate_program_steps or {}).get(chosen_id) or ()
        )
    )
    updated = {
        str(getattr(episode, "episode_id", "")): episode
        for episode in method.experience_episodes
    }
    winner_episode_relations = [
        str(getattr(updated.get(ep.episode_id, ep), "relation", ""))
        for ep, steps in result._episodes
        if tuple(steps) == tuple(result._winner_steps or ())
    ]
    return {
        "chosen_candidate_id": chosen_id,
        "chosen_candidate_source_skill_id": (
            performance.source_skill_of_candidate(chosen_id)
            if hasattr(performance, "source_skill_of_candidate")
            else None
        ),
        "chosen_steps": [
            {"op": op, "params": dict(params)}
            for op, params in chosen_steps
        ],
        "candidate_ids": list(trace.candidate_ids or ()),
        "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
        "winner_candidate_id": str(result._winner_candidate_id or ""),
        "winner_candidate_source_skill_id": (
            performance.source_skill_of_candidate(
                result._winner_candidate_id
            )
            if hasattr(performance, "source_skill_of_candidate")
            else None
        ),
        "winner_steps": [
            {"op": op, "params": dict(params)} for op, params in _steps(result)
        ],
        "winner_delayed_relations": winner_episode_relations,
        "probes": performance._plain(result.actual_probed_programs),
        "support_receipts_used": int(result.target_support_receipts_used),
        "slow_replay_receipts_used": int(result.slow_replay_receipts_used),
        "program_supply_verifier_requests": int(
            result.program_supply_verifier_requests
        ),
        "program_supply_verifier_blocked": int(
            result.program_supply_verifier_blocked
        ),
        "harm_count": int(result.harm_count),
        "harm_magnitude": float(result.harm_magnitude),
        "abstained": bool(result.abstained),
        "slow_event": performance._plain(result._slow_event),
        "support_b_gain": (
            float(result.delayed_utility)
            if result.delayed_utility is not None
            else None
        ),
        "consumer_fits": {
            face: after[face] - int(before.get(face, 0))
            for face in after
        },
    }


def _trial_fast_selects_pending(
    trial_method: Any,
    pending_steps: Sequence[tuple[str, Mapping[str, object]]],
) -> bool:
    trace = trial_method.last_trace
    chosen_id = str(trace.chosen_candidate_id or "")
    chosen_steps = tuple(
        (str(op), dict(params))
        for op, params in (
            dict(trace.candidate_program_steps or {}).get(chosen_id) or ()
        )
    )
    return chosen_steps == tuple(
        (str(op), dict(params)) for op, params in pending_steps
    )


def _winner_uses_skill(result: Any, skill_id: str | None) -> bool:
    if not skill_id:
        return False
    return (
        performance.source_skill_of_candidate(result._winner_candidate_id)
        == skill_id
    )


def _winner_is_fast_choice(result: Any, method: Any) -> bool:
    trace = method.last_trace
    winner_id = str(result._winner_candidate_id or "")
    return bool(
        winner_id
        and winner_id == str(trace.chosen_candidate_id or "")
    )


def _winner_matches_trace_steps(result: Any, method: Any) -> bool:
    trace = method.last_trace
    winner_id = str(result._winner_candidate_id or "")
    traced_steps = tuple(
        (str(op), dict(params))
        for op, params in (
            dict(trace.candidate_program_steps or {}).get(winner_id) or ()
        )
    )
    return bool(winner_id and traced_steps and _steps(result) == traced_steps)


def _budget_terminal_status(passed: bool) -> str:
    return "COMPLETE" if passed else "FAILED"


def _is_cell_llm_exhaustion(exc: shared_harness.Stop) -> bool:
    return exc.verdict == performance.CELL_LLM_EXHAUSTION_VERDICT


def _versioned_revision(
    before: Any, candidate: Any, slow_event: Mapping[str, Any]
) -> dict[str, Any]:
    operation = str(slow_event.get("operation") or "")
    target = str(slow_event.get("target_surface_id") or "")
    before_revisions = {
        str(skill.skill_id): int(skill.revision) for skill in before.skills
    }
    after_revisions = {
        str(skill.skill_id): int(skill.revision) for skill in candidate.skills
    }
    target_skill = ""
    prefix = "skill_library.entries/"
    suffix = ".body"
    if target.startswith(prefix) and target.endswith(suffix):
        target_skill = target[len(prefix):-len(suffix)]
    passed = bool(
        operation == "PATCH"
        and target_skill
        and target_skill in before_revisions
        and after_revisions.get(target_skill) == before_revisions[target_skill] + 1
    )
    return {
        "passed": passed,
        "operation": operation,
        "target_skill_id": target_skill or None,
        "revision_before": before_revisions.get(target_skill),
        "revision_after": after_revisions.get(target_skill),
    }


def _pending_skill_id(pending: Mapping[str, Any]) -> str | None:
    manifest = pending.get("manifest_applied")
    new_value = dict(getattr(manifest, "new_value", None) or {})
    if new_value.get("skill_id"):
        return str(new_value["skill_id"])
    target = str(getattr(manifest, "target_surface_id", "") or "")
    prefix = "skill_library.entries/"
    suffix = ".body"
    if target.startswith(prefix) and target.endswith(suffix):
        return target[len(prefix):-len(suffix)]
    return None


def _usage_checks(usage: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "support_a_within_cap": int(usage.get("support_a_fits") or 0)
        <= MAX_SUPPORT_A,
        "support_b_within_cap": int(usage.get("support_b_fits") or 0)
        <= MAX_SUPPORT_B,
        "cheap_probes_within_cap": int(usage.get("cheap_probes") or 0)
        <= performance.MAX_CHEAP_PROBES,
        "llm_calls_within_cap": int(usage.get("llm_calls") or 0)
        <= MAX_LLM_CALLS,
        "tokens_within_cap": int(usage.get("tokens") or 0) <= MAX_TOKENS,
        "accepted_updates_within_cap": int(
            usage.get("accepted_updates") or 0
        ) <= performance.MAX_UPDATES,
        "program_supply_verifier_within_cap": int(
            usage.get("program_supply_verifier_requests") or 0
        ) <= PROGRAM_SUPPLY_VERIFIER_CAP,
        "wall_seconds_within_cap": float(usage.get("wall_seconds") or 0.0)
        <= MAX_WALL_SECONDS,
    }


def _budget_exhausted_record(
    *,
    scope: str,
    exc: shared_harness.Stop,
    backend: Any,
    before_backend: tuple[int, int, int],
    bundle: Mapping[str, Any],
    fits_before: Mapping[str, int],
    dispatcher: Any,
    method: Any,
    started: float,
) -> dict[str, Any]:
    before_calls, before_input, before_output = before_backend
    after_calls, after_input, after_output = forecast_p1._backend_usage(backend)
    fits_after = _fit_counts(bundle)
    accounting = dispatcher.accounting()
    trace = getattr(method, "last_trace", None)
    fast_verifier = (
        forecast_p1._fast_verifier_requests(trace) if trace is not None else 0
    )
    usage = {
        "support_a_fits": (
            fits_after["support_a"] - int(fits_before.get("support_a", 0))
        ),
        "support_b_fits": (
            fits_after["support_b"] - int(fits_before.get("support_b", 0))
        ),
        "cheap_probes": int(
            fast_verifier + accounting["unique_candidate_verifier_requests"]
        ),
        "llm_calls": after_calls - before_calls,
        "input_tokens": after_input - before_input,
        "output_tokens": after_output - before_output,
        "tokens": (after_input - before_input) + (after_output - before_output),
        "accepted_updates": 0,
        "program_supply_verifier_requests": 0,
        "wall_seconds": round(time.time() - started, 3),
    }
    return {
        "scope": scope,
        "chosen_candidate_id": "",
        "chosen_steps": [],
        "winner_steps": [],
        "support_b_gain": 0.0,
        "winner_delayed_relations": ["ABSTAIN"],
        "abstained": True,
        "abstain_reason": str(exc.verdict),
        "budget_exhaustion_action": performance.CELL_LLM_EXHAUSTION_ACTION,
        "llm_budget_exhausted": True,
        "consumer_fits": {
            face: fits_after[face] - int(fits_before.get(face, 0))
            for face in fits_after
        },
        "usage": usage,
        "usage_checks": _usage_checks(usage),
    }


def _new_method(
    *, snapshot: Any, cell: Any, arm_backend: Any, root: Path, tag: str
) -> dict[str, Any]:
    return four_arms._new_state(
        snapshot=snapshot,
        agent=forecast_course._live_agent(cell.observation_block, arm_backend),
        store_root=root,
        tag=tag,
        episodes=(),
    )


def _reencounter(
    *,
    label: str,
    snapshot: Any,
    cell: Any,
    backend: Any,
    temp_root: Path,
    spec: Any,
    context: Any,
) -> dict[str, Any]:
    scope = "reencounter/%s" % label
    bundle = _face_bundle(cell, REENCOUNTER_ORIGIN)
    fits_before = _fit_counts(bundle)
    arm_backend = backend.new_arm_backend(
        scope_id=scope, maximum_calls=MAX_LLM_CALLS
    )
    before_calls, before_input, before_output = forecast_p1._backend_usage(backend)
    state = _new_method(
        snapshot=snapshot,
        cell=cell,
        arm_backend=arm_backend,
        root=temp_root / scope.replace("/", "_"),
        tag="state",
    )
    method = state["method"]
    request, features = _request(
        cell, REENCOUNTER_ORIGIN, "reencounter", spec, context
    )
    delayed_token = REENCOUNTER_ORIGIN + HORIZON
    dispatcher = performance._OriginDispatcher(
        {
            REENCOUNTER_ORIGIN: (
                "support_a", bundle["support_a"], REENCOUNTER_ORIGIN
            ),
            delayed_token: (
                "support_b", bundle["support_b"], REENCOUNTER_ORIGIN
            ),
        }
    )
    started = time.time()
    try:
        result = run_online_round(
            method,
            dispatcher,
            request,
            cell.values,
            origin=REENCOUNTER_ORIGIN,
            slow_agent=None,
            controller=state["controller"],
            store=state["store"],
            card_builder=lambda _episode: {},
            round_name="reencounter_%s" % label.lower(),
            budget=MAX_SUPPORT_A,
            allow_slow=False,
            horizon=HORIZON,
            period=PERIOD,
            domain="forecast_p4_dev_reencounter",
            fast_features=features,
            allow_fast_skill=False,
            runtime_prior_slot=False,
            pool_mode="full",
        )
    except shared_harness.Stop as exc:
        if not _is_cell_llm_exhaustion(exc):
            raise
        return _budget_exhausted_record(
            scope=scope,
            exc=exc,
            backend=backend,
            before_backend=(before_calls, before_input, before_output),
            bundle=bundle,
            fits_before=fits_before,
            dispatcher=dispatcher,
            method=method,
            started=started,
        )
    open_delayed(
        result,
        dispatcher,
        delayed_origin=delayed_token,
        store=state["store"],
    )
    after_calls, after_input, after_output = forecast_p1._backend_usage(backend)
    record = _round_record(
        result, method, bundle, fits_before=fits_before
    )
    record["usage"] = {
        "support_a_fits": record["consumer_fits"]["support_a"],
        "support_b_fits": record["consumer_fits"]["support_b"],
        "cheap_probes": int(
            forecast_p1._fast_verifier_requests(method.last_trace)
            + dispatcher.accounting()["unique_candidate_verifier_requests"]
        ),
        "llm_calls": after_calls - before_calls,
        "input_tokens": after_input - before_input,
        "output_tokens": after_output - before_output,
        "tokens": (after_input - before_input) + (after_output - before_output),
        "accepted_updates": 0,
        "wall_seconds": round(time.time() - started, 3),
    }
    record["usage_checks"] = _usage_checks(record["usage"])
    record["winner_is_fast_choice"] = _winner_is_fast_choice(result, method)
    record["winner_matches_trace_steps"] = _winner_matches_trace_steps(
        result, method
    )
    record["scope"] = scope
    return record


def _verdict(payload: Mapping[str, Any]) -> str:
    chain = dict(payload.get("chain") or {})
    first = dict(chain.get("first_fault") or {})
    if first.get("llm_budget_exhausted") is True:
        return "FIRST_CELL_LLM_BUDGET_EXHAUSTED_IDENTITY__H3_HELD"
    next_fast = dict(chain.get("next_fast") or {})
    if next_fast.get("llm_budget_exhausted") is True:
        return "NEXT_CELL_LLM_BUDGET_EXHAUSTED_IDENTITY__H3_HELD"
    event = dict(first.get("slow_event") or {})
    if not first.get("harm_count"):
        return "NO_NATURAL_FIRST_FAULT__H3_HELD"
    if event.get("stage") != "pending":
        return "SLOW_UPDATE_NOT_PENDING__H3_HELD"
    if chain.get("next_fast_used_pending_harness") is not True:
        return "NEXT_FAST_DID_NOT_USE_PENDING_HARNESS__H3_HELD"
    if chain.get("support_b_approved") is not True:
        return "INDEPENDENT_SUPPORT_B_REJECTED__H3_HELD"
    if chain.get("promotion_activated") is not True:
        return "PROMOTION_NOT_ACTIVATED__H3_HELD"
    if chain.get("versioned_revision") is not True:
        return "ADD_ONLY_PROMOTED_REENCOUNTERED__H3_HELD"
    if chain.get("reencounter_material_improvement") is not True:
        return "REVISION_REENCOUNTER_NO_MATERIAL_GAIN__H3_HELD"
    return "EXPOSED_DEV_REVISION_CHAIN_PASS__P4_EVOLUTION_STILL_HELD"


def run(*, backend_mode: str) -> dict[str, Any]:
    boundary = _assert_boundary()
    payload: dict[str, Any] = {
        "stage": STAGE,
        "status": "RUNNING",
        "started_at": _now(),
        "completed_at": None,
        "evidence_grade": EVIDENCE_GRADE,
        "release": boundary,
        "scope": {
            "dataset": "KDD_Cup_2018",
            "data_role": "EXPOSED_DEVELOPMENT_HELD_IN",
            "fresh_or_held_out_claim": False,
            "performance_rerun": False,
            "p4_evolution_release": False,
            "origins": [FAULT_ORIGIN, NEXT_FAST_ORIGIN, REENCOUNTER_ORIGIN],
            "origin_selection_rule": (
                "first three chronological exposed development origins; "
                "no Outcome-based origin selection"
            ),
            "natural_agent_first_fault_required": True,
            "controlled_card_or_forced_answer": False,
        },
        "budget": {
            "per_method_cell": {
                "support_a_max": MAX_SUPPORT_A,
                "support_b_max": MAX_SUPPORT_B,
                "llm_call_max": MAX_LLM_CALLS,
                "token_max": MAX_TOKENS,
                "wall_seconds_max": MAX_WALL_SECONDS,
                "cheap_probe_max": performance.MAX_CHEAP_PROBES,
            },
            "program_supply_verifier_only_cap": (
                PROGRAM_SUPPLY_VERIFIER_CAP
            ),
            "global_llm_call_cap": GLOBAL_LLM_CAP,
        },
        "program_space": {
            "fast_legal_inventory_source": "P1 frozen eligible Forecast registry",
            "fast_legal_operators": [],
            "slow_verified_whitelist_source": (
                "existing frozen Forecast B=8 Common-DSL order"
            ),
            "slow_verified_whitelist_operators": list(
                performance.PARALLEL_PROGRAMS
            ),
            "new_operator_count": 0,
            "outcome_selects_option": False,
        },
        "backend_mode": backend_mode,
        "chain": {},
        "llm_budget_instrument": {
            "classification": "PRE_CALL_HARD_GUARD",
            "frozen_per_cell_cap": MAX_LLM_CALLS,
            "global_cap": GLOBAL_LLM_CAP,
            "global_calls": 0,
            "scope_calls": {},
            "scope_caps": {},
            "call_records": [],
            "blocked_records": [],
        },
        "protocol_errors": [],
        "claim_boundary": {
            "p4_performance_run2_unchanged": True,
            "p4_evolution_before": "HELD__RQ3_NOT_EXERCISED",
            "p4_evolution_after": "HELD_PENDING_RESULT_REVIEW",
            "natural_final_release": False,
            "natural_final_outcome_reads": 0,
            "new_sha_or_hash_infrastructure": False,
        },
    }
    _write(payload)
    started = time.time()
    try:
        if backend_mode != "live":
            raise EvolutionDevBlocked("this development retry requires --backend live")
        base_cell, _selection, data = forecast_p1._load_exposed_cells()
        cells = {
            origin: performance._cell_at(base_cell, origin)
            for origin in (FAULT_ORIGIN, NEXT_FAST_ORIGIN, REENCOUNTER_ORIGIN)
        }
        bundles = {
            origin: _face_bundle(cells[origin], origin)
            for origin in (FAULT_ORIGIN, NEXT_FAST_ORIGIN)
        }
        payload["scope"].update(
            {
                "dataset": data["dataset"],
                "support_a_series_count": len(base_cell.support_a),
                "support_b_series_count": len(base_cell.support_b),
                "agent_inaccessible_identity_reference_fits_precomputed": 4,
            }
        )
        eligible = forecast_p1._eligible_programs()
        payload["program_space"]["fast_legal_operators"] = list(eligible)
        spec, context = forecast_p1._task_contract(
            eligible, maximum_candidates=performance.B_MAIN
        )
        from evaluation.functional.task_episode_harness.agentic.runner import (
            live_transport,
        )

        payload["backend_target"] = live_transport(
            default_model=shared_harness.SLOW_MODEL
        )
        backend = shared_harness._live_backend(
            GLOBAL_LLM_CAP,
            on_budget_change=lambda state: _persist_budget(payload, state),
        )

        with tempfile.TemporaryDirectory(prefix="forecast_p4_slow_dev_") as name:
            temp_root = Path(name)
            h0 = forecast_course._h0()
            source_card, source_contract = (
                forecast_p1._audited_forecast_supply_card()
            )
            shared_initial = forecast_course._install(
                h0,
                source_card,
                store_root=temp_root / "initial",
                tag="forecast_task_local_initial",
            )
            payload["initial_knowledge"] = {
                "source": source_contract,
                "k0_a5_same_initial_snapshot": True,
            }

            first_scope = "adaptation-first/A5-online"
            adaptation_backend = backend.new_arm_backend(
                scope_id=first_scope, maximum_calls=MAX_LLM_CALLS
            )
            first_before_backend = forecast_p1._backend_usage(backend)
            first_started = time.time()
            first_fits_before = _fit_counts(bundles[FAULT_ORIGIN])
            state = _new_method(
                snapshot=shared_initial,
                cell=cells[FAULT_ORIGIN],
                arm_backend=adaptation_backend,
                root=temp_root / "adaptation_a5",
                tag="state",
            )
            method = state["method"]
            slow_agent = TTHASlowAgent(method.fast_agent.core)
            request, features = _request(
                cells[FAULT_ORIGIN], FAULT_ORIGIN, "first_fault", spec, context
            )
            promotion_token = NEXT_FAST_ORIGIN
            dispatcher = performance._OriginDispatcher(
                {
                    FAULT_ORIGIN: (
                        "support_a",
                        bundles[FAULT_ORIGIN]["support_a"],
                        FAULT_ORIGIN,
                    ),
                    promotion_token: (
                        "support_b",
                        bundles[NEXT_FAST_ORIGIN]["support_b"],
                        NEXT_FAST_ORIGIN,
                    ),
                }
            )
            try:
                first = run_online_round(
                    method,
                    dispatcher,
                    request,
                    cells[FAULT_ORIGIN].values,
                    origin=FAULT_ORIGIN,
                    slow_agent=slow_agent,
                    controller=state["controller"],
                    store=state["store"],
                    card_builder=_card_builder(FAULT_ORIGIN),
                    round_name="first_fault",
                    budget=MAX_SUPPORT_A,
                    allow_slow=True,
                    horizon=HORIZON,
                    period=PERIOD,
                    domain="forecast_p4_dev_first_fault",
                    fast_features=features,
                    allow_fast_skill=False,
                    runtime_prior_slot=False,
                    pool_mode="full",
                    slow_typed_patch_options=typed_patch_options(),
                    program_supply_verifier=(
                        bundles[FAULT_ORIGIN]["support_a"]
                    ),
                    program_supply_verifier_budget=(
                        PROGRAM_SUPPLY_VERIFIER_CAP
                    ),
                    constrained_proposal_succeeds=None,
                )
            except shared_harness.Stop as exc:
                if not _is_cell_llm_exhaustion(exc):
                    raise
                payload["chain"]["first_fault"] = _budget_exhausted_record(
                    scope=first_scope,
                    exc=exc,
                    backend=backend,
                    before_backend=first_before_backend,
                    bundle=bundles[FAULT_ORIGIN],
                    fits_before=first_fits_before,
                    dispatcher=dispatcher,
                    method=method,
                    started=first_started,
                )
                first = None
            first_after_backend = forecast_p1._backend_usage(backend)
            first_active_wall = time.time() - first_started

            pending = (
                getattr(method, "_pending_update", None)
                if first is not None else None
            )
            trial = None
            trial_method = None
            trial_dispatcher = None
            trial_usage = None
            pending_steps: tuple[tuple[str, dict[str, object]], ...] = ()
            pending_skill_id = None
            revision_evidence: dict[str, Any] = {
                "passed": False,
                "operation": None,
                "target_skill_id": None,
                "revision_before": None,
                "revision_after": None,
            }
            if isinstance(pending, Mapping) and first.pending_patch_id:
                pending_steps = tuple(
                    (str(op), dict(params)) for op, params in pending["steps"]
                )
                candidate_snapshot = pending["receipt"].candidate_snapshot.snapshot
                pending_skill_id = _pending_skill_id(pending)
                revision_evidence = _versioned_revision(
                    method._active_snapshot(),
                    candidate_snapshot,
                    dict(first._slow_event or {}),
                )
                next_scope = "adaptation-next/A5-online"
                next_backend = backend.new_arm_backend(
                    scope_id=next_scope, maximum_calls=MAX_LLM_CALLS
                )
                next_before_backend = forecast_p1._backend_usage(backend)
                next_started = time.time()
                next_fits_before = _fit_counts(bundles[NEXT_FAST_ORIGIN])
                trial_state = _new_method(
                    snapshot=candidate_snapshot,
                    cell=cells[NEXT_FAST_ORIGIN],
                    arm_backend=next_backend,
                    root=temp_root / "pending_fast_replay",
                    tag="state",
                )
                trial_method = trial_state["method"]
                trial_request, trial_features = _request(
                    cells[NEXT_FAST_ORIGIN],
                    NEXT_FAST_ORIGIN,
                    "next_fast",
                    spec,
                    context,
                )
                trial_dispatcher = performance._OriginDispatcher(
                    {
                        NEXT_FAST_ORIGIN: (
                            "support_a",
                            bundles[NEXT_FAST_ORIGIN]["support_a"],
                            NEXT_FAST_ORIGIN,
                        )
                    }
                )
                try:
                    trial = run_online_round(
                        trial_method,
                        trial_dispatcher,
                        trial_request,
                        cells[NEXT_FAST_ORIGIN].values,
                        origin=NEXT_FAST_ORIGIN,
                        slow_agent=None,
                        controller=trial_state["controller"],
                        store=trial_state["store"],
                        card_builder=lambda _episode: {},
                        round_name="pending_harness_next_fast",
                        budget=MAX_SUPPORT_A,
                        allow_slow=False,
                        horizon=HORIZON,
                        period=PERIOD,
                        domain="forecast_p4_dev_next_fast",
                        fast_features=trial_features,
                        allow_fast_skill=False,
                        runtime_prior_slot=False,
                        pool_mode="full",
                    )
                except shared_harness.Stop as exc:
                    if not _is_cell_llm_exhaustion(exc):
                        raise
                    payload["chain"]["next_fast"] = (
                        _budget_exhausted_record(
                            scope=next_scope,
                            exc=exc,
                            backend=backend,
                            before_backend=next_before_backend,
                            bundle=bundles[NEXT_FAST_ORIGIN],
                            fits_before=next_fits_before,
                            dispatcher=trial_dispatcher,
                            method=trial_method,
                            started=next_started,
                        )
                    )
                    trial = None
                if trial is not None:
                    payload["chain"]["next_fast"] = _round_record(
                        trial,
                        trial_method,
                        bundles[NEXT_FAST_ORIGIN],
                        fits_before=next_fits_before,
                    )
                next_after = forecast_p1._backend_usage(backend)
                next_record = dict(payload["chain"].get("next_fast") or {})
                next_accounting = trial_dispatcher.accounting()
                next_fast_verifier = (
                    forecast_p1._fast_verifier_requests(
                        trial_method.last_trace
                    ) if trial_method.last_trace is not None else 0
                )
                trial_usage = {
                    "support_a_fits": int(
                        next_record.get("consumer_fits", {}).get(
                            "support_a", 0
                        )
                    ),
                    "support_b_fits": 0,
                    "cheap_probes": int(
                        next_fast_verifier
                        + next_accounting[
                            "unique_candidate_verifier_requests"
                        ]
                    ),
                    "llm_calls": next_after[0] - next_before_backend[0],
                    "input_tokens": next_after[1] - next_before_backend[1],
                    "output_tokens": next_after[2] - next_before_backend[2],
                    "tokens": (next_after[1] - next_before_backend[1])
                    + (next_after[2] - next_before_backend[2]),
                    "accepted_updates": 0,
                    "wall_seconds": round(time.time() - next_started, 3),
                }
                next_record["usage"] = trial_usage
                next_record["usage_checks"] = _usage_checks(trial_usage)
                payload["chain"]["next_fast"] = next_record

            fast_selected_pending = bool(
                trial is not None
                and pending_steps
                and _trial_fast_selects_pending(trial_method, pending_steps)
            )
            next_trace = (
                trial_method.last_trace
                if trial is not None and trial_method is not None else None
            )
            chosen_source = (
                performance.source_skill_of_candidate(
                    next_trace.chosen_candidate_id
                ) if next_trace is not None else None
            )
            edited_skill_retrieved = bool(
                pending_skill_id
                and next_trace is not None
                and pending_skill_id in tuple(next_trace.retrieved_skill_ids or ())
            )
            edited_skill_selected = bool(
                pending_skill_id and chosen_source == pending_skill_id
            )
            winner_matches_pending = bool(
                trial is not None and _steps(trial) == pending_steps
            )
            winner_uses_edited_skill = bool(
                trial is not None
                and _winner_uses_skill(trial, pending_skill_id)
            )
            winner_is_fast_choice = bool(
                trial is not None
                and _winner_is_fast_choice(trial, trial_method)
            )
            winner_matches_trace_steps = bool(
                trial is not None
                and _winner_matches_trace_steps(trial, trial_method)
            )
            used_pending = bool(
                fast_selected_pending
                and edited_skill_retrieved
                and edited_skill_selected
                and winner_matches_pending
                and winner_uses_edited_skill
                and winner_is_fast_choice
                and winner_matches_trace_steps
            )
            payload["chain"]["next_fast_used_pending_harness"] = used_pending
            payload["chain"]["next_fast_causal_evidence"] = {
                "pending_skill_id": pending_skill_id,
                "fast_selected_pending_steps": fast_selected_pending,
                "edited_skill_retrieved": edited_skill_retrieved,
                "chosen_candidate_source_skill_id": chosen_source,
                "edited_skill_selected": edited_skill_selected,
                "runtime_winner_matches_pending": winner_matches_pending,
                "runtime_winner_candidate_id": (
                    str(trial._winner_candidate_id or "")
                    if trial is not None else None
                ),
                "runtime_winner_source_skill_id": (
                    performance.source_skill_of_candidate(
                        trial._winner_candidate_id
                    ) if trial is not None else None
                ),
                "runtime_winner_uses_edited_skill": winner_uses_edited_skill,
                "runtime_winner_is_fast_choice": winner_is_fast_choice,
                "runtime_winner_matches_trace_steps": (
                    winner_matches_trace_steps
                ),
            }

            support_b_wall = 0.0
            if used_pending:
                support_b_started = time.time()
                open_delayed(
                    first,
                    dispatcher,
                    delayed_origin=promotion_token,
                    store=state["store"],
                )
                activated = bool(
                    first.approved_skill_id is not None
                    and activate_approved(first, state["store"])
                )
                support_b_wall = time.time() - support_b_started
            else:
                activated = False
            delayed_event = dict(
                first._delayed_event or {}
            ) if first is not None else {}
            payload["chain"]["support_b"] = {
                "origin": NEXT_FAST_ORIGIN,
                "face": "support_b",
                "event": performance._plain(delayed_event),
                "gain": (
                    float(first.delayed_utility)
                    if first is not None and first.delayed_utility is not None
                    else None
                ),
            }
            payload["chain"]["support_b_approved"] = (
                delayed_event.get("stage") == "approved"
            )
            payload["chain"]["promotion_activated"] = activated
            slow_event = dict(
                first._slow_event or {}
            ) if first is not None else {}
            payload["chain"]["update_operation"] = slow_event.get("operation")
            payload["chain"]["versioned_revision_evidence"] = revision_evidence
            payload["chain"]["versioned_revision"] = bool(
                revision_evidence["passed"]
            )

            if first is not None:
                payload["chain"]["first_fault"] = _round_record(
                    first,
                    method,
                    bundles[FAULT_ORIGIN],
                    fits_before=first_fits_before,
                )

            first_accounting = dispatcher.accounting()
            first_fast_verifier = (
                forecast_p1._fast_verifier_requests(method.last_trace)
                if method.last_trace is not None else 0
            )
            first_record = dict(payload["chain"].get("first_fault") or {})
            first_usage = {
                "support_a_fits": int(
                    bundles[FAULT_ORIGIN]["evaluator"].fits_by_face[
                        "support_a"
                    ] - first_fits_before["support_a"]
                ),
                "support_b_fits": int(
                    bundles[NEXT_FAST_ORIGIN]["evaluator"].fits_by_face[
                        "support_b"
                    ]
                ),
                "cheap_probes": int(
                    first_fast_verifier
                    + first_accounting[
                        "unique_candidate_verifier_requests"
                    ]
                    + int(first_record.get(
                        "program_supply_verifier_requests", 0
                    ))
                ),
                "llm_calls": first_after_backend[0] - first_before_backend[0],
                "input_tokens": (
                    first_after_backend[1] - first_before_backend[1]
                ),
                "output_tokens": (
                    first_after_backend[2] - first_before_backend[2]
                ),
                "tokens": (
                    first_after_backend[1] - first_before_backend[1]
                    + first_after_backend[2] - first_before_backend[2]
                ),
                "accepted_updates": int(activated),
                "program_supply_verifier_requests": int(
                    first_record.get(
                        "program_supply_verifier_requests", 0
                    )
                ),
                "wall_seconds": round(first_active_wall + support_b_wall, 3),
            }
            first_record["usage"] = first_usage
            first_record["usage_checks"] = _usage_checks(first_usage)
            payload["chain"]["first_fault"] = first_record

            if activated:
                frozen = method._active_snapshot()
                a5 = _reencounter(
                    label="A5-online",
                    snapshot=frozen,
                    cell=cells[REENCOUNTER_ORIGIN],
                    backend=backend,
                    temp_root=temp_root,
                    spec=spec,
                    context=context,
                )
                k0 = _reencounter(
                    label="K0-fixed",
                    snapshot=shared_initial,
                    cell=cells[REENCOUNTER_ORIGIN],
                    backend=backend,
                    temp_root=temp_root,
                    spec=spec,
                    context=context,
                )
                a5_gain = a5.get("support_b_gain")
                k0_gain = k0.get("support_b_gain")
                delta = (
                    float(a5_gain) - float(k0_gain)
                    if a5_gain is not None and k0_gain is not None
                    else None
                )
                payload["chain"]["reencounter"] = {
                    "origin": REENCOUNTER_ORIGIN,
                    "A5-online": a5,
                    "K0-fixed": k0,
                    "A5_minus_K0_delta_utility": delta,
                }
                a5_relations = set(a5.get("winner_delayed_relations") or ())
                a5_selected_edited_skill = bool(
                    pending_skill_id
                    and a5.get("winner_candidate_source_skill_id")
                    == pending_skill_id
                    and a5.get("winner_is_fast_choice") is True
                    and a5.get("winner_matches_trace_steps") is True
                )
                payload["chain"]["reencounter_causal_skill_use"] = (
                    a5_selected_edited_skill
                )
                payload["chain"]["reencounter_material_improvement"] = bool(
                    delta is not None
                    and delta >= MATERIAL
                    and a5_gain is not None
                    and float(a5_gain) >= MATERIAL
                    and "POSITIVE" in a5_relations
                    and a5_selected_edited_skill
                    and not bool(a5.get("llm_budget_exhausted"))
                )
            else:
                payload["chain"]["reencounter"] = None
                payload["chain"]["reencounter_causal_skill_use"] = False
                payload["chain"]["reencounter_material_improvement"] = False

        usage_cells: dict[str, Mapping[str, Any]] = {}
        for label in ("first_fault", "next_fast"):
            record = dict(payload["chain"].get(label) or {})
            if isinstance(record.get("usage"), Mapping):
                usage_cells[label] = dict(record["usage"])
        reencounter = dict(payload["chain"].get("reencounter") or {})
        for label in ("A5-online", "K0-fixed"):
            record = dict(reencounter.get(label) or {})
            if isinstance(record.get("usage"), Mapping):
                usage_cells["reencounter/%s" % label] = dict(record["usage"])
        cell_checks = {
            label: _usage_checks(usage)
            for label, usage in usage_cells.items()
        }
        budget_state = dict(payload.get("llm_budget_instrument") or {})
        payload["backend_observed_returned_model"] = getattr(
            backend, "first_returned_model", None
        )
        budget_validation = {
            "cells": cell_checks,
            "global_llm_calls_within_cap": int(
                budget_state.get("global_calls") or 0
            ) <= GLOBAL_LLM_CAP,
            "all_cells_within_caps": all(
                all(checks.values()) for checks in cell_checks.values()
            ),
        }
        budget_validation["passed"] = bool(
            budget_validation["global_llm_calls_within_cap"]
            and budget_validation["all_cells_within_caps"]
        )
        payload["budget_validation"] = budget_validation
        payload["verdict"] = (
            _verdict(payload)
            if budget_validation["passed"]
            else "BUDGET_INSTRUMENT_FAILURE__NO_SCIENTIFIC_VERDICT"
        )
        payload["status"] = _budget_terminal_status(
            budget_validation["passed"]
        )
        payload["completed_at"] = _now()
        payload["claim_boundary"]["p4_evolution_after"] = (
            "HELD__DEVELOPMENT_RESULT_REQUIRES_REVIEW"
        )
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "FAILED"
        payload["completed_at"] = _now()
        payload["protocol_errors"].append(
            {
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }
        )
    _write(payload)
    return payload


def preflight() -> dict[str, Any]:
    boundary = _assert_boundary()
    base_cell, _selection, data = forecast_p1._load_exposed_cells()
    cells = {
        origin: performance._cell_at(base_cell, origin)
        for origin in (FAULT_ORIGIN, NEXT_FAST_ORIGIN, REENCOUNTER_ORIGIN)
    }
    options = typed_patch_options()
    checks = {
        "boundary": all(boundary["checks"].values()),
        "three_exposed_origins_available": len(cells) == 3,
        "frozen_existing_programs_only": tuple(
            option["program_steps"][0]["op"] for option in options
        )
        == tuple(performance.PARALLEL_PROGRAMS),
        "no_duplicate_patch_ids": len({o["patch_id"] for o in options})
        == len(options),
        "program_supply_verifier_cap_is_positive": (
            PROGRAM_SUPPLY_VERIFIER_CAP > 0
        ),
        "natural_final_reads_zero": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "dataset": data["dataset"],
        "checks": checks,
        "output": OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("live",), default="live")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.preflight_only:
        result = preflight()
    else:
        result = run(backend_mode=args.backend)
    print(json.dumps(performance._plain(result), ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
