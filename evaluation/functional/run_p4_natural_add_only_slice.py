"""P4 NATURAL_ADD_ONLY_SLOW_PROGRAM_EDIT_SLICE (rev4, 2026-08-16).

One runner, one report.  Three arms fork from the same materialized h0:

  STATIC  memory=(), no Harness update
  A3      memory=(), batch-end Slow update allowed
  A5      Source positive/negative/conflict Episodes, then identical rules

Prequential: batch 1 origins {600,792,888,984} run with all Slow triggers off;
batch end runs at most one ``run_p4_group_update`` per update arm; delayed is
evaluated at 1032; batch 2 origins {1032,1080} verify real retrieval/supply/
execution and delayed harm veto.

This slice only judges whether the natural ADD lifecycle closes.  It does NOT
judge A5 > A3.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_a5a3_runtime_regression as reg  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)

from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    load_episodes_from_v6_reports,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.group_fault import (  # noqa: E402
    build_contrast_capsule,
    group_first_faults,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.p4_runner import run_p4_group_update  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

PERIOD = 24
HORIZON = 48
BUDGET = 2
POOL = ("winsorize", "outlier_mad", "hampel_filter")
BATCH1_ORIGINS = (600, 792, 888, 984)
BATCH2_ORIGINS = (1032, 1080)
DELAYED_BATCH1 = 1032
DOMAIN = "kdd_cup_2018"
REPORT_REL = (
    PROJECT_ROOT
    / "artifacts/functional/e2"
    / "w1_p4_natural_add_only_slice_report.json"
)


def _load_cohort(root: Path):
    cohort = reg._load(root)
    return cohort["roster"], cohort["values"]


def _source_episodes(root: Path) -> tuple[Any, ...]:
    """A5 Source positive/negative/conflict, rebuilt from already-exposed
    reports; task_consumer_key aligned with the KDD target harness."""
    episodes = load_episodes_from_v6_reports(
        root / "artifacts/functional/e2"
    )
    aligned = [
        dataclasses.replace(
            ep, task_consumer_key="forecast|ridge|sMASE"
        )
        for ep in episodes
    ]
    return tuple(aligned)


def _new_method(snapshot: Any, series0: np.ndarray, origin: int,
                memory: Sequence[Any]) -> TTHAMethod:
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(
            explore=True, operators=POOL,
            max_propose_candidates=3, force_pool=True,
        ),
        LocalPublicToolGateway(series0[:origin], task_kind="forecast"),
    )
    return TTHAMethod(sealed.TTHAFastAgent(core), snapshot, tuple(memory))


def _reset_backend(method: TTHAMethod, series0: np.ndarray,
                   origin: int) -> None:
    core = method.fast_agent.core
    core.backend = sealed.SealedProbeBackend(
        explore=True, operators=POOL,
        max_propose_candidates=3, force_pool=True,
    )
    core.tools = LocalPublicToolGateway(series0[:origin], task_kind="forecast")


def _round(method: TTHAMethod, executor: ScopeExecutor, values: Any,
           series0: np.ndarray, origin: int, arm: str) -> Any:
    _reset_backend(method, series0, origin)
    return run_online_round(
        method,
        executor,
        _request(series0, values, origin),
        values,
        origin=origin,
        slow_agent=None,
        controller=None,
        store=None,
        card_builder=lambda e: {},
        round_name=f"p4_{arm}_{origin}",
        budget=BUDGET,
        allow_slow=False,
        allow_group_slow=False,
        domain=DOMAIN,
        period=PERIOD,
        fast_features=dict(
            extract_public_features(series0[:origin], task_kind="forecast")
        ),
    )


def _round_record(result: Any, method: TTHAMethod, origin: int) -> dict[str, Any]:
    trace = method.last_trace
    return {
        "origin": origin,
        "candidate_ids": list(trace.candidate_ids or ()),
        "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
        "chosen_candidate_id": str(trace.chosen_candidate_id or ""),
        "winner_program": result.winner_program,
        "probes": [
            {
                "candidate_id": p["candidate_id"],
                "kind": p["kind"],
                "gain": p.get("gain"),
                "passed": p.get("passed"),
            }
            for p in result.actual_probed_programs
        ],
        "target_support_receipts_used": result.target_support_receipts_used,
        "slow_replay_receipts_used": result.slow_replay_receipts_used,
        "snapshot_sha": method._active_snapshot().harness_content_sha,
    }


def _card_for_group(group: Mapping[str, Any], origin: int,
                    values: Mapping[str, Any],
                    capsule: Mapping[str, Any] | None = None) -> dict[str, object]:
    failed_op = str(group.get("workflow") or "winsorize")
    alternatives = tuple(op for op in POOL if op != failed_op)
    facts: dict[str, object] = {}
    if capsule is not None:
        facts["contrast_capsule"] = dict(capsule)
    return {
        "pattern_id": "p4-natural-add",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": dict(resolver.window_context(values, origin, PERIOD)),
        "workflow": {"steps": [{"op": failed_op, "params": {}}]},
        "typed_patch_options": [
            {
                "patch_id": f"patch-replace-{failed_op}-with-{op}",
                "program_steps": [{"op": op, "params": {}}],
            }
            for op in alternatives
        ],
        "instruction": (
            "Choose exactly one ADD manifest from the typed_patch_options "
            "whitelist, or abstain. The runtime binds the chosen program "
            "steps; never invent a patch_id or program steps."
        ),
        "facts": facts,
    }


def _batch_update(method: TTHAMethod, executor: ScopeExecutor,
                  values: Mapping[str, Any], series0: np.ndarray,
                  store: SnapshotStore, controller: EditController,
                  arm: str, api_key: str, client: Any) -> dict[str, Any]:
    episodes = tuple(method._experience_episodes)
    groups = group_first_faults(episodes, min_group=2)
    if not groups:
        return {"stage": "no_group_family", "n_episodes": len(episodes)}
    group = groups[0]
    failed_op = str(group.get("workflow") or "winsorize")
    candidate_workflows = tuple(op for op in POOL if op != failed_op)
    capsule = build_contrast_capsule(
        group,
        all_episodes=episodes,
        target_domain_namespace=DOMAIN,
        candidate_workflows=candidate_workflows,
    )
    origin = max(
        int((getattr(e, "context_summary", {}) or {}).get("support_origin")
            or 0)
        for e in group["episodes"]
    )
    features = dict(
        extract_public_features(series0[:origin], task_kind="forecast")
    )
    view = resolve_harness_view(
        method._active_snapshot(), features, role="fast"
    )
    card = _card_for_group(group, origin, values, capsule)
    backend = AgictoChatCompletionsBackend(
        client=client, base_url=smoke.BASE_URL
    )
    slow_core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:origin], task_kind="forecast"),
        model=smoke.MODEL,
        base_url=smoke.BASE_URL,
    )
    slow = TTHASlowAgent(slow_core)
    ev = run_p4_group_update(
        method=method,
        group=group,
        capsule=capsule,
        trace=method.last_trace,
        episode=group["episodes"][-1],
        view=view,
        executor=executor,
        origin=origin,
        slow_agent=slow,
        controller=controller,
        store=store,
        card_builder=lambda _g, _c: card,
        evaluator_group=lambda steps, ep: executor.evaluate(
            tuple(steps),
            int((getattr(ep, "context_summary", {}) or {}).get(
                "support_origin"
            ) or 0),
        ),
        fast_features=features,
        allowed_operator_contracts=(),
    )
    record: dict[str, Any] = {
        "arm": arm,
        "stage": ev.get("stage"),
        "group": {
            "workflow": group.get("workflow"),
            "sign": group.get("sign"),
            "n_episodes": len(group.get("episodes") or ()),
        },
        "capsule_contrast_counts": {
            key: len(value)
            for key, value in capsule.get("contrast_cases", {}).items()
        },
        "source_provenance": capsule.get("source_provenance", {}),
        "update_event": {
            key: ev.get(key)
            for key in (
                "stage", "route", "verified_patch_ids", "typed_option_count",
                "choice_offered", "verified_choice_offered", "patch_id",
                "frozen_program", "support_gain", "support_passed",
                "group_replay", "holdout_gain", "edit_id", "error",
                "no_choice_offered",
            )
            if key in ev
        },
    }
    if ev.get("stage") == "pending":
        delayed = method.handle_feedback_delayed(
            lambda steps, _mode: executor.evaluate(
                tuple(steps), DELAYED_BATCH1
            ),
        )
        record["delayed_event"] = delayed
        record["snapshot_after_delayed_sha"] = (
            method._active_snapshot().harness_content_sha
        )
        if delayed.get("stage") == "approved":
            skill_ids = [
                skill.skill_id
                for skill in method._active_snapshot().skills
                if skill.skill_kind.value == "capability"
            ]
            record["approved_skill_id"] = (
                skill_ids[0] if skill_ids else None
            )
    return record


def _batch2_check(method: TTHAMethod, executor: ScopeExecutor,
                  values: Mapping[str, Any], series0: np.ndarray,
                  arm: str, skill_id: str | None) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    retrieved = supplied = executed = False
    delayed_utilities: list[float | None] = []
    for origin in BATCH2_ORIGINS:
        result = _round(method, executor, values, series0, origin, arm)
        trace = method.last_trace
        if skill_id is not None:
            retrieved = retrieved or skill_id in list(
                trace.retrieved_skill_ids or ()
            )
            supplied = supplied or (
                f"cand_skill_{skill_id}" in list(trace.candidate_ids or ())
            )
        executed = executed or any(
            p.get("candidate_id") == f"cand_skill_{skill_id}"
            and p.get("kind") in ("probe", "slow_replay")
            for p in result.actual_probed_programs
        )
        rounds.append(_round_record(result, method, origin))
        open_delayed(result, executor, delayed_origin=origin + HORIZON)
        delayed_utilities.append(result.delayed_utility)
    return {
        "rounds": rounds,
        "skill_retrieved": bool(retrieved),
        "skill_supplied": bool(supplied),
        "skill_executed": bool(executed),
        "delayed_utilities": delayed_utilities,
        "final_snapshot_sha": method._active_snapshot().harness_content_sha,
    }


def main() -> int:
    root = PROJECT_ROOT
    api_key = next(
        (
            os.environ.get(name, "").strip()
            for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
            if os.environ.get(name, "").strip()
        ),
        None,
    )
    if not api_key:
        print("no api key")
        return 2
    import openai  # noqa: PLC0415

    roster, values = _load_cohort(root)
    series0 = np.asarray(values[roster[0]["series_uid"]], dtype=np.float64)
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_kdd)
    baseline = compile_snapshot(
        root / "methods/ttha/harness/h0", verify_lock=False
    )
    baseline_materialized = SnapshotStore(
        root / ".p4_natural_baseline_store"
    ).materialize(baseline)
    source = _source_episodes(root)

    report: dict[str, Any] = {
        "experiment_id": "v1-p4-natural-add-only-slow-program-edit-slice",
        "note": (
            "P4 rev4 natural ADD lifecycle on development-exposed KDD "
            "T117. Same materialized h0 baseline forked into STATIC/A3/A5. "
            "Batch 1 all Slow triggers off; batch end at most one verified "
            "group Slow update; batch 2 verifies retrieval/supply/execution "
            "and delayed harm veto. P4 does not judge A5 > A3."
        ),
        "apparatus": {
            "domain": DOMAIN,
            "period": PERIOD,
            "budget": BUDGET,
            "pool": list(POOL),
            "batch1_origins": list(BATCH1_ORIGINS),
            "batch2_origins": list(BATCH2_ORIGINS),
            "delayed_batch1": DELAYED_BATCH1,
            "slow_model": smoke.MODEL,
            "slow_base_url": smoke.BASE_URL,
            "baseline_harness_content_sha": (
                baseline.harness_content_sha
            ),
            "baseline_runtime_bundle_sha": (
                baseline_materialized.runtime_bundle_sha
            ),
            "source_episode_ids": [ep.episode_id for ep in source],
        },
        "arms": {},
    }

    openai_client = smoke.CountingClient(
        openai.OpenAI(
            api_key=api_key, base_url=smoke.BASE_URL, timeout=180
        ),
        max_calls=24,
    )

    arms: dict[str, dict[str, Any]] = {}
    for arm, memory in (
        ("STATIC", ()),
        ("A3", ()),
        ("A5", source),
    ):
        store = SnapshotStore(root / f".p4_natural_store_{arm.lower()}")
        controller = EditController(
            store, surfaces=SurfaceRegistry(), router=FaultRouter()
        )
        method = _new_method(baseline, series0, BATCH1_ORIGINS[0], memory)
        batch1_rounds: list[dict[str, Any]] = []
        for origin in BATCH1_ORIGINS:
            result = _round(method, executor, values, series0, origin, arm)
            batch1_rounds.append(_round_record(result, method, origin))
        arms[arm] = {
            "batch1": {
                "rounds": batch1_rounds,
                "episode_count": len(method._experience_episodes),
                "slow_calls_in_batch": 0,
            },
            "update": None,
            "batch2": None,
            "approved_skill_id": None,
        }
        if arm != "STATIC":
            update = _batch_update(
                method, executor, values, series0, store, controller,
                arm, api_key, openai_client,
            )
            arms[arm]["update"] = update
            approved_skill_id = update.get("approved_skill_id")
            arms[arm]["approved_skill_id"] = approved_skill_id
        # Batch 2 always runs: if update did not approve, it verifies that
        # the baseline remains unchanged and no fake skill is retrieved.
        arms[arm]["batch2"] = _batch2_check(
            method, executor, values, series0, arm,
            arms[arm].get("approved_skill_id"),
        )
    report["arms"] = arms
    report["llm_calls"] = openai_client.calls

    def route_fired(arm_rec: dict[str, Any]) -> bool:
        update = arm_rec.get("update") or {}
        return update.get("stage") not in (
            None,
            "no_group_family",
            "no_verified_alternatives",
            "no_verified_options",
        )

    def lifecycle_pass(arm_rec: dict[str, Any]) -> bool:
        update = arm_rec.get("update") or {}
        delayed = update.get("delayed_event") or {}
        batch2 = arm_rec.get("batch2") or {}
        return bool(
            update.get("stage") == "pending"
            and delayed.get("stage") == "approved"
            and batch2.get("skill_retrieved")
            and batch2.get("skill_supplied")
            and batch2.get("skill_executed")
        )

    fired_arms = [arm for arm in ("A3", "A5") if route_fired(arms[arm])]
    if not fired_arms:
        verdict = "P4_VACUOUS_NO_ROUTE_FIRED"
    elif any(lifecycle_pass(arms[arm]) for arm in fired_arms):
        verdict = "NATURAL_ADD_ONLY_SLOW_PROGRAM_EDIT_DEV_PASS"
    elif any(not lifecycle_pass(arms[arm]) for arm in fired_arms):
        verdict = "NATURAL_ADD_ONLY_SLOW_PROGRAM_EDIT_DEV_NEGATIVE"
    else:
        verdict = "INCONCLUSIVE"
    report["verdict"] = verdict
    report["route_fired_arms"] = fired_arms
    report["lifecycle_pass_arms"] = [
        arm for arm in fired_arms if lifecycle_pass(arms[arm])
    ]
    REPORT_REL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_REL.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"== P4 verdict: {verdict} route_fired={fired_arms}")
    print(json.dumps(
        {
            arm: {
                "update_stage": (arms[arm].get("update") or {}).get("stage"),
                "delayed_stage": ((arms[arm].get("update") or {})
                                  .get("delayed_event") or {}).get("stage"),
                "skill_retrieved": (arms[arm].get("batch2") or {})
                .get("skill_retrieved"),
                "skill_supplied": (arms[arm].get("batch2") or {})
                .get("skill_supplied"),
                "skill_executed": (arms[arm].get("batch2") or {})
                .get("skill_executed"),
                "approved_skill_id": arms[arm].get("approved_skill_id"),
            }
            for arm in ("A3", "A5")
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
