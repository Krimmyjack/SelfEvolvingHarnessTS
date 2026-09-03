import copy

import pytest

from SelfEvolvingHarnessTS.methods.ttha.skill_acquisition import (
    CANDIDATE,
    apply_typed_patch,
    attach_delayed_outcomes,
    build_candidate_skill,
    build_policy_failure_dossier,
    collect_source_policy_episodes,
    execute_skill_card,
    execute_stop_on_first_positive,
    plan_skill_card_support_only,
    plan_support_only,
    read_active_skill_cards,
    run_failure_driven_update_cycle,
    validate_failure_driven_patch,
    validate_typed_patch,
)


SUPPLY = ("W_rowblock", "W_curation")


def _episode(rowblock_support, rowblock_query, curation_support, curation_query):
    return {
        "workflows": {
            "W_rowblock": {
                "workflow_id": "W_rowblock",
                "support_gain": rowblock_support,
                "query_gain": rowblock_query,
            },
            "W_curation": {
                "workflow_id": "W_curation",
                "support_gain": curation_support,
                "query_gain": curation_query,
            },
        }
    }


def _candidate():
    return build_candidate_skill(
        [],
        [
            _episode(0.4, 0.3, -0.1, -0.2),
            _episode(0.2, 0.1, 0.1, 0.05),
        ],
        capability_id="natural_forecasting_candidate_v1",
        task_context={"task": "forecasting", "consumer": "frozen-ridge"},
        workflow_supply=SUPPLY,
    )


def test_empty_memory_source_policy_episodes_build_candidate_only():
    candidate = _candidate()

    assert candidate["status"] == CANDIDATE
    assert candidate["source_prior"]["source_policy_episode_count"] == 2
    assert set(candidate["source_prior"]["workflow_order"]) == set(SUPPLY)
    assert candidate["evidence"]["promotion_status"] == "NOT_EVALUATED"


def test_current_support_stops_after_first_positive():
    calls = []
    responses = {
        "W_rowblock": {"support_gain": 0.2, "query_gain": 0.4},
        "W_curation": {"support_gain": 0.9, "query_gain": -1.0},
    }

    result = execute_stop_on_first_positive(
        SUPPLY,
        SUPPLY,
        lambda workflow_id: calls.append(workflow_id) or responses[workflow_id],
    )

    assert calls == ["W_rowblock"]
    assert result["selected_workflow"] == "W_rowblock"
    assert result["adaptation_curve"][-1]["fixed_query_gain"] == 0.4


def test_no_positive_support_preserves_identity():
    responses = {
        "W_rowblock": {"support_gain": -0.2, "query_gain": 1.0},
        "W_curation": {"support_gain": 0.0, "query_gain": 2.0},
    }

    result = execute_stop_on_first_positive(
        SUPPLY, SUPPLY, lambda workflow_id: responses[workflow_id]
    )

    assert result["selected_workflow"] == "IDENTITY"
    assert result["abstained"] is True
    assert result["adaptation_auc"] == 0.0


def test_support_only_plan_precedes_delayed_outcome_evaluation():
    support = {
        "W_rowblock": {"support_gain": 0.2},
        "W_curation": {"support_gain": 0.9},
    }
    plan = plan_support_only(
        SUPPLY,
        SUPPLY,
        lambda workflow_id: support[workflow_id],
        control="stop_on_first_positive",
    )

    assert plan["selected_workflow"] == "W_rowblock"
    assert plan["probed_workflows"] == ["W_rowblock"]
    assert "adaptation_curve" not in plan
    assert "adaptation_auc" not in plan
    assert all("fixed_query_gain" not in row for row in plan["support_planning_trace"])
    assert all("query_gain" not in row for row in plan["support_observations"])
    with pytest.raises(ValueError, match="only support_gain"):
        plan_support_only(
            SUPPLY,
            SUPPLY,
            lambda _workflow_id: {"support_gain": 0.2, "query_gain": 999.0},
            control="stop_on_first_positive",
        )

    evaluated = attach_delayed_outcomes(plan, {"W_rowblock": 0.4})
    legacy = execute_stop_on_first_positive(
        SUPPLY,
        SUPPLY,
        lambda workflow_id: {
            "support_gain": support[workflow_id]["support_gain"],
            "query_gain": 0.4 if workflow_id == "W_rowblock" else -1.0,
        },
    )

    assert evaluated["selected_workflow"] == legacy["selected_workflow"]
    assert evaluated["adaptation_curve"] == legacy["adaptation_curve"]
    assert evaluated["adaptation_auc"] == pytest.approx(0.3)


def test_delayed_outcome_attachment_fails_closed_when_unavailable():
    plan = plan_support_only(
        SUPPLY,
        SUPPLY,
        lambda workflow_id: {"support_gain": 0.2},
        control="stop_on_first_positive",
    )

    with pytest.raises(ValueError, match="unavailable"):
        attach_delayed_outcomes(plan, {})
    with pytest.raises(ValueError, match="finite"):
        attach_delayed_outcomes(plan, {"W_rowblock": float("nan")})


def test_support_only_skill_plan_scope_mismatch_never_probes():
    candidate = _candidate()
    candidate["applicability"] = {
        "cohort_topology": "multi_series_cross_sectional",
        "on_mismatch": "ABSTAIN",
    }
    calls = []

    plan = plan_skill_card_support_only(
        candidate,
        lambda workflow_id: calls.append(workflow_id) or {"support_gain": 1.0},
        allow_candidate_replay=True,
        execution_context={"cohort_topology": "single_series_temporal_origins"},
    )

    assert calls == []
    assert plan["selected_workflow"] == "IDENTITY"
    assert plan["applicability_matched"] is False
    assert "adaptation_auc" not in plan


def test_forbidden_typed_patch_is_rejected():
    candidate = _candidate()
    forbidden = {
        "patch_id": "rewrite-consumer",
        "operations": [
            {
                "operation": "PATCH_CONTROL",
                "target_surface": "consumer",
                "value": {"consumer": "new-model"},
            }
        ],
    }

    with pytest.raises(ValueError):
        validate_typed_patch(forbidden, candidate)


def test_patch_and_candidate_replay_never_auto_promote():
    candidate = _candidate()
    with pytest.raises(ValueError, match="not executable"):
        execute_skill_card(
            candidate,
            lambda _workflow_id: {"support_gain": 1.0, "query_gain": 1.0},
        )

    patched = apply_typed_patch(
        candidate,
        {
            "patch_id": "stop-first-positive-v1",
            "operations": [
                {
                    "operation": "PATCH_CONTROL",
                    "target_surface": "control",
                    "value": "stop_on_first_positive",
                }
            ],
        },
    )
    replay = execute_skill_card(
        patched,
        lambda _workflow_id: {"support_gain": 1.0, "query_gain": 0.2},
        allow_candidate_replay=True,
    )

    assert patched["status"] == CANDIDATE
    assert replay["status"] == CANDIDATE


def test_pre_evolution_control_can_be_overwritten_then_patch_stops_early():
    candidate = _candidate()
    responses = {
        "W_rowblock": {"support_gain": 0.2, "query_gain": 0.4},
        "W_curation": {"support_gain": 0.9, "query_gain": -1.0},
    }
    baseline = execute_skill_card(
        candidate,
        lambda workflow_id: responses[workflow_id],
        allow_candidate_replay=True,
    )
    patched = apply_typed_patch(
        candidate,
        {
            "patch_id": "stop-overwrite",
            "operations": [
                {
                    "operation": "PATCH_CONTROL",
                    "target_surface": "harness_update_policy",
                    "value": "stop_on_first_positive",
                }
            ],
        },
    )
    repaired = execute_skill_card(
        patched,
        lambda workflow_id: responses[workflow_id],
        allow_candidate_replay=True,
    )

    assert baseline["probed_workflows"] == ["W_rowblock", "W_curation"]
    assert baseline["adaptation_curve"][-1]["fixed_query_gain"] == -1.0
    assert repaired["probed_workflows"] == ["W_rowblock"]
    assert repaired["adaptation_curve"][-1]["fixed_query_gain"] == 0.4


def test_failure_dossier_and_patch_are_bounded_by_diagnosed_surfaces():
    observation_case = {
        "candidate_probe_order": ["W_rowblock", "W_curation"],
        "workflow_responses": {
            "W_rowblock": {"support_gain": -0.1, "query_gain": -0.2},
            "W_curation": {"support_gain": 0.3, "query_gain": 0.4},
        },
        "candidate_curve": [
            {"budget": 0, "fixed_query_gain": 0.0},
            {"budget": 1, "fixed_query_gain": 0.0},
            {"budget": 2, "fixed_query_gain": 0.4},
        ],
        "comparison_adaptation_auc": 0.3,
    }
    overwrite_case = {
        "candidate_probe_order": ["W_rowblock", "W_curation"],
        "workflow_responses": {
            "W_rowblock": {"support_gain": 0.2, "query_gain": 0.3},
            "W_curation": {"support_gain": 0.9, "query_gain": -0.5},
        },
        "candidate_curve": [
            {"budget": 0, "fixed_query_gain": 0.0},
            {"budget": 1, "fixed_query_gain": 0.3},
            {"budget": 2, "fixed_query_gain": -0.5},
        ],
    }
    dossier = build_policy_failure_dossier(
        [observation_case, overwrite_case],
        allowed_observations=[
            "phase_aligned_historical_policy_episode",
            "cohort_overview",
        ],
        allowed_controls=[
            "stop_on_first_positive",
            "keep_best_support_so_far",
        ],
    )
    patch = {
        "patch_id": "bounded-repair",
        "operations": [
            {
                "operation": "ADD_OBSERVATION",
                "target_surface": "observation",
                "value": "phase_aligned_historical_policy_episode",
            },
            {
                "operation": "PATCH_CONTROL",
                "target_surface": "harness_update_policy",
                "value": "stop_on_first_positive",
            },
        ],
    }

    codes = {row["code"] for row in dossier["categorical_first_faults"]}
    assert codes == {
        "GLOBAL_WORKFLOW_ORDER_NOT_TARGET_CONTEXTUALIZED",
        "CONFIRMED_POSITIVE_WORKFLOW_OVERWRITTEN",
    }
    assert validate_failure_driven_patch(patch, _candidate(), dossier) == patch

    patch["operations"][0]["value"] = "invented_observation"
    with pytest.raises(ValueError, match="outside"):
        validate_failure_driven_patch(patch, _candidate(), dossier)


def test_support_query_transport_fault_can_only_propose_bounded_composition():
    composition = {
        "type": "split_support_stability_gate",
        "minimum_target_feedback_units": 2,
    }
    dossier = build_policy_failure_dossier(
        [
            {
                "support_to_query_replays": [
                    {
                        "selected_program": "SEASONAL_RESIDUAL_TARGET",
                        "support_gains": {
                            "SEASONAL_RESIDUAL_TARGET": 0.2,
                            "LAST_VALUE_RESIDUAL_TARGET": -0.1,
                        },
                        "query_gain": -0.3,
                    }
                ]
            }
        ],
        allowed_observations=["phase_aligned_historical_policy_episode"],
        allowed_controls=["keep_best_support_so_far"],
        allowed_compositions=[composition],
    )
    patch = {
        "patch_id": "transport-repair",
        "operations": [
            {
                "operation": "COMPOSE_WORKFLOW",
                "target_surface": "workflow",
                "value": composition,
            }
        ],
    }

    assert dossier["categorical_first_faults"] == [
        {
            "surface": "workflow_composition",
            "code": "ONE_SUPPORT_PROBE_CAN_FALSELY_CONFIRM",
            "observed_behavior": (
                "a Program selected by a positive current-Support response is "
                "harmful on the paired Query cohort"
            ),
        }
    ]
    normalized = validate_failure_driven_patch(patch, _candidate(), dossier)
    patched = apply_typed_patch(_candidate(), normalized)
    assert patched["workflow_composition"] == composition
    assert patched["status"] == CANDIDATE

    patch["operations"][0]["value"] = {"type": "invented_composition"}
    with pytest.raises(ValueError, match="outside"):
        validate_failure_driven_patch(patch, _candidate(), dossier)


def test_cohort_topology_scope_patch_abstains_on_mismatch():
    scope = {
        "cohort_topology": "multi_series_cross_sectional",
        "on_mismatch": "ABSTAIN",
    }
    dossier = build_policy_failure_dossier(
        [
            {
                "source_cohort_topology": "multi_series_cross_sectional",
                "target_cohort_topology": "single_series_temporal_origins",
                "support_to_query_replays": [
                    {
                        "selected_program": "W_rowblock",
                        "support_gains": {"W_rowblock": 0.2},
                        "query_gain": -0.3,
                    }
                ],
            }
        ],
        allowed_observations=["phase_aligned_historical_policy_episode"],
        allowed_controls=["keep_best_support_so_far"],
        allowed_scopes=[scope],
    )
    patch = {
        "patch_id": "cohort-topology-scope-repair",
        "operations": [
            {
                "operation": "RESTRICT_SCOPE",
                "target_surface": "applicability",
                "value": scope,
            }
        ],
    }

    assert [row["code"] for row in dossier["categorical_first_faults"]] == [
        "SOURCE_SCOPE_OMITS_COHORT_TOPOLOGY"
    ]
    patched = apply_typed_patch(
        _candidate(),
        validate_failure_driven_patch(patch, _candidate(), dossier),
    )
    calls = []
    result = execute_skill_card(
        patched,
        lambda workflow_id: calls.append(workflow_id)
        or {"support_gain": 1.0, "query_gain": -1.0},
        allow_candidate_replay=True,
        execution_context={"cohort_topology": "single_series_temporal_origins"},
    )

    assert patched["applicability"] == scope
    assert calls == []
    assert result["selected_workflow"] == "IDENTITY"
    assert result["adaptation_auc"] == 0.0
    assert result["applicability_matched"] is False


def test_failure_driven_cycle_rejects_non_improving_composition_explicitly():
    composition = {
        "type": "split_support_stability_gate",
        "minimum_target_feedback_units": 2,
    }
    patch = {
        "patch_id": "transport-repair",
        "operations": [
            {
                "operation": "COMPOSE_WORKFLOW",
                "target_surface": "workflow",
                "value": composition,
            }
        ],
    }
    result = run_failure_driven_update_cycle(
        _candidate(),
        [
            {
                "support_to_query_replays": [
                    {
                        "selected_program": "W_rowblock",
                        "support_gains": {"W_rowblock": 0.2},
                        "query_gain": -0.1,
                    }
                ]
            }
        ],
        allowed_observations=["phase_aligned_historical_policy_episode"],
        allowed_controls=["keep_best_support_so_far"],
        allowed_compositions=[composition],
        propose_patch=lambda dossier: patch,
        replay_patch=lambda candidate: {
            "behavior_nontrivial": True,
            "policy_value_above_A3": False,
        },
        resolve_patch=lambda candidate, replays: {
            "status": "REJECTED",
            "reason": "no incremental policy value",
        },
    )

    assert result["candidate_after_patch"]["status"] == CANDIDATE
    assert result["resolved_skill"]["status"] == "REJECTED"
    assert result["resolved_skill"]["promotion_result"]["reason"] == (
        "no incremental policy value"
    )


def test_fast_path_reads_only_admitted_skill_cards():
    admitted = _candidate()
    admitted["status"] = "CROSS_DATASET_SUPPORTED"
    rejected = _candidate()
    rejected["capability_id"] = "rejected-skill"
    rejected["status"] = "REJECTED_AFTER_INDEPENDENT_DATASET_CONFIRMATION"

    active = read_active_skill_cards([admitted, rejected])

    assert [row["capability_id"] for row in active] == [
        admitted["capability_id"]
    ]
    assert rejected["status"] == "REJECTED_AFTER_INDEPENDENT_DATASET_CONFIRMATION"


def test_fresh_harm_state_update_restricts_without_rewriting_history():
    admitted = _candidate()
    admitted["status"] = "CROSS_DATASET_SUPPORTED"
    original = copy.deepcopy(admitted)

    active = read_active_skill_cards(
        [admitted],
        state_updates=[
            {
                "capability_id": admitted["capability_id"],
                "status": "RESTRICTED",
                "reason": "fresh Target harm",
            }
        ],
    )

    assert active == []
    assert admitted == original


def test_discovered_subset_reuses_a_richer_source_episode():
    richer_episode = _episode(0.4, 0.3, -0.1, -0.2)
    richer_episode["workflows"]["W_temporal_origin"] = {
        "workflow_id": "W_temporal_origin",
        "support_gain": 0.5,
        "query_gain": 0.4,
    }

    candidate = build_candidate_skill(
        [],
        [richer_episode],
        capability_id="bounded_discovery_candidate_v1",
        task_context={"task": "forecasting"},
        workflow_supply=["W_rowblock", "W_temporal_origin"],
    )

    assert candidate["workflow_supply"] == ["W_rowblock", "W_temporal_origin"]
    assert set(candidate["source_prior"]["workflow_order"]) == {
        "W_rowblock",
        "W_temporal_origin",
    }


def test_discovered_workflows_are_collected_as_source_policy_episode():
    calls = []
    compiled = [
        {"workflow_id": "W_rowblock", "bindings": {"period": 24}},
        {"workflow_id": "W_curation", "bindings": {}},
    ]

    def evaluate(context, workflow_id, bindings):
        calls.append((context["case_id"], workflow_id, bindings))
        return {
            "support_gain": 0.2 if workflow_id == "W_rowblock" else -0.1,
            "query_gain": 0.3 if workflow_id == "W_rowblock" else -0.2,
        }

    episodes = collect_source_policy_episodes(
        [{"case_id": "source-a", "workspace_ref": "opaque-local-ref"}],
        compiled,
        evaluate,
    )

    assert [row[1] for row in calls] == ["W_rowblock", "W_curation"]
    assert episodes[0]["source_case_id"] == "source-a"
    assert episodes[0]["workflows"]["W_rowblock"]["bindings"] == {"period": 24}
    assert episodes[0]["workflows"]["W_curation"]["query_gain"] == -0.2
