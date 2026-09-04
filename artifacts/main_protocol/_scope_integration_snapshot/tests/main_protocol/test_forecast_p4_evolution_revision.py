"""Focused contracts for the pre-registered Forecast Slow revision slice.

The tests are deliberately pure: they do not load NOAA values, call an Agent,
open Final, or write an experiment artifact.  Numeric examples below exercise
gate semantics only; they are not experimental observations.
"""
from __future__ import annotations

import copy

import pytest

from evaluation.main_protocol_p4 import (
    run_forecast_p4_evolution_revision as revision,
)


def _skill_v1() -> dict[str, object]:
    return {
        "skill_id": revision.OLD_SKILL_ID,
        "revision": 1,
        "body": [{"op": "outlier_iqr", "params": {}}],
        "allowed_tools": ["outlier_iqr"],
        "scope": {"task_kind": "forecast"},
        "risk": {
            "current_task_support_confirmation_required": True,
            "frozen_plan": {"program": "outlier_iqr", "excluded_series": []},
        },
        "status": "LOCAL_ACTIVE",
    }


def _skill_v2() -> dict[str, object]:
    skill = _skill_v1()
    skill["revision"] = 2
    skill["body"] = [{"op": "winsorize", "params": {}}]
    skill["allowed_tools"] = ["winsorize"]
    skill["risk"]["frozen_plan"]["program"] = "winsorize"
    return skill


def _patch_event() -> dict[str, object]:
    return {
        "operation": "PATCH",
        "target_surface_id": revision.EDIT_SURFACE,
        "patch_id": "forecast-existing-program-winsorize",
        "program_steps": [{"op": "winsorize", "params": {}}],
    }


def _passing_evidence() -> dict[str, object]:
    """A synthetic boundary case for deterministic verdict logic only."""
    return {
        "budget_valid": True,
        "llm_budget_exhausted_before_completion": False,
        "old_skill_qualified": True,
        "stage_1": {
            "v1_resolved": True,
            "fast_selected_v1": True,
            "runtime_executed_v1": True,
            "local_fault": True,
            "constrained_proposal_succeeds": False,
            "slow_status": "pending",
            "patch_valid": True,
            "support_replay_positive": True,
        },
        "stage_2": {
            "support_b_evaluations": 1,
            "support_b_relation": "POSITIVE",
            "promotion_activated": True,
            "version_chain_valid": True,
            "skill_id_preserved": True,
            "non_program_fields_preserved": True,
            "program_mirrors_consistent": True,
            "revision_before": 1,
            "revision_after": 2,
        },
        "stage_3": {
            "a5_v2_causally_used": True,
            "k0_v1_causally_used": True,
            "v2_minus_v1_utility": 0.005,
            "v2_minus_identity_utility": 0.005,
            "a5_harm_count": 0,
            "k0_harm_count": 0,
            "a5_harm_magnitude": 0.0,
            "k0_harm_magnitude": 0.0,
            "a5_active_revision": 2,
            "k0_active_revision": 1,
        },
    }


def _set_path(
    payload: dict[str, object], path: tuple[str, ...], value: object
) -> dict[str, object]:
    changed = copy.deepcopy(payload)
    cursor: dict[str, object] = changed
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[assignment]
    cursor[path[-1]] = value
    return changed


def test_primary_and_only_repeat_trajectories_are_frozen_same_domain_windows():
    assert revision.PRIMARY_ORIGINS == (8472, 8520, 8568)
    assert revision.REPEAT_ORIGINS == (8616, 8664, 8712)
    assert revision.HORIZON == 48

    primary = revision.trajectory_plan()
    repeat = revision.trajectory_plan(origins=revision.REPEAT_ORIGINS)
    assert [row["origin"] for row in primary] == list(
        revision.PRIMARY_ORIGINS
    )
    assert [row["origin"] for row in repeat] == list(
        revision.REPEAT_ORIGINS
    )
    assert [row["stage"] for row in primary] == [1, 2, 3]
    assert [row["role"] for row in primary] == [
        "old_skill_fault_and_program_patch",
        "independent_support_b",
        "independent_reencounter",
    ]
    assert all(row["dataset"] == "NOAA" for row in primary + repeat)
    assert all(
        row["data_role"] == "EXPOSED_DEVELOPMENT_HELD_IN"
        for row in primary + repeat
    )
    for plan in (primary, repeat):
        intervals = [
            (row["origin"], row["origin"] + row["horizon"])
            for row in plan
        ]
        assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def test_stage_two_has_no_llm_and_one_b_while_stage_three_uses_only_a():
    plan = revision.trajectory_plan()
    assert [
        (row["stage"], row["feedback_face"], row["llm_allowed"])
        for row in plan
    ] == [
        (1, "support_a", True),
        (2, "support_b", False),
        (3, "support_a", True),
    ]
    assert plan[1]["support_b_evaluations_per_arm"] == 1
    assert plan[1]["support_a_evaluations_per_arm"] == 0
    assert plan[2]["support_b_evaluations_per_arm"] == 0


def test_b8_is_one_whole_trajectory_ledger_per_arm_not_a_stage_reset():
    contract = revision.budget_contract()
    per_arm = contract["per_arm_whole_trajectory"]
    assert contract["operating_point"] == "B=8"
    assert per_arm == {
        "llm_call_max": 8,
        "token_max": 60_000,
        "full_consumer_fit_max": 8,
        "stage_1_support_a_max": 4,
        "stage_2_support_b_max": 1,
        "stage_3_support_a_max": 3,
        "cheap_probe_max": 24,
        "wall_seconds_max": 2_700,
        "accepted_update_max_by_arm": {
            "K0-fixed": 0,
            "A5-Slow": 1,
        },
        "counters_reset_between_stages": False,
    }
    assert contract["global"] == {
        "arms": 2,
        "llm_call_max": 16,
        "token_max": 120_000,
        "treatment_consumer_fit_max": 16,
        "identity_reference_fit_count": 3,
        "absolute_consumer_fit_max": 19,
        "cheap_probe_max": 48,
    }


def test_usage_validation_locks_stage_faces_and_the_shared_eight_call_cap():
    valid = {
        "llm_calls": 8,
        "llm_calls_by_stage": {"stage_1": 6, "stage_2": 0, "stage_3": 2},
        "tokens": 60_000,
        "full_consumer_fits": 8,
        "stage_1_support_a_fits": 4,
        "stage_2_support_b_fits": 1,
        "stage_3_support_a_fits": 3,
        "cheap_probes": 24,
        "accepted_updates": 1,
        "wall_seconds": 2_700,
    }
    assert revision.validate_usage("A5-Slow", valid) is True

    for path, value in (
        (("llm_calls",), 9),
        (("llm_calls_by_stage", "stage_2"), 1),
        (("tokens",), 60_001),
        (("full_consumer_fits",), 9),
        (("stage_1_support_a_fits",), 5),
        (("stage_2_support_b_fits",), 2),
        (("stage_3_support_a_fits",), 4),
        (("cheap_probes",), 25),
        (("accepted_updates",), 2),
        (("wall_seconds",), 2_700.001),
    ):
        assert revision.validate_usage(
            "A5-Slow", _set_path(valid, path, value)
        ) is False, path

    k0 = dict(valid, accepted_updates=0)
    assert revision.validate_usage("K0-fixed", k0) is True
    assert revision.validate_usage(
        "K0-fixed", dict(k0, accepted_updates=1)
    ) is False


def test_k0_and_a5_start_from_the_same_v1_with_only_writeback_different():
    assert revision.OLD_SKILL_ID == (
        "fast_winner_forecast_pooled_ridge_a1_smase_e1v2_outlier_iqr"
    )
    arms = revision.arm_contract()
    assert tuple(arms) == ("K0-fixed", "A5-Slow")
    assert arms["K0-fixed"]["initial_skill"] == arms["A5-Slow"][
        "initial_skill"
    ]
    assert arms["K0-fixed"]["initial_skill"] == {
        "skill_id": revision.OLD_SKILL_ID,
        "revision": 1,
        "program_steps": [{"op": "outlier_iqr", "params": {}}],
    }
    assert arms["K0-fixed"]["writeback_allowed"] is False
    assert arms["A5-Slow"]["writeback_allowed"] is True
    assert all(
        arm["raw_episode_input_to_fast"] is False for arm in arms.values()
    )
    assert revision.EDIT_SURFACE == (
        "skill_library.entries/"
        f"{revision.OLD_SKILL_ID}.body"
    )


def test_exact_body_program_patch_preserves_id_and_non_program_fields():
    result = revision.validate_patch_contract(
        before=_skill_v1(), after=_skill_v2(), event=_patch_event()
    )
    assert result["passed"] is True
    assert result["operation"] == "PATCH"
    assert result["target_surface_id"] == revision.EDIT_SURFACE
    assert result["changed_surfaces"] == [
        "body",
        "allowed_tools",
        "risk.frozen_plan.program",
    ]
    assert result["program_mirrors_consistent"] is True
    assert result["revision_before"] == 1
    assert result["revision_after"] == 2


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("add", "operation_not_patch"),
        ("scope_target", "target_not_exact_body"),
        ("no_op", "body_no_op"),
        ("scope_changed", "non_program_field_changed"),
        ("risk_changed", "non_program_field_changed"),
        ("allowed_tools_stale", "program_mirror_mismatch"),
        ("frozen_plan_stale", "program_mirror_mismatch"),
        ("revision_jump", "revision_not_incremented_by_one"),
        ("skill_id_changed", "skill_id_changed"),
    ],
)
def test_add_scope_risk_noop_and_invalid_versions_are_not_legal_program_patches(
    mutation: str, expected_reason: str
):
    before = _skill_v1()
    after = _skill_v2()
    event = _patch_event()
    if mutation == "add":
        event["operation"] = "ADD"
    elif mutation == "scope_target":
        event["target_surface_id"] = (
            f"skill_library.entries/{revision.OLD_SKILL_ID}.scope"
        )
    elif mutation == "no_op":
        after["body"] = copy.deepcopy(before["body"])
    elif mutation == "scope_changed":
        after["scope"] = {"task_kind": "classification"}
    elif mutation == "risk_changed":
        after["risk"]["current_task_support_confirmation_required"] = False
    elif mutation == "allowed_tools_stale":
        after["allowed_tools"] = ["outlier_iqr"]
    elif mutation == "frozen_plan_stale":
        after["risk"]["frozen_plan"]["program"] = "outlier_iqr"
    elif mutation == "revision_jump":
        after["revision"] = 3
    elif mutation == "skill_id_changed":
        after["skill_id"] = "replacement-skill"

    result = revision.validate_patch_contract(
        before=before, after=after, event=event
    )
    assert result["passed"] is False
    assert result["reason"] == expected_reason


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (
            ("budget_valid",),
            False,
            "BUDGET_INSTRUMENT_FAILURE__NO_SCIENTIFIC_VERDICT",
        ),
        (
            ("llm_budget_exhausted_before_completion",),
            True,
            "LLM_BUDGET_EXHAUSTED_BEFORE_CHAIN__H3_HELD",
        ),
        (
            ("old_skill_qualified",),
            False,
            "OLD_SKILL_QUALIFICATION_FAILED__H3_HELD",
        ),
        (
            ("stage_1", "fast_selected_v1"),
            False,
            "OLD_SKILL_NOT_CAUSALLY_USED__H3_HELD",
        ),
        (
            ("stage_1", "local_fault"),
            False,
            "NO_LOCAL_V1_FAULT__H3_HELD",
        ),
        (
            ("stage_1", "constrained_proposal_succeeds"),
            True,
            "PROGRAM_HYPOTHESIS_FALSIFIED__H3_HELD",
        ),
        (
            ("stage_1", "constrained_proposal_succeeds"),
            None,
            "SLOW_SAFE_ABSTAIN__H3_HELD",
        ),
        (
            ("stage_1", "slow_status"),
            "abstained",
            "SLOW_SAFE_ABSTAIN__H3_HELD",
        ),
        (
            ("stage_1", "patch_valid"),
            False,
            "UNAUTHORIZED_OR_INVALID_PATCH__H3_HELD",
        ),
        (
            ("stage_1", "support_replay_positive"),
            False,
            "PATCH_SUPPORT_REJECTED__H3_HELD",
        ),
        (
            ("stage_2", "support_b_relation"),
            "NEGATIVE",
            "INDEPENDENT_SUPPORT_B_REJECTED__H3_HELD",
        ),
        (
            ("stage_3", "a5_v2_causally_used"),
            False,
            "REVISED_SKILL_NOT_CAUSALLY_USED__H3_HELD",
        ),
        (
            ("stage_3", "v2_minus_v1_utility"),
            0.004999,
            "REVISION_REENCOUNTER_FAILED_ROLLED_BACK__H3_HELD",
        ),
        (
            ("stage_3", "a5_harm_count"),
            1,
            "REVISION_REENCOUNTER_FAILED_ROLLED_BACK__H3_HELD",
        ),
    ],
)
def test_terminal_branches_are_deterministic_and_keep_h3_held(
    path: tuple[str, ...], value: object, expected: str
):
    assert revision.derive_verdict(
        _set_path(_passing_evidence(), path, value)
    ) == expected


def test_budget_and_version_instrument_errors_precede_scientific_branches():
    evidence = _set_path(
        _passing_evidence(), ("stage_1", "fast_selected_v1"), False
    )
    evidence["budget_valid"] = False
    assert revision.derive_verdict(evidence) == (
        "BUDGET_INSTRUMENT_FAILURE__NO_SCIENTIFIC_VERDICT"
    )

    evidence = _set_path(
        _passing_evidence(), ("stage_1", "slow_status"), "abstained"
    )
    evidence = _set_path(
        evidence, ("stage_2", "version_chain_valid"), False
    )
    assert revision.derive_verdict(evidence) == (
        "VERSION_CHAIN_INVALID__NO_SCIENTIFIC_VERDICT"
    )


def test_complete_chain_pass_still_requires_the_prefrozen_replication():
    assert revision.derive_verdict(_passing_evidence()) == (
        "EXPOSED_DEV_SLOW_PATCH_CHAIN_PASS__REPLICATION_REQUIRED__H3_HELD"
    )


def test_failed_reencounter_rolls_back_v2_while_a_pass_keeps_it():
    v1 = _skill_v1()
    v2 = _skill_v2()

    failed = _set_path(
        _passing_evidence(), ("stage_3", "v2_minus_v1_utility"), 0.0
    )
    rolled_back = revision.apply_reencounter_outcome(
        v1_snapshot=v1, v2_snapshot=v2, evidence=failed
    )
    assert rolled_back["rolled_back"] is True
    assert rolled_back["active_snapshot"] is v1
    assert rolled_back["active_revision"] == 1
    assert rolled_back["new_revision_created"] is False
    assert rolled_back["verdict"] == (
        "REVISION_REENCOUNTER_FAILED_ROLLED_BACK__H3_HELD"
    )

    passed = revision.apply_reencounter_outcome(
        v1_snapshot=v1, v2_snapshot=v2, evidence=_passing_evidence()
    )
    assert passed["rolled_back"] is False
    assert passed["active_snapshot"] is v2
    assert passed["active_revision"] == 2
    assert passed["new_revision_created"] is False


def test_final_stays_zero_and_this_slice_adds_no_sha_or_manifest_layer():
    boundary = revision.boundary_contract()
    assert boundary == {
        "natural_final_outcome_reads": 0,
        "ucr_test_outcome_reads": 0,
        "sealed_ad_outcome_reads": 0,
        "noaa_2025_confirmation_reads": 0,
        "new_sha_added": False,
        "new_manifest_added": False,
        "p4_evolution_gate_before": "HELD",
        "p4_evolution_gate_after_single_pass": "HELD",
    }


def test_new_runner_reuses_the_existing_pre_backend_ninth_call_guard():
    """Backend guard behavior itself is covered by test_p4_llm_budget_guard."""
    contract = revision.budget_contract()
    assert contract["per_arm_whole_trajectory"]["llm_call_max"] == 8
    assert contract["ninth_call"] == {
        "reaches_backend": False,
        "budget_charged": False,
        "terminal_behavior": "BUDGET_EXHAUSTED_ABSTAIN_IDENTITY",
        "counter_scope": "whole_trajectory_per_arm",
    }


def test_old_skill_qualification_uses_exact_natural_lifecycle_fields():
    checks = revision._formation_checks()
    assert checks == {
        "natural_draft_written": True,
        "independent_promotion": True,
        "not_set_by_hand": True,
        "formation_target_noaa_development": True,
        "prior_same_domain_retrieval": True,
    }


@pytest.mark.parametrize(
    ("requested", "returned", "calls", "valid"),
    [
        ("gpt-5.6-sol", ("gpt-5.6-sol",), 8, True),
        ("gpt-5.6-sol", ("gpt-5.6-sol", "other"), 8, False),
        ("gpt-5.6-sol", ("other",), 1, False),
        ("gpt-5.6-sol", (), 1, False),
        ("gpt-5.6-sol", (), 0, True),
    ],
)
def test_returned_model_contract_is_fail_closed(
    requested: str, returned: tuple[str, ...], calls: int, valid: bool
):
    assert revision.validate_returned_models(
        requested_model=requested,
        returned_models=returned,
        calls=calls,
    ) is valid


def test_existing_output_is_never_overwritten_or_rerolled(tmp_path):
    output = tmp_path / "terminal.json"
    revision._refuse_existing_run_artifact(output)
    output.write_text('{"status":"FAILED"}\n', encoding="utf-8")
    with pytest.raises(revision.EvolutionRevisionBlocked, match="reroll"):
        revision._refuse_existing_run_artifact(output)


def test_fast_round_cannot_create_an_extra_skill_draft(monkeypatch):
    captured = {}

    def fake_round(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(revision, "run_online_round", fake_round)
    result, stopped = revision._run_fast(
        method=object(),
        state={"controller": object(), "store": object()},
        executor=object(),
        request=object(),
        features={},
        cell=type("Cell", (), {"values": {}})(),
        origin=revision.PRIMARY_ORIGINS[0],
        arm="A5-Slow",
        stage=1,
        budget=3,
    )
    assert stopped is False
    assert result is not None
    assert captured["allow_fast_skill"] is False


def test_unreached_stage_is_not_reported_as_budget_exhaustion():
    record = revision._result_record(
        None,
        object(),
        unavailable_reason="STAGE_NOT_REACHED__NO_APPROVED_V2",
    )
    assert record["abstained"] is True
    assert record["abstain_reason"] == "STAGE_NOT_REACHED__NO_APPROVED_V2"
