from __future__ import annotations

from types import SimpleNamespace

from evaluation.main_protocol_p4 import run_forecast_p4_evolution_dev as dev


def test_preflight_keeps_final_closed_and_uses_frozen_programs():
    result = dev.preflight()
    assert result["status"] == "PASS"
    assert result["checks"]["natural_final_reads_zero"] is True
    assert [
        option["program_steps"][0]["op"]
        for option in dev.typed_patch_options()
    ] == list(dev.performance.PARALLEL_PROGRAMS)


def test_next_fast_gate_reads_chosen_steps_not_runtime_winner():
    pending = (("winsorize", {"lower": 0.05, "upper": 0.95}),)
    trace = SimpleNamespace(
        chosen_candidate_id="autonomous",
        candidate_program_steps={
            "autonomous": (("hampel_filter", {"window": 7}),),
            "cand_skill_revision": pending,
        },
    )
    method = SimpleNamespace(last_trace=trace)
    assert dev._trial_fast_selects_pending(method, pending) is False

    trace.chosen_candidate_id = "cand_skill_revision"
    assert dev._trial_fast_selects_pending(method, pending) is True


def test_actual_winner_must_come_from_edited_skill():
    result = SimpleNamespace(
        _winner_candidate_id="cand_skill_forecast_local"
    )
    assert dev._winner_uses_skill(result, "forecast_local") is True

    result._winner_candidate_id = "autonomous"
    assert dev._winner_uses_skill(result, "forecast_local") is False
    assert dev._winner_uses_skill(result, None) is False


def test_actual_winner_must_be_the_fast_choice_with_matching_steps():
    steps = (("winsorize", {"lower": 0.05}),)
    trace = SimpleNamespace(
        chosen_candidate_id="cand_skill_forecast_local",
        candidate_program_steps={
            "cand_skill_forecast_local": steps,
            "autonomous": steps,
        },
    )
    method = SimpleNamespace(last_trace=trace)
    result = SimpleNamespace(
        _winner_candidate_id="autonomous",
        _winner_steps=steps,
    )
    assert dev._winner_is_fast_choice(result, method) is False
    assert dev._winner_matches_trace_steps(result, method) is True

    result._winner_candidate_id = "cand_skill_forecast_local"
    assert dev._winner_is_fast_choice(result, method) is True
    assert dev._winner_matches_trace_steps(result, method) is True


def test_only_cell_llm_exhaustion_can_become_identity_abstain():
    cell = dev.shared_harness.Stop(
        dev.performance.CELL_LLM_EXHAUSTION_VERDICT, "cell cap"
    )
    global_cap = dev.shared_harness.Stop(
        "LLM_BUDGET_EXCEEDED", "global cap"
    )
    assert dev._is_cell_llm_exhaustion(cell) is True
    assert dev._is_cell_llm_exhaustion(global_cap) is False


def test_budget_failure_is_not_execution_complete():
    assert dev._budget_terminal_status(True) == "COMPLETE"
    assert dev._budget_terminal_status(False) == "FAILED"


def test_versioned_revision_requires_same_skill_increment_by_one():
    before = SimpleNamespace(
        skills=(SimpleNamespace(skill_id="forecast_local", revision=2),)
    )
    after = SimpleNamespace(
        skills=(SimpleNamespace(skill_id="forecast_local", revision=3),)
    )
    evidence = dev._versioned_revision(
        before,
        after,
        {
            "operation": "PATCH",
            "target_surface_id": (
                "skill_library.entries/forecast_local.body"
            ),
        },
    )
    assert evidence == {
        "passed": True,
        "operation": "PATCH",
        "target_skill_id": "forecast_local",
        "revision_before": 2,
        "revision_after": 3,
    }

    skipped = SimpleNamespace(
        skills=(SimpleNamespace(skill_id="forecast_local", revision=4),)
    )
    assert dev._versioned_revision(
        before,
        skipped,
        {
            "operation": "PATCH",
            "target_surface_id": (
                "skill_library.entries/forecast_local.body"
            ),
        },
    )["passed"] is False


def test_usage_gate_checks_every_frozen_dimension():
    valid = {
        "support_a_fits": dev.MAX_SUPPORT_A,
        "support_b_fits": dev.MAX_SUPPORT_B,
        "cheap_probes": dev.performance.MAX_CHEAP_PROBES,
        "llm_calls": dev.MAX_LLM_CALLS,
        "tokens": dev.MAX_TOKENS,
        "accepted_updates": dev.performance.MAX_UPDATES,
        "wall_seconds": dev.MAX_WALL_SECONDS,
    }
    assert all(dev._usage_checks(valid).values())
    invalid = dict(valid, tokens=dev.MAX_TOKENS + 1)
    assert dev._usage_checks(invalid)["tokens_within_cap"] is False


def test_verdict_keeps_h3_held_on_independent_b_rejection():
    payload = {
        "chain": {
            "first_fault": {
                "harm_count": 1,
                "slow_event": {"stage": "pending"},
            },
            "next_fast_used_pending_harness": True,
            "support_b_approved": False,
        }
    }
    assert dev._verdict(payload) == (
        "INDEPENDENT_SUPPORT_B_REJECTED__H3_HELD"
    )
