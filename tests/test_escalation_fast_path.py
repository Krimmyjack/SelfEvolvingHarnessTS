import numpy as np
from SelfEvolvingHarnessTS.memory import EvidenceStore
from SelfEvolvingHarnessTS.memory.evidence_schema import build_memory_evidence
from SelfEvolvingHarnessTS.policy.escalation import (
    EscalationConfig,
    compile_fast_path_decision,
    decide_fast_path,
    execute_fast_path_decision,
    emit_fast_path_evidence,
    validate_fast_path_output,
)
from SelfEvolvingHarnessTS.policy.action_spec import action_menu_v1
from SelfEvolvingHarnessTS.policy.skill_memory_composer import TypedCandidate


def _record(*, uid="r1", snr=12.0, miss_rate=0.0):
    return {
        "uid": uid,
        "cell": "forecast|snrHigh|full",
        "snr": snr,
        "miss_rate": miss_rate,
        "X_p": [24.0, 0.05, 0.1, 0.1, 0.0, 0.1, 0.2, 0.05],
    }


def _menu(*allowed):
    return {"version": "test", "allowed_actions": list(allowed)}


def test_high_support_known_region_uses_deterministic_skill_without_composer():
    calls = []

    def composer(packet):
        calls.append(packet)
        return TypedCandidate(skill_id="median_smooth", action_id="v_median")

    decision = decide_fast_path(
        _record(),
        action_menu_meta=_menu("v_none", "v_median"),
        support_stats={"support_score": 0.1},
        config=EscalationConfig(max_support_score=0.5, min_deterministic_skill_score=0.55),
        composer=composer,
    )

    assert calls == []
    assert decision.route == "deterministic"
    assert decision.proposal_route == "deterministic"
    assert decision.action_id == "v_none"
    assert decision.candidate.skill_id == "identity"
    assert decision.safety.accepted is True


def test_weak_support_calls_composer_then_safety_falls_back_to_raw():
    calls = []

    def composer(packet):
        calls.append(packet)
        return TypedCandidate(skill_id="median_smooth", action_id="v_median")

    decision = decide_fast_path(
        _record(uid="weak", snr=-4.0, miss_rate=0.2),
        action_menu_meta=_menu("v_none", "v_median", "f0_median_w9"),
        support_stats={"support_score": 0.9},
        config=EscalationConfig(max_support_score=0.5, min_deterministic_skill_score=0.55),
        composer=composer,
    )

    assert len(calls) == 1
    assert decision.proposal_route == "llm_composer"
    assert decision.route == "raw_fallback"
    assert decision.action_id == "v_none"
    assert "weak_support" in decision.safety.reasons


def test_packet_contains_retrieved_skills_and_utility_bound_memory_rows():
    captured = {}
    memory = build_memory_evidence(
        task="forecast",
        pattern_region="forecast|snrLow|miss",
        skill_id="median_smooth",
        action_id="v_median",
        program={"steps": [{"op": "denoise_median", "params": {"window": 5}}]},
        raw_loss=2.0,
        selected_loss=1.5,
        support={"n": 7, "radius": 0.2},
        provenance={"case_id": "m1", "raw_loss": 2.0},
    )

    def composer(packet):
        captured["packet"] = packet
        return TypedCandidate(skill_id="median_smooth", action_id="v_median")

    decide_fast_path(
        _record(uid="memory", snr=-4.0, miss_rate=0.2),
        action_menu_meta=_menu("v_none", "v_median"),
        memory_rows=[memory],
        support_stats={"support_score": 0.9},
        config=EscalationConfig(max_support_score=0.5),
        composer=composer,
    )

    packet = captured["packet"]
    assert packet["schema"] == "skill_memory_evidence_packet_v1"
    assert packet["skills"][0]["rank"] == 1
    assert packet["memory"]["prior_fragments"][0]["schema"] == "memory_evidence_v1"
    assert packet["memory"]["prior_fragments"][0]["utility_delta_vs_raw"] == 0.5
    assert "raw_loss" not in str(packet)


def test_abstain_candidate_forces_raw_even_with_good_support():
    def composer(packet):
        return TypedCandidate(
            risk_rule={"rule_id": "conflict_abstain", "op": "abstain"},
            abstain_to_raw=True,
        )

    decision = decide_fast_path(
        _record(uid="conflict", snr=10.0, miss_rate=0.0),
        action_menu_meta=_menu("v_none", "v_median"),
        support_stats={"support_score": 0.1, "evidence_conflict": True},
        config=EscalationConfig(max_support_score=0.5),
        composer=composer,
    )

    assert decision.proposal_route == "llm_composer"
    assert decision.route == "raw_fallback"
    assert decision.action_id == "v_none"
    assert "candidate_abstain_to_raw" in decision.safety.reasons


def test_skill_action_mismatch_is_rejected_by_safety_gate():
    def composer(packet):
        return TypedCandidate(skill_id="identity", action_id="v_median")

    decision = decide_fast_path(
        _record(uid="bad_skill", snr=10.0, miss_rate=0.0),
        action_menu_meta=_menu("v_none", "v_median"),
        support_stats={"support_score": 0.1, "evidence_conflict": True},
        config=EscalationConfig(max_support_score=0.5),
        composer=composer,
    )

    assert decision.route == "raw_fallback"
    assert decision.action_id == "v_none"
    assert "skill_action_mismatch" in decision.safety.reasons

def test_accepted_candidate_compiles_to_action_program():
    menu = action_menu_v1()
    record = _record(uid="compile", snr=-4.0, miss_rate=0.2)
    decision = decide_fast_path(
        record,
        action_menu_meta=menu.to_dict(),
        support_stats={"support_score": 0.1},
        config=EscalationConfig(max_support_score=0.5, min_deterministic_skill_score=0.55),
    )

    compiled = compile_fast_path_decision(decision, record, menu)

    assert compiled.compiled is True
    assert compiled.action_id == decision.action_id
    assert compiled.program is not None
    assert compiled.program.to_dict()["note"] == f"tmpl:{decision.action_id}"


def test_raw_fallback_decision_does_not_compile_rejected_action_by_default():
    menu = action_menu_v1()

    def composer(packet):
        return TypedCandidate(skill_id="median_smooth", action_id="v_median")

    record = _record(uid="fallback_compile", snr=-4.0, miss_rate=0.2)
    decision = decide_fast_path(
        record,
        action_menu_meta=menu.to_dict(),
        support_stats={"support_score": 0.9},
        config=EscalationConfig(max_support_score=0.5),
        composer=composer,
    )

    compiled = compile_fast_path_decision(decision, record, menu)

    assert decision.route == "raw_fallback"
    assert compiled.compiled is False
    assert compiled.action_id == "v_none"
    assert compiled.program is None
    assert compiled.reason == "raw_fallback_not_compiled"

def test_execute_bridge_runs_accepted_program_and_returns_artifact():
    menu = action_menu_v1()
    record = _record(uid="execute", snr=-4.0, miss_rate=0.2)
    decision = decide_fast_path(
        record,
        action_menu_meta=menu.to_dict(),
        support_stats={"support_score": 0.1},
        config=EscalationConfig(max_support_score=0.5, min_deterministic_skill_score=0.55),
    )
    x = np.sin(np.linspace(0, 4 * np.pi, 96))
    x[10:12] = np.nan

    executed = execute_fast_path_decision(decision, record, menu, x)

    assert executed.status == "executed"
    assert executed.compiled.compiled is True
    assert executed.execution_ok is True
    assert executed.artifact.shape == x.shape
    assert np.all(np.isfinite(executed.artifact))


def test_execute_bridge_raw_fallback_returns_original_series_without_execution():
    menu = action_menu_v1()

    def composer(packet):
        return TypedCandidate(skill_id="median_smooth", action_id="v_median")

    record = _record(uid="execute_fallback", snr=-4.0, miss_rate=0.2)
    decision = decide_fast_path(
        record,
        action_menu_meta=menu.to_dict(),
        support_stats={"support_score": 0.9},
        config=EscalationConfig(max_support_score=0.5),
        composer=composer,
    )
    x = np.linspace(0.0, 1.0, 32)

    executed = execute_fast_path_decision(decision, record, menu, x)

    assert executed.status == "raw_fallback_not_compiled"
    assert executed.compiled.compiled is False
    assert executed.execution_ok is False
    assert np.array_equal(executed.artifact, x)

def test_validator_and_writeback_records_executed_ready_evidence():
    menu = action_menu_v1()
    store = EvidenceStore()
    record = _record(uid="writeback", snr=-4.0, miss_rate=0.2)
    decision = decide_fast_path(
        record,
        action_menu_meta=menu.to_dict(),
        support_stats={"support_score": 0.1},
        config=EscalationConfig(max_support_score=0.5, min_deterministic_skill_score=0.55),
    )
    x = np.sin(np.linspace(0, 4 * np.pi, 96))
    x[10:12] = np.nan
    executed = execute_fast_path_decision(decision, record, menu, x)

    validation = validate_fast_path_output(x, executed, task_type="forecast")
    evidence = emit_fast_path_evidence(
        record,
        decision,
        executed,
        validation,
        store=store,
        batch_id="b-writeback",
        harness_version=7,
    )

    assert len(store) == 1
    assert store.query_by_cell(record["cell"])[0] is evidence
    assert evidence.cell_id == record["cell"]
    assert evidence.harness_version == 7
    assert evidence.program["note"] == f"tmpl:{decision.action_id}"
    assert evidence.verification_result["passed"] is True
    assert evidence.verification_result["output_status"] == "executed"
    assert evidence.verification_result["downstream"]["validator"] == "role_b_proxy"
    assert evidence.routing["route"] == "deterministic"
    assert evidence.routing["safety"]["accepted"] is True


def test_writeback_marks_raw_fallback_not_passed_and_does_not_store_rejected_program():
    menu = action_menu_v1()
    store = EvidenceStore()

    def composer(packet):
        return TypedCandidate(skill_id="median_smooth", action_id="v_median")

    record = _record(uid="fallback_writeback", snr=-4.0, miss_rate=0.2)
    decision = decide_fast_path(
        record,
        action_menu_meta=menu.to_dict(),
        support_stats={"support_score": 0.9},
        config=EscalationConfig(max_support_score=0.5),
        composer=composer,
    )
    x = np.linspace(0.0, 1.0, 32)
    executed = execute_fast_path_decision(decision, record, menu, x)

    validation = validate_fast_path_output(x, executed, task_type="forecast")
    evidence = emit_fast_path_evidence(record, decision, executed, validation, store=store)

    assert len(store) == 1
    assert evidence.verification_result["passed"] is False
    assert evidence.verification_result["output_status"] == "raw_fallback_not_compiled"
    assert "weak_support" in evidence.verification_result["failure_signature"]
    assert evidence.program["source"] == "raw_fallback"
    assert evidence.program["steps"] == []
    assert evidence.execution_trace == []
    assert evidence.routing["candidate"]["action_id"] == "v_median"
    assert evidence.routing["selected_action"] == "v_none"


def test_custom_downstream_validator_payload_is_embedded_in_evidence():
    menu = action_menu_v1()
    record = _record(uid="custom_validator", snr=-4.0, miss_rate=0.2)
    decision = decide_fast_path(
        record,
        action_menu_meta=menu.to_dict(),
        support_stats={"support_score": 0.1},
        config=EscalationConfig(max_support_score=0.5, min_deterministic_skill_score=0.55),
    )
    x = np.sin(np.linspace(0, 2 * np.pi, 64))
    executed = execute_fast_path_decision(decision, record, menu, x)

    def validator(raw, artifact, context):
        return {
            "passed": True,
            "validator": "stub_downstream",
            "utility_delta_vs_raw": 0.25,
            "harm_delta_vs_raw": 0.0,
            "context_action": context["decision"].action_id,
        }

    validation = validate_fast_path_output(x, executed, task_type="forecast", validator=validator)
    evidence = emit_fast_path_evidence(record, decision, executed, validation)

    assert validation.passed is True
    assert evidence.verification_result["downstream"]["validator"] == "stub_downstream"
    assert evidence.verification_result["downstream"]["utility_delta_vs_raw"] == 0.25
    assert evidence.verification_result["downstream"]["context_action"] == decision.action_id

def test_mapping_candidate_with_unknown_skill_is_rejected_to_raw():
    def composer(packet):
        return {"skill_id": "unknown_skill", "action_id": "v_median"}

    decision = decide_fast_path(
        _record(uid="unknown_mapping", snr=10.0, miss_rate=0.0),
        action_menu_meta=_menu("v_none", "v_median"),
        support_stats={"support_score": 0.1, "evidence_conflict": True},
        config=EscalationConfig(max_support_score=0.5),
        composer=composer,
    )

    assert decision.route == "raw_fallback"
    assert decision.action_id == "v_none"
    assert "unknown_skill" in decision.safety.reasons


def test_escalation_api_is_exported_from_policy_package():
    from SelfEvolvingHarnessTS import policy

    assert policy.decide_fast_path is decide_fast_path
    assert policy.execute_fast_path_decision is execute_fast_path_decision
    assert policy.validate_fast_path_output is validate_fast_path_output
    assert policy.emit_fast_path_evidence is emit_fast_path_evidence
