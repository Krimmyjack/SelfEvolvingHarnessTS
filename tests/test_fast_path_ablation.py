import numpy as np

from SelfEvolvingHarnessTS.fast_path.ablation import (
    FastPathAblationArm,
    run_fast_path_ablation,
    summarize_fast_path_ablation_results,
    write_fast_path_ablation_report,
)
from SelfEvolvingHarnessTS.memory.evidence_schema import build_memory_evidence, build_memory_evidence_v2
from SelfEvolvingHarnessTS.policy.action_spec import action_menu_v1
from SelfEvolvingHarnessTS.policy.skill_memory_composer import TypedCandidate


def _record(uid="r1"):
    return {
        "uid": uid,
        "cell": "forecast|snrHigh|full",
        "snr": 12.0,
        "miss_rate": 0.0,
        "X_p": [24.0, 0.05, 0.1, 0.1, 0.0, 0.1, 0.2, 0.05],
    }


def _memory():
    return build_memory_evidence(
        task="forecast",
        pattern_region="forecast|snrHigh|full",
        skill_id="median_smooth",
        action_id="v_median",
        program={"steps": []},
        raw_loss=2.0,
        selected_loss=1.7,
        support={"n": 3},
        provenance={"case_id": "m1"},
    )


def _risk_memory(action_id="v_median"):
    return build_memory_evidence_v2(
        task="forecast",
        pattern_region="forecast|snrHigh|full",
        memory_type="risk",
        role="ban",
        skill_id="median_smooth",
        action_id=action_id,
        program={"steps": []},
        utility_delta_vs_raw=-0.4,
        harm_delta_vs_raw=0.4,
        support={"n": 2},
        evidence_refs=("risk:m1",),
        provenance={"case_id": "risk_m1"},
    )


def test_raw_ablation_arm_executes_v_none_as_baseline_not_fallback():
    x = np.linspace(0.0, 1.0, 32)

    results = run_fast_path_ablation(
        [_record()],
        {"r1": x},
        arms=[FastPathAblationArm.raw()],
        action_menu=action_menu_v1(),
    )

    assert len(results) == 1
    result = results[0]
    assert result.arm_name == "raw"
    assert result.decision.route == "raw"
    assert result.decision.action_id == "v_none"
    assert result.executed.status == "executed"
    assert result.validation.passed is True
    assert result.evidence.verification_result["passed"] is True
    assert result.evidence.routing["ablation_arm"] == "raw"


def test_memory_only_ablation_arm_removes_skill_surface_but_keeps_memory():
    captured = []

    def composer(packet):
        captured.append(packet)
        return TypedCandidate(skill_id=None, action_id="v_median", rationale="memory_only_stub")

    arm = FastPathAblationArm(
        name="memory_only_selector",
        use_skills=False,
        use_memory=True,
        use_composer=True,
        use_safety=False,
        composer=composer,
        support_stats={"support_score": 0.1, "needs_composition": True},
    )

    results = run_fast_path_ablation(
        [_record()],
        {"r1": np.sin(np.linspace(0, 2 * np.pi, 64))},
        arms=[arm],
        action_menu=action_menu_v1(),
        memory_by_uid={"r1": [_memory()]},
    )

    assert len(captured) == 1
    assert captured[0]["skills"] == []
    assert captured[0]["memory"]["prior_fragments"][0]["schema"] == "memory_evidence_v1"
    assert results[0].decision.composer_called is True
    assert results[0].decision.action_id == "v_median"
    assert results[0].evidence.routing["ablation_arm"] == "memory_only_selector"


def test_positive_memory_arm_filters_out_risk_memory_before_composer():
    captured = []

    def composer(packet):
        captured.append(packet)
        return TypedCandidate(skill_id=None, action_id="v_median", rationale="positive_memory_stub")

    arm = FastPathAblationArm(
        name="positive_memory_only",
        use_skills=False,
        use_memory=True,
        use_composer=True,
        use_safety=False,
        composer=composer,
        support_stats={"support_score": 0.1, "needs_composition": True},
        memory_mode="positive",
    )

    run_fast_path_ablation(
        [_record()],
        {"r1": np.sin(np.linspace(0, 2 * np.pi, 64))},
        arms=[arm],
        action_menu=action_menu_v1(),
        memory_by_uid={"r1": [_memory(), _risk_memory()]},
    )

    assert len(captured) == 1
    memory = captured[0]["memory"]
    assert len(memory["utility_memory"]) == 1
    assert memory["utility_memory"][0]["action_id"] == "v_median"
    assert memory["risk_memory"] == []
    assert memory["prior_fragments"] == []


def test_risk_memory_only_arm_abstains_to_raw_and_records_memory_mode():
    arm = FastPathAblationArm(
        name="risk_memory_only",
        use_skills=False,
        use_memory=True,
        use_composer=True,
        use_safety=True,
        composer=lambda packet: TypedCandidate(
            action_id="v_none",
            abstain_to_raw=True,
            rationale="risk_memory_only_abstain",
            evidence_refs=("risk:m1",),
        ),
        support_stats={"support_score": 0.1, "needs_composition": True},
        memory_mode="risk",
    )

    results = run_fast_path_ablation(
        [_record()],
        {"r1": np.sin(np.linspace(0, 2 * np.pi, 64))},
        arms=[arm],
        action_menu=action_menu_v1(),
        memory_by_uid={"r1": [_memory(), _risk_memory()]},
    )

    result = results[0]
    assert result.decision.candidate.abstain_to_raw is True
    assert result.decision.safety.accepted is False
    assert "candidate_abstain_to_raw" in result.decision.safety.reasons
    assert result.executed.status == "raw_fallback_not_compiled"
    assert result.evidence.routing["ablation_flags"]["memory_mode"] == "risk"


def test_ablation_summary_and_report_are_reproducible(tmp_path):
    def composer(packet):
        return TypedCandidate(skill_id=None, action_id="v_median", rationale="stub_report")

    arms = [
        FastPathAblationArm.raw(),
        FastPathAblationArm(
            name="memory_only_selector",
            use_skills=False,
            use_memory=True,
            use_composer=True,
            use_safety=False,
            composer=composer,
            support_stats={"support_score": 0.1, "needs_composition": True},
        ),
    ]

    results = run_fast_path_ablation(
        [_record()],
        {"r1": np.sin(np.linspace(0, 2 * np.pi, 64))},
        arms=arms,
        action_menu=action_menu_v1(),
        memory_by_uid={"r1": [_memory()]},
    )

    summary = summarize_fast_path_ablation_results(results)
    assert summary["n_results"] == 2
    assert summary["arm_order"] == ["raw", "memory_only_selector"]
    assert summary["arms"]["raw"]["n"] == 1
    assert summary["arms"]["raw"]["action_counts"] == {"v_none": 1}
    assert summary["arms"]["memory_only_selector"]["composer_called"] == 1
    assert summary["arms"]["memory_only_selector"]["action_counts"] == {"v_median": 1}

    report = write_fast_path_ablation_report(results, tmp_path, metadata={"slice": "unit"})
    assert report["metadata"]["slice"] == "unit"
    assert (tmp_path / "report.json").exists()
    records = (tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 2
    assert '"arm_name": "raw"' in records[0]


def test_ablation_summary_reports_lift_against_raw_arm_by_uid():
    def composer(packet):
        return TypedCandidate(skill_id=None, action_id="v_median", rationale="stub_lift")

    def validator(raw, artifact, context):
        action_id = context["decision"].action_id
        utility_by_action = {"v_none": 0.25, "v_median": 0.75}
        utility = utility_by_action[action_id]
        return {
            "validator": "fixed_utility_stub",
            "passed": True,
            "utility_delta_vs_raw": utility,
            "harm_delta_vs_raw": 0.0,
        }

    arms = [
        FastPathAblationArm.raw(),
        FastPathAblationArm(
            name="memory_only_selector",
            use_skills=False,
            use_memory=True,
            use_composer=True,
            use_safety=False,
            composer=composer,
            support_stats={"support_score": 0.1, "needs_composition": True},
        ),
    ]

    results = run_fast_path_ablation(
        [_record()],
        {"r1": np.sin(np.linspace(0, 2 * np.pi, 64))},
        arms=arms,
        action_menu=action_menu_v1(),
        memory_by_uid={"r1": [_memory()]},
        validator=validator,
    )

    summary = summarize_fast_path_ablation_results(results)

    assert summary["reference_arm"] == "raw"
    assert summary["arms"]["raw"]["mean_lift_vs_raw_arm"] == 0.0
    assert summary["arms"]["memory_only_selector"]["mean_lift_vs_raw_arm"] == 0.5


def test_ablation_summary_reports_phase4_safety_and_serve_metrics():
    def validator(raw, artifact, context):
        return {
            "validator": "fixed_stub",
            "passed": bool(context["executed"].execution_ok),
            "utility_delta_vs_raw": 0.0,
            "harm_delta_vs_raw": 0.0,
        }

    arms = [
        FastPathAblationArm.raw(),
        FastPathAblationArm(
            name="risk_memory_only",
            use_skills=False,
            use_memory=True,
            use_composer=True,
            use_safety=True,
            composer=lambda packet: TypedCandidate(
                action_id="v_none",
                abstain_to_raw=True,
                rationale="risk_memory_only_abstain",
                evidence_refs=("risk:m1",),
            ),
            support_stats={"support_score": 0.1, "needs_composition": True},
            memory_mode="risk",
        ),
    ]

    results = run_fast_path_ablation(
        [_record()],
        {"r1": np.sin(np.linspace(0, 2 * np.pi, 64))},
        arms=arms,
        action_menu=action_menu_v1(),
        memory_by_uid={"r1": [_risk_memory()]},
        validator=validator,
    )

    summary = summarize_fast_path_ablation_results(results)

    assert summary["arms"]["raw"]["serve_fraction"] == 1.0
    risk_arm = summary["arms"]["risk_memory_only"]
    assert risk_arm["serve_fraction"] == 0.0
    assert risk_arm["fallback_fraction"] == 1.0
    assert risk_arm["abstain_to_raw"] == 1
    assert risk_arm["safety_reason_counts"]["candidate_abstain_to_raw"] == 1
