import numpy as np

from SelfEvolvingHarnessTS.fast_path.ablation import (
    FastPathAblationArm,
    run_fast_path_ablation,
    summarize_fast_path_ablation_results,
    write_fast_path_ablation_report,
)
from SelfEvolvingHarnessTS.memory.evidence_schema import build_memory_evidence
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
