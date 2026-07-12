import json

import pytest

from SelfEvolvingHarnessTS.run_fast_path_ablation import run_demo_forecast_ablation


PHASE4_ARM_ORDER = [
    "raw",
    "deterministic_router",
    "skill_only_deterministic",
    "positive_memory_only",
    "risk_memory_only",
    "positive_risk_memory",
    "skill_memory_deterministic",
    "composer_skill",
    "composer_skill_memory",
    "composer_skill_memory_safety",
]


def test_run_demo_forecast_ablation_writes_no_api_report(tmp_path):
    report = run_demo_forecast_ablation(out_dir=tmp_path, n_records=2)

    assert report["metadata"]["slice"] == "synthetic_forecast_small"
    assert report["metadata"]["api_calls"] == 0
    assert report["metadata"]["reporter"] == "synthetic_oracle_proxy_v1"
    assert report["metadata"]["ablation_matrix"] == "phase4_memory_v2"
    assert report["summary"]["n_results"] == 20
    assert report["summary"]["reference_arm"] == "raw"
    assert report["summary"]["arm_order"] == PHASE4_ARM_ORDER
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "records.jsonl").exists()
    assert (tmp_path / "slow_path_proposals.jsonl").exists()
    assert len((tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 20
    assert report["summary"]["arms"]["raw"]["mean_lift_vs_raw_arm"] == 0.0
    assert report["summary"]["arms"]["positive_memory_only"]["mean_utility_delta_vs_raw"] is not None
    assert report["summary"]["arms"]["positive_memory_only"]["mean_lift_vs_raw_arm"] is not None
    assert report["summary"]["arms"]["risk_memory_only"]["fallback_fraction"] >= 0.0
    assert report["slow_path"]["n_proposals"] == 0
    assert report["slow_path"]["n_promotion_accepted"] == 0


def test_run_demo_forecast_ablation_promotes_only_independent_support(tmp_path):
    report = run_demo_forecast_ablation(out_dir=tmp_path, n_records=4)

    assert report["summary"]["n_results"] == 40
    assert report["metadata"]["api_calls"] == 0
    assert report["slow_path"]["min_support"] == 2
    assert report["slow_path"]["n_proposals"] > 0
    assert report["slow_path"]["n_promotion_accepted"] > 0

    rows = [
        json.loads(line)
        for line in (tmp_path / "slow_path_proposals.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        support = row["proposal"]["support"]
        assert support["n_unique_cases"] >= report["slow_path"]["min_support"]


def _ledger_records():
    return [
        {
            "uid": "r0",
            "cell": "forecast|snrLow|miss",
            "origin": "S_a",
            "snr": -4.0,
            "miss_rate": 0.2,
            "X_p": [24.0, 0.05, 0.1, 0.1, 0.0, 0.1, 0.2, 0.05],
            "L_test": {"v_none": 1.0, "v_median": 0.7, "v_winsor": 1.2},
        },
        {
            "uid": "r1",
            "cell": "forecast|snrLow|miss",
            "origin": "S_a",
            "snr": -5.0,
            "miss_rate": 0.1,
            "X_p": [24.0, 0.05, 0.1, 0.1, 0.0, 0.1, 0.2, 0.05],
            "L_test": {"v_none": 1.0, "v_median": 0.8, "v_winsor": 0.9},
        },
        {
            "uid": "r2",
            "cell": "forecast|snrHigh|full",
            "origin": "S_b",
            "snr": 12.0,
            "miss_rate": 0.0,
            "X_p": [24.0, 0.05, 0.1, 0.1, 0.0, 0.1, 0.2, 0.05],
            "L_test": {"v_none": 0.5, "v_median": 0.6, "v_winsor": 0.7},
        },
    ]


def test_oracle_ledger_inputs_use_only_prior_same_cell_memory():
    from SelfEvolvingHarnessTS.run_fast_path_ablation import build_oracle_ledger_ablation_inputs

    records, series_by_uid, memory_by_uid, losses_by_uid = build_oracle_ledger_ablation_inputs(
        _ledger_records(),
        n_records=3,
    )

    assert [row["uid"] for row in records] == ["r0", "r1", "r2"]
    assert set(series_by_uid) == {"r0", "r1", "r2"}
    assert losses_by_uid["r1"]["v_median"] == 0.8
    assert memory_by_uid["r0"] == []
    assert memory_by_uid["r2"] == []
    assert len(memory_by_uid["r1"]) == 2
    packet_rows = [row.to_packet_row() for row in memory_by_uid["r1"]]
    utility_rows = [row for row in packet_rows if row["memory_type"] == "utility"]
    risk_rows = [row for row in packet_rows if row["memory_type"] == "risk"]
    assert utility_rows[0]["action_id"] == "v_median"
    assert utility_rows[0]["utility_delta_vs_raw"] == pytest.approx(0.3)
    assert utility_rows[0]["provenance"]["source_uid"] == "r0"
    assert utility_rows[0]["provenance"]["source_uid"] != "r1"
    assert risk_rows[0]["action_id"] == "v_winsor"
    assert risk_rows[0]["harm_delta_vs_raw"] == pytest.approx(0.2)
    assert risk_rows[0]["provenance"]["source_uid"] == "r0"


def test_ledger_oracle_validator_scores_selected_action_from_l_test():
    from SelfEvolvingHarnessTS.policy.escalation import EscalationDecision, SafetyGateDecision
    from SelfEvolvingHarnessTS.policy.skill_memory_composer import TypedCandidate
    from SelfEvolvingHarnessTS.run_fast_path_ablation import ledger_oracle_validator

    decision = EscalationDecision(
        route="deterministic",
        proposal_route="deterministic",
        action_id="v_median",
        candidate=TypedCandidate(action_id="v_median"),
        packet={"provenance": {"source_uid": "r0"}},
        safety=SafetyGateDecision(accepted=True, serve_action_id="v_median", fallback_raw=False),
    )
    validator = ledger_oracle_validator({"r0": {"v_none": 1.0, "v_median": 0.7, "bad": 1.2}})

    result = validator([], [], {"decision": decision, "executed": type("E", (), {"execution_ok": True})()})

    assert result["validator"] == "ledger_l_test_oracle_v1"
    assert result["raw_loss"] == pytest.approx(1.0)
    assert result["selected_loss"] == pytest.approx(0.7)
    assert result["utility_delta_vs_raw"] == pytest.approx(0.3)
    assert result["harm_delta_vs_raw"] == pytest.approx(0.0)
    assert result["oracle_action"] == "v_median"
    assert result["regret_vs_oracle"] == pytest.approx(0.0)
    assert result["passed"] is True


def test_run_oracle_ledger_ablation_writes_no_api_report(tmp_path):
    from SelfEvolvingHarnessTS.run_fast_path_ablation import run_oracle_ledger_ablation

    records_path = tmp_path / "records.jsonl"
    records_path.write_text("\n".join(json.dumps(row) for row in _ledger_records()), encoding="utf-8")

    report = run_oracle_ledger_ablation(records_path=records_path, out_dir=tmp_path / "out", n_records=3)

    assert report["metadata"]["slice"] == "s2_oracle_ledger_small"
    assert report["metadata"]["api_calls"] == 0
    assert report["metadata"]["reporter"] == "ledger_l_test_oracle_v1"
    assert report["metadata"]["ablation_matrix"] == "phase4_memory_v2"
    assert report["summary"]["n_results"] == 30
    assert report["summary"]["reference_arm"] == "raw"
    assert report["summary"]["arm_order"] == PHASE4_ARM_ORDER
    assert report["summary"]["arms"]["raw"]["mean_lift_vs_raw_arm"] == 0.0
    assert (tmp_path / "out" / "report.json").exists()
    assert (tmp_path / "out" / "records.jsonl").exists()
    assert (tmp_path / "out" / "slow_path_proposals.jsonl").exists()
    report_text = (tmp_path / "out" / "report.json").read_text(encoding="utf-8")
    rows = (tmp_path / "out" / "records.jsonl").read_text(encoding="utf-8")
    assert "NaN" not in report_text
    assert "NaN" not in rows
    assert "L_test" not in rows
    assert "ledger_l_test_oracle_v1" in rows


def test_stub_composer_does_not_attach_memory_skill_when_registry_does_not_support_action():
    from SelfEvolvingHarnessTS.run_fast_path_ablation import stub_skill_memory_composer

    packet = {
        "skills": [{"name": "median_smooth", "allowed_actions": ["v_median"]}],
        "memory": {
            "prior_fragments": [
                {
                    "skill_id": "ledger_prior_best_action",
                    "action_id": "f0_median_w25",
                }
            ]
        },
    }

    candidate = stub_skill_memory_composer(packet)

    assert candidate.action_id == "f0_median_w25"
    assert candidate.skill_id is None


def test_stub_composer_attaches_registry_skill_when_it_supports_memory_action():
    from SelfEvolvingHarnessTS.run_fast_path_ablation import stub_skill_memory_composer

    packet = {
        "skills": [{"name": "median_smooth", "allowed_actions": ["v_median"]}],
        "memory": {
            "prior_fragments": [
                {
                    "skill_id": "ledger_prior_best_action",
                    "action_id": "v_median",
                }
            ]
        },
    }

    candidate = stub_skill_memory_composer(packet)

    assert candidate.action_id == "v_median"
    assert candidate.skill_id == "median_smooth"


def test_stub_composer_prefers_utility_memory_over_legacy_prior():
    from SelfEvolvingHarnessTS.run_fast_path_ablation import stub_skill_memory_composer

    packet = {
        "skills": [{"name": "median_smooth", "allowed_actions": ["v_median"]}],
        "memory": {
            "utility_memory": [
                {
                    "memory_type": "utility",
                    "role": "recommend",
                    "skill_id": "ledger_prior_best_action",
                    "action_id": "v_median",
                    "evidence_refs": ["utility:r0"],
                }
            ],
            "prior_fragments": [
                {
                    "skill_id": "legacy_prior",
                    "action_id": "v_winsor",
                }
            ],
        },
    }

    candidate = stub_skill_memory_composer(packet)

    assert candidate.action_id == "v_median"
    assert candidate.skill_id == "median_smooth"
    assert "utility:r0" in candidate.evidence_refs


def test_stub_composer_abstains_when_packet_contains_only_risk_memory():
    from SelfEvolvingHarnessTS.run_fast_path_ablation import stub_skill_memory_composer

    packet = {
        "skills": [{"name": "median_smooth", "allowed_actions": ["v_median"]}],
        "memory": {
            "risk_memory": [
                {
                    "memory_type": "risk",
                    "role": "ban",
                    "action_id": "v_median",
                    "evidence_refs": ["risk:r0"],
                }
            ],
            "utility_memory": [],
            "prior_fragments": [],
        },
    }

    candidate = stub_skill_memory_composer(packet)

    assert candidate.action_id == "v_none"
    assert candidate.abstain_to_raw is True
    assert candidate.risk_rule["source"] == "risk_memory"
    assert "risk:r0" in candidate.evidence_refs
