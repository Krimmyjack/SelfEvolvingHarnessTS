import json

from SelfEvolvingHarnessTS.memory.evidence_schema import MemoryEvidence, build_memory_evidence
from SelfEvolvingHarnessTS.policy.evidence_packet import build_evidence_packet


def test_memory_evidence_packet_row_carries_utility_and_harm_without_losses():
    ev = MemoryEvidence(
        task="forecast",
        pattern_region="forecast|snrLow|miss",
        skill_id="median_smooth",
        action_id="v_median",
        program={"steps": [{"op": "denoise_median", "params": {"window": 5}}]},
        utility_delta_vs_raw=0.12,
        harm_delta_vs_raw=0.0,
        support={"distance": 1.2, "threshold": 3.0},
        subgroup="S_season",
        validator_result={"passed": True, "split": "held_out_b"},
        failure_signature=None,
        source_domain="synthetic:S2",
        provenance={"record_uid": "u1", "L_test": "secret_loss"},
    )

    row = ev.to_packet_row()
    dumped = json.dumps(row, ensure_ascii=False)

    assert row["schema"] == "memory_evidence_v1"
    assert row["utility_delta_vs_raw"] == 0.12
    assert row["harm_delta_vs_raw"] == 0.0
    assert row["validator_result"]["passed"] is True
    assert "secret" not in dumped
    assert "L_test" not in dumped
    assert "raw_loss" not in dumped
    assert "selected_loss" not in dumped


def test_build_memory_evidence_computes_utility_and_harm_against_raw():
    helpful = build_memory_evidence(
        task="forecast",
        pattern_region="forecast|snrHigh|full",
        action_id="v_none",
        raw_loss=1.0,
        selected_loss=0.7,
        validator_result={"passed": True},
    )
    harmful = build_memory_evidence(
        task="forecast",
        pattern_region="forecast|snrHigh|full",
        action_id="v_savgol",
        raw_loss=1.0,
        selected_loss=1.4,
        validator_result={"passed": False, "failure_signature": "over_smooth"},
    )

    assert helpful.utility_delta_vs_raw == 0.3
    assert helpful.harm_delta_vs_raw == 0.0
    assert harmful.utility_delta_vs_raw == -0.4
    assert harmful.harm_delta_vs_raw == 0.4
    assert harmful.failure_signature == "over_smooth"


def test_memory_evidence_feeds_evidence_packet_memory_summary():
    ev = build_memory_evidence(
        task="forecast",
        pattern_region="forecast|snrLow|miss",
        skill_id="median_smooth",
        action_id="v_median",
        raw_loss=1.0,
        selected_loss=0.8,
        validator_result={"passed": True},
    )
    packet = build_evidence_packet(
        {
            "uid": "u2",
            "cell": "forecast|snrLow|miss",
            "snr": -4.0,
            "miss_rate": 0.1,
            "X_p": [12, 0.3, 0.4, 0.2, 0, 0.6, 0.5, 0.05],
        },
        skills=[],
        memory_rows={"prior_fragments": [ev.to_packet_row()], "failure_warnings": []},
        action_menu_meta={"version": "v1", "sha256": "abc", "allowed_actions": ["v_none", "v_median"]},
    )

    row = packet["memory"]["prior_fragments"][0]
    assert row["schema"] == "memory_evidence_v1"
    assert row["utility_delta_vs_raw"] == 0.2
    assert row["action_id"] == "v_median"