import json

from SelfEvolvingHarnessTS.policy.evidence_packet import build_evidence_packet
from SelfEvolvingHarnessTS.policy.skills import SKILLS_V1


def _all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _all_keys(value)


def test_evidence_packet_excludes_labels_and_raw_series():
    record = {
        "uid": "u1",
        "cell": "forecast|snrLow|miss",
        "origin": "S_secret",
        "snr": -5.0,
        "miss_rate": 0.2,
        "X_p": [1, 0.2, 0.3, 0.4, 0, 0.6, 0.7, 0.8],
        "L_test": {"v_none": "secret_loss"},
        "arms": {"dp_abstain": {"pick": "secret_pick"}},
        "X_t": ["secret_true_noise", "secret_true_missing"],
        "history": [1, 2, 3, 4],
    }
    memory = {
        "prior_fragments": [
            {
                "sim": 0.91,
                "cell": "forecast|snrLow|miss",
                "program": {"steps": [{"op": "impute_linear", "params": {}}]},
                "L_test": "secret_memory_loss",
            }
        ],
        "failure_warnings": [{"sim": 0.7, "signature": "nan_after_savgol", "arms": "secret"}],
    }

    packet = build_evidence_packet(
        record,
        skills=SKILLS_V1,
        memory_rows=memory,
        action_menu_meta={"version": "v1", "sha256": "abc", "allowed_actions": ["v_none"]},
    )
    dumped = json.dumps(packet, ensure_ascii=False)

    assert "secret" not in dumped
    assert not ({"L_test", "arms", "X_t", "history"} & set(_all_keys(packet)))


def test_evidence_packet_includes_skills_memory_and_action_menu():
    record = {
        "uid": "u1",
        "cell": "forecast|snrHigh|full",
        "snr": 12.0,
        "miss_rate": 0.0,
        "X_p": [24, 0.2, 0.8, 0.4, 0, 0.6, 0.7, 0.1],
    }
    memory = {
        "prior_fragments": [{"sim": 0.88, "cell": "forecast|snrHigh|full", "program": {"steps": []}}],
        "failure_warnings": [{"sim": 0.61, "signature": "over_smooth_season"}],
    }

    packet = build_evidence_packet(
        record,
        skills=SKILLS_V1,
        memory_rows=memory,
        action_menu_meta={"version": "v1", "sha256": "abc", "allowed_actions": ["v_none", "v_median"]},
    )

    assert packet["schema"] == "skill_memory_evidence_packet_v1"
    assert packet["pattern"]["cell"] == "forecast|snrHigh|full"
    assert packet["pattern"]["struct_feats"]["period"] == 24.0
    assert {card["name"] for card in packet["skills"]} >= {"identity", "median_smooth"}
    assert packet["memory"]["prior_fragments"][0]["sim"] == 0.88
    assert packet["action_menu"]["allowed_actions"] == ["v_none", "v_median"]
    json.dumps(packet, allow_nan=False)

def test_evidence_packet_includes_formal_composer_context_fields():
    record = {
        "uid": "u2",
        "cell": "forecast|snrLow|miss",
        "snr": -4.0,
        "miss_rate": 0.1,
        "X_p": [12, 0.3, 0.4, 0.2, 0, 0.6, 0.5, 0.05],
    }
    packet = build_evidence_packet(
        record,
        skills=[{"name": "median_smooth", "score": 0.9, "allowed_actions": ["v_median"]}],
        memory_rows={"prior_fragments": [], "failure_warnings": []},
        action_menu_meta={"version": "v1", "sha256": "abc", "allowed_actions": ["v_none", "v_median"]},
        support_stats={"distance": 2.1, "threshold": 3.3, "out_of_support": False},
        harm_stats={"harm_rate": 0.141, "mean_gain_vs_raw": 0.0963},
        risk_constraints=[{"rule_id": "force_raw_on_weak_support", "op": "abstain_to_raw"}],
        incumbent_decision={"action_id": "v_median", "abstained": False},
    )

    assert packet["support"]["out_of_support"] is False
    assert packet["harm_stats"]["harm_rate"] == 0.141
    assert packet["risk_constraints"][0]["rule_id"] == "force_raw_on_weak_support"
    assert packet["incumbent_decision"]["action_id"] == "v_median"
    assert packet["candidate_schema"]["type"] == "typed_candidate_v1"
    assert "skill_id" in packet["candidate_schema"]["allowed_fields"]
    assert "abstain_to_raw" in packet["candidate_schema"]["allowed_fields"]
    json.dumps(packet, allow_nan=False)
