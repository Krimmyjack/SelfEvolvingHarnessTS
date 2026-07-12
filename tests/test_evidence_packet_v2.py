"""P1 契约测试：EvidencePacket v2（连续证据 + trace + allowed_grammar；R1=禁二值读数）。"""
import json

import pytest

from SelfEvolvingHarnessTS.policy.evidence_packet import PACKET_SCHEMA_V2, build_evidence_packet_v2
from SelfEvolvingHarnessTS.policy.skills import SKILLS_V1

MENU_META = {"version": "v1", "sha256": "abc", "allowed_actions": ["v_none", "v_median"]}


def _record():
    return {
        "uid": "u1",
        "cell": "forecast|snrLow|miss",
        "snr": -4.0,
        "miss_rate": 0.2,
        "X_p": [24, 0.2, 0.3, 0.4, 0, 0.6, 0.7, 0.05],
    }


def _v2(**kw):
    base = dict(skills=SKILLS_V1, memory_rows=None, action_menu_meta=MENU_META)
    base.update(kw)
    return build_evidence_packet_v2(_record(), **base)


def test_v2_schema_grammar_and_channels():
    packet = _v2(
        continuous_evidence={"v_median": {"support_n": 12, "utility_q50": 0.031, "harm_rate": 0.14}},
        trace_summaries=[{"op": "denoise_median", "ok": False,
                          "error": "ValueError: kernel", "modified_fraction": 0.31}],
    )
    assert packet["schema"] == PACKET_SCHEMA_V2
    grammar = packet["allowed_grammar"]
    assert grammar["grammar"] == "program_spec_v1"
    assert grammar["max_steps"] == 3
    assert 25 in grammar["window_grid"]
    assert "impute_linear" in grammar["imputers"]
    assert "snr" in grammar["guard_features"]
    assert packet["continuous_evidence"]["v_median"]["utility_q50"] == 0.031
    assert packet["trace_summaries"][0]["modified_fraction"] == 0.31
    json.dumps(packet, allow_nan=False)


def test_v2_rejects_binary_readouts():
    # R1（slice v2 教训）：二值读数必输——bool 直接拒收
    with pytest.raises(ValueError):
        _v2(continuous_evidence={"v_median": {"is_good": True}})


def test_v2_rejects_non_numeric_and_non_finite():
    with pytest.raises(ValueError):
        _v2(continuous_evidence={"v_median": {"verdict": "harmful"}})
    with pytest.raises(ValueError):
        _v2(continuous_evidence={"v_median": {"utility": float("nan")}})


def test_v2_traces_are_leakage_linted():
    packet = _v2(trace_summaries=[{"op": "denoise_median", "L_test": "secret_loss",
                                   "oracle": "secret_oracle", "ok": True}])
    dumped = json.dumps(packet, ensure_ascii=False)
    assert "secret" not in dumped


def test_v2_carries_task_spec():
    from SelfEvolvingHarnessTS.policy.task_spec import classification_task_spec_v1
    packet = _v2(task_spec=classification_task_spec_v1())
    assert packet["task"]["task_type"] == "classification"
    assert packet["provenance"]["packet_schema"] == PACKET_SCHEMA_V2
