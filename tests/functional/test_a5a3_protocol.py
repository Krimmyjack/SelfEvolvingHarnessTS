"""Focused protocol assertions for the clean A5/A3 replay.

Permission replay v1 adds deterministic decision-input identity checks:
the post-Support Promotion decision may read current Target evidence and
Target-local history only; Source Experience must not enter that decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "evaluation" / "functional")):
    if path not in sys.path:
        sys.path.insert(0, path)

import evaluation.functional.task_episode_harness.a5a3 as a5a3_module
from evaluation.functional.task_episode_harness.a5a3 import (
    TARGET_EPISODES,
    _decision_input,
    _decision_input_fingerprint,
    _memory_agent_decision,
    _memory_summary,
    _sync_memory_summary,
    _trust_channel_mechanical_check,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
    EVIDENCE_SUPPORT,
    RELATION_POSITIVE,
    STATUS_LOCAL_DRAFT,
    build_episode,
)


def test_target_blocks_are_forward_non_overlapping_unique_roles() -> None:
    all_origins = [
        origin
        for spec in TARGET_EPISODES
        for origins in (spec["support_origins"], spec["delayed_origins"])
        for origin in origins
    ]
    assert len(all_origins) == len(set(all_origins)) == 8
    previous_delayed_max = None
    for spec in TARGET_EPISODES:
        support = spec["support_origins"]
        delayed = spec["delayed_origins"]
        assert max(support) < min(delayed)
        if previous_delayed_max is not None:
            assert previous_delayed_max < min(support)
        previous_delayed_max = max(delayed)


def test_delayed_update_replaces_support_only_memory_summary() -> None:
    support_episode = build_episode(
        episode_id="test_ep",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace="test",
        context_summary={},
        workflow_signature="winsorize",
        support_response={"gain": 0.04, "se_block": 0.03,
                          "gain_over_se": 1.33, "accepted": True},
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation=RELATION_POSITIVE,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_LOCAL_DRAFT,
    )
    import dataclasses

    delayed_episode = dataclasses.replace(
        support_episode,
        delayed_response={"evaluated": True, "gain": 0.05,
                          "se_block": 0.02, "gain_over_se": 2.5},
        local_status="LOCAL_ACTIVE",
    )
    memories = [_memory_summary(support_episode)]
    _sync_memory_summary(memories, delayed_episode)
    assert memories[0]["delayed_gain"] == 0.05
    assert memories[0]["delayed_se_block"] == 0.02
    assert memories[0]["delayed_gain_over_se"] == 2.5


def _sample_target_history() -> list[dict[str, object]]:
    return [{
        "episode_id": "a5a3_A3_target_01_attempt_0",
        "program": "winsorize",
        "support_gain": 0.02,
        "support_se_block": 0.14,
        "support_gain_over_se": 0.02 / 0.14,
        "delayed_gain": None,
        "delayed_se_block": None,
        "delayed_gain_over_se": None,
        "relation": "POSITIVE",
        "local_status": "LOCAL_ACTIVE",
    }]


def test_same_target_evidence_gives_identical_arm_free_decision_input() -> None:
    history = _sample_target_history()
    a3_input = _decision_input(
        program="winsorize",
        gain=0.02,
        se=0.14,
        gain_over_se=0.02 / 0.14,
        remaining=["hampel_filter"],
        above_threshold=True,
        target_memories=history,
    )
    a5_input = _decision_input(
        program="winsorize",
        gain=0.02,
        se=0.14,
        gain_over_se=0.02 / 0.14,
        remaining=["hampel_filter"],
        above_threshold=True,
        target_memories=history,
    )
    assert (
        _decision_input_fingerprint(a3_input)
        == _decision_input_fingerprint(a5_input)
    )
    assert "source_experiences" not in a3_input
    assert a3_input["target_experiences"] == [{
        "program": "winsorize",
        "support_gain": 0.02,
        "support_se_block": 0.14,
        "support_gain_over_se": 0.02 / 0.14,
        "delayed_gain": None,
        "delayed_se_block": None,
        "delayed_gain_over_se": None,
        "relation": "POSITIVE",
        "local_status": "LOCAL_ACTIVE",
    }]


def test_promotion_llm_payload_excludes_source_bank(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_call(
        payload: dict[str, object],
        system: str,
    ) -> dict[str, str]:
        captured["payload"] = payload
        captured["system"] = system
        return {"decision": "CONTINUE", "reason": "test stub"}

    monkeypatch.setattr(a5a3_module, "_call", fake_call)
    result = _memory_agent_decision(
        program="winsorize",
        gain=0.02,
        se=0.14,
        gain_over_se=0.02 / 0.14,
        remaining=["hampel_filter"],
        above_threshold=True,
        target_memories=_sample_target_history(),
    )
    payload = json.loads(json.dumps(captured["payload"]))
    assert result["decision"] == "CONTINUE"
    assert "source_experiences" not in payload
    expected_history = [
        {
            key: value
            for key, value in _sample_target_history()[0].items()
            if key != "episode_id"
        }
    ]
    assert payload["target_experiences"] == expected_history
    assert "Source Experience is excluded" in captured["system"]
    assert result["decision_input"] == payload


def _recorded_probe(
    *,
    program: str,
    gain: float,
    se: float,
    remaining: list[str],
    leak_source: bool = False,
) -> dict[str, object]:
    payload = _decision_input(
        program=program,
        gain=gain,
        se=se,
        gain_over_se=gain / se,
        remaining=remaining,
        above_threshold=gain >= 0.005,
        target_memories=[],
    )
    if leak_source:
        payload["source_experiences"] = [{"program": "outlier_mad"}]
    return {
        "attempt_index": 0,
        "program": program,
        "support_gain": gain,
        "support_se_block": se,
        "support_gain_over_se": gain / se,
        "agent_decision": {
            "decision": "CONTINUE",
            "reason": "stub",
            "decision_input": payload,
        },
        "mechanical_gate": "PASS" if gain >= 0.005 else "REJECT",
    }


def test_mechanical_check_reads_nested_decision_inputs() -> None:
    rows = [{
        "task_episode_id": "target_01",
        "A3": {
            "probes": [
                _recorded_probe(
                    program="winsorize",
                    gain=0.02,
                    se=0.14,
                    remaining=["hampel_filter"],
                ),
            ],
        },
        "A5": {
            "probes": [
                _recorded_probe(
                    program="winsorize",
                    gain=0.02,
                    se=0.14,
                    remaining=["hampel_filter"],
                ),
            ],
        },
    }]
    check = _trust_channel_mechanical_check(rows)
    assert check["runtime_same_evidence_pair_count"] == 1
    assert check["runtime_pairs_all_identical"] is True
    assert check["source_key_in_any_decision_input"] is False
    assert check["trust_channel_cut"] is True


def test_mechanical_check_detects_source_leak_in_nested_input() -> None:
    rows = [{
        "task_episode_id": "target_01",
        "A3": {
            "probes": [
                _recorded_probe(
                    program="winsorize",
                    gain=0.02,
                    se=0.14,
                    remaining=["hampel_filter"],
                ),
            ],
        },
        "A5": {
            "probes": [
                _recorded_probe(
                    program="winsorize",
                    gain=0.02,
                    se=0.14,
                    remaining=["hampel_filter"],
                    leak_source=True,
                ),
            ],
        },
    }]
    check = _trust_channel_mechanical_check(rows)
    assert check["source_key_in_any_decision_input"] is True
    assert check["runtime_same_evidence_pairs"][0][
        "decision_input_identical"
    ] is False
    assert check["trust_channel_cut"] is False
