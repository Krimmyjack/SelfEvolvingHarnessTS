"""Focused protocol assertions for the Workflow Generation A5/A3 slice.

No LLM and no Target outcome are opened by these tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "evaluation" / "functional")):
    if path not in sys.path:
        sys.path.insert(0, path)

import evaluation.functional.task_episode_harness.workflow_gen as wg_module
from evaluation.functional.task_episode_harness.workflow_gen import (
    WORKFLOW_GEN_SOURCE_TASKS,
    WORKFLOW_GEN_TARGET_TASK,
    _canonical_fingerprint,
    _generated_workflow_signature,
    _generation_call,
    _generation_payload,
    _target_spec,
)


def test_target_is_an_exposed_abstain_natural_episode_by_construction() -> None:
    spec = _target_spec()
    assert spec["task_episode_id"] == WORKFLOW_GEN_TARGET_TASK
    assert spec["support_origins"] == (1104, 1368, 1800)
    assert spec["delayed_origins"] == (2856, 3648, 3888)
    assert WORKFLOW_GEN_TARGET_TASK not in WORKFLOW_GEN_SOURCE_TASKS
    assert WORKFLOW_GEN_SOURCE_TASKS == (
        "natural_k1_01",
        "natural_k1_02",
        "natural_k1_04",
    )


def test_generation_payload_differs_only_in_source_experiences() -> None:
    target_spec = _target_spec()
    public_context = {
        "scope": frozenset({"T1", "T10"}),
        "observations": {
            "T1": {
                "local_robust_z_peak": 9.5,
                "local_robust_z_peak_bin": "high",
                "missing_fraction": 0.0,
                "level_excursion_score": 8.0,
            },
            "T10": {
                "local_robust_z_peak": 13.2,
                "local_robust_z_peak_bin": "high",
                "missing_fraction": 0.0,
                "level_excursion_score": 11.0,
            },
        },
        "representative_uid": "T10",
        "representative_features": {
            "task_kind": "forecast",
            "estimated_region_start_fraction": 0.0,
            "estimated_region_end_fraction": 0.8,
            "estimated_level_offset": 1.0,
        },
    }
    inventory = ()
    feedback = [{
        "program": "outlier_mad",
        "support_gain": 0.05,
        "support_se_block": 0.04,
        "support_gain_over_se": 1.25,
        "mechanical_gate": "PASS",
        "agent_decision": "CONTINUE",
    }]
    a3 = _generation_payload(
        target_spec=target_spec,
        public_context=public_context,
        inventory=inventory,
        target_feedback=feedback,
        source_experiences=[],
    )
    a5 = _generation_payload(
        target_spec=target_spec,
        public_context=public_context,
        inventory=inventory,
        target_feedback=feedback,
        source_experiences=[{
            "task_episode_id": "natural_k1_04",
            "program": "winsorize",
            "relation": "POSITIVE",
            "local_status": "LOCAL_ACTIVE",
        }],
    )
    a3_without_source = {
        key: value for key, value in a3.items()
        if key != "source_experiences"
    }
    a5_without_source = {
        key: value for key, value in a5.items()
        if key != "source_experiences"
    }
    assert set(a3) == set(a5)
    assert (
        _canonical_fingerprint(a3_without_source)
        == _canonical_fingerprint(a5_without_source)
    )
    assert a3["source_experiences"] == []
    assert len(a5["source_experiences"]) == 1


def test_generated_signature_is_a_safe_surface_id_component() -> None:
    signature = _generated_workflow_signature([
        ("repair_level_shift", {}),
        ("hampel_filter", {"window": 5}),
    ])
    assert signature == "generated_repair_level_shift_hampel_filter"
    assert "|" not in signature
    assert all(
        character.islower() or character.isdigit() or character == "_"
        for character in signature
    )


def test_generation_call_compiles_valid_proposal_and_records_abstain(
    monkeypatch,
) -> None:
    def fake_compile(proposal, inventory, public_context, *, generation):
        assert proposal["decision"] == "PROPOSE"
        return object()

    monkeypatch.setattr(wg_module, "_nf_call", lambda messages: {
        "decision": "PROPOSE",
        "steps": [{"op": "outlier_mad", "params": {}}],
        "requested_observations": [],
        "fallback": "IDENTITY",
        "experience_use": [],
    })
    monkeypatch.setattr(wg_module, "compile_workflow_proposal", fake_compile)
    payload = {
        "operator_inventory": (),
        "binding_context": {"features": {}},
    }
    _response, compiled, status = _generation_call(payload)
    assert status == "COMPILED"
    assert compiled is not None

    monkeypatch.setattr(wg_module, "_nf_call", lambda messages: {
        "decision": "ABSTAIN",
        "steps": [],
        "requested_observations": [],
        "fallback": "IDENTITY",
        "experience_use": [],
    })
    response, compiled, status = _generation_call(payload)
    assert status == "GENERATION_ABSTAIN"
    assert compiled is None
    assert response["decision"] == "ABSTAIN"
