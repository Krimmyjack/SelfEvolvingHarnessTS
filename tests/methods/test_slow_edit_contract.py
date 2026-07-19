from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (
    AgentProtocolError,
    TTHAAgentCore,
    validate_local_schema,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent


ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods/ttha/harness/h0"


def _base_add(applicability):
    h0 = compile_snapshot(H0_ROOT)
    skill = {
        "schema_version": "skill-entry/1",
        "skill_id": "contract_probe_v1",
        "skill_kind": "capability",
        "revision": 1,
        "body": "Use a bounded public-evidence repair candidate.",
        "observable_applicability": applicability,
        "allowed_tools": ["repair_level_shift"],
        "risk_guards": {"max_modified_fraction": 0.2},
    }
    return {
        "edit_manifest": {
            "edit_id": "edit-contract-probe-v1",
            "base_harness_sha": h0.harness_content_sha,
            "target_pattern_id": "pattern-contract-probe-v1",
            "target_surface_id": "skill_library.entries/contract_probe_v1",
            "operation": "ADD",
            "surface_precondition": {"kind": "ABSENT"},
            "dependency_precondition_shas": {},
            "new_value": skill,
            "observable_applicability": applicability,
            "predicted_agent_behavior_change": [
                "retrieve_skill:contract_probe_v1",
                "supply_operator:repair_level_shift",
                "supply_effect_distinct",
                "identity_retained",
                "scope_modified_fraction<=0.2",
            ],
            "predicted_data_effect": ["target utility improves"],
            "falsification_condition": ["predicted behavior is absent"],
        }
    }


@pytest.mark.parametrize(
    "applicability",
    [
        {"feature": "level_excursion_score", "op": ">=", "value": 2.5},
        {"feature": "level_excursion_score", "op": "==", "value": "high"},
        {"feature": "level_excursion_score", "op": "in", "value": ["medium", "high"]},
        {"const": True},
        {
            "all": [
                {"feature": "missing_fraction", "op": "==", "value": 0.0},
                {
                    "any": [
                        {"feature": "estimated_level_offset", "op": ">=", "value": 0.4},
                        {"feature": "estimated_level_offset", "op": "<=", "value": -0.4},
                    ]
                },
            ]
        },
        {"not": {"feature": "period_evidence_status", "op": "==", "value": "UNKNOWN"}},
    ],
)
def test_schema_valid_add_is_shape_valid_for_bounded_recursive_applicability(
    tmp_path, applicability
):
    payload = _base_add(applicability)
    schema = TTHAAgentCore.load_stage_schema("slow_edit_v1")
    validate_local_schema(payload, schema)
    manifest = TTHASlowAgent._manifest_from_payload(payload)
    controller = EditController(SnapshotStore(tmp_path / "snapshots"))
    shaped = controller.validate_shape(manifest)
    assert shaped.manifest.edit_id == "edit-contract-probe-v1"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(minimal_patch={"append_text": "not atomic"}),
        lambda row: row.update(
            predicted_agent_behavior_change=["the agent should do better"]
        ),
        lambda row: row["new_value"].pop("risk_guards"),
        lambda row: row.update(operation="PATCH"),
    ],
)
def test_run1_run2_contract_failures_are_rejected_before_controller(tmp_path, mutate):
    del tmp_path
    payload = _base_add({"const": True})
    mutate(payload["edit_manifest"])
    with pytest.raises(AgentProtocolError):
        validate_local_schema(
            payload,
            TTHAAgentCore.load_stage_schema("slow_edit_v1"),
        )
