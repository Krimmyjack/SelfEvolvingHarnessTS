import pytest

from SelfEvolvingHarnessTS.contracts.harness import (
    EditManifest,
    EditOperation,
    SkillKind,
    load_learned_skill_entry,
    load_memory_entry,
    load_skill_entry,
)
from SelfEvolvingHarnessTS.contracts.observables import validate_applicability


VALID_SKILL = {
    "schema_version": "skill-entry/1",
    "skill_id": "local_outlier_repair_v1",
    "skill_kind": "capability",
    "revision": 1,
    "body": "Prefer a local robust repair when public evidence supports it.",
    "observable_applicability": {
        "all": [{"feature": "local_robust_z_peak", "op": ">=", "value": 5.0}]
    },
    "allowed_tools": ["hampel_filter"],
    "risk_guards": {"max_modified_fraction": 0.05},
}


def test_valid_capability_skill_loads():
    skill = load_learned_skill_entry(VALID_SKILL)
    assert skill.skill_kind is SkillKind.CAPABILITY
    assert skill.allowed_tools == ("hampel_filter",)


def test_skill_entry_rejects_private_fields():
    with pytest.raises(ValueError, match="forbidden deployable field"):
        load_skill_entry({**VALID_SKILL, "injection_type": "spike"})


def test_applicability_rejects_unknown_or_oracle_features():
    with pytest.raises(ValueError, match="unknown observable feature"):
        validate_applicability(
            {"all": [{"feature": "injection_type", "op": "==", "value": "spike"}]}
        )


def test_deployable_memory_rejects_source_pattern_provenance():
    with pytest.raises(ValueError, match="forbidden deployable field"):
        load_memory_entry(
            {
                "schema_version": "memory-entry/1",
                "memory_id": "local-repair-caution-v1",
                "revision": 1,
                "body": "Keep the change local.",
                "observable_applicability": {"const": True},
                "risk_guards": {},
                "pattern_id": "pattern-private-source",
            }
        )


def test_add_manifest_requires_absent_precondition():
    with pytest.raises(ValueError, match="ABSENT"):
        EditManifest(
            edit_id="e1",
            base_harness_sha="a" * 64,
            target_pattern_id="p1",
            target_surface_id="skill_library.entries/local_outlier_repair_v1",
            operation=EditOperation.ADD,
            surface_precondition={"kind": "SHA", "sha": "b" * 64},
            dependency_precondition_shas={},
            new_value=VALID_SKILL,
            predicted_agent_behavior_change=("retrieve_new_skill",),
            predicted_data_effect=("target_gain",),
            falsification_condition=("skill_not_retrieved",),
        )


def test_applicability_enforces_feature_types_and_nonempty_nodes():
    with pytest.raises(ValueError, match="non-empty"):
        validate_applicability({"all": []})
    with pytest.raises(ValueError, match="bin label"):
        validate_applicability(
            {"feature": "missing_fraction", "op": ">", "value": "large"}
        )
