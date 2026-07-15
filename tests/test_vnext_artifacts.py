from __future__ import annotations

import pytest

from SelfEvolvingHarnessTS.vnext.artifacts import (
    MethodEvaluationArtifact, VNextMethodArtifact, VNextMethodArtifactV2,
)


def _sha(character: str) -> str:
    return character * 64


def test_method_artifact_computes_and_verifies_semantic_identity():
    artifact = VNextMethodArtifact(
        method_id="vnext_h0", benchmark_version="benchmark-v0.2",
        pattern_binding_sha=_sha("1"), action_menu_sha=_sha("2"),
        candidate_grammar_sha=_sha("3"), action_eligibility_sha=_sha("4"),
        harness_sha=_sha("5"), supplier_kind="deterministic_b1",
        selector_sha=_sha("6"), agent_model_id=None, prompt_sha=None,
        decoding_sha=None, dependency_fingerprint={"numpy": "2"},
        budget_sha=_sha("7"),
    )
    assert len(artifact.semantic_sha) == 64
    assert len(artifact.artifact_sha) == 64
    with pytest.raises(ValueError, match="semantic_sha"):
        VNextMethodArtifact(**{**artifact.__dict__, "semantic_sha": _sha("f")})


def test_evaluation_artifact_rejects_nonheadline_estimand():
    with pytest.raises(ValueError, match="published_joint_v02"):
        MethodEvaluationArtifact(
            _sha("1"), _sha("2"), _sha("3"), "joint", _sha("4"), _sha("5"),
            estimand="train_only",
        )


def test_v2_method_identity_binds_supplier_policy_and_comparison_without_fake_parent():
    artifact = VNextMethodArtifactV2(
        method_id="vnext-h0", benchmark_version="benchmark-v0.2",
        pattern_binding_sha=_sha("1"), action_menu_sha=_sha("2"),
        candidate_grammar_sha=_sha("3"), action_eligibility_sha=_sha("4"),
        harness_sha=_sha("5"), supplier_kind="deterministic_b3",
        supplier_policy_sha=_sha("6"), selector_sha=_sha("7"),
        method_input_contract_sha=_sha("8"), environment_sha=_sha("9"),
        seed_book_sha=_sha("a"), agent_model_id=None, prompt_sha=None,
        decoding_sha=None, dependency_fingerprint={"numpy": "2.2.6"},
        budget_sha=_sha("b"), comparison_base_sha=_sha("c"),
    )
    changed = VNextMethodArtifactV2(**{
        **artifact.__dict__, "supplier_policy_sha": _sha("d"), "semantic_sha": "",
    })
    assert artifact.semantic_sha != changed.semantic_sha
    assert artifact.parent_sha is None
    with pytest.raises(ValueError, match="atomic_edit requires"):
        VNextMethodArtifactV2(**{
            **artifact.__dict__, "atomic_edit": {"surface": "selector"},
        })
