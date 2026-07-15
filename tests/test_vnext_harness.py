from __future__ import annotations

import pytest

from SelfEvolvingHarnessTS.vnext.harness import (
    AtomicHarnessEdit, HarnessArtifactV1, HarnessArtifactV2,
    HarnessEditAuthorizationArtifactV1,
)


def test_atomic_edit_has_parent_chain_and_only_one_semantic_surface():
    h0 = HarnessArtifactV1.h0()
    edit = AtomicHarnessEdit("selector_threshold", 0.7, "planted selector witness")
    h1 = h0.apply_atomic(edit)
    assert h1.parent_sha == h0.artifact_sha
    assert h1.selector_threshold == 0.7
    assert h1.semantic_sha != h0.semantic_sha
    assert h1.atomic_edit == edit


def test_protected_and_unqualified_llm_surfaces_fail_loud():
    with pytest.raises(ValueError, match="protected"):
        AtomicHarnessEdit("operator_registry", {}, "must remain frozen")
    h0 = HarnessArtifactV1.h0()
    with pytest.raises(ValueError, match="M3"):
        h0.apply_atomic(AtomicHarnessEdit("llm_instruction", "new", "candidate"))


def test_v2_edit_authorization_is_separate_from_runtime_llm_efficacy():
    harness = HarnessArtifactV2.engineering_default()
    authorization = HarnessEditAuthorizationArtifactV1(
        h0_sha="1" * 64, evolution_prereg_sha="2" * 64,
        llm_trial_authorization_sha="3" * 64,
        allowed_surfaces=("selector_threshold", "llm_instruction"),
    )
    changed = harness.apply_atomic(
        AtomicHarnessEdit("selector_threshold", 0.65, "one frozen grid step"),
        authorization,
    )
    assert changed.parent_sha == harness.artifact_sha
    assert changed.selector_threshold == 0.65
    llm_changed = changed.apply_atomic(
        AtomicHarnessEdit("llm_instruction", "instruction-catalog-v1:item-2", "trial edit"),
        authorization,
    )
    assert llm_changed.llm_instruction is not None


def test_v2_rejects_unbounded_or_multi_slot_edits():
    harness = HarnessArtifactV2.engineering_default()
    authorization = HarnessEditAuthorizationArtifactV1(
        h0_sha="1" * 64, evolution_prereg_sha="2" * 64,
        llm_trial_authorization_sha=None,
        allowed_surfaces=("selector_threshold", "supplier_mix"),
    )
    with pytest.raises(ValueError, match="grid"):
        harness.apply_atomic(
            AtomicHarnessEdit("selector_threshold", 0.75, "too many steps"), authorization,
        )
    with pytest.raises(ValueError, match="one slot"):
        harness.apply_atomic(
            AtomicHarnessEdit(
                "supplier_mix", {"deterministic": 1, "random": 2, "llm": 0}, "two slots",
            ), authorization,
        )
