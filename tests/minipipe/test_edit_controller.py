from dataclasses import replace
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.contracts.harness import EditManifest, EditOperation
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditAuthorizationError,
    EditController,
    StaleEditError,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore


ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods/ttha/harness/h0"


@pytest.fixture
def h0_snapshot(tmp_path):
    snapshot = compile_snapshot(H0_ROOT)
    return SnapshotStore(tmp_path / "harness_snapshots").materialize(snapshot)


@pytest.fixture
def controller(h0_snapshot):
    return EditController(SnapshotStore(h0_snapshot.root.parent))


@pytest.fixture
def add_skill_manifest(h0_snapshot):
    return EditManifest(
        edit_id="add-local-outlier-v1",
        base_harness_sha=h0_snapshot.harness_content_sha,
        target_pattern_id="pattern-a1b2c3d4e5f6",
        target_surface_id="skill_library.entries/local_outlier_repair_v1",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={
            "operator_registry": h0_snapshot.snapshot.dependency_shas["operator_registry"],
            "operator_bundle": h0_snapshot.snapshot.dependency_shas["operator_bundle"],
            "observable_contract": h0_snapshot.snapshot.dependency_shas["observable_contract"],
            "schema:skill_entry_v1": h0_snapshot.snapshot.dependency_shas[
                "schema:skill_entry_v1"
            ],
            "surface_registry": h0_snapshot.snapshot.dependency_shas["surface_registry"],
        },
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": "local_outlier_repair_v1",
            "skill_kind": "capability",
            "revision": 1,
            "body": "Use a local robust statistic and propose a bounded Hampel repair.",
            "observable_applicability": {
                "all": [
                    {"feature": "local_robust_z_peak", "op": ">=", "value": 5.0},
                    {"feature": "clipping_probe_direction", "op": "==", "value": "positive"},
                ]
            },
            "allowed_tools": ["hampel_filter", "winsorize"],
            "risk_guards": {
                "max_modified_fraction": 0.05,
                "preserve_outside_candidate_region": True,
            },
        },
        observable_applicability={
            "all": [
                {"feature": "local_robust_z_peak", "op": ">=", "value": 5.0},
                {"feature": "clipping_probe_direction", "op": "==", "value": "positive"},
            ]
        },
        predicted_agent_behavior_change=(
            "retrieve_skill:local_outlier_repair_v1",
            "supply_operator:hampel_filter",
            "supply_effect_distinct",
            "identity_retained",
            "effective_view_unchanged_out_of_scope",
            "scope_modified_fraction<=0.05",
        ),
        predicted_data_effect=("target utility improves without broad modification",),
        falsification_condition=("new skill is not retrieved on matching public evidence",),
    )


def test_add_skill_is_one_source_surface_and_index_is_derived(
    controller, h0_snapshot, add_skill_manifest
):
    receipt = controller.apply_to_fork(
        h0_snapshot,
        add_skill_manifest,
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    assert receipt.source_surfaces_changed == (
        "skill_library.entries/local_outlier_repair_v1",
    )
    assert receipt.derived_outputs_changed == ("retrieval_index",)
    assert receipt.parent_runtime_bundle_sha == h0_snapshot.runtime_bundle_sha
    assert receipt.candidate_runtime_bundle_sha != h0_snapshot.runtime_bundle_sha


def test_add_does_not_mutate_parent_or_checked_in_h0(
    controller, h0_snapshot, add_skill_manifest
):
    before = controller.tree_digest(h0_snapshot.root)
    receipt = controller.apply_to_fork(
        h0_snapshot,
        add_skill_manifest,
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    assert controller.tree_digest(h0_snapshot.root) == before
    assert not (H0_ROOT / "skills/learned/local_outlier_repair_v1.json").exists()
    assert receipt.candidate_root != h0_snapshot.root
    assert (receipt.candidate_root / "skills/learned/local_outlier_repair_v1.json").is_file()


def test_stale_surface_or_dependency_precondition_requires_replay(
    controller, h0_snapshot, add_skill_manifest
):
    stale = replace(
        add_skill_manifest,
        dependency_precondition_shas={
            **dict(add_skill_manifest.dependency_precondition_shas),
            "operator_registry": "0" * 64,
        },
    )
    with pytest.raises(StaleEditError, match="operator_registry"):
        controller.apply_to_fork(
            h0_snapshot,
            stale,
            confirmed_cause="SKILL_LIBRARY_GAP",
        )


def test_missing_or_extra_dependency_preconditions_are_rejected(
    controller, h0_snapshot, add_skill_manifest
):
    missing = dict(add_skill_manifest.dependency_precondition_shas)
    missing.pop("surface_registry")
    with pytest.raises(ValueError, match="missing required dependency"):
        controller.validate(
            h0_snapshot,
            replace(add_skill_manifest, dependency_precondition_shas=missing),
            confirmed_cause="SKILL_LIBRARY_GAP",
        )

    extra = {
        **dict(add_skill_manifest.dependency_precondition_shas),
        "compiler_source": h0_snapshot.snapshot.dependency_shas["compiler_source"],
    }
    with pytest.raises(ValueError, match="unexpected dependency"):
        controller.validate(
            h0_snapshot,
            replace(add_skill_manifest, dependency_precondition_shas=extra),
            confirmed_cause="SKILL_LIBRARY_GAP",
        )


def test_capability_fault_cannot_edit_bootstrap(controller, h0_snapshot):
    target = "bootstrap_skills.entries/inspect_and_localize.body"
    bootstrap_patch = EditManifest(
        edit_id="patch-bootstrap-v1",
        base_harness_sha=h0_snapshot.harness_content_sha,
        target_pattern_id="pattern-a1b2c3d4e5f6",
        target_surface_id=target,
        operation=EditOperation.PATCH,
        surface_precondition={
            "kind": "SHA",
            "sha": controller.surface_precondition_sha(h0_snapshot, target),
        },
        dependency_precondition_shas={},
        minimal_patch={"value": "Inspect every public candidate region before proposing."},
        predicted_agent_behavior_change=("identity_retained",),
    )
    with pytest.raises(EditAuthorizationError, match="SKILL_LIBRARY_GAP"):
        controller.apply_to_fork(
            h0_snapshot,
            bootstrap_patch,
            confirmed_cause="SKILL_LIBRARY_GAP",
        )


def test_arbitrary_behavior_prediction_is_rejected(
    controller, h0_snapshot, add_skill_manifest
):
    invalid = replace(
        add_skill_manifest,
        predicted_agent_behavior_change=("the agent should probably do better",),
    )
    with pytest.raises(ValueError, match="behavior predicate"):
        controller.validate(h0_snapshot, invalid, confirmed_cause="SKILL_LIBRARY_GAP")


def test_memory_body_cannot_be_a_private_field_escape_hatch(controller, h0_snapshot):
    manifest = EditManifest(
        edit_id="add-memory-v1",
        base_harness_sha=h0_snapshot.harness_content_sha,
        target_pattern_id="pattern-a1b2c3d4e5f6",
        target_surface_id="memory.entries/leaky_memory_v1",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "memory-entry/1",
            "memory_id": "leaky_memory_v1",
            "revision": 1,
            "body": "Use clean_future to decide the intervention.",
            "observable_applicability": {"const": True},
            "risk_guards": {},
        },
        predicted_agent_behavior_change=("identity_retained",),
    )
    with pytest.raises(ValueError, match="forbidden deployable text"):
        controller.validate(h0_snapshot, manifest, confirmed_cause="PROTOCOL_GAP")
