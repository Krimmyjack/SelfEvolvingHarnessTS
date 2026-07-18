from __future__ import annotations

from pathlib import Path
import json

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.contracts.harness import EditManifest, EditOperation
from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import _manifest_json, run_cycles
from SelfEvolvingHarnessTS.evaluation.minipipe.fixtures.contract_policy import (
    ContractPolicyBackend,
    DeterministicContractValuator,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore


ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods/ttha/harness/h0"


def _run(root: Path, *, cycles: int = 2, resume: bool = False):
    return run_cycles(
        cycles=cycles,
        run_root=root,
        backend=ContractPolicyBackend(),
        valuator=DeterministicContractValuator(),
        resume=resume,
    )


def _seed_overly_strict_missing_skill(root: Path) -> Path:
    store = SnapshotStore(root / "harness_snapshots")
    parent = store.materialize(compile_snapshot(H0_ROOT))
    controller = EditController(store)
    target = "skill_library.entries/overly_strict_missing_v1"
    definition = controller.surfaces.resolve(target).definition
    manifest = EditManifest(
        edit_id="seed-overly-strict-missing-v1",
        base_harness_sha=parent.harness_content_sha,
        target_pattern_id="pattern-seeded-retrieval",
        target_surface_id=target,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={
            key: parent.snapshot.dependency_shas[key]
            for key in definition.required_dependency_keys
        },
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": "overly_strict_missing_v1",
            "skill_kind": "capability",
            "revision": 1,
            "body": "Use bounded linear imputation for deployment-visible missingness.",
            "observable_applicability": {
                "feature": "missing_fraction",
                "op": ">",
                "value": 0.5,
            },
            "allowed_tools": ["impute_linear"],
            "risk_guards": {
                "max_modified_fraction": 0.25,
                "preserve_outside_candidate_region": True,
            },
        },
        observable_applicability={
            "feature": "missing_fraction",
            "op": ">",
            "value": 0.5,
        },
        predicted_agent_behavior_change=(
            "retrieve_skill:overly_strict_missing_v1",
            "identity_retained",
        ),
        predicted_data_effect=("seed fixture only",),
        falsification_condition=("seed fixture could not be materialized",),
    )
    return controller.apply_to_fork(
        parent,
        manifest,
        confirmed_cause="SKILL_LIBRARY_GAP",
    ).candidate_root


def _seed_broad_localization_procedure(root: Path) -> Path:
    store = SnapshotStore(root / "harness_snapshots")
    parent = store.materialize(compile_snapshot(H0_ROOT))
    controller = EditController(store)
    target = "bootstrap_skills.entries/inspect_and_localize.body"
    definition = controller.surfaces.resolve(target).definition
    manifest = EditManifest(
        edit_id="seed-broad-localization-v1",
        base_harness_sha=parent.harness_content_sha,
        target_pattern_id="pattern-seeded-localization",
        target_surface_id=target,
        operation=EditOperation.PATCH,
        surface_precondition={
            "kind": "SHA",
            "sha": controller.surface_precondition_sha(parent, target),
        },
        dependency_precondition_shas={
            key: parent.snapshot.dependency_shas[key]
            for key in definition.required_dependency_keys
        },
        minimal_patch={
            "value": (
                "Inspect the public aggregate candidate span as one region. "
                "Procedure marker: broad_region_only/v1."
            )
        },
        predicted_agent_behavior_change=("identity_retained",),
        predicted_data_effect=("seed fixture only",),
        falsification_condition=("seed fixture could not be materialized",),
    )
    return controller.apply_to_fork(
        parent,
        manifest,
        confirmed_cause="LOCALIZATION_PROCEDURE_GAP",
    ).candidate_root


def test_two_cycles_promote_at_most_one_edit_and_reproduce_scientific_outputs(
    tmp_path,
):
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    assert len(first.cycles) == 2
    assert all(len(cycle.promoted_edit_ids) <= 1 for cycle in first.cycles)
    assert first.cycles[1].starting_snapshot_sha == first.cycles[0].ending_snapshot_sha
    assert first.normalized_behavior_shas == second.normalized_behavior_shas
    assert first.scientific_verdicts == second.scientific_verdicts
    assert first.lineage.verify_hash_chain() is True
    assert first.lineage.promotions
    for event in first.lineage.promotions:
        assert event.parent_snapshot_sha
        assert event.edit_manifest_sha
        assert event.paired_replay_report_sha
        assert event.final_core_regression_sha


def test_primary_artifacts_exist_in_their_correct_visibility_roots(tmp_path):
    result = _run(tmp_path / "artifacts", cycles=1)
    resumed = _run(tmp_path / "artifacts", cycles=1, resume=True)

    assert (result.private_root / "case_feedback.jsonl").is_file()
    assert (result.public_root / "failure_patterns.json").is_file()
    assert (result.public_root / "failure_patterns.md").is_file()
    assert (result.public_root / "edit_manifest.json").is_file()
    assert (result.private_root / "paired_replay_report.json").is_file()
    assert (result.run_root / "harness_lineage.jsonl").is_file()
    assert (result.private_root / "operator_capability_backlog.jsonl").is_file()
    assert resumed.active_snapshot_sha == result.active_snapshot_sha
    assert resumed.lineage.verify_hash_chain() is True


def test_retrieval_miss_runs_a_positive_applicability_patch_cycle(tmp_path):
    seeded = _seed_overly_strict_missing_skill(tmp_path / "seed")
    result = run_cycles(
        cycles=1,
        run_root=tmp_path / "retrieval-cycle",
        backend=ContractPolicyBackend(),
        valuator=DeterministicContractValuator(),
        h0_root=seeded,
    )
    payload = json.loads(
        (result.cycles[0].public_root / "edit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    retrieval_edits = [
        edit
        for edit in payload["edits"]
        if edit["target_surface_id"]
        == "skill_library.entries/overly_strict_missing_v1.observable_applicability"
    ]
    assert retrieval_edits
    assert retrieval_edits[0]["operation"] == "PATCH"
    assert any(
        event.event_kind == "EDIT_EVALUATED"
        and event.edit_manifest_sha
        and event.verdict in {"SUPPORTED_EDIT", "PARTIAL_RECOVERY"}
        for event in result.lineage.events
    )


def test_bootstrap_patch_reaches_a_scientific_reject_verdict(tmp_path):
    seeded = _seed_broad_localization_procedure(tmp_path / "patch-seed")
    result = run_cycles(
        cycles=1,
        run_root=tmp_path / "patch-cycle",
        backend=ContractPolicyBackend(),
        valuator=DeterministicContractValuator(),
        h0_root=seeded,
    )
    payload = json.loads(
        (result.cycles[0].public_root / "edit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    patch_ids = {
        edit["edit_id"]
        for edit in payload["edits"]
        if edit["target_surface_id"]
        == "bootstrap_skills.entries/inspect_and_localize.body"
    }
    assert patch_ids
    evaluated = [
        event
        for event in result.lineage.events
        if event.event_kind == "EDIT_EVALUATED"
        and event.verdict in {"BEHAVIOR_CHANGED_NO_GAIN", "DEAD_EDIT"}
    ]
    assert evaluated


def test_pending_edit_requeues_and_stale_duplicate_is_superseded(tmp_path):
    run_root = tmp_path / "pending-lifecycle"
    first = _run(run_root, cycles=1)
    manifests = json.loads(
        (first.cycles[0].public_root / "edit_manifest.json").read_text("utf-8")
    )["edits"]
    reports = json.loads(
        (first.cycles[0].private_root / "paired_replay_report.json").read_text(
            "utf-8"
        )
    )["reports"]
    stale = next(
        edit
        for edit in manifests
        if edit["target_surface_id"]
        == "skill_library.entries/observed_missing_repair_v1"
    )
    target_ids = next(
        report["target_case_ids"]
        for report in reports
        if report["edit_id"] == stale["edit_id"]
    )
    active = compile_snapshot(
        run_root / "harness_snapshots" / first.active_snapshot_sha
    )
    controller = EditController(SnapshotStore(run_root / "harness_snapshots"))
    pending_target = "skill_library.entries/observed_missing_repair_pending_v1"
    definition = controller.surfaces.resolve(pending_target).definition
    new_value = dict(stale["new_value"])
    new_value["skill_id"] = "observed_missing_repair_pending_v1"
    pending = EditManifest(
        edit_id="pending-missing-repair-v1",
        base_harness_sha=active.harness_content_sha,
        target_pattern_id=str(stale["target_pattern_id"]),
        target_surface_id=pending_target,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={
            key: active.dependency_shas[key]
            for key in definition.required_dependency_keys
        },
        new_value=new_value,
        observable_applicability=dict(stale["observable_applicability"]),
        predicted_agent_behavior_change=(
            "retrieve_skill:observed_missing_repair_pending_v1",
            "supply_operator:impute_linear",
            "supply_effect_distinct",
            "identity_retained",
        ),
        predicted_data_effect=("target utility improves",),
        falsification_condition=("incremental gain absent",),
    )
    rows = [
        {
            "schema_version": "pending-edit/1",
            "manifest": manifest,
            "cause_code": "SKILL_LIBRARY_GAP",
            "target_case_ids": target_ids,
            "origin_cycle_id": "cycle-000",
            "deferred_cycles": 0,
        }
        for manifest in (stale, _manifest_json(pending))
    ]
    (run_root / "private" / "pending_edits.json").write_text(
        json.dumps({"schema_version": "pending-edit-set/1", "edits": rows}),
        encoding="utf-8",
    )

    resumed = _run(run_root, cycles=2, resume=True)
    assert any(
        event.event_kind == "SUPERSEDED"
        and event.edit_manifest_sha == canonical_sha256(stale)
        for event in resumed.lineage.events
    )
    assert any(
        event.event_kind == "REQUEUED"
        and event.edit_manifest_sha == canonical_sha256(_manifest_json(pending))
        for event in resumed.lineage.events
    )
    state = json.loads(
        (run_root / "private" / "pending_edits.json").read_text("utf-8")
    )
    assert state["edits"] == []
