from __future__ import annotations

from pathlib import Path
import json

import numpy as np

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
from SelfEvolvingHarnessTS.methods.ttha.agent_core import AgentRole, TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent


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
    current = controller.apply_to_fork(
        parent,
        manifest,
        confirmed_cause="LOCALIZATION_PROCEDURE_GAP",
    ).candidate_snapshot
    capabilities = (
        (
            "observed_missing_repair_seed_v1",
            "impute_linear",
            {"feature": "missing_fraction", "op": ">", "value": 0.0},
        ),
        (
            "local_peak_repair_seed_v1",
            "hampel_filter",
            {"feature": "local_robust_z_peak", "op": ">=", "value": 4.0},
        ),
        (
            "bounded_level_repair_seed_v1",
            "repair_level_shift",
            {"feature": "level_excursion_score", "op": ">=", "value": 3.0},
        ),
    )
    for skill_id, operator_id, applicability in capabilities:
        target = f"skill_library.entries/{skill_id}"
        definition = controller.surfaces.resolve(target).definition
        seed = EditManifest(
            edit_id=f"seed-{skill_id}",
            base_harness_sha=current.harness_content_sha,
            target_pattern_id=f"pattern-seeded-{skill_id}",
            target_surface_id=target,
            operation=EditOperation.ADD,
            surface_precondition={"kind": "ABSENT"},
            dependency_precondition_shas={
                key: current.snapshot.dependency_shas[key]
                for key in definition.required_dependency_keys
            },
            new_value={
                "schema_version": "skill-entry/1",
                "skill_id": skill_id,
                "skill_kind": "capability",
                "revision": 1,
                "body": "Test-only planted capability for PATCH cycle isolation.",
                "observable_applicability": applicability,
                "allowed_tools": [operator_id],
                "risk_guards": {
                    "max_modified_fraction": 0.25,
                    "preserve_outside_candidate_region": True,
                },
            },
            observable_applicability=applicability,
            predicted_agent_behavior_change=("identity_retained",),
            predicted_data_effect=("seed fixture only",),
            falsification_condition=("seed fixture could not be materialized",),
        )
        current = controller.apply_to_fork(
            current,
            seed,
            confirmed_cause="SKILL_LIBRARY_GAP",
        ).candidate_snapshot
    return current.root


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
    assert (result.private_root / "cluster_purity.json").is_file()
    assert (result.private_root / "cycle_instrument_metrics.json").is_file()
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


def test_selection_patch_changes_negative_probe_choice_only(tmp_path):
    store = SnapshotStore(tmp_path / "selection-snapshots")
    parent = store.materialize(compile_snapshot(H0_ROOT))
    controller = EditController(store)
    target = "candidate_policy.selection_guidance"
    definition = controller.surfaces.resolve(target).definition
    catalog = [
        {
            "surface_id": target,
            "surface_template_id": target,
            "target_class": definition.target_class,
            "surface_type": definition.surface_type,
            "allowed_operations": list(definition.allowed_operations),
            "surface_precondition": {
                "kind": "SHA",
                "sha": controller.surface_precondition_sha(parent, target),
            },
            "required_dependency_keys": list(definition.required_dependency_keys),
            "dependency_precondition_shas": {
                key: parent.snapshot.dependency_shas[key]
                for key in definition.required_dependency_keys
            },
        }
    ]
    backend = ContractPolicyBackend()
    gateway = LocalPublicToolGateway(np.zeros(192), task_kind="forecast")
    core = TTHAAgentCore(backend, gateway)
    manifest = TTHASlowAgent(core).propose_edit(
        {
            "pattern_id": "pattern-negative-probe-selection",
            "cause_code": "PROBE_SELECTION_CONTRADICTION",
            "observable_signature": {
                "imputation_probe_direction": "negative"
            },
        },
        catalog,
        parent.snapshot,
    )
    assert manifest is not None
    assert manifest.target_surface_id == target
    candidate = controller.apply_to_fork(
        parent,
        manifest,
        confirmed_cause="PROBE_SELECTION_CONTRADICTION",
    ).candidate_snapshot

    features = {"imputation_probe_direction": "negative"}
    public_input = {
        "features": features,
        "inspection": {"inspected_region_fractions": [[0.0, 1.0]]},
        "candidates": [
            {
                "candidate_id": "identity",
                "kind": "identity",
                "program_sha": None,
                "steps": [],
            },
            {
                "candidate_id": "agent-0",
                "kind": "program",
                "program_sha": "0" * 64,
                "steps": [{"op": "impute_linear", "params": {}}],
            },
        ],
    }
    schema = core.load_stage_schema("fast_select_v1")
    before = core.run_stage(
        role=AgentRole.FAST,
        stage="select",
        case_id="m0-0001",
        public_input=public_input,
        harness_view=resolve_harness_view(parent.snapshot, features),
        output_schema_name="fast_select_v1",
        output_schema=schema,
        source_snapshot_sha=parent.runtime_bundle_sha,
    )
    after = core.run_stage(
        role=AgentRole.FAST,
        stage="select",
        case_id="m0-0001",
        public_input=public_input,
        harness_view=resolve_harness_view(candidate.snapshot, features),
        output_schema_name="fast_select_v1",
        output_schema=schema,
        source_snapshot_sha=candidate.runtime_bundle_sha,
    )
    assert before.payload["chosen_candidate_id"] == "agent-0"
    assert after.payload["chosen_candidate_id"] == "identity"


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
