from __future__ import annotations

from pathlib import Path

from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import run_cycles
from SelfEvolvingHarnessTS.evaluation.minipipe.fixtures.contract_policy import (
    ContractPolicyBackend,
    DeterministicContractValuator,
)


def _run(root: Path, *, cycles: int = 2, resume: bool = False):
    return run_cycles(
        cycles=cycles,
        run_root=root,
        backend=ContractPolicyBackend(),
        valuator=DeterministicContractValuator(),
        resume=resume,
    )


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
