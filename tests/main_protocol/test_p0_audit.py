from __future__ import annotations

from zipfile import ZipFile as StdlibZipFile

import pytest

from evaluation.main_protocol_p0 import run_p0


@pytest.fixture(scope="session")
def p0_report() -> dict:
    """Execute the audit once; it is intentionally a fail-closed readiness test."""
    return run_p0.build_report()


def test_protocol_is_locked_and_only_p1_is_released(p0_report: dict) -> None:
    assert p0_report["protocol_lock"]["status"] == "PASS"
    assert p0_report["gates"]["supersession"]["status"] == "PASS"
    assert p0_report["verdict"]["audit"] == "P0B_COMPLETE"
    assert p0_report["verdict"]["execution"] == "P0B_PASS__P1_BASELINE_SMOKE_RELEASED"
    assert p0_report["verdict"]["p1_release"] is True
    assert p0_report["verdict"]["live_outcome_release"] is False


def test_fresh_ucr_adapter_uses_train_only_and_macro_f1(p0_report: dict) -> None:
    assert p0_report["gates"]["exposure_fresh_pool"]["status"] == "PASS"
    assert p0_report["gates"]["adapter"]["status"] == "PASS"
    classification = next(
        row
        for row in p0_report["gates"]["adapter"]["tasks"]
        if row["task"] == "classification"
    )
    assert classification["status"] == "PASS_TRAIN_ONLY"
    assert [row["dataset"] for row in classification["checks"]] == [
        "Adiac",
        "ArrowHead",
    ]
    for check in classification["checks"]:
        assert check["primary_metric"] == "Macro-F1"
        assert check["archive"]["test_member_bytes_read"] is False
        assert check["split"]["surface_rows"]["fit"] > 0
        assert check["split"]["surface_rows"]["support_a"] > 0
        assert check["split"]["surface_rows"]["support_b"] > 0
    assert p0_report["sealed_read_invariants"]["ucr_test_member_bytes"] == 0
    assert p0_report["sealed_read_invariants"]["yahoo_sealed_41_csv_bytes"] == 0


def test_program_inventory_is_descriptive_and_does_not_drive_platform_work(
    p0_report: dict,
) -> None:
    gate = p0_report["gates"]["program_space"]
    assert gate["status"] == "PASS_DESCRIPTIVE_INVENTORY"
    assert gate["coverage_is_a_release_gate"] is False
    assert gate["minimum_p_effect"] is None
    by_task = {row["task"]: row for row in gate["tasks"]}
    for row in by_task.values():
        inventory = row["actual_single_step_inventory"]
        assert inventory["b_main"] == 4
        assert inventory["p_effect"] >= inventory["identity"]
        assert inventory["actual_coverage_percent"] > 0
        assert "p0_two_step_execution_probe" not in row


def test_minimal_baselines_and_cost_accounting_are_complete(p0_report: dict) -> None:
    baseline = p0_report["gates"]["baseline_smoke"]
    assert baseline["status"] == "PASS_MINIMAL_CONTRACT_SMOKE"
    assert baseline["p1_full_core_smoke"] == "PENDING"
    assert all(task["status"] == "PASS" for task in baseline["tasks"])
    assert all(task["k0_a5_same_initial_history"] for task in baseline["tasks"])
    assert baseline["common_contract"]["query_reads"] == 0
    assert baseline["common_contract"]["performance_or_headroom_claim"] is False

    cost = p0_report["gates"]["cost"]
    assert cost["status"] == "PASS_COST_ACCOUNTING_FREEZE"
    assert len(cost["method_roster"]["nonadaptive"]) == 8
    assert len(cost["method_roster"]["adaptive"]) == 5
    assert cost["totals"]["full_support_logical_evaluations_cap"] == 3630
    assert cost["p0b_observed_smoke"]["final_outcome_reads"] == 0


def test_ucr_parser_never_reads_test_member(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "Toy.zip"
    train = (
        "@problemName Toy\n"
        "@timestamps false\n"
        "@univariate true\n"
        "@classLabel true a b\n"
        "@data\n"
        "1,2,4,8:a\n"
        "2,3,5,9:b\n"
    )
    with StdlibZipFile(archive_path, "w") as archive:
        archive.writestr("Toy/Toy_TRAIN.ts", train)
        archive.writestr("Toy/Toy_TEST.ts", "this payload must remain unread")

    reads: list[str] = []

    class RecordingZipFile(StdlibZipFile):
        def read(self, name, *args, **kwargs):
            reads.append(name.filename if hasattr(name, "filename") else str(name))
            return super().read(name, *args, **kwargs)

    monkeypatch.setattr(run_p0, "ZipFile", RecordingZipFile)
    values, labels, manifest = run_p0._parse_ucr_train("Toy", archive_path)

    assert values.shape == (2, 4)
    assert labels.tolist() == [0, 1]
    assert reads == ["Toy/Toy_TRAIN.ts"]
    assert manifest["test_member_bytes_read"] is False
