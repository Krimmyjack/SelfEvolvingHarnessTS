import json

import pytest

from SelfEvolvingHarnessTS.evaluators.pattern_batch_scan import (
    build_pattern_batch_report,
    legacy_cell_groups,
    summarize_batch_groups,
)


def _records():
    return [
        {
            "uid": "a",
            "cell": "c1",
            "origin": "S_a",
            "snr": 1.0,
            "miss_rate": 0.0,
            "X_p": [0.0] * 8,
            "L_test": {"raw": 5.0, "clean": 1.0, "smooth": 4.0},
        },
        {
            "uid": "b",
            "cell": "c1",
            "origin": "S_a",
            "snr": 1.1,
            "miss_rate": 0.0,
            "X_p": [0.1] * 8,
            "L_test": {"raw": 5.1, "clean": 1.1, "smooth": 4.2},
        },
        {
            "uid": "c",
            "cell": "c2",
            "origin": "S_b",
            "snr": -1.0,
            "miss_rate": 0.2,
            "X_p": [0.2] * 8,
            "L_test": {"raw": 2.0, "clean": 3.0, "smooth": 1.0},
        },
        {
            "uid": "d",
            "cell": "c2",
            "origin": "S_c",
            "snr": -1.2,
            "miss_rate": 0.2,
            "X_p": [0.3] * 8,
            "L_test": {"raw": 1.0, "clean": 3.0, "smooth": 2.0},
        },
    ]


def test_summarize_batch_groups_measures_response_homogeneity():
    records = _records()
    groups = {"a": "left", "b": "left", "c": "right", "d": "right"}

    summary = summarize_batch_groups(records, groups)

    assert summary["n_records"] == 4
    assert summary["n_batches"] == 2
    assert summary["oracle_agreement"] == pytest.approx(0.75)
    assert summary["family_purity"] == pytest.approx(0.75)
    assert summary["within_batch_response_var"] > 0.0
    assert {row["batch_key"] for row in summary["batch_rows"]} == {"left", "right"}


def test_legacy_cell_groups_uses_record_cell():
    assert legacy_cell_groups(_records()) == {"a": "c1", "b": "c1", "c": "c2", "d": "c2"}


def test_build_pattern_batch_report_writes_artifacts(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("\n".join(json.dumps(r) for r in _records()), encoding="utf-8")
    out_dir = tmp_path / "out"

    report = build_pattern_batch_report(records_path, out_dir, k=2)

    assert "legacy_cell" in report["summaries"]
    assert "P0_kmeans" in report["summaries"]
    assert (out_dir / "report.json").exists()
    assert (out_dir / "batch_rows.csv").exists()
    assert (out_dir / "table.md").exists()
