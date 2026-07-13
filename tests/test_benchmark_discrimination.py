from __future__ import annotations

import pytest

from SelfEvolvingHarnessTS.benchmark.report import (
    DevDiscriminationRow,
    ReportProtocolError,
    build_dev_discrimination_report,
)


def _rows(role="dev_query", count=12, oracle_loss=0.99):
    return [
        DevDiscriminationRow(role, "d", "r", f"u{i}", 1.0, oracle_loss)
        for i in range(count)
    ]


def test_saturation_is_created_only_from_dev_query():
    report = build_dev_discrimination_report(_rows(), saturation_gap=0.02)
    assert report["split_role"] == "dev_query"
    assert report["cells"]["d|r"]["tag"] == "saturated_under_pool_v1"
    with pytest.raises(ReportProtocolError, match="Dev-Query"):
        build_dev_discrimination_report(_rows(role="final_query"), saturation_gap=0.02)


def test_small_dev_cell_is_retained_but_not_diagnosed():
    report = build_dev_discrimination_report(_rows(count=11), min_uid=12)
    assert report["cells"]["d|r"]["tag"] == "diagnostic_unavailable"
    assert report["cells"]["d|r"]["n_uid"] == 11


def test_non_saturated_cell_reports_h_ref_to_oracle_gain():
    report = build_dev_discrimination_report(_rows(oracle_loss=0.8))
    cell = report["cells"]["d|r"]
    assert cell["oracle_insample_gain_over_h_ref"] == pytest.approx(0.2)
    assert cell["tag"] == "discriminating_under_pool_v1"

