from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.benchmark.dev_eval import (
    aggregate_per_dose,
    apply_fixed_program,
    bind_dev_report_to_manifest,
    canonical_evaluation_context,
    oracle_transfer_with_coverage,
)
from SelfEvolvingHarnessTS.benchmark.baselines import ProgramLoss


def test_fixed_programs_preserve_length_and_raw_is_noop():
    values = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])
    raw = apply_fixed_program("raw", values, period=2)
    forward = apply_fixed_program("forward_fill", values, period=2)
    seasonal = apply_fixed_program("seasonal_fill", values, period=2)
    assert np.array_equal(raw, values, equal_nan=True)
    assert len(forward) == len(seasonal) == len(values)
    assert np.isfinite(forward).all()
    assert np.isfinite(seasonal).all()
    assert seasonal[2] == 1.0


def test_per_dose_disclosure_folds_replicates_but_not_scenario_or_dose():
    rows = [
        {"program_id": "raw", "cell_id": "d|r", "scenario": "block", "dose": 0.12, "loss": 1.0},
        {"program_id": "raw", "cell_id": "d|r", "scenario": "block", "dose": 0.12, "loss": 3.0},
        {"program_id": "raw", "cell_id": "d|r", "scenario": "scattered", "dose": 0.12, "loss": 5.0},
    ]
    table = aggregate_per_dose(rows)
    assert len(table) == 2
    assert table[0]["mean_smase"] == 2.0
    assert {row["scenario"] for row in table} == {"block", "scattered"}


def test_evaluation_context_ingests_full_history_before_taking_lookback():
    history = np.concatenate([np.arange(100, dtype=float), np.full(48, np.nan)])
    context, fill_rate = canonical_evaluation_context(history, lookback=48)
    assert np.isfinite(context).all()
    assert context.tolist() == [99.0] * 48
    assert fill_rate == 48 / 148


def test_transfer_oracle_discloses_query_cells_missing_from_support():
    support = [ProgramLoss("support_a", "a", "raw", "s", 1.0)]
    query = [
        ProgramLoss("dev_query", "a", "raw", "q1", 2.0),
        ProgramLoss("dev_query", "b", "raw", "q2", 3.0),
    ]
    selected, missing = oracle_transfer_with_coverage(support, query)
    assert [row.uid for row in selected] == ["q1"]
    assert missing == ("b",)


def test_dev_report_hash_and_numeric_timeouts_are_bound_to_sealed_manifest(tmp_path):
    path = tmp_path / "benchmark_manifest_v0.yaml"
    path.write_text('{"final_query_state":"sealed"}\n', encoding="utf-8")
    digest = bind_dev_report_to_manifest(
        path,
        b'{"split_role":"dev_query"}\n',
        {"prepare_p95_x2": 1.0, "trainer_p95_x2": 2.0},
    )
    payload = __import__("json").loads(path.read_text("utf-8"))
    assert payload["dev_discrimination_report_sha256"] == digest
    assert payload["final_query_state"] == "sealed"
    assert payload["timeouts_seconds"]["trainer"] == 2.0
