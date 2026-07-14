from __future__ import annotations

import numpy as np

import pytest

from SelfEvolvingHarnessTS.benchmark.dev_eval import (
    aggregate_per_dose,
    audit_h_ref_behaviour,
    bind_dev_report_to_manifest,
    canonical_evaluation_context,
    dual_headroom,
    fold_to_headline,
    mechanism_panel,
    oracle_transfer_with_coverage,
)
from SelfEvolvingHarnessTS.benchmark.baselines import ProgramLoss
from SelfEvolvingHarnessTS.benchmark.programs import apply_program


def test_headline_fold_is_cell_equal_not_series_micro():
    # A big cell (10 series, loss 1.0) and a small one (1 series, loss 11.0). The micro
    # mean is dragged to ~1.9 by sheer count; the frozen ladder weights the cells equally.
    rows = [
        ProgramLoss("dev_query", "big|regime", "raw", f"b{i}", 1.0) for i in range(10)
    ] + [ProgramLoss("dev_query", "small|regime", "raw", "s0", 11.0)]

    fold = fold_to_headline(rows)
    assert fold["by_cell_series_equal"] == {"big|regime": 1.0, "small|regime": 11.0}
    # One regime, two datasets -> dataset macro mean = 6.0.
    assert fold["by_regime_dataset_macro"]["regime"] == pytest.approx(6.0)
    assert fold["overall"] == pytest.approx(6.0)
    assert fold["series_micro_descriptive"] == pytest.approx(21.0 / 11.0)
    assert fold["overall"] != fold["series_micro_descriptive"]


def test_h_ref_audit_separates_no_op_from_help_and_harm():
    raw = [
        ProgramLoss("dev_query", "c|r", "raw", "same", 1.0),
        ProgramLoss("dev_query", "c|r", "raw", "better", 2.0),
        ProgramLoss("dev_query", "c|r", "raw", "worse", 1.0),
    ]
    h_ref = [
        ProgramLoss("dev_query", "c|r", "h_ref", "same", 1.0),
        ProgramLoss("dev_query", "c|r", "h_ref", "better", 1.5),
        ProgramLoss("dev_query", "c|r", "h_ref", "worse", 3.0),
    ]
    audit = audit_h_ref_behaviour(raw, h_ref)
    assert audit["indistinguishable_from_raw"] == 1
    assert audit["better_than_raw"] == 1
    assert audit["worse_than_raw"] == 1
    assert audit["mean_damage_where_worse"] == pytest.approx(2.0)
    # Net is positive: on this cell the reference ladder is worse than doing nothing.
    assert audit["net_vs_raw_series_micro"] > 0


def test_headroom_flags_the_cell_where_the_oracle_only_undoes_h_ref_harm():
    # H_ref hurt this cell (2.0 vs Raw 1.0). The oracle picks Raw back, so its "gain over
    # H_ref" is exactly H_ref's damage refunded -- and the true gain over Raw is zero.
    raw = [ProgramLoss("dev_query", "hurt|r", "raw", "u", 1.0)]
    h_ref = [ProgramLoss("dev_query", "hurt|r", "h_ref", "u", 2.0)]
    insample = [ProgramLoss("dev_query", "hurt|r", "raw", "u", 1.0)]

    headroom = dual_headroom(raw, h_ref, insample)
    cell = headroom["cells"]["hurt|r"]
    assert cell["oracle_reverts_to_raw"] is True
    assert cell["gain_over_h_ref"] == pytest.approx(1.0)
    assert cell["gain_over_raw"] == pytest.approx(0.0)
    assert cell["h_ref_self_harm"] == pytest.approx(1.0)
    assert headroom["cells_where_oracle_reverts_to_raw"] == ["hurt|r"]


def test_fixed_programs_preserve_length_and_raw_is_noop():
    values = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])
    raw = apply_program("raw", values, period=2)
    forward = apply_program("forward_fill", values, period=2)
    seasonal = apply_program("seasonal_fill", values, period=2)
    assert np.array_equal(raw, values, equal_nan=True)
    assert len(forward) == len(seasonal) == len(values)
    assert np.isfinite(forward).all()
    assert np.isfinite(seasonal).all()
    assert seasonal[2] == 1.0


def test_retrained_oracle_headroom_reports_the_picks_it_actually_made():
    # A retrained oracle carries its own id on every row, because it picks a program per
    # (cell, scenario, dose). The headroom table must therefore be told what it picked --
    # otherwise the "oracle reverted to Raw" flag, which is what separates real repair
    # space from H_ref's damage being refunded, silently reads False forever.
    raw = [ProgramLoss("dev_query", "hurt|r", "raw", "u", 1.0)]
    h_ref = [ProgramLoss("dev_query", "hurt|r", "h_ref", "u", 2.0)]
    oracle = [ProgramLoss("dev_query", "hurt|r", "oracle_transfer_retrained", "u", 1.0)]

    blind = dual_headroom(raw, h_ref, oracle)
    assert blind["cells"]["hurt|r"]["oracle_reverts_to_raw"] is False

    informed = dual_headroom(
        raw, h_ref, oracle, selection_by_cell={"hurt|r": ["raw", "raw", "winsorize"]}
    )
    cell = informed["cells"]["hurt|r"]
    assert cell["oracle_reverts_to_raw"] is True
    assert cell["oracle_raw_pick_share"] == pytest.approx(2 / 3)
    assert cell["gain_over_raw"] == pytest.approx(0.0)
    assert cell["gain_over_h_ref"] == pytest.approx(1.0)


def test_mechanism_panel_flags_the_cells_where_no_program_can_act():
    # This is the v0.1 finding, promoted to a permanent guard: when every program in the
    # pool scores identically on a defect, the pool has no action for it. That has to read
    # as a capability gap, never as "the data is saturated".
    rows = [
        {"program_id": p, "cell_id": "ds|r", "scenario": "level_shift", "dose": 0.05, "loss": 2.0}
        for p in ("raw", "forward_fill", "winsorize")
    ] + [
        {"program_id": "raw", "cell_id": "ds|r", "scenario": "block", "dose": 0.24, "loss": 3.0},
        {"program_id": "forward_fill", "cell_id": "ds|r", "scenario": "block", "dose": 0.24, "loss": 2.0},
        {"program_id": "winsorize", "cell_id": "ds|r", "scenario": "block", "dose": 0.24, "loss": 4.0},
    ]
    panel = mechanism_panel(rows)
    assert panel["cells_where_pool_cannot_act"] == ["ds|level_shift|0.05"]
    live = [row for row in panel["rows"] if row["scenario"] == "block"][0]
    assert live["programs_indistinguishable"] is False
    assert live["best_pool_program"] == "forward_fill"
    assert live["mechanism_of_best_program"] == "missing"


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


def test_timeout_binding_rejects_non_numeric_annotations(tmp_path):
    path = tmp_path / "benchmark_manifest_v0.yaml"
    path.write_text('{"final_query_state":"sealed"}\n', encoding="utf-8")
    # Timeouts are an enforcement threshold, so the binder takes numbers and nothing else.
    # Prose about how they were measured belongs in its own report key.
    with pytest.raises(ValueError, match="positive prepare/trainer"):
        bind_dev_report_to_manifest(
            path,
            b'{"split_role":"dev_query"}\n',
            {
                "prepare_p95_x2": 1.0,
                "trainer_p95_x2": 2.0,
                "trainer_scope": "closed_form_only",
            },
        )


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
