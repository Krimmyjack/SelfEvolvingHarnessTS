from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.method_api import (
    ContractVerdict,
    FeedbackAPI,
    FeedbackBudgetError,
    FeedbackHandle,
    MethodGateError,
    MethodSeriesView,
    PreparedSeries,
    PrivateSeriesEpisode,
    require_final_eligibility,
    run_support_dry_run,
    validate_prepared,
)
from SelfEvolvingHarnessTS.policy.task_spec import forecast_task_spec_v1


class NoOpMethod:
    method_id = "noop"

    def prepare(self, series_view, task_spec, observed_pattern_spec):
        return PreparedSeries(
            series_view.series_uid,
            series_view.degraded_inner_train,
            (),
            "original_units",
        )


def _episode(role: str = "support_a") -> PrivateSeriesEpisode:
    return PrivateSeriesEpisode(
        series_uid="u",
        degraded_inner_train=np.arange(20.0),
        future=np.arange(20.0, 24.0),
        regime_tag="seasonal",
        split_role=role,
    )


def test_method_view_excludes_private_fields_and_is_immutable():
    view = MethodSeriesView.from_episode(_episode())
    assert not hasattr(view, "future")
    assert not hasattr(view, "regime_tag")
    assert not hasattr(view, "split_role")
    with pytest.raises(ValueError):
        view.degraded_inner_train[0] = 9.0


def test_target_space_transform_is_forbidden():
    candidate = PreparedSeries("u", np.arange(10.0), ("znorm",), "original_units")
    verdict = validate_prepared(candidate, expected_length=10)
    assert verdict == ContractVerdict(False, "forbidden_target_space_transform")
    alias = PreparedSeries("u", np.arange(10.0), ("sliding_window",), "original_units")
    assert validate_prepared(alias, expected_length=10).code == "forbidden_target_space_transform"


def test_prepared_contract_rejects_shape_units_infinity_and_unknown_ops():
    cases = [
        (PreparedSeries("u", np.arange(9.0), (), "original_units"), "length_changed"),
        (PreparedSeries("u", np.ones((2, 5)), (), "original_units"), "dimensionality_changed"),
        (PreparedSeries("u", np.array([1.0] * 9 + [np.inf]), (), "original_units"), "non_finite_output"),
        (PreparedSeries("u", np.arange(10.0), (), "zscore"), "units_changed"),
        (PreparedSeries("u", np.arange(10.0), ("not-an-op",), "original_units"), "unknown_operator"),
    ]
    for prepared, code in cases:
        assert validate_prepared(prepared, expected_length=10).code == code


def test_feedback_is_support_a_closed_form_budgeted_and_accounted():
    calls = []
    api = FeedbackAPI(
        budget=1,
        evaluator=lambda handle, prepared: calls.append(handle.handle_id) or 0.5,
    )
    handle = FeedbackHandle("h", "support_a", "closed_form_inner_val")
    prepared = PreparedSeries("u", np.arange(10.0), (), "original_units")
    assert api.evaluate(handle, prepared) == 0.5
    assert calls == ["h"]
    assert len(api.call_records) == 1
    with pytest.raises(FeedbackBudgetError):
        api.evaluate(handle, prepared)
    forbidden = FeedbackAPI(budget=1, evaluator=lambda *_: 0.0)
    with pytest.raises(MethodGateError, match="Support-A"):
        forbidden.evaluate(
            FeedbackHandle("q", "final_query", "closed_form_inner_val"), prepared
        )

    invalid = FeedbackAPI(budget=1, evaluator=lambda *_: float("nan"))
    with pytest.raises(MethodGateError, match="non-finite"):
        invalid.evaluate(handle, prepared)
    assert invalid.remaining_budget == 0
    assert invalid.call_records[-1].status == "invalid_result"


def test_support_dry_run_is_deterministic_and_query_requires_both_gate_shas():
    method = NoOpMethod()
    task = forecast_task_spec_v1(horizon=48)
    first = run_support_dry_run(
        method,
        [_episode()],
        task,
        method_code_sha="a" * 64,
        config={"budget": 3},
    )
    second = run_support_dry_run(
        method,
        [_episode()],
        task,
        method_code_sha="a" * 64,
        config={"budget": 3},
    )
    assert first.artifact_sha == second.artifact_sha
    with pytest.raises(MethodGateError):
        require_final_eligibility("m", dry_run_sha=None, confirmation_sha=None)
    require_final_eligibility(
        "m", dry_run_sha=first.artifact_sha, confirmation_sha="b" * 64
    )
