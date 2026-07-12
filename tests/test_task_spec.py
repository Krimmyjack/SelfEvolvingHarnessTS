"""P0 契约测试：TaskSpec / MetricSpec（policy/task_spec.py）。

P0 目的：readiness 是 task/metric/model 条件化的（C6 FAIL + classify C1 符号翻转），
packet 与 grammar 必须显式携带任务契约，不再隐式 forecast。
"""
import json

import pytest

from SelfEvolvingHarnessTS.policy.task_spec import (
    LABEL_AVAILABILITY,
    MetricSpec,
    TASK_TYPES,
    TaskSpec,
    anomaly_task_spec_v1,
    classification_task_spec_v1,
    forecast_task_spec_v1,
)


def test_task_types_align_with_action_layer_order():
    from SelfEvolvingHarnessTS.policy.action_spec import _TASK_ORDER
    assert TASK_TYPES == _TASK_ORDER


def test_forecast_default_spec_fields_and_serialization():
    spec = forecast_task_spec_v1()
    assert spec.task_type == "forecast"
    assert spec.target_semantics == "future_values"
    assert spec.label_availability == "history_only"
    assert spec.metric.direction == "lower_is_better"
    assert spec.horizon is None
    d = spec.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["metric"]["name"] == spec.metric.name
    assert d["task_type"] == "forecast"


def test_classification_and_anomaly_defaults():
    clf = classification_task_spec_v1()
    assert clf.task_type == "classification"
    assert clf.target_semantics == "class_label"
    assert clf.label_availability == "train_labels"
    assert clf.metric.direction == "higher_is_better"
    ad = anomaly_task_spec_v1()
    assert ad.task_type == "anomaly_detection"
    assert ad.target_semantics == "anomaly_events"
    assert ad.label_availability == "unlabeled"


def test_invalid_task_type_rejected():
    with pytest.raises(ValueError):
        TaskSpec(
            task_type="regression",
            target_semantics="future_values",
            label_availability="history_only",
            metric=MetricSpec("nRMSE", "lower_is_better"),
            horizon=None,
            downstream_model_class="x",
            forbidden_modifications=(),
        )


def test_target_semantics_must_match_task_type():
    with pytest.raises(ValueError):
        TaskSpec(
            task_type="classification",
            target_semantics="future_values",
            label_availability="train_labels",
            metric=MetricSpec("accuracy", "higher_is_better"),
            horizon=None,
            downstream_model_class="x",
            forbidden_modifications=(),
        )


def test_label_availability_whitelist():
    assert set(LABEL_AVAILABILITY) == {"history_only", "train_labels", "unlabeled"}
    with pytest.raises(ValueError):
        forecast_task_spec_v1(label_availability="oracle_labels")


def test_horizon_only_for_forecast_and_positive():
    assert forecast_task_spec_v1(horizon=96).horizon == 96
    with pytest.raises(ValueError):
        forecast_task_spec_v1(horizon=0)
    with pytest.raises(ValueError):
        TaskSpec(
            task_type="classification",
            target_semantics="class_label",
            label_availability="train_labels",
            metric=MetricSpec("accuracy", "higher_is_better"),
            horizon=24,
            downstream_model_class="x",
            forbidden_modifications=(),
        )


def test_metric_direction_validated():
    with pytest.raises(ValueError):
        MetricSpec("nRMSE", "sideways")


def test_sha_stable_and_field_sensitive():
    a = forecast_task_spec_v1()
    b = forecast_task_spec_v1()
    assert a.sha() == b.sha()
    c = forecast_task_spec_v1(horizon=96)
    assert c.sha() != a.sha()
    d = classification_task_spec_v1()
    assert d.sha() != a.sha()


def test_forbidden_modifications_respect_registry_aliases():
    spec = forecast_task_spec_v1(forbidden_modifications=("impute_linear",))
    assert spec.is_op_forbidden("impute_linear")
    assert spec.is_op_forbidden("fill_gaps")  # 旧 alias ≡ impute_linear（S0.7-6 契约）
    assert not spec.is_op_forbidden("denoise_median")
    empty = forecast_task_spec_v1()
    assert not empty.is_op_forbidden("winsorize")
