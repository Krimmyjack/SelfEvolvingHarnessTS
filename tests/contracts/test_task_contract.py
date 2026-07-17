from SelfEvolvingHarnessTS.contracts.task import (
    MetricSpec,
    TaskSpec,
    anomaly_task_spec_v1,
    classification_task_spec_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.policy import task_spec as legacy


def test_legacy_task_contract_is_the_canonical_contract():
    assert legacy.MetricSpec is MetricSpec
    assert legacy.TaskSpec is TaskSpec
    assert legacy.forecast_task_spec_v1 is forecast_task_spec_v1
    assert legacy.classification_task_spec_v1 is classification_task_spec_v1
    assert legacy.anomaly_task_spec_v1 is anomaly_task_spec_v1


def test_canonical_task_sha_matches_legacy_semantics():
    task = forecast_task_spec_v1(horizon=12)
    assert task.to_dict() == {
        "task_type": "forecast",
        "target_semantics": "future_values",
        "label_availability": "history_only",
        "metric": {"name": "nRMSE", "direction": "lower_is_better"},
        "horizon": 12,
        "downstream_model_class": "dlinear_shared",
        "forbidden_modifications": [],
    }
    assert len(task.sha()) == 16
