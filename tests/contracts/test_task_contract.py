from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1


def test_canonical_task_sha_and_semantics():
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
