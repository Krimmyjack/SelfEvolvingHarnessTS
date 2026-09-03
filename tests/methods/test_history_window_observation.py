import json

import numpy as np

from SelfEvolvingHarnessTS.evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as runner,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_autonomous_natural_workflow_generation import (
    _augment_context_with_history_observation,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse


def test_agent_tool_request_executes_and_augments_public_context():
    class CapturingBackend:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return AgentResponse.valid(
                    {
                        "schema_version": "agent-envelope/1",
                        "kind": "tool_request",
                        "call_id": "history-call",
                        "tool_name": "compare_history_windows",
                        "arguments": {},
                    },
                    raw_response={"id": "history-tool-request"},
                )
            tool_result = json.loads(request.messages[-1]["content"])
            assert tool_result["schema_version"] == "tool-result/1"
            assert tool_result["tool_name"] == "compare_history_windows"
            assert "early_to_recent_change" in tool_result["public_result"]
            return AgentResponse.valid(
                {
                    "schema_version": "agent-envelope/1",
                    "kind": "stage_result",
                    "stage": "observe",
                    "payload": {"observation_complete": True},
                },
                raw_response={"id": "history-stage-result"},
            )

    time = np.arange(384, dtype=np.float64)
    series = [
        3.0 + np.sin(2.0 * np.pi * time / 24.0),
        7.0 + 0.02 * time + np.cos(2.0 * np.pi * time / 24.0),
    ]
    series[1][20:23] = np.nan
    backend = CapturingBackend()
    original = {
        "task": {"type": "forecast"},
        "periodicity": {"calendar_period": 24},
        "capability_memory": {"entry_count": 0},
    }
    augmented, metadata = _augment_context_with_history_observation(
        original,
        series,
        calendar_period=24,
        backend=backend,
        model="gpt-5.6-luna",
        base_url="https://api.agicto.cn/v1",
    )

    comparison = augmented["observations"]["history_window_comparison"]
    assert comparison["series_count"] == 2
    assert comparison["calendar_period"] == 24
    assert comparison["early"]["missing_run_count"] == 1
    assert comparison["recent"]["missing_run_count"] == 0
    assert metadata["agent_call_count"] == 2
    assert metadata["tool_call_count"] == 1
    assert len(backend.requests) == 2

    forbidden = {
        "dataset_id",
        "series_uid",
        "clean",
        "query_outcome",
        "support_outcome",
        "filesystem_path",
    }

    def check_public(value):
        if isinstance(value, dict):
            assert forbidden.isdisjoint(key.lower() for key in value)
            for nested in value.values():
                check_public(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                check_public(nested)

    check_public(comparison)


def test_missing_required_history_tool_retries_then_completes():
    class CapturingBackend:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return AgentResponse(
                    transport_ok=True,
                    raw_response={"id": "malformed-before-required-tool"},
                    assistant_text="not-json",
                    parsed_envelope=None,
                    parse_status="INVALID_AGENT_ENVELOPE",
                )
            if len(self.requests) == 2:
                feedback = json.loads(request.messages[-1]["content"])
                assert feedback["error_code"] == "AGENT_ENVELOPE_INVALID"
                return AgentResponse.valid(
                    {
                        "schema_version": "agent-envelope/1",
                        "kind": "stage_result",
                        "stage": "observe",
                        "payload": {"observation_complete": True},
                    },
                    raw_response={"id": "skipped-required-tool"},
                )
            if len(self.requests) == 3:
                feedback = json.loads(request.messages[-1]["content"])
                assert feedback["schema_version"] == "stage-validation-error/2"
                assert feedback["error_code"] == "REQUIRED_TOOL_MISSING"
                assert '"kind":"tool_request"' in feedback["required_outer_format"]
                assert "compare_history_windows" in feedback["public_message"]
                return AgentResponse.valid(
                    {
                        "schema_version": "agent-envelope/1",
                        "kind": "tool_request",
                        "call_id": "required-history-call",
                        "tool_name": "compare_history_windows",
                        "arguments": {},
                    },
                    raw_response={"id": "required-history-tool-request"},
                )
            tool_result = json.loads(request.messages[-1]["content"])
            assert tool_result["tool_name"] == "compare_history_windows"
            return AgentResponse.valid(
                {
                    "schema_version": "agent-envelope/1",
                    "kind": "stage_result",
                    "stage": "observe",
                    "payload": {"observation_complete": True},
                },
                raw_response={"id": "history-observation-complete"},
            )

    time = np.arange(384, dtype=np.float64)
    series = [
        3.0 + np.sin(2.0 * np.pi * time / 24.0),
        7.0 + 0.02 * time + np.cos(2.0 * np.pi * time / 24.0),
    ]
    backend = CapturingBackend()
    original = {
        "task": {"type": "forecast"},
        "periodicity": {"calendar_period": 24},
        "capability_memory": {"entry_count": 0},
    }
    augmented, metadata = _augment_context_with_history_observation(
        original,
        series,
        calendar_period=24,
        backend=backend,
        model="offline-test",
        base_url="https://api.agicto.cn/v1",
    )

    assert augmented["observations"]["history_window_comparison"]["series_count"] == 2
    assert metadata["agent_call_count"] == 4
    assert metadata["tool_call_count"] == 1
    assert len(backend.requests) == 4


def test_generated_program_only_touches_training_windows(monkeypatch):
    calls = []
    candidate = object()

    def record_training_application(raw, compiled):
        assert compiled is candidate
        values = np.asarray(raw, dtype=np.float64)
        calls.append(values.size)
        return values + 0.5, [{"ok": True}]

    def zero_prediction(np_module, *, x_train, targets, weights, x_eval):
        assert x_train.shape[0] == 2
        return np_module.zeros((x_eval.shape[0], targets.shape[1]), dtype=np.float64)

    monkeypatch.setattr(runner, "_apply_program", record_training_application)
    monkeypatch.setattr(runner, "_exact_weighted_ridge_prediction", zero_prediction)

    time = np.arange(320, dtype=np.float64)
    values = {
        "train": 10.0 + 0.01 * time + np.sin(2.0 * np.pi * time / 24.0),
        "ineligible": 15.0 + 0.01 * time + np.sin(2.0 * np.pi * time / 24.0),
        "eval": 20.0 + 0.01 * time + np.cos(2.0 * np.pi * time / 24.0),
    }
    conditions = runner._compile_scope_patch(
        {
            "decision": "RESTRICT_SCOPE",
            "program_op": "generated_op",
            "predicate": {
                "all": [{"field": "recent.coverage", "op": ">=", "value": 0.9}]
            },
        },
        common_program="generated_op",
        allowed_fields=("recent.coverage",),
    )
    assert conditions is not None
    assert runner._scope_matches({"recent": {"coverage": 0.95}}, conditions)
    assert not runner._scope_matches({"recent": {"coverage": 0.5}}, conditions)
    runner._assert_scope_dossier_public(
        {
            "environment": "A",
            "within_environment_ordinal": 0,
            "public_history_summary": {"recent": {"coverage": 0.95}},
        }
    )
    with np.testing.assert_raises(ValueError):
        runner._assert_scope_dossier_public({"dataset_id": "forbidden"})

    result = runner._evaluate(
        [
            {"series_uid": "train", "role": "train"},
            {"series_uid": "ineligible", "role": "train"},
            {"series_uid": "eval", "role": "eval"},
        ],
        values,
        candidate,
        {"anchors": (240,), "period": 24},
        origin=260,
        train_series_scope={"train"},
    )

    assert calls == [runner.CONTEXT_LENGTH + runner.HORIZON]
    assert result["behavior_point_count"] == runner.CONTEXT_LENGTH + runner.HORIZON
