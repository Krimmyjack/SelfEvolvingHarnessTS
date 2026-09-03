from types import SimpleNamespace
import sys

import pytest

from SelfEvolvingHarnessTS.runtime.agent_backend import (
    AgentRequest,
    AgentCallBudgetExceeded,
    AgentTransportError,
    AgictoChatCompletionsBackend,
    BudgetedAgentBackend,
)


class FakeCompletions:
    def __init__(self, content=None):
        self.calls = []
        self.content = content or (
            '{"schema_version":"agent-envelope/1",'
            '"kind":"stage_result","stage":"select",'
            '"payload":{"chosen_candidate_id":"identity"}}'
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            id="chatcmpl-m0-1",
            model="gpt-5.5",
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=8),
            model_dump=lambda **_: {
                "id": "chatcmpl-m0-1",
                "model": "gpt-5.5",
                "choices": [
                    {"message": {"content": message.content}, "finish_reason": "stop"}
                ],
            },
        )


def request_for_stage(**changes):
    values = {
        "case_id": "case-1",
        "role": "fast",
        "stage": "select",
        "call_index": 0,
        "replicate_id": "r0",
        "messages": (
            {"role": "system", "content": "Return agent-envelope/1 JSON only."},
            {"role": "user", "content": "Select from the public candidate pool."},
        ),
        "envelope_schema_sha": "4" * 64,
        "tool_schema_sha": "5" * 64,
        "tool_result_schema_sha": "6" * 64,
        "stage_schema_sha": "7" * 64,
        "public_case_view_sha": "1" * 64,
        "effective_harness_view_sha": "2" * 64,
        "tool_context_sha": "3" * 64,
    }
    values.update(changes)
    return AgentRequest.for_stage(**values)


def test_chat_request_uses_relay_alias_and_no_unproven_provider_features():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = AgictoChatCompletionsBackend(client=client)
    request = request_for_stage()

    result = backend.complete(request)
    payload = completions.calls[0]
    assert payload == {"model": "gpt-5.5", "messages": list(request.messages)}
    assert result.parsed_envelope["payload"]["chosen_candidate_id"] == "identity"


def test_constructor_passes_key_only_to_sdk_and_repr_does_not_expose_it(monkeypatch):
    calls = []
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    def fake_openai(**kwargs):
        calls.append(kwargs)
        return fake_client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    secret = "unit-test-secret"
    backend = AgictoChatCompletionsBackend(api_key=secret, timeout_seconds=17)
    assert calls == [
        {
            "api_key": secret,
            "base_url": "https://api.agicto.cn/v1",
            "timeout": 17,
        }
    ]
    assert secret not in repr(backend)


def test_successful_non_json_response_is_agent_behavior_not_transport_failure():
    completions = FakeCompletions(content="not-json")
    backend = AgictoChatCompletionsBackend(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    response = backend.complete(request_for_stage())
    assert response.transport_ok is True
    assert response.parse_status == "INVALID_AGENT_ENVELOPE"
    assert response.parsed_envelope is None


def test_recognized_timeout_is_mapped_to_transport_error():
    class TimeoutCompletions:
        def create(self, **_):
            raise TimeoutError("relay timed out")

    backend = AgictoChatCompletionsBackend(
        client=SimpleNamespace(chat=SimpleNamespace(completions=TimeoutCompletions()))
    )
    with pytest.raises(AgentTransportError, match="relay transport failed"):
        backend.complete(request_for_stage())


def test_budgeted_backend_counts_usage_and_hard_fails_before_extra_call():
    completions = FakeCompletions()
    delegate = AgictoChatCompletionsBackend(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    backend = BudgetedAgentBackend(delegate, maximum_calls=1)
    backend.complete(request_for_stage())
    assert backend.calls == 1
    assert backend.prompt_tokens == 10
    assert backend.completion_tokens == 8
    assert backend.returned_models == {"gpt-5.5"}
    with pytest.raises(AgentCallBudgetExceeded):
        backend.complete(request_for_stage(call_index=1))
    assert len(completions.calls) == 1


def test_task_context_changes_semantic_request_but_run_provenance_does_not():
    legacy = request_for_stage()
    task_bound = request_for_stage(task_context_sha="8" * 64)
    other_run = request_for_stage(
        task_context_sha="8" * 64,
        run_context_sha="9" * 64,
    )

    assert legacy.semantic_request_hash() != task_bound.semantic_request_hash()
    assert task_bound.semantic_request_hash() == other_run.semantic_request_hash()


# ---- P0 窄信封救援（2026-08-14，用户裁决：先零 LLM 单测再重跑 G1）----

from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    parse_agent_envelope,
    rescue_trailing_envelope,
)


def _tool_request(name="localize_regions", call_id="localize_regions_1"):
    return (
        '{"schema_version":"agent-envelope/1","kind":"tool_request",'
        f'"call_id":"{call_id}","tool_name":"{name}","arguments":{{}}}}'
    )


def _stage_result(payload='{"candidates":[]}', stage="propose"):
    return (
        '{"schema_version":"agent-envelope/1","kind":"stage_result",'
        f'"stage":"{stage}","payload":{payload}}}'
    )


# 真实观察样本 1（2026-08-14 B0 T100@600 rep0 propose 首响应）：
# tool_request localize_regions ×2（同 call_id 连发）
OBSERVED_DOUBLE_TOOL_REQUEST = _tool_request() + _tool_request()

# 真实观察样本 2（2026-08-14 B1 T100@600 rep1 propose 首响应形态）：
# tool_request summarize_series + 推测性 stage_result（空候选）
OBSERVED_TOOL_THEN_STAGE_RESULT = (
    _tool_request(name="summarize_series", call_id="summarize_series_1")
    + _stage_result()
)


def test_rescue_recovers_observed_double_tool_request():
    text, code = rescue_trailing_envelope(OBSERVED_DOUBLE_TOOL_REQUEST)
    assert code == "RECOVERED_TRAILING_ENVELOPE"
    # 首信封规范化文本（canonical 键序；不含第二个）
    envelope, status = parse_agent_envelope(text)
    assert status == "VALID_AGENT_ENVELOPE"
    assert envelope["kind"] == "tool_request"
    assert envelope["tool_name"] == "localize_regions"
    assert text.count("tool_request") == 1


def test_rescue_recovers_observed_tool_then_stage_result():
    text, code = rescue_trailing_envelope(OBSERVED_TOOL_THEN_STAGE_RESULT)
    assert code == "RECOVERED_TRAILING_ENVELOPE"
    envelope, status = parse_agent_envelope(text)
    assert status == "VALID_AGENT_ENVELOPE"
    assert envelope["tool_name"] == "summarize_series"
    # 第二个 speculative envelope 被丢弃
    assert "stage_result" not in text


def test_rescue_never_accepts_stage_result_first():
    double_stage = _stage_result() + _stage_result()
    assert rescue_trailing_envelope(double_stage) == (None, "NOT_RESCUED")


def test_rescue_rejects_garbage_second_document():
    text, code = rescue_trailing_envelope(
        _tool_request() + '{"kind":"not_an_envelope"}')
    assert (text, code) == (None, "NOT_RESCUED")


def test_rescue_rejects_three_documents_and_interleaved_text():
    assert rescue_trailing_envelope(
        _tool_request() + _tool_request() + _tool_request()
    ) == (None, "NOT_RESCUED")
    assert rescue_trailing_envelope(
        _tool_request() + " some prose " + _tool_request()
    ) == (None, "NOT_RESCUED")


def test_rescue_not_triggered_for_valid_single_document():
    valid = _tool_request()
    envelope, status = parse_agent_envelope(valid)
    assert status == "VALID_AGENT_ENVELOPE"
    assert rescue_trailing_envelope(valid) == (None, "NOT_RESCUED")


def test_backend_complete_rescues_double_json_and_normalizes_history():
    from SelfEvolvingHarnessTS.runtime.agent_backend import AgictoChatCompletionsBackend as _B

    class DoubleJSONCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                id="chatcmpl-m0-2",
                model="gpt-5.5",
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content=OBSERVED_DOUBLE_TOOL_REQUEST),
                    finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10,
                                      completion_tokens=8),
            )

    backend = _B(client=SimpleNamespace(
        chat=SimpleNamespace(completions=DoubleJSONCompletions())))
    response = backend.complete(request_for_stage())
    assert response.parse_status == "VALID_AGENT_ENVELOPE"
    assert response.parse_recovery == "RECOVERED_TRAILING_ENVELOPE"
    assert response.parsed_envelope["kind"] == "tool_request"
    # 对话写回的 assistant_text 必须是首信封（canonical 键序），不含原始双 JSON
    rewritten, status = parse_agent_envelope(response.assistant_text)
    assert status == "VALID_AGENT_ENVELOPE"
    assert rewritten["tool_name"] == "localize_regions"
    assert rewritten["call_id"] == "localize_regions_1"
    assert response.assistant_text.count("tool_request") == 1
