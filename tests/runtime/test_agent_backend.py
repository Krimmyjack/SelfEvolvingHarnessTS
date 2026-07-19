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
