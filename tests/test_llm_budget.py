import uuid

import pytest

from SelfEvolvingHarnessTS.llm import client as llm_client
from SelfEvolvingHarnessTS.llm.client import LLMClient


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "OK"}}]}


def _client_name(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def test_llm_client_stops_new_api_calls_at_call_budget(monkeypatch):
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: _FakeResponse())
    c = LLMClient(cache_name=_client_name("budget_calls"), temperature=0.0, max_api_calls=1)

    assert c("SYS", "USER-1") == "OK"
    with pytest.raises(RuntimeError, match="LLM budget exceeded"):
        c("SYS", "USER-2")

    assert c("SYS", "USER-1") == "OK"
    assert c.n_api == 1
    assert c.n_hit == 1


def test_llm_client_stops_before_estimated_cost_budget(monkeypatch):
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: _FakeResponse())
    c = LLMClient(
        cache_name=_client_name("budget_cost"),
        temperature=0.0,
        max_cost_usd=0.015,
        estimated_cost_per_call_usd=0.01,
    )

    assert c("SYS", "USER-1") == "OK"
    with pytest.raises(RuntimeError, match="cost"):
        c("SYS", "USER-2")

    stats = c.stats()
    assert stats["estimated_cost_usd"] == pytest.approx(0.01)
