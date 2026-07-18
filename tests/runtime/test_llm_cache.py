from dataclasses import replace

import pytest

from SelfEvolvingHarnessTS.runtime.agent_backend import (
    AgentRequest,
    AgentResponse,
    ReplayAgentBackend,
)
from SelfEvolvingHarnessTS.runtime.llm_cache import CachedAgentBackend, EffectiveRequestCache

@pytest.fixture
def agent_request():
    return AgentRequest.for_stage(
        case_id="case-1",
        role="fast",
        stage="select",
        call_index=0,
        replicate_id="r0",
        messages=(
            {"role": "system", "content": "Return agent-envelope/1 JSON only."},
            {"role": "user", "content": "Select from the public candidate pool."},
        ),
        envelope_schema_sha="4" * 64,
        tool_schema_sha="5" * 64,
        tool_result_schema_sha="6" * 64,
        stage_schema_sha="7" * 64,
        public_case_view_sha="1" * 64,
        effective_harness_view_sha="2" * 64,
        tool_context_sha="3" * 64,
        source_harness_snapshot_sha="e" * 64,
    )


def _identity_envelope():
    return {
        "schema_version": "agent-envelope/1",
        "kind": "stage_result",
        "stage": "select",
        "payload": {"chosen_candidate_id": "identity"},
    }


def test_full_snapshot_provenance_does_not_change_semantic_request_hash(agent_request):
    changed_provenance = replace(agent_request, source_harness_snapshot_sha="f" * 64)
    assert agent_request.semantic_request_hash() == changed_provenance.semantic_request_hash()


def test_effective_view_change_invalidates_cache(tmp_path, agent_request):
    backend = ReplayAgentBackend(
        [AgentResponse.valid(_identity_envelope(), raw_response={"id": "r1"})]
    )
    cached = CachedAgentBackend(backend, EffectiveRequestCache(tmp_path))
    first = cached.complete(agent_request)
    assert first.cache_receipt.hit is False
    changed = replace(agent_request, effective_harness_view_sha="9" * 64)
    with pytest.raises(KeyError, match="replay response exhausted"):
        cached.complete(changed)


def test_successful_malformed_response_is_cached(tmp_path, agent_request):
    malformed = AgentResponse(
        transport_ok=True,
        raw_response={"id": "r-bad", "choices": [{"message": {"content": "not-json"}}]},
        assistant_text="not-json",
        parsed_envelope=None,
        parse_status="INVALID_AGENT_ENVELOPE",
        provider_metadata={"model": "gpt-5.5"},
    )
    backend = ReplayAgentBackend([malformed])
    cached = CachedAgentBackend(backend, EffectiveRequestCache(tmp_path))
    assert cached.complete(agent_request).parse_status == "INVALID_AGENT_ENVELOPE"
    replayed = cached.complete(agent_request)
    assert replayed.raw_response["id"] == "r-bad"
    assert replayed.cache_receipt.hit is True
    assert backend.call_count == 1


def test_cache_record_contains_no_authorization_material(tmp_path, agent_request):
    backend = ReplayAgentBackend(
        [AgentResponse.valid(_identity_envelope(), raw_response={"id": "r-safe"})]
    )
    cached = CachedAgentBackend(backend, EffectiveRequestCache(tmp_path))
    cached.complete(agent_request)
    record = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "api_key" not in record
    assert "authorization" not in record.lower()
