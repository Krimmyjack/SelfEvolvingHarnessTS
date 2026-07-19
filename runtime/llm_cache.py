from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)

from .agent_backend import (
    AgentBackend,
    AgentRequest,
    AgentResponse,
    AgentTransportError,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


@dataclass(frozen=True)
class CacheKey:
    case_id: str
    role: str
    stage: str
    call_index: int
    replicate_id: str
    semantic_request_hash: str

    @classmethod
    def from_request(cls, request: AgentRequest) -> "CacheKey":
        return cls(
            case_id=request.case_id,
            role=request.role,
            stage=request.stage,
            call_index=request.call_index,
            replicate_id=request.replicate_id,
            semantic_request_hash=request.semantic_request_hash(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "stage": self.stage,
            "call_index": self.call_index,
            "replicate_id": self.replicate_id,
            "semantic_request_hash": self.semantic_request_hash,
        }

    def sha(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class CacheReceipt:
    hit: bool
    key_sha: str
    response_hash: str


def _response_payload(response: AgentResponse) -> dict[str, object]:
    return {
        "transport_ok": response.transport_ok,
        "raw_response": _plain(response.raw_response),
        "assistant_text": response.assistant_text,
        "parsed_envelope": _plain(response.parsed_envelope),
        "parse_status": response.parse_status,
        "finish_reason": response.finish_reason,
        "provider_metadata": _plain(response.provider_metadata),
    }


def _response_hash(response: AgentResponse) -> str:
    return canonical_sha256(_response_payload(response))


def _response_from_payload(payload: Mapping[str, object]) -> AgentResponse:
    parsed = payload.get("parsed_envelope")
    if parsed is not None and not isinstance(parsed, Mapping):
        raise ValueError("cached parsed_envelope must be an object or null")
    return AgentResponse(
        transport_ok=bool(payload["transport_ok"]),
        raw_response=payload["raw_response"],
        assistant_text=str(payload["assistant_text"]),
        parsed_envelope=parsed,
        parse_status=str(payload["parse_status"]),
        finish_reason=str(payload.get("finish_reason", "")),
        provider_metadata=payload.get("provider_metadata", {}),
    )


class EffectiveRequestCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, request: AgentRequest) -> Path:
        return self.root / f"{CacheKey.from_request(request).sha()}.json"

    def get(self, request: AgentRequest) -> AgentResponse | None:
        key = CacheKey.from_request(request)
        path = self.root / f"{key.sha()}.json"
        if not path.is_file():
            return None
        record = parse_json_document(path.read_bytes())
        if not isinstance(record, dict) or record.get("schema_version") != "agent-cache-record/1":
            raise ValueError("invalid Agent cache record")
        if record.get("key") != key.to_dict():
            raise ValueError("Agent cache key/path mismatch")
        response_payload = record.get("response")
        if not isinstance(response_payload, dict):
            raise ValueError("Agent cache response must be an object")
        response = _response_from_payload(response_payload)
        response_hash = _response_hash(response)
        if response_hash != record.get("response_hash"):
            raise ValueError("Agent cache response hash mismatch")
        return response.with_cache_receipt(
            CacheReceipt(hit=True, key_sha=key.sha(), response_hash=response_hash)
        )

    def put(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        if not response.transport_ok:
            raise AgentTransportError("transport-failure responses are not cacheable")
        key = CacheKey.from_request(request)
        key_sha = key.sha()
        response_payload = _response_payload(response)
        response_hash = canonical_sha256(response_payload)
        raw_response = _plain(response.raw_response)
        parsed_envelope = _plain(response.parsed_envelope)
        record = {
            "schema_version": "agent-cache-record/1",
            "key": key.to_dict(),
            "source_harness_snapshot_sha": request.source_harness_snapshot_sha,
            "task_context_sha": request.task_context_sha,
            "run_context_sha": request.run_context_sha,
            "relay_origin": request.base_url,
            "requested_model_alias": request.model,
            "sdk_version": request.sdk_version,
            "capability_flags": _plain(request.capability_flags),
            "messages": _plain(request.messages),
            "schema_hashes": {
                "envelope": request.envelope_schema_sha,
                "tool": request.tool_schema_sha,
                "tool_result": request.tool_result_schema_sha,
                "stage": request.stage_schema_sha,
                "public_case_view": request.public_case_view_sha,
                "effective_harness_view": request.effective_harness_view_sha,
                "tool_context": request.tool_context_sha,
            },
            "response": response_payload,
            "raw_response_hash": canonical_sha256(raw_response),
            "semantic_response_hash": (
                canonical_sha256(parsed_envelope) if parsed_envelope is not None else None
            ),
            "response_hash": response_hash,
            "provider_metadata": _plain(response.provider_metadata),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.root / f"{key_sha}.json"
        if path.exists():
            existing = parse_json_document(path.read_bytes())
            if not isinstance(existing, dict) or existing.get("response_hash") != response_hash:
                raise ValueError("immutable cache collision")
            return response.with_cache_receipt(
                CacheReceipt(hit=False, key_sha=key_sha, response_hash=response_hash)
            )
        encoded = canonical_json_bytes(record) + b"\n"
        handle, temporary_name = tempfile.mkstemp(prefix=".agent-cache-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                existing = parse_json_document(path.read_bytes())
                if not isinstance(existing, dict) or existing.get("response_hash") != response_hash:
                    raise ValueError("immutable cache collision")
            else:
                temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return response.with_cache_receipt(
            CacheReceipt(hit=False, key_sha=key_sha, response_hash=response_hash)
        )


class CachedAgentBackend:
    def __init__(self, delegate: AgentBackend, cache: EffectiveRequestCache) -> None:
        self.delegate = delegate
        self.cache = cache

    def complete(self, request: AgentRequest) -> AgentResponse:
        cached = self.cache.get(request)
        if cached is not None:
            return cached
        response = self.delegate.complete(request)
        return self.cache.put(request, response)


__all__ = [
    "CacheKey",
    "CacheReceipt",
    "CachedAgentBackend",
    "EffectiveRequestCache",
]
