from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)

from .errors import InfrastructureError


DEFAULT_AGENT_MODEL = "gpt-5.5"
DEFAULT_AGENT_BASE_URL = "https://api.agicto.cn/v1"
OPENAI_SDK_VERSION = "2.45.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_CAPABILITY_FLAGS = MappingProxyType(
    {
        "native_tools": False,
        "structured_outputs": False,
        "reasoning_controls": False,
        "provider_seed": False,
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _require_sha(value: str, *, field_name: str, optional: bool = False) -> None:
    if optional and value == "":
        return
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _validate_base_url(base_url: str) -> None:
    if not isinstance(base_url, str) or base_url != base_url.strip():
        raise ValueError("base_url must be canonical")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an HTTPS origin ending in /v1")


def _validate_messages(messages: tuple[Mapping[str, object], ...]) -> None:
    if not messages:
        raise ValueError("AgentRequest requires at least one message")
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError("messages must contain role and content only")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError("unsupported message role")
        if not isinstance(message["content"], str) or not message["content"]:
            raise ValueError("message content must be a non-empty string")
    canonical_json_bytes(messages)


@dataclass(frozen=True)
class AgentRequest:
    case_id: str
    role: str
    stage: str
    call_index: int
    replicate_id: str
    messages: tuple[Mapping[str, object], ...]
    envelope_schema_sha: str
    tool_schema_sha: str
    tool_result_schema_sha: str
    stage_schema_sha: str
    public_case_view_sha: str
    effective_harness_view_sha: str
    tool_context_sha: str
    source_harness_snapshot_sha: str = ""
    model: str = DEFAULT_AGENT_MODEL
    base_url: str = DEFAULT_AGENT_BASE_URL
    sdk_version: str = OPENAI_SDK_VERSION
    capability_flags: Mapping[str, bool] = field(
        default_factory=lambda: dict(_CAPABILITY_FLAGS)
    )
    cache_schema_version: str = "effective-request/1"

    def __post_init__(self) -> None:
        for field_name in ("case_id", "replicate_id", "model", "sdk_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be canonical non-empty text")
        if self.role not in {"fast", "slow"}:
            raise ValueError("role must be fast or slow")
        if not isinstance(self.stage, str) or not _CANONICAL_NAME.fullmatch(self.stage):
            raise ValueError("stage must be a canonical name")
        if isinstance(self.call_index, bool) or not isinstance(self.call_index, int) or self.call_index < 0:
            raise ValueError("call_index must be a non-negative integer")
        _validate_base_url(self.base_url)
        for field_name in (
            "envelope_schema_sha",
            "tool_schema_sha",
            "tool_result_schema_sha",
            "stage_schema_sha",
            "public_case_view_sha",
            "effective_harness_view_sha",
            "tool_context_sha",
        ):
            _require_sha(getattr(self, field_name), field_name=field_name)
        _require_sha(
            self.source_harness_snapshot_sha,
            field_name="source_harness_snapshot_sha",
            optional=True,
        )
        messages = tuple(_freeze_json(message) for message in self.messages)
        _validate_messages(messages)
        object.__setattr__(self, "messages", messages)
        expected_flags = dict(_CAPABILITY_FLAGS)
        if _plain(self.capability_flags) != expected_flags:
            raise ValueError("M0 capability_flags are fixed and all disabled")
        object.__setattr__(self, "capability_flags", _freeze_json(expected_flags))
        if self.cache_schema_version != "effective-request/1":
            raise ValueError("unsupported cache_schema_version")

    @classmethod
    def for_stage(cls, **values: object) -> "AgentRequest":
        return cls(**values)

    def semantic_request_hash(self) -> str:
        return canonical_sha256(
            {
                "provider": "agicto-chat-completions",
                "base_url": self.base_url,
                "model": self.model,
                "sdk_version": self.sdk_version,
                "capability_flags": self.capability_flags,
                "messages": self.messages,
                "envelope_schema_sha": self.envelope_schema_sha,
                "tool_schema_sha": self.tool_schema_sha,
                "tool_result_schema_sha": self.tool_result_schema_sha,
                "stage_schema_sha": self.stage_schema_sha,
                "public_case_view_sha": self.public_case_view_sha,
                "effective_harness_view_sha": self.effective_harness_view_sha,
                "tool_context_sha": self.tool_context_sha,
                "cache_schema_version": self.cache_schema_version,
            }
        )


def _validate_envelope(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("agent envelope must be an object")
    if value.get("schema_version") != "agent-envelope/1":
        raise ValueError("agent envelope schema_version mismatch")
    kind = value.get("kind")
    if kind == "stage_result":
        if set(value) != {"schema_version", "kind", "stage", "payload"}:
            raise ValueError("stage_result envelope has unexpected fields")
        if not isinstance(value["stage"], str) or not _CANONICAL_NAME.fullmatch(value["stage"]):
            raise ValueError("stage_result stage must be canonical")
        if not isinstance(value["payload"], dict):
            raise ValueError("stage_result payload must be an object")
    elif kind == "tool_request":
        expected = {"schema_version", "kind", "call_id", "tool_name", "arguments"}
        if set(value) != expected:
            raise ValueError("tool_request envelope has unexpected fields")
        for field_name in ("call_id", "tool_name"):
            if not isinstance(value[field_name], str) or not _CANONICAL_NAME.fullmatch(value[field_name]):
                raise ValueError(f"{field_name} must be canonical")
        if not isinstance(value["arguments"], dict):
            raise ValueError("tool arguments must be an object")
    else:
        raise ValueError("unknown agent envelope kind")
    canonical_json_bytes(value)
    return value


def parse_agent_envelope(assistant_text: str) -> tuple[Mapping[str, object] | None, str]:
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return None, "INVALID_AGENT_ENVELOPE"
    try:
        parsed = parse_json_document(assistant_text.encode("utf-8"))
        envelope = _validate_envelope(parsed)
    except (TypeError, ValueError, UnicodeError):
        return None, "INVALID_AGENT_ENVELOPE"
    return _freeze_json(envelope), "VALID_AGENT_ENVELOPE"


@dataclass(frozen=True)
class AgentResponse:
    transport_ok: bool
    raw_response: Mapping[str, object]
    assistant_text: str
    parsed_envelope: Mapping[str, object] | None
    parse_status: str
    finish_reason: str = ""
    provider_metadata: Mapping[str, object] = field(default_factory=dict)
    cache_receipt: object | None = None

    def __post_init__(self) -> None:
        canonical_json_bytes(self.raw_response)
        canonical_json_bytes(self.provider_metadata)
        if self.parsed_envelope is not None:
            canonical_json_bytes(self.parsed_envelope)
        object.__setattr__(self, "raw_response", _freeze_json(self.raw_response))
        object.__setattr__(self, "provider_metadata", _freeze_json(self.provider_metadata))
        if self.parsed_envelope is not None:
            object.__setattr__(self, "parsed_envelope", _freeze_json(self.parsed_envelope))

    @classmethod
    def valid(
        cls,
        envelope: Mapping[str, object],
        *,
        raw_response: Mapping[str, object],
        provider_metadata: Mapping[str, object] | None = None,
    ) -> "AgentResponse":
        parsed = _validate_envelope(_plain(envelope))
        return cls(
            transport_ok=True,
            raw_response=raw_response,
            assistant_text=canonical_json_bytes(parsed).decode("utf-8"),
            parsed_envelope=parsed,
            parse_status="VALID_AGENT_ENVELOPE",
            provider_metadata=provider_metadata or {},
        )

    def with_cache_receipt(self, receipt: object) -> "AgentResponse":
        return replace(self, cache_receipt=receipt)


class AgentTransportError(InfrastructureError):
    """A relay request did not yield a transport-success response."""


class AgentBackend(Protocol):
    def complete(self, request: AgentRequest) -> AgentResponse:
        raise NotImplementedError


class AgictoChatCompletionsBackend:
    def __init__(
        self,
        *,
        client: object | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_AGENT_BASE_URL,
        timeout_seconds: int | float = 120,
    ) -> None:
        _validate_base_url(base_url)
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a positive finite number")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if client is None:
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("a non-empty API key is required when no client is injected")
            openai = importlib.import_module("openai")
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_seconds,
            )
        self._client = client
        self._base_url = base_url
        self._timeout_seconds = float(timeout_seconds)

    def __repr__(self) -> str:
        return (
            "AgictoChatCompletionsBackend("
            f"base_url={self._base_url!r}, timeout_seconds={self._timeout_seconds!r})"
        )

    def complete(self, request: AgentRequest) -> AgentResponse:
        if request.base_url != self._base_url:
            raise ValueError("request base_url does not match backend origin")
        try:
            completion = self._client.chat.completions.create(
                model=request.model,
                messages=[_plain(message) for message in request.messages],
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            recognized = (
                isinstance(exc, (TimeoutError, ConnectionError))
                or type(exc).__name__
                in {
                    "APIConnectionError",
                    "APITimeoutError",
                    "RateLimitError",
                }
                or status_code in {408, 409, 429}
                or isinstance(status_code, int)
                and status_code >= 500
            )
            if recognized:
                raise AgentTransportError(
                    f"relay transport failed ({type(exc).__name__})"
                ) from None
            raise
        choices = getattr(completion, "choices", ())
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        assistant_text = getattr(message, "content", "")
        if not isinstance(assistant_text, str):
            assistant_text = ""
        envelope, parse_status = parse_agent_envelope(assistant_text)
        try:
            raw_response = completion.model_dump(mode="json")
        except (AttributeError, TypeError):
            raw_response = {
                "id": getattr(completion, "id", ""),
                "model": getattr(completion, "model", ""),
            }
        usage = getattr(completion, "usage", None)
        provider_metadata = {
            "response_id": getattr(completion, "id", ""),
            "returned_model": getattr(completion, "model", ""),
            "finish_reason": getattr(choice, "finish_reason", "") if choice else "",
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            },
        }
        return AgentResponse(
            transport_ok=True,
            raw_response=_plain(raw_response),
            assistant_text=assistant_text,
            parsed_envelope=envelope,
            parse_status=parse_status,
            finish_reason=provider_metadata["finish_reason"],
            provider_metadata=provider_metadata,
        )


class ReplayAgentBackend:
    def __init__(
        self,
        responses: Sequence[AgentResponse] | Mapping[str, AgentResponse],
    ) -> None:
        if isinstance(responses, Mapping):
            self._ordered: tuple[AgentResponse, ...] | None = None
            self._mapped = dict(responses)
        else:
            self._ordered = tuple(responses)
            self._mapped: dict[str, AgentResponse] | None = None
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def complete(self, request: AgentRequest) -> AgentResponse:
        if self._ordered is not None:
            if self._call_count >= len(self._ordered):
                raise KeyError("replay response exhausted")
            response = self._ordered[self._call_count]
        else:
            assert self._mapped is not None
            semantic_hash = request.semantic_request_hash()
            if semantic_hash not in self._mapped:
                raise KeyError(f"replay miss for semantic request {semantic_hash}")
            response = self._mapped[semantic_hash]
        self._call_count += 1
        return response

    def clone(self) -> "ReplayAgentBackend":
        source: Sequence[AgentResponse] | Mapping[str, AgentResponse]
        source = self._ordered if self._ordered is not None else self._mapped or {}
        return ReplayAgentBackend(source)


__all__ = [
    "AgentBackend",
    "AgentRequest",
    "AgentResponse",
    "AgentTransportError",
    "AgictoChatCompletionsBackend",
    "DEFAULT_AGENT_BASE_URL",
    "DEFAULT_AGENT_MODEL",
    "OPENAI_SDK_VERSION",
    "ReplayAgentBackend",
    "parse_agent_envelope",
]
