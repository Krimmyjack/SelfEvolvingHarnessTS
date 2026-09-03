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
# call_id 是纯关联标识：只用于同一轮内的重复检测（精确相等）和回显进
# tool-result/1（该 schema 对 call_id 无 pattern）。下游没有任何按小写
# 归一的查表、路径或名字解析，因此大小写不携带语义。实测（G2 T233 rerun）
# 模型会把序列 uid 直接拼进 call_id（inspect_T234_summary），uid 本身是
# 大写，于是四个 Task 以 AGENT_PROTOCOL_ERROR 结束。这里只放宽字母大小写，
# 不放宽首字符必须是字母、也不放宽字符集；tool_name 仍走 _CANONICAL_NAME
# ——它要去 declared_tools 里查表，是有语义的。
_CORRELATION_ID = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*$")
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
    task_context_sha: str = ""
    run_context_sha: str = ""
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
            self.task_context_sha,
            field_name="task_context_sha",
            optional=True,
        )
        _require_sha(
            self.run_context_sha,
            field_name="run_context_sha",
            optional=True,
        )
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
        identity = {
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
        if self.task_context_sha:
            identity["task_context_sha"] = self.task_context_sha
        return canonical_sha256(identity)


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
        if not isinstance(value["call_id"], str) or not _CORRELATION_ID.fullmatch(
            value["call_id"]
        ):
            raise ValueError("call_id must be canonical")
        if not isinstance(value["tool_name"], str) or not _CANONICAL_NAME.fullmatch(
            value["tool_name"]
        ):
            raise ValueError("tool_name must be canonical")
        if not isinstance(value["arguments"], dict):
            raise ValueError("tool arguments must be an object")
    elif kind == "no_proposal":
        expected = {"schema_version", "kind", "stage", "reason_code"}
        if set(value) != expected or value["stage"] != "edit":
            raise ValueError("no_proposal is valid for the edit stage only")
        if value["reason_code"] not in {
            "insufficient_public_evidence",
            "no_authorized_minimal_edit",
            "risk_too_high",
        }:
            raise ValueError("unknown no_proposal reason_code")
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


def _json_document_spans(text: str) -> list[tuple[int, int]]:
    """顶层 JSON 文档的 (start, end) 片段列表（brace 匹配 + 字符串/转义
    感知）。遇到未闭合文档即停止（其后无完整文档可言）。"""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escaped = False
        j = i
        closed = False
        while j < n:
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    spans.append((i, j + 1))
                    closed = True
                    break
            j += 1
        if not closed:
            break  # 未闭合——后续内容不再扫描
        i = j + 1
    return spans


def _whitespace_only(text: str) -> bool:
    return not text.strip()


def rescue_prose_wrapped_envelope(
    assistant_text: str,
) -> tuple[str | None, str]:
    """恢复"被普通说明文字包住的、恰好一个"合法 envelope。

    实测（G2 收口）：模型写了一句自然语言理由，后面跟一个语法完全合法的
    tool_request。严格解析拒绝，重试也拒绝，整个 Task 以
    AGENT_PROTOCOL_ERROR 结束——这不是模型给错了动作，是它多说了一句话。

    有界得很死，只补这一类：

      顶层 JSON 文档**恰好一个**，且该文档能通过现有 envelope schema。
      文档前后可以是普通文字。

    仍然拒绝：零个或两个以上 JSON 文档（多文档歧义由
    rescue_trailing_envelope 单独处理其已观察到的那一类）、文档本身不合
    schema。不放宽 schema，不放宽执行权限，不猜测模型意图——只是把它已经
    写对的那一个信封取出来。
    """
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return None, "NOT_RESCUED"
    spans = _json_document_spans(assistant_text)
    if len(spans) != 1:
        return None, "NOT_RESCUED"
    start, end = spans[0]
    try:
        document = parse_json_document(assistant_text[start:end].encode("utf-8"))
        envelope = _validate_envelope(document)
    except (ValueError, TypeError):
        return None, "NOT_RESCUED"
    return (
        canonical_json_bytes(envelope).decode("utf-8"),
        "RECOVERED_PROSE_WRAPPED_ENVELOPE",
    )


def rescue_trailing_envelope(
    assistant_text: str,
) -> tuple[str | None, str]:
    """窄信封救援（P0，用户裁决 2026-08-14）：只恢复**已观察到的**错误类——

      严格解析失败 且 恰好两个完整顶层 JSON 文档 且 两文档之间/前后只有
      空白 且 第一个是合法 tool_request 信封 且 第二个也是合法 agent
      envelope（任意 kind）且 尾部无其他非空内容。

    命中 → 返回 (规范化首信封 JSON 文本, "RECOVERED_TRAILING_ENVELOPE")；
    调用方须用首信封继续正常循环，并把对话中的 assistant_text 规范化
    为该文本（原始双 JSON 不得写回对话）。未命中 → (None, "NOT_RESCUED")。

    禁止泛化：stage_result 打头、第二个文档非法、三个及以上文档、任意
    夹杂文本均不救援（保持错误重试路径）。"""
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return None, "NOT_RESCUED"
    spans = _json_document_spans(assistant_text)
    if len(spans) != 2:
        return None, "NOT_RESCUED"
    (s1, e1), (s2, e2) = spans
    if not _whitespace_only(assistant_text[:s1]):
        return None, "NOT_RESCUED"
    if not _whitespace_only(assistant_text[e1:s2]):
        return None, "NOT_RESCUED"
    if not _whitespace_only(assistant_text[e2:]):
        return None, "NOT_RESCUED"
    try:
        first = parse_json_document(assistant_text[s1:e1].encode("utf-8"))
        _validate_envelope(first)
        second = parse_json_document(assistant_text[s2:e2].encode("utf-8"))
        _validate_envelope(second)
    except (TypeError, ValueError, UnicodeError):
        return None, "NOT_RESCUED"
    if first.get("kind") != "tool_request":
        return None, "NOT_RESCUED"
    return canonical_json_bytes(first).decode("utf-8"), "RECOVERED_TRAILING_ENVELOPE"


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
    # 窄信封救援标记（P0 2026-08-14）：空串 = 未救援；
    # "RECOVERED_TRAILING_ENVELOPE" = 双 JSON 拼接已按首 tool_request
    # 信封救援（assistant_text 已规范化为首信封）。
    parse_recovery: str = ""

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


class ReplayTapeMiss(InfrastructureError):
    """An immutable offline replay has no response for the effective request."""


class AgentBackend(Protocol):
    def complete(self, request: AgentRequest) -> AgentResponse:
        raise NotImplementedError


class AgentCallBudgetExceeded(InfrastructureError):
    """A bounded live run exhausted its preregistered relay-call budget."""


class BudgetedAgentBackend:
    def __init__(self, delegate: AgentBackend, *, maximum_calls: int) -> None:
        if (
            isinstance(maximum_calls, bool)
            or not isinstance(maximum_calls, int)
            or maximum_calls < 1
        ):
            raise ValueError("maximum_calls must be a positive integer")
        self.delegate = delegate
        self.maximum_calls = maximum_calls
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.returned_models: set[str] = set()

    def complete(self, request: AgentRequest) -> AgentResponse:
        if self.calls >= self.maximum_calls:
            raise AgentCallBudgetExceeded(
                f"Agent call budget exhausted at {self.maximum_calls}"
            )
        self.calls += 1
        response = self.delegate.complete(request)
        metadata = response.provider_metadata
        if isinstance(metadata, Mapping):
            usage = metadata.get("usage", {})
            if isinstance(usage, Mapping):
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
            returned = metadata.get("returned_model")
            if isinstance(returned, str) and returned:
                self.returned_models.add(returned)
        return response


def _relay_error_payload(completion: object) -> str | None:
    """The relay's error object, when a transport-success carries one.

    Returns a short public description when the payload has an ``error``
    member and no usable ``choices``; ``None`` otherwise, which is every
    ordinary response including a legitimately empty one.
    """
    choices = getattr(completion, "choices", None)
    if choices:
        return None
    error = getattr(completion, "error", None)
    if error is None:
        extra = getattr(completion, "model_extra", None)
        if isinstance(extra, Mapping):
            error = extra.get("error")
    if error is None:
        dump = getattr(completion, "model_dump", None)
        if callable(dump):
            try:
                payload = dump(mode="json")
            except (AttributeError, TypeError, ValueError):
                payload = None
            if isinstance(payload, Mapping):
                error = payload.get("error")
    if error is None:
        return None
    if isinstance(error, Mapping):
        code = error.get("code") or error.get("type") or "api_error"
        message = error.get("message") or ""
        return "%s: %s" % (code, str(message)[:200])
    return str(error)[:200]


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
        # 中转在上游过载时返回 HTTP 200，body 只有一个 error 对象、没有
        # choices（实测 2026-08-22：{"error":{"code":"api_error","message":
        # "Service load is too high, please try again later"}}，Claude 全族
        # 命中、同一时刻 gpt-5.6-luna 正常）。SDK 把 200 当成功解析，下面
        # 的 choices 取空、content 取空，于是一次服务宕机被构造成
        # transport_ok=True 的空回答，一路伪装成信封协议失败：
        # _RetryingTransport 不重试（没有异常），agent_core 对着空串把两次
        # 静态反馈重试用光，调用方最后读到 "invalid agent-envelope/1"。
        # #24/#25 两次 SLOW_ENVELOPE_PROTOCOL_FAILURE 都是这么来的。
        # 这里只认一件事：payload 里带 error 且没有可用 choices —— 那就是
        # 传输层没拿到答复，按 AgentTransportError 抛，让既有的退避重试和
        # INCONCLUSIVE_TRANSPORT 口径接手。正常的空回复（有 choices、
        # content 为空）不受影响，仍走原路。
        relay_error = _relay_error_payload(completion)
        if relay_error is not None:
            raise AgentTransportError(
                "relay returned an error payload with HTTP success: %s"
                % relay_error
            )
        choices = getattr(completion, "choices", ())
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        assistant_text = getattr(message, "content", "")
        if not isinstance(assistant_text, str):
            assistant_text = ""
        envelope, parse_status = parse_agent_envelope(assistant_text)
        parse_recovery = ""
        recovered_original = ""
        if parse_status != "VALID_AGENT_ENVELOPE":
            # P0 窄信封救援（用户裁决 2026-08-14）：双 JSON 拼接 → 首
            # tool_request 信封按正常循环处理；assistant_text 规范化为
            # 首信封（原始双 JSON 不得写回对话）。
            rescued_text, recovery = rescue_trailing_envelope(assistant_text)
            if rescued_text is None:
                rescued_text, recovery = rescue_prose_wrapped_envelope(
                    assistant_text
                )
            if rescued_text is not None:
                rescued_envelope, rescued_status = parse_agent_envelope(
                    rescued_text
                )
                if rescued_status == "VALID_AGENT_ENVELOPE":
                    # 原始输出保留在 raw_response 里；对话里换成规范化信封，
                    # 并记录恢复类型，避免"恢复"在读数里变成隐形。
                    original_text = assistant_text
                    assistant_text = rescued_text
                    envelope = rescued_envelope
                    parse_status = rescued_status
                    parse_recovery = recovery
                    recovered_original = original_text
        try:
            raw_response = completion.model_dump(mode="json")
        except (AttributeError, TypeError):
            raw_response = {
                "id": getattr(completion, "id", ""),
                "model": getattr(completion, "model", ""),
            }
        usage = getattr(completion, "usage", None)
        provider_metadata = {
            "recovered_original_text": recovered_original[:500],
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
            parse_recovery=parse_recovery,
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
                raise ReplayTapeMiss("NO_TAPE_ENTRY: ordered replay response exhausted")
            response = self._ordered[self._call_count]
        else:
            assert self._mapped is not None
            semantic_hash = request.semantic_request_hash()
            if semantic_hash not in self._mapped:
                raise ReplayTapeMiss(
                    f"NO_TAPE_ENTRY: semantic request {semantic_hash}"
                )
            response = self._mapped[semantic_hash]
        self._call_count += 1
        return response

    def clone(self) -> "ReplayAgentBackend":
        source: Sequence[AgentResponse] | Mapping[str, AgentResponse]
        source = self._ordered if self._ordered is not None else self._mapped or {}
        return ReplayAgentBackend(source)


__all__ = [
    "AgentCallBudgetExceeded",
    "AgentBackend",
    "AgentRequest",
    "AgentResponse",
    "AgentTransportError",
    "AgictoChatCompletionsBackend",
    "BudgetedAgentBackend",
    "DEFAULT_AGENT_BASE_URL",
    "DEFAULT_AGENT_MODEL",
    "OPENAI_SDK_VERSION",
    "ReplayAgentBackend",
    "ReplayTapeMiss",
    "parse_agent_envelope",
    "rescue_trailing_envelope",
]
