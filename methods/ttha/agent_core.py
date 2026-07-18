from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
)
from SelfEvolvingHarnessTS.contracts.public_boundary import assert_public_payload
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_AGENT_MODEL,
    AgentBackend,
    AgentRequest,
    AgentResponse,
)
from SelfEvolvingHarnessTS.runtime.errors import ProtocolViolation

from .public_tools import PublicToolGateway, PublicToolReceipt
from .retrieval import EffectiveHarnessView
from .schema_contracts import (
    LocalSchemaError,
    load_schema,
    load_stage_schema,
    validate_local_schema as _validate_local_schema,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
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


class AgentRole(str, Enum):
    FAST = "fast"
    SLOW = "slow"


class AgentProtocolError(ProtocolViolation):
    """The model returned an invalid local envelope or stage payload."""


def validate_local_schema(
    value: object,
    schema: Mapping[str, object],
    *,
    path: str = "payload",
) -> None:
    try:
        _validate_local_schema(value, schema, path=path)
    except LocalSchemaError as exc:
        raise AgentProtocolError(str(exc)) from exc


@dataclass(frozen=True)
class PublicAgentInput:
    case_id: str
    public_data: Mapping[str, object]
    public_case_view_sha: str

    @classmethod
    def create(cls, case_id: str, public_data: Mapping[str, object]) -> "PublicAgentInput":
        assert_public_payload(public_data)
        payload = _plain(public_data)
        return cls(case_id, _freeze_json(payload), canonical_sha256(payload))


@dataclass(frozen=True)
class AgentStageResult:
    role: AgentRole
    stage: str
    payload: Mapping[str, object]
    response: AgentResponse
    tool_receipts: tuple[PublicToolReceipt, ...]
    request_hashes: tuple[str, ...]
    no_proposal_reason: str | None = None


def _skill_prompt(skill: object) -> dict[str, object]:
    return {
        "skill_id": skill.skill_id,
        "skill_kind": skill.skill_kind.value,
        "body": skill.body,
        "allowed_tools": list(skill.allowed_tools),
        "risk_guards": _plain(skill.risk_guards),
    }


def _memory_prompt(memory: object) -> dict[str, object]:
    return {"memory_id": memory.memory_id, "body": memory.body, "risk_guards": _plain(memory.risk_guards)}


class TTHAAgentCore:
    def __init__(
        self,
        backend: AgentBackend,
        tools: PublicToolGateway,
        *,
        model: str = DEFAULT_AGENT_MODEL,
        base_url: str = DEFAULT_AGENT_BASE_URL,
    ):
        self.backend = backend
        self.tools = tools
        self.model = model
        self.base_url = base_url

    @staticmethod
    def load_stage_schema(name: str) -> Mapping[str, object]:
        return _freeze_json(load_stage_schema(name))

    def _messages(
        self,
        *,
        role: AgentRole,
        stage: str,
        public_input: Mapping[str, object],
        harness_view: EffectiveHarnessView,
        output_schema_name: str,
        output_schema: Mapping[str, object],
        tool_schemas: tuple[Mapping[str, object], ...],
    ) -> tuple[Mapping[str, object], ...]:
        resolved_harness = {
            "instruction": harness_view.instruction,
            "skills": [_skill_prompt(skill) for skill in harness_view.skills],
            "memories": [_memory_prompt(memory) for memory in harness_view.memories],
            "controls": _plain(harness_view.controls),
        }
        system = (
            harness_view.instruction
            + "\nRuntime rule: return exactly one agent-envelope/1 JSON value. "
            + "The outer envelope is mandatory; never return the stage payload by "
            + "itself and never use schema_name/content as substitute wrapper keys. "
            + "Do not emit PASS/FAIL judgments or hidden reasoning.\nResolved Harness: "
            + canonical_json_bytes(resolved_harness).decode("utf-8")
        )
        response_contract = {
            "outer_envelope_required": True,
            "bare_stage_payload_forbidden": True,
            "stage_result_template": (
                '{"schema_version":"agent-envelope/1","kind":"stage_result",'
                f'"stage":"{stage}","payload":<OBJECT SATISFYING '
                f'{output_schema_name}>}}'
            ),
            "tool_request_template": {
                "schema_version": "agent-envelope/1",
                "kind": "tool_request",
                "call_id": "unique_call_id",
                "tool_name": "one_allowed_local_tool_name",
                "arguments": {},
            },
        }
        user_payload = {
            "schema_version": "public-agent-input/1",
            "role": role.value,
            "stage": stage,
            "public_input": _plain(public_input),
            "allowed_local_tools": _plain(tool_schemas),
            "response_contract": response_contract,
            "stage_payload_schema_name": output_schema_name,
            "stage_payload_schema": _plain(output_schema),
        }
        return (
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": canonical_json_bytes(user_payload).decode("utf-8"),
            },
        )

    def run_stage(
        self,
        *,
        role: AgentRole | str,
        stage: str,
        case_id: str,
        public_input: Mapping[str, object],
        harness_view: EffectiveHarnessView,
        output_schema_name: str,
        output_schema: Mapping[str, object],
        source_snapshot_sha: str,
    ) -> AgentStageResult:
        role = AgentRole(role)
        validate_local_schema({}, {"type": "object"}, path="internal")
        tool_schemas = self.tools.schemas_for(role=role, stage=stage)
        tool_context_sha = (
            self.tools.context_sha
            if tool_schemas
            else canonical_sha256(
                {"schema_version": "empty-tool-context/1", "role": role.value, "stage": stage}
            )
        )
        declared_tools = {
            schema["name"]
            for schema in tool_schemas
            if isinstance(schema, Mapping) and isinstance(schema.get("name"), str)
        }
        public_agent_input = PublicAgentInput.create(case_id, public_input)
        messages = self._messages(
            role=role,
            stage=stage,
            public_input=public_agent_input.public_data,
            harness_view=harness_view,
            output_schema_name=output_schema_name,
            output_schema=output_schema,
            tool_schemas=tool_schemas,
        )
        envelope_schema = load_schema("agent_envelope_v1.json")
        tool_result_schema = load_schema("tool_result_v1.json")
        request_hashes: list[str] = []
        receipts: list[PublicToolReceipt] = []
        call_ids: set[str] = set()
        tool_rounds = 0
        call_index = 0
        while True:
            request = AgentRequest.for_stage(
                case_id=case_id,
                role=role.value,
                stage=stage,
                call_index=call_index,
                replicate_id="r0",
                messages=messages,
                envelope_schema_sha=canonical_sha256(envelope_schema),
                tool_schema_sha=canonical_sha256(_plain(tool_schemas)),
                tool_result_schema_sha=canonical_sha256(tool_result_schema),
                stage_schema_sha=canonical_sha256(_plain(output_schema)),
                public_case_view_sha=public_agent_input.public_case_view_sha,
                effective_harness_view_sha=harness_view.effective_harness_view_sha,
                tool_context_sha=tool_context_sha,
                source_harness_snapshot_sha=source_snapshot_sha,
                model=self.model,
                base_url=self.base_url,
            )
            request_hashes.append(request.semantic_request_hash())
            response = self.backend.complete(request)
            if response.parse_status != "VALID_AGENT_ENVELOPE" or response.parsed_envelope is None:
                raise AgentProtocolError("invalid agent-envelope/1 response")
            envelope = response.parsed_envelope
            if envelope["kind"] == "stage_result":
                if envelope["stage"] != stage:
                    raise AgentProtocolError("stage_result names the wrong stage")
                payload = envelope["payload"]
                validate_local_schema(payload, output_schema)
                return AgentStageResult(
                    role=role,
                    stage=stage,
                    payload=_freeze_json(payload),
                    response=response,
                    tool_receipts=tuple(receipts),
                    request_hashes=tuple(request_hashes),
                )
            if envelope["kind"] == "no_proposal":
                if role is not AgentRole.SLOW or stage != "edit":
                    raise AgentProtocolError(
                        "no_proposal is valid for the slow edit stage only"
                    )
                return AgentStageResult(
                    role=role,
                    stage=stage,
                    payload=MappingProxyType({}),
                    response=response,
                    tool_receipts=tuple(receipts),
                    request_hashes=tuple(request_hashes),
                    no_proposal_reason=str(envelope["reason_code"]),
                )
            if tool_rounds >= 8:
                raise AgentProtocolError("tool round limit exceeded")
            call_id = envelope["call_id"]
            tool_name = envelope["tool_name"]
            if call_id in call_ids:
                raise AgentProtocolError("duplicate tool call_id")
            if tool_name not in declared_tools:
                raise AgentProtocolError("undeclared tool requested")
            call_ids.add(call_id)
            receipt = self.tools.call(tool_name, envelope["arguments"])
            if receipt.tool_name != tool_name or receipt.context_sha != tool_context_sha:
                raise AgentProtocolError("public tool receipt identity mismatch")
            receipts.append(receipt)
            tool_result = {
                "schema_version": "tool-result/1",
                "call_id": call_id,
                "tool_name": tool_name,
                "ok": receipt.ok,
                "public_result": _plain(receipt.public_result),
                "receipt_sha": receipt.receipt_sha,
            }
            validate_local_schema(tool_result, tool_result_schema, path="tool_result")
            messages = (
                *messages,
                {"role": "assistant", "content": response.assistant_text},
                {
                    "role": "user",
                    "content": canonical_json_bytes(tool_result).decode("utf-8"),
                },
            )
            tool_rounds += 1
            call_index += 1


__all__ = [
    "AgentProtocolError",
    "AgentRole",
    "AgentStageResult",
    "PublicAgentInput",
    "TTHAAgentCore",
    "validate_local_schema",
]
