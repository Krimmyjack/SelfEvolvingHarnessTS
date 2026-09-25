"""LLM transport shared by both controllers (adapted from _scratch/ts_aug_source_grounding_pilot/backend.py:
same account/transport fault classification, stop-on-account-fault, one bounded retry on transient
transport faults, returned-model identity check, no fabricated token usage). Differences: explicit
configuration object instead of module constants; accounting goes through budget.Ledger; `messages`
may carry a multi-turn history (the W line's tool loop); a DryClient stands in for tests.
Credentials come only from CPA_API_KEY / OPENAI_API_KEY and are never written anywhere.
"""
from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass

from . import budget as _budget

_ACCOUNT_MARKERS = ("insufficient balance", "insufficient_quota", "insufficient quota", "payment required", "error code: 402",
                    "error code: 401", "invalid_api_key", "not enough available money", "quota exceeded", "billing")
_TRANSPORT_MARKERS = ("bad_response_status_code", "openai_error", "upstream", "bad gateway", "service unavailable",
                      "error code: 502", "error code: 503", "error code: 504", "timeout", "connection")


class AccountFault(RuntimeError):
    """The request never reached a model, or the model identity is wrong. Stops the line."""


class TransportFault(RuntimeError):
    pass


def classify_fault(detail: object) -> str:
    status = getattr(detail, "status_code", None)
    name = type(detail).__name__
    lowered = str(detail).lower()
    if status in (401, 402, 403) or name in ("AuthenticationError", "PermissionDeniedError"):
        return "ACCOUNT_OR_PERMISSION_FAULT"
    if any(m in lowered for m in _ACCOUNT_MARKERS):
        return "ACCOUNT_OR_PERMISSION_FAULT"
    if status in (500, 502, 503, 504) or name in ("InternalServerError", "APITimeoutError", "APIConnectionError"):
        return "TRANSPORT_TRANSIENT_FAULT"
    if any(m in lowered for m in _TRANSPORT_MARKERS):
        return "TRANSPORT_TRANSIENT_FAULT"
    return "OTHER_FAULT"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://127.0.0.1:8318/v1"
    requested_model: str = "cpa-grok-4.6"
    returned_model_required: str = "grok-4.6-build"
    temperature: float = 0.0
    per_request_timeout_s: float = 300.0
    transport_attempts: int = 2          # 1 send + at most 1 retry on a transient transport fault


class LiveClient:
    def __init__(self, cfg: LLMConfig, ledger: _budget.Ledger):
        key = next((os.environ.get(n, "").strip() for n in ("CPA_API_KEY", "OPENAI_API_KEY") if os.environ.get(n, "").strip()), "")
        if not key:
            raise AccountFault("no API key in CPA_API_KEY/OPENAI_API_KEY env vars")
        openai = importlib.import_module("openai")
        self.cfg = cfg
        self.client = openai.OpenAI(api_key=key, base_url=cfg.base_url, timeout=cfg.per_request_timeout_s, max_retries=0)
        self.ledger = ledger
        self.account_fault: str | None = None

    def complete(self, *, role: str, unit: str, messages: list, max_output_tokens: int) -> dict:
        """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]. One logical request."""
        if self.account_fault:
            raise AccountFault(self.account_fault)
        self.ledger.check_llm()
        last_exc, attempts = None, min(2, self.cfg.transport_attempts)
        for attempt in range(attempts):
            try:
                completion = self.client.chat.completions.create(model=self.cfg.requested_model, messages=messages, temperature=self.cfg.temperature,
                                                                 max_tokens=int(max_output_tokens), timeout=self.cfg.per_request_timeout_s)
                break
            except Exception as exc:  # noqa: BLE001
                detail = "%s: %s" % (type(exc).__name__, exc)
                kind = classify_fault(exc)
                if kind == "ACCOUNT_OR_PERMISSION_FAULT":
                    self.account_fault = detail
                    self.ledger.charge_llm(role, unit, attempt + 1, None, None, False, detail)
                    raise AccountFault(detail) from exc
                last_exc = exc
                if kind == "TRANSPORT_TRANSIENT_FAULT" and attempt + 1 < attempts:
                    time.sleep(2.0)
                    continue
                self.ledger.charge_llm(role, unit, attempt + 1, None, None, False, detail)
                raise TransportFault(detail) from exc
        else:
            self.ledger.charge_llm(role, unit, attempts, None, None, False, str(last_exc))
            raise TransportFault(str(last_exc))
        returned_model = str(getattr(completion, "model", "") or "")
        usage = getattr(completion, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else None
        ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else None
        choices = getattr(completion, "choices", ())
        choice = choices[0] if choices else None
        text = (getattr(getattr(choice, "message", None), "content", "") or "") if choice else ""
        finish = getattr(choice, "finish_reason", "") if choice else ""
        ok = returned_model == self.cfg.returned_model_required
        self.ledger.charge_llm(role, unit, attempt + 1, pt, ct, ok, "" if ok else "returned model %r" % returned_model)
        if not ok:
            self.account_fault = "unexpected returned model %r; required %r" % (returned_model, self.cfg.returned_model_required)
            raise AccountFault(self.account_fault)
        return {"text": text, "returned_model": returned_model, "finish_reason": finish, "usage": {"prompt_tokens": pt, "completion_tokens": ct},
                "http_attempts": attempt + 1}


class DryClient:
    """Deterministic stand-in for tests: returns queued responses in order, charges the ledger like the live client."""

    def __init__(self, ledger: _budget.Ledger, responses: list):
        self.ledger, self.responses, self.calls = ledger, list(responses), []

    def complete(self, *, role: str, unit: str, messages: list, max_output_tokens: int) -> dict:
        self.ledger.check_llm()
        if not self.responses:
            raise TransportFault("dry client has no more responses")
        text = self.responses.pop(0)
        self.calls.append({"role": role, "unit": unit, "messages": messages})
        self.ledger.charge_llm(role, unit, 1, sum(len(m["content"]) // 4 for m in messages), len(text) // 4, True, "dry")
        return {"text": text, "returned_model": "dry", "finish_reason": "stop", "usage": {"prompt_tokens": None, "completion_tokens": None}, "http_attempts": 1}
