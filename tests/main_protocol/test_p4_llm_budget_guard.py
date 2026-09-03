"""Focused checks for the P4 pre-call LLM budget instrument."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared
from evaluation.main_protocol_p4 import run_forecast_p4_performance as p4


class _Backend:
    def __init__(self, reached: list[tuple[str, int]]) -> None:
        self.calls = 0
        self._reached = reached

    def complete(self, request):
        self.calls += 1
        self._reached.append((request.stage, request.call_index))
        return SimpleNamespace(provider_metadata={})


def _request(
    stage: str,
    call_index: int,
    *,
    followup_schema: str | None = None,
):
    messages = [{"role": "user", "content": "initial"}]
    if followup_schema is not None:
        messages.append(
            {
                "role": "user",
                "content": json.dumps({"schema_version": followup_schema}),
            }
        )
    return SimpleNamespace(
        role="fast",
        stage=stage,
        call_index=call_index,
        messages=tuple(messages),
    )


def test_ninth_cell_call_is_blocked_before_backend_reach():
    reached: list[tuple[str, int]] = []
    ledger = shared._CountingBackend(
        lambda: _Backend(reached), 20, share_inner=True
    )
    arm = ledger.new_arm_backend(
        scope_id="Forward/E1/A5-online",
        maximum_calls=p4.MAX_LLM_CALLS,
    )
    requests = (
        _request("inspect", 0),
        _request("inspect", 1, followup_schema="tool-result/1"),
        _request("propose", 0),
        _request("propose", 1, followup_schema="stage-validation-error/2"),
        _request("select", 0),
        _request("select", 1, followup_schema="stage-validation-error/2"),
        _request("revise", 0),
        _request("revise", 1, followup_schema="stage-validation-error/2"),
    )
    for request in requests:
        arm.complete(request)

    with pytest.raises(shared.Stop) as blocked:
        arm.complete(_request("finalize", 0))

    assert blocked.value.verdict == "LLM_CELL_BUDGET_EXHAUSTED"
    assert reached == [
        ("inspect", 0),
        ("inspect", 1),
        ("propose", 0),
        ("propose", 1),
        ("select", 0),
        ("select", 1),
        ("revise", 0),
        ("revise", 1),
    ]
    state = ledger.budget_state()
    assert state["global_calls"] == 8
    assert state["scope_calls"] == {"Forward/E1/A5-online": 8}
    assert [row["purpose"] for row in state["call_records"]] == [
        "stage_initial",
        "tool_followup",
        "stage_initial",
        "validation_retry",
        "stage_initial",
        "validation_retry",
        "stage_initial",
        "validation_retry",
    ]
    assert state["blocked_records"][-1] == {
        "scope_id": "Forward/E1/A5-online",
        "scope_call": 9,
        "role": "fast",
        "stage": "finalize",
        "stage_call_index": 0,
        "purpose": "stage_initial",
        "reached_backend": False,
        "budget_charged": False,
    }


def test_resume_preserves_spent_count_and_other_arm_is_isolated():
    first_reached: list[tuple[str, int]] = []
    first = shared._CountingBackend(
        lambda: _Backend(first_reached), 20, share_inner=True
    )
    a5 = first.new_arm_backend(
        scope_id="Forward/E1/A5-online", maximum_calls=p4.MAX_LLM_CALLS
    )
    for call_index in range(p4.MAX_LLM_CALLS):
        a5.complete(_request("inspect", call_index))
    persisted = first.budget_state()

    resumed_reached: list[tuple[str, int]] = []
    resumed = shared._CountingBackend(
        lambda: _Backend(resumed_reached),
        20,
        share_inner=True,
        resume_state=persisted,
    )
    resumed_a5 = resumed.new_arm_backend(
        scope_id="Forward/E1/A5-online", maximum_calls=p4.MAX_LLM_CALLS
    )
    with pytest.raises(shared.Stop) as blocked:
        resumed_a5.complete(_request("select", 0))
    assert blocked.value.verdict == "LLM_CELL_BUDGET_EXHAUSTED"
    assert resumed_reached == []
    assert resumed.budget_state()["scope_calls"]["Forward/E1/A5-online"] == 8

    a3 = resumed.new_arm_backend(
        scope_id="Forward/E1/A3-reset", maximum_calls=p4.MAX_LLM_CALLS
    )
    a3.complete(_request("inspect", 0))
    state = resumed.budget_state()
    assert resumed_reached == [("inspect", 0)]
    assert state["global_calls"] == 9
    assert state["scope_calls"] == {
        "Forward/E1/A5-online": 8,
        "Forward/E1/A3-reset": 1,
    }


def test_spend_is_persisted_before_delegate_and_not_refunded_on_error():
    persisted: list[dict[str, object]] = []

    class _FailingBackend:
        calls = 0

        def complete(self, _request):
            self.calls += 1
            assert persisted[-1]["global_calls"] == 1
            assert persisted[-1]["scope_calls"] == {"Forward/E1/K0-fixed": 1}
            raise RuntimeError("transport witness")

    ledger = shared._CountingBackend(
        _FailingBackend,
        20,
        share_inner=True,
        on_budget_change=lambda state: persisted.append(state),
    )
    k0 = ledger.new_arm_backend(
        scope_id="Forward/E1/K0-fixed", maximum_calls=p4.MAX_LLM_CALLS
    )
    with pytest.raises(RuntimeError, match="transport witness"):
        k0.complete(_request("inspect", 0))

    assert ledger.budget_state()["scope_calls"] == {
        "Forward/E1/K0-fixed": 1
    }
    resumed = shared._CountingBackend(
        lambda: _Backend([]),
        20,
        share_inner=True,
        resume_state=ledger.budget_state(),
    )
    assert resumed.budget_state()["scope_calls"] == {
        "Forward/E1/K0-fixed": 1
    }


def test_p4_binds_every_adaptive_cell_to_the_frozen_eight_call_cap():
    payload = p4._initial_payload(
        release={}, replicas=("Forward",), backend_mode="scripted"
    )
    instrument = payload["llm_budget_instrument"]
    assert instrument["frozen_per_cell_cap"] == 8
    assert instrument["resume_count_preserved"] is True
    assert instrument["arm_counts_isolated"] is True
    assert p4.budget_contract(1)["per_method_cell"]["llm_call_max"] == 8
    assert p4.budget_contract(3)["global_llm_call_cap"] == 576
