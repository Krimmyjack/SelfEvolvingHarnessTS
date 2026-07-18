from dataclasses import replace

import numpy as np
import pytest

from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.runtime.candidate_pool import (
    CandidatePool,
    ProtocolChoiceError,
    effect_equivalent_to_identity as pool_effect,
    execute_selected,
)
from SelfEvolvingHarnessTS.runtime.decision_trace import BehaviorSignature, DecisionTrace


def _program_candidate(candidate_id="agent-0"):
    program = Program.from_steps([("impute_linear", {})], source="agent")
    return Candidate.program_candidate(candidate_id, program, source="agent")


def test_identity_is_injected_and_never_filtered():
    pool = CandidatePool.build([_program_candidate()], total_k=3)
    kept = pool.apply_risk(lambda candidate: False)
    assert kept.ids[0] == "identity"
    assert "identity" in kept.ids


def test_missing_choice_is_protocol_failure():
    pool = CandidatePool.build([_program_candidate()], total_k=3)
    with pytest.raises(ProtocolChoiceError, match="chosen_candidate_id"):
        pool.require_choice("")


def test_effect_equivalence_uses_shape_dtype_and_bytes():
    raw = np.asarray([1.0, 2.0], dtype=np.float64)
    assert pool_effect(raw, raw.copy()) is True
    assert pool_effect(raw, raw.astype(np.float32)) is False


def test_programs_are_deduplicated_by_semantics_and_budgeted_after_identity():
    first = _program_candidate("agent-0")
    duplicate = _program_candidate("agent-1")
    distinct = Candidate.program_candidate(
        "agent-2",
        Program.from_steps([("impute_ema", {})], source="agent"),
        source="agent",
    )
    pool = CandidatePool.build([first, duplicate, distinct], total_k=3)
    assert pool.ids == ("identity", "agent-0", "agent-2")


def test_identity_only_budget_does_not_consume_supplier_programs():
    pool = CandidatePool.build([_program_candidate()], total_k=1)
    assert pool.ids == ("identity",)


def test_identity_execution_returns_read_only_float64_copy():
    raw = np.asarray([1, 2, 3], dtype=np.int32)
    prepared, program = execute_selected(Candidate.identity(), raw)
    assert prepared.dtype == np.float64
    assert prepared.flags.writeable is False
    assert program is None
    raw[0] = 99
    assert prepared[0] == 1.0


def _trace() -> DecisionTrace:
    return DecisionTrace(
        case_id="case-a",
        public_observation_ids=("obs-request-a",),
        inspected_regions=((10, 20),),
        tool_calls=({"tool_name": "summarize_series", "latency_ms": 3},),
        retrieved_skill_ids=("inspect_and_localize",),
        retrieved_memory_ids=(),
        applicability_matches=("inspect_and_localize",),
        candidate_ids=("identity", "agent-0"),
        candidate_program_shas=(None, "a" * 16),
        chosen_candidate_id="agent-0",
        compilation_status="ok",
        execution_status="ok",
        modified_indices=(12, 13),
        verification_actions=("scope_checked",),
        effect_equivalent_to_identity=False,
        series_length=100,
    )


def test_behavior_signature_excludes_case_and_request_provenance():
    first = _trace()
    second = replace(
        first,
        case_id="case-b",
        public_observation_ids=("different-provider-request",),
        tool_calls=({"tool_name": "summarize_series", "latency_ms": 999},),
    )
    assert BehaviorSignature.from_trace(first).behavior_signature_sha == (
        BehaviorSignature.from_trace(second).behavior_signature_sha
    )
