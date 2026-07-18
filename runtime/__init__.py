"""Canonical program execution runtime."""

from .candidate_pool import (
    CandidatePool,
    ProtocolChoiceError,
    effect_equivalent_to_identity,
    execute_selected,
)
from .decision_trace import BehaviorSignature, DecisionTrace
from .executor import ExecutionResult, run_pipeline

__all__ = [
    "BehaviorSignature",
    "CandidatePool",
    "DecisionTrace",
    "ExecutionResult",
    "ProtocolChoiceError",
    "effect_equivalent_to_identity",
    "execute_selected",
    "run_pipeline",
]
