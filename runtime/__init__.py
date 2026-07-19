"""Canonical program execution runtime."""

from .candidate_pool import (
    CandidatePool,
    ProtocolChoiceError,
    effect_equivalent_to_identity,
    execute_selected,
)
from .agent_backend import (
    AgentBackend,
    AgentCallBudgetExceeded,
    AgentRequest,
    AgentResponse,
    AgentTransportError,
    AgictoChatCompletionsBackend,
    BudgetedAgentBackend,
    ReplayAgentBackend,
    ReplayTapeMiss,
)
from .decision_trace import BehaviorSignature, DecisionTrace
from .executor import ExecutionResult, run_pipeline
from .llm_cache import CacheKey, CacheReceipt, CachedAgentBackend, EffectiveRequestCache

__all__ = [
    "BehaviorSignature",
    "AgentBackend",
    "AgentCallBudgetExceeded",
    "AgentRequest",
    "AgentResponse",
    "AgentTransportError",
    "AgictoChatCompletionsBackend",
    "BudgetedAgentBackend",
    "CacheKey",
    "CacheReceipt",
    "CandidatePool",
    "CachedAgentBackend",
    "DecisionTrace",
    "ExecutionResult",
    "EffectiveRequestCache",
    "ProtocolChoiceError",
    "ReplayAgentBackend",
    "ReplayTapeMiss",
    "effect_equivalent_to_identity",
    "execute_selected",
    "run_pipeline",
]
