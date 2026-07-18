"""Stable contracts shared by methods, runtime, and evaluation."""

from .candidate import Candidate, CandidateKind
from .canonical import (
    CANONICALIZATION_VERSION,
    canonical_json_bytes,
    canonical_json_document_bytes,
    canonical_jsonl_bytes,
    canonical_sha256,
    canonical_text_bytes,
    parse_json_document,
)
from .harness import (
    EditManifest,
    EditOperation,
    HarnessSnapshot,
    MemoryEntry,
    SkillEntry,
    SkillKind,
    load_learned_skill_entry,
    load_memory_entry,
    load_skill_entry,
)
from .method import (
    ExecutionReceipt,
    Method,
    PreparationRequest,
    PreparationResult,
    PreparationStatus,
    PreparedSeries,
)
from .program import Program, ProgramStep
from .observables import OBSERVABLE_FEATURES, validate_applicability
from .task import (
    LABEL_AVAILABILITY,
    TARGET_SEMANTICS_BY_TASK,
    TASK_TYPES,
    MetricSpec,
    TaskSpec,
    anomaly_task_spec_v1,
    classification_task_spec_v1,
    forecast_task_spec_v1,
)

__all__ = [
    "Candidate",
    "CandidateKind",
    "CANONICALIZATION_VERSION",
    "EditManifest",
    "EditOperation",
    "ExecutionReceipt",
    "LABEL_AVAILABILITY",
    "Method",
    "HarnessSnapshot",
    "MemoryEntry",
    "OBSERVABLE_FEATURES",
    "PreparationRequest",
    "PreparationResult",
    "PreparationStatus",
    "PreparedSeries",
    "Program",
    "ProgramStep",
    "SkillEntry",
    "SkillKind",
    "TARGET_SEMANTICS_BY_TASK",
    "TASK_TYPES",
    "MetricSpec",
    "TaskSpec",
    "anomaly_task_spec_v1",
    "classification_task_spec_v1",
    "canonical_json_bytes",
    "canonical_json_document_bytes",
    "canonical_jsonl_bytes",
    "canonical_sha256",
    "canonical_text_bytes",
    "forecast_task_spec_v1",
    "load_learned_skill_entry",
    "load_memory_entry",
    "load_skill_entry",
    "parse_json_document",
    "validate_applicability",
]
