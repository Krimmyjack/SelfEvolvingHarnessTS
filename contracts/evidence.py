from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from .canonical import canonical_sha256
from .program import Program


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PATTERN_FORBIDDEN_FIELDS = frozenset(
    {"action", "program", "response", "utility", "outcome", "policy_verdict"}
)


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(nested) for key, nested in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(nested) for nested in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def _string_tuple(value: object, *, field: str, sha: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    result = tuple(
        _sha256(item, field=f"{field} entry") if sha else _nonempty(item, field=f"{field} entry")
        for item in value
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{field} entries must be unique")
    return result


@dataclass(frozen=True)
class ObservationScope:
    """A deployment-visible half-open interval bound to one immutable dataset."""

    dataset_sha: str
    series_id: str
    channel: str
    start: int
    end: int
    visibility: str

    def __post_init__(self) -> None:
        _sha256(self.dataset_sha, field="dataset_sha")
        _nonempty(self.series_id, field="series_id")
        _nonempty(self.channel, field="channel")
        _nonempty(self.visibility, field="visibility")
        if isinstance(self.start, bool) or not isinstance(self.start, int) or self.start < 0:
            raise ValueError("start must be a non-negative integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int) or self.end <= self.start:
            raise ValueError("end must be an integer greater than start")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "observation-scope/1",
            "dataset_sha": self.dataset_sha,
            "series_id": self.series_id,
            "channel": self.channel,
            "interval": {"start": self.start, "end": self.end, "closed": "left"},
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class ScopedObservationReceipt:
    """E0 scoped receipt kept parallel to the frozen M0 public-tool receipt."""

    tool_name: str
    arguments: Mapping[str, object]
    public_result: Mapping[str, object]
    context_sha: str
    scope: ObservationScope
    coverage: Mapping[str, object]
    reliability: Mapping[str, object]
    tool_version: str
    receipt_sha: str
    ok: bool = True

    @classmethod
    def create(
        cls,
        *,
        tool_name: str,
        arguments: Mapping[str, object],
        public_result: Mapping[str, object],
        context_sha: str,
        scope: ObservationScope,
        coverage: Mapping[str, object],
        reliability: Mapping[str, object],
        tool_version: str,
        ok: bool = True,
    ) -> "ScopedObservationReceipt":
        _nonempty(tool_name, field="tool_name")
        _nonempty(context_sha, field="context_sha")
        _nonempty(tool_version, field="tool_version")
        if not isinstance(scope, ObservationScope):
            raise TypeError("scope must be ObservationScope")
        for field, value in (
            ("arguments", arguments),
            ("public_result", public_result),
            ("coverage", coverage),
            ("reliability", reliability),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{field} must be an object")
        if not coverage or not reliability:
            raise ValueError("scoped receipt requires explicit coverage and reliability")
        payload = {
            "schema_version": "scoped-observation-receipt/1",
            "tool_name": tool_name,
            "arguments": _plain(arguments),
            "public_result": _plain(public_result),
            "context_sha": context_sha,
            "scope": scope.as_dict(),
            "coverage": _plain(coverage),
            "reliability": _plain(reliability),
            "tool_version": tool_version,
            "ok": ok,
        }
        return cls(
            tool_name=tool_name,
            arguments=_freeze(arguments),
            public_result=_freeze(public_result),
            context_sha=context_sha,
            scope=scope,
            coverage=_freeze(coverage),
            reliability=_freeze(reliability),
            tool_version=tool_version,
            receipt_sha=canonical_sha256(payload),
            ok=ok,
        )


@dataclass(frozen=True)
class PatternNode:
    """Observed state only; actions, responses, utility, and policy verdicts are forbidden."""

    pattern_id: str
    scope: ObservationScope
    observations: Mapping[str, object]
    receipt_shas: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.pattern_id, field="pattern_id")
        if not isinstance(self.scope, ObservationScope):
            raise TypeError("scope must be ObservationScope")
        if not isinstance(self.observations, Mapping) or not self.observations:
            raise ValueError("observations must be a non-empty object")
        forbidden = _PATTERN_FORBIDDEN_FIELDS.intersection(self.observations)
        if forbidden:
            raise ValueError(f"PatternNode cannot contain action/response fields: {sorted(forbidden)}")
        receipts = _string_tuple(self.receipt_shas, field="receipt_shas", sha=True)
        if not receipts:
            raise ValueError("PatternNode requires at least one receipt")
        object.__setattr__(self, "receipt_shas", receipts)
        object.__setattr__(self, "observations", _freeze(self.observations))


@dataclass(frozen=True)
class ScopedProgramBinding:
    """Binds an existing concrete Program to exactly one observation interval."""

    binding_id: str
    program: Program
    scope: ObservationScope
    bindings: Mapping[str, object]

    def __post_init__(self) -> None:
        _nonempty(self.binding_id, field="binding_id")
        if not isinstance(self.program, Program):
            raise TypeError("program must be Program")
        if not isinstance(self.scope, ObservationScope):
            raise TypeError("scope must be ObservationScope")
        if not isinstance(self.bindings, Mapping):
            raise TypeError("bindings must be an object")
        object.__setattr__(self, "bindings", _freeze(self.bindings))


@dataclass(frozen=True)
class CapabilityTemplate:
    capability_id: str
    applicability: Mapping[str, object]
    program_schema: Mapping[str, object]
    binding_schema: Mapping[str, object]
    risk_guards: Mapping[str, object]
    task_context: Mapping[str, object]
    consumer_context: Mapping[str, object]
    operator_registry_context: Mapping[str, object]

    def __post_init__(self) -> None:
        _nonempty(self.capability_id, field="capability_id")
        for field in (
            "applicability",
            "program_schema",
            "binding_schema",
            "risk_guards",
            "task_context",
            "consumer_context",
            "operator_registry_context",
        ):
            value = getattr(self, field)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field} must be an object")
            object.__setattr__(self, field, _freeze(value))


@dataclass(frozen=True)
class LocalBehaviorEvidence:
    """Evidence that a concrete program changed behavior; never a promotion proof alone."""

    evidence_id: str
    capability_id: str
    pattern_node_id: str
    program_binding_id: str
    behavior: Mapping[str, object]
    receipt_shas: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    evidence_kind = "local_behavior"

    def __post_init__(self) -> None:
        for field in ("evidence_id", "capability_id", "pattern_node_id", "program_binding_id"):
            _nonempty(getattr(self, field), field=field)
        if not isinstance(self.behavior, Mapping) or not self.behavior:
            raise ValueError("behavior must be a non-empty object")
        object.__setattr__(self, "behavior", _freeze(self.behavior))
        object.__setattr__(
            self, "receipt_shas", _string_tuple(self.receipt_shas, field="receipt_shas", sha=True)
        )
        object.__setattr__(self, "tags", _string_tuple(self.tags, field="tags"))

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": self.evidence_kind,
            "evidence_id": self.evidence_id,
            "capability_id": self.capability_id,
            "pattern_node_id": self.pattern_node_id,
            "program_binding_id": self.program_binding_id,
            "behavior": _plain(self.behavior),
            "receipt_shas": list(self.receipt_shas),
            "tags": list(self.tags),
        }

    @property
    def evidence_sha(self) -> str:
        return canonical_sha256(self.identity_payload())


@dataclass(frozen=True)
class PolicyInterventionEvidence:
    """Outcome evidence comparing an intervention policy with an explicit baseline."""

    evidence_id: str
    capability_id: str
    baseline_policy: str
    intervention_policy: str
    cohort_definition: Mapping[str, object]
    outcomes: Mapping[str, object]
    risk: Mapping[str, object]
    causal_scope: str
    source_refs: tuple[str, ...]
    receipt_shas: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    evidence_kind = "policy_intervention"

    def __post_init__(self) -> None:
        for field in (
            "evidence_id",
            "capability_id",
            "baseline_policy",
            "intervention_policy",
            "causal_scope",
        ):
            _nonempty(getattr(self, field), field=field)
        for field in ("cohort_definition", "outcomes", "risk"):
            value = getattr(self, field)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field} must be an object")
            object.__setattr__(self, field, _freeze(value))
        refs = _string_tuple(self.source_refs, field="source_refs")
        if not refs:
            raise ValueError("PolicyInterventionEvidence requires a source reference")
        object.__setattr__(self, "source_refs", refs)
        object.__setattr__(
            self, "receipt_shas", _string_tuple(self.receipt_shas, field="receipt_shas", sha=True)
        )
        object.__setattr__(self, "tags", _string_tuple(self.tags, field="tags"))

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": self.evidence_kind,
            "evidence_id": self.evidence_id,
            "capability_id": self.capability_id,
            "baseline_policy": self.baseline_policy,
            "intervention_policy": self.intervention_policy,
            "cohort_definition": _plain(self.cohort_definition),
            "outcomes": _plain(self.outcomes),
            "risk": _plain(self.risk),
            "causal_scope": self.causal_scope,
            "source_refs": list(self.source_refs),
            "receipt_shas": list(self.receipt_shas),
            "tags": list(self.tags),
        }

    @property
    def evidence_sha(self) -> str:
        return canonical_sha256(self.identity_payload())


Evidence = LocalBehaviorEvidence | PolicyInterventionEvidence


def _evidence_from_payload(payload: Mapping[str, object]) -> Evidence:
    kind = payload.get("kind")
    if kind == LocalBehaviorEvidence.evidence_kind:
        return LocalBehaviorEvidence(
            evidence_id=payload.get("evidence_id"),
            capability_id=payload.get("capability_id"),
            pattern_node_id=payload.get("pattern_node_id"),
            program_binding_id=payload.get("program_binding_id"),
            behavior=payload.get("behavior"),
            receipt_shas=payload.get("receipt_shas", ()),
            tags=payload.get("tags", ()),
        )
    if kind == PolicyInterventionEvidence.evidence_kind:
        return PolicyInterventionEvidence(
            evidence_id=payload.get("evidence_id"),
            capability_id=payload.get("capability_id"),
            baseline_policy=payload.get("baseline_policy"),
            intervention_policy=payload.get("intervention_policy"),
            cohort_definition=payload.get("cohort_definition"),
            outcomes=payload.get("outcomes"),
            risk=payload.get("risk"),
            causal_scope=payload.get("causal_scope"),
            source_refs=payload.get("source_refs", ()),
            receipt_shas=payload.get("receipt_shas", ()),
            tags=payload.get("tags", ()),
        )
    raise ValueError("unknown evidence kind")


class EvidenceDisposition(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class SignedEvidenceRecord:
    evidence: Evidence
    disposition: EvidenceDisposition
    recorded_by: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "signed-evidence-row/1",
            "disposition": self.disposition.value,
            "recorded_by": self.recorded_by,
            "evidence_sha": self.evidence.evidence_sha,
            "evidence": self.evidence.identity_payload(),
        }


class SignedEvidenceLedger:
    """One append-only log where "signed" means evidence polarity, not cryptography."""

    def __init__(self, ledger_id: str) -> None:
        self.ledger_id = _nonempty(ledger_id, field="ledger_id")
        self._records: list[SignedEvidenceRecord] = []

    def append(
        self,
        evidence: Evidence,
        disposition: EvidenceDisposition | str,
        *,
        recorded_by: str,
    ) -> SignedEvidenceRecord:
        if not isinstance(evidence, (LocalBehaviorEvidence, PolicyInterventionEvidence)):
            raise TypeError("evidence must be a supported evidence contract")
        disposition = EvidenceDisposition(disposition)
        recorded_by = _nonempty(recorded_by, field="recorded_by")
        if any(record.evidence.evidence_sha == evidence.evidence_sha for record in self._records):
            raise ValueError("evidence is already present in this ledger")
        record = SignedEvidenceRecord(evidence, disposition, recorded_by)
        self._records.append(record)
        return record

    def to_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(record.as_dict() for record in self._records)

    @classmethod
    def from_rows(
        cls,
        ledger_id: str,
        rows: Sequence[Mapping[str, object]],
    ) -> "SignedEvidenceLedger":
        ledger = cls(ledger_id)
        for row in rows:
            if row.get("schema_version") != "signed-evidence-row/1":
                raise ValueError("unsupported signed evidence row")
            payload = row.get("evidence")
            if not isinstance(payload, Mapping):
                raise ValueError("signed evidence row requires an evidence object")
            evidence = _evidence_from_payload(payload)
            if row.get("evidence_sha") != evidence.evidence_sha:
                raise ValueError("evidence SHA mismatch")
            ledger.append(
                evidence,
                EvidenceDisposition(row.get("disposition")),
                recorded_by=_nonempty(row.get("recorded_by"), field="recorded_by"),
            )
        return ledger

    def read(self, capability_id: str | None = None) -> tuple[SignedEvidenceRecord, ...]:
        if capability_id is None:
            return tuple(self._records)
        _nonempty(capability_id, field="capability_id")
        return tuple(
            record for record in self._records if record.evidence.capability_id == capability_id
        )

    def summarize(self, capability_id: str) -> Mapping[str, object]:
        records = self.read(capability_id)
        counts = {disposition.value: 0 for disposition in EvidenceDisposition}
        kind_counts = {"local_behavior": 0, "policy_intervention": 0}
        for record in records:
            counts[record.disposition.value] += 1
            kind_counts[record.evidence.evidence_kind] += 1
        return MappingProxyType(
            {
                "capability_id": capability_id,
                "evidence_count": len(records),
                "dispositions": MappingProxyType(counts),
                "evidence_kinds": MappingProxyType(kind_counts),
                "compiled_verdict": self.compile_verdict(capability_id).value,
            }
        )

    def compile_verdict(self, capability_id: str) -> EvidenceDisposition:
        """Compile only policy evidence; local behavior cannot promote a capability."""

        policy = [
            record
            for record in self.read(capability_id)
            if isinstance(record.evidence, PolicyInterventionEvidence)
        ]
        dispositions = {record.disposition for record in policy}
        if EvidenceDisposition.CONTRADICTED in dispositions:
            return EvidenceDisposition.CONTRADICTED
        if EvidenceDisposition.SUPPORTED in dispositions:
            return EvidenceDisposition.SUPPORTED
        return EvidenceDisposition.UNRESOLVED


__all__ = [
    "CapabilityTemplate",
    "EvidenceDisposition",
    "LocalBehaviorEvidence",
    "ObservationScope",
    "PatternNode",
    "PolicyInterventionEvidence",
    "ScopedProgramBinding",
    "ScopedObservationReceipt",
    "SignedEvidenceLedger",
    "SignedEvidenceRecord",
]
