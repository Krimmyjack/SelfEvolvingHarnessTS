from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.contracts.observables import (
    OBSERVABLE_FEATURES,
    validate_applicability,
)
from SelfEvolvingHarnessTS.contracts.public_boundary import assert_public_payload
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import CaseFeedback


_BEHAVIOR_ALLOWLIST = frozenset(
    {
        "schema_version",
        "inspected_region_fractions",
        "tool_names",
        "retrieved_skill_ids",
        "retrieved_memory_ids",
        "candidate_program_shas",
        "chosen_candidate_id",
        "compilation_status",
        "execution_status",
        "modified_region_fractions",
        "verification_actions",
        "effect_equivalent_to_identity",
    }
)
_PROBE_POINT_ALLOWLIST = frozenset(
    {"probe_id", "beta", "r_public", "modified_fraction", "response_shape", "receipt_sha"}
)


def _numeric_bin(feature: str, value: float) -> str:
    if feature in {
        "missing_fraction",
        "longest_missing_run_fraction",
        "estimated_region_start_fraction",
        "estimated_region_end_fraction",
    }:
        edges = (0.0, 0.01, 0.05, 0.20)
    elif feature == "period_change_score":
        edges = (0.0, 0.10, 0.25, 0.50)
    else:
        edges = (0.0, 1.0, 3.0, 6.0)
    labels = ("zero", "very_low", "low", "medium", "high")
    if value <= edges[0]:
        return labels[0]
    if value < edges[1]:
        return labels[1]
    if value < edges[2]:
        return labels[2]
    if value < edges[3]:
        return labels[3]
    return labels[4]


def _observable_signature(features: Mapping[str, object]) -> dict[str, object]:
    signature: dict[str, object] = {}
    for feature, value in sorted(features.items()):
        if feature not in OBSERVABLE_FEATURES:
            continue
        kind = OBSERVABLE_FEATURES[feature]
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            signature[feature] = _numeric_bin(feature, float(value))
        elif kind == "boolean" and isinstance(value, bool):
            signature[feature] = value
        elif kind == "string" and isinstance(value, str):
            signature[feature] = value
    return signature


def _sanitize_behavior(behavior: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key in sorted(set(behavior) & _BEHAVIOR_ALLOWLIST):
        sanitized[key] = behavior[key]
    assert_public_payload(sanitized)
    return sanitized


def _sanitize_probe_points(curves: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    points: list[Mapping[str, object]] = []
    for probe_id, raw in sorted(curves.items()):
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            point = {
                "probe_id": str(probe_id),
                "r_public": round(float(raw), 6),
            }
            points.append(MappingProxyType(point))
            continue
        if isinstance(raw, Mapping):
            raw_points = (raw,)
        elif isinstance(raw, (list, tuple)):
            raw_points = tuple(item for item in raw if isinstance(item, Mapping))
        else:
            continue
        for raw_point in raw_points:
            point = {
                str(key): raw_point[key]
                for key in sorted(set(raw_point) & _PROBE_POINT_ALLOWLIST)
            }
            point.setdefault("probe_id", str(probe_id))
            if isinstance(point.get("r_public"), (int, float)):
                point["r_public"] = round(float(point["r_public"]), 6)
            if isinstance(point.get("modified_fraction"), (int, float)):
                point["modified_fraction"] = round(float(point["modified_fraction"]), 6)
            assert_public_payload(point)
            points.append(MappingProxyType(point))
    return tuple(points)


@dataclass(frozen=True)
class FailurePatternEvidence:
    schema_version: str
    case_id: str
    first_stage: str | None
    fault_code: str
    cause_code: str
    actionability: str
    observable_signature: Mapping[str, object]
    observable_signature_hash: str
    probe_points: tuple[Mapping[str, object], ...]
    behavior_signature: Mapping[str, object]
    suspect_surface_templates: tuple[str, ...]
    public_tool_ids: tuple[str, ...]
    public_skill_ids: tuple[str, ...]
    intervention_receipt_ids: tuple[str, ...]
    observable_applicability: Mapping[str, object] | None = None

    @property
    def confirmed_surface(self) -> None:
        return None

    def with_applicability(
        self,
        applicability: Mapping[str, object],
    ) -> "FailurePatternEvidence":
        validate_applicability(applicability)
        return replace(
            self,
            observable_applicability=MappingProxyType(dict(applicability)),
        )

    def to_json(self) -> dict[str, object]:
        result = {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "first_stage": self.first_stage,
            "fault_code": self.fault_code,
            "cause_code": self.cause_code,
            "actionability": self.actionability,
            "observable_signature": dict(self.observable_signature),
            "observable_signature_hash": self.observable_signature_hash,
            "probe_points": [dict(point) for point in self.probe_points],
            "behavior_signature": dict(self.behavior_signature),
            "suspect_surface_templates": list(self.suspect_surface_templates),
            "public_tool_ids": list(self.public_tool_ids),
            "public_skill_ids": list(self.public_skill_ids),
            "intervention_receipt_ids": list(self.intervention_receipt_ids),
            "observable_applicability": (
                None
                if self.observable_applicability is None
                else dict(self.observable_applicability)
            ),
        }
        assert_public_payload(result)
        return result


def sanitize_case_feedback(feedback: CaseFeedback) -> FailurePatternEvidence:
    if not isinstance(feedback, CaseFeedback):
        raise TypeError("sanitizer accepts private CaseFeedback only")
    signature = _observable_signature(feedback.mechanism.observable_features)
    probe_points = _sanitize_probe_points(feedback.mechanism.r_public_curves)
    behavior = _sanitize_behavior(feedback.behavior.behavior_signature)
    tool_ids = tuple(str(value) for value in behavior.get("tool_names", ()))
    skill_ids = tuple(str(value) for value in behavior.get("retrieved_skill_ids", ()))
    receipt_ids = tuple(
        sorted(
            {
                str(point["receipt_sha"])
                for point in probe_points
                if isinstance(point.get("receipt_sha"), str)
            }
        )
    )
    result = FailurePatternEvidence(
        schema_version="failure-pattern-evidence/1",
        case_id=feedback.case_id,
        first_stage=feedback.fault_attribution.first_stage,
        fault_code=feedback.fault_attribution.fault_code,
        cause_code=feedback.fault_attribution.cause_code,
        actionability=feedback.fault_attribution.actionability,
        observable_signature=MappingProxyType(signature),
        observable_signature_hash=canonical_sha256(
            {"schema_version": "observable-signature/1", "features": signature}
        ),
        probe_points=probe_points,
        behavior_signature=MappingProxyType(behavior),
        suspect_surface_templates=tuple(
            feedback.update_attribution.suspect_surface_templates
        ),
        public_tool_ids=tool_ids,
        public_skill_ids=skill_ids,
        intervention_receipt_ids=receipt_ids,
    )
    result.to_json()
    return result


class PublicArtifactReader:
    def __init__(self, public_root: Path) -> None:
        self.public_root = Path(public_root).resolve()

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.public_root / candidate).resolve()
        if not resolved.is_relative_to(self.public_root):
            raise PermissionError("artifact path is outside the configured public root")
        if resolved.suffix not in {".json", ".jsonl", ".md"}:
            raise PermissionError("public artifact suffix is not allowlisted")
        return resolved

    def read_json(self, path: str | Path) -> object:
        resolved = self._resolve(path)
        if resolved.suffix == ".md":
            return resolved.read_text(encoding="utf-8")
        if resolved.suffix == ".json":
            return parse_json_document(resolved.read_bytes())
        rows = []
        for line_number, line in enumerate(resolved.read_bytes().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(parse_json_document(line))
            except ValueError as exc:
                raise ValueError(f"invalid public JSONL row {line_number}") from exc
        return rows


__all__ = [
    "FailurePatternEvidence",
    "PublicArtifactReader",
    "sanitize_case_feedback",
]
