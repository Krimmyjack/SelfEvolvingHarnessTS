from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.contracts.observables import validate_applicability
from SelfEvolvingHarnessTS.contracts.public_boundary import assert_public_payload
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import CaseFeedback

from .sanitize import FailurePatternEvidence, sanitize_case_feedback


def _plain_mapping(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, nested in value.items():
        if isinstance(nested, Mapping):
            result[str(key)] = _plain_mapping(nested)
        elif isinstance(nested, tuple):
            result[str(key)] = list(nested)
        else:
            result[str(key)] = nested
    return result


def _capability_state(cause_code: str) -> str:
    if cause_code in {"SKILL_LIBRARY_GAP", "OPERATOR_GAP"}:
        return "missing"
    if cause_code == "RETRIEVAL_MISS":
        return "misrouted"
    if cause_code in {
        "SKILL_CONTENT_GAP",
        "PROPOSAL_CONTROL_GAP",
        "SELECTION_MISS",
        "IMPLEMENTATION_MISMATCH",
        "EXECUTION_MISMATCH",
        "OUTCOME_GAP",
    }:
        return "insufficient"
    if cause_code == "RISK_GAP":
        return "risk"
    return "unknown"


@dataclass(frozen=True)
class FailurePatternCard:
    schema_version: str
    pattern_id: str
    support_count: int
    case_ids: tuple[str, ...]
    observable_signature: Mapping[str, object]
    observable_signature_hash: str
    contexts: Mapping[str, object]
    first_stage: str | None
    fault_code: str
    cause_code: str
    actionability: str
    common_behavior_signature: Mapping[str, object]
    sanitized_intervention_receipts: tuple[Mapping[str, object], ...]
    matched_success_case_ids: tuple[str, ...]
    counterexample_case_ids: tuple[str, ...]
    missing_success_contrast: bool
    suspect_surface_templates: tuple[str, ...]
    capability_state: str
    observable_applicability: Mapping[str, object] | None = None

    @property
    def confirmed_surface(self) -> None:
        return None

    def with_applicability(
        self,
        applicability: Mapping[str, object],
    ) -> "FailurePatternCard":
        validate_applicability(applicability)
        return replace(
            self,
            observable_applicability=MappingProxyType(dict(applicability)),
        )

    def to_json(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "pattern_id": self.pattern_id,
            "support_count": self.support_count,
            "case_ids": list(self.case_ids),
            "observable_signature": dict(self.observable_signature),
            "observable_signature_hash": self.observable_signature_hash,
            "contexts": dict(self.contexts),
            "first_stage": self.first_stage,
            "fault_code": self.fault_code,
            "cause_code": self.cause_code,
            "actionability": self.actionability,
            "common_behavior_signature": _plain_mapping(self.common_behavior_signature),
            "sanitized_intervention_receipts": [
                dict(receipt) for receipt in self.sanitized_intervention_receipts
            ],
            "matched_success_case_ids": list(self.matched_success_case_ids),
            "counterexample_case_ids": list(self.counterexample_case_ids),
            "missing_success_contrast": self.missing_success_contrast,
            "suspect_surface_templates": list(self.suspect_surface_templates),
            "capability_state": self.capability_state,
            "observable_applicability": (
                None
                if self.observable_applicability is None
                else dict(self.observable_applicability)
            ),
        }
        assert_public_payload(payload)
        return payload


@dataclass(frozen=True)
class ClusterPurityReceipt:
    pattern_id: str
    oracle_mechanism_purity: float | None
    best_intervention_purity: float | None
    target_surface_purity: float | None
    support_count: int
    low_mechanism_purity: bool
    receipt_sha: str


def _mode_mapping(evidences: Sequence[FailurePatternEvidence]) -> Mapping[str, object]:
    identities = [canonical_sha256(dict(evidence.behavior_signature)) for evidence in evidences]
    if not identities:
        return MappingProxyType({})
    winner = sorted(Counter(identities).items(), key=lambda item: (-item[1], item[0]))[0][0]
    selected = min(
        (evidence for evidence, identity in zip(evidences, identities) if identity == winner),
        key=lambda evidence: evidence.case_id,
    )
    return MappingProxyType(dict(selected.behavior_signature))


def _match_successes(
    evidence: FailurePatternEvidence,
    successes: Sequence[FailurePatternEvidence],
) -> tuple[str, ...]:
    exact = sorted(
        success.case_id
        for success in successes
        if success.observable_signature_hash == evidence.observable_signature_hash
    )
    return tuple(exact[:1])


def mine_failure_patterns(
    feedback_records: Sequence[CaseFeedback],
    *,
    successful_records: Sequence[CaseFeedback] = (),
    minimum_support: int = 2,
) -> tuple[FailurePatternCard, ...]:
    if minimum_support < 2:
        raise ValueError("recurring pattern support must be at least two")
    sanitized = tuple(sanitize_case_feedback(feedback) for feedback in feedback_records)
    successes = [
        sanitize_case_feedback(feedback)
        for feedback in (*feedback_records, *successful_records)
        if feedback.fault_attribution.fault_code == "NO_ACTIONABLE_FAULT"
    ]
    failures = [
        evidence for evidence in sanitized if evidence.fault_code != "NO_ACTIONABLE_FAULT"
    ]
    buckets: dict[tuple[object, ...], list[FailurePatternEvidence]] = defaultdict(list)
    for evidence in failures:
        key = (
            evidence.fault_code,
            evidence.cause_code,
            evidence.suspect_surface_templates,
            evidence.observable_signature_hash,
        )
        buckets[key].append(evidence)

    cards: list[FailurePatternCard] = []
    for key, raw_evidences in sorted(buckets.items(), key=lambda item: repr(item[0])):
        evidences = tuple(sorted(raw_evidences, key=lambda item: item.case_id))
        if len(evidences) < minimum_support:
            continue
        exemplar = evidences[0]
        matched = tuple(
            sorted(
                {
                    case_id
                    for evidence in evidences
                    for case_id in _match_successes(evidence, successes)
                }
            )
        )
        receipts_by_sha: dict[str, Mapping[str, object]] = {}
        for evidence in evidences:
            for point in evidence.probe_points:
                identity = str(point.get("receipt_sha", canonical_sha256(dict(point))))
                receipts_by_sha.setdefault(identity, point)
        public_content = {
            "schema_version": "failure-pattern-content/1",
            "case_ids": [evidence.case_id for evidence in evidences],
            "observable_signature": dict(exemplar.observable_signature),
            "first_stage": exemplar.first_stage,
            "fault_code": exemplar.fault_code,
            "cause_code": exemplar.cause_code,
            "actionability": exemplar.actionability,
            "behavior_signature": dict(_mode_mapping(evidences)),
            "probe_receipts": [dict(value) for _, value in sorted(receipts_by_sha.items())],
            "matched_success_case_ids": list(matched),
            "suspect_surface_templates": list(exemplar.suspect_surface_templates),
        }
        pattern_id = f"pattern-{canonical_sha256(public_content)[:12]}"
        card = FailurePatternCard(
            schema_version="failure-pattern-card/1",
            pattern_id=pattern_id,
            support_count=len(evidences),
            case_ids=tuple(evidence.case_id for evidence in evidences),
            observable_signature=exemplar.observable_signature,
            observable_signature_hash=exemplar.observable_signature_hash,
            contexts=MappingProxyType(
                {
                    "task_kind": exemplar.observable_signature.get("task_kind", "unknown"),
                    "probe_point_count": len(receipts_by_sha),
                }
            ),
            first_stage=exemplar.first_stage,
            fault_code=exemplar.fault_code,
            cause_code=exemplar.cause_code,
            actionability=exemplar.actionability,
            common_behavior_signature=_mode_mapping(evidences),
            sanitized_intervention_receipts=tuple(
                value for _, value in sorted(receipts_by_sha.items())
            ),
            matched_success_case_ids=matched,
            counterexample_case_ids=matched,
            missing_success_contrast=not matched,
            suspect_surface_templates=exemplar.suspect_surface_templates,
            capability_state=_capability_state(exemplar.cause_code),
        )
        card.to_json()
        cards.append(card)
    return tuple(sorted(cards, key=lambda card: card.pattern_id))


def _purity(values: Sequence[str]) -> float | None:
    if not values:
        return None
    return max(Counter(values).values()) / len(values)


def compute_cluster_purity(
    card: FailurePatternCard,
    private_feedback: Sequence[CaseFeedback],
    *,
    low_purity_threshold: float = 0.80,
) -> ClusterPurityReceipt:
    selected = [feedback for feedback in private_feedback if feedback.case_id in set(card.case_ids)]
    families = [
        feedback.mechanism.oracle_family
        for feedback in selected
        if feedback.mechanism.oracle_family is not None
    ]
    best_interventions: list[str] = []
    surfaces: list[str] = []
    for feedback in selected:
        curves = feedback.mechanism.r_private_curves
        numeric = {
            str(key): float(value)
            for key, value in curves.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric:
            best_interventions.append(max(numeric, key=numeric.get))
        surfaces.extend(feedback.update_attribution.suspect_surface_templates[:1])
    mechanism_purity = _purity([str(value) for value in families])
    payload = {
        "schema_version": "cluster-purity-receipt/1",
        "pattern_id": card.pattern_id,
        "oracle_mechanism_purity": mechanism_purity,
        "best_intervention_purity": _purity(best_interventions),
        "target_surface_purity": _purity(surfaces),
        "support_count": len(selected),
        "low_mechanism_purity": (
            mechanism_purity is not None and mechanism_purity < low_purity_threshold
        ),
    }
    return ClusterPurityReceipt(
        pattern_id=card.pattern_id,
        oracle_mechanism_purity=mechanism_purity,
        best_intervention_purity=payload["best_intervention_purity"],
        target_surface_purity=payload["target_surface_purity"],
        support_count=len(selected),
        low_mechanism_purity=bool(payload["low_mechanism_purity"]),
        receipt_sha=canonical_sha256(payload),
    )


__all__ = [
    "ClusterPurityReceipt",
    "FailurePatternCard",
    "compute_cluster_purity",
    "mine_failure_patterns",
]
