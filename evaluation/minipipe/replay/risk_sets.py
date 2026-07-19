from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import (
    CaseFeedback,
    CasePurpose,
    PrivateSyntheticCase,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.patterns import FailurePatternCard
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.sanitize import sanitize_case_feedback


_BIN_ORDINAL = {"zero": 0.0, "very_low": 0.25, "low": 0.5, "medium": 0.75, "high": 1.0}


def _signature_distance(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    keys = sorted(set(left) | set(right))
    distance = 0.0
    for key in keys:
        a = left.get(key)
        b = right.get(key)
        if isinstance(a, str) and a in _BIN_ORDINAL and isinstance(b, str) and b in _BIN_ORDINAL:
            distance += (_BIN_ORDINAL[a] - _BIN_ORDINAL[b]) ** 2
        elif a != b:
            distance += 1.0
    return distance**0.5


@dataclass(frozen=True)
class AutomaticRiskSetReceipt:
    pattern_id: str
    case_ids: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]
    receipt_sha: str


class AutomaticRiskSetBuilder:
    CATEGORY_ORDER = (
        "same_signature_clean",
        "nearest_genuine_event",
        "adjacent_severity_target",
        "baseline_success_regression",
        "opposite_probe_direction",
    )

    def build(
        self,
        pattern: FailurePatternCard,
        corpus: Sequence[PrivateSyntheticCase],
        baseline_feedback: Mapping[str, CaseFeedback],
    ) -> AutomaticRiskSetReceipt:
        cases = {case.case_id: case for case in corpus}
        evidence = {
            case_id: sanitize_case_feedback(feedback)
            for case_id, feedback in baseline_feedback.items()
        }
        pattern_ids = set(pattern.case_ids)
        categories: dict[str, tuple[str, ...]] = {}

        clean = sorted(
            case.case_id
            for case in corpus
            if case.purpose is CasePurpose.RISK_CLEAN
            and case.case_id in evidence
            and evidence[case.case_id].observable_signature_hash
            == pattern.observable_signature_hash
        )
        categories["same_signature_clean"] = tuple(clean)

        genuine = [
            case
            for case in corpus
            if case.purpose is CasePurpose.RISK_GENUINE_EVENT and case.case_id in evidence
        ]
        genuine.sort(
            key=lambda case: (
                _signature_distance(
                    pattern.observable_signature,
                    evidence[case.case_id].observable_signature,
                ),
                case.case_id,
            )
        )
        categories["nearest_genuine_event"] = tuple(case.case_id for case in genuine[:3])

        adjacent: list[str] = []
        for pattern_case_id in sorted(pattern_ids):
            source = cases.get(pattern_case_id)
            if source is None or source.purpose is not CasePurpose.TARGET:
                continue
            for candidate in corpus:
                if (
                    candidate.purpose is CasePurpose.TARGET
                    and candidate.seed == source.seed
                    and candidate.private_family == source.private_family
                    and candidate.private_severity != source.private_severity
                    and candidate.case_id not in pattern_ids
                ):
                    adjacent.append(candidate.case_id)
        categories["adjacent_severity_target"] = tuple(sorted(set(adjacent)))

        successes = sorted(
            case_id
            for case_id, feedback in baseline_feedback.items()
            if feedback.fault_attribution.fault_code == "NO_ACTIONABLE_FAULT"
            and (
                feedback.outcome.agent_decision_status == "CORRECT_IDENTITY"
                or feedback.outcome.repair_gain_g >= 0.0
            )
        )
        categories["baseline_success_regression"] = tuple(successes)

        proposed_probe: str | None = None
        for point in pattern.sanitized_intervention_receipts:
            if isinstance(point.get("probe_id"), str):
                proposed_probe = str(point["probe_id"])
                break
        opposite: list[str] = []
        if proposed_probe is not None:
            for case_id, feedback in baseline_feedback.items():
                value = feedback.mechanism.r_public_curves.get(proposed_probe)
                if isinstance(value, (int, float)) and float(value) < 0.0:
                    opposite.append(case_id)
        categories["opposite_probe_direction"] = tuple(sorted(opposite))

        ordered_ids: list[str] = []
        seen: set[str] = set(pattern_ids)
        for category in self.CATEGORY_ORDER:
            for case_id in categories[category]:
                if case_id in seen:
                    continue
                seen.add(case_id)
                ordered_ids.append(case_id)
        frozen_categories = MappingProxyType(
            {category: tuple(categories[category]) for category in self.CATEGORY_ORDER}
        )
        payload = {
            "schema_version": "automatic-risk-set/1",
            "pattern_id": pattern.pattern_id,
            "case_ids": ordered_ids,
            "categories": {
                category: list(frozen_categories[category])
                for category in self.CATEGORY_ORDER
            },
        }
        return AutomaticRiskSetReceipt(
            pattern_id=pattern.pattern_id,
            case_ids=tuple(ordered_ids),
            categories=frozen_categories,
            receipt_sha=canonical_sha256(payload),
        )


__all__ = ["AutomaticRiskSetBuilder", "AutomaticRiskSetReceipt"]
