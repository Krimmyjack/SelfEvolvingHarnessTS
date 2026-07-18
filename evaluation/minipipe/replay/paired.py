from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.contracts.harness import EditManifest
from SelfEvolvingHarnessTS.evaluation.minipipe.config import M0Rules
from SelfEvolvingHarnessTS.methods.ttha.harness.store import MaterializedSnapshot
from SelfEvolvingHarnessTS.runtime.errors import InfrastructureError

from .edit_controller import AppliedEditReceipt


class ReplayEvaluationStatus(str, Enum):
    OK = "OK"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"


class EditVerdict(str, Enum):
    DEAD_EDIT = "DEAD_EDIT"
    BEHAVIOR_CHANGED_NO_GAIN = "BEHAVIOR_CHANGED_NO_GAIN"
    TARGET_RECOVERED_WITH_HARM = "TARGET_RECOVERED_WITH_HARM"
    PARTIAL_RECOVERY = "PARTIAL_RECOVERY"
    SUPPORTED_EDIT = "SUPPORTED_EDIT"
    UNEXPECTED_GAIN = "UNEXPECTED_GAIN"
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def promotion_eligible(self) -> bool:
        return self is EditVerdict.SUPPORTED_EDIT


@dataclass(frozen=True)
class ReplayFacts:
    evaluation_status: ReplayEvaluationStatus
    prediction_verified: bool
    behavior_change_status: str
    target_outcome_status: str
    risk_status: str
    scope_status: str
    target_recovery_fraction: float
    median_target_improvement: float
    risk_set_miss: bool

    @classmethod
    def from_target_improvements(
        cls,
        improvements: Sequence[float],
        *,
        prediction_verified: bool,
        behavior_changed: bool,
        risk_pass: bool,
        scope_pass: bool,
        rules: M0Rules | Mapping[str, object],
        risk_set_miss: bool = False,
    ) -> "ReplayFacts":
        values = np.asarray(tuple(float(value) for value in improvements), dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("target replay requires finite improvements")
        gain_min = float(rules["candidate_gain_min"])
        recovery_fraction = float(np.mean(values >= gain_min))
        median = float(np.median(values))
        full = (
            recovery_fraction >= float(rules["target_recovery_fraction"])
            and median >= float(rules["target_median_gain_min"])
        )
        if full:
            target_status = "FULL_RECOVERY"
        elif median > float(rules["utility_tolerance"]):
            target_status = "PARTIAL_RECOVERY"
        else:
            target_status = "NO_GAIN"
        return cls(
            evaluation_status=ReplayEvaluationStatus.OK,
            prediction_verified=bool(prediction_verified),
            behavior_change_status="CHANGED" if behavior_changed else "UNCHANGED",
            target_outcome_status=target_status,
            risk_status="PASS" if risk_pass else "FAIL",
            scope_status="PASS" if scope_pass else "FAIL",
            target_recovery_fraction=recovery_fraction,
            median_target_improvement=median,
            risk_set_miss=bool(risk_set_miss),
        )


def derive_verdict(facts: ReplayFacts) -> EditVerdict:
    if facts.evaluation_status is ReplayEvaluationStatus.INFRASTRUCTURE_FAILURE:
        return EditVerdict.INCONCLUSIVE
    has_target_gain = facts.target_outcome_status in {
        "FULL_RECOVERY",
        "PARTIAL_RECOVERY",
    }
    if has_target_gain and (
        facts.risk_status != "PASS" or facts.scope_status != "PASS"
    ):
        return EditVerdict.TARGET_RECOVERED_WITH_HARM
    if not facts.prediction_verified:
        if has_target_gain and facts.behavior_change_status == "CHANGED":
            return EditVerdict.UNEXPECTED_GAIN
        return EditVerdict.DEAD_EDIT
    if facts.target_outcome_status == "NO_GAIN":
        return EditVerdict.BEHAVIOR_CHANGED_NO_GAIN
    if facts.target_outcome_status == "PARTIAL_RECOVERY":
        return EditVerdict.PARTIAL_RECOVERY
    if facts.target_outcome_status == "FULL_RECOVERY":
        return EditVerdict.SUPPORTED_EDIT
    raise ValueError(f"unknown target outcome status: {facts.target_outcome_status}")


@dataclass(frozen=True)
class OutOfScopePair:
    case_id: str
    new_skill_applicability_match: bool
    effective_view_equal: bool
    all_eligible_calls_reused: bool
    behavior_equal: bool

    @property
    def scope_status(self) -> str:
        return (
            "PASS"
            if self.effective_view_equal
            and self.all_eligible_calls_reused
            and self.behavior_equal
            else "FAIL"
        )


@dataclass(frozen=True)
class CaseRunReceipt:
    case_id: str
    utility_u: float
    effective_harness_view_sha: str
    behavior_signature_sha: str
    eligible_agent_calls: int
    cache_hit_flags: tuple[bool, ...]
    retrieved_skill_ids: tuple[str, ...] = ()
    supplied_operator_ids: tuple[str, ...] = ()
    supplied_effect_distinct: bool = False
    chosen_candidate_kind: str = "identity"
    identity_retained: bool = True
    modified_fraction: float = 0.0
    localization_iou: float | None = None
    run_context_sha: str = ""
    agent_decision_status: str = "ASSESSED"
    system_capability_status: str = "AVAILABLE_OR_UNKNOWN"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.utility_u)):
            raise ValueError("case replay utility must be finite")
        if self.chosen_candidate_kind not in {"identity", "program"}:
            raise ValueError("chosen candidate kind must be identity or program")
        if not 0.0 <= float(self.modified_fraction) <= 1.0:
            raise ValueError("modified fraction must lie in [0, 1]")
        if self.localization_iou is not None and not 0.0 <= float(
            self.localization_iou
        ) <= 1.0:
            raise ValueError("localization IoU must lie in [0, 1]")

    @property
    def all_eligible_calls_reused(self) -> bool:
        return (
            len(self.cache_hit_flags) >= self.eligible_agent_calls
            and all(self.cache_hit_flags[: self.eligible_agent_calls])
        )


class CaseRunner(Protocol):
    def run(
        self,
        snapshot: MaterializedSnapshot,
        case: object,
        cache: object,
    ) -> CaseRunReceipt: ...


@dataclass(frozen=True)
class PairedReplayReport:
    schema_version: str
    edit_id: str
    parent_runtime_bundle_sha: str
    candidate_runtime_bundle_sha: str
    rules_sha: str
    facts: ReplayFacts
    verdict: EditVerdict
    target_case_ids: tuple[str, ...]
    risk_case_ids: tuple[str, ...]
    out_of_scope_pairs: tuple[OutOfScopePair, ...]
    confirmed_surface: str | None
    attribution_additions: tuple[str, ...]
    infrastructure_error: str | None
    report_sha: str

    @property
    def promotion_eligible(self) -> bool:
        return self.verdict.promotion_eligible

    def to_private_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "edit_id": self.edit_id,
            "parent_runtime_bundle_sha": self.parent_runtime_bundle_sha,
            "candidate_runtime_bundle_sha": self.candidate_runtime_bundle_sha,
            "rules_sha": self.rules_sha,
            "facts": {
                "evaluation_status": self.facts.evaluation_status.value,
                "prediction_verified": self.facts.prediction_verified,
                "behavior_change_status": self.facts.behavior_change_status,
                "target_outcome_status": self.facts.target_outcome_status,
                "risk_status": self.facts.risk_status,
                "scope_status": self.facts.scope_status,
                "target_recovery_fraction": self.facts.target_recovery_fraction,
                "median_target_improvement": self.facts.median_target_improvement,
                "risk_set_miss": self.facts.risk_set_miss,
            },
            "verdict": self.verdict.value,
            "target_case_ids": list(self.target_case_ids),
            "risk_case_ids": list(self.risk_case_ids),
            "out_of_scope_pairs": [
                {
                    "case_id": pair.case_id,
                    "new_skill_applicability_match": pair.new_skill_applicability_match,
                    "effective_view_equal": pair.effective_view_equal,
                    "all_eligible_calls_reused": pair.all_eligible_calls_reused,
                    "behavior_equal": pair.behavior_equal,
                    "scope_status": pair.scope_status,
                }
                for pair in self.out_of_scope_pairs
            ],
            "confirmed_surface": self.confirmed_surface,
            "attribution_additions": list(self.attribution_additions),
            "infrastructure_error": self.infrastructure_error,
            "report_sha": self.report_sha,
        }


def _case_id(case: object) -> str:
    value = getattr(case, "case_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("replay case must expose a non-empty case_id")
    return value


class PairedReplayRunner:
    def __init__(
        self,
        case_runner: CaseRunner,
        *,
        rules: M0Rules,
        cache: object,
    ) -> None:
        self.case_runner = case_runner
        self.rules = rules
        self.cache = cache

    def _run_with_retry(
        self,
        snapshot: MaterializedSnapshot,
        case: object,
    ) -> tuple[CaseRunReceipt | None, str | None]:
        attempts = 1 + int(self.rules["infrastructure_retries"])
        last: InfrastructureError | None = None
        for _ in range(attempts):
            try:
                receipt = self.case_runner.run(snapshot, case, self.cache)
                if receipt.case_id != _case_id(case):
                    raise ValueError("case runner returned the wrong case identity")
                return receipt, None
            except InfrastructureError as exc:
                last = exc
        return None, type(last).__name__ if last is not None else "InfrastructureError"

    @staticmethod
    def _prediction_verified(
        predicates: Sequence[str],
        candidate_receipts: Sequence[CaseRunReceipt],
        out_pairs: Sequence[OutOfScopePair],
    ) -> bool:
        for predicate in predicates:
            if predicate.startswith("retrieve_skill:"):
                skill_id = predicate.split(":", 1)[1]
                passed = any(skill_id in receipt.retrieved_skill_ids for receipt in candidate_receipts)
            elif predicate.startswith("supply_operator:"):
                operator_id = predicate.split(":", 1)[1]
                passed = any(
                    operator_id in receipt.supplied_operator_ids
                    for receipt in candidate_receipts
                )
            elif predicate == "supply_effect_distinct":
                passed = any(receipt.supplied_effect_distinct for receipt in candidate_receipts)
            elif predicate.startswith("choose_candidate_kind:"):
                kind = predicate.split(":", 1)[1]
                passed = any(receipt.chosen_candidate_kind == kind for receipt in candidate_receipts)
            elif predicate == "identity_retained":
                passed = all(receipt.identity_retained for receipt in candidate_receipts)
            elif predicate == "effective_view_unchanged_out_of_scope":
                passed = all(pair.scope_status == "PASS" for pair in out_pairs)
            elif predicate.startswith("scope_modified_fraction<="):
                limit = float(predicate.split("<=", 1)[1])
                passed = all(receipt.modified_fraction <= limit for receipt in candidate_receipts)
            elif predicate.startswith("localization_iou>="):
                threshold = float(predicate.split(">=", 1)[1])
                passed = all(
                    receipt.localization_iou is not None
                    and receipt.localization_iou >= threshold
                    for receipt in candidate_receipts
                )
            else:
                passed = False
            if not passed:
                return False
        return True

    def _inconclusive(
        self,
        *,
        manifest: EditManifest,
        applied: AppliedEditReceipt,
        target_case_ids: tuple[str, ...],
        risk_case_ids: tuple[str, ...],
        error: str,
    ) -> PairedReplayReport:
        facts = ReplayFacts(
            evaluation_status=ReplayEvaluationStatus.INFRASTRUCTURE_FAILURE,
            prediction_verified=False,
            behavior_change_status="UNKNOWN",
            target_outcome_status="UNKNOWN",
            risk_status="UNKNOWN",
            scope_status="UNKNOWN",
            target_recovery_fraction=0.0,
            median_target_improvement=0.0,
            risk_set_miss=False,
        )
        return self._report(
            manifest=manifest,
            applied=applied,
            facts=facts,
            target_case_ids=target_case_ids,
            risk_case_ids=risk_case_ids,
            out_pairs=(),
            error=error,
        )

    def _report(
        self,
        *,
        manifest: EditManifest,
        applied: AppliedEditReceipt,
        facts: ReplayFacts,
        target_case_ids: tuple[str, ...],
        risk_case_ids: tuple[str, ...],
        out_pairs: tuple[OutOfScopePair, ...],
        error: str | None,
    ) -> PairedReplayReport:
        verdict = derive_verdict(facts)
        confirmed = applied.target_surface_id if facts.prediction_verified else None
        additions = (
            ("UPDATE_MISATTRIBUTION",)
            if not facts.prediction_verified
            and facts.evaluation_status is ReplayEvaluationStatus.OK
            else ()
        )
        payload = {
            "schema_version": "paired-replay-report/1",
            "edit_id": manifest.edit_id,
            "parent_runtime_bundle_sha": applied.parent_runtime_bundle_sha,
            "candidate_runtime_bundle_sha": applied.candidate_runtime_bundle_sha,
            "rules_sha": self.rules.rules_sha,
            "facts": {
                "evaluation_status": facts.evaluation_status.value,
                "prediction_verified": facts.prediction_verified,
                "behavior_change_status": facts.behavior_change_status,
                "target_outcome_status": facts.target_outcome_status,
                "risk_status": facts.risk_status,
                "scope_status": facts.scope_status,
                "target_recovery_fraction": facts.target_recovery_fraction,
                "median_target_improvement": facts.median_target_improvement,
                "risk_set_miss": facts.risk_set_miss,
            },
            "verdict": verdict.value,
            "target_case_ids": list(target_case_ids),
            "risk_case_ids": list(risk_case_ids),
            "out_of_scope_pairs": [
                {
                    "case_id": pair.case_id,
                    "new_skill_applicability_match": pair.new_skill_applicability_match,
                    "effective_view_equal": pair.effective_view_equal,
                    "all_eligible_calls_reused": pair.all_eligible_calls_reused,
                    "behavior_equal": pair.behavior_equal,
                    "scope_status": pair.scope_status,
                }
                for pair in out_pairs
            ],
            "confirmed_surface": confirmed,
            "attribution_additions": list(additions),
            "infrastructure_error": error,
        }
        return PairedReplayReport(
            schema_version="paired-replay-report/1",
            edit_id=manifest.edit_id,
            parent_runtime_bundle_sha=applied.parent_runtime_bundle_sha,
            candidate_runtime_bundle_sha=applied.candidate_runtime_bundle_sha,
            rules_sha=self.rules.rules_sha,
            facts=facts,
            verdict=verdict,
            target_case_ids=target_case_ids,
            risk_case_ids=risk_case_ids,
            out_of_scope_pairs=out_pairs,
            confirmed_surface=confirmed,
            attribution_additions=additions,
            infrastructure_error=error,
            report_sha=canonical_sha256(payload),
        )

    def run(
        self,
        *,
        parent: MaterializedSnapshot,
        candidate: MaterializedSnapshot,
        applied: AppliedEditReceipt,
        manifest: EditManifest,
        target_cases: Sequence[object],
        risk_cases: Sequence[object],
        out_of_scope_case_ids: Sequence[str] = (),
        stage_b_cases: Sequence[object] = (),
    ) -> PairedReplayReport:
        target_ids = tuple(_case_id(case) for case in target_cases)
        risk_ids = tuple(_case_id(case) for case in risk_cases)
        all_cases = tuple((*target_cases, *risk_cases))
        baseline: dict[str, CaseRunReceipt] = {}
        edited: dict[str, CaseRunReceipt] = {}
        for case in all_cases:
            receipt, error = self._run_with_retry(parent, case)
            if receipt is None:
                return self._inconclusive(
                    manifest=manifest,
                    applied=applied,
                    target_case_ids=target_ids,
                    risk_case_ids=risk_ids,
                    error=error or "InfrastructureError",
                )
            baseline[receipt.case_id] = receipt
        for case in all_cases:
            receipt, error = self._run_with_retry(candidate, case)
            if receipt is None:
                return self._inconclusive(
                    manifest=manifest,
                    applied=applied,
                    target_case_ids=target_ids,
                    risk_case_ids=risk_ids,
                    error=error or "InfrastructureError",
                )
            edited[receipt.case_id] = receipt

        out_ids = set(str(case_id) for case_id in out_of_scope_case_ids)
        out_pairs = tuple(
            OutOfScopePair(
                case_id=case_id,
                new_skill_applicability_match=False,
                effective_view_equal=(
                    baseline[case_id].effective_harness_view_sha
                    == edited[case_id].effective_harness_view_sha
                ),
                all_eligible_calls_reused=edited[case_id].all_eligible_calls_reused,
                behavior_equal=(
                    baseline[case_id].behavior_signature_sha
                    == edited[case_id].behavior_signature_sha
                ),
            )
            for case_id in sorted(out_ids)
            if case_id in baseline and case_id in edited
        )
        scope_pass = all(pair.scope_status == "PASS" for pair in out_pairs)
        in_scope_risks = [case_id for case_id in risk_ids if case_id not in out_ids]
        risk_pass = all(
            edited[case_id].utility_u - baseline[case_id].utility_u
            >= -float(self.rules["risk_epsilon"])
            for case_id in in_scope_risks
        )
        scored_target_ids = tuple(
            case_id
            for case_id in target_ids
            if not (
                edited[case_id].agent_decision_status == "CORRECT_IDENTITY"
                and edited[case_id].system_capability_status == "OPERATOR_GAP"
            )
        )
        target_baseline = [baseline[case_id] for case_id in scored_target_ids]
        target_edited = [edited[case_id] for case_id in scored_target_ids]
        improvements = [
            edited_receipt.utility_u - baseline_receipt.utility_u
            for baseline_receipt, edited_receipt in zip(target_baseline, target_edited)
        ]
        behavior_changed = any(
            baseline_receipt.behavior_signature_sha != edited_receipt.behavior_signature_sha
            for baseline_receipt, edited_receipt in zip(target_baseline, target_edited)
        )
        prediction = self._prediction_verified(
            manifest.predicted_agent_behavior_change,
            target_edited,
            out_pairs,
        )
        facts = ReplayFacts.from_target_improvements(
            improvements,
            prediction_verified=prediction,
            behavior_changed=behavior_changed,
            risk_pass=risk_pass,
            scope_pass=scope_pass,
            rules=self.rules,
        )
        provisional = derive_verdict(facts)
        if provisional in {
            EditVerdict.SUPPORTED_EDIT,
            EditVerdict.PARTIAL_RECOVERY,
            EditVerdict.UNEXPECTED_GAIN,
        } and stage_b_cases:
            stage_a_ids = set((*target_ids, *risk_ids))
            regressions = False
            for case in stage_b_cases:
                case_id = _case_id(case)
                if case_id in stage_a_ids:
                    continue
                baseline_receipt, error = self._run_with_retry(parent, case)
                if baseline_receipt is None:
                    return self._inconclusive(
                        manifest=manifest,
                        applied=applied,
                        target_case_ids=target_ids,
                        risk_case_ids=risk_ids,
                        error=error or "InfrastructureError",
                    )
                edit_receipt, error = self._run_with_retry(candidate, case)
                if edit_receipt is None:
                    return self._inconclusive(
                        manifest=manifest,
                        applied=applied,
                        target_case_ids=target_ids,
                        risk_case_ids=risk_ids,
                        error=error or "InfrastructureError",
                    )
                if edit_receipt.utility_u - baseline_receipt.utility_u < -float(
                    self.rules["risk_epsilon"]
                ):
                    regressions = True
            if regressions:
                facts = replace(facts, risk_status="FAIL", risk_set_miss=True)
        return self._report(
            manifest=manifest,
            applied=applied,
            facts=facts,
            target_case_ids=target_ids,
            risk_case_ids=risk_ids,
            out_pairs=out_pairs,
            error=None,
        )


__all__ = [
    "CaseRunReceipt",
    "CaseRunner",
    "EditVerdict",
    "OutOfScopePair",
    "PairedReplayReport",
    "PairedReplayRunner",
    "ReplayEvaluationStatus",
    "ReplayFacts",
    "derive_verdict",
]
