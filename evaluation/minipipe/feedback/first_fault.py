from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from SelfEvolvingHarnessTS.evaluation.minipipe.config import M0Rules, load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import (
    AssessmentStatus,
    BehaviorFeedback,
    CaseFeedback,
    FaultAttribution,
    MechanismFeedback,
    OutcomeFeedback,
    Stage,
    StageAssessment,
    UpdateAttributionFeedback,
)

from .router import FaultRouter


STAGE_ORDER = tuple(stage.value for stage in Stage)
_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "m0_rules.json"


@dataclass(frozen=True)
class CaseFacts:
    case_id: str
    is_target: bool = True
    private_family: str | None = None
    oracle_affected_indices: tuple[int, ...] = ()
    valuation_source: str = "UNSPECIFIED"
    ingestion_policy_id: str = "UNSPECIFIED"
    clean_u: float = -0.10
    corrupt_u: float = -0.40
    prepared_u: float = -0.38
    damage_d: float = 0.30
    chosen_gain: float = 0.02
    candidate_utilities: Mapping[str, float] = field(default_factory=dict)
    effect_distinct_candidate_ids: tuple[str, ...] = ()
    chosen_candidate_id: str = "identity"
    chosen_probe_directions: tuple[str, ...] = ()
    public_evidence_discriminative: bool = True
    agent_inspected_evidence: bool = True
    localization_required: bool = True
    localization_iou: float | None = 0.50
    mechanism_identified: bool | None = True
    mechanism_contradiction: bool = False
    public_probe_gains: Mapping[str, float] = field(default_factory=dict)
    private_probe_gains: Mapping[str, object] = field(default_factory=dict)
    period_diagnostic_pass: bool = False
    period_diagnostic: Mapping[str, object] = field(default_factory=dict)
    observable_features: Mapping[str, object] = field(default_factory=dict)
    curve_agreement_receipt_ref: str | None = None
    witness_receipt_refs: tuple[str, ...] = ()
    implied_mechanism_claims: tuple[str, ...] = ()
    expressibility_status: str = "PROVEN_EXPRESSIBLE"
    expressibility_cause: str | None = None
    required_transformation_class: str | None = None
    observable_witness_succeeded: bool = True
    oracle_witness_succeeded: bool = True
    capability_skill_exists: bool = True
    normal_retrieval: bool = True
    skill_retrieved: bool = True
    forced_skill_succeeds: bool = True
    constrained_proposal_succeeds: bool = False
    proposed_candidate_exists: bool = True
    compilation_ok: bool = True
    compiled_candidate_exists: bool = True
    execution_ok: bool = True
    execution_contract_ok: bool = True
    risk_delta_u: float | None = None
    scope_stable: bool = True
    over_restoration: bool = False
    target_window_gain: float | None = None
    outside_window_change: float | None = None
    counterpart_change: float | None = None
    non_target_collateral: float | None = None
    behavior_signature: Mapping[str, object] = field(default_factory=dict)
    decision_trace_ref: str | None = None
    compilation_status: str = "OK"
    execution_status: str = "OK"
    stage_evidence_refs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    private_receipt_refs: tuple[str, ...] = ()

    @classmethod
    def passing(cls, *, case_id: str) -> "CaseFacts":
        return cls(
            case_id=case_id,
            candidate_utilities={"identity": -0.40, "agent-0": -0.38},
            effect_distinct_candidate_ids=("agent-0",),
            chosen_candidate_id="agent-0",
            public_probe_gains={"clipping": 0.02, "denoising": 0.0},
            observable_features={"local_robust_z_peak": 5.0},
        )


@dataclass(frozen=True)
class AssessmentResult:
    feedback: CaseFeedback
    assessments: tuple[StageAssessment, ...]
    attribution: FaultAttribution


def _rule_id(name: str, rules: M0Rules) -> str:
    return f"{name}@{rules.rules_sha}"


def _assessment(
    facts: CaseFacts,
    rules: M0Rules,
    stage: Stage,
    status: AssessmentStatus,
    *,
    rule: str,
    fault: str | None = None,
    cause: str | None = None,
    surfaces: tuple[str, ...] = (),
) -> StageAssessment:
    refs = tuple(facts.stage_evidence_refs.get(stage.value, ()))
    return StageAssessment(
        stage=stage,
        status=status,
        fault_code=fault,
        cause_code=cause,
        evidence_refs=refs,
        decision_rule_id=_rule_id(rule, rules),
        suspect_surface_templates=surfaces,
    )


def _effective_candidates(facts: CaseFacts, rules: M0Rules) -> tuple[str, ...]:
    identity = float(facts.candidate_utilities.get("identity", facts.corrupt_u))
    distinct = set(facts.effect_distinct_candidate_ids)
    threshold = float(rules["candidate_gain_min"])
    return tuple(
        candidate_id
        for candidate_id, utility in facts.candidate_utilities.items()
        if candidate_id != "identity"
        and candidate_id in distinct
        and float(utility) - identity >= threshold
    )


def _supply_failure(facts: CaseFacts) -> tuple[str, str, tuple[str, ...]]:
    if facts.expressibility_status == "PROVEN_UNAVAILABLE":
        return "OPERATOR_GAP", "CAPABILITY_BACKLOG", ()
    if facts.expressibility_cause == "OBSERVABLE_DERIVATION_PROCEDURE_GAP":
        return (
            "OBSERVABLE_DERIVATION_PROCEDURE_GAP",
            "EDITABLE_M0",
            ("bootstrap_skills.entries/inspect_and_localize.body",),
        )
    if facts.expressibility_cause == "OBSERVABLE_FEATURE_SCHEMA_GAP":
        return "OBSERVABLE_FEATURE_SCHEMA_GAP", "OBSERVATION_CAPABILITY_BACKLOG", ()
    if facts.expressibility_status == "EXPRESSIBILITY_UNKNOWN":
        return "EXPRESSIBILITY_UNKNOWN", "EVIDENCE_BACKLOG", ()
    if facts.expressibility_status == "PROVEN_EXPRESSIBLE" and not facts.capability_skill_exists:
        return (
            "SKILL_LIBRARY_GAP",
            "EDITABLE_M0",
            ("skill_library.entries/{skill_id}",),
        )
    if facts.capability_skill_exists and facts.skill_retrieved:
        if facts.constrained_proposal_succeeds:
            return (
                "PROPOSAL_CONTROL_GAP",
                "EDITABLE_M0",
                ("candidate_policy.proposal_guidance",),
            )
        return (
            "SKILL_CONTENT_GAP",
            "EDITABLE_M0",
            ("skill_library.entries/{skill_id}.body",),
        )
    return "CANDIDATE_SUPPLY_UNKNOWN", "EVIDENCE_BACKLOG", ()


def _build_assessments(facts: CaseFacts, rules: M0Rules) -> tuple[StageAssessment, ...]:
    assessments: list[StageAssessment] = []
    if (
        facts.is_target
        and facts.chosen_candidate_id != "identity"
        and "negative" in facts.chosen_probe_directions
    ):
        assessments.append(
            _assessment(
                facts,
                rules,
                Stage.ELIGIBILITY,
                AssessmentStatus.FAIL,
                rule="selected_probe_direction",
                fault="PROBE_SELECTION_CONTRADICTION",
                cause="PROBE_SELECTION_CONTRADICTION",
                surfaces=("candidate_policy.selection_guidance",),
            )
        )
    elif facts.is_target and facts.damage_d < float(rules["critic_damage_min"]):
        assessments.append(
            _assessment(
                facts,
                rules,
                Stage.ELIGIBILITY,
                AssessmentStatus.FAIL,
                rule="critic_damage_min",
                fault="CRITIC_BLIND",
                cause="CRITIC_BLIND",
            )
        )
    else:
        assessments.append(
            _assessment(
                facts,
                rules,
                Stage.ELIGIBILITY,
                AssessmentStatus.PASS,
                rule="critic_damage_min",
            )
        )

    if not facts.public_evidence_discriminative:
        assessments.append(
            _assessment(
                facts,
                rules,
                Stage.OBSERVATION,
                AssessmentStatus.FAIL,
                rule="public_evidence_discriminative",
                fault="OBSERVATION_GAP",
                cause="OBSERVATION_GAP",
            )
        )
    elif not facts.agent_inspected_evidence:
        assessments.append(
            _assessment(
                facts,
                rules,
                Stage.OBSERVATION,
                AssessmentStatus.FAIL,
                rule="agent_inspected_public_evidence",
                fault="OBSERVATION_PROCEDURE_GAP",
                cause="OBSERVATION_PROCEDURE_GAP",
                surfaces=("bootstrap_skills.entries/inspect_and_localize.body",),
            )
        )
    else:
        assessments.append(
            _assessment(
                facts,
                rules,
                Stage.OBSERVATION,
                AssessmentStatus.PASS,
                rule="agent_inspected_public_evidence",
            )
        )

    if not facts.localization_required:
        localization = _assessment(
            facts,
            rules,
            Stage.LOCALIZATION,
            AssessmentStatus.NOT_APPLICABLE,
            rule="localization_not_applicable",
        )
    elif facts.localization_iou is None:
        localization = _assessment(
            facts,
            rules,
            Stage.LOCALIZATION,
            AssessmentStatus.UNKNOWN,
            rule="localization_iou_interval",
            fault="LOCALIZATION_UNKNOWN",
            cause="LOCALIZATION_UNKNOWN",
        )
    elif facts.localization_iou <= float(rules["localization_fail_iou_max"]):
        localization = _assessment(
            facts,
            rules,
            Stage.LOCALIZATION,
            AssessmentStatus.FAIL,
            rule="localization_iou_interval",
            fault="LOCALIZATION_MISS",
            cause="LOCALIZATION_PROCEDURE_GAP",
            surfaces=("bootstrap_skills.entries/inspect_and_localize.body",),
        )
    elif facts.localization_iou >= float(rules["localization_pass_iou_min"]):
        localization = _assessment(
            facts,
            rules,
            Stage.LOCALIZATION,
            AssessmentStatus.PASS,
            rule="localization_iou_interval",
        )
    else:
        localization = _assessment(
            facts,
            rules,
            Stage.LOCALIZATION,
            AssessmentStatus.UNKNOWN,
            rule="localization_iou_interval",
            fault="LOCALIZATION_UNKNOWN",
            cause="LOCALIZATION_UNKNOWN",
        )
    assessments.append(localization)

    mechanism_status = AssessmentStatus.PASS
    mechanism_fault: str | None = None
    mechanism_cause: str | None = None
    mechanism_surfaces: tuple[str, ...] = ()
    if facts.mechanism_contradiction:
        mechanism_status = AssessmentStatus.FAIL
        mechanism_fault = mechanism_cause = "MECHANISM_AMBIGUITY"
        mechanism_surfaces = ("skill_library.entries/{skill_id}.body",)
    elif not facts.period_diagnostic_pass:
        if facts.mechanism_identified is False or facts.mechanism_identified is None:
            mechanism_status = AssessmentStatus.UNKNOWN
            mechanism_fault = mechanism_cause = "MECHANISM_UNKNOWN"
    assessments.append(
        _assessment(
            facts,
            rules,
            Stage.MECHANISM,
            mechanism_status,
            rule="fixed_probe_discrimination",
            fault=mechanism_fault,
            cause=mechanism_cause,
            surfaces=mechanism_surfaces,
        )
    )

    if not facts.is_target:
        retrieval = _assessment(
            facts,
            rules,
            Stage.RETRIEVAL_POLICY,
            AssessmentStatus.NOT_APPLICABLE,
            rule="non_target_retrieval_not_attributed",
        )
    elif facts.capability_skill_exists and facts.forced_skill_succeeds and not facts.normal_retrieval:
        retrieval = _assessment(
            facts,
            rules,
            Stage.RETRIEVAL_POLICY,
            AssessmentStatus.FAIL,
            rule="forced_skill_replay",
            fault="POLICY_MISROUTING",
            cause="RETRIEVAL_MISS",
            surfaces=("skill_library.entries/{skill_id}.observable_applicability", "retrieval.rules"),
        )
    else:
        retrieval = _assessment(
            facts,
            rules,
            Stage.RETRIEVAL_POLICY,
            AssessmentStatus.PASS,
            rule="forced_skill_replay",
        )
    assessments.append(retrieval)

    effective = _effective_candidates(facts, rules)
    if not facts.is_target:
        supply = _assessment(
            facts,
            rules,
            Stage.CANDIDATE_SUPPLY,
            AssessmentStatus.NOT_APPLICABLE,
            rule="non_target_supply_not_required",
        )
    elif effective:
        supply = _assessment(
            facts,
            rules,
            Stage.CANDIDATE_SUPPLY,
            AssessmentStatus.PASS,
            rule="candidate_gain_min",
        )
    else:
        cause, _actionability, surfaces = _supply_failure(facts)
        supply = _assessment(
            facts,
            rules,
            Stage.CANDIDATE_SUPPLY,
            AssessmentStatus.FAIL
            if cause not in {"EXPRESSIBILITY_UNKNOWN", "CANDIDATE_SUPPLY_UNKNOWN"}
            else AssessmentStatus.UNKNOWN,
            rule="candidate_supply_expressibility_tree",
            fault="CANDIDATE_SUPPLY_GAP" if cause not in {"EXPRESSIBILITY_UNKNOWN", "CANDIDATE_SUPPLY_UNKNOWN"} else cause,
            cause=cause,
            surfaces=surfaces,
        )
    assessments.append(supply)

    identity_u = float(facts.candidate_utilities.get("identity", facts.corrupt_u))
    chosen_u = float(facts.candidate_utilities.get(facts.chosen_candidate_id, identity_u))
    regret = max((float(value) for value in facts.candidate_utilities.values()), default=identity_u) - chosen_u
    if not facts.is_target:
        selection = _assessment(
            facts,
            rules,
            Stage.CANDIDATE_SELECTION,
            AssessmentStatus.NOT_APPLICABLE,
            rule="non_target_selection_not_required",
        )
    elif effective and regret >= float(rules["selection_regret_min"]):
        selection = _assessment(
            facts,
            rules,
            Stage.CANDIDATE_SELECTION,
            AssessmentStatus.FAIL,
            rule="selection_regret_min",
            fault="SELECTION_MISS",
            cause="SELECTION_MISS",
            surfaces=("candidate_policy.selection_guidance",),
        )
    else:
        selection = _assessment(
            facts,
            rules,
            Stage.CANDIDATE_SELECTION,
            AssessmentStatus.PASS,
            rule="selection_regret_min",
        )
    assessments.append(selection)

    if facts.proposed_candidate_exists and not facts.compilation_ok:
        compilation = _assessment(
            facts,
            rules,
            Stage.COMPILATION,
            AssessmentStatus.FAIL,
            rule="canonical_compilation",
            fault="IMPLEMENTATION_MISMATCH",
            cause="IMPLEMENTATION_MISMATCH",
            surfaces=("candidate_policy.proposal_guidance",),
        )
    else:
        compilation = _assessment(
            facts,
            rules,
            Stage.COMPILATION,
            AssessmentStatus.PASS,
            rule="canonical_compilation",
        )
    assessments.append(compilation)

    if facts.compiled_candidate_exists and (
        not facts.execution_ok or not facts.execution_contract_ok
    ):
        execution = _assessment(
            facts,
            rules,
            Stage.EXECUTION,
            AssessmentStatus.FAIL,
            rule="execution_contract",
            fault="EXECUTION_MISMATCH",
            cause="EXECUTION_MISMATCH",
            surfaces=("verification.rules",),
        )
    else:
        execution = _assessment(
            facts,
            rules,
            Stage.EXECUTION,
            AssessmentStatus.PASS,
            rule="execution_contract",
        )
    assessments.append(execution)

    risk_failed = (
        facts.risk_delta_u is not None
        and facts.risk_delta_u < -float(rules["risk_epsilon"])
    ) or not facts.scope_stable
    if risk_failed or facts.over_restoration:
        outcome = _assessment(
            facts,
            rules,
            Stage.OUTCOME_RISK,
            AssessmentStatus.FAIL,
            rule="risk_epsilon_and_scope",
            fault="RISK_GAP",
            cause="RISK_GAP",
            surfaces=("verification.rules", "skill_library.entries/{skill_id}.risk_guards"),
        )
    elif facts.is_target and facts.chosen_gain < float(rules["candidate_gain_min"]):
        outcome = _assessment(
            facts,
            rules,
            Stage.OUTCOME_RISK,
            AssessmentStatus.FAIL,
            rule="candidate_gain_min",
            fault="OUTCOME_GAP",
            cause="OUTCOME_GAP",
            surfaces=("candidate_policy.proposal_guidance",),
        )
    else:
        outcome = _assessment(
            facts,
            rules,
            Stage.OUTCOME_RISK,
            AssessmentStatus.PASS,
            rule="risk_epsilon_and_scope",
        )
    assessments.append(outcome)
    if tuple(assessment.stage.value for assessment in assessments) != STAGE_ORDER:
        raise AssertionError("stage assessment order drifted")
    return tuple(assessments)


def _fold(assessments: tuple[StageAssessment, ...]) -> FaultAttribution:
    router = FaultRouter()
    for assessment in assessments:
        if assessment.status not in {AssessmentStatus.FAIL, AssessmentStatus.UNKNOWN}:
            continue
        cause = assessment.cause_code or assessment.fault_code or "CANDIDATE_SUPPLY_UNKNOWN"
        try:
            actionability = router.allowed_targets(cause).actionability
        except KeyError:
            actionability = "EVIDENCE_BACKLOG"
        return FaultAttribution(
            first_stage=assessment.stage.value,
            fault_code=assessment.fault_code or cause,
            cause_code=cause,
            actionability=actionability,
            suspect_surface_templates=assessment.suspect_surface_templates,
        )
    return FaultAttribution(
        first_stage=None,
        fault_code="NO_ACTIONABLE_FAULT",
        cause_code="NO_ACTIONABLE_FAULT",
        actionability="NONE",
        suspect_surface_templates=(),
    )


def assess_case(
    facts: CaseFacts,
    *,
    rules: M0Rules | None = None,
) -> AssessmentResult:
    rules = rules or load_m0_rules(_RULES_PATH)
    assessments = _build_assessments(facts, rules)
    attribution = _fold(assessments)
    identity_u = float(facts.candidate_utilities.get("identity", facts.corrupt_u))
    chosen_u = float(facts.candidate_utilities.get(facts.chosen_candidate_id, identity_u))
    regret = max((float(value) for value in facts.candidate_utilities.values()), default=identity_u) - chosen_u
    nrr = facts.chosen_gain / facts.damage_d if facts.damage_d > float(rules["critic_damage_min"]) else None
    correct_unavailable_identity = (
        facts.chosen_candidate_id == "identity"
        and facts.expressibility_status == "PROVEN_UNAVAILABLE"
    )
    agent_status = "CORRECT_IDENTITY" if correct_unavailable_identity else "ASSESSED"
    system_status = "OPERATOR_GAP" if facts.expressibility_status == "PROVEN_UNAVAILABLE" else "AVAILABLE_OR_UNKNOWN"
    feedback = CaseFeedback(
        schema_version="case-feedback/1",
        case_id=facts.case_id,
        outcome=OutcomeFeedback(
            valuation_source=facts.valuation_source,
            ingestion_policy_id=facts.ingestion_policy_id,
            clean_u=float(facts.clean_u),
            corrupt_u=float(facts.corrupt_u),
            prepared_u=float(facts.prepared_u),
            damage_d=float(facts.damage_d),
            repair_gain_g=float(facts.chosen_gain),
            nrr=nrr,
            over_restoration=bool(facts.over_restoration),
            selection_regret=regret,
            candidate_utilities=MappingProxyType(
                {str(key): float(value) for key, value in facts.candidate_utilities.items()}
            ),
            chosen_candidate_id=facts.chosen_candidate_id,
            target_window_gain=facts.target_window_gain,
            outside_window_change=facts.outside_window_change,
            counterpart_change=facts.counterpart_change,
            non_target_collateral=facts.non_target_collateral,
            agent_decision_status=agent_status,
            system_capability_status=system_status,
        ),
        mechanism=MechanismFeedback(
            localization_iou=facts.localization_iou,
            observable_features=MappingProxyType(dict(facts.observable_features)),
            r_public_curves=MappingProxyType(dict(facts.public_probe_gains)),
            r_private_curves=MappingProxyType(dict(facts.private_probe_gains)),
            curve_agreement_receipt_ref=facts.curve_agreement_receipt_ref,
            period_diagnostic=MappingProxyType(dict(facts.period_diagnostic)),
            witness_receipt_refs=tuple(facts.witness_receipt_refs),
            implied_mechanism_claims=tuple(facts.implied_mechanism_claims),
            expressibility_status=facts.expressibility_status,
            expressibility_cause=facts.expressibility_cause,
            oracle_family=facts.private_family,
            oracle_affected_indices=tuple(facts.oracle_affected_indices),
        ),
        behavior=BehaviorFeedback(
            decision_trace_ref=facts.decision_trace_ref,
            behavior_signature=MappingProxyType(dict(facts.behavior_signature)),
            inspected_discriminative_evidence=facts.agent_inspected_evidence,
            normal_retrieval=facts.normal_retrieval,
            forced_skill_succeeds=facts.forced_skill_succeeds,
            skill_retrieved=facts.skill_retrieved,
            compilation_status=facts.compilation_status,
            execution_status=facts.execution_status,
        ),
        update_attribution=UpdateAttributionFeedback(
            suspect_surface_templates=attribution.suspect_surface_templates,
            confirmed_surface=None,
            actionability=attribution.actionability,
        ),
        assessments=assessments,
        fault_attribution=attribution,
        private_receipt_refs=tuple(facts.private_receipt_refs),
    )
    return AssessmentResult(
        feedback=feedback,
        assessments=assessments,
        attribution=attribution,
    )


__all__ = [
    "AssessmentResult",
    "AssessmentStatus",
    "CaseFacts",
    "STAGE_ORDER",
    "Stage",
    "assess_case",
]
