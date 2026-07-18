from __future__ import annotations

import importlib.metadata
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.contracts.harness import EditManifest, HarnessSnapshot, SkillKind
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest, PreparationStatus
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1
from SelfEvolvingHarnessTS.evaluation.minipipe.config import M0Rules, load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import (
    CaseFeedback,
    CasePurpose,
    PrivateSyntheticCase,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import (
    CoreCorpus,
    build_core_corpus,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault import (
    AssessmentResult,
    CaseFacts,
    assess_case,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.patterns import (
    FailurePatternCard,
    mine_failure_patterns,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.expressibility import (
    ExpressibilityEvaluator,
    implied_mechanism_for_operator,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.panel import (
    M0_PROBE_SPECS,
    PrivateProbePanelReceipt,
    ProbePanel,
    PublicProbePanelReceipt,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    AppliedEditReceipt,
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.lineage import HarnessLineage
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.paired import (
    CaseRunReceipt,
    EditVerdict,
    PairedReplayReport,
    PairedReplayRunner,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.risk_sets import (
    AutomaticRiskSetBuilder,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.rolling_observed import (
    RollingObservedValuator,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (
    MaterializedSnapshot,
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_AGENT_MODEL,
    OPENAI_SDK_VERSION,
)
from SelfEvolvingHarnessTS.runtime.decision_trace import BehaviorSignature, DecisionTrace
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline
from SelfEvolvingHarnessTS.runtime.llm_cache import (
    CachedAgentBackend,
    EffectiveRequestCache,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RULES = Path(__file__).resolve().parent / "config" / "m0_rules.json"
_DEFAULT_H0 = _PACKAGE_ROOT / "methods" / "ttha" / "harness" / "h0"
_SURFACES = _PACKAGE_ROOT / "methods" / "ttha" / "harness" / "harness_surfaces.json"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    _atomic_write(
        path,
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows),
    )


def _manifest_json(manifest: EditManifest) -> dict[str, object]:
    return {
        "edit_id": manifest.edit_id,
        "base_harness_sha": manifest.base_harness_sha,
        "target_pattern_id": manifest.target_pattern_id,
        "target_surface_id": manifest.target_surface_id,
        "operation": manifest.operation.value,
        "surface_precondition": _plain(manifest.surface_precondition),
        "dependency_precondition_shas": _plain(
            manifest.dependency_precondition_shas
        ),
        "minimal_patch": _plain(manifest.minimal_patch),
        "new_value": _plain(manifest.new_value),
        "observable_applicability": _plain(manifest.observable_applicability),
        "predicted_agent_behavior_change": list(
            manifest.predicted_agent_behavior_change
        ),
        "predicted_data_effect": list(manifest.predicted_data_effect),
        "automatically_selected_risk_cases": list(
            manifest.automatically_selected_risk_cases
        ),
        "falsification_condition": list(manifest.falsification_condition),
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _backend_kind(backend: object) -> str:
    name = type(backend).__name__
    if name == "ReplayAgentBackend":
        return "offline-contract-replay/1"
    if name == "RecordingAgentBackend":
        return "offline-contract-policy-recording/1"
    if name == "ContractPolicyBackend":
        return "offline-contract-policy-author/1"
    if name == "AgictoChatCompletionsBackend":
        return "agicto-chat-completions"
    return f"injected:{type(backend).__module__}.{name}"


@dataclass(frozen=True)
class RunContext:
    schema_version: str
    runtime_bundle_sha: str
    backend_kind: str
    relay_base_url: str
    api_style: str
    requested_model_alias: str
    sdk_version: str
    provider_capability_flags: Mapping[str, bool]
    reported_response_models: tuple[str, ...]
    versions: Mapping[str, str]
    valuator_manifest_sha: str
    probe_specs_sha: str
    rules_sha: str
    corpus_sha: str
    platform_flags: Mapping[str, object]
    run_context_sha: str

    @classmethod
    def create(
        cls,
        *,
        snapshot: HarnessSnapshot,
        backend: object,
        valuator: object,
        rules: M0Rules,
        corpus: CoreCorpus,
        probe_specs_sha: str,
        model: str,
        base_url: str,
    ) -> "RunContext":
        payload = {
            "schema_version": "m0-run-context/1",
            "runtime_bundle_sha": snapshot.runtime_bundle_sha,
            "backend_kind": _backend_kind(backend),
            "relay_base_url": base_url,
            "api_style": "chat.completions",
            "requested_model_alias": model,
            "sdk_version": OPENAI_SDK_VERSION,
            "provider_capability_flags": {
                "native_tools": False,
                "structured_outputs": False,
                "reasoning_controls": False,
                "provider_seed": False,
            },
            "reported_response_models": [],
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": _version("torch"),
                "transformers": _version("transformers"),
                "chronos_forecasting": _version("chronos-forecasting"),
            },
            "valuator_manifest_sha": str(valuator.model_manifest_sha),
            "probe_specs_sha": probe_specs_sha,
            "rules_sha": rules.rules_sha,
            "corpus_sha": canonical_sha256(
                {
                    "schema_version": "m0-corpus-instance/1",
                    "case_private_shas": [
                        case.private_sha for case in corpus.all_cases
                    ],
                }
            ),
            "platform_flags": {
                "platform": platform.system().lower(),
                "machine": platform.machine().lower(),
                "device": "cpu",
                "deterministic_algorithms": True,
            },
        }
        return cls(
            schema_version="m0-run-context/1",
            runtime_bundle_sha=snapshot.runtime_bundle_sha,
            backend_kind=str(payload["backend_kind"]),
            relay_base_url=base_url,
            api_style="chat.completions",
            requested_model_alias=model,
            sdk_version=OPENAI_SDK_VERSION,
            provider_capability_flags=MappingProxyType(
                dict(payload["provider_capability_flags"])
            ),
            reported_response_models=(),
            versions=MappingProxyType(dict(payload["versions"])),
            valuator_manifest_sha=str(valuator.model_manifest_sha),
            probe_specs_sha=probe_specs_sha,
            rules_sha=rules.rules_sha,
            corpus_sha=str(payload["corpus_sha"]),
            platform_flags=MappingProxyType(dict(payload["platform_flags"])),
            run_context_sha=canonical_sha256(payload),
        )

    def to_json(self) -> dict[str, object]:
        payload = _plain(self.__dict__)
        return payload


@dataclass(frozen=True)
class _CaseEvaluation:
    case: PrivateSyntheticCase
    feedback: CaseFeedback
    assessment: AssessmentResult
    receipt: CaseRunReceipt
    trace: DecisionTrace
    public_case_json: Mapping[str, object]


def _agent_panel(receipt: PublicProbePanelReceipt) -> dict[str, object]:
    return {
        name: [
            {
                "beta": point.beta,
                "delta": point.r_public,
                "modified_fraction": point.modified_fraction,
                "response_shape": point.response_shape,
                "receipt_sha": point.receipt_sha,
            }
            for point in receipt.response_curves[name]
        ]
        for name in M0_PROBE_SPECS
    }


def _public_curves(receipt: PublicProbePanelReceipt) -> dict[str, object]:
    return {
        name: [point.to_public_dict(round_decimals=receipt.round_decimals) for point in points]
        for name, points in receipt.response_curves.items()
    }


def _private_curves(receipt: PrivateProbePanelReceipt) -> dict[str, object]:
    return {
        name: [
            {
                "probe_id": point.probe_id,
                "beta": point.beta,
                "r_private": point.r_private,
                "modified_fraction": point.modified_fraction,
                "response_shape": point.response_shape,
                "receipt_sha": point.receipt_sha,
            }
            for point in points
        ]
        for name, points in receipt.response_curves.items()
    }


def _localization_iou(
    regions: Sequence[tuple[int, int]],
    affected_indices: Sequence[int],
) -> float | None:
    affected = set(int(index) for index in affected_indices)
    if not affected:
        return None
    inspected = {
        index for start, end in regions for index in range(int(start), int(end))
    }
    union = affected | inspected
    return len(affected & inspected) / len(union) if union else None


def _program_steps(trace: DecisionTrace, candidate_id: str) -> list[tuple[str, dict[str, object]]]:
    raw = trace.candidate_program_steps.get(candidate_id, ())
    return [(str(op), dict(params)) for op, params in raw]


def _target_metrics(
    case: PrivateSyntheticCase,
    prepared: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if not case.oracle_affected_indices:
        return None, None, None
    target = np.asarray(case.oracle_affected_indices, dtype=int)
    prepared = np.asarray(prepared, dtype=np.float64).copy()
    finite_prepared = np.isfinite(prepared)
    if np.any(finite_prepared):
        indices = np.arange(prepared.size)
        prepared = np.interp(
            indices, indices[finite_prepared], prepared[finite_prepared]
        )
    else:
        prepared = np.zeros_like(prepared)
    outside = np.ones(case.clean_context.size, dtype=bool)
    outside[target] = False
    scale = max(float(np.std(case.clean_context)), 1e-8)
    before = case.corrupt_context[target]
    finite = np.isfinite(before)
    before_error = (
        float(np.sqrt(np.mean(np.square(before[finite] - case.clean_context[target][finite]))))
        if np.any(finite)
        else scale
    )
    after_error = float(
        np.sqrt(np.mean(np.square(prepared[target] - case.clean_context[target])))
    )
    outside_error = float(
        np.sqrt(np.mean(np.square(prepared[outside] - case.clean_context[outside])))
    )
    target_gain = (before_error - after_error) / scale
    outside_change = -outside_error / scale
    return target_gain, outside_change, max(0.0, -outside_change)


class _CycleCaseRunner:
    def __init__(
        self,
        *,
        backend: object,
        valuator: object,
        rules: M0Rules,
        run_context_sha: str,
        model: str,
        base_url: str,
    ) -> None:
        self.backend = backend
        self.valuator = valuator
        self.rules = rules
        self.run_context_sha = run_context_sha
        self.model = model
        self.base_url = base_url
        rolling = RollingObservedValuator(
            pipeline=valuator.pipeline,
            origins=tuple(int(value) for value in rules["public_probe_origins"]),
            horizon=int(rules["public_probe_horizon"]),
            min_finite_targets=int(rules["public_probe_min_finite_targets"]),
        )
        self.panel = ProbePanel(
            rolling_valuator=rolling,
            private_valuator=valuator,
            rules=rules,
        )
        self.expressibility = ExpressibilityEvaluator(valuator=valuator)
        self._public_panels: dict[str, PublicProbePanelReceipt] = {}

    def _public_panel(self, case: PrivateSyntheticCase) -> PublicProbePanelReceipt:
        if case.case_id not in self._public_panels:
            self._public_panels[case.case_id] = self.panel.run_public(
                case.to_public_view()
            )
        return self._public_panels[case.case_id]

    def evaluate(
        self,
        snapshot: MaterializedSnapshot,
        case: PrivateSyntheticCase,
    ) -> _CaseEvaluation:
        public_panel = self._public_panel(case)
        agent_panel = _agent_panel(public_panel)
        gateway = LocalPublicToolGateway(
            case.corrupt_context,
            task_kind="forecast",
            fixed_probe_panel=agent_panel,
        )
        public_view = (
            case.to_public_view()
            .with_features(gateway.public_features)
            .with_probe_panel(public_panel)
        )
        core = TTHAAgentCore(
            self.backend,
            gateway,
            model=self.model,
            base_url=self.base_url,
        )
        result, trace = TTHAFastAgent(core).prepare(
            PreparationRequest(
                case.case_id,
                case.corrupt_context,
                forecast_task_spec_v1(
                    horizon=48,
                    downstream_model_class="frozen_tsfm_m0",
                ),
                {},
            ),
            snapshot.snapshot,
            fixed_probe_panel=agent_panel,
        )
        prepared = (
            np.asarray(result.prepared.values, dtype=np.float64)
            if result.prepared is not None
            else case.corrupt_context.copy()
        )
        clean_receipt = self.valuator.evaluate(
            case.clean_context,
            case.clean_future,
            scale_context=case.clean_context,
        )
        corrupt_receipt = self.valuator.evaluate(
            case.corrupt_context,
            case.clean_future,
            scale_context=case.clean_context,
        )
        prepared_receipt = self.valuator.evaluate(
            prepared,
            case.clean_future,
            scale_context=case.clean_context,
        )
        candidate_utilities: dict[str, float] = {
            "identity": corrupt_receipt.utility_u
        }
        effect_distinct: list[str] = []
        supplied_operators: list[str] = []
        for candidate_id in trace.candidate_ids:
            if candidate_id == "identity":
                continue
            steps = _program_steps(trace, candidate_id)
            supplied_operators.extend(op for op, _params in steps)
            execution = run_pipeline(steps, case.corrupt_context, source="private_candidate_replay")
            if not execution.ok or execution.artifact is None:
                candidate_utilities[candidate_id] = corrupt_receipt.utility_u
                continue
            candidate_receipt = self.valuator.evaluate(
                execution.artifact,
                case.clean_future,
                scale_context=case.clean_context,
            )
            candidate_utilities[candidate_id] = candidate_receipt.utility_u
            same = np.array_equal(
                np.asarray(execution.artifact, dtype=np.float64),
                np.asarray(case.corrupt_context, dtype=np.float64),
                equal_nan=True,
            )
            if not same:
                effect_distinct.append(candidate_id)

        private_panel = self.panel.run_private(case, public_receipt=public_panel)
        expression = self.expressibility.evaluate(case)
        required_class = expression.required_transformation_class
        capability_skills = tuple(
            skill
            for skill in snapshot.snapshot.skills
            if skill.skill_kind is SkillKind.CAPABILITY
            and any(
                OPERATOR_METADATA[tool]["category"] == required_class
                for tool in skill.allowed_tools
                if tool in OPERATOR_METADATA
            )
        )
        capability_ids = {skill.skill_id for skill in capability_skills}
        retrieved_capability_ids = capability_ids & set(trace.retrieved_skill_ids)
        chosen_id = trace.chosen_candidate_id or "identity"
        if chosen_id not in candidate_utilities:
            chosen_id = "identity"
        target_gain, outside_change, collateral = _target_metrics(case, prepared)
        behavior = BehaviorSignature.from_trace(trace)
        implied = tuple(
            sorted(
                {
                    claim
                    for operator_id in supplied_operators
                    for claim in (implied_mechanism_for_operator(operator_id),)
                    if claim is not None
                }
            )
        )
        facts = CaseFacts(
            case_id=case.case_id,
            is_target=case.purpose is CasePurpose.TARGET,
            private_family=case.private_family,
            oracle_affected_indices=case.oracle_affected_indices,
            clean_u=clean_receipt.utility_u,
            corrupt_u=corrupt_receipt.utility_u,
            prepared_u=prepared_receipt.utility_u,
            damage_d=clean_receipt.utility_u - corrupt_receipt.utility_u,
            chosen_gain=prepared_receipt.utility_u - corrupt_receipt.utility_u,
            candidate_utilities=MappingProxyType(candidate_utilities),
            effect_distinct_candidate_ids=tuple(effect_distinct),
            chosen_candidate_id=chosen_id,
            public_evidence_discriminative=public_panel.status == "OK",
            agent_inspected_evidence=bool(trace.inspected_regions),
            localization_required=case.purpose is CasePurpose.TARGET,
            localization_iou=_localization_iou(
                trace.inspected_regions, case.oracle_affected_indices
            ),
            mechanism_identified=True,
            public_probe_gains=MappingProxyType(_public_curves(public_panel)),
            private_probe_gains=MappingProxyType(_private_curves(private_panel)),
            period_diagnostic_pass=case.private_family == "period_change",
            period_diagnostic=MappingProxyType(
                public_panel.period_diagnostic.to_public_dict()
            ),
            observable_features=MappingProxyType(dict(gateway.public_features)),
            curve_agreement_receipt_ref=private_panel.panel_sha,
            witness_receipt_refs=(
                expression.observable_witness.receipt_sha,
                expression.oracle_witness.receipt_sha,
            ),
            implied_mechanism_claims=implied,
            expressibility_status=expression.status.value,
            expressibility_cause=expression.cause_code,
            required_transformation_class=required_class,
            observable_witness_succeeded=expression.observable_witness.succeeded,
            oracle_witness_succeeded=expression.oracle_witness.succeeded,
            capability_skill_exists=bool(capability_skills),
            normal_retrieval=bool(retrieved_capability_ids) or not capability_skills,
            skill_retrieved=bool(retrieved_capability_ids),
            forced_skill_succeeds=bool(capability_skills),
            proposed_candidate_exists=any(
                candidate_id != "identity" for candidate_id in trace.candidate_ids
            ),
            compilation_ok=trace.compilation_status == "ok",
            compiled_candidate_exists=any(
                candidate_id != "identity" for candidate_id in trace.candidate_ids
            ),
            execution_ok=trace.execution_status == "ok",
            execution_contract_ok=result.status is not PreparationStatus.FAILED,
            risk_delta_u=(
                prepared_receipt.utility_u - corrupt_receipt.utility_u
                if case.purpose is not CasePurpose.TARGET
                else None
            ),
            scope_stable=(
                len(trace.modified_indices) / case.corrupt_context.size
                <= float(snapshot.snapshot.verification["max_modified_fraction"])
            ),
            over_restoration=(
                prepared_receipt.utility_u - clean_receipt.utility_u
                > float(self.rules["utility_tolerance"])
            ),
            target_window_gain=target_gain,
            outside_window_change=outside_change,
            non_target_collateral=collateral,
            behavior_signature=behavior.normalized_behavior,
            decision_trace_ref=behavior.behavior_signature_sha,
            compilation_status=trace.compilation_status,
            execution_status=trace.execution_status,
            private_receipt_refs=(
                clean_receipt.forecast_sha,
                corrupt_receipt.forecast_sha,
                prepared_receipt.forecast_sha,
                private_panel.panel_sha,
                expression.observable_witness.receipt_sha,
                expression.oracle_witness.receipt_sha,
            ),
        )
        assessment = assess_case(facts, rules=self.rules)
        view = resolve_harness_view(
            snapshot.snapshot,
            gateway.public_features,
            role="fast",
        )
        receipt = CaseRunReceipt(
            case_id=case.case_id,
            utility_u=prepared_receipt.utility_u,
            effective_harness_view_sha=view.effective_harness_view_sha,
            behavior_signature_sha=behavior.behavior_signature_sha,
            eligible_agent_calls=len(trace.public_observation_ids),
            cache_hit_flags=trace.agent_cache_hit_flags,
            retrieved_skill_ids=trace.retrieved_skill_ids,
            supplied_operator_ids=tuple(sorted(set(supplied_operators))),
            supplied_effect_distinct=bool(effect_distinct),
            chosen_candidate_kind=(
                "program" if chosen_id != "identity" else "identity"
            ),
            identity_retained="identity" in trace.candidate_ids,
            modified_fraction=len(trace.modified_indices) / case.corrupt_context.size,
            run_context_sha=self.run_context_sha,
            agent_decision_status=assessment.feedback.outcome.agent_decision_status,
            system_capability_status=assessment.feedback.outcome.system_capability_status,
        )
        return _CaseEvaluation(
            case=case,
            feedback=assessment.feedback,
            assessment=assessment,
            receipt=receipt,
            trace=trace,
            public_case_json=MappingProxyType(public_view.to_json()),
        )

    def run(
        self,
        snapshot: MaterializedSnapshot,
        case: object,
        cache: object,
    ) -> CaseRunReceipt:
        del cache
        if not isinstance(case, PrivateSyntheticCase):
            raise TypeError("M0 case runner requires PrivateSyntheticCase")
        return self.evaluate(snapshot, case).receipt


@dataclass(frozen=True)
class CycleSummary:
    cycle_id: str
    starting_snapshot_sha: str
    ending_snapshot_sha: str
    promoted_edit_ids: tuple[str, ...]
    behavior_shas: tuple[str, ...]
    verdicts: tuple[str, ...]
    public_root: Path
    private_root: Path


@dataclass(frozen=True)
class M0RunResult:
    run_root: Path
    public_root: Path
    private_root: Path
    cycles: tuple[CycleSummary, ...]
    normalized_behavior_shas: tuple[str, ...]
    scientific_verdicts: tuple[str, ...]
    active_snapshot_sha: str
    run_context: RunContext
    lineage: HarnessLineage


class M0CycleRunner:
    def __init__(
        self,
        *,
        run_root: Path,
        backend: object,
        valuator: object,
        rules: M0Rules,
        corpus: CoreCorpus,
        store: SnapshotStore,
        active: MaterializedSnapshot,
        run_context: RunContext,
        lineage: HarnessLineage,
        cache: EffectiveRequestCache,
        model: str,
        base_url: str,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.public_root = self.run_root / "public"
        self.private_root = self.run_root / "private"
        self.public_root.mkdir(parents=True, exist_ok=True)
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.valuator = valuator
        self.rules = rules
        self.corpus = corpus
        self.store = store
        self.active = active
        self.run_context = run_context
        self.lineage = lineage
        self.cache = cache
        self.model = model
        self.base_url = base_url
        self.case_runner = _CycleCaseRunner(
            backend=backend,
            valuator=valuator,
            rules=rules,
            run_context_sha=run_context.run_context_sha,
            model=model,
            base_url=base_url,
        )
        self.controller = EditController(store)
        self.risk_builder = AutomaticRiskSetBuilder()

    @staticmethod
    def _surface_catalog() -> list[dict[str, object]]:
        value = parse_json_document(_SURFACES.read_bytes())
        if not isinstance(value, dict) or not isinstance(value.get("surfaces"), list):
            raise ValueError("invalid Harness surface catalog")
        return [
            {
                "surface_id": surface["surface_template_id"],
                "target_class": surface["target_class"],
                "surface_type": surface["surface_type"],
                "allowed_operations": surface["allowed_operations"],
            }
            for surface in value["surfaces"]
            if isinstance(surface, dict)
        ]

    @staticmethod
    def _priority(
        card: FailurePatternCard,
        feedback_by_id: Mapping[str, CaseFeedback],
    ) -> tuple[float, float, float, str]:
        feedback = [feedback_by_id[case_id] for case_id in card.case_ids]
        median_damage = float(np.median([item.outcome.damage_d for item in feedback]))
        pass_count = sum(
            assessment.status.value == "PASS"
            for item in feedback
            for assessment in item.assessments
        )
        return (-card.support_count, -median_damage, -pass_count, card.pattern_id)

    @staticmethod
    def _applicability_out_of_scope(
        parent: MaterializedSnapshot,
        candidate: MaterializedSnapshot,
        cases: Sequence[PrivateSyntheticCase],
        public_features: Mapping[str, Mapping[str, object]],
    ) -> tuple[str, ...]:
        result = []
        for case in cases:
            features = public_features[case.case_id]
            left = resolve_harness_view(parent.snapshot, features)
            right = resolve_harness_view(candidate.snapshot, features)
            if left.effective_harness_view_sha == right.effective_harness_view_sha:
                result.append(case.case_id)
        return tuple(sorted(result))

    def _write_artifacts(
        self,
        *,
        cycle_id: str,
        evaluations: Sequence[_CaseEvaluation],
        patterns: Sequence[FailurePatternCard],
        manifests: Sequence[EditManifest],
        reports: Sequence[PairedReplayReport],
        operator_backlog: Sequence[Mapping[str, object]],
        incidents: Sequence[Mapping[str, object]],
    ) -> tuple[Path, Path]:
        cycle_root = self.run_root / "cycles" / cycle_id
        public = cycle_root / "public"
        private = cycle_root / "private"
        public.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)
        feedback_rows = [evaluation.feedback.to_private_json() for evaluation in evaluations]
        pattern_payload = {
            "schema_version": "failure-pattern-set/1",
            "cycle_id": cycle_id,
            "patterns": [card.to_json() for card in patterns],
        }
        manifest_payload = {
            "schema_version": "edit-manifest-set/1",
            "cycle_id": cycle_id,
            "edits": [_manifest_json(manifest) for manifest in manifests],
        }
        report_payload = {
            "schema_version": "paired-replay-report-set/1",
            "cycle_id": cycle_id,
            "reports": [report.to_private_json() for report in reports],
        }
        markdown = [f"# Failure patterns: {cycle_id}", ""]
        for card in patterns:
            markdown.extend(
                [
                    f"## {card.pattern_id}",
                    "",
                    f"- support: {card.support_count}",
                    f"- first fault: {card.fault_code} / {card.cause_code}",
                    f"- actionability: {card.actionability}",
                    f"- observable signature: `{canonical_json_bytes(dict(card.observable_signature)).decode('utf-8')}`",
                    "",
                ]
            )
        files: tuple[tuple[Path, bytes], ...] = (
            (private / "case_feedback.jsonl", b"".join(canonical_json_bytes(row) + b"\n" for row in feedback_rows)),
            (public / "failure_patterns.json", canonical_json_bytes(pattern_payload) + b"\n"),
            (public / "failure_patterns.md", "\n".join(markdown).encode("utf-8")),
            (public / "edit_manifest.json", canonical_json_bytes(manifest_payload) + b"\n"),
            (private / "paired_replay_report.json", canonical_json_bytes(report_payload) + b"\n"),
            (private / "operator_capability_backlog.jsonl", b"".join(canonical_json_bytes(row) + b"\n" for row in operator_backlog)),
            (private / "infrastructure_backlog.jsonl", b"".join(canonical_json_bytes(row) + b"\n" for row in incidents)),
        )
        for path, payload in files:
            _atomic_write(path, payload)
            destination_root = self.public_root if path.parent == public else self.private_root
            _atomic_write(destination_root / path.name, payload)
        cases_root = public / "cases"
        for evaluation in evaluations:
            _write_json(cases_root / f"{evaluation.case.case_id}.json", evaluation.public_case_json)
        return public, private

    def _final_regression_sha(
        self,
        snapshot: MaterializedSnapshot,
    ) -> str:
        rows = []
        for case in self.corpus.all_cases:
            receipt = self.case_runner.run(snapshot, case, self.cache)
            rows.append(
                {
                    "case_id": receipt.case_id,
                    "utility_u": receipt.utility_u,
                    "effective_harness_view_sha": receipt.effective_harness_view_sha,
                    "behavior_signature_sha": receipt.behavior_signature_sha,
                    "agent_decision_status": receipt.agent_decision_status,
                    "system_capability_status": receipt.system_capability_status,
                }
            )
        return canonical_sha256(
            {"schema_version": "final-core-regression/1", "rows": rows}
        )

    def run_cycle(self, cycle_index: int) -> CycleSummary:
        cycle_id = f"cycle-{cycle_index:03d}"
        starting = self.active
        evaluations = tuple(
            self.case_runner.evaluate(starting, case)
            for case in self.corpus.all_cases
        )
        feedback_by_id = {
            evaluation.case.case_id: evaluation.feedback
            for evaluation in evaluations
        }
        patterns = mine_failure_patterns(
            tuple(feedback_by_id.values()),
            minimum_support=2,
        )
        operator_backlog = tuple(
            {
                "schema_version": "operator-capability-backlog/1",
                "cycle_id": cycle_id,
                "pattern_id": card.pattern_id,
                "case_ids": list(card.case_ids),
                "cause_code": card.cause_code,
                "agent_decision_status": "CORRECT_IDENTITY_EXPECTED",
                "system_capability_status": "OPERATOR_GAP",
            }
            for card in patterns
            if card.cause_code == "OPERATOR_GAP"
        )
        actionable = sorted(
            (card for card in patterns if card.actionability == "EDITABLE_M0"),
            key=lambda card: self._priority(card, feedback_by_id),
        )[: int(self.rules["max_edits_per_cycle"])]
        public_features = {
            evaluation.case.case_id: evaluation.feedback.mechanism.observable_features
            for evaluation in evaluations
        }
        manifests: list[EditManifest] = []
        reports: list[PairedReplayReport] = []
        applied_by_edit: dict[str, AppliedEditReceipt] = {}
        pattern_by_edit: dict[str, FailurePatternCard] = {}
        incidents: list[Mapping[str, object]] = []
        slow_gateway = LocalPublicToolGateway(
            np.zeros(192, dtype=np.float64), task_kind="forecast"
        )
        slow_core = TTHAAgentCore(
            self.backend,
            slow_gateway,
            model=self.model,
            base_url=self.base_url,
        )
        slow_agent = TTHASlowAgent(slow_core)
        surface_catalog = self._surface_catalog()
        replay = PairedReplayRunner(
            self.case_runner,
            rules=self.rules,
            cache=self.cache,
        )
        cases_by_id = {case.case_id: case for case in self.corpus.all_cases}

        for card in actionable:
            try:
                manifest = slow_agent.propose_edit(
                    card.to_json(), surface_catalog, starting.snapshot
                )
                risk_receipt = self.risk_builder.build(
                    card, self.corpus.all_cases, feedback_by_id
                )
                manifest = replace(
                    manifest,
                    automatically_selected_risk_cases=risk_receipt.case_ids,
                )
                applied = self.controller.apply_to_fork(
                    starting,
                    manifest,
                    confirmed_cause=card.cause_code,
                )
                target_cases = tuple(cases_by_id[case_id] for case_id in card.case_ids)
                risk_cases = tuple(
                    cases_by_id[case_id]
                    for case_id in risk_receipt.case_ids
                    if case_id in cases_by_id
                )
                out_of_scope = self._applicability_out_of_scope(
                    starting,
                    applied.candidate_snapshot,
                    risk_cases,
                    public_features,
                )
                report = replay.run(
                    parent=starting,
                    candidate=applied.candidate_snapshot,
                    applied=applied,
                    manifest=manifest,
                    target_cases=target_cases,
                    risk_cases=risk_cases,
                    out_of_scope_case_ids=out_of_scope,
                    stage_b_cases=self.corpus.all_cases,
                )
            except Exception as exc:
                incidents.append(
                    MappingProxyType(
                        {
                            "schema_version": "cycle-incident/1",
                            "cycle_id": cycle_id,
                            "pattern_id": card.pattern_id,
                            "incident_type": type(exc).__name__,
                        }
                    )
                )
                continue
            manifests.append(manifest)
            reports.append(report)
            applied_by_edit[manifest.edit_id] = applied
            pattern_by_edit[manifest.edit_id] = card
            manifest_sha = canonical_sha256(_manifest_json(manifest))
            self.lineage.append(
                event_kind="EDIT_EVALUATED",
                cycle_id=cycle_id,
                parent_snapshot_sha=starting.runtime_bundle_sha,
                candidate_snapshot_sha=applied.candidate_runtime_bundle_sha,
                active_snapshot_sha=starting.runtime_bundle_sha,
                edit_manifest_sha=manifest_sha,
                paired_replay_report_sha=report.report_sha,
                verdict=report.verdict.value,
                scope_kind="scoped_candidate",
            )

        supported = sorted(
            (report for report in reports if report.verdict is EditVerdict.SUPPORTED_EDIT),
            key=lambda report: (
                -report.facts.target_recovery_fraction,
                -report.facts.median_target_improvement,
                report.edit_id,
            ),
        )
        winner = supported[0] if supported else None
        promoted_ids: tuple[str, ...] = ()
        for report in reports:
            if winner is not None and report.edit_id == winner.edit_id:
                continue
            manifest = next(item for item in manifests if item.edit_id == report.edit_id)
            event_kind = "PENDING" if report.verdict is EditVerdict.SUPPORTED_EDIT else "REJECTED"
            self.lineage.append(
                event_kind=event_kind,
                cycle_id=cycle_id,
                parent_snapshot_sha=starting.runtime_bundle_sha,
                candidate_snapshot_sha=applied_by_edit[report.edit_id].candidate_runtime_bundle_sha,
                active_snapshot_sha=starting.runtime_bundle_sha,
                edit_manifest_sha=canonical_sha256(_manifest_json(manifest)),
                paired_replay_report_sha=report.report_sha,
                verdict=report.verdict.value,
                scope_kind="scoped_candidate",
            )

        if winner is not None:
            manifest = next(item for item in manifests if item.edit_id == winner.edit_id)
            applied = applied_by_edit[winner.edit_id]
            cause = pattern_by_edit[winner.edit_id].cause_code
            self.controller.validate(starting, manifest, confirmed_cause=cause)
            final_regression_sha = self._final_regression_sha(
                applied.candidate_snapshot
            )
            self.store.set_active(applied.candidate_runtime_bundle_sha)
            self.active = applied.candidate_snapshot
            promoted_ids = (winner.edit_id,)
            self.lineage.append(
                event_kind="PROMOTED",
                cycle_id=cycle_id,
                parent_snapshot_sha=starting.runtime_bundle_sha,
                candidate_snapshot_sha=applied.candidate_runtime_bundle_sha,
                active_snapshot_sha=applied.candidate_runtime_bundle_sha,
                edit_manifest_sha=canonical_sha256(_manifest_json(manifest)),
                paired_replay_report_sha=winner.report_sha,
                final_core_regression_sha=final_regression_sha,
                verdict=winner.verdict.value,
                scope_kind="active_scoped_edit",
            )

        public, private = self._write_artifacts(
            cycle_id=cycle_id,
            evaluations=evaluations,
            patterns=patterns,
            manifests=manifests,
            reports=reports,
            operator_backlog=operator_backlog,
            incidents=incidents,
        )
        summary = CycleSummary(
            cycle_id=cycle_id,
            starting_snapshot_sha=starting.runtime_bundle_sha,
            ending_snapshot_sha=self.active.runtime_bundle_sha,
            promoted_edit_ids=promoted_ids,
            behavior_shas=tuple(
                evaluation.receipt.behavior_signature_sha for evaluation in evaluations
            ),
            verdicts=tuple(report.verdict.value for report in reports),
            public_root=public,
            private_root=private,
        )
        _write_json(
            private / "cycle_summary.json",
            {
                "schema_version": "m0-cycle-summary/1",
                "cycle_id": summary.cycle_id,
                "starting_snapshot_sha": summary.starting_snapshot_sha,
                "ending_snapshot_sha": summary.ending_snapshot_sha,
                "promoted_edit_ids": list(summary.promoted_edit_ids),
                "behavior_shas": list(summary.behavior_shas),
                "verdicts": list(summary.verdicts),
            },
        )
        return summary


def run_cycles(
    *,
    cycles: int,
    run_root: Path,
    backend: object,
    valuator: object,
    h0_root: Path = _DEFAULT_H0,
    rules_path: Path = _DEFAULT_RULES,
    model: str = DEFAULT_AGENT_MODEL,
    base_url: str = DEFAULT_AGENT_BASE_URL,
    resume: bool = False,
) -> M0RunResult:
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    run_root = Path(run_root).resolve()
    non_empty = run_root.exists() and any(run_root.iterdir())
    if non_empty and not resume:
        raise FileExistsError("run_root must be absent or empty for a new run")
    if resume and not non_empty:
        raise FileNotFoundError("resume requires a non-empty existing run_root")
    run_root.mkdir(parents=True, exist_ok=True)
    rules = load_m0_rules(Path(rules_path))
    corpus = build_core_corpus(rules)
    h0 = compile_snapshot(Path(h0_root))
    store = SnapshotStore(run_root / "harness_snapshots")
    if resume:
        active_pointer = parse_json_document(store.active_path.read_bytes())
        if not isinstance(active_pointer, dict) or set(active_pointer) != {
            "runtime_bundle_sha"
        }:
            raise ValueError("active Harness pointer is invalid")
        active_sha = str(active_pointer["runtime_bundle_sha"])
        active_root = store.root / active_sha
        active_snapshot = compile_snapshot(active_root)
        if active_snapshot.runtime_bundle_sha != active_sha:
            raise ValueError("active Harness pointer and snapshot lock disagree")
        active = MaterializedSnapshot(active_root, active_snapshot, None)
    else:
        active = store.materialize(h0)
        store.set_active(active.runtime_bundle_sha)
    cache = EffectiveRequestCache(run_root / "agent_cache")
    cached_backend = CachedAgentBackend(backend, cache)
    probe_specs_sha = canonical_sha256(
        {name: spec.implementation_sha for name, spec in M0_PROBE_SPECS.items()}
    )
    context = RunContext.create(
        snapshot=h0,
        backend=backend,
        valuator=valuator,
        rules=rules,
        corpus=corpus,
        probe_specs_sha=probe_specs_sha,
        model=model,
        base_url=base_url,
    )
    lineage = HarnessLineage(run_root / "harness_lineage.jsonl")
    if resume:
        stored_context = parse_json_document(
            (run_root / "private" / "run_context.json").read_bytes()
        )
        if not isinstance(stored_context, dict) or stored_context.get(
            "run_context_sha"
        ) != context.run_context_sha:
            raise ValueError("resume run context does not match the existing run")
        if not lineage.events or not lineage.verify_hash_chain():
            raise ValueError("resume requires a complete valid lineage")
        if lineage.events[0].metadata.get("run_context_sha") != context.run_context_sha:
            raise ValueError("lineage GENESIS is bound to another run context")
    else:
        _write_json(run_root / "private" / "run_context.json", context.to_json())
        lineage.append(
            event_kind="GENESIS",
            cycle_id="genesis",
            parent_snapshot_sha="",
            candidate_snapshot_sha=h0.runtime_bundle_sha,
            active_snapshot_sha=h0.runtime_bundle_sha,
            scope_kind="domain_naive_h0",
            metadata={
                "h0_harness_content_sha": h0.harness_content_sha,
                "h0_runtime_bundle_sha": h0.runtime_bundle_sha,
                "run_context_sha": context.run_context_sha,
            },
        )
    runner = M0CycleRunner(
        run_root=run_root,
        backend=cached_backend,
        valuator=valuator,
        rules=rules,
        corpus=corpus,
        store=store,
        active=active,
        run_context=context,
        lineage=lineage,
        cache=cache,
        model=model,
        base_url=base_url,
    )
    prior_summaries: list[CycleSummary] = []
    if resume:
        for cycle_root in sorted((run_root / "cycles").glob("cycle-*")):
            summary_path = cycle_root / "private" / "cycle_summary.json"
            if not summary_path.is_file():
                raise ValueError("resume found an incomplete cycle directory")
            value = parse_json_document(summary_path.read_bytes())
            if not isinstance(value, dict) or value.get("schema_version") != "m0-cycle-summary/1":
                raise ValueError("resume found an invalid cycle summary")
            prior_summaries.append(
                CycleSummary(
                    cycle_id=str(value["cycle_id"]),
                    starting_snapshot_sha=str(value["starting_snapshot_sha"]),
                    ending_snapshot_sha=str(value["ending_snapshot_sha"]),
                    promoted_edit_ids=tuple(value["promoted_edit_ids"]),
                    behavior_shas=tuple(value["behavior_shas"]),
                    verdicts=tuple(value["verdicts"]),
                    public_root=cycle_root / "public",
                    private_root=cycle_root / "private",
                )
            )
        if len(prior_summaries) > cycles:
            raise ValueError("requested cycle count precedes the resumed run")
        if prior_summaries and prior_summaries[-1].ending_snapshot_sha != active.runtime_bundle_sha:
            raise ValueError("active Harness pointer does not match the last completed cycle")
    new_summaries = tuple(
        runner.run_cycle(index)
        for index in range(len(prior_summaries), cycles)
    )
    summaries = tuple((*prior_summaries, *new_summaries))
    result = M0RunResult(
        run_root=run_root,
        public_root=runner.public_root,
        private_root=runner.private_root,
        cycles=summaries,
        normalized_behavior_shas=tuple(
            behavior_sha
            for summary in summaries
            for behavior_sha in summary.behavior_shas
        ),
        scientific_verdicts=tuple(
            verdict for summary in summaries for verdict in summary.verdicts
        ),
        active_snapshot_sha=runner.active.runtime_bundle_sha,
        run_context=context,
        lineage=lineage,
    )
    _write_json(
        run_root / "private" / "run_summary.json",
        {
            "schema_version": "m0-run-summary/1",
            "cycle_count": len(summaries),
            "active_snapshot_sha": result.active_snapshot_sha,
            "normalized_behavior_shas": list(result.normalized_behavior_shas),
            "scientific_verdicts": list(result.scientific_verdicts),
            "lineage_head_sha": lineage.events[-1].event_sha,
        },
    )
    return result


__all__ = [
    "CycleSummary",
    "HarnessLineage",
    "M0CycleRunner",
    "M0RunResult",
    "RunContext",
    "run_cycles",
]
