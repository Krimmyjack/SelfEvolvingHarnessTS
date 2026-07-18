from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import ValuationReceipt
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    AgentRequest,
    AgentResponse,
    ReplayAgentBackend,
)
from SelfEvolvingHarnessTS.runtime.llm_cache import CacheKey


FIXTURE_SOURCE = "contract_policy_not_openai"
_SPARSE_LOCALIZATION_MARKER = "sparse_localization_subregions/v1"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stage(stage: str, payload: Mapping[str, object]) -> AgentResponse:
    return AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "stage_result",
            "stage": stage,
            "payload": _plain(payload),
        },
        raw_response={
            "id": f"fixture-{canonical_sha256({'stage': stage, 'payload': payload})[:16]}",
            "model": "offline-contract-policy/1",
        },
        provider_metadata={
            "fixture_source": FIXTURE_SOURCE,
            "returned_model": "offline-contract-policy/1",
        },
    )


def _no_proposal(reason_code: str) -> AgentResponse:
    return AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "no_proposal",
            "stage": "edit",
            "reason_code": reason_code,
        },
        raw_response={
            "id": f"fixture-no-proposal-{reason_code}",
            "model": "offline-contract-policy/1",
        },
        provider_metadata={
            "fixture_source": FIXTURE_SOURCE,
            "returned_model": "offline-contract-policy/1",
        },
    )


def _public_input(request: AgentRequest) -> Mapping[str, object]:
    if len(request.messages) < 2:
        raise ValueError("contract-policy request has no public user payload")
    parsed = parse_json_document(str(request.messages[1]["content"]).encode("utf-8"))
    if not isinstance(parsed, dict) or parsed.get("schema_version") != "public-agent-input/1":
        raise ValueError("contract-policy request has invalid public input")
    public_input = parsed.get("public_input")
    if not isinstance(public_input, dict):
        raise ValueError("contract-policy request public_input must be an object")
    return public_input


def _resolved_harness(request: AgentRequest) -> Mapping[str, object]:
    system = str(request.messages[0]["content"])
    marker = "Resolved Harness: "
    if marker not in system:
        raise ValueError("contract-policy request has no resolved Harness")
    parsed = parse_json_document(system.split(marker, 1)[1].encode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("resolved Harness must be an object")
    return parsed


def _surface_contract(
    public: Mapping[str, object],
    *,
    surface_id: str | None = None,
    surface_template_id: str | None = None,
) -> Mapping[str, object]:
    catalog = public.get("writable_surface_catalog")
    if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes, bytearray)):
        raise ValueError("contract-policy request has no surface catalog")
    matches = [
        item
        for item in catalog
        if isinstance(item, Mapping)
        and (surface_id is None or item.get("surface_id") == surface_id)
        and (
            surface_template_id is None
            or item.get("surface_template_id") == surface_template_id
        )
    ]
    if len(matches) != 1:
        raise ValueError("surface catalog does not resolve one writable contract")
    return matches[0]


def _bin_is_positive(value: object) -> bool:
    return value in {"very_low", "low", "medium", "high"}


def _family_from_signature(signature: Mapping[str, object]) -> str | None:
    if _bin_is_positive(signature.get("missing_fraction")):
        return "missing"
    if signature.get("clipping_probe_direction") in {"positive", "overdose_collapse"}:
        return "impulsive_outlier"
    if signature.get("level_probe_direction") in {"positive", "overdose_collapse"}:
        return "level_shift"
    if signature.get("period_repair_available") is False and signature.get(
        "period_change_score"
    ) in {"medium", "high"}:
        return "period_change"
    if signature.get("estimated_region_end_fraction") not in {"high", None}:
        if signature.get("local_robust_z_peak") in {"medium", "high"}:
            return "impulsive_outlier"
    if signature.get("level_excursion_score") in {"medium", "high"}:
        return "level_shift"
    return None


def _skill_contract(family: str) -> tuple[str, str, Mapping[str, object], str]:
    if family == "missing":
        return (
            "observed_missing_repair_v1",
            "impute_linear",
            {"feature": "missing_fraction", "op": ">", "value": 0.0},
            "Use bounded linear imputation when deployment-visible missingness is present.",
        )
    if family == "impulsive_outlier":
        return (
            "local_peak_repair_v1",
            "hampel_filter",
            {
                "all": [
                    {"feature": "missing_fraction", "op": "==", "value": 0.0},
                    {"feature": "local_robust_z_peak", "op": ">=", "value": 4.0},
                    {
                        "feature": "estimated_region_end_fraction",
                        "op": "<",
                        "value": 0.95,
                    },
                ]
            },
            "Use a bounded Hampel repair for sparse public robust-z peaks away from the boundary.",
        )
    if family == "level_shift":
        return (
            "bounded_level_repair_v1",
            "repair_level_shift",
            {
                "all": [
                    {"feature": "missing_fraction", "op": "==", "value": 0.0},
                    {"feature": "level_excursion_score", "op": ">=", "value": 3.0},
                    {
                        "feature": "estimated_region_end_fraction",
                        "op": "<",
                        "value": 0.95,
                    },
                ]
            },
            "Use bounded structural level correction only for a closed public excursion.",
        )
    raise ValueError(f"no M0 capability skill is defined for {family}")


class ContractPolicyBackend:
    """Public-request-only fixture author; never used as scientific evidence."""

    backend_identity = "offline-contract-policy-author/1"

    def __init__(self) -> None:
        self.call_count = 0

    def clone(self) -> "ContractPolicyBackend":
        return ContractPolicyBackend()

    def complete(self, request: AgentRequest) -> AgentResponse:
        self.call_count += 1
        public = _public_input(request)
        if request.stage == "inspect":
            harness = _resolved_harness(request)
            features = public.get("features", {})
            if not isinstance(features, Mapping):
                features = {}
            start = float(features.get("estimated_region_start_fraction", 0.0))
            end = float(features.get("estimated_region_end_fraction", 1.0))
            start = min(max(start, 0.0), 0.999999)
            end = min(max(end, start + 1.0 / 192.0), 1.0)
            inspected = [[start, end]]
            skills = harness.get("skills", [])
            localization_body = next(
                (
                    str(skill.get("body", ""))
                    for skill in skills
                    if isinstance(skill, Mapping)
                    and skill.get("skill_id") == "inspect_and_localize"
                ),
                "",
            )
            if (
                _SPARSE_LOCALIZATION_MARKER in localization_body
                and float(features.get("local_robust_z_peak", 0.0)) >= 4.0
                and end - start >= 0.15
            ):
                # This fixture has only deployment-visible aggregate bounds.  A
                # patched procedural skill turns a broad sparse-peak span into
                # four narrow hypotheses instead of claiming the whole span.
                hypothesis_count = 2 if end - start < 0.22 else 4
                centers = np.linspace(
                    start,
                    end,
                    num=hypothesis_count,
                    endpoint=True,
                )
                half_width = 0.75 / 192.0
                inspected = [
                    [
                        max(0.0, float(center) - half_width),
                        min(1.0, float(center) + half_width),
                    ]
                    for center in centers
                ]
            return _stage(
                "inspect",
                {
                    "inspected_region_fractions": inspected,
                    "requested_public_tools": [],
                    "uncertainty": "low" if (end - start) < 0.75 else "high",
                },
            )
        if request.stage == "propose":
            harness = _resolved_harness(request)
            skills = harness.get("skills", [])
            capability = next(
                (
                    skill
                    for skill in skills
                    if isinstance(skill, Mapping)
                    and skill.get("skill_kind") == "capability"
                    and isinstance(skill.get("allowed_tools"), list)
                    and skill["allowed_tools"]
                ),
                None,
            )
            candidates: list[dict[str, object]] = []
            if capability is not None:
                operator_id = str(capability["allowed_tools"][0])
                candidates.append(
                    {
                        "candidate_id": "agent-0",
                        "steps": [{"op": operator_id, "params": {}}],
                    }
                )
            return _stage("propose", {"candidates": candidates})
        if request.stage == "select":
            candidates = public.get("candidates", [])
            chosen = "identity"
            for candidate in candidates if isinstance(candidates, list) else []:
                if isinstance(candidate, Mapping) and candidate.get("kind") == "program":
                    chosen = str(candidate["candidate_id"])
                    break
            return _stage(
                "select",
                {
                    "chosen_candidate_id": chosen,
                    "verification_actions": [
                        "scope_checked",
                        "identity_retained",
                    ],
                },
            )
        if request.stage == "edit":
            card = public.get("failure_pattern_card", {})
            if not isinstance(card, Mapping):
                raise ValueError("contract policy requires a failure pattern card")
            cause_code = str(card.get("cause_code", ""))
            pattern_id = str(card["pattern_id"])
            base_sha = str(public["base_harness_sha"])

            if cause_code == "LOCALIZATION_PROCEDURE_GAP":
                target = "bootstrap_skills.entries/inspect_and_localize.body"
                surface = _surface_contract(public, surface_id=target)
                manifest = {
                    "edit_id": f"edit-localization-{pattern_id[-8:]}",
                    "base_harness_sha": base_sha,
                    "target_pattern_id": pattern_id,
                    "target_surface_id": target,
                    "operation": "PATCH",
                    "surface_precondition": _plain(surface["surface_precondition"]),
                    "dependency_precondition_shas": _plain(
                        surface["dependency_precondition_shas"]
                    ),
                    "minimal_patch": {
                        "value": (
                            "Inspect public missingness, robust local deviation, level "
                            "continuity, and period-consistency evidence. When a high "
                            "robust-z signal spans a broad low-density region, preserve "
                            "multiple narrow localization hypotheses instead of merging "
                            "them into one interval. Procedure marker: "
                            f"{_SPARSE_LOCALIZATION_MARKER}."
                        )
                    },
                    "observable_applicability": None,
                    "predicted_agent_behavior_change": [
                        "localization_iou>=0.30",
                        "identity_retained",
                    ],
                    "predicted_data_effect": [
                        "localization evidence becomes actionable"
                    ],
                    "automatically_selected_risk_cases": [],
                    "falsification_condition": [
                        "predicted localization change absent",
                        "target utility gain absent",
                    ],
                }
                return _stage("edit", {"edit_manifest": manifest})

            if cause_code == "RETRIEVAL_MISS":
                harness = _resolved_harness(request)
                capability = next(
                    (
                        skill
                        for skill in harness.get("skills", [])
                        if isinstance(skill, Mapping)
                        and skill.get("skill_kind") == "capability"
                    ),
                    None,
                )
                if capability is None:
                    return _no_proposal("insufficient_public_evidence")
                skill_id = str(capability["skill_id"])
                tools = capability.get("allowed_tools", [])
                if not isinstance(tools, list) or not tools:
                    return _no_proposal("no_authorized_minimal_edit")
                operator_id = str(tools[0])
                target = (
                    f"skill_library.entries/{skill_id}.observable_applicability"
                )
                surface = _surface_contract(public, surface_id=target)
                applicability = {
                    "feature": "missing_fraction",
                    "op": ">",
                    "value": 0.0,
                }
                manifest = {
                    "edit_id": f"edit-retrieval-{pattern_id[-8:]}",
                    "base_harness_sha": base_sha,
                    "target_pattern_id": pattern_id,
                    "target_surface_id": target,
                    "operation": "PATCH",
                    "surface_precondition": _plain(surface["surface_precondition"]),
                    "dependency_precondition_shas": _plain(
                        surface["dependency_precondition_shas"]
                    ),
                    "minimal_patch": {"value": applicability},
                    "observable_applicability": applicability,
                    "predicted_agent_behavior_change": [
                        f"retrieve_skill:{skill_id}",
                        f"supply_operator:{operator_id}",
                        "supply_effect_distinct",
                        "identity_retained",
                    ],
                    "predicted_data_effect": ["target utility improves"],
                    "automatically_selected_risk_cases": [],
                    "falsification_condition": [
                        "skill remains unretrieved",
                        "target gain absent",
                    ],
                }
                return _stage("edit", {"edit_manifest": manifest})

            if cause_code != "SKILL_LIBRARY_GAP":
                return _no_proposal("no_authorized_minimal_edit")
            signature = card.get("observable_signature", {})
            if not isinstance(signature, Mapping):
                raise ValueError("failure pattern has no observable signature")
            family = _family_from_signature(signature)
            if family in {None, "period_change"}:
                raise ValueError("pattern has no authorable capability edit")
            skill_id, operator_id, applicability, body = _skill_contract(family)
            surface = _surface_contract(
                public,
                surface_template_id="skill_library.entries/{skill_id}",
            )
            manifest = {
                "edit_id": f"edit-{family.replace('_', '-')}-{pattern_id[-8:]}",
                "base_harness_sha": base_sha,
                "target_pattern_id": pattern_id,
                "target_surface_id": f"skill_library.entries/{skill_id}",
                "operation": "ADD",
                "surface_precondition": _plain(surface["surface_precondition"]),
                "dependency_precondition_shas": _plain(
                    surface["dependency_precondition_shas"]
                ),
                "new_value": {
                    "schema_version": "skill-entry/1",
                    "skill_id": skill_id,
                    "skill_kind": "capability",
                    "revision": 1,
                    "body": body,
                    "observable_applicability": applicability,
                    "allowed_tools": [operator_id],
                    "risk_guards": {
                        "max_modified_fraction": 0.25,
                        "preserve_outside_candidate_region": True,
                    },
                },
                "observable_applicability": applicability,
                "predicted_agent_behavior_change": [
                    f"retrieve_skill:{skill_id}",
                    f"supply_operator:{operator_id}",
                    "supply_effect_distinct",
                    "choose_candidate_kind:program",
                    "identity_retained",
                    "scope_modified_fraction<=0.25",
                ],
                "predicted_data_effect": ["target_utility_improves"],
                "automatically_selected_risk_cases": [],
                "falsification_condition": [
                    "predicted_behavior_absent",
                    "target_gain_absent",
                    "risk_regression",
                ],
            }
            return _stage("edit", {"edit_manifest": manifest})
        raise ValueError(f"unsupported contract-policy stage: {request.stage}")


def _response_payload(response: AgentResponse) -> dict[str, object]:
    return {
        "transport_ok": response.transport_ok,
        "raw_response": _plain(response.raw_response),
        "assistant_text": response.assistant_text,
        "parsed_envelope": _plain(response.parsed_envelope),
        "parse_status": response.parse_status,
        "finish_reason": response.finish_reason,
        "provider_metadata": _plain(response.provider_metadata),
    }


def _response_from_payload(value: object) -> AgentResponse:
    if not isinstance(value, dict):
        raise ValueError("offline replay response must be an object")
    parsed = value.get("parsed_envelope")
    if parsed is not None and not isinstance(parsed, dict):
        raise ValueError("offline parsed envelope must be an object or null")
    return AgentResponse(
        transport_ok=bool(value["transport_ok"]),
        raw_response=value["raw_response"],
        assistant_text=str(value["assistant_text"]),
        parsed_envelope=parsed,
        parse_status=str(value["parse_status"]),
        finish_reason=str(value.get("finish_reason", "")),
        provider_metadata=value.get("provider_metadata", {}),
    )


class RecordingAgentBackend:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self._rows: dict[str, dict[str, object]] = {}
        self.call_count = 0

    @property
    def rows(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            MappingProxyType(self._rows[key]) for key in sorted(self._rows)
        )

    def complete(self, request: AgentRequest) -> AgentResponse:
        response = self.delegate.complete(request)
        self.call_count += 1
        semantic_hash = request.semantic_request_hash()
        row = {
            "schema_version": "offline-agent-replay-row/1",
            "fixture_source": FIXTURE_SOURCE,
            "cache_key": CacheKey.from_request(request).to_dict(),
            "semantic_request_hash": semantic_hash,
            "response": _response_payload(response),
        }
        row["row_sha"] = canonical_sha256(row)
        existing = self._rows.get(semantic_hash)
        if existing is not None and existing != row:
            raise ValueError("contract policy produced a non-deterministic response")
        self._rows[semantic_hash] = row
        return response

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in self.rows)
        path.write_bytes(payload)
        return path


def load_replay_backend(path: Path) -> ReplayAgentBackend:
    mapped: dict[str, AgentResponse] = {}
    for line_number, raw in enumerate(Path(path).read_bytes().splitlines(), start=1):
        if not raw.strip():
            continue
        value = parse_json_document(raw)
        if not isinstance(value, dict):
            raise ValueError(f"offline replay row {line_number} is not an object")
        row_sha = value.pop("row_sha", None)
        if row_sha != canonical_sha256(value):
            raise ValueError(f"offline replay row {line_number} SHA mismatch")
        if value.get("fixture_source") != FIXTURE_SOURCE:
            raise ValueError("offline replay source identity mismatch")
        semantic_hash = str(value["semantic_request_hash"])
        key = value.get("cache_key")
        if not isinstance(key, dict) or key.get("semantic_request_hash") != semantic_hash:
            raise ValueError("offline replay CacheKey mismatch")
        response = _response_from_payload(value["response"])
        if semantic_hash in mapped:
            raise ValueError("duplicate semantic request in offline replay")
        mapped[semantic_hash] = response
    if not mapped:
        raise ValueError("offline replay tape is empty")
    return ReplayAgentBackend(mapped)


class _LastValuePipeline:
    def predict_quantiles(self, contexts, *, prediction_length, quantile_levels):
        import torch

        rows = []
        for context in contexts:
            values = np.asarray(context, dtype=np.float64).reshape(-1)
            observed = values[np.isfinite(values)]
            last = float(observed[-1]) if observed.size else 0.0
            rows.append(np.full(prediction_length, last, dtype=np.float32))
        mean = torch.as_tensor(np.stack(rows), dtype=torch.float32)
        return mean[:, :, None], mean


class DeterministicContractValuator:
    """Private fixture score: distance to the paired clean scale context."""

    valuation_source = "DETERMINISTIC_CONTRACT_FIXTURE"
    ingestion_policy_id = "fixture_mask_aware_plus_missing_penalty/v2"
    model_manifest_sha = canonical_sha256(
        {
            "schema_version": "deterministic-contract-valuator/2",
            "valuation_source": valuation_source,
            "ingestion_policy_id": ingestion_policy_id,
        }
    )

    def __init__(self) -> None:
        self.pipeline = _LastValuePipeline()

    @staticmethod
    def _array_sha(values: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()

    def evaluate(
        self,
        context: object,
        clean_future: object,
        *,
        scale_context: object,
    ) -> ValuationReceipt:
        raw = np.asarray(context, dtype=np.float64).reshape(-1)
        clean = np.asarray(scale_context, dtype=np.float64).reshape(-1)
        future = np.asarray(clean_future, dtype=np.float64).reshape(-1)
        if raw.shape != clean.shape or raw.size != 192 or future.size != 48:
            raise ValueError("deterministic contract valuator requires 192 + 48 values")
        finite = np.isfinite(raw)
        if not np.any(finite):
            filled = np.zeros_like(raw)
        else:
            indices = np.arange(raw.size)
            filled = np.interp(indices, indices[finite], raw[finite])
        scale = max(float(np.std(clean)), 1e-8)
        # Missingness is itself an observable preparation defect.  The private
        # fixture score therefore keeps a deterministic missing-value penalty
        # instead of making identity and linear imputation artificially equal.
        loss = float(
            np.sqrt(np.mean(np.square(filled - clean))) / scale
            + np.count_nonzero(~finite) / raw.size
        )
        if not math.isfinite(loss):
            raise ValueError("deterministic contract loss is non-finite")
        forecast_sha = canonical_sha256(
            {
                "kind": "contract-distance-mask-aware",
                "input": self._array_sha(raw),
            }
        )
        missing = int(np.count_nonzero(~finite))
        return ValuationReceipt(
            valuation_source=self.valuation_source,
            ingestion_policy_id=self.ingestion_policy_id,
            model_manifest_sha=self.model_manifest_sha,
            input_sha=self._array_sha(raw),
            filled_context_sha=self._array_sha(raw),
            future_sha=self._array_sha(future),
            forecast_sha=forecast_sha,
            loss_j=loss,
            utility_u=-loss,
            missing_count=missing,
            missing_fraction=missing / raw.size,
            fill_fraction=0.0,
            scale=scale,
            prediction_length=future.size,
            status="OK",
        )


__all__ = [
    "ContractPolicyBackend",
    "DeterministicContractValuator",
    "FIXTURE_SOURCE",
    "RecordingAgentBackend",
    "load_replay_backend",
]
