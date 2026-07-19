from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.contracts.observables import OUTLIER_Z_THRESHOLD
from SelfEvolvingHarnessTS.evaluation.minipipe.config import M0Rules
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import (
    PrivateSyntheticCase,
    PublicCaseView,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.rolling_observed import (
    RollingObservedValuator,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

from .features import PublicFeatureExtraction, extract_public_features


_BETAS = (0.25, 0.50, 0.75)
_MAPPING_VERSION = "m0-probe-map/3"
PROBE_INSTRUMENT_EPOCH = "probe-instrument/3"


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    betas: tuple[float, float, float]
    aggressiveness: tuple[float, float, float]
    mapping_version: str
    instrument_epoch: str
    required_detected_region: str
    operator_ids: tuple[str, ...]
    parameterization: tuple[Mapping[str, object], ...]
    implementation_sha: str


def _spec(
    name: str,
    *,
    aggressiveness: tuple[float, float, float],
    required_region: str,
    operator_ids: tuple[str, ...],
    parameterization: tuple[Mapping[str, object], ...],
) -> ProbeSpec:
    if len(parameterization) != len(_BETAS):
        raise ValueError("one canonical program template is required per probe beta")
    payload = {
        "name": name,
        "betas": list(_BETAS),
        "aggressiveness": list(aggressiveness),
        "mapping_version": _MAPPING_VERSION,
        "instrument_epoch": PROBE_INSTRUMENT_EPOCH,
        "required_detected_region": required_region,
        "operator_ids": list(operator_ids),
        "parameterization": [dict(item) for item in parameterization],
    }
    return ProbeSpec(
        name=name,
        betas=_BETAS,
        aggressiveness=aggressiveness,
        mapping_version=_MAPPING_VERSION,
        instrument_epoch=PROBE_INSTRUMENT_EPOCH,
        required_detected_region=required_region,
        operator_ids=operator_ids,
        parameterization=tuple(MappingProxyType(dict(item)) for item in parameterization),
        implementation_sha=canonical_sha256(payload),
    )


M0_PROBE_SPECS = MappingProxyType(
    {
        "imputation": _spec(
            "imputation",
            aggressiveness=(1.0 / 3.0, 2.0 / 3.0, 1.0),
            required_region="missing_positions",
            operator_ids=("impute_linear",),
            parameterization=tuple(
                {
                    "steps": [
                        {
                            "op": "impute_linear",
                            "params": {"strength": strength},
                        }
                    ]
                }
                for strength in (1.0 / 3.0, 2.0 / 3.0, 1.0)
            ),
        ),
        "clipping": _spec(
            "clipping",
            aggressiveness=(1.0 / 8.0, 1.0 / 5.0, 1.0 / 3.0),
            required_region="local_robust_z_points",
            operator_ids=("hampel_filter",),
            parameterization=tuple(
                {
                    "steps": [
                        {
                            "op": "hampel_filter",
                            "params": {
                                "window": 7,
                                "n_sigmas": n_sigmas,
                                "global_z_min": OUTLIER_Z_THRESHOLD,
                            },
                        }
                    ]
                }
                for n_sigmas in (8.0, 5.0, 3.0)
            ),
        ),
        "denoising": _spec(
            "denoising",
            aggressiveness=(1.0 / 3.0, 2.0 / 3.0, 1.0),
            required_region="entire_observed_context",
            operator_ids=("denoise_median",),
            parameterization=tuple(
                {
                    "steps": [
                        {
                            "op": "denoise_median",
                            "params": {"window": 5, "strength": strength},
                        }
                    ]
                }
                for strength in (1.0 / 3.0, 2.0 / 3.0, 1.0)
            ),
        ),
        "level_correction": _spec(
            "level_correction",
            aggressiveness=(1.0 / 3.0, 2.0 / 3.0, 1.0),
            required_region="estimated_excursion_region",
            operator_ids=("repair_level_shift",),
            parameterization=tuple(
                {
                    "steps": [
                        {
                            "op": "repair_level_shift",
                            "params": {
                                "region_start_fraction": {
                                    "feature": "estimated_region_start_fraction",
                                    "scale": 1.0,
                                },
                                "region_end_fraction": {
                                    "feature": "estimated_region_end_fraction",
                                    "scale": 1.0,
                                },
                                "estimated_offset": {
                                    "feature": "estimated_level_offset",
                                    "scale": strength,
                                },
                            },
                        }
                    ]
                }
                for strength in (1.0 / 3.0, 2.0 / 3.0, 1.0)
            ),
        ),
    }
)


def _changed_fraction(before: np.ndarray, after: np.ndarray) -> float:
    before_finite = np.isfinite(before)
    after_finite = np.isfinite(after)
    changed = before_finite != after_finite
    both = before_finite & after_finite
    changed[both] |= np.abs(before[both] - after[both]) > 1e-12
    return float(np.mean(changed))


def _plain_contract(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_contract(nested) for key, nested in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_contract(nested) for nested in value]
    return value


def _probe_arm_index(spec: ProbeSpec, beta: float) -> int:
    try:
        return spec.betas.index(float(beta))
    except ValueError as exc:
        raise ValueError("probe beta is outside the fixed dose schedule") from exc


def _resolve_parameter(value: object, features: Mapping[str, object]) -> object:
    if not isinstance(value, Mapping):
        return value
    if set(value) != {"feature", "scale"}:
        raise ValueError("probe parameter binding must contain feature and scale only")
    feature = value["feature"]
    scale = value["scale"]
    if not isinstance(feature, str) or feature not in features:
        raise ValueError(f"probe parameter references unavailable feature: {feature!r}")
    feature_value = features[feature]
    if (
        isinstance(feature_value, bool)
        or not isinstance(feature_value, (int, float))
        or isinstance(scale, bool)
        or not isinstance(scale, (int, float))
    ):
        raise ValueError("probe numeric binding requires numeric feature and scale")
    return float(feature_value) * float(scale)


def materialize_probe_program(
    probe_name: str,
    beta: float,
    features: Mapping[str, object],
) -> tuple[tuple[str, dict[str, object]], ...]:
    """Instantiate one fixed arm as a canonical runtime Program.

    Every value comes either from the versioned template or a declared public
    feature binding.  An unavailable public region produces explicit identity
    (an empty step tuple), never a hidden post-execution array edit.
    """

    if probe_name not in M0_PROBE_SPECS:
        raise ValueError(f"unknown fixed probe: {probe_name}")
    spec = M0_PROBE_SPECS[probe_name]
    template = spec.parameterization[_probe_arm_index(spec, beta)]
    raw_steps = template.get("steps")
    if not isinstance(raw_steps, list | tuple):
        raise ValueError("probe program template must contain steps")
    steps: list[tuple[str, dict[str, object]]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping) or set(raw_step) != {"op", "params"}:
            raise ValueError("probe step template must contain op and params only")
        op = raw_step["op"]
        params = raw_step["params"]
        if not isinstance(op, str) or not isinstance(params, Mapping):
            raise ValueError("probe step template has invalid op or params")
        steps.append(
            (
                op,
                {
                    str(key): _resolve_parameter(value, features)
                    for key, value in params.items()
                },
            )
        )
    if probe_name == "level_correction":
        start = float(features.get("estimated_region_start_fraction", 0.0))
        end = float(features.get("estimated_region_end_fraction", 0.0))
        offset = float(features.get("estimated_level_offset", 0.0))
        if end <= start or offset == 0.0:
            return ()
    return tuple(steps)


def public_probe_contracts(
    features: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize the fixed arm templates, optionally with current-context programs."""

    probes: dict[str, object] = {}
    for name, spec in M0_PROBE_SPECS.items():
        arms: list[dict[str, object]] = []
        for index, beta in enumerate(spec.betas):
            arm: dict[str, object] = {
                "beta": beta,
                "aggressiveness": spec.aggressiveness[index],
                "program_template": _plain_contract(spec.parameterization[index]),
            }
            if features is not None:
                steps = materialize_probe_program(name, beta, features)
                arm["current_context_program_steps"] = [
                    {"op": op, "params": params} for op, params in steps
                ]
                arm["current_context_applicable"] = bool(steps)
            arms.append(arm)
        probes[name] = {
            "required_detected_region": spec.required_detected_region,
            "operator_ids": list(spec.operator_ids),
            "arms": arms,
            "implementation_sha": spec.implementation_sha,
        }
    payload = {
        "schema_version": "fixed-probe-contracts/1",
        "mapping_version": _MAPPING_VERSION,
        "instrument_epoch": PROBE_INSTRUMENT_EPOCH,
        "probes": probes,
    }
    payload["contracts_sha"] = canonical_sha256(payload)
    return payload


def _execute_program(
    values: np.ndarray,
    steps: tuple[tuple[str, dict[str, object]], ...],
) -> np.ndarray:
    if not steps:
        return np.asarray(values, dtype=np.float64).copy()
    result = run_pipeline(steps, values, source="fixed_probe")
    if not result.ok or result.artifact is None:
        raise RuntimeError(f"fixed probe program failed: {steps!r}")
    output = np.asarray(result.artifact, dtype=np.float64)
    if output.shape != values.shape:
        raise RuntimeError("fixed probe program changed shape")
    return output


def _apply_probe(
    values: np.ndarray,
    probe_name: str,
    beta: float,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(values, dtype=np.float64).copy()
    features = extract_public_features(raw)
    steps = materialize_probe_program(probe_name, float(beta), features.mapping)
    output = _execute_program(raw, steps)
    return output, _changed_fraction(raw, output)


def _response_shape(
    responses: tuple[float | None, ...],
    *,
    gain_min: float,
    margin_min: float,
) -> str:
    if any(response is None for response in responses):
        return "unknown"
    values = tuple(float(response) for response in responses if response is not None)
    if max(values[:-1]) >= gain_min and values[-1] < max(values[:-1]) - margin_min:
        return "overdose_collapse"
    if max(values) >= gain_min:
        return "positive"
    if min(values) <= -gain_min:
        return "harmful"
    return "flat"


@dataclass(frozen=True)
class PeriodDiagnostic:
    pre_period: int
    post_period: int
    period_change_score: float
    acf_spectral_consistency: float
    reliability: float
    evidence_status: str
    repair_available: bool
    receipt_sha: str

    @classmethod
    def from_features(cls, features: PublicFeatureExtraction) -> "PeriodDiagnostic":
        payload = {
            "schema_version": "period-diagnostic/2",
            "pre_period": features.pre_period,
            "post_period": features.post_period,
            "period_change_score": features.mapping["period_change_score"],
            "acf_spectral_consistency": features.acf_spectral_consistency,
            "reliability": features.period_reliability,
            "evidence_status": features.period_evidence_status,
            "repair_available": False,
        }
        return cls(
            pre_period=features.pre_period,
            post_period=features.post_period,
            period_change_score=float(features.mapping["period_change_score"]),
            acf_spectral_consistency=features.acf_spectral_consistency,
            reliability=features.period_reliability,
            evidence_status=features.period_evidence_status,
            repair_available=False,
            receipt_sha=canonical_sha256(payload),
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "pre_period": self.pre_period,
            "post_period": self.post_period,
            "period_change_score": self.period_change_score,
            "acf_spectral_consistency": self.acf_spectral_consistency,
            "reliability": self.reliability,
            "evidence_status": self.evidence_status,
            "repair_available": self.repair_available,
            "receipt_sha": self.receipt_sha,
        }


@dataclass(frozen=True)
class PublicProbePoint:
    probe_id: str
    beta: float
    r_public: float | None
    modified_fraction: float
    response_shape: str
    receipt_sha: str

    def to_public_dict(self, *, round_decimals: int) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "beta": self.beta,
            "r_public": (
                None if self.r_public is None else round(self.r_public, round_decimals)
            ),
            "modified_fraction": round(self.modified_fraction, round_decimals),
            "response_shape": self.response_shape,
            "receipt_sha": self.receipt_sha,
        }


@dataclass(frozen=True)
class PublicProbePanelReceipt:
    panel_sha: str
    spec_bundle_sha: str
    instrument_epoch: str
    evaluator_manifest_sha: str
    input_sha: str
    feature_context_sha: str
    probe_contracts: Mapping[str, object]
    period_diagnostic: PeriodDiagnostic
    response_curves: Mapping[str, tuple[PublicProbePoint, ...]]
    status: str
    round_decimals: int

    @property
    def directions(self) -> dict[str, str]:
        return {
            name: points[0].response_shape
            for name, points in self.response_curves.items()
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": "public-probe-panel/2",
            "panel_sha": self.panel_sha,
            "spec_bundle_sha": self.spec_bundle_sha,
            "instrument_epoch": self.instrument_epoch,
            "evaluator_manifest_sha": self.evaluator_manifest_sha,
            "input_sha": self.input_sha,
            "feature_context_sha": self.feature_context_sha,
            "probe_contracts": _plain_contract(self.probe_contracts),
            "period_diagnostic": self.period_diagnostic.to_public_dict(),
            "points": [
                point.to_public_dict(round_decimals=self.round_decimals)
                for name in M0_PROBE_SPECS
                for point in self.response_curves[name]
            ],
            "status": self.status,
        }


@dataclass(frozen=True)
class PrivateProbePoint:
    probe_id: str
    beta: float
    r_private: float
    modified_fraction: float
    response_shape: str
    receipt_sha: str


@dataclass(frozen=True)
class PrivateProbePanelReceipt:
    panel_sha: str
    case_id: str
    input_private_sha: str
    evaluator_manifest_sha: str
    instrument_epoch: str
    response_curves: Mapping[str, tuple[PrivateProbePoint, ...]]
    public_private_curve_agreement: float | None
    status: str


class ProbePanel:
    def __init__(
        self,
        *,
        rolling_valuator: RollingObservedValuator,
        rules: M0Rules | Mapping[str, object],
        private_valuator: FrozenChronosValuator | None = None,
    ) -> None:
        self.rolling_valuator = rolling_valuator
        self.private_valuator = private_valuator
        self.rules = rules
        if tuple(float(value) for value in rules["probe_betas"]) != _BETAS:
            raise ValueError("M0 probe beta schedule differs from the fixed mapping")
        if tuple(self.rolling_valuator.origins) != tuple(
            int(value) for value in rules["public_probe_origins"]
        ):
            raise ValueError("rolling origins differ from the fixed ProbeAPI schedule")
        if self.rolling_valuator.horizon != int(rules["public_probe_horizon"]):
            raise ValueError("rolling horizon differs from the fixed ProbeAPI schedule")
        if self.rolling_valuator.min_finite_targets != int(
            rules["public_probe_min_finite_targets"]
        ):
            raise ValueError("rolling finite-target rule differs from the fixed ProbeAPI schedule")
        self.spec_bundle_sha = canonical_sha256(
            {
                name: spec.implementation_sha
                for name, spec in M0_PROBE_SPECS.items()
            }
        )

    def run_public(self, case: PublicCaseView) -> PublicProbePanelReceipt:
        if not isinstance(case, PublicCaseView):
            raise TypeError("public probe panel accepts PublicCaseView only")
        features = extract_public_features(case.values, task_kind=case.task_kind)
        probe_contracts = public_probe_contracts(features.mapping)
        baseline = self.rolling_valuator.evaluate(case.values)
        curves: dict[str, tuple[PublicProbePoint, ...]] = {}
        all_ok = baseline.mean_public_utility is not None
        for name, spec in M0_PROBE_SPECS.items():
            raw_points: list[PublicProbePoint] = []
            responses: list[float | None] = []
            for beta in spec.betas:
                arm = self.rolling_valuator.evaluate(
                    case.values,
                    prefix_transform=lambda prefix, _origin, n=name, b=beta: _apply_probe(
                        prefix,
                        n,
                        b,
                    )[0],
                )
                transformed, modified_fraction = _apply_probe(case.values, name, beta)
                del transformed
                response = (
                    None
                    if baseline.mean_public_utility is None
                    or arm.mean_public_utility is None
                    else arm.mean_public_utility - baseline.mean_public_utility
                )
                responses.append(response)
                point_payload = {
                    "schema_version": "public-probe-point/1",
                    "probe_id": name,
                    "beta": beta,
                    "r_public": response,
                    "modified_fraction": modified_fraction,
                    "spec_sha": spec.implementation_sha,
                    "input_sha": case.public_case_view_sha,
                }
                raw_points.append(
                    PublicProbePoint(
                        probe_id=name,
                        beta=beta,
                        r_public=response,
                        modified_fraction=modified_fraction,
                        response_shape="pending",
                        receipt_sha=canonical_sha256(point_payload),
                    )
                )
                all_ok &= response is not None
            shape = _response_shape(
                tuple(responses),
                gain_min=float(self.rules["probe_gain_min"]),
                margin_min=float(self.rules["probe_margin_min"]),
            )
            curves[name] = tuple(replace(point, response_shape=shape) for point in raw_points)
        period = PeriodDiagnostic.from_features(features)
        payload = {
            "schema_version": "public-probe-panel/2",
            "instrument_epoch": PROBE_INSTRUMENT_EPOCH,
            "spec_bundle_sha": self.spec_bundle_sha,
            "evaluator_manifest_sha": self.rolling_valuator.model_manifest_sha,
            "input_sha": case.public_case_view_sha,
            "feature_context_sha": features.feature_context_sha,
            "probe_contracts_sha": probe_contracts["contracts_sha"],
            "period_receipt_sha": period.receipt_sha,
            "point_shas": [
                point.receipt_sha
                for name in M0_PROBE_SPECS
                for point in curves[name]
            ],
            "status": "OK" if all_ok else "UNKNOWN",
        }
        return PublicProbePanelReceipt(
            panel_sha=canonical_sha256(payload),
            spec_bundle_sha=self.spec_bundle_sha,
            instrument_epoch=PROBE_INSTRUMENT_EPOCH,
            evaluator_manifest_sha=self.rolling_valuator.model_manifest_sha,
            input_sha=case.public_case_view_sha,
            feature_context_sha=features.feature_context_sha,
            probe_contracts=MappingProxyType(probe_contracts),
            period_diagnostic=period,
            response_curves=MappingProxyType(curves),
            status="OK" if all_ok else "UNKNOWN",
            round_decimals=int(self.rules["public_probe_round_decimals"]),
        )

    def run_private(
        self,
        case: PrivateSyntheticCase,
        *,
        public_receipt: PublicProbePanelReceipt | None = None,
    ) -> PrivateProbePanelReceipt:
        if self.private_valuator is None:
            raise ValueError("private probe panel requires a private valuator")
        if not isinstance(case, PrivateSyntheticCase):
            raise TypeError("private probe panel accepts PrivateSyntheticCase only")
        baseline = self.private_valuator.evaluate(
            case.corrupt_context,
            case.clean_future,
            scale_context=case.clean_context,
        )
        curves: dict[str, tuple[PrivateProbePoint, ...]] = {}
        paired_responses: list[tuple[float, float]] = []
        for name, spec in M0_PROBE_SPECS.items():
            responses: list[float] = []
            arms: list[tuple[float, float, str]] = []
            for beta in spec.betas:
                transformed, modified_fraction = _apply_probe(
                    case.corrupt_context,
                    name,
                    beta,
                )
                arm = self.private_valuator.evaluate(
                    transformed,
                    case.clean_future,
                    scale_context=case.clean_context,
                )
                response = arm.utility_u - baseline.utility_u
                responses.append(response)
                receipt_sha = canonical_sha256(
                    {
                        "schema_version": "private-probe-point/1",
                        "case_private_sha": case.private_sha,
                        "probe_id": name,
                        "beta": beta,
                        "r_private": response,
                        "modified_fraction": modified_fraction,
                        "spec_sha": spec.implementation_sha,
                    }
                )
                arms.append((response, modified_fraction, receipt_sha))
            shape = _response_shape(
                tuple(responses),
                gain_min=float(self.rules["probe_gain_min"]),
                margin_min=float(self.rules["probe_margin_min"]),
            )
            curves[name] = tuple(
                PrivateProbePoint(
                    probe_id=name,
                    beta=beta,
                    r_private=response,
                    modified_fraction=modified_fraction,
                    response_shape=shape,
                    receipt_sha=receipt_sha,
                )
                for beta, (response, modified_fraction, receipt_sha) in zip(spec.betas, arms)
            )
            if public_receipt is not None:
                for public, private in zip(public_receipt.response_curves[name], curves[name]):
                    if public.r_public is not None:
                        paired_responses.append((public.r_public, private.r_private))
        agreement: float | None = None
        if paired_responses:
            agreement = float(
                np.mean(
                    [
                        np.sign(public) == np.sign(private)
                        for public, private in paired_responses
                    ]
                )
            )
        payload = {
            "schema_version": "private-probe-panel/1",
            "instrument_epoch": PROBE_INSTRUMENT_EPOCH,
            "case_private_sha": case.private_sha,
            "evaluator_manifest_sha": self.private_valuator.model_manifest_sha,
            "point_shas": [
                point.receipt_sha
                for name in M0_PROBE_SPECS
                for point in curves[name]
            ],
            "public_private_curve_agreement": agreement,
        }
        return PrivateProbePanelReceipt(
            panel_sha=canonical_sha256(payload),
            case_id=case.case_id,
            input_private_sha=case.private_sha,
            evaluator_manifest_sha=self.private_valuator.model_manifest_sha,
            instrument_epoch=PROBE_INSTRUMENT_EPOCH,
            response_curves=MappingProxyType(curves),
            public_private_curve_agreement=agreement,
            status="OK",
        )


__all__ = [
    "M0_PROBE_SPECS",
    "PROBE_INSTRUMENT_EPOCH",
    "PeriodDiagnostic",
    "PrivateProbePanelReceipt",
    "ProbePanel",
    "ProbeSpec",
    "PublicProbePanelReceipt",
    "materialize_probe_program",
    "public_probe_contracts",
]
