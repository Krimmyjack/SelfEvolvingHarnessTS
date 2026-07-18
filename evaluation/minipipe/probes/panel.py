from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
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
_MAPPING_VERSION = "m0-probe-map/1"


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    betas: tuple[float, float, float]
    aggressiveness: tuple[float, float, float]
    mapping_version: str
    required_detected_region: str
    operator_ids: tuple[str, ...]
    implementation_sha: str


def _spec(
    name: str,
    *,
    aggressiveness: tuple[float, float, float],
    required_region: str,
    operator_ids: tuple[str, ...],
    mapping: Mapping[str, object],
) -> ProbeSpec:
    payload = {
        "name": name,
        "betas": list(_BETAS),
        "aggressiveness": list(aggressiveness),
        "mapping_version": _MAPPING_VERSION,
        "required_detected_region": required_region,
        "operator_ids": list(operator_ids),
        "mapping": dict(mapping),
    }
    return ProbeSpec(
        name=name,
        betas=_BETAS,
        aggressiveness=aggressiveness,
        mapping_version=_MAPPING_VERSION,
        required_detected_region=required_region,
        operator_ids=operator_ids,
        implementation_sha=canonical_sha256(payload),
    )


M0_PROBE_SPECS = MappingProxyType(
    {
        "imputation": _spec(
            "imputation",
            aggressiveness=_BETAS,
            required_region="missing_positions",
            operator_ids=("impute_linear",),
            mapping={"rule": "first_ceil_beta_times_detected_missing"},
        ),
        "clipping": _spec(
            "clipping",
            aggressiveness=(1.0 / 8.0, 1.0 / 5.0, 1.0 / 3.0),
            required_region="local_robust_z_points",
            operator_ids=("hampel_filter",),
            mapping={"n_sigmas": [8.0, 5.0, 3.0], "window": 7},
        ),
        "denoising": _spec(
            "denoising",
            aggressiveness=_BETAS,
            required_region="estimated_candidate_region",
            operator_ids=("denoise_median",),
            mapping={"window": 5, "blend": "(1-beta)*raw+beta*denoised"},
        ),
        "level_correction": _spec(
            "level_correction",
            aggressiveness=_BETAS,
            required_region="estimated_excursion_region",
            operator_ids=("repair_level_shift",),
            mapping={"blend": "raw+beta*(canonical_repair-raw)"},
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


def _execute(values: np.ndarray, op: str, params: dict[str, object]) -> np.ndarray:
    result = run_pipeline([(op, params)], values, source="fixed_probe")
    if not result.ok or result.artifact is None:
        raise RuntimeError(f"fixed probe operator failed: {op}")
    output = np.asarray(result.artifact, dtype=np.float64)
    if output.shape != values.shape:
        raise RuntimeError(f"fixed probe operator changed shape: {op}")
    return output


def _apply_probe(
    values: np.ndarray,
    probe_name: str,
    beta: float,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(values, dtype=np.float64).copy()
    features = extract_public_features(raw)
    output = raw.copy()
    if probe_name == "imputation":
        repaired = _execute(raw, "impute_linear", {})
        count = int(math.ceil(beta * len(features.missing_indices)))
        selected = features.missing_indices[:count]
        if selected:
            output[np.asarray(selected, dtype=int)] = repaired[np.asarray(selected, dtype=int)]
    elif probe_name == "clipping":
        thresholds = {0.25: 8.0, 0.50: 5.0, 0.75: 3.0}
        repaired = _execute(
            raw,
            "hampel_filter",
            {"window": 7, "n_sigmas": thresholds[float(beta)]},
        )
        selected = np.asarray(features.outlier_indices, dtype=int)
        if selected.size:
            output[selected] = repaired[selected]
    elif probe_name == "denoising":
        repaired = _execute(raw, "denoise_median", {"window": 5})
        selected = features.region_mask & np.isfinite(raw)
        output[selected] = (1.0 - beta) * raw[selected] + beta * repaired[selected]
    elif probe_name == "level_correction":
        repaired = _execute(raw, "repair_level_shift", {})
        selected = features.level_mask & np.isfinite(raw)
        output[selected] = raw[selected] + beta * (repaired[selected] - raw[selected])
    else:
        raise ValueError(f"unknown fixed probe: {probe_name}")
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
    evaluator_manifest_sha: str
    input_sha: str
    feature_context_sha: str
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
            "schema_version": "public-probe-panel/1",
            "panel_sha": self.panel_sha,
            "spec_bundle_sha": self.spec_bundle_sha,
            "evaluator_manifest_sha": self.evaluator_manifest_sha,
            "input_sha": self.input_sha,
            "feature_context_sha": self.feature_context_sha,
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
            "schema_version": "public-probe-panel/1",
            "spec_bundle_sha": self.spec_bundle_sha,
            "evaluator_manifest_sha": self.rolling_valuator.model_manifest_sha,
            "input_sha": case.public_case_view_sha,
            "feature_context_sha": features.feature_context_sha,
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
            evaluator_manifest_sha=self.rolling_valuator.model_manifest_sha,
            input_sha=case.public_case_view_sha,
            feature_context_sha=features.feature_context_sha,
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
            response_curves=MappingProxyType(curves),
            public_private_curve_agreement=agreement,
            status="OK",
        )


__all__ = [
    "M0_PROBE_SPECS",
    "PeriodDiagnostic",
    "PrivateProbePanelReceipt",
    "ProbePanel",
    "ProbeSpec",
    "PublicProbePanelReceipt",
]
