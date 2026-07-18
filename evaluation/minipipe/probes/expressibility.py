from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import PrivateSyntheticCase
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

from .features import extract_public_features


_CLASS_PATH = Path(__file__).with_name("transformation_classes.json")
_BASELINE_PATH = Path(__file__).resolve().parents[1] / "baselines" / "fixed_program_baseline_v1.json"
_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "m0_rules.json"


class ExpressibilityStatus(str, Enum):
    PROVEN_EXPRESSIBLE = "PROVEN_EXPRESSIBLE"
    PROVEN_UNAVAILABLE = "PROVEN_UNAVAILABLE"
    EXPRESSIBILITY_UNKNOWN = "EXPRESSIBILITY_UNKNOWN"


@dataclass(frozen=True)
class WitnessReceipt:
    witness_kind: str
    succeeded: bool
    program_sha: str | None
    gain: float | None
    baseline_model_input_sha: str | None
    witness_model_input_sha: str | None
    model_input_distinct: bool | None
    receipt_sha: str


@dataclass(frozen=True)
class ExpressibilityResult:
    status: ExpressibilityStatus
    cause_code: str
    observable_witness: WitnessReceipt
    oracle_witness: WitnessReceipt
    required_transformation_class: str | None
    missing_transformation_class: str | None
    joint_instrument_sha: str | None


def _witness(
    kind: str,
    succeeded: bool,
    *,
    gain: float | None = None,
    program_sha: str | None = None,
    baseline_model_input_sha: str | None = None,
    witness_model_input_sha: str | None = None,
) -> WitnessReceipt:
    distinct = (
        None
        if baseline_model_input_sha is None or witness_model_input_sha is None
        else baseline_model_input_sha != witness_model_input_sha
    )
    payload = {
        "schema_version": "expressibility-witness/2",
        "witness_kind": kind,
        "succeeded": bool(succeeded),
        "program_sha": program_sha,
        "gain": gain,
        "baseline_model_input_sha": baseline_model_input_sha,
        "witness_model_input_sha": witness_model_input_sha,
        "model_input_distinct": distinct,
    }
    return WitnessReceipt(
        witness_kind=kind,
        succeeded=bool(succeeded),
        program_sha=program_sha,
        gain=gain,
        baseline_model_input_sha=baseline_model_input_sha,
        witness_model_input_sha=witness_model_input_sha,
        model_input_distinct=distinct,
        receipt_sha=canonical_sha256(payload),
    )


def evaluate_expressibility(
    *,
    oracle_succeeds: bool,
    observable_succeeds: bool,
    required_feature_is_in_closed_vocabulary: bool,
    required_transformation_class: str | None = None,
    missing_transformation_class: str | None = None,
    complete_class_absence_proof: bool = False,
    joint_instrument_sha: str | None = None,
) -> ExpressibilityResult:
    observable = _witness("observable_parameterized", observable_succeeds)
    oracle = _witness("oracle_parameterized", oracle_succeeds)
    if observable_succeeds:
        status = ExpressibilityStatus.PROVEN_EXPRESSIBLE
        cause = "OBSERVABLE_WITNESS_SUCCEEDED"
        missing = None
    elif complete_class_absence_proof and missing_transformation_class:
        status = ExpressibilityStatus.PROVEN_UNAVAILABLE
        cause = "OPERATOR_CLASS_PROVEN_UNAVAILABLE"
        missing = missing_transformation_class
    elif oracle_succeeds and required_feature_is_in_closed_vocabulary:
        status = ExpressibilityStatus.EXPRESSIBILITY_UNKNOWN
        cause = "OBSERVABLE_DERIVATION_PROCEDURE_GAP"
        missing = None
    elif oracle_succeeds:
        status = ExpressibilityStatus.EXPRESSIBILITY_UNKNOWN
        cause = "OBSERVABLE_FEATURE_SCHEMA_GAP"
        missing = None
    else:
        status = ExpressibilityStatus.EXPRESSIBILITY_UNKNOWN
        cause = "FINITE_WITNESS_SEARCH_INCONCLUSIVE"
        missing = None
    return ExpressibilityResult(
        status=status,
        cause_code=cause,
        observable_witness=observable,
        oracle_witness=oracle,
        required_transformation_class=required_transformation_class,
        missing_transformation_class=missing,
        joint_instrument_sha=joint_instrument_sha,
    )


def _load_object(path: Path) -> dict[str, object]:
    value = parse_json_document(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"instrument must be a JSON object: {path.name}")
    return value


def _program_sha(program: Sequence[Sequence[object]]) -> str:
    return canonical_sha256({"schema_version": "witness-program/1", "steps": list(program)})


def _resolve_program(
    program: Sequence[Sequence[object]],
    *,
    pre_period: int,
) -> list[tuple[str, dict[str, object]]]:
    steps: list[tuple[str, dict[str, object]]] = []
    for entry in program:
        if len(entry) != 2 or not isinstance(entry[0], str) or not isinstance(entry[1], Mapping):
            raise ValueError("invalid witness program entry")
        params = dict(entry[1])
        if params.pop("period_from", None) == "pre_period":
            params["period"] = int(pre_period)
        steps.append((entry[0], params))
    return steps


class ExpressibilityEvaluator:
    """Grader-only positive witnesses with proof-safe unavailable classification."""

    def __init__(
        self,
        *,
        valuator: FrozenChronosValuator | None = None,
        class_path: Path = _CLASS_PATH,
        baseline_path: Path = _BASELINE_PATH,
    ) -> None:
        self.valuator = valuator
        self.classes = _load_object(class_path)
        self.baseline = _load_object(baseline_path)
        self.rules = load_m0_rules(_RULES_PATH)
        actual_categories = {OPERATOR_METADATA[name]["category"] for name in OPERATOR_NAMES}
        declared = set(self.classes.get("complete_operator_categories", []))
        if self.classes.get("declaration_complete") is not True or declared != actual_categories:
            raise ValueError("transformation-class declaration is not complete for the registry")
        required = self.classes.get("required_family_map")
        if not isinstance(required, dict):
            raise ValueError("required family map is absent")
        self.required_family_map = {str(key): str(value) for key, value in required.items()}
        self.joint_instrument_sha = canonical_sha256(
            {
                "schema_version": "joint-expressibility-instrument/1",
                "transformation_classes": self.classes,
                "witness_catalog": self.baseline,
                "corpus_definition": self.rules["corpus"],
            }
        )

    def evaluate(self, case: PrivateSyntheticCase) -> ExpressibilityResult:
        if not isinstance(case, PrivateSyntheticCase):
            raise TypeError("expressibility evaluator accepts PrivateSyntheticCase only")
        family = case.private_family
        required_class = self.required_family_map[family]
        actual_categories = {OPERATOR_METADATA[name]["category"] for name in OPERATOR_NAMES}
        if required_class not in actual_categories:
            return evaluate_expressibility(
                oracle_succeeds=False,
                observable_succeeds=False,
                required_feature_is_in_closed_vocabulary=True,
                required_transformation_class=required_class,
                missing_transformation_class=required_class,
                complete_class_absence_proof=True,
                joint_instrument_sha=self.joint_instrument_sha,
            )
        if self.valuator is None:
            return evaluate_expressibility(
                oracle_succeeds=False,
                observable_succeeds=False,
                required_feature_is_in_closed_vocabulary=True,
                required_transformation_class=required_class,
                joint_instrument_sha=self.joint_instrument_sha,
            )

        baseline = self.valuator.evaluate(
            case.corrupt_context,
            case.clean_future,
            scale_context=case.clean_context,
        )
        feature_context = extract_public_features(case.corrupt_context)
        catalog = self.baseline.get("observable_witnesses")
        if not isinstance(catalog, dict):
            raise ValueError("observable witness catalog is absent")
        programs = catalog.get(family, [])
        best_gain: float | None = None
        best_sha: str | None = None
        best_input_sha: str | None = None
        best_oracle_gain: float | None = None
        best_oracle_sha: str | None = None
        best_oracle_input_sha: str | None = None
        for raw_program in programs:
            if not isinstance(raw_program, list):
                raise ValueError("witness program must be a list")
            steps = _resolve_program(raw_program, pre_period=feature_context.pre_period)
            execution = run_pipeline(steps, case.corrupt_context, source="private_witness")
            if not execution.ok or execution.artifact is None:
                continue
            receipt = self.valuator.evaluate(
                execution.artifact,
                case.clean_future,
                scale_context=case.clean_context,
            )
            gain = receipt.utility_u - baseline.utility_u
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_sha = _program_sha(raw_program)
                best_input_sha = receipt.filled_context_sha
            oracle_artifact = case.corrupt_context.copy()
            affected = np.asarray(case.oracle_affected_indices, dtype=int)
            if affected.size:
                oracle_artifact[affected] = np.asarray(execution.artifact)[affected]
            oracle_receipt = self.valuator.evaluate(
                oracle_artifact,
                case.clean_future,
                scale_context=case.clean_context,
            )
            oracle_gain = oracle_receipt.utility_u - baseline.utility_u
            if best_oracle_gain is None or oracle_gain > best_oracle_gain:
                best_oracle_gain = oracle_gain
                best_oracle_sha = _program_sha(raw_program)
                best_oracle_input_sha = oracle_receipt.filled_context_sha
        threshold = float(self.rules["candidate_gain_min"])
        succeeded = (
            best_input_sha is not None
            and best_input_sha != baseline.filled_context_sha
            and best_gain is not None
            and best_gain >= threshold
        )
        oracle_succeeded = (
            best_oracle_input_sha is not None
            and best_oracle_input_sha != baseline.filled_context_sha
            and best_oracle_gain is not None
            and best_oracle_gain >= threshold
        )
        observable = _witness(
            "observable_parameterized",
            succeeded,
            gain=best_gain,
            program_sha=best_sha,
            baseline_model_input_sha=baseline.filled_context_sha,
            witness_model_input_sha=best_input_sha,
        )
        oracle = _witness(
            "oracle_parameterized",
            oracle_succeeded,
            gain=best_oracle_gain,
            program_sha=best_oracle_sha,
            baseline_model_input_sha=baseline.filled_context_sha,
            witness_model_input_sha=best_oracle_input_sha,
        )
        if succeeded:
            return ExpressibilityResult(
                status=ExpressibilityStatus.PROVEN_EXPRESSIBLE,
                cause_code="OBSERVABLE_WITNESS_SUCCEEDED",
                observable_witness=observable,
                oracle_witness=oracle,
                required_transformation_class=required_class,
                missing_transformation_class=None,
                joint_instrument_sha=self.joint_instrument_sha,
            )
        cause = (
            "OBSERVABLE_DERIVATION_PROCEDURE_GAP"
            if oracle_succeeded
            else "FINITE_WITNESS_SEARCH_INCONCLUSIVE"
        )
        return ExpressibilityResult(
            status=ExpressibilityStatus.EXPRESSIBILITY_UNKNOWN,
            cause_code=cause,
            observable_witness=observable,
            oracle_witness=oracle,
            required_transformation_class=required_class,
            missing_transformation_class=None,
            joint_instrument_sha=self.joint_instrument_sha,
        )


def implied_mechanism_for_operator(operator_id: str) -> str | None:
    metadata = OPERATOR_METADATA.get(operator_id)
    if metadata is None:
        return None
    reverse = _load_object(_CLASS_PATH).get("operator_category_to_implied_family")
    if not isinstance(reverse, dict):
        return None
    value = reverse.get(metadata["category"])
    return str(value) if value is not None else None


__all__ = [
    "ExpressibilityEvaluator",
    "ExpressibilityResult",
    "ExpressibilityStatus",
    "WitnessReceipt",
    "evaluate_expressibility",
    "implied_mechanism_for_operator",
]
