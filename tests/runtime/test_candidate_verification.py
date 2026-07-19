import numpy as np
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate
from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document
from SelfEvolvingHarnessTS.methods.ttha.agent_core import validate_local_schema


ROOT = Path(__file__).resolve().parents[2]


def _candidate(candidate_id: str, op: str, params=None) -> Candidate:
    return Candidate.program_candidate(
        candidate_id,
        Program.from_steps([(op, params or {})], source="agent"),
        source="agent",
    )


def test_identity_and_valid_imputation_receipts_are_public_mechanical_facts() -> None:
    raw = np.asarray([1.0, np.nan, 3.0])
    identity = verify_candidate(
        Candidate.identity(), raw, allowed_operators=("impute_linear",)
    )
    repaired = verify_candidate(
        _candidate("fill-gap", "impute_linear"),
        raw,
        allowed_operators=("impute_linear",),
    )

    assert identity.receipt.status == "valid"
    assert identity.receipt.effect_equivalent_to_identity is True
    assert repaired.receipt.status == "valid"
    assert repaired.receipt.execution_ok is True
    assert repaired.receipt.finite_output is True
    assert repaired.receipt.modified_fraction == 1 / 3

    forbidden = {
        "utility_u",
        "loss_j",
        "candidate_utilities",
        "clean_context",
        "injection_type",
        "selection_regret",
    }
    assert forbidden.isdisjoint(repaired.receipt.to_dict())
    schema = parse_json_document(
        (
            ROOT
            / "contracts"
            / "schemas"
            / "candidate_verification_receipt_v1.json"
        ).read_bytes()
    )
    validate_local_schema(repaired.receipt.to_dict(), schema)


def test_noop_is_visible_warning_not_silently_removed() -> None:
    raw = np.asarray([1.0, 2.0, 3.0])
    artifact = verify_candidate(
        _candidate("noop-impute", "impute_linear"),
        raw,
        allowed_operators=("impute_linear",),
    )
    assert artifact.selectable is True
    assert artifact.receipt.status == "warning"
    assert artifact.receipt.warnings == ("EFFECT_EQUIVALENT_TO_IDENTITY",)


def test_illegal_overflow_and_execution_failure_are_typed_rejections() -> None:
    raw = np.asarray([1.0, 2.0, 3.0, 4.0])
    illegal = verify_candidate(
        _candidate("illegal", "standardize"),
        raw,
        allowed_operators=("impute_linear",),
    )
    overflow = verify_candidate(
        _candidate("overflow", "winsorize", {"limits": 0.25}),
        raw,
        allowed_operators=("winsorize",),
        maximum_modified_fraction=0.1,
    )
    failure = verify_candidate(
        _candidate("failure", "not_registered"),
        raw,
        allowed_operators=("not_registered",),
    )

    assert illegal.receipt.rejection_code == "OPERATOR_NOT_ALLOWED"
    assert overflow.receipt.rejection_code == "MODIFICATION_FRACTION_EXCEEDED"
    assert failure.receipt.rejection_code == "EXECUTION_FAILED"
    assert not illegal.selectable and not overflow.selectable and not failure.selectable
