from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document
from SelfEvolvingHarnessTS.contracts.evidence import (
    CapabilityTemplate,
    EvidenceDisposition,
    PolicyInterventionEvidence,
    SignedEvidenceLedger,
)


@dataclass(frozen=True)
class M0EvidenceMigration:
    capability: CapabilityTemplate
    evidence: PolicyInterventionEvidence
    ledger: SignedEvidenceLedger


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an array")
    return value


def migrate_m0_level_shift_ledger(path: str | Path) -> M0EvidenceMigration:
    """Read the frozen M0 ledger into E0 contracts without rewriting release files."""

    ledger_path = Path(path)
    source = _object(parse_json_document(ledger_path.read_bytes()), field="M0 capability ledger")
    if source.get("schema_version") != "capability-ledger/1":
        raise ValueError("unsupported M0 capability ledger")
    release_id = source.get("release_id")
    if release_id != "m0-h2":
        raise ValueError("level-shift migration only supports the frozen m0-h2 release")
    capabilities = _sequence(source.get("capabilities"), field="capabilities")
    if len(capabilities) != 1:
        raise ValueError("m0-h2 must contain exactly one released capability")
    legacy = _object(capabilities[0], field="legacy capability")
    capability_id = legacy.get("capability_id")
    if capability_id != "level_shift_contrast_candidate":
        raise ValueError("m0-h2 level-shift capability is missing")

    skill_path = (
        ledger_path.parent
        / "harness"
        / "skills"
        / "learned"
        / f"{capability_id}.json"
    )
    skill = _object(parse_json_document(skill_path.read_bytes()), field="released capability skill")
    allowed_tools = tuple(str(value) for value in _sequence(skill.get("allowed_tools"), field="allowed_tools"))

    template = CapabilityTemplate(
        capability_id=str(capability_id),
        applicability=_object(skill.get("observable_applicability"), field="applicability"),
        program_schema={
            "kind": "typed_program",
            "allowed_operators": allowed_tools,
            "max_steps": 1,
        },
        binding_schema={
            "scope": "observation_interval",
            "required_public_bindings": (
                "estimated_region_start_fraction",
                "estimated_region_end_fraction",
                "estimated_level_offset",
            ),
        },
        risk_guards=_object(skill.get("risk_guards"), field="risk_guards"),
        task_context={
            "task_types": tuple(legacy.get("task_scope", ())),
            "data_regime": "synthetic",
            "evidence_origin": "legacy_migration",
        },
        consumer_context={
            "downstream_model_family": "Chronos",
            "binding": "fixed:m0",
            "scope": "forecasting_only",
        },
        operator_registry_context={
            "release_id": release_id,
            "allowed_operator_ids": allowed_tools,
        },
    )

    heldout = _object(legacy.get("heldout_reuse"), field="heldout_reuse")
    evidence = PolicyInterventionEvidence(
        evidence_id="m0-h2-level-shift-policy-replay",
        capability_id=str(capability_id),
        baseline_policy="identity_control",
        intervention_policy="m0-h2-released-harness",
        cohort_definition={
            "kind": "legacy_release_aggregate",
            "selected_targets": heldout.get("selected_targets"),
            "out_of_scope_case_count": heldout.get("out_of_scope_case_count"),
            "dataset_membership_available": False,
        },
        outcomes={
            "created": legacy.get("created"),
            "revised": legacy.get("revised"),
            "heldout_reuse": heldout,
        },
        risk={
            "risk_regressions": legacy.get("risk_regressions"),
            "false_trigger_count": legacy.get("false_trigger_count"),
            "known_boundaries": legacy.get("known_boundaries"),
            "instrument_dependencies": legacy.get("instrument_dependencies"),
        },
        causal_scope="legacy_release_aggregate_only:no_dataset_level_causal_claim",
        source_refs=(
            "artifacts/releases/m0-h2/capability_ledger.json#level_shift_contrast_candidate",
        ),
        receipt_shas=(),
        tags=("synthetic", "forecasting", "legacy_migration"),
    )
    migrated_ledger = SignedEvidenceLedger("m0-h2-e0-migration")
    migrated_ledger.append(
        evidence,
        EvidenceDisposition.UNRESOLVED,
        recorded_by="legacy-migration:v1",
    )
    return M0EvidenceMigration(template, evidence, migrated_ledger)


__all__ = ["M0EvidenceMigration", "migrate_m0_level_shift_ledger"]
