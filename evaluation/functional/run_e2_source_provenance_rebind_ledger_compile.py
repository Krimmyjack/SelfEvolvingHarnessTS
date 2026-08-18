"""Compile the frozen fresh-promotion report into a zero-fit signed ledger."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# contracts/__init__.py imports the ndarray-bearing method contract.  The compiler
# never calls it; allow this zero-fit evidence-only command in metadata environments.
if importlib.util.find_spec("numpy") is None:
    sys.modules["numpy"] = types.ModuleType("numpy")

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.contracts.evidence import (
    CapabilityTemplate,
    EvidenceDisposition,
    PolicyInterventionEvidence,
    SignedEvidenceLedger,
)


SCHEMA_VERSION = "e2-source-provenance-rebind-signed-ledger/1"
CAPABILITY_ID = "provenance_key_rebind_v0"
LEDGER_ID = "e2-provenance-key-rebind-fresh-promotion-ledger-v0"
RECORDED_BY = "run_e2_source_provenance_rebind_ledger_compile"
REPORT_SHA256 = "16117985e701151bf80fc6c5ad0e67c60ebcfc074b12ca7f4da8b1e62aa21354"
PLAN_SHA256 = "a59b32e97c184a482656423dfd17314c9dec3921744b772033b7880f01ec2137"
REPORT_RELATIVE_PATH = "artifacts/functional/e2/source_provenance_rebind_fresh_promotion_report.json"
PLAN_RELATIVE_PATH = "artifacts/functional/e2/source_provenance_rebind_fresh_promotion_plan.json"
OUTPUT_RELATIVE_PATH = "artifacts/functional/e2/source_provenance_rebind_signed_ledger.json"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def _read_pinned(path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, object]]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError(f"pinned artifact SHA mismatch: {path}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"pinned artifact is not an object: {path}")
    return payload, {"path": str(path), "expected_sha256": expected_sha,
                     "actual_sha256": actual, "hash_matches": True}


def _capability() -> tuple[CapabilityTemplate, dict[str, object]]:
    payload: dict[str, object] = {
        "capability_id": CAPABILITY_ID,
        "applicability": {
            "required": ["trusted_unique_input_keys", "trusted_unique_target_keys",
                         "equal_input_target_key_sets", "positional_key_mismatch"],
            "ineligible_when": "positional binding already intact",
        },
        "program_schema": {
            "operation": "key_only_whole_TargetRow_rejoin",
            "payload_or_outcome_reads": False,
            "target_key_moves_with_payload": True,
        },
        "binding_schema": {
            "immutable_key_fields": ["dataset_sha", "series_uid", "anchor", "horizon"],
            "key_visibility": "deployment_visible_trusted_provenance",
        },
        "risk_guards": {
            "intact_binding": "NO_OP",
            "duplicate_key": "ABSTAIN", "missing_key": "ABSTAIN",
            "cross_set_key": "ABSTAIN",
        },
        "task_context": {"task": "forecasting", "horizon": 48,
                         "evidence_scope": "injected_structural_only"},
        "consumer_context": {"class": "sklearn.linear_model.Ridge", "alpha": 1.0,
                             "fit_intercept": True, "solver": "svd",
                             "metric": "per-series normalized MAE", "fixed": True},
        "operator_registry_context": {
            "registration_status": "candidate_not_registered",
            "deployment_class": "MINIMAL_INTEGRITY_ADAPTER_CANDIDATE_NOT_MEMORY_CAPABILITY",
        },
    }
    template = CapabilityTemplate(**payload)
    return template, payload


def _policy_evidence(row: dict[str, Any], index: int) -> PolicyInterventionEvidence:
    dataset_id = str(row["dataset_id"])
    paired = row["paired_eval_rows"]
    harmed = sum(bool(item["harmed"]) for item in paired)
    return PolicyInterventionEvidence(
        evidence_id=f"{CAPABILITY_ID}:{dataset_id}:fresh-promotion-v0",
        capability_id=CAPABILITY_ID,
        baseline_policy="positional_incumbent",
        intervention_policy="key_rebind_repaired",
        cohort_definition={"dataset_id": dataset_id, "fresh_selected_uids": True,
                           "scientific_unit": "fresh_dataset_level",
                           "eval_series_count": len(paired)},
        outcomes={"mean_gain_incumbent_minus_repaired": row["mean_gain_incumbent_minus_repaired"],
                  "median_gain_incumbent_minus_repaired": row["median_gain_incumbent_minus_repaired"],
                  "positive_gain_count": row["positive_gain_count"],
                  "dataset_gate_pass": row["dataset_gate"]["pass"]},
        risk={"harm_definition": "gain < -0.005", "harmed_count": harmed,
              "eval_series_count": len(paired)},
        causal_scope="injected_structural_target_binding_dataset_level",
        source_refs=(f"{REPORT_RELATIVE_PATH}#/policy_intervention_evidence/{index}",),
        tags=("fresh_uid", "source_only", "injected_structural", "ridge_nmae"),
    )


def compile_ledger(report_path: Path, plan_path: Path) -> dict[str, object]:
    report, report_pin = _read_pinned(report_path, REPORT_SHA256)
    _, plan_pin = _read_pinned(plan_path, PLAN_SHA256)
    required = {
        "verdict": report.get("verdict") == "STRUCTURAL_PROVENANCE_REBIND_FRESH_PROMOTION_FAIL",
        "p0_exact_pass": report.get("p0_pre_fit_exact_gate", {}).get("pass") is True,
        "source_consumer_fit_count": report.get("consumer_fit_count") == 4,
        "fresh_selected_uids": report.get("intervention_family_fresh_on_selected_uids") is True,
        "promotion_ineligible": report.get("structural_source_promotion_eligible") is False,
        "target_query_closed": report.get("target_query_opened") is False,
        "report_binds_pinned_plan": report.get("plan_dependency", {}).get("actual_sha256") == PLAN_SHA256,
    }
    if not all(required.values()):
        raise ValueError(f"fresh promotion report contract failed: {required}")
    evidence_rows = report.get("policy_intervention_evidence")
    if not isinstance(evidence_rows, list) or len(evidence_rows) != 2:
        raise ValueError("expected exactly two dataset evidence rows")
    by_dataset = {row["dataset_id"]: (index, row) for index, row in enumerate(evidence_rows)}
    traffic_index, traffic = by_dataset["monash:traffic_hourly"]
    covid_index, covid = by_dataset["monash:covid_deaths"]
    assert traffic["dataset_gate"]["pass"] is True and traffic["positive_gain_count"] == 6
    assert covid["dataset_gate"]["pass"] is False and covid["positive_gain_count"] == 3
    assert float(covid["median_gain_incumbent_minus_repaired"]) < 0.0

    template, template_payload = _capability()
    ledger = SignedEvidenceLedger(LEDGER_ID)
    ledger.append(_policy_evidence(traffic, traffic_index), EvidenceDisposition.SUPPORTED,
                  recorded_by=RECORDED_BY)
    ledger.append(_policy_evidence(covid, covid_index), EvidenceDisposition.CONTRADICTED,
                  recorded_by=RECORDED_BY)
    rows = ledger.to_rows()
    roundtrip = SignedEvidenceLedger.from_rows(LEDGER_ID, rows)
    assert roundtrip.to_rows() == rows
    assert template.capability_id == CAPABILITY_ID
    summary = _plain(roundtrip.summarize(CAPABILITY_ID))
    assert summary["compiled_verdict"] == "contradicted"
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "zero_fit_fresh_promotion_signed_ledger_compile",
        "dependencies": {"fresh_promotion_report": report_pin, "fresh_plan": plan_pin,
                         "required_report_checks": required, "pass": True},
        "capability_template": template_payload,
        "ledger": {"ledger_id": LEDGER_ID, "rows": list(rows),
                   "roundtrip_from_rows_pass": True},
        "summary": summary, "compiled_verdict": "contradicted",
        "structural_integrity_contract_status": "supported",
        "utility_conditioned_capability_status": "contradicted",
        "deployment_class": "MINIMAL_INTEGRITY_ADAPTER_CANDIDATE_NOT_MEMORY_CAPABILITY",
        "p0_exact_is_utility_support": False,
        "promotion_decision": "ABSTAIN_DO_NOT_REGISTER",
        "target_query_opened": False, "consumer_fit_count": 0,
        "claim_limit": "P0 supports schema-correct key rebind integrity only; fresh Consumer utility is contradicted, so no capability registration, promotion, Memory, natural-defect, Pattern, Target, or Query claim.",
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=root / REPORT_RELATIVE_PATH)
    parser.add_argument("--plan", type=Path, default=root / PLAN_RELATIVE_PATH)
    parser.add_argument("--output", type=Path, default=root / OUTPUT_RELATIVE_PATH)
    args = parser.parse_args()
    payload = compile_ledger(args.report, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(args.output)
    print(payload["compiled_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
