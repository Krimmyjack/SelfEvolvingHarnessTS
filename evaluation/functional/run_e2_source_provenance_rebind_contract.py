"""Zero-fit Source contract for detecting, repairing, or abstaining from target-table
binding defects using immutable deployment-visible row keys."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    DISCOVERY_SUBSPLIT, HORIZON, SOURCE_DATASETS, TRAIN_ANCHORS,
    _load_roster_values, select_roster,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_label_binding_positive_control import (
    PERMUTATION_SHIFT, _bind_and_inverse_labels, _clean_training_rows,
    _permutation_order, _row_multiset,
)
SCHEMA_VERSION = "e2-source-provenance-rebind-contract/1"
PROGRAM_ID = "rebind_supervised_targets_v0"
OUTPUT_RELATIVE_PATH = "artifacts/functional/e2/source_provenance_rebind_contract_report.json"
PRIOR_REPORT_SHA256 = "37e91261c5d743594a45726f937ebad394f2d44b1e7b3608628d04c5feab4101"
@dataclass(frozen=True, order=True)
class RowKey:
    dataset_sha: str
    series_uid: str
    anchor: int
    horizon: int
@dataclass(frozen=True)
class InputRow:
    key: RowKey
    payload: np.ndarray
@dataclass(frozen=True)
class TargetRow:
    key: RowKey
    payload: np.ndarray

def _binding_status(
    input_rows: list[InputRow], target_rows: list[TargetRow]
) -> tuple[str, list[str]]:
    input_keys = [row.key for row in input_rows]
    target_keys = [row.key for row in target_rows]
    reasons: list[str] = []
    if len(input_keys) != len(set(input_keys)):
        reasons.append("DUPLICATE_INPUT_KEY")
    if len(target_keys) != len(set(target_keys)):
        reasons.append("DUPLICATE_TARGET_KEY")
    if set(input_keys) != set(target_keys):
        reasons.append("KEY_SET_MISMATCH")
    if reasons:
        return "UNRESOLVED", reasons
    if input_keys == target_keys:
        return "INELIGIBLE", ["POSITIONAL_BINDING_ALREADY_INTACT"]
    return "ELIGIBLE", ["UNIQUE_EQUAL_KEY_SETS_WITH_POSITIONAL_MISMATCH"]

def _rebind_supervised_targets_v0(
    input_rows: list[InputRow], target_rows: list[TargetRow]
) -> tuple[str, str, list[TargetRow] | None, list[str]]:
    """Join whole TargetRows to InputRows by key; never inspect target payloads."""
    status, reasons = _binding_status(input_rows, target_rows)
    if status == "UNRESOLVED":
        return status, "ABSTAIN", None, reasons
    if status == "INELIGIBLE":
        return status, "NO_OP", list(target_rows), reasons
    target_by_key = {row.key: row for row in target_rows}
    return status, "REPAIRED", [target_by_key[row.key] for row in input_rows], reasons

def _program_static_contract() -> dict[str, object]:
    source = "\n".join(
        inspect.getsource(fn) for fn in (_binding_status, _rebind_supervised_targets_v0)
    )
    tree = ast.parse(textwrap.dedent(source))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden = sorted(attributes & {"payload", "values", "loss", "clean_truth"})
    return {"program_ast_parses": True, "only_row_attribute_read": sorted(attributes),
            "forbidden_numeric_or_outcome_attribute_reads": forbidden,
            "pass": not forbidden}

def _dataset_sha(train_items: list[object]) -> tuple[str, dict[str, str]]:
    bindings = {(item.record.dataset_id, item.record.source_asset_sha256,
                 item.record.source_revision) for item in train_items}
    if len(bindings) != 1:
        raise ValueError("dataset deployment binding is not unique")
    dataset_id, asset_sha, revision = bindings.pop()
    metadata = {"dataset_id": dataset_id, "source_asset_sha256": asset_sha,
                "source_revision": revision}
    return hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(), metadata

def _derange_target_rows(
    dataset_id: str,
    input_rows: list[InputRow],
    clean_targets: list[TargetRow],
) -> list[TargetRow]:
    input_index = {row.key: index for index, row in enumerate(input_rows)}
    clean_by_key = {row.key: row for row in clean_targets}
    deranged: list[TargetRow | None] = [None] * len(input_rows)
    dataset_sha = input_rows[0].key.dataset_sha
    for anchor in TRAIN_ANCHORS:
        keys = [row.key for row in input_rows if row.key.anchor == anchor]
        ordered = _permutation_order(dataset_id, anchor, [key.series_uid for key in keys])
        for rank, recipient_uid in enumerate(ordered):
            source_uid = ordered[(rank + PERMUTATION_SHIFT) % len(ordered)]
            recipient = RowKey(dataset_sha, recipient_uid, anchor, HORIZON)
            source = RowKey(dataset_sha, source_uid, anchor, HORIZON)
            deranged[input_index[recipient]] = clean_by_key[source]
    if any(row is None for row in deranged):
        raise AssertionError("target-row derangement left an unassigned row")
    return [row for row in deranged if row is not None]

def _dataset_contract(dataset_id: str, train_items: list[object], values: dict) -> dict:
    x, clean_y, legacy_keys, diagnostics = _clean_training_rows(
        dataset_id=dataset_id, train_items=train_items, values_by_uid=values
    )
    old_deranged, old_oracle, _, old_p0 = _bind_and_inverse_labels(
        dataset_id=dataset_id,
        x_train=x,
        clean_y_train=clean_y,
        row_keys=legacy_keys,
    )
    dataset_sha, deployment_metadata = _dataset_sha(train_items)
    keys = [RowKey(dataset_sha, uid, anchor, HORIZON) for anchor, uid in legacy_keys]
    inputs = [InputRow(key, x[index]) for index, key in enumerate(keys)]
    clean_targets = [TargetRow(key, clean_y[index]) for index, key in enumerate(keys)]
    deranged = _derange_target_rows(dataset_id, inputs, clean_targets)
    status, action, repaired, reasons = _rebind_supervised_targets_v0(inputs, deranged)
    if repaired is None:
        repaired_matrix = np.empty((0, HORIZON), dtype=np.float64)
    else:
        repaired_matrix = np.asarray([row.payload for row in repaired], dtype=np.float64)
    deranged_matrix = np.asarray([row.payload for row in deranged], dtype=np.float64)
    input_matrix_after = np.asarray([row.payload for row in inputs], dtype=np.float64)
    clean_by_key = {row.key: row for row in clean_targets}
    fixed_points = sum(left.key == right.key for left, right in zip(inputs, deranged))
    checks = {
        "status": status,
        "action": action,
        "reason_codes": reasons,
        "input_key_unique": len(keys) == len(set(keys)),
        "target_key_unique": len(deranged) == len({row.key for row in deranged}),
        "key_sets_equal": {row.key for row in inputs} == {row.key for row in deranged},
        "zero_fixed_points": fixed_points == 0,
        "fixed_point_count": fixed_points,
        "input_matrix_unchanged": np.array_equal(x, input_matrix_after),
        "target_payload_multiset_unchanged": _row_multiset(clean_y)
        == _row_multiset(deranged_matrix),
        "target_key_travelled_with_payload": all(
            np.array_equal(row.payload, clean_by_key[row.key].payload) for row in deranged
        ),
        "defect_exactly_matches_prior_derangement": np.array_equal(
            deranged_matrix, old_deranged
        ),
        "repaired_target_max_abs_error_to_clean": float(
            np.max(np.abs(repaired_matrix - clean_y))
        ),
        "repaired_target_max_abs_error_to_prior_inverse_oracle": float(
            np.max(np.abs(repaired_matrix - old_oracle))
        ),
        "prior_inverse_manifest_p0_pass": bool(old_p0["pass"]),
    }
    checks["pass"] = (
        status == "ELIGIBLE"
        and action == "REPAIRED"
        and all(bool(checks[name]) for name in (
            "input_key_unique", "target_key_unique", "key_sets_equal",
            "zero_fixed_points", "input_matrix_unchanged",
            "target_payload_multiset_unchanged", "target_key_travelled_with_payload",
            "defect_exactly_matches_prior_derangement", "prior_inverse_manifest_p0_pass",
        ))
        and checks["repaired_target_max_abs_error_to_clean"] <= 1e-12
        and checks["repaired_target_max_abs_error_to_prior_inverse_oracle"] <= 1e-12
    )
    return {"dataset_id": dataset_id, "dataset_sha": dataset_sha,
            "deployment_metadata": deployment_metadata, "row_count": len(inputs),
            "clean_row_diagnostics": diagnostics, "checks": checks}

def _synthetic_p2() -> dict[str, object]:
    sha = hashlib.sha256(b"synthetic-provenance-contract").hexdigest()
    keys = [RowKey(sha, uid, 8, 2) for uid in ("a", "b", "c")]
    inputs = [InputRow(key, np.asarray([index])) for index, key in enumerate(keys)]
    targets = [TargetRow(key, np.asarray([index, index + 10])) for index, key in enumerate(keys)]

    def run(name: str, rows: list[TargetRow]) -> dict[str, object]:
        status, action, output, reasons = _rebind_supervised_targets_v0(inputs, rows)
        exact = output is not None and all(
            np.array_equal(row.payload, targets[index].payload)
            for index, row in enumerate(output)
        )
        return {"case": name, "status": status, "action": action, "exact": exact,
                "reason_codes": reasons, "abstained": output is None}

    other_sha = hashlib.sha256(b"other-dataset").hexdigest()
    cases = [
        run("intact", targets),
        run("valid_misbinding", targets[1:] + targets[:1]),
        run("duplicate_key", [targets[0], targets[0], targets[2]]),
        run("missing_key", targets[:2]),
        run("cross_set", targets[:2] + [TargetRow(RowKey(other_sha, "c", 8, 2), np.array([2, 12]))]),
    ]
    intact = cases[0]
    valid = cases[1]
    ambiguous = cases[2:]
    false_activation_rate = float(intact["status"] == "ELIGIBLE")
    valid_recall = float(valid["status"] == "ELIGIBLE" and valid["exact"])
    ambiguous_abstention_rate = sum(bool(row["abstained"]) for row in ambiguous) / len(ambiguous)
    passed = false_activation_rate == 0.0 and valid_recall == 1.0 and ambiguous_abstention_rate == 1.0
    return {
        "cases": cases,
        "clean_false_activation_rate": false_activation_rate,
        "valid_misbinding_exact_recall": valid_recall,
        "ambiguous_abstention_rate": ambiguous_abstention_rate,
        "gate": {"required": {"clean_false_activation_rate": 0.0,
                                "valid_misbinding_exact_recall": 1.0,
                                "ambiguous_abstention_rate": 1.0}, "pass": passed},
    }

def run_contract(registry: Path, split: Path, subsplit: Path, clean_root: Path,
                 prior_report_path: Path) -> dict[str, object]:
    roster, selection = select_roster(
        registry_path=registry, split_path=split, support_a_subsplit_path=subsplit
    )
    train = [item for item in roster if item.cohort == "train"]
    values = _load_roster_values(train, clean_root)
    datasets = {
        dataset_id: _dataset_contract(
            dataset_id,
            [item for item in train if item.record.dataset_id == dataset_id],
            values,
        )
        for dataset_id in sorted(SOURCE_DATASETS)
    }
    static_contract = _program_static_contract()
    p2 = _synthetic_p2()
    prior_bytes = prior_report_path.read_bytes()
    prior = json.loads(prior_bytes)
    prior_sha = hashlib.sha256(prior_bytes).hexdigest()
    prior_hash_matches = prior_sha == PRIOR_REPORT_SHA256
    prior_p1_pass = bool(prior["p1_positive_control_gate"]["pass"])
    p0_pass = all(bool(row["checks"]["pass"]) for row in datasets.values())
    p1_reused = p0_pass and prior_p1_pass and prior_hash_matches
    passed = p0_pass and bool(p2["gate"]["pass"]) and bool(static_contract["pass"]) and p1_reused
    p1_numbers = {
        row["dataset_id"]: {
            "mean_normalized_mae_degradation": row["mean_normalized_mae_degradation"],
            "median_normalized_mae_degradation": row["median_normalized_mae_degradation"],
            "positive_degradation_count": row["positive_degradation_count"],
            "gate": row["p1_gate"],
        }
        for row in prior["policy_intervention_evidence"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "source_only_provenance_rebind_contract_vertical_slice",
        "configuration": {
            "datasets": sorted(SOURCE_DATASETS), "split": "support_a",
            "subsplit": DISCOVERY_SUBSPLIT, "program_id": PROGRAM_ID,
            "row_key_schema": ["dataset_sha", "series_uid", "anchor", "horizon"],
            "row_key_immutable_frozen_dataclass": True, "program_reads": "immutable row keys only",
            "target_key_moves_with_payload": True,
            "incumbent_binding": "positional zip after whole-TargetRow derangement",
            "prototype_not_registered": True,
        },
        "roster": {"selection": selection, "loaded_value_cohort": "train_only",
                   "train_series_count": len(train)},
        "p0_exact_rebind_gate": {"max_error": 1e-12, "dataset_results": datasets,
                                 "conjunction_across_both_datasets": True,
                                 "failure_action": "zero-fit stop", "pass": p0_pass},
        "p1_reuse": {
            "prior_report_path": str(prior_report_path),
            "prior_report_sha256": prior_sha, "hash_matches": prior_hash_matches,
            "prior_schema_version": prior.get("schema_version"),
            "prior_verdict": prior.get("verdict"),
            "prior_consumer_fit_count": prior.get("consumer_fit_count"),
            "prior_p1_gate_pass": prior_p1_pass,
            "prior_dataset_numbers": p1_numbers,
            "p1_reused_by_exact_matrix_equivalence": p1_reused,
            "new_consumer_fit_count": 0,
        },
        "p2_applicability_contract": p2,
        "program_static_contract": static_contract,
        "information_wall": {
            "source_only": True, "support_a_discovery_train_only_values": True,
            "program_reads_values_loss_or_clean_truth": False,
            "uci_values_context_or_future_read": False,
            "support_b_values_context_or_future_read": False,
            "target_values_context_or_future_read": False,
            "query_values_context_or_future_read": False, "target_query_opened": False,
        },
        "new_consumer_fit_count": 0, "chronos_judge_call_count": 0,
        "agent_enabled": False, "memory_enabled": False, "promotion_eligible": False,
        "formal_transfer": False, "fresh": False, "prototype_not_registered": True,
        "next_if_pass": "FRED/NN5 intervention-family replay only; no promotion or Query",
        "target_query_opened": False,
        "verdict": "PROVENANCE_REBIND_CONTRACT_PASS" if passed else "PROVENANCE_REBIND_CONTRACT_FAIL",
        "claim_limit": "A zero-fit exposed-Source structural contract, not natural-defect, promotion, Target, or transfer evidence.",
    }

def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=root / "artifacts/frozen/benchmark_v02/series_registry.jsonl")
    parser.add_argument("--split", type=Path, default=root / "artifacts/frozen/benchmark_v02/split_manifest.json")
    parser.add_argument("--support-a-subsplit", type=Path, default=root / "artifacts/frozen/benchmark_v02/support_a_subsplit.json")
    parser.add_argument("--clean-root", type=Path, default=root / "data/benchmark_v0_2/clean_base")
    parser.add_argument("--prior-report", type=Path, default=root / "artifacts/functional/e2/source_label_binding_positive_control_report.json")
    parser.add_argument("--output", type=Path, default=root / OUTPUT_RELATIVE_PATH)
    args = parser.parse_args()
    report = run_contract(args.registry, args.split, args.support_a_subsplit, args.clean_root, args.prior_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(args.output)
    print(report["verdict"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
