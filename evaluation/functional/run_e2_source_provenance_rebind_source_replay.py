"""Replay key-only supervised-target rebind on exposed FRED-MD and NN5 Source data."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_coherent_missingness_positive_control import (
    CONTEXT_LENGTH, DATASET_SPECS, EVAL_SERIES_PER_DATASET, HORIZON, RECENT_V2,
    TRAIN_SERIES_PER_DATASET, _center_scale, _evaluation_matrices, select_roster,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_label_binding_positive_control import (
    PERMUTATION_SHIFT, _permutation_order, _row_multiset,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_provenance_rebind_contract import (
    InputRow, RowKey, TargetRow, _dataset_sha, _rebind_supervised_targets_v0,
)

SCHEMA_VERSION = "e2-source-provenance-rebind-source-replay/1"
POLICIES = ("positional_incumbent", "key_rebind_repaired")
CONTRACT_REPORT_SHA256 = "eecd7427748c2be26e8e18c78290b86170c4fea9a4d4477d52080e4c27419f92"
MEAN_GAIN_MIN = 0.005
MEDIAN_GAIN_MIN_EXCLUSIVE = 0.0
POSITIVE_GAIN_MIN_COUNT = 5
HARM_THRESHOLD = -0.005
HARM_RATE_MAX = 0.25
OUTPUT_RELATIVE_PATH = "artifacts/functional/e2/source_provenance_rebind_source_replay_report.json"


@dataclass(frozen=True)
class TrainingBundle:
    x_train: np.ndarray
    incumbent_y: np.ndarray
    repaired_y: np.ndarray
    diagnostics: dict[str, object]
    p0: dict[str, object]


def _derange_present_anchors(
    dataset_id: str, inputs: list[InputRow], clean_targets: list[TargetRow]
) -> list[TargetRow]:
    """Contract-equivalent whole-row derangement over the anchors present here."""

    input_index = {row.key: index for index, row in enumerate(inputs)}
    clean_by_key = {row.key: row for row in clean_targets}
    deranged: list[TargetRow | None] = [None] * len(inputs)
    anchors = sorted({row.key.anchor for row in inputs})
    for anchor in anchors:
        anchor_keys = [row.key for row in inputs if row.key.anchor == anchor]
        if len({key.dataset_sha for key in anchor_keys}) != 1:
            raise ValueError("anchor group crosses dataset provenance")
        ordered = _permutation_order(
            dataset_id, anchor, [key.series_uid for key in anchor_keys]
        )
        prototype = anchor_keys[0]
        for rank, recipient_uid in enumerate(ordered):
            source_uid = ordered[(rank + PERMUTATION_SHIFT) % len(ordered)]
            recipient = RowKey(
                prototype.dataset_sha, recipient_uid, anchor, prototype.horizon
            )
            source = RowKey(prototype.dataset_sha, source_uid, anchor, prototype.horizon)
            deranged[input_index[recipient]] = clean_by_key[source]
    if any(row is None for row in deranged):
        raise AssertionError("whole-row derangement left an unassigned target")
    return [row for row in deranged if row is not None]


def _training_bundle(spec: object, train_items: list[object], values: dict) -> TrainingBundle:
    items = sorted(train_items, key=lambda item: item.record.series_uid)
    if len(items) != TRAIN_SERIES_PER_DATASET:
        raise ValueError(f"unexpected training roster size: {spec.dataset_id}")
    dataset_sha, deployment_metadata = _dataset_sha(items)
    inputs: list[InputRow] = []
    clean_targets: list[TargetRow] = []
    scale_counts: dict[str, int] = {}
    for anchor in RECENT_V2.anchors:
        for item in items:
            uid = item.record.series_uid
            raw = values[uid]
            context = np.asarray(raw[anchor - CONTEXT_LENGTH : anchor], dtype=np.float64)
            target = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
            if context.shape != (CONTEXT_LENGTH,) or target.shape != (HORIZON,):
                raise ValueError(f"insufficient clean training row: {uid}/{anchor}")
            if not np.isfinite(context).all() or not np.isfinite(target).all():
                raise ValueError(f"non-finite clean training row: {uid}/{anchor}")
            center, scale, method = _center_scale(context)
            features = np.concatenate(
                ((context - center) / scale, np.zeros(CONTEXT_LENGTH, dtype=np.float64))
            )
            target = (target - center) / scale
            key = RowKey(dataset_sha, uid, anchor, HORIZON)
            inputs.append(InputRow(key, features.copy()))
            clean_targets.append(TargetRow(key, target.copy()))
            scale_counts[method] = scale_counts.get(method, 0) + 1
    expected = TRAIN_SERIES_PER_DATASET * len(RECENT_V2.anchors)
    if len(inputs) != expected:
        raise AssertionError("unexpected clean training row count")
    x_train = np.asarray([row.payload for row in inputs], dtype=np.float64)
    clean_y = np.asarray([row.payload for row in clean_targets], dtype=np.float64)
    deranged = _derange_present_anchors(spec.dataset_id, inputs, clean_targets)
    status, action, repaired, reasons = _rebind_supervised_targets_v0(inputs, deranged)
    incumbent_y = np.asarray([row.payload for row in deranged], dtype=np.float64)
    repaired_y = (
        np.asarray([row.payload for row in repaired], dtype=np.float64)
        if repaired is not None
        else np.empty((0, HORIZON), dtype=np.float64)
    )
    clean_by_key = {row.key: row for row in clean_targets}
    fixed_points = sum(left.key == right.key for left, right in zip(inputs, deranged))
    checks: dict[str, object] = {
        "status": status, "action": action, "reason_codes": reasons,
        "input_key_unique": len(inputs) == len({row.key for row in inputs}),
        "target_key_unique": len(deranged) == len({row.key for row in deranged}),
        "key_sets_equal": {row.key for row in inputs} == {row.key for row in deranged},
        "fixed_point_count": fixed_points, "zero_fixed_points": fixed_points == 0,
        "input_matrix_unchanged": np.array_equal(
            x_train, np.asarray([row.payload for row in inputs], dtype=np.float64)
        ),
        "target_payload_multiset_unchanged": _row_multiset(clean_y)
        == _row_multiset(incumbent_y),
        "target_key_travelled_with_payload": all(
            np.array_equal(row.payload, clean_by_key[row.key].payload) for row in deranged
        ),
        "repaired_target_max_abs_error_to_clean": float(
            np.max(np.abs(repaired_y - clean_y))
        ),
    }
    checks["pass"] = (
        status == "ELIGIBLE" and action == "REPAIRED"
        and all(bool(checks[name]) for name in (
            "input_key_unique", "target_key_unique", "key_sets_equal",
            "zero_fixed_points", "input_matrix_unchanged",
            "target_payload_multiset_unchanged", "target_key_travelled_with_payload",
        ))
        and checks["repaired_target_max_abs_error_to_clean"] <= 1e-12
    )
    return TrainingBundle(
        x_train, incumbent_y, repaired_y,
        {"dataset_sha": dataset_sha, "deployment_metadata": deployment_metadata,
         "scale_method_counts": scale_counts, "row_count": expected,
         "anchors": list(RECENT_V2.anchors), "context_and_targets_clean": True},
        {"dataset_id": spec.dataset_id, "checked_before_any_consumer_fit": True,
         "consumer_fit_count_at_check": 0, "checks": checks, "pass": checks["pass"]},
    )


def _evidence(spec: object, losses: dict[str, list[float]], uids: list[str],
              train_diagnostics: dict, eval_diagnostics: dict) -> dict[str, object]:
    paired = []
    for uid, incumbent, repaired in zip(
        uids, losses[POLICIES[0]], losses[POLICIES[1]]
    ):
        gain = incumbent - repaired
        paired.append({
            "series_uid": uid,
            "positional_incumbent_normalized_mae": incumbent,
            "key_rebind_repaired_normalized_mae": repaired,
            "gain_incumbent_minus_repaired": gain,
            "positive_gain": gain > 0.0, "harmed": gain < HARM_THRESHOLD,
        })
    gains = [float(row["gain_incumbent_minus_repaired"]) for row in paired]
    mean_gain, median_gain = statistics.fmean(gains), statistics.median(gains)
    positive_count = sum(bool(row["positive_gain"]) for row in paired)
    gate_pass = (
        mean_gain >= MEAN_GAIN_MIN and median_gain > MEDIAN_GAIN_MIN_EXCLUSIVE
        and positive_count >= POSITIVE_GAIN_MIN_COUNT
    )
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "scientific_unit": "dataset_level_exposed_source_structural_replay",
        "dataset_id": spec.dataset_id,
        "policy_mean_normalized_mae": {
            policy: statistics.fmean(rows) for policy, rows in losses.items()
        },
        "policy_median_normalized_mae": {
            policy: statistics.median(rows) for policy, rows in losses.items()
        },
        "mean_gain_incumbent_minus_repaired": mean_gain,
        "median_gain_incumbent_minus_repaired": median_gain,
        "positive_gain_count": positive_count,
        "train_cohort": {"series_count": TRAIN_SERIES_PER_DATASET,
                         "anchor_count_per_series": len(RECENT_V2.anchors),
                         "diagnostics": train_diagnostics},
        "eval_cohort": {"series_count": EVAL_SERIES_PER_DATASET,
                        "diagnostics": eval_diagnostics},
        "paired_eval_rows": paired,
        "dataset_gate": {"mean_gain_min": MEAN_GAIN_MIN,
                         "median_gain_must_exceed": MEDIAN_GAIN_MIN_EXCLUSIVE,
                         "positive_gain_min_count": POSITIVE_GAIN_MIN_COUNT,
                         "eval_series_count": EVAL_SERIES_PER_DATASET, "pass": gate_pass},
    }


def run_replay(registry: Path, split: Path, subsplit: Path, clean_root: Path,
               contract_report_path: Path) -> dict[str, object]:
    contract_bytes = contract_report_path.read_bytes()
    contract = json.loads(contract_bytes)
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    dependency_pass = (
        contract_sha == CONTRACT_REPORT_SHA256
        and contract.get("verdict") == "PROVENANCE_REBIND_CONTRACT_PASS"
    )
    roster, selection = select_roster(
        registry_path=registry, split_path=split, support_a_subsplit_path=subsplit
    )
    values = _load_values([item.record for item in roster], clean_root)
    bundles = {
        spec.dataset_id: _training_bundle(
            spec, [item for item in roster
                   if item.record.dataset_id == spec.dataset_id and item.cohort == "train"],
            values,
        )
        for spec in DATASET_SPECS
    }
    p0_by_dataset = {dataset_id: bundle.p0 for dataset_id, bundle in bundles.items()}
    p0_pass = dependency_pass and all(bool(row["pass"]) for row in p0_by_dataset.values())
    evidence_rows: list[dict[str, object]] = []
    fit_count = 0
    if p0_pass:
        for spec in DATASET_SPECS:
            bundle = bundles[spec.dataset_id]
            eval_items = [item for item in roster
                          if item.record.dataset_id == spec.dataset_id and item.cohort == "eval"]
            x_eval, y_eval, eval_uids, eval_diag = _evaluation_matrices(
                spec=spec, eval_items=eval_items, values_by_uid=values
            )
            losses: dict[str, list[float]] = {}
            targets = {POLICIES[0]: bundle.incumbent_y, POLICIES[1]: bundle.repaired_y}
            for policy in POLICIES:
                model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
                model.fit(bundle.x_train, targets[policy])
                fit_count += 1
                prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
                if prediction.shape != y_eval.shape or not np.isfinite(prediction).all():
                    raise RuntimeError(f"invalid Ridge prediction: {spec.dataset_id}/{policy}")
                losses[policy] = [float(v) for v in np.mean(np.abs(prediction - y_eval), axis=1)]
            evidence_rows.append(
                _evidence(spec, losses, eval_uids, bundle.diagnostics, eval_diag)
            )
    if p0_pass and fit_count != 4:
        raise AssertionError("successful P0 must lead to exactly four Ridge fits")
    harms = [bool(row["harmed"]) for evidence in evidence_rows
             for row in evidence["paired_eval_rows"]]
    harm_rate = sum(harms) / len(harms) if harms else None
    p1_pass = (
        p0_pass and fit_count == 4
        and all(bool(row["dataset_gate"]["pass"]) for row in evidence_rows)
        and harm_rate is not None and harm_rate <= HARM_RATE_MAX
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exposed_source_structural_provenance_rebind_replay",
        "configuration": {
            "datasets": [spec.dataset_id for spec in DATASET_SPECS],
            "split": "support_a_discovery", "policies": list(POLICIES),
            "geometry_id": RECENT_V2.geometry_id, "anchors": list(RECENT_V2.anchors),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "context_length": CONTEXT_LENGTH, "horizon": HORIZON,
            "training_features": "clean normalized 192 values plus 192 zero-mask values",
            "target_binding": "whole TargetRow fixed derangement, then key-only rebind",
            "consumer": {"class": "sklearn.linear_model.Ridge", "alpha": 1.0,
                         "fit_intercept": True, "solver": "svd"},
        },
        "contract_dependency": {
            "path": str(contract_report_path), "expected_sha256": CONTRACT_REPORT_SHA256,
            "actual_sha256": contract_sha, "hash_matches": contract_sha == CONTRACT_REPORT_SHA256,
            "verdict": contract.get("verdict"), "pass": dependency_pass,
        },
        "roster": {"selection": selection, "selected_value_series_count": len(roster)},
        "p0_pre_fit_gate": {"contract_dependency_required": True,
                            "dataset_results": p0_by_dataset,
                            "conjunction_across_both_datasets": True,
                            "failure_action": "zero-fit stop", "pass": p0_pass},
        "policy_intervention_evidence": evidence_rows,
        "p1_replay_gate": {
            "dataset_gates_conjunctive": True,
            "pooled_harm": {"definition": "gain < -0.005", "count": sum(harms),
                            "total": len(harms), "rate": harm_rate,
                            "rate_max": HARM_RATE_MAX,
                            "pass": harm_rate is not None and harm_rate <= HARM_RATE_MAX},
            "all_conditions_conjunctive": True, "pass": p1_pass,
        },
        "information_wall": {"source_only": True, "support_a_discovery_only": True,
                             "support_b_read": False, "target_or_query_read": False,
                             "target_query_opened": False},
        "consumer_fit_count": fit_count, "chronos_judge_call_count": 0,
        "intervention_family_new_on_these_datasets": True,
        "datasets_previously_exposed_to_other_diagnostics": True,
        "formal_fresh_evidence": False, "prototype_not_registered": True,
        "agent_enabled": False, "memory_enabled": False, "promotion_eligible": False,
        "formal_transfer": False, "numeric_pattern_claim_supported": False,
        "target_query_opened": False,
        "verdict": "STRUCTURAL_PROVENANCE_REBIND_SOURCE_REPLAY_PASS" if p1_pass
                   else "STRUCTURAL_PROVENANCE_REBIND_SOURCE_REPLAY_FAIL",
        "claim_limit": "Exposed-Source structural intervention-family replay only; no natural-defect, Pattern, promotion, Memory, Target, or transfer claim.",
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=root / "artifacts/frozen/benchmark_v02/series_registry.jsonl")
    parser.add_argument("--split", type=Path, default=root / "artifacts/frozen/benchmark_v02/split_manifest.json")
    parser.add_argument("--support-a-subsplit", type=Path, default=root / "artifacts/frozen/benchmark_v02/support_a_subsplit.json")
    parser.add_argument("--clean-root", type=Path, default=root / "data/benchmark_v0_2/clean_base")
    parser.add_argument("--contract-report", type=Path, default=root / "artifacts/functional/e2/source_provenance_rebind_contract_report.json")
    parser.add_argument("--output", type=Path, default=root / OUTPUT_RELATIVE_PATH)
    args = parser.parse_args()
    report = run_replay(args.registry, args.split, args.support_a_subsplit, args.clean_root,
                        args.contract_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(args.output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
