"""Plan or evaluate the exposed-Source E2-J0 Judge readability calibration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import statistics
import sys
import types
from pathlib import Path
from typing import Any

# Plan is metadata-only and must remain runnable without the numeric stack.
if importlib.util.find_spec("numpy") is None:
    sys.modules["numpy"] = types.ModuleType("numpy")

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes


SCHEMA_VERSION = "e2-j0-judge-readability-calibration/1"
PLAN_NAME = "source_judge_readability_calibration_plan.json"
REPORT_NAME = "source_judge_readability_calibration_report.json"
PLAN_RELATIVE_PATH = f"artifacts/functional/e2/{PLAN_NAME}"
REPORT_RELATIVE_PATH = f"artifacts/functional/e2/{REPORT_NAME}"
CONTEXT_LENGTH = 192
HORIZON = 48
ANCHORS = (240, 300, 360, 420, 480, 540)
SEEDS = (0, 1, 2)
NOMINAL_DOSES = (0.05, 0.10, 0.20)
SELECTED_ROW_COUNTS = (4, 7, 14)
TARGET_BLOCK = (18, 30)
TARGET_OFFSET = 2.0
SALT = "e2-j0-standardized-additive-target-block-v1"
POLICIES = ("clean_reference", "exact_repaired_repeat",
            "standardized_additive_target_block")
EXPECTED_FITS = 22
DEGRADATION_MIN = 0.02
PREDICTION_TOLERANCE = 1e-12
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260713
MDE_ONE_SIDED_Z = 2.4865
ESTIMAND_HELPER_SHA256 = {
    "evaluation/benchmark_v02/dev_eval.py":
        "69bd7d8fa4b82c02e81b140ed58bea49fe6d97e2a5fa3fc33998705b9109affb",
    "evaluation/benchmark_v02/metrics.py":
        "b7fc98eb5e620e8aeab603e28d69ec417ed1b8726fb53e7bf5388ed72db7afd4",
    "evaluation/functional/run_e2_source_cohort_policy_premise.py":
        "7c2fec5ddd0dc9a34fb43bddf7dfcbb36a6bb3af81bee2b80e611d48f6141498",
}
EXPECTED_RUNTIME_VERSIONS = {"numpy": "2.2.6", "scikit-learn": "1.7.2"}
DEPENDENCIES = {
    "monash:traffic_hourly": (
        "artifacts/functional/e2/source_provenance_rebind_validation_replay_report.json",
        "0c874fa28bb6fec3c25732f75cac0727dcc8927d697b860ea6324338dc6cc586",
        "STRUCTURAL_PROVENANCE_REBIND_VALIDATION_PASS",
        "support_a_validation",
    ),
    "legacy_monash:fred_md": (
        "artifacts/functional/e2/source_provenance_rebind_source_replay_report.json",
        "b38b25e64013591d7a15bb779459f7f47f802e1bc4568906d20be333d2619cbd",
        "STRUCTURAL_PROVENANCE_REBIND_SOURCE_REPLAY_PASS",
        "support_a_discovery",
    ),
}
SPECS = {
    "monash:traffic_hourly": {
        "train_stop": 928, "future_bounds": [928, 976], "period": 24,
        "frequency": "hourly",
    },
    "legacy_monash:fred_md": {
        "train_stop": 632, "future_bounds": [632, 680], "period": 12,
        "frequency": "monthly",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _runner_binding(root: Path) -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}


def _estimand_helper_bindings(root: Path) -> dict[str, dict[str, object]]:
    bindings: dict[str, dict[str, object]] = {}
    for relative, expected in ESTIMAND_HELPER_SHA256.items():
        actual = _sha(root / relative)
        if actual != expected:
            raise ValueError(f"estimand helper changed: {relative}")
        bindings[relative] = {"expected_sha256": expected,
                              "actual_sha256": actual, "hash_matches": True}
    return bindings


def _frozen_metadata(root: Path) -> tuple[dict[str, dict[str, Any]],
                                          dict[str, dict[str, Any]], dict[str, Any],
                                          dict[str, str]]:
    registry_path = root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    split_path = root / "artifacts/frozen/benchmark_v02/split_manifest.json"
    subsplit_path = root / "artifacts/frozen/benchmark_v02/support_a_subsplit.json"
    records = {row["series_uid"]: row for row in
               (json.loads(line) for line in registry_path.read_text("utf-8").splitlines())}
    assignments = {row["series_uid"]: row for row in _read_json(split_path)["assignments"]}
    subsplit = _read_json(subsplit_path)
    hashes = {path.relative_to(root).as_posix(): _sha(path)
              for path in (registry_path, split_path, subsplit_path)}
    return records, assignments, subsplit, hashes


def build_plan(root: Path) -> dict[str, object]:
    records, assignments, subsplit, metadata_hashes = _frozen_metadata(root)
    members = subsplit.get("members")
    if not isinstance(members, dict):
        raise ValueError("Support-A subsplit members missing")
    roster: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    dependency_rows: dict[str, object] = {}
    asset_shas: set[str] = set()
    for dataset_id, dependency in DEPENDENCIES.items():
        relative, expected_sha, expected_verdict, expected_subsplit = dependency
        path = root / relative
        actual_sha = _sha(path)
        report = _read_json(path)
        if actual_sha != expected_sha or report.get("verdict") != expected_verdict:
            raise ValueError(f"source roster dependency changed: {dataset_id}")
        selected = report.get("roster", {}).get("selection", {}).get(
            "selected_by_dataset", {}).get(dataset_id)
        if not isinstance(selected, dict):
            raise ValueError(f"source roster missing: {dataset_id}")
        train_uids, eval_uids = selected.get("train"), selected.get("eval")
        if not isinstance(train_uids, list) or not isinstance(eval_uids, list):
            raise ValueError(f"invalid source roster: {dataset_id}")
        if len(train_uids) != 12 or len(eval_uids) != 8:
            raise ValueError(f"source roster must be 12+8: {dataset_id}")
        spec = SPECS[dataset_id]
        train_groups, eval_groups = set(), set()
        dataset_assets: set[str] = set()
        subsplit_uids = set(members.get(expected_subsplit, []))
        for cohort, uids in (("train", train_uids), ("eval", eval_uids)):
            for uid in uids:
                record, assignment = records.get(uid), assignments.get(uid)
                if not record or not assignment or uid not in subsplit_uids:
                    raise ValueError(f"roster UID missing from frozen metadata: {uid}")
                expected_bounds = {"train": [0, spec["train_stop"]],
                    "validation": spec["future_bounds"],
                    "test": [spec["future_bounds"][1], spec["future_bounds"][1] + HORIZON]}
                eligible = (record.get("dataset_id") == dataset_id
                    and assignment.get("dataset_id") == dataset_id
                    and assignment.get("role") == "support_a"
                    and assignment.get("chronological_boundaries") == expected_bounds
                    and record.get("admission_reasons") == []
                    and record.get("natural_missing_count") == 0
                    and record.get("frequency") == spec["frequency"]
                    and "support_a" in record.get("roles_allowed", []))
                if not eligible:
                    raise ValueError(f"ineligible locked roster UID: {uid}")
                group = str(record["overlap_group"])
                (train_groups if cohort == "train" else eval_groups).add(group)
                dataset_assets.add(str(record["source_asset_sha256"]))
                roster.append({"dataset_id": dataset_id, "cohort": cohort,
                    "subsplit": expected_subsplit, "series_uid": uid,
                    "entity_id": record["entity_id"], "overlap_group": group,
                    "source_asset_sha256": record["source_asset_sha256"]})
        if set(train_uids) & set(eval_uids) or train_groups & eval_groups:
            raise ValueError(f"train/eval roster overlap: {dataset_id}")
        if len(train_groups) != 12 or len(eval_groups) != 8:
            raise ValueError(f"roster must contain 12/8 unique overlap groups: {dataset_id}")
        if len(dataset_assets) != 1:
            raise ValueError(f"dataset deployment asset is not unique: {dataset_id}")
        asset_shas.update(dataset_assets)
        revisions = {str(records[uid]["source_revision"]) for uid in train_uids + eval_uids}
        if len(revisions) != 1:
            raise ValueError(f"dataset deployment revision is not unique: {dataset_id}")
        deployment = {"dataset_id": dataset_id,
            "source_asset_sha256": next(iter(dataset_assets)),
            "source_revision": next(iter(revisions))}
        dataset_sha = hashlib.sha256(canonical_json_bytes(deployment)).hexdigest()
        audits[dataset_id] = {"train_count": 12, "eval_count": 8,
            "train_unique_overlap_group_count": len(train_groups),
            "eval_unique_overlap_group_count": len(eval_groups),
            "uid_disjoint": True, "overlap_group_disjoint": True,
            "source_asset_sha256": next(iter(dataset_assets)),
            "deployment_metadata": deployment, "dataset_sha": dataset_sha}
        dependency_rows[dataset_id] = {"path": relative,
            "expected_sha256": expected_sha, "actual_sha256": actual_sha,
            "hash_matches": True, "verdict": expected_verdict, "pass": True}
    if len(asset_shas) != 2 or any(row["dataset_id"].startswith("uci") for row in roster):
        raise ValueError("datasets must use two different non-UCI deployment assets")
    return {
        "schema_version": SCHEMA_VERSION, "phase": "plan", "plan_status": "READY",
        "scientific_role": "exposed_source_downstream_judge_readability_calibration",
        "runner_dependency": _runner_binding(root),
        "estimand_helper_dependencies": _estimand_helper_bindings(root),
        "runtime_version_pins": EXPECTED_RUNTIME_VERSIONS,
        "source_roster_dependencies": dependency_rows,
        "frozen_metadata_sha256": metadata_hashes,
        "configuration": {"datasets": SPECS, "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON, "train_anchors": list(ANCHORS),
            "policies": list(POLICIES), "ridge": {"alpha": 1.0,
                "fit_intercept": True, "solver": "svd"},
            "fit_schedule": {"control_fits_first": 4,
                "all_controls_must_be_deterministic_before_corrupt_fits": True,
                "corrupt_fits_after_control_gate": 18},
            "expected_consumer_fit_count": EXPECTED_FITS},
        "instrument": {"kind": "standardized_additive_target_block/v1",
            "row_key": ["dataset_sha", "series_uid", "anchor", "horizon"],
            "salt": SALT, "seeds": list(SEEDS),
            "nominal_doses": list(NOMINAL_DOSES),
            "rounding": "round-half-up over 72 rows",
            "selected_row_counts": list(SELECTED_ROW_COUNTS),
            "realized_doses": [count / 72 for count in SELECTED_ROW_COUNTS],
            "ordering": "sha256(salt|dataset_sha|RowKey|seed)",
            "nested_selection": "prefixes of one stable per-seed row order",
            "target_block_half_open": list(TARGET_BLOCK),
            "standardized_additive_offset": TARGET_OFFSET,
            "exact_repair": "copy clean selected block into corrupt copy; never subtract",
            "exact_repair_fit_source": {"seed": 0, "selected_row_count": 14}},
        "metric_and_power": {"prediction_space": "original_units_after_eval_inverse_scale",
            "reported_metrics": ["original_unit_mae", "smase"],
            "seasonal_scale_source": "clean original-unit pre-future train",
            "seasonal_scale_min_pairs": 32,
            "invalid_scale_action": "P0_FAIL_ZERO_FIT_NO_FLOOR",
            "power_component": "runner-local one-sided overlap-group bootstrap",
            "material_threshold": DEGRADATION_MIN,
            "bootstrap_b": BOOTSTRAP_B, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_quantile": 0.05,
            "mde80_one_sided_multiplier": MDE_ONE_SIDED_Z},
        "p0_pre_fit": {"failure_action": "zero-fit stop",
            "required_checks": ["selected_row_count_exact",
                "nested_selected_row_prefixes", "features_unchanged",
                "selected_block_additive_offset_exact",
                "unselected_targets_unchanged", "target_block_exterior_unchanged",
                "exact_repair_array_equal_clean", "scale_floor_count_zero",
                "eval_context_and_future_finite",
                "all_eval_seasonal_scales_valid"]},
        "gate": {"datasets_conjunctive": True, "scale_valid_required": "8/8",
            "endpoint_selected_row_count": 14,
            "endpoint_mean_smase_degradation_min": DEGRADATION_MIN,
            "endpoint_bootstrap_q05_must_exceed": 0.0,
            "endpoint_positive_uid_min_count": 6,
            "dose_response": "0 < mean(d05) <= mean(d10) <= mean(d20)",
            "e2_ready_requires_mde80_one_sided_lte": DEGRADATION_MIN,
            "clean_exact_repeat_tolerance": PREDICTION_TOLERANCE,
            "underpowered_classification": (
                "READABLE_AT_INJECTED_DOSE_BUT_UNDERPOWERED_FOR_EPSILON"),
            "pass_verdict": "JUDGE_READABILITY_CALIBRATION_PASS"},
        "roster_protocol_amendment": {
            "status": "EXPOSED_PREVIOUSLY_FIXED_ROSTER_REUSE_FOR_CALIBRATION_ONLY",
            "traffic": {"dataset_id": "monash:traffic_hourly",
                "subsplit": "support_a_validation",
                "identity_pin": DEPENDENCIES["monash:traffic_hourly"][1]},
            "fred": {"dataset_id": "legacy_monash:fred_md",
                "subsplit": "support_a_discovery",
                "identity_pin": DEPENDENCIES["legacy_monash:fred_md"][1]},
            "unified_discovery_roster": False,
            "fresh_or_held_out_claim_supported": False,
            "claim_scope": "downstream Judge calibration on exposed Source rosters only"},
        "roster": roster, "roster_audit": audits,
        "information_wall": {"series_values_loaded": False, "source_only": True,
            "exposed_rosters_only": True, "complete_frozen_metadata_read": True,
            "support_b_values_read": False, "uci_values_read": False,
            "target_values_read": False, "query_values_read": False,
            "target_closed": True, "query_closed": True, "target_query_opened": False},
        "claim_ceiling": "Judge readability calibration only; not Capability, Pattern, Memory, promotion, transfer, Target, or Query evidence.",
        "target_query_opened": False,
    }


def _verify_plan(root: Path, path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, object]]:
    actual_sha = _sha(path)
    if actual_sha != expected_sha:
        raise ValueError("plan SHA256 mismatch")
    plan = _read_json(path)
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("plan_status") != "READY":
        raise ValueError("plan is not ready")
    runner = _runner_binding(root)
    runner_match = runner == plan.get("runner_dependency")
    rebuilt_plan = build_plan(root)
    canonical_plan_match = (canonical_json_bytes(plan)
                            == canonical_json_bytes(rebuilt_plan))
    estimand_helpers_match = (plan.get("estimand_helper_dependencies")
                              == rebuilt_plan.get("estimand_helper_dependencies"))
    _, _, _, metadata_hashes = _frozen_metadata(root)
    metadata_match = metadata_hashes == plan.get("frozen_metadata_sha256")
    dependencies_match = True
    for dataset_id, dependency in DEPENDENCIES.items():
        relative, expected, verdict, _ = dependency
        payload = _read_json(root / relative)
        row = plan.get("source_roster_dependencies", {}).get(dataset_id, {})
        dependencies_match &= (_sha(root / relative) == expected
            and payload.get("verdict") == verdict and row.get("expected_sha256") == expected)
    return plan, {"path": str(path), "expected_sha256": expected_sha,
        "actual_sha256": actual_sha, "hash_matches": True,
        "canonical_rebuild_matches_input_plan": canonical_plan_match,
        "runner_sha_matches": runner_match, "frozen_metadata_matches": metadata_match,
        "estimand_helper_sha_matches": estimand_helpers_match,
        "source_dependencies_match": bool(dependencies_match),
        "roster_injection_blocked": canonical_plan_match,
        "pass": (canonical_plan_match and estimand_helpers_match
                 and runner_match and metadata_match
                 and dependencies_match)}


def _row_order(dataset_sha: str, seed: int,
               keys: list[tuple[str, str, int, int]]) -> list[int]:
    def rank(index: int) -> tuple[str, tuple[str, str, int, int]]:
        row_key = "|".join(str(value) for value in keys[index])
        material = f"{SALT}|{dataset_sha}|{row_key}|{seed}".encode("utf-8")
        return hashlib.sha256(material).hexdigest(), keys[index]
    return sorted(range(len(keys)), key=rank)


def _one_sided_bootstrap(np: Any, gains: dict[str, float],
                         clusters: dict[str, str]) -> dict[str, object]:
    by_cluster: dict[str, list[float]] = {}
    for uid, value in sorted(gains.items()):
        by_cluster.setdefault(clusters[uid], []).append(float(value))
    names = sorted(by_cluster)
    if len(names) < 2:
        raise ValueError("one-sided bootstrap requires at least two overlap groups")
    sums = np.asarray([sum(by_cluster[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(by_cluster[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(names), size=(BOOTSTRAP_B, len(names)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    se = float(np.std(means, ddof=1))
    return {"b": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED,
        "n_series": len(gains), "n_clusters": len(names),
        "effect": statistics.fmean(gains.values()),
        "bootstrap_q05": float(np.quantile(means, 0.05, method="linear")),
        "standard_error": se, "mde80_one_sided": MDE_ONE_SIDED_Z * se,
        "mde80_one_sided_multiplier": MDE_ONE_SIDED_Z,
        "material_threshold": DEGRADATION_MIN}


def _finish_failure(preflight: dict[str, object]) -> dict[str, object]:
    return {"schema_version": SCHEMA_VERSION, "phase": "evaluate",
        "scientific_role": "exposed_source_downstream_judge_readability_calibration",
        "plan_dependency": preflight, "p0_pre_fit_gate": {"pass": False},
        "consumer_fit_count": 0, "expected_consumer_fit_count": EXPECTED_FITS,
        "dataset_evidence": [], "judge_readability_gate": {"pass": False},
        "classification": "JUDGE_READABILITY_PREFLIGHT_OR_P0_FAIL",
        "e2_ready": False,
        "verdict": "JUDGE_READABILITY_CALIBRATION_FAIL",
        "capability_claim": False, "pattern_claim": False, "memory_claim": False,
        "promotion_eligible": False, "formal_transfer": False,
        "information_wall": {"complete_frozen_metadata_read": True,
            "selected_exposed_source_values_read": False,
            "support_b_values_read": False, "uci_values_read": False,
            "target_values_read": False, "query_values_read": False,
            "target_closed": True, "query_closed": True,
            "target_query_opened": False},
        "target_query_opened": False,
        "claim_limit": "Preflight failed; no fit and no Judge, Capability, Pattern, Memory, promotion, transfer, Target, or Query claim."}


def evaluate(root: Path, plan: dict[str, Any], preflight: dict[str, object]) -> dict[str, object]:
    if not bool(preflight["pass"]):
        return _finish_failure(preflight)
    runtime_versions: dict[str, str | None] = {}
    for package in EXPECTED_RUNTIME_VERSIONS:
        try:
            runtime_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime_versions[package] = None
    runtime_matches = runtime_versions == EXPECTED_RUNTIME_VERSIONS
    if not runtime_matches:
        report = _finish_failure(preflight)
        report.update({"runtime_version_check": {
                "expected": EXPECTED_RUNTIME_VERSIONS,
                "actual": runtime_versions, "pass": False},
            "classification": "IMPLEMENTATION_FAIL", "verdict": "IMPLEMENTATION_FAIL",
            "claim_limit": "Runtime version pin mismatch; no fit or response evidence."})
        return report
    import numpy as np
    from sklearn.linear_model import Ridge
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale, seasonal_scale, smase)
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import _center_scale

    records = {row.series_uid: row for row in read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl")}
    roster = plan["roster"]
    selected_records = [records[str(row["series_uid"])] for row in roster]
    values = _load_values(selected_records, root / "data/benchmark_v0_2/clean_base")
    consumer_fit_count = 0
    prepared: dict[str, dict[str, Any]] = {}
    p0_by_dataset: dict[str, object] = {}
    for dataset_id, spec in SPECS.items():
        train = [row for row in roster if row["dataset_id"] == dataset_id
                 and row["cohort"] == "train"]
        evaluate_rows = [row for row in roster if row["dataset_id"] == dataset_id
                         and row["cohort"] == "eval"]
        x_rows, target_rows, keys = [], [], []
        dataset_sha = str(plan["roster_audit"][dataset_id]["dataset_sha"])
        scale_floor_count = 0
        for anchor in ANCHORS:
            for row in train:
                uid, raw = str(row["series_uid"]), values[str(row["series_uid"])]
                context = np.asarray(raw[anchor-CONTEXT_LENGTH:anchor], dtype=np.float64)
                target = np.asarray(raw[anchor:anchor+HORIZON], dtype=np.float64)
                if context.shape != (CONTEXT_LENGTH,) or target.shape != (HORIZON,):
                    raise ValueError(f"invalid training window: {uid}/{anchor}")
                if not np.isfinite(context).all() or not np.isfinite(target).all():
                    raise ValueError(f"non-finite training window: {uid}/{anchor}")
                center, scale, method = _center_scale(context)
                scale_floor_count += int(method == "scale_floor_fallback")
                x_rows.append(np.concatenate(((context-center)/scale,
                                              np.zeros(CONTEXT_LENGTH))))
                target_rows.append((target-center)/scale)
                keys.append((dataset_sha, uid, anchor, HORIZON))
        x_train, clean_y = np.asarray(x_rows), np.asarray(target_rows)
        if x_train.shape != (72, 384) or clean_y.shape != (72, HORIZON):
            raise AssertionError(f"unexpected training geometry: {dataset_id}")
        x_before = x_train.copy()
        corrupt_targets: dict[tuple[int, int], Any] = {}
        canonical_repaired = None
        intervention_checks: list[dict[str, object]] = []
        selected_sets: dict[int, list[set[int]]] = {seed: [] for seed in SEEDS}
        for seed in SEEDS:
            order = _row_order(dataset_sha, seed, keys)
            for count in SELECTED_ROW_COUNTS:
                selected_indices = set(order[:count])
                selected_sets[seed].append(selected_indices)
                selected = sorted(selected_indices)
                unselected = sorted(set(range(72)) - selected_indices)
                corrupt = clean_y.copy()
                corrupt[selected, TARGET_BLOCK[0]:TARGET_BLOCK[1]] = (
                    clean_y[selected, TARGET_BLOCK[0]:TARGET_BLOCK[1]] + TARGET_OFFSET)
                repaired = corrupt.copy()
                repaired[selected, TARGET_BLOCK[0]:TARGET_BLOCK[1]] = (
                    clean_y[selected, TARGET_BLOCK[0]:TARGET_BLOCK[1]])
                selected_count_exact = len(selected_indices) == count
                block_delta_exact = bool(np.array_equal(
                    corrupt[selected, TARGET_BLOCK[0]:TARGET_BLOCK[1]],
                    clean_y[selected, TARGET_BLOCK[0]:TARGET_BLOCK[1]] + TARGET_OFFSET))
                unselected_unchanged = bool(np.array_equal(
                    corrupt[unselected], clean_y[unselected]))
                block_exterior_unchanged = bool(np.array_equal(
                    corrupt[:, :TARGET_BLOCK[0]], clean_y[:, :TARGET_BLOCK[0]])
                    and np.array_equal(corrupt[:, TARGET_BLOCK[1]:],
                                       clean_y[:, TARGET_BLOCK[1]:]))
                exact = bool(np.array_equal(repaired, clean_y))
                intervention_checks.append({"seed": seed, "selected_row_count": count,
                    "selected_row_keys": [list(keys[index]) for index in selected],
                    "selected_row_count_exact": selected_count_exact,
                    "selected_block_additive_offset_exact": block_delta_exact,
                    "unselected_targets_unchanged": unselected_unchanged,
                    "target_block_exterior_unchanged": block_exterior_unchanged,
                    "exact_repair_matches_clean": exact})
                corrupt_targets[(seed, count)] = corrupt
                if seed == 0 and count == max(SELECTED_ROW_COUNTS):
                    canonical_repaired = repaired
        nested = all(sets[0] < sets[1] < sets[2] for sets in selected_sets.values())
        if canonical_repaired is None:
            raise AssertionError("canonical exact repair was not constructed")

        x_eval, raw_future, eval_uids, centers, eval_scales = [], [], [], [], []
        seasonal_by_uid: dict[str, float] = {}
        invalid_scale_uids: list[str] = []
        nonfinite_eval_uids: list[str] = []
        eval_floor_count = 0
        for row in evaluate_rows:
            uid, raw = str(row["series_uid"]), values[str(row["series_uid"])]
            context = np.asarray(raw[spec["train_stop"]-CONTEXT_LENGTH:spec["train_stop"]],
                                 dtype=np.float64)
            future = np.asarray(raw[slice(*spec["future_bounds"])], dtype=np.float64)
            if context.shape != (CONTEXT_LENGTH,) or future.shape != (HORIZON,):
                raise ValueError(f"invalid evaluation window: {uid}")
            if not np.isfinite(context).all() or not np.isfinite(future).all():
                nonfinite_eval_uids.append(uid)
                continue
            center, scale, method = _center_scale(context)
            eval_floor_count += int(method == "scale_floor_fallback")
            x_eval.append(np.concatenate(((context-center)/scale,
                                          np.zeros(CONTEXT_LENGTH))))
            raw_future.append(future); eval_uids.append(uid)
            centers.append(center); eval_scales.append(scale)
            clean_train = np.asarray(raw[:spec["train_stop"]], dtype=np.float64)
            try:
                seasonal_by_uid[uid] = seasonal_scale(clean_train,
                    np.isfinite(clean_train), period=spec["period"], min_pairs=32)
            except (UndefinedSeasonalScale, ValueError):
                invalid_scale_uids.append(uid)
        p0_pass = (consumer_fit_count == 0 and scale_floor_count + eval_floor_count == 0
            and not invalid_scale_uids and not nonfinite_eval_uids
            and nested and np.array_equal(x_train, x_before)
            and np.array_equal(canonical_repaired, clean_y)
            and all(bool(row["selected_row_count_exact"])
                    and bool(row["selected_block_additive_offset_exact"])
                    and bool(row["unselected_targets_unchanged"])
                    and bool(row["target_block_exterior_unchanged"])
                    and bool(row["exact_repair_matches_clean"])
                    for row in intervention_checks))
        p0_by_dataset[dataset_id] = {"checked_before_any_fit": True,
            "consumer_fit_count_at_check": consumer_fit_count,
            "training_row_count": 72,
            "features_unchanged": bool(np.array_equal(x_train, x_before)),
            "clean_repaired_target_equality": bool(np.array_equal(
                canonical_repaired, clean_y)),
            "scale_floor_count": scale_floor_count + eval_floor_count,
            "eval_seasonal_scale_valid_count": len(seasonal_by_uid),
            "eval_seasonal_scale_invalid_uids": invalid_scale_uids,
            "eval_context_and_future_finite": not nonfinite_eval_uids,
            "eval_nonfinite_context_or_future_uids": nonfinite_eval_uids,
            "higher_doses_strictly_contain_lower_selected_rows": nested,
            "intervention_checks": intervention_checks, "pass": p0_pass}
        prepared[dataset_id] = {"x_train": x_train, "clean_y": clean_y,
            "repaired_y": canonical_repaired, "corrupt_targets": corrupt_targets,
            "x_eval": np.asarray(x_eval), "raw_future": np.asarray(raw_future),
            "eval_uids": eval_uids, "centers": np.asarray(centers),
            "eval_scales": np.asarray(eval_scales), "seasonal": seasonal_by_uid,
            "runtime_versions": runtime_versions}
    p0_pass = all(bool(row["pass"]) for row in p0_by_dataset.values())
    if not p0_pass:
        report = _finish_failure(preflight)
        report["p0_pre_fit_gate"] = {"dataset_results": p0_by_dataset,
                                     "failure_action": "zero-fit stop", "pass": False}
        report["information_wall"]["selected_exposed_source_values_read"] = True
        return report

    def fit_score(dataset_id: str, bundle: dict[str, Any],
                  targets: Any) -> tuple[list[dict[str, object]], Any]:
        nonlocal consumer_fit_count
        uids = bundle["eval_uids"]
        model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
        model.fit(bundle["x_train"], targets)
        consumer_fit_count += 1
        normalized = np.asarray(model.predict(bundle["x_eval"]), dtype=np.float64)
        if normalized.shape != (8, HORIZON) or not np.isfinite(normalized).all():
            raise RuntimeError(f"invalid Ridge prediction: {dataset_id}")
        original = normalized * bundle["eval_scales"][:, None] + bundle["centers"][:, None]
        rows = []
        for index, uid in enumerate(uids):
            mae = float(np.mean(np.abs(original[index]-bundle["raw_future"][index])))
            loss = smase(bundle["raw_future"][index], original[index],
                         scale=bundle["seasonal"][uid])
            rows.append({"series_uid": uid, "original_unit_mae": mae, "smase": loss})
        return rows, original

    controls: dict[str, dict[str, Any]] = {}
    control_evidence: list[dict[str, object]] = []
    for dataset_id in SPECS:
        bundle = prepared[dataset_id]
        uids = bundle["eval_uids"]
        clean_rows, clean_predictions = fit_score(dataset_id, bundle, bundle["clean_y"])
        exact_rows, exact_predictions = fit_score(dataset_id, bundle, bundle["repaired_y"])
        determinism_rows = []
        for index, uid in enumerate(uids):
            determinism_rows.append({"series_uid": uid,
                "prediction_max_abs_difference": float(np.max(np.abs(
                    clean_predictions[index]-exact_predictions[index]))),
                "original_unit_mae_abs_difference": abs(
                    clean_rows[index]["original_unit_mae"]-exact_rows[index]["original_unit_mae"]),
                "smase_abs_difference": abs(clean_rows[index]["smase"]-exact_rows[index]["smase"])})
        deterministic = all(max(row["prediction_max_abs_difference"],
            row["original_unit_mae_abs_difference"], row["smase_abs_difference"])
            <= PREDICTION_TOLERANCE for row in determinism_rows)
        controls[dataset_id] = {"clean_rows": clean_rows, "exact_rows": exact_rows,
            "determinism_rows": determinism_rows, "deterministic": deterministic}
        control_evidence.append({"dataset_id": dataset_id,
            "clean_reference_scores": clean_rows,
            "exact_repaired_repeat_scores": exact_rows,
            "clean_exact_repeat_determinism": {"tolerance": PREDICTION_TOLERANCE,
                "per_uid": determinism_rows, "pass": deterministic}})
    if consumer_fit_count != 4:
        raise AssertionError("control stage must perform exactly four fits")
    controls_deterministic = all(bool(row["deterministic"])
                                 for row in controls.values())
    if not controls_deterministic:
        return {"schema_version": SCHEMA_VERSION, "phase": "evaluate",
            "scientific_role": "exposed_source_downstream_judge_readability_calibration",
            "plan_dependency": preflight,
            "runtime_version_check": {"expected": EXPECTED_RUNTIME_VERSIONS,
                "actual": runtime_versions, "pass": True},
            "configuration": plan["configuration"], "instrument": plan["instrument"],
            "p0_pre_fit_gate": {"dataset_results": p0_by_dataset,
                "conjunction_across_both_datasets": True, "pass": True},
            "control_evidence": control_evidence,
            "control_stage_gate": {"both_datasets_deterministic": False,
                "failure_action": "stop before corrupt fits", "pass": False},
            "consumer_fit_count": 4, "corrupt_fit_count": 0,
            "expected_consumer_fit_count": EXPECTED_FITS,
            "dataset_evidence": [], "response_evidence_produced": False,
            "judge_readability_gate": {"pass": False},
            "classification": "IMPLEMENTATION_FAIL", "e2_ready": False,
            "verdict": "IMPLEMENTATION_FAIL",
            "capability_claim": False, "pattern_claim": False, "memory_claim": False,
            "promotion_eligible": False, "formal_transfer": False,
            "information_wall": {"complete_frozen_metadata_read": True,
                "selected_exposed_source_values_read": True,
                "support_b_values_read": False, "uci_values_read": False,
                "target_values_read": False, "query_values_read": False,
                "target_closed": True, "query_closed": True,
                "target_query_opened": False},
            "target_query_opened": False,
            "claim_limit": "Control mismatch is an implementation failure; no corrupt response evidence or scientific claim."}

    corrupt_fit_count = 0
    dataset_evidence: list[dict[str, object]] = []
    for dataset_id in SPECS:
        bundle = prepared[dataset_id]
        uids = bundle["eval_uids"]
        clean_rows = controls[dataset_id]["clean_rows"]
        exact_rows = controls[dataset_id]["exact_rows"]
        determinism_rows = controls[dataset_id]["determinism_rows"]
        deterministic = bool(controls[dataset_id]["deterministic"])
        corrupt_scores: dict[tuple[int, int], list[dict[str, object]]] = {}
        for seed in SEEDS:
            for count in SELECTED_ROW_COUNTS:
                rows, _ = fit_score(dataset_id, bundle,
                                    bundle["corrupt_targets"][(seed, count)])
                corrupt_fit_count += 1
                corrupt_scores[(seed, count)] = rows
        exact_by_uid = {str(row["series_uid"]): row for row in exact_rows}
        dose_response = []
        dose_mean_degradations: list[float] = []
        for nominal_dose, count in zip(NOMINAL_DOSES, SELECTED_ROW_COUNTS):
            per_uid = []
            for index, uid in enumerate(uids):
                smase_values = [float(corrupt_scores[(seed, count)][index]["smase"])
                                for seed in SEEDS]
                mae_values = [float(corrupt_scores[(seed, count)][index][
                    "original_unit_mae"]) for seed in SEEDS]
                smase_deltas = [value-float(exact_by_uid[uid]["smase"])
                                for value in smase_values]
                mae_deltas = [value-float(exact_by_uid[uid]["original_unit_mae"])
                              for value in mae_values]
                seed_mean_smase = statistics.fmean(smase_values)
                seed_mean_mae = statistics.fmean(mae_values)
                per_uid.append({"series_uid": uid,
                    "seed_mean_smase": seed_mean_smase,
                    "exact_repaired_smase": float(exact_by_uid[uid]["smase"]),
                    "per_seed_smase_degradation_corrupt_minus_exact": smase_deltas,
                    "seed_mean_smase_degradation": statistics.fmean(smase_deltas),
                    "seed_mean_original_unit_mae": seed_mean_mae,
                    "exact_repaired_original_unit_mae": float(
                        exact_by_uid[uid]["original_unit_mae"]),
                    "per_seed_original_unit_mae_degradation_corrupt_minus_exact": mae_deltas,
                    "seed_mean_original_unit_mae_degradation": statistics.fmean(mae_deltas)})
            mean_degradation = statistics.fmean(float(row[
                "seed_mean_smase_degradation"]) for row in per_uid)
            dose_mean_degradations.append(mean_degradation)
            dose_response.append({"nominal_dose": nominal_dose,
                "selected_row_count": count, "realized_dose": count/72,
                "first_average_over_seeds_per_uid": True,
                "per_uid": per_uid,
                "mean_smase_degradation_over_uids": mean_degradation})
        endpoint_rows = dose_response[-1]["per_uid"]
        endpoint_smase = {str(row["series_uid"]): float(
            row["seed_mean_smase_degradation"]) for row in endpoint_rows}
        clusters = {str(row["series_uid"]): str(row["overlap_group"])
                    for row in roster if row["dataset_id"] == dataset_id and row["cohort"] == "eval"}
        power_row = _one_sided_bootstrap(np, endpoint_smase, clusters)
        dose_trend = (0.0 < dose_mean_degradations[0]
                      <= dose_mean_degradations[1]
                      <= dose_mean_degradations[2])
        endpoint_mean = statistics.fmean(endpoint_smase.values())
        endpoint_positive_count = sum(value > 0 for value in endpoint_smase.values())
        gates = {"scale_valid_8_of_8": len(bundle["seasonal"]) == 8,
            "clean_exact_repeat_deterministic": deterministic,
            "endpoint_mean_smase_degradation_gte_0_02": endpoint_mean >= DEGRADATION_MIN,
            "endpoint_one_sided_cluster_bootstrap_q05_gt_0": (
                float(power_row["bootstrap_q05"]) > 0.0),
            "endpoint_positive_uid_count_at_least_6": endpoint_positive_count >= 6,
            "strict_dose_trend": dose_trend}
        response_pass = all(gates.values())
        mde_ready = float(power_row["mde80_one_sided"]) <= DEGRADATION_MIN
        dataset_e2_ready = response_pass and mde_ready
        dataset_evidence.append({"dataset_id": dataset_id,
            "eval_seasonal_scale_by_uid": bundle["seasonal"],
            "clean_reference_scores": clean_rows,
            "exact_repaired_repeat_scores": exact_rows,
            "clean_exact_repeat_determinism": {"tolerance": PREDICTION_TOLERANCE,
                "per_uid": determinism_rows, "pass": deterministic},
            "corrupt_scores": [{"seed": seed, "nominal_dose": nominal_dose,
                "selected_row_count": count, "realized_dose": count/72,
                "per_uid": corrupt_scores[(seed, count)]}
                for seed in SEEDS
                for nominal_dose, count in zip(NOMINAL_DOSES, SELECTED_ROW_COUNTS)],
            "endpoint_seed_mean_per_uid": endpoint_rows,
            "endpoint_mean_smase_degradation": endpoint_mean,
            "endpoint_positive_uid_count": endpoint_positive_count,
            "endpoint_one_sided_overlap_group_bootstrap": power_row,
            "dose_response": {"rows": dose_response,
                "mean_smase_degradations": dose_mean_degradations,
                "required_order": "0 < mean(d05) <= mean(d10) <= mean(d20)",
                "strict_order_pass": dose_trend},
            "dataset_gate": {"conditions": gates, "conjunctive": True,
                "response_pass": response_pass,
                "mde80_one_sided_lte_epsilon": mde_ready,
                "e2_ready": dataset_e2_ready}})
    if consumer_fit_count != EXPECTED_FITS or corrupt_fit_count != 18:
        raise AssertionError(f"expected exactly {EXPECTED_FITS} independent Ridge fits")
    response_passed = all(bool(row["dataset_gate"]["response_pass"])
                          for row in dataset_evidence)
    e2_ready = response_passed and all(bool(row["dataset_gate"]["e2_ready"])
                                       for row in dataset_evidence)
    if e2_ready:
        classification = "READABLE_AND_POWERED_FOR_EPSILON"
    elif response_passed:
        classification = "READABLE_AT_INJECTED_DOSE_BUT_UNDERPOWERED_FOR_EPSILON"
    else:
        classification = "JUDGE_READABILITY_RESPONSE_GATE_FAIL"
    return {"schema_version": SCHEMA_VERSION, "phase": "evaluate",
        "scientific_role": "exposed_source_downstream_judge_readability_calibration",
        "plan_dependency": preflight,
        "runtime_version_check": {"expected": EXPECTED_RUNTIME_VERSIONS,
            "actual": runtime_versions, "pass": True},
        "configuration": plan["configuration"], "instrument": plan["instrument"],
        "p0_pre_fit_gate": {"dataset_results": p0_by_dataset,
            "conjunction_across_both_datasets": True, "failure_action": "zero-fit stop",
            "pass": p0_pass},
        "consumer_fit_count": consumer_fit_count,
        "control_fit_count": 4,
        "corrupt_fit_count": corrupt_fit_count,
        "expected_consumer_fit_count": EXPECTED_FITS,
        "control_evidence": control_evidence,
        "control_stage_gate": {"both_datasets_deterministic": True,
            "corrupt_fits_authorized": True, "pass": True},
        "dataset_evidence": dataset_evidence,
        "response_evidence_produced": True,
        "judge_readability_gate": {"datasets_conjunctive": True,
            "response_pass_across_both_datasets": response_passed,
            "mde80_one_sided_lte_epsilon_required_for_e2_ready": True,
            "e2_ready": e2_ready, "pass": e2_ready},
        "classification": classification,
        "e2_ready": e2_ready,
        "verdict": "JUDGE_READABILITY_CALIBRATION_PASS" if e2_ready
                   else "JUDGE_READABILITY_CALIBRATION_FAIL",
        "capability_claim": False, "pattern_claim": False, "memory_claim": False,
        "promotion_eligible": False, "formal_transfer": False,
        "information_wall": {"complete_frozen_metadata_read": True,
            "selected_exposed_source_values_read": True,
            "support_b_values_read": False, "uci_values_read": False,
            "target_values_read": False, "query_values_read": False,
            "target_closed": True, "query_closed": True,
            "target_query_opened": False},
        "target_query_opened": False,
        "claim_limit": "Exposed-Source downstream Judge calibration only; not Capability, Pattern, Memory, promotion, transfer, Target, or Query evidence."}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("plan", "evaluate"))
    parser.add_argument("--plan-report", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "plan":
        report, output = build_plan(root), args.output or root / PLAN_RELATIVE_PATH
    else:
        if args.plan_report is None or args.expected_plan_sha256 is None:
            parser.error("evaluate requires --plan-report and --expected-plan-sha256")
        plan, preflight = _verify_plan(root, args.plan_report, args.expected_plan_sha256)
        report, output = evaluate(root, plan, preflight), args.output or root / REPORT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(output)
    print(report.get("verdict", report.get("plan_status")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
