"""Replay the fixed Linear cohort-policy effect on fresh Source validation series.

This Source-only replay reuses the exact development-premise training cohorts and
consumer protocol.  It changes only the evaluated series cohort: sixteen unexposed
``support_a_validation`` series per Source dataset, selected from registry metadata
before any selected values or futures are loaded.  It is not a Capability, Memory,
Agent, Target, Query, adaptation, or transfer evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
    SeriesRecord,
    read_registry_jsonl,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.split import (
    SplitAssignment,
    SplitManifest,
    SplitRole,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    CONTEXT_LENGTH,
    EVAL_CONTEXT_BOUNDS,
    EVAL_FUTURE_BOUNDS,
    GAP_BOUNDS,
    HARM_GAIN_THRESHOLD,
    HORIZON,
    MATERIAL_GAIN_MIN,
    RIDGE_ALPHA,
    ROBUST_SCALE_FLOOR,
    RosterItem,
    SOURCE_DATASETS,
    SUPPORT_A_SUBSPLIT_SCHEMA,
    TRAIN_ANCHORS,
    TRAIN_SERIES_PER_DATASET,
    _center_scale,
    _load_roster_values,
    _training_matrices,
    _validate_candidate,
)


SCHEMA_VERSION = "e2-source-cohort-policy-fresh-replay/1"
SCIENTIFIC_ROLE = "fresh_source_series_cohort_policy_direction_safety_replay"
PREMISE_SCHEMA_VERSION = "e2-source-cohort-policy-premise/1"
DISCOVERY_SUBSPLIT = "support_a_discovery"
VALIDATION_SUBSPLIT = "support_a_validation"
POLICIES = ("identity_minimal", "linear")

EVAL_SERIES_PER_DATASET = 16
ROSTER_SALT = "e2-source-cohort-fresh-replay-v1"
HARM_RATE_MAX = 0.25

OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_cohort_policy_fresh_replay_report.json"
)
PREMISE_REPORT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_cohort_policy_premise_report.json"
)

# This tuple is deliberately explicit and frozen.  Do not replace it with a glob:
# a glob could make a replay depend on its own output or on later artifacts.
EXPOSURE_REPORT_RELATIVE_PATHS = (
    "artifacts/functional/e2/natural_binding_diagnostic_report.json",
    "artifacts/functional/e2/natural_gap_geometry_sweep_report.json",
    "artifacts/functional/e2/natural_periodic_missing_headroom_report.json",
    "artifacts/functional/e2/natural_pseudogap_observation_report.json",
    "artifacts/functional/e2/natural_scope_operator_diagnostic_report.json",
    "artifacts/functional/e2/natural_short_gap_source_replay_report.json",
    "artifacts/functional/e2/natural_source_evidence_report.json",
    "artifacts/functional/e2/natural_source_promotion_report.json",
    PREMISE_REPORT_RELATIVE_PATH,
)

if OUTPUT_RELATIVE_PATH in EXPOSURE_REPORT_RELATIVE_PATHS:
    raise AssertionError("fresh replay output must not be an exposure input")


def _read_subsplit_members(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema_version") != SUPPORT_A_SUBSPLIT_SCHEMA:
        raise ValueError("unsupported Support-A subsplit schema")
    members = payload.get("members")
    counts = payload.get("counts")
    if not isinstance(members, dict) or not isinstance(counts, dict):
        raise ValueError("Support-A subsplit members/counts must be objects")

    result: dict[str, set[str]] = {}
    for name in (DISCOVERY_SUBSPLIT, VALIDATION_SUBSPLIT):
        raw = members.get(name)
        if not isinstance(raw, list) or not all(
            isinstance(uid, str) and uid for uid in raw
        ):
            raise ValueError(f"invalid {name} member list")
        if len(raw) != len(set(raw)):
            raise ValueError(f"duplicate UID in {name}")
        if counts.get(name) != len(raw):
            raise ValueError(f"{name} count disagrees with frozen metadata")
        result[name] = set(raw)
    if result[DISCOVERY_SUBSPLIT] & result[VALIDATION_SUBSPLIT]:
        raise ValueError("Support-A discovery and validation members overlap")
    return result


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)


def _read_exposure_uids(
    *, project_root: Path, registry_uids: set[str]
) -> tuple[set[str], dict[str, object]]:
    exposed: set[str] = set()
    counts_by_path: dict[str, int] = {}
    for relative_path in EXPOSURE_REPORT_RELATIVE_PATHS:
        path = project_root / relative_path
        payload = json.loads(path.read_text("utf-8"))
        report_uids = set(_iter_strings(payload)) & registry_uids
        exposed.update(report_uids)
        counts_by_path[relative_path] = len(report_uids)
    return exposed, {
        "path_list_is_explicit_and_frozen": True,
        "glob_used": False,
        "fresh_replay_output_included": False,
        "report_paths": list(EXPOSURE_REPORT_RELATIVE_PATHS),
        "registry_uid_count_by_report": counts_by_path,
        "unique_registry_uid_count": len(exposed),
        "scan_scope": (
            "registry UID strings only; no report outcome is used for ranking, "
            "training, fitting, gating, or evaluation"
        ),
    }


def _premise_train_uids(project_root: Path) -> dict[str, list[str]]:
    path = project_root / PREMISE_REPORT_RELATIVE_PATH
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema_version") != PREMISE_SCHEMA_VERSION:
        raise ValueError("unsupported cohort-policy premise report schema")
    premise_gate = payload.get("premise_gate")
    if not isinstance(premise_gate, dict) or premise_gate.get("selected_policy") != "linear":
        raise ValueError("development premise does not freeze Linear as its candidate")
    selection = payload.get("roster_selection")
    selected = selection.get("selected_by_dataset") if isinstance(selection, dict) else None
    if not isinstance(selected, dict):
        raise ValueError("development premise lacks its frozen roster")

    by_dataset: dict[str, list[str]] = {}
    for dataset_id in SOURCE_DATASETS:
        dataset_selection = selected.get(dataset_id)
        raw = dataset_selection.get("train") if isinstance(dataset_selection, dict) else None
        if not isinstance(raw, list) or not all(
            isinstance(uid, str) and uid for uid in raw
        ):
            raise ValueError(f"invalid premise train roster: {dataset_id}")
        if len(raw) != TRAIN_SERIES_PER_DATASET or len(raw) != len(set(raw)):
            raise ValueError(f"premise train roster must contain 32 unique UIDs: {dataset_id}")
        by_dataset[dataset_id] = list(raw)
    if set(by_dataset[SOURCE_DATASETS[0]]) & set(by_dataset[SOURCE_DATASETS[1]]):
        raise ValueError("premise train rosters overlap across datasets")
    return by_dataset


def _metadata_rank(record: SeriesRecord) -> str:
    payload = f"{ROSTER_SALT}\0{record.dataset_id}\0{record.entity_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_roster(
    *,
    project_root: Path,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
) -> tuple[list[RosterItem], dict[str, object]]:
    """Freeze the exact premise train cohort and fresh eval cohort from metadata."""

    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    assignments = {row.series_uid: row for row in manifest.assignments}
    subsplits = _read_subsplit_members(support_a_subsplit_path)
    train_uids_by_dataset = _premise_train_uids(project_root)
    exposed_uids, exposure_report = _read_exposure_uids(
        project_root=project_root, registry_uids=set(records)
    )

    roster: list[RosterItem] = []
    train_items_by_dataset: dict[str, list[RosterItem]] = {}
    for dataset_id in SOURCE_DATASETS:
        items: list[RosterItem] = []
        for uid in train_uids_by_dataset[dataset_id]:
            if uid not in subsplits[DISCOVERY_SUBSPLIT]:
                raise ValueError(f"premise train UID is not discovery: {uid}")
            record = records.get(uid)
            assignment = assignments.get(uid)
            if record is None or assignment is None:
                raise ValueError(f"premise train UID missing from frozen metadata: {uid}")
            if record.dataset_id != dataset_id:
                raise ValueError(f"premise train UID changed dataset: {uid}")
            _validate_candidate(record, assignment)
            items.append(RosterItem(record, assignment, "train"))
        train_items_by_dataset[dataset_id] = items
        roster.extend(items)

    raw_available_by_dataset: dict[str, int] = {}
    available_after_exposure_by_dataset: dict[str, int] = {}
    selected_by_dataset: dict[str, list[dict[str, str]]] = {}
    eval_items_by_dataset: dict[str, list[RosterItem]] = {}
    for dataset_id in SOURCE_DATASETS:
        candidates: list[tuple[SeriesRecord, SplitAssignment]] = []
        for uid in subsplits[VALIDATION_SUBSPLIT]:
            assignment = assignments.get(uid)
            if assignment is None:
                raise ValueError(f"validation UID absent from split manifest: {uid}")
            if assignment.dataset_id != dataset_id:
                continue
            record = records.get(uid)
            if record is None:
                raise ValueError(f"validation UID absent from registry: {uid}")
            _validate_candidate(record, assignment)
            candidates.append((record, assignment))

        raw_available_by_dataset[dataset_id] = len(candidates)
        fresh = [item for item in candidates if item[0].series_uid not in exposed_uids]
        available_after_exposure_by_dataset[dataset_id] = len(fresh)
        if len(fresh) < EVAL_SERIES_PER_DATASET:
            raise ValueError(f"fewer than 16 fresh validation series: {dataset_id}")
        if len({item[0].entity_id for item in fresh}) != len(fresh):
            raise ValueError(f"duplicate validation entity_id: {dataset_id}")
        ranked = sorted(fresh, key=lambda item: (_metadata_rank(item[0]), item[0].entity_id))
        selected = ranked[:EVAL_SERIES_PER_DATASET]
        items = [RosterItem(record, assignment, "eval") for record, assignment in selected]
        eval_items_by_dataset[dataset_id] = items
        roster.extend(items)
        selected_by_dataset[dataset_id] = [
            {
                "entity_id": item.record.entity_id,
                "series_uid": item.record.series_uid,
                "overlap_group": item.record.overlap_group,
                "selection_sha256": _metadata_rank(item.record),
            }
            for item in items
        ]

    train_uids = {item.record.series_uid for items in train_items_by_dataset.values() for item in items}
    eval_uids = {item.record.series_uid for items in eval_items_by_dataset.values() for item in items}
    if len(train_uids) != len(SOURCE_DATASETS) * TRAIN_SERIES_PER_DATASET:
        raise AssertionError("training roster must contain 64 unique premise series")
    if len(eval_uids) != len(SOURCE_DATASETS) * EVAL_SERIES_PER_DATASET:
        raise AssertionError("evaluation roster must contain 32 unique fresh series")
    if train_uids & eval_uids:
        raise AssertionError("training and evaluation series overlap")
    if eval_uids & exposed_uids:
        raise AssertionError("fresh evaluation roster contains an exposed registry UID")

    overlap_audit_by_dataset: dict[str, dict[str, object]] = {}
    for dataset_id in SOURCE_DATASETS:
        train_groups = {
            item.record.overlap_group for item in train_items_by_dataset[dataset_id]
        }
        eval_groups = {
            item.record.overlap_group for item in eval_items_by_dataset[dataset_id]
        }
        overlapping_groups = sorted(train_groups & eval_groups)
        if overlapping_groups:
            raise ValueError(
                "premise training and fresh validation rosters share overlap_group "
                f"metadata in {dataset_id}: " + ", ".join(overlapping_groups)
            )
        overlap_audit_by_dataset[dataset_id] = {
            "train_overlap_group_count": len(train_groups),
            "eval_overlap_group_count": len(eval_groups),
            "shared_overlap_groups": [],
            "disjoint": True,
        }

    return roster, {
        "fixed_before_selected_value_or_future_loading": True,
        "training_roster_source": PREMISE_REPORT_RELATIVE_PATH,
        "training_roster_exactly_reuses_development_premise_uids": True,
        "train_selected_uid_count_by_dataset": {
            dataset_id: len(train_uids_by_dataset[dataset_id])
            for dataset_id in SOURCE_DATASETS
        },
        "eval_subsplit": VALIDATION_SUBSPLIT,
        "eval_selection_rule": (
            "exclude registry UIDs found in the explicit frozen E2 exposure report list; "
            "within each Source dataset sort remaining support_a_validation records by "
            "SHA256(salt + NUL + dataset_id + NUL + entity_id), using entity_id as "
            "the final tie-break, then take the first 16"
        ),
        "selection_salt": ROSTER_SALT,
        "selection_uses_entity_metadata": True,
        "selection_is_content_independent": True,
        "selection_is_non_outcome_adaptive": True,
        "series_uid_used_only_as_reference": True,
        "raw_validation_available_by_dataset": raw_available_by_dataset,
        "available_after_exposure_exclusion_by_dataset": available_after_exposure_by_dataset,
        "selected_eval_count_by_dataset": {
            dataset_id: len(eval_items_by_dataset[dataset_id])
            for dataset_id in SOURCE_DATASETS
        },
        "selected_eval_by_dataset": selected_by_dataset,
        "train_eval_series_disjoint": True,
        "train_eval_overlap_group_disjoint": True,
        "overlap_group_audit_by_dataset": overlap_audit_by_dataset,
        "exposure": exposure_report,
    }


def _evaluation_matrices(
    items: list[RosterItem], values_by_uid: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    uids: list[str] = []
    scale_methods: dict[str, int] = {}
    for item in items:
        uid = item.record.series_uid
        values = values_by_uid[uid]
        context = np.asarray(values[slice(*EVAL_CONTEXT_BOUNDS)], dtype=np.float64).copy()
        future = np.asarray(values[slice(*EVAL_FUTURE_BOUNDS)], dtype=np.float64).copy()
        if context.shape != (CONTEXT_LENGTH,) or future.shape != (HORIZON,):
            raise ValueError(f"insufficient evaluation window: {uid}")
        if not np.isfinite(context).all() or not np.isfinite(future).all():
            raise ValueError(f"natural missingness enters evaluation window: {uid}")
        center, scale, scale_method = _center_scale(context)
        normalized = (context - center) / scale
        features = np.concatenate((normalized, np.zeros(CONTEXT_LENGTH, dtype=np.float64)))
        target = (future - center) / scale
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError(f"non-finite normalized evaluation data: {uid}")
        x_rows.append(features)
        y_rows.append(target)
        uids.append(uid)
        scale_methods[scale_method] = scale_methods.get(scale_method, 0) + 1
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    if x.shape != (EVAL_SERIES_PER_DATASET, 384) or y.shape != (
        EVAL_SERIES_PER_DATASET,
        HORIZON,
    ):
        raise AssertionError("unexpected fresh cohort evaluation matrix shape")
    return x, y, uids, {"scale_method_counts": scale_methods}


def _cohort_evidence(
    *,
    dataset_id: str,
    identity_losses: list[float],
    linear_losses: list[float],
    eval_uids: list[str],
    training_diagnostics: dict[str, dict[str, object]],
    evaluation_diagnostics: dict[str, object],
) -> dict[str, object]:
    if not (len(identity_losses) == len(linear_losses) == len(eval_uids)):
        raise AssertionError("paired cohort evidence lengths disagree")
    paired: list[dict[str, object]] = []
    for uid, identity_loss, linear_loss in zip(eval_uids, identity_losses, linear_losses):
        gain = identity_loss - linear_loss
        paired.append(
            {
                "diagnostic_role": "paired_series_diagnostic_not_causal_evidence",
                "series_uid": uid,
                "identity_minimal_normalized_mae": identity_loss,
                "linear_normalized_mae": linear_loss,
                "linear_gain_over_identity": gain,
                "harmed": gain < HARM_GAIN_THRESHOLD,
                "material_gain": gain >= MATERIAL_GAIN_MIN,
            }
        )
    identity_mean = statistics.fmean(identity_losses)
    linear_mean = statistics.fmean(linear_losses)
    mean_gain = statistics.fmean(float(row["linear_gain_over_identity"]) for row in paired)
    median_gain = statistics.median(
        float(row["linear_gain_over_identity"]) for row in paired
    )
    if identity_mean <= 0.0:
        raise ValueError("identity mean normalized MAE must be positive")
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "scientific_unit": "dataset_level_fresh_series_cohort",
        "dataset_id": dataset_id,
        "policy_contrast": "linear_vs_identity_minimal_training_cohort_preparation",
        "train_cohort": {
            "series_count": TRAIN_SERIES_PER_DATASET,
            "anchor_count_per_series": len(TRAIN_ANCHORS),
            "example_count": TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS),
            "diagnostics_by_policy": training_diagnostics,
        },
        "consumer_spec": {
            "class": "sklearn.linear_model.Ridge",
            "alpha": RIDGE_ALPHA,
            "fit_intercept": True,
            "solver": "svd",
            "input_dimension": 384,
            "output_dimension": HORIZON,
            "random_training_or_tuning": False,
        },
        "eval_cohort": {
            "series_count": EVAL_SERIES_PER_DATASET,
            "subsplit": VALIDATION_SUBSPLIT,
            "context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "future_bounds": list(EVAL_FUTURE_BOUNDS),
            "clean_input_and_zero_mask_shared_across_policies": True,
            "diagnostics": evaluation_diagnostics,
        },
        "identity_minimal_mean_normalized_mae": identity_mean,
        "linear_mean_normalized_mae": linear_mean,
        "mean_paired_gain": mean_gain,
        "median_paired_gain": median_gain,
        "relative_mean_gain": mean_gain / identity_mean,
        "harm_rate": sum(bool(row["harmed"]) for row in paired) / len(paired),
        "material_gain_rate": sum(bool(row["material_gain"]) for row in paired)
        / len(paired),
        "harm_definition": f"linear gain over identity < {HARM_GAIN_THRESHOLD}",
        "material_gain_definition": (
            f"linear gain over identity >= {MATERIAL_GAIN_MIN}"
        ),
        "paired_series_diagnostics": paired,
    }


def run_e2_source_cohort_policy_fresh_replay(
    *,
    project_root: Path,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    roster, selection = select_roster(
        project_root=project_root,
        registry_path=registry_path,
        split_path=split_path,
        support_a_subsplit_path=support_a_subsplit_path,
    )
    values_by_uid = _load_roster_values(roster, clean_root)

    evidence_rows: list[dict[str, object]] = []
    consumer_fit_count = 0
    for dataset_id in SOURCE_DATASETS:
        train_items = [
            item
            for item in roster
            if item.record.dataset_id == dataset_id and item.cohort == "train"
        ]
        eval_items = [
            item
            for item in roster
            if item.record.dataset_id == dataset_id and item.cohort == "eval"
        ]
        x_eval, y_eval, eval_uids, eval_diagnostics = _evaluation_matrices(
            eval_items, values_by_uid
        )
        policy_losses: dict[str, list[float]] = {}
        training_diagnostics: dict[str, dict[str, object]] = {}
        for policy in POLICIES:
            x_train, y_train, diagnostics = _training_matrices(
                train_items, values_by_uid, policy=policy
            )
            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd")
            model.fit(x_train, y_train)
            consumer_fit_count += 1
            prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
            if prediction.shape != y_eval.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid Ridge prediction: {dataset_id}/{policy}")
            policy_losses[policy] = [
                float(value) for value in np.mean(np.abs(prediction - y_eval), axis=1)
            ]
            training_diagnostics[policy] = diagnostics
        evidence_rows.append(
            _cohort_evidence(
                dataset_id=dataset_id,
                identity_losses=policy_losses["identity_minimal"],
                linear_losses=policy_losses["linear"],
                eval_uids=eval_uids,
                training_diagnostics=training_diagnostics,
                evaluation_diagnostics=eval_diagnostics,
            )
        )

    if consumer_fit_count != 4:
        raise AssertionError("expected exactly four independent Consumer fits")
    evidence_by_dataset = {str(row["dataset_id"]): row for row in evidence_rows}
    mean_gains = {
        dataset_id: float(evidence_by_dataset[dataset_id]["mean_paired_gain"])
        for dataset_id in SOURCE_DATASETS
    }
    median_gains = {
        dataset_id: float(evidence_by_dataset[dataset_id]["median_paired_gain"])
        for dataset_id in SOURCE_DATASETS
    }
    paired_rows = [
        paired
        for row in evidence_rows
        for paired in row["paired_series_diagnostics"]  # type: ignore[union-attr]
    ]
    pooled_harm_rate = sum(bool(row["harmed"]) for row in paired_rows) / len(paired_rows)
    mean_gate_pass = all(gain > 0.0 for gain in mean_gains.values())
    median_gate_pass = all(gain > 0.0 for gain in median_gains.values())
    harm_gate_pass = pooled_harm_rate <= HARM_RATE_MAX
    passed = mean_gate_pass and median_gate_pass and harm_gate_pass

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": SCIENTIFIC_ROLE,
        "configuration": {
            "datasets": list(SOURCE_DATASETS),
            "train_split": SplitRole.SUPPORT_A.value,
            "train_subsplit": DISCOVERY_SUBSPLIT,
            "eval_split": SplitRole.SUPPORT_A.value,
            "eval_subsplit": VALIDATION_SUBSPLIT,
            "policies": list(POLICIES),
            "fixed_candidate": "linear",
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "train_anchors": list(TRAIN_ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "train_gap_relative_to_context": list(GAP_BOUNDS),
            "eval_context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "eval_future_bounds": list(EVAL_FUTURE_BOUNDS),
            "standardization": {
                "center": "median of original finite context values",
                "primary_scale": "1.4826 * median absolute deviation",
                "fallback": (
                    "if primary scale < 1e-6, use observed population std when >=1e-6; "
                    "otherwise use 1e-6"
                ),
                "scale_floor": ROBUST_SCALE_FLOOR,
            },
            "original_missing_mask_appended": True,
            "consumer_input_dimension": 384,
            "agent_enabled": False,
            "memory_enabled": False,
            "adaptation_enabled": False,
        },
        "roster_selection": selection,
        "information_wall": {
            "complete_train_and_eval_rosters_fixed_before_selected_value_loading": True,
            "train_eval_series_disjoint": True,
            "train_eval_overlap_group_disjoint": True,
            "train_targets_end_at_or_before_index": 928,
            "eval_future_loaded_only_after_complete_roster_freeze": True,
            "source_datasets_only": list(SOURCE_DATASETS),
            "support_b_values_context_or_future_read": False,
            "uci_values_context_or_future_read": False,
            "dev_query_values_context_or_future_read": False,
            "final_query_values_context_or_future_read": False,
            "target_or_query_read": False,
            "target_query_opened": False,
            "exposure_reports_scanned_only_for_registry_uid_strings": True,
        },
        "consumer_fit_count": consumer_fit_count,
        "chronos_judge_call_count": 0,
        "policy_intervention_evidence": evidence_rows,
        "fresh_replay_gate": {
            "thresholds_frozen_before_replay": True,
            "per_dataset_mean_gain_strictly_positive": {
                "threshold": 0.0,
                "comparison": ">",
                "values": mean_gains,
                "pass": mean_gate_pass,
            },
            "per_dataset_median_gain_strictly_positive": {
                "threshold": 0.0,
                "comparison": ">",
                "values": median_gains,
                "pass": median_gate_pass,
            },
            "pooled_harm_rate": {
                "threshold": HARM_RATE_MAX,
                "comparison": "<=",
                "definition": f"linear gain over identity < {HARM_GAIN_THRESHOLD}",
                "value": pooled_harm_rate,
                "pass": harm_gate_pass,
            },
            "pass": passed,
        },
        "verdict": "FRESH_COHORT_REPLAY_PASSED" if passed else "FRESH_COHORT_REPLAY_FAILED",
        "promotion_eligible": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "At most one fresh Source validation series-cohort replay of the direction "
            "and safety of a fixed Linear-vs-Identity training-cohort preparation effect; "
            "not per-series causal evidence, Capability, Memory, Agent, Target, Query, "
            "adaptation, promotion, formal transfer, or cross-dataset transfer evidence."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/series_registry.jsonl",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/split_manifest.json",
    )
    parser.add_argument(
        "--support-a-subsplit",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/support_a_subsplit.json",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=project_root / "data/benchmark_v0_2/clean_base",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / OUTPUT_RELATIVE_PATH,
    )
    args = parser.parse_args()

    report = run_e2_source_cohort_policy_fresh_replay(
        project_root=project_root,
        registry_path=args.registry,
        split_path=args.split,
        support_a_subsplit_path=args.support_a_subsplit,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(args.output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
