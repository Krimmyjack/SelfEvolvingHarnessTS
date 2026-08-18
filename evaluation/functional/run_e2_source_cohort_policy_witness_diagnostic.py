"""Diagnose a development-only witness for selective Linear cohort preparation.

This runner reuses only the frozen development-premise roster and Source values.  It
asks whether pseudo-gap reconstruction margins can select training examples for
Linear preparation more safely than all-Linear and an activation-matched metadata
hash baseline.  It is a cohort/model-level mechanistic diagnostic, not individual
causal evidence, promotion evidence, or formal transfer.
"""
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
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
    SeriesRecord,
    read_registry_jsonl,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    _execute_program,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    CONTEXT_LENGTH,
    EVAL_CONTEXT_BOUNDS,
    EVAL_FUTURE_BOUNDS,
    EVAL_SERIES_PER_DATASET,
    GAP_BOUNDS,
    HARM_GAIN_THRESHOLD,
    HORIZON,
    MATERIAL_GAIN_MIN,
    RIDGE_ALPHA,
    ROBUST_SCALE_FLOOR,
    SOURCE_DATASETS,
    TRAIN_ANCHORS,
    TRAIN_SERIES_PER_DATASET,
    _center_scale,
    _evaluation_matrices,
    _load_roster_values,
    _standardized_features,
)


SCHEMA_VERSION = "e2-source-cohort-policy-witness-diagnostic/1"
SCIENTIFIC_ROLE = "development_source_cohort_applicability_witness_diagnostic"
PREMISE_SCHEMA_VERSION = "e2-source-cohort-policy-premise/1"
POLICIES = (
    "identity_minimal",
    "linear_all",
    "linear_witness",
    "linear_activation_matched_hash",
)
PSEUDO_GAP_WINDOWS = ((48, 72), (72, 96), (96, 120), (120, 144))
HASH_SALT = "e2-source-cohort-policy-witness-matched-hash-v1"
HARM_RATE_MAX = 0.25
PREMISE_REPORT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_cohort_policy_premise_report.json"
)
OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_cohort_policy_witness_diagnostic_report.json"
)


@dataclass(frozen=True)
class RosterItem:
    """The minimal premise-roster view needed by the reused matrix helpers."""

    record: SeriesRecord
    cohort: str


ExampleKey = tuple[str, int]


def _require_locked_configuration(configuration: object) -> None:
    if not isinstance(configuration, dict):
        raise ValueError("premise report lacks configuration")
    expected: dict[str, object] = {
        "datasets": sorted(SOURCE_DATASETS),
        "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
        "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
        "train_anchors": list(TRAIN_ANCHORS),
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "train_gap_relative_to_context": list(GAP_BOUNDS),
        "eval_context_bounds": list(EVAL_CONTEXT_BOUNDS),
        "eval_future_bounds": list(EVAL_FUTURE_BOUNDS),
        "consumer_input_dimension": 2 * CONTEXT_LENGTH,
    }
    for name, value in expected.items():
        if configuration.get(name) != value:
            raise ValueError(f"premise configuration changed at {name}")


def _read_premise_roster(
    *, premise_report_path: Path, registry_path: Path
) -> tuple[list[RosterItem], dict[str, object]]:
    """Resolve only the premise report's already-exposed train/eval UIDs."""

    premise = json.loads(premise_report_path.read_text("utf-8"))
    if premise.get("schema_version") != PREMISE_SCHEMA_VERSION:
        raise ValueError("unsupported cohort-policy premise report schema")
    _require_locked_configuration(premise.get("configuration"))
    selection = premise.get("roster_selection")
    selected = selection.get("selected_by_dataset") if isinstance(selection, dict) else None
    if not isinstance(selected, dict) or set(selected) != set(SOURCE_DATASETS):
        raise ValueError("premise report lacks exactly the two locked Source rosters")

    selected_uids: dict[str, dict[str, list[str]]] = {}
    all_uids: list[str] = []
    for dataset_id in SOURCE_DATASETS:
        raw_dataset = selected.get(dataset_id)
        if not isinstance(raw_dataset, dict):
            raise ValueError(f"invalid premise roster for {dataset_id}")
        selected_uids[dataset_id] = {}
        for cohort, expected_count in (
            ("train", TRAIN_SERIES_PER_DATASET),
            ("eval", EVAL_SERIES_PER_DATASET),
        ):
            raw = raw_dataset.get(cohort)
            if not isinstance(raw, list) or not all(
                isinstance(uid, str) and uid for uid in raw
            ):
                raise ValueError(f"invalid premise {cohort} UID list: {dataset_id}")
            if len(raw) != expected_count or len(raw) != len(set(raw)):
                raise ValueError(
                    f"premise {cohort} roster has wrong size or duplicates: {dataset_id}"
                )
            selected_uids[dataset_id][cohort] = list(raw)
            all_uids.extend(raw)
    if len(all_uids) != len(set(all_uids)):
        raise ValueError("premise train/eval rosters overlap")

    records = {record.series_uid: record for record in read_registry_jsonl(registry_path)}
    roster: list[RosterItem] = []
    metadata_audit: dict[str, dict[str, object]] = {}
    for dataset_id in SOURCE_DATASETS:
        dataset_items: list[RosterItem] = []
        for cohort in ("train", "eval"):
            for uid in selected_uids[dataset_id][cohort]:
                record = records.get(uid)
                if record is None:
                    raise ValueError(f"premise UID absent from frozen registry: {uid}")
                if record.dataset_id != dataset_id:
                    raise ValueError(f"premise UID changed dataset: {uid}")
                dataset_items.append(RosterItem(record=record, cohort=cohort))
        train_entities = [
            item.record.entity_id for item in dataset_items if item.cohort == "train"
        ]
        if len(train_entities) != len(set(train_entities)):
            raise ValueError(f"duplicate train entity_id prevents hash baseline: {dataset_id}")
        roster.extend(dataset_items)
        metadata_audit[dataset_id] = {
            "train_uid_count": TRAIN_SERIES_PER_DATASET,
            "eval_uid_count": EVAL_SERIES_PER_DATASET,
            "train_entity_ids_unique": True,
            "all_records_match_locked_dataset_id": True,
        }
    return roster, {
        "source": str(premise_report_path),
        "premise_fields_consulted": [
            "schema_version",
            "configuration",
            "roster_selection.selected_by_dataset",
        ],
        "premise_outcomes_or_policy_evidence_consulted": False,
        "selected_by_dataset": selected_uids,
        "metadata_audit_by_dataset": metadata_audit,
        "train_eval_uid_disjoint": True,
        "support_a_subsplit_read": False,
    }


def _corrupt_real_gap(clean_context: np.ndarray) -> np.ndarray:
    context = np.asarray(clean_context, dtype=np.float64)
    if context.shape != (CONTEXT_LENGTH,):
        raise ValueError("training context must have length 192")
    if not np.isfinite(context).all():
        raise ValueError("natural missingness enters training context")
    corrupt = context.copy()
    corrupt[slice(*GAP_BOUNDS)] = np.nan
    if int(np.isnan(corrupt).sum()) != GAP_BOUNDS[1] - GAP_BOUNDS[0]:
        raise AssertionError("real training corruption must inject exactly 24 NaNs")
    return corrupt


def _witness_decision(corrupt_context: np.ndarray) -> dict[str, object]:
    """Use only visible context values; the real gap must already be masked."""

    base = np.asarray(corrupt_context, dtype=np.float64).copy()
    if base.shape != (CONTEXT_LENGTH,):
        return {"status": "UNRESOLVED", "reason": "invalid_context_shape", "windows": []}
    real_gap = base[slice(*GAP_BOUNDS)]
    if not np.isnan(real_gap).all():
        return {
            "status": "UNRESOLVED",
            "reason": "real_gap_not_fully_masked_before_witness",
            "windows": [],
        }

    rows: list[dict[str, object]] = []
    for start, end in PSEUDO_GAP_WINDOWS:
        if start < 0 or end > CONTEXT_LENGTH or start >= end:
            return {
                "status": "UNRESOLVED",
                "reason": f"illegal_pseudo_window_{start}_{end}",
                "windows": rows,
            }
        if max(start, GAP_BOUNDS[0]) < min(end, GAP_BOUNDS[1]):
            return {
                "status": "UNRESOLVED",
                "reason": f"pseudo_window_overlaps_real_gap_{start}_{end}",
                "windows": rows,
            }
        truth = base[start:end].copy()
        if truth.shape != (end - start,) or not np.isfinite(truth).all():
            return {
                "status": "UNRESOLVED",
                "reason": f"pseudo_window_not_legally_visible_{start}_{end}",
                "windows": rows,
            }
        pseudo_corrupt = base.copy()
        pseudo_corrupt[start:end] = np.nan
        try:
            center, scale, scale_method = _center_scale(pseudo_corrupt)
            prepared = _execute_program("linear", pseudo_corrupt, observed_period=1)
        except (RuntimeError, ValueError) as exc:
            return {
                "status": "UNRESOLVED",
                "reason": f"pseudo_reconstruction_failed_{start}_{end}:{type(exc).__name__}",
                "windows": rows,
            }
        linear_nmae = float(np.mean(np.abs(prepared[start:end] - truth)) / scale)
        identity_nmae = float(np.mean(np.abs(center - truth)) / scale)
        margin = identity_nmae - linear_nmae
        if not all(np.isfinite(value) for value in (linear_nmae, identity_nmae, margin)):
            return {
                "status": "UNRESOLVED",
                "reason": f"non_finite_margin_{start}_{end}",
                "windows": rows,
            }
        rows.append(
            {
                "bounds": [start, end],
                "center_from_pseudo_masked_visible_values": center,
                "scale_from_pseudo_masked_visible_values": scale,
                "scale_method": scale_method,
                "linear_nmae": linear_nmae,
                "identity_center_fill_nmae": identity_nmae,
                "margin": margin,
            }
        )

    median_margin = float(statistics.median(float(row["margin"]) for row in rows))
    eligible = median_margin > 0.0
    return {
        "status": "ELIGIBLE" if eligible else "INELIGIBLE",
        "reason": (
            "all_four_windows_legal_and_median_margin_strictly_positive"
            if eligible
            else "all_four_windows_legal_but_median_margin_not_strictly_positive"
        ),
        "median_margin": median_margin,
        "windows": rows,
    }


def _hash_rank(dataset_id: str, entity_id: str, anchor: int) -> str:
    payload = f"{HASH_SALT}\0{dataset_id}\0{entity_id}\0{anchor}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _freeze_activations(
    *,
    dataset_id: str,
    train_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
) -> tuple[set[ExampleKey], set[ExampleKey], dict[str, object]]:
    """Freeze witness and matched-hash activations before any target is sliced."""

    witness_active: set[ExampleKey] = set()
    rows: list[dict[str, object]] = []
    hash_candidates: list[tuple[str, str, int, ExampleKey]] = []
    for item in train_items:
        uid = item.record.series_uid
        entity_id = item.record.entity_id
        values = values_by_uid[uid]
        for anchor in TRAIN_ANCHORS:
            start = anchor - CONTEXT_LENGTH
            if start < 0 or anchor > 928:
                raise AssertionError("training context crosses the frozen train boundary")
            corrupt = _corrupt_real_gap(values[start:anchor])
            decision = _witness_decision(corrupt)
            key = (uid, anchor)
            if decision["status"] == "ELIGIBLE":
                witness_active.add(key)
            rank = _hash_rank(dataset_id, entity_id, anchor)
            hash_candidates.append((rank, entity_id, anchor, key))
            rows.append(
                {
                    "series_uid": uid,
                    "entity_id": entity_id,
                    "anchor": anchor,
                    "semantic_role": "program_conditioned_pseudo_gap_witness_not_utility_support",
                    **decision,
                    "matched_hash_rank_sha256": rank,
                }
            )

    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    if len(rows) != expected:
        raise AssertionError("unexpected witness example count")
    ranked = sorted(hash_candidates, key=lambda row: (row[0], row[1], row[2]))
    hash_active = {row[3] for row in ranked[: len(witness_active)]}
    if len(hash_active) != len(witness_active):
        raise AssertionError("matched hash activation count differs from witness")
    for row in rows:
        key = (str(row["series_uid"]), int(row["anchor"]))
        row["witness_activates_linear"] = key in witness_active
        row["matched_hash_activates_linear"] = key in hash_active

    overlap = witness_active & hash_active
    union = witness_active | hash_active
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("ELIGIBLE", "INELIGIBLE", "UNRESOLVED")
    }
    return witness_active, hash_active, {
        "dataset_id": dataset_id,
        "example_count": expected,
        "pseudo_gap_windows": [list(window) for window in PSEUDO_GAP_WINDOWS],
        "eligibility_rule": (
            "ELIGIBLE iff all four pseudo windows are legal and the median of "
            "identity_center_fill_nmae - linear_nmae is strictly greater than zero"
        ),
        "identity_comparator_semantics": (
            "fill each temporarily masked pseudo window with the median center "
            "recomputed after that mask; equivalently standardized feature value zero"
        ),
        "status_counts": status_counts,
        "witness_activation_count": len(witness_active),
        "witness_activation_rate": len(witness_active) / expected,
        "matched_hash_activation_count": len(hash_active),
        "matched_hash_activation_rate": len(hash_active) / expected,
        "activation_counts_exactly_matched": True,
        "activation_overlap_count": len(overlap),
        "activation_overlap_rate_of_all_examples": len(overlap) / expected,
        "activation_jaccard": len(overlap) / len(union) if union else 1.0,
        "matched_hash_selection": {
            "salt": HASH_SALT,
            "rank_input_fields": ["dataset_id", "entity_id", "anchor"],
            "series_uid_content_future_or_outcome_used": False,
        },
        "per_training_example": rows,
    }


def _training_matrices(
    *,
    train_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
    policy: str,
    witness_active: set[ExampleKey],
    hash_active: set[ExampleKey],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    scale_methods: dict[str, int] = {}
    activation_count = 0
    for item in train_items:
        uid = item.record.series_uid
        values = values_by_uid[uid]
        for anchor in TRAIN_ANCHORS:
            context_start = anchor - CONTEXT_LENGTH
            target_end = anchor + HORIZON
            if context_start < 0 or target_end > 928:
                raise AssertionError("training example crosses the frozen train boundary")
            corrupt = _corrupt_real_gap(values[context_start:anchor])
            target = np.asarray(values[anchor:target_end], dtype=np.float64).copy()
            if target.shape != (HORIZON,) or not np.isfinite(target).all():
                raise ValueError(f"invalid training target: {uid}/{anchor}")
            key = (uid, anchor)
            if policy == "identity_minimal":
                linear_active = False
            elif policy == "linear_all":
                linear_active = True
            elif policy == "linear_witness":
                linear_active = key in witness_active
            elif policy == "linear_activation_matched_hash":
                linear_active = key in hash_active
            else:
                raise ValueError(f"unknown training cohort policy: {policy}")
            center, scale, scale_method = _center_scale(corrupt)
            features, _ = _standardized_features(
                corrupt,
                policy="linear" if linear_active else "identity_minimal",
                center=center,
                scale=scale,
            )
            normalized_target = (target - center) / scale
            if not np.isfinite(normalized_target).all():
                raise ValueError("standardized training target is non-finite")
            x_rows.append(features)
            y_rows.append(normalized_target)
            activation_count += int(linear_active)
            scale_methods[scale_method] = scale_methods.get(scale_method, 0) + 1

    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    if x.shape != (expected, 2 * CONTEXT_LENGTH) or y.shape != (expected, HORIZON):
        raise AssertionError("unexpected cohort training matrix shape")
    return x, y, {
        "scale_method_counts": scale_methods,
        "linear_activation_count": activation_count,
        "linear_activation_rate": activation_count / expected,
    }


def _policy_evidence(
    *,
    dataset_id: str,
    policy: str,
    losses: list[float],
    identity_losses: list[float],
    eval_uids: list[str],
    training_diagnostics: dict[str, object],
    evaluation_diagnostics: dict[str, object],
) -> dict[str, object]:
    if not (len(losses) == len(identity_losses) == len(eval_uids)):
        raise AssertionError("paired evidence lengths disagree")
    paired: list[dict[str, object]] = []
    for uid, loss, identity_loss in zip(eval_uids, losses, identity_losses):
        gain = identity_loss - loss
        paired.append(
            {
                "diagnostic_role": "paired_eval_series_diagnostic_not_individual_causal_evidence",
                "series_uid": uid,
                "normalized_mae": loss,
                "identity_minimal_normalized_mae": identity_loss,
                "gain_over_identity": gain,
                "harmed": gain < HARM_GAIN_THRESHOLD,
                "material_gain": gain >= MATERIAL_GAIN_MIN,
            }
        )
    gains = [float(row["gain_over_identity"]) for row in paired]
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "scientific_unit": "dataset_level_exposed_development_premise_cohort",
        "dataset_id": dataset_id,
        "policy": policy,
        "train_cohort": {
            "series_count": TRAIN_SERIES_PER_DATASET,
            "anchor_count_per_series": len(TRAIN_ANCHORS),
            "example_count": TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS),
            "diagnostics": training_diagnostics,
        },
        "eval_cohort": {
            "series_count": EVAL_SERIES_PER_DATASET,
            "context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "future_bounds": list(EVAL_FUTURE_BOUNDS),
            "clean_input_and_zero_mask_shared_across_all_four_models": True,
            "diagnostics": evaluation_diagnostics,
        },
        "consumer_spec": {
            "class": "sklearn.linear_model.Ridge",
            "alpha": RIDGE_ALPHA,
            "fit_intercept": True,
            "solver": "svd",
            "input_dimension": 2 * CONTEXT_LENGTH,
            "output_dimension": HORIZON,
            "random_training_or_tuning": False,
        },
        "mean_normalized_mae": statistics.fmean(losses),
        "mean_gain_over_identity": statistics.fmean(gains),
        "median_gain_over_identity": statistics.median(gains),
        "harm_rate": sum(bool(row["harmed"]) for row in paired) / len(paired),
        "material_gain_rate": sum(bool(row["material_gain"]) for row in paired)
        / len(paired),
        "harm_definition": f"gain over Identity < {HARM_GAIN_THRESHOLD}",
        "material_gain_definition": f"gain over Identity >= {MATERIAL_GAIN_MIN}",
        "paired_eval_series_diagnostics": paired,
    }


def _comparison_evidence(
    *,
    dataset_id: str,
    candidate: str,
    comparator: str,
    candidate_losses: list[float],
    comparator_losses: list[float],
    eval_uids: list[str],
) -> dict[str, object]:
    paired = []
    for uid, candidate_loss, comparator_loss in zip(
        eval_uids, candidate_losses, comparator_losses
    ):
        gain = comparator_loss - candidate_loss
        paired.append(
            {
                "diagnostic_role": "paired_eval_series_diagnostic_not_individual_causal_evidence",
                "series_uid": uid,
                "candidate_normalized_mae": candidate_loss,
                "comparator_normalized_mae": comparator_loss,
                "candidate_gain_over_comparator": gain,
                "candidate_harmed": gain < HARM_GAIN_THRESHOLD,
                "candidate_material_gain": gain >= MATERIAL_GAIN_MIN,
            }
        )
    gains = [float(row["candidate_gain_over_comparator"]) for row in paired]
    return {
        "evidence_type": "PolicyInterventionComparisonDiagnostic",
        "dataset_id": dataset_id,
        "candidate": candidate,
        "comparator": comparator,
        "mean_candidate_gain_over_comparator": statistics.fmean(gains),
        "median_candidate_gain_over_comparator": statistics.median(gains),
        "candidate_harm_rate": sum(bool(row["candidate_harmed"]) for row in paired)
        / len(paired),
        "candidate_material_gain_rate": sum(
            bool(row["candidate_material_gain"]) for row in paired
        )
        / len(paired),
        "paired_eval_series_diagnostics": paired,
    }


def run_e2_source_cohort_policy_witness_diagnostic(
    *,
    premise_report_path: Path,
    registry_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    roster, roster_audit = _read_premise_roster(
        premise_report_path=premise_report_path,
        registry_path=registry_path,
    )
    values_by_uid = _load_roster_values(roster, clean_root)  # type: ignore[arg-type]

    evidence_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    witness_diagnostics: dict[str, object] = {}
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
        witness_active, hash_active, witness_report = _freeze_activations(
            dataset_id=dataset_id,
            train_items=train_items,
            values_by_uid=values_by_uid,
        )
        witness_diagnostics[dataset_id] = witness_report
        x_eval, y_eval, eval_uids, eval_diagnostics = _evaluation_matrices(
            eval_items, values_by_uid  # type: ignore[arg-type]
        )
        policy_losses: dict[str, list[float]] = {}
        training_diagnostics: dict[str, dict[str, object]] = {}
        for policy in POLICIES:
            x_train, y_train, diagnostics = _training_matrices(
                train_items=train_items,
                values_by_uid=values_by_uid,
                policy=policy,
                witness_active=witness_active,
                hash_active=hash_active,
            )
            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd")
            model.fit(x_train, y_train)
            consumer_fit_count += 1
            prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
            if prediction.shape != y_eval.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid Ridge prediction: {dataset_id}/{policy}")
            policy_losses[policy] = [
                float(loss) for loss in np.mean(np.abs(prediction - y_eval), axis=1)
            ]
            training_diagnostics[policy] = diagnostics

        identity_losses = policy_losses["identity_minimal"]
        for policy in POLICIES:
            evidence_rows.append(
                _policy_evidence(
                    dataset_id=dataset_id,
                    policy=policy,
                    losses=policy_losses[policy],
                    identity_losses=identity_losses,
                    eval_uids=eval_uids,
                    training_diagnostics=training_diagnostics[policy],
                    evaluation_diagnostics=eval_diagnostics,
                )
            )
        for comparator in ("linear_all", "linear_activation_matched_hash"):
            comparison_rows.append(
                _comparison_evidence(
                    dataset_id=dataset_id,
                    candidate="linear_witness",
                    comparator=comparator,
                    candidate_losses=policy_losses["linear_witness"],
                    comparator_losses=policy_losses[comparator],
                    eval_uids=eval_uids,
                )
            )

    expected_fit_count = len(SOURCE_DATASETS) * len(POLICIES)
    if consumer_fit_count != expected_fit_count:
        raise AssertionError("expected exactly eight independent Consumer fits")
    evidence_by_key = {
        (str(row["dataset_id"]), str(row["policy"])): row for row in evidence_rows
    }
    witness_rows = [
        evidence_by_key[(dataset_id, "linear_witness")] for dataset_id in SOURCE_DATASETS
    ]
    hash_rows = [
        evidence_by_key[(dataset_id, "linear_activation_matched_hash")]
        for dataset_id in SOURCE_DATASETS
    ]
    witness_gains = {
        dataset_id: float(
            evidence_by_key[(dataset_id, "linear_witness")]["mean_gain_over_identity"]
        )
        for dataset_id in SOURCE_DATASETS
    }
    per_dataset_positive_pass = all(gain > 0.0 for gain in witness_gains.values())
    witness_paired = [
        paired
        for row in witness_rows
        for paired in row["paired_eval_series_diagnostics"]  # type: ignore[union-attr]
    ]
    hash_paired = [
        paired
        for row in hash_rows
        for paired in row["paired_eval_series_diagnostics"]  # type: ignore[union-attr]
    ]
    witness_harm = sum(bool(row["harmed"]) for row in witness_paired) / len(
        witness_paired
    )
    hash_harm = sum(bool(row["harmed"]) for row in hash_paired) / len(hash_paired)
    witness_dataset_equal_gain = statistics.fmean(
        float(row["mean_gain_over_identity"]) for row in witness_rows
    )
    hash_dataset_equal_gain = statistics.fmean(
        float(row["mean_gain_over_identity"]) for row in hash_rows
    )
    harm_cap_pass = witness_harm <= HARM_RATE_MAX
    gain_not_lower = witness_dataset_equal_gain >= hash_dataset_equal_gain
    harm_not_higher = witness_harm <= hash_harm
    at_least_one_strict = (
        witness_dataset_equal_gain > hash_dataset_equal_gain or witness_harm < hash_harm
    )
    matched_hash_dominance_pass = gain_not_lower and harm_not_higher and at_least_one_strict
    gate_pass = per_dataset_positive_pass and harm_cap_pass and matched_hash_dominance_pass

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": SCIENTIFIC_ROLE,
        "configuration": {
            "datasets": list(SOURCE_DATASETS),
            "premise_report": str(premise_report_path),
            "policies": list(POLICIES),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "train_anchors": list(TRAIN_ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "real_train_gap_relative_to_context": list(GAP_BOUNDS),
            "pseudo_gap_windows_relative_to_context": [
                list(window) for window in PSEUDO_GAP_WINDOWS
            ],
            "eval_context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "eval_future_bounds": list(EVAL_FUTURE_BOUNDS),
            "standardization_scale_floor": ROBUST_SCALE_FLOOR,
            "consumer_input_dimension": 2 * CONTEXT_LENGTH,
            "agent_memory_adaptation_enabled": False,
        },
        "roster": roster_audit,
        "information_wall": {
            "only_premise_train_and_eval_uids_resolved": True,
            "premise_policy_evidence_or_outcomes_consulted": False,
            "global_registry_metadata_loaded": True,
            "only_premise_roster_source_values_loaded": True,
            "support_a_validation_values_or_context_or_future_read": False,
            "support_b_values_or_context_or_future_read": False,
            "uci_or_query_values_or_context_or_future_read": False,
            "fresh_replay_report_read": False,
            "target_or_query_value_surface_read": False,
            "real_gap_masked_before_witness_receives_context": True,
            "real_gap_values_future_targets_or_eval_outcomes_used_by_witness": False,
            "training_targets_used_only_after_activations_frozen": True,
            "eval_future_used_only_for_post_fit_policy_evaluation": True,
            "clean_eval_matrix_shared_across_all_four_models_within_dataset": True,
        },
        "applicability_witness_diagnostics": witness_diagnostics,
        "consumer_fit_count": consumer_fit_count,
        "expected_consumer_fit_count": expected_fit_count,
        "chronos_judge_call_count": 0,
        "policy_intervention_evidence": evidence_rows,
        "policy_comparison_diagnostics": comparison_rows,
        "development_gate": {
            "name": "WITNESS_MECHANISM_PROMISING",
            "thresholds_fixed_without_tuning": True,
            "exact_definition": (
                "pass iff linear_witness has strictly positive mean gain over Identity "
                "in both datasets; its pooled harm rate is <=0.25; and versus the "
                "activation-matched hash its dataset-equal mean gain is not lower, its "
                "pooled harm rate is not higher, and at least one comparison is strict"
            ),
            "per_dataset_witness_mean_gain_strictly_positive": {
                "threshold": 0.0,
                "comparison": ">",
                "values": witness_gains,
                "pass": per_dataset_positive_pass,
            },
            "witness_pooled_harm_rate": {
                "threshold": HARM_RATE_MAX,
                "comparison": "<=",
                "harm_definition": f"gain over Identity < {HARM_GAIN_THRESHOLD}",
                "value": witness_harm,
                "pass": harm_cap_pass,
            },
            "activation_matched_hash_comparison": {
                "witness_dataset_equal_mean_gain": witness_dataset_equal_gain,
                "matched_hash_dataset_equal_mean_gain": hash_dataset_equal_gain,
                "witness_gain_not_lower": gain_not_lower,
                "witness_pooled_harm_rate": witness_harm,
                "matched_hash_pooled_harm_rate": hash_harm,
                "witness_harm_not_higher": harm_not_higher,
                "at_least_one_strictly_better": at_least_one_strict,
                "pass": matched_hash_dominance_pass,
            },
            "pass": gate_pass,
        },
        "verdict": (
            "WITNESS_MECHANISM_PROMISING"
            if gate_pass
            else "WITNESS_MECHANISM_NOT_PROMISING"
        ),
        "promotion": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "target_query": False,
        "target_query_opened": False,
        "claim_limit": (
            "At most an exposed-development Source mechanistic diagnostic of a "
            "cohort/model-level selective training-preparation policy; not individual "
            "causal or Utility-supported evidence, fresh-series evidence, Capability, "
            "promotion, formal transfer, Target, or Query evidence."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--premise-report",
        type=Path,
        default=project_root / PREMISE_REPORT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/series_registry.jsonl",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=project_root / "data/benchmark_v0_2/clean_base",
    )
    parser.add_argument("--output", type=Path, default=project_root / OUTPUT_RELATIVE_PATH)
    args = parser.parse_args()

    report = run_e2_source_cohort_policy_witness_diagnostic(
        premise_report_path=args.premise_report,
        registry_path=args.registry,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(args.output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
