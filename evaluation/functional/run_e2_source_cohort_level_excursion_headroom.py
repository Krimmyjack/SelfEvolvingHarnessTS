"""Screen development-only cohort headroom for observable level-excursion repair.

The screen injects one fixed transient level excursion into each exposed premise
training context and fits dataset-level Ridge consumers.  Hidden corruption geometry
is available only to the injector and private grader; the repair binding sees only
the canonical public-feature mapping.  This is a Source positive control, not
promotion, transfer, Query, or individual causal evidence.
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
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    CONTEXT_LENGTH,
    EVAL_CONTEXT_BOUNDS,
    EVAL_FUTURE_BOUNDS,
    EVAL_SERIES_PER_DATASET,
    HARM_GAIN_THRESHOLD,
    HORIZON,
    RIDGE_ALPHA,
    ROBUST_SCALE_FLOOR,
    SOURCE_DATASETS,
    TRAIN_ANCHORS,
    TRAIN_SERIES_PER_DATASET,
    _center_scale,
    _evaluation_matrices,
    _load_roster_values,
)
from SelfEvolvingHarnessTS.operators.s1_structural import repair_level_shift
from SelfEvolvingHarnessTS.runtime.public_features import extract_public_features


SCHEMA_VERSION = "e2-source-cohort-level-excursion-headroom/1"
SCIENTIFIC_ROLE = "development_source_cohort_level_excursion_positive_control"
PREMISE_SCHEMA_VERSION = "e2-source-cohort-policy-premise/1"
POLICIES = ("clean_upper_bound", "corrupt_identity", "observable_level_repair")
# Pre-run contract correction: the frozen public observer accepts excursion widths
# 40--96 and requires at least 12 post points.  This M0 severe topology is width 64
# and leaves 16 post points; the earlier proposed [144, 168) was mechanically
# undetectable at width 24 and therefore could not serve as a positive control.
EXCURSION_BOUNDS = (112, 176)
EXCURSION_SCALE_MULTIPLIER = 2.0
SIGN_SALT = "e2-source-cohort-level-excursion-sign-v1"
HARM_RATE_MAX = 0.25
PREMISE_REPORT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_cohort_policy_premise_report.json"
)
OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_cohort_level_excursion_headroom_report.json"
)


@dataclass(frozen=True)
class RosterItem:
    record: SeriesRecord
    cohort: str


@dataclass(frozen=True)
class TrainingExample:
    uid: str
    entity_id: str
    anchor: int
    clean: np.ndarray
    corrupt: np.ndarray
    repaired: np.ndarray
    target: np.ndarray
    corrupt_center: float
    corrupt_scale: float
    corrupt_scale_method: str


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
    """Resolve the premise report's exposed roster without consulting its outcomes."""

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
        for cohort, count in (
            ("train", TRAIN_SERIES_PER_DATASET),
            ("eval", EVAL_SERIES_PER_DATASET),
        ):
            raw = raw_dataset.get(cohort)
            if not isinstance(raw, list) or not all(
                isinstance(uid, str) and uid for uid in raw
            ):
                raise ValueError(f"invalid premise {cohort} UID list: {dataset_id}")
            if len(raw) != count or len(raw) != len(set(raw)):
                raise ValueError(f"wrong-sized or duplicate premise roster: {dataset_id}")
            selected_uids[dataset_id][cohort] = list(raw)
            all_uids.extend(raw)
    if len(all_uids) != len(set(all_uids)):
        raise ValueError("premise train/eval rosters overlap")

    records = {record.series_uid: record for record in read_registry_jsonl(registry_path)}
    roster: list[RosterItem] = []
    metadata_audit: dict[str, object] = {}
    for dataset_id in SOURCE_DATASETS:
        dataset_items: list[RosterItem] = []
        for cohort in ("train", "eval"):
            for uid in selected_uids[dataset_id][cohort]:
                record = records.get(uid)
                if record is None or record.dataset_id != dataset_id:
                    raise ValueError(f"premise UID missing or changed dataset: {uid}")
                dataset_items.append(RosterItem(record=record, cohort=cohort))
        train_entities = [
            item.record.entity_id for item in dataset_items if item.cohort == "train"
        ]
        if len(train_entities) != len(set(train_entities)):
            raise ValueError(f"duplicate premise train entity_id: {dataset_id}")
        roster.extend(dataset_items)
        metadata_audit[dataset_id] = {
            "train_uid_count": TRAIN_SERIES_PER_DATASET,
            "eval_uid_count": EVAL_SERIES_PER_DATASET,
            "train_entity_ids_unique": True,
            "records_match_locked_dataset_id": True,
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
    }


def _excursion_sign(dataset_id: str, entity_id: str, anchor: int) -> tuple[int, str]:
    payload = f"{SIGN_SALT}\0{dataset_id}\0{entity_id}\0{anchor}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (1 if int(digest[:2], 16) % 2 else -1), digest


def _inject_excursion(
    clean: np.ndarray, *, dataset_id: str, entity_id: str, anchor: int
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(clean, dtype=np.float64)
    if values.shape != (CONTEXT_LENGTH,) or not np.isfinite(values).all():
        raise ValueError("excursion injection requires a finite length-192 context")
    _, visible_scale, scale_method = _center_scale(values)
    sign, sign_digest = _excursion_sign(dataset_id, entity_id, anchor)
    signed_offset = sign * EXCURSION_SCALE_MULTIPLIER * visible_scale
    corrupt = values.copy()
    corrupt[slice(*EXCURSION_BOUNDS)] += signed_offset
    return corrupt, {
        "hidden_bounds": list(EXCURSION_BOUNDS),
        "hidden_sign": sign,
        "hidden_signed_offset": signed_offset,
        "visible_clean_context_scale": visible_scale,
        "visible_clean_context_scale_method": scale_method,
        "sign_sha256": sign_digest,
    }


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _observable_level_repair(values: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Bind only public mapping fields; no hidden geometry enters this function."""

    raw = np.asarray(values, dtype=np.float64)
    extraction = extract_public_features(raw)
    mapping = extraction.mapping
    score = mapping.get("level_excursion_score")
    start = mapping.get("estimated_region_start_fraction")
    end = mapping.get("estimated_region_end_fraction")
    offset = mapping.get("estimated_level_offset")
    public_view = {
        "level_excursion_score": score,
        "estimated_region_start_fraction": start,
        "estimated_region_end_fraction": end,
        "estimated_level_offset": offset,
    }
    valid = all(_finite_number(value) for value in (score, start, end, offset))
    if valid:
        valid = (
            float(score) > 0.0
            and 0.0 <= float(start) < float(end) <= 1.0
            and float(offset) != 0.0
        )
    if not valid:
        return raw.copy(), {
            "activated": False,
            "reason": "public_level_fields_invalid_zero_or_inactive",
            "public_mapping_view": public_view,
            "hidden_fields_consulted": False,
        }
    repaired = np.asarray(
        repair_level_shift(
            raw,
            region_start_fraction=float(start),
            region_end_fraction=float(end),
            estimated_offset=float(offset),
        ),
        dtype=np.float64,
    )
    if repaired.shape != raw.shape or not np.isfinite(repaired).all():
        raise RuntimeError("canonical observable level repair returned invalid output")
    modified = ~np.equal(repaired, raw)
    return repaired, {
        "activated": True,
        "reason": "valid_nonzero_public_level_binding",
        "public_mapping_view": public_view,
        "hidden_fields_consulted": False,
        "modified_point_count": int(np.count_nonzero(modified)),
        "mean_absolute_modification": float(np.mean(np.abs(repaired - raw))),
    }


def _private_grader(
    *, binding: dict[str, object], hidden: dict[str, object]
) -> dict[str, object]:
    """Score localization after binding; this output never feeds the observer/program."""

    public = binding["public_mapping_view"]
    if not isinstance(public, dict):
        raise TypeError("binding lacks its public mapping view")
    start_value = public.get("estimated_region_start_fraction")
    end_value = public.get("estimated_region_end_fraction")
    offset_value = public.get("estimated_level_offset")
    valid_region = (
        _finite_number(start_value)
        and _finite_number(end_value)
        and 0.0 <= float(start_value) < float(end_value) <= 1.0
    )
    if valid_region:
        estimated_start = min(
            CONTEXT_LENGTH - 1,
            max(0, int(np.floor(float(start_value) * CONTEXT_LENGTH))),
        )
        estimated_end = min(
            CONTEXT_LENGTH,
            max(estimated_start + 1, int(np.ceil(float(end_value) * CONTEXT_LENGTH))),
        )
        intersection = max(
            0,
            min(estimated_end, EXCURSION_BOUNDS[1])
            - max(estimated_start, EXCURSION_BOUNDS[0]),
        )
        union = (
            estimated_end
            - estimated_start
            + EXCURSION_BOUNDS[1]
            - EXCURSION_BOUNDS[0]
            - intersection
        )
        iou: float | None = intersection / union
    else:
        estimated_start = estimated_end = None
        iou = None
    true_offset = float(hidden["hidden_signed_offset"])
    offset_valid = _finite_number(offset_value)
    estimated_offset = float(offset_value) if offset_valid else None
    return {
        "diagnostic_role": "private_grader_only_not_binding_input",
        "hidden_bounds": list(EXCURSION_BOUNDS),
        "hidden_sign": int(hidden["hidden_sign"]),
        "hidden_signed_offset": true_offset,
        "estimated_bounds": (
            [estimated_start, estimated_end] if estimated_start is not None else None
        ),
        "localization_iou": iou,
        "estimated_offset": estimated_offset,
        "offset_sign_agrees": (
            estimated_offset * true_offset > 0.0 if estimated_offset is not None else None
        ),
        "offset_relative_absolute_error": (
            abs(estimated_offset - true_offset) / abs(true_offset)
            if estimated_offset is not None
            else None
        ),
        "used_by_observer_binding_or_program": False,
    }


def _build_training_examples(
    *,
    dataset_id: str,
    train_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
) -> tuple[list[TrainingExample], dict[str, object], dict[str, object]]:
    examples: list[TrainingExample] = []
    corrupt_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    for item in train_items:
        uid = item.record.series_uid
        values = values_by_uid[uid]
        for anchor in TRAIN_ANCHORS:
            start = anchor - CONTEXT_LENGTH
            target_end = anchor + HORIZON
            if start < 0 or target_end > 928:
                raise AssertionError("training example crosses frozen train boundary")
            clean = np.asarray(values[start:anchor], dtype=np.float64).copy()
            if clean.shape != (CONTEXT_LENGTH,) or not np.isfinite(clean).all():
                raise ValueError(f"invalid clean training context: {uid}/{anchor}")
            corrupt, hidden = _inject_excursion(
                clean,
                dataset_id=dataset_id,
                entity_id=item.record.entity_id,
                anchor=anchor,
            )
            # The observer and binding receive only the corrupt array, not ``hidden``.
            repaired, binding = _observable_level_repair(corrupt)
            grader = _private_grader(binding=binding, hidden=hidden)
            clean_repaired, clean_binding = _observable_level_repair(clean)
            clean_modified = ~np.equal(clean_repaired, clean)
            clean_rows.append(
                {
                    "series_uid": uid,
                    "entity_id": item.record.entity_id,
                    "anchor": anchor,
                    "activated": bool(clean_binding["activated"]),
                    "modified": bool(np.any(clean_modified)),
                    "modified_point_count": int(np.count_nonzero(clean_modified)),
                    "mean_absolute_modification": float(
                        np.mean(np.abs(clean_repaired - clean))
                    ),
                    "binding": clean_binding,
                }
            )
            corrupt_rows.append(
                {
                    "series_uid": uid,
                    "entity_id": item.record.entity_id,
                    "anchor": anchor,
                    "observer_binding": binding,
                    "private_grader": grader,
                    "sign_hash_input_fields": ["dataset_id", "entity_id", "anchor"],
                    "sign_sha256": hidden["sign_sha256"],
                }
            )
            # Target slicing occurs only after observer/binding and clean-risk diagnostics.
            target = np.asarray(values[anchor:target_end], dtype=np.float64).copy()
            if target.shape != (HORIZON,) or not np.isfinite(target).all():
                raise ValueError(f"invalid training target: {uid}/{anchor}")
            corrupt_center, corrupt_scale, scale_method = _center_scale(corrupt)
            examples.append(
                TrainingExample(
                    uid=uid,
                    entity_id=item.record.entity_id,
                    anchor=anchor,
                    clean=clean,
                    corrupt=corrupt,
                    repaired=repaired,
                    target=target,
                    corrupt_center=corrupt_center,
                    corrupt_scale=corrupt_scale,
                    corrupt_scale_method=scale_method,
                )
            )
    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    if len(examples) != expected:
        raise AssertionError("unexpected training example count")
    corrupt_activated = sum(bool(row["observer_binding"]["activated"]) for row in corrupt_rows)  # type: ignore[index]
    ious = [
        float(row["private_grader"]["localization_iou"])  # type: ignore[index]
        for row in corrupt_rows
        if row["private_grader"]["localization_iou"] is not None  # type: ignore[index]
    ]
    offset_sign_rows = [
        bool(row["private_grader"]["offset_sign_agrees"])  # type: ignore[index]
        for row in corrupt_rows
        if row["private_grader"]["offset_sign_agrees"] is not None  # type: ignore[index]
    ]
    corrupt_report = {
        "example_count": expected,
        "observer_activation_count": corrupt_activated,
        "observer_activation_rate": corrupt_activated / expected,
        "private_grader_summary": {
            "diagnostic_only_not_binding": True,
            "localized_example_count": len(ious),
            "mean_localization_iou": statistics.fmean(ious) if ious else None,
            "offset_sign_agreement_rate": (
                sum(offset_sign_rows) / len(offset_sign_rows) if offset_sign_rows else None
            ),
        },
        "per_training_example": corrupt_rows,
    }
    clean_activated = sum(bool(row["activated"]) for row in clean_rows)
    clean_modified_examples = sum(bool(row["modified"]) for row in clean_rows)
    clean_report = {
        "diagnostic_role": "zero_fit_clean_risk_diagnostic",
        "example_count": expected,
        "activation_count": clean_activated,
        "activation_rate": clean_activated / expected,
        "modified_example_count": clean_modified_examples,
        "modified_example_rate": clean_modified_examples / expected,
        "modified_point_count": sum(int(row["modified_point_count"]) for row in clean_rows),
        "per_training_example": clean_rows,
    }
    return examples, corrupt_report, clean_report


def _training_matrices(
    examples: list[TrainingExample], *, policy: str
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    scale_methods: dict[str, int] = {}
    for example in examples:
        if policy == "clean_upper_bound":
            raw = example.clean
        elif policy == "corrupt_identity":
            raw = example.corrupt
        elif policy == "observable_level_repair":
            raw = example.repaired
        else:
            raise ValueError(f"unknown level-excursion training policy: {policy}")
        normalized = (raw - example.corrupt_center) / example.corrupt_scale
        features = np.concatenate(
            (normalized, np.zeros(CONTEXT_LENGTH, dtype=np.float64))
        )
        target = (example.target - example.corrupt_center) / example.corrupt_scale
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("non-finite standardized training example")
        x_rows.append(features)
        y_rows.append(target)
        method = example.corrupt_scale_method
        scale_methods[method] = scale_methods.get(method, 0) + 1
    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    if x.shape != (expected, 2 * CONTEXT_LENGTH) or y.shape != (expected, HORIZON):
        raise AssertionError("unexpected level-excursion training matrix shape")
    return x, y, {
        "scale_method_counts_from_corrupt_context": scale_methods,
        "center_scale_and_target_normalization_source": "corrupt_context",
        "mask_feature_nonzero_count": int(np.count_nonzero(x[:, CONTEXT_LENGTH:])),
    }


def _dataset_evidence(
    *,
    dataset_id: str,
    policy_losses: dict[str, list[float]],
    eval_uids: list[str],
    training_diagnostics: dict[str, dict[str, object]],
    evaluation_diagnostics: dict[str, object],
) -> dict[str, object]:
    clean = policy_losses["clean_upper_bound"]
    corrupt = policy_losses["corrupt_identity"]
    repair = policy_losses["observable_level_repair"]
    if not (len(clean) == len(corrupt) == len(repair) == len(eval_uids)):
        raise AssertionError("paired evaluation evidence lengths disagree")
    paired: list[dict[str, object]] = []
    for uid, clean_loss, corrupt_loss, repair_loss in zip(
        eval_uids, clean, corrupt, repair
    ):
        degradation = corrupt_loss - clean_loss
        repair_gain = corrupt_loss - repair_loss
        paired.append(
            {
                "diagnostic_role": "paired_eval_series_diagnostic_not_individual_causal_evidence",
                "series_uid": uid,
                "clean_upper_bound_normalized_mae": clean_loss,
                "corrupt_identity_normalized_mae": corrupt_loss,
                "observable_level_repair_normalized_mae": repair_loss,
                "degradation_corrupt_minus_clean": degradation,
                "repair_gain_corrupt_minus_repair": repair_gain,
                "recovery_fraction": (
                    repair_gain / degradation if degradation != 0.0 else None
                ),
                "repair_harmed": repair_gain < HARM_GAIN_THRESHOLD,
            }
        )
    mean_losses = {policy: statistics.fmean(losses) for policy, losses in policy_losses.items()}
    degradation = mean_losses["corrupt_identity"] - mean_losses["clean_upper_bound"]
    repair_gain = (
        mean_losses["corrupt_identity"] - mean_losses["observable_level_repair"]
    )
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "scientific_unit": "dataset_level_exposed_development_premise_cohort",
        "dataset_id": dataset_id,
        "policy_mean_normalized_mae": mean_losses,
        "mean_degradation_corrupt_minus_clean_upper_bound": degradation,
        "mean_repair_gain_corrupt_minus_repair": repair_gain,
        "recovery_fraction": repair_gain / degradation if degradation != 0.0 else None,
        "repair_harm_rate": sum(bool(row["repair_harmed"]) for row in paired)
        / len(paired),
        "harm_definition": f"repair gain relative to corrupt < {HARM_GAIN_THRESHOLD}",
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
            "input_dimension": 2 * CONTEXT_LENGTH,
            "output_dimension": HORIZON,
            "random_training_or_tuning": False,
        },
        "eval_cohort": {
            "series_count": EVAL_SERIES_PER_DATASET,
            "context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "future_bounds": list(EVAL_FUTURE_BOUNDS),
            "clean_input_and_zero_mask_shared_across_all_three_models": True,
            "diagnostics": evaluation_diagnostics,
        },
        "paired_eval_series_diagnostics": paired,
    }


def run_e2_source_cohort_level_excursion_headroom(
    *, premise_report_path: Path, registry_path: Path, clean_root: Path
) -> dict[str, object]:
    roster, roster_report = _read_premise_roster(
        premise_report_path=premise_report_path,
        registry_path=registry_path,
    )
    values_by_uid = _load_roster_values(roster, clean_root)  # type: ignore[arg-type]
    evidence_rows: list[dict[str, object]] = []
    corruption_diagnostics: dict[str, object] = {}
    clean_risk_diagnostics: dict[str, object] = {}
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
        examples, corrupt_report, clean_report = _build_training_examples(
            dataset_id=dataset_id,
            train_items=train_items,
            values_by_uid=values_by_uid,
        )
        corruption_diagnostics[dataset_id] = corrupt_report
        clean_risk_diagnostics[dataset_id] = clean_report
        x_eval, y_eval, eval_uids, eval_diagnostics = _evaluation_matrices(
            eval_items, values_by_uid  # type: ignore[arg-type]
        )
        policy_losses: dict[str, list[float]] = {}
        training_diagnostics: dict[str, dict[str, object]] = {}
        for policy in POLICIES:
            x_train, y_train, diagnostics = _training_matrices(examples, policy=policy)
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
        evidence_rows.append(
            _dataset_evidence(
                dataset_id=dataset_id,
                policy_losses=policy_losses,
                eval_uids=eval_uids,
                training_diagnostics=training_diagnostics,
                evaluation_diagnostics=eval_diagnostics,
            )
        )

    expected_fit_count = len(SOURCE_DATASETS) * len(POLICIES)
    if consumer_fit_count != expected_fit_count:
        raise AssertionError("expected exactly six independent Consumer fits")
    degradation_by_dataset = {
        str(row["dataset_id"]): float(
            row["mean_degradation_corrupt_minus_clean_upper_bound"]
        )
        for row in evidence_rows
    }
    repair_gain_by_dataset = {
        str(row["dataset_id"]): float(row["mean_repair_gain_corrupt_minus_repair"])
        for row in evidence_rows
    }
    paired = [
        series
        for row in evidence_rows
        for series in row["paired_eval_series_diagnostics"]  # type: ignore[union-attr]
    ]
    pooled_harm_rate = sum(bool(row["repair_harmed"]) for row in paired) / len(paired)
    degradation_pass = all(value > 0.0 for value in degradation_by_dataset.values())
    repair_gain_pass = all(value > 0.0 for value in repair_gain_by_dataset.values())
    harm_pass = pooled_harm_rate <= HARM_RATE_MAX
    gate_pass = degradation_pass and repair_gain_pass and harm_pass

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": SCIENTIFIC_ROLE,
        "configuration": {
            "datasets": list(SOURCE_DATASETS),
            "policies": list(POLICIES),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "train_anchors": list(TRAIN_ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "excursion_bounds_relative_to_context": list(EXCURSION_BOUNDS),
            "excursion_bounds_rationale": (
                "M0 severe topology chosen before execution because width 64 is within "
                "the frozen public observer detectable width 40--96 and end 176 leaves "
                "16 post points, satisfying its >=12 post-point requirement"
            ),
            "excursion_scale_multiplier": EXCURSION_SCALE_MULTIPLIER,
            "excursion_scale_definition": (
                "2.0 times the robust scale computed from the finite clean context "
                "before excursion injection, using the premise median/MAD fallbacks"
            ),
            "sign_salt": SIGN_SALT,
            "sign_hash_input_fields": ["dataset_id", "entity_id", "anchor"],
            "eval_context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "eval_future_bounds": list(EVAL_FUTURE_BOUNDS),
            "standardization_scale_floor": ROBUST_SCALE_FLOOR,
            "all_mask_features_zero": True,
            "clean_upper_bound_semantics": (
                "clean input values with center, scale, and target normalization all "
                "derived from the paired corrupt context to isolate input repair"
            ),
            "consumer_input_dimension": 2 * CONTEXT_LENGTH,
        },
        "roster": roster_report,
        "information_wall": {
            "global_registry_metadata_loaded": True,
            "only_premise_roster_source_values_loaded": True,
            "premise_policy_evidence_or_outcomes_consulted": False,
            "fresh_replay_report_read": False,
            "support_a_validation_values_or_context_or_future_read": False,
            "support_b_values_or_context_or_future_read": False,
            "uci_target_or_query_values_or_context_or_future_read": False,
            "target_or_query_value_surface_read": False,
            "hidden_excursion_geometry_or_sign_available_to_observer_binding": False,
            "private_grader_diagnostics_used_by_binding_fitting_or_gate": False,
            "training_targets_used_only_after_observer_binding": True,
            "eval_future_used_only_for_post_fit_policy_evaluation": True,
        },
        "observer_binding_contract": {
            "public_extractor": "runtime.public_features.extract_public_features",
            "mapping_fields_read": [
                "level_excursion_score",
                "estimated_region_start_fraction",
                "estimated_region_end_fraction",
                "estimated_level_offset",
            ],
            "repair_operator": "operators.s1_structural.repair_level_shift",
            "activation_rule": (
                "call canonical repair only when all four public fields are finite, "
                "level score is positive, region satisfies 0<=start<end<=1, and "
                "estimated offset is nonzero; otherwise return identity"
            ),
            "hidden_or_private_grader_fields_read": False,
        },
        "corrupt_observer_and_private_grader_diagnostics": corruption_diagnostics,
        "zero_fit_clean_risk_diagnostic": clean_risk_diagnostics,
        "consumer_fit_count": consumer_fit_count,
        "expected_consumer_fit_count": expected_fit_count,
        "chronos_judge_call_count": 0,
        "policy_intervention_evidence": evidence_rows,
        "development_gate": {
            "thresholds_fixed_without_tuning": True,
            "exact_definition": (
                "pass iff mean corrupt-minus-clean degradation is strictly positive "
                "in both datasets, mean corrupt-minus-repair gain is strictly positive "
                "in both datasets, and pooled repair harm rate is <=0.25"
            ),
            "degradation_strictly_positive_by_dataset": {
                "values": degradation_by_dataset,
                "threshold": 0.0,
                "comparison": ">",
                "pass": degradation_pass,
            },
            "repair_gain_strictly_positive_by_dataset": {
                "values": repair_gain_by_dataset,
                "threshold": 0.0,
                "comparison": ">",
                "pass": repair_gain_pass,
            },
            "pooled_repair_harm_rate": {
                "value": pooled_harm_rate,
                "threshold": HARM_RATE_MAX,
                "comparison": "<=",
                "harm_definition": f"repair gain relative to corrupt < {HARM_GAIN_THRESHOLD}",
                "pass": harm_pass,
            },
            "clean_risk_is_reported_but_not_a_gate_input": True,
            "pass": gate_pass,
        },
        "verdict": (
            "LEVEL_COHORT_HEADROOM_PRESENT"
            if gate_pass
            else "LEVEL_COHORT_HEADROOM_WEAK"
        ),
        "promotion": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "query": False,
        "target_query_opened": False,
        "claim_limit": (
            "At most an exposed-development Source positive-control screen of "
            "cohort/model-level headroom and an observable repair binding under one "
            "synthetic transient level excursion; not individual causal evidence, "
            "Capability, promotion, formal transfer, Target, or Query evidence."
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
    report = run_e2_source_cohort_level_excursion_headroom(
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
