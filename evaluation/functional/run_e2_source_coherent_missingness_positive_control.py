"""Run the development-only coherent-missingness Consumer positive control.

Each training series is corrupted once as a complete artifact before any windows are
created.  A registered phase-median completion and a grader-only exact oracle are
compared with the corrupt incumbent under one fixed Ridge protocol.  This is neither
fresh evidence nor a promotion or Capability evaluation.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.linear_model import Ridge

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
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
    ROBUST_SCALE_FLOOR,
    _center_scale,
)
from SelfEvolvingHarnessTS.operators.registry import get_operator


SCHEMA_VERSION = "e2-source-coherent-missingness-positive-control/3"
SCIENTIFIC_ROLE = "development_method_facing_coherent_missingness_positive_control"
DISCOVERY_SUBSPLIT = "support_a_discovery"
SUPPORT_A_SUBSPLIT_SCHEMA = "benchmark-support-a-subsplit/2"
POLICIES = (
    "corrupt_identity",
    "grader_only_exact_oracle",
    "period_median_complete",
)
ORACLE_ONLY_POLICIES = POLICIES[:2]
MODES = ("full", "oracle_only")

TRAIN_SERIES_PER_DATASET = 12
EVAL_SERIES_PER_DATASET = 8
CONTEXT_LENGTH = 192
HORIZON = 48
RIDGE_ALPHA = 1.0
PHASE_CYCLES = 3
PHASE_MIN_DONORS = 2

P0_MEDIAN_RECOVERY_MIN = 0.50
P0_PHASE_BETTER_MIN_COUNT = 10
P1_ORACLE_MEAN_GAIN_MIN = 0.005
P1_ORACLE_MEDIAN_GAIN_MIN_EXCLUSIVE = 0.0
P1_ORACLE_POSITIVE_MIN_COUNT = 5
P1_PHASE_MEAN_GAIN_MIN_EXCLUSIVE = 0.0
P1_PHASE_HARM_THRESHOLD = -0.005
P1_PHASE_HARM_RATE_MAX = 0.25

EARLY_OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_coherent_missingness_positive_control_report.json"
)
RECENT_OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_coherent_missingness_recent_positive_control_report.json"
)
RECENT_ORACLE_OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/"
    "source_coherent_missingness_recent_oracle_diagnostic_report.json"
)


def _half_open_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


@dataclass(frozen=True)
class GeometrySpec:
    geometry_id: str
    anchors: tuple[int, ...]
    gap_bounds: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.geometry_id or self.geometry_id != self.geometry_id.strip():
            raise ValueError("geometry_id must be a canonical non-empty string")
        if not self.anchors or tuple(sorted(set(self.anchors))) != self.anchors:
            raise ValueError("geometry anchors must be unique and ascending")
        if any(anchor - CONTEXT_LENGTH < 0 for anchor in self.anchors):
            raise ValueError("geometry context starts before index zero")
        if not self.gap_bounds:
            raise ValueError("geometry must contain at least one gap")
        for start, stop in self.gap_bounds:
            if start < 0 or stop - start != 6:
                raise ValueError("every geometry gap must be a six-point half-open range")
        for index, bounds in enumerate(self.gap_bounds):
            if any(
                _half_open_overlap(bounds, other)
                for other in self.gap_bounds[index + 1 :]
            ):
                raise ValueError("geometry gaps overlap")
        target_bounds = tuple((anchor, anchor + HORIZON) for anchor in self.anchors)
        if any(
            _half_open_overlap(gap, target)
            for gap in self.gap_bounds
            for target in target_bounds
        ):
            raise ValueError("geometry gap overlaps a training target")

    @property
    def gap_point_count(self) -> int:
        return sum(stop - start for start, stop in self.gap_bounds)


EARLY_V1 = GeometrySpec(
    geometry_id="early_v1",
    anchors=(240, 264, 288, 312, 336, 360, 384),
    gap_bounds=((120, 126), (150, 156), (180, 186), (210, 216)),
)
RECENT_V2 = GeometrySpec(
    geometry_id="recent_v2",
    anchors=(240, 300, 360, 420, 480, 540),
    gap_bounds=(
        (234, 240),
        (294, 300),
        (354, 360),
        (414, 420),
        (474, 480),
        (534, 540),
    ),
)
GEOMETRY_BY_ID = MappingProxyType(
    {geometry.geometry_id: geometry for geometry in (EARLY_V1, RECENT_V2)}
)
OUTPUT_BY_GEOMETRY_ID = MappingProxyType(
    {
        EARLY_V1.geometry_id: EARLY_OUTPUT_RELATIVE_PATH,
        RECENT_V2.geometry_id: RECENT_OUTPUT_RELATIVE_PATH,
    }
)


def _policies_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "full":
        return POLICIES
    if mode == "oracle_only":
        return ORACLE_ONLY_POLICIES
    raise ValueError(f"unknown coherent-missingness mode: {mode!r}")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    period: int
    frequency: str
    train_stop: int
    validation_bounds: tuple[int, int]


DATASET_SPECS = (
    DatasetSpec(
        dataset_id="legacy_monash:fred_md",
        period=12,
        frequency="monthly",
        train_stop=632,
        validation_bounds=(632, 680),
    ),
    DatasetSpec(
        dataset_id="legacy_monash:nn5_daily",
        period=7,
        frequency="daily",
        train_stop=695,
        validation_bounds=(695, 743),
    ),
)
SPEC_BY_DATASET = {spec.dataset_id: spec for spec in DATASET_SPECS}
SOURCE_DATASETS = tuple(spec.dataset_id for spec in DATASET_SPECS)


@dataclass(frozen=True)
class RosterItem:
    record: SeriesRecord
    assignment: SplitAssignment
    cohort: str


@dataclass(frozen=True)
class PreparedSeries:
    clean: np.ndarray
    corrupt: np.ndarray
    oracle: np.ndarray
    phase: np.ndarray | None
    original_mask: np.ndarray
    p0: dict[str, object]


@dataclass(frozen=True)
class DatasetTrainingBundle:
    x_by_policy: dict[str, np.ndarray]
    y_train: np.ndarray
    diagnostics: dict[str, object]
    per_series_p0: list[dict[str, object]]
    p0: dict[str, object]


def _read_discovery_uids(path: Path) -> set[str]:
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema_version") != SUPPORT_A_SUBSPLIT_SCHEMA:
        raise ValueError("unsupported Support-A subsplit schema")
    members = payload.get("members")
    if not isinstance(members, dict):
        raise ValueError("Support-A subsplit members must be an object")
    raw = members.get(DISCOVERY_SUBSPLIT)
    if not isinstance(raw, list) or not all(isinstance(uid, str) and uid for uid in raw):
        raise ValueError("invalid support_a_discovery member list")
    if len(raw) != len(set(raw)):
        raise ValueError("duplicate support_a_discovery UID")
    counts = payload.get("counts")
    if not isinstance(counts, dict) or counts.get(DISCOVERY_SUBSPLIT) != len(raw):
        raise ValueError("support_a_discovery count disagrees with frozen metadata")
    return set(raw)


def _validate_candidate(
    record: SeriesRecord, assignment: SplitAssignment, spec: DatasetSpec
) -> None:
    uid = record.series_uid
    if record.dataset_id != spec.dataset_id or assignment.dataset_id != spec.dataset_id:
        raise ValueError(f"candidate dataset mismatch: {uid}")
    if record.regime_tag != assignment.regime_tag:
        raise ValueError(f"registry/split regime mismatch: {uid}")
    if record.admission_reasons != ():
        raise ValueError(f"ineligible Source discovery record: {uid}")
    if record.natural_missing_count != 0:
        raise ValueError(f"candidate has natural missingness: {uid}")
    if record.frequency != spec.frequency:
        raise ValueError(f"unexpected frequency for dataset semantics: {uid}")
    if SplitRole.SUPPORT_A.value not in record.roles_allowed:
        raise ValueError(f"record disallows Support-A: {uid}")
    if assignment.role is not SplitRole.SUPPORT_A:
        raise ValueError(f"discovery member is not Support-A: {uid}")
    boundaries = assignment.chronological_boundaries
    if boundaries is None:
        raise ValueError(f"missing chronological boundaries: {uid}")
    expected_test = (spec.validation_bounds[1], spec.validation_bounds[1] + HORIZON)
    if tuple(boundaries.get("train", ())) != (0, spec.train_stop):
        raise ValueError(f"unexpected train boundary: {uid}")
    if tuple(boundaries.get("validation", ())) != spec.validation_bounds:
        raise ValueError(f"unexpected validation boundary: {uid}")
    if tuple(boundaries.get("test", ())) != expected_test:
        raise ValueError(f"unexpected test boundary: {uid}")


def select_roster(
    *, registry_path: Path, split_path: Path, support_a_subsplit_path: Path
) -> tuple[list[RosterItem], dict[str, object]]:
    """Freeze exactly 12 train and 8 eval series per dataset from metadata only."""

    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    assignments = {row.series_uid: row for row in manifest.assignments}
    discovery_uids = _read_discovery_uids(support_a_subsplit_path)

    roster: list[RosterItem] = []
    selected_by_dataset: dict[str, dict[str, list[str]]] = {}
    available_by_dataset: dict[str, int] = {}
    required = TRAIN_SERIES_PER_DATASET + EVAL_SERIES_PER_DATASET
    for spec in DATASET_SPECS:
        candidates: list[tuple[SeriesRecord, SplitAssignment]] = []
        for uid in discovery_uids:
            assignment = assignments.get(uid)
            if assignment is None:
                raise ValueError(f"discovery UID absent from split manifest: {uid}")
            if assignment.dataset_id != spec.dataset_id:
                continue
            record = records.get(uid)
            if record is None:
                raise ValueError(f"discovery UID absent from registry: {uid}")
            _validate_candidate(record, assignment, spec)
            candidates.append((record, assignment))
        candidates.sort(key=lambda pair: pair[0].series_uid)
        available_by_dataset[spec.dataset_id] = len(candidates)
        if len(candidates) != required:
            raise ValueError(
                f"expected exactly {required} eligible clean discovery series: "
                f"{spec.dataset_id}, observed {len(candidates)}"
            )
        train = candidates[:TRAIN_SERIES_PER_DATASET]
        evaluate = candidates[TRAIN_SERIES_PER_DATASET:]
        roster.extend(RosterItem(record, assignment, "train") for record, assignment in train)
        roster.extend(RosterItem(record, assignment, "eval") for record, assignment in evaluate)
        selected_by_dataset[spec.dataset_id] = {
            "train": [record.series_uid for record, _ in train],
            "eval": [record.series_uid for record, _ in evaluate],
        }

    train_uids = {item.record.series_uid for item in roster if item.cohort == "train"}
    eval_uids = {item.record.series_uid for item in roster if item.cohort == "eval"}
    if len(train_uids) != len(DATASET_SPECS) * TRAIN_SERIES_PER_DATASET:
        raise AssertionError("training roster contains duplicate series")
    if len(eval_uids) != len(DATASET_SPECS) * EVAL_SERIES_PER_DATASET:
        raise AssertionError("evaluation roster contains duplicate series")
    if train_uids & eval_uids:
        raise AssertionError("training and evaluation rosters overlap")
    return roster, {
        "fixed_before_selected_value_loading": True,
        "selection_rule": (
            "per dataset, filter frozen support_a_discovery to eligible records with "
            "natural_missing_count=0; sort by series_uid; first 12 train, next 8 eval"
        ),
        "selection_features": [
            "support_a_discovery_membership",
            "dataset_id",
            "registry_eligibility",
            "natural_missing_count",
            "series_uid",
        ],
        "available_by_dataset": available_by_dataset,
        "selected_by_dataset": selected_by_dataset,
        "train_eval_series_disjoint": True,
    }


def _fixed_gap_mask(length: int, *, geometry: GeometrySpec) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, stop in geometry.gap_bounds:
        if not 0 <= start < stop <= length:
            raise ValueError("fixed gap falls outside the training artifact")
        if mask[start:stop].any():
            raise AssertionError("fixed gaps overlap")
        mask[start:stop] = True
    if int(mask.sum()) != geometry.gap_point_count:
        raise AssertionError("fixed artifact mask has the wrong point count")
    return mask


def _donor_diagnostics(
    corrupt: np.ndarray,
    mask: np.ndarray,
    *,
    period: int,
    geometry: GeometrySpec,
) -> dict[str, object]:
    donor_counts: dict[int, int] = {}
    for index in np.flatnonzero(mask):
        donor_counts[int(index)] = sum(
            1
            for cycle in range(1, PHASE_CYCLES + 1)
            if index - cycle * period >= 0
            and np.isfinite(corrupt[index - cycle * period])
        )
    fallback_points = [
        index for index, count in donor_counts.items() if count < PHASE_MIN_DONORS
    ]
    by_gap: list[dict[str, object]] = []
    for start, stop in geometry.gap_bounds:
        counts = [donor_counts[index] for index in range(start, stop)]
        by_gap.append(
            {
                "bounds": [start, stop],
                "min_prior_donor_count": min(counts),
                "max_prior_donor_count": max(counts),
                "fallback_point_count": sum(count < PHASE_MIN_DONORS for count in counts),
            }
        )
    return {
        "cycles": PHASE_CYCLES,
        "min_donors": PHASE_MIN_DONORS,
        "min_prior_donor_count": min(donor_counts.values()),
        "max_prior_donor_count": max(donor_counts.values()),
        "fallback_point_count": len(fallback_points),
        "fallback_points": fallback_points,
        "by_gap": by_gap,
    }


def _prepare_artifact_policies(
    clean_train: np.ndarray,
    *,
    spec: DatasetSpec,
    series_uid: str,
    geometry: GeometrySpec,
    mode: str,
) -> PreparedSeries:
    """Corrupt and execute the mode's policies once on a complete train artifact."""

    clean = np.asarray(clean_train, dtype=np.float64).copy()
    if clean.shape != (spec.train_stop,) or not np.isfinite(clean).all():
        raise ValueError(f"invalid clean train artifact: {series_uid}")
    mask = _fixed_gap_mask(spec.train_stop, geometry=geometry)
    corrupt = clean.copy()
    corrupt[mask] = np.nan
    if int(np.isnan(corrupt).sum()) != geometry.gap_point_count:
        raise AssertionError("corrupt artifact has the wrong injected NaN count")

    oracle = corrupt.copy()
    oracle[mask] = clean[mask]
    oracle_gap_max_abs_error = float(np.max(np.abs(oracle[mask] - clean[mask])))
    oracle_collateral_change_count = int(np.count_nonzero(oracle[~mask] != corrupt[~mask]))
    oracle_mechanical = (
        oracle_gap_max_abs_error == 0.0 and oracle_collateral_change_count == 0
    )

    phase: np.ndarray | None = None
    donor: dict[str, object] | None = None
    phase_fill_count: int | None = None
    phase_fill_rate: float | None = None
    phase_observed_change_count: int | None = None
    clean_scale: float | None = None
    clean_scale_method: str | None = None
    finite_median: float | None = None
    phase_gap_nmae: float | None = None
    baseline_gap_nmae: float | None = None
    recovery: float | None = None
    phase_better: bool | None = None
    phase_mechanical: bool | None = None
    if mode == "full":
        operator = get_operator("period_median_complete")
        phase = np.asarray(
            operator(
                corrupt,
                period=spec.period,
                cycles=PHASE_CYCLES,
                min_donors=PHASE_MIN_DONORS,
            ),
            dtype=np.float64,
        )
        if phase.shape != corrupt.shape:
            raise RuntimeError(
                f"period_median_complete changed artifact shape: {series_uid}"
            )
        donor = _donor_diagnostics(
            corrupt, mask, period=spec.period, geometry=geometry
        )
        phase_fill_count = int(np.count_nonzero(np.isfinite(phase[mask])))
        phase_fill_rate = phase_fill_count / geometry.gap_point_count
        phase_observed_change_count = int(np.count_nonzero(phase[~mask] != corrupt[~mask]))
        _, clean_scale, clean_scale_method = _center_scale(clean)
        finite_median = float(np.median(corrupt[np.isfinite(corrupt)]))
        baseline = corrupt.copy()
        baseline[mask] = finite_median
        phase_gap_nmae = float(
            np.mean(np.abs(phase[mask] - clean[mask])) / clean_scale
        )
        baseline_gap_nmae = float(
            np.mean(np.abs(baseline[mask] - clean[mask])) / clean_scale
        )
        if baseline_gap_nmae <= 0.0:
            raise ValueError(f"finite-median baseline has zero gap NMAE: {series_uid}")
        recovery = (baseline_gap_nmae - phase_gap_nmae) / baseline_gap_nmae
        phase_better = phase_gap_nmae < baseline_gap_nmae
        phase_mechanical = (
            phase_fill_rate == 1.0
            and phase_observed_change_count == 0
            and donor["fallback_point_count"] == 0
        )
    elif mode != "oracle_only":
        raise ValueError(f"unknown coherent-missingness mode: {mode!r}")

    p0 = {
        "series_uid": series_uid,
        "dataset_id": spec.dataset_id,
        "geometry_id": geometry.geometry_id,
        "mode": mode,
        "period": spec.period,
        "artifact_bounds": [0, spec.train_stop],
        "injected_gap_point_count": geometry.gap_point_count,
        "corrupt_artifact_mask_exact": (
            int(np.isnan(corrupt).sum()) == geometry.gap_point_count
            and np.array_equal(np.isnan(corrupt), mask)
        ),
        "donor_diagnostics": donor,
        "oracle_gap_max_abs_error": oracle_gap_max_abs_error,
        "oracle_collateral_change_count": oracle_collateral_change_count,
        "oracle_mechanical_checks_pass": oracle_mechanical,
        "phase_status": "completed" if mode == "full" else "not_run",
        "phase_fill_count": phase_fill_count,
        "phase_fill_rate": phase_fill_rate,
        "phase_observed_change_count": phase_observed_change_count,
        "phase_fallback_point_count": (
            donor["fallback_point_count"] if donor is not None else None
        ),
        "clean_train_scale": clean_scale,
        "clean_train_scale_method": clean_scale_method,
        "finite_value_median_completion_baseline": finite_median,
        "phase_gap_nmae": phase_gap_nmae,
        "finite_median_baseline_gap_nmae": baseline_gap_nmae,
        "gap_nmae_recovery_fraction": recovery,
        "phase_better_than_finite_median_baseline": phase_better,
        "phase_mechanical_checks_pass": phase_mechanical,
        "mechanical_checks_pass": (
            oracle_mechanical
            and (phase_mechanical is True if mode == "full" else True)
        ),
    }
    return PreparedSeries(clean, corrupt, oracle, phase, mask, p0)


def _training_bundle(
    *,
    spec: DatasetSpec,
    geometry: GeometrySpec,
    mode: str,
    train_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
) -> DatasetTrainingBundle:
    items = sorted(train_items, key=lambda item: item.record.series_uid)
    if len(items) != TRAIN_SERIES_PER_DATASET:
        raise ValueError(f"unexpected training roster size: {spec.dataset_id}")
    active_policies = _policies_for_mode(mode)
    x_rows: dict[str, list[np.ndarray]] = {policy: [] for policy in active_policies}
    y_rows: list[np.ndarray] = []
    p0_rows: list[dict[str, object]] = []
    scale_method_counts: dict[str, int] = {}
    target_checks: list[dict[str, object]] = []

    for item in items:
        uid = item.record.series_uid
        full_values = values_by_uid[uid]
        prepared = _prepare_artifact_policies(
            np.asarray(full_values[: spec.train_stop], dtype=np.float64),
            spec=spec,
            series_uid=uid,
            geometry=geometry,
            mode=mode,
        )
        p0_rows.append(prepared.p0)
        artifact_by_policy = {
            "corrupt_identity": prepared.corrupt,
            "grader_only_exact_oracle": prepared.oracle,
        }
        if mode == "full":
            if prepared.phase is None:
                raise AssertionError("full mode phase artifact was not prepared")
            artifact_by_policy["period_median_complete"] = prepared.phase
        for anchor in geometry.anchors:
            context_start = anchor - CONTEXT_LENGTH
            target_stop = anchor + HORIZON
            if context_start < 0 or target_stop > spec.train_stop:
                raise AssertionError("training window crosses the frozen train boundary")
            target_overlap_count = int(prepared.original_mask[anchor:target_stop].sum())
            if target_overlap_count != 0:
                raise AssertionError("training target overlaps the injected artifact mask")

            corrupt_context = prepared.corrupt[context_start:anchor]
            original_mask = ~np.isfinite(corrupt_context)
            center, scale, scale_method = _center_scale(corrupt_context)
            scale_method_counts[scale_method] = scale_method_counts.get(scale_method, 0) + 1

            # The label is deliberately sliced from the corrupt artifact.  Its lack of
            # overlap with the fixed mask makes it finite; no clean replacement occurs.
            target = np.asarray(prepared.corrupt[anchor:target_stop], dtype=np.float64).copy()
            if target.shape != (HORIZON,) or not np.isfinite(target).all():
                raise ValueError(f"invalid corrupt-artifact target: {uid}/{anchor}")
            normalized_target = (target - center) / scale
            y_rows.append(normalized_target)

            mask_features: list[np.ndarray] = []
            for policy in active_policies:
                context = np.asarray(
                    artifact_by_policy[policy][context_start:anchor], dtype=np.float64
                ).copy()
                normalized = (context - center) / scale
                normalized[~np.isfinite(normalized)] = 0.0
                features = np.concatenate((normalized, original_mask.astype(np.float64)))
                if not np.isfinite(features).all():
                    raise ValueError(f"non-finite training features: {uid}/{anchor}/{policy}")
                x_rows[policy].append(features)
                mask_features.append(features[CONTEXT_LENGTH:])
            masks_identical = all(
                np.array_equal(mask_features[0], mask) for mask in mask_features[1:]
            )
            if not masks_identical:
                raise AssertionError("original mask features differ across policies")
            target_checks.append(
                {
                    "series_uid": uid,
                    "anchor": anchor,
                    "target_bounds": [anchor, target_stop],
                    "target_overlap_with_artifact_mask_count": target_overlap_count,
                    "target_source": "corrupt_full_artifact",
                    "center_scale_source": "finite_values_of_corrupt_context_once",
                    "original_mask_identical_across_policies": masks_identical,
                }
            )

    expected = TRAIN_SERIES_PER_DATASET * len(geometry.anchors)
    x_by_policy = {
        policy: np.asarray(rows, dtype=np.float64) for policy, rows in x_rows.items()
    }
    y_train = np.asarray(y_rows, dtype=np.float64)
    if y_train.shape != (expected, HORIZON):
        raise AssertionError("unexpected coherent-missingness target matrix shape")
    for policy, matrix in x_by_policy.items():
        if matrix.shape != (expected, 2 * CONTEXT_LENGTH):
            raise AssertionError(f"unexpected training matrix shape: {policy}")
    masks_shared = all(
        np.array_equal(
            x_by_policy[POLICIES[0]][:, CONTEXT_LENGTH:],
            x_by_policy[policy][:, CONTEXT_LENGTH:],
        )
        for policy in active_policies[1:]
    )
    median_recovery = (
        statistics.median(
            float(row["gap_nmae_recovery_fraction"]) for row in p0_rows
        )
        if mode == "full"
        else None
    )
    phase_better_count = (
        sum(bool(row["phase_better_than_finite_median_baseline"]) for row in p0_rows)
        if mode == "full"
        else None
    )
    all_mechanical = all(bool(row["mechanical_checks_pass"]) for row in p0_rows)
    all_targets_nonoverlap = all(
        int(row["target_overlap_with_artifact_mask_count"]) == 0 for row in target_checks
    )
    p0_pass = all_mechanical and masks_shared and all_targets_nonoverlap
    if mode == "full":
        p0_pass = (
            p0_pass
            and median_recovery is not None
            and median_recovery >= P0_MEDIAN_RECOVERY_MIN
            and phase_better_count is not None
            and phase_better_count >= P0_PHASE_BETTER_MIN_COUNT
        )
    return DatasetTrainingBundle(
        x_by_policy=x_by_policy,
        y_train=y_train,
        diagnostics={
            "scale_method_counts": scale_method_counts,
            "full_artifact_policy_execution_count_per_policy": TRAIN_SERIES_PER_DATASET,
            "active_policies": list(active_policies),
            "phase_status": "completed" if mode == "full" else "not_run",
            "window_count": expected,
            "targets_are_shared_across_policies": True,
            "targets_sliced_directly_from_corrupt_artifact": True,
            "original_mask_features_shared_across_policies": masks_shared,
            "target_checks": target_checks,
        },
        per_series_p0=p0_rows,
        p0={
            "dataset_id": spec.dataset_id,
            "geometry_id": geometry.geometry_id,
            "mode": mode,
            "checked_before_any_consumer_fit": True,
            "consumer_fit_count_at_check": 0,
            "all_series_mechanical_checks_pass": all_mechanical,
            "all_training_targets_nonoverlapping_with_mask": all_targets_nonoverlap,
            "original_masks_identical_across_policies": masks_shared,
            "median_series_gap_nmae_recovery_fraction": median_recovery,
            "median_series_gap_nmae_recovery_min": (
                P0_MEDIAN_RECOVERY_MIN if mode == "full" else None
            ),
            "phase_better_than_baseline_series_count": phase_better_count,
            "phase_better_than_baseline_series_min_count": (
                P0_PHASE_BETTER_MIN_COUNT if mode == "full" else None
            ),
            "phase_recovery_strength_gate_status": (
                "required" if mode == "full" else "not_run"
            ),
            "train_series_count": TRAIN_SERIES_PER_DATASET,
            "pass": p0_pass,
        },
    )


def _evaluation_matrices(
    *,
    spec: DatasetSpec,
    eval_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    items = sorted(eval_items, key=lambda item: item.record.series_uid)
    if len(items) != EVAL_SERIES_PER_DATASET:
        raise ValueError(f"unexpected evaluation roster size: {spec.dataset_id}")
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    uids: list[str] = []
    scale_method_counts: dict[str, int] = {}
    context_bounds = (spec.train_stop - CONTEXT_LENGTH, spec.train_stop)
    for item in items:
        uid = item.record.series_uid
        values = values_by_uid[uid]
        context = np.asarray(values[slice(*context_bounds)], dtype=np.float64).copy()
        future = np.asarray(values[slice(*spec.validation_bounds)], dtype=np.float64).copy()
        if context.shape != (CONTEXT_LENGTH,) or future.shape != (HORIZON,):
            raise ValueError(f"insufficient held-out evaluation window: {uid}")
        if not np.isfinite(context).all() or not np.isfinite(future).all():
            raise ValueError(f"non-finite held-out clean evaluation window: {uid}")
        center, scale, scale_method = _center_scale(context)
        features = np.concatenate(
            ((context - center) / scale, np.zeros(CONTEXT_LENGTH, dtype=np.float64))
        )
        target = (future - center) / scale
        x_rows.append(features)
        y_rows.append(target)
        uids.append(uid)
        scale_method_counts[scale_method] = scale_method_counts.get(scale_method, 0) + 1
    x_eval = np.asarray(x_rows, dtype=np.float64)
    y_eval = np.asarray(y_rows, dtype=np.float64)
    if x_eval.shape != (EVAL_SERIES_PER_DATASET, 2 * CONTEXT_LENGTH):
        raise AssertionError("unexpected coherent-missingness evaluation input shape")
    if y_eval.shape != (EVAL_SERIES_PER_DATASET, HORIZON):
        raise AssertionError("unexpected coherent-missingness evaluation target shape")
    return x_eval, y_eval, uids, {
        "context_bounds": list(context_bounds),
        "future_bounds": list(spec.validation_bounds),
        "scale_method_counts": scale_method_counts,
        "clean_context_and_future": True,
        "zero_mask_feature_nonzero_count": int(
            np.count_nonzero(x_eval[:, CONTEXT_LENGTH:])
        ),
        "identical_matrix_shared_across_active_policies": True,
    }


def _dataset_evidence(
    *,
    spec: DatasetSpec,
    geometry: GeometrySpec,
    losses_by_policy: dict[str, list[float]],
    eval_uids: list[str],
    training_diagnostics: dict[str, object],
    evaluation_diagnostics: dict[str, object],
) -> dict[str, object]:
    if any(len(losses_by_policy[policy]) != len(eval_uids) for policy in POLICIES):
        raise AssertionError("paired coherent-missingness evidence lengths disagree")
    paired: list[dict[str, object]] = []
    for index, uid in enumerate(eval_uids):
        incumbent_loss = losses_by_policy["corrupt_identity"][index]
        oracle_loss = losses_by_policy["grader_only_exact_oracle"][index]
        phase_loss = losses_by_policy["period_median_complete"][index]
        oracle_gain = incumbent_loss - oracle_loss
        phase_gain = incumbent_loss - phase_loss
        paired.append(
            {
                "series_uid": uid,
                "corrupt_identity_normalized_mae": incumbent_loss,
                "grader_only_exact_oracle_normalized_mae": oracle_loss,
                "period_median_complete_normalized_mae": phase_loss,
                "oracle_gain_over_incumbent": oracle_gain,
                "phase_gain_over_incumbent": phase_gain,
                "oracle_positive_gain": oracle_gain > 0.0,
                "phase_harmed": phase_gain < P1_PHASE_HARM_THRESHOLD,
            }
        )
    oracle_gains = [float(row["oracle_gain_over_incumbent"]) for row in paired]
    phase_gains = [float(row["phase_gain_over_incumbent"]) for row in paired]
    oracle_mean = statistics.fmean(oracle_gains)
    oracle_median = statistics.median(oracle_gains)
    oracle_positive_count = sum(bool(row["oracle_positive_gain"]) for row in paired)
    phase_mean = statistics.fmean(phase_gains)
    oracle_gate_pass = (
        oracle_mean >= P1_ORACLE_MEAN_GAIN_MIN
        and oracle_median > P1_ORACLE_MEDIAN_GAIN_MIN_EXCLUSIVE
        and oracle_positive_count >= P1_ORACLE_POSITIVE_MIN_COUNT
    )
    phase_mean_gate_pass = phase_mean > P1_PHASE_MEAN_GAIN_MIN_EXCLUSIVE
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "scientific_unit": "dataset_level_development_same_family_cohort",
        "dataset_id": spec.dataset_id,
        "geometry_id": geometry.geometry_id,
        "period": spec.period,
        "policy_mean_normalized_mae": {
            policy: statistics.fmean(losses) for policy, losses in losses_by_policy.items()
        },
        "policy_median_normalized_mae": {
            policy: statistics.median(losses) for policy, losses in losses_by_policy.items()
        },
        "mean_oracle_gain_over_incumbent": oracle_mean,
        "median_oracle_gain_over_incumbent": oracle_median,
        "oracle_positive_gain_count": oracle_positive_count,
        "mean_phase_gain_over_incumbent": phase_mean,
        "train_cohort": {
            "series_count": TRAIN_SERIES_PER_DATASET,
            "anchor_count_per_series": len(geometry.anchors),
            "example_count": TRAIN_SERIES_PER_DATASET * len(geometry.anchors),
            "diagnostics": training_diagnostics,
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
            "diagnostics": evaluation_diagnostics,
        },
        "paired_eval_rows": paired,
        "dataset_gates": {
            "oracle": {
                "mean_gain_min": P1_ORACLE_MEAN_GAIN_MIN,
                "median_gain_must_exceed": P1_ORACLE_MEDIAN_GAIN_MIN_EXCLUSIVE,
                "positive_gain_min_count": P1_ORACLE_POSITIVE_MIN_COUNT,
                "eval_series_count": EVAL_SERIES_PER_DATASET,
                "pass": oracle_gate_pass,
            },
            "phase_mean": {
                "mean_gain_must_exceed": P1_PHASE_MEAN_GAIN_MIN_EXCLUSIVE,
                "pass": phase_mean_gate_pass,
            },
        },
    }


def _oracle_dataset_evidence(
    *,
    spec: DatasetSpec,
    geometry: GeometrySpec,
    losses_by_policy: dict[str, list[float]],
    eval_uids: list[str],
    training_diagnostics: dict[str, object],
    evaluation_diagnostics: dict[str, object],
) -> dict[str, object]:
    incumbent_losses = losses_by_policy["corrupt_identity"]
    oracle_losses = losses_by_policy["grader_only_exact_oracle"]
    if not (len(incumbent_losses) == len(oracle_losses) == len(eval_uids)):
        raise AssertionError("paired oracle-only evidence lengths disagree")
    paired: list[dict[str, object]] = []
    for uid, incumbent_loss, oracle_loss in zip(
        eval_uids, incumbent_losses, oracle_losses
    ):
        gain = incumbent_loss - oracle_loss
        paired.append(
            {
                "series_uid": uid,
                "corrupt_identity_normalized_mae": incumbent_loss,
                "grader_only_exact_oracle_normalized_mae": oracle_loss,
                "oracle_gain_over_incumbent": gain,
                "oracle_positive_gain": gain > 0.0,
            }
        )
    gains = [float(row["oracle_gain_over_incumbent"]) for row in paired]
    mean_gain = statistics.fmean(gains)
    median_gain = statistics.median(gains)
    positive_count = sum(bool(row["oracle_positive_gain"]) for row in paired)
    gate_pass = (
        mean_gain >= P1_ORACLE_MEAN_GAIN_MIN
        and median_gain > P1_ORACLE_MEDIAN_GAIN_MIN_EXCLUSIVE
        and positive_count >= P1_ORACLE_POSITIVE_MIN_COUNT
    )
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "scientific_unit": "dataset_level_development_oracle_fault_localization",
        "dataset_id": spec.dataset_id,
        "geometry_id": geometry.geometry_id,
        "mode": "oracle_only",
        "period": spec.period,
        "policy_mean_normalized_mae": {
            policy: statistics.fmean(losses) for policy, losses in losses_by_policy.items()
        },
        "policy_median_normalized_mae": {
            policy: statistics.median(losses) for policy, losses in losses_by_policy.items()
        },
        "mean_oracle_gain_over_incumbent": mean_gain,
        "median_oracle_gain_over_incumbent": median_gain,
        "oracle_positive_gain_count": positive_count,
        "train_cohort": {
            "series_count": TRAIN_SERIES_PER_DATASET,
            "anchor_count_per_series": len(geometry.anchors),
            "example_count": TRAIN_SERIES_PER_DATASET * len(geometry.anchors),
            "diagnostics": training_diagnostics,
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
            "diagnostics": evaluation_diagnostics,
        },
        "paired_eval_rows": paired,
        "dataset_gates": {
            "oracle": {
                "mean_gain_min": P1_ORACLE_MEAN_GAIN_MIN,
                "median_gain_must_exceed": P1_ORACLE_MEDIAN_GAIN_MIN_EXCLUSIVE,
                "positive_gain_min_count": P1_ORACLE_POSITIVE_MIN_COUNT,
                "eval_series_count": EVAL_SERIES_PER_DATASET,
                "pass": gate_pass,
            }
        },
    }


def _roster_report(
    roster: list[RosterItem], selection: dict[str, object]
) -> dict[str, object]:
    return {
        "selection": selection,
        "members": [
            {
                "dataset_id": item.record.dataset_id,
                "series_uid": item.record.series_uid,
                "entity_id": item.record.entity_id,
                "cohort": item.cohort,
                "split_role": item.assignment.role.value,
                "subsplit": DISCOVERY_SUBSPLIT,
                "natural_missing_count": item.record.natural_missing_count,
            }
            for item in sorted(
                roster,
                key=lambda row: (row.record.dataset_id, row.cohort, row.record.series_uid),
            )
        ],
    }


def run_e2_source_coherent_missingness_positive_control(
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
    geometry: GeometrySpec = EARLY_V1,
    mode: str = "full",
) -> dict[str, object]:
    active_policies = _policies_for_mode(mode)
    if mode == "oracle_only" and geometry != RECENT_V2:
        raise ValueError("oracle_only mode is frozen to recent_v2 geometry")
    roster, selection = select_roster(
        registry_path=registry_path,
        split_path=split_path,
        support_a_subsplit_path=support_a_subsplit_path,
    )
    values_by_uid = _load_values([item.record for item in roster], clean_root)

    bundles: dict[str, DatasetTrainingBundle] = {}
    for spec in DATASET_SPECS:
        train_items = [
            item
            for item in roster
            if item.record.dataset_id == spec.dataset_id and item.cohort == "train"
        ]
        bundles[spec.dataset_id] = _training_bundle(
            spec=spec,
            geometry=geometry,
            mode=mode,
            train_items=train_items,
            values_by_uid=values_by_uid,
        )

    p0_by_dataset = {
        dataset_id: bundle.p0 for dataset_id, bundle in bundles.items()
    }
    p0_pass = all(bool(row["pass"]) for row in p0_by_dataset.values())
    if not p0_pass:
        evidence_rows: list[dict[str, object]] = []
        consumer_fit_count = 0
        p1_report: dict[str, object] = {
            "status": "not_run",
            "reason": "P0 local recoverability conjunction failed",
            "gain_definition": (
                "corrupt_identity normalized MAE minus repaired-policy normalized MAE"
            ),
            "oracle_gate_by_dataset": {},
            "phase_mean_gate_by_dataset": (
                {} if mode == "full" else {"status": "not_applicable"}
            ),
            "pooled_phase_harm": {
                "status": "not_run" if mode == "full" else "not_applicable"
            },
            "all_conditions_are_conjunctive": True,
            "pass": False,
        }
        verdict = (
            "COHERENT_MISSINGNESS_ORACLE_CONSUMER_READABLE_NOT_ESTABLISHED"
            if mode == "oracle_only"
            else "COHERENT_MISSINGNESS_POSITIVE_CONTROL_NOT_ESTABLISHED"
        )
    else:
        evidence_rows = []
        consumer_fit_count = 0
        for spec in DATASET_SPECS:
            bundle = bundles[spec.dataset_id]
            eval_items = [
                item
                for item in roster
                if item.record.dataset_id == spec.dataset_id and item.cohort == "eval"
            ]
            x_eval, y_eval, eval_uids, eval_diagnostics = _evaluation_matrices(
                spec=spec,
                eval_items=eval_items,
                values_by_uid=values_by_uid,
            )
            losses_by_policy: dict[str, list[float]] = {}
            for policy in active_policies:
                model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd")
                model.fit(bundle.x_by_policy[policy], bundle.y_train)
                consumer_fit_count += 1
                prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
                if prediction.shape != y_eval.shape or not np.isfinite(prediction).all():
                    raise RuntimeError(
                        f"invalid Ridge prediction: {spec.dataset_id}/{policy}"
                    )
                losses_by_policy[policy] = [
                    float(loss) for loss in np.mean(np.abs(prediction - y_eval), axis=1)
                ]
            if mode == "oracle_only":
                evidence_rows.append(
                    _oracle_dataset_evidence(
                        spec=spec,
                        geometry=geometry,
                        losses_by_policy=losses_by_policy,
                        eval_uids=eval_uids,
                        training_diagnostics=bundle.diagnostics,
                        evaluation_diagnostics=eval_diagnostics,
                    )
                )
            else:
                evidence_rows.append(
                    _dataset_evidence(
                        spec=spec,
                        geometry=geometry,
                        losses_by_policy=losses_by_policy,
                        eval_uids=eval_uids,
                        training_diagnostics=bundle.diagnostics,
                        evaluation_diagnostics=eval_diagnostics,
                    )
                )

        expected_fit_count = len(DATASET_SPECS) * len(active_policies)
        if consumer_fit_count != expected_fit_count:
            raise AssertionError(
                f"expected exactly {expected_fit_count} independent Ridge Consumer fits"
            )

        oracle_gate_by_dataset = {
            str(row["dataset_id"]): row["dataset_gates"]["oracle"]  # type: ignore[index]
            for row in evidence_rows
        }
        oracle_conjunction_pass = all(
            bool(gate["pass"]) for gate in oracle_gate_by_dataset.values()  # type: ignore[index]
        )
        if mode == "oracle_only":
            p1_pass = oracle_conjunction_pass
            p1_report = {
                "status": "completed",
                "gain_definition": (
                    "corrupt_identity normalized MAE minus "
                    "grader_only_exact_oracle normalized MAE"
                ),
                "oracle_gate_by_dataset": oracle_gate_by_dataset,
                "phase_mean_gate_by_dataset": {"status": "not_applicable"},
                "pooled_phase_harm": {"status": "not_applicable"},
                "all_conditions_are_conjunctive": True,
                "pass": p1_pass,
            }
            verdict = (
                "COHERENT_MISSINGNESS_ORACLE_CONSUMER_READABLE"
                if p1_pass
                else "COHERENT_MISSINGNESS_ORACLE_CONSUMER_READABLE_NOT_ESTABLISHED"
            )
        else:
            phase_mean_gate_by_dataset = {
                str(row["dataset_id"]): row["dataset_gates"]["phase_mean"]  # type: ignore[index]
                for row in evidence_rows
            }
            paired_rows = [
                paired
                for row in evidence_rows
                for paired in row["paired_eval_rows"]  # type: ignore[union-attr]
            ]
            phase_harm_count = sum(bool(row["phase_harmed"]) for row in paired_rows)
            phase_harm_rate = phase_harm_count / len(paired_rows)
            pooled_phase_harm_pass = phase_harm_rate <= P1_PHASE_HARM_RATE_MAX
            p1_pass = (
                oracle_conjunction_pass
                and all(
                    bool(gate["pass"])
                    for gate in phase_mean_gate_by_dataset.values()  # type: ignore[index]
                )
                and pooled_phase_harm_pass
            )
            p1_report = {
                "status": "completed",
                "gain_definition": (
                    "corrupt_identity normalized MAE minus repaired-policy normalized MAE"
                ),
                "oracle_gate_by_dataset": oracle_gate_by_dataset,
                "phase_mean_gate_by_dataset": phase_mean_gate_by_dataset,
                "pooled_phase_harm": {
                    "harm_definition": f"phase gain < {P1_PHASE_HARM_THRESHOLD}",
                    "harm_count": phase_harm_count,
                    "eval_series_count": len(paired_rows),
                    "harm_rate": phase_harm_rate,
                    "harm_rate_max": P1_PHASE_HARM_RATE_MAX,
                    "pass": pooled_phase_harm_pass,
                },
                "all_conditions_are_conjunctive": True,
                "pass": p1_pass,
            }
            verdict = (
                "COHERENT_MISSINGNESS_POSITIVE_CONTROL_PRESENT"
                if p1_pass
                else "COHERENT_MISSINGNESS_POSITIVE_CONTROL_NOT_ESTABLISHED"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": SCIENTIFIC_ROLE,
        "geometry_id": geometry.geometry_id,
        "mode": mode,
        "diagnostic_only": mode == "oracle_only",
        "configuration": {
            "mode": mode,
            "geometry": {
                "geometry_id": geometry.geometry_id,
                "train_anchors": list(geometry.anchors),
                "gap_bounds_absolute_half_open": [
                    list(bounds) for bounds in geometry.gap_bounds
                ],
                "gaps_immediately_precede_matching_anchors": (
                    len(geometry.gap_bounds) == len(geometry.anchors)
                    and all(
                        stop == anchor
                        for (_start, stop), anchor in zip(
                            geometry.gap_bounds, geometry.anchors
                        )
                    )
                ),
                "all_gap_target_pairs_nonoverlapping": True,
            },
            "datasets": [
                {
                    "dataset_id": spec.dataset_id,
                    "period": spec.period,
                    "frequency": spec.frequency,
                    "train_bounds": [0, spec.train_stop],
                    "validation_bounds": list(spec.validation_bounds),
                }
                for spec in DATASET_SPECS
            ],
            "split": SplitRole.SUPPORT_A.value,
            "subsplit": DISCOVERY_SUBSPLIT,
            "policies": list(active_policies),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "artifact_gap_dose": {
                "gap_count": len(geometry.gap_bounds),
                "points_per_gap": [
                    stop - start for start, stop in geometry.gap_bounds
                ],
                "total_missing_point_count": geometry.gap_point_count,
                "fraction_of_train_artifact_by_dataset": {
                    spec.dataset_id: geometry.gap_point_count / spec.train_stop
                    for spec in DATASET_SPECS
                },
            },
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "artifact_first_protocol": (
                "inject one fixed mask into complete clean [0:train_stop] artifact; "
                "execute each active policy once on that artifact; only then windowize"
            ),
            "period_median_complete": {
                "status": "completed" if mode == "full" else "not_run",
                "registry_lookup": "get_operator('period_median_complete')",
                "period_source": "frozen dataset semantics",
                "cycles": PHASE_CYCLES,
                "min_donors": PHASE_MIN_DONORS,
            },
            "per_anchor_standardization": {
                "center": "median of finite corrupt-context values, computed once",
                "primary_scale": "1.4826 * median absolute deviation",
                "fallback": "population std when >=1e-6, otherwise 1e-6",
                "scale_floor": ROBUST_SCALE_FLOOR,
                "shared_across_three_policy_inputs_and_target": mode == "full",
                "shared_across_active_policy_inputs_and_target": True,
            },
            "consumer": {
                "class": "sklearn.linear_model.Ridge",
                "alpha": RIDGE_ALPHA,
                "fit_intercept": True,
                "solver": "svd",
                "expected_fit_count": len(DATASET_SPECS) * len(active_policies),
            },
            "agent_enabled": False,
            "memory_enabled": False,
            "promotion_enabled": False,
            "transfer_enabled": False,
        },
        "roster": _roster_report(roster, selection),
        "p0_local_recoverability_gate": {
            "dataset_results": p0_by_dataset,
            "per_series_results": {
                dataset_id: bundle.per_series_p0
                for dataset_id, bundle in bundles.items()
            },
            "all_datasets_required": True,
            "failure_action": "hard stop before all Ridge fits",
            "pass": p0_pass,
        },
        "policy_intervention_evidence": evidence_rows,
        "p1_consumer_sensitivity_gate": p1_report,
        "information_wall": {
            "roster_fixed_from_metadata_before_selected_value_loading": True,
            "source_only": True,
            "support_a_discovery_only": True,
            "train_eval_series_disjoint": True,
            "same_legacy_asset_family": True,
            "uci_values_context_or_future_read": False,
            "support_b_values_context_or_future_read": False,
            "target_values_context_or_future_read": False,
            "query_values_context_or_future_read": False,
            "target_query_opened": False,
            "grader_only_oracle_is_privileged": True,
        },
        "consumer_fit_count": consumer_fit_count,
        "chronos_judge_call_count": 0,
        "agent_enabled": False,
        "memory_enabled": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "fresh_evidence": False,
        "verdict": verdict,
        "claim_limit": (
            (
                "Diagnostic-only development fault localization on recent_v2 within "
                "two datasets from the same legacy Monash asset family. The exact "
                "oracle is privileged grader instrumentation and is not deployable. "
                "This is not fresh evidence, a Capability, promotion, Memory, Target, "
                "Query, or transfer result."
            )
            if mode == "oracle_only"
            else (
                "Development-only method-facing positive control within two datasets "
                "from the same legacy Monash asset family. The exact oracle is "
                "privileged grader instrumentation. This is not fresh evidence, a "
                "deployable oracle repair, a Capability, promotion, Memory, Target, "
                "Query, or transfer result."
            )
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
        "--geometry",
        choices=tuple(GEOMETRY_BY_ID),
        default=EARLY_V1.geometry_id,
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="full",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    geometry = GEOMETRY_BY_ID[args.geometry]
    if args.mode == "oracle_only" and geometry != RECENT_V2:
        parser.error("--mode oracle_only requires --geometry recent_v2")
    default_output = (
        RECENT_ORACLE_OUTPUT_RELATIVE_PATH
        if args.mode == "oracle_only"
        else OUTPUT_BY_GEOMETRY_ID[geometry.geometry_id]
    )
    output = args.output or project_root / default_output

    report = run_e2_source_coherent_missingness_positive_control(
        registry_path=args.registry,
        split_path=args.split,
        support_a_subsplit_path=args.support_a_subsplit,
        clean_root=args.clean_root,
        geometry=geometry,
        mode=args.mode,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
