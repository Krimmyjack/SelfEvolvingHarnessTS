"""M0a: zero-LLM outlier/level mask geometry census on public prefixes.

Frozen diagnostic question, not a Capability and not a Claim:

1. does splitting the public ``region_mask`` union back into an expanded
   outlier region and the un-expanded ``level_mask`` carry information that the
   union alone does not, and
2. does the union actually pollute the decision semantics of the public
   ``post_shift_support_sufficient`` (pss) field?

Discipline of this script:

* 0 LLM calls, 0 Support probes, 0 Outcome opened.  It only slices
  ``values[uid][:support_origins[0]]`` -- the same public prefix / origin cut
  the existing Task Episode censuses use -- and calls the frozen
  ``extract_public_features``.
* ``OBSERVABLE_FEATURES`` and ``extract_public_features`` are not modified.
  The two pss constants are imported from the extractor module, never copied.
* KDD W3 T211-T230 is never loaded; not even its Context is read.
* Weather reports field distribution and coverage only.  No Utility or gain
  symbol of any kind is read or computed for Weather (its aggregate utility is
  METRIC_UNREADABLE).
* No threshold is fitted anywhere.  The task_01 vs task_13..19 reading is
  descriptive (direction and observed-range overlap) only.

Pre-stated verdict rule (fixed before the numbers were seen):

    INFORMATIVE iff the four split-geometry fields are non-degenerate on the
    full-report cohorts (finite, not all-zero, not all-one) AND at least one of
      (i)  the ``e1v2_task_01`` representative value falls outside the observed
           min..max range of the ``e1v2_task_13..19`` representative values on
           at least one split-geometry field, or
      (ii) ``union_pss != level_only_pss`` on a non-zero fraction of the
           full-report decision points;
    otherwise OUTLIER_LEVEL_MASK_GEOMETRY_CANDIDATE_NOT_INFORMATIVE, which
    refutes this one candidate Observation and nothing else.

Run:

    python evaluation/functional/run_e2_m0a_mask_geometry_census.py

Writes ``artifacts/functional/e2/m0a_mask_geometry_census_v1.json`` and
``artifacts/functional/e2/m0a_mask_geometry_census_v1.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from SelfEvolvingHarnessTS.runtime.public_features import (  # noqa: E402
    _DOWNSTREAM_WINDOW_POINTS,
    _POST_SHIFT_SUPPORT_MIN_POINTS,
    _expand,
    extract_public_features,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
PROTOCOL_VERSION = "m0a_mask_geometry_census_v1"
TASK_KIND = "forecast"

# Rosters are the already-exposed ones: T233 e1v2_task_01..19 (g1.A5A3_MAX_N),
# electricity e1v2_task_01..09 (artifacts/functional/e2/r3_full_ab_electricity),
# Weather e1v2_task_01..19 (g1.WEATHER_MAX_N).
COHORT_TASK_COUNT = {"T233": 19, "electricity": 9, "weather": 19, "traffic": 1}
FULL_REPORT_COHORTS = ("T233", "electricity")
FIELD_ONLY_COHORTS = ("weather",)
COHORT_ORDER = FULL_REPORT_COHORTS + FIELD_ONLY_COHORTS

# ------------------------------------------------------------------- traffic
#
# traffic is structurally accepted by screening v2 but is not one of the three
# cohorts of the frozen v1 artifact and is not reachable through
# ``agentic.runner.load_cohort``.  The wiring below mirrors the traffic block
# of ``evaluation/functional/run_batch_composition_headroom.py`` -- its
# ``_TRAFFIC_*`` constants and the traffic branch of its local ``load_cohort``
# -- so the census reads exactly the window the batch recipe ran on.  That file
# is neither modified nor imported here; the mirror is checked field by field
# against the frozen recipe artifact by
# ``run_e2_m0a_mask_geometry_census_traffic.py``.
#
# A traffic run never touches the frozen v1 artifact: ``run()`` below refuses
# any cohort selection other than the frozen default, and the traffic census is
# driven by that separate runner, which writes its own file stem under its own
# protocol version.
EXTRA_COHORTS = ("traffic",)
_TRAFFIC_TRAIN: tuple[str, ...] = tuple(str(index) for index in range(12))
_TRAFFIC_EVAL: tuple[str, ...] = tuple(str(index) for index in range(12, 20))
_TRAFFIC_DEVELOPMENT_ORIGINS: tuple[int, ...] = (1104, 1368, 1800)
_TRAFFIC_SEALED_FROM_INDEX = 3072
# The recipe's own farthest read: its last delayed origin plus the frozen Task
# horizon.  The census reads strictly less than this -- the prefix rule cuts at
# the first Support origin -- and both are far inside the sealed boundary.
_TRAFFIC_RECIPE_FARTHEST_READ = 1848
_TRAFFIC_EXPOSURE = (
    "STRUCTURALLY_ACCEPTED_BUT_SOURCE_FAMILY_EXPOSURE_UNRESOLVED: "
    "PeMS SF Bay Area / monash:traffic_hourly family has unresolved prior "
    "exposure; this census reads the public prefix values[uid][:1104] only "
    "and does not open a sealed Outcome"
)

# Labels are taken verbatim from the frozen M0a instruction, not re-derived
# from any Outcome artifact.
HAMPEL_NEGATIVE_TASK = "e1v2_task_01"
OUTLIER_POSITIVE_TASKS = tuple("e1v2_task_%02d" % index for index in range(13, 20))

GEOMETRY_FIELDS = (
    "outlier_region_fraction",
    "level_region_fraction",
    "outlier_region_end_fraction",
    "level_region_end_fraction",
)
REFERENCE_FIELDS = (
    "outlier_point_fraction",
    "missing_region_end_fraction",
    "union_region_end_fraction",
    "union_region_fraction",
    "missing_fraction",
)


def _pss(end_fraction: float) -> bool:
    """The frozen public pss formula, evaluated on an arbitrary end fraction."""
    return bool(
        max(0.0, (1.0 - float(end_fraction)) * _DOWNSTREAM_WINDOW_POINTS)
        >= _POST_SHIFT_SUPPORT_MIN_POINTS
    )


def _end_fraction(mask: np.ndarray) -> float:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return 0.0
    return float((int(indices[-1]) + 1) / int(mask.size))


def _point_mask(size: int, indices: Sequence[int]) -> np.ndarray:
    mask = np.zeros(int(size), dtype=bool)
    if indices:
        mask[np.asarray(tuple(int(index) for index in indices), dtype=int)] = True
    return mask


def census_row(item: tuple[str, str, int, str, Any]) -> dict[str, Any]:
    """One decision point: one train series prefix at one Task's first origin."""
    cohort, task_id, cutoff, uid, prefix = item
    values = np.asarray(prefix, dtype=np.float64)
    size = int(values.size)
    extraction = extract_public_features(values, task_kind=TASK_KIND)
    mapping = dict(extraction.mapping)

    outlier_points = _point_mask(size, extraction.outlier_indices)
    outlier_region = _expand(outlier_points)
    missing_region = _expand(_point_mask(size, extraction.missing_indices))
    level_mask = np.asarray(extraction.level_mask, dtype=bool)
    union = missing_region | outlier_region | level_mask

    union_end = _end_fraction(union)
    outlier_end = _end_fraction(outlier_region)
    level_end = _end_fraction(level_mask)
    missing_end = _end_fraction(missing_region)

    union_pss = _pss(union_end)
    level_only_pss = _pss(level_end)
    mapping_pss = bool(mapping["post_shift_support_sufficient"])

    outlier_present = bool(outlier_region.any())
    level_present = bool(level_mask.any())
    if outlier_present and level_present:
        mask_class = "MIXED"
    elif outlier_present:
        mask_class = "OUTLIER_ONLY"
    elif level_present:
        mask_class = "LEVEL_ONLY"
    else:
        mask_class = "AMBIGUOUS"

    divergent = union_pss != level_only_pss
    contributors: list[str] = []
    if divergent:
        for name, end in (
            ("outlier", outlier_end),
            ("missing", missing_end),
            ("level", level_end),
        ):
            if end == union_end:
                contributors.append(name)
    if not divergent:
        source = "NONE"
    elif contributors == ["outlier"]:
        source = "OUTLIER"
    elif contributors == ["missing"]:
        source = "MISSING"
    elif set(contributors) == {"outlier", "missing"}:
        source = "BOTH"
    else:
        source = "OTHER:" + "+".join(contributors)

    row = {
        "cohort": cohort,
        "task_episode_id": task_id,
        "observation_cutoff": int(cutoff),
        "series_uid": uid,
        "prefix_points": size,
        "outlier_region_fraction": float(np.mean(outlier_region)),
        "level_region_fraction": float(np.mean(level_mask)),
        "outlier_region_end_fraction": outlier_end,
        "level_region_end_fraction": level_end,
        "missing_region_end_fraction": missing_end,
        "union_region_end_fraction": union_end,
        "union_region_fraction": float(np.mean(union)),
        "outlier_point_fraction": float(np.mean(outlier_points)),
        "missing_fraction": float(mapping["missing_fraction"]),
        "local_robust_z_peak": float(mapping["local_robust_z_peak"]),
        "level_excursion_score": float(mapping["level_excursion_score"]),
        "mask_class": mask_class,
        "union_pss": union_pss,
        "level_only_pss": level_only_pss,
        "mapping_post_shift_support_sufficient": mapping_pss,
        "pss_divergent": divergent,
        "pss_divergence_source": source,
        "sanity_union_pss_matches_mapping": union_pss == mapping_pss,
        "sanity_union_mask_reconstructed": bool(
            np.array_equal(union, np.asarray(extraction.region_mask, dtype=bool))
        ),
        "sanity_union_end_matches_mapping": (
            union_end == float(mapping["estimated_region_end_fraction"])
        ),
        "sanity_all_fields_finite": all(
            np.isfinite(value)
            for value in (
                float(np.mean(outlier_region)),
                float(np.mean(level_mask)),
                outlier_end,
                level_end,
                missing_end,
                union_end,
            )
        ),
    }
    return row


# ------------------------------------------------------------------ loading


def _traffic_csv_path() -> Path:
    """The same two candidate paths the batch-recipe tool uses, in that order."""
    candidates = (
        Path(r"C:/Users/辉/desktop/agent/shared_tsq_datasets")
        / "traffic/traffic.csv",
        Path(
            "/mnt/c/Users/辉/desktop/agent/shared_tsq_datasets/"
            "traffic/traffic.csv"
        ),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("traffic.csv not found under shared_tsq_datasets")


def _traffic_values(uids: Sequence[str], cutoff: int) -> dict[str, np.ndarray]:
    """Traffic columns, read from disk only as far as ``cutoff`` rows.

    ``max_rows=cutoff`` makes the window a structural guarantee rather than a
    comment: the CSV reader stops at the public prefix boundary, so no point at
    or past ``_TRAFFIC_SEALED_FROM_INDEX`` -- and none past the batch recipe's
    own farthest read either -- is ever loaded, let alone used.
    """
    from evaluation.functional.task_episode_harness.agentic import g3_sourcing

    cutoff = int(cutoff)
    if cutoff >= _TRAFFIC_SEALED_FROM_INDEX:
        raise RuntimeError(
            "traffic census cutoff %d reaches sealed_from_index=%d"
            % (cutoff, _TRAFFIC_SEALED_FROM_INDEX)
        )
    if cutoff > _TRAFFIC_RECIPE_FARTHEST_READ:
        raise RuntimeError(
            "traffic census cutoff %d reads past the batch recipe's farthest "
            "read %d" % (cutoff, _TRAFFIC_RECIPE_FARTHEST_READ)
        )
    _names, columns = g3_sourcing.load_csv_columns(
        _traffic_csv_path(), max_rows=cutoff
    )
    missing = [str(uid) for uid in uids if str(uid) not in columns]
    if missing:
        raise RuntimeError(
            "traffic screening roster missing from CSV columns: %s" % missing
        )
    return {
        str(uid): np.asarray(columns[str(uid)], dtype=np.float64)[:cutoff]
        for uid in uids
    }
def _traffic_work_items(
    uids: Sequence[str] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Decision points for traffic: one Task Episode origin, one prefix each.

    Only ``e1v2_task_01`` is wired for traffic, because that is the only Task
    the batch recipe ran on this cohort.  Its Support origins are the frozen
    screening development origins rather than the roster spec's, so the prefix
    rule ``values[uid][:support_origins[0]]`` cuts at 1104 and not at 3072.
    """
    from evaluation.functional.task_episode_harness.e1 import _frozen_task_roster

    spec = _frozen_task_roster()[0]
    task_id = str(spec["task_episode_id"])
    cutoff = int(_TRAFFIC_DEVELOPMENT_ORIGINS[0])
    selected = [str(uid) for uid in (_TRAFFIC_TRAIN if uids is None else uids)]
    series = _traffic_values(selected, cutoff)
    items: list[Any] = [
        ("traffic", task_id, cutoff, uid, series[uid].copy()) for uid in selected
    ]
    meta = {
        "cohort": "traffic",
        "exposure": _TRAFFIC_EXPOSURE,
        "train_uids": list(_TRAFFIC_TRAIN),
        "eval_uids": list(_TRAFFIC_EVAL),
        "censused_uids": selected,
        "task_episode_ids": [task_id],
        "observation_cutoffs": [cutoff],
        "prefix_rule": (
            "values[uid][:support_origins[0]] (public prefix only), with "
            "support_origins taken from the batch recipe's traffic development "
            "origins %s rather than from the frozen roster spec %s"
            % (
                list(_TRAFFIC_DEVELOPMENT_ORIGINS[:2]),
                [int(origin) for origin in spec["support_origins"]],
            )
        ),
        "window_provenance": {
            "development_origins": list(_TRAFFIC_DEVELOPMENT_ORIGINS),
            "recipe_support_origins": list(_TRAFFIC_DEVELOPMENT_ORIGINS[:2]),
            "recipe_delayed_origins": list(_TRAFFIC_DEVELOPMENT_ORIGINS[2:]),
            "census_cutoff": cutoff,
            "census_farthest_index_read": cutoff - 1,
            "recipe_farthest_index_read": _TRAFFIC_RECIPE_FARTHEST_READ,
            "sealed_from_index": _TRAFFIC_SEALED_FROM_INDEX,
            "csv_rows_loaded": cutoff,
        },
    }
    return items, meta


def _work_items(cohort: str, task_count: int) -> tuple[list[Any], dict[str, Any]]:
    if cohort == "traffic":
        if int(task_count) != 1:
            raise ValueError(
                "traffic is wired for e1v2_task_01 only "
                "(task_count=%d requested)" % int(task_count)
            )
        return _traffic_work_items()
    from evaluation.functional.task_episode_harness.agentic.runner import load_cohort
    from evaluation.functional.task_episode_harness.e1 import _frozen_task_roster

    cohort_data = load_cohort(PROJECT_ROOT, cohort)
    train_uids = [str(uid) for uid in cohort_data["train_uids"]]
    specs = list(_frozen_task_roster()[:task_count])
    items: list[Any] = []
    for spec in specs:
        task_id = str(spec["task_episode_id"])
        cutoff = int(spec["support_origins"][0])
        for uid in train_uids:
            series = np.asarray(cohort_data["values"][uid], dtype=np.float64)
            items.append((cohort, task_id, cutoff, uid, series[:cutoff].copy()))
    meta = {
        "cohort": cohort,
        "exposure": cohort_data["exposure"],
        "train_uids": train_uids,
        "task_episode_ids": [str(spec["task_episode_id"]) for spec in specs],
        "observation_cutoffs": [int(spec["support_origins"][0]) for spec in specs],
        "prefix_rule": "values[uid][:support_origins[0]] (public prefix only)",
    }
    return items, meta


def _mark_representatives(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Apply the frozen public_context scope / representative rule to the rows.

    The rule and its constants come from
    ``task_episode_harness.public_context``; only the already-computed
    ``local_robust_z_peak`` values are re-used so no series is extracted twice.
    """
    from evaluation.functional.task_episode_harness.public_context import (
        PUBLIC_CONTEXT_SCOPE_BIN,
        PUBLIC_CONTEXT_SCOPE_FEATURE,
    )
    from SelfEvolvingHarnessTS.contracts.observables import observable_numeric_bin

    by_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        row["scope_bin"] = observable_numeric_bin(
            PUBLIC_CONTEXT_SCOPE_FEATURE, float(row[PUBLIC_CONTEXT_SCOPE_FEATURE])
        )
        row["in_scope"] = row["scope_bin"] == PUBLIC_CONTEXT_SCOPE_BIN
        row["is_representative"] = False
        by_task.setdefault((row["cohort"], row["task_episode_id"]), []).append(row)

    representatives: dict[str, dict[str, Any]] = {}
    for (cohort, task_id), task_rows in by_task.items():
        scoped = {row["series_uid"]: row for row in task_rows if row["in_scope"]}
        if not scoped:
            representatives["%s|%s" % (cohort, task_id)] = {
                "representative_uid": None,
                "scope_size": 0,
            }
            continue
        uid = max(
            sorted(scoped),
            key=lambda key: float(scoped[key][PUBLIC_CONTEXT_SCOPE_FEATURE]),
        )
        scoped[uid]["is_representative"] = True
        representatives["%s|%s" % (cohort, task_id)] = {
            "representative_uid": uid,
            "scope_size": len(scoped),
        }
    return representatives


# ------------------------------------------------------------------ summary


def _field_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "finite_count": int(np.sum(np.isfinite(array))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "distinct_count": int(np.unique(array).size),
        "all_zero": bool(np.all(array == 0.0)),
        "all_one": bool(np.all(array == 1.0)),
        "non_degenerate": bool(
            np.all(np.isfinite(array))
            and not np.all(array == 0.0)
            and not np.all(array == 1.0)
        ),
    }


def _cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(row["mask_class"] for row in rows)
    total = len(rows)
    divergent = [row for row in rows if row["pss_divergent"]]
    return {
        "decision_point_count": total,
        "task_count": len({row["task_episode_id"] for row in rows}),
        "series_count": len({row["series_uid"] for row in rows}),
        "field_stats": {
            field: _field_stats([row[field] for row in rows])
            for field in GEOMETRY_FIELDS + REFERENCE_FIELDS
        },
        "mask_class_counts": dict(classes),
        "mask_class_fractions": {
            name: (count / total if total else 0.0) for name, count in classes.items()
        },
        "mixed_fraction": (classes.get("MIXED", 0) / total) if total else 0.0,
        "ambiguous_fraction": (classes.get("AMBIGUOUS", 0) / total) if total else 0.0,
        "pss": {
            "union_true_count": sum(1 for row in rows if row["union_pss"]),
            "level_only_true_count": sum(1 for row in rows if row["level_only_pss"]),
            "divergent_count": len(divergent),
            "divergent_fraction": (len(divergent) / total) if total else 0.0,
            "divergence_source_counts": dict(
                Counter(row["pss_divergence_source"] for row in divergent)
            ),
        },
        "representative_coverage": {
            "tasks_with_non_empty_scope": len(
                {row["task_episode_id"] for row in rows if row["in_scope"]}
            ),
            "in_scope_decision_points": sum(1 for row in rows if row["in_scope"]),
        },
    }


def _t233_contrast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive task_01 vs task_13..19 reading.  No threshold is fitted."""
    representative = {
        row["task_episode_id"]: row
        for row in rows
        if row["cohort"] == "T233" and row["is_representative"]
    }
    train_scope: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["cohort"] == "T233":
            train_scope.setdefault(row["task_episode_id"], []).append(row)

    def task_block(task_id: str) -> dict[str, Any]:
        rep = representative.get(task_id)
        group = train_scope.get(task_id, [])
        block: dict[str, Any] = {
            "task_episode_id": task_id,
            "representative_uid": rep["series_uid"] if rep else None,
            "mask_class": rep["mask_class"] if rep else None,
            "union_pss": rep["union_pss"] if rep else None,
            "level_only_pss": rep["level_only_pss"] if rep else None,
            "pss_divergent": rep["pss_divergent"] if rep else None,
            "pss_divergence_source": rep["pss_divergence_source"] if rep else None,
        }
        for field in GEOMETRY_FIELDS + ("union_region_end_fraction",):
            block[field] = rep[field] if rep else None
            block[field + "__train_scope_mean"] = (
                float(np.mean([item[field] for item in group])) if group else None
            )
        return block

    negative = task_block(HAMPEL_NEGATIVE_TASK)
    positives = [task_block(task_id) for task_id in OUTLIER_POSITIVE_TASKS]

    def overlap_on(suffix: str) -> dict[str, Any]:
        table: dict[str, Any] = {}
        for field in GEOMETRY_FIELDS:
            key = field + suffix
            positive_values = [
                block[key] for block in positives if block[key] is not None
            ]
            negative_value = negative[key]
            if not positive_values or negative_value is None:
                table[field] = {"readable": False}
                continue
            low, high = min(positive_values), max(positive_values)
            table[field] = {
                "readable": True,
                "negative_task_value": negative_value,
                "positive_min": low,
                "positive_max": high,
                "negative_inside_positive_range": bool(low <= negative_value <= high),
                "direction": (
                    "negative_below_positive_range"
                    if negative_value < low
                    else "negative_above_positive_range"
                    if negative_value > high
                    else "overlapping"
                ),
            }
        return table

    overlap = overlap_on("")
    scope_overlap = overlap_on("__train_scope_mean")
    separating = sorted(
        field
        for field, block in overlap.items()
        if block.get("readable") and not block["negative_inside_positive_range"]
    )
    scope_separating = sorted(
        field
        for field, block in scope_overlap.items()
        if block.get("readable") and not block["negative_inside_positive_range"]
    )
    negative_representative = negative["representative_uid"]
    positive_representatives = sorted(
        {
            block["representative_uid"]
            for block in positives
            if block["representative_uid"] is not None
        }
    )
    return {
        "labels_provenance": (
            "task_01 = hampel NEGATIVE, task_13..19 = outlier POSITIVE, taken "
            "verbatim from the frozen M0a instruction; no Outcome artifact was "
            "read by this script"
        ),
        "reading": "descriptive only: direction and observed-range overlap",
        "negative_task": negative,
        "positive_tasks": positives,
        "observed_range_overlap": overlap,
        "separating_fields": separating,
        "observed_range_overlap_train_scope_mean": scope_overlap,
        "separating_fields_train_scope_mean": scope_separating,
        "representative_series_confound": {
            "negative_task_representative_uid": negative_representative,
            "positive_task_representative_uids": positive_representatives,
            "representative_differs": bool(
                negative_representative not in positive_representatives
            ),
            "note": (
                "the frozen public_context representative rule selects a "
                "different train series for the negative Task than for the "
                "positive Tasks, so the representative-level contrast mixes a "
                "Task-origin difference with a series-identity difference; the "
                "train-scope-mean columns are reported as the cross-check"
            ),
        },
        "direction_stable_separating_fields": sorted(
            field
            for field in separating
            if field in scope_separating
            and overlap[field]["direction"] == scope_overlap[field]["direction"]
        ),
    }


def _verdict(
    full_summary: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> dict[str, Any]:
    non_degenerate = {
        field: bool(full_summary["field_stats"][field]["non_degenerate"])
        for field in GEOMETRY_FIELDS
    }
    all_non_degenerate = all(non_degenerate.values())
    separating = list(contrast["separating_fields"])
    divergent_fraction = float(full_summary["pss"]["divergent_fraction"])
    criterion_i = bool(separating)
    criterion_ii = divergent_fraction > 0.0
    informative = all_non_degenerate and (criterion_i or criterion_ii)
    return {
        "non_degenerate_by_field": non_degenerate,
        "all_geometry_fields_non_degenerate": all_non_degenerate,
        "criterion_i_range_separation": {
            "satisfied": criterion_i,
            "separating_fields": separating,
        },
        "criterion_ii_pss_divergence": {
            "satisfied": criterion_ii,
            "divergent_fraction": divergent_fraction,
        },
        "verdict": (
            "INFORMATIVE"
            if informative
            else "OUTLIER_LEVEL_MASK_GEOMETRY_CANDIDATE_NOT_INFORMATIVE"
        ),
        "scope_of_verdict": (
            "this verdict concerns only the outlier/level mask-geometry "
            "candidate Observation; it is not a Capability family termination"
        ),
    }


# ------------------------------------------------------------------ reports


def _sanity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "sanity_union_pss_matches_mapping",
        "sanity_union_mask_reconstructed",
        "sanity_union_end_matches_mapping",
        "sanity_all_fields_finite",
    )
    failures = [
        {
            "cohort": row["cohort"],
            "task_episode_id": row["task_episode_id"],
            "series_uid": row["series_uid"],
            **{key: row[key] for key in keys},
            "union_pss": row["union_pss"],
            "mapping_post_shift_support_sufficient": row[
                "mapping_post_shift_support_sufficient"
            ],
            "union_region_end_fraction": row["union_region_end_fraction"],
        }
        for row in rows
        if not all(row[key] for key in keys)
    ]
    return {
        "checked_rows": len(rows),
        "checks": list(keys),
        "all_pass": not failures,
        "failure_count": len(failures),
        "failures": failures[:20],
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    header = payload["provenance"]
    lines = [
        "# M0a outlier / level mask geometry census v1",
        "",
        "0 LLM calls. 0 Support probes. 0 Outcome opened. "
        "`OBSERVABLE_FEATURES` unchanged. `extract_public_features` unchanged. "
        "KDD W3 T211-T230 not read (INSTANCE_UNSEEN preserved).",
        "",
        "Frozen offline diagnostic only. Not a Capability, not a Claim, not a "
        "Promotion. No threshold is fitted anywhere in this report.",
        "",
        "- decision point = one train-series public prefix "
        "`values[uid][:support_origins[0]]` at one Task Episode origin",
        "- cohorts with the full report: %s" % ", ".join(FULL_REPORT_COHORTS),
        "- cohorts with field distribution and coverage only: %s "
        "(no Utility / gain symbol is read or computed; its aggregate utility "
        "is METRIC_UNREADABLE)" % ", ".join(FIELD_ONLY_COHORTS),
        "- pss constants imported from `runtime.public_features` "
        "(`_DOWNSTREAM_WINDOW_POINTS`, `_POST_SHIFT_SUPPORT_MIN_POINTS`), "
        "never copied as literals",
        "",
        "## 0. Coverage",
        "",
        "| cohort | tasks | train series | decision points | cutoffs |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for cohort in COHORT_ORDER:
        summary = payload["per_cohort"].get(cohort)
        meta = payload["cohort_meta"].get(cohort)
        if not summary or not meta:
            continue
        cutoffs = meta["observation_cutoffs"]
        lines.append(
            "| %s | %d | %d | %d | %d..%d |"
            % (
                cohort,
                summary["task_count"],
                summary["series_count"],
                summary["decision_point_count"],
                min(cutoffs),
                max(cutoffs),
            )
        )
    sanity = payload["sanity_check"]
    lines.extend([
        "",
        "Sanity check (`union_pss` == public "
        "`post_shift_support_sufficient`, union mask reconstruction, union end "
        "fraction, field finiteness) over all %d rows: **%s**."
        % (sanity["checked_rows"], "PASS" if sanity["all_pass"] else "FAIL"),
        "",
        "## a) Field non-degeneracy",
        "",
        "Full-report cohorts pooled (%s)." % " + ".join(FULL_REPORT_COHORTS),
        "",
        "| field | min | max | mean | distinct | all_zero | all_one | "
        "non_degenerate |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ])
    pooled = payload["full_report_pooled"]
    for field in GEOMETRY_FIELDS + REFERENCE_FIELDS:
        stats = pooled["field_stats"][field]
        lines.append(
            "| `%s` | %.6f | %.6f | %.6f | %d | %s | %s | %s |"
            % (
                field,
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["distinct_count"],
                stats["all_zero"],
                stats["all_one"],
                stats["non_degenerate"],
            )
        )
    lines.extend([
        "",
        "All %d pooled rows have finite values on every reported field: %s."
        % (
            pooled["decision_point_count"],
            all(
                pooled["field_stats"][field]["finite_count"]
                == pooled["field_stats"][field]["count"]
                for field in GEOMETRY_FIELDS + REFERENCE_FIELDS
            ),
        ),
        "",
        "Per-cohort field stats, including the field-distribution-only cohort, "
        "are in the JSON under `per_cohort.<cohort>.field_stats`.",
        "",
        "## b) T233 `task_01` (hampel NEGATIVE) vs `task_13..19` "
        "(outlier POSITIVE)",
        "",
        "Labels are taken verbatim from the frozen M0a instruction. "
        "Values are the representative series of each Task under the frozen "
        "`public_context` scope rule. Descriptive reading only.",
        "",
        "| task | label | representative | mask_class | "
        "outlier_region_frac | level_region_frac | outlier_end | level_end | "
        "union_end | union_pss | level_only_pss |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    contrast = payload["t233_contrast"]

    def contrast_line(block: Mapping[str, Any], label: str) -> str:
        def number(value: Any) -> str:
            return "n/a" if value is None else "%.6f" % float(value)

        return "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            block["task_episode_id"],
            label,
            block["representative_uid"] or "n/a",
            block["mask_class"] or "n/a",
            number(block["outlier_region_fraction"]),
            number(block["level_region_fraction"]),
            number(block["outlier_region_end_fraction"]),
            number(block["level_region_end_fraction"]),
            number(block["union_region_end_fraction"]),
            block["union_pss"],
            block["level_only_pss"],
        )

    lines.append(contrast_line(contrast["negative_task"], "hampel NEGATIVE"))
    for block in contrast["positive_tasks"]:
        lines.append(contrast_line(block, "outlier POSITIVE"))
    lines.extend([
        "",
        "Observed-range overlap (no threshold fitted):",
        "",
        "| field | task_01 | task_13..19 min | task_13..19 max | "
        "task_01 inside range | direction |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ])
    for field in GEOMETRY_FIELDS:
        block = contrast["observed_range_overlap"][field]
        if not block.get("readable"):
            lines.append("| `%s` | n/a | n/a | n/a | n/a | UNREADABLE |" % field)
            continue
        lines.append(
            "| `%s` | %.6f | %.6f | %.6f | %s | %s |"
            % (
                field,
                block["negative_task_value"],
                block["positive_min"],
                block["positive_max"],
                block["negative_inside_positive_range"],
                block["direction"],
            )
        )
    lines.extend([
        "",
        "Fields on which `task_01` falls outside the positive group's observed "
        "range: %s."
        % (
            ", ".join("`%s`" % field for field in contrast["separating_fields"])
            or "none"
        ),
        "",
        "Caveat, and it is load-bearing for how this table may be read: %s"
        % contrast["representative_series_confound"]["note"]
        + " (negative Task representative `%s`, positive Task representatives %s)."
        % (
            contrast["representative_series_confound"][
                "negative_task_representative_uid"
            ],
            ", ".join(
                "`%s`" % uid
                for uid in contrast["representative_series_confound"][
                    "positive_task_representative_uids"
                ]
            )
            or "none",
        ),
        "",
        "Same overlap reading on the train-scope mean over all train series of "
        "each Task, which removes the representative-series identity change:",
        "",
        "| field | task_01 mean | task_13..19 min | task_13..19 max | "
        "task_01 inside range | direction |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ])
    for field in GEOMETRY_FIELDS:
        block = contrast["observed_range_overlap_train_scope_mean"][field]
        if not block.get("readable"):
            lines.append("| `%s` | n/a | n/a | n/a | n/a | UNREADABLE |" % field)
            continue
        lines.append(
            "| `%s` | %.6f | %.6f | %.6f | %s | %s |"
            % (
                field,
                block["negative_task_value"],
                block["positive_min"],
                block["positive_max"],
                block["negative_inside_positive_range"],
                block["direction"],
            )
        )
    lines.extend([
        "",
        "Fields that separate in the **same direction** under both the "
        "representative reading and the train-scope-mean reading: %s. Fields "
        "that separate only under the representative reading, or that flip "
        "direction between the two readings, are not a stable descriptive "
        "separation and are reported as such."
        % (
            ", ".join(
                "`%s`" % field
                for field in contrast["direction_stable_separating_fields"]
            )
            or "none"
        ),
        "",
        "## c) mixed / ambiguous shares",
        "",
        "| cohort | MIXED | OUTLIER_ONLY | LEVEL_ONLY | AMBIGUOUS | "
        "mixed_frac | ambiguous_frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for cohort in COHORT_ORDER:
        summary = payload["per_cohort"].get(cohort)
        if not summary:
            continue
        counts = summary["mask_class_counts"]
        lines.append(
            "| %s | %d | %d | %d | %d | %.4f | %.4f |"
            % (
                cohort,
                counts.get("MIXED", 0),
                counts.get("OUTLIER_ONLY", 0),
                counts.get("LEVEL_ONLY", 0),
                counts.get("AMBIGUOUS", 0),
                summary["mixed_fraction"],
                summary["ambiguous_fraction"],
            )
        )
    lines.extend([
        "",
        "`MIXED` = expanded outlier region and `level_mask` are both non-empty, "
        "so the union folds two mechanisms into one region. `AMBIGUOUS` = both "
        "are empty.",
        "",
        "## d) `union_pss` != `level_only_pss`",
        "",
        "| cohort | decision points | divergent | fraction | source breakdown |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for cohort in COHORT_ORDER:
        summary = payload["per_cohort"].get(cohort)
        if not summary:
            continue
        pss = summary["pss"]
        lines.append(
            "| %s | %d | %d | %.4f | %s |"
            % (
                cohort,
                summary["decision_point_count"],
                pss["divergent_count"],
                pss["divergent_fraction"],
                json.dumps(pss["divergence_source_counts"], sort_keys=True),
            )
        )
    pooled_pss = pooled["pss"]
    lines.extend([
        "",
        "Full-report cohorts pooled: %d / %d = %.4f divergent, sources %s."
        % (
            pooled_pss["divergent_count"],
            pooled["decision_point_count"],
            pooled_pss["divergent_fraction"],
            json.dumps(pooled_pss["divergence_source_counts"], sort_keys=True),
        ),
        "",
        "`OUTLIER` / `MISSING` / `BOTH` name the region whose expanded tail "
        "attains the union's last True index, i.e. the region that pushed the "
        "union end fraction up and flipped pss away from the level-only "
        "reading. When `missing_fraction > 0` the frozen extractor forces "
        "`level_mask` to all-zero, so a `MISSING` divergence is a structural "
        "consequence of that branch, not an independent measurement.",
        "",
        "## e) Verdict",
        "",
    ])
    verdict = payload["verdict"]
    lines.extend([
        "**%s**" % verdict["verdict"],
        "",
        "Pre-stated rule: non-degenerate split-geometry fields AND "
        "(range separation on >=1 field OR non-zero pss divergence).",
        "",
        "- all four split-geometry fields non-degenerate: %s"
        % verdict["all_geometry_fields_non_degenerate"],
        "- criterion (i) range separation: %s (%s)"
        % (
            verdict["criterion_i_range_separation"]["satisfied"],
            ", ".join(
                "`%s`" % field
                for field in verdict["criterion_i_range_separation"][
                    "separating_fields"
                ]
            )
            or "no separating field",
        ),
        "- criterion (ii) pss divergence fraction: %s (%.4f)"
        % (
            verdict["criterion_ii_pss_divergence"]["satisfied"],
            verdict["criterion_ii_pss_divergence"]["divergent_fraction"],
        ),
        "",
        verdict["scope_of_verdict"] + ".",
        "",
    ])
    if verdict["verdict"] == "INFORMATIVE":
        lines.extend([
            "### M0b minimal field-set suggestion (not implemented here)",
            "",
            payload["m0b_suggestion"],
            "",
        ])
    else:
        lines.extend([
            "This refutes the outlier/level mask-geometry candidate "
            "Observation only. It does not close the Capability family and it "
            "does not authorize any new Gate, Schema or infrastructure.",
            "",
        ])
    lines.extend([
        "## Provenance",
        "",
        "- extractor: `%s`" % header["extractor"],
        "- expansion: `%s`" % header["expansion"],
        "- pss formula: `%s`" % header["pss_formula"],
        "- rosters: %s" % json.dumps(header["rosters"], sort_keys=True),
        "- roster reconstruction: %s" % header["roster_reconstruction"],
        "- not read: %s" % ", ".join(header["not_read"]),
        "",
    ])
    return "\n".join(lines) + "\n"


def _m0b_suggestion(
    pooled: Mapping[str, Any],
    contrast: Mapping[str, Any],
    verdict: Mapping[str, Any],
) -> str:
    separating = list(contrast["direction_stable_separating_fields"])
    unstable = [
        field
        for field in verdict["criterion_i_range_separation"]["separating_fields"]
        if field not in separating
    ]
    coverage_fields = [
        field
        for field in separating
        if field in ("outlier_region_fraction", "level_region_fraction")
    ]
    tail_fields = [
        field
        for field in separating
        if field
        in ("outlier_region_end_fraction", "level_region_end_fraction")
    ]
    divergent_fraction = float(pooled["pss"]["divergent_fraction"])
    sources = pooled["pss"]["divergence_source_counts"]
    parts: list[str] = []
    if coverage_fields and tail_fields:
        parts.append(
            "The descriptive separation shows up in both coverage "
            "(%s) and tail position (%s), so the union folds two distinguishable "
            "quantities."
            % (
                ", ".join("`%s`" % field for field in coverage_fields),
                ", ".join("`%s`" % field for field in tail_fields),
            )
        )
    elif coverage_fields:
        parts.append(
            "The descriptive separation is a coverage difference (%s); the tail "
            "positions of the two masks overlap on this cohort."
            % ", ".join("`%s`" % field for field in coverage_fields)
        )
    elif tail_fields:
        parts.append(
            "The descriptive separation is a tail-position difference (%s); the "
            "coverage fractions of the two masks overlap on this cohort."
            % ", ".join("`%s`" % field for field in tail_fields)
        )
    else:
        parts.append(
            "No field separates task_01 from the positive group by observed "
            "range in a direction-stable way; the informative part is the pss "
            "decision itself, not the per-field geometry."
        )
    if unstable:
        parts.append(
            "%s separate only on the representative series and lose the "
            "separation or flip direction on the train-scope mean, so they are "
            "not carried into the suggestion."
            % ", ".join("`%s`" % field for field in unstable)
        )
    if divergent_fraction > 0.0:
        parts.append(
            "`post_shift_support_sufficient` disagrees with its level-only "
            "reading on %.2f%% of the full-report decision points (sources %s), "
            "so the field as currently wired reports a support margin that the "
            "level mechanism does not own."
            % (100.0 * divergent_fraction, json.dumps(sources, sort_keys=True))
        )
    else:
        parts.append(
            "`post_shift_support_sufficient` never disagrees with its "
            "level-only reading on these cohorts, so union pollution is not "
            "demonstrated as a pss decision defect here."
        )
    minimal: list[str] = []
    if tail_fields or divergent_fraction > 0.0:
        minimal.append("`level_region_end_fraction`")
    if coverage_fields or "level_region_fraction" in separating:
        minimal.append("`level_region_fraction`")
    if not minimal:
        minimal.append("`level_region_end_fraction`")
    if any(
        key in sources for key in ("OUTLIER", "BOTH")
    ) or "outlier_region_end_fraction" in separating:
        minimal.append("`outlier_region_end_fraction`")
    parts.append(
        "Minimal M0b field set to consider wiring, in this order: %s. "
        "A level-only pss reading is the single decision-relevant derived "
        "field; the outlier tail field is only needed to explain why the union "
        "reading differs. Nothing beyond these is justified by M0a, and M0a "
        "does not implement any of them."
        % ", ".join(dict.fromkeys(minimal))
    )
    return " ".join(parts)


# ------------------------------------------------------------------ entry


def run(
    *,
    cohorts: Sequence[str],
    max_tasks: int | None,
    workers: int,
) -> int:
    # ``run`` writes the frozen v1 artifact, which the batch-recipe tool reads
    # verbatim by provenance.  Any narrower or wider selection would silently
    # rewrite it with a different cohort set, so it is refused before a single
    # series is loaded.  Cohorts outside the frozen three -- traffic -- have
    # their own runner and their own file stem.
    if tuple(cohorts) != tuple(COHORT_ORDER) or max_tasks is not None:
        raise SystemExit(
            "run() only reproduces the frozen %s artifact, and only for the "
            "frozen cohort set %r with no --max-tasks; requested %r "
            "max_tasks=%r. The non-frozen cohorts %r have their own runners: "
            "for traffic run evaluation/functional/"
            "run_e2_m0a_mask_geometry_census_traffic.py, which writes its own "
            "artifact and never touches this one."
            % (
                PROTOCOL_VERSION, list(COHORT_ORDER), list(cohorts), max_tasks,
                list(EXTRA_COHORTS),
            )
        )
    items: list[Any] = []
    cohort_meta: dict[str, Any] = {}
    for cohort in cohorts:
        task_count = COHORT_TASK_COUNT[cohort]
        if max_tasks is not None:
            task_count = min(task_count, max_tasks)
        cohort_items, meta = _work_items(cohort, task_count)
        items.extend(cohort_items)
        cohort_meta[cohort] = meta
        print(
            "LOADED %s tasks=%d series=%d decision_points=%d"
            % (cohort, task_count, len(meta["train_uids"]), len(cohort_items)),
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for index, item in enumerate(items, start=1):
            rows.append(census_row(item))
            print("ROW %d/%d %s %s %s" % (index, len(items), item[0], item[1], item[3]),
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, row in enumerate(pool.map(census_row, items), start=1):
                rows.append(row)
                if index % 25 == 0 or index == len(items):
                    print("ROW %d/%d" % (index, len(items)), flush=True)

    representatives = _mark_representatives(rows)
    order = {cohort: index for index, cohort in enumerate(COHORT_ORDER)}
    rows.sort(
        key=lambda row: (
            order.get(row["cohort"], 99),
            row["task_episode_id"],
            row["series_uid"],
        )
    )

    sanity = _sanity(rows)
    per_cohort = {
        cohort: _cohort_summary([row for row in rows if row["cohort"] == cohort])
        for cohort in cohorts
    }
    full_rows = [row for row in rows if row["cohort"] in FULL_REPORT_COHORTS]
    pooled = _cohort_summary(full_rows) if full_rows else {}
    provenance = {
        "extractor": "SelfEvolvingHarnessTS.runtime.public_features."
        "extract_public_features (unmodified)",
        "expansion": "runtime.public_features._expand (radius=2, the same "
        "helper the union uses)",
        "pss_formula": "max(0, (1 - end_fraction) * _DOWNSTREAM_WINDOW_POINTS) "
        ">= _POST_SHIFT_SUPPORT_MIN_POINTS, constants imported",
        "rosters": {
            cohort: cohort_meta[cohort]["task_episode_ids"] for cohort in cohorts
        },
        "roster_reconstruction": (
            "rosters come from the existing agentic.runner.load_cohort. For "
            "electricity and Weather that helper applies the repo's frozen "
            "outcome-blind substrate preflights to pick the same exposed "
            "columns; those preflights are the pre-Outcome Judge-readability "
            "guards, not an Outcome read. The census itself only reads "
            "values[uid][:support_origins[0]]."
        ),
        "not_read": [
            "KDD W3 T211-T230 (INSTANCE_UNSEEN preserved, no Context read)",
            "any sealed Outcome (NOAA, g3_final_query_outcome, delayed truth)",
            "any Weather Utility or gain symbol (METRIC_UNREADABLE)",
        ],
        "zero_llm": True,
        "zero_support_probe": True,
        "zero_outcome_opened": True,
        "observable_features_unchanged": True,
        "extract_public_features_unchanged": True,
        "thresholds_unchanged": True,
    }

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "provenance": provenance,
        "row_unit": "cohort x task_episode_id x train series uid at "
        "observation_cutoff = support_origins[0]",
        "cohort_meta": cohort_meta,
        "full_report_cohorts": list(FULL_REPORT_COHORTS),
        "field_distribution_only_cohorts": list(FIELD_ONLY_COHORTS),
        "sanity_check": sanity,
        "per_cohort": per_cohort,
        "full_report_pooled": pooled,
        "task_representatives": representatives,
        "rows": rows,
    }

    if not sanity["all_pass"]:
        payload["verdict"] = {
            "verdict": "M0A_SANITY_CHECK_FAILED",
            "reason": "union_pss did not reproduce the public "
            "post_shift_support_sufficient on every decision point; the census "
            "stopped before any separability or pss reading was made",
        }
        E2.mkdir(parents=True, exist_ok=True)
        (E2 / "m0a_mask_geometry_census_v1.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        (E2 / "m0a_mask_geometry_census_v1.md").write_text(
            "# M0a outlier / level mask geometry census v1\n\n"
            "**STOPPED: M0A_SANITY_CHECK_FAILED.**\n\n"
            "0 LLM, 0 Support, 0 Outcome opened, `OBSERVABLE_FEATURES` "
            "unchanged, KDD W3 not read.\n\n"
            "`union_pss` did not reproduce the public "
            "`post_shift_support_sufficient` on %d of %d decision points. "
            "No separability or pss reading was made. First failures are in "
            "the JSON under `sanity_check.failures`.\n"
            % (sanity["failure_count"], sanity["checked_rows"]),
            encoding="utf-8",
        )
        print("SANITY_CHECK_FAILED failures=%d" % sanity["failure_count"], flush=True)
        return 2

    contrast = _t233_contrast(rows)
    payload["t233_contrast"] = contrast
    verdict = _verdict(pooled, contrast)
    payload["verdict"] = verdict
    payload["m0b_suggestion"] = _m0b_suggestion(pooled, contrast, verdict)

    E2.mkdir(parents=True, exist_ok=True)
    json_path = E2 / "m0a_mask_geometry_census_v1.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    md_path = E2 / "m0a_mask_geometry_census_v1.md"
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print("wrote", json_path, flush=True)
    print("wrote", md_path, flush=True)
    print("sanity_all_pass", sanity["all_pass"], flush=True)
    print("verdict", verdict["verdict"], flush=True)
    print(
        "pooled_divergent_fraction %.4f sources %s"
        % (
            pooled["pss"]["divergent_fraction"],
            json.dumps(pooled["pss"]["divergence_source_counts"], sort_keys=True),
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=list(COHORT_ORDER),
        choices=list(COHORT_ORDER),
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    return run(
        cohorts=list(args.cohorts),
        max_tasks=args.max_tasks,
        workers=int(args.workers),
    )


if __name__ == "__main__":
    raise SystemExit(main())
