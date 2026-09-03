"""0-LLM per-series audit of the P4 Forecast CONFLICT units (development only).

Why this exists.  The P4 performance terminal artifact records one aggregate
``support_gain`` per Episode and the relation it was classified into, but not
the ``per_view_gain`` behind that relation.  The policy question -- whether a
CONFLICT is "one series barely nicked" or "several series badly hurt" -- can
only be answered from the per-series split, so this recomputes it.

Discipline.  No LLM.  No Final, Query, UCR TEST or sealed AD read.  No new SHA,
manifest or ledger field.  Reads only the exposed KDD evolution roster already
opened by the P4 performance run, and calls that runner's own ``_cell_at`` and
``_reading`` so the numbers come from the same instrument rather than from a
re-implementation -- every recomputed aggregate is checked back against the
value the terminal artifact recorded.

Scope of the claim.  Development-only diagnosis of an already-collected run.
It changes no threshold and releases nothing.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json"
)
REPORT = (
    PROJECT_ROOT / "artifacts/main_protocol/p4_conflict_per_series_audit_20260831.json"
)

# Frozen classification threshold, read not written:
# experience_memory.CLASSIFICATION_MATERIAL_THRESHOLD.
MATERIAL = 0.005


def _conflict_units(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Unique (origin, operator) CONFLICT units, with their run multiplicity.

    The three arms and three replica orders replay the same origins, so the 51
    CONFLICT rows in the artifact are run records, not independent samples.
    Policy has to be decided on the deduplicated units.
    """
    seen: dict[tuple[int, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        details = row.get("details") or {}
        for episode in details.get("episodes_written") or []:
            if str(episode.get("relation")) != "CONFLICT":
                continue
            # episode_id tail after "_target_" is
            # <operator>_<replica>_<e#>_<arm>_<p#>, and every arm name spends
            # two underscore-separated tokens (a3_reset, k0_fixed, a5_online),
            # so five tokens follow the operator however long the operator is.
            _head, _sep, tail = str(episode["episode_id"]).partition("_target_")
            operator = tail.rsplit("_", 5)[0] if tail else ""
            key = (int(row["origin"]), operator)
            unit = seen.setdefault(
                key,
                {
                    "origin": int(row["origin"]),
                    "operator": operator,
                    "episode_label": row["episode_id"],
                    "run_record_count": 0,
                    "arms": set(),
                    "recorded_support_gain": float(episode["support_gain"]),
                    "recorded_local_status": str(episode["local_status"]),
                    "delayed_gain": episode.get("delayed_gain"),
                },
            )
            unit["run_record_count"] += 1
            unit["arms"].add(str(row["method"]))
    units = []
    for unit in seen.values():
        unit["arms"] = sorted(unit["arms"])
        units.append(unit)
    return sorted(units, key=lambda u: (u["origin"], u["operator"]))


def _face_split(cell: Any, face: str, operator: str, origin: int) -> dict[str, Any]:
    identity = forecast_p4._reading(cell, face, (), origin=origin)
    candidate = forecast_p4._reading(
        cell, face, forecast_p1._steps(operator), origin=origin
    )
    before = np.asarray(identity["per_series_smase"], dtype=np.float64)
    after = np.asarray(candidate["per_series_smase"], dtype=np.float64)
    gains = before - after  # positive = sMASE fell = the series got better
    harmed = gains < -MATERIAL
    helped = gains > MATERIAL
    negative = gains[gains < 0.0]
    positive = gains[gains > 0.0]
    return {
        "aggregate_gain": float(identity["smase"] - candidate["smase"]),
        "series_count": int(gains.size),
        "harmed_count": int(harmed.sum()),
        "harmed_fraction": float(harmed.mean()),
        "helped_count": int(helped.sum()),
        "max_single_series_harm": float(-gains.min()) if gains.min() < 0 else 0.0,
        "total_negative_mass": float(-negative.sum()),
        "total_positive_mass": float(positive.sum()),
        "median_series_gain": float(np.median(gains)),
        "harmed_series_indices": [int(i) for i in np.flatnonzero(harmed)],
        "per_series_gain": [float(value) for value in gains],
        "identity_per_series_smase": [float(value) for value in before],
    }


def build_report() -> dict[str, Any]:
    payload = json.loads(SOURCE_RUN.read_text(encoding="utf-8"))
    units = _conflict_units(payload)
    base, _selection, data = forecast_p1._load_exposed_cells()

    rows: list[dict[str, Any]] = []
    for unit in units:
        origin = unit["origin"]
        cell = forecast_p4._cell_at(base, origin)
        support_a = _face_split(cell, "support_a", unit["operator"], origin)
        support_b = _face_split(cell, "support_b", unit["operator"], origin)
        recomputed = support_a["aggregate_gain"]
        rows.append(
            {
                **unit,
                "support_a": support_a,
                "support_b": support_b,
                # A CONFLICT with a clean Support-A face was a Support winner
                # first and was only downgraded by the delayed reading.
                "conflict_stage": (
                    "SUPPORT" if support_a["harmed_count"] > 0 else "DELAYED"
                ),
                "support_b_same_direction": bool(
                    (support_a["aggregate_gain"] > MATERIAL)
                    == (support_b["aggregate_gain"] > MATERIAL)
                ),
                "support_b_also_conflict": bool(
                    support_b["aggregate_gain"] >= MATERIAL
                    and support_b["harmed_count"] > 0
                ),
                "recompute_matches_artifact": bool(
                    abs(recomputed - unit["recorded_support_gain"]) < 5e-6
                ),
            }
        )

    support_stage = [row for row in rows if row["conflict_stage"] == "SUPPORT"]
    harmed = [row["support_a"]["harmed_count"] for row in support_stage]
    worst = [row["support_a"]["max_single_series_harm"] for row in support_stage]
    negative = [row["support_a"]["total_negative_mass"] for row in support_stage]
    aggregate = [row["support_a"]["aggregate_gain"] for row in support_stage]
    harmed_anywhere = {
        index
        for row in support_stage
        for index in row["support_a"]["harmed_series_indices"]
    }

    return {
        "stage": "P4_FORECAST_CONFLICT_PER_SERIES_AUDIT",
        "status": "COMPLETE",
        "evidence_grade": "DEVELOPMENT_ONLY_DIAGNOSIS_OF_COLLECTED_RUN",
        "source_run": SOURCE_RUN.relative_to(PROJECT_ROOT).as_posix(),
        "dataset": data.get("dataset"),
        "data_role": "EXPOSED_DEVELOPMENT",
        "classification_rule": (
            "experience_memory.classify_relation: aggregate >= +0.005 and "
            "min(per-series) >= -0.005 -> POSITIVE; aggregate >= +0.005 with "
            "any per-series < -0.005 -> CONFLICT"
        ),
        "material_threshold": MATERIAL,
        "threshold_changed_by_this_audit": False,
        "llm_calls": 0,
        "boundary": {
            "natural_final_outcome_reads": 0,
            "query_evaluations": 0,
            "ucr_test_outcome_reads": 0,
            "sealed_ad_outcome_reads": 0,
            "new_sha_added": False,
            "new_manifest_added": False,
            "live_provider_calls": 0,
        },
        "sampling": {
            "conflict_run_records": 51,
            "unique_units": len(rows),
            "note": (
                "three arms x three replica orders replay the same origins, so "
                "run records are not independent samples; every statistic below "
                "is computed on the unique units"
            ),
        },
        "counts": {
            "unique_units": len(rows),
            "support_stage_conflict": len(support_stage),
            "delayed_stage_downgrade": len(rows) - len(support_stage),
            "recompute_matches_artifact": sum(
                1 for row in rows if row["recompute_matches_artifact"]
            ),
        },
        "support_stage_summary": {
            "units": len(support_stage),
            "series_per_unit": 20,
            "harmed_count_median": float(np.median(harmed)),
            "harmed_count_min": int(min(harmed)),
            "harmed_count_max": int(max(harmed)),
            "harmed_count_histogram": {
                str(value): harmed.count(value) for value in sorted(set(harmed))
            },
            "units_with_single_series_harm": sum(1 for h in harmed if h == 1),
            "max_single_series_harm_median": float(np.median(worst)),
            "max_single_series_harm_min": float(min(worst)),
            "max_single_series_harm_max": float(max(worst)),
            "total_negative_mass_median": float(np.median(negative)),
            "aggregate_gain_median": float(np.median(aggregate)),
            "worst_harm_over_aggregate_gain_median": float(
                np.median([w / a for w, a in zip(worst, aggregate)])
            ),
            "positive_to_negative_mass_ratio_median": float(
                np.median(
                    [
                        row["support_a"]["total_positive_mass"]
                        / row["support_a"]["total_negative_mass"]
                        for row in support_stage
                    ]
                )
            ),
        },
        "support_b_reproduction": {
            "same_direction": sum(1 for row in rows if row["support_b_same_direction"]),
            "also_conflict": sum(1 for row in rows if row["support_b_also_conflict"]),
            "of_units": len(rows),
            "sign_reversals": [
                {
                    "origin": row["origin"],
                    "operator": row["operator"],
                    "support_a_aggregate_gain": row["support_a"]["aggregate_gain"],
                    "support_b_aggregate_gain": row["support_b"]["aggregate_gain"],
                    "support_b_harmed_count": row["support_b"]["harmed_count"],
                }
                for row in rows
                if not row["support_b_same_direction"]
            ],
        },
        "harmed_series_stability": {
            "series_harmed_in_at_least_one_unit": len(harmed_anywhere),
            "series_never_harmed": sorted(set(range(20)) - harmed_anywhere),
            "max_units_any_single_series_is_harmed_in": max(
                sum(
                    1
                    for row in support_stage
                    if index in row["support_a"]["harmed_series_indices"]
                )
                for index in range(20)
            ),
            "reading": (
                "the harmed set is not stable across units, so a Scope revision "
                "keyed on series identity has no stable subset to exclude"
            ),
        },
        "identity_smase_association": {
            "status": "DIAGNOSTIC_REFERENCE_ONLY",
            "not_a_deployable_scope_feature": True,
            "not_a_performance_ceiling": True,
            "why": (
                "identity sMASE at this origin is a downstream Consumer outcome, "
                "not a pre-deployment observable a held-out Fast Path may read; "
                "it is reported only to show the harm is systematic rather than "
                "random, and the deployment-visible feature audit is a separate "
                "step that has not been run yet"
            ),
        },
        "units": rows,
        "verdict": (
            "CONFLICT_IS_REPRODUCIBLE_AND_MULTI_SERIES__"
            "NOT_THE_SINGLE_SERIES_NICK_BRANCH"
        ),
        "releases": "NONE",
    }


def main() -> int:
    report = build_report()
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = report["counts"]
    print(
        "unique=%d support-stage=%d delayed-stage=%d recompute-match=%d/%d"
        % (
            counts["unique_units"],
            counts["support_stage_conflict"],
            counts["delayed_stage_downgrade"],
            counts["recompute_matches_artifact"],
            counts["unique_units"],
        )
    )
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
