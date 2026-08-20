"""M0a traffic extension: the same mask-geometry census, on the traffic batch.

Why this exists.  The batch-recipe tool already produced an adopted plan on
traffic -- ``outlier_iqr`` with training series ``6`` reverted to identity --
and its reverted-series geometry section came out empty, because the frozen
census ``artifacts/functional/e2/m0a_mask_geometry_census_v1.json`` covers
T233, electricity and Weather and has no traffic rows.  This runner fills that
hole so the question "is the geometry of a dropped series observable in
advance" has a third batch to look at.

What it is and is not:

* it is a **descriptive geometry census**.  Every field is computed by the
  frozen ``census_row`` of ``run_e2_m0a_mask_geometry_census`` -- imported, not
  reimplemented -- so a traffic row is field-for-field the same object as a
  frozen-census row;
* it is **not authorization evidence**.  No Skill is written, no Episode is
  formed, no Fast or Slow path is entered, no Gate or Schema is proposed;
* it **does not modify the frozen census**.  It writes its own file stem under
  its own protocol version and never opens the frozen artifact for writing;
* **no threshold is fitted anywhere**, and no adoption rule reads anything in
  here.  The excluded-vs-retained contrast is reported after the fact about a
  decision that was already made by real retrains on the Support window.

Discipline: 0 LLM calls, 0 Support probes, 0 Outcome opened.  The only data
touched is ``traffic.csv`` rows 0..1103 -- the CSV reader is given
``max_rows=1104`` so nothing further is read from disk at all -- which is the
public prefix ``values[uid][:support_origins[0]]`` under the batch recipe's
traffic development origins (1104, 1368 Support; 1800 delayed).  The recipe's
own farthest read is 1848; ``sealed_from_index`` is 3072; neither is
approached.

Run:

    python evaluation/functional/run_e2_m0a_mask_geometry_census_traffic.py

Writes ``artifacts/functional/e2/m0a_mask_geometry_census_traffic_v1.json`` and
``artifacts/functional/e2/m0a_mask_geometry_census_traffic_v1.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

from run_e2_m0a_mask_geometry_census import (  # noqa: E402
    GEOMETRY_FIELDS,
    PROTOCOL_VERSION as FROZEN_PROTOCOL_VERSION,
    REFERENCE_FIELDS,
    _TRAFFIC_DEVELOPMENT_ORIGINS,
    _TRAFFIC_EVAL,
    _TRAFFIC_RECIPE_FARTHEST_READ,
    _TRAFFIC_SEALED_FROM_INDEX,
    _TRAFFIC_TRAIN,
    _cohort_summary,
    _field_stats,
    _mark_representatives,
    _sanity,
    _traffic_work_items,
    census_row,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
PROTOCOL_VERSION = "m0a_mask_geometry_census_traffic_v1"
OUT_JSON = E2 / "m0a_mask_geometry_census_traffic_v1.json"
OUT_MD = E2 / "m0a_mask_geometry_census_traffic_v1.md"

FROZEN_CENSUS = E2 / "m0a_mask_geometry_census_v1.json"
RECIPE_ARTIFACT = E2 / "batch_recipe_traffic_v1.json"
SCREENING_ARTIFACT = E2 / "g3_candidate_screening_v2.json"

# The nine fields the batch-recipe tool quotes in its reverted-series geometry
# section, in its order.  Mirrored here so the traffic table lines up with the
# tables the other two batches already produced; the recipe tool is not
# imported and not modified.
CONTRAST_FIELDS: tuple[str, ...] = (
    "outlier_region_fraction",
    "level_region_fraction",
    "outlier_region_end_fraction",
    "level_region_end_fraction",
    "union_region_fraction",
    "union_region_end_fraction",
    "outlier_point_fraction",
    "local_robust_z_peak",
    "level_excursion_score",
)
ALL_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(GEOMETRY_FIELDS + REFERENCE_FIELDS + CONTRAST_FIELDS)
)


# ------------------------------------------------------------------ provenance
def _recipe_facts() -> dict[str, Any]:
    """Roster, windows and adopted plan, read verbatim from the recipe artifact.

    Read-only.  The traffic wiring in the census module is a hand mirror of the
    recipe tool's ``_TRAFFIC_*`` constants, so it is checked here against the
    artifact the recipe run actually wrote rather than trusted.  A mismatch is
    fatal: a census on a different roster or a different origin would not be
    describing the batch whose plan it is meant to explain.
    """
    recipe = json.loads(RECIPE_ARTIFACT.read_text(encoding="utf-8"))
    train = [str(uid) for uid in recipe["train_series"]]
    evaluation = [str(uid) for uid in recipe["eval_series"]]
    support = [int(origin) for origin in recipe["support_origins"]]
    delayed = [int(origin) for origin in recipe["delayed_origins"]]
    checks = {
        "train_roster_matches": train == list(_TRAFFIC_TRAIN),
        "eval_roster_matches": evaluation == list(_TRAFFIC_EVAL),
        "support_origins_match": support == list(_TRAFFIC_DEVELOPMENT_ORIGINS[:2]),
        "delayed_origins_match": delayed == list(_TRAFFIC_DEVELOPMENT_ORIGINS[2:]),
        "census_cutoff_is_first_support_origin": (
            int(_TRAFFIC_DEVELOPMENT_ORIGINS[0]) == support[0]
        ),
        "recipe_farthest_read_inside_sealed": (
            _TRAFFIC_RECIPE_FARTHEST_READ < _TRAFFIC_SEALED_FROM_INDEX
        ),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise RuntimeError(
            "traffic census wiring disagrees with the frozen recipe artifact "
            "on %s" % failed
        )
    adopted = recipe["adopted_plan"]
    trace = [
        {
            "program": str(row["program"]),
            "excluded_series": [str(uid) for uid in row["excluded_series"]],
            "stability_check": str(row.get("stability_check")),
        }
        for row in recipe.get("adoption_trace", [])
    ]
    return {
        "recipe_artifact": RECIPE_ARTIFACT.relative_to(PROJECT_ROOT).as_posix(),
        "recipe_protocol_version": recipe.get("protocol_version"),
        "task_episode_id": str(recipe["task_episode_id"]),
        "train_series": train,
        "eval_series": evaluation,
        "support_origins": support,
        "delayed_origins": delayed,
        "adopted_program": str(adopted["program"]),
        "adopted_kind": str(adopted["kind"]),
        "excluded_series": [str(uid) for uid in adopted["excluded_series"]],
        "adoption_trace": trace,
        "wiring_checks": checks,
        "wiring_check_all_pass": True,
    }


def _screening_z_peak() -> dict[str, Any]:
    """The screening's per-series ``local_robust_z_peak``, for cross-reference.

    Read-only, and reported on a different window: screening computed it on
    ``values[uid][:3072]`` (the frozen roster's first Support origin, which is
    also traffic's ``sealed_from_index``), while this census computes it on
    ``values[uid][:1104]``.  The two numbers are therefore not comparable point
    for point and no conclusion is drawn from their difference.
    """
    if not SCREENING_ARTIFACT.is_file():
        return {"available": False}
    screening = json.loads(SCREENING_ARTIFACT.read_text(encoding="utf-8"))
    candidate = next(
        (
            row for row in screening.get("candidates", [])
            if str(row.get("candidate")) == "traffic"
        ),
        None,
    )
    if candidate is None:
        return {"available": False}
    per_series = candidate["public_phenomena"]["per_series"]
    return {
        "available": True,
        "artifact": SCREENING_ARTIFACT.relative_to(PROJECT_ROOT).as_posix(),
        "window": "values[uid][:3072] (frozen roster support_origins[0])",
        "census_window": "values[uid][:1104] (recipe development origin)",
        "comparability": (
            "different prefix lengths; the two z-peak columns are reported "
            "side by side and are not differenced or thresholded"
        ),
        "local_robust_z_peak": {
            str(uid): float(row["local_robust_z_peak"])
            for uid, row in per_series.items()
        },
    }


# -------------------------------------------------------------------- contrast
def _group_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mask-class counts, pss counts and per-field spread for one group."""
    classes = Counter(str(row["mask_class"]) for row in rows)
    fields: dict[str, Any] = {}
    for field in CONTRAST_FIELDS:
        values = [float(row[field]) for row in rows]
        if not values:
            fields[field] = {
                "count": 0, "mean": None, "min": None, "median": None,
                "max": None,
            }
            continue
        array = np.asarray(values, dtype=np.float64)
        fields[field] = {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "max": float(np.max(array)),
        }
    return {
        "series": [str(row["series_uid"]) for row in rows],
        "series_count": len(rows),
        "mask_class_counts": dict(classes),
        "union_pss_true_count": sum(1 for row in rows if row["union_pss"]),
        "level_only_pss_true_count": sum(
            1 for row in rows if row["level_only_pss"]
        ),
        "pss_divergent_count": sum(1 for row in rows if row["pss_divergent"]),
        "fields": fields,
    }


def _ranks(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Descending rank of each series on each contrast field.  Rank 1 = highest.

    A rank is a description of where a series sits inside the batch it was
    measured with.  It is not a score and nothing is thresholded on it.  Ties
    are broken by series uid, so a rank alone can understate a series: the
    per-field contrast records exact-equality ties alongside the rank.
    """
    table: dict[str, dict[str, int]] = {}
    for field in CONTRAST_FIELDS:
        ordered = sorted(
            rows, key=lambda row: (-float(row[field]), str(row["series_uid"]))
        )
        table[field] = {
            str(row["series_uid"]): index
            for index, row in enumerate(ordered, start=1)
        }
    return table


def _contrast(
    rows: Sequence[Mapping[str, Any]],
    excluded: Sequence[str],
    label: str,
    basis: str,
) -> dict[str, Any]:
    """One descriptive excluded-vs-retained reading.  No threshold is fitted."""
    excluded = [str(uid) for uid in excluded]
    by_uid = {str(row["series_uid"]): row for row in rows}
    excluded_rows = [by_uid[uid] for uid in excluded if uid in by_uid]
    retained_rows = [
        row for row in rows if str(row["series_uid"]) not in set(excluded)
    ]
    left = _group_block(excluded_rows)
    right = _group_block(retained_rows)
    ranks = _ranks(rows)

    per_field: list[dict[str, Any]] = []
    for field in CONTRAST_FIELDS:
        low, high = left["fields"][field], right["fields"][field]
        if low["mean"] is None or high["mean"] is None:
            continue
        overlaps = not (low["min"] > high["max"] or low["max"] < high["min"])
        per_field.append({
            "field": field,
            "excluded_mean": low["mean"],
            "retained_mean": high["mean"],
            "excluded_range": [low["min"], low["max"]],
            "retained_range": [high["min"], high["max"]],
            "retained_median": high["median"],
            "observed_ranges_overlap": overlaps,
            "direction": (
                "excluded_higher" if low["mean"] > high["mean"]
                else "excluded_lower" if low["mean"] < high["mean"]
                else "equal"
            ),
            "excluded_ranks_descending": {
                uid: ranks[field][uid] for uid in excluded if uid in ranks[field]
            },
            "batch_size_for_rank": len(rows),
            "excluded_values": {
                uid: float(by_uid[uid][field]) for uid in excluded
                if uid in by_uid
            },
            # Exact-equality ties matter for reading a rank: a series can sit
            # at rank 2 while holding the batch maximum.
            "tied_retained_series": {
                uid: sorted(
                    (
                        str(other["series_uid"]) for other in retained_rows
                        if float(other[field]) == float(by_uid[uid][field])
                    ),
                    key=lambda item: (
                        int(item) if item.isdigit() else 1 << 30, item
                    ),
                )
                for uid in excluded if uid in by_uid
            },
        })
    return {
        "label": label,
        "reading": (
            "DESCRIPTIVE ONLY. No threshold is fitted, no Observation is wired, "
            "and nothing in this block feeds any adoption rule, Gate or verdict."
        ),
        "selection_basis": basis,
        "excluded": left,
        "retained": right,
        "fields_with_non_overlapping_observed_ranges": [
            row["field"] for row in per_field
            if not row["observed_ranges_overlap"]
        ],
        "per_field": per_field,
    }


# ------------------------------------------------------- comparability check
def _frozen_reproduction_check() -> dict[str, Any]:
    """Does this working tree still reproduce the frozen census rows exactly?

    The traffic rows are only comparable with the frozen census if the extractor
    still computes the frozen values.  ``runtime.public_features`` carries M0b
    working-tree edits, so this is checked rather than assumed: ``census_row``
    is re-run on the ``e1v2_task_01`` train prefixes of both full-report
    cohorts and every field is compared with the frozen artifact at exact
    equality.  Read-only on both sides; the frozen artifact is never written.
    """
    from evaluation.functional.task_episode_harness.agentic.runner import (
        load_cohort,
    )

    frozen = json.loads(FROZEN_CENSUS.read_text(encoding="utf-8"))
    frozen_rows = {
        (str(row["cohort"]), str(row["task_episode_id"]), str(row["series_uid"])):
        row
        for row in frozen["rows"]
    }
    cutoffs = {
        cohort: int(frozen["cohort_meta"][cohort]["observation_cutoffs"][0])
        for cohort in frozen["full_report_cohorts"]
    }
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for cohort_name, cutoff in sorted(cutoffs.items()):
        cohort = load_cohort(PROJECT_ROOT, cohort_name)
        for uid in [str(item) for item in cohort["train_uids"]]:
            series = np.asarray(cohort["values"][uid], dtype=np.float64)
            row = census_row(
                (cohort_name, "e1v2_task_01", cutoff, uid, series[:cutoff].copy())
            )
            reference = frozen_rows.get((cohort_name, "e1v2_task_01", uid))
            if reference is None:
                mismatches.append({
                    "cohort": cohort_name, "series_uid": uid,
                    "problem": "row absent from the frozen artifact",
                })
                continue
            checked += 1
            for key, value in row.items():
                if key not in reference:
                    mismatches.append({
                        "cohort": cohort_name, "series_uid": uid, "field": key,
                        "problem": "field absent from the frozen artifact",
                    })
                elif value != reference[key]:
                    mismatches.append({
                        "cohort": cohort_name, "series_uid": uid, "field": key,
                        "recomputed": value, "frozen": reference[key],
                    })
    return {
        "claim": (
            "the current working tree reproduces the frozen census rows "
            "field-for-field at exact equality, so a traffic row computed now "
            "is comparable with the frozen T233 / electricity / Weather rows"
        ),
        "scope": (
            "e1v2_task_01 train prefixes of both full-report cohorts, %s"
            % sorted(cutoffs)
        ),
        "checked_rows": checked,
        "mismatch_count": len(mismatches),
        "all_exact": not mismatches,
        "mismatches": mismatches[:20],
        "why_it_is_checked": (
            "runtime/public_features.py carries uncommitted M0b edits; they are "
            "additive mapping keys plus a refactor of the pss formula into a "
            "helper, and this check is what makes that additivity a measured "
            "fact rather than a reading of the diff"
        ),
    }


# ------------------------------------------------------------------- assembly
def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[int, str]:
        uid = str(row["series_uid"])
        return (int(uid) if uid.isdigit() else 1 << 30, uid)

    return sorted(rows, key=key)


def _census(uids: Sequence[str], role: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    items, meta = _traffic_work_items(uids)
    rows = [census_row(item) for item in items]
    representatives = _mark_representatives(rows)
    for row in rows:
        row["roster_role"] = role
    return _sorted_rows(rows), meta, representatives


def build_payload(
    *, include_eval: bool = True, reproduction_check: bool = True
) -> dict[str, Any]:
    facts = _recipe_facts()
    repro = _frozen_reproduction_check() if reproduction_check else {
        "claim": "skipped by --no-reproduction-check", "all_exact": None,
    }

    train_rows, meta, train_representatives = _census(_TRAFFIC_TRAIN, "train")
    eval_rows: list[dict[str, Any]] = []
    eval_representatives: dict[str, Any] = {}
    if include_eval:
        eval_rows, _eval_meta, eval_representatives = _census(
            _TRAFFIC_EVAL, "eval"
        )

    train_sanity = _sanity(train_rows)
    all_sanity = _sanity(train_rows + eval_rows)
    summary = _cohort_summary(train_rows)
    field_stats = {
        field: _field_stats([row[field] for row in train_rows])
        for field in ALL_FIELDS
    }
    degenerate = sorted(
        field for field in ALL_FIELDS if not field_stats[field]["non_degenerate"]
    )

    adopted_excluded = list(facts["excluded_series"])
    adopted_contrast = _contrast(
        train_rows,
        adopted_excluded,
        "adopted plan `%s`, reverted %s"
        % (
            facts["adopted_program"],
            ", ".join(adopted_excluded) or "nothing",
        ),
        "chosen by the batch recipe's greedy Support-window mask search, every "
        "step validated by a real retrain; the geometry below played no part "
        "in that decision and is read off afterwards",
    )
    other_contrasts = [
        _contrast(
            train_rows,
            row["excluded_series"],
            "unadopted masked candidate `%s`, would have reverted %s "
            "(adoption trace: %s)"
            % (
                row["program"],
                ", ".join(row["excluded_series"]) or "nothing",
                row["stability_check"],
            ),
            "the other masked plan the same search produced; `NOT_REACHED` "
            "means the adoption rule stopped at the first candidate that "
            "passed the delayed stability check and never judged this one. It "
            "is reported so the descriptive reading is not conditioned on the "
            "winning mask alone",
        )
        for row in facts["adoption_trace"]
        if row["excluded_series"] and row["excluded_series"] != adopted_excluded
    ]

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "descriptive mask-geometry census of the traffic batch; a third "
            "batch of samples for the question of whether the geometry of a "
            "dropped training series is observable in advance"
        ),
        "not_authorization_evidence": (
            "no Skill is written, no Episode is formed, no Fast or Slow path is "
            "entered, no Gate or Schema is proposed, and no execution right is "
            "granted or implied"
        ),
        "does_not_modify_frozen_census": {
            "frozen_artifact": FROZEN_CENSUS.relative_to(PROJECT_ROOT).as_posix(),
            "frozen_protocol_version": FROZEN_PROTOCOL_VERSION,
            "opened_for_writing": False,
            "sha256_after_this_run": _sha256(FROZEN_CENSUS),
            "note": (
                "the batch-recipe tool reads the frozen census verbatim by "
                "provenance; this run writes a separate stem and leaves it "
                "byte-identical"
            ),
        },
        "row_unit": (
            "one traffic series public prefix values[uid][:1104] at the batch "
            "recipe's first Support origin, under Task Episode %s"
            % facts["task_episode_id"]
        ),
        "row_function": (
            "run_e2_m0a_mask_geometry_census.census_row, imported unmodified, "
            "so every field is computed exactly as in %s"
            % FROZEN_PROTOCOL_VERSION
        ),
        "frozen_census_reproduction_check": repro,
        "recipe_provenance": facts,
        "cohort_meta": meta,
        "screening_cross_reference": _screening_z_peak(),
        "sanity_check": train_sanity,
        "sanity_check_including_eval": all_sanity,
        "summary_train": summary,
        "field_stats_train": field_stats,
        "degenerate_fields_train": degenerate,
        "task_representatives_train": train_representatives,
        "task_representatives_eval": eval_representatives,
        "adopted_plan_contrast": adopted_contrast,
        "other_candidate_contrasts": other_contrasts,
        "rows": train_rows,
        "eval_rows": eval_rows,
        "eval_rows_note": (
            "the eval series are censused with the identical prefix rule and "
            "the identical row function, and are excluded from every summary, "
            "contrast and rank in this artifact; the recipe's mask search only "
            "ever reverts training series"
        ),
        "provenance": {
            "extractor": "SelfEvolvingHarnessTS.runtime.public_features."
            "extract_public_features (unmodified)",
            "expansion": "runtime.public_features._expand (radius=2)",
            "pss_formula": "max(0, (1 - end_fraction) * "
            "_DOWNSTREAM_WINDOW_POINTS) >= _POST_SHIFT_SUPPORT_MIN_POINTS, "
            "constants imported from the extractor module",
            "zero_llm": True,
            "zero_support_probe": True,
            "zero_outcome_opened": True,
            "observable_features_unchanged": True,
            "extract_public_features_unchanged": True,
            "thresholds_unchanged": True,
            "no_threshold_fitted": True,
            "csv_rows_read": int(meta["window_provenance"]["csv_rows_loaded"]),
            "not_read": [
                "KDD W3 T211-T230 (INSTANCE_UNSEEN preserved)",
                "any sealed Outcome (NOAA, g3_final_query_outcome, delayed "
                "truth)",
                "traffic rows at or past index %d, and in fact any traffic row "
                "at or past index %d"
                % (
                    _TRAFFIC_SEALED_FROM_INDEX,
                    int(meta["window_provenance"]["census_cutoff"]),
                ),
            ],
        },
    }
    return payload


# --------------------------------------------------------------------- report
def _number(value: Any) -> str:
    return "n/a" if value is None else "%.6f" % float(value)


def _repro_line(payload: Mapping[str, Any]) -> str:
    repro = payload["frozen_census_reproduction_check"]
    if repro.get("all_exact") is None:
        return (
            "Comparability with the frozen census was **not** re-checked in "
            "this run (%s)." % repro["claim"]
        )
    return (
        "Comparability with the frozen census: `census_row` was re-run on the "
        "%d frozen `e1v2_task_01` train rows of the full-report cohorts and "
        "compared field for field at exact equality -- **%s**, %d mismatches. "
        "This matters because `runtime/public_features.py` carries uncommitted "
        "M0b edits; the check makes their additivity a measured fact rather "
        "than a reading of the diff, and it is what licenses putting a traffic "
        "row next to a frozen one."
        % (
            repro["checked_rows"],
            "PASS" if repro["all_exact"] else "FAIL",
            repro["mismatch_count"],
        )
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    facts = payload["recipe_provenance"]
    meta = payload["cohort_meta"]
    window = meta["window_provenance"]
    summary = payload["summary_train"]
    rows = payload["rows"]
    lines: list[str] = []

    lines += [
        "# M0a mask geometry census -- `traffic` v1",
        "",
        "**Descriptive geometry census. Not authorization evidence. Does not "
        "modify the frozen M0a census.**",
        "",
        "Every field below is produced by `census_row` of "
        "`evaluation/functional/run_e2_m0a_mask_geometry_census.py`, imported "
        "unmodified, so a traffic row is field-for-field the same object as a "
        "row of the frozen `%s` artifact. That artifact is not opened for "
        "writing by this run; the batch-recipe tool keeps reading it verbatim "
        "by provenance (sha256 after this run "
        "`%s`)."
        % (
            payload["does_not_modify_frozen_census"]["frozen_protocol_version"],
            payload["does_not_modify_frozen_census"]["sha256_after_this_run"],
        ),
        "",
        "0 LLM calls. 0 Support probes. 0 Outcome opened. `OBSERVABLE_FEATURES` "
        "unchanged, `extract_public_features` unchanged, **no threshold is "
        "fitted anywhere in this report**. No Skill, Episode, Gate, Schema or "
        "execution right follows from it.",
        "",
        "## 0. Why traffic",
        "",
        "The batch recipe adopted `%s` on traffic with training series %s "
        "reverted to identity, and its reverted-series geometry table came out "
        "empty: the frozen census covers T233, electricity and Weather and has "
        "no traffic rows. This run fills that hole so the question *is the "
        "geometry of a dropped series observable in advance* has a third batch "
        "of samples. The census is computed **after** the fact and had no part "
        "in the decision it describes."
        % (
            facts["adopted_program"],
            ", ".join("`%s`" % uid for uid in facts["excluded_series"]) or "none",
        ),
        "",
        "## 1. Coverage and window",
        "",
        "| item | value |",
        "| --- | --- |",
        "| Task Episode | `%s` |" % facts["task_episode_id"],
        "| train series censused | %d (`%s`) |"
        % (len(rows), "`, `".join(str(row["series_uid"]) for row in rows)),
        "| eval series censused | %d (reported separately, in no summary) |"
        % len(payload["eval_rows"]),
        "| prefix rule | `values[uid][:support_origins[0]]` |",
        "| recipe Support origins | %s |" % window["recipe_support_origins"],
        "| recipe delayed origins | %s |" % window["recipe_delayed_origins"],
        "| census cutoff | %d (farthest index read %d) |"
        % (window["census_cutoff"], window["census_farthest_index_read"]),
        "| recipe farthest index read | %d |"
        % window["recipe_farthest_index_read"],
        "| `sealed_from_index` | %d |" % window["sealed_from_index"],
        "| CSV rows loaded from disk | %d |" % window["csv_rows_loaded"],
        "",
        "The CSV reader is called with `max_rows=%d`, so the window is "
        "structural: no traffic row past index %d is loaded at all, which is "
        "well inside both the recipe's own farthest read (%d) and the sealed "
        "boundary (%d)."
        % (
            window["csv_rows_loaded"],
            window["census_farthest_index_read"],
            window["recipe_farthest_index_read"],
            window["sealed_from_index"],
        ),
        "",
        "Exposure: %s" % meta["exposure"],
        "",
        "The roster and both origin sets are checked against the frozen recipe "
        "artifact `%s` before anything is computed; all %d checks pass."
        % (
            facts["recipe_artifact"],
            len(facts["wiring_checks"]),
        ),
        "",
    ]

    sanity = payload["sanity_check"]
    all_sanity = payload["sanity_check_including_eval"]
    lines += [
        "## 2. Sanity",
        "",
        "The same four checks the frozen census runs -- `union_pss` reproduces "
        "the public `post_shift_support_sufficient`, the union mask "
        "reconstructs `region_mask`, the union end fraction matches "
        "`estimated_region_end_fraction`, every field is finite -- over the %d "
        "train rows: **%s**. Including the %d eval rows: **%s**."
        % (
            sanity["checked_rows"],
            "PASS" if sanity["all_pass"] else "FAIL",
            len(payload["eval_rows"]),
            "PASS" if all_sanity["all_pass"] else "FAIL",
        ),
        "",
        "%s" % _repro_line(payload),
        "",
        "## 3. The 12 train series",
        "",
        "| series | mask_class | outlier_region_frac | level_region_frac | "
        "outlier_end | level_end | union_frac | union_end | outlier_point_frac "
        "| z_peak | level_excursion | union_pss | level_only_pss | divergent |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | --- | --- | --- |",
    ]
    excluded_set = set(facts["excluded_series"])
    for row in rows:
        uid = str(row["series_uid"])
        mark = " **(reverted)**" if uid in excluded_set else ""
        lines.append(
            "| `%s`%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | "
            "%s | %s |"
            % (
                uid, mark, row["mask_class"],
                _number(row["outlier_region_fraction"]),
                _number(row["level_region_fraction"]),
                _number(row["outlier_region_end_fraction"]),
                _number(row["level_region_end_fraction"]),
                _number(row["union_region_fraction"]),
                _number(row["union_region_end_fraction"]),
                _number(row["outlier_point_fraction"]),
                _number(row["local_robust_z_peak"]),
                _number(row["level_excursion_score"]),
                row["union_pss"], row["level_only_pss"], row["pss_divergent"],
            )
        )
    lines.append("")

    stats = payload["field_stats_train"]
    lines += [
        "## 4. Field non-degeneracy (12 train rows)",
        "",
        "Degeneracy is read exactly as the frozen census reads it: a field is "
        "non-degenerate when it is finite everywhere and neither all-zero nor "
        "all-one.",
        "",
        "| field | min | max | mean | distinct | all_zero | all_one | "
        "non_degenerate |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for field in ALL_FIELDS:
        row = stats[field]
        lines.append(
            "| `%s` | %s | %s | %s | %d | %s | %s | %s |"
            % (
                field, _number(row["min"]), _number(row["max"]),
                _number(row["mean"]), row["distinct_count"],
                row["all_zero"], row["all_one"], row["non_degenerate"],
            )
        )
    degenerate = payload["degenerate_fields_train"]
    lines += [
        "",
        "Degenerate on this batch: %s."
        % (", ".join("`%s`" % field for field in degenerate) or "none"),
        "",
        "## 5. `mask_class` distribution",
        "",
        "| class | count | fraction |",
        "| --- | ---: | ---: |",
    ]
    for name in ("MIXED", "OUTLIER_ONLY", "LEVEL_ONLY", "AMBIGUOUS"):
        count = summary["mask_class_counts"].get(name, 0)
        lines.append(
            "| `%s` | %d | %.4f |"
            % (name, count, count / max(1, summary["decision_point_count"]))
        )
    pss = summary["pss"]
    lines += [
        "",
        "`MIXED` = expanded outlier region and `level_mask` both non-empty; "
        "`AMBIGUOUS` = both empty.",
        "",
        "## 6. `union_pss` vs `level_only_pss`",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        "| decision points | %d |" % summary["decision_point_count"],
        "| `union_pss` true | %d |" % pss["union_true_count"],
        "| `level_only_pss` true | %d |" % pss["level_only_true_count"],
        "| divergent | %d |" % pss["divergent_count"],
        "| divergent fraction | %.4f |" % pss["divergent_fraction"],
        "| divergence sources | %s |"
        % json.dumps(pss["divergence_source_counts"], sort_keys=True),
        "",
        "`OUTLIER` / `MISSING` / `BOTH` name the region whose expanded tail "
        "attains the union's last True index, i.e. the region that pushed the "
        "union end fraction up and flipped pss away from the level-only "
        "reading.",
        "",
    ]

    def contrast_section(contrast: Mapping[str, Any], heading: str) -> None:
        left, right = contrast["excluded"], contrast["retained"]
        lines.append(heading)
        lines.append("")
        lines.append("**%s**" % contrast["reading"])
        lines.append("")
        lines.append(
            "Group split: %s (%s) vs %d retained (%s). Selection basis: %s."
            % (
                ", ".join("`%s`" % uid for uid in left["series"]) or "none",
                json.dumps(left["mask_class_counts"], sort_keys=True),
                right["series_count"],
                json.dumps(right["mask_class_counts"], sort_keys=True),
                contrast["selection_basis"],
            )
        )
        lines.append("")
        if len(left["series"]) > 1:
            lines.append(
                "The excluded group has more than one member, so its per-series "
                "values are listed before the group summary:"
            )
            lines.append("")
            lines.append(
                "| field | " + " | ".join("`%s`" % uid for uid in left["series"])
                + " |"
            )
            lines.append(
                "| --- | " + " | ".join("---:" for _ in left["series"]) + " |"
            )
            for field_row in contrast["per_field"]:
                values = field_row["excluded_values"]
                lines.append(
                    "| `%s` | " % field_row["field"]
                    + " | ".join(
                        _number(values.get(uid)) for uid in left["series"]
                    )
                    + " |"
                )
            lines.append("")
        lines.append(
            "| field | excluded mean | retained min | retained median | "
            "retained max | ranges overlap | direction | excluded rank (of %d) |"
            % len(rows)
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
        for field_row in contrast["per_field"]:
            ranks = ", ".join(
                "%s:%d" % (uid, rank)
                for uid, rank in sorted(
                    field_row["excluded_ranks_descending"].items(),
                    key=lambda item: item[1],
                )
            )
            lines.append(
                "| `%s` | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    field_row["field"],
                    _number(field_row["excluded_mean"]),
                    _number(field_row["retained_range"][0]),
                    _number(field_row["retained_median"]),
                    _number(field_row["retained_range"][1]),
                    field_row["observed_ranges_overlap"],
                    field_row["direction"],
                    ranks or "n/a",
                )
            )
        lines.append("")
        lines.append(
            "Rank is descending within the whole %d-series batch (1 = highest "
            "value); ties are broken by series uid, so a rank can understate a "
            "series that holds a shared maximum."
            % len(rows)
        )
        lines.append("")
        ties: list[str] = []
        for field_row in contrast["per_field"]:
            for uid, tied in field_row["tied_retained_series"].items():
                if tied:
                    ties.append(
                        "`%s` ties `%s` exactly on `%s` (%s)"
                        % (
                            uid,
                            "`, `".join(tied),
                            field_row["field"],
                            _number(field_row["excluded_values"][uid]),
                        )
                    )
        if ties:
            lines.append(
                "Exact-equality ties with retained series: %s." % "; ".join(ties)
            )
            lines.append("")
        separated = contrast["fields_with_non_overlapping_observed_ranges"]
        lines.append(
            "Fields whose observed ranges do not overlap between the two "
            "groups: %s."
            % (", ".join("`%s`" % field for field in separated) or "none")
        )
        lines.append("")

    contrast_section(
        payload["adopted_plan_contrast"],
        "## 7. Descriptive contrast: the reverted series vs the retained ones",
    )
    for index, contrast in enumerate(payload["other_candidate_contrasts"], start=8):
        contrast_section(
            contrast,
            "## %d. Descriptive contrast: %s" % (index, contrast["label"]),
        )

    section = 8 + len(payload["other_candidate_contrasts"])
    screening = payload["screening_cross_reference"]
    if screening.get("available"):
        lines += [
            "## %d. Screening `local_robust_z_peak`, side by side" % section,
            "",
            "The screening artifact `%s` computed `local_robust_z_peak` on `%s`; "
            "this census computes it on `%s`. **Different windows: the two "
            "columns are printed side by side and are never differenced, "
            "ranked together or thresholded.**"
            % (
                screening["artifact"], screening["window"],
                screening["census_window"],
            ),
            "",
            "| series | screening z_peak (prefix 3072) | census z_peak (prefix "
            "1104) |",
            "| --- | ---: | ---: |",
        ]
        for row in rows:
            uid = str(row["series_uid"])
            mark = " **(reverted)**" if uid in excluded_set else ""
            lines.append(
                "| `%s`%s | %s | %s |"
                % (
                    uid, mark,
                    _number(screening["local_robust_z_peak"].get(uid)),
                    _number(row["local_robust_z_peak"]),
                )
            )
        lines.append("")
        section += 1

    lines += [
        "## %d. What this does not say" % section,
        "",
        "- It does not say that any field predicts exclusion. One batch, one "
        "reverted series, and a rank is not a threshold.",
        "- It does not fit, propose or imply a threshold, cut point or rule on "
        "any field, and nothing here is wired into a Gate, Schema, Observation "
        "or adoption rule.",
        "- It does not revisit the adopted plan. That plan was chosen by real "
        "retrains on the Support window and is unchanged by this artifact.",
        "- It does not touch the frozen census. The batch-recipe tool still "
        "reads `%s` verbatim; this run wrote `%s` instead."
        % (
            payload["does_not_modify_frozen_census"]["frozen_artifact"],
            OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
        ),
        "- It is not authorization evidence: %s."
        % payload["not_authorization_evidence"],
        "",
        "## Provenance",
        "",
        "- row function: `%s`" % payload["row_function"],
        "- extractor: `%s`" % payload["provenance"]["extractor"],
        "- expansion: `%s`" % payload["provenance"]["expansion"],
        "- pss formula: `%s`" % payload["provenance"]["pss_formula"],
        "- recipe artifact read for roster, windows and plan: `%s` (`%s`)"
        % (facts["recipe_artifact"], facts["recipe_protocol_version"]),
        "- CSV rows read from disk: %d" % payload["provenance"]["csv_rows_read"],
        "- not read: %s"
        % "; ".join(payload["provenance"]["not_read"]),
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- entry
def run(*, include_eval: bool = True, reproduction_check: bool = True) -> int:
    payload = build_payload(
        include_eval=include_eval, reproduction_check=reproduction_check
    )
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)

    summary = payload["summary_train"]
    print("sanity_all_pass", payload["sanity_check"]["all_pass"], flush=True)
    print(
        "frozen_reproduction_all_exact",
        payload["frozen_census_reproduction_check"]["all_exact"],
        flush=True,
    )
    print(
        "mask_class", json.dumps(summary["mask_class_counts"], sort_keys=True),
        flush=True,
    )
    print(
        "pss_divergent %d/%d = %.4f %s"
        % (
            summary["pss"]["divergent_count"],
            summary["decision_point_count"],
            summary["pss"]["divergent_fraction"],
            json.dumps(
                summary["pss"]["divergence_source_counts"], sort_keys=True
            ),
        ),
        flush=True,
    )
    print(
        "degenerate_fields",
        json.dumps(payload["degenerate_fields_train"]),
        flush=True,
    )
    print(
        "non_overlapping_fields_adopted",
        json.dumps(
            payload["adopted_plan_contrast"][
                "fields_with_non_overlapping_observed_ranges"
            ]
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="census the 12 train series only (eval rows are descriptive extra)",
    )
    parser.add_argument(
        "--no-reproduction-check",
        action="store_true",
        help="skip the exact-equality re-check against the frozen census rows",
    )
    args = parser.parse_args(argv)
    return run(
        include_eval=not args.no_eval,
        reproduction_check=not args.no_reproduction_check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
