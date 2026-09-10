"""R4E: is the pattern readable once the effect is split into its two channels?

R4D-A split every ``(program, position, face, uid)`` effect into ``route =
L_pr - L_rr`` (the shared model changed because *somebody's* training corpus was
prepared) and ``ctx = L_pp - L_pr`` (this sequence's own serving context was
prepared), with ``total = route + ctx`` being the library's ``-g``.  R4A only
ever tested identifiability against ``total`` and read WEAK everywhere.  astra's
point is that "where the effect comes from" and "which information predicts who
benefits" are two different questions: one shared model change can still be
modulated, series by series, by the state of each series.  So this package
re-runs the R4A identifiability reading once per channel.

Nothing here fits anything.  The per-series losses come from
``_scratch/r4d_a_three_cell_store.json`` (produced by the authorised R4D-A run,
12 physical fits already spent there); this script only indexes them.  The only
raw data touched is the with-missing KDD2018 variant, read strictly before each
origin -- no horizon is read at all, so the ORACLE row of R4A has no counterpart
here by construction.

Thresholds, feature set and verdict words are the frozen ones: R4A's
``INFORMATIVE_AUC`` / ``WEAK_AUC`` and R4C's quantile grid and ``MATERIAL``
margin.  The statistics functions (``auc``, ``feature_reading``, ``stump_fold``,
``Reader``, ``mechanism_features``) are imported from R4A rather than
re-implemented, and ``serving_observables`` is imported from R4C.

Run: ``python -m evaluation.main_protocol_p4.audit_r4e_channel_identifiability``
0 Consumer fits, 0 LLM calls, 0 held-out reads, 0 horizon reads.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import audit_cross_fitted_targeting as targeting
from evaluation.main_protocol_p4 import audit_r4a_pattern_identifiability as r4a
from evaluation.main_protocol_p4 import audit_r4c_imputation_donor_conditions as r4c
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE = PROJECT_ROOT / "_scratch/r4d_a_three_cell_store.json"
R4A_ARTIFACT = PROJECT_ROOT / "artifacts/main_protocol/r4a_pattern_identifiability.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/r4e_channel_identifiability.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/r4e_channel_identifiability.md"

CONTEXT = int(preflight.CONTEXT)

PROGRAMS = ("ANCESTOR", "W2_pmc_then_outlier_mad")
CHANNELS = ("route", "ctx", "total")
FACES = ("support_face", "delayed_face")
FACE_GROUPS = ("support_face", "delayed_face", "both_faces")

# --- frozen constants, all inherited; nothing is tuned here ---
SEVERE = r4a.SEVERE_HARM                  # 0.30, loss units
INFORMATIVE_AUC = r4a.INFORMATIVE_AUC     # 0.70
WEAK_AUC = r4a.WEAK_AUC                   # 0.60
QUANTILES = r4c.QUANTILES                 # (1/3, 1/2, 2/3)
MATERIAL = r4c.MATERIAL                   # 0.005
CTX_ACTIVE_EPS = 1e-12

# R4A's 21 deployment-visible quantities (gap_period_aligned is NOT_COMPUTABLE
# in R4A and is not part of VISIBLE_FEATURES there either) plus R4C's six
# serving-side action conditions = 27.
FEATURES = tuple(r4a.VISIBLE_FEATURES) + tuple(r4c.SERVING_FEATURES)
TARGETS = ("helped", "severe_harm")


# ---------------------------------------------------------------------------
# 1. rows
# ---------------------------------------------------------------------------


def build_rows(entries: Mapping[str, Any],
               cards: Mapping[tuple[int, str], dict[str, float]]) -> list[dict]:
    """One row per (program, position, face, uid), carrying all three channels.

    ``g`` is kept per channel in *gain* orientation (``-channel``) so that the
    R4A utility functions, which were written for gains, can be reused without
    changing their sign convention.
    """
    rows: list[dict] = []
    for entry in entries.values():
        program = str(entry["program"])
        if program not in PROGRAMS:
            continue
        origin = int(entry["origin"])
        for index, uid in enumerate(entry["eval_uids"]):
            uid = str(uid)
            row = {
                "program": program,
                "position": int(entry["position"]),
                "face": str(entry["face"]),
                "origin": origin,
                "block": str(entry["block"]),
                "uid": uid,
                "status": str(entry["status"][index]),
                "serving_unmodified": bool(entry["serving_unmodified"][index]),
                "features": cards[(origin, uid)],
            }
            for channel in CHANNELS:
                row[channel] = float(entry[channel][index])
            rows.append(row)
    return rows


def channel_rows(rows: Sequence[Mapping[str, Any]], channel: str) -> list[dict]:
    """Recast rows for one channel into the shape ``r4a.feature_reading`` reads.

    For ``ctx`` the rows whose serving window was not modified are dropped: their
    ``ctx`` is exactly zero by construction, so they carry neither a helped nor a
    harmed reading, and keeping them would put a hard zero in both classes.
    """
    out = []
    for row in rows:
        value = float(row[channel])
        if channel == "ctx" and abs(value) <= CTX_ACTIVE_EPS:
            continue
        out.append({
            **{key: row[key] for key in
               ("program", "position", "face", "origin", "block", "uid",
                "features")},
            "value": value,
            "g": -value,
            "helped": 1 if value < 0.0 else 0,
            "severe_harm": 1 if value > SEVERE else 0,
        })
    return out


# ---------------------------------------------------------------------------
# 2. reading 1 -- per-channel identifiability (R4A logic, unchanged)
# ---------------------------------------------------------------------------


def _pick_best(readings: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, dict | None]:
    best_name, best = None, None
    for name in FEATURES:
        for target in TARGETS:
            reading = readings[name][target]
            value = reading.get("lodo_mean_auc_oriented")
            if value is None:
                continue
            if best is None or value > best["lodo_mean_auc_oriented"]:
                best_name, best = "%s|%s" % (name, target), dict(reading)
    return best_name, best


def _flatten(readings: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [readings[name][target] for name in FEATURES for target in TARGETS]


def identifiability(subset: Sequence[Mapping[str, Any]],
                    blocks: Sequence[str]) -> dict[str, Any]:
    if not subset:
        return {"n_rows": 0, "verdict": "NOT_EVALUABLE"}
    readings = {
        name: {target: r4a.feature_reading(subset, name, target, blocks)
               for target in TARGETS}
        for name in FEATURES
    }
    best_name, best = _pick_best(readings)
    cell_verdict, why = r4a.verdict(best, _flatten(readings))
    values = np.array([row["value"] for row in subset], dtype=np.float64)
    # The frozen rule takes the minimum over the folds that *have* an AUC, so a
    # fold with no positive instance is skipped rather than failed.  The verdict
    # is not changed here, but the support of every INFORMATIVE entry is written
    # down so that a verdict resting on two positives in two folds cannot be
    # read as if it rested on four.
    informative = []
    for name in FEATURES:
        for target in TARGETS:
            reading = readings[name][target]
            if not r4a._is_informative(reading):
                continue
            folds = reading["folds"]
            scored = [block for block, fold in folds.items()
                      if fold["auc_oriented"] is not None]
            positives = [fold["positives_test"] for fold in folds.values()
                         if fold["positives_test"] is not None]
            informative.append({
                "feature_target": "%s|%s" % (name, target),
                "lodo_mean": reading["lodo_mean_auc_oriented"],
                "lodo_min": reading["lodo_min_auc_oriented"],
                "folds_with_an_auc": len(scored),
                "folds_total": len(folds),
                "folds_without_a_positive_instance":
                    [block for block, fold in folds.items()
                     if fold["auc_oriented"] is None],
                "positives_per_fold": positives,
                "min_positives_in_a_scored_fold": (
                    min((folds[block]["positives_test"] for block in scored),
                        default=None)),
                "supported_by_every_fold": bool(len(scored) == len(folds)),
            })
    return {
        "n_rows": len(subset),
        "n_unique_uids": len({row["uid"] for row in subset}),
        "n_helped": int(sum(row["helped"] for row in subset)),
        "n_severe": int(sum(row["severe_harm"] for row in subset)),
        "helped_rate": float(np.mean([row["helped"] for row in subset])),
        "severe_rate": float(np.mean([row["severe_harm"] for row in subset])),
        "mean_channel_value": float(values.mean()),
        "blocks": list(blocks),
        "best_feature": best_name,
        "best_reading": best,
        "features_clearing_every_fold": informative,
        "n_features_clearing_every_fold": len(informative),
        "features_clearing_every_fold_with_full_support":
            [item["feature_target"] for item in informative
             if item["supported_by_every_fold"]],
        "verdict": cell_verdict,
        "verdict_reason": why,
        "per_feature": readings,
    }


# ---------------------------------------------------------------------------
# 3. reading 2 -- pattern-selected treatment vs one fixed choice for everybody
# ---------------------------------------------------------------------------


def _deployment_utility(total: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    """Whole-population utility of "selected get P, the rest keep identity".

    The gain of deploying P on a series is ``-total``; identity is exactly 0 by
    definition of the three-cell decomposition (``L_rr`` is the identity arm).
    The denominator is always the whole served population of the fold.
    """
    realised = np.where(selected, -total, 0.0)
    return {
        "utility": float(realised.mean()),
        "harmed": int((realised < 0.0).sum()),
        "worst_single_series": float(realised.min()) if realised.size else 0.0,
        "treated_fraction": float(selected.mean()) if selected.size else 0.0,
    }


def selection_readout(subset: Sequence[Mapping[str, Any]], feature: str,
                      blocks: Sequence[str]) -> dict[str, Any]:
    """Rule: deploy P iff ``feature <op> tau``; tau from the training folds only.

    ``(op, tau)`` maximises the whole-population utility on the training folds,
    is then read once on the held-back block, and is compared against the two
    fixed choices a deployer could make without any pattern -- always-P and
    always-identity -- and against the per-series oracle, which is an upper
    bound and is not deployable.
    """
    x = np.array([float(row["features"].get(feature, np.nan)) for row in subset],
                 dtype=np.float64)
    total = np.array([row["total"] for row in subset], dtype=np.float64)
    block = np.array([row["block"] for row in subset])
    folds: dict[str, Any] = {}
    for name in blocks:
        test = block == name
        train = ~test
        if not test.any() or not train.any():
            continue
        finite = x[train][np.isfinite(x[train])]
        if finite.size < 5:
            continue
        best = None
        for q in QUANTILES:
            tau = float(np.quantile(finite, q))
            for op in (">=", "<="):
                sel = ((x[train] >= tau) if op == ">=" else (x[train] <= tau))
                sel = sel & np.isfinite(x[train])
                score = _deployment_utility(total[train], sel)["utility"]
                if best is None or score > best[0]:
                    best = (score, op, tau, q)
        _score, op, tau, q = best
        sel = ((x[test] >= tau) if op == ">=" else (x[test] <= tau))
        sel = sel & np.isfinite(x[test])
        rule = _deployment_utility(total[test], sel)
        always_p = _deployment_utility(total[test],
                                       np.ones(int(test.sum()), dtype=bool))
        always_id = _deployment_utility(total[test],
                                        np.zeros(int(test.sum()), dtype=bool))
        oracle = _deployment_utility(total[test], -total[test] > 0.0)
        folds[name] = {
            "rule": "deploy P iff %s %s %.6f (train quantile %.3f)" % (
                feature, op, tau, q),
            "n_test": int(test.sum()),
            "rule_utility": rule,
            "always_p": always_p,
            "always_identity": always_id,
            "oracle_per_series_UPPER_BOUND_NOT_DEPLOYABLE": oracle,
            "vs_always_p": rule["utility"] - always_p["utility"],
            "vs_always_identity": rule["utility"] - always_id["utility"],
            "harmed_minus_always_p": rule["harmed"] - always_p["harmed"],
            "harmed_minus_always_identity": rule["harmed"] - always_id["harmed"],
            "worst_minus_always_p": (rule["worst_single_series"]
                                     - always_p["worst_single_series"]),
            "worst_minus_always_identity": (rule["worst_single_series"]
                                            - always_id["worst_single_series"]),
        }
    if not folds:
        return {"feature": feature, "folds": {}, "verdict": "NOT_EVALUABLE"}
    vs_p = [f["vs_always_p"] for f in folds.values()]
    vs_id = [f["vs_always_identity"] for f in folds.values()]
    beats_both = sum(1 for a, b in zip(vs_p, vs_id) if a > 0.0 and b > 0.0)
    mean_p, mean_id = float(np.mean(vs_p)), float(np.mean(vs_id))
    binding = min(mean_p, mean_id)
    verdict = ("SELECTION_BEATS_FIXED"
               if beats_both >= 3 and binding >= MATERIAL
               else "FIXED_CHOICE_NOT_BEATEN")
    return {
        "feature": feature,
        "n_folds": len(folds),
        "folds_beating_both_fixed": beats_both,
        "mean_vs_always_p": mean_p,
        "mean_vs_always_identity": mean_id,
        "binding_mean_margin": binding,
        "mean_harmed_minus_always_p": float(
            np.mean([f["harmed_minus_always_p"] for f in folds.values()])),
        "mean_harmed_minus_always_identity": float(
            np.mean([f["harmed_minus_always_identity"] for f in folds.values()])),
        "mean_worst_minus_always_p": float(
            np.mean([f["worst_minus_always_p"] for f in folds.values()])),
        "mean_worst_minus_always_identity": float(
            np.mean([f["worst_minus_always_identity"] for f in folds.values()])),
        "mean_oracle_utility_NOT_DEPLOYABLE": float(np.mean(
            [f["oracle_per_series_UPPER_BOUND_NOT_DEPLOYABLE"]["utility"]
             for f in folds.values()])),
        "mean_always_p_utility": float(np.mean(
            [f["always_p"]["utility"] for f in folds.values()])),
        "verdict": verdict,
        "folds": folds,
    }


def selection_table(subset: Sequence[Mapping[str, Any]],
                    blocks: Sequence[str]) -> dict[str, Any]:
    per_feature = {name: selection_readout(subset, name, blocks)
                   for name in FEATURES}
    best_name = None
    for name, reading in per_feature.items():
        if reading.get("binding_mean_margin") is None:
            continue
        if (best_name is None
                or reading["binding_mean_margin"]
                > per_feature[best_name]["binding_mean_margin"]):
            best_name = name
    any_beat = [name for name, reading in per_feature.items()
                if reading["verdict"] == "SELECTION_BEATS_FIXED"]
    return {
        "n_rows": len(subset),
        "best_feature_by_binding_margin": best_name,
        "best_reading": per_feature[best_name] if best_name else None,
        "features_with_SELECTION_BEATS_FIXED": any_beat,
        "verdict": ("SELECTION_BEATS_FIXED" if any_beat
                    else "FIXED_CHOICE_NOT_BEATEN"),
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# 4. reading 4 -- does this store reproduce R4A's total-channel reading?
# ---------------------------------------------------------------------------


def r4a_crosscheck(table: Mapping[str, Any]) -> dict[str, Any]:
    """R4A's best feature was picked over its 21 visible quantities only."""
    if not R4A_ARTIFACT.exists():
        return {"available": False,
                "why": "artifacts/main_protocol/r4a_pattern_identifiability.json absent"}
    published = json.loads(R4A_ARTIFACT.read_text(encoding="utf-8"))["main_table"]
    rows = []
    for program in PROGRAMS:
        for face in FACES:
            cell = table["%s|total|%s" % (program, face)]
            here_name, here = None, None
            for name in r4a.VISIBLE_FEATURES:
                for target in TARGETS:
                    reading = cell["per_feature"][name][target]
                    value = reading.get("lodo_mean_auc_oriented")
                    if value is None:
                        continue
                    if here is None or value > here:
                        here_name, here = "%s|%s" % (name, target), value
            there = published.get("%s|%s" % (program, face))
            there_name = there["best_visible_feature"] if there else None
            there_value = ((there["best_visible_reading"] or {})
                           .get("lodo_mean_auc_oriented") if there else None)
            agrees = bool(here_name == there_name and there_value is not None
                          and abs(here - there_value) <= 5e-6)
            rows.append({
                "cell": "%s|%s" % (program, face),
                "r4e_best_feature": here_name,
                "r4e_lodo_mean": here,
                "r4a_best_feature": there_name,
                "r4a_lodo_mean": there_value,
                "n_rows_r4e": cell["n_rows"],
                "n_rows_r4a": there["n_rows"] if there else None,
                "agrees": agrees,
                "abs_difference": (None if there_value is None
                                   else abs(here - there_value)),
            })
    return {
        "available": True,
        "rule": ("same best (feature|target) over R4A's 21 visible quantities and "
                 "the same LODO mean to 5e-6"),
        "all_agree": all(row["agrees"] for row in rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 5. run
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    payload_store = json.loads(STORE.read_text(encoding="utf-8"))
    entries = payload_store["entries"]

    variant = preflight.load_variant()
    nan_count = int(sum(int(np.isnan(series).sum()) for series in variant.values()))
    if nan_count <= 0:
        raise AssertionError(
            "the loaded variant has no NaN; this is the filled cache, not "
            "EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING")
    reader = r4a.Reader(variant)

    # --- feature cards, one per (origin, uid); no horizon is ever read ---
    # Keyed by (position, face): two positions in different cohort blocks can
    # share the same origin while serving different uids, so the origin alone
    # does not identify the cell.
    cells: dict[tuple[int, str], tuple[int, list[str]]] = {}
    for entry in entries.values():
        if str(entry["program"]) not in PROGRAMS:
            continue
        cells.setdefault((int(entry["position"]), str(entry["face"])),
                         (int(entry["origin"]),
                          [str(uid) for uid in entry["eval_uids"]]))
    cards: dict[tuple[int, str], dict[str, float]] = {}
    serving_checks = {"contexts": 0, "cross_check_failed": 0}
    for _key, (origin, uids) in sorted(cells.items()):
        matrix, names = targeting.series_features(variant, uids, origin)
        for uid in uids:
            reader.account_external(uid, origin)
        for index, uid in enumerate(uids):
            if (origin, uid) in cards:
                continue
            context = reader.window(uid, origin - CONTEXT, origin)
            card = {name: float(matrix[index][column])
                    for column, name in enumerate(names)
                    if name in r4a.OBSERVABLE_NUMERIC}
            card.update({key: value
                         for key, value in r4a.mechanism_features(context).items()
                         if key in r4a.VISIBLE_FEATURES})
            serving, ok = r4c.serving_observables(context)
            card.update(serving)
            serving_checks["contexts"] += 1
            serving_checks["cross_check_failed"] += 0 if ok else 1
            missing = [name for name in FEATURES if name not in card]
            if missing:
                raise AssertionError("feature card is missing %s" % missing)
            cards[(origin, uid)] = card

    rows = build_rows(entries, cards)
    blocks = sorted({row["block"] for row in rows})
    statuses = sorted({row["status"] for row in rows})
    if statuses != ["OK"]:
        raise AssertionError("the three-cell store carries non-OK statuses: %s"
                             % statuses)

    # --- reading 1, per (program, channel, face group) ---
    table: dict[str, Any] = {}
    for program in PROGRAMS:
        for channel in CHANNELS:
            base = [row for row in rows if row["program"] == program]
            recast = channel_rows(base, channel)
            dropped = len(base) - len(recast)
            for group in FACE_GROUPS:
                subset = ([row for row in recast if row["face"] == group]
                          if group in FACES else list(recast))
                present = sorted({row["block"] for row in subset})
                cell = identifiability(subset, present)
                cell["channel"] = channel
                cell["program"] = program
                cell["face_group"] = group
                if channel == "ctx":
                    in_group = ([row for row in base if row["face"] == group]
                                if group in FACES else base)
                    cell["ctx_instances_serving_modified"] = len(subset)
                    cell["ctx_instances_serving_unmodified"] = (
                        len(in_group) - len(subset))
                    cell["ctx_rows_dropped_total_for_program"] = dropped
                table["%s|%s|%s" % (program, channel, group)] = cell

    # --- reading 2, per (program, face group); always on total ---
    selection: dict[str, Any] = {}
    for program in PROGRAMS:
        base = [row for row in rows if row["program"] == program]
        for group in FACE_GROUPS:
            subset = ([row for row in base if row["face"] == group]
                      if group in FACES else base)
            present = sorted({row["block"] for row in subset})
            selection["%s|%s" % (program, group)] = selection_table(subset, present)

    # --- reading 3, channel comparison on the merged faces ---
    comparison: dict[str, Any] = {}
    for program in PROGRAMS:
        per_channel = {}
        for channel in CHANNELS:
            cell = table["%s|%s|both_faces" % (program, channel)]
            best = cell.get("best_reading") or {}
            per_channel[channel] = {
                "best_feature": cell.get("best_feature"),
                "lodo_mean": best.get("lodo_mean_auc_oriented"),
                "lodo_min": best.get("lodo_min_auc_oriented"),
                "verdict": cell.get("verdict"),
                "n_rows": cell.get("n_rows"),
            }
        ranked = [name for name in CHANNELS
                  if per_channel[name]["lodo_mean"] is not None]
        ranked.sort(key=lambda name: per_channel[name]["lodo_mean"], reverse=True)
        route_feature = per_channel["route"]["best_feature"]
        ctx_feature = per_channel["ctx"]["best_feature"]
        comparison[program] = {
            "per_channel": per_channel,
            "most_identifiable_channel": ranked[0] if ranked else None,
            "route_and_ctx_share_the_best_feature":
                bool(route_feature is not None and route_feature == ctx_feature),
            "route_best_feature": route_feature,
            "ctx_best_feature": ctx_feature,
        }

    crosscheck = r4a_crosscheck(table)

    ctx_active = sum(1 for row in rows if abs(row["ctx"]) > CTX_ACTIVE_EPS)
    payload = {
        "task": "R4E_CHANNEL_IDENTIFIABILITY",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_class": ("MECHANISM / NEGATIVE-capable; development reading; "
                           "not a capability claim"),
        "question": ("route and ctx are two different channels of the same "
                     "effect; does either of them carry a deployment-visible "
                     "pattern that total hides, and does selecting the "
                     "treatment on that pattern beat choosing one treatment for "
                     "everybody?"),
        "boundary": {
            "consumer_fits": 0,
            "llm_calls": 0,
            "held_out_reads": reader.held_out_reads,
            "horizon_reads": 0,
            "max_time_index_read": reader.max_index_read,
            "held_out_frontier": r4a.MAX_ALLOWED_INDEX,
            "held_out_origins": list(r4a.HELD_OUT_ORIGINS),
            "raw_series_reads": reader.reads,
            "existing_files_edited": 0,
            "prediction_store_written": False,
            "new_sha_or_hash": 0,
            "git_commits": 0,
            "sub_agents_spawned": 0,
            "files_written": [str(OUT_JSON.relative_to(PROJECT_ROOT)),
                              str(OUT_MD.relative_to(PROJECT_ROOT))],
        },
        "provenance": {
            "three_cell_store": str(STORE.relative_to(PROJECT_ROOT)),
            "three_cell_store_generated_at": payload_store.get("generated_at"),
            "three_cell_store_entries": len(entries),
            "physical_fits_already_spent_in_that_store":
                payload_store["boundary"]["physical_fits"],
            "loader": ("evaluation.main_protocol_p4.preflight_natural_gap_variant"
                       ".load_variant"),
            "loader_nan_count": nan_count,
            "data_version": preflight.DATA_VERSION,
            "context_steps": CONTEXT,
            "feature_sources": {
                "observable_vocabulary_12": ("audit_cross_fitted_targeting"
                                             ".series_features, as R4A uses it"),
                "mechanism_9": ("audit_r4a_pattern_identifiability"
                                ".mechanism_features; gap_period_aligned is "
                                "excluded, as it is in R4A, because the "
                                "definition is degenerate on a 48-step horizon"),
                "serving_action_conditions_6": ("audit_r4c_imputation_donor_"
                                                "conditions.serving_observables; "
                                                "meaningful for W2, computed for "
                                                "ANCESTOR only as a reference "
                                                "column since the ancestor does "
                                                "not run period_median_complete"),
            },
            "statistics_reused_from_r4a": ["Reader", "auc", "_average_ranks",
                                           "_frozen_edges", "stump_fold",
                                           "feature_reading", "verdict"],
            "serving_cross_check": serving_checks,
            "rows": {
                "unit": "(program, position, face, uid)",
                "total_rows": len(rows),
                "unique_uids": len({row["uid"] for row in rows}),
                "positions": len({row["position"] for row in rows}),
                "blocks": blocks,
                "statuses": statuses,
                "rows_dropped_for_status": 0,
                "ctx_rows_with_a_modified_serving_window": ctx_active,
                "ctx_rows_with_an_untouched_serving_window":
                    len(rows) - ctx_active,
                "repeated_measures": ("the same uid recurs at every position of "
                                      "its block; rows are not independent and "
                                      "no significance is claimed"),
            },
        },
        "definitions": {
            "route": "L_pr - L_rr, the shared-model channel (loss units)",
            "ctx": "L_pp - L_pr, this series' own serving-context channel",
            "total": "route + ctx = L_pp - L_rr = -g in the library convention",
            "helped_c": "channel < 0 (the loss went down)",
            "severe_c": "channel > %.2f" % SEVERE,
            "ctx_restriction": ("rows with |ctx| <= %g are dropped from the ctx "
                                "channel: their serving window was not modified, "
                                "so ctx is exactly zero by construction"
                                % CTX_ACTIVE_EPS),
            "auc": ("Mann-Whitney U with average ranks; each LODO fold is "
                    "oriented by the sign the training blocks give"),
            "deployment_utility": ("mean over the whole served population of the "
                                   "fold of (-total if the rule selects the "
                                   "series else 0); identity is the L_rr arm and "
                                   "scores exactly 0"),
            "frozen_thresholds": {
                "informative_auc": INFORMATIVE_AUC,
                "weak_auc": WEAK_AUC,
                "severe": SEVERE,
                "selection_material_margin": MATERIAL,
                "selection_quantiles": list(QUANTILES),
            },
            "verdict_rules": {
                "PATTERN_INFORMATIVE": ("some quantity has every LODO fold >= "
                                        "%.2f with one direction across folds"
                                        % INFORMATIVE_AUC),
                "PATTERN_WEAK": ("best LODO mean in [%.2f, %.2f), or the folds "
                                 "disagree on direction"
                                 % (WEAK_AUC, INFORMATIVE_AUC)),
                "PATTERN_UNINFORMATIVE": "best LODO mean < %.2f" % WEAK_AUC,
                "SELECTION_BEATS_FIXED": ("at least 3 of 4 folds beat both "
                                          "always-P and always-identity and the "
                                          "smaller of the two mean margins is "
                                          ">= %.3f" % MATERIAL),
                "FIXED_CHOICE_NOT_BEATEN": "anything else",
            },
        },
        "features": {
            "observable_vocabulary": list(r4a.OBSERVABLE_NUMERIC),
            "mechanism": list(r4a.SPIKE_FEATURES + r4a.GAP_FEATURES),
            "serving_action_conditions": list(r4c.SERVING_FEATURES),
            "n_features": len(FEATURES),
            "excluded": list(r4a.NOT_COMPUTABLE),
        },
        "reading_1_channel_identifiability": table,
        "reading_2_selection_vs_fixed_choice": selection,
        "reading_3_channel_comparison": comparison,
        "reading_4_r4a_crosscheck": crosscheck,
    }
    return payload


# ---------------------------------------------------------------------------
# 6. rendering
# ---------------------------------------------------------------------------


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 6)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    return value


def _num(value: Any, fmt: str = "%.6f") -> str:
    return "n/a" if value is None else fmt % value


def _bar(text: Any) -> str:
    return str(text).replace("|", "\\|")


def _md(p: Mapping[str, Any]) -> str:
    boundary = p["boundary"]
    rows = p["provenance"]["rows"]
    lines = [
        "# R4E per-channel identifiability -- machine reading", "",
        "Evidence class: %s" % p["evidence_class"], "",
        "Boundary: fits=%d, llm=%d, held_out_reads=%d, horizon_reads=%d, "
        "max_time_index_read=%d (frontier %d), existing_files_edited=%d, "
        "new_sha_or_hash=%d."
        % (boundary["consumer_fits"], boundary["llm_calls"],
           boundary["held_out_reads"], boundary["horizon_reads"],
           boundary["max_time_index_read"], boundary["held_out_frontier"],
           boundary["existing_files_edited"], boundary["new_sha_or_hash"]),
        "",
        "Rows: %d (program, position, face, uid); %d unique uids over %d "
        "positions and blocks %s; statuses %s, dropped for status %d.  ctx "
        "channel keeps the %d rows whose serving window was modified and drops "
        "the %d whose ctx is exactly zero."
        % (rows["total_rows"], rows["unique_uids"], rows["positions"],
           ", ".join(rows["blocks"]), ", ".join(rows["statuses"]),
           rows["rows_dropped_for_status"],
           rows["ctx_rows_with_a_modified_serving_window"],
           rows["ctx_rows_with_an_untouched_serving_window"]),
        "",
        "Features: %d = 12 vocabulary numbers + 9 mechanism quantities + 6 "
        "serving action conditions.  Serving cross-check failed %d of %d "
        "contexts." % (p["features"]["n_features"],
                       p["provenance"]["serving_cross_check"]["cross_check_failed"],
                       p["provenance"]["serving_cross_check"]["contexts"]),
        "",
        "## Reading 1 -- per-channel LODO identifiability", "",
        "| program | channel | face | rows | helped | severe | best "
        "(feature\\|target) | LODO mean | LODO min | folds | clears every fold "
        "(of which on 4/4 folds) | verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- "
        "| --- |",
    ]
    for key, cell in p["reading_1_channel_identifiability"].items():
        if not cell.get("n_rows"):
            lines.append("| %s | | | 0 | | | | | | | | %s |"
                         % (_bar(key), cell.get("verdict")))
            continue
        best = cell.get("best_reading") or {}
        folds = ", ".join(
            "%s=%s" % (block, _num(fold.get("auc_oriented")))
            for block, fold in (best.get("folds") or {}).items())
        lines.append(
            "| %s | %s | %s | %d | %d | %d | %s | %s | %s | %s | %d (%d) | %s |"
            % (cell["program"], cell["channel"], cell["face_group"],
               cell["n_rows"], cell["n_helped"], cell["n_severe"],
               _bar(cell["best_feature"]),
               _num(best.get("lodo_mean_auc_oriented")),
               _num(best.get("lodo_min_auc_oriented")), folds,
               cell["n_features_clearing_every_fold"],
               len(cell["features_clearing_every_fold_with_full_support"]),
               cell["verdict"]))
    lines += ["", "Support of every INFORMATIVE entry (the frozen rule takes the "
              "minimum over the folds that have an AUC, so a fold with no "
              "positive instance is skipped, not failed):", ""]
    for key, cell in p["reading_1_channel_identifiability"].items():
        for item in cell.get("features_clearing_every_fold") or ():
            lines.append(
                "- `%s` / `%s`: LODO mean %s, min %s over %d of %d folds; "
                "positives per fold %s; folds without a positive instance: %s"
                % (_bar(key), _bar(item["feature_target"]),
                   _num(item["lodo_mean"]), _num(item["lodo_min"]),
                   item["folds_with_an_auc"], item["folds_total"],
                   item["positives_per_fold"],
                   ", ".join(item["folds_without_a_positive_instance"]) or "none"))
    lines += ["", "## Reading 2 -- select on a pattern vs one fixed choice "
              "(utility uses total, denominator is the whole served population)",
              "",
              "| program | face | best feature | mean vs always-P | mean vs "
              "always-identity | folds beating both | harmed vs P | harmed vs "
              "identity | worst vs P | always-P utility | oracle (not "
              "deployable) | verdict |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
              "---: | ---: | --- |"]
    for key, cell in p["reading_2_selection_vs_fixed_choice"].items():
        program, group = key.split("|")
        best = cell.get("best_reading") or {}
        lines.append("| %s | %s | %s | %s | %s | %s/%s | %s | %s | %s | %s | %s "
                     "| %s |" % (
            program, group, _bar(cell.get("best_feature_by_binding_margin")),
            _num(best.get("mean_vs_always_p"), "%+.6f"),
            _num(best.get("mean_vs_always_identity"), "%+.6f"),
            best.get("folds_beating_both_fixed"), best.get("n_folds"),
            _num(best.get("mean_harmed_minus_always_p"), "%+.2f"),
            _num(best.get("mean_harmed_minus_always_identity"), "%+.2f"),
            _num(best.get("mean_worst_minus_always_p"), "%+.6f"),
            _num(best.get("mean_always_p_utility")),
            _num(best.get("mean_oracle_utility_NOT_DEPLOYABLE")),
            cell["verdict"]))
    lines += ["", "Per-fold rules of the best feature, per program (merged "
              "faces):", ""]
    for program in PROGRAMS:
        cell = p["reading_2_selection_vs_fixed_choice"]["%s|both_faces" % program]
        best = cell.get("best_reading") or {}
        for block, fold in (best.get("folds") or {}).items():
            lines.append("- `%s` / %s: %s -> utility %s, always-P %s, "
                         "always-identity %s, oracle %s (n=%d, treated %s)"
                         % (program, block, fold["rule"],
                            _num(fold["rule_utility"]["utility"]),
                            _num(fold["always_p"]["utility"]),
                            _num(fold["always_identity"]["utility"]),
                            _num(fold["oracle_per_series_UPPER_BOUND_NOT_"
                                      "DEPLOYABLE"]["utility"]),
                            fold["n_test"],
                            _num(fold["rule_utility"]["treated_fraction"])))
    lines += ["", "## Reading 3 -- which channel is more identifiable", ""]
    for program, cell in p["reading_3_channel_comparison"].items():
        parts = ", ".join(
            "%s: %s (%s, %s)" % (channel, _num(value["lodo_mean"]),
                                 value["best_feature"], value["verdict"])
            for channel, value in cell["per_channel"].items())
        lines.append("- `%s`: most identifiable channel = **%s**; %s; route and "
                     "ctx share the best feature: %s"
                     % (program, cell["most_identifiable_channel"], parts,
                        cell["route_and_ctx_share_the_best_feature"]))
    lines += ["", "## Reading 4 -- cross-check against the published R4A total "
              "reading", "",
              "| cell | R4E best | R4E LODO mean | R4A best | R4A LODO mean | "
              "rows R4E/R4A | agrees |",
              "| --- | --- | ---: | --- | ---: | ---: | --- |"]
    crosscheck = p["reading_4_r4a_crosscheck"]
    if crosscheck.get("available"):
        for row in crosscheck["rows"]:
            lines.append("| %s | %s | %s | %s | %s | %s/%s | %s |" % (
                _bar(row["cell"]), _bar(row["r4e_best_feature"]),
                _num(row["r4e_lodo_mean"]), _bar(row["r4a_best_feature"]),
                _num(row["r4a_lodo_mean"]), row["n_rows_r4e"], row["n_rows_r4a"],
                row["agrees"]))
        lines += ["", "All four cells agree: **%s**" % crosscheck["all_agree"]]
    else:
        lines += ["", crosscheck.get("why", "not available")]
    lines += ["", "## Per-feature detail", "",
              "The full per-feature LODO tables, frozen-bin stumps and "
              "per-fold selection rules are in "
              "`artifacts/main_protocol/r4e_channel_identifiability.json`.", ""]
    return "\n".join(lines)


def main() -> int:
    payload = _round(run())
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    boundary = payload["boundary"]
    print("fits=%d llm=%d held_out_reads=%d horizon_reads=%d "
          "max_time_index_read=%d" % (
              boundary["consumer_fits"], boundary["llm_calls"],
              boundary["held_out_reads"], boundary["horizon_reads"],
              boundary["max_time_index_read"]))
    for key, cell in payload["reading_1_channel_identifiability"].items():
        if not cell.get("n_rows"):
            print("%-48s %s" % (key, cell.get("verdict")))
            continue
        best = cell.get("best_reading") or {}
        print("%-48s %-22s n=%-4d best=%-38s lodo=%s" % (
            key, cell["verdict"], cell["n_rows"], cell["best_feature"],
            _num(best.get("lodo_mean_auc_oriented"))))
    for key, cell in payload["reading_2_selection_vs_fixed_choice"].items():
        best = cell.get("best_reading") or {}
        print("selection %-40s %-24s best=%-30s vsP=%s vsID=%s" % (
            key, cell["verdict"], cell["best_feature_by_binding_margin"],
            _num(best.get("mean_vs_always_p"), "%+.6f"),
            _num(best.get("mean_vs_always_identity"), "%+.6f")))
    print("r4a_crosscheck all_agree=%s"
          % payload["reading_4_r4a_crosscheck"].get("all_agree"))
    print("wrote %s" % OUT_JSON)
    print("wrote %s" % OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
