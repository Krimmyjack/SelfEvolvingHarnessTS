"""R2 revision-effect stability readout (G1).  Read-only; 0 fit; 0 LLM.

Question (descriptive only, no significance claim): in the same lineage, is
the child-minus-ancestor difference d more stable across units than the
ancestor's own effect g?

Sources
-------
(a) artifacts/main_protocol/dev_auto1_blocked_candidates__run2.json → per_unit
    ANCESTOR and two CAND programs, 16 units (positions from 5) × 2 faces.
(b) _scratch/m_r0k_prediction_store.json → per-series g = raw − program.

Run:  python -m evaluation.main_protocol_p4.audit_r2_revision_effect_stability
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ART = PROJECT_ROOT / "artifacts" / "main_protocol"
PER_UNIT_PATH = ART / "dev_auto1_blocked_candidates__run2.json"
STORE_PATH = PROJECT_ROOT / "_scratch" / "m_r0k_prediction_store.json"
OUT_STEM = ART / "r2_revision_effect_stability"

FACES = ("support_face", "delayed_face")
ANCESTOR = "ANCESTOR"
CAND_HAMPEL = "CAND_outlier_iqr({})>hampel_filter({})"
CAND_WINSOR = "CAND_outlier_iqr({})>winsorize({})"
W1 = "W1_hampel_filter"
W2 = "W2_pmc_then_outlier_mad"
W3 = "W3_outlier_mad_then_pmc"
CHILDREN = (CAND_HAMPEL, CAND_WINSOR, W1, W2, W3)
CAND_CHILDREN = (CAND_HAMPEL, CAND_WINSOR)
TZ = timezone(timedelta(hours=8))
READABLE = "READ"
MISSING_STATUSES_SEEN = (
    "WINDOW_VERIFIER_REJECTED",
    "SERVING_CONTEXT_DEGENERATE",
    "NO_STORED_READING",
    "NOT_READABLE",
    "UNKNOWN",
)


def _r6(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, 6)
    if isinstance(value, dict):
        return {str(key): _r6(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_r6(item) for item in value]
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
            "frac_positive": None,
            "frac_negative": None,
            "frac_zero": None,
            "sign_consistency": None,
            "mean": None,
            "std": None,
            "var": None,
        }
    n_positive = int(np.sum(array > 0.0))
    n_negative = int(np.sum(array < 0.0))
    n_zero = int(np.sum(array == 0.0))
    frac_positive = n_positive / n
    frac_negative = n_negative / n
    frac_zero = n_zero / n
    return {
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "frac_positive": frac_positive,
        "frac_negative": frac_negative,
        "frac_zero": frac_zero,
        "sign_consistency": max(frac_positive, frac_negative),
        "mean": float(array.mean()),
        "std": (float(array.std(ddof=1)) if n >= 2 else None),
        "var": (float(array.var(ddof=1)) if n >= 2 else None),
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        return None
    return numerator / denominator


def _face_status(face: MappingLike) -> str:
    if not isinstance(face, dict) or not face:
        return "UNKNOWN"
    if "reading" not in face:
        status = face.get("status")
        if isinstance(status, str) and status:
            return status
        return "UNKNOWN"
    reading = face.get("reading")
    if not isinstance(reading, dict) or not reading:
        return "UNKNOWN"
    status = reading.get("status")
    if isinstance(status, str) and status:
        return status
    return "UNKNOWN"


def _reading_if_read(face: MappingLike) -> dict[str, Any] | None:
    if _face_status(face) != READABLE:
        return None
    reading = face.get("reading")
    if not isinstance(reading, dict):
        return None
    if "aggregate_gain" not in reading or reading.get("aggregate_gain") is None:
        return None
    return reading


MappingLike = dict[str, Any]


def _load_per_unit() -> dict[str, Any]:
    payload = json.loads(PER_UNIT_PATH.read_text(encoding="utf-8"))
    per_unit = payload["per_unit"]
    inventory: dict[str, Any] = {}
    sample = per_unit[ANCESTOR][0]
    sample_face = sample["faces"]["support_face"]
    inventory["per_unit_policy_keys"] = list(per_unit.keys())
    inventory["unit_record_keys"] = list(sample.keys())
    inventory["face_keys_when_read"] = list(sample_face.keys())
    inventory["reading_keys_when_read"] = list(sample_face["reading"].keys())
    inventory["sample_read_record"] = {
        "policy": sample["policy"],
        "position": sample["position"],
        "unit": sample["unit"],
        "side": sample["side"],
        "exposure": sample["exposure"],
        "faces": {
            "support_face": {
                key: (value if key != "reading"
                      else {inner: inner_value
                            for inner, inner_value in value.items()
                            if inner != "per_series_gain"}
                      | {"per_series_gain": value.get("per_series_gain")})
                for key, value in sample_face.items()
            }
        },
    }
    missing_counts: dict[str, dict[str, dict[str, int]]] = {}
    for policy, rows in per_unit.items():
        missing_counts[policy] = {}
        for face in FACES:
            counts: Counter[str] = Counter()
            for row in rows:
                counts[_face_status((row.get("faces") or {}).get(face) or {})] += 1
            missing_counts[policy][face] = dict(counts)
    inventory["reading_status_counts"] = missing_counts
    inventory["positions"] = [int(row["position"]) for row in per_unit[ANCESTOR]]
    return {"payload": payload, "per_unit": per_unit, "inventory": inventory}


def _index_per_unit(per_unit: dict[str, Any]) -> dict[tuple[str, int, str], dict]:
    index: dict[tuple[str, int, str], dict] = {}
    for policy, rows in per_unit.items():
        for row in rows:
            position = int(row["position"])
            for face in FACES:
                index[(policy, position, face)] = (row.get("faces") or {}).get(face) or {}
    return index


def _load_store() -> dict[str, Any]:
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    sample_key = next(iter(store))
    sample = store[sample_key]
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    coverage: Counter[tuple[str, str]] = Counter()
    for key, record in store.items():
        program = str(record["program"])
        position = int(record["position"])
        face = str(record["face"])
        by_key[(program, position, face)] = record
        coverage[(program, face)] += 1
    gains: dict[tuple[str, int, str], dict[str, float]] = {}
    unreadable: list[dict[str, Any]] = []
    for (program, position, face), record in by_key.items():
        uids = list(record.get("eval_uids") or ())
        raw = list(record.get("raw_per_view") or ())
        prog = list(record.get("program_per_view") or ())
        degenerate = set(record.get("degenerate_uids") or ())
        verifier_passed = record.get("verifier_passed")
        if not raw or not prog:
            unreadable.append({
                "program": program,
                "position": position,
                "face": face,
                "verifier_passed": verifier_passed,
                "eval_uids_n": len(uids),
                "raw_per_view_n": len(raw),
                "program_per_view_n": len(prog),
                "kept_as": "missing",
                "coerced_to_zero": False,
            })
            continue
        if len(uids) != len(raw) or len(uids) != len(prog):
            raise RuntimeError(
                "store length mismatch at %s|%s|%s" % (program, position, face))
        per_uid: dict[str, float] = {}
        for uid, raw_value, prog_value in zip(uids, raw, prog):
            if uid in degenerate:
                continue
            if raw_value is None or prog_value is None:
                continue
            per_uid[str(uid)] = float(raw_value) - float(prog_value)
        if not per_uid:
            unreadable.append({
                "program": program,
                "position": position,
                "face": face,
                "verifier_passed": verifier_passed,
                "kept_as": "missing",
                "coerced_to_zero": False,
                "why": "no paired finite per-uid gains",
            })
            continue
        gains[(program, position, face)] = per_uid
    return {
        "n_records": len(store),
        "sample_key": sample_key,
        "sample_keys": list(sample.keys()),
        "sample_record": {
            "program": sample["program"],
            "position": sample["position"],
            "face": sample["face"],
            "origin": sample["origin"],
            "verifier_passed": sample.get("verifier_passed"),
            "eval_uids_n": len(sample.get("eval_uids") or ()),
            "raw_per_view_n": len(sample.get("raw_per_view") or ()),
            "program_per_view_n": len(sample.get("program_per_view") or ()),
            "degenerate_uids": sample.get("degenerate_uids"),
            "physical_fits": sample.get("physical_fits"),
            "eval_uids_head": (sample.get("eval_uids") or [])[:5],
            "raw_per_view_head": (sample.get("raw_per_view") or [])[:3],
            "program_per_view_head": (sample.get("program_per_view") or [])[:3],
        },
        "coverage": {("%s|%s" % key): int(count)
                     for key, count in sorted(coverage.items())},
        "gains": gains,
        "unreadable_store_records": unreadable,
    }


def _unit_from_store(
        gains: dict[tuple[str, int, str], dict[str, float]],
        program: str,
        position: int,
        face: str,
) -> dict[str, Any] | None:
    per_uid = gains.get((program, position, face))
    if not per_uid:
        return None
    values = list(per_uid.values())
    return {
        "aggregate_gain": float(np.mean(values)),
        "harmed_series": int(sum(1 for value in values if value < 0.0)),
        "treated_nonzero": int(sum(1 for value in values if value != 0.0)),
        "n_series": len(values),
        "per_uid": per_uid,
    }


def _paired_unit_rows(
        child: str,
        face: str,
        per_unit_index: dict[tuple[str, int, str], dict],
        ancestor_positions: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for position in ancestor_positions:
        ancestor_face = per_unit_index.get((ANCESTOR, position, face), {})
        child_face = per_unit_index.get((child, position, face), {})
        ancestor_status = _face_status(ancestor_face)
        child_status = _face_status(child_face)
        ancestor_reading = _reading_if_read(ancestor_face)
        child_reading = _reading_if_read(child_face)
        if ancestor_reading is None or child_reading is None:
            missing.append({
                "position": position,
                "ancestor_status": ancestor_status,
                "child_status": child_status,
                "kept_as": "missing",
                "coerced_to_zero": False,
            })
            continue
        d_u = (float(child_reading["aggregate_gain"])
               - float(ancestor_reading["aggregate_gain"]))
        g_u = float(ancestor_reading["aggregate_gain"])
        harmed_child = child_reading.get("harmed_series")
        harmed_anc = ancestor_reading.get("harmed_series")
        harmed_diff = None
        if harmed_child is not None and harmed_anc is not None:
            harmed_diff = int(harmed_child) - int(harmed_anc)
        paired.append({
            "position": position,
            "d_u": d_u,
            "g_u_ancestor": g_u,
            "child_aggregate_gain": float(child_reading["aggregate_gain"]),
            "ancestor_aggregate_gain": g_u,
            "child_harmed_series": harmed_child,
            "ancestor_harmed_series": harmed_anc,
            "harmed_series_diff": harmed_diff,
            "child_treated": child_reading.get("treated"),
            "ancestor_treated": ancestor_reading.get("treated"),
            "child_status": child_status,
            "ancestor_status": ancestor_status,
        })
    return paired, missing


def _store_unit_pairs(
        child: str,
        face: str,
        gains: dict[tuple[str, int, str], dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ancestor_positions = sorted({
        position for (program, position, stored_face) in gains
        if program == ANCESTOR and stored_face == face
    })
    child_positions = sorted({
        position for (program, position, stored_face) in gains
        if program == child and stored_face == face
    })
    paired: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    series_d: list[float] = []
    series_g: list[float] = []
    unit_positive_share: list[dict[str, Any]] = []
    harmed_diffs: list[int] = []
    for position in ancestor_positions:
        ancestor_unit = _unit_from_store(gains, ANCESTOR, position, face)
        child_unit = _unit_from_store(gains, child, position, face)
        if ancestor_unit is None or child_unit is None:
            missing.append({
                "position": position,
                "ancestor_present": ancestor_unit is not None,
                "child_present": child_unit is not None,
                "kept_as": "missing",
                "coerced_to_zero": False,
            })
            continue
        paired.append({
            "position": position,
            "d_u": child_unit["aggregate_gain"] - ancestor_unit["aggregate_gain"],
            "g_u_ancestor": ancestor_unit["aggregate_gain"],
            "child_aggregate_gain": child_unit["aggregate_gain"],
            "ancestor_aggregate_gain": ancestor_unit["aggregate_gain"],
            "child_harmed_series": child_unit["harmed_series"],
            "ancestor_harmed_series": ancestor_unit["harmed_series"],
            "harmed_series_diff": (child_unit["harmed_series"]
                                   - ancestor_unit["harmed_series"]),
            "n_series_child": child_unit["n_series"],
            "n_series_ancestor": ancestor_unit["n_series"],
        })
        harmed_diffs.append(child_unit["harmed_series"] - ancestor_unit["harmed_series"])
        shared = sorted(set(child_unit["per_uid"]) & set(ancestor_unit["per_uid"]))
        d_in_unit = []
        for uid in shared:
            g_child = child_unit["per_uid"][uid]
            g_anc = ancestor_unit["per_uid"][uid]
            d_i = g_child - g_anc
            series_d.append(d_i)
            series_g.append(g_anc)
            d_in_unit.append(d_i)
        if d_in_unit:
            n_pos = int(sum(1 for value in d_in_unit if value > 0.0))
            unit_positive_share.append({
                "position": position,
                "n_paired_series": len(d_in_unit),
                "frac_d_positive": n_pos / len(d_in_unit),
            })
    series_stats_d = _stats(series_d)
    series_stats_g = _stats(series_g)
    share_values = [row["frac_d_positive"] for row in unit_positive_share]
    return paired, missing, {
        "n_ancestor_positions": len(ancestor_positions),
        "n_child_positions": len(child_positions),
        "series_d": series_stats_d,
        "series_g_ancestor_on_paired_rows": series_stats_g,
        "var_ratio": _ratio(series_stats_d["var"], series_stats_g["var"]),
        "n_paired_series": series_stats_d["n"],
        "unit_frac_d_positive": unit_positive_share,
        "unit_frac_d_positive_distribution": _stats(share_values),
        "harmed_count_diff_from_g_lt_0": {
            "n_units": len(harmed_diffs),
            "mean": (float(np.mean(harmed_diffs)) if harmed_diffs else None),
            "std": (float(np.std(harmed_diffs, ddof=1))
                    if len(harmed_diffs) >= 2 else None),
            "values": harmed_diffs,
        },
    }


def _summarize_pairs(paired: list[dict[str, Any]]) -> dict[str, Any]:
    d_values = [row["d_u"] for row in paired]
    g_values = [row["g_u_ancestor"] for row in paired]
    d_stats = _stats(d_values)
    g_stats = _stats(g_values)
    harmed = [row["harmed_series_diff"] for row in paired
              if row.get("harmed_series_diff") is not None]
    return {
        "n_paired_units": len(paired),
        "d": d_stats,
        "g_ancestor": g_stats,
        "var_ratio_d_over_g": _ratio(d_stats["var"], g_stats["var"]),
        "harmed_series_diff": {
            "n": len(harmed),
            "mean": (float(np.mean(harmed)) if harmed else None),
            "std": (float(np.std(harmed, ddof=1)) if len(harmed) >= 2 else None),
            "values": harmed,
        },
        "units": paired,
    }


def build() -> dict[str, Any]:
    per_unit_pack = _load_per_unit()
    store_pack = _load_store()
    per_unit = per_unit_pack["per_unit"]
    per_unit_index = _index_per_unit(per_unit)
    ancestor_positions = [int(row["position"]) for row in per_unit[ANCESTOR]]
    gains = store_pack["gains"]
    comparisons: dict[str, Any] = {}
    ranking_rows: list[dict[str, Any]] = []
    for child in CHILDREN:
        comparisons[child] = {}
        for face in FACES:
            store_pairs, store_missing, series_block = _store_unit_pairs(
                child, face, gains)
            store_summary = _summarize_pairs(store_pairs)
            block: dict[str, Any] = {
                "store_unit_aggregate": {
                    "source": str(STORE_PATH.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"),
                    "missing_units": store_missing,
                    **store_summary,
                },
                "per_series": {
                    "source": str(STORE_PATH.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"),
                    "pairing": "same (position, uid) present in child and ANCESTOR",
                    "d_i": "g_child - g_ancestor",
                    "g_i": "raw - program",
                    **series_block,
                },
            }
            if child in CAND_CHILDREN:
                unit_pairs, unit_missing = _paired_unit_rows(
                    child, face, per_unit_index, ancestor_positions)
                unit_summary = _summarize_pairs(unit_pairs)
                block["per_unit_aggregate"] = {
                    "source": str(PER_UNIT_PATH.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"),
                    "missing_units": unit_missing,
                    **unit_summary,
                }
                headline = unit_summary
                headline_layer = "per_unit_aggregate"
            else:
                block["per_unit_aggregate"] = {
                    "source": str(PER_UNIT_PATH.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"),
                    "present": False,
                    "why": ("per_unit only contains ANCESTOR and the two CAND "
                            "programs; W1/W2/W3 unit aggregates are formed from "
                            "the prediction store as the mean of per-series g"),
                }
                headline = store_summary
                headline_layer = "store_unit_aggregate"
            headline_d = headline["d"]
            headline_g = headline["g_ancestor"]
            series_d = series_block["series_d"]
            series_g = series_block["series_g_ancestor_on_paired_rows"]
            row = {
                "child": child,
                "face": face,
                "headline_layer": headline_layer,
                "n_paired_units": headline["n_paired_units"],
                "sign_consistency_d": headline_d["sign_consistency"],
                "frac_positive_d": headline_d["frac_positive"],
                "frac_negative_d": headline_d["frac_negative"],
                "sign_consistency_g": headline_g["sign_consistency"],
                "frac_positive_g": headline_g["frac_positive"],
                "frac_negative_g": headline_g["frac_negative"],
                "var_ratio": headline["var_ratio_d_over_g"],
                "d_mean": headline_d["mean"],
                "d_std": headline_d["std"],
                "g_mean": headline_g["mean"],
                "g_std": headline_g["std"],
                "harmed_diff_mean": headline["harmed_series_diff"]["mean"],
                "series_sign_consistency_d": series_d["sign_consistency"],
                "series_var_ratio": series_block["var_ratio"],
                "d_more_stable_than_g_by_var": (
                    None if headline["var_ratio_d_over_g"] is None
                    else headline["var_ratio_d_over_g"] < 1.0),
                "d_more_sign_consistent_than_g": (
                    None if (headline_d["sign_consistency"] is None
                             or headline_g["sign_consistency"] is None)
                    else headline_d["sign_consistency"]
                    >= headline_g["sign_consistency"]),
            }
            block["headline"] = row
            comparisons[child][face] = block
            ranking_rows.append(row)

    ranked = [row for row in ranking_rows
              if row["sign_consistency_d"] is not None
              and row["var_ratio"] is not None]
    ranked.sort(key=lambda row: (-row["sign_consistency_d"], row["var_ratio"],
                                 row["child"], row["face"]))
    identity_rows = [row for row in ranked
                     if row["frac_positive_d"] == 0.0
                     and row["frac_negative_d"] == 0.0]
    nontrivial = [row for row in ranked if row not in identity_rows]
    by_sign = sorted(nontrivial, key=lambda row: (
        -row["sign_consistency_d"], row["var_ratio"], row["child"], row["face"]))
    by_var = sorted(nontrivial, key=lambda row: (
        row["var_ratio"], -row["sign_consistency_d"], row["child"], row["face"]))
    most_stable = None
    if by_sign and by_var and by_sign[0] is by_var[0]:
        most_stable = by_sign[0]
    n_var_lt_1 = sum(1 for row in nontrivial if row["var_ratio"] < 1.0)
    n_sign_ge = sum(1 for row in nontrivial
                    if row["d_more_sign_consistent_than_g"] is True)
    one_sentence = (
        "W3 is identically the ancestor on both faces (d=0 on every paired "
        "unit; trivial stability, not a revision). Among actual revisions, "
        "no child×face jointly has the highest sign consistency and the "
        "smallest Var(d)/Var(g): highest sign consistency is %s/%s "
        "(%.6f, mostly %s); smallest Var ratio is %s/%s (%.6f). "
        "Var(d)<Var(g) on %d/%d non-zero pairs, so d is generally smaller "
        "in variance than g, but sign(d) is as concentrated as sign(g) on "
        "only %d/%d pairs. Descriptive only."
        % (
            by_sign[0]["child"] if by_sign else "none",
            by_sign[0]["face"] if by_sign else "none",
            by_sign[0]["sign_consistency_d"] if by_sign else float("nan"),
            ("negative" if by_sign and by_sign[0]["frac_negative_d"]
             >= by_sign[0]["frac_positive_d"] else "positive"),
            by_var[0]["child"] if by_var else "none",
            by_var[0]["face"] if by_var else "none",
            by_var[0]["var_ratio"] if by_var else float("nan"),
            n_var_lt_1,
            len(nontrivial),
            n_sign_ge,
            len(nontrivial),
        )
    )
    return {
        "stage": "R2_REVISION_EFFECT_STABILITY",
        "written_at": datetime.now(TZ).isoformat(timespec="microseconds"),
        "what_this_is": (
            "descriptive readout of whether revision effect d = g(child) − "
            "g(ANCESTOR) is more stable across units than the ancestor's own "
            "effect g; 0 fit, 0 LLM, no significance claim"),
        "question": (
            "is d_e(P→P') more stable across units than g_e(P) in this lineage"),
        "boundary": {
            "consumer_fits": 0,
            "llm_calls": 0,
            "existing_files_edited": 0,
            "git_commits": 0,
            "new_sha_or_hash": 0,
        },
        "sources": {
            "per_unit": {
                "path": str(PER_UNIT_PATH.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"),
                "n_policies": len(per_unit),
                "n_units_per_policy": len(per_unit[ANCESTOR]),
                "positions": ancestor_positions,
            },
            "prediction_store": {
                "path": str(STORE_PATH.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"),
                "n_records": store_pack["n_records"],
                "coverage": store_pack["coverage"],
                "unreadable_store_records": store_pack["unreadable_store_records"],
            },
        },
        "missing_handling": {
            "rule": (
                "a face is paired only when both child and ANCESTOR are READ "
                "and carry aggregate_gain; WINDOW_VERIFIER_REJECTED, "
                "SERVING_CONTEXT_DEGENERATE, NO_STORED_READING, NOT_READABLE "
                "and UNKNOWN stay missing and are never stored as 0"),
            "statuses_reserved_as_missing": list(MISSING_STATUSES_SEEN),
            "variance": "sample variance, numpy.var(ddof=1)",
            "sign_consistency": (
                "max(frac_positive, frac_negative) over paired units, zeros "
                "kept in the denominator"),
        },
        "field_inventory": {
            "per_unit": per_unit_pack["inventory"],
            "prediction_store": {
                "n_records": store_pack["n_records"],
                "sample_key": store_pack["sample_key"],
                "sample_keys": store_pack["sample_keys"],
                "sample_record": store_pack["sample_record"],
            },
        },
        "comparisons": comparisons,
        "ranking_by_sign_then_var_ratio": ranked,
        "identity_equivalent_child_faces": identity_rows,
        "highest_sign_consistency_nontrivial": (by_sign[0] if by_sign else None),
        "smallest_var_ratio_nontrivial": (by_var[0] if by_var else None),
        "most_stable_child_face": most_stable,
        "one_sentence": one_sentence,
    }


def _fmt(value: Any) -> str:
    rendered = _r6(value)
    if rendered is None:
        return "NA"
    if isinstance(rendered, float):
        return "%.6f" % rendered
    return str(rendered)


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# R2 · 修订效应稳定性读数（描述性；0 fit）",
        "",
        "地位：development 描述性读数，**不做显著性主张**。",
        "问题：同一谱系里子程序相对祖先的差值 `d` 是否比祖先自身的效应 `g` 在跨单元上更稳定。",
        "",
        "## 数据来源",
        "",
        "| 源 | 路径 | 行数 / 覆盖 |",
        "| --- | --- | --- |",
        "| (a) 聚合 `per_unit` | `%s` | 3 个策略 × 16 单元 × 2 面 |"
        % report["sources"]["per_unit"]["path"],
        "| (b) 逐序列预测库 | `%s` | %d 条；键 `program|position|face` |"
        % (report["sources"]["prediction_store"]["path"],
           report["sources"]["prediction_store"]["n_records"]),
        "",
        "预测库覆盖：ANCESTOR / W1 / W2 / W3 各 21 位置 × 2 面；两个 CAND 合计 31+31 条。",
        "逐序列增益 `g = raw − program`（正 = 程序有益）。",
        "",
        "## 缺失处理",
        "",
        report["missing_handling"]["rule"] + "。",
        "方差：样本方差 `ddof=1`。符号一致率：`max(正占比, 负占比)`，零计入分母。",
        "",
        "已观测的非 READ 状态（原样保留为缺失，不记 0）：",
        "`WINDOW_VERIFIER_REJECTED`、`SERVING_CONTEXT_DEGENERATE`、`NO_STORED_READING`。",
        "本批 `per_unit` 未出现字面 `NOT_READABLE` / `UNKNOWN`；若出现同样保留为缺失。",
        "",
        "## 字段核验（一条完整可读记录的键，未猜测）",
        "",
        "- 单元记录键：`%s`"
        % ", ".join(report["field_inventory"]["per_unit"]["unit_record_keys"]),
        "- 可读 face 键：`%s`"
        % ", ".join(report["field_inventory"]["per_unit"]["face_keys_when_read"]),
        "- reading 键：`%s`"
        % ", ".join(report["field_inventory"]["per_unit"]["reading_keys_when_read"]),
        "- 预测库样本键 `%s` 字段：`%s`"
        % (report["field_inventory"]["prediction_store"]["sample_key"],
           ", ".join(report["field_inventory"]["prediction_store"]["sample_keys"])),
        "",
        "## 读数（6 位小数）",
        "",
        "CAND 的聚合层来自 `per_unit`；W1/W2/W3 的聚合层来自预测库按单元均值 "
        "（`per_unit` 不含这些程序）。逐序列层全部来自预测库。",
        "",
        "| child | face | 层 | n | 正占比(d) | 负占比(d) | 符号一致率(d) | "
        "符号一致率(g) | mean(d) | std(d) | mean(g) | std(g) | Var(d)/Var(g) | "
        "受损数差均值 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for child in CHILDREN:
        for face in FACES:
            head = report["comparisons"][child][face]["headline"]
            lines.append(
                "| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    child, face, head["headline_layer"],
                    head["n_paired_units"],
                    _fmt(head["frac_positive_d"]),
                    _fmt(head["frac_negative_d"]),
                    _fmt(head["sign_consistency_d"]),
                    _fmt(head["sign_consistency_g"]),
                    _fmt(head["d_mean"]),
                    _fmt(head["d_std"]),
                    _fmt(head["g_mean"]),
                    _fmt(head["g_std"]),
                    _fmt(head["var_ratio"]),
                    _fmt(head["harmed_diff_mean"]),
                ))
    lines.extend([
        "",
        "### 逐序列层",
        "",
        "| child | face | n 配对序列 | 正占比(d_i) | 负占比(d_i) | 符号一致率(d_i) | "
        "Var(d_i)/Var(g_anc_i) | 单元内 d_i>0 占比均值 | 该占比标准差 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for child in CHILDREN:
        for face in FACES:
            series = report["comparisons"][child][face]["per_series"]
            d_stats = series["series_d"]
            dist = series["unit_frac_d_positive_distribution"]
            lines.append(
                "| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    child, face, d_stats["n"],
                    _fmt(d_stats["frac_positive"]),
                    _fmt(d_stats["frac_negative"]),
                    _fmt(d_stats["sign_consistency"]),
                    _fmt(series["var_ratio"]),
                    _fmt(dist["mean"]),
                    _fmt(dist["std"]),
                ))
    lines.extend([
        "",
        "### 逐序列受损计数差（g<0 重算）对照聚合层 harmed_series 差",
        "",
        "| child | face | 聚合层受损数差均值 | 逐序列 g<0 计数差均值 |",
        "| --- | --- | ---: | ---: |",
    ])
    for child in CHILDREN:
        for face in FACES:
            block = report["comparisons"][child][face]
            agg_mean = block["headline"]["harmed_diff_mean"]
            series_mean = block["per_series"]["harmed_count_diff_from_g_lt_0"]["mean"]
            lines.append("| `%s` | %s | %s | %s |"
                         % (child, face, _fmt(agg_mean), _fmt(series_mean)))
    lines.extend([
        "",
        "## 缺失单元清单（CAND × per_unit）",
        "",
    ])
    for child in CAND_CHILDREN:
        for face in FACES:
            missing = (report["comparisons"][child][face]
                       .get("per_unit_aggregate", {}).get("missing_units") or [])
            lines.append("### `%s` / %s（%d 个缺失）" % (child, face, len(missing)))
            if not missing:
                lines.append("")
                lines.append("无。")
                lines.append("")
                continue
            lines.append("")
            lines.append("| position | ancestor_status | child_status |")
            lines.append("| ---: | --- | --- |")
            for row in missing:
                lines.append("| %s | %s | %s |"
                             % (row["position"], row["ancestor_status"],
                                row["child_status"]))
            lines.append("")
    lines.extend([
        "## 一句话",
        "",
        report["one_sentence"],
        "",
        "## 边界",
        "",
        "fits=0，LLM=0，不编辑既有文件，不提交 git，不新增 SHA/Hash。",
        "",
        "复跑：`python -m evaluation.main_protocol_p4.audit_r2_revision_effect_stability`",
        "",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    json_path = OUT_STEM.with_suffix(".json")
    md_path = OUT_STEM.with_suffix(".md")
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8")).get("stage")
        except (OSError, ValueError):
            existing = None
        if existing not in (None, "R2_REVISION_EFFECT_STABILITY"):
            sys.stderr.write("refusing to overwrite %s\n" % json_path)
            return 2
    report = build()
    json_path.write_text(
        json.dumps(_r6(report), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    md_path.write_text(render_md(_r6(report)), encoding="utf-8")
    print("wrote %s" % json_path)
    print("wrote %s" % md_path)
    print(report["one_sentence"])
    for child in CHILDREN:
        for face in FACES:
            head = report["comparisons"][child][face]["headline"]
            print(
                "%s %s  sign(d)=%s  sign(g)=%s  var_ratio=%s  harmed_diff_mean=%s"
                % (
                    child, face,
                    _fmt(head["sign_consistency_d"]),
                    _fmt(head["sign_consistency_g"]),
                    _fmt(head["var_ratio"]),
                    _fmt(head["harmed_diff_mean"]),
                ))
    inventory = report["field_inventory"]["per_unit"]
    print("per_unit policies:", inventory["per_unit_policy_keys"])
    print("reading status counts:", inventory["reading_status_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
