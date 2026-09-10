"""R4A-b: is the per-series gain a stable property at all?

R4A closed two doors at once: the condition that decides whether a cleaning
program helps a series is not in the series' own visible past (best LODO mean
0.63-0.68, every cell ``PATTERN_WEAK``) and not in its own future either
(``NO_FUTURE_ADVANTAGE_DETECTED``, ORACLE AUC <= 0.606).  That leaves a prior
question, which this package asks instead of asking for a better feature: is
there anything to condition *on*?  A per-series gain that does not reproduce
when the same series is served 48 steps later, and whose variance sits neither
on the series nor on the cohort, is not a hidden pattern -- it is noise, and the
whole identifiability programme would be aimed at the wrong layer.

Three readings carry the answer.  Cross-face persistence asks whether the same
(position, uid) keeps its gain when the origin moves by 48.  A variance
decomposition splits each face's gain into a position (cohort x origin) main
effect, a uid main effect taken inside each cohort, and a residual.  A
unit-level AUC asks how far one could get with the cohort's own mean gain as a
condition -- an in-sample ceiling, never deployable, quoted only to bound what a
cohort-layer condition could ever buy.

Nothing here fits anything.  Per-series readings come from the m_r0k prediction
store, produced by an authorised package; this script only indexes it.  Row
construction, drops and the ``local_robust_z_peak`` card are imported from
``audit_r4a_pattern_identifiability`` so the two packages cannot drift apart.

Verdict vocabulary and its three thresholds are frozen by the R4A-b brief and
are not tuned here.

Run: ``python -m evaluation.main_protocol_p4.audit_r4a_persistence_and_variance``
0 Consumer fits, 0 LLM calls, 0 held-out reads.
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
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import run_hec1 as hec1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE = r4a.STORE
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/r4a_persistence_and_variance.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/r4a_persistence_and_variance.md"

CONTEXT = r4a.CONTEXT
DELAYED_OFFSET = r4a.DELAYED_OFFSET
SEVERE_HARM = r4a.SEVERE_HARM
MAX_ALLOWED_INDEX = r4a.MAX_ALLOWED_INDEX
OBSERVABLE_NUMERIC = r4a.OBSERVABLE_NUMERIC
FACES = r4a.FACES
TARGETS = r4a.TARGETS

# W3_outlier_mad_then_pmc is numerically identical to ANCESTOR in this store
# (R4A section 5.2).  It is verified below and then skipped, so that an alias is
# never counted as a second independent reading.
MAIN_PROGRAMS = ("ANCESTOR", "W1_hampel_filter", "W2_pmc_then_outlier_mad")
ALIAS_PROGRAM = "W3_outlier_mad_then_pmc"
REFERENCE_PROGRAMS = r4a.REFERENCE_PROGRAMS

# --- frozen verdict thresholds (R4A-b brief; set before any number was seen) ---
UID_SHARE_SERIES = 0.30
SPEARMAN_SERIES = 0.50
POSITION_SHARE_COHORT = 0.50
SPEARMAN_NOISE = 0.30
# Reading aid for the z_peak strength-vs-direction contrast, not part of the
# frozen vocabulary: how much larger |rho| against |g| must be to be called
# "clearly higher" than |rho| against g.
ZPEAK_MARGIN = 0.10

Z_PEAK = "local_robust_z_peak"


# ---------------------------------------------------------------------------
# 1. rank and moment statistics (no fits, no new dependency)
# ---------------------------------------------------------------------------


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return None
    sx, sy = float(x.std()), float(y.std())
    if sx <= 0.0 or sy <= 0.0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return None
    return pearson(r4a._average_ranks(x), r4a._average_ranks(y))


def quartiles(values: Sequence[float]) -> dict[str, float | None]:
    finite = np.array([v for v in values if v is not None and math.isfinite(v)],
                      dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "q25": None, "median": None, "q75": None,
                "min": None, "max": None, "mean": None}
    return {
        "n": int(finite.size),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    """Chance-corrected agreement of two binary labels on the same units."""
    if a.size == 0:
        return {"kappa": None, "observed_agreement": None,
                "expected_agreement": None, "rate_a": None, "rate_b": None}
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    return {
        "kappa": None if pe >= 1.0 else float((po - pe) / (1.0 - pe)),
        "observed_agreement": po,
        "expected_agreement": pe,
        "rate_a": pa,
        "rate_b": pb,
    }


def null_share(n_levels: int, n_rows: int) -> float | None:
    """What a between-group share looks like when nothing is going on.

    Under exchangeable noise, E[SS_between / SS_total] = (k-1)/(n-1) for k
    groups.  A share must be read against this, because a factor with many
    levels and few rows per level -- the thin cohorts here have only two or
    three positions per uid -- absorbs a large share of the variance simply by
    having enough degrees of freedom to do so.
    """
    if n_rows < 2 or n_levels < 1:
        return None
    return float((n_levels - 1) / (n_rows - 1))


def one_way_share(values: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """SS_between / SS_total for a single grouping factor."""
    if values.size < 2:
        return {"share": None, "ss_between": None, "ss_total": None,
                "n_groups": None, "n_rows": int(values.size)}
    mu = float(values.mean())
    ss_total = float(((values - mu) ** 2).sum())
    ss_between = 0.0
    levels = np.unique(groups)
    for level in levels:
        mask = groups == level
        ss_between += float(mask.sum()) * (float(values[mask].mean()) - mu) ** 2
    share = None if ss_total <= 0.0 else float(ss_between / ss_total)
    expected = null_share(int(levels.size), int(values.size))
    return {
        "share": share,
        "share_null_expectation": expected,
        "share_excess_over_null": (None if share is None or expected is None
                                   else float(share - expected)),
        "ss_between": ss_between,
        "ss_total": ss_total,
        "n_groups": int(levels.size),
        "n_rows": int(values.size),
    }


def additive_two_way(values: np.ndarray, uids: np.ndarray,
                     positions: np.ndarray) -> dict[str, Any]:
    """Marginal uid and position shares plus the additive-model residual.

    The design inside one cohort is the same 20 uids repeated at every position,
    so for the main programs this is balanced and the three shares sum to one.
    They are reported separately anyway, together with their sum, because a
    dropped unit (the two CAND reference rows) makes the design unbalanced and
    the sum then drifts off 1.0 -- silently renormalising would hide that.
    """
    if values.size < 4:
        return {"uid_share": None, "position_share": None,
                "residual_share": None, "shares_sum": None,
                "n_rows": int(values.size), "n_uids": None, "n_positions": None}
    mu = float(values.mean())
    ss_total = float(((values - mu) ** 2).sum())
    uid_levels, position_levels = np.unique(uids), np.unique(positions)
    uid_mean = {level: float(values[uids == level].mean()) for level in uid_levels}
    position_mean = {level: float(values[positions == level].mean())
                     for level in position_levels}
    ss_uid = sum(float((uids == level).sum()) * (uid_mean[level] - mu) ** 2
                 for level in uid_levels)
    ss_position = sum(float((positions == level).sum())
                      * (position_mean[level] - mu) ** 2
                      for level in position_levels)
    residual = np.array(
        [value - uid_mean[uid] - position_mean[position] + mu
         for value, uid, position in zip(values, uids, positions)],
        dtype=np.float64)
    ss_residual = float((residual ** 2).sum())
    if ss_total <= 0.0:
        return {"uid_share": None, "position_share": None,
                "residual_share": None, "shares_sum": None,
                "n_rows": int(values.size), "n_uids": int(uid_levels.size),
                "n_positions": int(position_levels.size)}
    shares = (ss_uid / ss_total, ss_position / ss_total, ss_residual / ss_total)
    uid_null = null_share(int(uid_levels.size), int(values.size))
    position_null = null_share(int(position_levels.size), int(values.size))
    return {
        "uid_share": float(shares[0]),
        "position_share": float(shares[1]),
        "residual_share": float(shares[2]),
        "shares_sum": float(sum(shares)),
        "uid_share_null_expectation": uid_null,
        "position_share_null_expectation": position_null,
        "uid_share_excess_over_null": (None if uid_null is None
                                       else float(shares[0] - uid_null)),
        "position_share_excess_over_null": (None if position_null is None
                                            else float(shares[1] - position_null)),
        "ss_total": ss_total,
        "n_rows": int(values.size),
        "n_uids": int(uid_levels.size),
        "n_positions": int(position_levels.size),
    }


# ---------------------------------------------------------------------------
# 2. the five readings
# ---------------------------------------------------------------------------


def cross_face_persistence(rows: Sequence[Mapping[str, Any]],
                           program: str) -> dict[str, Any]:
    """Reading 1.  Does a (position, uid) keep its gain 48 steps later?"""
    paired: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["program"] != program:
            continue
        paired.setdefault(row["position"], {}).setdefault(row["uid"], {})[
            row["face"]] = row["g"]
    per_position: dict[str, Any] = {}
    pooled_s, pooled_d = [], []
    for position in sorted(paired):
        gs, gd = [], []
        for uid in sorted(paired[position]):
            faces = paired[position][uid]
            if "support_face" in faces and "delayed_face" in faces:
                gs.append(faces["support_face"])
                gd.append(faces["delayed_face"])
        if not gs:
            continue
        gs_a, gd_a = np.array(gs, dtype=np.float64), np.array(gd, dtype=np.float64)
        pooled_s.extend(gs)
        pooled_d.extend(gd)
        per_position[str(position)] = {
            "n_pairs": int(gs_a.size),
            "spearman": spearman(gs_a, gd_a),
            "pearson": pearson(gs_a, gd_a),
            "mean_g_support": float(gs_a.mean()),
            "mean_g_delayed": float(gd_a.mean()),
        }
    support = np.array(pooled_s, dtype=np.float64)
    delayed = np.array(pooled_d, dtype=np.float64)
    severe_s = (-support > SEVERE_HARM).astype(np.float64)
    severe_d = (-delayed > SEVERE_HARM).astype(np.float64)
    helped_s = (support > 0.0).astype(np.float64)
    helped_d = (delayed > 0.0).astype(np.float64)
    carried = (float(severe_d[severe_s == 1].mean())
               if float(severe_s.sum()) > 0 else None)
    base = float(severe_d.mean()) if severe_d.size else None
    return {
        "n_positions_paired": len(per_position),
        "n_pairs": int(support.size),
        "per_position": per_position,
        "spearman_by_position": quartiles(
            [cell["spearman"] for cell in per_position.values()]),
        "pearson_by_position": quartiles(
            [cell["pearson"] for cell in per_position.values()]),
        "pooled_spearman": spearman(support, delayed),
        "pooled_pearson": pearson(support, delayed),
        "severe_harm_carry_over": {
            "n_severe_support": int(severe_s.sum()),
            "p_delayed_severe_given_support_severe": carried,
            "delayed_base_rate": base,
            "lift": (None if carried is None or not base else
                     float(carried / base)),
            "excess": None if carried is None or base is None
            else float(carried - base),
        },
        "helped_kappa": cohens_kappa(helped_s, helped_d),
        "severe_harm_kappa": cohens_kappa(severe_s, severe_d),
    }


def variance_decomposition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reading 2.  Position main effect, uid main effect inside cohorts, rest."""
    values = np.array([row["g"] for row in rows], dtype=np.float64)
    positions = np.array([row["position"] for row in rows])
    uids = np.array([row["uid"] for row in rows])
    cohorts = np.array([row["block"] for row in rows])
    position_only = one_way_share(values, positions)
    per_cohort: dict[str, Any] = {}
    for cohort in sorted(set(cohorts.tolist())):
        mask = cohorts == cohort
        per_cohort[cohort] = additive_two_way(values[mask], uids[mask],
                                              positions[mask])
    weights: list[float] = []
    columns: dict[str, list[float]] = {
        "uid_share": [], "position_share": [], "residual_share": [],
        "uid_share_null_expectation": [], "position_share_null_expectation": [],
    }
    for cell in per_cohort.values():
        if cell.get("uid_share") is None:
            continue
        weights.append(float(cell["n_rows"]))
        for name in columns:
            columns[name].append(cell[name])
    total_weight = float(sum(weights))

    def weighted(items: Sequence[float]) -> float | None:
        if not items or total_weight <= 0.0:
            return None
        return float(sum(w * v for w, v in zip(weights, items)) / total_weight)

    uid_weighted = weighted(columns["uid_share"])
    uid_null_weighted = weighted(columns["uid_share_null_expectation"])
    return {
        "position_share_one_way": position_only["share"],
        "position_one_way_detail": position_only,
        "within_cohort": per_cohort,
        "uid_share_weighted": uid_weighted,
        "within_cohort_position_share_weighted": weighted(columns["position_share"]),
        "residual_share_weighted": weighted(columns["residual_share"]),
        "uid_share_null_expectation_weighted": uid_null_weighted,
        "uid_share_excess_over_null_weighted": (
            None if uid_weighted is None or uid_null_weighted is None
            else float(uid_weighted - uid_null_weighted)),
        "position_share_null_expectation_weighted":
            weighted(columns["position_share_null_expectation"]),
        "weighting": "rows per cohort",
        "cohort_row_counts": {cohort: cell["n_rows"]
                              for cohort, cell in per_cohort.items()},
    }


def zpeak_strength_vs_direction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reading 3.  Does z_peak track how hard the program bites, or which way?"""
    x = r4a._matrix(rows, Z_PEAK)
    g = np.array([row["g"] for row in rows], dtype=np.float64)
    severe = np.array([row["severe_harm"] for row in rows], dtype=np.float64)
    helped = np.array([row["helped"] for row in rows], dtype=np.float64)
    rho_abs = spearman(x, np.abs(g))
    rho_signed = spearman(x, g)
    auc_severe = r4a.auc(x, severe)
    auc_helped = r4a.auc(x, helped)
    strength_only = (
        rho_abs is not None and rho_signed is not None
        and abs(rho_abs) >= abs(rho_signed) + ZPEAK_MARGIN)
    return {
        "spearman_with_abs_g": rho_abs,
        "spearman_with_g": rho_signed,
        "pearson_with_abs_g": pearson(x, np.abs(g)),
        "pearson_with_g": pearson(x, g),
        "pooled_auc_severe_harm_raw": auc_severe,
        "pooled_auc_helped_raw": auc_helped,
        "severe_harm_direction": (
            None if auc_severe is None else
            "higher z_peak -> more severe harm" if auc_severe > 0.5
            else "higher z_peak -> less severe harm" if auc_severe < 0.5
            else "no direction"),
        "reads_strength_not_direction": bool(strength_only),
        "margin_rule": ("|rho(z_peak, |g|)| >= |rho(z_peak, g)| + %.2f"
                        % ZPEAK_MARGIN),
    }


def unit_level_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reading 4.  The cohort's own mean gain used as a condition, in-sample.

    The score is constant inside a unit, so this AUC is an in-sample ceiling on
    what any cohort-layer condition could buy, computed with the answer already
    in hand.  It is not deployable and is not comparable to a LODO reading.
    """
    unit_mean: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        unit_mean.setdefault((row["position"], row["face"]), []).append(row["g"])
    means = {unit: float(np.mean(values)) for unit, values in unit_mean.items()}
    score = np.array([means[(row["position"], row["face"])] for row in rows],
                     dtype=np.float64)
    out: dict[str, Any] = {
        "n_units": len(means),
        "unit_mean_g_min": float(min(means.values())) if means else None,
        "unit_mean_g_max": float(max(means.values())) if means else None,
    }
    for target in TARGETS:
        y = np.array([row[target] for row in rows], dtype=np.float64)
        raw = r4a.auc(score, y)
        out[target] = {
            "auc_raw": raw,
            "auc_oriented": None if raw is None else float(max(raw, 1.0 - raw)),
        }
    return out


def best_visible_pooled(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled AUC of the deployment vocabulary, same orientation convention."""
    out: dict[str, Any] = {}
    for target in TARGETS:
        y = np.array([row[target] for row in rows], dtype=np.float64)
        best_name, best_value, best_raw = None, None, None
        per_feature = {}
        for name in OBSERVABLE_NUMERIC:
            raw = r4a.auc(r4a._matrix(rows, name), y)
            per_feature[name] = raw
            if raw is None:
                continue
            oriented = max(raw, 1.0 - raw)
            if best_value is None or oriented > best_value:
                best_name, best_value, best_raw = name, oriented, raw
        out[target] = {
            "feature": best_name,
            "auc_oriented": best_value,
            "auc_raw": best_raw,
            "per_feature_auc_raw": per_feature,
        }
    return out


def cell_verdict(spearman_median: float | None, position_share: float | None,
                 uid_share: float | None) -> tuple[str, str]:
    """The frozen four-way vocabulary, checked in the brief's order."""
    if uid_share is None or position_share is None:
        return "NOT_READABLE", "a share is not computable"
    sm = spearman_median
    if uid_share >= UID_SHARE_SERIES and sm is not None and sm >= SPEARMAN_SERIES:
        return "SERIES_LEVEL_CONDITION", (
            "uid share %.6f >= %.2f and cross-face Spearman median %.6f >= %.2f"
            % (uid_share, UID_SHARE_SERIES, sm, SPEARMAN_SERIES))
    if position_share >= POSITION_SHARE_COHORT and uid_share < UID_SHARE_SERIES:
        return "COHORT_LEVEL_CONDITION", (
            "position share %.6f >= %.2f and uid share %.6f < %.2f"
            % (position_share, POSITION_SHARE_COHORT, uid_share,
               UID_SHARE_SERIES))
    if (sm is not None and sm < SPEARMAN_NOISE
            and uid_share < UID_SHARE_SERIES
            and position_share < POSITION_SHARE_COHORT):
        return "MOSTLY_NOISE", (
            "cross-face Spearman median %.6f < %.2f, uid share %.6f < %.2f, "
            "position share %.6f < %.2f"
            % (sm, SPEARMAN_NOISE, uid_share, UID_SHARE_SERIES,
               position_share, POSITION_SHARE_COHORT))
    return "MIXED", (
        "none of the three frozen patterns holds (Spearman median %s, position "
        "share %.6f, uid share %.6f)"
        % ("n/a" if sm is None else "%.6f" % sm, position_share, uid_share))


# ---------------------------------------------------------------------------
# 3. main
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    store = json.loads(STORE.read_text(encoding="utf-8"))
    forward = contract.ordering("forward")
    readable = hec1.readable_uids()

    variant = preflight.load_variant()
    nan_count = int(sum(int(np.isnan(series).sum()) for series in variant.values()))
    if nan_count <= 0:
        raise AssertionError(
            "the loaded variant has no NaN; this is the filled cache, not "
            "EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING")
    reader = r4a.Reader(variant)

    positions = sorted({int(entry["position"]) for entry in store.values()})
    position_block = {p: str(forward[p]["block"]) for p in positions}

    geometry_checks = []
    for key, entry in store.items():
        expected = int(forward[int(entry["position"])]["origin"]) + (
            DELAYED_OFFSET if str(entry["face"]) == "delayed_face" else 0)
        if int(entry["origin"]) != expected:
            geometry_checks.append({"key": key, "origin": int(entry["origin"]),
                                    "expected": expected})

    # The alias check is re-run here rather than trusted from R4A, because the
    # decision to drop the W3 rows depends on it.
    alias_identical = alias_compared = 0
    alias_max_abs_diff = 0.0
    for entry in store.values():
        if entry["program"] != ALIAS_PROGRAM or not entry.get("verifier_passed"):
            continue
        twin = store.get("ANCESTOR|%d|%s" % (int(entry["position"]), entry["face"]))
        if twin is None:
            continue
        alias_compared += 1
        mine = np.asarray(entry["program_per_view"], dtype=np.float64)
        theirs = np.asarray(twin["program_per_view"], dtype=np.float64)
        alias_max_abs_diff = max(alias_max_abs_diff,
                                 float(np.max(np.abs(mine - theirs))))
        if np.array_equal(mine, theirs):
            alias_identical += 1

    # --- feature cards, one per (origin, uid); only the pre-origin context ---
    wanted: dict[int, list[str]] = {}
    for entry in store.values():
        if entry["program"] not in MAIN_PROGRAMS + (ALIAS_PROGRAM,) + \
                REFERENCE_PROGRAMS:
            continue
        wanted.setdefault(int(entry["origin"]), [])
        for uid in entry["eval_uids"]:
            if str(uid) not in wanted[int(entry["origin"])]:
                wanted[int(entry["origin"])].append(str(uid))
    cards: dict[tuple[int, str], dict[str, float]] = {}
    for origin, uids in wanted.items():
        matrix, names = targeting.series_features(variant, uids, origin)
        for uid in uids:
            reader.account_external(uid, origin)
        for index, uid in enumerate(uids):
            cards[(origin, uid)] = {
                name: float(matrix[index][column])
                for column, name in enumerate(names)
                if name in OBSERVABLE_NUMERIC
            }

    rows, skipped = r4a.build_rows(store, cards, position_block)
    scoped = [row for row in rows if row["program"] != ALIAS_PROGRAM]

    table: dict[str, Any] = {}
    persistence: dict[str, Any] = {}
    for program in MAIN_PROGRAMS + REFERENCE_PROGRAMS:
        program_rows = [row for row in scoped if row["program"] == program]
        if not program_rows:
            continue
        persistence[program] = cross_face_persistence(program_rows, program)
        spearman_median = persistence[program]["spearman_by_position"]["median"]
        for face in FACES:
            subset = [row for row in program_rows if row["face"] == face]
            if not subset:
                continue
            g = np.array([row["g"] for row in subset], dtype=np.float64)
            decomposition = variance_decomposition(subset)
            verdict, why = cell_verdict(
                spearman_median,
                decomposition["position_share_one_way"],
                decomposition["uid_share_weighted"])
            table["%s|%s" % (program, face)] = {
                "role": "main" if program in MAIN_PROGRAMS else "reference",
                "n_rows": len(subset),
                "n_unique_uids": len({row["uid"] for row in subset}),
                "n_positions": len({row["position"] for row in subset}),
                "cohorts": sorted({row["block"] for row in subset}),
                "b1": {
                    "helped_rate": float(np.mean([row["helped"] for row in subset])),
                    "severe_harm_rate":
                        float(np.mean([row["severe_harm"] for row in subset])),
                    "mean_g": float(g.mean()),
                    "sd_g": float(g.std(ddof=0)),
                },
                "cross_face_spearman_median": spearman_median,
                "variance": decomposition,
                "zpeak": zpeak_strength_vs_direction(subset),
                "unit_level_condition": unit_level_condition(subset),
                "best_visible_series_level": best_visible_pooled(subset),
                "verdict": verdict,
                "verdict_reason": why,
                "verdict_inputs": {
                    "cross_face_spearman_median": spearman_median,
                    "position_share_one_way":
                        decomposition["position_share_one_way"],
                    "uid_share_weighted": decomposition["uid_share_weighted"],
                },
            }

    verdict_counts: dict[str, int] = {}
    for cell in table.values():
        if cell["role"] == "main":
            verdict_counts[cell["verdict"]] = verdict_counts.get(
                cell["verdict"], 0) + 1

    payload = {
        "task": "R4A-b persistence and variance of the per-series gain",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_class": "MECHANISM / NEGATIVE-capable; not a capability claim",
        "question": (
            "R4A showed the condition is neither in the series' visible past nor "
            "in its own future.  Is the per-series gain a stable property at "
            "all -- a series-layer property, a cohort x origin property, or "
            "mostly noise?"),
        "boundary": {
            "consumer_fits": 0,
            "llm_calls": 0,
            "held_out_reads": reader.held_out_reads,
            "max_time_index_read": reader.max_index_read,
            "held_out_frontier": MAX_ALLOWED_INDEX,
            "held_out_origins": list(r4a.HELD_OUT_ORIGINS),
            "held_out_span": list(r4a.HELD_OUT_SPAN),
            "raw_series_reads": reader.reads,
            "horizon_reads": 0,
            "oracle_quantities_used": 0,
            "files_written": [str(OUT_JSON.relative_to(PROJECT_ROOT)),
                              str(OUT_MD.relative_to(PROJECT_ROOT))],
            "existing_files_edited": 0,
            "prediction_store_written": False,
            "new_sha_or_hash": 0,
        },
        "provenance": {
            "prediction_store": str(STORE.relative_to(PROJECT_ROOT)),
            "prediction_store_entries": len(store),
            "prediction_store_physical_fits_already_spent":
                int(sum(int(entry.get("physical_fits") or 0)
                        for entry in store.values())),
            "data_version": r4a.DATA_VERSION,
            "loader": ("evaluation.main_protocol_p4.preflight_natural_gap_variant"
                       ".load_variant"),
            "loader_archive": str(preflight.WITH_MISSING.relative_to(PROJECT_ROOT)),
            "loader_nan_count": nan_count,
            "filled_cache_not_used": str(preflight.CACHE.relative_to(PROJECT_ROOT)),
            "row_construction_function": ("evaluation.main_protocol_p4"
                                          ".audit_r4a_pattern_identifiability"
                                          ".build_rows, imported unchanged"),
            "feature_card_function": ("evaluation.main_protocol_p4"
                                      ".audit_cross_fitted_targeting"
                                      ".series_features, reused; only the twelve "
                                      "numeric observables are kept, no mechanism "
                                      "and no ORACLE quantity"),
            "auc_function": ("evaluation.main_protocol_p4"
                             ".audit_r4a_pattern_identifiability.auc"),
            "readable_uids": len(readable),
            "block_definition": ("evaluation.main_protocol_p4.hec1_contract"
                                 ".ordering('forward'); position -> (block, origin)"),
            "geometry": {
                "context_steps": CONTEXT,
                "horizon_steps": r4a.HORIZON,
                "delayed_face_offset": DELAYED_OFFSET,
                "store_origin_mismatches": geometry_checks,
            },
            "positions_present": positions,
            "position_to_block": {str(p): b for p, b in position_block.items()},
            "cohort_positions": {
                block: sorted(p for p, b in position_block.items() if b == block)
                for block in sorted(set(position_block.values()))},
            "alias_check": {
                "program": ALIAS_PROGRAM,
                "entries_compared_against_ancestor": alias_compared,
                "entries_numerically_identical": alias_identical,
                "max_abs_difference": alias_max_abs_diff,
                "is_an_alias_of_ancestor":
                    bool(alias_compared and alias_identical == alias_compared),
                "handling": ("skipped from every table; its readings would be "
                             "ANCESTOR's to the last digit"),
            },
            "row_construction": {
                "unit": "(program, position, face, uid)",
                "repeated_measures": ("the same twenty uids recur at every "
                                      "position of their cohort, so rows are not "
                                      "independent; no significance is claimed"),
                "degenerate_uid_rows_dropped": skipped["degenerate_uid_rows"],
                "why_dropped": ("a degenerate uid is outside the executed scope, "
                                "so its reading is the untreated one by "
                                "construction"),
                "verifier_failed_entries_dropped": skipped["verifier_failed"],
                "entries_out_of_scope": skipped["programs_out_of_scope"],
                "total_rows_built": len(rows),
                "rows_after_dropping_the_alias": len(scoped),
                "unique_uids": len({row["uid"] for row in scoped}),
            },
        },
        "definitions": {
            "gain": "g = raw_per_view - program_per_view (positive = program helped)",
            "helped": "g > 0",
            "severe_harm": "-g > %.2f" % SEVERE_HARM,
            "cross_face_pair": ("the same (position, uid) read at origin and at "
                                "origin+%d" % DELAYED_OFFSET),
            "position_share_one_way": ("SS_between(position) / SS_total over all "
                                       "rows of the face"),
            "uid_share_weighted": ("inside each cohort, the uid main effect of an "
                                   "additive (uid + position) decomposition, "
                                   "SS_uid / SS_total; averaged over the four "
                                   "cohorts weighted by rows"),
            "residual_share_weighted": ("the same decomposition's residual "
                                        "SS(g - mean_uid - mean_position + mean) "
                                        "/ SS_total, row-weighted"),
            "share_null_expectation": ("(levels - 1) / (rows - 1), the expected "
                                       "between-group share under exchangeable "
                                       "noise; a diagnostic only, it changes no "
                                       "frozen threshold"),
            "unit_level_condition": ("the unit's own mean gain used as a score for "
                                     "its own rows; in-sample ceiling, NOT "
                                     "deployable, not comparable to a LODO number"),
            "best_visible_series_level": ("pooled AUC over the twelve numeric "
                                          "observables, oriented as "
                                          "max(auc, 1-auc); pooled, not LODO"),
            "verdict_rules": {
                "SERIES_LEVEL_CONDITION": "uid share >= %.2f and cross-face "
                                          "Spearman median >= %.2f"
                                          % (UID_SHARE_SERIES, SPEARMAN_SERIES),
                "COHORT_LEVEL_CONDITION": "position share >= %.2f and uid share "
                                          "< %.2f" % (POSITION_SHARE_COHORT,
                                                      UID_SHARE_SERIES),
                "MOSTLY_NOISE": "cross-face Spearman median < %.2f and uid share "
                                "< %.2f and position share < %.2f"
                                % (SPEARMAN_NOISE, UID_SHARE_SERIES,
                                   POSITION_SHARE_COHORT),
                "MIXED": "none of the above",
                "evaluation_order": ["SERIES_LEVEL_CONDITION",
                                     "COHORT_LEVEL_CONDITION", "MOSTLY_NOISE",
                                     "MIXED"],
                "note": ("the cross-face Spearman median is a program-level "
                         "quantity by construction -- one pairing serves both "
                         "faces -- so the two faces of a program share it"),
            },
        },
        "cross_face_persistence": persistence,
        "main_table": table,
        "verdict_counts_main_cells": verdict_counts,
    }
    return payload


def _md(payload: Mapping[str, Any]) -> str:
    boundary = payload["boundary"]
    provenance = payload["provenance"]
    lines = ["# R4A-b persistence and variance -- machine reading", ""]
    lines += [
        "Evidence class: %s" % payload["evidence_class"],
        "",
        "Boundary: fits=%d, llm=%d, held_out_reads=%d, max_time_index_read=%d "
        "(frontier %d), horizon_reads=%d." % (
            boundary["consumer_fits"], boundary["llm_calls"],
            boundary["held_out_reads"], boundary["max_time_index_read"],
            boundary["held_out_frontier"], boundary["horizon_reads"]),
        "Loader: `%s`, NaN=%d." % (provenance["loader"],
                                   provenance["loader_nan_count"]),
        "Rows built %d; after dropping the `%s` alias %d; %d unique uids. "
        "Dropped: %d degenerate-uid rows, %d verifier-failed entries, "
        "%d out-of-scope entries." % (
            provenance["row_construction"]["total_rows_built"],
            provenance["alias_check"]["program"],
            provenance["row_construction"]["rows_after_dropping_the_alias"],
            provenance["row_construction"]["unique_uids"],
            provenance["row_construction"]["degenerate_uid_rows_dropped"],
            len(provenance["row_construction"]["verifier_failed_entries_dropped"]),
            provenance["row_construction"]["entries_out_of_scope"]),
        "",
        "## Verdicts per (program, face)",
        "",
        "| cell | rows | cross-face Spearman median | position share | "
        "uid share (weighted) | uid null (weighted) | residual share | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def num(value: Any) -> str:
        return "n/a" if value is None else "%.6f" % value

    for key, cell in payload["main_table"].items():
        lines.append("| %s%s | %d | %s | %s | %s | %s | %s | %s |" % (
            r4a._bar(key), "" if cell["role"] == "main" else " (ref)",
            cell["n_rows"], num(cell["cross_face_spearman_median"]),
            num(cell["variance"]["position_share_one_way"]),
            num(cell["variance"]["uid_share_weighted"]),
            num(cell["variance"].get("uid_share_null_expectation_weighted")),
            num(cell["variance"]["residual_share_weighted"]),
            cell["verdict"]))
    lines += ["", "## Reading 1: cross-face persistence (per program)", "",
              "| program | pairs | positions | Spearman q25/median/q75 | "
              "pooled Spearman | pooled Pearson | P(delayed severe \\| support "
              "severe) | delayed base rate | helped kappa |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for program, cell in payload["cross_face_persistence"].items():
        carry = cell["severe_harm_carry_over"]
        lines.append("| %s | %d | %d | %s / %s / %s | %s | %s | %s | %s | %s |" % (
            r4a._bar(program), cell["n_pairs"], cell["n_positions_paired"],
            num(cell["spearman_by_position"]["q25"]),
            num(cell["spearman_by_position"]["median"]),
            num(cell["spearman_by_position"]["q75"]),
            num(cell["pooled_spearman"]), num(cell["pooled_pearson"]),
            num(carry["p_delayed_severe_given_support_severe"]),
            num(carry["delayed_base_rate"]),
            num(cell["helped_kappa"]["kappa"])))
    lines += ["", "## Reading 2: variance shares inside each cohort", "",
              "`null` is (levels-1)/(rows-1), the share the factor would take "
              "under exchangeable noise. It is a diagnostic and changes no "
              "frozen threshold.", "",
              "| cell | cohort | rows | positions | uid share | uid null | "
              "position share | position null | residual | shares sum |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, cell in payload["main_table"].items():
        for cohort, part in cell["variance"]["within_cohort"].items():
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r4a._bar(key), cohort, part["n_rows"], part["n_positions"],
                num(part["uid_share"]),
                num(part.get("uid_share_null_expectation")),
                num(part["position_share"]),
                num(part.get("position_share_null_expectation")),
                num(part["residual_share"]), num(part.get("shares_sum"))))
    lines += ["", "## Reading 3: z_peak, strength or direction", "",
              "| cell | Spearman with \\|g\\| | Spearman with g | "
              "pooled AUC severe_harm (raw) | direction | "
              "reads strength not direction |",
              "| --- | --- | --- | --- | --- | --- |"]
    for key, cell in payload["main_table"].items():
        z = cell["zpeak"]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            r4a._bar(key), num(z["spearman_with_abs_g"]),
            num(z["spearman_with_g"]), num(z["pooled_auc_severe_harm_raw"]),
            z["severe_harm_direction"], z["reads_strength_not_direction"]))
    lines += ["", "## Reading 4: unit-level condition against the visible "
              "vocabulary", "",
              "All four AUC columns are pooled and in-sample. The unit-level "
              "columns are a ceiling computed with the answer in hand and are "
              "**not deployable**.", "",
              "| cell | unit-level AUC helped | best visible AUC helped | "
              "unit-level AUC severe_harm | best visible AUC severe_harm | "
              "best visible feature (severe_harm) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for key, cell in payload["main_table"].items():
        unit = cell["unit_level_condition"]
        visible = cell["best_visible_series_level"]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            r4a._bar(key), num(unit["helped"]["auc_oriented"]),
            num(visible["helped"]["auc_oriented"]),
            num(unit["severe_harm"]["auc_oriented"]),
            num(visible["severe_harm"]["auc_oriented"]),
            visible["severe_harm"]["feature"]))
    lines += ["", "## Verdict rules (frozen before any number was seen)", ""]
    for name, rule in payload["definitions"]["verdict_rules"].items():
        if isinstance(rule, str):
            lines.append("- `%s`: %s" % (name, rule))
    lines += ["", "Alias check: `%s` compared on %d entries, %d numerically "
              "identical to ANCESTOR, max |diff| = %s; %s." % (
                  provenance["alias_check"]["program"],
                  provenance["alias_check"]["entries_compared_against_ancestor"],
                  provenance["alias_check"]["entries_numerically_identical"],
                  num(provenance["alias_check"]["max_abs_difference"]),
                  provenance["alias_check"]["handling"]), ""]
    return "\n".join(lines)


def main() -> int:
    payload = r4a._round(run())
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    boundary = payload["boundary"]
    print("fits=%d llm=%d held_out_reads=%d max_time_index_read=%d" % (
        boundary["consumer_fits"], boundary["llm_calls"],
        boundary["held_out_reads"], boundary["max_time_index_read"]))
    for key, cell in payload["main_table"].items():
        variance = cell["variance"]
        print("%-52s sp_med=%-9s pos=%-9s uid=%-9s res=%-9s %s" % (
            key,
            "n/a" if cell["cross_face_spearman_median"] is None
            else "%.6f" % cell["cross_face_spearman_median"],
            "n/a" if variance["position_share_one_way"] is None
            else "%.6f" % variance["position_share_one_way"],
            "n/a" if variance["uid_share_weighted"] is None
            else "%.6f" % variance["uid_share_weighted"],
            "n/a" if variance["residual_share_weighted"] is None
            else "%.6f" % variance["residual_share_weighted"],
            cell["verdict"]))
    print("verdict counts (main cells): %s" % payload["verdict_counts_main_cells"])
    print("wrote %s" % OUT_JSON)
    print("wrote %s" % OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
