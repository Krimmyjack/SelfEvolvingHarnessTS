"""R4A: can a deployment-visible pattern say which series a program helps?

The premise under test is the second half of the project's claim: not "the same
treatment flips sign across series" (already read at M0 level), but "the
condition that decides the sign can be read off data that is visible before the
origin".  Every earlier attempt used one bag of generic global fingerprints; this
one derives the quantities from the *mechanism* of the programs actually in the
store -- clipping programs get spike-recurrence quantities, gap-repair programs
get gap quantities -- and puts them next to the twelve-name deployment
vocabulary and next to an ORACLE row that is allowed to look inside the horizon.

Nothing here fits anything.  The per-series readings come from the m_r0k
prediction store, which was produced by an authorised package; this script only
indexes it.  The only data it touches beyond that is the raw with-missing
KDD2018 variant, read strictly at indices below the held-out frontier.

Definitions, thresholds and verdict rules are frozen in
``docs/R4A_PATTERN_IDENTIFIABILITY_TASKBOOK_2026-09-07.md`` and are not tuned
here.  Where a taskbook definition cannot be computed as written, the quantity is
reported ``NOT_COMPUTABLE_AS_DEFINED`` rather than replaced.

Run: ``python -m evaluation.main_protocol_p4.audit_r4a_pattern_identifiability``
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

from contracts import observables
from evaluation.main_protocol_p4 import audit_cross_fitted_targeting as targeting
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as p4b
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import run_hec1 as hec1
from SelfEvolvingHarnessTS.methods.ttha import public_tools

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE = PROJECT_ROOT / "_scratch/m_r0k_prediction_store.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/r4a_pattern_identifiability.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/r4a_pattern_identifiability.md"

DATA_VERSION = preflight.DATA_VERSION
CONTEXT = preflight.CONTEXT
HORIZON = preflight.HORIZON
PERIOD = preflight.PERIOD
DELAYED_OFFSET = 48

MAIN_PROGRAMS = (
    "ANCESTOR",
    "W1_hampel_filter",
    "W2_pmc_then_outlier_mad",
    "W3_outlier_mad_then_pmc",
)
REFERENCE_PROGRAMS = (
    "CAND_outlier_iqr({})>hampel_filter({})",
    "CAND_outlier_iqr({})>winsorize({})",
)
FACES = ("support_face", "delayed_face")

# --- frozen thresholds (taskbook 2 and 4; set before any number was seen) ---
SEVERE_HARM = 0.30
SPIKE_Z = 3.0
INFORMATIVE_AUC = 0.70
WEAK_AUC = 0.60
ORACLE_AUC = 0.75
ORACLE_MARGIN = 0.10

# --- held-out frontier (taskbook 1) ---
HELD_OUT_SPAN = (80, 120)
HELD_OUT_ORIGINS = (4056, 4296, 4536, 4776, 5016)
MAX_ALLOWED_INDEX = min(HELD_OUT_ORIGINS)

SPIKE_FEATURES = (
    "spike_recurrence",
    "spike_recency",
    "spike_tail_count",
    "spike_head_count",
    "spike_sign_balance",
    "spike_peak_over_tail_sd",
)
GAP_FEATURES = (
    "gap_tail_fraction",
    "gap_recurrence",
    "gap_longest_run_steps",
)
NOT_COMPUTABLE = {
    "gap_period_aligned": (
        "the definition asks for the share of context gaps whose within-day "
        "phase (period=24) coincides with the horizon's phases, but the "
        "horizon is 48 = 2 x 24 steps and therefore covers every one of the 24 "
        "phases; the quantity is identically 1.0 wherever a gap exists and 0/0 "
        "otherwise, so it separates nothing.  Reported as defined rather than "
        "swapped for another definition."
    ),
}
OBSERVABLE_NUMERIC = tuple(
    name for name, kind in observables.OBSERVABLE_FEATURES.items() if kind == "number"
)
ORACLE_FEATURES = (
    "oracle_future_spike",
    "oracle_future_gap",
    "oracle_future_level_shift",
)
VISIBLE_FEATURES = SPIKE_FEATURES + GAP_FEATURES + OBSERVABLE_NUMERIC
TARGETS = ("helped", "severe_harm")


# ---------------------------------------------------------------------------
# 1. boundary-accounted reads
# ---------------------------------------------------------------------------


class Reader:
    """Every raw-series read passes through here so the frontier is checked.

    The held-out zone is a set of (series, origin) pairs, but the cheap and
    stronger invariant is the one the taskbook names: no time index at or past
    4056 is ever touched, for any series.  A violation raises rather than
    records, because a boundary that reports itself after the fact has already
    been crossed.
    """

    def __init__(self, variant: Mapping[str, np.ndarray]) -> None:
        self._variant = variant
        self.max_index_read = -1
        self.held_out_reads = 0
        self.held_out_uids = frozenset(hec1.block_uids(HELD_OUT_SPAN))
        self.reads = 0

    def _account(self, uid: str, stop: int) -> None:
        self.reads += 1
        last = int(stop) - 1
        if uid in self.held_out_uids and last >= MAX_ALLOWED_INDEX:
            self.held_out_reads += 1
        if last >= MAX_ALLOWED_INDEX:
            raise AssertionError(
                "read of %s reaches index %d, at or past the held-out frontier %d"
                % (uid, last, MAX_ALLOWED_INDEX)
            )
        self.max_index_read = max(self.max_index_read, last)

    def window(self, uid: str, start: int, stop: int) -> np.ndarray:
        self._account(uid, stop)
        return np.asarray(self._variant[uid][int(start):int(stop)], dtype=np.float64)

    def account_external(self, uid: str, stop: int) -> None:
        """For windows the reused project function slices internally."""
        self._account(uid, stop)


# ---------------------------------------------------------------------------
# 2. candidate pattern quantities (taskbook 3)
# ---------------------------------------------------------------------------


def _robust_z(window: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Robust z on the context, with the deployed centre/scale convention."""
    center, scale = public_tools._robust_center_scale(window)
    if center is None or scale is None:
        return np.full(window.shape, np.nan), float("nan"), float("nan")
    scale = max(float(scale), observables.PUBLIC_ROBUST_Z_MAD_FLOOR)
    return (window - float(center)) / scale, float(center), scale


def mechanism_features(context: np.ndarray) -> dict[str, float]:
    """Spike- and gap-recurrence quantities, all strictly pre-origin."""
    z, _center, scale = _robust_z(context)
    finite = np.isfinite(context)
    spike = np.isfinite(z) & (np.abs(z) >= SPIKE_Z)
    idx = np.flatnonzero(spike)
    n = context.size
    windows = [spike[start:start + 48] for start in range(0, n, 48)]

    out: dict[str, float] = {}
    out["spike_recurrence"] = float(np.mean([bool(w.any()) for w in windows]))
    out["spike_recency"] = (
        float((n - int(idx[-1])) / n) if idx.size else 1.0
    )
    out["spike_tail_count"] = float(spike[-48:].sum())
    out["spike_head_count"] = float(spike[:-48].sum())
    if idx.size:
        positive = int((z[idx] > 0).sum())
        negative = int((z[idx] < 0).sum())
        out["spike_sign_balance"] = float((positive - negative) / idx.size)
    else:
        out["spike_sign_balance"] = 0.0
    # The tail's robust SD is taken on the z series, so the ratio is a ratio of
    # two robust scales and does not move with the series' physical units.
    _tz, _tc, tail_scale = _robust_z(context[-48:])
    peak = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else float("nan")
    tail_in_z = tail_scale / scale if scale and math.isfinite(scale) else float("nan")
    out["spike_peak_over_tail_sd"] = (
        float(peak / tail_in_z)
        if math.isfinite(peak) and math.isfinite(tail_in_z) and tail_in_z > 0
        else float("nan")
    )

    missing = ~finite
    out["gap_tail_fraction"] = float(missing[-48:].mean())
    out["gap_recurrence"] = float(
        np.mean([bool((~finite[start:start + 48]).any()) for start in range(0, n, 48)])
    )
    runs = public_tools._observed_runs(missing)
    out["gap_longest_run_steps"] = float(
        max((stop - start for start, stop in runs), default=0)
    )
    # Kept only to demonstrate the degeneracy recorded in NOT_COMPUTABLE.
    out["gap_period_aligned"] = float(1.0) if missing.any() else float("nan")
    return out


def oracle_features(context: np.ndarray, horizon: np.ndarray) -> dict[str, float]:
    """ORACLE only.  Reads truth inside [origin, origin+48); never deployable."""
    center, scale = public_tools._robust_center_scale(context)
    out: dict[str, float] = {}
    if center is None or scale is None:
        return dict.fromkeys(ORACLE_FEATURES, float("nan"))
    scale = max(float(scale), observables.PUBLIC_ROBUST_Z_MAD_FLOOR)
    z = (horizon - float(center)) / scale
    spike = np.isfinite(z) & (np.abs(z) >= SPIKE_Z)
    out["oracle_future_spike"] = float(bool(spike.any()))
    out["oracle_future_gap"] = float((~np.isfinite(horizon)).mean())
    tail = context[-48:]
    future_mean = float(np.nanmean(horizon)) if np.isfinite(horizon).any() else np.nan
    tail_mean = float(np.nanmean(tail)) if np.isfinite(tail).any() else np.nan
    out["oracle_future_level_shift"] = (
        float((future_mean - tail_mean) / scale)
        if math.isfinite(future_mean) and math.isfinite(tail_mean)
        else float("nan")
    )
    return out


# ---------------------------------------------------------------------------
# 3. statistics, all rank- and count-based (no fits, no new dependency)
# ---------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or sorted_values[stop] != sorted_values[start]:
            ranks[order[start:stop]] = 0.5 * (start + stop + 1)
            start = stop
    return ranks


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney U as AUC; ties get average ranks.  None if a class is empty."""
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    n1 = int(labels.sum())
    n0 = int(labels.size - n1)
    if n1 == 0 or n0 == 0:
        return None
    ranks = _average_ranks(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _quantile_bins(train: np.ndarray) -> np.ndarray:
    edges = np.unique(np.quantile(train, [0.25, 0.5, 0.75]))
    return edges


def brier_fold(train_x: np.ndarray, train_y: np.ndarray,
               test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float | None]:
    """Fold-internal empirical-rate-per-bin prediction against the B1 constant.

    Bin edges are the training side's quartiles: the frozen deployment edges are
    used for the stump in ``stump_fold``, where the taskbook names them, but they
    place every [0,1]-valued mechanism quantity in one bin and would make this
    row vacuous rather than negative.
    """
    keep_train = np.isfinite(train_x)
    keep_test = np.isfinite(test_x)
    if keep_train.sum() < 4 or keep_test.sum() == 0:
        return {"brier_b1": None, "brier_feature": None, "delta": None}
    train_x, train_y = train_x[keep_train], train_y[keep_train]
    test_x, test_y = test_x[keep_test], test_y[keep_test]
    base = float(train_y.mean())
    edges = _quantile_bins(train_x)
    train_bin = np.digitize(train_x, edges)
    test_bin = np.digitize(test_x, edges)
    rates = {}
    for b in np.unique(train_bin):
        rates[int(b)] = float(train_y[train_bin == b].mean())
    predicted = np.array([rates.get(int(b), base) for b in test_bin], dtype=np.float64)
    brier_feature = float(np.mean((predicted - test_y) ** 2))
    brier_b1 = float(np.mean((base - test_y) ** 2))
    return {
        "brier_b1": brier_b1,
        "brier_feature": brier_feature,
        "delta": brier_b1 - brier_feature,
    }


def _frozen_edges(feature: str) -> tuple[float, ...]:
    return tuple(
        float(edge)
        for edge in observables._NUMERIC_BIN_EDGES.get(feature, (0.0, 1.0, 3.0, 6.0))
    )


def stump_fold(feature: str, train_x: np.ndarray, train_g: np.ndarray,
               test_x: np.ndarray, test_g: np.ndarray) -> dict[str, Any]:
    """One frozen-bin stump, chosen on the training blocks, read on the test one.

    Utility is always divided by the whole served population of the fold, so a
    rule that treats three series cannot look large by shrinking its own
    denominator.
    """
    keep_train = np.isfinite(train_x)
    keep_test = np.isfinite(test_x)
    if keep_train.sum() == 0 or keep_test.sum() == 0:
        return {"rule": None, "utility_rule": None, "utility_treat_all": None,
                "vs_treat_all": None, "vs_treat_none": None, "treated_fraction": None}
    n_train = float(train_g.size)
    n_test = float(test_g.size)
    best = None
    for edge in _frozen_edges(feature):
        for direction in (">=", "<="):
            selected = (
                (np.isfinite(train_x)) & (train_x >= edge)
                if direction == ">=" else
                (np.isfinite(train_x)) & (train_x <= edge)
            )
            utility = float(train_g[selected].sum() / n_train)
            if best is None or utility > best[0]:
                best = (utility, edge, direction)
    _train_utility, edge, direction = best
    selected = (
        (np.isfinite(test_x)) & (test_x >= edge)
        if direction == ">=" else
        (np.isfinite(test_x)) & (test_x <= edge)
    )
    utility_rule = float(test_g[selected].sum() / n_test)
    utility_all = float(test_g.sum() / n_test)
    return {
        "rule": "%s %s %.6f" % (feature, direction, edge),
        "utility_rule": utility_rule,
        "utility_treat_all": utility_all,
        "vs_treat_all": utility_rule - utility_all,
        "vs_treat_none": utility_rule,
        "treated_fraction": float(selected.sum() / n_test),
    }


# ---------------------------------------------------------------------------
# 4. rows
# ---------------------------------------------------------------------------


def build_rows(store: Mapping[str, Any], cards: Mapping[tuple[int, str], dict],
               position_block: Mapping[int, str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    skipped = {"verifier_failed": [], "degenerate_uid_rows": 0,
               "programs_out_of_scope": 0}
    for key, entry in store.items():
        program = str(entry["program"])
        if program not in MAIN_PROGRAMS + REFERENCE_PROGRAMS:
            skipped["programs_out_of_scope"] += 1
            continue
        if not entry.get("verifier_passed"):
            skipped["verifier_failed"].append(key)
            continue
        position = int(entry["position"])
        face = str(entry["face"])
        origin = int(entry["origin"])
        degenerate = set(str(uid) for uid in entry.get("degenerate_uids") or ())
        raw = np.asarray(entry["raw_per_view"], dtype=np.float64)
        prog = np.asarray(entry["program_per_view"], dtype=np.float64)
        for index, uid in enumerate(entry["eval_uids"]):
            uid = str(uid)
            if uid in degenerate:
                # Not inside the executed scope: its reading is the untreated
                # one by construction and carries no helped/harmed information.
                skipped["degenerate_uid_rows"] += 1
                continue
            gain = float(raw[index] - prog[index])
            rows.append({
                "program": program,
                "position": position,
                "face": face,
                "origin": origin,
                "block": position_block[position],
                "uid": uid,
                "g": gain,
                "helped": 1 if gain > 0 else 0,
                "harmed": 1 if gain < 0 else 0,
                "severe_harm": 1 if -gain > SEVERE_HARM else 0,
                "features": cards[(origin, uid)],
            })
    return rows, skipped


def _matrix(rows: Sequence[Mapping[str, Any]], feature: str) -> np.ndarray:
    return np.array([float(row["features"].get(feature, np.nan)) for row in rows],
                    dtype=np.float64)


def feature_reading(rows: Sequence[Mapping[str, Any]], feature: str, target: str,
                    folds: Sequence[str]) -> dict[str, Any]:
    """Pooled AUC plus leave-one-block-out folds, oriented on the training side."""
    x = _matrix(rows, feature)
    y = np.array([row[target] for row in rows], dtype=np.float64)
    g = np.array([row["g"] for row in rows], dtype=np.float64)
    blocks = np.array([row["block"] for row in rows])
    pooled = auc(x, y)
    per_fold: dict[str, Any] = {}
    for block in folds:
        test = blocks == block
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            per_fold[block] = {"auc_raw": None, "auc_oriented": None,
                               "train_auc": None, "n_test": int(test.sum()),
                               "positives_test": None, "brier": None}
            continue
        train_auc = auc(x[train], y[train])
        raw_auc = auc(x[test], y[test])
        if raw_auc is None or train_auc is None:
            oriented = None
        else:
            oriented = raw_auc if train_auc >= 0.5 else 1.0 - raw_auc
        per_fold[block] = {
            "auc_raw": raw_auc,
            "auc_oriented": oriented,
            "train_auc": train_auc,
            "n_test": int(test.sum()),
            "positives_test": int(y[test].sum()),
            "brier": brier_fold(x[train], y[train], x[test], y[test]),
        }
    oriented = [f["auc_oriented"] for f in per_fold.values()
                if f["auc_oriented"] is not None]
    raws = [f["auc_raw"] for f in per_fold.values() if f["auc_raw"] is not None]
    directions = {bool(value >= 0.5) for value in raws}
    stumps = {}
    for block in folds:
        test = blocks == block
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            stumps[block] = None
            continue
        stumps[block] = stump_fold(feature, x[train], g[train], x[test], g[test])
    vs_all = [s["vs_treat_all"] for s in stumps.values()
              if s and s["vs_treat_all"] is not None]
    vs_none = [s["vs_treat_none"] for s in stumps.values()
               if s and s["vs_treat_none"] is not None]
    briers = [f["brier"]["delta"] for f in per_fold.values()
              if f.get("brier") and f["brier"]["delta"] is not None]
    return {
        "pooled_auc": pooled,
        "folds": per_fold,
        "lodo_mean_auc_oriented": float(np.mean(oriented)) if oriented else None,
        "lodo_min_auc_oriented": float(np.min(oriented)) if oriented else None,
        "direction_consistent": bool(len(directions) == 1) if raws else None,
        "brier_delta_mean": float(np.mean(briers)) if briers else None,
        "stump": {
            "folds": stumps,
            "mean_vs_treat_all": float(np.mean(vs_all)) if vs_all else None,
            "mean_vs_treat_none": float(np.mean(vs_none)) if vs_none else None,
        },
    }


def _is_informative(reading: Mapping[str, Any]) -> bool:
    return bool(reading.get("lodo_min_auc_oriented") is not None
                and reading["lodo_min_auc_oriented"] >= INFORMATIVE_AUC
                and reading.get("direction_consistent"))


def verdict(best: Mapping[str, Any] | None,
            all_readings: Sequence[Mapping[str, Any]] = ()) -> tuple[str, str]:
    """INFORMATIVE asks whether *any* visible quantity clears every fold.

    The reason is carried with the verdict because the frozen rule lets an AUC
    well under 0.60 read as WEAK when the folds disagree on direction, and a
    reader who sees only the label would take that for a weak signal rather than
    for no signal with an unstable sign.
    """
    if any(_is_informative(reading) for reading in all_readings):
        return "PATTERN_INFORMATIVE", "every LODO fold clears %.2f" % INFORMATIVE_AUC
    if best is None or best.get("lodo_mean_auc_oriented") is None:
        return "NOT_READABLE", "no feature has a readable fold"
    if best["lodo_mean_auc_oriented"] >= WEAK_AUC:
        return "PATTERN_WEAK", "best LODO mean in [%.2f, %.2f)" % (
            WEAK_AUC, INFORMATIVE_AUC)
    if not best["direction_consistent"]:
        return "PATTERN_WEAK", (
            "best LODO mean %.6f is below %.2f, but the folds disagree on sign, "
            "which the frozen rule reads as WEAK"
            % (best["lodo_mean_auc_oriented"], WEAK_AUC))
    return "PATTERN_UNINFORMATIVE", "best LODO mean below %.2f" % WEAK_AUC


def _flatten(readings: Mapping[str, Mapping[str, Any]],
             names: Sequence[str]) -> list[Mapping[str, Any]]:
    return [readings[name][target] for name in names for target in TARGETS]


def _pick_best(readings: Mapping[str, Mapping[str, Any]],
               names: Sequence[str]) -> tuple[str | None, dict | None]:
    best_name, best = None, None
    for name in names:
        for target in TARGETS:
            reading = readings[name][target]
            value = reading.get("lodo_mean_auc_oriented")
            if value is None:
                continue
            if best is None or value > best["lodo_mean_auc_oriented"]:
                best_name, best = "%s|%s" % (name, target), dict(reading)
    return best_name, best


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 6)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


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
    reader = Reader(variant)

    positions = sorted({int(entry["position"]) for entry in store.values()})
    position_block = {p: str(forward[p]["block"]) for p in positions}
    blocks = sorted(set(position_block.values()))

    # geometry as the code states it, not as the taskbook assumed it
    geometry_checks = []
    for key, entry in store.items():
        position, face, origin = (int(entry["position"]), str(entry["face"]),
                                  int(entry["origin"]))
        expected = int(forward[position]["origin"]) + (
            DELAYED_OFFSET if face == "delayed_face" else 0)
        if origin != expected:
            geometry_checks.append({"key": key, "origin": origin,
                                    "expected": expected})

    # Two rows of the main table are only distinct if the two programs actually
    # produced different numbers; a composition whose second operator is inert
    # would otherwise be counted as an independent reading of the same effect.
    equivalence = {}
    for program in MAIN_PROGRAMS + REFERENCE_PROGRAMS:
        identical = compared = 0
        for key, entry in store.items():
            if entry["program"] != program:
                continue
            twin = store.get("ANCESTOR|%d|%s" % (int(entry["position"]),
                                                 entry["face"]))
            if twin is None or not entry.get("verifier_passed"):
                continue
            compared += 1
            if np.array_equal(np.asarray(entry["program_per_view"]),
                              np.asarray(twin["program_per_view"])):
                identical += 1
        equivalence[program] = {
            "entries_compared_against_ancestor": compared,
            "entries_numerically_identical": identical,
            "is_an_alias_of_ancestor": bool(compared and identical == compared),
        }

    # --- feature cards, one per (origin, uid) ---
    cells: dict[tuple[int, str, int], dict[str, Any]] = {}
    for entry in store.values():
        position, face = int(entry["position"]), str(entry["face"])
        cells.setdefault((position, face, int(entry["origin"])), {
            "uids": [str(uid) for uid in entry["eval_uids"]],
            "origin": int(entry["origin"]),
        })
    cards: dict[tuple[int, str], dict[str, float]] = {}
    for cell in cells.values():
        origin, uids = cell["origin"], cell["uids"]
        matrix, names = targeting.series_features(variant, uids, origin)
        for uid in uids:
            reader.account_external(uid, origin)
        for index, uid in enumerate(uids):
            if (origin, uid) in cards:
                continue
            context = reader.window(uid, origin - CONTEXT, origin)
            horizon = reader.window(uid, origin, origin + HORIZON)
            card = {name: float(matrix[index][column])
                    for column, name in enumerate(names)
                    if name in OBSERVABLE_NUMERIC}
            card.update(mechanism_features(context))
            card.update(oracle_features(context, horizon))
            cards[(origin, uid)] = card

    rows, skipped = build_rows(store, cards, position_block)

    # --- per (program, face) main table ---
    table: dict[str, Any] = {}
    for program in MAIN_PROGRAMS + REFERENCE_PROGRAMS:
        for face in FACES:
            subset = [row for row in rows
                      if row["program"] == program and row["face"] == face]
            if not subset:
                continue
            g = np.array([row["g"] for row in subset], dtype=np.float64)
            present_blocks = sorted({row["block"] for row in subset})
            readings = {
                name: {target: feature_reading(subset, name, target, present_blocks)
                       for target in TARGETS}
                for name in VISIBLE_FEATURES + ORACLE_FEATURES
            }
            best_name, best = _pick_best(readings, VISIBLE_FEATURES)
            three = [b for b in present_blocks if b in ("[0:40]", "[40:80]",
                                                        "[80:120]")]
            readings_three = {
                name: {target: feature_reading(subset, name, target, three)
                       for target in TARGETS}
                for name in VISIBLE_FEATURES
            }
            best_three_name, best_three = _pick_best(readings_three, VISIBLE_FEATURES)
            oracle_row = {
                name: {
                    "pooled_auc_severe_harm":
                        readings[name]["severe_harm"]["pooled_auc"],
                    "lodo_mean_auc_severe_harm":
                        readings[name]["severe_harm"]["lodo_mean_auc_oriented"],
                }
                for name in ORACLE_FEATURES
            }
            best_visible_severe = max(
                ((name, readings[name]["severe_harm"]["pooled_auc"])
                 for name in VISIBLE_FEATURES
                 if readings[name]["severe_harm"]["pooled_auc"] is not None),
                key=lambda pair: max(pair[1], 1.0 - pair[1]),
                default=(None, None),
            )
            visible_pooled = (
                None if best_visible_severe[1] is None
                else max(best_visible_severe[1], 1.0 - best_visible_severe[1])
            )
            best_stump = None
            for name in VISIBLE_FEATURES:
                for target in TARGETS:
                    stump = readings[name][target]["stump"]
                    value = stump["mean_vs_treat_all"]
                    if value is None:
                        continue
                    if best_stump is None or value > best_stump["mean_vs_treat_all"]:
                        best_stump = {
                            "feature": name,
                            "selected_on": target,
                            "rules_per_fold": {
                                block: (fold or {}).get("rule")
                                for block, fold in stump["folds"].items()},
                            "vs_treat_all_per_fold": {
                                block: (fold or {}).get("vs_treat_all")
                                for block, fold in stump["folds"].items()},
                            "mean_vs_treat_all": value,
                            "mean_vs_treat_none": stump["mean_vs_treat_none"],
                        }
            cell_verdict, cell_why = verdict(
                best, _flatten(readings, VISIBLE_FEATURES))
            three_verdict, three_why = verdict(
                best_three, _flatten(readings_three, VISIBLE_FEATURES))
            table["%s|%s" % (program, face)] = {
                "role": "main" if program in MAIN_PROGRAMS else "reference",
                "n_rows": len(subset),
                "n_unique_uids": len({row["uid"] for row in subset}),
                "n_positions": len({row["position"] for row in subset}),
                "blocks": present_blocks,
                "b1": {
                    "helped_rate": float(np.mean([row["helped"] for row in subset])),
                    "harmed_rate": float(np.mean([row["harmed"] for row in subset])),
                    "severe_harm_rate":
                        float(np.mean([row["severe_harm"] for row in subset])),
                    "mean_g": float(g.mean()),
                    "worst_single_series_harm": float(max(0.0, -g.min())),
                },
                "best_visible_feature": best_name,
                "best_visible_reading": best,
                "verdict": cell_verdict,
                "verdict_reason": cell_why,
                "best_visible_feature_three_named_blocks": best_three_name,
                "verdict_three_named_blocks": three_verdict,
                "verdict_three_named_blocks_reason": three_why,
                "best_frozen_bin_stump": best_stump,
                "oracle_row": oracle_row,
                "best_visible_pooled_auc_severe_harm": {
                    "feature": best_visible_severe[0],
                    "auc_max_of_both_directions": visible_pooled,
                },
                "per_feature": readings,
            }

    # --- transport flip, unit = (program, position, uid) ---
    by_unit: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in rows:
        by_unit.setdefault((row["program"], row["position"], row["uid"]), {})[
            row["face"]] = row["g"]
    flip_rows = []
    for (program, position, uid), faces in by_unit.items():
        if "support_face" not in faces or "delayed_face" not in faces:
            continue
        support_origin = int(forward[position]["origin"])
        flip_rows.append({
            "program": program,
            "position": position,
            "uid": uid,
            "block": position_block[position],
            "g": faces["support_face"],
            "transport_flip": 1 if (faces["support_face"] > 0
                                    and faces["delayed_face"] < 0) else 0,
            "features": cards[(support_origin, uid)],
        })
    flip_table: dict[str, Any] = {}
    for program in MAIN_PROGRAMS + REFERENCE_PROGRAMS:
        subset = [row for row in flip_rows if row["program"] == program]
        if not subset:
            continue
        present_blocks = sorted({row["block"] for row in subset})
        eligible = [row for row in subset if row["g"] > 0]
        readings = {
            name: feature_reading(subset, name, "transport_flip", present_blocks)
            for name in VISIBLE_FEATURES
        }
        best_name, best = None, None
        for name, reading in readings.items():
            value = reading["lodo_mean_auc_oriented"]
            if value is None:
                continue
            if best is None or value > best["lodo_mean_auc_oriented"]:
                best_name, best = name, dict(reading)
        flip_verdict, flip_why = verdict(best, list(readings.values()))
        flip_table[program] = {
            "n_units": len(subset),
            "n_support_helped_units": len(eligible),
            "flip_rate_over_all_units":
                float(np.mean([row["transport_flip"] for row in subset])),
            "flip_rate_over_support_helped":
                float(np.mean([row["transport_flip"] for row in eligible]))
                if eligible else None,
            "best_visible_feature": best_name,
            "best_visible_reading": best,
            "verdict": flip_verdict,
            "verdict_reason": flip_why,
            "per_feature": readings,
        }

    # --- structural upper bound, across programs ---
    bound_rows = []
    for key, cell in table.items():
        if cell["role"] != "main":
            continue
        visible = cell["best_visible_pooled_auc_severe_harm"][
            "auc_max_of_both_directions"]
        for name, reading in cell["oracle_row"].items():
            value = reading["pooled_auc_severe_harm"]
            if value is None or visible is None:
                continue
            oriented = max(value, 1.0 - value)
            bound_rows.append({
                "cell": key,
                "oracle_feature": name,
                "oracle_pooled_auc": oriented,
                "best_visible_pooled_auc": visible,
                "margin": oriented - visible,
                "triggers": bool(oriented >= ORACLE_AUC
                                 and oriented - visible >= ORACLE_MARGIN),
            })
    triggered = [row for row in bound_rows if row["triggers"]]
    structural = {
        "verdict": ("CONDITION_PARTLY_IN_THE_FUTURE" if triggered
                    else "NO_FUTURE_ADVANTAGE_DETECTED"),
        "rule": ("an ORACLE quantity reaches AUC >= %.2f on severe_harm and "
                 "beats the best visible quantity by >= %.2f"
                 % (ORACLE_AUC, ORACLE_MARGIN)),
        "triggering_rows": triggered,
        "max_margin_row": max(bound_rows, key=lambda row: row["margin"])
        if bound_rows else None,
        "all_rows": bound_rows,
    }

    payload = {
        "task": "R4A pattern identifiability (development mechanism reading)",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_class": "MECHANISM / NEGATIVE-capable; not a capability claim",
        "boundary": {
            "consumer_fits": 0,
            "llm_calls": 0,
            "held_out_reads": reader.held_out_reads,
            "max_time_index_read": reader.max_index_read,
            "held_out_frontier": MAX_ALLOWED_INDEX,
            "held_out_origins": list(HELD_OUT_ORIGINS),
            "held_out_span": list(HELD_OUT_SPAN),
            "raw_series_reads": reader.reads,
            "files_written": [str(OUT_JSON.relative_to(PROJECT_ROOT)),
                              str(OUT_MD.relative_to(PROJECT_ROOT))],
            "prediction_store_written": False,
            "new_sha_or_hash": 0,
        },
        "provenance": {
            "prediction_store": str(STORE.relative_to(PROJECT_ROOT)),
            "prediction_store_entries": len(store),
            "prediction_store_physical_fits_already_spent":
                int(sum(int(entry.get("physical_fits") or 0)
                        for entry in store.values())),
            "data_version": DATA_VERSION,
            "loader": ("evaluation.main_protocol_p4.preflight_natural_gap_variant"
                       ".load_variant"),
            "loader_archive": str(preflight.WITH_MISSING.relative_to(PROJECT_ROOT)),
            "loader_nan_count": nan_count,
            "filled_cache_not_used": str(preflight.CACHE.relative_to(PROJECT_ROOT)),
            "feature_card_function": ("evaluation.main_protocol_p4"
                                      ".audit_cross_fitted_targeting.series_features"
                                      " (the same one smoke_live_scope._feature_cards"
                                      " calls), reused, not recomputed"),
            "robust_z_convention": ("SelfEvolvingHarnessTS.methods.ttha.public_tools"
                                    "._robust_center_scale, the deployed one"),
            "roster_source": ("evaluation.main_protocol_p4.run_hec1.readable_uids /"
                              " block_uids over artifacts/p4s_main_experiment_supply"
                              ".json"),
            "readable_uids": len(readable),
            "block_definition": ("evaluation.main_protocol_p4.hec1_contract"
                                 ".ordering('forward'); position -> (block, origin)"),
            "geometry": {
                "context_steps": CONTEXT,
                "horizon_steps": HORIZON,
                "period": PERIOD,
                "delayed_face_offset": DELAYED_OFFSET,
                "source": ("p4b_contract.CONTEXT_LENGTH, "
                           "preflight_natural_gap_variant.HORIZON, "
                           "hec1_contract.FACES"),
                "store_origin_mismatches": geometry_checks,
            },
            "program_equivalence_against_ancestor": equivalence,
            "blocks_present": blocks,
            "positions_present": positions,
            "position_to_block": {str(p): b for p, b in position_block.items()},
            "taskbook_deviations": [
                {
                    "what": "the store spans four cohort blocks, not three",
                    "taskbook": ("section 4.2 freezes LODO over [0:40] / [40:80] / "
                                 "[80:120] and asks for three folds"),
                    "observed": ("positions 23-25 sit on [120:160]; block [80:120] "
                                 "carries only positions 16 and 17"),
                    "handled": ("leave-one-block-out is run over every block "
                                "present (primary) and, separately, over the three "
                                "named blocks (verdict_three_named_blocks)"),
                },
                {
                    "what": ("W3_outlier_mad_then_pmc is numerically identical to "
                             "ANCESTOR in the store"),
                    "taskbook": "section 1 lists four independent main programs",
                    "observed": ("see provenance.program_equivalence_against_ancestor; "
                                 "period_median_complete placed after outlier_mad "
                                 "changes nothing on these cells, so the W3 row is "
                                 "an alias, not a second reading"),
                    "handled": ("the row is kept and reported, but must not be "
                                "counted as independent evidence"),
                },
            ],
            "row_construction": {
                "unit": "(program, position, face, uid)",
                "repeated_measures": ("the same uid recurs at every position of "
                                      "its block, so rows are not independent; no "
                                      "significance is claimed"),
                "degenerate_uid_rows_dropped": skipped["degenerate_uid_rows"],
                "why_dropped": ("a degenerate uid is outside the executed scope, so "
                                "its reading is the untreated one by construction"),
                "verifier_failed_entries_dropped": skipped["verifier_failed"],
                "entries_out_of_scope": skipped["programs_out_of_scope"],
                "total_rows": len(rows),
                "unique_uids": len({row["uid"] for row in rows}),
            },
        },
        "definitions": {
            "gain": "g = raw_per_view - program_per_view (positive = program helped)",
            "helped": "g > 0",
            "harmed": "g < 0",
            "severe_harm": "-g > %.2f" % SEVERE_HARM,
            "transport_flip": "support face helped and delayed face harmed",
            "spike": "|robust z| >= %.1f on the 192-step context" % SPIKE_Z,
            "auc": "Mann-Whitney U with average ranks; folds oriented on the train side",
            "brier_bins": "training-side quartiles, fold-internal",
            "stump_bins": "contracts.observables._NUMERIC_BIN_EDGES, frozen",
            "utility": "G(S) = sum(g over S) / (all served rows in the fold)",
            "not_computable": NOT_COMPUTABLE,
            "verdict_rules": {
                "PATTERN_INFORMATIVE": ("every LODO fold >= %.2f on helped or "
                                        "severe_harm and one direction across folds"
                                        % INFORMATIVE_AUC),
                "PATTERN_WEAK": ("LODO mean in [%.2f, %.2f) or folds disagree on "
                                 "direction" % (WEAK_AUC, INFORMATIVE_AUC)),
                "PATTERN_UNINFORMATIVE": "LODO mean < %.2f" % WEAK_AUC,
            },
        },
        "features": {
            "mechanism_visible": list(SPIKE_FEATURES + GAP_FEATURES),
            "observable_vocabulary": list(OBSERVABLE_NUMERIC),
            "oracle": list(ORACLE_FEATURES),
            "not_computable": list(NOT_COMPUTABLE),
        },
        "main_table": table,
        "transport_flip": flip_table,
        "structural_upper_bound": structural,
    }
    return payload


def _bar(text: Any) -> str:
    """A cell key carries a '|', which would otherwise split the table column."""
    return str(text).replace("|", "\\|")


def _md(payload: Mapping[str, Any]) -> str:
    lines = ["# R4A pattern identifiability -- machine reading", ""]
    boundary = payload["boundary"]
    lines += [
        "Evidence class: %s" % payload["evidence_class"],
        "",
        "Boundary: fits=%d, llm=%d, held_out_reads=%d, max_time_index_read=%d "
        "(frontier %d)." % (boundary["consumer_fits"], boundary["llm_calls"],
                            boundary["held_out_reads"],
                            boundary["max_time_index_read"],
                            boundary["held_out_frontier"]),
        "Loader: `%s`, NaN=%d." % (payload["provenance"]["loader"],
                                   payload["provenance"]["loader_nan_count"]),
        "",
        "## Per (program, face)",
        "",
        "| cell | rows | uids | helped | severe_harm | mean g | best visible "
        "(feature / target) | LODO mean | LODO folds | verdict | why |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, cell in payload["main_table"].items():
        best = cell["best_visible_reading"] or {}
        folds = ", ".join(
            "%s=%s" % (block, "n/a" if fold["auc_oriented"] is None
                       else "%.6f" % fold["auc_oriented"])
            for block, fold in (best.get("folds") or {}).items())
        lines.append(
            "| %s%s | %d | %d | %.6f | %.6f | %.6f | %s | %s | %s | %s | %s |" % (
                _bar(key), "" if cell["role"] == "main" else " (ref)",
                cell["n_rows"], cell["n_unique_uids"],
                cell["b1"]["helped_rate"], cell["b1"]["severe_harm_rate"],
                cell["b1"]["mean_g"], _bar(cell["best_visible_feature"]),
                "n/a" if best.get("lodo_mean_auc_oriented") is None
                else "%.6f" % best["lodo_mean_auc_oriented"],
                folds, cell["verdict"], cell["verdict_reason"]))
    lines += ["", "## Frozen-bin stump, utility over the whole served population",
              "",
              "| cell | treat-all utility | best stump | mean vs treat-all | "
              "mean vs treat-none |",
              "| --- | --- | --- | --- | --- |"]
    for key, cell in payload["main_table"].items():
        stump = cell["best_frozen_bin_stump"]
        if stump is None:
            continue
        rules = sorted({rule for rule in stump["rules_per_fold"].values() if rule})
        lines.append("| %s | %.6f | %s | %+.6f | %+.6f |" % (
            _bar(key), cell["b1"]["mean_g"], "; ".join(rules),
            stump["mean_vs_treat_all"], stump["mean_vs_treat_none"]))
    lines += ["", "## ORACLE row (severe_harm, pooled AUC)", "",
              "| cell | oracle feature | oracle AUC | best visible AUC | margin |",
              "| --- | --- | --- | --- | --- |"]
    for row in payload["structural_upper_bound"]["all_rows"]:
        lines.append("| %s | %s | %.6f | %.6f | %+.6f |" % (
            _bar(row["cell"]), row["oracle_feature"], row["oracle_pooled_auc"],
            row["best_visible_pooled_auc"], row["margin"]))
    lines += ["", "Structural verdict: **%s**"
              % payload["structural_upper_bound"]["verdict"], "",
              "## transport_flip", "",
              "| program | units | flip rate (all) | flip rate (support helped) | "
              "best visible | LODO mean | verdict | why |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for program, cell in payload["transport_flip"].items():
        best = cell["best_visible_reading"] or {}
        lines.append("| %s | %d | %.6f | %s | %s | %s | %s | %s |" % (
            _bar(program), cell["n_units"], cell["flip_rate_over_all_units"],
            "n/a" if cell["flip_rate_over_support_helped"] is None
            else "%.6f" % cell["flip_rate_over_support_helped"],
            cell["best_visible_feature"],
            "n/a" if best.get("lodo_mean_auc_oriented") is None
            else "%.6f" % best["lodo_mean_auc_oriented"],
            cell["verdict"], cell["verdict_reason"]))
    lines += ["", "## Not computable as defined", ""]
    for name, why in payload["definitions"]["not_computable"].items():
        lines.append("- `%s`: %s" % (name, why))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = _round(run())
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    boundary = payload["boundary"]
    print("fits=%d llm=%d held_out_reads=%d max_time_index_read=%d" % (
        boundary["consumer_fits"], boundary["llm_calls"],
        boundary["held_out_reads"], boundary["max_time_index_read"]))
    for key, cell in payload["main_table"].items():
        best = cell["best_visible_reading"] or {}
        print("%-46s %-24s best=%s lodo=%s" % (
            key, cell["verdict"], cell["best_visible_feature"],
            "n/a" if best.get("lodo_mean_auc_oriented") is None
            else "%.6f" % best["lodo_mean_auc_oriented"]))
    print("structural: %s" % payload["structural_upper_bound"]["verdict"])
    print("wrote %s" % OUT_JSON)
    print("wrote %s" % OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
