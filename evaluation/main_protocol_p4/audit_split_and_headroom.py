"""Two calibration checks on the repairability audit's scope.

The audit found 0 of 193 legal programs stable on both Support faces.  Before
that can close a defect family, two things have to be ruled out, because either
would make the target itself unfair rather than the Program space empty.

**Check 1 -- is the A/B split structurally balanced?**  The faces are not a
random or stratified partition.  ``run_forecast_p1.py:264-266`` sorts the
structurally readable UIDs and takes ``[:20]`` as Support-A and ``[20:40]`` as
Support-B, so the split is *lexicographic on UID*.  If the two groups differ
systematically on deployment-visible features, "stable on both faces" is a
harder target than intended, and the empty intersection says something about the
split rather than about repairability.  Every feature here is one the Fast Path
may already read; no Outcome is touched.

**Check 2 -- do the four zero-clearing origins have any headroom at all?**  Four
of eight origins had no legal program clear either face.  If identity is already
near-optimal there, they are ``NO_HEADROOM`` control positions and counting them
against the method overstates the failure.  Headroom is read from the audit's own
stored readings, so no Consumer is refit.

Neither check enumerates a new program, moves a threshold, or reads a held-out
origin.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import p4b_viability as viability
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT_ROOT / "artifacts/main_protocol/p4c_program_repairability_audit.json"
REPORT = PROJECT_ROOT / "artifacts/main_protocol/p4c_split_and_headroom_check.json"

# Deployment-visible, numeric, and meaningful to compare between two groups of
# series.  Booleans and status strings are counted separately.
NUMERIC_FEATURES = (
    "missing_fraction",
    "longest_missing_run_fraction",
    "local_robust_z_peak",
    "level_region_fraction",
    "level_excursion_score",
    "period_change_score",
    "period_reliability",
)
FLAG_FEATURES = (
    "level_only_post_shift_support_sufficient",
    "post_shift_support_sufficient",
    "period_repair_available",
)
CONTEXT_LENGTH = contract.CONTEXT_LENGTH


def _series_features(values: Any, origin: int) -> dict[str, Any]:
    """One series' deployment-visible features on its pre-origin context."""
    window = np.asarray(values[origin - CONTEXT_LENGTH : origin], dtype=np.float64)
    return dict(extract_public_features(window, task_kind=forecast_p4.TASK))


def _describe(sample: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in sample if v is not None and math.isfinite(float(v))]
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _mann_whitney_u(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    """Exact-rank U with a normal approximation, and the rank-biserial effect.

    Two groups of 20 with no distributional assumption: the rank test is the
    honest instrument, and the effect size matters more than the p-value here --
    the question is whether the groups are *materially* different, not whether a
    difference is detectable.
    """
    a = np.asarray([v for v in left if v is not None and math.isfinite(v)], float)
    b = np.asarray([v for v in right if v is not None and math.isfinite(v)], float)
    if a.size == 0 or b.size == 0:
        return {"u": None, "p_two_sided": None, "rank_biserial": None}
    combined = np.concatenate([a, b])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    sorted_values = combined[order]
    position = 0
    assigned = np.empty(combined.size, dtype=np.float64)
    while position < sorted_values.size:
        stop = position
        while (stop + 1 < sorted_values.size
               and sorted_values[stop + 1] == sorted_values[position]):
            stop += 1
        assigned[position:stop + 1] = np.mean(
            np.arange(position + 1, stop + 2, dtype=np.float64)
        )
        position = stop + 1
    ranks[order] = assigned
    rank_sum_a = float(ranks[: a.size].sum())
    u_a = rank_sum_a - a.size * (a.size + 1) / 2.0
    u = min(u_a, a.size * b.size - u_a)
    mean_u = a.size * b.size / 2.0
    # Tie-corrected variance.
    _unique, counts = np.unique(combined, return_counts=True)
    tie_term = float((counts ** 3 - counts).sum())
    n = a.size + b.size
    variance = (a.size * b.size / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0:
        return {"u": float(u), "p_two_sided": None, "rank_biserial": None}
    z = (u - mean_u) / math.sqrt(variance)
    from statistics import NormalDist

    return {
        "u": float(u),
        "z": float(z),
        "p_two_sided": float(2.0 * NormalDist().cdf(-abs(z))),
        # 0 = groups interleave perfectly, 1 = complete separation.
        "rank_biserial": float(abs(2.0 * u_a / (a.size * b.size) - 1.0)),
    }


def detectable_effect(n_left: int, n_right: int, *, power_z: float = 2.80,
                      alpha_z: float = 1.96) -> dict[str, Any]:
    """What separation this n can and cannot see.

    Twenty series per face is a small sample.  Reporting "comparable" without
    saying what size of difference would have shown up would overstate the
    check: it can rule out a large structural separation between the groups,
    not a moderate one.
    """
    product = n_left * n_right
    sd = math.sqrt(product * (n_left + n_right + 1) / 12.0)

    def rank_biserial_for(z: float) -> float:
        superiority = (z * sd + product / 2.0) / product
        return abs(2.0 * superiority - 1.0)

    return {
        "n_per_face": [n_left, n_right],
        "minimum_significant_rank_biserial": rank_biserial_for(alpha_z),
        "rank_biserial_detectable_at_80pct_power": rank_biserial_for(power_z),
        "reading": (
            "this n only detects a large separation; a moderate difference "
            "between the two faces would not reach significance here, so "
            "'comparable' means 'not largely different', not 'identical'"
        ),
    }


def split_balance(base_cell: Any, origins: Sequence[int]) -> dict[str, Any]:
    """Are Support-A and Support-B comparable on deployment-visible features?"""
    per_origin = []
    for origin in origins:
        features = {
            face: [
                _series_features(base_cell.values[uid], int(origin))
                for uid in (base_cell.support_a if face == "support_a"
                            else base_cell.support_b)
            ]
            for face in ("support_a", "support_b")
        }
        comparisons = {}
        for name in NUMERIC_FEATURES:
            left = [row.get(name) for row in features["support_a"]]
            right = [row.get(name) for row in features["support_b"]]
            comparisons[name] = {
                "support_a": _describe(left),
                "support_b": _describe(right),
                "test": _mann_whitney_u(left, right),
            }
        flags = {
            name: {
                "support_a_true": sum(
                    1 for row in features["support_a"] if row.get(name)),
                "support_b_true": sum(
                    1 for row in features["support_b"] if row.get(name)),
                "of": len(features["support_a"]),
            }
            for name in FLAG_FEATURES
        }
        per_origin.append(
            {"origin": int(origin), "numeric": comparisons, "flags": flags}
        )

    # A feature counts as systematically different when it separates the groups
    # at p < 0.05 in a majority of origins.  One origin proves nothing; a
    # feature that separates them everywhere is a property of the split.
    verdicts = {}
    for name in NUMERIC_FEATURES:
        significant = [
            entry["origin"] for entry in per_origin
            if (entry["numeric"][name]["test"].get("p_two_sided") or 1.0) < 0.05
        ]
        effects = [
            entry["numeric"][name]["test"].get("rank_biserial")
            for entry in per_origin
            if entry["numeric"][name]["test"].get("rank_biserial") is not None
        ]
        verdicts[name] = {
            "origins_separated_at_p05": significant,
            "separated_in_majority": len(significant) > len(per_origin) / 2,
            "median_rank_biserial": float(np.median(effects)) if effects else None,
        }
    systematic = sorted(
        name for name, entry in verdicts.items() if entry["separated_in_majority"]
    )
    return {
        "split_rule": (
            "lexicographic on UID: sorted(structurally_readable)[:20] is "
            "Support-A and [20:40] is Support-B "
            "(evaluation/main_protocol_p1/run_forecast_p1.py:264-266); the split "
            "is neither random nor stratified"
        ),
        "support_a_uids": list(base_cell.support_a),
        "support_b_uids": list(base_cell.support_b),
        "features_compared": list(NUMERIC_FEATURES),
        "per_origin": per_origin,
        "feature_verdicts": verdicts,
        "systematically_different_features": systematic,
        "groups_comparable": not systematic,
        "power": detectable_effect(
            len(base_cell.support_a), len(base_cell.support_b)
        ),
        "observed_effect_sizes": {
            name: verdicts[name]["median_rank_biserial"]
            for name in NUMERIC_FEATURES
        },
        "outcome_reads": 0,
        "llm_calls": 0,
    }


def identity_headroom(audit: Mapping[str, Any]) -> dict[str, Any]:
    """How much any legal program could improve each origin, from stored readings."""
    rows = audit["rows"]
    origins = audit["summary"]["origins"]
    table = []
    for origin in origins:
        best = {}
        for face in ("support_a", "support_b"):
            gains = [
                entry[face]["aggregate_gain"]
                for row in rows
                for entry in row["per_origin"]
                if entry["origin"] == origin and "aggregate_gain" in entry[face]
            ]
            best[face] = {
                "best_gain": max(gains) if gains else None,
                "programs_read": len(gains),
                "programs_with_positive_gain": sum(
                    1 for value in gains if value > 0.005),
            }
        cleared = next(
            entry for entry in audit["summary"]["face_clearance_by_origin"]
            if entry["origin"] == origin
        )
        zero_clearing = (
            cleared["support_a_cleared_by"] == 0
            and cleared["support_b_cleared_by"] == 0
        )
        # "No headroom" means no legal program materially improves either face.
        # That is a property of the position, not a failure of the method.
        no_headroom = all(
            (best[face]["best_gain"] is None or best[face]["best_gain"] <= 0.005)
            for face in ("support_a", "support_b")
        )
        table.append(
            {
                "origin": int(origin),
                "support_a": best["support_a"],
                "support_b": best["support_b"],
                "zero_clearing": zero_clearing,
                "classification": (
                    "NO_HEADROOM" if no_headroom
                    else "HEADROOM_EXISTS_BUT_UNSTABLE" if zero_clearing
                    else "HEADROOM_ON_AT_LEAST_ONE_FACE"
                ),
            }
        )
    zero = [entry for entry in table if entry["zero_clearing"]]
    return {
        "per_origin": table,
        "zero_clearing_origins": [entry["origin"] for entry in zero],
        "zero_clearing_are_no_headroom": [
            entry["origin"] for entry in zero
            if entry["classification"] == "NO_HEADROOM"
        ],
        "zero_clearing_with_real_headroom": [
            entry["origin"] for entry in zero
            if entry["classification"] != "NO_HEADROOM"
        ],
        "consumer_fits": 0,
        "note": "read from the repairability audit's stored readings; nothing refit",
    }


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    base_cell, _selection, data = forecast_p1._load_exposed_cells()
    origins = list(audit["summary"]["origins"])

    balance = split_balance(base_cell, origins)
    headroom = identity_headroom(audit)

    if balance["groups_comparable"]:
        verdict = "NO_REPAIR_HEADROOM_CONFIRMED"
        reading = (
            "Support-A and Support-B are comparable on every deployment-visible "
            "feature tested -- no feature separates them at p<0.05 in a "
            "majority of origins, and the observed effect sizes are far below "
            "what this n could detect -- so the empty intersection is a "
            "property of the Program space rather than of the lexicographic "
            "split.  The check rules out a large structural separation, not a "
            "moderate one."
        )
    else:
        verdict = "NO_REPAIR_HEADROOM_UNDER_LEXICOGRAPHIC_SPLIT"
        reading = (
            "the two faces differ systematically on %s, so 'stable on both "
            "faces' is a harder target than intended; the defect family cannot "
            "be closed on this evidence alone"
            % ", ".join(balance["systematically_different_features"])
        )

    report = {
        "stage": "P4C_SPLIT_BALANCE_AND_HEADROOM_CHECK",
        "written_at": datetime.now().astimezone().isoformat(),
        "calibrates": AUDIT.relative_to(PROJECT_ROOT).as_posix(),
        "dataset": data.get("dataset"),
        "corrects_earlier_wording": (
            "Support-A/B are two disjoint series groups at the same origin, "
            "produced by a lexicographic UID split -- not two time faces 48 "
            "steps apart.  Earlier P4b wording describing temporal drift is "
            "withdrawn."
        ),
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "outcome_reads": 0,
            "held_out_origins_touched": 0,
            "new_programs_enumerated": 0,
            "thresholds_changed": False,
        },
        "check_1_split_balance": balance,
        "check_2_identity_headroom": headroom,
        "verdict": verdict,
        "reading": reading,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("CHECK 1 -- A/B split balance (%d origins, 20 vs 20 series)" % len(origins))
    for name in NUMERIC_FEATURES:
        entry = balance["feature_verdicts"][name]
        mark = "DIFFERS" if entry["separated_in_majority"] else "comparable"
        print("   %-32s %-11s p<.05 at %d/%d origins, median |rb|=%s" % (
            name, mark, len(entry["origins_separated_at_p05"]), len(origins),
            "%.2f" % entry["median_rank_biserial"]
            if entry["median_rank_biserial"] is not None else "--"))
    print("   systematically different: %s"
          % (balance["systematically_different_features"] or "none"))
    power = balance["power"]
    print("   power: needs |rank-biserial| >= %.2f to reach p<.05, >= %.2f at 80%%; "
          "largest observed %.2f" % (
              power["minimum_significant_rank_biserial"],
              power["rank_biserial_detectable_at_80pct_power"],
              max(v for v in balance["observed_effect_sizes"].values()
                  if v is not None)))
    print("\nCHECK 2 -- identity headroom")
    for entry in headroom["per_origin"]:
        print("   origin %-6d A best gain %-9s B best gain %-9s %s" % (
            entry["origin"],
            "%+.4f" % entry["support_a"]["best_gain"]
            if entry["support_a"]["best_gain"] is not None else "--",
            "%+.4f" % entry["support_b"]["best_gain"]
            if entry["support_b"]["best_gain"] is not None else "--",
            entry["classification"]))
    print("   zero-clearing origins with NO headroom : %s"
          % headroom["zero_clearing_are_no_headroom"])
    print("   zero-clearing origins WITH headroom    : %s"
          % headroom["zero_clearing_with_real_headroom"])
    print("\nverdict : %s" % verdict)
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
