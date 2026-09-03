"""Where the natural gaps actually are -- and why none of them reached P1-P4c.

The Forecast line's artifacts all label their dataset "KDD Cup 2018 with
missing values", but ``data/kdd2018/series_cache.npz`` was built from
``kdd_cup_2018_dataset_without_missing_values.tsf``: the cache holds zero NaN.
Every imputation operator in the eligible menu has therefore been probed against
a series that had nothing to impute, and the seven missingness features in the
deployment-visible risk audit were constants (grouped AUC exactly 0.500).

This audit reads the *with*-missing archive that already sits beside it in
``raw/`` and reports the gap geometry per origin.  It writes no cache, rebuilds
no roster and touches no existing artifact: swapping the data source changes
every identity baseline, so that is a separate decision, not a side effect of
measuring.  0 LLM calls, 0 Consumer fits, 0 held-out reads.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WITH_MISSING = (
    PROJECT_ROOT / "data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip"
)
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4d_natural_gap_roster.json"

CONTEXT, HORIZON = 192, 48
P4B_ORIGINS = (1176, 1416, 1656, 1896, 2136, 2376, 2616, 2856)
OLD_P4_ORIGINS = (600, 648, 744, 792, 840, 888, 936)
ROSTER_SIZE = 40  # the P1 target cell: Support-A [:20] + Support-B [20:40]


def missing_masks() -> dict[str, np.ndarray]:
    """Per-UID boolean gap mask from the with-missing .tsf inside the archive."""
    with zipfile.ZipFile(WITH_MISSING) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            stream = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            masks: dict[str, np.ndarray] = {}
            in_data = False
            for line in stream:
                if not in_data:
                    in_data = line.strip().lower() == "@data"
                    continue
                fields = line.rstrip().split(":")
                masks[fields[0]] = np.array(
                    [value.strip() == "?" for value in fields[-1].split(",")],
                    dtype=bool,
                )
    return masks


def longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def alignment(masks: dict[str, np.ndarray]) -> dict[str, Any]:
    """The swap is only drop-in if UIDs and lengths correspond one-to-one."""
    cache = np.load(CACHE, allow_pickle=True)
    names = [str(value) for value in cache["names"]]
    values = cache["values"]
    present = [name for name in names if name in masks]
    same_length = [
        name for name in present
        if masks[name].size == np.asarray(values[names.index(name)]).size
    ]
    nan_in_cache = int(
        sum(
            int(np.isnan(np.asarray(row, dtype=np.float64)).sum())
            for row in values
        )
    )
    return {
        "cache_series": len(names),
        "uid_overlap": len(present),
        "length_match": len(same_length),
        "drop_in_swappable": len(present) == len(names) == len(same_length),
        "nan_points_in_current_cache": nan_in_cache,
        "cache_built_from": "kdd_cup_2018_dataset_without_missing_values.tsf",
        "artifacts_label_it": "KDD Cup 2018 with missing values",
        "label_is_wrong": nan_in_cache == 0,
    }


def per_origin(masks: dict[str, np.ndarray], uids: list[str],
               origins: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    for origin in origins:
        windows = [masks[uid][origin - CONTEXT:origin + HORIZON] for uid in uids]
        fractions = np.array([window.mean() for window in windows])
        runs = [longest_run(window) for window in windows]
        rows.append(
            {
                "origin": int(origin),
                "series": len(uids),
                "mean_missing_fraction": round(float(fractions.mean()), 4),
                "max_series_missing_fraction": round(float(fractions.max()), 4),
                "series_with_any_gap": int((fractions > 0).sum()),
                "longest_gap_run": int(max(runs)),
                "context_only_mean": round(
                    float(np.mean([w[:CONTEXT].mean() for w in windows])), 4
                ),
                "horizon_only_mean": round(
                    float(np.mean([w[CONTEXT:].mean() for w in windows])), 4
                ),
            }
        )
    return rows


def build() -> dict[str, Any]:
    masks = missing_masks()
    align = alignment(masks)
    total = int(sum(mask.size for mask in masks.values()))
    gaps = int(sum(int(mask.sum()) for mask in masks.values()))
    # Sorted-UID order is the order P1's structural filter walks; the first 40
    # readable UIDs form the target cell.  Readability is recomputed on whatever
    # data is actually loaded, so this is the roster's shape, not its identity.
    uids = sorted(masks)[:ROSTER_SIZE]
    return {
        "stage": "P4D_NATURAL_GAP_ROSTER",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_DATA_INVENTORY",
        "data_role": "EXPOSED_DEVELOPMENT",
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "caches_rebuilt": 0,
            "existing_artifacts_modified": 0,
        },
        "source_alignment": align,
        "corpus": {
            "series": len(masks),
            "points": total,
            "missing_points": gaps,
            "missing_fraction": round(gaps / total, 5),
            "series_with_any_gap": sum(
                1 for mask in masks.values() if bool(mask.any())
            ),
        },
        "roster_uids": uids,
        "p4b_p4c_origins": per_origin(masks, uids, P4B_ORIGINS),
        "old_p4_origins": per_origin(masks, uids, OLD_P4_ORIGINS),
        "verdict": (
            "NATURAL_GAPS_ARE_ABUNDANT_BUT_WERE_REMOVED_UPSTREAM"
            if align["label_is_wrong"] else "CACHE_ALREADY_CARRIES_GAPS"
        ),
        "what_this_does_not_claim": [
            "does not supersede any P1-P4c number: those were measured "
            "correctly on the without-missing cache and remain valid for it",
            "does not assert the swapped roster keeps the same 80 readable "
            "UIDs; P1's structural filter must be rerun on gapped data",
            "does not measure any imputation gain; no Consumer was fit",
        ],
        "releases": "NONE",
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    align, corpus = report["source_alignment"], report["corpus"]
    print("drop-in swappable : %s (%d/%d UIDs, %d/%d lengths)" % (
        align["drop_in_swappable"], align["uid_overlap"], align["cache_series"],
        align["length_match"], align["cache_series"]))
    print("current cache NaN : %d" % align["nan_points_in_current_cache"])
    print("corpus gaps       : %d / %d (%.3f%%), %d/%d series affected" % (
        corpus["missing_points"], corpus["points"],
        100 * corpus["missing_fraction"], corpus["series_with_any_gap"],
        corpus["series"]))
    print("%8s %10s %10s %9s %8s" % (
        "origin", "mean miss", "max miss", "with gap", "run"))
    for row in report["p4b_p4c_origins"]:
        print("%8d %10.4f %10.4f %9d %8d" % (
            row["origin"], row["mean_missing_fraction"],
            row["max_series_missing_fraction"], row["series_with_any_gap"],
            row["longest_gap_run"]))
    print("verdict           : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
