"""Verified inputs for freezing a second training cohort -- the freeze itself is a ruling.

P4D/P4G swept six origins over one training corpus: the anchors are frozen at
``[312 ... 852]`` and every one of them clears ``anchor + 48 <= origin`` past 900,
so the Program's intervention was identical at all six and a single fitted Ridge
was scored at six forecast windows.  Adding origins therefore adds evaluation
windows, not training conditions.  A genuine second condition needs different
training *series*.

P1 already reserves that supply: the structural filter takes ``[:20]`` and
``[20:40]`` as the target cell and ``[40:60]`` / ``[60:80]`` as a disjoint
selection cell.  This audit recomputes readability on the gapped variant and
reports whether the selection cell forms, how its gaps sit, and which origins
are evaluable on it -- the same three questions the target cell had to answer.

It freezes nothing and selects nothing.  Choosing the cohort and the anchor
block is a pre-registration act; this only makes sure the candidate is usable
before that ruling is made.  0 LLM calls, 0 Consumer fits, 0 held-out reads.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
    UndefinedSeasonalScale,
    seasonal_scale,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4j_candidate_cohort.json"

CONTEXT, HORIZON = preflight.CONTEXT, preflight.HORIZON
PERIOD, MIN_PAIRS = preflight.PERIOD, preflight.MIN_SEASONAL_PAIRS
SWEPT_ORIGINS = (1176, 1896, 2136, 2376, 2616, 2856)
# Candidate origins for the second cohort, spaced like the swept block and
# reaching further out; evaluability is measured, not assumed.
CANDIDATE_ORIGINS = (3096, 3336, 3576, 3816, 4056, 4296, 4536, 4776)


def longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def evaluability(variant: dict[str, np.ndarray], uids: Sequence[str],
                 origins: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for origin in origins:
        counts, undefined, fractions, runs = [], [], [], []
        too_short = [
            uid for uid in uids if variant[uid].size < origin + HORIZON
        ]
        if too_short:
            rows.append({
                "origin": int(origin), "usable": False,
                "reason": "series shorter than origin + horizon",
                "short_series": len(too_short),
            })
            continue
        for uid in uids:
            raw = variant[uid]
            truth = raw[origin:origin + HORIZON]
            count = int(np.isfinite(truth).sum())
            counts.append(count)
            gaps = ~np.isfinite(raw[origin - CONTEXT:origin + HORIZON])
            fractions.append(float(gaps.mean()))
            runs.append(longest_run(gaps))
            if count == 0:
                undefined.append({"uid": uid, "why": "no observed truth"})
                continue
            try:
                seasonal_scale(raw[:origin], np.isfinite(raw[:origin]),
                               period=PERIOD, min_pairs=MIN_PAIRS)
            except UndefinedSeasonalScale as exc:
                undefined.append({"uid": uid, "why": str(exc)[:80]})
        rows.append({
            "origin": int(origin),
            "usable": not undefined,
            "min_observed_truth": int(min(counts)),
            "median_observed_truth": int(np.median(counts)),
            "mean_missing_fraction": round(float(np.mean(fractions)), 4),
            "longest_gap_run": int(max(runs)),
            "not_evaluable_series": undefined,
        })
    return rows


def anchor_geometry(origins: Sequence[int]) -> dict[str, Any]:
    """Which anchors are live at which origin -- the fact that made six one."""
    anchors = [int(a) for a in forecast_p1._config()["anchors"]]
    live = {
        str(origin): [a for a in anchors if a + HORIZON <= int(origin)]
        for origin in origins
    }
    identical = len({tuple(v) for v in live.values()}) == 1
    return {
        "frozen_anchors": anchors,
        "live_anchors_by_origin": live,
        "identical_across_origins": identical,
        "reading": (
            "every anchor clears the filter at every candidate origin, so "
            "changing the origin alone still leaves the training corpus fixed; "
            "a second training condition must change the series or the anchors"
            if identical else "the live anchor set varies across these origins"
        ),
    }


def build() -> dict[str, Any]:
    variant = preflight.load_variant()
    anchors = [int(a) for a in forecast_p1._config()["anchors"]
               if a + HORIZON <= forecast_p1.ORIGIN]
    readable = [
        uid for uid in sorted(variant)
        if preflight._fit_readable(variant[uid], anchors)
    ]
    target = {"support_a": readable[:20], "support_b": readable[20:40]}
    selection = {"support_a": readable[40:60], "support_b": readable[60:80]}
    overlap = sorted(
        set(target["support_a"] + target["support_b"])
        & set(selection["support_a"] + selection["support_b"])
    )
    cohort_uids = selection["support_a"] + selection["support_b"]

    return {
        "stage": "P4J_CANDIDATE_COHORT",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_SUPPLY_CHECK",
        "data_version": preflight.DATA_VERSION,
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "anything_frozen_by_this_audit": False,
        },
        "why": (
            "adding origins adds evaluation windows, not training conditions; "
            "a second cohort is what makes a new corpus and a new fitted model"
        ),
        "structurally_readable": len(readable),
        "cohort_1_target_cell": {
            **target,
            "formable": len(readable) >= 40,
            "used_by": "P4D / P4F / P4G",
        },
        "cohort_2_selection_cell": {
            **selection,
            "formable": len(readable) >= 80,
            "disjoint_from_cohort_1": not overlap,
            "overlap": overlap,
        },
        "anchor_geometry": anchor_geometry(
            list(SWEPT_ORIGINS) + list(CANDIDATE_ORIGINS)
        ),
        "cohort_2_evaluability_on_swept_origins": evaluability(
            variant, cohort_uids, SWEPT_ORIGINS
        ),
        "cohort_2_evaluability_on_candidate_origins": evaluability(
            variant, cohort_uids, CANDIDATE_ORIGINS
        ),
        "what_still_needs_a_ruling": [
            "which cohort is the primary validation cohort",
            "whether an anchor block disjoint from [312...852] is also declared, "
            "as the optional temporal-robustness axis",
            "the origin list, frozen before any O1 arm runs",
            "that the result is labelled development cohort holdout, not "
            "Final/held-out, with the training cohort/face as the statistical "
            "unit and origins as repeated evaluation points inside it",
        ],
        "releases": "NONE",
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    second = report["cohort_2_selection_cell"]
    print("structurally readable : %d" % report["structurally_readable"])
    print("cohort 2 formable     : %s | disjoint from cohort 1: %s" % (
        second["formable"], second["disjoint_from_cohort_1"]))
    print("cohort 2 A            : %s ..." % second["support_a"][:5])
    print("cohort 2 B            : %s ..." % second["support_b"][:5])
    print("anchors identical across all listed origins: %s"
          % report["anchor_geometry"]["identical_across_origins"])
    for label in ("cohort_2_evaluability_on_swept_origins",
                  "cohort_2_evaluability_on_candidate_origins"):
        print("--- %s" % label.replace("cohort_2_evaluability_on_", ""))
        for row in report[label]:
            if not row.get("usable") and "reason" in row:
                print("   %6d  UNUSABLE (%s)" % (row["origin"], row["reason"]))
                continue
            print("   %6d  usable=%-5s miss %.3f  min truth %2d  run %3d  "
                  "not-evaluable %d" % (
                      row["origin"], row["usable"], row["mean_missing_fraction"],
                      row["min_observed_truth"], row["longest_gap_run"],
                      len(row["not_evaluable_series"])))
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
