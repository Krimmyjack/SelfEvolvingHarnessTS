"""Verified inputs for freezing the main experiment's cohorts -- not the freeze.

Static / A3 / A5 needs three things cohorts 1 and 2 cannot supply: a Target the
Scope machinery has never been fitted on, a Source whose Skills can be audited
into it, and a held-out block for that Target that stays closed until the arms
are frozen.  Choosing them is a pre-registration act.  Measuring whether they
*exist* is not, and doing it first means the ruling is made against readings
rather than against hope.

Three questions per candidate cohort, the same three the target cell had to
answer: does a 20/20 two-face cell form on gapped data, is it disjoint from
everything already read, and which origins can actually be scored on it.

The full structurally-readable list is persisted this time.  Recomputing it
costs minutes of ``_fit_readable`` on 270 series, and every later cohort
question needs the same list.

0 LLM calls, 0 Consumer fits, 0 held-out reads.  Nothing is frozen here.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import audit_candidate_cohort as cohorts
from evaluation.main_protocol_p4 import phase2_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4s_main_experiment_supply.json"

CONTEXT, HORIZON = preflight.CONTEXT, preflight.HORIZON

#: Cohorts 1 and 2 are spent: development and confirmation respectively.
SPENT = {"cohort_1": (0, 40), "cohort_2": (40, 80)}
#: What the main experiment would need, in the order P1's filter yields them.
CANDIDATES = {
    "cohort_3_target": (80, 120),
    "cohort_4_source": (120, 160),
}
#: Development origins already read on this data version.
READ_ORIGINS = tuple(contract.ORIGINS) + (1416, 1656, 1896)
#: Candidates for the Target's held-in and held-out blocks.  Evaluability is
#: measured; the split between them is a ruling, not a computation.
CANDIDATE_ORIGINS = (3096, 3336, 3576, 3816, 4056, 4296, 4536, 4776, 5016, 5256)


def readable_uids(variant: dict[str, np.ndarray]) -> list[str]:
    anchors = [
        int(a) for a in forecast_p1._config()["anchors"]
        if a + HORIZON <= forecast_p1.ORIGIN
    ]
    return [
        uid for uid in sorted(variant)
        if preflight._fit_readable(variant[uid], anchors)
    ]


def cohort(readable: Sequence[str], span: tuple[int, int]) -> dict[str, Any]:
    start, stop = span
    block = list(readable[start:stop])
    return {
        "slice": "readable[%d:%d]" % span,
        "support_a": block[:20],
        "support_b": block[20:40],
        "formable": len(block) >= 40,
    }


def build() -> dict[str, Any]:
    variant = preflight.load_variant()
    readable = readable_uids(variant)
    spent = {name: cohort(readable, span) for name, span in SPENT.items()}
    spent_uids = {
        uid for entry in spent.values()
        for uid in entry["support_a"] + entry["support_b"]
    }

    proposed: dict[str, Any] = {}
    for name, span in CANDIDATES.items():
        entry = cohort(readable, span)
        uids = entry["support_a"] + entry["support_b"]
        entry["disjoint_from_spent_cohorts"] = not (set(uids) & spent_uids)
        entry["evaluable_on_read_origins"] = cohorts.evaluability(
            variant, uids, sorted(READ_ORIGINS))
        entry["evaluable_on_candidate_origins"] = cohorts.evaluability(
            variant, uids, CANDIDATE_ORIGINS)
        entry["usable_read_origins"] = [
            row["origin"] for row in entry["evaluable_on_read_origins"]
            if row.get("usable")
        ]
        entry["usable_candidate_origins"] = [
            row["origin"] for row in entry["evaluable_on_candidate_origins"]
            if row.get("usable")
        ]
        proposed[name] = entry

    target = proposed.get("cohort_3_target", {})
    source = proposed.get("cohort_4_source", {})
    shared = sorted(
        set(target.get("usable_candidate_origins") or ())
        & set(source.get("usable_candidate_origins") or ())
    )
    return {
        "stage": "P4S_MAIN_EXPERIMENT_SUPPLY",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_SUPPLY_CHECK",
        "data_version": contract.DATA_VERSION,
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "anything_frozen_by_this_audit": False,
        },
        "structurally_readable": len(readable),
        "readable_uids": readable,
        "spent_cohorts": {
            name: {**entry, "role": (
                "DEVELOPMENT" if name == "cohort_1" else "DEVELOPMENT_CONFIRMATION")}
            for name, entry in spent.items()
        },
        "proposed_cohorts": proposed,
        "origins": {
            "already_read_on_this_data_version": sorted(READ_ORIGINS),
            "candidates_measured": list(CANDIDATE_ORIGINS),
            "usable_on_both_proposed_cohorts": shared,
            "note": (
                "the split of these into Target held-in and held-out is a "
                "ruling; this audit only reports which can be scored at all"
            ),
        },
        "what_still_needs_a_ruling": [
            "which cohort is the Target and which is the Source",
            "how the usable origins split into Target held-in and held-out",
            "that held-out stays closed until Static/A3/A5 are frozen",
            "the LLM budget per arm, and that A3 and A5 receive the same one",
        ],
        "reminder": (
            "adding origins does not add training conditions: the anchors are "
            "frozen at [312...852] and all of them clear anchor+48<=origin past "
            "900, so a new cohort is what makes a new corpus and a new model"
        ),
        "releases": "NONE",
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("structurally readable : %d" % report["structurally_readable"])
    for name, entry in report["proposed_cohorts"].items():
        print("--- %s (%s)" % (name, entry["slice"]))
        print("    formable %s | disjoint from spent %s" % (
            entry["formable"], entry["disjoint_from_spent_cohorts"]))
        print("    A: %s ..." % entry["support_a"][:4])
        print("    usable read origins      : %s" % entry["usable_read_origins"])
        print("    usable candidate origins : %s"
              % entry["usable_candidate_origins"])
    print("usable on both proposed cohorts: %s"
          % report["origins"]["usable_on_both_proposed_cohorts"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
