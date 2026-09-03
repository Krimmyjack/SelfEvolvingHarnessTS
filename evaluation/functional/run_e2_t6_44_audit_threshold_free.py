"""#44-audit -- threshold-free reread of the M0-C IForest and PCA arms.

#44a-r2 showed that aegists_iforest_v1's contamination=0.1 freezes the
Query threshold among ordinary windows on this near-clean training block,
and that any training-side operation that changes the outlier rate first
moves that threshold.  M0-C's four-program all-negative event-F1 reading
therefore has a competing explanation: the programs may have been judged
on a displaced threshold rather than on ranking quality.  The PCA
Consumer's threshold is the training residual 0.90 quantile -- the same
family of artefact.

This book rereads the same 24 EXPOSED series x five-program menu on the
two threshold-bearing Consumers, using only the continuous anomaly score
and the existing pointwise AUPRC.  No new threshold is introduced.  The
in-service contamination / residual-quantile parameters are not touched.
The supervised v3 arm is not rerun: its mechanism is positive-row
erosion, independently measured and threshold-free already.

The exam, frozen
----------------
Roster / split / menu / acting bytes: identical to M0-C.
Readout: development_exposed_eval [int(0.7n), n) against real Yahoo
point labels.  Programs act on the held-in block only; the Query array
handed to score_series is the untouched raw series.

Scores (no threshold enters the ranking):

  * C-a IForest: existing anomaly_scores = -decision_function(window).
    decision_function = score_samples - offset_, a per-model constant,
    so the ranking equals score_samples.  Flags (decision < 0) are
    used only for the companion event-F1, which is taken from the same
    fit and is not a judgment input.
  * C-c PCA: existing anomaly_scores = window RMS residual, mapped to
    the window-ending point by score_region's established alignment.
    The frozen residual quantile is used only for companion event-F1.

AUPRC is the existing consumers.aegists_iforest_v1.auprc -- stepwise
average precision, None when the eval region has no positive label.
Zero-event series (real_14 / real_18 class) are listed and excluded
from every macro.  Identity AUPRC is a NEW ANCHOR: no prior AUPRC
anchor exists on this roster.

Judgment, pre-registered, gates not rewritten after seeing numbers:

  * THRESHOLD_ARTIFACT_CONFIRMED -- all four cleaning programs' AUPRC
    macro Δ fall in ±0.005
  * DATA_HARM_CONFIRMED -- some program has AUPRC macro Δ ≤ −0.005
    and harmed series > 2/24
  * MIXED -- some programs artefact, some confirmed harm; or leftover
    (outside the artefact band but no program meets the harm bar),
    annotated per program
  * INSTRUMENT_UNREADABLE -- AUPRC uncomputable on a majority of the
    event-bearing series

Usage:
  python evaluation/functional/run_e2_t6_44_audit_threshold_free.py --run
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from consumers import aegists_iforest_v1 as iforest  # noqa: E402
from consumers import pca_reconstruction_v1 as pca  # noqa: E402

import run_e2_t6_natural_a5_a3 as t6  # noqa: E402

PROTOCOL_VERSION = "t6_44_audit_threshold_free_v1"
RUN_ID = "yahoo_m44_audit_v1"

DATA_ROOT = PROJECT_ROOT / "data" / "benchmark_yahoo_s5_v1"
WORK_DIR = DATA_ROOT / "work"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_44_audit_threshold_free.json"
OUT_MD = E2 / "t6_44_audit_threshold_free.md"
M0C = E2 / "t6_m0c_consumer_flip.json"

PROGRAMS: tuple[str, ...] = t6.PROGRAMS
CLEANING: tuple[str, ...] = tuple(p for p in PROGRAMS if p != "identity")
CONSUMERS: tuple[str, ...] = ("c_a_iforest", "c_c_pca")

FIT_CAP = 280
MATERIAL = 0.005
HARMED_MAX_FOR_ARTIFACT = 2
ANCHOR_TOLERANCE = 1e-12
MAJORITY_UNREADABLE = 0.50


class Stop(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__("%s: %s" % (code, detail))
        self.code = code
        self.detail = detail


class FitBudget:
    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0
        self.by_arm: dict[str, int] = {}

    def spend(self, arm: str, n: int = 1) -> None:
        if self.used + n > self.cap:
            raise Stop("CONSUMER_FIT_BUDGET_EXCEEDED",
                       "AD fit budget exhausted at %d" % self.cap)
        self.used += n
        self.by_arm[arm] = self.by_arm.get(arm, 0) + n


def _json_text(doc: Any) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fnum(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


# =========================================================================== #
# Part A -- substrate and one exam
# =========================================================================== #
def load_substrate() -> dict[str, Any]:
    """The 24 EXPOSED series through the canonical loader.  Read only."""
    pack = t6._load_yahoo_l1_roster()
    rows = pack["rows"]
    order = list(pack["order"])
    if len(order) != 24:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "expected the 24 EXPOSED series, got %d" % len(order))
    freeze_n = len(pack["freeze"]["roster"])
    if freeze_n < 65:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "yahoo freeze roster shorter than expected: %d" % freeze_n)
    vault_out = DATA_ROOT / "vaults" / "held_out"
    series: dict[str, Any] = {}
    for uid in order:
        rec = rows[uid]
        n = int(rec["length"])
        cut = int(rec["windows"]["heldout"][0])
        raw = np.asarray(rec["values"], dtype=np.float64)
        series[uid] = {
            "n": n,
            "cut": cut,
            "raw": raw,
            "work_path": WORK_DIR / uid,
            "eval_events": t6._point_events_from_vault(vault_out / uid, cut, n),
        }
    return {"order": order, "series": series, "freeze_roster_n": freeze_n}


def _score_cell(module: Any, raw: np.ndarray, cut: int, n: int,
                program: str, eval_events: list[list[int]]) -> dict[str, Any]:
    """One fit on the prepared held-in block; score the raw Query."""
    train = t6._apply_program(raw[:cut], program)
    model = module.fit_series(train)
    reading = module.score_series(model, raw, (cut, n), eval_events)
    return {
        "auprc": _fnum(reading.get("auprc")),
        "f1": float(reading["f1"]),
        "precision": float(reading["precision"]),
        "recall": float(reading["recall"]),
        "truth_events": int(reading["truth_events"]),
        "predicted_events": int(reading["predicted_events"]),
        "matched_events": int(reading["matched_events"]),
        "flagged_points": int(reading["flagged_points"]),
        "scored_points": int(reading["scored_points"]),
        "zero_scale": bool(reading.get("zero_scale")),
    }


def exam(substrate: dict[str, Any], budget: FitBudget) -> dict[str, Any]:
    """24 x 5 x 2 fits.  Programs never touch eval-region bytes."""
    order = substrate["order"]
    series = substrate["series"]
    modules = {"c_a_iforest": iforest, "c_c_pca": pca}
    cells: dict[str, Any] = {arm: {uid: {} for uid in order}
                             for arm in CONSUMERS}
    for arm, module in modules.items():
        for uid in order:
            rec = series[uid]
            raw = rec["raw"]
            cut, n = int(rec["cut"]), int(rec["n"])
            events = rec["eval_events"]
            for program in PROGRAMS:
                budget.spend(arm)
                cells[arm][uid][program] = _score_cell(
                    module, raw, cut, n, program, events)
    return cells


def _event_bearing(cells: dict[str, Any], order: list[str]) -> list[str]:
    """A series is event-bearing when identity eval has at least one event.

    Taken from C-a identity; C-c sees the same truth windows.
    """
    return [uid for uid in order
            if int(cells["c_a_iforest"][uid]["identity"]["truth_events"]) > 0]


def _zero_event(cells: dict[str, Any], order: list[str]) -> list[str]:
    return [uid for uid in order if uid not in _event_bearing(cells, order)]


def _attach_deltas(cells: dict[str, Any], order: list[str]) -> None:
    for arm in CONSUMERS:
        for uid in order:
            base_a = cells[arm][uid]["identity"]["auprc"]
            base_f = float(cells[arm][uid]["identity"]["f1"])
            for program in PROGRAMS:
                row = cells[arm][uid][program]
                auprc = row["auprc"]
                row["auprc_delta"] = (
                    None if auprc is None or base_a is None
                    else float(auprc) - float(base_a))
                row["f1_delta"] = float(row["f1"]) - base_f


def _macro(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def aggregate(cells: dict[str, Any], order: list[str]) -> dict[str, Any]:
    bearing = _event_bearing(cells, order)
    zero = _zero_event(cells, order)
    out: dict[str, Any] = {}
    for arm in CONSUMERS:
        programs: dict[str, Any] = {}
        for program in PROGRAMS:
            auprcs = []
            deltas = []
            f1s = []
            f1_deltas = []
            harmed: list[str] = []
            improved: list[str] = []
            unreadable: list[str] = []
            for uid in bearing:
                row = cells[arm][uid][program]
                if row["auprc"] is None or row["auprc_delta"] is None:
                    unreadable.append(uid)
                    continue
                auprcs.append(float(row["auprc"]))
                deltas.append(float(row["auprc_delta"]))
                f1s.append(float(row["f1"]))
                f1_deltas.append(float(row["f1_delta"]))
                if float(row["auprc_delta"]) <= -MATERIAL:
                    harmed.append(uid)
                elif float(row["auprc_delta"]) >= MATERIAL:
                    improved.append(uid)
            programs[program] = {
                "macro_auprc": _macro(auprcs),
                "macro_auprc_delta": _macro(deltas),
                "macro_f1": _macro(f1s),
                "macro_f1_delta": _macro(f1_deltas),
                "n_scored": len(auprcs),
                "n_event_bearing": len(bearing),
                "unreadable_series": unreadable,
                "harmed": len(harmed),
                "harmed_series": harmed,
                "improved": len(improved),
                "improved_series": improved,
                "worst": min(deltas) if deltas else None,
                "best": max(deltas) if deltas else None,
            }
        out[arm] = {
            "event_bearing_series": bearing,
            "zero_event_series": zero,
            "programs": programs,
        }
    return out


def _program_label(row: dict[str, Any]) -> str:
    delta = row["macro_auprc_delta"]
    if delta is None:
        return "UNREADABLE"
    if abs(float(delta)) <= MATERIAL:
        return "ARTIFACT_BAND"
    if float(delta) <= -MATERIAL and int(row["harmed"]) > HARMED_MAX_FOR_ARTIFACT:
        return "HARM"
    if float(delta) <= -MATERIAL:
        return "WEAK_NEGATIVE"
    return "POSITIVE_OR_OUTSIDE"


def judge_arm(block: dict[str, Any]) -> dict[str, Any]:
    programs = {p: dict(block["programs"][p]) for p in CLEANING}
    bearing = list(block["event_bearing_series"])
    n_bearing = len(bearing)
    unread_share = {}
    for program in CLEANING:
        unread = len(programs[program]["unreadable_series"])
        unread_share[program] = (
            unread / n_bearing if n_bearing else 1.0)
    majority_unread = bool(
        n_bearing == 0
        or sum(1 for p in CLEANING
               if unread_share[p] > MAJORITY_UNREADABLE) >= 2)
    labels = {p: _program_label(programs[p]) for p in CLEANING}
    artefact = [p for p, lab in labels.items() if lab == "ARTIFACT_BAND"]
    harm = [p for p, lab in labels.items() if lab == "HARM"]
    if majority_unread:
        verdict = "INSTRUMENT_UNREADABLE"
        reason = (
            "AUPRC uncomputable on a majority of event-bearing series "
            "for at least two cleaning programs"
        )
    elif all(lab == "ARTIFACT_BAND" for lab in labels.values()):
        verdict = "THRESHOLD_ARTIFACT_CONFIRMED"
        reason = (
            "all four cleaning programs' AUPRC macro Δ fall in ±%.3f"
            % MATERIAL
        )
    elif harm and artefact:
        verdict = "MIXED"
        reason = (
            "some programs sit in the artefact band, some meet the "
            "DATA_HARM bar; labelled per program"
        )
    elif harm:
        verdict = "DATA_HARM_CONFIRMED"
        reason = (
            "at least one program has AUPRC macro Δ ≤ −%.3f and "
            "harmed series > %d/24"
            % (MATERIAL, HARMED_MAX_FOR_ARTIFACT)
        )
    else:
        verdict = "MIXED"
        reason = (
            "not all programs fall in ±%.3f and no program meets the "
            "DATA_HARM bar (harmed > %d/24); leftover, labelled per program"
            % (MATERIAL, HARMED_MAX_FOR_ARTIFACT)
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "per_program_label": labels,
        "artefact_programs": artefact,
        "harm_programs": harm,
        "majority_unreadable": majority_unread,
        "unread_share": unread_share,
        "gate": {
            "artefact_band": MATERIAL,
            "harm_macro_at_most": -MATERIAL,
            "harmed_gt": HARMED_MAX_FOR_ARTIFACT,
            "harmed_denominator": 24,
            "zero_event_excluded_from_macro": True,
        },
    }


def sign_agreement(cells: dict[str, Any], order: list[str]) -> dict[str, Any]:
    """Descriptive: AUPRC Δ vs F1 Δ sign, event-bearing cells only."""
    bearing = _event_bearing(cells, order)
    out: dict[str, Any] = {}
    for arm in CONSUMERS:
        total = 0
        agree = 0
        both_zero = 0
        disagree = 0
        skipped = 0
        cells_out: list[dict[str, Any]] = []
        for uid in bearing:
            for program in CLEANING:
                row = cells[arm][uid][program]
                a_delta = row["auprc_delta"]
                f_delta = row["f1_delta"]
                if a_delta is None:
                    skipped += 1
                    continue
                total += 1
                sa, sf = _sign(float(a_delta)), _sign(float(f_delta))
                same = sa == sf
                if same:
                    agree += 1
                    if sa == 0:
                        both_zero += 1
                else:
                    disagree += 1
                cells_out.append({
                    "uid": uid, "program": program,
                    "auprc_delta": float(a_delta),
                    "f1_delta": float(f_delta),
                    "same_sign": same,
                })
        out[arm] = {
            "status": "POST_HOC_DESCRIPTIVE",
            "not_a_judgment": (
                "sign agreement is reported because the book asks for it; "
                "it does not open or close a verdict"
            ),
            "event_bearing_cells": total,
            "skipped_unreadable": skipped,
            "agree": agree,
            "disagree": disagree,
            "both_zero": both_zero,
            "rate": (agree / total) if total else None,
            "cells": cells_out,
        }
    return out


def reproduce_m0c_f1(cells: dict[str, Any], order: list[str]
                     ) -> dict[str, Any]:
    """Companion F1 must match the landed M0-C matrix bitwise.

    Same fits, same score_series, same flags.  Confirms this book is a
    reread of that exam and not a new Consumer.
    """
    if not M0C.exists():
        return {"status": "M0C_ARTIFACT_MISSING"}
    landed = json.loads(M0C.read_text(encoding="utf-8"))
    pairs = 0
    exact = 0
    worst = 0.0
    mismatches: list[dict[str, Any]] = []
    for uid in order:
        for arm in CONSUMERS:
            for program in PROGRAMS:
                want = landed["per_series"][uid][arm][program]
                got = cells[arm][uid][program]
                for key in ("f1", "eval_delta"):
                    source = ("f1_delta" if key == "eval_delta" else "f1")
                    reference = float(want[key])
                    value = float(got[source])
                    gap = abs(value - reference)
                    pairs += 1
                    if value == reference:
                        exact += 1
                    worst = max(worst, gap)
                    if gap > ANCHOR_TOLERANCE:
                        mismatches.append({
                            "uid": uid, "arm": arm, "program": program,
                            "field": key, "landed": reference,
                            "reproduced": value,
                        })
    return {
        "status": "REPRODUCED" if not mismatches else "ANCHOR_MISMATCH",
        "artifact": M0C.name,
        "pairs_compared": pairs,
        "pairs_bitwise_equal": exact,
        "max_abs_gap": worst,
        "tolerance": ANCHOR_TOLERANCE,
        "mismatches": mismatches[:20],
        "note": (
            "event-F1 companion, not a judgment input; confirms the "
            "same fits and the same score_series path as M0-C"
        ),
    }


def new_auprc_anchors(agg: dict[str, Any]) -> dict[str, Any]:
    """First identity AUPRC reading on this roster.  New-anchor identity."""
    out: dict[str, Any] = {"identity": "NEW_ANCHOR"}
    for arm in CONSUMERS:
        row = agg[arm]["programs"]["identity"]
        out[arm] = {
            "identity": "NEW_ANCHOR",
            "why": (
                "no prior AUPRC identity reading exists for this roster / "
                "split / Consumer; this number is the first anchor, not a "
                "reproduction of a landed artefact"
            ),
            "macro_auprc": row["macro_auprc"],
            "n_scored": row["n_scored"],
            "n_event_bearing": row["n_event_bearing"],
            "zero_event_excluded": list(agg[arm]["zero_event_series"]),
        }
    return out


def numeric_fingerprint(cells: dict[str, Any], order: list[str]) -> str:
    compact: dict[str, Any] = {}
    for arm in CONSUMERS:
        compact[arm] = {}
        for uid in order:
            compact[arm][uid] = {}
            for program in PROGRAMS:
                row = cells[arm][uid][program]
                compact[arm][uid][program] = {
                    "auprc": row["auprc"],
                    "auprc_delta": row["auprc_delta"],
                    "f1": row["f1"],
                    "f1_delta": row["f1_delta"],
                    "truth_events": row["truth_events"],
                    "predicted_events": row["predicted_events"],
                    "flagged_points": row["flagged_points"],
                }
    return hashlib.sha256(
        json.dumps(compact, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# =========================================================================== #
# Part B / C -- run, judge, write
# =========================================================================== #
def run() -> int:
    substrate = load_substrate()
    order = substrate["order"]
    series = substrate["series"]
    work_hashes_before = {uid: _sha(series[uid]["work_path"]) for uid in order}

    official = FitBudget(FIT_CAP)
    cells = exam(substrate, official)
    _attach_deltas(cells, order)

    verify = FitBudget(FIT_CAP)
    again = exam(substrate, verify)
    _attach_deltas(again, order)
    fp1 = numeric_fingerprint(cells, order)
    fp2 = numeric_fingerprint(again, order)
    two_run = {
        "status": "BITWISE_IDENTICAL" if fp1 == fp2 else "DRIFT",
        "official_fingerprint": fp1,
        "verification_fingerprint": fp2,
        "official_fits": official.used,
        "verification_fits": verify.used,
        "note": (
            "two complete 24 x 5 x 2 exams in one process; the official "
            "cap (280) is charged only to the first exam; the second is "
            "a verification replay and is reported separately"
        ),
    }
    if fp1 != fp2:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "two-run numeric fingerprint drifted")

    work_hashes_after = {uid: _sha(series[uid]["work_path"]) for uid in order}
    originals_untouched = work_hashes_before == work_hashes_after
    if not originals_untouched:
        raise Stop("PROTOCOL_BREACH", "a Yahoo work CSV changed during the exam")

    agg = aggregate(cells, order)
    judgments = {arm: judge_arm(agg[arm]) for arm in CONSUMERS}
    signs = sign_agreement(cells, order)
    f1_anchor = reproduce_m0c_f1(cells, order)
    auprc_anchor = new_auprc_anchors(agg)

    if f1_anchor.get("status") == "ANCHOR_MISMATCH":
        for arm in judgments:
            judgments[arm]["verdict"] = "INSTRUMENT_UNREADABLE"
            judgments[arm]["reason"] = (
                "companion event-F1 did not reproduce the landed M0-C "
                "matrix; the reread is not of the same exam"
            )

    zero = _zero_event(cells, order)
    zero_detail = {
        uid: {
            "truth_events": int(
                cells["c_a_iforest"][uid]["identity"]["truth_events"]),
            "excluded_from_auprc_macro": True,
            "identity_auprc": {
                arm: cells[arm][uid]["identity"]["auprc"] for arm in CONSUMERS},
            "identity_f1": {
                arm: float(cells[arm][uid]["identity"]["f1"])
                for arm in CONSUMERS},
        }
        for uid in zero
    }

    iforest_spec = iforest.spec()
    pca_spec = pca.spec()
    if float(iforest.FOREST_KWARGS["contamination"]) != 0.1:
        raise Stop("PROTOCOL_BREACH",
                   "in-service IForest contamination was not 0.1")
    if pca.THRESHOLD_QUANTILE != 0.90 or pca.RANK != 3:
        raise Stop("PROTOCOL_BREACH",
                   "in-service PCA rank/threshold were not 3 / 0.90")

    outside_findings = [
        {
            "id": "score_samples_vs_decision_function",
            "reading": (
                "IForest AUPRC uses the existing -decision_function ranking.  "
                "That ranking is identical to -score_samples because "
                "decision_function = score_samples - offset_ is a per-model "
                "constant shift.  No new threshold was introduced to break "
                "the tie."
            ),
            "action": "reported, not repaired",
        },
        {
            "id": "pca_point_mapping_is_window_end",
            "reading": (
                "PCA maps the window residual to the window-ending point "
                "via the existing score_region alignment; this book does "
                "not invent a second mapping."
            ),
            "action": "reported, not repaired",
        },
        {
            "id": "companion_f1_still_thresholded",
            "reading": (
                "event-F1 still uses each Consumer's in-service threshold "
                "(IForest decision < 0; PCA residual > training 0.90 "
                "quantile).  That is the old-calibre companion, taken from "
                "the same fit, and is not an input to the verdict."
            ),
            "action": "reported, not repaired",
        },
        {
            "id": "contamination_left_untouched",
            "reading": (
                "aegists_iforest_v1.FOREST_KWARGS['contamination'] remains "
                "0.1.  This audit does not retune a Consumer in service."
            ),
            "action": "reported, not repaired",
        },
    ]

    obligations = {
        "llm_calls": 0,
        "supervised_v3_rerun": False,
        "fit_budget_used_official": official.used,
        "fit_budget_cap": FIT_CAP,
        "fit_budget_respected": bool(official.used <= FIT_CAP),
        "fits_by_arm_official": dict(official.by_arm),
        "verification_fits": verify.used,
        "two_run_bitwise": two_run["status"],
        "yahoo_sealed_41_reads": 0,
        "yahoo_exposed_24_reads": len(order),
        "freeze_roster_n": substrate["freeze_roster_n"],
        "noaa_nab_smd_beyond_17520_reads": 0,
        "eval_region_bytes_processed": 0,
        "new_threshold_introduced": False,
        "contamination_parameter_edited": False,
        "work_originals_untouched": bool(originals_untouched),
        "m0c_f1_reproduction": f1_anchor.get("status"),
        "identity_auprc_anchor_identity": "NEW_ANCHOR",
        "zero_event_excluded_from_macro": True,
        "zero_event_series": zero,
        "methods_package_touched": False,
        "gates_rewritten_after_seeing_numbers": False,
    }

    per_series = {
        uid: {
            arm: {
                program: {
                    "auprc": cells[arm][uid][program]["auprc"],
                    "auprc_delta": cells[arm][uid][program]["auprc_delta"],
                    "f1": cells[arm][uid][program]["f1"],
                    "f1_delta": cells[arm][uid][program]["f1_delta"],
                    "truth_events": cells[arm][uid][program]["truth_events"],
                    "predicted_events": cells[arm][uid][program][
                        "predicted_events"],
                    "matched_events": cells[arm][uid][program][
                        "matched_events"],
                    "flagged_points": cells[arm][uid][program][
                        "flagged_points"],
                }
                for program in PROGRAMS
            }
            for arm in CONSUMERS
        }
        for uid in order
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "entry": "--run",
        "book": "#44-audit threshold-free reread",
        "evidence_class": "INSTRUMENT / EVIDENCE_INTEGRITY",
        "development_only": True,
        "claim_cap": (
            "a development evidence-integrity audit of the M0-C IForest "
            "and PCA event-F1 negatives.  It decides whether those "
            "negatives are a threshold artefact or data harm.  It is not "
            "a headroom probe, does not retune contamination, and never "
            "enters a Yahoo capability claim"
        ),
        "held_fixed": {
            "roster": order,
            "split": "held-in [0, int(0.7n)) / eval [int(0.7n), n)",
            "menu": list(PROGRAMS),
            "acting_bytes": "the training substrate only; Query bytes raw",
            "primary_metric": (
                "pointwise AUPRC on development_exposed_eval, existing "
                "aegists_iforest_v1.auprc, no new threshold"
            ),
            "companion_metric": (
                "event F1 from the same score_series call; not a "
                "judgment input"
            ),
        },
        "consumers": {
            "c_a_iforest": {
                "module": "consumers/aegists_iforest_v1.py",
                "score": (
                    "-decision_function; ranking-identical to "
                    "score_samples; no threshold in AUPRC"
                ),
                "spec": iforest_spec,
            },
            "c_c_pca": {
                "module": "consumers/pca_reconstruction_v1.py",
                "score": (
                    "window RMS residual mapped to the window-ending "
                    "point by the existing score_region alignment"
                ),
                "spec": pca_spec,
            },
            "c_b_supervised": {
                "rerun": False,
                "reason": (
                    "mechanism is positive-row erosion, independently "
                    "measured and already threshold-free"
                ),
            },
        },
        "eval_zone": "development_exposed_eval",
        "true_held_out": "remaining 41 sealed series; unread this book",
        "new_auprc_anchors": auprc_anchor,
        "m0c_f1_reproduction": f1_anchor,
        "zero_event_series": zero_detail,
        "per_series": per_series,
        "aggregate": agg,
        "judgment": judgments,
        "sign_agreement_auprc_vs_f1": signs,
        "determinism": {
            "two_run": two_run,
            "work_originals_sha_unchanged": bool(originals_untouched),
        },
        "outside_findings": outside_findings,
        "cost": {
            "llm": 0,
            "ad_fits_official": official.used,
            "ad_fit_cap": FIT_CAP,
            "ad_fits_by_arm_official": dict(official.by_arm),
            "ad_fits_verification": verify.used,
            "forecast_retrains": 0,
        },
        "obligations": obligations,
    }
    OUT_JSON.write_text(_json_text(payload), encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "judgment": {arm: row["verdict"] for arm, row in judgments.items()},
        "new_auprc_anchors": {
            arm: auprc_anchor[arm]["macro_auprc"] for arm in CONSUMERS},
        "m0c_f1": f1_anchor.get("status"),
        "two_run": two_run["status"],
        "zero_event": zero,
        "fits_official": official.used,
        "fits_verification": verify.used,
        "originals_untouched": originals_untouched,
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)
    return 0


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return ("%+." + str(digits) + "f") % float(value)


def _markdown(payload: dict[str, Any]) -> str:
    order = list(payload["held_fixed"]["roster"])
    lines = [
        "# #44-audit -- threshold-free reread of M0-C IForest and PCA",
        "",
        "evidence class: %s (development).  %s" % (
            payload["evidence_class"], payload["claim_cap"]),
        "",
        "## Verdict",
        "",
    ]
    for arm in CONSUMERS:
        row = payload["judgment"][arm]
        lines.append("- **%s**: **%s**" % (arm, row["verdict"]))
        lines.append("  - %s" % row["reason"])
        lines.append("  - per-program: %s" % row["per_program_label"])
    lines.extend([
        "",
        "## Instrument gates",
        "",
        "- companion event-F1 vs landed M0-C: **%s**, %s/%s pairs bitwise "
        "equal, max gap %s" % (
            payload["m0c_f1_reproduction"].get("status"),
            payload["m0c_f1_reproduction"].get("pairs_bitwise_equal"),
            payload["m0c_f1_reproduction"].get("pairs_compared"),
            payload["m0c_f1_reproduction"].get("max_abs_gap")),
        "- two-run numeric fingerprint: **%s**" % (
            payload["determinism"]["two_run"]["status"]),
        "- work originals SHA unchanged: **%s**" % (
            payload["determinism"]["work_originals_sha_unchanged"]),
        "- identity AUPRC identity: **NEW_ANCHOR** (no prior AUPRC "
        "anchor on this roster)",
        "",
        "## New identity AUPRC anchors",
        "",
        "| Consumer | identity | macro AUPRC | n scored / event-bearing | "
        "zero-event excluded |",
        "|---|---|---|---|---|",
    ])
    for arm in CONSUMERS:
        row = payload["new_auprc_anchors"][arm]
        lines.append("| %s | NEW_ANCHOR | %s | %d / %d | %s |" % (
            arm, _fmt(row["macro_auprc"]).lstrip("+"),
            row["n_scored"], row["n_event_bearing"],
            ", ".join(row["zero_event_excluded"]) or "none"))
    lines.extend([
        "",
        "## AUPRC macro (event-bearing series only; zero-event excluded)",
        "",
    ])
    for arm in CONSUMERS:
        lines.extend([
            "### %s" % arm,
            "",
            "| program | macro AUPRC | macro Δ | harmed /24 | improved | "
            "worst | label |",
            "|---|---|---|---|---|---|---|",
        ])
        labels = payload["judgment"][arm]["per_program_label"]
        for program in PROGRAMS:
            row = payload["aggregate"][arm]["programs"][program]
            label = "identity" if program == "identity" else labels[program]
            lines.append(
                "| %s | %s | %s | %d | %d | %s | %s |" % (
                    program,
                    _fmt(row["macro_auprc"]).lstrip("+"),
                    _fmt(row["macro_auprc_delta"]),
                    row["harmed"], row["improved"],
                    _fmt(row["worst"], 4), label))
        lines.append("")
        lines.append("harmed series by program:")
        for program in CLEANING:
            row = payload["aggregate"][arm]["programs"][program]
            names = ", ".join(row["harmed_series"]) or "none"
            lines.append("- %s (%s): %s" % (
                program, payload["judgment"][arm]["per_program_label"][program],
                names))
        lines.append("")
    lines.extend([
        "## Companion event-F1 (same fits; not a judgment input)",
        "",
    ])
    for arm in CONSUMERS:
        lines.extend([
            "### %s" % arm,
            "",
            "| program | macro F1 | macro Δ |",
            "|---|---|---|",
        ])
        for program in PROGRAMS:
            row = payload["aggregate"][arm]["programs"][program]
            lines.append("| %s | %s | %s |" % (
                program, _fmt(row["macro_f1"]).lstrip("+"),
                _fmt(row["macro_f1_delta"])))
        lines.append("")
    lines.extend([
        "## AUPRC Δ vs F1 Δ sign agreement (descriptive)",
        "",
        "| Consumer | agree | disagree | both zero | rate | skipped |",
        "|---|---|---|---|---|---|",
    ])
    for arm in CONSUMERS:
        row = payload["sign_agreement_auprc_vs_f1"][arm]
        lines.append("| %s | %d | %d | %d | %s | %d |" % (
            arm, row["agree"], row["disagree"], row["both_zero"],
            "n/a" if row["rate"] is None else "%.3f" % row["rate"],
            row["skipped_unreadable"]))
    lines.extend([
        "",
        payload["sign_agreement_auprc_vs_f1"]["c_a_iforest"]["not_a_judgment"],
        "",
        "## Per-series AUPRC",
        "",
    ])
    for arm in CONSUMERS:
        lines.extend([
            "### %s" % arm,
            "",
            "| series | identity | iqr Δ | mad Δ | hampel Δ | winsorize Δ | "
            "events |",
            "|---|---|---|---|---|---|---|",
        ])
        for uid in order:
            cell = payload["per_series"][uid][arm]
            events = int(cell["identity"]["truth_events"])
            ident = cell["identity"]["auprc"]
            if events == 0:
                lines.append(
                    "| %s | excluded (zero-event) | n/a | n/a | n/a | n/a | "
                    "0 |" % uid)
                continue
            lines.append("| %s | %s | %s | %s | %s | %s | %d |" % (
                uid,
                _fmt(ident).lstrip("+") if ident is not None else "n/a",
                _fmt(cell["outlier_iqr"]["auprc_delta"], 4),
                _fmt(cell["outlier_mad"]["auprc_delta"], 4),
                _fmt(cell["hampel_filter"]["auprc_delta"], 4),
                _fmt(cell["winsorize"]["auprc_delta"], 4),
                events))
        lines.append("")
    cost = payload["cost"]
    lines.extend([
        "## Zero-event eval series (listed; not in any macro)",
        "",
    ])
    for uid, row in payload["zero_event_series"].items():
        lines.append("- %s: truth_events=%d; identity AUPRC %s; identity F1 %s"
                     % (uid, row["truth_events"], row["identity_auprc"],
                        row["identity_f1"]))
    if not payload["zero_event_series"]:
        lines.append("- none")
    lines.extend([
        "",
        "## Budget",
        "",
        "- LLM: %d; official AD fits: %d / %d; verification fits: %d" % (
            cost["llm"], cost["ad_fits_official"], cost["ad_fit_cap"],
            cost["ad_fits_verification"]),
        "- official by arm: %s" % ", ".join(
            "%s=%d" % (k, v) for k, v in cost["ad_fits_by_arm_official"].items()),
        "",
        "## Obligation self-report",
        "",
    ])
    for key in sorted(payload["obligations"]):
        lines.append("- %s: %s" % (key, payload["obligations"][key]))
    lines.extend(["", "## Outside findings (reported, not repaired)", ""])
    for row in payload["outside_findings"]:
        lines.append("- **%s**: %s" % (row["id"], row["reading"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    argv = sys.argv[1:]
    if "--run" in argv:
        return run()
    print("usage: --run")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stop as exc:  # noqa: BLE001
        print(json.dumps({"verdict": exc.code, "detail": exc.detail},
                         ensure_ascii=False, indent=1))
        raise SystemExit(3) from exc
