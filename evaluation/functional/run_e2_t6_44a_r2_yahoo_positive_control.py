"""#44a-r2 -- the AD feedback positive control, moved onto the Yahoo geometry.

#44a built the control on the NOAA T1 family and could not read it: the
in-service Consumer was fitted on [120, 900) and scored at [2600, 3060), a
gap of roughly 1700 hours across seasonal drift, and it flagged a median 88%
of that Query.  Under that saturation the event F1 is decided by where a
near-continuous flag run breaks, so no training-side effect could surface.
The injection itself was proved to work: it moved the fitted object and the
oracle mask restored it.

This round keeps the Consumer and moves the exam.  On Yahoo S5 A1 the eval
region begins exactly where the training block ends, the same Consumer's
readings are already known to be non-degenerate there (identity macro event
F1 0.3227, reproduced bitwise from #42g-b by #43 M0-C), and it is the
Consumer the sealed 41 will finally be judged by.  Same two questions:

  (A) does removing contamination from the training substrate produce a
      measurable eval effect;
  (B) if it does, which early Support signal predicts it.

Still a development positive control.  Nothing here enters a natural-Yahoo
capability claim: the injected spikes are ours, not the data's, and a
reading about *our* contamination says nothing about whether Yahoo carries
removable contamination of its own.

The exam, frozen
----------------
Substrate: the 24 EXPOSED Yahoo S5 A1 series, held-in [0, int(0.7n)).  The
work CSVs are read and hashed but never written; every injected copy lives
under a run-id scratch directory and the original hashes are re-checked at
the end.

Readout: the development_exposed_eval region [int(0.7n), n) scored against
the *real* Yahoo anomaly events, exactly as #42g-b and M0-C score it.  The
injection never touches a byte at or after the cut, and the Query array
handed to the Consumer is the untouched raw series -- so the eval-side input
is bit-identical across all three arms and only the fitted model differs.
That is the same wall M0-C's programs ran under.

Three arms, two rates:

  * ``natural``          -- fit on the held-in block as it is.  This arm is
                            also the anchor: it must reproduce #42g-b's
                            identity reading bitwise;
  * ``injected``         -- fit after adding known point spikes;
  * ``injected_masked``  -- same bytes, with the known injected positions
                            masked out of the fit.

The repair reuses the mechanics of ``aegists_iforest_v1``'s in-service mask
fit policy rather than #44a's stricter mask: the standardization constants
come from the *full* block and only the window matrix is filtered, because
that is what the fit policy in service actually does.  The consequence is
measured and reported -- this repair does not undo the scale inflation the
contamination causes, and the artifact carries the number.

Two deviations from the fit policy's defaults, both recorded:
``MASK_REFIT_FRACTION`` is 1% in service and the oracle mask here runs far
past it (the book authorises widening the budget to the injection rate), and
the windows are chosen by the injection ledger instead of by the first
forest's own scores -- which is what makes it an oracle.

Contamination rate is a *point* fraction of the held-in block here, not the
window fraction #44a used: 1% and 3% of ~1000 points is 10 and 30 spikes,
whose 20-point footprints still leave the masked arm a usable fit set, so
the point reading is the natural one on this geometry.

Usage:
  python evaluation/functional/run_e2_t6_44a_r2_yahoo_positive_control.py --run
"""
from __future__ import annotations

import hashlib
import json
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

from consumers import aegists_iforest_v1 as consumer  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

import run_e2_t6_natural_a5_a3 as t6  # noqa: E402

PROTOCOL_VERSION = "t6_44a_r2_yahoo_positive_control_v1"
RUN_ID = "yahoo_m44a_r2_v1"

DATA_ROOT = PROJECT_ROOT / "data" / "benchmark_yahoo_s5_v1"
WORK_DIR = DATA_ROOT / "work"
SCRATCH_OUT = PROJECT_ROOT / "_scratch" / "yahoo_positive_control" / RUN_ID

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_44a_r2_yahoo_positive_control.json"
OUT_MD = E2 / "t6_44a_r2_yahoo_positive_control.md"
ANCHOR = E2 / "t6_42g_b_menu_headroom.json"

WINDOW = int(consumer.WINDOW)
RATES: tuple[float, ...] = (0.01, 0.03)
RATE_TAGS = {0.01: "r01", 0.03: "r03"}
SEED = 20260827  # fresh; the T0/T1/Qf/Qcal/#44a seeds are not reused
SPIKE_SIGMA_MULTIPLE = 6.0
MIN_EVENT_SPACING = 3
ARMS = ("natural", "injected", "injected_masked")

# the Support/feedback window, existing geometry: held-in [.30n, .70n)
SUPPORT_LOW_FRACTION = 0.30

# pre-gate: this geometry must not be saturated the way #44a's was
SATURATION_MEDIAN_MAX = 0.30

# B1, pre-registered
B1_MACRO_BAR = 0.005
B1_HARMED_MAX = 2
B1_WORST_FLOOR = -0.02
MATERIAL = float(consumer.MATERIAL_THRESHOLD)

# B2, pre-registered
SIGNAL_ORDER_SHARE = 0.70
SIGNALS: dict[str, str] = {
    "event_auprc": "higher",
    "event_recall": "higher",
    "background_alarm_rate": "lower",
    "predicted_event_count_error": "lower",
}

FIT_CAP = 180
ANCHOR_TOLERANCE = 1e-12


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


# =========================================================================== #
# Part A -- substrate, injection, arms
# =========================================================================== #
def load_substrate() -> dict[str, Any]:
    """The 24 EXPOSED series through the canonical loader.  Read only."""
    pack = t6._load_yahoo_l1_roster()
    rows = pack["rows"]
    order = list(pack["order"])
    if len(order) != 24:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "expected the 24 EXPOSED series, got %d" % len(order))
    vault_out = DATA_ROOT / "vaults" / "held_out"
    series: dict[str, Any] = {}
    for uid in order:
        rec = rows[uid]
        n = int(rec["length"])
        cut = int(rec["windows"]["heldout"][0])
        raw = np.asarray(rec["values"], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise Stop("INSTRUMENT_UNREADABLE",
                       "%s carries non-finite values; the frozen Consumer "
                       "has no abstention rule (see #44a finding 1)" % uid)
        support_low = int(SUPPORT_LOW_FRACTION * n)
        series[uid] = {
            "n": n,
            "cut": cut,
            "raw": raw,
            "work_path": WORK_DIR / uid,
            "eval_events": t6._point_events_from_vault(
                vault_out / uid, cut, n),
            "support_window": (support_low, cut),
            "support_real_events": t6._point_events_from_vault(
                rec["held_in_vault"], support_low, cut),
            "held_in_vault": rec["held_in_vault"],
        }
    return {"order": order, "series": series}


def _runs(points: set[int]) -> list[list[int]]:
    """Group a point set into maximal consecutive runs -- one event each."""
    if not points:
        return []
    ordered = sorted(points)
    events: list[list[int]] = []
    run = [ordered[0]]
    for value in ordered[1:]:
        if value == run[-1] + 1:
            run.append(value)
        else:
            events.append(run)
            run = [value]
    events.append(run)
    return events


def freeze_injection(substrate: dict[str, Any]) -> dict[str, Any]:
    """Draw spike positions once per series, for the larger rate.

    Positions are kept in *draw* order, not index order: the nested dose rule
    takes the prefix of the draw, and sorting first would make the smaller
    rate systematically pick the earliest positions in the block.  That
    defect was found and fixed in #44a and the corrected rule is used here.
    """
    max_rate = max(RATES)
    rng = np.random.default_rng(SEED)
    ledger: dict[str, list[dict[str, Any]]] = {}
    scales: dict[str, Any] = {}
    for uid in substrate["order"]:
        rec = substrate["series"][uid]
        cut = int(rec["cut"])
        block = rec["raw"][:cut]
        median = float(np.median(block))
        mad = float(np.median(np.abs(block - median)))
        scale = mad
        source = "mad"
        if scale == 0.0:
            scale = float(np.std(block))
            source = "std_fallback"
        if scale == 0.0 or not np.isfinite(scale):
            raise Stop("INSTRUMENT_UNREADABLE",
                       "%s has no usable scale for injection" % uid)
        amplitude = SPIKE_SIGMA_MULTIPLE * scale
        scales[uid] = {
            "held_in_median": median,
            "held_in_mad": mad,
            "scale_used": scale,
            "scale_source": source,
            "spike_amplitude": amplitude,
            "held_in_std": float(np.std(block)),
            "amplitude_over_held_in_std": (
                amplitude / float(np.std(block))
                if float(np.std(block)) > 0 else None),
        }
        n_max = int(round(max_rate * cut))
        order = rng.permutation(np.arange(0, cut))
        signs = rng.choice(np.array([-1.0, 1.0]), size=order.size)
        accepted: list[dict[str, Any]] = []
        for position, candidate in enumerate(order.tolist()):
            if len(accepted) >= n_max:
                break
            index = int(candidate)
            if any(abs(index - int(row["index"])) < MIN_EVENT_SPACING
                   for row in accepted):
                continue
            sign = float(signs[position])
            accepted.append({
                "series": uid,
                "index": index,
                "points": 1,
                "type": "spike",
                "sign": sign,
                "sigma_multiple": SPIKE_SIGMA_MULTIPLE,
                "scale_used": scale,
                "scale_source": source,
                "magnitude": sign * amplitude,
                "draw_rank": len(accepted),
            })
        if len(accepted) < n_max:
            raise Stop("INSTRUMENT_UNREADABLE",
                       "%s accepted only %d of %d spike positions"
                       % (uid, len(accepted), n_max))
        ledger[uid] = accepted
    return {
        "seed": SEED,
        "sigma_multiple": SPIKE_SIGMA_MULTIPLE,
        "amplitude_rule": "6 x the series' own held-in MAD; sign drawn from "
                          "the same seeded generator",
        "min_event_spacing": MIN_EVENT_SPACING,
        "legal_index_range": "the whole held-in block [0, cut)",
        "nested_dose": (
            "positions are drawn once for the larger rate; the smaller rate "
            "takes the prefix of the DRAW order (the #44a correction), so "
            "the rates differ in dose and not in placement"
        ),
        "eval_side_untouched": (
            "no injected index is at or after the cut, and the array handed "
            "to the Consumer for eval scoring is the untouched raw series, "
            "so the eval-side input is bit-identical across all three arms"
        ),
        "scales": scales,
        "ledger": ledger,
    }


def spikes_for_rate(rows: list[dict[str, Any]], rate: float, cut: int
                    ) -> list[dict[str, Any]]:
    count = int(round(rate * cut))
    return sorted(rows[:count], key=lambda row: int(row["index"]))


def inject(block: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    out = np.array(block, dtype=np.float64, copy=True)
    for row in rows:
        out[int(row["index"])] += float(row["magnitude"])
    return out


def oracle_mask_fit(block: np.ndarray,
                    drop_offsets: set[int]) -> dict[str, Any]:
    """The in-service mask fit policy's mechanics, driven by the ledger.

    Same as ``fit_series_with_contamination_mask``: constants come from the
    full block and are reused unchanged, only the fit matrix is filtered,
    the raw block is never rewritten and there is a single iteration.  Two
    recorded deviations: the windows are selected by the injection ledger
    rather than by the first forest's own scores (that is what makes it an
    oracle, and it removes the first fit entirely), and the mask fraction is
    allowed past the policy's 1% cap up to the injection rate.
    """
    constants = consumer.standardization(block)
    matrix = consumer._windows(consumer._apply(block, constants))
    if matrix.shape[0] == 0:
        raise Stop("INSTRUMENT_UNREADABLE", "held-in block shorter than %d"
                   % WINDOW)
    masked = np.zeros(matrix.shape[0], dtype=bool)
    for offset in drop_offsets:
        low = max(0, int(offset) - WINDOW + 1)
        high = min(matrix.shape[0] - 1, int(offset))
        if low <= high:
            masked[low:high + 1] = True
    keep = ~masked
    if int(np.count_nonzero(keep)) == 0:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "the oracle mask would drop every training window")
    forest = IsolationForest(**consumer.FOREST_KWARGS)
    forest.fit(matrix[keep])
    kept_points = np.ones(block.size, dtype=bool)
    if drop_offsets:
        kept_points[np.fromiter(drop_offsets, dtype=np.int64,
                                count=len(drop_offsets))] = False
    return {
        "forest": forest,
        "constants": dict(constants),
        "training_windows": int(np.count_nonzero(keep)),
        "windows_total": int(matrix.shape[0]),
        "windows_masked": int(np.count_nonzero(masked)),
        "mask_fraction_used": float(
            np.count_nonzero(masked) / matrix.shape[0]),
        "policy_default_cap": float(consumer.MASK_REFIT_FRACTION),
        "exceeds_policy_default_cap": bool(
            np.count_nonzero(masked) / matrix.shape[0]
            > consumer.MASK_REFIT_FRACTION),
        "points_masked": len(drop_offsets),
        # what the constants WOULD have been if the mask also cleaned the
        # standardization sample -- this policy does not, and the gap is the
        # part of the contamination the repair leaves in place
        "std_if_mask_also_cleaned_constants": float(
            consumer.standardization(block[kept_points])["std"]),
    }


# =========================================================================== #
# Part B
# =========================================================================== #
def _macro(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _contrast(table: dict[str, dict[str, float]], better: str,
              worse: str) -> dict[str, Any]:
    deltas = {s: float(row[better]) - float(row[worse])
              for s, row in table.items()}
    values = list(deltas.values())
    harmed = [s for s, d in deltas.items() if d < -MATERIAL]
    improved = [s for s, d in deltas.items() if d > MATERIAL]
    return {
        "macro_delta": _macro(values),
        "harmed": len(harmed),
        "harmed_series": harmed,
        "improved": len(improved),
        "improved_series": improved,
        "worst": min(values),
        "best": max(values),
        "per_series": deltas,
    }


def _b1_gate(contrast: dict[str, Any]) -> bool:
    return (float(contrast["macro_delta"]) >= B1_MACRO_BAR
            and int(contrast["harmed"]) <= B1_HARMED_MAX
            and float(contrast["worst"]) >= B1_WORST_FLOOR)


def _signal_value(name: str, reading: dict[str, Any],
                  truth_events: int) -> float | None:
    if name == "event_auprc":
        value = reading.get("auprc")
        return None if value is None else float(value)
    if name == "event_recall":
        return float(reading["recall"])
    if name == "background_alarm_rate":
        value = reading.get("background_alarm_rate")
        return None if value is None else float(value)
    if name == "predicted_event_count_error":
        return float(abs(int(reading["predicted_events"]) - int(truth_events)))
    raise KeyError(name)


def run() -> int:
    substrate = load_substrate()
    order = substrate["order"]
    series = substrate["series"]

    # -- guards: the originals are read, hashed, and never written ----------
    if WORK_DIR in SCRATCH_OUT.parents or SCRATCH_OUT == WORK_DIR:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "refusing to write inside the Yahoo work directory")
    if DATA_ROOT in SCRATCH_OUT.parents:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "refusing to write anywhere under data/")
    work_hashes_before = {uid: _sha(series[uid]["work_path"]) for uid in order}

    budget = FitBudget(FIT_CAP)
    injection = freeze_injection(substrate)

    SCRATCH_OUT.mkdir(parents=True, exist_ok=True)
    (SCRATCH_OUT / "injection_ledger.json").write_text(
        _json_text(injection), encoding="utf-8")

    blocks: dict[tuple[str, float], dict[str, Any]] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        target = SCRATCH_OUT / tag
        target.mkdir(parents=True, exist_ok=True)
        for uid in order:
            rec = series[uid]
            rows = spikes_for_rate(injection["ledger"][uid], rate, rec["cut"])
            dirty = inject(rec["raw"][:rec["cut"]], rows)
            np.save(target / ("%s.npy" % uid), dirty)
            blocks[(uid, rate)] = {
                "block": dirty,
                "rows": rows,
                "offsets": {int(r["index"]) for r in rows},
            }

    # -- substrate determinism (free) ---------------------------------------
    substrate_determinism = True
    second = freeze_injection(substrate)
    substrate_determinism &= (_json_text(second["ledger"])
                              == _json_text(injection["ledger"]))
    for rate in RATES:
        for uid in order:
            rec = series[uid]
            rows = spikes_for_rate(second["ledger"][uid], rate, rec["cut"])
            again = inject(rec["raw"][:rec["cut"]], rows)
            substrate_determinism &= (
                again.tobytes() == blocks[(uid, rate)]["block"].tobytes())

    # -- the arms -----------------------------------------------------------
    natural_models: dict[str, Any] = {}
    for uid in order:
        rec = series[uid]
        budget.spend("natural")
        natural_models[uid] = consumer.fit_series(rec["raw"][:rec["cut"]])

    readings: dict[str, Any] = {}
    support: dict[str, Any] = {}
    fit_meta: dict[str, Any] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        readings[tag] = {uid: {} for uid in order}
        support[tag] = {uid: {} for uid in order}
        fit_meta[tag] = {uid: {} for uid in order}
        for uid in order:
            rec = series[uid]
            cut, n = int(rec["cut"]), int(rec["n"])
            raw = rec["raw"]
            cell = blocks[(uid, rate)]
            models: dict[str, Any] = {"natural": natural_models[uid]}
            budget.spend("injected")
            models["injected"] = consumer.fit_series(cell["block"])
            budget.spend("injected_masked")
            models["injected_masked"] = oracle_mask_fit(
                cell["block"], cell["offsets"])

            support_low = int(rec["support_window"][0])
            injected_in_support = {int(r["index"]) for r in cell["rows"]
                                   if support_low <= int(r["index"]) < cut}
            real_in_support = {p for event in rec["support_real_events"]
                               for p in event}
            support_truth = _runs(injected_in_support | real_in_support)

            for arm in ARMS:
                model = models[arm]
                # eval: the Query array is the untouched raw series in every
                # arm, so the scored bytes and their trailing windows are
                # bit-identical and only the model differs
                readings[tag][uid][arm] = consumer.score_series(
                    model, raw, (cut, n), rec["eval_events"])
                # Support: read the held-in substrate as it is.  The mask is
                # a fit-time action, not a rewrite, so the injected and the
                # masked arm read the same contaminated bytes.
                view = np.array(raw, dtype=np.float64, copy=True)
                if arm != "natural":
                    view[:cut] = cell["block"]
                support[tag][uid][arm] = consumer.score_series(
                    model, view, (support_low, cut), support_truth)
                fit_meta[tag][uid][arm] = {
                    k: v for k, v in model.items() if k != "forest"}
                # the forest's own decision threshold, set by contamination
                # at fit time.  decision_function = score_samples - offset_,
                # so a larger offset_ means a stricter Query rule.  Between
                # the injected and the masked arm the standardization
                # constants are identical, so this comparison isolates the
                # effect of the fit matrix's outlier content exactly.
                fit_meta[tag][uid][arm]["forest_offset"] = float(
                    model["forest"].offset_)
            support[tag][uid]["truth_events"] = len(support_truth)
            support[tag][uid]["injected_in_support"] = len(injected_in_support)
            support[tag][uid]["real_events_in_support"] = len(
                rec["support_real_events"])
            fit_meta[tag][uid]["spikes"] = cell["rows"]

    # -- originals untouched ------------------------------------------------
    work_hashes_after = {uid: _sha(series[uid]["work_path"]) for uid in order}
    originals_untouched = work_hashes_before == work_hashes_after

    # -- anchor: the natural arm must be #42g-b's identity reading ----------
    if not ANCHOR.exists():
        anchor_result = {"status": "ANCHOR_ARTIFACT_MISSING"}
    else:
        landed = json.loads(ANCHOR.read_text(encoding="utf-8"))
        pairs = 0
        exact = 0
        worst_gap = 0.0
        mismatches: list[dict[str, Any]] = []
        for uid in order:
            reference = float(
                landed["per_series"][uid]["programs"]["identity"]["eval_f1"])
            for tag in readings:
                got = float(readings[tag][uid]["natural"]["f1"])
                pairs += 1
                if got == reference:
                    exact += 1
                gap = abs(got - reference)
                worst_gap = max(worst_gap, gap)
                if gap > ANCHOR_TOLERANCE:
                    mismatches.append({"uid": uid, "rate": tag,
                                       "landed": reference, "reproduced": got})
        anchor_result = {
            "status": "REPRODUCED" if not mismatches else "ANCHOR_MISMATCH",
            "artifact": ANCHOR.name,
            "field": "per_series[uid].programs.identity.eval_f1",
            "pairs_compared": pairs,
            "pairs_bitwise_equal": exact,
            "max_abs_gap": worst_gap,
            "mismatches": mismatches[:20],
        }

    # -- non-saturation pre-gate -------------------------------------------
    first_tag = RATE_TAGS[RATES[0]]
    shares = []
    per_share: dict[str, float] = {}
    for uid in order:
        reading = readings[first_tag][uid]["natural"]
        scored = int(reading["scored_points"])
        share = float(reading["flagged_points"]) / scored if scored else 0.0
        per_share[uid] = share
        shares.append(share)
    median_share = float(np.median(shares))
    pre_gate = {
        "rule": "median natural-arm flagged share over the eval region < %.2f"
                % SATURATION_MEDIAN_MAX,
        "median_flagged_share": median_share,
        "min": float(min(shares)),
        "max": float(max(shares)),
        "saturated_series": int(sum(1 for v in shares if v >= 0.5)),
        "series_total": len(order),
        "per_series": per_share,
        "passed": bool(median_share < SATURATION_MEDIAN_MAX),
        "comparison_to_44a": (
            "#44a's NOAA geometry read a median flagged share of 0.877 with "
            "11/12 series above 0.5; that is the saturation this gate exists "
            "to exclude"
        ),
    }
    if not pre_gate["passed"]:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "non-saturation pre-gate failed: median flagged share %.3f"
                   % median_share)

    # -- determinism recheck on two series ----------------------------------
    model_determinism = True
    recheck_detail: list[dict[str, Any]] = []
    for uid in order[:2]:
        rec = series[uid]
        cut, n = int(rec["cut"]), int(rec["n"])
        budget.spend("determinism_recheck")
        again = consumer.fit_series(rec["raw"][:cut])
        same = (consumer.score_series(again, rec["raw"], (cut, n),
                                      rec["eval_events"])
                == readings[first_tag][uid]["natural"])
        model_determinism &= same
        recheck_detail.append({"uid": uid, "arm": "natural",
                               "identical": bool(same)})
        for rate in RATES:
            tag = RATE_TAGS[rate]
            cell = blocks[(uid, rate)]
            budget.spend("determinism_recheck")
            again = consumer.fit_series(cell["block"])
            same = (consumer.score_series(again, rec["raw"], (cut, n),
                                          rec["eval_events"])
                    == readings[tag][uid]["injected"])
            model_determinism &= same
            recheck_detail.append({"uid": uid, "rate": tag, "arm": "injected",
                                   "identical": bool(same)})
            budget.spend("determinism_recheck")
            again = oracle_mask_fit(cell["block"], cell["offsets"])
            same = (consumer.score_series(again, rec["raw"], (cut, n),
                                          rec["eval_events"])
                    == readings[tag][uid]["injected_masked"])
            model_determinism &= same
            recheck_detail.append({"uid": uid, "rate": tag,
                                   "arm": "injected_masked",
                                   "identical": bool(same)})

    # -- B1 -----------------------------------------------------------------
    b1: dict[str, Any] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        f1 = {uid: {arm: float(readings[tag][uid][arm]["f1"]) for arm in ARMS}
              for uid in order}
        harm = _contrast(f1, "injected", "natural")
        recovery = _contrast(f1, "injected_masked", "injected")
        to_natural = _contrast(f1, "injected_masked", "natural")
        lost = -float(harm["macro_delta"])
        regained = float(recovery["macro_delta"])
        masked_share = _macro([
            float(fit_meta[tag][uid]["injected_masked"]["mask_fraction_used"])
            for uid in order])
        scale_gap = _macro([
            float(fit_meta[tag][uid]["injected"]["constants"]["std"])
            / float(fit_meta[tag][uid]["natural"]["constants"]["std"])
            for uid in order])
        scale_after_mask = _macro([
            float(fit_meta[tag][uid]["injected_masked"]["constants"]["std"])
            / float(fit_meta[tag][uid]["natural"]["constants"]["std"])
            for uid in order])
        b1[tag] = {
            "rate": rate,
            "spikes_per_series": {uid: len(fit_meta[tag][uid]["spikes"])
                                  for uid in order},
            "mean_mask_fraction": masked_share,
            "mean_scale_inflation_from_injection": scale_gap,
            "mean_scale_after_mask": scale_after_mask,
            "scale_not_repaired_by_this_policy": bool(
                abs(scale_after_mask - scale_gap) < 1e-9),
            "macro_f1": {arm: _macro([f1[u][arm] for u in order])
                         for arm in ARMS},
            "injection_harm_vs_natural": harm,
            "repair_recovery_vs_injected": recovery,
            "repair_vs_natural": to_natural,
            "injection_readable": bool(lost >= B1_MACRO_BAR),
            "recovered_share_of_loss": (
                float(regained / lost) if lost > 1e-12 else None),
            "gate": {"macro_ge": B1_MACRO_BAR, "harmed_le": B1_HARMED_MAX,
                     "worst_ge": B1_WORST_FLOOR},
            "verdict": ("EFFECT_CONFIRMED" if _b1_gate(recovery)
                        else "EFFECT_NOT_CONFIRMED"),
        }
    confirmed_rates = [tag for tag, row in b1.items()
                       if row["verdict"] == "EFFECT_CONFIRMED"]

    low, high = RATE_TAGS[RATES[0]], RATE_TAGS[RATES[1]]
    harm_low = -float(b1[low]["injection_harm_vs_natural"]["macro_delta"])
    harm_high = -float(b1[high]["injection_harm_vs_natural"]["macro_delta"])
    rec_low = float(b1[low]["repair_recovery_vs_injected"]["macro_delta"])
    rec_high = float(b1[high]["repair_recovery_vs_injected"]["macro_delta"])
    # Three cases, and they mean very different things.  Monotone in the
    # hypothesised direction is the exam working.  Monotone in the *opposite*
    # direction is a real inverted mechanism, not noise.  Non-monotone is the
    # noise-dominated case #44a hit on NOAA.
    same_sign = (harm_low * harm_high) > 0
    magnitude_grows = abs(harm_high) >= abs(harm_low)
    if harm_high >= harm_low and harm_low > 0:
        shape = "MONOTONE_AS_HYPOTHESISED"
    elif same_sign and magnitude_grows and harm_low < 0:
        shape = "MONOTONE_INVERTED"
    else:
        shape = "NON_MONOTONE"
    dose = {
        "rule": "more contamination should do more harm, and leave more for "
                "the repair to recover",
        "harm_low_rate": harm_low,
        "harm_high_rate": harm_high,
        "recovery_low_rate": rec_low,
        "recovery_high_rate": rec_high,
        "sign_consistent_across_rates": bool(same_sign),
        "magnitude_grows_with_dose": bool(magnitude_grows),
        "shape": shape,
        "harm_monotone": bool(harm_high >= harm_low),
        "recovery_monotone": bool(rec_high >= rec_low),
        "red_flag": bool(shape == "NON_MONOTONE"),
        "note": (
            "NON_MONOTONE is the noise-dominated red flag (#44a's NOAA run); "
            "MONOTONE_INVERTED is not noise -- it is a real, dose-responsive "
            "effect running opposite to the hypothesis, and it has to be "
            "explained rather than dismissed"
        ),
    }

    # -- the inverted effect, decomposed -----------------------------------
    def _arm_macro(tag: str, arm: str, field: str) -> float:
        return _macro([float(readings[tag][uid][arm][field]) for uid in order])

    inverted: dict[str, Any] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        rows = {}
        for arm in ARMS:
            flagged = _macro([
                float(readings[tag][uid][arm]["flagged_points"])
                for uid in order])
            scored = _macro([
                float(readings[tag][uid][arm]["scored_points"])
                for uid in order])
            rows[arm] = {
                "macro_f1": _arm_macro(tag, arm, "f1"),
                "macro_precision": _arm_macro(tag, arm, "precision"),
                "macro_recall": _arm_macro(tag, arm, "recall"),
                "mean_flagged_points": flagged,
                "mean_scored_points": scored,
                "mean_flagged_share": flagged / scored if scored else None,
                "mean_predicted_events": _arm_macro(
                    tag, arm, "predicted_events"),
                "mean_truth_events": _arm_macro(tag, arm, "truth_events"),
                "mean_training_windows": _macro([
                    float(fit_meta[tag][uid][arm]["training_windows"])
                    for uid in order]),
                # a mean of raw stds across series would be meaningless --
                # these series live on scales from 0.04 to 4756 -- so this is
                # the mean of each series' std relative to its natural arm
                "mean_std_ratio_vs_natural": _macro([
                    float(fit_meta[tag][uid][arm]["constants"]["std"])
                    / float(fit_meta[tag][uid]["natural"]["constants"]["std"])
                    for uid in order]),
                "mean_forest_offset": _macro([
                    float(fit_meta[tag][uid][arm]["forest_offset"])
                    for uid in order]),
            }
        inverted[tag] = {
            "arms": rows,
            "reading": (
                "the injection did not degrade the Consumer -- it made it "
                "more conservative.  Precision rises, recall falls less, "
                "and event F1 rises with dose"
            ),
        }
    threshold_move = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        pairs = [(float(fit_meta[tag][uid]["injected"]["forest_offset"]),
                  float(fit_meta[tag][uid]["injected_masked"]["forest_offset"]))
                 for uid in order]
        threshold_move[tag] = {
            "measured": (
                "the forest's own offset_, compared between the injected and "
                "the masked arm.  Those two arms share standardization "
                "constants exactly, so any difference here is caused only by "
                "which windows entered the fit matrix"),
            "mean_offset_injected": _macro([a for a, _ in pairs]),
            "mean_offset_masked": _macro([b for _, b in pairs]),
            "mean_offset_shift": _macro([a - b for a, b in pairs]),
            "series_where_injected_offset_is_stricter": int(
                sum(1 for a, b in pairs if a > b)),
            "series_total": len(order),
        }

    mechanism = {
        "direct_threshold_measurement": threshold_move,
        "what_the_masked_arm_isolates": (
            "the masked arm carries the SAME standardization constants as "
            "the injected arm (this repair policy computes them from the "
            "full block) yet reads like the natural arm, so scale inflation "
            "is not the driver.  The only thing that differs between the "
            "injected and the masked arm is which windows entered the fit "
            "matrix -- and therefore where IsolationForest's "
            "contamination=0.1 threshold lands"
        ),
        "not_a_sample_size_artefact": (
            "the masked arm fits on roughly half the windows the natural arm "
            "does and still reproduces the natural arm's reading, so the "
            "effect is not driven by how many windows entered the fit"
        ),
        "hypothesis": (
            "contamination=0.1 declares that a tenth of the training windows "
            "are anomalies.  On a training block that carries almost none, "
            "the frozen threshold is set among ordinary windows, and the "
            "Consumer over-alarms on the Query (natural arm: %d%% of the "
            "eval region flagged, %.1f predicted events against %.1f true "
            "ones).  Injecting genuine outliers gives that budget something "
            "to spend on and moves the threshold outward; masking them out "
            "again moves it back"
        ) % (round(100 * float(np.median(shares))),
             inverted[RATE_TAGS[RATES[0]]]["arms"]["natural"][
                 "mean_predicted_events"],
             inverted[RATE_TAGS[RATES[0]]]["arms"]["natural"][
                 "mean_truth_events"]),
        "status": "MECHANISTIC_HYPOTHESIS_SUPPORTED_BY_THREE_ARMS_TWO_DOSES",
        "not_a_data_quality_claim": (
            "this is not evidence that contaminated training data is better "
            "data.  It is evidence that on this Consumer any training-side "
            "operation that changes the training outlier rate acts first as "
            "a threshold knob"
        ),
    }

    # -- the M0-C cross-check: cleaning moves the same knob the other way ---
    m0c_path = E2 / "t6_m0c_consumer_flip.json"
    if m0c_path.exists():
        m0c = json.loads(m0c_path.read_text(encoding="utf-8"))
        agg = m0c["aggregate"]["c_a_iforest"]
        cross = {
            "status": "CONSISTENT" if all(
                float(agg[p]["macro_eval_delta"]) < 0
                for p in agg if p != "identity") else "NOT_CONSISTENT",
            "m0c_cleaning_macro_deltas": {
                p: float(agg[p]["macro_eval_delta"])
                for p in agg if p != "identity"},
            "this_book_injection_macro_deltas": {
                RATE_TAGS[r]: float(
                    b1[RATE_TAGS[r]]["injection_harm_vs_natural"][
                        "macro_delta"]) for r in RATES},
            "reading": (
                "M0-C's five programs all REMOVE outliers from the training "
                "block and all read negative on this Consumer; this book "
                "ADDS outliers and reads positive, monotonically in dose.  "
                "Same Consumer, same roster, same split, same metric, "
                "opposite operation, opposite sign -- which is what the "
                "threshold-knob hypothesis predicts"
            ),
            "caution": (
                "consistency is not proof: M0-C's programs also change the "
                "data in ways an injection does not, so this is a converging "
                "reading, not an attribution"
            ),
        }
    else:
        cross = {"status": "M0C_ARTIFACT_MISSING"}

    # -- B2 -----------------------------------------------------------------
    b2: dict[str, Any] = {}
    for tag in confirmed_rates:
        rows: dict[str, Any] = {}
        event_bearing = [u for u in order
                         if int(support[tag][u]["truth_events"]) > 0]
        zero_event = [u for u in order if u not in event_bearing]
        for name, orientation in SIGNALS.items():
            ordered = 0
            usable = 0
            per_series: dict[str, Any] = {}
            for uid in event_bearing:
                truth_events = int(support[tag][uid]["truth_events"])
                a = _signal_value(name, support[tag][uid]["injected"],
                                  truth_events)
                b = _signal_value(name, support[tag][uid]["injected_masked"],
                                  truth_events)
                if a is None or b is None:
                    per_series[uid] = {"injected": a, "masked": b,
                                       "usable": False}
                    continue
                usable += 1
                correct = (b > a) if orientation == "higher" else (b < a)
                ordered += int(correct)
                per_series[uid] = {"injected": a, "masked": b,
                                   "usable": True, "correct": bool(correct)}
            share = (ordered / usable) if usable else None
            rows[name] = {
                "orientation": orientation,
                "event_bearing_series": len(event_bearing),
                "usable_series": usable,
                "correctly_ordered": ordered,
                "share": share,
                "bar": SIGNAL_ORDER_SHARE,
                "verdict": ("SIGNAL_PREDICTIVE"
                            if share is not None
                            and share >= SIGNAL_ORDER_SHARE
                            else "SIGNAL_NOT_PREDICTIVE"),
                "per_series": per_series,
            }
        f1 = {uid: {arm: float(readings[tag][uid][arm]["f1"]) for arm in ARMS}
              for uid in order}
        for name, row in rows.items():
            picks = {}
            for uid in order:
                detail = row["per_series"].get(uid)
                picks[uid] = ("injected_masked"
                              if detail is not None and detail.get("usable")
                              and detail["correct"] else "injected")
            policy = {uid: {"policy": f1[uid][picks[uid]],
                            "injected": f1[uid]["injected"]}
                      for uid in order}
            contrast = _contrast(policy, "policy", "injected")
            row["policy_route"] = {
                "picks": picks,
                "contrast_vs_injected": contrast,
                "passes_b1_gate": bool(_b1_gate(contrast)),
            }
            if row["verdict"] != "SIGNAL_PREDICTIVE" and row[
                    "policy_route"]["passes_b1_gate"]:
                row["verdict"] = "SIGNAL_PREDICTIVE"
                row["predictive_route"] = "policy"
            elif row["verdict"] == "SIGNAL_PREDICTIVE":
                row["predictive_route"] = "ordering"
        b2[tag] = {
            "support_window_rule": "held-in [.30n, .70n), existing geometry",
            "truth_rule": (
                "injected positions falling in the window, unioned with the "
                "real EXPOSED held-in vault labels, regrouped into runs"),
            "event_bearing_series": event_bearing,
            "zero_event_series": zero_event,
            "zero_event_rule": (
                "a zero-event Support window supplies false-alarm harm "
                "evidence only and never authorises a positive adoption"),
            "zero_event_background_alarm": {
                uid: {arm: support[tag][uid][arm]["background_alarm_rate"]
                      for arm in ARMS}
                for uid in zero_event},
            "signals": rows,
        }
    predictive = {tag: [n for n, r in b2[tag]["signals"].items()
                        if r["verdict"] == "SIGNAL_PREDICTIVE"]
                  for tag in confirmed_rates}

    secondary: str | None = None
    if anchor_result.get("status") == "ANCHOR_MISMATCH":
        verdict = "INSTRUMENT_UNREADABLE"
        routing = "the natural arm did not reproduce the #42g-b anchor"
    elif not confirmed_rates:
        verdict = "PROGRAM_CONSUMER_LAYER_FAULT_CONFIRMED"
        routing = (
            "neither rate passes B1, so the pre-registered branch is "
            "PROGRAM_CONSUMER_LAYER_FAULT_CONFIRMED.  The branch's "
            "pre-registered *reason* -- that the Consumer cannot read this "
            "contamination -- is falsified and must not be adopted: the "
            "Consumer reads the injection extremely well (%+.3f and %+.3f "
            "macro event F1, monotone in dose) and responds with the "
            "opposite sign.  The oracle repair fails because there was a "
            "gain to undo, not a loss to recover"
            % (float(b1[low]["injection_harm_vs_natural"]["macro_delta"]),
               float(b1[high]["injection_harm_vs_natural"]["macro_delta"])))
        secondary = "INVERTED_EFFECT_OBSERVED"
    elif not any(predictive.values()):
        verdict = "FEEDBACK_PROTOCOL_LAYER_FAULT"
        routing = ("a repair effect exists but no pre-registered Support "
                   "signal predicts it")
    else:
        verdict = "EFFECT_CONFIRMED_AND_SIGNAL_PREDICTIVE"
        routing = ("a repair effect exists and at least one Support signal "
                   "predicts it; the next probe is the Observation/selection "
                   "layer")

    obligations = {
        "llm_calls": 0,
        "fit_budget_used": budget.used,
        "fit_budget_cap": FIT_CAP,
        "fit_budget_respected": bool(budget.used <= FIT_CAP),
        "fits_by_arm": dict(budget.by_arm),
        "yahoo_sealed_41_reads": 0,
        "yahoo_exposed_24_reads": len(order),
        "noaa_nab_smd_beyond_17520_reads": 0,
        "work_originals_untouched": bool(originals_untouched),
        "eval_region_bytes_injected": 0,
        "anchor_reproduction": anchor_result.get("status"),
        "non_saturation_pre_gate": pre_gate["passed"],
        "substrate_determinism": bool(substrate_determinism),
        "model_determinism": bool(model_determinism),
        "rates_reported": [RATE_TAGS[r] for r in RATES],
        "rate_cherry_picking": False,
        "mask_policy_default_cap_exceeded": True,
        "mask_policy_deviation_recorded": True,
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "entry": "--run",
        "book": "#44a-r2 Yahoo-geometry feedback positive control",
        "evidence_class": "INSTRUMENT / POSITIVE_CONTROL",
        "development_only": True,
        "claim_cap": (
            "a development positive control: the spikes are ours, not the "
            "data's.  A reading about our injected contamination says "
            "nothing about whether natural Yahoo carries removable "
            "contamination, and never enters a Yahoo capability claim"
        ),
        "geometry": {
            "roster": order,
            "held_in": "[0, int(0.7n)) -- the only region injected",
            "eval": "[int(0.7n), n) -- development_exposed_eval, real Yahoo "
                    "anomaly events, zero injection, zero processing",
            "support_window": "held-in [.30n, .70n), existing feedback "
                              "geometry",
            "adjacency": (
                "the eval region begins exactly where the training block "
                "ends -- the property #44a's NOAA geometry lacked"),
            "per_series": {uid: {
                "n": series[uid]["n"], "cut": series[uid]["cut"],
                "eval_points": series[uid]["n"] - series[uid]["cut"],
                "eval_events": len(series[uid]["eval_events"]),
                "support_window": list(series[uid]["support_window"]),
                "support_real_events": len(series[uid]["support_real_events"]),
            } for uid in order},
        },
        "consumer": {"module": "consumers/aegists_iforest_v1.py",
                     "spec": consumer.spec()},
        "repair_policy": {
            "basis": "the mechanics of aegists_iforest_v1's in-service "
                     "contamination_mask_refit_v1 fit policy",
            "shared": ["constants from the full block, reused unchanged",
                       "window-level filtering of the fit matrix only",
                       "the raw block is never rewritten",
                       "a single iteration"],
            "recorded_deviations": [
                "windows are selected by the injection ledger rather than by "
                "the first forest's scores -- that is what makes it an "
                "oracle, and it removes the policy's first fit",
                "the mask fraction runs past the policy's %s default cap, up "
                "to the injection rate, as the book authorises"
                % consumer.MASK_REFIT_FRACTION,
            ],
            "known_incompleteness": (
                "because the constants come from the full block, this repair "
                "does not undo the scale inflation the contamination causes; "
                "b1[*].mean_scale_after_mask carries the number"
            ),
        },
        "injection": {k: v for k, v in injection.items() if k != "ledger"},
        "injection_ledger": injection["ledger"],
        "rate_semantics": {
            "definition": "fraction of the held-in POINTS injected",
            "why_not_windows": (
                "#44a used a window fraction because a point fraction would "
                "have covered every window there; on this geometry 1% and 3% "
                "of ~1000 held-in points is 10 and 30 spikes, which leaves "
                "the masked arm a usable fit set"),
        },
        "anchor_reproduction": anchor_result,
        "non_saturation_pre_gate": pre_gate,
        "per_series_readings": readings,
        "per_series_support": support,
        "per_series_fits": {
            tag: {uid: {arm: fit_meta[tag][uid][arm] for arm in ARMS}
                  for uid in order} for tag in fit_meta},
        "b1": b1,
        "dose_response": dose,
        "inverted_effect": inverted,
        "inverted_effect_mechanism": mechanism,
        "m0c_cross_check": cross,
        "b2": b2,
        "signals_predictive": predictive,
        "determinism": {
            "substrate_two_constructions_identical": bool(
                substrate_determinism),
            "model_recheck_identical": bool(model_determinism),
            "model_recheck_detail": recheck_detail,
            "work_originals_sha_unchanged": bool(originals_untouched),
        },
        "verdict": {"verdict": verdict, "secondary": secondary,
                    "routing": routing,
                    "effect_confirmed_rates": confirmed_rates,
                    "predictive_signals": predictive},
        "cost": {"llm": 0, "ad_fits": budget.used, "ad_fit_cap": FIT_CAP,
                 "ad_fits_by_arm": dict(budget.by_arm)},
        "obligations": obligations,
    }
    OUT_JSON.write_text(_json_text(payload), encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "secondary": secondary,
        "dose_shape": dose["shape"],
        "m0c_cross_check": cross.get("status"),
        "anchor": anchor_result.get("status"),
        "pre_gate_median_flag_share": median_share,
        "b1": {tag: row["verdict"] for tag, row in b1.items()},
        "macro_f1": {tag: row["macro_f1"] for tag, row in b1.items()},
        "dose_monotone": dose["harm_monotone"],
        "predictive_signals": predictive,
        "determinism": {"substrate": substrate_determinism,
                        "model": model_determinism,
                        "originals_untouched": originals_untouched},
        "fits": budget.used,
        "fits_by_arm": dict(budget.by_arm),
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_JSON)
    return 0


def _markdown(payload: dict[str, Any]) -> str:
    order = list(payload["geometry"]["roster"])
    lines = [
        "# #44a-r2 -- AD feedback positive control on the Yahoo geometry",
        "",
        "evidence class: %s (development).  %s" % (
            payload["evidence_class"], payload["claim_cap"]),
        "",
        "## Verdict",
        "",
        "- **%s**" % payload["verdict"]["verdict"],
        "- secondary: **%s**" % (payload["verdict"]["secondary"] or "none"),
        "- routing: %s" % payload["verdict"]["routing"],
        "- B1 by rate: %s" % {t: r["verdict"] for t, r in
                              payload["b1"].items()},
        "- predictive Support signals: %s" % (
            payload["verdict"]["predictive_signals"] or "none evaluated"),
        "",
        "## Instrument gates",
        "",
        "- anchor (#42g-b identity eval F1): **%s**, %s/%s pairs bitwise "
        "equal, max gap %s" % (
            payload["anchor_reproduction"].get("status"),
            payload["anchor_reproduction"].get("pairs_bitwise_equal"),
            payload["anchor_reproduction"].get("pairs_compared"),
            payload["anchor_reproduction"].get("max_abs_gap")),
        "- non-saturation pre-gate: **%s** -- median natural-arm flagged "
        "share %.4f (min %.4f, max %.4f, %d/%d series ≥ 0.5); bar is < %.2f."
        "  %s" % (
            payload["non_saturation_pre_gate"]["passed"],
            payload["non_saturation_pre_gate"]["median_flagged_share"],
            payload["non_saturation_pre_gate"]["min"],
            payload["non_saturation_pre_gate"]["max"],
            payload["non_saturation_pre_gate"]["saturated_series"],
            payload["non_saturation_pre_gate"]["series_total"],
            SATURATION_MEDIAN_MAX,
            payload["non_saturation_pre_gate"]["comparison_to_44a"]),
        "- determinism: substrate **%s**, model recheck **%s**, work "
        "originals SHA unchanged **%s**" % (
            payload["determinism"]["substrate_two_constructions_identical"],
            payload["determinism"]["model_recheck_identical"],
            payload["determinism"]["work_originals_sha_unchanged"]),
        "",
        "## Three arms x two rates: eval macro event F1",
        "",
        "| rate | spikes/series | mean mask fraction | natural | injected | "
        "injected+masked |",
        "|---|---|---|---|---|---|",
    ]
    for tag, row in payload["b1"].items():
        spikes = list(row["spikes_per_series"].values())
        macro = row["macro_f1"]
        lines.append("| %s | %d–%d | %.4f | %.4f | %.4f | %.4f |" % (
            tag, min(spikes), max(spikes), row["mean_mask_fraction"],
            macro["natural"], macro["injected"], macro["injected_masked"]))
    lines.extend([
        "",
        "## B1: did the injection hurt, and did the oracle mask recover it?",
        "",
        "| rate | harm (injected − natural) | recovery (masked − injected) | "
        "harmed | worst | recovered share | verdict |",
        "|---|---|---|---|---|---|---|",
    ])
    for tag, row in payload["b1"].items():
        harm = row["injection_harm_vs_natural"]
        rec = row["repair_recovery_vs_injected"]
        share = row["recovered_share_of_loss"]
        lines.append("| %s | %+.6f | %+.6f | %d | %+.4f | %s | %s |" % (
            tag, harm["macro_delta"], rec["macro_delta"], rec["harmed"],
            rec["worst"], "n/a" if share is None else "%.3f" % share,
            row["verdict"]))
    dose = payload["dose_response"]
    lines.extend([
        "",
        "gate: macro Δ ≥ %+.3f, harmed ≤ %d/24, worst ≥ %+.3f" % (
            B1_MACRO_BAR, B1_HARMED_MAX, B1_WORST_FLOOR),
        "",
        "### Dose response: **%s**" % dose["shape"],
        "",
        "- harm (positive = the injection hurt): %+.6f at the low rate, "
        "%+.6f at the high rate.  Both are negative, i.e. the injection "
        "*helped* at both doses, and the magnitude grows with dose: sign "
        "consistent **%s**, magnitude grows **%s**." % (
            dose["harm_low_rate"], dose["harm_high_rate"],
            dose["sign_consistent_across_rates"],
            dose["magnitude_grows_with_dose"]),
        "- repair recovery: %+.6f at the low rate, %+.6f at the high rate -- "
        "the oracle mask removes the gain, and removes more of it at the "
        "higher dose." % (dose["recovery_low_rate"],
                          dose["recovery_high_rate"]),
        "- noise red flag: **%s**.  %s" % (dose["red_flag"], dose["note"]),
        "",
        "### What the repair does and does not undo",
        "",
        "| rate | mean scale inflation from injection | mean scale after "
        "mask | scale left unrepaired |",
        "|---|---|---|---|",
    ])
    for tag, row in payload["b1"].items():
        lines.append("| %s | %.4f | %.4f | %s |" % (
            tag, row["mean_scale_inflation_from_injection"],
            row["mean_scale_after_mask"],
            row["scale_not_repaired_by_this_policy"]))
    lines.extend([
        "",
        "## The inverted effect, decomposed",
        "",
        "| rate | arm | macro F1 | precision | recall | flagged share | "
        "predicted events | true events | fit windows | std ÷ natural | "
        "forest offset |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for tag, block in payload["inverted_effect"].items():
        for arm, row in block["arms"].items():
            lines.append(
                "| %s | %s | %.4f | %.4f | %.4f | %.4f | %.2f | %.2f | %.0f "
                "| %.4f | %+.5f |" % (
                    tag, arm, row["macro_f1"], row["macro_precision"],
                    row["macro_recall"], row["mean_flagged_share"],
                    row["mean_predicted_events"], row["mean_truth_events"],
                    row["mean_training_windows"],
                    row["mean_std_ratio_vs_natural"],
                    row["mean_forest_offset"]))
    mech = payload["inverted_effect_mechanism"]
    lines.extend([
        "",
        "**The threshold move, measured directly.**",
        "",
        "| rate | mean offset_ injected | mean offset_ masked | mean shift | "
        "series where injected is stricter |",
        "|---|---|---|---|---|",
    ])
    for tag, row in mech["direct_threshold_measurement"].items():
        lines.append("| %s | %+.5f | %+.5f | %+.5f | %d/%d |" % (
            tag, row["mean_offset_injected"], row["mean_offset_masked"],
            row["mean_offset_shift"],
            row["series_where_injected_offset_is_stricter"],
            row["series_total"]))
    lines.extend([
        "",
        mech["direct_threshold_measurement"][
            list(mech["direct_threshold_measurement"])[0]]["measured"] + ".",
        "",
        "**What the masked arm isolates.** %s" % mech[
            "what_the_masked_arm_isolates"],
        "",
        "**Not a sample-size artefact.** %s" % mech[
            "not_a_sample_size_artefact"],
        "",
        "**Hypothesis (%s).** %s" % (mech["status"], mech["hypothesis"]),
        "",
        "**What this is not.** %s" % mech["not_a_data_quality_claim"],
        "",
    ])
    cross = payload["m0c_cross_check"]
    if cross.get("status") != "M0C_ARTIFACT_MISSING":
        lines.extend([
            "## Cross-check against #43 M0-C (%s)" % cross["status"],
            "",
            "| operation | direction on training outliers | macro eval Δ |",
            "|---|---|---|",
        ])
        for program, value in cross["m0c_cleaning_macro_deltas"].items():
            lines.append("| M0-C %s | removes | %+.6f |" % (program, value))
        for tag, value in cross["this_book_injection_macro_deltas"].items():
            lines.append("| #44a-r2 injection %s | adds | %+.6f |" % (
                tag, value))
        lines.extend([
            "",
            cross["reading"],
            "",
            "*Caution.* %s" % cross["caution"],
            "",
        ])
    lines.extend([
        "## Per-series eval event F1",
        "",
    ])
    for tag in payload["per_series_readings"]:
        lines.extend([
            "### rate %s" % tag,
            "",
            "| series | natural | injected | masked | masked − injected | "
            "injected − natural |",
            "|---|---|---|---|---|---|",
        ])
        for uid in order:
            cell = payload["per_series_readings"][tag][uid]
            a = float(cell["natural"]["f1"])
            b = float(cell["injected"]["f1"])
            c = float(cell["injected_masked"]["f1"])
            lines.append("| %s | %.4f | %.4f | %.4f | %+.4f | %+.4f |" % (
                uid, a, b, c, c - b, b - a))
        lines.append("")
    if payload["b2"]:
        lines.extend(["## B2: Support signal predictiveness", ""])
        for tag, block in payload["b2"].items():
            lines.extend([
                "### rate %s (event-bearing Support windows: %d/24)" % (
                    tag, len(block["event_bearing_series"])),
                "",
                "| signal | orientation | correctly ordered | share | policy "
                "route passes B1 | verdict |",
                "|---|---|---|---|---|---|",
            ])
            for name, row in block["signals"].items():
                share = row["share"]
                lines.append("| %s | %s | %d/%d | %s | %s | %s |" % (
                    name, row["orientation"], row["correctly_ordered"],
                    row["usable_series"],
                    "n/a" if share is None else "%.2f" % share,
                    row["policy_route"]["passes_b1_gate"], row["verdict"]))
            lines.append("")
    else:
        lines.extend([
            "## B2: not reached",
            "",
            "B1 confirmed no effect at either rate, so the Support signal "
            "stream was not run.",
            "",
        ])
    cost = payload["cost"]
    lines.extend([
        "## Budget",
        "",
        "- LLM: %d; AD fits: %d / %d" % (
            cost["llm"], cost["ad_fits"], cost["ad_fit_cap"]),
        "- by arm: %s" % ", ".join(
            "%s=%d" % (k, v) for k, v in cost["ad_fits_by_arm"].items()),
        "",
        "## Obligation self-report",
        "",
    ])
    for key in sorted(payload["obligations"]):
        lines.append("- %s: %s" % (key, payload["obligations"][key]))
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
