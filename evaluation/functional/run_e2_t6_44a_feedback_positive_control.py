"""#44a -- the AD feedback positive control.  0 LLM, fully deterministic.

Why this round exists.  #42h found no safe feedback unit, #42j closed the
mask fit policy, and #43 M0-C read all five menu programs as harmful on all
three AD Consumers.  Every one of those is a reading taken on an exam where
nobody knows whether there was anything to clean.  Before the line can ask
"does the Harness optimise data", it has to answer a prior instrument
question on an exam whose answer is known by construction:

  (A) does removing contamination from the training substrate produce a
      measurable delayed effect at all, under this Consumer and this
      scoring semantics?
  (B) if it does, which early Support signal predicts it?

Nothing here is a capability reading and nothing here may be carried into a
claim about natural Yahoo data.  It is a development positive control whose
only job is to route the first fault to a layer.

The exam, frozen
----------------
Substrate: the 12 NOAA stations of the T1 injection family, read from the
already-open ``_scratch/phase_t/injected`` copies.  No file under ``data/``
is opened by this runner.

Geometry, verified against the three ledgers before any fit (see
``verify_geometry``):

  * series length 8760 (one hourly year), well inside the 17520 wall;
  * training substrate / P's measured action region: [120, 900);
  * Qf, the formal Query: [2100, 2560) -- NOT opened by this book, kept as
    an independent confirmation surface;
  * Qcal, the calibration Query: [2600, 3060), four injected events per
    station, and the only region this book scores;
  * the qcal copy and the qf copy are byte-identical on [120, 900), and the
    t1 copy differs from the qcal copy exactly at the union of the two
    ledgers' points.  That is what licenses using the qcal copy as a
    substrate whose training block is pristine.

Three arms, two contamination rates, one Consumer (aegists_iforest_v1,
window 20, the in-service AD Consumer):

  * ``clean``                  -- fit on the pristine training block.  A
                                  *reference*, not an upper bound: the short
                                  canon forbids calling it a bound before the
                                  reading exists;
  * ``contaminated_identity``  -- fit on the training block after injecting
                                  known point spikes;
  * ``contaminated_repaired``  -- same bytes, but the known injected
                                  positions are masked out of the fit: out of
                                  the standardization sample and out of the
                                  window matrix.  The exact positions come
                                  from the injection record, which is what
                                  makes this an *oracle* repair.

The Query is never processed, never masked and never rewritten; the arms
differ only in what the Consumer was allowed to fit on.

Contamination rate, and why it is a window fraction
---------------------------------------------------
``r`` is the fraction of *training windows* the contamination reaches, not
the fraction of training points.  The Consumer is window-based, the repair
is a window-level mask, and the repo's existing contamination primitive
(``contamination_mask_refit_v1``) is already parameterised by a window
fraction.  Under a point-fraction reading, r = 15% would mark 117 of 780
points, whose 20-point footprints would cover every one of the 761 windows,
so the repaired arm would have nothing left to fit and the exam would test
nothing.  The realized point-level rate is reported alongside, so both
readings are on the record.

The two rates are a nested dose series: positions are drawn once per station
for the larger rate and the smaller rate takes the ordered prefix, so the
rates differ in dose and not in placement luck.

Injection rule, inherited
-------------------------
Type, sign/multiple cycle, spacing, boundary exclusion, sigma rule and the
filter order are the T1 rules, restricted to the point-spike family (the
book fixes one contamination type).  ``sigma_local`` is always computed on
the pristine substrate, so no injected value is ever a scale source -- the
T1 ``delta_scale_dependence`` rule.  The seed is fresh (20260826); the T0,
T1, Qf and Qcal seeds are not reused.

Missing data
------------
The NOAA copies carry NaN.  The frozen Consumer has no abstention rule and
scikit-learn will not fit or score a NaN row, so the adapter here applies
the project's established T0/v3 abstention canon at the adapter layer
rather than editing the frozen Consumer: a window containing a non-finite
point never enters the fit, and on the Query it is forced to not flag and is
excluded from the AUPRC ranking.  Every abstention is counted and reported.
``assert_adapter_matches_consumer`` proves the adapter reduces to
``consumer.fit_series`` bitwise when nothing is masked and nothing is
missing.

Usage:
  python evaluation/functional/run_e2_t6_44a_feedback_positive_control.py --run
"""
from __future__ import annotations

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

# =========================================================================== #
# constants -- frozen by the book
# =========================================================================== #
PROTOCOL_VERSION = "t6_44a_feedback_positive_control_v1"
RUN_ID = "m44a_v1"

INJECTED = PROJECT_ROOT / "_scratch" / "phase_t" / "injected"
T1_DIR = INJECTED / "t1"
QCAL_DIR = INJECTED / "t1b_query" / "qcal"
QF_DIR = INJECTED / "t1b_query" / "qf"
SCRATCH_OUT = PROJECT_ROOT / "_scratch" / "phase_t" / RUN_ID

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_44a_feedback_positive_control.json"
OUT_MD = E2 / "t6_44a_feedback_positive_control.md"

SERIES_LENGTH = 8760
TRAIN_BLOCK = (120, 900)
QCAL_REGION = (2600, 3060)
QF_REGION = (2100, 2560)  # declared, never opened by this book
WINDOW = int(consumer.WINDOW)
TRAIN_WINDOWS = TRAIN_BLOCK[1] - TRAIN_BLOCK[0] - WINDOW + 1  # 761

# Support window: T1's own triple-window geometry is a *forecasting* origin
# geometry sited at [1104, 1392), outside the training block and carrying no
# AD event labels, so it cannot serve as the AD Support window.  The book's
# stated fallback applies: the last 20% of the training region.
SUPPORT_FRACTION = 0.20
SUPPORT_WINDOW = (
    TRAIN_BLOCK[1] - int(SUPPORT_FRACTION * (TRAIN_BLOCK[1] - TRAIN_BLOCK[0])),
    TRAIN_BLOCK[1],
)  # [744, 900)

RATES: tuple[float, ...] = (0.05, 0.15)
RATE_TAGS = {0.05: "r05", 0.15: "r15"}

SEED = 20260826  # fresh; T0/T1/Qf/Qcal used 20260822..20260825
SPIKE_CYCLE: tuple[tuple[float, float], ...] = (
    (1.0, 6.0), (-1.0, 6.0), (1.0, 10.0), (-1.0, 10.0),
)
MIN_EVENT_SPACING = WINDOW      # disjoint window footprints
BOUNDARY_EXCLUSION = 25         # T1 convention
SIGMA_PREFIX = 168              # T1 convention
MAD_SCALE = 1.4826
MIN_SIGMA_PREFIX_FINITE = SIGMA_PREFIX // 2

ARMS = ("clean", "contaminated_identity", "contaminated_repaired")

# B1 gate, pre-registered
B1_MACRO_BAR = 0.005
B1_HARMED_MAX = 1
B1_WORST_FLOOR = -0.02
MATERIAL = float(consumer.MATERIAL_THRESHOLD)  # 0.005

# B2, pre-registered
SIGNAL_ORDER_SHARE = 0.70
SIGNALS: dict[str, str] = {
    # signal -> orientation of "repair should be better than contaminated"
    "event_auprc": "higher",
    "event_recall": "higher",
    "background_alarm_rate": "lower",
    "predicted_event_count_error": "lower",
}

FIT_CAP = 100


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
    return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=False) + "\n"


# =========================================================================== #
# Part A -- geometry verification, then the substrate
# =========================================================================== #
def _ledger_points(ledger: dict[str, Any]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for series, events in ledger.items():
        points: set[int] = set()
        for event in events:
            start = int(event["index"])
            points.update(range(start, start + int(event["points"])))
        out[series] = points
    return out


def _diff_indices(a: np.ndarray, b: np.ndarray) -> set[int]:
    """NaN-aware: two NaNs at the same position are not a difference."""
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return set(np.flatnonzero(~same).tolist())


def verify_geometry() -> dict[str, Any]:
    """Read the three protocols and prove the layout before anything is fit."""
    for directory in (T1_DIR, QCAL_DIR):
        if not (directory / "protocol.json").is_file():
            raise Stop("INSTRUMENT_UNREADABLE",
                       "missing protocol at %s" % directory)
    t1p = json.loads((T1_DIR / "protocol.json").read_text(encoding="utf-8"))
    qcp = json.loads((QCAL_DIR / "protocol.json").read_text(encoding="utf-8"))
    qfp = json.loads((QF_DIR / "protocol.json").read_text(encoding="utf-8"))
    names = sorted(t1p["ledger"])
    if len(names) != 12:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "expected 12 T1 stations, found %d" % len(names))
    t1r = _ledger_points(t1p["ledger"])
    qcr = _ledger_points(qcp["ledger"])
    qfr = _ledger_points(qfp["ledger"])

    per_station: dict[str, Any] = {}
    for series in names:
        t1 = np.load(T1_DIR / ("%s.npy" % series))
        qc = np.load(QCAL_DIR / ("%s.npy" % series))
        qf = np.load(QF_DIR / ("%s.npy" % series))
        if not (t1.size == qc.size == qf.size == SERIES_LENGTH):
            raise Stop("INSTRUMENT_UNREADABLE",
                       "%s is not %d long" % (series, SERIES_LENGTH))
        ledger_match_t1 = _diff_indices(t1, qc) == (t1r[series] | qcr[series])
        ledger_match_qf = _diff_indices(qf, qc) == (qfr[series] | qcr[series])
        train_identical = bool(np.array_equal(
            qc[TRAIN_BLOCK[0]:TRAIN_BLOCK[1]],
            qf[TRAIN_BLOCK[0]:TRAIN_BLOCK[1]], equal_nan=True))
        if not (ledger_match_t1 and ledger_match_qf and train_identical):
            raise Stop(
                "INSTRUMENT_UNREADABLE",
                "%s failed the ledger/geometry check (t1=%s qf=%s train=%s)"
                % (series, ledger_match_t1, ledger_match_qf, train_identical))
        block = qc[TRAIN_BLOCK[0]:TRAIN_BLOCK[1]]
        per_station[series] = {
            "length": int(qc.size),
            "t1_diff_matches_ledgers": ledger_match_t1,
            "qf_diff_matches_ledgers": ledger_match_qf,
            "qcal_qf_training_block_identical": train_identical,
            "non_finite_in_training_block": int(
                np.count_nonzero(~np.isfinite(block))),
            "non_finite_in_qcal_region": int(np.count_nonzero(
                ~np.isfinite(qc[QCAL_REGION[0]:QCAL_REGION[1]]))),
            "qcal_truth_events": len(qcp["ledger"][series]),
            "t1_train_injected_points": len(t1r[series]),
        }
    return {
        "status": "VERIFIED",
        "stations": names,
        "series_length": SERIES_LENGTH,
        "training_block": list(TRAIN_BLOCK),
        "qcal_region": list(QCAL_REGION),
        "qf_region": list(QF_REGION),
        "qf_opened_this_book": False,
        "support_window": list(SUPPORT_WINDOW),
        "support_window_basis": (
            "T1's own triple-window geometry (support origins 1104/1152/1200, "
            "delayed 1248/1296/1344) is a forecasting origin geometry sited "
            "outside the training block and carries no AD event labels, so it "
            "cannot serve as the AD Support window; the book's stated "
            "fallback -- the last 20% of the training region -- is used"
        ),
        "substrate_choice": (
            "the qcal copy is the substrate: its training block is pristine "
            "(byte-identical to the qf copy there, and the t1 copy differs "
            "from it exactly at the union of the t1 and qcal ledger points), "
            "and it already carries the four known Qcal events this book "
            "scores against"
        ),
        "per_station": per_station,
        "qcal_ledger": {s: qcp["ledger"][s] for s in names},
        "read_paths": [str(T1_DIR), str(QCAL_DIR), str(QF_DIR)],
        "data_dir_opened": False,
    }


def _sigma_local(pristine: np.ndarray, index: int) -> dict[str, Any] | None:
    """T1's sigma rule, on the pristine substrate, NaN-aware."""
    prefix = pristine[index - SIGMA_PREFIX:index]
    finite = prefix[np.isfinite(prefix)]
    if finite.size < MIN_SIGMA_PREFIX_FINITE:
        return None
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = MAD_SCALE * mad
    source = "mad"
    if sigma == 0.0:
        sigma = float(np.std(finite))
        source = "std"
    if sigma == 0.0 or not np.isfinite(sigma):
        return None
    return {"sigma_local": sigma, "sigma_source": source}


def freeze_injection(pristine: dict[str, np.ndarray],
                     names: list[str]) -> dict[str, Any]:
    """Draw the spike positions once, for the largest rate.

    T1's filter order is kept exactly: block fit, minimum spacing,
    sigma_local validity, target finiteness.  The cycle counter is global
    across the stations in roster order and advances only on an accepted
    event, as T1 specifies.
    """
    legal_low = max(TRAIN_BLOCK[0] + BOUNDARY_EXCLUSION, SIGMA_PREFIX,
                    TRAIN_BLOCK[0] + WINDOW - 1)
    legal_high = TRAIN_BLOCK[1] - BOUNDARY_EXCLUSION - 1
    if legal_high + WINDOW - 1 >= TRAIN_BLOCK[1] + WINDOW - 1:
        pass  # footprint stays inside the fit region by construction
    max_rate = max(RATES)
    n_max = int(round(max_rate * TRAIN_WINDOWS / WINDOW))
    rng = np.random.default_rng(SEED)
    cycle_counter = 0
    ledger: dict[str, list[dict[str, Any]]] = {}
    skips: dict[str, list[dict[str, Any]]] = {}
    for series in names:
        base = pristine[series]
        order = rng.permutation(np.arange(legal_low, legal_high + 1))
        accepted: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for candidate in order.tolist():
            if len(accepted) >= n_max:
                break
            index = int(candidate)
            if any(abs(index - int(row["index"])) < MIN_EVENT_SPACING
                   for row in accepted):
                continue
            sigma = _sigma_local(base, index)
            if sigma is None:
                skipped.append({"index": index, "why": "sigma_local_invalid"})
                continue
            if not np.isfinite(base[index]):
                skipped.append({"index": index, "why": "target_non_finite"})
                continue
            sign, multiple = SPIKE_CYCLE[cycle_counter % len(SPIKE_CYCLE)]
            cycle_counter += 1
            accepted.append({
                "series": series,
                "index": index,
                "points": 1,
                "type": "spike",
                "sign": float(sign),
                "sigma_multiple": float(multiple),
                "sigma_local": float(sigma["sigma_local"]),
                "sigma_source": sigma["sigma_source"],
                "magnitude": float(sign) * float(multiple)
                * float(sigma["sigma_local"]),
                "cycle_slot": int((cycle_counter - 1) % len(SPIKE_CYCLE)),
                "draw_rank": len(accepted),
            })
        if len(accepted) < n_max:
            raise Stop("INSTRUMENT_UNREADABLE",
                       "%s accepted only %d of %d spike positions"
                       % (series, len(accepted), n_max))
        # Kept in acceptance order, NOT sorted by index: the nested dose rule
        # takes the prefix of the *draw*, and sorting here would make the
        # smaller rate systematically pick the earliest positions in the
        # block instead of a uniform sample of it.
        ledger[series] = accepted
        skips[series] = skipped
    return {
        "seed": SEED,
        "legal_index_range": [legal_low, legal_high],
        "min_event_spacing": MIN_EVENT_SPACING,
        "boundary_exclusion": BOUNDARY_EXCLUSION,
        "spike_cycle": [[s, m] for s, m in SPIKE_CYCLE],
        "n_spikes_at_max_rate": n_max,
        "nested_dose": (
            "positions are drawn once for the largest rate; a smaller rate "
            "takes the ordered prefix, so the rates differ in dose and not "
            "in placement"
        ),
        "filter_order": ["block fit", "minimum spacing",
                         "sigma_local validity", "target finiteness"],
        "delta_scale_dependence": (
            "every delta = sign * sigma_multiple * sigma_local(pristine); no "
            "injected value is ever used as a scale source"
        ),
        "ledger": ledger,
        "skips": skips,
    }


def spikes_for_rate(rows: list[dict[str, Any]], rate: float
                    ) -> list[dict[str, Any]]:
    """Prefix of the draw order, then sorted by index for application."""
    count = int(round(rate * TRAIN_WINDOWS / WINDOW))
    return sorted(rows[:count], key=lambda row: int(row["index"]))


def contaminate(block: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    """Apply the spikes to a copy of the training block.  Never in place."""
    out = np.array(block, dtype=np.float64, copy=True)
    for row in rows:
        offset = int(row["index"]) - TRAIN_BLOCK[0]
        out[offset] = out[offset] + float(row["magnitude"])
    return out


# =========================================================================== #
# the adapter: the frozen Consumer's primitives, plus abstention and masking
# =========================================================================== #
def adapter_fit(block: np.ndarray, drop_offsets: set[int]) -> dict[str, Any]:
    """Fit the frozen Consumer on a block with some points masked out.

    ``drop_offsets`` are block-relative indices excluded from the fit: out of
    the standardization sample and out of every window they touch.  With an
    empty mask and no missing data this is ``consumer.fit_series`` exactly --
    ``assert_adapter_matches_consumer`` proves it.
    """
    keep_points = np.ones(block.size, dtype=bool)
    if drop_offsets:
        keep_points[np.fromiter(drop_offsets, dtype=np.int64,
                                count=len(drop_offsets))] = False
    constants = consumer.standardization(block[keep_points])
    matrix = consumer._windows(consumer._apply(block, constants))
    if matrix.shape[0] == 0:
        raise Stop("INSTRUMENT_UNREADABLE", "training block shorter than %d"
                   % WINDOW)
    masked_window = np.zeros(matrix.shape[0], dtype=bool)
    for offset in drop_offsets:
        low = max(0, int(offset) - WINDOW + 1)
        high = min(matrix.shape[0] - 1, int(offset))
        if low <= high:
            masked_window[low:high + 1] = True
    finite_window = np.isfinite(matrix).all(axis=1)
    keep = finite_window & ~masked_window
    if int(np.count_nonzero(keep)) == 0:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "every training window was masked or non-finite")
    forest = IsolationForest(**consumer.FOREST_KWARGS)
    forest.fit(matrix[keep])
    return {
        "forest": forest,
        "constants": dict(constants),
        "training_windows": int(np.count_nonzero(keep)),
        "windows_total": int(matrix.shape[0]),
        "windows_dropped_non_finite": int(
            np.count_nonzero(~finite_window)),
        "windows_dropped_masked": int(
            np.count_nonzero(masked_window & finite_window)),
        "points_masked": len(drop_offsets),
    }


def adapter_score(model: dict[str, Any], series: np.ndarray,
                  region: tuple[int, int],
                  truth_rows: list[list[int]]) -> dict[str, Any]:
    """Score [lo, hi) of the raw series; non-finite windows abstain.

    The event arithmetic is the frozen Consumer's own, so this reading is in
    the same semantics as every other AD reading in the line.
    """
    low, high = int(region[0]), int(region[1])
    if low < WINDOW - 1:
        raise ValueError("region start %d leaves no full trailing window" % low)
    fed = series[low - (WINDOW - 1):high]
    matrix = consumer._windows(consumer._apply(fed, model["constants"]))
    finite = np.isfinite(matrix).all(axis=1)
    scores = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    if int(np.count_nonzero(finite)):
        scores[finite] = model["forest"].decision_function(matrix[finite])
    flags = np.zeros(matrix.shape[0], dtype=bool)
    flags[finite] = scores[finite] < consumer.DECISION_THRESHOLD
    indices = np.arange(low, high, dtype=np.int64)
    assert indices.size == matrix.shape[0]
    predicted = consumer.merge_events(indices, flags)
    truth = [[int(r) for r in rows if low <= int(r) < high]
             for rows in truth_rows]
    truth = [rows for rows in truth if rows]
    row = dict(consumer.event_f1(truth, predicted))
    labels = consumer._point_labels(indices, truth)
    row.update({
        "region": [low, high],
        "predicted_event_spans": [[int(s), int(e)] for s, e in predicted],
        "flagged_points": int(np.count_nonzero(flags)),
        "scored_points": int(np.count_nonzero(finite)),
        "abstained_points": int(np.count_nonzero(~finite)),
        "auprc": consumer.auprc(-scores[finite], labels[finite]),
        "background_alarm_rate": consumer.background_alarm_rate(
            indices[finite], flags[finite],
            [[r for r in rows if r in set(indices[finite].tolist())]
             for rows in truth]),
    })
    return row


def assert_adapter_matches_consumer(pristine: dict[str, np.ndarray],
                                    names: list[str],
                                    budget: FitBudget) -> dict[str, Any]:
    """The adapter must be the frozen Consumer when nothing is masked.

    Only stations whose training block has no missing point can carry the
    assertion -- with a NaN present the frozen Consumer cannot run at all,
    which is the very reason the adapter exists.
    """
    checked: list[dict[str, Any]] = []
    for series in names:
        block = pristine[series][TRAIN_BLOCK[0]:TRAIN_BLOCK[1]]
        if int(np.count_nonzero(~np.isfinite(block))):
            continue
        budget.spend("instrument_check")
        reference = consumer.fit_series(block)
        budget.spend("instrument_check")
        adapted = adapter_fit(block, set())
        probe = consumer._windows(consumer._apply(block, adapted["constants"]))
        same_constants = reference["constants"] == adapted["constants"]
        same_windows = (int(reference["training_windows"])
                        == int(adapted["training_windows"]))
        same_scores = np.array_equal(
            reference["forest"].decision_function(probe),
            adapted["forest"].decision_function(probe))
        checked.append({
            "series": series,
            "constants_identical": bool(same_constants),
            "training_windows_identical": bool(same_windows),
            "decision_function_bitwise_identical": bool(same_scores),
        })
        if not (same_constants and same_windows and same_scores):
            raise Stop("INSTRUMENT_UNREADABLE",
                       "adapter diverges from the frozen Consumer on %s"
                       % series)
        if len(checked) >= 2:
            break
    if not checked:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "no station with a fully finite training block to anchor "
                   "the adapter equivalence assertion")
    return {"status": "ADAPTER_EQUALS_FROZEN_CONSUMER", "checked": checked}


# =========================================================================== #
# Part B -- the two judgment streams
# =========================================================================== #
def _macro(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _contrast(per_station: dict[str, dict[str, float]], better: str,
              worse: str) -> dict[str, Any]:
    deltas = {s: float(row[better]) - float(row[worse])
              for s, row in per_station.items()}
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


def run(argv: list[str]) -> int:
    geometry = verify_geometry()
    names = list(geometry["stations"])
    qcal_ledger = geometry["qcal_ledger"]

    pristine: dict[str, np.ndarray] = {}
    qcal_series: dict[str, np.ndarray] = {}
    for series in names:
        array = np.load(QCAL_DIR / ("%s.npy" % series))
        qcal_series[series] = array
        pristine[series] = array  # pristine on [0, 2600); Qcal events after
    qcal_truth = {
        series: [list(range(int(e["index"]), int(e["index"]) + int(e["points"])))
                 for e in qcal_ledger[series]]
        for series in names
    }

    budget = FitBudget(FIT_CAP)
    equivalence = assert_adapter_matches_consumer(pristine, names, budget)
    injection = freeze_injection(pristine, names)

    # -- substrates, isolated under a run id; never over the T1 originals ----
    for guard in (T1_DIR, QCAL_DIR, QF_DIR):
        if SCRATCH_OUT == guard or guard in SCRATCH_OUT.parents:
            raise Stop("INSTRUMENT_UNREADABLE",
                       "refusing to write inside the frozen injection tree")
    SCRATCH_OUT.mkdir(parents=True, exist_ok=True)
    (SCRATCH_OUT / "injection_ledger.json").write_text(
        _json_text(injection), encoding="utf-8")

    substrates: dict[tuple[str, float], dict[str, Any]] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        target = SCRATCH_OUT / tag
        target.mkdir(parents=True, exist_ok=True)
        for series in names:
            rows = spikes_for_rate(injection["ledger"][series], rate)
            block = pristine[series][TRAIN_BLOCK[0]:TRAIN_BLOCK[1]]
            dirty = contaminate(block, rows)
            np.save(target / ("%s.npy" % series), dirty)
            offsets = {int(r["index"]) - TRAIN_BLOCK[0] for r in rows}
            substrates[(series, rate)] = {
                "block": dirty, "rows": rows, "offsets": offsets}

    # -- determinism of the substrate construction (free, no fits) ----------
    substrate_determinism = True
    second_injection = freeze_injection(pristine, names)
    substrate_determinism &= (_json_text(second_injection["ledger"])
                              == _json_text(injection["ledger"]))
    for rate in RATES:
        for series in names:
            rows = spikes_for_rate(second_injection["ledger"][series], rate)
            block = pristine[series][TRAIN_BLOCK[0]:TRAIN_BLOCK[1]]
            again = contaminate(block, rows)
            substrate_determinism &= (
                again.tobytes()
                == substrates[(series, rate)]["block"].tobytes())

    # -- the three arms -----------------------------------------------------
    # clean does not depend on the rate, so it is fitted once per station
    clean_models: dict[str, Any] = {}
    for series in names:
        block = pristine[series][TRAIN_BLOCK[0]:TRAIN_BLOCK[1]]
        budget.spend("clean")
        clean_models[series] = adapter_fit(block, set())

    readings: dict[str, Any] = {}
    support: dict[str, Any] = {}
    fit_meta: dict[str, Any] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        readings[tag] = {series: {} for series in names}
        support[tag] = {series: {} for series in names}
        fit_meta[tag] = {series: {} for series in names}
        for series in names:
            raw = qcal_series[series]
            cell = substrates[(series, rate)]
            models = {"clean": clean_models[series]}
            budget.spend("contaminated_identity")
            models["contaminated_identity"] = adapter_fit(
                cell["block"], set())
            budget.spend("contaminated_repaired")
            models["contaminated_repaired"] = adapter_fit(
                cell["block"], cell["offsets"])
            support_truth = [[int(r["index"])] for r in cell["rows"]
                             if SUPPORT_WINDOW[0] <= int(r["index"])
                             < SUPPORT_WINDOW[1]]
            for arm in ARMS:
                model = models[arm]
                readings[tag][series][arm] = adapter_score(
                    model, raw, QCAL_REGION, qcal_truth[series])
                # Support reads the held-in substrate as it is: the repair is
                # a fit-time mask, not a rewrite, so arms 2 and 3 read the
                # same contaminated bytes and differ only by their model.
                held_in = (cell["block"] if arm != "clean"
                           else pristine[series][TRAIN_BLOCK[0]:TRAIN_BLOCK[1]])
                full = np.array(raw, dtype=np.float64, copy=True)
                full[TRAIN_BLOCK[0]:TRAIN_BLOCK[1]] = held_in
                support[tag][series][arm] = adapter_score(
                    model, full, SUPPORT_WINDOW, support_truth)
                fit_meta[tag][series][arm] = {
                    k: v for k, v in model.items() if k != "forest"}
                fit_meta[tag][series][arm]["constants"] = dict(
                    model["constants"])
            support[tag][series]["truth_events"] = len(support_truth)
            support[tag][series]["truth_indices"] = [
                int(r[0]) for r in support_truth]
            fit_meta[tag][series]["spikes"] = cell["rows"]
            fit_meta[tag][series]["contaminated_windows"] = int(
                models["contaminated_repaired"]["windows_dropped_masked"]
                + 0)

    # -- model-level determinism recheck on two stations --------------------
    recheck_names = names[:2]
    model_determinism = True
    recheck_detail: list[dict[str, Any]] = []
    for series in recheck_names:
        raw = qcal_series[series]
        for rate in RATES:
            cell = substrates[(series, rate)]
            for arm, drop in (("contaminated_identity", set()),
                              ("contaminated_repaired", cell["offsets"])):
                budget.spend("determinism_recheck")
                again = adapter_fit(cell["block"], drop)
                reading = adapter_score(again, raw, QCAL_REGION,
                                        qcal_truth[series])
                same = (reading == readings[RATE_TAGS[rate]][series][arm])
                model_determinism &= same
                recheck_detail.append({"series": series, "rate": rate,
                                       "arm": arm, "identical": bool(same)})
        budget.spend("determinism_recheck")
        again = adapter_fit(
            pristine[series][TRAIN_BLOCK[0]:TRAIN_BLOCK[1]], set())
        reading = adapter_score(again, raw, QCAL_REGION, qcal_truth[series])
        same = (reading == readings[RATE_TAGS[RATES[0]]][series]["clean"])
        model_determinism &= same
        recheck_detail.append({"series": series, "rate": None,
                               "arm": "clean", "identical": bool(same)})

    # -- B1 -----------------------------------------------------------------
    b1: dict[str, Any] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        f1 = {series: {arm: float(readings[tag][series][arm]["f1"])
                       for arm in ARMS} for series in names}
        harm = _contrast(f1, "contaminated_identity", "clean")
        recovery = _contrast(f1, "contaminated_repaired",
                             "contaminated_identity")
        to_clean = _contrast(f1, "contaminated_repaired", "clean")
        lost = -float(harm["macro_delta"])
        regained = float(recovery["macro_delta"])
        b1[tag] = {
            "rate": rate,
            "n_spikes": int(round(rate * TRAIN_WINDOWS / WINDOW)),
            "realized_window_rate": (
                int(round(rate * TRAIN_WINDOWS / WINDOW)) * WINDOW
                / TRAIN_WINDOWS),
            "realized_point_rate": (
                int(round(rate * TRAIN_WINDOWS / WINDOW))
                / (TRAIN_BLOCK[1] - TRAIN_BLOCK[0])),
            "macro_f1": {arm: _macro([f1[s][arm] for s in names])
                         for arm in ARMS},
            "contamination_harm_vs_clean": harm,
            "repair_recovery_vs_contaminated": recovery,
            "repair_vs_clean": to_clean,
            "contamination_readable": bool(lost >= B1_MACRO_BAR),
            "recovered_share_of_loss": (
                float(regained / lost) if lost > 1e-12 else None),
            "gate": {"macro_ge": B1_MACRO_BAR, "harmed_le": B1_HARMED_MAX,
                     "worst_ge": B1_WORST_FLOOR},
            "verdict": ("EFFECT_CONFIRMED" if _b1_gate(recovery)
                        else "EFFECT_NOT_CONFIRMED"),
        }
    confirmed_rates = [tag for tag, row in b1.items()
                       if row["verdict"] == "EFFECT_CONFIRMED"]

    # -- B2, only for the rates that passed B1 ------------------------------
    b2: dict[str, Any] = {}
    for tag in confirmed_rates:
        rows: dict[str, Any] = {}
        event_bearing = [s for s in names
                         if int(support[tag][s]["truth_events"]) > 0]
        zero_event = [s for s in names if s not in event_bearing]
        for name, orientation in SIGNALS.items():
            ordered = 0
            usable = 0
            per_series: dict[str, Any] = {}
            for series in event_bearing:
                truth_events = int(support[tag][series]["truth_events"])
                a = _signal_value(
                    name, support[tag][series]["contaminated_identity"],
                    truth_events)
                b = _signal_value(
                    name, support[tag][series]["contaminated_repaired"],
                    truth_events)
                if a is None or b is None:
                    per_series[series] = {"contaminated": a, "repaired": b,
                                          "usable": False}
                    continue
                usable += 1
                correct = (b > a) if orientation == "higher" else (b < a)
                ordered += int(correct)
                per_series[series] = {"contaminated": a, "repaired": b,
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
                            if share is not None and share >= SIGNAL_ORDER_SHARE
                            else "SIGNAL_NOT_PREDICTIVE"),
                "per_series": per_series,
            }
        # the conservative-policy route: pick per station what the signal
        # prefers, then put that policy through the same B1 gate
        f1 = {series: {arm: float(readings[tag][series][arm]["f1"])
                       for arm in ARMS} for series in names}
        for name, row in rows.items():
            picks = {}
            for series in names:
                detail = row["per_series"].get(series)
                if detail is None or not detail.get("usable"):
                    picks[series] = "contaminated_identity"  # conservative
                else:
                    picks[series] = ("contaminated_repaired"
                                     if detail["correct"]
                                     else "contaminated_identity")
            policy = {series: {"policy": f1[series][picks[series]],
                               "contaminated_identity":
                                   f1[series]["contaminated_identity"]}
                      for series in names}
            contrast = _contrast(policy, "policy", "contaminated_identity")
            row["policy_route"] = {
                "picks": picks,
                "contrast_vs_contaminated_identity": contrast,
                "passes_b1_gate": bool(_b1_gate(contrast)),
            }
            if row["verdict"] != "SIGNAL_PREDICTIVE" and row[
                    "policy_route"]["passes_b1_gate"]:
                row["verdict"] = "SIGNAL_PREDICTIVE"
                row["predictive_route"] = "policy"
            elif row["verdict"] == "SIGNAL_PREDICTIVE":
                row["predictive_route"] = "ordering"
        b2[tag] = {
            "support_window": list(SUPPORT_WINDOW),
            "event_bearing_series": event_bearing,
            "zero_event_series": zero_event,
            "zero_event_rule": (
                "a zero-event Support window supplies false-alarm harm "
                "evidence only and never authorises a positive adoption"
            ),
            "zero_event_background_alarm": {
                series: {
                    arm: support[tag][series][arm]["background_alarm_rate"]
                    for arm in ARMS}
                for series in zero_event
            },
            "signals": rows,
        }

    predictive = {
        tag: [name for name, row in b2[tag]["signals"].items()
              if row["verdict"] == "SIGNAL_PREDICTIVE"]
        for tag in confirmed_rates
    }

    # -- did the contamination actually reach the fit? ----------------------
    bite: dict[str, Any] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        per: dict[str, Any] = {}
        for series in names:
            meta = fit_meta[tag][series]
            clean_std = float(meta["clean"]["constants"]["std"])
            dirty_std = float(
                meta["contaminated_identity"]["constants"]["std"])
            repaired_std = float(
                meta["contaminated_repaired"]["constants"]["std"])
            per[series] = {
                "std_clean": clean_std,
                "std_contaminated": dirty_std,
                "std_repaired": repaired_std,
                "std_inflation_from_contamination": dirty_std / clean_std,
                "std_residual_after_repair": repaired_std / clean_std,
                "windows_masked_by_repair": int(
                    meta["contaminated_repaired"]["windows_dropped_masked"]),
                "fit_windows_clean": int(meta["clean"]["training_windows"]),
                "fit_windows_repaired": int(
                    meta["contaminated_repaired"]["training_windows"]),
            }
        inflation = [per[s]["std_inflation_from_contamination"]
                     for s in names]
        residual = [per[s]["std_residual_after_repair"] for s in names]
        bite[tag] = {
            "reading": (
                "whether the injected contamination changed the object the "
                "Consumer actually fits, and whether the oracle mask undid it"
            ),
            "median_std_inflation_from_contamination": float(
                np.median(inflation)),
            "max_std_inflation_from_contamination": float(max(inflation)),
            "median_std_residual_after_repair": float(np.median(residual)),
            "contamination_reached_the_fit": bool(
                float(np.median(inflation)) > 1.02),
            "repair_restored_the_fit": bool(
                abs(float(np.median(residual)) - 1.0) < 0.02),
            "per_series": per,
        }

    # -- is the delayed estimand able to resolve anything? ------------------
    calibration: dict[str, Any] = {}
    for rate in RATES:
        tag = RATE_TAGS[rate]
        per = {}
        for series in names:
            row = {}
            for arm in ARMS:
                reading = readings[tag][series][arm]
                scored = int(reading["scored_points"])
                row[arm] = {
                    "flagged_share": (int(reading["flagged_points"]) / scored
                                      if scored else None),
                    "predicted_events": int(reading["predicted_events"]),
                    "truth_events": int(reading["truth_events"]),
                }
            per[series] = row
        shares = [per[s]["clean"]["flagged_share"] for s in names
                  if per[s]["clean"]["flagged_share"] is not None]
        calibration[tag] = {
            "reading": (
                "share of the Qcal region the Consumer flags.  A detector "
                "that flags most of the Query is saturated: its event F1 is "
                "then decided by where the flag run happens to break, not by "
                "what it learned from the training substrate"
            ),
            "median_flagged_share_clean_arm": float(np.median(shares)),
            "min_flagged_share_clean_arm": float(min(shares)),
            "max_flagged_share_clean_arm": float(max(shares)),
            "saturated_series_clean_arm": int(
                sum(1 for v in shares if v >= 0.5)),
            "series_total": len(names),
            "per_series": per,
        }

    # -- routing ------------------------------------------------------------
    if not confirmed_rates:
        readable = [tag for tag, row in b1.items()
                    if row["contamination_readable"]]
        reached = all(bite[tag]["contamination_reached_the_fit"]
                      for tag in bite)
        restored = all(bite[tag]["repair_restored_the_fit"] for tag in bite)
        saturated = all(
            calibration[tag]["median_flagged_share_clean_arm"] >= 0.5
            for tag in calibration)
        verdict = "PROGRAM_CONSUMER_LAYER_FAULT"
        parts = ["no rate produced a readable repair effect, so the first "
                 "fault is at the Program/Consumer layer"]
        if not readable:
            parts.append(
                "the contamination did not materially harm the Consumer "
                "either, so nothing was there to repair")
        if reached and restored:
            parts.append(
                "this is NOT an injection failure: the contamination did "
                "change the fitted object (median std inflation %.3fx) and "
                "the oracle mask did restore it (median std residual %.3fx), "
                "so the exam's mechanics worked and the delayed reading "
                "still did not move" % (
                    float(np.median([
                        bite[t]["median_std_inflation_from_contamination"]
                        for t in bite])),
                    float(np.median([
                        bite[t]["median_std_residual_after_repair"]
                        for t in bite]))))
        if saturated:
            parts.append(
                "the delayed estimand is the suspect: the Consumer flags a "
                "median %.0f%% of the Qcal region even in the clean arm, so "
                "its event F1 is decided by where a near-continuous flag run "
                "breaks rather than by what it learned from the substrate"
                % (100.0 * float(np.median([
                    calibration[t]["median_flagged_share_clean_arm"]
                    for t in calibration]))))
        routing = "; ".join(parts)
    elif not any(predictive.values()):
        verdict = "FEEDBACK_PROTOCOL_LAYER_FAULT"
        routing = ("the repair effect is real but no pre-registered Support "
                   "signal predicts it; the first fault is at the feedback "
                   "protocol layer")
    else:
        verdict = "EFFECT_CONFIRMED_AND_SIGNAL_PREDICTIVE"
        routing = ("a repair effect exists and at least one Support signal "
                   "predicts it; the next probe is the Observation/selection "
                   "layer (can the Agent find it)")

    obligations = {
        "llm_calls": 0,
        "fit_budget_used": budget.used,
        "fit_budget_cap": FIT_CAP,
        "fit_budget_respected": bool(budget.used <= FIT_CAP),
        "fits_by_arm": dict(budget.by_arm),
        "yahoo_sealed_41_reads": 0,
        "yahoo_reads": 0,
        "noaa_2025_new_reads": 0,
        "nab_smd_beyond_17520_reads": 0,
        "data_directory_opened": False,
        "qf_opened": False,
        "frozen_injection_tree_written": False,
        "substrate_determinism": bool(substrate_determinism),
        "model_determinism": bool(model_determinism),
        "adapter_equivalence": equivalence["status"],
        "rates_reported": [RATE_TAGS[r] for r in RATES],
        "rate_cherry_picking": False,
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "entry": "--run",
        "book": "#44a AD feedback positive control",
        "evidence_class": "INSTRUMENT / POSITIVE_CONTROL",
        "development_only": True,
        "claim_cap": (
            "a development positive control on injected NOAA copies; its "
            "conclusions never enter a natural-Yahoo capability claim and "
            "say nothing about whether natural data carries removable "
            "contamination"
        ),
        "clean_arm_naming": (
            "the clean arm is a *reference*, not an upper bound; the short "
            "canon forbids calling it a bound before the reading exists"
        ),
        "geometry": geometry,
        "consumer": {
            "module": "consumers/aegists_iforest_v1.py",
            "spec": consumer.spec(),
            "adapter": (
                "the runner adds two things the frozen Consumer does not "
                "have and must not be edited to have: an oracle fit mask, "
                "and the T0/v3 abstention canon for non-finite windows "
                "(never fitted, forced not to flag on the Query, excluded "
                "from the AUPRC ranking)"
            ),
            "adapter_equivalence": equivalence,
        },
        "injection": {
            k: v for k, v in injection.items() if k not in {"ledger", "skips"}
        },
        "injection_ledger": injection["ledger"],
        "injection_skips": injection["skips"],
        "rate_semantics": {
            "definition": "fraction of the 761 training windows reached",
            "why": (
                "the Consumer is window-based and the repair is a "
                "window-level mask; under a point-fraction reading r = 15% "
                "would cover every window and leave the repaired arm nothing "
                "to fit, so the exam would test nothing"
            ),
            "training_windows": TRAIN_WINDOWS,
            "per_rate": {RATE_TAGS[r]: {
                "target_window_rate": r,
                "n_spikes": int(round(r * TRAIN_WINDOWS / WINDOW)),
                "realized_window_rate": (
                    int(round(r * TRAIN_WINDOWS / WINDOW)) * WINDOW
                    / TRAIN_WINDOWS),
                "realized_point_rate": (
                    int(round(r * TRAIN_WINDOWS / WINDOW))
                    / (TRAIN_BLOCK[1] - TRAIN_BLOCK[0])),
            } for r in RATES},
        },
        "arms": {
            "clean": "fit on the pristine training block",
            "contaminated_identity": "fit on the contaminated block, no repair",
            "contaminated_repaired": (
                "fit on the contaminated block with the known injected "
                "positions masked out of the standardization sample and out "
                "of every window they touch"),
        },
        "per_series_readings": {
            tag: {series: {arm: readings[tag][series][arm] for arm in ARMS}
                  for series in names}
            for tag in readings
        },
        "per_series_support": support,
        "per_series_fits": fit_meta,
        "b1": b1,
        "b2": b2,
        "signals_predictive": predictive,
        "contamination_bite": bite,
        "query_calibration": calibration,
        "determinism": {
            "substrate_two_constructions_identical": bool(
                substrate_determinism),
            "model_recheck_identical": bool(model_determinism),
            "model_recheck_detail": recheck_detail,
        },
        "verdict": {
            "verdict": verdict,
            "routing": routing,
            "effect_confirmed_rates": confirmed_rates,
            "predictive_signals": predictive,
        },
        "cost": {
            "llm": 0,
            "ad_fits": budget.used,
            "ad_fit_cap": FIT_CAP,
            "ad_fits_by_arm": dict(budget.by_arm),
        },
        "obligations": obligations,
    }
    OUT_JSON.write_text(_json_text(payload), encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "b1": {tag: row["verdict"] for tag, row in b1.items()},
        "macro_f1": {tag: row["macro_f1"] for tag, row in b1.items()},
        "predictive_signals": predictive,
        "determinism": {"substrate": substrate_determinism,
                        "model": model_determinism},
        "fits": budget.used,
        "fits_by_arm": dict(budget.by_arm),
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_JSON)
    return 0


def _markdown(payload: dict[str, Any]) -> str:
    geometry = payload["geometry"]
    names = list(geometry["stations"])
    lines = [
        "# #44a -- AD feedback positive control",
        "",
        "evidence class: %s (development).  %s" % (
            payload["evidence_class"], payload["claim_cap"]),
        "",
        "## Verdict",
        "",
        "- **%s**" % payload["verdict"]["verdict"],
        "- routing: %s" % payload["verdict"]["routing"],
        "- B1 by rate: %s" % {tag: row["verdict"]
                              for tag, row in payload["b1"].items()},
        "- predictive Support signals: %s" % (
            payload["verdict"]["predictive_signals"] or "none evaluated"),
        "",
        "## Geometry, as understood and verified",
        "",
        "- series length %d (hourly year), inside the 17520 wall; 12 NOAA "
        "stations from the T1 injection family." % geometry["series_length"],
        "- training substrate / P's action region: %s" % geometry[
            "training_block"],
        "- Qcal (the only region scored here): %s, four known events per "
        "station." % geometry["qcal_region"],
        "- Qf: %s -- **not opened by this book**, kept as an independent "
        "confirmation surface." % geometry["qf_region"],
        "- Support window: %s.  %s" % (geometry["support_window"],
                                       geometry["support_window_basis"]),
        "- substrate choice: %s" % geometry["substrate_choice"],
        "- verification: all 12 stations pass -- the t1 copy differs from the "
        "qcal copy exactly at the union of their ledger points, the qf and "
        "qcal copies are byte-identical on the training block, and no "
        "injected event point is missing.",
        "",
        "## Three arms x two rates: Qcal macro event F1",
        "",
        "| rate | spikes | window rate | point rate | clean | contaminated | "
        "repaired |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, row in payload["b1"].items():
        macro = row["macro_f1"]
        lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            tag, row["n_spikes"], row["realized_window_rate"],
            row["realized_point_rate"], macro["clean"],
            macro["contaminated_identity"], macro["contaminated_repaired"]))
    lines.extend([
        "",
        "## B1: did contamination hurt, and did the oracle repair recover it?",
        "",
        "| rate | harm (contaminated − clean) | recovery (repaired − "
        "contaminated) | harmed | worst | recovered share of loss | verdict |",
        "|---|---|---|---|---|---|---|",
    ])
    for tag, row in payload["b1"].items():
        harm = row["contamination_harm_vs_clean"]
        rec = row["repair_recovery_vs_contaminated"]
        share = row["recovered_share_of_loss"]
        lines.append("| %s | %+.6f | %+.6f | %d | %+.4f | %s | %s |" % (
            tag, harm["macro_delta"], rec["macro_delta"], rec["harmed"],
            rec["worst"],
            "n/a" if share is None else "%.3f" % share, row["verdict"]))
    lines.extend([
        "",
        "gate: macro Δ ≥ %+.3f, harmed ≤ %d/12, worst ≥ %+.3f" % (
            B1_MACRO_BAR, B1_HARMED_MAX, B1_WORST_FLOOR),
        "",
        "## Per-station Qcal event F1",
        "",
    ])
    for tag in payload["per_series_readings"]:
        lines.extend([
            "### rate %s" % tag,
            "",
            "| station | clean | contaminated | repaired | repair − "
            "contaminated | contaminated − clean |",
            "|---|---|---|---|---|---|",
        ])
        for series in names:
            cell = payload["per_series_readings"][tag][series]
            clean = float(cell["clean"]["f1"])
            dirty = float(cell["contaminated_identity"]["f1"])
            fixed = float(cell["contaminated_repaired"]["f1"])
            lines.append("| %s | %.4f | %.4f | %.4f | %+.4f | %+.4f |" % (
                series, clean, dirty, fixed, fixed - dirty, dirty - clean))
        lines.append("")
    if payload["b2"]:
        lines.extend(["## B2: Support signal predictiveness", ""])
        for tag, block in payload["b2"].items():
            lines.extend([
                "### rate %s (event-bearing Support windows: %d/12)" % (
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
            "B1 did not confirm an effect at either rate, so the Support "
            "signal stream was not run -- the book stops the round at the "
            "Program/Consumer layer rather than hunting for a signal that "
            "predicts an effect that is not there.",
            "",
        ])
    lines.extend([
        "## Did the contamination reach the fit, and did the mask undo it?",
        "",
        "| rate | median std inflation | max | median std residual after "
        "repair | reached fit | repair restored |",
        "|---|---|---|---|---|---|",
    ])
    for tag, row in payload["contamination_bite"].items():
        lines.append("| %s | %.4f | %.4f | %.4f | %s | %s |" % (
            tag, row["median_std_inflation_from_contamination"],
            row["max_std_inflation_from_contamination"],
            row["median_std_residual_after_repair"],
            row["contamination_reached_the_fit"],
            row["repair_restored_the_fit"]))
    lines.extend([
        "",
        "## Can the delayed estimand resolve anything? (Qcal flag saturation)",
        "",
        "| rate | median flagged share (clean arm) | min | max | series with "
        "≥50% flagged |",
        "|---|---|---|---|---|",
    ])
    for tag, row in payload["query_calibration"].items():
        lines.append("| %s | %.3f | %.3f | %.3f | %d/%d |" % (
            tag, row["median_flagged_share_clean_arm"],
            row["min_flagged_share_clean_arm"],
            row["max_flagged_share_clean_arm"],
            row["saturated_series_clean_arm"], row["series_total"]))
    lines.append("")

    determinism = payload["determinism"]
    cost = payload["cost"]
    lines.extend([
        "## Determinism",
        "",
        "- substrate: two independent constructions byte-identical: **%s**"
        % determinism["substrate_two_constructions_identical"],
        "- model/reading recheck on 2 stations x 2 rates x 3 arms identical: "
        "**%s**" % determinism["model_recheck_identical"],
        "- adapter equals the frozen Consumer when nothing is masked: **%s**"
        % payload["consumer"]["adapter_equivalence"]["status"],
        "",
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
    if "--verify-geometry" in argv:
        print(_json_text({k: v for k, v in verify_geometry().items()
                          if k != "qcal_ledger"}))
        return 0
    if "--run" in argv:
        return run(argv)
    print("usage: --run | --verify-geometry")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stop as exc:  # noqa: BLE001
        print(json.dumps({"verdict": exc.code, "detail": exc.detail},
                         ensure_ascii=False, indent=1))
        raise SystemExit(3) from exc
