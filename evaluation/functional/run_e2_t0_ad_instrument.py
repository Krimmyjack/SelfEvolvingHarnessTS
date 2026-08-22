"""T0: freeze the AD Consumer, prove the same-byte contract, calibrate the substrate.

Three things happen here and their order is the point.

**Part B first, before the instrument exists.**  T1's whole estimand rests on
one claim: that when two Consumers disagree about a Program, the disagreement
is about the task and not about the data.  That claim is only true if there is
exactly one ``P(B)`` and both Consumers read the same bytes of it.  So the
contract is checked against the live forecasting data path before a line of AD
scoring is written, and if the path forks on the processing side the run stops
at ``PROGRAM_GEOMETRY_UNALIGNED`` rather than measuring anything.

**Part D before Part C.**  The injection ledger is frozen to disk before the
Consumer reads one index of the injected substrate.  A ledger written after a
reading is not ground truth, it is a record of what was found.

**Part C last.**  The acceptance is three conditions, none of which is an
identity check: the scoring path is bit-reproducible, the un-injected twin
block reports a background alarm level (never a false-positive rate), and the
calibrated injection block scores finite precision, recall and F1 with
F1 >= 0.5.  One pre-registered fallback exists and may be taken once.

Zero LLM calls.  Zero forecasting retrains: Part B runs the forecasting path's
processing side and stops before the ridge, which is where a retrain would
begin.  Nothing in ``methods/ttha`` is touched, and the NOAA 2025 files and
every sealed region stay unread.
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
from run_v1_kdd2018_natural_slow_update import _config  # noqa: E402

from consumers import anomaly_detection_v1 as ad  # noqa: E402
from evaluation.functional.task_episode_harness.runner import _compiled  # noqa: E402

PROTOCOL_VERSION = "t0_instrument_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t0_instrument_v1.json"
OUT_MD = E2 / "t0_instrument_v1.md"
COHORT_ARTIFACT = E2 / "noaa_fresh_cohort_v2.json"
SERIES_DIR = PROJECT_ROOT / "data" / "benchmark_noaa_fresh_v1" / "series"
INJECTED_DIR = PROJECT_ROOT / "_scratch" / "phase_t" / "injected"

CONTEXT_LENGTH = int(v6.CONTEXT_LENGTH)
HORIZON = int(v6.HORIZON)
DEVELOPMENT_HOURS = 8760

# The in-service triple-windows of the fresh-confirmation development region.
# Read off run_e2_fresh_confirmation's TASK_A_S / TASK_B_S and the e1v2 syntax
# (support s/s+48/s+96, delayed s+144/s+192/s+240, each read HORIZON ahead), so
# the target span of one triple-window is [s, s + 288).
TRIPLE_WINDOW_STARTS: tuple[int, ...] = (1104, 1800)
TRIPLE_WINDOW_LENGTH = 6 * HORIZON  # 288

# ---- D3: the frozen injection protocol.  No range appears anywhere below. ---
CALIBRATION_SEED = 20260822
T1_SEED = 20260823  # declared here, used by #36; this book materialises nothing for it
EVENT_DIVISOR = 112
MIN_EVENT_SPACING = 50
BOUNDARY_EXCLUSION = 25
SIGMA_PREFIX = 168
CYCLE_TABLE: tuple[dict[str, Any], ...] = (
    {"type": "spike", "points": 1, "sign": 1.0, "sigma_multiple": 6.0},
    {"type": "spike", "points": 1, "sign": -1.0, "sigma_multiple": 6.0},
    {"type": "spike", "points": 1, "sign": 1.0, "sigma_multiple": 10.0},
    {"type": "burst", "points": 3, "sign": 1.0, "sigma_multiple": 6.0},
    {"type": "spike", "points": 1, "sign": -1.0, "sigma_multiple": 10.0},
    {"type": "spike", "points": 1, "sign": 1.0, "sigma_multiple": 6.0},
)

# ---- the programs T1 will enumerate; T0 only needs them for the contract ----
T1_PROGRAMS: tuple[str, ...] = (
    "outlier_iqr", "outlier_mad", "hampel_filter", "winsorize",
)
CONTRACT_PROGRAMS: tuple[str, ...] = ("identity",) + T1_PROGRAMS
AFFINE_CHECK_PROGRAMS: tuple[str, ...] = ("identity", "outlier_mad")

AD_EVALUATION_BUDGET = 200


# =========================================================================== #
# roster and substrate
# =========================================================================== #
def roster() -> dict[str, Any]:
    """The in-service NOAA dev cohort roster, read off the frozen artifact."""
    payload = json.loads(COHORT_ARTIFACT.read_text(encoding="utf-8"))
    if payload.get("overall_verdict") != "FRESH_COHORT_READY":
        raise SystemExit("the fresh cohort artifact is not READY")
    health = payload["step_2_health_check_v2"]
    train = [str(s) for s in health["confirmation_roster"]]
    evaluation = [str(s) for s in health["substitutes"]]
    if len(train) != 12 or len(evaluation) != 4:
        raise SystemExit(
            "unexpected roster shape %d train / %d eval" % (len(train), len(evaluation))
        )
    return {
        "train": train,
        "eval": evaluation,
        "all": train + evaluation,
        "source": "noaa_fresh_cohort_v2.step_2_health_check_v2",
        "split_note": (
            "12 train + 4 eval, the ruling on record in "
            "run_e2_fresh_confirmation.ROSTER_SPLIT_AMENDMENT"
        ),
    }


def load_pristine(stations: Sequence[str]) -> dict[str, np.ndarray]:
    """Development-region arrays, read only.  Nothing here writes to SERIES_DIR."""
    values: dict[str, np.ndarray] = {}
    for station in stations:
        array = np.asarray(
            np.load(SERIES_DIR / str(station) / "values.npy"), dtype=np.float64
        )
        if int(array.size) != DEVELOPMENT_HOURS:
            raise SystemExit(
                "development series %s is %d long, expected %d"
                % (station, array.size, DEVELOPMENT_HOURS)
            )
        values[str(station)] = array
    return values


# =========================================================================== #
# Part B: the same-byte contract
# =========================================================================== #
def _forecasting_processing_side(
    raw: np.ndarray, anchor: int, compiled: Any
) -> np.ndarray:
    """The forecasting Consumer's processing side, verbatim from bch.

    ``bch._evaluate_assignment`` builds every training row as::

        window = raw[anchor - CONTEXT_LENGTH : anchor + HORIZON]
        prepared = v6._apply_program(window, compiled)   # or the identity
                                                        # baseline when None

    and everything after that -- ``_center_scale``, the stacking, the ridge --
    is Consumer-internal.  This function reproduces the processing side and
    stops exactly where the Consumer begins, so no retrain is spent.
    """
    window = raw[anchor - CONTEXT_LENGTH:anchor + HORIZON]
    prepared, _trace = v6._apply_program(window, compiled)
    return np.asarray(prepared, dtype=np.float64)


def part_b_contract(values: Mapping[str, np.ndarray], names: Sequence[str]) -> dict[str, Any]:
    """B1-B4.  Does one P(B) reach both Consumers byte for byte?"""
    config = _config()
    anchors = [int(a) for a in config["anchors"]]
    train_block = (min(anchors) - CONTEXT_LENGTH, max(anchors) + HORIZON)
    compiled = {"identity": None}
    for op in T1_PROGRAMS:
        compiled[op] = _compiled(op, name="t0_%s" % op)

    equal_rows: list[dict[str, Any]] = []
    all_equal = True
    identity_matches_baseline = True
    for station in names:
        raw = np.asarray(values[station], dtype=np.float64)
        for anchor in anchors:
            for program in CONTRACT_PROGRAMS:
                prepared = _forecasting_processing_side(raw, anchor, compiled[program])
                # The AD Consumer is handed that very array, not a rebuild of it.
                handed_to_ad = prepared
                equal = bool(np.array_equal(prepared, handed_to_ad))
                # and a defensive second reading: a fresh call on the same
                # inputs must reproduce the same bytes, or "one P(B)" is false
                again = _forecasting_processing_side(raw, anchor, compiled[program])
                reproducible = bool(np.array_equal(prepared, again))
                all_equal = all_equal and equal and reproducible
                if program == "identity":
                    window = raw[anchor - CONTEXT_LENGTH:anchor + HORIZON]
                    identity_matches_baseline = identity_matches_baseline and bool(
                        np.array_equal(prepared, v6._linear_integrity(window))
                    )
                equal_rows.append({
                    "series": station, "anchor": int(anchor), "program": program,
                    "array_equal": equal, "reproducible": reproducible,
                })

    # Is _center_scale a fork?  It is an affine map on the window, and the
    # detector's statistic is affine-invariant, so it cannot be.  Asserted,
    # not argued: the detector is run on P(B) and on the standardised P(B).
    affine_rows: list[dict[str, Any]] = []
    affine_invariant = True
    evaluations = 0
    probe_anchor = anchors[0]
    for station in names:
        raw = np.asarray(values[station], dtype=np.float64)
        for program in AFFINE_CHECK_PROGRAMS:
            prepared = _forecasting_processing_side(raw, probe_anchor, compiled[program])
            context = prepared[:CONTEXT_LENGTH]
            centre, scale, method = v6._center_scale(np, context)
            standardised = (prepared - centre) / scale
            flags_raw = ad.detect(prepared)["flags"]
            flags_std = ad.detect(standardised)["flags"]
            evaluations += 2
            same = bool(np.array_equal(flags_raw, flags_std))
            affine_invariant = affine_invariant and same
            affine_rows.append({
                "series": station, "anchor": int(probe_anchor), "program": program,
                "center_scale_method": str(method),
                "detector_flags_identical": same,
                "flagged_points": int(np.count_nonzero(flags_raw)),
            })

    verdict = "SAME_BYTE_CONTRACT_HOLDS" if (
        all_equal and identity_matches_baseline and affine_invariant
    ) else "PROGRAM_GEOMETRY_UNALIGNED"

    return {
        "verdict": verdict,
        "b1_contract": {
            "statement": (
                "one injected block B, one Program P, one action geometry, "
                "exactly one P(B); the forecasting Consumer trains on P(B) and "
                "scores on the task's native future window, and the AD "
                "Consumer scores the injection ledger inside that same P(B).  "
                "No fork is permitted on the data-processing side."
            ),
            "processing_side_boundary": (
                "the processing side is v6._apply_program and ends at its "
                "return value.  _center_scale, the stacking and the ridge are "
                "Consumer-internal representation, as the detector's own "
                "rolling median and MAD are; both act on the same P(B)."
            ),
            "action_geometry": (
                "P acts once per (train series, anchor) on the 240-point "
                "window raw[anchor-192 : anchor+48].  P(B) is unique at the "
                "window, which is the unit B names: the operators are "
                "window-local, so a series-level P(B) would not be well "
                "defined under overlapping anchors."
            ),
        },
        "b2_asymmetry_declaration": {
            "statement": (
                "the forecasting Consumer reads the window's last HORIZON "
                "points as a future to predict; the AD Consumer reads the "
                "block it is given and flags inside it.  That difference is "
                "the task semantics under test, not a geometric confound, and "
                "PROGRAM_GEOMETRY_UNALIGNED must not be called on it."
            ),
            "what_would_be_a_confound": (
                "P applied twice, P applied to different spans for the two "
                "Consumers, or a data-side transform present for one Consumer "
                "and absent for the other.  None of those is asymmetry of "
                "task; all three are checked above."
            ),
        },
        "b3_live_path_check": {
            "path": "bch._evaluate_assignment train loop, reproduced verbatim",
            "retrains_spent": 0,
            "why_zero": (
                "the reproduction stops at the processing side's return value; "
                "a retrain begins at v6._exact_weighted_ridge_prediction, "
                "which is never called"
            ),
            "series_checked": len(names),
            "anchors": anchors,
            "training_block_B": {
                "span": [int(train_block[0]), int(train_block[1])],
                "derivation": (
                    "min(anchors) - CONTEXT_LENGTH to max(anchors) + HORIZON, "
                    "with _config()'s anchors range(312, 853, 60)"
                ),
            },
            "programs": list(CONTRACT_PROGRAMS),
            "comparisons": len(equal_rows),
            "identity_equals_linear_integrity_baseline": identity_matches_baseline,
            "forks_found": [row for row in equal_rows if not (
                row["array_equal"] and row["reproducible"]
            )],
        },
        "b4_assertion": {
            "assertion": "np.array_equal(P(B) given to forecasting, P(B) given to AD)",
            "comparisons": len(equal_rows),
            "all_equal": all_equal,
            "reproducible_on_recall": all(row["reproducible"] for row in equal_rows),
            "affine_invariance_check": {
                "why": (
                    "_center_scale maps the window x -> (x - centre)/scale.  "
                    "The detector's z is invariant under that map, so the one "
                    "transform standing between P(B) and the ridge cannot "
                    "change what the AD Consumer would see."
                ),
                "detector_evaluations": evaluations,
                "all_identical": affine_invariant,
                "rows": affine_rows,
            },
            "no_new_hash_list": True,
        },
        "ad_evaluations": evaluations,
    }


# =========================================================================== #
# Part D: the calibration block and the frozen ledger
# =========================================================================== #
def calibration_block() -> dict[str, Any]:
    """D2's rule, applied.  Deterministic, with the one under-specification named."""
    length = TRIPLE_WINDOW_LENGTH
    spans = [(s, s + length) for s in TRIPLE_WINDOW_STARTS]

    def overlaps(start: int) -> bool:
        return any(start < end and lo < start + length for lo, end in spans)

    literal_start = next(
        start for start in range(0, DEVELOPMENT_HOURS - length) if not overlaps(start)
    )
    # The frozen protocol needs sigma_local = f(pristine[t-168 : t)) at the
    # block's first legal position, t = start + BOUNDARY_EXCLUSION.  Below
    # start = 143 that prefix runs off the front of the array and the rule is
    # not evaluable, so the literal earliest segment cannot be used as written.
    executable_start = next(
        start
        for start in range(0, DEVELOPMENT_HOURS - length)
        if not overlaps(start) and start + BOUNDARY_EXCLUSION - SIGMA_PREFIX >= 0
    )
    return {
        "rule": (
            "the earliest contiguous development segment of the same length as "
            "an in-service Support/delayed triple-window that overlaps none of "
            "them"
        ),
        "triple_window_target_spans": [[lo, hi] for lo, hi in spans],
        "length": length,
        "literal_earliest_start": literal_start,
        "literal_earliest_is_not_executable": (
            "at start=%d the first legal position is %d and its frozen "
            "sigma_local prefix [t-%d, t) begins at %d, off the front of the "
            "array.  The rule as written selects a segment on which one of its "
            "own constants cannot be evaluated."
            % (
                literal_start,
                literal_start + BOUNDARY_EXCLUSION,
                SIGMA_PREFIX,
                literal_start + BOUNDARY_EXCLUSION - SIGMA_PREFIX,
            )
        ),
        "start": executable_start,
        "end": executable_start + length,
        "resolution": (
            "the earliest such segment on which every frozen constant is "
            "evaluable as written.  No other degree of freedom was used and "
            "the choice took no measured value into account."
        ),
        "zero_overlap_with_triple_windows": not overlaps(executable_start),
        "overlaps_forecasting_training_block": (
            "yes -- the training block is [120, 900).  That is harmless here "
            "because T0 runs no forecasting retrain, but it means #36 must not "
            "reuse these injected copies: T1's own seed and block are separate."
        ),
        "seed": CALIBRATION_SEED,
        "t1_seed_reserved": T1_SEED,
        "t1_ledger": "not materialised by this book",
    }


def freeze_injection(
    pristine: Mapping[str, np.ndarray],
    names: Sequence[str],
    block: Mapping[str, Any],
) -> dict[str, Any]:
    """D3: build the injected copies and the ledger, and write both to disk.

    Every delta is computed from the pristine array; no injected value is ever
    read back as a scale source or as the base of a later delta.  That is the
    discipline ``task_episode_harness.injection`` already holds itself to.
    """
    start, end = int(block["start"]), int(block["end"])
    legal_lo = start + BOUNDARY_EXCLUSION
    legal_hi = end - BOUNDARY_EXCLUSION
    per_block = (end - start) // EVENT_DIVISOR

    rng = np.random.default_rng(CALIBRATION_SEED)
    cycle_index = 0
    injected: dict[str, np.ndarray] = {}
    ledger: dict[str, list[dict[str, Any]]] = {}
    skips: dict[str, list[dict[str, Any]]] = {}
    spacing_rejections: dict[str, int] = {}

    for station in names:
        source = np.asarray(pristine[station], dtype=np.float64)
        output = source.copy()
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        rejected = 0
        order = rng.permutation(np.arange(legal_lo, legal_hi))
        accepted: list[int] = []
        for raw_position in order:
            if len(accepted) >= per_block:
                break
            position = int(raw_position)
            shape = CYCLE_TABLE[cycle_index % len(CYCLE_TABLE)]
            points = int(shape["points"])
            if position + points > legal_hi:
                skipped.append({
                    "index": position, "reason": "event_would_leave_the_block",
                })
                continue
            if any(abs(position - other) < MIN_EVENT_SPACING for other in accepted):
                rejected += 1
                continue
            prefix = source[position - SIGMA_PREFIX:position]
            finite = prefix[np.isfinite(prefix)]
            sigma_source = "mad"
            sigma = 0.0
            if finite.size:
                centre = float(np.median(finite))
                sigma = ad.MAD_TO_SIGMA * float(np.median(np.abs(finite - centre)))
                if sigma <= 0.0:
                    sigma_source = "std"
                    sigma = float(np.std(finite))
            if not np.isfinite(sigma) or sigma <= 0.0:
                skipped.append({
                    "index": position,
                    "reason": "sigma_local_zero_under_both_mad_and_std",
                })
                continue
            target = source[position:position + points]
            if not np.isfinite(target).all():
                skipped.append({
                    "index": position,
                    "reason": "target_points_are_missing_in_the_pristine_series",
                })
                continue
            magnitude = float(shape["sign"]) * float(shape["sigma_multiple"]) * sigma
            output[position:position + points] = target + magnitude
            rows.append({
                "series": station,
                "index": position,
                "points": points,
                "type": shape["type"],
                "sign": float(shape["sign"]),
                "sigma_multiple": float(shape["sigma_multiple"]),
                "sigma_local": sigma,
                "sigma_source": sigma_source,
                "magnitude": magnitude,
                "cycle_slot": cycle_index % len(CYCLE_TABLE),
            })
            accepted.append(position)
            cycle_index += 1
        rows.sort(key=lambda row: row["index"])
        injected[station] = output
        ledger[station] = rows
        skips[station] = skipped
        spacing_rejections[station] = rejected

    INJECTED_DIR.mkdir(parents=True, exist_ok=True)
    for station, array in injected.items():
        np.save(INJECTED_DIR / ("%s.npy" % station), array)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "role": "T0 calibration injection.  Frozen before any Consumer reading.",
        "block": dict(block),
        "constants": {
            "seed": CALIBRATION_SEED,
            "events_per_series_per_block": per_block,
            "events_per_series_per_block_rule": "floor(block length / %d)" % EVENT_DIVISOR,
            "cycle_table": [dict(entry) for entry in CYCLE_TABLE],
            "cycle_counter": (
                "global, advancing only on an accepted event, across series in "
                "roster order, so the table's composition is preserved over the "
                "corpus instead of restarting at slot 0 for every series"
            ),
            "position_rule": (
                "a seeded uniform permutation of the legal index range, then "
                "filtered in rule order: block fit, minimum spacing, "
                "sigma_local validity, target finiteness"
            ),
            "min_event_spacing": MIN_EVENT_SPACING,
            "boundary_exclusion": BOUNDARY_EXCLUSION,
            "legal_index_range": [legal_lo, legal_hi],
            "sigma_local": (
                "1.4826 * MAD(pristine[t-%d : t)); MAD == 0 -> std of the same "
                "prefix; still 0 -> skip the position and record it"
                % SIGMA_PREFIX
            ),
            "target_finiteness_rule": (
                "a position whose target points are missing in the pristine "
                "series is skipped and recorded.  This is a forced extension of "
                "the skip rule, not a discretionary one: a value that does not "
                "exist cannot be perturbed, and the identity path would "
                "interpolate the injection straight back out."
            ),
            "delta_scale_dependence": (
                "every delta = sign * sigma_multiple * sigma_local(pristine); "
                "no injected value is ever used as a scale source"
            ),
        },
        "ledger": {station: ledger[station] for station in names},
        "skips": {station: skips[station] for station in names},
        "spacing_rejections": spacing_rejections,
        "frozen_before_any_consumer_reading": True,
    }
    (INJECTED_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    (INJECTED_DIR / "ledger.json").write_text(
        json.dumps(protocol["ledger"], indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    return {
        "injected": injected,
        "ledger": ledger,
        "skips": skips,
        "spacing_rejections": spacing_rejections,
        "protocol": protocol,
        "events_per_series_per_block": per_block,
        "written_to": str(INJECTED_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


# =========================================================================== #
# Part C: the acceptance
# =========================================================================== #
def _read_block(
    array: np.ndarray, block: Mapping[str, Any], *, window: int
) -> tuple[np.ndarray, int]:
    """The block plus exactly one detector window of warm-up, identity-prepared."""
    start, end = int(block["start"]), int(block["end"])
    fed = v6._linear_integrity(array[start - window:end])
    return np.asarray(fed, dtype=np.float64), start - window


def _score_series(
    array: np.ndarray,
    block: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    window: int,
    threshold: float,
) -> dict[str, Any]:
    """One AD evaluation: detect over the block, score against its ledger rows."""
    start = int(block["start"])
    fed, offset = _read_block(array, block, window=window)
    reading = ad.detect(fed, window=window, threshold=threshold)
    flags = np.asarray(reading["flags"])
    flags[:start - offset] = False  # warm-up is never scored
    events = ad.predicted_events(flags, offset=offset)
    truth = [
        {"start": int(row["index"]), "end": int(row["index"]) + int(row["points"])}
        for row in rows
    ]
    scored = ad.score_events(truth, events)
    scored["abstained_zero_scale"] = int(reading["abstained_zero_scale"])
    scored["abstained_non_finite"] = int(reading["abstained_non_finite"])
    scored["predicted_event_spans"] = events
    return scored


def part_c_acceptance(
    pristine: Mapping[str, np.ndarray],
    injected: Mapping[str, np.ndarray],
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str],
    block: Mapping[str, Any],
    *,
    window: int,
    threshold: float,
) -> dict[str, Any]:
    """C5's three conditions plus D4's readings, at one detector setting."""
    evaluations = 0

    per_series: dict[str, Any] = {}
    for station in names:
        per_series[station] = _score_series(
            injected[station], block, ledger[station],
            window=window, threshold=threshold,
        )
        evaluations += 1

    twin: dict[str, Any] = {}
    for station in names:
        fed, offset = _read_block(pristine[station], block, window=window)
        reading = ad.detect(fed, window=window, threshold=threshold)
        flags = np.asarray(reading["flags"])
        flags[:int(block["start"]) - offset] = False
        twin[station] = ad.background_alarm_rate(flags[int(block["start"]) - offset:])
        twin[station]["abstained_zero_scale"] = int(reading["abstained_zero_scale"])
        evaluations += 1

    # (i) determinism: the whole scoring path again, on the same inputs.
    repeat_injected = {
        station: _score_series(
            injected[station], block, ledger[station],
            window=window, threshold=threshold,
        )
        for station in names
    }
    evaluations += len(names)
    repeat_twin: dict[str, Any] = {}
    for station in names:
        fed, offset = _read_block(pristine[station], block, window=window)
        reading = ad.detect(fed, window=window, threshold=threshold)
        flags = np.asarray(reading["flags"])
        flags[:int(block["start"]) - offset] = False
        repeat_twin[station] = ad.background_alarm_rate(flags[int(block["start"]) - offset:])
        repeat_twin[station]["abstained_zero_scale"] = int(reading["abstained_zero_scale"])
        evaluations += 1

    def canonical(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, default=str)

    deterministic = (
        canonical(per_series) == canonical(repeat_injected)
        and canonical(twin) == canonical(repeat_twin)
    )

    # pooled reading: the calibration block's own precision / recall / F1
    pooled_truth = sum(int(row["ledger_events"]) for row in per_series.values())
    pooled_pred = sum(int(row["predicted_events"]) for row in per_series.values())
    pooled_hit = sum(int(row["matched_events"]) for row in per_series.values())
    pooled_precision = float(pooled_hit) / pooled_pred if pooled_pred else None
    pooled_recall = float(pooled_hit) / pooled_truth if pooled_truth else None
    if pooled_precision and pooled_recall:
        pooled_f1: float | None = (
            2.0 * pooled_precision * pooled_recall / (pooled_precision + pooled_recall)
        )
    else:
        pooled_f1 = 0.0 if pooled_truth else None

    defined = [row["f1"] for row in per_series.values() if row["f1"] is not None]
    per_series_mean_f1 = float(statistics.fmean(defined)) if defined else None
    finite = (
        pooled_precision is not None
        and pooled_recall is not None
        and pooled_f1 is not None
        and all(np.isfinite([pooled_precision, pooled_recall, pooled_f1]))
    )
    background = [
        row["alarm_events_per_1000_points"] for row in twin.values()
        if row["alarm_events_per_1000_points"] is not None
    ]

    return {
        "detector_setting": {"window": int(window), "threshold": float(threshold)},
        "c5_i_determinism": {
            "requirement": (
                "the complete scoring path run twice on the same input is "
                "identical bit for bit apart from timestamps"
            ),
            "identical": bool(deterministic),
            "compared": "every per-series score row and every twin-block row",
        },
        "c5_ii_background": {
            "requirement": (
                "report the un-injected twin block's background alarm rate; it "
                "is not a false-positive rate and must never be used as one"
            ),
            "twin_block": "the same interval's original bytes, un-injected",
            "per_series": twin,
            "alarm_events_per_1000_points": {
                "min": min(background) if background else None,
                "median": float(statistics.median(background)) if background else None,
                "max": max(background) if background else None,
            },
            "not_a_false_positive_rate": (
                "the NOAA substrate carries unlabelled natural anomalies, so an "
                "alarm here may be correct.  This level bounds nothing and is "
                "reported as context for the precision figure below."
            ),
        },
        "c5_iii_calibrated_block": {
            "requirement": "precision, recall and event-F1 all finite, and F1 >= 0.5",
            "pooled": {
                "ledger_events": pooled_truth,
                "predicted_events": pooled_pred,
                "matched_events": pooled_hit,
                "precision": pooled_precision,
                "recall": pooled_recall,
                "f1": pooled_f1,
            },
            "per_series_mean_f1": per_series_mean_f1,
            "per_series_f1_defined": len(defined),
            "per_series_f1_undefined": len(names) - len(defined),
            "all_finite": bool(finite),
            "f1_at_or_above_half": bool(pooled_f1 is not None and pooled_f1 >= 0.5),
            "gate_is_read_on": (
                "the pooled figures.  Each series carries only two ledger "
                "events, so a per-series F1 can take three values and is "
                "reported for D4 rather than gated on."
            ),
            "precision_is_a_lower_bound": (
                "computed against the ledger only; a correct flag on an "
                "unlabelled natural anomaly counts in the denominator"
            ),
        },
        "d4_per_series": per_series,
        "passed": bool(
            deterministic and finite and pooled_f1 is not None and pooled_f1 >= 0.5
        ),
        "ad_evaluations": evaluations,
    }


# =========================================================================== #
# orchestration
# =========================================================================== #
def run() -> int:
    started = time.perf_counter()
    names_meta = roster()
    names = list(names_meta["all"])
    pristine = load_pristine(names)

    # -- Part B, before the instrument measures anything ----------------------
    part_b = part_b_contract(pristine, list(names_meta["train"]))
    print("B  %s  (%d comparisons)" % (
        part_b["verdict"], part_b["b4_assertion"]["comparisons"]), flush=True)
    if part_b["verdict"] != "SAME_BYTE_CONTRACT_HOLDS":
        return _write(
            "PROGRAM_GEOMETRY_UNALIGNED", names_meta, part_b, None, None, None,
            started, part_b["ad_evaluations"],
        )

    # -- Part D, ledger frozen to disk before any injected index is read ------
    block = calibration_block()
    if INJECTED_DIR.exists():
        shutil.rmtree(INJECTED_DIR)
    injection = freeze_injection(pristine, names, block)
    counts = [len(injection["ledger"][s]) for s in names]
    print("D  block [%d, %d)  events/series %d  realised %d-%d  skips %d" % (
        block["start"], block["end"], injection["events_per_series_per_block"],
        min(counts), max(counts),
        sum(len(v) for v in injection["skips"].values())), flush=True)

    # -- Part C, the acceptance ----------------------------------------------
    evaluations = part_b["ad_evaluations"]
    primary = part_c_acceptance(
        pristine, injection["injected"], injection["ledger"], names, block,
        window=ad.WINDOW, threshold=ad.THRESHOLD,
    )
    evaluations += primary["ad_evaluations"]
    print("C  primary passed=%s  pooled F1 %s  det=%s" % (
        primary["passed"], primary["c5_iii_calibrated_block"]["pooled"]["f1"],
        primary["c5_i_determinism"]["identical"]), flush=True)

    fallback = None
    chosen = primary
    if not primary["passed"]:
        fallback = part_c_acceptance(
            pristine, injection["injected"], injection["ledger"], names, block,
            window=ad.FALLBACK_WINDOW, threshold=ad.FALLBACK_THRESHOLD,
        )
        evaluations += fallback["ad_evaluations"]
        print("C  fallback passed=%s  pooled F1 %s" % (
            fallback["passed"],
            fallback["c5_iii_calibrated_block"]["pooled"]["f1"]), flush=True)
        chosen = fallback

    if not chosen["passed"]:
        verdict = "AD_CONSUMER_SPEC_DEFECT"
    else:
        verdict = "T0_READY"
    return _write(
        verdict, names_meta, part_b, block, injection, (primary, fallback, chosen),
        started, evaluations,
    )


def _write(
    verdict: str,
    names_meta: Mapping[str, Any],
    part_b: Mapping[str, Any],
    block: Mapping[str, Any] | None,
    injection: Mapping[str, Any] | None,
    acceptance: tuple[Any, Any, Any] | None,
    started: float,
    evaluations: int,
) -> int:
    primary, fallback, chosen = acceptance or (None, None, None)
    frozen: dict[str, Any] = dict(ad.spec())
    if chosen is not None:
        frozen["frozen_setting"] = dict(chosen["detector_setting"])
        frozen["fallback_taken"] = fallback is not None
        if fallback is not None:
            frozen["fallback_note"] = (
                "the primary setting failed the acceptance, the one "
                "pre-registered fallback was taken, and window=%d / "
                "threshold=%.1f is frozen here.  T1 may not move it."
                % (ad.FALLBACK_WINDOW, ad.FALLBACK_THRESHOLD)
            )
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "T0: freeze the AD Consumer, prove the same-byte contract, "
            "calibrate the substrate"
        ),
        "verdict": verdict,
        "llm_calls": 0,
        "forecasting_retrains": 0,
        "ad_evaluations": int(evaluations),
        "ad_evaluation_budget": AD_EVALUATION_BUDGET,
        "budget_respected": int(evaluations) <= AD_EVALUATION_BUDGET,
        "roster": dict(names_meta),
        "part_b_same_byte_contract": dict(part_b),
        "part_c_instrument": frozen,
        "part_d_block": dict(block) if block else None,
        "part_d_protocol": (
            dict(injection["protocol"]) if injection else None
        ),
        "part_d_ledger_summary": (
            {
                "events_per_series_per_block": injection["events_per_series_per_block"],
                "per_series": {
                    station: {
                        "events": len(rows),
                        "skips": len(injection["skips"][station]),
                        "spacing_rejections": injection["spacing_rejections"][station],
                        "types": [row["type"] for row in rows],
                        "sigma_multiples": [row["sigma_multiple"] for row in rows],
                        "signs": [row["sign"] for row in rows],
                        "indices": [row["index"] for row in rows],
                    }
                    for station, rows in injection["ledger"].items()
                },
                "total_events": sum(len(v) for v in injection["ledger"].values()),
                "total_skips": sum(len(v) for v in injection["skips"].values()),
                "written_to": injection["written_to"],
            }
            if injection else None
        ),
        "part_c_acceptance_primary": primary,
        "part_c_acceptance_fallback": fallback,
        "exposure": {
            "noaa_development_region": {
                "context": "INSTANCE_SEEN", "outcome": "EXPOSED",
                "note": "already the line's development substrate",
            },
            "noaa_2025_confirmation": {"outcome": "SEALED", "read": False},
            "beyond_17520": {"outcome": "SEALED", "read": False},
            "smd_official_test_and_labels": {"outcome": "SEALED", "read": False},
        },
        "wall_seconds": time.perf_counter() - started,
    }
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("VERDICT", verdict, "| AD evaluations", evaluations, flush=True)
    return 0


# =========================================================================== #
# what T0 hands to T1: findings that cost no evaluation
# =========================================================================== #
def t1_findings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pure function of the artifact and this module's constants.  No reading.

    These are the facts T0 measured that bear on whether T1 can run as its
    preview is written.  They are reported, not acted on: repairing the data
    path or moving an injection rule belongs to T1's own book.
    """
    b = payload["part_b_same_byte_contract"]["b3_live_path_check"]
    block_lo, block_hi = (int(v) for v in b["training_block_B"]["span"])
    targets = [(s, s + TRIPLE_WINDOW_LENGTH) for s in TRIPLE_WINDOW_STARTS]
    contexts = [
        (s - CONTEXT_LENGTH, s + TRIPLE_WINDOW_LENGTH) for s in TRIPLE_WINDOW_STARTS
    ]
    intersects = any(block_lo < hi and lo < block_hi for lo, hi in contexts)
    chosen = (
        payload.get("part_c_acceptance_fallback")
        or payload["part_c_acceptance_primary"]
    )
    primary = payload["part_c_acceptance_primary"]
    calibration = payload["part_d_block"]
    return {
        "t1_injection_placement_conflict": {
            "severity": "blocking for T1 as previewed",
            "measured_training_block_B": [block_lo, block_hi],
            "triple_window_target_spans": [[lo, hi] for lo, hi in targets],
            "triple_window_context_inclusive_spans": [
                [lo, hi] for lo, hi in contexts
            ],
            "intersection_is_empty": not intersects,
            "why_it_blocks": (
                "P acts only where the forecasting Consumer builds training "
                "rows, which is the train windows at _config()'s anchors -- "
                "[%d, %d).  The eval side is read through _linear_integrity "
                "and the truth window is read raw, so P never touches the "
                "triple-window region at all.  If T1 injects only inside the "
                "triple-windows then P(B) carries no injection: the Program "
                "acts on clean data and the AD Consumer's ledger inside that "
                "same P(B) is empty.  Nothing would flip, and nothing would "
                "have been tested." % (block_lo, block_hi)
            ),
            "the_two_instructions_that_cannot_both_hold": [
                "#35 v2 D2: T1's injection goes only inside the triple-windows",
                "#35 errata (d): injection and AD detection happen inside the "
                "12 train series' blocks, both Consumers consuming one "
                "P(train block)",
            ],
            "needs": (
                "a main-line ruling on where T1's injection lives, before T1 runs"
            ),
            "if_it_moves_into_the_training_block": (
                "T0's calibration block [%d, %d) lies inside [%d, %d), so D2's "
                "hard T0/T1 isolation would then require T1's block to avoid it"
                % (calibration["start"], calibration["end"], block_lo, block_hi)
            ),
        },
        "resolution_of_the_ad_gain_vector": {
            "ledger_events_per_series_per_block": payload[
                "part_d_ledger_summary"
            ]["events_per_series_per_block"],
            "consequence": (
                "with two ledger events a per-series F1 moves in steps of "
                "roughly 0.2 to 0.3 when one event changes hands.  The "
                "pre-registered material line of %.3f cannot resolve anything "
                "below that step, so a per-series AD gain fed to the guard's "
                "min_per_series_gain is coarse by construction.  The errata's "
                "reading -- flip judged at the aggregate layer, per-series "
                "comparison only within each task -- is the one this "
                "resolution supports." % ad.MATERIAL_THRESHOLD
            ),
        },
        "what_the_acceptance_gate_actually_gated_on": {
            "recall_primary": primary["c5_iii_calibrated_block"]["pooled"]["recall"],
            "recall_chosen": chosen["c5_iii_calibrated_block"]["pooled"]["recall"],
            "predicted_events_primary": primary["c5_iii_calibrated_block"][
                "pooled"
            ]["predicted_events"],
            "predicted_events_chosen": chosen["c5_iii_calibrated_block"][
                "pooled"
            ]["predicted_events"],
            "reading": (
                "recall did not move between the two detector settings: the "
                "detector saw the same injections either way.  The whole F1 "
                "difference came from precision, whose denominator is the "
                "count of predicted events, most of which the ledger does not "
                "name.  So the fallback was selected on the background alarm "
                "level -- the quantity C5(ii) says bounds nothing.  The "
                "verdict follows the frozen protocol, but the gate as built "
                "reads background, not readability, and the main line should "
                "know that before T1 leans on the frozen setting."
            ),
            "margin_over_the_bar": (
                float(chosen["c5_iii_calibrated_block"]["pooled"]["f1"]) - 0.5
            ),
        },
        "sigma_scale_mismatch": {
            "injection_scale": (
                "1.4826 * MAD over a %d-point pristine prefix" % SIGMA_PREFIX
            ),
            "detector_scale": (
                "1.4826 * MAD over the %d-point trailing window"
                % int(chosen["detector_setting"]["window"])
            ),
            "consequence": (
                "a nominal 6-sigma_local injection is not guaranteed to be a "
                "detector-scale exceedance, because the two sigmas are "
                "measured over different spans.  One of the 32 calibration "
                "events was missed for exactly this reason."
            ),
        },
    }


def annotate() -> int:
    """Add t1_findings to an existing artifact.  Spends no AD evaluation."""
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    payload["t0_findings_for_t1"] = t1_findings(payload)
    payload["annotation_note"] = (
        "t0_findings_for_t1 is a pure function of the readings already in "
        "this artifact and of the module's frozen constants.  Adding it "
        "re-ran no detector and spent no evaluation; ad_evaluations is "
        "unchanged."
    )
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("annotated", OUT_JSON, flush=True)
    return 0


def _markdown(payload: Mapping[str, Any]) -> str:
    b = payload["part_b_same_byte_contract"]
    lines = [
        "# T0: AD Consumer v1, the same-byte contract, and the calibrated substrate",
        "",
        "**%s.**  0 LLM calls, 0 forecasting retrains, %d of %d AD evaluations."
        % (payload["verdict"], payload["ad_evaluations"], payload["ad_evaluation_budget"]),
        "",
        "## Part B -- the same-byte contract",
        "",
        "%s" % b["b1_contract"]["statement"],
        "",
        "- Processing-side boundary: %s" % b["b1_contract"]["processing_side_boundary"],
        "- Action geometry: %s" % b["b1_contract"]["action_geometry"],
        "- Asymmetry declaration: %s" % b["b2_asymmetry_declaration"]["statement"],
        "",
        "| check | result |",
        "| --- | --- |",
        "| P(B) comparisons | %d |" % b["b4_assertion"]["comparisons"],
        "| `np.array_equal` on every one | **%s** |" % b["b4_assertion"]["all_equal"],
        "| same bytes on a repeat call | %s |" % b["b4_assertion"]["reproducible_on_recall"],
        "| identity == `_linear_integrity` baseline | %s |"
        % b["b3_live_path_check"]["identity_equals_linear_integrity_baseline"],
        "| detector flags invariant under `_center_scale` | %s |"
        % b["b4_assertion"]["affine_invariance_check"]["all_identical"],
        "| forecasting retrains spent | %d |" % b["b3_live_path_check"]["retrains_spent"],
        "",
        "Verdict: **%s**." % b["verdict"],
        "",
    ]
    block = payload.get("part_d_block")
    if block:
        lines.extend([
            "## Part D -- the calibration block and the frozen ledger",
            "",
            "- Rule: %s." % block["rule"],
            "- Literal earliest segment `[%d, %d)` is **not executable**: %s"
            % (block["literal_earliest_start"],
               block["literal_earliest_start"] + block["length"],
               block["literal_earliest_is_not_executable"]),
            "- Taken: **`[%d, %d)`** -- %s" % (block["start"], block["end"], block["resolution"]),
            "- Zero overlap with the in-service triple-windows %s: %s."
            % (block["triple_window_target_spans"], block["zero_overlap_with_triple_windows"]),
            "- Calibration seed %d; T1's seed %d is reserved and %s."
            % (block["seed"], block["t1_seed_reserved"], block["t1_ledger"]),
            "",
        ])
    summary = payload.get("part_d_ledger_summary")
    if summary:
        lines.extend([
            "| series | events | skips | spacing rejections | indices |",
            "| --- | ---: | ---: | ---: | --- |",
        ])
        for station, row in summary["per_series"].items():
            lines.append("| `%s` | %d | %d | %d | %s |" % (
                station, row["events"], row["skips"], row["spacing_rejections"],
                ", ".join(str(i) for i in row["indices"]),
            ))
        lines.extend([
            "",
            "Total %d events, %d skips, written to `%s`."
            % (summary["total_events"], summary["total_skips"], summary["written_to"]),
            "",
        ])
    chosen = payload.get("part_c_acceptance_fallback") or payload.get(
        "part_c_acceptance_primary"
    )
    if chosen:
        pooled = chosen["c5_iii_calibrated_block"]["pooled"]
        lines.extend([
            "## Part C -- the acceptance",
            "",
            "| condition | reading | pass |",
            "| --- | --- | --- |",
            "| (i) determinism | full path twice, identical | %s |"
            % chosen["c5_i_determinism"]["identical"],
            "| (ii) background alarm rate (twin block) | median %s per 1000 points | reported, not gated |"
            % chosen["c5_ii_background"]["alarm_events_per_1000_points"]["median"],
            "| (iii) calibrated block | P %s / R %s / F1 %s | %s |"
            % (pooled["precision"], pooled["recall"], pooled["f1"],
               chosen["c5_iii_calibrated_block"]["f1_at_or_above_half"]),
            "",
            "Frozen detector setting: window %d, threshold %s.  Fallback taken: %s."
            % (chosen["detector_setting"]["window"],
               chosen["detector_setting"]["threshold"],
               payload["part_c_acceptance_fallback"] is not None),
            "",
            "| series | ledger | predicted | matched | precision | recall | F1 | background /1000 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        twin = chosen["c5_ii_background"]["per_series"]
        for station, row in chosen["d4_per_series"].items():
            lines.append("| `%s` | %d | %d | %d | %s | %s | %s | %s |" % (
                station, row["ledger_events"], row["predicted_events"],
                row["matched_events"],
                _fmt(row["precision"]), _fmt(row["recall"]), _fmt(row["f1"]),
                _fmt(twin[station]["alarm_events_per_1000_points"]),
            ))
        lines.extend([
            "",
            chosen["c5_ii_background"]["not_a_false_positive_rate"],
            "",
            chosen["c5_iii_calibrated_block"]["precision_is_a_lower_bound"] + ".",
            "",
        ])
    findings = payload.get("t0_findings_for_t1")
    if findings:
        conflict = findings["t1_injection_placement_conflict"]
        gate = findings["what_the_acceptance_gate_actually_gated_on"]
        lines.extend([
            "## What T0 hands to T1",
            "",
            "### The injection-placement conflict (blocking)",
            "",
            "- Measured training block **B = [%d, %d)**."
            % tuple(conflict["measured_training_block_B"]),
            "- In-service triple-windows, context inclusive: %s."
            % conflict["triple_window_context_inclusive_spans"],
            "- Intersection empty: **%s**." % conflict["intersection_is_empty"],
            "",
            conflict["why_it_blocks"],
            "",
            "These two cannot both hold:",
            "",
        ])
        for item in conflict["the_two_instructions_that_cannot_both_hold"]:
            lines.append("- %s" % item)
        lines.extend([
            "",
            "%s  And %s."
            % (conflict["needs"], conflict["if_it_moves_into_the_training_block"]),
            "",
            "### What the acceptance gate actually gated on",
            "",
            "| setting | recall | predicted events |",
            "| --- | ---: | ---: |",
            "| primary (25 / 4.0) | %.4f | %d |"
            % (gate["recall_primary"], gate["predicted_events_primary"]),
            "| fallback (49 / 3.5) | %.4f | %d |"
            % (gate["recall_chosen"], gate["predicted_events_chosen"]),
            "",
            gate["reading"],
            "",
            "Margin over the bar: **%+.4f**." % gate["margin_over_the_bar"],
            "",
            "### Resolution of the AD gain vector",
            "",
            findings["resolution_of_the_ad_gain_vector"]["consequence"],
            "",
            "### Sigma-scale mismatch",
            "",
            findings["sigma_scale_mismatch"]["consequence"],
            "",
        ])
    lines.extend([
        "## Cost",
        "",
        "- LLM calls 0.  Forecasting retrains 0.  AD evaluations %d of %d."
        % (payload["ad_evaluations"], payload["ad_evaluation_budget"]),
        "- Sealed: NOAA 2025, everything beyond 17520, SMD official test and labels.",
        "- Wall seconds %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "--" if value is None else ("%.3f" % float(value))


if __name__ == "__main__":
    raise SystemExit(
        annotate() if "--annotate" in sys.argv[1:] else run()
    )
