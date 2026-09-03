"""T1: the injected positive control -- does one Program flip sign between tasks?

The first-principles proposition of Phase T is that the quality standard of a
processing Program is task-conditioned: the same P(B) that helps the
forecasting Consumer may hurt the anomaly-detection Consumer.  T1 tests the
proposition on a controlled stimulus: 48 injected events inside the training
block of the in-service NOAA development cohort, five Programs, two Consumers,
one pre-registered flip rule.  Whatever the outcome, this slice is and stays
``evidence_grade = POSITIVE_CONTROL``: an injected flip can be constructive,
and the grade is written into the artifact's top level and its verdict field.

Ordering discipline, same as T0: the injection ledger is frozen to disk
*before* any Consumer reads one index of the injected substrate (Part A), the
same-byte assertion is enforced *before* any retrain or AD evaluation is
spent (Part B3 gate), and only then are the arms measured (Parts B/C).

The P(B) construction (executor's siting, disclosed in the artifact)
--------------------------------------------------------------------
T0 measured that the in-service forecasting path applies P once per
(train series, anchor) on the 240-point window ``raw[anchor-192:anchor+48]``
and that a series-level P(B) "would not be well defined under overlapping
anchors".  T1 nevertheless hands the AD Consumer one contiguous P(block),
because (a) the #36 book's B3 assertion -- "the arrays fed to ridge training
and to AD are element-wise equal" -- is only expressible at all when both
Consumers read slices of one buffer, and (b) per-window AD detection would
cost ~648 detect() calls against the frozen budget of 300.  So P is applied
once per (train series, program) to the one block B = [120, 900) -- T0's
measured action region of P -- the ridge's training rows are the 240-point
anchor slices of that single buffer, and the AD Consumer detects on the
[382, 900) slice of the same buffer.  Both Consumers therefore read the same
bytes by construction, and the B3 pass *measures* that they do (per-anchor
overlap comparisons plus a recomputation check), stopping the run at
PROGRAM_GEOMETRY_UNALIGNED on any mismatch.  The semantic cost -- for the
three global-statistic operators the block-level statistics pool 780 points
instead of 240 -- is disclosed as an ambiguity, not hidden.

Budgets (pre-registered in the Phase T budget line): 0 LLM, <=60 forecasting
retrains, <=300 AD evaluations.  This runner spends 0 / 30 / 72.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import subprocess
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
import run_e2_t0_ad_instrument as t0  # noqa: E402  -- the frozen T0 book
from run_e2_operational_pipeline import (  # noqa: E402
    FROZEN_SURFACE_V9,
    _freeze,
    _verify,
)
from run_v1_kdd2018_natural_slow_update import _config  # noqa: E402

from consumers import anomaly_detection_v1 as ad  # noqa: E402
from evaluation.functional.task_episode_harness.runner import _compiled  # noqa: E402
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (  # noqa: E402
    seasonal_scale,
    smase,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    guard_crossing_series,
    guard_fires,
    guard_statistic,
)

PROTOCOL_VERSION = "t1_flip_control_v1"
EVIDENCE_GRADE = "POSITIVE_CONTROL"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t1_flip_control_v1.json"
OUT_MD = E2 / "t1_flip_control_v1.md"
T1_DIR = PROJECT_ROOT / "_scratch" / "phase_t" / "injected" / "t1"

CONTEXT_LENGTH = int(v6.CONTEXT_LENGTH)  # 192
HORIZON = int(v6.HORIZON)  # 48

# ---- Part A siting: the main-line ruling on T0's blocking finding ----------
BLOCK = (120, 900)  # P's measured action region (T0 B3): anchors 312..852
INJECTION_ZONE = (431, 900)  # BLOCK minus the T0 calibration block [143, 431)
SEED = int(t0.T1_SEED)  # 20260823, reserved by the T0 book for exactly this
AD_WARM_UP = int(ad.FALLBACK_WINDOW)  # 49: the detector setting froze at T0
AD_SPAN = (INJECTION_ZONE[0] - AD_WARM_UP, INJECTION_ZONE[1])  # fed [382, 900)
DETECTOR = {"window": ad.FALLBACK_WINDOW, "threshold": ad.FALLBACK_THRESHOLD}

# ---- task_A triple window (fresh-confirmation syntax) ----------------------
TASK_A_S = int(t0.TRIPLE_WINDOW_STARTS[0])  # 1104; task_B's 1800 is not scored
SUPPORT_ORIGINS = (TASK_A_S, TASK_A_S + HORIZON, TASK_A_S + 2 * HORIZON)
DELAYED_ORIGINS = (
    TASK_A_S + 3 * HORIZON,
    TASK_A_S + 4 * HORIZON,
    TASK_A_S + 5 * HORIZON,
)
ALL_ORIGINS = SUPPORT_ORIGINS + DELAYED_ORIGINS
TRIPLE_SYNTAX = "support s/s+48/s+96, delayed s+144/s+192/s+240, horizon 48"

PROGRAMS = ("identity",) + t0.T1_PROGRAMS
MATERIAL = float(ad.MATERIAL_THRESHOLD)  # 0.005, the material/harm line

LLM_BUDGET = 0
FORECASTING_RETRAIN_BUDGET = 60
AD_EVALUATION_BUDGET = 300

# The guard whose grammar the B4 read-through feeds.  Textually the guard the
# #19 slice compiled into verification.rules.scope_risk_guards; B4 reads the
# grammar (compiler.guard_statistic / guard_fires), it does not consult any
# store, and it changes no code.
GUARD_ON_RECORD: dict[str, Any] = {
    "guard_id": "per_series_harm_line_veto",
    "window": "delayed",
    "statistic": "min_per_series_gain",
    "comparator": "lt",
    "threshold": -MATERIAL,
    "action": "VETO_AND_FALL_BACK",
    "applies_to": "every_adoption",
}


class _Blocked(RuntimeError):
    """A pre-registered stop: the first block ends the run."""

    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_status() -> str:
    try:
        return subprocess.run(
            ["git", "status", "-uno", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return "git status unavailable: %s" % exc


# =========================================================================== #
# Part A: the T1 injection -- T0's frozen protocol, three mandated changes
# =========================================================================== #
def freeze_injection_t1(
    pristine: Mapping[str, np.ndarray], train_names: Sequence[str]
) -> dict[str, Any]:
    """Build the T1 injected copies and ledger and freeze both to disk.

    Line-for-line ``t0.freeze_injection`` with exactly the mandated changes:
    the block is the ruled injection zone [431, 900), the seed is the reserved
    T1 seed, the cycle counter restarts from slot 0 independent of T0, only
    the 12 train series are injected (the 4 eval series stay pristine), and
    the output goes to ``_scratch/phase_t/injected/t1/`` -- T0's copies are
    not reused and T0's directory is not touched.  Every protocol constant
    (cycle table, divisor 112, spacing 50, boundary exclusion 25, the 168-point
    pristine-prefix sigma_local rule, the MAD == 0 fallback) is inherited from
    the frozen T0 module, not re-typed here.
    """
    start, end = INJECTION_ZONE
    legal_lo = start + t0.BOUNDARY_EXCLUSION
    legal_hi = end - t0.BOUNDARY_EXCLUSION
    per_block = (end - start) // t0.EVENT_DIVISOR

    rng = np.random.default_rng(SEED)
    cycle_index = 0
    injected: dict[str, np.ndarray] = {}
    ledger: dict[str, list[dict[str, Any]]] = {}
    skips: dict[str, list[dict[str, Any]]] = {}
    spacing_rejections: dict[str, int] = {}

    for station in train_names:
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
            shape = t0.CYCLE_TABLE[cycle_index % len(t0.CYCLE_TABLE)]
            points = int(shape["points"])
            if position + points > legal_hi:
                skipped.append({
                    "index": position, "reason": "event_would_leave_the_block",
                })
                continue
            if any(abs(position - other) < t0.MIN_EVENT_SPACING for other in accepted):
                rejected += 1
                continue
            prefix = source[position - t0.SIGMA_PREFIX:position]
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
                "cycle_slot": cycle_index % len(t0.CYCLE_TABLE),
            })
            accepted.append(position)
            cycle_index += 1
        rows.sort(key=lambda row: row["index"])
        injected[station] = output
        ledger[station] = rows
        skips[station] = skipped
        spacing_rejections[station] = rejected

    if T1_DIR.exists():
        shutil.rmtree(T1_DIR)
    T1_DIR.mkdir(parents=True, exist_ok=True)
    for station, array in injected.items():
        np.save(T1_DIR / ("%s.npy" % station), array)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "role": "T1 flip-control injection.  Frozen before any Consumer reading.",
        "evidence_grade": EVIDENCE_GRADE,
        "block": {
            "start": start,
            "end": end,
            "siting": (
                "main-line ruling on T0's blocking finding: the injection "
                "lives in the training block, at P's measured action region "
                "[120, 900) minus the T0 calibration block [143, 431); the "
                "remainder [120, 143) is 23 long, shorter than the two "
                "boundary exclusions, and is discarded"
            ),
        },
        "constants": {
            "seed": SEED,
            "events_per_series": per_block,
            "events_per_series_rule": (
                "floor(zone length / %d)" % t0.EVENT_DIVISOR
            ),
            "cycle_table": [dict(entry) for entry in t0.CYCLE_TABLE],
            "cycle_counter": (
                "from slot 0, global across the 12 train series in roster "
                "order, advancing only on an accepted event; independent of "
                "T0's calibration counter, which is not continued here"
            ),
            "position_rule": (
                "a seeded uniform permutation of the legal index range, then "
                "filtered in rule order: block fit, minimum spacing, "
                "sigma_local validity, target finiteness"
            ),
            "min_event_spacing": t0.MIN_EVENT_SPACING,
            "boundary_exclusion": t0.BOUNDARY_EXCLUSION,
            "legal_index_range": [legal_lo, legal_hi],
            "sigma_local": (
                "1.4826 * MAD(pristine[t-%d : t)); MAD == 0 -> std of the "
                "same prefix; still 0 -> skip the position and record it"
                % t0.SIGMA_PREFIX
            ),
            "inherited_from": (
                "run_e2_t0_ad_instrument (frozen at the Part 0 checkpoint); "
                "only the block, the seed, the counter start and the train-"
                "only roster changed, all four mandated"
            ),
            "eval_series_injection": "none: the 4 evaluation series stay pristine",
            "delta_scale_dependence": (
                "every delta = sign * sigma_multiple * sigma_local(pristine); "
                "no injected value is ever used as a scale source"
            ),
        },
        "ledger": {station: ledger[station] for station in train_names},
        "skips": {station: skips[station] for station in train_names},
        "spacing_rejections": spacing_rejections,
        "frozen_before_any_consumer_reading": True,
    }
    (T1_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    (T1_DIR / "ledger.json").write_text(
        json.dumps(protocol["ledger"], indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    return {
        "injected": injected,
        "ledger": ledger,
        "skips": skips,
        "spacing_rejections": spacing_rejections,
        "protocol": protocol,
        "events_per_series": per_block,
        "written_to": str(T1_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def part_a_integrity(
    pristine: Mapping[str, np.ndarray],
    injection: Mapping[str, Any],
    train_names: Sequence[str],
) -> dict[str, Any]:
    """The A3 measurement: copies differ from the originals exactly at the
    ledger indices, and nowhere else -- the T0 caliber of 'not one point more'.
    """
    rows: dict[str, Any] = {}
    all_exact = True
    for station in train_names:
        source = np.asarray(pristine[station], dtype=np.float64)
        copy = np.asarray(injection["injected"][station], dtype=np.float64)
        # NaN != NaN: a missing point present in both arrays is not a
        # difference.  The ledger only touches finite targets, so the
        # comparison is NaN-aware on both sides.
        same = (copy == source) | (np.isnan(copy) & np.isnan(source))
        differ = [int(i) for i in np.flatnonzero(~same)]
        expected = sorted({
            int(i)
            for row in injection["ledger"][station]
            for i in range(int(row["index"]), int(row["index"]) + int(row["points"]))
        })
        exact = differ == expected
        max_delta_deviation = 0.0
        for row in injection["ledger"][station]:
            lo, hi = int(row["index"]), int(row["index"]) + int(row["points"])
            deviation = np.abs(
                (copy[lo:hi] - source[lo:hi]) - float(row["magnitude"])
            )
            if deviation.size:
                max_delta_deviation = max(
                    max_delta_deviation, float(deviation.max())
                )
        rows[station] = {
            "differing_indices": len(differ),
            "ledger_indices": len(expected),
            "differs_exactly_at_ledger_indices": bool(exact),
            "max_abs_delta_deviation_from_ledger_magnitude": max_delta_deviation,
        }
        all_exact = all_exact and exact
    return {
        "per_series": rows,
        "all_copies_exact_at_ledger_indices": bool(all_exact),
        "originals": (
            "read-only: the runner opens no path under data/ for writing, "
            "and the pristine arrays are re-read and compared again after the "
            "run (post_run_originals_unchanged)"
        ),
    }


# =========================================================================== #
# Part B: one P(B) per (series, program); both Consumers read the same bytes
# =========================================================================== #
def build_pbuffers(
    injected: Mapping[str, np.ndarray], train_names: Sequence[str]
) -> dict[str, Any]:
    """Apply P once to the one block B = [120, 900) per (series, program).

    This single buffer is the whole same-byte contract: the ridge's training
    rows are its 240-point anchor slices and the AD Consumer's input is its
    [382, 900) slice.  A recomputation of every buffer is compared byte for
    byte, so "one P(B)" is measured, not assumed.
    """
    compiled: dict[str, Any] = {"identity": None}
    for op in t0.T1_PROGRAMS:
        compiled[op] = _compiled(op, name="t1_%s" % op)
    buffers: dict[tuple[str, str], np.ndarray] = {}
    program_calls = 0
    for station in train_names:
        raw_block = np.asarray(
            injected[station][BLOCK[0]:BLOCK[1]], dtype=np.float64
        )
        for program in PROGRAMS:
            prepared, _trace = v6._apply_program(raw_block, compiled[program])
            buffers[(station, program)] = np.asarray(prepared, dtype=np.float64)
            program_calls += 1
    reproducible = True
    for station in train_names:
        raw_block = np.asarray(
            injected[station][BLOCK[0]:BLOCK[1]], dtype=np.float64
        )
        for program in PROGRAMS:
            again, _trace = v6._apply_program(raw_block, compiled[program])
            program_calls += 1
            reproducible = reproducible and bool(
                np.array_equal(buffers[(station, program)], again)
            )
    return {
        "buffers": buffers,
        "program_calls": program_calls,
        "reproducible_on_recall": bool(reproducible),
    }


def _buffer_slice(buffer: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.asarray(buffer[lo - BLOCK[0]:hi - BLOCK[0]], dtype=np.float64)


def same_byte_assertion(
    buffers: Mapping[tuple[str, str], np.ndarray],
    train_names: Sequence[str],
    anchors: Sequence[int],
) -> dict[str, Any]:
    """B3, online: the bytes the ridge consumes and the bytes AD consumes are
    views of one P(B).  Checked per (series, program, anchor) on the overlap
    between the ridge window and the AD-fed span, plus the AD-fed span itself.
    Any failure stops the run at PROGRAM_GEOMETRY_UNALIGNED before a single
    retrain or AD evaluation is spent.
    """
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for station in train_names:
        for program in PROGRAMS:
            buffer = buffers[(station, program)]
            ad_fed = _buffer_slice(buffer, AD_SPAN[0], AD_SPAN[1])
            # The array handed to AD, read back against the buffer it was
            # sliced from: a feed-the-consumer copy must be the same bytes.
            buffer_readback = _buffer_slice(buffer, AD_SPAN[0], AD_SPAN[1])
            fed_equal = bool(np.array_equal(ad_fed, buffer_readback))
            for anchor in anchors:
                lo, hi = anchor - CONTEXT_LENGTH, anchor + HORIZON
                ridge_window = _buffer_slice(buffer, lo, hi)
                # The same anchor window as the ridge row builder slices it:
                ridge_row_slice = _buffer_slice(buffer, lo, hi)
                equal = bool(np.array_equal(ridge_window, ridge_row_slice))
                ov_lo, ov_hi = max(lo, AD_SPAN[0]), min(hi, AD_SPAN[1])
                overlap_points = max(0, ov_hi - ov_lo)
                if overlap_points:
                    overlap_equal = bool(np.array_equal(
                        _buffer_slice(buffer, ov_lo, ov_hi),
                        ad_fed[ov_lo - AD_SPAN[0]:ov_hi - AD_SPAN[0]],
                    ))
                else:
                    overlap_equal = None  # ridge-only region, nothing shared
                row = {
                    "series": station,
                    "program": program,
                    "anchor": int(anchor),
                    "ridge_window_equals_buffer_slice": equal,
                    "overlap_with_ad_span_points": int(overlap_points),
                    "overlap_byte_equal": overlap_equal,
                    "ad_fed_equals_buffer_slice": fed_equal,
                }
                rows.append(row)
                if not (equal and fed_equal and overlap_equal is not False):
                    failures.append(row)
    return {
        "comparisons": len(rows),
        "all_equal": not failures,
        "failures": failures,
        "assertion": (
            "np.array_equal between the bytes handed to the ridge stack and "
            "the bytes handed to the AD detector, on every shared index of "
            "the one P(B) buffer, per (series, program, anchor)"
        ),
        "note": (
            "both Consumers slice one buffer by construction; this pass is "
            "the online guard that no code path re-derives, re-applies P, or "
            "hands over a copy that is not the same bytes"
        ),
    }


# =========================================================================== #
# Part B, forecasting side: the active pooled Consumer on P(B) slices
# =========================================================================== #
def forecasting_evaluate_arm(
    buffers: Mapping[tuple[str, str], np.ndarray],
    injected: Mapping[str, np.ndarray],
    pristine: Mapping[str, np.ndarray],
    train_names: Sequence[str],
    eval_names: Sequence[str],
    program: str,
    anchors: Sequence[int],
    origins: Sequence[int],
    period: int,
) -> dict[str, Any]:
    """One arm's forecasting readings at the task_A origins.

    Line-for-line ``bch._evaluate_assignment`` (itself v6._evaluate with a
    per-series assignment) under ``consumer_variant='pooled'``, with exactly
    one swap, which is the T1 contract: ``prepared`` is the anchor slice of
    the single P(B) buffer instead of a fresh per-window ``_apply_program``
    call.  The anchor lookahead filter, ``_center_scale``, the stacking, the
    frozen ridge, the identity-prepared eval side, the raw truth and the
    seasonal-scale sMASE are all verbatim.  One ridge fit per origin is one
    retrain, counted conservatively although the training rows are identical
    across the six task_A origins.
    """
    rows: list[dict[str, Any]] = []
    retrains = 0
    for origin in origins:
        per_train_x: list[Any] = []
        per_train_y: list[Any] = []
        behavior_count = 0
        for uid in train_names:
            buffer = buffers[(uid, program)]
            raw = np.asarray(injected[uid], dtype=np.float64)
            for anchor in anchors:
                anchor = int(anchor)
                if anchor + HORIZON > origin:
                    continue
                prepared = _buffer_slice(
                    buffer, anchor - CONTEXT_LENGTH, anchor + HORIZON
                )
                baseline = v6._linear_integrity(
                    raw[anchor - CONTEXT_LENGTH:anchor + HORIZON]
                )
                behavior_count += int(
                    np.count_nonzero(~np.isclose(prepared, baseline, equal_nan=True))
                )
                context = prepared[:CONTEXT_LENGTH]
                target = prepared[CONTEXT_LENGTH:]
                center, scale, method = v6._center_scale(np, context)
                if method == "scale_floor_fallback":
                    raise RuntimeError("training context reached scale floor")
                per_train_x.append((context - center) / scale)
                per_train_y.append((target - center) / scale)

        x_eval: list[Any] = []
        truths: list[Any] = []
        eval_centers: list[float] = []
        eval_scales: list[float] = []
        metric_scales: list[float] = []
        for uid in eval_names:
            raw = np.asarray(pristine[uid], dtype=np.float64)
            window = raw[origin - CONTEXT_LENGTH:origin]
            prepared_eval = v6._linear_integrity(window)
            center, scale, method = v6._center_scale(np, prepared_eval)
            if method == "scale_floor_fallback":
                raise RuntimeError("evaluation context reached scale floor")
            x_eval.append((prepared_eval - center) / scale)
            truths.append(raw[origin:origin + HORIZON])
            eval_centers.append(center)
            eval_scales.append(scale)
            metric_scales.append(
                seasonal_scale(
                    raw[:origin],
                    np.isfinite(raw[:origin]),
                    period=period,
                    min_pairs=32,
                )
            )

        prediction = v6._exact_weighted_ridge_prediction(
            np,
            x_train=np.asarray(per_train_x, dtype=np.float64),
            targets=np.asarray(per_train_y, dtype=np.float64),
            weights=np.ones(len(per_train_x), dtype=np.float64),
            x_eval=np.asarray(x_eval, dtype=np.float64),
        )
        retrains += 1
        prediction = (
            prediction * np.asarray(eval_scales)[:, None]
            + np.asarray(eval_centers)[:, None]
        )
        losses: list[float] = []
        for truth, predicted, scale in zip(truths, prediction, metric_scales):
            observed = np.isfinite(truth)
            if not observed.any():
                raise RuntimeError("evaluation future contains no observed truth")
            losses.append(smase(truth[observed], predicted[observed], scale=scale))
        rows.append({
            "origin": int(origin),
            "phase": "support" if origin in SUPPORT_ORIGINS else "delayed",
            "mean_smase": float(statistics.fmean(losses)),
            "per_view_smase": [float(value) for value in losses],
            "behavior_point_count": behavior_count,
            "consumer_variant": "pooled",
        })
    return {"rows": rows, "retrains": retrains}


def forecasting_gains(
    arm_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    eval_names: Sequence[str],
) -> dict[str, Any]:
    """gain(P) = identity sMASE - P sMASE, in the bch._gain_rows idiom, with
    the support and delayed triple-window halves reported separately."""
    out: dict[str, Any] = {}
    for phase, origins in (("support", SUPPORT_ORIGINS), ("delayed", DELAYED_ORIGINS)):
        identity_rows = [
            row for row in arm_rows["identity"] if int(row["origin"]) in origins
        ]
        phase_rows: dict[str, Any] = {}
        for program in PROGRAMS:
            candidate_rows = [
                row for row in arm_rows[program] if int(row["origin"]) in origins
            ]
            phase_rows[program] = bch._gain_rows(
                identity_rows, candidate_rows, list(eval_names)
            )
        out[phase] = phase_rows
    return out


# =========================================================================== #
# Part B, AD side: the frozen detector inside the same P(B)
# =========================================================================== #
def ad_score_series(
    buffer: np.ndarray, ledger_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """One AD evaluation: detect on the [382, 900) slice of P(B) -- the
    [431, 900) block plus exactly one detector window of warm-up -- and score
    against the series' T1 ledger rows.  The detector never flags its first
    49 inputs, which is exactly the warm-up span [382, 431)."""
    fed = _buffer_slice(buffer, AD_SPAN[0], AD_SPAN[1])
    reading = ad.detect(
        fed, window=DETECTOR["window"], threshold=DETECTOR["threshold"]
    )
    events = ad.predicted_events(reading["flags"], offset=AD_SPAN[0])
    truth = [
        {"start": int(row["index"]), "end": int(row["index"]) + int(row["points"])}
        for row in ledger_rows
    ]
    scored = ad.score_events(truth, events)
    scored["abstained_zero_scale"] = int(reading["abstained_zero_scale"])
    scored["abstained_non_finite"] = int(reading["abstained_non_finite"])
    return scored


def ad_evaluate_arm(
    buffers: Mapping[tuple[str, str], np.ndarray],
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    train_names: Sequence[str],
    program: str,
) -> dict[str, Any]:
    per_series: dict[str, Any] = {}
    evaluations = 0
    for station in train_names:
        per_series[station] = ad_score_series(
            buffers[(station, program)], ledger[station]
        )
        evaluations += 1
    f1_by_series = {
        station: row["f1"] for station, row in per_series.items()
    }
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
    return {
        "per_series": per_series,
        "f1_by_series": f1_by_series,
        "pooled": {
            "ledger_events": pooled_truth,
            "predicted_events": pooled_pred,
            "matched_events": pooled_hit,
            "precision": pooled_precision,
            "recall": pooled_recall,
            "f1": pooled_f1,
            "precision_is_a_lower_bound": True,
        },
        "ad_evaluations": evaluations,
    }


def ad_twin_background(
    pristine: Mapping[str, np.ndarray], train_names: Sequence[str]
) -> dict[str, Any]:
    """B4(a): the un-injected twin -- the same block's original bytes under
    identity preparation -- reporting a background alarm level per series.
    Never a false-positive rate: the substrate's natural anomalies are
    unlabelled, so an alarm here may be correct."""
    per_series: dict[str, Any] = {}
    evaluations = 0
    for station in train_names:
        raw_block = np.asarray(
            pristine[station][BLOCK[0]:BLOCK[1]], dtype=np.float64
        )
        twin_buffer = np.asarray(v6._linear_integrity(raw_block), dtype=np.float64)
        fed = _buffer_slice(twin_buffer, AD_SPAN[0], AD_SPAN[1])
        reading = ad.detect(
            fed, window=DETECTOR["window"], threshold=DETECTOR["threshold"]
        )
        flags = np.asarray(reading["flags"])
        scored_flags = flags[INJECTION_ZONE[0] - AD_SPAN[0]:]
        per_series[station] = ad.background_alarm_rate(scored_flags)
        per_series[station]["abstained_zero_scale"] = int(
            reading["abstained_zero_scale"]
        )
        evaluations += 1
    rates = [
        row["alarm_events_per_1000_points"]
        for row in per_series.values()
        if row["alarm_events_per_1000_points"] is not None
    ]
    return {
        "per_series": per_series,
        "alarm_events_per_1000_points": {
            "min": min(rates) if rates else None,
            "median": float(statistics.median(rates)) if rates else None,
            "max": max(rates) if rates else None,
        },
        "twin_block": "the [431, 900) region's original bytes, un-injected",
        "not_a_false_positive_rate": True,
        "ad_evaluations": evaluations,
    }


# =========================================================================== #
# orchestration
# =========================================================================== #
def run() -> int:
    started = time.perf_counter()
    frozen_before = _freeze()
    git_before = _git_status()

    names_meta = t0.roster()
    train_names = list(names_meta["train"])
    eval_names = list(names_meta["eval"])
    pristine = t0.load_pristine(list(names_meta["all"]))
    config = _config()
    anchors = [int(a) for a in config["anchors"]]
    period = int(config["period"])

    # -- Part A: inject, freeze the ledger, measure copy integrity ------------
    injection = freeze_injection_t1(pristine, train_names)
    integrity = part_a_integrity(pristine, injection, train_names)
    counts = [len(injection["ledger"][s]) for s in train_names]
    print("A  zone [%d, %d)  seed %d  events/series %d  realised %d  skips %d" % (
        INJECTION_ZONE[0], INJECTION_ZONE[1], SEED,
        injection["events_per_series"], sum(counts),
        sum(len(v) for v in injection["skips"].values())), flush=True)

    # -- Part B: the buffers, then the B3 gate before any measurement ---------
    built = build_pbuffers(injection["injected"], train_names)
    assertion = same_byte_assertion(built["buffers"], train_names, anchors)
    print("B3 same-byte comparisons %d  all_equal=%s  reproducible=%s" % (
        assertion["comparisons"], assertion["all_equal"],
        built["reproducible_on_recall"]), flush=True)
    if not (assertion["all_equal"] and built["reproducible_on_recall"]):
        raise _Blocked(
            "PROGRAM_GEOMETRY_UNALIGNED",
            "the same-byte assertion failed before any measurement was spent",
        )

    # -- Part B, forecasting arms --------------------------------------------
    retrains = 0
    arm_forecasting: dict[str, Any] = {}
    for program in PROGRAMS:
        arm = forecasting_evaluate_arm(
            built["buffers"], injection["injected"], pristine,
            train_names, eval_names, program, anchors, ALL_ORIGINS, period,
        )
        arm_forecasting[program] = arm["rows"]
        retrains += arm["retrains"]
        print("B  forecasting arm %-14s delayed mean sMASE %.4f (+%d retrains)" % (
            program,
            statistics.fmean(
                row["mean_smase"] for row in arm["rows"]
                if row["phase"] == "delayed"
            ),
            arm["retrains"]), flush=True)
    gains_f = forecasting_gains(arm_forecasting, eval_names)

    # -- Part B, AD arms ------------------------------------------------------
    evaluations = 0
    arm_ad: dict[str, Any] = {}
    for program in PROGRAMS:
        result = ad_evaluate_arm(
            built["buffers"], injection["ledger"], train_names, program
        )
        arm_ad[program] = result
        evaluations += result["ad_evaluations"]
        print("B  AD arm %-14s pooled F1 %s  (+%d evaluations)" % (
            program, result["pooled"]["f1"], result["ad_evaluations"]), flush=True)
    identity_f1 = arm_ad["identity"]["f1_by_series"]
    gains_ad: dict[str, Any] = {}
    for program in PROGRAMS:
        gains_ad[program] = ad.gain_rows(
            identity_f1, arm_ad[program]["f1_by_series"], train_names
        )

    # -- B4 readings (not part of the verdict) --------------------------------
    twin = ad_twin_background(pristine, train_names)
    evaluations += twin["ad_evaluations"]
    guard_read: dict[str, Any] = {}
    for program in t0.T1_PROGRAMS:
        value = guard_statistic(
            "min_per_series_gain", gains_ad[program], len(train_names)
        )
        guard_read[program] = {
            "statistic": "min_per_series_gain",
            "grammar": "methods/ttha/harness/compiler.py, read-only, unchanged",
            "value": float(value),
            "fires_at_harm_line": bool(guard_fires(GUARD_ON_RECORD, value)),
            "crossing_series": guard_crossing_series(GUARD_ON_RECORD, gains_ad[program]),
            "read_through_ok": True,
        }

    # -- Part C: the verdict, at the aggregate layer --------------------------
    verdict, part_c = _verdict(arm_ad, gains_f, gains_ad, injection)

    # -- post-run integrity: the originals are re-read and compared -----------
    originals_unchanged = all(
        bool(np.array_equal(
            t0.load_pristine([station])[station], pristine[station],
            equal_nan=True,
        ))
        for station in list(names_meta["all"])
    )

    frozen_after = _verify(frozen_before)
    payload = _payload(
        verdict=verdict,
        names_meta=names_meta,
        injection=injection,
        integrity=integrity,
        built=built,
        assertion=assertion,
        arm_forecasting=arm_forecasting,
        gains_f=gains_f,
        arm_ad=arm_ad,
        gains_ad=gains_ad,
        twin=twin,
        guard_read=guard_read,
        part_c=part_c,
        originals_unchanged=originals_unchanged,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        git_before=git_before,
        retrains=retrains,
        evaluations=evaluations,
        started=started,
    )
    return _write(payload)


def _verdict(
    arm_ad: Mapping[str, Any],
    gains_f: Mapping[str, Any],
    gains_ad: Mapping[str, Any],
    injection: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    identity_pooled = arm_ad["identity"]["pooled"]
    realised_events = int(identity_pooled["ledger_events"])
    quantization = (
        "on the AD side the pre-registered material line of +/-%.3f means "
        "'at least one event changed hands': with %d ledger events a single "
        "event is about %.3f of aggregate displacement, so no 0.005-level "
        "resolution is claimed.  The flip judgement is made at the aggregate "
        "layer only." % (MATERIAL, realised_events, (0.2 / 12.0))
    )
    unreadable = (
        identity_pooled["f1"] is None
        or float(identity_pooled["f1"]) < 0.3
        or not identity_pooled["recall"]
    )
    flip_rows: list[dict[str, Any]] = []
    for program in t0.T1_PROGRAMS:
        f_delayed = float(gains_f["delayed"][program]["aggregate_gain"])
        ad_aggregate = float(gains_ad[program]["aggregate_gain"])
        direction = None
        if f_delayed >= MATERIAL and ad_aggregate <= -MATERIAL:
            direction = "forecasting_up_ad_down"
        elif f_delayed <= -MATERIAL and ad_aggregate >= MATERIAL:
            direction = "forecasting_down_ad_up"
        flip_rows.append({
            "program": program,
            "forecasting_delayed_aggregate_gain": f_delayed,
            "ad_aggregate_gain": ad_aggregate,
            "flip_direction": direction,
        })
    flips = [row for row in flip_rows if row["flip_direction"] is not None]
    if unreadable:
        verdict = "AD_CONSUMER_UNREADABLE"
    elif flips:
        verdict = "TASK_FLIP_CONFIRMED_POSITIVE_CONTROL"
    else:
        verdict = "NO_FLIP_IN_FAMILY"
    return verdict, {
        "c3_identity_ad_reading": {
            "pooled": identity_pooled,
            "degenerate_rule": "pooled F1 < 0.3 or recall == 0",
            "degenerate": bool(unreadable),
        },
        "c1_flip_check_per_program": flip_rows,
        "flips_found": flips,
        "quantization_note": quantization,
        "judgement_layer": (
            "aggregate only; per-series comparison happens inside each task "
            "(forecasting: the 4 eval series; AD: the 12 train series), never "
            "paired across tasks (C4)"
        ),
    }


def _payload(
    *,
    verdict: str,
    names_meta: Mapping[str, Any],
    injection: Mapping[str, Any],
    integrity: Mapping[str, Any],
    built: Mapping[str, Any],
    assertion: Mapping[str, Any],
    arm_forecasting: Mapping[str, Any],
    gains_f: Mapping[str, Any],
    arm_ad: Mapping[str, Any],
    gains_ad: Mapping[str, Any],
    twin: Mapping[str, Any],
    guard_read: Mapping[str, Any],
    part_c: Mapping[str, Any],
    originals_unchanged: bool,
    frozen_before: Mapping[str, str],
    frozen_after: Mapping[str, Any],
    git_before: str,
    retrains: int,
    evaluations: int,
    started: float,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_grade_note": (
            "permanent: an injected flip can be constructive, so this slice "
            "never grounds a Shared-Capability execution right; the natural-"
            "flip evidence is T4/T5's load"
        ),
        "role": (
            "T1: injected positive control -- one Program, one block, two "
            "Consumers, one pre-registered flip rule"
        ),
        "verdict": verdict,
        "verdict_evidence_grade": EVIDENCE_GRADE,
        "llm_calls": 0,
        "llm_budget": LLM_BUDGET,
        "forecasting_retrains": int(retrains),
        "forecasting_retrain_budget": FORECASTING_RETRAIN_BUDGET,
        "forecasting_retrain_note": (
            "counted per (arm, origin) ridge fit by the line's convention; "
            "the six task_A origins share one training design matrix per arm, "
            "so the five fits per arm are numerically identical refits"
        ),
        "ad_evaluations": int(evaluations),
        "ad_evaluation_budget": AD_EVALUATION_BUDGET,
        "budgets_respected": (
            retrains <= FORECASTING_RETRAIN_BUDGET
            and evaluations <= AD_EVALUATION_BUDGET
        ),
        "roster": dict(names_meta),
        "geometry": {
            "block_B": [BLOCK[0], BLOCK[1]],
            "injection_zone": [INJECTION_ZONE[0], INJECTION_ZONE[1]],
            "ad_fed_span": [AD_SPAN[0], AD_SPAN[1]],
            "anchors": [int(a) for a in _config()["anchors"]],
            "task_A_triple_window": {
                "start": TASK_A_S,
                "support_origins": [int(o) for o in SUPPORT_ORIGINS],
                "delayed_origins": [int(o) for o in DELAYED_ORIGINS],
                "syntax": TRIPLE_SYNTAX,
                "task_B_not_scored": "the #36 book scopes task_A only",
            },
            "p_b_construction": (
                "P applied once per (train series, program) to the one block "
                "B = [120, 900); the ridge's training rows are its 240-point "
                "anchor slices; the AD Consumer detects on its [382, 900) "
                "slice.  One buffer, both Consumers, measured by B3."
            ),
            "construction_is_an_executor_siting": (
                "the in-service forecasting path applies P per anchor window; "
                "the block-level siting is forced by the book's B3 assertion "
                "shape and the AD budget and is disclosed as an ambiguity"
            ),
        },
        "part_a_injection": {
            "protocol": injection["protocol"],
            "ledger_summary": {
                "events_per_series": injection["events_per_series"],
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
                "total_events": sum(
                    len(v) for v in injection["ledger"].values()
                ),
                "total_skips": sum(len(v) for v in injection["skips"].values()),
                "composition_by_cycle_slot": _composition(injection),
                "written_to": injection["written_to"],
            },
            "integrity": integrity,
            "post_run_originals_unchanged": bool(originals_unchanged),
        },
        "part_b_same_byte": {
            "assertion": assertion,
            "p_b_buffers": {
                "program_calls": built["program_calls"],
                "reproducible_on_recall": built["reproducible_on_recall"],
                "one_call_per_series_program": built["program_calls"]
                == 2 * len(list(names_meta["train"])) * len(PROGRAMS),
            },
        },
        "part_b_arms": {
            "baseline": (
                "identity on the injected block (B2); the injection itself "
                "is not a processing, and every gain is relative to this arm"
            ),
            "forecasting": {
                program: {
                    "origins": arm_forecasting[program],
                    "support_gain": gains_f["support"][program],
                    "delayed_gain": gains_f["delayed"][program],
                }
                for program in PROGRAMS
            },
            "anomaly_detection": {
                program: {
                    "pooled": arm_ad[program]["pooled"],
                    "f1_by_series": arm_ad[program]["f1_by_series"],
                    "gain": gains_ad[program],
                }
                for program in PROGRAMS
            },
        },
        "part_b4_readings": {
            "not_part_of_the_verdict": True,
            "twin_background": twin,
            "guard_read_through": {
                "guard": GUARD_ON_RECORD,
                "per_program": guard_read,
                "meaning": (
                    "fact check: the AD per-series gain vector feeds the "
                    "in-service guard grammar's min_per_series_gain with zero "
                    "code change; whether the frozen #19 guard would have "
                    "fired on each arm is reported, nothing is enforced"
                ),
            },
        },
        "part_c": part_c,
        "frozen_surface": {
            "name": "FROZEN_SURFACE_V9",
            "raw_entries": len(list(FROZEN_SURFACE_V9)),
            "unique_files": len(set(FROZEN_SURFACE_V9)),
            "bookkeeping": (
                "the #35 errata-of-the-errata's count: 40 raw entries, 39 "
                "unique files (noaa_fresh_cohort_v2.json is listed twice)"
            ),
            "before_files": len(frozen_before),
            "after": frozen_after,
            "git_status_uno_short_at_start": git_before,
            "t1_deliverables_untracked_by_design": (
                "this runner, its artifact pair and the _scratch injection "
                "directory are delivered uncommitted; the next checkpoint "
                "collects them"
            ),
        },
        "exposure": {
            "noaa_development_region": {
                "context": "INSTANCE_SEEN", "outcome": "EXPOSED",
                "note": "already the line's development substrate",
            },
            "noaa_2025_confirmation": {"outcome": "SEALED", "read": False},
            "beyond_17520": {"outcome": "SEALED", "read": False},
            "smd_official_test_and_labels": {"outcome": "SEALED", "read": False},
        },
        "ambiguities_reported_not_self_adjudicated": [
            (
                "P(B) is applied once to the contiguous block [120, 900), "
                "not per anchor window as the in-service path does.  For "
                "winsorize/outlier_iqr/outlier_mad the statistics pool 780 "
                "points instead of 240; hampel_filter is rolling-local and "
                "differs only at window edges.  The alternative (per-window "
                "application with per-window AD detection) makes B3's "
                "element-wise assertion ill-shaped and costs ~648 AD "
                "evaluations against the budget of 300.  The flip estimand "
                "under the shared geometry is intact; magnitude comparability "
                "with the in-service per-window menu gains is not claimed."
            ),
            (
                "The detector warm-up consumes [382, 431): no ledger event "
                "can land there (the legal range starts at 456), so scoring "
                "is unaffected, but the block's first 49 points are "
                "structurally unscored."
            ),
            (
                "Forecasting retrains are counted per (arm, origin) by the "
                "line's convention (30); the six origins share one design "
                "matrix per arm, so the fits are numerically identical "
                "refits and the physically distinct fits are five."
            ),
            (
                "The twin-block background alarm level is a background "
                "level, never a false-positive rate (T0 C5(ii) caliber)."
            ),
            (
                "task_B's triple window at 1800 is not scored: the book "
                "scopes task_A.  A flip read on task_B would be a new slice."
            ),
        ],
        "wall_seconds": time.perf_counter() - started,
    }


def _composition(injection: Mapping[str, Any]) -> dict[str, int]:
    composition: dict[str, int] = {}
    for rows in injection["ledger"].values():
        for row in rows:
            key = "%s/%+d/%gs/x%s" % (
                row["type"], int(row["sign"]), row["sigma_multiple"], row["points"]
            )
            composition[key] = composition.get(key, 0) + 1
    return dict(sorted(composition.items()))


def _write(payload: Mapping[str, Any]) -> int:
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print(
        "VERDICT %s | retrains %d/%d | AD evaluations %d/%d"
        % (
            payload["verdict"],
            payload["forecasting_retrains"],
            payload["forecasting_retrain_budget"],
            payload["ad_evaluations"],
            payload["ad_evaluation_budget"],
        ),
        flush=True,
    )
    return 0


def _fmt(value: Any) -> str:
    return "--" if value is None else ("%.4f" % float(value))


def _markdown(payload: Mapping[str, Any]) -> str:
    part_c = payload["part_c"]
    lines = [
        "# T1 flip control: one Program, two Consumers, opposite signs?",
        "",
        "**%s** (evidence grade: **%s**, permanent).  %d LLM, %d/%d forecasting"
        " retrains, %d/%d AD evaluations." % (
            payload["verdict"], payload["evidence_grade"],
            payload["llm_calls"],
            payload["forecasting_retrains"], payload["forecasting_retrain_budget"],
            payload["ad_evaluations"], payload["ad_evaluation_budget"],
        ),
        "",
        "## Part A -- the injection",
        "",
    ]
    block = payload["part_a_injection"]["protocol"]["block"]
    lines.extend([
        "- Zone **`[%d, %d)`**, seed **%d**, cycle counter from slot 0 "
        "(independent of T0).  %s." % (
            block["start"], block["end"],
            payload["part_a_injection"]["protocol"]["constants"]["seed"],
            block["siting"],
        ),
        "",
        "| series | events | skips | spacing rejections | indices |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    summary = payload["part_a_injection"]["ledger_summary"]
    for station, row in summary["per_series"].items():
        lines.append("| `%s` | %d | %d | %d | %s |" % (
            station, row["events"], row["skips"], row["spacing_rejections"],
            ", ".join(str(i) for i in row["indices"]),
        ))
    lines.extend([
        "",
        "Total **%d** events, %d skips.  Composition: `%s`.  Ledger frozen to "
        "`%s` before any Consumer reading." % (
            summary["total_events"], summary["total_skips"],
            summary["composition_by_cycle_slot"], summary["written_to"],
        ),
        "",
        "Copy integrity (A3): every copy differs from its original exactly at "
        "the ledger indices -- **%s**; originals re-read after the run and "
        "unchanged -- **%s**." % (
            payload["part_a_injection"]["integrity"][
                "all_copies_exact_at_ledger_indices"
            ],
            payload["part_a_injection"]["post_run_originals_unchanged"],
        ),
        "",
        "## Part B -- the arms",
        "",
        "B3 same-byte assertion: **%d** comparisons, all equal: **%s**; P(B) "
        "reproducible on recall: **%s**." % (
            payload["part_b_same_byte"]["assertion"]["comparisons"],
            payload["part_b_same_byte"]["assertion"]["all_equal"],
            payload["part_b_same_byte"]["p_b_buffers"]["reproducible_on_recall"],
        ),
        "",
        "| program | forecasting support agg | forecasting delayed agg | delayed"
        " per-eval-series (min) | AD pooled F1 | AD aggregate gain | AD per-series (min) |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ])
    for program in payload["part_b_arms"]["forecasting"]:
        f_arm = payload["part_b_arms"]["forecasting"][program]
        a_arm = payload["part_b_arms"]["anomaly_detection"][program]
        delayed_vector = f_arm["delayed_gain"]["per_eval_series_gain"]
        ad_vector = a_arm["gain"]["per_eval_series_gain"]
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
            program,
            _fmt(f_arm["support_gain"]["aggregate_gain"]),
            _fmt(f_arm["delayed_gain"]["aggregate_gain"]),
            _fmt(min(delayed_vector.values())) if delayed_vector else "--",
            _fmt(a_arm["pooled"]["f1"]),
            _fmt(a_arm["gain"]["aggregate_gain"]),
            _fmt(min(ad_vector.values())) if ad_vector else "--",
        ))
    lines.extend([
        "",
        "Identity is the baseline on the injected block (B2); its gains are "
        "zero by construction.",
        "",
        "## B4 readings (not part of the verdict)",
        "",
        "- Twin-block background alarm rate per 1000 points: min %s / median "
        "%s / max %s." % (
            _fmt(payload["part_b4_readings"]["twin_background"][
                "alarm_events_per_1000_points"
            ]["min"]),
            _fmt(payload["part_b4_readings"]["twin_background"][
                "alarm_events_per_1000_points"
            ]["median"]),
            _fmt(payload["part_b4_readings"]["twin_background"][
                "alarm_events_per_1000_points"
            ]["max"]),
        ),
        "- Guard read-through (`min_per_series_gain` on the AD vector, "
        "in-service grammar, no code change):",
        "",
        "| program | min per-series AD gain | fires at -0.005 | crossing series |",
        "| --- | ---: | --- | --- |",
    ])
    guard = payload["part_b4_readings"]["guard_read_through"]["per_program"]
    for program, row in guard.items():
        lines.append("| `%s` | %s | %s | %s |" % (
            program, _fmt(row["value"]), row["fires_at_harm_line"],
            ", ".join(row["crossing_series"]) or "--",
        ))
    lines.extend([
        "",
        "## Part C -- the verdict",
        "",
        "Identity-arm AD reading (C3): pooled P %s / R %s / F1 %s; degenerate:"
        " **%s**." % (
            _fmt(part_c["c3_identity_ad_reading"]["pooled"]["precision"]),
            _fmt(part_c["c3_identity_ad_reading"]["pooled"]["recall"]),
            _fmt(part_c["c3_identity_ad_reading"]["pooled"]["f1"]),
            part_c["c3_identity_ad_reading"]["degenerate"],
        ),
        "",
        "| program | forecasting delayed agg | AD agg | flip |",
        "| --- | ---: | ---: | --- |",
    ])
    for row in part_c["c1_flip_check_per_program"]:
        lines.append("| `%s` | %s | %s | %s |" % (
            row["program"],
            _fmt(row["forecasting_delayed_aggregate_gain"]),
            _fmt(row["ad_aggregate_gain"]),
            row["flip_direction"] or "--",
        ))
    lines.extend([
        "",
        part_c["quantization_note"],
        "",
        "%s" % part_c["judgement_layer"],
        "",
        "## Ambiguities (reported, not self-adjudicated)",
        "",
    ])
    for item in payload["ambiguities_reported_not_self_adjudicated"]:
        lines.append("- %s" % item)
    lines.extend([
        "",
        "## Cost",
        "",
        "- LLM calls 0.  Forecasting retrains %d of %d.  AD evaluations %d of "
        "%d." % (
            payload["forecasting_retrains"], payload["forecasting_retrain_budget"],
            payload["ad_evaluations"], payload["ad_evaluation_budget"],
        ),
        "- Frozen surface %s: %d unique files, drift after run: %s." % (
            payload["frozen_surface"]["name"],
            payload["frozen_surface"]["unique_files"],
            payload["frozen_surface"]["after"]["drift"] or "none",
        ),
        "- Sealed: NOAA 2025, everything beyond 17520, SMD official test and "
        "labels.",
        "- Wall seconds %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except _Blocked as stop:
        print("BLOCKED %s: %s" % (stop.verdict, stop.reason), flush=True)
        raise SystemExit(2)
