"""T1b: the training-side flip control -- does the flip survive two trained Consumers?

T1 showed one Program acting once on one block flips sign between a
forecasting Consumer (trained on P) and a *training-free* AD detector.  The
one question this book authorizes: when the AD Consumer also *trains* on the
same P(training block) and both Consumers are scored on fixed, independent,
unprocessed Query bytes, does the flip still hold -- is the training data's
utility itself task-flipped?

* Part A freezes the trainable AD Consumer (consumers/anomaly_detection_
  trainable_v1.py) and gates its readability on the calibration Query only.
* Part B freezes two Query injection regions (Qf/Qcal) with ledgers on disk
  before any training or scoring; the Query is never P-processed.
* Part C trains both Consumers on the same P(B) bytes (forecasting readings
  reused from the frozen T1 artifact behind a guard, zero retrains expected)
  and scores AD on Qf.
* Part D judges at the aggregate layer, with the event-quantization note.

evidence_grade = POSITIVE_CONTROL, permanently.
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
import run_e2_t1_flip_control as t1  # noqa: E402  -- the frozen T1 book
from run_e2_operational_pipeline import (  # noqa: E402
    FROZEN_SURFACE_V9,
    _freeze,
    _verify,
)
from run_v1_kdd2018_natural_slow_update import _config  # noqa: E402

from consumers import anomaly_detection_v1 as ad  # noqa: E402

# v2 mode (main-line ruling after the v1 AD_TRAINABLE_SPEC_DEFECT stop):
# feature geometry x[t-48:t+1] (current point included), the gate/judgment
# metric is the macro average of per-series event F1, and Qcal is fully read
# before any Qf byte is scored.  v3 mode (same book, continued): the only
# change is the feature family -- the single z_t statistic from the T0
# detector's own detect() path at explicit 49/3.5; single-shot gate, no
# fallback.  Default stays the delivered v1 configuration, so the v1
# artifact remains reproducible as written.
MODE = "v1"
if "--v2" in sys.argv[1:]:
    MODE = "v2"
if "--v3" in sys.argv[1:]:
    MODE = "v3"
V2_MODE = MODE in ("v2", "v3")  # the v2-line ordering and macro metric
V3_MODE = MODE == "v3"
if MODE == "v3":
    from consumers import anomaly_detection_trainable_v3 as adt  # noqa: E402
elif MODE == "v2":
    from consumers import anomaly_detection_trainable_v2 as adt  # noqa: E402
else:
    from consumers import anomaly_detection_trainable_v1 as adt  # noqa: E402

PROTOCOL_VERSION = "t1b_training_flip_%s" % MODE
EVIDENCE_GRADE = "POSITIVE_CONTROL"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / ("t1b_training_flip_%s.json" % MODE)
OUT_MD = E2 / ("t1b_training_flip_%s.md" % MODE)
# gate and judgment metric: macro per-series event F1 in v2/v3, pooled in v1
GATE_METRIC = "pooled_f1" if MODE == "v1" else "macro_f1"
T1_ARTIFACT = E2 / "t1_flip_control_v1.json"
T1_DIR = t1.T1_DIR  # _scratch/phase_t/injected/t1, read-only this round
QUERY_DIR = PROJECT_ROOT / "_scratch" / "phase_t" / "injected" / "t1b_query"

# ---- Part B siting (book-frozen) --------------------------------------------
QF_REGION = (2100, 2560)
QCAL_REGION = (2600, 3060)
QF_SEED = 20260824
QCAL_SEED = 20260825
TASK_SPANS_TO_AVOID = ((912, 1392), (1608, 2088))  # task_A/task_B context+horizon
DEV_LIMIT = 8760

PROGRAMS = t1.PROGRAMS  # identity + the four T1 programs
TRAIN_BLOCK = t1.BLOCK  # [120, 900); P's measured action region
MATERIAL = float(ad.MATERIAL_THRESHOLD)  # 0.005

LLM_BUDGET = 0
# v3 book: forecasting retrains 0 (C3 keeps reusing the T1 artifact); the
# re-measure path is not authorized, so a broken C3 guard stops the run.
FORECASTING_RETRAIN_BUDGET = 0 if V3_MODE else 40
# classifier fits and scorings each count once; the v3 slice carries its own
# 120 cap, and the T1b cumulative cap was raised 300 -> 400 by the main line
AD_EVALUATION_BUDGET = 120 if V3_MODE else 300
T1B_CUMULATIVE_CAP = 400
# AD evaluations spent by the delivered v1/v2 runs (three v1 executions
# ~26+50+50, two v2 executions 38+38 -- the duplicate executions produced
# bit-identical readings and are counted anyway)
T1B_AD_EVALUATIONS_BEFORE_V3 = 202

# Part 0 checkpoint per slice.  The first v3 delivery embedded the v1-era
# checkpoint (a6ba53d) -- corrected by an appended erratum in
# t1b_training_flip_v3.* (the frozen field there is not rewritten); the
# runner itself is fixed here so every future run cites its own slice.
PART0_CHECKPOINT = {
    "v1": {
        "commit": "a6ba53d",
        "files": 6,
        "note": (
            "#37 deliverables + the V9 registry's authorized ssi hash move "
            "(37d31cb8... -> f39c13f3..., T2_OBSERVATION_TOUCHED) + "
            "main-line ledger/roadmap revisions"
        ),
    },
    "v2": {
        "commit": "a6ba53d",
        "files": 6,
        "note": (
            "v2 had no Part 0 of its own; the reference names the "
            "then-latest checkpoint (T2 wiring + #37)"
        ),
    },
    "v3": {
        "commit": "359eec5",
        "files": 9,
        "note": (
            "T1b v1/v2 deliverables (trainable v1/v2 consumers, the runner, "
            "the v1/v2 artifacts) + two main-line docs revisions (v1/v2 "
            "double-stop ruling, instrument supersession, v3 authorization, "
            "#41b prep-book siting)"
        ),
    },
}

READABILITY_GATE_F1 = 0.5  # A3: identity(B)-trained, pooled event F1 on Qcal


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


def _dir_shas(root: Path) -> dict[str, str]:
    return {
        str(path.name): _sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }


# =========================================================================== #
# Part B: the two Query injections -- T0's frozen protocol, per-region seed
# =========================================================================== #
def freeze_query_injection(
    pristine: Mapping[str, np.ndarray],
    train_names: Sequence[str],
    region: tuple[int, int],
    seed: int,
    out_dir: Path,
    role: str,
) -> dict[str, Any]:
    """Line-for-line t1.freeze_injection_t1 with the block/seed/out-dir as
    parameters -- the two regions are this book's only authorized changes.
    The cycle counter starts from slot 0 per region (the T1 convention: each
    freeze is its own seeded draw, independent of every other).
    """
    start, end = int(region[0]), int(region[1])
    for forbidden in TASK_SPANS_TO_AVOID:
        if start < forbidden[1] and forbidden[0] < end:
            raise _Blocked(
                "SCHEMA_BLOCKED",
                "query region %s overlaps a task span %s" % (region, forbidden),
            )
    if end > DEV_LIMIT:
        raise _Blocked("SCHEMA_BLOCKED", "query region leaves the dev region")
    legal_lo = start + t0.BOUNDARY_EXCLUSION
    legal_hi = end - t0.BOUNDARY_EXCLUSION
    per_block = (end - start) // t0.EVENT_DIVISOR

    rng = np.random.default_rng(int(seed))
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

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for station, array in injected.items():
        np.save(out_dir / ("%s.npy" % station), array)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "role": role,
        "evidence_grade": EVIDENCE_GRADE,
        "block": {"start": start, "end": end},
        "constants": {
            "seed": int(seed),
            "events_per_series": per_block,
            "events_per_series_rule": (
                "floor(region length / %d)" % t0.EVENT_DIVISOR
            ),
            "cycle_table": [dict(entry) for entry in t0.CYCLE_TABLE],
            "cycle_counter": (
                "from slot 0, global across the 12 train series in roster "
                "order, advancing only on an accepted event; each Query "
                "region is its own seeded freeze, independent of T0 and T1"
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
                "run_e2_t0_ad_instrument via run_e2_t1_flip_control (both "
                "frozen); only the region, the seed and the output directory "
                "changed, all three book-mandated"
            ),
            "query_processing": (
                "none, ever: the Query is scored unprocessed (B3); P is only "
                "ever applied to the training block [120, 900)"
            ),
            "delta_scale_dependence": (
                "every delta = sign * sigma_multiple * sigma_local(pristine); "
                "no injected value is ever used as a scale source"
            ),
        },
        "ledger": {station: ledger[station] for station in train_names},
        "skips": {station: skips[station] for station in train_names},
        "spacing_rejections": spacing_rejections,
        "frozen_before_any_training_or_scoring": True,
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    (out_dir / "ledger.json").write_text(
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
        "history_isolation": history_isolation_assertions(
            pristine, injected, ledger, (start, end), train_names
        ),
        "written_to": str(out_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


# =========================================================================== #
# v2 execution convention: 168-step history isolation, checked byte-level
# =========================================================================== #
def history_isolation_assertions(
    pristine: Mapping[str, np.ndarray],
    injected: Mapping[str, np.ndarray],
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    region: tuple[int, int],
    train_names: Sequence[str],
) -> dict[str, Any]:
    """Two byte-level guarantees around the 168-step sigma prefix:

    (a) every accepted event's ledger ``sigma_local`` recomputes exactly from
    the *pristine* [t-168, t) prefix -- injected bytes never enter a scale
    source (in the working copy, later events' 168-prefixes can contain
    earlier events, since the spacing 50 < 168; only sourcing from pristine
    keeps the isolation);
    (b) the 168-step pre-region prefix of every injected copy is byte-equal
    to the pristine series -- scoring features that reach back before the
    region never read an injected byte.
    """
    start = int(region[0])
    per_series: dict[str, Any] = {}
    holds = True
    for station in train_names:
        source = np.asarray(pristine[station], dtype=np.float64)
        copy = np.asarray(injected[station], dtype=np.float64)
        history_pristine = bool(np.array_equal(
            copy[start - t0.SIGMA_PREFIX:start],
            source[start - t0.SIGMA_PREFIX:start],
            equal_nan=True,
        ))
        sigma_from_pristine = True
        for row in ledger[station]:
            position = int(row["index"])
            prefix = source[position - t0.SIGMA_PREFIX:position]
            finite = prefix[np.isfinite(prefix)]
            sigma = 0.0
            sigma_source = "mad"
            if finite.size:
                centre = float(np.median(finite))
                sigma = ad.MAD_TO_SIGMA * float(np.median(np.abs(finite - centre)))
                if sigma <= 0.0:
                    sigma_source = "std"
                    sigma = float(np.std(finite))
            if not (
                sigma == float(row["sigma_local"])
                and sigma_source == row["sigma_source"]
            ):
                sigma_from_pristine = False
                break
        per_series[station] = {
            "pre_region_168_pristine": history_pristine,
            "sigma_prefix_from_pristine": sigma_from_pristine,
        }
        holds = holds and history_pristine and sigma_from_pristine
    return {
        "rule": (
            "168-step history isolation: sigma prefixes are sourced from "
            "pristine bytes only (recomputed and compared per event), and "
            "the 168-step pre-region prefix of every scored copy is "
            "byte-equal to pristine"
        ),
        "per_series": per_series,
        "holds": bool(holds),
    }


# =========================================================================== #
# B4: the oracle smoke (robust-z 49/3.5, report only, never a verdict input)
# =========================================================================== #
def oracle_smoke(
    copies: Mapping[str, np.ndarray],
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    train_names: Sequence[str],
    region: tuple[int, int],
) -> dict[str, Any]:
    warm = int(ad.FALLBACK_WINDOW)
    per_series: dict[str, Any] = {}
    evaluations = 0
    for station in train_names:
        array = np.asarray(copies[station], dtype=np.float64)
        fed = array[region[0] - warm:region[1]]
        reading = ad.detect(
            fed, window=ad.FALLBACK_WINDOW, threshold=ad.FALLBACK_THRESHOLD
        )
        events = ad.predicted_events(reading["flags"], offset=region[0] - warm)
        truth = [
            {"start": int(row["index"]), "end": int(row["index"]) + int(row["points"])}
            for row in ledger[station]
        ]
        scored = ad.score_events(truth, events)
        scored["abstained_zero_scale"] = int(reading["abstained_zero_scale"])
        per_series[station] = scored
        evaluations += 1
    pooled = {
        "ledger_events": sum(int(r["ledger_events"]) for r in per_series.values()),
        "predicted_events": sum(int(r["predicted_events"]) for r in per_series.values()),
        "matched_events": sum(int(r["matched_events"]) for r in per_series.values()),
    }
    pooled["f1"] = None
    if pooled["ledger_events"]:
        precision = (
            pooled["matched_events"] / pooled["predicted_events"]
            if pooled["predicted_events"] else 0.0
        )
        recall = pooled["matched_events"] / pooled["ledger_events"]
        pooled["f1"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0 else 0.0
        )
    return {
        "per_series": per_series,
        "pooled": pooled,
        "ad_evaluations": evaluations,
        "role": "injection-visibility oracle only; not a verdict input",
    }


# =========================================================================== #
# Part C: both Consumers trained on the same P(B) bytes
# =========================================================================== #
def training_set(
    buffers: Mapping[tuple[str, str], np.ndarray],
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    train_names: Sequence[str],
    program: str,
    window: int,
) -> dict[str, Any]:
    """Stack the 12 series' trailing-window samples from P(B), labels from the
    T1 ledger (burst: all 3 points), standardization per series from that
    series' own P(B) bytes."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    constants: dict[str, Any] = {}
    exclusions: dict[str, Any] = {}
    for station in train_names:
        buffer = np.asarray(buffers[(station, program)], dtype=np.float64)
        events = {
            int(point)
            for row in ledger[station]
            for point in range(int(row["index"]), int(row["index"]) + int(row["points"]))
        }
        if V3_MODE:
            # the v3 feature family: one detect() pass over P(B) yields z_t;
            # undefined-z points never enter the fit and are counted by cause
            feats = adt.block_features(buffer)
            z = np.asarray(feats["z"], dtype=np.float64)
            constants[station] = {
                "median": None,
                "scale": None,
                "source": "none_v3_z_feature",
            }
            exclusions[station] = feats["counts"]
            absolute = np.arange(buffer.size, dtype=np.int64) + TRAIN_BLOCK[0]
            y_all = np.array(
                [1 if int(t) in events else 0 for t in absolute],
                dtype=np.float64,
            )
            finite = np.isfinite(z)
            xs.append(z[finite, None])
            ys.append(y_all[finite])
            continue
        const = adt.standardization_constants(buffer)
        constants[station] = const
        x, indices = adt.features_for_range(
            buffer, window, buffer.size,
            window=window, median=const["median"], scale=const["scale"],
        )
        absolute = indices + TRAIN_BLOCK[0]
        y = np.array(
            [1 if int(t) in events else 0 for t in absolute], dtype=np.float64
        )
        xs.append(x)
        ys.append(y)
    return {
        "X": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "constants": constants,
        "exclusions": exclusions if V3_MODE else None,
    }


def fit_arm(
    buffers: Mapping[tuple[str, str], np.ndarray],
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    train_names: Sequence[str],
    program: str,
    window: int,
) -> dict[str, Any]:
    data = training_set(buffers, ledger, train_names, program, window)
    model = adt.fit(data["X"], data["y"])
    model["constants"] = data["constants"]
    model["training_samples"] = int(data["X"].shape[0])
    model["training_positives"] = int(model["n_pos"])
    model["training_exclusions"] = data["exclusions"]
    return model


def score_arm_on_query(
    model: Mapping[str, Any],
    copies: Mapping[str, np.ndarray],
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    train_names: Sequence[str],
    region: tuple[int, int],
    window: int,
) -> dict[str, Any]:
    per_series: dict[str, Any] = {}
    evaluations = 0
    for station in train_names:
        const = model["constants"][station]
        per_series[station] = adt.score_query_series(
            model,
            copies[station],
            region,
            ledger[station],
            window=window,
            median=const["median"],
            scale=const["scale"],
        )
        evaluations += 1
    pooled = adt.pooled_f1(per_series)
    macro_fn = getattr(adt, "macro_f1", None)
    macro = macro_fn(per_series) if macro_fn is not None else None
    auprcs = [
        float(row["auprc"]) for row in per_series.values() if row["auprc"] is not None
    ]
    return {
        "per_series": per_series,
        "f1_by_series": {
            station: row["f1"] for station, row in per_series.items()
        },
        "pooled_f1": pooled,
        "macro_f1": macro,
        "auprc_by_series": {
            station: row["auprc"] for station, row in per_series.items()
        },
        "auprc_mean": float(np.mean(auprcs)) if auprcs else None,
        "undefined_points_by_series": {
            station: int(row.get("undefined_points", 0))
            for station, row in per_series.items()
        },
        "undefined_points_total": sum(
            int(row.get("undefined_points", 0)) for row in per_series.values()
        ),
        "undefined_attribution_total": _sum_attribution(per_series),
        "ad_evaluations": evaluations,
    }


def _sum_attribution(per_series: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    totals = {"warm_up": 0, "zero_scale": 0, "non_finite": 0}
    for row in per_series.values():
        for cause, count in (row.get("undefined_attribution") or {}).items():
            totals[cause] = totals.get(cause, 0) + int(count)
    return totals


# =========================================================================== #
# forecasting reuse: the frozen T1 readings behind the C3 guard
# =========================================================================== #
def forecasting_reused(names_meta: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the delayed gains from the T1 artifact's recorded per-origin
    rows with the same bch._gain_rows code, and cross-check them against the
    recorded c1 aggregates.  Zero retrains."""
    payload = json.loads(T1_ARTIFACT.read_text(encoding="utf-8"))
    fore = payload["part_b_arms"]["forecasting"]
    eval_names = list(names_meta["eval"])
    delayed = set(t1.DELAYED_ORIGINS)
    rebuilt: dict[str, Any] = {}
    for program in PROGRAMS:
        identity_rows = [
            row for row in fore["identity"]["origins"]
            if int(row["origin"]) in delayed
        ]
        candidate_rows = [
            row for row in fore[program]["origins"]
            if int(row["origin"]) in delayed
        ]
        rebuilt[program] = bch._gain_rows(identity_rows, candidate_rows, eval_names)
    recorded = {
        row["program"]: float(row["forecasting_delayed_aggregate_gain"])
        for row in payload["part_c"]["c1_flip_check_per_program"]
    }
    cross_check = {
        program: {
            "rebuilt_aggregate": rebuilt[program]["aggregate_gain"],
            "recorded_aggregate": recorded[program],
            "abs_diff": abs(rebuilt[program]["aggregate_gain"] - recorded[program]),
        }
        for program in recorded
    }
    consistent = all(row["abs_diff"] < 1e-9 for row in cross_check.values())
    return {
        "delayed_gains": rebuilt,
        "cross_check_vs_recorded_c1": cross_check,
        "consistent_with_recorded": bool(consistent),
        "source": "artifacts/functional/e2/t1_flip_control_v1.json, read-only",
        "retrains": 0,
    }


def forecasting_remeasure(
    buffers: Mapping[tuple[str, str], np.ndarray],
    injected_t1: Mapping[str, np.ndarray],
    pristine: Mapping[str, np.ndarray],
    names_meta: Mapping[str, Any],
    anchors: Sequence[int],
    period: int,
) -> dict[str, Any]:
    """The pre-authorized fallback if the C3 guard breaks: re-run T1's own
    per-arm evaluation (6 retrains per arm, 30 total, inside the 40 budget)."""
    train_names = list(names_meta["train"])
    eval_names = list(names_meta["eval"])
    arm_rows: dict[str, Any] = {}
    retrains = 0
    for program in PROGRAMS:
        arm = t1.forecasting_evaluate_arm(
            buffers, injected_t1, pristine,
            train_names, eval_names, program, anchors, t1.ALL_ORIGINS, period,
        )
        arm_rows[program] = arm["rows"]
        retrains += int(arm["retrains"])
    if retrains > FORECASTING_RETRAIN_BUDGET:
        raise _Blocked(
            "SUBSTRATE_GUARD_FAIL",
            "re-measurement would spend %d retrains, over the %d budget"
            % (retrains, FORECASTING_RETRAIN_BUDGET),
        )
    gains = t1.forecasting_gains(arm_rows, eval_names)
    return {
        "delayed_gains": {
            program: gains["delayed"][program] for program in PROGRAMS
        },
        "cross_check_vs_recorded_c1": None,
        "consistent_with_recorded": None,
        "source": "re-measured live after the C3 guard broke (authorized, <=40)",
        "retrains": retrains,
    }


# =========================================================================== #
def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run() -> int:
    """Thin wrapper: a pre-registered stop still lands on disk -- the artifact
    records the verdict, the stages completed before the block, and the costs
    spent so far (the book's deliverables exist for every verdict)."""
    completed: dict[str, Any] = {}
    try:
        return _run_inner(completed)
    except _Blocked as stop:
        print("BLOCKED %s: %s" % (stop.verdict, stop.reason), flush=True)
        return _write_blocked(_blocked_payload(stop, completed))


def _run_inner(completed: dict[str, Any]) -> int:
    started = time.perf_counter()
    frozen_before = _freeze()
    git_before = _git_status()
    t1_shas_before = _dir_shas(T1_DIR)

    names_meta = t0.roster()
    train_names = list(names_meta["train"])
    pristine = t0.load_pristine(list(names_meta["all"]))
    config = _config()
    anchors = [int(a) for a in config["anchors"]]
    period = int(config["period"])

    evaluations = 0
    retrains = 0
    completed.update({
        "started": started,
        "frozen_before": frozen_before,
        "git_before": git_before,
        "t1_shas_before": t1_shas_before,
        "names_meta": names_meta,
        "pristine_shas": {
            station: _array_sha(pristine[station])
            for station in list(names_meta["all"])
        },
        "evaluations": evaluations,
        "retrains": retrains,
    })

    # -- Part B: the two Query injections, frozen before any training/scoring --
    qf = freeze_query_injection(
        pristine, train_names, QF_REGION, QF_SEED, QUERY_DIR / "qf",
        "T1b formal Query Qf.  Frozen before any training or scoring.",
    )
    qcal = freeze_query_injection(
        pristine, train_names, QCAL_REGION, QCAL_SEED, QUERY_DIR / "qcal",
        "T1b calibration Query Qcal; serves the A3 readability gate only.",
    )
    qf_shas = _dir_shas(QUERY_DIR / "qf")
    qcal_shas = _dir_shas(QUERY_DIR / "qcal")
    for name, region, inj in (("qf", QF_REGION, qf), ("qcal", QCAL_REGION, qcal)):
        counts = [len(inj["ledger"][s]) for s in train_names]
        print("B  %s region [%d, %d)  events/series %d  realised %d  skips %d" % (
            name, region[0], region[1], inj["events_per_series"], sum(counts),
            sum(len(v) for v in inj["skips"].values())), flush=True)
    integrity_qf = t1.part_a_integrity(pristine, qf, train_names)
    integrity_qcal = t1.part_a_integrity(pristine, qcal, train_names)
    completed.update({
        "stage": "part_b_injection",
        "qf": qf, "qcal": qcal,
        "integrity_qf": integrity_qf, "integrity_qcal": integrity_qcal,
        "qf_shas": qf_shas, "qcal_shas": qcal_shas,
    })

    # -- B4: oracle smoke, report only -----------------------------------------
    # v2 ordering convention: Qcal is fully read (oracle + gate, fallback
    # included) before any Qf byte is scored; Qf's oracle smoke sits behind
    # the gate.  v3 book B3: cite the recorded values, never re-run.
    if V3_MODE:
        oracle = {
            "qcal": {
                "pooled": {"f1": 0.7457627118644068},
                "ad_evaluations": 0,
                "cited_from": (
                    "t1b_training_flip_v1/v2 B4 oracle smoke (robust-z 49/3.5, "
                    "injection-visibility only); not re-run this slice"
                ),
            },
            "qf": {
                "pooled": {"f1": 0.6976744186046512},
                "ad_evaluations": 0,
                "cited_from": (
                    "t1b_training_flip_v1 B4 oracle smoke (the v2 run stopped "
                    "at the gate and never read Qf); not re-run this slice"
                ),
            },
        }
        print("B4 oracle cited from record: Qcal 0.7458 / Qf 0.6977 "
              "(not re-run)", flush=True)
    elif V2_MODE:
        oracle = {
            "qcal": oracle_smoke(qcal["injected"], qcal["ledger"], train_names, QCAL_REGION),
            "qf": None,
        }
        evaluations += oracle["qcal"]["ad_evaluations"]
        print("B4 oracle Qcal pooled F1 %s (Qf deferred behind the gate)" % (
            oracle["qcal"]["pooled"]["f1"]), flush=True)
    else:
        oracle = {
            "qf": oracle_smoke(qf["injected"], qf["ledger"], train_names, QF_REGION),
            "qcal": oracle_smoke(qcal["injected"], qcal["ledger"], train_names, QCAL_REGION),
        }
        evaluations += oracle["qf"]["ad_evaluations"] + oracle["qcal"]["ad_evaluations"]
        print("B4 oracle Qf pooled F1 %s  Qcal pooled F1 %s" % (
            oracle["qf"]["pooled"]["f1"], oracle["qcal"]["pooled"]["f1"]), flush=True)
    completed.update({
        "stage": "b4_oracle_smoke",
        "oracle": oracle,
        "evaluations": evaluations,
    })

    # -- Part C first half: the buffers and the C2 same-byte gate ---------------
    injected_t1 = {
        station: np.asarray(
            np.load(T1_DIR / ("%s.npy" % station)), dtype=np.float64
        )
        for station in train_names
    }
    t1_ledger = json.loads((T1_DIR / "ledger.json").read_text(encoding="utf-8"))
    built = t1.build_pbuffers(injected_t1, train_names)
    assertion = t1.same_byte_assertion(built["buffers"], train_names, anchors)
    print("C2 same-byte comparisons %d  all_equal=%s  reproducible=%s" % (
        assertion["comparisons"], assertion["all_equal"],
        built["reproducible_on_recall"]), flush=True)
    if not (assertion["all_equal"] and built["reproducible_on_recall"]):
        raise _Blocked(
            "PROGRAM_GEOMETRY_UNALIGNED",
            "the same-byte assertion failed before any training was spent",
        )
    # B3 online: P is only ever applied inside build_pbuffers, to the 780-point
    # training block.  The Query copies enter only score_query_series, raw.
    query_never_processed = {
        "rule": "v6._apply_program is called only inside t1.build_pbuffers, on "
                "[120, 900) slices; the Query arrays are never an argument",
        "program_calls": int(built["program_calls"]),
        "expected_program_calls": 2 * len(train_names) * len(PROGRAMS),
        "holds": int(built["program_calls"]) == 2 * len(train_names) * len(PROGRAMS),
    }
    if not query_never_processed["holds"]:
        raise _Blocked(
            "PROGRAM_GEOMETRY_UNALIGNED",
            "unexpected program-application count; the query isolation "
            "argument no longer holds by construction",
        )
    completed.update({
        "stage": "c2_same_byte",
        "assertion": assertion,
        "query_never_processed": query_never_processed,
        "built_meta": {
            "program_calls": int(built["program_calls"]),
            "reproducible_on_recall": bool(built["reproducible_on_recall"]),
        },
    })

    # -- Part A gate: readability on Qcal only ----------------------------------
    window = int(adt.FEATURE_WINDOW)
    fallback_taken = False
    gate_model = fit_arm(
        built["buffers"], t1_ledger, train_names, "identity", window
    )
    evaluations += 1
    gate = score_arm_on_query(
        gate_model, qcal["injected"], qcal["ledger"], train_names,
        QCAL_REGION, window,
    )
    evaluations += gate["ad_evaluations"]
    print("A3 gate window %d  identity-trained Qcal %s %s (pooled %s)" % (
        window, GATE_METRIC, gate[GATE_METRIC], gate["pooled_f1"]), flush=True)
    if V3_MODE and (
        gate[GATE_METRIC] is None or float(gate[GATE_METRIC]) < READABILITY_GATE_F1
    ):
        gate["training_exclusions"] = gate_model["training_exclusions"]
        completed.update({
            "stage": "a3_readability_gate",
            "window": window,
            "fallback_taken": False,
            "gate": gate,
            "gate_passed": False,
            "evaluations": evaluations,
        })
        raise _Blocked(
            "SUPERVISED_AD_PC_FAMILY_CLOSED",
            "v3 is single-shot with no fallback: the identity-trained "
            "threshold head on z_t reads Qcal at macro-averaged per-series "
            "event F1 %s, below %.2f; the supervised-AD positive-control "
            "family closes on three consistent specifications (v1/v2/v3)"
            % (gate[GATE_METRIC], READABILITY_GATE_F1),
        )
    if gate[GATE_METRIC] is None or float(gate[GATE_METRIC]) < READABILITY_GATE_F1:
        fallback_taken = True
        window = int(adt.FALLBACK_FEATURE_WINDOW)
        gate_model = fit_arm(
            built["buffers"], t1_ledger, train_names, "identity", window
        )
        evaluations += 1
        gate_fallback = score_arm_on_query(
            gate_model, qcal["injected"], qcal["ledger"], train_names,
            QCAL_REGION, window,
        )
        evaluations += gate_fallback["ad_evaluations"]
        print("A3 gate FALLBACK window %d  %s %s (pooled %s)" % (
            window, GATE_METRIC, gate_fallback[GATE_METRIC],
            gate_fallback["pooled_f1"]), flush=True)
        if (
            gate_fallback[GATE_METRIC] is None
            or float(gate_fallback[GATE_METRIC]) < READABILITY_GATE_F1
        ):
            completed.update({
                "stage": "a3_readability_gate",
                "window": window,
                "fallback_taken": fallback_taken,
                "gate": gate,
                "gate_fallback": gate_fallback,
                "gate_passed": False,
                "evaluations": evaluations,
            })
            raise _Blocked(
                "AD_TRAINABLE_SPEC_DEFECT",
                "the identity-trained classifier reads Qcal at %s event F1 "
                "%s under the primary window and %s under the fallback; both "
                "below %.2f" % (
                    "macro-averaged per-series" if V2_MODE else "pooled",
                    gate[GATE_METRIC], gate_fallback[GATE_METRIC],
                    READABILITY_GATE_F1,
                ),
            )
        gate = {"primary": gate, "fallback": gate_fallback}
    if V3_MODE:
        gate["training_exclusions"] = gate_model["training_exclusions"]
    completed.update({
        "stage": "a3_gate_passed",
        "window": window,
        "fallback_taken": fallback_taken,
        "gate": gate,
        "gate_passed": True,
        "evaluations": evaluations,
    })

    # -- v2 ordering: only now, behind the passed gate, Qf is first read -------
    if MODE == "v2":
        oracle["qf"] = oracle_smoke(
            qf["injected"], qf["ledger"], train_names, QF_REGION
        )
        evaluations += oracle["qf"]["ad_evaluations"]
        completed.update({"oracle": oracle, "evaluations": evaluations})
        print("B4 oracle Qf pooled F1 %s (behind the passed gate)" % (
            oracle["qf"]["pooled"]["f1"]), flush=True)

    # -- Part C second half: the arms -------------------------------------------
    arm_ad: dict[str, Any] = {}
    for program in PROGRAMS:
        model = fit_arm(built["buffers"], t1_ledger, train_names, program, window)
        evaluations += 1
        result = score_arm_on_query(
            model, qf["injected"], qf["ledger"], train_names, QF_REGION, window,
        )
        evaluations += result["ad_evaluations"]
        result["model"] = {
            "positive_weight": model["positive_weight"],
            "n_pos": model["n_pos"],
            "n_neg": model["n_neg"],
            "training_samples": model["training_samples"],
            "standardization_sources": {
                uid: row["source"] for uid, row in model["constants"].items()
            },
            "training_exclusions": model["training_exclusions"],
        }
        arm_ad[program] = result
        print("C  AD arm %-14s Qf pooled F1 %s  mean AUPRC %s" % (
            program, result["pooled_f1"], result["auprc_mean"]), flush=True)
    completed.update({
        "stage": "c_arms",
        "arm_ad": arm_ad,
        "evaluations": evaluations,
    })

    # -- C3: forecasting readings, reused behind the guard ----------------------
    fore = forecasting_reused(names_meta)
    guard = {
        "t1_copy_shas_unchanged": None,  # filled post-run
        "pbuffers_recomputed_twice_byte_identical": bool(
            built["reproducible_on_recall"]
        ),
        "rebuilt_gains_match_recorded_c1": bool(fore["consistent_with_recorded"]),
    }
    if not fore["consistent_with_recorded"]:
        if V3_MODE:
            completed.update({
                "stage": "c3_guard",
                "evaluations": evaluations,
                "retrains": retrains,
            })
            raise _Blocked(
                "SUBSTRATE_GUARD_FAIL",
                "the C3 guard broke (rebuilt gains disagree with the recorded "
                "c1) and the v3 book authorizes zero forecasting retrains, "
                "so no re-measurement is permitted",
            )
        print("C3 guard broke: rebuilt gains disagree with the recorded c1; "
              "re-measuring within the 40-retrain budget", flush=True)
        fore = forecasting_remeasure(
            built["buffers"], injected_t1, pristine, names_meta, anchors, period
        )
        retrains += int(fore["retrains"])
        guard["remeasurement_spent"] = retrains
        if retrains > FORECASTING_RETRAIN_BUDGET:
            completed.update({
                "stage": "c3_guard_remeasure",
                "evaluations": evaluations,
                "retrains": retrains,
            })
            raise _Blocked(
                "SUBSTRATE_GUARD_FAIL",
                "re-measurement spent %d retrains, over the %d budget"
                % (retrains, FORECASTING_RETRAIN_BUDGET),
            )
    gains_f_delayed = fore["delayed_gains"]
    completed.update({
        "stage": "c3_forecasting",
        "fore": {
            "source": fore["source"],
            "retrains": fore["retrains"],
            "consistent_with_recorded": fore["consistent_with_recorded"],
            "cross_check": fore["cross_check_vs_recorded_c1"],
        },
        "evaluations": evaluations,
        "retrains": retrains,
    })

    # -- Part D: the verdict, aggregate layer ------------------------------------
    flips: list[dict[str, Any]] = []
    for program in t0.T1_PROGRAMS:
        f_gain = float(gains_f_delayed[program]["aggregate_gain"])
        a_base = arm_ad["identity"][GATE_METRIC]
        a_prog = arm_ad[program][GATE_METRIC]
        a_gain = (
            None
            if a_base is None or a_prog is None
            else float(a_prog) - float(a_base)
        )
        direction = None
        if a_gain is not None:
            if f_gain >= MATERIAL and a_gain <= -MATERIAL:
                direction = "forecasting_up_ad_down"
            elif f_gain <= -MATERIAL and a_gain >= MATERIAL:
                direction = "forecasting_down_ad_up"
        flips.append({
            "program": program,
            "forecasting_delayed_aggregate_gain": f_gain,
            "ad_train_gain_pooled": a_gain,
            "ad_gain_metric": GATE_METRIC,
            "flip_direction": direction,
        })
    found = [row for row in flips if row["flip_direction"] is not None]
    verdict = (
        "TRAINING_SIDE_TASK_FLIP_CONFIRMED_POSITIVE_CONTROL"
        if found else "NO_TRAINING_SIDE_FLIP"
    )

    # -- post-run integrity ------------------------------------------------------
    originals_unchanged = all(
        bool(np.array_equal(
            t0.load_pristine([station])[station], pristine[station],
            equal_nan=True,
        ))
        for station in list(names_meta["all"])
    )
    t1_shas_after = _dir_shas(T1_DIR)
    guard["t1_copy_shas_unchanged"] = t1_shas_before == t1_shas_after
    query_shas_after = {
        "qf": _dir_shas(QUERY_DIR / "qf"),
        "qcal": _dir_shas(QUERY_DIR / "qcal"),
    }
    query_copies_unchanged = (
        query_shas_after["qf"] == qf_shas and query_shas_after["qcal"] == qcal_shas
    )
    if not guard["t1_copy_shas_unchanged"]:
        completed.update({
            "stage": "post_run_integrity",
            "evaluations": evaluations,
            "retrains": retrains,
        })
        raise _Blocked(
            "SUBSTRATE_GUARD_FAIL", "a T1 copy moved during the run"
        )

    frozen_after = _verify(frozen_before)
    payload = _payload(
        verdict=verdict,
        names_meta=names_meta,
        qf=qf, qcal=qcal,
        integrity_qf=integrity_qf, integrity_qcal=integrity_qcal,
        oracle=oracle,
        window=window, fallback_taken=fallback_taken, gate=gate,
        built=built, assertion=assertion,
        query_never_processed=query_never_processed,
        arm_ad=arm_ad, flips=flips,
        fore=fore, guard=guard,
        originals_unchanged=originals_unchanged,
        query_copies_unchanged=query_copies_unchanged,
        frozen_before=frozen_before, frozen_after=frozen_after,
        git_before=git_before,
        retrains=retrains, evaluations=evaluations,
        started=started,
    )
    return _write(payload)


def _payload(**kw: Any) -> dict[str, Any]:
    names_meta = kw["names_meta"]
    qf, qcal = kw["qf"], kw["qcal"]
    flips = kw["flips"]
    found = [row for row in flips if row["flip_direction"] is not None]
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "T1b: one P(training block), two *trained* Consumers, both scored "
            "on a fixed independent unprocessed Query -- is the training "
            "data's utility itself task-flipped?"
        ),
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_grade_note": (
            "permanent: an injected flip can be constructive, so this slice "
            "can never become transfer evidence by later reinterpretation"
        ),
        "verdict": kw["verdict"],
        "verdict_evidence_grade": EVIDENCE_GRADE,
        "mode": MODE,
        "execution_conventions_v2": (
            {
                "feature_geometry": (
                    "x[t-48:t+1] (window 49) / x[t-24:t+1] (fallback 25), "
                    "the current point included -- the only method change "
                    "vs v1; alpha, class weight, labels, programs, training "
                    "block and Queries unchanged"
                ),
                "ordering": (
                    "Qcal is fully read (oracle, gate, fallback) before any "
                    "Qf byte is scored; Qf's oracle smoke sits behind the "
                    "passed gate"
                ),
                "history_isolation_168": (
                    "per-event sigma prefix recomputed from pristine bytes "
                    "and compared to the ledger; the 168-step pre-region "
                    "prefix of every scored copy asserted byte-equal to "
                    "pristine (see part_b_queries.*.history_isolation)"
                ),
                "metric": (
                    "the gate and the aggregate judgment read the "
                    "macro-averaged per-series event F1; the pooled F1 is "
                    "reported alongside as a secondary reading"
                ),
                "relation_to_v1": (
                    "v1 (strictly trailing window, pooled F1) stopped at "
                    "AD_TRAINABLE_SPEC_DEFECT in t1b_training_flip_v1; v2 "
                    "was ordered by the main line as the minimal repair"
                ),
            }
            if MODE == "v2" else None
        ),
        "execution_conventions_v3": (
            {
                "feature_family": (
                    "the only change vs v2: the feature is the single "
                    "statistic z_t from anomaly_detection_v1.detect(values, "
                    "window=49, threshold=3.5)['scores'] -- explicit 49/3.5 "
                    "(T0's frozen fallback parameters), never the 25/4.0 "
                    "file defaults; no re-standardization.  Head, labels, "
                    "P(B), arms, Queries, scoring and ordering all frozen "
                    "along v2"
                ),
                "single_shot": (
                    "no fallback: a gate miss closes the supervised-AD "
                    "positive-control family (SUPERVISED_AD_PC_FAMILY_CLOSED)"
                ),
                "abstention": (
                    "T0 semantics: undefined z is excluded from the fit "
                    "(counted by warm_up / zero_scale / non_finite), forced "
                    "to not flag at Query time, excluded from the AUPRC "
                    "ranking with the count reported, never zeroed"
                ),
                "oracle": (
                    "B4 readings cited from record (Qcal 0.7458 / Qf "
                    "0.6977), not re-run this slice"
                ),
                "relation_to_v1_v2": (
                    "v1/v2 (raw trailing windows x linear ridge, both "
                    "geometries) closed by credible negative; v3 changes "
                    "only the feature family"
                ),
            }
            if V3_MODE else None
        ),
        "budgets": {
            "llm_calls": 0,
            "llm_budget": LLM_BUDGET,
            "forecasting_retrains": kw["retrains"],
            "forecasting_retrain_budget": FORECASTING_RETRAIN_BUDGET,
            "ad_evaluations": kw["evaluations"],
            "ad_evaluation_budget": AD_EVALUATION_BUDGET,
            "ad_evaluation_counting": (
                "each classifier fit counts once; each per-series Query "
                "scoring counts once; each oracle detect counts once"
            ),
            "t1b_cumulative_cap": T1B_CUMULATIVE_CAP,
            "t1b_ad_evaluations_before_v3": (
                T1B_AD_EVALUATIONS_BEFORE_V3 if V3_MODE else None
            ),
            "t1b_ad_evaluations_cumulative": (
                T1B_AD_EVALUATIONS_BEFORE_V3 + kw["evaluations"]
                if V3_MODE else None
            ),
        },
        "budgets_respected": (
            kw["retrains"] <= FORECASTING_RETRAIN_BUDGET
            and kw["evaluations"] <= AD_EVALUATION_BUDGET
            and (
                not V3_MODE
                or T1B_AD_EVALUATIONS_BEFORE_V3 + kw["evaluations"]
                <= T1B_CUMULATIVE_CAP
            )
        ),
        "roster": {
            "train": names_meta["train"],
            "eval": names_meta["eval"],
            "source": names_meta["source"],
        },
        "geometry": {
            "training_block": [TRAIN_BLOCK[0], TRAIN_BLOCK[1]],
            "qf_region": [QF_REGION[0], QF_REGION[1]],
            "qcal_region": [QCAL_REGION[0], QCAL_REGION[1]],
            "task_spans_avoided": [list(s) for s in TASK_SPANS_TO_AVOID],
            "dev_limit": DEV_LIMIT,
        },
        "part0_checkpoint": dict(PART0_CHECKPOINT[MODE]),
        "part_a_consumer": {
            "spec": adt.spec(),
            "feature_window_used": kw["window"],
            "fallback_taken": kw["fallback_taken"],
            "readability_gate": {
                "rule": (
                    "identity(B)-trained classifier, %s event F1 on Qcal "
                    ">= %.2f; the formal Query never participates" % (
                        "macro-averaged per-series" if V2_MODE else "pooled",
                        READABILITY_GATE_F1,
                    )
                ),
                "gate_metric": GATE_METRIC,
                "pooled_f1": (
                    kw["gate"]["pooled_f1"]
                    if "pooled_f1" in kw["gate"]
                    else kw["gate"].get("primary", {}).get("pooled_f1")
                ),
                "macro_f1": (
                    kw["gate"]["macro_f1"]
                    if "macro_f1" in kw["gate"]
                    else kw["gate"].get("primary", {}).get("macro_f1")
                ),
                "fallback_pooled_f1": (
                    kw["gate"].get("fallback", {}).get("pooled_f1")
                    if isinstance(kw["gate"], Mapping) else None
                ),
                "fallback_macro_f1": (
                    kw["gate"].get("fallback", {}).get("macro_f1")
                    if isinstance(kw["gate"], Mapping) else None
                ),
                "passed": True,
            },
            "training_exclusions": kw["gate"].get("training_exclusions"),
            "gate_query_undefined_points_total": kw["gate"].get(
                "undefined_points_total"
            ),
            "gate_query_undefined_attribution": kw["gate"].get(
                "undefined_attribution_total"
            ),
        },
        "part_b_queries": {
            name: {
                "region": kw[name]["protocol"]["block"],
                "protocol": kw[name]["protocol"],
                "ledger_summary": {
                    "events_per_series": kw[name]["events_per_series"],
                    "total_events": sum(
                        len(v) for v in kw[name]["ledger"].values()
                    ),
                    "total_skips": sum(
                        len(v) for v in kw[name]["skips"].values()
                    ),
                    "written_to": kw[name]["written_to"],
                },
                "history_isolation": kw[name].get("history_isolation"),
                "integrity": (
                    kw["integrity_qf"] if name == "qf" else kw["integrity_qcal"]
                ),
            }
            for name in ("qf", "qcal")
        },
        "part_b4_oracle_smoke": kw["oracle"],
        "part_c": {
            "c2_same_byte": kw["assertion"],
            "pbuffers": {
                "program_calls": kw["built"]["program_calls"],
                "reproducible_on_recall": kw["built"]["reproducible_on_recall"],
                "construction": (
                    "t1.build_pbuffers on the read-only T1 injected copies; "
                    "one P(B) per (series, program), fed to both Consumers"
                ),
            },
            "b3_query_never_processed": kw["query_never_processed"],
            "c3_guard": kw["guard"],
            "forecasting_source": {
                "source": kw["fore"]["source"],
                "retrains": kw["fore"]["retrains"],
                "consistent_with_recorded": kw["fore"]["consistent_with_recorded"],
                "cross_check": kw["fore"]["cross_check_vs_recorded_c1"],
            },
            "ad_arms": {
                program: {
                    "pooled_f1": kw["arm_ad"][program]["pooled_f1"],
                    "macro_f1": kw["arm_ad"][program]["macro_f1"],
                    "f1_by_series": kw["arm_ad"][program]["f1_by_series"],
                    "auprc_by_series": kw["arm_ad"][program]["auprc_by_series"],
                    "auprc_mean": kw["arm_ad"][program]["auprc_mean"],
                    "undefined_points_total": kw["arm_ad"][program][
                        "undefined_points_total"
                    ],
                    "undefined_points_by_series": kw["arm_ad"][program][
                        "undefined_points_by_series"
                    ],
                    "model": kw["arm_ad"][program]["model"],
                }
                for program in PROGRAMS
            },
        },
        "part_d": {
            "c5_mechanism_note": (
                "mandatory for the negative reading: repair restores the "
                "positive-label positions' features to normal shape, so the "
                "classifier loses its separating signal and Query detection "
                "degrades; if detection does NOT degrade, that is a credible "
                "negative -- no re-injection, no program swap, no re-roll"
            ),
            "consumer_family_calibration_notes_v3": (
                [
                    "the AD Consumer is a threshold head learned on the "
                    "task-native sufficient statistic (a learnable robust-z "
                    "in effect); a flip verdict speaks only for this "
                    "Consumer family",
                    "this result only proves that the training-data utility "
                    "flip is readable by an instrument when the task-native "
                    "sufficient statistic is visible -- it does not prove "
                    "the Harness discovered that representation by itself, "
                    "and it does not claim generalization to natural "
                    "anomaly data",
                ]
                if V3_MODE else None
            ),
            "quantization_note": (
                (
                    "Qf carries 48 events (12 series x 4); with the macro "
                    "average, one event changing hands moves one series' "
                    "recall by 1/4 and the aggregate by roughly 0.02, so the "
                    "+-0.005 line means 'at least one event changes hands', "
                    "not 0.005-level resolution"
                )
                if V2_MODE else (
                    "Qf carries 48 events; one event moving hands shifts the "
                    "pooled aggregate by about 0.02, so the +-0.005 line means "
                    "'at least one event changes hands', not 0.005-level "
                    "resolution"
                )
            ),
            "flip_check_per_program": flips,
            "flips_found": found,
            "forecasting_delayed_per_series": {
                program: kw["fore"]["delayed_gains"][program][
                    "per_eval_series_gain"
                ]
                for program in PROGRAMS
            },
            "ad_gain_per_series": {
                program: (
                    None
                    if kw["arm_ad"]["identity"]["pooled_f1"] is None
                    else {
                        uid: (
                            None
                            if kw["arm_ad"][program]["f1_by_series"][uid] is None
                            or kw["arm_ad"]["identity"]["f1_by_series"][uid] is None
                            else float(kw["arm_ad"][program]["f1_by_series"][uid])
                            - float(kw["arm_ad"]["identity"]["f1_by_series"][uid])
                        )
                        for uid in kw["names_meta"]["train"]
                    }
                )
                for program in PROGRAMS
            },
        },
        "integrity_post_run": {
            "originals_unchanged": kw["originals_unchanged"],
            "t1_copies_unchanged": kw["guard"]["t1_copy_shas_unchanged"],
            "query_copies_unchanged": kw["query_copies_unchanged"],
        },
        "frozen_surface": {
            "name": "FROZEN_SURFACE_V9 (post-T2 registry)",
            "raw_entries": len(list(FROZEN_SURFACE_V9)),
            "unique_files": len(set(FROZEN_SURFACE_V9)),
            "verify_after_run": kw["frozen_after"],
        },
        "git_status_before": kw["git_before"],
        "sealed_discipline": (
            "NOAA 2025 / beyond_17520 / SMD test+labels: zero reads; robust-z "
            "(49/3.5) served as injection-visibility oracle only, never as the "
            "main AD Consumer"
        ),
        "commit_discipline": (
            "deliverables not committed (the Part 0 checkpoint excepted); no "
            "spawn; the other line untouched"
        ),
        "ambiguities_reported_not_self_adjudicated": _ambiguities(),
        "wall_seconds": time.perf_counter() - kw["started"],
    }
    return payload


def _ambiguities() -> list[str]:
    notes = [
        "the cycle counter restarts from slot 0 per Query region (the T1 "
        "convention: each freeze is its own seeded draw); the T1b book "
        "does not name the counter rule, only the seeds",
        "Query features for the first scored points read pristine bytes "
        "preceding the region (up to 48 back for window 49) -- the same "
        "trailing geometry the T0 detector uses, and never P-processed "
        "bytes; the v2 isolation block asserts the 168-step pre-region "
        "prefix byte-equal to pristine in every scored copy",
    ]
    if V3_MODE:
        notes += [
            "the v2-line metric ruling holds: the gate and the judgment "
            "read the macro average of per-series event F1; the pooled F1 "
            "is kept as a secondary reading alongside",
            "the A3 gate reads the same macro average on Qcal; a "
            "per-series reading of the gate was not pre-registered",
            "'168-step history isolation' is implemented as (a) each "
            "injected event's sigma prefix recomputed from pristine bytes "
            "and compared to the ledger, and (b) the 168-step pre-region "
            "prefix of every scored copy asserted byte-equal to pristine; "
            "if the main line meant a different isolation, the assertion "
            "block is the single place to adjust",
            "the z feature for the training block reads no bytes before "
            "the block (detect's warm-up starts at the block's own index "
            "0, so training rows begin at block index 49), while Query "
            "features read the 49 pristine pre-region bytes -- the same "
            "training/query asymmetry canon as v1/v2",
            "the detect() threshold 3.5 is passed explicitly per the book "
            "but is inert for the feature path (only ['scores'] is used, "
            "never ['flags']); it is recorded for provenance",
        ]
    elif V2_MODE:
        notes += [
            "the v2 gate and judgment read the macro average of per-series "
            "event F1 (main-line ruling); the pooled F1 is kept as a "
            "secondary reading alongside, and both are in the artifact",
            "the A3 gate reads the same macro average on Qcal; a "
            "per-series reading of the gate was not pre-registered",
            "'168-step history isolation' is implemented as (a) each "
            "injected event's sigma prefix recomputed from pristine bytes "
            "and compared to the ledger, and (b) the 168-step pre-region "
            "prefix of every scored copy asserted byte-equal to pristine; "
            "if the main line meant a different isolation, the assertion "
            "block is the single place to adjust",
        ]
    else:
        notes += [
            "the aggregate AD gain is the pooled event-F1 difference over "
            "Qf's 48 events, matching the book's quantization note; the "
            "per-series F1 differences are reported alongside",
            "the A3 gate reads pooled Qcal F1 (the 48-event granularity), "
            "consistent with the aggregate-layer judgment; a per-series "
            "reading of the gate was not pre-registered",
        ]
    return notes


def _write(payload: Mapping[str, Any]) -> int:
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict:", payload["verdict"], flush=True)
    return 0


# =========================================================================== #
# pre-registered stops still deliver: the artifact records the verdict and
# every stage measured before the block
# =========================================================================== #
def _blocked_payload(
    stop: _Blocked, completed: Mapping[str, Any]
) -> dict[str, Any]:
    evaluations = int(completed.get("evaluations", 0))
    retrains = int(completed.get("retrains", 0))
    names_meta = completed.get("names_meta") or t0.roster()
    gate = completed.get("gate")
    gate_fallback = completed.get("gate_fallback")

    # post-run integrity is cheap and deterministic even on a stopped run
    originals_unchanged = None
    if "pristine_shas" in completed:
        current = t0.load_pristine(list(names_meta["all"]))
        originals_unchanged = all(
            _array_sha(current[station]) == completed["pristine_shas"][station]
            for station in list(names_meta["all"])
        )
    t1_copies_unchanged = None
    if "t1_shas_before" in completed:
        t1_copies_unchanged = _dir_shas(T1_DIR) == completed["t1_shas_before"]
    query_copies_unchanged = None
    if "qf_shas" in completed:
        query_copies_unchanged = (
            _dir_shas(QUERY_DIR / "qf") == completed["qf_shas"]
            and _dir_shas(QUERY_DIR / "qcal") == completed["qcal_shas"]
        )
    frozen_after = (
        _verify(completed["frozen_before"]) if "frozen_before" in completed else None
    )

    if "qf" in completed:
        queries_block: Any = {
            name: {
                "region": completed[name]["protocol"]["block"],
                "protocol": completed[name]["protocol"],
                "ledger_summary": {
                    "events_per_series": completed[name]["events_per_series"],
                    "total_events": sum(
                        len(v) for v in completed[name]["ledger"].values()
                    ),
                    "total_skips": sum(
                        len(v) for v in completed[name]["skips"].values()
                    ),
                    "written_to": completed[name]["written_to"],
                },
                "history_isolation": completed[name].get("history_isolation"),
                "integrity": (
                    completed["integrity_qf"]
                    if name == "qf" else completed["integrity_qcal"]
                ),
            }
            for name in ("qf", "qcal")
        }
    else:
        queries_block = "not_reached"

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "T1b: one P(training block), two *trained* Consumers, both scored "
            "on a fixed independent unprocessed Query -- is the training "
            "data's utility itself task-flipped?"
        ),
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_grade_note": (
            "permanent: an injected flip can be constructive, so this slice "
            "can never become transfer evidence by later reinterpretation"
        ),
        "verdict": stop.verdict,
        "verdict_evidence_grade": EVIDENCE_GRADE,
        "mode": MODE,
        "blocked_reason": stop.reason,
        "stopped_at_stage": completed.get("stage", "before_part_b"),
        "budgets": {
            "llm_calls": 0,
            "llm_budget": LLM_BUDGET,
            "forecasting_retrains": retrains,
            "forecasting_retrain_budget": FORECASTING_RETRAIN_BUDGET,
            "ad_evaluations": evaluations,
            "ad_evaluation_budget": AD_EVALUATION_BUDGET,
            "ad_evaluation_counting": (
                "each classifier fit counts once; each per-series Query "
                "scoring counts once; each oracle detect counts once"
            ),
            "t1b_cumulative_cap": T1B_CUMULATIVE_CAP,
            "t1b_ad_evaluations_before_v3": (
                T1B_AD_EVALUATIONS_BEFORE_V3 if V3_MODE else None
            ),
            "t1b_ad_evaluations_cumulative": (
                T1B_AD_EVALUATIONS_BEFORE_V3 + evaluations if V3_MODE else None
            ),
        },
        "budgets_respected": (
            retrains <= FORECASTING_RETRAIN_BUDGET
            and evaluations <= AD_EVALUATION_BUDGET
            and (
                not V3_MODE
                or T1B_AD_EVALUATIONS_BEFORE_V3 + evaluations
                <= T1B_CUMULATIVE_CAP
            )
        ),
        "roster": {
            "train": names_meta["train"],
            "eval": names_meta["eval"],
            "source": names_meta["source"],
        },
        "geometry": {
            "training_block": [TRAIN_BLOCK[0], TRAIN_BLOCK[1]],
            "qf_region": [QF_REGION[0], QF_REGION[1]],
            "qcal_region": [QCAL_REGION[0], QCAL_REGION[1]],
            "task_spans_avoided": [list(s) for s in TASK_SPANS_TO_AVOID],
            "dev_limit": DEV_LIMIT,
        },
        "part0_checkpoint": dict(PART0_CHECKPOINT[MODE]),
        "part_a_consumer": {
            "spec": adt.spec(),
            "feature_window_used": completed.get("window"),
            "fallback_taken": completed.get("fallback_taken"),
            "readability_gate": {
                "rule": (
                    "identity(B)-trained classifier, %s event F1 on Qcal "
                    ">= %.2f; the formal Query never participates" % (
                        "macro-averaged per-series" if V2_MODE else "pooled",
                        READABILITY_GATE_F1,
                    )
                ),
                "gate_metric": GATE_METRIC,
                "pooled_f1": (
                    gate["pooled_f1"] if "pooled_f1" in gate
                    else gate.get("primary", {}).get("pooled_f1")
                ) if isinstance(gate, Mapping) else None,
                "macro_f1": (
                    gate["macro_f1"] if "macro_f1" in gate
                    else gate.get("primary", {}).get("macro_f1")
                ) if isinstance(gate, Mapping) else None,
                "fallback_pooled_f1": (
                    gate_fallback.get("pooled_f1")
                    if isinstance(gate_fallback, Mapping)
                    else gate.get("fallback", {}).get("pooled_f1")
                    if isinstance(gate, Mapping) else None
                ),
                "fallback_macro_f1": (
                    gate_fallback.get("macro_f1")
                    if isinstance(gate_fallback, Mapping)
                    else gate.get("fallback", {}).get("macro_f1")
                    if isinstance(gate, Mapping) else None
                ),
                "passed": bool(completed.get("gate_passed", False)),
            },
            "training_exclusions": (
                gate.get("training_exclusions")
                if isinstance(gate, Mapping) else None
            ),
            "gate_query_undefined_points_total": (
                gate.get("undefined_points_total")
                if isinstance(gate, Mapping) else None
            ),
            "gate_query_undefined_attribution": (
                gate.get("undefined_attribution_total")
                if isinstance(gate, Mapping) else None
            ),
        },
        "part_b_queries": queries_block,
        "part_b4_oracle_smoke": completed.get("oracle", "not_reached"),
        "part_c": {
            "c2_same_byte": completed.get("assertion", "not_reached"),
            "pbuffers": completed.get("built_meta", "not_reached"),
            "b3_query_never_processed": completed.get(
                "query_never_processed", "not_reached"
            ),
            "ad_arms": (
                "not_reached"
                if "arm_ad" not in completed
                else {
                    program: {
                        "pooled_f1": completed["arm_ad"][program]["pooled_f1"],
                        "f1_by_series": completed["arm_ad"][program][
                            "f1_by_series"
                        ],
                        "auprc_mean": completed["arm_ad"][program]["auprc_mean"],
                    }
                    for program in completed["arm_ad"]
                }
            ),
            "forecasting_source": "not_reached"
            if "fore" not in completed
            else completed["fore"],
        },
        "part_d": (
            "not_reached -- the run stopped at %s before the aggregate-layer "
            "judgment; no flip claim is made or implied"
            % completed.get("stage", "before_part_b")
        ),
        "integrity_post_run": {
            "originals_unchanged": originals_unchanged,
            "t1_copies_unchanged": t1_copies_unchanged,
            "query_copies_unchanged": query_copies_unchanged,
        },
        "frozen_surface": {
            "name": "FROZEN_SURFACE_V9 (post-T2 registry)",
            "raw_entries": len(list(FROZEN_SURFACE_V9)),
            "unique_files": len(set(FROZEN_SURFACE_V9)),
            "verify_after_run": frozen_after,
        },
        "git_status_before": completed.get("git_before"),
        "sealed_discipline": (
            "NOAA 2025 / beyond_17520 / SMD test+labels: zero reads; robust-z "
            "(49/3.5) served as injection-visibility oracle only, never as the "
            "main AD Consumer"
        ),
        "commit_discipline": (
            "deliverables not committed (the Part 0 checkpoint excepted); no "
            "spawn; the other line untouched"
        ),
        "ambiguities_reported_not_self_adjudicated": _ambiguities(),
        "wall_seconds": time.perf_counter() - float(
            completed.get("started", time.perf_counter())
        ),
    }
    return payload


def _blocked_markdown(payload: Mapping[str, Any]) -> str:
    gate = payload["part_a_consumer"]["readability_gate"]
    lines = [
        "# T1b -- training-side task flip (POSITIVE_CONTROL)",
        "",
        "- verdict: **%s** (pre-registered stop; the first block ends the run)"
        % payload["verdict"],
        "- reason: %s" % payload["blocked_reason"],
        "- stopped at stage: `%s`" % payload["stopped_at_stage"],
        "- evidence grade: %s (permanent)" % payload["evidence_grade"],
        "- LLM %d/%d, forecasting retrains %d/%d, AD evaluations %d/%d" % (
            payload["budgets"]["llm_calls"], payload["budgets"]["llm_budget"],
            payload["budgets"]["forecasting_retrains"],
            payload["budgets"]["forecasting_retrain_budget"],
            payload["budgets"]["ad_evaluations"],
            payload["budgets"]["ad_evaluation_budget"],
        ),
        "- Part 0 checkpoint: `%s` (%d files)" % (
            payload["part0_checkpoint"]["commit"],
            payload["part0_checkpoint"]["files"],
        ),
        "",
        "## Part A -- the A3 readability gate",
        "",
        "- instrument: `%s`" % payload["part_a_consumer"]["spec"]["sited_at"],
        "- rule: %s" % gate["rule"],
        "- gate metric: %s; primary window: %s (pooled %s); fallback window: "
        "%s (pooled %s); passed: %s" % (
            gate["gate_metric"],
            gate[gate["gate_metric"]],
            gate["pooled_f1"],
            (
                gate["fallback_macro_f1"]
                if gate["gate_metric"] == "macro_f1"
                else gate["fallback_pooled_f1"]
            ),
            gate["fallback_pooled_f1"],
            gate["passed"],
        ),
    ]
    if payload.get("mode") == "v3":
        lines.append(
            "- v3 is single-shot (no fallback); training exclusions "
            "(warm_up / zero_scale / non_finite): %s" % json.dumps(
                payload["part_a_consumer"].get("training_exclusions"),
                ensure_ascii=False,
            )
        )
    lines += [
        "",
        "## Part B -- the two Query regions",
        "",
    ]
    queries = payload["part_b_queries"]
    if isinstance(queries, Mapping):
        for name in ("qf", "qcal"):
            block = queries[name]["protocol"]["block"]
            summary = queries[name]["ledger_summary"]
            lines.append(
                "- %s [%d, %d): %d events/series, %d total, %d skips; ledger "
                "frozen before any training or scoring" % (
                    name.upper(), block["start"], block["end"],
                    summary["events_per_series"], summary["total_events"],
                    summary["total_skips"],
                )
            )
    else:
        lines.append("- not reached")
    oracle = payload["part_b4_oracle_smoke"]
    if isinstance(oracle, Mapping):
        qf_f1 = oracle["qf"]["pooled"]["f1"] if oracle.get("qf") else None
        qcal_f1 = oracle["qcal"]["pooled"]["f1"] if oracle.get("qcal") else None
        lines.append(
            "- B4 oracle smoke (robust-z 49/3.5, visibility only): Qf pooled "
            "F1 %s, Qcal pooled F1 %s%s" % (
                qf_f1, qcal_f1,
                ""
                if qf_f1 is not None
                else " (Qf never read -- the run stopped before the gate "
                     "released it)",
            )
        )
    lines += [
        "",
        "## Part C / Part D",
        "",
        "- C2 same-byte: %s" % json.dumps(
            payload["part_c"]["c2_same_byte"], ensure_ascii=False
        ),
        "- B3 query-never-processed: %s" % json.dumps(
            payload["part_c"]["b3_query_never_processed"], ensure_ascii=False
        ),
        "- AD arms: %s"
        % (
            "not reached"
            if payload["part_c"]["ad_arms"] == "not_reached"
            else "partial, see JSON"
        ),
        "- Part D: %s" % payload["part_d"],
        "",
        "## Discipline",
        "",
        "- %s" % payload["sealed_discipline"],
        "- %s" % payload["commit_discipline"],
        "- originals unchanged post-run: %s; T1 copies unchanged: %s; Query "
        "copies unchanged: %s" % (
            payload["integrity_post_run"]["originals_unchanged"],
            payload["integrity_post_run"]["t1_copies_unchanged"],
            payload["integrity_post_run"]["query_copies_unchanged"],
        ),
        "",
        "## Ambiguities (reported, not self-adjudicated)",
        "",
    ]
    for note in payload["ambiguities_reported_not_self_adjudicated"]:
        lines.append("- %s" % note)
    lines.append("")
    return "\n".join(lines)


def _write_blocked(payload: Mapping[str, Any]) -> int:
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_blocked_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict:", payload["verdict"], flush=True)
    return 2


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "%+.4f" % value
    return str(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# T1b -- training-side task flip (POSITIVE_CONTROL)",
        "",
        "- verdict: **%s**" % payload["verdict"],
        "- evidence grade: %s (permanent)" % payload["evidence_grade"],
        "- LLM %d/%d, forecasting retrains %d/%d, AD evaluations %d/%d" % (
            payload["budgets"]["llm_calls"], payload["budgets"]["llm_budget"],
            payload["budgets"]["forecasting_retrains"],
            payload["budgets"]["forecasting_retrain_budget"],
            payload["budgets"]["ad_evaluations"],
            payload["budgets"]["ad_evaluation_budget"],
        ),
        "- Part 0 checkpoint: `%s` (%d files)" % (
            payload["part0_checkpoint"]["commit"],
            payload["part0_checkpoint"]["files"],
        ),
        "",
    ]
    if payload.get("mode") == "v2" and payload.get("execution_conventions_v2"):
        conv = payload["execution_conventions_v2"]
        lines += [
            "## v2 execution conventions (main-line ruling)",
            "",
            "- feature geometry: %s" % conv["feature_geometry"],
            "- ordering: %s" % conv["ordering"],
            "- history isolation: %s" % conv["history_isolation_168"],
            "- metric: %s" % conv["metric"],
            "- relation to v1: %s" % conv["relation_to_v1"],
            "",
        ]
    if payload.get("mode") == "v3" and payload.get("execution_conventions_v3"):
        conv = payload["execution_conventions_v3"]
        lines += [
            "## v3 conventions (same book, continued)",
            "",
            "- feature family: %s" % conv["feature_family"],
            "- single shot: %s" % conv["single_shot"],
            "- abstention: %s" % conv["abstention"],
            "- oracle: %s" % conv["oracle"],
            "- relation to v1/v2: %s" % conv["relation_to_v1_v2"],
            "",
        ]
    if (
        payload.get("mode") == "v3"
        and payload["verdict"] == "TRAINING_SIDE_TASK_FLIP_CONFIRMED_POSITIVE_CONTROL"
        and payload["part_d"].get("consumer_family_calibration_notes_v3")
    ):
        lines += [
            "## Mandatory calibration notes carried by the confirmed verdict",
            "",
        ]
        for note in payload["part_d"]["consumer_family_calibration_notes_v3"]:
            lines.append("- %s" % note)
        lines.append("")
    lines += [
        "## Part A -- the trainable AD Consumer",
        "",
        "- instrument: `%s`" % payload["part_a_consumer"]["spec"]["sited_at"],
        "- feature window used: %d%s" % (
            payload["part_a_consumer"]["feature_window_used"],
            " (fallback taken)" if payload["part_a_consumer"]["fallback_taken"] else "",
        ),
        "- A3 readability gate: %s" % json.dumps(
            payload["part_a_consumer"]["readability_gate"], ensure_ascii=False
        ),
        "",
        "## Part B -- the two Query regions",
        "",
    ]
    for name in ("qf", "qcal"):
        block = payload["part_b_queries"][name]["protocol"]["block"]
        summary = payload["part_b_queries"][name]["ledger_summary"]
        lines.append(
            "- %s [%d, %d): %d events/series, %d total, %d skips; ledger frozen "
            "before any training or scoring" % (
                name.upper(), block["start"], block["end"],
                summary["events_per_series"], summary["total_events"],
                summary["total_skips"],
            )
        )
    lines += [
        "",
        "B4 oracle smoke (robust-z 49/3.5, injection visibility only, never a "
        "verdict input): Qf pooled F1 %s, Qcal pooled F1 %s%s" % (
            payload["part_b4_oracle_smoke"]["qf"]["pooled"]["f1"],
            payload["part_b4_oracle_smoke"]["qcal"]["pooled"]["f1"],
            " (cited from record, not re-run this slice)"
            if payload.get("mode") == "v3" else "",
        ),
        "",
        "## Part C -- both Consumers trained on the same P(B)",
        "",
        "- C2 same-byte: %d comparisons, all_equal=%s, reproducible=%s" % (
            payload["part_c"]["c2_same_byte"]["comparisons"],
            payload["part_c"]["c2_same_byte"]["all_equal"],
            payload["part_c"]["pbuffers"]["reproducible_on_recall"],
        ),
        "- B3 query-never-processed: %s" % payload["part_c"]["b3_query_never_processed"]["holds"],
        "- C3 guard: %s" % json.dumps(payload["part_c"]["c3_guard"], ensure_ascii=False),
        "- forecasting readings: %s (retrains %d)" % (
            payload["part_c"]["forecasting_source"]["source"],
            payload["part_c"]["forecasting_source"]["retrains"],
        ),
        "",
        "| program | forecasting delayed agg | AD macro F1 | AD pooled F1 | AD train gain | mean AUPRC | undef pts | flip |",
        "|---|---|---|---|---|---|---|---|",
    ]
    arms = payload["part_c"]["ad_arms"]
    flips = {row["program"]: row for row in payload["part_d"]["flip_check_per_program"]}
    for program in sorted(arms):
        row = flips.get(program, {})
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            program,
            _fmt(row.get("forecasting_delayed_aggregate_gain")),
            _fmt(arms[program]["macro_f1"]),
            _fmt(arms[program]["pooled_f1"]),
            _fmt(row.get("ad_train_gain_pooled")),
            _fmt(arms[program]["auprc_mean"]),
            arms[program]["undefined_points_total"],
            row.get("flip_direction") or "—",
        ))
    lines += [
        "",
        "Per-series vectors (forecasting delayed and AD gain) are in the JSON "
        "under `part_d`.",
        "",
        "## Discipline",
        "",
        "- %s" % payload["sealed_discipline"],
        "- %s" % payload["commit_discipline"],
        "- originals unchanged post-run: %s; T1 copies unchanged: %s; Query "
        "copies unchanged: %s" % (
            payload["integrity_post_run"]["originals_unchanged"],
            payload["integrity_post_run"]["t1_copies_unchanged"],
            payload["integrity_post_run"]["query_copies_unchanged"],
        ),
        "",
        "## Ambiguities (reported, not self-adjudicated)",
        "",
    ]
    for note in payload["ambiguities_reported_not_self_adjudicated"]:
        lines.append("- %s" % note)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    try:
        return run()
    except _Blocked as stop:
        print("BLOCKED %s: %s" % (stop.verdict, stop.reason), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
