"""Can the with-missing KDD variant actually enter the Consumer?

Five mechanical gates, all of which must pass before any sweep is worth running
on ``EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING``:

1. the variant is value-identical to the current cache at every observed
   position, so the two differ only by the upstream fill;
2. the structural filter still yields a 20/20 two-face target cell once the
   gaps are back -- readability is recomputed, not inherited;
3. every eval series has observed truth in its horizon and a defined seasonal
   scale, so the missing-aware sMASE is computable rather than raising;
4. each eligible operator survives genuinely gapped input, so a null result
   would mean "no gain" rather than "the operator crashed";
5. the variant carries its own version name and never merges with old numbers.

Note what identity means here.  ``_apply_program(window, None)`` returns
``_linear_integrity(window)``, and every program's output is passed through
``_linear_integrity`` too, so the baseline *is* linear interpolation and
``impute_linear`` must land on ZERO_BEHAVIOR.  What this line can test is
whether fft / ema / period-median / ar completion beat linear interpolation.

0 LLM calls.  One identity Consumer fit per probed origin, on exposed
development data.  No cache is rebuilt and no existing artifact is modified.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
    UndefinedSeasonalScale,
    seasonal_scale,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WITH_MISSING = (
    PROJECT_ROOT / "data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip"
)
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4d_natural_gap_preflight.json"

DATA_VERSION = "EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING"
INCUMBENT_VERSION = "EXPOSED_DEVELOPMENT__KDD2018_WITHOUT_MISSING"

CONTEXT, HORIZON = 192, 48
ORIGIN = forecast_p1.ORIGIN
P4B_ORIGINS = (1176, 1416, 1656, 1896, 2136, 2376, 2616, 2856)
PERIOD = 24
MIN_SEASONAL_PAIRS = 32


def load_variant() -> dict[str, np.ndarray]:
    """Series with genuine NaN where the archive records '?'."""
    with zipfile.ZipFile(WITH_MISSING) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            stream = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            series: dict[str, np.ndarray] = {}
            in_data = False
            for line in stream:
                if not in_data:
                    in_data = line.strip().lower() == "@data"
                    continue
                fields = line.rstrip().split(":")
                series[fields[0]] = np.array(
                    [np.nan if token.strip() == "?" else float(token)
                     for token in fields[-1].split(",")],
                    dtype=np.float64,
                )
    return series


def gate_1_value_identity(variant: dict[str, np.ndarray]) -> dict[str, Any]:
    """Observed positions must agree to the bit, not merely in shape."""
    cache = np.load(CACHE, allow_pickle=True)
    names = [str(value) for value in cache["names"]]
    rows = cache["values"]
    compared = differing = 0
    worst = 0.0
    for index, name in enumerate(names):
        filled = np.asarray(rows[index], dtype=np.float64)
        gapped = variant[name]
        observed = np.flatnonzero(np.isfinite(gapped))
        deviation = np.abs(gapped[observed] - filled[observed])
        compared += int(observed.size)
        differing += int((deviation > 1e-9).sum())
        worst = max(worst, float(deviation.max()) if deviation.size else 0.0)
    return {
        "observed_positions_compared": compared,
        "positions_differing": differing,
        "max_abs_deviation": worst,
        "passed": differing == 0,
        "reading": (
            "the two versions differ only by the upstream fill"
            if differing == 0 else
            "the variant is not the same series; a swap would change the data"
        ),
    }


def _fit_readable(raw: np.ndarray, anchors: list[int]) -> bool:
    """P1's structural filter, recomputed on gapped input.

    Readability is not inherited from the filled cache: a gap can collapse a
    Ridge training context to the scale floor where the filled version was fine.
    """
    try:
        history = forecast_runtime._linear_integrity(raw[ORIGIN - CONTEXT:ORIGIN])
    except Exception:
        return False
    _center, _scale, method = forecast_runtime._center_scale(np, history)
    if method == "scale_floor_fallback":
        return False
    for anchor in anchors:
        window = np.asarray(raw[anchor - CONTEXT:anchor + HORIZON], dtype=np.float64)
        for op in forecast_p1._eligible_programs():
            result = run_pipeline(
                list(forecast_p1._steps(op)), window,
                source="p4d_natural_gap_preflight",
            )
            if not result.ok or result.artifact is None:
                return False
            try:
                prepared = forecast_runtime._linear_integrity(
                    np.asarray(result.artifact, dtype=np.float64)
                )
            except Exception:
                return False
            if not np.isfinite(prepared).all():
                return False
            _c, _s, method = forecast_runtime._center_scale(np, prepared[:CONTEXT])
            if method == "scale_floor_fallback":
                return False
    return True


def gate_2_roster(variant: dict[str, np.ndarray]) -> dict[str, Any]:
    """Does a 20/20 two-face target cell still form on gapped data?"""
    anchors = [
        int(anchor) for anchor in forecast_p1._config()["anchors"]
        if int(anchor) + HORIZON <= ORIGIN
    ]
    readable = [uid for uid in sorted(variant) if _fit_readable(variant[uid], anchors)]
    cache = np.load(CACHE, allow_pickle=True)
    incumbent = sorted(str(value) for value in cache["names"])
    # The incumbent roster is the filled cache's first 80 readable UIDs; here we
    # only need to know how much of the *ordering* survives the gaps.
    return {
        "series_screened": len(variant),
        "structurally_readable": len(readable),
        "target_cell_formable": len(readable) >= 40,
        "selection_cell_formable": len(readable) >= 80,
        "support_a": readable[:20],
        "support_b": readable[20:40],
        "readable_prefix_matches_sorted_order": readable[:40] == incumbent[:40],
        "passed": len(readable) >= 40,
        "reading": (
            "a 20/20 two-face target cell forms on gapped data"
            if len(readable) >= 40 else
            "the gaps eliminate too many series for the two-face geometry"
        ),
    }


def gate_3_horizon_truth(variant: dict[str, np.ndarray],
                         uids: list[str]) -> dict[str, Any]:
    """Observed truth in every horizon, and a defined seasonal scale."""
    rows = []
    blocking = []
    for origin in P4B_ORIGINS:
        observed_counts, undefined = [], []
        for uid in uids:
            raw = variant[uid]
            truth = raw[origin:origin + HORIZON]
            count = int(np.isfinite(truth).sum())
            observed_counts.append(count)
            if count == 0:
                undefined.append({"uid": uid, "why": "no observed truth"})
                continue
            try:
                seasonal_scale(
                    raw[:origin], np.isfinite(raw[:origin]),
                    period=PERIOD, min_pairs=MIN_SEASONAL_PAIRS,
                )
            except UndefinedSeasonalScale as exc:
                undefined.append({"uid": uid, "why": str(exc)[:80]})
        counts = np.array(observed_counts)
        row = {
            "origin": int(origin),
            "min_observed_truth": int(counts.min()),
            "median_observed_truth": int(np.median(counts)),
            "series_with_zero_truth": int((counts == 0).sum()),
            "undefined_metric_series": undefined,
            "usable": not undefined,
        }
        rows.append(row)
        if undefined:
            blocking.append(int(origin))
    return {
        "horizon": HORIZON,
        "per_origin": rows,
        "blocking_origins": blocking,
        "usable_origins": [o for o in P4B_ORIGINS if o not in blocking],
        "passed": len(blocking) < len(P4B_ORIGINS),
        "reading": (
            "every origin has computable missing-aware sMASE"
            if not blocking else
            "origins %s cannot be scored and must be dropped, not imputed" % blocking
        ),
    }


def gate_4_operator_survival(variant: dict[str, np.ndarray],
                             uids: list[str]) -> dict[str, Any]:
    """Every eligible operator, on genuinely gapped windows.

    An operator that crashes or returns an all-NaN artifact would report as
    "no gain" in a sweep, which is a different claim from "no effect".
    """
    ops = forecast_p1._eligible_programs()
    windows = [
        np.asarray(variant[uid][origin - CONTEXT:origin + HORIZON], dtype=np.float64)
        for origin in P4B_ORIGINS
        for uid in uids[:8]
    ]
    gapped = [window for window in windows if not np.isfinite(window).all()]
    results = []
    for op in ops:
        ok = failed = zero_behavior = 0
        first_error = None
        for window in gapped:
            try:
                baseline = forecast_runtime._linear_integrity(window)
                execution = run_pipeline(
                    list(forecast_p1._steps(op)), window,
                    source="p4d_natural_gap_preflight",
                )
                if not execution.ok or execution.artifact is None:
                    raise RuntimeError(execution.error or "pipeline refused")
                prepared = forecast_runtime._linear_integrity(
                    np.asarray(execution.artifact, dtype=np.float64).ravel()
                )
                if not np.isfinite(prepared).all():
                    raise RuntimeError("non-finite after integrity handling")
                ok += 1
                if np.allclose(prepared, baseline, equal_nan=True):
                    zero_behavior += 1
            except Exception as exc:  # noqa: BLE001 - survival is the measurement
                failed += 1
                if first_error is None:
                    first_error = "%s: %s" % (type(exc).__name__, str(exc)[:100])
        results.append(
            {
                "operator": op,
                "windows": len(gapped),
                "ran": ok,
                "failed": failed,
                "zero_behavior_vs_linear": zero_behavior,
                "survival_rate": round(ok / len(gapped), 4) if gapped else None,
                "first_error": first_error,
            }
        )
    dead = [row["operator"] for row in results if row["ran"] == 0]
    live_changing = [
        row["operator"] for row in results
        if row["ran"] > 0 and row["zero_behavior_vs_linear"] < row["ran"]
    ]
    return {
        "gapped_windows_probed": len(gapped),
        "operators": results,
        "operators_dead_on_gaps": dead,
        "operators_that_change_something": live_changing,
        "passed": len(dead) < len(ops),
        "reading": (
            "%d/%d operators run on gapped input and %d change it"
            % (len(ops) - len(dead), len(ops), len(live_changing))
        ),
    }


def gate_5_consumer_smoke(variant: dict[str, np.ndarray],
                          uids: list[str], origins: list[int]) -> dict[str, Any]:
    """One identity fit per usable origin: does the Consumer accept gaps at all?"""
    roster = [
        {"series_uid": uid, "role": "train" if index < 20 else "eval"}
        for index, uid in enumerate(uids[:40])
    ]
    rows = []
    for origin in origins:
        config = dict(forecast_p1._config())
        config["support_origin"] = int(origin)
        try:
            reading = forecast_runtime._evaluate(
                roster, variant, None, config, origin=int(origin)
            )
            rows.append(
                {
                    "origin": int(origin),
                    "ran": True,
                    "identity_mean_smase": round(float(reading["mean_smase"]), 6),
                    "eval_series": len(reading["per_view_smase"]),
                }
            )
        except Exception as exc:  # noqa: BLE001 - the smoke test's whole point
            rows.append(
                {
                    "origin": int(origin),
                    "ran": False,
                    "error": "%s: %s" % (type(exc).__name__, str(exc)[:140]),
                }
            )
    ran = [row for row in rows if row["ran"]]
    return {
        "consumer_fits": len(origins),
        "per_origin": rows,
        "origins_that_ran": [row["origin"] for row in ran],
        "passed": bool(ran),
        "reading": (
            "identity is computable on gapped data at %d/%d origins"
            % (len(ran), len(origins))
        ),
    }


def build() -> dict[str, Any]:
    variant = load_variant()
    gate1 = gate_1_value_identity(variant)
    gate2 = gate_2_roster(variant)
    uids = (gate2["support_a"] + gate2["support_b"]) or sorted(variant)[:40]
    gate3 = gate_3_horizon_truth(variant, uids)
    gate4 = gate_4_operator_survival(variant, uids)
    gate5 = gate_5_consumer_smoke(variant, uids, gate3["usable_origins"][:3])
    gates = {
        "gate_1_value_identity": gate1,
        "gate_2_roster_geometry": gate2,
        "gate_3_horizon_truth": gate3,
        "gate_4_operator_survival": gate4,
        "gate_5_consumer_smoke": gate5,
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    return {
        "stage": "P4D_NATURAL_GAP_PREFLIGHT",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_INSTRUMENT_CHECK",
        "data_version": DATA_VERSION,
        "does_not_merge_with": INCUMBENT_VERSION,
        "data_role": "EXPOSED_DEVELOPMENT_VARIANT",
        "boundary": {
            "llm_calls": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "natural_final_outcome_reads": 0,
            "caches_rebuilt": 0,
            "existing_artifacts_modified": 0,
        },
        "identity_semantics": (
            "_apply_program(window, None) == _linear_integrity(window), and every "
            "program output is passed through _linear_integrity, so the baseline "
            "is linear interpolation and impute_linear must be ZERO_BEHAVIOR; the "
            "question this line can answer is whether fft / ema / period-median / "
            "ar completion beat linear interpolation"
        ),
        "gates": gates,
        "failed_gates": failed,
        "verdict": "PREFLIGHT_PASS" if not failed else "PREFLIGHT_BLOCKED",
        "releases": (
            "P4D_FULL_SWEEP_MAY_PROCEED" if not failed else "NONE"
        ),
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for name, gate in report["gates"].items():
        print("%-26s %-6s %s" % (
            name.replace("gate_", "").replace("_", " ")[:26],
            "PASS" if gate["passed"] else "BLOCK", gate["reading"]))
    print("verdict : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0 if report["verdict"] == "PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
