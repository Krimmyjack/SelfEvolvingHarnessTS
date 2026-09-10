"""DEV-TRAIN-1 numeric-kernel smoke: synthetic arrays only.

Zero LLM.  Zero real-data / Outcome reads.  At most 30 pooled Ridge fits.
Does not require that every non-identity Workflow change the coefficients.
Does not use object ids or a new SHA as freeze proof.

Run:  python -m evaluation.main_protocol_p4.smoke_dev_train1
"""
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import patch

import numpy as np

from evaluation.functional.run_batch_composition_headroom import _evaluate_assignment
from evaluation.main_protocol_p4.dev_train1_evaluator import (
    CONTEXT_LENGTH,
    HORIZON,
    DegenerateContext,
    FrozenRidge,
    MissingAssignmentError,
    PreparationFailed,
    build_training_design,
    compile_steps,
    fit_assignment,
    predict_assignment,
    score_predictions,
    _prepare_train_window,
)

MAX_FITS = 30
FITS = {"n": 0}


def _fit(design: dict[str, Any]) -> FrozenRidge:
    FITS["n"] += 1
    return fit_assignment(design)


def _old_pooled(roster, values, assignment, config, origin: int) -> dict[str, Any]:
    FITS["n"] += 1
    return _evaluate_assignment(roster, values, assignment, config, origin=int(origin))


def _series(seed: int, n: int, spikes: tuple[tuple[int, float], ...] = ()) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64)
    out = 8.0 + np.sin(2.0 * np.pi * t / 24.0) + 0.05 * rng.randn(n)
    for index, value in spikes:
        out[int(index)] = float(value)
    return out


def _synthetic() -> tuple[list, dict, dict, int]:
    n = 480
    values = {
        "t0": _series(1, n, ((70, 800.0), (90, -800.0))),
        "t1": _series(2, n),
        "e0": _series(3, n),
        "e1": _series(4, n),
    }
    roster = [
        {"series_uid": "t0", "role": "train"},
        {"series_uid": "t1", "role": "train"},
        {"series_uid": "e0", "role": "eval"},
        {"series_uid": "e1", "role": "eval"},
    ]
    return roster, values, {"anchors": (240,), "period": 24}, 312


def _check(rows: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    row = {"check": name, "passed": bool(passed)}
    if detail is not None:
        row["detail"] = detail
    rows.append(row)


def _close(a: Any, b: Any) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                            rtol=0.0, atol=1e-12, equal_nan=True))


def run() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    roster, values, config, origin = _synthetic()
    identity_train = {"t0": None, "t1": None}
    identity_eval = {"e0": None, "e1": None}
    winsor = compile_steps((("winsorize", {"limits": 0.05}),))

    # 1. heterogeneous windows match applying each program alone
    het = build_training_design(
        roster, values, {"t0": winsor, "t1": None}, config, origin=origin
    )
    solo_t0 = build_training_design(
        [{"series_uid": "t0", "role": "train"}],
        values, {"t0": winsor}, config, origin=origin,
    )
    solo_t1 = build_training_design(
        [{"series_uid": "t1", "role": "train"}],
        values, {"t1": None}, config, origin=origin,
    )
    w_t0 = [w for w in het["windows"] if w["series_uid"] == "t0"]
    w_t1 = [w for w in het["windows"] if w["series_uid"] == "t1"]
    _check(rows, "heterogeneous_windows_match_solo_apply",
           len(w_t0) == 1 and len(w_t1) == 1
           and _close(w_t0[0]["context"], solo_t0["windows"][0]["context"])
           and _close(w_t0[0]["target"], solo_t0["windows"][0]["target"])
           and _close(w_t1[0]["context"], solo_t1["windows"][0]["context"])
           and int(w_t1[0]["moved_points"]) == 0
           and int(w_t0[0]["moved_points"]) == int(solo_t0["windows"][0]["moved_points"]))

    # 2–3. all-identity and uniform program match old pooled _evaluate_assignment
    d_id = build_training_design(
        roster, values, identity_train, config, origin=origin
    )
    m_id = _fit(d_id)
    p_id = predict_assignment(m_id, roster, values, identity_eval, config, origin=origin)
    s_id = score_predictions(p_id, roster, values, config, origin=origin)
    old_id = _old_pooled(roster, values, identity_train, config, origin)
    _check(rows, "identity_matches_old_pooled_evaluate_assignment",
           s_id["complete"] is True
           and _close(s_id["per_view_smase"], old_id["per_view_smase"])
           and int(d_id["behavior_point_count"]) == 0
           and int(old_id["behavior_point_count"]) == 0,
           {"ours": s_id["per_view_smase"], "old": old_id["per_view_smase"]})

    uniform = {"t0": winsor, "t1": winsor}
    d_u = build_training_design(roster, values, uniform, config, origin=origin)
    m_u = _fit(d_u)
    p_u = predict_assignment(m_u, roster, values, identity_eval, config, origin=origin)
    s_u = score_predictions(p_u, roster, values, config, origin=origin)
    old_u = _old_pooled(roster, values, uniform, config, origin)
    _check(rows, "uniform_program_matches_old_pooled",
           s_u["complete"] is True
           and _close(s_u["per_view_smase"], old_u["per_view_smase"])
           and int(d_u["behavior_point_count"]) == int(old_u["behavior_point_count"]),
           {"ours": s_u["per_view_smase"], "old": old_u["per_view_smase"],
            "moved": d_u["behavior_point_count"]})

    # 4. fit once, predict twice, no second fit
    with patch.object(np.linalg, "solve", side_effect=AssertionError("prediction refit")):
        p_again = predict_assignment(m_id, roster, values, identity_eval, config, origin=origin)
    _check(rows, "fit_once_predict_twice_no_refit",
           int(m_id.consumer_fits) == 1
           and int(p_id["consumer_fits"]) == 0
           and int(p_again["consumer_fits"]) == 0
           and _close(p_id["predictions"], p_again["predictions"]))

    # 5. JSON-compatible restore; freeze is coefficients, not id()
    blob = json.dumps(m_u.to_dict(), allow_nan=False)
    restored = FrozenRidge.from_dict(json.loads(blob))
    p_rest = predict_assignment(restored, roster, values, identity_eval, config, origin=origin)
    _check(rows, "json_state_restore_predicts_equal",
           _close(p_u["predictions"], p_rest["predictions"])
           and restored.coefficients is not m_u.coefficients
           and int(restored.consumer_fits) == 1
           and int(p_rest["consumer_fits"]) == 0)

    # 6. true input_only W(X) differs from recombination Xp (W saw y)
    # The complete target block shifts the winsor quantiles. A single extreme
    # in a periodic fixture need not change either clipping boundary.
    y_sensitive = np.linspace(0.0, 1.0, 480)
    y_sensitive[240:288] = np.arange(1000.0, 1048.0)
    vs = dict(values)
    vs["t0"] = y_sensitive
    in_only = build_training_design(
        roster, vs, {"t0": winsor, "t1": None}, config, origin=origin,
        training_design="input_only",
    )
    rec_xp = build_training_design(
        roster, vs, {"t0": winsor, "t1": None}, config, origin=origin,
        training_design="recombination", recombination="Xp_y0",
    )
    rec_full = build_training_design(
        roster, vs, {"t0": winsor, "t1": None}, config, origin=origin,
        training_design="recombination", recombination="Xp_yp",
    )
    whole = build_training_design(
        roster, vs, {"t0": winsor, "t1": None}, config, origin=origin,
        training_design="whole_window",
    )
    x_in = [w for w in in_only["windows"] if w["series_uid"] == "t0"][0]["context"]
    x_rec = [w for w in rec_xp["windows"] if w["series_uid"] == "t0"][0]["context"]
    _check(rows, "input_only_x_differs_from_recombination_xp_y_sensitive",
           in_only["y_access"] is False
           and rec_xp["y_access"] is True
           and not _close(x_in, x_rec)
           and _close(
               [w for w in rec_full["windows"] if w["series_uid"] == "t0"][0]["context"],
               [w for w in whole["windows"] if w["series_uid"] == "t0"][0]["context"],
           )
           and _close(
               [w for w in rec_full["windows"] if w["series_uid"] == "t0"][0]["target"],
               [w for w in whole["windows"] if w["series_uid"] == "t0"][0]["target"],
           ))
    _check(rows, "recombination_is_not_no_y_access",
           rec_xp["y_access"] is True and rec_full["y_access"] is True
           and whole["y_access"] is True and in_only["y_access"] is False)
    rec_base = build_training_design(
        roster, vs, {"t0": winsor, "t1": None}, config, origin=origin,
        training_design="recombination", recombination="X0_y0",
    )
    _check(rows, "recombination_counts_retained_not_discarded_changes",
           rec_base["behavior_point_count"] == 0
           and rec_base["windows"][0]["source_workflow_moved_points"] > 0)

    # 7. missing key / failed program / short window are explicit failures
    missing_ok = False
    try:
        build_training_design(roster, values, {"t0": None}, config, origin=origin)
    except MissingAssignmentError as exc:
        missing_ok = "t1" in exc.missing and exc.extra == []
    failed_ok = False
    try:
        build_training_design(
            roster, values,
            {"t0": compile_steps((("definitely_not_an_operator", {}),)), "t1": None},
            config, origin=origin,
        )
    except PreparationFailed:
        failed_ok = True
    short_ok = False
    short_values = {uid: arr[:100].copy() for uid, arr in values.items()}
    try:
        build_training_design(roster, short_values, identity_train, config, origin=origin)
    except PreparationFailed:
        short_ok = True
    none_is_identity = build_training_design(
        roster, values, identity_train, config, origin=origin
    )["behavior_point_count"] == 0
    _check(rows, "missing_assignment_and_failed_program_do_not_become_identity",
           missing_ok and failed_ok and short_ok and none_is_identity)

    # 8. B(X) may differ from B([X,y])[:192] at a missing-value boundary
    gap = values["t1"].copy()
    # window at anchor 240 is series[48:288]; X/y cut is series index 240
    gap[238:242] = np.nan
    vg = dict(values)
    vg["t1"] = gap
    d_in = build_training_design(
        roster, vg, identity_train, config, origin=origin, training_design="input_only"
    )
    d_w = build_training_design(
        roster, vg, identity_train, config, origin=origin, training_design="whole_window"
    )
    rec_in = [w for w in d_in["windows"] if w["series_uid"] == "t1"][0]
    rec_w = [w for w in d_w["windows"] if w["series_uid"] == "t1"][0]
    _check(rows, "input_only_BX_may_differ_from_whole_window_X0",
           not _close(rec_in["context"], rec_w["context"])
           and _close(rec_in["context"], rec_in["input_baseline_context"])
           and _close(rec_w["context"], rec_w["baseline_full_context"])
           and not _close(rec_in["input_baseline_context"], rec_in["baseline_full_context"]))

    sparse = np.full(CONTEXT_LENGTH + HORIZON, np.nan)
    sparse[0] = 0.0
    sparse[CONTEXT_LENGTH:] = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + HORIZON)
    sparse_whole = _prepare_train_window(
        sparse, None, training_design="whole_window", recombination=None)
    sparse_input_failed = False
    try:
        _prepare_train_window(sparse, None, training_design="input_only", recombination=None)
    except PreparationFailed:
        sparse_input_failed = True
    _check(rows, "undefined_BX_does_not_reject_valid_full_window",
           sparse_whole["input_baseline_context"] is None
           and np.isfinite(sparse_whole["context"]).all() and sparse_input_failed)

    shape_rejected = False
    try:
        build_training_design(roster, dict(values, t0=np.ones((2, 480))),
                              identity_train, config, origin=origin)
    except PreparationFailed:
        shape_rejected = True
    _check(rows, "multidimensional_series_not_silently_flattened", shape_rejected)

    # 9. eval identity on a non-identity shared model is not forced gain 0
    d_het_fit = build_training_design(
        roster, values, {"t0": winsor, "t1": None}, config, origin=origin
    )
    m_het = _fit(d_het_fit)
    p_het = predict_assignment(m_het, roster, values, identity_eval, config, origin=origin)
    s_het = score_predictions(p_het, roster, values, config, origin=origin)
    design_changed = not _close(d_het_fit["x_train"], d_id["x_train"]) or not _close(
        d_het_fit["y_train"], d_id["y_train"]
    )
    pred_changed = not _close(p_het["predictions"], p_id["predictions"])
    _check(rows, "eval_identity_on_treated_model_not_forced_gain_zero",
           design_changed and pred_changed and s_het["complete"] is True
           and s_id["complete"] is True
           and not _close(s_het["per_view_smase"], s_id["per_view_smase"]),
           {"static": s_id["per_view_smase"], "treated_train": s_het["per_view_smase"]})

    # 10. predict does not read post-origin bytes
    leaked = dict(values)
    leaked["e0"] = values["e0"].copy()
    leaked["e0"][origin + 5] = 1.0e9
    p_leak = predict_assignment(m_id, roster, leaked, identity_eval, config, origin=origin)
    _check(rows, "predict_ignores_post_origin_bytes",
           _close(p_id["predictions"], p_leak["predictions"])
           and p_id["predictions"].shape == (2, HORIZON))

    missing_truth = dict(values, e1=values["e1"].copy())
    missing_truth["e1"][origin:origin + HORIZON] = np.nan
    partial = score_predictions(p_id, roster, missing_truth, config, origin=origin)
    _check(rows, "missing_truth_keeps_full_population_unknown",
           partial["mean_smase"] is None and partial["complete"] is False
           and len(partial["per_view_smase"]) == 2
           and partial["per_view_smase"][0] is not None
           and partial["per_view_smase"][1] is None)

    # geometry constants used by the kernel
    _check(rows, "frozen_geometry_is_192_48_alpha_1",
           CONTEXT_LENGTH == 192 and HORIZON == 48
           and np.asarray(m_id.coefficients).shape == (193, 48)
           and float(m_id.ridge_alpha) == 1.0
           and bool(m_id.intercept_unpenalized))

    spent = int(FITS["n"])
    _check(rows, "synthetic_fit_budget", spent <= MAX_FITS and spent >= 1,
           {"consumer_fits": spent, "cap": MAX_FITS})
    return {
        "suite": "DEV_TRAIN1_NUMERIC",
        "checks": rows,
        "passed": all(row["passed"] for row in rows),
        "consumer_fits": spent,
        "llm_calls": 0,
        "real_data_reads": 0,
    }


if __name__ == "__main__":
    report = run()
    for row in report["checks"]:
        print("PASS" if row["passed"] else "FAIL", "|", row["check"])
        if row.get("detail") is not None and not row["passed"]:
            print("      ", json.dumps(row["detail"], ensure_ascii=False, default=str)[:240])
    print("passed:", report["passed"], "| fits:", report["consumer_fits"],
          "| llm:", report["llm_calls"])
    sys.exit(0 if report["passed"] else 1)
