"""Can deployment-visible features predict which series a cleaner will harm?

This is the follow-up the CONFLICT per-series audit left open.  That audit
showed the harm is systematic rather than random, but it showed it with the
Consumer's own identity sMASE at the deployment origin -- a downstream outcome
a held-out Fast Path may not read, and therefore not something a Skill Scope
may key on.  The open question is whether any *pre-deployment* observable
carries the same signal, because that is what decides whether a CONFLICT can
drive a bounded Slow Scope/Risk revision or whether Slow should legally abstain.

Every feature here is computed from ``values[:origin]`` or from the operator's
own footprint on that pre-origin window.  Nothing reads the evaluation horizon,
the Consumer outcome at this origin, Query, Final, UCR TEST or sealed AD.
No LLM.  No new SHA or manifest.  Changes no threshold and releases nothing.

Validation is leave-one-origin-out: every unit and series sharing an origin is
held out together, so a fold is never scored on an origin it trained on.  The
220 (unit x series) observations are not independent -- they replay the same 20
series across 7 origins -- and pooled random splits would flatter the result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from SelfEvolvingHarnessTS.contracts.observables import (
    OUTLIER_Z_THRESHOLD,
    PUBLIC_ROBUST_Z_MAD_FLOOR,
)
from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT = (
    PROJECT_ROOT / "artifacts/main_protocol/p4_conflict_per_series_audit_20260831.json"
)
REPORT = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p4_deployment_visible_risk_audit_20260831.json"
)

MATERIAL = 0.005
HORIZON = forecast_p4.HORIZON
PERIOD = forecast_p4.PERIOD
CONTEXT = 192  # the recent-window length window_context itself reports on


def _robust_z(window: np.ndarray) -> np.ndarray:
    centre = float(np.median(window))
    mad = float(np.median(np.abs(window - centre)))
    scale = max(1.4826 * mad, PUBLIC_ROBUST_Z_MAD_FLOOR)
    return np.abs(window - centre) / scale


def _longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def _footprint(window: np.ndarray, operator: str) -> dict[str, float]:
    """How much of the pre-origin window this operator actually rewrites.

    Deployment-visible: running the cleaner on observed history and looking at
    what it touched needs no outcome and no future values.
    """
    compiled = forecast_p1.forecast_runtime._compiled_bound_program(
        {"op": str(operator), "params": {}},
        environment="p4_deployment_visible_risk_audit",
    )
    prepared, _trace = forecast_p1.forecast_runtime._apply_program(window, compiled)
    delta = np.abs(np.asarray(prepared, dtype=np.float64) - window)
    scale = max(float(np.median(np.abs(window - np.median(window)))) * 1.4826,
                PUBLIC_ROBUST_Z_MAD_FLOOR)
    touched = delta > 1e-12
    return {
        "modified_fraction": float(touched.mean()),
        "modified_longest_run": float(_longest_run(touched)),
        "modified_mean_abs_delta_scaled": float(delta.mean() / scale),
        "modified_max_abs_delta_scaled": float(delta.max() / scale),
    }


def _series_features(
    base: Any, uid: str, origin: int, operator: str, backtest: float
) -> dict[str, float]:
    values = np.asarray(base.values[uid], dtype=np.float64)
    window = values[origin - CONTEXT : origin]
    z = _robust_z(window)
    outlier = z > OUTLIER_Z_THRESHOLD
    features: dict[str, float] = {
        # Pre-origin backtest error: the legal stand-in for identity sMASE.
        # Evaluated one horizon earlier, so its own evaluation span still ends
        # at origin and never touches the deployment horizon.
        "backtest_smase_prev_origin": float(backtest),
        "outlier_density": float(outlier.mean()),
        "outlier_max_robust_z": float(z.max()),
        "outlier_longest_run": float(_longest_run(outlier)),
    }
    features.update(_footprint(window, operator))
    features.update(
        {
            key: float(value)
            for key, value in resolver.window_context(
                {uid: values}, origin, PERIOD
            ).items()
        }
    )
    return features


def _auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Rank AUC for separating harmed (positive) from helped (negative)."""
    if positive.size == 0 or negative.size == 0:
        return None
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (positive.size * negative.size))


def _grouped_auc(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[float | None, list[dict[str, Any]]]:
    """Mean within-origin AUC, weighted by the comparable pairs each origin has.

    Scoring inside an origin is what keeps an origin-level offset from being
    read as discrimination between series.
    """
    per_group: list[dict[str, Any]] = []
    total_pairs = 0.0
    weighted = 0.0
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        positive = scores[mask & (labels == 1)]
        negative = scores[mask & (labels == 0)]
        auc = _auc(positive, negative)
        pairs = float(positive.size * negative.size)
        per_group.append(
            {
                "origin": int(group),
                "harmed": int(positive.size),
                "helped": int(negative.size),
                "auc": auc,
            }
        )
        if auc is not None:
            weighted += auc * pairs
            total_pairs += pairs
    return (weighted / total_pairs if total_pairs else None), per_group


def build_report() -> dict[str, Any]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    units = [row for row in audit["units"] if row["conflict_stage"] == "SUPPORT"]
    base, _selection, data = forecast_p1._load_exposed_cells()

    rows: list[dict[str, Any]] = []
    for unit in units:
        origin = int(unit["origin"])
        operator = str(unit["operator"])
        roster = base.roster("support_a")
        uids = [str(row["series_uid"]) for row in roster if row["role"] == "eval"]
        # One-horizon-earlier identity reading, per series, entirely pre-origin.
        previous = forecast_p4._reading(
            forecast_p4._cell_at(base, origin - HORIZON),
            "support_a",
            (),
            origin=origin - HORIZON,
        )["per_series_smase"]
        gains = unit["support_a"]["per_series_gain"]
        if len(uids) != len(gains):
            raise RuntimeError(
                "eval roster (%d) does not line up with per-series gains (%d)"
                % (len(uids), len(gains))
            )
        for index, (uid, gain) in enumerate(zip(uids, gains)):
            if abs(gain) <= MATERIAL:
                continue  # neutral: neither harmed nor helped, carries no label
            rows.append(
                {
                    "origin": origin,
                    "operator": operator,
                    "series_index": index,
                    "series_uid": uid,
                    "gain": float(gain),
                    "harmed": int(gain < -MATERIAL),
                    "features": _series_features(
                        base, uid, origin, operator, previous[index]
                    ),
                    "reference_identity_smase": float(
                        unit["support_a"]["identity_per_series_smase"][index]
                    ),
                }
            )

    names = sorted(rows[0]["features"])
    matrix = np.array([[row["features"][name] for name in names] for row in rows])
    labels = np.array([row["harmed"] for row in rows])
    groups = np.array([row["origin"] for row in rows])

    univariate = []
    for column, name in enumerate(names):
        scores = matrix[:, column]
        auc, per_group = _grouped_auc(scores, labels, groups)
        if auc is None:
            continue
        univariate.append(
            {
                "feature": name,
                "grouped_auc": auc,
                # A feature that ranks harm backwards is just as informative;
                # report the distance from chance so both directions are visible.
                "abs_distance_from_chance": abs(auc - 0.5),
                "per_origin": per_group,
            }
        )
    univariate.sort(key=lambda row: -row["abs_distance_from_chance"])

    # Leave-one-origin-out multivariate: no fold is ever scored on an origin it
    # was fitted on, so a per-origin offset cannot leak across the split.
    out_of_fold = np.full(labels.shape, np.nan)
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        if len(set(labels[train].tolist())) < 2:
            continue
        scaler = StandardScaler().fit(matrix[train])
        model = LogisticRegression(max_iter=2000, C=0.5).fit(
            scaler.transform(matrix[train]), labels[train]
        )
        out_of_fold[test] = model.predict_proba(scaler.transform(matrix[test]))[:, 1]
    scored = ~np.isnan(out_of_fold)
    multivariate_auc, multivariate_groups = _grouped_auc(
        out_of_fold[scored], labels[scored], groups[scored]
    )

    reference = np.array([row["reference_identity_smase"] for row in rows])
    # identity sMASE ranks harm the other way round (clean series get hurt), so
    # negate before scoring it on the same "higher = more likely harmed" scale.
    reference_auc, reference_groups = _grouped_auc(-reference, labels, groups)

    best = univariate[0] if univariate else None
    verdict = (
        "DEPLOYMENT_VISIBLE_FEATURES_DO_NOT_SEPARATE_HARM"
        if multivariate_auc is None or multivariate_auc < 0.65
        else "DEPLOYMENT_VISIBLE_FEATURES_CARRY_USABLE_RISK_SIGNAL"
    )

    return {
        "stage": "P4_DEPLOYMENT_VISIBLE_RISK_AUDIT",
        "status": "COMPLETE",
        "evidence_grade": "DEVELOPMENT_ONLY_DIAGNOSIS_OF_COLLECTED_RUN",
        "upstream_audit": AUDIT.relative_to(PROJECT_ROOT).as_posix(),
        "dataset": data.get("dataset"),
        "data_role": "EXPOSED_DEVELOPMENT",
        "llm_calls": 0,
        "threshold_changed_by_this_audit": False,
        "boundary": {
            "natural_final_outcome_reads": 0,
            "query_evaluations": 0,
            "ucr_test_outcome_reads": 0,
            "sealed_ad_outcome_reads": 0,
            "deployment_horizon_reads": 0,
            "new_sha_added": False,
            "new_manifest_added": False,
            "live_provider_calls": 0,
        },
        "design": {
            "units": len(units),
            "labelled_observations": len(rows),
            "harmed": int(labels.sum()),
            "helped": int((labels == 0).sum()),
            "neutral_dropped": 20 * len(units) - len(rows),
            "origin_groups": sorted(set(groups.tolist())),
            "validation": "leave-one-origin-out; AUC scored within held-out origin",
            "feature_count": len(names),
            "features": names,
            "all_features_pre_origin": True,
        },
        "univariate_grouped_auc": univariate,
        "multivariate_leave_one_origin_out": {
            "grouped_auc": multivariate_auc,
            "per_origin": multivariate_groups,
            "model": "standardised logistic regression, C=0.5",
        },
        "reference_identity_smase": {
            "grouped_auc": reference_auc,
            "per_origin": reference_groups,
            "status": "DIAGNOSTIC_REFERENCE_ONLY",
            "not_a_deployable_scope_feature": True,
            "why": (
                "identity sMASE at the deployment origin is a Consumer outcome, "
                "not a pre-deployment observable; it is scored here only as the "
                "yardstick the deployment-visible features are compared against"
            ),
        },
        "best_single_deployment_visible_feature": best,
        "verdict": verdict,
        "releases": "NONE",
    }


def main() -> int:
    report = build_report()
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    design = report["design"]
    print(
        "labelled=%d (harmed=%d helped=%d neutral-dropped=%d) over %d origins"
        % (
            design["labelled_observations"],
            design["harmed"],
            design["helped"],
            design["neutral_dropped"],
            len(design["origin_groups"]),
        )
    )
    print(
        "multivariate LOGO grouped AUC = %s"
        % report["multivariate_leave_one_origin_out"]["grouped_auc"]
    )
    print(
        "identity-sMASE reference grouped AUC = %s (not deployable)"
        % report["reference_identity_smase"]["grouped_auc"]
    )
    for row in report["univariate_grouped_auc"][:8]:
        print("  %-52s grouped AUC = %.3f" % (row["feature"], row["grouped_auc"]))
    print("verdict: %s" % report["verdict"])
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
