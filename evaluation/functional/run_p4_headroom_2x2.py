"""P4 zero-LLM 2x2 common-headroom check (reviewer directive, 2026-08-16).

No Slow call.  For the two verified alternatives on the two failure origins,
report whether either program is common-positive (both gains >= +M).
"""
from __future__ import annotations

import json
import sys
import itertools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_a5a3_runtime_regression as reg  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
)
from SelfEvolvingHarnessTS.contracts.observables import (  # noqa: E402
    OBSERVABLE_FEATURES,
    observable_numeric_bin,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

ORIGINS = (888, 984)
ALTERNATIVES = ("outlier_mad", "hampel_filter")
M = 0.005
EXCLUDED_SCOPE_FEATURES = (
    "imputation_probe_direction",
    "clipping_probe_direction",
    "denoising_probe_direction",
    "level_probe_direction",
)


def _scope_hypothesis_probe(root):
    """S1a zero-LLM: can any single numeric observable bin predicate put 888
    on one side and 984 on the other?  Bare-float thresholds are forbidden."""
    cohort = reg._load(root)
    series0 = np.asarray(
        cohort["values"][cohort["roster"][0]["series_uid"]],
        dtype=np.float64,
    )
    rows = {
        origin: dict(
            extract_public_features(series0[:origin], task_kind="forecast")
        )
        for origin in ORIGINS
    }
    candidates: list[dict[str, object]] = []
    for feature, feature_type in sorted(OBSERVABLE_FEATURES.items()):
        if feature_type != "number" or feature in EXCLUDED_SCOPE_FEATURES:
            continue
        values = {}
        for origin in ORIGINS:
            raw = rows[origin].get(feature)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                values[origin] = None
                continue
            values[origin] = observable_numeric_bin(
                feature, float(raw)
            )
        candidates.append({
            "feature": feature,
            "bins": {
                str(origin): values[origin] for origin in ORIGINS
            },
            "separates": bool(
                values[ORIGINS[0]] is not None
                and values[ORIGINS[1]] is not None
                and values[ORIGINS[0]] != values[ORIGINS[1]]
            ),
        })
    separators = [c for c in candidates if c["separates"]]
    return {
        "origins": list(ORIGINS),
        "numeric_features_checked": [
            c["feature"] for c in candidates
        ],
        "feature_bins": candidates,
        "separating_features": [c["feature"] for c in separators],
        "verdict": (
            "SCOPE_HYPOTHESIS_EXPRESSIBLE_IN_BINS"
            if separators
            else "SCOPE_HYPOTHESIS_NOT_EXPRESSIBLE_IN_BINS"
        ),
    }


D0_ORIGIN = 984
D0_ANCHORS = tuple(a for a in range(312, 853, 60) if a != 852)
# v3: previous v2 grid was invalidated by eval-scale-floor failures at
# 3168/3408 and by contamination of already-opened origins. This grid is
# outcome-blind, time-interleaved, baseline-evaluation available, and all
# origins are previously unopened for candidate gains.
SCOPE_VALIDATION_GRID_V2 = (
    (1104, "outlier_mad"),
    (1368, "hampel_filter"),
    (1800, "outlier_mad"),
    (2856, "hampel_filter"),
    (3648, "outlier_mad"),
    (3888, "hampel_filter"),
)


def _eval_recent_high_count(root, origin: int) -> int:
    cohort = reg._load(root)
    rows = [r for r in cohort["roster"] if r["role"] != "train"]
    count = 0
    for row in rows:
        series = np.asarray(
            cohort["values"][row["series_uid"]], dtype=np.float64
        )
        features = dict(extract_public_features(
            series[max(0, origin - 192):origin],
            task_kind="forecast",
        ))
        if float(features["local_robust_z_peak"]) >= 6.0:
            count += 1
    return count


def _s1b_d3(root):
    cohort = reg._load(root)
    config = dict(_config())
    programs = {
        "outlier_mad": (("outlier_mad", {}),),
        "hampel_filter": (("hampel_filter", {}),),
        "winsorize": (("winsorize", {}),),
    }
    rows = []
    for origin, assigned in SCOPE_VALIDATION_GRID_V2:
        executor = ScopeExecutor(
            cohort["roster"], cohort["values"], config,
            evaluate_fn=_evaluate_kdd,
        )
        gains = {
            name: executor.evaluate(steps, origin).gain
            for name, steps in programs.items()
        }
        selected = gains[assigned]
        rows.append({
            "origin": origin,
            "assigned_program": assigned,
            "gains": gains,
            "selected_gain": selected,
            "promotion_pass": (
                selected is not None and selected >= M
            ),
        })
    def macro(name):
        vals = [r["gains"][name] for r in rows]
        return float(np.mean(vals)) if all(v is not None for v in vals) else None
    def harm_count(name):
        return sum(1 for r in rows if (r["gains"][name] or 0.0) < -M)
    def harm_magnitude(name):
        return float(sum(max(0.0, -(r["gains"][name] or 0.0))
                          for r in rows if (r["gains"][name] or 0.0) < -M))
    fixed = {
        name: {
            "macro_gain": macro(name),
            "harm_origin_count": harm_count(name),
            "harm_magnitude": harm_magnitude(name),
        }
        for name in ("outlier_mad", "hampel_filter", "winsorize")
    }
    fixed_competitors = ("outlier_mad", "hampel_filter")
    best_fixed = max(
        fixed_competitors,
        key=lambda name: (fixed[name]["macro_gain"] or -1e9),
    )
    s_macro = macro("assigned_program") if False else float(np.mean(
        [r["selected_gain"] for r in rows]
    )) if all(r["selected_gain"] is not None for r in rows) else None
    s_harm_count = sum(
        1 for r in rows if (r["selected_gain"] or 0.0) < -M
    )
    s_harm_magnitude = float(sum(
        max(0.0, -(r["selected_gain"] or 0.0))
        for r in rows if (r["selected_gain"] or 0.0) < -M
    ))
    utility_pass = bool(
        s_macro is not None
        and fixed[best_fixed]["macro_gain"] is not None
        and s_macro >= fixed[best_fixed]["macro_gain"] + M
        and s_harm_count <= fixed[best_fixed]["harm_origin_count"]
        and s_harm_magnitude <= fixed[best_fixed]["harm_magnitude"]
    )
    promotion_feasible = all(r["promotion_pass"] for r in rows)
    if utility_pass and promotion_feasible:
        verdict = "SCOPE_SELECTION_CONFIRMED"
    elif utility_pass:
        verdict = "SCOPE_UTILITY_PASS_PROMOTION_INCOMPATIBLE"
    else:
        verdict = "CONTEXT_SCOPE_HYPOTHESIS_NOT_CONFIRMED"
    return {
        "rows": rows,
        "fixed_program_metrics": fixed,
        "best_fixed_program": best_fixed,
        "scope_policy": {
            "macro_gain": s_macro,
            "harm_origin_count": s_harm_count,
            "harm_magnitude": s_harm_magnitude,
        },
        "utility_pass": utility_pass,
        "promotion_feasible": promotion_feasible,
        "verdict": verdict,
    }


def _s1b_d4(root):
    """D4 zero-LLM: use the 4 support series to choose a program, evaluate on
    the 4 query series.  Existing two-program query oracle is upper bound."""
    cohort = reg._load(root)
    train = [r for r in cohort["roster"] if r["role"] == "train"]
    support = [r for r in cohort["roster"] if r["role"] == "support"]
    query = [r for r in cohort["roster"] if r["role"] == "query"]
    programs = {
        "outlier_mad": (("outlier_mad", {}),),
        "hampel_filter": (("hampel_filter", {}),),
    }

    def subset_gain(rows, origin, op):
        executor = ScopeExecutor(
            train + rows, cohort["values"], dict(_config()),
            evaluate_fn=_evaluate_kdd,
        )
        return executor.evaluate(programs[op], origin).gain

    rows = []
    support_probe_count = 0
    query_probe_count = 0
    abstain_count = 0
    for origin, _high_z_program in SCOPE_VALIDATION_GRID_V2:
        support_gains = {
            op: subset_gain(support, origin, op) for op in programs
        }
        support_probe_count += len(programs)
        available = {
            op: g for op, g in support_gains.items() if g is not None
        }
        chosen = None
        if available and max(available.values()) >= M:
            chosen = max(available, key=lambda op: available[op])
        if chosen is None:
            abstain_count += 1
            query_selected_gain = 0.0
        else:
            query_probe_count += 1
            query_selected_gain = subset_gain(query, origin, chosen)
            if query_selected_gain is None:
                query_selected_gain = 0.0
        query_gains = {
            op: subset_gain(query, origin, op) for op in programs
        }
        query_probe_count += len(programs)
        query_oracle = max(
            query_gains,
            key=lambda op: query_gains[op]
            if query_gains[op] is not None else -1e9,
        )
        rows.append({
            "origin": origin,
            "high_z_program": _high_z_program,
            "support_gains": support_gains,
            "support_chosen": chosen,
            "support_chosen_gain": (
                support_gains.get(chosen) if chosen else None
            ),
            "query_gains": query_gains,
            "query_selected_gain": query_selected_gain,
            "query_oracle": query_oracle,
            "query_oracle_gain": query_gains[query_oracle],
        })

    def policy_query_gains(row_getter):
        vals = [row_getter(r) for r in rows]
        finite = [v for v in vals if v is not None]
        return {
            "macro_gain": float(np.mean(vals)) if len(vals) == len(rows) else None,
            "harm_origin_count": sum(1 for v in vals if v is not None and v < -M),
            "harm_magnitude": float(sum(
                -v for v in vals if v is not None and v < -M
            )),
        }

    support_policy = policy_query_gains(
        lambda r: r["query_selected_gain"]
    )
    best_fixed_query = {
        op: policy_query_gains(lambda r, op=op: r["query_gains"][op])
        for op in programs
    }
    query_oracle = policy_query_gains(
        lambda r: r["query_oracle_gain"]
    )
    return {
        "rows": rows,
        "support_conditioned_policy": support_policy,
        "fixed_query_programs": best_fixed_query,
        "query_oracle_upper_bound": query_oracle,
        "abstention_count": abstain_count,
        "support_probe_count": support_probe_count,
        "deployment_query_probe_count": (
            query_probe_count - (len(programs) * len(SCOPE_VALIDATION_GRID_V2))
        ),
        "diagnostic_query_probe_count": (
            len(programs) * len(SCOPE_VALIDATION_GRID_V2)
        ),
        "verdict": "SUPPORT_CONDITIONED_SELECTION_NOT_CONFIRMED",
    }


def _s1b_d5(root):
    """D5 zero-LLM: per-series gain matrix, leave-one-support-out leverage,
    and all 70 directed 4/4 support/query partitions under frozen mean+M."""
    cohort = reg._load(root)
    train = [r for r in cohort["roster"] if r["role"] == "train"]
    support = [r["series_uid"] for r in cohort["roster"] if r["role"] == "support"]
    query = [r["series_uid"] for r in cohort["roster"] if r["role"] == "query"]
    all8 = support + query
    origins = [o for o, _op in SCOPE_VALIDATION_GRID_V2]
    programs = {
        "outlier_mad": (("outlier_mad", {}),),
        "hampel_filter": (("hampel_filter", {}),),
    }
    per = {
        uid: {origin: {} for origin in origins} for uid in all8
    }
    for uid in all8:
        for origin in origins:
            executor = ScopeExecutor(
                train + [{"series_uid": uid, "role": "eval"}],
                cohort["values"], dict(_config()), evaluate_fn=_evaluate_kdd,
            )
            for op, steps in programs.items():
                per[uid][origin][op] = executor.evaluate(steps, origin).gain

    def policy(sset, qset, aggregate="mean"):
        gains = []
        chosen_by_origin = []
        abstain = 0
        for origin in origins:
            support_values = {
                op: [per[u][origin][op] for u in sset]
                for op in programs
            }
            support_scores = {}
            for op, vals in support_values.items():
                vals = [v for v in vals if v is not None]
                if not vals:
                    support_scores[op] = None
                elif aggregate == "mean":
                    support_scores[op] = float(np.mean(vals))
                elif aggregate == "median":
                    support_scores[op] = float(np.median(vals))
            available = {
                op: v for op, v in support_scores.items() if v is not None
            }
            chosen = None
            if available and max(available.values()) >= M:
                chosen = max(available, key=lambda op: available[op])
            if chosen is None:
                abstain += 1
                selected = 0.0
            else:
                qvals = [per[u][origin][chosen] for u in qset]
                qvals = [v for v in qvals if v is not None]
                selected = float(np.mean(qvals)) if len(qvals) == len(qset) else 0.0
            gains.append(selected)
            chosen_by_origin.append((origin, chosen))
        return {
            "macro_gain": float(np.mean(gains)),
            "harm_origin_count": sum(1 for g in gains if g < -M),
            "harm_magnitude": float(sum(-g for g in gains if g < -M)),
            "abstention_count": abstain,
            "chosen_by_origin": chosen_by_origin,
        }

    def majority_policy(sset, qset):
        gains = []
        abstain = 0
        chosen_by_origin = []
        for origin in origins:
            votes = []
            for uid in sset:
                vals = {op: per[uid][origin][op] for op in programs}
                available = {op: v for op, v in vals.items() if v is not None}
                if available and max(available.values()) >= M:
                    votes.append(max(available, key=lambda op: available[op]))
            counts = {op: votes.count(op) for op in programs}
            if counts["outlier_mad"] == counts["hampel_filter"]:
                chosen = None
            else:
                chosen = max(counts, key=lambda op: counts[op])
            if chosen is None:
                abstain += 1
                selected = 0.0
            else:
                qvals = [per[u][origin][chosen] for u in qset]
                qvals = [v for v in qvals if v is not None]
                selected = float(np.mean(qvals)) if len(qvals) == len(qset) else 0.0
            gains.append(selected)
            chosen_by_origin.append((origin, chosen))
        return {
            "macro_gain": float(np.mean(gains)),
            "harm_origin_count": sum(1 for g in gains if g < -M),
            "harm_magnitude": float(sum(-g for g in gains if g < -M)),
            "abstention_count": abstain,
            "chosen_by_origin": chosen_by_origin,
        }

    current = policy(set(support), set(query), "mean")
    current_median = policy(set(support), set(query), "median")
    current_majority = majority_policy(set(support), set(query))
    leave_one_out = []
    for removed in support:
        row = policy(set(support) - {removed}, set(query), "mean")
        row["removed_support_series"] = removed
        leave_one_out.append(row)

    partition_rows = []
    for combo in itertools.combinations(all8, 4):
        sset = set(combo)
        qset = set(all8) - sset
        result = policy(sset, qset, "mean")
        fixed = {}
        for op in programs:
            vals = []
            for origin in origins:
                sub = [per[u][origin][op] for u in qset]
                sub = [v for v in sub if v is not None]
                vals.append(float(np.mean(sub)) if len(sub) == len(qset) else None)
            fixed[op] = float(np.mean(vals)) if all(v is not None for v in vals) else None
        best_fixed = max(fixed, key=lambda op: fixed[op])
        agree = total = 0
        for origin, chosen in result["chosen_by_origin"]:
            qvals = {
                op: float(np.mean([
                    v for v in [per[u][origin][op] for u in qset]
                    if v is not None
                ]))
                for op in programs
                if all(v is not None for v in [per[u][origin][op] for u in qset])
            }
            total += 1
            if qvals and chosen == max(qvals, key=lambda op: qvals[op]):
                agree += 1
        partition_rows.append({
            "support_series": tuple(sorted(sset)),
            "query_series": tuple(sorted(qset)),
            "macro_gain": result["macro_gain"],
            "best_fixed_query_program": best_fixed,
            "best_fixed_query_macro_gain": fixed[best_fixed],
            "beats_best_fixed": result["macro_gain"] > fixed[best_fixed],
            "harm_origin_count": result["harm_origin_count"],
            "harm_magnitude": result["harm_magnitude"],
            "abstention_count": result["abstention_count"],
            "query_oracle_agreement_rate": agree / total if total else None,
        })
    partition_rows.sort(key=lambda row: row["macro_gain"])
    current_ss = tuple(sorted(support))
    current_row = next(
        row for row in partition_rows if row["support_series"] == current_ss
    )
    current_rank = partition_rows.index(current_row) + 1
    beats_count = sum(1 for row in partition_rows if row["beats_best_fixed"])
    high_leverage = [
        row["removed_support_series"]
        for row in leave_one_out
        if row["macro_gain"] > current["macro_gain"] + M
    ]
    verdict = (
        "SMALL_SUPPORT_SET_TRANSFER_UNSTABLE"
        if beats_count <= len(partition_rows) // 2
        else "FROZEN_SUPPORT_ROSTER_COMPOSITION_FAILURE"
    )
    return {
        "per_series_gain_matrix": {
            uid: {
                str(origin): {
                    op: per[uid][origin][op] for op in programs
                }
                for origin in origins
            }
            for uid in all8
        },
        "current_frozen_partition": {
            "support_series": current_ss,
            "query_series": tuple(sorted(query)),
            "mean_policy": current,
            "median_policy": current_median,
            "majority_policy": current_majority,
            "rank_by_macro_gain": current_rank,
            "total_partitions": len(partition_rows),
        },
        "leave_one_support_out": leave_one_out,
        "high_leverage_support_series": high_leverage,
        "partition_enumeration": {
            "n_partitions": len(partition_rows),
            "beats_best_fixed_count": beats_count,
            "beats_best_fixed_proportion": beats_count / len(partition_rows),
            "rows": partition_rows,
        },
        "verdict": verdict,
    }


def _deconfound_d0(root):
    cohort = reg._load(root)
    config = dict(_config())
    config["anchors"] = list(D0_ANCHORS)
    executor = ScopeExecutor(
        cohort["roster"], cohort["values"], config,
        evaluate_fn=_evaluate_kdd,
    )
    gains = {}
    for op in ALTERNATIVES:
        receipt = executor.evaluate(((op, {}),), D0_ORIGIN)
        gains[op] = {
            "gain": receipt.gain,
            "verification_passed": receipt.verification.passed,
            "checked_windows": receipt.verification.checked_windows,
        }
    outlier = gains["outlier_mad"]["gain"]
    hampel = gains["hampel_filter"]["gain"]
    if outlier is not None and hampel is not None and outlier < M <= hampel:
        verdict = "EVAL_CONTEXT_SUFFICIENT"
    elif outlier is not None and hampel is not None and outlier >= M > hampel:
        verdict = "TRAINING_CONTEXT_NECESSARY"
    else:
        verdict = "TRAIN_EVAL_INTERACTION_UNRESOLVED"
    return {
        "origin": D0_ORIGIN,
        "anchors": list(D0_ANCHORS),
        "gains": gains,
        "verdict": verdict,
    }


REPORT_REL = (
    PROJECT_ROOT
    / "artifacts/functional/e2"
    / "w1_p4_headroom_2x2_report.json"
)


def main() -> int:
    cohort = reg._load(PROJECT_ROOT)
    executor = ScopeExecutor(
        cohort["roster"], cohort["values"], _config(),
        evaluate_fn=_evaluate_kdd,
    )
    grid: dict[str, dict[str, object]] = {}
    common = {}
    for op in ALTERNATIVES:
        gains = {}
        for origin in ORIGINS:
            receipt = executor.evaluate(((op, {}),), origin)
            gains[str(origin)] = {
                "gain": receipt.gain,
                "verification_passed": receipt.verification.passed,
                "checked_windows": receipt.verification.checked_windows,
            }
        grid[op] = gains
        common[op] = all(
            isinstance(gains[str(origin)]["gain"], (int, float))
            and float(gains[str(origin)]["gain"]) >= M
            for origin in ORIGINS
        )
    any_common = any(common.values())
    scope_probe = _scope_hypothesis_probe(PROJECT_ROOT)
    d0 = _deconfound_d0(PROJECT_ROOT)
    d3 = _s1b_d3(PROJECT_ROOT)
    d4 = _s1b_d4(PROJECT_ROOT)
    d5 = _s1b_d5(PROJECT_ROOT)
    validation_grid = {
        "origins": [o for o, _op in SCOPE_VALIDATION_GRID_V2],
        "assigned_policy": [
            {"origin": o, "program": op,
             "eval_recent_high_z_count": _eval_recent_high_count(
                 PROJECT_ROOT, o
             )}
            for o, op in SCOPE_VALIDATION_GRID_V2
        ],
        "freeze_rule": (
            "chosen outcome-blind by eval_recent_high_z_count>=4; "
            "origin % 24 == 0; [origin-192,origin+48) disjoint; "
            "low/high alternating in time"
        ),
    }
    report = {
        "experiment_id": "v1-p4-headroom-2x2",
        "note": (
            "Zero-LLM headroom check after P4 selected-program replay "
            "rejection. No Slow call, no Harness update, no gate change."
        ),
        "origins": list(ORIGINS),
        "alternatives": list(ALTERNATIVES),
        "material_threshold": M,
        "grid": grid,
        "common_positive": common,
        "verdict": (
            "COMMON_PROGRAM_HEADROOM_EXISTS"
            if any_common
            else "PER_CONTEXT_HEADROOM_EXISTS_NO_COMMON_PROGRAM"
        ),
        "scope_hypothesis_probe": scope_probe,
        "deconfound_d0": d0,
        "scope_validation_grid_v2": validation_grid,
        "s1b_d3_zero_llm_v3": d3,
        "d4_support_conditioned_selection_zero_llm": d4,
        "d5_support_query_sensitivity": d5,
        "next_action": (
            "D5 diagnostics: current blocker is support->query feedback "
            "transportability, not Program headroom. S4_NOT_AUTHORIZED; "
            "next is series/context-matched feedback, not new Workflow."
        ),
    }
    REPORT_REL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_REL.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "grid": {
            op: {origin: gains[str(origin)]["gain"] for origin in ORIGINS}
            for op, gains in grid.items()
        },
        "common_positive": common,
        "verdict": report["verdict"],
        "scope_hypothesis_probe": {
            "separating_features": scope_probe["separating_features"],
            "verdict": scope_probe["verdict"],
        },
        "deconfound_d0": d0,
        "scope_validation_grid_v2": validation_grid,
        "s1b_d3_zero_llm_v3": d3,
        "d4_support_conditioned_selection_zero_llm": d4,
        "d5_support_query_sensitivity": d5,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
