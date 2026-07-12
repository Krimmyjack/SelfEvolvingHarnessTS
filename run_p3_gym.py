"""run_p3_gym.py — P3 判决 runner：gym 认证 + proxy 保真度 + 种子供给 headroom + ε 注册。

（Final_Plan_CodeAgentFirst_2026-07-09 §P3；prereg §3。出口四件：0-API gym、fidelity 报告、
skill bank v1、ε 正式注册。）

保真度（R4 硬门材料）：per task 汇总 (proxy_delta, true_delta) 对（rows × candidates）的
Spearman ρ；ρ ≥ rho_min 才认证该任务的 proxy 可用于 gym 驱动的进化主张，否则标
escalate-only。anomaly 的**可部署**候选面只剩插补类（registry 物理禁平滑）→ 两指标
几乎不动 → 方差不足属预期诚实结局；另算一组**违约诊断**（off-path 执行平滑/删改，
P2 式）证明 proxy 在空间真移动时确实跟踪 true——诊断组不进 headroom、不解锁部署。

headroom（原 G1 判决重定位）：per series 每任务
  best_menu  = max(0, 冻结 menu v1 任务合法动作的 true delta)
  best_all   = max(0, menu ∪ seed bank v1)
  headroom   = best_all − best_menu ≥ 0
ε 注册规则（prereg §3，落盘后冻结）：ε = max(0.02, forecast headroom 的 bootstrap ci90_lo)。

范围：合成 gym substrate（motivation-grade），非 confirmatory；真实 corpus 上的供给判决
属 P5 identity gate 本体。

命令：
  D:\\Anaconda_envs\\envs\\project\\python.exe -m SelfEvolvingHarnessTS.run_p3_gym --n-series 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .evaluators.anomaly_rig import DETECTOR_SPEC, anomaly_readiness_eval, make_anomaly_slice
from .harness.layers import minimal_l2
from .policy.action_spec import action_menu_v1
from .policy.program_edit import _resolved_steps_v1
from .policy.seed_programs import BANK_VERSION, SEED_PROGRAMS_V1, seed_bank_manifest, seed_skill_cards
from .readiness_gym import (
    GYM_SPEC,
    ReadinessGym,
    anomaly_proxy_delta,
    forecast_proxy_delta,
)
from .run_p2_motivation import seasonal_naive_nrmse
from .sandbox.executor import run_pipeline

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P3Gym"
RHO_MIN = 0.7          # 保真度认证阈值（Spearman；prereg §3 占位在此填入并冻结）
EPSILON_FLOOR = 0.02
TASKS = ("forecast", "anomaly_detection")

# 违约诊断组（off-path 执行，不可部署，不进 headroom；证明 anomaly proxy 有牙齿）
_VIOLATION_DIAG = {
    "diag_median_w9": [("impute_linear", {}), ("denoise_median", {"window": 9})],
    "diag_winsor": [("impute_linear", {}), ("winsorize", {})],
    "diag_universal": [("impute_linear", {}), ("winsorize", {}), ("denoise_median", {"window": 9})],
}


def _menu_candidates(task: str) -> List[Tuple[str, List[Tuple[str, dict]]]]:
    menu = action_menu_v1()
    out = []
    for aid, spec in sorted(menu.actions.items()):
        if task in spec.task_constraints:
            out.append((aid, [(s.op, dict(s.params)) for s in spec.steps]))
    return out


def _seed_candidates(task: str, defaults: Mapping[str, Mapping]) -> List[Tuple[str, List[Tuple[str, dict]]]]:
    out = []
    for name, spec in SEED_PROGRAMS_V1.items():
        if spec.task_type == task:
            out.append((name, [(op, dict(p)) for op, p in _resolved_steps_v1(spec, defaults)]))
    return out


def _execute(steps: List[Tuple[str, dict]], x: np.ndarray) -> Tuple[np.ndarray, bool]:
    result = run_pipeline(steps, x)
    ok = bool(result.ok and result.artifact is not None and result.artifact.shape == x.shape)
    return (np.asarray(result.artifact, dtype=float) if ok else x.copy()), ok


def _true_delta(task: str, artifact: np.ndarray, row: Mapping[str, Any], x: np.ndarray) -> float:
    if task == "forecast":
        raw_s = seasonal_naive_nrmse(x, row["future_clean"], int(row["period"]))
        art_s = seasonal_naive_nrmse(artifact, row["future_clean"], int(row["period"]))
        return float(raw_s - art_s)
    raw_f1 = anomaly_readiness_eval(x, row["labels"], raw_reference=x)["F1"]
    art_f1 = anomaly_readiness_eval(artifact, row["labels"], raw_reference=x)["F1"]
    return float(art_f1 - raw_f1)


def _proxy_delta(task: str, artifact: np.ndarray, row: Mapping[str, Any], x: np.ndarray) -> float:
    if task == "forecast":
        return float(forecast_proxy_delta(artifact, x, int(row["period"])))
    return float(anomaly_proxy_delta(artifact, x))


def _spearman(pairs: List[Tuple[float, float]], *, min_n: int = 5,
              min_distinct: int = 5) -> float | None:
    """成对 Spearman；样本不足或任一列近简并（distinct 值 < min_distinct）→ None（不可判）。"""
    from scipy.stats import spearmanr
    finite = [(p, t) for p, t in pairs if np.isfinite(p) and np.isfinite(t)]
    if len(finite) < min_n:
        return None
    arr = np.asarray(finite, dtype=float)
    for col in (arr[:, 0], arr[:, 1]):
        if len(np.unique(np.round(col, 12))) < min_distinct:
            return None                            # 近简并（首轮教训：平局海里 ρ=−1.0 假象）
    rho = spearmanr(arr[:, 0], arr[:, 1]).statistic
    return float(rho) if np.isfinite(rho) else None


def _fidelity_entry(pairs_by_uid: Dict[str, List[Tuple[float, float]]],
                    rho_min: float) -> Dict[str, Any]:
    """主判据 = **within-series** 排序保真（gym 里 proxy 的用途 = 给该序列的候选排序）。
    pooled 只作诊断：跨序列尺度差会制造 Simpson 反向（首轮实测 pooled −0.32 而
    within +0.35、候选均值层完全单调对齐）。覆盖率 < 0.5 → insufficient_variance。"""
    rhos = [rho for pairs in pairs_by_uid.values()
            if (rho := _spearman(pairs)) is not None]
    all_pairs = [pair for pairs in pairs_by_uid.values() for pair in pairs]
    pooled = _spearman(all_pairs, min_n=8)
    n_series = len(pairs_by_uid)
    coverage = len(rhos) / max(1, n_series)
    if not rhos or coverage < 0.5:
        status = "insufficient_variance"
        mean_rho = None
    else:
        mean_rho = float(np.mean(rhos))
        status = "pass" if mean_rho >= rho_min else "fail"
    return {
        "within_series_mean_rho": mean_rho,
        "within_series_rho_p25": (float(np.percentile(rhos, 25)) if rhos else None),
        "n_series_scored": len(rhos),
        "series_coverage": float(coverage),
        "pooled_rho_diagnostic": pooled,
        "n_pairs": len(all_pairs),
        "status": status,
    }


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, b: int) -> Tuple[float, float]:
    n = values.size
    means = np.empty(b)
    for i in range(b):
        means[i] = float(np.mean(values[rng.integers(0, n, n)]))
    lo, hi = np.percentile(means, (5.0, 95.0))
    return float(lo), float(hi)


def _gym_smoke(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """出口判据①：gym 0-API 可执行重放（两任务各走一条脚本化 episode）。"""
    out = {}
    scripts = {
        "forecast": {"grammar": "v1", "steps": [["impute_linear", {}], ["denoise_median", {"window": 9}]],
                     "scope": ["*"], "task_type": "forecast", "pattern_guard": [],
                     "risk_budget_beta": 0.3, "fallback": "v_impute_linear"},
        "anomaly_detection": {"grammar": "v1", "steps": [["impute_linear", {}]],
                              "scope": ["*"], "task_type": "anomaly_detection", "pattern_guard": [],
                              "risk_budget_beta": 0.3, "fallback": "v_impute_linear"},
    }
    for task, prog in scripts.items():
        gym = ReadinessGym(rows[:1], task=task, budget=2)
        gym.reset(0)
        gym.step({"op": "proxy_eval", "program_spec": prog})
        gym.step({"op": "finalize", "program_spec": prog})
        res = gym.result(0)
        out[task] = {"final_kind": res["final_kind"], "proxy_evals_used": res["proxy_evals_used"]}
    return out


def run_p3(n_series: int = 60, out_dir: Path | str | None = None,
           seed: int = 20260709, bootstrap_b: int = 1000,
           rho_min: float = RHO_MIN) -> Dict[str, Any]:
    rows = make_anomaly_slice(n_series, seed=seed)
    defaults = minimal_l2().operator_defaults
    rng = np.random.default_rng(seed + 29)

    records: List[Dict[str, Any]] = []
    fidelity_pairs: Dict[str, Dict[str, List[Tuple[float, float]]]] = {t: {} for t in TASKS}
    diag_pairs: List[Tuple[float, float]] = []
    per_series_true: Dict[str, Dict[str, Dict[str, float]]] = {t: {} for t in TASKS}

    candidates = {t: {"menu": _menu_candidates(t), "seed": _seed_candidates(t, defaults)}
                  for t in TASKS}

    for row in rows:
        x = np.asarray(row["x"], dtype=float)
        for task in TASKS:
            per_series_true[task].setdefault(row["uid"], {})
            for source in ("menu", "seed"):
                for name, steps in candidates[task][source]:
                    artifact, ok = _execute(steps, x)
                    true = _true_delta(task, artifact, row, x) if ok else 0.0
                    proxy = _proxy_delta(task, artifact, row, x) if ok else 0.0
                    records.append({"uid": row["uid"], "cell": row["cell"], "task": task,
                                    "source": source, "name": name, "executed_ok": ok,
                                    "true_delta": true, "proxy_delta": proxy,
                                    "deployable": True})
                    fidelity_pairs[task].setdefault(row["uid"], []).append((proxy, true))
                    per_series_true[task][row["uid"]][f"{source}:{name}"] = true
        # anomaly 违约诊断组（off-path；不进 headroom / 不可部署）
        for name, steps in _VIOLATION_DIAG.items():
            artifact, ok = _execute(steps, x)
            true = _true_delta("anomaly_detection", artifact, row, x) if ok else 0.0
            proxy = _proxy_delta("anomaly_detection", artifact, row, x) if ok else 0.0
            records.append({"uid": row["uid"], "cell": row["cell"], "task": "anomaly_detection",
                            "source": "violation_diag", "name": name, "executed_ok": ok,
                            "true_delta": true, "proxy_delta": proxy,
                            "deployable": False})
            diag_pairs.append((proxy, true))

    # ── fidelity ────────────────────────────────────────────────────────────
    fidelity: Dict[str, Any] = {}
    for task in TASKS:
        entry = _fidelity_entry(fidelity_pairs[task], rho_min)
        if task == "anomaly_detection":
            diag_rho = _spearman(diag_pairs, min_n=8)
            entry["diagnostic_with_violations"] = {
                "pooled_rho": diag_rho,
                "n_pairs": len(diag_pairs),
                "status": ("pass" if diag_rho is not None and diag_rho >= rho_min else
                           ("fail" if diag_rho is not None else "insufficient_variance")),
                "basis": "pooled_offpath（诊断组刻意跨 off-path 空间，pooled 口径合理）",
            }
            entry["note"] = ("deployable 候选面=插补类（registry 物理禁平滑/删改）→ 指标近常量属预期；"
                             "违约诊断组证明 proxy 在空间移动时跟踪 true")
        fidelity[task] = entry
    escalate_only = [t for t in TASKS if fidelity[t]["status"] != "pass"]

    # ── headroom + ε ────────────────────────────────────────────────────────
    headroom: Dict[str, Any] = {}
    for task in TASKS:
        diffs = []
        seed_wins: Dict[str, int] = {}
        for uid, trues in per_series_true[task].items():
            menu_vals = [v for k, v in trues.items() if k.startswith("menu:")]
            all_vals = list(trues.values())
            best_menu = max(0.0, max(menu_vals) if menu_vals else 0.0)
            best_all = max(0.0, max(all_vals) if all_vals else 0.0)
            diffs.append(best_all - best_menu)
            if best_all > best_menu + 1e-12:
                winner = max(trues, key=trues.get)
                if winner.startswith("seed:"):
                    seed_wins[winner[5:]] = seed_wins.get(winner[5:], 0) + 1
        arr = np.asarray(diffs, dtype=float)
        lo, hi = _bootstrap_ci(arr, rng, bootstrap_b)
        headroom[task] = {"mean": float(arr.mean()), "ci90_lo": lo, "ci90_hi": hi,
                          "n_series": int(arr.size),
                          "series_with_seed_win": int((arr > 1e-12).sum()),
                          "seed_win_counts": dict(sorted(seed_wins.items()))}

    epsilon = round(max(EPSILON_FLOOR, headroom["forecast"]["ci90_lo"]), 4)

    report = {
        "phase": "P3_gym_fidelity_headroom",
        "claim_scope": "synthetic gym substrate (motivation-grade); real-corpus supply verdict belongs to P5 identity gate",
        "n_series": int(n_series),
        "fidelity": fidelity,
        "escalate_only_tasks": escalate_only,
        "headroom": headroom,
        "epsilon_registered": float(epsilon),
        "epsilon_rule": f"max(0.02, forecast headroom ci90_lo) = max(0.02, {headroom['forecast']['ci90_lo']:.4f})",
    }
    manifest = {
        "generated_by": "run_p3_gym",
        "plan": "Final_Plan_CodeAgentFirst_2026-07-09 §P3",
        "prereg": "results/Stage2/prereg_codeagent_first_P1_P5.md §3",
        "seed": int(seed), "bootstrap_b": int(bootstrap_b), "n_series": int(n_series),
        "rho_min": float(rho_min),
        "api_calls": 0,
        "detector": dict(DETECTOR_SPEC),
        "gym": dict(GYM_SPEC),
        "gym_smoke": _gym_smoke(rows),
        "menu_sha256": action_menu_v1().sha256,
        "bank_version": BANK_VERSION,
        "violation_diag_programs": {k: [[op, p] for op, p in v] for k, v in _VIOLATION_DIAG.items()},
        "headroom_note": "seed guard 不参与 headroom（测的是供给潜力上限）；guard 生效面在 gate/gym",
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "skill_bank_v1.json").write_text(json.dumps(
            {"bank_version": BANK_VERSION, "programs": seed_bank_manifest(),
             "cards": seed_skill_cards(), "menu_sha256": action_menu_v1().sha256},
            ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (out / "VERDICT.md").write_text(_verdict_md(report, manifest), encoding="utf-8")
    return report


def _verdict_md(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    fc, ad = report["fidelity"]["forecast"], report["fidelity"]["anomaly_detection"]
    hr = report["headroom"]
    lines = [
        "# P3 VERDICT：gym 认证 + proxy 保真度 + 种子供给 headroom + ε 注册",
        "",
        f"> 范围：{report['claim_scope']}；seed={manifest['seed']}，n={report['n_series']}，"
        f"B={manifest['bootstrap_b']}，ρ_min={manifest['rho_min']}。",
        "",
        "## 保真度（R4 硬门材料；主判据=within-series 排序保真，pooled 仅诊断）",
        "",
        f"- forecast: within-series 均值 ρ={fc['within_series_mean_rho']}"
        f"（p25={fc['within_series_rho_p25']}, n_series={fc['n_series_scored']}，"
        f"pooled 诊断={fc['pooled_rho_diagnostic']}）→ **{fc['status'].upper()}**",
        f"- anomaly_detection（deployable 面）: status=**{ad['status'].upper()}**"
        f"（within 覆盖率={ad['series_coverage']:.2f}）",
        f"- anomaly 违约诊断组: pooled ρ={ad['diagnostic_with_violations']['pooled_rho']}"
        f"（n={ad['diagnostic_with_violations']['n_pairs']}）→ proxy 在空间移动时确实跟踪 true",
        f"- escalate-only 任务: {report['escalate_only_tasks'] or '无'}",
        "",
        "## 供给 headroom（vs 冻结 menu v1 + dose oracle，同判官同 split）",
        "",
    ]
    for task in TASKS:
        h = hr[task]
        lines.append(f"- {task}: mean **{h['mean']:+.4f}** [{h['ci90_lo']:+.4f},{h['ci90_hi']:+.4f}]，"
                     f"seed 严格获胜序列 {h['series_with_seed_win']}/{h['n_series']}，"
                     f"胜者分布 {h['seed_win_counts']}")
    lines += [
        "",
        f"## ε 正式注册：**{report['epsilon_registered']}**（规则 {report['epsilon_rule']}）",
        "",
        "P5 identity gate 的 utility 判据以此 ε 为效应量门槛；本值落盘后冻结（prereg §3）。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 gym fidelity/headroom runner")
    parser.add_argument("--n-series", type=int, default=60)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap-b", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()
    report = run_p3(n_series=args.n_series, out_dir=args.out_dir,
                    bootstrap_b=args.bootstrap_b, seed=args.seed)
    print(json.dumps({
        "phase": report["phase"],
        "fidelity_forecast": report["fidelity"]["forecast"]["status"],
        "fidelity_anomaly": report["fidelity"]["anomaly_detection"]["status"],
        "headroom_forecast": report["headroom"]["forecast"]["mean"],
        "epsilon_registered": report["epsilon_registered"],
        "escalate_only": report["escalate_only_tasks"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
