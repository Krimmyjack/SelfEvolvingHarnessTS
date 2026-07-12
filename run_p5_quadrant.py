"""run_p5_quadrant.py — P5-B pattern-vs-domain 四象限迁移（真 Monash，论文实验 2；prereg §5.0 冻结）。

轴（冻结）：pattern = 退化网格 preset cell（G_hi/lo × full/miss，E-1.1/F0 谱系的可观测结构轴）；
domain = 数据集 config（nn5_daily / tourism_monthly / fred_md）。12 真实基底 × 4 preset = 48
episodes。判官 = seasonal-naive（period=系列自身周期）nRMSE vs **真实未来**（true 判官）。

迁移协议：对每个 target episode，四象限源组 = 其余 episodes 按 (domain 同/异 × pattern 同/异)
过滤且**排除同 series_uid**（防同基底泄漏）；recipe = 源组均值最优动作；regret = target 的
per-series oracle delta − recipe delta。假设：diff-domain/same-pattern regret <
same-domain/diff-pattern（配对差，grouped bootstrap 按 series_uid）。

诚实注记：本轴测的是"退化结构 pattern"迁移；内在结构 pattern 需更大语料（future work）。

命令：D:\\Anaconda_envs\\envs\\project\\python.exe -m SelfEvolvingHarnessTS.run_p5_quadrant
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .data.load_real import FORECAST_PRESETS, _forecast_from_signal, load_signals
from .harness.layers import minimal_l2
from .run_p2_motivation import seasonal_naive_nrmse
from .sandbox.executor import run_pipeline

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P5Quadrant"
QUADRANTS = ("same_domain_same_pattern", "same_domain_diff_pattern",
             "diff_domain_same_pattern", "diff_domain_diff_pattern")

_ACTIONS: Dict[str, List[Tuple[str, dict]]] = {
    "v_none": [("impute_linear", {})],
    "f0_median_w9": [("impute_linear", {}), ("denoise_median", {"window": 9})],
    "f0_median_w15": [("impute_linear", {}), ("denoise_median", {"window": 15})],
    "f0_median_w25": [("impute_linear", {}), ("denoise_median", {"window": 25})],
    "v_winsor_savgol": [("impute_linear", {}), ("winsorize", {}), ("denoise_savgol", {})],
}


def _episode_seed(config: str, item: str, preset: str) -> int:
    return int(hashlib.sha256(f"{config}|{item}|{preset}".encode()).hexdigest()[:8], 16)


def _build_episodes() -> List[Dict[str, Any]]:
    defaults = minimal_l2().operator_defaults
    episodes: List[Dict[str, Any]] = []
    for sig in load_signals():
        for preset in FORECAST_PRESETS:
            raw = _forecast_from_signal(sig, preset, seed=_episode_seed(sig.config, sig.item_id, preset))
            x = np.asarray(raw.history, dtype=float)
            raw_score = seasonal_naive_nrmse(x, raw.future, raw.period)
            deltas: Dict[str, float] = {}
            for name, steps in _ACTIONS.items():
                resolved = [(op, {**defaults.get(op, {}), **ov}) for op, ov in steps]
                result = run_pipeline(resolved, x)
                ok = bool(result.ok and result.artifact is not None
                          and result.artifact.shape == x.shape)
                artifact = np.asarray(result.artifact, dtype=float) if ok else x.copy()
                deltas[name] = float(raw_score - seasonal_naive_nrmse(artifact, raw.future, raw.period))
            episodes.append({
                "uid": f"{raw.series_uid}|{preset}",
                "series_uid": raw.series_uid,
                "domain": raw.origin,
                "pattern": preset,
                "period": int(raw.period),
                "deltas": deltas,
                "oracle_action": max(deltas, key=deltas.get),
                "oracle_delta": float(max(deltas.values())),
            })
    return episodes


def _quadrant_of(target: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    same_d = source["domain"] == target["domain"]
    same_p = source["pattern"] == target["pattern"]
    if same_d and same_p:
        return "same_domain_same_pattern"
    if same_d:
        return "same_domain_diff_pattern"
    if same_p:
        return "diff_domain_same_pattern"
    return "diff_domain_diff_pattern"


def _grouped_bootstrap_ci(values_by_group: Dict[str, List[float]], rng: np.random.Generator,
                          b: int) -> Tuple[float, float]:
    groups = sorted(values_by_group)
    means = np.empty(b)
    for i in range(b):
        picked = rng.choice(groups, size=len(groups), replace=True)
        pooled = [v for g in picked for v in values_by_group[g]]
        means[i] = float(np.mean(pooled))
    lo, hi = np.percentile(means, (5.0, 95.0))
    return float(lo), float(hi)


def run_p5_quadrant(out_dir: Path | str | None = None, bootstrap_b: int = 2000) -> Dict[str, Any]:
    episodes = _build_episodes()
    rng = np.random.default_rng(20260710)
    actions = sorted(_ACTIONS)

    per_target: List[Dict[str, Any]] = []
    for target in episodes:
        row: Dict[str, Any] = {"uid": target["uid"], "series_uid": target["series_uid"],
                               "domain": target["domain"], "pattern": target["pattern"],
                               "oracle_action": target["oracle_action"],
                               "oracle_delta": target["oracle_delta"],
                               "regret": {}}
        for quadrant in QUADRANTS:
            sources = [s for s in episodes
                       if s["series_uid"] != target["series_uid"]
                       and _quadrant_of(target, s) == quadrant]
            if not sources:
                row["regret"][quadrant] = None
                continue
            recipe = max(actions, key=lambda a: float(np.mean([s["deltas"][a] for s in sources])))
            row["regret"][quadrant] = float(target["oracle_delta"] - target["deltas"][recipe])
            row.setdefault("recipe", {})[quadrant] = recipe
        per_target.append(row)

    quadrant_mean: Dict[str, float] = {}
    quadrant_cov: Dict[str, int] = {}
    for quadrant in QUADRANTS:
        vals = [r["regret"][quadrant] for r in per_target if r["regret"][quadrant] is not None]
        quadrant_cov[quadrant] = len(vals)
        quadrant_mean[quadrant] = float(np.mean(vals)) if vals else None

    # 假设检验：per-target 配对差 regret(dd_sp) − regret(sd_dp)，按 series_uid 分组 bootstrap
    diff_by_series: Dict[str, List[float]] = {}
    for r in per_target:
        a, b = r["regret"]["diff_domain_same_pattern"], r["regret"]["same_domain_diff_pattern"]
        if a is not None and b is not None:
            diff_by_series.setdefault(r["series_uid"], []).append(float(a - b))
    all_diffs = [v for vs in diff_by_series.values() for v in vs]
    diff_mean = float(np.mean(all_diffs)) if all_diffs else None
    ci = _grouped_bootstrap_ci(diff_by_series, rng, bootstrap_b) if diff_by_series else (None, None)

    report = {
        "phase": "P5B_pattern_vs_domain_quadrant",
        "data": "real Monash (nn5_daily, tourism_monthly, fred_md), 12 base signals x 4 presets",
        "n_episodes": len(episodes),
        "quadrant_mean_regret": quadrant_mean,
        "quadrant_coverage": quadrant_cov,
        "hypothesis_dd_sp_vs_sd_dp": {
            "paired_diff_mean": diff_mean,
            "ci90": [ci[0], ci[1]],
            "n_pairs": len(all_diffs),
            "direction_holds": (diff_mean is not None and diff_mean < 0.0),
            "ci_excludes_zero": (ci[0] is not None and ci[1] < 0.0),
            "reading": "负 = diff-domain/same-pattern 迁移优于 same-domain/diff-pattern（pattern 承重）",
        },
        "claim_scope": "退化结构 pattern 轴（E-1.1/F0 谱系）；内在结构 pattern 需更大语料（future work）",
    }
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "generated_by": "run_p5_quadrant",
        "prereg": "results/Stage2/prereg_codeagent_first_P1_P5.md §5.0（P5-B 冻结）",
        "pattern_axis": "degradation_preset_cell", "domain_axis": "dataset_config",
        "actions": {k: [[op, ov] for op, ov in v] for k, v in _ACTIONS.items()},
        "judge": "seasonal_naive nRMSE vs real future (H_FORECAST), true judge",
        "leakage_guard": "sources exclude same series_uid",
        "bootstrap_b": int(bootstrap_b), "grouped_by": "series_uid",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in per_target:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap-b", type=int, default=2000)
    args = parser.parse_args()
    report = run_p5_quadrant(out_dir=args.out_dir, bootstrap_b=args.bootstrap_b)
    print(json.dumps({"quadrant_mean_regret": report["quadrant_mean_regret"],
                      "hypothesis": report["hypothesis_dd_sp_vs_sd_dp"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
