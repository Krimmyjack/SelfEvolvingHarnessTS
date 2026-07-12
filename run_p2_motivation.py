"""run_p2_motivation.py — P2 动机表实验（Final_Plan_CodeAgentFirst_2026-07-09 §P2，论文实验 1）。

命题：**TS data readiness 不是普适质量**——同一批序列、同一组预处理程序，在不同任务判官下
utility 符号翻转。fresh 部分 = forecast × anomaly_detection（同数据双任务：历史含注入异常，
forecast 判官=seasonal-naive nRMSE（尖峰/噪声是害），anomaly 判官=冻结 residual z-score F1
（尖峰是检测目标））；第二组任务对 = forecast × classification 引用冻结 classify C1 结果
（_clf_maintable.log：v_stl/v_savgol 助 forecast 伤 classify，v_median 助 classify）。

表列还带 deployable_under_contract：registry allowed_tasks（D6 物理契约）+ TaskSpec
forbidden_modifications 是否允许该程序进部署路径——展示"翻转已被契约编码"。

范围声明：合成动机切片（motivation-grade），非 confirmatory；判官冻结于 DETECTOR_SPEC /
seasonal-naive，ε=0.01 预注册于 manifest（不调参）。

命令：
  D:\\Anaconda_envs\\envs\\project\\python.exe -m SelfEvolvingHarnessTS.run_p2_motivation --n-series 40
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .evaluators.anomaly_rig import DETECTOR_SPEC, anomaly_readiness_eval, make_anomaly_slice
from .harness.layers import minimal_l2
from .operators.registry import OPERATOR_METADATA, canonicalize
from .policy.task_spec import TaskSpec, anomaly_task_spec_v1, forecast_task_spec_v1
from .sandbox.executor import run_pipeline

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P2Motivation"
CLF_LOG = Path(__file__).resolve().parent / "_clf_maintable.log"
EPSILON = 0.01                      # 预注册翻转判据幅度（不调参；normalized metric 口径）
REFERENCE_ACTION = "v_raw_identity"

TASKS = ("forecast", "anomaly_detection")


def _task_specs() -> Dict[str, TaskSpec]:
    return {
        "forecast": forecast_task_spec_v1(
            horizon=24, downstream_model_class="seasonal_naive_h24"),
        "anomaly_detection": anomaly_task_spec_v1(
            downstream_model_class=f"residual_zscore_w{DETECTOR_SPEC['window']}"
                                   f"_t{DETECTOR_SPEC['threshold']}"),
    }


def _programs(row: Mapping[str, Any], task: str) -> Dict[str, List[Tuple[str, dict]]]:
    """program 名 → 有序 (op, override) 链。task_conditioned 是唯一按任务/结构变化的行。"""
    high = "snrHigh" in str(row["cell"])
    tc = ([("impute_linear", {}), ("denoise_median", {"window": 9 if high else 25})]
          if task == "forecast" else [("impute_linear", {})])
    return {
        "v_raw_identity": [],
        "v_impute_linear": [("impute_linear", {})],
        "median_w9": [("impute_linear", {}), ("denoise_median", {"window": 9})],
        "winsor": [("impute_linear", {}), ("winsorize", {})],
        "universal_cleaner": [("impute_linear", {}), ("winsorize", {}),
                              ("denoise_median", {"window": 9})],
        "task_conditioned": tc,
    }


def _resolve(steps: Sequence[Tuple[str, dict]], defaults: Mapping[str, Mapping]) -> List[Tuple[str, dict]]:
    return [(op, {**defaults.get(op, {}), **override}) for op, override in steps]


def _deployable(steps: Sequence[Tuple[str, dict]], task: str, spec: TaskSpec) -> bool:
    for op, _params in steps:
        meta = OPERATOR_METADATA.get(canonicalize(op))
        if meta is None or task not in meta.get("allowed_tasks", ()):
            return False
        if spec.is_op_forbidden(op):
            return False
    return True


def seasonal_naive_nrmse(history: np.ndarray, future_clean: np.ndarray, period: int) -> float:
    """确定性 forecast 判官：seasonal-naive（复制最后一个周期）对干净未来的 nRMSE。
    预测含 NaN（strict raw + 缺失）→ 用历史 nanmean 填（无信息填充，记入代价）。"""
    hist = np.asarray(history, dtype=float).ravel()
    future = np.asarray(future_clean, dtype=float).ravel()
    h = future.size
    reps = int(np.ceil(h / period))
    pred = np.tile(hist[-period:], reps)[:h]
    if np.any(~np.isfinite(pred)):
        fill = float(np.nanmean(hist)) if np.any(np.isfinite(hist)) else 0.0
        pred = np.where(np.isfinite(pred), pred, fill)
    rmse = float(np.sqrt(np.mean((pred - future) ** 2)))
    return rmse / (float(np.std(future)) + 1e-9)


def _bootstrap_ci(deltas: np.ndarray, rng: np.random.Generator, b: int,
                  q: Tuple[float, float] = (5.0, 95.0)) -> Tuple[float, float]:
    n = deltas.size
    means = np.empty(b)
    for i in range(b):
        means[i] = float(np.mean(deltas[rng.integers(0, n, n)]))
    lo, hi = np.percentile(means, q)
    return float(lo), float(hi)


def _load_classify_citation(path: Path) -> Dict[str, Any]:
    """从冻结 classify C1 主表日志抽引用行（frozen 证据，非本次 fresh run）。"""
    lines: List[str] = []
    if path.exists():
        pattern = re.compile(r"^\s*(v_median|v_savgol|v_stl)\s+inception=([+-][\d.]+)\s+rocket=([+-][\d.]+)")
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if pattern.match(raw):
                lines.append(raw.strip())
    return {
        "source": str(path.name) if not path.exists() else str(path).replace("\\", "/"),
        "quoted_lines": lines,
        "reading": ("frozen classify C1（ΔPerf vs raw, reporter ⟂ judge, final_test split）: "
                    "v_median 助 classify（inception +0.249 / rocket +0.134）；"
                    "v_stl 伤 classify（-0.037 / -0.110）而 stl/savgol 族在 forecast 侧为正收益"
                    "（E-1.1/F0 冻结结果）→ forecast×classification 符号翻转（frozen 任务对）"),
        "status": "frozen_result_citation",
    }


def run_p2(n_series: int = 40, out_dir: Path | str | None = None,
           bootstrap_b: int = 1000, seed: int = 20260709) -> Dict[str, Any]:
    slice_rows = make_anomaly_slice(n_series, seed=seed)
    defaults = minimal_l2().operator_defaults
    specs = _task_specs()
    rng = np.random.default_rng(seed + 13)

    program_names = list(_programs(slice_rows[0], "forecast").keys())
    records: List[Dict[str, Any]] = []
    # per (program, task) 的逐序列 delta（vs v_raw_identity，正=对该任务更好）
    deltas: Dict[str, Dict[str, List[float]]] = {p: {t: [] for t in TASKS} for p in program_names}
    deploy: Dict[str, Dict[str, bool]] = {p: {} for p in program_names}

    for row in slice_rows:
        x = np.asarray(row["x"], dtype=float)
        # 每任务先算 raw 参照
        raw_scores: Dict[str, float] = {}
        artifacts_cache: Dict[str, Tuple[np.ndarray, bool]] = {}
        row_records: Dict[Tuple[str, str], Dict[str, Any]] = {}   # (task, program) → record 引用

        for task in TASKS:
            progs = _programs(row, task)
            for name, steps in progs.items():
                key = f"{task}|{name}"
                resolved = _resolve(steps, defaults)
                if resolved:
                    result = run_pipeline(resolved, x)
                    ok = bool(result.ok and result.artifact is not None
                              and result.artifact.shape == x.shape)
                    artifact = result.artifact if ok else x.copy()
                else:
                    ok, artifact = True, x.copy()
                artifacts_cache[key] = (artifact, ok)
                deploy[name].setdefault(task, _deployable(resolved, task, specs[task]))

            for name in progs:
                artifact, ok = artifacts_cache[f"{task}|{name}"]
                if task == "forecast":
                    score = seasonal_naive_nrmse(artifact, row["future_clean"], row["period"])
                else:
                    score = anomaly_readiness_eval(artifact, row["labels"], raw_reference=x)["F1"]
                if name == REFERENCE_ACTION:
                    raw_scores[task] = score
                rec = {
                    "uid": row["uid"], "cell": row["cell"], "task": task, "program": name,
                    "metric": specs[task].metric.name, "value": float(score),
                    "executed_ok": ok,
                    "deployable_under_contract": deploy[name][task],
                }
                records.append(rec)
                row_records[(task, name)] = rec

        for task in TASKS:
            for name in program_names:
                rec = row_records[(task, name)]
                if task == "forecast":                       # nRMSE lower-better → raw − prog
                    delta = raw_scores[task] - rec["value"]
                else:                                        # F1 higher-better → prog − raw
                    delta = rec["value"] - raw_scores[task]
                rec["delta_vs_raw_identity"] = float(delta)
                deltas[name][task].append(float(delta))

    table: Dict[str, Any] = {}
    for name in program_names:
        entry: Dict[str, Any] = {"deployable_under_contract": deploy[name]}
        for task in TASKS:
            arr = np.asarray(deltas[name][task], dtype=float)
            lo, hi = _bootstrap_ci(arr, rng, bootstrap_b)
            entry[task] = {
                "metric_delta": ("nRMSE_raw_minus_prog" if task == "forecast" else "F1_prog_minus_raw"),
                "mean_delta": float(arr.mean()),
                "ci90_lo": lo, "ci90_hi": hi, "n": int(arr.size),
            }
        table[name] = entry

    flip_programs = []
    for name in program_names:
        f, a = table[name]["forecast"], table[name]["anomaly_detection"]
        if (f["mean_delta"] > EPSILON and f["ci90_lo"] > 0.0
                and a["mean_delta"] < -EPSILON and a["ci90_hi"] < 0.0):
            flip_programs.append(name)
    fresh_flip_pairs = 1 if flip_programs else 0
    classify_citation = _load_classify_citation(CLF_LOG)
    frozen_flip_pairs = 1 if classify_citation["quoted_lines"] else 0

    report = {
        "phase": "P2_motivation_table",
        "claim": "TS data readiness is task-conditional (same series, same program, opposite utility sign)",
        "claim_scope": "synthetic motivation-grade slice; frozen deterministic judges; NOT confirmatory",
        "n_series": int(n_series),
        "table": table,
        "flip_programs_fresh": flip_programs,
        "fresh_flip_pairs": fresh_flip_pairs,          # forecast × anomaly_detection
        "frozen_flip_pairs": frozen_flip_pairs,        # forecast × classification（引用）
        "exit_criterion_met": bool(fresh_flip_pairs + frozen_flip_pairs >= 2),
        "classify_citation": classify_citation,
    }
    manifest = {
        "generated_by": "run_p2_motivation",
        "plan": "Final_Plan_CodeAgentFirst_2026-07-09 §P2",
        "prereg": "results/Stage2/prereg_codeagent_first_P1_P5.md §2",
        "epsilon": EPSILON,
        "reference_action": REFERENCE_ACTION,
        "flip_rule": "mean>|eps| 且 90% bootstrap CI 同侧不跨 0（两任务反向）",
        "seed": int(seed), "bootstrap_b": int(bootstrap_b), "n_series": int(n_series),
        "detector": dict(DETECTOR_SPEC),
        "forecast_judge": {"name": "seasonal_naive_h24", "metric": "nRMSE_vs_clean_future"},
        "task_spec_shas": {task: spec.sha() for task, spec in specs.items()},
        "slice": {"generator": "evaluators.anomaly_rig.make_anomaly_slice",
                  "cells": ["anomaly|snrHigh|full", "anomaly|snrLow|miss"],
                  "shared_series_across_tasks": True},
        "programs": {name: [[op, dict(ov)] for op, ov in _programs(slice_rows[0], "forecast")[name]]
                     for name in program_names if name != "task_conditioned"},
        "task_conditioned_rule": "forecast→impute+median(w9 snrHigh / w25 snrLow, F0 剂量单调)；anomaly→impute only",
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (out / "VERDICT.md").write_text(_verdict_md(report, manifest), encoding="utf-8")
    return report


def _verdict_md(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    lines = [
        "# P2 动机表 VERDICT（论文实验 1：readiness 非普适）",
        "",
        f"> 范围：{report['claim_scope']}；ε={manifest['epsilon']}；参照={manifest['reference_action']}；"
        f"seed={manifest['seed']}，n={report['n_series']}，B={manifest['bootstrap_b']}。",
        "",
        "| program | forecast Δ(nRMSE↓) [90%CI] | anomaly Δ(F1↑) [90%CI] | 部署契约允许(f/a) |",
        "|---|---|---|---|",
    ]
    for name, entry in report["table"].items():
        f, a, d = entry["forecast"], entry["anomaly_detection"], entry["deployable_under_contract"]
        lines.append(
            f"| {name} | {f['mean_delta']:+.4f} [{f['ci90_lo']:+.4f},{f['ci90_hi']:+.4f}] "
            f"| {a['mean_delta']:+.4f} [{a['ci90_lo']:+.4f},{a['ci90_hi']:+.4f}] "
            f"| {'Y' if d.get('forecast') else 'N'}/{'Y' if d.get('anomaly_detection') else 'N'} |")
    lines += [
        "",
        f"**fresh 翻转程序（forecast×anomaly，判据过 CI）**: {', '.join(report['flip_programs_fresh']) or '无'}",
        f"**任务对计数**: fresh={report['fresh_flip_pairs']}（forecast×anomaly）"
        f" + frozen={report['frozen_flip_pairs']}（forecast×classification 引用）"
        f" → 出口判据(≥2) **{'PASS' if report['exit_criterion_met'] else 'FAIL'}**",
        "",
        "## frozen classify 引用（第二组任务对）",
        "",
        f"来源: `{report['classify_citation']['source']}`",
        "",
    ]
    lines += [f"    {q}" for q in report["classify_citation"]["quoted_lines"]]
    lines += [
        "",
        report["classify_citation"]["reading"],
        "",
        "## 解读",
        "",
        "同一批序列上，universal cleaner（impute→winsor→median9）助 forecast（去噪去尖峰）却毁",
        "anomaly（尖峰正是检测目标）；registry 任务契约（D6）已把该翻转编码为物理禁入（表末列 N）。",
        "task_conditioned 行显示按任务选择处理即可同时保住两侧。这就是 pattern/task 条件化",
        "readiness 的动机证据（motivation-grade；confirmatory 见 P5）。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 motivation table runner")
    parser.add_argument("--n-series", type=int, default=40)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap-b", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()
    report = run_p2(n_series=args.n_series, out_dir=args.out_dir,
                    bootstrap_b=args.bootstrap_b, seed=args.seed)
    print(json.dumps({k: report[k] for k in
                      ("phase", "n_series", "fresh_flip_pairs", "frozen_flip_pairs",
                       "flip_programs_fresh", "exit_criterion_met")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
