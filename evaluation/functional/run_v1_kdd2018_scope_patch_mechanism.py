"""KDD_CUP_2018_SCOPE_PATCH_MECHANISM（P4.5，用户裁决 2026-08-11）。

**不是**"为 outlier_mad 写死 span_fraction<0.01 规则"（case patch）；
**是**"最小 Program-conditioned Observation → Typed Scope Patch → replay"
的 Scope 学习机制切片。用户四点修正全部采纳：
  1. Observation 与真实执行 Scope 对齐——复用实际训练窗口
     （scope_executor.training_windows——评估应用的同一窗口集合）逐窗口
     应用程序，聚合：每窗口 affected fraction / 修改簇数 / 跨度 /
     跨窗口覆盖率 / series 间分布 / cohort 聚合比例；
  2. 阈值只是 Target-local 内容——即使成立也只属于 KDD/当前 Task/
     outlier_mad 的 Skill Scope，不成为跨域 Shared 规则；
  3. 证据不足保持 Draft——不自动安装全局 Skill；未通过 replay →
     不写 active snapshot（LOCAL_DRAFT_REQUIRES_SUPPORT 语义，不退回
     const:true+LOCAL_ACTIVE）；
  4. 由 Harness 生成 Scope Patch——Runtime 从成功/冲突对照区间机械生成
     ≤2 个有界候选（阈值=midpoint，非人工值）；选择器（dev 确定性——
     分离度排序）模拟 Slow Agent 选择/abstain；Runtime 编译 Applicability。

流程（用户裁决 1-5 步的 development 切片；fresh Context 确认与薄入口
不在本 runner）：
  a. 已暴露窗口上对齐真实执行窗口，验证 localization 信号（9 点）；
  b. Runtime 从对照区间生成 ≤2 Scope Patch（midpoint）；
  c. 选择器选择/abstain；
  d. 编译 Applicability（task_kind + scope 条件）；
  e. 正向 Context（@888 对齐观测）replay：匹配（检索）；
     冲突 Context（@984）replay：不匹配（不自动优先）；
     removal：原行为；
  f. 通过才冻结；未过 → 不安装（如实记录）。

判定（预注册）：
  SCOPE_PATCH_MECHANISM_PASS（dev-level）：对齐观测信号成立 + Scope
    Patch 生成/编译/选择链工作 + 正/负 Context replay 行为正确
  ALIGNED_SIGNAL_LOST : 对齐观测下 localization 信号消失 →
    UNIDENTIFIABLE（不生成 Patch）
  SCOPE_PATCH_SELECTION_ABSTAINED : 有候选但最佳 margin_ratio < 0.5
    （证据不足——不冻结不安装；Draft 语义——用户修正 3）
  SCOPE_PATCH_REPLAY_FAILED : 生成/编译成功但 replay 行为不正确 →
    不安装（Draft 语义）
  PROTOCOL_FAILURE : 数据/装配失败

零新数据（已暴露窗口）/零新 LLM（确定性选择器）。阈值由 Runtime 从
对照区间机械生成（Target-local），不写死任何人工值。

用法：
  python evaluation/functional/run_v1_kdd2018_scope_patch_mechanism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
from SelfEvolvingHarnessTS.contracts.candidate import (  # noqa: E402
    Candidate,
    CandidateKind,
)
from SelfEvolvingHarnessTS.contracts.program import Program  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (  # noqa: E402
    CompiledWorkflow,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
)
from run_v1_kdd2018_program_effect_context_diagnostic import (  # noqa: E402
    _points_p41,
    _points_p43,
    _points_headroom,
    _series_values,
    _steps_for,
    _utility_class,
)

PERIOD = 24
HORIZON = 48
M = 0.005
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_kdd2018_scope_patch_mechanism_report.json"
ALIGNED_FEATURES = (
    "window_affected_mean", "window_affected_max", "window_clusters_mean",
    "window_span_mean", "window_span_max", "coverage",
    "series_coverage", "series_fraction_std", "cohort_aggregate_fraction")


def _aligned_observation(executor: ScopeExecutor, origin: int,
                         op: str,
                         max_anchor: int | None = None) -> dict[str, float | int]:
    """对齐真实执行窗口的 Program-effect Observation：对
    training_windows(origin)（评估应用的同一窗口集合——窗口
    [anchor−192, anchor+48]）逐窗口应用程序，聚合修改几何（用户
    修正 1）。max_anchor 用于共享锚点对照（审查 2026-08-11 决定性
    发现 2）。零 outcome 读取。"""
    steps = _steps_for(op)
    program = Program.from_steps(list(steps), source="p45_observation")
    candidate = Candidate(candidate_id="p45_obs", kind=CandidateKind.PROGRAM,
                          program=program, source="p45_observation")
    compiled = CompiledWorkflow(candidate, (), tuple(program.steps))
    windows = [w for w in executor.training_windows(origin)
               if max_anchor is None or int(w[1]) <= int(max_anchor)]
    per_window_affected: list[float] = []
    per_window_clusters: list[int] = []
    per_window_span: list[float] = []
    per_series_affected: dict[str, list[float]] = {}
    all_points = 0
    all_changed = 0
    for uid, _anchor, window in windows:
        after, _trace = v6._apply_program(window, compiled)
        changed = np.nonzero(np.abs(after - window) > 1e-12)[0]
        frac = float(changed.size) / float(window.size)
        per_window_affected.append(frac)
        per_series_affected.setdefault(str(uid), []).append(frac)
        all_points += int(window.size)
        all_changed += int(changed.size)
        if changed.size:
            runs: list[int] = []
            start = prev = int(changed[0])
            for idx in changed[1:]:
                idx = int(idx)
                if idx == prev + 1:
                    prev = idx
                else:
                    runs.append(prev - start + 1)
                    start = prev = idx
            runs.append(prev - start + 1)
            per_window_clusters.append(len(runs))
            per_window_span.append(float(changed[-1] - changed[0] + 1)
                                   / float(window.size))
    n = len(per_window_affected)
    n_series = len(per_series_affected)
    series_means = [float(np.mean(v)) for v in per_series_affected.values()]
    modified_series = [uid for uid, v in per_series_affected.items()
                       if any(f > 0.0 for f in v)]
    obs: dict[str, float | int] = {
        "n_windows": n,
        "window_affected_mean": round(float(np.mean(per_window_affected)), 6)
        if n else 0.0,
        "window_affected_max": round(float(np.max(per_window_affected)), 6)
        if n else 0.0,
        "window_clusters_mean": round(float(np.mean(per_window_clusters)), 4)
        if per_window_clusters else 0.0,
        "window_span_mean": round(float(np.mean(per_window_span)), 6)
        if per_window_span else 0.0,
        "window_span_max": round(float(np.max(per_window_span)), 6)
        if per_window_span else 0.0,
        # 审查修正（2026-08-11）：coverage = 有修改窗口占比（per_window_span
        # 只在有修改时 append）——原实现用 per_window_affected 恒为 1.0
        "coverage": round(float(len(per_window_span)) / n, 6) if n else 0.0,
        "series_coverage": round(len(modified_series) / n_series, 6)
        if n_series else 0.0,
        "series_fraction_std": round(float(np.std(series_means)), 6)
        if len(series_means) > 1 else 0.0,
        "cohort_aggregate_fraction": round(all_changed / all_points, 6)
        if all_points else 0.0,
    }
    return obs


def _scope_patch_candidates(
    pp_rows: Sequence[Mapping[str, Any]],
    other_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Runtime 从成功/冲突对照区间机械生成 ≤2 Scope Patch（用户修正 4）：
    特征在 pp 与 other 上区间分离 → 阈值 = 两区间端点 midpoint（非人工
    值）。op='<'（pp 更低）或 '>'（pp 更高）。附规范化分离裕度
    margin_ratio（跨特征可比——选择器用）。"""
    out: list[dict[str, Any]] = []
    for f in ALIGNED_FEATURES:
        if len(out) >= 2:
            break
        pp_vals = [float(r[f]) for r in pp_rows if r.get(f) is not None]
        ot_vals = [float(r[f]) for r in other_rows if r.get(f) is not None]
        if not pp_vals or not ot_vals:
            continue
        span = max([*pp_vals, *ot_vals]) - min([*pp_vals, *ot_vals])
        if max(pp_vals) < min(ot_vals):
            threshold = (max(pp_vals) + min(ot_vals)) / 2.0
            margin = min(ot_vals) - max(pp_vals)
            out.append({"feature": f, "op": "<", "threshold": threshold,
                        "pp_range": [min(pp_vals), max(pp_vals)],
                        "other_range": [min(ot_vals), max(ot_vals)],
                        "margin_ratio": round(margin / span, 4)
                        if span > 0 else 0.0})
        elif max(ot_vals) < min(pp_vals):
            threshold = (min(pp_vals) + max(ot_vals)) / 2.0
            margin = min(pp_vals) - max(ot_vals)
            out.append({"feature": f, "op": ">", "threshold": threshold,
                        "pp_range": [min(pp_vals), max(pp_vals)],
                        "other_range": [min(ot_vals), max(ot_vals)],
                        "margin_ratio": round(margin / span, 4)
                        if span > 0 else 0.0})
    return out


def _compile_applicability(card_sig: Mapping[str, object],
                           patch: Mapping[str, Any]) -> dict[str, object]:
    """Runtime 编译 Applicability：card 公开签名 + Scope Patch 条件
    （用户修正 2：Target-local——只在当前 Skill 上生效）。"""
    leaves: list[dict[str, object]] = []
    for k, v in card_sig.items():
        if isinstance(v, (str, int, float, bool)):
            leaves.append({"feature": str(k), "op": "==", "value": v})
    leaves.append({"feature": str(patch["feature"]), "op": str(patch["op"]),
                   "value": float(patch["threshold"])})
    return {"all": leaves}


def _replay_retrieval(applicability: Mapping[str, object],
                      obs: Mapping[str, float | int]) -> bool:
    """dev replay：给定 Context 的对齐观测下 applicability 是否匹配
    （= 该 Context 下 Skill 是否被检索）。ctx = 公开特征（task_kind）+
    program-effect 对齐观测——与部署时 fast 入口上下文同构。"""
    ctx = {"task_kind": "forecast"}
    ctx.update({k: float(v) for k, v in obs.items()
                if isinstance(v, (int, float))})
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        evaluate_applicability,
    )
    matched, _ = evaluate_applicability(applicability, ctx)
    return bool(matched)


def main() -> int:
    root = PROJECT_ROOT
    values = _series_values(root)
    points = [*_points_p41(root), *_points_p43(root),
              *_points_headroom(root)]
    for p in points:
        p["utility"] = _utility_class(
            float(p["support_gain"]),
            None if p.get("delayed_gain") is None
            else float(p["delayed_gain"]))

    # ---- a. 对齐真实执行窗口的观测（每点按其评估时的 cohort 重建 executor）----

    def _executor_for(series_name: str) -> ScopeExecutor:
        if series_name == "T117":
            rel = "w1_kdd2018_frozen_cohort_p41.jsonl"  # K1
        else:
            rel = "w1_kdd2018_frozen_cohort.jsonl"      # K0
        rows = [json.loads(line)
                for line in (root / "artifacts/functional/e2" / rel)
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        roster = [{"series_uid": str(r["series_name"]),
                   "role": str(r["role"])} for r in rows]
        vals = {str(r["series_name"]): values[str(r["series_name"])]
                for r in rows}
        return ScopeExecutor(roster, vals, _config(),
                             evaluate_fn=_evaluate_kdd)

    for p in points:
        executor = _executor_for(str(p["series"]))
        p["aligned"] = _aligned_observation(
            executor, int(p["origin"]), str(p["op"]))
        for f in ALIGNED_FEATURES:
            p[f] = p["aligned"][f]

    # ---- 共享锚点对照（审查 2026-08-11 决定性发现 2）：pp@888 与 np@984
    # 在共享锚点（≤840）上的对齐观测逐位相同而效用相反 → 表观分离全部
    # 来自窗口集合组成（@984 多出 anchor-852 组），非效用信号 ----
    exec_k1 = _executor_for("T117")
    shared_obs_888 = _aligned_observation(exec_k1, 888, "outlier_mad",
                                          max_anchor=840)
    shared_obs_984 = _aligned_observation(exec_k1, 984, "outlier_mad",
                                          max_anchor=840)
    shared_diff = {f: abs(float(shared_obs_888[f]) - float(shared_obs_984[f]))
                   for f in ALIGNED_FEATURES}
    shared_no_signal = bool(all(d == 0.0 for d in shared_diff.values()))
    shared_anchor_control = {
        "max_anchor": 840,
        "obs_pp_888": shared_obs_888,
        "obs_np_984": shared_obs_984,
        "max_abs_diff_per_feature": shared_diff,
        "identical_on_shared_anchors": shared_no_signal,
    }
    print(f"== shared-anchor control: identical={shared_no_signal} "
          f"max_diff={max(shared_diff.values())}")
    print("== aligned observations:")
    for p in points:
        a = p["aligned"]
        print(f"  {p['series']}@{p['origin']} {p['op']:<14} "
              f"class={p['utility']:<9} "
              f"aff_mean={a['window_affected_mean']} "
              f"aff_max={a['window_affected_max']} "
              f"clusters={a['window_clusters_mean']} "
              f"span_mean={a['window_span_mean']} span_max={a['window_span_max']} "
              f"cov={a['coverage']} s_cov={a['series_coverage']} "
              f"s_std={a['series_fraction_std']} agg={a['cohort_aggregate_fraction']}")

    # ---- b. Runtime 生成 ≤2 Scope Patch（成功 vs 冲突/其他）----
    om = [p for p in points if p["op"] == "outlier_mad"]
    om_pp = [p for p in om if p["utility"] == "pp"]
    om_other = [p for p in om if p["utility"] != "pp"]
    patches = _scope_patch_candidates(om_pp, om_other)
    print(f"== runtime scope patch candidates ({len(patches)}): "
          f"{json.dumps(patches)}")

    if not patches:
        print(json.dumps({
            "verdict": "ALIGNED_SIGNAL_LOST",
            "reason": "no aligned observation feature separates pp from "
                      "other on outlier_mad", }, indent=1))
        REPORT_REL.write_text(json.dumps({
            "experiment_id": "v1-kdd2018-scope-patch-mechanism",
            "note": "P4.5 development 机制切片（零新数据/零新 LLM）",
            "verdict": "ALIGNED_SIGNAL_LOST",
            "points": [{k: p[k] for k in
                        ("series", "origin", "op", "utility", "aligned")}
                       for p in points],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    # ---- c. 选择器（dev 确定性，模拟 Slow Agent 选择/abstain）----
    # 规范化裕度排序（跨特征可比——margin/观测全距）；abstain 条件：
    # 无候选 margin_ratio ≥ 0.5（阈值落在模糊区——证据不足，用户修正 3）。
    patches.sort(key=lambda p: float(p["margin_ratio"]), reverse=True)
    chosen = patches[0]
    if float(chosen["margin_ratio"]) < 0.5:
        verdict = "SCOPE_PATCH_SELECTION_ABSTAINED"
        reason = (f"best candidate {chosen['feature']} margin_ratio="
                  f"{chosen['margin_ratio']} < 0.5 — evidence insufficient "
                  f"to freeze a scope")
        print(json.dumps({"verdict": verdict, "reason": reason,
                          "candidates": patches}, indent=1))
        REPORT_REL.write_text(json.dumps({
            "experiment_id": "v1-kdd2018-scope-patch-mechanism",
            "note": "P4.5 development 机制切片（零新数据/零新 LLM）",
            "points": [{k: p[k] for k in
                        ("series", "origin", "op", "support_gain",
                         "delayed_gain", "utility", "aligned")}
                       for p in points],
            "shared_anchor_control": shared_anchor_control,
            "patches": patches,
            "verdict": verdict, "reason": reason,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    # ---- d. Runtime 编译 Applicability（Target-local）----
    card_sig = {"task_kind": "forecast"}  # P4.1 card 的公开签名
    applicability = _compile_applicability(card_sig, chosen)
    print(f"== compiled applicability: {json.dumps(applicability)}")

    # ---- e. 正/负 Context dev replay ----
    om_ctx = {p["origin"]: p for p in om}
    positive_origin = 888   # 双正 Context
    conflict_origin = 984   # 冲突 Context（不自动优先）
    nearzero_origin = 600   # 近零/双负 Context
    replays = {
        "positive_888": _replay_retrieval(
            applicability, om_ctx[positive_origin]["aligned"]),
        "conflict_984": _replay_retrieval(
            applicability, om_ctx[conflict_origin]["aligned"]),
        "nearzero_600": _replay_retrieval(
            applicability, om_ctx[nearzero_origin]["aligned"]),
    }
    print(f"== replays: {json.dumps(replays)}")
    checks = {
        "aligned_signal": bool(patches),
        "patch_generated_by_runtime": bool(
            patches and all(p["threshold"] == (max(p["pp_range"])
                                               + min(p["other_range"])) / 2.0
                            for p in patches)),
        "shared_anchor_no_signal": bool(shared_no_signal),
        "positive_context_retrieves": bool(replays["positive_888"]),
        "conflict_context_not_auto": bool(not replays["conflict_984"]),
        "nearzero_context_not_auto": bool(not replays["nearzero_600"]),
    }
    if all(checks.values()):
        verdict = "SCOPE_PATCH_MECHANISM_PASS"
        reason = (f"runtime-generated scope patch {chosen['feature']}"
                  f" {chosen['op']} {chosen['threshold']:.6g} compiles to "
                  f"applicability; positive context retrieves, conflict/"
                  f"near-zero contexts do not auto-prioritize")
    else:
        verdict = "SCOPE_PATCH_REPLAY_FAILED"
        reason = f"checks: {json.dumps(checks)}"
    print(f"== verdict: {verdict}")
    print(f"== reason: {reason}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-scope-patch-mechanism",
        "note": "P4.5 development 机制切片（零新数据/零新 LLM；阈值由 "
                "Runtime 从对照区间机械生成——Target-local，不写死人工值；"
                "通过只证机制，不追溯 fresh）",
        "points": [{k: p[k] for k in
                    ("series", "origin", "op", "support_gain",
                     "delayed_gain", "utility", "aligned")} for p in points],
        "shared_anchor_control": shared_anchor_control,
        "runtime_patch_candidates": patches,
        "chosen_patch": chosen,
        "compiled_applicability": applicability,
        "replays": replays,
        "checks": checks,
        "verdict": verdict,
        "reason": reason,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
