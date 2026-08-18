"""KDD_CUP_2018_PROGRAM_EFFECT_CONTEXT_DIAGNOSTIC（P4.4，用户裁决
2026-08-11：先诊断、不盲目换 virgin 数据）。

first fault（用户裁决，P4.3 之后）：
  PROGRAM_CONDITIONED_SCOPE_OBSERVATION_INSUFFICIENT——A+B 修复后
  applicability 只剩 card 的 task_kind==forecast（可达但区分力弱）；
  @984 实测：Skill 被检索/执行/removal 恢复 ✓，但 outlier_mad
  Support −0.0608（负）、delayed +0.2567（正）——Support/delayed 冲突。

本诊断回答（零新数据/零新 LLM——只读已暴露报告 + 已暴露窗口的程序
应用几何）：
  "部署时可见的 Program effect（候选作用几何）能否区分 outlier_mad 的
  效用翻转（双正 / 近零或双负 / Support 负-delayed 正冲突）？"

三类已知 Context（全部来自已测点——报告提取，不重测）：
  1. 双正      : outlier_mad @888  T117  sg +0.1199 / dg +0.1113（P4.1）
  2. 近零/双负 : outlier_mad @600  K0    sg −0.0014 / dg −0.1175（P2）
  3. 冲突      : outlier_mad @984  T117  sg −0.0608 / dg +0.2567（P4.3）
  对照（同几何、跨算子泛化检查）：winsorize @600/@792（双正 T117）、
  @888（双负 T117）、@984（冲突 T117）、@600（冲突 K0）、hampel @600
  （双负 K0）。

候选作用几何（对训练窗口 [0, origin) 应用程序——部署时对候选可算）：
  affected_fraction / 修改点簇数 / 最大连续修改区间占比 / 修改跨度占比 /
  修改幅度 vs 原序列 robust scale / center 变化 / scale 变化。

判定（预注册）：
  OBSERVABLE_SCOPE_SEPARATION_FOUND : 存在特征 f——outlier_mad 三点按
    类分离（双正 vs 其余区间不重叠）**且** winsorize 对照点类间方向一致
    （跨算子泛化）**且**方向可解释 → 冻结最小 Scope 用该特征（P4.5）。
  UNIDENTIFIABLE_WITH_CURRENT_OBSERVATIONS : 其他——不消费 fresh 数据；
    Skill 保持 LOCAL_DRAFT（每次 Support，不自动优先）。

用法：
  python evaluation/functional/run_v1_kdd2018_program_effect_context_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

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

PERIOD = 24
HORIZON = 48
M = 0.005  # MATERIAL_THRESHOLD
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_kdd2018_program_effect_context_diagnostic_report.json"
FEATURES = ("affected_fraction", "n_clusters", "max_run_fraction",
            "span_fraction", "mag_ratio", "center_change", "scale_change")


def _series_values(root: Path) -> dict[str, np.ndarray]:
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return {n: np.asarray(values[i], dtype=np.float64)
            for i, n in enumerate(names)}


def _k0_first_series(root: Path) -> str:
    row = json.loads(next(
        line for line in (root / "artifacts/functional/e2"
                          / "w1_kdd2018_frozen_cohort.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()))
    return str(row["series_name"])


def _points_p41(root: Path) -> list[dict[str, Any]]:
    """P4.1 报告（T117）探测点：A3 臂 R1/R2 winsorize、R3 p1 winsorize +
    p2 outlier_mad（含 delayed）。"""
    rep = json.loads((root / "artifacts/functional/e2"
                      / "w1_kdd2018_cross_domain_a5_a3_report.json")
                     .read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    a3 = rep["arms"]["A3"]["rounds"]
    for rd in a3:
        for p in rd["probes"]:
            out.append({
                "series": "T117", "origin": rd["origin"], "op": p["op"],
                "support_gain": p["gain"], "delayed_gain": p["delayed_gain"],
                "source": "p41", })
    return out


def _points_p43(root: Path) -> list[dict[str, Any]]:
    """P4.3 报告（T117@984）：ADOPT 执行 outlier_mad、REMOVE 执行
    winsorize（binding replay 报告——不是 P4.2 adoption 报告）。"""
    rep = json.loads((root / "artifacts/functional/e2"
                      / "w1_kdd2018_applicability_binding_replay_report.json")
                     .read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for arm, op in (("ADOPT", "outlier_mad"), ("REMOVE", "winsorize")):
        a = rep["arms"][arm]
        out.append({
            "series": "T117", "origin": rep["origin"], "op": op,
            "support_gain": a["support_gain"],
            "delayed_gain": a.get("delayed_gain"),
            "source": "p43", })
    return out


def _points_headroom(root: Path) -> list[dict[str, Any]]:
    rep = json.loads((root / "artifacts/functional/e2"
                      / "w1_kdd2018_headroom_diagnosis_report.json")
                     .read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for row in rep["results"]:
        if row.get("support_gain") is None:
            continue
        out.append({
            "series": _k0_first_series(root), "origin": rep["origin"],
            "op": row["op"], "support_gain": row["support_gain"],
            "delayed_gain": row["delayed_gain"], "source": "headroom"})
    return out


def _utility_class(sg: float, dg: float | None) -> str:
    """效用类（M=0.005）：pp=双正 / nn=双负 / np=support负-delayed正
    （冲突）/ pn=support正-delayed负 / near_zero=support近零。"""
    if dg is None:
        return "no_delayed"
    if abs(sg) < M:
        return "near_zero"
    if sg >= M and dg >= M:
        return "pp"
    if sg <= -M and dg <= -M:
        return "nn"
    if sg <= -M and dg >= M:
        return "np"
    if sg >= M and dg <= -M:
        return "pn"
    return "mixed"


def _steps_for(op: str) -> tuple[tuple[str, dict], ...]:
    return ((op, dict(wiring.contract_params(op, PERIOD))),)


def _geometry(series: np.ndarray, origin: int,
              op: str) -> dict[str, float | int]:
    s0 = np.asarray(series[:origin], dtype=np.float64)
    steps = _steps_for(op)
    program = Program.from_steps(list(steps), source="p44_diagnostic")
    candidate = Candidate(candidate_id="p44_diagnostic",
                          kind=CandidateKind.PROGRAM,
                          program=program, source="p44_diagnostic")
    compiled = CompiledWorkflow(candidate, (), tuple(program.steps))
    after, _trace = v6._apply_program(s0, compiled)
    changed = np.nonzero(np.abs(after - s0) > 1e-12)[0]
    n = int(s0.size)
    geo: dict[str, float | int] = {"n_changed": int(changed.size)}
    if changed.size == 0:
        geo.update({f: 0.0 for f in FEATURES})
        return geo
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
    robust_scale = 1.4826 * float(np.median(np.abs(
        s0 - np.median(s0))))
    delta = np.abs(after - s0)[changed]
    geo["affected_fraction"] = round(changed.size / n, 6)
    geo["n_clusters"] = len(runs)
    geo["max_run_fraction"] = round(max(runs) / n, 6)
    geo["span_fraction"] = round((int(changed[-1]) - int(changed[0]) + 1)
                                 / n, 6)
    geo["mag_ratio"] = (round(float(np.median(delta)) / robust_scale, 6)
                        if robust_scale > 0 else 0.0)
    geo["center_change"] = round(float(np.median(after)
                                       - np.median(s0)), 6)
    mad0 = float(np.median(np.abs(s0 - np.median(s0))))
    mad1 = float(np.median(np.abs(after - np.median(after))))
    geo["scale_change"] = round(mad1 / mad0 - 1.0, 6) if mad0 > 0 else 0.0
    return geo


def _separates(values_a: Sequence[float], values_b: Sequence[float]) -> bool:
    """两组取值区间不重叠（max(a) < min(b) 或 max(b) < min(a)）。"""
    if not values_a or not values_b:
        return False
    return bool(max(values_a) < min(values_b)
                or max(values_b) < min(values_a))


def main() -> int:
    root = PROJECT_ROOT
    values = _series_values(root)
    points = [*_points_p41(root), *_points_p43(root),
              *_points_headroom(root)]
    for p in points:
        series = values[str(p["series"])]
        p["steps"] = list(_steps_for(str(p["op"])))
        p["geometry"] = _geometry(series, int(p["origin"]), str(p["op"]))
        p["utility"] = _utility_class(
            float(p["support_gain"]),
            None if p.get("delayed_gain") is None
            else float(p["delayed_gain"]))
        for f in FEATURES:
            p[f] = p["geometry"][f]
    print("== points:")
    for p in points:
        dg = p["delayed_gain"]
        dg_s = "None" if dg is None else f"{float(dg):+.4f}"
        print(f"  {p['series']}@{p['origin']} {p['op']:<14} "
              f"sg={float(p['support_gain']):+.4f} dg={dg_s}  "
              f"class={p['utility']:<9} "
              f"aff={p['affected_fraction']} clusters={p['n_clusters']} "
              f"maxrun={p['max_run_fraction']} span={p['span_fraction']} "
              f"mag={p['mag_ratio']} center={p['center_change']} "
              f"scale={p['scale_change']}")

    # ---- 分离性检查（预注册）----
    om = [p for p in points if p["op"] == "outlier_mad"]
    om_pp = [p for p in om if p["utility"] == "pp"]
    om_other = [p for p in om if p["utility"] != "pp"]
    wins = [p for p in points if p["op"] == "winsorize"]
    wins_pp = [p for p in wins if p["utility"] == "pp"]
    wins_other = [p for p in wins if p["utility"] != "pp"]
    checks: dict[str, Any] = {}
    candidates: list[str] = []
    for f in FEATURES:
        om_sep = _separates([float(p[f]) for p in om_pp],
                            [float(p[f]) for p in om_other])
        w_sep = _separates([float(p[f]) for p in wins_pp],
                           [float(p[f]) for p in wins_other])
        # 跨算子方向一致：pp 在 outlier_mad 上相对 other 的方向 == winsorize 上
        om_dir = (np.median([float(p[f]) for p in om_pp])
                  > np.median([float(p[f]) for p in om_other]))
        w_dir = (np.median([float(p[f]) for p in wins_pp])
                 > np.median([float(p[f]) for p in wins_other]))
        consistent = bool(om_dir == w_dir)
        checks[f] = {"om_separates": om_sep, "winsorize_separates": w_sep,
                     "direction_consistent": consistent,
                     "om_pp": [float(p[f]) for p in om_pp],
                     "om_other": [float(p[f]) for p in om_other],
                     "wins_pp": [float(p[f]) for p in wins_pp],
                     "wins_other": [float(p[f]) for p in wins_other]}
        if om_sep and consistent:
            candidates.append(f)
    print(f"== feature candidates: {candidates}")

    if candidates:
        verdict = "OBSERVABLE_SCOPE_SEPARATION_FOUND"
        reason = (f"features with outlier_mad class separation + winsorize "
                  f"direction consistency: {candidates} — **candidate "
                  f"identification level**（审查 2026-08-11 附条件裁定）："
                  f"om_pp 仅 1 点、两候选特征同族、winsorize 逐点不可分；"
                  f"阈值冻结门控于 P4.5 dev replay 验证之后")
    else:
        verdict = "UNIDENTIFIABLE_WITH_CURRENT_OBSERVATIONS"
        reason = ("no geometry feature separates utility classes on both "
                  "outlier_mad (3 pts) and winsorize (4 pts)")
    print(f"== verdict: {verdict}")
    print(f"== reason: {reason}")

    caveats = [
        "om_pp 仅 1 点（T117@888）——分离是 1-vs-2；7 特征无多重比较校正，"
        "单点分离误报风险高",
        "winsorize 逐点不可分（winsorize_separates=false）：T1@600 np 是"
        "最局部化的 winsorize 点（span 0.79/cl 15 均低于全部 pp 值）——"
        "span/n_clusters 上不存在能正确分类 winsorize 点的阈值；跨算子"
        "泛化仅为中位数方向一致",
        "n_clusters 与 span_fraction 是同一 localization 信号族（9 点上"
        "近乎共线）——非两次独立确认；候选特征应表述为单一信号族",
        "系列内反例：T117 winsorize 600→888→984 效用翻转（pp→nn→np）"
        "而几何几乎不变（aff≈0.09 恒定）——几何-效用共变主要见于 om 侧，"
        "且与算子固有行为（outlier_mad 只改少量局部点）混叠，不可解读为"
        "'作用几何充分区分效用'",
        "几何作用域为 [0, origin) 整前缀，评估应用为逐 anchor 窗口"
        "（[anchor−336, anchor+48]）——几何是部署时代理量，非评估复现",
        "hampel @600 已计算但未参与分离检查（并入 other 不影响结论）",
        "near_zero 仅指 support 近零：T1@600 om 的 delayed 为 material 负"
        "（−0.1175）——类名不得误读为'无效果'",
        "未冻结任何阈值——本次判定只提名候选特征；阈值冻结严格门控于"
        "P4.5 dev replay（扩展系列/origin 网格、跨算子逐点验证）之后",
    ]

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-program-effect-context-diagnostic",
        "note": "P4.4 development 诊断（零新数据/零新 LLM——只读已暴露报告 "
                "与已暴露窗口的程序应用几何；不消费 virgin）",
        "material_threshold": M,
        "points": [{
            "series": p["series"], "origin": p["origin"], "op": p["op"],
            "support_gain": p["support_gain"],
            "delayed_gain": p["delayed_gain"],
            "utility": p["utility"], "source": p["source"],
            "geometry": p["geometry"],
        } for p in points],
        "separation_checks": checks,
        "candidate_features": candidates,
        "verdict": verdict,
        "reason": reason,
        "caveats": caveats,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
