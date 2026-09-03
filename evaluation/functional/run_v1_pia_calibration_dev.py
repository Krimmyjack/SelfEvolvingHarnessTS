"""PIA_CALIBRATION_DEV（P2 失败原因分支 → P3，2026-08-13：Program
Impact Analysis——first-order Response Sketch 校准——development
exposure——零新 Claim——零 LLM——零新评估（全部 gold 来自已暴露报告）。

背景：P1 block2 闭包（BLOCK2_FAMILY_NO_HEADROOM——连续两个 family 无
headroom，用户 P1 停止条件触发）。按用户 P1-P6 自动推进序：P2 按失败
原因自动分支——失败原因 = NO_COMMON_PROGRAM_HEADROOM + 评估成本瓶颈
（今晚 100+ 完整 Consumer 评估）→ P3 PIA 校准：验证廉价 first-order
Response Sketch 能否预筛候选、减少 full-Support 评估量。

规格（SHIFT_REPORT NEXT_BRANCH #2 预注册）：Program ΔX/ΔY →
first-order Response Sketch vs gold——只验 top-k recall / sign
agreement / harmful FP / full-Support 减少量，**绝不接批准**（PIA 只做
筛选定位，批准权永远在 Support 实测）。

Gold（全部读已暴露报告——零新评估——全量纳入不挑 outcome）：
  1. census winsorize/outlier_mad probes：wave3 24 行 + block2 20 行
     （cand_winsorize + cand_outlier_mad——含 gain=0 行）
  2. replacement headroom：wave3 family 18 行（3 ops × 6 窗——含
     winsorize 自替代行）+ block2 family 6 行（× 3 窗）——steps 用
     contract_params 重建（与 census 装置一致——参数来自代码常量非
     outcome）
  3. supply search：54 行（9 候选 × 6 窗口——steps 从报告直读）
  合计 122 行 (steps, series, origin, gold_gain)。（审计修正
  2026-08-13：docstring 原枚举 92 行/block2 4 窗口/headroom 12+6 为
  计数错误——实际 122 行/20 窗口/18+6；选择规则本身合规——全量纳入。）

Sketch（first-order 定义——程序干预幅度）：对 series[:origin] 顺序
应用 steps 的算子函数（operators.s1_outlier 直接 numpy 调用——不跑
verifier/编译/下游模型——sketch 的廉价性即在此），δ = x' − x：
  modified_fraction = |{i: |δ_i| > 1e-9}| / n
  delta_mean = mean(δ)（有符号——sign 预测的来源）
  norm_mag = mean(|δ|) / std(window)（归一化干预幅度——ranking 来源）
恒等式（机械安全门）：modified_fraction=0 ⇒ x'=x ⇒ 下游模型输入不变
⇒ gain=0——R0 跳过零风险（T105 outlier_mad gain=0.0 与无修改一致）。

指标（预注册——SHIFT_REPORT 四项）：
  - sign agreement：material 行（|gold| ≥ M）且 modified_fraction>0
    上 sign(delta_mean)==sign(gold) 比例（随机基线 0.5）
  - top-k recall（k=1,2）：每窗口内候选按 norm_mag 排名，top-k 含该
    窗口 gold best（max gain ≥M）的比例——只在候选集 ≥2 的窗口计
    （单候选窗口恒真——不污染）；随机基线 = k/n_candidates
  - harmful FP：sign 预测正（delta_mean>0）但 gold ≤ −M 的行数/率
  - full-Support 减少量：R0（modified_fraction=0 → skip）+ Rτ
    （norm_mag < τ → skip，τ ∈ {1e-4,1e-3,3e-3,1e-2,3e-2,1e-1}）的
    saved / missed-positive（被 skip 且 gold ≥M）/ skipped-harmful
    （被 skip 且 gold ≤ −M——"省得对"）曲线

verdict（预注册）：
  PIA_SKETCH_SAFE_SCREENING_USEFUL  : 安全门过（R0 零 missed-
    positive）且（R0 ∪ Rτ(1e-2)）省 ≥10% 且 top-2 recall ≥ 1.5×
    随机基线
  PIA_SKETCH_SAFE_SCREENING_MARGINAL: 安全门过但未达上面
  PIA_SKETCH_SAFETY_VIOLATION       : R0 存在 |gold|≥M 的零修改行
    （机械恒等式被破坏——装置/读数 bug 须调查）
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_pia_calibration_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.operators.s1_outlier import (  # noqa: E402
    hampel_filter,
    outlier_mad,
    winsorize,
)

M = resolver.MATERIAL_THRESHOLD
PERIOD = 24
E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_pia_calibration_dev_report.json"
OPS_FN = {"winsorize": winsorize, "outlier_mad": outlier_mad,
          "hampel_filter": hampel_filter}
TAUS = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"


def _load_series(root: Path, uid: str) -> np.ndarray:
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index(uid)], dtype=np.float64)


def _apply_steps(series_arr: np.ndarray, origin: int,
                 steps: Sequence[Sequence[Any]]) -> np.ndarray:
    """first-order sketch 的程序应用：算子函数直接作用训练窗口
    （顺序 compose——与编译 pipeline 的差异正是 sketch 的廉价性）。"""
    y = series_arr[:origin].astype(np.float64)
    for op, params in steps:
        fn = OPS_FN[op]
        y = fn(y, **dict(params))
    return y


def _sketch(series_arr: np.ndarray, origin: int,
            steps: Sequence[Sequence[Any]]) -> dict[str, float]:
    x = series_arr[:origin].astype(np.float64)
    y = _apply_steps(series_arr, origin, steps)
    delta = y - x
    n = max(x.size, 1)
    std = float(np.std(x))
    if std < 1e-12:
        std = 1.0
    return {
        "modified_fraction": float(np.mean(np.abs(delta) > 1e-9)),
        "delta_mean": float(np.mean(delta)),
        "delta_mean_abs": float(np.mean(np.abs(delta))),
        "norm_mag": float(np.mean(np.abs(delta)) / std),
    }


def _assemble_gold(root: Path) -> tuple[list[dict[str, Any]],
                                        dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    notes: dict[str, Any] = {"sources": {}}

    census = json.loads((E2 / "w1_batch_census_dev_report.json")
                        .read_text(encoding="utf-8"))
    b2 = json.loads((E2 / "w1_block2_census_ec_dev_report.json")
                    .read_text(encoding="utf-8"))
    supply = json.loads((E2 / "w1_program_supply_dev_report.json")
                        .read_text(encoding="utf-8"))

    # 1. census probes（wave3 + block2——winsorize/outlier_mad 全行）
    n_probe = 0
    for rep, tag in ((census, "wave3"), (b2, "block2")):
        for sid, rounds in (rep.get("development_rounds") or {}).items():
            for r in rounds:
                for cid, gain in r.get("probes") or []:
                    if gain is None:
                        continue
                    op = cid.replace("cand_", "")
                    if op not in OPS_FN:
                        continue
                    steps = ((op, dict(wiring.contract_params(op,
                                                             PERIOD))),)
                    rows.append({"source": f"census_probe_{tag}",
                                 "series": sid, "origin": r["origin"],
                                 "steps": steps, "gain": float(gain)})
                    n_probe += 1
    notes["sources"]["census_probes"] = n_probe

    # 2. replacement headroom（wave3 + block2——contract_params 重建）
    n_hr = 0
    for rep, tag in ((census, "wave3"), (b2, "block2")):
        fams = rep.get("development_families") or []
        if not fams:
            continue
        hr = fams[0].get("replacement_headroom") or {}
        for alt, block in hr.items():
            if alt not in OPS_FN:
                continue
            steps = ((alt, dict(wiring.contract_params(alt, PERIOD))),)
            for e in block.get("per_episode_gains") or []:
                if e.get("gain") is None:
                    continue
                rows.append({"source": f"headroom_{tag}",
                             "series": e["series"], "origin": e["origin"],
                             "steps": steps, "gain": float(e["gain"])})
                n_hr += 1
    notes["sources"]["headroom"] = n_hr

    # 3. supply search（steps 直读）
    n_sup = 0
    for s in supply.get("search") or []:
        steps = tuple((op, dict(params))
                      for op, params in s.get("steps") or [])
        for e in s.get("per_window_gains") or []:
            if e.get("gain") is None:
                continue
            rows.append({"source": f"supply_{s.get('label')}",
                         "series": e["series"], "origin": e["origin"],
                         "steps": steps, "gain": float(e["gain"])})
            n_sup += 1
    notes["sources"]["supply"] = n_sup
    return rows, notes


def main() -> int:
    root = PROJECT_ROOT
    rows, notes = _assemble_gold(root)
    if not rows:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no gold rows"}, indent=1))
        return 0
    # sketch 计算（零新评估——只读 series + 算子应用）
    for r in rows:
        series_arr = _load_series(root, r["series"])
        try:
            r["sketch"] = _sketch(series_arr, r["origin"], r["steps"])
        except Exception as exc:  # noqa: BLE001
            r["sketch"] = {"error": f"{type(exc).__name__}: {exc}"}
    sketched = [r for r in rows if "error" not in (r.get("sketch") or {})]
    if not sketched:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "all sketches failed"}, indent=1))
        return 0

    def _sign(v: float) -> int:
        return 1 if v >= M else (-1 if v <= -M else 0)

    # ---- sign agreement（material 行且 modified_fraction>0）----
    sign_rows = [r for r in sketched
                 if r["sketch"]["modified_fraction"] > 1e-9
                 and abs(r["gain"]) >= M]
    agree = sum(1 for r in sign_rows
                if _sign(r["sketch"]["delta_mean"]) == _sign(r["gain"]))
    sign_agreement = (agree / len(sign_rows)) if sign_rows else None

    # ---- harmful FP（预测正但 gold 有害）----
    harmful_fp = [r for r in sketched
                  if r["sketch"]["delta_mean"] > 0 and r["gain"] <= -M]
    # ---- benign FN（预测非正但 gold 正——信息补充）----
    benign_fn = [r for r in sketched
                 if r["sketch"]["delta_mean"] <= 0 and r["gain"] >= M]

    # ---- top-k recall（窗口内候选按 norm_mag 排名）----
    windows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in sketched:
        windows.setdefault((r["series"], r["origin"]), []).append(r)
    recall: dict[int, float] = {}
    random_baseline: dict[int, float] = {}
    n_windows_used = 0
    for key, cands in windows.items():
        n = len(cands)
        if n < 2:
            continue  # 单候选窗口恒真——不污染
        best = max(r["gain"] for r in cands)
        if best < M:
            continue  # 窗口无 gold positive——recall 无靶
        ranked = sorted(cands, key=lambda r: -r["sketch"]["norm_mag"])
        n_windows_used += 1
        for k in (1, 2):
            hit = any(r["gain"] == best for r in ranked[:k])
            recall[k] = recall.get(k, 0) + (1 if hit else 0)
            random_baseline[k] = random_baseline.get(k, 0) + min(k, n) / n
    for k in (1, 2):
        recall[k] = (recall.get(k, 0) / n_windows_used
                     if n_windows_used else None)
        random_baseline[k] = (random_baseline.get(k, 0) / n_windows_used
                              if n_windows_used else None)

    # ---- full-Support 减少量（R0 + Rτ sweep）----
    n_total = len(sketched)
    n_zero = sum(1 for r in sketched
                 if r["sketch"]["modified_fraction"] <= 1e-9)
    zero_miss = [r for r in sketched
                 if r["sketch"]["modified_fraction"] <= 1e-9
                 and abs(r["gain"]) >= M]
    r0 = {"skipped": n_zero, "saved_fraction": n_zero / n_total,
          "missed_positive": sum(1 for r in zero_miss if r["gain"] >= M),
          "skipped_harmful": sum(1 for r in zero_miss
                                 if r["gain"] <= -M)}
    tau_table = []
    for tau in TAUS:
        skipped = [r for r in sketched if r["sketch"]["norm_mag"] < tau]
        tau_table.append({
            "tau": tau,
            "skipped": len(skipped),
            "saved_fraction": len(skipped) / n_total,
            "missed_positive": sum(1 for r in skipped if r["gain"] >= M),
            "skipped_harmful": sum(1 for r in skipped if r["gain"] <= -M),
        })

    # ---- 判定（预注册）----
    safety_ok = r0["missed_positive"] == 0
    # 审计修复 4（2026-08-13）：saved_ok 按预注册口径取 R0 ∪ Rτ(1e-2)
    # 并集（原为双计——R0 ⊂ Rτ 因零修改行 norm_mag=0）
    union_saved = len({i for i, r in enumerate(sketched)
                       if r["sketch"]["modified_fraction"] <= 1e-9
                       or r["sketch"]["norm_mag"] < 1e-2}) / n_total
    saved_ok = union_saved >= 0.10
    recall_ok = bool(recall.get(2) is not None
                     and recall[2] >= 1.5 * (random_baseline.get(2) or 1.0))
    if not safety_ok:
        verdict = "PIA_SKETCH_SAFETY_VIOLATION"
    elif saved_ok and recall_ok:
        verdict = "PIA_SKETCH_SAFE_SCREENING_USEFUL"
    else:
        verdict = "PIA_SKETCH_SAFE_SCREENING_MARGINAL"

    report = {
        "experiment_id": "v1-pia-calibration-dev",
        "note": "P2 失败原因分支 → P3 PIA 校准：first-order Response "
                "Sketch（程序干预幅度——算子直接应用）vs 已暴露 gold"
                "——只做筛选定位，绝不接批准——development exposure"
                "——零新 Claim——零 LLM——零新评估",
        "gold": {"n_rows": len(rows), "n_sketched": len(sketched),
                 **notes},
        "metrics": {
            "sign_agreement": sign_agreement,
            "sign_agreement_n": len(sign_rows),
            "harmful_fp": len(harmful_fp),
            "harmful_fp_rate": (len(harmful_fp) / len(sketched)
                                if sketched else None),
            "benign_fn": len(benign_fn),
            "topk_recall": recall,
            "topk_random_baseline": random_baseline,
            "n_windows_used": n_windows_used,
        },
        "screening": {"r0": r0, "tau_sweep": tau_table},
        "harmful_fp_rows": [
            {"series": r["series"], "origin": r["origin"],
             "source": r["source"], "steps": r["steps"],
             "gain": r["gain"], "sketch": r["sketch"]}
            for r in harmful_fp],
        "verdict": verdict,
    }
    print("== sign_agreement:", sign_agreement, "n=", len(sign_rows))
    print("== topk recall:", recall, "random:", random_baseline,
          "windows:", n_windows_used)
    print("== harmful_fp:", len(harmful_fp), "/", len(sketched))
    print("== r0:", json.dumps(r0))
    print("== tau sweep:", json.dumps(tau_table))
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
