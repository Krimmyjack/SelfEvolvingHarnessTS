"""PUBLIC_LEVEL_EXCURSION_BINDING_PREMISE（用户裁决 2026-08-10）。

默认 repair_level_shift 被 H0 拒绝（修改 45.7% > 0.35）不是"没有 headroom"——
算子支持显式局部区域（region_start_fraction/region_end_fraction/estimated_
offset），公开特征提取器（extract_public_features）产生对应字段。验证
"局部几何绑定"是否工作：

  第一步（只观察，不读 outcome）：UCI offset=40 三决策点（648/744/840）
    调用 extract_public_features(series[:origin])：
    - 识别到有起点终点的局部 excursion（level_mask 非空、start < end）
    - 区域宽度 ≤ 0.35（H0 修改分数约束）
    - excursion 在窗口结束前返回（level 区域后仍有 post 段）
    - estimated offset 非零
    - level-excursion evidence 可靠（score > 0）
  第二步（绑定局部 Program）：参数全来自公开 mapping
    repair_level_shift(region_start_fraction=estimated_region_start_fraction,
                       region_end_fraction=estimated_region_end_fraction,
                       estimated_offset=estimated_level_offset)
    不允许手工指定区域/offset。
  第三步（verifier 过才测 headroom）：ScopeExecutor.evaluate（H0 verifier
    + Support）@origin → 冻结 → delayed @origin+HORIZON。

Verdict（预注册，每决策点 + cohort 级）：
  NO_BOUNDED_EXCURSION_OBSERVED（无闭合局部 excursion → 停止方向，
    LLM abstain 正确）
  BOUND_REGION_TOO_WIDE（区域 > 0.35 → H0 无合法局部修复权限，不放宽）
  BOUND_CANDIDATE_VERIFIER_PASS_NO_HEADROOM（局部 Program 合法但
    Support/delayed 无改善——这时才说该 Context 无 level-shift headroom）
  BOUND_CANDIDATE_DELAYED_STABLE_HEADROOM（合法且双正 → 进下一实验：
    CandidatePool [denoise_median, bound repair] LLM 选择测试）

development 数据（已暴露），不称 fresh。

用法：
  python evaluation/functional/run_v1_public_level_excursion_binding_premise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.runtime.public_features import extract_public_features  # noqa: E402

DOMAIN = "uci_electricity_load_diagrams"
OFFSET = 40
PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
MAX_FRACTION = 0.35  # H0 max_modified_fraction（部署约束）
ORIGINS = (648, 744, 840)
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_public_level_excursion_binding_premise_report.json")


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(root, offset=OFFSET)
    series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                         dtype=np.float64)
    executor = ScopeExecutor(tgt_roster, tgt_values, config,
                             evaluate_fn=sealed.v6._evaluate)

    rows: list[dict[str, object]] = []
    cohort_has_stable_headroom = False
    for origin in ORIGINS:
        row: dict[str, object] = {"origin": origin}
        # ---- 第一步：公开 Observation（只观察，不读 outcome）----
        fe = extract_public_features(series0[:origin], task_kind="forecast")
        mapping = dict(fe.mapping)
        s_frac = float(mapping["estimated_region_start_fraction"])
        e_frac = float(mapping["estimated_region_end_fraction"])
        offset = float(mapping["estimated_level_offset"])
        score = float(mapping["level_excursion_score"])
        level_idx = np.flatnonzero(fe.level_mask)
        n = int(series0[:origin].size)
        l_start = float(level_idx[0] / n) if level_idx.size else None
        l_end = float((level_idx[-1] + 1) / n) if level_idx.size else None
        row["observation"] = {
            "region_start_fraction": s_frac,
            "region_end_fraction": e_frac,
            "region_width": e_frac - s_frac,
            "estimated_level_offset": offset,
            "level_excursion_score": score,
            "level_mask_start_fraction": l_start,
            "level_mask_end_fraction": l_end,
            "level_mask_present": bool(level_idx.size),
        }
        # 观察判定
        closed_excursion = bool(
            level_idx.size and s_frac < e_frac and offset != 0.0
            and score > 0.0)
        width_ok = (e_frac - s_frac) <= MAX_FRACTION
        returns_before_end = bool(
            l_end is not None and l_end < 1.0)
        row["observation_checks"] = {
            "closed_excursion": closed_excursion,
            "width_le_0.35": width_ok,
            "returns_before_window_end": returns_before_end,
            "offset_nonzero": bool(offset != 0.0),
            "evidence_score_positive": bool(score > 0.0),
        }
        print(f"== @{origin}: region=[{s_frac:.4f},{e_frac:.4f}] "
              f"width={e_frac - s_frac:.4f} offset={offset:.4f} "
              f"score={score:.3f} level_mask=({l_start},{l_end})")

        # ---- 第二/三步：绑定局部 Program → verifier → Support → delayed ----
        if not closed_excursion:
            row["verdict"] = "NO_BOUNDED_EXCURSION_OBSERVED"
        elif not width_ok:
            row["verdict"] = "BOUND_REGION_TOO_WIDE"
        else:
            steps = (("repair_level_shift", {
                "region_start_fraction": s_frac,
                "region_end_fraction": e_frac,
                "estimated_offset": offset,
            }),)
            rs = executor.evaluate(steps, origin)
            gs = (float(rs.gain) if rs.gain is not None else None)
            passed = bool(rs.verification.passed)
            rd = executor.evaluate(steps, origin + HORIZON)
            gd = (float(rd.gain) if rd.gain is not None else None)
            row["bound_candidate"] = {
                "steps": [{"op": "repair_level_shift", "params": {
                    "region_start_fraction": s_frac,
                    "region_end_fraction": e_frac,
                    "estimated_offset": offset}}],
                "verifier_passed": passed,
                "support_gain": gs,
                "delayed_gain": gd,
            }
            if not passed:
                row["verdict"] = "BOUND_REGION_TOO_WIDE"
            elif (gs is not None and gs >= M
                  and gd is not None and gd >= -M):
                row["verdict"] = "BOUND_CANDIDATE_DELAYED_STABLE_HEADROOM"
                cohort_has_stable_headroom = True
            else:
                row["verdict"] = "BOUND_CANDIDATE_VERIFIER_PASS_NO_HEADROOM"
            print(f"== @{origin}: bound candidate verifier={passed} "
                  f"support={gs} delayed={gd} -> {row['verdict']}")
        rows.append(row)

    # cohort 级判定：任一决策点稳定 headroom → 进下一实验
    verdict = ("BOUND_CANDIDATE_DELAYED_STABLE_HEADROOM"
               if cohort_has_stable_headroom
               else "NO_BOUNDED_EXCURSION_OBSERVED")
    print(f"== cohort verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-public-level-excursion-binding-premise",
        "dataset": DOMAIN, "cohort_offset": OFFSET,
        "origins": list(ORIGINS),
        "material_threshold": M, "max_fraction": MAX_FRACTION,
        "points": rows,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
