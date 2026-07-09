# S0.6a 主表重锚 — decision（修复后新基线）

日期：2026-07-02
脚本：`run_main_table --task forecast --seeds 2 --n-per-signal 4`（judge=chronos，reporter=lstm_scratch+dlinear_scratch ⟂ judge）
在 **S0.1(F1)+S0.2(F2)+S0.3(F3)+S0.4+S0.5 修复后**的 splits 上跑（final_test 现为 series_uid 分组、基底不泄漏）。

## 结果（4 forecast cells，final_test≥4）

| cell | oracle ΔPerf | winner |
|---|---:|---|
| forecast\|snrHigh\|full | +0.049 | v_stl |
| forecast\|snrHigh\|miss | +0.327 | v_median |
| forecast\|snrLow\|full  | +0.035 | v_stl |
| forecast\|snrLow\|miss  | +0.068 | v_stl |

- **per-cell oracle mean ΔPerf = +0.120**
- **single-best-global = v_median（mean ΔPerf +0.086）**
- **oracle − single-best gap = +0.033 = 条件化的价值（C1 信号）**
- 参照：minimal mean = −0.005；degraded mean = −0.039
- winner 非同质：snrHigh|miss 强偏好 v_median（+0.327），其余 cell 偏好 v_stl → 初步支持 per-cell 异质性存在（严格良定性判定留 E-1.1）。

## D-0.1 判据评估

- 判据：若修复前后 headline 数字变化 > 20%，后续只引用修复后数字。
- **repo 内无留存的修复前 results/ 产物**（glob 已确认，见 plan §一.6 事实核实）→ 无法做逐数字漂移比较。
- **裁决**：本次 = 修复后 canonical 基线。此前文档/memory 中的 C1/oracle 旧数值一律标记为**修复前口径、仅供追溯**，不与本表混用。所有后续 Stage 1/2 对比实验以本 anchor 为锚。

## 重要说明（进 E-7.2 危害证据档）

- oracle−single-best gap = +0.033 只是 7 变体凸包内的条件化价值；是否**统计良定**（超出 series 抽样噪声）由 **E-1.1**（group bootstrap by series_uid）裁决——本表 gap 接近 EPS_NARROW(0.03) 量级，良定性不能从本表直接断言。
- snrHigh|miss 的 +0.327 量级远大于其他 cell（与 thresholds.py:17 "趋势 cell 效应尺度 10–28×" 线索一致）→ 提示 per-cell ε / 效应尺度异质，E-6.1 前置 se_Δ 表须 per-cell。
