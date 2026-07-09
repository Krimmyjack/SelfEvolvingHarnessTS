# E-1.1 v2.1 — S0.7-8 边界修复后完整重跑（v2↔v2.1 对照；A-31b/A-32a）

日期：2026-07-03
脚本：`run_variance_decomp.py --n-seeds 20 --boot 300 --perm 500 --out results/E1_1_v2_1`（1397.6s）
前置：**S0.7-8 边界语义修复**（`denoise_median`：零填充 medfilt → `ndimage.median_filter(mode="reflect")`；
`moving_average`：convolve mode="same" → symmetric+valid；savgol 显式 `interp`、wavelet 显式 `symmetric`）
+ S0.7-6 registry 契约。测试 195+8 过（含 `tests/test_boundary_semantics.py`）。
对照 v2（`results/E1_1_v2/`，边界修复前）与 v1（`results/E1_1_operator_pool_v1/`）——**均冻结不覆盖**。

**A-32a 落盘清单**：完整响应矩阵 ✅ / 依赖指纹 ✅ / `boundary_modes`（symmetric/interp）✅ /
fallback 台账 ✅（`denoise_stl→{stl:109, savgol:211}`，与 v2 **完全一致** → 边界修复与 STL 正交）/
bootstrap 配置 ✅（B=300, perm=500, kfold=5）/ 本 decision ✅ / v2→v2.1 逐动作 diff ✅（下节）。
本目录未用 alias（requested≡canonical，全 canonical 名）。

## v2 → v2.1 逐动作 diff（`diagnostics/diff_v1_v2.py E1_1_v2 E1_1_v2_1`）

**仅 v_median 变化（320/320 uid，max|Δ|=2.67）；其余 6 动作 bit 级一致（0/320）** →
全部变化唯一归因于 denoise_median 边界修复，因果链闭合（与 v1→v2 的 STL 隔离同款）。
且与 A-31b 诊断点估计逐位吻合（1.0146/0.9076/1.6823/1.6478）——诊断→正式重跑内部一致。

| cell | v_median v2(零填充) | **v_median v2.1** | Δ | 该 cell v2.1 次优 |
|---|---:|---:|---:|---|
| snrHigh\|full | 1.3642 | **1.0146** | −0.3496 | v_stl 1.7459 |
| snrHigh\|miss | 1.1228 | **0.9076** | −0.2152 | v_winsor_savgol 1.3994 |
| snrLow\|full | 1.7060 | **1.6823** | −0.0237 | v_winsor_savgol 1.6888 |
| snrLow\|miss | 1.7089 | **1.6478** | −0.0611 | v_winsor_savgol 1.7893 |

**v_median 现在 4/4 cell 全胜**（v2 中 snrLow|full 曾是 v_winsor_savgol 1.6888）。全局单动作均值
（A-33b 三 estimand 显式区分，勿混用）：
- `uid_weighted_mean`（按 cell uid 数 88/78/72/82 加权）：v_median **1.3010** ≪ 次优 v_winsor_savgol 1.6658；
- `cell_equal_mean`（cell 等权，与下节三层分解同口径）：v_median **1.3131**；
- 二者均**不等于** bootstrap 三层分解的 `L0_global_oracle`=**1.3623**（oracle 逐 cell 选 best action +
  refit-in-bootstrap，estimand 不同 → 勿与上二者互比）。

## 判据裁决（v2.1，A-32b 三分 gain 命名）

### D-1.1a/b 良定性 — 3/4（比 v2 的 2/4 **更**良定，winner 全部 v_median）
| cell | winner | win_rate | gap CI | WD |
|---|---|---:|---|:--:|
| snrHigh\|full | v_median | 1.00 | [+0.449, +1.042] | ✅ |
| snrHigh\|miss | v_median | 1.00 | [+0.233, +0.707] | ✅ |
| snrLow\|miss | v_median | 0.99 | [+0.021, +0.265] | ✅（v2 中不良定 → 新增） |
| snrLow\|full | v_median | 0.62 | [−0.113, +0.079] | ❌（near_tie 0.50） |

### 三层分解 — routing_gain 进一步塌缩，97.5% 分位落到 ε 下（B=300，贴近边界，措辞保守）
- L0 1.3623 → L1 1.3580 → L2-UB 1.3391（bootstrap 重拟合口径，cell 等权）。
- **routing_gain(deg) = +0.0043，CI[+0.0000, +0.0283]（B=300）**。L0≥L1 构造性 caveat 仍在
  （下界触 0 → 非标准零检验）。**bootstrap 97.5% 分位（0.0283）落到 ε=0.03 之下，但仅差 0.0017 且 B 偏小**
  → 措辞取保守形式：*当前池上退化路由 oracle 增益的 bootstrap 97.5% 分位不高于实用阈值*，
  **不**写成"95% 置信确认低于阈值"（分位贴边界、B=300、分位本身有蒙特卡洛噪声，A-33b）。
  v2 时上界 0.040>ε 只能说"未建立"；v2.1 的信息增量是**方向性的**（分位跨过 ε），非强确认。
  正式判决前将以更大 B（full-refit group bootstrap，见 nested CI 升级 A-33c）复核该分位稳定性。
- routing_gain(pattern UB) = +0.0189，CI[+0.0090, +0.0298]（in-sample 上界，稳定）。
- share 18.5/81.5 仍**严禁 headline**。

### processing_gain — 从 +0.303 涨到 **+0.4607**（v_none→L1，cell 等权，原始 OOF 口径）
| cell | none→L1 | 备注 |
|---|---:|---|
| snrHigh\|full | +0.8958 | |
| snrHigh\|miss | +0.6798 | |
| snrLow\|full | **+0.1175** | 池最弱处（witness 所在地） |
| snrLow\|miss | +0.1498 | |
- "判官不敏感/语料不需预处理"备择被进一步强否定；池在 snrLow 依旧最弱 → E-3.3 方向不变。

### witness 状态（正式确认 A-31b 初判）
- **snrLow|full 存活**：v1-STL 响应列 1.5406 vs v2.1 池最优 v_median 1.6823 → 池外动作仍好 **0.1417**（≫ε）。
- **snrLow|miss 死亡**：v2.1 v_median 1.6478 已优于 v1-STL 1.6752 → 该 cell witness 由边界修复解释。
- witness 缩为**单 cell**；P1①（强平滑成为 ≥1 个 snrLow cell 的 held-out selected action）判定基准=本文件。

## 综合裁决（v2.1 = E-3.3 正式基线）
1. **v_median 全局单动作在修复后统治当前池**（4/4 point winner、3/4 稳固良定、全局均值领先 0.36）——
   "H* 局部稳定"的证据更强，同时"不同 cell 需不同动作"的证据更弱。
2. **routing_gain(deg) 的 bootstrap 97.5% 分位（0.0283, B=300）方向性地落到 ε 之下**（差 0.0017、贴边界，
   措辞保守；构造性 caveat 在案，正式判决前更大 B / full-refit group bootstrap 复核，A-33b/c）。
3. **供给假设照常由 E-3.3 判定，但赌注变清晰**：witness 只剩 snrLow|full 一处（差 0.14），
   Family 0 剂量扫描若能在该 cell 追平 1.54 量级即复现；若 F0 失败则 F1–F5 承重。
   **Δ_supply 基线 = 本目录响应矩阵对应的 pool_v2.1**（A-31b：对 v2 未修基线跑会虚报 +0.16 量级）。
4. 交互 2/4 存活分层置换（同 v2）；rank 依赖池（eff_rank 1.93–3.44 随 cell 大幅波动）。

## 产物
`results/E1_1_v2_1/{report.json,response_matrix.csv,decision.md}`。对照目录：`E1_1_v2/`（边界修复前）、
`E1_1_operator_pool_v1/`（算子身份修复前）。诊断脚本：`diagnostics/{diff_v1_v2,boundary_diag}.py`。
