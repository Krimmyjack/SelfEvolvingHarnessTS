# R4C · 周期填补的作用条件与相对价值（主线 Fable，2026-09-07；口径与判词在任何数字之前冻结）

## 0. 承接与问题

R4A/R4A-b 的序列级特征全部描述**服务窗口的形态**；astra（`idea-stage/observation_mechanism/notes.md`）
补算显示多数实例的效果来自训练侧模型变化、且现有词表未表达程序的**实际作用条件**（周期填补面对
的缺失点只有约 31.6% 有足够同相位供体，其余退回线性）。本包按 astra 的三段拆法只做前两段，
对象选"周期填补"：

1. **作用条件可识别**：程序在当前实例上到底做了什么？（哪些缺口被周期填补、哪些退回线性、供体是否一致、
   周期填补与线性填补差多少；训练窗与服务窗分开）
2. **相对价值可识别**：这些作用条件能否在冻结折外改善"用不用周期填补"这个决策相对**合法 incumbent** 的
   全人群效用？

第三段（LLM 能否利用）不在本包。

## 1. 数据、目标量与边界

- 逐序列读数唯一来源 `_scratch/m_r0k_prediction_store.json`。程序 `W2_pmc_then_outlier_mad`
  （= `period_median_complete` → `outlier_mad`）与 `ANCESTOR`（= `outlier_mad`，自带 `interp_nan` 线性填补）
  各 21 位置 × 2 面 × 20 uid。
- **目标量** `d_i = g_i(W2) − g_i(ANCESTOR)`，g = raw − program 逐序列增益。d 就是"周期填补替代线性填补"
  这一步的边际效果（其后 MAD 两边相同）。同时记 `|d_i|`、`sign(d_i)`、`d_i < −0.30`（严重边际伤害）。
  合法 incumbent = ANCESTOR（K0 卡的程序）。
- 原始序列：含缺失版 KDD2018，loader 同 R4A（`preflight_natural_gap_variant.load_variant`，NaN=503712）。
- 服务窗 = origin 前 192 步；训练窗几何以 `scoped_serving_evaluator.py` / `run_hec1.py` 实际实现为准
  （冻结 anchors、保留条件 `anchor + 48 ≤ origin`），须在工件 provenance 写明并核对。
- **算子行为以源码为准**：`period_median_complete` 的 period / cycles / min_donors 与退回规则从 `operators/`
  实际实现读取（astra 记为 period=24、cycles=3、min_donors=2，需核实）；判定"某缺失点是否被周期填补"
  必须复用或逐行翻译该实现，不得自拟规则。
- 边界：0 Consumer fit；0 LLM；不读 horizon 真值；任何 (uid, 时间索引) < 4056；held-out 禁区
  `[80:120]` × {4056, 4296, 4536, 4776, 5016}；不编辑现有文件；不提交 git；0 新 SHA。

## 2. 作用条件观察量（全部 origin 前可算）

**服务窗（每 (uid, origin)）**

| 名称 | 定义 |
| --- | --- |
| `srv_gap_points` | 服务窗 NaN 点数 |
| `srv_period_filled_points` | 其中按源码规则会被周期中位数填补的点数 |
| `srv_period_filled_frac` | 上两者之比（无缺口记 0） |
| `srv_donor_dispersion` | 周期填补点上供体的稳健离散（供体 MAD / 服务窗稳健尺度）的中位数；无周期填补点记 `NA` |
| `srv_fill_divergence` | 周期填补点上 \|周期中位数填补值 − 线性填补值\| / 服务窗稳健尺度 的均值；无周期填补点记 `NA` |
| `srv_tail48_period_filled` | 最后 48 步内被周期填补的点数 |

**训练窗（每单元共享；按实际训练几何对 20 条被服务序列的全部训练窗汇总）**

| 名称 | 定义 |
| --- | --- |
| `trn_gap_points` | 训练语料 NaN 点数 |
| `trn_period_filled_points` / `trn_period_filled_frac` | 同上口径 |
| `trn_fill_divergence` | 同上口径的均值 |
| `trn_series_touched` | 训练语料中至少有一个点被周期填补的序列数 |

## 3. 读数（阈值现在冻结）

**3.1 作用条件一致性检查（机制必要条件）**
若一个实例的服务窗与其单元训练语料都没有任何点被周期填补，则 W2 ≡ ANCESTOR，应有 `|d_i| < 1e-9`。
报：该类实例数、其中 |d| < 1e-9 的比例；反之，|d| > 1e-9 的实例中"服务窗或训练语料至少一处被周期填补"
的比例。判词：≥ 0.95 → `ACTION_CONDITION_CONSISTENT`；否则 `ACTION_CONDITION_INCONSISTENT`（先查机制再谈价值）。

**3.2 作用强度与效应强度**
`srv_fill_divergence`、`srv_period_filled_points`、`trn_fill_divergence` 与 |d_i| 的 Spearman（pooled 与按面）。

**3.3 方向可识别性**
每个观察量对 `sign(d_i) > 0` 与 `d_i < −0.30` 的 AUC，按四块留一（[0:40]/[40:80]/[80:120]/[120:160]），
报三/四折与均值；只在 `srv_period_filled_points > 0` 的实例上算（其余 d 应为 0，见 3.1）。

**3.4 相对价值（决策读数，分母 = 全服务人群）**
候选规则形如 `使用 W2 当且仅当 <观察量> <op> <阈值>`，否则使用 ANCESTOR；阈值只在训练折上按该观察量的
三分位/中位数选（不扫阈值），在测试折读全人群效用 `U = Σ_i [rule_i · g_i(W2) + (1−rule_i) · g_i(ANC)] / N`。
对照：`always W2`、`always ANCESTOR`（incumbent）、`oracle per-series`（上界，标 ORACLE 不可部署）。
报每折 `U(rule) − U(always ANC)`、`U(rule) − U(always W2)`、受损序列数差、最坏单序列伤害差。

**判词（冻结）**
- `VALUE_CONDITION_INFORMATIVE`：某规则在 ≥ 3/4 折上同时优于 always-ANC 与 always-W2，且四折均值优于
  always-ANC ≥ 0.005（material 线），受损序列数不多于 always-ANC；
- `VALUE_CONDITION_WEAK`：仅优于 always-W2 或仅 2/4 折成立；
- `VALUE_CONDITION_UNINFORMATIVE`：其余。

**3.5 单元级**
`trn_period_filled_frac` / `trn_fill_divergence` 与单元均值 d̄_u 的 Spearman（21 位置 × 2 面）。

## 4. 交付

- 脚本 `evaluation/main_protocol_p4/audit_r4c_imputation_donor_conditions.py`（`python -m` 可复跑，0 fit）；
- 工件 `artifacts/main_protocol/r4c_imputation_donor_conditions.json` / `.md`（含 boundary、provenance：
  算子参数出处、训练几何出处、行数与剔除、全部读数 6 位小数、两条判词）；
- 报告 `docs/R4C_IMPUTATION_DONOR_CONDITIONS_RESULT_2026-09-07.md`（中文，正典 §10 五问 + "这份结果不是什么"）。

## 5. 预写分流

| 结果 | 下一步 |
| --- | --- |
| 3.1 不一致 | 先修机制理解（源码级），价值读数不可解释，不进第三段 |
| 一致 + `VALUE_CONDITION_INFORMATIVE` | 该观察量进入 Observation 候选；下一包做 LLM 输入消融（同模型、同预算、有/无作用条件信息） |
| 一致 + WEAK/UNINFORMATIVE | 记"作用条件可识别但不改变决策价值"，转向训练/服务作用分解（需两模型拟合，随 per-channel 运行采集） |
