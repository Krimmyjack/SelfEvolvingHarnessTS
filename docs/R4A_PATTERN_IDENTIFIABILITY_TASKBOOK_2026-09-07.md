# R4A · 条件可识别性任务书（主线 Fable，2026-09-07；阈值与口径在任何数字计算之前冻结）

## 0. 要回答的问题

项目前提有两半：(i) 同一处理随 Consumer / cohort / 局部模式反号——已验证（M0 级）；
(ii) 决定处理有效与否的条件能从**部署时可见**的数据模式里读出来——**未验证**，间接证据
多为负（W61 LODO 0.5038、W54 AUROC 0.497、P4d `FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE`、
D1 AUC 0.45–0.65、HEC-1 34 个 Support 安全候选只有 10 个传到 +144）。但这些测试全部是
"一袋通用全局指纹 + 阈值/树"，从未从**处理机制**反推该看什么，也从未把 outcome 侧的
"条件在未来"假设量化。

本包用零新 fits 回答三件事：

1. **机制导出的模式量**是否比现有 12 维通用观测更能区分同一程序的受益/受害序列？
2. Support 面到 delayed 面的**符号翻转**能否被任何 origin 前可见的量预判？
3. （ORACLE 诊断，永不作为可部署特征）**未来窗口是否出现同类事件**能否解释伤害？若它解释力
   远高于任何 origin 前可见量，则"条件部分在未来"成立，这是可识别性的结构性上界。

## 1. 数据与边界

- 逐序列读数**唯一来源**：`_scratch/m_r0k_prediction_store.json`（263 条；键
  `program|position|face`；每条含 `origin`、`eval_uids`、`raw_per_view`、`program_per_view`）。
  逐序列增益 `g = raw − program`（正 = 程序有益）；单序列伤害 `h = program − raw`。
  该库由已授权包 m_r0k / DEV-AUTO-1 产生，重读 0 fit。**不得触发任何新的 Consumer fit。**
- 程序覆盖：`ANCESTOR`（= `outlier_mad`，全 21 位置 × 2 面）、`W1_hampel_filter`、
  `W2_pmc_then_outlier_mad`、`W3_outlier_mad_then_pmc`（各 42 条）、两个 `CAND_outlier_iqr>…`
  组合（31 条）、其余零星程序只作参考不入主表。
- 位置：0–17、23、24、25（HEC-1 forward 课程位置）；u18–u22 不在库中，本包也不补。
  面：`support_face`（origin）与 `delayed_face`（origin+48）。
- 原始序列：含缺失版 KDD2018（数据身份 `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`），
  loader 以 `run_m_r0k_scope_workflow_development.py` / `run_hec1.py` 实际使用者为准；使用前
  必须核实缓存内 NaN 数 > 0（`data/kdd2018/series_cache.npz` 是**无缺失**版，不得误用）。
- **held-out 禁区**：序列块 `readable[80:120]`（T180…T22）× origin ∈ {4056, 4296, 4536,
  4776, 5016} 的任何 (series, origin) 对。库内最大 origin 3864，ORACLE 诊断最多读到
  origin+48+48 = 3960 < 4056；脚本须断言"读取的任何 (uid, 时间点) 均 < 4056"。
- 0 LLM；0 SHA/Hash 新增；不编辑任何现有文件；不提交 git。

## 2. 分析单位与标签

单位 = (program, position, face, uid)。标签：

- `helped` = g > 0；`harmed` = g < 0；`severe_harm` = h > 0.30（与权威门单序列线一致）。
- `transport_flip`（按 (program, position, uid)）= Support 面 helped 而 delayed 面 harmed。
- 汇总口径同 M3：先按 (program, face) 给出 **B1 = 程序级均值/基率**，所有特征只报**相对
  B1 的增量**。

## 3. 候选模式量（全部只用 origin 之前的数据；context = 192 步）

**机制导出（裁剪类程序：outlier_mad / hampel / winsorize / outlier_iqr）**

| 名称 | 定义 |
| --- | --- |
| `spike_recurrence` | 把 context 切成 4 个 48 步窗，含 robust-z（context 中位数/MAD）≥3 尖峰的窗口占比 |
| `spike_recency` | 最近一个 ≥3 尖峰距 origin 的步数 / 192（无尖峰记 1.0） |
| `spike_tail_count` | 最后 48 步内 ≥3 尖峰点数 |
| `spike_head_count` | 前 144 步内 ≥3 尖峰点数 |
| `spike_sign_balance` | 正向尖峰数 − 负向尖峰数，除以总尖峰数（无尖峰记 0） |
| `spike_peak_over_tail_sd` | context 最大 |z| 除以最后 48 步的稳健标准差比 |

**机制导出（缺口类程序：W2 / W3 含 `period_median_complete`）**

| 名称 | 定义 |
| --- | --- |
| `gap_tail_fraction` | 最后 48 步缺失占比 |
| `gap_recurrence` | 4 个 48 步窗中含缺失的窗口占比 |
| `gap_longest_run_steps` | context 内最长连续缺失长度（步） |
| `gap_period_aligned` | 缺失点落在与 horizon 同一日内相位（period=24）的占比 |

**通用基线（现有观测词表，`contracts/observables.py` 的 12 个数值项；优先复用项目里
计算 feature card 的函数，找不到则至少重算 `missing_fraction`、`longest_missing_run_fraction`、
`local_robust_z_peak`，并注明是重算）**

**ORACLE 诊断量（读 [origin, origin+48) 真值；只用于机制归因，报告中必须标 ORACLE）**

| 名称 | 定义 |
| --- | --- |
| `oracle_future_spike` | horizon 内是否出现 ≥3 尖峰（以 context 的中位数/MAD 为尺度） |
| `oracle_future_gap` | horizon 内缺失占比 |
| `oracle_future_level_shift` | horizon 均值与 context 最后 48 步均值之差 / context 稳健 SD |

## 4. 读数与判词（阈值现在冻结）

对每个 (program ∈ {ANCESTOR, W1, W2, W3}, face)：

1. B1 基率：helped 率、severe_harm 率、mean g。
2. 每个特征单独：预测 `helped` 与 `severe_harm` 的 AUC；以及 Brier 相对 B1 的增量；
   **按序列块留一（LODO over blocks [0:40] / [40:80] / [80:120]）**，报三折各自与均值。
3. 冻结分箱 stump（`contracts/observables._NUMERIC_BIN_EDGES`，缺省 (0,1,3,6)）的
   LODO 增益：用特征 ≥/≤ 阈值选中子集，报选中子集全人群效用 G(S)=Σ g·1[i∈S] 相对
   "全治"与"全不治"的差；**分母永远是全服务人群**。
4. `transport_flip` 的 AUC（origin 前特征能否预判翻转）。
5. ORACLE 行单列：`oracle_*` 对 `severe_harm` 的 AUC，与最佳可见特征并排。

**判词（每 program × face 一条）**

- `PATTERN_INFORMATIVE`：存在可见特征，LODO 三折 AUC 均 ≥ 0.70（helped 或 severe_harm 任一），
  且三折方向一致；
- `PATTERN_WEAK`：最佳可见特征 LODO 均值 AUC ∈ [0.60, 0.70) 或三折方向不一致；
- `PATTERN_UNINFORMATIVE`：最佳可见特征 LODO 均值 AUC < 0.60。

**结构性上界判词（跨 program）**

- `CONDITION_PARTLY_IN_THE_FUTURE`：某 ORACLE 量对 severe_harm 的 AUC ≥ 0.75 且比最佳可见
  特征高 ≥ 0.10；
- 否则 `NO_FUTURE_ADVANTAGE_DETECTED`。

样本量提醒：每 (program, face) 约 21 位置 × 20 序列 ≈ 400 行，但同一序列跨位置是重复测量，
块级 LODO 只有 3 折；本包是 development 机制读数，不做显著性主张。

## 5. 交付

- 脚本：`evaluation/main_protocol_p4/audit_r4a_pattern_identifiability.py`（一个文件；
  `python -m evaluation.main_protocol_p4.audit_r4a_pattern_identifiability` 可复跑；0 fit）。
- 工件：`artifacts/main_protocol/r4a_pattern_identifiability.json` 与同名 `.md`。
- 报告：`docs/R4A_PATTERN_IDENTIFIABILITY_RESULT_2026-09-07.md`，按正典 §10 五问收口，
  并明确写出：每 program × face 的判词、最佳特征及其三折 AUC、transport_flip 可预判性、
  ORACLE 行、以及"这份结果不是什么"。
- 已有的 `_scratch/m_r0k_prediction_store.json`、`data/`、`contracts/`、`evaluation/` 只读。

## 6. 并行读数（Grok，只读）

- **G1 · 修订效应稳定性（R2）**：用 `artifacts/main_protocol/dev_auto1_blocked_candidates__run2.json`
  的 `per_unit`（ANCESTOR 与两个 CAND，16 单元 × 2 面）以及预测库中 W1/W2/W3 与 ANCESTOR
  的逐序列读数，计算每单元 d_u = gain(child) − gain(ancestor)（聚合与逐序列两种），
  报符号一致率、Var(d)/Var(g_ancestor)、每单元受损数差；0 fit。
- **G2 · 可识别性证据台账**：把项目文档中所有"Observation 能否识别条件"的既有测量
  （W54、W61、P4d Targeter、D1、HEC-1 transfer、E2.59/E2.62/E2.84、p4y oracle bound、m_r0k L1/L2）
  按 [测量 | 特征/方法 | 数据 | 数字 | 判词 | 文件行号] 列成一张表；只读。

## 7. 收口后的分流（预写）

| R4A 结果 | 下一步 |
| --- | --- |
| 某 program `INFORMATIVE` | 该特征进入 Scope 谓词候选与 LLM 上下文；R2 为正则批一个有界的确定性选择规则实验 |
| 全部 `WEAK/UNINFORMATIVE` 且 `CONDITION_PARTLY_IN_THE_FUTURE` | 论文主张改写为"条件在决策时刻不可观测 → 只能反馈治理"；申请 M4 种植阳性对照（≈400 fits）证明仪器能学回已知规则 |
| 全部 `UNINFORMATIVE` 且 `NO_FUTURE_ADVANTAGE` | 机制假设本身可疑；回到 Consumer / evaluator 层（§6 梯子第一行）审 Ridge 对裁剪的响应 |
