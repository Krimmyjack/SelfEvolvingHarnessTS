# R4A · 条件可识别性结果（2026-09-07）

证据类别：**MECHANISM / NEGATIVE**（development 机制读数，不是能力证据，不做显著性主张）。
阈值与口径在看到任何数字之前由任务书冻结，本报告未调整任何阈值。

- 脚本：`evaluation/main_protocol_p4/audit_r4a_pattern_identifiability.py`
  （`python -m evaluation.main_protocol_p4.audit_r4a_pattern_identifiability` 可复跑）
- 工件：`artifacts/main_protocol/r4a_pattern_identifiability.json` / `.md`
- 任务书：`docs/R4A_PATTERN_IDENTIFIABILITY_TASKBOOK_2026-09-07.md`

**一句话结论**：八个 (program, face) 主格全部落在 `PATTERN_WEAK`，没有一个可见量在留一块的
每一折上达到 0.70；同时 ORACLE 量也没有优势（最高 0.606145，全部低于可见量最好读数），
结构性判词是 `NO_FUTURE_ADVANTAGE_DETECTED`。也就是说：**这一轮既没有找到"条件可从部署时
可见模式读出"的证据，也没有找到"条件躲在未来窗口里"的解释**——两个方向同时为负。
唯一非空信号在效用面而不在 AUC 面：对 `W1_hampel_filter`（唯一一个全治为负的程序），
冻结分箱 stump `spike_peak_over_tail_sd >= 6.000000` 在四折中每一折都优于全治，
四折均值 +0.081614。

## 0. 边界自检与复跑口径

| 项 | 值 |
| --- | --- |
| Consumer fits | 0（只索引 `_scratch/m_r0k_prediction_store.json`，263 条，其中已花费的 physical_fits = 496 全部是历史的） |
| LLM 调用 | 0 |
| held-out 读数 | 0 |
| 读到的最大时间下标 | 3911（禁区前沿 4056；ORACLE 最深一次是 origin 3864 的 horizon 末点） |
| 原始序列读取次数 | 2520 |
| 新增 SHA/Hash | 0 |
| 写入文件 | 只有上述两个工件；预测库、`data/`、`contracts/`、`evaluation/` 现有文件未改 |

数据身份与 loader 已核实并写入工件 `provenance`：
`evaluation.main_protocol_p4.preflight_natural_gap_variant.load_variant`，
读 `data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip`，
**NaN 计数 503712 > 0**，数据身份 `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`；
无缺失版 `data/kdd2018/series_cache.npz` 未使用。
Feature card 复用 `audit_cross_fitted_targeting.series_features`（即 `smoke_live_scope._feature_cards`
调用的同一个函数），robust-z 中心/尺度复用 `methods/ttha/public_tools._robust_center_scale`，
因此没有"重算"项。几何常量以代码为准并已核对：context = 192、horizon = 48、delayed 面 = origin+48，
库内 42×N 条记录的 origin 与 `hec1_contract.ordering('forward')` 无一处不符。

行数：主表每个 (program, face) 420 行（21 位置 × 20 uid），共 4400 行，80 个不同 uid。
去重说明：**同一 uid 在同一块的每个位置重复出现**，行与行不独立；因此 LODO 只按块留一，
不做显著性检验。剔除项：40 个 degenerate uid 行（不在执行 scope 内，其读数按构造等于未处理），
8 条 verifier 未通过的条目（全部属于两个 CAND 程序）。

## 1. 每 (program, face) 的 B1 与判词

LODO = 按序列块留一；每折的方向在训练侧确定后再读测试块，表中给的是**定向后**的 AUC。

| cell | helped 率 | severe_harm 率 | mean g | 最佳可见特征 → 目标 | 四折均值 | 四折 | 判词 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ANCESTOR / support | 0.692857 | 0.066667 | 0.245427 | `local_robust_z_peak` → severe_harm | 0.633725 | [0:40]=0.525021、[40:80]=0.544966、[80:120]=0.868421、[120:160]=0.596491 | `PATTERN_WEAK` |
| ANCESTOR / delayed | 0.683333 | 0.073810 | 0.224645 | `local_robust_z_peak` → severe_harm | 0.676036 | [0:40]=0.717803、[40:80]=0.702528、[80:120]=0.684685、[120:160]=0.599129 | `PATTERN_WEAK` |
| W1_hampel_filter / support | 0.459524 | 0.292857 | -0.031355 | `local_robust_z_peak` → severe_harm | 0.639015 | [0:40]=0.624490、[40:80]=0.605546、[80:120]=0.547315、[120:160]=0.778708 | `PATTERN_WEAK` |
| W1_hampel_filter / delayed | 0.509524 | 0.235714 | 0.045469 | `spike_peak_over_tail_sd` → helped | 0.637288 | [0:40]=0.641711、[40:80]=0.559970、[80:120]=0.545000、[120:160]=0.802469 | `PATTERN_WEAK` |
| W2_pmc_then_outlier_mad / support | 0.640476 | 0.142857 | 0.272318 | `spike_peak_over_tail_sd` → severe_harm | 0.648404 | [0:40]=0.703210、[40:80]=0.662823、[80:120]=0.683983、[120:160]=0.543599 | `PATTERN_WEAK` |
| W2_pmc_then_outlier_mad / delayed | 0.642857 | 0.178571 | 0.185003 | `local_robust_z_peak` → severe_harm | 0.656777 | [0:40]=0.724267、[40:80]=0.684158、[80:120]=0.541126、[120:160]=0.677560 | `PATTERN_WEAK` |
| W3_outlier_mad_then_pmc / support | 0.692857 | 0.066667 | 0.245427 | 同 ANCESTOR（数值全等） | 0.633725 | 同 ANCESTOR | `PATTERN_WEAK` |
| W3_outlier_mad_then_pmc / delayed | 0.683333 | 0.073810 | 0.224645 | 同 ANCESTOR（数值全等） | 0.676036 | 同 ANCESTOR | `PATTERN_WEAK` |

八个主格全部 `PATTERN_WEAK`，触发条件都是"最佳可见特征四折均值落在 [0.60, 0.70)"，
没有任何 (特征, 目标) 组合满足"每一折 ≥ 0.70 且方向一致"。若严格按任务书字面只用它点名的
三个块（[0:40] / [40:80] / [80:120]）重算，`W1_hampel_filter / support` 掉到
`PATTERN_UNINFORMATIVE`（最佳读数 < 0.60），其余七格仍是 `PATTERN_WEAK`——去掉的正是
[120:160] 这个 W1 上读数最高（0.778708）的块。两个版本都写在工件里。

机制方向值得记一笔：最常胜出的可见量是 `local_robust_z_peak`，而它的原始 AUC 一律 **低于** 0.5
（例如 ANCESTOR/support 汇总 0.436316，四折 0.474979 / 0.455034 / 0.131579 / 0.403509），
即**尖峰越不显著的序列越容易被裁剪类程序重伤**——没有东西可裁时，裁剪只在改坏形状。
这与机制预期一致，但强度不足以支撑一条谓词。

参考行（不进主表）：两个 `CAND_outlier_iqr>…` 组合同样全部 `PATTERN_WEAK`；其中
`CAND_outlier_iqr({})>winsorize({}) / delayed` 的 `local_robust_z_peak` → severe_harm 三折均值
高达 0.804630（0.699217 / 0.748571 / 0.966102），但最低折 0.699217 差 0.000783 未过 0.70 线，
且该程序只有 280 行、缺一个块，不足以翻案；按冻结规则它仍是 WEAK，不作为阳性引用。

## 2. 冻结分箱 stump 的全人群效用差

分箱用 `contracts/observables._NUMERIC_BIN_EDGES`（缺省 (0, 1, 3, 6)）；规则在训练块上选，
读测试块；分母永远是该折的全服务人群。

| cell | 全治效用 | 训练侧选出的最佳规则 | 相对全治（四折均值） | 相对全不治（四折均值） |
| --- | --- | --- | --- | --- |
| ANCESTOR / support | 0.245427 | `spike_recurrence >= 0.000000`（= 全选） | +0.000000 | +0.267773 |
| ANCESTOR / delayed | 0.224645 | `spike_sign_balance >= 0.000000` | +0.000277 | +0.231133 |
| W1_hampel_filter / support | -0.031355 | `spike_peak_over_tail_sd >= 6.000000` | **+0.081614** | +0.046130 |
| W1_hampel_filter / delayed | 0.045469 | `spike_head_count >= 1.000000` | +0.050479 | +0.121932 |
| W2_pmc_then_outlier_mad / support | 0.272318 | `spike_recurrence >= 0.000000`（= 全选） | +0.000000 | +0.267159 |
| W2_pmc_then_outlier_mad / delayed | 0.185003 | `spike_recurrence >= 0.000000`（= 全选） | +0.000000 | +0.239236 |
| W3_outlier_mad_then_pmc（两面） | 同 ANCESTOR | 同 ANCESTOR | 同 ANCESTOR | 同 ANCESTOR |

读数有两层：

1. **凡是全治已经为正的程序，冻结词表挑不出比"全治"更好的子集**（+0.000000，选中比例 1.0），
   与 P4d 的 `FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE` 同向。这不只是特征弱：缺省边界 (0,1,3,6)
   对取值落在 [0,1] 的机制量只剩 "≤0" 与 "≥0" 两个可用切点，**冻结的部署词表在语法上就
   表达不出这些量的阈值**。
2. **唯一的例外是 W1_hampel_filter**，它也是唯一全治为负的程序（-0.031355）。规则
   `spike_peak_over_tail_sd >= 6.000000` 在四折中每一折都优于全治：
   [0:40] +0.023504、[40:80] +0.034366、[80:120] +0.096316、[120:160] +0.172268，
   选中比例 0.278571 / 0.316667 / 0.450000 / 0.150000；折内相对全不治为
   +0.031788 / -0.023832 / +0.012907 / +0.163657（四折中三折为正）。
   即：**hampel 只对"确有突出尖峰"的序列有效，这一条件恰好落在冻结边界 6.0 上可以表达。**
   这是本包唯一可继续追的正向线索，但它的 AUC 面读数仍只有 0.637288，属于"效用可选、
   排序不可信"的形态。

Brier 的折内分箱经验率预测相对 B1 常数预测的增量全部在 ±0.011 以内（例如 W1/support 上
`local_robust_z_peak` 为 +0.010167，ANCESTOR/support 为 -0.000497），没有实质改善。

## 3. transport_flip 可预判性

`transport_flip`（Support 面 helped 且 delayed 面 harmed）本身规模不小：
ANCESTOR 全单元 0.219048、在 Support 已受益的单元中 0.316151；W1 分别为 0.207143 / 0.450777；
W2 为 0.214286 / 0.334572。也就是说 **Support 面成功的序列里有三分之一到接近一半在 +48 步就翻号**。

但没有任何 origin 前可见量能预判它：最佳可见特征的四折均值 AUC 为
ANCESTOR/W3 `gap_recurrence` 0.548406、W1 `level_excursion_score` 0.527003、
W2 `missing_fraction` 0.559865，全部低于 0.60，且四折方向不一致（冻结规则把
"方向不一致"也记为 `PATTERN_WEAK`，因此判词是 WEAK 而不是 UNINFORMATIVE——这里的
WEAK 表示"没有信号且符号不稳"，不表示"弱信号"，工件的 `verdict_reason` 字段逐格写明）。
两个 CAND 参考行分别为 0.607093 与 0.546944（后者 `PATTERN_UNINFORMATIVE`）。

**结论：符号会翻，但翻不翻在决策时刻读不出来。**

## 4. ORACLE 行与结构性上界

ORACLE 量读 [origin, origin+48) 真值，只用于机制归因，**永不作为可部署特征**。
对 severe_harm 的汇总 AUC（两向取大，与可见量同一口径）：

| cell | oracle_future_spike | oracle_future_gap | oracle_future_level_shift | 最佳可见量 |
| --- | --- | --- | --- | --- |
| ANCESTOR / support | 0.536990 | 0.539723 | 0.518313 | 0.582407 |
| ANCESTOR / delayed | 0.513517 | 0.606145 | 0.540924 | 0.666929 |
| W1 / support | 0.564630 | 0.530700 | 0.502614 | 0.638184 |
| W1 / delayed | 0.568819 | 0.501054 | 0.548287 | 0.623305 |
| W2 / support | 0.559722 | 0.532407 | 0.524907 | 0.650023 |
| W2 / delayed | 0.549275 | 0.504309 | 0.525990 | 0.683459 |

主表 24 行 ORACLE 比较的 margin **全部为负**，最大的一条是 -0.042684
（ANCESTOR/support 的 `oracle_future_gap` 0.539723 对可见量 0.582407）；
主表最高的 ORACLE 读数 0.606145 远低于 0.75 的绝对线（两个 CAND 参考行也一样，
其中最高的是 `CAND_outlier_iqr>hampel_filter / delayed` 的 `oracle_future_level_shift`
0.670395）。判词：

> **`NO_FUTURE_ADVANTAGE_DETECTED`**

这条判词不依赖比较口径：即便不与可见量比较，没有一个 ORACLE 量越过 0.75。
含义是：伤害既不能由"未来窗口出现同类事件"解释，也不能由 origin 前可见模式解释。
把"条件在未来"当作可识别性失败的结构性借口，这一轮**没有得到支持**。

## 5. 与任务书前提不符的事实（三处）

1. **块结构是四块不是三块。** 库覆盖的位置 0–17、23–25 映射到
   `[0:40]`（位置 0–6）、`[40:80]`（7–15）、`[80:120]`（16–17）、`[120:160]`（23–25）；
   任务书 §4.2 冻结的是"三折 LODO"。处理：主判词按**实际存在的全部四块**留一（四折，
   使用全部数据），另按任务书点名的三块单出一份（`verdict_three_named_blocks`），
   两者都在工件里；差异只影响 W1/support 一格。另注意 `[80:120]` 只有两个位置（16、17），
   是最薄的一折，ANCESTOR/support 上 0.868421 这个高读数就来自这一折。
   `[80:120]` 在低 origin 上参与课程是合法的：held-out 禁区是 (block, origin) 对，
   本包读到的该块最大下标是 1511。
2. **W3_outlier_mad_then_pmc 与 ANCESTOR 数值全等。** 42/42 条 `program_per_view` 逐值相同，
   即 `period_median_complete` 接在 `outlier_mad` 之后在这些 cell 上是恒等的（W1 与 W2
   对 ANCESTOR 均 0/42 相同，最大逐点差分别为 5.821865、2.343799）。因此主表的八格里
   只有**六格是独立读数**，W3 两行是别名，不得当作第二次独立验证。工件的
   `provenance.program_equivalence_against_ancestor` 记录了这次比对。
3. **`gap_period_aligned` 按定义不可计算**，记为 `NOT_COMPUTABLE_AS_DEFINED`，未替换定义。
   原因是算术上的：horizon = 48 = 2 × period 24，覆盖全部 24 个日内相位，
   于是"缺失点落在与 horizon 同一相位的占比"对任何含缺失的序列恒为 1.0，无缺失时是 0/0，
   分不开任何东西。

另有两处与用户简报的细节差异，不影响判词：`_scratch` 库中 8 条 CAND 条目 `verifier_passed`
为 False（位置 5、6 的两面）已剔除，故两个 CAND 的行数是 240/280 而非 31×20；
`eval_uids` 经核实等于 `run_hec1.block_uids(span)[:20]`，即每块的 support_a 半边。

## 6. 正典 §10 五问

**Harness 行为改变了什么。** 没有改变。本包 0 fit、0 LLM、不编辑任何现有文件，
只新增一个只读审计脚本与两个工件。它改变的是**下一步该做什么的依据**，不是系统行为。

**数据上观察到了什么。** (i) 同一裁剪程序在同一批序列上确实有大比例的受害样本
（W1/support severe_harm 率 0.292857，ANCESTOR 只有 0.066667），程序间差异很大；
(ii) 机制导出的六个尖峰量与四个缺口量，没有一个比现有 12 维词表明显更好——两族的最好读数
都停在 0.63–0.68 的四折均值，且最好的可见量常常仍是词表里的 `local_robust_z_peak`；
(iii) 唯一有效用价值的读数出现在唯一全治为负的程序上（W1，冻结阈值 6.0，四折一致优于全治
+0.081614）；(iv) Support→delayed 翻号率在已受益单元里高达 0.316151–0.450777，却完全不可预判；
(v) ORACLE 未来量同样无解释力，最高 0.606145。

**当前最大方法不确定性。** 不是"特征够不够多"，而是**这些伤害是否有稳定的、序列层面的条件**。
四折之间读数摆动很大（同一特征同一目标可以从 0.131579 摆到 0.868421），块内 uid 又跨位置重复，
真实自由度远小于 420 行。因此本包区分不了两种解释：a) 条件存在但需要更强表示；
b) 逐序列增益的大部分方差来自 Consumer/评估侧的噪声，序列层面根本没有稳定条件可学。
`NO_FUTURE_ADVANTAGE_DETECTED` 让 b) 的嫌疑显著上升，因为连允许作弊的 ORACLE 都解释不了伤害。

**是否仍与目标一致。** 一致，但结论方向必须改写。项目前提的后一半
（"决定处理有效与否的条件能从部署时可见的数据模式里读出来"）在 development 数据上
**再次为负，且这次是从机制侧发起的、带 ORACLE 上界的负结果**。按任务书 §7 的预写分流，
落在"全部 WEAK/UNINFORMATIVE"这一行，但**不落在** `CONDITION_PARTLY_IN_THE_FUTURE` 那一格，
所以"条件在未来 → 只能反馈治理"这条改写路径本身也缺少证据支撑。

**下一项最小纵向切片。** 按 AGENTS §6 的 first-fault 梯子，最早阻塞已经上移到
"无可读正效应 → Consumer / evaluator" 这一行。建议的最小切片，二选一、不并行：
(A) **仪器信噪比切片**：在固定 (uid, origin) 上重复 Consumer 读数，估计逐序列 g 的重测方差，
若 Var(噪声) 与 Var(g) 同量级，则本包与 W54/W61/D1 的全部负结果都被这一个数解释，
可识别性问题应当搁置；(B) **W1 阈值 6.0 的确定性规则切片**：把
`spike_peak_over_tail_sd >= 6.000000` 作为一条冻结的确定性 Scope 谓词，只在 W1 上、
只在已授权 development 单元上跑一次前瞻验证（预算 ≈ 每单元 2 fits），检验 §2 的
+0.081614 是不是留一里的幸存者。**推荐先做 (A)**：如果仪器噪声本身能吃掉效应，(B) 无论
结果如何都不可解释。

## 7. 这份结果不是什么

- **不是能力证据，也不是泛化证据。** 全部读数来自 development 变体
  `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，held-out 块 × held-out origin
  一个都没读（最大下标 3911 < 4056）。任何一格都不能引用为"系统能识别条件"。
- **不是显著性结论。** 每格 420 行里只有 80 个不同 uid，同一 uid 跨 21 个位置重复计入，
  折数只有 4；本包不做检验，也不给区间。四折之间的摆动应当按"不确定"读，不按"异质性"读。
- **不是对 12 维观测词表的判决。** 结论是"这批机制量没有赢过词表"，不是"词表足够好"——
  两族的绝对水平都不足以支撑谓词。
- **不是对 imputation / 缺口方向的判决。** 缺口族四个量里有一个按定义不可计算，
  其余三个也只在 W2（唯一含 `period_median_complete` 的独立程序）上有意义，样本仅 420 行。
- **不是"未来窗口无关"的物理结论。** `NO_FUTURE_ADVANTAGE_DETECTED` 只说明本包定义的
  三个 ORACLE 量在 48 步 horizon 上解释不了 severe_harm，不排除别的未来量或更长窗口。
- **W3 那两行不是独立证据**（与 ANCESTOR 数值全等），`CAND_…>winsorize` 的 0.804630
  也不是阳性读数（最低折未过线、块不全、参考行）。
- **不改变任何已冻结的协议、roster、split 或预算**，也没有新增 SHA、目录或平台层。
