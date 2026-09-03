# P4 Forecast CONFLICT 逐序列诊断（2026-08-31）

> **数据源勘误（2026-09-01 追加，不改动本文任何数字与判词）**
>
> 本文标注的数据集 `KDD Cup 2018 with missing values` **是错的**。实际使用的
> `data/kdd2018/series_cache.npz` 建自 `..._without_missing_values.tsf`，缓存内
> NaN 计数为 0；天然缺口（17.119%，270/270 条序列）已被上游填补消除。
>
> 逐值核验见 `artifacts/main_protocol/p4d_natural_gap_roster.json` 与
> `p4d_natural_gap_preflight.json`：两版本在 2,438,652 个观测位置上最大偏差
> `0.000e+00`。**本文数字在该（without）版本上测得正确，全部保留。**
>
> 但结论范围收窄为**无缺口的 outlier / level / denoise 场景**。identity 自身即
> `_linear_integrity` 线性插补，故全部 imputation 算子在本文数据上退化为恒等，
> 从未真正受考——本文任何负结论**不关闭 imputation 方向**。
>
> 含缺失版本记为独立数据身份 `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，
> 与本文结果平行、**不并表**。详见 `AGENTS.md` §8.1。


**证据等级**：`DEVELOPMENT_ONLY_DIAGNOSIS_OF_COLLECTED_RUN`
**机器报告**：`artifacts/main_protocol/p4_conflict_per_series_audit_20260831.json`
**复跑**：`python -m evaluation.main_protocol_p4.audit_conflict_per_series`
**放行**：`NONE`。本诊断不改任何阈值，不开 Final/Query/UCR TEST/sealed AD，不新增
SHA/manifest，`llm_calls = 0`。

---

## 0. 这份诊断回答什么

P4 性能终态工件对每条 Episode 只记录一个聚合 `support_gain` 和它被判入的
relation，不记录 relation 背后的 `per_view_gain`。因此"一条 CONFLICT 到底是
*20 条里蹭破 1 条*，还是 *好几条被重伤*"无法从工件表面读出。这是决定
CONFLICT 门去留的那个数，所以本诊断把它确定性重算出来。

诊断对象是已收集的运行 `p4_forecast_performance_b8_llm8_run2_20260830.json`
（KDD Cup 2018 with missing values，`EXPOSED_DEVELOPMENT`）。

## 1. 口径与忠实性

分类规则读自 `experience_memory.classify_relation`（阈值 ±0.005，本诊断只读不写）：

```
aggregate >= +0.005 且 min(per-series) >= -0.005   -> POSITIVE
aggregate >= +0.005 且 存在 per-series < -0.005    -> CONFLICT
aggregate <  -0.005                                -> NEGATIVE
```

重算复用 P4 性能 Runner 自己的 `_cell_at` 与 `_reading`，不是另写一份实现。
**13 个单元的重算聚合值与终态工件记录的 `support_gain` 全部吻合（|Δ| < 5e-6，
13/13）**，因此下面的逐序列拆分与当初那次运行同仪器、同数值。

## 2. 采样：51 条运行记录 = 13 个独特单元

三个 arm × 三种 replica 顺序重放同一批 origin，所以工件里的 51 条 CONFLICT 是
**运行记录，不是独立样本**。全部统计按去重后的独特 (origin × operator) 单元计算。

进一步分两个阶段：

| 阶段 | 单元数 | 含义 |
| --- | --- | --- |
| **Support 阶段 CONFLICT** | **11** | Support-A 面即存在受害序列，当场被拦，从未取得部署权 |
| **delayed 改判** | **2** | origin 696 的 `outlier_mad` 与 `winsorize`：Support-A 面 **0/20 受害**，是干净的 POSITIVE，**先成为了 Support winner**，随后被 delayed 读数改判 |

> **终态 `POSITIVE = 0` 不等于历史过程中从未形成过 Support winner。** origin 696
> 的两条就形成过。下面第 3 节的伤害画像只用 11 个 Support 阶段单元。

## 3. Support 阶段伤害画像（11 单元 × 20 序列）

| origin | operator | 聚合增益 | 受害数 | 最大单序列伤害 | 负收益总量 | 正收益总量 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 600 | winsorize | 0.0626 | 5/20 | 0.1803 | 0.502 | 1.753 |
| 648 | outlier_mad | 0.2615 | 4/20 | 0.2003 | 0.325 | 5.556 |
| 744 | hampel_filter | 0.5135 | 2/20 | 0.4968 | 0.729 | 10.998 |
| 744 | outlier_mad | 0.2963 | 2/20 | 0.6707 | 0.704 | 6.630 |
| 744 | winsorize | 0.3040 | 3/20 | 0.6038 | 0.713 | 6.792 |
| 792 | outlier_mad | 0.2648 | 6/20 | 0.7726 | 2.210 | 7.507 |
| 840 | outlier_mad | 0.3187 | 2/20 | 0.5145 | 0.521 | 6.895 |
| 888 | outlier_mad | 0.3194 | 5/20 | 0.7205 | 1.329 | 7.717 |
| 888 | winsorize | 0.3115 | 4/20 | 0.3921 | 0.873 | 7.103 |
| 936 | outlier_mad | 0.3180 | 6/20 | 0.3294 | 0.792 | 7.151 |
| 936 | winsorize | 0.2130 | 7/20 | 0.6187 | 1.270 | 5.530 |

汇总：

- 受害序列数 **median = 4/20（20%）**，range 2–7（10%–35%）
- 直方图 `{2:3, 3:1, 4:2, 5:2, 6:2, 7:1}` — **没有任何一个单元是 1/20**
- 最大单序列伤害 median **0.5145**（range 0.180–0.773）
- 聚合增益 median 0.3040 → **最大单序列伤害 / 聚合增益 的中位比值 = 1.99**
- 正:负 质量比 median **9:1**

**判词**：`CONFLICT_IS_REPRODUCIBLE_AND_MULTI_SERIES__NOT_THE_SINGLE_SERIES_NICK_BRANCH`。
不落在"多数 1/20 轻微受损"那一支。

## 4. Support-B 复现：12/13 同向，12/13 同为 CONFLICT

这些 CONFLICT 是**可复现的结构性质，不是采样噪声**——对"把 CONFLICT 当作 Slow
的学习信号"是有利条件，因为信号稳定。

**唯一符号反转：origin 744 × hampel_filter**，A 面 +0.5135（2/20 受害）→ B 面
**−0.1902（13/20 受害）**。这一条若被部署会造成真实的聚合伤害；**严格门在这里
确实拦下了一次真实伤害**，这是它的一条具体正当性证据，不是纯保守。

## 5. 受害序列不稳定 → 按 series ID 的 Scope 修订不可行

```
600 winsorize     [2, 9, 13, 14, 15]      888 outlier_mad   [2, 5, 7, 16, 17]
648 outlier_mad   [5, 6, 13, 19]          888 winsorize     [2, 5, 12, 17]
744 hampel_filter [0, 1]                  936 outlier_mad   [2, 6, 10, 16, 17, 18]
744 outlier_mad   [2, 7]                  936 winsorize     [3, 7, 8, 10, 15, 16, 17]
744 winsorize     [2, 10, 11]             792 outlier_mad   [9, 10, 11, 16, 17, 19]
840 outlier_mad   [1, 6]
```

20 条序列里 **19 条至少受害过一次**，最频繁的 `series[2]` 也只有 6/11，只有
`series[4]` 从未受害。**按序列身份排除没有稳定子集可排除**，这条路可以直接排除，
不必再花预算试。

## 6. identity sMASE 关联 —— 仅诊断参考

在 220 个 (unit × series) 观测上，受害序列的 identity sMASE median 1.370、受益
序列 2.000，`corr = +0.464`，`AUC = 0.714`；机制上讲得通（清洗算子帮助 identity
本来预测得差的序列，伤害 identity 已经预测得好的序列——它削掉的"离群点"在干净
序列里是真实信号）。**0/11 单元存在能完全分离受害与受益的硬阈值。**

> **本节读数标记为 `DIAGNOSTIC_REFERENCE_ONLY`。**
>
> - identity sMASE 是**当前 origin 的下游 Consumer 结果**，不是 held-out Fast
>   Path 可读的部署前观测量，**不得写进 Skill Scope**。
> - AUC 0.714 **不是性能上限**，只是"伤害是系统性的而非随机的"这一点的佐证。
> - 220 个观测共享同一批 series/origin，**不是完全独立样本**。
>
> 它唯一的用途是排除"CONFLICT 是噪声、放宽门即可"这一解释，并说明值得去做
> 第 7 节那项审计。

## 7. 结论与下一步

**结论**：CONFLICT 具有真实且稳定的平均收益（正:负质量 9:1，B 面 12/13 复现），
但局部伤害也真实且较重（median 4/20 受害，最坏单序列损失约为聚合增益的 2 倍）。
它应当成为 **Slow 的修订输入**，而不能直接部署。

**未决**：Slow 的修订面必须建立在**部署可见特征**上。已明确排除的修订面——

- 按 series ID 排除（第 5 节已证不可行）
- 直接使用真实 identity sMASE（第 6 节，属下游结果）
- "允许 k 条序列受害后直接部署"（跳过 Slow，偏离方法）
- 修改安全阈值

**下一步**：部署可见特征的 0-LLM 分组预测审计——已执行，见第 8 节。

- 若部署可见特征能稳定区分风险 → `CONFLICT → Slow 提出一次 Scope/Risk PATCH →
  原候选仍无执行权 → 修订版走 Support-B → 通过才形成 v2 → 独立重遇验证`
- 若预测能力不足 → 保留严格门，Slow 合法弃权，停止在这批数据上追求 H3 正结果

**不改 CONFLICT 门，不跑新的 live，不推进 P4-Evolution。**

---

## 8. 部署可见特征审计（2026-08-31，同日执行）

**机器报告**：`artifacts/main_protocol/p4_deployment_visible_risk_audit_20260831.json`
**复跑**：`python -m evaluation.main_protocol_p4.audit_deployment_visible_risk`

设计：11 个 Support 阶段单元 × 20 序列 → 216 个有标签观测（46 受害 / 170 受益，
4 个中性丢弃），22 个特征**全部由 `values[:origin]` 或算子在 pre-origin 窗口上的
足迹算出**，不读部署 horizon、不读 Consumer 结果。验证为
**leave-one-origin-out**（7 组），AUC 在留出 origin 内部计分——同一 origin 的所有
单元与序列一起留出，避免 origin 级偏移被读成序列间的区分力。

### 8.1 结果：部署可见特征不能分离伤害

| | 分组 AUC |
| --- | ---: |
| 多变量 LOGO（标准化 logistic，C=0.5，22 特征） | **0.587** |
| 最佳单特征 `modified_longest_run`（方向正确） | 0.565 |
| `backtest_smase_prev_origin`（合法代理，**方向相反**） | 0.434 |
| *（参考，不可部署）* 当前 origin 的 identity sMASE | *0.712* |

多变量结果在各 origin 间**不稳定**：0.333 / 0.453 / 0.484 / 0.556 / 0.580 /
0.638 / 0.674 —— 三个 origin 低于随机。不是"弱而一致的信号"，是噪声。

六个特征恰为 0.500（`recent.coverage`、`recent.missing_run_count`、
`recent.maximum_missing_run_length` 及其三个 change 对应项）：这批 KDD eval 面上
缺失轴不携带任何信息。

### 8.2 为什么最自然的合法代理会失效

`backtest_smase_prev_origin`（前一个 horizon 的 identity 逐序列 sMASE）是
identity sMASE 的自然合法替身，但它 AUC 0.434 —— **排序方向相反**。原因是
**序列的可预测性跨一个 horizon 就不稳定**：

| origin | 600 | 648 | 744 | 792 | 840 | 888 | 936 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rank-corr(前一 origin, 本 origin) | 0.544 | 0.838 | 0.165 | 0.132 | 0.203 | 0.504 | 0.238 |

汇总 `corr = 0.365`，rank-corr 中位约 0.24。

所以第 6 节那条关联是真的（"清洗帮助难预测的序列、伤害好预测的序列"），但它
**只在部署 origin 上成立，无法从历史推知**——需要预知的那个量本身就不持久。
这不是"换个更好的特征或模型"能解决的问题，是这批数据上的结构性限制。

### 8.3 分支判定

判词：`DEPLOYMENT_VISIBLE_FEATURES_DO_NOT_SEPARATE_HARM`。

数据落在第二支：**保留严格门 → Slow 在此类 CONFLICT 上合法弃权 → 停止在这批
数据上追求 H3 正结果**。

边界（这是"在这批数据上不成立"，不是"原理上不可能"）：

- 7 个 origin 组、216 个观测、46 个受害样本，统计分辨率有限；
- 试过的是 harness 自己的 observable 词表 + 回测误差 + 算子足迹，不是穷举；
- 46 个正例分布在 7 组，容量更大的模型只会过拟合，所以未再加模型类。

第 5 节（按 series ID 不可行）与本节（按部署可见特征不可行）合起来，把
`CONFLICT → Slow Scope/Risk 有界修订` 这条路在**当前数据 + 当前 observable 词表**
下关闭。是否据此正式判 Slow 弃权、并把 Forecast 支线的主张改到伤害/成本轴，属
Planner 裁决。
