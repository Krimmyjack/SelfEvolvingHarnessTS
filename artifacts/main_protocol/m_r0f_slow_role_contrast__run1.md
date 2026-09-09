# M-R0f · 第 2 步：只改角色说明的开发对照（run1）

**stage** `M_R0F_SLOW_ROLE_CONTRAST` · **收据** `m_r0f_slow_role_contrast__run1.json`
**传输** `https://api.nowaterapi.xyz/v1` · `gpt-5.6-sol`（与 attempt 3 同一路）
**成本** 2 次真实 Slow、0 Consumer fit，在 2/10 上限内 · `run_fault: null`

## 零、为什么改的是角色说明

第 1 步（`m_r0e`）已定位输入影响：`rows[:60]` 那份切片仍含 **9 条可行三元组**、最优 stump
与全量同为 `local_robust_z_peak >= 6.0`。**截断没有丢掉可行信号**，因此按既定分支规则，
本轮优先验证角色说明。

## 一、唯一被改的因素

`harness_view.instruction`（role=slow）。在**调用缝**上替换——`_RoleOverrideCore` 包住
`core.run_stage`，只换视图的 `instruction` 字段。**未改快照、未写 store、未新增 SHA**；收据
同时记录两段原文与各自 sha256，并显式标注 `effective_harness_view_sha_is_stale`。

保持与 attempt 3 完全一致：模型、传输、`slow_scope_clause_v1` schema、冻结 12 词表、
`public_input` 正文（**含 `rows[:60]` 截断**）、policy、阈值、候选空间、replay 屏、
k=1 的 Draft 状态与 Active 谱系。

**没有告诉模型**：任何特征名、任何方向或阈值、影子搜到了什么、上一次调用弃权过。
**弃权仍然可用**：`no_proposal` 信封由 `agent_core` 无条件提供，新说明还明写了它。

| | attempt 3（快照原文） | run1（外环说明） |
| --- | --- | --- |
| 角色自述 | TTHA **preparation** Agent | 外环 **Scope** Agent |
| 对 gain | "Never … infer **candidate utility**" | "Reading those gains is your job: public measurements" |
| 要产出 | effect-distinct **PROGRAM** candidates + **identity** | 一条 scope clause，程序固定 |
| 阶段 | "**Inspect** before proposing" | "no inspect stage: a single edit call" |
| 收尾 | "**abstain** when public evidence does not justify a repair" | "if the evidence genuinely supports no such feature, return no_proposal" |

## 二、结果：模型不再弃权，开始提议

| 调用 | `no_proposal_reason` | 返回 |
| --- | --- | --- |
| 1 | `null` | `outlier_region_end_fraction >= 0` |
| 2 | `null` | `outlier_region_end_fraction <= 0` |

对照 attempt 3：同一步、同一证据、同一模型，**唯一变量是那段说明**，弃权变成了两次提议。

**这是 n=1 对 n=1。** 一次弃权与一次提议不是效应量，只是一条可复现的定点记录。

## 三、两次提议都没通过工具校准

`clause_from_slow` 对 `outlier_region_end_fraction` 两个方向都判 `NO_FEASIBLE_THRESHOLD`。
0 成本复算出的逐边缘读数：

| 方向 | 边缘 | treated | aggregate | 失败的线 |
| --- | --- | --- | --- | --- |
| `<=` | 0.0 | 21 | 0.098846 | harmed_fraction、single_series_harm |
| `<=` | 1.0 / 3.0 / 6.0 | 57 | 0.439761 | harmed_fraction、single_series_harm |
| `>=` | 0.0 | 57 | 0.439761 | harmed_fraction、single_series_harm |
| `>=` | 1.0 | 1 | 0.345778 | coverage_floor |
| `>=` | 3.0 / 6.0 | 0 | — | 全部（选不到行） |

该特征**并非缺失**：100 行全部携带它，取值落在 [0, 1]。

## 四、真正的阻断点：这个特征根本没有可用的候选栅格

`frozen_bin_edges` 从观测契约读边缘，**没有登记的数值特征回落到 `(0.0, 1.0, 3.0, 6.0)`**。
12 个词表特征里，**6 个走的是这个回落**：

| 特征 | 冻结边缘 | 本步可行三元组 |
| --- | --- | --- |
| missing_fraction | (0.0, 0.01, 0.05, 0.2) | 2 |
| longest_missing_run_fraction | (0.0, 0.01, 0.05, 0.2) | 1 |
| **local_robust_z_peak** | (0.0, 1.0, 3.0, 6.0) ← 回落 | 1 |
| estimated_region_start_fraction | (0.0, 0.01, 0.05, 0.2) | 3 |
| estimated_region_end_fraction | (0.0, 0.01, 0.05, 0.2) | 0 |
| level_region_fraction | (0.0, 1.0, 3.0, 6.0) ← 回落 | 0 |
| level_region_end_fraction | (0.0, 1.0, 3.0, 6.0) ← 回落 | 0 |
| **outlier_region_end_fraction** | (0.0, 1.0, 3.0, 6.0) ← 回落 | **0** |
| level_excursion_score | (0.0, 1.0, 3.0, 6.0) ← 回落 | 0 |
| estimated_level_offset | (0.0, 1.0, 3.0, 6.0) ← 回落 | 0 |
| period_change_score | (0.0, 0.1, 0.25, 0.5) | 0 |
| period_reliability | (0.0, 0.25, 0.5, 0.75) | 1 |

模型选中的 `outlier_region_end_fraction` 是 **[0,1] 上的分数**，却按 z 分形状的
`(0.0, 1.0, 3.0, 6.0)` 取候选阈值：3.0 与 6.0 永远选不到任何行，1.0 退化。它的有效栅格实际
只剩 0.0 一个切点。`local_robust_z_peak` 同样走回落，但它是 z 分，3.0/6.0 恰好有意义——
影子的胜者正落在它上面。

**这不是模型选了个荒唐的特征。** 程序是 `outlier_mad`，选 outlier 相关的特征在主题上完全
合理；是这个特征的阈值栅格与它的量纲不匹配。

整体看：96 个三元组里只有 **8 个可行**（8.3%），分布在 5 个特征上。模型要一次命中，且
除了一行「上次这个方向不可行」之外拿不到任何反馈。

## 五、预算的结构性交互（记录，不改）

候选终态是 `OUTER_LLM_BUDGET_SPENT`，不是 `NO_FEASIBLE_THRESHOLD_AFTER_RETRIES`：
`OUTER_LLM_PER_STEP = 2` 在 `retries_per_candidate = 2`（即 3 次尝试）之前先耗尽。
**每个候选实际最多两次提议**，第三次结构上不可达。

第二次调用收到的 `already_refused` 只有 `{feature, direction, outcome}` 三个字段；模型据此
换了方向、没换特征——在它拿到的信息下这是合理动作。

## 六、状态是否正确写入

**正确：什么都没写。** 两次提议都没过校准，`record_revision` 未到达。

| 项 | 前 | 后 |
| --- | --- | --- |
| revisions | 0 | **0** |
| current_scope | `z_peak>=3.0` | **未变** |
| verification_attempts | 1 | **1** |
| state | REVISABLE | **REVISABLE** |
| Draft 数 | 1 | **1（无第二壳）** |
| deployable | false | **false** |

`later_units_touched=0`、`skills_activated=0`、`sealed_reads=0`、`evaluation_face_reads=0`、
`thresholds_changed=0`、`production_code_changed=0`、`snapshot_stores_written=0`。

## 七、发车前的 0 成本排练

用脚本化 core 走过两种形状（弃权 / 提议），确认：override 确实送出（sent sha ≠ snapshot sha）；
提议形状下 `CALIBRATED → preflight 通过 → 屏 → DRAFT_REVISED`，`revisions 0→1`、
工具忽略脚本给的 4.25 改用冻结边缘 6.0、无第二壳；以及 attempt 2 那个 mappingproxy 形状可
序列化。0 LLM、0 fit。

## 八、这轮不能得出的结论

1. **不能**说角色说明是弃权的原因——n=1 对 n=1。
2. **不能**说换了说明就能产出好子句——两次提议都不可行。
3. **不能**说 `outlier_region_end_fraction` 的栅格错配是历史 0 修订的原因——它只解释本次这两次
   提议为何被拒。
4. 下游收益仍**未测**：没有子句写进状态，没有 replay 屏读数，没有触碰任何后续 unit。
