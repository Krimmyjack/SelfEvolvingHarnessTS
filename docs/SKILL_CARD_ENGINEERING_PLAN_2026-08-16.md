# Skill Card 工程推进计划（2026-08-16）

> **定位纠正（本计划的前提）**
> 本项目要交付的是**工程能力**，不是现象证明。两个能力：
> - **C1 Agent 驱动的数据 Readiness**：结合任务需求与时序模式，自主分析数据质量问题、生成针对性的数据准备策略
> - **C2 反馈驱动的自适应优化**：用下游任务表现验证处理效果，并利用成功/失败经验持续优化 Harness
>
> 因此**每一步的完成判据是「机制跑通、可复现、可撤销」，不是统计显著性**。
> 2026-08-16 之前的七轮实验都用了科学判据（能否事前预测 / p 值），是重心放错。
> 它们的有效产物只有两条，作为本计划的输入而非结论：
> 1. **事前预测不可得** → 「针对性」必须由 *探测+验证* 实现，不能由 *分析后决策* 实现。这是架构决策依据。
> 2. **顺序先验可得且稳定** → 累积经验的可用形式是**排序先验**（相对随机 gain/probe +26%/+31%，两个 cohort 一致）。这是卡的内容。

---

## 一、当前工程状态

| 组件 | 状态 | 证据 |
|---|---|---|
| Fast 生成合法 Workflow | ✅ 通 | FULLOP2 24/24 |
| Target Support 验证效果 | ✅ 通 | ScopeExecutor / 210+420 格实测 |
| Episode 写入（成功/失败/冲突） | ✅ 通 | experience_memory |
| Skill 生命周期（形成→批准→检索→重绑定→再探测） | ✅ 通 | `CONTEXT_BOUND_SKILL_REBINDING_DEV_PASS` |
| 在线循环（fast 改序 / slow 触发 / 坏 patch 拒绝 / removal 恢复） | ✅ 通 | `OPERATIONAL_TARGET_LOCAL_SELF_EVOLUTION_DEV_PASS` 4 场景 8/8 |
| **经验固化为持久工件** | ❌ **`skills/learned/` 是空的** | 只有 .gitkeep |
| 结合**任务需求**选策略 | ❌ 任务轴被 `allowed_tasks` 硬编码，anomaly rig 不存在 | |
| 探测预算优化 | ⚠️ F4 已确认（省 28%、零漏判）但未接进队列 | |

**结论：C2 的闭环差最后一段——经验被写下来了，但从未凝结成一个能改变下一轮行为的持久工件。**
这是当前唯一的承重缺口。

---

## 二、动手前必须知道的三个硬事实（今天查出来的）

**F-1 `skill-entry/1` 字段集是封闭的。**
`_require_exact_fields` + `_reject_forbidden_fields` 只允许 8 个字段：
`schema_version / skill_id / skill_kind / revision / body / observable_applicability / allowed_tools / risk_guards`。
`load_learned_skill_entry` 另强制 `skill_kind = capability`。
→ 早先设计文档 `DOMAIN_ORDERING_CARD_DESIGN.md` §3 里的顶层 `scope / evidence / ranking_key / order`
**会被 loader 直接拒绝**。必须改走 `risk_guards`（自由 JSON，可规范序列化）+ `body`（人类可读）。

**F-2 `validate_applicability` 只接受 `const / all / any / not / 叶子谓词`。**
→ scope 限定（task × consumer_family × domain）**不能**做成 applicability 的兄弟字段，只能进 `risk_guards`。

**F-3 窗口验证器 `maximum_modified_fraction=0.35` 挡掉 26 个算子里的 13 个**
（全部 `targeting_mode=global`：denoise 5 / decompose 3 / shape 5）。
→ 「生成针对性策略」的实际可用池是 10–13 个。这是环境约束，必须写进 Skill 可见的契约，
否则 Fast 会反复提出注定被拒的候选，且「完整池 vs 受限池」的对比测的是 verifier 不是 Skill。

---

## 三、五步计划（每步都小，判据都是工程判据）

### E1 让卡真正存在 —— 打通 C2 的最后一段
把三臂运行里已验证的 `Card` 逻辑（计数 → score → 排序）从 scratchpad 脚本搬进 harness 工件。

- **产出**：`skills/learned/ordering_<domain>_<task>_<consumer_family>.json`，`skill_kind=capability`；
  `body` = 人类可读的排序说明；`risk_guards` = `{scope, evidence 计数, ranking_key(λ), order}`。
- **完成判据（全部为机制性）**
  1. 卡由 Episode 计数**确定性**生成（同输入同输出，无 LLM 参与）
  2. 通过 `compiler` 编译，进 snapshot
  3. 被 `retrieval` 检索到并**改变下一轮探测顺序**（有 before/after 决策轨迹为证）
  4. `removal` 后行为**恢复原状**（复用既有 removal 控制）
  5. 一个端到端 integration test
- **不做**：suppression / 禁用-恢复机制（第一轮只做排序，否则改进无法归因）
- **成本**：小。逻辑已验证，只是搬家 + 适配 F-1/F-2。

### E2 把在线循环真的接到卡上
E1 只证明卡能改变顺序；E2 让它在真实 `run_online_round()` 里连续生效。

- **产出**：online_loop 读 active snapshot 里的 ordering card 决定候选顺序；每轮结束按 Episode 更新卡、走既有 `requires_target_support` + replay 批准门。
- **完成判据**：连续 N 轮不中断；卡的 revision 单调递增；每次更新有 receipt；坏更新被现有门拒绝；随时可回滚到任一 revision。
- **注意**：λ（风险姿态）**不随 Source 继承**——三臂实验显示继承 Source 先验时连带继承了 Source 的风险偏好，导致 A3 反而优于 A5。λ 必须 Target-local 学。
- **成本**：中。

### E3 任务轴 —— 让「结合任务需求」从硬编码变成真的机制
这是 C1 里「结合任务需求」四个字目前唯一没有兑现的部分。

- **做什么**：建最小 anomaly rig（`residual_zscore_detector` + F1），复用现有 378 格与 4 个算子。
  评估窗注入已知异常，比较「训练数据经清洗 vs 未清洗」的检出 F1。
- **产出**：同一算子在 forecast 与 anomaly_detection 下的效用表。
- **完成判据（工程口径）**：rig 能跑、能出稳定数、能被 TaskSpec 切换；
  **不要求**「必须出现符号翻转」——出现了是 C1 的强证据，没出现说明 `allowed_tasks` 的硬门可以放宽，两种结果都推进工程。
- **限制须入档**：两个数据集无真实异常标注，用注入式合成异常。
- **成本**：中偏大（唯一需要新建组件的一步）。

### E4 每任务一张卡 —— 「针对性策略」的工程体现
E3 通过后，让同一套机制在两个任务上各自学出一张卡。

- **完成判据**：两张卡的 `order` **不同**；交叉使用（把 forecast 的卡用在 anomaly 任务上）性能下降；
  切换 TaskSpec 时 retrieval 取到正确的卡。
- **意义**：这就是「固定清洗规则做不到、可进化 Harness 做得到」的最小可运行演示——
  而且它只需要 (task × consumer_family) 这个**粗粒度**条件化，不需要七轮都没找到的逐格 trigger。

### E5 把 F4 接进探测队列 —— 「训练效率」
- **做什么**：候选进队列前先过 F4 材料性上界，过滤掉不可能产生材料效应的候选。
- **已确认**：新 series 上零违反、零漏判、cohort A 省 28% 探测。
- **完成判据**：接线后端到端仍通过 E1–E2 的全部判据，且探测数下降。须注明它省的是
  **Outcome/Support 反馈**，不一定省候选模型训练计算。
- **成本**：小。

---

## 四、明确不做

| 项 | 理由 |
|---|---|
| Context Rule Card / Pattern trigger / Scope 拆分 | 七轮（F1/F2/F3/Pattern/MEMO/probe_direction，预测与排序两种框架）证明逐格 trigger 当前不可得。保持暂缓，把「平台高度」当待测量而非待修缺陷 |
| 三臂 A5/A3 完整判据 | 单任务内 headroom 太小（cohort A oracle 上限 +0.0756 vs STATIC +0.0446）。等 E3/E4 把任务轴打开后再谈跨任务的 A5/A3 |
| 扩 `skill-entry` 契约 / 新增 `card-entry/1` | 现在动契约，等 E3 出结果卡的 scope 主键可能变，会再动一次。先走 `risk_guards` |
| 自由文本 Guidance patch | G3/P4 两次 `PATCH_REJECTED`，family 已关闭。Slow 只能提议分组，计数与批准必须确定性 |
| 为每个字段建测试矩阵 / SHA 体系 / Ledger | 根 AGENTS.md §1 反过度工程 |

---

## 五、顺序与依赖

```
E1 (卡存在)  →  E2 (在线连续生效)  →  E5 (F4 接队列)
                      ↓
                 E3 (anomaly rig)  →  E4 (每任务一张卡)
```

E1 是所有后续步骤的前置，且成本最小——**建议立即开始 E1**。
E3 可与 E1/E2 并行准备（它不依赖卡）。

---

## 六、与既有纪律的对齐

- 只追加不改写：新实验写新 section，冻结 verdict 不追溯修改
- LLM 不批准自己的 patch：批准权在确定性 compiler + replay + in-domain feedback
- 配对同场验证：patch 验证必须同期 AB/BA，禁止引用历史基线数字
- 每实验最多一个 runner package + 一个主报告 section + 一个承重 integration test

---

## 七、执行记录

### E1 —— 完成（2026-08-16）`ORDERING_CARD_VERTICAL_SLICE_DEV_PASS`

按本地评审修订版执行。**计划 §3 的 E1 描述已被评审推翻两处**，以本节为准：

1. 「经验从未固化」是**错误表述**。`SnapshotStore._write_snapshot_tree` 本来就会把 learned skill
   写进 `skills/learned/<id>.json`，`.gitkeep` 只在无 capability skill 时才写。h0 为空 = 基线还没学过，
   不是机制缺失。真正缺的是**多条 Episode → 聚合成持久排序控制工件**——所以 E1 是**新增一种
   Memory/Control Update**，不是「搬家」。
2. 计划原说「塞进 risk_guards 即可」。**不成立**：`retrieval.resolve_harness_view` 不读 `risk_guards`
   （已代码核实：该字段在 retrieval.py 中只出现在序列化处）。改为双层门——
   `observable_applicability` 用合法叶子谓词 `task_kind == forecast` 门住任务（检索层），
   Runtime 再机械精确匹配 domain / downstream_model_class / program_family。

产出：`methods/ttha/ordering_card.py` + `online_loop.py` 内一个极小的 Runtime ordering consumer +
`tests/functional/test_ordering_card.py`（4 例）。**h0 未修改。**

七条机制判据全过：确定性生成 / 编译进 snapshot / **实测改变探测顺序**
（`[cand_winsorize, cand_outlier_mad]` → `[cand_outlier_mad, cand_winsorize]`）/ permutation 不变量 /
不供应候选 / scope 三维度各错一个均不生效 / 删卡后逐位恢复。

回归：`tests/functional` **146 passed / 1 failed**，唯一失败是上手文档已记录的既有 f1 失败。

自查发现并修掉一个**空过测试**：首版证据集排出的顺序与基线同序，断言恒真；已补一个与基线相反的
证据集并加入 `probe_order_before != after` 的非空断言。

### 后续顺序（按评审调整，取代 §5 的原顺序）

```
E1 ✅ → E2（立即回到核心三臂 STATIC/A3/A5）→ E5（单独接 F4）→ E3/E4 暂缓
```

E2 的冻结条件：同候选供给、**同 lambda**（在 development/source 上预先固定，三臂完全相同）、
同 Support 预算；唯一变量只能是 Source Episode 是否用于初始化排序。Source 与 Target 计数**分开**存，
不合池。**不做** per-revision receipt / 独立审批系统 / revision ledger——排序卡是 Episode 的确定性
派生视图，不是自由 patch，行为安全由下一轮 Support 保证。

### E2 —— 完成（2026-08-16）`ONLINE_ORDERING_CARD_PREQUENTIAL_DEV_PASS`

runner `evaluation/functional/run_e2_three_arm_ordering.py`；checkpoint
`artifacts/functional/e2/e2_three_arm_checkpoint.json`；承重测试并入
`tests/functional/test_ordering_card.py`（新增 2 例，共 6 例）。零 LLM、零 virgin series、1212 秒。

**协议在跑目标流之前就落盘**（checkpoint 的 `declared_before_target_stream`，先写文件再跑）。

#### 机制判据（工程口径，全过）

| 判据 | 结果 |
|---|---|
| 同候选供给（逐格硬断言三臂相同） | **420 次检查 / 0 违反** |
| 仪器一致性（实测 probe 数 vs 冻结表重放） | **630 arm-cell / 0 不一致** |
| 连续重建（revision 单调递增、编译进 snapshot、被 Runtime 取到） | 每臂每 cohort **7 个 revision**，逐轮断言 `ordering_card_id` 非空 |
| 卡本身的价值 | gain/probe 比随机 **+40.4% / +51.8%**，达 oracle 的 **84.7% / 90.9%** |

回归 `tests/functional`：**148 passed / 1 failed**（唯一失败仍是既有的 `test_f1_forecast_pilot`）。

#### 三臂结果（这是 E2 真正要回答的）

| cohort | STATIC | A3 | A5 |
|---|---|---|---|
| A（kdd2018，150 格） | gain +11.9984 / 208 probes / 56 harm | +11.7104 / 213 / 54 | **+11.9984 / 208 / 56（与 STATIC 逐格相同）** |
| B（metr_la，60 格） | +35.0603 / 76 / 22 | 同左 | 同左 |

- **cohort A**：A3 的**全部**劣势在第一个 batch（gain −0.2881、probes +5）；batch 2–6 三臂逐批完全相同。
  → Source 先验的价值 = **一次性省掉 1 个 batch（25 格 / 41 条 Support 反馈）的冷启动**，不是持续优势。
  `N_first_effective`：A3 = **42**（第 42 条反馈时首次出现行为上有效的 Target-local 重排）；A5 = None。
- **cohort B**：三臂完全相同。原因是 A3 无证据时按 tie_break 排序，其首位 `outlier_mad` 恰好也是
  Source 先验的首位。→ **cohort B 对 A5-vs-A3 不携带信息**，两个方向都不能引用它。

#### 本轮发现的结构性限制（写进档，是下一步的首要问题）

**stop-on-first-positive 会删失 rank 1 以下的证据。**
cohort A 的 A5 改了 6 次卡（rev2/3/5 都翻转过 rank 2↔3），但 **rank 1 始终是 `winsorize`**——
于是这些更新在行为上完全不可观测，A5 与 STATIC 逐格相同。A3 收敛到
`['winsorize','outlier_mad','outlier_iqr']`，A5 停在 `['winsorize','outlier_iqr','outlier_mad']`：
rank 2/3 不同、行为相同。

含义：**当前更新规则没有机制纠正一个错误的 rank-1 Source 先验**——能证伪它的算子正是最少被探测的那个。
这不是实现缺陷，是「排序卡 + 停在首个正向」的结构性后果。
任何补救（强制探索槽等）**必须三臂同步施加**，否则会重演早先那次 STATIC 被豁免探索导致的伪优势。

#### λ 的诚实表述

λ 在本设置下**几乎是惰性旋钮**：cohort A 上 λ∈[0,10] 的 Source dress rehearsal 完全平坦
（gain/probe 0.03270、harm 0.6154 全域不变），因为 λ 改 rank 2/3 但不改 rank 1，而 rank 1 主导结果；
cohort B 上 λ≥3 才真正改 rank 1。不能说「λ 被成功拟合」，只能说**风险姿态旋钮的作用域被 rank-1 主导性压掉了**。

#### E2 立住了什么 / 没立住什么

立住：C2 闭环最后一段真的接上了（Episode → 确定性聚合 → 持久工件 → 编译 → 检索 → 改变真实行为 →
再更新，连续 6 批不中断）；卡只重排不改供给（420 次断言）；排序先验有跨 cohort 一致的价值。

**没立住**：没有证明 A5 优于 A3（A 只有一个 batch 的冷启动差、B 无信息）；没有证明卡能纠正错误的
Source 先验（rank-1 删失使其当前不可验证）；完全没触及任务轴。

---

## 八、本计划已结束（2026-08-16）

E1 ✅ / E2 ✅ 完成并封存。**E3/E4 作废**，被
`docs/SLOW_PROGRAM_RULE_CARD_PLAN_2026-08-16.md` 取代。

原因：E2 测出 Ordering Card 的剩余天花板只有 +18.0%（cohort A）/ +10.1%（cohort B），
且供给层 `fast_propose_v1.maxItems=3` 把可学习状态压到约 1.6 bit/scope。
主线转回「固定说明书 + Slow Agent 可修改的结构化 Rule Card」。

**Ordering Card 的最终定位**：Source Memory 的辅助 Control prior，
**不是** Harness Evolution 的主体。留作已完成的基础设施 + 一条能力边界结论。

已补：最小权限守卫 `contracts/harness.py:333 _reject_mixed_card_authority`
（`card_kind==ordering-control/1` 且 body 含 `Frozen program steps:` → 加载期拒绝）+ 1 测试。
