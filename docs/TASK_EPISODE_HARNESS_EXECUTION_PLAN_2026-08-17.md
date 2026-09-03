# Task Episode 驱动的时序 Data Readiness Harness：唯一执行任务书

> **后续执行指针（2026-08-17）**：本文件保留截至 §17.7 的历史协议、结果与
> verdict。§17.8 及其他尚未执行的未来动作，已由
> `docs/EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN_2026-08-17.md` 取代；后续只执行该
> 文件，不再向本文件追加新路线。

版本：rev1（2026-08-17）  
执行对象：本地实现 Agent  
历史状态（截至 §17.7）：`A5_SOURCE_GENERATION_ADVANTAGE_DEV_SIGNAL`（workflow_generation_dev_v1：A3 生成 novel `repair_level_shift` 且 Support −0.0283 未成 Draft；A5 生成/reuse `outlier_mad`，Support +0.0572 成 Draft，delayed +0.0440 激活。A5 提案是 fixed-pool 已探候选的复用，Support outcome 未新开，只新开 delayed；这是 development 单轨迹信号，不是“Source 帮助生成/组合新 Workflow”的证明。原“下一开放项：novelty 约束”已由上方后续执行指针撤回。）

## 0. 文档地位与执行纪律

本文件截至 §17.7 曾是当前阶段的唯一执行任务书；后续执行地位以上方
“后续执行指针”为准。

它取代以下文档中的全部未完成/未来执行步骤，但不改写其中已经产生的历史事实、
verdict、rows、报告或代码记录：

- `docs/INJECTED_TESTBED_DEMO_PLAN_2026-08-17.md`
- `docs/SLOW_PROGRAM_RULE_CARD_PLAN_2026-08-16.md`

旧文档只作历史审计，不得继续从中挑选步骤执行。若旧文档与本文件冲突，以本文件为准。

执行期间遵守仓库根目录 `AGENTS.md`，尤其是：

- 每轮只验证一个主要方法假设；
- 不新增通用 Hash/SHA、Ledger、Receipt、Manifest、Schema 或大型测试矩阵；
- 当前实验最多一个逻辑 Runner、一个主报告、一个必要 smoke/integration test；
- 受控注入只用于验证机制，不得替代自然时序数据上的最终能力；
- Slow/LLM 只能提出方案，不能批准自己的方案；
- 当前 Query future 保持 sealed，只有 delayed evaluator 可以打开；
- 不得为了“跑出正结果”修改 Consumer、Metric、Gate、split 或阈值。

---

## 1. 主坐标：项目目的不得再漂移

### 1.1 项目最终目标

构建一个面向时序数据的 Data Readiness Harness：它能够读取 Task、Consumer、
时序 Pattern、数据作用几何和历史 Action–Response Experience，自主观察数据、提出或
修改 Typed Workflow、在 Target Support 上验证，并把成功、失败和冲突经验写回，形成
可撤销的 Target-local Skill。

项目不是：

- 传统 `Context -> Operator` Router；
- 只学习算子排序的控制器；
- 纯清洗 AutoML；
- 纯 Consumer/模型选择；
- 围绕 Gate、统计校准、Schema 或报告搭建的平台工程。

### 1.2 当前核心里程碑

在相同 Target downstream-feedback 上限下：

```text
A3：空 Source Experience，从 Target-only adaptation from scratch 开始
A5：读取 Source 的 Positive / Negative / Conflict Task Episodes
```

验证 A5 能否比 A3：

1. 更少 probe 地形成第一个有效 Target-local Skill；
2. 少经历有害 Workflow；
3. 不降低最终 delayed utility；
4. 在证据不足时更合理地 abstain 或请求 Observation。

Source Experience 是**提案与诊断先验**，不是执行权。Target Support 决定是否形成
`LOCAL_DRAFT`；独立 delayed feedback 决定保留、限制或撤销。

### 1.3 本轮唯一主要变化

```yaml
one_allowed_change: 将主要反馈/裁定单位从单个 origin Action–Response
                    改为完整 Target adaptation Task Episode
```

本轮不同时引入：新 Gate、新 Pattern 模型、新归因分类器、新 novelty gate、复合多绑定
执行器、额外 Consumer 或新 Task。

---

## 2. 为什么改为完整 Task Episode

当前 K1/n=8/per-origin Program selection 已关闭。主要原因不是 Harness 机械链缺失，
而是单 origin 的效用标签在当前规模下不稳定：同一个 Program 的 winner 会随少量 eval
series 的组成翻转。

新的问题不再是：

> 在 origin=984，哪一个滤波器高出 0.05？

而是：

> 面对一个完整 Target forecasting 数据准备任务，Harness 能否在固定 Support 预算内，
> 通过观察、尝试和反馈形成一个在独立 delayed block 上无害且有效的本地 Skill？

反馈单位变粗不等于删除细粒度数据：

- `series × origin × Program` 继续作为诊断、Scope 定位和 action credit；
- Task Episode 的完整下游结果才作为学习、Draft 和 delayed 核销的主反馈。

简写：

```text
日志粒度 = 细
反馈/裁定粒度 = 完整任务
```

---

## 3. Task Episode 的冻结定义

### 3.1 Episode 边界

一个 Task Episode 从 Target 数据任务首次呈现开始，到下列终态之一结束：

- 形成一个 `LOCAL_DRAFT`，随后在 delayed block 上保留或撤销；
- 在预算内没有可接受 Workflow，终止为 abstain/no-skill；
- 机械或信息墙错误导致 `INCONCLUSIVE`。

一个 Task Episode 内可以有多个顺序 Action–Response attempt，但只有一个完整任务终态。

### 3.2 真实 Episode 只记录实际发生的轨迹

```yaml
task_episode:
  task_episode_id:
  public_context:
    task_consumer:
    cohort_pattern_composition:
    available_observations:
    allowed_operator_contracts:
  attempts:
    - attempt_index:
      observations_used:
      proposed_workflow:
      proposed_scope_summary:
      support_macro_gain:
      support_harm_summary:
      decision: CONTINUE | LOCAL_DRAFT | ABSTAIN
  terminal:
    final_skill_or_abstain:
    probes_used:
    cumulative_support_harm:
  delayed:
    evaluated:
    macro_gain:
    harm_summary:
    final_status: LOCAL_ACTIVE | RESTRICTED | EPISODE_ONLY
```

`attempts` 只能包含实际 probe 并支付 Target Support 成本的 Workflow。不得把未实际执行的
全候选结果写成 Agent Experience。

### 3.3 私有审计信息必须与 Memory 分开

以下信息只进入报告的 `private_audit`，不得进入 Agent prompt、Experience Memory、
检索 Context 或 Skill：

- injection recipe；
- 哪些 series 被注入何种 fault 的真实标签；
- oracle Scope；
- 未 probe 候选的完整 outcome matrix；
- Query/delayed future。

注入 fault ground truth 只能评价诊断/Scope 是否正确，不能替代 downstream utility：

```text
fault_type 匹配某个 Operator family
≠
该 Operator 在当前 Consumer 上必然是最佳 Workflow
```

Workflow 的 WIN/LOSS/HARM 只能由冻结的下游评估得到。

### 3.4 不新建 Episode Schema

复用 `methods/ttha/experience_memory.py` 中现有 `ExperienceEpisode`：

- 每个实际 Action–Response 仍写一条现有 Episode；
- 在 `context_summary` 内增加普通字段：
  - `task_episode_id`
  - `attempt_index`
  - `observations_used`
  - `scope_summary`
- Runner 按 `task_episode_id` 形成任务级聚合视图；
- delayed 继续原位更新同一 Action–Response Episode；
- 不新增数据库、Memory Store 或第二套生命周期。

---

## 4. Feedback、信息墙与执行权

### 4.1 一次 Target feedback 的成本单位

一次 probe 定义为：

> 在冻结的 Target Support block 上完整执行一个 scoped Workflow、重新训练固定 Consumer，
> 并返回完整任务 macro utility 与诊断分解。

Support block 由固定 eval series × K 个彼此不重叠/足够分离的 origins 构成。单个 cell
不是独立 feedback budget。

### 4.2 Support 主反馈

主反馈是候选相对 identity/incumbent 的完整任务 macro gain。

同时保留：

- per-series mean gain；
- per-origin gain；
- harm count/magnitude；
- modified coverage；
- execution/legality 信息。

这些细分字段用于诊断，不得再次被当作 per-origin Router 标签。

### 4.3 不在本轮改全局 Gate

本轮只改变 feedback estimand 的输入，不改写全局 Gate 语义：

- Support 完整任务 macro gain 达到现有材料阈值，才可形成 `LOCAL_DRAFT`；
- delayed 完整任务结果继续走现有单侧 harm veto；
- per-series harm 先完整报告；只有真实运行证明宏平均掩盖了不可接受的集中伤害，才另开
  Risk surface 实验；
- 不采用“每条 eval series 都必须 ≥ +M”的新合取门，因为它不是单侧 harm veto，且会
  重新制造 null-arm 严格占优。

T1–T3 的 Program 激活只能调用 `TTHAMethod.handle_feedback_delayed`；它是当前真正拥有
snapshot 更新权、并执行 `delayed_gain >= -M` 单侧 harm veto 的路径。不得使用
`online_loop._update_delayed_status` 或历史 runner helper 代替激活裁定。若 Episode 的
relation 命名与 snapshot 激活状态不一致，原样同时报告，不在本轮顺手重写 Gate/状态机。

### 4.4 停止规则

每个 Target Task Episode 最多 `B=3` 次完整 Support probe（如成本探针证明无法承受，
只允许在看任何 Target outcome 前一次性冻结更小的 B）。

- 首个通过 Support 的 Workflow → 形成 `LOCAL_DRAFT`，停止继续 probe；
- B 次均未通过 → abstain/no-skill；
- delayed 只对最终 Draft 打开一次；
- delayed 不得反过来修改本次已经执行的 Target 决策，只更新未来 Memory/Skill 状态。

---

## 5. Source Experience 在 A5 中能做什么

Source Memory 不得被限制为 Ordering Card。它可以作为 Slow/Fast 的提案依据，影响：

- 先调用哪种 TS-native Observation；
- 如何解释 cohort/series Pattern；
- `REUSE / MODIFY / COMPOSE / ADD / REQUEST_OBSERVATION / ABSTAIN`；
- Workflow family、参数与 Target-local Scope；
- 哪些已知失败或冲突应规避；
- 是否先做更便宜的 proxy/probe。

Source Memory 不能：

- 直接批准或激活 Skill；
- 读取 Target Query future；
- 绕过 Target Support；
- 把 Source 的 series ID/注入标签当作 Target Scope；
- 根据 dataset 名称直接映射 Workflow。

A5 应收到最相似的正向、负向和冲突完整任务轨迹摘要；A3 为空。两臂随后使用相同的
Target Support 上限和确定性 Runtime。

---

## 6. Pattern 与 Scope 在新设计中的位置

Pattern 不再承担“预测单个 origin 上两个 Ridge gain 谁更高”的任务。

Pattern 的用途改为：

1. 描述完整 Target Task 的数据组成；
2. 在一个 task 内识别哪些 series 可能接受相似处理；
3. 检索具有相似任务/数据组成/作用几何的 Source Episodes；
4. 帮助 Agent 决定需要补 Observation、修改 Scope、组合 Workflow 还是 abstain。

第一版不使用单个全局 9 元组代表整个 corpus。若需检索 Context，只使用已有公开字段的
轻量组合：

```text
Task / Consumer
+ cohort 中各类 series-level Pattern 的计数或比例分箱
+ 当前候选 Program family 与作用范围
+ 已发生的正向/负向/冲突 Action–Response
```

不建设 embedding、向量库、Pattern Graph 或 learned retrieval。

### 6.1 Scope 的第一阶段边界

当前 `_evaluate` 只支持一个 Program + 一个 `train_series_scope`，现有 Context-bound
Skill 只重绑定参数，不支持一个 Skill 内多个 `scope -> Program` 绑定。

因此第一阶段只使用：

```text
一种 fault family
+ 一部分 train series 有 fault
+ 一部分 train series clean
+ 一个 non-identity Program 仅作用于 Agent 识别出的 scope
```

Agent 必须根据部署可见 Observation 选择 Scope。Runner 可用私有注入标签评分，但不得把
标签或 oracle Scope交给 Agent。

混合 `gap + spike + clean` 的复合 Workflow 只能在第一阶段通过后启动，见 T4。

---

## 7. 分阶段执行计划

### T0 — Episode substrate qualification（零 LLM）

> **执行口径已由 Planner 裁定并冻结于 §13。本节 T0 中 `gap -> impute_linear` 的配对已作废（§12.1）。**

目标：先证明完整任务反馈本身可读，再接 Agent。

#### 数据任务

先选一个 family，例如 `gap -> impute_linear`：

- 一部分 train series 注入 gap；
- 其余 train series 保持 clean；
- eval series 不接受 oracle 标签输入；
- 候选最多为 identity、一个匹配 Program、一个机制不同的错误/过宽候选。

#### 评估边界

- Support 与 delayed 使用互不重叠的 origin blocks；
- 两个 block 共享 Task/Consumer/Metric/训练协议；
- 先在 development recipes 调 injection strength；
- strength 只在 T0 调，达到可读性后冻结；
- 不把 `series × origin` cells 当 IID 样本，不使用旧实验 SE 推导 `n=24` 功效。

#### 通过条件

在预定 development tasks 上：

- matching scoped Workflow 在 Support 与 delayed 上方向一致；
- 相对 identity 有材料性正向 headroom；
- 错误或过宽 Workflow 明显更差或产生可诊断 harm；
- 同一个任务标签不随预定 eval 半区轻易翻转。

若失败：`TASK_EPISODE_SUBSTRATE_UNREADABLE`。停止，不接 LLM、不建 Memory、不改 Gate。

#### 成本探针

先跑一个 Task 的完整候选矩阵计时；按实测成本一次性冻结 Source/Target Task 数量。不得预先
假定 20+10，也不得在看 A5/A3 outcome 后增加任务数量。

### T1 — 单个真实 Task Episode 纵向切片

目标：在一个 development Target task 上跑通：

```text
公开 Context
→ Agent Observation
→ scoped Workflow proposal
→ 完整 Task Support probe
→ 写 Action–Response Experience
→ LOCAL_DRAFT 或 abstain
→ 独立 delayed
→ LOCAL_ACTIVE / RESTRICTED / EPISODE_ONLY
```

要求：

- Agent 不看 injection recipe/oracle Scope；
- 最多 B 次 probe；
- 实际 probe 的每条轨迹立即写现有 ExperienceEpisode；
- Task-level 聚合视图能重建完整尝试顺序；
- final Skill removal/revocation 后行为恢复；
- Program 激活/撤销由 `TTHAMethod.handle_feedback_delayed` 及现有 revocation 路径拥有，
  Runner 不自行判定；
- 只做一个必要端到端测试，不扩测试矩阵。

通过标签：`TASK_EPISODE_TARGET_LOCAL_LOOP_DEV_PASS`。  
若机械链失败：`TASK_EPISODE_LOOP_INCONCLUSIVE`。  
若链跑通但没有 Skill：记录合法负结果，不为通过而调 Gate。

### T2 — Source Task Experience bank

目标：产生 A5 可读取的合法 Source Experience。

要求：

- Source 与 Target 至少按基础 series pool 隔离；若只换 injection recipe 而共享同一基础
  series，必须标记为 `RECIPE_TRANSFER_ONLY`，不得称跨域；
- Source Episode 只包含实际执行轨迹；
- positive、negative、conflict、abstain 均保留；
- Memory 不含 injection recipe、oracle Scope、未 probe outcome；
- Source bank 构造完成后冻结，不能根据 Target 结果回头挑 Episode；
- 不归纳 Shared Capability；只作为 Target proposal prior。

### T3 — 核心 A5 vs A3 配对实验

对每个 Target Task，从同一个 materialized baseline fork：

| 项目 | A3 | A5 |
|---|---|---|
| Agent/model | 相同 | 相同 |
| Task/Consumer/Metric | 相同 | 相同 |
| Operator DSL/Workspace | 相同 | 相同 |
| Target Support 上限 B | 相同 | 相同 |
| 停止规则/delayed block | 相同 | 相同 |
| Target 初始 Skill | 空 | 空 |
| Source Task Episodes | 空 | signed Source bank |

唯一实验变量是 Source Experience。

#### LLM 前置断言（零 LLM）

1. 当前 Target task 存在至少一个可执行、在 development substrate 上有 headroom 的候选；
2. A5 的 candidate-conditioned Source Experience 非空；
3. A3/A5 的规范化决策输入只在 Source Experience 部分不同；
4. Target Support 与 delayed blocks 未打开；
5. Agent 看到的 Scope 候选来自公开 Observation，不是注入标签。

若 A3/A5 决策承重输入相同：`A5_A3_ARM_DISTINCTION_INERT`，不发 LLM。

#### 主指标

1. `probes_to_first_local_draft`；
2. `delayed_confirmed_skill`（是/否）；
3. `cumulative_support_harm` 与 harm attempt 数；
4. `final_delayed_macro_utility`；
5. abstention/no-skill；
6. Agent 动作类型：REUSE/MODIFY/COMPOSE/ADD/REQUEST_OBSERVATION/ABSTAIN。

per-origin/per-series 数值只作诊断和失败归因，不作为主胜负标签。

#### 初始工程判读

- `A5_FASTER_SAFER_DEV_PASS`：A5 总 probe 更少，material harm 不增，delayed-confirmed
  Skill 数不低于 A3；
- `A5_SPEED_ONLY`：更快但 harm 或 delayed 变差；
- `A5_A3_NO_SOURCE_EFFECT`：决策与轨迹基本相同；
- `A5_NEGATIVE_TRANSFER`：A5 没有速度收益且安全/delayed 更差；
- `INCONCLUSIVE`：信息墙、机械链、输入区分或仪器失败。

上述为 development 工程判读。若进入论文确认，任务数、统计方法和 fresh target roster 必须
在一次成本/功效探针后另行冻结；不得把 development PASS 直接包装成跨域结论。

### T4 — 混合故障与复合 scoped Workflow（T3 后）

只有 T3 证明完整 Task Episode 与 Source Experience 能改变 Target 适配轨迹后，才增加一个
最小 runner-local 多绑定能力：

```text
scope_gap   -> impute_linear
scope_spike -> outlier_mad
scope_clean -> identity
```

要求：

- 只扩展当前 evaluator，不建设通用 Scope 平台；
- Agent 从公开 series Pattern 推断各 scope；
- oracle injection group 只作评分；
- 比较 composite scoped Workflow、最佳单一全局 Workflow、identity；
- 证明的是 Context-conditioned Data Readiness，不是固定 Operator routing。

### T5 — 未见 family 的 Harness Update 与自然任务

T4 后再引入未见 `level_shift` 或自然 fault：

- Source Memory 没有直接答案；
- Agent 必须 REQUEST_OBSERVATION、MODIFY/ADD Workflow 或 abstain；
- 成功后才允许讨论 case novelty 或 library growth；
- 随后至少在一个自然数据任务上复用同一 Task Episode 循环。

受控注入只能证明机制，不能作为项目终点。

---

## 8. 当前计划中停止/暂缓的内容

在 T3 完成前全部暂缓：

- 5 类 LLM first-fault attribution 主实验；
- novelty gate 和 Hamming case 去重；
- 预设“case 库 Phase1 长到 2、Phase2 长到 3”的展示；
- 全局 Gate 改写；
- per-origin routing/Scope 谓词搜索；
- Ordering Card 继续扩展；
- 混合 fault 复合 Workflow；
- 新 Task/Consumer/anomaly rig；
- embedding、Pattern Graph、向量数据库；
- 新 Schema、Hash、Receipt、Manifest 或通用 Runner 平台。

如果注入器、5 类投影或 novelty gate 已有未提交实现：

- 注入器可保留并用于 T0；
- 5 类投影/novelty 代码不得进入 T0–T3 的决策链；
- 不删除用户已有实现，但标记 `DEFERRED_DIAGNOSTIC`；
- 不为它们补测试、Schema 或 dashboard。

---

## 9. 最小代码与交付预算

### 9.1 允许的最小改动

T0–T3 合计优先复用现有模块，只允许：

1. 一个 Task Episode Runner package（可用子命令/phase 依次运行 T0–T3）；
2. 一个主报告；
3. 一个必要的端到端 smoke/integration test；
4. `ExperienceEpisode.context_summary` 中增加普通任务分组字段；
5. 一个多-origin Support/delayed 聚合函数；
6. 一个把同 task_episode_id 的实际 Action–Response 汇成 Task dossier 的普通函数。

不新增抽象基类、Registry、第二套 Store 或通用 Pipeline。

### 9.2 主报告

统一写入：

```text
artifacts/functional/e2/w1_task_episode_harness_report.json
```

至少包含：

```yaml
protocol:
  feedback_unit:
  support_blocks:
  delayed_blocks:
  target_probe_budget:
  source_target_separation:
private_audit:
  injection_specs:
  oracle_checks:
substrate:
  matched_vs_identity:
  support_delayed_agreement:
task_episodes:
  # 仅实际执行轨迹
source_bank:
  positive_count:
  negative_count:
  conflict_count:
  abstain_count:
a5_a3:
  per_target_paired_trajectories:
  aggregate_metrics:
mechanical_checks:
verdict:
```

private audit 与 Agent-visible task episodes 必须在 JSON 中分为两个顶层字段，禁止混合。

### 9.3 Dashboard

Dashboard 不是 T0–T3 的阻塞项。T1 通过后可复用现有生成器，只显示真实内容：

- 每个 Task Episode 的观察—提案—Support—Draft—delayed 时间线；
- A3/A5 probe 与 harm 的成对曲线；
- Skill 的形成/撤销；
- positive/negative/conflict Memory 数量。

不得用预设 case 数增长替代真实 Harness 更新。

---

## 10. 执行者每阶段回报格式

每完成一个阶段，只回报：

```text
Harness 行为发生了什么变化
真实或受控任务上观察到了什么
第一个未解决阻塞是什么
是否仍与 A5 vs A3 和 Context-conditioned Data Readiness 主线一致
下一项最小动作
```

测试数量、文件数量、SHA 和 dashboard 不是主要进展，只能列为必要附注。

---

## 11. 当前立即执行项

> **执行口径已由 Planner 裁定并冻结于 §13。本节 T0 中 `gap -> impute_linear` 的配对已作废（§12.1）。**

执行者现在只做 T0：

1. 复用/完成注入器；
2. 构造一个单 fault + clean scope 的完整任务；
3. 冻结 Support/delayed origin blocks；
4. 运行 identity / matching scoped Program / 一个错误或过宽候选；
5. 检查完整 Task feedback 是否在 Support 与 delayed 上可读；
6. 跑一个 Task 的耗时探针；
7. 写入同一个主报告后停下回报。

T0 之前及期间：不发 Slow LLM，不建 novelty gate，不做 5 类归因，不改 Gate，不跑 A5/A3。

T0 通过后才启动 T1。不得跨阶段提前实现 T4/T5。


---

## 12. T0 落地口径（审查提案 → Planner 裁定后冻结，2026-08-17）

### 12.0 本节地位

本节流程：主控 Agent 提出代码核对结果与修改提案 → **Planner 逐条裁定** → 冻结为本节。

**审查意见本身不具备覆盖任务书的效力。** 本节生效的是裁定结果，不是提案。
今后任何审查/核对产出一律先记为提案，与 §0–§11 冲突时提交 Planner 裁定，
不得自行声明"冲突时以本节为准"。

裁定台账（保留否决项，供审计）：

```text
12.1 identity 恒等修复          ACCEPTED   （阻塞项，必须改）
12.2 outlier 配对 + 臂的命名     ACCEPTED with NARROWING（撤销"过宽臂"预设）
12.3 SE_block 可读性门槛         NARROWED   （改为聚类单位 + 带符号 + 仅限 T0）
12.4 强制更换 eval series        WITHDRAWN  （提案被否决，仅保留命名约束）
12.5 复用现有注入配方            ACCEPTED
12.6 Observation 测在 train 侧   ACCEPTED
12.7 level_shift 有界性          BACKLOG    （T5 事项，不进 T0）
12.8 接口核对结果                CODE NOTE  （无行动项）
```

### 12.1 【ACCEPTED·阻塞】T0 主配对必须更换：`gap -> impute_linear` 恒等于 identity

`_evaluate` 的 identity 臂不是原始数据，而是：

```text
run_e2_autonomous_natural_workflow_generation.py:695   baseline = _linear_integrity(window)
run_e2_autonomous_natural_workflow_generation.py:543   _linear_integrity = np.interp 填非有限位置
operators/s1_impute.py:35                              impute_linear(strength=1.0) = interp_nan(raw)
```

两者是同一个操作。真实 T117 窗口 `[408:648]` 注入 24 点连续 NaN 后实测：

```text
impute_linear         相对 baseline 改动点数 = 0      maxabs = 0
impute_fft            改动 24                          maxabs = 12.38
impute_ema            改动 24                          maxabs = 11.27
period_complete(p=24) 改动 24                          maxabs = 41.60
```

按 §7/§11 原写法跑 T0，matched Workflow 的 gain 恒等于 0、`behavior_point_count = 0`，
输出会是 `TASK_EPISODE_SUBSTRATE_UNREADABLE`——但这不是负结果，是**正控选错**。

**冻结口径：matched 算子必须赢过"线性插补"，不是赢过"什么都不做"。**

`missing(长 gap) -> period_complete` 保留为 T4 的第二个 family，不在 T0 使用。

### 12.2 【ACCEPTED·收窄】T0 三臂及其命名

```text
fault:  impulsive_outlier

arms:
  identity
  oracle-scoped outlier_mad     matched candidate
  oracle-scoped hampel_filter   mechanism-different / overactive comparator
```

实测（同一 T117 窗口，3 点 +6sd spike，相对 `_linear_integrity` baseline）：

```text
outlier_mad     改动  3 点   maxabs 71.07     全局 MAD 裁剪，点式
hampel_filter   改动 23 点   maxabs 186.61    滚动 MAD，触发更频繁
```

**命名裁定（撤销原提案措辞）**：23 点改动**只证明 hampel 更活跃，不证明它对下游有害**。
不得称其为"过宽臂"，不得预设其必须失败。正式名称为
`mechanism-different / overactive comparator`。

**T0 必须允许 hampel 胜出。** 若 hampel 反而更好，如实报告，不调参、不换臂、不改判据。

**Scope 口径分期**：T0 使用 oracle scope，这**仅是零 LLM 正控**，用于确认 substrate 可读。
自 T1 起，Scope 必须由 Agent 从公开 Observation 推断；oracle scope 退回 `private_audit`
只作评分，不得再进入执行链。

### 12.3 【NARROWED】T0 可读性门槛：聚类单位 SE、带符号、仅限本阶段

原提案 `|macro gain| / SE_block >= 3` 有两处缺陷，按裁定修正：

**(a) SE 必须按聚类单位计算，不得把 `series × origin` 当 IID 样本**

```text
第一步：对每条 eval series，先在 K 个 Support origins 上求 mean gain
第二步：SE_block = 这些 per-series mean gain 的标准误   （n = eval series 条数）
```

**(b) matched 候选必须带符号为正，不得用绝对值**

```text
matched (outlier_mad)：  support_macro_gain / SE_block >= 3        必须为正
comparator (hampel)：    报告 gain / SE_block（带符号）与 |gain| / SE_block，不设通过要求
```

用绝对值会让一个**稳定且强负**的 matched Workflow 通过门槛，与门槛的目的相反。

**(c) 适用范围**

本门槛**只是 T0 的 substrate qualification**，不升级为全局 Gate、不适用于未来任何 Skill 的
激活判定。§4.3"本轮不改全局 Gate"不受本节影响。

**(d) 调参方向单一**

注入强度是 T0 唯一允许调的旋钮（§7）。只允许朝"提高可读性"方向调，
**不得反向放宽本门槛以让路径可跑**（D-6）。达到可读后立即冻结。

### 12.4 【WITHDRAWN】不强制 T0 更换 eval series

原提案要求 delayed 同时更换 eval series 集合。**该提案被否决。**

否决理由（采纳）：Support/delayed 只换 origin 时，评价的是"已形成的数据准备方案与模型，
在后续时间到达时继续接受评价"——这**正是 delayed 的真实语义**，不是缺陷。若同时更换
eval series，问题就变成"Support 消费者上的反馈能否迁移到一批不同消费者"，
而这正是 D4/D5 已暴露不稳定的地方，会让 T0 因反馈 transport 失败，
而非因 Task Episode 不可读——失败原因不可分辨。

**保留的部分（仅命名约束，不改设计）**：

```text
run_v1_kdd2018_natural_slow_update.py:80   anchors = list(range(312, 853, 60))   最大 852
run_e2_autonomous_natural_workflow_generation.py:692   if anchor + HORIZON > origin: continue
```

任何 `origin >= 900` 都收全 10 个 anchor → 120 训练窗 → 同一个 Ridge。故 Support 与 delayed
是**同一拟合产物在不同未来窗口上的评价**。据此：报告与结论中不得声称 delayed 证明了
"跨重拟合的泛化"。这是措辞约束，不是设计变更。

**T0 口径**：同一冻结 eval roster，Support/delayed 使用不重叠且充分分离的 origin blocks。

**Backlog（T3 或之后的 robustness，不在 T0）**：增加 group-disjoint eval series 检查。

### 12.5 【ACCEPTED】复用现有注入配方

```text
evaluation/minipipe/corpus/injections.py:37   inject_target(seed, family, severity)
  families = missing | impulsive_outlier | level_shift | period_change
  severity = mild | severe
  返回 InjectionResult.affected_indices  <- 即 private_audit 要的 oracle scope
```

它为合成 240 长序列而写，不能直接套 10898 长的 KDD 序列；**复用的是配方**：
注入位置、以 `np.std(context)` 为单位的幅度、severity 分档、`affected_indices` 契约。
好处是注入强度与已有 minipipe artifact 可比，且省一个设计周期。

T0 用 `family = impulsive_outlier`。其现有配方为
`mild = 2 点 ±6·scale`，`severe = 4 点 ±10·scale`，符号交替。

### 12.6 【ACCEPTED】Observation 必须测在 train series 上

本设计中：故障注入在 **train** 序列，程序只作用于 train 窗
（`run_e2_autonomous_natural_workflow_generation.py:699`），eval 序列干净且未被程序触碰。
因此 Agent 的观察特征**必须在 train 序列上测**——Scope 应从被处理对象上识别。

这与此前 S1b 的修正方向相反（那里在 T117 上测特征是 bug，因为 loss 定义在 eval 序列上）。
本条明确记录，防止有人按旧结论"改回去"。

### 12.7 【BACKLOG·T5】level_shift 注入必须有界

不影响 T0，仅登记备查：`operators/s1_structural.py:228 repair_level_shift` 只撤销
"上去又回来"的暂态偏移、**保留持续型 regime 变更**。若 T5 注入持续型 level shift，
matched 算子将退化为恒等。现有注入器的 level_shift 是 `corrupt[start:end] += amp*scale`
（有界），符合要求。T5 启动时再取用本条。

### 12.8 【CODE NOTE】接口核对结果（无行动项）

```text
methods/ttha/experience_memory.py:55    class ExperienceEpisode
methods/ttha/experience_memory.py:60    context_summary: Mapping[str, object]
methods/ttha/experience_memory.py:172   _context_distance 只遍历 cohort / local_pattern / program_geometry
methods/ttha/method.py:1305             TTHAMethod.handle_feedback_delayed
methods/ttha/online_loop.py:162         _update_delayed_status（确认排除）
run_e2_...py:672                        train_series_scope 参数存在
```

- 在 `context_summary` 顶层新增 `task_episode_id` / `attempt_index` /
  `observations_used` / `scope_summary` **不会扰动检索距离**（`_context_distance` 只走那三个维度）。
  §3.4 可原样实施。
- `handle_feedback_delayed` 是 Program 激活的正确唯一入口。附注：其 docstring 明确
  guidance pending 不走这条批准链，将来不得假定它覆盖 guidance patch。

---

## 13. T0 冻结执行口径（Planner 裁定版，执行者按此运行）

```yaml
fault:
  family: impulsive_outlier
  scope: 一部分 train series 注入，其余 train series 保持 clean
  strength: T0 唯一可调旋钮；达到可读性后立即冻结

arms:
  - identity
  - oracle-scoped outlier_mad          # matched candidate
  - oracle-scoped hampel_filter        # mechanism-different / overactive comparator
                                       # 不预设有害，允许其胜出

support_delayed:
  eval_roster: 同一冻结 roster，不更换
  origins:     不重叠且充分分离的 origin blocks

report:
  - macro_gain
  - per_series_mean_gain
  - SE_block                    # 先 per-series 对 K origins 求均值，再对这些均值求标准误
  - gain_over_se                # 带符号
  - split_half_stability        # 8 条 eval series 的全部 70 个 4/4 划分
  - modified_point_count
  - one_task_wall_clock         # 成本探针

substrate_pass:
  - outlier_mad macro_gain > 0
  - outlier_mad gain_over_se >= 3
  - delayed 与 Support 同方向
  - 标签在预定半区不轻易翻转

on_fail:
  label: TASK_EPISODE_SUBSTRATE_UNREADABLE
  action: 停止回报；不接 LLM、不建 Memory、不改 Gate、不调判据
```

**split-half 的预登记对照数**：D5 在同一 8 条 eval series 上做过全部 70 个 4/4 划分，
per-origin winner 的划分一致率为 **45.7%**（50% = 纯噪声）。若 T0 的 substrate 可读，
本项应显著高于该数。此数仅作对照与解释，**不作通过条件**（通过条件只有上面四条）。

---

## 14. 【提案·待 Planner 裁定】T0 注入强度标定报告（零 LLM，2026-08-17）

状态：`PROPOSAL_PENDING_ADJUDICATION`。本节**不修改 §13**，不自动生效。
裁定通过后由 Planner 把 14.3 的设置写入 §13 的 `fault.strength`。

依据 §7 与 §12.3(d)：注入强度是 T0 唯一可调旋钮，只允许朝提高可读性方向调，
达到可读后立即冻结。判据（§12.3）在本次运行**之前**已由 Planner 冻结，未作任何改动。

### 14.1 关键发现：只有打进训练窗 target 区的故障才伤模型

训练窗为 `raw[anchor-192 : anchor+48]`；其中 `[:192]` 是回归输入 context，
`[192:]` 是回归标签 target（`run_e2_autonomous_natural_workflow_generation.py:706-708`）。

6 条 faulty train series，随机位置注入，6 个 origin 的 per-series 均值聚合：

```text
注入区域      amp  n/series | damage = inj - clean   d/SE | outlier_mad 回收   r/SE
target          8       40  |            +0.17770  +4.47 |          +0.21307  +3.18
target         16       40  |            +0.53989  +5.75 |          +0.52131  +4.58
target         16      120  |            +1.28711  +7.91 |          +1.18549  +7.09
context         8       40  |            -0.04737  -0.95 |          -0.00303  -0.12
context        16      120  |            -0.08157  -1.60 |          +0.03932  +1.79
both            8       40  |            -0.09976  -2.13 |          +0.00484  +0.16
both           16       40  |            -0.10635  -2.01 |          -0.00364  -0.10
```

机制（已核对代码，非推测）：

- `_center_scale`（`run_e2_cross_series_curation.py:410`）是 **median/MAD 稳健标准化**，
  context 侧尖峰被吸收；Ridge 在 120 行上又把少数离群特征平均掉。
- 最小二乘的**标签**没有任何稳健性，target 侧尖峰直接偏移拟合系数。
- 程序作用于整个 240 窗（`:699`），因此也清洗 target，回收才成立。这是训练数据，
  不是 eval truth，不构成泄漏。

**未解释的观察（登记为开放项，不解释掉）**：`both` 的伤害不等于 target 与 context 之和
（预期约 +0.09，实测 −0.10）。存在交互或非线性。本轮不追查；执行者用自己的 runner
复现时若数值不同，以其实现为准并报告差异。

### 14.2 四条通过条件的可达性（§13 判据，未改）

target 区注入，6/12 train series 带故障，oracle scope，Support=[1104,1368,1800]、
delayed=[2856,3648,3888]，SE 按 per-eval-series 聚类单位计算，split-half = 8 条序列的
全部 35 个无序 4/4 划分：

```text
amp= 8  n=40   SUPPORT outlier_mad   macro=+0.24346  SE=0.03737  g/SE=+6.52  pos=8/8  split=35/35
               DELAYED outlier_mad   macro=+0.18268  SE=0.11309  g/SE=+1.62  pos=6/8  split=35/35
               SUPPORT hampel_filter macro=+0.02024  SE=0.04640  g/SE=+0.44  pos=4/8  split=11/35
               DELAYED hampel_filter macro=-0.08199  SE=0.05496  g/SE=-1.49  pos=2/8  split=27/35

amp=16  n=40   SUPPORT outlier_mad   macro=+0.58631  SE=0.08422  g/SE=+6.96  pos=8/8  split=35/35
               DELAYED outlier_mad   macro=+0.45630  SE=0.16884  g/SE=+2.70  pos=8/8  split=35/35
               SUPPORT hampel_filter macro=+0.11698  SE=0.08046  g/SE=+1.45  pos=6/8  split=30/35
               DELAYED hampel_filter macro=-0.11299  SE=0.06958  g/SE=-1.62  pos=3/8  split=31/35
```

两个设置均满足 §13 全部四条：macro>0、g/SE≥3、delayed 同方向、半区不翻转。
split-half 100% 对照 D5 同一 8 条 eval series 上的 45.7%。

`hampel_filter` 的失败是**测出来的**：§12.2 事先允许其胜出，判据未作任何调整。

### 14.3 提案冻结设置

```yaml
proposed_strength:
  region:    training-window target only    # raw[anchor : anchor+48]
  amplitude: 8.0 x nanstd(series[120:900])
  count:     40 spikes per faulty series, 随机位置与符号, seed=7
  faulty:    6 of 12 train series
```

选 amp=8 而非 16 的理由：§12.3(d) 要求"达到可读后立即冻结"，8 是**清过全部四条门槛的
最弱设置**。若执行者在真实 runner 上复现得更弱，amp=16（delayed g/SE +2.70）为唯一备选，
不得再往上加。

### 14.4 必须随报告披露的两项限制

**(a) target 集中注入不是"真实故障"。** 真实传感器尖峰同时命中 context 与 target，
而 `both` 配置下净效应接近零。本设置是为 substrate qualification 刻意做的 target 集中，
**不得描述为真实故障场景**。真实性检验属于 §7 T5 的自然数据任务。

可以诚实陈述的版本：*在本 consumer 上，输入侧稳健性已由 median/MAD 标准化解决，
readiness 的瓶颈在标签侧* —— 这是一条测出来的发现，但**范围仅限本 rig、本算子族、
本 consumer**，不得外推。

**(b) 曝光账本。** 本次标定已在 origins `[1104, 1368, 1800, 2856, 3648, 3888]` 上，
对 identity / outlier_mad / hampel_filter 三臂、6 组注入配置打开了 outcome cells。
这 6 个 origin 与 8 条 eval series **永久标记为 development exposed**。
T3 若需要 fresh target，不得复用；需要新 series 时须另行取得用户曝光批准。

### 14.5 成本探针

单次 `_evaluate` ≈ **0.685 s**（120 训练窗 × 192 维 Ridge，含载入）。
一次完整 probe（3 臂 × 3 origin）≈ 6 s。§4.4 的 `B=3` 预算在算力上没有约束力，
Task Episode 数量可由设计需要决定，不受成本限制。

---

## 15. 【提案·待 Planner 裁定】§14 与 T0 实执行的冲突消解（2026-08-17）

状态：`PROPOSAL_PENDING_ADJUDICATION`。回应报告中的 `open_planner_item`。
**本节取代 §14.3 的强度提案**（§14.1/14.2 的测量事实保留，§14.3 的具体数值作废，理由见 15.3）。

**T0 已冻结的 verdict `TASK_EPISODE_SUBSTRATE_READABLE_UNDER_S13` 不撤销**，它按其字面口径为真，
作为历史行保留。本节提出的是**重新冻结强度并重跑**，不是推翻既有结论。

### 15.1 两份结果不冲突：各测了独立两轴的一半

注入几何有两条独立轴，此前两方各只覆盖一半：

```text
轴 A  region：故障落在训练窗 context 区 [0,192) 还是 target 区 [192,240)
轴 B  phase ：每窗注入位置固定（相对 anchor 恒定）还是随机
执行者跑的是 (context, fixed)；§14 跑的是 (target, random) 与 (context, random)
缺 (target, fixed)
```

补齐 2×2（执行者冻结的 support block `[1104,1800,2856]`，逐训练窗 4 尖峰，6/12 faulty，
clean identity = 1.3682）：

```text
region   phase    amp | inj_identity  x_worse |  mad_gain      SE    g/SE   pos
context  fixed     10 |       5.0392     3.68 |   +3.7803  0.4670  +8.10   8/8
context  fixed      3 |       1.3711     1.00 |   -0.0116  0.0339  -0.34   3/8
context  random    10 |       1.4556     1.06 |   +0.1495  0.0711  +2.10   6/8
context  random     3 |       1.3588     0.99 |   +0.0295  0.0279  +1.06   5/8
target   fixed     10 |       1.4617     1.07 |   +0.0781  0.0574  +1.36   5/8
target   fixed      3 |       1.3846     1.01 |   -0.0295  0.0553  -0.53   5/8
target   random    10 |       1.9261     1.41 |   +0.4338  0.0672  +6.46   8/8
target   random      3 |       1.4887     1.09 |   +0.0337  0.0158  +2.13   7/8
```

`(context, fixed, amp=10)` 复现出 inj_identity = **5.039**，对照执行者报告的 **4.926**——
复现忠实，双方实现一致。

**§14.1 的结论需修正**：原文称"context 侧注入不伤模型"。该结论**只对随机相位成立**。
固定相位的 context 注入极具破坏性——每一行训练数据在同一特征位置都有巨大尖峰，
Ridge 会拟合出一个在干净 eval context 上无对应物的大系数。执行者报告中
"固定相位可能形成跨窗对齐"的疑虑成立，且机制比其描述的更强。

### 15.2 决定性差别：一个是可标定旋钮，一个是悬崖

```text
(context, fixed)   amp 3 -> 1.00x 无伤        amp 10 -> 3.68x 摧毁      中间无可选点
(target, random)   amp 4 -> 1.11x   ...  amp 8 -> 1.28x                单调剂量-反应
```

§12.3(d) 要求"达到可读后立即冻结"。**该指令在悬崖型旋钮上无法执行**——不存在"最弱的
通过设置"，只有"关"和"灾难"。

### 15.3 提案：在两个都通过 §13 的配置中选一个

**这不是改判据（D-6 不适用），是在都过关的选项里做选择。** §13 四条一字未动。
选择依据全部为实测，无事后新增标准：

```text
1 可标定性     target+random 单调可挑点；context+fixed 无中间态，无法执行 §12.3(d)
2 基线完整性   1.18x  vs  3.68x。identity 被打坏 3.68 倍时，+3.578 的"增益"
               是在回收自己制造的灾难，属稻草人基线
3 机制诚实性   context+fixed 的收益来自模型记忆了一个相位锁定的人工痕迹，
               不是数据质量故障；执行者已自行标出
4 种子稳健性   target+random @ amp6 在 seed 11/23/47 上全部通过（见 15.4）
```

**提案冻结设置（取代 §14.3）**：

```yaml
region:    training-window target only        # raw[anchor : anchor+48]
phase:     random per (series, anchor)
amplitude: 6.0 x nanstd(window context)       # 每窗 scale，沿用 minipipe _scale 语义
count:     4 spikes per training window per faulty series
faulty:    6 of 12 train series
blocks:    沿用执行者已冻结的 support [1104,1800,2856] / delayed [1368,3648,3888]
scheme:    沿用执行者的逐训练窗注入实现
```

**采纳执行者实现的部分**：其"逐训练窗注入"几何优于 §14.3 的"整序列撒 n=40"——
每窗剂量一致，不受窗重叠影响。§14.3 的 `n=40/series, seed=7` 数值作废。

选 amp=6 的理由：它是清过 g/SE ≥ 3 的**最弱**设置（amp 5 → +2.44 未过），
符合 §12.3(d)。备选只有 amp=7。

### 15.4 提案设置的完整 §13 复核（3 个种子）

```text
amp=6 seed=11 | SUP.mad +0.143 g/SE +3.12 sh 35/35 | SUP.ham -0.263 g/SE -3.98 sh 35/35
              | DEL.mad +0.187 g/SE +2.55 sh 35/35 | DEL.ham -0.116 g/SE -1.39 sh 28/35
amp=6 seed=23 | SUP.mad +0.062 g/SE +5.48 sh 35/35 | SUP.ham -0.379 g/SE -3.77 sh 35/35
              | DEL.mad +0.188 g/SE +4.13 sh 35/35 | DEL.ham -0.328 g/SE -5.63 sh 35/35
amp=6 seed=47 | SUP.mad +0.123 g/SE +3.80 sh 35/35 | SUP.ham -0.257 g/SE -3.77 sh 35/35
              | DEL.mad +0.044 g/SE +2.09 sh 35/35 | DEL.ham -0.230 g/SE -3.04 sh 35/35
```

§13 四条：macro>0 ✓；SUP g/SE ≥ 3 ✓（3.12 / 5.48 / 3.80）；delayed 同方向 ✓（全正）；
半区不翻转 ✓（mad 全 35/35）。comparator 稳定为负，但仍未预设其必输——是测出来的。

### 15.5 附带提案（独立裁定项）

建议 T0 报告新增一个**披露字段**，不作通过条件：

```yaml
baseline_degradation_ratio:  injected_identity_smase / clean_identity_smase
```

理由：§13 四条能检出"反馈是否可读"，但检不出"基线是否被打成稻草人"。
本轮正是靠这个比值（3.68 vs 1.18）才分辨出两个同样"通过"的配置。

**明确标注**：这是事后发现的口径缺口。**提案只作披露，不追溯改变任何已冻结 verdict**，
是否升格为通过条件由 Planner 另行裁定。

### 15.6 若否决本提案

则冻结执行者的 `(context, fixed, severe)` 配方进 T1，但报告与对外表述中必须写明：

- identity 基线被注入劣化 3.68 倍，`+3.578` 的增益是相对该劣化基线；
- 故障为相位锁定人工痕迹，非真实数据质量故障；
- 该强度不可标定（amp 3 无伤、amp 10 摧毁），不得称已按 §12.3(d) 取最弱可读设置。


---

## 16. Source 权限修复与 permission replay（2026-08-17，已执行）

状态：`SOURCE_PERMISSION_BOUNDARY_REPLAY_PASS`。本节是唯一执行项，不产生新的 A5/A3 转移 Claim。

### 16.1 clean_replay_v2 的通道分解（审计结论，已修正采纳）

`A5_NEGATIVE_TRANSFER_CLEAN_REPLAY_V2` 的效应来自两个独立通道，不能作为一个假设处理：

```text
TRUST 通道   target_01 候选顺序两臂相同；同一 probe、同一 gain、同一 gate，
             仅 A3 CONTINUE / A5 TRUST_DRAFT 不同。
             它解释 utility 差（0.16672），不解释 harm 差。
排序通道     target_02/03 的候选路径不同。
             它解释全部额外 probe 与 harm 差（0.2355）。
```

因此“Source 只用于候选排序、不进入 Support 后 Promotion 决策”的修复：
- 必要，但只切断 TRUST/utility 通道；
- 不会改变排序通道的 harm。排序 harm 是独立阻塞，不随本次修改处理。
- `g/SE ≥ 3` 是 T0 可读性正控，**不是 Runtime Gate**。不事后升级为 A5/A3 Gate。
- `MATERIAL_THRESHOLD = 0.005` 仍冻结，不在本节处理。

### 16.2 唯一 Harness 修改：权限边界（§5 原文恢复）

`evaluation/functional/task_episode_harness/a5a3.py`：

```text
候选提案/排序   _memory_initial_order  ← Source Experiences + Target Experiences
Support 后决策  _memory_agent_decision ← 仅 last_probe + remaining_programs
                                          + threshold + allowed + target_experiences
```

实现要点：
- `_run_arm` 拆为 `source_memories` 与 `target_memories` 两条独立流；Source bank 只参与 initial order。
- `_decision_input` 是无 arm 标识的确定性 payload；`target_experiences` 投影剔除内部 `episode_id`
  （否则 arm 前缀会进入决策输入，破坏 A3/A5 同证据比较）。
- Prompt/Memory 组织/Context 特征/候选池/门控均未改。
- 聚焦测试 `tests/functional/test_a5a3_protocol.py`：6 passed（含确定性输入一致、
  LLM payload 无 `source_experiences`、机械检查读嵌套 `decision_input`、泄漏会判 FAIL）。

### 16.3 机械检查口径（不用一次随机 LLM 输出当证据）

`_trust_channel_mechanical_check` 定义相同证据为：

```text
program + support_gain + support_se_block + support_gain_over_se
+ remaining_programs + target_experiences
```

对每个相同证据对逐字节比较 decision input。本次行为 replay 结果：

```text
runtime_same_evidence_pairs = 3
  target_01 #0 outlier_mad    identical
  target_01 #1 hampel_filter  identical
  target_01 #2 winsorize      identical
runtime_pairs_all_identical  = true
source_key_in_any_decision_input = false
verdict = SOURCE_PERMISSION_BOUNDARY_REPLAY_PASS
```

观察到的 LLM 决策（仅描述性，不构成结论）：

```text
target_01 #0  两臂都 CONTINUE
target_01 #1  两臂都 CONTINUE
target_01 #2  输入逐字节相同，但 A3 = REQUEST_OBSERVATION，A5 = TRUST_DRAFT
```

#2 这一对再次证明：单次随机 LLM 输出不能验证权限边界，确定性输入检查才是判据。

### 16.4 本轮行为 replay 的聚合（描述性事实，不作 Claim）

```text
A3  10 probes / 0 drafts / 0 delayed approved / harm 0.6752
A5  11 probes / 2 drafts / 2 delayed approved / harm 0.9190
```

- A5 本轮 utility 数字高于 A3，但这同样是采样行为，**不得**反向宣称为 A5 更优。
- `delayed_confirmed` 自本版起改名 `delayed_approved`：它只表示现有机械 Gate 批准，
  不表示统计上已确认。历史 `clean_replay_v2` 字段名保持原样。
- 保留的历史判定：`A5_NEGATIVE_TRANSFER_CLEAN_REPLAY_V2` 仅表示“当前 A5 设计
  在该 development cohort 上发生描述性负迁移”，它足以授权本节 Control 修复；
  “A5 具有可复现负迁移”**尚未建立**（resolution census 最大 |g/SE| ≈ 2.27，
  决定 utility 的 delayed g/SE 是 +1.07 对 −0.39，都在 1 SE 内）。

### 16.5 下一项顺序（已冻结，不并行）

1. 排序通道 harm 作为独立阻塞保留，下一轮单独给规格，不与 TRUST 修复同改。
2. 正式 Runner 恢复多-origin Task Episode：每 Episode 多个 Support origins +
   多个 Delayed origins，所有 origins 全程唯一、严格前向。
   只有长序列可用时间块不足时才申请增加 eval series；不无限扩 series。
3. 在 development 上 TRUST 权限正确、且排序先验不再明显增加 harm 后，
   才申请 virgin cohort 正式曝光。暂不批准 virgin cohort，也不要求多 Context
   （当前 Source bank 基本只有一个公开 Context，混入 Context 检索会混淆假设）。


### 16.6 origin 容量反例确认（零 LLM，零 outcome）

对“降低 SE 只能增加 eval series”的反例已用 `/tmp/sehts-venv`（numpy 2.5.2）复核：

```text
CONTEXT_LENGTH = 192，HORIZON = 48，最大训练 anchor = 852
first valid origin >= 901；selected eval series 最小长度 = 10898
按 240 宽（192 context + 48 truth）互不重叠块计数：
可用块 = 42
需求   = 4 Episodes × (3 Support origins + 3 Delayed origins) = 24
结论   = 当前 eval series 就足够，无需先扩 series
```

示例形状（仅为容量演示，**不是**下一版正式 Runner 的冻结 origin 表）：

```text
ep1 support [901,1141,1381]   delayed [1621,1861,2101]
ep2 support [2341,2581,2821]  delayed [3061,3301,3541]
ep3 support [3781,4021,4261]  delayed [4501,4741,4981]
ep4 support [5221,5461,5701]  delayed [5941,6181,6421]
```

下一版正式 Runner 的 origin 表将按同一原则单独预注册：全程唯一、严格前向、
块间不重叠；正式冻结前不打开任何 outcome。


---

## 17. Workflow Generation A5/A3 development slice（预注册，2026-08-17）

状态：`PREREGISTERED_BEFORE_OUTCOME`。本切片的生成候选 outcome 尚未打开；
以下口径先冻结，再执行 `evaluation/functional/task_episode_harness/workflow_gen.py`
（Runner phase `workflow-generation`）。

### 17.1 问题与范围

固定池排序已暂存为窄负结果。本切片回答下一个主线问题：

> Source Experience 能否帮助 LLM 生成更有效的数据适配 Workflow？

范围：
- development diagnostic：一次自然 Task Episode、每臂一次随机生成轨迹；
- 不消耗 virgin cohort；Target 是已曝光自然数据；
- 不新增 Gate/Schema/retriever/统计平台；使用现有
  `compile_workflow_proposal`、`_evaluate_origins`、`_arm_metrics`、
  `TTHAMethod.handle_fast_winner` / `handle_feedback_delayed`。

### 17.2 Target 与 Source 冻结

```text
Target task       natural_k1_03（已曝光；固定池最终 AGENT_ABSTAIN）
Support origins   [1104, 1368, 1800]
Delayed origins   [2856, 3648, 3888]
Scope             沿用 natural_k1_03 已冻结 agent_scope；recompute 不一致则
                  RuntimeError，不发 LLM
固定池反馈        该 Episode 的 3 条实际 Support 轨迹，两臂都可见
Source tasks      natural_k1_01 / natural_k1_02 / natural_k1_04
                  natural_k1_03 自身的轨迹从 Source 中排除
A3                只读 Target Context + 固定池反馈 + operator inventory
A5                额外读 Source 成功/失败/冲突 Experience
```

固定池 best support gain = `outlier_mad +0.05719`。生成候选超过它记
`beats_fixed_best_support=true`（描述性，不作胜负）。

### 17.3 生成与 Runtime 闭包

- 每臂恰好 1 次 LLM generation call，A3 先、A5 后，模型与 prompt 相同；
- 提案 1–4 个 EXECUTABLE canonical operator，可单步、可组合；也可 ABSTAIN；
- `compile_workflow_proposal` 编译；非法/未知算子/错绑定 fail closed；
- Support probe 使用同一 Support origins 与同一 scope；
- **Target Support 决定 Draft**：`macro_gain >= MATERIAL_THRESHOLD(0.005)` 才
  Draft；本切片没有 LLM TRUST/CONTINUE 决策；
- Draft 后 `handle_fast_winner` 使用真实 `compiled.program.execution_steps()`
  （组合 Workflow 也走同一 Skill body），delayed 用同一 compiled 候选；
- `handle_feedback_delayed` 照常批准/拒绝；Source 没有批准权；
- 生成候选若 instrument 失败（pipeline raise），记录
  `INSTRUMENT_FAILED`，不伪装成负经验。

### 17.4 输入完整性预断言（零 LLM）

A3/A5 的 generation payload 去掉 `source_experiences` 后必须逐字节一致；
两臂唯一差异是 `source_experiences` 内容。检查失败则不发 LLM。

### 17.5 预注册 verdict 表（只看 Draft 是否形成，不引入显著性）

```text
两臂都无 Draft   WORKFLOW_GENERATION_NO_EFFECTIVE_CANDIDATE_DEV
仅 A5 有 Draft    A5_SOURCE_GENERATION_ADVANTAGE_DEV_SIGNAL
仅 A3 有 Draft    A3_TARGET_ONLY_GENERATION_ADVANTAGE_DEV_SIGNAL
两臂都有 Draft    WORKFLOW_GENERATION_BOTH_EFFECTIVE_DEV
```

所有标签均带 `_DEV` 或 `DEV_SIGNAL`：一次随机 LLM 轨迹 + Source/Target 同
K1 cohort、origin block 大量重叠，不得外推为可复现结论。若两臂都生成不出
有效 Workflow，首阻塞记为 Program Supply/headroom，而不是 Source 失败。

### 17.6 已知限制（预注册，不事后补充）

1. 固定池与生成候选共享部分 evaluation series，且自然 Episode 之间 origin
   blocks 重叠；这是“已曝光 development 数据”的固有属性，只作诊断。
2. `representative_binding_series` 规则冻结为：scope 内
   `local_robust_z_peak` 最大的 train series，前缀到第一个 Support origin
   `1104`；本次预计算得到 `T126`。它只用于把公开 binding feature 解析成
   `compile_workflow_proposal` 所需的数值 Context。
3. 每臂一次生成调用：LLM 随机性未控制，单轨迹不作统计结论。
4. 本切片不比较 fixed-pool 排序能力，也不重复 clean_replay_v2。


### 17.7 执行结果（2026-08-17，单轨迹 development diagnostic）

输入预断言通过：A3/A5 去掉 `source_experiences` 后逐字节一致；
A3 0 条 Source，A5 9 条 Source 轨迹。

```text
A3  COMPILED
    proposal = [repair_level_shift]，bindings 来自预注册 T126 Context
    support  = -0.02830  SE 0.05312  g/SE -0.533
    Draft    = false；无 delayed
    意义     = Target-only 生成了一条 fixed-pool 之外的 novel Workflow，失败

A5  COMPILED
    proposal = [outlier_mad] params {}
    experience_use = natural_k1_01/02/04（task id 粒度，模型输出）
    support  = +0.05719  SE 0.05448  g/SE +1.050
    Draft    = true；delayed = +0.04402，SE 0.09188，LOCAL_ACTIVE
    意义     = Source 帮助选择了 fixed-pool 中已知正向候选并推进到 delayed
```

诊断字段：
- A5 `fixed_pool_reuse=true`，`support_gain_matches_fixed_feedback=true`。
  A5 的 Support gain 与 natural_k1_03 fixed-pool `outlier_mad` 完全一致，
  本次只新开了 delayed outcome，没有新开 Support outcome。
- A3 `fixed_pool_reuse=false`：novel 生成真实发生，但没有 headroom。
- 因此 `A5_SOURCE_GENERATION_ADVANTAGE_DEV_SIGNAL` 的含义只能是：
  “本轨迹中 Source 让 A5 选到可推进候选、Target-only 选择的新候选失败”。
  它**不是**“Source 帮助生成了更有效的新 Workflow”的证据。

### 17.8 结果派生的下一项单 Control 修改（待批准）

预注册 verdict 表未覆盖“候选与 target fixed-pool 已探候选重复”的降级，
本轮按描述字段披露，不改 verdict。下一轮只允许一个修改：

```text
GENERATION_NOVELTY_GUARD
生成提案的 op 序列不得与当前 Target 的 fixed-pool 已探轨迹完全相同。
```

- 这是生成阶段的合法性约束，不是新的统计 Gate；不改 MATERIAL_THRESHOLD、
  不新增 Runtime 决策者、不扩算子池。
- A5 仍可 REUSE/MODIFY Source Workflow，但必须相对 Target 已探轨迹有
  新的算子序列或参数绑定，否则按 `COMPILATION_FAILED:NON_NOVEL_CANDIDATE`
  记录，不发 Support。
- 通过后再在同一 natural_k1_03 上重开一轮 development 诊断，才可能回答
  原始问题“Source 能否帮助生成更有效的 Workflow”。
- 当前不重跑、不消耗 virgin cohort。
