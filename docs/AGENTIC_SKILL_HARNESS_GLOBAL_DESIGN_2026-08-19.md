# Agentic Skill Harness 全局实现设计 v0.5

~~~text
status: G1_G2_COMPLETE / G3_SKILL_ONLY_SOURCE_TRANSFER_PENDING
date: 2026-08-19
scope: GLOBAL_ARCHITECTURE_AND_IMPLEMENTATION
implementation: CORE_PIPELINE_RUNNING / SOURCE_TO_SKILL_ROUTE_NOT_YET_TESTED
reviewers: USER / OPUS / KIMI / GPT
evidence_cutoff: RAW_SOURCE_EPISODES_TO_FAST_REJECTED / ELECTRICITY_DEVELOPMENT_EXPOSED
~~~

## 0. 文档定位

本文把此前分散在 Fast Path、Experience、Skill Card、General guidance、有限归因和
Task Episode 实验中的设计统一成一套可实现的 Agentic Harness。

它回答五个问题：

1. General Skill、Specific Skill Card 和 Experience Episode 分别装什么；
2. Episode 如何在 Runtime / Slow 内部形成或修改 Skill，以及 Fast Agent 如何只消费 Skill、
   Workspace 观察与已发生的 Target Support；
3. Fast Path 如何执行、验证并累积经验；
4. Slow Path 如何根据失败修改 General、Specific 或其他 Harness Surface；
5. 如何按“先接通全局闭环、再依据真实 first fault 调整”的方式实施，而不继续逐零件建实验。

本文不改写历史实验，也不把既有 development 结果升级成 fresh Claim。旧文档继续保留其
实验记录和局部协议价值；本设计经审核冻结后，作为后续整体实现的架构入口。

相关旧文档的定位：

- BOUNDED_SKILL_CARD_ATTRIBUTION_DESIGN_2026-08-18.md：保留有限归因细节；
- EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN_2026-08-17.md：保留 E0/E1/G1 历史路线；
- TASK_EPISODE_HARNESS_EXECUTION_PLAN_2026-08-17.md：保留 Task Episode 仪器与执行记录；
- V1_FASTPATH_FRAMEWORK.md：保留早期确定性 Fast Path 组装证据。

本文不新增 Schema、通用 Ledger、Hash Chain、向量库、Pattern Graph 或长期实验平台。

本文的知识流以仓库本地 `AGENTS.md` 为最高优先级。历史文档中把 Raw/aggregated Episode
直接送入 Fast Agent 的路线仅保留为历史实验记录，不再属于本项目主线。

---

## 1. 锁定目标

项目最终目标仍是：

> 针对时序数据中质量标准随任务、模型和局部时序模式变化的问题，构建 Agent 驱动的
> Data Readiness Harness，实现从数据与模式理解、Workspace 工具调用、Workflow 适配、
> 下游反馈验证，到 Harness 自我更新的闭环。

近期承重里程碑仍是 A5 vs A3：

> 在相同 Target downstream-feedback budget 下，使用由 Source 成功、失败和冲突
> Experience 形成的 Source-derived Skill 的 A5，能否比没有该 Source-derived Skill 的 A3
> 更快、更安全地形成有效 Target-local Skill。

固定边界：

- A3 和 A5 使用相同 Agent 流程、Workspace 工具、Operator DSL、Runtime、Consumer、
  Judge 和 Target Support 预算；
- 唯一实验差异是 A5 能读取在 Target Outcome 打开前冻结的 Source-derived General Skill
  （或另有独立多域证据授权的 Shared Capability），A3 没有该 Source-derived Skill；
- Raw/aggregated Source Episode 不进入任一 Fast Prompt；Source Target-local Card 不因跨域
  Context 粗匹配获得执行权；
- 当前 Query future 始终 sealed；
- Tool observation 不等于 Target Support，也不读取 Outcome；
- Consumer、Metric、训练协议和 estimand 属于实验仪器，不是 Harness 可编辑面。

---

## 2. “全局实现 + 事后调”的准确含义

本计划采用以下推进方式：

~~~text
先实现一条完整、可运行的核心纵向闭环
→ 在已曝光 development 数据上整体验证
→ 记录第一个真实 first fault
→ 每轮只修改一个主要 Harness Surface
→ 闭环稳定后再冻结 fresh A5/A3
~~~

“全局实现”指一次接通完整核心路径，而不是一次建设所有未来能力。

本轮需要整体接通：

- General / Specific Skill 检索，以及 Episode 到 Runtime / Slow 的内部证据路由；
- Fast Agent 多步观察、提案与选择；
- Workspace 工具调用；
- Typed Workflow 编译、逐作用单元绑定与执行；
- Support / delayed feedback；
- Episode 写入与 Target-local Skill 生命周期；
- Runtime first-fault attribution；
- Slow 单 Surface 更新和 replay。

本轮明确不一次建设：

- 所有 Operator 专用观察工具；
- 所有归因 Cause 的自动修复器；
- Shared Capability 自动归纳平台；
- learned retrieval / embedding；
- 多 Task、多 Consumer 的完整矩阵；
- 新 Card Schema、复杂 Memory Store 或 Evidence Ledger。

实验前只保留四项最低检查：

1. 信息墙没有泄漏；
2. Workflow 可编译、可执行，提案意图与实际执行动作一致；
3. 打开 Outcome 前，对全部 train / eval action units 运行 substrate 合法性检查，包括与
   正式 evaluator 相同的 scale-floor 前提；
4. Judge 和报告能产生可读结果，并明确区分 real Support probes、charged probe cost、
   Task 数与 draft 数。任一臂 instrument-unreadable 时，整个 paired Task 必须从均值、
   方差、样本量公式和行为统计中排除并单独记录，不得标成两臂打平。

其余细节通过完整运行后的 first fault 调整，不在实现前逐项证明。

---

## 3. 术语与对象

| 对象 | 定义 | 是否直接执行数据 |
| --- | --- | --- |
| Experience Episode | 一次合法 Action–Response 的事实记录，保留成功、失败、冲突和 abstain | 否 |
| General Skill | 可跨 Episode 复用的调查方法、决策纪律和带证据范围的软先验 | 否 |
| Specific Skill Card | 某个 Domain / Context 内验证过的 Workflow 模板、适用范围和风险 | 提供候选模板，不直接绕过 Runtime |
| Shared Capability | 多 Domain 重复成立后获得更强跨域复用权的 Capability | 可优先进入候选；当前 A5 仍需 Target Support，只有后续 A4 扩权才讨论低/零探测执行 |
| Context | Task、Consumer、数据结构、局部 Pattern、可用 Skill 和本轮工具结果的动态集合 | 否 |
| Agent Decision Trajectory | Agent 观察、调用工具、生成候选、验证和选择的多步过程 | 否 |
| Typed Workflow / Program | 最终作用到数据上的有类型 Operator 序列、Scope 和参数绑定 | 是 |
| Fast Path | 当前任务内的检索、观察、提案、Support 验证、执行与经验写入 | 是 |
| Slow Path | 根据跨 Episode 失败或冲突修改 Harness 的过程 | 通过修改影响未来执行 |

必须保持以下区分：

~~~text
Skill != 当前任务的最终 Workflow
Workflow != Agent 的完整决策轨迹
Experience != 有执行权的 Skill
Tool observation != downstream Outcome
~~~

---

## 4. 三层知识设计

### 4.1 Experience Episode：事实层

任何合法 Action–Response 都立即记录：

- Task / Consumer / objective；
- 部署时可见 Context；
- 工具调用与公开结果摘要；
- proposed / compiled / executed Workflow；
- Scope 与参数绑定来源；
- Support response；
- delayed response 到达后原位更新；
- Positive / Negative / Conflict / Immaterial / Abstain；
- instrument validity。

Episode 不自动获得提案权或执行权。它的作用是：

- Runtime 检索和对照相似成功、失败与冲突，并做 first-fault attribution；
- Slow Path 从有边界的证据 census 形成或修改 General / Specific Skill；
- Fast / Runtime 生命周期将同域重复结果归纳为 Target-local Skill；
- 多域重复结果为 Shared Capability 提供证据。

Raw Episode、逐行 Episode 列表以及由 Episode 聚合出的独立候选菜单均不得进入 Fast
Prompt。Fast 只能看到已编译 Skill 携带的简洁 scope、risk、provenance 与 evidence-strength
注释；这些注释不能绕过 Skill formation，也不单独授予提案优先级或执行权。

前期继续用现有普通对象或 JSON 集合，不建设独立 Episode 平台。

### 4.2 General Skill：通用方法、纪律与软先验

General 不是“匹配所有数据的一张大 Specific Card”。它包含三类知识：

General 也不是写死的效用 Router。基础调查流程与确定性安全纪律可以由研究者初始化；由
经验归纳出的效用或风险内容必须经 Runtime / Slow 编译成可追溯、可撤销的 Skill 后才进入
Agent Context。是否“写死”不由作者身份决定，而由权限决定：经验 General 不得机械决定最终
Workflow，不得绕过 Workspace 观察、Runtime 校验或当前 Target Support。

#### A. 调查与工具使用方法

例如：

- 先检查哪个粒度的 Pattern；
- 什么情况下需要查看 series / channel / interval；
- 何时补充 scope heterogeneity、作用几何或参数前提；
- 何时停止观察并进入 Support；
- 何时因信息不足而 abstain。

#### B. 决策与风险纪律

例如：

- 正向先验不得绕过 Target Support；
- 冲突证据存在时不得把单个成功例外写成主动组合规则；
- 工具结果显示作用对象异构时不得使用单一任务级参数；
- 没有合法替代 Workflow headroom 时不得把失败解释为 Memory 不够。

其中“Runtime 必须怎样绑定参数”属于机械契约，不应只写成自然语言 guidance。General
只能要求 Agent 获取必要证据、选择合法动作；最终绑定由 Runtime 实现和强制。

#### C. 带证据范围的经验先验

General 可以保存跨 Context 重复出现的效用或风险先验，但必须遵守：

- 明确 evidence scope、适用 Context 和 provenance；
- 正向知识只能作为 guarded candidate 到达，必须由当前 Target Support 确认；
- 风险或回避知识默认是 soft prior，可用于降级、先补 Observation 或要求 Support；
- 只有确定性的结构安全条件，才可形成 Runtime hard guard；
- 一条规则在新 Domain 失效后，不再宣称已稳定跨域，而是收缩证据范围或降级为待确认先验。

因此，General 既不是纯方法论，也不是无条件的跨域效用规则。

A5 的运行时知识顺序冻结为：General Skill 提供调查方法、决策纪律和带证据边界的软先验，
当前 Domain 合法匹配的 Specific / Target-local Skill 提供本地 Workflow 模板，Workspace
观察与已发生 Target Support 决定当前动作。完整正/负/冲突 Episode 只供 Runtime / Slow
归因和 Skill 更新；Fast 不读取其原文或独立聚合表。

Slow 可以从完整 Episode contrast 编译简洁 General Skill，但必须保留证据范围与 provenance，
且不得把统计表直接伪装成候选菜单。Skill 是可修改的经验载体，不是对 Raw Episode Prompt 的
改名。

当前 Weather 证据表明：proposal deprioritization 虽能改变顺序，却可能使 Agent 先试一个
替代候选后仍回头试原候选，从而增加试错。因此“风险先验能够跨域减少成本”目前不是已冻结
事实，只能保留为需当前 Context / Support 验证的软先验。

当前阶段优先复用已有 Harness instruction / candidate policy / bootstrap Skill 表达
General，不立即新增 General Skill Schema。具体序列化形式由实现评审决定。

### 4.3 Specific Skill Card：Target-local 可执行经验

Specific 回答：

> 在当前 Domain 的这种可观察 Context 下，什么 Workflow 曾经有效，适用范围和风险是什么？

其逻辑内容包括：

- observable applicability；
- Workflow / Program 模板；
- 必要 Workspace tool / Observation 前提；
- allowed tools；
- risk / abstention；
- 本地 Support 与 delayed 状态。

Specific 的执行权只在当前 Domain 内成立。新 Domain 上：

- 只有当前 Domain 内匹配的 LOCAL_ACTIVE Card 才可直接进入可执行候选；
- Source 或未匹配的 Target-local Card 不进入 Fast Prompt，也不作为证据模板变相提供候选；
- 不得因 Card 存在就直接执行；
- 其历史证据只能由 Runtime / Slow 用于形成 General / Shared Skill，或在新 Domain 重新形成
  Target-local Skill；
- 不要求匹配率必须高。

一次正向 Action–Response 只先形成 Positive Episode。形成卡的顺序是：

~~~text
Positive Episode
→ 当前选择切片改善
→ LOCAL_DRAFT
→ 同 Domain held-in / 后续反馈确认
→ LOCAL_ACTIVE
~~~

### 4.4 Shared Capability：后续扩权，不是当前前置

只有多个 Domain 的相似可观察 Context 中重复出现正向和风险证据，才讨论 Shared
Capability。它不要求当前阶段建设第三套存储或自动 Promotion 平台。

---

## 5. Context 是动态工作状态，不是一行固定特征

### 5.1 初始 Context

Fast Agent 开始时至少获得：

- TaskSpec / Consumer / Metric / objective；
- Dataset / Cohort 规模与可见 DataSemantics；
- series / channel / interval 的初始公开摘要；
- Operator contracts 与执行前提；
- Target Support 预算；
- active General Skill 与当前 Domain 合法匹配的 Specific / Target-local Skill。

初始 Fast Context 不包含 Raw Episode、Episode rows、`source_experiences` 或独立的 Episode
aggregate。Source 证据必须先经过 Runtime / Slow 形成 Skill。

Dataset 名称可用于读取 Target-local Skill，但不得作为跨域相似性的理由。

### 5.2 动态 Context

Agent 每次调用 Workspace 工具后，确定性结果追加到本轮 Context：

~~~text
initial_context
+ tool_observation_1
+ tool_observation_2
+ retrieved_skill_scope_and_risk
+ occurred_target_support
= current_decision_context
~~~

因此 Agent 不需要在首次 Prompt 中收到所有预计算特征，也不依赖一条代表序列概括整个
Cohort。

### 5.3 决策粒度

Context 和 Action 至少允许以下层级：

~~~text
dataset / cohort
→ series / channel
→ interval / local pattern
→ program action unit
~~~

同一个 Dataset 可以同时存在多个 Specific Skill 和多个不同 Workflow。任务级判断不能
自动广播为所有 series / interval 的参数。

---

## 6. Workspace Tool 契约

### 6.1 Tool 的职责

Workspace Tool 负责确定性读取部署时公开数据并返回事实。Agent 决定调用什么工具、何时
停止观察以及提出什么 Workflow。

工具必须：

- 不读取 Query future / delayed Outcome；
- 不返回 Consumer Utility；
- 输出确定、可重算；
- 绑定明确的数据对象和时间窗口；
- 只返回完成当前决策所需的摘要或定位；
- 调用次数有界；
- 不接受任意文件路径或任意代码执行。

### 6.2 初始工具集合

优先复用现有：

- summarize_series；
- localize_regions；
- read_fixed_probe_panel（已有固定公开探测时）。

G1 不新增 `inspect_level_excursion_scope`。绑定审计和已曝光数据 replay 已确认 legacy
external binding 错误，同时确认 `repair_level_shift` 的 intrinsic 路径能在当前 action unit 内
形成局部动作。该 replay 没有证明 Agent 缺少新的定位 Observation，因此不授权新工具；只有
后续 Fast 轨迹把“Agent 看不到一个部署时公开且会改变 Program 选择的事实”定位为 first fault，
才允许增加一个最小 operator-specific 工具。不得预建通用 Pattern 工具平台。

### 6.3 三种成本分别记账

- Workspace tool calls：公开数据观察成本；
- LLM calls：Agent 推理成本；
- Target Support / delayed cells：不可再生反馈成本。

charged probe cost 与真实 Support probe 数继续分开。工具调用不伪装成 Support。

---

## 7. Fast Path：Agent 多步决策轨迹

正式 Fast Path 采用：

~~~text
TaskSpec + initial Context
→ retrieve active General Skill / current-Domain Specific or Target-local Skill
→ INSPECT：Agent 调用受限 Workspace tools
→ PROPOSE：Agent 生成 1..B 个 Typed Workflow 候选
→ COMPILE：Runtime 校验 Operator、Scope、Binding provenance 与信息墙
→ SUPPORT：在相同 Target feedback budget 下探测
→ SELECT：执行、尝试下一个候选、请求 Observation 或 abstain
→ WRITE EPISODE：立即保存 Action–Response
→ LOCAL SKILL：满足同域条件时形成 DRAFT / ACTIVE
→ DELAYED UPDATE：未来 outcome 到达后更新同一 Episode 和 Skill
~~~

### 7.1 A3 / A5 公平性

两臂必须使用同一条 Agentic Fast Path：

| 项 | A3 | A5 |
| --- | --- | --- |
| Workspace tools | 相同 | 相同 |
| Tool-call bound | 相同 | 相同 |
| Operator inventory | 相同 | 相同 |
| Target Support budget | 相同 | 相同 |
| Runtime / Judge | 相同 | 相同 |
| Target-local 学习权 | 有 | 有 |
| Source-derived General Skill | 空 | 有，在 Target Outcome 打开前冻结 |
| Raw/aggregated Source Episodes | 空 | 空 |
| 跨域 Source Target-local Card | 无执行权 | 无执行权 |

不得给 A5 额外 Support，不得把工具调用当成免费 Outcome，也不得让 A3 使用 A5 运行后形成
的 Target-local Skill。A5 的额外输入必须通过 resolved Harness / Skill view 到达，不能通过
`source_experiences`、Episode aggregate 或未匹配 Card 旁路 Skill formation。

### 7.2 Runtime 拥有机械正确性

Agent 可以：

- 选择观察工具；
- 选择 Operator / Workflow；
- 提议 Scope 语义；
- 引用公开工具结果；
- 选择继续试验或 abstain。

Runtime 必须：

- 验证 Observation 来源；
- 编译 Typed Program；
- 按 series / channel / interval / window 分派 action unit，并强制参数所有权；
- 拒绝把代表对象参数广播给不兼容 action units；
- 强制 Task、Operator、Risk 和信息墙；
- 记录实际执行 bytes 与提案意图是否一致。

自然语言 General guidance 不替代 Runtime Binding Contract。

### 7.3 参数与定位所有权

每个 Operator 的动态参数必须声明唯一所有者：

~~~text
RUNTIME_BOUND
  Runtime 根据绑定到同一 action unit、同一坐标系的合法公开 Observation 提供参数。

OPERATOR_INTRINSIC
  Runtime 只把当前 action unit 交给 Operator；Operator 在该单元内部定位并计算参数，
  找不到合法目标时返回 identity。
~~~

同一参数不得同时由两者声明。`targeting_mode=external_region` 与
`targeting_mode=intrinsic` 必须和实际实现一致；Runtime 不得在 intrinsic Operator 外再建一套
重复定位器。

当前 `repair_level_shift` 的 legacy contract 声明为 external region，但函数同时已有 intrinsic
路径，契约与实现不一致。G1 不再使用已确认失真的“代表序列 full-prefix 参数广播”路径。

已曝光 Support replay 表明 intrinsic 路径把中位改动点数缩小约 7--24 倍，因而机械几何修复
成立；但跨 cohort 的可复用效用未成立。G1 若保留该 family，只能保留
`OPERATOR_INTRINSIC + Target Support required` 的探索候选权限，不得由 legacy Card、General
条款或 legacy Episode 直接提升优先级。legacy 与 intrinsic 是不同 Program 语义；两者证据
不得因共享 operator 名称而合并。该结论不授权继续调阈值或新建 binder。

---

## 8. Workflow 的两层含义

为避免后续文档混淆，统一使用两个术语。

### 8.1 Agent Decision Trajectory

这是整个多步过程：

~~~text
检索 → 观察 → 工具调用 → 再观察 → 提案 → Support → 选择
~~~

它作为 Trace / Episode 的一部分记录，但不是直接执行在数据上的 Program。

### 8.2 Typed Workflow / Program

这是最终数据处理动作：

~~~text
Operator sequence
+ per-unit Scope
+ owner-resolved parameters（RUNTIME_BOUND 或 OPERATOR_INTRINSIC）
+ risk / fallback
~~~

示例：

~~~text
Agent trajectory:
  inspect scope
  → call bounded existing Workspace tools
  → compare current observations with the active Skill's scope, risk and evidence boundary
  → propose repair_level_shift family or abstain

Typed Workflow:
  for each eligible series/window:
      repair_level_shift()  # OPERATOR_INTRINSIC: localize inside this action unit
~~~

Specific Card 可以保存 Typed Workflow 模板和观察前提，但不得保存一次运行的动态数值作为
跨任务固定参数。

---

## 9. Fast Path 的经验与 Skill 更新

每次运行结束按以下顺序处理：

1. 保存完整但公开合法的 Episode；
2. Positive、Negative、Conflict、Abstain 一律保留；
3. 当前选择切片上有材料改善，可形成 LOCAL_DRAFT；
4. 同 Domain 后续未参与选择的历史切片确认后，成为 LOCAL_ACTIVE；
5. 新结果翻转时，收缩 Scope、增加 risk、拆分 Skill 或保留冲突，不追溯删除原 Domain
   已成立的本地经验；
6. 只有多 Domain 重复成立，才讨论 Shared Capability。

正向 Fast Path 可以：

- 新建 Specific Card；
- 更新已有 Card 的本地证据；
- 形成新的候选 Workflow 模板。

它不能仅凭一次正向结果：

- 写入硬 General 规则；
- 获得跨域直接执行权；
- 删除相反 Experience。

---

## 10. Slow Path：失败驱动的 Harness Update

### 10.1 触发

Slow Path 可以由以下情况触发：

- 负向或冲突 Experience；
- 可避免的重复 Support 试错；
- Agent 请求现有 Observation 无法提供的信息；
- Specific Card 在相似 Context 下失效；
- Program intent 与实际执行不一致；
- 同一 first fault 跨 Task 重复。

一次差结果足以触发诊断，但不自动授权 General 修改。

### 10.2 Runtime 先归因

LLM 不自报 Cause。Runtime 使用合法 Episode、工具结果、Program trace 和反馈确定：

- instrument / utility 是否可读；
- 问题发生在 Observation、Workflow/Binding、Decision、Scope/Risk 还是执行机械层；
- 是 SPECIFIC 还是 GENERAL 范围；
- 本轮唯一可编辑 Surface。

对外有限原因继续使用：

- CONTEXT_GAP；
- WORKFLOW_GAP；
- DECISION_GAP；
- NO_ACTIONABLE_EVIDENCE。

机械执行错误、信息泄漏和 Judge 不可读走确定性出口，不强塞给 Slow。

Program Binding 的处理规则：

- 若提案意图与执行 bytes 不一致，属于 IMPLEMENTATION / INSTRUMENT 出口；
- 若绑定机械执行正确，但现有 Context 无法为作用对象提供正确参数，属于
  WORKFLOW_GAP 或必要的 Observation + Program Binding 耦合修复；
- 未证明存在替代 headroom 时，不直接重写 Program。

### 10.3 Slow 只编辑一个 Surface

Runtime 向 Slow 提供：

- Cause；
- Repair scope；
- 唯一 Surface catalog；
- 对齐到可编辑对象的正、负、冲突证据；
- distinct Task 计数与 provenance；
- 现有 General 条款或 Specific Card 原文；
- 允许的 PATCH / ADD / ABSTAIN。

Slow 可以修改：

- General 调查/决策 guidance；
- Specific body / applicability / risk；
- Observation 或工具使用策略；
- Program / Binding 模板；
- bounded Program Supply；
- Memory / Control policy。

每轮只能修改一个主要机制。只有合法 Scope 无法由现有 Observation 表达时，才允许
Observation + Scope 作为一次耦合修复。

### 10.4 Slow 不批准自己

~~~text
Slow proposal
→ deterministic compiler
→ binding / information-wall validation
→ exposed replay
→ 后续 in-domain feedback
→ accept / restrict / reject
~~~

General 的每个主动条款都必须获得独立证据。完整 census 必须同时包含正、负、冲突和
immaterial，不得单边取样。

证据 provenance 的授权规则冻结为：`GUIDANCE_CONDITIONED` Episode 可以反驳、弱化或撤销
已有条款，但不能单独授权新的主动推荐、主动降权或例外组合条款。新主动条款必须达到冻结的
`UNGUIDED` distinct-Task 证据门槛，避免 guidance 用自己诱发的行为循环证明自己。

---

## 11. General / Specific 的更新边界

| 证据形态 | 默认写入层 | 到达 Fast Path 的权限 |
| --- | --- | --- |
| 单次合法运行 | Experience | 不直接到达；仅供 Runtime / Slow 与本地 Skill 生命周期使用 |
| 单 Domain 重复正向 | Specific LOCAL_DRAFT / ACTIVE | 当前 Domain 候选 |
| 单 Domain 风险或冲突 | Specific Scope / Risk 或 Slow evidence census | 经 Skill 更新后局部降级、Support 或 abstain |
| 多 Task 同 first fault | General 候选 | 需逐条款证据审计 |
| 多 Domain 重复正向 | Shared candidate 或 General guarded candidate | 当前 Target Support 前不直接执行 |
| 确定性结构安全条件 | Runtime Risk contract | 可硬阻断 |
| 跨域经验风险先验 | General soft prior | 降权、补观察或要求 Support；默认不硬禁 |
| 仅由既有 guidance 诱发的重复行为 | GUIDANCE_CONDITIONED Experience | 供 Slow 反驳/撤销旧条款；不直接到达 Fast，不得单独授权新主动条款 |

此前两种过度表述均不采用：

1. “General 只允许方法论，不能包含效用知识”——过窄；
2. “风险/回避知识天然比正向知识更能跨域”——当前证据不足。

冻结口径是：

> General 可以包含方法、决策纪律和带证据范围的经验先验；正向与负向经验都必须声明
> 适用范围，并由当前 Context、Runtime 风险和 Target Support 决定实际权限。

---

## 12. 当前代码的可复用基础

已有能力：

- TTHAAgentCore 已支持多轮 tool request / tool result；
- FastAgent 已有 inspect / propose / select 三阶段；
- LocalPublicToolGateway 已有受限公开工具；
- Typed Workflow compiler 和 Operator contracts 已存在；
- Support / delayed evaluator、Episode、Target-local Skill 生命周期已存在；
- Runtime attribution、General Surface patch、Slow 单 Surface 修改和行为 replay 已跑通；
- G1 Agentic Pipeline 与 G2 exposed-data shakedown 已完成，机械协议错误、instrument-unreadable
  与 infrastructure failure 在收口运行中均为零；
- paired-arm concurrency 已通过已曝光 Task smoke，保留跨 Task barrier 并约有 2 倍墙钟提速；
- electricity development 已验证 Workspace tool、Typed Program、Episode、Target-local Skill 与
  Slow/replay 链路，但不具备 fresh 身份。

当前尚未接通的承重路径：

- Source Episode 已能被保存、普查并交给 Runtime / Slow，但“Source evidence -> Skill -> Fast”
  尚未形成一次正式 A5/A3 对比；
- 当前 Runner 曾增加独立 `source_experiences` inlet 并把 13 条 T233 Episode 直接送给 Fast；
  该路线已由 development 负结果拒绝，不是待优化入口；
- true fresh G3 尚未运行；fresh sourcing 作为后台问题，不阻塞 exposed development 上的正确
  Skill-only 路线。

因此下一步不是从零建设 Agent，也不是继续改 Episode Prompt，而是复用现有闭环，把 T233
Episode 经 Runtime census 与 Slow 编译为 Source-derived Skill，再让 A5 Fast 只消费该 Skill。

---

## 13. 全局实施路线

### G0：冻结总体契约（已完成，v0.5 同步 Skill-only 路线）

本文件经 USER / OPUS / KIMI / GPT 审核后，冻结：

- 术语；
- General / Specific Skill 与 Episode 内部证据权限；
- A3 / A5 唯一差异；
- 信息墙；
- Agent 与 Runtime 所有权；
- Operator 参数与定位所有权；
- Episode、Specific、General 的证据权限与 provenance；
- 主读数；
- 一轮一个 first fault 的调整纪律。

本步不跑 LLM、不打开 Outcome。

新 fresh 数据的候选来源与长度/序列数等公开元数据可以在 G1/G2 期间做只读普查并保持
Outcome sealed；不在 G2 冻结前实现专用 loader、挑选实例或冻结最终 roster。

### G1：一次接通完整核心 Pipeline（已完成）

在一个逻辑 Runner package 中接通：

1. Context-conditioned retrieval；
2. General / Specific Skill 输入与 Episode 的 Runtime / Slow 内部证据路由；
3. FastAgent inspect / propose / select；
4. bounded Workspace tools；
5. Runtime per-unit compile / parameter-owner dispatch；
6. Support / execute / abstain；
7. Episode / Local Skill lifecycle；
8. deterministic attribution；
9. Slow single-surface edit；
10. replay 回到 Fast Path。

只允许一个主报告和一个承重 integration test。现有单元测试保留，不为此建设新测试矩阵。

G1 是集成实现，不要求每个子组件先各自形成实验结论。完成判据：

- 一条命令跑完整闭环；
- Agent 至少真实调用一次 Workspace 工具；
- 通过同输入同期对照或确定性引用 Trace，证明工具结果改变后续候选、Scope 或 abstention；
- Runtime 生成并执行 Typed Workflow；
- Episode 被写入；
- 正向结果能形成或复用 Target-local Skill；
- 负向/冲突能触发 Slow 或明确 NO_ACTIONABLE；
- Slow patch 通过 replay 后确实改变下一轮行为；
- 仪器失败的 paired Task 被排除而非标成打平；主报告明确区分 real/charged、Task/draft，
  且 fresh regression 不会被 replay verdict 遮蔽。

主报告同时记录探索集中度，但不把它设为 Gate：

- 每臂 `distinct_canonical_program_count`；
- 每臂 `distinct_operator_name_count / executable_operator_name_count`；
- 每臂 `top1_canonical_program_attempt_fraction`。

这些读数只用于判断 Agent 是否长期收敛在极少数 Program 上。Program 组合数与 Operator 数
不得混作同一分母；A3/A5 对同一 Task 的重复尝试在行为频率中可以分别计数，但在
Action--Response 证据授权中仍必须按底层 cohort + Task 去重。

### G2：已曝光数据上的整体 Shakedown（已完成）

使用已曝光的 KDD e31、T233 和 Weather development 数据：

- 不称 fresh；
- 不重新选择阈值追求正向结论；
- 只检查完整链路和当前 first fault；
- 每次发现 first fault，只修改一个 Surface；
- 同期 replay 检查修改是否改变预期位置；
- 保留负结果，避免直到成功为止的重试。

首个已确认 first fault 是 `repair_level_shift` 的 Program Binding contract：

- Weather 与 T233 各 12 个已审计 Task 的 legacy region 平均约 0.89 为 non-level 区域，
  24/24 都把 level offset 用于 non-level region；
- Weather 12/12 的 full-prefix 绝对 region 与实际训练窗口零重叠；
- T233 虽有绝对重叠，legacy 绑定仍平均修改约 93% 的 240-point window；
- 代表对象向 scope 广播的风险已确认。

第一项 bounded replay 已完成：在 e31 / T233 / Weather 的已曝光 Support cells 上比较 legacy
explicit binding 与既有 intrinsic 路径，零 LLM、零 fresh、未修改 registry：

- intrinsic 在 47/47 个 Task 上找到局部目标，identity 为 0；“intrinsic 会大量安全退回
  identity”的预测被否证；
- 中位改动点数由 e31 `55800 -> 8070`、T233 `65100 -> 3561`、Weather
  `4200 -> 174`，支持 `INTRINSIC_GEOMETRY_LOCALIZED`；
- 按实际 Runtime Draft 门 `gain >= 0.005`，intrinsic 在 Weather 为 19/19 material positive、
  0 material harmful，但 e31 为 5 positive / 7 harmful、T233 为 6 positive / 7 harmful；
- 按 `gain/SE >= 3` 计数为 1/47，但该门是 T0 substrate readability positive control，冻结文档
  明确禁止把它升级成 Runtime 或 A5/A3 Gate；
- replay 只重跑 Support origins，没有 delayed/held-in 确认，也没有通过正式 Agent/compiler
  路径生成 intrinsic Program。

因此冻结结论不是 `NO_PROGRAM_HEADROOM_UNDER_EITHER_BINDING`，而是：

~~~text
LEGACY_BINDING_CONTRACT_INVALID
INTRINSIC_GEOMETRY_LOCALIZED
WEATHER_INTRINSIC_SUPPORT_SIGNAL_ONLY
NO_CROSS_COHORT_REUSABLE_HEADROOM_CONFIRMED
~~~

`gain >= 0.005` 继续只决定 Target `LOCAL_DRAFT`；Target-local 执行权仍需 held-in/delayed
反馈；Shared/General 复用还需跨 Domain 证据。G1 可以把 intrinsic 版本保留为同预算下需
Support 的探索候选，但必须停用 legacy binding，并隔离所有由 legacy 行为产生的 Card、guidance
和 Experience 的执行授权。G2 后续价值是检查整体 Pipeline 能否从真实轨迹得到这一已知答案，
而不是继续为该 family 寻找阈值或 Observation。

现有历史提案的去重 Support census 还给出：裸 `outlier_mad` 为 24 个 material-positive、
3 个 material-negative distinct Task；legacy 裸 `repair_level_shift` 为 25 positive、23 negative。
该计数按底层 cohort + Task 去重，避免把 e31 replay 重复计为新证据。它证明当前已曝光数据中
存在比 legacy level-shift 更稳定的已观察候选，并提示历史探索高度集中；但它只覆盖 Agent
实际提出过的 Context，存在 proposal-selection bias，不是 21 个 Operator 的全 Task
counterfactual headroom census，也没有自动获得 delayed/Shared 权限。

因此 G1/G2 不先跑全 Operator 大矩阵。G2 观察完整 Agentic Pipeline 是否能利用合法 Skill、
Workspace Observation 与当前 Target Support 形成正向候选，并用内部 Episode 驱动后续归因
和 Skill 更新；只有它没有形成任何正向 Workflow，且 first fault 明确落在 Program
Supply/headroom unknown 时，才对至多两个由公开 Context 与历史 Episode 支持的候选做
bounded replay。不得为“先知道所有答案”遍历全部 Operator。

### G3：Skill-only Source Transfer（当前阶段）

#### G3-D0：Raw Source Episode inlet（已拒绝的替代路线）

在已曝光 electricity development 上，A3 使用空 Source bank，A5 通过独立
`source_experiences` inlet 直接收到 13 条 T233 Episode。8/9 个 paired Task 可读，结果为：

| 主读数 | A3 | A5 |
| --- | ---: | ---: |
| 首个 material-positive 所需 Task | 2 | 8 |
| harmful probes | 3 | 8 |
| cumulative harm | 0.083 | 0.626 |
| LOCAL_ACTIVE | 3 | 0 |
| prompt tokens | 426,704 | 601,985 |

A5 的首探 Program family 与 Source bank family 完全重合，并遗漏 electricity 上有效但 bank
中不存在的 outlier family。该实验原报告的数值保留，架构裁决记为：

~~~text
RAW_SOURCE_EPISODES_TO_FAST_REJECTED
~~~

这是一个可信的错误接口负结果：它拒绝“Raw/flat Episode bank 直接进入 Fast Prompt”，不代表
Skill-only A5 失败，也不代表 Experience 对 Skill formation 没有价值。不得通过重排、加权、
检索、压缩或聚合同一批 Episode 后再次直接送给 Fast 来追求正结果。

#### G3-D1：Source evidence -> Skill -> Fast（下一条唯一主线）

在继续寻找 fresh 数据之前，先在同一已曝光 electricity 上验证正确知识流：

~~~text
T233 Episodes
-> deterministic signed Source evidence census
-> Slow PATCH or ABSTAIN
-> compile and freeze a Source-derived General Skill
-> A5 Fast receives that Skill only
-> A3 receives the same baseline Harness without that Skill
-> paired electricity development comparison
~~~

Source-derived Skill 必须在读取任何 electricity Outcome 前冻结。Slow 返回 ABSTAIN 是合法负
结果，不得由研究者代写 Skill。Fast payload 中不得出现 Raw Episode、Episode rows、
`source_experiences` 或独立 aggregate；两臂除 Source-derived Skill 外完全相同。T233
Target-local Card 不提供给 electricity，也不修改 applicability、probe policy、Judge 或
Operator supply。

#### G3-F：后续 fresh confirmation

G3-D1 使正确 Pipeline 可读后，再选择新的未曝光自然数据：

- A3 / A5 同一 Agentic Pipeline；
- 相同 Workspace tool、LLM 和 Support budget；
- A5 只增加在 Target Outcome 前冻结的 Source-derived General Skill；A3 不含该
  Skill；Raw/aggregated Episode 与跨域 Target-local Card 对两臂都不可见；
- 打开 Outcome 前通过信息墙、Program/Binding、train+eval substrate 与报告读数完整性四项
  最低检查；
- 预注册 Tool calls、LLM calls、real Support probes、harm、首次正向 Workflow、
  LOCAL_DRAFT / ACTIVE 形成时间和 delayed utility；
- 一次性打开 fresh Outcome；
- 结果可以是正向或可信负向。

fresh sourcing 可与 G3-D1 并行做只读背景筛查，但不得因候选数据难找而阻塞 Skill-only
development 主线，也不得降低既定信息墙或数据合法性条件来凑 roster。

### G4：实验后扩展

只有 G3 的 first fault 要求时，才扩展：

- 新 Operator-specific tool；
- 新 Observation；
- 新 Task / Consumer；
- Shared Capability；
- 更强检索；
- 新 General Surface；
- Scope split / deprecation。

---

## 14. Weather 的当前定位

Weather 主实验已完成，不再是“是否先跑”的待决定事项。

当前可支持：

- Runtime attribution → clause evidence alignment → Slow autonomous patch →
  Fast behavior change 的机制链成立；
- autonomous General guidance 在 Weather 上明确改变了候选顺序；
- 仪器可读、无 ADD collision、Source Specific package 19/19 未匹配。

当前不支持：

- A5 比 A3 更快或更安全；
- current General guidance 带来跨域 Utility；
- post_shift_support_sufficient 是稳定的跨域效用边界；
- guidance 有显著 harm。

Weather 中 A5 真实 Support probes 为 31，A3 为 21；A5 仍探测裸
repair_level_shift 18 次，A3 为 19 次。说明当前 deprioritization 主要改变顺序，未避免
该候选，且可能增加前置试错。

Weather 已永久成为 development-exposed。后续用于：

- Context / Binding 诊断；
- Agentic Pipeline replay；
- Slow 更新的行为检查。

不得再称 fresh，也不得根据其 Outcome 调参后回头形成 Weather fresh Claim。

后续 Program Binding 审计确认 legacy action geometry 不可靠，但不能据此追溯改写 Weather
“当前实现下未证明收益”的结果。后续 intrinsic Support replay 在同一已曝光 Weather 上得到
19/19 material positive、0 material harmful；它是 development-only 的 Target-local 候选信号，
没有 delayed 确认，不能追溯改写原 A5/A3 结论，也不能升级成跨域 Utility。

---

## 15. 复杂度预算

首轮整体实现默认预算：

~~~text
new logical runner packages: 1
new primary reports: 1
new required integration tests: 1
new general schemas: 0
new hash systems: 0
new ledgers/registries: 0
new retrieval platforms: 0
new TS-native tools in G1: 0
~~~

优先复用：

- AgentCore tool loop；
- FastAgent inspect / propose / select；
- LocalPublicToolGateway；
- existing compiler / executor；
- current Episode / Skill lifecycle；
- current attribution / EditController / SnapshotStore；
- existing exposed data和报告。

---

## 16. 明确暂缓

- 不把全部 raw series 直接塞给 LLM；
- 不把 Raw/aggregated Source 或 Target Episode 直接塞给 Fast Agent，也不把 Episode aggregate
  当作独立候选菜单；
- 不允许 Agent 自由读取文件路径；
- 不把 Dataset ID 当跨域 Context；
- 不让 Slow 自报 Cause 或批准 Patch；
- 不因 General 有规则就绕过 Target Support；
- 不把 General guidance 当作硬 `Context -> Operator` 路由；
- 不让 Specific Card 自动跨 Domain 执行；
- 不同时修改 Observation、Program、Guidance 和 Risk；
- 不在 intrinsic Operator 外重复建设 Runtime 定位器；
- 不为工具调用建设通用 Receipt / SHA 平台；
- 不为所有归因分支预建修复器；
- 不在 Weather 上继续制造 fresh 结论；
- 不为了 demo 把项目降格成 Context-to-Operator Router。

---

## 17. 多方审核后的冻结决定与剩余实现选择

以下决定不改变总体架构：

1. General 的近期载体：
   - 当前不新增 Schema；
   - 基础调查方法与决策纪律优先复用 bootstrap Skill / Harness instruction；
   - `candidate_policy.proposal_guidance` 只能作为可修订的辅助 Context，不得成为经验
     General 的唯一载体或直接执行命令；
   - 经验性 General 必须由 Runtime census 与 Slow 从完整 Episode contrast 中形成或修改，
     通过 resolved Harness / Skill view 到达 Fast；原始 Episode contrast 不随它进入 Fast。

2. Specific Card 到 Agent 的表达：
   - 当前 Domain 匹配的 LOCAL_ACTIVE Card 可进入候选；
   - Source 或未匹配 Target-local Card 不进入 Fast Prompt，不提供跨域候选或证据模板；
   - 其 Positive / Negative / Conflict Experience 只供 Runtime / Slow 形成 General / Shared
     Skill 或在目标域重新形成 Target-local Skill；
   - 已有四个新 roster 上 Specific Card 匹配均为零（0/11、0/27、0/12、0/19），因此
     fresh A5 不得把跨域 Specific 命中作为经验入口。

3. 当前第一项 TS-native tool：
   - G1 只复用现有受限 Workspace tools；
   - 不预建 `inspect_level_excursion_scope`；
   - intrinsic replay 已证明局部动作可由 Operator 自身形成，当前不授权新定位工具；
   - 只有后续自然轨迹的新 first fault 证明 Agent 缺少会改变选择的公开事实时，才重新授权
     一个最小 operator-specific tool。

4. Program Binding：
   - Agent 只选择 Program family；
   - Runtime 针对每个 action unit 分派执行并强制唯一参数所有权；
   - RUNTIME_BOUND 与 OPERATOR_INTRINSIC 互斥；
   - `repair_level_shift` 的 legacy external binding 停用，intrinsic 版本只能作为需 Target
     Support 的探索候选；
   - legacy 与 intrinsic 的 Experience / Card / guidance 权限隔离，不按 operator 名合并；
   - 不拆新 localization artifact，不继续为该 family 调阈值。

5. Agent 调用预算：
   - 复用现有 tool-round 上限；
   - 或在首轮实验预注册更小上限；
   - Tool、LLM、Support 三种成本分开。

6. 新 fresh 数据：
   - G3-D1 Skill-only development 与候选来源的只读筛查可以并行，fresh sourcing 不阻塞
     正确知识流的 development 验证；
   - 只有正确 Pipeline 在 G3-D1 可读后，才冻结 fresh loader、实例与最终 roster；
   - Weather、T233、e31 不再承担 fresh 身份。

---

## 18. 总体验收

### 18.1 工程能力成立

满足以下条件即可认为 Agentic Skill Harness v1 核心闭环实现：

- active General 与当前 Domain 合法匹配的 Specific / Target-local Skill 进入 Agent Context；
- Raw/aggregated Episode 不进入 Fast；Runtime / Slow 能用完整正负冲突 Episode 形成、修改或
  拒绝 Skill，并保留 provenance；
- Fast / Runtime 能从同域 Action--Response 形成或更新 Target-local Skill；
- Agent 自主调用受限 Workspace 工具；
- 工具结果参与 Workflow 选择；
- Runtime 按正确作用单元和唯一参数所有权执行 Typed Workflow；
- Support / delayed 形成 Experience；
- 正向路径能形成或复用 Specific Card；
- 失败路径能触发 Runtime attribution 和 Slow single-surface update；
- replay 证明更新改变后续 Fast Path；
- A3 / A5 能在同一路径、同反馈预算下运行；
- A5 的 Source 增量通过 Source-derived Skill 到达，A3/A5 Fast payload 都不含 Raw Episode；
- instrument-unreadable paired Task 被排除，报告读数不会把仪器失败或 fresh regression
  隐藏成正常行为。

### 18.2 方法 Claim 成立

工程闭环成立不等于 A5 方法收益成立。主 Claim 仍需 fresh G3：

> 只增加 Source-derived Skill 的 A5，是否在相同 Target feedback budget 下，比没有该
> Source-derived Skill 的 A3 更快且更安全地形成有效 Target-local Skill。

若 fresh 结果为可信负向，则关闭对应 Capability family，回到最早 first fault；不关闭
Agentic Harness 总目标。

---

## 19. 一句话架构

> General 教 Agent 如何调查、如何约束决策并提供带证据范围的软先验；Specific 提供
> 当前 Domain 已验证的 Workflow 模板；Experience 保留原始正负冲突事实，供 Runtime
> 归因、Fast/Runtime 本地 Skill 生命周期与 Slow Skill 更新使用，但不直接进入 Fast；Fast
> Agent 只结合 Skill、动态 Workspace Context 与已发生 Target Support 生成当前 Typed Workflow；
> Runtime 负责强制参数所有权和机械正确性，intrinsic Operator 可在 action unit 内自行定位；
> Slow Path 根据 first fault 只修改一个 Harness Surface，再回到 Fast Path 验证。
