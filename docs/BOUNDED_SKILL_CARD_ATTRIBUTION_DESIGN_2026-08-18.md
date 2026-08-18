# 有限归因与 General / Specific Harness Update 设计 rev2.0

~~~text
status: REVIEWED_FORWARD_DESIGN
scope: ATTRIBUTION_AND_ROUTING
implementation:
  C1_CONTEXT_OBSERVATION_PATH: COMPLETE
  GENERAL_PROPOSAL_GUIDANCE_PATH: PENDING
~~~

## 0. 本次修订解决什么

本设计只回答一个问题：

> 当 Action–Response Experience 暴露失败、冲突或可避免的试错时，Runtime 应把
> first fault 归到哪个有限原因、允许修改哪一层知识、再由 Slow Agent 编译什么修改？

rev1.1 将原因直接绑定为 CARD_ABSENT / PROGRAM_MISMATCH /
SCOPE_MISMATCH / RISK_MISMATCH。真实 E1 证据证明这个冻结层级过窄：
repair_level_shift 的有害提案并非来自任何匹配 Skill Card，而来自统一 Operator
Inventory、固定 proposal prompt 与 LLM 自由推理。此时没有一张 Card 可修，却存在真实、
可避免的 Support probe 成本。

因此 rev2.0 冻结三件彼此独立的事：

~~~text
Cause        = 失败发生在哪一种机制
Repair scope = 修一条局部知识，还是修可复用的通用行为
Surface      = 本次实际获授权修改的唯一位置
~~~

旧四因不再作为对外归因词汇；其字段名仅保留为 Specific Skill 的候选 Surface。现有
25 个内部 fault code 继续作为执行标签，不另建第三套 taxonomy。

本设计不新增归因 Schema、Evidence Ledger、Hash、Gate、Memory 平台或通用状态机。

---

## 1. 两层可进化知识

### 1.1 Specific：Context-conditioned executable Skill

Specific Skill 是现有 skill-entry/1：

~~~text
observable_applicability
+ canonical executable body
+ risk_guards
+ allowed_tools
~~~

它回答“在这个可观察 Context 下具体执行什么”，只能通过合法 Program、Support 和
delayed feedback 获得执行权。可用动作仍是：

- ADD 一张新卡；
- 修改 Program/body；
- 修改或拆分 applicability；
- 收紧/调整 risk/abstention。

### 1.2 General：跨 Episode 复用的 Harness guidance

General 不是一张“匹配所有 Context 的大 Card”，也不是把原始负轨迹塞进 Fast Prompt。
它是可复用的 Harness 行为知识，例如：

- 如何从公开 Observation 构造决策 Context；
- 在什么可观察前提下应优先、降级或暂缓某类 Workflow 提案；
- 候选选择与 abstention 的通用指导。

当前最小 General Surface 复用现有 candidate_policy.proposal_guidance。本阶段不新增
General Skill Schema；只有真实结果证明现有 Surface 无法承载时才重新裁决。

### 1.3 Experience 仍是证据层

Positive、Negative、Conflict 与 Abstain Episode 均立即保存，但不自动获得提案权或
执行权。Runtime 可用完整 Episode（包括 context_summary）做归因；Fast Path 不得因此
接收整段原始轨迹。

---

## 2. 封闭原因集合

对外只保留三个原因和一个无动作终局。

| Cause | 确定性含义 | 典型修复方向 |
| --- | --- | --- |
| CONTEXT_GAP | 当前公开 Context 无法合法表达已观察到的效用边界，或合法特征没有进入实际决策表示 | Observation / projection / Specific scope |
| WORKFLOW_GAP | Context 已足够，当前可生成或已绑定的 Program 仍不能解决缺口，而有合法替代 Workflow headroom | Specific ADD/body 或有限 Program Supply |
| DECISION_GAP | Context、可执行候选和相关正负证据已经足够，但提案、选择或 abstention 仍重复作出可避免的错误 | Specific risk/priority 或 General proposal/selection guidance |
| NO_ACTIONABLE_EVIDENCE | Utility 不可读、证据不重复、没有 headroom、多个机制不可分，或没有合法反事实 | 不编辑，保留 Episode / abstain |

CARD_ABSENT 是系统状态，不再是原因；缺卡可能由 Context、Workflow 或 Decision 中任一
机制导致。GENERAL_PATTERN_FAILURE 是 Repair scope 的升级条件，也不是第四个原因。

### 2.1 CONTEXT_GAP 的两个确定性出口

Runtime 必须区分：

1. **信息存在但未被使用**：已有部署可见特征能稳定区分正负区域，但当前 projection、
   applicability 或提案条件没有使用它。允许进入一次 Observation/Scope 修复。
2. **信息不存在**：当前公开特征无法区分真实的相反结果。输出
   OBSERVATION_REQUIRED，不得让 LLM 事后发明阈值或字段。

只有当一个必要新 Observation 与 Scope 不可分时，才允许按 AGENTS.md 将二者作为同一
次耦合修复；不得同时修改 Program、Memory、Risk 或 Judge。

### 2.2 WORKFLOW_GAP 与 DECISION_GAP 的边界

两者不靠自然语言优先级仲裁，而看候选在失败发生前是否已经可用：

- 合法替代 Workflow 未进入当前可生成/可编译集合，或当前 Program 机制本身无 headroom
  → WORKFLOW_GAP；
- 正确 Workflow、正确 abstention 或负向边界已经能被当前提案层使用，仍选择/重复探测
  有害候选 → DECISION_GAP；
- 证据无法判断候选是否“可用但没选” → NO_ACTIONABLE_EVIDENCE。

当前阶段只实现已经真实观察到的 DECISION_GAP 路径，不为另外两因预建通用机制。

---

## 3. Runtime 归因，Slow 只编译修改

LLM 不得自报 Cause。Runtime 使用已经打开且合法的 Episode 计算确定性事实：

~~~text
instrument_valid
utility_readable
distinct_task_episode_count
outcome_identity_not_reused
program_identity
public_context_condition
positive / negative / conflict relation
candidate_available_before_failure
proposal / compile / probe / decision stage
verified_alternative_headroom
~~~

有限 first-fault 顺序是：

~~~text
0. 仪器或 Utility 不可读
   → INSTRUMENT_INVALID / NO_ACTIONABLE_EVIDENCE

1. 同一 Program 的正负结果是否需要 Context 条件才能解释？
   是，但当前合法表示不能区分
   → CONTEXT_GAP

2. Context 已解决后，合法有效 Workflow 是否不在当前可生成/可编译集合？
   是 → WORKFLOW_GAP

3. Context、候选或正确 abstention 已可用，仍重复作出可避免提案/探测？
   是 → DECISION_GAP

4. 仍有两个机制同时成立且不能用现有证据消歧
   → NO_ACTIONABLE_EVIDENCE
~~~

Runtime 随后只暴露一个 Surface catalog。Slow 可以读取最小的正、负、冲突对照，输出
PATCH / ADD / ABSTAIN，但不能改变 Cause、扩大 Repair scope 或批准自己的修改。

---

## 4. SPECIFIC / GENERAL 升级规则

一次有效差结果即可触发 Slow **诊断**；不必为了调用 Slow 等到 General 门槛。但写入权限
分层：

### 4.1 SPECIFIC

仅修改一张 Target-local Skill 或增加一张 Context-conditioned Skill。要求：

- 当前 Domain 内存在合法 Action–Response；
- Context 与 Program 绑定可表达；
- 只开放一个 Card Surface；
- Support/replay 与 delayed 决定保留、限制或撤销。

### 4.2 GENERAL

只有 Runtime 同时确认以下条件，才允许从 Specific 升为 General：

1. 至少 2 个不同 Task Episode；
2. 同一 Program mechanism；
3. 同一 Cause / first fault；
4. 共享同一个已经可表达的决定性公开 Context 条件；
5. 同一 Outcome 不重复计数；
6. 若修改会抑制/降级某类 Workflow，必须有至少一个相反 Context 的正向对照，证明不是
   全局禁用。

计数必须发生在 Context 已解决之后。粗分箱内正负混杂不得分别凑数后宣布为两个 General
规律。

General 修改仍只开放一个 Surface。当前不允许一次同时修改 Observation、
proposal guidance 和 mechanical deny-list。

### 4.2.1 逐条款证据原则（Planner 裁定 2026-08-18）

上述计数一律以 **distinct Task Episode** 为单位。A3/A5 会在同一 Task、同一冻结
Outcome cell 上各探一次，因此 attempt 数（14、8 这类）是重复计数，只能作诊断，
不得作证据量。

进一步，门槛不只约束 GENERAL 写入权限，也逐条约束 guidance 文本里的每一条款：

> 单条 conflict 可以阻止全局禁用，但不能授权新的主动推荐或例外组合；每个主动
> 子句必须独立满足 General 重复证据门槛。

Runtime 交给 Slow 的证据必须是**完整、去重的计数普查**，形状为：

~~~text
canonical program
  x public context condition
  x POSITIVE / NEGATIVE / IMMATERIAL
  -> distinct_task_count
   + attempt_count（仅诊断）
~~~

不得按 relation 或按程序做单边过滤。为此不建设语义解析器，也不新增 Gate：完整
census、Slow 约束与人工审查 rubric 已足够。

**触发本条的实证**：G1 首次实现把 conflict 槽写成"只保留组合程序正例"，Slow 因此
只看到 1 条 `[outlier_mad, repair_level_shift]` 在 false Context 下为正，而看不到
同程序同条件的 4 条负例，于是写出"false 时须配对"的主动子句。该子句在 fresh 侧
把有害算子注入了本来干净的程序。first fault 记为 IMPLEMENTATION_MISMATCH /
RUNTIME_CONTRAST_SAMPLING_BIAS，不是 guidance 机制负结果。

### 4.3 NONE

证据不足、headroom 不存在、Utility 不可读、当前 fresh roster 没有可证伪对照，均进入
NONE。不因“已经调用 Slow”而强行产出修改。

---

## 5. Cause × Repair scope × Surface

| Cause | SPECIFIC 候选 Surface | GENERAL 候选 Surface | 当前状态 |
| --- | --- | --- | --- |
| CONTEXT_GAP | Skill applicability；必要时 Observation+Scope 一次耦合修复 | public Observation derivation/projection | C1 已完成一个最小实例 |
| WORKFLOW_GAP | Skill ADD/body | bounded Program Supply / workflow guidance | 未触发，不实现 |
| DECISION_GAP | risk/abstention/局部优先级 | candidate_policy.proposal_guidance 或后续 selection guidance | 当前唯一下一切片 |

表格不是自动路由。每次只能由真实 first fault 选中其中一个格子。

Specific applicability 仍由 Runtime 拥有：Slow 只能提议使用哪个已观测特征或已有语义，
Runtime 校验存在性、可分性、信息墙与 AST 可达性后生成 Scope。不得拆掉防止 LLM 编造
不可达条件的现有保护。

---

## 6. 当前实证如何落入本设计

### 6.1 C1：CONTEXT_GAP

development Episode 中，repair_level_shift 在不同 Task 上出现稳定效用翻转。C1 确认：

~~~text
post_shift_support_sufficient =
    (1 - estimated_region_end_fraction) * 240 >= 24
~~~

其中 240 = 192 Context + 48 Horizon，24 为冻结周期；它是机制/任务几何导出的布尔
Observation，不是 Outcome 阈值扫描。它在 development 中区分：

- 正向：task07 / task11 / task12；
- 负向或无 headroom：task10 / task13 / task14 / task15 / task16。

基于该 Observation，E0b 形成一张窄的 LOCAL_ACTIVE Specific Skill。fresh task17..27
全部为 false，所以新卡覆盖 0/11 是正确 Scope 行为，不是待填补的覆盖缺口。

该结果证明一个最小 Observation→Specific Skill 闭环；它不是独立 fresh 正向泛化证据。

### 6.2 当前 first fault：DECISION_GAP / GENERAL

代码与 Episode 已确认：

- repair_level_shift 在无匹配 Skill、无 Source prior 的 A3 中仍会被提出；
- 提案源是统一 Operator Inventory、固定 proposal prompt 与 LLM 推理；
- compiled proposal 在 _agent_decision 前必先消耗 Support probe；
- 目标 Experience 的 Fast 摘要丢失 Context，现有 General proposal guidance 又未接入 E1；
- 负向 Context 上存在重复、可避免的有害 probe，而正向 Context 证明该算子不能全局禁用。

因此当前 Cause 是 DECISION_GAP，Repair scope 是 GENERAL，唯一候选 Surface 是：

~~~text
candidate_policy.proposal_guidance
~~~

它不是“制造负向 Skill Card”，也不是 forbidden_operators 全局封禁。

---

## 7. 下一条最小纵向切片

唯一允许实现：

~~~text
Runtime 用完整 Episode + C1 Observation 确认 DECISION_GAP / GENERAL
→ Slow 从正负对比中 PATCH proposal_guidance
→ E1 proposal payload 真正消费该 guidance
→ exposed replay 检查行为
→ fresh false-context paired development 检查 probe / harm
~~~

这是一项 Harness 行为，虽然需要接通三个位置：

1. **归因侧**：E1 调用确定性 cause route；
2. **路由侧**：无匹配 Skill 但重复出现 Context-resolved decision fault 时，能到达
   PROPOSAL_CONTROL_GAP，不再落入不可编辑的 CANDIDATE_SUPPLY_UNKNOWN；
3. **执行侧**：proposal payload 消费已授权的 candidate_policy.proposal_guidance。

三处缺一不可，但不得借此实现另外两个 Cause、完整 25-code 重映射或通用归因平台。

### 7.1 Development 验收

**Planner 裁定 2026-08-18 追加两条读数纪律**：

* `task_probe_cost` 是"未拿到 LOCAL_ACTIVE 即记 B+1"的**惩罚记账**，不是探测次数。
  凡主读数写 Support probe count 之处，一律读 `real_support_probe_count`。
* exposed replay 的 true 侧只证明"没有全局删除该算子"。若 patch 后的 guidance 在
  true Context 下**主动要求**提出该算子，该检验按构造不可能失败，因此它不证明
  "没有过度强制某一配置"。exposed replay PASS 降格为 wiring/behavior replay，
  不是效用验证。

exposed replay 同时检查：

- post_shift_support_sufficient=false：不再优先提出已知有害的
  repair_level_shift，减少 Support probe/harm；
- post_shift_support_sufficient=true：不得把该 Workflow 全局禁用，正向提案资格保留；
- 其他 Program、Judge、Scope、Risk、Memory 与反馈预算不变。

若 replay 通过，fresh task17..27 只能验证 **负向侧的试错成本/安全性**，因为 11 个 Task
全部为 false。它不能验证正向侧保持、完整 warm-start 或 Scope 泛化。

### 7.2 失败也可解释

- Slow ABSTAIN / patch 编译失败 → General guidance supply 尚不可用；
- guidance 已进入 payload，但 LLM 仍重复有害提案 → guidance-only Control 机制负结果；
- 只有后一结果重复成立后，才讨论已有 forbidden_operators 钩子的
  Context-conditioned mechanical Risk 接线；不得现在同时实现；
- CAUSE_NOT_CONFIRMED 不自动证明 Cause 错误，也可能是当前 replay/反馈分辨率不足。

---

## 8. 明确暂缓

- 不实现负向/deny-list Skill Card Schema；
- 不把所有 Negative Experience 原样送进 Fast Prompt；
- 不同时接入 ordering card、proposal guidance 和 forbidden list；
- 不全量重写 fault_routes.json；
- 不实现三 Cause 的通用分类器或覆盖率 dashboard；
- 不启用 DEPRECATE、向量检索、Pattern Graph、新 Memory 平台；
- 不打开 sealed confirmation；
- 不用 fresh 11 个单侧 Task 声称完整 A5 warm-start。

---

## 9. 能支持的项目主张

若下一切片通过，可作出的 development 主张是：

> Harness 能把 Context-resolved 的负向 Action–Response Experience 归因为 General
> Decision Gap，由 Slow Agent 编译一项受限 proposal-guidance 修改，并在不全局禁用该
> Workflow 的前提下减少后续相似 Context 的无效或有害 Support 试错。

这比“修改了一张 Card”更接近 Harness 自适应，但仍只覆盖固定 Consumer、一个 Program
mechanism 和 development evidence。它不证明跨数据集 Shared Capability、多模型泛化或
全部 Cause 已实现。

