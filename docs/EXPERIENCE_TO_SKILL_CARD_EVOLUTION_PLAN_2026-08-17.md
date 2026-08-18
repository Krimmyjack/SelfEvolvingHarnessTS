# Experience → Skill Card：Slow Agent 自进化后续执行计划

版本：rev2.2（2026-08-18）  
状态：`ACTIVE_FORWARD_PLAN / C1_OBSERVATION_FOUND / E0B_SPECIFIC_SKILL_CREATED / G1_GENERAL_DECISION_GUIDANCE_NEXT`  
执行对象：本地实现 Agent  

## 0. 文档地位

本文件是 `docs/TASK_EPISODE_HARNESS_EXECUTION_PLAN_2026-08-17.md`
完成 §17.7 后的**唯一向前执行计划**。旧任务书中的实验数据、历史 verdict 和代码事实
继续保留；其中尚未执行的未来动作由本文件取代，尤其包括：

- §17.8 的全局 `GENERATION_NOVELTY_GUARD`：撤回；
- 继续把大量原始 trajectory 直接塞入 Prompt 来研究候选排序：停止作为主线；
- 在当前阶段扩展 `memory.entries`、`DEPRECATE`、新 Gate、新 Schema 或新检索平台：暂缓。

本计划服从仓库根目录 `AGENTS.md`：一次只改变一个主要 Harness 行为，先完成最小
自然纵向能力，再决定是否扩展。

### 0.1 rev2 审校裁定

| 审校建议 | 裁定 | 落点 |
| --- | --- | --- |
| 修复真实 public features 与 task-specific observation cutoff | 采纳，且列为 E0 唯一前置 | C0 |
| 给宽卡强制增加“必须窄/必须含数据特征”规则 | 不采纳 | 宽卡保留为 guarded DRAFT |
| 将 7/8 同号、`gain/SE>=3` 或新 harm 上限升级为 Runtime Gate | 不采纳 | 保留诊断字段，Gate 不变 |
| 明确 origin / series / Task Episode 三层聚合 | 采纳 | §4.3 |
| E1 用 paired Task 指标、12-Task pilot、最多 30 Task | 采纳 | §5.4–§5.6 |
| `first LOCAL_ACTIVE` 继续作为主统计样本 | 不采纳 | 降为 trajectory 描述量 |
| 宽 DRAFT 的真实冲突可触发单 Surface PATCH | 采纳，条件触发 | E2 |
| 立即补 `DEPRECATE`、dashboard、新检索平台或统计基础设施 | 暂缓 | §8 |

这些不是两条并行研究路线。唯一顺序仍是：先修 C0 进水口，再测 E0 ADD 生命周期，再做
E1 A5/A3；只有 E1 的真实失败/冲突满足条件，才进入 E2 PATCH。

### 0.2 rev2.2 当前证据与向前裁定

rev2.1 的 C0、E0、E1-v1/v2/v3、E0b 与 C1 记录均作为历史证据保留，不重写 Outcome。
长期核心里程碑仍是同预算 A5 vs A3；以下改动只修复这条证据链暴露的第一个 Harness
Update 阻塞，不改变项目终点或比较臂。

- C1 已确认 repair_level_shift 的大幅负效用是真实下游效应，并形成最小 Observation
  post_shift_support_sufficient；
- E0b 已用该 Observation 形成一张窄的 LOCAL_ACTIVE Specific Skill；
- fresh task17..27 全部不满足该 Observation，因此 0/11 coverage 是正确 abstention
  边界，不是继续造 Specific Card 的理由；
- Negative Experience Actionability Audit 证明有害提案并非来自匹配 Skill Card，而来自
  全量 Operator Inventory、固定 proposal prompt 与 LLM 自由推理；
- 当前 E1 不调用归因路由，旧 router 在 skill_retrieved=false 时到不了
  PROPOSAL_CONTROL_GAP，proposal 路径也不消费 candidate_policy.proposal_guidance。

因此旧 §3.2 / §6 的“只修改 Skill Card 字段”不再是唯一 E2 路径。有限归因统一采用
[有限归因与 General / Specific Harness Update 设计](./BOUNDED_SKILL_CARD_ATTRIBUTION_DESIGN_2026-08-18.md)
rev2.0：Cause、Repair scope 与 Surface 分离。当前只开放一个用于恢复 A5 证据链的子切片：

~~~text
DECISION_GAP
→ GENERAL
→ candidate_policy.proposal_guidance
→ proposal path 消费
→ exposed replay
→ fresh false-context development
~~~

---

## 1. 锁定目标

```yaml
locked_goal: >
  构建能够从时序 Action–Response Experience 中归纳、验证、修改并复用
  Context-conditioned Skill Card 的 Data Readiness Harness。
current_milestone: >
  在相同 Target downstream-feedback 预算下，A5 能否利用 Source 的成功、失败和
  冲突 Experience，比 A3 从空 Source 开始更快、更安全地形成有效 Target-local Skill。
scientific_question: >
  Slow Agent 能否把少量相似且对照化的 Experience 转化为结构化可执行 Skill Card，
  并在自然 Target Support 与独立 delayed feedback 下存活和改变后续行为？
smallest_measurable_output: >
  一张由 Slow Agent 提议、Runtime 绑定 Program、Target Support 形成 LOCAL_DRAFT、
  delayed 更新为 LOCAL_ACTIVE 或 RESTRICTED，并能在下一 Context 被真实检索的 Skill Card。
current_bottleneck: >
  C1 已解决当前 repair_level_shift 的 Context 分辨率，但 Negative Experience 尚不可
  行动：E1 不调用 cause routing；现有 router 在没有匹配 Skill 时落到不可编辑 backlog；
  proposal payload 不消费已经声明的 candidate_policy.proposal_guidance。
one_allowed_change: >
  只接通 DECISION_GAP / GENERAL / candidate_policy.proposal_guidance 这一条纵向行为。
  允许为同一行为连接归因、路由和 proposal 消费三处；不同时实现其他 Cause、
  ordering card、negative Skill Schema 或 mechanical deny-list。
deferred_axes:
  - memory.entries 独立 Harness Memory 面
  - DEPRECATE manifest operation
  - 全局 novelty guard
  - learned embedding / vector database / Pattern Graph
  - 新 Gate、General Skill Schema、SHA、Ledger、Receipt
  - ordering card 与 forbidden_operators 接线
  - 三 Cause 通用分类器与 25-code 全量重映射
  - 多 Task、多 Consumer、Shared Capability 零探测执行
definition_of_done: >
  当前子切片完成时，Runtime 能确定性确认 General Decision Gap，Slow 只获得
  proposal_guidance Surface，该 guidance 真实进入 proposal path；exposed replay
  同时检查 false Context 的 probe/harm 下降与 true Context 不被全局禁用。该结果只
  修复后续 A5/A3 的经验行动能力，不单独替代长期 A5/A3 里程碑。
concept_diff: none
```

---

## 2. 统一后的知识模型

前期只让三个逻辑对象承担主线责任，但不建设三套存储系统：Experience 是证据，
Specific Skill 是可执行局部能力，General Guidance 是跨 Episode 复用的 Harness 行为。

### 2.1 ExperienceEpisode：证据，不是执行知识

继续使用现有 `ExperienceEpisode` 保存每次合法 Action–Response：

```text
可观察 Context
+ 实际执行的 Workflow / Program geometry
+ Support gain / SE / gain_over_se
+ delayed gain / SE / gain_over_se（若已到达）
+ POSITIVE / NEGATIVE / CONFLICT / ABSTAIN
+ local status
```

规则：

- 每次真实 probe 后立即记录；成功、失败、冲突和 abstain 都保留；
- Episode 不自动获得候选权或执行权；
- 不把全部历史轨迹原样塞给 Agent；
- 复用现有 `SignedEpisodeRetriever` 和 candidate-conditioned 检索，不新增检索器；
- 每次 Slow 输入最多包含一个最相似 Positive、一个 Negative、一个 Conflict，外加当前
  Target 失败/反馈；缺哪一类就如实为空，不补造。

### 2.2 SkillEntry：蒸馏后的可执行知识

Slow Agent 的主要持久产物统一为现有 `skill-entry/1`：

```text
observable_applicability   # 在什么公开 Context 下适用
body                       # Runtime-owned canonical Program
risk_guards                # 何时应停止、限制或要求 Target probe
allowed_tools              # 从已编译 Program 派生
evidence references        # 写在现有可容纳位置或报告中，不新增 schema
```

权限边界：

- Slow Agent 提出 Skill 意图、Typed Workflow 与 Risk/Scope 修订理由；ADD 时的精确
  applicability 仍由 Runtime 从当前 Card 的公开 signature 生成；
- `compile_workflow_proposal` 验证算子、参数和公开绑定；
- Runtime 把编译后的 canonical Program 写入 Skill body；
- Slow 自由文本不得直接成为可执行 Program；
- Source Skill/Experience 只能帮助提案，不能在 Target 直接激活；
- Target Support 决定 `LOCAL_DRAFT`，独立 delayed 决定
  `LOCAL_ACTIVE / RESTRICTED / revoke`。

Applicability 的所有权保持现状：Runtime 只从 Failure/Task Card 的
`observable_signature` 机器生成 `observable_applicability`；Slow 不得自由编造特征。
`_applicability_is_wide`、`requires_target_support=true` 与批准前可达性检查原样保留。

宽 Scope 不等于错误，窄 Scope 也不自动等于正确：

- 宽卡表示“尚未获得边界证据的假设”，只能作为需要当前 Target Support 的 DRAFT 候选；
- 窄卡表示一个可证伪的 Context 主张，仍须经过 Support 与 delayed；
- 本计划不新增“每张卡必须含某个数据特征”或“必须足够窄”的硬规则。

### 2.3 General Guidance：可复用的 Harness 行为

General 不是一张匹配所有 Context 的 Skill Card。当前只复用已声明的
`candidate_policy.proposal_guidance`，表达“在什么公开前提下优先、降级或暂缓某类
Workflow 提案”。它不直接执行 Operator，也不绕过 Target Support。

General 写入必须满足：至少两个不同 Task Episode、同一机制、同一 first fault、同一
已解决的公开 Context 条件、Outcome 不重复计数；抑制性规则还必须有相反 Context 的
正向对照。Slow 负责从最小正负对比中编译 guidance，Runtime 负责归因、授权和 replay。
本阶段不新增 General Skill Schema。

### 2.4 当前暂缓的 `memory.entries`

`memory-entry/1` 保留兼容性，但本阶段不新增、不读取、不删除。它与 Skill Card 高度重叠，
当前没有独立承重用途。若未来观察到“有价值的非执行指导无法由 Episode + Skill 表达”，
再以该真实阻塞决定是否启用。

---

## 3. Slow Agent 的三种核心进化动作

### 3.1 归纳：Experience → ADD Skill

触发条件：当前 Target 没有可直接复用的合法 Target-local Skill，且存在至少一条真实
Episode。宽 Scope、带 `requires_target_support` 的卡可以继续进入候选池接受当前
Target Support，但不得因此自动取得执行权。

Slow Agent 可以：

1. 从相似成功轨迹归纳共同 Context、Workflow 和 Scope；
2. 同时读取最相似失败/冲突作为边界；
3. 输出一个结构化 `ADD skill_library.entries/{skill_id}`，或 `ABSTAIN`；
4. 一次调用最多一个 Skill 提案，不重试、不为通过而改 Prompt。

`REUSE / MODIFY / NEW` 仅作为提案来源诊断：三者都合法。不得为了证明“生成”而禁止
正确复用；非 novel 提案也不得伪装成编译失败。

### 3.2 修订：Cause、Repair scope 与 Surface 分离

一次有效差结果即可触发 Slow 诊断，但不能凭单点证据获得 General 写入权。Runtime 先按
归因 rev2.0 确定一个 Cause：

```text
CONTEXT_GAP
WORKFLOW_GAP
DECISION_GAP
NO_ACTIONABLE_EVIDENCE
```

再确定 Repair scope：

- SPECIFIC：ADD 或修改一张 Context-conditioned Skill；
- GENERAL：修改跨 Episode 复用的 Observation / proposal / selection guidance；
- NONE：保留 Episode，不编辑。

最后只开放一个 Surface。Card 的 body/applicability/risk 是 Specific Surface，不再被当成
原因本身；CARD_ABSENT 也只是状态。详细确定性谓词、General 升级门槛与终止条件以
归因 rev2.0 为准。

### 3.2.1 当前唯一 General 修改

当前真实 first fault 已满足：

```text
DECISION_GAP / GENERAL / candidate_policy.proposal_guidance
```

Slow 只可依据 C1 后的 Context-resolved 正负对比编译 guidance；Runtime 必须保证正向
Context 中 repair_level_shift 仍可被提出，负向 Context 中不再被无条件优先探测。
不同时接 ordering card、forbidden_operators、negative Card 或 selection guidance。

现有 `RESTRICTED` 与 `revoke_deployed_skill` 继续承担 Specific Skill 撤销语义；
本阶段不为 `DEPRECATE` 修改 edit manifest。

### 3.3 阶段 C0：E0 前置的 Context 进水口修复

本阶段只修一件事：把现有公开 Observation 从真实、部署可见的 Task 前缀送入当前向前
Runner。它是 Instrument/Observation 接线，不是新的 Scope 学习方法。

已核到的缺口位于 Task Episode Runner：`a5a3.py`、`natural_flow.py`、
`normal_flow.py`、`workflow_gen.py` 等路径把 `fast_features` 写成相同的
`task_kind=forecast + local_robust_z_peak=high`，并复用固定 cutoff。C0 只为当前向前路径
抽出一个共享 helper，不追溯清理全部历史 phase。

冻结规则：

1. Observation 只能从当前 Task 的公开前缀计算；cutoff 固定为该 Episode 第一条
   Support origin，严禁读取当前或未来 Support/Query outcome；
2. 复用现有 `extract_public_features`、冻结 numeric bins 与 Task/Card 构造逻辑；不新增
   feature、Schema、embedding 或自由浮点阈值；
3. Task-level `fast_features` 与 Card `observable_signature` 必须来自同一个确定性、
   outcome-blind 投影。优先复用现有 workflow-generation 的 representative-series
   规则；若复用时需要收窄，只允许在现有公开特征中按冻结代码顺序选择一个会随计划中
   Task Context 变化的 bin 特征，不按 gain 选择；
4. 不同 Task 必须使用各自的 cutoff，禁止继续把 `1104` 或字面量 `"high"` 作为所有
   Task 的共同特征输入；
5. Runtime 的 `_applicability_from_card`、`_applicability_is_wide`、
   `requires_target_support` 和 `_applicability_reachable` 原样保留。

发任何 E0 LLM 调用前，只在已曝光 Context 上做零 outcome census：

```text
逐 Task 从公开前缀计算规范化 task-level signature
→ 至少存在两个不同 signature
→ 选定一个与 E0 Target 匹配的 Context 和一个自然不匹配 Context
```

若现有词汇和冻结投影仍不能产生两个 task-level signature，停止为
`TASK_CONTEXT_INLET_NOT_DISTINGUISHABLE`，不得用 outcome 反推特征，也不得直接进入 E0。

C0 完成判据：

- 相同公开 Task 输入重复计算得到相同 `fast_features`；
- 至少两个不同 Task/cutoff 得到不同规范化 signature；
- Card signature 的每个叶子都存在于 Fast feature space，当前 Context 可达；
- 所有输入严格来自当前 cutoff 之前；
- 只新增或修改当前向前 Runner 的一个共享 helper 与一个聚焦测试，不回填全部历史
  Runner，不修改 Runtime 或 Gate。

通过标签：`TASK_CONTEXT_INLET_BINDING_PASS`。

---

## 4. 阶段 E0：自然 Experience → Slow ADD Skill 纵向切片

启动条件：`TASK_CONTEXT_INLET_BINDING_PASS`。C0 未通过时不得运行 E0。

### 4.1 唯一问题

> 现有自然 Episode 能否被 Slow Agent 蒸馏为一张真实落盘、经过 Target feedback、
> 并改变下一轮检索行为的 Target-local Skill Card？

这是机制与最小能力检查，不比较 A5 是否优于 A3，不消耗 virgin cohort。
E0 只跑一条 A5-shaped Source-Evidence 路径，不开两臂；A3/A5 对照从 E1 开始。

现有 run-local store 已经出现过 learned Skill，完整 operator inventory 也已遍历 canonical
operator registry；因此 E0 不再声称“第一次证明 ADD surface 或算子供给存在”。它要新增的
证据是：**自然 Experience → Slow 结构化 ADD → Runtime-owned Program → Task feedback →
下一 Context 的受约束复用**这一整条链。

### 4.2 数据与信息墙

- 使用已经曝光的自然 K1 Task Episodes；不得为 E0 新开 virgin data；
- Target 冻结为已曝光的 `natural_k1_03`；
- Source 冻结为 Target 之外已经发生的 `natural_k1_01/02/04`，经现有检索器缩为
  bounded contrast bundle；
- Target 当前 future 保持 sealed；Support 与 delayed 各固定 `K=3` 个严格前向、互不重叠
  origin；已有曝光数据只能形成 development 机制结论；
- Slow 只看到公开 Context、当前 Target 历史反馈和至多三条对照 Source Evidence。

E0 是 outcome-aware 的 development positive control。发 LLM 前只重放已曝光缓存做一次
零 LLM headroom 断言：`natural_k1_03` 的现有合法、已验证候选中至少一条 Task Episode
Support macro gain `>= M`（现有记录中 `outlier_mad = +0.05719`）。该值只证明正确答案
存在，不进入 Slow 输入，也不指定 Slow 的最终答案；不穷举所有 1–4 步组合。断言不成立则
记录 `E0_NO_KNOWN_PROGRAM_HEADROOM` 并停止，不临时换 Target。E1 的 Target roster 必须
outcome-blind 冻结，不得沿用 E0 的 outcome-aware 选案规则。

### 4.3 冻结的 Task Episode 聚合与批准尺子

本节只写明代码已经使用的 estimand 和 Gate，不新增统计门：

```text
对每条 eval series：先在 block 的 K 个 origins 上平均 gain
Task macro gain：再对全部 eval series 取平均
SE_block：在“每条 series 的 K-origin 均值”之间计算标准误
M = MATERIAL_THRESHOLD = 0.005

Support：verifier 通过、macro gain 有限且 >= M → LOCAL_DRAFT / pending
Delayed：verifier 通过、macro gain 有限且 >= -M → 可批准 LOCAL_ACTIVE
         macro gain < -M → RESTRICTED / revoke，active snapshot 不更新
```

三个统计层次必须分开：

```text
origin                 = Task 内的测量基底，不是反馈标签
per-series K-origin 均值 = 有信息量的聚合轴
Task Episode macro       = Runtime 本轮 DRAFT/ACTIVE 的决策单位
```

`K=3` 的目的主要是避免单一时间点/regime 的偶然性，不把三个 origin 当三个独立样本，
也不宣称 SE 会按 `sqrt(K)` 下降。既有 T0 rig 的审计显示，加 origin 的收益明显递减；该
数值只用于本计划的成本与数据选择，不外推为所有 dataset 的定律。

`M=0.005` 是现有 Runtime 的材料性/权限阈值，不是统计显著性线。`per_origin_gain`、
正/负 series 数、`SE_block`、`gain/SE` 全部报告，用于解释可读性与后续数据设计，但不在
E0 中升级为新 Gate。因此，本轮**不采纳**“7/8 同号”“`gain/SE>=3`”或新单侧 harm
上限作为激活条件；那会同时改变 Risk/Judge 机制。Delayed 保持现有单侧 veto 语义：只要
没有超过 `-M` 的整体伤害即可继续批准，不把 delayed 当作第二次正向显著性证明。

同一 block 内允许个别 origin 或 series 为负。E0 的分支 verdict 只描述机制走到了哪一步；
`SLOW_ADD_SUPPORT_REJECTED` 不自动证明 Program 无效，
`EXPERIENCE_TO_SKILL_ADD_MECHANISM_PASS` 也不自动证明自然 Capability 稳健。

### 4.4 单次执行链

```text
Target Context
→ SignedEpisodeRetriever 返回 bounded contrast bundle
→ Slow Agent：ADD Skill 或 ABSTAIN
→ Runtime 编译 Typed Workflow 并绑定 canonical body
→ 写入仅供 replay 的 candidate snapshot（尚无执行权）
→ Target Support replay
→ 写 ExperienceEpisode
→ 通过后才进入 run-local learned store / LOCAL_DRAFT；拒绝则 active snapshot 不变
→ 独立 delayed
→ LOCAL_ACTIVE / RESTRICTED / revoke
→ 下一已曝光 Context 的真实 retrieval 检查
```

### 4.5 完成判据

Slow 选择 ADD 后，以下机械判据全部满足才记：

```text
EXPERIENCE_TO_SKILL_ADD_MECHANISM_PASS
```

1. Slow 输入确实包含当前 Target Episode；A5-shaped 路径的 Source 对照包非空；
2. Slow 返回合法 ADD；明确 ABSTAIN 是合法终局，但只记 `SLOW_ADD_ABSTAINED`，不继续
   套用 3–7，也不记机制 PASS；
3. candidate Skill body 与 Runtime 编译后的 ordered Program steps 完全一致；
4. Support 拒绝时 active snapshot 不变；通过时 Skill 才进入 run-local learned store；
5. Support/Delayed 结果原位更新对应 Episode 与 Skill 状态；
6. 下一 Context 的复用按 Scope 类型分支验证：
   - 非宽 Scope：在 C0 冻结的 matching Context 命中，在 non-matching Context 不命中；
   - 宽 Scope：必须保留 `requires_target_support=true`，只能作为降级候选进入当前
     Target Support，不得自动占 ACTIVE 优先位或直接执行；
7. `RESTRICTED/revoke` 后下一轮不再执行。

结果分支必须如实保留：

```text
SLOW_ADD_ABSTAINED
SLOW_ADD_COMPILATION_FAILED
SLOW_ADD_SUPPORT_REJECTED
SLOW_ADD_DELAYED_RESTRICTED
EXPERIENCE_TO_SKILL_ADD_MECHANISM_PASS
```

只有最后一项表示完整机制通过；它仍不是跨域 A5 优势。

E0 另报告一个 Scope 子标签：

```text
E0_NARROW_SCOPE_RETRIEVAL_PASS     # matching/non-matching 均实测
E0_WIDE_SCOPE_GUARDED_REUSE_PASS   # 宽假设被正确降级，尚无 Context 边界 Claim
```

两者都可支持 ADD 生命周期的机械 PASS；只有第一项支持“这张卡实现了
Context-conditioned retrieval”的窄 Claim。

E0 默认先跑一次。若唯一失败来自 LLM 的合法 `ABSTAIN`、合法但无效的 Program，或
可编译性失败，而不是数据泄漏/Runtime binding/评估器机械错误，则最多追加两次完全相同
输入、Prompt、模型和参数的诊断调用；不得修改中间结果后继续跑到通过。报告保留首轮 verdict
并列出 `complete_path_count / attempts`：任一次完整闭环只证明机制存在，至少 `2/3` 才记
`SLOW_ADD_REPEATABLE_DEV`。机械错误应先修复，不靠重复掩盖。

### 4.6 复杂度预算

- 复用现有 `task_episode_harness` Runner package，不另建第二套框架；
- 新增至多一个 phase/module，例如 `skill_evolution.py`；
- 主结果继续写现有 `w1_task_episode_harness_report.json`，不再新建第二份主报告；
- 只新增一个聚焦 integration test：ADD → compile → retrieval → delayed/revoke；
- 不修改 Gate、Schema、surface registry、operator registry 或全局 Prompt 体系；
- 不清理历史 SHA、store、报告或旧 Runner。

---

## 5. 阶段 E1：核心 development A5 vs A3

E0 通过后进入 E1，不先扩平台。E0 最多提供一张 Source-domain Skill Card 和 bounded
contrast Episodes，所以本阶段的准确口径是**单卡 + 对照经验 warm start**，不是“成熟
Skill library”。

E1 的硬启动条件是：E0 留下一张在 run-local store 中可检索、未
`RESTRICTED/revoke`、且通过 §4.5 窄卡或宽卡复用检查的 Source Card。若 E0 终止于
`ABSTAINED / COMPILATION_FAILED / SUPPORT_REJECTED / DELAYED_RESTRICTED`，则 E1 延后并
记录 `E1_SOURCE_CARD_UNAVAILABLE`。不得把 A5 临时改成“只有 Source Episodes、没有 Card”
继续比较；那会改变本计划的 estimand，必须另行立项。

E1 使用与 E0/K1 Source base series 不重叠的 development Target cohort；不把当前 K1
已曝光结果包装成新结论，也不在本阶段消耗封存确认集。Source Card 在 Target 只能作为
提案/排序先验，不能直接批准或激活。

### 5.1 启动前置、数据角色与任务规模

打开 E1 paired Target outcome 前必须完成以下前置。除第 5 项只打开隔离的 development
calibration outcome 外，其余检查均为零 outcome：

1. 用现有 exposure/roster 信息确认 Target base series 与 E0/K1 Source 不重叠；
2. 若采用 Time-Series-Library/TSQAgent 范围的数据，先把一个明确 dataset 登记为
   `DEVELOPMENT`，另一个登记为 `SEALED_CONFIRMATION`；前者可做本轮 pilot，后者不得读取
   实例 Context 或 Outcome；
3. 确认 development dataset 能形成至少 12 个预注册 paired Task Episode，并另留一个与
   paired roster 在 base series 和 outcome block 上都不重叠的 development calibration
   slice；每个正式 Episode 的
   Support/Delayed 各固定 `K=3` 个 origin，严格前向、互不重叠，任何 Outcome cell 只承担
   一次反馈角色；不足则停止，不静默缩小样本；
4. 零 outcome census 必须显示计划 Task 中至少有两个不同的规范化 task-level Context
   signature；否则停止 `E1_TARGET_CONTEXTS_INERT`，不把同一 Task 的滑窗重测包装成
   Context 多样性；
5. 在不打开 paired Target roster 的前提下，对 calibration slice 做一次零 LLM headroom
   positive control：只运行预先冻结的合法 canonical single-step inventory，不搜索组合；
   至少一个候选在 Task Episode Support 上达到现有 `macro gain >= M` 才继续。该结果只记入
   `private_audit`，不进入 Source/Target Memory，不指定或排序 E1 候选。否则停止为
   `E1_DEVELOPMENT_SUBSTRATE_NO_KNOWN_HEADROOM`，不得把“两臂都无可用 Workflow”解释为
   Source Card 无效；
6. 冻结每 Episode 最大 Target probe 预算 `B`、两臂 LLM 预算、Task roster 与执行顺序；
   Task 上按 `A3→A5 / A5→A3` 交错；
7. 数据选择优先足够多的 eval series。当前 8-series rig 的已曝光审计显示 origin 平均存在
   明显收益递减，因此 E1 development 优先选择约 32 条或更多 eval series 的现成 dataset；
   这只是当前方差估计导出的优先条件，不是跨 dataset 的新 Gate。若候选 dataset 少于该数，
   必须由 12-Task pilot 的实际 paired SE 决定是否继续。

`K=3` 用于覆盖时间 regime，而不是把三个 origin 当独立样本。新数据的确切名称、series
数量、Task roster 和 `B` 必须在打开 Outcome 前补入本节或同一报告的 preregistration
块，不能在结果出来后替换 Task。

同臂复本不作为 development 首轮前置。LLM 随机性属于方法行为，不追求同 Prompt 同答案；
只有 first fault 明确指向采样方差时，才在不改 Prompt/数据/预算的条件下做有上限诊断。
正式封存集确认前，再冻结独立轨迹复本与同臂噪声地板。

### 5.2 两臂

| Arm | 初始知识 | Slow 输入 | Target 执行权 |
| --- | --- | --- | --- |
| A3 | 空 Source Experience/Skill | 当前 Target Context、Target Episodes | 仅 Target Support + delayed |
| A5 | 一张 Source-domain Skill Card + bounded Source contrast Episodes | 与 A3 相同内容，外加 Source 先验 | 仍仅 Target Support + delayed |

两臂必须使用相同：Target Task Episodes、Consumer、Metric、完整 operator inventory、Support
预算、delayed blocks、LLM 设置和每 Episode 最大 proposal/probe 数。去掉 Source 块后，
两臂规范化输入必须逐字节一致。

Source-domain Skill 在 Target 只能被 Slow `REUSE/MODIFY` 为 Target 候选，不能零探测直接
激活；这不是 Shared Capability 实验。

### 5.3 Prequential Task 流

每个 Target Task Episode：

```text
计算当前公开 Task Context
→ 检索 Target-local Skill 与 A5-only Source prior
→ 宽 DRAFT 可进入候选池，但必须接受当前 Target Support
→ 没有可用候选时，Slow 最多 ADD 一次；允许 REUSE / MODIFY / NEW / ABSTAIN
→ 每次 Target Support probe 后立即写 Episode
→ Draft / continue / request observation / abstain
→ delayed 到达后更新 Skill 与 Episode
→ 下一 Episode 才能读取本轮 Target Experience/Skill
```

复用是正确行为，不强制每个新 Context 都 ADD。计划同时记录 `ADD / guarded reuse /
direct Target-local reuse / abstain` 次数，用来判断系统是在积累新卡、验证宽假设，还是复用
已成立的局部能力。

不得把同一 Outcome 同时作为 Support、delayed 或下一 Episode 的独立证据。Task blocks
必须全程唯一、严格前向、互不重叠。

### 5.4 指标与样本单位

主比较使用**每个 Task 一对 A5−A3 差值**，不再用两臂总和制造视觉差异：

| 指标 | 样本单位 | 方向 / 处理 |
| --- | --- | --- |
| `task_probe_cost` | paired Task | 形成 LOCAL_ACTIVE 时取实际 Support probes；否则截尾为 `B+1`，越低越好 |
| `harmful_probe_count` | paired Task | Support gain < `-M` 的尝试数，越低越好 |
| `cumulative_support_harm` | paired Task | 所有负 gain 绝对值之和，越低越好 |
| `task_local_active` | paired Task | 本 Task 是否形成 delayed 后仍可用的本地 Skill |
| `task_delayed_utility` | paired draft Task | 仅两臂都形成 draft 时进入配对 delayed 比较 |
| `abstention/request_observation` | paired Task | 描述何时停止或请求信息 |

每个配对数值指标必须报告：`paired_mean(A5-A3) ± SE`、样本数和逐 Task 差值；两臂总和只作
辅助显示。

`target_probes_to_first_local_active` 保留为**整条 adaptation trajectory 的描述量**；一条
轨迹只贡献一个值，Task 数增加不会把它变成多个独立样本。它不再单独承担 E1 主 verdict。

`delayed_survival_rate` 的有效样本数是 draft 数，不是 Task 数。报告两臂 draft 数；只有
两臂都形成 draft 的配对 Task `>=8` 时，才允许给出 comparative delayed 结论，否则标记
`DELAYED_COMPARISON_UNREADABLE`。

`gain/SE`、per-series 符号和 `SE_block` 继续作为 Agent 可见可靠性与报告字段，不升级为
新 Runtime Gate。

### 5.5 两段式 development 样本计划

最小有意义效应预先固定为：

```text
Delta_probe = 每个 Target Task 至少少 1 次 Support probe
且 A5 不增加 harmful_probe_count / cumulative_support_harm
```

执行规则：

1. 先跑 `N0=12` 个冻结的 paired Task Episode；
2. pilot 结束后只用 `d_i = task_probe_cost_A5 - task_probe_cost_A3` 的居中差值估计 paired
   SD `s`，按双侧 `alpha=0.05`、power `0.80` 计算
   `N_req = ceil(7.84 * s^2 / Delta_probe^2)`；
3. 同时只按 paired-draft 产出率 `q` 估计 delayed 所需 Task：`q=0` 时为不可读；否则
   `N_draft_req = ceil(8 / q)`；不读取 delayed 正负来决定追加；
4. `N_final = max(12, N_req, N_draft_req)`；`N_final<=30` 时只按预冻结 roster 向后补足
   一次，`N_final>30` 或可用 Task 不足时停止为
   `E1_PRACTICAL_RESOLUTION_INSUFFICIENT`；
5. 不查看中途显著性来决定追加，不换 Task，不运行到出现正例；最大 `N=30`。

现有 4-Task replay 只用于 pilot 方差起点：实测 paired probe SD 约 `1.26`，暗示检出每 Task
少 1 probe 约需 13–14 个 Task；该 SD 来自极小样本，不作为最终定数。历史真实成本约为每个
双臂 Task 72.5 秒、约 6.2 次 LLM 调用，因此 12–30 Task 仍是可承受的 development 范围，
但不能再引用 T0 的 4.4 秒零 LLM 成本。

这套公式是 development 容量规划，不是最终论文功效保证：Prequential Task 共享同一条
适应轨迹，存在序列依赖。E1 报告必须同时给出按时间顺序的逐 Task 差值；正式封存确认要以
独立 adaptation trajectories / dataset 为推断单位，不能把同一轨迹内的 Task 假装成完全
独立重复。

### 5.6 Development 判读

本阶段不要求 LLM 对同一 Prompt 给出确定性输出。随机选择/生成是 Agent 能力的一部分；
公平性由同预算、同 Context、配对 Task、交错 arm 顺序和预注册停止规则保证。
下列 95% CI 只用于 development 可读性和是否值得进入封存确认，不声称已完成正式独立重复。

```text
A5_SKILL_CARD_WARM_START_DEV_SIGNAL：
  Source 确实进入 A5 输入并改变至少一个 Task 的行为；
  paired mean(task_probe_cost_A5 - task_probe_cost_A3) <= -1；
  该差值的 95% CI 上界 < 0；
  paired mean harmful_probe_count 与 cumulative_support_harm 均不高于 A3；
  task_local_active_count 不低于 A3；
  若 paired drafts >= 8，A5 delayed utility / survival 不低于 A3。

A5_SKILL_CARD_NEGATIVE_TRANSFER_DEV：
  Source 确实改变行为，且 paired harm 的 95% CI 下界 > 0，
  或在可读 delayed 样本上 A5 明确更差。

A5_A3_SKILL_INPUT_INERT：
  Source Card/Evidence 未进入规范化输入，或所有 Task 的提案与控制行为完全未变。

E1_PRACTICAL_RESOLUTION_INSUFFICIENT：
  N_req > 30、可用 Task 不足，或 paired delayed drafts < 8 而 Claim 必须依赖 delayed。

其余：A5_SKILL_CARD_NO_BENEFIT_DEV
```

若 probe/harm 满足 warm-start 条件但 delayed 不可读，只能记
`A5_SUPPORT_EFFICIENCY_DEV_SIGNAL / DELAYED_UNREADABLE`，不得升级为完整“更快且更安全”。
所有标签均为 development；只有稳定正向信号、Harness 冻结且封存数据仍未打开时，才申请
`SEALED_CONFIRMATION`。

---

## 6. 阶段 G1 / E2：Negative Experience → General Decision Guidance

E2 仍只由真实失败或冲突触发，但触发源不再限定为“某张 Skill Card 失败”。本轮已经满足
一个更早的真实入口：

```text
同一 Program 在 Context-resolved 的多个 Episode 中反复产生真实 harm
+ 正向 Context 证明不能全局禁用
+ 有害候选来自 General proposal path，而非匹配 Specific Skill
→ DECISION_GAP / GENERAL
```

### 6.1 唯一允许的实现

只接通以下一条端到端行为：

```text
完整 Episode + post_shift_support_sufficient
→ Runtime 确定性归因
→ PROPOSAL_CONTROL_GAP
→ Slow PATCH candidate_policy.proposal_guidance
→ A5 proposal payload 消费；A3 保持 base guidance
```

该行为需要同时修三个现有断点：

1. **归因侧**：E1 必须调用确定性 cause route；不能只在编辑时构造 FaultRouter；
2. **路由侧**：当 skill_retrieved=false、但重复的 Context-resolved harmful proposal
   已成立时，必须可到达 PROPOSAL_CONTROL_GAP，而不是
   CANDIDATE_SUPPLY_UNKNOWN / EVIDENCE_BACKLOG；
3. **执行侧**：E1 proposal payload 与 system instruction 必须真实消费获授权的
   candidate_policy.proposal_guidance。

三处共同实现一个 Harness 行为，不得拆成三个平行研究假设，也不得顺手扩充其他 route。

### 6.2 Slow 与 Runtime 权限

- Runtime 用完整 ExperienceEpisode 的 context_summary 计算 Cause；不把 raw trajectory
  或整个 Episode bank 塞入 Fast Prompt；
- Slow 只接收最小正向、负向和冲突对比以及一个 Surface catalog；
- Slow 可以 PATCH guidance 或 ABSTAIN，不能新增 Cause、改 Specific Card、改
  Observation、改 Risk、改 Judge 或启用 forbidden list；
- Runtime 验证实际 diff 只发生在 proposal_guidance；
- 一次失败即可触发 Slow 诊断，但 GENERAL 写入必须满足归因 rev2.0 的重复 Episode 与
  正向对照门槛。

### 6.3 两级验证

先用已曝光 development Episode 做 replay：

```text
false Context: repair_level_shift 不再被无条件优先提出，probe/harm 下降
true Context : repair_level_shift 仍保留提案资格，不得全局 suppression
```

若任一侧失败，停止并记录 guidance-only 机制结果，不接 mechanical deny-list。

只有两侧 replay 通过，才可在 task17..27 做同预算 paired development：

- A3：base proposal guidance；
- A5：Source/历史 Experience 归纳出的 patched guidance；
- 主读数：Support probe count、harmful probe count、cumulative Support harm；
- 辅助读数：abstention、其他 Workflow proposal、最终 Utility；
- 不因减少 repair_level_shift 就默认判 PASS，必须记录是否错过其他有效 Workflow。

fresh 11 个 Task 全部 post_shift_support_sufficient=false，所以该阶段只能提供
**负向侧 safety/cost development evidence**。它不能验证正向侧泛化、完整 warm-start 或
Shared Capability；正向保持只能来自 exposed replay，后续双侧自然 roster 另行冻结。

### 6.4 可解释终局

- Runtime 不能唯一确认当前 Cause → G1_CAUSE_NOT_ACTIONABLE；
- Slow ABSTAIN 或 patch 不合法 → G1_GUIDANCE_SUPPLY_FAILED；
- guidance 被消费但行为不变 → G1_GUIDANCE_INERT；
- false 侧改善、true 侧退化 → G1_GLOBAL_SUPPRESSION_REJECTED；
- 两侧 exposed replay 通过 → G1_GENERAL_GUIDANCE_REPLAY_PASS；
- 随后的 fresh 单侧 paired 结果单独报告，不把 replay PASS 自动升级为 A5 warm-start。

---

## 7. Claim—Evidence 对照

| Claim | 必需证据 | 阶段 | 当前状态 |
| --- | --- | --- | --- |
| Task Context 的公开 Observation 真正进入 applicability | public-prefix 实算、task-specific cutoff、两个 signature、可达性 | C0 | PASS |
| Slow 能把 Experience 变成结构化 Skill Card | 自然 ADD、Runtime body binding、落盘与编译；不是泛化 ADD 首证 | E0 | PASS |
| Skill Card 能改变下一轮行为 | 窄卡 matching/non-matching；或宽卡 guarded reuse；撤销后消失 | E0 | PASS（development mechanism） |
| 单张 Source Skill/Evidence 能缩短 Target 冷启动 | A5 vs A3 同预算、paired probes/harm/delayed | E1 | 未成立；v2 有 Scope bypass，v3 matching capacity 不足 |
| 新 Observation 能表达原本不可分的 Program 边界并形成 Specific Skill | 机制导出的 Observation、正负分离、held-in replay、窄卡批准 | C1/E0b | PASS（development） |
| Negative Experience 能修改 General proposal 行为并减少重复试错 | Runtime 归因、Slow 单 Surface patch、正负 replay、fresh 单侧 paired cost/harm | G1 | NEXT |
| 能跨域低探测直接执行 Shared Capability | virgin cross-domain confirmation | 后续 | 暂缓 |

这里不把以下内容当作主线证据：测试数量、Schema 数量、Card 文件数量、排序变化、一次
Support 正 gain、LLM 是否输出同一句话。

---

## 8. 明确停止与暂缓

### 8.1 当前停止

- Ordering Card 作为 C2 主证据；
- 全局 `GENERATION_NOVELTY_GUARD`；
- 把所有 Experience 原样送入 Prompt；
- 为得到 signed bank 人工制造失败；
- 为一个自然负结果不断搜索新的 fault/operator family；
- 继续用 per-origin winner 作为主要反馈标签。
- 把纯负向经验伪装成 executable Skill Card。

### 8.2 当前暂缓

- `memory.entries/{memory_id}` 主线使用；
- `DEPRECATE` operation；
- 同时开放多个 PATCH Surface；
- ordering card、proposal guidance 与 forbidden list 同时接线；
- General Skill 新 Schema 与全量 fault taxonomy 重构；
- learned retrieval、embedding、向量数据库；
- 新 Consumer、异常检测、多任务和 Shared Capability；
- virgin Target 曝光。

`DEPRECATE` 的暂缓不是遗漏：当前只有个位数 run-local cards，现有
`RESTRICTED/revoke` 足以控制执行权。只有真实测到 learned cards 造成检索干扰、过期卡
占位或维护负担时，才把 create/modify 之外的剪枝操作重新列为当前阻塞。

---

## 9. 执行顺序与交付格式

### 9.1 顺序

```text
已完成并保留：
  C0 PASS
  → E0 ADD / narrow retrieval mechanism PASS
  → E1-v1 no-benefit development
  → E1-v2 protocol isolation repair；发现 Source Scope bypass
  → E1-v3 Scope routing repair；matching Target capacity insufficient
  → E0b Observation insufficient
  → C1 post_shift_support_sufficient
  → E0b Specific Skill created；fresh coverage 0/11（正确边界）

当前唯一向前切片：
  G1-A  Runtime DECISION_GAP / GENERAL attribution
  → G1-B Slow PATCH proposal_guidance + proposal path consumption
  → exposed false/true Context replay
      ├─ 任一侧失败：停止，记录 guidance-only 机制终局
      └─ 两侧通过：task17..27 单侧 fresh paired development
  → 交审

后续而非本轮：
  在同时含 true/false Context 的新自然 roster 上恢复完整 A5 vs A3
  → 有稳定双侧 development signal 后才申请 sealed confirmation
```

### 9.2 每阶段回报

```text
Harness 行为发生了什么改变
真实 Task Episode 上观察到了什么
形成/修改了哪张 Skill Card，状态是什么
Source Experience 是否改变 Target 成本或风险
当前第一个方法阻塞
下一项唯一允许的 Harness 修改
```

基础设施、测试和报告路径仅放在附注。

---

## 10. 当前立即执行项

本地实现 Agent 下一步只执行 G1，不并行其他 Cause、E1-v3、sealed 或 E2 Card PATCH：

1. 用已有 development Episode 与 post_shift_support_sufficient 做零 LLM 确定性归因，
   确认重复 Episode、Outcome 不重复、同一 Program/first fault 与正向对照；
2. 让无匹配 Skill 的当前案例能窄路由到 PROPOSAL_CONTROL_GAP；不全量改写
   fault_routes；
3. 只授权 candidate_policy.proposal_guidance，调用 Slow 至多一次生成 PATCH 或 ABSTAIN；
4. 让 E1 proposal path 真实消费 patch；A3 使用 base guidance，A5 使用 patched guidance；
5. 在已曝光 task07/10/11/12/14/15/16 上做正负 replay。false 侧必须少提或不提已知
   有害 repair_level_shift；true 侧必须保留其提案资格；
6. 增加一个聚焦协议测试，覆盖单 Surface 权限、guidance 进入 payload、正负 Context
   不发生全局 suppression；不新建测试矩阵；
7. replay 不通过即停止。通过后才运行 task17..27 的 11-Task paired development，
   主报 probes/harm，并检查是否错过其他有效 Workflow；
8. 结果写入既有主报告的新 G1 节，保留全部历史对象；停止交审。

复杂度预算：

- 不新建 Runner package，优先修改现有 e1.py 与最小 route 实现；
- 最多一个现有/聚焦测试文件、一个既有主报告；
- 不新增 Schema、SHA、Memory 格式、negative card、ordering card、deny-list、Gate 或平台；
- 不打开 noaa_global_hourly sealed confirmation；
- 不把 fresh 单侧结果写成完整 A5 warm-start。
