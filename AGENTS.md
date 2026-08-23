# SelfEvolvingHarnessTS 项目执行正典

本文件适用于本仓库及其所有子目录，是项目目标、实验优先级和证据边界的
长期权威。历史任务书与报告保留为证据记录；与本文件冲突时，以本文件和
用户当前指令为准。

## 1. 项目目标

本项目针对时序数据中“质量”标准随 Task、Consumer/模型和局部时序 Pattern
变化的问题，构建 Agent 驱动的 Data Readiness Harness：

```text
Task / Consumer / Data Context
× executable Workflow
× downstream Action–Response Evidence
→ Target-local Data Readiness Skill
```

Harness 应读取部署时可见的多尺度 Context，自主生成、验证和修改数据准备
Workflow；使用真实下游反馈记录成功、失败、冲突和 abstain；据此更新
Observation、Program、Scope/Risk、Skill、Memory 或决策策略，并在无反馈的
部署数据上安全复用冻结结果。

项目不得默认收缩成传统数值 Router、固定 Operator 选择、纯清洗 AutoML、
纯 Consumer 优化、模型选择或围绕 Gate/SHA/报告建设的工程项目。固定
Forecasting、Consumer 和五程序菜单只是已经验证或正在验收的实验支架，
不是最终系统边界。

最终系统验收不是“积累了多少 Runner”，而是同一个 operational Harness 入口能够
接收不同 `TaskSpec` 与 Consumer adapter，共用 Workspace、Typed Workflow、Memory、
Risk 和 Skill 生命周期，完成 `held-in adaptation → freeze → held-out Fast-only`。
实验 Runner 只负责提供数据、信息墙和计分，不得按 Task/Consumer 名称指定答案。
最终收口还必须保留至少一次自然数据上的 `Static/A3/A5` 同场验收：A5 代表完整
“累积知识 + Target 校准”系统，A3/Static 只用于拆解其贡献；不能用只有 A3 的
组件实验替代完整系统结果。

## 2. 统一主张：积累与适应是同一个自进化循环

不得把项目写成“Target 从零适应是主线，跨域经验只是可选增强”。完整 Harness 的
常态工作方式是：过去 Domain 的合法经验经整合成为可复用知识，新 Domain 的
held-in 反馈再对这份知识做校准、收缩、扩展或否决，最后把冻结状态部署到 held-out。
`Source` 与 `Target` 是同一时间循环中的证据角色，不是两套系统；今天的 Target 在
结果合法打开并进入下一轮后，也可以成为明天的 Source。

```text
K_t：跨 Domain/Task/Consumer 累积的经审计 Skill/Capability
+ 新 Target 的部署可见 Context
→ held-in 多轮适应：复用、probe、写回、修订 Scope/Risk/Workflow、Fast replay
→ 形成并持续更新 Target-local Skill
→ freeze
→ held-out Fast-only 验收
→ 已打开证据只进入下一代 K_(t+1)，不得追溯修改本次结果
```

### 2.1 完整系统臂与消融臂

- `A5` 是完整系统臂：以经审计的累积 Source-derived Skill/Capability 为起点，
  再用当前 Target held-in 反馈适应。
- `A3` 是去掉跨域积累的消融臂：同一 Harness 从公共 h0 开始，仅靠 Target
  held-in 反馈适应。它用于测量“历史积累贡献了多少”，不是项目的最终产品形态。
- `Static` 是去掉 Harness 适应的消融臂：用于测量“自适应本身贡献了多少”。

完整评价应同时报告 `A5 vs Static`（端到端系统收益）、`A5 vs A3`（累积经验的
边际贡献）与 `A3 vs Static`（Target 本地适应的边际贡献）。实验上分臂是为了归因，
不代表架构上把 accumulation、transfer 和 adaptation 拆成互不相干的路线。不能预先
规定 A5 的数值收益一定占多数，但必须让累积知识在架构中承担一等角色，并由读数
检验它实际贡献了多少。

若某一候选 Source Skill 未通过 development 行为验收，只关闭该候选/版本进入
本次考场的资格；它不把 A5 或跨域积累降为永久可选项。此时可以先跑 `A3 vs Static`
定位 Target 适应能力，但该结果只是完整系统的组件验收，不能单独宣称最终 Harness
目标已经完成。

### 2.2 可复用知识的合法通路

原始跨轨迹/跨域 Source Episode 不得直接进入 Fast Prompt；合法知识流是：

```text
Source Episode → deterministic census → Slow consolidation
→ audited frozen Source-derived Skill → Fast read-only use
→ Target held-in evidence 对其确认、修订或否决
```

两臂比较时必须使用相同 Target held-in 反馈预算；A5 只多冻结的累积知识，不得
多看 Target Outcome。Source-derived Skill 通常是要求 inspect、probe、avoid 或
abstain 的 soft prior；除非满足 Shared Capability 的严格证据门，它不得绕过当前
Target Support 自动取得执行权。

### 2.3 Pattern 与 Task/Consumer 是主方法 Context

Pattern 不是附属解释图，而是决定历史知识何时应复用、何时应由 held-in 反馈修订的
核心 Context。基本单位是：

```text
observable Pattern × Program geometry × Task/Consumer
→ Action–Response
```

Harness 应区分 series/channel/interval 级 Pattern、Program 作用几何和
Task/Consumer 质量语义。Dataset/cohort 名称及其代理不得作为跨域相似性或 Skill
Scope 的理由。若 Observation 无法区分反号结果，只能记录 Observation 缺口、依靠
少量 Target 反馈、abstain 或终止该 family；不得以继续扩 Memory 代替可观察 Context。

M0 只能证明同一处理随 Consumer/模型反号的现象存在；M1 必须证明同一 Harness
读取该 Context 后会自主改变 Workflow，并通过 held-in 反馈与 held-out 终态安全
验收。M0 的模型轴必须保持 Task、数据 roster/split、窗口、Program 及其作用字节、
反馈预算、Metric 和最终评价目标不变，只改变 Consumer/模型的归纳偏置；否则只能
记为混合变化。M1 可以向 Agent 暴露部署时合法可见的 Consumer 结构、训练目标和
接口语义，但 Runner 禁止按模型名或 Consumer ID 映射 Workflow，且各设置必须走
同一 Harness 入口、候选菜单和生命周期。

## 3. Held-in / held-out 正典语义

### Held-in：反馈可用的适应区

Held-in 是可在预冻结反馈预算内持续交互的**域内适应环境**，不是只消费一次就
丢弃的单个 Support batch。冻结前，Harness 可以在同一 held-in 数据域上运行
`r1 ... rR` 多轮 self-harness 循环；前一轮形成的 Episode、Target-local Draft、
Risk、Observation 或经验证的 Harness Patch 可以进入后一轮：

```text
读取当前冻结知识与 held-in 历史
→ Fast 提案 / probe
→ Support 与 held-in delayed feedback
→ Episode 写回与 first-fault
→ 必要时 Slow 只修改一个 Harness 面
→ 确定性审计 / replay
→ 下一 held-in 轮，直到预算或停止条件命中
```

允许在后续轮次重新使用 held-in 数据和已经发生的反馈来修正 Harness；每次 Consumer
评估、重训或反馈调用仍须计入总预算。对同一数据切片或同一 Outcome 的重复 replay
不得冒充新的独立证据，报告需区分新反馈、缓存重放和重复观测。多轮次数上限、反馈
总预算、可用 held-in 窗口和停止规则须在打开 fresh Target outcome 前冻结；Agent
在该边界内自主决定 Workflow 和更新顺序。

以下行为只能发生在 held-in：

- 下游 Consumer 的即时 Support 和后续 delayed feedback；
- Positive / Negative / Conflict / ABSTAIN Episode 写入；
- Slow Path first-fault 分析；
- Target-local Draft 的形成、批准、限制和撤销；
- 预冻结总反馈预算内的多轮 probe、Fast replay 和 Harness 修改。

这里的 delayed 是 held-in 内未参与本次选择的后续反馈，用于防止同批自提自批；
它不是最终 held-out。

### Held-out：零反馈的 Fast-only 部署区

进入 held-out 前必须冻结 Static* / A3* / A5*。运行期间只允许读取部署时可见
Context、冻结 Skill 和不读取 Outcome 的确定性合法性检查。禁止：

- `open_delayed` 或任何下游反馈回传；
- Slow Path；
- 新增、修改、批准、限制或撤销 Skill；
- 按 held-out Outcome 写 Experience；
- 看结果后重试、调参、换 Workflow 或重跑改法。

所有臂的 Workflow 与输出冻结后，外部 evaluator 才可一次性打开 Outcome，
只用于最终计分和报告，不得回流本次 Harness。打开即 `outcome_exposure=EXPOSED`，
该数据不再是 fresh/virgin Target。

最终统一协议固定为：

```text
held-in iterative adaptation (r1 ... rR) → freeze → held-out Fast-only deployment
→ offline one-shot evaluation
```

## 4. Harness 知识与执行权

- `Experience Episode`：一次合法 Action–Response 即可记录；成功、失败、冲突
  和 abstain 均收纳，但不自动获得执行权。
- `Target-local Skill`：在当前 Domain held-in Support 上形成，由同域 held-in
  delayed feedback 更新；冻结后可在同域 held-out 使用。
- `Shared Capability`：只有多个 Domain 的相似可观察 Context 中存在重复正向
  与风险证据时才归纳；零/低 probe 跨域执行权需要更强 fresh 证据。

Memory 的收纳与扩大执行权必须分开。当前轨迹内已发生的 Target Episode 可用于
后续 held-in 轮；跨轨迹/跨域 Episode 必须先整合为经审计 Skill。LLM 只能提出
Patch，不能批准自己的 Patch；执行权由 deterministic compiler/replay 和下游反馈
决定。

Fast Path 可以读取：active bootstrap/General Skill、合法适用的 Source-derived
或 Target-local Skill、当前部署可见 Workspace Observation/工具结果，以及当前
held-in 轨迹中已经发生的 Target Support。Fast Path 禁止读取：

- raw 或逐行 Source/Target Episode bank；
- `source_experiences`、`raw_episode_bank` 或等价 Episode 列表 prompt 字段；
- 绕过 Skill 形成、把 Episode 确定性聚合后直接变成候选菜单的旁路；
- 未匹配当前 Domain 的 Source Target-local Card；
- 当前 Query future、delayed Outcome 或最终 held-out Outcome。

历史 `T233 raw Source Episodes → Fast Agent` 属可信拒绝路线
`RAW_SOURCE_EPISODES_TO_FAST_REJECTED`，保留作机制证据，不得通过重排、加权、
检索或聚合相同 raw Episode 修复后重新接回 Fast。

## 5. 当前状态锁（C23 后）

- Forecasting 纵向切片和 pooled Source 加速已有有界正证据；既有设计冻结复用，
  不因 AD 支线结果重写。
- 多 Task 基础设施已接通：TaskSpec/Consumer 键、双任务 Runtime、AD adapter、
  Support/delayed 生命周期与风险撤权均已有机制或 development 证据。
- AD Source family 已使用 4 cohort、40 Episode 封顶；不得下载或扩充第五个
  Source cohort。
- `source_investigation_ad_v3` 的 development 验收为
  `RISK_PRIOR_EFFECT_AMBIGUOUS`：送达成立，但无可归因行为收益；v3 归档。
- 当前知识下只关闭 `source_investigation_ad_v3` 进入本次 #42g 的资格；A5 不进入
  该考场，且不得修改 v3、重抽 AdExchange 或新造 Risk Skill 来凑入场。该裁定不
  关闭完整系统对跨域积累的需要，也不把 A3 提升为最终系统形态。
- 当前下一项组件门是准备未曝光 Target，并执行 `#42g: Static vs A3` 的严格
  held-in→freeze→held-out 协议，先验收 AD Target-local 适应端口。它通过后仍欠
  一个具备合格累积知识的 `Static/A3/A5` 同场完整系统验收。
- Pattern 是统一方法的核心 Context；#42e2 只是一个非阻塞的单特征诊断，不得
  阻塞 #42f/#42g。若继续，只能使用既有 4-cohort 响应表和一个预注册最小
  Observation，不再扩 Source 数据。
- AdExchange、NOAA 2025 及已打开的 Source outcome 只能作 development/replay，
  不得再次称为 fresh。

## 6. 单假设与 first-fault 纪律

每轮只改变 Observation、Program、Scope/Risk、Memory 或 Harness Update 中的
一个主要行为机制。先定位最早阻塞：

```text
无可读正效应              → Consumer / evaluator / training protocol
有效应但无合法 Program     → Program Supply
Program 存在但无候选       → Observation / localization / supply
候选存在但选错            → selection / retrieval
意图与执行字节不一致      → execution / binding
Support 成功但后续失败     → Scope / overfitting / risk
结果可用但无法归因        → instrument / credit assignment
```

不得把每次失败解释为 Memory 不足。只有失败共享可观察 Context、相同 first fault，
且存在可验证替代 Workflow headroom 时，才新增或修改 Skill。无 Program headroom
时不建设复杂 Observation；无决策缺陷时不建设 Harness Update。

## 7. 反过度工程

- 前期禁止建设或扩展通用 SHA/Hash、hash chain、复杂 Receipt/Manifest、形式化
  Evidence Ledger、大型测试矩阵或平台层。
- 只有具体的数据混淆、串线、泄漏或不可解释结果需要一个决策性哈希时，才允许
  增加这一个；超过一个或需要新抽象时必须停止复核。
- 每个实验默认最多一个逻辑 Runner package、一个主报告和一个必要 smoke；
  plan/evaluate 双入口只用于 sealed-data boundary，不发展第二套框架。
- 现有历史 SHA、Runner 和工件保留，不因清理欲望迁移、补全或重写。
- 基础设施、测试、文档、Gate 和状态机不算方法进展。

判断任何新 Gate、Schema、测试或抽象前先问：

> 不做它，当前能产生方法证据的核心实验是否真的无法运行或解释？

答案不是明确的“是”就暂缓。

## 8. 数据与证据纪律

- Development 可看 Context 和 Outcome，只用于开发与 first-fault；不得包装成
  fresh、held-out 泛化或 Capability 正证据。
- Fresh Target 的 held-in/held-out split、Consumer、Program、反馈预算、Judge
  和 roster 必须在 Outcome 打开前冻结。
- Proxy 用于候选定位和低成本 credit；最终 Utility 必须来自真实下游 Consumer。
- Consumer、Metric、训练协议、聚合 estimand 与 split 是实验仪器，不得包装成
  Harness 自进化。
- 受控注入与 positive control 只验证机制，不替代自然数据能力证据。
- 正结果和可信负结果都可关闭具体 family/candidate；不得据此删除完整系统中的
  累积知识、Target 校准或 Context 条件化角色。

报告必须区分：

```text
CAPABILITY / MECHANISM / INFRASTRUCTURE / INSTRUMENT / NEGATIVE / INCONCLUSIVE
```

## 9. Agent 自主性与协作

研究者只提供 TaskSpec、Workspace 工具、Typed Operator DSL、信息墙和反馈预算；
不得根据已见 Outcome 手工指定最终 Workflow、正确 Skill/Scope 或处理顺序。

主 Agent 负责核心方法决策。委派深度固定为一层：只有根 Agent 可创建子 Agent，
子 Agent 不得继续 spawn；所有委派必须携带本文件的方向与反过度工程约束。

Windows 已有 Conda 环境 `project`；需要 Windows 原生 Python/测试/实验时可先运行
`conda activate project`。这是可用环境，不是必须改变当前 Shell 的要求。

## 10. 阶段性交付

每个大阶段只需回答：

```text
Harness 行为改变了什么
真实或可控数据上观察到了什么
当前最大方法不确定性是什么
是否仍与项目目标一致
下一项最小纵向切片是什么
```

当前路线状态、历史证据与任务编号详见：

- `docs/ROADMAP_POST_V1_2026-08-22.md`
- `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`
- `docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md`
