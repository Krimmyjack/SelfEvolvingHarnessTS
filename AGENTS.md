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

## 5. 当前状态锁（Main Protocol P4 拆分放行后，2026-08-30）

- Main Protocol 当前里程碑为：P0b 完成；P1 三任务基础合同完成；P2
  Forecast 风险控制生命周期机制通过；P3 三任务统一纵向接线通过。历史判词
  `P3_UNIFIED_VERTICAL_INTEGRATION_PASS__P4_HELD` 保持不变。后续拆分门控仅放行
  Forecast/Classification 的 `P4-Performance`（H1/H2）；`P4-Evolution`（H3）
  继续 HELD；`P4-AD` 只放行条件化与安全验收。Natural Final 继续封存。
  历史 split-1 拆分裁定见 `artifacts/main_protocol/p4_split_gate_20260830.json`。
  Forecast P4-Performance 的四次 B=4 live 尝试均作为非科学 FAILED 仪器记录保留：
  前两次被外部服务的 `model_price_error` 终止，第三次暴露事后预算检查故障，第四次
  在 Forward/E1/A3-reset 第 5 次调用前被修复后的硬守卫正确阻断；完整 unit 仍为
  0，不形成性能或科学判词。历史 canonical 检查点
  `artifacts/main_protocol/p4_forecast_performance_20260830.json` 不得覆盖。

  用户随后以前瞻性 split-2 将**仅 Forecast P4-Performance** 的 operating point
  统一提高到 B=8。split-2 合同为 7 次 Support-A + 1 次独立 Support-B、24 probes、
  6 LLM calls、60,000 tokens、最多 1 次 accepted update、2700 秒；A3-reset、
  K0-fixed、A5-online 使用完全相同的资源向量，A5 无预算例外。H2 等预算对照同步为
  `Parallel Best-of-N@8`，按冻结顺序评估 7 个既有单步 Common-DSL 候选，再只对
  Support-A winner 做 1 次 Support-B。该修订发生在 B=4 故障之后，B4/B8 不得
  合并；Classification 仍为 B=4，AD 仍只做安全验收。split-2 裁定写入
  `artifacts/main_protocol/p4_split_gate_forecast_b8_20260830.json`，未来 B8 输出为
  `artifacts/main_protocol/p4_forecast_performance_b8_20260830.json`。B8 live 已于
  2026-08-30 19:35+08:00 按 Forward→Reverse→Interleaved 全场启动；第一 replica
  只作运行健康观察，不按科学读数决定续跑。独立只读监视已同步启动，launch receipt
  为 `.aris/runs/forecast-p4-performance-b8-20260830/launch.json`。该 L6 运行随后在
  Reverse/E8/K0-fixed 的第 7 次调用前被 cell 硬守卫阻断，保留为
  `BUDGET_INSTRUMENT_LIMIT__NO_SCIENTIFIC_VERDICT`，不得覆盖或形成科学判词。

  用户最新以前瞻性 split-3 将三个 adaptive method-cell 的 LLM 上限统一为 8；
  其余 B=8 向量、方法、数据、seed、Prompt、Consumer 和阈值不变，A5 仍无预算
  例外。第 9 次调用继续在后端前阻断且不计费，但
  `LLM_CELL_BUDGET_EXHAUSTED` 现在只令该 cell 原子丢弃局部状态、identity abstain，
  随后继续其他 arm/unit/replica；全局预算、token/time、传输、协议和数据错误仍
  fail closed。A3-reset、K0-fixed、A5-online 的 cell 耗尽次数/率作为成本效率结果
  分别报告。新 gate 使用 `p4_split_gate_forecast_b8_llm8_20260830.json`。split-3
  首次发车因启动终端把临时凭据拼接两次而在首个 backend request 得到 401；0 个
  unit 完成，作为非科学认证仪器失败保留。干净重试输出改用
  `p4_forecast_performance_b8_llm8_run2_20260830.json`，不改变实验字段，也不覆盖
  split-2/split-3 失败工件。Natural Final 读取仍为 0，且未新增
  SHA/Manifest/Hash 基础设施。run2 已于 2026-08-30 21:23+08:00 按三 replicas
  发车，receipt 为
  `.aris/runs/forecast-p4-performance-b8-llm8-run2-20260830/launch.json`；独立只读
  monitor 同步运行，第一 replica 只作健康观察，不按科学结果决定是否续跑。
- P3 Classification 在已暴露 TRAIN 上用真实 Macro-F1 跑通了一次受控 Scope
  策略重放：Hampel 卡在 Epilepsy2 两面为正，在 PowerCons Support-A 为负后，
  机械收窄策略仍保留 Epilepsy2、并在同一 PowerCons 重放时停止供应旧卡。该
  重放不写 Harness、不形成 pending、不经独立 Support-B 批准、不增加 revision，
  也不是独立 re-encounter；因此 Classification Treatment 仍记 `NO_TREATMENT`，
  证据上限为 `MECHANICAL_SCOPE_NARROWING_REPLAY`。卡不是 Source 自然学得，
  失败归因不是 Agent 自主完成；不得据此声称非终止性 Skill 修订、自然性能、
  Source 学习或跨数据迁移。
- P3 AD 只完成 Yahoo 已暴露 24 条的 r1 TRAIN / Support-A / Support-B、固定
  IForest 和 identity Adapter 接线；未调用 Agent，未写 Episode/Skill/Store。
  #44a-r2 的主解释固定为 `INVERTED_EFFECT_OBSERVED`：Consumer 能读到变化，但
  清洗与 Event-F1 方向相反。AD 不承担正向性能主张，也不更换 Consumer、Metric
  或 event matching 追求正号；只承担 Task/Consumer 条件化、信号保护、安全拒绝
  与无负迁移证据。Classification 的 production revision reachability 仍未形成，
  因而不改变独立的 P4-Evolution HELD 判词。
- “生成→撤销”只算终止性风险控制，不算持续 Skill 进化。后续正式
  Evolution 必须分开报告 Skill 存活率、修订成功率和重遇收益；若只观察到
  ADD/REVOKE，最高只能判为 `RISK_CONTROL_ONLY`。至少需要一条“局部冲突
  →有限改 Scope/参数/Workflow→独立重验→存活→后续相似场景改善”的自然
  证据，才可支持完整持续进化主张。

- P4b 获准作为同源时间留出的前瞻性 bounded-risk 实验；旧 P4 不覆盖，Natural
  Final 继续关闭。已收缩为纯门实验（strict vs bounded 两臂）：审计 Source 卡在
  该批 origin 上 Scope 匹配 0/24，跨域积累 treatment 为空，故本轮不产出 §2.1
  的 `A5 vs A3` 读数；同一事实对旧 P4 的归因更正见
  `artifacts/main_protocol/p4_source_treatment_empty_correction_20260831.json`。
  **已收口（2026-08-31）**：48/48 held-in 完成，bounded 在 Support-A 上准入 6 次
  （strict 3 次），但独立 Support-B 全数拒绝 ⇒ 0 Active Skill；判词
  `BOUNDED_GATE_STILL_BLOCKING / blocking_face = SUPPORT_B`，held-out 未开启。
  结果见 `docs/P4B_BOUNDED_RISK_GATE_RESULT_2026-08-31.md`。

- Forecasting 纵向切片和 pooled Source 加速已有有界正证据；既有方法设计冻结
  复用，不因 AD 支线结果重写。
- 多 Task 基础设施与最小接线已完成到 #42k/#42k-b：Task/Consumer Context
  fail-closed、T6 Context 携带、候选帽与 H0 lock/Runner 键已校正。它们是实验
  资格与兼容修复，不是新的 Capability 证据。
- #42j 的主判为 `FIT_POLICY_NOT_QUALIFIED`：在 Yahoo 已曝光 24 条、现役
  IForest Consumer 与六候选供给下，mask fit-policy 未过宏效用、harm 和 worst
  三门。`f1_pooled` 仅为边缘 development 线索，未获 Support/晋升授权。该结果
  关闭当前 IForest × 现役供给/反馈切片的继续扩建，不关闭 AD、多 Consumer 或
  完整 Harness；不得继续追加第七程序或 U4/U5 来拟合这 24 条。
- AD Source family 已使用 4 cohort、40 Episode 封顶；
  `source_investigation_ad_v3` 已因行为效果不可归因而归档。不得追加第五个 Source
  cohort、修改 v3 或把相同生 Episode 再接回 Fast。
- Yahoo S5 A1 已下载 67 条，结构门 roster 为 65 条；前 24 条 outcome 已曝光，
  只能作 development，剩余 41 条保持 sealed，须等 development 管线形成可冻结
  状态后才可用于一次性验收。
- #42l 系列已收口：合法 `ABSTAINED` 的分类语义可信，相关旧测试/集成路由已按
  first-fault 处理或封存；该阶段只修仪器，不产 Capability 证据。
- #43 M0-C 已完成。在 Yahoo 已曝光 24 条、三个 AD Consumer 与现五清洗程序下，
  12 个程序级宏效用均为负；IForest/PCA 的预注册正负翻转未确认，PCA 也无安全
  headroom。该结果只关闭此数据、Consumer 与菜单组合上的继续正效应探针；不得
  增加第四 Consumer 或第七程序拟合这 24 条，也不得外推为 AD 无优化空间。
- 不再扩建或重跑 P3/#44a。P4-Performance 直接使用自然 Agent/反馈流程收集
  Forecast、Classification 的 H1/H2 证据；受控卡不得替代性能证据。
  P4-Evolution 只有在自然链条形成 pending → 独立 Support-B → promotion →
  versioned revision → 独立 re-encounter 后才可重新裁定 H3。
- 最终系统仍是 `A5 = 经审计的跨域积累 + Target held-in 多轮校准`；A3 与 Static
  仅为消融臂。后续仍欠可行 Consumer 上的 Target-local 生命周期/replay，以及
  sealed 41 条上的 Static/A3、具备合格积累知识时的 A5 同场验收。
- AdExchange、Yahoo 前 24 条、NOAA 2025 及已打开的 Source outcome 只能作
  development/replay，不得再次称为 fresh。

### 5.1 天然缺口线收口（2026-09-01）

收口文档 `docs/P4D_NATURAL_GAP_LINE_CLOSURE_2026-09-01.md`；数据源勘误见 §8.1。
数据身份 `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，与 P1–P4c 平行、
**不并表**。本轮全程 0 LLM、0 held-out 读、0 UCR TEST 字节，未调任何阈值、
未新增任何算子、未改动冻结的 P1 Common DSL。

**一、Forecast 出现开发级正向 Program headroom。**
`period_median_complete → outlier_*` 在 origin 2856 双面稳定（A 面 +0.56~+0.65、
B 面 +0.29~+0.32，两面受害序列 3–4/20，均在 0.20/0.30 预算内）；匹配对照确认
**缺口依赖**（已填补版 17 臂全不稳定）、**组合必要**（单算子在 2856 全不稳定）与
**顺序效应**（前向减反向 +0.21~+0.28）。它不是 Agent/Skill/held-out 正结果，
准确名称是：

```text
首个天然缺口上的、双面有界的组合程序正证据
```

**二、泛化仍未解决。** 仅 2/6 origin 存在稳定效果；没有程序跨多个 origin 稳定；
绑定约束首先在同一 origin 内的跨序列组（1176/2616 的 A/B 准入集完全不相交，
1896/2376 有一面准入数为零）。**因此 `P4-Evolution`、Natural Final 均不放行。**

**三、Targeting 整体判负，保留一个案例。** 正式判词继续是
`FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE`：跨面 12 折上 best-fixed +0.2629、
交叉拟合 Targeter +0.1908、per-series oracle +0.6106，Targeter 对 +0.3477 的
oracle 空间捕获率为负。origin 2856-B 的案例写作：

```text
一个合法的交叉拟合实例，将固定选择的风险失败转为预算通过
```

**不得**写成「Targeting 已有效」或「项目完整机制已得到证明」。绑定门的准确表述是：
**直接绑定门是最大单序列伤害；当前 15 维特征、深度 3 树和六程序菜单无法稳定
提前识别该风险。尚不能唯一归因于特征、模型容量或样本量。**

**四、Classification 当前菜单无稳定 headroom。** `ONE_FACE_POSITIVE_ONLY`，
UCR TEST 保持零读取。两条线的 first fault 不同：

- **Forecast**：有效 Program 已存在，卡在 Observation/选择与跨 origin 泛化。
- **Classification**：现有修复菜单大多无行为（13 个评估算子中 10 个改动点数为 0），
  全局变换又被局部修改门拒绝（5 个平滑算子全部
  `COHORT_MODIFICATION_FRACTION_EXCEEDED`），卡在 Program Space/验证契约。

**下一轮约束**：不得继续在这 6 个 origin、这 12 个折上拟合特征或树模型；新的自然
结构特征与未评价 origin 必须在打开前冻结，然后做一次真正的外部验证；全局可逆表示
变换需要**独立风险契约**，不得简单放宽现有 10% 修改门——该门度量被修改点的比例，
而可逆全局变换按定义修改 100% 的点，取值恒为 1.0，与温和程度和可逆性无关。

**方法学纪律**：按程序计数的结论必须先按逐序列增益向量去重。18 个算子中 4 个在
含缺口数据上零行为，故 396 个程序里存在大量别名；本轮 19 个稳定对去重后仅 7 个
不同效果。

**追加更正（见收口文档 §10，工件 `p4h_training_intervention_geometry.json`）**：

- 训练窗口 anchors 冻结为 `[312…852]`，保留条件 `anchor + 48 ≤ origin`，任何
  ≥900 的 origin 都让十个 anchor 全部通过，且 P4 路径从未覆盖 anchor 列表。语料
  指纹在两面上跨六个 origin **完全相同**，而 `x_train`/`y_train` 只来自训练窗口，
  故**对给定 (program, face)，六个 origin 共用同一个已拟合 Ridge**。因此
  **「跨 origin」应准确读作「跨评价窗口」**：它证明的是同一训练干预所得模型在多个
  预测窗口上的时间稳定性，不是该程序在六套独立训练条件上都能重新训练成功。
  origin-2856 的正结果不受影响。
- **单纯新增 origin 不再视为真正的外部训练泛化。** 阶段二主几何为**更换训练序列**
  （新 development cohort，生成新语料与新模型）；更换 anchor block 只作可选的时间
  稳健性测试，不得替代。仍属 **development cohort holdout，非 Final/held-out**；
  统计单位是训练 cohort/face，origin 只是其内部重复评价点。
- 窗口验证器**全有或全无**：Forecast `MAX_MODIFIED_FRACTION = 0.35`，
  `ScopeExecutor._verify` 逐训练窗口独立判定，200 个窗口中 1 个超限即拒掉整个程序、
  且因语料跨 origin 不变而在所有 origin 上永久被拒。故「396 → 171 可读」的收缩
  **有一部分来自合法性门而非性能失败**，二者不得混谈。
- `outlier_iqr` 的 Support-A 读数是 `WINDOW_VERIFIER_REJECTED (1 windows)`，
  **不是 gain=0**；此前的「恒为 +0.000」表述已撤回（误报源于报告表达式把缺失值
  强制成 0）。该算子在 support_b 语料上改动 108/200 个窗口、1042 个点。

### 5.2 实验主体更正：主协议至今测的是训练语料策展（2026-09-01）

收口见 `docs/P4D_NATURAL_GAP_LINE_CLOSURE_2026-09-01.md` §11；机械演示见
`artifacts/main_protocol/p4n_serving_side_gap.json`（0 LLM）。

**两个任务都不处理被服务的数据。** Forecast 的 `_evaluate` 只在 `train_rows`
循环内调用 `_apply_program`，评价 context 是 `_linear_integrity(raw[origin-192:
origin])`、真值是 raw 切片；Classification 的 `_prepared_fit` 只处理
`cell.fit_values`，`cell.surface(face)` 直接进 `model.predict`。

**Scope 无法豁免被伤害的序列。** `train_series_scope` 存在但主协议从未传过；它
过滤训练行，而 `roster("support_a")` 训练于 Support-B、评价于 Support-A，两集
不相交。实测 Scope 限制到 1/20 条训练序列，仍移动 20/20 条评价序列。Skill 的
`program_geometry.scope` 取值只有 `training_rows` / `historical_origins`。

因此 **P4/P4b/P4c/P4d/P4f/P4g/P4k/P4m（含 Agent 臂）测的都是"全局程序、训练语料
策展"，不是 AGENTS §1 描述的 Pattern-conditioned、Target-local Scoped Harness。**
这些负结果**不得**外推为完整 Scoped Harness 无效；`FEATURES_DO_NOT_BEAT_A_FIXED_
CHOICE` 缩窄为**仅对开环树 Router 成立**；`AUC 0.587` 只约束部署前静态预测，不
约束能读 Support-A 真实反馈的 Slow。Scope 亦非 `A5 == K0` 的唯一成因（Source 卡
不可达、严格门、0 Active Skill 同样参与），它是目前最上游的机制缺口。

**裁定（选乙）**：补齐 serving-side scoped pipeline，双管线——选中序列走
`prepared train → program model → prepared serve context`，未选序列走
`raw train → raw model → raw serve context`，使 Scope 外序列与 Static 逐位相等；
额外 Consumer fit 必须计费。Forecast 三表面为 `train_context+train_target` /
`serve_context`（只用 origin 之前）/ `evaluation_truth`（始终 raw）；
Classification 的 fit 与 serve features 同样处理、labels 不处理。

顺序：serving-side evaluator → ScopeSpec（存部署可见特征谓词，不存 UID）→
0-LLM 生命周期预检 → Static / A3 / A5 主实验。阶段二的 O1 表示算子按门控关闭；
不再优先建风险感知树 Targeter；不再寻找适合全部序列的全局程序。

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
- 新任务的 SHA 预算默认是 0。例外只限协议唯一真源、外部下载原始包，或已经出现
  具体串线/泄漏风险的密封材料；每个被校验物料最多保留一个直接服务当前决策的
  校验值。同一物料不得再派生 member/index/split/manifest/inventory 等多层哈希。
- 禁止为候选、Episode、Skill、split 索引、Runner 状态或报告批量生成逐项 SHA，
  禁止 candidate manifest hash、inventory digest、hash chain 和哈希平台。路径、
  冻结 seed/索引及机器可读语义字段足够时，必须使用这些信息而不是新增 SHA。
- SHA 只属于字节完整性仪器，不能作为方法证据、实验进展或 Gate 通过理由；若确需
  第二层派生哈希或新的哈希抽象，必须停止并先取得用户明确批准。
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

### 8.1 数据源勘误（2026-09-01，追加式，不覆盖历史数字）

Forecast 线（P1–P4c）全部工件把数据集标注为 `KDD Cup 2018 with missing values`，
**该标注是错的**。`data/kdd2018/series_cache.npz` 建自
`kdd_cup_2018_dataset_without_missing_values.tsf`，缓存内 **NaN 计数为 0**。

机械核验（`artifacts/main_protocol/p4d_natural_gap_roster.json`、
`p4d_natural_gap_preflight.json`，0 LLM）：两版本 UID 270/270、长度 270/270 对应，
**2,438,652 个观测位置逐值比对最大偏差 `0.000e+00`**，即 without 版 = with 版
经上游填补。天然缺口规模为 503,712 / 2,942,364 点 = **17.119%**，270/270 条序列
全部含缺失。

因此：

- **历史数字与判词全部保留、不覆盖、不重算**。它们在 without 版本上测得正确。
- 但它们能支持的结论范围收窄为**无缺口的 outlier / level / denoise 场景**。
  P4c 的 `NO_REPAIR_HEADROOM_CONFIRMED` **不关闭 imputation 方向**：identity 自身
  即 `_linear_integrity`（`run_e2_autonomous_natural_workflow_generation.py:543`），
  在无缺口数据上全部 imputation 算子退化为恒等，从未真正受考。
- 部署可见风险审计的 22 维特征中，7 维缺失类特征在该数据上是常数
  （grouped AUC 恰为 0.500），故 `AUC 0.587` 实为 15 维有效特征的读数。

含缺失版本记为**独立数据身份**，与既有结果平行、不合并：

```text
EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING   # 含缺失变体
EXPOSED_DEVELOPMENT__KDD2018_WITHOUT_MISSING        # 既有 P1–P4c 全部结果
```

同一数据域、同一批 UID，但数值条件不同：**不算 fresh 数据，不影响 UCR TEST 与
Natural Final，也不得与旧结果并表**。结构可读性在含缺口数据上重算后为 239/270，
roster 成员与旧线不同，两条线的 Support-A/B 不可互相引用。

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
- `docs/P4D_NATURAL_GAP_LINE_CLOSURE_2026-09-01.md`
