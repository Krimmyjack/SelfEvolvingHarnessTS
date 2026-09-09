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

### 5.3 决策单位更正：逐序列，不是 cohort 代表序列（2026-09-07）

本节更正的是**主线设计记录**，不是某一次实验的读数。历史读数全部保留。

**一、决策单位是单条序列及其当前合法窗口。** 到 DEV-KNOW-1 为止，主协议每个
单元只构造**一个** `PreparationRequest`：`_a5_request(cell.observation_block, ...)`，
而 `observation_block = values[support_a[0]][:origin]`
（`run_forecast_p4_performance.py:302` 起）——即该 block 的**第一条 eval 序列**。
Fast 读一条序列的可见 Context，产出一个 compiled 程序，由 `scoped_evaluate` 广播到
被服务的 20 条序列。这是仪器形状，不是 §1/§2.3 描述的方法：正确形状是每条序列
携带自己的 UID、values、部署可见 observed pattern、公开特征与自身历史，各自生成、
执行并被计分。

**二、共享的是 Agent、工具与 Skill，不是同一个处理程序。** 同一批决策共用一个
**冻结的知识版本**、同一个 Agent、同一套公开工具与同一个 Typed Operator DSL；
程序逐序列生成。允许不同序列选中相同程序，也允许 identity；**程序多样性不是成功
条件**，同质化的赋值本身是一条读数，不是失败。

**三、Scope 保留为适用条件、作用区间与执行边界，不再替代逐序列程序生成。**
Scope 的三个合法含义是：知识卡的 `observable_applicability`（这条知识对哪些序列
成立）、算子的作用几何、以及被服务序列集合。§5.2 的 serving-side 双管线是 Scope
的**执行端**；它不是"用一个谓词挑出一个全局程序"的替代物。逐序列决策下，执行
边界就是该决策所属的那条序列。

**四、旧的组级实验与判词保留，但测试范围明确。** P4 / P4b / P4c / P4d /
DEV-AUTO-1 / DEV-AUTO-2 / DEV-AUTO-3 / DEV-KNOW-1 等测的都是
「cohort 广播式单程序 + 组级知识写回」这一形状。它们的负结果与判词继续有效，
但**不得**当作原逐序列方法的验收结果；逐序列读数的分母是单条序列，与组级读数
**不并表、不相减**。

**五、与 R4A 的关系（2026-09-07 深夜）。** R4A 测得逐序列增益不是序列的稳定属性
（同序列同程序 g@origin 对 g@origin+48 的 Spearman 0.017–0.109，块内 ICC≈0）。
这**不改变决策单位**——观察、决策与执行本来就应该逐序列发生——但它约束**能对
逐序列条件化期待什么**：本节要求的是"按序列观察、决策、执行并逐条计分"，不是
"逐序列增益可由部署前特征预测"。R4A 关闭的是 series 级 Scope **修订**类实验，
不是逐序列决策本身。

**六、知识更新发生在批次边界。** 同一批内所有序列用同一个冻结知识版本；合法反馈
到达后，在批次边界由 Slow 提出**共享知识**更新，经未参与构造的后续反馈验证，
再供下一批 Fast 逐序列使用。单条序列的 delayed 损失不在批次中途撤销共享知识，
只作为边界的证据。首个按本节执行的开发级验证是 DEV-SEQ-1
（`docs/DEV_SEQ1_PER_SEQUENCE_2026-09-07.md`）。

### 5.4 本包方法边界：可编辑面与开发级采用规则（DEV-SEQ-2，2026-09-08）

追加式记录，只说明 DEV-SEQ-2 这一包在方法上放宽了什么、收紧了什么，不改写 5.3 的
决策单位结论，也不改动任何历史数字。

**一、两处工程修复。** (1) **窗口绑定**：可见特征卡此前按 series UID 存一份，构造段
第一个窗口先到先得（`run_dev_seq1.knowledge_boundary` 的 `setdefault`）。这些课程单元
是同一条课程沿时间推进，同一 UID 会在 u7 和 u8 各出现一次，于是所有 u8 决策都以 u7 的
特征被描述给 Slow、被验证器计分。现按 `(series_uid, origin)` 绑定，缺窗口记 miss，不借
别的窗口。(2) **验证去重**：一次真实读数被同一序列同一程序的两条 Episode 引用时，旧 L3
按 member 迭代会计成两个测量。现按预测缓存自己的语义键（unit, origin, 训练配置, typed
steps）加被服务 UID 记一个测量；**不删除任何 Episode**，不同窗口/程序/参数/训练配置仍是
不同测量，也不新增哈希体系。修复后重算 DEV-SEQ-1 run2 的 L2：覆盖从"6 条序列中的 1 条"
变成"7 个失败窗口中的 2 个"，**L2 仍然不过**——修复不构成批准旧卡的理由，报告因此把
"旧 L2 拒绝"与"没有有用条件"分开陈述。

**二、可编辑面从一个扩到四个（仍是软指导）。** DEV-SEQ-1 只开放
`skill_library.entries/{skill_id}` 的 ADD。本包按任务书开放四类对象，每次仍只改一处，
沿用既有编译器、分支与前置条件：新增情境化指导条目（ADD）；已有**指导型**条目的
`.body` / `.observable_applicability`（PATCH）；`bootstrap_skills.entries/{skill_id}.body`
（PATCH）；`candidate_policy.proposal_guidance` 与 `.selection_guidance`（PATCH）。
K0 中唯一的 capability 卡是冻结 Program 卡（body 为 `Frozen program steps:`），属执行体
而非指导，两个"已有指导条目"面因此在 K0 无实例、不供给。**confirmed_cause 由模型选中的
surface 机械推导**，不预先裁定故障类；五类故障标签退回统计标签与事实提示，不再对已授权
的指导面行使否决权。仍然关闭：`instruction.core`、retrieval top_k、候选槽数、risk_guards、
verification、算子/特征计算器、Consumer、评分与训练/服务几何；仍然禁止 `Frozen program
steps:` 出现在任何 body 或 patch 值里、禁止 `supplies_candidates` 与 serving Scope。
`edit_preflight` 对未列面 fail-closed。

**三、采用判据换成真实新旧 Fast，不再是谓词覆盖。** DEV-SEQ-1 的 L1/L2/L3 说的是"条件
覆盖了多少失败""旧程序是否还失败"，那是关于谓词的陈述，不是关于指导是否有用的陈述。
本包保留它们作**诊断**（廉价、可算、不设门），把门换成：在同一个 withheld 单元上用同一
人群、同一预算真跑两臂 Fast，B 相对 A 的配对 delayed 增益达到既有 MATERIAL，且 B 在该
单元通过既有权威门，读数完整可比。**不改常数、不改风险分母。** 这是**本包的开发级采用
标记**，不是生产部署许可，也不授予 Shared Capability。无论是否通过，合法候选都在 u10/u11
跑完：未通过者标为"未采用候选的隔离诊断分支"，其后续读数不计晋升/部署收益，也不反过来
追认验证单元通过。

**四、模型身份。** 本包固定 `deepseek-v4-pro`，并核对**返回**的模型标识而非请求别名
（DEV-SEQ-1 请求 `deepseek-chat`，该中转以 `deepseek-v4-flash` 应答）。因此 DEV-SEQ-2 与
DEV-SEQ-1 是不同实验，两者的数字不并表。

**五、人口规模按实测成本定。** 该非 Flash 模型每条逐序列决策约 149–260 墙钟秒。形成段
保持 10 条（边界需要足够材料才能成组），两臂各 6 条（同一固定 roster 顺序的前 6 条），
两个后续单元都跑满。开跑前写死，按成本而非按收益选，A/B 永远同一人群同一分母。配对 n
很小，报告如实说明，不做补偿。

**六、事后新增的开发级停止规则：零暴露不跑两臂（2026-09-08，必须留痕）。** 原任务书要求
"只要有合法候选且仪器健康就把两臂在后续单元跑完"，这条**未区分两种情况**：(a) Fast 读到了
指导但效果不好——仍应跑完，不能见负数就停；(b) 剩余样本**全部读不到**指导——继续只是在重复
父知识下的运行，无法检验这条指导。本包属于 (b)，因此新增：候选编译后先做**零成本暴露检查**
（`dev_seq2_knowledge.exposure_check`，在父/候选两个快照上渲染 Fast 实际读取的字节，覆盖全部
预定臂决策，0 fits / 0 LLM），命中为 0 则记 `CANDIDATE_COMPILED_BUT_NEVER_EXPOSED`、两臂不跑、
treatment 记 `COMPILED_BUT_NEVER_EXPOSED__UTILITY_UNTESTED`。**它不是覆盖门**——不要求条件解释
多数失败，只问"要花钱测的东西有没有被呈现"。适用该规则的组**不得作为完整两臂比较呈现**，
后续也**不得只挑指导确实加载了的组来展示**；一次这样的运行也不足以建立"噪声底"。

**七、账户/权限错误不是弃权（2026-09-08）。** `402 Insufficient Balance` 之类的错误意味着请求
**从未到达模型**，因此不得记为 Slow 弃权、不得盲重试、不得替换未授权模型；分类为
`ACCOUNT_OR_PERMISSION_FAULT`，边界结果单列 `BOUNDARY_ACCOUNT_OR_PERMISSION_FAULT`。

执行与结果见 `docs/DEV_SEQ2_SLOW_UPDATE_TO_FAST_RESULT_2026-09-08.md`。

**八、DEV-SEQ-3 的方法变化（2026-09-08，追加式）。** (1) **特征说明改为事实**：21 个可见特征
逐条给出公式与计算窗口，删除 DEV-SEQ-2 断言的漂移话术（只有一条被测过，一条序列上的测量不是
特征的性质）；明确区分**特征窗**（`values[:origin]`，全历史）、**服务窗**（`values[origin-192:origin]`，
Consumer 唯一可见的 192 点）与**修改区**（执行时才决定），并写明该词汇**不测量修改区**。
(2) **动作事实**：每条成败记录带"服务窗改动点数/总点数"与"共享训练材料改动点数/总点数"，
由既有 `_prepare`、既有 `_linear_integrity` 基线与既有容差**重放**已合法打开的窗口取得——
不重训、不花 Consumer 时间、不建新 schema；烟测证实训练侧计数与评价器自报的
`behavior_point_count` **逐位相等**。训练改动标注为**共享训练材料**的属性而非被评价序列的属性；
计数是动作不是效果；UNKNOWN 保持。(3) **批内并行**：批次串行、批内 4 路并发，每序列独立会话与
guard，拟合与共享缓存写入串行，汇总按固定 roster 顺序；实测加速 **×3.74**，读数与串行逐位相同、
不多花 fit、并发计费精确。(4) **行为链必须按暴露拆开读**：在全人群上数五个环节会把采样差异报成
行为改变——本包实测未暴露配对已有 70% 观察不同、19% 交付不同；因此环节计数只在**暴露子集**上
成立，未暴露子集单列为采样离散。(5) **过程字段必须进对照**：只比 `deployed_label`/Support/delayed
会漏掉模型是否服从——本包初版即因此误判。逐决策必须同时比 `chosen_candidate_id`、
`candidate_programs`、`tool_calls`、`history_episodes_in`。(6) **select 阶段没有交付否决权**
（`online_loop`：`chosen_candidate_id` 只定探测顺序，首个通过 Support 准入的候选取得部署权）：
实测 agent 的选择只决定 **53/116 = 45.7%** 的最终交付，选 identity 的 20 条中运行时照样部署 12 条。
**任何经 select 起作用的指导都不可能阻止一次部署**——设计卡片实验前必须先处理这条约束。
但那 12 条 delayed 合计 +3.5842（7 正 / 5 负），Support 门并非单向有害，不得据此断言运行时该改。
(7) **"未匹配"不等于"同知识"，但这可以直接测**：两臂 Episode 历史会分叉，并经
`method.py:374`→`fast_agent.py:804` 进入 Fast 上下文，所以"未暴露即同知识"是个待验命题而非前提。
验法是把两臂历史各走一遍 `resolve_experience_contrast_pack`→`render_experience_pack`，比**渲染文本**
而不是比条数：本包实测 60 对中条数不同 25 对、渲染文本不同仅 6 对、暴露决策上 0 对。先测再断言；
未测时只能写"来源尚未区分"。全人群终点**保留为主读数**（它回答整套机制对服务人群的净影响），
暴露子集与过程差异**加报**，不事后换口径。(8) **同知识配对多数是逐位相同的**：51 对严格同知识里
41 对完全相同，离散集中在少数序列。据此不得把"配对有差"泛化成"模型普遍抖动"。
执行与结果见 `docs/DEV_SEQ3_GUIDANCE_DELIVERY_RESULT_2026-09-08.md`。

**九、DEV-SEQ-4 的方法变化（2026-09-08）。** (1) **同起点对照**：要区分"卡片起作用"与"模型在抖"，
必须让两臂拿**同一份决策前历史**并**重复**同一格。历史按检查点重建后要**对照当时记录的
`history_episodes_in` 逐条核验**（本包 60/60 精确），否则重建出的"同起点"是假的。
(2) **对照要带邻近未匹配情境**：由卡片自身子句的**相对违反幅度**选出，规则开跑前定死。它们提供
**本设计内**的抖动基线；没有它，匹配面的任何差异都无法与抖动区分（本包基线 ±1，只有 0/3→3/3 超出）。
(3) **select 阶段没有交付否决权**（见八·(6)），因此**给 Slow 的输入必须包含这条机制事实**——
否则它会反复写在架构上不可能生效的指导。这是输入完善，不是替它写答案：陈述机制、附上
"覆盖并非单向有害（delayed 合计 +3.5842）"、不指定改哪个面。实测它据此**删掉了自己那条不可能实现的
预测**并改向候选提议。(4) **修订轮要把上一版的原文与它自己的预测还给 Slow**，并把
`.body`/`.observable_applicability` 一起放进目录，**明说保持不变是合法答案**；不得要求它必须 ADD、
必须换程序、必须给出修改。(5) **预算记账必须抗杀**：只在正常退出时落盘的预算，会在 OOM 时连同
花费一起丢失，下一个进程静默少算包内总额——每一步花钱后立刻落盘。(6) **恢复绝不重调 Slow**：
按 SHA 从 store 重新编译那条已编译的修订并校验哈希；再问一次既花钱又不会返回同一段文本，
等于偷换实验。(7) **一格一个 OS 进程**是本机内存下唯一可靠的长跑方式（进程退出才释放内存）。
执行与结果见 `docs/DEV_SEQ4_GUIDANCE_ADOPTION_RESULT_2026-09-08.md`。



### 5.5 当前推进重点：部署口径对齐（2026-09-08）

用户已同意推进第一包 `DEV-DEPLOY-1` 并同步文档，执行规格见
`docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_TASK_2026-09-08.md`；任务书可分发，不等于
执行者已收到或实验已启动。第一包不调用 Slow、不写新 Skill、不开放正式密封数据。

先对历史 Fast 原选择与 Support 交付做同历史影子审计，再在已曝光 development 上
真实模拟 `freeze → Fast-only → 输出冻结 → 外部评分`，同时跑不看当格反馈的确定性
基线。菜单 oracle 与效应尺度只作解释，不扩成前置平台。后续才比较旧知识与 Slow
自主修订知识对这种部署能力的增量；完整 A5/A3/Static 目标不缩减。

开发期 Support 选策与最终 Fast-only 的控制权不同。历史“53/116 = 45.7%”是选择与
交付一致率，不是 LLM 的因果贡献比例；选择 identity 不会否决候选池中的准入者，
不等于所有选择指导在所有运行条件下均无效。影子评分只改变一次动作、不重建全课程。
无反馈部署的 origin 预测分数由外部评价器给出；保持同一 P 到 +48 的读数是另一个
持续性问题，按新任务书单列，不改写旧 SEQ 包的 delayed 主终点。

路线检查点为 **2026-09-15**。届时根据真实部署能力、合法菜单空间及修订增量，决定
继续、明确换一项机制/环境，或停止当前机制的小修；故障导致未完成不判方法失败，
也不自动无限延期。旧 SEQ-4 收口独立记录，不为同一张卡追加新一轮小包。

**进度（2026-09-08/09，DEV-DEPLOY-1 第一包三项全部完成）**：影子审计（A）、
冻结 Fast-only 真实运行（B，120 次真实决策）、0-LLM 确定性基线（C）均已
完成，见 `docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_RESULT_2026-09-08.md`。Part A
独立重算精确复现了本节 (6) 记录的历史读数（53/116=45.7% 选择即交付；12 条
"选 identity 仍被改写"净值 +3.5842，7 正/5 负），可信度已交叉验证。Part B
开跑前重算的 K0 父知识快照 SHA 与 DEV-SEQ-3 g2 `knowledge-frozen` 臂历史
记录逐字节一致，单条决策烟测复现历史 probe 增益精确一致，随后 120/120 决策
0 故障完成：**拿掉 Support 准入门、且不带同批次 Episode 历史后，冻结
Fast-only 在 origin 面（三单元等权 0.058/0.097，两组）系统性低于不做任何
自适应的单一固定基线 D-fixed（0.118），高于 raw/D-safe（0），也明显低于
同一知识、同一模型但经准入门交付的历史水平（0.210）**。这与 §5.4(6) 及
本节上一条"准入门净贡献为正"方向一致，但**归因需要拆两层**（Fable 可行性
检查 2026-09-08 23:08 已预先指出，结果文档 §0/§2.4 已补记）：Part B 相对
历史同时去掉了准入门**和**同批 u9→u10→u11 的 Episode 历史累积，0.21 的
总差距不能整段记到准入门一项；唯一单独隔离"只去掉准入门"这一个变量的
证据仍是 §5.4(6) 与本节上一条引用的影子审计读数（0.0193 vs 0.2175）。
u9 单元风险画像尤为明显（harmed_fraction 0.65、单序列最大伤害 0.84，
两组均未过既有四线中的三线）。三问完整回答见结果文档
§0；不改变本节其余判词，不视为对 DEV-SEQ-4 的启动或终止，不追加第二次
Fast-only 冷跑。

**第二包（2026-09-09，已完成）**：DEV-DEPLOY-2 比较旧知识 F 与三种 Slow
修订方式（普通反思 R、比较式 C、去新增效果 C-minus）对冻结部署能力的
增量，见 `docs/DEV_DEPLOY2_FEEDBACK_TO_FROZEN_FAST_RESULT_2026-09-09.md`。
两组 C-minus 均弃权（缺新增效果信息时 Slow 不产出修订，确认增量依赖
效果反馈而非中性事实重组织）；C 在两组 origin 面均跑赢 F（+0.047、
+0.030），但两组均未跑赢同批数据校准的固定基线 D-new-fixed（0.284）——
与第一包同一模式：修订后的 Fast 仍不敌一个不做任何自适应的单一固定
程序。R 方向不一致（g1 明显更差、g2 略好于 F），样本小（各组40条配对），
不构成稳定结论。条件化增量尚未成立；后续默认优先简单 R，不追加同类
补丁包。过程中一次账户余额耗尽事故（连锁影响120条决策）已妥善处理：
原始故障记录全部保留备份，136条故障决策已用同一模型身份（`gpt-5.6-sol`，
经新中转核验）补跑，104条原有效决策未受影响；过程中修复了三处真实
代码缺陷（故障分类器漏判账户错误、CLI 未按故障退出、批次不完整时的
均值/UNKNOWN 语义），详见结果文档 §0 与 `docs/DECISIONS.md`。

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

### 9.1 研究协作分工

以下是参与本项目研发的协作者分工，不是被实验 Harness 的内部角色。
默认按任务设一名执行负责人，不要求每项工作串行经过所有协作者。

- **Astra（主助手）：方法、任务书与证据整合。** 维护研究问题和实验优先级，起草
  可执行任务书，复核影响结论的缺陷，整合结果及主张边界。负责结论不等于免于复核，
  不以个人裁定替代工件，也不把普通实现细节升级成三方审批。
- **Fable：系统整合、执行可行性与对抗核查。** 检查环境、恢复、交接和跨线一致性，
  对计划提出有依据的反方意见，对执行收据与报告的不一致作初核；后续主维护
  `docs/DECISIONS.md`。反方角色不豁免过强推论，不逐项审批常规补丁或预算分摊。
- **Opus / Kimi：主执行负责人，可相互替代。** 承担复杂代码修改、接线、必要测试、
  实验执行与结果分析。在已授权的范围、预算和停止条件内连续完成整包交付，
  普通实现问题自行解决并集中汇报。同一任务由其中一人主责，不默认双人重做。
- **Grok：高吞吐辅助执行。** 优先承担文献初筛、资料整理、日志与工件批量核对、
  数据资产清点、明确规则下的只读检查等可分解任务。复杂推理、关键补丁和决定
  方法路线的结论不由其单独定案，交主执行负责人或主助手核实；保留来源与未知项。

此分工不新增审批角色，不改变根 Agent 的最终整合责任和一层委派限制，也不因
协作者名称或可用 token 多而扩大任务、数据或工具授权。

**用户决定研究路线、重要协议变化和投入边界。** 普通实现由当包执行者连续完成。
决策记录区分“提议 / 用户批准 / 执行者收到 / 完成”；收到任务后无需等待日志维护者
在线才能执行已授权工作。记录由一人维护，其他线提供事实与回执，避免并发写同一页。
事实争议查代码/工件，仍影响路线的分歧才升级用户；任何角色均不得把建议写成批准。

### 9.2 以结果为中心的推进节奏

- **整包派工、连续执行。** 一次说明目标、范围、总资源上限和交付终点；执行者在
  授权内完成实现、必要测试、实验和分析，不把每个小修补拆成新的三方交接。
  已冻结的按臂预算、信息墙与停止条件仍须遵守，额度内的常规安排不逐笔请示。
- **开发试错与正式验证分开。** 在已授权的 development 范围内优先取得可解释的
  方法效果和自然修订链；正式比较及 fresh/密封终验前冻结必要协议。小规模验收
  是进入整体课程比较的台阶，不替代完整系统结果，也不自动授权下一阶段。
- **审核聚焦真正阻断。** 数据泄漏、对照或计分失效、实质性超预算风险、主要方法
  或冻结边界变更必须及时升级；其余不影响实验解释的字段、展示、成本分摊与普通
  修补随执行收尾，不单独停工。只暂停受影响部分，其他已授权工作继续。
- **关键点复核，不循环转接。** 必要的非作者复核集中覆盖影响结论的改动；已闭合
  问题不无故重开，后续只核新增差集和具体风险。可直接交接时由协作者交接，不让
  用户反复转述已确认的上下文；需要用户决定的只呈报实质选择或新增授权。
- **交付以研究结果为主。** 优先报告实际尝试、收益/伤害/成本、失败原因及下一项
  最有信息量的实验。测试数、文档数和边界修复数只是支撑，不替代方法进展；
  不得通过放宽判词、隐藏负结果或反复挑配置来追求正号。

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
