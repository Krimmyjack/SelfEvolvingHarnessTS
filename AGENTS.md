# SelfEvolvingHarnessTS 项目执行正典

本文件适用于本仓库及其所有子目录，是项目目标、实验优先级和证据边界的
长期权威。历史任务书与报告保留为证据记录；与本文件冲突时，以本文件和
用户当前指令为准。

**当前执行入口（2026-09-23 晚）**：后续设计为“离线学习 Workflow、部署时依据数据执行”（§5.11）。任务族筛选（§5.11.1）与 Planner 任务书 `DEV-AUG-OFFLINE-SKILL`（电力 / 交通 / 太阳能，A→B→C→D）均已收口（§5.11.2 阶段 A、§5.11.3 全包）；按任务书停止，下一包未起草、未启动。批级自主研究 Workflow 仍以 §5.9 为准；实体划分开发路线的历史收口见 §5.10。§5 中早期路线与收口保留为历史记录，不据其旧“下一步”另开实验。

- **最新已收口**：`DEV-AUG-TASK-FAMILY-SCREEN`（2026-09-23 12:20–14:36，0 LLM）：五个候选域（电力、交通，新增太阳能、空气质量、风电）的 Source 池 38 案 × 12 个统一程序 × 3 seed，共 1368 拟合，0 失败。按拟合前冻结的规则只携带 D01、D03，满足 F2 的携带族为 0，**按规则不建议直接做无卡 / 共享卡 / 域卡实验**。实质：增强偏好在数据集族层面（族内 0 对结构—处理关联过门槛）；四族强增强大幅有益（Preset 对 None +16 ~ +36 pp），交通相反（Preset −6.2、最差 −35），交通族最优比 Preset 好 7.1 pp、8/8 复现，但被 F1 的写法排除。固定程序层面各族选本族最佳、相对统一默认的开发优势约 1.4–2.9 pp（三域时约一半来自交通、一半来自太阳能，随打平的统一默认而变；不是域卡对共享卡的实测效果）。划分文档 `docs/AUG_TASK_FAMILY_DIVISION_V1.json`（五族，PROPOSED）。详见 §5.11.1。
- **上上包已收口**：`DEV-DOMAIN-AUG-TEMPORAL-COVERAGE`（数值 2026-09-22 02:14，报告/核账 09-23）COMPLETE：6 学习时期 / 2 选卡时期 / 4 测试时期、实体隔离，16 个 MetaTest 案。F_domain − F0 **+2.35 pp**（D01 +4.85、D02 −0.14；6/5/5；A12 时期 −2.38），D01 收益基本是回到预设；F_domain − F_shared **−0.04**（共享卡对 F0 +2.40）；F_domain − Fixed_dev **−1.42**；部署 token 比 F0 少 80%、新评估 13 vs 62。Fixed_dev 出现域差异（D01 P_NoMixRecipe、D02 Edit[-resample-random_conv]）。已知 30.91M token + 16 次已接受的未知 usage（上界约 32.9M）。判读 WEAK_POSITIVE_VS_NO_CARD / NOT_STABLE / DOMAIN_ORGANISATION_NOT_SUPPORTED，详见 §5.10.8。
- **上一包已收口**：`DEV-DOMAIN-AUG-MATCHED-LAG-FEEDBACK`（2026-09-21 13:34）COMPLETE：四案 31 候选在同实体上回放到 h1 = t−768、h2 = t−384（各自 672 小时训练、2000 步、三 seed，186 次历史拟合 + 6 次接线，0 LLM），近端/延后各四起点评分；四种固定 selector 同池比较（pp of None，两域等权）：R_HistLate 对 R_CA +0.10（2 胜 2 负，最大伤害 −4.17）、对 Fixed_dev −0.73（0/2/2）；R_HistNear 对 R_CA +0.85、对 Fixed_dev +0.02，3/4 与延后同交付；成对同向率 c_a 0.717 / hist_near 0.729 / hist_late 0.741。判词 DEFAULT_RECOVERY_ONLY：两次胜利都是回到 P_NoMixRecipe，两次离开预设都变差，延后位置无独有贡献；历史阶段墙钟 16.7 min（3 路）。详见 §5.10.7。
- **更早一包**：`DEV-DOMAIN-AUG-FIXED-POOL-SELECTION`（2026-09-21 02:41）COMPLETE：同候选池、同合法反馈视图下学习提交决策卡；每域三次独立 Slow 提案全部收敛为“默认预设 P_NoMixRecipe，除非其 C_A 对 None 两起点一致为负”；Select 三卡同交付、按 token 破同；Replay 8 案有卡对无卡 −0.14 pp（3/2/3）、对 C_A argmin −0.20、与 Fixed_dev（两域均 NoMix）8/8 逐案相同（否决分支 0/12 触发）、对 R_uniform +3.19；无卡 S0 在 11/12 决策等于 C_A argmin；40 请求 / 1.33M token / 13.9 min / 0 拟合；CAPABILITY = NO_INCREMENT（固定偏好复现），详见 §5.10.5。上一包 `DEV-DOMAIN-AUG-DECISION-PRIORITY` 收口见 §5.10.3。
- **当前状态**：`DEV-AUG-OFFLINE-SKILL` 收口（2026-09-23 21:02）：零反馈部署下 F_domain − F0 **+17.66 pp**（95% 聚类区间 [+11.5, +24.8]，32/0/8）、F_shared − F0 +16.29、F_domain − F_shared +1.37（区间跨零；差距全在交通）；收益来自纠正“按材料外观否决强增强”的偏差，交付物均为统一固定程序（F_domain − NoMix −1.01、− Fixed_source +0.71）。详见 §5.11.3。Natural Final 继续封存；第二 Consumer PatchTST 冻结方案迁移已收口（§5.11.6：域卡方案对无卡 −3.96，退步集中在电力），PatchTST 条件化离线学卡已收口（§5.11.7：新卡对原 MLP 卡 +3.22、对无卡 −2.47，PatchTST 上增强空间小），TSFM（Time-MoE）条件化学卡在服务器运行（§5.11.8），TSFM 仅准备方案。 朴素卡对照已收口（§5.11.4：域卡 − 朴素卡 +6.44，主要来自太阳能）；`DEV-AUG-MAIN-COMPARISON` 已收口（§5.11.5：主表九臂齐全，F_domain − AutoDA +24.8，AutoDA 适配后对 None −2.7）。
- **当前重点**：离线学到的域经验能否帮助 Agent 在新案例上零反馈地构造更有效的增强（§5.11）；主比较是同工具、同预算下的无卡、共享卡、域卡。不再在电力 / 交通两域上换反馈字段或提示词小修。

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

## 5. 历史状态锁与阶段更新（当前推进见 §5.10）

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

### 5.6 Skill 核心主线重申与后续审计限定（2026-09-09）

用户本轮确认：以 General / Specific Skill 指导每条序列当前合法窗口的 Workflow
生成；成败反馈由 Runtime / Slow 在批次边界用于有限 Skill 修订。主要修订对象是
指导正文、适用条件等，不是把项目收缩为选择一个冻结算子或不断增加程序卡。
这是 §1–2 目标的重申，不代表 8 月 28 日代码已完全符合目标，也不授权 git 回退。

General / Specific 是知识的语义角色；可复用既有 bootstrap、观察/提议/选择指导与
情境 Skill 的实现，不强制新增 Schema 或存储平台。模板可作参考，但不能让 Runner
暗中替代 Fast 生成。当前 Program 支持 1–4 步线性 Workflow；单步可以合理，不以
强制多步、多卡或更少弃权证明学习。Skill 适用条件、程序作用区间与评价人口分开。

后续应先修读数与真实证据入口，再在同一对齐后的生成执行器上比较新旧 Skill。
修执行器与修知识是两种变化，不能把前者的收益归因于反馈学习。具体代码发现、
方法假说与后续范围见 `docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md` §15。
本轮只进行只读核查与文档更新，不启动新实验，不改变 Consumer/训练/评分/风险边界。

**对 §5.5 完成回执的追加限定，历史数字不覆盖**：DEPLOY-2 的 240 行包含 18 条
空选择回退（g1 F/R/C 为 2/1/1，g2 为 13/1/0）；DEPLOY-1 也有 9 条。Runner 忽略
`prepare()` 返回的 FAILED 状态，内部失败可带 trace 返回并被记成 identity / 0，
外层 `faults=[]` 不能证明成功决策。原始逐次错误 receipt 缺失，不能唯一归因于中转
或模型。未来主机制读数须区分故障、正常 identity 与合法 raw 回退，不删行换分母；
若另报实际故障兜底的端到端效果，明确其含义，不能伪装成主动选择的机制证据。
因此 C−F 正差不能称两次干净复现；C-minus 两次弃权也不能单独证明有效性依赖反馈。
同反馈固定基线必须保留，但其风险亦须同报，不将高均值写成所有目标上的支配关系。

当前 Fast 仍选择完整准备—训练—服务管线，不能把逐序列 loss 自动解释为只清洗该
序列造成的效果。若后续转为固定 Consumer 的仅本序列处理，需另行明确干预几何。
已批准过的旧包与报告不因本节自动重跑、取消或改写；2026-09-15 路线检查点保留。

### 5.7 用户确认的训练与预测组织方式（2026-09-09）

用户明确确认目标：**每条训练序列各自生成 Workflow → 汇总处理后的训练数据 →
训练一个共享模型；预测时也逐条准备输入。** General / Specific Skill 仍指导逐序列
生成，同一批内知识冻结，Runtime 负责执行、汇总与 Consumer 拟合，Slow 在合法反馈
到达后的批次边界更新知识。Fast 不读取当前评价对象的下游效果。

“一个共享模型”指每个完整训练轮次/实验臂内汇总异质准备数据所得的 Consumer，
不是每种程序分别训练一个模型再按预测序列路由，也不是强制使用固定预训练模型。
这更新了 §5.2/§5.3 中双管线执行端作为未来主线的适用范围；历史代码与读数保留。
当前 `per_sequence.py` 仍从各程序对应的模型预测中逐行取数，**尚未实现本节目标**。

新组织方式下，单条训练序列的修改可能经共享模型影响多条评价序列；逐序列损失
仍可报告，但不能视为该序列处理的独立因果效应。不能拼接旧程序预测缓存中的行，
冒充异质训练材料共同拟合后的结果。旧 R4/SEQ/DEPLOY 的条件化结论只约束其原设置，
不直接验收或否定本节目标；更改组织方式本身也不证明学习有效。

本轮是用户设计确认与文档落实，未改训练代码、未启动实验。迁移应保留现有 Skill、
DSL、时间边界和历史工件；训练窗口及目标的具体处理权限、训练/预测角色的一致性
需在实施规格中明确，不由本节暗中决定。不新增哈希平台、不打开密封数据。
既有信息研究及待实施要点见设计备忘录 §17；9/15 检查点保留。

#### 5.7.1 DEV-TRAIN-1 实施规格确认（2026-09-09，追加）

用户在讨论训练目标权限后批准继续推进。首包20 train/20 eval，主臂允许 Workflow
准备整段历史训练窗口（192输入+48目标），评价真值始终 raw。保留真正只向程序提供
X 的输入侧对照；整窗口处理后取回 X 不等于未读取 y。输出分量分解只解释已处理
X/y 的条件效果，不宣称可加总归因。其形成段读数可交 Slow，后续评价不得回灌。

首包先比较 Static、形成段校准固定 Workflow、旧 Skill、一次 Slow 修订后的 Skill；
跨域积累另包检验。每臂最后一次知识更新后重新生成其训练赋值、拟合并冻结一个
共享模型，后续逐条准备预测输入但不重训、不由当格反馈覆盖 Fast。u9–u13 均为
曝光 development，不更名为 fresh/密封。原始 Episode 仍只供 Slow/Runtime。

当前用户指定普通实施由 Grok 主责，关键变更由根 Agent 复核，困难任务才交 Opus-5；
这细化 §9.1 的旧分工。实验调用 cpa-grok-4.6，明确入口 http://127.0.0.1:8318/v1，
不得默换模型或放宽远程 HTTP。具体执行以路线图顶部“执行定稿”为准，原草案存档
不再授权五臂/10人口/per-channel副表。此条记录批准，不代表代码已验收或效果已成立。

#### 5.7.2 新主线取消通用修改比例硬门（2026-09-10，用户批准）

用户明确回复“可以取消，同时其他的设计上呢”。此批准取消后续 Forecast TRAIN
主线的通用 35% 修改点比例上限；兼容现有字段可用 `maximum_modified_fraction=1.0`
表示不按修改比例淘汰。训练、预测、固定基线和新旧 Skill 臂权限一致，不能仅对白名单
算子放行。修改比例继续记录，但不当作伤害大小或方法有效性的代理。

该变化只对新的、显式标记的运行生效。保留旧 0.35 配置、canonical h0、历史读数；
新旧权限的决定/模型不能复用为同一次实验。真正生效的 Fast/actionable-menu、任务
约束、训练窗口与预测窗口验证必须一致，不得只改某处常量而留下隐形旧门。

不放宽合法观察时间、评价真值不可修改、输出/执行边界及结果风险口径；不授予 Slow
修改裁判的权限。不将此批准外推到 Classification/AD、任意代码编辑、新算子或密封
评价。取消比例门是动作权限改变，不是知识修订效果。其余设计建议见备忘录 §18，
待验证建议不自动成为已实施规则。此条记录批准，不宣称实现或新实验已完成。

### 5.8 当前研究判断与下一步计划（2026-09-11，文档记录）

用户要求将两轮外部调研及本轮讨论的看法、想法和后续计划落实到文档，统一入口为
[Harness 研究方向与计划](docs/HARNESS_RESEARCH_DIRECTION_AND_PLAN.md)。本节记录当前
研究判断与拟议工作，不将建议写成已批准实施或已取得结果。

核心主题继续是面向数据就绪／数据增强的 Agent Harness 自进化；General／Specific、
Fast／Slow、逐序列 Workflow、共享 Consumer 及完整 A5 定义保留。调查方法、处置经验和
条件化程序复用可共同积累；不要求所有有效知识都表现为“研究技能”，也不以非 identity
率、程序多样性或卡片增长证明学习。

下一步建议先用现有预测数据就绪环境检验“依据当前合法证据主动选择下一项对照，并把
有效调查／构造方法用于后续任务”。已有 General 编辑和角色对比不作为新发现。公共
能力边界修复由各臂共同获得；对照中的其余训练赋值固定，权威反馈来自真实共享拟合，
不能拆成逐序列独立因果标签。Fast 不直接读取当前下游效果，Slow 仍在批次边界工作，
最终 held-out 保持冻结 Fast-only。

反馈取得策略比较保持工具、权限与预算相同，允许所选实验不同；更新器比较才固定
同一份反馈材料。固定 H 的任务内搜索不等于 A3，完整积累仍由同 Target 预算的 A5/A3
评价。轻量增强保留为扩展候选，当前不并行迁移或以纯数值计划优化替代原主题。

现有 TRAIN3 General 草案保持未批准状态；其同材料再改指导不能直接作为上述新机制
实验。本文档任务不启动实验、不开放密封材料、不改历史读数。9/15 路线检查点保留；
具体实现、数据与预算在后续实施规格中定稿，不由外部报告的时间和阈值自动授权。

### 5.9 批级自主研究 Workflow（2026-09-13，用户确认的新设计）

用户确认：Fast 面向一批训练数据生成处理策略，策略覆盖观察、决策、构造、实验和执行；整体训练价值作为反馈，在贯通 Fast 后优化效果及经验进化。用户进一步要求落实文档并安排后续任务。

本节是对 §5.3、§5.6–5.8 中“Fast 必须逐序列独立生成”及对应下一步顺序的显式更新。后续批级开发以本节为准；历史实验及判词保留原设置，不并表、不重写。项目积累＋Target适应目标、完整A5定义、共享Consumer和最终E信息墙保留。

- **单位统一**：Fast接收完整Job的Task、Consumer、全批可见Context和预算，生成并执行批级Workflow。一个候选的全部原始/派生材料共同训练一个Consumer；配对seed各自训练，不拼接不同模型的逐实体预测。
- **批级不等于统一算子**：Workflow可按实际观察对实体/窗口分组、设置默认和例外，也可全体相同。不能从第一条数据决定后广播；不以多样性或每实体都改善为门。全人口宏效用为主，局部读数用于诊断。
- **Fast/Slow分工**：Fast在held-in内可取得当前合法C_A反馈，组织观察、构造、比较、停止与commit；同一Job知识冻结。Slow在Job边界读取过程与合法延迟反馈，修改观察/构造/实验/决策指导或Specific适用条件；每次聚焦一个主要行为机制。
- **知识通路**：原始跨作业Episode仍只供Slow/Runtime，Fast读取冻结General/Specific及当前合法工具结果。匹配可先取得候选Skill，再补查其要求的T观察；MATCH不等于效果已证明。
- **交付控制**：新批级协议由Fast根据C_A提交已真实拟合的完整候选；C_B在commit冻结后作延迟检查，不能再由旧argmin覆写当前交付。此变化只适用于新明确标记的运行。最终E仍在所有输出冻结后统一开放，不回流本轮。
- **可编辑边界**：允许观察、材料构造、实验组织、决策指导的有限改进；Consumer、评分、训练目标、预算与标签权限不交给Slow改。观察与材料诊断不冒充真实训练价值。
- **推进顺序**：M1完成可运行闭环，M2优化Fast净收益/成本，M3检验Source积累与A3/A5。不得把每一个局部headroom预检都升级为M1建设前置门；也不将M1完成写成性能或持续学习通过。§6的单机制归因纪律用于后续效果试验，不阻止本次必要接口共同接通。
- **首包历史状态（非最新进度）**：W首包三个分支已真实COMPLETE，31次拟合（含对齐1次）、12次实验LLM；Source候选实际形成且后续3/3请求加载。完成一次批级最小开发闭环，未支持Fast净收益或新H积累增益，不晋升M2/M3。实际收口见首包任务书§10.3与 `_scratch/dev_batch_research_workflow_v1/REPORT.md`；本包已停止，Natural Final及旧P4冻结状态不变。

统一框架：[批级研究Workflow规格](docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md)。
首包：[DEV-BATCH-RESEARCH-WORKFLOW-V1](docs/DEV_BATCH_RESEARCH_WORKFLOW_V1_TASK_2026-09-13.md)。
当前路线见 docs/HARNESS_RESEARCH_DIRECTION_AND_PLAN.md §6.30。本次没有批准git提交、覆盖历史包或新增哈希。

### 5.10 域 Skill 的实体划分开发 setting（2026-09-19，用户确认重定）

用户确认尽快按项目需求重定实验：同域不同序列/实体组承担 Skill 学习、选优和复用验证，恢复有界完整增强构造，不将研究压成固定配方的组件开关。本节更新当前开发任务的案例组织、动作范围和推进顺序；§1–3 的完整系统目标与 Natural Final 边界保留。

**旧包状态与可复用教训**

`DEV-TEMPO-AUG-DOMAIN-SKILL-OPTIMIZE` 已按原规格完成并停止，身份为 `EXPOSED_DEVELOPMENT_REPLAY`，出口 `NO_REPLAY_INCREMENT / GUARD_LEARNED_BUT_UNTRIGGERED`。卡学会“只在配方族内按 C_A 排序，其他公共参照的 C_A 领先不覆盖”；开发阶段确实改变提交，但 L6/L7 守卫没有触发，两批有卡交付与 NoMix 相同，相对无卡平均 −0.05 pp。L4 的 +36.1 pp 是相对影子 C_A argmin，不是真实无卡臂增量。

这说明历史经验能够改变提交规则，但尚未证明新案例效用；不得外推为 Slow 学不会、公共参照普遍有害或条件化无价值。旧卡的“只留在配方族/不分组”不能作为新两域的默认知识。报告见 [_scratch/dev_tempo_aug_domain_skill_optimize/REPORT.md](_scratch/dev_tempo_aug_domain_skill_optimize/REPORT.md)。保留旧任务原预算、实际用量与 amendment，不以实际用量追改原上限；不据旧任务重跑。

**当前实验设置与方法边界**

- **两层划分**：外层 Source / Select / MetaTest 按实体互斥；内层仍按时间切 Consumer 训练窗口和评价块。先分实体，再切窗口，不能将重叠窗口随机拆成独立 Skill 案例。
- **当前数据**：D01 Electricity、D02 Traffic。每域 Source 128 条序列/8 案例、Select 32 条/2 案例、MetaTest 32 条/2 案例；每案例 16 条序列，共同训练一个共享 Consumer。三个集合覆盖同样的两个时间切点，具体名单和几何读冻结 JSON。窗口数、实体数和 seed 数均不能冒充独立研究经历数。
- **学习主体**：Skill 是可执行的观察、假说、构造、实验、提交与停止指导，含 Workflow 与可选 Principles；完整数值处理程序是 Fast 使用这些指导后的产物，两者不能混称。Opus 是开发执行者；实验内 Fast/Slow 及模型身份按任务书，角色不能混淆。
- **域内复用**：每域从本轮 Source 的真实研究轨迹与合法后期效果生成多张候选，Select 实测选卡，随后按已知域身份加载到新实体组。无画像 Router。原始跨案例 Episode 只给 Slow；Fast 只读冻结卡与本案例合法工具结果。新包不导入旧 RD02 卡或人工按域指定答案。
- **完整构造**：七个既有 TempoPFN 原语的合法 1–3 步组合、按观察条件化赋值、公共 NoMix 及关闭 0–2 组件的局部编辑都可用。参数分布、顺序和互斥规则沿用任务书；不开放任意代码、新算子、Consumer 调参或采样权重。NoMix 不具强制起点地位，显式组合不得被 edit-only 接口挡住。
- **权限与经验分开**：不能从“单实体 loss 不等于其材料的独立因果贡献”推出“不准分组”；未尝试不等于有害。Skill 可提出有证据的候选优先级或提交偏好，但软指导不改工具权限；不强制复杂、分组、用满槽或偏离默认来制造行为差异。
- **提交控制**：Fast 可依据当前合法观察、C_A 和冻结 Skill 提交任一自身已评估方案或公共参照；Runner 不强制改为 C_A argmin，不预置“3/3 才提交”或“必须留在 NoMix 家族”。C_A 有限且可能与后期冲突；恢复完整空间和更换 split 本身不保证解决该问题。
- **必要比较**：同工具、同预算比较有卡、无卡、通用指导、随机搜索与开发选出的固定方案，四公共参照继续同场。主拆解是经验学习是否改善新实体任务，不要求无卡先胜 NoMix 才允许学 Skill；只胜 None 不能记为 Skill 增量。若学得固定偏好照实报告，不强制将其写成动态适应。
- **信息权限**：Source/Select 的合法后期评分用于本轮学习/选优；MetaTest 有当前 C_A 支持反馈，但卡全程冻结、无 Slow。C_B/E 在输出冻结后统一开放，不回流本轮。MetaTest 是开发级新实体任务，不是 §3 的零反馈 held-out。数据源历史曝光身份保留为 `SERIES_DISJOINT_DEVELOPMENT`，不称全新终验。
- **执行与状态**：本包已连续完成接线、16 个 Source 案例、两域形成、12 条 Select 分支、20 条 MetaTest 分支和报告，并停止。实际结果与偏差见 §5.10.1；不自动插入旧批次诊断、局部修订或下一包。后续常规技术问题在已授权范围内解决，数据边界、方法与预算的实质变化按 §9.2 处理。

执行规格：[DEV-DOMAIN-AUG-ENTITY-SPLIT](docs/DEV_DOMAIN_AUG_ENTITY_SPLIT_TASK_2026-09-19.md)；实体与时间名单：[DOMAIN_AUG_ENTITY_SPLIT_V1.json](docs/DOMAIN_AUG_ENTITY_SPLIT_V1.json)。数据、模型、预算、输出与停止条件由任务书限定。本节不授权新封存区、git commit、新增 SHA/Hash，也不将本开发组件的完成等同于完整 A5 或持续进化通过。

#### 5.10.1 实体划分包实际收口（2026-09-20）

报告：[_scratch/dev_domain_aug_entity_split/REPORT.md](_scratch/dev_domain_aug_entity_split/REPORT.md)；机器读数：[result.json](_scratch/dev_domain_aug_entity_split/result.json)。身份仍是 `SERIES_DISJOINT_DEVELOPMENT`，不是 Natural Final；本次没有多轮进化或跨域共享 Skill 对照。

- **方法读数**：F_domain 相对 F0 为 +0.74 pp（D01 +2.24、D02 −0.75；四案例 1 正、1 负、2 同交付），相对 Random_B4 +1.66 pp，相对 Fixed_dev/NoMix +0.01 pp。均以各案例 None 均值为分母。域卡对 None +10.88、固定 NoMix +10.87，不能把这约 11 pp 全归于 Agent 学习，也不从接近零推断总体等效。
- **具体行为**：D01_Q02 的卡优先评估并提交 Comp[censor]，比 F0 好 4.48 pp（三 seed 同向）；F0 构造了 censor 却未评估，已用满 24 次工具调用。D02_Q02 卡提交联合编辑，比 F0 差 1.50 pp。MetaTest 十二条 Fast 的提交均等于自身池 C_A argmin；当前可观察作用主要是改变候选评估与预算分配，不是已学会更好的提交排序。
- **选卡限度**：每域三卡在两个 Select 案例均同交付、J 并列，按更少评估破同。两域 Fixed_dev 都为 NoMix；卡片内容有域差异，不等于已经证明 per-domain 优于共享卡。完整构造接口已被实际使用，Source 出现显式原语、单/双编辑及条件化方案；不能再称“接口只有 NoMix 开关”。
- **成本口径**：MetaTest F_domain/F0 为 809,220/1,759,785 个已记录 token，新增评估 7/14。50% 是新增候选拟合节省；四案例公共参照共 48 拟合也计入独立部署时，为 69/90 拟合（约少 23%，按工作量计算，不是本包缓存后的物理费用）。Source＋形成＋Select 共 11,195,705 个已记录 token、447 拟合，另有 12 次接线；0.60M 只指 Slow 形成调用，不能代表完整学习成本。
- **运行与完整性**：609 次拟合全部成功，账本 309 请求、约 15.10M 已记录 token、付费墙钟 4h17m。预算按用户指示修订，原上限与 amendment 分列保留。Select 双进程事故覆盖两份响应并重跑两条 W3 轨迹，另有一次已接受的未知 usage；不能将重建后的 token 写成完整精确总额或声称协议零偏差。D02-W3 正文含 S06/S08 编号，虽未选中仍保留为文本契约偏差。原报告与工件不追改。
- **推进边界**：保留新 setting 与 D01 的候选优先级正例，同时保留 D02 的伤害；不强制回到旧时间重放或把“只采用固定方案”写成 Skill 改进。若后续把固定程序纳入系统采用集，须与研究 Skill 本身的增量分开报告；当前不启动该修改或新实验。

#### 5.10.2 下一包：研究决策优先级与并行复验（2026-09-20，待派工）

用户要求安排后续任务，并将可独立的拟合/API 调用并行。本包复用上一包每域
12 个已完成实体案例作为下一代学习材料；保留其历史测试身份和旧报告，不重跑
Source。新名单按原洗牌继续取每域 96 个未用实体：2 个 Select、4 个 MetaTest
案例，与上一包及彼此互斥；仍为 SERIES_DISJOINT_DEVELOPMENT。

方法重点是从真实决策点、当时可见证据和离线后期反馈学习观察/构造/评估优先级。
材料外观不等于下游效用；D01_Q02 的 censor 未评估既有主动降优先级也有工具消耗，
不能仅归因于没有预算提示。每域三次独立 Slow 提案，实测与旧卡比较；新实体比较
无卡、旧卡、新卡、随机、固定程序；本轮以旧卡替代再跑通用指导，保留四公共参照。
共同契约修复与上下文去重由所有臂共享；不预写
哪个域该用哪个原语，不强制复杂/分组，不把 NoMix 改成强制起点或唯一准入门。

并行采用单协调器、跨案例最多 4 路/API 最多 4 在途，全包数值重任务目标 3 路，
同案例依赖顺序与标签屏障保留。并行前修复按全局计数差分摊费用/判断缓存、请求
编号竞争和双恢复覆盖；沿用现役数值环境，内存不足时降低并发而不改训练语义。
不建设新调度/哈希平台。

规格：[DEV-DOMAIN-AUG-DECISION-PRIORITY](docs/DEV_DOMAIN_AUG_DECISION_PRIORITY_TASK_2026-09-20.md)；
名单：[DOMAIN_AUG_DECISION_PRIORITY_V1.json](docs/DOMAIN_AUG_DECISION_PRIORITY_V1.json)。
本节写于派工前（READY_FOR_DISPATCH）；实际执行与收口见 §5.10.3。本节不开放 Natural Final 或额外域，
不更新既有科学判词。完整范围、采用/比较规则、规模、费用记录与停止条件读任务书。

#### 5.10.3 决策优先级包实际收口（2026-09-20 12:50）

报告：[_scratch/dev_domain_aug_decision_priority/REPORT.md](_scratch/dev_domain_aug_decision_priority/REPORT.md)；方法与接线 [METHOD.md](_scratch/dev_domain_aug_decision_priority/METHOD.md)；
机器读数 [result.json](_scratch/dev_domain_aug_decision_priority/result.json)。身份 `SERIES_DISJOINT_DEVELOPMENT`；8 个新实体案例共享两个切点，不是 Natural Final。

- **方法读数**（pp of None，正 = 前者好，域内四案例等权再两域等权）：F_new − F_old **+0.45**（D01 0.00，四案例交付完全相同；D02 +0.89，2 胜 2 平，SE 0.36 / 1.77）；
  F_new − F0 **−0.44**（2 胜 2 平 4 负）；F_new − RandomSearch_B4 −1.38（3/0/5）；F_new − Fixed_dev −0.78（0/5/3）；F_new − NoMix −0.11；F_new − None +8.20。
  各臂对 None：Random_B4 +9.58、Fixed_dev +8.97、F0 +8.64、NoMix +8.31、F_new +8.20、F_old +7.75。判词 CAPABILITY = NO_INCREMENT / INCONCLUSIVE；
  不能记为 Skill 增量，也不从近零推断等效。
- **采用决定**：H_deploy 两域都按同一 J 选了固定程序（D01 Edit[-random_conv] J 0.922；D02 P_NoMixRecipe J 0.911，与 W_new 并列后固定优先）；对 F0 +0.33、
  对 F_old +1.23、对 F_new +0.78、对 NoMix +0.67，全部来自固定程序本身；学习卡不在采用集内。固定程序进入采用集的小幅收益与研究 Skill 无增量分开记录。
- **学到什么**：六次独立 Slow 提案全部形成互不相同的卡（无 KEEP）；D01 三张都收敛到“预设 C_A 不差于 None 时不被 C_A 领先的编辑覆盖”（源于父包 V02/S04 的后期反转），
  W_new = N2（禁 shock 关闭覆盖）在 Select 与 MetaTest 的交付与旧卡完全相同，只减少材料检查（每案例 1 次 vs 3 次）；D02 W_new = N1 是“预设 C_A ≤ None 则 0 评估直接提交预设”的
  固定偏好，在 2/4 新案例 1 次调用即提交，2/4 案例走评估分支（Q03 多评 Comp[amplitude] 得 +1.65 vs 旧卡；Q06 锁预设放弃了公共 C_A argmin，−1.49 vs F0）。
- **研究过程**：共同材料语义段下，构造未评估 F0 1 / F_old 0 / F_new 0，“未验证写成有害”启发式 0/24 条轨迹；三条 Fast 臂提交 = 自身池 C_A argmin 各 7/8；C_A/E 冲突再现
  （D02_Q05 三臂池内 E 最优均为未提交的 NoMix）。0-LLM 随机搜索（4 槽 + C_A argmin）是平均最强臂，Fast 卡臂只用 0–2 槽。
- **运行与完整性**：402 拟合全部成功、0 重试、771 缓存；245 逻辑请求 / 248 HTTP；9.09M token；付费墙钟 1 h 25 min。并行实际使用：活跃案例 4、HTTP 峰值 4/4、
  数值池 2（接线实测内存规则，计划 3）；Select 16 条轨迹 + 标签 19.7 min。三次中转瞬断（1 次 300 s 超时、2 次 500）均由同一逻辑请求的有界重试成功，冻结规则各记 1 次未知 usage
  并停止派发，操作者逐次显式接受并恢复（无重复付费调用、无标签泄漏）；两处无科学影响的技术修复（写一次记录幂等、付费时钟顺序并按账本事件回填）。原任务书上限保留，未触顶。
- **推进边界**：保留 D02_Q03 的候选供给正例与 D02_Q06 的锁定伤害；不把“采用固定程序”写成 Skill 改进；不自动追加提示词修订或新实验。报告提出的唯一下一项是
  0-LLM 可部分回答的诊断（同 4 槽下随机供给 + C_A argmin 与 Fast 自选供给 + C_A argmin 的差距来源），需用户另行授权才启动。

#### 5.10.4 同候选池的提交决策学习（2026-09-21 起草；实际执行与收口见 §5.10.5）

用户要求继续推进并保留独立 API 并行。下一项为
[DEV-DOMAIN-AUG-FIXED-POOL-SELECTION](docs/DEV_DOMAIN_AUG_FIXED_POOL_SELECTION_TASK_2026-09-21.md)，
状态 READY_FOR_DISPATCH，起草时未启动实验。

复用既有 F0 的实际已评估材料：每案四公共参照加其自造候选，同案所有 selector
看到相同池与合法反馈。Learn 仅父包两域 S01–S08；Select 用父包 Q01/Q02；
Replay 用后包 Q03–Q06。Slow 不读旧卡或包含 Q 结果的总结。外层实体互斥，
结果早已曝光，明确记 EXPOSED_DEVELOPMENT_REPLAY，不恢复独立终验身份。

每域三次独立 Slow 提案，Select 实测后冻结一张提交 Skill；Replay 比较有卡、
无卡、C_A argmin、开发选定固定公共程序及池内均匀选择期望。当前 C_A 已有
两起点、三 seed；本包整理已有细分反馈，不新增时间覆盖，不预设 3/3 或 NoMix
守卫。不要求先证明相关性或显著胜固定方案才测试学习；各比较的实际差值照报。

这是完整研究 Harness 的提交组件实验，完整原语组合/条件化/编辑构造能力保留，
不将最终系统改成固定菜单。正常 38 次主要 LLM 请求，至多 4 路 HTTP；单协调器，
新增 Consumer 拟合/预测推理/标签/SHA/commit 均为 0。范围、纠正/重试及未知 usage 规则、
同池比较和收口终点按任务书；不自动开反馈重设计、缺口修复或下一包。

#### 5.10.5 同池提交决策包实际收口（2026-09-21 02:41）

报告：[_scratch/dev_domain_aug_fixed_pool_selection/REPORT.md](_scratch/dev_domain_aug_fixed_pool_selection/REPORT.md)；方法与接线 [METHOD.md](_scratch/dev_domain_aug_fixed_pool_selection/METHOD.md)；
机器读数 [result.json](_scratch/dev_domain_aug_fixed_pool_selection/result.json)。身份 `EXPOSED_DEVELOPMENT_REPLAY`（实体互斥保留，后期结果历史已曝光；两个共享切点）。

- **方法读数**（pp of None，正 = 前者好，域内四 Replay 案例等权再两域等权）：S_W* − S0 **−0.14**（D01 −1.21、D02 +0.92；3 胜 2 同交付 3 负；最大伤害 −5.20 于 D01_Q05；
  跨案例 SD 3.3/2.0 远大于 seed SE 0.43/0.17）；S_W* − R_CA −0.20（3/1/4）；S_W* − Fixed_dev **0.00（8/8 同交付）**；S_W* − R_uniform +3.19；S_W* − None +8.31；S0 − R_CA −0.05（0/7/1）。
  各臂对 None：H_select 8.91、R_CA 8.51、S0 8.45、S_W* = Fixed_dev = NoMix 8.31、R_uniform 5.12。判词 CAPABILITY = NO_INCREMENT（固定偏好复现）。
- **学到什么**：六次独立 Slow 提案（每域三次、无旧卡、只读 S01–S08）全部形成互不相同但同机制的卡：默认提交预设，只有预设对 None 的 C_A 配对差在两起点（D01 另要求三 seed 且超 seed SE）一致为负才离开预设、改提交 C_A 领先者。
  该否决条件在 16 个 Learn 池中 3 次为真（正是预设后期失败的三个案例），在 4 个 Select + 8 个 Replay 池中 0 次为真；有卡 Agent 12/12 次提交预设。动态分支未被实测，也未被否定。
- **无卡行为**：共同合法视图下的单次提交 S0 在 11/12 决策等于池内 C_A argmin；池内对 NoMix 的候选机会平均 3.5 pp 存在，但 S0/R_CA 只兑现 +0.14，C_A/E 冲突再现（D01_Q06 预设 C_A 全正、E −4.28）。
- **采用读数**：Fixed_dev 两域均 P_NoMixRecipe；H_select D01 = R_CA（与 S0 并列）、D02 = Fixed_dev（与 S_W* 并列）；对 S0 +0.46，是 Select 选择效应，不是 Skill 增量。
- **运行与完整性**：40 逻辑请求 = 40 HTTP（38 主 + 2 次 S0 格式纠正，纠正后选择不变）、0 瞬断、0 未知 usage、0 操作者接受；1.19M 输入 / 0.13M 输出 token；付费墙钟 13.9 min；0 拟合 / 预测 / 标签 / SHA / commit。
  偏差：Slow 输入实际 88k/84k token，高于任务书“约 60K”目标（为保留每案全部候选与反例未再删减，后端接受）；W* 两域由 Select token 破同产生。smoke 27/27、接线（R_CA = 父包 shadow argmin、池 E 比值一致）通过。
- **推进边界**：按任务书 §10 第三/四种结果停止本包；不把“采用固定预设”写成 Skill 改进，也不据本包宣布没有可学信息。候选下一设计（T 内多历史伪切点 / 匹配延后间隔反馈，单独计训练成本）留给用户/Planner 决定，未起草、未启动。

#### 5.10.6 下一包：同实体、同候选的历史延后反馈（2026-09-21，待派工）

规格：[DEV-DOMAIN-AUG-MATCHED-LAG-FEEDBACK](docs/DEV_DOMAIN_AUG_MATCHED_LAG_FEEDBACK_TASK_2026-09-21.md)，
状态 READY_FOR_DISPATCH。用户要求先试历史同间隔反馈，再决定接回完整 Agent；
本次只起草任务书，未启动训练。

冻结 D01/D02 各 Q03、Q04（上一包复验名单的编号前缀），共四案、31 份原 F0
已评估候选，保留组合/编辑/条件化。每案原实体上取 h=t−768、t−384，
每 h 仍训练此前 672 小时，分别评价 [h,h+192) 近端和 [h+192,h+384) 延后块，
各四个 48 小时预测起点。显式新增历史读取仅本案原 roster 的 [t−1440,t)；
不能在原 672 点 T 内缩短训练来凑切点。历史 scaler/特征/条件赋值各自重算。

比较当前 C_A、历史近端、历史延后 argmin 和其他实体 Q01/Q02 上已冻结的固定
程序；当前 E 只复用已有缓存，在四案选择冻结后连接。近/远策略共用历史模型，
以区分一般历史平均与延后位置作用。身份 EXPOSED_DEVELOPMENT_REPLAY；
回放当前候选不等于候选在历史切点时已独立提出，不称 fresh 或无偏回测。

主拟合最多 186、接线 6、同配置重试最多 4，总尝试上限 196；LLM/HTTP/SHA/commit
均为 0。数值任务默认全包 2 并行、实测资源允许时最多 3，单协调器与恢复锁沿用。
不改变 Consumer、原增强随机流或完整构造权限；报告新增历史评分、推理和成本，
不得写成零训练/零推理。完成本包即停，下一轮 Agent 学习不自动授权。

#### 5.10.7 历史同间隔反馈包实际收口（2026-09-21 13:34）

报告：[_scratch/dev_domain_aug_matched_lag_feedback/REPORT.md](_scratch/dev_domain_aug_matched_lag_feedback/REPORT.md)；方法与接线 [METHOD.md](_scratch/dev_domain_aug_matched_lag_feedback/METHOD.md)；
机器读数 [result.json](_scratch/dev_domain_aug_matched_lag_feedback/result.json)、[selections.json](_scratch/dev_domain_aug_matched_lag_feedback/selections.json)、[feedback.json](_scratch/dev_domain_aug_matched_lag_feedback/feedback.json)。
身份 `EXPOSED_DEVELOPMENT_REPLAY`，历史实例身份 `historical_calibration`；回放当前候选不等于它们在 h 时已独立提出。入口 `evaluation/main_protocol_p4/batch_research_domain_aug_matched_lag_feedback.py`。

- **方法读数**（pp of None，正 = 前者好，域内两案等权再两域等权，主块 E）：R_HistLate − R_CA **+0.10**（D01 +1.15、D02 −0.95；2 胜 2 负；D01_Q03 +2.49、D02_Q03 +2.28 都是换回 P_NoMixRecipe，D01_Q04 −0.20、D02_Q04 **−4.17** 都是离开预设）；
  R_HistLate − Fixed_dev **−0.73**（0 胜 2 同交付 2 负）；R_HistNear − R_CA +0.85、− Fixed_dev +0.02（1/2/1）；R_HistLate − R_HistNear −0.74（3/4 同交付，D02_Q04 近端选 Edit[-random_conv] +0.26 而延后选 Edit[-shock] −2.72）；R_CA − Fixed_dev −0.83。
  对 None：R_HistNear 14.58 ≈ Fixed_dev 14.57 > R_HistLate 13.84 ≈ R_CA 13.74。判词 **DEFAULT_RECOVERY_ONLY**（也具“混合/变差”特征）。
- **机会与对齐**：池内比 Fixed_dev 好的候选在 3/4 案存在（可得 +1.22 / +4.98 / +1.45），R_HistLate 0 次兑现，R_CA 兑现 1 次；候选成对与 E 同向率 c_a 0.717、hist_near 0.729、hist_late 0.741（差 1–2 对），单 h 只有 0.65–0.67 且逐案摆动到 0.25–0.32；两个 h 常给出相反排序（D02_Q03 条件化候选对预设 h1 +4% / h2 −55%）。
  D02_Q04 中跨 seed、起点、两 h 完全一致的历史延后信号（Edit[-shock] 优于预设）在当前 t 反转（−2.72），跨起点/seed 一致性门槛救不了该案；D01_Q03 是历史反馈修正 C_A 排序的唯一清楚正例。
- **构造与接线**：条件化程序在各 h 的 T-only 表上重解析（D02_Q03 C05 在 h1 变 12/4，D02_Q04 C04 变 10/6 与 11/5），字面阈值保留；当前 t 重编译 12/12 复现旧 assignment；接线 14/14（None×3 seed C_A 与 F0 逐位相同、8 个材料逐字节相同）；smoke 21/21（含 h 之后行投毒测试：历史 scaler/材料/权重不变）；8 实例一致性检查通过（近端起点 0–1 == C_A 逐起点分数）。
- **运行与费用**：192 次拟合（6 接线 + 186 主）全部成功、0 重试、0 缓存、0 LLM/HTTP/SHA/commit；62 个历史物理材料（0 别名）、1116 次新增推理 + 372 次复用；并发按接线实测取 3（单 worker 峰值 406 MB）；历史阶段墙钟 16.7 min，单次拟合无竞争 6.8 s / 3 路竞争 12.7 s；worker 内构造 571 s、评分 11.6 s。新反馈的额外成本 = 当前池 93 次拟合的 2 倍历史训练；近/远共用模型。
  偏差：接线删去“在当前 t 跑历史评分器”一项（会读 C_B/E 原始行），其余无预算/案例/公式偏差。
- **推进边界**：按任务书 §9 第二/四种结果停止；不把“回到预设”写成动态能力，也不据四案两切点宣布历史反馈无信息。唯一可提的下一项（需授权）：把逐 h、逐块的历史配对差作为信息字段而非 argmin 决策交给完整 Fast，有卡/无卡臂共享该信息与预算；本包未起草、未启动。

#### 5.10.8 跨时期域 Skill 学习包实际收口（数值 2026-09-22 02:14；报告 2026-09-23）

任务书：[DEV-DOMAIN-AUG-TEMPORAL-COVERAGE](docs/DEV_DOMAIN_AUG_TEMPORAL_COVERAGE_TASK_2026-09-21.md)（回执 §10）；名单 [DOMAIN_AUG_TEMPORAL_COVERAGE_V1.json](docs/DOMAIN_AUG_TEMPORAL_COVERAGE_V1.json)。报告：[_scratch/dev_domain_aug_temporal_coverage/REPORT.md](_scratch/dev_domain_aug_temporal_coverage/REPORT.md)；
方法与运行期修订 [METHOD.md](_scratch/dev_domain_aug_temporal_coverage/METHOD.md)；机器读数 [result.json](_scratch/dev_domain_aug_temporal_coverage/result.json)、[tables.md](_scratch/dev_domain_aug_temporal_coverage/tables.md)。
身份 `SERIES_DISJOINT_TEMPORAL_DEVELOPMENT`：父包实体组按 12 个等宽时间锚点重排（Source 6 时期 × 每域 12 案、Select 2 时期 × 2 案、MetaTest 4 时期 × 2 组 = 每域 8 案），三角色实体集合互斥、逐案时间屏障通过；不是 Natural Final，MetaTest 带本地 C_A。

- **方法读数**（pp of None，16 案等权）：F_domain − F0 **+2.35**（D01 +4.85、D02 −0.14；A09 +6.07 / A10 +5.53 / A11 +0.21 / A12 −2.38；6/5/5；最大伤害 −4.79）；F_domain − F_shared **−0.04**（3/8/5），F_shared − F0 +2.40；
  F_domain − Random_B4 +1.55、− Menu_CA +2.91、− Fixed_dev **−1.42**、− NoMix +0.94、− None +14.64。各臂对 None：Fixed_dev 16.07 最高。H_deploy：D01 = Fixed_dev、D02 = F_domain（Select J 差 0.012，MetaTest 反转），H_deploy − F0 +2.59、− Fixed_dev −1.18。
- **学到什么**：9 次独立 Slow 全部成卡；被选三卡同核心——P_NoMixRecipe 为 E 先验默认，C_A 仅在 seed 一致且超过 SE 时让步，稀疏条件化 / 单原语 Comp / FixedMixup 不作首选。D01 卡 7/8 案 0 评估直接提交预设，D01 的 +4.85 基本等于“F0 离开预设的代价”；D02 卡 8/8 案评估 1–2 个统一编辑、5 次离开预设，有得（D02_Q_A10_G01 +4.94）有失（D02_Q_A12_G02 −4.44、对 Fixed_dev −12.64）。
  卡臂在 MetaTest 未评估任何条件化方案，构造被软偏好收缩到“预设 + 统一编辑”。共享卡的反 mixup 先验在 D02_Q_A09_G01 造成 −13.49。当前兑现的经验大部分两域通用，分域组织无可测额外价值。
- **线索**：Fixed_dev 首次出现域差异（D01 = P_NoMixRecipe，D02 = Edit[-resample-random_conv]，与相邻编辑的 J 差 < 0.008）；这是开发集选优产物，不算 Slow 学会的领域 Workflow。
- **成本**：部署每案 F_domain 80k token / 0.8 新评估 vs F0 400k / 3.9（−80% / −79%），未扣一次性学习（717 拟合、439 请求、19.87M token）。全包拟合 1383 / 1988、逻辑请求 720 / 1618、HTTP 734 / 1626；已知 30.91M token，16 次未知 usage 全部接受（1 次用户显式、15 次瞬断由用户常设决定下的 operator_loop 逐次接受），保守上界约 32.9M；付费时钟 8 h 03 min（含约 98 min 停机等待）；数值 2 路、HTTP 峰值 4。
- **偏差**：主机内存 kill（4 条 Source 轨迹整条重起、2 拟合丢失）；传输额外尝试 14 次超过规划 8 次预留（预留被折进阶段 HTTP 帽，未单设计数门；每请求至多一次重试保持）；census 高于规划目标、低于硬上限；文本检查 “may” 误判（约 0.85M prompt token）；8 h 预警未单独写出（报告补记）；package_status 残留暂停字段已在收口时归档。无泄漏、无计分失效、无按结果换实体/日期/卡片。
- **推进边界**：按任务书 §9 停止。不能再说“学习指导完全没效果”，也不能宣布 per-domain Skill 成功；尚缺领域专用经验相对通用经验的独立价值，以及动态研究超过固定程序的收益。后续应围绕已出现的具体领域差异设计，而不是只换 Slow 提示词或重复同种选卡；下一包未起草、未启动，由用户 / Planner 决定。

### 5.11 离线学习 Workflow、零反馈部署（2026-09-23，用户确认的新设计）

用户转述 Planner 安排并确认执行：后续按“离线学习 Workflow，部署时依据数据执行”设计。Skill 固定下来，但生成的处理方案可以随新数据变化。设计参考 Eval-Skill（arXiv 2606.07040：每域静态 Skill、离线探索与多例选优、测试时无反馈）；本项目的迁移方式不称原论文复现。

- **目标**：检验离线学到的域经验，能否帮助 Agent 在新案例上构造更有效的增强。
- **开发（Source）**：充分探索。允许比较方案、训练和读取反馈；成功、失败和反例都保留，供 Slow 学习。
- **选卡（Select）**：模拟真实部署。候选卡面对新开发案例时，只能看合法历史和材料诊断；提交后由外部评分选优。增加案例覆盖，不再只用每域两例选卡。
- **测试（Test）**：零下游反馈。Fast 可以观察、构造组合、检查材料，然后提交；之后才训练最终 Consumer、统一评分。当前案例的公共基线 C_A 分数也不提供。
- **构造空间**：保留完整空间。原语组合、条件化处理和 NoMix 编辑都可用，不强制从 NoMix 开始。
- **主比较**：相同工具和预算下的无卡、共享卡、域卡。NoMix 和开发选出的固定方案保留作参照。
- **推进顺序**：先完成任务族与案例划分，再冻结实验规模；这一段不启动付费运行。最值得花力气的是找出结构可观察、处理差异能重复出现的任务族，让 Slow 有可学的经验。已有 Harness 接线继续复用。

#### 5.11.1 任务族筛选收口（2026-09-23 14:36，0 LLM）

报告：[_scratch/dev_aug_task_family_screen/REPORT.md](_scratch/dev_aug_task_family_screen/REPORT.md)；机器读数见同目录 plan.json（拟合前冻结）、result.json、tables.md、associations.json、family_set_options.json；划分文档：[docs/AUG_TASK_FAMILY_DIVISION_V1.json](docs/AUG_TASK_FAMILY_DIVISION_V1.json)（五族，状态 PROPOSED_BY_SCREEN_NOT_FROZEN）。入口 `evaluation/main_protocol_p4/batch_research_aug_task_family_screen.py`。

- **设置**：D01 电力、D02 交通，新增 D03 太阳能、D04 空气质量（KDD 无缺失版，按非零平坦段过滤）、D05 风电（分钟 → 小时）。每域锚点表 Source < Select < Test，相邻角色间隔 ≥ 1056 h；实体池拟合前固定，D01/D02 的 Select/Test 只用从未使用过的实体。只用 Source 池：每域按结构第一主成分切 4 组 × 2 锚点（D04 为 3 组），12 个统一程序 × 3 seed，E 由评估器直接评分。
- **读数（pp of None）**：
  - Preset 对 None：电力 +16.2、交通 −6.2、太阳能 +28.9、空气质量 +28.5、风电 +36.3。
  - 族最优：电力 Comp[censor] +18.9（对 Preset +2.7，6/8）；交通 Comp[censor] +0.9（对 Preset +7.1，**8/8**）；太阳能与空气质量为 Preset；风电 Edit[-calendar] +36.5（≈ Preset）。
  - 域对处理差值的方差解释 11%–67%；族内结构—处理关联 0 对过族错误率门槛。
  - 案例指纹留一 30/38，电力与交通互相混淆 4 例。
  - C_A 最优等于 E 最优 13/38。
- **规则判定**：携带 {D01, D03}，满足 F2 的携带族 0，按规则不建议实验。交通因 F1 措辞（要求族最优比 None 好 ≥ 3 pp）未被携带；这一缺口在最终结果前已向用户提示，规则未事后修改。
- **含义**：
  - 卡 vs 无卡：零反馈下“是否该强增强”影响 +16 ~ +36 pp（交通为反向风险）。
  - 域卡 vs 共享卡：固定程序层面的族级选择差距约 1.4–2.9 pp（按族集合）；三域 {D01, D02, D03} 相对统一最佳 Comp[shock]，为电力 +0.15、交通 +4.33、太阳能 +4.12。统一最佳与 Preset 等四个程序只差 0.4 pp 以内。这个数不是实测效果，也不是 Workflow 上限（09-23 复核更正措辞）。
- **规模估算**：{D01, D02, D03} lean 约 1440 拟合、25–68M token；full 约 2016 拟合、39–91M token。其中零反馈轨迹 token 是假设，冻结前宜先实测。
- **运行**：1368 拟合 0 失败；拟合到 500 次时，被 Claude Code 在系统内存告急时回收，用户释放内存后 4 路从断点续跑，已完成单元格按绑定跳过，读数无影响。`entity_case.DOMAIN_INDEX` 新增 D03–D05（加法）；构建进程 torch 单线程。
- **复核意见（2026-09-23，用户转来，未批准为任务书）**：
  - 继续做电力、交通、太阳能三域学习实验，旧判词保留，下一实验的准入理由另定：交通属于“增强收益小、选错损失大”的情境，按已知域加载卡，不需要画像路由门。
  - 复用筛选表作为脚本实验记录交给 Slow，辅以少量真实 Fast 轨迹；主比较为零反馈的无卡、共享卡、域卡，保留选卡覆盖。
  - 下一包开头先实测零反馈轨迹成本，再冻结支出；不人工写入各域答案。
- **待决定**：按规则停止，或前瞻性修订携带集合（Select/Test 池未触碰）；并冻结 Source F0 案例数、候选卡数、选卡目标 J 与零反馈轨迹预算。另一个可选项：把 zero_fraction 等结构字段加入 Fast 可见 overview（所有臂共享）。

#### 5.11.2 DEV-AUG-OFFLINE-SKILL 阶段 A 检查点（2026-09-23 16:55）

任务书：[DEV-AUG-OFFLINE-SKILL](docs/DEV_AUG_OFFLINE_SKILL_TASK_2026-09-23.md)（Planner 定稿，用户转来直接执行）。报告：[_scratch/dev_aug_offline_skill/REPORT_STAGE_A.md](_scratch/dev_aug_offline_skill/REPORT_STAGE_A.md)。入口 `evaluation/main_protocol_p4/batch_research_aug_offline_skill.py`，零反馈循环 `methods/ttha/batch_zero_feedback.py`（新路径；旧 `run_job` 不改）。身份 `DEVELOPMENT_ZERO_FEEDBACK_REUSE`，不是 Natural Final。

- **实现**：
  - Fast 没有 evaluate / compare，请求和工具输出都经过“下游分数通道”检查；commit 冻结未拟合的方案（COMMITTED_UNFITTED），全部冻结后才由外部评估器拟合、评分。
  - 请求改为紧凑案例卡加批级分位数，逐实体表按需获取；`zero_fraction`、`flat_nonzero_fraction`、`weekly_excess_r2` 按筛选原公式接入，并真实参与编译。
  - 每条轨迹上限：6 请求 / 18 工具 / 4 材料 / 20 万 token，最后一次请求只开放 commit。
  - 缓存：按赋值键复用筛选材料和三 seed E，筛选目录只读。
  - 接线 7/7（重训筛选材料的 C_A / E 与缓存逐位相同），smoke 20/20。
- **读数**（pp of None，域内 8 例等权）：

  | | 电力 | 交通 | 太阳能 | 三域等权 |
  |---|---:|---:|---:|---:|
  | F0 − None | | | | +0.04 |
  | F0 − NoMix | −17.22（0/0/8） | +5.88（4/0/4） | −27.38（0/0/8） | −12.91 |
  | F0 − Fixed_source | | | | −16.18 |
  | F0 − Comp[shock] | | | | −13.31 |
  | F0 − 菜单事后最优 | | | | −19.03 |

  固定方案与事后最优都在同一批 Source 案例上选出，属样本内比较。
- **行为**：
  - 0/24 提交 NoMix。提交为单独 amplitude 7、单独 calendar 6、温和条件化方案 5、None 3，其余 3 例。
  - 20/24 构造过 NoMix 编辑版，看过最大改动窗口后以“破坏日形态 / 抬高夜间零值”为由放弃。
  - 这是一个与域相关的系统性偏差：电力、太阳能上错失强增强，交通上温和偏好反而占优。离线经验有明确可学空间，但卡能否改变零反馈决策尚未实测。
- **用量**：
  - 135 逻辑请求 = 135 HTTP，0 额外传输，0 未知用量。
  - 3.43M token（上限 5M）；每条轨迹均值 142.7k、P90 161.6k、最大 171k，约 5.6 请求 / 15.8 工具 / 3.6 材料。
  - 拟合 42 次（提交 39 + 接线 3），0 失败；缓存命中 11/24 案。
  - 付费墙钟 49.5 min。
  - 首次启动因目录创建顺序缺陷全部报错，0 付费、0 拟合，已修复重启。
- **待决定**：B–D 预算。点估计约 34.5M、保守约 49.4M，拟定上限 57M token / 1,312 请求 / 1,258 拟合。另一可选项：压缩 inspect_material 窗口与 inspect_data segment 的输出，预计每条降 20–30%，B–D 所有臂一致适用。确认后 B→C→D→报告连续执行。

#### 5.11.3 DEV-AUG-OFFLINE-SKILL 全包收口（2026-09-23 21:02）

报告：[_scratch/dev_aug_offline_skill/REPORT.md](_scratch/dev_aug_offline_skill/REPORT.md)；方法 [METHOD.md](_scratch/dev_aug_offline_skill/METHOD.md)；机器读数 [result.json](_scratch/dev_aug_offline_skill/result.json)、[tables.md](_scratch/dev_aug_offline_skill/tables.md)；卡片 `slow/`、`freeze/cards_frozen.json`。身份 `DEVELOPMENT_ZERO_FEEDBACK_REUSE`。预算由用户 / Planner 在阶段 A 后冻结（57M token / 1312 请求 / 1258 拟合），保留工具输出不压缩。

- **形成与选卡**：8 次 Slow（1 次契约纠错）全部成卡，共同核心为“材料外观不是下游效用”加按观察触发的统一默认：太阳能夜间零值 → NoMix；电力零值为 0、lag24 高 → shock / censor；交通按字段分档 → Edit[-random_conv] / FixedMixup / censor；共享卡 s2 默认 NoMix，weekly 与极值双高 → None。Select 96 条轨迹，J 选出 D01 d1、D02 d2、D03 d2（四卡 J 相同，按 token 破同）、共享 s2。
- **测试读数**（40 例 × 3 臂，pp of None，域内等权再三域等权，两向聚类重采样）：
  - F_domain − F0 **+17.66** [+11.5, +24.8]（电力 +17.4 / 交通 +2.8 / 太阳能 +32.8；32/0/8；留一时期 +15.5–+19.3；seed +15.2–+19.7）。
  - F_shared − F0 +16.29 [+9.3, +24.2]（交通 −2.1；29/0/11）。
  - F_domain − F_shared +1.37 [−1.2, +4.5]（19/8/13；交通 +4.9：共享卡的 None 规则在 11/16 交通案触发，而 Test 期交通 NoMix 对 None 为 +9.9，Source 期为 −6.2）。
  - 对 None：F0 +4.47、F_shared +20.76、F_domain +22.13、NoMix +23.14、Fixed_source +21.42。
- **归因**：卡臂只交付统一程序（域卡 shock 16 / Edit[-random_conv] 9 / NoMix 8 / censor 6 / FixedMixup 1；共享卡 NoMix 29 / None 11），没有条件化方案。F_domain − NoMix −1.01、− Fixed_source +0.71，即补回了 F0 与 NoMix 差距的 94.6%。正结果是“离线经验纠正零反馈 Agent 的系统性偏差、回到固定程序水平”，不是超过固定程序的动态构造。
- **成本**：
  - 部署每条轨迹：F0 145.7k、F_domain 61.4k（−58%）、F_shared 22.7k（−84%）token。
  - 一次性学习：A 3.43M + Slow 0.87M + Select 4.14M = 8.44M token、264 拟合，另复用筛选 1368 拟合。
  - Test：9.34M token、605 拟合。
  - B–D 合计 14.35M / 57M token、770 请求、827 拟合（0 失败）；全包拟合 869 / 1300；4 次瞬断均按接受类别自动接受，各有 incident。
- **偏差**：
  - 两次启动接线缺陷，均 0 付费。
  - Test 33/120 时控制器被 Claude Code 因宿主内存回收；用户确认后续跑，4 条在途轨迹整条重开，丢失 2 次在途拟合。
  - 阶段 A 的机器预算记录 1261 已更正为 1258。
- **推进边界**：按任务书完成 D 段和报告后停止，不自动追加修订轮、画像路由或新域。可考虑的下一项（交用户 / Planner）：在程序排序真正随案例 / 时期翻转的族上检验超过固定程序的动态能力；卡规则对时期漂移敏感（交通），是具体的研究对象。

#### 5.11.4 朴素指导卡对照收口（DEV-AUG-OFFLINE-SKILL-NAIVE-CONTROL，2026-09-23 晚）

报告：[_scratch/dev_aug_offline_skill/naive/REPORT_NAIVE.md](_scratch/dev_aug_offline_skill/naive/REPORT_NAIVE.md)。身份：已曝光 Test 上的事后机制消融；原主实验与冻结卡不变。

- **设置**：同一 Slow、每域一张朴素卡，只给任务、Consumer、算子说明与 Source T 观察，通用原则与学习卡相同；一次生成加一次格式纠错，不选优；在 40 个 Test 案例上零反馈运行，预算不变。
- **读数**（pp of None，三域等权，聚类 95% 区间）：
  - 域卡 − 朴素卡 **+6.44** [+1.5, +11.1]：电力 −1.9、交通 −0.5、太阳能 **+21.8**；25/0/15。
  - 朴素卡 − 无卡 +11.22 [+5.5, +16.9]。
  - 共享卡 − 朴素卡 +5.07 [−0.5, +10.0]。
  - 朴素卡对 None +15.69，其中电力 +23.5，高于域卡。
- **改变的判断**：朴素卡凭任务知识写出“夜间零值序列跳过 regime / shock / resample / conv”，太阳能交付温和方案；学习卡依据 Source 证据直接提交 NoMix。
- **部署成本**：朴素卡每条轨迹 143k token（≈ 无卡），域卡 61k（−57%）。
- **运行**：5.87M 已知 token / 9M、114 / 126 拟合；2 例 INCOMPLETE；另有 4 次停机在途请求用量未记。一次仪器纠正（卡片文本检查把算子参数名 `t0` 误判为实体编号，停机后豁免 `t0`，按统一规则用已返回的回复重新校验，不新发调用）。

#### 5.11.5 主实验补齐收口（DEV-AUG-MAIN-COMPARISON，2026-09-24 凌晨）

任务书：[DEV-AUG-MAIN-COMPARISON](docs/DEV_AUG_MAIN_COMPARISON_TASK_2026-09-23.md)（回执在文末）。报告：[_scratch/dev_aug_main_comparison/REPORT.md](_scratch/dev_aug_main_comparison/REPORT.md)；方法 [METHOD_CORE.md](_scratch/dev_aug_main_comparison/METHOD_CORE.md)；主表 [main_table.md](_scratch/dev_aug_main_comparison/main_table.md)；摘要事实 [ABSTRACT_FACTS.md](_scratch/dev_aug_main_comparison/ABSTRACT_FACTS.md)。0 LLM。

- **主表**（40 个 Test 案例，G = pp of None，三域等权）：NoMix +23.1、F_domain +22.1、Fixed_source +21.4、Fixed_global +21.2、F_shared +20.8、F_naive +15.7、F0 +4.5、None 0、AutoDA −2.7。
- **成对差**：F_domain − F0 +17.7 [+11.5, +24.8]；− F_naive +6.4 [+1.5, +11.1]；− F_shared +1.4 [−1.2, +4.5]；− NoMix −1.0；− AutoDA +24.8 [+14.6, +33.2]。
- **AutoDA-Timeseries**：官方模块原样调用，受控协议适配（同一 MLP、2000 步 AdamW、Catch22 特征、只增强输入）。三个登记配置在 Source 上都劣于不增强，选中 C3；Test 上对 None −2.7 [−5.5, +0.7]。机制读数：选择分布几乎不动，样本几乎总被增强。这不是官方 benchmark 复现，不能写成“AutoDA 无效”。
- **成本**：部署 token 为 F0 146k、F_naive 143k、F_domain 61k、F_shared 23k；AutoDA 0 token，但每次联合训练 102 s，约为普通拟合的 8.2 倍。拟合 151 / 180。
- **不能写**：域卡可靠胜过共享卡；超过 NoMix 或固定程序；条件化构造带来收益。经验来源消融与第二 Consumer 后置。

#### 5.11.6 第二 Consumer 与 TSFM 准备（2026-09-24，用户授权）

用户明确要求“今晚跑第二 consumer，同时准备上 TSFM 的方案，后续再开放作用域”。本次授权覆盖 PatchTST 的本机真实训练：冻结 OFFLINE-SKILL 的卡与已交付材料，沿用三域/40 Test/三 seed，检验跨 Consumer 材料迁移；不重跑 Fast/Slow、不称模型条件化学习。该授权更新顶部历史“模型切换未授权”在此有限范围的状态，Natural Final 继续关闭。

执行任务书：`docs/DEV_AUG_PATCHTST_TRANSFER_TASK_2026-09-24.md`；输出 `_scratch/dev_aug_patchtst_transfer/`。TSFM 只准备 `docs/TSFM_AUGMENTATION_EXTENSION_PLAN_2026-09-24.md`，不启动 TSFM 训练。PatchTST 已于 09-24 01:21 本机启动（Windows 控制器初始 PID 50788；两次 Source 控制台/未知中断后修复无窗口启动，恢复 PID 41796），三次真实接线通过；主机内存约2.9GB，采用单路训练。

**收口（09-24 05:39）**：582 次成功拟合，[REPORT.md](_scratch/dev_aug_patchtst_transfer/REPORT.md)。冻结的 MLP 派生交付在 PatchTST 上：域卡方案对无卡 −3.96 pp（区间跨零）、对朴素卡 −7.88、对 NoMix −3.63；退步几乎全来自电力（域卡 16/16 交付 C_shock，−20.2 pp），太阳能 NoMix 仍 +4.7。PatchTST 不增强（nMSE 0.307；电力 0.350 / 交通 0.330 / 太阳能 0.241）已胜过 MLP 上所有臂，NoMix 从 MLP 的 +23.1 降到 −1.7：换强 Consumer 后增强空间缩小、方案排序改变。含义限于冻结方案迁移，不代表针对 PatchTST 重新学习后的结果。

#### 5.11.7 PatchTST 条件化离线学卡（DEV-AUG-PATCHTST-OFFLINE-SKILL，2026-09-24 启动）

任务书与执行前盘点：[DEV-AUG-PATCHTST-OFFLINE-SKILL](docs/DEV_AUG_PATCHTST_OFFLINE_SKILL_TASK_2026-09-24.md)（Planner 安排，用户确认预算与四项决定）。入口 `evaluation/main_protocol_p4/batch_research_aug_patchtst_offline_skill.py`，输出 `_scratch/dev_aug_patchtst_offline_skill/`。身份 `EXPOSED_DEVELOPMENT_REVALIDATION`：Test 40 例已曝光，只是开发复验；可用的未评分实体只有交通足够（电力 13、太阳能 9，不足 16），用户决定不加独立验证。

- **设计**：Consumer 冻结为迁移包的 PatchTST（lr 1e-4、500 步），Fast/Slow/朴素卡看到 PatchTST 描述，其余零反馈协议不变。Source 24 例 × 12 固定方案 + 24 条无卡轨迹均在 PatchTST 上训练评分；Slow 每域 2 张卡，只读 PatchTST 证据；原 Select 选卡；Test 四臂（新卡 / 原 MLP 卡 / 无卡 / PatchTST 朴素卡）同预算，参照 None、NoMix、PatchTST Source 固定方案；迁移结果单列。
- **预算**（用户确认）：36M token、1450 逻辑请求、1800 拟合、24 h；API 12 路并行；GPU 按内存 1–2 路。
- **启动前**：smoke 11/11；接线 PASS（同进程第二次拟合与迁移包模型逐位相同，E 相同；单拟合 22 s，进程峰值 1.23 GB）。09-24 11:42 启动，控制器 PID 10144。实际进度以 `status.json` 为准，本节在收口时补读数。
- **收口（09-24 16:10 COMPLETE）**：报告 [_scratch/dev_aug_patchtst_offline_skill/REPORT.md](_scratch/dev_aug_patchtst_offline_skill/REPORT.md)。
  - Test（pp of PatchTST None，三域等权）：无卡 +0.38、朴素卡 −0.32、新卡 −2.09、NoMix −1.68、Source 固定方案 −3.40、原 MLP 卡 −5.31。
  - 新卡 − 原 MLP 卡 **+3.22** [−1.10, +8.36]（24/6/10；电力 +13.4 来自去掉 shock，太阳能 −4.45）；新卡 − 无卡 −2.47 [−11.65, +3.42]（电力 Q4 一例 −93.9 主导，NoMix 同案 −90.8；Source 最优电力编辑方案 +5.6 在 Test 为 −11.2，时期反转）；新卡 − Source 固定方案 +1.31（27/40 同交付）。
  - 含义：方法能按 Consumer 重新学出不同规则并挽回迁移损失；但 PatchTST 上增强空间小且随时期反转，没有超过无卡或不增强。部署 token 新卡 65k vs 无卡 145k。22.50M token、1392 拟合尝试、付费 4.38 h。
- **运行记录**：13:28 左右控制器被 Claude Code 内存回收（会话空闲、系统内存告急），当时无在途 LLM 请求，在途 72 次拟合中 34 次已写完、38 次重训；用户 14:15 在独立窗口按原命令续跑（PID 23232）。现场记录 `incidents/controller_reaped_1.json`；续跑后批次编号从 1 重排，覆盖了前一次运行的批次说明文件，拟合记录本身完整。
- **BN 机制诊断（09-24 晚，Planner 安排）**：[DEV-AUG-PATCHTST-BN-DIAGNOSTIC](docs/DEV_AUG_PATCHTST_BN_DIAGNOSTIC_TASK_2026-09-24.md)。电力 16 例 × 3 seed 的 None / NoMix / Edit 已有模型，冻结学习参数，只用 T 父窗重估 BN 运行统计（0 LLM、0 优化器更新；原评分逐位复现，参数逐位不变）。**BN 运行统计不是负收益的原因**：相对各自 None，NoMix −9.58 → −10.67，Edit −11.19 → −12.62；Q1–Q3 变化约 ±1 pp，Q4 全体（包括 None 自身）更差；Q4_G1 第二起点的负偏差（增强 −1.30 / −1.37 对 None −0.52）重估后没有缩小；系统读数 −6.88 → −7.76。两次前向对训练过程本身的影响未检验（需要重训）。原包结果的稳健性补充：新卡 − 无卡在均值 −2.47、中位数 +1.27、截尾均值（每侧 10%）+0.53、去掉 Q4_G1 −0.54；原全样本主结果保留。

#### 5.11.8 TSFM 条件化学卡与服务器迁移（2026-09-24 起草；用户确认）

- **TSFM 包**：[DEV-AUG-TSFM-OFFLINE-SKILL](docs/DEV_AUG_TSFM_OFFLINE_SKILL_TASK_2026-09-24.md)，入口 `evaluation/main_protocol_p4/batch_research_aug_tsfm_offline_skill.py`。设计与 §5.11.7 同构，Consumer 为 Time-MoE-50M，每个方案都从同一预训练权重微调；先在 Source 上用不增强校准 lr × 每步父窗数 × 步数；不保存模型，训练后用 E 输入窗冻结预测，屏障之后再评分；付费前自动接线（展开与官方 `generate` 逐位一致、同进程重复逐位一致、子视图生效、双卡逐位一致）。在服务器 1–2 × RTX 5880 上由用户运行；Windows smoke 8/8，GPU 接线待本机空闲后实测。
- **修订 1（用户选 B，09-24 晚）**：第一版（48 步展开 MSE、实体 T scaler、β2 0.999）在服务器校准时 12 个微调配置全部不如零样本（0.1985 对 0.1959），已停。改为 Time-MoE 原生微调：官方逐位置四头 Huber 损失 + 0.02 路由辅助损失、每个窗口按自身 192 点输入标准化（训练与服务一致）、AdamW β2 0.95；校准网格 lr {1e-5, 5e-5, 1e-4} × 每步父窗 {8, 32} × 步数 {25, 50, 100, 200}。输出目录 `_scratch/dev_aug_tsfm_native_offline_skill/`，与第一版分开。新增 `--selftest`（不调 LLM），`server/run_tsfm.sh` 启动付费阶段前自动运行。GPU 调度改为按每批任务估算显存（本机实测：不增强 / 增强，每步 8 窗 3.9 / 5.9 GB，每步 32 窗 9.9 / 约 16 GB）选卡，最近 90 秒内已派发但尚未占用显存的任务也计入。
- **修订 2（09-24 晚）**：服务器原生版校准仍不如零样本（最优微调 0.2227 对零样本 0.206，最优点在网格最弱一角）；32 路并发下中转 152 次尝试 44 次瞬断，6/24 条 Source 无卡轨迹被打断，按原代码会被当作不增强流入证据，已停机。修改：调用失败打断的轨迹在补跑后仍未完成时本阶段停止；传输上限可上调并记修订（token / 请求 / 拟合 / 墙钟上限不变）；并发降为 8；`SEH_STOP_AFTER=source` 在 Source 屏障后停止，并补做 24 个 Source 案例的零样本核对（只供操作者决定，不进入任何证据）。本机测试 8/8，更新包 `update_v6.tar`。之后选 A（续跑全包）还是 B（收口），由 Source 零样本核对结果决定。
- **修订 3（09-24 晚，只改传输）**：v6 的只跑 Source 续跑因中转故障停下（24 条完成 20 条，4 条被打断且未混入证据；18:16 起上游中断约 7 分钟，20:14 起中转超时被误调为 120 s，已改回）。改为 stream=true 并带 include_usage；流中途断开或读超时统一判为瞬断；每个请求仍最多重试一次，重试前等 90 s；阶段末尾有被打断的轨迹时先等 600 s 再补跑一轮。已实测两个中转的流式响应都带 usage，本机测试 6/6（含一次真实调用），服务器模拟测试 5/5；已直接部署到服务器，原文件备份为 `.bak_v6`。
- **Source 零样本核对（09-24 21:25，STOPPED_AT_SOURCE）**：流式请求 9/9 成功，补完被打断的轨迹。24 个 Source 案例上，12 个方案微调后全部不如 Time-MoE 零样本（三域 pp：最好 random_conv −29.7，不增强 −40.1，最差 amplitude −48.5；每个方案至多胜 6/24；逐案事后取最好仍为 −18 / −14 / −49）。相对微调后的不增强，增强只挽回 +0.6 ~ +5.9。Time-MoE 零样本是四个 Consumer 中绝对误差最低的（电力 0.167，PatchTST 0.222）。校准看的 C_A+C_B 只差 8%，到 E 块差到 40%，离 T 越远，微调损伤越大。全包至此 184 次请求、3.5M token、962 次拟合；是否续跑（A）或收口（B）待用户 / Planner 决定。
- **TSFM 切入点（09-24 晚，不调 LLM；任务书 §8–9）**：
  - 零样本与全参数微调预测的混合：所有权重、所有方案都为负，排除。
  - 冻结主干、只训练预测头：损伤从 −40 缩到约 −3；增强再挽回约 1 pp，但无一方案超过零样本。
  - **推理上下文准备**：零样本改用更长的历史上下文（Time-MoE 最长支持 4096 点）。Source 上 672 点 +25.9 pp（23/24）；按 Source 冻结的逐域选择（电力 2688 点、交通和太阳能 1344 点，均用 T scaler 标准化），在 40 个 Test 案例上 **+36.1 pp（37 胜 3 负）**，三域 Test nMSE 从 0.247 降到 0.163（PatchTST 不增强 0.307，MLP 用 NoMix 0.330；输入长度不同，只作量级参照）。
  - 这是 TSFM 上唯一大幅且可迁移的正向杠杆，属于“为 Consumer 准备推理输入”，不训练模型。
- **服务器迁移包**：`C:\Users\辉\desktop\agent\server_export\`（仓库外，由 `make_export.py` 从本仓库只读生成）。代码副本只改可移植性（读 JSON 时映射 Windows 绝对路径、`SEH_*` 环境变量控制数据根目录 / LLM 地址 `https://cpa.cpa-lab.me/v1` / 并行）。数据打包在 `data_pack/`。服务器上的拟合换了硬件，不能和 Windows 缓存的拟合放进同一张表。

#### 5.11.9 第三 Consumer：DLinear 的 Source 筛查（DEV-AUG-DLINEAR-SOURCE，2026-09-24）

任务书 [DEV-AUG-DLINEAR-SOURCE](docs/DEV_AUG_DLINEAR_SOURCE_TASK_2026-09-24.md)（Planner 安排，用户要求在本机跑）。入口 `evaluation/main_protocol_p4/batch_research_aug_dlinear_source.py`，输出 `_scratch/dev_aug_dlinear_source/`。Consumer 为官方 DLinear（cure-lab/LTSF-Linear 0c11366），训练循环与 PatchTST 包相同，在 CPU 上训练；不调 LLM，只看 Source，不学卡，不进入 Test。

- **校准**：只用不增强，6 个 Source 案例，看 C_A+C_B；选中 lr 1e-3、2000 步。
- **24 案 × 4 方案 × 3 seed**（pp of DLinear None，三域等权）：NoMix +7.98（电力 +17.9 / 交通 −12.1 / 太阳能 +18.1）、censor +11.03（+18.6 / +0.0 / +14.5）、shock +10.23（+18.2 / −5.0 / +17.5）；收益集中在 A1 时期，A2 很小或为负。
- **跨 Consumer**（同 24 案、同 4 方案，已有读数）：不增强的 E 由好到差为 PatchTST < DLinear < MLP，NoMix 收益依次为 +0.66 / +7.98 / +12.95；shock 在 MLP 上 +13.4，在 PatchTST 上 −3.8。Consumer 越弱，增强空间越大，方案排序随 Consumer 变化。
- **成本**：拟合 315 次；受空闲内存限制只开 1 个 CPU 进程，墙钟 47 min。按任务书停止。

#### 5.11.10 DLinear 条件化离线学卡（DEV-AUG-DLINEAR-OFFLINE-SKILL，2026-09-24/25，本机）

任务书 [DEV-AUG-DLINEAR-OFFLINE-SKILL](docs/DEV_AUG_DLINEAR_OFFLINE_SKILL_TASK_2026-09-24.md)，报告 [_scratch/dev_aug_dlinear_offline_skill/REPORT.md](_scratch/dev_aug_dlinear_offline_skill/REPORT.md)。与 §5.11.7 同构，Consumer 为官方 DLinear（CPU），复用 DLinear Source 包的 288 次拟合；传输为流式，被打断的轨迹不得当作“不增强”流入结果。

- **Test**（40 例，DLinear None 的 pp）：新卡 +8.64、原 MLP 卡 +8.80、朴素卡 +8.88、无卡 +1.11；NoMix +8.49，Source 固定方案 +7.64。
- **新卡 − 无卡 +7.53，区间 [+0.58, +14.23]**（26/1/13，去掉任一时期仍为正）；新卡与原 MLP 卡、朴素卡之间无可分辨差别（−0.16、−0.23，区间都跨零）。卡把无卡 Agent 从“条件化小改动或不增强”纠正回统一固定程序的水平（电力 16/16 为 Source 固定方案）。
- **跨 Consumer**：卡相对无卡的收益随增强空间单调变化，依次是 MLP +17.66、DLinear +7.53、PatchTST −2.47（NoMix 对 None 分别为 +23.1 / +8.5 / −1.7）。
- **成本**：22.2M token、1002 次请求、1441 次拟合（0 失败）；本机网络切换导致 10 条轨迹被打断，阶段末全部续跑完成。

#### 5.11.11 TSFM 上下文准备卡（DEV-TSFM-CONTEXT-CARD，2026-09-24/25，本机，评分查缓存）

任务书 [DEV-TSFM-CONTEXT-CARD](docs/DEV_TSFM_CONTEXT_CARD_TASK_2026-09-24.md)，报告 [_scratch/dev_tsfm_context_card/REPORT.md](_scratch/dev_tsfm_context_card/REPORT.md)。TSFM 转向“推理上下文准备”，微调线收口（§5.11.8 修订 4–5）。

- **设置**：Time-MoE-50M 零样本。零反馈 Agent 为每个案例选择上下文长度 × 标准化方式，共 10 种组合，评分查缓存的零样本上下文筛查。所有臂的历史访问范围、候选动作和 Consumer 说明完全相同。
- **Test**（40 例，以 192 点、逐窗口标准化为基线的 pp）：
  - 各臂：无卡 +31.30，共享卡 +35.95，**域卡 +37.67**；参照中统一 672 点 +34.43，逐域固定选择 +36.07。
  - **域卡 − 无卡 +6.37 [+2.49, +12.85]（35/2/3）**；**域卡 − 统一 672 点 +3.25 [+1.10, +5.78]**；域卡 − 逐域固定选择 +1.60（区间跨零）；域卡 − 共享卡 +1.72（区间刚好跨零）。
  - 域卡每条轨迹的 token 比无卡少约 33%。
- **无卡的偏差**：只用逐窗口标准化，在太阳能等场景会截短到 336 点。卡学到的是交通取最长上下文并用 T scaler 标准化、太阳能不截短、电力在最近几周与 T 同分布时加长。
- **意义**：本项目第一次出现学到的卡明显超过 Source 选出的统一固定方案。方法可以统一表述为：离线学习 Consumer 所需的数据准备经验，部署时对弱 Consumer 生成训练材料，对强预训练 Consumer 准备推理上下文。
- **身份**：开发复验。各阶段的上下文读数都已曝光，评分是服务器数值。
- **补充实验（09-25 凌晨，用户批准；任务书 §6）**：
  - **朴素卡对照**：只凭常识写的卡建议短上下文，Test 上 35/40 选了 336 点，得分 +17.5。域卡 − 朴素卡 **+20.15 [+11.8, +29.1]**，朴素卡 − 无卡 −13.8。
  - **新起点留出**：40 个从未评分过的预测起点，全部冻结部署。域卡 − 无卡 **+3.94 [+0.15, +8.17]**；域卡 − 朴素卡 **+20.11 [+10.8, +29.5]**；token 少约 35%。域卡 − 统一 672 点为 +1.15 [−1.0, +3.1]，没有复现 Test 上的 +3.25。
  - **可写的结论**：离线经验（不是常识）让零反馈 Agent 为 TSFM 准备上下文更可靠，且与 Source 选出的最佳固定上下文相当。不写“超过固定方案”。
- **摘要草稿**：`docs/ABSTRACT_DRAFT_VLDB_2026-09-25.md`（v2，暂不提 PatchTST）。

### 5.12 分类增强：跨任务验证的接线与盘点（DEV-CLS-AUG-WIRING，2026-09-25）

任务书 [DEV-CLS-AUG-WIRING](docs/DEV_CLS_AUG_WIRING_TASK_2026-09-25.md)（Planner 意见经用户转来；用户追加“验证过了到服务器上跑，需要多路的并行开起来验证”）；报告 [_scratch/dev_cls_aug_wiring/REPORT.md](_scratch/dev_cls_aug_wiring/REPORT.md)。定位：同一套离线学习—零反馈部署流程能否迁到分类任务的**接线与耗时预检**；0 LLM、只读 TRAIN、不学卡、不终评。旧分类线（修复算子、RidgeClassifier）的结论不外推为“分类增强无空间”。

- **已定设计**（Planner）：域 = UCR 数据集族，Source / Select / Test 按数据集分开（目标 2 族 × 4/2/6），同源放同侧；TRAIN ≥ 100 且每类 ≥ 30，Source/Select 的 TRAIN 按类 2:1 分拟合/反馈；逐序列 z 标准化 → 训练批次内按概率增强 → FCN，不再重标准化；干净与增强样本同批前向，不增加步数；Source 筛查投入规则为至少两个非同源 Source 数据集上 |ΔMacro-F1| ≥ 1 pp 且三 seed 同向（本包不执行）。
- **实现**：`methods/ttha/cls_aug_data.py`（numpy：TRAIN 解析、z 标准化、划分、指标）、`methods/ttha/cls_aug.py`（FCN 的 Torch 重实现，按 dl-4-tsc 与 Keras 语义对齐；AutoDA 官方算子封装）、`evaluation/main_protocol_p4/cls_aug_wiring.py`。
- **读数**：本机 6 次、服务器 24 次完整拟合（2000 epoch）全部通过技术检查；服务器同 seed 跨卡逐位相同，与本机不逐位相同（正式拟合只在服务器跑）。服务器 4、5 号卡上不增强单次约 1–5 GPU 分钟（ElectricDevices / Crop 最重），增强臂 2–3 倍；单卡 8–12 路吞吐为单路 4.6–6 倍。PowerCons（反馈 60 条）三 seed Macro-F1 标准差 1.6–3.7 pp，与 1 pp 门槛同量级。
- **算子事实**：AutoDA 默认启用 {Raw, Scale, Jitter, Downsampling, Resampling, FreqWarp}；Scale 是逐点乘性噪声（s = 0.5 恒等），MagnitudeWarp 是向随机正弦插值，TimeWarp 在 s ≥ 0.1 时 30–83% 的时间映射非单调，DRC/Perm/Rotation 在单变量 GPU 输入上报错。Planner 的五类候选中只有 Jitter、WindowSliceWarp 语义对得上。
- **数据**：timeseriesclassification.com 当前整站 403；改用 UCR 官方归档 UCRArchive_2018.zip（服务器 `data/ucr_archive_2018/`），本地 17 个 .ts 与之逐一核对完全一致。合格 44 个、独立来源 24 个；按同源同侧，没有族能凑满 12 个（Image 12 个 / 5 个来源，Sensor 7 个 / 6 个来源）。已报告缺口，未降门槛。
- **待定**（Planner / 用户）：名单结构、候选算子与强度（含 p）、投入规则与噪声的关系；定下后一次冻结完整实验规模与预算。

#### 5.12.1 主包 DEV-CLS-AUG-OFFLINE-SKILL（2026-09-25 午启动，进行中）

任务书 [DEV-CLS-AUG-OFFLINE-SKILL](docs/DEV_CLS_AUG_OFFLINE_SKILL_TASK_2026-09-25.md)；名单 [CLS_AUG_ROSTER_V1.json](docs/CLS_AUG_ROSTER_V1.json)（方案 a：旧线从未打开 TEST 的 12 个来源全部作 Test，曾打开的 12 个分 8 Source / 4 Select）。Planner 最终修订：四算子（Jitter、Scaling、MagWarp、WSlice）按固定顺序组成 11 个程序、整条程序 0.5 概率；Source/Select 一次分层 2:1 划分 × 3 seed（用户：卡少不跑三折）；Source 11 程序全网格，Select/Test 只训练实际交付；六臂（不增强、无卡、朴素卡、经验卡、Source 最佳固定程序、冻结随机程序），经验卡与朴素卡各两张、对称选卡；Test 交付全部冻结后一次打开官方 TEST。
- **补充曝光事实**：早期 E2 线有 6 份报告打开过 22 个 UCR 数据集的官方 TEST（`artifacts/functional/e2/*_report.json`），不只 Epilepsy2；映射到 24 个来源恰好 12 净 / 12 曝光。
- **执行**：拟合在服务器（只用空闲卡，Source 网格优先），LLM 在本机；Agent 只见匿名化的 TRAIN 统计与工具，不见任何分类器分数。无卡 Agent 在 8 个 Source、12 个 Test 上全部完成（20/20），交付高度集中在 MagWarp。偏差：一次 1584 次三折计划启动约 1 分钟即按修订停掉（0 结果）；一次调度器被自身 `pkill` 模式误杀（worker 未受影响，已改锚定模式）；为让 Source 网格优先，停掉 12 个在跑的 Test 拟合（约 20 分钟工作量，之后重跑）。

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
