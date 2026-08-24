# Post-V1 路线图(2026-08-22 冻结)

本文件是 V1 里程碑(v7 九环闭合)之后的总体安排,供主线、执行 Agent 与外审共同使用。
逐轮台账见 `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`(canonical,遇冲突以台账为准)。
当前系统形态、证据切面与数据曝光的短表见
`docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md`；Task/Consumer 相对的数据质量与
反馈语义见 `docs/DATA_QUALITY_AND_FEEDBACK_MODEL.md`。

## 当前活动路线（2026-08-24；覆盖下方旧“立即待办/Phase S 下一步”措辞）

- #42j/#42l 已收口，#43 M0-C = C32：Yahoo 已曝光 24 条 × 三个 AD Consumer ×
  当前五清洗程序的 12 个宏效用均为负，IForest/PCA 预注册翻转未确认，PCA 无安全
  headroom。该结果只关闭这一数据/Consumer/菜单组合的继续探针；不新增第四
  Consumer、第七程序或 U4/U5，也不得外推为 AD 无优化空间。
- 当前下一门为 #44a AD held-in 反馈正控：固定一个 Consumer，只注入一种已知训练
  污染，先分别验证污染造成 delayed 伤害与 repair 恢复真实任务效用，再验证
  Support 是否预测该恢复。它是 development 正控，不构成自然 Yahoo 能力证据。
- #44a 通过后按单假设定序：#44b 只考 correct/neutral/shuffled 合同的 Context
  因果；M1b 在正确合同下考双侧多轮生命周期（可修时形成正 Skill、不该动时
  abstain），随后才可冻结管线打开 Yahoo 剩余 41 条。
- 终考报告 Static/A3；只有存在经审计且确实改变行为的积累知识时才加入 A5。
  A5 仍是产品形态，A3 与 Static 只是归因消融。
- 下方 #26/#27、Phase S 和 C23 状态段保留作历史，不得用于签发当前任务。

## 0. 已入账状态(带工件指针,不得复述超出其口径)

| 结论 | 口径上限 | 工件 |
|---|---|---|
| FRESH_A5_DELIVERS | pooled 首正成本 −43.9%(69 vs 123 重训),最终质量与 harm 同冷启动;per_channel = A5_TIE_TRANSFER_BOUNDARY;NOAA 2025 fresh 区域在反馈消耗式适应中一次性打开（非 Fast-only held-out） | `fresh_confirmation_v1.*` + `fresh_confirmation_v1_adjudication.md`(canonical 措辞) |
| LOCAL_LIFECYCLE_CLOSES | Target-local Skill 形成/晋级/持久化/召回,development 级 | `local_skill_recall_v1.*` |
| SLOW_CLOSES_SCOPE_GAP_BY_VETO | 银行重放,containment 非效用改善 | `slow_scope_update_v2.*` |
| BANKED_CHAIN_CLOSES_IN_K_MODELS 2/2 | 6–9 环端到端,gpt-5.6-sol 与 gpt-5.6-luna | `operational_pipeline_v6.*` |
| DEVELOPMENT_OPERATIONAL_PIPELINE_CLOSES_POST_FIX_ON_GPT_5_6_SOL | 九环一次连续无人闭合;判据 (i) 现场成立;DEVELOPMENT 级,非 fresh,非 A5>A3 新结果,不声称后端无关 | `operational_pipeline_v7.*` |

仪器事实(可引用):选择器 5/5 且非写死(v7 越过干净 task_A 选中 task_C);聚合盲区已复现 5 次;
冻结 v2 阶梯三条路径均有 live 样本;两钥匙门多次拦住不该复用的技能;信封错误分类学 + 三计数制已落地;
引用≠遵从 n=7 只记不断因果;三次 guard 提案参数逐字收敛按"选择空间窄"记账;
Luna 证据敏感(#19 弃权为实话),Opus 两次协议失败系上游宕机已划销复位。

## 1. 立即待办(顺序固定)

1. 两笔提交(#26 报文已授权):修复提交(恰四文件:`runtime/agent_backend.py`、
   `methods/ttha/harness/h0/snapshot.lock.json`、`operational_pipeline_v5.json/.md`;
   修复 f8a50ee 新克隆 INSTRUMENT_DRIFT)+ 里程碑提交(runner + v7 工件 + 台账)。
2. #27(任务书已发,聊天记录 2026-08-22 02:00 前后):BY_RESCOPE 语法 +1、银行三臂对照、
   guard 误伤半考。判定集与预算见原书。交付 `operational_pipeline_v8.*`。

## 2. 阶段计划

每阶段遵守:一轮一个方法面改动;预注册判定集;预算封顶;负结果照常入账并可关闭该阶段。

### Phase R — 完成(2026-08-22)
- R2 结果:v9 COMPILER_REJECTS(信封/manifest 契约错位,已按外审指令统一契约,
  升级条款入纪律)→ v10 **LIVE_RESCOPE_CONTAINS_WITHOUT_COLLATERAL**
  (task_D 同窗对照 RESCOPE +0.0959 vs VETO 0.0,双双清零伤害,零误伤,
  保留序列逐位不动)。遗留:记录缺陷第四例(#29 修)、纯 clone 不可导入
  (symlink 未入库,backlog)、Opus 复跑(可选)。

### Phase R 原书(存档)
- R1 = #27 完成(v8):RESCOPE_PRESERVES_GAIN_ELIMINATES_HARM + SLOW_PROPOSES_RESCOPE
  + NO_FALSE_POSITIVE + AS_PROPOSED_CONTAINS_WITHOUT_COLLATERAL;几何偏离已追认
  (评估侧 identity 路由 + 投影,非掩码;同窗选择 caveat 常挂)。
- R2 = #28(书已发):Part 0 检查点(.gitattributes 治 CRLF、#27 交付、台账、路线图)
  → Part A 接线(identity_routed_eval_series 进 plan label / Experience / Skill 生命周期,
  per_origin_gain 显式 null+理由,Opus carry-in 清零)→ Part B live 单轨
  (--rescope-live,两动作平权,gpt-5.6-sol,一抽不重掷);
  判定 LIVE_RESCOPE_CONTAINS_WITHOUT_COLLATERAL / LIVE_RESCOPE_COLLATERAL /
  SLOW_PREFERS_VETO(如实)/ 沿 v7 判定集;预算 LLM ≤8、重训 ≤200。
  交付 `operational_pipeline_v9.*`。

### Phase O — Opus 读数(随时,不阻塞)
- Claude 服务恢复后银行复跑一次(≈1 LLM);仍宕机记 INCONCLUSIVE_TRANSPORT 不耗额度。
  只更新后端表格,不改任何 claim。

### Phase C — 整合与 claim 冻结(quota 紧张时可提前)
- 台账已很长:把 STAGE_REPORT 重构为「冻结 claim 表(每条带口径上限与工件指针)+
  开放问题表 + 仪器事实表」三段;不新增结论,只整理。0 LLM。
  这是任何新会话/新 Agent 接手前的推荐第一步。

### Phase S — Shared Capability 跨域(下一个方法里程碑,设计定稿 2026-08-22)

**科学问题(可证伪)**:把 traffic(旧线)与 NOAA(本线)两域的正/负/冲突 Experience
与风险证据,确定性归纳为一个 Shared Capability candidate(Workflow 卡 + per-series
harm guard,适用条件全部用部署时可观察 Context 表达,禁止数据集名);在第三个
未消费域上同预算对照 A5''(带 candidate)vs A3''(冷启动),检验三件事:
(i) 到首个 delayed-positive 的试错成本是否下降(#17 同口径);
(ii) harm 是否不增加——guard 应在第三域同样拦截/路由(风险面可迁移性);
(iii) 卡上 Context 条件的匹配判断与实际收益方向是否一致(Scope 的可证伪检验)。
(iii) 是超出 #17 的增量:#17 考的是"菜单省成本",S 考的是"条件化的执行权"。

**阶段门**(每步一书、逐书封顶、过门才发下一书):
- S0 域盘点(0 LLM):从台账/旧线工件/data/ 实测盘点候选第三域,逐域标注
  context_exposure(UNSEEN/AGGREGATE_SEEN/INSTANCE_SEEN)与 outcome_exposure
  (SEALED/EXPOSED)及结构(长度/序列数/频率/缺陷可见性)。分层:
  Tier1=不同域族(electricity 未消费切片、T233 等),Tier2=同族新区(NOAA 新
  国家/地区站点)。门 = 至少一个 Tier1 或 Tier2 候选结构合格且 INSTANCE 级未见;
  失败 → NO_ELIGIBLE_THIRD_DOMAIN,Phase S 阻塞,回用户(数据获取决策)。
- S1 健康检查(development 区,零 outcome 打开):只输出冻结聚合统计
  (结构/缺失/缺陷 prevalence);判定 PROCEED_UNCHANGED / STOP_FOR_LOW_PREVALENCE;
  不得据此调阈值或挑个体(章程 §6 Feasibility 纪律)。主选失败换备选;
  双失败 → 该 capability 的 Context 在可得域中不出现,如实关闭并回用户。
- S2 候选编译(0 LLM 确定性):从两域银行 Experience 编译 candidate 并冻结
  version;门 = 干重放(dry replay)在两个源域上复现已知结局(candidate 必须
  能解释自己的训练证据);失败 → NO_COMPRESSIBLE_SHARED_CONTEXT(证据压不进
  可观察条件),按章程回 Observation 面或关闭,不得用数据集名硬编码。
- S3 development 试运行:第三域 development 区,A5'' vs A3'' 同预算,#17 同
  口径读数 + guard 行为;预注册判定集(DELIVERS / TIE_BOUNDARY / HARMS /
  GUARD_MISSES);通过才允许 S4。
- S4 冻结确认(fresh):Program/Scope/Judge/roster 冻结后一次性打开第三域
  held-out outcome;判定 SHARED_CAPABILITY_{DELIVERS,TIE,HARMS};版本一次性,
  打开后改方法必须立新 version,原 cohort 不再 fresh。

**失败分流**(每格指向被起诉的面,不许混):
- S0 无候选 → 数据供给,用户决策;
- S1 低 prevalence → 该 Context 不出现,Scope 事实,停车不算方法失败;
- S2 压不进条件 → Context 表示不足 → 按章程一轮只许加一个 Observation,
  或诚实关闭 family;
- S3 无成本优势但无 harm → capability 级迁移边界(同 #17 per_channel 先例),
  记边界、不晋级、candidate 保持两域局部;
- S3 guard 未拦到第三域 harm → 风险面不可迁移 → first-fault 归因一次 +
  至多一个有界修复,禁调参循环;再失败记负结论;
- S3 负迁移(A5'' 比 A3'' 更差/更害)→ 最有价值的失败:Context 匹配但效用
  翻转 → 按章程 §2.1 记 Scope 不足证据、拆分 Scope、保留反例,candidate
  在该域降级 LOCAL_DRAFT;
- S4 在 S3 通过后失败 → development-to-fresh 缺口,如实报,一次性版本已耗,
  不得重开。

**预算**:S0/S1/S2 均 0 LLM(S1 有少量特征计算);S3 LLM ≤30、重训 ≤400;
S4 参照 #17(≈200 重训)。总重训 ≤700 上限,S1 报告后由用户拍板正式额度。

**状态(2026-08-22 终版)**:S2 = PASS,candidate v2 冻结(iqr/mad 两域证据、
双向 LODO 4/4+4/4、authorization=GUIDANCE、target_support_required=true);
S1b/#32 = JUDGE_UNREADABLE(异构逐机映射)、#33 = JUDGE_UNREADABLE_ALIGNED_
MAPPING(ch18 对齐映射)→ **SMD 在当前 fixed forecasting family 下关闭**,
且该关闭对"份额门=3×均分"的几何修正稳健(spread 实质失败仍在)。
**Phase S 停在数据供给**:真 Tier1 新域获取(用户供数则继续)/ 停车封存 v2;
NOAA 新区域不得冒充第三域。v2 为已冻结正资产,随新域到来即可复活 S1b→S3。
**当前推荐的下一刀 = O7 guard held-out 化**(与新域无关,用已有 NOAA 银行,
解全部 guard/RESCOPE claim 上最大的同窗选择 caveat)。

### 契约性收尾(小,择机并入任一轮 Part A)
- SELECTION_MISS 适配器窄口径修复(在册缺陷,两次兑现;修复后跑一次 0-LLM 归因回归)。
- 记录家族规则已根治,后续新阶段沿用"进入阶段前挂载记录"。

### 停车场(有观察证据前不排期)
- per_channel 迁移边界(#17):需先有"什么 Observation 能区分 per-channel Context"的假设。
- 第二 Slow 面/补丁积累:仅当自然出现第二类 harness 故障时立项,不预建。
- 供给面(菜单可采纳率、#22 型自堵):留观察,shortlist 稳定性表继续随轮追加。

## 3. 常备纪律(每份任务书默认携带)

1. 一轮一个方法面改动;仪器修复与方法改动不得混刀。
2. 预注册判定集;标签名不副实时 raw 保留 + 主线改判并存(先例:#17/#23/#24)。
3. 不重掷、不做前向概率;非 iid 只报观察频数。
4. 后端/证据出处断言必须带工件路径并核验(教训:#25 初稿)。
5. 三计数制:valid_decision_samples / protocol_failed_draws / llm_calls_spent;
   协议失败独立上限 2;transport 记 INCONCLUSIVE 不耗额度。
6. 证据等级三档:INSTRUMENT / MECHANISM(banked)/ DEVELOPMENT(live);
   fresh 是保留字,仅限 sealed outcome 一次性打开。
7. BY_VETO=containment;"无殃及"仅指不相关 episode 决策不变;保收益措辞属 RESCOPE。
8. 冻结面跑前跑后核对;v* 原档只增不改;交付不 commit,统一检查点、显式 add。
9. 反 SHA 扩张:沿现行清单版式,不建新哈希体系;git diff 即可。
10. 子 Agent 不得 spawn 下级;每书注明;beyond_17520 零读取;运行期间另一线停笔。

## 3.5 范围锁定与 Phase T/M/X(2026-08-22,取代上文 Phase S 的"下一步"地位)

**项目范围(用户裁定)**:当前 Phase T/M 的自然验证切片限定单变量，避免在同一轮
同时引入多变量 Adapter；这不是最终 Harness 的永久数据形态边界。多变量支持在
当前单变量纵向链路跑通后再做最小适配。Task/Consumer、模型、Domain、Pattern
全部可变；质量标准随任务/模型/模式变化是第一性命题，forecasting 线的全部已
入账证据只覆盖一个 family。

**主张架构 v2(2026-08-23,用户 + sol 会审采纳;取代"A5 在陌生域更快找到
正解"式单条总主张,台账同日节为准)**:总主张 = 构建面向多 Task、多
Consumer、多 Domain 的时序数据质量适应 Harness——依据部署时可观察的局部
时序 Pattern、候选 Program 的作用几何与 Task/Consumer 的质量语义,在新域
held-in 上多轮自主生成/验证/更新 Target-local Workflow Skill 与 Harness,在同域 held-out
上验收效用与风险;当 Source 与 Target 共享可观察决策 Context 时,冻结
Source-derived Skill 进一步减少适应成本与负迁移。
**系统形态澄清(2026-08-23 用户质询裁定)**:系统形态上积累与适应不可拆
——最终交付是单一进化 Harness,跨域积累 → 新域 held-in 适应 → 冻结
held-out 部署为同一生命周期;held-in/held-out 只切反馈可得性,不切知识
可用性,系统臂(A5/全量)的跨域 Skill 在两侧全程在场,held-out 禁学习
不禁知识使用。L1/L2 是归因刻度不是产品拆分:A3 是"关闭积累"的消融臂,
存在目的是让胜利可归因。弃用"主线/增强线"命名(产品化误读),改称
L1 = 必要条件层、L2 = 积累贡献层。"积累承担大部分、适应只是校准"是
权重假设:控制世界旧证据支持(W47 A5 AdaptAUC 1.0 vs A3 0.875;W56
A4 零射 6/6 正),自然世界证据当前只支持条件加速器(FRESH −43.9% 终态
平;#42e1 先验对 Target 实误、未采用反避损)。渐近方向 = 库成熟后逐
新域以递减 held-in 预算复测 L2,重复测量支持时该假设即升格(亦为旧 A4
零/少探针重新入场路径),测到为止,不预设。四层主张与主比较:
- L1 核心适应性:新域 held-in 适应 → 冻结 → held-out 终态(A3 vs 静态
  默认;无条件承重,不依赖任何 Source 先验);
- L2 Source 先验价值:同 held-in 预算下 A5 vs A3,同一 held-out 验收
  (FRESH_A5_DELIVERS −43.9% 为其预测线首个有界正例);
- L3 Pattern 机制:完整 Context vs 去 Pattern/错配——收益须来自
  Pattern × Program 几何 × Task/Consumer 匹配,非 dataset 记忆
  (消融臂只跑已曝光域);
- L4 载体条件化:同一处理随 Task/Consumer 反号且 Harness 随之改变行为
  (M0/M1;T1b 注入正控为先例)。
旧 A4(少/零 probe 直接执行)= 远期可选目标;但“经审计的跨域积累参与新域
held-in 校准”是完整 Harness 的组成,不是可永久删除的增强项。L1–L4 只作
证据归因坐标。**当前组件闭环 = #42g Static/A3 + M0 + M1;项目最终闭环还
必须包含至少一次自然数据上的 Static/A3/A5 同场验收。**M0 仅证"反号现象
存在"的正控,载体条件化能力须 M1 坐实;只有 A3 的正结果不能替代完整 Harness。
**承重读数纪律**:能力对比以冻结终态在 held-out 的效用与安全承重
(utility / harmed series / worst-series gain / abstention / 方向一致率);
首正成本、试验数、feedback 消耗为解释性辅读数,不得单独支撑能力主张。
held-in 预算由已曝光 development 轨迹预先冻结,禁挑选"刚好让 A5 赢"的
紧预算;保留适应过程曲线;宽预算下终态打平而 A5 更快仍记有效加速;
处女 held-out 只被冻结态触碰。
**协议语义钉(2026-08-23,用户定义立为正典)**:held-in = 反馈可用区
——Support(即时)与 delayed(后续)反馈、Episode 写入、Slow 修正、
Target-local Skill 形成/批准/撤权全部只在其内。held-in 是可在预冻结总反馈
预算与停止规则内运行 `r1...rR` 的 self-harness 适应环境,不是一次性 batch:
前轮 Episode/Skill/Risk/Harness Patch 可进入后轮 Fast replay;每次 Consumer
调用仍计预算,同一 Outcome 的缓存重放或重复读取不得计作独立新证据;
held-out = 零反馈
Fast-only 部署区——冻结态检索 Skill + 部署可观察 Context 自主出
Workflow,禁 open_delayed / 反馈回传 / Slow / Skill 变更 / 按 outcome
写 Experience / 看结果重试;计分 = 全臂输出冻结后外部 evaluator
一次性开 outcome,开即 EXPOSED。#42g 四段 = held-in iterative adaptation(多轮 r1…rR;轮数上限、
总反馈预算、可用窗口、停止规则书面预冻结,禁预指定 Workflow 或
逐轮答案;2026-08-23 C24 轮钉)→
freeze(Static*/A3*/A5*)→ held-out deployment(Fast-only)→
offline evaluation;Part C 需新增 Fast-only 部署入口(现役 replay
均含 open_delayed,仪器缺口在册),Part D 沿用 LabelWall +
evaluate_released;#42f 获取 census 时即预冻结:文件 roster、
cohort 定义、**按序列内时间边界**的 held-in/held-out 切分
(2026-08-23 sol 修正采纳,supersede"按序列切")、标签隔离方式;
held-in outcome 至 #42g Part A 才开,held-out outcome 封存至
Part D 离线计分。首正成本 = held-in 读数;FRESH_A5_DELIVERS 属
"fresh 区域反馈消耗式适应"读数,与冻结部署 held-out 读数不可互替。
**数据三角色细分(2026-08-24 用户+sol 会审)**:外墙内再分
Target base-train(下游拟合 + 无标签 Context,零反馈)与
feedback-bearing adaptation windows(Support/delayed 反馈窗,
真正应控小的量),连同 frozen held-out 构成三角色;今后不再把
整个前缀称"适应集"。无固定比例教条(Self-Harness 43/21 参考,
sol 核);claim 须显式报告反馈窗份额。#42g 现行 = 70/30 外墙
第一版组件实验(反馈窗份额 40%),不中途改;后续选项 = 剩余
41 条 sealed 序列预注册更小份额(sol 20% 方案备案)确认,或
下一新域冻结三分协议;已开标签 24 条不得换比例重称 fresh。
**历史执行锁(C23 后,2026-08-23；已由文件顶部当前活动路线取代)**:#42e1 已以
`RISK_PRIOR_EFFECT_AMBIGUOUS` 收口,v3 归档,当前 #42g 的 AD L2
关闭且不得通过修改 v3、重抽 AdExchange 或追加 Source cohort 重开。该裁定只
关闭 v3 进入本次考场的资格,不关闭完整系统中的 A5/累积知识角色。下一组件门
固定为 #42f(冻结未曝光 Target 的 held-in/held-out 边界)→ #42g
(Static vs A3),用于验收 AD Target 校准端口,不作为最终产品形态。#42e2 若已
分发可按冻结单假设完成一次,但它是 Pattern 机制诊断,不阻塞 #42f/#42g,
也不自动为当前 #42g 增加 A5;
任何未来 Pattern-conditioned Source Skill 必须建立新 version、另过
development 行为验收,进入后续独立正式考场。
**#42f 数据获取顺序**:先盘点本地已有候选与 exposure;只有本地不存在满足
任务语义、时间 held-in/held-out 和 outcome=SEALED 的 Target 时,才按任务书与
用户授权下载。新域用于一次干净验收,不是继续堆 Source 数据。
**独有性声称前置**:相关工作检索(载体条件化与 Pattern 机制在既有
harness 进化文献中的存在性)完成前,不得声称"独有",入 #46 前债。

**阶段图**:
- **Phase T — Task-conditioned quality**(当前主线):同一单变量 Pattern 下
  forecasting 与 anomaly detection 双 Consumer 的方向翻转与条件化适配。
  T0 仪器定义+底物普查(#35)→ T1 注入正控(#36,改判输入侧)→
  T2 TaskSpec/Consumer 观察接线审计(#37)→ **T1b 训练侧翻转正控(#38,
  双 Consumer 同训练字节、独立未处理 Query 计分)** → T3 任务条件化决策
  (#39,门控在训练侧翻转上;C1 用方差参照判据)→ T4 冲突 Experience
  写入与按任务检索(#40)→ T5 生命周期闭环(#41)→ T6 fresh 跨域确认
  (#42,需新域,见 O9)。机制阶段全部用注入正控 + 自有 dev 数据,
  不需新数据。
  进度(2026-08-22):T0 = `T0_READY`(#35,AD Consumer 冻结于 49/3.5,
  回退已用尽);T1 = `TASK_FLIP_CONFIRMED_POSITIVE_CONTROL`(#36,C12,
  4/4 程序同向,guard 语法对 AD 向量直读通过);T2 =
  `TASK_CONTEXT_GAP_PATCHED`(#37,task_spec 三项入公开视图,三重
  物化验收;卡层 task 维已在;T4 范围钉死 = episode 键 task 分量 +
  卡词汇 consumer 特征);T3 v1 零消耗撤回(estimand 不对称,sol 审核
  采纳,C12 改判输入侧);T1b v1/v2 双停 `AD_TRAINABLE_SPEC_DEFECT`
  (根因 = 特征族×线性头表示缺口,可信负关闭"原始窗×线性 ridge"规格族;
  仪器取代:trainable_v1/v2 化石,v3 接任);T1b v3 =
  `TRAINING_SIDE_TASK_FLIP_CONFIRMED_POSITIVE_CONTROL`(C13:winsorize
  臂 F +0.4059 / AD 宏 −0.1672,程序特异翻转;门 0.6109 一次过;
  T1b 关卷,累计 AD 280/400、LLM 0、重训 0);T3 =
  `TASK_CONDITIONED_PROPOSALS_CONFIRMED`(C14:完全分离 1.0>0.5,
  聚合 3/3+3/3,Risk F 0/3(先验-经验缺口:hampel 先验安全实则
  风险键外)/ AD 3/3;第二次抽样裁定有效;LLM 12/12);
  T4 = `PARTIAL_EXPERIENCE_CONDITIONING`(C15:F Risk 0/3→3/3
  冲突卡纠偏成立;AD 3/3→0/3 回退,机制 = 卡表达范围无 abstain
  通道;键/写入/检索全绿;键统一落地,方言负债与 bundle 盲区入册
  路由 #41b);  #40b = `TASK_SEPARATION_REGRESSION`(C16:abstain 通道使 AD Risk
  0/3→2/3、F 3/3 保持;分离门败于共有合法弃权词 identity 进双臂
  shortlist,非任务趋同;判据教训入册);T4 收束 = Memory 机制部分
  成立、安全闭环未完成,不做卡序第三修(防答案 Router);
  T5(#41)= `INCOMPLETE_LLM_BUDGET`(C17,687af6e:单入口双任务
  真实运行、三 delayed 全 CONFLICT 全拒、F 自写经验改变下一轮探索
  且归因字节级成立、零串写零泄漏——止损线首考存活;未证 = AD r2、
  真实轨迹 Skill 激活复用、Memory 直选安全计划;预算算术为主线
  协议缺陷,不重跑,未证项移交自然阶段);收尾追认(任务化 ID 破
  五测全修,手拼副本收口为 fast_winner_skill_id() 单一出处;MKL
  崩溃 = 既有环境故障路由 #41b);
  #41b-lite = V10_READY_FOR_T6 @ 5dee103(V10 四十成员零漂移);
  #42 v1 = NATURAL_DATA_SHAPE_INELIGIBLE(有效 first-fault,四路
  越权补救零触碰,LabelWall 构造时丢弃,freshness 未烧;根因改判 =
  A4 门与行序 Consumer 契约错配,书面缺陷主线共领);
  #42a = **T6_NATURAL_PLAN_READY**(C18:三类 delayed 层独立集齐
  2P/4N/1C,契约 20/20 无损,29/29 窗口全映射,自然跨 cohort 效用
  翻转两例 = 核心前提自然实证;known_cause r1 delayed 近零分辨率
  为已知限制随卷);主线独立核验封存后已置位 evaluate_released=true;
  evaluate 首次释放收回(sol 发现执行体 stub:runner:1285 一次未调
  run_online_round,note 字符串 + 退出码 0 双重陷阱,gateway 绑
  zeros(1) 占位;SEALED 完好,flag 已翻回 false;主线释放清单新增
  "核验可执行体存在");
  #42b 执行体已补(759 行,八 cell smoke 未覆盖正向分支——顺序
  教训:纵向最小切片对仪器同样适用);
  #42c = LIFECYCLE_FIXTURE_CLOSED(单 cell 全链真实走通含
  LOCAL_ACTIVE 正向分支,读数机械复现 v2 bank,16/20 fits 0 LLM;
  flag 事件裁定 = 主线蓄意置位非误触,执行方拒跑上报 = 正确,
  永久口径"flag 必要非充分");
  正式 evaluate 完成 = **C19**(盘面 SAFER_NOT_FASTER + 竞态影子
  样本 NO_ADVANTAGE,承重取交集:未加速一致、无负迁移一致、
  安全优势 1/2 不稳;**附带首次自然生命周期闭合**:CPM A3 r2
  outlier_mad LOCAL_ACTIVE;AdExchange 已曝光不再 virgin;
  one-shot 站规新增 run-id 隔离 + 启动锁 + 杀后验尸);
  **[15:2x supersede]** 修面路由撤回——架构回退裁定:仓库级
  AGENTS.md override(L21-99,压过 workspace 章程)禁 Episode/
  ContrastPack 直入 Fast,前例 T233 = RAW_SOURCE_EPISODES_TO_FAST_
  REJECTED 且路线锁禁"检索/聚合式修复";#40 卡线 → #42 A5 直连 =
  重建被否决路线;C19 主线审定 = INCONCLUSIVE_CONCURRENT_EXECUTION_
  COLLISION + 受控旁路,NAB Target 降 development 永久非 virgin;
  C14-C16 改标旁路机制证据;T5/域内轨迹检索合法无损(override
  L44 明许当前轨迹 Support 历史);C18 bank 成立(整合原料);
  当前 = **校正后 A5 主链(照 override L86-95 既定切片)**:
  (1) 关直连接线 →(2) 新方法切片:Source census → Slow 整合 →
  冻结 Source-derived Skill(NAB 20 卡为原料)→(3) 已曝光轨迹
  development replay 机制验证 →(4) O9 第二未曝光域 census →
  正式 A5vsA3 v2(A5 = h0 + 冻结 Skill,单进程站规)→(5) M0。
  workspace AGENTS.md 已由 sol 改为纯路由(仓库级 override 唯一
  权威);开书前必读仓库级 override 立为站规。
  #42d 书 r1 修订(本地 agent 评审三事实核验采纳):census 证据
  单元改 episode_id、cell = program × 可观察条件((cohort,round,
  program) 分组实测 20 组全单例,v1 证据线不可满足,主线认领);
  Slow 整合复用 T233 既有 source_skill.py 审计链(AD 薄封装,
  本体零改动);replay 改同跑配对 A3/A5'(历史读数按 C19 降附录);
  bank 实况:全池无条件时无程序可获授权,整合器必须靠可观察
  Context 分辨,ABSTAIN 为合法出口;待 sol 复核后分发。
  r2(sol 保真审计采纳):两证据 Runner 手写旧 ID 坐实(recall:411 /
  fresh:1866,重跑必 ValueError——响亮失败非静默错数);两道门
  收紧确认为 #41 T5 A4 在册授权(六旧正例读数全过新门);#42d
  前置 Part 0b 保真收口硬门(修 ID → 公有 fast_winner_skill_id +
  0-LLM 缓存重放,绿 = FORECASTING_COMPAT_RESTORED 续跑,
  红即停);T6 直连行现为 :1619,Part A 删除对象确认。
  r3(sol 三修采纳 + 主线 claim 勘误):FRESH_A5_DELIVERS 在册
  (NOAA 2025 fresh 适应区 pooled 首正成本 −43.9%,非 Fast-only held-out;per-channel 迁移
  边界)——主线"预测线从未证明"表述错误收回,#42f 定位更正为
  已证优势向第二任务+独立域的推广考试;证据门回归已证水准
  (TRY/RISK 均 ≥2 个不同 Source cohort 计票,min=1 撤回);
  pss 实测完美复刻 cohort 身份且 bank 仅 3 布尔特征(2 常量 +
  1 代理)→ 本 bank 合法条件化结构性不可用,census 增 cohort
  代理检查义务;source_skill.py 改最小参数化(默认路径字节等同
  断言);  预注册推演:TRY 无候选,RISK 唯一候选 hampel(CONFLICT
  计伤害待 sol 终审)或全 ABSTAIN → 跳 Part D 报
  CONSOLIDATION_NO_ELIGIBLE_SKILL,扩 Source 路由
  (realTraffic/realTweets 作第 3/4 Source cohort)。
  **#42d 跑完 = C20**(r1+r2+r3 预注册全兑现):0b 兼容恢复
  6/6、直连已拆、census 逐字合预注册、SOURCE_SKILL_WRITTEN
  (risk-only hampel)、D = SCOPE_CORRECT_NO_APPLICABLE
  (hampel 双臂 0 提案,无坍缩;LLM 25/40,fit 12/120)——
  架构纠偏成功,降权效果未获触发机会,不得声称更快/更安全;
  v1 Skill 文本时序惰性缺陷(RISK 挂未来观察 t=0 永不点火,
  与 T233 惰性同族;审计只查词汇不查时序)→ 收口书 v2(sol
  四修):allowlist 提交 + 密钥扫描前置(raw/临时 Store/另一线
  untracked 显式排除)、时序审计机械化(hampel 非硬禁、VERIFY
  精确两阶段)、判定 = SKILL_V2_FROZEN_PENDING_BEHAVIOR_REPLAY
  ("不重跑"推理撤回,v1 判定不背书 v2)、0-LLM 送达断言;
  **行为重放立为任何 h0s_vN 进正式考场的前置**;两新 Runner
  化石(禁扩建,入 #46 债);本轮性质 = 机制/适配修复;
  收口书跑完 = SKILL_V2_FROZEN_PENDING_BEHAVIOR_REPLAY;sol 补
  两缺口(plan_v2 入库自足;h0s 临时快照永不提交、从冻结 entry
  确定性重建、禁重调 Slow);ad_source_skill.py issue_v2() 冻结;
  #42e0(sol 直发短书,主线追认)= **C21 RISK_SKILL_NO_
  TRIGGERING_CANDIDATE** @ ad4f7b82:v2 送达 4/4-0/4、无全局
  坍缩、hampel 双臂 0、两臂同激活 mad +0.0111(轨迹事实不归因,
  池宽差异如实报);禁再抽 NAB 钓触发;下一步 = #42e Source 扩充
  (realTraffic/realTweets 第 3/4 cohort,冻结门重整合,停止线 =
  SOURCE_EVIDENCE_INSUFFICIENT_FOR_ACTIONABLE_TRANSFER)→ 有
  h0s_v3 则 #42e1 行为重放 → 才谈 fresh Target;
  排期编号后移:#42e = Source 扩充(realTraffic/realTweets,
  TRY 解锁唯一路径)→ #42f = 新域 census(Yahoo S5 优先)→
  #42g = 正式 A5vsA3 v2(禁用惰性 h0s 消耗处女域);
  主张架构 v2 落位(2026-08-23,全文见本节首):#42e r1 重定位 =
  当前 Source 表示下的 actionability census,停止判词改名
  NO_ACTIONABLE_SOURCE_SKILL + 强制 first-fault 三分
  (Observation 不足 / Program 无重复 headroom / Source family
  不适配),家族封顶禁新下载;新增 #42e2 = winsorize 反号最小
  Observation 判别(严格在 #42e 之后——2-cohort 库上完美分离
  特征即 cohort 代理,合法性按 census v3 代理判据审;0 新数据);
  #42g 重塑两层协议(L1 A3 vs 静态默认无条件,L2 A5 vs A3
  条件开,承重 = held-out 终态,首正成本降辅读数);Pattern
  insight 重定义 = 可观察 Pattern × Program 几何 × Task/Consumer
  → Action-Response,检索单位禁 dataset 名。
  **#42e 已跑完 = C22 SOURCE_RISK_ONLY_TRIGGERABLE**(主线读工件
  核验:4 cohort / 40 卡,17 新文件门全过;TRY=[],RISK=[hampel,
  iqr, mad](严格伤害 3/3/3,hampel 扩展 4,全部零正例);winsorize
  aws 唯一正例 + 3 cohort 伤害 = 无授权;合法 Scope=0(pss 降
  single-cohort indicator 仍禁,另两特征恒常);skill v3 =
  source_investigation_ad_v3,h0s_v3 f2054da1…,v1/v2 superseded
  不入;v3 首次可触发(v2 只降权 hampel 历史 0 提案;v3 纳入
  mad/iqr 历史提案 4/2 次);LLM 1/8、fit 170/240;r1 增补页因
  执行交叉降为事后解释规则,family 已封顶禁新下载);  排序更正:
  **#42e1(v3 一次性行为验收,书已发)先于 #42e2**;#42g L2 仅当
  #42e1 = RISK_PRIOR_BEHAVIOR_EFFECTIVE 才开;#45 重定位 =
  forecasting 复现/回归验证(NOAA 2025 已开不得再称 fresh);
  距离改按能力门计:底座成 + forecasting 纵向切片成,缺 AD 纵向
  切片(#42g L1)与载体条件化切片(M0+M1),主体两场 = #42g L1
  与 M1,增强线不得拖住。
  **#42e1 = C23 RISK_PRIOR_EFFECT_AMBIGUOUS**(Part 0 = 20218007b;
  送达 4/4-0/4、h0s_v3 从 entry 重建哈希相符;触发场合实存——
  CPM r2 双臂各探 mad +0.0593 → delayed +0.0111 → LOCAL_ACTIVE
  终态打平;A5 未少探 mad 反多列 hampel 且提案不引用降权条款,
  无可归因行为效应;阶梯外新格主线追认,今后判定集必带 residual
  格);**v3 归档,#42g L2(AD)当前知识下关闭**,#42g 简化为
  L1 主考(Static vs A3)——主线不受增强线拖累首次兑现;
  风险事实入册:v3 的无条件 mad 降权与本 Target 唯一自然正向 mad
  证据方向冲突;若它严格阻断 mad 会损失该正例,但该反事实并未发生,
  不得写成已观测负迁移。合法 Pattern/Context 分辨力因此成为候选
  first-fault,尚未被证明为唯一必要修复;
  增强线 A(AD)四轮弧线收口(#42d→#42e0→#42e→#42e1,诚实
  诊断性负结果);**#42e2 书已发**(纯测量 0 LLM / 0 fit:
  isolated-extreme × winsorize 单假设判别 + 代理审计 + LOCO,
  mad 附录;一轮一假设)。
  **sol 排期审核采纳(2026-08-23)**:唯一承重两场 = #42g L1 与
  #44 M1(M0 = 前提正控非承重,#42f 备考,#45 回归,#46 收口);
  #42e2 r1 = 顶格判词改 PATTERN_CANDIDATE_CLUE(正例侧单 cohort
  为结构性上限)+ 非阻塞钉 + 完美分离两读法区分义务;#42g 三态
  写死(Static = identity + 一条预注册固定通用清洗双基线);
  M0 纯度钉(只变 Consumer 归纳偏置)+ M1 禁按模型名派答案;
  #45 压缩为 #46 前轻量回归;#46 交付 = 单一 Harness 入口处理
  不同 TaskSpec/Consumer 的最终纵向系统;项目 AGENTS.md 对齐
  核验属实;workspace ccfa.yaml 最小状态更新完成(stage 块 +
  authority_note,历史段保留)。
  最终定序:#42e2 → #42f → #42g L1 → #43 M0 → #44 M1 →
  #45 轻量回归 → Static/A3/A5 三臂同场验收(合格累积 Skill +
  sealed Target 齐备后,#46 前必经;用户二次校正 2026-08-23)
  → #46 系统整合。
  **#42e2 = C24 PATTERN_NO_DISCRIMINATION**(主线读工件核验:
  aws 正例与 traffic/tweets 负例同为 isolated_dominant=True,
  唯一 False = kc 长 burst 结构;C1 反侧率 0.25<0.75、C2 单
  cohort 指示器、C3 剔 aws 不可读;family 关闭不入 Scope;
  mad 附录与双基线漏产记偏差主线豁免,教训 = 书面义务分
  "必跑/仅通过时跑"且漏产须自行申报;first-fault =
  OBSERVATION_INSUFFICIENT 限本特征×本程序;L3 零活跃候选,
  增强线 B 暂闭至 #42g 后);**#42f 书已发**(本地盘点 → 预注册
  规则冻结 roster/cohort/0.7n 序列内时间边界/标签双仓隔离;
  禁下载禁开标签;NO_LOCAL_SEALED_CANDIDATE → 用户决策)。
  **#42f = C25 TARGET_FROZEN_SEALED**(停报 → 用户授权下载;
  Yahoo S5 A1 镜像 67 文件 @24958b84,roster 65[MIN_LENGTH=1000,
  与书面 800 结果等价,已申报],单 cohort yahoo_s5_a1,0.7n 双仓
  封存,主线隔离抽查通过;**镜像来源 caveat 承重**:对外引用
  公开基准数字须注 mirror copy);  **#42g 正式书已分发**
  (v0 + sol r1 七条:字典序前 24、两轮窗口写死[.30/.40/.50 与
  .50/.60/.70]、cohort 级 Fast 粒度 + DEPLOYMENT_GRANULARITY_
  UNSUPPORTED 格、训练底物评价语义[held-out Query 原始字节
  不处理]、LLM≤24 / fit≤240、identity 主 Static + hampel
  "固定清洗压力基线" + SAFETY_ONLY 改名、判定优先序;两处
  运行前停报门 = delayed 模型复用烟测、部署粒度)。
  **#42g 首跑 = C26 PROTOCOL_BREACH(裁定改判)**:执行方报
  ADAPTATION_HARMS_HELDOUT 被撤回——冻结态 h0 无 learned
  Skill,计分却按"末轮 winner 优先"(runner:5811)把未批准的
  outlier_mad 当 A3* 部署,违 R1-4;Slow 静默关闭(:5717);
  held-in 轨迹未落盘。保留读数(development):outlier_mad 全
  cohort → held-out 宏 F1 0.2624 vs identity 0.3227,伤 7/24,
  逐序列 5 改善/12 平/7 伤;hampel 自伤 12/24。反馈稀疏实证:
  held-in 反馈窗合计 14 事件(14/24 全空)vs held-out 38。
  前 24 条 EXPOSED 永久 development;41 条 sealed 完好 =
  确认场(50/20/30 三分协议)。AD 特设原则 P1–P6 入台账;
  两常备格 NO_FROZEN_ADAPTATION_STATE / FEEDBACK_EVENT_
  STARVATION;三机械修复(绑定断言/三段落盘顺序/Slow 按书面
  配置)入 #42g-b Part 0b。**#42g-b 诊断书已发**(0 LLM,
  EXPOSED 24 条五程序全响应表 + headroom 判定树 + 稀疏表
  工件化 + 选择缺失核查,fit≤150);定序更新:#42g-b 诊断 →
  #42g-c 确认(41 条 sealed,50/20/30)→ #43 M0 → #44 M1。
  **#42g-b 收口 = C27 PARTIAL_SERIES_HEADROOM_ONLY(采纳)**:
  B1 四程序全局宏 Δ 全负(winsorize −0.092/14 最重,变形
  强度与受害同序);B2 局部赢家 iqr6(特征不可见)/mad5/
  hampel7;B3 反馈 estimand 无偏好(有事件 10 条仍全负,
  14/24 零事件)。机制读法 = "去污染不变形":AD 就绪惩罚
  变形,预测就绪奖励变形;菜单系预测遗产但未枯竭。
  first-fault 移至 Scope/Observation 与 Program Supply
  并列;**#42g-b2 派生读数书已发**(0 LLM/0 fit:oracle-
  scope 天花板、方向一致率、赢家重叠;决策门 ≥+0.02 →
  Scope 线 / <+0.01 → Supply 线;一致率 ≤50% → 反馈单元
  按事件质量重设)。分叉裁定后才动 41 条与 AD-native 增补。
  **sol 预读定分叉(待 b2 工件复核)**:oracle +0.0375(14/24)
  = Scope 潜力不薄;方向一致 17/40=42.5% + 保守 policy −0.0070
  = 现役反馈非可学信号。契约 v0r1 定稿(改名 anomaly_background_
  model_quality_contract_v1 限 Consumer 族;删新 Receipt 与
  no_contamination_evidence)。**执行序改钉:b2 复核 →
  Feedback Unit 重设计(EXPOSED 24,预注册候选单元,判据 =
  一致率 + policy-regret)→ 契约 wiring(纯代码)→ Scope/
  Observation 线 → EXPOSED 24 闭合 → 41 sealed 三臂终考
  (50/20/30)**;"契约 + Scope 合并一书"作废;41 条不动。
  **#42g-b2 收口 = C28 FEEDBACK_UNIT_REDESIGN 门生效**:
  executor 与 sol 独立重聚合逐数一致(仪器交叉验证通过);
  A3 新载重 = 三程序赢家两两 ≤2、三交 ∅、并集 14 →
  "单一可清洗子群"证伪,Scope 线须程序条件化 Observation;
  hampel = 全局最害(12/24)且 oracle 最大贡献者(+0.383/6 条)
  = 高方差算子,series 条件化最强单点证据。  **#42h 反馈单元
  书已发**(四候选 U0–U3 预注册;主判据 C2 policy 模拟;
  选定即冻结,41 条只用不调);序:#42h → 契约 wiring →
  Scope 线 → EXPOSED 闭合 → #42g-c 三臂终考。
  **#42h 收口 = C29 FEEDBACK_UNIT_UNRESOLVED**:四单元全败
  安全门(U2 全区池化 +0.005 仍 worst −0.071);U0 锚字节
  复现;U0/U1 因预注册歧义塌缩(锚复现义务立功)。裁定:
  五菜单"操作性枯竭"(headroom 存在但无合法反馈机制可安全
  收割),P3 解锁 → **Supply 线:一个 AD-native 程序
  contamination_mask_refit_v1(检测评分遮罩重拟合,值不
  改写)**。序改:#42i 纯代码(契约 wiring + 程序注册 +
  仪器扩展)→ #42j census(六程序,同安全门)→ 合格则
  多轮 replay → #42g-c;不合格则弃权框架 #42g-c(A5 正确
  弃权 vs 强采纳消融受伤)。两结局均良定义;41 条不动。
  **文献调研(grok)采纳为设计输入**:三范式表(无监督统计
  /重构一类/监督事件)与实证链互证;M0 配对获文献先验
  (iforest 清洗负 vs AE 族清洗正 = 预期反号翻转),预注册;
  pre-#42j 预期 = held-in 污染轻 + iforest 稳健,mask-refit
  headroom 可能薄,不合格属文献一致结局;Yahoo 病理 caveat
  (Wu&Keogh 四病;稀疏表 38 vs 14 即位置偏置显形)附着
  所有 Yahoo 结论;TSB 资格门进未来选数流程(41 条禁预跑);
  引证核验入 #46 相关工作债。
  **外部静态审查采纳(2026-08-24)**:C29"操作性枯竭"收窄为
  "事件 F1 读出族枯竭"(量化台阶批评成立;连续 Support 信号
  未测,AUPRC 代码已录)→ iforest 现役菜单 capability 路径
  重开。**#42j 重塑 = 六程序 census + Support 信号仪器研究**
  (连续信号只作 Support 排序,判官/晋升保持 event-F1+安全门);
  其后 pre-M0 卫生书(TaskEvaluationContract 缓冻结 + 三泄漏
  fail-closed:online_loop:584/:339/:354、method:239);信号
  可用 → Stage C 局部生命周期正控(EXPOSED 24)→ replay →
  #42g-c capability;全哑 → M0 AE 主场 + 弃权框架。AD 观测
  扩展守标签合法性红线(score margin/flag 率合法;事件密度/
  calibration 含异常等标签项仅 post-hoc)。M0 必做不变。
  **#42i = CODE_LANDED @ 9983e5f**:契约 v0r1 全 wired(15/15
  测试);mask 落地为 consumer fit policy(sol r1,追认优于
  注册表方案;16/16 测试);census 菜单参数化(6-程序 Yahoo
  须 #42j 授权);consumer 文件 provenance 缺口自报闭合。
  **#42j 已发**(六程序 census + 四 Support 信号仪器研究;
  U0 锚复现义务;fit≤200;Stage C 解锁 = ≥1 信号 policy
  过安全门)。
  **外部复审五断言核实(2026-08-24)**:TaskContext 未进主
  runner(预期态);树上 1 失败测试实证;M0b 特征系另线
  已落地(union 污染修复,共享受益)。provenance 权限原则
  采纳(outlier_iqr +6/−0 断言待验)。
  **sol 排程审核全采纳(2026-08-24 15:2x)**:#42j r1 收紧
  (单主判 FIT_POLICY_QUALIFIED;一遍 168 fits,锚同批复算;
  四信号降 development 诊断,仅 AUPRC 预注册候选);卫生书
  解散 → 最小接线修复(context 携带 + max_candidates 适应期
  =2 修正 + 禁静默回退 + group card 从 TaskSpec)全分支必做,
  其余按 #42j 单分支(信号 → ConsumerFeedbackContract + 五
  程序正控;mask → 类型提升接线;双无效 → 转 M0);
  modified_fraction = 测试过时,对齐文档语义,三分记延迟债;
  H0 lock 仅拒跑时重生成;shuffled 合同诊断归 M1;弃权四
  条件(检索引用/改变行为/省试错/消融受伤)方可归因;replay
  双钉(Outcome 不重复计证;机制正控 ≠ 能力读数);41 条
  主张帽 = EXPOSED 24 先验只算"同域跨 series 积累";M0 三
  Consumer 设计,mask 不承担跨 Consumer 反号主张。41 条不动。
  **#42j r1 收口 = C31 FIT_POLICY_NOT_QUALIFIED**(三门全败
  +0.001375/8 伤/−0.167;预登记文献预测命中;Δ_oracle_6
  +0.0478,mask 系高方差局部算子)。f1_pooled 诊断 +0.0059/
  0 伤(零事件排除+保守门的结构功劳)距门 +0.0009 处分辨率
  边缘,不授权、不支撑 iforest Stage C;三件套(零事件排除/
  保守门/全策略晋升)入 ConsumerFeedbackContract 设计库。
  **路由:双无效 → 转 M0**(sol 停止规则;iforest 线收束为
  经审计弃权候选)。**#42k 接线修复书直派 Opus 5**(候选帽
  v2 + fail-closed 三处 + T6 context 携带 + 过时测试对齐);
  其后 = M0 三 Consumer 设计书。41 条不动。
  **#42k/#42k-b = CODE_LANDED**(四修复零回归;lock 重生成
  内容 SHA 未变;两 runner 键铸造;树 689P/14F/0E,44 项
  lock 掩盖失败清零,揭出弃权错判族 = pre-M0 阻断)。
  **#42l 系列收口**(6 根因全测试侧,弃权语义无恙;解释器
  伪影更正后有效树况 696P/7F/0E;LOCALIZATION 集成路由
  结构性死亡 → xfail 封存记债)。**#43 M0-C = C32 双阴性**:
  12 读数全宏负(最正 −0.00306),翻转与 PCA 可行动性均否;
  文献先验(重构族清洗正)证伪;锚复现逐位零差。条件化
  证据重定位:任务级翻转(forecast 益 vs AD 三族害)已立;
  AD 内条件化显形于机制(边界变形/训练证据侵蚀/重构失真)。
  **停止正效应探针**(第四 Consumer 与第七程序同禁)——
  identity = 本 cohort × 本菜单就绪正解,预冻结弃权分支
  生效:**M1a 合同因果 → M1b replay → 41 条弃权框架终考**。
  **sol 文档审读修正(2026-08-24 23:0x)**:原序缺"反馈有效性
  门"——插入 **#44a AD 反馈正控**(0 LLM 确定性:held-in 注入
  一种已知污染 → identity vs 已知 repair → 先证 delayed
  event-F1 真升 → 再查 Support 信号预测力;底物候选 T1 注入
  基建优先;四层分流树定 first-fault)。**总序定稿:#44a →
  #44b 合同/Context 因果考(correct/neutral/shuffled,不与生命周期混刀)
  → M1b Agent 双侧多轮 replay(正控场该动会动 + Yahoo development
  场该停会停;Memory 能表达并保留 Positive/Negative/Conflict/Abstain,
  但禁为填类别强造自然结果)→
  41 条 sealed 终考**。弃权 = Risk 能力非唯一故事;短正典
  docs/DATA_QUALITY_AND_FEEDBACK_MODEL.md 建立;
  Data_Quality_Disgussion.md 定位研究档案;AGENTS.md 状态
  陈旧提请 sol 更新。
  **#42l 在飞**(弃权错判族修复;门 = 点名失败全绿 + 零新增
  + 方法代码零非预期变化)。**sol M0 审核采纳,总序定稿**:
  #42l → **#43 M0-C consumer-flip**(0 LLM,EXPOSED 24,
  fit≤280:PCA 仪器门[窗20/full-SVD/固定rank/训练残差阈值/
  fixture 先行] + 三 Consumer 响应矩阵[iforest 每序列 120 /
  监督 v3 pooled 5 / PCA 每序列 120];主对比 iforest vs PCA,
  监督系辅助;双独立判定 FLIP_CONFIRMED 与 RECONSTRUCTION_
  HEADROOM_QUALIFIED,仅后者解锁 M1)→ M1a Context 因果
  (correct/neutral/shuffled)→ M1b 多轮生命周期 replay →
  41 条 sealed Static/A3/(合格时 A5)。Phase M 两线分工:
  本线 = M0-C,另线 = M0-Obs(a/b),M0-C 禁改 public_features
  等四处;主张口径 = "处理效用随 Consumer protocol 改变"。
- **Phase M — Model-conditioned quality**:T 闭合后,固定任务与数据,只变
  模型结构,考同一处理的 Gain/Harm 翻转与模型感知适配(M0 正控 → M1 闭环)。
  M0 配对候选(2026-08-22 由 T1b 裁定派生,届时再定稿):同一 AD 任务下
  监督型事件分类器 vs 重构型检测器(TimesNet/AE 族,本地 AegisTS 有源码)
  ——重构族把训练异常当污染、清洗预期帮它,监督族预期反号,天然的
  模型条件化翻转正控。
- **Phase X — 跨域确认(#45 重定位,2026-08-23)**:原"fresh 确认"改称
  **既有 forecasting 能力复现/回归验证**——NOAA 2025 outcome 已于
  2026-08-21(FRESH_A5_DELIVERS)一次性打开,只能 replay/development/
  回归引用,不得再称 fresh;两台 ID 修复后的证据 Runner(recall /
  fresh_confirmation)在此服役。新的 forecasting fresh 证据需另一
  outcome=SEALED 域(是否获取 = 用户决策点)。candidate v2 在此复活;
  AD family 同构重走;跨任务经验只作对照/冲突证据,不自动获得执行权。
- Phase S 状态:停车封存,资产 = candidate v2(两域证据+双向 LODO 4/4+4/4)。
  SMD 排除理由(改判,2026-08-22):多变量机器遥测的**形态不匹配**,非任务
  窄化;JUDGE_UNREADABLE 系数据-仪器形态错配,非方法失败。
- 供给挂账 O9:T6/X 需未消费单变量域;T4/T5 自然翻转证据需带标签的单变量
  AD 数据(Yahoo S5 / NAB / UCR-AD 类,曝光状态待 census);筛选标准已补
  "任务语义与实体结构必须匹配目标 Consumer"一道门。
  **O9 落定(2026-08-23)**:NAB 官方固定 ref,raw 不入库;
  Source(outcome 开放)= realAWSCloudwatch 字典序前 8 +
  realKnownCause 字典序前 6;Target = realAdExchange 六条全纳
  (CPC×3/CPM×3,聚合披露"一条正常"实例 SEALED);
  **realTraffic/realTweets 改判 INSTANCE_SEEN**(sol 检索暴露部分
  异常窗)不得作 virgin Target;Consumer = aegists_iforest_v1
  (薄接本地 AegisTS IForest 结构,借结构不搬栈)。

**AD Consumer 仪器纪律**:确定性检测器(固定参数)+ 事件级 F1(固定容差,
一对一贪心匹配)+ 既有 Support/delayed 三联窗语义 + 逐序列增益向量
(guard/选择器/RESCOPE 无改动直读);材料线与害线沿 ±0.005(预注册常数,
只经仪器审计可改);落位 evaluation/functional/consumers/(实验仪器,
不入 methods/ttha/)。
**同字节契约(T1 estimand 承重面)**:同一注入块 B、同一 Program P、
同一作用几何产出唯一 P(B),两 Consumer 读同一字节,处理侧零分叉;
预测读未来/检测读块内的不对称属任务语义,非几何混杂。
**编号口径(2026-08-22 顺延)**:#34 不存在;T1b(#38)以修复书身份并入
主链后,主任务书 = #35–#46 共 12 张:#38=T1b,#39=T3,#40=T4,#41=T5,
#42=T6,#43=M0,#44=M1,#45=X 复活,#46=最终整合。
**沉淀口径(2026-08-22,回应"不只是实验脚本"之问)**:仪器
(consumers/runner/注入协议/翻转表)是项目的永久测量资产但不是
Harness 本体;本体增长点 = T2 task_spec 观察(已落)→ T4 任务键控
冲突 Experience 入 methods/ttha(episode 键 task 分量 + 卡词汇
consumer 特征,Phase T 首次大改本体)→ T5 延迟反馈改变后续行为
→ #41b 实验层入口收敛 → #46 最终整合。章程纪律:考出 Harness
决策缺陷才动本体,故 T0–T3 先造秤、T4 起改机器;#39 的三种结局
分别指认 T4 该动哪一面。
**Phase T 合并判据与止损线(2026-08-22,sol 提议、主线定名采纳)**:
#39–#41 必须收敛到同一个可运行 Harness,否则该实现路线止损。
#41(T5)验收链 = 单一 Harness 入口 → 接收 TaskSpec / Consumer
adapter → 检索任务相关成功/失败/冲突 Experience → Agent 自主生成
Workflow → Runtime 执行并取得该任务下游反馈 → 写回 Task/Consumer
条件化 Experience → 形成/修订 Target-local Skill → 下一任务读取后
行为改变。允许变化的只有 TaskSpec、Consumer adapter、数据 Context。
禁止:写死任务→动作映射;每任务一套 Harness;runner 手工指定
Workflow;答案键代替真实执行与反馈;为过关新增 Consumer/Gate/
Schema。#41 若仍需 runner 手工拼接 → `PHASE_T_NO_HARNESS_
CAPABILITY`,止损并升用户检查点。#39–#41 三轮零新基础设施,任何
新组件先过章程之问("不加它,#39–#41 是否无法运行或解释?")。
T4 硬约束:冲突 Episode 必须经正常 Runtime 写入路径产生(重放
T1b 臂或真实执行),禁止手工插行;Context 字段 = 可观察特征,
非数据集名、非答案表。仪器处置口径:注入底物 / trainable v1/v2 /
SMD 诊断为脚手架,证据归档不删;v3 Consumer 为可复用测量资产
(T4/T5 反馈源、M0 配对),不在丢弃之列。自然阶段薄接 AegisTS/
TSLib 成熟单变量 AD Consumer 与 O9 数据(Yahoo S5/NAB/UCR 均
单变量带标签,与 T6 排期合流),不再自造分类器。
**整备书位置(2026-08-22 裁定)**:实验层沉积(96 个 run_e2 runner、
主管线 5000+ 行叠 V3→V9 清单、仪器分叉)不在机制轮中途重构;唯一整备
窗口 = **#41(T5)收口之后、#42(T6)fresh 冻结之前**,以修复书 #41b
执行,行为保持 + 重放验证(重跑 2 个指定历史 runner 逐字节比对工件,
有差异即整体回退),产出 V10 后 T6/X 两次 fresh 均在整备后代码上冻结
与打开;禁止落在任何冻结与打开之间。范围五项封顶:现役评估入口抽取、
仪器名册(现役/化石取代关系)、V3–V8 死清单移出活管线归档、
RUNNER_INDEX(入口 runner vs 化石 runner,零删除)、重放验证。
**Supersession(2026-08-23)**:五项方案撤回——A1 前提过时(评估
入口已在 online_loop.py:289/:722 独立承载)、重放预算不可兑现
(t1b --v3 实测 78 AD 评估)、余项不阻塞 T6;整备窗口以
**#41b-lite** 兑现(--really-refresh 检查点 + 四项验证 + 最小 V10,
experience_memory.py 纳入清单,0 LLM/0 重训/0 AD 评估);
名册/RUNNER_INDEX/V3–V8 归档延后 #46;键方言只登记;
bundle-sha 不扩依赖(违反反过度工程条款)。
0 LLM,不计方法进展,报告作附注。**提前触发条款**:T5 前任一书在
沉积层本身上摔倒(import 断裂、仪器分叉误接、冻结清单自相矛盾、
first-fault 因 runner 纠缠无法归因)即构成仪器层 first fault,
可把 #41b 提前为有界修复,范围不变。**即刻站规**:每书 ≤1 新 runner、
复用现役评分件/窗口/阶梯;新仪器只进 consumers/ 且创建分叉的同轮
必须在台账落一行取代关系(trainable_v1/v2 的取代关系在 T1b 报告
落地时补记)。
**Phase T 前段预算(预注册)**:T0 = 0 LLM / 0 forecasting 重训 /
≤200 AD 评估;T1 = 0 LLM / ≤60 forecasting 重训 / ≤300 AD 评估;
T2 起随书预注册。冻结面口径 = 现行 FROZEN_SURFACE(当前 V9,39 项)
+ git diff,#35 v2 书中"v7 注册表"字样按此勘正;#41b-lite 起为
V10(40 项,含 experience_memory.py)。

**主线操作纪律(2026-08-23,由当日三次折返的共同病根立规——
"拿名义规范当事实,不核实测状态")**:
1. 预算一律从工件实测成本推导(如 t1b --v3 = 78 AD 评估、T5 实测
   ~5 call/轮),禁按名义步数拍;重试/validation 流量计入常规。
2. 合法性门与形状契约一律从 estimand 与仪器实际输入契约推导
   (Consumer 按行序开窗就不得要求时间戳严格递增),禁审美门。
3. 释放任何冻结路径前,主线必须核验:封存状态 + 协议冻结 +
   **可执行体真实存在且非占位**;"已实现"的报告语句须行级证据。
4. 机械修复快车道:纯机械、可自测的仪器修复(stub 清除、占位数据
   替换、烟测断言修正、字面量改 import)授权执行方修完自测通过
   直接续跑,只报结果不逐步等裁定;方法面、协议面、封存面、
   预算面改动仍需主线裁定后动手。
5. 接口/ID/键统一类改动的兼容修复清单必须包含**所有承载在册
   证据的 Runner 与入口**(以台账证据行反查),不得只修当轮
   报错项;修复后以零消耗缓存重放证明旧正例在新接口下仍复现
   (2026-08-23,#42d r2 由两 forecasting 证据 Runner 手写旧 ID
   缺口立规)。

**文档更新纪律(2026-08-23 明文化)**:两文档制,不新建第三份
长期文档。台账(STAGE_REPORT)= 事件驱动:每次裁定当轮写入
结果、歧义裁定与教训,不隔夜、不攒批。roadmap = 状态驱动:
进度线每轮裁定后同步;常设规则只在被 supersede 或新增站规时改,
supersession 同轮标注、原文不删。执行方与子 Agent 不改这两份
文档,只报告;主线是唯一写入者。工件内主线只允许触碰
release 类字段且必须同轮登账。

**附属书编号(2026-08-23)**:主书号 #35–#46 不变;修复/补全/
收尾书以字母后缀挂靠主书(#41b-lite、#42a、#42b),不占主号、
不改主链计数。

**#42 正判后的复核预留**:NATURAL/provisional 升格需第二自然
单变量域独立复核(候选 Yahoo S5 / UCR-AD,获取时同走曝光
census、结构门与行序契约);排期在 #42 出判定后与 #43(M0)
一并定夺,不预建仪器。

## 4. 保留给用户的决策点

1. Phase S 的第三域选择与总预算(S0 盘点报告出来后拍板)。
2. 论文化时点:何时把冻结 claim 表转成论文骨架(建议 Phase R 收尾或 Phase S S3 后)。
3. Opus 复跑的优先级(纯后端读数,可无限期搁置)。
