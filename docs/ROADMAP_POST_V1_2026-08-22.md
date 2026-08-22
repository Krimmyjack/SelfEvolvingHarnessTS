# Post-V1 路线图(2026-08-22 冻结)

本文件是 V1 里程碑(v7 九环闭合)之后的总体安排,供主线、执行 Agent 与外审共同使用。
逐轮台账见 `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`(canonical,遇冲突以台账为准)。

## 0. 已入账状态(带工件指针,不得复述超出其口径)

| 结论 | 口径上限 | 工件 |
|---|---|---|
| FRESH_A5_DELIVERS | pooled 首正成本 −43.9%(69 vs 123 重训),最终质量与 harm 同冷启动;per_channel = A5_TIE_TRANSFER_BOUNDARY;NOAA held-out 2025 一次性打开 | `fresh_confirmation_v1.*` + `fresh_confirmation_v1_adjudication.md`(canonical 措辞) |
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

**项目范围(用户裁定)**:数据形态限定单变量;Task/Consumer、模型、Domain、
Pattern 全部可变。质量标准随任务/模型/模式变化是第一性命题;forecasting 线
的全部已入账证据只覆盖一个 family。

**阶段图**:
- **Phase T — Task-conditioned quality**(当前主线):同一单变量 Pattern 下
  forecasting 与 anomaly detection 双 Consumer 的方向翻转与条件化适配。
  T0 仪器定义+底物普查(#35)→ T1 注入正控(#36,POSITIVE_CONTROL 等级)
  → T2 TaskSpec/Consumer 观察接线审计 → T3 任务条件化决策(Agent 平权
  双任务,禁 Router 硬编码)→ T4 冲突 Experience 写入与按任务检索 →
  T5 生命周期闭环(反馈改变下一任务行为)→ T6 fresh 跨域确认(需新域,
  见 O9)。机制阶段 T0–T5 全部用注入正控 + 自有 dev 数据,不需新数据。
  进度(2026-08-22):T0 = `T0_READY`(#35,AD Consumer 冻结于 49/3.5,
  回退已用尽);T1 = `TASK_FLIP_CONFIRMED_POSITIVE_CONTROL`(#36,C12,
  4/4 程序同向,guard 语法对 AD 向量直读通过);当前 = T2(#37)。
- **Phase M — Model-conditioned quality**:T 闭合后,固定任务与数据,只变
  模型结构,考同一处理的 Gain/Harm 翻转与模型感知适配(M0 正控 → M1 闭环)。
- **Phase X — 跨域 fresh 确认**:各 family 内 A5 vs A3(forecasting 线的
  candidate v2 在此复活;AD family 同构重走)。跨任务经验只作对照/冲突证据,
  不自动获得执行权。
- Phase S 状态:停车封存,资产 = candidate v2(两域证据+双向 LODO 4/4+4/4)。
  SMD 排除理由(改判,2026-08-22):多变量机器遥测的**形态不匹配**,非任务
  窄化;JUDGE_UNREADABLE 系数据-仪器形态错配,非方法失败。
- 供给挂账 O9:T6/X 需未消费单变量域;T4/T5 自然翻转证据需带标签的单变量
  AD 数据(Yahoo S5 / NAB / UCR-AD 类,曝光状态待 census);筛选标准已补
  "任务语义与实体结构必须匹配目标 Consumer"一道门。

**AD Consumer 仪器纪律**:确定性检测器(固定参数)+ 事件级 F1(固定容差,
一对一贪心匹配)+ 既有 Support/delayed 三联窗语义 + 逐序列增益向量
(guard/选择器/RESCOPE 无改动直读);材料线与害线沿 ±0.005(预注册常数,
只经仪器审计可改);落位 evaluation/functional/consumers/(实验仪器,
不入 methods/ttha/)。
**同字节契约(T1 estimand 承重面)**:同一注入块 B、同一 Program P、
同一作用几何产出唯一 P(B),两 Consumer 读同一字节,处理侧零分叉;
预测读未来/检测读块内的不对称属任务语义,非几何混杂。
**编号口径**:#34 不存在,主任务书 = #35–#45 共 11 张;有界修复书另计。
**Phase T 前段预算(预注册)**:T0 = 0 LLM / 0 forecasting 重训 /
≤200 AD 评估;T1 = 0 LLM / ≤60 forecasting 重训 / ≤300 AD 评估;
T2 起随书预注册。冻结面口径 = 现行 FROZEN_SURFACE(当前 V9,39 项)
+ git diff,#35 v2 书中"v7 注册表"字样按此勘正。

## 4. 保留给用户的决策点

1. Phase S 的第三域选择与总预算(S0 盘点报告出来后拍板)。
2. 论文化时点:何时把冻结 claim 表转成论文骨架(建议 Phase R 收尾或 Phase S S3 后)。
3. Opus 复跑的优先级(纯后端读数,可无限期搁置)。
