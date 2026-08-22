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

### Phase S — Shared Capability 跨域(下一个方法里程碑)
按章程:Shared Capability 需多 Domain 重复正向+风险证据,严格 Promotion 只适用此层。
- S0 域盘点(0 LLM):列出未被本线 outcome 消费的候选第三域(electricity/weather/
  traffic 各自的曝光状态查台账与旧线报告),给出 context_exposure / outcome_exposure 标注。
- S1 健康检查(development 区,不开 outcome):结构、长度、缺陷 prevalence;
  判定 PROCEED / STOP_FOR_LOW_PREVALENCE。
- S2 候选编译(0 LLM 确定性):从 NOAA + 旧线 traffic 的正/负/冲突 Experience 归纳
  Shared Capability candidate(含 guard 作为风险面),冻结 version;
  凡与决策相关的 Context 必须是部署时可观察特征,禁止 Dataset 名称做相似性理由。
- S3 development 试运行:A5'(带 Shared candidate)vs A3'(空 Source)同预算,
  与 #17 同口径的成本/质量/harm 读数;判定后才允许 S4。
- S4 冻结确认:roster/程序/Judge 冻结后一次性打开第三域 held-out outcome
  (fresh 纪律,`context_exposure`/`outcome_exposure` 注明)。
- 每步一书,预算逐书封顶;S 阶段总重训预算建议 ≤600,由用户确认。

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

## 4. 保留给用户的决策点

1. Phase S 的第三域选择与总预算(S0 盘点报告出来后拍板)。
2. 论文化时点:何时把冻结 claim 表转成论文骨架(建议 Phase R 收尾或 Phase S S3 后)。
3. Opus 复跑的优先级(纯后端读数,可无限期搁置)。
