# 阶段报告:批次配方线 — 冻结 claim / 开放问题 / 仪器事实

重构日期:2026-08-22(Phase C)。**这是新会话 / 新 Agent 接手本线的第一阅读件。**

本次重构**不新增任何结论**,只把已有内容重排为三张表;逐轮流水账原文一字未删,
移至文末第 4 节「历史台账」。遇口径冲突,以本文件第 1 节的 canonical 措辞为准;
遇本文件与工件冲突,以工件为准(每条都给了指针,请直接核)。
规划与纪律见 `docs/ROADMAP_POST_V1_2026-08-22.md`。

**证据等级四档**(常备纪律第 6 条):
`INSTRUMENT` 仪器自证 / `MECHANISM` 银行重放 / `DEVELOPMENT` 已曝光窗 live /
`FRESH` sealed outcome 一次性打开(保留字)。

---

## 1. 冻结 claim 表

每行 = 结论 / 口径上限(canonical 措辞,不得超出复述)/ 证据等级 / 工件指针 / 已知 caveat。

### 1.1 里程碑结论

| # | 结论 | 口径上限(canonical) | 等级 | 工件 |
|---|---|---|---|---|
| C1 | `FRESH_A5_DELIVERS` | pooled 首个正采纳成本 −43.9%(69 vs 123 重训);最终质量与 harm 同冷启动;per_channel = `A5_TIE_TRANSFER_BOUNDARY`;NOAA held-out 2025 一次性打开 | FRESH | `fresh_confirmation_v1.*` + `fresh_confirmation_v1_adjudication.md`(裁定附录为 canonical 措辞) |
| C2 | `LOCAL_LIFECYCLE_CLOSES` | Target-local Skill 形成 / 晋级 / 持久化 / 召回四步成立 | DEVELOPMENT | `local_skill_recall_v1.*` |
| C3 | `SLOW_CLOSES_SCOPE_GAP_BY_VETO` | 银行重放;**containment,不是效用改善**——伤害清零的同时聚合增益一并放弃 | MECHANISM | `slow_scope_update_v2.*` |
| C4 | `BANKED_CHAIN_CLOSES_IN_K_MODELS` 2/2 | 6–9 环端到端,`gpt-5.6-sol` 与 `gpt-5.6-luna` 各一次;两模型首抽即提出同一 guard,0 次信封重试 | MECHANISM | `operational_pipeline_v6.*` |
| C5 | `DEVELOPMENT_OPERATIONAL_PIPELINE_CLOSES_POST_FIX_ON_GPT_5_6_SOL` | 九环一次连续、无人接力闭合;行为改变判据 (i) 现场成立(guard 读 −0.102763 开火 → identity) | DEVELOPMENT | `operational_pipeline_v7.*` |
| C6 | `RESCOPE_PRESERVES_GAIN_ELIMINATES_HARM` | 确定性三臂:task_C 无 guard +0.0297 带 1 受害 / VETO 0 / **RESCOPE +0.0611 零受害剔 1**;task_D +0.0495 带 2 受害 / 0 / **+0.0959 零受害剔 2** | MECHANISM | `operational_pipeline_v8.*` |
| C7 | `LIVE_RESCOPE_CONTAINS_WITHOUT_COLLATERAL` | live 单轨,`gpt-5.6-sol`:task_D 路由恰为两条越线序列,+0.049504 带 2 受害 → **+0.095879 零受害**;保留的两条序列读数逐位不动;VETO 孪生同轨为 identity 0.0 | DEVELOPMENT | `operational_pipeline_v10.*` |
| C12 | `TASK_FLIP_CONFIRMED_POSITIVE_CONTROL`(raw);**ADJUDICATED_ESTIMAND = INPUT_SIDE_TASK_FLIP** | 同字节契约下(同一注入训练块、同一 Program 施用恰一次、两 Consumer 消费同一字节,断言 600/600),离群修复族 **4/4 程序同向翻转**:forecasting delayed 聚合 +0.0648~+0.4059,AD 聚合 −0.0455~−0.2808;镜像方向 0 例;identity 基线不退化(F1 0.6667)。**改判(2026-08-22)**:F 侧为训练数据效用、AD 侧为推理输入效用(AD Consumer 零训练直接检测 P(B)),消费模式不对称——本行只承载"清洗推理输入抹除检测信号",训练侧翻转由 T1b(#38)另证 | POSITIVE_CONTROL | `t1_flip_control_v1.*` |
| C13 | `TRAINING_SIDE_TASK_FLIP_CONFIRMED_POSITIVE_CONTROL` | 双 Consumer 都在同一 P(B) 上训练(C2 同字节 600/600)、固定未处理独立 Qf 计分:**winsorize 臂 forecasting delayed +0.4059(四 eval 序列全正)且 AD 宏 F1 增益 −0.1672(≈8 事件,9/12 序列为负)**——同一臂恰为 forecasting 最强臂;iqr/mad/hampel 双侧同向为正(+0.0086/+0.0413/+0.0413),**翻转为程序特异**,非"清洗必伤 AD";门 0.6109 一次过(v1→v2→v3 = 0.1709→0.1765→0.6109,解锁在特征族);AUPRC 五臂全同(0.8878,单特征正斜率下排序≡z)→ 翻转全部承载在判决边界位置;双句口径注记强制随行 | POSITIVE_CONTROL | `t1b_training_flip_v3.*` |
| C14 | `TASK_CONDITIONED_PROPOSALS_CONFIRMED` | 空店(0 Guidance/0 Experience/0 learned Skill + 3 个任务中立 h0 bootstrap 常量,六 draw 同快照)、零 Outcome 泄漏、prompt 仅 task_spec 字节差(剥除后两臂 canonical sha 相同 96ce9fc9…)下,6/6 有效 draw **完全分离**:跨任务 9 对 Jaccard 距离全 1.0 > 同任务最大 0.5;聚合方向 3/3+3/3(F top-1 hampel_filter×3,AD top-1 identity×3);Risk 层 F 0/3(hampel 最差序列 −0.0904 出局)/ AD 3/3——空店下 Risk 不达为预注册预期,构成 T4 入口证据;**第二次抽样**(首跑 mappingproxy 写盘崩溃致 draw 1–4 丢失;暴露片段 F5=outlier_mad/AD6=identity 方向一致且风险层更优,无择果签名,主线裁定有效;LLM 12/12 收口);只证提案条件化,不含执行/采纳,任务语义部署可见 | POSITIVE_CONTROL | `t3_task_exam_v1.*` |
| C15 | `PARTIAL_EXPERIENCE_CONDITIONING` | 10 条 T1b 重放 Episode 经 Runtime 真实写入(同 TTHAMethod 读回,DELAYED / EPISODE_ONLY / 不晋级不授执行权)后重跑 #39 考试:**F Risk 0/3→3/3**(hampel→outlier_iqr×3,冲突卡直接对应,draw-3 原话在档);**AD Risk 3/3→0/3 回退**(identity→hampel×3);分离增强(min 跨 1.0 > max 同 0.0);键/写入/检索门全绿(B4 7/7、C2 三向 7/7、零跨任务取卡);回退机制定位 = **卡表达范围**——identity 判 ABSTAIN 后 ContrastPack 三格无处安放、_hard_filter 剔除、孤立"聚合改善"卡无对照即误导(AD 三 draw reason 逐字为证);证明 Memory 机器能承载风险更正,不证明 Agent 发现新知 | POSITIVE_CONTROL | `t4_conflict_experience_v1.*` |
| C16 | `TASK_SEPARATION_REGRESSION`(raw,预注册优先级第 3 格);**T4 收束裁定 = Memory 机制部分成立,安全闭环未完成** | abstain 通道(Runtime 天然可达:relation=ABSTAIN ∧ 签名 identity 只豁免 informative-operator membership,其余过滤照过;identity 实测不在 Operator registry,豁免为通道可达的必要条件)使 **AD Risk 0/3→2/3**(两 draw 逐字引用 no-action baseline),**F 3/3 保持**;分离门失效诊断 = `identity` 为两任务共有合法弃权词同时进两臂 shortlist(top-1 层完全不相交,6/9 跨对仍 1.0)——非任务趋同;判据教训:未来提案考试 top-1 分离为主、shortlist 为副并设共有弃权例外;不做 T4c 卡序第三修(防调成答案 Router);AD6 hampel 提案的正确归宿 = T5 真实执行权门 | POSITIVE_CONTROL | `t4_conflict_experience_v2.*` |
| C17 | `INCOMPLETE_LLM_BUDGET`(T5 #41 raw;定性 = 部分成功 + 协议预算缺陷,非 Harness 失败,主线认领算术错误) | **已证**:统一 Harness 单入口真实运行双 TaskSpec/Consumer(零第二套、Runner 零指定 Workflow——止损线首考存活);风险接线生效(三次 delayed 全 CONFLICT 全拒,聚合正逐序列害不激活 Skill;B2/B6 撤权格 22/22);F r1 自写 Experience 改变 F r2(避开已知有害 outlier_mad、probe 2→1,r1/r2 TaskSpec/Context/Consumer 字节相同,归因断言 2/2);任务键零串写零 Skill 泄漏。**未证**:AD r2(预算 16 处斩)、真实轨迹 Skill 激活与复用(仅 Part B 脚本层)、Memory 引导直选安全计划(r2 重选已冲突 hampel 靠 Runtime 门拦;全程未探 iqr/winsorize——探索为先验支持贪心,剪坏不引新,只入册不授权修) | POSITIVE_CONTROL | `t5_lifecycle_v1.*` @ 687af6e |
| C18 | `T6_NATURAL_PLAN_READY`(#42a):自然 NAB Source 在 **delayed 层独立集齐三类** Action–Response(POSITIVE ×2 / NEGATIVE ×4 / CONFLICT ×1 + identity ABSTAIN ×4),零 Target outcome 读取下冻结 A5-vs-A3 | row-order 契约 20/20 无损通过(行数/值逐元素等同,零特例);29/29 官方窗口全映射;20 条 Episode 走真实写入路径全 EPISODE_ONLY;**自然数据自带跨 cohort 效用翻转**(winsorize:aws 双轮 POSITIVE 0/8 害 vs known r2 NEGATIVE −0.0747 3/6 害;hampel:aws r1 NEGATIVE vs known r2 CONFLICT 聚合+0.0400 2/6 害)= 项目核心前提的自然实证;Support→delayed 翻转 8 次单列 = Support 单层不可作批准信号的自然证据;确定性复核逐位相同;AD fit 152/200,LLM 0 | NATURAL / provisional(单 Target 域双 cohort,不声称普适跨域) | `t6_nab_frozen_plan_v2.*`(v1 保留为契约诊断) |
| C19 | 正式 A5vsA3 自然评估:盘面判定 `SOURCE_EXPERIENCE_SAFER_NOT_FASTER`,**强制双样本注记**(启动竞态产生影子完整样本 = `NO_SOURCE_EXPERIENCE_ADVANTAGE`,后写者覆盖;执行方未选样、双份如实报);**承重取两样本交集:Source 经验未加速 Target 适配(一致)、无负迁移(一致)、安全优势 1/2 样本不稳** | **附带历史性正结果:首次自然数据完整生命周期闭合**(CPM A3 r2:空店自主提案 outlier_mad,Support +0.0593/3 序列 0 害 → Draft → delayed +0.0111 POSITIVE → LOCAL_ACTIVE)= T5 未证项自然兑现;CPC A3 r1 探索→负例写回→r2 收敛 = Target 内学习自然生效;**机制(第一阻塞证据)**:A5 四 cell 检索渲染 20 卡(sha 稳定)→ 提案池全坍缩 identity-only、0 次非 identity 试验、0 Target 反馈消耗——跨域负例卡转译为全局保守而非 context 条件化谨慎;墙两次未破,合计 LLM 45/48、fit 27/120 | NATURAL / provisional | `t6_nab_evaluate_v2.json`(影子样本仅存执行方读数摘录) |

### 1.2 反复观察到的机制事实

| # | 事实 | 口径上限(canonical) | 计数 | 工件 |
|---|---|---|---|---|
| C8 | 聚合观察粒度看不见该故障类 | 同一 episode:逐序列读数 → `RISK_GAP` at `OUTCOME_RISK`;纯聚合读数 → `NO_ACTIONABLE_FAULT`。**该故障类在聚合观察粒度下不可见的直接证据** | **6 个不同 episode 情形,16 条机器记录**:`a5_pooled`(#18/#19)、#21 task_C、#23 T1/T2/T3、live task_C(v7/v9/v10 为同窗同方案的确定性重复,计 1)。差额全部是逐位复现的重放 | `slow_scope_update_v1/v2.*`、`operational_pipeline_v1/v4/v5/v6/v7/v9/v10.*` |
| C9 | 两钥匙(弱晋级 + 强当窗确认 + v2 门)是实测承重件 | 三次探针晋级全在噪声带(g/se 1.09 / 0.29 / 0.56);随后 5 次复用尝试被拦(3 次当窗确认失败、2 次确认通过但 v2 门拦)+ 1 次合法放行且保持正向;**零次复用越过 aggregate harm 门** | 6 次复用尝试 | `fresh_confirmation_v1.*` |
| C10 | 引用 ≠ 遵从 | Agent 自称引用的 Guidance 条款与它实际提出的 shortlist 之间没有强制关系;`skill_clause_use` 只记自称,无校验 | 机器字段 `clause_shortlist_overlap` 有 **6 条读数,overlap 分别为 1,1,1,2,1,1——全部 ≥1**。首个**零重叠**样本(#22,n=3:两次重叠且成功 +0.306,一次背离且停摆)**早于该字段,只存在于第 4 节散文里,无机器记录** | `operational_pipeline_v3/v7/v9/v10.*` |
| C11 | shortlist 对抽样敏感 | 同窗、同卡、同配置的 FULL_PRICE task_A 抽样共 7 次,得到 **3 种不同 shortlist** | `[repair_level_shift, outlier_mad]` ×3(v1/v9/v10);`[repair_level_shift, outlier_iqr]` ×3(v3 的 K=3 三轨,轨间一致);`[outlier_mad, outlier_iqr]` ×1(v7)。阶梯路径:`GATE_PASS_ADOPT_NAMED` ×5、`GATE_FAIL_FALLBACK_SUPPORT_WINNER` ×2 | `operational_pipeline_v1/v3/v7/v9/v10.*` |

### 1.3 全表通用 caveat

1. **同窗选择**:guard 读 delayed 窗,它产生的否决/路由也在同一个 delayed 窗上计分——VETO 与 RESCOPE 同构。C6/C7 **没有**证明在一个窗上选出的路由能在另一个窗上存活;本线至今未为此开过 held-out 窗。
2. **聚合抬升是算术必然**:`aggregate_gain` 是逐序列向量的均值,去掉负项当然抬升。RESCOPE 臂高于"无 guard"臂不是白得的收益——被路由的序列服务 identity,得零。
3. **后端标注**:C5/C7 是 `gpt-5.6-sol` 上的读数,C4 是 sol + luna。任何"后端无关"的说法都没有证据。
4. **n 与独立性**:每 cell / 每轨迹 n=1,无重复无区间;C4 的 2/2、C8 的 16 条、C11 的 7 次里,凡标注为重放或同窗重复的都**不是独立观察**。C1 的成本对比中,"协议补全的乐观反事实 ≈99 vs 144" 是推算不是实测;首正读数 69 vs 123 不受此影响。
5. **DEVELOPMENT ≠ FRESH**:C2/C5/C7 的窗口 outcome 由 #17 一次性打开过,这些轮次不产生新的 fresh 证据,也不产生新的 A5>A3 结果。
6. **BY_VETO 的"无殃及"**只指不相关 episode 的决策逐位不变,不表示 guard 免费(C3)。保收益的措辞只属于 RESCOPE(C6/C7)。
7. **POSITIVE_CONTROL 等级(C12)**:翻转由注入构造,AD 的 ground truth 恰是修复类程序可移除的事件——它验证的是仪器链能读到任务条件化翻转,**不构成自然数据上存在该翻转的声明**;自然翻转证据承重在后续自然标签轮。C12 的增益幅度与现役逐窗菜单增益**不可比**(P 整块施用,全局统计算子统计域 780 点 vs 逐窗 240 点),只承载方向。**估计对象不对称(2026-08-22 改判)**:T1 的 AD 零训练、直接检测 P(B),C12 只承载输入侧;输入侧本身是真实部署场景(流式清洗后检测),资产保留不弃;训练侧由 T1b 检验。
8. **C13 承载面与外推禁令**:v3 Consumer 单特征、正斜率,五臂 Qf 排序恒等(AUPRC 全同 0.8878),训练侧翻转全部表现在学到的判决边界位置,不表现在排序层;更丰富的 Consumer 可能两层都动,本读数不外推。翻转为程序特异(仅 winsorize),不支持"任一清洗都伤 AD 训练"的泛化。双句口径注记(阈值头族发声 + 不证明 Harness 自发现表示/不声称自然泛化)永久随 C13。

---

## 2. 开放问题表

| # | 问题 | 现状 | 卡在谁 | 指针 |
|---|---|---|---|---|
| O1 | Phase S 第三域 + 总预算(**已了结,#30–#33**) | S0/S1/S2/S1b 已跑:SMD 唯一候选,S2 编译过,S1b JUDGE_UNREADABLE 且对门槛修正稳健 → SMD 关闭,Phase S 停车封存(candidate v2 冻结);第三域供给并入 O9 | 无(Phase X 复活时再开) | 本文件范围锁定段;`shared_capability_candidate_v2.*` |
| O2 | per_channel 迁移边界 | #17 判 `A5_TIE_TRANSFER_BOUNDARY`;停车场项 | 需先有"什么 Observation 能区分 per-channel Context"的假设 | `fresh_confirmation_v1.*`;ROADMAP 停车场 |
| O3 | 供给面自堵(菜单可采纳率) | #22 出现过一次"shortlist 无一可采纳 → 诚实 identity";属仪器按设计弃权,非故障 | 留观察,shortlist 稳定性表(C11)随轮追加 | `operational_pipeline_v2.*` |
| O4 | `SELECTION_MISS` 适配器窄口径 | 在册缺陷,已两次兑现(#18 a3_pooled、#23);**冻结中,未修** | 修复后需跑一次 0-LLM 归因回归 | ROADMAP §契约性收尾 |
| O5 | Opus 读数 | 中继宕机已证实为上游(HTTP 200 + error payload);fix (c) 已上线;carry-in 已清零 | 随时可跑,≈1 LLM,不阻塞任何 claim | `operational_pipeline_v5/v6.*` |
| O6 | **纯 clone 不可导入** | 字节口径已修(`.gitattributes` 的 lock `-text`,新 checkout 实测通过 `stage_p2_precondition`);但 `SelfEvolvingHarnessTS/` 自指 symlink **未入库**,`git ls-files` 0 条,纯 clone 连 import 都过不去 | backlog | 本文件 §3.6 |
| O7 | 同窗选择的 held-out 化 | C6/C7 的核心 caveat;需一个"guard 在 A 窗选出、在 B 窗计分"的设计 | 未立项 | 本文件 §1.3 第 1 条 |
| O8 | 上一检查点漏提交 2 文件(**已闭合,#30 Part 0**) | `compiler.py` 与 `h0/snapshot.lock.json` 已入库;#35 A2 实测 `git ls-files --error-unmatch` 两条均命中(#35 勘误 (b) 据此撤回) | 无 | 本文件 §3.6 |
| O9 | **Phase T/X 数据供给** | T6/X 需未消费单变量 fresh 域;T4/T5 自然翻转证据需带标签单变量 AD 数据(Yahoo S5 / NAB / UCR-AD 类,曝光状态待 census);census 标准已补"任务语义与实体结构匹配 Consumer"门 | 最早 #39 承重,T0–T3 不阻塞 | 本文件范围锁定段;ROADMAP §3.5 |

---

## 3. 仪器事实表

### 3.1 记录家族:四例与根治规则

同一个病:**记录写晚了、写错地方、或被过滤掉**,导致工件缺字段或字段名不副实。

| 例 | 症状 | 根治 |
|---|---|---|
| 一 | `stage_slow` 先抛后记,原因码丢失(#18/#21) | 进入阶段前把 sink 挂到 payload 上,阶段内边写边记 |
| 二 | `_public()` 按名过滤,把整个 store/card 区块删掉(#21 v2) | 改为下划线 + 类型排除,不按名字过滤 |
| 三 | `payload["trajectory"]` 在 task_C 之后才赋值,早停即丢(#22) | **每个阶段的容器在进入该阶段之前挂载** |
| 四 | `per_eval_series_delayed_after_gate` 在发生路由时取自冻结读者重测的**未路由**方案,与同一行的 `harmed_after_gate` / `delayed_after_gate` 自相矛盾(#28 v10 task_D),并多花约 3 次重训 | 路由发生时直接取 gate 投影收据(`_after_gate_per_series`) |

**根治规则**:阶段进入前挂载记录容器;**凡冻结读者产出的字段都是 pre-route 量,必须如此命名或归档**(代码常量 `FROZEN_READER_RULE`)。

### 3.2 契约链:三例与升级条款

同一个病:**同一份契约被两处各存一份,宽的那处放行、严的那处拒绝,且拒绝点没有重试**。

| 例 | 缺口 | 修法 |
|---|---|---|
| 一 | 信封校验器缺 `maxLength`(与自身 `minLength` 不对称),EditController 有 → `COMPILER_REJECTS`(#21) | `schema_contracts._validate_local_schema` 补 `maxLength`(#22 A1) |
| 二 | `_open_stores_v2` 硬断言 dependency drift 恰为两键,后续每加一个依赖就多偏一键 | 未修:该门已不再运行,#21 结果以 precondition 形式前推 |
| 三 | 信封说 `predicted_agent_behavior_change.items` 是 `{"type":"string","minLength":1}`,EditController 按 `behavior_predicate_v1` 校验 → Agent 把词表里的 `<matching ^…$>` 原样抄进 manifest,信封 0 重试放行,`apply_to_fork` 拒绝(#28 v9) | 信封 items 直接复用 `load_stage_schema("slow_edit_v1")["$defs"]["behavior_predicate"]`;`_behavior_vocabulary()` 只留可照抄的枚举值,正则族移入 `_behavior_patterns()` 并带**由 pattern 自身生成并回验**的示例(#29) |

**升级条款(用户 2026-08-22 下达)**:若再暴露一处**独立的** Proposal/Manifest 契约错位,不得继续逐字段打补丁——应判定这套自定义简化 Proposal Schema 重复拥有契约,整体改为复用真实 Schema。

### 3.3 选择器语义

`select_scope_risk_episode`:在已完成 / 已采纳 / delayed 已揭示的 episode 中,取**最早**一个"逐 eval 序列 delayed 增益最小值 < −0.005"者;都不满足时返回 `NO_ELIGIBLE_SCOPE_RISK_EPISODE`。
已验证**不是写死在 task_A 上**:#23 选中 task_A,v7/v9/v10 中 task_A 干净(min +0.017726)、task_B 采纳 identity,选择器自行走到 task_C。

### 3.4 三计数制

`valid_decision_samples`(≤2)/ `protocol_failed_draws`(独立累计上限 2)/ `llm_calls_spent`。
transport 失败**两者都不消耗**,记 `INCONCLUSIVE_TRANSPORT`;得提案即止,不重掷。
`claude-opus-5` 的 carry-in 已由 2 清零,注记「重分类 `INCONCLUSIVE_TRANSPORT`×2(上游宕机证实)」。

### 3.5 信封错误分类学

`ENVELOPE_PROTOCOL`(消耗一次 protocol_failed_draw)/ `COMPILER_CONTROLLER`(编辑真的到过控制器才写 `COMPILER_REJECTS`)/ `TRANSPORT`(连续 3 次 → `INCONCLUSIVE_TRANSPORT`)/ `RUNTIME`。
**中继坑**:agicto 中继会以 HTTP 200 返回 `{"error":{…"Service load is too high"…}}` 且无 `choices`。SDK 视为成功,后端曾据此构造空 `AgentResponse`,重试层无异常可重试,于是把上游宕机记成 Agent 协议失败(#24/#25 两次)。fix (c) 在 `runtime/agent_backend.py` 把该形状抬为 `AgentTransportError`。

### 3.6 冻结面 / 检查点惯例

- 冻结面清单跑前 `_freeze()`、跑后 `_verify()`,漂移即 `CONCURRENT_WRITE_ABORT`;当前 39 项。
- **哈希口径 = 工作树字节**。`core.autocrlf=true` + `* text=auto` 会让 checkout 与工作树不一致;`h0/snapshot.lock.json` 因此在 `.gitattributes` 里标了 `-text`,新 checkout 实测已能通过 `stage_p2_precondition`。
- **依赖链**:`compiler.py` / `harness_surfaces.json` / `runtime/agent_backend.py` / `methods/ttha/schema_contracts.py` 任一变动 → `dependency_shas` → `runtime_bundle_sha` → 必须重生成 h0 lock,并同步 `P1_POST_SHA`。每次只应有**一个** dependency 键移动,`harness_content_sha` 不得变。
- 交付不 commit,统一检查点、显式 `add`(不用 `add -A`);`v*` 原档只增不改。
- `SelfEvolvingHarnessTS/` 是指向仓库根的自指 symlink 且**未入库**;git 跟踪的是裸路径。

**仪器规则第五条(验收夹具自钉)**:验收夹具**必须自钉快照或自行回卷**——
复制被测 store 后立刻把 active 指回工件里记录的那个 sha,断言写在自己钉住的
状态上;**禁止读活的可变 scratch 目录**。两次教训:#25 的 A1 夹具因签名漂移
静默失效(`fake_store` 缺 `label`、`fake_slow` 缺 `fault_step`,注入的中止从未
触发却"通过"),#29 的契约夹具因 v10 已把 `rescope_live` 打了补丁,
`guards 0 → 1` 变成 `1 → 1` 而红——被测代码没有回归,红的是夹具的环境耦合。
推论:**夹具本身也要有"会红"的证据**——注入一次必失败的输入,确认它确实红。

---

## 4. 历史台账(2026-08-22 重构前全文,一字未删)

以下为本文件重构前的完整原文,逐字保留。上面三节是它的索引,不是它的替代:
凡上面没写进去的细节——尤其逐轮的失败、改判、以及被否决的解释——都在这里。

# 阶段报告:批次配方线(Batch Recipe Line)

日期:2026-08-21
范围:2026-08-20 ~ 08-21 的配方线工作(组合头寸 → 掩码搜索 → 配方工具 v1/v2 → Consumer 条件化 → 窗口外复核 → Agent 挂载 → 工具对照 → 经验冷暖对照与 LOCO 轮换)。
性质:按 AGENTS.md §11 的阶段性交付。所有数字均出自逐项审计过的冻结工件;代码栈经独立只读审查(`code_review_batch_recipe_stack_v1.md`,裁定 NUMBERS_TRUSTWORTHY)。

## 1. 方法 / Harness 行为发生了什么改变

- 新增一个确定性 0-LLM **批次配方工具**(`run_batch_composition_headroom.py --mode recipe`):菜单扫描 → 掩码贪心剔除搜索 → 冻结采纳规则。规则经一次带版本号的修正(v1 → v2:identity 成为延迟窗口在位者),修正由活体失败案例驱动并归档。
- 配方工具获得 **Consumer 结构条件化**能力(pooled / per_channel 变体,仅存在于实验 runner)。
- Fast Agent 的 Workspace 工具供给新增 **batch_recipe**(`agentic/gateway.py`,唯一 Harness 改动面;binding=None 时与旧行为完全等价)。该工具把延迟窗口数字带进 Fast 上下文,因此**永不绑进产出授权证据的 Task Episode run**(信息墙注记已写入工具描述与工件)。
- Experience 首次装入**内容已验证为正**的条目,并以 provenance 严格隔离:`batch_recipe_tool_v2_engineering` / `agent_hand_rolled_engineering` / `budgeted_search_engineering`,一律 `counts_as_unguided_exploration=false`,不作授权证据、不写 Skill、不授 TRY。
- 新增**预算化配方搜索**仪器(受限评估预算下的 shortlist 决策),用于经验价值的冷暖对照。

## 2. 真实数据上观察到了什么

证据链(全部为已曝光 development 数据;工件在 `artifacts/functional/e2/`):

| # | 结果 | 关键数字 | 工件 |
|---|---|---|---|
| 1 | 每个批次存在可验证正向方案 | 6/6 cell 延迟非负(+0.016 ~ +1.047) | `batch_recipe_v2_all_cells_v1` |
| 2 | 方案随批次变 | 三 cohort 冠军程序互不相同;Agent 历史习惯的 RLS 零次夺冠 | 同上 |
| 3 | 方案随 Consumer 结构变(核心论题) | 同批换 Consumer:程序/掩码/是否处理全翻;稳健 Consumer 吃掉清洗头寸(T233 identity 损失 1.458→0.738,处理增益 +0.117→+0.030) | `consumer_conditioned_recipe_v1` |
| 4 | 池化互作是 Consumer 结构性的 | 换 per-channel 集成后互作 −0.23/−0.25/−0.37 → −0.05/−0.02/−0.07 | 同上 |
| 5 | 采纳方案窗口外成立 | W1 方案在两个新窗口 12/12 延迟非负(v2 规则首次 out-of-selection 验证) | `batch_recipe_windows_v1` |
| 6 | 程序选择半稳、掩码窗口局部 | program_stable 3/6,mask_stable 1/6(且为空集凑数) | 同上 |
| 7 | Agent 闭环行为成立 | 初见调工具 / 复访复用 / Consumer 变化重搜,8 次 LLM 零失败;E3 在看得见旧方案时选择重搜 | `agent_recipe_mount_v1` |
| 8 | 工具对 Agent 的价值 | 同预算 3/3 全胜(延迟差 +1.047/+1.047/+0.225),LLM 省 2.6 倍;徒手臂两次弃权、一次选 hampel 留 +0.22 在桌上 | `agent_recipe_mount_notool_v1` |
| 9 | 经验价值(同成本质量) | #4:目标 A capture 0.367→0.962;#5 轮换:暖 4 / 冷 1 / 平 1,最差 −0.006;暖臂 4/6 逐位或近逐位复现全搜索答案(capture 1.000×3、0.989),评估预算仅 2/7 | `warm_vs_cold_recipe_search_v1`、`warm_vs_cold_rotation_v1` |
| 10 | 无经验 LLM 提案不随上下文条件化 | 冷臂 8/8 个目标给出雷同 shortlist(#4 两目标一张单、#5 六目标一张单),两次采纳延迟为负方案;暖臂 5 张不同单子、全含 outlier_iqr、随 cohort 与 Consumer 变化;hampel 暖臂 0/6 入选 vs 冷臂 6/6 | 同上 |

辅助发现:几何字段无法静态预测掩码剔除(批内反例 + 跨 Consumer 反例双重否定,`m0a_mask_geometry_census_traffic_v1`);traffic 的 union-pss 语义失真达 10/12(全部源于 outlier 区域,继续支持 M0 线的污染判断)。

## 3. 当前最大的方法不确定性

1. **成本维度未测量**:shortlist 填满无代价,"更省预算"结构上不可触发;主张定格为"同成本质量更好"。若要量成本需改读数结构(质量 − λ×评估数)或放宽上限,属下一版仪器设计。
2. **预算化采纳缺 identity 在位者门**(v2 配方规则有、预算化仪器没继承),导致一次两臂双输给 identity 的退化平局。对称不偏,但下一版必须预注册补上。
3. **迁移边界已现形**:批次特异答案(electricity×per_channel 的 denoise)在 LOCO 经验下结构上不可知,暖臂以 0.001 之差落败。跨批次经验的适用范围 = 机制在多批次重复出现之处;批次特异答案必须由目标本地探索补足(对应框架中的 A3 残差通道)。
4. **LLM 行为读数均为单次采样**(每 episode n=1,无重复方差定标);行为结论(路由/复用/单子分化)方向一致且跨 8 个 episode 重复,但单点数字不应过度解读。
5. **受害账是窗口局部性质**:聚合延迟增益窗口外保持(12/12),但逐序列受害数窗口外可增;受害控制依赖本地重搜掩码。
6. Weather 效用不可读(METRIC_UNREADABLE)未解;sealed 库存(NOAA、KDD W3)未动。

## 4. 与用户原始方向是否一致

一致,且是原始两支柱的第一个最小完整实现:

- "任务与模式感知的数据 Readiness 优化" → 配方随批次、随 Consumer 结构条件化(证据 1–4);
- "反馈驱动的 Harness 自适应进化" → 规则 v1→v2 由失败案例驱动;工具挂载源于徒手 Agent 的失败分析;经验使提案分布 context-conditioned(证据 9–10)。

A5-vs-A3 里程碑的**工程原型**在配方层建立:同一搜索仪器成本下(两臂含掩码搜索内部评估各约 100 次,近乎相等)质量 4/6 胜、最差损失有界、负迁移受控(暖臂零次灾难性失败,冷臂两次延迟为负)。与正式里程碑的三点差距(sol 复核 2026-08-21,已采纳):暖臂直接读经验表,未经 Slow 编译的 Source Skill 通道;仪器内部 Consumer 评估未计入成本口径,"更快"维度未测量;未形成进入生命周期的 Target-local Skill。桥接实验及其仪器修正重放已交付正向读数(§5 第 1 项),三点差距至 2026-08-21 全部闭合:真实 Skill 通道(#10 集成)、Target-local 生命周期(#11 探针 + #15 持久化召回)、fresh 跨域确认(#17 收官,FRESH_A5_DELIVERS)。fresh 读数的诚实边界:优势为**适配效率**(pooled 首正成本 69 vs 123,−44%),最终质量与 harm 两臂持平(task_C delayed 差 +0.000000),n=1/cell;"更快"在 1/2 cell 决定性成立,另一 cell 为迁移边界平局;"更安全"表现为零次复用越过 aggregate harm 门(5 拦 1 放),而非 harm 差——pooled 确认两臂各 1 条受害序列,记为 Scope/Risk 缺口。外审终裁(sol,#17 后):接受 FRESH_A5_DELIVERS 与 per_channel TIE 改记;对外 claim 以 fresh_confirmation_v1_adjudication.md 为准(曝光口径=fresh Outcome 非全新 Domain);Slow 首故障改判 TARGET_LOCAL_SCOPE_RISK_GAP——非探针晋级阈值(1.09 SE 晋级技能后续合法交付,抬线会误杀),而是聚合正增益掩盖单序列显著受害。未滑向 Router:决策由 Agent 读上下文做出,经验只调制提案与风险跳过;identity/弃权始终在位。

## 5. 下一个最有价值的纵向切片

1. **RECIPE_EXPERIENCE_TO_SKILL 桥接**(已跑 2026-08-21,recipe_skill_bridge_v1):预注册判定 SKILL_LOSES_SIGNAL(12 次 LLM,T2 一票 A5_LOSES)。门无关的硬读数:送达 3/3(A5 逐字引用编译条款)、A5 命名方案 delayed 三目标全高于 A3——**信号未在编译中丢失,无条款失灵**。败票溯源为仪器复制缺陷:采纳门失败后直落 identity,漏移 v2 回退阶梯("先回落最佳全批方案若 delayed 为正"),命中 3/6 arm-target、方向不偏袒;缺陷源自任务书转述而非执行。成本首测(修正前口径):总重训 A3 194 / A5 175,达首个 delayed 正采纳 A5 累计 66 次 vs A3 156 次。重放已完成(recipe_skill_bridge_v2_replay,0 LLM、0 新增重训,输入 sha256 前后一致):修正读数 **SKILL_BRIDGE_DELIVERS**(3/3 A5_WINS:+0.245/+0.016/+1.117;T2 薄胜按预注册困难目标口径读作非劣性超预期;T3_A3 回退按 support 选择移出 +0.203、敏感性检验显示读数不依赖 support>0 资格线,修正不自利)。成本读数(固定目标顺序下的首次有效方案发现成本;三目标互不回灌,不构成跨目标在线学习率):A5 达首个 delayed 正采纳累计 66 次重训 vs A3 156 次;总重训 A5 175 < A3 194(两目标更省、一目标多 2 次)。v1 缺陷复盘为两笔(回退阶梯漏移、bar 取多候选 delayed 最大值),均由重放按 ADOPTION_RULE_V2 原文修正;v1 判定原档保留。遗留:#6 runner 同型门缺陷不重放(run 内回灌使重放无效),标签维持原档;两个回退采纳的 harm 账 v1 未落盘,置 null 注明。外审终裁(sol,2026-08-21):**DEVELOPMENT_A5_ADAPTATION_EFFICIENCY_SIGNAL_POSITIVE / FORMAL_A5_NOT_ESTABLISHED**;正式 A5 前的剩余清单:溯源策略声明、真实 Skill store/检索接入、Target-local 生命周期(三项合并为下一集成切片),harm 补账、重复采样、fresh 确认(各有队列位)。桥接卷宗(v1+replay+本节)就此关闭,不再回读审计。集成切片已跑(2026-08-21,skill_store_integration_v1):**INTEGRATION_DELIVERS**——检索送达 3/3(通道真实性:两臂 public_input 逐字节相同,卡文仅经 store→检索→Resolved Harness 注入,store/检索/生命周期代码零改动)、方向 3/3 为正(T1/T3 逐位复现修正读数,T2 边距收窄 40% 为实测通道损耗)、5 个 LOCAL_DRAFT 经真实代码路径形成;溯源策略以实测载体保证落档(GUIDANCE 授权、无免确认 TRY)。清单前两项闭合;生命周期余 DRAFT→ACTIVE:延迟读数参与了选择故诚实不升级,以 0-LLM 后续窗探针切片收尾。已知接口债:检索门词表无法表达 cohort/consumer 作用域(本轮 LOCO 隔离靠命名空间),收官书中裁决(合并卡旁路或词表扩展)。探针切片已跑(lifecycle_probe_v1,0 LLM、5 次评估):**LIFECYCLE_CLOSES**——3/5 DRAFT 经真实 `_update_delayed` 路径升 LOCAL_ACTIVE(探针窗逐字引 task_05 roster,读数未参与选择;traffic 两 DRAFT 因 sealed 边界如实 UNAVAILABLE,未挪窗);证据字段 SUPPORT→DELAYED 齐备(gain/se 3.8-5.8)。in-selection 膨胀首次量化:T1 采纳时 +0.2448 → 探针 +0.0936(−62%),为"延迟读数参与选择不得作确认证据"纪律提供量级;T2 两例探针反升,膨胀非普遍律。注意事项:RESTRICTED/中性带分支未被执行到;A3 的 DRAFT 同样升级且 T2_A3 探针(+0.0567)高于 T2_A5(+0.0405)——升级是通道属性非卡片因果,T2 弱证据档案再添一笔。sol 清单第 3 项闭合,余 harm 补账(收官仪器)、重复采样(可选)、fresh(用户门)。外审复核(sol,#10 后):总判定 DEVELOPMENT_A5_POSITIVE + REAL_GUIDANCE_CHANNEL_DELIVERS + TARGET_LOCAL_LIFECYCLE_PARTIAL;first fault 后移至 TARGET_LOCAL_SKILL_PERSISTENCE_AND_PROMOTION_GAP——#11 已验更新机制(episode 级),但技能从未经 handle_fast_winner 持久化为可检索 Target-local Skill、从未在下一任务被召回复用;以持久化+召回切片闭合(冻结 Guidance/编译器/v2 阶梯/菜单/预算,只动生命周期面)。Slow 自更新层与 fresh 的先后:本线裁定 fresh 主张不依赖 Slow(考试期 Slow 冻结),层 2 排 fresh 后,sol 的保守排序作为备选记录。持久化+召回切片已跑(local_skill_recall_v1,2 LLM):**LOCAL_LIFECYCLE_CLOSES**——3 个 Target-local Skill 经真实 handle_fast_winner→handle_feedback_delayed(引 #11 task_05 探针证据,只读不重测)入 store 为 approved/LOCAL_ACTIVE(元数据寄放 risk_guards,schema 无状态字段);task_06 上 A5@T1/T2 与 A3@T2 自然检索命中并复用,0 RECALL_MISS、0 REUSE_HARMFUL,对照 A3@T1 如预期只能重搜。域内学习曲线首读:复用格 60/75/69→15 次重训、LLM 2→0(省下的正是 shortlist 与逐序列掩码轮),对照格 69→63 仍付全价。范围上限如实入账:复用为 Runner 直供非 Agent 择选;单候选下延迟门退化为符号检验(本轮无候选间延迟竞赛);三个复用方案 delayed 跨窗一致缩水(仍全正、未触 −0.005 害线,A3@T2 距材料线仅 +0.000271);task_06 delayed 参与采纳门、依设计不作晋级证据,技能停留 LOCAL_ACTIVE。sol 层 1(Fast adaptation pipeline)至此全链闭合,层 2(Slow 自更新)与收官解耦。
2. **负路径自更新演示**(已跑 2026-08-21,negative_path_adaptation_v1):预注册判定 GATE_SAVES_BUT_NO_LEARNING(9 次 LLM)。实测:廉价自主弃权存在(E1,1 次评估自选 identity);负经验驱动复用拒绝成立(E2/E3 逐字引用失败记录改道);控制组无过度泛化(E4 capture 1.000,+0.387)。两处失真如实入账:E3 被门误拒的恰是 W4 全搜索最优(bar 由负 support 不可采纳方案设定);W4 实有 +0.029 头寸,"停手收敛"的前提(多窗零头寸 cell)不成立,该问题搁置不重跑。第四环"失败驱动自适应"证据形态定格为"负经验→改道",非"负经验→停手"。
3. **收官确认实验**(需用户批准,烧 sealed 数据):仓库核验(2026-08-21)修正储备清单——代码指定的 sealed 确认集仅 noaa_global_hourly(e1.py `SEALED_CONFIRMATION_DATASET`,从未触碰);KDD ≥3072 已是 e1v2 开发任务空间,非储备;screening 无其他通过候选(PSM/SWaT 被否)。健康检查第一段已跑(noaa_health_check_v1,0 LLM、0 重训):**NOAA_STRUCTURE_FAIL**——物化仅 19 条 × 1024 点(结构线 5760),第 2 步未执行,outcome 一格未烧。同轮取证修正两个前提:(a)"从未触碰"不成立,旧线已有 9 份 NOAA outcome 报告(A5_NOT_BETTER 等),registry `certified_virgin` 与 `EXPOSED_FAMILIES` 曝光记录矛盾,按保守读法视为已消费;(b)现物化不存在 ≥3072 索引,收官协议装不下。修正后的收官路径:用 raw 74 台站 × 2024-2025 两年小时数据,以"未消费台站 + 日历分区(2024=开发区,2025=确认区)"盲规则重新物化 fresh cohort(family 级 AGGREGATE_SEEN、instance 级 UNSEEN 的诚实层级),健康检查 v2 通过后,一次性在 2025 确认区对决完整管线(全源合并卡 + 真实 store/检索 + v2 原文语义仪器)vs 冷启动;主读数 sol 成本度量,双候选 harm 落账。重物化与确认各设一道用户批准门。重物化第一段已跑(noaa_fresh_cohort_v1,0 LLM、确认区零读取):**INSUFFICIENT_UNCONSUMED_STATIONS**——v1 普查的"74"系 csv 文件数而非台站数,2024 唯一台站 64,已消费 44(registry 40 ∪ 报告/roster/配置点名),fresh 池 20 < 预注册下限 24;任何计法下均不足(最宽口径 23)。裁定:下限 24→20 为 outcome 前修正(零 csv 打开、零数值读取,普查只暴露了供给数而非任何结果),依据=下限本为 12 席 roster 的健康检查损耗余量,20 仍留 8 席;旧线 p0 曾扫描全部 64 站但 62 个拒绝读数无一存留、本线方法开发未用任何 NOAA 数值,20 站按 instance=SCANNED_BY_RETIRED_SCREENING_NO_SURVIVING_READOUT / outcome=SEALED 诚实披露,不冒充 UNSEEN。#16 = 落地修正 + 物化 20 站 2024 开发区 + 健康检查 v2(roster=字典序前 12 个通过站);确认区 2025 数据需考期按盲规则获取,不可得时以 2024 内日历切分做 outcome 前替代修正。落地已跑(noaa_fresh_cohort_v2,0 LLM、0 重训):**FRESH_COHORT_READY**——census 与 #14 清单逐字一致,物化 20 站 × [0,8760) 开发区,16/20 站体检 PASS(判据逐字引 v1 健康检查:长度 5760/缺失率 ≤0.3425/substrate 平线筛),roster=字典序前 12 个 PASS 站 + 4 替补;2025 csv 零打开,确认区零读取,v1 原档未动。遗留:JUDGE_UNREADABLE 未测(0 retrain,weather 家族有 METRIC_UNREADABLE 前科),收官考卷以开发区读判性预检为第 0 阶段硬门,不可读则停且 2025 不下载。收官已跑(fresh_confirmation_v1,12 LLM、513 重训、398s;19 项冻结面跑前跑后 sha256 一致,无并发写;roster 按 outcome 前修正取 12 train + 4 eval 替补,2025 存在性 16/16、替补级联未触发):**FRESH_A5_DELIVERS**——pooled 主 cell A5_WINS(首正成本 69 vs 123 重训,总成本 99 vs 195;A5 卡引 R1-2/R1-1/R3-1 首任务即中 outlier_mad 全批 delayed +0.306,A3 首任务门全败落 identity;两臂 task_C 收敛同一方案 +0.0297、harm 各 1);per_channel 由主线保守改记 **A5_TIE_TRANSFER_BOUNDARY**(预注册 WINS 与 TIE 条款同时成立且无优先级,首正差 3 次重训系噪声量级,按不利于己方向落档,总判定不敏感)。实质定格:两 cell task_C delayed 差均 +0.000000,**质量零信号,全部优势在成本与首正速度**;"只在 task_A 形成 Draft"的协议规则放大 pooled 总成本差(协议补全的乐观反事实 ≈ 99 vs 144 [57+66+6+15,含探针成功前提],非实测;首正读数 69 vs 123 不受此影响);pooled 胜负吊在单次 shortlist(n=1/cell,无重复无区间);三次探针晋级全在噪声带(g/se 1.09/0.29/0.56),随后 5 次复用尝试被拦(3 次当窗确认、2 次确认通过但 v2 门拦)+ 1 次合法放行且保持正向,零次复用越过 aggregate harm 门——**弱晋级+强确认的两钥匙结构是本轮实测的安全承重件**("无门必受害"系已测读数上的反事实推断,非独立消融);task_C 系 2024 初训练窗预测 2025-02 的移位读数(冻结仪器形状,两臂对称);pooled task_C 两臂同付 1/4 eval 序列 −0.1256、被聚合 +0.0297 掩盖(12+4 方差代价兑现,逐序列 harm 粒度记入未来 Scope/Risk 面)。确认年实际覆盖至 2025-08 末(尾部 ~3000 槽源端无数据,可用区 [8784,~14490)),beyond_17520 仍 SEALED。**冻结版本唯一一次开卷已用掉**,此后仅允许 0 评估确定性重放;曝光台账终版:development_2024 与 confirmation_2025 均 SEEN/EXPOSED。

4. **Slow Scope/Risk 自更新切片**(#18,已跑 2026-08-21,slow_scope_update_v1;checkpoint 9a0f953 先行入库 14 文件,父 eed0a01):判定 **SLOW_DECLINES_PATCH**(执行侧标 SLOW_ABSTAINS;该标签在重跑前已由主线预注册,执行侧未同步,事后同义并档)。B1 确定性归因(0 LLM):a5_pooled 逐序列读数 → RISK_GAP at OUTCOME_RISK,纯聚合读数 → NO_ACTIONABLE_FAULT——**该故障类在聚合观察粒度下不可见的直接证据**;a3_pooled 双读数 SELECTION_MISS at CANDIDATE_SELECTION(掩码 +0.2399 在自己桌上却取全批 +0.1912,#15 歧义 5 同类第二例,recorded-uncorrected)。B2:Slow 在授权面(两卡 risk_guards + verification.rules)上三次带理由弃权(no_authorized_minimal_edit / insufficient_public_evidence;首轮 reason 因 runner 缺陷丢失,修复后验证生效),弃权采样纪律(每轮 ≤2 次字节相同抽样)用尽即停;B3/B4 依判定未执行。成本 3 LLM / 0 重训 / 59.4s;32 项冻结面前后零漂移。主线裁定(经执行方复核审计修订,2026-08-21 晚):**family 不关闭;first fault 由单因 OBSERVATION_PROJECTION_GAP 修订为复合 EDIT_SURFACE_DEFECT**——(i) 授权面非最小:两处特定 Skill risk_guards 要求整对象替换且含无契约生命周期字段(frozen_plan/local_status,动之即断 direct recall),第三处为整份 verification 文档;(ii) 发明键 scope_risk_guards 无运行时读取方,散文断言不可验证,即便提案也不绑定行为(安慰剂面);(iii) guard 执行上下文无逐序列观察(原诊断降格为组件)。判别证据:字节相同提示两码不一致(任务欠定)+ **risk_too_high 零次使用**(Slow 从未否定方向,只否定供给面)。结论改记:**归因环成立且有判别力,编辑环未被真正考到,三次弃权系 Slow 对病态编辑任务的正确行为**;KIMI 报告经 Opus 复核全数属实(33→32 更正成立)。B1 两条硬读数独立入账、不随 B2 命运:聚合折叠盲点为机器自身的两次折叠演示;a3 SELECTION_MISS 为同 episode 第二缺陷、由非专用机器捡出。#19 修订为一次复合仪器缝合(写入面:scope_risk_guards 以空列表预登记 h0 verification 文档 + SurfaceRegistry json_pointer,语义由注册表承载,最小编辑=向已存在空列表追加一条;执行面:采纳门 guard 评估器实际读取该列表并绑定逐序列 delayed 向量;非回归门=空列表下四 task_C episode 逐位复现),缝合冻结后 Slow 同纪律重试(≤2 次字节相同抽样,逐次记录 backend,不在旧面上再抽样);KIMI 全弃权→Opus 追试一轮,跨 backend 全弃权才判 SLOW_DECLINES_PATCH_FINAL。veto/collapse 边界预注册落地(聚合 <0 = REPLAY_AGGREGATE_COLLAPSE,=0 = closes_by_veto 且弃益 +0.0297 入账,>0 = closes_by_rescope);#18 三次弃权的 backend 身份 #19 回填分列。#19 已跑(slow_scope_update_v2,3 LLM / 111 重训 / 57.9s):**SLOW_CLOSES_SCOPE_GAP_BY_VETO**,带预注册双标注 obtained_after_abstention(总第 3 次抽样得提案)与 backend_dependent(gpt-5.6-luna 五次全弃权 [#18 ×3 + #19 ×2,字节相同提示],claude-opus-5 一次即提案——语法修复必要但对弱 backend 不充分,系 Harness 对服务模型鲁棒性的真实发现)。链路全通:缝合(O1 逐序列透传 + scope_risk_guards 空列表经整文档面迁移 [四 fork 单键 diff 回执] + 评估器落 harness 本体 compiler.py)→ 非回归门四 cell 逐位复现(111 重训全在此,B3 增量 0)→ Slow 提案 per_series_harm_line_veto(min_per_series_gain < −0.005 → VETO_AND_FALL_BACK,applies_to=every_adoption;阈值取预注册害线非定制值,rationale 仅引公开数字,附 4 条自证伪条件)→ compiler PATCH_APPLIED(touched 恰为授权面单键)→ replay:两臂 pooled outlier_mad→identity(guard 触发,99999904140 不再越线,聚合 +0.029688→0,弃益双臂显式入账;fallback 之 Support 冠军即被否方案,故落 identity,增量 0 重训),per_channel 零变化(回归通过),B4 依规零 Experience(全 cell 终 identity,skip 原因逐 cell 在册)。**完整自进化环(失败→Runtime 归因→Slow 单面修改→确定性校验→replay 行为改善)首次闭合**,sol 清单最后一项翻绿;闭合等级=development replay(非 fresh),形态=BY_VETO 非 BY_RESCOPE。主线裁定歧义 (c):veto 为合约正确行为(害线是冻结风险契约,聚合增益不赎回逐序列违约),弃益为守约价格;rescope headroom(掩码保收益)结构上存在、replay 未付新搜索,留作可选后续切片,不属本 family 闭合条件。冻结面 v2=33 项(4 项双 hash 重入册 + compiler.py 新增),28 项零漂移;harness_surfaces.json CRLF/LF 双口径以工作树字节为准注明;单条 schema 约束使多 guard 共存需新授权轮(最小性设计使然)。全部改动留工作树,与 #18 交付物待统一检查点(#20)。#20 已收(Opus 只读复核全过,提交 531049f,11 文件,父 9a0f953):复核六点入案——(1) CRLF 注记应上移进工件(committed blob LF 归一 d26a5215,其 CRLF 形态=在册 e222daff),立为常规;(2) **缝合部分性**:evaluate_scope_risk_guards 只做读取+触发判断,veto 动作仍在实验 runner,tracked 通路读 guard 而不执行——评为已暴露的下一接线缺口;(3) **B2 信息量降格**:O1 后语法内可见 per-series 向量的统计量唯一(min_per_series_gain),"Slow 找到了修法"不成立,Slow 贡献=阈值/动作/范围选择 + rationale + 四条自证伪,对外引用一律降格版;(4) BY_VETO 边界系 #17 数字后、replay 前所划,时序在案;(5) manifest.json 属冻结面但被 gitignore 无 HEAD 基线,下一检查点反 ignore 入库;(6) 同族披露:提案 claude-opus-5、复核 Opus 5、主线同为 Claude 族,缓解=判定链确定性(compiler/replay/sha)不依赖判断权威;KIMI 在修复面上两次弃权理由收敛为 insufficient_public_evidence(比 #18 两码不一致更稳),证明 no_authorized_minimal_edit 一半确被修复。外审(sol)阶段命名 **V1_TECHNICAL_ROUTE_ESTABLISHED**:总体路线成立(不回头改架构),能力在完善(veto 非 rescope、fresh 优势单 cell、Slow 提案 backend 依赖、多卡同库检索未解、单 capability family);claim 七行表照录 adjudication 口径。下一里程碑采纳 sol:**OPERATIONAL_PIPELINE_CLOSES_IN_ONE_RUN**(#21,一命令一轨迹一连续 store,从源编译到 Slow 更新后的下一任务决策,无人工接力);前置 Part 0.5 =按审计点 (2) 把 veto 执行提升进 tracked 采纳通路(完成 #19 R3 原规格,接线非方法,非回归门=四 episode 逐位重放);BY_RESCOPE、shortlist 稳定性、多卡检索、第二 family 依次排后。#21 已跑(operational_pipeline_v1,LLM 4 / 单轨迹重训 54 + P2 门 111 另记 / 36 项冻结面零漂移):**COMPILER_REJECTS——连续链前五环全通,第六环断**。前五环:接线完成(P1 veto 执行入 harness 本体 +128/−1,harness 决策、调用方测量;P2 四 episode 逐位复现)→ Guidance 卡由运行自主重编译(entry/body sha 与 #17 pooled 卡逐字节同)→ task_A 全价采纳 outlier_mad(+0.3064;**Agent 引 R3-1 自发弃掩码轮,21 次重训拿到 #17 同格 69 次的同一方案同读数,条款改变行为的 n=1 样本**)→ 探针晋级 → task_B 召回、当窗确认不过、诚实 identity → task_C 召回复用、伤害逐位复现 → 归因 RISK_GAP@OUTCOME_RISK(仅本轮自产证据)。断点:Slow(claude-opus-5)**提案非弃权**,编译抛 rationale must be 1..600——**机制不对称实锤:maxLength 存在于公示语法与编译器两端,却缺席于专职教契约的信封校验器(_validate_local_schema 实现九种约束独缺 maxLength),违规静默穿过重试层、死在无二次机会的门上**。主线裁定歧义(4):记 CONTRACT_ENFORCEMENT_CHAIN_GAP,不记 Slow 失败(提案跨过弃权障碍,死因格式非语义;同型空转威胁一切 stage schema)。两处报告层缺陷自曝:stage_slow 先抛后记(提案原文丢失,#18 同型)、_public() 滤 store 键(快照读回补注)。#22 = 三项已证仪器修复(信封补 maxLength 对称实现、先记后抛、store 键过滤)+ 全轨迹重跑 v2;P2 门不重跑(修复均在提案/报告路径与测量正交,P1 文件 sha 不变为证);新增预注册 PIPELINE_RUNS_NO_FAULT_SAMPLE(重跑轨迹若自然无 RISK_GAP 抵达 task_D,如实记环 6-9 本样本未考,不重掷不播种)。预跑注记(脚本化提案,非读数):task_D@10560 伤害将复发且加重(两序列),guard 有望现场触发判据(i)。#22 已跑(operational_pipeline_v2,LLM 2 / 轨迹重训 57 + 运行后重建 24 另记 / 37 项冻结面零漂移):原始标签 LOCAL_LIFECYCLE_BREAK,主线改判 **NO_ADOPTABLE_PLAN_SAMPLE**(生命周期机制未被触及——task_A shortlist [rls, hampel] 在冻结 v2 阶梯下无一可采纳:hampel support +0.0393 过线但全批 delayed −0.0261,阶梯只回退 delayed 为正的赢家 → 诚实 identity;运行后 0-LLM 重测证实,仪器按设计弃权非故障;标签升入预注册集)。Part A 三修复全部生效并当场验收(A1 信封 maxLength 对称实现,预跑证 #21 死因已消除;A2 先记后抛双路径验收;A3 store 区块入工件),但真实链条未走到 Slow,A1 修好的链本轮未被真的走一遍。三笔新观察:(a) **引用≠遵从首个样本**——Agent 引全部三条 TRY 条款后 shortlist 与条款命名程序零重叠(n=3:两次重叠且成功 +0.306,一次背离且停摆;skill_clause_use 只记自称,无校验);(b) 同配置三抽样三个不同 shortlist,单轨迹一锤对 task_A 抽样高度敏感——协议升级为**预注册 K=3 多轨迹**(每轨新鲜 store、全部入账、不早退不重掷,聚合=闭合/自堵/断裂分布,兼答 sol 排队的 shortlist 稳定性项);(c) 同族记录缺陷第三处(trajectory 迟挂 payload),根治规则=每阶段记录进入前挂载。h0 lock 单键过期(ttha:schema_contracts),重生成值已预算,#23 Part 0 落地并统一检查点。#23 已跑(operational_pipeline_v3;Part 0 检查点 d59b993 11 文件 + h0 lock 重生成核对一致;Part A 记录根治 10/10 边界中止验收 + clause_shortlist_overlap 机器计算字段 + v5 冻结 38 项;Part B LLM 6/24、重训 303/450、冻结零漂移):聚合 PIPELINE_NEVER_EXERCISES_SLOW(raw verdict 保留),主线裁定经 sol 外审收紧:新增 ADJUDICATED_FIRST_FAULT = **ATTRIBUTION_EPISODE_ROUTING_GAP**——sol 行号级定位硬接线(stage_attribution 只收 task_C 记录 L850、主流程固定送 step_c L1528、Slow 门只看 task_C.harmed_before_gate L1534),Runtime 未在本次运行已完成、已采纳、delayed 已揭示的 episode 中找最早可行动失败;不是 NO_FAULT_SAMPLE、不是 LOCAL_LIFECYCLE_BREAK、不是 SLOW_FAILURE。三轮重试 canonical 重述:#21 前五环成功+Slow 真实提案+编译因契约链缺口拒收(非 Slow 无能力);#22 task_A 无可采纳方案诚实回退(非生命周期坏);#23 技能形成/晋级/持久化/9 召回全发生但 task_A 风险 episode 未被送入归因(非无故障样本)。注记:task_C 的 SELECTION_MISS 系适配器窄口径把一次正确弃权(当窗 Support +0.0859 过线、delayed −0.0869 被 v2 阶梯合法拦)读成选择失误,#21 在册缺陷第二次兑现,留案冻结不修。三轨不构成三个独立风险样本(同一窗口同一 Program 的确定性结果,三条独立决策轨迹重复遇到同一失败)。教训成规则:非 iid 抽样只报观察频数,不做前向概率(#23 前的 ≈96% 与 #24 初稿的 ≈1/6 同错,均删)。三轨全长自然跑通 1–5 环 ×3:task_A 三次真独立抽样(provider_seed=False,T3 掩码搜索 87≠21 重训)收敛同一决策 outlier_iqr 全批 delayed +0.0669;技能形成→弱晋级(g/se 0.298)→持久化→9 次召回全机制首次在管线内自然走通,但 0/9 复用成功(B/D 当窗 Support 不过线,C 被 delayed 门拦)——两钥匙第三次拦住本不该复用的技能,安全面正读数,Target-local 跨窗存活 0/9 为本 cohort 供给面事实。**承重发现(歧义 1)**:故障样本三次都存在——task_A 采纳聚合 +0.0669 掩盖 99999923908 −0.0621,教科书式 Scope/Risk 故障;协议把归因硬接线在 task_C,三次从样本旁走过。管线未达 Slow 的直接原因是**归因窗口接线**,不是菜单可采纳率、不是抽样方差。K=3 独立性低于预期(三次同 shortlist),确认修窗口优先于加 K。→ #24(经 sol 四处收紧后定稿):唯一改动=归因 episode 选择接线,语义为 **earliest eligible scope/risk episode**(仅本次连续运行中已完成、已采纳、delayed 已揭示的 episode,按时间序找首个越 −0.005 害线者原样送现有 _attribute,记录 selected step/episode id/未选原因;无越线返回 NO_ELIGIBLE_SCOPE_RISK_EPISODE 而非全局无故障;不读未采纳候选/未来 task_D/旧工件/sealed outcome;是 lane 不是通用归因平台)。顺序:先 0-LLM 确定性重放(v3 三轨银行台账,三轨必须都选 task_A、归因必须出 RISK_GAP、SELECTION_MISS 不得抢先、验证 selector 读取边界)→ T1 补考 6–9 环(BY_VETO 为 containment:消除 −0.0621 受害同时主动放弃 +0.0669 聚合,"无殃及"仅指不相关 episode 决策未被意外改变,保收益等 BY_RESCOPE)→ 一条 post-fix live 单轨(--post-fix-live,判定 DEVELOPMENT_OPERATIONAL_PIPELINE_CLOSES_POST_FIX;NOAA outcome 已曝光,此为 development 级证据非 fresh confirmation;自堵则如实记 NO_ADOPTABLE_PLAN_SAMPLE 不重掷)。B+C 双闭合的 claim 上限=V1 development 级连续管线闭合,非新 fresh 证据、非 A5>A3 新结果;闭合后才进 BY_RESCOPE。#24 已跑(operational_pipeline_v4;LLM 3/12 全在 B2 一次抽样、重训 0/250、冻结 38 项零漂移、v1/v2/v3 原档未动):Part A 选择器 5/5 验收——三轨都选中 task_A(min −0.062068)、归因三轨 RISK_GAP@OUTCOME_RISK、SELECTION_MISS 未抢先、读取边界 ×4、合成台账返 NO_ELIGIBLE_SCOPE_RISK_EPISODE,**路由缺口仪器级闭合**(T3 无掩码完整证据轨作对照,T1/T2 重建不完整但同向,权威检验留给 live)。Part B 断在 B2:claude-opus-5 三次内部重试均未产出可解析 agent-envelope/1(AgentProtocolError),决策从未被观察,EditController 零调用;raw 标签 COMPILER_REJECTS 系 stage_slow 把非 transport 异常一律归此的分类缺陷(记录家族第四处),主线改判 **SLOW_ENVELOPE_PROTOCOL_FAILURE**(升一等标签)。歧义 2 裁定沿 #19 先例(transport 级故障 ≠ 弃权,单列 inconclusive):信封失败不耗 #18 弃权采样额度,但每次必须留 raw response(本轮未留——无法区分格式不合还是证据束截断,并入记录家族修复)。Part C 未跑(门控成立)。backend 依赖性(信封契约对模型敏感)入册为可移植性事实,本轮不修 opus 兼容。**#25 初稿含决定性事实错误,经 sol 外审纠正并由主线工件核实**(slow_scope_update_v2.json b2_patch:rounds[0] KIMI 两抽全弃权,rounds[1] OPUS 一抽即产出被接受提案;工件原文"grammar repair was necessary but not sufficient for the weaker backend"):正确历史 = Luna 四抽零提案仅合法弃权,Opus 为唯一提过案后端(#19/#21),#24 系其三抽首次信封失败。初稿"换 KIMI 求提案"方向反了,已废。教训成规则:任务书中后端/证据出处断言必须带工件路径并核验,不得凭记忆。→ #25 定稿:Part 0 统一检查点;Part A = stage_slow 错误分类学 + 复用 agent_core 异常已附诊断(有界文本/全长/finish reason/token/截断标志,不存全 raw),不建 v7 哈希清单(v6 基线 + git diff 足够,反 SHA 扩张纪律);Part B = **保持 Opus 同配置再取一个 outer draw**,三计数制(valid_decision_samples/protocol_failed_draws/llm_calls_spent),协议失败独立上限 2(#24 已占 1)→ 再失败即 SLOW_ENVELOPE_PROTOCOL_FAILURE_EXHAUSTED 停手;合法提案→编译/B4;合法弃权→计首个有效决策样本,按纪律至多再取一个;Part C 门控不变,backend 同 B 实际闭合者,若最终使用 Luna 则 claim 必须带 ON_GPT_5_6_LUNA 后缀,不得声称确认原 Opus 配置或后端无关。#25 已跑(operational_pipeline_v6;LLM 2/12、重训 0/250、墙钟 78s、冻结零漂移、v1–v5 原档未动;Part 0 检查点未执行,并入 #26):**BANKED_CHAIN_CLOSES_IN_K_MODELS,2/2 —— 九环链 6–9 环首次被真实走完**:选择器→fold(RISK_GAP@OUTCOME_RISK;仅聚合视角仍 NO_ACTIONABLE_FAULT)→Slow 真实提案(gpt-5.6-sol 与 gpt-5.6-luna 各一抽即合法,0 信封重试)→EditController 全链(#22 maxLength 修复首次真实过考,两份 rationale 均 <600 字符)→guard 安装→银行重裁定(task_A outlier_iqr→identity,99999923908 的 −0.0621 消除、+0.0669 聚合主动放弃,containment 非效用改善;task_B/C 逐位不变,unrelated_episodes_that_moved=[];两模型除 guard_id/rationale 文字外数值读数逐位相同)。#24 信封失败根因查明 = Claude 上游宕机(Service load is too high),非格式不合、非证据截断,两个先验假设皆误;fix (c) 分类学保留(未来宕机记 INCONCLUSIVE_TRANSPORT 不耗额度)。执行方因宕机现场改钉 sol+luna 双后端(各 1 draw、三计数制),主线追认为合格现场适应。两模型 guard 参数逐字收敛的信息量有限(语法唯一逐序列 statistic + 公示害线,选择空间窄,#19 老账例证)。Luna #19 双弃权 vs 本轮即提案 → 采纳"证据敏感"解读,#19 "weaker backend" 叙述加脚注不再作能力结论。裁定:PROTOCOL_FAILED_CARRY_IN['claude-opus-5']=2 划销重分类为 INCONCLUSIVE_TRANSPORT×2(外部宕机证实),Opus 额度复位,服务恢复后可选银行复跑取 Opus 读数(排队不阻塞)。裁定:Part C 不等 Opus,gpt-5.6-sol 跑,claim 带 ON_GPT_5_6_SOL 后缀。**#26 已跑,V1 里程碑落袋(operational_pipeline_v7;LLM 3/8、重训 126/200、冻结 39 项零漂移、v1–v6 原档未动):DEVELOPMENT_OPERATIONAL_PIPELINE_CLOSES_POST_FIX_ON_GPT_5_6_SOL(仪器层原值 OPERATIONAL_PIPELINE_CLOSES 留档,改名仅在报告层)——九环链首次在一次连续无人运行中自主闭合**:task_A GATE_FAIL_FALLBACK_SUPPORT_WINNER 采纳 outlier_mad(+0.3064,逐序列全正,最差 +0.0177;冻结 v2 阶梯三条路径至此全部有 live 样本)→ 技能 ACTIVE(探针 +0.2058)→ B 窗诚实弃权 → C 窗 GATE_PASS 复用(+0.0297 聚合掩盖 99999904140 −0.1256)→ **选择器越过干净 task_A 自主选中 task_C**(证明非写死;task_A min +0.0177 不合格、task_B identity 无可归咎)→ 归因 RISK_GAP@OUTCOME_RISK(聚合视角第五次复现 NO_ACTIONABLE_FAULT 盲区)→ Slow 一抽即提案(三计数 1/0/1,0 信封重试,参数与前三次逐字收敛:唯一逐序列 statistic + 公示害线,继续按选择空间窄记账)→ EditController 接受(快照 2afeed810cad→05a4cbd16c0d,单面改动)→ task_D 读新快照哈希核对一致,伤害复发且更重(99999904140 −0.0827 + 99999963862 −0.1028,聚合 +0.0495 仍在掩盖)→ **guard 现场开火 VETO_AND_FALL_BACK→identity,判据 (i) 成立**。证据等级 DEVELOPMENT 写死工件:不产生 fresh 证据、不产生 A5>A3 新结果、不声称后端无关。叙述更新:v7 技能在 C/D 两窗 GATE_PASS 复用成功,#23 "跨窗存活 0/9" 系弱技能(g/se 0.298)个案而非 cohort 普遍事实。未考项:guard 只考过"救得下"(veto 正确开火),未考过"不误伤"(B/C 本就 identity,无殃及检验空跑)——排入下轮与 BY_RESCOPE 同考。检查点:f8a50ee 漏 4 文件(agent_backend.py fix(c)、h0 lock、v5 工件),新克隆会 INSTRUMENT_DRIFT 拒跑,修复提交已授权;v7 交付物另打里程碑提交。队列:BY_RESCOPE(保收益修复,#27)> Opus 银行复跑(服务恢复后)> SELECTION_MISS 适配器(冻结中)。#27 已跑(operational_pipeline_v8;LLM 1/6、重训 0/50、冻结 39 项零漂移;检查点 ce8bc2d 修复四文件 + 15d7469 里程碑 v7):**RESCOPE_PRESERVES_GAIN_ELIMINATES_HARM**(确定性三臂:task_C 无guard +0.0297带1受害 / VETO 0 / RESCOPE +0.0611零受害剔1;task_D +0.0495带2受害 / 0 / +0.0959零受害剔2)+ **SLOW_PROPOSES_RESCOPE_MASK_HARMED_SERIES**(一抽即中,0 信封重试;两动作平权信封下模型选 RESCOPE 且自主把 applies_to 收窄到 reused_skill_adoption_only——语法变宽读数即变,部分证伪"收敛=空间窄"的悲观半)+ B1 NO_FALSE_POSITIVE(干净1条不触发、越线6条全触发、identity 5条门不进入;n=1系逐序列向量历史深度非筛选)+ B4 AS_PROPOSED_GUARD_CONTAINS_WITHOUT_COLLATERAL(整条v7轨迹重裁定,task_A零误伤有一半由收窄结构保证,如实注记)。**几何偏离主线追认**:掩码几何(train_uids)与逐序列读数(eval_uids)交集为空,照字面实现即安慰剂;替代语义=评估侧 identity_routed_eval_series + 已测向量投影(前提 aggregate==mean(per-series) 在70个银行字典上核至2.8e-17),条件:(a)命名诚实=逐序列identity路由非掩码;(b)一切claim挂同窗选择caveat(路由用被打分的同窗,VETO同病);(c)live前完成下游接线(该键目前零消费者,per_origin_gain投影中显式丢弃)。canonical措辞:RESCOPE聚合高于无guard是去负项的算术必然,收益真实形态=受害序列改服务identity,非加性增益。前置发现:h0 lock字节口径未修(core.autocrlf在checkout转CRLF,新克隆仍INSTRUMENT_DRIFT),授权一行.gitattributes(snapshot.lock.json -text)入下一检查点并克隆验证。台账/路线图的modified是主线自己的修订,授权入检查点。carry-in清零随#28 Part A落地。→ #28:Part 0检查点(.gitattributes+全部#27交付+台账+路线图);Part A接线(identity_routed_eval_series进plan label/Experience/Skill生命周期读数,per_origin_gain显式null+理由,carry-in清零);Part B live单轨(--rescope-live,两动作平权语法,gpt-5.6-sol,一抽不重掷)。**#28+契约统一轮已跑,Phase R 收官**(检查点 5728897 含 .gitattributes,克隆实测字节口径修复、--gate-only 过;A 接线 19/19,probe/persist 冻结读者拿 plain plan 属正确设计;更正两处:lock 期望值 03d08251 非书中陈旧值 327f0118,SelfEvolvingHarnessTS 符号链接未入库纯 clone 不可导入→backlog)。v9 live 首跑 **COMPILER_REJECTS**(预注册集,保留为诊断):新契约链缺口与 #21/#22 同族异字段——_behavior_vocabulary 把正则谓词渲染成字面 token,信封 schema 仅 minLength 放行占位符,EditController manifest 真 oneOf 在 apply_to_fork 无退路拒绝;RESCOPE 语法三处全过,非方法问题。契约统一(外审指令,主线追认):信封 items 直接复用 EditController 同一 behavior_predicate 对象(非副本),词表 31 枚举/3 正则分列且示例回验后才给,8/8 验收(拒绝层升至可重试的 STAGE_SCHEMA_INVALID);**升级条款入常备纪律:再现一处独立 Proposal/Manifest 错位即判简化 Schema 重复拥有契约,整体改复用真 Schema**。v10 唯一授权 live 单轨:**LIVE_RESCOPE_CONTAINS_WITHOUT_COLLATERAL**(LLM 3/8,重训 129/200;task_D 快照核对一致,guard 开火恰路由两条越线序列,+0.0495→+0.0959,保留序列逐位未动,误伤检查以独立 measured 前门向量核对零 finding;VETO 孪生同窗对照 0.0 vs +0.0959;Slow 一抽即中零重试,predicted_agent_behavior_change 无占位符;Experience 写入 "outlier_mad -route 2")。记录缺陷第四例(冻结读者丢路由键打在 _step 重测路径,per_eval_series_delayed_after_gate 报未投影向量,多花 3 重训;判定不依赖该字段):裁定下轮修——post-gate 字段一律以 gate 投影收据为唯一来源,不对 routed plan 重测;家族原则=冻结读者只拿 plain plan,其产出字段必须标注 pre-route。n=3 全选 RESCOPE(v8/v9/v10 同后端同证据面)只记不断言偏好。同窗选择、算术必然、DEVELOPMENT/ON_GPT_5_6_SOL 纪律均写死 v10 工件。→ #29:检查点(4 文件 812+/17−)+ 记录缺陷修复 + Phase C claim 冻结整合(交接件)。

## 工件索引

配方线:`batch_composition_headroom_v1`、`masked_single_program_v1`、`batch_recipe_{electricity,T233,traffic}_v1`、`consumer_conditioned_recipe_v1`、`batch_recipe_v2_all_cells_v1`、`batch_recipe_windows_v1`
Agent 线:`agent_recipe_mount_v1`、`agent_recipe_mount_notool_v1`、`warm_vs_cold_recipe_search_v1`、`warm_vs_cold_rotation_v1`
审查与普查:`code_review_batch_recipe_stack_v1.md`、`m0a_mask_geometry_census_traffic_v1`
Runner:`run_batch_composition_headroom.py`、`run_e2_m0a_mask_geometry_census{,_traffic}.py`、`run_e2_agent_recipe_mount{,_notool}_micro.py`、`run_e2_batch_recipe_windows.py`、`run_e2_warm_vs_cold_recipe_search.py`、`run_e2_warm_vs_cold_rotation.py`

---

### #29(2026-08-22,重构后首条增量)

检查点 eac0582(8 files;v9/v10 原样入库)。清单再漏 2 项(compiler.py、h0 lock)→ O8 开立,新克隆 INSTRUMENT_DRIFT 复发(内容口径);主线裁定:自 #30 起检查点清单由执行方 `git status` 实测生成、主线只审排除项。Part A 记录缺陷第四例修复 6/6(NoMeasurement 夹具强制"不重测",顺带省 3 重训;FROZEN_READER_RULE 落为代码常量)。Part B 三段重构交付(190 行,原文一字未删入第 4 节)。**C10 削弱裁定:接受**——机器字段 n=6 全部 overlap≥1,零重叠样本(#22)早于字段仅存散文,claim 降级为"引用与 shortlist 无强制关系、无校验",从工件数数优先于任务书记忆。**仪器规则第五条采纳**(#29 歧义 3,两次夹具事故):验收夹具必须自钉快照/自行回卷,禁止读活的可变 scratch——待 #30 写入 §3.6。Phase S 设计定稿与 S0+S1 任务书、继任恢复提示见 `ROADMAP_POST_V1_2026-08-22.md` §Phase S 与 `SUCCESSOR_BRIEF_2026-08-22.md`。

### #30(2026-08-22)

检查点 5854713(O8 关闭,克隆实测 precondition True;仪器规则第五条入 §3.6,附加推论:夹具要有"会红"的证据)。**S0 = THIRD_DOMAIN_AVAILABLE,主选 smd,无备选**:复用 g3_candidate_screening_v3 预注册标准与 census(未另立,正确);曝光台账十域全 INSTANCE_SEEN/EXPOSED 或纸面淘汰,NOAA Tier2 离线不可物化(raw 恰 64 站=44+20);smap/msl 因退化通道落——首跑三候选全过系二值命令位假象,可用通道门(cardinality>20)为 SWaT 同理由延伸,主线追认非抬门。仪器观察入册:public_phenomenon_census 读不存在的 outlier_fraction 键,三析取恒缺一项;不推翻 psm/swat 旧裁(各落于他因),该 census 再用须先修。**S1(smd)= PROCEED_UNCHANGED**(24 可用通道,23 带公共现象 vs 门 4),substrate_shape_warning 比判定重要:插补家族 smd 零底物(0/24 vs NOAA 20/20),阶跃家族方向相反(smd 中位 6.45 vs NOAA 全零)。主线裁定:(1) 接受 smd 为第三域,对外表述限定"离群修复能力族在 server telemetry 域";(2) **S2 前置=candidate 只落两语料共同可行使的家族(离群修复族+算子无关 harm guard),插补/阶跃 out-of-scope 并写进卡**;(3) **S2 硬门=凭 provenance 恢复 28 机边界,禁止数据推断,不可恢复则 NO_ELIGIBLE_THIRD_DOMAIN**;(4) 官方 partition 按机器定义、经用户确认,8760 块仅 S1 声明用;(5) z 峰只引中位数。→ 用户裁定(2026-08-22):**有条件接受 SMD(仅候选身份)**,八点修订约束 #31:(1) machine=entity、metric=channel,38 异构指标不得当 38 条同质序列;(2) 边界优先官方 provenance,本地缺失允许官方源获取,失败记 SMD_ACQUISITION_BLOCKED 不判项目无第三域;(3) 官方 train=development/held-in、test=sealed held-out,禁套 NOAA 8760 切分;(4) Source 证据按独立 domain/episode 去重,NOAA v1–v10 同窗不算多票;(5) 干重放双向 leave-one-domain-out(traffic 编译→NOAA 检验、NOAA 编译→traffic 检验),同库重放只算内部一致性不算迁移证据;(6) candidate 固定 status=SHARED_CANDIDATE / authorization=GUIDANCE / target_support_required=true / grants_confirmation_free_try=false;(7) S2 过后先 S1b(实体结构+独立 Source 证据+仅开 SMD development outcome+Judge 可读性+outlier headroom;判定集 SMD_TARGET_READY / SMD_ENTITY_MAPPING_BLOCKED / NO_INDEPENDENT_SOURCE_SUPPORT / JUDGE_UNREADABLE / NO_PROGRAM_HEADROOM)再谈 S3;(8) 分段预算 S2 0/0、S1b 0/≤100、S3 ≤30/≤350、S4 ≤200、contingency ≤50、总 ≤700,前段 PASS 才释放下段。核心理由:同源重放一致不得误当共享能力证据;SMD 的可读预测效用与算子 headroom 未证。#31 修订版已发。

### #32–#33(2026-08-22,sol 代发书,主线追认)

#32(S1b,49380d8):A 部全过——执行前条件审计(outlier_point_fraction 降证据、逐序列离散度移出执行前,最小合法 gate 集非空);双向逐算子矩阵:iqr/mad 两域支持,hampel/winsorize 仅 traffic → contrast only,NOAA 同窗去重 13→4;**candidate v2 冻结**(shared_programs=[outlier_iqr, outlier_mad],gates 2 条,四固定字段,先于任何 SMD outcome 写盘);双向 LODO:traffic→NOAA 4/4、NOAA→traffic 4/4,同源重放仅标内部一致性。B 部 28 机映射与 12+4 roster 合法(e1v2 窗型未缩放)。C1 Judge → **JUDGE_UNREADABLE**(主/备两映射 8 cell 中 6 死于份额门,ep2 delayed spread 18.35/11.11 实质超标)。重训 24/100,LLM 0,test/labels 全程 SEALED。
#33(对齐映射补考,f2ca51f):I={1,2,5,13,14,15,18,19,20,21,27,30},选 ch18(28 机中位基数 1069.5),CHANNEL_INDEX_ALIGNED_PROXY(列对齐,不主张物理同义);Judge 四格全不过(ep1 support spread 9.868+份额 0.526 实质失败;其余三格边距 0.001/0.009/0.134)→ **JUDGE_UNREADABLE_ALIGNED_MAPPING,SMD 在当前 fixed forecasting family 下资格关闭**。重训 12/100。两轮纪律无瑕:0.001 边距不调门、first-fault 停、v2 未回改。
**主线裁定**:(1) 门槛几何缺陷入册(仪器事实):0.40 份额门 rationale='3×均分'按 8-eval 标定,复用到 4-eval 实为 1.6×均分;规则改为**份额门=3×均分**(4-eval 即 0.75),数字 0.40 废止为几何特例。(2) **重读不重开**:几何修正下 #33 仍死 ep1 support(spread 9.87,2-9 占 52.6%)与 ep2 delayed(5.134),#32 主映射仍死 ep2 delayed(18.35);spread 门在 4-eval 只松不紧。**SMD 关闭对门槛修正稳健,维持关闭**。(3) S2 正资产与 SMD 失败解耦:v2 为已冻结交付物;SMD 之败是仪器在 server telemetry 上的可读性边界(章程 §5 Consumer/Metric 属实验仪器),非能力之败、非 Phase S 方法失败。(4) 242MB 原始 SMD 不入库(entity_structure 已含 upstream ref+校验)。(5) Phase S 供给决策回用户:真 Tier1 获取 / 停车封存 v2;NOAA 新区域不得冒充第三域(同源家族,载不动 Shared 跨域 claim)。(6) 与新域无关的最高价值下一刀=O7 guard held-out 化(A 窗选出、B 窗计分,用已有 NOAA 银行),解全部 guard claim 最大 caveat。

### 范围锁定与 Phase T 转向(2026-08-22,用户裁定,主线确认)

用户最终范围:**数据形态限定单变量;Task/Consumer、模型、Domain、Pattern 全部可变**——项目是"质量标准随任务/模型/模式变化"的 Data Readiness Harness,不是预测清洗 Harness。sol 两次收窄(→单变量→单变量预测)被用户纠正,主线确认:至今全部已入账证据都在 forecasting 一个 family 内,原始命题"同一处理在预测与异常检测上方向相反"从未被测过——Phase T 即回到该第一性命题。裁定:(1) SMD 排除理由改判"多变量形态不匹配"(非任务窄化),JUDGE_UNREADABLE 系数据-仪器形态错配非方法失败;(2) **筛选标准补一道门:数据原始任务语义与实体结构必须与目标 Consumer 匹配**,不得只看长度与离群 prevalence(census 标准修订);(3) Phase S 停车封存(v2 冻结带双向 LODO),在 Phase X 内按 family 复活;(4) 新阶段图 = **Phase T(task-conditioned:forecasting vs anomaly detection 双 Consumer)→ Phase M(model-conditioned)→ Phase X(各 family 内跨域 fresh)**;(5) T0–T5 不需新数据(注入正控+自有 dev 数据),T6 fresh 域与 AD 自然标签数据为已知供给问题挂 **O9**;(6) AD Consumer 按仪器纪律最小化建设:确定性检测器+事件级 F1+既有三联窗语义+**逐序列增益向量**(guard/选择器/RESCOPE 全套机器无改动直读);(7) T1 判定必须标 POSITIVE_CONTROL 等级(注入翻转可能是构造性的),自然翻转证据留 T4/T5,措辞不得混用。#35(T0)、#36(T1)任务书已发。

**#35 修订裁定(2026-08-22,sol 评审 REVISE_BEFORE_DISTRIBUTION,主线全部采纳)**:#35 v1 收回不分发,v2 修订六点:(1) AD Consumer 落位 `evaluation/functional/consumers/`(实验仪器,不得入 methods/ttha/);(2) 冻结**同字节契约**——同一注入块 B、同一 Program P、同一作用几何产出唯一 P(B),两 Consumer 读同一字节,处理侧零分叉;预测读未来/检测读块内的不对称属任务语义,非几何混杂;违约判 PROGRAM_GEOMETRY_UNALIGNED,否则 T1 只能叫 TASK_AND_GEOMETRY_FLIP;(3) T0 校准注入与 T1 正式注入硬隔离(不同 seed、不同块),T0 回退一旦启用当场冻结参数,T1 不得再调;#35/#36 不合并执行(主线收回"可连续跑"提议);(4) 可读性验收弃恒等式,改三条:同输入重复跑逐位一致、无注入孪生块只报 background alarm rate(不得冒充 FPR)、校准注入块 P/R/F1 有限且 F1≥0.5;(5) 事件 F1 钉死一对一贪心匹配、最小事件间距、窗口边界排除;(6) 注入协议禁用范围表达:密度、构成循环表、符号、两档强度与爆发占比、间距、边界排除、σ 来源(注入前合法前缀)、MAD=0 处理全部预注册为常数或确定性规则。另两项规划裁定:#39 注入正控证据**永久**标 evidence_grade=POSITIVE_CONTROL、不授 Shared Capability 执行权,自然标签 AD 数据承重在 #41;编号口径 = #34 不存在,主任务书 #35–#45 共 11 张,有界修复书另计。

**#35 分发勘误(2026-08-22,三方阅读代理交叉核对)**:(a) 纪律段"v7 注册表"勘正为现行 `FROZEN_SURFACE_V9`(39 项,机制为 runner 内清单+工作树 sha256,非独立注册表文件);(b) Part 0 预期入列追加 O8 两文件(`methods/ttha/harness/compiler.py` 与 `h0/snapshot.lock.json`,历史漏提交,属预期入列非漂移,本轮顺手闭 O8);(c) `methods/ttha/consumers/` 已存在但仅有 `__init__.py`,AD Consumer 仍按裁定落 `evaluation/functional/consumers/`,不得挪用前者;(d) T1 契约注记:forecasting 逐序列增益定义在 4 条 eval 序列上、注入与 AD 检测发生在 12 条 train 序列块内——同字节契约不受影响(两 Consumer 消费同一份 P(train 块)),但翻转判定在聚合层,逐序列对照只在各自任务内做,guard 直读检查用 AD 向量;(e) 校准块锚点:task_A 三联窗上下文+horizon 跨度约 [912,1392),[0,912) 为最早无重叠区,校准块按"最早等长段"规则落位,以 T0 B3 实测为准。

**勘误的勘误(#35 A2 实测后)**:(a) `FROZEN_SURFACE_V9` 实测 **40** 项非 39(§3.6 旧文过期);(b) **撤回**——O8 两文件已于 #30 检查点入库,§2 表行系过期未更(现已更);(c) 执行方按 v1 误建的 `methods/ttha/consumers/` 已删,现不存在。

### #35 T0 结果与主线裁定(2026-08-22)

判定 **T0_READY** 接受(0 LLM / 0 forecasting 重训 / AD 评估 176/200;检查点 7278317 / 4343d0a / 3314620;SAME_BYTE_CONTRACT_HOLDS,处理侧无分叉)。AD Consumer v1 冻结于**回退参数(窗 49 / 阈 3.5)**:primary(25/4.0)校准 F1=0.4052 未过线,回退 F1=0.5082 过线仅 +0.0082,回退已用尽。校准块 [143,431)(字面规则选出的 [0,288) 其首合法位 σ 前缀越出数组头,取最早可执行段,两读数在册);32 事件 0 skip;唯一漏检系 σ 尺度错配(注入幅度用 168 点前缀尺度、检测用 49 点尾随尺度,recall 31/32)。

七条歧义主线裁定:(1) **T1 注入落位 = 训练块**(阻塞项)——实测 P 只作用训练区 [120,900)、从不触及三联窗 [912,1392)/[1608,2088),原 D2 "T1 只在三联窗内注入"为主线笔误,按同字节契约改为:注入 12 条 train 序列的 **[431,900)**(避开 T0 校准块 [143,431);[120,143) 长 23 不足边界排除弃用),seed 20260823,eval 4 序列零注入;(2) 验收门实际按 background alarm 水平选中回退(两设置 recall 同为 31/32,F1 差全来自 precision 分母),记为**仪器内张力**,门照冻结协议有效,不改;(3) 过线 +0.0082 如实入册;(4) **AD 增益事件量子化**:每序列事件数少时 per-series F1 以 0.2–0.3 跳变,±0.005 材料线在 AD 侧的实义 = "至少一个事件易手",T1 翻转判定在聚合层,不声称 0.005 级分辨率;(5) σ 尺度错配保留为已知仪器事实,不改协议(改注入尺度会把刺激与仪器耦合,更糟);(6) 校准块落位与勘误 (e) 一致;(7) C5(iii) pooled 读法接受,per-series 不设门。#36(T1)据此定稿发出。

### #36 T1 结果与主线裁定(2026-08-22)

判定 **TASK_FLIP_CONFIRMED_POSITIVE_CONTROL** 接受,入册 **C12**(0 LLM / forecasting 重训 30/60 / AD 评估 72/300;检查点 26391a6;同字节断言 600/600 零分叉;冻结面零漂移——注册表口径修正:**40 原始项 / 39 去重**,两读数并存)。四臂全部同向翻转:forecasting delayed 聚合 {iqr +0.2723, mad +0.3255, hampel +0.0648, winsorize +0.4059},AD 聚合 {−0.1046, −0.0455, −0.2808, −0.2681};C3 不退化(identity pooled F1 0.6667)。**接线事实(独立入册)**:AD 逐序列向量直读现役 guard 语法 `min_per_series_gain`(compiler 原函数,零代码改动),四臂全部正确触发 −0.005 害线——#19 缝合的 Scope/Risk 机器对第二任务向量**原生可读**,T2+ 的接线风险实测下降。

五条歧义裁定:(1) **P 整块施用接受**——同字节契约本意即"一次施用、共享字节";代价已入 §1.3 caveat 7(幅度与逐窗菜单不可比,只承载方向);(2) warm-up 吃掉 [382,431) 结构性不可评,入册;(3) hampel 的 support/delayed 方向劈叉(−0.0890 / +0.0648)与 99999923908 再次越 forecasting 害线(−0.0904)是 **T3 任务条件化决策会撞上的真实读数**,单独留案;(4) 翻转方向单一(全为 F↑/AD↓),镜像方向(利 AD 害 F 的程序)本 family 未测——停车场项,T4 冲突证据设计时再议;(5) task_B 窗未评,书面 scope 如此,不补。执行方首跑 NaN-naive 比较缺陷已自查修复并重跑(两跑 ledger 字节相同、全臂读数逐位相同),处理方式接受。#37(T2 观察接线审计)发出。

### #37 T2 结果与主线裁定(2026-08-22)

判定 **TASK_CONTEXT_GAP_PATCHED** 接受(0 LLM / 0 重训 / 0 AD 评估;检查点 be02ab2)。审计确立三件事:(1) 公开视图确无 task/consumer/质量语义(task_kind="forecast" 硬编码于特征提取处且被 OBSERVATION_FIELDS 丢弃)——缺口由 `task_spec` 三项字段(task_id / consumer_id / quality_semantics)补上,`run_e2_skill_store_integration.py` +49/−0 纯增量,B3 三重物化验收全过(剥字段后 canonical sha 与真实 episode 录档 `public_input_sha256` 逐位相等;旧→新 diff 恰为新增字段;F→AD diff 恰为字段值之差);(2) **卡层 task 维度已在**(两卡 gate `task_kind=="forecast"`,词汇域含三任务)——T3 检索合法性过滤可直接分任务;(3) **T4 范围就此钉死** = episode 键补 task 分量(现为 `batch:<cohort>|consumer:<variant>` 无 task)+ 卡词汇补 consumer 特征,不多不少。四条歧义裁定:英文质量语义接受(提示词语言);`per_channel_ridge_a1` 命名入册 canonical;schema_version 不升版接受(升版破坏 B3 验收,现无读者依赖,读者出现时再议);AD 变体走 override 物化属实——live caller 由 #38 接线。ssi 为 V9 冻结成员(`37d31cb8…→f39c13f3…`),注册表更新授权在 #38 Part 0 执行(#18/#19 先例)。#38(T3)发出。

### #38(T3)撤回与 T1b 转向(2026-08-22,sol 审核,主线采纳)

#38 v1(T3)在零消耗状态撤回(未建 runner,未花 LLM/重训)。sol 指出的 **estimand 不对称**成立:T1 中 forecasting 在 P(B) 上训练、在未处理未来窗计分(训练数据效用),AD 零训练直接检测 P(B)(推理输入效用)——翻转声明混杂了"任务不同"与"消费模式不同",而项目主命题承重训练侧。裁定:(1) C12 改判 `ADJUDICATED_ESTIMAND = INPUT_SIDE_TASK_FLIP`,raw 判定保留,输入侧为真实部署场景、资产保留;(2) T0 robust-z 改标**注入可见性仪器 / 烟测 oracle**,不再是主 AD Consumer(冻结参数 49/3.5 在该角色下沿用);(3) T2 不受影响(task_spec 字段值无关训练模式;可训练 AD 的 consumer_id 注册 `ad_ridge_train_v1`,替换字段值即可);(4) 新 #38 = **T1b 训练侧任务翻转正控**:两 Consumer 都在同一 P(B) 上训练、都在固定未处理独立 Query 上计分;可训练 AD 复用仓库自有加权 ridge 闭式解(不引入 sklearn),标签取自 T1 ledger 且不随 P 变化;**双 Query 隔离**(主线加严,沿 T0 校准/正式隔离纪律):可读性门与回退选择只看校准 Query [2600,3060) seed 20260825,正式 Query [2100,2560) seed 20260824 只在计分时打开,两区均避开 task_A/task_B 全部上下文+horizon 跨度;(5) T3 修订为 #39,门控在训练侧翻转确认上,C1 改为**方差参照判据**(跨任务提案差异须大于同任务重复抽样差异,K=3+3),原 Jaccard<1 判据废止(C11 抽样方差在册);(6) 编号顺延:#38=T1b,#39=T3,#40=T4,#41=T5,#42=T6,#43=M0,#44=M1,#45=X 复活,#46=整合,主书 12 张。**预期机制注记**(负结果解读用):修复使正标签位置的特征恢复正常形态,训练出的分类器无从分离 → Query 检测退化;若未退化即为可信负读数,T3 保持暂停、回用户决策。

### #38(T1b)v1/v2 双停与 v3 授权(2026-08-22,主线裁定)

**Part 0 入账**:sha a6ba53d,6 文件(#37 交付物 + V9 注册表 ssi 授权移动 37d31cb8…→f39c13f3…,`T2_OBSERVATION_TOUCHED` 沿先例 + 两份 docs 修订);live 哈希与注册表一致,授权移动非漂移。

**v1 判定成立并补根因**:`AD_TRAINABLE_SPEC_DEFECT` + `FEATURE_LABEL_GEOMETRY_MISMATCH / CURRENT_EVENT_NOT_OBSERVABLE`——排他尾随窗 [t−49,t) 不含 x_t,分类器被要求判断它看不见的点。**定性为主线规格缺陷**(把 forecasting 滞后几何照搬进点事件检测),非执行错误。门读数 49:0.1709 / 回退 25:0.2606(Qcal pooled),oracle 同批字节 0.7458 → 注入可见、规格不可读。

**v2 判定成立**:含当前点窗 + 宏平均 + 先 Qcal 后 Qf 顺序下,门仍两次不达(宏 0.1765 / 0.2942),较 v1 仅 +0.03 量级 → 缺陷超出窗几何,定位到**特征族×线性头**:线性 ridge 读原始标准化窗无法表达 |x_t−med|/MAD 类稳健统计。v1+v2 合并为"原始窗×线性 ridge×稀疏点标签"规格族的**可信负结果**(两种几何一致)。Qf 全程零读取,五臂未释放,零翻转声称。

**过程诚实入账(两起,均未污染读数)**:fit 广播 bug(weights[:,None] 对 1-D 标签广播成 (n,n))修复于任何门读数之前,修后两跑逐位一致;_Blocked 路径落盘遗漏补齐,注入目录 28 文件 sha 逐字节相同。成本累计 ≈202/300(v1 三跑 ~126 + v2 两跑 76),LLM 0、重训 0。接受。

**歧义裁定(四条全部立为正典)**:(1) 循环计数器每 Query 区从槽 0 起步(沿 T1 独立 seeded draw 惯例);(2/3) 宏平均(逐序列 event-F1 均值)为门与判定主口径,pooled 降为副读数,量子化注记 ≈0.02/事件(每序列 4 事件粒度);(4) 168 隔离的执行解释 = σ 前缀 pristine 重算 + 区前 168 步字节与 pristine 相等断言;Query 特征读区前 49 原始字节 = 与 T0 检测器同尾随几何,零泄漏。

**仪器取代关系(站规补记)**:`anomaly_detection_trainable_v1` = 化石(排他窗规格缺陷);`trainable_v2` = 化石(含点窗,族内可信负);现役接替 = `trainable_v3`(任务原生特征,本轮授权)。`anomaly_detection_v1`(robust-z 49/3.5)继续任注入可见性 oracle。

**外部分析裁定(sol+grok,主线采纳)**:主因 = Consumer 表示缺口,采纳。TIMECLAW 无可训练 AD Consumer(LLM 感知问答 + 工具,无可抄);AegisTS 可借**原则**(Consumer 任务原生、观察层与效用层分离)不可借**栈**——MiniRocket/InceptionTime 为整序列分类几何,与点事件错配;TimesNet/LSTMAD 本地有源码(a-evolve/AegisTS),不采用的真实原因是 estimand 不匹配(重构族把训练异常当污染,清洗可能双任务同改善,不构成翻转正控),留给日后自然无监督 AD 线;冻结 robust-z 阈值不能任主 Consumer(不从 P(B) 学习)。**整序列分类定性修正(sol,主线采纳)**:用户已定 Task/Consumer 可变,该路不是项目身份变更,而是**新 Capability family 与新实验问题**;若 supervised-AD 正控族关闭,主线可在既定范围内自行裁定转向并在台账记明理由,阶段汇报时告知用户,不构成强制检查点。

**v3 授权(同书同 runner --v3,非新书;sol 预分发审查修订版)**:sol 在分发前证伪了三特征方案——`detect()` 只公开 scores/flags/弃权计数,不公开逐点 median/MAD,f2 无法在"复用同一路径、禁止重写、原档不动"下实现;"MAD=0→[0,0,0]"亦非 T0 弃权语义(零向量过含截距 ridge 可能误报);两处均为主线书稿缺陷,分发前改正。修订后:唯一改动 = **单特征 f1 = z_t**,经 `anomaly_detection_v1.detect(values, window=49, threshold=3.5)["scores"]` 同一路径一次取得(**显式传冻结回退参数,文件默认为 25/4.0,不许吃默认**),禁止重写 median/MAD;非有限 score 全部沿 T0 弃权语义:训练时该点不入拟合并记数(正类权重只在入拟合行上计算),Query 时强制不触发,AUPRC 只用有限特征点的 Consumer score 排序计算并报未定义点数,ledger 事件不可评分自然记漏检。头/标签/P(B)/臂/Qcal/Qf/计分/顺序全沿 v2 冻结。**无回退、单发**:门不达 → `SUPERVISED_AD_PC_FAMILY_CLOSED`(三规格一致可信负;转回主线在既定范围内裁定,阶段汇报告知用户)。预注册口径注记(两句均强制入工件):该 Consumer ≈ 在任务原生充分统计量上学习的阈值头,翻转只对该 Consumer 族发声;**v3 若闭合,只证明在任务原生充分统计量可见时,训练数据效用翻转可被仪器读出——不证明 Harness 自己发现了该表示,也不代表自然异常数据上的泛化**。Part 0 修正:`_scratch/t1b_query` 在 .gitignore(L26),不入版本库,继续作只读运行底物(完整性走 sha 快照);Part 0 只提交代码、v1/v2 工件与 docs。预算:v3 切片 AD ≤120,T1b 累计上限 300→400(两次诚实停机耗损,主线批准),LLM 0、重训 0 不变。

### 架构健康评估裁定(2026-08-22,主线采纳)

外部评估结论采纳:方法核(methods/ttha 22 py + contracts/runtime/operators)小而分权,九环对应模块而非脚本堆;实验层(evaluation/functional 257 py、run_e2 约 96 个、e2 工件 496 份、主管线 5000+ 行叠 V3→V9 清单、AD 仪器三叉)是**刻意保留的实验化石层**,不是产品包——按章程旧 runner 与旧证据不删,故只会单调增高。裁定:(1) **机制轮中途不重构**(章程问题"不做它核心实验是否无法运行"当前答案为否);(2) 唯一整备窗口 = T5(#41)收口后、T6(#42)fresh 冻结前,修复书 **#41b**,行为保持 + 双 runner 重放逐字节验证,产出 V10,使 T6 与 X 两次 fresh 打开都发生在整备后代码上;不放在 #46,因 fresh 轮是 first-fault 归因成本最高处,不该跑在沉积峰值上;(3) 提前触发条款与即刻站规见路线图 §3.5(仪器分叉须同轮落取代关系一行,trainable_v1/v2 待 T1b 报告落地补记)。整备不计方法进展(章程 §9),报告作附注。

### #38(T1b)v3 结果与 T1b 关卷(2026-08-22,主线裁定)

**判定采纳**:`TRAINING_SIDE_TASK_FLIP_CONFIRMED_POSITIVE_CONTROL`(C13 入册)。Part 0 sha 359eec5(9 文件)入账。A4 门一次过(Qcal 宏 0.6109;解锁链 0.1709→0.1765→0.6109 定位在特征族)。护栏全绿:C2 同字节 600/600、P(B) 重算逐位一致、B3 120=120、C3 复用 T1 工件且重建增益与录档一致(0 重训)、V9 39 文件零漂移、anomaly_detection_v1.py 零改动、注入拷贝 sha 前后一致(复冻结仅 protocol.json version 标签 v2→v3)。

**承重发现两条**:(1) **程序特异性**——iqr/mad/hampel 两任务同向为正,唯 winsorize(forecasting 最强臂)翻转;训练数据质量是任务×程序联合条件化的,且为 T3 提供了比任务语义字符串更细的答案钥匙(AD 合宜集 = {identity, iqr, mad, hampel},唯一有害臂 = winsorize;F 合宜集 = 四个修复臂)。(2) **AUPRC 五臂全同**(0.8878)canonical 化:单特征正斜率下 Consumer 排序≡z 排序,翻转全部承载在 0.5 阈值头的映射位置(caveat 8)。

**歧义裁定(四条全部沿执行解释立为正典)**:168 隔离沿 v2 实现;threshold=3.5 对特征路径惰性(只用 scores 不用 flags)作 provenance 事实;训练块特征自块内 index 49 起、Query 读区前 49 原始字节的训练/查询不对称沿 v1/v2 正典(代价 49 行/序列,不实质);循环计数器每区槽 0 起步沿旧。仪器事实入档:序列 72329003935 zero_scale 12;Qcal 未定义点 289/5520 全归因基底缺测段。

**T1b 全卷收官**:v1/v2 = "原始窗×线性 ridge"规格族可信负 + v3 = 训练侧翻转正控确认;成本累计 AD 评估 280/400、LLM 0、重训 0。**T3(#39)门控条件满足,解锁。**

### #39(T3)预分发修订(2026-08-22,sol 审核 7 点 + 主线 3 处收紧,分发前并入)

1. **定位收窄**:本书只证任务条件化**提案**,执行/采纳/反馈留 T4/T5;弃用 `TASK_CONDITIONED_DECISION_CONFIRMED` 名。判定阶梯:完全分离+聚合方向 → `TASK_CONDITIONED_PROPOSALS_CONFIRMED`;再加 3/3 逐序列 Risk → `TASK_CONDITIONED_RISK_AWARE_PROPOSALS_CONFIRMED`;分离成立方向不满 → `TASK_SEPARATED_DIRECTION_INAPPROPRIATE`(主线补名);分离不成立 → `SAMPLING_VARIANCE_DOMINATES`。
2. **Part 0 纯元数据勘误**:v3 runner/工件内嵌检查点引用仍为 a6ba53d/6 文件,实际 359eec5/9 文件;以**追加 erratum 字段**更正(冻结工件不静默改写),读数零改动,不重跑。
3. **AD task_spec 语义串升级并冻结**(注册在 ad_ridge_train_v3 名下,逐字):"good preparation preserves the event signal needed to train the AD Consumer and improves event F1 on an untouched query";旧 T2 串保持绑定 ad_v1_49_35(输入侧),不删;新串只述任务目标,零任务→动作映射。
4. **店态冻结**:新建空店快照(0 Guidance / 0 Experience / 0 Skill),六 draw 读同一快照;"优先空店"措辞废止。
5. **旧语义冲突清障**:ssi 输入的 target.consumer_variant=pooled 等 forecasting 遗留字段不得把 AD 臂重新描述成 forecasting;A1 跑前烟测断言 task_spec 为唯一权威任务描述;两臂可加同一条中性说明,禁任务→动作映射;两臂完整 prompt 逐字入档。
6. **双层答案键且 runner 内从冻结工件推导(禁手抄)**,harm 线 −0.005,identity 增益按定义 0:聚合方向层预期 F={iqr,mad,hampel,winsorize}、AD={identity,iqr,mad,hampel};Risk 层(逐序列无越线)预期 F={iqr,mad,winsorize}(hampel 最差序列 −0.0904 出局)、AD={identity,abstain};推导键与预期不符按歧义上报。**预注册解读**:空店无经验下 Risk 层不达属预期可能,直接构成 T4 冲突 Experience 的入口证据,不是缺陷判定;AD 逐序列 4 事件粒度下单事件易手 ≈0.2+,Risk 层近似"是否避免触碰",量子化注记随行。
7. **协议钉死**:后端 = gpt-5.6-sol;重试每 draw ≤1(6+6=12 封顶,修正原书 ≤2 与总额 12 的冲突);abstain 在 Jaccard 中记 {__ABSTAIN__} 单元素集(避免空集距离未定义);OFF_MENU 该 draw 无效、不重掷、排除出距离矩阵并自动破坏该臂 3/3;无效 draw >2/6 → `EXAM_PROTOCOL_UNREADABLE`。

### #39(T3)结果与关卷(2026-08-23,主线裁定)

**判定采纳**:`TASK_CONDITIONED_PROPOSALS_CONFIRMED`(C14 入册)。Part 0 sha bd5922d(6 文件)入账;t1b v3 工件勘误以追加 erratum 落点(a6ba53d/6→359eec5/9),冻结字段未改写,核可。A1 烟测 9/9:两臂 prompt 剥除 task_spec 后 canonical sha 相同(96ce9fc9…),system 同一份(0a657752…),AD 臂零 forecast/sMASE 词、F 臂零 anomaly 词,信息墙零增益/翻转泄漏;prompts_verbatim 三 sha 入档。

**第二次抽样裁定(有效)**:首跑 6 draw 死于 mappingproxy 写盘缺陷,draw 1–4 不可恢复,暴露片段仅 F5=outlier_mad / AD6=identity。裁定理由:(1) 死因为实现缺陷非读数不利;(2) 暴露片段本身即过关方向,无择果动机;(3) 第二次抽样 F 侧结果(hampel,Risk 出局)较暴露片段(outlier_mad,Risk 合宜)更不利,与择果签名相反。首跑片段存档,不计入判定;LLM 预算 12/12 收口。

**承重发现**:(1) 任务轴第一条 Harness 侧证据——仅 task_spec 字节差即完全翻转提案(F {hampel,mad} → AD {identity});(2) **先验-经验风险缺口**:F 臂 3/3 选 hampel,LLM 先验把中值滤波当安全牌,而冻结键中 hampel 恰是唯一被逐序列 harm 线踢出 F Risk 键的温和臂(−0.0904),先验以为激进的 winsorize 反而四序列全正——此缺口无经验不可弥合,即 T4 headroom 的直接演示;(3) AD 臂 3/3 纯 identity 短单 = 谨慎剖面,与 C13 合宜集吻合;(4) 同任务 top-1 稳定 3/3(C11 的 7 抽 3 单为不同店态/上下文,不互推)。

**店态口径修正(正典)**:"空店" = 0 Guidance / 0 Experience / 0 learned Skill + 机内 h0 bootstrap 常量(本次 3 个,任务中立,六 draw 同快照 c8c1e452…);#40 种子店必须在同一快照上**加且仅加**经验条目。工作树未提交 ROADMAP +27 行为主线沉淀/止损口径编辑,#40 Part 0 一并提交。

**当前最早阻塞声明(sol 复核一致,主线入册)**:任务识别已闭合(C14 两层拆分:提案条件化已证,风险感知未证);第一阻塞移至"如何利用任务化正/负/冲突 Experience 避免局部伤害"——#40 即为此而设。**缩 Scope 边界裁定**:#40 考试菜单为保 #39 基线可比,冻结为 5 程序 + abstain,不含 rescope 动作;Agent 自由文本中自发提缩 Scope 按现行协议记 OFF_MENU 入档(有价值行为观察,非协议破损、非缺陷);缩 Scope 作为正式动作属 T5 真实执行环(guard 语法已有 RESCOPE_MASK_HARMED_SERIES)。

### #40(T4)预分发修订(2026-08-23,sol 审核 5+2 点,主线核实采纳并定键)

事实核实:`fast_agent.py:772` 运行时规范键 = `task_type|downstream_model_class|metric`;`ordering_card.py` scope 四键分立(task/domain/downstream_model_class/program_family),domain 本为独立维度;ssi 三处(L368/866/1222)写 `batch:<cohort>|consumer:<variant>`,无 task 分量——方言分裂属实,原书第三格式若落地即 T5 假闭合。修订:

1. **键统一(主线定夺)**:不造第三方言。写入与检索共用运行时规范键 helper(`task_type|downstream_model_class|metric`;AD 值由现役 helper 从 task_spec 推导,不得手造);cohort/domain 只进 domain scope 字段与 Context,**不进任务硬键**(护 T6/X 跨域检索);旧工件不迁移不补写。A1 从"键加 task 分量"改为"ssi 写入路径接运行时规范键 helper"。
2. **生命周期分类必须看见"聚合正、局部有害"**:预注册机械分类(冻结阈值推导,非答案键)——identity→ABSTAIN;agg≥+0.005 且 min(per-series)≥−0.005→POSITIVE;agg≥+0.005 且 ∃per-series<−0.005→CONFLICT;agg<−0.005→NEGATIVE;近零→NEUTRAL。产出:F = iqr/mad/winsorize POSITIVE + hampel CONFLICT;AD = winsorize NEGATIVE + iqr/mad/hampel CONFLICT + identity 中性。不修则 hampel 被记 POSITIVE,T4 反而强化 #39 错误选择。
3. **写入必须落 Runtime**:build_episode → TTHAMethod.append_experience_episode → **同一 TTHAMethod 的 prepare/检索可读**(报告内构造不算写入);10 条全部 evidence_level=DELAYED、local_status=EPISODE_ONLY、不晋 LOCAL_DRAFT、不授 TRY/自动执行权、普通唯一 ID——历史重放非新证据,防重复计数(章程)。
4. **B4 改类别验收**(防 ID 排序塑形):F 必回 hampel 局部伤害 CONFLICT + 至少一个 {iqr,mad,winsorize} 无害 POSITIVE;AD 必回 winsorize NEGATIVE + 至少一个修复程序 CONFLICT;两臂零跨任务取卡;卡载事实摘要(consumer/聚合方向/harmed count/min gain),禁"应选 X";不钉死实例名。
5. **判定补 PARTIAL_EXPERIENCE_CONDITIONING 统一兜底**:F 1/3–2/3 改善、分离丢失、不安全位移等全落此格,必报字段 = F/AD 安全次数、分离读数、AD 风险回退标志;原 EXPERIENCE_SHIFT_RISK_REGRESSION 并入必报字段不单列。
6. 小修 a:撤销 B3 新店 sha,保留 h0 快照标识,报 0→10、10 个 episode ID、检索日志(反过度工程)。
7. 小修 b:C2 拆三向比对(T4-F vs #39-F 仅经验块异;T4-AD vs #39-AD 仅经验块异;T4 两臂互比仅 TaskSpec+各自经验块异)。

科学定位(sol 措辞入册):#39 证明 Agent 能读懂不同任务;#40 检验 Harness 写入的任务化成功/失败/局部冲突经验能否纠正 F 的风险盲点,同时不破坏 AD 的保守选择。通过即 MECHANISM + POSITIVE_CONTROL 级 Memory 能力;执行/采纳/Delayed 写回/Local Skill 更新留 T5。

### #40(T4)结果与 #40b 修复切片授权(2026-08-23,主线裁定)

**判定采纳**:`PARTIAL_EXPERIENCE_CONDITIONING`(C15 入册),按预注册兜底格,执行方零自裁。Part 0 fd29501(5 文件)入账。Memory 面 diff 4 文件 +269/−16;V9 触碰 = ssi + methods/ttha/method.py,注册表授权移动在 #40b Part 0 执行。测试 36/36 改动前后各一跑。

**歧义裁定**:(1) 执行读法即原意——任务硬键(task_consumer_key,经 task_spec 工厂铸造)与单元键(cell_key,batch|consumer,承载 leave-one-cohort-out 语义)分立两函数、三处集中铸造、cell_key 输出字节不变;书面"三处接同一 helper"为主线措辞缺陷,ssi L124 死方言常量拆除核可。(2) **方言负债普查入册,路由 #41b**:task_episode_harness/t1.py:88(被 ~15 模块 import)、run_v1_fastpath.py:50、run_v1_fastpath_framework.py:62(第二方言 forecast|ridge_smase)、run_v1_guidance_evolution.py:5834、experience_memory.load_episodes_from_v6_reports 内三条历史 Episode 键 = forecast|ridge_smase(现役检索永不可命中);依"旧工件不迁移不补写"本轮零触碰。(3) **bundle 身份盲区入册停放**:runtime_bundle_sha 的 dependency_shas 按名收录 fast_agent/method 而不含 experience_memory,Memory-only 改动不动 bundle 身份;T5 证据冻结继续以 V9+git diff 为准,修复候选停放 #41b/整合,本轮不修。(4) 39/40 悬案关闭:V9 原始 40 条、唯一 39 文件,重复项 = artifacts/functional/e2/noaa_fresh_cohort_v2.json,#35 勘误与执行方读数各自成立。

**A2 分类丰富(答案键再证)**:AD 侧三温和修复全 CONFLICT(iqr 5/12 harmed min −0.27;mad/hampel 各 4/12 min −0.20),winsorize NEGATIVE(9/12,−0.50);Risk 键 {identity} 由工件推导第三次一致。

**方法发现(承重,入册)**:孤立的"聚合改善"事实卡在无对照可比时误导谨慎剖面——F 臂因同屏有无害 POSITIVE 卡而受益,AD 臂无对照则"Aggregate direction: improved"单独站立即成误导。**经验呈现必须能表达"什么都不做是安全读数"**。AD 回退与 #39 的 F 犯错同构(聚合诱导),证明该错误类是呈现层通病而非任务特异。

**#40b 授权(同 Memory 面有界修复)**:唯一改动 = 卡表达范围三件套——(i) _hard_filter abstain 资格;(ii) ContrastPack abstain 第四通道;(iii) ABSTAIN 事实句(零祈使)。**卡序(聚合先于风险)本轮不动**,预注册为唯一后备面。重考照 #40 协议逐字。

**#40b 预分发修订(2026-08-23,sol 三硬修 + 一命名 + 一删减,主线采纳、勘察按后备条款收紧)**:(1) **A1 改为 Runtime 天然可达**:relation=ABSTAIN 且 workflow_signature=identity 的 Episode 只绕过 informative-operator membership 检查,仍须通过 response_validity / task_consumer_key / pattern_view 等全部其他过滤;不依赖 identity ∈ allowed_operators(其在真实 Runtime 来自 Operator registry,identity 属 incumbent/no-op 未必注册——原写法 = 测试专用通道假闭合);unknown 照滤。(2) **B1 改为重新物化**:#40 十条未持久化(住在当次 TTHAMethod 实例),新进程无"#40 店";在全新 TTHAMethod 实例中自 #40 v1 工件重新物化同 10 条、再经 append_experience_episode 写入,逐条断言 to_dict 与 v1 工件一致;new_independent_evidence = 0,不得称新增试验/新增独立证据;"沿用店"为主线事实错误,入册。(3) **判定按优先级补全**:RETRIEVAL_MISS → EXAM_PROTOCOL_UNREADABLE(>2/6)→ TASK_SEPARATION_REGRESSION(分离失效,附双臂 Risk 次数)→ CONFIRMED(F 3/3 ∧ AD 3/3 ∧ 分离)→ EXPERIENCE_SHIFT_RISK_REGRESSION(F<3/3 ∧ AD=3/3)→ CARD_CHANNEL_INSUFFICIENT(F=3/3 ∧ AD=0/3)→ 其余合法混合(含 AD 1–2/3)全落 PARTIAL_EXPERIENCE_CONDITIONING。(4) 通道结构字段唯一命名 `ContrastPack.abstain`(卡面可称 no-action baseline);旧行为断言收窄为"abstain is None 时最终渲染 prompt 字节不变",不苛求 to_dict 序列化不变。(5) T5 静态勘察保留但收紧:仅 CONFIRMED 触发,固定四入口(operational pipeline runner / fast_agent / online_loop / method),≤10 条缺口,只读零修复,不追踪全仓库。F 保 3/3 + AD 回 3/3 + 分离保持 → `CONFLICT_EXPERIENCE_CONDITIONS_PROPOSALS_CONFIRMED`,T4 关卷。

### #40b 结果与 T4 收束、#41(T5)转向(2026-08-23,主线裁定)

**判定保留**:`TASK_SEPARATION_REGRESSION`(C16),按预注册优先级第 3 格,执行方零自裁。Part 0 fbba86f(10 文件)+ V9 两项授权移动(ssi 第二次 f39c13f3…→0dbe61d9…;method.py 首次 e9c27af3…→cd28df33…,T4_MEMORY_TOUCHED 沿先例;注册表自身触碰与 a6ba53d 同构非漂移)入账。A4 旧行为断言 4/4、B1 重新物化 10/10 字段级全等(字面 to_dict 比对因 #40 工件只存 16 字段摘要不可执行——落法接受,本轮工件已补 episodes_to_dict 使下轮可字面比对)、B2 烟测 5/5、B3 三向 7/7、测试 36/36。

**分离失效诊断立为正典**:非任务趋同——top-1 层完全不相交(F 全 outlier_iqr;AD identity/identity/hampel),6/9 跨对仍 1.0;失效源 = abstain 卡使 `identity`(两任务共有合法弃权项)同时进两臂 shortlist,同抬跨任务重叠与同任务离散。**判据教训(只前瞻不追溯)**:未来提案考试以 top-1 分离为主判据,shortlist Jaccard 降副读数并为共有合法弃权项设例外;本轮不改判、不为此新造考试。

**T4 收束(sol 审计口径入册)**:任务键控写入/检索 = 支持(POSITIVE_CONTROL);Experience 修正 F 风险盲点 = 支持(3/3);abstain 通道改善 AD = 部分支持(0/3→2/3);双任务安全提案闭合 = 不支持;真实执行/Delayed 更新/Local Skill = 尚无证据;独立性 same-context,provisional。**不做 T4c 卡序修**:同一正控、同一模型、同一答案面上第三次调呈现 = 把 Memory 调成答案 Router;AD6 的 hampel 提案应由 T5 的 Support 风险门与 delayed CONFLICT 拦截,不由卡面措辞消灭。

**注册表事实(执行方实测,T5 第一阻塞)**:identity 不在 OPERATOR_METADATA;outlier_iqr/outlier_mad/hampel_filter/winsorize 的 allowed_tasks = ('forecast','classification') 不含 anomaly_detection → 真实九环入口下 AD TaskSpec 拿到空算子集(fast_agent.py:65 过滤);online_loop.py:117 写回硬编码 forecast|ridge|sMASE;delayed 生命周期只读聚合标量。**T3/T4 已证 Agent 与 Memory 会任务条件化,但均为 proposal exercise;#41(T5)= 接入同一真实 Harness 的单入口双 Consumer 生命周期闭合**,sol 草案主线采纳 + 四钉:Part B 全绿门控 Part C、≤1 新 runner 复用现役 run_online_round/open_delayed 入口、Part C 全新空店(行为改变证据必须来自轨迹内自写经验,不预种 T4 卡)、写回键回归断言(初版为逐字节等同,后修正,见下)。

**#41 分发前修订(sol 评审六缺口,2026-08-23,全采纳;主线复核三处代码事实属实)**:(1) 单 `_pending_update` 槽(method.py:268)+ ADD 带 ABSENT 前置(:574)→ F/AD 轮次交错开 delayed,否则 pending 被覆盖或第二次 ADD 硬失败;(2) handle_feedback_delayed 现行门 = `dg ≥ −0.005` 即批准(:1335-1343),读不到逐序列伤害且 NEUTRAL 也扩权 → **method.py 列入必要接线面**,收紧为 classify_relation=POSITIVE 才批准(本轮唯一行为机制,任务一致适用,NEUTRAL 不再扩权不算回归);(3) Skill ID `fast_winner_{op}` 三处字面(:568/:572/:578)跨任务撞名 → 任务化无哈希 ID,applicability 本轮只声称 task_kind 隔离(consumer 特征不在 Observation 词表,不偷加);(4) **主线四钉之键断言为主线自误**:旧硬编码 forecast|ridge|sMASE 本身是方言(真实 F Consumer=pooled_ridge_a1),逐字节等同断言只会逼出假绿或假红 → 改双断言(legacy fixture 等旧字面;实际 F 等 task_consumer_key(actual),允许且预期不同),键迁移显式入回报;(5) 菜单等同需同一函数生成两臂相同 allowlist/forbidden set 并断言 operator contract ID 集全等;adapter 只做 steps→task-native 读数,不得决定 relation/winner/Risk/Skill(防第二套 Harness);(6) B6 烟测格(POSITIVE→Draft→delayed 聚合正逐序列害→CONFLICT→撤权)+ 判定集补 AGENT_PROTOCOL_UNREADABLE / INCOMPLETE_LLM_BUDGET / NO_CONFLICT_FEEDBACK_SAMPLE / EXPERIENCE_RETRIEVED_BEHAVIOR_UNCHANGED;LLM 预算 12→**16**(4 轮×3 阶段名义 12 + 4 重试余量)。不加只读预检书。

### #41(T5)结果、八歧义裁定与收尾追认(2026-08-23,主线裁定)

**判定维持 `INCOMPLETE_LLM_BUDGET`(C17)**,不升格不重跑。Part A 11/11(两臂 contract ID 集全等 = 四离群程序;identity 实测不在 OPERATOR_METADATA 由 Runtime 保留)、Part B 22/22、Part C 三轮完成 + AD r2 于第 16 次调用处斩(AgentCallBudgetExceeded)。**预算缺陷为主线协议错误第二例**(#39 重试算术后再犯):现役 prepare 三阶段各带 validation_retries=1,真实 ~5 call/轮,四轮理论上限 24 ≠ 书面 16;缺陷入账,不以第二次抽样覆盖第一次。**不重跑理由(sol 口径采纳)**:runner 每次全新 Method/Store 无 resume 入口,重跑 = 第二次抽样;为注入正控建恢复基础设施不值;未证项(AD r2、真实 Skill 激活复用、Memory 直选安全计划)移交自然阶段承重。C17 claim 语句以台账行为准,不得表述为"任务条件化生命周期已闭合";独立性 same-context/provisional。

**八歧义裁定**:(1) 预算算术 = 主线认领,见上;(2) A5 任务化 ID 撞坏另一线 5 测试 = 收尾修复追认(见下);(3) handle_feedback_support(Slow 侧)未同步收紧 = 正确,单假设纪律;已知不对称入册,Slow 下次启用时的第一接线项;(4) F 臂 roster 12 内 8/4 切分 = 器械选择,如实披露,POSITIVE_CONTROL 下接受;(5) AD adapter 两约定接受并入册:动作区 = 三个 verifier 窗拼接 [120,840)(48 注入事件中 44 在内)、读数以 −macro_f1 交执行器使其 baseline−candidate 算术直接给出 F1 增益(约定在 adapter 与工件明示,不埋代码);(6) r1/r2 同 origin 使 memo 免费重读 = 书面字节等同要求的确定性后果,接受,仅 cache miss 计预算;(7) part_c.llm_calls 原始字段误读 backend.returned_models,以零成本 --annotate 更正保留 raw 值、不重测 = 接受(异常在第 16 次后才抛,语义钉死真值 16);(8) 快照店落 %TEMP%/t5s = 环境事实接受(64 字符 sha 目录 + 任务化长文件名 + Desktop 同步进程句柄 → os.replace WinError 5,读似生命周期故障实非),入仪器名册环境注记,不建基础设施。

**收尾追认(sol 方案经用户分发,视同授权;0 LLM / 0 重训)**:687af6e 七件入库,t5 工件两次被烟测覆写两次从提交还原、终态与提交逐字节一致——处置正确。V9 登记:method.py 第二次(cd28…→ccf2b837…,T4 条目不覆盖)、e1.py 首次(3891…→e550…),after 值回读核符。**分类出入接受且是本轮最有价值的发现**:skill_evolution.py:782 与 e0b.py:996 各有一份手拼 `fast_winner_{sig}` 副本(书面未点到),ID 规则改动后与 manifest 失配 → E0_MECHANICAL_ERROR;收口为 method.py 公开 `fast_winner_skill_id()` 单一出处,两调用方改调——手拼方言在无人看管处静默腐烂的直接实证,键方言教训的第二次应验。e1.py 走新谓词 `_is_local_skill_id()` 兼容旧拼法,旧 store 仍可读。test_f1_forecast_pilot = savgol→scipy lstsq 原生 MKL 崩溃,eb4a03c 干净 worktree 复现 = 既有环境故障,与 #41 无关,路由 #41b(skip 标记 + 原因串)。复跑:四测试 18 passed、Part B 22/22、Part A 11/11、冻结面 ok、全量 functional 248 passed/3 deselected。收尾六件未提交,归 #41b Part 0。

**下一步裁定**:**#41b 先行,#42 随后**——与 sol"直接进 #42"的唯一分歧。理由:(a) 整备窗口为既定裁决(T5 收口后、T6 fresh 冻结前,唯一窗口,V10 为 T6/X 冻结面);(b) O9 勘察(2026-08-23):本地无 Yahoo S5/NAB/UCR 任何供给,#42 今日物理上无数据可冻;(c) T5 大接线 + 收尾补丁堆未提交,直接在其上冻 fresh 证据重蹈仪器漂移债。#41b 五项封顶 + 挂账清偿附录(键方言、v6 不可达、bundle 盲区、MKL skip、%TEMP% 注记),0 LLM,重放门控。

### #41b v1 撤回 → #41b-lite;#42 定稿(2026-08-23,sol 审核采纳,主线核验)

**#41b v1 撤回(同轮 supersession,主线认领)**:(1) A1 前提过时——run_online_round/open_delayed 实测已在 online_loop.py:289/:722 独立承载,T5 runner L99 直接 import,无物可抽;(2) A5 重放预算不可兑现——t1b --v3 工件记录 ad_evaluations=78 > 书面 ≤40,必然中断(**主线预算算术第三例**;根因 = 照搬 08-22 五项裁决未对 T5 后代码重新推导);(3) 名册/INDEX/归档不阻塞 T6 延后 #46;键方言只登记不扫仓;**runtime_bundle_sha 扩依赖违反 AGENTS.md 反过度工程条款,撤销**——Memory 覆盖改由 experience_memory.py 直接纳入 V10 既有清单;MKL 保持挂账不加 skip。**运维发现(sol,载重)**:git status --short 当前为空但六个收尾文件逐一与 HEAD 实异(stat 缓存未识别保时间戳修改)→ Part 0 必须先 `git update-index --really-refresh` 再生成清单,否则空提交假成功。

**Freshness 改判**:NAB realTraffic/realTweets 在 sol 检索中暴露部分异常窗 → context/outcome = INSTANCE_SEEN,不得作 virgin Target;**#42 Target = realAdExchange 六条全纳**(CPC×3/CPM×3,MIT,单变量带时间戳;公开聚合披露"六条中一条正常"、实例身份未知,记 aggregate disclosure,实例 outcome SEALED);Source(outcome 可开放)= realAWSCloudwatch 字典序前 8 + realKnownCause 字典序前 6,禁按标签/结果/headroom 换序列。

**#42 定稿 = sol 全文 + 主线六条补注**(见分发件):Part 0 幂等化(lite 已提交则退化为核验)、evaluate 后端现在钉死(gpt-5.6-sol @ 现役 endpoint,冻结协议一部分)、plan 工件记曝光标签(Source=INSTANCE_SEEN/EXPOSED,Target=INSTANCE_SEEN/SEALED + 聚合披露行;T6 全程在 V10 上)、确定性复核限 2 个 cell(保 plan AD fit ≤200:枚举 140 + 复核 ~14)、evidence_grade = NATURAL/provisional 单域双 cohort 不声称普适跨域、标准纪律块(不 spawn/歧义上报/交付不 commit 除 Part 0/NOAA 2025 与 beyond_17520 与 SMD 墙/另一线停笔/不重掷)。**#41 证据上限表(sol)与 C17 一致,维持;进度正式从"正控链是否接通"切回项目承重问题:自然数据上 Source 经验能否减少 Target 试错并降低 harm。**

### #41b-lite 收口、#42 v1 first-fault 与 #42a 契约修正(2026-08-23,主线裁定)

**#41b-lite = V10_READY_FOR_T6 @ 5dee103**:六件逐文件入库(method.py ccf2…/e1.py e550… 回读相符),t5 工件与 687af6e 逐字节一致,三份 untracked 测试未删未入,V10 四十成员 drift=[]。

**#42 v1 = NATURAL_DATA_SHAPE_INELIGIBLE(有效 first-fault,freshness 未烧)**:NAB v1.1(tag→commit 0dcd730)20 文件门表全出,4 条时间列非单调(Target exchange-2 cpc/cpm 各 1 重复戳;Source ec2_request_latency ×11、machine_temperature DST fold ×1),其余四项 20/20 过;roster 字典序诚实;**LabelWall 构造时即丢弃 Target 窗口**(非请求时拒绝,target_values_retained_in_memory=false,键存在性不泄露"哪条正常")为在册封存机制;执行方四路补救(弃 exchange-2/去重/放宽非递减/换域)全识别为越权,只报不选——满分处置;evaluate 协议未冻入失败面(正确);成本 0/0/0。

**根因改判(sol,主线共领)**:`TIMESTAMP_ORDER_GATE_MISALIGNED_WITH_INDEX_BASED_CONSUMER`——Consumer 行序开窗/行号切分/不用真实间隔,Program 只碰值序列,"严格递增"非本 estimand 执行前提(NAB 只承诺 ordered/timestamped/single-valued)。**A4 门为书面缺陷,主线通则入册:合法性门必须从 estimand 与仪器实际输入契约推导,不从"干净时序"审美拍。**

**#42a(已分发)= 合法契约修正,非放宽门槛**:修约仅依据形状门合法读取的结构信息,零 outcome 通道(AD fit 0/200,Part B/C 未执行);row-order 契约统一适用 20 文件(行序 = 物理行序,严格递增降为诊断统计,禁排序/去重/聚合/重采样/插值/删行,行数与值逐元素等同为验收);标签契约:真值窗口按 timestamp 语义逐行判归属,重复戳双行同窗同事件,未映射窗口计数必报;v1 工件保留为诊断,v2 于 Target 封存下一次冻结;--plan-v2 同 Runner 不建新 Runner;0 LLM / AD fit ≤200。

**七歧义裁定**:(1) 四路补救全否,由 #42a 第五路取代;(2)(3) 统一契约下 Source 侧同款缺陷消解(known_cause 保 6),"Target 失败停/Source 失败报"口径追认;(4) **C3 三类门只读 delayed_relation**(sol 裁定采纳),Support 单独记账观察 Support→delayed 翻转,不得凑票;(5) Query 前置 19 点取原始字节 = t1b 成规升为 T6 契约条款(Program 不得经 query 前缀后门进入推理输入);(6) background_alarm_rate 分母冻结 = 全部真值窗口外可评分点;(7) 窗口-采样映射:包含式匹配,v2 起未映射窗口必报。**#42a 验收线(预立)**:v2 工件须含契约修订 freshness 行(修约发生于 Target context INSTANCE_SEEN 之后、任何 outcome 打开之前)、20 文件行数/值等同读数、未映射窗口计数、C3 delayed 层三类判定、V10 零漂移。

### #42a 结果 = T6_NATURAL_PLAN_READY(C18);主线核验与 evaluate 释放(2026-08-23)

预立五项验收线全部由主线对 v2 工件独立读取坐实(非仅采信报告):曝光标签 Source INSTANCE_SEEN/EXPOSED、Target INSTANCE_SEEN/SEALED + aggregate_disclosure 行;all_ok/rows_preserved_everywhere = true 且契约叙述明写代价量化;v1 工件保留为"产生此契约的诊断";target_key_requests=[] / breached=false / 值未驻内存 / evaluate_released=false + release_rule 主线专属;V10 drift=[]。**主线于 2026-08-23 12:2x 置位 evaluate_released=true——本工件唯一主线触碰字段,置位即释放已冻结 --evaluate,协议其余任何字段零改动。**

**三歧义裁定**:(1) 重复戳双行同事件的代价已量化且 29/29 全映射 = 接受;(2) Episode 粒度 (cohort, round, program) 追认——cohort 为决策单元,逐序列 gain 向量与 harm 证据随卡保留,符合章程"cohort 层检查组合效用"口径;(3) **known_cause r1 delayed 近零分辨率**(四条非 identity |gain|≤0.0004,而 Support 层三条 CONFLICT)= 反馈分辨率的数据事实,入册为已知限制随卷进 evaluate:A5 若检索到这些卡,低信息量是自然属性非故障,不得据此调窗/换 roster/重掷。

**科学注记(承重)**:自然 Source bank 未经任何注入即呈现章程立项前提——同一 Program 跨 cohort 效用翻转(winsorize、hampel 两例齐全),意味着 A5 臂若只按 Program 名检索必然踩雷,必须靠可观察 Context 分辨;Support→delayed 八次翻转则是 T5 A4"delayed 才批准"接线的自然数据背书。

### evaluate 释放收回:执行体 stub(2026-08-23,sol 发现,主线复核属实)

runner:1285 `_evaluate_released()` 为占位体:脚手架真实(Consumer/LabelWall/bank/FitBudget/h0/逐臂全新 Method/Store/钉死后端工厂),但循环体只 append note 字符串 **"executed through run_online_round / open_delayed" 而一次未调**(:1334-1337),`_ = (run_online_round, …)` 显式丢弃 import(:1342),打印 EVALUATE_RELEASED_RUN_NOT_PERFORMED 退出码 0;另 `_evaluate_agent()` gateway 绑 `np.zeros(1)` 占位数据(:1363)。**分类 = 执行体缺失 instrument fault,非方法/数据失败;C18 与 SEALED 完好(墙对象未被使用,target_key_requests=[] 已核)。** stub 为 #42 禁跑期合规产物且自报 RUN_NOT_PERFORMED,但 note 字符串与退出码 0 是双重误导陷阱,#42b 必须清除。**主线登账**:12:2x 置位 → 12:4x 收回(evaluate_released 已翻回 false);释放前核了封存与协议冻结、未核可执行体存在——**释放清单永久新增:释放任何被冻结路径前,必须核验其可执行体真实存在且非占位**(报告语句"双入口都已实现"以后须以行级证据支撑)。#42b 极小补全书已发:仅补 `_evaluate_released()` 本体,0-LLM Source 侧机械烟测过后,主线二次核验封存、二次置位,唯一一次正式 --evaluate。

### #42b 后止损:单 cell fixture 为最后一道预运行检查(2026-08-23,sol 止损方案采纳)

#42b 执行体一次补 759 行,八 cell smoke 未覆盖 LOCAL_ACTIVE 正向分支,另暴露预算/读数/判定细节问题(正式报告待收,现状经 sol 代码复核转述)。**顺序教训入册**:正确顺序 = 单 cell 全链烟测 → 补执行体 → 冻结协议 → 开 Target;实际把冻结与放行做在验证之前——纵向最小切片原则对实验仪器同样适用,一次写 759 行本身就是违反。sol 与主线各认各的失察(均未第一时间打开代码验证"已实现")。**止损裁定(机械快车道首次使用,接力压到一圈)**:主线已再次置位 evaluate_released=true,并以指令把"单 cell fixture(AWS r2 + winsorize,≤20 fits,0 LLM)全绿"设为运行正式 --evaluate 的硬前置:绿 → 同 session 直接续跑正式 evaluate 不再回报等待;红 → 最多修一个明确机械断点一次,仍红 → 报 RUNNER_UNUSABLE 停;**不再新增 #42d、八 cell smoke 扩展、文档审计或新 Gate;Target 打开后绝不现场调参**。

### #42c = LIFECYCLE_FIXTURE_CLOSED;flag 事件裁定;正式 evaluate 放行(2026-08-23 13:4x)

**fixture 全绿(MECHANICAL_FIXTURE,不计方法证据)**:单 cell(source_aws r2 全 8 序列,identity+winsorize,脚本后端)走 #42b 补全体同一路径 `--evaluate-lifecycle-fixture → _run_cells → run_online_round → open_delayed → activate_approved`,全链真实:Support +0.02679 POSITIVE → Draft pending → delayed 真开真分类 +0.00780 POSITIVE → Episode LOCAL_ACTIVE → activate_approved=True(计数 1);Support/delayed 宏 F1 与 v2 bank 同 cell 逐字相符(identity 0.0357/0.1372,winsorize 0.0625/0.145)= 已曝光 Source 读数的机械复现;16/20 fits,0 LLM;support_trial_budget=1 仅 fixture 入口,冻结默认 2 未动。四处修正只读核验 4/4(共享 LLM 单账/每轮重绑 Gateway/delayed 不混层/十格机械落地,PENDING_ADJUDICATION 字样 0 次);旧 smoke 不重跑,B3 由本 fixture 闭合;旧 0-LLM 工件无法区分新旧 llm 记账口径 = 如实接受。

**flag 事件裁定**:13:28:12 的 evaluate_released false→true 为**主线蓄意置位**(上轮已同步写入指令/台账/roadmap,时间吻合),非误触;执行方见 flag 为 true 仍拒跑并上报 = 正确行为,**永久口径:release flag 是必要条件非充分条件,运行授权只来自指令前置条件**。二次核验封存由 #42c 读数坐实(fixture/smoke target_key_requests=[]、fixture 墙 released=False、t6_nab_evaluate_v2.json 不存在、V10 未触碰)。**正式 --evaluate 全部前置齐备,主线放行:冻结协议零改动,一次性运行,Target 打开后绝不现场调参。**

### 正式 evaluate 结果 = C19;竞态双样本裁定;第一阻塞指认(2026-08-23 15:0x)

**竞态裁定**:Windows 启动器多次空转被杀,其中一个"已杀"进程实际完成整跑并先落盘(影子件 28,758B:两臂 CPM r2 均激活 outlier_mad,判 NO_SOURCE_EXPERIENCE_ADVANTAGE,LLM 22/fit 12),随后 bash-10 整跑覆盖(现盘件 27,372B:判 SAFER_NOT_FASTER,LLM 23/fit 15)。"唯一一次"纪律被基础设施破坏,非执行方选样(后写者胜 + 双份如实上报);**承重结论取交集**(见 C19),盘面判定保留但单独不承重。两次合计 LLM 45/48、fit 27/120、墙均未破;Target outcome 已按计划曝光,AdExchange 自此不再 virgin(一次性用途已兑现)。**永久站规(一次性运行纪律)**:凡 one-shot 运行,启动前必须以唯一 run-id 后缀隔离 Store 与工件路径 + 启动锁;"已杀"必须验尸(进程确认终止)后才准重启;违者即使结果完好也记协议破坏。

**其余歧义**:(2) evaluate 工件缺受害序列实例名 = 仪器缺口,记入下轮仪器修正清单(harm 计数在,不阻塞);(3) 共享账 per-arm llm_calls 为快照不可按臂求和 = 口径已明,以单账为准;(4) known_cause 低信息卡 + A5 全面 abstain = 数据/检索事实追认,非故障。

**第一阻塞指认(主线读数,待 sol 对齐后开下一书)**:经验条件化的机械链全部工作(检索命中、渲染稳定、行为确实改变),但改变方向 = **全局保守**:A5 读到 4 张负例卡与近零信息卡后,四个 cell 提案池坍缩 identity-only,0 试验 0 反馈消耗——卡上下文(aws/known_cause 基础设施指标)与 Target 上下文(广告 CPC/CPM)的差异未被检索/渲染/提案任何一层折算。候选修面(一轮只动一面):(a) 检索第二段 context 匹配(渲染时按 context 距离过滤或标注),(b) bank 组成(负例占比与 r1 低分辨率卡),(c) ContrastPack 呈现(负例的 scope 语义)。证据倾向 (a):卡本身带 Context 字段,是使用层没有折算。确认性复考必须换第二自然域(Yahoo S5/UCR-AD,本域已曝光,同域重考 = 二次抽样);已曝光轨迹可用于修面后的 replay 机制验证。M0 暂缓,先破此阻塞。
**[同日 supersede,见下节]** 上述修面路由全部撤回:(a)(b)(c) 三条都是在修复已被仓库级路线锁否决的 Episode→Fast 直连,违反 "不得以检索/聚合方式修复 REJECTED 路线" 明文。

### 架构回退裁定:Episode→Fast 直连非法,#40 线证据整体改标(2026-08-23 15:2x,sol 发现,主线复核仓库级 override 属实,主线认领失职)

**事实**:仓库级 AGENTS.md L21-99 "Repository-specific architecture override" 明文(且声明压过 workspace 章程与历史计划):唯一主线知识流 = Episode → 确定性 Runtime 证据 → (a) 域内生命周期 → Target-local Skill 或 (b) first-fault → Slow 整合 → General/Specific Skill → Fast;**Fast 不得接收 raw/row-wise Source/Target Episode bank、source_experiences 字段、绕过 Skill formation 的 Episode 聚合**;A5 = Target Outcome 前冻结的 Source-derived Skill,**差异不得以直接注入 Source Episode 实现**;前例 T233 已留档 RAW_SOURCE_EPISODES_TO_FAST_REJECTED 且路线锁禁止"检索/聚合同批 Episode 供 Fast 直接消费"式修复。#40 ContrastPack → #42 A5 二十卡直连(runner:1577 → method.py:343 → fast_agent.py:758 渲染约 4 张事实卡入 prompt)= 重建被否决路线。**主线失职**:Phase T 全程未读仓库级 override,按 workspace 章程行事;立规:开书前必读仓库级 override,两章程冲突即停升用户。workspace AGENTS.md 由 sol 修正中(Source Experience → Runtime/Slow 整合 → Skill → Fast;禁 Episode/ContrastPack 直入 prompt)。

**证据改标(不删除,双层入账)**:C19 主线审定改为 **INCONCLUSIVE_CONCURRENT_EXECUTION_COLLISION + DIRECT_EXPERIENCE_CARD_TO_FAST 受控旁路**,原始工件判定字段不改;前轮"两样本交集承重"降级为 development 诊断;NAB AdExchange Target 降为 development evidence,永久非 virgin,不再正式运行;review_independence = same-context,正式接受状态 = blocked。C14/C15/C16(T4 卡线)改标为**旁路机制证据**(卡条件化的 POSITIVE_CONTROL 机制研究成立,不构成主线 A5 能力证据)。**边界读法(主线裁定,请 sol 写入 AGENTS.md)**:Fast 可读"当前适应轨迹内已发生的 Target Support 历史"(override L44 明许)——故 T5 域内轨迹检索与 #42 中 CPC A3 r2 读own r1 负例合法,C17/CPM A3 r2 自然闭合完全无损;非法的是**跨轨迹/跨域 Episode bank** 直入 Fast。C18(Source bank 本身)完全成立——bank 是整合的合法原料,非法的只是直连用法。**"劝退系统"发现改写为本裁定的实证支撑**:#42 意外让被否决路线在自然数据上考了一次,复现了路线锁预言的失败形态(未整合的负例 → 无差别保守)——旁路负结果有价值,记 RAW/DIRECT_SOURCE_EXPERIENCE_TO_FAST 自然域第二例。

**校正后排期(与仓库级 override L86-95 既定下一切片直接对齐)**:(1) 关闭 Source-episode→Fast 直连接线(边界如上);(2) **新方法切片 = Source 证据 census → Slow 整合 → 冻结 Source-derived Skill**(NAB 20 卡 bank 为原料:aws winsorize 双轮 POSITIVE → 带 scope 的 soft-prior 能力 Skill 候选;known_cause 负例 → context 条件化 Risk Skill 候选;r1 低信息 → ABSTAIN;冻结不得接触任何新 Target outcome);(3) 已曝光 NAB 轨迹上 development replay 机制验证(Skill 版 A5' 池是否不再无差别坍缩);(4) O9 第二未曝光域(Yahoo S5/UCR-AD)census → **正式 A5vsA3 v2**(A5 = h0 + 冻结 Source-derived Skill,单进程站规);(5) M0 顺延其后,#46 不变。

### #42d 分发前修订 r1(2026-08-23 15:4x,本地 agent 评审,主线核验采纳;分组粒度错误主线认领)

**三事实核验**:(1) `task_episode_harness/agentic/source_skill.py` 实存,且正是 T233 判例后建的正确架构整合器(六段 WHEN/OBSERVE/TRY/RISK/VERIFY/FALLBACK、UNGUIDED/conditioned 证据溯源、留一授权审计、确定性包含审计[禁造算子/特征/数值阈/cohort 泄露]、skill-entry/1 capability 载荷不携冻结程序、ABSTAIN 合法)——#42d v1 Part C 属重复造轮,主线开书前未盘点既有仪器,记失察;(2) 分组单例坐实:v2 工件 /source_bank/rows 按 (cohort,round,program) 实测 **20 组全部单例**(该键即 episode 粒度),v1"组内 ≥2 delayed POSITIVE"永不可满足,CONSOLIDATION_NO_ELIGIBLE_SKILL 在 v1 书面下为必然而非可能;(3) 历史 A3 非因果基线按 C19 本就成立,replay 改同跑配对。**bank 实况预读(承重)**:全池无条件 cell 下无任何程序可获 TRY 授权(winsorize 2 正 [aws r1/r2] 被 known_cause r2 负所阻;hampel/iqr/mad 各携负例或 CONFLICT)——授权只可能出现在能分辨 aws/known_cause 的可观察条件 cell 内,整合器被迫直面章程考点"靠 Context 分辨跨 cohort 翻转";两正例同 cohort 异轮窗为已知限制,census 必记、VERIFY 必载。

**修订内容(r1,其余条款逐字不变)**:Part B 证据单元 = episode_id,cell = program × 可观察条件(限卡上既有布尔 pattern 特征,series_count 类 cohort 代理排除,禁新造阈值);delayed_relation 承重(CONFLICT→反证阻 TRY 可计 RISK,identity ABSTAIN 不计票);全体 UNGUIDED 须经运行配置断言。Part C 复用 source_skill.py 经 AD 薄封装(skill_id=source_investigation_ad_v1、applicability task_kind==anomaly_detection、NAB 禁词表;**本体零改动**——预测线归档仪器,T233 工件依赖其原样行为);min_distinct_tasks=1 预注册(TRY = 同 cell ≥2 不同 episode 正证据过留一 + 零反证;RISK = ≥1 负 + 零正);单次 Slow ADD-or-ABSTAIN,≤1 条目(v1 ≤6 随旧分组设计作废),LLM ≤8。Part D 同跑配对:单进程 8 cell(双 cohort × 双轮 × A3/A5'),LLM ≤32 / fit ≤120,one-shot 站规全套;#42 历史读数降描述性附录、判定禁引,#42-A5 坍缩(0 非恒等尝试)仅作定性参照。总账 LLM ≤40 / fit ≤120 / 重训 0。**成熟度定性采纳本地 agent 表述**:#42d = "复用预测线 Skill 生命周期,补齐 AD 原生 Evidence→Context Scope→Skill 适配";"AD 侧仪器齐"仅指 Consumer/评分/契约/接线。**与评审的一处分歧**:"复用"以 import 薄封装实现,不按字面在 source_skill.py 本体改常量。

### #42d r2:sol 保真审计采纳,Part 0b 保真收口前置(2026-08-23 16:0x,主线逐条核验)

**sol 三类指认全部坐实(主线独立读码)**:(1) 两个承载 forecasting 正结果的 Runner 手写旧 ID——run_e2_local_skill_recall.py:411 与 run_e2_fresh_confirmation.py:1866 均为 `"fast_winner_%s" % workflow_signature`,方法层现写任务化新 ID(公有出处 method.py:1494 `fast_winner_skill_id`,其 docstring 本就记载"各处私拷贝已静默失联"),重跑必在 ID 断言处 ValueError(recall:285 / fresh:2027)——**响亮失败非静默错数,风险性质 = 暂不可复现,非证据污染**;T5 收尾只修了报错五测,未清点未运行的证据 Runner,主线缺口认领。(2) T6 直连现行号 :1619(`memory = bank_episodes if arm == "A5" else ()`),#42d Part A 删除对象确认。(3) 两道门收紧属实且**均为 #41 T5 A4 在册授权**(代码注释自引授权出处:online_loop.py:421 "Support = POSITIVE 才形成 winner/Draft"、method.py:1458 "批准条件改 classify_relation == POSITIVE……本轮授权的行为变化")——定性 = 在册收紧非静默漂移,不回退;sol 核对六个历史 forecasting Local Skill 读数(support 0.0118–0.1071 / delayed 0.0238–0.2058)全部过新门且无逐序列 harm 上传,旧正例在新门下判定不变(待 Part 0b 机械复现坐实)。

**裁定**:采纳 sol 最小收口,并入 #42d 为 **Part 0b 保真收口硬门**(置 Part A 前;0 LLM / 0 fit / 0 重训):修两处手写 ID → 公有 `fast_winner_skill_id`;以两 Runner 工件缓存读数 0-LLM 重放全部历史正例过现役双门,断言 Draft→ACTIVE、新 ID 检索命中、有缓存快照处 Skill 内容与提案入口渲染稳定、两 Runner 过 ID 断言点不再 ValueError;全绿 = FORECASTING_COMPAT_RESTORED 继续 Part A,任一红 = FORECASTING_COMPAT_BROKEN 停报不进 A–E。不重跑全量 fresh confirmation(#45 X 专属)。**对齐说明**:sol "不新增分组规则/阈值/Memory 通道/第二套 Skill" 四约束与 r1 相容——r1 的 cell 粒度(program × 可观察条件)与 min_distinct_tasks 皆 source_skill.py 自有审计粒度/参数的预注册,新增件只有 AD evidence/context 适配。**source_skill.py 定位澄清(sol,入册)**:旧架构资产而非 forecasting 正结果主源(其初版 Skill 曾致负迁移,后加 provenance 门并不授权 TRY);forecasting 正证据主链 = recipe evidence → Skill bridge → Store/检索 → Target-local 生命周期;#42d = 复用其审计与六段结构 + 保留已证 Store/Fast 使用方式/生命周期。**新站规(主线操作纪律第 5 条)**:接口/ID/键统一类改动的兼容清单必须包含所有承载在册证据的 Runner,不得只修当轮报错项。

### #42d r3:证据门回归已证水准 + 主线 claim 勘误(2026-08-23 16:2x,sol 三修全采纳,主线核验)

**主线勘误(承重)**:主线上轮口头表述"Source 迁移优势从未在预测线上证明"**错误**,与在册 `fresh_confirmation_v1_adjudication.md`(2026-08-21)矛盾——overall_verdict = **FRESH_A5_DELIVERS(维持)**,pooled Consumer 首正成本 69 vs 123(−43.9%),held-out utility/harm 与冷启动持平,per-channel = A5_TIE_TRANSFER_BOUNDARY(迁移边界)。canonical claim 以该附录 §对外 Claim 上限为准:预测线已部分证明 Source-derived Skill 降低适配成本(一次 fresh temporal/cohort outcome,非全新 Domain,AGGREGATE_SEEN);未证 = 独立数据域、AD 任务、其他 Consumer。**#42f 定位更正 = 已证优势向第二任务 + 独立域的推广考试,非首次证明。**正结果主链确认(sol,采纳)= 确定性 recipe compiler(R1/R2 均 ≥2 distinct cohorts,0 LLM)→ Skill 卡 → 真实 Store/检索/生命周期;source_skill.py 为 T233 线 Context 条件化整合器,无正能力结果在册,复用限六段载体 + 授权/包含审计 + ABSTAIN。

**三修核验与采纳**:(1) min_distinct_tasks=1 = 暗降门,撤回——在册门 g1.py:102 GENERAL_EVIDENCE_MIN_DISTINCT_TASKS=2、risk_skill.py:49 RISK_MIN_DISTINCT_TASKS=2、recipe R1/R2 ≥2 cohorts;r3 计票单位改 distinct Source cohort,TRY/RISK 均 ≥2。(2) pss 实测完美复刻 cohort 身份(aws 十条全 False / known_cause 十条全 True),且 bank 布尔特征仅 3 个(level_only_pss 恒 True、period_repair_available 恒 False)——**本 bank 合法可观察条件化结构性不可用**,census 新增 cohort 代理检查普遍义务,完美一致者禁作 Scope;r1"授权只可能出现在条件 cell"预读作废,"靠 Context 分辨"考点移交扩 Source/第二域轮次;pss union 语义缺陷另在跨域理论文档 §M0 在案。(3) source_skill.py 模板 import 时烧入 skill_id、build_skill_payload 直读模块常量,r1 wrapper-常量方案不可行收回;改最小参数化(可选参数、旧默认逐字不变)+ 默认路径字节等同断言测试;r2"本体零改动"由此 supersede。

**预注册推演(跑前入册)**:TRY 无候选(winsorize 正证据仅 aws 单 cohort);RISK 唯一候选 hampel_filter(aws r1 NEGATIVE + kc r2 CONFLICT = 2 cohort 伤害,全 pool 零正)——成立取决于 CONFLICT 计伤害(主线预注册:计,理由 = CONFLICT 定义即含逐序列伤害,合冻结不对称原则"somewhere 重复伤害即可降权";census 严格/扩展双口径并报;**r3 唯一新解释点,交 sol 终审可改严**)。两分支均合法:risk-only Skill → Part D 主读数 = 有边界劝退 vs 全局劝退(hampel 提案率降、其余算子探索保持,坍缩 = UNCHANGED_COLLAPSE);全 ABSTAIN → 跳 Part D 判 CONSOLIDATION_NO_ELIGIBLE_SKILL,路由扩 Source(realTraffic/realTweets 作第 3/4 Source cohort,Source 侧用途合法)。预算不变(LLM ≤40 / fit ≤120 / 重训 0)。

### #42d 结果 = 架构纠偏成功,risk-only Skill 未获触发(C20);sol 审核采纳;收口书发出(2026-08-23 17:3x)

**四段判定照收(预注册全兑现,主线读 skill 工件独立核验)**:0b **FORECASTING_COMPAT_RESTORED 6/6**(两手写 ID 改调公有 fast_winner_skill_id,6 条缓存正例 Draft→approved POSITIVE→ACTIVE、新任务化 ID 可检索;残留拼接三处入册[T5 前缀检查/method 构造器本体/docstring],均非查询点);A 直连已拆,双臂构造期 Memory 断言为空,A5' 只经快照带 Skill;B census TRY=[] / RISK=[hampel_filter] 与预注册逐字一致(matches_preregistration=true,双口径伤害账 extended 承重/strict 并记),pss 完美代理坐实禁 Scope;C **SOURCE_SKILL_WRITTEN**(risk-only,LLM 1/8;source_skill.py 最小参数化默认路径 3/3 字节等同;known_limits 记 winsorize 单 cohort);D **SCOPE_CORRECT_NO_APPLICABLE**(8 cell 同跑,run-id 20260823T090444Z 隔离 + 启动锁;hampel 双臂 0 提案,非 identity 试验各 1,无 identity-only 坍缩;墙未破;target_outcome_read=false 于整合段)。成本 LLM 25/40 / fit 12/120 / 重训 0。首次启动死于 Windows 无 pgrep(未读 Target),验尸确认后单次重启——one-shot 站规首次实战通过。执行方拒绝把 mad/iqr 差异贴成更快/更安全 = 满分处置。

**C20(canonical,sol 表述采纳)**:风险 Skill 已合法送达;本次未出现 hampel 候选,降权效果未获验证机会;未观察到全局坍缩。不得声称 A5 更快/更安全;A5' CPM r2 outlier_mad LOCAL_ACTIVE(+0.0111)与 A3 CPM r2 outlier_iqr delayed NEUTRAL 拒批(−0.0028,害及 exchange-4_cpm)为单轨迹观测差异,不归因 Skill(该 Skill 未推荐 mad)。evidence_grade:形成 = NATURAL/provisional,重放 = development。观察注记(非承重):CPM r2 × outlier_mad 二度自然闭合(C19 竞态样本 A3 臂、本轮 A5' 臂)。

**v1 Skill 文本时序惰性缺陷(sol 发现,主线读工件坐实,承重)**:OBSERVE 令读"available programs 的 delayed_relation"(提案时点不存在);RISK 把降权挂在"repeated observations show delayed NEGATIVE/CONFLICT"(未来 Target 观察)上——**t=0 永不点火,与 T233"送达但惰性"同族失败**;VERIFY 混同 Support/delayed 两阶段并要求 distinct tasks(Source 整合证据单位,非 Target-local 要求)。仪器缺口定性:包含审计查词汇来源不查时序合法性。**修法(sol 方案采纳)**:预测侧 source_skill.py 零再改;AD 薄适配层新增确定性时序审计(WHEN/OBSERVE 禁阶段结果词汇;RISK 须为常备降权知识、明示可被当前 Task 证据推翻;VERIFY 两阶段有序、禁 distinct-task 要求)+ Slow 提示补阶段语义,重发一次 → source_investigation_ad_v2,冻结 h0s_v2;v1 在册标 superseded 不删不入 h0s_v2;**Part D 不重跑**(hampel 双臂 0 提案,文本修正不改本轮行为读数,判定维持)。

**其余裁定**:两个新 run_e2_* 入口(809/318 行)违"不建新 Runner"纪律,既产真实工件**化石处理**——不删、禁扩建、入 #46 名册债;fresh Target 禁用当前 h0s(sol rec 3,惰性 Skill 不得消耗处女域);工作树 771 行 porcelain 积压(#42b/c/evaluate/#42d 全部未提交)由收口书 Part 0 一次入库。**排期更新(编号后移)**:收口书(提交 + Skill v2)→ **#42e = Source 扩充**(realTraffic/realTweets 第 3/4 Source cohort;winsorize 第二正 cohort 为 TRY 解锁唯一路径)→ **#42f = 新域获取 census**(Yahoo S5 优先)→ **#42g = 正式 A5vsA3 v2**。

**收口书 sol 四修采纳(同日 supersede)**:(1) Part 0 由"积压全量"改**精确 allowlist** + 显式排除(NAB raw 永不入库、临时 Store/lock/cache、另一线三份 untracked 测试、无关改动)+ 两份 stdout 与待提交文本**密钥扫描前置**(Authorization/api key 模式,命中脱敏或排除);(2) **主线"Part D 不重跑因文本修正不改行为"推理撤回**——v2 恰是把惰性提示改为常备降权,可能改变候选池与弃权行为;v1 判定只属 v1、不为 v2 背书;Part S 判定改 **SKILL_V2_FROZEN_PENDING_BEHAVIOR_REPLAY**,并立前置:**任何 h0s_vN 进正式考场前必须先过一次 development 同跑行为重放**(挂 #42e/#42g,不在收口书内跑);(3) 时序审计规则机械化:OBSERVE/WHEN 禁 support_relation/delayed_relation/approval/Skill 状态词汇、只准提案时部署可见 public Context;RISK = census 默认降权但**明示强公共 Pattern 下 hampel 仍可作受限探测候选,不得硬禁**;VERIFY 精确两阶段(当前 Target Support relation=POSITIVE 才成 Draft;随后 delayed relation=POSITIVE 才批准/保活);全文禁 distinct-task;(4) 冻结交付清单明确:t6_nab_42d_source_skill_v2.json/.md、v2 entry 或完整 h0s_v2 快照、**0-LLM 静态送达断言**(A5 Fast 视图可检索 source_investigation_ad_v2、A3 不可、双臂 Memory 构造期空——只证合法送达,不证行为收益)。**本轮性质入册 = 机制/适配修复,不构成新 Capability 正证据。**判定空间改 CHECKPOINT_FAILED;PART0_COMMITTED × {SKILL_V2_FROZEN_PENDING_BEHAVIOR_REPLAY | SLOW_CONSOLIDATION_UNREADABLE}。

### 收口书结果;#42e0(sol 直发)= C21 无触发候选;#42e 书发出(2026-08-23 18:1x,主线追认入典)

**收口书兑现**:Part S = SKILL_V2_FROZEN_PENDING_BEHAVIOR_REPLAY(v2 合法生成送达,无行为证据;claim 上限 = 不得说改变提案/减少 harm/加速适配;same-context/provisional)。sol 两缺口裁定:(1) t6_nab_frozen_plan_v2.json/.md(~267KB,T6 Runner 与整合的直接输入;raw 不入库故 plan 必须入库才自足;墙记录 target_key_requests=[]/breached=false 可安全提交)→ 已由 #42e0 Part 0 补入;(2) **h0s 临时快照永不入库**——后续一律从 t6_nab_42d_source_skill_v2.json["entry"] 确定性重建加入 h0,**禁重调 Slow 再生成**。工程裁定:ad_source_skill.py 本轮膨胀至 ~420 行(混入 Slow 调用/Store 物化/工件写盘),已非"薄适配";不返工,**issue_v2() 冻结为一次性形成仪器,禁扩建**。

**#42e0(sol 直拟直发短书,主线本节追认其判定阶梯与读数入正典;正典文档仍由主线单笔)= C21 RISK_SKILL_NO_TRIGGERING_CANDIDATE @ ad4f7b82**:五件逐文件入库(plan_v2 原字节、evaluate_released 保持 true 不改);送达断言 4/4-0/4、双臂 Source Memory held=0;--replay-skill-v2 于现役 T6 Runner,同 #42d 8 cell 同协议单进程一次;hampel 双臂提案/选中/探测 0/0/0;A5 CPM r2 池 [identity, mad] 探测 mad +0.0593 → LOCAL_ACTIVE,delayed +0.0111;A3 同 cell 池 [identity, mad, iqr] 同 mad 同读数——**池宽差异非 hampel 专属效应,如实不归因**;LLM 22/32、fit 12/24、Slow 0、墙未破。**claim 阶梯(sol,canonical)**:送达且无 Episode 直连 = 支持(4/4-0/4);无全局探索坍缩 = 本 development 轨迹内部分支持;v2 降低 hampel 风险 = 证据不可用(双臂 0);A5 更快/更安全 = 不支持(两臂同一次 mad 探测同 delayed)。**站规**:禁在 NAB 上重复抽样钓 hampel 触发(碰运气 = 调样本);first-fault 定位 = 问题不在送达/Memory/生命周期/Fast 执行,在 Source 证据只形成了无触发场合的风险提示。

**#42e 排期裁定(sol 方案采纳)**:Source 扩充先于 fresh Target——realTraffic/realTweets 作第 3/4 Source cohort(仅 Source,outcome 本轮打开,永久不得再称 fresh Target);Consumer/五程序菜单/窗口/双层计分零改动;冻结门重整合(TRY ≥2 独立 cohort POSITIVE 且合法 Scope 零反证;RISK ≥2 cohort 伤害零正;dataset 名与完美 cohort 代理禁 Scope;4-cohort 下代理判据升级为"单 cohort 指示器或完整 cohort 划分复刻");**停止线** = 4 cohort 后仍无可行动 TRY 且无可触发风险知识 → SOURCE_EVIDENCE_INSUFFICIENT_FOR_ACTIONABLE_TRANSFER(本 Source family 停)[判词命名同日 supersede,见下节 r1];有 h0s_v3 → #42e1 development 行为重放 → 才消耗 Yahoo S5 等 fresh Target。

### 主张架构 v2 采纳;#42g 两层重塑;#42e 重定位 r1(2026-08-23 20:5x,用户提案 + sol 评审,主线裁定)

**背景(用户勘定,承重)**:用户明确项目第一诉求 = 数据质量在不同 Task/Consumer/Domain 上的**适应性**——Harness 在新域 held-in 上适应后,在同域 held-out 上取得更好终态;"A5 比 A3 更快找到首正解"不是错,但只是"Source 经验复用是否有价值"的子命题,不得再作总主张。sol 评审整体采纳该重构,附两处硬修正(#42e 定位、预算反挑选),均采纳。

**主张架构 v2(canonical,取代单条 A5vsA3 式总主张;全文入 roadmap §3.5)**:总主张 = 面向多 Task/Consumer/Domain 的时序数据质量适应 Harness,依据部署时可观察局部 Pattern、Program 作用几何与 Task/Consumer 质量语义,在新域 held-in 上自主生成/验证/更新 Target-local Skill,在同域 held-out 上验收效用与风险;Source 与 Target 共享可观察决策 Context 时,冻结 Source-derived Skill 进一步减少适应成本与负迁移。四层:**L1 核心适应性**(A3 vs 静态默认,无条件承重)/ **L2 Source 先验价值**(A5 vs A3 同预算同 held-out;FRESH_A5_DELIVERS −43.9% 为其预测线首个有界正例)/ **L3 Pattern 机制**(完整 Context vs 去 Pattern/错配,收益须来自 Pattern–Program–Consumer 匹配而非 dataset 记忆)/ **L4 载体条件化**(同一处理随 Task/Consumer 反号且 Harness 随之改行为,M0/M1;T1b 为注入正控先例)。旧 A4(少/零 probe Shared Capability)= 远期可选,非成立前提。L1/L4 不依赖 Source 迁移成败;L2/L3 为更强迁移贡献。

**#42g 两层重塑(采纳)**:L1 层无条件——新 Target 域按序列切 held-in/held-out,A3 从空白在 held-in 固定预算适应,冻结终态,held-out 上对照静态默认(identity + 一条固定通用清洗,发书时预注册);主读数 = held-out 终态 utility / harmed series / worst-series gain / abstention / held-in→held-out 方向一致率。L2 层条件开(存在可行动 Source Skill)——A5 = 冻结 Source Skill + 同 held-in 预算,同一 held-out 验收。**承重读数纪律(新立)**:能力对比以冻结终态 held-out 效用与安全承重;首正成本/试验数/feedback 消耗降为解释性辅读数,不得单独支撑能力主张。**反挑选钉(sol)**:held-in 预算由已曝光 development 轨迹预先冻结,禁挑"刚好让 A5 赢"的紧预算;保留适应过程曲线;宽预算下终态打平而 A5 更快仍记有效加速。处女 held-out 只被冻结态触碰;L3 机制消融只跑已曝光域。

**#42e 重定位 + 主线勘误(承重)**:主线上轮称 #42e 为 Pattern insight"生死考"、预设失败归因"自然语料证据密度不足"——过头,正式收回(chat 层表述,文档未污染;与 r3 勘误同族)。#42e = **当前 Source 表示下的 actionability census**:现有 3 布尔特征 + 五程序菜单 + 冻结 ≥2-cohort 门的组合能否形成可行动 Skill;负判不得表述为"共同 Pattern 不存在"。停止判词改名 **NO_ACTIONABLE_SOURCE_SKILL**(原名把归因烧进判词,废),判此词必须附 first-fault 区分:OBSERVATION_INSUFFICIENT(跨 cohort 反号存在但合法特征两侧无差)/ PROGRAM_NO_REPEATED_HEADROOM(响应表稀疏,无程序在 ≥2 cohort 重复同向)/ SOURCE_FAMILY_MISMATCH;允许 UNDETERMINED 但须列区分所需最小后续观测。任何判定下不再为本 family 下载新数据。r1 增补页已发(定位/判词/家族封顶三条,其余条款逐字不变)。

**Pattern insight 重定义(采纳,canonical)**:可迁移单元 = 部署时可观察局部 Pattern × Program 作用几何 × Task/Consumer 质量语义 → Action–Response;检索单位禁 Dataset 名与"有 outlier"级粗标签;Observation 升级按四块 Context 框架(Data Pattern / Program Geometry / Task-Consumer Semantics / Response Evidence)。pss 失败只证明粗特征不够,不否定 insight。

**新条目 #42e2(winsorize 反号判别,规划)**:已曝光 NAB 上,对 winsorize AWS 正 / KC 负反号检验是否存在部署时可见、与 winsorize 作用几何直接相关的局部 Pattern 差异(isolated spike vs 持续 burst、修改点比例/集中度、是否覆盖事件本体、周期相位/持续长度、cohort 内覆盖结构);一次只加一个最小合法 Observation;仍不可分 → 诚实关闭该 Pattern/Program family。**时序钉(主线补充)**:严格排在 #42e 之后——2-cohort 库上任何完美分离特征形式上即 cohort 代理(pss 判例必然复发),判别特征合法性按 census v3 代理判据审(≥2 cohorts per side 才可入 Skill Scope,不足只记机制线索);0 新数据下载,0 fresh 消耗。

**方法重点转向(用户/sol 共识入册)**:多任务基础设施(T1b 反号、T3 TaskSpec 条件化、T5 单入口生命周期、任务化键/ID、双层风险门、AD Consumer/契约/墙)已齐;所欠 = 当前 Observation 能否表达可跨数据复用的 Action-relevant Pattern——后续方法改动优先级从"继续扩 Memory/证据门"转向最小 Pattern/Context Observation 改进。**债务新增**:相关工作检索(载体条件化 + Pattern 机制在既有 harness 进化文献中的存在性),正式声称"独有贡献"前必须完成,挂 #46 前任意窗口。

### #42e 结果 = C22(SOURCE_RISK_ONLY_TRIGGERABLE);sol 四修采纳;#42e1 书发出(2026-08-23 21:1x,主线读工件核验)

**执行交叉与 r1 处置(主线认领)**:r1 增补页签发时 #42e 已跑完(census/skill/expansion v3 工件在盘,Part 0 前置提交 c43810b 在册)——签发与执行交叉,主线未先核跑态。依 sol 裁定:r1 降为**结果定位与后续解释规则的事后修订**,非预注册增补;不重跑 #42e。实际判定 SOURCE_RISK_ONLY_TRIGGERABLE 属原判定集不变项;停止判词改名(NO_ACTIONABLE_SOURCE_SKILL)未被触发,且 family 已封顶不再有适用场合,随卷封存;定位条款(actionability census,负判不得表述为"共同 Pattern 不存在")与家族封顶条款照常生效。

**C22 canonical(主线读 census_v3.md / source_skill_v3.md 独立核验,非仅采信转述)**:4 Source cohort(aws / known_cause / real_traffic / real_tweets),20 旧 + 20 新 = 40 卡,17 个新文件形状门全过(traffic 7 + tweets 10)。TRY = [];RISK = [hampel_filter, outlier_iqr, outlier_mad]——严格伤害 cohort 数 3/3/3(hampel 另有 kc CONFLICT,扩展口径 4),三者全池零正例;winsorize = aws 正(全库唯一正例)+ kc/traffic/tweets 三 cohort 伤害 → authorization null(混合证据,既非 TRY 亦非 RISK)。合法 Pattern Scope = 0:pss 从完美划分复刻降为 single-cohort indicator(kc True / 其余三 False)仍禁;level_only、period_repair 四 cohort 恒常无分辨力。Skill v3 = source_investigation_ad_v3(h0s_v3 f2054da1d1…bd7914):TRY 段 = NO_AUTHORIZED_ACTIVE_RECOMMENDATION;RISK 段 = 默认降权三算子、强公共 Pattern 下仍可受限探测;VERIFY 两阶段正确;containment + temporal 双审计过;v1/v2 superseded 不删不入 h0s_v3;entry 入 JSON 供确定性重建、临时 Store 未提交(收口规则兑现);Slow 1/2、census LLM 1/8、AD fit 170/240。**与 v2 的承重差异**:v2 只降权 hampel(历史 0 提案,永不可触发);v3 纳入 mad/iqr(既有 development 轨迹提案 4/2 次)→ 风险条款首次可触发,#42e1 因此才有信息量。

**claim 表(sol 表述采纳 canonical)**:四 cohort 形成可执行正向 TRY 先验 = 不支持;形成跨 cohort 风险降权知识 = 当前 NAB Source/IForest/五菜单范围内支持;风险 Skill 改善/保护 Target 适应 = 不可判,须 #42e1;当前 Pattern 支持条件化迁移 = 不支持(合法 Scope=0);"自然数据不存在共同 Pattern" = 不能推出;"Source 跨域迁移已成立" = 不能推出。最强合规表述:"四个 NAB 自然 Source cohort 上,四种修复程序没有形成重复的正向 TRY 证据;hampel/iqr/mad 形成跨 cohort 无条件风险先验,其中 iqr/mad 在既有 development 轨迹中有可触发候选,其 Target 行为效用尚未验证。"review_independence = same-context,provisional。**研究含义定性(sol,主线共签)**:有价值的诊断/负向结果,非 Source transfer 成功;数据量不是缺口,再扩 cohort 无新正向 headroom,"当前自然 AD Source 证据支持广泛谨慎,不支持任何主动修复建议;能否把谨慎变成安全而不妨碍 Target 自适应的行为 = #42e1 唯一问题。"

**v3 三风险入册(即 #42e1 读数面)**:(1) 非 Pattern-conditioned——有效 Scope 仅 task_kind==anomaly_detection,属任务级全局降权,不能承担共同 Pattern 贡献;(2) 可能压制有益探索——mad 曾在 CPM r2 delayed +0.0111 成唯一自然 LOCAL_ACTIVE;闭环风险:降权 → 不探测 → Target Support 永无机会推翻 → 先验不可被否("强 Pattern 下可受限探测"当前无可执行定义,合法 Scope 为空);(3) winsorize 相对抬高——3/4 cohort 伤害却因混合证据免降权,identity 仍在,候选质量可能只是位移,须专项读数。

**sol 四修全采纳**:(i) #42e 收口入库、不重跑,r1 事后化(见上);(ii) 排序更正 = **#42e1 先于 #42e2**;#42e2 为增强线 B 独立实验,不得为救 v3 调 Observation,不阻塞 L1;(iii) 主线"最短故事 = #42g L1 + M0"表述**收回**——M0 只证反号现象存在(正控),载体条件化能力主张须 M1(Harness 读取 Consumer Context 并安全改变适配行为);最小可辩护闭环 = #42g L1 + M0 + M1(chat 层勘误,文档未染);(iv) **#45 重定位**——NOAA 2025 outcome 已于 FRESH_A5_DELIVERS(2026-08-21)一次性打开,#45 不得再称 fresh confirmation,改为既有 forecasting 能力复现/回归验证(两台 ID 修复后的证据 Runner 在此服役);新 fresh 证据需另一 outcome=SEALED forecasting 域,获取与否 = 用户决策点。

**结构追认与"概念回退"正典口径**:本地 agent"一条主线、两条增强线" = 主张架构 v2 的同义映射(主线 = L1[+L4 最终系统能力],增强线 A = L2,增强线 B = L3),无冲突。#42g L1 **不是重新验证方法**:它是冻结的预测线设计(Episode→Runtime 证据→Skill→生命周期、双门、A5 冻结先验语义全部不动)经已建多任务基础设施移植到 AD 后的一次**集成端口验收**——变的只有 Consumer adapter 与数据;失败按 first-fault 只修一个 AD 适配面(Program headroom / Observation / 检索提案 / Risk Scope),不改生命周期、不造第二套 Skill 系统、不加新 Gate。**距离口径更正(sol 采纳,supersede 主线"一层半/按书数计")**:按能力门计——底座(Runtime/Memory/生命周期/多任务接线)成、forecasting 纵向切片成(适应 + Source 加速各有有界证据);未完成 = AD 纵向切片(#42g L1)、载体条件化切片(M0+M1)、AD Source 加速(#42e1 待验)、合法共同 Pattern(#42e2 起步);决定项目主体的两场 = **#42g L1 与 M1**,#42e1/#42e2/#42g L2 为增强,不得拖住主线两场。

**#42e1 书发出(见分发件)**:development 一次性同跑行为验收;v3 文本冻结(验收前禁改);判定五格 + 边界退化格(NO_TRIGGERING_OCCASION 如实报,禁钓鱼);唯一开关 = #42g L2 是否加 A5 臂,任何判定不构成 Capability 正证据;Part 0 先入库 #42e 五工件 + Runner 修改 + 两 docs(allowlist + 密钥扫描)。

### held-in/held-out 协议语义钉:用户定义立为正典(2026-08-23 21:3x,sol 形式化,主线采纳并补三钉)

**歧义与裁定**:主线此前流程表述把"delayed feedback 批准/撤权"与"held-out 验收"写得相邻,可误读为 held-out 结果仍可回写 Skill——chat 层歧义,主线认领;用户定义立为正典。**held-in = 反馈可用区**:下游 Consumer 反馈(Support 即时 + delayed 后续)全部在其内,Episode 写入、Slow 修正、Target-local Skill 形成/批准/撤权、固定反馈预算全部只发生于此(delayed 的作用是防"同批反馈上自提自批",它不是 held-out)。**held-out = 零反馈部署区**:冻结终态 H* 只走 Fast——检索冻结 Skill + 读部署时可观察 Context 自主出 Workflow,运行期不知处理好坏。

**held-out 禁行清单(预注册为 #42g 协议违规判词素材)**:open_delayed;下游反馈回传 Agent;Slow;新增/修改/批准/撤销 Skill;按 held-out outcome 写 Experience;看结果后重试/调参/换 Workflow。可用:部署时可观察 Context、冻结 Source-derived Skill、held-in 形成的 Target-local Skill、不读 Outcome 的确定性合法性检查。**计分协议**:全部臂(Static/A3/A5)held-out 输出冻结后,外部 evaluator 一次性打开 outcome,只算效用/harm/abstention、只写报告、不回流本次 Harness;打开即 EXPOSED,不得重跑改法(与 one-shot 站规同族)。**三臂语义**:Static = 无 held-in 适应冻默认态;A3 = 空 Source + held-in 固定预算适应后冻 A3*;A5 = 冻结 Source Skill + 与 A3 相同 held-in 预算后冻 A5*;三者同一 held-out。首正成本/试验数/反馈消耗 = held-in 读数;终态效用/harm = held-out 外部评价读数。

**存量证据重标注(零改判)**:T5/#42d/#42e0/#42e1(在途)全程在已曝光 development 数据上,其 delayed 进生命周期属 held-in 语义内,继续作 development/机制证据,从未主张 held-out 泛化。**FRESH_A5_DELIVERS 协议类型注记(主线补钉一)**:NOAA 2025 打开后的运行是"fresh 区域上的反馈消耗式适应"(首正成本在适应过程中测得),按新分类属适应效率读数,非冻结部署 held-out 读数;其 L2 首个有界正例地位不变,但 #42g L2 主读数(冻结终态 held-out)是更严的另一统计量,两者不可互替。

**#42g 四段结构(预注册)**:Part A held-in adaptation(Support/delayed/Slow/Skill 更新全许,预算预冻结)→ Part B freeze(Static*/A3*/A5* 落盘)→ Part C held-out deployment(Fast-only,零反馈零更新)→ Part D offline evaluation(输出全冻后一次性开 outcome,只计分)。**仪器缺口预告(主线补钉二,#42g 书前置项)**:现役 T6 Runner 无 Fast-only 零反馈部署模式(现有 replay 入口均含 open_delayed),Part C 需新增部署入口;Part D 沿用 LabelWall + evaluate_released 既有封存/释放机制(照 #42a/#42c 先例,含"核验可执行体存在"清单)。**#42f 前置义务(主线补钉三)**:获取 census 时即冻结 held-in/held-out 划分(按序列、保 cohort 结构),held-out outcome 自获取起入墙封存;划分冻结先于任何 outcome 打开。

### #42e1 结果 = C23(RISK_PRIOR_EFFECT_AMBIGUOUS);增强线 A(AD)收口,#42g L2 关闭;#42e2 书发出(2026-08-23 21:4x,主线裁定)

**判定追认(阶梯外新格)**:执行方走完五格预注册阶梯逐一不中——非坍缩(A5 非恒等 1 = A3)、非 BLOCKS(读数 4 空,A5 同样拿到 mad+)、非位移(winsorize 双臂 0)、非 EFFECTIVE(A5 风险算子探测未更少)、非 INERT(CPM r2 池宽异,A5 多列 hampel)——新立 RISK_PRIOR_EFFECT_AMBIGUOUS 而不强贴标签,正确处置,主线追认入判定集。**阶梯缺格教训(主线认领)**:书面未预置"方向相反的歧义关联"格;今后行为验收类判定集必带 residual 格(OBSERVED_BUT_UNCLASSIFIED → 上报主线),沿常备纪律 2 精神。

**C23 canonical(主线依回报入典)**:Part 0 = 20218007b(八件,密钥扫描 clean,runner `authorization:` 形参与 docs `authorization=GUIDANCE` 记科学用语);Part A 从冻结 entry 重建 h0s_v3 未调 Slow、哈希 f2054da1… 相符、送达 4/4-0/4、构造期 Memory held=0;Part B run-id 20260823T213436Z 一次性(锁 + 无 leftover 首次独占),LLM 22/32、fit 12/24、重训 0,墙 breached=false(target_key_requests=6 = evaluate 释放路径对 CPM delayed 计分的合法读取,与 #42e0 同口径)。前六格双臂 identity-only、三风险算子提案 0,信息量集中 CPM r2:A3 池 [identity, mad] 探 mad Support +0.0593 → delayed +0.0111 POSITIVE → LOCAL_ACTIVE;A5 池 [identity, mad, hampel] 先探 mad 同读数同终态(hampel 列 probe_order 第二未实探);**A5 未少探 mad,反多列 hampel,提案理由未引用降权条款(cites_risk_knowledge=false;cell_blob_cites=true 系工件 skill_id 字符串非理由引用——"关联≠因果"与在册"引用≠遵从 n=7"成对)**。读数 5 全程 NO_OVERRIDE_OCCASION;读数 10 无位移。v3 六段未改;交付未 commit。

**科学定性(主线,承重)**:送达成立、触发场合实存(mad 被双臂探测),但**未观察到可归因的行为效应**——"送达≠采用"家族最干净一例。**风险事实(反直觉,入册)**:v3 的无条件 mad 降权与本 Target 唯一自然正向 mad 证据方向冲突;若降权被严格执行且致跳过 mad 将损失 +0.0111,但该反事实未发生,不得写成已观测负迁移或"v3 实为误导"。当前同时存在两个候选缺口:census 层缺合法 Pattern/Context 分辨力、Fast 层未操作化先验(提案不引用);#42e1 无法区分二者,故 Pattern 条件化只是下一项有界测量假设,不是已证明的唯一必要修复。机制假设留案不归因:RISK 段点名算子可能经提示显著性反向增加提案可得性(#42e0 v2 仅点名 hampel 时 A5 池无 hampel;本轮 v3 点名三算子后 hampel 现于 A5 池——各 n=1 方向一致,留观察不立项)。

**决策**:v3 归档(v1/v2/v3 链全 superseded 保留不删);**#42g L2(AD)= 当前知识下关闭**,A5 臂不入 #42g,#42g 简化为 L1 主考(Static vs A3)——"主线不受增强线拖累"的设计首次兑现。#42e2 即使产出合法 Pattern 条件化候选,也不得自动重开当前 #42g L2:新候选必须建立新 version、另过 development 行为验收,只可进入后续独立正式考场。**增强线 A(AD Source 先验)四轮弧线收口**:#42d(形成未触发)→ 收口 + #42e0(送达无触发场合)→ #42e(census:仅无条件谨慎)→ #42e1(触发场合存在仍无可归因效应)——诚实的诊断性负结果,claim 沿 C22 表不变。evidence_grade = DEVELOPMENT/same-context。

**#42e2 书发出(见分发件)**:增强线 B 首实验,纯测量(0 LLM / 0 AD fit / 0 重训 / 0 新数据);单一 Observation 假设(冻结)= winsorize 作用几何相关的孤立尖峰结构 isolated_dominant(robust z≥4 沿在册词表,isolated_fraction≥0.5,禁扫阈值);判别 + 代理审计(episode 级,两侧各 ≥2 cohort)+ LOCO + 打乱地板/dataset-ID 天花板双基线;mad 反号表零成本描述性附录;判定三格 + 仪器格;一轮一假设,不判定不换特征重跑;交付不 commit。

**C23 后执行锁(防主线再次被增强线拖住)**:#42e2 若已启动,只允许按冻结假设完成这一轮,不得换特征、扩 Source 或追加重放;它的任何结果都不阻塞 #42f/#42g L1。当前无条件下一能力门 = #42f 冻结未曝光 Target → #42g Static vs A3 的 held-in adaptation / freeze / held-out Fast-only / offline evaluation。A5 不进入当前 #42g;Pattern-conditioned 新 Skill 若未来形成,须另立 version、另过 development 行为验收并进入后续独立考场。

### #41b-lite 执行与最小 V10(2026-08-23,执行方报告)

**Part 0 检查点(0 LLM / 0 重训 / 0 AD 评估)**:`git update-index --really-refresh` 后 `git status` 实测 8 件修改(六收尾文件 + 两 docs;刷新前 stat 缓存确实吞改——载重运维发现兑现),逐文件 add、全程未用 `git add -A`。轮始发现 t5_lifecycle_v1.json/.md 为上一次复跑烟测的 CRLF 覆写(未还原),从 687af6e 逐字节还原并核符(处置同 #41 追认先例;本轮自测覆写一次后再次还原)。核验:V9 登记回读——method.py `after_t5` = ccf2b837…a4fa3、e1.py `after_t5` = e5501fe9…1097f,两者恰等于各自收尾后工作树哈希(登记语义 = #41 全程终态)。三份既有 untracked 测试(closeout 时间戳三件 = `test_e1_v2_protocol_repair.py` / `test_skill_evolution_e0.py` / `test_skill_revocation.py`)只跑不入库、不删除;MKL/Savgol 崩溃保持挂账(`test_f1_forecast_pilot` 原样不动,零 skip 标记写入)。

**Part A 最小 V10**:清单机制 = runner 内注册表(`FROZEN_SURFACE_V*` + 逐文件 before/after 注册,非独立注册表文件)。成员 = V9 成员收尾后哈希(39 唯一路径逐文件登记,before = V9 末次注册值、after = 检查点值,全部 carry 无新移动)+ `experience_memory.py` 新增纳入(`FROZEN_SURFACE_V10` = V9 + 1,40 唯一/41 原始条目含历史重复项,`_freeze` 切 V10);零代码移动;`runtime_bundle_sha` 依赖图零改动(Memory 覆盖走清单成员资格,#41b v1 的扩依赖裁决已撤销遵循)。键方言单断言 = T5 Part A A3(a)(b) 复跑:legacy fixture 与 `TASK_CONSUMER_KEY_FALLBACK` 字面量 `forecast|ridge|sMASE` 相等;online_loop 现役无引号硬编码残留,写回键全部经 `task_consumer_key(task_spec)`(:165),唯一幸存提及 = 解释移除的注释。

**Part B 四项验证**:`git diff --check` 干净;四测试 18 passed(`test_g1_proposal_guidance` + 上述三件 untracked,共 18 items);T5 `--smoke-only` 复跑 Part A 11/11(含 A3 键断言 (a)/(b))、Part B **22/22**、`llm_calls=0`;冻结面测试前后两测均 `{"files": 40, "drift": [], "ok": true}`(V10 唯一路径口径;V9 为 39 唯一/40 原始,历史二口径并存已由 #35 勘误记)。

**判定:`V10_READY_FOR_T6`**(另两候选不成立:CHECKPOINT_CONTENT_MISSING——内容俱在,含轮始工件脏态还原;BEHAVIOR_REGRESSION——18 + 22/22 + 11/11 全绿)。Part 0 提交 = 8 文件(六收尾文件 + 两 docs),sha 于回报与 #42 Part 0 幂等核验记入。纪律:O9 零下载零读取(检查点仅对既有冻结成员做字节测量,不开频率/标签值);NOAA 2025 / beyond_17520 / SMD 零读取;零 spawn;另一线停笔(其三件 untracked 测试保持其所有权与内容);歧义如实上报:清单成员资格对 bundle 盲区的覆盖照裁决执行(盲区条目仍在册),V10 切换后 T5 runner 冻结面口径自动 39→40 属授权移动非漂移。

### #31(2026-08-22,S2)

检查点 46ed5e2(8 files)。**CANDIDATE_COMPILES + LODO_TRANSFER_SUPPORTED(双向)**。Part A 硬门过:官方 OmniAnomaly 28 机文件获取(242.3 MB 仅 scratchpad),内容匹配定位(精确字节索引查表,非信号推断),28/28 逐元素一致、无缝铺满 [0,708405),拼接序非数字非字典(machine-1-5 起 machine-3-1 终,在册禁重推);官方 train=dev/held-in、test=sealed(仅报总行数 708420);#30 悬案澄清:[0,8760) 整块落在 machine-1-5 train 内,系一台机器 24 通道被当 24 序列报。Part B:证据池去重 21→12(traffic 8 + noaa 4;13→4 塌掉的 9 条全为同键重放);卡 shared_outlier_repair_with_per_series_guard_v1,四固定字段照裁定(SHARED_CANDIDATE/GUIDANCE/support_required/no_free_try),programs=[hampel,iqr,mad,winsorize]+算子无关 per-series guard(VETO+RESCOPE),适用条件全部部署时可观察(缺失可为零/z峰≥4/outlier_fraction>0/离散度即需 guard),插补+阶跃 out-of-scope 带底物指针。Part C:执行方自查废掉首版两条循环判据后,C1 traffic→noaa 4/4 SUPPORTED(3 条聚合藏害全被 guard 抓)、C2 noaa→traffic 置险 4/4 SUPPORTED(2/2)、C3 12/12 仅标 INTERNAL_CONSISTENCY_ONLY。**跨域量化副产品(升入 C8 证据链):12 行证据 5 条受害全部聚合为正,聚合单独捕获 0 次**。主线裁定:(a) C1 几乎不可证伪、C2 为信息方向,两向 SUPPORTED 挂 n=4 caveat;(b) 两侧证据均 in-selection,卡方向读数不得表述为 out-of-selection,该级证据只能来自 S3/S4;(c) #18/#19 缺口经核为零成本(去重键下与已入池行同票),提取器形状留案不修;(d) 242 MB 不入库,补记 28 文件 sha256 + 来源 ref 使重获取确定;(e) 实体粒度(NOAA 单变量实体 vs SMD 38 通道实体)为 S1b 第一项。S1b 预算解锁(0 LLM / ≤100 重训),书已发。
