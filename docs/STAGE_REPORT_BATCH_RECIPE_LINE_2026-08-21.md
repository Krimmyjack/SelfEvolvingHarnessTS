# 阶段报告:批次配方线 — 冻结 claim / 开放问题 / 仪器事实

重构日期:2026-08-22(Phase C)。**这是新会话 / 新 Agent 接手本线的第一阅读件。**

本次重构**不新增任何结论**,只把已有内容重排为三张表;逐轮流水账原文一字未删,
移至文末第 4 节「历史台账」。遇口径冲突,以本文件第 1 节的 canonical 措辞为准;
遇本文件与工件冲突,以工件为准(每条都给了指针,请直接核)。
规划与纪律见 `docs/ROADMAP_POST_V1_2026-08-22.md`。
当前系统形态、证据切面与数据使用短表见
`docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md`。

**证据等级四档**(常备纪律第 6 条):
`INSTRUMENT` 仪器自证 / `MECHANISM` 银行重放 / `DEVELOPMENT` 已曝光窗 live /
`FRESH` sealed outcome 一次性打开(保留字)。

---

## 1. 冻结 claim 表

每行 = 结论 / 口径上限(canonical 措辞,不得超出复述)/ 证据等级 / 工件指针 / 已知 caveat。

### 1.1 里程碑结论

| # | 结论 | 口径上限(canonical) | 等级 | 工件 |
|---|---|---|---|---|
| C1 | `FRESH_A5_DELIVERS` | pooled 首个正采纳成本 −43.9%(69 vs 123 重训);最终质量与 harm 同冷启动;per_channel = `A5_TIE_TRANSFER_BOUNDARY`;NOAA 2025 fresh 区域在反馈消耗式适应中一次性打开（非 Fast-only held-out） | FRESH | `fresh_confirmation_v1.*` + `fresh_confirmation_v1_adjudication.md`(裁定附录为 canonical 措辞) |
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

### sol 排期审核采纳:承重更正、四钉、#45 压缩、#46 交付定义;#42e2 r1;ccfa.yaml 最小更新(2026-08-23 22:1x,主线核验裁定)

**两处核验(当时口径;L1/L2 层级读法由下文“用户二次校正”supersede)**:(1) 项目 AGENTS.md 与当时主张架构 v2 对齐属实;(2) workspace ccfa.yaml 实读 = 旧 W/E 线元数据(stage 停 2026-08-03,next_actions 停 W61,claims C0–C4 为 A5 主导旧叙事,code_root 指旧布局)——按 sol 建议完成最小状态更新:仅改 stage 块与 authority_note,claims/experiments/next_actions/risks 保留为历史证据不删不改。

**承重更正(组件口径)**:近期两场组件承重 = **#42g(Target 校准端口)与 #44 M1(Consumer 条件化端口)**;#42e2 = Pattern 线索、#42f = 备考、#43 M0 = 前提正控、#45 = 回归。下文用户二次校正后,这两场不再等同项目最终收口;完整系统还欠自然数据 Static/A3/A5 同场验收。

**四钉采纳**:(钉1)#42e2 非阻塞 + claim 封顶——正例侧仅 aws 单 cohort,顶格判词改 **PATTERN_CANDIDATE_CLUE**,本库内永不得表述为"已证跨 cohort 共同 Pattern",升级只可能来自未来新域自然证据;r1 增补页发出(含完美分离时 single_cohort_indicator 必然触发的两种读法区分义务:特征即 cohort 代理[组内无变差,永禁] vs 正例侧结构性重合[组内有变差,记线索上限]);sol"聚合与判词小修"原文未随转述到达,r1 按其意涵落实,分发前若原文另有具体聚合口径以 sol 文本为准合并。(钉2)#42f 义务细化——Yahoo S5 不作同质域:预冻结文件 roster、cohort 定义、**按序列内时间边界**的 held-in/held-out 切分(supersede 主线此前"按序列切"表述;时间切分与部署语义及 NOAA 2025 时间 held-out 先例一致)、标签隔离方式;held-in outcome 至 #42g Part A 才开,held-out outcome 封存至 Part D 离线计分。(钉3)#42g 三态写死——Static = 无 held-in 适应,identity + 一条预注册固定通用清洗双基线;A3 = 空 Source 在 held-in 反馈适应;A3* = 冻结后 held-out Fast-only(禁 open_delayed/Slow/Skill 更新/看结果重试)。(钉4)M0 纯度 + M1 反作弊——M0 只变 Consumer 归纳偏置,Task/数据/窗口/Program/最终评价目标全同,否则测混合变化;M1 = Agent 读合法 Consumer Context 自主改 Workflow,Runner 禁按模型名派答案。

**尾段定形**:#45 压缩为 #46 前轻量回归,不再发展成大实验;#46 交付定义 = "同一个 Harness 入口处理不同 TaskSpec/Consumer"的最终纵向系统 + 清债(化石名册/RUNNER_INDEX/V3–V8 归档/键方言/相关工作检索),不再造方法。经下文二次校正,#46 前还必须具备自然数据 Static/A3/A5 同场结果,否则只能收口组件而非完整系统。**当前定序**:#42e2 一次性收尾 → #42f 冻结新域与 held-in/held-out → #42g Target 校准组件 → #43 M0 → #44 M1 → #45 轻量回归 → 完整三臂验收 → #46 系统整合。

### 系统形态 vs 归因刻度:用户质询裁定(2026-08-23 22:2x)

**质询**:主线/增强线分开是否违背项目目的——"主线像在 held-in 上快速找一个 Skill";held-in 只是适应集合,不代表跨域经验不起作用;用户预期 = 积累整体起大部分作用,适应是贴合新数据特征的调整;积累与 transfer 不应切得太开。

**裁定**:(1) 系统形态维持用户观:最终交付是单一进化 Harness,积累 → 适应 → 部署为同一生命周期,产品上不可拆;held-in/held-out 只切反馈可得性不切知识可用性——系统臂跨域 Skill 在两侧全程在场,held-out 禁学习不禁知识使用。(2) L1/L2 = 归因刻度非产品拆分:A3 是"关闭积累"的消融臂,使胜利可归因;评审与导师必问"赢的是进化环还是先验",不设 A3 无法回答。(3) "在 held-in 快速找 Skill"不是 L1 定义:承重读数 = 冻结终态 held-out utility/safety,速度只是辅读数(承重读数纪律已册)。(4) 权重问题是实证问题:控制世界旧证据支持积累主导(W47 A5 AdaptAUC 1.0 vs A3 0.875;W56 A4 零射 6/6 正、宏增益 +0.176),自然世界当前只支持条件加速器(FRESH −43.9% 终态平;#42e1 送达不采用且 v3 对 Target 实误——若积累默认主导、适应只轻调,错误先验会沉没新域;"先验软、反馈硬"是负迁移证据逼出的安全设计)。当前总主张维持"适应为必要主干、积累为条件加速器(需 Pattern/Context 条件化)",不得超前声称积累主导。(5) 渐近方向入册:库成熟后逐新域以递减 held-in 预算复测 L2,重复测量支持时"积累主导"即升格,同路重新入场旧 A4 零/少探针——用户终局以测量抵达,不以预设抵达。(6) 命名收敛:弃"主线/增强线",改称 L1 = 必要条件层、L2 = 积累贡献层,今后书内沿用新名。**不变项**:#42g 仍 Static vs A3 承重(必要条件 + 当前无行为验收合格的 AD Skill 可组 A5 臂;v3 已按"对 Target 实误"归档,塞入主考 = 注入已知坏先验);L2 门逐域常开,forecasting 侧正例已在库。

### 用户二次校正:完整系统臂优先于证据层级叙事(2026-08-23,本轮文档化)

**对上段的语义收紧**:“自然证据目前只支持条件加速器”是当前读数上限,不能反向
定义项目架构为“A3 主体 + A5 可选增强”。产品/方法形态固定为 **A5 = 经审计的
跨域积累 + 当前 Target held-in 校准**;A3 是删除积累的消融,Static 是删除适应的
消融。accumulation、transfer、adaptation 是同一自进化循环的相邻阶段;Source 与
Target 是时间角色,不是两套 Harness。可以且必须用分臂回答各部分贡献,但不能用
A3 单臂组件结果替代完整系统结果。是否“积累承担大部分数值收益”仍由实验回答,
不预写结论。

**对排期的影响**:#42g 因 v3 未通过行为验收仍只跑 Static/A3,但降格为 **AD
Target 校准端口的组件验收**,不再称项目最终主考。v3 关闭只关闭该 candidate/该
考场,不关闭 A5 角色。#46 前必须另有至少一次自然数据上的 Static/A3/A5 同场
held-in→freeze→held-out 验收;否则只能报告组件成立,不能报告完整 Harness 收口。
Pattern/Context 也不是可丢弃的解释性增强,而是控制跨域知识何时复用、何时由
Target 反馈修订的核心方法 Context。当前数据与证据全貌见
`docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md`。

**C23 后执行锁(防单一诊断支线拖住组件门)**:#42e2 若已启动,只允许按冻结假设完成这一轮,不得换特征、扩 Source 或追加重放;它的任何结果都不阻塞 #42f/#42g。当前下一能力门 = #42f 冻结未曝光 Target → #42g Static vs A3 的 held-in adaptation / freeze / held-out Fast-only / offline evaluation。A5 不进入当前 #42g只因 v3 未过行为验收;Pattern-conditioned 新 Skill 若未来形成,须另立 version、另过 development 行为验收并进入后续独立考场。#42g 只验 Target 校准组件,不替代后续完整三臂系统验收。

### 用户三次校正:held-in 是多轮 self-harness 适应环境(2026-08-23,本轮文档化)

**正典补充**:held-in 不是“一次 Support 后即丢弃”的一次性集合。freeze 之前,
Harness 可在预冻结的总反馈预算、可用窗口和停止规则内运行 `r1...rR` 多轮:
Fast 提案/probe → Support 与 held-in delayed → Episode 写回/first-fault → 必要时
Slow 单面修正与确定性审计 → Fast replay。前一轮形成的 Episode、Target-local
Skill、Risk 或 Harness Patch 可在后一轮继续使用和修订,这正是 self-harness 的
域内演化环。每次 Consumer 评估/重训仍计预算;同一 Outcome 的缓存重放或重复读取
不得冒充独立新证据。held-out 定义不变:freeze 后零反馈、零更新、Fast-only,
外部一次性计分不得回流本次 Harness。后续 #42g 与自然三臂任务书均须显式给出
最大轮数/总反馈预算/停止规则,不得把“一个 cell 一轮”误写成方法边界。

### 文档体系定形与外部写入追认;全项目总结交付(2026-08-23 22:3x,主线)

**外部写入追认**:上节"用户二次校正"与 `PROJECT_STATE_AND_DATA_MAP_2026-08-23.md`、项目 AGENTS.md §2 三臂归因/§5 状态锁为同批用户授权写入(非主线执笔)。主线逐条复核采纳,全部入典:产品形态 = A5(经审计跨域积累 + Target held-in 校准),A3/Static 为归因消融;#42g 降格为 AD Target 校准端口组件验收;**#46 前必须另有至少一次自然数据 Static/A3/A5 三臂同场 held-in→freeze→held-out 验收**(需合格累积 Skill + sealed Target 同时具备,数据供给为用户决策点);前节(4)中"适应为必要主干、积累为条件加速器"降为当前读数上限陈述,不作架构定义。

**文档地图定形(两文档制条款修订)**:AGENTS.md = 长期方法权威(sol 维护)| ROADMAP = 排期与常设规则(主线)| STAGE_REPORT = 逐轮台账(主线)| PROJECT_STATE_AND_DATA_MAP = 人读导航快照(状态/数据曝光/Pattern 历史;与工件冲突以工件为准)| ccfa.yaml = workspace 元数据(已最小更新)。此外不再新增长期文档;历史 docs/*(V1 计划、Phase S 预注册等)保留为证据记录不删不改。

**本轮交付**:应用户令产出全项目总结(目标/技术路线/证据现状/未来计划,chat 交付,不新建文档);内容与本台账、roadmap、数据图核对零冲突。

### #42e2 结果 = C24(PATTERN_NO_DISCRIMINATION);增强线 B(AD Pattern 机制)暂闭;#42g 多轮 held-in 钉;#42f 书发出(2026-08-23 22:5x,主线读工件核验)

**判定追认**:`PATTERN_NO_DISCRIMINATION`,阶梯应用正确(非仪器故障 → C1 不达标即出格,2–4 格未到);family isolated-extreme × winsorize 关闭,不入任何 Skill Scope,未形成 Skill;evidence_grade = MECHANISM/development,非 Capability 证据;不阻塞 #42f/#42g L1,不重开 #42g L2。

**C24 canonical(主线读 md+json 工件核验)**:Part 0 = 3411bb5(HEAD 核符,五件 = #42e1 runner 修改 + 两工件 + 两 docs;密钥扫描 clean);回报特征表缺 aws 两行系转述截断非工件缺失——实读 aws r1 0.7045/True/max_run 172、r2 0.7890/True/172。响应表(delayed 实读):正类 = aws r1/r2;adverse = kc r2、traffic r1/r2、tweets r1;archive = kc r1、tweets r2(NEUTRAL);conflict 0;positive_distinct_cohorts=1(仅 aws)。C1 = {all_positive_same_side=true(True 侧),adverse_opposite_rate=0.25<0.75} 不分离;C2 = {False→仅 kc,single_cohort_indicator=true,usable_as_scope=false};C3 = 剔 aws fold 正类空 LOCO_UNREADABLE,其余三 fold direction_holds 全 false。窗口 = 各 episode train 区间(r1 [0,0.40n)/r2 [0,0.70n))与 bank 形成一致;z≥4 / isolated_fraction≥0.5 未扫描;MAD=0 走现役 public_features floor,无新 epsilon;排除 NA=0。成本 0 LLM / 0 fit / 0 重训 / 0 新数据;未改 v1/v2/v3 与 h0;交付未 commit。

**书面义务偏差两项(不承重,主线豁免)**:mad 反号描述性附录、置换地板/dataset-ID 天花板双基线未产出(json 无对应键,主线核验发现,执行方未申报)。判定不受影响:C1 冻结判据独立成立,两项均预注册为"不得改变主判"的描述参照。豁免理由 = family 已关闭无当前消费者;未来任何新 L3 假设若需 mad 响应表,可从同一冻结 bank 确定性重生成(0 成本)。**教训立规**:今后书面义务须标注"必跑 / 仅通过时跑"两类;执行方漏产任何预注册项须在回报中自行申报,不得静默。

**科学定性(主线,承重)**:干净的机制负结果——孤立尖峰几何**不能解释 winsorize 反号**:aws 正例与 traffic/tweets 负例同为 isolated_dominant=True,唯一 False 的 kc 是长 burst 结构(max_run 140/222 vs traffic 19/22),该特征实质是 kc 结构指示器(C2 单 cohort 指示器与之相符)。first-fault = OBSERVATION_INSUFFICIENT,**限定于 isolated-extreme × winsorize**;不否定 Pattern 总命题、winsorize 的 Context 条件化价值、Source 积累或方法本体。**L3(AD Pattern 机制)当前零活跃候选,增强线 B 暂闭至 #42g 之后**;任何新 Pattern 假设 = 新预注册 family、非阻塞、一轮一假设。

**#42g 多轮 held-in 协议钉(采纳,与协议语义钉合并)**:Part A = held-in r1 → feedback/Episode/Skill 或 Harness 更新 → r2 → … → rR → freeze → held-out Fast-only。任务书冻结:最大轮数 R、总反馈预算、可用窗口、停止规则;**禁预指定 Workflow 或逐轮答案**;delayed 语义沿在册(防同批自提自批);held-out 零反馈不变。

**#42f 书发出(见分发件)**:备考非承重;Part 0 = #42e2 交付检查点;Part A = 本地单变量带标签 AD 候选盘点(存在性/许可/结构/标签形态/曝光状态,禁开任何标签值,禁下载);Part B = 仅当候选合格且布局符合预期,按预注册规则冻结:roster(结构门,不看标签)、cohort 如实登记、每序列 [0,0.7n)/[0.7n,n) 时间边界(与 NAB r1/r2 train 前缀惯例一致)、标签双仓隔离(held-in 仓 #42g Part A 才开;held-out 仓 Part D 才开)、raw 不入库;判定集 = TARGET_FROZEN_SEALED / NO_LOCAL_SEALED_CANDIDATE(→ 下载为用户决策)/ LAYOUT_UNEXPECTED_STOP / INSTRUMENT_UNREADABLE / OBSERVED_BUT_UNCLASSIFIED;0 LLM / 0 fit / 0 重训。

### 依赖梳理与今夜设计预冻结:#42g 预分发稿 v0、M0 设计钉、M1 schema 草案(2026-08-23 23:1x,主线;待 sol 审 + #42f 回填,未分发)

**依赖图**:#42f→#42g 硬依赖(roster/双仓);Fast-only 入口 = #42g Part C 硬前置但与 #42f 无关,并入 #42g Part 0b 实施(烟测只用 dev fixture,sealed 域仅正式运行触碰);M0 数据无依赖(用已曝光 dev 数据),仪器前置 = 第二 Consumer(AegisTS 重构族)薄适配,排期纪律仍 #42g 先行;M0→M1 硬依赖(翻转存在性/配对/效应量定预算);#45 独立仅占带宽;三臂同场 = 合格积累 Skill + 新 sealed 域(用户供给决策,#42g 后拍板;候选:forecasting 侧复活 candidate v2[Skill 已在,缺域] vs AD 侧[Skill 域双缺])+ 复用 #42g 仪器;#46 收尾。

**#42g v0 预冻结决策**:Static 双基线 = identity + hampel_filter 菜单默认参(选型理由 = 领域惯用稳健清洗默认,非取其 NAB 伤害史;对比对象 = best-of 双基线,该选择只可能抬高 A3 门槛);roster 上限 = 字典序前 24(NAB 先例);R_max = 2,r1 train [0,0.40n) / r2 [0,0.70n)(与 bank 形成惯例同构),窗口全 ≤0.7n,停止规则 = 预算尽/一轮无新提案/轮尽;Fast-only 入口硬断言 = open_delayed=0、Slow=0、store 哈希前后不变、逐序列部署日志,违规硬失败;判定集 = ADAPTATION_DELIVERS_HELDOUT / ADAPTATION_AVOIDS_DEFAULT_HARM(A3 平 identity 而 hampel 实害 = 拒有害默认的安全价值格)/ ADAPTATION_TIE / ADAPTATION_HARMS_HELDOUT / PROTOCOL_BREACH / INSTRUMENT_UNREADABLE / OBSERVED_BUT_UNCLASSIFIED;阈值 v0 = 材料线 ±0.005 先例 + worst ≥−0.02 + 伤害份额 1/24(主线拍值待 sol);预算挂公式(LLM ≤24,AD fit ≤200,按 #42e1 每 cell 实测外推)待 #42f 实测回填;方向一致率 = A3* 活跃序列 held-in delayed 号 vs held-out 号。

**M0 设计钉**:配对 = 监督型事件分类器 vs 重构型检测器(AegisTS 薄适配入 consumers/,确定性训练,同轮登记);纯度清单 = 仅 Consumer 归纳偏置可变,Task/数据/窗口/Program 字节/预算/Metric/最终评价目标全同;两段式 = S0 自然 census(40-episode bank 双 Consumer 反号表,A 侧走缓存 0 新 fit、B 侧 ≤300 fit)→ 仅当零翻转才 S1 注入正控(T1b 同族);预期方向预注册 = 清洗训练异常助重构族/损监督族;判定集含 NATURAL/INJECTED_ONLY 分格 + residual。

**M1 schema 草案(M0 后定稿)**:Consumer Context 合法字段 = family / uses_labels / training_objective / anomaly_treatment_in_train / scoring,禁模型名与 ID;反作弊双检 = Runner 只传 schema + 改名不变行为测试;验收链复用 #42g 协议仪器。

### #42f 结果 = C25(TARGET_FROZEN_SEALED);授权下载;#42g r1(sol 七条)追认,书已分发(2026-08-24 00:2x,主线核验裁定)

**C25 canonical(主线读冻结工件 + 隔离抽查)**:执行流 = 本地盘点判 NO_LOCAL_SEALED_CANDIDATE(停报保留)→ 用户授权下载续跑(书面预设决策点,流程正确)。来源链入档:官方 GitHub 404 → HF 官方 gated → 镜像 muditdham/time-series-auto-encoder@24958b84,A1Benchmark 67 文件,raw 不入库。布局 = 公开 schema(timestamp,value,is_anomaly 内联点标签),无官方 train/test → 0.7n 外墙兼容,单自然 cohort yahoo_s5_a1。**长度门偏差(已申报,结果等价)**:书面 ≥800,实用 T6 现役 MIN_LENGTH=1000(仪器一致性);被剔 real_54/real_62 均 n=741,双门下 roster 同为 65。切分 = 每序列 [0,int(0.7n)) / [int(0.7n),n);事件映射 = 连续 is_anomaly==1 → merge_events 行集合,合成 fixture 三例通过(未触真标签)。隔离抽查(主线):work 65 文件仅 timestamp,value(real_1 头行实读)、vaults 65+65、real_54/62 确不在 work。曝光如实:值载入后 context = INSTANCE_SEEN;held-in 标签封至 #42g Part A,held-out 封至 Part D。0 LLM / 0 fit / 0 重训;交付两件未 commit。

**镜像来源 caveat(承重,claim 措辞义务)**:数据为社区镜像副本非官方 Webscope 发行;冻结 JSON 逐文件 sha 锁定本副本身份;考场内部效度不受影响(三臂同字节),但对外引用 Yahoo S5 公开基准数字的可比性未经官方副本核验,今后任何 claim 须注 "mirror copy, provenance recorded"。

**#42g r1(sol 七条)追认,与 v0 合并为正式书(用户已分发,以 r1 为准)**:R1-1 实况回填(roster 65,考场 = 字典序前 24,总长 34,507,min/med/max = 1420/1439/1461,held-in 24,140 行 / held-out 10,367 行;其余 41 条双区标签本轮 evaluator 亦不得请求);R1-2 两轮窗口写死(r1 train [0,.30n) / Support [.30n,.40n) / delayed [.40n,.50n);r2 train [0,.50n) / Support [.50n,.60n) / delayed [.60n,.70n);标签墙按窗序逐开,未来窗与 held-out 不得预载;R_max=2 固定执行,仅预算尽/协议违规/仪器不可读可早停)——supersede v0 的 [0,.40n)/[0,.70n) train 映射;R1-3 Fast 决策粒度保持现役 cohort 级(不新增 24 次独立 Agent 决策),逐序列日志记同一冻结决策的 Scope/实际应用/abstain/输出,LLM≤24 内维持不了则判 **DEPLOYMENT_GRANULARITY_UNSUPPORTED** 且不开正式 held-out;R1-4 评价语义 = 三臂同用完整 held-in [0,.70n) 为下游 fit 底物(identity 不处理 / hampel 冻结默认参 / A3* 只读冻结态 Fast 输出),Program 只作用训练底物,held-out Query 原始字节不处理,事件匹配零额外容差沿现役行集合重叠一对一贪心;R1-5 预算 LLM≤24、AD fit≤240(held-in 上界 144 + held-out 三臂 72 + 余量 24),delayed 必须复用同轮已拟合模型,烟测不复用则正式运行前停报重裁,禁运行中加码;R1-6 Static 口径 = identity 主 Static,hampel 改称"固定清洗压力基线"(主线"领域默认"措辞收回),best_static_macro = 两完整静态臂宏均 F1 全局最大、禁逐序列 oracle 选臂,AVOIDS_DEFAULT_HARM 改名 **ADAPTATION_SAFETY_ONLY**(只算安全结果,不表述为 held-out 效用提升);R1-7 判定优先序 = PROTOCOL_BREACH → INCOMPLETE_LLM_BUDGET / CONSUMER_FIT_BUDGET_EXCEEDED → TARGET_FEEDBACK_UNREADABLE → DEPLOYMENT_GRANULARITY_UNSUPPORTED → INSTRUMENT_UNREADABLE → 科学结果格 → OBSERVED_BUT_UNCLASSIFIED。

### 数据三角色术语钉与比例语义裁定;#42g 不中途改(2026-08-24 00:3x,用户 + sol 会审,主线采纳)

**语义更正(用户直觉 → 三角色拆分)**:用户直觉 = "held-in 应少于 held-out(适应数据非训练数据)";sol 拆分数据三角色采纳为正典:(1) **Target base-train** = 下游模型拟合 + Agent 无标签 Context,零 Harness 反馈;(2) **feedback-bearing adaptation windows** = Support/delayed 反馈窗——真正应当控小的量;(3) **frozen held-out** = 零反馈验收。应小的是 (2),非 (1)+(2) 前缀整体。今后文档与任务书不再把整个前缀称"适应集",改用三角色命名。

**参考核对(sol 实核,主线快搜未复现仓库、未独立复核,记参考值)**:本地 Self-Harness = 43 train/held-in 任务 + 21 heldout ≈ 67/33,无"held-in 必须更小"规则;可借鉴点 = held-in 多轮反复用于 Harness 改进、heldout 用于候选验收,非具体比例。**无固定比例教条;claim 义务 = 显式报告反馈窗份额。**

**#42g 现行书裁定**:不中途改协议,按"70/30 外墙下第一版组件实验"跑完。实际三角色在 r1 墙设计中已隔离(train 前缀无标签,墙按 Support/delayed 窗序逐开),仅命名混用;本轮反馈窗份额 = **40%**(r1/r2 各 Support+delayed 10%×4)。回报与主线裁定的措辞义务:结果 scope 注记"40% 反馈窗份额下的适应组件验收"。

**后续选项登记(不排期,#42g 后拍)**:(a) 剩余 41 条 sealed 序列预注册更小反馈窗份额做后续确认——sol 20% 方案备案:base-train [0,.50n),r1 S [.50,.55)/D [.55,.60),r2 train [0,.60n)、S [.60,.65)/D [.65,.70),held-out 30%,满足"反馈数据 < held-out"且保两轮自进化;已开标签的 24 条不得换比例重称 fresh;(b) 下一新域直接冻结 base-train / adaptation / held-out 三分协议。

### #42g L1 结果裁定 = C26:PROTOCOL_BREACH(部署绑定);ADAPTATION_HARMS_HELDOUT 撤回;AD 特设设计原则;#42g-b 诊断书发出(2026-08-24 01:2x,主线依 sol 代码审计裁定)

**改判(执行方科学格拒绝,按 R1-7 优先序)**:C26 = **PROTOCOL_BREACH**,注记 HELDOUT_WORKFLOW_ATTRIBUTION_UNAVAILABLE;ADAPTATION_HARMS_HELDOUT 不入 claim 表。三处字节级违规(sol 代码审计,主线核):(i) runner:5811 `winner = cells[-1].winner` 优先于 `deploy.applied_program`——冻结态 = h0、skills/learned/ 空、delayed 未批准、Fast-only 实际未采,`outlier_mad` 仍被强行计为 A3* 部署,违 R1-4"A3* 只读冻结态 Fast 输出";正确绑定下 A3* ≡ identity;(ii) runner:5717 `slow_agent=None, allow_slow=False`,违书面"Episode/Slow/Skill 生命周期全许",静默降配;(iii) 首进程死于 OUT_L1 落盘前,held-in 逐格表/该次 LLM-fit 计数/方向一致率永久缺失,补跑 `--l1-score-heldout-only` 未先补轨迹——held-in/freeze/deploy 三段工件未在开 held-out 前分别落盘。执行流两阶段:held-in 一次跑 20260824T002018Z;Part D 首崩于别名映射(`extreme-deviation-mad`→`outlier_mad`,崩于开仓前,非换样本),修复后仅补计分。Part 0 = 21bff04(四件);Fast-only 烟测 = DEPLOY_FAST_ONLY_SMOKE_OK(曝光 fixture,open_delayed=0,store 不变,未读 Yahoo)。执行方漏产自报义务兑现(C24 规),但未按 R1-7 首查协议格——**新站规:Part D 必须断言 scored_program == deploy.applied_program,阶梯行走从协议格开始**。

**保留读数(development,EXPOSED,可靠)**:Yahoo A1 前 24 条,`outlier_mad` 作全 cohort 训练底物处理 → held-out macro 事件 F1 0.2624 vs identity 0.3227(Δ −0.0602),受害 7/24,worst −0.6667(real_11);hampel 压力基线 0.265(−0.057)自伤 12/24;identity = best_static。**逐序列异质:改善 5 / 平 12 / 伤 7**——cohort 粒度掩盖 series 条件化,恰为项目核心命题(Pattern × Program × Consumer)的自然证据素材,但当前无合法 Observation 区分 5 与 7。计分 fit=72,补跑 LLM=0。前 24 条双区标签 EXPOSED 永久 development;**41 条双区 sealed 完好**。

**反馈稀疏事实(sol 读 EXPOSED 标签,主线采信,#42g-b 工件化义务)**:held-in 反馈窗事件极稀——r1 Support 3 事件(21/24 空窗)、r1 delayed 3(23/24 空)、r2 Support 5(20/24 空)、r2 delayed 5(19/24 空),反馈区合计 14 事件、14/24 序列全程无事件;held-out 38 事件、仅 2/24 空。事件质量沿时间非平稳,反馈分布与 held-out 错配;空窗上 macro 事件 F1 主要读误报。解释:难成 delayed POSITIVE、无 Target-local Skill、held-in 选择不预测 held-out——"有效反馈事件太少",非"时间点太少"。

**两常备格立规**:NO_FROZEN_ADAPTATION_STATE(冻结态与初始 h0 等价且无 learned Skill 时,A3 列按构造 = identity 报告,不单独消耗 held-out);FEEDBACK_EVENT_STARVATION(反馈窗事件数不足支撑判定时如实报,不得记能力负结果)。**修复三件(机械,#42g-b Part 0b,字节可验)**:(a) A3* 计分只取 deploy.applied_program + Part D 绑定断言;(b) held-in/freeze/deploy 三段工件各自落盘且先于任何 held-out 打开(顺序断言);(c) Slow 接线按书面配置,禁静默降配。

**AegisTS/TimeClaw 对照教训(sol 代码读证入册)**:AegisTS = 整份数据反复清洗 + 每动作重训轻量下游取密集 reward + 离线 meta 选择器(FMMS/VUS-PR)+ 冻结 RL 策略零射新数据集(RLclean.py:365/:1208,detectors/main.py:45);TimeClaw = 任务实例 ~50/50 切分 + train 实例直给答案记录工具轨迹 + 重复成功提炼 Memory/工具(相似任务 held-out 通过率门)+ test 冻结检索求解(common.py:70,tsrbench.py:355,agents.py:97,summarize.py:100)。两者反馈单元均远大于 #42g 设置;我们的差异化(真下游反馈在线适应 + 零反馈冻结部署)保留,反馈密度设计向"held-in 整池 + 多轮重用"修正。

**AD 特设设计原则(主线裁定,回应用户"AD 与预测不同需特设")**:P1 反馈以事件质量计,不以时间份额计——AD 的效用信号只住在事件上(预测任务每个 origin 都有连续 sMASE 信号,AD 空窗只有误报面),每书报告反馈窗事件数,starvation 格兜底,拒绝以聚合标签预窥换取设计便利(sealed 纯度优先,靠扩 roster 降 starvation 概率);P2 series 级条件化 Scope 一等公民,abstain-by-default,禁全 cohort 强推(5/12/7 异质为证);P3 Program Supply 现实性——五菜单系 forecasting 承继,清洗训练底物对检测器可能本质反向(T1b/M0 翻转逻辑),AD-native 候选仅在 headroom census 证菜单枯竭后新增一个;P4 生命周期硬门——仅 LOCAL_ACTIVE Skill 可冻结部署,NO_FROZEN_ADAPTATION_STATE 省 held-out;P5 held-in 整池反馈 + 多轮重用(撤销窗内顺序墙,Support/delayed 角色切分保留防自批);P6 确认场 = 41 条 sealed,50% base-train / 20% adaptation / 30% held-out 三分协议。

**#42g-b 诊断书发出(见分发件)**:0 LLM,EXPOSED 24 条上五程序 × 24 序列全响应表(held-in 底物 fit 一次、双区计分),headroom 判定树 + 事件稀疏表工件化 + 选择缺失核查;主判 = NO_MENU_HEADROOM / GLOBAL_HEADROOM_EXISTS / PARTIAL_SERIES_HEADROOM_ONLY(副 flag:SELECTION_MISS)/ INSTRUMENT_UNREADABLE / OBSERVED_BUT_UNCLASSIFIED;fit ≤150。诊断出结果前:不下载、不改 Harness 本体、41 条不动。

### #42g-b 结果裁定 = C27:PARTIAL_SERIES_HEADROOM_ONLY 采纳;"去污染不变形"机制读法;AD 就绪缺口两问裁定;#42g-b2 派生读数书发出(2026-08-24 01:5x,主线)

**裁定**:采纳,无改判。Part 0 = 92d5bf2(六件 allowlist,密钥命中均科学用语,Yahoo raw/vault untracked);三机械修复字节验收((a) A3* 只认 deploy.applied_program + Part D 断言;(b) held-in/freeze/deploy 三段落盘先于任何 development_exposed_eval 打开;(c) Slow 由 L1_ALLOW_SLOW/L1_SLOW_AGENT 注入,本书 False/None)。120/150 fit、0 LLM、41 条 sealed 零读取、methods/ 零改动;非 Capability 主张。

**判定事实**:B1 全局无可行动 headroom——四非 identity 程序 development_exposed_eval 宏 Δ 全负且受害 >2/24(iqr −0.030/5;mad −0.060/7;hampel −0.057/12;winsorize −0.092/14)。B2 局部 headroom 存在:iqr 6 条(现役 public feature 关联不可见)、mad 5 条(可见)、hampel 7 条(可见)、winsorize 4 条未过门;关联仅副读数,未扫阈值未出 Scope。B3 现役反馈 estimand(四窗并集 [.30n,.70n))无程序被偏好:全体 24 全负,有事件 10 条仍全负,零事件 14 条一律 −0.0714;稀疏事实 = 14/24 反馈窗零事件,r1 Support 有事件仅 3 条。对 #42g L1 的 first-fault 读法:A3*=outlier_mad 伤 held-out 与无全局菜单 headroom 一致,不能单独证明 Agent 选错(轨迹丢失、映射曾错、反馈 estimand 也不偏好任何非 identity)。**claim cap 在册**:若引用菜单枯竭类结论,上限 = "现役 Consumer、五程序菜单及预注册全局/局部门下无可行动 headroom",不得写成"AD 数据不存在可处理空间";真 held-out 仍是未读的 41 条。

**机制读法(主线)**:受害排序 = 变形强度排序——winsorize(无条件裁全尾,最强变形)−0.092/14 > hampel(滚动改写)−0.057/12 > mad/iqr(点删除,最接近纯去污染)且后者局部赢面最大(5/6 条)。机制:清洗压缩训练底物的合法尾部 → 检测器学到过窄"正常边界" → held-out 原始字节的合法尾波动被读成异常 → 误报塌 event F1。局部获益 5–7 条 ≈ held-in 真含污染事件的序列。= Pattern × Program × Consumer 条件化在自然数据上的显形,与 M0(仅换 Consumer)设计互证。

**用户两问裁定**:Q1(Harness 是否只有预测就绪的指导、缺 AD 就绪定义)= 是,准确表述:就绪定义一贯 consumer 相对(效用定义),预测任务也无公理化定义;预测线拥有的是三件匹配——菜单动作(压平不规则以利拟合)对准预测 Consumer 失效模式 + 多域积累 Pattern→Program 证据 + 与之共生的观测特征;AD 三件皆无。AD 可定义的一般原则 = **去污染而不变形**(底物须忠实代表 normal 含合法尾部,只删真异常);预测就绪允许甚至奖励变形,AD 就绪惩罚变形——两目标在"删真垃圾"重合、在"重塑分布"背反。此原则 Harness 尚未编码于菜单/Pattern/特征,正是跨域积累要发现的对象。Q2(headroom 低是否因算子系预测遗产)= 对了一大半:机制排序为证;但 census = PARTIAL 非 NO_MENU——菜单未枯竭,局部可用而(a)无全局规则(b)iqr 赢家现役特征不可见。first-fault 移至 Scope/Observation 与 Program Supply 并列待分叉。

**#42g-b2 派生读数书发出(0 LLM / 0 fit,纯重聚合已落盘工件)**:A1 oracle-scope 天花板(每序列取菜单含 identity 最优的宏 Δ + 程序被选分解);A2 方向一致率(限 10 有事件序列,sign(反馈窗 Δ) vs sign(held-out Δ) + argmax 一致,恢复 L1 丢失读数);A3 赢家集合重叠结构。决策门(非科学门):Δ_oracle ≥ +0.02 → SCOPE_LINE_MATERIAL;< +0.01 → SUPPLY_LINE_FIRST;之间 BORDERLINE;一致率 ≤50% → flag FEEDBACK_UNIT_REDESIGN(反馈单元按事件质量重设)。字段缺失允许 ≤120 fit 确定性重算并自报。41 条 sealed 不动;AD-native 特征/Workflow 押后至分叉裁定。

### sol 三分诊断字节核验追认;"契约=积累基质"边界钉;anomaly 契约 v0 设计钉(2026-08-24 02:0x,主线)

**sol 三分诊断(AD 语义缺失 / Observation 不够 / Program Supply 不够)追认,七处引证主线字节核验全通过**:(1) forecast_task_quality_contract_v1 完整六段(objective/preserve/harms/evidence/verification/abstention,task.py:336);(2) anomaly 只有薄 TaskSpec(anomaly_task_spec_v1,task.py:578,注释"P2 最小 anomaly rig 预留口径"),无 anomaly_task_quality_contract_v1 / anomaly_task_context_v1;(3) 词表早已预留——preserve_anomaly_evidence(QUALITY_OBJECTIVES,task.py:43)、anomaly_events(PRESERVATION_VOCABULARY,:51)、event_erasure(HARM_VOCABULARY)、文件头"anomaly 下 spike 是信号非缺陷"(:9):**代码词表层早知 AD 关键区分,从未组装成契约**;(4) public_features.py:27-31 = 240=192+48 Ridge/forecast 语义烧进公开特征;(5) registry.py:36-41 = 四点式离群算子对 AD 仅"物理禁→机械可试","allowed_tasks 只表达可试不表达有益";(6) runner:174-177 五菜单 frozen since T3;(7) bootstrap 指导机制词表 = missingness/极值/level shift/period change,无"污染 vs 证据"区分。**加强性发现(主线)**:event_erasure_guard 在 classification 技能线已实际运转(multiskill fast-path 决策回执 RETRIEVE_AND_ABSTAIN + event_erasure_risk,fallback IDENTITY)——"弃权保事件证据"机制有活先例,缺的是 AD 实例化;支持 sol"非总体设计失败,缺适配层"结论。

**sol first-fault 排序采纳(作先验)**:主 = AD 原生 Observation/series 级 Scope 不足;次 = AD-native Program Supply 可能不足;已排除 = Agent 太弱(#42g-b 系 0 LLM 确定性全枚举)——排除项限本 census,#42g L1 原 Agent 行为因轨迹丢失不可裁,仍归 PROTOCOL_BREACH。**排序与 #42g-b2 决策门关系**:sol 排序基于赢家条数(6/5/7)未见幅度;Δ_oracle 门(≥+0.02 Scope 线 / <+0.01 Supply 线)仍是权威裁定器,sol 排序记为预期结果。

**两处边界钉(主线对 sol 方案的修正)**:(一)**契约 = 积累基质,不是手写答案**——researcher 只写"轴"(危害/保留词表、安全默认、验证维度),Pattern→Program 条件化规则("尖峰序列勿 winsorize"类)必须由 Experience/Skill 证据长出,禁手写入契约;否则 M1 反作弊批评适用,积累主张死亡。(二)sol 步骤 4 措辞精确化:协议本就保证 held-out Query 原始字节不处理(R1-4),AD-native 程序的真靶 = **训练底物去污染且不变形正常边界**,非"不擦 Query"。

**anomaly_task_quality_contract_v1 设计钉 v0(待 sol 审,零仓写)**:objective = preserve_anomaly_evidence(词表在册);preserve = anomaly_events + normal_boundary_fidelity(新词:合法尾部与方差结构)+ temporal_order + series_length;harms = event_erasure(在册)+ normal_boundary_shrinkage(新)+ false_alarm_amplification(新)+ unnecessary_modification + out_of_scope_change;evidence_expectations = 与 forecast 同三件;verification_dimensions = 沿用六件 + tail_mass_change_receipt(报告项非门:程序前后底物尾部质量变化机械回执);abstention_conditions = 四件 + no_contamination_evidence(无训练污染证据默认 identity,把 abstain-by-default 制度化);anomaly_task_context_v1 = spec(判官名落实为 T6 实际 Consumer)+ 本契约 + deployment_constraints(固定 Consumer/maximum_candidates/max modified fraction)。**排程**:契约钉现在送 sol 审(与 #42g-b2 并行,零仓写);b2 回报分叉后一本书合并 wiring(contracts/task.py + context + 测试)与分叉侧第一步(Scope 线 = 选局部正响应程序找部署可见非代理 Observation / Supply 线 = 一个 AD-native 去污染程序);单执行线纪律不并行两本。

### sol 契约 v0r1 三修正采纳;审阅侧预读定分叉;执行序改钉 = Feedback Unit → 契约 wiring → Scope(2026-08-24 02:1x,主线)

**契约三修正全采纳**:(1) **改名限 Consumer 族**——anomaly_background_model_quality_contract_v1,只服务学习背景/正常边界类 Consumer(现役 aegists_iforest_v1);未来监督事件分类器另立契约(可能需保留训练异常样本)——与 M0 Consumer 条件化命题同构,防"一个 Consumer 的好数据冒充全 AD 公理";anomaly_task_context_v1 = TaskSpec + 本契约 + 固定 Consumer 约束。(2) **删 tail_mass_change_receipt**:反轻量纪律、量未定义、阈值体系风险;尾部压缩/修改比例/方差变化降为实验报告普通诊断字段,不进 Contract/Gate/Receipt。(3) **删 no_contamination_evidence 枚举**:现无合法 contamination Observation,硬写则全局默认 identity 或诱导 Agent 把极值冒充污染;弃权语义由既有 insufficient_public_evidence 承担,专门词汇等真实 contamination Observation 形成后再议。**契约 v0r1 定稿 = sol 推荐版**:objective preserve_anomaly_evidence;preserve = observed_values_outside_suspect_region + temporal_order + series_length + anomaly_events + normal_boundary_fidelity(唯一新 preserve 词);harms = event_erasure + normal_boundary_shrinkage(新)+ false_alarm_amplification(新)+ unnecessary_modification + future_information_use + out_of_scope_change;evidence/verification/abstention 全沿用现有词表零新增;anomaly_events 注释钉:"指保护 downstream event discrimination 所需证据,不表示训练区内任何疑似异常点都禁止删除"。两红线(轴不写答案;Query 墙非 Program 目标)sol 确认。

**审阅侧预读(sol 零写入重聚合,记 review-side 证据,待 b2 工件复核;两独立确定性计算必须一致,不一致 = 仪器故障)**:oracle 上界 **+0.0375**(14/24 局部正)→ 按预注册门 ≥+0.02 = SCOPE_LINE_MATERIAL;材料线方向一致 **17/40 = 42.5%** ≤50% → FEEDBACK_UNIT_REDESIGN;保守 feedback-selected policy 后段宏 **−0.0070**(伤 2,worst −0.10)——现役反馈单元驱动的选择 ≈ identity 减 epsilon。预期 b2 主判 = SCOPE_LINE_MATERIAL + flag FEEDBACK_UNIT_REDESIGN 并存。**读法钉**:一致率 ≈ 掷硬币 + policy ≈ 微负,结论是"非可信号"(non-informative),不下"反信号"重话;Scope 潜力不薄(+0.0375)但在反馈修好前不可学、不可归因。

**执行序改钉(sol 因果论证采纳:Scope 学习消费反馈,反馈是噪声则 Scope 不可归因)**:b2 工件复核 → **书 n+1 = Feedback Unit 重设计**(EXPOSED 24 上开发:预注册 2–3 候选反馈单元[事件质量池化 / 空窗单侧误报语义 / 整 held-in 池化],判据 = 方向一致率 + 对 oracle 的 policy-regret,选定即冻结,41 条上用新单元作确认)→ **书 n+2 = 契约 wiring**(contracts/task.py + context + 测试,纯代码不跑行为)→ **Scope/Observation 线**(修好的反馈单元上,选局部正响应程序找部署可见非代理 Observation)→ EXPOSED 24 闭合 → **41 sealed 三臂终考**(50/20/30,反馈窗份额按新单元语义届时复核)。原"契约 wiring + Scope 第一步合并一书"作废。契约文字审查可并行;落码与行为实验不并行。**b2 不中途加读数**:sol 的 policy 读数记审阅侧,Feedback Unit 书将以"现役单元基线"身份预注册重算。41 条 sealed 不动。

### #42g-b2 收口 = C28:FEEDBACK_UNIT_REDESIGN 门生效;仪器交叉验证通过;赢家不相交约束入册;#42h 反馈单元书发出(2026-08-24 02:2x,主线)

**采纳,无改判**。0 LLM / 0 fit,纯重聚合 t6_42g_b_menu_headroom.json,无 fallback;Part 0 = eef656b(五件:headroom json/md + runner 0b/--menu-headroom-v1 + docs 两件;密钥命中均科学用语;Yahoo raw/vault untracked);41 条 sealed 未读。**仪器交叉验证**:executor A1/A2 与 sol 审阅侧零写入预读逐数一致(Δ_oracle +0.0375;正贡献 14/24;合计一致率 0.425)——两条独立确定性计算相同,重聚合仪器可信;sol 的保守 policy 读数(−0.0070)按钉留审阅侧,#42h 以 U0 基线身份预注册重算复现。

**A1**:Δ_oracle = +0.0375,正贡献 14/24(real_21 +0.005003 边缘过线);程序分解:identity 被选 10 / hampel 6(贡献和 +0.383,**最大贡献者**)/ mad 3(+0.318)/ iqr 3(+0.069)/ winsorize 2(+0.131)。**hampel 同时是全局最害(12/24)与 oracle 最大贡献者 = 高方差算子,series 条件化命题的最强单点证据**。报告 caveat 正确:oracle 只证"若每条都被正确识别的菜单局部空间上限",不证部署时存在合法可观察 Scope,非 Capability 证据。**A2**(10 有事件序列,材料线 ±0.005):合计 0.425 ≤ 0.50;identity 10/10 全 NEUTRAL×NEUTRAL 正确剔除;mad 7/10 唯一有牵引;hampel 1/10(4 次反馈 NEG→eval POS,2 次反向)——小 n 描述性记录,不下"系统性反向"结论;iqr 5/10、winsorize 4/10 且两者反馈侧零 POSITIVE;argmax best-set 交集 3/10(13、3、30)。**A3(新增载重)**:iqr∩mad∩hampel = ∅,两两重叠 ≤2,并集(含 winsorize)= 14——**"单一可清洗子群"故事证伪**;Scope 线(押后)须程序条件化 Observation 而非单子群检测器;单一全局程序吃不到 oracle,与 B1 全负互证。

**门裁定**:合计一致率 0.425 ≤ 0.50 → FEEDBACK_UNIT_REDESIGN 优先于 SCOPE_LINE_MATERIAL(Δ_oracle +0.0375 ≥ +0.02 成立但按预注册优先序 + sol 序改钉不走 Scope);门不入 claim 表。交付 t6_42g_b2_derived_readouts.json/.md 未 commit(#42h Part 0 义务)。

**#42h 书发出(见分发件)**:EXPOSED 24 条上四候选反馈单元预注册禁增删——U0 现役基线(四窗并集,窗宏 F1 Δ;义务复现 −0.0070)/ U1 读出修正(同区,区内事件池化单一 Δ,零事件序列降误报单侧通道)/ U2 区扩展(全 held-in [0,.70n),读出同 U1)/ U3 cohort 池化(全区跨序列 micro 合一)。判据:C2 policy 模拟为主(保守门 Δ>+0.005 才离开 identity,eval 宏 Δ + 受害 + worst + 对 oracle regret),C1 方向一致率(材料线,双侧序列)报告 + 弱则 flag FEEDBACK_DIRECTION_WEAK 不阻塞,C3 双侧覆盖;并列(≤0.005)取反馈份额更小/改动更小者;胜者须 C2 > 0,否则 FEEDBACK_UNIT_UNRESOLVED 回设计台。义务:[0,.30n) 事件计数补稀疏表缺格。fit ≤150、0 LLM;**选定即冻结,#42g-c(41 条)只用不调**,反馈份额与事件计数作协议事实报告。

### #42h 收口 = C29:FEEDBACK_UNIT_UNRESOLVED;菜单"操作性枯竭"裁定 → Supply 线解锁;弃权即知识注记;#42i 纯代码书发出(2026-08-24 09:4x,主线)

**采纳,无改判**。Part 0 = 80fd455(四件;密钥命中均科学用语;Yahoo raw/vault untracked);0 LLM / 120 fit / 0 重训;methods/ 未改;41 条 sealed 未开。**U0 锚逐字节复现**:macro −0.00697816676077546,harmed 2,worst −0.1,choices = identity 21 / hampel 2(real_19、real_23)/ mad 1(real_29)——审阅侧与执行侧第三次独立一致。

**预注册歧义裁定**:书面 U0"四窗宏"与 sol 锚不符(窗宏读数 = −0.00762 / worst −0.20 / 19-1-2-1-1);executor 以锚复现为准将 U0 绑到 #42g-b 并集池化 estimand,U1 随之与 U0 塌缩——处理正确、自报完整,追认。主线自认书面 U0 定义不精确;**站规:锚复现义务是有效预注册手段(本轮靠它抓住歧义),保留**。**安全门追认(应用版预注册)**:macro > +0.005 且 harmed ≤ 2/24 且 worst ≥ −0.02。四单元全不合格:U0/U1 −0.006978(宏败);U2 全 held-in 区 +0.005059 但 worst −0.071(安全败);U3 cohort 反馈四程序全 NEGATIVE → 全 identity,C2 = 0(宏败;其 C1=1.0 系"两边都伤"的 4 点描述性,非可迁移正向)。稀疏补格:[0,.30n) 有事件 9 条 / 反馈并集 10 / 全 held-in 14;零事件序列不得授权正向采用。

**根因与解锁裁定**:约束不在读出公式——U2 已用尽 held-in 全部事件质量仍不过安全门;根因 = 菜单效应薄而散(oracle +0.0375,赢家不相交)× 事件稀疏非平稳(held-out 38 vs held-in 14)。**裁定:现役五菜单在本 cohort 上"操作性枯竭"——局部 headroom 存在但无合法反馈机制可安全收割;P3 解锁条件(#42h 预承诺)生效,批准新增一个 AD-native 程序。禁再造 U4/U5(单元对 24 条过拟合风险);41 条不动。**

**弃权即知识注记**:C26–C29 链构成"本 cohort × iforest 族 × 点清洗菜单 → 正确策略 = identity/弃权"的 development 证据;该负迁移守卫本身是可积累 Experience(A5 应携带的跨域知识形态)。两种结局都给 #42g-c 良定义三臂终考:新程序合格 = capability 正结局;仍不合格 = 弃权/安全正结局(A5 正确弃权零伤害 vs 强采纳消融受伤)——AD 线不停摆。**多轮 replay 门追认(executor 提案)**:任何单元/程序合格后、开 41 条前,必须在 EXPOSED 24 上真实多轮 Harness replay(Support/delayed 分回执;delayed 不得批准自己产生的提案)。

**#42i(纯代码,不跑行为)+ #42j(census)设计钉**:#42i = 契约 v0r1 wiring(contracts/task.py:anomaly_background_model_quality_contract_v1 + anomaly_task_context_v1 + 测试)+ AD-native 程序注册 **contamination_mask_refit_v1**(机制 = raw 底物 fit → 评分 → 固定分位遮罩 top ≤1% 含毗邻段延伸 → 重 fit;单次迭代;禁参数扫描;intrinsic targeting;AD-only;**值不改写、仅对 fit 不可见**——"去污染不变形"直接实例;遮罩实现[NaN 或行删]由 executor 提案、sol 审,须与 consumer 窗口化兼容并烟测)+ census 仪器扩至六程序;Part 0 提交 #42h 交付。#42j = EXPOSED 24 六程序响应表 + 同安全门资格判(macro>+0.005 / harmed≤2 / worst≥−0.02)+ U2 池化方向与策略模拟;合格 → 多轮 replay 书 → #42g-c;不合格 → 弃权框架 #42g-c。

### 外部文献调研(grok,AD 训练数据质量)审读采纳为设计输入;M0 文献背书预测预注册;pre-#42j 预期注记;Yahoo 病理 caveat 扩栏(2026-08-24 10:1x,主线)

**审读判定**:主体成立,采纳为**设计输入**(非 claim 来源);任何引证进 claim 表或 #46 前须独立核验(Wu & Keogh TKDE 2023、Li et al. 2022 污染训练、SRR TMLR 2022、RoSAS 2023、TranAD/USAD、Deep SVDD、Tatbul NeurIPS 2018、TSB-UAD VLDB 2022、TSB-AD NeurIPS 2024 主线记忆可证为真实文献,gist 相符;PHM 2025、arXiv:2601.00005 暂不可核记 leads;"Yahoo ~86% 平凡可解"等具体数字按近似引用处理,核验入 #46 相关工作债)。**三范式表 = 契约 Consumer 分族的文献版**:无监督统计族(iforest/robust-z:尺度稳、轻污染可扛)/ 重构-一类族(AE/SVDD:需未污染正常轮廓 + 工况覆盖)/ 监督事件族(正例保留 + 标签几何对齐)。**与我们实证链互证**:identity 全局最优 + 四清洗全负 = "iforest 对污染稳健、对变形敏感"的文献预测吻合;5/12/7 逐序列异质 = "污染危害看结构非比例"吻合;契约 Consumer 分族(sol 改名裁定)与"没有第四种对大家都好的干净序列"同构。

**M0 文献背书预测(预注册,结果前登记)**:M0 第二 Consumer 若取重构/AE 族,清洗类程序预期**正向**——与 iforest 族的全负**反号** = 模型条件化翻转有文献先验背书,M0 配对设计(iforest vs AE 同任务同数据)从"探索"升级为"有向验证"。**pre-#42j 预期注记(结果前登记)**:本 cohort held-in 污染轻(14/24 有事件且事件稀)+ iforest 族稳健 → contamination_mask_refit_v1 的 headroom 可能同样薄;**若 #42j 不合格 = 文献一致结局**,弃权框架 #42g-c 依然良定义;capability 正故事更可能落在 M0 的 AE Consumer。#42j 照跑不撤:便宜、补全 Supply 线裁定、且 mask-refit 是 M0 配对的天然程序候选。

**Yahoo 病理 caveat 扩栏**:Wu & Keogh 四病(平凡可解 / 密度失真 / 错标 / run-to-failure 位置偏置)适用于 Yahoo S5;**我们的稀疏表(held-out 38 事件 vs held-in 窗 14)即位置偏置在本数据上的显形**——前段少事件部分是基准构造使然,非自然事实;所有 Yahoo 结论(含未来 #42g-c)附病理 caveat 与镜像 provenance caveat 并列。TSB 式资格门(in-context 信号、密度真实性、类型变异)进**未来**数据集选择流程;**41 条 sealed 禁预跑资格计算**(标签不可读),病理描述仅计分后补。**Skill 风险条款带 Consumer**:"Memory 跨任务劝退"failure-mode 命名采纳——Source Skill 写"操作有害"必须限定 Consumer 族,与契约分族、M1 反作弊互证。**标签几何注记**:现役 event-F1 零容差 + merge_events 点映射与文献警告相容;Tatbul range-based P/R 记 #46 相关工作债。**序不变**:#42i(已发)→ #42j → 合格 replay / 不合格弃权 #42g-c;41 条不动。

### 外部静态审查(pro 模型)裁定:引证核实;**C29 枯竭裁定收窄为"F1 读出族"**;#42j 重塑为 census+仪器研究;TaskEvaluationContract 采纳(缓冻结);观测扩展标签合法性红线(2026-08-24 10:4x,主线)

**引证核验(六处全实 + 两处加强)**:method.py:236-239 任务键解析失败静默回退 ("forecast","ridge","sMASE");experience_memory.py:76-77 任务硬键 task_type|downstream_model_class|metric.name + 回退常量(注::84-85 铸键函数对错误形状已 fail-loud,"静默造第四种方言比崩掉更贵"——fail-closed 学说仓内已有,reviewer 建议与之同宗);online_loop.py:583-584 默认 group card 硬编码 {"task_kind":"forecast"},**另有 :339/:354 两处 getattr 默认 "forecast"(主线加强发现)**;ad_scope_adapter.py 头注自证"报 -macro_f1 以沿用 lower-is-better 算术",**且内藏 CONTEXT_LENGTH=192/HORIZON=48(主线加强发现:forecasting 几何寄生在 AD 适配器内)**;per-series AUPRC 已录(runner + aegists_iforest_v1 + trainable v1–3);MATERIAL_THRESHOLD=0.005(runner:180)。**盲区注记**(其未见 contracts/tests/docs):其 Mod4/Stage A(冻结工件三件套 + 绑定断言 + NO_FROZEN_ADAPTATION_STATE)= C26 裁定 + #42g-b Part 0b 已落地;质量契约侧 sol v0r1 已定稿(其 TaskEvaluationContract 为**评价语义侧**,互补不重复);其 #42g 读法与 C26 逐条一致 = 独立收敛,增信。

**C29 修订(主线自我修正,载重)**:原裁定"无合法反馈机制可安全收割"**过宽**——#42h 四单元全属事件 F1 读出族;reviewer 量化台阶批评成立:少事件序列 event F1 最小跳变 ≈0.2–0.33,±0.005 逐序列材料线低于量化台阶(A2/#42h 的逐序列方向读数部分读的是量化噪声);连续代理信号(AUPRC / score margin / flag 率)**从未测过**且代码已录 AUPRC。**收窄为:"事件 F1 读出族反馈机制枯竭;连续 Support 信号未测"。** iforest × 现役菜单的 capability 正路径重新打开(Stage B→C);mask-refit 解锁不撤(便宜 + 文献支持的 data-centric refinement)。

**采纳与排程**:(1) **#42j 重塑 = 六程序 census + Support 信号仪器研究**一书:六程序 × 24 × 双区 event F1 + 每(序列,程序)held-in AUPRC/score margin/flag 率;mask-refit 安全门资格判;各候选信号的方向一致率 + policy 模拟(同安全门);**信号纪律钉:连续信号只作 Support 侧排序/起草,最终判官与晋升门保持 event-F1 + 安全门(宏/受害/worst),禁代理指标晋升**;fit ≤180、0 LLM;正文待 #42i 落地后发。(2) **pre-M0 卫生书**(#42j 后):TaskEvaluationContract v0(薄冻结对象:metric 方向/Support-delayed 信号/materiality/harm/聚合/晋升单元/保护信号/允许禁止族/弃权;**缓冻结——内容依赖 #42j 的信号与量化发现**)+ 三泄漏修复(online_loop:584 硬编码、:339/:354 默认值、method:239 解析回退)多任务模式 fail-closed + 测试。(3) **Stage C 条件采纳**:#42j 若发现可用信号 → EXPOSED 24 局部生命周期正控书(正组 → LOCAL_DRAFT → delayed → LOCAL_ACTIVE;负组不激活/撤权;其余 identity)= 首个多任务正结果;→ replay → #42g-c capability 框架。(4) Stage D 指标(coverage/abstention/首个有效局部 Skill 成本/LOCAL_ACTIVE 部署命中率)入 #42g-c 设计。(5) 三层迁移税则(程序性共享 / 机制共享+Task profile 解释 / 主动 TRY 默认 Consumer-specific / Target-local Outcome 不跨用)+ 权限层语言(PROCEDURAL_SHARED / CONSUMER_LOCAL / TARGET_LOCAL_ACTIVE)采纳为 A5/Stage E 设计语言,schema 实现押后。

**标签合法性红线(主线对 reviewer 的修正)**:其 AD 观测扩展清单混入标签依赖项——事件密度/事件时长、正常区间长度、"calibration 区间是否含异常"、"处理动作是否改变真实事件区域"在部署时**不可观测**,作 Observation 违反 label-safe 协议(M1 反作弊纪律适用);合法子集 = score margin、flag 率(阈值相对)、阈值校准稳定性、train/calibration/query 三区分布差异(label-free)、检出事件几何(来自分数非标签);标签依赖项只能作 post-hoc 协议事实报告。AD Observation 扩展按此红线设计。**序更新**:#42i(已发)→ #42j(重塑)→ 卫生书 → [信号可用 → Stage C 正控 → replay → #42g-c capability | 全哑 → M0 AE 主场 + #42g-c 弃权框架] → #43 M0(必做,与 AD 结局无关)→ #44 M1 → …;41 条不动。

### #42i 收口 = CODE_LANDED 采纳;fit-policy 绑定追认;provenance 缺口自报闭合;#42j 书发出(2026-08-24 10:5x,主线)

**CODE_LANDED @ 9983e5f 采纳**(11 文件 +3254/−13):契约 wiring 全落——anomaly_background_model_quality_contract_v1 + anomaly_task_context_v1,三新词表 token 同步至模块与双 JSON schema(task_quality_contract_v1.json / task_context_v1.json),anomaly_events 注释钉入,红线保持(零 Pattern→Program 字段),契约测试 15/15;mask 以 **consumer fit policy** 落地(fit_series_with_contamination_mask + consumer_id_for,窗口级遮罩 ≤1%,1 exec = 2 fits,常量字节不变,raw+Query 零触,16/16 测试);census 菜单参数化(PROGRAMS_V2 / FIT_POLICY_PROGRAMS / _program_fit_cost 1vs2 / --menu-size=N,默认 5,**menu-size=6 fixture-only,Yahoo 运行须 #42j 授权**;feedback_unit_v1 锁 5 保 U0 锚)。methods/ 与 operators/registry.py 零改;0 LLM / 0 fit(仅 fixture)/ 0 Yahoo / 41 未读;failure-not-laundered 清单在册(五处真实测试失败修复后重断言)。

**r1 fit-policy 绑定追认(优于主线原注册表方案)**:"对 fit 不可见"本义即 fit 侧行为而非底物变换;注册表纯度保持;窗口级匹配 iforest 几何;成本诚实。**两旗标**:(a) harness 可达性——若 #42j 资格通过,Agent 采纳路径(program 别名 vs consumer 配置)须另行合法接线,届时与 sol 定,现在不预建;(b) program-vs-consumer 表征张力(census 侧作第六"程序",Memory 键侧系 consumer 变体)留给 TaskEvaluationContract 设计。**provenance 自报追认**:aegists_iforest_v1.py 此前 untracked——#42g/b/b2/h 全部跑在未提交 consumer 上,缺口现已闭合;此前各 sha 不能重建 consumer 代码,工件数字仍有效但代码 provenance 带缺口 caveat。**偏差注记**:本次 Part 0 未含主线两 docs(执行方判前会话遗留、超范围)——可接受,#42j Part 0 必须补提交。

**#42j 书发出(见分发件)**:一书两用 = 六程序 census(mask fit-policy 资格判,安全门同前:宏>+0.005 / 受害≤2 / worst≥−0.02)+ Support 信号仪器研究(F1-pooled / AUPRC / score margin / flag-rate 四信号,方向一致率描述 + policy 模拟同安全门);**五程序 F1-pooled policy 锚 −0.006978 复现义务先行**;六程序新 oracle(Δ_oracle_6 + 分解);fit ≤200、0 LLM、EXPOSED 24 only;Stage C 解锁条件 = ≥1 信号 policy 模拟过安全门;信号纪律钉(连续信号仅 Support 侧,判官/晋升恒 event-F1+安全门)。

### 外部复审(pro,含 contracts/tests)裁定:五断言全核实;ConsumerFeedbackContract 采纳;Stage C 前置义务钉;M0 线协调事实首录(2026-08-24 14:5x,主线)

**核验(五项断言五项属实)**:(1) PreparationRequest.task_context 可选(contracts/method.py:48,校验仅在传入时 :57-70;注 :67-68 run_dependency_binding 已强制 context,严格钩子在);(2) fast_agent 全接线但 `if task_context is not None` 门控(:254/:576/:603/:692/:1084/:1119 等;maximum_candidates 与 max_modified_fraction 传入时真实生效);(3) **T6 runner task_context 零命中**——"合同已定义未进主路径"属实;(4) **树上 1 失败测试实跑坐实**:tests/runtime/test_candidate_verification.py::test_identity_and_valid_imputation… 期望 NaN 填补计 modified_fraction=1/3,runtime 报 0.0(modified_indices=(),填补被排除)——语义歧义非简单 bug;(5) **M0b 特征在树**(public_features.py:287-323:level_region_fraction / level_region_end_fraction / outlier_region_end_fraction / level_only_post_shift_support_sufficient,注释自证替代 union 读法)= **另一执行线(M0 线)已落地工作,主线台账首录此协调事实**;其 union 污染修复对 AD 线共享受益。

**时序校正(主线对复审)**:"合同未进主路径"当前是**预期状态**——#42i 系纯代码书,census 0 LLM 不走 PreparationRequest,合同落地后尚无行为实验。复审要求转译为 **Stage C 前置义务(硬门)**:(a) 行为书 runner 构造唯一 anomaly_task_context_v1,全 PreparationRequest 携带;(b) 严格模式 fail-closed(主实验缺 context → INVALID_TASK_CONTEXT,禁静默当 forecast);(c) inspect/propose/select/trace 同 TaskContext SHA 断言,三臂同 context;(d) correct/neutral/shuffled 合同诊断(M1 式仪器验证:合同须可观察地改变推理);(e) **H0 snapshot lock 重生成 + verify_lock=True**(复审报 mismatch/content-sha 一致 bundle 漂移,采信为待验,付费实验前不得绕过)。

**采纳清单**:(1) **ConsumerFeedbackContract**(取代 TaskEvaluationContract 方案;与 TaskQualityContract 分职 = 质量语义 vs 反馈/晋升语义):support/delayed 信号、materiality(Consumer-specific,废全局 0.005 复用)、decision/aggregation/promotion 单元、zero_event_policy(starvation 制度化)、uncertainty_rule;**内容缓冻结至 #42j 数字**。(2) 类型提升 PreparationAction = DataTransformProgram | ConsumerFitPolicy(解 #42i 旗标 b 表征张力;最小实现禁平台化)。(3) modified_fraction 三分(observed_value_modified / missing_value_filled / total_output_changed;DeploymentConstraintSpec 指明所 cap 字段,AD 事件保护 cap observed_value_modified;禁只改测试、禁把填补算有害)。(4) Evidence-provenance 权限原则(独立发现证据可扩权;Skill 在场自我诱导的确认证据可维护/反驳、不可自我扩权)用于 A5/Stage E;**其 outlier_iqr +6/−0(5 正例系 Skill 在场后产生)断言列验证义务,核实前不得入 claim**。(5) 红线细化:标签派生观测仅限已打开反馈区合法,未开区/held-out 永远非法。(6) Runner 拆四窄接口记债,AD 线收口后做(#42j 在飞不动 runner)。**卫生书范围定稿**:ConsumerFeedbackContract v0 + 类型提升 + 三泄漏 fail-closed + modified_fraction 三分 + H0 lock 重生成 + 旧设计文档 classification-first 表述修正。**序不变**:#42j(在飞)→ 卫生书 → Stage C(带前置义务)/ M0 路由;41 条不动。

### sol 排程审核全采纳:#42j r1 收紧;卫生书解散为最小接线修复;弃权四条件;replay 双钉;41 条主张帽;M0 三 Consumer;max_candidates 冲突证实(2026-08-24 15:2x,主线)

**主线两错自认**:(1) #42j 书预算矛盾——Part A 锚复现若重跑五程序 = 120 fits,加 Part B 168 = 288,超自定 200 帽;(2) 卫生书范围膨胀成多机制工程回合,违一轮一因果纪律。**sol 修正全采纳**。**max_candidates=1 冲突证实**(task.py:705,docstring :679 把"部署单程序"与"适应期探索 ≤2"混同;fast_agent :1084-1088 真消费该值)——修法:适应期 context maximum_candidates=2,held-out 部署本就单冻结 Workflow 无需候选帽。

**#42j r1 收紧(Part 0 = 2b6d520 已交,结果未产,来得及)**:单主判 = FIT_POLICY_QUALIFIED / NOT_QUALIFIED(mask-refit 是否有真实安全 headroom),一遍共 168 fits(旧五程序 24×5=120 + mask 24×2=48),U0 锚从同批拟合复算不重跑;四 Support 信号照常落盘但降为 development 诊断,不授资格、不自动冻结反馈合同;仅 AUPRC 预注册为未来 Support 候选(margin/flag-rate 正方向不天然明确,只作诊断);Stage C 解锁语言从书中移除,路由归主线。

**卫生书解散 → 最小多任务接线修复(全分支必做)**:T6 请求真正携带 anomaly_task_context_v1(含 max_candidates=2 适应期修正);非预测任务禁静默回退(method.py:239、online_loop:340/:584,group card 从当前 TaskSpec 生成);评分只绑冻结部署 + 无冻结不开 held-out(0b 已落,重断言)。**分支专属**:信号可用 → 仅接 ConsumerFeedbackContract(薄 dataclass,#42j 数字冻结)+ 原五程序局部生命周期正控;mask 合格 → 仅做 PreparationAction = DataTransformProgram | ConsumerFitPolicy 最小接线(验证 Agent 可合法提案执行 mask);双合格 → 先修 Feedback(菜单已证局部 oracle headroom),mask 后接;双无效 → 停 IForest 上 U4/U5/第七程序,转 M0。**modified_fraction 裁定**:现行语义系文档化意图("只统计被修改的已有观测值"),失败测试属过时——最小修 = 测试对齐文档语义 + docstring 澄清;三分 Receipt 记延迟债(未来需 cap 填补比例时再议)。**H0 lock**:仅在现役 Runtime 因漂移拒跑时机械重生成一次,不建验证体系;decision-bearing 运行禁静默绕过(既有站规)。**shuffled 合同诊断移出工程门**:归 M1 能力实验(卫生阶段只断言约束被执行)。**ad_scope_adapter 192/48**:先查是否在 T6 活路径,禁见常量就全仓清理。

**弃权四条件(采纳,收紧对策 B)**:弃权只有同时满足——检索并引用合法带 Scope 历史知识 / 确实改变行为 / 相比 A3 减少试错或 harm / 强制行动消融确实受伤——才是可归因积累优势;否则 identity 只是未行动,判"本 slice 无能力演示",故事落 M0。**replay 双钉**:同一 Outcome 不得重复记独立证据;已知正/负 subgroup 仅作机制正控,能力读数必须 Agent 在全 24 条自主形成 Scope。**41 条主张帽**:若用 EXPOSED 24 派生先验,只能主张"同域跨 series 积累",跨域迁移主张需 forecasting 线 Source 先验。**held-in < held-out 形态重申**(大无反馈底物 + 小可多轮反馈适应区 + 大零反馈验收区)。**M0 三 Consumer 设计**(AE/重构 + 监督事件分类 + IForest 自然参照;同数据同处理同窗口只换 Consumer;不预设 AE 必翻正——文献预测保持"已登记预测"身份,设计须可证伪);**mask-refit 系 Consumer-specific fit policy,不得承担"同一 Program 跨 Consumer 反号"主张**(该主张由 data-transform 程序承担)。AD 有界路线若最终无正向局部 Skill → 启动 classification 作为新增 Task slice(非项目改向)。

### #42j r1 收口 = C31:FIT_POLICY_NOT_QUALIFIED(三门全败,预登记预测命中);f1_pooled 诊断线索入库;路由 = 转 M0;#42k 直派 Opus 5;分发工作流变更首录(2026-08-24 15:3x,主线)

**C1 主判采纳**:contamination_mask_refit_v1 三门全败——eval 宏 +0.001375(< +0.005)、受害 8/24(> 2)、worst −0.167(< −0.02);不升 Operator。**pre-#42j 预期注记命中**(文献一致:轻污染 + iforest 稳健)。Δ_oracle_6 = +0.0478(vs 5 程序 +0.0375,差 +0.0103;mask 为 5/24 序列最优)——mask 有 series 条件化局部价值、无全局安全 headroom,与 hampel 同型(高方差算子)。预算 178/200(120+48+10,identity 重拟合 10 系信号 delta 自报);U0 锚第四次字节复现;旧五程序读数从 42g-b 工件加载 96 对字节一致零漂移;t6_42h 重生成侧效应 = 内容 byte-exact 仅时间戳/格式,采纳无碍。

**C2 诊断读法钉(不授权)**:f1_pooled 方向率 0.417 / policy +0.0059 / 0 伤 / worst −0.004——通过安全门算术,但系**结构性改进的功劳:零事件序列不授权 + 保守门**(#42h U0 全 24 授权计 −0.007,本轮仅 10 双侧序列可授权 → +0.0059);+0.0059 距门仅 +0.0009,处仪器分辨率边缘(单事件翻转 ≈0.008–0.014),方向率亚掷硬币——**不足以支撑 iforest Stage C 的 LLM 支出**。AUPRC(唯一预注册候选)+0.0031 未过;margin/flag-rate 永久诊断。**线索入库**:零事件排除 + 保守门 + 全策略晋升三件套进 ConsumerFeedbackContract 设计库,M0 的 AE Consumer 反馈协议直接受益。

**路由裁定**:形式状态 = mask 未过 + 无信号获授权 → **sol 双无效停止规则生效:不在 IForest 上造第七程序/U4,转 M0**。iforest 线收束为"经审计弃权候选",待 41 条终考设计(弃权四条件适用:须检索引用合法 Scope 知识、改变行为、省试错、消融受伤)。**#42k 最小接线修复书直派**(五件:Part 0 提交 #42j 交付 + t6_42h 侧效应 + docs;anomaly context 候选帽 v2[适应期 2];fail-closed 三处[method:239 畸形即 raise、online_loop:339/354 直读 task_type、:584 group card 从 TaskSpec 生成];T6 行为路径 context 携带;过时测试对齐文档语义,三分 Receipt 记延迟债;H0 lock 不动、拒跑才报)。**工作流变更首录**:用户授权主线直接分发子代理执行——难度低 → grok 4.6,难度高 → Opus 5,长任务同 session 复用缓存;#42k 系 methods/ 窄修有生命周期破坏风险 → Opus 5。执行方零 spawn 纪律不变。

### #42k 收口裁定:四修复落地零回归;INSTRUMENT_UNREADABLE 归因既有 H0 lock 债 → 授权一次机械重生成;t6_42h stub 覆盖 = 证据损失裁恢复 HEAD;B1 阻断修 runner 不放宽(2026-08-24 16:2x,主线)

**执行事实**:Part 0 = a26f1e7(#42j 交付 + docs,4 文件 +2120);代码 = c44dea0(8 文件 +184/−22):候选帽 v2(maximum_candidates=2,constraint_id anomaly-fixed-aegists-iforest-v2,全仓无 v1 残留)、B1 畸形键 raise INVALID_TASK_SCOPE(None 走文档化默认)、B2 两处直读 task_type、B3 group card task_kind 从 request.task_spec 闭包生成、Part C 三处 PreparationRequest 全携带单例 anomaly_task_context_v1 + SHA 断言 + 3 新测试、Part D 过时测试对齐文档语义 + 注释。全仓 pytest:改动后 645P/49F vs HEAD 基线 641P/50F,**新增回归 0,修复既有失败 1**;目标窄跑 80P。0 LLM / 零数据读取 / 41 未触 / methods/ 仅书内三处。

**判定裁定**:执行者报 INSTRUMENT_UNREADABLE(pytest 中 "snapshot lock mismatch" 100 次)——归因核实为**既有仪器债**(h0 目录相对 HEAD 干净,基线同败),非本书引入;sol 站规"仅在现役 Runtime 拒跑时机械重生成一次"触发条件已到,**授权重生成**(须验证 harness_content_sha 与旧 lock 一致 = 内容未变仅 bundle 漂移;重生成后判定翻 CODE_LANDED)。

**t6_42h 侧效应改判(证据损失)**:#42j 执行者"内容 byte-exact 仅时间戳/格式差"的自查**对 .md 不成立**——工作副本系 14 行 runner 自动 stub,HEAD(9983e5f)是 89 行完整报告(含 U0 锚、U1 塌缩自报、稀疏表、四单元对照);json 仅 1e-16 浮点重排,承重读数不变。#42k 执行者提交前 diff 复核抓住,未提交、未回滚,处置正确。**裁定:两文件恢复 HEAD 版本;站规新增——runner 重生成工件时禁止覆盖已提交报告路径(重跑输出须走 run-id 隔离路径),锚复现类操作产生的副本一律另存**。

**B1 阻断裁定:修 runner,不放宽闸门**。run_e2_fresh_confirmation.py(:1764-1769,:1858)与 run_e2_local_skill_recall.py(:784-792,:389)把 experience_memory 的 2 段 cell_key 灌进任务硬键字段,违 experience_memory.py:72-76"两键分立不可互换"——正是 fail-closed 要抓的方言病;修法 = 两处改用 task_consumer_key(task_spec) 铸 3 段键。在线主链(online_loop:165)不受影响。**书外发现处置**:online_loop:370 `_scope_now["task"]` 同型 getattr-forecast 隐患,授权按 B2 同法修;methods/h_ref_v02 未跟踪残留使三条架构测试红 + 另线 test_skill_revocation.py 语法错(Python 3.12+ f-string)——均属另一执行线遗留,主线不动,报用户协调;仓根 SelfEvolvingHarnessTS/ 系 Windows Junction(同一文件双路径)注记在案。**续派同一执行者 session(缓存复用):恢复 t6_42h、重生成 lock、修两 runner 键铸造、修 :370,全绿(除另线既有项)后翻 CODE_LANDED。**

### #42k-b 收口 = CODE_LANDED 追认;lock 漂移归因清白;树况 689P/14F/0E;揭出弃权错判族 = pre-M0 阻断项;#42l 诊断书续派(2026-08-24 17:3x,主线)

**CODE_LANDED 追认**(29bed7e lock 重生成 + 93a68ce 三文件 +20/−6)。**F2 lock 归因清白**:harness_content_sha 新旧同值(53b1c803…654f),漂移仅 8 个依赖 SHA,其中 3 个归因 #42k(candidate_verification 注释/task_contract/method)、4 个归因 #42i 及更早(双 schema/operator bundle/registry/fast_agent)——解释了 HEAD 即失配;符合"内容未变仅 bundle 漂移"的放行条件。**F3 亮点**:两 runner 改用 ssi 现役工厂 `_runtime_task_consumer_key`(不新造方言),实测 3 段键正确解析、旧 2 段键仍被 INVALID_TASK_SCOPE 拒绝(B1 未放宽),cohort 信息保留本职字段;F1 恢复经 blob sha 逐字节核验(CRLF/LF stat 假阳性识别正确);F4 :370 已修。**全仓 pytest:645P/49F/9E → 689P/14F/0E,零回归,44 项 lock 失败清零**;耗时 219s→2645s 属预期(58 项测试首次真正执行测试体)。

**剩余 14 项 worktree 隔离归因(零项归因 #42k/#42k-b)**:8 项系**此前被 lock 掩盖的既有缺陷首次可见**——其中 6 项同族(tests/methods/test_ttha_agent 5 项 + minipipe 1 项):`AgentProtocolError: stage_result names the wrong stage`,使本应 ABSTAINED 的路径错判 FAILED;另 2 项 test_f1_forecast_pilot / test_m0_release 断言待诊。3 项架构测试既有;1 项系 methods/h_ref_v02 未跟踪残留(另线);2 项系另线未跟踪测试文件。**主线裁定:弃权错判族 = pre-M0 阻断项**——AD 弃权终考与 M0 行为实验的读数都依赖 ABSTAINED/FAILED 正确区分,**#42l 诊断修复书续派**(同 session):定位 stage 信封错名根因,最小修复 + 零回归,协议语义有歧义即停报。**另线遗留报用户协调**:h_ref_v02 使 1 条架构测试红;test_skill_revocation.py 语法错使无 --ignore 的整仓收集中断;docs 台账文件存在 CRLF stat 假阳性注记。

### sol M0 计划审核全采纳:主张改口径;主/辅对比拆分;双独立判定;M1 拆 a/b;#43 M0-C 命名与两线边界;#42l 门改稳健口径(2026-08-24 18:4x,主线)

**四修全采纳**:(1) 三臂拟合协议不同(iforest 每序列无标签窗 20 / 监督 v3 有标签跨序列 pooled 窗 49 / PCA 每序列无标签窗 20),主张改为**"数据处理效用随 Consumer protocol 改变"**,禁称"只换模型结构";**主对比 = iforest vs PCA**(同 label-free,检验归纳偏置翻转),监督 v3 = 辅助对比(检验监督语义下事件证据必须保留)。(2) 监督 v3 禁逐序列拟合(单序列 held-in 常无正类):沿 T1b 方式,每程序 pooled fit 一次逐序列评分;预算 = iforest 120 + 监督 5 + PCA 120 = 245,帽 280;mask-refit 不再跑,引 #42j 侧道读数。(3) **双独立判定**:CONSUMER_UTILITY_FLIP_CONFIRMED(同程序材料级反号,主判限 iforest vs PCA 对)与 RECONSTRUCTION_HEADROOM_QUALIFIED(PCA ≥1 程序过安全门宏/受害/worst)拆开;仅后者通过才解锁 C-c 生命周期/M1;可能出现"翻转成立但无安全可行动 Workflow"。(4) **M1 拆两问**:M1a Consumer Context 因果(correct/neutral/shuffled → 提案按语义改变)先行,M1b 多轮生命周期 replay(Support→Draft→delayed→Active/撤权→freeze)在后;可共 Runner,禁合成一个判词。

**三问裁定采纳**:① C-c = 确定性窗口 PCA,最小定义冻结——窗 20 与 iforest 几何一致;标准化只用训练底物;full SVD 禁 randomized;固定 rank 禁 Yahoo 扫描;阈值只由训练重构残差定;Query 原始字节不处理;合成 fixture 先行验证(污染移动重构边界 / Query 零接触 / 确定性),不过即 INSTRUMENT_UNREADABLE,禁现场换 rank/阈值;语义只代表"确定性低秩重构族",禁外推 AE/TimesNet。② M0 = EXPOSED 24 上 0 LLM development 机制测量,不留封存确认,41 条零消耗。③ **Phase M 两线分工**:新实验命名 **#43 M0-C(consumer-flip)**,另线记 M0-Obs(a/b)(Observation 几何侧);M0-C 禁改 runtime/public_features.py、test_public_feature_calibration.py、run_e2_m0a_*、另线未跟踪 run_t233_supply_obs_ab.py;M0-C 只新增一个重构 Consumer、一个逻辑 Runner、一份主报告——两线零概念冲突零文件撞车。

**#42l 门改稳健口径**(另线工作树影响剩余失败数):弃"14F→≤6F"数字门,改为**"#42l 点名的失败全部转绿;无新增失败;Runtime/方法代码零非预期变化"**。**总序定稿**:#42l → #43 M0-C(PCA 仪器门 + 三 Consumer 响应矩阵)→ 翻转判定 + PCA actionability 判定 → M1a Context 因果 → M1b 多轮 replay → 41 条 sealed Static/A3/(合格时 A5)。M0-C 书主线已冻结备发,#42l 落地即派。

### #42l 收口:PROTOCOL_AMBIGUITY_STOP 采纳;"同族"前提更正(6 根因全测试侧)= 弃权语义无恙,pre-M0 阻断解除;#7 裁测试侧;#8 押考古;Chronos 缓存环境阻断;#42l-b 续派(2026-08-24 19:4x,主线)

**前提更正(主线自我修正)**:"6 项同族 stage 错名"判断错误——8 项实为 **6 个互不相同根因**,全部为被 lock 染红期间(2026-08-08 空池跳 select、2026-08-13 NaN 填补裁定、2026-08-19 2799d0f repair_level_shift 参数收权、actionability 探测)有意协议变更后**测试未同步**,零代码侧缺陷;"wrong stage" 系 canned 响应错位的次生症状。**载重结论:runtime ABSTAINED/FAILED 语义从未破——弃权错判威胁解除,M0 读数此轴可信。** 2799d0f 自述"51 failures before and after"与 lock 染红互证:**协议变更在红树期落地 = 测试同步盲区**,系 lock 卫生第二笔学费。

**6 修追认**(7254195,+107/−19,全测试侧,零校验放宽);**+10.0 幅度重接空心断言的自报处理追认为范例**(逐点实测算子灵敏度 + 引"无命中→恒等"文档契约 + 断言换 call_count 轴——非放宽,是把空心断言接回被测机制)。树况 694P/9F/0E,本书零回归(stash 隔离法证明);另线并发改动工作区(AGENTS.md/README/4 docs,1 项架构测试自行转绿)注记。

**#7 裁定 = 测试侧**:冻结发布工件系历史证据——姊妹测试明断 runtime_bundle_sha != EXPECTED 为正常,AGENTS.md §7 历史 SHA 保留不迁移;修法 = 断言对齐工件内记录的历史常量(自洽性),内容兼容性由姊妹测试的 content sha 断言继续承担;**冻结发布工件禁改写**。**#8 押考古后裁**:两读法均有档案支撑(SKILL_CONTENT_GAP 与 LOCALIZATION_PROCEDURE_GAP 系 fault_routes.json 并存合法类、LOCALIZATION 分支仍在),缺"翻转时点"——先 git 考古找出植入场景 cause_code 何时/因何提交翻转,再裁测试过时 vs 归类回归,防洗白。**Chronos 环境阻断**:FrozenModelUnavailable(chronos-bolt-small@772f3d25,local_files_only)——主线实查:**钉定 revision 快照目录在默认 HF 缓存中存在**,故非模型丢失,疑快照内文件缺损(同 session OMP libiomp5md 崩溃殃及)或加载路径差异;triage 入 #42l-b(先诊断快照完整性与精确报错;若确缺损,授权按钉定 revision 772f3d25 重取——模型权重系仪器非 outcome 数据)。**cycle.py:1521/1539 静默默认记债**(伪造 no_authorized_minimal_edit,与 getattr-forecast 同型;本轮两条经核系 fixture 真实返回)。repair_level_shift 宽度非单调灵敏度特征登记。**序:#42l-b(小书:#7 修 + #8 考古 + Chronos triage)→ #43 M0-C(书已冻结)。**

### 用户站规:验证经济学(2026-08-24 21:0x,用户裁定,主线记录)

**验证分级,关注承重点,不为每个小修跑全仓**:(1) 小修(测试对齐/单点修复类)只验目标测试与直接受影响模块;(2) 全仓 pytest 只在**相关修改批次结束后**统一跑,最多一两次,不逐书跑;(3) 涉及 production 代码(如 first-fault 路由)的修改仍须跑相关模块 + 下一实验的 smoke;(4) 简单执行类任务(Part 0 提交、文档提交、机械小修)派 grok 4.6 提速,复杂/协议承重任务仍走 Opus 5。适用即刻生效(#42l-b 已按旧规完成,不受影响);#7 目标测试过即可;#8 考古必须完成——若仅旧测试,局部验证即结;若需改 production 路由,相关模块 + M0-C smoke。

### #42l-b 收口 = CODE_LANDED;解释器伪影更正(有效树况 696P/7F/0E);#8 裁 fixture 侧修;两站规;#42l-c 派 grok、#43 M0-C 派 Opus(2026-08-24 21:1x,主线)

**CODE_LANDED 追认**(f612183,+25/−1):#7 按读法一修——断言对齐工件历史常量,且补 `f1_runtime_bundle_sha != historical_m0_runtime_bundle_sha`(更贴工件真实主张:"F1 重绑 runtime 而 authoring content 未动");冻结工件零改写(提交后复验);内容兼容性由姊妹断言继续承担未削弱。**解释器伪影更正(执行者自报)**:#42l 的全仓计数(694P/9F)与"Chronos 缓存失效"归因**作废**——网络中断重启丢 conda activate,pytest 跑在 base Anaconda(无 chronos-forecasting,torch/transformers 版本偏离 manifest);**有效数字 = 696P/7F/0E vs 基线 689P/14F,零回归**;Chronos 快照从未缺失(config 1121B + safetensors 182MB 俱在,HF 环境变量全净),模型未重取(授权前置"文件缺损"不成立,处置正确);OMP 冲突同源 base 环境。**站规二条**:(1) 每次 pytest/脚本运行打印并核验解释器路径;(2) chronos.py:140 宽 `except Exception` 包装记债——ImportError/ModuleNotFoundError 应透出原文,否则环境漂移一律误报成模型问题。

**#8 裁定 = fixture 侧修(保路由覆盖)**:考古采信——翻转提交 2799d0f(worktree 二分实跑:父提交 1 passed / 该提交 1 failed 与今日逐字同;规则侧 git -S 证 first_fault.py 自引入未动、fault_routes 未动、该提交文件清单零 feedback/router;产出侧 failure_patterns 证植入的错误区域仍在生效、localization 仍 miss,但候选在 intrinsic 语义下于该 fixture 退化 no-op → CANDIDATE_SUPPLY_GAP 抢先,归因链走不到 LOCALIZATION)。**裁定理由**:该测试的目的是覆盖 LOCALIZATION_PROCEDURE_GAP → bootstrap PATCH 这条**仍在役**的路由;对齐期望 = 静默丢覆盖。修法 = fixture 场景改为在 intrinsic 语义下产生真实修改(参照 #42l +10.0 幅度范例),使 localization miss 重归 first fault;production 零改动;按用户验证经济学局部验证即结。**2799d0f 教训入册**:红树期行为性变更有未记账影响面(提交信息只记 actionable pool 6→7,实际静默改变 minipipe first-fault 归属并断 3 处测试)。worktree prune(仅清失效元数据)透明记录。**派发**:#42l-c → grok 4.6(fixture 修 + 局部验证 + docs 提交);#43 M0-C → Opus 5 新 session(书已冻结:PCA 仪器门 + 三 Consumer 响应矩阵,fit≤280,0 LLM,验证按经济学限 fixture 门 + 自身矩阵 + 新模块单测)。

### #42l-c 收口:停报采纳;主线 fixture 前提再更正(LOCALIZATION 集成路由结构性死亡);裁 xfail 封存 + 记债;docs 已提交(2026-08-24 21:3x,主线)

**停报采纳,执行纪律好**(三档实测禁扫描、到不了即停、零期望改动零 production 改动)。**主线两处更正**:(1) 我"fixture 加幅可重接路由"的裁定前提错误——实测三档(1.5×/10×/20×)候选均可变 non-noop,但 36/36 条 LOCALIZATION 恒 NOT_APPLICABLE:2799d0f 后全仓无 external_region 算子,`program_requires_external_localization` 恒 False → cycle 关闭 localization 评估,**该路由在集成路径结构性死亡**,与幅度无关;(2) 前记"CANDIDATE_SUPPLY_GAP 抢先"系 stage/fault 层混记,本 fixture 实际 cause_code = SKILL_CONTENT_GAP(5)/OPERATOR_GAP(6)/RETRIEVAL_MISS(1)。**改裁**:该集成测试 xfail 封存(reason 注明 2799d0f + 本裁定;LOCALIZATION_PROCEDURE_GAP 单测覆盖仍在 test_first_fault.py 手工 CaseFacts);**记债**:LOCALIZATION 集成路由休眠——若未来任一算子重新声明 external_region,须重接路由 + 配 fixture;ContractPolicyBackend 对 SKILL_CONTENT_GAP 无授权分支(bootstrap patch 集成覆盖需 minipipe 线复工时重设计)。该债不承重 AD/M0 线,按用户经济学押后。T1 完成:docs 两件提交 2055c4d(真实 diff +37/−4 与 +42/−0,非 stat 假阳性)。局部验证 90P/1F(唯一失败即目标测试,待 xfail)。**#42l-d 落地**:872d57b,strict xfail 封存(L7 + L253-261,测试体零改),该文件 5P/1xf——#42 清障系列全闭,我们名下失败清零。

### #43 M0-C 收口 = C32:双阴性(FLIP_NOT_CONFIRMED + HEADROOM_NOT_QUALIFIED);文献先验证伪;条件化证据重定位;停止正效应探针裁定;监督侧"训练证据侵蚀"后验发现(2026-08-24 21:5x,主线)

**判定采纳**:C1 CONSUMER_UTILITY_FLIP_NOT_CONFIRMED——三 Consumer × 四清洗 12 读数**全部宏负**(矩阵最正格 −0.00306,无一侧 ≥ +0.005,翻转无从谈起);C2 RECONSTRUCTION_HEADROOM_NOT_QUALIFIED(PCA 最好程序 mad:−0.0418/8 伤/worst −0.60 三门全败);副 flag SUPERVISED_EVIDENCE_PRESERVATION_NOT_CONFIRMED(预注册 recall 口径未塌,三程序 recall Δ 恰 0)。**仪器可信度最强口径达成**:C-a 240 对读数与 #42g-b 工件逐位相同(max gap 0.0);245 fits 矩阵端到端三跑逐字节一致;三臂 identity 宏 F1 0.323/0.199/0.378 均非退化。预算 245/280、0 LLM、41 条零读取;代码 79ef5c0 + 555fab7(methods/ 零改动,禁改清单未开);报告未 commit(下书 Part 0 收)。预注册预测 3 中 1:iforest 负成立;**PCA"清洗正"文献先验证伪**;监督 recall 塌未发生——两落空原口径记录无事后重释。

**条件化证据重定位(主线裁定)**:总命题的 Consumer/Task 条件化证据**不死,换位**——(1) **任务级翻转已立**:同批清洗程序 forecasting 线获益(W47/W56)vs AD 三 Consumer 族一致受害 = Task 条件化的最强自然证据(T1b 受控版 + M0-C 自然版互证);(2) AD 内部条件化以**机制与幅度**而非符号显形:iforest = 边界变形→误报;监督 = **训练证据侵蚀**;PCA = 重构失真;(3) 监督侧后验发现载重:pooled 正例行 identity 369 → iqr 187 / mad 184——clipping 压平异常邻域后 robust-z 零尺度未定义,正例行**退出拟合而非被错标**,= 契约 event_erasure 危害的实证形态,标注后验、监督臂再用前须单独预注册确认。矩阵 96 格中 14 格 iforest 与 PCA 实质异侧(后验描述)。rank-3 fixture 方差 ≥0.90 vs Yahoo 中位 0.830(71.7% 低于门)诊断在案,未重扫(纪律守住);pooled 协议 roster 规模依赖与 real_14/18 零 eval 事件注记。

**停止正效应探针裁定(扩 sol 停止规则)**:#42j 已关"第七程序",本裁定关"第四 Consumer"——24 条 EXPOSED 在菜单(#42g-b/j)、反馈单元(#42h)、fit-policy(#42j)、三 Consumer 族(M0-C)四个维度全部给出一致答案:**identity 是本 cohort × 本菜单的就绪正解**。继续加维度找正 = 对 24 条过拟合。执行者"第四 Consumer 更便宜"建议记档不采。**caveat 保留**:本轮只关"确定性低秩重构使菜单转正"假设,不关 M0 概念,不许可"效用与 Consumer 无关"反向外推(三协议一 roster 一冻结菜单不构成该空间)。**路由 = 预冻结的弃权分支生效**:M1a 合同因果 → M1b 多轮生命周期 replay(弃权收敛 + 强制行动消融受伤)→ 41 条终考(Static/A3/A5,弃权四条件)。M1a 判读因全负矩阵而更锐:每格"正确答案"已知(任何清洗提案 = 有害),合同读取的行为效应可逐格对答案。

### 用户文档 Data_Quality_Disgussion.md 审读:三项设计输入采纳(2026-08-24 22:5x,主线)

审读用户桌面导出会话(约 2600 行,T1b 史 + 代码/数据盘点 + 文献测绘)。大部分与主线两日裁定独立收敛(identity 最安全 / mask-refit 属抗污染细支线且轻污染下增益薄 / 清洗与 AD 目标函数反向 / 聚类不做)。**三项超出典章,采纳为设计输入**:(1) **#46 related-work 定位骨架**——AD"数据质量"文献三分:训练污染线(Li/SRR/CLEANet,从 fit 丢样本)/ 标签基准线(Wu&Keogh/TSB)/ **"可执行数据准备→冻结检测器变好"≈ 空白**,"Agent 按 Task/Consumer 优化训练底物"几乎无人做 = 本项目独特性主张候选;引证(Impute4TSC 2025、label-guided imputation 2025、CLEANet/TSAD-C/WIRACAD 等)挂核验债。(2) **分类切片设计钉**:仓内已有双分类契约(global_coarse vs local_event)→ winsorize 条件化可在**分类任务内部**重问(M0-C 未测出的翻转以更好设计重试:密标签无事件饥饿、菜单合法、契约现成);选数据须有可读质量缺陷(禁拿干净 z-norm UCR 重演 identity 困境);Impute4TSC"插补 × 分类器匹配"线 = 天然正控场。(3) **注入式正控仪器**(Li/SRR 式):held-in 注入已知污染 → 仅 fit 去污染 → Query 不动 → 先证 oracle 恢复再考 Harness 发现;定位 development 正控,永不入自然 Yahoo 能力声明——回应用户"加噪构建数据"之问的合规版本。**谨慎注记**:文档中部规划批注系历史地层(#42d-f 时代),已被后续裁定吸收,禁复活过时排程;全部文献条目按设计输入待遇,进 claim 前核验。**合流裁定**:AD 线收官考"正确地不动"(M1a/M1b/41 条),分类切片接棒"该动时会动并变好"(任务内双契约翻转 + 缺测正控)——两线合成总命题完整证明。

### sol 文档审读采纳:路线补"反馈有效性门";弃权非唯一故事;Memory 四类经验钉;#44a 反馈正控设计;短正典文档建立(2026-08-24 23:0x,主线)

**sol 三修正全采纳**:(1) **路线缺口(主线自认)**——原 M1a→M1b→41 条序列跳过"AD held-in 学习反馈是否有效"一关:M1a 只证合同致谨慎、M1b 只证失败经验致少做,均不证 Harness 会优化数据;F1 反馈单元已证无安全合格者(#42h)、f1_pooled 仅边缘线索未授权(#42j)——**最终 Judge 有了,AD 的 held-in 学习反馈还没验证好**。修复 = 注入式反馈正控(#44a)插在 M1b 与终考之前。(2) **弃权非唯一故事**:正确弃权 = Risk/安全能力,不能替代正向能力;完整 Harness 须同时证"不该动会停 + 真有可修缺陷时能发现、执行、成 Skill";**Memory 合法内容 = Positive + Negative + Conflict + Abstain 四类,缺正例则系统学成劝退器**(M1b/Experience 设计钉)。(3) 文献段引用审计义务重申(具体比例/阈值/论文结论逐篇核验后方可入典)。**canonical claim-cap 措辞定稿**:"在 Yahoo 已曝光 24 条 × 现三 Consumer × 现五程序下,无全局安全处理;存在局部 headroom,但现役 Observation 与反馈无法安全收割"——禁说 Yahoo 完美/AD 无空间/Harness 不适 AD/Slow 未归因好/堆失败经验即可解。

**#44a AD 反馈正控设计钉(sol 版采纳,最小纵向切片)**:固定一个 AD Consumer → 仅向 held-in 训练底物注入**一种**已知污染 → identity vs 一个已知 exact/mask repair → Query 原样 → **先确定性证明 repair 的 delayed event-F1 真升** → 再查哪种早期 Support 信号能预测该提升(记录:有事件区 AUPRC/事件排序、事件 recall、正常背景误报率、事件数与不确定性)。规则:零事件窗只供误报伤害证据不得批正向;Support 只成 Draft;独立 held-in delayed event-F1 方可批准;held-out 永远零反馈。**分流树**:repair 无正效应 → Program/Consumer 层;repair 有效 Support 预测不了 → 反馈协议层;Support 可读 Agent 选不到 → Observation/Scope/selection 层;Agent 找到并过 delayed → 才进 Experience/M1b 与 sealed 验收。底物候选:T1 注入基建(12 NOAA 站 52 npy,注入事件已知可读 oracle F1≈0.7)优先于向 Yahoo 24 再注入(其 held-in 事件稀,delayed 效应难测),执行者提案、sol 审。**总序修订**:#44a(确定性,0 LLM)→ #44b 合同/Context 因果考(correct/neutral/shuffled,不与生命周期混刀)→ M1b Agent 双侧多轮 replay(正控场该动会动 + Yahoo development 场该停会停;Memory 能表达并保留四类证据但禁为填类别强造结果)→ 41 条终考。**文档定位**:Data_Quality_Disgussion.md = 研究讨论档案非路线权威;短正典 docs/DATA_QUALITY_AND_FEEDBACK_MODEL.md 建立(五节:Consumer 相对定义/各任务质量语义/三层反馈/证据支持与不支持/下一项);AGENTS.md 状态已更新到 C32/#44a。

**sol 复审边界补丁**:#44a 先分开验证两项必要对照——污染相对 clean reference 确实造成 delayed 伤害、repair 相对 contaminated identity 确实恢复效用;clean 未经读数不得预称 upper bound。注入位置/clean reference 只归 evaluator,不得进入 Agent Observation 或 Support 特征;两个冻结污染率逐率完整报告,不得择优率授权。#44b 不把合同扰动与双场生命周期合并,避免同时改变合同与数据 Context 后无法归因。四类 Memory 是表示/保留能力与机制覆盖义务,不是每次自然运行的配额。

### #44a 收口 = C33:PROGRAM_CONSUMER_LAYER_FAULT,细定位 = 判官饱和(远距几何);新 scope 条件入册;主线更正 Yahoo 注入判断;#44a-r2 发出(2026-08-24 23:5x,主线)

**判定采纳,归因细化**:两率 B1 均 EFFECT_NOT_CONFIRMED,B2 按规未跑。但非注入失败——注入进拟合(尺度抬高 1.02x/1.13x,尖峰 4.5σ)、oracle 遮罩还原拟合(0.1% 内)、全程确定性。**真 first fault = 判官饱和**:iforest fit [120,900) 计分 Qcal [2600,3060)(跨约 1700 小时季节漂移),clean 臂即对 Qcal 中位 88% 点报警(11/12 站 ≥50%,一站 100%),merge_events 塌段后 event-F1 由旗标断点决定;逐站噪声 ±0.17~0.37 = 效应(±0.03)十倍。**r05 伤害 −0.0331 可读、修复找回 26%(宏 +0.0087 过宏门,安全两门败);r15 三倍剂量零伤害(+0.0005)且 6/12 站宏 F1 反升(尺度膨胀→报警减少→精度偶升)——剂量-反应非单调 = 噪声主导估计量,饱和态 AD 效用读数一律不可信**。执行者自查注入前缀排序缺陷(靠 1/120 反常自抓,修正后 r05 由 +0.016 翻 −0.033,r15 逐位不变)——范例级自查入册。预算 74/100;Qf 未开;T1 冻结拷贝零触;Part 0 = d69bce1(M0-C 工件 + docs 三件),代码 3e5d777。

**三项入册**:(1) **新 scope 条件**:aegists_iforest_v1 的效用读数仅在"eval 区邻接训练区"几何有效(Yahoo 属此);远距跨漂移几何饱和——附着于既有全部 iforest 读数。(2) NaN 弃权适配(执行者按 T0/v3 正典补在适配器层,未改役中 Consumer)追认;记债:任何非 Yahoo 真实数据使用前升为 Consumer 一等特性并配门。(3) "重议 Consumer 选择而非注入"的教训采纳,但**不违停止探针令**:停止令针对 Yahoo 24 的 headroom 探针;正控场的判官选择是仪器资格问题,不是能力探针。

**主线自我更正 + #44a-r2 发出**:先前否 Yahoo 注入的理由("held-in 事件稀")仅适用 Support 窗,不适用 eval 读出——**r2 = 正控场搬回 Yahoo 几何**:EXPOSED 24 held-in 注入已知点尖峰(per-series 6×MAD,率 {1%,3%},seed 固定,拷贝走 _scratch run-id,原件零触),三臂 natural / +inject / +inject+oracle-mask(复用 mask fit-policy 机械),读出 = development_exposed_eval 真实事件 F1(配对消真实异常底噪);邻接几何 + 已验 Consumer + 同终考判官;B1 同安全门;B2(若过)Support 信号用注入已知位置 + 反馈窗真标签(事件饥饿因注入自解);非饱和预门(clean 臂旗标率中位 <30% 断言)。fit ≤180、0 LLM。**#44b 设计随之改良**:双侧考可单场化——同 Yahoo 域内"注入序列该动 / 干净序列该停",同一 Consumer 同一契约,纯底物条件化对照,科学上更紧。

### 主线重定向裁定(用户 + sol,主线自认偏移):正向迁移 = 主线,弃权 = 约束与副读数;#44a 升格为 AD 最后资格门并补法定程序臂;M1a/M1b/41 终考冻结待门;分类切片启动设计(2026-08-25 00:1x,主线)

**偏移自认**:M0-C 双阴性后,主线把"弃权结局"从保险升格为实际路线(M1a/M1b/41 终考全按弃权框架预冻结)——用户质询成立,sol 裁定采纳:**安全是正向能力的约束(有效适配收益 + harm 不超限 + 不适用时 abstain),不是主线替代品**;"没有有效适配 → 把 abstain 升格为主要成果"不可承担有效迁移主张。AD 负结果保留为边界案例;弃权保留为副能力。

**#44a 升格为 AD 最后资格门(sol 三链条件采纳)**:必须依次证明 (i) 注入缺陷伤害真实 delayed utility;(ii) **Agent 可执行的合法 Workflow**(非 oracle)能恢复 utility——oracle-mask 仅证机制,evaluator 独知位置的修复不算,缺法定动作仍 = Program Supply gap;(iii) held-in Support 预测独立 delayed 恢复方向。**在跑的 #44a-r2 收口时补法定程序臂**(iqr/mad 于注入底物,≈96 fits 附录;预注册预测:点尖峰注入正是此二算子的机制对题——若转正,现役菜单在"真污染场"即有法定正解,无需第七算子)。**硬分流**:三链全过 → AD 保留为正向能力切片(注入开发场上做 Target-local Skill → A5/A3);oracle 有效而法定无效 → 只许设计**一个**机制不同的 AD-native Workflow,禁批量扩菜单;效应/反馈不过 → **暂停当前 AD family**,分类切片接主线。**M1a/M1b 与 Yahoo-41 终考冻结待门**:缺正向 Program headroom 时不得继续(弃权主线支出停止);41 条继续封存。

**主线恢复图(sol 版采纳)**:核心 = **A5 vs A3 正向迁移**——多 Source 产 Positive/Negative/Conflict Episode → Slow 整合为带 Context 边界的 Skill → 新 Target 上 A3(空)vs A5(带审计 Skill)同预算对决;承重读数 = 首个有效 Target-local Skill 试错成本 / LLM+fit 消耗 / delayed utility / harm / held-out 终态(预测线 FRESH_A5_DELIVERS 69 vs 123 为既有正证)。**分类切片启动设计**(与 #44a 收口并行,设计零支出):一个 Consumer × 一个可读缺陷 × identity + 1–2 个 Workflow;先证 Workflow 真改善下游,再进 Harness;资格门 = 有可读质量缺陷(禁干净 z-norm UCR);候选形态 = 任务内双契约翻转 + Impute4TSC 式缺测正控。**扩算子纪律重申**:找有效算子是对的(无 headroom 时修 Program Supply 属正当),错的是在已曝光 24 条上迭代加算子;正确路径 = 冻结缺陷 → 机制匹配族 → identity+至多两候选 → 无 Agent 确认 headroom → Support 可预测 → Harness 学习 → 未参与选择的数据验收。

### #44a-r2 收口 = C34:INVERTED_EFFECT(注入抬分;机制 = contamination 分位阈值,已直接测量);**既有 AD 读数获竞争解释**;阈值审计探针提案;今夜停(2026-08-25 00:2x,主线)

**判定采纳**:PROGRAM_CONSUMER_LAYER_FAULT_CONFIRMED + 副判 INVERTED_EFFECT_OBSERVED;执行者**拒绝沿用该分支预注册理由**("iforest 读不出污染"被证伪——Consumer 读得极好,只是反号响应)的处理追认为范例(分支名成立、理由证伪必须自报,禁静默复用)。仪器全绿:非饱和预门过(eval 旗标率中位 0.134 vs NOAA 0.877)、natural 臂与 #42g-b 锚 48/48 逐位同、四次完整执行读数全同、eval 零注入零处理、24 份 work CSV SHA 不变、预算 130/180、两率全报。

**核心事实**:注入 6×MAD 尖峰使 eval 宏 F1 0.3227→0.4467(1%)/0.5694(3%),剂量单调;oracle 遮罩还原(0.3252/0.3259)。**机制已直接测量非推断**:masked 与 injected 臂共享标准化常数,森林 offset_ 隔离拟合矩阵贡献——injected 臂阈值 23/24 序列更严(+0.0369→+0.0486 随剂量),遮罩后回 natural 0.005 内;`IsolationForest(contamination=0.1)` 把阈值定为训练分数 10% 分位,训练块近净时该分位落在正常窗之间 → Query 撕碎(natural 臂 1.6 真事件报 10.7,精度 0.2525/召回 0.8347)。注入非"好数据":injected 臂召回反降(0.83→0.72),得分升纯因误报塌。**在此 Consumer 上,任何改变训练离群率的训练侧操作首先是阈值旋钮,其次才关数据质量。**

**竞争解释入册(载重,决策相关)**:M0-C 五程序全部**移除**训练离群点、四清洗全负;本书**添加**离群点、读正且剂量单调——符号恰为阈值机制所预测。"五清洗对该 Consumer 有害" vs "五清洗把失准阈值推向坏方向"**观测上不可区分**,仅后者关乎数据质量。波及范围:本线 Yahoo cohort 全部 AD 效用读数(#42g-b/#42h/#42j/M0-C)均透过该失准取得;**PCA 臂同类伪影候选**(其阈值 = 训练重构残差分位,清洗压残差散布 → 阈紧 → 误报升,同向机制);**监督臂不受波及**(正例行侵蚀 369→184 系独立测得的数据侧机制,任务级条件化主张经此臂仍立)。M0-C 数字不被推翻(逐位可复现),被质疑的是**它们测的是什么**;"identity = 本 cohort 就绪正解"的停止裁定获竞争解释——可能仍成立,但须审计后重述。`contamination=0.1` = 无人审计的自由参数,杠杆(+0.247)三倍于全菜单最大效应(−0.092)。

**阈值审计探针提案(待 sol/用户,今夜不派)**:对已有响应矩阵做**阈值无关重读**——AUPRC/排序质量读数(只需分数不需阈值)重算 iforest 与 PCA 两臂五程序 × 24(≤240 fits,确定性 0 LLM):若阈值无关读数下清洗转中性 → M0-C 负读数系阈值伪影,claim 重写("清洗移动失准阈值",方法论贡献);若仍负 → 数据质量伤害独立成立,原 claim 保留并加固。**定位 = 证据完整性审计**(审计判官,非探 headroom),不违停止探针令与主线重定向;它决定 #46 的 AD 证据链怎么写,优先级最高。**书外发现入册**:役中 mask fit-policy 不修标准化(常数取自整块);r03 遮罩达训练窗 48%(点污染需点级修复原语,窗级遮罩钝);real_1/real_28 在邻接几何下仍饱和(旗标率至 0.984);Consumer 失准(精度 0.25)系本线 Yahoo 几何上的普适事实,比 #44a 饱和发现更重。代码 d7b3b1e;Part 0 = ae8821b;r2 报告未 commit(下书 Part 0 收)。**今夜停机**:审计探针 + 分流重裁 + 分类切片设计,明早连同 C34 一并交 sol。

### sol 七点主线方向采纳定序;三线并发派发(2026-08-25 00:5x,主线)

**七点全采纳**:(1) AD family 收口冻结——只许最小 AUPRC/阈值混杂证据审计;禁弃权主线复活、禁扩菜单、禁读 Yahoo 41;(2) 下一步 = 本地已曝光 Forecasting 数据上当前代码形态的 **A5 vs A3 development replay**(held-in 总预算内多轮;freeze 后 held-out Fast-only 一次计分;验证 Source Skill → Target 校准 → Target-local Skill → frozen deployment 完整正向链;禁下载/重选数据/扩仪器);(3) replay 过后启动 Classification 正向切片,先 Grok 限时调研 + 只盘点既有 40 UCR zip 与 controlled classification 资产,不下载;(4) 分类先无 Agent 资格门(一 Consumer × 一缺陷 × identity+≤2 Workflow;须证合法 Program headroom + Support→delayed 方向可读);(5) 资格过后才 Source Episode→Skill→Target A5/A3 同预算→freeze→held-out(主读数:首正成本/LLM+fit/delayed utility/harm/held-out 终态);(6) 安全弃权仅约束;禁新 SHA 体系/复杂 Gate/Schema/Runner 平台/全仓清债;(7) 三个用户检查点 = Forecasting 主链无法复现 / Classification 无合法 headroom / 需换 Task-Consumer-缺陷 family 时停报,其余主线自主。**模型路由(用户令)**:多数任务 grok 4.6 fast 提速,攻坚才 Opus;遇问题可用 subagent 调研 Agent 文件夹内其他任务做法与领域论文。

**三线并发派发**:(A) **#44-audit AD 证据审计**(grok):iforest+PCA 两臂五程序 × 24 阈值无关重读(eval 区 AUPRC/平均精度,只需分数不需阈值),≤280 fits、0 LLM;判定 THRESHOLD_ARTIFACT_CONFIRMED(清洗 AUPRC 宏 Δ 全落 ±0.005 内)/ DATA_HARM_CONFIRMED(仍 ≤−0.005)/ MIXED / INSTRUMENT_UNREADABLE;监督臂不重跑(机制独立)。(B) **#45-Frep Forecasting A5/A3 开发 replay**(Opus):复刻 FRESH_A5_DELIVERS 配置于已曝光开发数据 + 当前 HEAD;A3 = 空 Source h0,A5 = +冻结 Source Skill,同预算;判定 CHAIN_REPRODUCED / CHAIN_BROKEN(stage)(= 用户检查点①停报)/ A5_NO_ADVANTAGE / INSTRUMENT_UNREADABLE。(C) **CLS-survey 分类调研**(grok,限时零仓写,提前并行——纯研究不违序):分类 TS 数据质量/反馈信号文献 + 40 UCR zip 名录级盘点(禁开内容)。

### CLS-survey 收口采纳;分类资格门书 CLS-1 预冻结(待 #45-Frep 链通即派)(2026-08-25 01:1x,主线)

**调研采纳(零仓写,凭证分档规范)**:40 UCR zip 中仅 DodgerLoopWeekend 官方含 NaN(tsml 名单交叉验证);KeplerLightCurves 已官方插补不可当缺测集;现役 loader 拒非有限值且逐行再 z-norm(run_e2_task_context_label_evidence_witness.py:54-70)。**分类活 Consumer = ridge-raw-plus-difference-v1(runner 内联);rocket_ridge 仅 TaskSpec 字符串无实现**——survey 纠正主线此前认知,ROCKET 留作过门后第二 Consumer(M0 条件化用),不当第一刀。注册表插补算子 7 个(impute_linear/fft/ema、period_complete/median_complete、impute_ssm[hard_fail]、impute_ar);fast_agent._MISSING_ONLY_OPS 门控:**Observation 无缺测信号(coverage==1 且 max_missing_run==0)时插补算子供应前即被跳过——注入后必须核验 Observation 出现缺测信号,否则 Fast 结构性不可能提案插补**(载重接线事实)。文献:Rhodes ICMLA 2025 证"插补 × 分类器条件化"(RMSE 最优 ≠ accuracy 最优;RF 与 kNN 排名不同;线性插值对 RF 伤害更大)= 分类切片版 Consumer 条件化的文献先例;Impute4TSC gist 确认、细节未查到(核验债);分类线既有注入先例全是 impulse 族,缺测注入系新仪器。分类无事件饥饿,但小 n = 量化饥饿(Coffee n=28 一步 0.036),材料线须 n 自适应(≥1/n 预注册),禁抄 AD 0.005。

**CLS-1 资格门书预冻结(sol 点 4 形状;#45-Frep CHAIN 通过后即派,不通过则按检查点①停)**:Consumer = ridge-raw-plus-difference-v1 + classification_global_coarse 契约(缺测系可修损伤非保护事件;local_event+erasure_guard 留尖峰族);底物 = ECG200 或 GunPoint(现役 loader 已接,中等 n,平衡二分类;执行者按 n 与类平衡二选一并陈述);缺陷 = held-in 训练行单机制 MCAR 缺测注入(固定 seed,率 15% 单率起步,点缺 + 短段两形态中预注册一种),Query/官方 TEST 零挖零插(项目墙,异于文献 train+test 都插);三臂 = clean reference / injected identity / injected+impute;菜单 = identity + impute_linear + impute_ema(禁 hampel/winsorize/znorm/ssm);判定 B1 = 注入伤害 delayed accuracy(预注册伤害门 Δacc ≤ −0.05)且合法插补显著恢复;B2 = Support(held-in 子集 accuracy,只起草)方向预测独立 delayed;义务 = 注入后 Observation 缺测信号核验(coverage<1);定位 development 正控非自然 UCR 能力;0 LLM 确定性。三失败模式与缓解入册(仪器/Query 墙、无 headroom 2.0、Support 不预测)。

### #44-audit 收口 = C35:iforest DATA_HARM_CONFIRMED(清洗伤害扛过阈值审计,主证加固);PCA MIXED(部分伪影);claim 重写定稿(2026-08-25 01:2x,主线)

**判定采纳**:C-a iforest = **DATA_HARM_CONFIRMED**——四清洗 AUPRC 宏 Δ 全 ≤ −0.005 且受害 >2/24,排序质量真受损,M0-C 的 F1 全负**不能**归因于 contamination 阈值位移;C-c PCA = **MIXED**——mad/winsorize 落 ±0.005 伪影带,hampel 真伤(−0.027/3 受害),iqr 弱负(−0.0054/1)。监督 v3 未重跑(机制独立)。**仪器**:240/280 fits,companion F1 与 M0-C 落地工件 480/480 逐位同(审计仪器与原仪器同源性证明);两跑 BITWISE_IDENTICAL;新锚 = identity AUPRC iforest 0.451 / PCA 0.576(22 有事件序列;real_14/18 单列);AUPRC-F1 符号一致率 iforest 0.602 / PCA 0.489(PCA 的分叉正是阈值旋钮 vs 排序质量的可视化)。Part 0 = 085ba79。

**claim 重写定稿(#46 用)**:(1) "清洗伤害 AD 背景族(iforest)"**加固**——在事件 F1 与阈值无关 AUPRC 双口径下均成立,且经受了 C34 竞争解释的正面挑战;(2) M0-C 矩阵加脚注:PCA 列的负读数部分系阈值伪影(mad/winsorize),hampel 独立真伤;"12/12 全负"叙述改为"iforest 4/4 真伤 + 监督 4/4(机制=证据侵蚀)+ PCA 1 真伤 2 伪影 1 弱";(3) 任务级条件化主张(forecast 益 vs AD 害)**完整存活**(iforest 数据伤害 + 监督证据侵蚀均阈值无关);(4) C34 的失准发现与本审计**并立为真**:判官失准(contamination=0.1,杠杆 +0.247)与清洗真伤(AUPRC 全负)同时成立,互不推翻——前者是仪器学发现(修判官属 Consumer 重规格,超就绪范围),后者是数据质量事实;(5) "identity = 本 cohort 就绪正解"恢复无竞争解释状态。AD family 按 sol 点 1 继续冻结;审计闭环,无后续 AD 支出。

### #45-Frep 收口 = C36:CHAIN_REPRODUCED(主链在 HEAD 存活,sol 点 2 门过);held-out 对比裁无效(F1 违规 + F2 结构偏置);首正成本优势存活;CLS-1 发车(2026-08-25 01:3x,主线)

**判定采纳:CHAIN_REPRODUCED**——Source Skill(卡重编译 SHA 逐字节同)→ held-in 校准(2 轮+探针)→ Target-local Skill(3 起草/3 批准/3 晋升,0 撤权)→ freeze(四臂快照字节稳定)→ held-out Fast-only(部署绑定断言 4/4,scored==applied)在 #42i/k/l 手术后的 HEAD 上完整走通。预算 13/40 LLM、528 重训、零下载零封存读取;0-LLM 预检与计分账本分列;单次计分零重掷。**A5/A3 读数裁定**:首正成本优势**存活且为本轮唯一 claim 级读数**(pooled 84 vs 123,−31.7%,原证 69 vs 123 同向;LLM 非确定性动幅度不动方向[F5]);**held-out 终态对比(A5 弃权 0.0 vs A3 +0.1525)裁定无效**——F1:继承的 stage-4 部署梯队在 held-out 块上开 delayed 读数作采纳门(a3_pooled 取 bar/confirmation 两读),违 AGENTS.md §3,系原 runner 既有缺陷原样继承;F2:部署成本按构造不对称(有冻结技能臂付 9-15 重训召回,无技能臂在**计分块上**白得全价搜索 69 重训+2 LLM)——结构性偏向 A3,该对比不测积累价值。执行者按书面字义打 CHAIN_REPRODUCED、把原预注册条款读数(FRESH_A5_FAILS)如实放报告头而不自造定义——处理追认。

**债三笔**:(1) **原 FRESH_A5_DELIVERS 考古注记义务**——其 held-out 数字是否同经违规 delayed 门取得,#46 前须查明并注记(首正成本主张 69 vs 123 系 held-in 侧,不受波及);(2) F2 部署协议缺陷:未来任何 A5/A3 终态对比前必须修部署对称性(无技能臂不得在计分块搜索);(3) F4:该路径 compile_snapshot(verify_lock=False) 绕锁,lock 已在 #42k-b 修复,此路径应回 True(小修,记债)。F3(19 冻结文件中 5 个已移动,兼容)注记。**过夜三线合并结论**:CLS-survey 定资格门设计 + #44-audit 定 AD claim 措辞(iforest 真伤加固/PCA 脚注)+ #45-Frep 定主链健康(活,带三笔债)——**sol 七点序的门全部满足,CLS-1 分类资格门书发车(grok,0 LLM 确定性)**。

### CLS-1 收口 = INSTRUMENT_UNREADABLE(主线书面设计缺陷自认);CLS-1-r2 改注入形态续派(2026-08-25 01:4x,主线)

**判定采纳,归因 = 主线设计缺陷**:书面「各行 15% 点 MCAR」×「identity=丢含 NaN 训练行」结构碰撞——L=150 时 P(整行完整)=0.85^150≈2.6e-11,identity fit 区 35/35 全丢,B1 无定义;非用户检查点②。执行侧干净:Observation 缺测信号成立(coverage 0.853/max_run 5,_MISSING_ONLY_OPS 不会跳过插补——结构前提已验);clean delayed acc 0.820;两 impute 臂可拟合(0.793);两跑 BITWISE_IDENTICAL;TEST/zip SHA 不变;fit 3/50;底物选 GunPoint(理由在册)。**预警入册**:插补后 vs clean 仅 −0.0267(< 0.05 伤害门)——散点 MCAR + 平滑序列 + 线性插补太易恢复,注入形态须换。Part 0 = 03b07bf(过夜交付 6 文件;#45-Frep runner 未在 allowlist 仍未跟踪,记下书收)。

**CLS-1-r2 改形态(续派)**:注入改**行子集 + 连续段**——held-in 50% 行受灾(seed 固定),灾行内 2 段连续缺测 × 10–15 点(≈13–20% 行长);identity=丢灾行(余 50% 完好行可拟合,伤害通道 = 训练数据损失);连续段更伤 raw‖diff 特征且区分 linear vs EMA。门不变(伤害 ≤ −0.05;恢复 ≥50% 且类 recall 不恶化 >0.05;Support 方向可读)。**底物阶梯预注册**:GunPoint 先行;若 INJURY_NOT_READABLE 则 ECG200 一次(唯一换底物机会,非扫描);再不可读 → 停报(缺陷 family 需与 sol 重议,接近检查点②性质)。其余(seed/Support 切分/TEST 零触/fit 帽 50/0 LLM)沿用。

### CLS-1-r2 收口 = INJURY_NOT_READABLE_BOTH(预注册停报);结构洞见:训练侧缺测对干净 TEST 的分类是弱通道;三候选重设计待 sol(2026-08-25 01:5x,主线)

**判定采纳**:仪器已修好(identity 有定义:GunPoint 17/35 行、ECG200 36/70 行;Observation 缺测信号两边成立 max_run=15;两跑 BITWISE_IDENTICAL;16/50 fits;TEST/zip SHA 不变),但两底物均读不出 ≥0.05 伤害——GunPoint 丢 18/35 灾行 delayed Δacc = **0.000**(ridge 在 17 行 = 35 行,训练数据高度冗余);ECG200 丢 34/70 行 **+0.020 反号**(多数类 recall 升 0.797→0.922,少数类降);两 impute 臂相对 clean 各约 −0.03。按预注册阶梯停报,非检查点②(headroom 未及测,伤害先造不出来)。

**主线结构洞见(晨议核心)**:文献(Rhodes 2025 等)的插补效应在 train+test **都缺**的设定下取得——插补质量主要在**推断时**咬合;我们的协议墙(TEST 零触且干净)恰好关掉了这条主通道。**对分类任务,训练侧-only 的缺测是弱伤害通道**(ridge 对训练行损失/轻噪声高度稳健,任务边界宽)——这与预测/AD 不同(那两者的 Consumer 本体就是训练产物)。资格门若要开,伤害通道必须换。**三候选重设计(待 sol 裁)**:(A) identity 语义改"零填充最小默认"(而非丢行)→ 可全行受灾,零填充真实扭曲 raw‖diff 特征,伤害可读性高;识别风险 = 零填充是否算诚实 identity(Consumer 侧文档化最小策略,类比 AD adapter 弃权正典)。(B) 缺陷族换**值损坏**(段平坦化/噪声爆发,非 NaN)→ identity 无法用丢行逃逸,特征真实受损;风险 = 与 W43 impulse 族部分重叠,须机制区分(段级 vs 点级)。(C) 缩小训练集(fit 行 ≤10)使每行都承重 → 行损失通道复活;风险 = 量化台阶变粗(n 小)。**另可考虑(D):承认分类切片按此协议墙难产正效应,回 sol 重议切片选择**——但 (A)/(B) 未试前不宜下此结论。今夜停机;CLS-1/r2 两报告由下书 Part 0 收。

### sol 晨间六点裁定采纳;#45-Frep-b 部署修复书发车(2026-08-25 09:4x,主线)

**六点全采纳**:(1) AD 探索冻结——AUPRC 审计保留为负向边界,不追加实验/Consumer/Program/弃权路线,Yahoo 41 不读,AD 不作当前核心 Capability 贡献;(2) 下一书**只修** Frep 两协议缺陷:held-out 零反馈零搜索;A5/A3 对称只用冻结状态;优先从既有快照重新部署,禁顺带修历史债或扩仪器;(3) 修后重裁 A5/A3(首正成本/LLM+fit/held-out 终态/harm),**不得为得正结果继续调协议**;(4) Forecasting 收口后,当前 HEAD 重放既有 controlled classification dynamic-binding capability,验证第二 Task 正向生命周期与 A5/A3;(5) 其后关闭 train-only missingness family,分类新缺陷 = **单一连续段值损坏**(主线候选 B 获准),identity + ≤2 Workflow,本地 UCR,先过无 Agent 资格门;(6) 安全/abstain 仅约束;主线承重 = A5 比 A3 更快更少试错形成有效 Target-local Skill 且 held-out 不差或更好。

**#45-Frep-b 发车(Frep session 续派)**:范围 = 仅部署与计分段重做——(a) 修 F1:部署阶段零 delayed 打开、零 held-out 反馈;(b) 修 F2:两臂对称,只从冻结快照部署(有 ACTIVE 技能臂 Fast-only 召回;无技能臂部署其冻结态既定 incumbent/identity,禁计分块搜索);(c) 优先复用 Frep 四臂既有字节稳定快照与 held-in 读数,只重跑部署+计分;快照不可用即停报。读数 = sol 点 3 四项;判定 = A5A3_TERMINAL_READJUDICATED / SNAPSHOT_UNAVAILABLE / INSTRUMENT_UNREADABLE;预期 0 LLM、重训 ≤100。**排队(不派)**:点 4 分类受控能力重放书(Frep-b 收口后设计);点 5 值损坏资格门(重放后)。

### #45-Frep-b 收口 = C37:A5A3_TERMINAL_READJUDICATED——修复后 pooled 终点翻向 A5(+0.276 差);主线承重三项全立(development 级);CLS-replay 发车(2026-08-25 10:2x,主线)

**判定采纳**:F1/F2 修复(driver 局部,行号在册:DEPLOY_RULE :302-345 / 冻结态两源 :346-412 / 零 outcome 读部署 :413-449 / 纯度回执 :564-612;役中 fc.stage_4 零改动,methods/ 零改动)后四臂部署纯度证明——各臂 1 次 delayed 打开(均为评分器一次性计分,0 用于采纳)、0 候选评估、0 LLM、部署成本全同(9 重训,spread 0);四快照重编译 SHA 自复现且与 #45-Frep 发布值一致,部署后字节不变。**重裁四读数(pooled)**:首正成本 84 vs 123(不变);held-out 终态 **A5 +0.059385(伤 1)vs A3 −0.216513(伤 4/4),差 +0.275898**,对照作废读数 −0.152549;per_channel 精确平局(同 applied 字节,迁移边界照旧)。敏感性钉:改"无技能⇒identity"规则 A3 pooled = 0,A5 仍高于材料带——方向稳健,幅度不稳健。**机制陈述(载重)**:非"A5 同方案打分更高"——A3 冻结 incumbent repair_level_shift 系 held-in delayed 正向(+0.162837)而尾段崩(−0.216513,4/4 伤);A5 的 outlier_iqr held-in 较弱(+0.066941)而泛化(+0.059385)。**= held-in 反馈单独可选中不泛化方案,Source 先验引导的收敛泛化了——积累价值的机制级可视化**;旧协议以 held-out 重搜(0.369 泄漏量)遮蔽此事。

**主线承重口径(sol 点 6)在预测线三项全立(development 级 caveat:已曝光数据、单次配对跑方向 only、per_channel 平局 = 该 cell 积累无边际作用)**:A5 更快(−31.7% 首正成本)、held-out 更好(+0.276)、伤害更少(1 vs 4)。**G1-G5 入册**:G1 fc.stage_4 缺陷仍在役(他调用方继承,债);G2 #45-Frep 工件终态列作废(链与 held-in 读数仍有效,终态只准引 frep-b 工件);G3 FreshSearch 构造器 3 次决策无关 identity 基线 fit 触 held-out support 起点(对称无用,役中修出范围);G4 **held-in 正向方案 held-out 4/4 伤 = Scope/过拟合行素材**(A3 教训 = 未来 Scope 线证据);G5 per_channel 无边际。两披露采纳(两遍部署 13 读数字节同验证;03b07bf 已含 Frep 工件更正)。Part 0 = 098ec40(7 文件)。**Forecasting 线收口;按 sol 点 4 CLS-replay 发车**(controlled classification dynamic-binding 于 HEAD 重放,第二任务正向生命周期验证)。

### CLS-replay 收口 = C38:REPLAY_REPRODUCED(第二任务生命周期 HEAD 验活);X4 与 Frep-G4 同源主题入册;CLS-2 值损坏资格门发车(2026-08-25 10:5x,主线)

**判定采纳**:受控分类 dynamic-binding 能力在 HEAD 精确复现——plan/evaluate/罐装 fast-path 三腿 canonical SHA 逐字节同(全链无 RNG 零 LLM);live-LLM 腿协议零失败(schema/合同/禁字段/未晋升技能全 0,五 context 决策/能力 ID/步骤逐字同),波动仅自由文本与 token 计。守卫逐段复现:H1 精确绑定 6/6、H0 失配 6/6、event_erasure_guard 六触发全 ABSTAIN_KEEP_INCUMBENT 伤害恒 0;两个非平凡回执(ToeSegmentation1 零增益回滚弃 +0.0526 真实收益、SonyAIBO 平局倒向 H0 失 0.178)逐字复现。TaskContext 走 runner 级 legacy typed 路径(classification-local-event-quality-v1),未接 T6 inlet,如实记录未顺手接线。执行者 base 解释器事故当场自抓、作废重跑,零污染。Part 0 = 84aabd1。sol 点 4 门过。

**X 系发现入册**:**X4 载重——A5−A4 = −0.01576,零反馈 source-only 臂反超完整 A5**(根因即两个保守回执;冻结门只测 A5>A3 从不测 A5>A4);**与 #45-Frep G4(held-in 正向方案 held-out 全伤)同源主题:held-in 反馈可误导——预测线上选中不泛化方案,分类线上保守规则丢真实收益**。裁定:第二任务不得读成"A5 占优";"A5 vs A4"缺口列 #46 前必答问题(哪些场景反馈净增值、哪些场景先验已足);该主题升格为**跨任务重复观测**,#46 细化主张素材。X1(live fast-path 静默覆写历史 plan 证据,§7 抵触)记债——证据保全修复列下一工程窗;X2 零回归测试保护、X3 材料指纹缺口、X5 provider token 口径注记。TEST 打开 6 条系原书已 EXPOSED 材料 development 重放,合规。

**CLS-2 值损坏资格门发车(sol 点 5;train-only missingness family 正式关闭)**:缺陷 = **单一连续段噪声爆发**(每灾行 1 段,段长 15–20% 行长,幅度 4–6× 行 std,50% 行受灾分层保类平衡,seed 固定)——与 W43 点脉冲族机制区分(连续段 vs 孤立点);identity 无法丢行逃逸(值损坏无 NaN 标记),直接拟合于损坏特征 = 伤害通道;菜单 = identity + hampel_filter + 执行者从注册表侦察的第二合法段修复候选(≤2,机制匹配,无则单 Workflow 并报);门沿 r2(伤害 ≤ −0.05/恢复 ≥50%/类 recall 守卫/Support 方向);底物 GunPoint/ECG200 二选一(选 clean acc 更有下落空间者);0 LLM,fit ≤50。

### CLS-2 收口 = INJURY_NOT_READABLE;三连击模式定性 = Consumer 钝感;检查点③停报待用户/sol(2026-08-25 11:1x,主线)

**判定采纳**:GunPoint 上连续段 5σ 噪声爆发(50% 行受灾、段长 15-20% 行长)对 identity 的 delayed Δacc 仅 **−0.0133**(门 −0.05);hampel 零恢复,outlier_mad 反伤至 0.6933(类 1 recall 0.378——全局裁剪对连续段过伤);Observation **可见**(local_robust_z_peak 18.2→67.2,hit 行 23/25 上升,max 368.9 ≫ 阈 4.0;coverage 恒 1.0 属预期)= 值损坏版结构前提:**Agent 看得见,判官感不到**。执行干净(0 LLM,4/50 fit,两跑逐位,TEST 零触,菜单侦察表 13 算子 + 选型理由在册,幅度/受灾率未扫)。书外:loader 行 z-norm 使 5×std 恒为 5.0;Support 比 delayed 更伤(抽样倾斜);identity 净精度不动系类交换(0.934→0.868 vs 0.703→0.743)。

**三连击模式定性(CLS-1/r2/CLS-2)**:每行缺测 × 丢行 = 未定义;丢一半训练行 = 零伤害(数据冗余);连续段 5σ 值损坏 = −0.013(拟合穿透损坏)。**结论:ridge-raw-plus-diff 在这批 UCR 上对"合理烈度的训练侧攻击"实质免伤(≥0.05 门下)**——文献(Rhodes:kNN/RF 敏感;TANDEM:RNN 0.58→0.47)的效应都在更敏感的 Consumer 上取得。"Observation 可见 + Consumer 钝感"本身是分类版条件化素材(同一损坏,敏感判官受伤、稳健判官免疫)。**触发 sol 检查点③(改 Consumer/缺陷 family 须停报)——停,不再单方面迭代**。三选项呈用户/sol:(A) 资格门 Consumer 换 kNN(k=3,确定性,sklearn 现成,文献背书的敏感判官;ridge 保留为第二 Consumer 做条件化对照——"kNN 受伤 ridge 免疫"= 分类版 M0 素材;主线推荐);(B) 接受"可见但不伤"关闭该族,升级烈度轴(幅度/段长/率,有扫描味,不推荐);(C) 换更窄边界底物。CLS-2 工件未 commit 待收。

### sol 检查点③裁定采纳:证据表收紧;两步序(CLS-OP 共享 Harness 闭合 → CLS-3 配对 Consumer 资格);A3/A4/A5 三臂报告标准;双线发车(2026-08-25 11:4x,主线)

**主线两处自我修正(sol 指正)**:(1) "三连击"虚计——CLS-1 系主线设计碰撞非 Consumer 证据;有效证据 = r2 + CLS-2 两条且同判官,仅支持"ridge 对缺测/连续段噪声钝感",不支持"分类难融入"更不支持"Agent 不会分类准备"(C38 已证 ridge 对类条件 impulse 敏感且可修——**不同缺陷 family 的 headroom 不同**);(2) kNN 定位纠偏——禁"换敏感判官取正结果"的 Consumer 挑选;正确定位 = **配对固定双 Consumer(ridge+kNN),同数据同损坏同菜单,测 Consumer-conditioned response**;若 ridge 免伤(identity 正解)而 kNN 受伤可修 = "同一质量问题的重要性取决于 Consumer"的干净多 Consumer 证据;Agent 不选 Consumer(环境给定),Agent 按 Consumer Context 决定处理与否。**证据表收紧在册**(八行:成立/不成立/未测逐条,C38 六 Target H1 绑定 6/6、A5−A3 +0.0833、零负 Target、A5−A4 −0.0158)。**canonical 表述**:"Classification 已有受控正向 Capability,尚未接进当前共享 Harness 的真实 Experience→Target-local Skill 生命周期"。**A3/A4/A5 三臂报告标准采纳**:分类主实验必报三臂(A4 = Source 直用零适应),主比较 A5 vs A3,A4 答"Target 反馈是增益还是带偏",禁隐藏 A4>A5。

**两步序采纳并双线发车**:**第一步 CLS-OP(Opus)**= 用 C38 已知正向 dynamic-binding family 做共享 T6 Harness 的最小 operational replay——Source Positive/Negative/Conflict **真实 Episode 入 experience_memory**(禁预编译答案旁路)→ A3 空 / A4 Source 直用 / A5 Source+适应,同 held-in 预算 → Target-local Skill → freeze → Fast-only;复用现有 Observer 与 center-excluded local median Workflow;不找新缺陷;允许且仅允许一个薄 classification evaluate_fn 适配器(循 ad_scope_adapter 先例);development 级。通过 = 第二 Task 真正进同一 Harness。**第二步 CLS-3(grok,并行,设计全由 sol 钉死零开放决策)**= 配对 Consumer 资格:GunPoint + 复用 CLS-2 注入账本;ridge 与 kNN(k=3 欧氏 uniform 冻结);菜单 identity+hampel+MAD 禁增;0 LLM 禁扫描(k/距离/强度/数据集);过门五条件(kNN 伤达线/≥1 合法修复/类 recall 守卫/Support 预测/ridge 保持钝感成差异);四出口表(伤且修 → Context-conditioned Agent 实验;伤不修 → Program Supply gap 不跑 Agent;不伤 → 关闭连续段族;Support 不预测 → Feedback first-fault)。

### CLS-3 收口 = KNN_INJURED_NO_REPAIR;配对 Consumer 差异证据到手;反馈误导第三例入册;CLS-4 段修复补给发车(2026-08-25 11:5x,主线)

**判定采纳(sol 预登记出口对号)**:五条件 1/5(kNN 伤 −0.120 过线 ✔)、5(ridge 钝 −0.0133 ✔)成立,2/3/4 败——hampel 在 kNN delayed 上逐字节等于 identity(局部窗修不动 15-20% 连续段),outlier_mad 继续伤(kNN 类 1 recall 0.689→0.203)且 Support 反抬(0.60→0.73)。**配对差异成立:kNN − ridge injury = −0.107**——"同一可观察质量问题对不同 Consumer 重要性不同"的分类版证据到手(Observation 可见 + ridge 免伤 + kNN 受伤)。出口 = Program Supply gap,**不跑 Agent;连续段族不关闭**;下一步 = 补给,禁扫 k/换底物。执行干净:CLS-2 账本零重抽回放(SHA 同),ridge 四臂与 CLS-2 逐字节复现,8/30 fit,0 LLM,两跑逐位。

**反馈误导第三例(载重,跨任务主题加固)**:Support 序 clean>mad>hampel>identity vs delayed 序 clean=hampel>identity>mad——**Support 会选中 delayed 最差臂**。与预测线 G4(held-in 正向方案 held-out 全伤)、分类 X4(A5<A4,保守回执丢真实收益)并列为第三例:**"held-in/Support 反馈在三个不同场景下系统性误导"= #46 候选头条主题之一**(它同时论证:晋升必须走独立 delayed、Support 只许起草——我们的三层反馈模型不是过度设计,是被三次实证救回来的)。

**CLS-4 段修复补给发车(解锁条件满足:八臂矩阵证菜单对该缺陷枯竭;一个机制匹配算子)**:新算子 repair_burst_segment(intrinsic):滚动 robust-z 检测连续高偏差段(|z|>3.5 且 run ≥8,参数冻结禁扫)→ 段两端线性插值替换;allowed_tasks = classification 单任务起步(扩域需证据);注册表契约字段如实填(destructive=yes/值改写);配单测(确定性/边界/无段时恒等)。落地后同账本重跑资格:identity + hampel(阴性对照)+ repair_burst_segment,双 Consumer 十臂,同五条件判(2/3/4 重判,1/5 沿用已证)。出口沿 sol 四格表。0 LLM,fit ≤30。

### CLS-4 收口 = REPAIR_INSUFFICIENT;主线停迭代,连续段族补给挂起;反馈误导第四例;战略选项呈用户/sol(2026-08-25 12:0x,主线)

**判定采纳**:repair_burst_segment 入库(5ef9726,单测 7/7 + 族测 17P;registry 复查无同机制;classification 单任务;参数冻结未扫)但真实账本上检出塌——IoU 0.142、点 precision 0.160、**干净行 56% 被误改写**;kNN 恢复 −0.444(与 outlier_mad 塌到同一决策面,执行者验证非串档),ridge 上该修法再伤 0.140("ridge 不需修"再证)。根因机制性:全局 z 误判类形状峰谷为段、5σ 高斯洞把真段拆短于 min_run=8——非阈值问题。**反馈误导第四例**:Support 仍把 burst/mad 抬到 identity 上,delayed 两者最差。执行干净(新 fit 2/30、引用臂零重跑、两跑逐位、0 LLM)。Part 0 = 3656355。算子留库,其测得局限如实在册。

**主线裁定:连续段族补给挂起,停止在同一账本上第三次迭代检测器(过拟合风险)**。已入袋不失:配对 Consumer 差异证据(kNN −0.120/ridge −0.0133)与"现役注册表对连续高斯爆发无合法修复"的 Supply 边界事实。**战略选项呈用户/sol**:(A) 主线推荐——**连续段族封存为诚实 Supply gap,Context-conditioned Agent 实验改用 C38 impulse family**(那里缺陷+合法修复[center-excluded local median]+Consumer 敏感性三件俱全,A5−A3 +0.0833 已证;配对 kNN 差异可在 impulse 族上补测一次资格,若同样分化则 Agent 实验两条件俱备);(B) 第三次补给(局部窗 z + 容洞 run 合并的机制改良检测器)——机械上有据但同账本三迭代有钓鱼味;(C) 等 CLS-OP 收口后合盘再定。CLS-OP(共享 Harness 三臂)仍在飞,不受影响。

### sol 执行裁定:现选 C、CLS-OP 过后转 A;禁 B;主线两处计数修正;ccfa.yaml 更新义务(2026-08-25 13:2x,主线)

**裁定采纳**:连续段损坏族**封存**(Hampel/MAD 无效或有害、段修复 v1 检出塌,两次 Supply 失败,禁同账本第三迭代 = 禁 B);**当前先 C(等 CLS-OP 收口),通过后转 A**(C38 impulse 族上补最小 ridge/kNN 配对 → 若分化则 Consumer-conditioned Agent 实验);CLS-OP 不通过则只修其暴露的第一个断点,不扩多 Consumer。repair_burst_segment 留库作负向经验,**正式 Agent 菜单暂排除**(已知有害动作不得暴露给 Agent,直到独立正向证据)。**主线两处计数修正(sol 指正)**:(1) C38 +0.0833 = 窄受控族正向迁移证据,非 Consumer 敏感性证据(后者需 impulse 族 ridge/kNN 配对另证);(2) 反馈误导独立场景数 = **三**(forecasting G4 / C38 X4 / GunPoint Support 倒序)——CLS-3/CLS-4 同账本复现不重复计证。**分类线诚实状态(sol 三句)**:窄修复正例在手;delayed Judge 可用而小样本 Support 不可靠(必报五读数:acc/AUC、每类 recall、worst-class harm、Support-delayed 方向一致性、A3/A4/A5 含 A4 带偏检查);算子供给仅够 impulse 一族。**ccfa.yaml 陈旧(仍停 #44a 前)——CLS-OP 收口后立即更新**(义务)。后续序:CLS-OP → [过:UCR 未用 Target 机械选一做确认(不下载不广选)→ impulse 配对 → Agent 实验 | 不过:修第一断点] → 最终整合(Forecasting 正证 + Classification 补证 + AD 边界负证 + 跨域总主张)。

### CLS-OP 收口 = C39:SECOND_TASK_LIFECYCLE_CLOSED(里程碑成立)+ SUPPLY_STARVED_BY_WINDOW_VERIFIER(对决被饿死);first fault 字节级定位;校验器语义修复提案呈 sol;ccfa.yaml 已更新(2026-08-25 14:4x,主线)

**里程碑判定采纳**:第二任务全链过共享 Harness——run_online_round→open_delayed→activate_approved 跑通;**真实 Episode 5 条**(Source 3 NEUTRAL + 1 CONFLICT,Target 1 CONFLICT,全经 classify_relation 机械判定,零预编译);Slow 1 次调用产六段 Source Skill 过 7/7 确定性审计(证据全 immaterial → TRY = NO_AUTHORIZED_ACTIVE_RECOMMENDATION,执行权 withheld——审计纪律正确);三臂同预算,freeze 后 Fast-only 部署纯度 6/6(零 delayed 采纳/零 Slow/零 LLM/store 不变/scored==applied);TaskContext 全请求携带 SHA 断言;唯一授权适配器 cls_scope_adapter.py 循 ad_scope_adapter 先例。LLM 74/80、fit 16/600。**但对决未发生**:9/14 held-in 轮零合法 Support receipt,三臂全冻 identity,A5−A3 = A5−A4 = 0.0000 双 Target——里程碑成立、读数无信号,两句并记,三臂表不得读作任何 capability 结论。

**first fault(0-LLM 定位,采信)**:ScopeExecutor.verify 系 **cohort 全有或全无**——任一窗口拒即全候选拒;预测/AD cohort ≈ 十余窗,分类 = **每 fit 行一窗**(42–1260),"至少一行超 maximum_modified_fraction=0.10"几乎必然。机制对口的 hampel_filter 均改 6.55%/6.78% 却因 157/1260、18/95 行超线整体被拒(0.20 下均过);0.10 下唯一幸存者系数值 no-op(denoise_median window=1)。执行者拒绝自行放松重跑("为正结果调协议"红线)——追认为范例。**修复提案(呈 sol,选项已避结果污染)**:主线推荐**语义修复而非数字调整**——modified_fraction 约束从"每窗一票否决"改为 **cohort 聚合口径**(总修改点/总点数 ≤ 0.10,数字不动;辅以逐行分布报告不设否决)——理由:契约本意是约束修改质量总量,预测/AD 的十余窗几何下两种语义近等价,分类每行一窗几何使 per-window 语义退化为必拒;此修复几何不变量、不引入任何结果导出的新数字(0.20/超线行占比等方案均已被探测数字污染,不取)。sol 裁后 CLS-OP 同 roster 同菜单同预算重跑(只改这一面)。

**自报事项追认**:runner 新文件必要;maximum_candidates 2→3 循 #42k 先例(适应期探索语义)——补呈 sol 确认;support pool 四分出 delayed 面(family 无此面,共享生命周期必需);观测窗收 3200 点(actionability 探测 215s 成本,executor 仍作用全行)——均记录在案。**债与发现**:verify_candidate.selectable 不要求 effect distinctness(no-op 占提案槽,饿死轮的次生因);run_online_round 拒绝不带 rejection code(无法自诊);registry 无 bound center-excluded local-median(C38 程序系 legacy oracle-bound,共享菜单最近亲 = hampel);stable_task_event 条件判官饱和(incumbent acc 1.0);**operators/registry.py 多线共写竞态注记**(CLS-4 的 5ef9726 在 CLS-OP 执行中落地——主线自查:系本线两并行派发所致,立站规:**涉 operators/ 或共享 runner 写入的书不并行**)。ccfa.yaml stage 块已更新(post_CLS_OP,gate = 校验器语义裁定→重跑→impulse 配对→Agent 实验)。

### sol 校验器裁定 + 六步序采纳;主线三处修正自认;CLS-OP-r2-prep 发车(2026-08-25 15:1x,主线)

**sol 裁定采纳**:(1) 校验器语义修复获准——maximum_modified_fraction=0.10 数字不动,"任一窗超限全拒"改"总修改点/总点数",逐行分布保留作诊断,禁 0.20(已被结果提示);(2) **r2 单面纪律**:maximum_candidates 保持 C39 实际使用值 3(追认 executor 修正为基线),r2 不得再动任何其他门——否则不可归因;(3) **先双 0-LLM 门再花 LLM**:smoke(非恒等候选能否获合法非 no-op 回执)+ Program headroom(现共享菜单是否真有正向候选)——主线自认曾预设 hampel 会 POSITIVE,无据;C38 验证的是 center-excluded local median + 动态绑定,注册表无此精确 Workflow;(4) headroom 无 → 唯一允许的补给 = **接入 C38 已验证 Workflow 为 Typed Workflow**(只提供可执行动作,Scope 与采用仍归 Agent),再查 headroom,仍无则停 family——**校验器只修一次,禁无限修仪器**;(5) 分类迁移成立判定门:r2 的 A5>A3 仅 development 级,**须加一次未使用本地 UCR Target 冻结确认**(机械选,不下载),Yahoo 41 系 AD 资产禁挪用;(6) 写作与非阻塞债(runner 拆分/rejection code 等)继续押后。**六步序定稿**:verifier 修复+smoke → 0-LLM headroom →(有正向 → CLS-OP-r2 三臂[须真实形成非恒等 Target-local Skill;报首正成本/delayed/harm/abstention/A4 带偏判]| 无 → 接 C38 Workflow → 再查)→ 未用 UCR Target 确认 → impulse ridge/kNN 配对(有差异才跑 Consumer-conditioned Agent)→ 整合写作。

**CLS-OP-r2-prep 发车(Opus 续派,0 LLM)**:Part A 校验器语义修复(实现于 ScopeExecutor.verify 路径,聚合口径;跑受影响测试子集证预测/AD 线零回归——若聚合语义改变既有线行为,如实报字节证据停裁);Part B smoke(修后非恒等候选在 CLS-OP 同材料上获合法回执,排除 no-op);Part C headroom census(共享菜单全部 classification 合法非恒等候选 × Source cells + Target held-in 面,确定性 Δacc 表,材料线判有无正向候选);判定 = HEADROOM_EXISTS(→ r2 发车)/ NO_MENU_HEADROOM(→ C38 Workflow 接入书)/ VERIFIER_FIX_REGRESSES(停)/ INSTRUMENT_UNREADABLE。

### CLS-OP-r2-prep 收口 = HEADROOM_EXISTS;校验器修复零回归落地;CLS-OP-r2 三臂发车(2026-08-25 15:5x,主线)

**判定采纳**:双门全过。**Part A 修复范例级**——scope 开关实现(默认 per_window,全部既有调用方字节不变,仅本执行器选入 cohort 聚合;0.10 不动;非 fraction 门仍逐窗否决;两口径诊断都产出),零回归证明用"只换单文件字节跑两遍"法(避 stash 殃及另线),40 失败集合逐条相同(sha 相等;38 条系 h0 锁失效先在);新单测 8 项含"唯一分歧案例"与"超线窗计数无否决权"。**Part B**:修复解锁 hampel/repair_level_shift 于 4/5 cell(GunPointAgeSpan 0→2 非恒等存活),非普遍放松(全局平滑族 cohort 比值 0.86-1.00 仍全拒;Lightning2 outlier 族仍拒);no-op 判据按 prepared 字节恒等,10 no-op 零 fit 支出。**Part C**:GunPointAgeSpan hampel Support +0.5000/worst Δrecall +0.4000、delayed +0.3000/+0.2000(材料线 0.1000,n=10 粒度粗注记在案);Phalanges hampel +0.0222 与 repair_level_shift 双面过 guard。fit 46/120、0 LLM。

**债与注记**:h0 锁再失效(CLS-4 的 5ef9726 改 operator_bundle_sha 所致——并行竞态站规的实证第二笔),38 测试红,机械重生成条件触发 → r2 Part 0b 执行;effect-distinctness 债升格(修复前 GunPointAgeSpan 唯一供给全是 no-op,r2 后优先偿);repair_level_shift 分歧注记(Lightning2 Source 拒 vs Phalanges Target 喜——r2 Memory 将见此分歧,真实 CONFLICT 素材);GunPointAgeSpan n=10 粒度 → sol 第 5 步未用 Target 确认因此必要。**CLS-OP-r2 发车**:全链重跑(Source 形成 + Slow 整合 + 三臂 + freeze + Fast-only)于修复后校验器,同 roster 同菜单同预算,maximum_candidates=3,fresh run-id 禁复用 C39 冻结态;预注册预期(可证伪):Source 段 hampel 存活或产 POSITIVE Episode → Slow 或授权 TRY → A5 获真实先验;须真实形成非恒等 Target-local Skill;报首正成本/delayed/harm/abstention/A4 带偏判。

### CLS-OP-r2 收口 = C40:CLS_LIFECYCLE_OK_NO_ADVANTAGE——分类首个真实 Target-local Skill(+0.2690 零类伤)与"咨询性先验净伤害"并立;CLS-CONF 确认实验发车;OBSERVE 指名规则假设呈 sol(2026-08-25 17:0x,主线)

**判定采纳**。**正面(载重)**:分类线首个非恒等 Target-local Skill 经共享 Harness 全链形成——A3/GunPointAgeSpan:hampel 探测 → Support +0.5000(两类 recall 不跌)→ POSITIVE → Draft → delayed +0.4000 → 批准 ACTIVE → FROZEN_ACTIVE_SKILL_RECALL 部署 → **held-out 0.8513 vs identity 0.5823 = +0.2690,两类 recall +0.2812/+0.2564,零伤害**;首正成本 LLM 6/候选执行 1;Support-delayed 方向 2:0。功效注记:held-in 面仅 10 行(+0.5 = 5 行差),held-out 316 行同号佐证,held-in 读数不单独引用。Phalanges 三臂全 identity = 弱场正确弃权(探测发生、守卫拦截,与 headroom≠生命周期 POSITIVE 门的两把尺一致)。**负面(同等载重)**:A5−A3 = **−0.2690**——Source 卡每轮被检索(非检索错)、无执行权供不出候选、但 OBSERVE 文本指名 outlier_mad/winsorize 使 A5 两轮提案落 level-shift 族(该 cell cohort 修改比 0.69 必拒)、零合法读数(非反馈带偏)→ 冻结 identity;机制自动归类 **prior_delivered_but_steered_the_proposal_elsewhere:咨询性文本先验有行为效应且方向为负**——先验误导 = 与"反馈误导"并立的新形态,第四个独立场景。A4 结构性 identity(TRY 空)。预注册 P1/P4 HELD,P2/P3 FALSIFIED(Source 仍无 POSITIVE、TRY 仍空:修复打开探测序但证据仍分裂)。

**执行质量**:Part 0b 锁重生成 content sha 一致(cb03eb6);fresh run-id,C39 状态零复用(Source 快照 sha 异);Source Episode 4→9(6N/2C/1NEG,6/6 cell 有回执);LLM 69/90、fit 41/600;纯度 6/6;中断恢复自证(无残件,LLM 0 起算);未为正结果动任何门。**发现四则**:AddTargetExistsError 生命周期缺口(同步骤新 id 候选不识别为重部署,二轮确认证据未入 Skill 记录——债);0-LLM 预检有真实预测力(census 预测与实测逐项符,建议固化为标准前置——采纳为站规);headroom 尺与生命周期 POSITIVE 尺分开报(站规);单候选轮放大先验影响。**排程**:CLS-CONF 发车(sol 步 5:未用 UCR Target 机械选一,同注入模板同预算,A3+identity 两臂,验 +0.2690 非孤例);**OBSERVE 指名规则假设呈 sol**(单面:"TRY 为空时 OBSERVE 不得指名算子",同账本重跑 A5 臂可证伪——先验误导的治理修复,与契约红线同宗)。

### CLS-CONF-r2 停摆:PREDICTION_GATE_FAILED——字面 token 规则下合格集空,与预注册 8 件/Computers 不符;0 LLM 收工(2026-08-25 20:3x,执行方)

**判定**:**PREDICTION_GATE_FAILED**(停摆类)。选中 Target = **None**。机器重算合格集 `[]`,预注册 8 件(Computers / FreezerRegularTrain / GunPointMaleVersusFemale / GunPointOldVersusYoung / PowerCons / SemgHandGenderCh2 / WormsTwoClass / Yoga)全部因认领 runner 文件文本含 `fit_only_artifact` 或 `stable_task_event` 出局。按书:不得消耗 LLM、不得现场放宽或收紧规则。Part B 两臂未开。

**选靶归属摘要(40 zip, r1 census 缓存复用)**:其余排除条件未动——不可载入 2(DodgerLoopWeekend、KeplerLightCurves);TRAIN 行数不在 [40,400] 的照旧出局;无非二分类。收窄后的"未使用"=认领 `evaluation/functional/run_*` 文件不含两条件 token。在行数合格且可载入的二分类里,预测 8 件与 Earthquakes/Ham/Herring/Wine/ECG200/GunPoint/GunPointAgeSpan/HouseTwenty/Lightning2/ToeSegmentation1 等一律命中 token。

**Earthquakes 排除证明**:认领方仅 `run_e2_action_credit_candidate_ordering.py`;该文件文本同时含 `fit_only_artifact` 与 `stable_task_event`。与主线预注册一致。

**预测 8 件为何全灭(书外发现,规则未改)**:主线预算把"只被 integrated_context / source_prior 认领"当成"未在 impulse 条件对下使用"。字面 grep 否证——(1) `run_e2_integrated_context_harness_evolution.py` 的 `TARGET_DATASETS` 含 Computers/PowerCons/Yoga/SemgHandGenderCh2/WormsTwoClass,且 `CONDITIONS = ("fit_only_artifact", "stable_task_event")`,这五件**确实在该条件对下跑过**,不是 token 误伤;(2) `run_e2_source_prior_evidence_fusion.py` 文本含 `stable_task_event`(读 W56 planned scope 字段),FreezerRegularTrain / GPMvF / GPOvY 因此过度排除——它们可能从未在该条件对下计分,但书规定"宁可过度排除",执行方未改成语义"对该数据集跑过该条件"。本地池在此字面规则下为空。

**两臂**:未跑。无 held-in 轨迹、无 Skill/冻结、无 held-out accuracy/逐类 recall。cohort 校验器与 `maximum_candidates=3` 已在既有 runner 落地,未改 `methods/`。

**成本**:LLM 0/40;Consumer fit 0/200;墙钟 2.1 s(选靶干跑);下载 0。解释器 `D:\Anaconda_envs\envs\project\python.exe`。

**提交**:Part 0 r1 停摆件 `2d055ea`(内容未改);本书 runner + r2 隔离工件 `t6_cls_conf_r2_unused_target.json/.md`(未覆写 r1) + 本节。义务自报:规则未放宽/未收紧;LLM 0;fit 0;下载 0;`methods/` 零改动;他线文件(`AGENTS.md`/`README.md`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP`)未碰。

### T1 最小 Fast 可见性修复:无行动资格经验卡不进 Fast 提案视图(2026-08-26,执行方)

**谓词定义(单一行为谓词,落在 Fast 检索闸口)**:一张**经验卡**(唯一识别特征 = `risk_guards.sections` 且其中含 `TRY` 段——Slow 编译的 Source 卡/recipe compilation 卡都是这个形状;bootstrap 程序卡与带 `Frozen program steps:` 的 Target-local 卡都没有该字段,故结构上不在谓词射程内)当且仅当同时满足下述两条时为 **inert**,`role == "fast"` 时整卡不进 `EffectiveHarnessView`(不进 Fast prompt、不参与提案检索、不参与 `_skill_frozen_candidates`):(1) **无授权 TRY** = TRY 段空、等于 `NO_AUTHORIZED_ACTIVE_RECOMMENDATION`、或匹配 `NO_[A-Z0-9_]+` 全大写哨兵;(2) **无重复 scoped RISK** = 不同时具备【机器 Context Scope 严于资格门:`observable_applicability` AST 递归里存在非 `task_kind` 的 feature 叶,`{"const": True}` 与纯 `task_kind` 均不算】与【显式重复证据字段:`risk_guards.evidence_distinct_task_count >= 2`,复用 `risk_skill.py:246` 既有字段,未新增 Schema】。保留侧:TRY 点名授权算子的卡照旧可见;RISK 已 scoped 且计数的卡照旧可见;非经验卡一律不受影响。`role == "slow"` 完全不变——卡仍在 store、仍被 Slow 解析,说过什么仍可审计与修订。

**落点(单文件、单闸口)**:`methods/ttha/retrieval.py:274`(`resolve_harness_view` 主循环内,紧随 `_RESTRICTED_GUARD` 之后、`bootstrap` 的 `continue` 之后),谓词实现 `:195-238`,两个辅助 `:158-169` / `:172-192`,常量 `:145-155`。同文件 `import re` 一行。连带机械件:`methods/ttha/harness/h0/snapshot.lock.json` —— `retrieval.py` 是 `_dependency_shas()`(compiler.py:234)登记的 runtime 依赖,改动必须 `--write-lock` 重生成,否则全部 `compile_snapshot(H0_ROOT)` 报 lock mismatch。重生成后**只有两键变化**:`dependency_shas.ttha:retrieval` 785d6994→d438149f、`runtime_bundle_sha` c3917fcc→4abf3bec;`harness_content_sha` 53b1c803 逐字节不变(编排内容未动)。

**两项测试(新文件 `tests/methods/test_inert_experience_card_visibility.py`,零 LLM)**:(1) 聚焦单测 `test_only_the_card_with_nothing_to_authorize_leaves_the_fast_view` —— 用 `source_skill.build_skill_payload` 造同 applicability(`task_kind == classification`)两卡,TRY 哨兵+自由文本 RISK 卡不在 Fast 视图、TRY 点名 `hampel_filter` 卡仍在;三张 bootstrap 程序卡(`build_contrastive_candidates`/`inspect_and_localize`/`select_or_identity_and_verify`)在有卡与无卡两种 snapshot 下均仍在 Fast 视图;inert 卡仍在 `role="slow"` 视图。**PASSED**。(2) C40 fixture 检查 `test_the_c40_source_card_is_excluded_by_the_predicate` —— 直接读 `artifacts/functional/e2/t6_cls_op_r2_three_arms.json` 的 `part_b.source_skill_entry`(真 `source_investigation_cls_v1`:TRY = 哨兵、RISK 自由文本点名 `repair_level_shift`、applicability 仅 `task_kind`),经 `load_skill_entry` 入 snapshot 后不在 Fast 视图、在 Slow 视图。**PASSED**。

**回归与不变性**:`tests/methods/test_ttha_agent.py` **22 passed**(与新文件同跑共 24 passed / 4.13s);另跑 `tests/functional/test_source_derived_skill.py` **22 passed**(其 `SECTIONS_OK` 卡 TRY 点名 `outlier_iqr`,仍进 Fast——保留侧活证)。**A3 冷启动不变性**:h0 profile = `h0-domain-naive`(capability 库必空),bootstrap 卡在主循环里先 `continue`,谓词结构上到不了它们;实测 h0 Fast 视图 `effective_harness_view_sha` 改动前后同为 `95be1f62050790b4784e47d6b3270c9a3511b2001ebfb6e57c3b77ebb03039d6`(forecast 与 classification 两组特征均如此,skill_ids 仍为三张 bootstrap 卡),即 A3 冷启动 Fast 视图逐字节不变(以 `git stash` 基线对照实测)。

**结构发现(未修,按纪律不扩面)**:(1) `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:3675-3701` 的 `a5_deficit_mechanism` 分类器把"卡未出现在 `retrieved_skill_ids`"直接判为 `retrieval_binding_miss`——修复生效后该 runner 若重跑,会把**故意的治理性扣留**误标成检索绑定失误;该 runner 属既有 runner,本轮未碰,T2 重放读数时须按此校正解读。(2) `deterministic_recipe_compilation_card` 一族(`run_e2_skill_store_integration.py:314` / `run_e2_fresh_confirmation.py:780`)在无 priority clause 时把 TRY 写成自由文本"No priority clause was compiled: …"而非哨兵,机器不可识别,故不触发本谓词;要覆盖须让该编译器写哨兵,属另一面改动,本轮不做。(3) `tests/functional/test_target_local_risk_skill.py:457` 记录的"第二条 Fast 通道"(runner 侧 Target-local Skill 直查 snapshot,不走 `resolve_harness_view`)只供 frozen-program 型 Skill,不搬运经验卡文本,本谓词不需要在那里重复落地。

**义务自报**:改动文件 4 件 —— `methods/ttha/retrieval.py`、`methods/ttha/harness/h0/snapshot.lock.json`(机械重生成)、`tests/methods/test_inert_experience_card_visibility.py`(新增)、本节;未碰 `operators/`/`contracts/`/`runtime/`/`evaluation/` 既有 runner,未碰他线文件(`AGENTS.md`/`README.md`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP`——工作树中它们由他线持有修改,本轮未 stage)。**LLM 调用 0;下载 0;consumer fit 0;未跑全仓 pytest**。解释器 `D:\Anaconda_envs\envs\project\python.exe`。未处理任何背景债,未顺带清理代码。

### T2 C40 A5 单臂机制重放:VISIBILITY_INVARIANT_HOLDS(2026-08-26,执行方)

**判定**:**VISIBILITY_INVARIANT_HOLDS**(机制级;不产 capability)。同一张无权卡 `source_investigation_cls_v1` 从 r2 账本安装进 fresh A5 store(Slow 可见、store 在场),`--r2-replay-a5` 只重跑 GunPointAgeSpan A5:held-in r1/r2 → freeze → Fast-only。**卡在预检 Fast 视图、两轮 `retrieved_skill_ids`、部署 Fast 视图共 3 面确定性缺席**。C40「每轮唯一候选都是 level-shift」模式打破:r1 仍为 `repair_local_level_shift`(verifier 拒,0 Support);r2 改为 `outlier_iqr` 并获得合法 Support 回执(gain 0.0,NEUTRAL,delayed 未开)。探索通道恢复见证 = 该 Support 回执;提案族与 A3 冷启动(hampel)不同型,不要求。终局 held-out accuracy **0.5823 = identity**(信息位,不判定;不要求等于 A3 的 0.8513)。

**后端锁定**:r2 工件只记 `obligations.backend = live Fast Agent`;代码路径还原并实测 = Fast `gpt-5.6-sol` @ `https://api.agicto.cn/v1`(probe `ok`,returned_model 同名)。未换中转/模型。协议帽沿 r2:LLM 90/Fast 82、fit 600;本臂实耗 Fast 9(与 r2 A5 GunPoint 9 同量级)+ probe 1 不计入臂帽、fit 3/600、墙钟 545.9 s。

**`retrieval_binding_miss` 误标**:`r2_annotate` 分类器(~:3675-3701)见卡不在 `retrieved_skill_ids` 会标 `retrieval_binding_miss`。本迹若过该分类器也会如此。这是**故意的治理扣留**,不是检索失败;分类器与共享 runner 既有逻辑**未改**。

**提交**:runner 最小入口 `--r2-replay-a5`;隔离工件 `t6_cls_op_r2_a5_replay.json/.md`(未覆写 r2 三臂件);本节。`methods/`/`runtime/`/`contracts/`/`operators/` 零改动;他线文件未碰;下载 0;未跑全仓 pytest。解释器 `D:\Anaconda_envs\envs\project\python.exe`。

### CLS-DEV-ECG200:本地轻底物 development 级 conf 生命周期收口——DEV_CHAIN_NO_POSITIVE(2026-08-26 11:4x,执行方)

**判定**:**DEV_CHAIN_NO_POSITIVE**(development;非独立确认;禁 `CLS_CHAIN_CONFIRMED`)。门算式与 `_conf_verdict` 相同:非 identity Target-local Skill 未形成;A3−Static held-out accuracy = 0.0 < max(0.005, 1/100)=0.01;逐类 recall 无伤害(worst Δ=0.0)。部署纯度 `all_pure=true`。ECG200 曾被 W48/W49/curvature 在同一 impulse 条件对下用过(见 `t6_cls_conf_r3_selection.json`),本跑只作开发数据复用。

**两臂读数**(held-out n=100,官方 TEST 未注入):

| 臂 | Skill | held-out acc | vs identity | recall 0/1 | recall Δ | Support-delayed | 部署 |
|---|---|---|---|---|---|---|---|
| A3 | 无 | 0.6000 | +0.0000 | 0.6389 / 0.5781 | 0.0 / 0.0 | 0:0 | `FROZEN_LEDGER_NO_INCUMBENT_IDENTITY` |
| STATIC | 无 | 0.6000 | +0.0000 | 0.6389 / 0.5781 | 0.0 / 0.0 | 0:0 | `FROZEN_LEDGER_NO_INCUMBENT_IDENTITY` |

**A3 提案轨迹**:两轮 `retrieved_skill_ids` 仅 bootstrap 三卡(`build_contrastive_candidates` / `inspect_and_localize` / `select_or_identity_and_verify`),无 Source 卡。r1:`repair_level_excursion` verifier 拒,0 Support,abstain。r2:选 `remove_broad_extreme_deviations`=`outlier_mad`,Support −0.1429 → NEGATIVE(delayed 未开);`repair_early_level_shift` verifier 拒。冻结后 Fast-only = identity。

**与 GunPointAgeSpan 正例形态**:同型 = A3 冷启动 + bootstrap 三卡 + cohort 校验器 + `maximum_candidates=3` + held-in r1/r2 → freeze → held-out Fast-only,管线端到端收口(conf 机制首个完整样本)。异型 = GunPoint r1 即 hampel Support +0.50 / delayed +0.40 → Skill → held-out +0.2690;本跑从未提案 hampel,冻结后诊断 `hampel_filter` 在此底物被 `COHORT_MODIFICATION_FRACTION_EXCEEDED` 拒(cohort 比 0.128>0.10,47/70 窗超 per-window 帽)。Scope 编译开发应读作「同 impulse 族、不同 Program 几何/校验器命运」,不得把本跑与 GunPoint 并成第二 hampel 正例。

**成本**:LLM 10/40;fit 12/200;墙钟 runner 账 229.2 s / 进程 285.5 s(帽 5400 s,未触发 `COMPUTE_BUDGET_EXCEEDED`);下载 0。

**提交**:runner 最小入口 `--conf-dev-run`(`--dataset` 默认 ECG200);隔离工件 `t6_cls_conf_dev_ecg200.json/.md`;本节。`methods/`/`runtime/`/`contracts/`/`operators/` 零改动;未触 `data/ucr_conf_downloaded/`;他线文件未碰;未跑全仓 pytest。解释器 `D:\Anaconda_envs\envs\project\python.exe`。

### S1a-r2 收口:HEADROOM_WITHOUT_LEGAL_TRANSFER_PATH;两簇可学均不足通道;9 单元池无 2+1 重组案(2026-08-26 16:xx,执行方)

**判定**:**HEADROOM_WITHOUT_LEGAL_TRANSFER_PATH**(development;不升级课程;S1b 仍冻结)。0 LLM / 0 fit / 0 注入重跑;密封 oracle 只读重聚合;r1 工件未覆写。解释器 `D:\Anaconda_envs\envs\project\python.exe`。提交前 HEAD 将记入工件 `git_head`。

**三条合法性规则(出处)**:(a) Target-local 禁跨单元进 Fast——正典 AGENTS.md:174-175 / 76-81 / 184-191;现役卡 applicability 仅 `task_kind`(shared harness :610-613 → method.py:89-105 → retrieval.py:278-282),T1 惰性闸口拦不住非经验卡,照跑测的是宽 Scope bug。(b) 可授权 Source 证据须 Support 与 delayed 均为 `classify_relation==POSITIVE`(method.py:742-757 / 1466-1492;online_loop.py:201-204;experience_memory.py:434-439;阈值 0.005,禁自造;oracle 仅有拼接 held-in 池,作两门代理)。(c) 仅未引导正例可授权新 Shared TRY(source_skill.py:217-257);未引导=Fast 无同 Program 族 TRY/capability 卡。

**9 单元可学性**(oracle 集算子 × 现役批准语义):GPA hampel LEARNABLE(held-in +0.375);GunPoint hampel LEARNABLE(+0.467);Herring hampel **HELDOUT_ONLY**(held-in 0 / held-out +0.046875,`s1_oracle/Herring__impulse_v2.json:579`);ECG200 repair_burst **HELDOUT_ONLY**(held-in 0 / held-out +0.04);Toe / Lightning2 repair_burst LEARNABLE;Wine/Ham N/A;GunPoint burst iqr HELDOUT_ONLY。**hampel 簇可学 2/3**(独立家族 1=GunPointFamily,GPA↔GP pattern_view 字节相等);**repair_burst 簇可学 2/3**(ECG200 与 Herring 同型:考官可见、学生不可学)。

**正反序合法时间线**:携带禁行后,正序单元 3(GunPoint)后 Slow 可写 Scope-v1 候选(形式 2 正例、交非空)但独立家族=1;authorization_audit LOO min=1<2 → TRY 不授权 → T1 inert → Fast 不可见。反序正向在 4/6,候选成型即课终,同样无 Fast TRY。两序「首次合法 Fast 可见分歧」= **不存在**。每转移合法性列见 `s1a_r2_legal_treatment_audit.md`。

**重组搜索**(现有 9 单元、不扩池):无满足「≥2 可学正例在前、≥1 可学匹配场在后」的 6 单元排列。hampel 匹配场=自身且独立家族 1;burst-repair 可学仅 Toe+L2,Scope 交后无第三可学匹配场。不存在待批重组案。

**S1b 规格**(文字,未写代码):runner 层域绑定——cell 构建给 Target-local 打 domain;跨单元携带过滤异域 Target-local;Source-derived 按 Scope v1 五轴放行。此为协议合规(正典已写,防把宽 Scope bug 当考题),不是改菜单/预算/oracle。methods 第③步仍是长期 Scope 编译;本书不动 methods/;③ 落地后拆 runner 墙。

**成本**:0 LLM / 0 fit / 墙钟 0.02s / 下载 0。**义务**:未跑任何适应臂、未重算 oracle 数值;`methods/`/`runtime/`/`contracts/`/`operators/` 零改;他线文件未碰;未跑全仓 pytest。

**书外**:现役 LOO 要 3 个未引导正例才授权 TRY(r1 把 2 当成够);ECG200 burst-repair 亦 HELDOUT_ONLY,仲裁「改建于该簇」不被密封 held-in 支持;Toe hampel held-in 正但非 oracle 集且 period_change_score 与 GPA/GP 不合。

**提交**:`artifacts/functional/e2/s1a_r2_legal_treatment_audit.json/.md`(新,不覆 r1);runner `--legal-r2`;本节(他书未提交台账条目一并入库,未删改既有正文)。

### S1-v2 终掷(课程 v4)正序 r1:第三次 `TREATMENT_EMPTY`;按终掷硬帽转为**系统性结论**,全停呈机制重审(2026-08-28 00:4x,执行方)

**判词:`TREATMENT_EMPTY`**(development)。**终掷硬帽生效**:三次处理组落空不是第四次重排的理由,而是一个结论——S1-v2 全停,机制重审。r2 不发。**成本 115/280 LLM、77/900 fit、3206.0 s / 21600 s、0 下载**;三课累计 306 LLM(总帽 500 内)。

**三答执行确认**:**①** 采样重复授权 —— 已写入冻结件(SIGNAL 才续 r2);本跑负判,未续。**②** 判据第三项 live Support 通过率 —— 已入件并附实证账:GPA 冷发现 3/3 且 Support 3/3;GPMvF-impulse 3/4(M-1 对半实证);GunPoint-impulse 未实测(正因如此仅作备份);PowerCons-impulse 冷发现 2/4、**Support 贴线 0/2**(+0.0714 / +0.0357)。**③** 备份产例 C —— 已入件,语义如实写明:C 是"第二个正例的第三次机会",不是第三个正例;任一边界凑满 2 个未引导正例即编卡,此后 C 的正例因卡已在 A5 视野属**受引导、计零**(既有 UNGUIDED 规则,非本课特设)。

**冻结件 v4**(`s1v2_course_freeze_v4.json/.md`,定名 **discovery-and-support-reliable development curriculum**,8 单元):GPA(产例A)→ BeetleFly-impulse(identity A)→ GPMvF-impulse(产例B)→ GunPoint-impulse(**备份产例C**)→ [边界]→ GPOvY(受益强 5.00×)→ **PowerCons-impulse(受益弱)**→ Herring(HELDOUT_ONLY)→ BirdChicken-burst(identity B)。Δ_material = 1/20 + 1/26 = **0.088462**。必写注记全入件:终掷硬帽、判据第三项与实证账、备份产例语义、**家族注记**(卡证据若成型即 GPA+GPMvF 同属 GunPoint 族、强受益 GPOvY 亦同族 → development mechanism 级,不得报跨族)、**PowerCons 密封 5.00× 与 live 贴线的落差**(归因提案参数绑定:密封 oracle 按自调参数打分,臂按契约绑定提案)、以及沿 v3 的排除令语义/前曝光/采样重复/自然自举对照。

**本跑发生了什么(A5 三个产例位逐一)**:**产例A GPA** —— A5 只提了 `repair_localized_level_shift`(level_shift 族),**verifier_rejected**,全程未提 hampel;同一单元同一协议下 **K0-fixed 提了 hampel 并拿到 +0.2690**。**产例B GPMvF** —— A5 提 hampel,Support **+0.1905 POSITIVE**、delayed +0.0526、`LOCAL_ACTIVE`、获批、部署 **+0.1867**。**产例C GunPoint-impulse(备份)** —— A5 只提 `repair_local_level_excursion`(level_shift),**verifier_rejected**,随后选 identity。于是**未引导正例 1 个**,八个边界审计全判 `fewer_than_2_distinct_unguided_positive_tasks`(前两个边界更早,判 `no_program_family_met_the_supply_rule`),**卡从未编译**,受益场注入 **0/2**(ITT)。

**四臂读数(ITT,8 单元累计)**:Static regret +1.2267 / 均值 0;A3-reset **+1.2267** / +0.0000 / 11 probe / 38 LLM / 19 fit / **0 次非 identity 部署**;**A5-online +1.0400** / +0.0233 / 12 probe / 38 LLM / 23 fit / 1 次部署;K0-fixed **+0.7710** / +0.0570 / 12 probe / 39 LLM / 27 fit / 2 次部署。**四臂 harm 全零、worst-class 全零**。**门核算**:A5 对 A3 regret 差 **+0.1867 > Δ 0.0885**(数值上过!)但**归因门不过**——零产卡零注入,该差值来自 A5 自己在产例B 的同单元学习,与"课程内自产知识跨单元复利"无关,按判词逻辑不计信号;对 K0 **−0.2690**(更差)。成本门 probe 差 **−1**,未过。**训练计算效率**:fit 墙钟 A5 965 s / A3 1068 s / K0 1102 s / Static 52 s,三适应臂无材料差异;time-to-threshold 不可算。**分层预测**无法检验(两受益场零注入)。

**系统性结论(三课汇总,这是本条的承重内容)**:把三课的 **A5 产例位**拉平看——v2 两位(0 次提 hampel)、v3 两位(2 次提、1 次过 Support)、v4 三位(1 次提、1 次过 Support)——共 **7 个产例位:提出 hampel 3 次(43%)、过 Support 双门 2 次(29%)**。供给档要 **2 个独立未引导正例**;在每位约 0.29 的成功率下,三产例位凑满两个的概率约 **0.19**。**这不是课程构成问题,是价格与供给率不匹配**:重排课程、升级判据、加备份产例三招都试过,三次都空。**首处故障因此从"提案语义"进一步收窄为"提案语义 × 供给档价格"**:通道(W-1 读端 / G-3 条件化与否决 / P0 产端)三书已分别证通,可达性链在 P0 上 7/7 全通;缺的是**能稳定产出两个独立正例的提案器**。

**书外发现**:(1) **level_shift 偏好仍在烧提案位**:v4 的产例A与C,A5 唯一的非 identity 提案都是 level_shift 族,且**两次都被 verifier 拒**——即冷 proposer 不仅偏爱错误族(C40 起已记),而且偏爱到连校验都过不了,提案位等于白丢。这是比"没提对族"更具体的一条:**提案器的候选合法性也不达标**。(2) **同单元跨臂对照是最干净的采样证据**:v4 产例A 上 K0 提了 hampel(+0.2690)、A5 没提;两臂同底物同协议同预算,唯一差别是 LLM 采样。臂间差异当前由采样方差主导这一点,现在有了单单元级证据。(3) 备份产例 C 的语义在本跑未被触发(卡未成型),其"受引导计零"分支仍未获实测,如实记。(4) PowerCons 作受益弱场的密封-live 落差已入件;本跑它在四臂上全 identity、regret +0.1333,与"贴线"预期一致。

**提交**:`evaluation/functional/run_e2_s1v2_forward_course.py`(v4 冻结 + 输出路径 + LLM 帽 280)、`artifacts/functional/e2/s1v2_course_freeze_v4.json/.md`、`artifacts/functional/e2/s1v2_v4_forward_run1.json/.md/.checkpoint.json`、本节。`methods/`/`contracts/`/`runtime/`/`operators/` 零改;`AGENTS.md`/`README`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP`/设计稿 未碰;密钥零出现;未跑全仓 pytest。

### S1-v2 课程 v3(discovery-reliable)正序 r1:`TREATMENT_EMPTY`,但性质与 v2 完全不同——管线全通,卡差一个正例(2026-08-27 23:2x,执行方)

**判词:`TREATMENT_EMPTY`**(development)。r2 **不发**(停发规则照旧;是否补发采样重复由仲裁定)。**成本 95/250 LLM、73/900 fit、4228.1 s / 21600 s、0 下载**。仲裁批案 A 只改产例侧,其余(判分/ITT/材料门 Δ=0.102632/预算/对半协议/K0 纯度/oracle 纪律/checkpoint)全部沿 r2 冻结与原书。

**冻结件 v3**(`s1v2_course_freeze_v3.json/.md`):课程定名 **discovery-reliable development curriculum**(产例按历史冷发现率实证选取)。正序 = GPA(产例A,冷发现 2/2)→ BeetleFly-impulse(identity A)→ PowerCons-impulse(产例B,2/3)→ **[边界]** → GPOvY(受益强 5.00×)→ GPMvF-impulse(受益弱 2.00×)→ Herring(HELDOUT_ONLY)→ BirdChicken-burst(identity B)。**排除令语义修订注记已入件**:护住"卡是课程内自产"的约束不是"该单元曾在别处当过源",而是 (i) K0 空、(ii) 受益单元≠产例单元;产例上课程内重挣正是课程该做的事,不破卡自产性(sol 已核)。受益单元前曝光注记沿 r2;`replicate_kind=sampling`(注入无 RNG,出处已引);**GunPoint 家族重叠如实注记**(GPA 与 GPOvY/GPMvF 同名族、单元级不重叠,不得报作跨族能力)。

**与 v2 空场的性质区分(本条是本轮主结论)**:v2 的空场是**结构性**的——受益侧根本不存在(池内可学 hampel 单元全被占用或 HELDOUT_ONLY),且产例上臂**从未提出**目标族;v3 的空场是**采样差一发**——管线每一环都通了,只是产例 B 的 2/3 抽签这次没中。证据:**产例 A(GPA)一轮命中**——A5 自提 `hampel_extreme_deviation`,Support **+0.4500 → POSITIVE**,delayed **+0.4000**,Episode `LOCAL_ACTIVE`,Skill 获批,部署 held-out **+0.2690**(regret −0.0063,略优于菜单 oracle);**产例 B(PowerCons-impulse)也提出了正确的族**——A5 一轮自提 hampel,但 Support 只读到 **+0.0357**,经 `classify_relation` 判 **CONFLICT → EPISODE_ONLY**(聚合过材料线但逐序列有害),不计未受引导正例。于是七个边界的供给档审计一致给出 `fewer_than_2_distinct_unguided_positive_tasks`(未受引导正例 **1**、受引导 0、对立 0,族=`hampel_filter`),**卡 1/2 未编**,受益场注入 **0/2**(ITT)。**产例实证率更新:GPA 3/3、PowerCons-impulse 2/4。**

**四臂读数(ITT,七单元累计)**:Static regret +0.8200 / 均值 0 / 0 probe;A3-reset **+0.6333** / +0.0267 / 11 probe / 33 LLM / 21 fit / 1 次非 identity 部署;**A5-online +0.5510** / +0.0384 / 9 probe / 32 LLM / 23 fit / 1 次部署;K0-fixed **+0.3643** / +0.0651 / 6 probe / 30 LLM / 22 fit / 2 次部署。**三臂 harm 全零,worst-class 全零**(v2 的那次 A5 harm 未复现)。**门核算**:A5 对 A3 regret 差 **+0.0823 < Δ_material 0.1026**(未过);对 K0 **−0.1867**(更差)。成本门 probe 差 **+2 ≥ 可转化单元数 2**(数值上过),但**归因门不过**(零产卡、零注入),按判词逻辑归因优先,故不计入信号。**训练计算效率**:consumer fit 墙钟 A5 1309 s / A3 1385 s / K0 1388 s / Static 128 s,三适应臂无材料差异;time-to-threshold 不可算。

**分层预测对表**:预注册"A5 优势集中强余量受益场"——**无法检验**,因为两受益场 A5 均零注入;更值得记的是 A5 在两个受益单元上 **0 probe、0 提案、llm 3**(弃权),而 A3 在弱受益场自行挣到 hampel(+0.1867)。A5 单元 1 挣得的 Target-local hampel 按设计带域戳、不跨单元生效,故受益场对 A5 而言等同冷启动——**这一轮 A5 比 A3 更沉默,且因此丢掉了 +0.1867**,如实记为负面读数。

**书外发现**:(1) **产例 B 的失败点从"提案"移到了"Support 分级"**:+0.0357 聚合为正却因逐序列有害判 CONFLICT。这是 `classify_relation` 的既有正确行为(聚合正、逐序列害不得扩权),但它意味着**"冷发现率"这个先验只覆盖到"提出",没有覆盖"过 Support 双门"**——产例选择判据还需要第三项:该单元在对半协议下的 **Support 通过率**,而不只是历史 earn 率。(2) K0-fixed 连续两跑 regret 最低,两跑都靠在 GPMvF 上自行挣到 hampel;在处理组为空时四臂对比仍不具解释力,不得读作"K0 优于 A5"。(3) 供给档在"1 个正例"处稳定拒绝七次,零凑数,与 P0 单测一致。

**提交**:`evaluation/functional/run_e2_s1v2_forward_course.py`(v3 冻结 + 输出路径切换)、`artifacts/functional/e2/s1v2_course_freeze_v3.json/.md`、`artifacts/functional/e2/s1v2_v3_forward_run1.json/.md/.checkpoint.json`、本节。`methods/`/`contracts/`/`runtime/`/`operators/` 零改;`AGENTS.md`/`README`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP`/设计稿 未碰;密钥零出现;未跑全仓 pytest。

### S1-v2 正序第一跑:`TREATMENT_EMPTY`;产例未产,课程内自产知识为零,按预注册停发第二跑(2026-08-27 19:4x,执行方)

**判词:`TREATMENT_EMPTY`**(development)。按设计稿"课程内未产生任何 Fast 可见知识 → **立即停**"执行,**第二跑不发**。**成本 96/250 LLM(总帽 500 未过半)、68/900 fit、3284.6 s / 21600 s、0 下载**。

**两项裁定执行确认**:**(a)** GPOvY(强,对半 5.00×)与 GPMvF-impulse(弱,2.00×)作双受益单元,两者机器 Scope 均命中产例 19 叶交集且 held-in LEARNABLE;双源单元 GPA/PowerCons-impulse 维持排除;**前曝光注记已强制写入** `s1v2_course_freeze_r2.json/.md` 与 run1 工件正文(两受益单元可转化性系 PS-2/W-1/M-1 已知,臂内无泄漏,新颖主张 = 课程内自产 + ITT 端到端复利)。**(b)** regret 门 `Δ_material` 改为双受益单元对半材料线之和 = 1/20 + 1/19 = **0.102632**,常数已代入冻结件;成本门(可转化单元均省 ≥1 probe)不动。

**课程冻结件 r2**(`s1v2_course_freeze_r2.json/.md`,r1 停报件保留):PowerCons__burst_cls2(产例A,5.00×)→ BeetleFly__impulse_v2(identity A)→ GunPoint__impulse_v2(产例B,3.00×)→ **[Slow 边界预计产卡]** → GPOvY__impulse_v2(受益强)→ GPMvF__impulse_v2(受益弱)→ Herring__impulse_v2(HELDOUT_ONLY)→ BirdChicken__burst_cls2(identity B)。七单元 + 一边界(书内"8"含边界步);预计首分叉 = 位置 4。

**首处断裂在产例侧,不在传递侧。** 七个边界的供给档审计**每一次**都判 `fewer_than_2_distinct_unguided_positive_tasks`,且合格 Episode 始终只有 **1 条、族为 `outlier_iqr`**:A5 在产例 A 上自提并部署了 outlier_iqr(+0.0444),在产例 B(GunPoint impulse)上**一无所获**。课程选产例的依据是"hampel 是其密封答案且对半余量 ≥2×",而**臂在两个产例上都没有提出 hampel**——余量说的是"若探到该族则读得出",不是"会探到该族"。于是供给卡从未编译、候选从未入池、受益单元零注入(ITT 主分析:0/2),`TREATMENT_EMPTY` 成立。这与本线既有诊断同族:S1c 漏斗 11/15 "未提出"、PS-1 `NO_PROPOSAL_SHIFT`、K 消融 `proposal_semantics_insufficient`——**瓶颈是提案语义,不是供给通道**(供给通道 W-1/G-3/P0 三书已分别证读端、条件化、产端均通)。

**四臂读数(ITT,七单元累计)**:A3-reset regret +0.9474 / 均值 gain +0.0000 / harm 0 / 8 probes / 30 LLM / 15 fit;**A5-online** +0.9029 / +0.0063 / **harm 1** / 10 probes / 33 LLM / 22 fit;K0-fixed +0.7607 / +0.0267 / harm 0 / 12 probes / 33 LLM / 24 fit;Static +0.9474 / 0 / harm 0 / 0 probes。A5 对 A3 的 regret 差 +0.0444 **未过** Δ_material=0.1026;对 K0 为 **−0.1423(更差)**;成本门亦未过(probe 差 −2,A5 反而多探)。**训练计算效率**:consumer fit 墙钟 A5 1038 s / A3 1095 s / K0 1053 s / Static 89 s,三适应臂无材料差异;time-to-threshold 不可算(A5 未达任何质量阈)。**A5 唯一一次 harm 事件**来自单元 1 的 outlier_iqr 部署(worst-class −0.0667),即课程内自产的**非目标族**知识本身带来的代价。

**书外发现**:(1) **"异注入 seed"在本底物上不可实现**——`run_e2_t6_cls_op_shared_harness.py:3896-3901` 已入册:该族注入无 RNG 可播种(固定符号模板 + 确定性等距 fit/support 划分),"新注入种子"是虚构。故两跑只能是**采样重复**(唯一随机源是 Fast Agent);本书如实改标 `replicate_kind=sampling` 并把出处写进工件,不伪装成注入重复。第二跑因 `TREATMENT_EMPTY` 未发,该问题本轮不构成实际偏差,但**反序/重复书必须先解决重复语义**。(2) K0-fixed 本跑 regret 最低(+0.7607),来自它在 GPMvF 上自行挣到 hampel(+0.1867)——臂间差异当前由 agent 采样方差主导,**在处理组为空时四臂对比不具解释力**,不得读作"K0 优于 A5"。(3) 产例选择判据需要升级:现依据"密封答案 + 余量",应加一条**可提出性**先验(该族在该单元曾被冷 proposer 自然提出过的实证),否则产例侧会继续空转。(4) 供给档在**零知识**下的行为正确:七次审计全部给出同一条明确拒绝理由,没有任何一次凑数产卡。

**提交**:`evaluation/functional/run_e2_s1v2_forward_course.py`(r2 冻结 + 四臂课程驱动 + `--finalize`)、`artifacts/functional/e2/s1v2_course_freeze_r2.json/.md`、`artifacts/functional/e2/s1v2_forward_run1.json/.md`、本节。`methods/`/`contracts/`/`runtime/`/`operators/` 零改;`AGENTS.md`/`README`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP`/设计稿 未碰;密钥零出现;未跑全仓 pytest。

### S1-v2 Part 0 停报:`COURSE_NOT_CONSTRUCTIBLE`;本地池无第五个可学 hampel 单元,未烧一次 LLM(2026-08-27 18:2x,执行方)

**判词:`COURSE_NOT_CONSTRUCTIBLE`**(Part 0 算术预检,**0 LLM / 0 fit / 0 下载**)。按设计稿"推演不通 → 停报,不烧 LLM"执行,四臂正序 ×2 **未发车**;LLM 500 帽、fit 900 帽分文未动。

**对半协议余量重算(全池,复用 M-1 role-concat 算术:Support=r1s+r2s、delayed=r1d+r2d,双门保留)**:清 2× 的 hampel 单元共 **6 个**——GPA 7.00×、GPOvY 5.00×、PowerCons-burst 5.00×、PowerCons-impulse 5.00×、GunPoint-impulse 3.00×、GPMvF-impulse 2.00×(季度余量分别为 3.75/4.15/2.22/2.44/1.40/1.35,对半协议一律抬升,与 M-1 因果结论同向)。其余带 hampel 的单元(BeetleFly-burst、GPMvF-burst、Herring、MoteStrain、SonyAIBO2-burst、ToeSeg2-burst)**全部 HELDOUT_ONLY**。

**产例侧成立**:排除四个他书已用单元(GPA/PowerCons-impulse 双源、GPOvY PS-2/W-1 考场、GPMvF-impulse M-1)后,恰剩 **PowerCons__burst_cls2(5.00×)与 GunPoint__impulse_v2(3.00×)** 两个未被占用的 ≥2× 可学 hampel 单元,`task_episode_id` 互异,五轴 Scope 交集 **19 叶非空**——课程内自产供给卡的证据对是存在的。

**受益侧不成立(第一处未满足条件)**:候选受益单元必须"机器 Scope 匹配 ∧ held-in 可学"。剩余六个带 hampel 单元里只有 `GunPointMaleVersusFemale__burst_cls2` 机器匹配,而它 **HELDOUT_ONLY**——"考官可见、学生不可学",held-in 反馈按构造批不了该族。**若照书内"取机器匹配且余量最高者"的宽松条款收下它,等于把 `NO_TRANSFER` 预先写死在池的性质上而非 Harness 的性质上**,正是 S1a-r2 已经踩过、而本预检存在的目的就是拦住的那一类。故本书把宽松条款读作"放宽余量档,不放宽 held-in 可学性",并如实报停。

**反事实(供主线裁决,已入工件)**:四个被排除单元**若释放则全部合格**为受益单元——GPA 7.00×、GPOvY 5.00×、PowerCons-impulse 5.00×、GPMvF-impulse 2.00×,四者机器 Scope 全匹配且 held-in LEARNABLE。**代价分级**:释放任一双源单元会让卡变成"部分带资进场",直接消解本考"课程内自产 vs 带资进场"的区分;释放 GPOvY 或 GPMvF-impulse 只损失与已读过该单元的书的独立性,代价小得多。**本书不自行释放任何排除项**——排除名单是设计稿冻结件,改它属方法决策。

**书外发现**:(1) 首版选课把 `GunPointMaleVersusFemale__burst_cls2` 当受益单元选中并判 `S1V2_COURSE_FROZEN`,其 menu oracle 实为 `outlier_iqr`、census HELDOUT_ONLY;若照此发车,六单元全程会在预计首分叉位置产出一个**注定否决**的读数,而判词会写成 `NO_TRANSFER` —— 预检收紧后当场翻为停报。**这是本轮最有价值的一次自查:一个"能跑通"的课程不等于一个"能证伪"的课程**。(2) 同一次收紧还发现治理槽会把 BeetleFly 的 impulse 与 burst 两个视图同时选进 identity 与 HELDOUT_ONLY 槽,已改为按 dataset 去重。(3) 若课程当时成立,`Δ_material = max_u(1/n_slice_u)` 会是 **1/7 ≈ 0.143**(由 GunPoint-impulse 对半协议最粗切片 n=7 决定)——这个 regret 门比任何单元的实际 held-out 余量都苛刻,说明**即便课程构成,材料门本身也需要主线复核**:要么课程避开 n 极小的单元,要么门的定义改用加权而非 max。此点单列供裁决。

**提交**:`evaluation/functional/run_e2_s1v2_forward_course.py`(Part 0 冻结件生成器;live 入口读冻结件的判词把关,课程未冻结时拒跑)、`artifacts/functional/e2/s1v2_course_freeze.json/.md`、本节。`methods/`/`contracts/`/`runtime/`/`operators/` 零改;`AGENTS.md`/`README`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP`/设计稿 未碰;密钥零出现;未跑全仓 pytest。

### P0 收口:供给档在**产卡端**落地;`SUPPLY_TIER_PRODUCTION_REACHABLE`(2026-08-27 18:0x,执行方)

**判词:`SUPPLY_TIER_PRODUCTION_REACHABLE`**(infrastructure;确定性,**0 LLM / 0 fit / 0.3 s / 0 下载**)。仲裁核码定案兑现:W-1 只接通读卡端,产卡端此前唯一的正向出口是 TRY 档(LOO 使其实要 3 个未受引导正例),两源课程照跑必然要么产不出可消费的卡、要么产出权限过强的旧卡。本书补上**供给档出口**。

**分层实现(全在 evaluation 层,methods 零改)**:`agentic/source_skill.py` 新增 `SUPPLY_TIER_MIN_DISTINCT_TASKS=2` / `SUPPLY_CARD_KIND`、`supply_tier_audit`、`five_axis_scope`、`supply_applicability`、`build_supply_card_payload`、`compile_supply_tier`。**两档共用同一套子句词汇**(只算未受引导证据、对立证据一票否决、distinct `task_episode_id` 为计数单位),**唯一差别是计数与是否套 LOO**:供给档 2 个、不套 LOO;TRY 档 `authorization_audit` 一字未动,仍是 LOO 地板。编译**全程机械模板填充,零 LLM 撰文**——理由入码:TRY 子句是一个论证,需要 Slow 撰文;供给候选不是论证,是"一个 Program 加上它被挣到的 Scope",模板化正好堵住模型悄悄放宽主张的那一处。卡形固定为 `supplies_candidates=true / grants_execution=false / requires_target_support=true`,Frozen program = 共同 Program(参数取证据侧共同缺省),Scope = 五轴交集(task_kind/consumer/metric + 持久化 pattern 叶交 + program geometry)。**保守条款依据**:同族存在未解决 NEGATIVE 即不产卡,直接沿用 `authorization_audit` 既有的"对立证据从任一出处都阻断"(source_skill.py:225-226);受引导正例计零,沿用既有 UNGUIDED 规则(:219-221);另加两条机械拒绝——身份轴不一致、pattern 交为空。

**六项聚焦测试**(新 `tests/functional/test_supply_tier_compiler.py`,13 项,0 LLM):(a) 2 个合格 Episode → 产卡且四权限字段逐字正确 + 模板对输入顺序不变;(b) 1 个不产、同一 task 两次仍算 1;(c) 受引导正例计零;(d) **两档互不干扰证明**——同一 2 正例下 `authorization_audit` 判 `does_not_survive_leave_one_out`、`authorized_try_operators` 为空,而供给档产卡,且 `build_skill_payload` 的 TRY 载荷不含 `authority` 块;(e) 卡过 `load_skill_entry`、Scope AST 双向机器可判(在域命中/离域不命中)、T1 谓词对其正确无操作、W-1 读端 `_supply_rung_candidates` 消费成功、离域零供给,另加未契约叶被丢弃并记录;(f) 同族 NEGATIVE 阻断、异族 NEGATIVE 不阻断、身份轴不一致、pattern 交空各产正确拒绝理由。**回归 121 passed**(W-1/G-3 供给测试 + guard 12 测 + T1 + agent 套件 + risk/source/ordering card)。**h0 锁无需重生成**:本书零 methods 改动,`agentic/source_skill.py` 不在 dependency_shas,`compile_snapshot(verify_lock=True)` 通过,`harness_content_sha` 仍 `53b1c803…0654f`。

**生产路径可达性(真实双源,7 环全通)**:输入 = PS-0 重挣 GPA `ps0_srcA_1`(Support +0.4000 / delayed +0.4000)与 PS-0c 重挣 PowerCons `ps0_srcB_4`(+0.0714 / +0.5000),各带 21 个持久化 pattern 叶,均为 A3-reset 空店所挣故未受引导。链路:① 持久化 Episode 读取 → ② 供给档审计+模板编译(`source_skill.py:566`)→ ③ 卡形八项断言(`source_skill.py:466`)→ ④ **真实 EditController apply 与重编译**(`run_e2_s1_curriculum_four_arms.py:1058`;PS-1 当年就是在这一环被 AST 形状打回)→ ⑤ `resolve_harness_view(fast)` 在域送达、离域withhold(`retrieval.py:241`)→ ⑥ `_supply_rung_candidates` dry 入池、离域零供给(`fast_agent.py:386`)→ ⑦ 双门在该路径上存在(`online_loop.py:435` Support 门、`online_loop.py:833` delayed 门,后者即 W-1 同权修)。**编译结果**:17 叶机器 AST(16 pattern 叶 + task_kind),4 个未契约叶如实丢弃并记录。工件 `artifacts/functional/e2/p0_supply_tier_reachability.json/.md`。

**书外发现**:(1) 首版模板对输入顺序敏感(provenance 串与 sources 列表随读入序变),被 (a) 的确定性断言抓到,已改为按 `task_episode_id` 排序后编译——编译产物是证据的函数,不是读取顺序的函数。(2) schema-代码叶集漂移仍在:`contracts/observables.OBSERVABLE_FEATURES` 是 `contracts/schemas/observable_feature_v1.json` 的超集,四个叶(`level_only_post_shift_support_sufficient`、`level_region_end_fraction`、`level_region_fraction`、`outlier_region_end_fraction`)只能进 `scope_v1` 记录不能进机器 AST;本书照 PS-1 先例丢弃并留痕,未修契约。(3) 本卡 17 叶比 W-1 手写卡的 16 叶多一叶(`period_change_score`),因为机械交集不做人工取舍——两源在该叶上确实一致。(4) 供给档目前只在同族单卡时出卡:两族同时合格判 `more_than_one_family_qualified` 不出卡,留给 Slow,不由模板替它选。

**提交**:`evaluation/functional/task_episode_harness/agentic/source_skill.py`、`tests/functional/test_supply_tier_compiler.py`、`evaluation/functional/run_e2_p0_supply_tier_reachability.py`、`artifacts/functional/e2/p0_supply_tier_reachability.json/.md`、本节。`methods/`/`contracts/`/`runtime/`/`operators/` 零改;`AGENTS.md`/`README`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP` 未碰;密钥零出现;未跑全仓 pytest。

### G-3 收口:三场小课程;条件化两场全过、迁移场未复现;判词 `FIELD1_NO_CONVERSION`(2026-08-27 16:0x,执行方)

**总判:`FIELD1_NO_CONVERSION`**(development-mechanism, pilot)。16 跑全部完成。**成本 116/150 LLM、86/120 fit、4050.8 s / 10800 s**;0 下载。**阈值/授权语义零改动**:卡不重编(W-1 同一张双源 hampel 卡)、MATERIAL、LOO、T1 谓词、incumbent 规则、`classify_relation` 三档全未动;methods 只多一处 inspect 层接线。

**Part 0(inspect 层一行修)**:`fast_agent.prepare` 把 inspect 阶段按 W-1 同法单独包一层(捕获集与外层相同,仅在视野里确有供给卡时降级为"agent 零贡献"继续)。判据依据:`inspected_regions=()` 时 `verify_candidate` 的 `outside` 恒 False,故**无须任何"退化全窗口"决定**;修改分数帽与卡自带 region 参数照旧生效。配 2 项聚焦单测(inspect 抛错时供给候选仍入池并获测;同形无旗卡仍失败)。h0 锁机械重生成:`harness_content_sha` **不变** `53b1c803…0654f`,`runtime_bundle_sha` 0d66c4d3…→c3427b4e…。**live 兑现**:g3_f1_a5_1 与 a5_2 各有一轮 `supply_without_agent_program=True`——**W-1 遗留的"去耦合未 live 观测"缺口在本书补上**。

**Part 1 场地机械选定(0 LLM,先于任何 live 跑)**:用卡的 16 叶机器 AST 对 45 份密封单元逐一求值(密封件只作考卷,不入任何臂视野)。**Scope 判别力读数:45 单元中仅 8 个命中 WHEN,其中 6 个的密封答案正是 hampel**——WHEN 轴不是橡皮图章。选定:**场① `GunPointMaleVersusFemale__impulse_v2`**(16/16 命中、hampel 为 oracle、ps0b ROBUST_LEARNABLE 1.35×;同名族 GunPointFamily 与源 A 同族,如实注记);**场② `ShapeletSim__impulse_v2`**(缺 3 叶:estimated_level_offset / estimated_region_start_fraction / level_excursion_score,oracle=identity);**场③ `ToeSegmentation2__impulse_v2`**(16/16 命中但 oracle=identity、hampel held-out −0.023)。**书内点名的 Wine 落选**:Wine 缺 `estimated_region_start_fraction`,机器 WHEN 不匹配,按书内后备条款改用"匹配但 oracle≠hampel"的全池扫描结果。**排除规则**:卡自身两个源单元(GPA、PowerCons impulse)与 ps2p 考场单元(GPOvY)一律不得作场①——PowerCons impulse 余量最大(2.44×)但正是源 B,用它等于自证。

**三场读数**:场②(Scope 不匹配)**注入 0/4**,四跑全 identity 部署、harm 0 —— **无 Scope 泄漏**;场③(Scope 匹配但 Target 不同意)**注入 4/4、获测 15 探针、材料正 0/4、部署 hampel 0/4**,四跑全 identity、harm 0 —— **否决成立**;场①(Scope 匹配正向)A5-scoped 注入 3/4,但**材料正 0/4、部署 hampel 1/4**(且那一跑 `g3_f1_a5_4` 的 hampel 来自 **agent 自提**而非注入),均值 held-out **+0.0467**;A3 对照同为 **+0.0467**(a3_2 自行提出 hampel,+0.1867)。**成本对照**:A5 26 LLM / 28 fit / 7 探针,A3 25 LLM / 17 fit / 4 探针——A5 更贵而不更好。

**判词按预注册取 `FIELD1_NO_CONVERSION`**:两条治理红灯(FIELD2_SCOPE_LEAK / FIELD3_VETO_FAILED)均未触发,但场①未达"≥2/4 转化且正收益差"。**结论一句话**:供给机制的**条件化与数据主权已被两场独立证实**(该出现时出现、不该出现时不出现、出现了也可被 Target 否决),但"Scope 匹配即转化"**未在 GPOvY 之外复现**——+0.2127 目前仍是单单元读数。

**书外发现**:(1) GPMvF 上 hampel 的 Support 在 A5 三次注入中一次都未记为材料正:a5_1 探针读 +0.1818 却被 delayed −0.10 否决(否决链正确),另两次未过 Support 门——与 ps0b 记的 GPMvF 余量仅 1.35×、held-in 切片 0.18/0/0.20/0.22 一致,即**该单元本身的确认面比 GPOvY(4.15×)弱得多**,场①的阴性更像单元质量而非通道故障。(2) A3 在 GPMvF 上自行提出并部署了 hampel(a3_2 +0.1867),说明该族在此单元对冷 proposer **并非不可达**——这正是"供给机制的边际价值"应当被质疑的地方,如实记。(3) 场③ 15 次探针零转化、零 harm,是 Target 数据主权最干净的一次读数。(4) 执行顺序中途改为**治理优先**(场②③ 先跑,场① A5/A3 交错):理由是墙钟若截断,应当截在正向场而非红灯场;各跑状态相互独立,顺序非实验变量,如实记。

**提交**:`methods/ttha/fast_agent.py`(Part 0)、`methods/ttha/harness/h0/snapshot.lock.json`(机械)、`tests/functional/test_supply_rung_wiring.py`(+2 项)、`evaluation/functional/run_e2_g3_three_field_course.py`(新 runner:`--select` / `--run` / `--resume` / `--finalize`)、`artifacts/functional/e2/g3_three_field_course.json/.md`、本节。`AGENTS.md`/`README`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP` 未碰;`contracts/`/`runtime/`/`operators/` 零改;密钥零出现于任何提交物;回归 105 passed;未跑全仓 pytest。

### W-1 收口:supplies_candidates 生产接线;`SUPPLY_RUNG_PRODUCTION_CONFIRMED`,GPOvY held-out **+0.2127 ×4/4**(2026-08-27 13:1x,执行方)

**判词:`SUPPLY_RUNG_PRODUCTION_CONFIRMED`**(development-mechanism, pilot)。12 跑协议 ps2p_run1..12,0 下载。**成本 74/150 LLM、48/160 fit、3043.4 s / 12600 s 墙钟**。**阈值/授权语义零改动**:MATERIAL=0.005、LOO、T1 谓词、`_incumbent_after_delayed`、`classify_relation` 三档、`risk_skill` 计数全部未动;谁得 `supplies_candidates` 旗仍由外部编译决定,methods 只读旗不发旗;无新 Skill 类、无权限平台、无 Schema 改动。

**接线两处(methods 手术)**:(1) `fast_agent.py` 新增 `_supplies_candidates` / `_supply_rung_candidates`,并把 propose 阶段单独包一层 try——**捕获集与外层完全相同**,仅在视野里确有供给卡时把失败降级为"agent 零提案"而非整轮失败(此前合并点在 propose 成功路径**下游**,agent 协议失败会连带毁掉卡的冻结程序;失败仍可见:propose 不进 `stages`)。(2) `online_loop.open_delayed` 给 `deployed_existing_skill` 路由补上 delayed 裁决出口——判据不新造,直接读 winner Episode 刚拿到的 `_update_delayed_status` 三档分级(LOCAL_ACTIVE 才批准)。**这是同权修复而非扩权**:此前供给候选比 agent 自提 winner 权利**更少**——PS-2 run9/run12 Support +0.6364/+0.6000 POSITIVE、delayed +0.30、Episode LOCAL_ACTIVE,却因 `approved_skill_id` 恒为 None 而被 ledger 规则拒绝,部署回落 identity。

**12 跑协议表**:A5-scoped **4/4 入池 / 4/4 获 Support 回执 / 4/4 材料正 / 4/4 delayed 批准 / 4/4 部署,held-out accuracy gain 每跑 +0.2127**(sealed oracle 上界 +0.184,实测略高于预注册"≈+0.18 级");A5-neutral 2/4 入池、**0/4 材料正、0/4 部署**(安慰剂链正确,`resample_uniform` 被测但读不出效果);A3 无卡全 0;**三臂 harm 全零**;探索槽 4/4 保留(每个入池轮都与 agent 自提候选共存,select 从未选中供给候选而 Support 预算仍给了它一个位)。工件 `artifacts/functional/e2/ps2p_production_validation.json/.md`。

**PS-2 读数订正(承重)**:PS-2 的 `_inject_funnel` 把 Support 记在"select 选中"上,但 harness 对池内每个成员都在轮预算内发 Support 位——run9/run12 实为完整 Support+delayed 走通却被记成 0/4。本书改为按注入自己探针写下的 Episode 归因(episode_id 带 Skill id,agent 同签名程序不会误计)。故 PS-2 的 `POOL_ENTRY_WITHOUT_CONVERSION` 中"零 Support"一句应读作计分口径产物;真正未兑现的是**部署**,由上述第 (2) 处修复解开。

**测试**:新增 `tests/functional/test_supply_rung_wiring.py`(11 项,0 LLM):(a) propose 抛错时供给候选仍入池且获测 + (e) 同形但 `supplies_candidates=false` 的卡不入池(读的是权限位不是形状);(b) agent 有提案时其首选先测、预算只够一个时探索槽不被挤占;(c) 池含供给候选时总数 ≤ maximum_candidates;(d) 无捷径四项(delayed 确认→批准;delayed 拒→`_incumbent_after_delayed` 不动 ledger;delayed 害→撤销且不批准;Support 非材料正→无 Draft、无 winner、Episode 仍如实写)。回归 95 passed(B 的 12 项 guard 测试 + T1 惰性 + `test_ttha_agent` + risk/source/ordering card 套)。**h0 锁机械重生成**(`ttha:fast_agent` 在 dependency_shas 内):`harness_content_sha` **不变** `53b1c803…0654f`,`runtime_bundle_sha` 4abf3bec…→0d66c4d3…。未跑全仓 pytest。

**书外发现**:(1) **残留耦合在 inspect 层**——ps2p_run2/run8 仍 `card_in_view_not_in_selectable_pool`,两轮均 llm=2、`chosen=""`、agent 零程序;同一张 neutral 卡在 run5/run11 正常入池,说明 `_supply_rung_candidates` 非空,故失败发生在**本书 try 之前的 inspect 阶段**(inspect + 一次重试 = 2 call)。同款一行处理应当可行:`inspected_regions=()` 时 `verify_candidate` 的 `outside` 恒 False,不需要任何"退化全窗口"决定;但需再跑一次 12 跑验证,本书预算不够,列为下一书具名项。**因此"注入去耦合"在本轮 live 未被直接观测**(每个入池轮都恰好有 agent 程序共存),只有单测证据。(2) select 层四跑全未选中供给候选,却四跑全部转化——**转化不经 select**,由探针序 + Support 预算完成;这既是好消息(机械通道不依赖 LLM 选择)也是提醒:PS-2"选择盲区"断点仍在,只是不再挡路。(3) A3 run7 亦出现 agent 零程序轮(llm=3),说明该早停与卡无关,是 GPOvY 上 proposer 自身的失败率。

**语义纪律**:本条只主张"经验以机械通道供给了一个待验证候选、Target 反馈裁决并批准了它",**不主张 agent 学会了提出 hampel**(自提族仍为 burst/outlier_threshold,与 PS-1 基线一致);GPOvY 与 source A 同属 GunPointFamily,系机制隔离而非跨族迁移主张;引导下的正例对 Source 跨域授权**计零**;pilot 级,不冻结生产设计。

**提交**:`methods/ttha/fast_agent.py`、`methods/ttha/online_loop.py`、`methods/ttha/harness/h0/snapshot.lock.json`(机械)、`evaluation/functional/run_e2_ps2_mechanical_supply.py`(生产模式 `--prod-run`)、`tests/functional/test_supply_rung_wiring.py`、`artifacts/functional/e2/ps2p_production_validation.json/.md`、本节。`AGENTS.md`/`README`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP` 未碰;`contracts/`/`runtime/`/`operators/` 零改;密钥零出现于任何提交物。

### B 收口:guard 管道三处修复落地,中档在分类线可达(2026-08-26 19:xx,执行方)

**判定:`GUARD_TIER_REACHABLE_AT_TWO_UNITS`**(INFRASTRUCTURE,非能力证据)。0 LLM / 0 fit / 0 下载。解释器 `D:\Anaconda_envs\envs\project\python.exe`。**阈值/语义零改动**:`RISK_MIN_DISTINCT_TASKS`、`MIN_DISTINCT_TASKS`、`MATERIAL=0.005`、LOO 语义、T1 谓词逻辑、TRY 授权规则、risk_skill 计数与 payload 全部逐字未动;无配置开关、无新平台、无 Schema 改动。

**三处修复**:(1) `methods/ttha/online_loop.py:_write_target_episode` 的 `context_summary` 写 `task_episode_id = domain`——该串就是本 Episode 的 `domain_namespace`,由调用方在开跑前从 cell 标识机械拼出(分类线 = `dataset/condition`),单元内跨探测跨 r1/r2 恒定、换单元必变,且不含 Outcome/delayed/held-out 标签(同一 Episode 早已带着它);语义与预测线 `e1._make_episode` 一致。(2) `agentic/source_skill.py` 新增 `risk_guard_rows`(纯投影,读审计既有 `pooled_negative` 去重计数,只保留"分裂族不发声"这半条既有 clause-kind 规则,`>=2` 的比较仍归审计与 retrieval)+ `build_skill_payload(risk_evidence=...)` 写 `risk_guards.evidence_distinct_task_count` 与结构化 `deprioritized_scoped_evidence`(算子名 + scope + 计数);不传参时字节与修前完全相同,body 一字未加。(3) `run_e2_t6_cls_op_shared_harness.py:_risk_lifecycle`(在 `_run_round` 尾调用)接通既有 `agentic/runner.run_risk_skill_lifecycle`——只建 `_ArmState`、调既有函数、把返回的 snapshot 写回 method 引用,无新生命周期。

**可达性差分(0 LLM,真实 Episode + 一个合成第二单元)**:证据 = `ECG200/fit_only_artifact` 的 `outlier_mad` NEGATIVE(−0.1429,取自 `t6_cls_conf_dev_ecg200.json`)+ 合成 `Wine/fit_only_artifact` 害证。六级链路修前全部不可达、修后 n=2 全部可达:`_task_of` `["",""]`→两个真串;census 去重 1→2;`risk_candidates` `[]`→`["outlier_mad"]`;分类线编译 无调用点→`target_risk_outlier_mad`;`resolve_harness_view(role="fast")` 空→含该 guard;形态 = `skill_kind=safety` / `allowed_tools=[]` / `_skill_frozen_candidates` 空 / `evidence_distinct_task_count=2` 的结构化 avoid。工件 `artifacts/functional/e2/b_guard_pipeline_reachability.json/.md`。

**结构发现(未修,超出本书)**:(a) 经验卡分支只走到一半——修 2 已把计数落到卡上,但 `SOURCE_APPLICABILITY` 仅 `task_kind==classification`,`retrieval._scopes_beyond_task_kind` 要求更细,故现役 source 卡仍判 inert;换成更细 Scope 后同一张卡即被送达(差分表两行对照)。(b) `CENSUS_CONDITION_KEY = support_reproduces_fit_signal` 不在 `contracts/observables.OBSERVABLE_FEATURES` 里,**根本无法作为 applicability leaf**——分类线要给中档一个真 Context Scope 需动 observable 契约,不是接线能修。(c) 分类 Episode 不写 `context_summary.task_signature`,故 minted guard 的 applicability 为 `{"const": True}`(既有 `applicability_from` 语义);要收窄须写 signature = Scope 改动。(d) `risk_skill_payload` 把 `negative_task_ids` 渲进 body,故 dataset/condition 串会作为**证据出处**出现在后续单元的 Fast prompt(适用性仍只由 applicability 决定);改 body 是语义改动。(e) 既存 `tests/functional/test_skill_revocation.py` 在 Python 3.10 下语法不可解析(第 166 行多行 f-string 为 3.12+ 语法),先于本书存在,未碰。

**测试**:新增 `tests/functional/test_guard_pipeline_reachability.py`(12 项,0 LLM;含三处各自的聚焦单测与差分断言,`__main__` 生成工件)。回归全绿:`test_ttha_agent` / `test_inert_experience_card_visibility` / `test_source_derived_skill` / `test_target_local_risk_skill` / `test_source_skill_default_bytes` 共 87 passed;`test_ordering_card` / `test_p2_online_route_abstain` / `test_n5_growth` / `test_n3` 15 passed。**h0 快照锁未重生成**:`snapshot.lock.json` 的 `dependency_shas` 不含 `online_loop`,`retrieval.py` 未改,`compile_snapshot(verify_lock=True)` 通过,`harness_content_sha` 仍为 `53b1c803…0654f`。未跑全仓 pytest。

**提交**:`methods/ttha/online_loop.py`、`evaluation/functional/task_episode_harness/agentic/source_skill.py`、`evaluation/functional/run_e2_t6_cls_op_shared_harness.py`、新测试文件、`artifacts/functional/e2/b_guard_pipeline_reachability.json/.md`、本节(含他书未提交台账条目一并入库,未删改既有正文)。`AGENTS.md`/`README`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP` 未碰未提交;`contracts/`/`runtime/`/`operators/`/`risk_skill.py` 零改。

### S1b-r3 微书收口:delayed 拒批的 winner 不再被部署(canon-vs-code bug 修复)(2026-08-26 22:0x,opus)

**主线裁定**:三级反馈教义——Support 只立 Draft,delayed 批准才 Active;部署一个 delayed 已拒的程序违背已批语义,系 canon-vs-code bug,S1c 前必修。0 LLM、0 下载、19 fit、33 s。

**最小修复**(共享 runner,新函数 `run_e2_t6_cls_op_shared_harness._incumbent_after_delayed`,轮体一行改调用):winner 存在且 `approved_skill_id` 非 None 才写入 ledger incumbent;被拒则**不采纳该程序**,回退到"上一轮真正过双门的 Workflow",没有则 identity(`FROZEN_LEDGER_NO_INCUMBENT_IDENTITY`)。**语义零改动**:Support/delayed 判定、`classify_relation`、`MATERIAL=0.005`、LOO、T1 谓词、risk_skill 全部逐字未动;无开关、无新 Schema。轮记录新增三个只读字段(`winner_delayed_approved` / `incumbent_before_round` / `incumbent_after_round`)供审计。S1b runner 的自有轮体改为调用同一函数,规则只有一处定义。

**受影响入口清单**(共享 runner `_run_round` 全部 5 个调用点 + S1b runner 1 个,修复一致生效):(1) `run()` Part B Source 轮(`allow_fast_skill=False`,**不部署**,行为不变仅 incumbent 字段变);(2) `run()` Part C 三臂 Target 赛(`--run`/`--smoke` 及经 `run()` 的 conf-dev / dev-wine / r2-prep 派生入口,**部署,行为改变**);(3) `conf_run()` A3 臂(**部署,行为改变**);(4) `micro()`(`allow_fast_skill=False`,只打印,**不部署**);(5) `r2_replay_a5()`(**部署,行为改变**);(6) `run_e2_s1_curriculum_four_arms._run_round`(**部署,行为改变**)。既有已提交工件未重跑、未覆写。

**聚焦单测** `tests/functional/test_delayed_rejected_winner_not_deployed.py`(8 项,0 LLM/0 fit):Support 正+delayed 拒 → incumbent 清除、`_frozen_recall` 部署 identity;Support 正+delayed 批 → 照常部署该程序(现行为不变);无 winner 轮不动 ledger;r1 批 hampel、r2 拒 winsorize 的双轮序列 → 仍部署 hampel(拒批只否决新候选,不丢弃已批 incumbent);另加一条防回潮断言(两处轮体必须走同一函数,旧的直接赋值写法不得复现)。回归:`test_guard_pipeline_reachability` 12 项 + 本文件 8 项 = 20 passed。

**smoke 复跑(r3,工件 `s1_smoke_cell1_r3.json/.md`,不覆写 r2)**:九门全绿(新增 `no_delayed_rejected_winner_deployed`)。三适应臂由 `FROZEN_LEDGER_INCUMBENT/winsorize/+0.1993/worst-class −0.0720/harm=True/regret −0.1890` 改为 `FROZEN_LEDGER_NO_INCUMBENT_IDENTITY/identity/+0.0000/worst-class 0.0000/harm=False/**regret +0.0103**`,与 Static 齐平;Static 完全不变。**harm 事件归零、负 regret 消失**——r2 那条"伤类换准确率把 regret 刷成负"的书外发现,根因正是本 bug:被 delayed 判定为 NEGATIVE 的程序本不该进部署账。每臂 fit 由 7 降至 6(部署 identity 命中基线缓存)。17 项状态隔离断言、域绑定合成探针、oracle 三读取面探针全部照跑全绿;反馈面仍为 `live`(三条 NEGATIVE Support 回执不变——修的是部署,不是判定)。

**提交**:共享 runner 修复、S1b runner(同规则 + `--smoke-suffix`)、新测试、`s1_smoke_cell1_r3.json/.md`、本节。**义务自报**:阈值与判定语义零改动;受影响入口清单见上;S1c 未跑;未跑全仓 pytest;`methods/`/`runtime/`/`contracts/`/`operators/` 零改动;密封 oracle 零改写;零下载。

### S1b-r2 收口:切片可读性地板重选课程,反馈面活证到手,仪器阻断解除(2026-08-26 21:4x,opus)

**主线裁定采纳**:r1 的"最小总点数"排序反向选择了反馈面,系主线规则错误;r2 改为可读性地板 + 可读性排序。执行:0 LLM、0 下载、22 fit、34 s。r1 课程完整保留于 `artifacts/functional/e2/s1_curriculum_frozen_r1.json/.md`,未丢弃。

**新冻结课程(r2,正序)**:`MiddlePhalanxOutlineCorrect__impulse_v2`(害证A,最小切片 45)→ `DistalPhalanxOutlineCorrect__burst_cls2`(可学A,44)→ `PowerCons__impulse_v2`(害证B,12)→ `FreezerRegularTrain__burst_cls2`(identityA,10)→ `GunPointOldVersusYoung__impulse_v2`(可学B,10)→ `ECG200__impulse_v2`(HELDOUT_ONLY,7)→ `Ham__impulse_v2`(identityB,8);反序严格逆转。**7 单元全部无空切片,最小切片全课程 ≥7 行**(r1 为 6/7 单元 ≤2 行、3 个 `r2_delayed` 为 0 行)。切片数直接读密封 oracle 的 `cell.slice_rows`,零新 fit。**降档轨迹**:害证/identity/HELDOUT_ONLY 三组均在地板 5 + 严格跨课程家族去重下满额,零降档;可学组在地板 5 严格家族下只能填 1(Phalanx 与 PowerCons 两族已被害证组占用,GunPoint 族只能出 1 个),按声明的松弛阶梯先走完 5→4→3 严格档仍不满,再回到地板 5 允许同族,取 `DistalPhalanxOutlineCorrect__burst_cls2` 为**具名同族重复**(PhalanxFamily,与害证A 同族但不同底物)。**家族**:6 独立家族 / 7 单元,重复 1(PhalanxFamily);7 个底物互不重复(r1 曾有 MoteStrain 底物重复,已消除)。

**规则解释一处,须主线确认**:必要条件 `|pooled held-in 读数| ≥ 1/最小切片行数` 只对害证组与可学组生效。identity 组按定义 oracle set = identity、HELDOUT_ONLY 组按定义 held-in = 0,literal 套用该式会让这两组在阶梯每一档都归零候选(工件 `literal_application_counterfactual` 逐组记数),与"2+2+2+1"结构自相矛盾;对这两组,"若真有材料级效应则本可看见"正是地板本身。已写入 `selection_rules.necessary_condition_scope`。

**smoke 读数(单元 1 = MiddlePhalanx,45 行切片,scripted backend)**:八门全绿,`S1B_SMOKE_WIRED`。**反馈面活证到手**:三个适应臂各拿到一条 **NEGATIVE** Support 回执(winsorize,slice support gain +0.1333、delayed −0.1111),`feedback_surface_evidence_mode = live`,不需退回算术旁证;旁证仍一并记账(outlier_iqr pooled −0.0944、outlier_mad +0.1389,均 ≥ 1/45 = 0.0222)。四臂表:Static identity/gain 0;A3/K0/A5 三臂同样冻结部署 winsorize、held-out gain **+0.1993**、worst-class **−0.0720**、harm_event True、regret **−0.1890**。状态隔离 17 项、域绑定合成探针、oracle 隔离(三读取面探针全 blocked、判分只读 1 键、臂阶段泄漏 0)全部沿用 r1 实现并全绿。

**书外发现三条**:(1) **regret 单指标可被"伤类换准确率"反向刷分**——本单元 menu-oracle 是受 class-harm 约束的 `repair_level_shift`(+0.0103),而 winsorize 以 worst-class −0.072 换来 +0.1993,于是 regret 为负而 harm 为真。预注册判读已把 regret 与 harm/worst-class 非劣并列,此例是该设计必要性的实证,S1c 报告不得单引 regret。(2) **delayed 否决的 winner 仍被冻结部署**——`handle_feedback_delayed` 正确拒批(approved_skill_id 为 none),但共享 runner 轮体里由 Support winner 写入的 `state['incumbent']` 未被清除,`_frozen_recall` 据此部署。属继承自共享 runner 的部署规则,本书只记账不修(改它是行为变更,需自己的切片)。工件 `deploy_rule_observation`。(3) **guard 通道可行性已可算**:`outlier_iqr` 在两个害证单元上均合法且可读地有害(−0.0944 / −0.1852,分别 ≥ 1/45 与 1/12),预期最早在正序第 3 单元后成型;但成型仍取决于提案阶段是否在两个单元上都采样到它,这是 Agent 行为、非算术保证。工件 `guard_channel_feasibility`。

**提交**:runner(选课函数 r2 重写 + 三个新读数)、`s1_curriculum_frozen.json/.md`(r2)、`s1_curriculum_frozen_r1.json/.md`(留档)、`s1_smoke_cell1.json/.md`、本节。未跑全程课程;未跑全仓 pytest;`methods/`/`runtime/`/`contracts/`/`operators/` 与共享 runner 零改动;密封 oracle 零改写。S1c 仍不在本书。

### S1b 收口:课程机械冻结 + 四臂 runner 就绪 + 单元 1 smoke `S1B_SMOKE_WIRED`;实测仪器阻断 S1c(2026-08-26 21:2x,opus)

**交付**:新独立 runner `evaluation/functional/run_e2_s1_curriculum_four_arms.py`(`--select-curriculum` / `--smoke`);`methods/`、`runtime/`、`contracts/`、`operators/` 与共享 runner 零改动,只 import 复用。0 真实 LLM(smoke 走 scripted sealed-probe backend,9 次为脚本探测调用非 LLM 支出)、13 fit、11.2 s。

**机械选课(7 单元,双序冻结)**:正序 `MoteStrain__impulse_v2`(害证A)→ `ECGFiveDays__impulse_v2`(可学A,+0.5714)→ `Coffee__impulse_v2`(害证B)→ `SonyAIBORobotSurface1__burst_cls2`(identityA)→ `GunPoint__impulse_v2`(可学B,+0.4667)→ `BeetleFly__burst_cls2`(HELDOUT_ONLY 诱惑场)→ `MoteStrain__burst_cls2`(identityB);反序严格逆转。害证判据 = mad/iqr 合法(cohort 改动 ≤0.10)∧ held-in headroom ≤ −0.005,取最小总点数的两个异族;四类无缺额(shortfalls 空)。**域命名空间取 unit_id(dataset__injection)而非 dataset**:全课程同一 condition,若按 dataset 则同底物两单元在 `risk_skill._task_of` 里塌成一个 Task,e64c684 修 1 打通的 guard 计数会重新失效。**书外后果**:identityB 与害证A 同底物 MoteStrain(声明规则只在组内要求异族),6 个独立家族 / 7 单元,已记入工件 `family_census`。K0 = h0 三张 bootstrap + 既有惰性 `source_investigation_cls_v1`(从 `t6_cls_op_r2_three_arms.json` 读出,0 LLM),已断言 `allowed_tools=[]`、body 无 frozen steps、**C40 Target-local hampel 未装入**。

**smoke 四臂读数(单元 1,各 1 轮)**:四臂全部 identity 冻结部署、held-out gain +0.0000、regret 均 +0.1142(menu-oracle = hampel_filter);Static 0 轮 1 fit,三适应臂各 4 fit / 2 probe / 1 浪费探针。状态隔离 17 项断言全绿:Static 零轮零 Episode 存储不变、A3 起点=h0 且 K0 卡从未进其 store、K0-fixed 单元边界重置回 K0 且零 Episode 携带、A5 与 K0-fixed 同起点且携带 2 条 Episode Memory;**K0 卡在 K0-fixed/A5 store 中但从未进入任何 Fast 视图**(T1 惰性闸口实测生效)。域绑定:钩子 1 无新卡可盖章(本单元无 POSITIVE Support),故另以合成条目走同一决策函数做**构造性探针**——异域 Target-local capability 被丢弃 / 同域被保留;Scope-v1 五轴对匹配 Scope 放行、空交集/错 Consumer/Pattern 失配一律拒绝(单元 1 与单元 2 唯一相异轴 = `period_change_score`)。

**oracle 隔离(实测,非声明)**:模块 import 时包裹 `builtins.open`/`io.open`/`os.open`/`Path.open`/`Path.read_text`/`Path.read_bytes`,臂阶段命中 `s1_oracle/` 即抛 `OracleIsolationBreach`;臂阶段主动探针三个读取面全部 blocked;判分阶段仅读 1 个键(`MoteStrain__impulse_v2.json`),臂阶段泄漏 0。**本 smoke 的自身发现**:首版只包 `builtins.open`/`io.open`,`Path.read_text` 绕过——臂本可静默读 oracle 且不留日志;已修并以三面探针固定。

**阻断性仪器发现(须主线在 S1c 前裁定)**:声明的"最小总点数"选课规则把课程选到了**读不出关系的材料**上。7 单元中 6 个的最小 held-in 切片 ≤2 行,其中 **3 个 `r2_delayed` 切片为 0 行**(MoteStrain 两单元、BeetleFly__burst_cls2)——冻结的两轮协议在这些单元上没有 delayed 面可开。单元 1 实测:两次探测 gain 均为 0.0,两条 Episode 全 NEUTRAL,**未产生任何害证,guard 无物可编译**。即 S1 重塑后的主考通道(≥2 独立害证 → Fast guard)在当前冻结课程上不可能被点亮。可选修法(均属协议改动,本书未动):给 `support_pool_rows` 设下限 / 害证与 identity 组改取最大总点数 / 四切片不再取自本已极小的 support 池的四等分。工件 `s1_smoke_cell1.json/.md` 的 `instrument_census` 已给出 7 单元逐格 slice 分布与最小可表达 gain。

**判词**:`S1B_SMOKE_WIRED`(七门全绿:四臂状态隔离 / 域绑定钩子 / 判分出 regret 表 / oracle 隔离 / 隔离墙自检 / 预算 / 部署纯净)。仅为接线证据,非能力证据,课程未跑。**S1c 不应在仪器裁定前发车**。

**提交**:新 runner、`artifacts/functional/e2/s1_curriculum_frozen.json/.md`、`artifacts/functional/e2/s1_smoke_cell1.json/.md`、本节(含他书未提交台账条目一并入库,未删改既有正文)。封存 oracle 件零改写;未跑全仓 pytest;零下载。

### S1-diag 收口(仲裁改版):行为漏斗 + 候选帽配对;判读 proposal_semantics_insufficient;无 diag-r2/r3(2026-08-27 00:3x)

**原书 Part B 全菜单排序未跑**(作废,本会话无探索性排名物)。**Part A(0 LLM)**:工件未存全部原始提案——最早可得层 = 编译+verifier 后的 `pool`(`proposal_count` = 非恒等池长)。S1c 单层:在当前课程/Prompt/候选预算/这一次运行中,15 个含菜单正解的臂-单元机会**冻结部署命中 1 次**(A3-PowerCons-hampel);同一 15 次里菜单正解被执行 4 次(MiddlePhalanx 三臂 `repair_level_shift` delayed 未过 + 上述 1 次部署)。断点 `not_proposed=11 / executed_not_passed=3 / deployed=1`。不得写成固有难度或稳定概率。历史 CLS-OP 三件(r2 three-arms 8 轮 / a5-replay 2 / ECG200 conf-dev 2)分层单列,未混算。A5 六次 Slow 各 1/6 LLM,空携带因无可编译证据(非预算饿死)。**Part B(LLM 12/12,0 fit)**:同 observation/instruction/菜单,只改 K=3 vs K=5,只提案不执行,oracle 事后判定。K=5 三单元(PowerCons/GPOVY/Distal)均未提出菜单正解,且候选数仍为 0–1(槽未用满);ECG200 未跑、MiddlePhalanx 帽满未提案。**冻结判读 = proposal_semantics_insufficient**(不得再拆 observation vs 策略偏置)。无截断解释。探针隔离于 `s1_cold_policy_map_probes/`,不入未来臂视野。分级 hypothesis 轨含义按合理假设待因果实验验证照写。本书封顶。

**提交**:`evaluation/functional/run_e2_s1_cold_policy_map.py`(新诊断 runner,既有 methods/runtime/contracts/operators/课程 runner 零改);`artifacts/functional/e2/s1_cold_policy_map.json/.md`;探针目录;本节。

### M-1 收口:MARGIN_GATING_CONFIRMED——GPMvF 对半协议供给候选双门转化 2/4,held-out +0.1867,harm 0(2026-08-27 16:3x)

**核心正效果移动:是。** 唯一变量(四分→角色拼接对半:Support n=21 / delayed n=19,单轮双门)下,A5-scoped **供给候选经双门转化 2/4**(m1_a5_1/2),部署 held-out **+0.1867**,harm 0。冻结判词 **`MARGIN_GATING_CONFIRMED`**:确认面余量门控成立,余量分层进 Gate 4。算术先行 0 fit:Support 4.00× / delayed 2.00×,均 ≥2×(G3 四分余量 1.35×,材料正 0/4 不重跑)。漏斗:卡 4/4 在视野;入池 2/4;材料正 2/4;供给双门 2/4。a5_3/4 卡在视野但 inject=False(同族 prepare/identity-only 漏注入),agent 自提 hampel 亦部署——**不计供给转化**。A3 冷提案 3/4 同增益部署(a3_4 identity):对半面本身可读,门控的是确认面余量而非只是供给通道。对半读数只作余量机制证据,**不得与四分基线作能力比较**。pilot;GunPointFamily 同族;引导正例计零。成本 LLM 29/100、fit 45/100、墙钟 1068.7s/7200s;returned_model=`gpt-5.6-sol`;下载 0;methods/contracts/runtime/operators 零改;密钥零出现。

**提交**:`evaluation/functional/run_e2_m1_margin_gate.py`;`artifacts/functional/e2/m1_margin_gate.json/.md`;本节(含他书未提交的 Gate 3 收口 / A′ 发车 / 提速四点,未删改既有正文)。不提交 checkpoint / `AGENTS.md` / `README` / `PROJECT_STATE*` / `SUCCESSOR_BRIEF*`。

### SA-0 Part A 收口(审计件已落,执行者配额死亡);阶梯 v2 裁定一处更正:撤权网当前未上膛;L1 中途读数入案(2026-08-28 03:2x,主线)

**SA-0 状态**:执行者写完 `artifacts/functional/e2/sa0_wiring_audit.json/.md`(Part A 四项审计,全部引行;0 LLM/0 fit/0 git/密封件零读)后,在起笔设计稿时死于模型配额(resource_exhausted)。Part B 设计稿按分层纪律回归主线自写(方法级设计本属主线,不再占执行者);Part C 反事实量化仍待 L1 落地。**记录缺陷入案**:该 JSON 的 obligations 块预列了两份未写出的文件(设计稿、counterfactual),以本条为准更正,不改执行者工件。

**四项审计判定(一行版,详见审计件)**:A-1 归因面**部分足够**——卡→Support 可机械归因(`cand_skill_<id>` 前缀),卡→delayed/部署只有签名级 join;缺 4 个纯增字段(`episodes[].source_skill_id`/版本戳/逐单元 `scope_match`/逐卡受引导标记)。A-2 修订面——**`revision` 是静态著作字段,全仓无自增**(contracts/harness.py:111);Scope 收窄的 PATCH 面已授权但**无写者**(harness_surfaces.json:65-77,AST 支持 not,三值逻辑缺叶弃权=宁少供);回滚只在快照粒度(store 内容寻址+血缘)。A-3 混合反馈现状——**产线可达分支内对卡一律零写回**:正向不累计(转化只新铸 Target-local 卡)、Support 拒不可见、delayed 拒不可达。A-4 撤权钝度——两条机制按构造最钝(全卡全域),但对供给卡**均未接通**。

**裁定更正(主线自纠,盖 02:3x 条对应句)**:02:3x 条"Target 反证即 `restricted_by_target_feedback` 收回检索(retrieval.py:143/269 现役)"**只对读端成立**;写端触发 `stage==existing_skill_revalidated` 被 `_is_local_skill_id` 门在本地卡复用路径(runner.py:947-958、e1.py:689-690/133),**对供给卡不可能触发**;替代写者 `revoke_deployed_skill`(online_loop.py:814-817)不被本 harness 调用。**更正后表述:单例供给卡的当前安全背书 = verifier + 当前 Target Support/delayed 双门 + harm 否决,无任何事后收回;SA-1 上膛前,阶梯 v2 的安全论证不得引用撤权网**。L1 判词逻辑不受影响(永不被撤 ≠ 被错撤),此判由审计件明文确认。

**L1 中途读数(经审计件转录,L1 在飞不干预)**:T1 已过半——档价常数已落工作树(source_skill.py:319 `SUPPLY_TIER_MIN_DISTINCT_TASKS = 1`,TRY 档 :250-258 未动,新聚焦测试 test_supply_tier_compiler.py);单例卡已编译;**Scope 匹配预检 1/5(仅 GPOvY),预注册"匹四不匹一"在此轴已证伪**——逐叶重算:GunPoint 与 PowerCons 各只差 `period_change_score` 一叶(偶然携带),Herring 差 2/17,BirdChicken 差 3/17;按书 ≥1 匹配续跑 T2。**三条跨项发现一并入案**:卡 Scope 带重复 task_kind 叶(编译器 :467-474,applicability 分数双计);四个 pattern 叶因 edit schema 无契约被静默丢弃(有效 Scope 比记录宽,排除规则同盲区);n=1 退化交集"窄到无可修订"直接约束 SA-1 课程设计。**12 条开放问题(Q1-Q12)呈 sol**,全文见审计件;设计稿 `docs/SA1_SKILL_ADAPTATION_DESIGN_2026-08-28.md`(主线自写)一并呈。

### 夜间自主推进令(用户就寝授权);SA-0(适应线设计+接线审计)并行发车;L1 在飞(2026-08-28 02:4x,主线)

**用户定调(原话要义)**:夜间产出须服务论题"Skill = 由经验初始化、再由后续 Gain/Harm 持续修订的可学习组件"(纯靠正向一次性生成完美 Skill 不现实;现有 Skill 本就适应于特定数据,应视为可持续学习/适应的部分);晨间要实质性正向进展。**主线映射**:该论题两半——前半"经验初始化+低权入场+当前 Target 检验"由 **L1 在飞实考**;后半"按反馈持续修订"机制未建,凌晨不动治理刀,但以 **SA-0** 推进到"白天可批、次日可建":设计稿 + 接线审计,机制建设仍待 sol 批(治理语义底线预置:**单调收窄可自主、任何扩权/升档按阶梯证据定价**,防夜改治理复发)。

**SA-0 发车(opus,0 LLM,只读代码 + 新建文件,零 git 操作)**:Part A 接线审计四项(全部引 file:line):① 归因面完整性(supplied 候选逐单元结果对卡的记账:source_skill_id/candidate_origin/双门结果,W-1/G-3 仪器是否足以支撑修订归因)② 修订面(SkillEntry.revision 语义;`restricted_by_target_feedback` 的 PATCH 由谁写/触发条件/粒度;observable_applicability 可否 PATCH 收窄;版本可回滚性)③ 混合反馈现状(卡在 X/Y 转化、Z 被拒时今日系统行为)④ 撤权钝度(一次被拒是否会废掉他处正向的卡)。Part B `docs/SA1_SKILL_ADAPTATION_DESIGN_2026-08-28.md`:Skill=可更新假设的字段定义;按反馈类型的修订规则(正向→证据累计;冲突→结构化 Scope 排除,自失败单元 frozen pattern view 机械编译,禁自由文本;负向/害→分域限制;verifier 拒→几何注记);治理(收窄自主/扩权定价/版本化/回滚);SA-1 实验设计(K0-fixed v0 冻结 vs A5-adaptive v0+修订,主读数=修订后单元的 probe 浪费/regret/harm 差)与可证伪预测。Part C(仅当 `l1_ladder_v2_replay_r1.json` 届时已落,不等待):实账反事实量化(单调收窄能省什么、钝撤权会亏什么)。**防撞分工**:L1 执行者独占 methods+runner+工件提交+STAGE_REPORT 执行方条目;SA-0 零 git、只写自有新文件、禁触 STAGE_REPORT/密封件。晨间主线统一裁定与汇总;capstone 开封与 SA-1 发车均待用户+sol。

### S2 forecast 资产盘点(执行方):七项只读判定已落册;agentic 路线对 forecast 为分类特化(2026-08-28 16:2x,执行方)

**首行读数:现役 forecast 仪器在、S1 四臂不能翻旗跑 forecast、现役 12+8/12+4 cell 过不了新 ≥20 行门、+31.7% 是 v1 冻结前 Guidance 卡年代。** 授权链 = 台账 16:0x 条 S2 设计前置(发 grok,0 LLM 只读盘点)。成本 **0 LLM / 0 fit / 下载 0**。工件 `artifacts/functional/e2/s2_forecast_asset_inventory.json/.md`。

**七项(各一句)**:① 现役键 `forecast|ridge|sMASE`(TEH 回落,`experience_memory.py`:77 / `t1.py`:88)与 `forecast|pooled_ridge_a1|sMASE`(T5,`run_e2_t5_lifecycle_dual_consumer.py`:262-267);S1 四臂 `TASK_KIND=classification`(:275-277)且 :1683 拒他 kind。② 注册表 1919 行;现役 electricity/T233 12+8、traffic 12+8、NOAA Frep 12+4 对「对半每面 ≥20 行」FAIL;仅重切 traffic_hourly 862 / electricity 370 / metr_la 207 / nn5 91 才过。③ +31.7% = Frep 回放 84 vs 123(`t6_45_frep_a5a3_replay.md`:95),`CHAIN_REPRODUCED`,机制 = Guidance 卡非阶梯/修订;v1 冻结下重挣须过新门+新宿主+只读 Skill 层。④ Frep 有 `fresh_batch_guidance_*` 与 `fast_winner_forecast_*`;仓内无已提交可检索 forecast Episode 库。⑤ forecast 注入 = impulsive_outlier/gap/T0 cycle/minipipe/benchmark;**无** impulse_v2。⑥ 生命周期 Consumer 只有 ridge×sMASE(pooled+per_channel);DLinear/kNN 探针未接 G1/S1/T5。⑦ `#31` shared 卡无 `task_kind` 轴(预测域编译);AD v3 已归档(`AGENTS.md`:211);分类卡惰性守卫缺 forecast 宿主。

**S2 成本承重**:现代课程(S1 四臂+修订+五轴)是分类特化,不是通用 `task_kind` 分发器;G1 agentic 是 forecast 写死(`runner.py`:351-372)但不承载该课程。S2 不能拨旗,须付适配器或新 runner。

**义务**:只读 + 两新工件 + 本条;methods/contracts/runtime/operators 与密封件(Epilepsy2/s1_oracle/D2)未碰;未跑全仓 pytest;零子代理。**first fault:无。**

### S2a G0 第二源清扫(执行方):electricity 一次切完;**S2_HOST_READY_FAIL_BOTH_SOURCES**(2026-08-28 17:0x,执行方)

**S2a 状态: FAIL_BOTH_SOURCES**。授权链 = 台账 17:0x 条「S2a-G0 首败裁定……第二源清扫续令」。成本 **0 LLM / 75 本扫 fit(累计 180/300) / 下载 0 / 24.9s**。未改模板/门/菜单,未按读数增删,未开 Part C。

**预声明切法**(池容量,非读数):现役 loader = `shared_tsq_datasets/electricity/electricity.csv`(TSL 321 通道,UCI 族;注册表 370×1024 不能承载同构 origin 1104/1800)。cell 宽 60 → 最多 5 个 impulse(300 列),余 21 未用;gap 沿用 `traffic_gap_00`(角色 = Scope 不匹配,其 LEARNABLE 不碍守卫)。注入现役 `impulsive_outlier` 原件。oracle 密封 `s2a_oracle/electricity_impulsive_outlier_0{0-4}.json`。

**electricity 资格**:5/5 LEARNABLE two_x,oracle 全 `winsorize`,余量 1.14–1.50(远超门 0.10);贴线 0;identity 0。**两源合并**:impulse 11/11 强 LEARNABLE,weak 0,identity 0,gap 1(沿用)。课形凑不齐。工件 `s2a_g0_electricity_sweep.json/.md`。**后备三选现交主线呈 sol**:(i) 扩 metr_la/nn5;(ii) 课形收缩撤贴线/identity;(iii) identity 改 clean 条件 cell。**first fault:S2_HOST_READY_FAIL_BOTH_SOURCES — no_near_line_weak_beneficiary; no_identity_field。**

### S2a Part B(执行方):考场重切+oracle 已跑;**S2_HOST_READY 不过,全书停**(2026-08-28 16:5x,执行方)

**首行读数:S2_HOST_READY 失败;first fault = 机械 6+1 cell 全强 LEARNABLE,缺贴线弱受益与 identity 场。** 授权链 = 台账 16:5x 条。成本 **0 LLM / 105 fit / 下载 0 / 78.8s**。未改数据、未改门、未开 Part C。

**重切**:`monash:traffic_hourly` 经 `shared_tsq_datasets/traffic/traffic.csv` 文件序,7 cell × (40 train / 20+20 对半 / 20 official held-out);材料线 `max(0.005,1/20)=0.05`,2×=0.10;注入现役 `impulsive_outlier`×6 + `gap`×1(`injection.py` 原件未改);Consumer = pooled ridge + sMASE,origin 1104/1800。oracle 密封于 `artifacts/functional/e2/s2a_oracle/`。

**资格**:6/6 impulse 均 LEARNABLE 且 headroom 0.88–1.27(远超 2×);0 贴线弱;0 identity;`gap_00` 亦 LEARNABLE(headroom 0.148)。缺课形所需「强+贴线弱+identity」。**禁改数据/门凑格,停呈。** 工件 `s2a_host_ready.json/.md`。**first fault:S2_HOST_READY — no_near_line_weak_beneficiary; no_identity_field。**

### S2a Part A(执行方):v1.1 三处分发落地;分类 146 原封全绿;冻结清单升 v1.1(2026-08-28 16:5x,执行方)

**首行读数:v1.1 适配过门;分类 146 绿;清单 sha `68ea0f07…`。** 授权链 = 台账 16:5x 条 sol 核准 v1.1 受限修订。成本 **0 LLM / 0 fit / 下载 0**。

**三处**:① `run_e2_s1_curriculum_four_arms.py` 三常数经 `bind_curriculum_identity` 参数化,缺省仍 classification;② `_scope_v1_admits` 放行 forecast 为合法轴值,仍按卡 `task_kind` 对绑定身份匹配;③ `skill_revision.contracted_axes` 按 `task_kind` 分发现役 `extract_public_features`(forecast 分支 `runtime/public_features.py:265-331`;分类默认 = 历史调用)。29 文件清单仅 `skill_revision.py` 字节变(`88bec25c…` → `f1c856e8…`)。

**测试**:3 新聚焦(`test_s2a_v11_adapter.py`) + 分类原 146 套件 = **149 passed**;未跑全仓 pytest。冻结清单 `skill_memory_v1_freeze.json/.md` 升 v1.1,`inventory_sha256 68ea0f07dc12e57b4c623d94979dfbff7b7a926fc6a9b1d05fccb7bc04f11494`。**first fault:无。** 续 Part B 考场重切。

### S2a-G0 判 `FAIL_BOTH_SOURCES` 收口(主线);**冲突场定义的移植修正案 (iv) 成型**:CONFLICT 语义系承重触发,"贴线余量"系分类底物偶然形态;四选呈 sol(2026-08-28 17:1x,主线)

**清扫核可**:electricity 一次切完 5/5 全强 LEARNABLE(oracle 均 `winsorize`,余量 1.14–1.50),两源合并 **11/11 impulse 强 LEARNABLE、贴线 0、identity 0**(提交 `951d85a`);执行者禁凑格纪律守住,未开 Part C+D。**书外正向注记**:electricity oracle 均 winsorize 而 traffic 侧非同一算子——两源程序几何有别,Scope 轴素材更富。

**主线复盘发现(承重,自认设计稿抄错)**:设计把冲突场写成"贴线弱受益",系照抄分类底物的**偶然形态**;分类线真正触发 R2 的是 `classify_relation` 的 **CONFLICT 判**(SA-1 r2 实录:PowerCons#1 与 Herring 均 CONFLICT——聚合正、逐序列害),非"低于材料线"。forecast 的**忠实移植** = "pooled 正 ∧ 逐序列害 ≥ 杠"的 cell 即冲突场;#31 谱系(5/5 聚合藏害、guard 全抓)早证该形态在预测域天然存在。11 cell 系预声明全报集合,按冻结 CONFLICT 语义指派角色**非按结果挑数**;但角色定义超出已批设计 → 须 sol 核,主线不自行执行。

**四选呈 sol(主线推荐序 iv > iii > i > ii)**:**(iv)** 对既有 11 cell 重算逐序列分解,CONFLICT 形态 cell 即冲突场(0 LLM,fit ≤60,机制忠实);**(iii)** identity 场以 clean 条件 cell 替代("不修没坏的"读数,协议变体);**(i)** 扩源 metr_la/nn5 仅当 (iv) 零命中;**(ii)** 课形收缩为末位(修订环标 forecast 未考会掏空 S2a 主张核心)。发车待 sol。

### S2a-G0 首败裁定(主线):v1.1 修订**已生效**(149 绿);资格门败于"注入-底物多样性失配";第二源清扫续令(设计内);后备三选预置(2026-08-28 17:0x,主线)

**Part A 核可(真实进展)**:v1.1 三处分发落地,3 聚焦 + 分类 146 项回归**原封全绿**(P1 立),冻结清单升 v1.1(提交 `04f5703`)——修订案生效,Skill/Memory 语义零改自证。**Part B 首败定性**:traffic_hourly 机械重切 6 cell 在现役 `impulsive_outlier` 模板下**全部强 LEARNABLE**(余量 0.88–1.27 ≫ 门 0.10),无贴线弱场、无 identity 场(提交 `4ec3aff`,执行者拒凑格正确)。**注记一条**:gap 场的守卫角色 = Scope 不匹配(卡不该匹),其 LEARNABLE(+0.148)不碍守卫资格,资格门败点仅在贴线/identity 缺失。**定性入册**:forecast ridge/sMASE 对 impulsive_outlier 一致高敏且菜单一致可修,与分类线"同族三命运"成对照——不同 Task×Consumer 对同族缺陷的敏感度结构不同,系条件化论题正向素材,非机制失败。

**续行令(设计内,不呈 sol)**:设计原文"traffic_hourly **或** UCI electricity",第二源未扫,清扫未穷尽——令执行者按同冻结规则一次性完成 electricity(370 序列)重切清扫,全读数报告,再判 `S2_HOST_READY`。**后备三选预置**(若第二源仍无贴线/identity,呈 sol):(i) 扩源至盘点内 metr_la/nn5(超设计名单);(ii) 课形收缩——撤贴线与 identity 要求,S2a 只考"阶梯+Scope+双门"可移植,修订环如实标 forecast 未考;(iii) identity 场以 clean 条件 cell 替代("不修没坏的"读数,协议变体)。禁:调模板/调门/按结果挑数。

### sol 核准 S2 设计并更名 **S2a**(机制可移植性+跨任务隔离);v1.1 修订案批;S2b/G3 条件排队;S2a 发车(2026-08-28 16:5x,主线)

**sol 裁定(全采)**:方向对,批 v1.1 受限适配与 S2-G0/G1/G2,但**证据等级定名 S2a**——预测 Skill 系预测任务内重学,证"同一机制服务不同任务";分类卡沉默证"任务边界有效";**未证"分类中学到的任务中立程序知识正向帮助预测"**,后者系缓议 G3(例:"先查反馈可读性、证据不足弃权"→减少无效 probe/降适应成本,而非把 hampel 动作迁过去),预测闭环跑通后另做小型 **S2b**;**S2a+S2b 合起来才算闭合 Stage 2 多任务承诺**。冷发现风险确认:两掷帽合理,仍空则**候选发现正式移交 Stage 3 决策策略进化**,不再改课。设计稿头部已加核准记录并更名。

**主线两点执行细化(入书)**:(a) K0-fixed 在 forecast 上 K0 为空,**预注册其读数 ≡ A3-reset(构造性等价,P6)**,充当免费采样方差复本——若两臂材料级分歧,即为方差尺度警报,不作能力读数;(b) 族外守卫场用 **gap 缺陷族 cell**(forecast 内的 BirdChicken 对应物)。**S2a-G0/G1/G2 一书发车(grok 续话,盘点同线)**:Part A v1.1 修订(三处分发,146 项分类回归原封全绿为生效条件,冻结清单升 v1.1)→ Part B 考场重切 + 双层 oracle 资格门(`S2_HOST_READY`,不过即停)→ Part C 四臂 live(两掷帽,ITT)→ Part D 分类卡三面全零守卫随跑(非零即机制级 first fault)。预算:A/B 0 LLM;C+D ≤120 LLM / ≤300 fit / ≤6h。

### S2 设计稿定稿呈 sol(主线):机制任务可移植性 + 跨任务条件化守卫;含唯一一处冻结修订案请求(2026-08-28 16:4x,主线)

**盘点收口**(执行方 `cc8e045`,七项全引证):S1 四臂 runner 系分类特化(三常数写死 + `_scope_v1_admits :1683` 拒他任务 + `skill_revision.py:78` 特征提取写死)——**S2 必须付适配成本**;现役 forecast cell(12+8/12+4)反馈容量全部不过新门,须从 monash traffic_hourly(862)/UCI electricity(370)重切;+31.7% 正账系 v1 前 Guidance 卡年代,只作历史参照不得与 S2 读数并写;无现成 forecast Episode 库(冷发现率未知 = 最大风险);注入用现役 `impulsive_outlier`;#31 AD 卡无 task_kind 轴不能作守卫,跨任务负控由分类 supply 卡独任。

**设计稿 `docs/S2_DESIGN_DRAFT_2026-08-28.md` 呈 sol**,要点:**G0** 宿主适配(语义零改仅分发)+ 考场重切(对半每面 ≥20)+ 双层 oracle 资格门(M-1 余量 ≥2× 产例判据继承),门 `S2_HOST_READY`;**G1** 机制重挣(课程形状照抄 SA-1 已证形状:产例→阶梯 v2 产卡→五轴转化→贴线冲突→R2 收窄→再遇位),判词 `S2_MECHANISM_PORTABLE / S2_PARTIAL / TREATMENT_EMPTY`,冷发现风险预置两掷硬帽;**G2** 跨任务守卫免费随跑(分类卡三面全零,非零即机制级 first fault);G3(程序性迁移)缓议另立。**冻结修订案请求(唯一)**:runner 三常数参数化 + task 轴放行 + 特征提取分发,Skill/Memory 语义零改,落地后清单升 v1.1、分类 146 项回归原封全绿为生效条件。预算:G0 0 LLM;G1+G2 ≤120 LLM。**待 sol 核准修订案与设计稿后发车(grok)。**

### Skill/Memory v1 冻结生效(主线):三债清、146 绿、29 文件清单入册;S2 设计进入资产盘点(2026-08-28 16:0x,主线)

**裁定**:执行方三笔修复核可——Q1 `SCOPE_OVERREACH` 因码语义正确(扩/缩方向分离,`RETRIEVAL_MISS` 原义不动,fault_routes.json:15 / router.py:22-25/71-77 / skill_revision.py:54);Q7 卡面 `scope_unreachable_axes` 纯增声明(source_skill.py:665);Q11 交集编译器 task_kind 去重仅及新卡(source_skill.py:570-571)。146 测试绿;0 LLM/0 fit;提交 `edefd15`/`8645739`/`0e5a2bf`;py3.12 测试债(他线文件)挂账注明。**冻结即时生效**:`skill_memory_v1_freeze.json/.md`,29 文件,总 sha `a5c98d40…`;此后 Skill/Memory 结构性改动须 sol 级修订案;**Stage 3 许可面 = instruction/决策策略层;Stage 2 只读 Skill 层**(照 15:5x 条边界)。

**S2 设计前置(发 grok,0 LLM 只读盘点)**:预测线资产清点——forecast 任务的 runner/harness 入口与 task_kind 支持现状(引行)、数据资产与反馈样本容量(对照"对半后每面 ≥20 行"新门)、A5>A3 +31.7% 正账的工件与机制年代、预测侧 Episode/store/卡现状、缺陷注入族实现、Consumer 实现。盘点回来主线出 S2 设计稿:主探 = **机制任务可移植性**(阶梯/承重轴/修订环在 forecast cell 上重挣,兼为主实验预测线密封考铺路)+ **跨任务惰性守卫**(分类卡在 forecast cell 零检索零供给,task_kind 轴负控);"算子中立程序性迁移"机制选项与双 Consumer 0-LLM oracle 扫描选项一并呈 sol。

### sol 裁 B 采纳:CAP-2 以 `POOL_EXHAUSTED` 终局收口,不放宽 TRAIN 门;措辞修正入典;分类开发线转入收口与 Skill/Memory v1 冻结(2026-08-28 15:5x,主线)

**裁定(sol,全采;主线自认 A 倾向失当)**:不放宽 TRAIN 门。理由系机制级而非程序级,**优于主线 15:2x 的 A 案辩护**——M-1 已证反馈可读性是转化前提;DodgerLoopGame 20 行 TRAIN 对半后 Support/delayed 各 ~10 行、材料线 0.10,系已实测"贴线不可读"区域(PowerCons 在 40 行面尚且贴线 0/2);主动选弱反馈靶大概率再得不可解读 neutral,偶然为正亦难逃"看池改门"质疑。主线低估自家 M-1 仪器质量先验,记档。**CAP-2 终判**:`POOL_EXHAUSTED / EVIDENCE_UNAVAILABLE_UNDER_RESOURCE_CONSTRAINT`。**措辞修正入典(论文口径)**:非"方法无法进行密封正迁移",而是"**当前公开 UCR、既定规模与新鲜度约束下,无可用 Scope 匹配密封靶,该证据当前 unavailable**"——记资源约束,不记方法负结果。

**路线图(sol,采)**:① 分类开发线正式收口(能力复证/修订复证/密封安全成立)→ ② 冻结 Skill/Memory v1,不再围绕单个数据集改结构 → ③ 补 Stage 2(跨任务程序知识迁移)与 Stage 3(Gain/Harm 驱动 Instruction/策略自修订)→ ④ 论文级主实验(足量反馈样本、多独立 family、新鲜池、独立开发/密封测试集)统一承担密封正迁移。**主线两点增益(入主实验设计输入)**:(a) **密封正迁移槽位不必由分类线补**——主实验多任务,预测线数据资源(Monash)充裕且已有 A5>A3 +31.7% 成本优势正账,能力级密封考应设计在反馈样本天然充足的任务线;分类线以四行终态定格贡献。(b) **v1 冻结令须同时划定 Stage 3 许可触碰面**(instruction/策略层),防 S3 开工即撞冻结墙。

**机制债核对(sol 清单 vs 实况)**:候选去重记录已落地(`b95a853`);余三笔规格清晰——Q1(SCOPE_OVERREACH 因码替换收窄授权令牌)、Q7(不可达轴卡面声明)、Q11(交集编译器重复 task_kind 叶去重)——按 15:0x 路由令发 **grok**(v1 冻结准备书:三修复+聚焦测试+冻结清单 sha 落册);py3.12 测试债系他线 untracked 文件,不碰,冻结文注明。**终态报告** `docs/CLS_LINE_FINAL_REPORT_2026-08-28.md` 主线自写入库。

### Skill/Memory v1 冻结准备(执行方):Q1/Q7/Q11 三笔机制债落地;冻结清单 29 文件;0 LLM(2026-08-28 15:5x,执行方)

**首行读数:三笔规格债已修、Skill/Memory v1 冻结清单已落册。** 授权链 = sol 裁 B + 台账 15:5x 条(显式解除此前"禁触 fault_routes/router"令)。成本 **0 LLM / 0 fit / 下载 0**。提交 `edefd15`(三修复+3 项聚焦测试)/ `8645739`(冻结清单)。

**Q1 授权令牌**:`fault_routes.json:15` 新增因码 `SCOPE_OVERREACH`(target_class 仅 `applicability`);`router.py:22-25`/`71-77` 映射收窄方向,非 applicability 即拒。SA-1 借用点 `skill_revision.py:54` `APPLICABILITY_CAUSE` 从 `RETRIEVAL_MISS` 换成它;R2/R3 收窄 PATCH 走新令牌。`RETRIEVAL_MISS` 路由字节未改,扩方向原义保留。

**Q7 不可达轴**:供给卡编译器 `source_skill.py:665` 把 edit schema 丢弃的 pattern 轴写入卡面新字段 `risk_guards.scope_unreachable_axes`(纯增,匹配仍只读机器 AST);n≥2 交集路径与 n=1 同字段。

**Q11 重复叶**:交集编译器 `source_skill.py:570-571` 对 `task_kind` 叶去重;只影响此后新编卡,历史卡零追改。聚焦测试断言新卡叶数 = 去重后值。

**测试**:3 项新聚焦 + SA-1 5 项聚焦 + 供给/检索既有 + 105 相关套件(含 first_fault 路由)合计 **146 passed**;未跑全仓 pytest。`tests/functional/test_skill_revocation.py` 他线 untracked、py3.12 f-string 收集债,未碰,冻结文挂账。

**冻结清单**:`artifacts/functional/e2/skill_memory_v1_freeze.json/.md`。29 个 Skill/Memory 机制面承载文件(种子 + import 图补全:retrieval/fast/online/experience/ordering/method/compiler/store/surfaces、contracts/harness+observables+canonical+三 schema、source_skill/risk_skill、三写回落点 skill_revision/runner/e1、fault_routes/router/edit_controller、public_features)。**inventory_sha256 `a5c98d40384b342d059d943296676b455fa155baff4b8db212c4b98fb4475fc7`**。冻结声明:"结构性改动须 sol 级修订案;Stage 3 许可触碰面 = instruction/决策策略层;Stage 2 只读 Skill 层"。

**义务**:阈值/门/菜单/prompt/模板零改;三写回语义本体(R1 追加 / R2 收窄 / R3 降权)零改;密封件与他线文件(`AGENTS.md`/`README`/`PROJECT_STATE*`/`SUCCESSOR_BRIEF*`/`ROADMAP`)未碰;零子代理;零下载;未跑全仓 pytest。**first fault:无。**

### CAP-2 收口裁定(主线):`CAP2_CANDIDATE_POOL_EMPTY` 核可;**公开档案该格资源耗尽入典为结构性事实**;放宽案与 r3a 先例正面冲突,呈 sol 裁(2026-08-28 15:2x,主线)

**裁定**:执行方停手正确、判名正确(既非"三件全不匹"亦非"结构全灭",另名呈裁合规);**反钓鱼自证核可**——`DodgerLoopGame` 只差 TRAIN 带一条(20 vs 下限 40)即可凑出候选,执行者点名而不动手,纪律满分;成本 0 下载 / 0 LLM / 0 fit / 12 min;提交 `c37948f`/`9fa0c79`。**first fault 归属主线**:CAP-2 §1 合取与"字典序取前 3"在当前档案上不相容,冻结件缺零候选分支——起草缺口,自认。

**结构性事实入典**(承接 2026-08-25 "本地分类数据预算已尽"):公开 UCR 单变量档案 190 行中,"二分类∧等长∧单变量∧全新可封"仅 5 名,过全部 §1 硬门者 **0**;反事实自证(撤 ROSTER 排除仍 0;撤双名单排除仅回流 11 个本地已持有名);metadata 现拉与 2026-08-25 归档副本逐字节同 sha(`1e336629…`)。**该格资源已被本线用尽,此非本跑失败,系档案边界。**

**两案呈 sol(主线倾向 A,但因先例冲突不自行执行)**:**(A) CAP-2b 单候选例外**——TRAIN 带仅限本批下调至 [20,400],唯一新增合格者 = DodgerLoopGame(20/138/288,总 45,504 点,相容门与点数门均过);代价如实预declare:对半双门分辨率减半(Support/delayed 各 n=10,材料线 0.10),弱门高杠;仍反钓鱼(一候选一考即停,考完无论判词收线)。**与 r3a 先例"行数门放宽以纳入 Coffee = 看池改规,拒绝"正面冲突,须 sol 明示区分或推翻**;可辩点:r3a 时存在替代路径(下载批),今替代路径已被机器证明穷尽;放宽发生在任何值读取之前,结构盲;修改常数的后果(门分辨率)透明入册。**(B) 承认耗尽,收口成文**——分类线终态四行:dev 能力复证 / dev 修订复证 / sealed 条件化+安全已证 / **sealed 族内正迁移"公开资源冻结协议下不可考"**如实入册,留作方法边界而非未完成项。B 在 A 失败后仍可退,A 的边际成本 = 一次下载 + ≤20 LLM。若 sol 批 A,执行按 15:0x 路由令发 **grok**(机械执行冻结协议)。

### 常备纪律澄清:模型分层优先于续话惯性(用户提醒,主线自认偏离,2026-08-28 15:0x,主线)

用户重申 2026-08-25 17:41 分层令。主线自认:今日全部委派走同一 opus 续话线,其中 SA-1 r2(纯采样重复)与 CAP-2 选靶/下载/顺序开封段按纪律属 grok 任务,用贵了;根因 = 把"子代理复用(同模型 resume)"排在了"难度分层"之前。**裁定优先级:难度分层 > 续话惯性——线内任务难度降级时新开 grok,不续 opus;例外仅限一次性密封操作或紧接手术的验证需护栏连续性**。opus 另有配额可用性风险(SA-0 前科)。在飞 CAP-2 不中途换手(协议中段换手风险大于算力成本),此后机械类书一律 grok。

### sol 裁 ① 采纳(密封条件考批次);CAP-2 协议前置冻结(含反钓鱼硬条款);发车(2026-08-28 15:0x,主线)

**sol 判读与裁定(全采)**:capstone NEUTRAL 语义确认 = "安全侧成功,能力侧未被考到,不是正迁移失败也不是成功";证据版图四行确认(卡收益 dev 已复证 / 修订 dev 已复证但收益主为避免重复错误 / 密封条件化+安全已证 / **密封 Scope 匹配场正迁移仍缺**);**选 ①:预冻结三候选密封批**,一次写死候选/顺序/停止规则,所有开封结果全报,首个 Scope 匹配靶进行能力考,三件全不匹即停止并承认 Scope 覆盖有限;CatsDogs 不用(计算规模不适配,维持原令)。

**CAP-2 冻结**(`artifacts/functional/e2/cap2_sealed_batch_protocol_freeze.md`,先于任何档案元数据浏览):结构盲选(二分类/等长/单变量/TRAIN∈[40,400]/**总点数≤10万**/长度过模板相容门[机械导出引行]/名称排除本地 40+ROSTER 全名单),字典序前 3 一次写死,过滤轨迹先于下载入工件;每 zip 落库记 sha256(新基线纪律);顺序考:结构失败=槽死亡不递补;不匹=记逐叶差异停用顺延;首匹=能力考(CAP-1 骨架,TEST 子集 min(官方,500) 行种子 20260828,材料线随 n 换算,三臂与卡照旧);**反钓鱼硬条款(主线加于 sol 四条之上):首匹靶考完即停批,判词不满意亦不得续开下一候选**;三件全不匹 → `SCOPE_COVERAGE_LIMITED` 书面承认,不追加第 4 件。预算:下载 ≤3 / LLM ≤60 / fit ≤80 / 墙钟 ≤3h。授权链 = sol 裁 ① + 用户转达 + 本条。发车(opus 续话)。

### capstone 收口裁定(主线):`CAPSTONE_NEUTRAL` 核可;**密封域条件化+安全面全过,能力侧未被考(靶在族外)**;sha 缺基线裁定;三主张证据状态定格;下一步菜单呈用户+sol(2026-08-28 14:5x,主线)

**裁定**:判词 `CAPSTONE_NEUTRAL`(A5−A3 = +0.000000,harm 0,三臂同 accuracy 0.5336)按 CAP-1 §6 核可;开封记录清洁(单次,授权链全,首读范围最小);成本 9 LLM / 9 fit / 5.6 min(帽 45/75/90);预测对表 P1/P2 立、**P3 破如实**(卡 Scope 差且仅差一叶:`estimated_level_offset` 卡要 low、靶读 zero);提交 `b95a853`/`8568ca8`。

**承重解读(NEUTRAL 在本案的语义)**:(1) **条件化主张在密封处女域按设计工作**——族外卡零广播零供给零转化,C40 病在 sealed 级未复发;去重记号如实报"Scope 未匹"未误记;**修订环零反馈零幻影修订**(修订确定性的对偶读数)。(2) **安全链在从未见过的域上端到端成立**——双适应臂各自独立提出的有害动作(repair_level_shift −0.2750 / outlier_mad −0.0500)全被 Support 拦下,双臂正确弃权,harm 0;Epilepsy2 系无 headroom 靶,identity 即正解,三臂同分即满分安全读数。(3) **能力侧未被考到**:A5−A3 ≡ 0 系构造使然(A5 唯一独有知识出了 Scope),不是对卡内容的读数;且未匹之叶系 **S1a 先冻家族轴成员**——家族轴按规格工作,是靶在族外,非 L1 偶然叶复发。附带:卡的 hampel 本轮无人自提,故"卡拒绝出场"未获"拒绝即最优"的对照证明,如实记。

**sha 缺基线裁定(执行方呈报,主线核可)**:CAP-0 之封 = 其实际记录的 11 项不变量(ROSTER 字节/成员名/尺寸/换行/记录数),全过即 `SEAL_INTACT` 成立;**缺 sha 基线 ≠ 不符**,继续开封的判断正确;sha256 首算入册为后续基线;本书"按 CAP-0 校验 zip sha"措辞系主线起草过度指定,记档自纠。

**三主张证据状态定格(全日汇总)**:① 带 Scope 经验卡端到端收益——**development 级已证**(+0.6860,GunPoint 族内,两跑逐字复现),sealed 级未证(靶族外未考);② 反馈驱动修订——**development 级复合已证**(两跑同向、修订确定性、再遇位行为差),sealed 级正确空转(治理读数);③ 条件化+安全——development 级(G-3 谱系)+ **sealed 级本日新证**(族外零供给、无 headroom 弃权、harm 0)。

**下一步菜单(呈用户+sol,主线推荐 ①)**:**① 密封条件考批次**——授权下载 K=3 个按冻结规则机械选出的候选(二分类、TRAIN∈[40,400]、总点数≤10 万、长度相容 impulse-v2 模板、不在本地 40 件与已用之列),顺序开封:族外开封 = 又一条 sealed 条件化读数并自动放行下一个;**首个族内靶跑能力考**(族内与否只能开封后判定,系密封的固有代价,规则全部先冻);每考 ≤10 LLM,总成本一次下载授权。**② 接受现状收口成文**——三主张按上表定级,NEUTRAL 如实写,族内密封正迁移留开放问题。**③ D2/CatsDogs 不推荐**——仍密封,但 ~1.48 万点/行,任何行数下总点数远超冻结计算门(BinaryHeartbeat COMPUTE_BUDGET_EXCEEDED 前科),开封只会重复仪器-规模失配;维持 sol 2026-08-26"不得因已下载强用"原令。

### SA-1 r2 收口裁定(主线):出口 A 核可;**复合措辞解锁**;"修订确定性"入典;两处更正记档;capstone 按出口 A 发车(2026-08-28 13:5x,主线)

**裁定**:执行方判定 CAP-1b **出口 A**(G0/G1/G2 全立,拒绝事件 exercised)核可;提交 `0f10ec4`,代码零改动自证(`git diff cf2eb12 HEAD` 实验面全空)。**复合措辞就此解锁(两跑同向)**:"反馈驱动的 Skill 修订在固定协议下可复现"——r2 且比 r1 更干净(可归因 probe 省 0→1;Herring 真实被拒触发第二次收窄,P4 在 r2 立,七预测全立)。**新性质入典,定名"修订确定性"**:两跑卡版本链前四版**内容 sha 逐字节相同且同序**——修订体是触发读数的确定性函数,系统随机性只在 LLM 提案层,治理层可审计可复现。regret 不变量逐字节复现(卡对无卡 +0.6860;臂间 +0.0000)。

**两处更正记档**:(1) **CAP-1b 第 13 行前提更正(执行方对主线,成立)**:"现 HEAD 代码面未变"字面不成立——`5ff76b5` 对 SA-1 runner 跑后判分/渲染区有 +182/−8(含 `_verdict` 一处**收紧**:机制差须可归因于收窄);实验面未动,r1 工件系同一 HEAD 重渲,两跑同字节判分,可比性成立;冻结件本体不追改,以本条为准。(2) r2 输出路径 workaround(runner 硬编码 r1 路径,产物事后搬移、r1 件恢复并逐字节校验)记仪器债,capstone 走独立 runner(PREP-1)不受累。**诚实边界三条带入终考**:修订环两跑只买到成本未买到质量(臂间 regret 恰 0);R3 仅离线背书;v4 无过度排除课程内未受检。

**capstone 发车(出口 A,开封 Epilepsy2)**:授权链 = CAP-1b + 本条 + sol 令("无论结果停止重复,进终考")。**沿 CAP-1 冻结骨架逐字执行**(TEST 476 行种子 20260827/sha `7e1c4088…`、TRAIN mod-4 对半双门、材料线 0.025/0.025、consumer ridge、菜单 sha `48e09ec4…`、每臂 LLM ≤15 / fit ≤25 / 合计 ≤90 min、判词 `CAPSTONE_POSITIVE/NEGATIVE/NEUTRAL` 阈值照冻结),**仅按 CAP-1b 替换两处**:解锁条件(§7 旧条款作废)与 A5 池(原"S1-v2 正序终态池"随线退役,改装 **SA-1 同源 scope-v2 单例供给卡 + R1-R3 修订环开启**);三臂骨架保留(Static / A3 冷 h0 / A5-adaptive)。**去重记号仪器**按 CAP-1b 预声明先行落地(评估层从既有字段推导:scope 匹配 ∧ 池无 `cand_skill_` ∧ 自提同程序 → `dedup_swallowed=true`,禁改 methods)。**预注册预测**:密封校验过(TEST 清单 sha 对上);A5 harm=0、零越权;卡 Scope 家族轴按注入族判定匹配(可证伪);头条 A5−A3 = 考题本身,不预测。开封动作与密封复核一并入执行方条目。

### sol 裁定采纳(r2 一次→无论结果停→进终考);主线程序性自纠(CAP-1 冻结件为准);CAP-1b 前置冻结(三出口);r2 发车(2026-08-28 12:5x,主线)

**sol 裁定(全采)**:SA-1 定性 = "生成→使用→反馈→修订→再使用闭环首次真实跑通,'持续修订带来稳定提升'未充分证明";**只重复一次 r2,协议/Scope/课程/阈值零改,无论结果停止重复,立即进 capstone,无 r3**;r2 通过门放宽到机制级(反馈驱动更新发生 + 行为按预期改变 + 零害零越权),**不要求数字复现**;r2 未复现则记"适应机制可运行但稳定性不足",capstone 改考"带 Scope 经验卡端到端收益"。**重要纠正(sol 对主线,成立)**:CAP-1 冻结件要求"S1-v2 正序两次信号+反序确认",单次 SA-1 未字面满足,不得立即开 Epilepsy2。

**主线程序性自纠(记档)**:12:4x 条"开封条件字面已达"以主线 11:0x 自立条目为参照系——主线条目不能悄悄顶替冻结工件,属越权表述,**收回**;且 CAP-1 原条件引用已退役的 S1-v2 线,字面不可满足,唯一合法路径 = 另立修订件。

**主线一处收紧(在 sol 方案上)**:新解锁规则**前置冻结**于 r2 结果可见之前(sol 原步骤 2 在 r2 后冻结,残留"按结果写门"风险),并补第三出口——**`unexercised`**(全课程无拒绝事件,PowerCons#1 采样转化即触发):既非复现亦非反证,机制证据维持 n=1,走 B 形态但判词不得写 refuted。P4 已示范"没写下来的第三条路"必然发生,故三出口全部预声明。**CAP-1b 冻结件已落**:`artifacts/functional/e2/cap1b_capstone_unlock_amendment.md`——r2 机制门 G0(安全,必须)/G1(≥1 写回且 sha 变化,必须)/G2(条件式:拒绝发生→R2 必触发+再遇位行为差);出口 A = capstone 考 A3 vs A5-adaptive(完整主张)/ B = 拒绝发生但 G2 破 → A3 vs K0-fixed(仅卡主张)/ C = unexercised → 同 B 判词有别;capstone 靶/骨架/预算/判分沿 CAP-1,仅解锁条件替换;去重记号仪器(P4)预声明为 r2 后 capstone 前落地;Epilepsy2 开封以 CAP-1b + r2 收口裁定为记录授权。**r2 发车(opus 续话,纯采样重复,LLM ≤120)。**

### SA-1 收口裁定(主线):判词维持;**归因三分账**(轴规则 +0.6860 / 修订环 1 次避拒 / 安全零害);P4 第三路径入案;capstone 开封案呈用户+sol(2026-08-28 12:4x,主线)

**裁定**:执行方判词 `SA1_DEVELOPMENT_SIGNAL`(单跑措辞)核可;四段全过、止损未触发;成本 78/150 LLM、69/300 fit、2227 s;提交 `cf2eb12`(代码)/`5ff76b5`(工件+台账)。回归 105 绿;`test_skill_revocation.py` py3.12 f-string 收集失败**先于本书**,挂账不修。

**归因三分账(本条承重,三笔分立入典、禁互相挪用)**:**(a) 承重轴效应(sol 裁 ③)= +0.6860**——两带卡臂 distinct 五单元累计 regret **+0.0850** vs 无卡臂 **+0.7710**,与 v4 尾段同底可比:叶 Scope 卡(L1)回收 +0.2127,承重五轴卡回收 +0.6860,**多回收 +0.4733**,与 SA-0b 反事实预估同向且更优(GunPoint 实转 regret −0.0667,略优 oracle);development 级、GunPoint 族内、双门零害。**(b) 修订环效应(sol 裁 ②)= 恰 1 次避免挨拒**,位于预注册再遇位 PowerCons#2(A5-adaptive 零供给零挨拒,K0-fixed 再供给再挨拒;P5 干净成立);卡版本链 v0 `00503481`→v1(R1@GunPoint)→v2(R1@GPOvY)→v3 `89728a4a`(R2@PowerCons#1,排除 `period_change_score==very_low`),全走冻结 EditController、sha 版本化;probe 层归因为零(agent 自提回填空槽),raw"省 2 探 2 拒"中仅此 1 拒可归因,如实。**(c) 安全面**:harm 全零、regret 非劣(两带卡臂逐字同分)、无升档无扩 Scope(受引导计零红线未破)。**用户论题两半就此各有实证**:前半(经验初始化低权入场)= L1+轴规则;后半(按反馈持续修订)= 本跑版本链与再遇位读数。

**书外三项入案**:(1) **P4 第三路径**——Herring 匹配在视野,agent 自提同一冻结程序,**候选去重吞掉机械供给**(池内无 `cand_skill_` 条目),非拒非排除;行为正确(防重复探测)但遮蔽供给归因,记仪器注意项:自提与供给同程序时应补写去重记号(小修,留后续,不阻塞)。(2) R3 离线只落降权未落排除(ECG200 害证与证据在 12 契约轴上不可区分,规则拒绝发明),live 零 harm 未现场考——R3 现场覆盖留待自然发生,不造 harm。(3) Q1 残留照令未动(收窄 PATCH 以 `RETRIEVAL_MISS` 作授权令牌,语义错位),留 sol 队列。

**capstone 开封案(呈用户+sol)**:11:0x 冻结的开封条件——"SA-1 预注册信号成立(材料级改善 ∧ regret 非劣 ∧ harm 0)"——**字面已达**(P5 成立、非劣、零害),但修订环自身效应量 = 1 次避拒(计数型读数,无冻结材料线可援),且单跑措辞纪律在案。**主线建议:先跑 SA-1 采样 r2(~80 LLM / ~40 min,同课同协议冻结),两跑同向再开 Epilepsy2**——一次性密封靶配复合级证据;若用户+sol 认单跑已足,条件字面满足亦可直开。开封后终考形态建议:CAP-1 冻结件 + scope-v2 单例卡 + 修订环开启(A5-adaptive 形态)对照 A3。

### sol 三裁全采(capstone 缓开 / SA-1 最小版批 / Q9 承重轴规则);主线三条操作化;SA-1 发车(2026-08-28 11:0x,主线)

**sol 裁定(全采)**:**①** capstone 暂不解锁——L1 +0.2127 认可为机制正结果,但系边界重放,未满足 CAP-1 原定完整演化条件;Epilepsy2 唯一,留待 Skill 修订产生正信号后开封(主线撤回昨日"capstone 先跑"推荐,资源论+门完整性论均成立)。**②** SA-1 批最小版:先补四归因字段,再只接三种写回(**正向→追加证据 / 冲突→收窄 Scope / 负向→降权-排除**);不同时处理 12 问,不建新平台;离线 replay 过后直接跑一次短 A3 / K0-fixed / A5-adaptive。**③** Q9 采**预冻结承重轴规则**:供给卡初始 Scope = Task × Consumer × Metric × **Pattern family** × Program geometry,不塞偶然 observation 叶;后续冲突/负反馈由 Slow 加排除条件;单例卡仍仅候选建议权,不能执行。

**主线操作化(三条,防日后重议)**:(1) **capstone 开封条件即时冻结**:SA-1 预注册信号成立(A5-adaptive 对 K0-fixed 在 probe 浪费/重复挨拒上材料级改善 ∧ regret 非劣 ∧ harm 0)→ 呈用户+sol 终批;单跑只记 `SA1_DEVELOPMENT_SIGNAL`,复合措辞仍须采样重复(既有纪律)。(2) **Scope 规则 v2(供给档承重五轴)**:Pattern family 轴必须引用**先于 L1 冻结**的既有定义(S1a 簇资格判定的 Pattern 交集/缺陷族分类),执行者引 file:line 注明来源;无机械定义即停呈,禁按 L1 结果挑叶、禁现场发明。Q11 重复 task_kind 叶随新编译器自然消失;`period_change_score` 类偶然叶不再入初始 Scope。(3) **收窄 PATCH 走既有冻结 EditController 通道**(L1 装卡同路),不触 minipipe fault 路由(Q1 无需新因码,零新平台);**R3 负向分支以历史负例离线 replay 验证**(v3 PowerCons CONFLICT、ECG200 outlier_mad 害证),live 短课预期零 harm、不承担 R3 测试。

**SA-1 发车(opus 续话,四段)**:Part 0 四归因字段(`episodes[].source_skill_id` / `source_skill_revision`=卡内容 sha / `round.scope_match_by_skill_id` / `guidance_conditioned_by_skill_id`)+聚焦测试 → Part 0.5 Scope v2 编译器,离线重编单例卡并出新匹配表(**门:尾段 impulse 四单元全匹、BirdChicken-burst 不匹**)→ Part 1 三写回接通(证据追加走 `risk_guards` PATCH append-only;收窄走 `observable_applicability` PATCH,排除叶限特征契约保证轴、只许由实拒单元编译;负向降权/排除同面)+离线 replay 门(含 R3 历史负例)→ Part 2 live 短课 r1。**课程冻结(六位,全为已审计单元,零新数据)**:GunPoint → GPOvY → PowerCons(#1) → Herring → **PowerCons(#2 再遇,机制探针)** → BirdChicken(族外对照)。三臂 = A3-reset(无卡)/ K0-fixed(scope-v2 单例卡,冻结不修订)/ A5-adaptive(同卡 + R1-R3);卡种子 = v4 GPMvF 记录 Episode(单例、供给权限、ITT 主判);**再遇位读数单列为机制读数,不计首行 regret(首行只算 distinct 单元)**。**预注册预测(可证伪)**:v2 卡匹 GunPoint/GPOvY/PowerCons/Herring、不匹 BirdChicken;GunPoint+GPOvY 供给转化;PowerCons#1 被拒 → **恰一次收窄 PATCH(卡 v1,内容 sha 版本化)**;Herring 被拒或被 v1 排除(均合法,如实报);**PowerCons#2:A5-adaptive 零供给零挨拒,K0-fixed 再供给再挨拒(核心机制差)**;A5 对 K0 probe ≥1 省、regret 非劣、harm 全零。**预算**:LLM ≤150 / fit ≤300 / 墙钟 ≤5h / 下载 0。**止损**:离线门任一不过即停呈 first fault;live 中 PowerCons#1 意外转化 → 机制读数改由 Herring 拒绝承载,如实报;拒绝发生而 R2 未触发 → 单一尸检,禁改课禁调门禁 r2。

### CAP-2 Stage 1 停手呈报(执行方):**候选池为空**(`CAP2_CANDIDATE_POOL_EMPTY`),零下载、零开封;§1 合取在官方档案上无一名通过;**冻结件无"零候选"分支,需主线裁定**(2026-08-28 15:4x,执行方)

**首行读数:CAP-2 结局 = 三候选名单无法组成——`CAP2_CANDIDATE_POOL_EMPTY`(既非"三件全不匹",也非"结构全灭")。** 工件 `artifacts/functional/e2/cap2_selection.json/.md`(全轨迹 190 行逐名放行/排除)+ `cap2_sequential_exam.json/.md`(状态 `NOT_ENTERED`)。成本 **下载 0 zip、LLM 0、fit 0、墙钟约 12 min**;ROSTER.md 未追加(无落库对象);`data/ucr_conf_downloaded/` 无新目录。

**元数据可复现性(先说这条,因为它决定结论强度)**:本次现拉 `https://timeseriesclassification.com/aeon-toolkit/metadata.csv` 与 2026-08-25 CLS-CONF-dl 归档副本**逐字节相同**(同 7253 B、同 sha256 `1e336629…`,190 行),故该普查稳定、选靶可由同一公开表复现。

**§1 合取逐项(全部公开元数据机械判定,零数值零标签)**:二分类 ∧ 等长(`Length != 0`)∧ 单变量(`Channels == 1`)∧ TRAIN∈[40,400] ∧ 总点数 `(TrainSize+TestSize)×Length ≤ 100,000` ∧ 注入模板相容门 ∧ 名称 ∉ 本地 40 ∪ ROSTER 全名单。**相容门系调用现役实现导出而非现场复述**(引行:`run_e2_t6_cls_op_shared_harness.py:430`(`_v2_segment_length` = `round(length/150)`)、`:455`(segment≤0 即 raise)、`:464`(段尾溢出即 raise)、`run_e2_task_context_label_evidence_witness.py:88`(`_bound_positions` 四点互异且距两端 ≥3)、`:37`(`SPIKE_FRACTIONS` 四值)、`run_e2_t6_cls_op_shared_harness.py:301`(`maximum_modified_fraction` = 0.10))。**注记:0.10 修改帽这一支从不单独决定任何候选**——伪影足迹恒为 `4×round(L/150)/L ≈ 0.027`,远在帽内;真正承重的是段长(L 需 ≥76)与端距几何。

**结果:池 190 → 二分类 58 → 二分类∧等长∧单变量 48 → 合格 0。** 其中 39 名"已在本地 40 件",4 名"已在 ROSTER"(BinaryHeartbeat / CatsDogs / Epilepsy2 / ItalyPowerDemand),其余均折于 TRAIN 带或点数或相容门。

**"全新可封"名单只有 5 个,逐一如实报(全在工件)**:`Chinatown`(TRAIN 20 < 40;且 L=24 段长 0、端距被拒)、**`DodgerLoopGame`(TRAIN 20 < 40 —— 唯一失败项,相容门与点数均过)**、`ElectricDeviceDetection`(TRAIN 623 > 400;1,123,840 点)、`RightWhaleCalls`(TRAIN 10934;51.6M 点)、`SharePriceIncrease`(TRAIN 965;115,800 点;L=60 段长 0)。

**空池是真的,不是解析口径造成的(反事实三行已入工件)**:按冻结口径合格 0;**完全撤掉 ROSTER 排除仍为 0**;同时撤掉两处名称排除也只重新放行 11 个**本线早已本地持有**的名字(ECG200/GunPoint 四件/Ham/Herring/Lightning2/PowerCons/ToeSegmentation1/Wine)。即:公开档案里"二分类∧等长∧单变量∧小规模"这一格已被本线基本用尽,这是结构性事实而非过滤实现缺陷。

**为何不套用 `SCOPE_COVERAGE_LIMITED`(重要,防日后误引)**:§3.6 的该判词是**关于卡 Scope 的**结论,须在三候选各自开封并算出 pattern view 之后才成立。本次零开封、零 Scope 判定,套用它会主张本跑并不具备的证据。同理也不是"结构全灭"(§3.1 的结构失败须在下载后由 loader 判定)。故另名 `CAP2_CANDIDATE_POOL_EMPTY` 呈裁。

**明确未做的那一步(反钓鱼自证)**:`DodgerLoopGame` 只差 TRAIN 带一条(20 vs 下限 40),把带宽放宽到 20 即可立刻凑出候选——**这正是本书明令禁止的动作,故未做,改为如实呈报**。执行方对"放宽 §1 某条 / 更换池 / 记为结构性缺口"三条路径均无裁量权。

**预测对表(4 条)**:P1 过滤轨迹完整且选靶工件先于任何下载提交 ✔(下载恒为 0,轨迹 190 行全文入册);P2 开封密封纪律 —— **不适用**(零开封,故无 sha256 入册对象,首读范围为空);P3 能力考 A5 harm=0 且零越权 —— **不适用**(未跑臂);P4 匹配与否与头条判词不预测 —— 如实报为"未进入匹配判定阶段"。

**义务**:`methods/` `contracts/` `runtime/` `operators/` 零改;阈值/菜单/模板/prompt/模型零改;新增仅 CAP-2 选靶壳层(`evaluation/functional/run_e2_cap2_sealed_batch.py`,提交 `c37948f`);未按值或按预期族挑候选(每一条均为公开元数据字段或冻结模板的机械后果);未越字典序(无可取者);下载 0;`s1_oracle` 与既有密封件未碰;未跑全仓 pytest;零子代理;他线文件未碰。

### CAPSTONE 终考收口(执行方):Epilepsy2 已开封,判词 **`CAPSTONE_NEUTRAL`**;A5−A3 = +0.000000;harm = 0;**卡 Scope 差一叶未匹,条件未进入**(2026-08-28 14:3x,执行方)

**首行读数:CAPSTONE 判词 `CAPSTONE_NEUTRAL`;A5−A3 accuracy = +0.000000;harm = 0。** 工件 `artifacts/functional/e2/capstone_epilepsy2_final.json/.md`(+ 开封记录 `capstone_epilepsy2_unseal_record.json`);成本 **9 LLM(每臂帽 15,Static 0 / A3 4 / A5 5)、9 fit(每臂帽 25)、5.58 min(合计帽 90 min)、下载 0**;单次开封,不重跑。

**开封记录(一行)**:2026-08-28T06:28:56Z 开封 `data/ucr_conf_downloaded/D3_reserve/EpilepticSeizures.zip`(16220082 B,sha256 `72ebe5b2be97…`),首次读取范围 = `EpilepticSeizures_TRAIN.ts` + `EpilepticSeizures_TEST.ts` 全量解析为浮点数组;`val.ts` / `.txt` / `.png` 未开;授权链 = CAP-1b + 主线 13:5x 条 + sol 令。

**Part 0(0 LLM / 0 fit,开封前)**:① **去重记号仪器**按 CAP-1b 预声明落地——评估层从既有字段推导 `dedup_swallowed`(scope 匹配 ∧ 池无 `cand_skill_` ∧ 自提同程序签名),6 项聚焦测试(含一项断言其保持惰性),`methods/` 零改。② **密封复核 11/11 全过,`SEAL_INTACT` 复认**:ROSTER 字节数 16220082、六成员名逐字、TRAIN/TEST/val 未压缩尺寸(272040 / 38806349 / 68077)、原始换行数(87 / 11427)、`@data` 后记录数(80 / 11420)全部对上 CAP-0。**一处如实呈报的规格缺口:CAP-0 从未记录 zip 的 sha256**,故"对 CAP-0 记录校验 zip sha"这一项无基线可比;本书首次计算并入册为后续基线,CAP-0 实际记录的每一项均已复核且相符。③ 自证:TEST 476 行清单由 seed 20260827 重生成且 sha 对上 `7e1c4088…`;菜单名 sha 对上 `48e09ec4…`(序列化 = `json.dumps(menu, separators=(',',':'))`);h0 `runtime_bundle_sha c3427b4e…` / `harness_content_sha 53b1c803…` 对上;mod-4 四分 20/20/20/20 → Support 40 / delayed 40 复现冻结表;条件对 = `fit_only_artifact`。

**两处替换(按 CAP-1b 授权,已引台账)**:§7 开封条件作废(其条件"S1-v2 正序×2 + 反序×1"随 S1-v2 三次 `TREATMENT_EMPTY` 退役后**字面不可满足**),runner 改为从工件机械重导授权链(CAP-1b 在库 ∧ SA-1 r2 记 exit A ∧ G0/G1/G2 全立);**旧 §7 校验器保留且仍被求值并写入工件**(它会拒绝),以便读者逐字看到替换改了什么。§3 A5 池替换:A5 = K0 起源 + SA-1 同源 scope-v2 单例供给卡(v0,内容 sha `00503481…`,装卡前对齐 SA-1 种子 sha 并校验权限为 supply-only)+ R1-R3 修订环开启;三臂骨架保留。

**三臂读数(TEST 476 行干净子集,单次开启)**:Static / A3 / A5 **全部部署 identity**,accuracy 一致 **0.5336134**,逐类 recall {0: 0.4651, 1: 0.5487},逐类 delta 全 0.0,harm 全零;**A5−A3 = +0.000000**、worst-class 差 +0.000000 → 按 CAP-1 §6 三分落 `CAPSTONE_NEUTRAL`(|Δ| < 0.005 且 worst 差 ≥ −0.005 且 harm=0)。A5−Static = A3−Static = +0.000000(§2.1 附报,不替代判词)。

**判词为何是 NEUTRAL——承重解释**:**卡的 Scope 在 Epilepsy2 上未匹配,差且仅差一叶**:`estimated_level_offset` 卡要 `low`,Epilepsy2 读 `zero`(applicability 分 8)。卡因此从未进入 A5 的 Fast 视野,供给 0、`dedup_swallowed` 0(去重记号如实报"Scope did not match",未误记为被吞)。**这正是 CAP-1b 预声明的合法读数**("不保证匹配——不匹配本身是合法读数",按 ITT 记)。**故能力侧主张本次未被考到**:A5 与 A3 的基底差异只剩 K0 起源(h0+三 bootstrap+惰性 Slow 卡)vs 冷 h0,A5 独有的那一件知识出了 Scope,Δ 恒为 0 系构造使然,不是对卡内容的读数。**并且这一叶是先于 L1 冻结的 S1a 家族交集成员,不是 L1 那种偶然叶**——家族轴按规格工作,是 Epilepsy2 在族外。

**附带发现(对后续选靶承重)**:**Epilepsy2 对两个适应臂都没有 headroom**。A3 与 A5 各自独立提出同两个族(`repair_level_shift`、`outlier_mad`),Support 读数全为负(−0.2750 / −0.0500),无 Draft 形成,双臂弃权、部署 identity——弃权是正确行为而非未尝试。**卡自身的程序 `hampel_filter` 本轮无人提案,故其在 Epilepsy2 上的价值未被测量**:本考只证明卡拒绝了,未证明拒绝是最优。修订环相应地正确空转(无供给、无归因于卡的拒绝 → R1/R2/R3 均不写,版本链停在 v0);权限四旗未动、卡未被重铸、无升档、受引导计零红线未破。

**预测对表(4 条,誊自主线)**:P1 密封校验全过 ✔(11/11);P2 A5 harm=0 且零越权 ✔;**P3 卡 Scope 按注入族匹配 ✘ 破**(machine_match False,供给 0)——这是本考唯一被证伪的预注册项,如实记;P4 头条不预测(照三分如实报)。

**本考许可与不许可什么(诚实边界)**:**不构成能力主张**——条件化主张的条件未进入。**也不构成对 SA-1 的反证**——r1/r2 的 exit-A 机制证据不受"卡从未声称的靶"影响。**它确实在一次性开封的密封材料上证成了安全侧那一半**:以阶梯最低档买到的单例卡,面对其证据从未覆盖的域**主动不供给**,零成本、零 harm、零越权。**能力级判词需要一个落在卡家族之内的密封靶,本线当前没有**——此为下一步选靶的承重输入,不由本书裁定。

**义务**:`methods/` `contracts/` `runtime/` `operators/` 零改;阈值/菜单/模板/prompt/模型零改(全部常量读自 `cap1_capstone_protocol_freeze.json`,注入模板 = `cls._inject_v2` @ `helpers['positions']` 原件);`s1_oracle` 未碰;下载 0;单次开封无重跑;未跑全仓 pytest;零子代理;他线文件未碰。

### SA-1 r2 收口(执行方,纯采样重复):**CAP-1b 出口 A** — G0/G1/G2 三门全立;七条预注册全立(r1 破的 P4 在 r2 立);无 r3(2026-08-28 13:5x,执行方)

**首行读数:CAP-1b 出口 = A;G0 立 / G1 立 / G2 立(exercised)。** capstone 形态按 CAP-1b 表 = **A3-reset vs A5-adaptive**(scope-v2 单例卡 + R1-R3),头条主张 = 带 Scope 经验卡 + 反馈修订的端到端收益。工件 `artifacts/functional/e2/sa1_minimal_r2.json/.md`(+ `.checkpoint.json`),r1 三件未覆写(见下自证)。成本 **80 LLM(书面帽 120)、71/300 fit、2580 s / 10800 s、下载 0**。

**卡版本链 v0 `00503481` → v1 `0eae563f`(R1@GunPoint) → v2 `3ef7202e`(R1@GPOvY) → v3 `89728a4a`(R2@PowerCons#1) → v4 `64c1cb59`(R2@Herring)。** 与 r1 链**前四版内容 sha 逐字节相同且同序**,r2 多出 v4 —— 这不是分歧而是多走了一步:**同样的读数编出同样的修订内容**,说明修订体是触发它的读数的确定性函数;而 r1 从未见到 Herring 的拒绝(其供给被去重吃掉)。

**三门逐项**:**G0** harm 全零、三臂 worst-class 最差均 0.0000;**零扩 Scope**(A5 逐单元 Scope 判定是 K0 冻结卡的子集,无一处反向);无非阶梯规则、无非写回产出的版本、两处授权面之外零 PATCH;卡在视野的每个单元均记 `guidance_conditioned=true`,受引导计零红线未破。**G1** 4 次写回(R1×2、R2×2),版本链长 5,五个内容 sha 互异。**G2 exercised**:两个 Scope 匹配单元发生供给候选被拒(PowerCons#1、Herring,均 CONFLICT),**两次 R2 均如期触发**;首遇拒发生于 PowerCons#1,故读再遇位——**PowerCons#2:A5-adaptive scope=False / 零供给 / 零挨拒 / 1 probe;K0-fixed scope=True / 再供给 / 再挨拒 / 2 probe**,条件式子项成立。

**预测对表:七条全立(r1 为 6 立 1 破)。** P1 匹四不匹一、P2 GunPoint+GPOvY 双转化、P3 恰一次首遇收窄且 sha 版本化、**P4 立**(Herring 走了 P4 明文允许的"被拒"支路:供给入池、Target 判 CONFLICT)、P5 再遇位机制差、P6 probe ≥1 省且 regret 非劣(**本跑 1 省 1 拒全部可归因于收窄**)、P7 harm 全零。

**r1-r2 并排(数字不复现不构成失败,按 CAP-1b)**:**复现的**——三臂累计 regret 逐字节相同(A3 +0.7710 / K0 +0.0850 / A5 +0.0850)、regret 差 +0.0000、卡对无卡差 +0.6860、harm 全零、K0 供给 5 / 转化 2 / 挨拒 3、避免挨拒可归因数 1。**未复现的**——raw probe 省 2→1、raw 避免挨拒 2→1(r1 那额外一次来自 Herring 去重,r2 无去重);**可归因 probe 省 0→1**(r1 收窄腾出的槽被 agent 自己的提案填回,r2 没有);A5 供给 3→4、挨拒 1→2;逐 cell 有 8/18 格 probe 数不同(采样)。**去重普查本跑为空**——P4 第三路径未再现;去重不是拒绝,不入 G2 条件式。

**两条必须留档的边界**:(1) **v4 的"无过度排除"在课程内未受检**。两条排除子句对六个单元各自恰好只排除其触发单元(#3 只排 PowerCons、#4 只排 Herring,均按冻结 pattern view 逐叶核算),但 v4 落在位置 4,其后唯一的族内单元 PowerCons#2 已被 v3 排除,故"v4 单独会保留的活单元"本课不存在——该性质只对六份冻结视图成立,未经 live 检验。(2) **修订环本跑仍只买到成本、没买到质量**:两个带卡臂 regret 恰好相同(+0.0000),预注册只声称非劣;而 +0.6860 那个大数与 r1 逐字节相同,归 **Part 0.5 家族 Pattern 轴**(把 Scope 覆盖从 L1 n=1 卡的 1/5 抬到 4/5),不归修订环。R3 两跑均未现场触发(harm 全零、无 NEGATIVE 供给),仍只有离线历史重放背书——CAP-1b 已预声明不读 R3。

**协议同一性自证 + 一处对冻结件的事实更正**:课程冻结件、种子卡(v0 sha `00503481…`)、阈值、预算口径、后端协议逐字节同 r1;`git diff cf2eb12 HEAD` 在 `methods/`、`contracts/`、`runtime/`、`operators/`、`evaluation/minipipe/`、`run_e2_s1_curriculum_four_arms.py`、`task_episode_harness/`(含 `source_skill.py`/`skill_revision.py`)**全空**——实验面零改动。**但 CAP-1b 第 13 行"现 HEAD 代码面未变"字面不成立**:`5ff76b5` 对 `run_e2_sa1_minimal.py` 有 +182/-8,全部落在跑后判分/渲染区(`_attribution`、`_headline`、`_honest_boundaries`、`_markdown`、`--finalize` 承接,以及 `_verdict` 一处**收紧**——机制差须可归因于收窄而非仅 raw);live 回路、修订步与逐单元判分未动。**决定性的一点:r1 已入库工件本身就是用这同一份 HEAD 代码重渲的**,故 r1 与 r2 由同一字节判分,并排表可比。此为对冻结件前提的事实更正,不是协议变更。**工件路径系事后搬移而非参数化**(runner 硬编码 r1 路径,改之即代码改动,为本书所禁):r2 写在 r1 文件名上,产物随后搬为 `sa1_minimal_r2.*`,r1 三件由 `5ff76b5` 恢复并与跑前副本**逐字节校验相同**(json/md/checkpoint 三项 hash 全等)。

**判分脚本声明**:CAP-1b 三门读数与并排表由一份**未入库**的 `_scratch` 分析脚本算出,输入全部是两份跑件里已有的字段(逐单元 `scope_match_by_skill_id`、`card` 读数、`revisions[].receipts`、`card_version_chain`、冻结 `pattern_views`),因此每条判定都可只凭 JSON 复核;不入库正是为了守住"代码零 diff"这条义务。runner 生成的 `honest_boundaries` 系按 r1 结局写的静态模板,其中数句对 r2 为假(P4 第三路径、probe 归因),已在 r2 件中替换为 r2 自己的边界,模板原文完整保留在 `honest_boundaries_from_r1_template` 键下,未静默丢弃。

**收口与交接**:按 sol 令 **r2 后无论出口即停,不跑第三次**;capstone 不在本书内,待主线按 CAP-1b 出口 A 另书发车(预声明的去重记号仪器为 r2 后、capstone 前落地项)。证据等级 development,单元均已曝光,种子系 v4 记录 Episode;**两跑买到的是机制三门同向两次**(CAP-1b 所求),不是数字复现(CAP-1b 明文不要求)。**义务**:代码零改动零提交;阈值/TRY/RISK/执行/部署门与 prompt-模型-预算协议零改;`fault_routes.json`/`router.py` 零改;密封件(Epilepsy2、s1_oracle 键)零读;零新数据/单元/算子/Consumer;零下载;零子代理;未跑全仓 pytest;他线文件未碰。

### SA-1 最小闭环收口(执行方):四段全过,判词 `SA1_DEVELOPMENT_SIGNAL`;**核心正效果移动:是,但只值 1 次避免挨拒**(2026-08-28 12:1x,执行方)

**首行读数:核心正效果移动 = 是,probe 省 2(raw)/ 避免挨拒 2(raw)/ regret 差 +0.0000。归因后**:2 次 probe 省**零条**归因于收窄,2 次避免挨拒中**恰 1 次**归因于收窄——即预注册的再遇位 PowerCons#2。工件 `artifacts/functional/e2/sa1_minimal_r1.json/.md`(+ 门件 `sa1_minimal_gates.json`、冻结件 `sa1_course_freeze.json`);成本 **78/150 LLM、69/300 fit、2227 s / 15120 s、下载 0**;Part 0/0.5/1 全程 0 LLM / 0 fit。

**Part 0(四归因字段,纯增零行为)**:`episodes[].source_skill_id`(`online_loop.py` 从 `cand_skill_<id>` 前缀落记录,与 `fast_agent.py:365-369` 同源解码)、`episodes[].source_skill_revision`=**卡内容 sha**(`source_skill.skill_content_sha`,不动 `SkillEntry.revision`、零铸卡点改动)、`round.scope_match_by_skill_id`、`round.guidance_conditioned_by_skill_id`(以"是否在视野"定义,替代按单元位次派生)。门:在 **L1 r1 记录上离线回填**——GPOvY 行 `source_skill_id=l1_ladder_v2_supply_v1`、`scope_match=true`、`guidance_conditioned=true`、revision sha 非空,其余各行 `source_skill_id=null`;5 项聚焦测试 + 既有 105 项全绿(`test_skill_revocation.py` 因 py3.12 f-string 语法在 3.10 环境不可收集,**先于本书存在**,未改)。

**Part 0.5(Scope 规则 v2,承重五轴)**:Pattern 家族轴取 **S1a 簇资格判定的 Pattern 交集**——`run_e2_s1a_curriculum_oracle_audit.py:619`(`_compatible_clusters`,交集helper `:609`),冻结于 `s1a_curriculum_audit.json → part_b.clusters[hampel_filter].pattern_intersection`(10 叶,**先于 L1**,非现场发明、非按结果挑叶)。编出卡 **8 叶 / 8 不同特征**(task_kind 重复叶自然消失,Q11 关闭);**匹配表恰如预注册:GunPoint / GPOvY / PowerCons / Herring 全匹,BirdChicken 不匹**(唯一差异叶 `estimated_level_offset` low vs medium)。L1 那条决定两次未匹的偶然叶 `period_change_score` 不在家族定义内。**Q7 照旧诚实声明**:3 条家族叶(`level_region_end_fraction` / `level_region_fraction` / `outlier_region_end_fraction`)edit schema 无契约被丢,有效 Scope 比记录宽,排除规则继承同盲区。

**Part 1(三写回 + 离线 replay 门)**:全部 PATCH 走 L1 装卡的冻结 EditController(SHA 前置),`fault_routes.json`/`router.py` 零改动。(a) R1 以 L1 r1 的 GPOvY 转化追加一行 `risk_guards.evidence_ledger`,Scope 与权限档字节未动;(b) R2 由 **v3 PowerCons CONFLICT** 驱动,从拒绝单元 binned view 机械编译出 `not(period_change_score == very_low)`,PATCH 进 `observable_applicability`;R3 由 **ECG200 outlier_mad 害证** 驱动——**如实报:该单元在 12 条契约保证轴上与卡的证据完全一致,无可区分轴,故只落结构化降权注记、未产生排除**(规则拒绝现场发明轴);(c) 收窄后对非拒绝单元匹配逐字不变,只 PowerCons 由 true→false。四版内容 sha 互异、快照血缘完整、`set_active(父 sha)` 可回滚。**Q1 残留照实记**:`.observable_applicability` 面在冻结路由表里只被 `RETRIEVAL_MISS` 授权,而该因码语义是"该检索到却没检索到"(扩的方向);本书按令不新增因码、不改路由,直接以它作收窄 PATCH 的授权令牌,并把语义错位留作 Q1。

**Part 2(live 短课 r1,六位三臂)**:GunPoint → GPOvY → PowerCons#1 → Herring → PowerCons#2(再遇) → BirdChicken;A3-reset(无卡)/ K0-fixed(卡 v0 冻结)/ A5-adaptive(同卡 + R1-R3),两个带卡臂除"可否修订"外逐项相同(均不携带 Episode)。**卡版本链 v0 `00503481` → v1 `0eae563f`(R1@GunPoint) → v2 `3ef7202e`(R1@GPOvY) → v3 `89728a4a`(R2@PowerCons#1)**。逐位:1 GunPoint 两卡臂均转化 regret −0.0667(A3 +0.4067);2 GPOvY 均转化 −0.0286(A3 +0.1841);3 PowerCons#1 均供给均被拒(CONFLICT)→ **恰一次收窄 PATCH**;4 Herring;5 PowerCons#2 **A5 零供给零挨拒 / K0 再供给再挨拒**;6 BirdChicken 两臂均不匹(族外对照成立)。累计 regret(仅 distinct 五位)A3 **+0.7710** / K0 **+0.0850** / A5 **+0.0850**;harm 全零,worst-class 最差 0.0000。

**预测对表 7 条:6 立 1 破,破的照报。** 立——P1 匹四不匹一 ✔;P2 GunPoint+GPOvY 双转化 ✔;P3 PowerCons#1 被拒→恰一次收窄 PATCH、sha 版本化 ✔;P5 再遇位 A5 零供给零挨拒 / K0 再供给再挨拒 ✔(**本书唯一干净的机制差**);P6 regret 非劣 ✔(probe 项见下);P7 harm 全零 ✔。**破——P4**:Herring 既没被拒也没被 v1 排除。实况是第三种走法:卡匹配、卡在视野,但 **Fast agent 自己提了同一冻结程序,机械供给被候选去重吃掉**,故池中无 `cand_skill_` 条目;结局与 K0-fixed 相同(identity 部署、+0.0469),只差一个 probe。

**两条必须写进台账的诚实边界**:(1) **probe 省数不归收窄**——2 次 raw 省分别来自 Herring(候选去重)与 BirdChicken(agent 提案数波动),再遇位收窄虽移除了供给候选与随之而来的挨拒,**agent 立刻用自己的提案填回了那个槽**,故该位 probe 数持平。本跑可归因的机制差 = **1 次避免挨拒**,位置与冻结件点名的完全一致。(2) **+0.6860 那个大数属于 Part 0.5,不属于修订环**——两个带卡臂同为 +0.0850、无卡臂 +0.7710;这五个 distinct 单元与 identity 基线正是 v4 尾段那五个(v4 尾段四臂亦为 +0.7710),故可与 L1 的 +0.2127 同底比较:**把 Scope 覆盖从 1/5 抬到 4/5 的是家族 Pattern 轴,不是修订**。修订环本跑买到的是成本(少挨一次拒),不是质量(regret 差恰为 0.0000),预注册只声称非劣,不多claim。

**证据等级与措辞**:development,单跑,单元均已曝光,种子系 v4 记录 Episode 而非现场重挣;判词按纪律只记 `SA1_DEVELOPMENT_SIGNAL`,复合措辞仍须采样重复(本书未授权 r2)。**止损未触发**:PowerCons#1 如期被拒且 R2 如期触发,无尸检。**义务**:阈值/TRY/RISK/执行/部署门与 prompt-模型-预算协议零改;`fault_routes.json`/`router.py` 零改;密封件(Epilepsy2、s1_oracle 键)零读——本书一切 Pattern view 均由 `extract_public_features` 现算于已建 cell(生产路径);零新数据/单元/算子/Consumer;零下载;零子代理;未跑全仓 pytest;他线文件(AGENTS/README/PROJECT_STATE*/SUCCESSOR_BRIEF*/ROADMAP)未碰。

### L1 收口裁定(主线):判词维持 `L1_SIGNAL`×2;阶梯 v2 机制证成;缺口定名"Scope 面积价";SA-0b 反事实简报;三项待裁呈用户+sol(2026-08-28 10:3x,主线)

**裁定**:执行方两跑判词与门核算全部核可(**+0.2127 ≥ Δ 0.088462,2.4×;harm/worst-class 全零或正向;两跑读数逐字同向,复合措辞许可**;成本 40/120 LLM、24 min)。**里程碑定名**:首次完整落地"课程内自产 Episode → 单例供给卡 → Scope 检索 → 机械供给 → 当前 Target 双门 → 材料级 regret 改善"的端到端复利链(development 级、GunPoint 族内、单场转化、重放级对照,四注记照录)。用户论题前半("Skill 由经验初始化、天然窄、低权入场、由当前数据检验")就此有实证;**methods 本夜零改动**(惰性谓词天然放行供给卡,carve-out 未动用)。网络中断致执行者终回执未达,工作已全部入库(`74978c0`/`ecbe116`),按工件收口,无缺口。

**新知识入典(比 SIGNAL 更重要的定价发现)**:低价的真实代价是 **Scope 面积**——n=1 退化交集把预测 4 场压到 1 场(`period_change_score` 一叶决定 GunPoint/PowerCons 两次未匹)。阶梯 v2 的完整定价画像 = **通道稳定(两跑逐字同向)、入口稀缺(产例 0.29/位)、面积按价收缩(1 证据 = 1 场)**。

**SA-0b 反事实简报(主线自算,0 LLM,基于 r1/r2 + v4 账本;原 Part C 任务就此收口)**:(a) "冲突→收窄"反事实在本账本**空转**——唯一匹配场两跑均转化,零拒绝事件,无可修订素材;SA-1 须自带冲突场(设计稿 §5 已按此把 PowerCons 前置)。(b) 钝撤权反事实同空转(供给候选零 CONFLICT/NEGATIVE)。(c) **Scope 面积机会成本可量化**:GunPoint 若匹配(差一叶)并按其强余量转化,尾段 regret ≈0.15,预注册 ≤0.20 即达——**残余 0.5583 中 +0.4067(73%)由一叶之差解释**;PowerCons(Support 贴线 0/2)与 Herring(held-in=0)即便匹配也大概率零转化。损失集中度 100% 在 GunPoint,此即 Q9(单例卡初始 Scope 宽度定价)的实账输入。

**三项待裁(呈用户+sol)**:(A) **capstone 开封**——"两跑同向 SIGNAL"已达,但 CAP-1 冻结的开封条件系为 S1-v2 原判词设计,L1 系机制修订后的重放,判词等价性须 sol 核;开封另需用户确认(密封 Epilepsy2,下载条款照 2026-08-27 16:2x)。(B) **SA-1 发车**——设计稿 `docs/SA1_SKILL_ADAPTATION_DESIGN_2026-08-28.md` + 接线审计 `sa0_wiring_audit.*` 已呈,Q1-Q12 待裁;其中 Q6(四个归因字段,纯仪器)主线建议先行落地。(C) **Q9 Scope 宽度**——0.4067/一叶的实账入册后,是否为供给档定义事先冻结的轴选择规则(禁按结果挑轴)。主线推荐:A 与 B 并行呈裁,批后 capstone 先跑(头条),SA-1 Part 0 仪器随后。

**提交**:`sa0_wiring_audit.json/.md` + `SA1_SKILL_ADAPTATION_DESIGN_2026-08-28.md` + 本节(主线提交)。

### L1 收口(执行方):两跑 `L1_SIGNAL` 同向;**核心正效果移动:是,+0.2127**(A5 尾段 regret 0.7710→0.5583,门 0.088462,harm 0)(2026-08-28 04:4x,执行方)

**首行读数:核心正效果移动 = 是,+0.2127。** r1 与 r2 逐字相同(0.7710 → 0.5583,门 0.088462,harm 0),**两跑同向,准复合措辞**。成本合计 **40/120 LLM(每跑 20)、33/300 fit、1428 s / 14400 s、下载 0**。

**T1 离线门六检全过(0 LLM/0 fit,6.0 s)**:① 档价常数 `SUPPLY_TIER_MIN_DISTINCT_TASKS = 1`(`source_skill.py:319`);**TRY 档零触**——`authorization_audit` 的 `loo_minimum` 与 `active_try_authorized` 逐字未动(`source_skill.py:250`),供给档与 TRY 档双向不干扰测试仍全绿。② 单元 3 边界**确定性编译**出单例卡(两次编译字节相同):权限四旗 `supplies_candidates=true / grants_execution=false / reorders=false / suppresses=false`,`evidence.source_count=1`,证据 = GPMvF 未引导正例(Support +0.1905 / delayed +0.0526 / 部署 +0.1867);**双门"强正例"由行构造层强制**(`run_e2_s1v2_forward_course.py:1627` 只收 `relation==POSITIVE ∧ local_status==LOCAL_ACTIVE`)。③ **惰性谓词天然不拦**:供给卡无 `risk_guards.sections`,`_experience_card_sections` 返回 None,`_is_inert_experience_card`(`retrieval.py:195`)在读任何子句前即返 False——**无需 carve-out,本夜 methods 改动数 = 0**。④ Scope 匹配预检见下。⑤ 注入干跑:卡在匹配单元入 Fast 视野、`_supply_rung_candidates`(`fast_agent.py:386`)物化候选、verifier 接受其几何(selectable)。⑥ 受引导计零:同一批 Episode 标 conditioned 后 `unguided_positive` 归 0(`source_skill.py:366`),单例卡无法为自己升档或扩 Scope。

**断点方式声明(诚实边界)**:采用**边界重放**——从 v4 记录的单元 3 后边界恢复,**产例阶段 1-3 不重跑**,唯一变量 = 档价。**携带**:按新价从记录的 GPMvF Episode 编译出的供给卡,经冻结 EditController 装到 K0 上。**未携带**:A5 的内存 Episode 对象与其单元 3 的 Target-local capability——Episode 行以卡的 evidence 块存活;Target-local Skill 带域戳,对任何尾段单元本就不适用。如实声明以免高估归因。

**Scope 匹配预检:预测 4/5,实测 1/5(重大预测落空,单列)**。命中仅 `GPOvY`;`GunPoint`/`PowerCons`/`Herring` **均不匹**,`BirdChicken-burst` 不匹(如预测)。**原因是修订案自身的代价**:n=1 时"五轴退化交集"= 该单条 Episode 的全部已录 pattern 叶,比两 Episode 交集**窄得多**——低价买到的卡,适用面也随之收缩。这是阶梯 v2 的真实定价后果,不是实现缺陷,应入典。

**逐单元表(A5 尾段,候选来源分列;r1/r2 同形)**:位置 4 GunPoint(供给 0 / 自提 2 / identity / regret +0.4067);**位置 5 GPOvY(供给 1、被探 1 / 自提 1 / 部署 `hampel_filter` / held-out +0.2127 / regret −0.0286 / worst-class +0.1879)**;位置 6 PowerCons(供给 0 / identity / +0.1333);位置 7 Herring(供给 0 / identity / +0.0469;r2 零提案弃权);位置 8 BirdChicken(供给 0 / identity / 0)。**唯一转化来自供给候选**,且发生在预注册的强受益场。

**预测对表(12 条:7 立 5 破,破的全报)**:立——卡边界确定编译 ✔;Scope 匹 GPOvY ✔;Scope 不匹 BirdChicken ✔;GPOvY 转化 ✔;PowerCons 不部署 ✔;Herring 不部署 ✔;harm=0 ✔。**破**——Scope 匹 GunPoint ✘、匹 PowerCons ✘、匹 Herring ✘(n=1 Scope 过窄,见上);GunPoint 转化 ✘(未匹即无注入);**A5 尾段 regret ≤0.20 ✘(实测 0.5583)**——预测把"四场转化"折进去了,实际只有一场转化,故只回收 +0.2127 而非 ≥0.57。

**门核算**:regret 改善 **+0.2127 ≥ Δ_material 0.088462** ✔;harm 事件 **0** ✔;worst-class 最差 **+0.1879**(正向)✔。**对照口径**:v4 尾段四臂 regret 全为 **+0.7710**(全 identity),系**重放级冻结对照**,非同期对照——L1 只重跑 A5 尾段,其余三臂沿用 v4 读数,**不得表述为同期四臂比较**。

**证据等级与边界**:development(单元均已曝光、GunPoint 族内、单场转化、对照为重放级)。**新颖主张仅限**:证据价降至 1 强正例后,课程内自产的单例供给卡**确实**编译、被送达、被 Target 双门裁决、并在强余量场产生材料级 regret 改善且零 harm。**不主张**跨族能力,不主张多场复利。

**书外发现**:(1) 低价的真实成本是 **Scope 面积**——n=1 退化交集把可作用面从预测的 4 场压到 1 场;若要恢复覆盖,需要的是第二条 Episode(回到 2 价)或一条**显式的 Scope 泛化规则**,而后者本夜明令不动。(2) 两跑读数**逐字相同**(+0.2127 / −0.0286 / +0.1879),说明该转化在当前采样下**高度可复现**,与产例侧 0.29/位的抽签形成对照——**通道稳定,入口稀缺**。(3) r2 的 Herring 出现零提案弃权(r1 为 1 探针),弃权行为本身正确(held-in=0),但再次显示 A5 在无卡可用的场上倾向沉默。

**提交**:代码 `74978c0`(档价 2→1 + L1 机制 + 测试更新,68 项供给/检索测试全绿);工件本次提交(`l1_ladder_v2_replay_r1.json/.md`、`_r2.json/.md` 及两份 checkpoint;v4 件未覆写)。`methods/` 本夜零改动;TRY/RISK 档、执行与部署门、MATERIAL、prompt/模型/预算协议零触;密封件(Epilepsy2、s1_oracle)零进臂视野;密钥零出现;未跑全仓 pytest。**capstone 今夜不开封**,CAP-1 就绪清单随交付另附。

### 阶梯修订案 v2(供给档证据价 2→1,sol 案采纳);L1 = v4 课程断点重放发车;主线自认 0.85 外推数学错误(2026-08-28 02:3x,主线)

**机制重审裁定(sol 提案,用户核可,主线定稿)**:三入口中 **(i) 增轮否**——主线 0.85 系把"每位两轮后 0.29"误作单轮率重复计数,sol 纠正成立且主线独立复算一致(单轮 p≈0.155,R=4 每位 ≈0.49、三位凑二 ≈0.49,提案相关性使真值更低),抽样增轮不修"价格-供给率失配"根因;**(iii) 收缩主张否**(用户定调正效果优先)。**(ii) 采,原则化版本:证据价与权限相称**——**1 条强正例**(Support+delayed 双门 POSITIVE)即许编 **supply-only 卡**(`supplies_candidates=true / grants_execution=false / reorders=false / suppresses=false`),不占 agent 自主探索槽,候选仍过 verifier + 当前 Target Support/delayed 双门,harm 否决兜底;**2 条独立未引导正例 → 现行 Source 供给档(交集 Scope)不变;TRY 档(LOO)/RISK 档/执行与部署门全不动**。**Scope 归纳规则 v1 修订**:供给档许 **n=1**,五轴与"dataset 名禁作轴"照旧,取值 = 该单一 Episode 已录字段本身(退化交集,天然最窄)。**防自举**:单例卡引导所得正例按既有 UNGUIDED 规则计零,不得据以升档或扩 Scope;Target 反证即 `restricted_by_target_feedback` 收回检索(retrieval.py:143/269 现役)——**"负反馈→撤权"半环已在产线,系单例低价的事后兜底;"冲突→收窄 + 版本化修订"(用户"Skill 持续适应"愿景)留 SA 线白天设计呈 sol,不入今夜刀口**。谱系:C40"权力越大 Scope 越窄"之对偶(权力最小→门最低+Scope 最窄)。备忘不动刀:单例卡 prose 影响面定价(机械通道 only)记 Stage-3 议题,今晚不加第二刀。

**L1 发车(重放非重掷,opus 续话)**:v4 冻结件原样;**从 v4 A5 单元 3 后边界断点恢复**(GPMvF 未引导正例已在账,档价改 1 → 卡确定性编译;不重掷产例阶段,归因唯一变量 = 档价;重跑整课至少一中仅 ~0.64,弃),A5 尾段(GunPoint→GPOvY→PowerCons→Herring→BirdChicken)live 重放,其余臂用 v4 冻结读数作重放级对照;ITT 主判,Δ_material=0.088462 沿用;r1 SIGNAL → 采样 r2,两跑同向才许复合措辞;**development 级**(单元均已曝光、GunPoint 族内),capability 级留密封 Epilepsy2(CAP-1,明早呈批,今夜不开封)。**终掷硬帽语义**:L1 非第四次课程重排,系 (ii) 机制修订后的重放;**L1 亦空则为阶梯 v2 负证据,直接入机制重审案卷,无 L2 重试**。**T1 离线门(0 LLM)六检**:① 供给档价常数定位改 1(引 file:line;TRY 档 LOO 零触)② 单例卡边界编译重放(v4 checkpoint store)③ T1 惰性谓词兼容(supply 卡不得被 `_is_inert_experience_card` 拦;若结构性冲突,最小 carve-out = `authority.supplies_candidates==true` 非惰性,引行+聚焦测试)④ Scope 匹配预检表(尾段 5 单元 frozen pattern view)⑤ 注入干跑(候选物化 + probe 槽)⑥ 受引导计零复核。**预注册预测(可证伪)**:卡边界确定编译;Scope 匹 GunPoint/GPOvY/PowerCons/Herring、不匹 BirdChicken-burst;转化 GunPoint+GPOvY、PowerCons 不转(Support 贴线 0/2 实测)、Herring 弃权(held-in=0);harm 全零;A5 尾段 regret +0.7710 → ≤0.20。**预算帽**:LLM ≤120 / fit ≤300 / 墙钟 ≤4h / 下载 0。禁令:无新单元/无 prompt-模型改/密封件零触/TRY-RISK 档零触/密钥零出现/未授权不 spawn。

### v4 终掷判 TREATMENT_EMPTY(第三次,提交 9894a5d);硬帽生效,S1-v2 全停;系统性结论定案呈机制重审(2026-08-28 00:5x,主线)

**读数**(115/280 LLM,77 fit,3206s,三课累计 306 LLM;harm 四臂全零):产例 B/GPMvF 命中(A5 自提 hampel,Support +0.1905 POSITIVE、delayed +0.0526、部署 +0.1867);产例 A/GPA 与备份 C/GunPoint 上 A5 唯一非 identity 提案均为 level_shift 族且**均被 verifier 拒**;卡 1/2 未编,受益注入 0/2。regret 门数值过但**归因门不过**(差值来自产例 B 同单元学习,非跨单元复利)。**系统性结论(三课汇总)**:A5 产例位 7,提对族 3(43%),过双门 2(**29%/位**);供给档价 2 正例下三产例凑满概率 **≈0.19**;重排课程、判据升级、备份产例三招俱试——**首处故障定名"提案语义 × 供给档价格"失配**;通道本身三书已证(W-1 读端/G-3 条件化否决/P0 产端)。**单点铁证**:v4 产例 A 同单元同协议,K0 提 hampel +0.2690、A5 提 level_shift 被拒——采样方差主导臂差,单元级证据。**新具体缺陷**:冷 proposer 的 level_shift 偏好连合法性校验都不过(提案位纯浪费),系 prompt 或候选契约绑定问题(Stage-3 答案题第三道)。**机制重审三入口(明早呈用户+sol,主线倾向 i>ii>iii)**:(i) **产例单元轮次增额**(R=2→4,纯评估层协议参数,零治理改动,每位命中 ~0.3→~0.75,三产例凑二 ~0.85;需用户/sol 特批解除主线自立的终掷帽一次);(ii) **供给档价格修订至 1 正例+最窄 Scope**(权限梯逻辑支持[建议权本就低价],G-3 否决机制兜底,但改已批阶梯须 sol 级修订案);(iii) **接受负结论重构主张**:复用链因果证据已齐(W-1 +0.2127/M-1 门控/G-3 治理),课程内自举演示受 proposer 吞吐限制,论文以 pilot 卡演示为主线、自举瓶颈为已量化边界。备份产例"受引导计零"分支未触发未实测;PowerCons 密封-live 落差(提案参数绑定)入件。

### v3 r1 收口(提交 a1f5134):TREATMENT_EMPTY 但性质跃迁——产例 A 全链自产成功;判据补第三项;课程 v4 终掷(2026-08-27 23:4x,主线)

**读数**(95/250 LLM,73 fit,4228s,harm 全零):**产例 A/GPA 课程内全链自产成功**(A5 r1 自提 hampel,Support +0.45 POSITIVE,delayed +0.40,部署 +0.2690,regret 略优 oracle)——"课程自产知识"首次真实发生;产例 B/PowerCons **族提对了**(hampel),Support 读 +0.0357 判 CONFLICT(聚合正逐序列害,EPISODE_ONLY)——**断点从提案语义移到 Support 分级**;卡 1/2 未编,受益注入 0/2。**v2 vs v3 性质区分入案**:结构性空场 vs 采样差一发(管线全通)。附带负读数如实记:A5 在两受益场 0 提案弃权,丢掉 A3 自挣的 +0.1867;K0 连续两跑靠 GPMvF 自挣 hampel 得最低 regret(处理组空,不可比)。供给档七拒零凑数照旧。**诊断升级(执行者)**:冷发现率先验只覆盖"提出",不覆盖"过 Support 双门";live 读数与密封余量算术的落差另源于提案参数差(agent 的 hampel 变体参数 ≠ oracle 默认)。**主线三答组成课程 v4(终掷)**:(b) 产例判据 = 发现率 × live Support 通过率 → 产例 B 换 GPMvF(M-1 对半 3/4 实证),PowerCons-impulse 降弱受益场(Support 面实测贴线,诚实弱分层);(c) GunPoint-impulse 作备份产例 C(任一边界凑满 2 正例即编卡,保险成本 ~20 LLM);(a) v4 即修正后重试,实例化概率 ~0.8;**硬帽:v4 系 S1-v2 最后一次课程尝试,第三次空场即系统性结论,全停呈机制重审**;家族注记(卡证据或为 GunPoint 族内相关)如实入件。课程 v4:GPA→identityA→GPMvF→GunPoint(备份产例)→[边界]→GPOvY(强受益)→PowerCons-impulse(弱受益)→Herring(HELDOUT)→identityB。

### sol 批案 A(用户定调:正效果优先);S1-v2 课程 v3 重冻发车(2026-08-27 22:0x,主线)

**裁定**:案 A 立即推进,B 收编为一行命名(discovery-reliable development curriculum)与模块化对照框架(自然自举课程=发现模块负结果保留;本课程=积累模块正控),**不加任何新声明/资格工序**;发现率瓶颈记后续优化点(Stage 3 的 Instruction/策略进化),不占关键路径;GPA/PowerCons 作产例的排除令语义修订获 sol 核(约束对象=K0 空 + 受益≠产例单元;课程内重挣不破自产性)。**主线补裁**:GunPoint 冗余产例不加(与 GPA 同族,证据计数添乱;GPA 2/2+PowerCons-impulse 2/3 实证命中率 + 每单元两轮已足)。**课程 v3**:GPA(产例A)→identity A→PowerCons-impulse(产例B)→[边界产卡]→GPOvY(受益强)→GPMvF(受益弱)→Herring(HELDOUT_ONLY)→identity B;判分/预算/ITT/材料门沿 r2 冻结;重复=采样重复(如实标);r1 SIGNAL 自动续采样 r2(分支③);TREATMENT_EMPTY 即停。发 opus 续话。

### S1-v2 正序 r1 判 TREATMENT_EMPTY(提交 cf06343);断裂在产例可提出性;分支①启动呈裁(2026-08-27 18:5x,主线)

**判定**(96/250 LLM,68/900 fit,3285s;r2 按预注册停发):两项裁定执行确认(双受益+前曝光注记+Δ_material=0.102632);七边界审计全拒 `fewer_than_2_distinct_unguided_positive_tasks`——全课程合格 Episode 仅 1 条且族为 outlier_iqr(产例 A 自提 iqr:held-in 批 +0.0444、held-out worst −0.0667 harm 1;产例 B 零收获);**agent 在两产例均未提出 hampel,供给卡从未编译,受益注入 0/2(ITT)**。**断裂定位(执行者原句入典)**:"余量说的是'若探到该族则读得出',不是'会探到该族'"——课程判据混淆可读性与可提出性;提案语义瓶颈第四次现身(S1c 漏斗/PS-1/K 消融/本轮自举层)。**供给档零知识行为正确**(七拒同因,零凑数);处理组空时四臂对比无解释力(K0 最低 regret 系 GPMvF 自然挣得 hampel 的采样方差,不得读作 K0>A5)。**仪器事实**:本注入族无 RNG,"异注入 seed"虚构,重复=采样重复(replicate_kind=sampling 已改标;反序/重复书须先修重复语义)。**分支①裁决件(呈用户/sol,主线荐案 A)**:产例选择判据加**可提出性先验**(该族在该单元被冷 proposer 实证提出过)——实证发现账:GPA 2/2、PowerCons-impulse 2/3、GPMvF-halved 3/4、GPOvY 0/5+;**案 A**:产例改 GPA+PowerCons-impulse(+GunPoint-impulse 冗余第三产例),受益仍 GPOvY+GPMvF——关键论证:双源单元排除令本意防"卡带资进场",约束对象应为 **K0 与受益≠产例**,产例场上 agent 课程内**重新挣得**新 Episode 不违自产性(K0 仍空,卡仍课程自产);须 sol 核此排除令语义修订;**案 B**:判"自然自举形态在提案语义未解前不可考",演化主张重构为"复用链因果已证(W-1/G-3/M-1)+ 自举吞吐量已量化 + 复利曲线在发现可靠课程上演示";**案 C**:攻提案语义本身(observation 契约扩展,重手术,先不动)。荐 A(含 B 的措辞收编:课程定名 discovery-reliable curriculum,如实注记)。

### 跑批间隙并行预备(sol 清单采纳)+ 分支任务书预写(2026-08-27 18:1x,主线)

**PREP-1 发车(grok,零 live/零 CPA/零 D3/不碰在飞文件/不写本档)**:A = 独立 ITT 聚合器(四臂表+知识形成时间线+材料门核算+三档自动判词,run1 部分 checkpoint 只读验证);B = capstone runner 预建(常数全从 CAP-1 冻结件读、**开封连锁**[无 S1-v2 双门判词解锁文件即拒读 D3]、合成数据 smoke + checkpoint/resume 演练)。**分支任务书预写(判词落地即按对应支发车,无需现场起草)**:①TREATMENT_EMPTY → 停,不跑反序;查 Episode→supply_tier 产卡链(冻结件预计产卡边界 vs 实际,断层定位表),呈裁;②处理组现但无优势(NO_EVOLUTION_SIGNAL 系)→ 记可信负结果(ITT 表全文入典),按 first-fault 字段查成本面或受益场转化断点,呈 sol 议机制层结论;③正序 r1 SIGNAL → r2 已在同书自动续;双跑 SIGNAL → 发反序 ×1(同课程逆排、同预算、判词 REVERSE_CONFIRMED/NOT);④反序确认 → 写解锁文件,capstone 按 CAP-1 + PREP-1 runner 一键发车(单次验收)。**间隙禁做清单(sol)**:改 methods/S1 runner/Scope/课程/阈值、开 Epilepsy2、重占 CPA 后端、按中途结果调课、清无关债——照办。监控:主线按 checkpoint/预算/后端健康轻量只读巡检。

### S1-v2 Part 0 停:COURSE_NOT_CONSTRUCTIBLE(受益侧空);主线两项裁定后重冻发车(2026-08-27 17:3x,主线)

**预检停报**(提交 2e7c527,0 LLM/0 fit):产例对成立(PowerCons-burst 5.00× + GunPoint-impulse 3.00×,19 叶交)但受益侧空——剩余 Scope 匹配 hampel 单元唯 GPMvF-burst 且系 HELDOUT_ONLY,"能跑通≠能证伪";执行者自查推翻首版错误冻结(曾选 HELDOUT_ONLY 作受益,将把 NO_TRANSFER 写死在池性质里),live 入口现读冻结件把关。对半余量重算表在案(对半一律抬升,与 M-1 同向)。**主线裁定 (a)**:释放 GPOvY(5.00×)+GPMvF-impulse(2.00×)作双受益单元——代价同类(实验者层已知转化倾向,臂内无泄漏),双受益将 M-1 余量分层预测内建进课程(强场应转化/弱场应边缘);前曝光注记强制入报告(新颖主张=课程内自产+ITT 端到端复利,非受益单元可转化性);**不释放双源单元**(防"带资进场")。**主线裁定 (b)**:regret 材料门从 max_u(1/n_slice) 改为**受益单元材料线之和**(max 被 GunPoint n=7 顶到 0.143 超一切真实 headroom;门本义=处理可作用处的可读改善);成本门不动。课程重冻:产例 A→identity A→产例 B→[边界产卡]→受益强(GPOvY)→受益弱(GPMvF-imp)→HELDOUT_ONLY→identity B(8 单元);立即重冻发车,正序 ×2。

### P0 收口(提交 98fc1fd):SUPPLY_TIER_PRODUCTION_REACHABLE;S1-v2 前置清零,正序发车(2026-08-27 17:2x,主线)

**P0**(0 LLM/0 fit,121 回归+13 新测全绿,h0 锁无需动):供给档产卡端落地于 evaluation 层(source_skill.py:294-566:2 个 distinct 未引导正例+五轴交非空 → 机械模板卡 supplies/no-exec/requires-support;1 例不产;引导计零;同族未解决 NEGATIVE 阻断[引旧规:225];两族同时合格不出卡留 Slow);**两档互不干扰双向证明**(同 2 正例下 TRY 档仍 does_not_survive_LOO、供给档正常出卡;TRY 载荷无 authority 块);**七环生产链真实双源全通**(含 PS-1 当年被打回的 EditController apply 环);模板顺序敏感缺陷被确定性断言当场抓修(编译产物=证据的函数非读序的函数,入教训);schema-代码四叶漂移续记账。**发车裁定**:sol 预授权条件全部满足(P0+判分冻结),S1-v2 正序 ×2 发车(opus)——课程冻结件作 Part 0(从 ps0b 对半余量机械推导:2 产例单元+≥1 受益单元+2 identity+1 HELDOUT_ONLY,排除 GPA/PowerCons-impulse[源]/GPOvY[已用]/GPMvF[M-1 用],处理组存在性算术预检不过即停不烧 LLM);A5-online 边界整合 = guard lifecycle + compile_supply_tier(P0 出口);ITT 主分析;材料门公式按冻结课程代入常数;TREATMENT_EMPTY 即停;正序×2 异注入 seed;判词封顶 S1V2_FORWARD_SIGNAL;预算总 LLM≤500(每跑≤250)/fit≤900/墙钟≤6h 硬。

### CAP-1 收口(提交 fd1ed90):终考协议冻结完毕;开封条件写死(2026-08-27 17:1x,主线)

TEST 子集改 seed=20260827 确定性随机 476 行(清单 sha256 7e1c4088…,总点 98,968 不变,mod-24 标 SUPERSEDED);终考协议冻结:三臂(Static/A3 冷/A5=S1-v2 同 K0+正序终态池)、TRAIN 对半按行号奇偶(Support/delayed 各 40,材料线 0.025)、ridge+accuracy+逐类 recall、菜单 blob 8de9545b/名单 sha 48e09ec4、每臂 LLM≤15/fit≤25/总墙钟≤90min、单次验收、判词 CAPSTONE_POSITIVE=A5−A3≥0.005 ∧ worst-class≥−0.005 ∧ harm 0;**开封缺一不可:S1-v2 正序×2 均 SIGNAL ∧ 反序×1 确认,届时自动开封无需再授权**。D3 全程零开封。0 LLM/0 fit。

### sol 发车裁定:S1-v2 方向批、原稿缓发;P0(供给档编译语义)+ CAP-1(协议冻结)并行发车(2026-08-27 17:0x,主线)

**M-1 证据等级定名**:development mechanism evidence(单单元;A3 对半亦 3/4 自然转化——反馈可读性惠及全系统非 A5 独占;delayed 恰 2.00× 系边界证据),支持进 S1-v2,非能力结论。**S1-v2 两处未对齐(sol 核码)**:(1) 设计稿"2 正例产供给卡"vs 现役 authorization_audit 仍 LOO≥2(等价 3 正例)——须正式分层:**2 独立未引导正例 → 仅 supplies_candidates(不执行不部署,受 Target 双门);3+LOO → 才可能更强 Shared TRY**(证据价格与权限对齐,非降门);(2) **Slow 编译器不会产 W-1 可消费的卡**——现路径产旧式 Active Skill(Frozen program 无 supplies_candidates=true/grants_execution=false 语义)——W-1 只接读端,产端缺失;照跑则 treatment empty 或测错对象(runner:1129/fast_agent:377)。**P0 发车(opus)**:在 Slow/source_skill 编译路径实现供给档语义(2 个 distinct task_episode_id、未引导、五轴 Scope 交非空 → 机械模板产卡:supplies_candidates=true/grants_execution=false/requires_target_support=true/Scope=交集/证据引用;1 正例不产;引导正例计零;3+LOO 旧档不动)+ **零成本生产路径可达性检查**(真实双源 Episode → Slow 编译 → 卡形 → T1/Scope/Fast 入池 → 双门,预算 ≤4 LLM);聚焦测试全套。**CAP-1 发车(grok)**:TEST 子集从 mod-24 系统抽样改**固定公开 seed 确定性随机 476 行**(防类别/批次排序结构偏差,标签仍封)+ 冻结完整终考协议(三臂/TRAIN 内切分/Consumer/metric/菜单/Scope/初始态/预算/seeds/repeats/停止条件/判词);CAP-0 记为密封+算力合格,**CAP-1 冻结后才许自动开封**。**S1-v2 判分收紧(入设计终稿)**:主分析 ITT(Scope 合格注入失败=A5 系统失败,条件转化率另报);材料门数值化;冻结具体课程名单/顺序/transfer graph/预计产卡边界与首分叉单元;**反序不替代重复——预冻正序 ×2(异 seed)+ 信号后反序 ×1**;效率措辞主用"适应/搜索效率",训练计算效率仅 fit 墙钟/time-to-threshold 支撑部分。P0 落地 → 冻结终稿 → 直接跑正序,**不再增资格轮**。

### CAP-0 收口(提交 010c0d1):SEAL_INTACT;capstone 靶冻结 Epilepsy2 子集(2026-08-27 16:4x,主线)

密封证据链五层全过(ROSTER 声明/磁盘字节与时间戳/git 唯一提交 10f9fee/下载脚本仅 namelist/全仓零数值 loader,80-11420-178 系下载前 metadata census 非 zip 读取);结构 MATCH。**子集冻结:k=24,TEST 行号 ≡0(mod 24) 取 476 行,总点 (80+476)×178=98,968 ≤ 100,000**;三臂同子集;S1-v2 过预注册门后按协议**自动开封**,无需再授权;开封前禁 oracle/fit/标签/数值。0 LLM/0 fit,D3 零数值零标签读取。

### M-1 收口(提交 55f1d1e):MARGIN_GATING_CONFIRMED——余量门控获受控因果证据;S1-v2 参数回填待 sol(2026-08-27 16:4x,主线)

**判词依据**(29/100 LLM,45/100 fit,1069s):算术先行(0 fit)——对半拼接 Support n=21 余量 4.00×、delayed n=19 余量 2.00×(四分基线 1.35×);唯一变量承诺守住(同卡/Scope/算子/Consumer/帽/底物/种子);**供给候选双门转化 2/4(四分基线 0/4),部署 held-out +0.1867,harm 0**。**深层读数**:A3 冷提案在可读面自然转化 3/4——门控主体是确认面余量,非供给通道特权;"反馈可读性是就绪学习前提"获实证。**注记**:注入漏(prepare→identity-only)仍在(2/4,PS-2/G3/M-1 三场同族),按 sol"不再修 wiring"裁定不追修,S1-v2 按 50-75% 注入可靠性如实设计;delayed 余量恰在 2.00× 杠上;对半读数仅作机制证据不与四分比能力。**S1-v2 设计稿参数已回填**(docs/S1V2_DESIGN_DRAFT_2026-08-27.md:对半协议全程、余量按对半重算、注入可靠性入设计、K0 不装双源卡、处理组存在性预检、TREATMENT_EMPTY 即停、训练效率=fit 墙钟+time-to-threshold),**待 sol 终审即发**。CAP-0(Epilepsy2 密封审计)在飞。

### 提速方案四点收紧 + capstone 改道 Epilepsy2(sol,2026-08-27 16:2x,主线采纳)

(1) **M-1 控制 S1-v2**:框架可先写,课程参数/分层规则/发车必须等 M-1 判词;否证则 S1-v2 不得照旧跑(16:12 时 M-1 仍 PROBE_ONLY,live 0/8,仅完成余量预检);(2) **S1-v2 正序先行**:正序确认新知识产生并影响后续单元 → 有信号才反序与重复;treatment empty 立即停;(3) **效率措辞**:仅记 LLM/probe/fit 数只可称"适应/搜索效率";兑现任务书"训练效率"须加 consumer fit 墙钟或 time-to-threshold(随 S1-v2 低成本记录,采);(4) **capstone 改道**:sol 核官方元数据——严格新下载规则(非本地+二分类+单变量+长度≥150+TRAIN+TEST 全样本×长度×通道≤10 万)实际仅剩 DodgerLoopGame 一件且系本地近亲,**凑不满池**;改为:机械审计 D3=Epilepsy2(80×178 TRAIN,TEST 11420 行)密封完整性 → 若 Outcome 从未打开,预注册**不读标签的确定性 TEST 子集规则**(行号规则,总点数≤10 万)冻结为 capstone Target,S1-v2 过预注册门后按冻结协议自动开封,无需再等授权;密封失效才启动新下载。**周六头条系顺利分支非承诺**;三门定位:M-1 决反馈余量假设 / S1-v2 决复利存在性 / capstone 决处女 Target 成立性。审计书发 99d4e625 谱系(不写 STAGE_REPORT,防与 M-1 撞);S1-v2 设计稿今日出待 sol。

### sol 裁定 A′ 采纳:Gate 3 定名"治理通过、效用复现失败";M-1 反馈余量因果小门发车;主线自纠(2026-08-27 16:0x,主线)

**Gate 3 官方判词**(sol 措辞):条件化通过、有害候选否决通过、正向跨单元复现未通过;不换数据集追第二正例、不宣称全过。**当前可主张**:历史经验已能被有 Scope 地送到 Target 并由当前数据安全否决;**尚不可主张**:稳定改善不同 Target/已成复利。**主线自纠**:方案 A(直接带分层跑 S1-v2)系又一次"门未验完跳大实验"——余量门控仅 2 个事后观察点,先小门实证。**M-1 设计(A′,发 grok)**:因果括号——固定卡/Scope/算子/Consumer/预算,**唯一变量 = held-in 反馈切片分配**(四分→对半,行数翻倍,Support→delayed 双门结构保留);主方向 = GPMvF-对半(G3 已有四分基线 0/4):A5-scoped×4 + A3×4;判词 MARGIN_GATING_CONFIRMED(供给候选转化 ≥2/4 经双门)/ MARGIN_GATING_REJECTED(余量 ≥2× 仍 0/4 → 机制重审按止损线)/ AMBIGUOUS(1/4,附 GPOvY-降档确认方向的追加案呈裁);**协议变体标签纪律:对半协议读数只作余量机制证据,不与四分基线作能力比较**。成立 → 余量分层写入 Gate 4;不成立 → 机制重审。后续序(sol):M-1 → S1-v2(K0-fixed/A5-online 同起点、唯 A5 跨单元写回、须观察到课程中新知识影响后续单元)→ fresh capstone → Stage 2 → Stage 3。

### Gate 3 收口(提交 19c6b22):FIELD1_NO_CONVERSION——治理两场过、迁移场未复现;"余量门控转化"假设立案(2026-08-27 15:5x,主线)

**核心正效果移动:上午是(+0.2127×4);下午复现场否(GPMvF 差 0.0000,A5 多 11 fit/3 探针)。** **场②条件化过**:16 叶 WHEN 对 45 密封单元仅命中 8、其中 6 个密封答案即 hampel(Scope 判别力实证);场外 0/4 注入。**场③数据主权过**:ToeSeg2(hampel held-out −0.023)4/4 注入、15 探针、0 获批、identity 零害——否决链最干净读数。**场①未复现**:GPMvF 注入 3/4、材料正 0/4;唯一 hampel 部署系 agent 自提(a3_2 冷启动也自提部署 hampel +0.1867);a5_1 探针 +0.1818 被 delayed −0.10 否决(链条工作正确,该单元 Support/delayed 自相分歧)。**机制解释(立案为假设,非定论)**:转化受确认面余量门控——GPOvY 4.15× 转 4/4,GPMvF 1.35× 转 0/4,与 ps0b 切片(0.18/0/0.20/0.22)一致;通道无故障(候选已达、已测),是 Target 反馈面读不出收益。**结构性事实**:本地池 Scope 内强余量未用单元已尽(仅剩 GunPoint 1.40× 同弱档),场① 本地重试预期同败,不再试。**Part 0 inspect 修落地且 live 兑现**(supply_without_agent_program=True ×2,W-1 缺口补上);执行者中途改治理优先顺序(红灯场先跑)如实记,批准。**裁决叉呈用户/sol**:(A) 接受"机制身份完整 + 转化余量门控"打包,Gate 4(S1-v2)带余量分层设计前进(预测:A5 优势集中于高余量单元——把场①结果转为设计输入);(B) 判 Gate 3 未过整体重议。主线荐 A。

### W-1 收口(提交 70b6d8d):SUPPLY_RUNG_PRODUCTION_CONFIRMED——分类线首个 A5 正效果入账;Gate 1+2 通过;Gate 3 发车(2026-08-27 13:2x,主线)

**核心正效果移动:是。** A5-scoped 4/4 全链走通(入池→Support 回执→材料正→delayed 批准→部署),**GPOvY held-out gain +0.2127 ×4**(超密封上界预估 +0.184);A5-neutral 2/4 入池 0/4 材料正 0/4 部署(安慰剂链正确);三臂 harm 全零;探索槽 4/4 保留;成本 74/150 LLM、48/160 fit。**接线内容**:修 1 = fast_agent.py:373-408/:1005-1052 注入与 propose 失败解耦(单测证实,live 未直接观测——残留耦合定位于 inspect 层上游,一行级修复候选已具名,折入 Gate 3 验证);**修 2 = online_loop.py:775/:817-836 供给 winner 补上 agent 自提 winner 本就有的 delayed 裁决出口——此为真正解锁**(PS-2 run9/12 实已走完 Support+delayed,被无裁决分支丢弃;**PS-2 漏斗记账订正入案**:Support 应按注入探针 Episode 归因,非 select 选中)。**机制注记**:转化全程未经 LLM 选择层(探针序+Support 预算完成)——选择盲区仍在但不再挡路。同权证明在案(补权非扩权);阈值/授权语义零改动;h0 content sha 不变;95 回归+11 新测全绿。语义纪律:只主张"经验机械供给候选、Target 反馈裁决并批准",不主张 agent 学会提 hampel;GPOvY 同族注记;pilot。**四行现状块**:当前目标 = 证明供给机制条件化(非复读)|最新判词 = SUPPLY_RUNG_PRODUCTION_CONFIRMED(+0.2127×4)|当前 first fault = inspect 层残留耦合(一行修候选)|下一门 = Gate 3 三单元小课程。**Gate 3 发车(opus)**:①Scope 匹配正向场(机械选自 PS-0b ROBUST 且非 GPOvY,A3+A5-scoped 各×4,读正收益与成本差)②Scope 不匹配场(pattern 轴不交单元,A5-scoped ×4,断言零注入——条件化直接证明)③identity/harm 否决场(Wine:hampel 合法但 Support≈0+类伤,A5-scoped ×4,断言获测但 Target 否决、identity 部署零害——数据主权直接证明);inspect 一行修折入并以注入可靠性读数验证(预期 ①③ 注入 4/4、② 0/4);判词 GATE3_PASSED 或按场失败标签;LLM ≤150/fit ≤120/墙钟 ≤3h。

### 止损线与推进方式改革(sol 周度审计,用户裁可,主线全采并自认三重漂移,2026-08-27 12:1x)

**周度诚实计账**:七天内核心主张(分类 A5>A3 via 积累)零推进;工程/诊断/可信度进展明显但不等价方法进度。**主线自认三失**:(1) 报告美化——把断点定位写成"论文级成果",致停滞被叙事掩盖(第三次同类漂移);(2) 时序错误——先建七单元重实验后验处理组存在,正确序为 1 单元→3 单元→7 单元;(3) 审计递归——r 系与 PS 系连环,主实验不断后移。**报告改革(立即生效)**:一切报告首行 = "核心正效果今日移动:是/否 + 数字";诊断发现一行带过,禁用"论文级/信息量大"类修辞于非正结果。**硬止损线(sol,逐字采)**:此后仅允许 ①W-1 生产接线(在飞,恰为允许项)②一次生产路径 PS-2 重跑(含于 W-1)③至多一次针对明确 first-fault 的修正;仍不能回收 GPOvY +0.184 → **全停,与用户和 sol 重审"经验以候选供应影响搜索"机制本身**,不再加 runner/权限层/PS-3。**通过后的门序**:Gate 3 = 三单元小课程(Scope 匹配正向场/Scope 不匹配惰性场/identity-harm 拒绝场,三场全对才证明机制非复读 Hampel)→ Gate 4 = S1-v2 四臂演化课程 → Gate 5 = Stage 2/fresh capstone/Stage 3。**暂停清单(全采)**:AD 新实验、新 Consumer、新算子、新数据下载、七单元课程、Stage 2/3、新权限平台、长篇论文整编、新诊断 runner——直到小型正向闭环成立。**文档**:W-1 落地后更新 AGENTS.md 状态锁 + 四行现状块(当前目标/最新判词/当前 first fault/下一门),不做长文档运动。

### PS-2 正式收口(提交 853f367):POOL_ENTRY_WITHOUT_CONVERSION;两断点定位;生产接线书发车(2026-08-27 12:0x,主线)

**判词**(12 跑协议落盘,134/150 LLM,checkpoint+resume 经受住 run7 两次 500):卡 4/4 在视野;注入仅 2/4 入池(**断点一:注入与 agent 提案活动耦合**——prepare 落 identity-only 池时 cand_skill_* 不发射,proposal_count=0/LLM 2-4 同现);入池轮**选择层零选中**(含过校验的 no-op,**断点二:选择盲区**),Support/delayed/部署全零;自提仍 burst/outlier_threshold 零 hampel(机械入池不改提案族,与 PS-1 一致);安慰剂零假阳;全程零害。GPOvY +0.18 headroom 未回收——**正效果第一级未兑现,断点已定位**。**生产接线书发车(opus,五件套齐)**:root cause = supplies_candidates 死旗+注入耦合+选择盲区;principle = bootstrap 4b 承诺+sol 权限梯"供一待验证候选"档+4f 探索槽保留;最小面 = methods/ttha Fast 候选组装与探针序;可证伪实验 = PS-2 协议在生产路径重跑(预期注入 4/4、供给候选获测、GPOvY 真 headroom 下转化、no-op 被测但 Support≈0 不获批);回退 = revert 单提交。**语义规格**:supplies_candidates 卡的冻结程序在 Scope 匹配时无条件物化为候选(去耦合);供给候选在既有 Support 预算内占探针位(agent 首选仍第一,探索槽 4f 保留;同帽不外加);grants_execution=false 与批准链原封;授权政策(谁得旗)不进 methods,留修订案。周冲刺表第 1 日照计划执行中;待用户:capstone 下载预授权、sol 包转呈。

### 主线定向指令(用户,2026-08-27 11:12):产出以"有效正效果"为唯一中心;负证据/边界降级为附录防御

用户重申项目目的(任务书原文重贴):最终产出是**方法设计 + 有效的正效果**(自适应提升下游模型性能与训练效率),**不是**"坚实的负证据/边界/条件"类结论。主线执行调整:(1) 报告与规划的重心从"断点钉得漂亮"回到"正效果推进了多少";(2) 实验排序以能产出正面演示者优先(PS-2 收口 → S1-v2 演化曲线 → capstone 处女靶正效果),边界测绘类工序除非阻塞正面路径否则不再排;(3) 论文架构:正效果为主章(预测线迁移正例 + 分类线经验加速就绪 + 训练效率读数),负证据/治理/边界全部降入消融与附录;(4) 已 bank 的正面资产清单:Frep A5>A3(成本 −31.7%/held-out 优/害 1v4)、GPA hampel +0.2690(可再生)、PowerCons +0.0833(可再生)、GPOvY +0.184 headroom(PS-2 目标回收物)、两任务全生命周期、全程零 harm 纪录。排障链(七层)重定性为:为正效果扫清路障的过程记录,其价值以"解锁了 PS-2/S1-v2 正面演示"计,不独立成章。

### PS-2 收口:POOL_ENTRY_WITHOUT_CONVERSION——机械入池 2/4,选择层未选中,无批准部署(2026-08-27 11:3x)

**判词 `POOL_ENTRY_WITHOUT_CONVERSION`(冻结表,协议记录)**:隧道恢复后续跑 12 跑落盘。A5-scoped 冻结 hampel **入池 2/4**(run9/12),run3/6 未入;入池的两跑 **select 均未选中** → Support/delayed/部署全零。A5-neutral 入池 2/4,0 虚假部署(非 PLACEBO_CONVERSION)。harm 0;探索槽保留(DRAFT 合并未删自提槽;入池轮均与自提共存)。**不得写成 agent 学会了 hampel**:自提族仍是 burst/outlier_threshold,相对 PS-1 无提案能力改善。机械档成功措辞仍只许"经验以机械通道供给候选、Target 反馈裁决"——本场未走到裁决。

**inject=False 归因(结构化,非 stdout)**:卡 **4/4 进 Fast 视野**(检索出卡)。未入池轮池内只有 identity、`proposal_count=0`、chosen 空、LLM 2–4、自提空——属 **卡在视野但 prepare 落到 identity-only 池**(提案早停/编译路径未发出 `cand_skill_*`),不是检索未出卡。入池轮则自提与注入共存。断层主位:**selection 不选** + 半数轮机械通道未发出候选。

**成本**:LLM 134/150(含 attempt1 67);fit 63/160;本次墙钟 3930s/7200s;下载 0;returned_model=`gpt-5.6-sol`。run7 曾 500,checkpoint 保住 1–6 后 resume。pilot;GPOVY 同族;引导正例计零(本场无部署)。

**提交**:协议工件 `ps2_mechanical_supply.json/.md`;runner 续跑/漏斗/miss 归因;本节(未删改既有 PS-2 BACKEND 段)。不提交他线文件与 checkpoint。

### PS-2 部分收口(提交 1f0a921):BACKEND_UNAVAILABLE(隧道死亡,11/12 非协议);实现已落地可续跑;整夜链条总结(2026-08-27 06:3x,主线)

**判词 BACKEND_UNAVAILABLE(冻结表,不升格)**:机械入池实现已落地(卡尾 Frozen program steps 走生产路径 _parse_frozen_steps→_skill_frozen_candidates→cand_skill_*,requires_target_support=true 入 DRAFT 合并、占 3 帽一槽不外加,grants_execution=false 全程同权无捷径;中性程序 resample_uniform 密封核为 numeric_no_op);attempt1 跑完 11/12 时中转 500 且 runner 终局落盘设计吞掉协议记录(**教训:长协议必须逐跑 checkpoint,已补 +--resume**);attempt2 隧道整段不可达(trycloudflare 发送即断),禁回退旧后端遵守。stdout 11 行仅作补记:scoped 已完成 3 跑 2 跑入池、从未 selected/部署;neutral 3/4 入池、0 虚假部署;自提仍 burst/outlier_threshold 零 hampel;run5/6 inject=False 疑检索未出卡——**均不作因果定案**。成本 67 LLM/31 fit。**待办**:隧道恢复(需用户侧重启)后 --resume 重开 12 跑,判词表不得改。**整夜断点链(总结,均有工件)**:S1c 处理组空 → diag 提案语义不足(11/15 未提出,K=5 不救) → PS-0 记录层不存 Context+PowerCons 一行假象(瓶颈拆双名:发现 vs 确认) → PS-0b 确认面审计(hampel 簇存活/burst 系拼池假象) → PS-0c 双源就绪 → PS-1 文本档惰性(卡在视野零影响零安慰剂效应) → PS-1b 根因 = bootstrap 4b 承诺注入从未接线 + 文本影响系修辞函数(C40 对照)→ **设计原则:知识影响必须走类型化机械通道** → PS-2 机械档实现落地待续跑。**永久资产**:round record 持久化(pattern+全提案)、确认面审计法与 ROBUST/FRAGILE/UNREADABLE 分级、源再生性标准、安慰剂对照双卡设计、checkpoint+resume、8 个提交。

### PS-1b 收口(提交 01dd11c):TEXT_RUNG_INERT 冻结;根因 = bootstrap 4b 承诺的 runtime 注入从未接线(supplies_candidates 死旗);PS-2 发车(2026-08-27 03:4x,主线)

**尸检定案**(0 LLM):非 WIRING_FAULT——scoped 卡完整进提案调用 system(与 C40 同槽 skills[3],8/8 轮在视野);位置不解释分裂。**双重根因**:(1) 措辞——C40 短 RISK+具体名+否定句即可带偏,PS-1 卡"prioritise exploring"+六条 hedge 抹平影响(A5-scoped 2/4 跑仍提竞品族);(2) **spec-vs-wiring 缺口**——bootstrap `build_contrastive_candidates` 4b 明文"Source prior 由 runtime 注入,勿自己复述提案",但 supplies_candidates 在 Fast 零读取、卡无冻结程序则 _skill_frozen_candidates 无物可供:**系统对 agent 承诺机械注入、叫它别抄先验,却从未兑现注入**——与 guard 管道同族的第 N 个"框架已表达、实现未接线"(五问 Q3 型)。**设计原则入典**:文本是不可治理的影响通道(无 hedge 强到成灾[C40]、有 hedge 惰性到零[PS-1],影响力系修辞函数非权限函数)→ **知识影响必须走类型化机械通道**。模型混淆评级低(两场 returned_model 同为 gpt-5.6-sol)。书外:supplies_candidates 死旗、C40 算子名被 risk_guards.sections 双写、schema-代码叶集漂移(16/20)。**PS-2 发车(grok 续)**:实现 4b 已承诺语义(评估层)——scoped 卡携冻结 hampel 程序机械入池(同帽不外加、与自提候选完全同权、Support/delayed 仍握批准),安慰剂携合法中性 no-op 程序(对照"任何入池"vs"正确候选"),GPOVY 三臂×4;判词 MECHANICAL_RUNG_CONFIRMED / POOL_ENTRY_WITHOUT_CONVERSION / **PLACEBO_CONVERSION(批准链失灵警报单列)**;语义纪律:机械档成功表述为"经验以机械通道供给候选、Target 反馈裁决",不得包装为提案能力改善。

### PS-1 收口(提交 5bf4d64):NO_PROPOSAL_SHIFT——卡进视野但零提案影响;C40 对照竖起"耦合层"新问题;PS-1b 尸检发车(2026-08-27 03:2x,主线)

**Part 1**:PowerCons 第二试挣到(hampel Support +0.0714/delayed +0.50/批准部署;首试仍 level-shift 未提 hampel)——双源就绪(GPA[agicto 时代]+PowerCons[新中转],跨后端注记在案)。**PS-1 判词 NO_PROPOSAL_SHIFT(pilot)**:12 跑三臂 hampel 提案率全 0/4;scoped/neutral 卡 4/4 进 Fast 视野;六段漏斗全零;harm 全零;三臂帽与预算相等断言过;**连 PLACEBO_EFFECT 都无——卡存在本身亦不改行为**。后端:新中转首用成功(returned_model=gpt-5.6-sol,首探 TLS 500 重试通),密钥零泄漏。Part 0 五轴交 19 叶;runner 级修复一处(observable_feature_v1.json schema 与 Python OBSERVABLE_FEATURES 叶集不一致,机器 AST 投影 16 schema 合法叶——schema-代码漂移记债)。**核心新问题(耦合层)**:C40 无权自由文本卡曾把 A5 提案两轮锁死 level-shift(文本引导力过强成灾),PS-1 结构化建议卡却完全惰性——"技能上下文→提案生成"的耦合在两场景表现相反,断点未定位(渲染位置/提案指令措辞/卡语言强度/布线)。**权限梯档位辨析(主线)**:本次卡按 suggestion-only 约束不携程序,supplies_candidates=true 实际无物可供——测的是"文本暗示"档;梯上"**供一待验证候选**"档(卡携具体程序机械入池、execution 关、Support/delayed 仍握批准)未测,系 sol 梯原文既有档位。**PS-1b 发车(grok,0 LLM)**:离线重建两场景渲染与提案指令、布线核查(卡文本是否进提案调用上下文)、语言强度对比、模型混淆评级;裁决 WIRING_FAULT(修线重跑 PS-1)/ TEXT_RUNG_INERT(升机械档 PS-2:scoped 卡携 hampel 程序,安慰剂携中性 no-op 程序)/ MIXED。

### PS-1b 收口:TEXT_RUNG_INERT——卡在提案上下文完整渲染,文本档天然弱耦合;PS-2 升机械入池档(2026-08-27 03:2x)

**渲染**:两卡同槽(skills[3],三 bootstrap 之后)。PS-1 scoped body 1810 字符,`hampel_filter` 在复原 system ~80%(散文+`program_geometry`);C40 RISK `repair_level_shift` 在 ~85% 且 `risk_guards.sections` 双写。observation 在 user,技能只在 system Resolved Harness。T1 只改 retrieval 惰性过滤,`_skill_prompt`/`_messages`/instruction/bootstrap 自 C40 未改布局;C40 live 在 T1 前故卡进 Fast。**指令**:`instruction.md` 明确使用 retrieved Harness content,无"仅 observation 提案";bootstrap 4b 写 runtime 注入先验、禁止复制。**布线**:prepare 单 view 进 propose;`retrieved_skill_ids` 八轮全含 scoped 卡;其中两轮编出 `outlier_threshold`——提案调用有卡仍不提 hampel。`supplies_candidates` 在 Fast 零读取(`ordering_card.py` 仅默认 False);无 `Frozen program steps` 故不进 `_skill_frozen_candidates`。**语言**:C40 短 RISK 点名+否定句;PS-1 TRY-HYPOTHESIS 六条 hedge。**裁决 `TEXT_RUNG_INERT`**(非 WIRING_FAULT/MIXED)。**PS-2**:scoped 携 hampel 冻结程序经 `supplies_candidates` 机械入池(execution 关);安慰剂改合法中性 no-op 程序;12 跑协议不变。**模型混淆低**(同 returned `gpt-5.6-sol`;中转路径不同已注)。0 LLM/0 fit/0 下载/零改代码。

**提交**:`artifacts/functional/e2/ps1b_coupling_autopsy.json/.md`;上条 PS-1 收口段(他书未提交,连同提交,未删改既有正文)。不提交他线 `AGENTS.md`/`README.md`/`PROJECT_STATE*`/`SUCCESSOR_BRIEF*`。

### PS-2 停在 BACKEND_UNAVAILABLE:评估层机械入池已接线,12 跑协议未完成(2026-08-27 06:2x)

**实现(评估层,未改 methods/runtime/contracts/operators)**:scoped/neutral 卡对在 body 末尾携带 `Frozen program steps:`,走既有 `_skill_frozen_candidates` → `cand_skill_<id>`;`requires_target_support=true` 故 DRAFT 合并 `(*agent[:1], *draft[:1])`,注入占 `maximum_candidates=3` 一槽、不外加、无执行捷径。安慰剂冻结 `resample_uniform{}`(密封 oracle:合法、verifier 过、numeric_no_op、held-in/held-out=0);scoped 冻结 `hampel_filter{}`(文献默认,不抄密封 oracle 的 window=3/n_sigmas=0.1)。compile-only:解析+EditController 应用+冻结步存活、token 比 1.127、中性散文零算子名。

**判词 `BACKEND_UNAVAILABLE`(冻结表)**:attempt 1 跑完并打印 11/12 后,`ps2_run12` inspect 遇 relay `InternalServerError`;runner 当时只在 12 跑返回后落盘,11 条结构化记录随栈一起丢。attempt 2 加了逐跑 checkpoint 与 unit 级重试,但新中转 probe 连续 `APIConnectionError`(trycloudflare 隧道,HTTP 发送即断),无旧中转回退。不得把 stdout 11 行升格为协议表,也不得把"有的跑 inject=True"写成 `MECHANICAL_RUNG_CONFIRMED`。

**stdout 补记(非协议)**:A5-scoped 已完成 3 跑中 2 跑打印 inject=True 且从未 selected/approved/deployed;A5-neutral 4 跑中 3 跑 inject=True、0 虚假部署;自提族仍是 burst/outlier_threshold(相对 PS-1 零 hampel 自提,未见提案能力改善)。run3/9 打印 support=True 且 selected=False——漏斗可能把同签名 Episode 误记到注入候选,未核结构化行前不作断层定案。run5/6 inject=False 且 agent=-、LLM 3/4,提示检索未出卡或提案早停,机械入池并非 4/4。成本:LLM 67/150,fit 31/160,墙钟 5236s(attempt 1)+后续 probe;下载 0。语义纪律仍成立:即便入池,成功表述也只能是"经验以机械通道供给候选、由 Target 反馈裁决"。

**提交**:`evaluation/functional/run_e2_ps2_mechanical_supply.py`;`artifacts/functional/e2/ps2_cards/`;`artifacts/functional/e2/ps2_mechanical_supply.json/.md`;本节(连同上条未提交的 PS-1b 收口四行,未删改既有正文)。不提交他线 `AGENTS.md`/`README.md`/`PROJECT_STATE*`/`SUCCESSOR_BRIEF*`。隧道恢复后可用 `--resume` 重开 12 跑,不得改判词表。

### PS-0b 收口(提交 346d30c):SECOND_SOURCE_AVAILABLE(hampel 簇存活);burst 簇 LEARNABLE 系拼池假象;PS-0c+PS-1 发车(2026-08-27 01:2x,主线)

**审计**(0 LLM/51 fit/9.6s,29 对单元×算子四分逐片):hampel 簇 ROBUST 6、独立家族 2(GunPointFamily+PowerCons),五轴 Pattern 交 11 叶可用(period_change_score 未入交);**GPA 4/4 余量 3.75×、PowerCons impulse 3/4 余量 2.44×、GPOVY 考场 4/4 余量 4.15×**。burst 簇死:Toe1/L2 仅 1/4 达线(**census LEARNABLE 的拼池假象——pooled vs sliced 的第三课**),ECGFiveDays 3/4 但最粗片 1 行、余量 0.57×。iqr/mad/level-shift/winsorize 零 ROBUST。**分层严谨**:单元级确认面不平反 S1c PowerCons Episode(一行假象维持取消);PS-1 须**新挣** PowerCons live Episode;禁从密封 oracle 编卡。协议变体(对半):ROBUST 7→10、iqr 新增双源、burst 仍死——报告级呈 sol,不采纳。**注记**:重挣尝试与 canonical cell 注入种子不同,切片面逐 run 有波动(srcB_2 读 0.0 与 canonical 3/4 不矛盾),PowerCons 重挣仍 take-what-comes。**PS-0c+PS-1 发车(grok,新中转首用)**:Part 1 重挣 PowerCons(≤2 试,新后端 probe 核身,与源 A'[agicto]跨后端注记——Episode 效度系 consumer 读数不依赖后端,提案采样行为差异由 PS-1 内部对照臂吸收)→ Part 2 挣到即按已提交 PS-1 冻结协议全程(Part 0 复验[新 round record 有持久化 pattern 叶]→ SkillEntry 双卡 → 12 跑 GPOVY → 判词表含 SHIFT_WEAK 无追加)→ 未挣到停报 PS1_SOURCES_NOT_REEARNED_FINAL(hampel 双源 episode 级死,切片协议问题升 sol)。预算 LLM≤180/fit≤160/墙钟≤2.5h。

### PS-0c+PS-1 收口:PowerCons 次试重挣成功;PS-1 判词 NO_PROPOSAL_SHIFT(2026-08-27 03:1x)

**后端**(新中转首用,无旧后端回退):probe 核身 host=`orbit-words-principle-alberta.trycloudflare.com`,请求模型 `cpa-gpt-5.6-sol`,returned_model=`gpt-5.6-sol`。首探 TLS EOF 500,同中转重试后通。源 A'(GPA/`ps0_srcA_1`)仍为 agicto 时代 Episode;效度是 consumer 读数。密钥未入任何工件/提交。

**Part 1**:`PowerCons__impulse_v2` A3-reset 同构重挣。`ps0_srcB_3` 失(族=`level_shift,outlier_threshold`,未提 hampel);`ps0_srcB_4` 挣到(族=`hampel,level_shift`,Support +0.0714, delayed +0.50, delayed 批准并部署)。Part 0 五轴交可用(19 叶超 task_kind;`period_change_score` 仍不入交:GPA=zero vs PowerCons=very_low)。

**Part 2**:GPOVY 三臂×4。卡进 Fast 视野(A5-neutral/A5-scoped 各 4/4 served)。hampel 提案率 **A3 0/4、A5-neutral 0/4、A5-scoped 0/4**。六段漏斗全零。无有效 Skill,harm 0。预算三臂全等(`maximum_candidates=3`,LLM cap 12,fit cap 10)。**判词 `NO_PROPOSAL_SHIFT`**(冻结表,无追加)。pilot;GPOVY 与 GPA 同族——机制隔离,非跨族 capability;引导下正例计零(本场亦无正例)。

**Runner 级修复(如实)**:首编卡被 EditController 拒(`observable_feature_v1.json` 不含 `level_region_*` / `outlier_region_end_fraction` / `level_only_post_shift_support_sufficient`,Python `OBSERVABLE_FEATURES` 有)。机器 AST 只投影 schema 合法 16 叶;body 与 scope_v1 仍带全交。methods/runtime/contracts/operators 未改。inert 卡审计扫到 `hampel_filter`——来自 `risk_guards.scope_v1.program_geometry`,body 无算子名。

**成本**:LLM 81/180;fit 26/160;墙钟 5729s/9000s;下载 0。

**提交**:`evaluation/functional/run_e2_ps0c_ps1.py`;`run_e2_ps1_arms.py`(最小 applicability 投影);`artifacts/functional/e2/ps0c_reearn_powercons.json/.md`+`ps0c_dual_source.json`;`ps1_proposal_shift_r2.json/.md`;`ps1_cards/`;上条 PS-0b 发车段(他书未提交,连同提交)。不提交他线改动的 `AGENTS.md`/`README.md`/`PROJECT_STATE*`/`SUCCESSOR_BRIEF*`。

### PS-0 收口(提交 bbd5fc5):GPA 复挣成功/PowerCons 源资格取消;瓶颈拆双名;PS-0b 确认面审计发车(2026-08-27 01:0x,主线)

**Part 1**:round record 落盘 fast_features_binned(叶级对拍过)+ 全提案账本;顺带揪出并修复族标注缺口(原靠扫自由文本 id,S1c 全部 probe 算子列表实为空)。**Part 2**:源 A'(GPA)首试复挣(Support+0.40/delayed+0.40/部署+0.2690,**可复现源**;胜路细节:agent 选了 level-shift 被 verifier 拒,Support 预算走到 probe_order 第二项 hampel——**ordering 而非 selection 在关键路径**);源 B'(PowerCons)两试皆失:srcB_1 未提出族(发现失败),srcB_2 提出并探得 Support 恰 0.0(**确认失败,hypothesis 卡不可治**);S1c 的 +0.0714 = 14 行切片 1 行,信号在分辨率地板——**PowerCons 判"一行假象",源资格取消**。**Part 3 未跑**:PS1_SOURCES_NOT_REEARNED,双源规则守住;PS-1 runner 以 sol 统一架构(SkillEntry 四权限字段/inert 中性卡/预算相等断言/SHIFT_WEAK 无追加)已提交待第二源。**教义候选(呈 sol)**:瓶颈双名制——发现失败 vs 确认失败,各配 hypothesis 卡 / 切片分辨率两种疗法;**源资格再生性原则**:授权源须确认面余量 ≥2× 分辨率地板(离线可验),或经复挣存活。**PS-0b 发车(grok,0 LLM,fit≤300)**:全池非 identity oracle 单元的确认面审计——按实际两轮切片逐片重算 oracle 算子读数,分类 ROBUST_LEARNABLE / FRAGILE(一行假象类)/ UNREADABLE;判各簇是否存活独立家族双源(hampel:GPA+?;burst:Toe1/L2/ECGFiveDays[+0.571 大余量]);附"对半 vs 四分"切片协议变体的报告级分析(不采纳,呈 sol);判词 SECOND_SOURCE_AVAILABLE(点名)/ NO_ROBUST_PAIR(切片协议问题呈 sol)。

### PS-0b 收口:SECOND_SOURCE_AVAILABLE;hampel = GPA + PowerCons 单元面;burst 凑不出双源(2026-08-27 01:1x)

**对象** 22 个非 identity oracle 单元 × 29 算子对(含并列)。cell 与密封 `slice_rows` 逐单元对齐;同一 consumer(ridge-raw-plus-difference-v1 / accuracy);fit 复用(每单元 1 次 identity + 每算子 1 次处理,共 51/300)。0 LLM。

**四分分级(冻结)**:ROBUST 7 / FRAGILE 11 / UNREADABLE 11。
- **hampel ROBUST 6**:GPA 4/4(+0.375,余量 3.75×) / GP 4/4(余量 1.40×,最粗 n=3) / GPMVF 3/4(1.35×) / GPOVY 4/4(4.15×) / PowerCons impulse 3/4(+0.143/+0.429/+0.214/0,余量 2.44×) / PowerCons burst 3/4(2.22×)。独立家族 **2**(GunPointFamily, PowerCons);五轴 Pattern 交可用(11 叶;`period_change_score` 未入交:GPA=zero vs PowerCons=very_low)。
- **burst**:Toe1 **FRAGILE 1/4**(唯 r2_delayed +1.0 on n=2);Lightning2 **FRAGILE 1/4**(唯 r2_support +0.50);ECGFiveDays **ROBUST 3/4** 但最粗片 **1 行**(材料线 1.0,余量 0.57×)——census +0.571 是 7 行池,不是高质量源。簇独立 ROBUST 家族 = 1,凑不出双源。
- 其余簇(iqr/mad/level-shift/winsorize)零 ROBUST。

**判词 `SECOND_SOURCE_AVAILABLE`**(单元 × oracle 算子确认面,不是给 S1c PowerCons Episode 平反)。PS-1 路径:源 A = GPA(已复挣 Episode);源 B = PowerCons **单元**(oracle 默认 hampel 3/4);考场仍 **GPOVY**(原场,现 4/4 ROBUST,与 GPA 同族,不得作跨族 capability 主张)。S1c PowerCons Episode 仍取消;若 PS-1 坚持双 live Episode,须对 PowerCons **新挣** oracle-default/稳定 hampel,禁从密封 oracle 编卡。

**协议变体(不采纳)**:对半后 ROBUST 7→10;hampel 双源仍在;iqr 新够双源(Distal burst + GPMVF burst);burst 仍不够(Toe1/L2 仍 FRAGILE)。改切片协议救不了 burst +0.571。

**提交**:`evaluation/functional/run_e2_ps0b_confirmation_surface_audit.py`(新诊断 runner;methods/runtime/contracts/operators/既有 runner 零改);`artifacts/functional/e2/ps0b_confirmation_surface_audit.json/.md`;本节。密封 oracle 只读;产物隔离于臂视野。

### 后端切换指令(用户,2026-08-27 00:46)

自下一本书起,live agent 后端切换为用户提供的新中转(base_url = trycloudflare orbit-words-principle-alberta,model = cpa-gpt-5.6-sol);密钥存本地未跟踪文件 `_scratch/agent_backend.ps1`(已核 .gitignore 覆盖,禁入任何提交物/工件/文档),执行者 dot-source 后运行。**在飞 PS 链豁免**:按 sol 后端锁定规则于原后端(gpt-5.6-sol@agicto)跑完,保持与 S1c 基线可比;若其 live 段遭遇原后端连接失败,按 BACKEND_UNAVAILABLE 停报后由主线以新配置续跑。此后跨后端的历史对比须注明后端变更;新跑工件照例记录 probe 与 returned_model。

### 常备纪律:principle-driven 五问协议(sol)+ 主线锚定与模型路由强化(用户,2026-08-27 00:4x)

**五问协议(一切方法改动前必答)**:①现象与证据等级?②因果链最早断在哪层(Observation/proposal/selection/verification/feedback/Skill compilation/Scope/authority/implementation wiring)?③现有框架原则能否表达正确行为(仅实现未接通或参数/表示不当)?④能表达→优先修现有实现;唯现有抽象确实无法表达才许新增机制;⑤新机制须统一解释一类问题并直接解锁核心实验,禁单数据集/单次失败专用补丁。**每次方法改动附五件套**:root cause/与既有 principle 关系/最小修改面/可证伪实验/回退方案。**Runner、Gate、Schema、文档不算方法进展**。总框架(Task/Consumer/Pattern Context→Workflow proposal→downstream feedback→Episode→Scoped Skill/Memory→Target 校准→freeze/deploy)判定基本合理;候选发现问题优先补齐"低权限 Skill 影响 proposal"的既有语义路径,不建平行系统。**用户加持二则**:一切推进须锚定项目目的/主线不偏移;模型路由再强化——**默认 grok-4.6,唯真困难任务用 opus**。主线自查(今晚账):PS-0 记录修复 = 五问③"实现未接通"型✓;PS-1 = 检验既有权限格✓;被撤回的 Hypothesis 新类 = ④违例被 sol 拦下(引以为戒);在飞 opus 书(PS 链)属实现+实验复合,保留;此后简单书一律 grok。

### S1-diag 收口(提交 94cf58e):proposal_semantics_insufficient;行为漏斗定量;K 消融否证截断说(2026-08-27 00:5x,主线)

**漏斗**(15 个含正解臂-单元机会,仅 S1c 层,历史 CLS-OP 分层单列未混):**11 未提出**(提案召回)/ 3 执行未过(MiddlePhalanx 三臂齐执 repair_level_shift、delayed 全拒)/ 1 走通(A3-PowerCons-hampel);措辞按"这一次运行"纪律。**K 消融**(12 LLM,只提案不执行,oracle 判 prompt 外):K=5 仍无一单元提出正解,且**候选数停 0-1 连帽不满**(GPOvY 双 K 空提案;PowerCons 双 K 皆 level-shift)——截断解释被否证,"原 K 提过未执行"亦无,冻结判 **proposal_semantics_insufficient**(不再拆 observation/策略偏置,留待未来 observation 扩展消融)。**附带定性**:冷 proposer 重度 level-shift 偏好(与 C40 A5 同款);Slow 空携带 = 无可编译证据非预算饿死;**仪器缺口确认:工件未存原始提案,最早可得层为编译后 pool**(PS-0 Part 1 的全提案落盘正中此缺)。**对 PS-1 的含义**:瓶颈在提案语义 → supplies_candidates=true 恰是把正确族直接放进 pool 的机制,若有效应当显著;分级 hypothesis 仍按"待验假设"表述。探针隔离存放。全菜单排序确认未跑。

### sol 统一裁定:不建 Hypothesis 新类,PS-1 = 现有 SkillEntry 的 proposal-only 权限格实验;七项待定拍板;在飞书已中断修正(2026-08-27 00:4x,主线)

**统一 Principle(入典)**:唯一知识对象 Episode→Skill;Scope 定可见处,Authority 定影响步;权限梯 = 只存储→重排候选→供一待验证候选→执行冻结 Workflow——同一 SkillEntry 的权限差,非四种新 Skill。`ordering_card.py:214` 既有四权限字段,**PS-1 即检验空格 supplies_candidates=true ∧ grants_execution=false**;禁增 PreparationHypothesis/第四 Active Skill/新权限平台(主线原提案系过度工程,撤回)。**在飞修正(已中断送达)**:A5-neutral 改全权限关闭、零算子名的 inert SkillEntry(原 period no-op 设计非真中性,作废);三臂候选帽与 LLM 预算严格相等(supplies 候选计入同帽);本轮定级 **pilot**,不得冻结生产设计。**七项拍板(sol)**:(1) 灰区判 SHIFT_WEAK,取消自动追加,充分重复留后续一次预冻实验;(2) 引导下正例可成 Target-local,对 Source 跨域扩权**计零**(不折权,防新任意参数);(3) 独立 family≠Scope 相容,五轴交塌缩至 task_kind 级即 SCOPE_INTERSECTION_TOO_WIDE 停编;(4) 成本材料线不拍百分比,报"到首个有效 Skill 的 LLM/probe/fit/轮次",非劣下少一次完整 probe 即清楚改善;(5) 害证家族归并暂缓(安全辅助非正向承重);(6) HELDOUT_ONLY 切片诊断排 PS-1 后;(7) Stage 2 扩写与论文主张定高,待 PS-1 正证据。**最小工程清单**:机器可执行 Scope + 现有 Skill 的 proposal-only 使用方式,仅此两项。**文档债**:AGENTS.md:198 状态锁停在 C32/#44a 严重滞后,PS-1 收口后仅更新"当前状态锁与下一门",不做大同步。

### PS-1 停于 Part 0(SOURCE_PROVENANCE_INSUFFICIENT,提交 3eda222);持久化缺口定性;PS-0 发车(2026-08-27 00:3x,主线)

**门判**(0 LLM/0 fit/<1s):双源 1-4 项全过(A:GPA hampel,DELAYED 级/LOCAL_ACTIVE/unguided[仅 bootstrap 三卡]/Support+0.50/delayed+0.40;B:PowerCons hampel,A3-reset 冷启动 by construction/Support+0.0714/delayed+0.50;家族独立);**轴 5 失败**:两执行记录均未持久化 OBSERVABLE_FEATURES 的 20 个非 task_kind 叶任何一个——witness/observer 统计不在 observable 契约内,不能作 applicability 叶。**系统性发现**:Fast 每轮都算 `extract_public_features` 产出的 binned pattern view(fast_features)传给 run_online_round,**分类线无任何 runner 落盘**;仅存两处(r3 census、s1_oracle 密封件)均在隔离横幅下,取之即"披 Scope 外衣的 oracle 泄漏"。执行者按"存量字段求交,禁事后重构"停手正确;Parts 1-3 协议已预冻入工件(未见任何 outcome);未建不可运转机械。**两条规划注记**:源 A 早于本 runner 线,工件无字段槽,须重挣;源 B 的 cell 记录躺在冻结课程条目内含 oracle 选课元数据,卡编译器不得读 course[]——合法 Context 只能来自 round record(恰是不存 pattern 的那个)。**PS-0 发车(续 opus 会话)**:Part 1 记录层修复(round record 落盘 binned pattern view + **全部原始提案清单**[兼修行为漏斗顶层缺口],零行为改动);Part 2 双源重挣(GPA/PowerCons 各 ≤2 次预注册 A3-reset 单 cell 跑,take-what-comes,漏采如实报——漏采本身是发现瓶颈数据);Part 3 双源齐则自动续跑 PS-1 冻结协议(Part 0 复验→编卡→12 跑)。**灰区规则预钉(A 层项 1)**:hampel 提案率分离落灰区(如 scoped 2/4 vs 对照 ≤1/4)→ 自动追加一批预冻 4 重复;仍灰判 SHIFT_WEAK 停报。预算:LLM≤220/fit≤200/墙钟≤3h。不写 STAGE_REPORT(diag 在飞)。

### 用户全权授权;PS-1(proposal-shift 因果实验)主线自主定稿并发车(2026-08-27 00:1x,主线)

**授权**:用户明示"大胆推进,以确实推进项目、解决难题、取得有效效果为准"。主线执行原则:砍批复往返,保效度纪律。**PS-1 设计定稿(sol 骨架 + 主线落地参数)**:目的 = 机制隔离验证"有 Scope 的结构化 hypothesis 能否改善候选搜索"(先于任何完整课程)。**双源**(待书内 Part 0 provenance 硬门):GPA hampel Episode(C40 r2 工件,A3 r1,Support+0.50/delayed+0.40)+ PowerCons hampel Episode(S1c 工件 unit3 A3,unguided by construction,Support 1/1,delayed 批准,+0.0833)——独立家族,S1c 自产解决了 GPA↔GP 同族问题。**考场 = GPOvY**(hampel-oracle 且 LEARNABLE[+0.415 held-in/+0.184 headroom],S1c 实测三臂全漏采[A3 一探即弃]——冷基线的漏采已有实录;系 GPA 同族,机制隔离实验可接受,报告须注明不得作跨族 capability 主张)。**三臂 × N=4 预冻重复**:A3(无卡)/ A5-neutral(**同结构同长度安慰剂卡**,指向合法无关无害族[period 完整性类 no-op],控"有卡即改行为"混淆)/ A5-scoped(双源按冻结 Scope 规则 v1 机械编译的 hypothesis 卡:五轴 Scope+建议探索 hampel 族+证据出处方向不确定性+强制"须 held-in 验证,失败回 identity")。**prior 槽实现为 runner 层实验机制**(生产 T1/授权规则零改动;卡经专用槽入 Fast prompt,评估层,工件明示 experimental slot;生产化须另走修订案)。**判词表(预注册)**:SOURCE_PROVENANCE_INSUFFICIENT(Part 0 停)/ PROPOSAL_SHIFT_CONFIRMED(scoped 的 hampel 族提案率材料级高于双对照 ∧ ≥1 次转化为批准并部署的本地 Skill ∧ harm 不升 ∧ neutral≈A3)/ SHIFT_WITHOUT_CONVERSION / NO_PROPOSAL_SHIFT / **PLACEBO_EFFECT**(neutral 偏离 A3 即卡存在效应,单列)。预算:12 跑(3 臂×4),LLM≤150/fit≤120/墙钟≤2.5h。evidence_grade=development-mechanism。执行:续 S1b opus 会话;**本书不写 STAGE_REPORT**(与在飞 diag 防撞,主线代记)。

### sol 双裁采纳:瓶颈定名"候选发现";S1-diag 中断改版;proposal-shift 因果实验立为下一主考;主线两处措辞受纠(2026-08-26 23:5x,主线)

**核心洞见入典(sol)**:当前 Harness 瓶颈在**发现候选**,不在验证/晋升/部署——链路"随机提中→才有 Episode→重复提中→才可编译→才入下一单元"把知识形成绑在低命中首环;系统现状 = 会审核经验,不擅用经验提高下次搜索命中率。**方法句**:Experience 不应只在候选被发现后负责审批记账,还应在有 Scope、有权限约束下改善下一次候选搜索。**主线两处措辞受纠(自认)**:(1) 1/15 只能表述为"当前课程/Prompt/预算/单次运行下 15 机会命中 1 次"——冷策略低召回,非"数据就绪固有难度"、非稳定概率;(2) "分级 hypothesis 解决问题"是合理假设非 S1c 已证事实,须走完整因果链(合法独立 Episode→机器 Scope 匹配→专用 prior 槽→提案分布实测改变→Target 自批→成本/regret 改善且 harm 不升)。**S1-diag 中断改版(已送达)**:Part B 全菜单排序作废(菜单入 prompt 即提示,截断/提醒不可分);改候选帽配对消融(同 observation/prompt,仅 K→5,提案不执行,oracle 判定 prompt 外);Part A 升级行为漏斗(菜单可用→原始提案→选择→verifier→执行→Support→delayed→部署,按 runner/Prompt/Context/arm 分层,先核验工件存否全部原始提案);封顶无 diag-r2/r3。**guard 主考暂停**(适合安全辅助,不承载正向进化主张;exact-name 太脆,家族归并不得本轮落地——MAD/IQR/winsorize 行为不等价,只能作有界政策候选经固定家族本体+课程回归门评估)。**下一主考 = proposal-shift 因果实验**(机制隔离先于完整课程):同 Target 同预算三臂 A3(无历史)/ **A5-neutral(等长无关结构化卡,安慰剂)**/ A5-scoped(合法匹配 hypothesis),查:卡被正确检索/提案分布实测改变/正确族出现率升/不挤占探索/Support-delayed-部署-成本改善;**须预冻结多 seed 或重复提案**(单跑不作数)。**前置**:Scope 机器可执行 + Source 正例须真实独立合法 Episode(oracle 普查不得直接变 Memory;GPA↔GunPoint 同族不得冒充两独立源)。**新事实(主线补)**:S1c 自产了第二个独立家族合法 hampel 正例——PowerCons A3(unguided by construction、Support 1/1、delayed 批准、部署 +0.0833),与 GPA(C40)构成候选双源,**独立性问题可能已被今晚数据自解**,待 provenance/Scope 审计核验。**管线**:diag(在飞)→ GPA+PowerCons 双源 provenance/独立性/Scope 审计(0 LLM)→ 明日 proposal-shift 实验设计与实现(hypothesis 卡编译+prior 槽+安慰剂卡+预冻 seed,设计呈 sol 后跑)。

### S1c 收口(提交 258c28d):主线判词 TREATMENT_NOT_INSTANTIATED;冷探索漏采率首次量化;主线并轨分级授权(2026-08-26 23:4x,主线)

**读数**(199/400 LLM、105/900 fit、58min、harm 全 0、错误晋升 0、无 delayed 拒批部署——部署修复站住):A3 仅单元 3 批准 hampel(+0.0833/worst-class +0.0444),其余三臂全课 identity;A5 六次 Slow 边界共 6 LLM 未写出可携带卡;环 1 = 同名害证未在两害证单元同采(A5 在 PowerCons **0 probe**);环 2 = 无物可测。**判词**:runner 按预注册算式出 NEGATIVE_TRANSFER,主线按 23:3x 裁定覆盖为 **TREATMENT_NOT_INSTANTIATED**(A5 全程无跨单元知识可用,差值系采样方差;执行者书外发现 1 同判:first-fault 是提案采样非 carry 污染)。**新量化发现(比判词值钱)**:5 个含菜单正解的单元 × 3 适应臂 = 15 次机会,live proposer 仅命中 1 次(A3-PowerCons-hampel)——**冷探索对菜单 headroom 的命中率 ≈ 1/15**;GPOVY(+0.1841 headroom)A3 一探即弃。这把项目论题反向坐实:无积累知识的 agent 系统性错过可得价值,问题只在如何合法供给知识。**统一诊断(三通道同根)**:TRY 饿于家族独立供给、guard 饿于同名采样巧合、可学正解饿于提案熵——**知识形成速率被小预算下的提案采样熵限死**。**主线并轨**:(1) 演化主考正式移至**分级授权轨**——其 hypothesis 卡由**历史已入库证据**编译(GPA +0.375/GunPoint +0.467 的 Support 正例已在账,零新采样依赖),卡入 Fast 后定向集中采样(TRY 建议本身就是采样导向器),课程回归门测其效果;待 sol 批的修订案即刻起草;(2) 害证家族级归并计数 = 第一份 Stage-3 式带证据规则提案(今晚运行数据为证据);(3) 冷命中率 1/15 入论文(动机章:经验缺席的代价)。S1 严格门基线数据齐备(四臂 × 7 单元 × live),作分级门对照的对照组存档。今日不再发车。

### S1c 运行中裁定(sol 实时判读,主线采纳):处理组未实例化;判词改 TREATMENT_NOT_INSTANTIATED(2026-08-26 23:3x,主线)

**实况**(运行健康:后端探测过、预算内、已至单元 7、checkpoint 正常):A5-online 六单元 `carry skills=[]`。负例分布 = MiddlePhalanx:repair_level_shift NEGATIVE(设计的可读害算子 iqr **未被采样**);FreezerRegularTrain(identity 单元):outlier_iqr NEGATIVE;其余 NEUTRAL/CONFLICT/无合法 probe——**两负例不同名,exact-name×2-unit 的 guard 门永不满足**,risk_authorized_operators=[]、Fast 视图零 guard。**裁定**:预注册未测环之一(同名采样)实测答案 = 否;处理组未实例化,四臂数值差不得归因经验积累;单元 3(A3 +0.0833 vs A5 identity)非负迁移(A5 无任何跨单元知识可用);**最终判词 = TREATMENT_NOT_INSTANTIATED(runner 若输出 NEGATIVE_TRANSFER/NO_TRANSFER 不予采纳,主线判词覆盖)**。执行五条(sol):单元 7 自然跑完不杀;原始结果保存但 A5/A3 数值不作迁移解释;**不跑反序、不重掷同一课程**;报告必列每单元 A5 NEGATIVE 算子/同名跨单元重复有无/guard 成型与入 Fast 与后续行为变化;无 guard 即以"随机提案未产生合法处理组"收口。**机制教训(升级仪器纪律)**:S1a 可达性审计只验了**代码可达**,未建模**证据生成过程(agent 采样分布)**——guard 通道的实例化概率依赖"LLM 恰好两次踩同一算子",此依赖过强,**exact-name 计数 + 自由采样的 guard 实验不适合作主线承重**;后续资格审计必须附"采样感知的处理组实例化概率"估计(本轮 28 臂-单元的 live 提案分布恰是第一份实证素材)。**主线走向**:演化主线权重移向分级授权轨(2 可学正例 → hypothesis 卡 → 课程回归门——其处理组由正常探索+材料级 Support 构成,**不依赖害证采样巧合**,待 sol 批的修订案价值上升);家族级害证归并(mad/iqr/winsorize 并族计数)可作 Stage-3 式带证据规则提案(本轮即其证据),不临时改。

### 部署规则修复落地;S1c 正序全程跑发车(2026-08-26 22:2x,主线)

**修复**(提交 987a13c,0 LLM/19 fit/33s):共享函数 `_incumbent_after_delayed`——winner 仅在 approved_skill_id 置位时上账;拒批不清既往已批 incumbent(r1 批 hampel、r2 拒 winsorize → 站 hampel);无既往则 identity。两轮体同源单定义;三只读审计字段;8 新测 + 12 guard 测全绿;r3 smoke 如预测:三臂 identity、harm 零、regret +0.0103,新增站岗门 no_delayed_rejected_winner_deployed。受影响入口清单在案(run Part C/conf_run/r2_replay/S1 runner)。**记录修正**:r2 的"负 regret=卖类博弈"实例系本 bug 症状(被拒程序上了部署账本),一般原则(regret 配 harm 读)保留。**S1c 发车**(续 grok 会话):冻结课程 r2 正序 7 单元 × 四臂,live 后端(先 probe 核身),帽 LLM≤400/fit≤900/墙钟≤3h 硬;判词封顶 S1_DEVELOPMENT_EVOLUTION_SIGNAL,判读铁律 regret 配 harm/worst-class 非劣;重点观测两个未测环:同名算子害证是否在两害证单元均被采样、guard 入视图后 proposer 是否避开;工件 s1_course_forward_run1 隔离。

### S1c 收口:NEGATIVE_TRANSFER;两未测环均未点亮;单顺序单跑 development(2026-08-26 23:2x,执行方)

**判定**:**NEGATIVE_TRANSFER**(development;单顺序单跑;封顶仍是 S1_DEVELOPMENT_EVOLUTION_SIGNAL,本跑未达)。解释器 `D:\Anaconda_envs\envs\project\python.exe`。后端核身 live:`gpt-5.6-sol` @ `https://api.agicto.cn/v1`,returned_model=`gpt-5.6-sol`;probe 不计入课程帽。课程/预算零改:冻结 r2 正序 7 单元,帽 LLM 199/400、fit 105/900、墙钟 3509s/10800s。

**四臂累计**(regret 必须配 harm/worst-class 读,禁单引):Static / K0-fixed / A5-online 三臂全课程 identity,cum held-out +0.0000,cum regret +0.3859,harm 0。**A3-reset 唯一分离**:单元 3 PowerCons 批准 `hampel_filter`(FROZEN_ACTIVE_SKILL_RECALL,held-out +0.0833,regret +0.0500 vs oracle +0.1333,worst-class +0.0444,harm False),cum held-out +0.0833,cum regret +0.3026。A5 在该单元 0 probe / 1 fit(只部署 identity),故质量劣于冷启动,判 NEGATIVE_TRANSFER。harm 四臂全 0、worst-class 非负。

**两未测环**:①同名算子害证未在两害证单元同时落库——A3/A5 只在 MiddlePhalanx 写到 `repair_level_shift` NEGATIVE,K0 只在 PowerCons 写到 `outlier_iqr`+`repair_level_shift` NEGATIVE,无交集;算术上可成型的 `outlier_iqr` 从未被 A5 采样为害证。②guard 从未进入 A5 Fast 视图(位置 3 后无成型),proposer 尊重问题无物可测。winsorize Support+/delayed− 本跑 0 条(live 未提案 winsorize;与 scripted smoke 不同)。A5 六次单元间 Slow 各 1 LLM,carry 技能始终空。

**提交**:runner `--run-course --order forward`;`s1_course_forward_run1.json/.md`;本节(含他书未提交的 S1c 发车条,未删改既有正文)。**义务**:课程/预算未动;后端身份 gpt-5.6-sol@agicto,未换后端;两未测环读数见上;`methods/`/`runtime/`/`contracts/`/`operators/` 与共享 runner 零改;oracle 隔离墙自检三面 blocked、判分只读 7 课键;密封件未覆写;未跑全仓 pytest;零下载。

### S1b-r2 收口:仪器解锁 + smoke 抓获两发现;部署规则 canon-vs-code bug 拦停 S1c;修复书发车(2026-08-26 22:0x,主线)

**r2 交付**(提交 4cca785,0 真 LLM/22 fit/34s):新 7 单元课程全底物互异、最小切片 7-45 行、零空切片;降档轨迹全记录(可学组走完 5→4→3 严格阶梯后按声明回到地板 5 允许同族重复,DistalPhalanx burst 具名入选);r1 课程存档未覆写。smoke 八门全绿且拿到**活证**:三适应臂各得 winsorize 的 NEGATIVE Support 回执(slice +0.1333/delayed −0.1111),feedback_surface_evidence_mode=live,S1c 仪器阻塞解除。**规则解释确认(主线)**:必要条件 |读数|≥1/切片只约束害证/可学组;identity/HELDOUT_ONLY 组定义读数为零,地板即可读性要求——解释正确,反事实已记。**发现一(regret 可被卖类游戏)**:三适应臂全冻结 winsorize,裸 held-out +0.1993 越过 menu oracle(+0.0103,类伤约束下),regret −0.1890 与 harm 事件同现——预注册"regret 必须与 harm/worst-class 非劣配对"实战立功;**S1c 判读禁单引 regret,站规**。**发现二(部署规则 canon-vs-code bug,拦停 S1c)**:handle_feedback_delayed 正确拒批(approved_skill_id=none),但 Support 立的 ledger incumbent 未被清除,_frozen_recall 照常部署被拒程序——三级反馈"delayed 才批准"教义在部署环失守;共享 runner 轮体规则,r2 记录未修(正确纪律)。**主线裁定:S1c 前必修**(canon-vs-code,B/T1 同类;污染 harm/regret 读数且系统性利好激进臂);修复 = delayed 拒批时清除/不采信 incumbent,部署回退 identity,配聚焦测试+单元 1 smoke 复跑(期望三臂改部署 identity、harm 事件归零、regret 回正);续 opus 会话执行,sol 异步复核。**发现三**:iqr 在两害证单元均合法可读有害(−0.0944@1/45、−0.1852@1/12),guard 前向位置 3 后原则可成型,实际取决于提案采样。

### S1b 交付但 S1c 拦停:选课规则"最小点数"反向选择反馈面(主线规则错);S1b-r2 修正发车(2026-08-26 21:4x,主线)

**S1b 交付**(提交 dbd840a,0 真 LLM/13 fit/11s smoke):四臂 runner + 机械冻结 7 单元双序 + 单元 1 smoke 七门全过(S1B_SMOKE_WIRED);17 项状态隔离断言过,K0 惰性卡在两臂 store 中且 Fast 三面零出现(T1 闸在活料上工作);oracle 隔离墙自检暴露并修复 Path.read_text 旁路(初版只包 builtins/io.open——**隔离审计自身也要正向自测**,入教训);合成条目验证域绑定双墙(异域 Target-local 丢弃/五轴 Scope 拒空交集与错 Consumer)。**S1c 拦停原因(主线选课规则错误)**:"各组内总点数最小"反向选择反馈面——6/7 单元最小 held-in 切片 ≤2 行、3 单元 r2_delayed 为空,probe 增益量化为 0.0、Episode 全 NEUTRAL,**guard 通道物理不可点火**;执行者按声明规则忠实执行并停给主线,纪律正确。**连带仪器缺口**:r2/r3 可学性标签基于拼接全池,臂协议实读四分切片,标签系统性偏乐观——"协议常数泄漏底物假设"第四例。**S1b-r2 修正规则(主线声明,续 opus 会话执行)**:(1) 全课程单元准入加**切片可读性地板**:两轮协议下最小切片行数 ≥5,且 |pooled 增益/害| ≥ 1/最小切片行数(必要条件筛,零新 fit);地板 5 不满足组配额时按 5→4→3 阶梯降,每步记录,禁止静默重选;(2) 害证/可学组的组内择优从"点数最小"改为"最小切片行数最大",identity 组同受地板;(3) 家族相异改为**全课程跨组去重**(7 底物尽量互异,池不足时同族需报);(4) 重冻结双序、重跑单元 1 smoke(期望见到至少一条非 NEUTRAL 回执的可读面证据)。S1c 待 r2 冻结课程复核后发。

### 用户-sol 大讨论裁定:"3"门考古定性;分级授权+课程回归门立为待批修订案;两轨一门合流方案(2026-08-26 20:4x,主线)

**"3"门考古(sol,采纳)**:3 = GENERAL_EVIDENCE_MIN_DISTINCT_TASKS=2(0818 设计,g1.py:99,a1d879a9)× LOO 稳定门(0819 预测线 Source Skill 自我确认事故后加,source_skill.py:217,a2fb69ae)的叠加隐含值,n−1≥2⇒n≥3;正典只写"多 Domain 重复证据"未写死 3;系预测线治理事故的遗产被分类线继承,**非统计校准值,可议但不可为便利而降**。主线补层次辨析:LOO 防自证,UNGUIDED 已管 provenance;对"不能执行、不挤探索槽、须过新 Target held-in"的软先验,3 的边际安全收益小、覆盖代价实测为零覆盖(r2/r3)。**Self-Harness 调研裁定(grok 报告+sol 四修正,采纳)**:其晋升货币 = 有界单钩子补丁+43/21 全量回归不退步门,非正例计数;其 heldout 实为 validation(参与每轮采择,多轮可间接过拟合)、门为 split 宏平均(非逐题非劣)、paper/README 数字不一致须分别引用、"同模自改"系实验配置非框架强制。可借:单 Surface 有界改/LLM 只提案确定性门批准/候选版本化可拒可滚/完整回归后晋升;不可搬:知识压成全局源码/宏平均替代 per-unit harm+worst-class/反复用 heldout 称终验/去 Scope。**修订案立项(待 sol+用户正式批准,未批不动工)**:分级授权表(1 scoped 正例=Target-local;2 独立 Scope 相容正例=结构化 Source hypothesis,占独立 prior 槽、不执行、不挤 A3 探索、无自由文本算子暗示;过冻结 development 课程回归门 [累计 regret 非增 ∧ harm/worst-class 非劣 ∧ ≥1 单元材料级改善 ∧ identity 单元零错误晋升]=A5 可见软先验;3+LOO 或 fresh 复现=更强 Shared TRY;新 Target held-in 终批;sealed Target 一次性验收)。**前置工程**:Scope 编译修复(四步序第③步)+observable 契约扩展(B 结构发现 2)升格为修订案关键路径——分类卡现状无五轴机器 Scope,"结构化 hypothesis"无载体。**两轨一门合流**:轨 1 = 在飞 S1b→S1c 按现行严格门跑(即 sol 步 4 对照的严格门基线臂,不停);轨 2 = 修订案批准后 Scope 修复→分级门实现→同课程同预算重跑→对表(覆盖率/regret/harm/错误建议率/探索挤占/确认率/成本)→**采纳标准 = harm 与错误晋升非劣前提下覆盖与 regret/成本改善,非"让 S1 跑通"**→胜则冻结新政策→Toe+Lightning2 双源 hypothesis(真独立家族)→课程回归→sealed 终验(D2 算力/新下载届时议)。元注:修订案流程本身即 Stage 3 机制的人工在环预演,论文"治理自进化"活案例。

### B 收口:GUARD_TIER_REACHABLE_AT_TWO_UNITS;S1b 发车(2026-08-26 20:3x,主线)

**B 判定**(提交 e64c684,0 LLM/0 fit/45min):三处接线落地(online_loop:181-189 写 task_episode_id=domain_namespace,与预测线 e1:937 口径对齐;source_skill:293-336/:488/:514-524 传 evidence_distinct_task_count 结构化 guard;cls runner:716-754/:822 接通既有 run_risk_skill_lifecycle),**阈值/语义逐字未动**;12 新聚焦测试+103 回归全绿;h0 锁无需重生成(dependency_shas 不含 online_loop);差分表:census 去重害证 1→2、risk_candidates 空→[outlier_mad]、Fast 视图无→含 guard(safety/allowed_tools=[]/不供候选/count=2)。**结构发现四条记账不修**:经验卡 Scope 仍卡在 SOURCE_APPLICABILITY 只有 task_kind(收窄系 Scope 改动);census 条件键不在 OBSERVABLE_FEATURES,分类中档要真 Context Scope 须动 observable 契约(正路,后议);分类 Episode 不写 task_signature 致 minted guard applicability={"const":True};guard body 渲染 dataset 串作证据出处。**小债**:method._snapshot 私有属性回写(必要,否则 guard 下轮不可见)后续补公开 setter。**S1b 发车(opus)**:四臂课程 runner——Static/A3-reset/K0-fixed/A5-online;K0 按 r2 合法性编译(排除 C40 Target-local hampel);域绑定三钩子照 r2 规格;A5-online 单元间走真 Slow 整合(含活化的 risk lifecycle,Slow 预算入总账);判分 = 密封 oracle regret 主指标 + sol 收紧门全套(质量/harm 非劣前提下 regret 或成本材料级改善);课程由 runner 按声明规则从密封 census 机械选定(2+ 害证生成单元 [mad/iqr 合法且 held-in 有害] + 2 可学正向 + 2 identity + 可选 HELDOUT_ONLY,正反序冻结);oracle 件仅判分组件读,零入臂视野;交付 = runner + 单元 1 四臂 smoke + 冻结课程表,**S1c 另书待主线复核 smoke 后发**。

### S1a-r3 收口:POOL_EXHAUSTED_FOR_TRY_CHANNEL;供给-价格地图定稿;S1 重塑为 guard+程序通道实验;B 发车(2026-08-26 18:5x,主线)

**普查定局**(提交 87ee7c1,0 LLM/342 fit/436s;38 单元预声明一轮算完,资格链终结,无 r4):无任何簇同时具备 ≥3 独立家族 LEARNABLE + ≥1 额外 LEARNABLE 考场。最近缺口:**repair_burst 家族够(ToeSeg1/Lightning2/ECGFiveDays 各自成族)差 1 个可学考场**(ECG200 与 ToeSeg2-burst 均 HELDOUT_ONLY);**hampel 可学 6 但独立家族仅 2**(GunPoint 族 ×4 并一 + PowerCons)。新知:ECGFiveDays impulse 强可学(+0.571/+0.322)、PowerCons 双模板可学、HELDOUT_ONLY 类扩至 9 单元(反馈可学性边界的分布性证据)。SonyAIBO 两 impulse 系 v2 公式 L<75 段长=0 的构造失败,如实记未调参。**供给-价格结论(论文级)**:治理对行动权的标价 TRY=3 独立可学正例+可学考场——本地池付不起;guard=2 独立害证——**付得起**(mad/iqr 在多单元非法/有害,课程内可自然产生 ≥2 独立害证 Episode)。**主线裁定**:(1) **B 发车**(opus):按 r2 行号修三处管道(online_loop 写 task_episode_id / source_skill 传 evidence_distinct_task_count / 分类线接通 risk lifecycle),只修可达性不动任何阈值语义,各配聚焦测试+r2 式可达性差分复审;(2) **S1 重塑**:四臂课程改考 guard+程序通道(A5-online 在 ≥2 独立害证后形成 Fast guard → 后续单元跳过死路的成本/harm 优势;域内学习与 HELDOUT_ONLY 正确弃权作质量读数),TRY 通道如实标注"供给未及价,未测";课程重组以害证制造与 identity/可学单元混布为原则,S1b 书待 B 落地后定稿;(3) **TRY 供给三选项呈 sol(下周议,不阻本周跑)**:预注册第三注入 family 一次性普查(本地,零获取成本,但新 family 成新簇未必补旧簇考场)/ 密封下载批(烧处女数据于 development,与 sol"处女度留终验"原则冲突,主线不荐)/ 晋升门校准(以 S1 实测误授权率议,禁以便利议)。

### S1a-r2 裁定:HEADROOM_WITHOUT_LEGAL_TRANSFER_PATH 定案;三条路呈裁(2026-08-26 16:1x,主线)

**判定**(提交 e74c021,0 LLM/0 fit/0.02s):正反序均无合法 Fast 可见分歧;9 单元池 0 个合法重组排列。**定量根因**:现役 TRY 授权 = LOO 摘一后 ≥2 ⇒ **须 3 个未受引导可学正例**(source_skill.py:217-257,r1 误算 2 够);hampel 簇可学 2/3 且 GPA↔GP pattern_view 字节相等、独立家族=1;burst 簇可学 2/3(ECG200 与 Herring 同为 HELDOUT_ONLY:held-in=0 而 held-out 正——"考官可见学生不可学"单元定型为一类);guard 档管道断裂依旧(三处,r1 已引行)。**非 TREATMENT_EMPTY**:域内可学在,缺跨单元合法通道。可学性判据全部引现役代码行(agg≥+0.005 且逐 view≥−0.005;Support Draft 门;delayed 须 classify_relation==POSITIVE)。**S1b 域绑定三钩子规格入账**(cell 构建打 domain_namespace/跨单元丢异域 Target-local/Source-derived 按 Scope v1 五轴放行;methods 第③步落地后拆 runner 墙)。**三条路呈用户与 sol**:(A) **S1a-r3 一次性扩池普查**(主线荐):全部剩余本地合适底物 × {impulse-v2, burst} 预声明一批、oracle 一轮算完、带可学性与家族独立性列,判"是否存在 ≥3 独立可学正例 + 课后匹配场";判词含 POOL_EXHAUSTED 出口——单批 take-what-comes,与顺序钓鱼有本质区别;(B) **guard 管道三处修复**(canon-vs-code 的 bug 修复:三分策略系已批教义、管道使中档成死码;修后 n≥2 害证可合法入 Fast,identity 单元的处理组激活;methods 手术,opus,修后须 r2 式可达性复审);(C) **晋升门校准问题呈 sol 作协议议题**:"3 个未引导正例"的门是设计校准还是任意常数?r2 审计即证据;实验便利不得作为调门理由,故此路只议不动。主线推荐 A+B 并行不悖(A 零治理改动,B 与 S1 无关也该修);S1b 维持冻结待 A/B 结果。

### S1a-r3 收口:POOL_EXHAUSTED_FOR_TRY_CHANNEL;本地池穷尽无 3+1 TRY 通道;无 r4(2026-08-26 18:xx,执行方)

**判定**:**POOL_EXHAUSTED_FOR_TRY_CHANNEL**(development;不升级课程;无 r4;S1b 仍冻结)。0 LLM;解释器 `D:\Anaconda_envs\envs\project\python.exe`。池先冻结后评分:本地 40 zip → 入选 19 底物 × {impulse-v2, burst}=38 单元;排除 21(8=r1 已测,11=TRAIN 点数>100000 含 HandOutlines,1=非有限 DodgerLoopWeekend,1=无 TRAIN KeplerLightCurves)。合并 r1 9 单元一起算。可学性复用 r2 谓词(`classify_relation==POSITIVE`;experience_memory.py:411-451;method.py:742-757 / 1466-1492)。

**各簇独立可学**(名称前缀 + 仅在 LEARNABLE 成员上的 pattern_view 字节归并):hampel LEARNABLE 6 / 独立家族 **2**(GunPointFamily + PowerCons;PowerCons 是新独立家族,仍差 1 族);repair_burst LEARNABLE 3 / 独立家族 **3**(ToeSegmentation + Lightning2 + ECGFiveDays=ECGFamily)但无第 4 个 LEARNABLE 考场(Toe2 burst 与 ECG200 均为 HELDOUT_ONLY);outlier_iqr 3/2;outlier_mad 3/2(Phalanx+TwoLeadECG);其余 ≤1。最近缺口 = repair_burst:**差 1 个可学考场**,不是差家族。SonyAIBO 两 impulse 因 v2 段长=round(L/150)=0 记 construction fail,未从名单删除。

**成本**:0 LLM / 342 fit / 600;墙钟 435.5 s / 5400 s;下载 0;r3 密封 36 份(2 构造失败无密封件)。**义务**:未跑适应臂;未调注入/扫参;未事后增删池;`methods/`/`runtime/`/`contracts/`/`operators/` 零改;r1/r2 工件与 9 份旧密封未覆写;他线文件未碰;未跑全仓 pytest。

**书外**:独立家族若对全部单元(含 identity)做 pattern 并查,会经 BeetleFly identity 把 GunPoint/PowerCons/ECG 塌成一族;已改为只在 LEARNABLE 成员上归并,密封数值未重算。无课程草案。无 r4。

**提交**:runner `--census-r3`;新密封 `s1_oracle/*`(仅新单元);`s1a_r3_pool_census.json/.md`;本节(含他书未提交的 r2 裁定条,未删改既有正文)。

### S1a-r1 判词收窄(sol 两点承重批评成立,主线自认失职);S1a-r2 发车;S1b 继续冻结(2026-08-26 15:5x,主线)

**批评一(成立,主线双重失职)**:S1a 时间线的"首次分歧 = GPA Target-local hampel 直接带入 Wine"违反正典(Target-local 限本域,AGENTS.md:170;跨域须 Episode→census→Slow→Source-derived;主线自己入典的 C40 修正案原文在案)——且该通道在代码上可走恰因四步序第③步(Scope 编译修复)未落地、Target-local 卡 applicability 仍仅 task_kind;照此跑 S1b 测到的是宽 Scope bug 而非演化,回落"复制答案"旧单元。主线裁定时未查时间线合法性,失职。**新站规:状态时间线每次转移必须标注放行它的正典条款(合法性列)**。**批评二(成立)**:资格门只查 held-out headroom 未查 held-in 可学性——Herring hampel held-in=0/held-out+0.047,Target 反馈不会批准,系"考官可见、学生不可学"单元。**主线推演的结构后果**:携带禁行+授权证据须未受旧卡引导(防循环授权)下,正序 Source-TRY 卡最早单元 3 后成型、唯一匹配场 Herring 不可学;反序正向在 4/6 位、卡成型即课终——现课程恐判 HEADROOM_WITHOUT_LEGAL_TRANSFER_PATH;repair_burst_segment 簇(3 员)若 held-in 可学,合法课程或需改建于该簇。**S1a-r1 判词收窄为:课程存在菜单 headroom 与条件化响应结构;"A5 拥有合法可学可跨单元生效的处理通道"未证**。**S1a-r2 发车(grok,0 LLM/0 fit,纯重聚合+走码)**:①禁 Target-local 跨单元携带 ②仅 held-in Support/delayed 亦材料级正向的单元计可授权 Source 证据(批准语义引用现役代码行,禁自发明阈值)③仅未受旧 Skill 引导的正例可授权新 Shared TRY ④重画合法状态时间线(Source-derived 何时成型/匹配哪些后续单元/Target 反馈能否批准),每转移标正典条款 ⑤正反序分别重算;附:GunPoint↔GPA 同族域证据独立性弱化注记;给 S1b 的"runner 层执行域绑定"实现规格。判词:LEGAL_EVOLUTION_TREATMENT_QUALIFIED(才批 S1b)/ HEADROOM_WITHOUT_LEGAL_TRANSFER_PATH(换课程结构)/ TREATMENT_EMPTY。

### S1a-r1 裁定:FULL_CURRICULUM_QUALIFIED;guard 档结构性不可达;S1b 按现状放行建议(2026-08-26 15:1x,主线)【判词被上条收窄;"按现状放行 S1b"建议撤回】

**门判**(提交 837b537,0 Fast LLM/0 Slow/69 fit/103s):9 单元双层 oracle 密封;正向 7、identity 2;**hampel 簇过资格门**(GPA+GunPoint+Herring,Pattern 交非空含 local_robust_z_peak=high 类部署可见特征);课程冻结 6 单元 = GPA→Wine→GunPoint→Ham→Herring→GunPoint-burst + 严格反序。**新证据**:repair_burst_segment 在 impulse-v2 下系 ECG200/Toe/Lightning2 的 menu oracle(CLS-4 burst 族中同算子为害)——同算子双命运,条件化再添一证;第二簇入账不进本课。**记录修正**:"ECG200=identity"(上午)被 oracle 层修正为"存在 repair_burst_segment 正解(+0.16 残差可回收)";hampel 被帽拒的三底物命运表不变;Wine/Ham identity 距 clean 上界残差 0.11-0.15 = Program Supply 缺口量化。**结构性发现(处理通道图)**:三分策略 guard 档在分类线不可达——online_loop.py:180-193 不写 task_episode_id(计数塌缩)、source_skill.py:472-478 不写 evidence_distinct_task_count、分类 harness 不调 run_risk_skill_lifecycle;主线"guard 中途解锁"预测在实现层证伪;有效通道 = Target-local capability 携带(单元 2 起首次分歧)+ Slow 授权 Fast-TRY(单元 3 后,LOO min_distinct=2)。**K0 纪律:C40 Target-local hampel 不得入 K0**(否则 K0-fixed 全课漏答案)。**主线放行建议:S1b 按现状跑**(审计时间线按现状算;缺口系过闭非泄漏;修后重跑即 guard 通道价值消融 S1-r2);guard 管道三处修复入具名 backlog。**预测入典(可证伪)**:四臂分歧主现单元 3/5,单元 2/4 近平;单元 6 考 impulse 系 TRY 卡被 Scope 压住不跨族误发(治理读数)。S1b/S1c 未自动衔接,待批。

### 纲领换轨定案:三段式演化实验(主线自立,sol 六点修订采纳);S1a-r1 发车(2026-08-26 13:2x,主线)

**换轨(用户驱动,主线定稿)**:证明单元从"动作 Skill 跨底物复制"(C38→CLS-CONF 阶梯,被三底物命运表证伪,退役)改为"**整台 Harness 知识状态随经历复利**"。三段式:S1 任务内演化课程(分类)/ S2 跨任务程序性迁移探针(S1 产出算子中立程序知识后;预测 cell 主探,AD 作 Task 卡惰性副守卫)/ S3 受治理 Instruction/策略自修订(**移出关键路径但留 ROADMAP 具名里程碑**;首考靶 = v2 教训自主重发现)。#38-46 的管道工程与能力测绘全部继承为地基。**sol 六点修订全采**:(1) 四臂 = Static(不适应)/ A3-reset(每单元 H0)/ **K0-fixed(同课前知识 K0,单元间禁写回,单元内正常 held-in)**/ A5-online(同 K0 起点,持续整合)——原"A5-冻结"废止(单元 1 特殊化混淆先验与演化);(2) **S1a 先证处理组存在**:每类经验编译终态、Fast 可见时点、Scope 是否来自部署可见特征、A5-online 与 K0-fixed 首次状态分歧单元;`evidence_distinct_task_count≥2` 须核现役编译器实际计数语义(risk_skill.py:246)+ 两害证同 Program/Context/first-fault;主线补:**Wine mad 害证系 0-LLM 预检非 Episode,课前池 mad 害证仅 ECG200 n=1,第二计数单元须课程内产生**;(3) 课程须 ≥2 个独立 Scope 相容正向单元,否则停在 S1a(主线加预注册双出口:FULL_CURRICULUM_QUALIFIED / SAFETY_ONLY_CURRICULUM 带决策件停);防钓鱼 = 候选池一次性预声明 + oracle 一轮算完;(4) **oracle 双层**:menu oracle(现菜单最优,无安全改进则 identity)判决策正确性 + readiness upper bound(clean/exact repair 上界)报残差——Program Supply 缺口不得洗成"数据无需处理";oracle 工件全程隔离于 Harness/Agent/Memory 视野;课程定名 development positive-control curriculum;(5) 门收紧:主读数 = 每单元 regret(menu-oracle 效用−实际冻结部署效用),报累计 regret / held-out 效用与 worst-class harm / 错误晋升 / Target 边际成本(LLM/fit/probe)/ 含 Slow 整合的累计总成本;**A5-online 须在质量与 harm 非劣前提下 regret 或成本至少一项材料级改善**;fit 数改称适应/搜索成本;(6) 单顺序单跑只记 `S1_DEVELOPMENT_EVOLUTION_SIGNAL`,复合措辞需 ≥2 套预冻结反平衡顺序(S1a-r1 一并冻结)。**批准范围:仅 S1a-r1(课程资格 + 双层判分器 + 状态可达性审计),禁自动衔接 S1b/S1c。**

### S1a-r1 收口:FULL_CURRICULUM_QUALIFIED;双层 oracle 密封;处理组非空(2026-08-26 14:xx,执行方)

**总判定:`FULL_CURRICULUM_QUALIFIED`**(development positive-control curriculum;非 fresh)。预声明 9 单元池一轮算完,禁事后扩池/扫注入。0 Fast LLM,0 Slow 排演(卡形态可由现役谓词+既有 `source_investigation_cls_v1` 推演),69 fit / 500,墙钟 103.3 s / 5400 s。

**Part A 双层 oracle**(ridge × `fit_only_artifact`;held-out = 官方干净 TEST;上界 = exact-repair 干净训练)。正向 7 / identity 2。hampel 簇:GunPointAgeSpan(+0.2627,残差 0.0095)/ GunPoint(+0.4067,残差 0.1133)/ Herring(+0.0469,残差 0.0781)。`repair_burst_segment` 簇:ECG200(+0.040,残差 0.160)/ ToeSegmentation1(+0.031,残差 0.083)/ Lightning2(+0.098,残差 0.164)——**菜单有解 ≠ 上界已达**;Wine/Ham oracle=identity 但 identity 残差 0.148/0.114,不得写成"无需处理"。GunPoint×burst 拉伸 oracle=`outlier_iqr`(+0.0133,残差 0.100)。密封件 `artifacts/functional/e2/s1_oracle/` 头部声明不得进入任何臂 prompt/store/检索。

**Part B 冻结课程**(hampel 簇过门:同 Task/Consumer + Pattern 交非空且禁用 dataset 名 + 同一 Program 几何):正序 `GunPointAgeSpan__impulse_v2 → Wine__impulse_v2 → GunPoint__impulse_v2 → Ham__impulse_v2 → Herring__impulse_v2 → GunPoint__burst_cls2`;反序为其严格逆。含 3 正向 / 2 identity / 1 burst。第二相容簇(`repair_burst_segment`×3)入账但不作本课动作族。

**Part C**:课前 K0 = `source_investigation_cls_v1`(TRY 弃权,无 `evidence_distinct_task_count`)→ **Slow-only**。C40 GPA hampel 是 Target-local capability,不得装进 K0。`evidence_distinct_task_count` 按 `context_summary.task_episode_id` 计(risk_skill.py:72-74/246);分类 `online_loop.py:180-193` 不写该字段,两单元害证会塌成 count=1;分类 harness 不调用 `run_risk_skill_lifecycle`;source 卡编译器也不写该字段 → **Fast-guard 结构性不可达**。A5-online vs K0-fixed 第一处 Fast 可见差异 = **单元 2 起始**(单元 1 形成的 Target-local Skill 被 A5 带入,K0-fixed 回 K0);单元 3(第二 hampel 正向)后 Slow 可授权 Fast-TRY。处理组非空,非 `TREATMENT_EMPTY`。

**提交**:独立 runner `evaluation/functional/run_e2_s1a_curriculum_oracle_audit.py`(不改共享 runner);密封 oracle 9 份;审计 `s1a_curriculum_audit.json/.md`;本节(连同主线未提交的换轨发车条与 Wine 关族条)。`methods/`/`runtime/`/`contracts/`/`operators/` 零改动;未触 `data/ucr_conf_downloaded/`;未跑正式臂;未跑全仓 pytest。解释器 `D:\Anaconda_envs\envs\project\python.exe`。

### Wine 预检触发关闭条款:impulse×hampel family 关闭;三底物命运表定型(2026-08-26 12:2x,主线)

**门判 FAMILY_CLOSURE_RECOMMENDED(提交 436cc71,0 LLM/2 fit/1.2s)**:v2 等比模板下 hampel 合法(0.0297<0.10,L=150 不变性五检过)但 headroom +0.0 < 门 0.0588,且 worst-class Δrecall −0.5556;唯一合法非空动作即 hampel,全局裁剪族(mad 0.245/iqr 0.164)超帽被拒。按 sol 预设条款 **impulse×hampel family 关闭,不再换数据追结果**;Part C 未跑。**三底物命运表(同一注入 family)**:GunPointAgeSpan(150 点)= 正向 +0.2690 / ECG200(96 点)= 校验器几何拒(v2 亦无解:round(1/150×96)=1 同 v1,halo 数学在短行不可压)/ Wine(234 点)= 合法但零 headroom+类伤——**同 family 三种命运,直接支撑"数据就绪由底物几何×Consumer×缺陷共同条件化"的核心论题,负边界证据入账**。书外仪器注:observer 末段 2 点回收咬偏(recovered_all_nodes=false),记债不修。**分类线定格状态**:生命周期已通(ECG200 229s 端到端)+ 治理链已证(T1/T2)+ 单域正例 n=1(GunPoint)+ 跨底物边界已测绘;Shared Skill 归纳与 impulse 系 A5>A3 跨域主考不可行,后续路线呈 sol:(a) 跨域正迁移头条押回预测线 Frep(已有 A5>A3 +31.7% 成本优势),分类章定位"第二任务生命周期+治理+条件化边界";(b) 以机制推理预注册设计新分类缺陷 family(不许数据挖掘式选家族);(c) 补 ridge/kNN 双 Consumer impulse 对照完善条件化证据章。D2/D3 继续封存,IPD 维持暂停,r3a 使用地图固化与 junction 移除等入 housekeeping 待发。

### sol 裁定:帽不动/注入模板等比缩放/暂停 IPD/Wine 预检制 dev 验证/否则关 family(2026-08-26 12:06)

(1) **0.10 修改帽保留,不为结果调高**;(2) 注入模板改为**随序列长度等比缩放**,比例从 GunPoint 正控机械换算、禁止扫描,并保证注入占比低于修改帽;(3) **ItalyPowerDemand 暂停**(24 点,现协议必然更不适配);(4) 许可**一次**本地 development 机制验证:长度≥150、规模小,**推荐 Wine**(57×234);**先 0-LLM"程序合法性+headroom"预检**(hampel 可执行 ∧ 确实正向)通过才花 LLM 跑 Harness;(5) 若仍无合法正向 headroom → **关闭 impulse×hampel family,不再换数据追结果**。定位一句话:分类 Harness 已能运行且会正确拒绝有害处理(ECG200 零害守住);缺第二个几何相容正向场;Shared Skill 归纳与真 A5>A3 跨域主考均未到条件。ECG200 反馈面未被证伪(outlier_mad Support −0.1429 被正确不晋升)。

### CLS-DEV-WINE:注入模板 v2 等比缩放 + 0-LLM 预检——FAMILY_CLOSURE_RECOMMENDED(2026-08-26 12:1x,执行方)

**判定**:**FAMILY_CLOSURE_RECOMMENDED**(development;非独立确认;禁 `CLS_CHAIN_CONFIRMED`)。冻结门 = `hampel_filter` 合法 ∧ held-in headroom ≥ max(0.005, 1/n_heldin)。Wine n_heldin=17,线 = 0.0588。hampel **合法**(cohort 修改分数 0.0297 < 0.10 帽)但 headroom = **+0.0000**,且 worst-class Δrecall = −0.5556(类交换、净精度不动)。按书停 Part C,不换数据,不花 LLM。

**v2 换算**:原常量 `run_e2_task_context_label_evidence_witness.py:37` `SPIKE_FRACTIONS=(0.08,0.20,0.80,0.92)`;`:38` `SPIKE_AMPLITUDE=16.0`;`:95-100` `_inject` 每位点写 1 点 → v1 段长=1。公式 `段长=round(1/150×L)`。L=150 不变性断言 **passed**(段长/位点/幅度/分数/注入字节五检全过)。Wine L=234 → 段长 2,注入 8/234≈0.0342,hampel 理论上界 16/234≈0.0684,均 < 0.10。ECG200 参考位 L=96 → 段长仍 1,理论 12/96=0.125 仍超帽(只作常数对照,未再跑该底物)。

**预检全表**(cohort 0.10 帽;consumer = ridge-raw-plus-difference-v1 / accuracy;held-in = 全 support 池 17 行):合法且非空动作仅 `hampel_filter`(0.0297,+0.0000);`identity` 与插补/resample/denoise_median 合法但 no-op; `outlier_mad`/`outlier_iqr`/`winsorize`/`repair_*` 及全局平滑族一律 `COHORT_MODIFICATION_FRACTION_EXCEEDED`。无第二合法正向动作。

**成本**:LLM 0/0;fit 2/200;墙钟 runner 账 1.2 s;下载 0。Part C `--dev-wine-run` 未开。

**提交**:runner 新入口 `--dev-wine-precheck`/`--dev-wine-run`(v2 仅这两口;既有 `--conf-*`/`--r2-*` 仍走 v1);工件 `t6_cls_dev_wine_precheck.json/.md`;本节。`methods/`/`runtime/`/`contracts/`/`operators/` 零改动;未触 `data/ucr_conf_downloaded/`;他线文件未碰;未跑全仓 pytest。解释器 `D:\Anaconda_envs\envs\project\python.exe`。

### 中午统一摘要:上午四步全收口;ECG200 dev 负结果定性为"校验器-几何失配";正式确认前置问题呈 sol(2026-08-26 12:0x,主线)

**上午成绩单**:(1) D1 BinaryHeartbeat 09:55 终止,COMPUTE_BUDGET_EXCEEDED+INSTRUMENT_SCALE_MISMATCH,提交 10f9fee;(2) T1 可见性修复落地(谓词 retrieval.py:274 单闸口,C40 卡 Fast 恒拦/Slow 保留,A3 视图 sha 逐字节不变,聚焦+fixture+回归 22/22 全绿,h0 锁机械重生成仅 retrieval sha 与 runtime_bundle_sha 移动,提交 03f2c1b);(3) T2 重放判 VISIBILITY_INVARIANT_HOLDS(卡装 store/Slow 可见,Fast 三面恒缺席;C40 双轮 level-shift 锁破,r2 探 outlier_iqr 获合法 Support;后端锁定 gpt-5.6-sol@agicto 核对通过;9 LLM/546s,提交 168cc99)——**惰性不变量成立,且确认"去误导只还冷启动,不送发现"**;(4) CLS-DEV-ECG200 判 **DEV_CHAIN_NO_POSITIVE**(229s/10 LLM/12 fit,管线首次正常规模端到端收口,提交 96db0e9)。**ECG200 负结果机制定性(比结果本身重要)**:hampel 在该底物被 cohort 校验器拒——修改分数 0.128>帽 0.10(47/70 窗超帽);注入模板固定段长,96 点短行占比天然高;唯一合法回执 outlier_mad 系 Support 有害;A3 守 identity(0.6000=Static,零害)。**定性:同 impulse family、不同 Program 几何/校验器命运——不是"修复不迁移"的证据;GunPoint 正例仍 n=1**。此系"协议常数泄漏底物假设"第三例(前:AD contamination=0.1、选靶漏计算门)。**呈 sol 决策件**:(A) 几何失配修哪端——注入模板按长度等比缩放(改注入,保帽),或 cohort 帽按 task_context 参数化(改帽,须防"按注入调帽"倒挂),或两者;(B) **ItalyPowerDemand 24 点/行,同模板+帽下大概率结构不可用**——正式确认选靶须同时过总点数门(≤100k)与长度-模板相容门(下限从模板常量机械导出),下载授权前必须先裁 (A);(C) 是否许可第二个本地轻底物 dev 跑(定位为注入几何×校验器机制研究,非钓正例;候选须长度≥GunPoint 级)。**队列状态**:sol 夜间队列全部履行(T0 终止/T1 过/T2 过/T3-T4 按门跳过)+重构后 dev 跑完成;无在飞任务;等 sol 裁 (A)(B)(C) 后发车。

### 夜间-早晨统一状态摘要:D1 计算不可行终止;sol 重构分类数据使用策略;队列重启(2026-08-26 09:5x,主线)

**T0 时间线**:BinaryHeartbeat 两臂 21:36 开跑,选靶+下载 26min 正常;A3 臂 22:02 起,**12.3h 仅完成 r1(probes=1, winner=None, delayed=None)**,进程全程单核 ~91% 真算非挂(累计 CPU ~11h);执行子代理 08:16 网络死亡,跑批本体独立存活由主线看护;09:55:06 按用户+sol 明确裁定人工终止。**判定:COMPUTE_BUDGET_EXCEEDED(主)+ INSTRUMENT_SCALE_MISMATCH(次)——非科学负结果,CLS-CONF 问题保持 OPEN,r1 无 winner 系局部观察不得引为不复现证据**。部分工件保留(选靶 census/ROSTER/终端轨迹/store 快照)。性能实证:378 万总点数底物单轮数小时,瓶颈为随点数放大的管线热路径(留 profiling 定位),LLM/fit 帽不封单位算量——选靶规则漏"行数×长度"的规则债被坐实。**sol 重构裁定(全采)**:(1) 立即停,判计算不可行;(2) **当前分类开发改用本地 ECG200**(100/100×96≈1.92 万点,证据等级明确 development)——本地 18 件跑过条件对者不可再包装为全新独立确认,但完全可用于接线/生命周期/Scope 编译开发/A3-A5 机制调试/已曝光 development 复现;(3) 最终独立确认另用满足计算门的轻量未曝光 Target(候选 ItalyPowerDemand 67×24,届时另行下载授权);(4) **选靶规则修正:先过公开结构计算门(例:总点数≤100,000)再机械排序,禁止单纯字典序**;(5) CatsDogs/Epilepsy2 继续封存,不得因已下载强用。**核心原则入典:开发数据可重复用;唯最终承重确认需处女数据**。**队列重启**:复活 dl 执行者出终止记录+提交 → T1 最小可见性修复(opus)→ T2 C40 单臂重放(grok,锁原后端)→ ECG200 development 级 conf 机制跑(接续 Scope 编译开发)→ 终考事宜(D2 处置/可行性门正式化/ItalyPowerDemand)另呈 sol。

### 夜间队列四点补丁(sol,2026-08-26 00:35)

(1) Fit/LLM 帽只限调用次数不限单次算量,清晨收尾是乐观估计非承诺;(2) 判定分型:进程退出/异常=INSTRUMENT_UNREADABLE;等 LLM 网络=按 transport timeout;CPU+I/O+日志全静且调用栈长期不变=疑似挂死;**仅"人为在明确墙钟上限处终止"才记 COMPUTE_BUDGET_EXCEEDED**;store 无写入单独不作依据;(3) **T2 后端锁定**:C40 replay 的模型/base URL/Prompt/预算/候选菜单必须与 C40 原跑一致,禁止顺带切新中转站——否则行为变化无法归因;(4) T4 启动条件完整式:D1 confirmed ∧ T1 targeted checks passed ∧ T2 visibility invariant passed ∧ T3 交集非空;smoke/profiling 只用已曝光 D1/fixture,零读 D2 值与标签。入典纪律:夜间各段只记简短机器结果,分支确定后统一一次状态摘要。

### 夜间串行条件队列裁定(sol 版全采 + 主线两处收紧,2026-08-26 00:2x,主线)

**T0(在飞)**:BinaryHeartbeat D1 两臂跑完为先。挂死判据改为:进程 CPU 连续 15min <5% 且 stdout/临时件/store 三面同静,或异常/内存持续异常增长;**store 暂无写入不构成挂死**(计算密集阶段批量落盘)。若因时长必须人工终止 → 判 **COMPUTE_BUDGET_EXCEEDED**(非科学负结果),**不得以 D3 替换慢 D1**(D3 仅结构/载入失败);运行期间禁改其在用 runner、methods/、共享注册表。**T1(D1 收口后,opus)**:最小 Fast 可见性修复,仅覆盖已观测 first-fault:"无授权 TRY 且无重复 scoped RISK → 整卡不进 Fast proposal view,只留 Slow"。复杂度上限:一个行为谓词+一个聚焦测试+一个既有 C40 fixture/integration 检查;禁止顺带实现完整三分 guard 平台/清债/重构 Memory(三分策略保留为教义,今晚只落地第一档)。**T2(T1 过后,grok)**:C40 已曝光账本单臂机制重放,只跑 A5,LLM 后端/Prompt 协议/预算不变;主判 = 无权卡确定性不在 Fast 视图 + level-shift 不再被固定诱导 + 探索通道恢复;**不要求 accuracy 精确等于 A3**(LLM 采样方差);只证治理,不产 capability。**T3(仅当 D1=CLS_CHAIN_CONFIRMED,主线自跑,0 LLM)**:按冻结五轴规则对 GunPointAgeSpan+BinaryHeartbeat 机械求交,输出 SCOPE_CANDIDATE_AVAILABLE / SCOPE_INTERSECTION_EMPTY;不授权 candidate、不开 D2。**T4(仅当 T1/T2/T3 全过,opus)**:构建终考三臂 runner(**新建独立 runner 文件**,不碰共享 runner;臂集 STATIC/A3/A4/A5 做成可选参数,**最终臂集留明早用户授权定**——sol 文本 T4 与早晨段有出入,主线裁定参数化),同预算/held-in 多轮/freeze/held-out Fast-only 骨架;用已曝光 fixture/D1 做 smoke;对长序列路径 profiling 并**交付 CatsDogs(约 1.48 万点)预计墙钟数**;只生成 plan,零读取 D2 数值与标签。**分支停止**:D1 不复现 → 完成 T1/T2 后停(关闭 impulse 跨数据集 family,不得开 D2 找第三正例);D1 仪器/计算停摆 → 转性能 first-fault 报告;T1/T2 不过 → 停,不扩修硬闯;Scope 交集空 → 双 Skill 留 Target-local,Shared 路线停。**明早唯一动作**:四门读数(D1 复现 / T1 修复过 / T2 探索恢复 / Scope 非空)全过后,由用户授权开 D2 跑终局三臂;开封前另决:接受 CatsDogs 计算成本,或按公开结构冻结总点数可行性门(**选靶规则遗漏计算规模——只限 TRAIN 行数未限行数×长度——记为规则债**,该门只能按结构定,不得按效用结果换靶)。夜间全程单执行线,不并行写共享文件,不处理背景债,不自动开 D2。

### 用户授权方案③精简版 + 下载选靶规则冻结 + CLS-CONF-dl 发车(2026-08-25 21:1x,主线)

**授权条款(用户原文要义)**:按冻结、outcome-blind 规则下载**最多 3 个**新 UCR 二分类数据集;D1 用于 CLS-CONF,D2 保持 sealed 用于通过后的 A5 vs A3,D3 仅作**结构性载入失败**备用;**不得因科学结果为负递补**;隔离用目录/roster/来源记录维持,**不新增**通用 SHA、双仓或 Evidence Ledger。**选靶规则(冻结于执行者接触档案元数据前)**:池 = 官方 UCR 单变量档案中,类数=2 ∧ 等长 ∧ 单变量 ∧ 官方 TRAIN 行数∈[40,400] ∧ 名称不在本地 `data/ucr_task_context` 现有 zip 之列者;按字典序取前 3,依序定角色 D1/D2/D3(池不足 3 则取实有并如实报);来源优先 timeseriesclassification.com 逐数据集 zip,记录 URL+日期;打包格式可机械转换为本地 loader 期望格式(值与标签不得改动)。**结构性失败分类(冻结)**:(a) 下载/打包失败 (b) loader 拒载(非有限值/格式)(c) 序列长度低于注入模板既有最小长度常量(执行者须引用代码常量行,不得现场发明门槛);仅此三类允许 D3 顶替失败者角色;**第一次 LLM 消耗后 D1 锁定,任何递补窗口关闭**;第二次结构性失败 → 停摆报告,不得追加第 4 件。**主线本轮不设名单预测**(三连预测失败后改制:对外部档案元数据无可靠先验,门改为"执行者必须在下载前把元数据过滤轨迹全文写入工件"供事后审计)。发车:下载+转换+CLS-CONF 两臂一书(grok);D2/D3 落库即密封(独立目录+roster 记录,开封需未来书面记录)。

### CLS-CONF-r3a 收口:语义审计后池仍空(结构性事实定案);选项①死亡;主线荐"一次授权密封下载批"(2026-08-25 21:0x,主线)

**审计判定(提交 e83f898,零 LLM/fit/下载)**:20 认领 runner = EXECUTES 10 / INCIDENTAL 0 / NO_TOKEN 10,每判引用 file:line;18 件 TRAIN∈[40,400] 可载入二分类**全部** condition_pair_used,合格 0 件。主线第三次门用预测(FreezerRegularTrain)亦被否证:source_prior 的 evaluate 复用 W55/W56 planner(:123/:354/:409),对其四件真执行条件对注入并开 TEST——r2"仅字段名"判断不成立。**三轮停摆链(r1 全提及→r2 字面 token→r3a 语义)各自诚实且机器可复核,门纪律三连胜:每次都拦在花 LLM 之前**。结构性事实定案:**本地 40 件 UCR 在 impulse 条件对下的确认预算 = 0**(此前 r1 的"20 runner 瓜分 40 件"再收紧)。census 卫生附注:Ham 认领系 "Hampel" 子串假阳性(实际仍由 W50b 出局);cls_op 文件点名远大于实际 _build_cell 五件——资格判定已按实际 roster。**决策收敛**:① 死亡(语义池空,机器证据);行数门放宽以纳入 Coffee 28 行等 = 看池改规,拒绝;② 切分级确认(同数据集冻结新切分)仍可行但弱——答不了底物泛化,sol 已定性为退路;③ 下载。**主线荐 ③ 之"一次授权密封下载批"**(将 sol 条件树中步 ④ 本就需要的下载提前合并,一次授权两用):预注册机械规则选自官方 UCR 档案的二分类、TRAIN∈[40,400]、不在本地 40 件之列者,字典序取前 K(建议 K=6);第 1 件即开用于 CLS-CONF 两臂,其余 K-1 件**密封入库**(照 #42f 镜像纪律:来源 URL/校验和/日期入册,双库,步 ④ 前不得开封);载入失败/资格不符则按序递补(规则先冻);全程 outcome-blind(下载时无任何注入读数存在)。待用户授权与 sol 核准后发车(下载+CONF 两臂一书,grok)。

### CLS-CONF-r2 PREDICTION_GATE_FAILED:主线预测被机器否证,Computers 出局;r3a 语义审计发车(2026-08-25 20:3x,主线)

**门判**:r2 按字面 token 规则重算,合格集为空 ≠ 预注册 8 件/Computers,零 LLM 停手(2.1s 干跑;Part 0 提交 2d055ea=r1 停摆件,3673366=r2 选靶记录)。**这次错在主线**:预测把"仅被 integrated_context/source_prior 认领"当成"未在 impulse 条件对下用过"——机器证据显示 `run_e2_integrated_context_harness_evolution.py` 的 TARGET_DATASETS+CONDITIONS 证明 Computers/PowerCons/Yoga/SemgHandGenderCh2/WormsTwoClass **五件真在该条件对下跑过**(语义出局,**sol 批准 Computers 的前提失效**);`run_e2_source_prior_evidence_fusion.py` 仅含 stable_task_event 作 W56 planned scope 字段名读取(token 碰撞,FreezerRegularTrain/GPMvF/GPOvY 三件系字面过度排除,语义上大概率合格)。执行者纪律正确:未现场改规则,停在门上。**站规强化(两轮连续手推名单出错后:r1 执行者 Earthquakes、r2 主线 integrated_context)**:候选池每条排除/放行必须携带引用到 file:line 的机器证据,禁止任何一方手推名单入书。**r3a 发车(grok,审计 only,零 LLM)**:对全部 20 个认领 runner 逐个分类 EXECUTES_CONDITION_PAIR / INCIDENTAL_TOKEN / NO_TOKEN,每判引用证据行;按审计结果机械重算 40 件资格表(条件对未用 ∧ 二分类 ∧ [40,400] ∧ 可载入),字典序选靶;**无论结果停给主线复核,两臂留 r3b**;若选中者为 GunPoint* 近亲须特别标出(独立性弱化,呈 sol)。语义规则本身系 sol 已批口径("没在本次 impulse 缺陷-修复条件对下看过结果"),r3a 属正确执行而非改规则;全程 outcome-blind(仍无任何候选的注入读数)。预测(仅供门用,不入规则):至少 FreezerRegularTrain 150 行合格且字典序居首。

### CLS-CONF-r3a 停在选靶:CANDIDATE_POOL_EMPTY——语义审计后合格集仍空;source_prior 经调用链实为 EXECUTES;0 LLM 收工(2026-08-25 20:4x,执行方)

**判定**:**CANDIDATE_POOL_EMPTY**(审计停点,两臂未开)。选中 Target = **None**。20 个认领 runner 分类 = EXECUTES_CONDITION_PAIR 10 / INCIDENTAL_TOKEN 0 / NO_TOKEN 10。机械重算:condition_pair_used = 认领方被判 EXECUTES 且数据集在其**实际 roster**(常量/调用,非文件名提及);合格 = 未用条件对 ∧ 二分类 ∧ TRAIN∈[40,400] ∧ 可载入。40 件无一同时满足。GunPoint 同族警示未触发(无选中者)。

**20-runner 一行证据**(详表见 `artifacts/functional/e2/t6_cls_conf_r3_selection.md`):action_credit `TARGET_DATASETS`+`ARTIFACT`/`EVENT`+`_condition_inputs`(:34/:40/:275);curvature 存疑归严(报告字段读 + TRAIN fit-only `_inject` :202);integrated `TARGET_DATASETS`+`CONDITIONS`+evaluate 循环注入(:38/:46/:392);pattern_mass NO_TOKEN(:28);program_binding 双条件 `_condition_inputs`(:199/:223);promoted 复用 W55 planner + `condition="fit_only_artifact"`(:113/:330);s0_census NO_TOKEN(仅注释提及);source_outlier NO_TOKEN(roster=monash/metr_la,Ham 系 Hampel 子串误认领);**source_prior EXECUTES**(evaluate 复用 W56/W55 planner 双条件注入 + `_prepare_train_execution` 再注入 fit_only_artifact 后打开 TEST,:123/:354/:409——否证 r2/发车书"仅字段名读取");cls1/cls1_r2/cls2/cls3/cls4 NO_TOKEN(MCAR/burst,非本条件对);cls_op EXECUTES 实际 roster 仅 SOURCE+TARGET 五件(:132/:134/:1316);impulse_repair NO_TOKEN(monash/fred);W48 witness / W49 transfer / W50b confirmation 均为 EXECUTES;temporary_excursion NO_TOKEN。

**资格表摘要**:可载入二分类 38;不可载入 2(DodgerLoopWeekend 非有限值、KeplerLightCurves 缺 TRAIN)。TRAIN∈[40,400] 且可载入的 18 件(Computers 250 / Earthquakes 322 / ECG200 100 / FreezerRegularTrain 150 / GunPoint 50 / GunPointAgeSpan 135 / GPMvF 135 / GPOvY 136 / Ham 109 / Herring 64 / HouseTwenty 40 / Lightning2 60 / PowerCons 180 / SemgHandGenderCh2 300 / ToeSegmentation1 40 / Wine 57 / WormsTwoClass 181 / Yoga 300)全部 `condition_pair_used`。r2 预测 8 件出局链:Computers/PowerCons/Yoga/Semg/Worms ← integrated :38+:46+:392;FreezerRegularTrain/GPMvF/GPOvY ← source_prior 调用链 :38+:123+:409。主线门用预测"FreezerRegularTrain 字典序居首"被否证。

**两臂**:未跑。无 held-in、无 Skill/冻结、无 held-out。未改 `methods/`/`runtime/`/`contracts/`/`operators/`;未覆写 r1/r2 工件。census 缓存名单与 40 zip stem 一致,未重扫。解释器 `D:\Anaconda_envs\envs\project\python.exe`。

**提交**:隔离工件 `t6_cls_conf_r3_selection.json/.md`(未覆写 r1/r2)+本节(连同主线未提交的 r3a 发车条)。义务自报:LLM 0;fit 0;下载 0;`methods/` 零改动;他线文件(`AGENTS.md`/`README.md`/`PROJECT_STATE`/`SUCCESSOR_BRIEF`/`ROADMAP`)未碰;存疑归严仅 curvature 1 条。

### sol 批准 CLS-CONF 口径①(Computers)+ Scope 归纳规则冻结 v1 + CLS-CONF-r2 发车(2026-08-25 20:1x,主线)【Computers 前提已被 r2 机器证据否证,见上条】

**sol 裁定**:当前非实验失败,系"发车前发现确认数据集不够独立,正在修正选靶"。Earthquakes 剔除确认;机器重算后 8 件合格,预注册字典序选 **Computers**——它在其他类型实验用过,但从未在本 impulse 缺陷-修复条件对下看过结果,**可承担 development 级独立确认**。条件树:Computers 复现 → GunPointAgeSpan+Computers 构成两独立正例 → 按冻结规则归纳候选 Scope → **下载真正未用的新分类 Target 做终局同预算 A5 vs A3**;不复现 → 不得强行生成 Shared Skill,GunPoint hampel 只保留为 Target-local。census/junction/坏链修复定性为**实验准备红利,非方法成果**。一句话定位:分类正向能力已有单域正例,现在准备第二独立域确认;确认后才有资格进入真正跨域 A5 vs A3 主考。

**Scope 归纳规则 v1(冻结于 2026-08-25 20:1x,先于主线见到 CLS-CONF-r2 任何逐 Episode Observation 细节;GunPointAgeSpan 侧细节虽已见,本规则只定轴名,值由数据交集机械产生,不含任何 GunPoint 特化值)**:Source-derived candidate 的机器 Scope = 五轴合取:(1) task_kind,(2) consumer_id,(3) metric,(4) 部署可见局部 Pattern 标签(仅取 Episode 当时已记录的 observation/public_features 字段,禁止事后重构),(5) Program 作用几何类。各轴取值 = **全部 n≥2 支持正例 Episode 已录字段的交集**;任一必轴交集为空 → 不生成 Shared candidate,各正例保留 Target-local;dataset 名不得作为轴。生成的 candidate 进入新 Target 时仍受 Target Support 门与 delayed 批准约束(三分可见性策略第三档)。

### CLS-CONF 停摆:INSTRUMENT_UNREADABLE(候选池空)+ 分类数据预算耗尽事实 + 主线裁定建议(2026-08-25 18:2x,主线)

**判定**:CLS-CONF 未开跑即停——预注册候选池为空。全仓 census(修复两仪器缺陷后:自指 junction 使 `os.walk` 重入、计数乘 2 的幂,按 realpath 去重+跳 reparse point 修复,扫描 25-30min→约 1min;`data/tsquality` 坏链接使 `rglob` 抛 OSError,改 `os.walk` 剪枝)证实 `data/ucr_task_context` 全部 40 个数据集均系 20 个既有分类 runner 的 roster 成员;唯二偶然提及者(DodgerLoopWeekend 非有限值+20 行、KeplerLightCurves 打包缺 TRAIN)独立失格。执行者未事后放宽规则(0 LLM/0 fit),纪律正确。**C40 +0.2690 是否孤例仍未回答**。Part 0 提交 9cf4ceb/9fbdf64(r2 工件+h0 锁重生成后 40F→2F,38 条 lock mismatch 清零,残留 2F 归另线);runner 改动 324e8fc;conf 工件未 commit(停摆件)。**主线复核纠错**:执行者"可放行"名单含 Earthquakes 系笔误——工件 `claiming_runners` 显示其属 `action_credit_candidate_ordering`,恰是其书外发现 4 点名的同注入条件对(fit_only_artifact/stable_task_event)前线,选项 1 下必须排除。**结构性事实(建议固化进 PROJECT_STATE_AND_DATA_MAP)**:本地分类数据预算已尽——20 runner 瓜分 40 数据集;census 表为首份完整归属清单;两条早期 runner(action_credit_candidate_ordering、task_risk_confirmation_adaptation_curve)用过与 C38 相同条件对。**三选项与主线建议**:①"未使用"收窄为"未在 impulse 条件对族下使用"(outcome-blind:看过的只是各线用了哪些数据集,未见任何候选在本注入模板下的读数,不构成拿答案挑确认集;放行 8 件=Computers 250/FreezerRegularTrain 150/GPMvF 135/GPOvY 136/PowerCons 180/SemgHGC2 300/WormsTwoClass 181/Yoga 300,字典序机械选靶落 Computers,非 GunPoint 近亲,证词不弱化;排除名单须由 runner 机器化重算——grep 各认领 runner 的条件 token,不得手抄)②切分级确认(弱,不答底物泛化,仅作 ① 不可行的退路)③授权下载(现纪律禁;**建议保留给步 ④ 终局 A5 vs A3**——头条实验值得真处女数据,且本地预算已尽,步 ④ 迟早需要)。**主线推荐 ①+③ 组合**:CLS-CONF 用 ①(Computers),步 ④ 提前规划密封下载批。待 sol 裁定后发车;发车书按模型分层归 grok(机械重算+重跑,零方法设计)。

### 常备纪律补充:子代理复用(用户指示,2026-08-26 20:55,主线)

同一条工作线的后续任务**优先 resume 既有子代理续话**(同模型;上下文长度控制,不开到 1M 级),不必每书新开:续用可保留仓库熟悉度与既有纪律吸收,省 ramp-up。新开仅限:新工作线、或旧会话上下文已臃肿/无关。当前可复用锚:S1 审计线(grok,S1a-r1/r2/r3 谱系)、methods 手术线(opus,T1/B 谱系)、S1b 构建线(opus,在飞)。

### 常备纪律:执行方模型分层(用户指示,2026-08-25 17:41,主线)

**委派模型路由**:低中难度、边界清晰的任务(跑既定 runner/replay、报告誊写与格式核对、文件整理、allowlist 内小修、按明确规格的机械改动)→ **grok 4.6 fast**(cursor-grok-4.6-xhigh-fast)提速;高难度攻坚(方法级设计、跨多文件手术、复杂归因调试、共享 Harness 核心代码改动)→ **opus**(耗时长,适合攻坚)。每次发书前先判难度定模型,默认能用 grok 就不占 opus;主线(根 Agent)始终负责任务书设计、裁定与整合,不外包。

### C40 裁定三处再修正(sol 终版,主线独立核后全采):根因改写为"惰性失效缺位";禁事后补 Scope;三分可见性策略(2026-08-25 17:35,主线)

**修正一(根因改写,覆盖下条目的原因排序)**:"Source 零 POSITIVE"降为输入条件,非根因——反事实下若治理正确,零正例只预测 A5≈A3(冷启动等价),不预测 A5<A3;−0.269 的差值只能由"无行动资格知识未惰性失效"(无可行动证据→仍生成 capability 卡→Scope 仅 classification→进 Fast 改提案)制造。系统在执行层已做对(verifier 拒供应),漏在影响层;完整不变量:**无行动资格知识须在所有影响行为的层上惰性(执行层+提案可见层)**。C40 重放即验此不变量,判读用机制级标准(提案不再被 level-shift 固定、能获合法 Support 回执),不用端点 accuracy 相等(LLM 方差)。**修正二(禁事后补 Scope)**:GunPoint hampel Skill 不得手工按已见结果挑 Pattern 条件(n=1 Scope 归纳不可辨识,任何特征合取都"符合"=拿结果选条件)。正确路径:旧 Skill 保留为 GunPoint Target-local → 按**事先冻结的编译规则**从当时已录合法 Observation 生成新 scoped candidate → 独立 Source/Target 验证;历史 Episode 若未录足部署可见 Pattern,诚实停留本域,不得凭描述补成 Shared Capability。**主线追加收紧:编译规则须在查看 CLS-CONF 逐 Episode Observation 细节前冻结**(结论级成败可知,字段级规则不依赖看数据即可写,如"Scope=Task×Consumer×Metric+pattern_view+Program 几何,取 n≥2 支持 Episode 交集"),否则泄漏在规则层复发。**修正三(不建权限状态机)**:工程上仅三分:无授权 TRY 且无重复 scoped RISK→Slow-only;有重复 scoped 害证→Fast 仅得**结构化** avoid/downweight guard(非自由文本——C40 卡 RISK 恰是自由文本+算子名);有重复 scoped 正证→Fast 可得 TRY/候选仍受 Target Support 门。现有字段够用。当前 repair_level_shift 负例 n=1,连 guard 不够格,整卡 Slow-only。**修订后排程**:CLS-CONF 收口 → 可见性单面修复 → C40 development replay(验不变量,非 capability)→ 冻结最小 Scope 编译规则(先于看 CLS-CONF Observation 细节)→ 仅当 CLS-CONF 与 GunPoint 构成两个独立同 Consumer 同可观察 Pattern 同 Program 几何正例时生成 Source-derived candidate → 第三个未用 Target(**看 Outcome 前机械确定**)同预算 A5 vs A3;若不复现或无共同可观察 Context,不得强并 Shared Skill。**工作纪律入册(第二次治理漂移后)**:每项治理工序必须能指出其解锁正向迁移主线哪一步,否则不排期;主线仍=有 Scope 的正向 A5→Target 迁移,治理是可信度保障非主体。

### C40 归因深挖裁定(用户+sol):first-fault = Scope 编译丢失 + 无权卡 Fast 可见;主线四处受纠;晋升路径教义;四步修复序(2026-08-25 17:3x,主线)【原因排序与"补 Scope"表述被上条 17:35 修正覆盖】

**归因定稿(sol 排序采纳)**:−0.269 系"负迁移机会损失"非直接数据伤害(A5=identity=0.5823,A3=0.8513)。原因排序:根因 = Source 无真实正向先验(9 Episode 零 POSITIVE,TRY 空);行为 first-fault = **无授权文本仍影响提案**(具体点名 repair_level_shift 者系 **RISK 段**——执行者报告称 OBSERVE 与工件不符,段级归因不可靠,恰证按段立规是打地鼠);放大因 = 单候选轮(实为每轮 1 候选非"四发");本轮排除 = 反馈不可靠(A3 三面同向)与菜单无算子(hampel +0.269)。**更早根因 = Episode→Skill 编译丢 Scope**:Episode 检索严(Task×Consumer×Metric+pattern,experience_memory:277),编译后机器 applicability 仅 task_kind==classification(工件 :1648,runner :157)——"comparable context_condition"未成机器条件,整卡向全分类广播;检索(retrieval:145)只按 applicability 匹配数排序,Skill ID 里的 Consumer 名不是条件。**连带警示:GunPointAgeSpan 正向 hampel Skill 同样仅 task_kind Scope(runner :530)——本域冻结部署有效,升跨域 Source 前必须补合格 Scope**。

**主线四处受纠(自认)**:(1) OBSERVE 段规则押错段;(2) "宽 WHEN 配有执行权卡"方向反——正确 = **权力越大 Scope 越窄**(宽 Scope 只配算子中立程序性指导;算子相关 TRY/RISK 须窄机器 Scope;无授权 TRY 且无重复授权 RISK → 整卡留 Slow 不进 Fast);(3) 双遍提案否决(破同预算 + 人为保 A5 不输,押后二阶段);(4) "先验与反馈都需治理"降为次级洞见,主线仍 = A5 更快更好形成有效 Target-local Skill。**晋升路径教义入册**(无权卡的三个合法出口):同域 Target-local(Support+delayed 批准,限本域)/ 跨域正向 Capability(多相似机器可识别 Context 重复正向 → Slow 新 revision:明确 Scope+授权 TRY,先 probe)/ 跨域风险 Skill(重复害证 → 授权 RISK,可降权避用弃权,不得越权荐他算子);旧卡不悄扩权,新证据产新版。**四步修复序(sol)**:①修 Fast 可见性(无授权 TRY/RISK 的 Source 卡只留 Slow)→ ②C40 已曝光账本单面重放 A5(机制测试非 capability:看 A5 是否恢复正常探索)→ ③修 Skill Scope 编译(Target-local 限本域;跨域绑 Task/Consumer/Metric+部署可见 Pattern+Program 几何;dataset 名不作跨域依据)→ ④真 A5 vs A3 正向迁移(需真实 Positive Source 池:CLS-CONF 正例 + GunPointAgeSpan hampel Episode 转 Source → 第三个未用 Target)。**排程**:CLS-CONF 在飞;落地后按 ①+② 一书发车(同 session);③ 随后;④ 待正例池成。

### #41b-lite 执行与最小 V10(2026-08-23,执行方报告)

**Part 0 检查点(0 LLM / 0 重训 / 0 AD 评估)**:`git update-index --really-refresh` 后 `git status` 实测 8 件修改(六收尾文件 + 两 docs;刷新前 stat 缓存确实吞改——载重运维发现兑现),逐文件 add、全程未用 `git add -A`。轮始发现 t5_lifecycle_v1.json/.md 为上一次复跑烟测的 CRLF 覆写(未还原),从 687af6e 逐字节还原并核符(处置同 #41 追认先例;本轮自测覆写一次后再次还原)。核验:V9 登记回读——method.py `after_t5` = ccf2b837…a4fa3、e1.py `after_t5` = e5501fe9…1097f,两者恰等于各自收尾后工作树哈希(登记语义 = #41 全程终态)。三份既有 untracked 测试(closeout 时间戳三件 = `test_e1_v2_protocol_repair.py` / `test_skill_evolution_e0.py` / `test_skill_revocation.py`)只跑不入库、不删除;MKL/Savgol 崩溃保持挂账(`test_f1_forecast_pilot` 原样不动,零 skip 标记写入)。

**Part A 最小 V10**:清单机制 = runner 内注册表(`FROZEN_SURFACE_V*` + 逐文件 before/after 注册,非独立注册表文件)。成员 = V9 成员收尾后哈希(39 唯一路径逐文件登记,before = V9 末次注册值、after = 检查点值,全部 carry 无新移动)+ `experience_memory.py` 新增纳入(`FROZEN_SURFACE_V10` = V9 + 1,40 唯一/41 原始条目含历史重复项,`_freeze` 切 V10);零代码移动;`runtime_bundle_sha` 依赖图零改动(Memory 覆盖走清单成员资格,#41b v1 的扩依赖裁决已撤销遵循)。键方言单断言 = T5 Part A A3(a)(b) 复跑:legacy fixture 与 `TASK_CONSUMER_KEY_FALLBACK` 字面量 `forecast|ridge|sMASE` 相等;online_loop 现役无引号硬编码残留,写回键全部经 `task_consumer_key(task_spec)`(:165),唯一幸存提及 = 解释移除的注释。

**Part B 四项验证**:`git diff --check` 干净;四测试 18 passed(`test_g1_proposal_guidance` + 上述三件 untracked,共 18 items);T5 `--smoke-only` 复跑 Part A 11/11(含 A3 键断言 (a)/(b))、Part B **22/22**、`llm_calls=0`;冻结面测试前后两测均 `{"files": 40, "drift": [], "ok": true}`(V10 唯一路径口径;V9 为 39 唯一/40 原始,历史二口径并存已由 #35 勘误记)。

**判定:`V10_READY_FOR_T6`**(另两候选不成立:CHECKPOINT_CONTENT_MISSING——内容俱在,含轮始工件脏态还原;BEHAVIOR_REGRESSION——18 + 22/22 + 11/11 全绿)。Part 0 提交 = 8 文件(六收尾文件 + 两 docs),sha 于回报与 #42 Part 0 幂等核验记入。纪律:O9 零下载零读取(检查点仅对既有冻结成员做字节测量,不开频率/标签值);NOAA 2025 / beyond_17520 / SMD 零读取;零 spawn;另一线停笔(其三件 untracked 测试保持其所有权与内容);歧义如实上报:清单成员资格对 bundle 盲区的覆盖照裁决执行(盲区条目仍在册),V10 切换后 T5 runner 冻结面口径自动 39→40 属授权移动非漂移。

### #31(2026-08-22,S2)

检查点 46ed5e2(8 files)。**CANDIDATE_COMPILES + LODO_TRANSFER_SUPPORTED(双向)**。Part A 硬门过:官方 OmniAnomaly 28 机文件获取(242.3 MB 仅 scratchpad),内容匹配定位(精确字节索引查表,非信号推断),28/28 逐元素一致、无缝铺满 [0,708405),拼接序非数字非字典(machine-1-5 起 machine-3-1 终,在册禁重推);官方 train=dev/held-in、test=sealed(仅报总行数 708420);#30 悬案澄清:[0,8760) 整块落在 machine-1-5 train 内,系一台机器 24 通道被当 24 序列报。Part B:证据池去重 21→12(traffic 8 + noaa 4;13→4 塌掉的 9 条全为同键重放);卡 shared_outlier_repair_with_per_series_guard_v1,四固定字段照裁定(SHARED_CANDIDATE/GUIDANCE/support_required/no_free_try),programs=[hampel,iqr,mad,winsorize]+算子无关 per-series guard(VETO+RESCOPE),适用条件全部部署时可观察(缺失可为零/z峰≥4/outlier_fraction>0/离散度即需 guard),插补+阶跃 out-of-scope 带底物指针。Part C:执行方自查废掉首版两条循环判据后,C1 traffic→noaa 4/4 SUPPORTED(3 条聚合藏害全被 guard 抓)、C2 noaa→traffic 置险 4/4 SUPPORTED(2/2)、C3 12/12 仅标 INTERNAL_CONSISTENCY_ONLY。**跨域量化副产品(升入 C8 证据链):12 行证据 5 条受害全部聚合为正,聚合单独捕获 0 次**。主线裁定:(a) C1 几乎不可证伪、C2 为信息方向,两向 SUPPORTED 挂 n=4 caveat;(b) 两侧证据均 in-selection,卡方向读数不得表述为 out-of-selection,该级证据只能来自 S3/S4;(c) #18/#19 缺口经核为零成本(去重键下与已入池行同票),提取器形状留案不修;(d) 242 MB 不入库,补记 28 文件 sha256 + 来源 ref 使重获取确定;(e) 实体粒度(NOAA 单变量实体 vs SMD 38 通道实体)为 S1b 第一项。S1b 预算解锁(0 LLM / ≤100 重训),书已发。
