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

### #31(2026-08-22,S2)

检查点 46ed5e2(8 files)。**CANDIDATE_COMPILES + LODO_TRANSFER_SUPPORTED(双向)**。Part A 硬门过:官方 OmniAnomaly 28 机文件获取(242.3 MB 仅 scratchpad),内容匹配定位(精确字节索引查表,非信号推断),28/28 逐元素一致、无缝铺满 [0,708405),拼接序非数字非字典(machine-1-5 起 machine-3-1 终,在册禁重推);官方 train=dev/held-in、test=sealed(仅报总行数 708420);#30 悬案澄清:[0,8760) 整块落在 machine-1-5 train 内,系一台机器 24 通道被当 24 序列报。Part B:证据池去重 21→12(traffic 8 + noaa 4;13→4 塌掉的 9 条全为同键重放);卡 shared_outlier_repair_with_per_series_guard_v1,四固定字段照裁定(SHARED_CANDIDATE/GUIDANCE/support_required/no_free_try),programs=[hampel,iqr,mad,winsorize]+算子无关 per-series guard(VETO+RESCOPE),适用条件全部部署时可观察(缺失可为零/z峰≥4/outlier_fraction>0/离散度即需 guard),插补+阶跃 out-of-scope 带底物指针。Part C:执行方自查废掉首版两条循环判据后,C1 traffic→noaa 4/4 SUPPORTED(3 条聚合藏害全被 guard 抓)、C2 noaa→traffic 置险 4/4 SUPPORTED(2/2)、C3 12/12 仅标 INTERNAL_CONSISTENCY_ONLY。**跨域量化副产品(升入 C8 证据链):12 行证据 5 条受害全部聚合为正,聚合单独捕获 0 次**。主线裁定:(a) C1 几乎不可证伪、C2 为信息方向,两向 SUPPORTED 挂 n=4 caveat;(b) 两侧证据均 in-selection,卡方向读数不得表述为 out-of-selection,该级证据只能来自 S3/S4;(c) #18/#19 缺口经核为零成本(去重键下与已入池行同票),提取器形状留案不修;(d) 242 MB 不入库,补记 28 文件 sha256 + 来源 ref 使重获取确定;(e) 实体粒度(NOAA 单变量实体 vs SMD 38 通道实体)为 S1b 第一项。S1b 预算解锁(0 LLM / ≤100 重训),书已发。

