# DEV-TEMPO-AUG-DECISION-LEARNING

日期：2026-09-19。
负责人：Opus（项目研发执行者）；Planner：Astra。
状态：可分发执行规格。用户将本书交给执行者后，按本书范围连续实施；文档写入本身不代表实验已启动。
唯一新运行目录：_scratch/dev_tempo_aug_decision_learning/。

## 0. 本包要解决什么

目标是改进从经验中学习研究策略的能力，取得真实交付收益。保留批级 Fast/Slow、七工具、TempoPFN 增强、共享 Consumer 与域级 Workflow，不把系统降为固定配方查表，也不要求无卡 Fast 先胜过简单方法。

主假设：相比现有“从轨迹提炼/修补规则”的学习契约，将同一批证据围绕“可观察条件—实际决定—延后效果—强简单对照”组织，并要求每条复用规则处理支持与反例，能够生成更有用的 Workflow。

本包比较的是“Slow 任务契约 + 输入组织 + 规则证据表达”的联合改动，不将收益单独归因于某句话、某字段或去掉某条禁令。其他方法部分在新旧两条学习线间保持相同。

必须回答：
1. 新契约是否减少无依据的旧规则继承，产生可检验的决策假设？
2. 候选经真实试用与一次修订后，是否改变材料构造或最终交付？
3. 新契约选出的卡是否胜过旧契约选出的卡、无卡、同预算随机搜索以及开发选出的固定方案？
4. 把固定方案纳入系统采用集后，系统实际选择了什么，质量与完整成本如何？

正结果方向是“学会何时、如何值得偏离有证据支持的默认处理”。默认处理也由合法开发数据决定，研究者不手工指定 NoMixRecipe、AmpResample 或其他答案。

## 1. 对辅助意见的采用与修正

采用：
- 每条规则，包括继承的规则，都须说明证据与限制；
- 旧卡作为待检验历史假说，移出新契约的轨迹正文，保留精确关联；
- 固定方案进入系统采用比较，不能只作报告中的旁观对照；
- 每种契约两次互不读取输出的形成调用，机会与预算相同；
- 必要的零拟合读数整理并入本包，不另立诊断项目。

不采用以下推论或门槛：
- “两包 argmin 0/14，因此 C_A 无信息”：候选集合、别名和共享 Job 不独立；精确命中最优不等于成对排序能力。只能按正确单位描述，不能据此禁止学习。
- “画像能分开域，因此能判断经验适用”：域可分性不等于处理收益可预测。本包单域、无 Router，不做跨域画像距离项目。
- “提示词字节不同/条件看似不触发，便可零成本判断行为有无差异”：一般自然语言 Workflow 的行为必须真实执行。字节检查只确认知识实际加载、严格别名与技术不可达，不能代替效用试验。
- “3/3 seed 是安全门”：它约束训练重复间一致性，不能保证未来时间效果；不作为研究者预设的通用提交规则。
- “J+成本”未定义加权分数：本包质量优先，成本单列；只在数值平局时依预定成本次序决胜。
- 固定方案获选不冒充学习成功；本包不以整理负结果为主要目标，也不为出现正号改数据、指标、预算或判词。

## 2. 必读、复用与允许改动

必读：
- 项目 AGENTS.md，尤其 §2.2、§3、§5.9、§6–9。
- docs/DEV_TEMPO_AUG_WORKFLOW_LEARNING_LOOP_TASK_2026-09-19.md。
- evaluation/main_protocol_p4/batch_research_tempo_aug_workflow_learning_loop.py。
- evaluation/main_protocol_p4/batch_research_tempo_aug_workflow_skill.py。
- methods/ttha/domain_skill.py、methods/ttha/batch_research.py。
- methods/ttha/batch_base/tempo_aug.py、tempo_source.py，以及现役账本、标签与恢复代码。
- 父包 _scratch/dev_tempo_aug_workflow_learning_loop/ 与前包 _scratch/dev_tempo_aug_workflow_skill/ 的冻结配置和原始工件。

建议新增一个入口 evaluation/main_protocol_p4/batch_research_tempo_aug_decision_learning.py，复用现有模块。共享层只加必要的显式 opt-in；不复制训练器、账本或第二套 Harness。

Opus 是实施者，不是实验中的 Slow/Fast。实验模型继续请求 cpa-grok-4.6，核验返回 grok-4.6-build；沿用父包真实代理入口、temperature=0、输出上限 12,000。不另做付费连通探针，不静默换模型，不输出凭据。

旧运行包和参考仓库只读；不修改 AGENTS，不 git commit，不清理其他执行线文件；新增 SHA/Hash 为 0。不得打开 2016 年封存区。本包不增加算子、TSFM、Router、采样降权或 Consumer 调参。

## 3. 数据与信息权限：全部是开发实验

沿用 RD02 排序后的 12 站。每 Job：
T=[t-672,t)，输入 192、预测 48；
C_A origins=[t,t+48]；
C_B origins=[t+96,t+144]；
E origins=[t+192,t+240,t+288,t+336]。

| 阶段 | Job | t | job_index | 用途 |
|---|---|---:|---:|---|
| 初始证据 | S1/S2/V1/T2 | 3360/4560/5760/8160 | 1/2/3/4 | 前包四条无卡轨迹、全部合法 T/C_A/C_B |
| 初始证据 | T4/Q2 | 10560/12960 | 5/6 | 前包每批无卡、旧 W1、旧 W2，合计六条轨迹 |
| Practice | L1/L2 | 15360/16560 | 7/8 | 两契约所有新候选真实试用，C_B 供修订 |
| Select | L3/L4 | 18960/20160 | 9/10 | 父子与无卡真实运行，C_B 选优与系统采用 |
| 开发复验 | L5/L6/L7 | 21360/22560/24480 | 11/12/13 | 冻结后比较两条学习线，最终评分 E |

所有 Job 前缀均为 RD02_。本表全部结果已有不同程度的研究者曝光，L5–L7 也已经评分，必须标记 EXPOSED_DEVELOPMENT_REPLAY。它们不能再称 fresh、独立测试或正式泛化验收。

初始 Slow 只读表中前六 Job、前包那十条轨迹的 T/C_A/C_B，不将学习闭环包的 L1–L7 经验提前混入。
Practice 本包实际运行后，其 T/C_A/C_B 才进入修订；Select、开发复验的任何反馈不进入 Slow。
所有 Slow/Fast 请求均禁止 E、含 E 的报告结论、研究者根据 L5–L7 写出的答案。
历史 E 即使已曝光也不作为本包学习输入；新包相关评分在最终统一阶段才读取/计算。
本包没有新增独立学习情境；同一 Job 的多臂、多个 seed 和重复使用不可当独立样本。

不重做全仓曝光普查。使用现有清单核对表内身份、几何与可评分性即可。资格失败保留，不换时点/实体/阈值。
最大原始行读取上界仍为 24864 exclusive。RD01B、PRSA row>=24864 与其他 Final 不开放。

## 4. 共同运行配置与缓存

全臂继承父包最终配置：
- MLP、2000 updates、AdamW、batch=64、训练 seed=[20269181,20269182,20269183]；
- 原合法父对、T scaler、Linear 服务输入、原始观测真值、missing-aware normalized MSE；
- 父/子损失各 0.5；None 沿用原无增强路径；
- 七原语、53 程序、最多三步、原互斥/执行顺序；允许统一或最多八条实体条件规则；
- 四公共参照 None、FixedMixup、P_AmpResample、P_NoMixRecipe，全部有合法 C_A、均可直接提交；
- 每条 Fast 每 Job 最多 4 个新完整 assignment、16 次 LLM、24 次工具、400,000 输入+输出 token；
- rendered Workflow + Principles 总上限 6000 字符，scope=const:true，已知域加载；
- Fast 可提交自身任何已评估材料，不强制 argmin C_A，不强制更多观察、分组、多步、换算子或用满预算。

材料 RNG 不变：tempo_source_v2_entity_program；
SeedSequence([2026091800, job_index, program_index, entity_index])。
同一实体、程序、Job 不因臂或生成顺序改变随机材料。

可以只读复用两旧包和上一学习闭环包的物理缓存，必须绑定数据、roster、T、父对、scaler、完整 assignment、RNG、训练 seed、Consumer、环境与服务几何。缓存不匹配则正常新拟合，不重新解释旧分数。
旧模型/数值不能被新包写回；评分产物放在新 root。不能把其他臂候选或未来分数暴露给当前 Fast。

本包 Fast 轨迹原则上重新运行：不能用旧包相同卡片的历史轨迹冒充此次真实试用。物理训练缓存命中可省计算，逻辑候选额度和方法独立运行成本不减。
本包严格别名只在同 Job、同冻结知识、同初始工具状态及实际模型输入一致时复用一次执行，并显式记一次，不能当独立重复。

Fast 基础系统提示与工具合同对所有臂相同，继续使用父包 FAST_SYSTEM。新处理仅通过冻结卡进入 Fast，不把新的 Slow 教学要求暗中加给全部 Fast。

## 5. 同一证据，两种学习契约

### 5.1 公共事实与默认方案

重用原 census 的事实范围：逐 Job 特征与实际观察、行动顺序、完整候选/assignment、逐 seed/origin C_A/C_B、commit/理由、成本和失败。
数值按物理材料去重；行为轨迹不删除。缺失不填零。三个 seed 和两个 origin 不冒充独立批次。

从初始六 Job 的四公共参照计算：
J_source(p)=mean_j[mean_s C_B(p,j,s)/mean_s C_B(None,j,s)]。
最小者记 P_source；容差 1e-12，固定 ID 次序决平局：None、FixedMixup、P_AmpResample、P_NoMixRecipe。
这只是初始开发默认及证据摘要，不是 E oracle，也不是强制提交规则；两种 Slow 都获得相同结果和底层数值。
P_source 后续不按 Practice/Select 重写。它不绕过 Slow 直接注入 Fast；固定策略作为独立对照执行。

在一个 evidence_summary.json 内，零拟合整理初始六 Job 的公共参照成对 C_A→C_B 同向率、排序与损失差。每 Job 先单独给值再等权汇总；同材料/别名去重，平局单列。
保留逐 seed 配对差及 SE，n=3 的 95% t 区间用 df=2，不能用 2SE 冒充。
不从六个 Job 拟合所谓“可分辨阈值”，不据此定门或把反号自动归因于漂移/噪声。
此摘要及全部底层事实同供新旧契约。

### 5.2 旧契约 O

两次形成均使用父包 SLOW_A1 的任务正文与原 evidence_review schema；不使用读过首次输出的 SLOW_A2。
census 保持旧式组织（轨迹内带 knowledge_loaded），附上两线共同的 P_source 和零拟合摘要。
candidate_id 在每次输出仍可为 W1/W2，由运行器加 O1_/O2_ 前缀区分；不让 ID 改变正文。
这是旧契约在共同独立生成安排下的对照，不声称逐字复现上一整包。

修订使用父包 SLOW_C；输入本包共同 Practice 事实、父卡及固定方案比较表。旧式 branch-local counterexamples 保留，两线共同拥有底层信息。

### 5.3 新契约 N：可直接用于系统提示的核心正文

在共同角色、权限、反馈定义、工具/预算和禁止未来泄漏条款之后加入以下语义，英文实现不得改变含义：

    Your goal is to learn a reusable research Workflow that improves the delayed
    forecasting utility of Fast's actual committed material under the fixed budget.
    Compare the research policy with the source-selected fixed default and the
    other public references. Prediction quality is primary; costs are reported
    separately. Saving experiments alone is not evidence of a quality improvement.

    Historical cards are hypotheses being evaluated, not instructions for you.
    No recommendation or prohibition inherits authority merely because a past card
    contained it. Evidence requirements apply equally to retained and new rules.
    Legal grouping and multi-step plans remain available; neither is mandatory.

    Distinguish the C_A evidence available to Fast at the decision from the C_B
    evidence used to evaluate whether that decision should be reused. Acknowledge
    limited time coverage. Agreement across training seeds is not evidence of
    agreement across future time periods.

    Propose up to two alternative Workflows, or KEEP. Explain each alternative's
    decision hypothesis relative to the simple default: what observable information
    could justify a different construction, experiment allocation, or commit;
    what outcome would contradict that hypothesis; and what to do when the
    available information cannot distinguish the alternatives.

    For every reusable rule, including any retained rule, identify its supporting
    observations, delayed outcomes, and counterexamples. If there is no supporting
    evidence, mark it as an exploratory hypothesis, not an established principle.
    A counterexample need not force deletion, but explain why the rule is retained,
    narrowed, changed into a test, or left unresolved. Do not handle it merely by
    appending a generic warning while claiming the rule has been validated.

    Do not invent unmeasured causal explanations or performance. Do not impose a
    universal 3/3-seed gate, a universal trust/distrust of C_A, or a preferred
    primitive unless the supplied evidence supports the scoped recommendation.
    A short, uniform or stopping Workflow is legitimate; so is conditional
    construction. Material complexity or diversity is not an objective.

新输入采用：
1. 公共合同与默认方案事实；
2. 按 Job 的 decision_cases：可见信息、实际决定、数值后果、对默认的差、其他已评估材料；
3. 完整材料数值表与动作序列；
4. historical_hypotheses：旧卡原文每份仅一次，由 trajectory 的 card_ref 精确关联。

旧卡内容不删，身份映射不丢，失败与不利证据不丢；新输入只是去重与重排相同信息。禁止用另一个 LLM 替 Slow 先写“正确规则”。

两线科学事实等价需在本包一个 smoke 中核对：Job、material、assignment、所有 CA/CB seed/origin 数值、action、commit、加载卡原文、失败与成本都能对应。没有新 hash 或通用证据平台。

新输出保留兼容的候选外壳，在不可部署的 evidence_review 中以 decision_rules 替代泛化的 kept/dropped 清单：
- rule：建议正文对应的具体规则；
- observable_condition：Fast 合法可获得的条件，缺信息时说明要检查什么；
- change_vs_default：构造、实验安排或提交何处可能不同；
- support_refs / counter_refs：实际 evidence_ref，可为空但必须注明证据不足；
- evidence_status：supported_in_these_cases / exploratory / unresolved；
- counterexample_handling：保留/收窄/转为实验/放弃的理由与限制；
- predicted_behavior_and_failure：下一次预期行为及何种结果将反驳它。

这些字段不是研究者规定的答案；不能凭人工感觉“仍然保守”就拒卡或追加调用。
evidence_review 不作为原始历史直接注入 Fast。可部署 Workflow/Principles 不能包含历史 Job、实体 ID、行号答案表或逐案例成绩。

初始两线均 Workflow-only、principles=null。真实试用后的修订阶段两线均允许 Principles，容量一致。

## 6. A→E 连续流程

### A. 两线各两次形成：共四次科学调用，最多八张初版卡

调用顺序 O1、N1、N2、O2。每次只读各自契约下的同一初始事实，不读取其他形成输出，也不读取本包 Practice。
每次最多两个候选；KEEP 合法。每次至多一次纯格式/引用合同纠正，不因内容重复、保守或不受欢迎要求重写。

temperature 保持 0。两次分开的调用只提供重复形成机会，不保证统计独立或输出多样性；如逐字相同，记别名和有效候选数，不调温度追求不同。
四次全部完成并冻结后才进入 Practice。技术失败不等于 KEEP；任何科学形成调用若因 API/预算未完成，应按恢复规则处理并标记形成技术未完成，不能把缺少候选算作正常弃权后宣称完成双契约比较。
一线无候选则该线记 NO_PROPOSAL，另一线继续；两线都无候选则收口为形成阶段未提出候选，不启动伪造的卡片比较。

形成后只做零成本结构/加载检查：schema、长度、引用存在、scope、实际注入。
不做“条件看起来不触发所以不试”的人工筛卡；无法仅由静态文本证明行为相同，不能据此别名。

### B. Practice：L1/L2 上真实试用全部合法初版卡

最多八候选×两批=16 条 Fast。L1 用候选 ID 排序，L2 反序；新旧线交错。
所有分支 commit/失败状态冻结后，统一开 C_B；不产生 E。

两线分别按
J_P(v)=mean_{L1,L2}[mean_s C_B(commit_v)/mean_s C_B(None)]
保留各自最好的一个父版本，平局按形成调用次序和候选 ID。
本包保留一父/线是将预算用于新旧契约对照，不等于只生成或只试用一个方案；不可减少两线各两次形成和全部候选试用。

选择父版本是分配修订预算，不要求先胜过无卡或固定方案。每线有完整父版本都获得一次修订机会。

### C. 两个父版本各一次反馈修订，允许 Workflow + Principles

两线接收相同范围的事实：
初始证据、两线全部本包 Practice 轨迹与数值、父卡、分支内最佳已评估材料、P_source 及四公共参照、成本。
不得只把对本父卡有利的其他分支交给它。
Practice 结果供形成后修订属于合法学习；Select/开发复验结果仍禁止进入。

O 使用旧 SLOW_C。
N 使用 §5.3 核心要求并明确：
- 区分未执行、执行但没有收益、选择有误、候选未覆盖等可直接观察的事实；
- 训练随机性、时间变化、材料机制只能作为待检验解释，不能由一次反号自动定因；
- 相比父版本改变一个主要决策机制，允许重写相关流程及配套 Principles，不限制为在句尾加一道门；
- 每项新增/保留原则都要处理证据与反例；KEEP 是完整合法结果；
- 预期触发变化的场景必须说清，未触发不能报“修订已被验证”。

每线最多一个子版本，一次科学修订加至多一次合同纠正。不得再做第二次科学重写。
父/子都保留，KEEP 记真实别名。

### D. Select：L3/L4 上选择卡片与系统策略

每 Job 运行 F0、O_parent、O_child、N_parent、N_child，最多五条 Fast；不存在或严格别名项合并。
L3 按上述顺序，L4 反序。统一 commit 后开 C_B，不开 E。之后不再调用 Slow。

J_S(v)=mean_{L3,L4}[mean_s C_B(commit_v)/mean_s C_B(None)]。

分别冻结：
- W_O：O 父子中最低 J_S 的非空卡；
- W_N：N 父子中最低 J_S 的非空卡；
- Fixed_dev：四公共固定程序中最低 J_S 的那一个；
- H_O：在 W_O、F0、四公共固定策略之间按同一 J_S 选择；
- H_N：在 W_N、F0、四公共固定策略之间按同一 J_S 选择。

若某线没有非空卡，仍可按固定策略/F0 定义该线系统，单独记无学习提案；其卡片对比标不可用，不填 0。
技术不完整不等于科学弃权，不悄悄删除失败候选后宣称全线公平完成。

数值平局容差 1e-12：
- 父子卡平局父优先，再按候选 ID；
- 系统平局先选固定策略（无需研究 LLM），同类固定按 §5.1 顺序；非固定之间依 Select 平均逻辑拟合数、平均实际 token、固定 ID 次序决胜。
不设临时显著性门，不混用 J+任意成本惩罚；不把宽区间判为等效。

Fixed_dev 必须在首次开发复验前冻结。它进入两线系统采用集，不能再排除。
这会改变采用层，但对 O/N 完全相同；采用层收益与新 Slow 契约收益分开报告。
原始 L3/L4 结果先前已曝光，选择仅属开发，不是独立科学验证。

### E. L5–L7 开发复验：运行卡，即使系统选了固定方案

每 Job：
- F0：无卡；
- F_O：W_O；
- F_N：W_N；
- RandomSearch_B4：继承父包两统一+两条件化分布及冻结 RNG，四个新完整 assignment；
- Menu_CA、Fixed_dev、P_source、四公共固定参照由对应真实模型评分，不新增研究 LLM。

不存在卡的线不冒充 F0 卡；严格相同知识及请求可别名一次并说明。
即使 H_N/H_O 选择固定策略，仍真实运行本线最佳非空卡，检验候选潜力和采用质量；不能因系统选择不用卡就跳过学习机制读数。

顺序：
L5=[F0,F_O,F_N,Random]；
L6=[F_N,Random,F0,F_O]；
L7=[F_O,F0,Random,F_N]。

Random 只复用原分布/seed/合法性与去重规则，不按旧 E 挑抽样种子。所有研究臂看到相同四公共参照，仅能看到本臂自己新评估的候选。
三 Job 全部分支完成/记失败并冻结交付 → 统一 C_B → 所有所需 E 预测冻结 → 统一 E 评分。
新候选未有 E 缓存则正常评分；不能先看一批 E 再决定其余批次的方法。

H_O/H_N 是在 Select 冻结的执行策略别名，不在复验时重新按 E/CB 或当前 CA 选择不同策略。
固定策略的独立部署只需训练其交付模型；为本包公平研究准备四个参照的公共实验开销另列，不能给固定策略虚构四倍部署成本。

## 7. 结果与定性核查：不拿格式改善冒充效果

主比较：
d_contract(j,s)=E(F_O,j,s)-E(F_N,j,s)；
d_skill=E(F0)-E(F_N)；
d_default=E(Fixed_dev)-E(F_N)；
d_system=E(H_O)-E(H_N)。
另报 F_N/H_N 相对 Random、Menu、None、P_source。

主汇总 G=mean_j[100×mean_s d(j,s)/mean_s E(None,j,s)]，三 Job 等权。
单位为相对该 Job None 平均损失的百分点，不能称相对 Fixed 的百分比。
同时给实际损失、逐 Job/seed 差、配对 SE 与 n=3 的 t95% 区间；起点全部保留。
同 seed 跨 Job 也不等于多份独立 Agent 决策；不把九个 seed×Job 单元伪装成九个独立任务。两次 Slow 调用不够估计可靠生成方差。

按实际卡片/轨迹给短证据表，不调用额外评分 LLM：
- 旧禁令是否继续存在，其证据实际支持到了什么范围；
- 反例是否改变执行规则，还是只出现在解释文字中；
- 新卡预言的条件是否真实发生，Fast 是否采取预言中的不同决定；
- 候选、观察顺序、利用中间反馈、commit 与停止如何改变；
- 对同一 Job，哪个完整材料改善/恶化了真实延后效果。

没有触发某条规则时记“这条变化未被考察”，不能以提交文字中提到规则算执行证明。
明确禁止用“3/3 seed”充当时间泛化证明，不预设保守规则一定错误或复杂方案一定更好。
若对子卡仅有 Select 配对读数、复验未选中子卡，不能宣称取得其独立复验效果。

最终报告期才追加零拟合 C_A→C_B、C_A→E、C_B→E 描述：
- 四公共参照为共同候选集主表，每 Job 六个非重复程序对，别名/平局单列；
- 分支私有集合另表，不把大小不同集合的 argmin 混成“随机准确率”；
- 同时给选择损失/遗憾，不能只给命中率；
- 这些读数不再回流本包卡片，不拟合事后阈值，不触发追加实验。
不计算跨域画像分离度来替代“特征能否预测动作收益”。

成本三层：历史资产、本包新增形成/试用/修订/选择、每批独立部署。实际物理缓存成本与无缓存逻辑成本都给。
若质量逐位相同且省调用，可以报告使用侧效率收益；计入形成成本计算有条件的摊销批次数。不能一律写“无从摊销”，也不能忽略同质量的更便宜 Menu/固定策略。

## 8. 实现验收与失败恢复

本包一个合成 smoke，检查新增风险即可：
1. 新旧事实等价、旧卡移动后引用可追溯、初始/Practice/Select/E 权限正确；
2. 两次独立形成互不读取输出、规则引用和 6000 字符处理、新正文实际加载；
3. 父/子/KEEP/NO_PROPOSAL/技术失败与严格别名正确；
4. Fixed_dev 纳入 H_O/H_N，系统固定时仍执行隔离卡，平局处理正确；
5. 缓存不会泄露其他臂候选与标签，学习失败不自动启动后续阶段。

跑相关 controller tests 与父包 smoke，已知历史不相关失败单列，不借机修平台。
可用脚本+缓存完成接线则 0 拟合；确需真实接线最多 3 次，仅用原 RD02_T1 的已曝光几何和三 seed，不读 E。

沿用已有环境与 OpenMP 初始化；禁止 KMP_DUPLICATE_LIB_OK 掩盖冲突。
首次付费前冻结所有实际 prompt、数据表、候选/调用顺序、阶段预算、压缩规则；用实际序列化尺寸走现有字节预留检查，不能只按估算 token。
两线只能按共同事实规则去重/压缩；超尺寸不能只删一线的不利案例或数值。技术尺寸问题不得记 KEEP。
零成本检查通过后连续完成，不因定性读数“不够新颖”插入人工反复退卡。

未知 usage 保持 UNKNOWN 并按现役规则阻断收费续跑，用户接受未知费用不解除标签边界。
技术拟合同配置最多重试一次，Practice/Select/复验各最多两次额外拟合，共六次。
全包额外原样 HTTP 最多四次，每请求最多一次。保持原请求，不在恢复时重问 Slow 换卡。
标签已打开的受影响阶段不补跑。普通无方法变化 bug 可在未开相应标签时修，保留事实；重大方法/数据/预算变化交用户决定。

## 9. 资源上限与预计安排

最大候选数按 O/N 各四张初版、一父一子，全部互异计算：

| 阶段 | 常规拟合上限 | Fast 分支 / 调用上限 | Slow 请求上限（科学+纠正） | token 阶段帽 |
|---|---:|---:|---:|---:|
| 只读整理/接线 | 3 | 0/0 | 0 | 0 |
| A 形成 | 0 | 0/0 | 4+4 | 2,400,000 |
| B Practice | 2×(4+8×4)×3=216 | 16/256 | 0 | 3,200,000 |
| C 修订 | 0 | 0/0 | 2+2 | 2,400,000 |
| D Select | 2×(4+5×4)×3=144 | 10/160 | 0 | 2,800,000 |
| E 开发复验 | 3×(4+4×4)×3=180 | 9/144 | 0 | 2,400,000 |
| 合计 | 543 | 35/560 | 12 | 13,200,000 |

总拟合尝试帽 549（含六次重试），逻辑 LLM 请求帽 572，HTTP 尝试帽 576。
每分支仍受原 400K token 帽，阶段和全包上限优先；这些不是承诺每条分支能用满其局部上限。
同一个总账本，恢复不重置；阶段预算按开始时已用量加本阶段额度冻结，不将结余转给下一阶段。
物理缓存应显著减少实际拟合，但不保证命中率，不通过变更 seed 追命中。

数值活跃上限 4 小时；首次真实拟合或付费调用起的执行活跃上限 8 小时。实现和人工等待单列，资源竞争不自动扩帽。
调度估算：实现与检查约 2–4 小时，实验约 2–4 小时，报告约半小时；以实际代理与缓存为准，不承诺固定完成时刻。
不得为节约时间跳过修订、删除固定对照或只挑有差异的 Job；上限耗尽如实保留技术不完整。

## 10. 交付与下一步

同一 root 交付：
- REPORT.md：首页直接回答四主问题，逐 Job 质量/伤害/完整成本；
- METHOD.md：两契约共同部分与实际差集、阶段信息权限、开发身份；
- frozen_config.json、source/default/Select 的冻结选择；
- 两线实际请求、原始响应、初版/父/子、关键证据引用和运行轨迹；
- result.json/tables.md，模型/预测沿用现有结构；
- 一份合并检查记录，零拟合摘要不另起项目。

报告至少展示一条“旧规则/假说→支持和反例→新决定→实际触发/未触发→真实延后结果”的链。
不能把选优胜出、格式合规或因果解释文字写成质量已改善。

完成即停止付费运行，并给下一项具体建议：
- 若新契约在多个复验批次取得对旧契约/无卡的有实际幅度增益，且对固定/随机的比较支持研究过程价值：提出新时间段验收方案，列明所需数据授权；本包不打开。
- 若候选有效但系统选择固定/无卡：分开报告候选潜力与采用问题，不追溯改选择。
- 若系统仅因固定策略纳入而改善：归于采用层，不能归于 Slow 学习。
- 若交付仍相同但成本下降：报告效率与摊销，同时判断固定/菜单是否仍支配。
- 若新契约仍没有交付改善：不要自动再修一轮提示词，也不自动转成“固定配方卡+3/3门+画像匹配”。提出一个有证据的任务/学习条件调整建议，保留用户的性能改进目标。

任务书末尾追加实际收口，项目既有计划文档只追加短回执，旧数字不覆盖。不自动启动第二域或最终验收，不把本包开发结果包装为完整原论文复现/持续进化验收。

## 11. 给 Opus 的直接执行指令

按本书实施 DEV-TEMPO-AUG-DECISION-LEARNING。目标是找出能改善交付的经验学习方式，不是继续解释负结果。
先完成同事实双契约输入、固定策略采用层和必要 smoke；通过后按 A→E 连续执行。
保留两契约各两次独立形成、全部候选试用、各一父一次修订、父子选优与最佳非空卡复验。
不以静态字节检查猜测行为，不手写正确增强答案，不开放封存区，不新增 hash，不改 Consumer。
完整成本与正负读数都报告，不能为正号换规则；整包完成即停止。用户转发本书即授权本书列明的开发实施和预算范围，未转发前不视为已启动。

## 12. 实际收口（2026-09-19 13:25，执行者追加；正文未改）

- **执行 COMPLETE（A→E 全部完成；三复验批次 F0/F_O/F_N/Random/Menu 五臂配对完整；0 拟合失败、0 分支失败；全部身份 EXPOSED_DEVELOPMENT_REPLAY）。** 报告 `_scratch/dev_tempo_aug_decision_learning/REPORT.md`，方法与差集 `METHOD.md`，数值 `result.json` / `tables.md`，冻结 `frozen_config.json` + `frozen_config_amendment_1.json`，两线原文 `formation_a/{O1,N1,N2,O2}.json`、`revision/revise_*.json`，证据 `evidence/`，冻结选择 `formation_a/frozen_choices.json` / `test_catalog.json`，账本 `budget.json`，事故 `INCIDENT_2026-09-19_smoke_stray_worker.md`。入口 `evaluation/main_protocol_p4/batch_research_tempo_aug_decision_learning.py`（复用 W worker 与 L census/parsers；共享层无改动）。smoke 25/25 + controller tests 26 + 父包 smoke 13/13；接线 PASS 0 拟合。
- A：O1、N1、N2、O2 四次 PROPOSE，8 张互异卡（O 1745–2260 字符：public-gate commit / T-structure then ablate / conservative_public_commit / feature_gated_allocation；N 734–1630 字符：ca_skeptical_public_stop / feature_gated_public_family / stop_commit_public_recipe / recipe_default_ca_collapse_exception）；O1、N1 各一次合同纠正。零拟合摘要：P_source = P_NoMixRecipe（J_source 0.915），六批公共参照 C_A→C_B 同向率 19/36。
- B：16 分支 COMPLETE。L1 八卡全交付 P_NoMixRecipe；L2 四张 N 卡仍交付 P_NoMixRecipe（C_B 1.287×None），四张 O 卡交付 FixedMixup / ResampleOnly / P_AmpResample / res_only。父：O1_W2（J_P 0.7597）、N1_W1（0.9329，四 N 卡并列）。
- C：O → O1_W2-r2（正向 level-shift 路由改写，带 Principles）；N → N1_W1-r2（配方非 3/3 且 Amp 均值胜 None 时开两次配对消融，写明触发场景，带 Principles）。
- D：J_S 无卡 0.8876、O 父=子 0.7776、N 父=子 0.5600 = P_NoMixRecipe 0.5600；W_O = O1_W2、W_N = N1_W1、Fixed_dev = P_NoMixRecipe、**H_O = H_N = P_NoMixRecipe**（N 卡并列，固定优先）。父子交付逐 seed 相同，子卡新增分支未触发。
- E：F_N 三批全部提交 P_NoMixRecipe（0 构造）；**d_contract +9.36 pp（L5 +28.1 [+++]，L6/L7 0）、d_skill +12.59（L5 +28.1、L6 +9.7 [++−]、L7 0）、d_default 0.00（9 seed 单元全 0）、d_system 0.00**；F_N 对 Random/Menu +9.36、对 None +13.76（L7 −9.2，对 FixedMixup −21.5）；t95 区间全部跨零。F_O 在 L5 自造消融后提交 P_AmpResample（−28.1 对配方）。复验公共参照同向率 C_A→C_B 7/18、C_A→E 8/18、C_B→E 7/18。
- 判读：新契约产生了逐规则带证据状态与反例处理的可检验决策规则，但学到的是"总是提交固定默认"，没有学会何时/如何偏离；其对旧卡/无卡/随机/菜单的正号全部等于固定程序自身优势（对 Fixed_dev 精确 0）；两条旧禁令（不分组、不 3 步）被标 supported 而从未检验；系统收益全部归采用层（固定程序纳入采用集），不归 Slow 学习。属 §10 "候选并列固定而系统选固定" + "仅因固定策略纳入而改善"。
- 成本：物理拟合 36（0 失败）、缓存命中 579、109 请求 / 109 HTTP、4.05M 入 / 0.23M 出 token、1 次 UNKNOWN（用户接受）；A–D 增量 83 请求 / 3.33M+0.19M token / 15 物理（96 逻辑）拟合；复验 26 请求 / 21 物理（102 逻辑）拟合；数值活跃 13.1 min，账本墙钟 98.5 min。账本外 UNKNOWN：实现期 smoke 误起 worker 约 10 min（估 0.3–0.9M token）、被杀 request 4。
- 偏差（均有记录、用户裁定）：harness 内存保护杀掉驱动 → request 4 UNKNOWN 接受 + 原样重发一次；N 修订纠正被预留拒绝 1,631 token → 修订阶段帽 +8,192；付费前 N 提示追加输出长度句。未做第二次科学重写、人工筛卡、静态别名、Consumer/算子/Router 改动、封存区读取、新 hash、git commit。
- 下一项建议（未执行）：0-LLM/0-拟合读数——用 13 批 × 4 公共参照冻结分数核对是否存在任何 T 侧可观察字段在 ≥10 批上与"配方对 None 的延后比值"稳定同向；没有则把下一包目标改为反馈块设计而非 Slow 契约，有则以该字段为条件变量重开一次单机制契约试验。本包不打开新数据、不再修提示词。
- 追加（13:40，用户要求）：0-LLM 字段读数已做（`_scratch/dev_tempo_aug_decision_learning/field_readout/`，REPORT §9）：13 批上配方对 None 的 C_B 比值仅 7/13 可分辨；62 个 T 侧字段统计中 0 个在置换检验 p<0.05 下分开害/益批次（最好 11/13，多数基线 9/13，p ≥ 0.108）；E（n=7）无信息。结论：可观察 T 字段中不存在"何时偏离默认"的条件变量，下一包目标应为反馈块设计而非 Slow 契约。
