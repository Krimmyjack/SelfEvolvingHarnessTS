# DEV-TEMPO-AUG-DOMAIN-SKILL-OPTIMIZE
日期：2026-09-19（分发前补充：Source 决策摘要与提交机制读数；预算和批次不变）  

> **状态更新：旧包已执行完成并停止；后续规格已被替代。** 实际结果为 NO_REPLAY_INCREMENT / GUARD_LEARNED_BUT_UNTRIGGERED，见 [_scratch/dev_tempo_aug_domain_skill_optimize/REPORT.md](../_scratch/dev_tempo_aug_domain_skill_optimize/REPORT.md) 及本任务书完成回执。保留原执行规格、预算、实际用量和 amendment，不按结果追改原预算。下一包使用 [DEV-DOMAIN-AUG-ENTITY-SPLIT](DEV_DOMAIN_AUG_ENTITY_SPLIT_TASK_2026-09-19.md)，不据本旧稿重跑或自动追加实验。

执行负责人：Opus（项目开发者；实验内 Fast/Slow 沿用已验证模型）  
性质：开发优化包，EXPOSED_DEVELOPMENT_REPLAY；不是论文完整复现或最终泛化验收。  
交付位置：`_scratch/dev_tempo_aug_domain_skill_optimize/`。

## 1. 目标与本次变化

把已有增强研究变成一张可以实际帮助 RD02 Fast 的域 Skill：从较早批次的研究过程和真实后期效果学习，在开发批上试用、修订一次，冻结后比较有卡和无卡的交付。

这次主要改变学习反馈：**允许 Slow 学习较早批次已经曝光的 E；在 Practice 和 Select 中，把本包 commit 后统一取得的 E 明确作为开发反馈使用。** 当前批 Fast 仍只能看合法 T、当前 C_A 和冻结 Skill。最后两个复验批次的结果不回流本包。

NoMixRecipe 是全部方法共享的强工具，也是保留的固定对照。无需先证明无卡 Fast 超过它，才允许学习 Skill；也不要求每张卡必须构造复杂新方案。主要问题是“学得的指导相对同工具、同预算、无指导的 Fast 改善多少”，固定对照帮助识别改善是否只是学会了一个常用配方。

已知依据：上一包 Fast 使用了编辑工具，在四批上相对 NoMix 平均 +1.40 pp，结果不稳定；其评估集中于统一单组件关闭。不能据此断言条件化无用，也不能预先指定去掉 calendar、保留 shock/censor 就是正确答案。本包给 Slow 原始数值与冲突，让它提出处理指导。

本次复核补充（用于设计解释，不整体复制到实验模型提示词）：
- 四公共参照的滚动重放已完成，见 temporal_feedback_replay/REPORT.md：历史策略对当前 C_A 选择 +9.58 pp，但与固定 NoMix 全部同交付。不重复同一实验，也不预设历史反馈一定带来动态选择增量。
- 编辑包的 7.98 − 6.58 = 1.40 pp 是相对事后 E 池最优的描述性分解；约 82.5% 未兑现。这个 oracle 有事后选择偏差，不是合法选择器保证能拿到的收益。L7 一批贡献约 81.8% 的选择损失；不能据总量断言每批都由同一机制失败，或把瓶颈强度说成精确“四倍”。
- 5/7 是跨四批平均的组件消融符号比较；它提示当前反馈与后期效果存在冲突，不证明固定方向的偏置机制。C_A/E 都是原有 48 小时预测，区别在评价日历时段，不能把它写成长预测步长实验。
- 4/4 commit 等于 C_A argmin 说明原 Fast 沿用了这个选择规则，不能证明其他提示/经验指导没有改变决策的可能。实际正差照实保留，单批集中不能直接判定“纯运气”或稳定能力。

本任务书交给执行者后，在以下范围内连续完成实现、运行和收口。普通接线、输入尺寸、展示和可恢复参数错误自行处理；不逐阶段重新等待 Planner。未开始执行前完成配置冻结。

## 2. 保留的实验主体

- 数据：RD02 既有 12 站 roster、合法窗口和评分覆盖规则。
- Consumer：现役共享 MLP，2000 updates，原优化器、目标与 3 个配对训练 seed，均从 recipe_edit_pilot 的冻结配置继承；不改 Consumer。
- 工具：现有七工具和 EditAdapter；四公共参照为 None、FixedMixup、P_AmpResample、P_NoMixRecipe。
- 新方案：现有 P_NoMixRecipe_Edit，最多关闭两个原语；允许统一赋值或最多 8 条合法实体条件规则。沿用已有参数与特征谓词，不新增算子、强度、概率或特征学习器。
- 材料几何：沿用整段历史训练窗口 192+48 的 parent/child 处理、0.5/0.5 损失。仅历史 child 训练目标可变，评价真值不变。
- RNG：沿用 tempo_source_v2 / recipe_edit_v1、同一 Job index 和实体的 program-52 父流；关闭组件不重抽、不调整其他步骤随机状态。
- Fast 系统提示、工具说明、4 新方案 / 16 请求 / 24 工具调用上限全臂相同。仅 experiment_guidance 是否加载及内容不同。
- 模型入口、返回身份与环境继承上一包：cpa-grok-4.6，实际返回 grok-4.6-build；不可静默换模型。Opus 不充当实验内 Slow。
- 不做画像 Router；同域加载。无强制分组、强制三步、强制用满名额、强制离开默认，也无新“禁止分组”规则。

## 3. 数据分工与可见性

| 阶段 | Job | t / 既有 job_index | 作用 |
|---|---|---|---|
| Source | S1、S2、V1、T2、T4、Q2 | 3360、4560、5760、8160、10560、12960；沿用原 index | 原无卡过程及已合法打开的分数 |
| Source | L1、L2、L3 | 15360/7、16560/8、18960/9 | 原无卡过程；L1/L3 另加编辑包证据 |
| Practice | L4 | 20160/10 | 两张卡实际试用；之后允许一次修订 |
| Select | L5 | 21360/11 | 无卡、父卡、子卡对比并冻结最终卡 |
| Replay | L6、L7 | 22560/12、24480/13 | 冻结卡与无卡配对复验 |

表中简称均加 RD02_ 前缀。没有新切点、新人口或新数据授权；row >= 24864 保持封存。

Source 共 9 个时间批次，其中只有 L1/L3 具有这次局部编辑空间的直接研究证据。不得把窗口数、材料数、3 个 seed 或重复轨迹当作额外独立案例。

所有这些批次已在历史研究中曝光。本包可以控制本轮模型的读取顺序，不能恢复研究者未见过结果的状态；报告始终写“开发复验”。

### 3.1 Source 白名单

设 W = `_scratch/dev_tempo_aug_workflow_skill`，P = `_scratch/dev_tempo_aug_recipe_edit_pilot/pilot`。

读取：
- W/source/RD02_{S1,S2,V1,T2}_no_skill；
- W/select/RD02_{T4,Q2}_no_skill；
- W/target/RD02_{L1,L2,L3}_f_no_skill；
- P/RD02_{L1,L3}_{f_edit,random_edit,ablation}；
- `_scratch/dev_tempo_aug_temporal_feedback_replay/historical_late/RD02_{V1,T2,T4,Q2}/` 中已打开的历史后期评分。

分支轨迹在 branch/trace.jsonl，标签通常在 branch/job/c_b_scores.json、e_scores.json；以实际元数据绑定为准。缺失的 E 记 unavailable，不为补表自动重训或打开额外标签。

普查保留：实际观察与材料检查、构造及合法失败、每个真实评估方案的完整赋值、逐 seed C_A/C_B/E（已有者）、commit、理由、成本。明确区分完整统一配方、条件化赋值及受评估池限制的选择。

排除：L4 及以后的旧结果、编辑包 L5/L7 的结果、汇总了未来批次的 REPORT/记忆结论、历史旧卡中的禁令原文。不把旧卡当作必须继承的母版。**本任务书全文也不直接拼入 Slow/Fast 请求**；从其中提取权限与任务契约，实验数值只能由本节白名单生成。

相同 TA 编号在不同目录可能对应不同材料。普查按来源路径、Job、完整赋值、profile 和 seed 语义核对；不按局部 material_id 跨包合并，也不建立新哈希系统。

### 3.2 标签角色

- Slow 形成：只读上面 Source 白名单，包括已曝光的后期 E。
- Fast：只读本 Job 的 T、工具结果、C_A 和本阶段冻结 Skill；不读其他 Job 原始轨迹或本 Job C_B/E。
- Practice：两卡均 commit 后统一 C_B，再冻结 E 预测并评分；这些 E 可给本次修订 Slow。
- Select：全部分支 commit 后统一评分；E 只用于选卡和固定对照，不再触发第二次修订。
- Replay：L6/L7 所有分支输出冻结后，统一 C_B → E 预测冻结 → E 评分；结果仅用于收口。
- 不将 E 全局改名为 C_A 绕过权限。新 study 显式按阶段/Job 白名单开放 development_late；旧包的 E 拒绝行为保持原样。

## 4. 学习目标与 Skill 契约

初始形成用两次 Slow 调用，各产出一张候选 W1/W2。第二次可见第一张卡并提出不同的研究策略；这不是两次独立统计重复。不要额外调用模型挑出好看的生成结果。

输入明确写：

> 你的目标是在后续同域批次改善最终下游预测，并节省无价值实验。历史 C_A 解释 Fast 当时为何选择；历史后期分数用于判断哪种指导值得复用、修改或保留为假说。二者冲突时要保留冲突，不能把 C_A 领先等同于后期更好，也不能先验规定忽略全部 C_A。
>
> 从实际可见的数据问题和动作—效果证据提出可执行 Workflow：需要查什么、何时构造什么有区分力的完整方案、如何使用反馈、何时停止以及如何提交。可以使用已有强配方；也可以提出单关闭、双关闭、条件化处理或不增强。未尝试的动作不等于有害。
>
> 单实体预测变化不证明其自身材料的独立贡献；这不禁止按观测分组，分组方案的效用仍由完整共享模型评估。
>
> 不确定性可以成为追加检查、有限试验或保留基线的理由。一次“不分胜负”不自动变成永久的算子禁令。历史结论只能支持其实际覆盖的处理与情境。
>
> 请特别写清最终提交策略：当前 C_A 在决策中承担什么角色，历史学得的偏好何时保留、何时让位于当前证据；若没有依据偏离 C_A 最小值，也可明确保留该规则。你可以建议提交已评估但 C_A 不是最低的方案，不能使用当前批的隐藏后期结果。不要把“必须显著占优”“必须 3/3”当作未经验证的默认答案。

输出用现有 Markdown/body 载体，不另造框架。每张卡正文 <=6000 字符，scope 为已知域内 const:true，包含：
1. 可执行 Workflow：观察、构造、试验、commit/stop；
2. 可选 Principles：条件、动作、理由和例外；
3. 对主要规则的简短 support/counter 来源以及 supported / hypothesis / unresolved 状态；
4. 希望相对无卡改变的一个具体决定，尤其是提交规则；写清哪些条件由当前工具计算，哪些属于学得的历史偏好。

不得写 Job ID、站点 ID 或原点到答案的查表，不手工给阈值拟合未来结果。已有证据可形成稳定偏好，但要能区分“域内常用方案”和“何时偏离”的条件证据。

不要求固定配方通过显著性门后才允许卡试用；不要求所有卡必须体现复杂分支。允许合法 KEEP/没有修订，但不得用 fixture 或 Generic 冒充学得卡。若形成始终没有合法卡，在已给格式纠错次数后停止并报告原因。

## 5. 连续运行流程

### A. 最小接线与冻结

先在 Source 普查内完成一个小型决策摘要，开发耗时上限 20 分钟，0 LLM、0 拟合、0 推理、0 新标签，不另开诊断包：
1. 列出“规范化程序 × Source Job”的已评分覆盖表。统一关闭同一组件可以作为同一策略跨批比较；不同条件谓词、不同关闭集合或不同 profile 不能因为名字相似合并。缺失格保留缺失，不用该候选在其他批的成绩代填。
2. 每个 Source Job 的每条原 Fast 轨迹，仅在其自身已评估池内列 commit、影子 C_A argmin、事后 E 最优与差距；有 E 才算。区分新增编辑提供的机会与已有公共方案的选择机会。oracle 明确标“事后诊断”，不当成部署可得信号。
3. 扩展的时间检查只用 **L1→L3** 这一个完整配对：四公共参照加七统一单关闭，按两端均有三 seed 分数的程序交集。以 L1 后期 E/None 均值选一个历史推荐，去 L3 已有对应材料中检查，与 L3 的当前 C_A 选择、NoMix 对照。它只有一个历史批和一次跨批检查，只作 Slow 的案例证据，不能证明滚动平均或动态匹配有效。若实际没有完整绑定，记不可算并继续主流程。
4. 已完成的四公共历史重放只引用 Source 白名单内对应行；不把 L4–L7 汇总混入形成。其他私有/条件化材料保留为原 Job 的案例，不扩展成虚构的跨批面板。

摘要并入同一 census，保留来源、覆盖数和冲突，不直接送 Fast。**这项读数不设进入试卡的效用门，也不搜索多个 k、均值/中位数、阈值或按未来 E 分桶的选择器。** 超时保留已取得的摘要，继续下面已授权流程。

复用 recipe_edit_pilot 的 adapter、材料缓存和 worker；复用 workflow_learning_loop 的 Skill 载体、Slow 调用、阶段控制。新增一个 study 入口即可。

必须处理的接线差集：
- recipe_edit_pilot 的 knowledge、臂名和随机/消融启动当前有固定配置；新 study 按配置加载卡并关闭随机、七消融。
- 新 Fast 臂名仍接受相同代理、账本和参数校验；不能只保护名叫 f_edit 的分支。
- 全局 no-E 检查不能直接删除；本 study 增加 §3 的白名单语义。
- 保留并核实已有 commit 权限：可提交自身任一已评估方案或公共参照，运行时不得改写为 C_A argmin。脚本 smoke 以一个非 C_A 最小方案做合法提交检查；不要求真实 Agent 必须偏离 argmin。
- 分支模型/材料/评分绑定自己的目录，读取实际评估集合；同一方案可共享物理缓存，但不得共享 Agent 会话历史。

一个无 LLM smoke 覆盖新增知识加载、阶段标签权限及一个编辑方案执行；若需真实接线，最多 3 拟合，复用既有 RD02_T1 接线 Job，C_B only。禁止 smoke 启动真实 Fast/Slow worker。无需重复已通过的全部逐位 parity 或故障矩阵。

在首个付费调用前冻结本包配置、实际 Fast system、Slow 模板、数据分工和预算。压缩仅删除重复/冗余文本，保留候选身份、动作、关键观察及分数；按真实请求字节检查预留额度，留出规定的格式纠错空间。

### B. 两卡形成及 Practice

按 §4 生成 W1/W2，然后在 L4 各跑一次 Fast。两卡拥有相同公共参照和 4 个新方案名额。不额外跑无卡 Practice，也不重跑 Source。

两卡均提交后，按 §3 开 C_B/E。用实际提交的 E/None 三 seed 均值选一个修订父卡；绝对平局 <=1e-12 时依次按较少新评估、较少 token、W1 顺序破同。

### C. 一次反馈修订

一次 Slow 读取 Source、两个 Practice 的事实、父卡全文与原来的预测，输出一个子卡或 KEEP。聚焦一个主要决策机制，允许改变观察、候选优先级、比较/提交规则中的一处；无需为了产生版本而改动。

明确给出“哪个实际反馈支持改动、预期在哪类合法输入下改变决定”。未触发的分支保持未检验，不强迫运行时触发。KEEP 是父卡别名，不重复跑。

### D. Select 与冻结

L5 跑 F0（无卡）、W1、W2，以及非别名子卡，最多 4 条 Fast；每条预算相同。全部提交后统一开 C_B/E。

卡片分数：
J(W) = mean_seed E(L5, actual_commit_W) / mean_seed E(L5, None)。

在卡片候选中选 J 最小者为 W*；平局按较少新评估、较少 token、原始卡顺序、子卡最后。固定对照或 F0 胜出不会令 W* 消失；仍完成已计划的有卡/无卡比较，并如实报告 Select 差距。

同时形成 Fixed_dev：从本阶段实际评估的公共方案与**统一**编辑方案中，按同一 E/None 选一个固定程序；平局公共顺序优先，其余按规范化程序文本顺序。复制其程序定义，不能复制 L5 模型、实体 ID 或输出。条件化方案不进入这个静态对照。

W*、Fixed_dev 和两 Replay Job 配置在 Replay 首次模型调用前冻结。只允许这一轮选优；不据 L5 再改卡。

### E. 两批冻结开发复验

L6/L7 各跑 F0 与 F_skill(W*)，共 4 条 Fast。L6 先 F0、L7 先 F_skill；同批资料与预算相同。

公共 None、NoMix 等直接报告；Fixed_dev 在两个 Job 按同一固定程序重新生成正确材料，必要时各 3 拟合。它可以与已有公共方案别名，不重复训练。

不启动新 Random、Generic 或消融全表。已有随机结果只可作历史背景，不能拼接成这包公平配对臂。Menu_CA 可由本包四公共分数零成本重放，标成影子读数。

L6/L7 所有 commit 冻结后统一评分，收口并停止。无自动下一轮修订，无按结果替换批次或增加 seed。

## 6. 缓存与执行资源

优先复用物理材料、模型和 C_A，不复用旧 Fast 会话来假冒本包新对照；最多重跑 10 条 Fast。

公共缓存候选：
- L4：workflow_skill/target/RD02_L4_common/RD02_L4；
- L5：recipe_edit_pilot/pilot/RD02_L5_common/RD02_L5；
- L6：decision_learning/test/RD02_L6_common/RD02_L6；
- L7：recipe_edit_pilot/pilot/RD02_L7_common/RD02_L7。
以上均位于对应 _scratch 包下；缺失时可用同 Job 原父包 common，先核现有语义绑定。不得将旧 C_B/E 分数随物理缓存带入 Fast 请求。

| 阶段 | 物理拟合上界 | LLM 请求上界 | token 上界 |
|---|---:|---:|---:|
| 接线 | 3 | 0 | 0 |
| 两卡形成（含一次格式纠错） | 0 | 3 | 800,000 |
| Practice：2 Fast | 36 | 32 | 800,000 |
| 一次修订（含一次格式纠错） | 0 | 2 | 500,000 |
| Select：最多 4 Fast | 60 | 64 | 1,600,000 |
| Replay：4 Fast + Fixed_dev | 78 | 64 | 1,600,000 |
| 同配置技术拟合重试预留 | 3 | 0 | 0 |
| **整包上限** | **180** | **165** | **5,300,000** |

上界按每 Job 4 公共×3 seed、每条 Fast 4 新方案×3 seed 计算，缓存命中不占物理拟合但仍占该臂逻辑评估槽。失败尝试也计入整包上限。

单 Fast 400K tokens、16 请求、24 工具、4 新方案，输出帽 12K；不因臂表现改变配额。整包 HTTP 上限 169，最多 4 次额外传输尝试、同请求最多额外一次；有 UNKNOWN usage 则暂停新增付费调用并记录，不能把未知当零。无未知费用的可恢复中断可在本包原预算内恢复，不重生成已冻结卡。

使用一个持久账本，阶段结余不扩大后续阶段配额。拟合/请求/HTTP/token 同时守卫；Slow 的一次格式纠错只修合约，不引导效果方向。账号/权限错误不盲重试。

付费阶段墙钟上限 4 小时，编码另计；按既有速度预计运行约 1–3 小时，缓存命中和代理速度影响较大，不能保证。达到预算时如实留不完整格，不加预算、不改判据追正号。

## 7. 主读数与结果使用

收益统一为：
Delta_j(A,B) = 100 × [mean_s E_j(A) - mean_s E_j(B)] / mean_s E_j(None)。
正号表示 B 更好；单位是各 Job 的 None 均值百分点，不是相对 A 的百分比改善。

报告：
1. Replay 的 F_skill 对 F0：逐 Job、逐 seed 向量、两 Job 等权均值及最大伤害。
2. F_skill 对 NoMix、Fixed_dev、None；四公共参照 E 原值保留，必要时加 Menu_CA 影子结果。
3. Practice 到 Select 的规则修订、实际触发、材料/commit 变化。修订效用只在 Select 读，因 Replay 未跑全部父卡，不声称隔离出了后续“进化增量”。
4. 观察、新方案、逻辑/物理拟合、token 与一次性形成成本。记录条件化是否实际使用；未使用不等于其没有价值。
5. 每条新 Fast 的实际 commit 与其自身池的影子 C_A argmin 是否相同；两者在本阶段已评分 E 上的配对差。这样可拆开“卡改变了候选供给”与“同一池中改变了提交”。再列自身池的事后可用收益、选择损失和贡献最大的 Job；不把 oracle 差距直接写成可实现提升。影子选择只用于读数，不覆写真实 commit。

保留已有配对 SE/t95 计算，但不把显著性当开发试用或修订的前置门。两个 Job、三个 seed 只能提供开发线索；训练 seed 不代替时间批次。均值为正但单批集中时直接说明，不新增统计诊断包。

若卡改善无卡，即使仍不及 NoMix，也可以继续开发该学习机制；若胜无卡而与 Fixed_dev 同材料，收益记为学得稳定方案偏好，不称自适应研究增量。若表现更差，保存具体失误并提出下一项方法改动，不自动重复提示词搜索。若仍仅省成本，分别报告收益和成本，不能把“未显著”解释成“严格相等”。

## 8. 交付与完成条件

一个入口：
`evaluation/main_protocol_p4/batch_research_tempo_aug_domain_skill_optimize.py`。

一个输出目录，保留：
- REPORT.md：首页直接展示 W1/W2/子卡/W* 的中文可读内容或准确摘要、一个“观察→指导→动作→效果”实例，以及主结果；
- result.json / tables.md：真实值和缺失状态；
- frozen_config.json、budget.json、必要的阶段日志、实际请求与卡片；
- METHOD.md 可并入 REPORT，避免重复文档。

任务书末尾追加实际收口：阶段完成情况、选中卡、收益/伤害/成本、故障及剩余不确定性。共享改动 opt-in；原包只读。SHA/Hash 预算 0，不 git commit，不改 AGENTS，不修改参考仓库或其他执行线。

本包止于上述结果交付，不启动第二域、TSFM、封存区或额外分析。

## 9. 后续大致安排（规划，不属于本包运行授权）

1. 本包：在 RD02 完成“有后期监督的形成→试用→修订→冻结开发复验”，得到能展示且可继续优化的实际 Skill。
2. 随后：复用同一学习入口补第二域，使用该域独立合法开发证据形成自己的 Skill，比较域卡、共享指导和无卡。先盘点已有证据，不重新建 Harness；无需预先保证每域都打赢 NoMix。
3. 最后：确定完整方法与主要对照后，对真正未使用的数据做一次冻结验证。数据开放范围、预算另行明确；不能把当前 L6/L7 包装成那次验收。

这三步保留 per-domain 学习的项目主体；本包只解决第一步。

## 10. 实际执行回执（执行者完成后追加）

待执行。

### 10.1 执行回执（Opus，2026-09-19 21:26 完成，追加式）

- 阶段：接线 PASS（RD02_T1，3 拟合，fixture 卡进入请求，非 C_A argmin 提交未被改写，C_B only）→ 形成 2 次 Slow（W1「配方族内按 C_A 选，公共参照不覆盖」、W2「固定筛 calendar-off/conv-off」）→ Practice L4 两卡均交付 Edit[-tp_random_conv]（E 对预设 +10.05 pp，对影子 C_A argmin=None +36.1 pp；同分，按较少新评估选 W1 为父）→ 修订 Slow **KEEP**（父卡在 L4 被完整执行，commit 既是族内 C_A 最优也是自身池 E 最低）→ Select L5：f0 交付 Edit[-tp_shock]（J 0.606），W1/W2 均交付预设（J 0.5696，对 f0 +3.63 pp；影子 argmin=P_AmpResample 被卡规则拒绝，避免 −28.1 pp）→ **W\* = W2**（同分，较少新评估）；**Fixed_dev = Edit[-tp_calendar]**（L5 E/None 0.558）→ Replay L6/L7。
- 主结果（Replay）：F_skill − F0 = **−0.05 pp**（L6 −0.10，L7 0.00；0/1/1；最大伤害 −0.10）；F_skill − NoMix = 0（两批同交付预设）；F_skill − Fixed_dev = −1.12 pp（L6 +0.52，L7 −2.77）；F_skill − None = −0.87（L6 +7.43，L7 −9.18）。卡的"公共不覆盖"守卫在 L6/L7 未触发（C_A argmin 已在族内），影子 argmin = commit 4/4。判读：`NO_REPLAY_INCREMENT / GUARD_LEARNED_BUT_UNTRIGGERED`。条件化规则 9 条 Fast 中 0 次使用。
- 费用：物理拟合 30（0 失败 0 重试；帽 180），缓存命中 192；请求 59（Slow 3 + Fast 56；帽 165），HTTP 59，token in/out 1,993,203 / 85,658（帽 5.3M），未知 0；新增 SHA 0，git commit 0。账本墙钟 18:24→21:26（含 18:40→20:54 空转）；真实付费运行约 55 min。
- 故障与修正：修订 Slow 载荷 306 KB 超冻结预留（阶段帽 已用 574k + 500k）→ 驱动 RuntimeError 退出；Source 修订输入改为"材料数字 + commit 摘要"（144 KB）续跑，未丢失/重复付费调用；formation 载荷用 minimal 层（265 KB）按逐次调用预留发送。空转 2 h 后按用户指示"预算超出不作停止条件"，把墙钟帽 4 h→8 h（`amendment_1_wall_cap.json`；frozen_config 原文未改）。
- 实现：`evaluation/main_protocol_p4/batch_research_tempo_aug_domain_skill_optimize.py`（Source 白名单普查含 development_late、决策摘要、两次形成/一次修订 Slow、Practice/Select/Replay 阶段、Fixed_dev 视图、读数）；复用 recipe_edit_pilot 的 EditAdapter/缓存/worker 与 learning_loop 的 Skill 载体、Slow 调用、预留检查。输出根 `_scratch/dev_tempo_aug_domain_skill_optimize/`（REPORT.md §0–§4、tables.md、result.json、frozen_config.json、budget.json、evidence/、formation/、revision/、practice/、select/、replay/）。本包停止，未启动第二域/TSFM/封存区。
