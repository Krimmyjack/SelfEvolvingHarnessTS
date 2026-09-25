# DEV-BATCH-RESEARCH-SKILL-REVISION：使用反馈后的经验修订

日期：2026-09-14。状态：**已执行完成并收口（2026-09-14 03:31）；原协议合规；见 §13。** 原文其余部分保持派工时原样。
负责人：当前 W 线执行 Opus。用户转发并要求按本任务书执行后，连续完成实现、必要检查、运行和收口，不把常规步骤拆成新的交接。
权威关系：AGENTS.md §5.9 → BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md → 本任务书。历史包保持原协议和读数。

## 1. 唯一科学问题

原有 Source Skill 真实使用后，Slow 读取「原指导—实际行为—合法 C_A/C_B—成本」并修订一次，能否使后续批次的 Fast 比使用旧 Skill 更有用？无 Skill、通用指导和固定菜单分别排除自身搜索、普通提醒和菜单复用的解释。

本包的主要变化是**一次有实际使用反馈的知识修订**。不同时更换 Consumer、算子、指标、候选数、训练重复数或 Fast 交付权；不修改统计门来追求正号。仍只开放 experiment_guidance 一处正文及其适用条件，按一个整体行为机制修改。

本包是开发级修订试验，不是完整 A3/A5 课程或正式 Skill 晋升。旧 Source 和修订候选都保持 CANDIDATE_TEST_ONLY。新旧 H 在每个 Target Job 内冻结；Target 内不得再调 Slow。

### 为什么现在做

前包已经证明 Source 会送达并改变行为。a93 的有用材料由 F0/Generic 同样找到；a89 的提交理由采用了相对 None 的支持，而相对 Fixed 的优势仍不确定。后一项只是 Planner 的开发线索，**不得把这句话或根据 E 得出的正确答案写入 Slow 输入**。由 Slow 自己依据合法记录诊断，也允许它认为没有足够证据修改。

零观察、统一配方、直接复用和 KEEP 都可以合理。不能用观察次数、文本变化、非 identity 数量或更保守来定义成功。

## 2. 接手文件与写入范围

先读：
1. AGENTS.md（尤其 §5.9、§6–9）；
2. docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md；
3. docs/DEV_BATCH_RESEARCH_SOURCE_PROCESS_TASK_2026-09-13.md §12；
4. _scratch/dev_batch_research_source_process/ 的冻结知识、Source 输入、原始轨迹、C_A/C_B 文件和收口状态；
5. methods/ttha/batch_research.py；evaluation/main_protocol_p4/batch_research_runtime.py、batch_research_source_process.py、run_batch_research_roundtrip.py；
6. methods/ttha/batch_base/ 的 spec/context/data/train/commit 及现有控制测试。

复用现役 W 入口、七工具、真实共享训练器、Source census 与 apply_update。允许为本包新增一个薄 study 模块/分支，以及必要的显式 roster 参数传递；不另建 Agent 框架、估值器、聚类平台或证据数据库。

唯一实验输出目录：
`_scratch/dev_batch_research_skill_revision/`

本任务书可以追加实际收口。历史报告、原始包、P/G 工作树、参考仓库和 AGENTS.md 不写；不 git commit/push；0 新 SHA/Hash。已有未提交改动一律保留。

最后的恢复检查已经存在：labels_boundary_violations 在构造账本前拒绝已 finalize/已开标签的恢复。本包直接复用，不再建立独立审计包。旧运行仍不得恢复。

## 3. 冻结 H_old 与合法修订输入

### 3.1 H_old

从前包 frozen_knowledge.json 取 f_source 的原样 Knowledge，保留正文、scope、版本与引用；与前包已保存的实际加载内容对应。不重新调用初始 Source 形成，不人工删改旧卡。

Generic 也从前包冻结 generic_body/对应 Knowledge 原样取，不根据本轮结果优化文案。

### 3.2 本轮 Source 白名单

- 前包的 source_evidence.json / slow_evidence.json 中原已合法的 a55/a65/a75/a85 记录及其来源标记。
- 前包 a89、a93 的 f0、f_generic、f_source、f_program、random 五类分支，共十条实际执行轨迹及合法 C_A、commit 后 C_B。
- 原卡与实际渲染指导、当前工具事实、完整配方、材料诊断、拟合数量、已知 token、未知费用和失败状态。

保留正、负、混合、同材料与失败信息；公共基线去重，不将缓存重复当成新证据。不按哪个案例更支持修订而筛选。

a89 的提前开 E、整个旧包运行中预算变更、费用未知必须以确定性元数据标明。a93 可标本 Job 局部阶段顺序成立；不能把它称为原包无偏的独立验证。这些记录是已曝光开发中的使用反馈，不能在修订后追认旧卡曾通过验证。

### 3.3 防止把 E 答案塞给 Slow

构建器只从白名单轨迹、overview、候选 C_A 与独立 c_b_scores 等已许可字段取值；不把完整 REPORT.md、result.json、e_scores、冻结 E 预测、Planner 对 E 的解释或两个新 Target 的信息交给 Slow。旧 REPORT 可供执行者理解状态，但不能成为自动提炼输入。

保留按事件解析得到的证据引用；复用旧 census 中的去重方式。检查实际发送文本的字段来源，不能仅以「没有 E 字样」证明没有 E 值。

不在提示词中指定「以后选 Fixed」「不要 FreqMask」「多观察」或「把两槽拆开调用」。当前比较对象、风险与后续实验如何组织，由 Slow 从输入中提出。

## 4. 一次自然修订

Slow 输入：H_old + 第3节确定性使用证据 + 公共工具/Consumer/权限说明。
目标说明仅要求它核对原指导意图与执行结果，选择 KEEP 或提出一项有限修订，说明预期改变哪类后续研究决定及证据限制。不得要求它必须修改。

沿用 apply_update 的 KEEP / PROPOSE 契约：
- 唯一允许 hook：experiment_guidance；
- body ≤1200 字符，适用条件使用现有合法 batch 字段；
- const:true 表示明确无条件；null/缺失不等于无条件；
- 正文不使用数据集、Job、实体 ID、组号或绝对日期作为处理条件；证据定位保留在 evidence_refs；
- 可以保留或调整 scope，也可保留直接构造建议；不以新卡必须异质化/必须动态调用为约束；
- rationale 说明所改主要机制，沿用现有字段，不新增复杂 Schema。

一次修订调用，最多一次只针对解析/契约错误的纠正，最多2次逻辑请求。合法 KEEP 不纠正；内容不讨喜、没有指定条件、仍偏好统一方案均不构成重抽理由。

冻结原始响应、H_old、H_new 与实际 diff 后才能打开新 Target 的 T。

**早停出口**：
- KEEP，或仅改版本/引用而行动正文及 scope 不变：NO_REVISION_PROPOSED / NO_ACTIONABLE_EDIT，完成报告，**不启动整套 Target 拟合**，不用同知识双跑制造修订差。
- 形成失败：FORMATION_FAILED，按失败语义收口，不以 H0 代替新版。
- 有合法实质编辑：继续下述全部预定 Target，不因第一个 Target 的 C_A 好坏加卡、调参或换数据。
- 新版未匹配并不自动等于没有 treatment：如果旧版会加载、新版不加载，撤去指导就是实际变化，仍照常测试并报告。

## 5. 两个 Target：同一后续时间、不同训练人口

### 5.1 明确的数据安排变化

本包不再沿第一组32实体继续挑两个相邻比例点。前包 a93 完整工作负载止于24840；若要求后续672小时训练及384小时校准/评价，Electricity剩余部分只够一个与其不重叠的完整作业。

本包因此明确采用 **同一个后续截止点 + 两个不相交的32实体组**。这是当前任务的显式人口配置扩展；不修改历史默认 roster，不作为两次独立时间复现或新数据集泛化。

数据：现有 TSLib Electricity CSV，321实体，T=26304；不是 Monash。沿用现有路径解析，原文件只读。
两组都标 EXPOSED_DEVELOPMENT；第二组不保证历史上从未被其他代码读取，不称 fresh、sealed 或 Natural Final。

### 5.2 冻结 roster

按CSV原始字符串列名排序，排除date；G0取[0:32]，G1取[32:64]。本任务书制定时只读表头核对，未读取新 T/C/E 数值。

G0：
`0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 121, 122, 123, 124, 125, 126`

G1：
`127, 128, 129, 13, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 154, 155`

同组内所有臂用同一人口；候选始终覆盖全部32实体。先固定人口再查资格，资格失败如实停止受影响 Job，不换列补满或筛选有利实体。

### 5.3 冻结时间范围

两个独立 Job ID：a97_g0、a97_g1。两者 t=25512（2019-05-30 02:00:00）。

|内容|范围/起点（行号，左闭右开）|
|---|---|
|T|[24840,25512)，672小时|
|C_A origins|25512、25560|
|C_A targets|[25512,25608)|
|C_B origins|25608、25656|
|C_B targets|[25608,25704)|
|E origins|25704、25752、25800、25848|
|E targets|[25704,25896)|
|完整工作负载结束|25896 < 26304|

G0/G1标签时间相同，但实体不相交。训练数据与模型分别形成；共同日历冲击仍可能相关，不能把两个 Job 当成两个独立时间环境。

新 Target 的 C_B/E 不进入任何 Fast/Slow。两个 Job 之间不更新 H、不传递轨迹或候选；全部 commit 冻结后才开放 C_B/E。不得因其他开发线曾读过分数而把这些分数接入本包生成路径。

### 5.4 roster 的必要实现约束

现有 spec.JobSpec.roster 和 data.load_slice 默认绑定前32列，必须显式传递本包 roster 至观察、父对、scaler、拟合子进程、commit、C_B 和 E。
- 用不可变 Job 配置传参/持久化，旧入口不传时仍用原默认；不靠修改 DATASETS 全局变量或临时 monkeypatch 切换人口。
- 两组 run/job/模型/缓存绑定不同，不能因为同一个 t、material_id、seed 就交叉复用。
- 不新建伪 dataset 名称，也不在 Fast 提示里按组名规定策略。
- 所有实验臂共享同一适配；这是共同仪器变化，不归因于 Skill。
- 只补这项必要参数贯通，不迁移 P/G 树，不建设新 registry/manifest。

## 6. 共同训练与七种交付

训练几何沿用前包：L=192、H=48、每实体433个合法父对、全组13856父对、stride=1、T内每实体scaler、父/子各0.5权重、原样算子实现与R/U语义。
共享MLP 192→128→64→48，AdamW lr=1e-3、weight_decay=1e-4、2000更新、parent batch=64。模型/评分不交给Slow。
冻结训练seeds：20260925、20260926、20260927；所有候选配对初始化与父batch，增强随机流沿用现有规则。三seed不是三条Agent轨迹，也不是集成；实际交付取第一个seed。

|臂|初始知识/控制器|目的|
|---|---|---|
|F0|空H0，现役Fast自主研究|无积累基线|
|F_generic|前包原样通用experiment_guidance|一般研究建议|
|F_old|冻结H_old|修订前基线，主要对照|
|F_new|冻结H_new|反馈修订后的效果|
|Menu|0 LLM；下述固定两候选与确定性提交|固定菜单解释|
|Fixed|全体R-TimeMixup w=0.25|强固定基线|
|None|identity|不增强锚点|

四条Fast走同一个run_job、公共工具和反馈；每Job最多2个新完整候选，各训3seed，初始均有真实评估的None/Fixed。允许直接commit，也允许全体统一或条件化材料。实际工具观察与构造由Fast自主决定；不新增强制roundtrip或2SE采用门。

**Menu 固定行为**：每Job真实构造、共同训练均匀U-TimeMixup w=0.25和均匀FreqMask mu=0.10；与已评估的Fixed、None一起按C_A三seed均值取最小。完全平局的顺序固定为Fixed、None、U-TimeMixup、FreqMask。不带旧卡的解释、风险条款或适用建议，不读取任何历史结果。候选在制定本任务书时已固定，不能读新版或Target后改菜单。它对应前包实际两槽菜单，不是公平随机搜索，也不能代替未来完整论文的随机搜索对照。

公共基线在Job内共享。私有候选仅本臂去别名/复用公共基线，不把他臂候选或反馈暴露给Fast。不同臂偶然构造同材料时，分别按原私有拟合规则执行并记录同材料，不从旧任务复制模型。

## 7. 预算、顺序与缓存

有合法编辑时计划：
`2 Jobs × (6公共基线fit + 4 Fast臂×6fit + Menu 6fit) = 72次拟合`。

|项目|冻结上限|
|---|---|
|物理拟合尝试|74，72计划 + 2次同配置非科学故障重试预留|
|每Fast Job|16逻辑请求、24工具尝试、2新候选、2次可恢复工具输入纠正|
|Slow|1次修订 + 最多1次契约纠正|
|全包逻辑LLM请求|130 = 8条Fast×16 + Slow 2|
|全包HTTP尝试|260；每逻辑请求最多1次同请求传输重传|
|已返回usage停止边界|3,000,000 token；未知费用另列，规则见下|
|实验活跃墙钟|10,800秒，暂停时间不重置；实现/分析时间另报|
|单fit进程|300秒，且不得超过全包剩余时间|
|LLM身份|现授权中转cpa-grok-4.6，实际返回grok-4.6-build；不换模型|

费用预留不是后端保证上界。出现失败尝试或返回响应缺usage，保留未知状态；按现役规则完成至多一次预先允许的同请求重传，之后拒绝新逻辑收费调用。执行者不得自行添加--accept-unknown-usage来豁免本任务预算；需要用户明确接受未知费用才可续跑，并保留原账本。没有该授权时扣留标签、保存现场即可，不收费探活。

执行顺序固定：
- a97_g0：F0 → F_generic → F_old → F_new → Menu；
- a97_g1：Menu → F_new → F_old → F_generic → F0。
两个Job在首次Target调用前一并冻结配置。Scope不匹配不改顺序、不换Job；节余不兑换新候选/新Source/新seed。

本包计新增完整成本；H_old形成、历史Source资产、前包使用反馈的已发生费用另列，不称历史经验免费。未知历史费用继续UNKNOWN。仅Fast token下降不能写成总成本下降。

## 8. 冻结、暂停与不可重跑边界

1. 实现、必要smoke、Source白名单、Menu、Generic、人口/时间/seed/预算先冻结；此时不读新Target数值。
2. Slow只处理第3节证据；H_old/H_new冻结。KEEP等无干预出口在此收口。
3. 只打开两个Target的T，做固定人口资格、roster/数组绑定及scope检查；不按T特征更改已冻结的新旧知识或Target选择。
4. 公共基线与各臂真实运行，Fast仅取得自己的C_A；每个commit绑定已拟合完整方案和预定交付seed。
5. 所有计划臂完成或明确终止后冻结所有commit。随后对已完整拟合候选打开C_B，只作延迟诊断，本包不再调用Slow。
6. 全部合格候选的E预测存盘并建立全局屏障，再统一评分E。观察到E后不得修改候选、知识或参数。

方法级解析/契约失败只记该方法不完整，其余健康臂按预算继续；不能兜底Fixed冒充COMPLETE。整体预算/账户/信息墙问题停止全包并扣留标签。
恢复复用冻结知识和已完成响应/模型，只从未发送的下一次调用继续；剩余时间变化如实记录，不称整个请求逐字相同。
显式finalize表示终止剩余实验，之后禁止任何方法恢复；已有C_B/E标记也禁止恢复。不得为了凑满72fit反复重启。

## 9. 必要检查，不重开旧审计

一个现有smoke内覆盖本次新增差集：
- 显式G0 roster与旧默认第一组在相同合法T上的父对/scaler/材料数值一致，0拟合；G1确实读取其32列，跨组缓存/模型绑定拒绝。
- 完整roster和时间配置往返至独立拟合/评分子进程；用已有数值路径/小型合成fixture检查配置传递，不新增真实对齐fit。
- Slow收到H_old及合法使用证据，E/Target信息不进入实际请求；PROPOSE仅能覆盖experiment_guidance，KEEP不产生收费Target假对照。
- F0/Generic/Old/New渲染各自知识，Menu没有历史解释文本；scope状态与真正发送内容一致。
- 复用已通过的暂停/恢复/终止检查；只有本包修改了相关路径才补针对性验证，不重新扫全部历史包。

若roster路径仍需改动，修好后再收费。若需要更换Consumer、额外动作、调整预算/时间人口，先提出具体差异，不自行执行新科学设置。

## 10. 结果与判读

**主读数**：每Job，已commit方案的三seed E均值之差
`Δ_revision = loss(F_old) - loss(F_new)`，正值表示修订更好。
同时列每seed原值、配对差、seed SE、首seed实际单模型交付；不能在首seed与均值间择优。
辅助比较：F0/Generic/Menu/Fixed/None减F_new；C_A、C_B、E分列。主目标始终全32实体normalized MSE宏均值，不删难例、不用中位数替换。
两个Job分别报告；等权汇总只作描述，不能把2组×3seed包装成6次独立学习复现。

过程检查只回答：
- 原指导提出什么，Slow依据哪些实际事件修改了什么；预期改变哪个决定？
- 新版在当前批次何时加载/未加载；与旧版的具体观察、候选、对照、停止或commit有何不同？
- 材料是否改变、是否真实共同训练；若相同，效果差是否只是重复调用差异？
- 新版相对Menu/Generic多出的收益或完整成本变化是什么？

|观察|允许的结论/下一步|
|---|---|
|KEEP/无行动编辑/形成失败|本轮无修订treatment，记录原因；不重抽、不补跑Target凑结果|
|行为变了但效用未改善|该次修订尚无净效用；不因格式更好而晋升|
|只胜旧卡，与Generic/Menu相当|纠正旧指导有局部价值，额外研究技能增量未成立|
|材料只是一个新的固定菜单/固定程序并有收益|可报告构造/程序复用收益；不能仅凭这点宣称研究过程技能或Pattern适应成立|
|新卡在后续两组均有方向一致、具有实际量级的增益，并优于F0/Generic/Menu，或有可解释的完整成本优势|值得进入多轨迹与完整F/A3/A5短课程；仍非本包证明泛化|
|只在一个组有利/符号混合/效应小于当前分辨率|限定该配置并保留不确定；不改门、追加seed直到变正|

不新增小n显著性或2SE晋升门。三个seed仅测固定候选的训练随机性，一个Slow输出和每臂每Job一条Fast轨迹不能给出LLM更新效果的可靠方差。质量差异不明确也不等于已证明等效；成本更低但质量更差应报告权衡。

候选集合内的E最优与实际commit差只作回顾性分解；不得拼不同模型的逐实体预测，也不得据它修正本次交付。
新修订如果相对Menu没有增量，不再启动同数据、同E上的第三版文案包。本包至多提出一个后续方法假设，完成即停止。

## 11. 交付与回执

一个主REPORT.md，配result.json、frozen_config.json（含两组roster/日期/权限）、合法Source输入、Slow原始响应、H_old/H_new、逐分支请求/轨迹/材料/模型/冻结预测及成本账本。文件名可沿用现有约定，不为清单额外造平台。
主报告首页回答：
1. 原经验经实际使用反馈后修订了什么，还是KEEP？
2. 修订造成了什么具体研究决定和材料变化？
3. 两个新批次中，旧/新/F0/Generic/Menu/Fixed/None的完整交付效果与成本如何？
4. 当前最有证据支持的解释及唯一下一项工作是什么？

执行完成、协议合规、数值/绑定检查、失败/未知费用分别记录。不把测试通过当成方法有效。
在本任务书末尾追加真实完成状态与主报告位置；不要更新AGENTS、晋升canonical Skill、重跑旧包或自动展开A3/A5。

## 12. 转发执行指令

请按本任务书连续完成实现、必要检查、一次自然修订、预定对照和收口。先完成公共roster参数贯通再收费；原Skill和已有使用证据复用，不重训Source。KEEP是合法终点；有效修订才进入两个Target，不能见负数换题。全部数据按development解释，0新Hash，不动其他执行线，不自行接受未知费用或解除E边界。

## 13. 实际完成状态（2026-09-14，追加式）

主报告：`_scratch/dev_batch_research_skill_revision/REPORT.md`；机器数据 `result.json`、`audit.json`、`frozen_config.json`、`slow_result.json`（含 H_old/H_new/diff）、`usage_evidence.json`、`slow_evidence.json`、各分支轨迹/材料/模型/冻结预测、`budget.json`。

- **执行完成**：10/10 方法分支 COMPLETE，72/72 计划拟合成功（0 失败、0 重试），1 次 Slow 修订（PROPOSE，0 契约纠正），27 次模型请求 = 27 次 HTTP（0 传输故障、0 未知 usage），已知 token 1,177,186，账本活跃墙钟 4,191 s。
- **原协议合规**：修订冻结（02:23:02）先于任何 Target T 打开（02:23:15）；全部 commit 冻结（03:26:56）后开 C_B；全局 E 屏障（03:30:51）后统一评分；无暂停/恢复/finalize，未使用 `--accept-unknown-usage`。`audit.json`：120 份冻结 E 预测按显式人口复算最大差 1.4e−14，roster/seed/commit 绑定、跨臂同基线、跨组不同、每次 Fast 请求只加载本臂冻结正文、Slow 请求不含 Target/E 标识、Menu 0 请求、usage 与账本一致。
- **修订内容**：Slow 依据 a89/a93 两条 F_source 轨迹及其 commit 后 C_B（加 a65/a85 三条历史事件）删去"两槽必用 U+FreqMask"与被证伪的"R 有帮助时 FreqMask≈identity"，加入"C_A 胜 identity 不能确定 FreqMask 优于 R""符号混合也算不确定""均匀方案/复用/停止都合法"；scope 仍 const:true。
- **结果**：F_new 两组 3/3 加载；实验组织改变（分槽推进、6 次 compare、commit 理由对准在位方案），但候选材料与 F_old/Menu 完全相同（均匀 U 0.25 + 均匀 FreqMask 0.1），交付相同（g0 Fixed、g1 None）。Δ_revision = loss(F_old) − loss(F_new) 在 C_A/C_B/E 上两组均精确为 0；对 F0/Menu 也为 0；对 Generic g0 +0.0016（SE 0.043）、g1 0；对 None g0 +0.172（3/3）、g1 0；对 Fixed g0 0、g1 −0.0008。成本：F_new 比 F_old 多 84,676 Fast token，另加修订 100,354。判读：**行为变了但效用未改善——该次修订尚无净效用，不晋升**；修订针对的决策情形（非在位方案以小幅、符号混合的 C_A 优势胜出）在两组上均未出现。H_old/H_new 保持 CANDIDATE_TEST_ONLY。
- **唯一后续假设**（未启动）：用 0 LLM、只读 T+C_A 的固定材料共训在同一截止点其余 32 实体组上筛出"在位方案不是 C_A 最低且优势在 2 seed_se 内或符号混合"的批次，再在那里比较 F_old/F_new（含 Generic/Menu/Fixed/None）；找不到则转向让 Fast 产生在位方案之外的 C_A-更好候选，不再修卡。
- **代码**：新增 `evaluation/main_protocol_p4/batch_research_skill_revision.py`；`run_batch_research_roundtrip.py --study skill_revision`（Menu 臂、seeds/roster 贯通、先开两组 T、KEEP 早停、smoke）；`batch_base` 的 spec/data/context/train/commit 增加可选 `roster`（默认路径不变，`resolve_job` 接受 `a97_g1`，人口随 scaler 冻结）。未 git commit、0 新 SHA、未写 AGENTS/旧包/P-G 树。

