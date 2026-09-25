# DEV-BATCH-RESEARCH-PAIRED-REFINEMENT：固定 Fast 的反馈驱动材料修订

日期：2026-09-14。状态：READY_FOR_EXECUTOR；任务书已定稿，根 Agent 未启动实验。
主执行者：接收本任务的 Opus。Planner：Astra。
权威关系：项目 AGENTS.md §5.9 → BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md §17 → 本任务书。

用户转发并要求执行本任务后，执行者在下列范围内连续完成实现、必要 smoke、真实运行、统一评分和报告。普通实现选择自行处理，不把每格训练拆成审批。本任务不授权下一包、Slow 课程、git 提交或新增哈希。

## 0. 本包直接做什么

先让固定 Harness 下的 Fast 自己观察整批数据、构造处理方案、真实共同训练，再用当前反馈修订材料，检验能否产生有用交付。不是研究者拆完一个历史赢家再让 Fast 照抄。

唯一主要机制：**在已有完整训练方案上做一个可核对的单参数修订，使下一次训练回答一个明确的材料改动问题。**

三个搜索槽：第一份自由构造；若继续，第二份为已训练父方案的单参数修订；第三份可再修订或另行探索。任意阶段可交付已有方案、停止，不能为了满足流程强造第三份材料。

首轮固定 H0、0 Slow。不使用历史卡、D1、旧 E 排名或未评分草案池向 Fast 指定答案。已有 G3 正结果仅是研究者选择本机制的依据；原报告和原数字不改。

## 1. 主问题与对照

主比较：新 Harness 的 F_refine 是否比同预算自由构造 F_free 得到更好的完整共享模型交付，或用更低完整成本达到相当质量？

同时区分：
- 对 None 的处理收益；
- 对 Fixed、Menu 的新增价值；
- 对 Random-global 的搜索收益；
- 对 Random-local 的增量：避免把普通局部调参的收益全算给 Agent。

第一轮可以先取得“Fast 构造的材料有用”的开发结果；若只交付公共 Fixed，收益仍归固定处理。未胜过简单搜索，不宣称 Agent 独有优势。是否多观察、是否真的改了参数，只是过程读数。进化与历史积累本轮不测。

## 2. 接手文件与实现范围

必读：
- AGENTS.md §5.9、§6–9；
- docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md §16–17；
- _scratch/dev_batch_research_roundtrip/REPORT.md：旧串行反馈试验；
- _scratch/dev_batch_research_exploration_quota/REPORT.md：文字与硬配额的零交付差；
- methods/ttha/batch_research.py；
- methods/ttha/batch_base/{policy,materials,train,context,spec,feedback,budget}.py；
- evaluation/main_protocol_p4/{batch_research_runtime,run_batch_research_v1,run_batch_research_roundtrip,batch_research_draft_construction,batch_research_exploration_quota}.py。

建议新增一个薄 study：batch_research_paired_refinement.py，沿用现役 runner --study paired_refinement。
输出只写 _scratch/dev_batch_research_paired_refinement/；任务书完成后追加实际收口节。

允许必要的兼容扩展：
1. 薄 Adapter 支持“由父 policy 改一个参数”的 build_material 形式；
2. branch/resume_branch 接收本包显式 Limits 与 evidence_roundtrip，默认旧值不变；
3. Runner 增加本 study 的 prepare/run/resume/report 分发及配置；
4. 原有 trace/material_spec 中保存 parent/edit/实际差异；不另建知识或审计平台。

具体代码陷阱：当前 run_batch_research_v1.branch/resume_branch 都硬编码默认两槽 Limits，resume_branch 还硬编码 evidence_roundtrip=False。必须显式传递本包的三槽/32工具/往返配置并在恢复时保持；不能只改启动路径。Runner 也有 study 白名单、planned_fits、fit_plan 和 jobs 的硬编码，需同步本 study，不以旧“72次/两槽”作实际账本。
已有 prepare/load_pools 分发可复用；本包随机流只给零 LLM 对照，不向 Fast 提供草案。不要为适配分发伪造“所有臂都有候选池”，不要修改旧 study 全局常量。

旧包、P/G 工作树、参考仓库、AGENTS.md 不写。不装依赖、不换 Consumer，不重新审计已闭合故障路径。

## 3. 两个 Job：G5，按顺序取整组，不筛成绩

来源为本地 TSLib Electricity CSV，321 实体列、T=26304：
/mnt/c/Users/辉/desktop/agent/shared_tsq_datasets/electricity/electricity.csv
Windows 路径沿用 spec.DATASETS，解析后须为同一文件；不称 Monash。

原始字符串 ID 排序的 [160:192]，记 G5（G4 后的下一整组）：
~~~
242,243,244,245,246,247,248,249,25,250,251,252,253,254,255,256,
257,258,259,26,260,261,262,263,264,265,266,267,268,269,27,270
~~~

t=24*floor(a*26304/24)。两 Job 明确绑定同一份 G5，不能落回默认 G0。

|Job|job_index|T 区间|C_A origins|C_B origins|E origins|
|---|---:|---|---|---|---|
|a65_g5|0|[16416,17088)|17088,17136|17184,17232|17280,17328,17376,17424|
|a85_g5|1|[21672,22344)|22344,22392|22440,22488|22536,22584,22632,22680|

目标长度48。两完整工作负载分别结束17472、22728；彼此不相交。相同人口两个时间截点，不是独立同分布样本，也不是未见来源。

Planner 本轮仅解析两段 T 的指定列核资格：各672行，非有限0、零尺度0，最小总体标准差分别17.4072796266和14.3136965531；未解析本包 C/E 数值。执行者仍在冻结配置后由现役 Runner 检查资格/roster/scaler。
任一 Job 不合格则记 ELIGIBILITY_FAILED，不替换列、人口或时间；另一个合格 Job 可继续。两者均为 EXPOSED_DEVELOPMENT，不声称 untouched/Natural Final；整文件历史曝光事实保持。

## 4. 公共训练与工具语义

沿用现役：
- T672小时，L192/H48，stride1，433父对/实体、13856父对/Job；
- 每实体 scaler 只由 T 估计，派生材料不另归一化；
- 原始/派生视图各0.5；identity 沿用现役数值与权重；
- 每个完整材料×seed 共同训练一个 MLP：192→128→64→48，ReLU，无dropout/BN；
- AdamW，lr=.001，weight_decay=.0001，2000更新，父batch64；
- 主指标 normalized MSE 的实体/origin宏均值，所有实体保留；MAE/MASE辅助，不改采用目标。

配对训练 seeds 固定为 20261004、20261005、20261006；实际交付第一 seed，不挑最优 seed、不集成。初始化/父batch按现役公式对齐，增强流仍为900090+i+100000*step，材料在训练重复前冻结。

合法动作沿用现役：TimeMixup w={.10,.25,.50}，FreqMask/FreqMix mu={.05,.10,.20}，donor R/U；最多2步、8规则，first-match。本包不扩取值、不开放跨实体 donor、不改变执行公式。
None、FixedMixup（均匀R-TimeMixup .25）是公共已评估基线，各3seed，每Job只物理拟合一次。

两 Fast 均完整读取当前 overview 和公共基线 C_A；都可用现役 inspect_data/inspect_material/compare，额外材料诊断不自动前置。都无 H 历史、无 Source Episode、无旧赢家注入。

两 Fast 均 evidence_roundtrip=True，复用既有机制，不将串行本身认作新贡献。新结果必须出现在后续请求中再用于构造；不是原两个槽一次写完、只改报告称呼。相同 T/反馈可见时序、Limits 和工具说明用于两臂。

环境优先复用上一包解释器。只读冻结实际 Python/torch/device，所有格相同；上一包实际为Python3.13.5/torch2.12.1+cpu。环境检测放独立子进程，避免控制进程 import torch 与 numpy/MKL 双OpenMP问题。不得写成未经核实的旧CUDA环境。

## 5. 新增的父方案派生形式：两 Fast 都可使用

保留原 build_material({plan_id,policy})。
本 study 同时接受：
~~~json
{
  "plan_id": "P2",
  "parent_plan_id": "P1",
  "parameter_edit": {
    "location": "default",
    "rule_index": null,
    "step_index": 0,
    "parameter": "mu",
    "new_value": 0.20
  },
  "rationale": "Brief current-evidence hypothesis and the comparison being tested."
}
~~~

这是语法示例，不是推荐的算子或参数方向。location=default 时 rule_index 必须null；location=rule 时为现存规则的0起始索引。step_index指向现存步骤。parameter只能是该步骤的w或mu；new_value为现役允许且不同于原值的数值，bool不作数值。

父方案必须为本臂当前Job已完成全部3seed、且真实C_A已回传的方案；可以是公共Fixed，不可以是外臂方案、未来计划或仅构造未训练方案。None本身无强度参数，不自动为它添加步骤。

执行：
- 深拷贝父方案经验证的policy，只改指定数值叶子；rationale可换成当前解释，其他操作/条件/顺序不动。
- 由现役编译器展开，而非手工改32行assignment。保留父方案，不覆盖旧数组/评分。
- 指定叶子需作用于至少一个当前实体；不允许修改不生效分支后声称发生了材料干预。
- 记录父/子policy、参数旧/新值、实际改变的实体和数组差异。未改实体的材料/权重与父方案逐位一致；改变实体仍按统一冻结随机流生成。
- 实际数组改变不等于有用；由完整共同拟合检验。逐实体预测差仍非该实体训练材料的因果贡献。

本 study 两 Fast 统一使用语义去重：新plan_id若与已注册基线/本臂既有assignment相同，在材料/拟合前返回现役ToolInputError并给已有ID；用旧ID的evaluate为缓存读取，不占新槽、不推动第二/第三阶段。这是为了避免“改名重放”绕过第二槽合同，不能额外赠送拟合。初始化/恢复公共计划使用原ID，不能被误伤。用现有规范JSON/assignment直接比较，0哈希。

这只是小型父policy派生助手，不增加通用patch语言。两 Fast 都看到相同的语法、当前合法父方案提示和错误反馈；旧study保持原行为。

## 6. 五个方法：三槽同预算，包含普通局部搜索

|臂|构造/反馈组织|最多新材料|LLM|
|---|---|---:|---|
|F_refine|自由起点→单参数配对修订→自由再探索或修订，可提前停止|3|Fast|
|F_free|同工具、同往返，自主决定全部构造；派生助手可选|3|Fast|
|Random-global|冻结随机流的前三份不同合法完整方案|3|0|
|Random-local|同一随机起点→随机单参数修订→另一随机完整方案|3|0|
|Menu|均匀U-Mixup .25、均匀FreqMask .10|2|0|

F_refine：
- 第一份新评估从完整policy自由构造；构造前可观察，或直接commit已有基线。
- 第一份完成并回传后，若使用第二份新评估，必须通过§5的派生形式构造，父方案由Fast从本臂已评估方案中选择。
- 第三份可自由构造或派生。无必须评满三槽、必须放大强度、必须改变分组的要求。
- 第二槽父子关系在现役check_evaluate钩子中、槽计数和拟合预留前验证。错误按本包共同的2次ToolInputError纠正预算返回；不自动帮改、不自动训练。
- 为防事前把第二方案写死，F_refine在第一份新评估完成前不能用派生形式预造第二方案；往返机制保证第一份evaluate之后同响应中的build会延期。事前另造的自由方案不能冒充第二槽派生，但可留到第三槽。
- 任何时候可commit任一完整评估方案，包括None/Fixed；没有新的commit质量门。

F_free：
- 同样的工具、往返和三槽上限；随时可派生，但没有必须派生或必须何时派生的要求。
- 同样可停止、返回基线。它是本包的同预算自由构造对照，不拿旧“两槽、无往返”的历史分数替代。

冻结公共系统语义沿用FAST_SYSTEM；仅如实增补父方案构造语法及三槽事实。两臂共用一条简短过程说明：
“Use current batch evidence and actual paired C_A feedback to seek useful complete training material. State a brief testable hypothesis in the material rationale, name the comparison, and revise or stop as useful. Stronger, weaker, uniform and conditional constructions are all hypotheses, not defaults. Extra observations, edits and fits are not goals.”
F_refine另外声明上述第二槽合同；F_free明确无此要求。冻结实际发出的全文；不加“必须更大胆/优先频域/保护周期就轻处理”等效果提示。

本处理包括公开的修订骨架及其执行约束；不声称已单独识别文字与Runtime各自贡献。

## 7. 零LLM对照的确定性定义

### 7.1 共用随机全局起点

两Job在首次拟合前，只用T编译冻结：
~~~
proposal_seed(j) = 2026091700 + 1000*job_index + j, j=0..63
按j升序调用现役 policy.random_policy(seed)
按assignment排除None、Fixed和先前已收录者
保留前4份，编号R1..R4
~~~
保持生成器default/rules分布，rationale可改中性说明。不是整个DSL的均匀分布，不按距离、复杂度、代理或成绩筛选。
这些候选仅给Random对照，Fast请求中不出现。公共T准备工作/随机流生成成本单列。
64次不够4份：随机对照记SUPPLY_INCOMPLETE，不换流；健康Fast/Menu可继续，涉及随机的主比较缺失而非伪造结果。

Random-global评估R1/R2/R3；从None/Fixed及自身已评估材料按三seed C_A宏均值最小者交付，完全相等时Fixed、None、候选评估顺序优先。

### 7.2 Random-local

1. 首份同为R1（材料与Random-global相同，各臂按自身预算真实训练，不共享私有评分）。
2. 在None/Fixed/R1中，列出有非重复合法单参数邻居的完整父方案。按父方案C_A三seed宏均值最小者选择父方案；平局Fixed、None、评估顺序。None没有邻居，因此不被隐式加步骤。
3. 枚举该父方案default以及规则顺序中的现存w/mu叶子，再枚举不同合法值，按§5编译；排除无当前实体作用及与本臂既有assignment重复者。语义重复邻居只保留第一个路径，形成确定顺序。
4. 使用RandomState(2026091800+1000*job_index).randint(n_neighbors)抽一个邻居，记L2，真实评估。该流只在第二槽抽一次，恢复不重抽。记录全部候选地址/值及所抽索引即可，不加哈希。
5. 第三份取R2/R3/R4中首个与本臂已注册assignment不重复者。既不依据其他臂评分选择，也不把第三槽再改成一次看分挑邻居。
6. 最终交付规则同Random-global。

若异常地无任何邻居，记录LOCAL_NEIGHBORHOOD_EMPTY，停止并从已完成方案交付；不无限重抽。固定基线的未试w通常提供合法邻居，此处必须按实际语义验证。
它是一份不使用LLM的局部搜索参照，不能称与Fast起点相同：只与Random-global共享R1。

### 7.3 Menu

只评估原有MenuU=均匀U-TimeMixup .25及MenuFM=均匀FreqMask .10，按C_A在自身与None/Fixed中交付；完全平局Fixed、None、MenuU、MenuFM。保留少用一槽的真实成本，不制造无意义第三次训练。

## 8. 冻结预算、顺序与故障语义

物理拟合预算：
~~~
每Job公共基线                     2×3 = 6
F_refine/F_free/Random-global/local 4×3×3 = 36
Menu                              2×3 = 6
每Job最高48，两Job最高96
另留2次同配置非科学故障重试：总尝试硬帽98
~~~

|资源|上限|
|---|---:|
|Job / 完整搜索臂|2 / 每Job5臂|
|Fast轨迹|2臂×2Job=4|
|每Fast请求 / 工具尝试 / 新评估|16 / 32 / 3|
|每Fast可纠正工具错误|2|
|逻辑LLM请求 / HTTP尝试|64 / 128|
|已知输入+输出token停止边界|3,000,000|
|计划物理拟合 / 含重试尝试上限|96 / 98|
|整包实验墙钟（含暂停）|10,800秒|
|单fit子进程 / 单HTTP请求|最多300秒，且不超过剩余包墙钟|
|Slow / Source形成 / 新SHA / git提交|全部0|

相比上包增加一个搜索槽与局部搜索参照，已在此明确计费；不能继续套用72拟合配置。公共基线共享；偶然相同的私有候选按各臂原路径分别训练，本臂精确缓存/公共别名按实际情况记账。没有新增原种子复现拟合预算，必要一致性用已有公共材料与新对照自然重复检查。

固定执行顺序：
- a65_g5：F_refine → F_free → Random-global → Random-local → Menu；
- a85_g5：Menu → Random-local → Random-global → F_free → F_refine。
两Job都先完成T资格和随机流冻结，再启动任何拟合。一个Job的观察/轨迹/私有评分不进入另一个Job。

模型沿用现役本地中转与cpa-grok-4.6请求别名，核对返回grok-4.6-build；凭据从现有配置读取，不写入任务书/日志，不付费探活、不静默换模型。
未知usage按现役规则保留UNKNOWN。同请求传输重试至多一次；未知出现后阻止新的逻辑收费请求。已有适用于本包的用户明确授权可按授权留痕执行，否则执行者不得自行接受新的未知费用、把预留估计记成实际用量或保证上界。
恢复要求用户明确接受未知费用（若涉及），且通过现役labels_boundary_violations；已开C_B/E不可续跑未完成方法。恢复从原请求与原材料继续，不重调模型选答案、不重抽随机对照。
方法级契约/权限失败记METHOD_INCOMPLETE，其他健康臂继续；全局预算/账户/信息墙问题停止并扣留标签。失败不冒充主动停止或基线交付。未知费用与阶段合规分列：执行冻结协议允许的授权恢复本身不等于协议违反。

## 9. 实现与最小检查后连续运行

先做一套本包smoke，0实验LLM、0真实拟合：
- 派生默认/规则参数时只有指定叶子变化，父方案不变；inactive分支、原值、非法值、未训练/外臂父方案被拒。
- 未改实体材料一致；实际改变落在指定规则作用实体。返回真实变化量，不返回效果预测。
- 两Fast都能使用同一派生助手；只有F_refine第二新槽受约束；0/1槽commit合法，第三槽自由，缓存/改名不能绕过规则。
- 第一份反馈确实回到后续请求再构造第二份；恢复保留三槽Limits、往返模式、父子关系和已消耗计数，不能落回旧默认。
- Random-global/local的起点、邻居枚举、一次抽样及去重可确定性复现，0分数预筛；请求中不泄露其候选。
- G5贯穿观察、材料、模型和评分；失败阶段扣留标签。复用已有控制器测试，旧恢复检查不另建审计包。

smoke通过后执行者连续完成：
1. 冻结实现版本说明、配置、实际提示/合同、预算、人口、随机流定义和环境；
2. 只开两Job的T，资格与roster检查，生成零LLM对照随机流并冻结；
3. 共同基线，按冻结顺序完成全部计划方法（含失败判定），阶段内只有T+C_A；
4. 全部commit或方法终止状态冻结后，统一开放C_B作延迟检查；
5. 全局冻结所有合格模型的E预测，之后才统一读E真值评分；
6. 从真实记录收口，停止。不能看E后补草案、追加seed、改提示或恢复失败方法。

开发允许修复不改变实验定义的实现问题；一旦收费/拟合开始，方法、预算、候选规则及信息边界不得临场修改。涉及解释的配置变化应收口旧尝试并报告，不能悄悄续成同一实验。已授权预算内普通操作不逐笔请示。

## 10. 结果计算与主报告首页

每Job的配对差定义正值=F_refine更好：
- 主：F_free − F_refine；
- 必报：None、Fixed、Menu、Random-global、Random-local − F_refine；
- 子机制：每个实际修订的 parent − child，分别在C_A/C_B/E上保留逐seed差；
- 全局/局部随机对照差，用于观察普通局部结构的作用。

按完整共同拟合模型计算三seed差、均值、SE与符号；实际第一seed交付另列。实体/窗口向量用于贡献诊断，不作为独立训练重复。两Job不混成6个独立样本，也不把三个评分面视作独立重复。父候选或对照未完整运行则比较UNKNOWN，不能推算其分数。
无新的2SE/显著性采用门，commit仍由Fast选择自己已完成方案；主指标不改，难例不删。评价所有已完成模型便于诊断，但不能用跨模型逐实体最优拼出“真实oracle”。

首页只回答四问：
1. Fast实际上怎样构造、读取反馈与修订？哪对parent/child真实不同，哪次停止/换方向？区分建议、实际工具和数组。
2. 实际交付对None是否有收益、相对Fixed/Menu/两种随机搜索/自由Fast怎样？收益来自新材料还是公共基线？完整费用与未知费用分别是多少？
3. 配对修订有何贡献？子材料被采用/被拒的原因，已测更好候选是没交付还是没构造；仅新增参数对照或更多调用不算收益。
4. 当前到哪个台阶、最大剩余不确定性和唯一后续建议是什么？

判读：
- 构造与修订真实完成但交付无改善：框架行为成立，本机制效用未成立。
- 仅胜None，或交付公共Fixed：分别记处理收益/固定处理收益，不宣称Agent或Skill增量。
- Random-local解释同等改善：普通局部搜索有用，LLM额外价值未建立。
- 相对自由Fast/简单搜索更好或同质量更省：有限Fast增量线索；单数据集、两Job、每臂一条LLM轨迹，不直接晋升稳定能力。
- 两Job差分不开/方向冲突：保留INCONCLUSIVE或限定范围，不追加到正号。
- 0/1槽停止导致未出现parent-child：报告该固定Harness在当前Job的实际选择与端到端结果，不能自动重跑直到触发修订。

后续Slow需要的Episode可以照常保留，但本包不调用Slow、不写新Skill、不修改H0、不自动进入课程。

## 11. 工件和终点

只用一个逻辑Runner和一个主报告。沿用现有config/budget/result/trace/material/model/prediction文件；补充父子语义、随机邻居选择与必要检查到现有记录，不建立新哈希或全仓库台账。
每个新材料保存完整policy、assignment、scaler/权重绑定、父方案及单叶子编辑（若有）；原始LLM请求/响应及真实费用可回查。报告数值从这些真实模型/评分重算。

报告路径：_scratch/dev_batch_research_paired_refinement/REPORT.md。
完成后只在本任务书追加§12实际收口：执行状态、协议状态、结果、成本、失败/授权恢复（若有）、修改文件与未做事项。完成即停止，不启动下一实验，不git commit/push，0新增SHA。

## 12. 实际收口（执行者追加，2026-09-14）

**执行状态**：COMPLETE。10/10 臂交付；93 次物理拟合（计划最多 96，a65 F_free 只用两槽）全部成功，0 训练重试，66 次缓存命中；28 个逻辑请求 / 29 次 HTTP / 27 次有效返回（`grok-4.6-build`）；已知 token 1,170,038 入 + 29,322 出；包墙钟 5512 s（含 1138 s 停机等待）。0 Slow、0 新 SHA、未 git 提交。

**协议状态**：`audit.json` 全部数值/绑定/阶段/流/派生/合同/信息墙检查通过（141 个 E 格由原始预测与显式 G5 roster 重算，最大差 7.1e-15；随机供给由公共 T 确定性复现且未出现在任何 Fast 请求中；每个派生子方案 = 编译器对父 policy 改一叶的展开，未改分支实体数组逐位相同；F_refine 第一槽为完整 policy、第二槽为反馈后构造的派生方案；四条轨迹 0 次拒绝）。`original_protocol_compliant=false`：request 16（a85_g5_f_free 首次请求）两次 `APIConnectionError`（本机 8318 中转当时拒绝连接）→ 2 次未知费用 → 冻结规则停机、标签扣留（未开任何 C_B/E）；用户回复"现在好了，继续"（记录于 `operator_decision_resume.json`）后 `--resume --accept-unknown-usage`，重发请求与未答请求除 `remaining.seconds` 外逐字相同，恢复 0 次新增拟合；`frozen_protocol_followed_including_authorized_resume=true`。未知费用保持 UNKNOWN（预留估计 175,174 非上界保证）。另一次启动因本机用电池且内存耗尽（单次拟合约 100 s，将超墙钟）在 0 次 LLM 请求时被执行者中止，现场留为 `_scratch/dev_batch_research_paired_refinement_attempt1_env_aborted/`，定义未改。

**结果**（三 seed E，正值 = F_refine 更好）：
- a65_g5：五臂全部交付 None，所有臂间差为 0。C_A 上 None 最优，E 上所有增强方案都比 None 好（Fixed 好 0.060，FreqMask .1 好 0.134，随机 R3 好 0.146）：该切点 C_A 与 E 排序 36 对中 23 对反向。
- a85_g5：F_refine 交付公共 FixedMixup（E 0.14239），为五臂交付中最差：F_free −0.00370 [−−−]、Random-global −0.00987 [−−−]、Menu −0.00591 [−−−]、Random-local −0.00249 [−+−]；对 None +0.00630 [+++]（固定处理收益）。
- 配对修订（5 次真实 parent→child，含 Random-local 2 次）：C_A 上无一可辨优于父方案，全部未采用；a85 的 Fixed w .25→.5 子方案 E 比父方案好 0.00716 [+++]，被 C_A 判差（−0.00112 [−+−]）而未交付；F_free 自由构造了同一材料也未交付。Fast 在两 Job 都拟合了比自己交付更好的材料——"没交付"为主导模式，"没构造"仅对各 Job 的最优材料（均为随机臂的带规则方案）成立。
- 按 §10 判读：框架行为成立，本机制效用未成立；a85 记固定处理收益；普通局部搜索（Random-local 同样修订一次、同样不采用）解释了 F_refine 的可见行为，LLM 额外价值未建立；两 Job 一平一负，不追加正号。
- 成本：F_refine 在 a65 用 F_free 2.4 倍 token 得同一交付，在 a85 用 1.2 倍得更差交付。

**唯一后续建议（提议，未授权）**：先做 0 LLM 的选择信号诊断包（汇总 G3/G4/G5 全部已拟合方案，量化 argmin-C_A 交付相对集合内 E 最优的遗憾，检验仅用持有内信息的替代规则），再决定是否继续 Fast 侧干预。

**修改文件**：新增 `evaluation/main_protocol_p4/batch_research_paired_refinement.py`；`run_batch_research_v1.py`（branch/resume_branch 显式 `limits`/`evidence_roundtrip`，默认旧值不变；恢复时被拒绝的 evaluate 不再计为已消耗槽）；`run_batch_research_roundtrip.py`（study 分发、配置、`smoke_paired_refinement`、CLI 选项、`FIT_PLAN`/`POOL_NAME`/`FROZEN_MARKER` 属性化读取）；`batch_research_source_process._guidance_readout`（兼容无 `policy` 参数的派生 build；旧 study 的 result.json 复现逐字节相同）。26 个控制器测试与 7 个 study smoke 全部通过。旧 study 常量、旧包目录、AGENTS.md、框架文档未改。

**未做**：未看 E 后补方案/补 seed/改提示/重跑任何臂；未调用 Slow、未写 Skill、未改 H0；未启动下一实验。
