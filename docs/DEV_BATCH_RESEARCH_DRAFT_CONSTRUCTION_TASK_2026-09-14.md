# DEV-BATCH-RESEARCH-DRAFT-CONSTRUCTION：草案辅助的批级材料构造

日期：2026-09-14。状态：**原 G2 运行已因两作业资格失败收口；下一次 G3 运行采用 §13 人口修订，尚未由 Planner 启动。**

**执行入口：§13 覆盖正文中的 G2 人口、Job 名和输出目录，其余协议保持原义。原 G2 条款保留为失败运行的历史规格，不得原地改写旧运行记录。**
主执行者：接收本任务的 Opus。Planner：Astra。模式：M2 开发级方法实验。
权威关系：AGENTS.md §5.9 → BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md → 本任务书。

## 0. 直接说明：本包在做什么

给 Fast 四份依据当前 T 实例化、尚未训练评分的合法构造草案，让它观察数据，自主挑选、修改或另行构造，然后用真实共享训练检验。

同时比较：自由构造、拿草案但只能挑选、拿草案且可以修改、随机搜索、固定菜单。区分三个来源：草案供给本身，Agent 选择，Agent 构造。

这个改变不引入新算子，也不将随机方案写成 Skill。它是一项拟检验的候选构造辅助机制，效果未知；不是完整 Source/A3/A5 实验，更不是已经成立的新贡献。

用户转发并要求执行本任务后，执行者在以下范围内完成实现、一次必要 smoke、真实运行、统一评分与报告；普通实现选择自行解决。不得把每格拟合再拆成审批，不自行追加下一包。

## 1. 唯一主问题与依据

**给定相同的 T、Consumer 和两次新材料评估预算，未评分的多样化草案能否帮助 Fast 构造出比自由构造、只选草案、随机搜索和固定菜单更有用的完整训练材料？**

依据只用于解释研究者为什么提出这个机制，不作为发给 Fast 的答案：

- 最新 SKILL-REVISION 中，新旧卡都构造 U-TimeMixup 0.25 + FreqMask 0.1，交付与 Menu 相同，新卡成本更高。实验组织改变未带来构造增量。
- FAST-BASELINES 的 a95 曾出现 Random 的完整材料在 C_A/C_B/E 都优于当次 Fast 交付。因此至少在那个已曝光作业，不能把差距全部归因于反馈或不存在可用材料。
- 现役 DSL 已支持混合强度、频域强度、R/U donor、条件分组、至多两步组合。新机制尝试改变 Fast 的构造起点，不先扩大这些范围。

不能将 a95 的随机赢家、既有 E 排名或旧卡的菜单作为本包草案。草案由预先冻结的随机生成规则和当前 T 产生，不取自历史 Episode。

本包可编辑变量是**是否提供未评分草案，以及是否允许在草案之外构造**。反馈、算子、训练、评分、交付规则保持同一套。F_select 的构造限制是明确的消融，不伪装成与 F_edit 完全相同的动作权限。

## 2. 接手文件与修改范围

必读：

- `docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md`，特别是 §13；
- `docs/DEV_BATCH_RESEARCH_SKILL_REVISION_TASK_2026-09-14.md` §13 与 `_scratch/dev_batch_research_skill_revision/REPORT.md`；
- `_scratch/dev_batch_research_fast_baselines/REPORT.md` §2–3，只作研究者依据；
- `methods/ttha/batch_research.py`；
- `methods/ttha/batch_base/{spec,policy,context,materials,train,commit}.py`；
- `evaluation/main_protocol_p4/{batch_research_runtime,run_batch_research_v1,run_batch_research_roundtrip,batch_research_skill_revision}.py`。

复用现有 run_job、Adapter、显式 roster、配对种子、私有候选、公共基线、暂停/恢复和标签屏障。建议增加一个薄 study 模块与现役 runner 的 `--study draft_construction` 分支；具体文件拆分由执行者决定。

允许本包新增：草案生成与 T-only 冻结、向指定臂提供草案、F_select 的编译后集合检查、结果中的草案谱系分析。不要新建搜索平台、Skill 平台、哈希或第二套训练器。

工件写入 `_scratch/dev_batch_research_draft_construction/`。旧实验目录、P/G 工作树、AGENTS.md、历史卡和数字不写。完成后仅追加本任务书的实际收口节；不 git commit/push，不晋升 canonical Skill。

## 3. 两个 Target：先定人口和时间，不筛 C_A

来源：本地 TSLib Electricity CSV，321 列、26304 小时，不称 Monash。
路径：`/mnt/c/Users/辉/desktop/agent/shared_tsq_datasets/electricity/electricity.csv`，Windows 路径沿用 spec.DATASETS。

两作业均使用原始字符串 ID 排序后的 `[64:96]`，记 G2；规划时只核了 CSV 表头，未读本包数值或 C/E：

```text
156,157,158,159,16,160,161,162,163,164,165,166,167,168,169,17,
170,171,172,173,174,175,176,177,178,179,18,180,181,182,183,184
```

这是对本包的明确人口扩展，复用 SKILL-REVISION 已完成的可选 roster 接口。不能用默认 G0 代替；人口贯通观察、scaler、父对、材料、拟合、commit、C_B/E。

切点按 `t=24*floor(a*26304/24)`，固定 a=.65/.85。这两个比例和 G2 在任何当前分数打开前确定；不按旧的 G0 表现选配方，也不根据本包资格或结果换组。

|Job|T 区间|C_A origins|C_B origins|E origins|
|---|---|---|---|---|
|a65_g2|[16416,17088)|17088,17136|17184,17232|17280,17328,17376,17424|
|a85_g2|[21672,22344)|22344,22392|22440,22488|22536,22584,22632,22680|

目标窗均长48。两作业完整工作负载分别为 [16416,17472)、[21672,22728)，不相交；当前两个作业的训练材料重新构造、重新拟合，不是同一模型增加评价窗口。
日期：a65_g2 的 T 为 2018-05-16 02:00 至 2018-06-13 02:00，E 目标结束 2018-06-29 02:00；a85_g2 的 T 为 2018-12-21 02:00 至 2019-01-18 02:00，E 目标结束 2019-02-03 02:00。

数据均标 EXPOSED_DEVELOPMENT。它们是同一来源、同一人口的两个时间作业，不宣称独立同分布、未见来源、untouched 或 Natural Final。不同臂不共享当前私有反馈；两个 Job 不传递轨迹、草案或知识。

配置冻结后只用 T 做资格检查。任一 Job 不合格则记 ELIGIBILITY_FAILED，不换人口或时间；另一个合格 Job 可继续。C_A 非有限等运行故障按现役语义记录，不把失败当主动 identity。

## 4. 冻结草案生成：0 评分、0 历史赢家

每个 Job 在公共基线训练之前，使用 `context.open_job(..., stage='material', roster=G2)` 取得当前 T 观察。只用 T 编译，不读取 C_A/C_B/E、旧模型损失或其他臂候选。

复用 `policy.random_policy(seed)` 原有分布；不声称在整个策略空间均匀。按固定流依次生成，编译为当前32实体的 assignment：

```text
job_index = 0 for a65_g2, 1 for a85_g2
proposal_seed(j) = 2026091400 + 1000*job_index + j, j = 0..63
按 j 升序：
    p = random_policy(proposal_seed(j))
    assignment = compile_policy(p, T_overview.entities).assignment
    排除与 None / Fixed 的 assignment 相同者
    排除与已收录草案 assignment 相同者
    保留最先出现的4份，编号 D1..D4
```

去重直接比较已有 assignment/规范 JSON，不新增 hash。只按确定性语义去重，不按预测、代理分、材料是否“合理”、操作复杂度或人工偏好淘汰。64次仍不足4份则该 Job 标 DRAFT_POOL_INCOMPLETE，不换种子，不带不完整池进入主要比较。

冻结每份 policy、proposal_seed、展开 assignment、规则覆盖数/identity数。当前材料值尚未生成也可，只要通过现役编译；按需 build/inspect_material 走原接口。冻结时间必须早于任何本包 C_A 打开。

向 F_select/F_edit 的初始 overview 加同一份 `unscored_drafts`：D1..D4、完整 policy、T-only 编译覆盖信息、生成分布说明。明确这些是随机生成的未评分建议，没有历史证据、质量顺序或推荐等级。不要把4份草案加入 evaluated candidates，不虚构模型、分数或 fit_id。

F_free 不收到草案，overview 的原有信息不变；Random 使用相同草案池的 D1/D2；Menu 不读取池。草案池是待检验的新供给方式，不能作为公共“接口修复”偷偷加给 F_free。

两池都先生成并冻结，再启动任何拟合。不得在看完第一个 Job 结果后重抽第二池。

## 5. 五个方法与公共基线

所有 Fast 采用空 H0、相同现役 FAST_SYSTEM 和工具事实，不加载旧卡、新卡或 Generic，不调用 Slow。只增加说明相应草案/构造权限所必需的中性合同，禁止加入历史结果或“应当更大胆”之类效果暗示。

|臂|初始材料信息|允许行为|主要解释|
|---|---|---|---|
|F_free|原有 overview + None/Fixed 的真实 C_A|现役全 DSL 自由构造、观察、评估、commit|无草案的现役 Fast|
|F_select|F_free 信息 + D1..D4 未评分草案|可观察、挑草案、评估、比较和停止；新评估仅限4草案|Agent 选择的作用|
|F_edit|与 F_select 相同|可复用草案、修改参数/规则/步骤，也可另行构造合法方案|草案辅助的构造 Workflow|
|Random|同一个 D1..D4 池|确定性评估 D1/D2；从已评估集合按 C_A 选|随机供给/搜索解释，0 LLM|
|Menu|预先固定 U-Mixup .25 + FreqMask .1|评估两份；从已评估集合按 C_A 选|固定菜单解释，0 LLM|

公共锚点：None = identity；Fixed = 全体 R-TimeMixup w=.25。每 Job 两份公共材料各3 seed，共6次拟合，之后各臂只复用公共基线。

**三个 Fast 都最多评估2份新的完整材料，每份3seed。** 允许只用1槽或直接提交公共基线，不强制消耗预算，不强制修改草案。build/inspect 本身沿用工具次数限制，不等于新 fit。不可把节余转成第3份评估或新草案池。

F_select 的限制在薄 Adapter 的 build_material 编译后检查：assignment 必须等于本 Job 的 D1..D4 或公共基线；文本不同但材料相同可接受，真实改变则在执行前按已有可纠正工具错误返回，最多2次纠正。不得先训练再检查。F_edit/F_free 均保持完整原 DSL，不增加额外构造强制条件。

F_edit 可在已有 rationale 中注明来源草案及修改意图；这不是新必填 Schema，缺说明不追加调用。执行器按真实编译结果检查：与草案/菜单是否相同、改变了哪些参数/条件/步骤、影响哪些实体。模型自述不作为“确实修改”的唯一证据。

F_select 与 F_edit 都能先评估一个，再决定第二个；不新增强制暂停/roundtrip。允许两者自主批量操作。F_free 也可使用当前 C_A 迭代构造。

Random/Menu 的精确选择：C_A 三seed均值最小；只在完全相等时按 Fixed、None、D1、D2（Menu 为 U、FM）顺序。Fast 可以提交任一已完整评估方案，不由 Runtime 改写。不要把看过 C_A 后的 winner 重命名为事先指定的 incumbent。

## 6. 数值与训练配置

保持现役训练几何：L192/H48，T672小时，全433父窗/实体、13856父对，stride1；全批材料共同训练一个 MLP。父/子权重各.5，identity 的子槽等于父；T内每实体 mean/std scaler 固定，派生材料不再归一化。

MLP 192→128→64→48、ReLU、无dropout/BN；AdamW lr=.001、weight_decay=.0001、2000更新、parent batch64。主分数 normalized MSE 的实体/origin宏均值；原MAE/MASE保留为辅助，未定义项如实计数。

动作仍为：TimeMixup w={.10,.25,.50}、R/U；FreqMask mu={.05,.10,.20}；FreqMix mu={.05,.10,.20}、R/U；每配方至多2步、规则至多8条。沿用完整共同训练，不按实体拼接模型预测。

新配对训练 seeds：20260928、20260929、20260930。初始化、父batch按现役函数对齐；增强 seed 与模型 seed 分离，沿用 900090+i+100000*step。每臂自己的材料冻结后重复，不在3seed中重新调用 Fast 或重新抽草案。

实际交付固定第一seed20260928；三seed均值用于研究读数，不是ensemble，不挑seed。环境优先复用上一包实际可运行解释器；只读确认 Python/torch/device 并在开跑前冻结，所有臂相同。上一包工件实际记 torch2.12.1+cpu，不能未经核实写成旧CUDA环境。无需安装依赖或迁移Consumer；CPU/GPU选择不交给Fast，开跑后不切换。

本包不复用旧模型缓存；只在本 Job 复用公共基线与本臂已有别名。偶然相同的私有候选仍沿用现役各臂独立拟合，不共享私有分数。所有缓存键与模型记录包含显式 roster、Job、seed、材料语义。

## 7. 预算与运行顺序

最大计划物理拟合：

```text
每 Job：公共 None/Fixed 2×3 = 6
       五方法各至多2新方案×3seed = 30
两个 Job：2×(6+30) = 72
另留2次同配置非科学故障重试：拟合尝试硬帽74
```

|资源|上限|
|---|---:|
|Slow / Source 形成|0|
|Fast 轨迹|3臂×2Job = 6|
|每Fast Job逻辑请求 / 工具尝试|16 / 24|
|每Fast Job可纠正工具输入错误|2|
|全包逻辑LLM请求|96|
|全包HTTP尝试|192，单请求至多一次原请求重传|
|已知输入+输出token停止边界|3,000,000；未知费用仍按下述规则处理|
|拟合尝试|74，含至多2次预留重试|
|活跃实验墙钟|10,800秒，含暂停时的墙钟；实现时间另列|
|单fit子进程|300秒且不超过全包剩余墙钟|

草案生成/编译/材料观察为0拟合，但记录时间、构造次数和存储；不把它们称作免费。三条 Agent 臂都有相同预算帽，未用完是结果。Random/Menu 0实验LLM，但候选拟合全计费。

固定执行顺序：
- a65_g2：F_free → F_select → F_edit → Random → Menu。
- a85_g2：Menu → Random → F_edit → F_select → F_free。

各 Job 在第一条臂启动前只拟合一次公共基线。两 Job 不共享候选/结果。所有Job/臂完成或按协议明确终止前均不打开C_B/E。观察首Job运行健康不改变另一个Job配置、候选池或继续条件。

模型沿用用户授权的本地中转与请求别名 cpa-grok-4.6，核对实际返回 grok-4.6-build；凭据从现有配置安全读取，不写进任务书/日志、不付费探活、不默换模型。

失败请求或无usage响应保留 UNKNOWN。按原规则至多一次同请求重传；未知usage出现后拒绝新的逻辑收费调用。执行者不能自行加 --accept-unknown-usage。只有用户明确接受未知费用、且标签仍被扣留，才可恢复；不把token预留估计改记成真实用量或保证上界。

同配置拟合重试只用于已分类的非科学故障，不因负结果、non-finite或差值不清楚重跑。方法级契约失败记 METHOD_INCOMPLETE，其他健康臂继续；全局预算/账户/信息墙故障暂停全包并扣留标签。不用None/Fixed伪装一个未完成方法。

## 8. 实现和必要检查

本包只需一个逻辑Runner、一个主报告和一个必要smoke。优先复用已完成的原子预算/unknown usage/labels_boundary_violations，旧恢复审计不重开。

smoke 聚焦新增路径，0拟合、0实验LLM：

1. 草案池只调用material/T路径，按冻结流可重复生成；别名与公共基线排除仅依赖编译后的assignment，64次上限生效。可用小型合成overview检查，不将其作为方法证据。
2. F_select/F_edit获得同池；F_free/Menu请求中无草案字段或他臂结果。所有臂原工具事实和配对训练定义一致。
3. F_select可构建任一池内assignment及公共基线，池外改动在fit前被拒；F_edit同一合法改动可通过原DSL。限制不扩散至旧study/F_free。
4. 未评分草案不能直接commit或作为已评估反馈；拟合两槽按现役计数工作；跨Job/roster引用被既有绑定拒绝。

原必要控制测试运行一次即可；不为本包重训对齐模型或批量重跑历史实验。实施前核对正在运行的其他线，避免写其文件；常规代码冲突由执行者保留原行为解决。

## 9. 阶段顺序与停止边界

1. 完成薄实现和smoke，写本包config，冻结本任务所列协议、3个训练seed、两Job、人口、草案生成流、臂顺序、预算与提示合同。
2. 只开两Job的T，做资格检查、生成并冻结草案池；不接触本包C_A。生成规则不得因看到草案内容而调整。
3. 运行公共基线与各臂。训练子进程只读T+C_A，材料构造只读T。保存请求、工具返回、实际数组、模型与C_A。
4. 全部预定臂完成/明确终止，冻结全部commit，之后才开放C_B。C_B只作本包延迟诊断，无Slow，不改当前交付。
5. 对全部合格已拟合候选冻结E预测，建立全局屏障，再统一读取E评分。未拟合草案不得补训/补分，不能用它们作oracle。
6. 报告后收口。任何暂停复用冻结池与完成响应，不重抽草案、不重问已完成请求；现役标签边界守卫先于恢复费用接受。finalize或任一C_B/E已开后不恢复未完成方法。

程序性中断不能自动开放标签。健康情况下完成全包后统一评分；未知费用、主动终止等情况使用现役显式resume/finalize规则，保留不完整分母。不开Natural Final，不改旧包。

## 10. 主读数：构造、选择、交付分开

每Job报告完整材料共同训练的三seed C_A/C_B/E、第一seed实际交付、全成本；正差统一表示 F_edit 更好：

- 主比较一：Δ_free = loss(F_free) − loss(F_edit)。草案辅助是否胜过现役自由构造？
- 主比较二：Δ_select = loss(F_select) − loss(F_edit)。允许修改/另造是否超过只挑草案？
- 必要参照：Δ_random、Δ_menu，以及相对Fixed/None。

所有比较在开跑前列明；不在结果出来后挑有利一列称主结果。不跨Job直接把不同损失尺度相加后掩盖单Job负值；每Job单列。三seed报告配对差、均值与SE，不新增2SE/显著性采用门；两作业不是六条独立Agent轨迹。

过程只回答真实发生的事：

- F_edit是否使用草案？保留、修改或另造了什么？按实际assignment而非措辞定。
- 修改涉及强度、donor、算子、步骤/顺序还是条件作用范围？未变化的材料是否保持？
- 哪次当前合法观察/反馈先于该修改？同一响应中先写好后执行的动作不能叙述成已读新反馈后的适应。
- F_select是否比直接D1/D2搜索选得更好？F_edit是否仅复用了同一个好草案？
- 找到了更好的已评估材料但未交付，与没有构造出来分开。
- 额外token/时间是否换来质量或拟合节省？包含整个草案生成、基线、所有失败和私有拟合。

若某修改的父草案没有真实拟合，不能声称测得该次编辑的独立因果收益；只能报告两种完整研究策略的端到端差。逐实体预测差仍不是对应实体训练材料的独立价值。

集合内最优只在各臂实际完整拟合的候选中计算；跨臂联合集合仅作回顾性诊断，不授予任何Agent未见反馈，不拼接不同模型的逐实体预测。

## 11. 结果出口及后续接回Skill的方法

|观察|可支持的结论与下一步|
|---|---|
|F_edit相对F_free、F_select和简单对照有跨两个Job同向的实用收益，或质量相当且完整成本更低|有范围的M2开发证据；下一包设计Source/Target短课程，不直接宣布普遍有效|
|F_select与F_edit同样好，二者好于Random/Menu|主要支持基于当前数据选择草案；不强称材料修改增量|
|Random同样好或更好|新增空间由随机供给已解释，Agent增量未成立；不将随机生成器包装成学习|
|只胜过F_free，未胜过Menu/Random|草案起点可能修复自由生成弱点，端到端竞争优势未成立|
|材料或交付相同，成本增加|本次辅助机制无观察到的净效用；不以更长轨迹、更多观察、更多规则晋升|
|有用候选未交付，C_A/C_B/E反转|保留构造与决策分解，不按E改本次选择；最多提出一项后续决策假设|
|两Job方向不一或差值分不开|证据有限，不追加seed/换组直到正号|
|资格/技术故障导致不完整|报告实际完成分母，不将其当方法失败或identity成功|

“实用”须同时给绝对/相对收益和完整成本，保留所有对照；本包不事后设晋升阈值，不自动进入大课程。

若出现值得后续研究的构造/选择经验，下一包仍走既有合法路线：完整Episode的T/C_A/commit后C_B → Slow在批次边界提炼一个有限构造/观察原则 → 冻结后在变化的训练材料上重新实例化 → F/A3/A5同Target预算比较 → 最终Fast-only。历史原始草案/赢家不得直接伪装成Specific输入Fast；Source所学若只是复用好程序，需要保留程序复用对照。

本包0 Slow/Source更新，不能报告知识积累、跨域适应或Skill提升。若未获增量，本包只提出一个具体方法假设并停止，不把本任务变成不断抽草案和修改提示词的auto-research循环。

## 12. 交付与可直接转发的执行指令

交付：一份REPORT.md；config与两池冻结记录；result.json；原始请求/工具轨迹；真实材料/模型/冻结预测；现役预算/屏障文件；可复跑入口；一个必要smoke。保留实际别名/来源，不新建逐候选manifest或hash平台。

报告首页只回答：
1. 草案辅助让Fast构造了什么不同的材料，还是原样选择/继续旧菜单？
2. 相对自由构造、只选草案、随机搜索、固定菜单，质量和完整成本怎样？
3. 增量来自供给、选择、修改，还是未被交付的候选？哪些归因尚不能做？
4. 当前最大不确定性，以及仅一项下一步建议？

可转发：

> 请执行本任务书 DEV-BATCH-RESEARCH-DRAFT-CONSTRUCTION。先落实薄study与必要smoke，再按固定G2人口、a65/a85两Job、五臂和72次计划拟合连续完成真实运行、统一评分与报告。只测未评分草案是否改善批级Fast的构造/选择，0 Slow，不加载旧卡，不筛C_A批次，不新增算子、不放宽信息墙。预算和失败/暂停规则以本文为准；普通实现问题自行解决，完成即停止，不追加实验、不提交git。请如实区分草案本身有效、Agent选择有效和Agent修改有效，不把正号强行归给Harness。


## 13. G3 人口修订（2026-09-14，Planner 裁定，供用户转发执行）

### 13.1 依据与证据范围

原包 `_scratch/dev_batch_research_draft_construction/` 已收口为两 Job 的 ELIGIBILITY_FAILED。两段 T 中实体182均为常数0，现役 std≤1e-6 资格规则拒绝；原账本为0拟合、0请求、0token，两个草案池均未生成。已有空分支 `all_e_predictions_frozen.json` 的 branches=[]，不等于生成过预测或读取了E。原包不恢复、不覆盖，不计为方法负结果。

采用原始字符串 ID 排序中紧随 G2 的下一整组 `[96:128]`（G3），切点不变。依据是确定的相邻整组顺序与执行者已报告的两段 T 资格结果，不依据 C_A/C_B/E、基线效果或增强收益挑组。Planner 本次另核 CSV 表头绑定；未独立重算 G3 的数值资格，执行者在新运行阶段2仍按原函数验证。

这是失败后的显式人口修订，不冒充原 G2 预注册运行。全部数据保持 EXPOSED_DEVELOPMENT，结论适用于具备现役训练资格的批次；不能外推为常数/断流序列处理能力。

### 13.2 唯一配置变化与冻结人口

- 人口：G2 `[64:96]` → G3 `[96:128]`，32实体整组替换，不逐实体补位，不将182的std强行floor后放行。
- Job：`a65_g2` → `a65_g3`；`a85_g2` → `a85_g3`。时间索引、窗口、日期与§3相同。
- 新输出目录：`_scratch/dev_batch_research_draft_construction_g3/`；旧目录只读保留，不能用旧目录resume。

G3 显式 roster（原始字符串字典序）：

```text
185,186,187,188,189,19,190,191,192,193,194,195,196,197,198,199,
2,20,200,201,202,203,204,205,206,207,208,209,21,210,211,212
```

`job_index=0` 对应 a65_g3，`job_index=1` 对应 a85_g3。草案流仍为 `2026091400+1000*job_index+j`，j=0..63；不得因Job改名重新选seed或生成规则。训练seeds仍为20260928/29/30。五臂、原提示合同、2候选预算、全部动作/参数、Consumer、反馈/评分/commit、两Job臂顺序、故障和信息墙规则完全沿用正文。

### 13.3 执行与终点

1. 复用已经完成的study实现，只更新G3常量、Job/ORDER/ROSTERS绑定、人口说明与新root。原G2 frozen_config与报告保留原值；不要机械替换整个仓库中的G2。
2. 跑一次现有 draft-construction smoke，聚焦显式人口/Job/输出绑定与草案流保持；既有恢复故障修复和历史study审计不重开。不增对齐拟合、不装依赖、不加hash或git提交。
3. 在新root写冻结配置，按阶段2只读取两Job的T做资格检查；合格后两池均在任何C_A打开前生成冻结。若又有资格失败，按原停止分支收口，不自动推进G4、不更改切点。
4. 按原阶段顺序连续运行、冻结commit、读C_B、冻结全部E预测后统一评分；报告后停止。预算仍是72次计划拟合/74次尝试硬帽、96逻辑请求/192HTTP、3M已知token停止边界、10800秒实验墙钟、0Slow。原失败包实际0拟合/0LLM单列，既有实现与资格检查时间另列，不因新目录隐藏历史成本。
5. 报告注明本次为G3修订运行及原包资格失败原因。关于182的早期6389个非零记录，如需引用，应注明来自两个T窗之外的额外历史补查；仅两段T全零不能证明从某时刻起永久断流。此项文字准确性随新报告处理，不开新审计包。

可直接转发：

> 确认下一次运行使用 G3（字符串排序[96:128]），a=.65/.85及所有方法、种子、草案流、臂顺序和预算保持不变。新开 `_scratch/dev_batch_research_draft_construction_g3/`，保留原G2资格失败现场；不要删182补位或放宽零尺度规则。按本任务书§13完成必要绑定修改和一次现有smoke，再连续运行、统一评分、报告并停止，不追加筛组或实验。

## 14. 实际完成状态（2026-09-14，执行者收口）

**执行**：G2 预注册运行两作业 ELIGIBILITY_FAILED（实体 182 在两段 T 内全零；0 拟合、0 请求、0 token，池未生成；目录只读保留）。按 §13 改用 G3 后在 `_scratch/dev_batch_research_draft_construction_g3/` 完成：两池在第一次拟合前冻结，10 臂全部尝试，9 臂 COMPLETE 并统一评分，**a65_g3 F_free METHOD_INCOMPLETE**（请求 T 外 `inspect_data segment sub_range=[0,336)`，现役 PermissionError 终止规则，0 私有拟合，无 commit，不以基线伪装）。63 次物理拟合（≤72）、19 请求 = 19 HTTP、0 传输故障、0 未知 usage、788,102 已知 token、活跃墙钟 61.0 分钟；无暂停/恢复；`audit.json` 全绿（105 份 E 预测重算最大差 3.6e-15）。

**结果（E 三 seed 均值，正值 = F_edit 更好）**：a65_g3 Δ_free 无定义、Δ_select +0.0022（SE 0.0029）、Δ_random +0.0015（SE 0.0014）、Δ_menu +0.0022，全部噪声内且 C_A/E 排序相反；a85_g3 Δ_free −0.00007、Δ_select −0.0028（3/3）、Δ_random **−0.0139**（3/3）、Δ_menu −0.0075（3/3）。a85 的随机草案 D1（条件化 FreqMix+FreqMask / U-Mixup 0.5）是全包 C_A/C_B/E 三块最优；a85 F_select 两槽给了均匀草案 D2/D4（未调用观察工具，overview 特征已在手），a65 F_select 评估了条件化 D4 后提交 None；F_edit 两作业未观察到实际复用或可识别的草案编辑（a65 reasoning 曾拟评估 D1/D2 后放弃），构造与 F_free 同族 U-Mixup 方案，交付等于 F_free。本包证明的是 a85 有一个好候选被漏评，未证明漏评原因。

**判读（§11）**：a85 "Random 同样好或更好"——增量由随机供给解释，Agent 增量未成立；a65 "差值分不开"（分辨率不足，不推广）。草案供给本身有价值（一例），Agent 选择未捕获、未观察到 Agent 修改；F_edit 不晋升，0 Slow、无知识积累结论。后续（Planner 裁定，框架 §14）：检验"两个评估名额中保留一个不由 Agent 当前偏好决定的探索机会"——自主选择 / 文字指导 / Runtime 可执行配额 / 全随机，同池同预算，看 E、配对差与完整成本；不采用"先评估差异最大"（无验证过的距离度量），材料诊断暂缓，0 Slow；未启动。

**实现**：`evaluation/main_protocol_p4/batch_research_draft_construction.py`（草案流、DraftAdapter、每臂合同、固定候选臂、准备/恢复/报告），`batch_research_runtime.Adapter.check_compiled` 空钩子，`run_batch_research_v1.branch/resume_branch` 的 `adapter_factory`，Runner `--study draft_construction`；smoke `DRAFT_CONSTRUCTION_SMOKE_OK` + `FAULT_PATH_SMOKE_OK`，旧四 study smoke 与 26 单元测试通过。控制进程改为子进程只读确认 torch 环境（同进程 import torch 与 numpy MKL 撞出第二个 OpenMP 运行时）。未 git commit，未新增哈希，旧包目录零写入。
