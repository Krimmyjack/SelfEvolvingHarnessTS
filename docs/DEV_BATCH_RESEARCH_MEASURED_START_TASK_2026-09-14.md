# DEV-BATCH-RESEARCH-MEASURED-START：从当前批次的真实试验起点继续研究

日期：2026-09-14。状态：READY_FOR_EXECUTOR，未启动实验。
执行负责人：接收本任务的 Opus；方法与结果整合：Astra。
上位规则：AGENTS.md §5.9、§6–9；框架 §18。用户转发并要求执行后，按本任务连续完成实现、必要 smoke、真实运行和报告。本文本身不是实验已获执行、已经开跑或取得结果的记录。

## 0. 本包要回答什么

当前 Fast 能构造合法、有时有用的材料，但实际探索常局限于少数方案；上一包强制单参数修订没有改善交付。选择重放也没有找到可直接替换当前交付方式的规则。因此本包只改变一个主要机制：**把两个搜索槽用于确定性的随机初始化并真实训练，再让 Fast 根据这些当前批次的结果，自由使用剩余两个槽。**

工作流：完整批次 T → 两份冻结随机完整方案 → 各自共享重训并取得 C_A → Fast 观察、构造/修改、评估、停止、commit。所有材料仍共同训练一个 Consumer；不是单实体模型路由。

这两份是当前 Job 的新实验，不是 Source、历史最佳程序或 Skill。研究者不挑好分起点；Fast 可修改它们，也可完全离开它们。没有必须局部修订、必须更强、必须异质、必须评满的要求。

需要拆开的解释：
- 两份随机试验本身是否已经提供全部收益？
- 在相同两个已测起点上，Fast 后续两槽是否优于普通全局/局部随机搜索？
- 把前两槽交给随机初始化，是否优于让 Fast 自己从零决定全部四槽？

首轮 0 Slow；不形成或晋升 Skill。固定 H0 的构造、实际交付和完整费用是主要终点。RB 仅附带影子读数。

## 1. 接手与历史证据

完整读项目 AGENTS.md。重点复用：
- docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md §17–18；
- _scratch/dev_batch_research_paired_refinement/REPORT.md；
- _scratch/dev_batch_research_selection_replay/REPORT.md（已补 a85_g5 Menu 的 RB 反例）；
- methods/ttha/batch_research.py 与 batch_base/{policy,materials,train,feedback,budget,context,spec}.py；
- evaluation/main_protocol_p4/{batch_research_runtime,run_batch_research_v1,run_batch_research_roundtrip,batch_research_paired_refinement}.py。

历史依据不进入 Fast 请求。G3 的 D1、G5 的随机赢家及 E 排名不作为本包供给内容。本包与旧 DRAFT-CONSTRUCTION 的区别是“已完成真实训练的当前实验起点”，不是再给几份未评分草案；与 PAIRED-REFINEMENT 的区别是起点来源，且取消必须改一叶的第二槽合同。

输出目录：_scratch/dev_batch_research_measured_start/。
建议新增薄 study：batch_research_measured_start.py，并接现役 --study measured_start；不另建 Runner。旧包、其他执行线、AGENTS.md、参考仓库不写；不装依赖、不提交 git、0 新哈希。

## 2. 数据：下一个两截点均合格的完整人口组

TSLib Electricity，本地 electricity.csv，321 实体列，T=26304；不是 Monash。
解析路径：
/mnt/c/Users/辉/desktop/agent/shared_tsq_datasets/electricity/electricity.csv

按原始字符串 ID 排序。G5 已用，下一组 G6=[192:224] 在 a65 的训练段中实体298零尺度；因此按“两截点均合格、整组顺序取下一组”的规则采用 G7=[224:256]，不在组内换列。

G7 固定32实体：
~~~
30,300,301,302,303,304,305,306,307,308,309,31,310,311,312,313,
314,315,316,317,318,319,32,33,34,35,36,37,38,39,4,40
~~~

|Job|job_index|T 区间|C_A origins|C_B origins|E origins|
|---|---:|---|---|---|---|
|a65_g7|0|[16416,17088)|17088,17136|17184,17232|17280,17328,17376,17424|
|a85_g7|1|[21672,22344)|22344,22392|22440,22488|22536,22584,22632,22680|

公式 t=24*floor(a*26304/24)，L=192、H=48、训练672小时。两个完整工作负载不相交，同一人口、同一数据集，不当成独立来源。

Planner 只读两段 T 的 G6/G7 指定列核资格，未解析这些 Job 的 C/E：
- G6/a65：298 的 std=0；G6/a85 合格；
- G7：两个 T 均672行、非有限0、零尺度0，最小总体std分别18.1685744075、15.8529443953。
Runner 在配置冻结后重核 G7；再失败则本 Job ELIGIBILITY_FAILED，不继续找 G8、不改时间。另一合格 Job 可继续。
数据统一标 EXPOSED_DEVELOPMENT；不因换人口就称 untouched/Natural Final。

## 3. 公共 Consumer、材料、随机流

沿用现役数值：
- stride=1，433父对/实体、13856父对/Job；
- T-only 的实体 scaler；原/子各0.5质量，identity按既有实现；不对子材料重新归一化；
- 共享 MLP 192→128→64→48，ReLU；AdamW lr=.001、weight_decay=.0001、2000更新、父batch64；
- normalized MSE 实体/origin宏均值为主；原MAE/合法MASE辅助，保留难例和全部人口；
- 动作及参数域不扩：TimeMixup w={.10,.25,.50}，FreqMask/FreqMix mu={.05,.10,.20}，R/U，至多2步骤、8规则、现役first-match；
- 配对训练seeds：20261010、20261011、20261012；交付第一seed，不挑seed、不集成；
- augmentation seed=900090+job_index+100000*step；初始化/父batch随机流沿现役公式，材料先冻结再重复训练。

None 和 FixedMixup（均匀R-TimeMixup .25）为公共基线，各3seed，每Job只真实拟合一次。
环境沿用上一包解释器，单独子进程读取实际 Python/torch/device 后冻结；上包为3.13.5/2.12.1+cpu。不要在控制进程import torch触发已知MKL问题，不重配环境追速度。

## 4. 随机初始化：有分数，不挑赢家，不额外赠送预算

两Job均在任何拟合之前，只用T完成随机流编译：
~~~
proposal_seed(j)=2026092300+1000*job_index+j，j=0..63
沿j顺序调用现役policy.random_policy
按完整assignment排除None/Fixed和已收录的语义重复
保留前4份，记R1/R2/R3/R4
~~~

不按异质性、材料距离、代理分或已知效果过滤。生成器不是整个DSL均匀分布；如实保存完整policy与assignment。原有规范assignment比较足够，0哈希。

先对R1、R2各做3seed真实完整共享训练，获取当前C_A。它们不论好坏都作为初始化传入 F_start、Random-global、Random-local；不能因不如None就重抽，也不能先看R3/R4分数再换起点。

这三条方法共享同一份当前初始化的物理模型/评分，科学预算各记已用2个材料槽、6次拟合等价成本；共同缓存只是实验节约。F_free与Menu不看R1/R2内容及评分，除非各自独立构造并评估出相同材料。

64次不足4份记SUPPLY_INCOMPLETE，不换流；完整依赖该供给的三方法不冒充完成，F_free/Menu可继续。R1/R2真实拟合故障按§8处理，未完成起点不以默认材料替换。

Fast可见的起点内容：
- R1/R2的完整policy、assignment、真实C_A逐seed/origin/实体反馈、可用model/material引用；
- 明确它们是CURRENT_JOB_RANDOM_INITIALIZATION，已占用2/4搜索槽；
- 与None/Fixed同样可inspect_material、compare、作为父方案、直接commit。
这不是只塞一段总结文本；工具注册和评分绑定必须真实可用。不伪造Fast曾经提出/调用过这两份方案的轨迹，不混作历史经验。

## 5. 五方法与一个不追加拟合的起点参照

|方法|前两槽|后两槽|总搜索槽上限|
|---|---|---|---:|
|F_start|确定性R1/R2，真实训练后给Fast|Fast自由构造、可选派生、评估或停止|4=2+2|
|F_free|Fast自己决定|Fast自己决定|4|
|Random-global|同R1/R2|冻结R3/R4|4|
|Random-local|同R1/R2|两次随机数值参数邻居，依次读C_A|4|
|Menu|均匀U-Mixup .25、FreqMask .10|无|2|

另记 Start-only：在None/Fixed/R1/R2中按三seed C_A均值选择，初始化完成后冻结该选择，额外拟合/LLM为0。它是只做随机起步即停止的真实基线，不是Agent。

F_start与F_free共用：
- 当前完整T overview、Consumer说明、公共None/Fixed C_A；
- 现役七工具、evidence_roundtrip=True、语义去重；
- 已有父policy单参数派生助手可选，无第二槽强制派生，无“必须等反馈才能构造”的额外限制；
- 任何时刻可commit自己完整评估的方案，包括基线；不是Runtime改写为argmin；
- H0空历史、0 Source卡、0 Slow；两Job相互不传轨迹。

两 Fast 沿用上一包公共中性过程说明，工具说明只更新实际剩余预算及起点来源，不加算子/强度方向提示，不教它必须接受/优先修改随机起点：
“Use current batch evidence and actual paired C_A feedback to seek useful complete training material. State a brief testable hypothesis in the material rationale, name the comparison, and revise or stop as useful. Stronger, weaker, uniform and conditional constructions are all hypotheses, not defaults. Extra observations, edits and fits are not goals.”

F_start首请求已包含R1/R2真实结果，最多再评估2份；F_free首请求无它们，可评估4份。两者区别是前两槽的配置决定权和结果供给，不单独宣称“某一文字提示”的因果作用。

Random-global从None/Fixed/R1/R2/R3/R4中按C_A均值最小交付。
所有确定性控制器平局顺序：Fixed、None、R1、R2、之后本方法的评估顺序。Menu平局Fixed、None、MenuU、MenuFM。F_free/F_start实际commit仍归Fast，不强制遵守该平局规则。

## 6. Random-local：已有局部搜索的两步延续

初始化R1/R2后，对k=0,1依次执行：
1. 在自身全部完整评估方案中找有合法、未试数值邻居者，按C_A三seed均值选最好的父方案；平局按§5。
2. 复用现役enumerate_neighbours：default→规则顺序→step顺序→合法值顺序，只改一处w或mu；排除inactive叶子、同值和本方法已注册assignment。语义重复邻居保留第一个路径。
3. 用RandomState(2026092400+1000*job_index+k).randint(n_neighbors)抽一个，冻结选择并真实训练；不得读外臂分数。
4. 下一步可从新方案或旧方案继续。无需子方案C_A胜父才允许结束；最终从所有本臂已完成方案按C_A交付。
没有合法邻居则记录LOCAL_NEIGHBORHOOD_EMPTY并停止，不能改成临时全局重抽。
复用现役derive_policy与派生差异记录；不用人工挑“有希望”的邻居。

## 7. 实现保持薄层，特别避免预算与信息混淆

新增薄study和一个必要Adapter；复用现役shared/restore机制。
允许用控制器现有初始Candidate入口承载R1/R2，但必须在元数据/请求/报告里标记初始化角色；接口中的baselines名称不能把两次试验变成科学上的免费基线。

简单可行的计数方式：
- F_start控制器Limits.max_new_evaluations=2，另明示初始化已用2；
- F_free控制器Limits.max_new_evaluations=4；
- 报告分列initializer_evaluations、Fast_new_evaluations、总搜索材料数与逻辑拟合成本；
- resume保留起点、已耗槽、材料及原请求，不重新生成/拟合R1/R2，不恢复成默认两槽自由Fast。

两臂都可复用RefineAdapter的派生/去重部分，但不激活f_refine第二槽约束；不可通过改旧study全局常量实现。
F_free意外构造同R1/R2时按自己正常新评估计槽并真实拟合，不泄露另一方法的初始化分数；私有相同材料维持旧包的独立拟合计费。共享仅限公共N/Fixed及三条初始化方法明确共享的R1/R2。

起点供给不由Fast申请，因此提前停止F_start仍付出两个初始化槽；Start-only也包含该成本。对于自由Fast，在还没用满四槽时停止是真实节省，不替它补齐。

## 8. 总预算、顺序与停止

物理拟合上限：
~~~
每Job：
公共None/Fixed                    2×3 = 6
三方法共用R1/R2初始化              2×3 = 6
F_start后续                       2×3 = 6
F_free全程                        4×3 =12
Random-global后续R3/R4             2×3 = 6
Random-local后续                   2×3 = 6
Menu                              2×3 = 6
共48，两Job96；非科学故障同配置重试另2次，总尝试硬帽98。
Start-only、R0/RB影子比较新增拟合=0。
~~~

每个4槽搜索方法独立运行口径均为：6次公共基线+12次搜索拟合=18次；F_start不是“6次拟合”方法。另报缓存后的本包实际物理成本、完整方法成本及边际继续研究成本。

|资源|上限|
|---|---:|
|Job / 方法分支|2 / 10|
|Fast轨迹|4|
|每Fast逻辑请求 / 工具尝试 / 可纠正工具错误|16 / 40 / 2|
|全包逻辑请求 / HTTP尝试|64 / 128|
|已知输入+输出token停止边界|3,000,000|
|计划拟合 / 所有启动与重试合计尝试帽|96 / 98|
|实验墙钟，所有启动与暂停累计|10,800秒|
|单拟合 / 单HTTP|最多300秒且不超过包剩余墙钟|
|Slow / 实验Source / 新SHA / git提交|0|

顺序：
- 两Job先完成T资格、四份随机policy冻结；
- 各Job先公共基线，再其R1/R2；
- a65_g7：F_start→F_free→Random-global→Random-local→Menu；
- a85_g7：Menu→Random-local→Random-global→F_free→F_start。

Runner需正确处理本包的48/96拟合计划和每臂剩余槽，不能沿用旧72或默认两槽配置。实现/smoke墙钟单列，真实运行开始后的初始化、等待、重启/重试均算实验成本；不能换目录清零已经付出的成本。不得为对齐另外加一次真实复现拟合。

沿现役中转/model配置：cpa-grok-4.6、返回grok-4.6-build，0额外付费探活、0静默换模型；凭据不写报告/请求明文日志。
传输重试至多一次；未知费用保持UNKNOWN并阻止下一逻辑付费调用，适用的用户明确授权才可接受未知费用后恢复。标签必须仍扣留，沿现役labels_boundary_violations。
按冻结规则的授权恢复不自动等于协议违规；费用未知、标签阶段、是否改协议分别记录。
方法权限/契约失败按METHOD_INCOMPLETE处理，不冒充停止或基线；其他健康方法继续。全局费用/时间/账户/标签边界失败停止并扣留标签。不得因负结果重抽起点、换参数、换Job或追加seed。
初始R1/R2不完整时依赖它们的F_start/两随机方法不可有效完成，F_free/Menu健康部分继续，相关比较记UNKNOWN。

## 9. RB影子对照：只随本包计分，不改变决策

已有反例：
- G5/a65 Menu：两origin三seed一致反向，RB的E改善0.133736；
- G5/a85 Menu：同样符号结构，RB的E恶化0.005909。
因此本包不采用“出现冲突就相信最后一个origin”的规则，不等待触发后无限续包。

全部实际commit冻结后、C_B/E打开之前，按每臂自身完整评估集合冻结：
- R0：三seed、两个C_A origin宏均值argmin；
- RB：三seed、最后一个C_A origin宏均值argmin；
- 平局沿§5确定顺序；不覆盖Fast实际commit，不回传影子选择影响构造。

令d[r,o]=C_A(R0选中方案,r,o)−C_A(RB选中方案,r,o)。
不同方案且三个seed均d[r,0]<0、d[r,1]>0时记一致方向冲突；同方案记SAME_SELECTION，其余记MIXED。不使用“E最优方案”定义冲突，不加事后阈值。
后续只从常规E评分计算RB−R0、逐seed符号，所有Job/臂都报告。RB相对实际Fastcommit可另列，不能与RB−R0混淆。
它是固定候选集合上的选择重放，不能代表“若Fast过程中采用RB会构造什么”。不形成Skill、不把29个旧臂或本包10个臂当作独立Job复现；新读数与两个旧例分开标前瞻/回顾，不为累计正号换权重。

## 10. 一次必要smoke后连续运行

只验证新增差集，0真实拟合、0实验LLM：
- R1/R2真实Candidate、工具引用和C_A可见；F_free/Menu不收到它们；没有外Job/E字段；
- 起点按T确定性生成、未按分筛；初始化占2槽，F_start只能再2，F_free可4，缓存不增槽；
- 派生助手可选且无强制第二槽；初始方案可作为父方案，旧包行为不变；
- resume保留同起点、已用槽/费用和请求；旧信息墙检查复用；
- R0/RB在标签开启前冻结，tie/冲突定义只依赖C_A，实际commit不被覆盖。
旧公共控制器与受改路径回归一次即可，不另造大审计矩阵。

通过后：
1. 冻结实际配置/提示、人口、随机流、预算与环境；
2. T-only资格与随机方案冻结；
3. 各Job初始化、方法搜索与commit，仅T+C_A；
4. 所有方法完成/失败状态冻结，R0/RB影子选择冻结；
5. 一次打开C_B作延迟读数；
6. 全局冻结所有合格模型E预测，再开E真值评分；
7. 写报告、追加任务书实际收口，停止。

## 11. 首页只回答四问

1. **Fast有没有真正产生有用交付？** 对None、Fixed、Menu的配对E；区分公共基线、R1/R2复用、Fast新构造材料，第一seed实际交付单列。
2. **收益是谁带来的？** 主读数F_free−F_start；同起点Random-global/Random-local−F_start；Start-only−F_start。正值均表示F_start更好。同时报告完整成本，不以多观察、用了起点、写了修改解释当成功。
3. **真实过程改变在哪里？** 是否利用R1/R2结果、提出了什么实际材料、是否改变构造方向、哪次停止；列实际工具/材料证据，不将模型自述当因果证明。保留已评估未交付/本臂未探索差距，仍为回顾性有限集合诊断。
4. **RB有何附带读数，最大剩余问题是什么？** 不用它代替主要Fast结果；最多提出一项后续建议。

三seed只估训练重复，不等于三条Agent轨迹；两Job同源，不以0.005或0.01统一划“噪声/有效”。报告均值、配对SE、每seed差和真实损失尺度；不给小n自动显著门。缺失比较UNKNOWN。

判读：
- 只复用R1/R2且与Start-only相同：随机初始化提供了全部交付质量，Fast继续研究未显示增量，额外LLM成本保留。
- Fast新材料有收益但不胜同预算简单搜索：固定研究链可能有用，LLM独有增量未成立。
- 相对F_free与同起点随机对照改善，且完整成本可接受：有限的Fast机制线索；本包不自动进入M3。
- 同交付/更差/差异不清：据实收口，不追加直到正号。
- 起点、实际材料改变、更多观察本身不能晋升；如实区分“没构造”“没评估”“没交付”。

工件沿现有config/budget/result/trace/material/model/prediction格式，初始来源和成本写入现有记录；不新增平台。报告：
_scratch/dev_batch_research_measured_start/REPORT.md
完成后本任务书追加§12实际执行/协议/费用/结论/变更，停止，不提交git、不自动启动下一包。

## 12. 实际收口（执行者追加，2026-09-15）

**执行状态**：COMPLETE。10/10 臂交付；96/96 计划拟合全部成功，0 重试，108 次缓存命中（R1/R2 由三个初始化臂共用）；29 次逻辑请求 / 29 次 HTTP 全部返回 `grok-4.6-build`；已知 token 1,355,177 入 + 31,701 出；0 未知费用；包墙钟 4355 s（23:44:45→00:57:20），无暂停、无恢复。0 Slow、0 新 SHA、未 git 提交。

**协议状态**：`audit.json` 全部检查通过，`original_protocol_compliant=true`：随机流在第一次拟合前冻结；R1/R2 每 Job 在 `<job>_init/` 真实训练一次（各 6 次拟合），F_start/Random-global/Random-local 引用同一组模型；Start-only 在初始化完成时冻结；F_start 请求携带 R1/R2（`trained=true`、真实 C_A、`role=CURRENT_JOB_RANDOM_INITIALIZATION`、`overview.initialization` 标明 2/4 槽），F_free/Menu 的请求与目录不含 R1/R2；F_start 2 槽、F_free 4 槽如实计数；四条轨迹 0 拒绝、往返无越序；控制臂交付 = 冻结 argmin，Random-local 两步抽样可复现；R0/RB 影子在 C_B 打开前冻结且未改任何交付；168 个 E 格由原始预测与显式 G7 roster 重算，最大差 1.8e-15。

**结果**（三 seed E，正值 = F_start 更好）：
- F_start 两 Job 都交付新构造材料（a65 P4：R2 默认步骤只施于高 last168_std 实体；a85 UniFM02：去门控的均匀 FreqMask .2），C_A 上都超过起点 R2，但 E 上对 None 无收益（a65 −0.0017 [+−−]，a85 −0.0021 [+−−]），对 Menu 两 Job 3/3 更差（−0.0061、−0.0095）。
- 主读数 F_free − F_start：a65 −0.0034 [−−−]，a85 −0.0095 [−−−]——四槽全自由的 Fast 在两 Job 都好于"随机起点 + 两槽继续"。
- 同起点：Random-global − F_start a65 −0.0023 [++−]、a85 −0.0227 [−−−]（R3 全场最优）；Random-local − F_start a65 −0.0057 [−−−]、a85 +0.0043 [+−−]；Start-only − F_start a65 −0.0057 [−−−]（继续研究使交付变差）、a85 +0.0043 [+−−]（分不开）。
- 过程：两条 F_start 轨迹都真实读取并利用了 R1/R2（比较、检视材料、对 R2 做消融/剂量式构造，a85 用了一次派生形式），这是与 G3 草案被无视的实质区别；但两 Job 上集合内 E 最优都是起点 R1（分别好 0.0096、0.0100），因 C_A 不是最低而未被任何持有它的臂交付。
- RB 影子：10 臂中 0 个一致方向冲突、5 SAME、5 MIXED；MIXED 中 RB 两处均值更好但 seed 混合、三处更差（a85 F_free 3/3 更差 0.0095）；不改交付，不采用。
- 判读：固定研究链行为成立；Fast 新材料 C_A 有收益、E 无；不胜自由 Fast、简单搜索与菜单；一负一分不清，据实收口，不追加。

**唯一后续建议（提议，未授权）**：0 LLM、0 新拟合的分辨率测量——用 G3–G7 全部已拟合材料估计同材料 seed 间 E 方差与材料间 E 差的比例，以及 C_A 差距分档下 C_A/E 排序同向频率，给出 C_A 选择失去意义的经验差距；在此之前不再增加 Fast 侧构造/流程包。

**修改文件**：新增 `evaluation/main_protocol_p4/batch_research_measured_start.py`；`run_batch_research_roundtrip.py`（study 分发/配置/`smoke_measured_start`；两个按 study 属性读取的钩子 `after_baselines`/`before_labels`，旧 study 不受影响）；`batch_research_paired_refinement.refinement_events(baseline_ids=)` 可选参数（默认不变）。26 个控制器测试与 8 个 study smoke 通过；G5 `result.json` 重生成逐字节相同。

**未做**：未看 E 后补方案/补 seed/改提示/重跑；未重抽起点；未调用 Slow、未写 Skill；未启动下一实验。
