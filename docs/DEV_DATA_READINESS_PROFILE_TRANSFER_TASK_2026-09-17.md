# DEV-DATA-READINESS-PROFILE-TRANSFER

日期：2026-09-17。Planner：Astra；开发执行负责人：Opus。
状态：待用户转发执行；本文形成时0新拟合、0实验LLM。
输出根：_scratch/dev_data_readiness_profile_transfer/。
配套计划：[后续交付计划](DATA_READINESS_COMPLETION_PLAN_2026-09-17.md)。

## 0. 目标与授权终点

完成数据就绪“域内形成/选优 → 冻结目录 → 按画像加载 → 新批次Fast”的完整实测。
主要问题：仅根据当前T的结构画像匹配Skill，与直接给定域标签加载相比，
改变了什么决策、交付效用与成本。已知域Skill对无卡增量同时测量，不设“无卡先赢”的前置门。

用户转发并指示执行后，连续完成实现、smoke、Source、形成、Select、冻结、Test和报告；
不止于BUILD_COMPLETE。范围内收费运行按本书总帽执行，普通实现问题自主解决。
本文不表示Planner已启动实验，不授权超预算、换模型、增加域或打开密封材料。

Workflow仍是观察、假设、构造、实验、选择和停止的执行指导，不是固定算子表。
不改Consumer、评分、样本合法性、公共Fast提示/权限，不启动第三轮RD02修订。
原始跨作业轨迹只给Slow，不能直接给Fast或Router。

## 1. 复用与修复

完整阅读项目AGENTS.md及父包任务书，优先复用：
- methods/ttha/domain_skill.py：Skill、形成/选优/冻结、known_domain/profile_match。
- methods/ttha/batch_base/readiness.py：缺失数据、材料、训练、提交和评分。
- evaluation/main_protocol_p4/batch_research_data_readiness.py：Adapter、七工具、FAST_SYSTEM、Generic、阶段执行。
- batch_research_data_readiness_r2.py：两Job开发选优、预算、冻结和标签扣留。
- run_batch_research_v1.py、methods/ttha/batch_research.py：Fast与恢复路径。
- 父包dev_data_readiness_domain_skill_v1与dev_data_readiness_domain_workflow_evolve_r2只读。

允许一个新study入口和必要画像函数，不建新平台。保留其他执行线改动。
新增SHA/Hash预算0，不git commit，不覆盖历史工件。实际收口追加本任务末尾。

### 1.1 恢复未评估材料

先查另一个修复任务是否已完成，避免并发写同一函数。
未修则最小修复：已构造未评估材料保留ID，可继续evaluate，
但不能带入要求C_A的初始已评估候选列表；不得隐式评分、消耗名额或重新生成。
账本、旧轨迹、已完成候选保留；成功请求不重发。

零LLM/零拟合测试：build后中断→恢复→同ID evaluate/commit；
预算不重置，已开标签仍拒绝恢复。旧RD02_T4_Generic不补跑。
已有修复则仅复用及核差集，不重新组织故障审计。

## 2. 两域与固定批次

全部EXPOSED_DEVELOPMENT。Test是本次冻结策略的新计划评估，不称Natural Final，
不保证这些原始数据在整个项目中从未被看过。

### 2.1 RD01B：KDD with-missing的新固定人口

RD01B是RD01的新cohort，不是第三个独立数据域；旧RD01人口、Job映射、结果不变。
数据仍为data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip，
禁止使用NaN=0的without-missing缓存。

人口规则已固定：UID按字符串排序，从下标80起，排除父包旧RD01的32个UID；
长度>=9744；在下表五个T均满足>=336有限值、有限值std>1e-6、>=32合法父窗；
按顺序取前32个。只按T可训练性，不按缺失率排名或C/E收益。

Planner仅数值化这些T得到：
T172,T178,T190,T196,T202,T219,T22,T222,T223,T224,T225,T226,T227,T228,T229,T23,
T230,T233,T234,T235,T236,T239,T24,T240,T241,T243,T244,T246,T247,T25,T254,T256

| Job | 用途 | t | T平均缺失率 | 每实体最少合法父窗 |
|---|---|---:|---:|---:|
| RD01B_S1 | Source | 4560 | 15.178571% | 68 |
| RD01B_S2 | Source；第一开发选优Job | 5760 | 4.110863% | 89 |
| RD01B_V1 | 第二开发选优Job | 6960 | 9.263393% | 53 |
| RD01B_Q1 | Test | 8160 | 1.446243% | 139 |
| RD01B_Q2 | Test | 9360 | 1.274182% | 112 |

使用了全部计划T的资格信息，属于按可训练性筛选的设置，不称未筛选总体。
未读取本包C_A/C_B/E数值；T合格不保证评价覆盖合格。

### 2.2 RD02：原北京PM2.5全部12站

保留原CSV、全部12站和字段。复用R2的freeze/frozen.json中开发选中的RD02-C1-r2，
正文、Principles、scope不改。状态是开发选中、测试未证实增量，不因R2的E改选旧卡。

| Job | t | T | C_A | C_B | E |
|---|---:|---|---|---|---|
| RD02_Q1 | 11760 | [11088,11760) | [11760,11856) | [11856,11952) | [11952,12144) |
| RD02_Q2 | 12960 | [12288,12960) | [12960,13056) | [13056,13152) | [13152,13344) |

Planner只读T核对：Q1缺失1.847718%、最少132父窗；Q2缺失2.703373%、216；
均12/12合格。未读取这些Job的C/E数值。

### 2.3 通用几何与绑定

每个t：T=[t-672,t)；C_A origins=t,t+48；C_B=t+96,t+144；
E=t+192,t+240,t+288,t+336；L=192、H=48。KDD保留每条原始起始时间。

开跑前按现役实现复算T/人口/窗口数，与表不符先查绑定；核既有exposure和密封区声明。
碰明确密封区暂停该部分，不自行换人口或时间。
新增别名/Job需贯通scaler、cache、fit子进程、commit、标签阶段、resume，
不能只在父进程monkeypatch；旧Job逐项保持原义。

T不合格或C_A不可评分按原规则记录/跳该Job，不补批次。
每实体每评价块25%覆盖规则不变。Planner当前T检查不代表评价资格已通过。

## 3. 冻结公共环境

- 仅准备训练X和预测输入，不改原始y、合法父窗集合、样本权重和scaler。
- 整批材料共同训练一个MLP，每seed独立，不拼不同模型逐实体预测。
- 原MLP、AdamW、2000 updates、batch64和所有训练常数保持。
- 原10种readiness算子/参数、最多3步/8规则；允许统一或条件化方案。
- 七工具、FAST_SYSTEM、Generic、两公共基线完整复用。
- 每Fast最多16请求、24工具动作、2个新完整候选、2次现役工具参数纠错。
- Consumer seeds=[20260922,20260923,20260924]；不是3次独立Agent运行。
- Linear/Seasonal每Job各3fits，共用真实公共模型。
- 用与父包/R2数值一致的已安装解释器；R2实际用base Anaconda，
  不机械切到版本不同的conda project。核依赖记录，不安装升级追结果。
- 请求cpa-grok-4.6，返回必须grok-4.6-build，temperature=0；
  沿用本地代理/安全凭据，不写key、不加付费探活。模型不符停止受影响比较。

## 4. 数据就绪画像

显式readiness profile版本，不改旧增强profile的解释。复用T-only观察公式。
Router只收以下11字段：
missing_fraction、longest_gap、head_gap、tail_gap、lag24_corr、lag168_corr、
constant_run_fraction、robust_deviation_fraction、robust_z_max、
recent168_level_shift、recent168_volatility_ratio。

每字段给批内p25/median/p75和valid_fraction；全未知分位数保持null，不补0。
不先填补T再计算缺失统计。不新建embedding、估值器或特征学习层；
Fast原有观察权限不变。

Runtime完整保存身份用于复现，但Router输入不得包含：
来源名、站点/UID、路径、Job ID、绝对日期/行号、实体数N、n_valid计数、
合法父窗总数、raw finite_mean/std、任何C_A/C_B/E/排名/收益或Skill正文。
32和12不能成为识别域的捷径；valid_fraction只说明统计缺失。

兼容性保留task、Consumer结构/训练约定、readiness几何、L/H、小时采样和T长度；
去掉entities_per_batch强相等要求。禁止readiness匹配旧增强Skill。

每域一份DomainProfile，内部按时间保留三个开发T画像：
RD01B为S1/S2/V1；RD02为父包S1/S2/V1。
不压成单个不可追溯均值；不以Test画像更新原型。
资格预检读过的Test T也不进入Slow或参考画像。

## 5. Stage A：RD01B域内形成与选优

### 5.1 Source

S1、S2各跑一次无卡Fast，真实观察、构造、C_A、commit；
两条完成或明确失败后统一开C_B，不开开发E。
仅一条完整可用则以该条形成并标证据缩减；两条都无可用过程则不调用Slow。

### 5.2 Slow

一次提案，KEEP或最多W1/W2；最多一次格式纠错，科学KEEP/差效果不重抽。
复用readiness SLOW_SYSTEM、Workflow+Principles<=1200字符合同。
只读本包RD01B Source的T观察、工具/材料、C_A、commit后C_B、成本/失败。
不得传旧包E赢家、报告结论或Test画像。
要求可执行研究指导，不指定算子、不强制改数据/探索或必须出卡。

### 5.3 开发实测

固定S2与V1比较NO_SKILL、W1、W2。
S2无卡及公共基线引用本包Source；只新跑存在的W1/W2。
V1公共基线新拟合，分别跑无卡/W1/W2。
顺序：S2 W1→W2；V1 W2→NO_SKILL→W1；不存在的候选省略。
全部新开发臂尝试结束并commit冻结后统一开它们C_B，不开E。

每选项实际commit的C_B先3seed均值，再两个Job等权平均。
只选两个Job都完整可评分的选项；严格平局NO_SKILL→W1→W2。
S2已被Slow看到，这是训练/开发选优，不是独立验证。

选NO_SKILL则保留，不再抽卡、不把落选卡升格入目录。
缺完整选择条件记SELECTION_INCOMPLETE，RD01B采用无卡；
仍可做预定测试，但不能宣称该域Skill选优成功。

### 5.4 冻结目录

冻结RD01B选中卡（如有）、RD02既定C1、各域无卡状态、参考画像；
必须早于任何Test Router/Fast调用。RD02不再形成或选优。
目录不足两卡仍可继续测试，明确单卡适用/弃权，不称多Skill选择获证。

## 6. Stage B：一次画像匹配

复用SELECTED/ABSTAIN接口。scope改用readiness允许字段集，
不可沿用增强spec.OBS_FIELDS误拒/忽略合法缺失字段。
可加可选fields/compatibility参数或wrapper，不全局覆盖旧常量。

Router只看§4 query/参考画像、兼容性、匿名目录ID、冻结applicability_summary。
summary只含可观察条件，不载入收益、排名或推荐配方；可确定性过滤历史/来源字段，
但不重写Skill正文。匿名目录ID为C1/C2，映射仅在Runtime，无RD01/RD02串。
Q1用正序，Q2用逆序，两域同规则，不按真实标签排候选。

四个Test路由均先于Test训练及C_A读取，结果冻结到整个Job结束。
只选已冻结可路由Skill或弃权；证据字段须实际输入过；不改Skill、不生成配方。
scope由程序按当前T检查。路由不改Fast权限/预算。
Router解释不额外注入Fast；只加载原卡正文、本Job公共输入。

跨来源选择但兼容/scope匹配，可在本包实际运行，属于待测画像路由，
不事先授予跨域Capability，也不改全项目Shared Capability规则。
真实域标签一致率仅诊断：同域不等于最有效，异域也不能未评分就判无效。

格式/传输故障记ROUTING_FAILED，不能伪装ABSTAIN；不重抽直到匹配满意。
没有可路由卡时免调用，记NO_CANDIDATE_SKILLS。

## 7. Stage C：四Test五臂

| 臂 | 内容 | 用途 |
|---|---|---|
| F0 | 同Fast无卡 | 公共研究起点 |
| F_generic | 原Generic | 通用建议对照 |
| F_known | 真实域标签加载冻结卡，缺卡则无卡 | 已知域Skill作用 |
| F_match | 模型按画像选卡或弃权 | 完整匹配系统 |
| Random | 原random_policy两方案、按C_A提交 | 同预算普通搜索 |
| 公共基线 | Linear/Seasonal各3seed | 固定处理 |

Job顺序：RD01B_Q1、RD02_Q1、RD01B_Q2、RD02_Q2。
臂顺序：
- RD01B_Q1：F0、F_known、F_match、F_generic、Random；
- RD02_Q1：F_generic、F_match、F_known、F0、Random；
- RD01B_Q2：F_known、F0、F_generic、F_match、Random；
- RD02_Q2：F_match、F_generic、F0、F_known、Random。

Random policy seeds按上述Job依次：
(2026091801,2026091802)、(2026091803,2026091804)、
(2026091805,2026091806)、(2026091807,2026091808)，不看结果重抽。

### 7.1 等价执行复用

在Test效果产生前，根据实际渲染Knowledge正文、scope、hook和公共初始输入判等价：
F_match与F_known相同，或其中一个等同F0时，只运行最早一条Fast；
另一个标IDENTICAL_EXECUTION_REFERENCE，引用同轨迹/交付。
不同guidance不能因最后效果相同而合并；Generic不事后合并。
Router成本仍计F_match；共享执行不是独立LLM复现。
材料模型缓存沿用既有语义，所有缓存/alias分别报告。

这种配对复用避免把相同指导两次LLM采样差写成路由增量；
即使无一等价，下列预算仍覆盖所有臂。

### 7.2 信息墙

Fast只看本Job T和本轨迹C_A；所有跨Job知识、画像和路由已冻结。
预测输入按现役origin时序权限读；未来真实y仅由外部评分器读取。
四Job所有计划臂完成或明确未完成之后统一开C_B，
再全局冻结E预测，最后统一开E评分。Q1后不修卡/画像，不按结果停止Q2。
C_A不可评分按原规则跳Job；其他块缺失各自记缺失，不填0/换人口。

保留固定线性服务输入的影子评分，0新拟合，不参与选卡或交付。
主读数是准备管线效用，影子是固定服务输入时训练材料差，不能互相替代。

## 8. 一个总账本与阶段上限

| 阶段 | 正常拟合上限 | 逻辑请求上限 | token分配 |
|---|---:|---:|---:|
| RD01B Source两Job | 24 | 32 | 700,000 |
| Slow含最多一次格式纠错 | 0 | 2 | 250,000 |
| 开发选优 | 36 | 80 | 1,600,000 |
| 四Test Router | 0 | 4 | 50,000 |
| 四Test五臂及公共基线 | 144 | 256 | 3,400,000 |
| 合计 | 204 | 374 | 6,000,000 |

全包另有2次同配置故障拟合重试，物理尝试总帽206；HTTP尝试帽748。
实验墙钟8小时，自首次收费/拟合开始沿用现役账本。
阶段余额不转移，恢复不重置；故障2次是全包共享，不是每阶段2次。
所有Slow/Router/Fast/纠错/失败都计费，不另开付费接线包。

算术：Source=2×(6公共+6候选)；
Select=2×6(S2新卡)+6(V1公共)+3×6(V1各臂)；
Test每Job=6公共+5×6新候选=36。
alias、提前停止、等价执行可节省，不可用于追加Job/候选。

报告本包增量、历史形成和独立部署成本。公共基线不重复乘臂数；
RD02形成成本按父包另列，不宣称取得卡免费。
未知usage保持未知；预留估计不是保证上界。

## 9. 停止与恢复

沿用labels-withheld。未知usage停后续收费，本任务不预先接受未来未知费用，
旧事故授权不自动覆盖新事故；零成本诊断可继续。
每阶段最多一次用户授权恢复，标签须仍扣留，请求/状态/候选保持，
accept-unknown-usage不能解除标签边界。

首次收费前普通实现bug自主修复。真实运行后实质协议变化需明确amendment，
不能覆盖旧尝试。全包终止需评分时，先冻结永久放弃的臂、确认无在途调用，再finalize。
已开标签不resume；方法未完成不伪装基线交付。

预算不足记BUDGET_INCOMPLETE，不借其他阶段帽。
KEEP、NO_SKILL、合法提前停止不是故障，不追加采样。

## 10. 最小smoke

只测新风险，复用已通过控制：
1. 恢复build未evaluate计划；开标签后拒绝。
2. 新别名/cut贯通子进程、cache、标签；旧Job不变。
3. 缺失画像null/分位数、12/32人口；Router原始请求无身份/N/C/E。
4. readiness scope可用；跨profile拒绝；弃权、scope失败、技术失败分开。
5. 等价执行事前判定，Fast只收费一次、Router单列。
6. 合成完整链路的冻结/E屏障和预算不重置。

一次综合smoke足够，不为通过项数重复旧矩阵。
真实接线使用Source首个正常拟合，失败计总帽，不另付费探针。

## 11. 结果与固定判读

每Job报告commit、每seed E、均值、配对SE和成本，正差表示研究臂更好：
- Δ_skill=E(F0)-E(F_known)；
- Δ_routing=E(F_known)-E(F_match)；
- Δ_system=E(F0)-E(F_match)；
- Generic、Random、Linear、Seasonal分别与F_match比较。

跨域不直接平均不同尺度原始NMSE差：先以同Job Linear E作分母算相对差，
再域内Job等权、两域等权，同时保留原值。分母0/非有限则该相对项缺失，
不添eps追结果；缺Job标PARTIAL并给实际分母，不当完整四Job主结果。
seed不是独立Job，小样本SE不是未来泛化证书。

必须分开报告：
- 路由选择/弃权/scope/故障和真实标签一致率；匹配合理不等于有效。
- 单卡目录只能测适用/弃权，不是充分的多卡选择能力。
- 与known同加载则Δ_routing=0，Router额外费用保留。
- 工具行为、材料、评估集合、commit各自变化。
- 未构造、已评未交付、后续块反号分别分析，不全归因于噪声。
- RD01B新人口只支持该cohort；两来源均为空气质量，不称跨行业。
- 原T缺失率与实际入训X缺失率/窗口数并列，明确完整y筛选范围。
- 准备管线效用与固定服务输入影子分开。

出口：
框架跑通但无收益 → FRAMEWORK_COMPLETE + NO_UTILITY_EVIDENCE；
known有收益、match保留 → 路由保持线索，不称路由提高质量；
known有收益、match丢失 → 匹配/scope线索，不追测试改卡；
仅单批/seed混合 → 局部线索；
无卡/Generic/Random同样好 → 额外价值未建立；
不完整/不可评分 → 明列可比项，不把缺失写胜负。

不以更保守、多观察、出两卡、认对域标签替代效用；
不因结果自动追加R3、算子或评价信号改动。

## 12. 交付与停止

一个REPORT.md首页回答：
1. 学出了什么可执行guidance，目录有几张卡，哪域仍无卡？
2. 匹配是否改变实际加载、行为与材料？
3. 系统对无卡/Generic/Random的效用和完整成本？
4. 匹配对已知域加载增加或损失了什么？
5. 最大未解问题；仅提一个下一步，不执行。

复用result/config/frozen/routing/轨迹/模型/预算，不建平行证据平台。
给可复算结果表与简短METHOD，讲清研发Opus、系统内Slow/Router/Fast、
两层切分和权限。原报告/CLAIMS不改，勘误集中附本报告，不开新审计包。

本包未授权之后的Natural Final、新域、第三次修订或论文性能宣称。
完成实测即停止；有收益写收益，无收益也交付完整原型和结果。

## 13. 实际收口（2026-09-17，Opus）

主报告 `_scratch/dev_data_readiness_profile_transfer/REPORT.md`，方法说明 `METHOD.md`，数值 `result.json`，协议核对 `protocol_checks.json`，事件与修复 `package_config_amendment_1.json`。

**执行 COMPLETE，协议合规，完成即停；出口 FRAMEWORK_COMPLETE + NO_UTILITY_EVIDENCE。**

- **成本**：拟合尝试 110/206（2 次失败，1 次重试），请求 81/374，已知 token 3,057,037/6M，0 次未知 usage，账本约 1.31 h。
- **恢复修复**：`resume_branch` 与 `run_job` 可以恢复“已构造未评估”的方案；smoke 已验证；本次运行中没有触发恢复。
- **RD01B**：
  - RD01B_S1 的 C_A 不可评分（实体 7、8 无观测）；
  - Slow 只凭 RD01B_S2 提出 W1、W2；
  - RD01B_S2 的 C_B 不可评分（实体 25 无观测），开发选优为 SELECTION_INCOMPLETE，RD01B 无卡；
  - 仅作描述：V1 上无卡提交的 C_B 最低（1.053）。
- **目录**：只有 RD02-C1-r2 一张卡。
- **路由**：RD01B_Q1、RD02_Q1、RD02_Q2 弃权（两个 RD02 本域批次都弃权），RD01B_Q2 跨域选中 RD02 卡；标签一致率 0/1。
- **Test**（相对同 Job 的 Linear E，域内等权后两域等权；正值表示后者更好）：
  - Δ_skill +2.5%（仅 RD02_Q1 的 +0.197，2/3 seed，约 1 个 SE）；
  - Δ_routing −4.6%（RD02_Q1 弃权丢失 −0.197；RD01B_Q2 跨域加载 −0.108，3/3）；
  - Δ_system −2.1%；
  - F_match 对 Generic −3.7%，对 Linear −3.5%，对 Seasonal −4.3%；对 Random 为 PARTIAL。
- **技术事件**：
  - RD01B_Q1 的 Random 因拟合进程 OMP 冲突（denoise_savgol）失败，未补跑、未替换；
  - 驱动进程在目录冻结前同类崩溃，没有在途付费调用，修复后续跑；
  - worker 改为先初始化 OpenMP 再加载 torch，数值中性核对通过。
- **下一步**（未执行）：0 LLM，测量 readiness 画像在同域跨时间与跨域之间的距离分布。

