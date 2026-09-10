# EXTERNAL_REVIEW_GPT6PRO_2026-09-09

**SelfEvolvingHarnessTS 外部独立评审｜第一部分：A、B、C**  
审阅日期：2026-09-09。仓库快照：本次上传的 `9.8.zip`。  
本部分完整回答 A–C，并核对全部 11 条内部读数；D–G 尚未在本文件中展开。本文是外部意见，不是执行指令、晋升记录或路线决定。

## 执行摘要

1. **[中]** 逐序列选择是条件性合理的研究方向，但现有证据不足以把“更多特征和卡片”作为 9/15 的主要押注。
2. **[高]** Consumer 的 25% 差距混入了训练材料变化，不能归因于参数共享本身。
3. **[高]** 70–80% 是训练模型路径的绝对分量份额，不是“单条清洗向其他序列外溢”的实证。
4. **[高]** 严格同知识 51 对的 MATERIAL 计数存在算术矛盾，须先对账再用于功效判断。
5. **[高]** 当前最需要的是部署一致性证据和依赖结构明确的效应范围，而不是把行数、暴露数或材料阈值当统计精度。

## 审阅范围与证据边界

按要求先读 `AGENTS.md`，再读置顶状态、决策记录、Slow 设计笔记 §13、已冻结的第一包任务书、指定结果文档及代码。未以大型阶段归档账本作为入口。只对允许的 development JSON 做离线读取和算术复核；未执行项目实验、拟合 Consumer、调用项目 Agent、编辑仓库或更新知识。

**跳过内容：**密封/held-out 数据与结果、TARGET_HELD_IN 相关工件、Yahoo 密封集，以及被标记为 sealed/UNREAD 的样本或结果。边界规则本身属于可读协议，不把其中出现的禁区名称当成可读取数据的授权。可选数据资产审计属于混合清单，其密封/未读条目未用于推断；本报告的数据身份以已开放的指定文档为准。压缩包中未找到 `docs/EXTERNAL_BRIEFS_GPT6PRO_2026-09-08.md`。

本报告区分：**公开文献事实**、**仓库中可复算的事实**、**文档报告但缺少完整逐项收据的事实**、以及**外部推断/建议**。外部论文均给出可核验的 arXiv ID、DOI 或会议信息；参考文献表给出具体核查位置。文中行号均指本次 ZIP 内文件，而非后续 GitHub 版本。

三个全篇适用的口径：

- `raw/identity` 不是“将原始 NaN 直接交给 Ridge”。它已经经过 evaluator 的 `_linear_integrity` 基础完整性处理；这里研究的是**相对这条基础路径的额外准备价值**。出处：`AGENTS.md:647–678`；`docs/R4D_A_ACTION_DECOMPOSITION_RESULT_2026-09-08.md:25–32`。
- 旧 SEQ 结果以 delayed 为主；第一包把 **origin 当次决策、事后且不回传的评分**定义为主终点，把原程序在 +48 的表现定义为持续性副终点。这不是把 Support 偷渡回部署。出处：`docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_TASK_2026-09-08.md:92–100`。
- 截至该快照，第一包是已授权、待执行状态；DEV-SEQ-4 的完整主终点仍 UNKNOWN。以下不预判尚不存在的结果。出处：`docs/DECISIONS.md:12–15`；`docs/STATE_ONE_PAGE_2026-09-03.md:3–31`。

---

## A. 逐序列预处理选择的可学习性与所需规模

### ① 文献与公开事实

**没有可直接移植的“达到 N 条序列就稳定超过固定选择”的门槛。** 下列论文的学习单位、反馈预算和部署方式并不相同；它们能提供尺度参照，不能共同构成一条适用于本项目的样本复杂度曲线。

| 工作及可核验身份 | 实际学习规模与信息 | 与本项目的距离 |
|---|---|---|
| **FFORMA**，Montero-Manso et al.，International Journal of Forecasting，2020，DOI **10.1016/j.ijforecast.2019.02.011** [A1] | M4 的 **100,000 条序列**；趋势、季节性、相关性、熵、稳定性等特征，学习候选预测模型的组合权重。 | 学的是预测模型组合，不是固定 Consumer 下的清洗选择。利用序列可见历史构造训练验证，不能把全部 M4 成绩解释为严格未见序列上的跨任务转移。 |
| **FFORMS**，Talagala et al.，Journal of Forecasting，2023，DOI **10.1002/for.2963** [A2] | 真实参考序列、合成序列及频率分组；随机森林把时间序列特征映射到预测方法。公开稿使用 M1/M3 等参考材料，也利用 M4 可见训练部分构造合成材料。 | 支持“特征可帮助选模型”，不支持“20 个身份的重复窗口足够”，也不是全程与目标序列历史完全隔离的元学习例证。 |
| **Time series model selection with a meta-learning approach; evidence from a pool of forecasting algorithms**，Barak et al.，2019，arXiv **1908.08489** [A3] | NN5 共 **111 条序列**；正文报告 80%/20% 训练测试划分，因而训练侧约 **89 条**；使用 24 个候选特征，包含趋势、季节强度、ACF、熵、曲率等。报告若干元学习器胜过最佳固定预测器。 | 是“训练序列少于 100”的正向先例，**不是总数据少于 100、跨域稳定胜出**的证据。论文还使用重复交叉验证和特征筛选；其描述不足以让我确认所有筛选均严格嵌套于外层训练折。 |
| **auto-sklearn**，Feurer et al.，*Efficient and Robust Automated Machine Learning*，NIPS 2015 [A4] | **140 个分类数据集、38 个元特征**；元学习提供初始化，随后继续在新任务上用验证反馈做 Bayesian optimization 和集成。 | 其成功包含“新任务还允许搜索”的价值，不能直接转述成无反馈 Fast 的成功。 |
| **BoostClean**，Krishnan et al.，2017，arXiv **1711.01299** [A5] | 在 12 个数据集上，从检测/修复候选中利用预测验证信号优化组合。 | 12 是实验任务数，不是“12 条元训练样本就学会跨任务清洗”。清洗搜索与 Consumer 组合也不同于固定 Consumer 的部署策略。 |

**对“少于 100 条成功”的直接回答：有小训练集的正向报告，但没有在本次核查中找到足以保证本项目这种设定的小样本稳定成功证据。** [A3] 是可核验的小训练集例子；不能因此否定小样本，也不能把一次同域分割的胜出当成所需规模的上限。另有 FFORMS 引述的 NN3 小样本研究，但我未取得足够的一手实验材料，因此不把二手引用升级为本报告的承重证据。

**CleanML 的结论更接近“效用依赖条件”，不是“效用已可预测”。** Li et al.，ICDE 2021，arXiv **1904.09483** [A6]，覆盖 14 个真实数据集、5 类错误、7 类模型。缺失值实验中填补也可能不如删除；其对照不是未经处理的 NaN。异常清理常无明显帮助，验证选择能降低但不能普遍保证消除负作用。论文没有给出可直接用于本项目的“公开特征→清洗收益符号”泛化定律。

**训练时利用即时反馈、部署时不再需要反馈：有相邻先例，但必须认清迁移的对象。** Ren et al.，*Learning to Reweight Examples for Robust Deep Learning*，ICML 2018，arXiv **1803.09050** [A7]，用干净验证信号学习训练样本权重，最后部署训练好的预测模型。被带到部署的是模型，而不是一个已证明能迁移到新域的清洗选择器。Duan et al.，**RL²**，arXiv **1611.02779** [A8]，确实学习如何探索新任务，但新任务交互仍输入奖励，不能作为“新任务全程零反馈”的同义先例。

因此，对“训练中学会何时该试、何时该停，随后迁移到无反馈部署”的回答是：**原则上可行的学习范式存在；训练期改进搜索，与冻结后能够替代搜索，是两个需要分别验证的主张。** 上述文献没有替本项目证明第二个主张。

### ② 推断

**判断：条件性合理；作为当前 9/15 的唯一或主要押注，不合理。** 这里反对的是在未证明部署口径收益前，继续主要依赖增加静态特征、加卡和条件收窄；不是断言逐序列条件化不可能。

首先，**效应重测不稳定，不等于状态到效应的映射不可学习**。设某个公开状态为独立变化的二值量，程序收益恰好随该状态正负变化，则相邻时点收益相关可以为零，当前状态却能完美预测收益。低重测相关主要打击“这条 UID 过去被帮过，所以以后继续帮”的记忆，不自动打击“当前窗口出现条件 X，所以现在做 P”的策略。

其次，**事后 oracle 空间不等于可学习空间**。以共同部署动作集、相同总体与同一收益口径定义：

$$
H_{\mathrm{oracle}}
=E[\max_a g_a]-\max_a E[g_a],
$$

$$
H_X
=E[\max_a E(g_a\mid X)]-\max_a E[g_a]
\le H_{\mathrm{oracle}}.
$$

这是决策问题的区分，不是新的实验指标提案。前者允许偷看每个动作的结果，后者只能利用部署时可见信息。即使前者为正，只要所有动作的条件期望排序基本不随 X 改变，后者仍可能接近零。当前 0.07–0.13 还是 **P 与 identity 两选一**的事后空间，不是完整工作流空间，更不是 LLM 的专属空间。

第三，**“27 个特征未赢”只否定了已检验的规则族和分割**。它不能证明非线性组合、机制性观测、当前可执行动作的影响范围都没有信息。反过来，小样本里有一个 AUC 较高的特征，也不能证明它能优化全人群总收益：识别服务侧严重伤害，不等于识别训练路径与服务路径相加后的净收益。

第四，当前最紧的规模瓶颈不是“100 个行样本太少”这么简单，而是**独立的效应翻转情形、训练环境和合法决策证据太少**。20 条相同序列在 5 个单元出现，不会自动变成 100 个独立元任务。若各序列的最优动作始终相同，再增加同类序列也只能更精确地证明固定动作强；真正有用的是可见条件能够解释、且在不同独立单元重现的帮助/伤害异质性。

**需要多大规模？本次证据无法给一个可信的点数。** 已核验的文献从约 89 条训练序列的同域报告，到数千/十万条序列或百级数据集的研究都有；差异来自效应幅度、动作集合、元特征、额外反馈和验证方式，而非一个共同阈值。对于广泛可迁移的结论，百级以上的独立序列及多个不同训练/机制环境比当前重复窗口更有说服力；这只是证据覆盖要求，不是“增加到某个 N 必然成功”的预算请求。

### ③ 对仓库的核对

**A-1｜准备的收益与选择的增量，应分开保留。** `docs/DEV_SEQ1_PER_SEQUENCE_2026-09-07.md:179–207` 的两次运行相对 raw 的 delayed 均值为 **+0.061771、+0.068199**；相对固定程序为 **+0.004461、+0.010890**。这支持准备本身有价值；但既不能把全部 +0.06 归给逐序列选择，也不能只凭现有重复把 +0.010890 判为“纯噪声”。后者超过 MATERIAL，是否可重复则是另一问题。该形成段实际为 **3 个单元 × 10 条序列 × 2 次运行**，不是当前 20×5 全课程的统一读数。

**A-2｜“相关约为零”需要指定相关系数。** `docs/R4A_B_PERSISTENCE_AND_VARIANCE_RESULT_2026-09-07.md:59–72` 中 Spearman 为 **0.065363、0.017177、0.108911**，但 Pearson 为 **0.363516、0.281393、0.286426**。可写“相邻窗口的收益排序稳定性弱”，不能不加限定地写“收益完全无相关”。该审计涉及 80 个 UID、21 个位置、4 个块；不能与当前 20×5 混成一个分母。

**A-3｜六个设置的正确判词是“未建立超过固定选择”，不是“全都更差”。** `docs/R4E_CHANNEL_IDENTIFIABILITY_RESULT_2026-09-08.md:117–141` 的规则是单特征分位阈值，训练折选择方向和阈值；五格退化为 always-P，另一格均值比 always-P 高 **+0.000548**，远低于 MATERIAL，且只在一折胜出。应把“不足以胜出”与“严格落后”分开。

**A-4｜0.83 的可识别性目前比简报表述窄。** 同文 `:103–113`：W2 的 pooled Consumer、delayed、服务 context 路径中，局部鲁棒 z 峰值与严重伤害的定向四折均值 AUC 为 **0.826536**；方向是 **z 较低更容易受严重伤害**。这是按 4 个块留一，不是按 20 条序列留一。此外，代码 `evaluation/main_protocol_p4/audit_r4e_channel_identifiability.py:111–123` 以 **`abs(ctx) > CTX_ACTIVE_EPS`**筛选该通道样本，即用结果分解量定义活跃子集。它不自动等价于部署前可观察的“实际修改了服务窗”。若要部署这一规则，必须先证明一个不看损失的替代筛选条件，而不能携带这个事后筛选器。 此外，代码 `:140–148` 的 `_pick_best` 按跨折 AUC 读数选择特征/目标；最佳关系仍有探索性筛选，不能把四折结果直接当成预先指定该特征的独立确认。

**A-5｜当前最大的未对齐项在执行链，而不只是特征。** `methods/ttha/online_loop.py:763–780,884–953` 显示默认路径按选择/候选顺序探测，并让 Support 准入结果取得交付权；identity 也不清空候选池。新第一包明确改为真实 Fast-only 的交付模拟，见任务书 `:71–100`。在它完成前，无法把旧运行时的全部收益归于冻结 Fast 的可学习能力。

### ④ 对决定的建议，以及仍不知道的东西

**结论。** 当前证据支持“数据准备有作用”，但尚不支持“在这套小课程上，持续知识更新能让无反馈 Fast 稳定优于固定策略”。同样，没有证据证明所有条件化方法必败。

**对 9/15 的建议。** 保持第一包作为既有部署假设的检验，不把它改写成新特征竞赛。决定时区分三种结果：冻结 Fast-only 若连校准后的确定性选择都未稳定超过，应降低当前逐序列选择路线作为主贡献的权重；若 Fast-only 能赢，但更新知识的额外价值仍无证据，应保留“自主准备/候选供给”的较窄主张，不提前写成自进化；若优势只存在于有 Support 的旧运行时，就应承认收益依赖在线验证，而不是靠更多同类卡片解释过去。

以上是决策解释，不要求增加实验参数、数据或预算，也不替内部团队决定停哪条线。

**仍需内部补充：**第一包真实结果；每个数值对应的 roster/训练几何和动作空间；0.826536 子集能否由纯动作事实重建；当前 P/identity oracle 是否在相同风险约束下仍有余量；以及严格同知识配对的逐项收据。现有材料不能回答“再补多少序列一定足够”，我不建议用一个没有依据的样本数为路线续期。

---

## B. “Consumer 结构决定数据准备的价值与风险”是不是新结果

### ① 文献与公开事实

**广义结论已知，具体机制量化可能有价值；当前还不是干净的新因果结果。**

Montero-Manso 与 Hyndman 的 *Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality*，International Journal of Forecasting 2021，arXiv **2008.00444** [B1]，并不支持“异质序列必然应局部建模”。它区分有限容量模型与足够表达力的全局方法，并说明全局方法不必要求序列同质。Hewamalage、Bergmeir、Bandara 的 *Global Models for Time Series Forecasting: A Simulation Study*，arXiv **2012.12485** [B2]，显式改变生成机制、长度、异质性和模型复杂度；结论同样不是局部或全局无条件占优。

CleanML [A6] 的异常值实验显示模型相关差异，例如依赖样本距离的 KNN 对异常清理更敏感；但缺失值部分各模型总体趋势相近。它支持“清理效用不是数据的静态属性”，**不支持单凭模型类别就预言某个 25% 的幅度**。

“改变训练数据会改变其他样本的预测”有成熟概念：Koh 与 Liang，*Understanding Black-box Predictions via Influence Functions*，ICML 2017，PMLR 70:1885–1894 [B3]，研究训练点对测试预测的影响；Ghorbani 与 Zou，*Data Shapley: Equitable Valuation of Data for Machine Learning*，ICML 2019，PMLR 97:2242–2251 [B4]，按训练数据对模型效用的边际贡献估值。它们都不能直接给本项目三格分解赋予“70% 因果中介比例”的含义。

**本次没有核实到一篇直接完成“只清洗某条时序训练序列，测量共享预测器对其他序列的外溢，并对照局部模型”的一手论文。** 这限定了我的检索结论，不构成“该现象首次发现”的证明。影响函数和数据估值是应比较的相关概念，而不是能替代本项目识别设计的现成解释。

### ② 推断

**新颖性判断：部分已知。** 建议把想说的东西拆成三个层级。

| 主张层级 | 当前判断 | 可防守的措辞 |
|---|---|---|
| 清洗效果随 Consumer 而变化 | 已知，单独新颖性弱 | 本项目在特定天然缺失预测环境中观察到强烈的 Consumer/训练设定依赖。 |
| 准备收益和伤害主要来自重新拟合训练模型，而不是当前服务窗改动 | 属于可解释的既有机制，但具体路径审计可有贡献 | 在明确的三格、固定路径分解下，训练模型路径占绝对分量的大多数。 |
| 清洗一条序列，通过共享模型伤害其他序列 | 概念上已知；本项目尚未验证这个具体干预 | 暂不作事实主张，需要“单条训练序列修改、其他不变”的干预才成立。 |

**何时局部可能胜出？** 在容量受限的共同参数模型中，序列规律相反、局部历史有足够信息且共享表示又无法表达这种差异时，共享约束会引入偏差；局部拟合可能更好。但短序列、可共享规律和较高估计方差又可能使全局更好。一个推理例子是两组 AR 系数符号相反的序列：不带分组信息的单一系数可能相互抵消，而每组自己的系数不会。这个例子说明可能性，不能解释本项目的实际幅度。[B1–B2]

**25% 是否符合已知模式？** “有可能出现”符合；“因此正好证明共享有害”不符合。当前不但改变了系数是否共享，还把预测器从“用另外 20 条序列训练”换成“用被预测序列自己的历史训练”。这改变了可用信息、训练分布和有效样本数；局部模型赢，可能首先是拿到了更相关的历史。

**70–80% 不等于共享外溢。** 即使完全没有跨序列参数共享，清理一条序列的历史后重新拟合自己的模型，仍然会有训练侧作用。仓库的 per-channel 结果也保留了很大的训练侧分量。因此，训练介导和跨序列共享是两个正交问题。

此外，三格分解天然带有**路径顺序**：

$$
L_{pp}-L_{rr}=(L_{pr}-L_{rr})+(L_{pp}-L_{pr}).
$$

先更换模型、再更换服务 context 时，第二项包含“在准备后模型条件下改 context”的效果。缺少 $L_{rp}$ 就不能知道换一个顺序，分量是否相同；也不能排除模型与服务变换的交互。这里的绝对份额是有用的诊断，但不是无条件唯一的因果贡献划分。

### ③ 对仓库的核对

**B-1｜25% 的面发生了压缩。** `docs/R4D_B_PERCHANNEL_RESULT_2026-09-08.md:16–25`：两面合并的 Static sMASE 从 **1.494869 降到 1.119398**，相对下降约 **25.12%**；仅 delayed 则从 **1.527425 降到 1.246216**，相对下降约 **18.41%**。不能把 25% 不加限定地用作 delayed 主终点改善。

**B-2｜共享与材料确实混杂。** 同文 `:22–25` 明写 per-channel 使用自身 10 个训练窗，pooled 使用块内另外 20 条序列的 200 个窗。`docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md:360–365` 已对这点作出较新、更谨慎的更正。应采用较新的限定，而不是让旧结果文档中的强措辞重新升级为结论。

**B-3｜严重伤害的变化比摘要复杂。** 同文 `:89–97` 的四组计数依次是 **3/28、9/31、8/60、6/75**（per-channel/pooled）。比率范围约 **0.080–0.290**，不是四格都在 1/5–1/10。与此同时受损占比没有同等幅度下降，准备的平均收益也减少。可以说严重尾部明显改善，不能说风险全面消失、或收益毫无代价地保留。

**B-4｜70–80% 的分子分母可定位。** `docs/R4D_A_ACTION_DECOMPOSITION_RESULT_2026-09-08.md:20–34,88–94` 使用：

$$
\frac{\sum_i |\mathrm{route}_i|}
{\sum_i |\mathrm{route}_i|+\sum_i |\mathrm{ctx}_i|}.
$$

MAD 的 support/delayed 约为 **0.768/0.804**，W2 约为 **0.714/0.703**。这是**绝对分量份额**，不是净收益百分比；route 与 ctx 可能抵消。更重要的是，`L_pr` 换的是基于程序处理后训练材料拟合的模型，并没有只修改被评分的那条序列。

**B-5｜现有逐序列 evaluator 的“逐序列”不是一种任意混合清洗后联合训练。** `evaluation/main_protocol_p4/per_sequence.py:24–48,181–191` 与 `docs/DEV_SEQ1_PER_SEQUENCE_2026-09-07.md:88–111` 使用 K+1 结构：每个 distinct program 对应一套训练模型，服务时按序列选择模型/程序。给序列 i 指派 P，并不等价于只清理 i 的训练样本、然后让所有序列重新共享一个混合训练模型。因此现有任务内“给 i 换程序”的后果，不能重述为“i 的清洗向 j 外溢”。第一包 `:24–30` 明确保留这一结构。

### ④ 对决定的建议，以及仍不知道的东西

**结论。** 目前最稳妥的定位是：“相同额外准备程序的净效用及严重尾部，对训练/Consumer 设定高度敏感；在现有三格诊断中，训练模型路径占主要绝对分量。”不要把它改成“我们发现了参数共享引发的清洗外溢，并证明局部 Consumer 优于共享 Consumer”。

**最小补充实验建议：增加一个匹配训练材料的桥接对照，而不是大因子实验。** 不改第一包，把它作为后续可选的 development 机制检验：以已用 per-channel 的自身历史窗为共同、合法训练材料，增加 **pooled-on-the-same-material** 一臂，与原 per-channel 比较。两者使用相同预测对象、时间边界、训练窗集合、表示、基础填补和 P；区别只在参数是否被约束为共享。

更严格地说，把同一个联合训练目标写成：

$$
\frac1n\sum_i\left[\frac1{m_i}\sum_j
\ell(y_{ij},x_{ij}^{\top}\beta_i)+\lambda\|\beta_i\|^2\right].
$$

局部臂允许各 $\beta_i$ 不同；共享臂施加 $\beta_i=\beta$。这样损失归一化与正则化含义明确，不能只说“两个 sklearn 调用都传同一个 alpha，所以已控制”。原 pooled-on-other-series 可以保留作背景第三点，但不能替代这条匹配比较。

桥接臂需要同时有 identity 与同一个既有 P，核心读数应是清洗增量的差中之差：

$$
\Gamma=
\{L_{\mathrm{shared},P}-L_{\mathrm{shared},I}\}
-\{L_{\mathrm{local},P}-L_{\mathrm{local},I}\}.
$$

只比较两个 Static 不能回答清洗敏感性。这里需要确认现有 per-channel 收据可复用且训练窗不包含对应预测未来；如果不可复用，就不能把它说成只需一个新拟合。桥接实验只识别**这套共同材料上参数共享的影响**，不证明任意 Consumer 或其他数据域。

缺失的第四格 $L_{rp}$ 是检验路径交互的另一问题；单条训练序列清洗是检验外溢的又一问题。只有坚持更强的中介/外溢主张时，才需相应补证据；不要把三种问题捆成一个必须大改的包。

**审稿人最可能的三个问题与应准备的回答：**

| 审稿人问题 | 应准备的回答及其边界 |
|---|---|
| 你改了共享方式，还是让模型获得了被预测序列自己的历史？ | 明确承认旧 25% 是混合对照；展示匹配材料的共享约束实验，报告原始损失和准备增量，不能只展示有利的 Static 差距。 |
| 70–80% 为什么叫单序列外溢或因果归因？ | 给出三格公式、绝对值分母、训练集合和路径顺序。若没有单条训练序列干预，就撤回“外溢已验证”；若没有第四格，就只称固定路径分解。 |
| 是否只是一个不合适的 pooled Ridge 给清洗制造了空间？ | 在相同合法校准条件下，展示强固定策略与匹配 Consumer 的收益/风险前沿。若更合适的 Consumer 吸收了大部分准备收益，必须把它写成结果，而不是移除该基线。不能用当前一个域推出通用框架优势。 |

**仍需内部补充：**逐臂训练窗身份、变换与标准化顺序、正则化归一化、fit 缓存的语义身份，以及各严重伤害格的共同分母。现有材料未告诉我匹配后共享约束到底贡献了多大差距。

---

## C. 小样本、强依赖下的配对分析计划与可辨别效应

### ① 文献与公开事实

MacKinnon、Nielsen、Webb，*Cluster-Robust Inference: A Guide to Empirical Practice*，arXiv **2205.03285** [C1]，强调簇大小、不平衡、杠杆和相关结构影响推断；不存在一个保证安全的通用簇数。按行数套普通标准误，或把 bootstrap 次数增加到很大，都不能弥补独立信息不足。

Cai、Canay、Kim、Shaikh，*On the implementation of Approximate Randomization Tests in Linear Models with a Small Number of Clusters*，arXiv **2102.09058** [C2]，提供少簇符号随机化的算法和条件。其“少到五簇”不是“任意五簇都能做 5% 双侧显著检验”：非随机化双侧检验在 5% 水平要到至少六个有效符号簇才有非平凡拒绝能力，且仍须满足簇间独立/适当极限对称等前提。

Ibragimov 与 Müller，*t-Statistic Based Correlation and Heterogeneity Robust Inference*，Journal of Business & Economic Statistics，2010，DOI **10.1198/jbes.2009.08046** [C3]，允许用少量组估计量推断，但依赖组估计量近似独立正态等条件；不是给三个重叠时间块算均值后套 t 分布即可。

**分层贝叶斯收缩是另一种明确模型假设的途径，不存在一个“达到几簇后自动可信”的下限。** 下述模型与先验敏感性方案是本报告建议，而非上述频率学派论文的结论。后验区间可在极少数据下计算，但可计算不等于数据已识别组间方差。

### ② 推断

**建议以“有限课程的完整配对效应 + 明示假设的敏感性分析”为主，以满足条件的簇级符号翻转为条件性推断；不把复杂模型作为默认精度补丁。**

必须先分开三个目标：

| 目标 | 允许说什么 | 不能混入什么 |
|---|---|---|
| 固定本次序列、窗口、Consumer 和已实现输出 | 本批课程的配对差、受损、成本与逐单元影响是实际描述量。 | 不需要用虚假的 IID 标准误为这批已观察值“增加可信度”。 |
| 固定这些任务，重新调用随机 Fast | 推断调用随机性下的期望表现，需要可交换的重复和相同可见输入。 | 同一确定性程序被重复评分，不构成新的调用随机性样本。 |
| 推广到新序列、时间或训练环境 | 需要对应层级的独立样本或明确生成模型。 | 20 个 UID 并不能自动支持新时间、新环境、跨域泛化。 |

共享拟合也需要谨慎解释：**同一个确定性已冻结模型被多人读取，不自动产生额外随机噪声；但当目标包括重采样训练材料或共同时间环境时，模型和环境是共同来源。** 不能机械地说所有共享 fit 的行只能算一例，也不能机械地说各 UID 不同所以独立。必须先写明究竟固定了什么、想推广到什么。

#### 推荐方法、最低簇数与失败条件

| 方法 | 本项目中的适当用途 | 最低独立簇数：可诚实给出的答案 |
|---|---|---|
| 簇级置换/符号翻转 | 有真实簇级随机分配时检验尖锐零假设；或在原假设下整个簇贡献向量允许独立符号翻转时检验中心。 | 计算上少簇也能做；非随机化 5% 双侧检验的离散分辨率要求至少 **6 个非零有效簇**。这只是必要条件，不是有效性或功效保证。 |
| 按 UID 整条轨迹聚类 bootstrap | 把时点与训练环境固定，且 UID 之间确有独立性依据时，作条件敏感性分析。 | 无统一下限；当前 20 个 UID、效应集中在少数 UID 时，不建议把 percentile 区间当唯一主证据。三个时间块不能靠 bootstrap 变成大量独立时间环境。 |
| 分层贝叶斯收缩 | 报告共同均值、UID 异质性及明显的先验敏感性；对异常长尾可用厚尾误差。 | 没有数据无关的最低值；只有三个时间层级时，时间方差和跨层协方差高度依赖假设。应报告多套合理先验，不得把收缩后的窄区间写成“数据已经精确证明”。 |

作为可选的贝叶斯敏感性模型，可以在固定本次时间点和拟合的条件下写 $d_{it}=\mu+u_i+\gamma_t+\epsilon_{it}$，其中 UID 效应收缩、时间效应加零和约束、误差使用厚尾分布；报告不同合理收缩强度下的 $\mu$，而非只给一套后验区间。它针对固定三时点的条件均值。只有三个时间点时，不建议同时自由估计时间、共享 fit 和交互的多个方差，再声称已获得新环境的不确定性；若必须对这些随机来源外推，当前数据很可能不能区分它们。此项不是第一包必须增加的建模任务。

原始配对差的均值为零，并不足以保证符号翻转精确。还需整个向量在零假设下具有相应对称性；仅凭“两臂做过相同任务”不等于处理标签随机分配。把所有行各自随机翻号，尤其会破坏同 UID、多窗口与共同路径依赖。[C1–C3]

若确实只能辩护 **3 个独立时间簇**，双侧枚举最小 p 值为 $2/2^3=0.25$；4 簇为 0.125，5 簇为 0.0625。若有依赖需合并，情况更弱。这个离散限制不妨碍观察很大的有限课程效应，但妨碍声称用这种检验已获得 5% 水平的跨时间证明。

#### 暴露子集如何预注册

暴露应按**打分前冻结、两臂共同的可见输入**定义，例如“同一公共 Context 与工具状态下，既定 applicability 会使候选卡呈现给 Fast”。应同时保存未暴露分母。不要用“B 最后选了不同程序”“部署后受到伤害”或“分解后的 ctx 非零”来定义确认性暴露子集。

建议同时冻结三个读数：全人群均值是主终点；预定义可达子集的均值是机制副终点；从呈现、调用、选择到交付的实际触发率是过程终点。实际检索/交付可能被 treatment 改变，其条件子集只能作为过程诊断，不能无条件称为子群因果效应。多个特征阈值可探索，但探索完不能把其中最好的重新命名为预注册。

对共同权重，恒等式是：

$$
\Delta_{\mathrm{all}}=p\Delta_E+(1-p)\Delta_{\neg E}.
$$

只有已论证未暴露面没有效应——包括没有历史、候选顺序、共享更新等间接影响——才能简化为 $p\Delta_E$。所以 1/40 暴露时，全人群 +0.005 对应子群 +0.20 的说法是**附加零外溢假设下的稀释换算**，不是功效定理。一个暴露对象足以贡献很大的已观察均值，但不能识别该类对象的稳定平均效应。

### ③ 对仓库的核对

#### C-1. 实际配对工件是什么结构

直接读取了：

- `artifacts/main_protocol/dev_seq3_guidance_delivery__g2.json`；
- `artifacts/main_protocol/dev_seq3_guidance_delivery__contrasts.json`；
- `artifacts/main_protocol/dev_seq3_exposure_check__g1.json`；
- `artifacts/main_protocol/dev_seq4_revision.json`。

g2 的实际决策位于 `validation.units[*].sequences` 与 `followup_units[*].sequences`。每行包含 `position`、`series_uid`、`arm`、`origin`、`delayed_origin`、`delayed_gain`、`chosen_candidate_id`、`candidate_programs`、`probes`、`deployed_label`、`history_episodes_in`、`fast_view_bytes` 与 `edited_text_is_in_this_sequences_view` 等字段。

因此应按 **(position, series_uid)** 配对不同臂，不能仅按 UID 配对。u9/u10/u11 共 60 对；A/B 各 60 条，合计 120 次决策。3 对呈现了新卡、57 对未呈现。`history_episodes_in` 是数量，`fast_view_bytes` 是长度；**两者都不是渲染文本逐位相同的证明**。

#### C-2. 51 对的 MATERIAL 计数存在可定位矛盾

从逐决策 JSON 重算，并与 contrasts 的 `q3_and_q4_per_group.g2.exposure_split.same_knowledge_pairs` 对照：

| 未暴露的 57 对 | 重算值 |
|---|---:|
| 均值 B−A | −0.0083910526 |
| 样本标准差 | 0.0574367556 |
| 最小 / 最大 | −0.313522 / +0.157526 |
| 恰为零 | 46 |
| 非零 | 11 |
| `abs(B−A) >= 0.005` | **9** |

但 `docs/DEV_SEQ3_GUIDANCE_DELIVERY_RESULT_2026-09-08.md:321–338` 声称：剔除 6 对历史文本不同者，剩余严格同知识 **51 对，41 对为零，6 对达 MATERIAL**；被剔除的 6 对中 **5 对为零**，均值 −0.000665。

这几句不能同时为真：剔除的 6 对至多只有一对超过阈值，所以 9 个达阈值者至少应剩 8 个，不可能只剩 6 个。若采用文档的剔除均值，则唯一非零剔除项约为 **−0.003990**，本身不到 MATERIAL，故剩余应仍为 **9 个**。

**本次没有取得逐项的“渲染文本相等”收据，不能独立重建哪 6 个配对被剔除。** 我没有按收益反推出成员来代替历史审计。结论是“计数矛盾须核对”，不是私自修订严格 51 对名单。下面的 51 对标准差和尺度计算仅在文档关于剔除集合的描述成立时给出。

#### C-3. 可辨别效应的敏感性范围，不是假精度

依照上述条件性剔除描述，51 对的均值为 **−0.009300**，样本标准差约 **0.060717**。仅为展示乐观尺度，假设这些差值独立、近似正态且该方差可迁移，则：

$$
\mathrm{SE}_{\mathrm{IID}}\approx0.00850,\qquad
\delta_{80\%}\approx(1.96+0.842)\frac{0.060717}{\sqrt{51}}
\approx0.0238.
$$

这里 80% 是假设性功效、5% 是双侧检验水平，**0.005 则是效应量门槛**。这不是已证明的实际功效，也不是给现有差值颁发置信区间。

再作仅含 UID 内相关、平均每 UID 2.55 行的简化设计效应敏感性：$DE=1+(2.55-1)\rho$。

| 假设的 UID 内相关 ρ | 仅供尺度解释的有效 n | 正态近似下 80% 可辨别全人群增量 |
|---:|---:|---:|
| 0 | 51.0 | 约 0.0238 |
| 0.1 | 44.2 | 约 0.0256 |
| 0.3 | 34.8 | 约 0.0288 |
| 0.5 | 28.7 | 约 0.0317 |
| 1.0 | 20.0 | 约 0.0380 |

这些不是对 ρ 的估计，也没有包含跨 UID 的共同时间/训练影响、不均衡簇大小和长尾问题；不能把 **0.0238–0.0380** 当成有保证的上下界。更保守的课程级依赖下，现有材料甚至不足以校准一个可靠的有限 MDE 上限。

另一个直接诊断：在可复算的 57 对里，分别删除一个 UID 后，均值范围约为 **−0.01177 至 −0.00298**。这不是置信区间，但说明少数 UID 对结论量级的影响很大。当前分布主要是零加少数较大差值，不宜把其标准差当稳定的“LLM 噪声地板”。

若全人群目标仍是 +0.005，零外溢假设下，不同暴露比例对应的子群增量为：

| 暴露比例 | 达到全人群 +0.005 所需的子群均值增量 |
|---:|---:|
| 1/40 = 2.5% | +0.20 |
| 5% | +0.10 |
| 10% | +0.05 |

以上只是均值换算。不能再把稀释后的门槛和单次子群差值相比较，就宣布有统计把握；也不能因为暴露少就先验宣布收益不可能有价值。

#### C-4. 另外两处与统计协议直接相关的实现风险

`dev_seq3_guidance_delivery__contrasts.json` 的 `q3_and_q4_per_group.g2.outcomes.independence` 报告 `distinct_series=10`、`distinct_windows=34`，但同一 outcomes 块对应的是 **20 个 UID、60 个 (UID, origin) 配对**。这些数字可能来自 formation 的统计口径；需要标注作用对象，不能直接作为 outcomes 的独立样本数。

`evaluation/main_protocol_p4/per_sequence.py:241–245` 会把缺少 assignment 键与明确选择 identity 都走到零收益；`:250–274` 遇到 UnitFault 后可以对剩余可读序列求均值。第一包 `docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_TASK_2026-09-08.md:155–165` 则要求主总体完整、缺失必须保留 UNKNOWN。**这是包装层必须拦截的潜在口径不一致，不是已证明过去主结果受污染。** 当前 g2 的 60 对均可读，不能据此臆造一个已经发生的删样本偏差。

### ④ 对决定的建议，以及可实现的分析协议

**结论。** 当前可辩护的最强读数是有限课程的完整配对效果和机制事实；不能把 51 对当成 51 次独立实验，不能把“6 对超过 MATERIAL”解释成假阳性率，也不能用未显著超过零证明 ±0.005 内等价。

**建议：把以下协议作为独立 0-LLM、0-fit 审计，而不是重写第一包。** 它读取结果与收据、产生审计报告，不改变候选、知识或原始主终点。已经看过的 DEV-SEQ-3 数据只能做回顾性审计；对后续未出结果的读数，才可以把分析选择称为预注册。

#### 第一步：冻结 estimand、数据白名单与缺失规则

**假设：**第一包明确的 roster、两个重复运行和主要评分面能够从收据确认；任何缺失不等于 identity。

保存一个小型分析配置：允许的文件路径、合法 development unit/UID 集合、主要比较、面、固定权重、MATERIAL、统计检验水平、依赖目标、暴露谓词版本。不得递归读取任意 artifacts；白名单之外不读。当前全人群是固定单元内 20 条序列，先按单元等权求平均，再按单元等权汇总；每个 run 单独报告，运行均值加报，不能选好的一次。

#### 第二步：建立配对及因果角色表

**假设：**能够从同一 Task/Consumer/训练定义中识别真正的同任务配对。

每个配对保留：`run, unit, origin, uid, task, consumer, training_geometry, baseline_signature, selected_program, delivered_program, history_hash, visible_view_hash, exposure, gain, fault`。其中哈希缺失则记 UNKNOWN；不能用字符数代替文本哈希。程序 ID 不同未必是语义不同，程序 ID 相同也不能代替参数、数据边界和训练身份核验。

冻结两种不同角色：全人群对照用于效用；严格同知识、同可见历史的配对用于调用变异诊断。后者不覆盖前者，也不自动是知识的随机化因果实验。

#### 第三步：先做无需分布假设的核算

**假设：**配对完整，gain 的符号一致，MATERIAL 使用的是绝对差阈值。

输出配对差、各 unit 均值、全人群均值、暴露/未暴露的加权恒等式、零值数、正负数、达 MATERIAL 数、最大正负差、严重伤害、覆盖、逐 UID/逐 unit 删除后的均值变化。禁止删去极端负值后把剩余当主结果。UNKNOWN 个体和可读子集分开显示；主总体不完整时主读数为 UNKNOWN。

#### 第四步：先定义依赖图，再选推断方式

**假设：**连接依据是当前推断目标下的共同随机来源或抽样来源，不是“共享任意配置字段”。

同 UID 重复窗口要保留为一个轨迹；时间窗重叠按实际区间计算，不按 unit 编号猜测；共同训练材料/随机拟合、反馈历史延续和同步环境影响另行记录。固定数据条件下的 Fast 调用变异，与对新训练环境的泛化，分别给出依赖分组。若只能辩护一个共同课程簇，则不报跨环境标准误。

若有充分依据将 UID 轨迹视为独立，仅作“条件于这三个时点和这些拟合”的 UID 聚类敏感性。若跨 UID 时间影响仍重要，需以更粗的块作为主限制；不要从 20、3、60 中挑能给最小 p 值的那个。

#### 第五步：有前提才做簇级符号翻转

**假设：**预先说明真实随机化零假设，或说明簇贡献向量的符号不变性；只假设总体均值为零不够。

令固定总体权重为 $w_i$，簇贡献为 $S_g=\sum_{i\in g}w_i(d_i-\delta_0)$。对整个簇统一乘以 ±1，枚举 $2^G$ 种符号，统计量取 $|\sum_g S_g|$。保持原来的总体权重，不能为了凑“簇均值”悄悄改变估计对象。多次随机枚举不能创造比真实符号空间更细的证据。

没有足够有效簇时仍可报告精确 p 值及其最小可能值，但必须写“本设计无法以该检验在该水平拒绝”，而不是写“方法等价”或“不存在效应”。要反演区间，还需明确允许反演的中心/移位模型；不能把尖锐零假设检验自动变成对任意异质平均处理效应的精确区间。

#### 第六步：可辨别效应按情景扫描，不解一个样本数

**假设：**同知识配对在相同部署链上对未来变异有参考价值；这是假设，不是既有事实。旧 Support 链与新 Fast-only 不同，故只能把旧分布当压力情景。

情景至少分为：固定任务的独立调用；按 UID 保留完整向量的依赖；共同时间/课程影响较强的依赖。只有对应收据可用时，才重采样实际簇；严格 51 对成员未核实前，不对它伪造簇结构。

在每个可执行情景下，用中心化残差作压力底座，对全人群增量扫描 **0、0.005、0.01、0.02、0.04** 等尺度，并扫描既定暴露宽度及其邻近范围。只在预定义暴露位置注入增量 $\delta/p$，不把一个只作用 5% 决策的机制模拟成给所有决策加同一个常数。报告零效应拒绝率、正效应检出率、方向错误率及风险门通过率；也报告大尾部事件频率/幅度改变时结论如何变化。未观察到的罕见严重伤害率不能从大量零差值可靠估出。

这些情景给的是**假设性功效曲线**。若没有可信的独立簇或噪声生成情景，就返回 `POWER_NOT_IDENTIFIED`，不能输出一个看似精确的“需要 N=…”答案。

#### 可直接落成审计脚本的伪代码

```python
# 只读 ZIP 与明确给出的开放 development 收据；不 import 项目运行器。
# report_dir 位于仓库外。任何 required 字段缺失都不补成零。
MATERIAL = 0.005

payload = read_allowlisted_json(
    zip_path,
    "artifacts/main_protocol/dev_seq3_guidance_delivery__g2.json"
)
units = list(payload["validation"]["units"].values())
units += payload["followup_units"]

rows = {}
for u in units:
    assert u["position"] in allowed_development_positions
    for r in u["sequences"]:
        k = (u["position"], r["series_uid"], u["arm"])
        assert k not in rows, "重复记录不能靠覆盖消失"
        assert r["series_uid"] in expected_roster[u["position"]]
        rows[k] = r

pairs = []
for position, uid in expected_pair_keys:
    a = rows.get((position, uid, "knowledge-frozen"))
    b = rows.get((position, uid, "slow-candidate"))
    if a is None or b is None or not finite_gains(a, b, face="delayed"):
        pairs.append(unknown_pair(position, uid))
        continue
    assert_same_task_and_training_identity(a, b, receipts)
    pairs.append({
        "key": (position, uid),
        "delta": b["delayed_gain"] - a["delayed_gain"],
        "edit_present": b["edited_text_is_in_this_sequences_view"],
        # 长度、Episode 条数不是文本相同的证明。
        "same_visible_text": equality_from_receipt_or_UNKNOWN(
            receipts, position, uid
        ),
    })

main = whole_population_mean_or_UNKNOWN(pairs, frozen_weights)
not_present = [p for p in pairs if p["edit_present"] is False]
assert len(not_present) == 57  # 此断言仅针对当前 g2 历史工件
stats57 = descriptive_stats(not_present, threshold=MATERIAL)
assert stats57["zero_count"] == 46
assert stats57["material_count"] == 9

# 检查文档断言是否能同时成立；不按结果构造被剔除的 UID。
assertion_check = check_exclusion_arithmetic(
    original_material=9, excluded_n=6, excluded_zero=5,
    claimed_remaining_material=6
)
# 应返回 INCONSISTENT；禁止吞掉此告警。

if missing_text_equality_receipts(receipts):
    strict51 = "NOT_RECONSTRUCTIBLE_FROM_THIS_ZIP"
else:
    strict51 = [p for p in not_present if p["same_visible_text"] is True]
    audit_document_counts(strict51)  # 不一致即停止其确认性统计

write_descriptive_report(main, stats57, assertion_check)

# 以下只能使用冻结的主终点与该终点的完整配对，不能自动用旧 delayed 代替。
for scenario in preregistered_dependency_scenarios:
    if not scenario.identification_assumptions_supported(receipts):
        write_status(scenario, "INFERENCE_NOT_IDENTIFIED")
        continue
    clusters = scenario.cluster_keys(primary_pairs, receipts)
    write_influence_diagnostics(primary_pairs, clusters, frozen_weights)
    if scenario.allows_cluster_sign_flips_under_null:
        result = exact_cluster_sign_test(
            primary_pairs, clusters, frozen_weights, null_effect=0.0
        )
        write_result_with_discrete_resolution(result)
    if scenario.supports_power_simulation:
        write_power_sensitivity_grid(
            primary_pairs, scenario,
            effects=[0.0, 0.005, 0.01, 0.02, 0.04],
            exposure_masks=frozen_exposure_scenarios,
            preserve_whole_cluster_vectors=True,
            include_rare_tail_stress=True
        )
```

这段伪代码刻意让历史 g2 审计与未来主终点分析分开；不能因为旧工件键名是 `delayed_gain`，就把第一包的主终点换成 delayed。

#### 当前规模下不能无额外假设得到的结论

| 不应声称的结论 | 具体缺口 |
|---|---|
| 全人群真实增量稳定处于 ±0.005 内，因此方法等价 | 未建立足够窄且有效的等价区间；不显著不是等价。 |
| 51 对给出可靠的 LLM 假阳性率或通用噪声地板 | 计数尚矛盾、历史成员收据缺失、重复结构与执行链不同。 |
| 三个时间点的显著均值证明跨时间、跨域有效 | 少量且可能依赖的时间/训练环境不能支撑该外推。 |
| 暴露子群平均效用已被稳定识别 | 1–3 次暴露可能只涉及同一 UID；子群总体和外溢假设未识别。 |
| 低收益重测相关证明所有条件化学习不可行 | 时序持久性与当前状态可预测性不同。 |
| 0.83 AUC 已经给出可部署的高收益决策器 | 通道、面、事后子集与总效用不一致，且尚无独立部署验证。 |
| 选择/交付一致率 45.7% 就是 LLM 因果贡献率 | 候选生成、探测排序、运行时筛选没有被分别干预。 |
| 某次 zero-feedback 模拟成功已经证明 Source→Target 知识积累 | 当前仍是已曝光 development；没有对应的冻结迁移对照。 |

**仍需内部补充：**51 对的逐项可见文本/历史文本相等收据；fit 与训练材料的语义身份；每次 LLM 随机种子实际控制的层级；新第一包完整、未回传的 origin 评分；预定义暴露谓词与明确的泛化目标。补不齐时应降低主张，不用更复杂的统计软件掩盖缺口。

---

## 参考文献及核查位置

以下只列本部分实际用于论证的一手材料。arXiv 首次年份与期刊发表年份可能不同，已尽量区分；没有把未核验的 2025–2026 新论文补进 A–C 充数。

**[A1]** Montero-Manso, P.; Athanasopoulos, G.; Hyndman, R. J.; Talagala, T. S. **FFORMA: Feature-based forecast model averaging.** *International Journal of Forecasting*, 36(1), 86–92, 2020. **DOI: 10.1016/j.ijforecast.2019.02.011.** 核查：作者公开论文稿的数据构造、M4 规模与特征表；不要把预测模型组合等同于清洗选择。

**[A2]** Talagala, T. S.; Hyndman, R. J.; Athanasopoulos, G. **Meta-learning how to forecast time series.** *Journal of Forecasting*, 42(6), 1476–1501, 2023. **DOI: 10.1002/for.2963.** 方法 FFORMS。核查：作者公开稿 §4.1 参考数据与模拟序列来源、特征及预测方法选择；公开稿日期为 2022 年，发表年份为 2023 年。

**[A3]** Barak, S.; Nasiri, M.; Rostamzadeh, M. **Time series model selection with a meta-learning approach; evidence from a pool of forecasting algorithms.** 2019. **arXiv:1908.08489.** 核查：§4.1 的 111 条 NN5；§4.3–4.6 的 24 特征、筛选及 80/20 分割；Table 4 的固定预测器与元学习器结果。少于 100 指训练分割，不指总语料。

**[A4]** Feurer, M.; Klein, A.; Eggensperger, K.; Springenberg, J. T.; Blum, M.; Hutter, F. **Efficient and Robust Automated Machine Learning.** *Advances in Neural Information Processing Systems 28*, **NIPS 2015**. 核查：§3.1，140 个参考数据集、38 个元特征、元学习初始化后继续 Bayesian optimization；§6 评价。

**[A5]** Krishnan, S.; Franklin, M. J.; Goldberg, K.; Wu, E. **BoostClean: Automated Error Detection and Repair for Machine Learning.** 2017. **arXiv:1711.01299.** 核查：候选检测/修复、验证信号驱动的组合与 12 数据集评价；不是跨域冻结元选择器的直接证明。

**[A6]** Li, P.; Rao, X.; Blase, J.; Zhang, Y.; Chu, X.; Zhang, C. **CleanML: A Study for Evaluating the Impact of Data Cleaning on ML Classification Tasks.** **ICDE 2021**. **arXiv:1904.09483**, 核查版本 v3。核查：实验设置、§V-B/Table 11 缺失值对照、§V-C/Table 12 异常值与模型差异。没有把 NaN 未处理数据作为所有实验的共同对照。

**[A7]** Ren, M.; Zeng, W.; Yang, B.; Urtasun, R. **Learning to Reweight Examples for Robust Deep Learning.** **ICML 2018**. **arXiv:1803.09050.** 核查：小型干净验证集提供的元梯度，作用于训练例权重；部署对象是最终预测模型。

**[A8]** Duan, Y.; Schulman, J.; Chen, X.; Bartlett, P. L.; Sutskever, I.; Abbeel, P. **RL²: Fast Reinforcement Learning via Slow Reinforcement Learning.** 2016. **arXiv:1611.02779.** 核查：循环策略输入包括动作、奖励和终止标记；新任务内仍有反馈。

**[B1]** Montero-Manso, P.; Hyndman, R. J. **Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality.** *International Journal of Forecasting*, **2021**. **arXiv:2008.00444**, 核查版本 v3。核查：§2–3 表达能力与泛化关系、§5 的异质性和模型容量实验。

**[B2]** Hewamalage, H.; Bergmeir, C.; Bandara, K. **Global Models for Time Series Forecasting: A Simulation Study.** **arXiv:2012.12485**，首次 2020，核查版本 v3（2021）。核查：§2 控制数据复杂度、异质性和数据量的实验设置，以及结果/结论；本报告未依赖未经核实的期刊 DOI。

**[B3]** Koh, P. W.; Liang, P. **Understanding Black-box Predictions via Influence Functions.** **ICML 2017**, *PMLR* 70:1885–1894. 核查：训练样本权重变化如何影响测试预测；影响函数不是大幅清洗干预的自动精确归因。

**[B4]** Ghorbani, A.; Zou, J. **Data Shapley: Equitable Valuation of Data for Machine Learning.** **ICML 2019**, *PMLR* 97:2242–2251. 核查：给定效用定义下的训练数据边际贡献；不要与三格路径份额混用。

**[C1]** MacKinnon, J. G.; Nielsen, M. Ø.; Webb, M. D. **Cluster-Robust Inference: A Guide to Empirical Practice.** **arXiv:2205.03285**, 2022，核查版本 v1。核查：簇内相关、簇大小不均、少簇、多向聚类及 bootstrap 的适用条件。

**[C2]** Cai, Y.; Canay, I. A.; Kim, D.; Shaikh, A. M. **On the implementation of Approximate Randomization Tests in Linear Models with a Small Number of Clusters.** **arXiv:2102.09058**, 2021，核查版本 v2。核查：Algorithm 2.1、§2.1 关于 5%/10% 水平和簇数的离散限制、§4 方法前提。

**[C3]** Ibragimov, R.; Müller, U. K. **t-Statistic Based Correlation and Heterogeneity Robust Inference.** *Journal of Business & Economic Statistics*, 28(4), 453–468, 2010. **DOI:10.1198/jbes.2009.08046.** 核查：近似独立正态组估计量的前提，不是任意少簇样本的无条件保证。

---

## 仓库内不一致清单

### 11 条内部读数逐条核对

| 编号 | 核对结论 | 不一致、必要限定与出处 |
|---:|---|---|
| **1** 数据准备 delayed 平均 +0.06 | **数值近似成立，但需限定实验人口。** | DEV-SEQ-1 两次是 +0.061771/+0.068199，相对基础 raw 路径；实际形成段为 3×10×2，不是统一的 20×5。`docs/DEV_SEQ1_PER_SEQUENCE_2026-09-07.md:179–207`；raw 含基础填补，见 `AGENTS.md:647–678`。 |
| **2** 逐序列比固定高 +0.005/+0.011，是噪声量级 | **数值近似成立，“属于噪声”尚未证明。** | 精确值 +0.004461/+0.010890；后者超过 MATERIAL。不能用不同人口/执行链的重复差分直接判其无效，也不能据单次优势称稳定有效。`docs/DEV_SEQ1_PER_SEQUENCE_2026-09-07.md:179–207`。 |
| **3** 相邻窗口增益相关约零 | **只对较弱的秩相关成立。** | Spearman 0.017–0.109，Pearson 0.281–0.364。低持久性不否定可见状态条件化。`docs/R4A_B_PERSISTENCE_AND_VARIANCE_RESULT_2026-09-07.md:59–72,128–146`；`AGENTS.md:441–446` 已有更正。 |
| **4** 70–80% 经共享模型通道，处理一条主要影响共享模型 | **前半是特定绝对份额，后半混淆干预单位。** | 三格替换的是程序处理整套训练材料后的模型；未做只修改一条训练序列的外溢干预。份额使用绝对值，不是净增益或唯一因果比例。`docs/R4D_A_ACTION_DECOMPOSITION_RESULT_2026-09-08.md:20–34,88–94`；`evaluation/main_protocol_p4/per_sequence.py:24–48`。 |
| **5** per-channel Static 好 25%，严重伤害到 1/5–1/10 | **方向成立，但面、比例和混杂需修正。** | 25.12% 是两面合并，delayed 为 18.41%；四格伤害比 3/28、9/31、8/60、6/75，并非都在摘要区间。自身 10 窗与他人 200 窗混入。`docs/R4D_B_PERCHANNEL_RESULT_2026-09-08.md:16–25,89–97`；设计笔记 `:360–365` 已承认材料混杂。 |
| **6** 六设置均不敌固定，oracle 0.07–0.13，唯一稳定关系 AUC≈0.83 | **主判断应收窄；不是六格严格落后，也不是所有关系唯一。** | 多数与 always-P 完全相同，一格 +0.000548。oracle 精确差距约 0.065844–0.130807，限 P/identity。AUC 为 pooled/W2/delayed/ctx 的四块留一、较低 z 对严重伤害；support 另有 helped_ctx 关系。ctx 子集按结果分量非零筛选。`docs/R4E_CHANNEL_IDENTIFIABILITY_RESULT_2026-09-08.md:103–141`；`evaluation/main_protocol_p4/audit_r4e_channel_identifiability.py:111–123`。 |
| **7** 加卡两组 delayed −0.119/−0.111，挤掉原程序探测位 | **数字成立于附录 sol 复验，不能覆盖同文全部实验。** | −0.119086/−0.111169 在 sol 附录；正文 deepseek 两组最终差为零。损失几乎集中于同一个 u23：旧 MAD 有效，新候选 IQR 未过且无 MAD 回退。支持候选/探测路径干扰，不是“两独立环境证明普遍知识越多越差”。`docs/DEV_KNOW1_ACCUMULATED_SKILL_TWO_ARMS_2026-09-07.md:13–30,424–459`。 |
| **8** Slow 三次条件几乎不触发：0/36、3/60、1/40；子句基率 6.7% | **暴露数基本成立，“连续三次新写条件”不准确。** | SEQ-2 两次各 0/18；SEQ-3 g2 3/60，而 g1 另有 0/60；SEQ-4 修改的是 body，沿用 applicability 后为 1/40。它不是第三次独立写出新条件。简报的 6.7% 在本次核对中未独立复算，需补具体子句、形成材料名单与分母，暂不列为已验证事实。`docs/DEV_SEQ2_SLOW_UPDATE_TO_FAST_RESULT_2026-09-08.md:7–15`；`docs/DEV_SEQ3_GUIDANCE_DELIVERY_RESULT_2026-09-08.md:128–150`；`docs/DEV_SEQ4_GUIDANCE_ADOPTION_RESULT_2026-09-08.md:117–148`。 |
| **9** Fast 只决定 45.7%；identity 覆盖净收益 +3.58；自写 32 次 | **计数大体成立，但“决定比例”已被最新正典纠正。** | 45.7%=53/116 是可观察选择与最终交付一致率，不是 53/120，更不是因果贡献。identity 20 次中被覆盖 12 次；delayed 合计 +3.5842，7 正 5 负。实际 75 次部署中 43 recalled、32 searched，32 的程序标签重算为 MAD24/IQR6/Hampel2；不等于发明了新算法族。`docs/DEV_SEQ3_GUIDANCE_DELIVERY_RESULT_2026-09-08.md:231–253`；`AGENTS.md:574–578`；g2 JSON 的 `deployed_via`、`deployed_label`。 |
| **10** 严格同知识 51 对、41 零、6 对超过 MATERIAL | **存在待修的算术矛盾；严格成员也缺收据。** | 57 对 JSON 为 46 零、9 达阈值；文档说剔除 6 对中 5 零，所以剩余不可能只有 6 达阈值。按其剔除均值应仍是 9。不能把 6/51 当波动概率。`docs/DEV_SEQ3_GUIDANCE_DELIVERY_RESULT_2026-09-08.md:321–338`；`dev_seq3_guidance_delivery__contrasts.json:q3_and_q4_per_group.g2.exposure_split.same_knowledge_pairs`。详见 C。 |
| **11** Slow 看见 30 失败：21 Support 拒绝、9 delayed 翻负；23 成功 | **直接对上，但“30 次部署失败”将是错误解释。** | `dev_seq4_revision.json:boundary.card.failing_decisions` 的 `symptom` 重算为 SUPPORT_NEGATIVE21、SUPPORT_POSITIVE_DELAYED_NEGATIVE9；`matched_successes` 长度23。21 个被拒候选没有对应真实部署 delayed 失败，部分字段应保持 UNKNOWN；30 是从完整输入分组后选择的诊断组，不是完整 86 条 Episode 的总体失败率。`docs/DEV_SEQ4_GUIDANCE_ADOPTION_RESULT_2026-09-08.md:110–115,152–154`。 |

### 额外需要进入内部核对、但不能擅自改成项目决定的事项

**I-1｜旧文档中的“1/40 必然没有分辨力”应降格。** `docs/DEV_SEQ4_GUIDANCE_ADOPTION_RESULT_2026-09-08.md:146–150` 的绝对表述不成立：单个 +0.20 就能给 40 个对象带来 +0.005 均值。真正缺的是稳定效应推断与足够暴露，而非算术上不可能。较新的 `docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md:383–386`、置顶状态已收窄；引用旧段必须附更正。

**I-2｜特征边际范围重叠不能证明没有可观察条件。** `docs/DEV_SEQ1_PER_SEQUENCE_2026-09-07.md:286–299` 从已有公开特征的正负例范围重叠，推到条件不存在，逻辑过强；范围重叠不排除联合模式、非线性或概率性区分。结论只能是当前筛查没有找到足够的条件。

**I-3｜“select 无法阻止部署”只能用于局部机制，不是全链因果定理。** `docs/DEV_SEQ3_GUIDANCE_DELIVERY_RESULT_2026-09-08.md:231–253` 有绝对化句子。代码默认路径确实不把 identity 当 veto，但选择仍可改探测顺序；指导还可能改变候选供给。较新的 `AGENTS.md:563–582` 应优先。`online_loop.py:895–953` 还含不同 admission/selection policy 分支，不能把一个默认分支概括为所有配置。

**I-4｜旧运行时与新 Fast-only 不只相差 Support。** 新运行还清空原始 Episode 注入；因此跨包直接相减混合了反馈兜底、历史可见性和随机调用等变化。第一包的同历史影子审计只识别历史轨迹上的即时覆盖差，不是完整无 Support 课程的反事实。出处：`docs/DECISIONS.md:44–45`；`docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_TASK_2026-09-08.md:51–90`；`methods/ttha/method.py:350–379`；`methods/ttha/fast_agent.py:893–904`。

**I-5｜UNKNOWN/identity 与完整总体均值的接口约定需在外层核验。** `evaluation/main_protocol_p4/per_sequence.py:241–274` 可把缺 assignment 当 identity、可在 UnitFault 后返回可读子集均值，而第一包要求完整总体。此处列为执行验收风险；本次完整 g2 结果没有发现因此删掉配对的事实。

**I-6｜formation 的样本计数不应进入 outcomes 的独立性标签。** contrasts 的 `outcomes.independence` 为 10/34，而实际 outcome 配对为 20 UID×3 origin。需核对字段语义与来源，不应取其中一个较大的数字为统计检验提供分母。

**I-7｜0.83 的筛选实现比自然语言更接近事后通道条件。** R4E 的注释把排除项解释为“服务窗未修改”，但实现判断的是损失分解量 `ctx` 是否接近零。二者可能一致，但代码没有以动作改点数直接保证等价。若要将其升为部署规则，必须提供不依赖结果的等价性收据。

**I-8｜原始数据版本与基本 raw 含义需继续锁定。** `AGENTS.md:647–678` 已记录误把 without-missing 变体当作 with-missing 的历史问题；本报告没有把那批零缺失实验当自然缺失证据。任何跨阶段汇总，必须先统一数据版本、训练 roster、评分面和基础填补路径，不能仅用“KDD”或“raw”两个标签做跨表拼接。

---

**本部分的最终边界：**上述结果足以支持对当前解释链的收窄与对第一包的严格验收；不足以替项目决定最终保留哪条研究路线。尚未展开的 D–G 将涉及 shield/Slow 输入、环境候选、近期 agentic 文献与反方实验，不能视为本文件已完成的内容。
