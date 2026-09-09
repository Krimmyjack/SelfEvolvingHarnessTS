# 整体机制文献广度表（Grok 辅助，供 Kimi 深读挑选）

日期：2026-09-05  
角色：广度检索与原始证据整理，**不是**方法冻结，**不是**第二套实验计划。  
边界：只调研；不跑项目实验、不读密封数据、不改代码/合同/AGENTS/既有工件。公开检索未粘贴未发表项目数字或结果表。

相对 2026-09-05 两份 V2 地图的增量：V2 核心是 SPIBB / 保守 bandit / 实体准入 / 分层池化 / FFORMA 软路由 / 清洗 AutoML。本表补 **Agent 真正改什么**（ADAS / DGM / AFlow / FunSearch / AlphaEvolve / STOP / Promptbreeder / TextGrad / GEPA / GPTSwarm）、**已核验的提议–评价成本**、**负迁移与干扰**、**训练语料归因**、**冻结 Consumer 上的 test-time 适应（含伤害）**、**目标误指定**。不把这些工作收成「SPIBB + 分层模型 + 新准入门」购物单。

---

## 0. 给 Kimi 的使用说明

- 优先深读标 **核心** 的 12–18 篇；标 **V2已有** 的除非要核对成本，否则不必再铺。
- 标 **反例** 的必须读：它们不支持「收窄/保守/换 Consumer 必然更好」。
- 成本列只写论文或作者页写明的数字；没写则 **未知**，不估。
- 几乎所有进化工作都用 **即时、可核验的任务分数**（准确率 / pass@1 / 数学界）。它们 **不是** 延迟、噪声、无干净标签的下游 Consumer 效用。这是本项目与文献的最大错位，不是小缺口。

---

## 1. 六个问题群：V2 已覆盖 vs 本轮新增

| 群 | V2 主要落点 | 本轮补什么 |
| --- | --- | --- |
| 1 决策目标 | 保守约束、拒绝选项、coverage–risk | 目标误指定 / specification gaming；「永远 baseline」如何饿死学习；探索–部署分离 |
| 2 效应异质性 | CATE/DTR、特征无预测力 | 负迁移的形式定义；无干扰假设破裂；何时该判「没有稳定规律」 |
| 3 干预×Consumer | 训练准备 vs serving 路由 | 影响函数（改训练集如何改预测）；冻结 TSFM 的 TTA（含 TTA 伤害） |
| 4 Agent 进化对象 | Voyager / WikiSkill / TimeClaw / AegisTS | ADAS、DGM、AFlow、FunSearch、AlphaEvolve、STOP、Promptbreeder、TextGrad、GEPA、GPTSwarm：**改代码 / 工作流图 / 提示 / 脚手架** |
| 5 迁移与持续学习 | 零样本转移程序不转移实体账本 | 负迁移；MAML（改参数初始化，不是改 Workflow）；DarwinX 跨基准 harness 转移 |
| 6 评价与搜索效率 | 延迟 bandit、k=2 确认昂贵 | **实际迭代次数、评价次数、美元成本**；验证集选择偏差 |

---

## 2. 紧凑证据表（28 篇）

列含义：RQ=研究问题；改什么；反馈；关键假设；自然数据；成本（已核）；迁移；来源。

### A. Agent / 工作流 / 脚手架进化（V2 几乎未作为主表）

| # | 文献 | 年/状态 | RQ | 修改对象 | 反馈 | 关键假设 | 自然数据 | 提议/评价成本（已核） | 迁移 | 链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Hu, Lu, Clune, **ADAS** / Meta Agent Search | ICLR 2025；arXiv:2408.08435 | 自动发明 agent 设计 | **代码中的 agent 程序**（提示、工具、控制流） | 验证集准确率 / F1 | 设计可写成图灵完备代码；有即时任务分数 | ARC、DROP、MGSM、GPQA、MMLU；非时序、非数据准备 | **25 轮**；Meta=GPT-4，被发现 agent=GPT-3.5；DROP val=128，GPQA val=32。美元总成本 **未知**。作者承认搜索昂贵 | 报告跨任务与跨模型转移 | https://arxiv.org/abs/2408.08435 ；https://github.com/ShengranHu/ADAS ；ICLR: https://proceedings.iclr.cc/paper_files/paper/2025/hash/36b7acf6f6010652b3f2a433774a66fe-Abstract-Conference.html |
| 2 | Zhang, Hu, Lu, Lange, Clune, **DGM** | arXiv:2505.22954 **v3 2026-03-12**；预印本 | 经验上自改代码，放弃可证有益 | **agent 自身源代码**（工具、上下文、评审机制） | 编码基准 pass 率 | 有沙箱；有可跑测试的基准；修改可编译 | SWE-bench / Polyglot（真实 GitHub issue，但是**编码任务**不是数据准备） | **80 代，每代 1 个新 agent**；SWE 并行 2。分阶段评价：先 **10** 题验「还能改代码」，再 **50** 题（SWE-bench-verified-mini）。作者估 **约 USD 22,000 / 约两周** 一次 SWE 跑。无自改进消融 39% vs 50%；无开放探索（只爬当前最优）**23% vs 50%**。谱系在第 4、56 代曾掉分后恢复 | 报告跨模型；作者自问「再跑更久能否超闭源」= **未知** | https://arxiv.org/abs/2505.22954 ；https://sakana.ai/dgm/ ；https://github.com/jennyzzt/dgm |
| 3 | Zhang et al., **AFlow** | ICLR 2025 Oral；arXiv:2410.10762v4 | 在代码工作流图上搜索 | **LLM 节点+边的工作流代码** | 验证集执行分 | 工作流可执行；有任务 GT | HumanEval, MBPP, MATH, GSM8K, HotpotQA, DROP | **N=20 轮**；每候选在 val 上执行 **5 次**；数据 **20% val / 80% test**；早停 top-k 不变 5 轮。对比实验里 ADAS 设 30 轮。论文自身 **美元成本未知**。第三方 GAIA 对照报 AFlow 训练约 $22.50（**不是**六基准原文数字，标第三方） | 同任务 test；不是跨数据集数据准备 Skill | https://arxiv.org/abs/2410.10762 ；https://github.com/FoundationAgents/AFlow |
| 4 | Novikov et al., **AlphaEvolve** | 2025 技术报告 arXiv:2506.13131 | 进化整文件算法 | **整文件代码**（非单函数） | 自动评价器（可跑小时级） | 必须有可计算的评价函数 | 数学公开题 + Google 内部调度/电路 | 相对 FunSearch：**数千次 LLM 样本即可**（FunSearch 量级 **10^6**）。单次评价可并行约 **100 计算小时**。Klarna 部署报道约 **6000** 候选程序 / 三周（**公司博客，非论文表**） | 75% 公开题复现 SOTA，20% 改进；内部基础设施 | https://arxiv.org/abs/2506.13131 ；https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ |
| 5 | Romera-Paredes et al., **FunSearch** | **Nature 2023** | 在函数空间搜索可验证发现 | **短 Python 函数**（约 10–20 行） | 确定性评分器 | 评价快（≤20 min CPU）；可大规模采样 | cap set、装箱启发式 | **约 10^6 LLM 样本**；约 15 sampler + 150 CPU evaluator。作者选**快而弱**的模型换样本量 | 装箱启发式跨实例规模；组合构造对更大 n **常不泛化**（后续复现注记） | https://doi.org/10.1038/s41586-023-06924-6 ；https://github.com/google-deepmind/funsearch |
| 6 | Fernando et al., **Promptbreeder** | arXiv:2309.16797；OpenReview | 自指进化：任务提示 **和** 变异提示 | **自然语言提示**，不是程序 DSL | 训练集 batch 准确率 | 有带标签训练集 | GSM8K 等推理；ETHOS 分类 | 论文：**种群 50，通常 20–30 代**；二元锦标赛。OpenReview：一代 ≈ POP_SIZE 次评价；图中约 **2000** 次评价。精确「每次有效修改要多少提议」**未知** | 主要同任务 test；换 LLM 有附录 | https://arxiv.org/abs/2309.16797 |
| 7 | Zelikman et al., **STOP** | COLM 2024；arXiv:2310.02304v3 | 脚手架递归改进自己 | **调用 LLM 的 Python 脚手架**（不是权重） | 下游小任务效用 | GPT-4 级模型；有效用函数 | 小集合下游任务（**不是**大规模自然时序） | 论文未给总提议次数。作者明确：**不是**完整递归自改进（模型本身不变）。有沙箱逃逸频率评估 | 未知 | https://arxiv.org/abs/2310.02304 ；https://github.com/microsoft/stop |
| 8 | Yuksekgonul et al., **TextGrad** | **Nature 2025** 639:609–616；arXiv:2406.07496 | 用文本梯度反传改进复合系统 | **计算图中的文本变量**（提示、代码片段等） | LLM 自然语言批评 | 存在可查询的评价 LLM；前向是可微类比的文本图 | GPQA、LeetCode-Hard、分子、放疗计划 | GPQA GPT-4o 51%→55%；LeetCode-Hard 相对 +20%。**总反向步数 / 美元未知** | 框架声称跨任务不改代码；不是跨 Consumer 数据准备 | https://doi.org/10.1038/s41586-025-08661-4 ；https://github.com/zou-group/textgrad |
| 9 | Zhuge et al., **GPTSwarm** | ICML 2024 PMLR 235 | agent 当可优化图 | **节点提示 + 边连接** | 任务准确率 | 图参数可优化（含 RL 边） | Mini Crosswords、MMLU、HumanEval、GAIA | 边优化约 **10 次迭代**：0.465→0.575（±0.0275）；换 GPT-4-Turbo 到 0.800。总 LLM 调用 **未知** | 优化边再换更强模型 | https://proceedings.mlr.press/v235/zhuge24a.html ；https://openreview.net/forum?id=uTC9AFXIhg |
| 10 | Agrawal et al., **GEPA** | ICLR 2026 Oral；arXiv:2507.19457v2 | 语言反思进化提示，对比 RL | **系统中的提示** | 轨迹 + 自然语言反馈，不只标量 | 度量能返回可教的文字反馈 | HotpotQA、IFBench 等 | 相对 GRPO（报 24k rollout）：**最多约 35× 更少**；部分任务约 **678** rollout 达最优。相对 MIPROv2 +10% 量级 | 两模型；不是跨数据准备域 | https://arxiv.org/abs/2507.19457 |
| 11 | **DarwinX** | arXiv:2608.07545 **2026-07 预印本** | 在冻结模型上选择 harness 种群 | **harness / 脚手架**，模型冻结 | 各基准自带 verifier | preserve-and-extend：不回归已有覆盖 | Terminal-Bench、WebArena 等 | 「一轮」平均约 +17 分。**提议次数未知** | 称 Terminal-Bench harness **原样**转到 SWE-bench Verified | https://arxiv.org/abs/2608.07545 |

### B. 决策目标、误指定、保守策略的副作用

| # | 文献 | 年/状态 | RQ | 修改对象 | 反馈 | 关键假设 | 自然数据 | 成本 | 与项目 | 链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | Krakovna et al., Specification gaming | 2020 DeepMind **博客**（非正式论文） | 字面目标 ≠ 意图 | 奖励/适应度定义 | 被游戏的标量目标 | 目标可被漏洞满足 | 约 60 个案例汇编 | 不适用 | **反例**：优化「伤害↓」或「门通过」可以牺牲效用。收窄同时降伤害和效用，首先要问是不是 **目标写错**，不是自动加门 | https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/ |
| 13 | Sambasivan et al., Data Cascades | CHI 2021 | 数据问题如何级联到下游 | 数据实践（不是算法） | 访谈 53 人 | 高风险部署 | 健康/保护等实践 | 不适用 | 92% 经历至少一次 cascade。支持「数据准备是一等机制」，**不**支持某一 Scope 规则 | https://doi.org/10.1145/3411764.3445518 |
| 14 | Wu et al., Conservative Bandits | ICML 2016 | **V2已有** | 臂选择 | 即时奖励 | 随机 bandit | 合成 | 理论 regret | 「相对 baseline 不太差」；**不是**永久禁止探索。V2 已用 | http://proceedings.mlr.press/v48/wu16.pdf |
| 15 | Laroche et al., SPIBB | ICML 2019 | **V2已有** | 策略在低计数 (s,a) bootstrap | 离线轨迹 | 已知 baseline；状态可数 | MDP / DQN | 未知 | 证据不足回 baseline。**不是** UID 表。V2 已用 | https://proceedings.mlr.press/v97/laroche19a.html |

### C. 异质性、负迁移、干扰（CATE 套不上时）

| # | 文献 | 年/状态 | RQ | 修改对象 | 反馈 | 关键假设 | 自然数据 | 成本 | 与项目 | 链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | Wang, Dai, Póczos, Carbonell, Negative Transfer | CVPR 2019 | 形式定义负迁移并过滤源 | 源样本权重/门 | 目标标签 | 有源和目标标签 | 视觉域适应四基准 | 未知 | **反例**：源「相关」仍可伤害目标。跨数据集 Skill 默认复用需要 **可拒绝**，不是分层模型必对 | https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.html ；arXiv:1811.09751 |
| 17 | Hudgens & Halloran, Interference | JASA 2008 | 单元间干扰下的因果效应 | 组内处理分配 | 潜在结果 | 组间无干扰、组内可干扰 | 疫苗/住房券 | 不适用 | 改 A 的训练语料改变共享模型再服务 B = **干扰**。实体级 τ_e = Y(program)−Y(raw) **识别失败**。不要叫 CATE | https://doi.org/10.1198/016214508000000292 ；PMC2600548 |
| 18 | Finn, Abbeel, Levine, **MAML** | ICML 2017 | 少步梯度适应新任务 | **网络初始化参数** | 任务内标签/回报 | 任务可微、任务分布可采样 | Omniglot / MiniImageNet / RL | 每任务少量梯度步 | **对照**：这是参数适应，不是 Workflow 结构修订，也不是 Skill 库。若项目需要的是程序组合进化，MAML 不是答案 | https://proceedings.mlr.press/v70/finn17a.html |
| 19 | Koh & Liang, Influence Functions | ICML 2017 | 训练点如何改变预测 | 无（诊断） | 损失与 Hessian | 光滑损失；近似在非凸上仍可用 | 图像/线性模型 | 相对留一重训便宜 | 对应「改训练语料」几何：服务点的变化应追溯到哪些训练行。**不是**部署门 | https://proceedings.mlr.press/v70/koh17a.html |
| 20 | Li et al., CleanML | PVLDB 2021 | **V2已有反例** | 清洗算子 | 下游 ML | 真实错误，无单一 GT | 表格 ML | 未知 | 清洗可伤害；train vs test 清洗是不同决策 | https://chu-data-lab.github.io/downloads/CleanML.pdf |

### D. 冻结 Consumer 与 test-time 适应（含失败）

| # | 文献 | 年/状态 | RQ | 修改对象 | 反馈 | 关键假设 | 自然数据 | 成本 | 与项目 | 链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 21 | Kim et al., **TAFAS** | AAAI 2025 | 非平稳预测的 TTA | 校准模块；主干可冻结 | **部分已观测的未来真值**（预测后不久可观测） | 部署中真值会延迟到达 | 多种 TSF 基准 | 未知 | 预测 TTA 比视觉 TTA 多一个「真值稍后可见」。**不是**换 Consumer 必解 | https://doi.org/10.1609/aaai.v39i17.33965 |
| 22 | PETSA | arXiv:2506.23424 **2025 预印本** | 参数高效 TTA | 输入/输出低秩校准；**冻结预测器** | 测试窗损失 | 校准模块足够 | TSF 基准 | 未知 | 改的是适配器不是清洗程序 | https://arxiv.org/abs/2506.23424 |
| 23 | Wu et al., TTA for non-stationary TS | arXiv:2602.00073 **2026 预印本** | 合成漂移 vs 金融市场 TTA | 归一化仿射；主干冻结 | 无标签窗 / 熵 | 漂移可跟踪 | ETT 合成漂移；SPY/QQQ/EURUSD | 未知 | **反例**：合成漂移上 TTA 有时降误差；金融上简单 BN 更新更稳，**更激进的 norm-only TTA 甚至伤害**。换适应强度 ≠ 换 Consumer，但同属「动部署时参数」 |
| 24 | COSA | ICLR 2026 Poster | 输出空间残差校正 | **预测残差适配器**，不改数据 | 已观测真值、无泄漏协议 | 基模型冻结 | 多种 TSF | 称相对无 TTA +13.9–17.0%（论文自报） | 几何是 **改输出** 不是改训练语料或 serving context | https://openreview.net/forum?id=L7Z5wBMPrW |
| 25 | Montero-Manso et al., **FFORMA** | IJF 2020 | **V2已有反例** | 方法混合权重 | 预测误差 | 有元特征 | M4 | 未知 | 平均优于硬选择 | https://doi.org/10.1016/j.ijforecast.2019.02.011 |

### E. V2 已覆盖、本表只钉住以免重复深读

| # | 文献 | 备注 |
| --- | --- | --- |
| 26 | Voyager, arXiv:2305.16291 | 可执行技能 + 环境成功标签。V2 已有 |
| 27 | WikiSkill, arXiv:2608.27454 **预印本** | Wiki/Skill 分层 + val 回滚。依赖 GT。V2 已有 |
| 28 | TimeClaw, arXiv:2606.05404 **预印本** | TS harness 进化用 held-out 成功率。V2 已有 |
| — | DomainBed, Gulrajani & Lopez-Paz ICLR 2021 | 域泛化 ERM 常赢。V2 已有 |
| — | Geifman & El-Yaniv NeurIPS 2017 | 选择分类 / coverage–risk。V2 已有 |

---

## 3. 已核验成本对照（只写论文写明的）

| 系统 | 改什么 | 有效搜索预算（已核） | 评价器 | 未报告 |
| --- | --- | --- | --- | --- |
| FunSearch | 短函数 | ~10^6 LLM 样本 | 确定性评分 | 每次「科学发现」对应多少无效样本 |
| AlphaEvolve | 整文件 | **数千** LLM 样本（作者对比 FunSearch） | 可小时级自动评价 | 单题精确样本数 |
| ADAS | agent 代码 | **25** 元迭代 | 小验证集（32–128） | 美元；每迭代多少 inner 调用 |
| AFlow | 工作流代码 | **20** 轮 × 每候选 val 上 **5** 次执行；20% val | 任务 GT | 原文美元；每轮展开多少新图 |
| DGM | agent 源码 | **80** 代 × 1 新 agent；先 10 后 50 题 | SWE/Polyglot 测试 | 50 题上的选择偏差大小；完整 Verified(500) 每代是否都跑（文中按信心扩评价集） |
| DGM 美元 | 同上 | 作者附录约 **USD 22,000** / SWE 一次 | API | 是否含失败编译体 |
| Promptbreeder | 提示 | 种群 50，20–30 代；图约 2000 评价 | 训练 batch 准确率 | 官方 API 账单 |
| GEPA | 提示 | 相对 GRPO 最多 ~35× 少；部分任务 ~678 rollout | 带文字反馈的度量 | 每任务绝对美元 |
| GPTSwarm 边优化 | 图边 | 约 10 迭代（Crossword） | 准确率 | 总 token |
| STOP | 脚手架 | **未知** | 小任务效用 | 全部 |
| TextGrad | 文本变量 | **未知**（只报终点分数） | LLM 批评 | 步数 |

共同结构（给 Kimi，不是推荐实现）：**提议器（常是 LLM）× 可执行修改 × 独立评价器 × 档案/种群**。LLM 贡献的是 **提议**，不是批准。批准来自评价器。本项目若评价器是延迟、有噪声、且会干扰其他实体的 Consumer，则上述论文的「val 准确率门」**不能**直接当 Skill 晋升规则。

---

## 4. 反例与失败（不要只收支持收窄/分层/新门的论文）

1. **CleanML**：清洗可降低下游表现。  
2. **FFORMA**：硬选择输给混合。  
3. **负迁移 (Wang 2019)**：源数据可伤害目标；需要过滤/拒绝，不是默认共享。  
4. **干扰 (Hudgens 2008)**：共享 Consumer 下「谁被准备」改变「谁被服务」的潜在结果。  
5. **金融 TTA (Wu 2026 预印本)**：更激进的 test-time 更新可以变差。  
6. **DGM 消融**：只爬当前最优 → 23% vs 50%；谱系允许中间掉分。若项目禁止一切暂时变差的修订，可能堵死 DGM 那种「垫脚石」。  
7. **DGM / STOP**：自改代码会出现奖励黑客 / 沙箱逃逸（STOP 论文评估了逃逸；DGM 强调沙箱与人工监督）。  
8. **Specification gaming**：优化字面门（伤害、覆盖）可以违背意图（效用）。  
9. **DomainBed**：复杂域泛化算法在公平调参下常不赢 ERM。  
10. **FunSearch 泛化**：为固定实例进化的程序换规模常失效。  
11. **ADAS/AFlow**：用同一任务的 val 选工作流再上 test——**选择偏差**；不是新数据集 held-out Skill。  
12. **MAML**：少样本成功不等于 Workflow 可迁移；它迁移的是参数敏感性。

---

## 5. 对「目标对齐」类观察的判断（供 Kimi，不是结案）

任务书中的发展观察：一条收窄在后续窗 **降低伤害也降低效用**；选择目标与全人群评价目标不一致。

**能排除的**

- 「Scope 收窄是只降伤害、不降效用的免费午餐。」  
- 「当前选择目标与部署评价目标已经对齐。」  
- 「只要再加一个更严的安全门，效用自然回来。」（门更严通常进一步降覆盖/效用，见选择分类。）

**不能证明的**

- 必须改用分层 Bayes / SPIBB / 实体准入（V2 偏好）。  
- 必须放弃 Scope、改 Workflow 组合（本表进化文献的偏好，同样未在该观察上受检）。  
- 必须换 Consumer（TTA 文献说冻结主干 + 小适配器有时够，有时更糟）。  
- 异质性来自实体而非时间/Pattern（该观察没有识别实验）。  
- 不修改永远正确（保守策略在 DGM/探索文献里会停在局部最优）。

更接近文献的读法：这是 **目标误指定或选择–评价分裂** 的症状，与 Krakovna 的 specification gaming、Geifman 的 coverage–risk、以及「对选中子群优化却用全人群计分」是同一类问题。它要求把 **探索目标、晋升目标、报告目标** 写成三个可证伪量，而不是证明某一架构。

---

## 6. 建议 Kimi 深读顺序（仍不是冻结）

1. **DGM**（改什么 = 脚手架代码；成本与消融最完整；允许中间变差）。  
2. **ADAS + AFlow**（工作流/agent 代码搜索；val/test 泄漏风险对照本项目 held-in/out）。  
3. **FunSearch vs AlphaEvolve**（评价器质量决定进化能否「发现」；样本量差三个数量级）。  
4. **Wang 负迁移 + Hudgens 干扰 + Koh 影响函数**（跨域 Skill 与训练语料干预的识别假设）。  
5. **TAFAS + 金融 TTA 反例**（冻结 Consumer 时还能动什么）。  
6. **GEPA vs GRPO**（标量奖励 vs 语言反馈的样本效率——若本项目只有 sMASE 标量，GEPA 的优势可能消失）。  
7. **Krakovna + Data Cascades**（目标与数据工作，不是新模块）。

不要默认输出「SPIBB + 分层 + 置信序列 + 新状态机」。三条互替路线应在 **修改对象** 上互斥，例如：（i）只进化可执行 Workflow 组合；（ii）只进化冻结 Consumer 上的小适配器；（iii）只进化晋升/拒绝规则（不做程序搜索）。每条删掉另外两条的组件。具体路线由 Kimi 写，本文件不写实验计划。
