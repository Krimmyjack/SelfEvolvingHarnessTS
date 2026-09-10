# SelfEvolvingHarnessTS：整体机制与研究定位文献调研

## A. 主报告：项目真正缺的机制是什么

### 1. 总判断

SelfEvolvingHarnessTS 当前真正缺的，不是更精细的 Scope，不是新的准入门，也不是把现有规则换成 CATE、bandit 或层级模型。

更准确地说，缺的是：

> **一套“几何一致、目标一致、探索与部署分离”的 Evidence → Structural Revision 机制：让昂贵、延迟、带噪声的 Consumer feedback 能够合法地修改 Agent 的 Workflow / Observation / Skill / Harness，并证明这种修改改善的是未来总体决策，而不是只改善当前被选择的人群或开发窗口。**

这里有三个必须首先对齐的“单位”：

1. **Revision Unit：** Agent 究竟修改什么——Workflow、Skill、Observation procedure、Memory abstraction，还是 Harness control？
2. **Intervention Unit：** 这个修改实际改变什么——一条 series、一个 serving context、整个训练池，还是模型路由？
3. **Evaluation Unit：** 最终效用在哪个总体上定义——被处理样本、完整 population、未来时间窗口、整个 dataset，还是另一个 Consumer？

旧框架已经明确希望从 `Observe → Workflow → Feedback → Experience → Skill → Harness Patch` 形成长期闭环，而不是固定 Router。 但本轮文献说明：如果上述三个单位没有先对齐，那么“学习更多 Context”“收窄 Scope”“记更多 entity evidence”都有可能只是更精确地优化一个错误 estimand。

---

### 2. 决策目标：搜索和部署必须是两种问题

部署时，我建议把目标写成**相对当前部署祖先的 population-level 增量效用**：

\[
\Delta V(H')
=
V_{\mathrm{future,pop}}(H')
-
V_{\mathrm{future,pop}}(H_{\mathrm{deploy}})
\]

并要求：

\[
\text{Integrity}=1,\quad
\text{Risk}\le \delta,\quad
\text{Cost}\le B .
\]

Coverage、abstention/no-op 单独报告，而不要全部揉进一个任意加权 reward。

这与 SPIBB 一类安全策略改进工作的合理部分一致：当某个状态—动作区域证据不足时回到已知 baseline；但 SPIBB 的结论不是“证据不足的东西永远不准探索”，而是“不能把不可靠策略直接部署”。

这里本轮相对旧方案有一个重要修正：

> **安全祖先应该主要约束 deployment lineage，而不应垄断 exploration lineage。**

DGM 是很好的反例。它不是每次只保留当前最高分 agent，而是维护开放式 archive，从不同历史 agent 继续产生后代；这种 open-ended archive 是其优于单一路径自改进的重要组成部分。DGM 明确把“stepping stones”视为开放式搜索的核心。

因此应区分：

- **Contract-invalid candidate**：泄漏、非法访问、程序无效——搜索阶段都不能执行；
- **Deployment-unqualified candidate**：目前效用差、风险大、支持不足——不能部署，但可以在沙箱/开发环境中作为祖先、反例或变异素材继续被研究。

#### 什么时候“不修改”是合理动作？

No-op 合理，当：

- 当前声明的可编辑空间没有 material headroom；
- 现有合法信息不足以区分正负动作，而且线上代价高；
- 相对部署基线的预期增量效用扣除风险和成本后非正；
- 当前 modification 的 value-of-information 也很低。

但保守机制开始**阻止学习**，当：

> 获得“正向证据”的前提，是先有执行权限；而执行权限又要求事前已有正向证据。

这是一个自封闭循环。部署可以保守，**离线探索和信息获取不能被同一个 promotion gate 饿死**。

你目前观察到“一次收窄后来同时降低 harm 和 utility”，首先应解释为**目标/约束权衡诊断**，而不是证明还需要继续收窄。

---

### 3. 可学习的效应异质性：先识别，再决定用不用 CATE

Task、Consumer、Pattern、entity、time 五类变量不是同一层面的“features”。

**Task** 改变的是“什么算好”。同样的平滑，对 forecasting 和 anomaly detection 可以有相反符号。

**Consumer** 改变的是干预到最终 utility 的传递函数。相同 prepared data 进入 Ridge、global TSFM、局部模型或 routing system，效应不必相同。

**Pattern** 最有希望成为可迁移 effect modifier，但只有满足三个条件才值得进入核心方法：部署前可观察；在多个独立实体/时间段出现；对 action response 有稳定的 out-of-sample 增量预测力。

**Entity** 可能是持久的潜在机制，也可能只是 dataset-specific ID proxy。不能因为 entity history 比 Pattern 好预测，就直接推出“实体资格”是机制。

**Time** 不是普通 feature。它决定 evidence age、regime drift 和效应稳定性。

这里建议用一个四分诊断，而不是一见 sign flip 就加新 Gate：

| 失败类型 | 应看到什么证据 |
|---|---|
| 信息不足 | 增加合法 Observation / probe 后，未来 action-effect prediction 明显改善 |
| 目标错误 | 同一预测在 selection cohort 看似正确，但按 population target 重计后 policy 排名改变 |
| 估计器错误 | 在 estimand 已正确、干预可识别时，预测 effect 与独立 intervention/replay 持续不符 |
| 无稳定规律 | 在足够多未见 entity/window 上，所有声明的可观察变量都不能稳定超越 task×consumer-only baseline |

R-learner 等异质处理效应方法很强，但前提是 treatment、outcome、propensity 等满足相应识别条件。 对你的项目，它可以成为**某种干预几何成立后的估计器候选**，不能先被指定为研究答案。

更值得注意的是 policy learning 文献：Athey–Wager 直接学习满足应用约束的 treatment policy，说明“必须先把每个人的 CATE 估得非常准”并不是唯一道路。

**对本项目的可证伪预测：**

> 如果 Task × Consumer × observable Pattern 真有可迁移异质性，那么在 leave-entity-out + future-window evaluation 上，它应比只用 Task × Consumer 的 policy 更准确地预测 action 的方向/排序；如果加入 entity ID 才有效，而换时间后消失，则当前获得的是局部记忆，而不是 Pattern-level knowledge。

---

### 4. 干预与 Consumer 几何：这是当前最容易被低估的问题

这是本轮最重要的新认识之一。

#### 情形 A：修改训练语料，再训练 pooled/global Consumer

假设对第 \(i\) 条 series 做 preparation \(a_i\)，然后所有 prepared series 一起训练共享模型：

\[
M = Train(D(a_1,\ldots,a_n)).
\]

此时第 \(i\) 条 series 的未来结果一般是：

\[
Y_i(a_1,\ldots,a_n),
\]

而不是简单的 \(Y_i(a_i)\)。

也就是说，一个 training-series treatment 会通过模型参数影响其他 series。标准 unit-level causal inference 中常见的 no-interference 假设已经破裂。Hudgens–Halloran 正是研究这一类“一个单位的 treatment 可以影响其他单位 outcome”的因果问题。

因此：

> pooled Consumer 下的 per-series gain 可以是很好的 diagnosis / credit signal，但不能未经证明直接叫做“该 entity 的 causal treatment effect”。

真正承担 promotion 的 counterfactual 应尽量保持为：

- group/set intervention；
- 完整 prepared training pool；
- 完整 retraining；
- future population outcome。

如果成本太高，Influence / leave-one-out / proxy 可以做候选筛选，但必须明确它们是 approximation。

#### 情形 B：冻结模型，只修改 serving context

模型参数不变，只对当前 series 输入 context 做 transformation，此时 unit-level potential outcome 更接近：

\[
Y_i(a_i)
\]

，识别问题简单得多。

#### 情形 C：数据准备同时触发 model routing

这时 action 其实是：

\[
a=(\text{preparation},\text{route/model}).
\]

若 routing 跟着 treatment 改变，就不能把最终 gain 单独归因给 data preparation，除非：

1. 冻结 routing；或
2. 从一开始就把 joint action 当成研究 estimand。

这也解释为什么“换一个 Consumer”不能自动解决当前问题——那相当于换了干预几何和研究问题。

时序领域还有两个有用反例。FFORMA 已经证明，使用 time-series features 训练数值 meta-model 来分配 forecasting 方法权重可以非常有效。 但如果 SelfEvolvingHarnessTS 最终只学到一个类似 FFORMA 的 feature→program numerical router，那么它虽然可能是强 baseline，却已经不是你想研究的 Agent Harness evolution。

另一方面，global forecasting 理论表明，global model 并不要求所有 series 来自相似过程，global/local 的差异涉及容量和 pooling，而不是简单的“异质就必须换 local model”。 所以不能预设“换 Consumer geometry 就会让 effect heterogeneity 消失”。

---

### 5. Agent 真正应该进化什么？

文献可以清楚分成四类：

**参数适应。** MAML、TTA 一类方法改变模型参数、初始化或小型 adapter；例如 TAFAS 在 TS forecasting 中利用逐渐可见的真实值适配 forecaster。 这不是你的核心问题，因为你的 LLM / Consumer 可以保持冻结。

**经验检索。** 从 Memory 取一个旧策略/轨迹并复用。它可以提高性能，但“检索成功”本身不是结构进化。

**结构修订。** ADAS 让 meta-agent 直接编写新的 agent code；AFlow 搜索代码表示的 agent workflow；DGM 修改自己的 agent code；Self-Harness 从失败轨迹提取 weakness，再生成 minimal harness edits 并 regression validate。

**经验→可复用结构。** Evo-Harness 更接近你的长期目标：将一次次噪声 execution context 编译为 structured skill harness，使冻结 agent 在后续任务改变行为。

所以 SelfEvolvingHarnessTS 要保住“self-evolving”这个研究身份，至少需要证明：

\[
\text{feedback}_{t}
\rightarrow
\Delta H_t
\rightarrow
\pi_{H_{t+1}}(\text{Workflow}\mid X)
\neq
\pi_{H_t}(\text{Workflow}\mid X)
\rightarrow
\text{future utility improvement}.
\]

只改变：

- 一个 UID 的 allow/deny；
- 一个 numerical score；
- 一个固定 program 的 threshold；
- 一个 top-k retrieval list；

都不足以单独证明 Harness evolution。

LLM 最合理的贡献不是最终裁判，而是：

> **在巨大、离散、结构化的 Workflow/Harness revision space 中生成有机制解释的候选，并利用 success/failure contrast 提出下一种可证伪结构。**

裁判仍是执行环境和 Consumer。

FunSearch 的成功非常能说明这个分工：LLM 负责产生程序，系统化 evaluator 负责评分；而它之所以可以扩展到约 \(10^6\) 个 LLM 样本，是因为问题拥有便宜、清晰、自动化的 `evaluate()`。

AutoTTS 也把自己的关键贡献明确写成**环境设计**：利用预采集 trajectory，让候选 controller 可以反复、便宜地 replay，完整 discovery 仅约 \$39.9 / 160 分钟。

这反过来揭示了你的真实科学 Gap：

> **TS data preparation 的 candidate 会改变训练语料，常常必须重训 Consumer；反馈又延迟、有噪声。你没有 FunSearch/AutoTTS 那种天然廉价、即时、近确定性的 `evaluate()`。如何把这种 Consumer feedback 转化为可信的结构修订，本身就是方法问题。**

---

### 6. 迁移与持续学习：真正可迁移的不是“这个 entity 可以做 Hampel”

更合理的知识层次是：

**较可能跨环境迁移：**

- Operator/Workflow 的语义与前置条件；
- 怎样观察一个问题；
- 某种 failure mechanism；
- Workflow skeleton / control procedure；
- 哪种 Task × Consumer 下什么信息必须保留；
- 某个 action 的已知 contraindication；
- 如何验证或证伪一个 Skill。

**通常需要目标环境重学：**

- effect magnitude；
- 阈值；
- 当前 entity state；
- 最近 regime；
- 当前 Consumer calibration；
- action 的具体参数；
- 是否当前仍然成立。

负迁移文献对此有直接警告。Wang et al. 将 negative transfer 定义为：同一个算法加入 source information 后，target risk 比 target-only 版本更差。

因此你的跨域实验最重要的 baseline 不是“无 Memory”，而是：

> **同一算法、同一 Target budget，Target-only vs Source-assisted。**

否则无法确认 Source Skill 到底是迁移知识，还是只是让系统多试了几个候选。

这里也不需要默认 hierarchical model。层级 pooling 是一种可能的 estimator；你的 scientific claim 应该是“是否存在可迁移的 procedural abstraction”，而不是“分层统计模型优于不分层”。

一个真正的 Skill 更像：

> **一个有 Context、执行 procedure、支持证据、反例、适用边界、证伪条件和 evidence age 的可修订假设。**

而不是：

> `Pattern X → Program Y`。

---

### 7. 评价与搜索效率

进化论文的预算实际上差异巨大，而且绝大多数论文**没有报告“每获得一次有效修改平均需要几个 proposal”**这个可直接比较的数字。因此下面只写原文能够核验的内容。

- **ADAS / Meta Agent Search（ICLR 2025）**：meta-agent 编写 agent code，逐轮进入 archive；论文验证其跨 domain/model transfer，但每次有效结构改善对应多少无效 proposal，并无统一报告。
- **AFlow（ICLR 2025）**：代码化 workflow + MCTS + execution feedback；论文报告六个 benchmark 平均提升 5.7%。具体“一个有效 edit 对应几次 proposal”未知。
- **DGM（arXiv:2505.22954，当前为预印本）**：开放 archive、自修改 source code；SWE-bench 20.0%→50.0%，Polyglot 14.2%→30.7%。有效 edit proposal ratio 未统一报告。
- **Self-Harness（2026 预印本）**：三个 base model held-out pass rate 从 40.5→61.9、23.8→38.1、42.9→57.1；accessible primary report 没有给一个可复用的“proposals / accepted edit”总成本数字，因此标 **unknown**。
- **TTHE（2026 预印本）**：test-time population harness evolution，执行轨迹 proxy 代替 gold label；论文自己把 proxy reliability 指为核心挑战。
- **AutoTTS（2026 预印本）**：完整 discovery 报 \$39.9、160 分钟；依赖 cached reasoning trajectories，因此评价候选无需重新调用 solver LLM。
- **FunSearch（Nature 2024）**：约 \(10^6\) LLM samples；它选择“更快但较弱”的代码模型来扩大搜索吞吐量。
- **Evo-Harness（2026 预印本）**：证明一次 execution 可以编译为持续 Harness knowledge，但精确“每次有效 Skill revision 需要多少 proposal”仍不能从摘要中得到，应标 unknown。

这意味着你不能照搬“进化论文一般每轮 3 个 proposal”之类数字。更关键的 benchmark 应是：

\[
\text{Utility gain per Consumer evaluation}
\]

和：

\[
\text{future gain per full retraining}.
\]

另外，持续在同一个 development window 上搜索 Harness 会产生 adaptive overfitting。TTHE 尤其值得注意：它属于 test-time adaptation，本身允许利用 test execution traces；这与“学完后在真正未来数据是否仍有效”是不同 claim。

因此你的承重实验应优先采用：

```text
Discovery window
→ Harness revision
→ 冻结
→ next chronological window / independent entities
→ evaluate
```

而不是在同一开发数据上 proposal、selection、报告最终分数。

---

## 本轮相对旧报告新增的五点认识

旧报告正确地看到 Harness evolution 需要 observable components、changeset、failure dossier、regression validation 和 rejected experience；旧框架也已经明确不希望系统退化为固定 Router。

但本轮有五个明显增量：

1. **探索安全 ≠ 部署安全。** DGM 说明低当前分数的 archive member 仍可能成为之后的重要 stepping stone；旧的 best-config/rollback 思想适合 deployment，不足以定义 exploration。
2. **Consumer geometry 先于 CATE/Scope。** pooled retraining 下有 interference，entity treatment effect 甚至可能不是正确 estimand。
3. **目标对齐比估计器精度更先。** 选择 cohort 和 population objective 不一致时，提升 effect estimator 只会更准确地优化错误目标。
4. **Evolution 的瓶颈很可能在 evaluator/environment。** ADAS、AFlow、FunSearch、AutoTTS 的成功都依赖可反复评价的 candidate；你这里的 delayed full-Consumer feedback 反而是最独特、最难的一环。
5. **跨域知识应被定义为“可证伪 procedural abstraction”，不是 entity authorization。** Negative transfer 要求任何 Source-assisted 方法都和等预算 Target-only 明确比较。

因此旧报告里的 predicted attribution、rollback、changeset、risk sentinel 都可以继续作为工程手段，但**没有任何一个单独回答项目当前的科学 Gap**。

---

# B. 核心文献证据表

| 文献 | 年份/状态 | 核心机制与必要假设 | 对本项目的意义 / 不适用处 |
|---|---|---|---|
| ActiveClean | PVLDB 2016 | 在有限 cleaning budget 下优先清洗更影响统计模型的记录；凸损失假设 | 早期 task-aware data cleaning；但有明确 cleaning operator，不解决 Harness structural evolution。 |
| SPIBB | ICML 2019 | 不确定区域 bootstrap 到 baseline；要求离线轨迹与可信 baseline | 支持部署 fallback；不支持禁止离线探索。 |
| Negative Transfer | CVPR 2019 | Source information 必须相对同一 target-only algorithm 评价 | 直接定义跨 dataset Skill 的最低科学对照。 |
| FFORMA | IJF 2020 | series features→forecast method weights；有大量训练 series/损失 | 是 Pattern numerical Router 的强 baseline，也是项目不应退化成的最终形式。 |
| R-learner | Biometrika 2021 | 正确 treatment/outcome/propensity 识别下估 CATE | 可作局部 estimator；pooled retrain interference 时不能硬套。 |
| Policy Learning with Observational Data | Econometrica 2021 | 在可识别 causal effect 下直接优化受约束 treatment policy | 说明不必先完美估 CATE；仍不能跳过因果识别。 |
| Locality & Globality | IJF 2021 | global/local forecasting 的表达能力与 generalization analysis | 警告不能简单把“换 local Consumer”视为异质性解决方案。 |
| Hudgens & Halloran | JASA 2008 | unit outcomes 可受其他 units treatment 影响 | pooled training-data preparation 的最关键识别警告。 |
| ADAS / Meta Agent Search | ICLR 2025 | meta-agent 写新的 agent code，archive 提供搜索历史 | 证明 LLM 可负责 structural proposal；假设 benchmark evaluator 足够直接。 |
| AFlow | ICLR 2025 | 搜索代码化 agent workflow graph，execution feedback 驱动 MCTS | 与 Workflow evolution 最直接；但其 benchmark feedback 远比 TS Consumer 便宜直接。 |
| TAFAS | AAAI 2025 | 部分未来真值到达后适配 forecaster | 展示另一种 Consumer-side adaptation；它改变模型，不是数据准备 Harness。 |
| DGM | arXiv:2505.22954，2026 当前预印本 | agent 自改代码 + open-ended archive + benchmark validation | 最强的“不要只保留安全/最高分祖先”反例。 |
| AegisTS | arXiv:2605.04902，2026 预印本 | hierarchical RL 优化 MTS cleaning order + method，双阶段 reward | 证明 downstream-aware TS pipeline optimization 已存在；你必须超越固定 operator policy learning。 |
| AutoTTS | arXiv:2605.08083，2026 预印本 | LLM 发明 controller，cached trajectory 提供 cheap/frequent feedback | 最重要启示是“environment/evaluator design”，不是它的具体 controller。 |
| Self-Harness | arXiv:2606.09498，2026 预印本 | Weakness Mining → minimal Harness Proposal → regression Validation | 最接近结构性 Harness repair；但其 feedback 是可直接验证的 coding task。 |
| TTHE | arXiv:2607.08124，2026 预印本 | test-time population harness evolution，execution trace proxy judge | 证明 unlabeled trace 可驱动 harness adaptation，同时凸显 proxy-selection 可靠性问题。 |
| Evo-Harness | arXiv:2608.15071，2026 预印本 | noisy one-shot context → reusable skill harness | 与“经验编译为可修订 Skill”最接近；仍依赖任务本身能给较强反馈。 |
| FunSearch | Nature 2024 | LLM program generation + systematic executable evaluator + population | 给出搜索规模反例：若 evaluator 便宜，可以用约百万样本；你的 evaluator 不具备这一条件。 |

原文入口：[ADAS / ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/36b7acf6f6010652b3f2a433774a66fe-Abstract-Conference.html) · [DGM](https://arxiv.org/abs/2505.22954?utm_source=chatgpt.com) · [Self-Harness](https://arxiv.org/abs/2606.09498?utm_source=chatgpt.com) · [TTHE](https://arxiv.org/abs/2607.08124?utm_source=chatgpt.com) · [AutoTTS](https://arxiv.org/abs/2605.08083?utm_source=chatgpt.com) · [Evo-Harness](https://arxiv.org/abs/2608.15071?utm_source=chatgpt.com) · [AegisTS](https://arxiv.org/abs/2605.04902?utm_source=chatgpt.com)

---

# 六个问题群的机制—假设—可证伪预测

| 方向 | 核心机制 | 必要假设 | 不适用时 | 本项目可证伪预测 |
|---|---|---|---|---|
| 决策目标 | population incremental utility + explicit baseline/no-op；search/deploy 分离 | population target 可定义；risk/cost 可测 | selection objective 与 population objective 不同 | 开放 sandbox archive 提高 future candidate headroom，而 deployment harm 不增加；否则 archive openness 无价值 |
| 效应异质性 | 用 Task/Consumer/Pattern 解释 treatment response | effect modifier 部署前可见且跨 entity/time 稳定 | pooled interference；entity/time sign 极不稳定 | Pattern-aware model 在 leave-entity/window-out 上优于 task×consumer-only；否则不宣称可迁移 Pattern effect |
| Consumer geometry | 先明确 training-pool / serving-context / routing treatment | intervention unit 清晰 | 多个机制同时移动 | 冻结 route 后若 preparation 排名大幅变化，之前的“data effect”其实混入 routing effect |
| Agent evolution | feedback 持久改变 Workflow/Harness generation distribution | evaluator 能区分有意义修改 | 只改变 retrieval/UID/score | H1 必须产生行为/结构上不同的 candidate distribution，并在 next-window 带来 gain |
| 迁移 | transfer procedural hypothesis，target local 校准 | source→target 存在稳定 invariants | source knowledge造成 negative transfer | equal target budget 下 Source-assisted 优于 Target-only；否则关闭跨域 Skill claim |
| 搜索效率 | cheap screen → expensive full Consumer；独立 future validation | cheap signal 保持候选排序 | proxy rank 不稳定 | 被 cheap screen 淘汰的 candidate 很少成为 full-eval winner；否则 proxy 不应进入核心搜索 |

---

# C–D. 三条相互替代的最小方法路线

## 路线 A：Open-ended Structural Harness Evolution

### 一句话

让 LLM 直接作为 **Harness/Workflow engineer**：根据完整 success/failure trace 修改 Workflow、Observation procedure、Skill 或 Harness control；维护开放探索档案，但只有独立验证通过的完整 policy 才部署。

### 必须保留

- Task × Consumer × observable Pattern；
- executable Workflow / typed structural edit；
- real Consumer evaluator；
- success + failure experience；
- hard legality / leakage boundary；
- no-op deployment fallback。

### 可以删除或降级

- persistent UID qualification 作为主方法；
- 当前某个固定 Scope narrowing family；
- numerical effect Router 作为核心；
- “每个旧 Skill 必须先安全晋升才能继续被搜索”的 lineage。

### 最低验证实验

固定同样的：

```text
LLM
edit space
proposal budget
Consumer evaluation budget
development data
```

比较：

1. Frozen Harness；
2. Random structural edits；
3. Deterministic heuristic edits；
4. LLM structural Harness evolution。

必须同时证明：

- LLM updater 改变了候选/Workflow 分布；
- independent next-window population utility 提高；
- 相同部署 gate 下 harm 未增加；
- 至少一次有用 modification 来源于非当前-best ancestor，或证明开放 archive 没必要。

### 退出条件

若在预注册 budget 内：

> LLM structural updater 不能稳定超过 equal-budget random/deterministic edit，或开发集 gain 在 next-window 消失，

则停止把“open-ended self-Harness evolution”作为主 claim，转路线 B/C。

---

## 路线 B：Evidence-to-Skill Compilation and Revision

### 一句话

不开放整个 Harness 自改，而把核心进化对象收缩成**可证伪 procedural Skill**：

```text
WHEN / OBSERVE
→ HOW TO PREPARE
→ HOW TO VERIFY
→ CONTRAINDICATION
→ EVIDENCE FOR / AGAINST
→ FALSIFIER
```

反馈可以 Add / Revise / Split / Merge / Retire Skill；Fast Agent 根据 Skill 生成当前 Workflow，而不是 UID 查表。

这是本轮我认为**最平衡、也最接近你最初研究目标**的路线。

### 必须保留

- dynamic Workflow generation；
- Pattern observation；
- Task/Consumer-conditioned quality meaning；
- positive/negative/conflict episodes；
- Skill revision；
- Consumer-grounded verification。

### 可以删除

- arbitrary whole-Harness rewriting；
- entity eligibility state machine 作为中心机制；
-复杂 numerical CATE/Router；
-大规模 open-ended archive。

entity evidence 可以留在 Skill 的 evidence refs 中，但不是 Skill 本体。

### 最低验证实验

同一个 Source→Target setting，等预算比较：

1. Target-only Agent；
2. Raw Episode retrieval；
3. Compiled Skill；
4. Compiled + feedback-revised Skill。

成功必须是：

```text
Source experience
→ Skill abstraction
→ unseen Target 的 candidate/search 改变
→ 少用 Consumer evaluations 或更高 future utility
→ Target feedback 再修订 Skill
→ 下一个 window 再提升
```

如果只是“旧经验卡让 LLM 更容易选到 Hampel”，不算。

### 退出条件

如果 Compiled Skill：

- 不优于 raw episode retrieval；
- 不优于 equal-budget Target-only；
- 或 revision 后 next-window 不能改善；

则没有“Skill abstraction”增量，停止该路线。

---

## 路线 C：Consumer-Geometry-First Set-Level Workflow Evolution

### 一句话

为了先把科学问题做干净，暂时只选一种 Consumer geometry：

> **训练数据 preparation → pooled/global Consumer 完整重训 → future population utility。**

Agent 不学习“第 i 条 series 应不应该处理”的 unit CATE，而直接进化**整个训练池的数据准备 policy/workflow**。

Pattern/entity 信号只用于 diagnosis 和 proposal，不作为需要独立成立的 causal effect。

### 必须保留

- Agent 自主生成/修订 Workflow；
- Pattern/Task/Consumer observations；
- complete retraining evaluator；
- full-population objective；
- next-window validation；
- no-op baseline。

### 可以删除

- per-entity deployment qualification 主机制；
- unit-level treatment effect claim；
- model routing；
- 同时支持 training preparation + serving context 两套 geometry；
- 复杂 entity history model。

### 最低验证实验

对完整 training pool：

```text
H0 Workflow
vs
Agent revision
vs
Random revision
vs
Deterministic revision
```

每个候选完整重新训练同一 Consumer，在未来 population 上评价。

细粒度 per-series effect 只用于解释 failure，不直接 promotion。

### 退出条件

若完整 set-level effect：

- 在 next-window 无法复现；
- 或只有在 routing/model 一起变化时才存在；
- 或 cheap feedback 完全无法预测 full retrain ordering，

则说明“pooled training-data readiness”这一 geometry 本身当前不适合作为主要自进化 substrate。

此时可以另立 **frozen serving-context adaptation** 研究问题，但那是换 estimand，不是给当前方法打补丁。

---

# 三条路线如何选

如果首要目标是**论文创新性和 Agent 身份最强**，选 A。

如果首要目标是**保留 Agent 自主性，同时把实验成本和研究风险控制住**，我优先推荐 **B**。

如果首要目标是**尽快得到因果解释最干净、不会继续陷入 per-entity Scope 争论的主实验**，选 C。

三条不能一起作为 v1。最多选一条主路线，另外两条作为 competing hypotheses / fallback。

---

# E. 对当前“目标对齐”实验的判断

当前这个实验很重要，但它是：

> **alignment diagnostic，而不是 positive method proof。**

## 它已经能够排除什么

第一，它排除了：

> “只要把当前 Scope 再收窄，安全性改善就会免费转化成更好的整体方法。”

至少在当前开发证据中，收窄同时降低 harm 和 utility，因此安全—效用存在真实 trade-off。

第二，它排除了：

> “当前选择目标与总体评价目标可以不加区分地互换。”

当前观察已经说明 selection criterion 和 population evaluation target 并非自然等价。

第三，它说明：

> deterministic Scope revision 在执行层确实可以改变行为。

所以当前问题不能再简单归因于“规则没执行成功”。

第四，它使“继续加 entity admission gate”失去自动正当性。新的 Gate 若要成为方法，必须证明它改善最终 target，而不仅仅使行为更保守。

## 它不能证明什么

它不能证明：

- entity-local evidence 是正确的最终表示；
- UID qualification 是必要机制；
- Pattern 没有稳定规律；
- 没有更好的 effect model；
- no-op 对总体永远最优；
- 换 Consumer 会解决问题；
- 当前 Harness 架构必须保留；
- 低当前分数/不安全的 candidate 不能作为 sandbox exploration ancestor；
- Agent 不能形成可迁移 Workflow/Skill；
- 未来窗口不存在另一种可学习 heterogeneity。

最关键的是：

> 当前实验只比较了一个有限的 narrowing / target-alignment 方案，并没有比较“结构性 Harness revision 是否能改变搜索空间本身”。

因此不能从：

```text
某个 scope 收窄效果不好
```

外推出：

```text
应该建立更复杂的 entity-level qualification system
```

也不能外推出：

```text
effect heterogeneity 根本无法迁移
```

---

# 最终研究定位

我建议把论文问题从：

> “怎样安全决定哪个 entity 可以执行某个 preparation program？”

抬回：

> **How can an LLM agent turn noisy, delayed, consumer-grounded data-preparation feedback into reusable and revisable workflow knowledge that improves future decisions across changing time-series contexts?**

其中真正独特的 Gap 是：

1. 时序 data readiness 的 utility 是 Task/Consumer conditional；
2. 数据 preparation 可能作用于共享训练池，因此 credit 存在 interference；
3. Consumer feedback 昂贵、延迟、带噪，无法像 FunSearch / ADAS 那样廉价评价大量候选；
4. 可迁移知识必须从局部 action–response 中被**编译为结构性 Workflow/Skill hypothesis**；
5. 探索过程中必须容纳失败和暂时不安全的假设，但部署必须相对安全 baseline 独立验证；
6. 最终证据必须来自未来/独立场景，而不是被反复用于 Harness 搜索的开发窗口。

因此，当前最值得写进 Introduction 的机制缺口不是：

> “已有方法没有考虑 missing fraction / entity evidence。”

而是：

> **Existing data-preparation methods typically optimize a fixed pipeline or policy, while recent self-evolving agent systems assume cheap and directly verifiable feedback. Time-series data readiness sits between these regimes: the appropriate intervention is task- and consumer-dependent, may interact through a shared downstream model, and is judged only through expensive and delayed downstream feedback. The central challenge is therefore not merely choosing a preprocessing action, but learning how to revise and transfer the agent’s data-preparation workflow from such feedback without conflating exploration, deployment safety, and population utility.**

这比继续沿“更好的 Scope”往下推进，更接近 SelfEvolvingHarnessTS 最初应该回答的研究问题。