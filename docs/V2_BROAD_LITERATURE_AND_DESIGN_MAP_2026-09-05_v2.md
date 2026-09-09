# V2 Broad Literature and Design Map

**Project:** SelfEvolvingHarnessTS / Fable  
**Date:** 2026-09-05  
**Status:** Independent literature and design review  
**Boundary:** 本轮只做调研与方法设计地图；不修改代码、阈值、合同、既有工件，不运行实验，不提交 Git。

---

# 1. Executive Summary

## 1.1 当前最重要的 8 个结论

### 结论 1：`没有实体自身证据 → 不给予执行权` 的方向是合理的，但最成熟的对应物不是 “entity allowlist”

当前 Method v2 把 transferable `Program + PatternScope` 与 target-local `EntityEvidence` 分开，并规定未评估实体不部署。这个设计直接针对项目当前观察到的两个事实：部署期特征不能可靠预测未来受害者；新进入 Scope 的实体承担了大量严重伤害。

跨领域最接近的成熟形式是：

- **Safe Policy Improvement / baseline bootstrapping**
- **Conservative contextual bandit**
- **Selective deployment / reject option**
- 再叠加 **personalization**

SPIBB 的核心原则就是：

> 对证据不足的 state-action，不相信新策略，而是 bootstrap 回已有 baseline。

这和本项目中的 `raw / incumbent fallback` 非常接近。

因此论文中不建议把这个机制命名成：

> entity allowlist

更准确的是：

> **evidence-bounded personalized deployment with baseline fallback**

关键文献：
- Laroche et al., *SPIBB*, ICML 2019: https://proceedings.mlr.press/v97/laroche19a.html
- Thomas et al., *High Confidence Policy Improvement*, ICML 2015: https://proceedings.mlr.press/v37/thomas15.html
- Kazerouni et al., *Conservative Contextual Linear Bandits*, NeurIPS 2017: https://papers.nips.cc/paper_files/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html

---

### 结论 2：当前 `positives >= k` 是一个很好的安全 baseline，但不适合作为最终最成熟的方法

目前 v2 用：

\[
positives_e \ge k
\land
\neg quarantined_e
\]

决定某 entity 是否有 program-model 执行权。

它的问题是完全没有区别：

- 两次非常强的正反馈；
- 两次刚好越过 material line 的弱反馈；
- 两次非常久以前的反馈；
- 同一 regime 下的两次反馈；
- 两个独立窗口的反馈。

也没有显式表达 uncertainty。

更成熟的替代方式是：

\[
LCB_\alpha(\Delta U_{e,p})>\epsilon
\]

同时：

\[
UCB_\alpha(Risk_{e,p})<\delta
\]

即执行权由**后验效用与风险的不确定性**控制，而不是纯计数控制。

High-Confidence Policy Improvement、SPIBB 和 Conservative Contextual Bandits 都是这一思想的成熟来源。

---

### 结论 3：解决 “ID memorization” 最成熟、最适合你们的方法不是更复杂的 Pattern Router，而是 hierarchical partial pooling

这是本轮调研中我认为最重要的设计发现。

`IntelligentPooling` 面对的场景和你们非常相似：

- 不同 user 对相同 treatment 反应不同；
- 每个 user 自身样本很少；
- response 会随时间变化；
- 又不能给每个人完全独立训练一个 policy。

它使用 **mixed-effects Bayesian reward model**，自动学习应该在多大程度上共享群体信息、多大程度上个性化。

迁移到你们：

\[
g_{e,t,p}
=
f_\theta(
Pattern_{e,t},
Program_p,
Task,
Consumer
)
+
b_e
+
\epsilon
\]

其中：

\[
b_e\sim\mathcal N(0,\sigma_e^2)
\]

新 entity 没有历史时：

\[
b_e\approx0
\]

但 uncertainty 很大。

随着自身反馈增加，entity posterior 才逐渐偏离 population/Pattern prior。

所以：

> **Pattern 决定 prior；entity evidence 决定 posterior execution right。**

这比：

```text
PatternScope
+
ID -> positive count
```

更能抵抗“只是 ID lookup”的审稿攻击。

关键文献：
- Tomkins et al., *IntelligentPooling: Practical Thompson Sampling for mHealth*, Machine Learning 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC8494236/

---

### 结论 4：不能简单把当前问题包装成 CATE / Individual Treatment Effect

Athey–Wager policy learning、Generalized Random Forest 等领域确实研究：

\[
X_i\rightarrow\text{who should receive treatment}
\]

并且可以估计 heterogeneous treatment effect。

但是当前 pooled Consumer 存在一个很重要的问题：

> 对训练 entity A 的数据进行 preparation，会改变共享的 program model，随后可能影响服务 entity B。

也就是存在 interference。

因此单纯定义：

\[
\tau_e=
Y_e(program)-Y_e(raw)
\]

然后把它解释成 entity-level causal treatment effect，很可能违反常规的 unit-level independence / no-interference 假设。

所以当前论文中更安全的术语应当是：

> **paired downstream Action–Response effect / evidence**

而不是直接声称 causal ITE。

相关文献：
- Athey & Wager, *Policy Learning With Observational Data*, Econometrica 2021: https://doi.org/10.3982/ECTA15732
- Athey, Tibshirani & Wager, *Generalized Random Forests*, Annals of Statistics 2019: https://doi.org/10.1214/18-AOS1709

---

### 结论 5：training preparation 和 serving/model routing 拆开，是当前 v2 最值得保留的部分之一

最合理的最终决策至少应该有两个变量：

\[
a^{train}_{s,t}
\in
\{raw,prepare\}
\]

以及：

\[
a^{route}_{e,t}
\in
\{M_{raw},M_{program}\}
\]

如果 serving context 本身也要 transform，则进一步：

\[
a^{ctx}_{e,t}
\in
\{raw,prepare\}
\]

这使得系统能够分别回答：

1. Program 本身有效吗？
2. 哪些 training rows 应参与 program model？
3. 哪些 serving entities 应看到 transformed context？
4. 哪些 entities 应切换到 program model？

当前项目把这些决策合起来时，route 同时放大收益和尾部伤害。这个拆分不是额外工程复杂化，而是在修复当前 estimand 混合问题。

Learning-to-Defer 和 FFORMA 等工作提供了 hard/soft routing 的成熟机制类比。

关键文献：
- Mozannar & Sontag, *Consistent Estimators for Learning to Defer to an Expert*, ICML 2020: https://proceedings.mlr.press/v119/mozannar20b.html
- Montero-Manso et al., *FFORMA*, IJF 2020: https://doi.org/10.1016/j.ijforecast.2019.02.011

---

### 结论 6：“没有证据永远 Raw”虽然安全，但会产生永久低 coverage，应该升级成 Safe Probe

Conservative bandit 并不是：

> 不知道就永远执行 baseline。

而是：

> 在保证 baseline safety budget 的同时有选择地探索。

所以 target cold start 更合理的状态机是：

```text
UNSEEN
  ↓
PROBE
  ↓
PROBATION
  ↓
ACTIVE
  ↓
QUARANTINED / REVOKED
```

Source-domain knowledge 可以决定：

- 什么值得 probe；
- 哪些 Program 风险高，应 avoid；
- probe 顺序是什么；

但不能直接给予：

> full execution right。

因此完整原则应该是：

> **Transfer what to probe, not whom to trust.**

---

### 结论 7：HEC-1 的真正失败和 delayed/aggregated bandit 的 attribution problem 很接近

Delayed, Aggregated Anonymous Bandit 研究了更极端的情况：

> reward 晚到，而且 arrival 后甚至不知道对应哪个 action。

项目中的 pooled downstream reward 虽然不是完全 anonymous，但如果不拆：

```text
Program
training scope
serving context
model route
```

本质上也会产生 attribution 混合。

因此下一版 Harness 必须把：

> **Decision Receipt**

作为一等状态，而不是普通 log。

关键文献：
- Pike-Burke et al., *Bandits with Delayed, Aggregated Anonymous Feedback*, ICML 2018: https://proceedings.mlr.press/v80/pike-burke18a.html

---

### 结论 8：最推荐的方法不是 v2 原样，也不是推翻 v2，而是 “Hierarchical Evidence-Bounded Skill + Two-Decision Deployment”

最终我最推荐：

> **Architecture B + Architecture C**

即：

1. `Program / Pattern / Task / Consumer` 形成 transferable response prior；
2. 每个 target entity 有 target-local posterior；
3. execution right 由 posterior confidence 决定；
4. unknown entity 进行 bounded probe；
5. training preparation 与 serving route 分开；
6. stale evidence decay；
7. serious harm quarantine；
8. reusable Skill 的修改仍需 downstream validation、versioning 和 re-encounter。

当前 hard entity ledger 可以留下来作为非常重要的 **Architecture A baseline**。

---

# 2. Literature Map

本轮共筛查约 **50 篇候选工作**，保留以下核心论文作为设计地图。

## A. Time-Series Data Preparation / Transformation Selection

| Paper | Year / Venue | Problem | Feedback | Unit | Transfer / Safety | 与项目关系 |
|---|---|---|---|---|---|---|
| Salles et al., *Nonstationary Time Series Transformation Methods* | 2019, KBS | 不同非平稳 transformation 如何影响预测 | validation forecast loss | series | validation selection | **最重要自然 Readiness 动机** |
| TSPred | 2022, Neurocomputing | preprocessing→model→inverse→evaluation pipeline | rolling validation | series/dataset | explicit pipeline | 强 Static/Search baseline |
| FFORMS | 2023, Journal of Forecasting | series features→best model | historical forecasting error | series | meta-learning | Pattern→Action 的直接近邻 |
| FFORMA | 2020, IJF | features→model mixture weights | forecast loss | series | soft routing | Soft raw/program route 类比 |
| AutoAI-TS | 2021, SIGMOD | TS preprocessing/model/HPO automation | validation | dataset | pipeline search | AutoML reviewer baseline |
| AegisTS | 2026, preprint | 多质量问题 agentic cleaning | upstream + downstream reward | dataset/pipeline | hierarchical RL | TS cleaning 最近工作 |

稳定链接：
- Salles 2019: https://doi.org/10.1016/j.knosys.2018.10.041
- TSPred: https://doi.org/10.1016/j.neucom.2021.09.067
- FFORMS: https://doi.org/10.1002/for.2963
- FFORMA: https://doi.org/10.1016/j.ijforecast.2019.02.011
- AutoAI-TS: https://arxiv.org/abs/2102.12347
- AegisTS: https://arxiv.org/abs/2605.04902

Salles 等在 262 条竞赛与真实宏观时序上的实验发现，合适 transformation 对大量序列带来很大预测改善，而且 validation 可以帮助选择适合 particular series 的 transformation。

TSPred 将 transformation、pre/post-processing、decomposition、forecasting 与 accuracy assessment 显式组合为 pipeline。

FFORMS 则直接建立：

\[
TS\ features\rightarrow best\ model
\]

说明基于 series characteristics 学 action prior 本身是成熟思路。

AegisTS 进一步把 MTS cleaning 定义成 order + cleaning-method joint optimization，并用 upstream/downstream 双阶段 reward，但重点仍然是 missing、outlier、constraint violations 等 cleaning issues。

---

## B. Downstream-Utility-Driven Data Cleaning

| Paper | Venue | Feedback | Adaptation unit | Transfer | 关键启发 |
|---|---|---|---|---|---|
| ActiveClean | 2016 | model loss / gradients | record | weak | 清最影响模型的记录 |
| BoostClean | 2017 | clean test labels / model accuracy | detector-repair | weak | **反例：依赖 clean labels** |
| Learn2Clean | WWW 2019 | downstream quality metric | prep pipeline | Q-policy | RL 搜 preprocessing sequence |
| Raha | SIGMOD 2019 | ≤20 labeled target tuples | tuple/value | historical detector configs | Source prior + tiny target supervision |
| Baran | PVLDB 2020 | ≤20 labeled target tuples | value/tuple | pretrained context | Shared prior + target calibration |
| HoloClean | 2017 | constraints/statistical signals | cell/value | probabilistic | 没有下游 delayed utility |

稳定链接：
- ActiveClean: https://arxiv.org/abs/1601.03797
- BoostClean: https://arxiv.org/abs/1711.01299
- Learn2Clean: https://doi.org/10.1145/3308558.3313602
- Raha: https://doi.org/10.1145/3299869.3324956
- Baran: https://doi.org/10.14778/3407790.3407801
- HoloClean: https://arxiv.org/abs/1702.00820

ActiveClean 会根据当前模型结构优先清理对模型结果影响最大的 records，而不是先追求一个与任务无关的 clean score。

Raha 和 Baran 很值得借鉴，因为它们都体现：

> 大量 target labels 并非必须，历史/共享知识 + 少量 target-local evidence 可以工作。

---

## C. AutoML / Algorithm Selection / Meta-Learning

最关键不是 AutoML 的算法细节，而是三种 knowledge representation。

### Feature → action

FFORMS：

\[
\phi(series)\rightarrow model
\]

### Feature → soft action weights

FFORMA：

\[
\phi(series)
\rightarrow
(w_1,\ldots,w_K)
\]

### Dataset/task embedding

Dataset2Vec 学习跨 schema 的 dataset-level representation，并用于 meta-learning/HPO。

Task2Vec 用 probe network Fisher information 表示 task，并据此选择适合新 task 的 pretrained expert。

稳定链接：
- Dataset2Vec: https://doi.org/10.1007/s10618-021-00737-9
- Task2Vec: https://openaccess.thecvf.com/content_ICCV_2019/html/Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.html

这些工作共同说明：

> Dataset/Pattern representation 可以承担跨域 **prior**。

但项目现有诊断又说明：

> prior 不能被误写成 execution certificate。

---

## D. Safe Online Learning / Contextual Bandits

### High-Confidence Policy Improvement

候选 policy 只有在概率意义上的 performance lower bound 足够高时才被返回。

链接：
https://proceedings.mlr.press/v37/thomas15.html

### SPIBB

最关键的一句话：

> When the target does not know, bootstrap to the baseline.

链接：
https://proceedings.mlr.press/v97/laroche19a.html

这是当前 `raw fallback` 最强的理论类比。

### Conservative Contextual Linear Bandit

探索 policy 的同时，整个 trajectory 上保持相对 baseline 的安全约束。

链接：
https://papers.nips.cc/paper_files/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html

### Delayed Rewards

Gigli & Stella 进一步说明：部分 reward 尚未 mature 时，需要显式建模 delay uncertainty，而不能把“尚未观察到成功”直接视为失败。

链接：
https://proceedings.mlr.press/v244/gigli24a.html

---

## E. Personalized Decision / Dynamic Treatment

最相关的是 **IntelligentPooling**。

它使用 Gaussian mixed-effects reward model，实现：

- population sharing；
- person-specific effect；
- limited data rapid personalization；
- nonstationarity extension。

链接：
https://pmc.ncbi.nlm.nih.gov/articles/PMC8494236/

这基本就是 RQ2 想寻找的成熟答案。

Athey & Wager 的 policy learning 说明可以基于 heterogeneous benefit 学“谁应该 treatment”，但其 causal identification assumptions 需要非常谨慎地迁移到 pooled Consumer。

链接：
https://doi.org/10.3982/ECTA15732

---

## F. Continual Learning / Drift

经典 nonstationary bandit 使用：

- Discounted UCB
- Sliding-Window UCB

让旧证据自然衰减。

链接：
https://doi.org/10.1007/978-3-642-24412-4_16

最近的 distributionally robust policy learning 则直接研究 concept drift 下 worst-case policy value。

链接：
https://proceedings.mlr.press/v267/wang25cm.html

对项目而言不需要立刻上复杂 DRO，但一定需要：

```text
last_evidence_time
evidence_age
decay
change/reset
version
```

否则一个很久以前的 positive receipt 会永久授予执行权，这在 temporal readiness 中不合理。

---

## G. Selective Prediction / Conformal Risk Control

Selective Classification 的核心不是提高所有样本 accuracy，而是允许：

> coverage ↓，换取 conditional risk ↓。

链接：
https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

这正好支持：

```text
DEPLOY
PROBE
ABSTAIN
```

作为一等选择。

Adaptive Conformal Inference 则进一步证明，在非平稳 sequence 下，calibration 本身也可以持续更新。

链接：
https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html

但是要注意：

> conformal coverage guarantee ≠ downstream policy utility guarantee。

不能直接写成“conformal 可以证明 Program safety”。

---

## H. Agent Memory / Skill Library / Harness Evolution

### Reflexion

把反馈转化成 verbal reflective memory，而不修改模型权重。

https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html

### ExpeL

自主收集 training-task experiences，抽取自然语言 insight，推理时检索相关经验。

https://ojs.aaai.org/index.php/AAAI/article/view/29936

### Voyager

持续维护 executable skill library，在环境交互中构造与复用技能。

https://arxiv.org/abs/2305.16291

### WikiSkill

进一步把：

```text
raw execution experience
→ accumulated wiki knowledge
→ executable skill
```

显式分开。

https://arxiv.org/abs/2608.27454

这与项目非常接近。

但是 WikiSkill 最关键的缺失是：

> 它没有你们这种 entity-specific downstream harm 下的 execution-right ledger。

### Self-Harness

三个步骤：

```text
Weakness Mining
→ Harness Proposal
→ Proposal Validation
```

而且强调 bounded/minimal harness modifications 和 held-out regression testing。

https://arxiv.org/abs/2606.09498

这是你们最接近的 Harness lifecycle 工作。

### TTHE

直接在 test-time 利用 unlabeled execution traces 修改 executable harness。

https://arxiv.org/abs/2607.08124

但它和你们最大的制度差异是：

> TTHE 允许 evaluation stream 本身驱动 adaptation；你们 Final held-out 明确禁止反馈写回。

### TimeClaw

TimeClaw 的 Explore→Compare→Distill→Reinject 特别支持：

> experience 应该先比较，再蒸馏，最后才 reusable。

https://arxiv.org/abs/2605.10038

另一个 TimeClaw generalist Harness 版本也使用 executable temporal tools、experience-driven capability evolution 和 episodic multimodal memory：

https://arxiv.org/abs/2606.05404

---

## I. Routing / Mixture of Experts

Learning-to-Defer 把 predictor 与 rejector 分开：

\[
x\rightarrow
\begin{cases}
model\\
expert/defer
\end{cases}
\]

对项目而言完全可以变成：

\[
x_e\rightarrow
\begin{cases}
M_{raw}\\
M_{program}
\end{cases}
\]

链接：
https://proceedings.mlr.press/v119/mozannar20b.html

FFORMA 则提供 soft mixture：

\[
\hat y
=
\sum_m w_m(x)\hat y_m
\]

所以一个合理的实验性替代方案是：

\[
\hat y_e
=
(1-w_e)\hat y_{raw}
+
w_e\hat y_{program}
\]

但 soft routing **不天然更安全**：

只要 \(w_e>0\)，bad program model 就会污染 prediction。

因此只能作为实验 arm。

---

## J. Cross-Domain Transfer / Dataset-as-Unit Evaluation

WILDS 最重要的不是方法，而是 evaluation philosophy：

> 真实 shift 必须以自然 domain/dataset split 明确体现，而且 OOD performance 要与 ID 分开报告。

https://proceedings.mlr.press/v139/koh21a.html

DomainBed 更进一步指出：

> domain generalization 中如果 model selection protocol 不一致，比较本身就失真；在公平条件下简单 ERM 往往非常强。

https://openreview.net/forum?id=lQdXeXDoWtI

这个警告和你们项目高度相关：

> A5、A3、Static 必须共享同一 target feedback budget、Consumer、Program space、风险门和 sealed evaluation。

否则所谓 “cross-domain evolution” 很容易变成额外调参预算带来的收益。

---

# 3. Closest-Work Matrix

符号：

- Y = 主要支持
- P = 部分/近似
- N = 不具备

| Work | No clean truth | Delayed downstream | Executable prep Program | Task/Consumer | Entity-local evidence | Cross-domain knowledge | Promotion/revoke | Held-in→held-out | Harness self-edit |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Method v2 target** | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| AegisTS | Y/P | P | Y | P | N | weak | N | P | N |
| TSPred | Y | N | Y | P | N | pipeline reuse | N | Y | N |
| Learn2Clean | Y | N | Y | model/metric | N | weak | N | P | N |
| FFORMS | Y | historical | N | Y | N | Y | N | meta-test | N |
| SPIBB | N/A | batch reward | N | state/action | P | representation-level | **Y** | offline | N |
| IntelligentPooling | N/A | sequential | N | Y | **Y** | population prior | posterior action | online | N |
| Self-Harness | evaluator | outcome | Harness Program | model/task | N | persistent Harness | **Y** | **Y** | **Y** |
| TTHE | Y | execution proxy | Harness Program | task | N | persistent | proxy commit | test-stream | **Y** |
| WikiSkill | task dependent | outcome | **Y Skill** | task | N | **Y** | weak | benchmark | **Y** |
| TimeClaw | Y | execution | executable routines | Y | N | **Y** | weak | train/test | experience evolution |
| Baran | small labeled target | N | repairs | context | target labels | **Y** | N | target eval | N |

## Matrix verdict

目前没有发现一篇工作同时完整覆盖：

\[
\boxed{
CrossDomainSkill
+
ExecutableDataProgram
+
DelayedConsumerFeedback
+
TargetLocalExecutionRight
+
Revision/Revocation
+
SealedFastOnly
}
\]

但是必须明确：

> 每一个单独组件几乎都有 prior art。

所以 novelty 不能写成：

- 第一个用 memory；
- 第一个做 safe routing；
- 第一个做 preprocessing search；
- 第一个做 personalized selection；
- 第一个修改 Harness。

比较可能成立的是：

> **新的问题组合 + transferable Skill 与 target-local execution authority 的分层机制 + governed cross-domain evaluation。**

Novelty 目前仍应标记：

> **uncertain**

---

# 4. Design Pattern Cards

## Pattern Card 1 — Evidence-Bounded Eligibility

### 核心机制

\[
Deploy(e,p)
=
\mathbf 1[
LCB_\alpha(\Delta U_{e,p})>\epsilon
]
\]

并且：

\[
RiskUCB_{e,p}<\delta
\]

否则 baseline raw。

### 来源

SPIBB / HCOPE / Conservative Bandit。

### 对应 HEC-1

解决：

- new entrant severe harm；
- PatternScope 无法稳定预测 future safety。

### 风险

coverage 下降。

### 最小实验

Development replay：

```text
hard k-count
vs
LCB posterior
```

比较：

- cumulative gain
- harm
- coverage
- time-to-active

---

## Pattern Card 2 — Hierarchical Partial Pooling

### 核心机制

\[
\Delta_{e,t,p}
=
\beta^\top z_{e,t,p}
+b_e+\epsilon
\]

\[
b_e\sim N(0,\sigma_b^2)
\]

其中：

\[
z=
Pattern
\times Program
\times Task
\times Consumer
\]

### 对应 HEC-1

不再要求 deployment-visible feature 完美预测 victim，而只要求它对 population prior 有一点信息。

### 风险

如果所有 entity response 完全独立，则 population sharing 会伤害。

### 最小实验

三臂：

```text
Global only
Hard entity
Hierarchical
```

---

## Pattern Card 3 — Safe Exploration with Abstention

三状态：

```text
RAW
PROBE
DEPLOY
```

新 entity 不是：

```text
RAW forever
```

而是：

```text
RAW by default
+ bounded informative probe
```

### 评价

- probe cost
- probe harm
- activation rate
- eventual coverage
- utility

---

## Pattern Card 4 — Pending Reward Ledger

每个 action 保存：

```text
receipt_id
entity
Program
Program_version
Pattern
train_action
context_action
route_action
Consumer
decision_time
feedback_due
feedback_status
outcome
```

状态：

```text
PENDING
MATURED
CENSORED
```

### 关键纪律

PENDING 不能算：

```text
failure
```

也不能算：

```text
support
```

---

## Pattern Card 5 — Temporal Evidence Decay

例如：

\[
w_i=
2^{-\Delta t/h}
\]

其中 \(h\) 为固定 half-life。

或者 sliding window。

### Drift 后行为

```text
ACTIVE
→ PROBATION
```

必要时：

```text
posterior -> group prior
```

而不是删除历史 receipt。

---

## Pattern Card 6 — Two-Stage Preparation / Routing

### Training decision

\[
q_{train}(s)
\]

### Serving decision

\[
q_{route}(e)
\]

如果需要：

\[
q_{context}(e)
\]

### 对应 D5

把：

> 谁改变 program model

和：

> 谁消费 program model

分开。

---

## Pattern Card 7 — Soft Raw / Program Mixture

\[
\hat y_e
=
(1-w_e)\hat y_{raw}
+w_e\hat y_{program}
\]

其中：

\[
w_e=P(\Delta U_e>0\mid evidence)
\]

或其他 monotonic mapping。

### 风险

可能把局部 bad model contamination 传播到更多 entity。

因此不是默认方法。

---

## Pattern Card 8 — Cross-Domain Skill Compiler

分三层：

```text
Experience
↓
Knowledge
↓
Executable Skill
```

### Experience

immutable receipts。

### Knowledge

```text
Pattern × Program × Task/Consumer
→ response/risk hypothesis
```

不含 target entity IDs。

### Skill

```text
ProgramSpec
Pattern prior
Risk
Version
Provenance
```

Target execution right 不放 Skill 内。

这是 WikiSkill 思想最值得迁移的地方。

---

# 5. Candidate Method Architectures

## Architecture A — Evidence-Bounded Entity Eligibility++

这是对当前 v2 的最小增强。

### State

```text
TransferableSkill:
  Program
  PatternScope
  Task
  Consumer
  risk_signatures
  version
  provenance

TargetEntityState:
  entity_id
  posterior_gain
  posterior_uncertainty
  harms
  last_window
  pending
  status
```

### Held-in

PatternScope：

> 决定 probe eligibility。

Entity posterior：

> 决定 deploy eligibility。

### Fast-only

```text
if ACTIVE:
    program route
else:
    raw
```

### Cold start

```text
UNSEEN -> PROBE
```

### Drift

```text
ACTIVE -> PROBATION
```

### 为什么可能失败

仍然高度 entity-specific。

### 成本

最低。

### 地位

应该保留成 main safety baseline。

---

## Architecture B — Hierarchical / Meta-Response Skill

这是我最推荐替代 hard ID ledger 的方案。

### Model

\[
\Delta U_{e,t,p}
=
f_\theta(
Pattern_{e,t},
Program_p,
Task,
Consumer
)
+
b_{dataset}
+
b_e
+
\epsilon
\]

### Transferable

\[
f_\theta,\sigma_b, Program
\]

### Target-local

\[
b_{dataset},b_e
\]

entity ID 永不跨 dataset transfer。

### Cold start

新 entity：

\[
b_e=0
\]

但 variance 大。

所以 source knowledge 可以说：

> seasonal + Program A historically promising

但不能直接说：

> entity 47 should deploy A。

### Execution Right

\[
LCB(\Delta U_{e,p})>\epsilon
\]

且 risk gate 通过。

### 为什么比 hard ledger 更合理

因为它同时解决：

#### Group-only 的问题

不会假设所有同 Pattern entities 一样。

#### ID-only 的问题

不会让每个 entity 从零独立学习。

### 主要风险

如果 pooled Consumer interference 很强：

\[
\Delta U_e
\]

并不是 entity-local response。

所以 Architecture B 与 Architecture C 应组合。

---

## Architecture C — Decoupled Train-Preparation / Serving Route

### State 1

```text
TrainPrepState
```

回答：

> 哪些 training series 进入 program-model prepared corpus？

### State 2

```text
ServeRouteState
```

回答：

> 哪些 entity 应消费 program model？

二者不共用一个 eligibility。

### Held-in

可以做：

\[
2\times2
\]

```text
raw train / raw route
raw train / program route
prepared train / raw route
prepared train / program route
```

如果 context 也是决策，再单独扩展。

### Cold start

Serving route 默认 Raw。

Training eligibility 可以更广，因为其风险和 serving risk 不完全相同。

### 成本

最高，需要更多 Consumer fits。

### 价值

它是当前 D5 证据最直接的方法响应，而不是从文献硬凑出来的模块。

---

# 6. Recommended Synthesis

最终推荐不是 A/B/C 三选一，而是：

\[
\boxed{
B\ \text{Hierarchical Evidence}
+
C\ \text{Decision Decoupling}
}
\]

A 作为关键 baseline。

## 6.1 Transferable State

```text
TransferableSkill:
    program_spec
    program_semantics
    task_consumer_signature
    pattern_response_prior
    population_response_parameters
    risk/failure_signatures
    known_contraindications
    version
    provenance
    source_dataset_support
```

不允许：

```text
target entity ID
target held-out outcome
source execution entitlement
```

## 6.2 Target-local State

```text
TargetState:
    dataset_posterior
    entity_posteriors
    pending_probe_receipts
    execution_status
    quarantine
    recency
    train_prep_state
    route_state
```

新 dataset 到来时 reset。

## 6.3 Fast Held-In Loop

```text
observe PatternCard

retrieve portable Skill hypotheses

for candidate Program:
    infer group/pattern prior

    if local posterior insufficient:
        RAW or bounded PROBE
    else:
        posterior safety gate

execute

record immutable receipt

when downstream feedback matures:
    update target posterior
    update risk
    update drift state

if repeated structural failure:
    propose bounded Skill/Harness edit

paired validate

if valid:
    version/promote
else:
    reject
```

## 6.4 Execution Right

推荐第一版：

\[
Deploy(e,p)=
[
LCB_{90/95\%}(\Delta U_{e,p})>\epsilon
]
\land
[
RiskUCB_{e,p}<\delta
]
\land
[
\neg quarantine
]
\]

这里的 90% / 95% 等数值不能根据 exposed replay 挑，应在正式 contract 前冻结。

## 6.5 Training Side

development 中比较两个合法候选：

### Option 1

prepare 所有 PatternScope matches。

### Option 2

只 prepare evidence-qualified entities。

选定后进入正式 protocol。

不能 Final 后再决定。

## 6.6 Serving Side

默认：

```text
ACTIVE      -> Program model
uncertain   -> Raw model
quarantine  -> Raw model
```

## 6.7 Promotion / Revision / Revocation

### Local execution lifecycle

```text
UNSEEN
→ PROBE
→ PROBATION
→ ACTIVE
→ QUARANTINED
```

### Skill lifecycle

```text
DRAFT
→ VALIDATED
→ ACTIVE
→ REVISABLE
→ revised version
→ independent validation
→ ACTIVE_v2
```

严重失败可以：

```text
ACTIVE -> REVOKED
```

---

# 7. Self-Evolution Definition

这个部分建议未来直接进论文。

## Level 1 — Target-local Deployment Adaptation

包含：

- entity posterior update；
- route change；
- quarantine；
- probe schedule；
- eligibility update。

这只能叫：

> adaptation / personalization

不能作为完整 self-evolution 证据。

## Level 2 — Reusable Skill Evolution

必须出现完整链：

```text
Failure
↓
Attribution
↓
Persistent Skill/Harness Revision
↓
Independent Validation
↓
Promotion
↓
Later Similar Re-encounter
↓
Revised Skill > Frozen Ancestor
```

才可以叫：

> reusable Skill evolution。

## Level 3 — Cross-Domain Harness Evolution

更强：

```text
Source datasets
↓
Reusable Skill accumulation
↓
Unseen target
↓
Same target feedback budget
↓
A5 learns/adapts faster or safer than A3
↓
held-out survives
```

其中最关键控制是：

\[
Budget_{A5,target}=Budget_{A3,target}
\]

否则 reviewer 会认为 A5 只是有更多 optimization budget。

## 不能叫 Self-Evolution 的事件

- Memory append；
- `positives += 1`；
- entity allowlist update；
- routing table update；
- 一个 Skill ADD 后直接 REVOKE；
- LLM 写了新版 Skill 但行为没变化；
- 行为变化但 downstream 没验证；
- exposed replay 改善。

---

# 8. Experiment Implications

## 8.1 数据结构

建议最终实验采用：

```text
Domain
└── Dataset
    └── Series
        └── Windows / Origins
```

统计含义分别是：

### Window

repeated measurement。

不能当 independent sample。

### Series

entity adaptation / safety unit。

### Dataset

**跨域主实验的主要统计单位。**

### Domain

transfer axis。

## 8.2 Dataset-as-unit Main Experiment

完整：

```text
Source D1,D2,...
↓
A5 knowledge accumulation

Unseen Target D*
↓
held-in A/B
↓
target calibration
↓
freeze
↓
sealed Q*
```

每个 Target 都比较：

```text
Static
A3-reset
K0-fixed
A5-online
Validation search
```

## 8.3 Main Baselines

### Data readiness / search

1. Raw
2. Best fixed safe Program
3. TSPred-style validation search
4. FFORMS-style Pattern→Program mapper
5. Random-valid Program search

### Agent

6. Frozen LLM
7. Retrieval-only memory
8. deterministic Harness updater
9. random Harness edit
10. LLM Harness updater

### Evolution

11. Static
12. A3 target-only
13. K0 fixed
14. A5 full

### Method v2 family

15. hard entity count
16. hierarchical posterior
17. feature-only router
18. train/route coupled
19. train/route decoupled
20. optional soft route

---

# 8.4 Treatment-Instantiation Metrics

HEC-1 的经验表明，这组指标必须成为正式结果的一部分。

每个 arm 报：

```text
planned unit
complete Fast decision
candidate supplied
non-identity candidate
probe instantiated
admitted
deployed
online/frozen discordant
budget-mediated discordant
outcome matured
Skill retrieved
Skill affected action
Skill revised
revision validated
revision re-encountered
```

否则：

\[
online=frozen
\]

可能只是：

> 两边都没有实际 treatment。

---

# 8.5 Intermediate Evaluation：不只看最终 End-to-End

建议论文的中间性能分成六级。

## E0 — Natural Headroom

回答：

> 自然数据上到底有没有 preparation space？

指标：

- raw vs best fixed；
- menu oracle；
- per-series oracle；
- non-raw winner proportion；
- action entropy。

## E1 — Transformation Fidelity

只在 controlled positive-control 上。

例如 injected：

```text
missing
outlier
timestamp irregularity
```

测：

- imputation RMSE；
- outlier F1；
- restoration error；
- invertibility；
- distortion。

Baseline：

```text
linear interpolation
seasonal interpolation
Hampel
Winsorize
AegisTS
```

注意：

> E1 只能证明 repair capability，不能证明 natural readiness。

## E2 — Response / Targeting Quality

最重要。

对 development-only oracle action：

\[
p_i^*
\]

评价：

- Top-1 / Top-k oracle recall；
- normalized regret；
- sign prediction；
- calibration；
- harm prediction；
- abstention。

Baseline：

- global fixed；
- FFORMS style；
- hard entity；
- hierarchical；
- nearest-neighbour；
- random。

## E3 — Execution-Right Quality

回答：

> 哪些 decision 得到了合理执行权？

指标：

### Safety

\[
P(gain<-\delta)
\]

### Coverage

\[
P(Deploy)
\]

### Calibration

例如 predicted positive probability vs empirical positive rate。

必须画：

> safety–coverage curve。

否则 hard ledger 很容易因为 deployment 极少而“看起来最安全”。

## E4 — Harness Revision Quality

评价：

### Diagnosis accuracy

failure 是否定位到了真正有害行为？

### Behavioral realization

Harness edit 后目标行为改变了吗？

### Revision precision

被 promote 的 revisions 中多少最终 held-out 正向？

### Non-target regression

目标 cohort 修复后其他 cohort 是否受损？

Baseline：

```text
random edit
deterministic heuristic edit
LLM edit
```

## E5 — Cross-Domain Evolution Quality

重点不是只看 final utility。

应报：

### A5 vs A3

\[
\Delta U
\]

### Adaptation speed

例如：

\[
\#ConsumerFits
\rightarrow
first\ safe\ positive
\]

### Safe coverage speed

\[
feedback\ units\rightarrow coverage
\]

### Reusable revision rate

真正发生：

```text
revision
→ validation
→ later re-encounter
```

的 Skill 数。

## E6 — End-to-End

最终：

- downstream task utility；
- mean / median gain；
- harm rate；
- max single-series harm；
- worst-dataset；
- coverage；
- abstention；
- LLM cost；
- Consumer fit cost。

---

# 8.6 Risk Metrics

只报 average gain 不够。

建议：

\[
MeanGain
\]

\[
MedianGain
\]

\[
HarmRate=
P(g_i<-\delta)
\]

\[
MaxHarm
\]

\[
Q_{0.05}(g)
\]

以及 CVaR-style：

\[
CVaR_{\alpha}(-g)
\]

OPRA 提供 contextual-bandit 下通过 CDF 估计 CVaR 等 risk functionals 的正式框架：
https://proceedings.neurips.cc/paper/2021/hash/c7502c55f8db540625b59d9a42638520-Abstract.html

但当前系统如果没有 randomized logging propensity / overlap：

> 不可以直接声称自己能合法使用 IS/DR OPE。

---

# 8.7 Ridge 与 TSFM 的顺序

建议继续保持：

> 先让 Ridge 把 treatment identity、route、safety mechanism 讲清楚，再在方法站住后换 TSFM 验 Consumer generalization。

TSFM 不应该重新开启 method search。

---

# 9. Novelty and Reviewer-Risk Map

## Risk 1 — “只是 ID lookup”

这是当前 v2 最大 reviewer risk。

### reviewer 会看到

```text
EntityEvidence[entity]
```

然后说：

> 你只是记住了哪些 series 曾经成功。

### 必须化解

1. transferable state 不包含 entity ID；
2. hierarchical prior baseline；
3. unseen entity cold-start；
4. leave-entity-out；
5. leave-dataset-out；
6. A5 应在还没有 entity history 时就降低 probe cost 或提高 candidate quality。

如果这些做不到：

> hard entity ledger 不适合作为 headline method。

---

## Risk 2 — “只是 Contextual Bandit”

这个批评部分成立。

确实可以抽象成：

\[
context
\rightarrow action
\rightarrow delayed reward
\]

所以不要否认。

真正区别应该是：

- action 是 executable data transformation；
- action 改变训练数据和 downstream model；
- 可能存在跨 entity interference；
- Skill/Harness 是 persistent editable state；
- source→target knowledge accumulation 是 claim；
- Final feedback wall 比普通 online bandit 更严格。

---

## Risk 3 — “只是 AutoML”

如果 A5 最终只是在每个 Target：

```text
试几个 preprocessing
选最好一个
```

那就是 TSPred / Learn2Clean / AutoML 范式。

必须证明：

\[
A5>A3
\]

至少在以下之一：

- equal-budget held-out utility；
- adaptation speed；
- fewer Consumer fits；
- less harm；
- faster coverage。

否则 cross-domain evolution 不成立。

---

## Risk 4 — “只是 Agent Memory”

Memory 写入不是贡献。

需要证明：

```text
stored experience
↓
compiled reusable knowledge
↓
changes executable Skill/Harness
↓
validated
↓
later helps
```

这也是为什么 WikiSkill 是危险近邻。

---

## Risk 5 — “只是 Self-Harness on TS”

Self-Harness 已经有：

```text
weakness
proposal
validation
```

你们真正不同必须来自：

- downstream Consumer-mediated readiness；
- no clean truth；
- entity-level risk；
- delayed feedback；
- cross-domain knowledge；
- target-local execution authority；
- data/model route decomposition。

---

## Risk 6 — “Pattern Scope 没有作用”

论文应该写：

> Pattern is a prior, not a certificate.

而不是继续寻找更复杂 Pattern features，直到 AUC 看起来好。

---

## Risk 7 — “安全只是因为一直 Raw”

所以所有 safety table 必须同时给：

```text
harm
gain
coverage
abstention
```

最好画 Pareto frontier。

---

# 10. Falsification and Stop Rules

## 10.1 Entity Evidence 方案什么时候算失败？

在 fresh Target 上如果：

1. continuing entities 不比 new entrants 更稳定；
2. entity evidence 对 future safety 没有增量预测；
3. coverage 长期很低；
4. 同 safety 下 hierarchical/probe 方法 utility 明显更高；

则停止继续调：

```text
k=1 / 2 / 3
```

hard entity allowlist 退出 main method。

---

## 10.2 Hierarchical 方案什么时候失败？

如果：

- population/Pattern effect 接近 0；
- model 最后只剩 entity intercept；
- leave-entity-out 无 transfer；
- source prior 经常有害；
- posterior uncertainty 与 future safety 无关；

则结论是：

> 当前 observable state 对 response generalization 不足。

此时应该：

```text
probe + abstain
```

而不是继续堆 feature engineering。

---

## 10.3 Route Decoupling 什么时候失败？

如果 fresh 数据上：

- pooled route 不再承担主要 harm；
- train/route decomposition 不稳定；
- decoupling 减少 harm 的同时几乎完全消灭 gain；
- per-channel 没有改善 safety–utility frontier；

则不能硬把 per-channel 做成最终系统。

---

## 10.4 Cross-Domain Self-Evolution 什么时候失败？

最关键停止规则：

多个 fresh Target 上：

\[
A5-A3\approx0
\]

并且：

- adaptation speed 无改善；
- coverage 无改善；
- safety 无改善；
- source Skills 经常被 target revoke；
- 没有 reusable revision→re-encounter；

那么必须关闭：

> cross-domain self-evolution claim。

不要再开新 domain 找一个正结果。

转 Track B：

> **governed target-local adaptive Data Readiness under delayed downstream feedback**

仍然可能是一篇完整且有价值的论文。

---

## 10.5 LLM Harness Optimizer 什么时候失败？

公平条件：

```text
same Harness edit space
same Program space
same target evidence
same candidate budget
same downstream fits
```

若：

```text
LLM updater
≤ deterministic updater
≈ random edit
```

或者：

- behavior discordance 仍然稀疏；
- Skill edits 多是文字变化；
- revision 不存活；
- later re-encounter 没收益；

那么：

> LLM 不应是 headline 方法。

它只保留：

> proposal supplier。

---

# 11. 最终推荐的方法蓝图

一句话版本：

> **The transferable object is a Program-level readiness Skill; the deployable object is a target-local posterior execution right.**

完整：

```text
SOURCE DOMAINS
    ↓
Experience receipts
    ↓
Cross-domain consolidation
    ↓
Transferable Skill:
    Program
    Pattern/Task/Consumer prior
    failure knowledge
    uncertainty hyper-prior
    version/provenance
    ↓

NEW TARGET
    ↓
reset target-local state
    ↓
Pattern/context
    ↓
hierarchical prior
    ↓
bounded probe
    ↓
delayed downstream feedback
    ↓
entity posterior
    ↓
LCB execution-right gate
    ↓
TrainPrep decision
+
ServeRoute decision
    ↓
ACTIVE / RAW / QUARANTINE
    ↓

repeated failure?
    ↓ yes
bounded Harness/Skill revision
    ↓
paired validation
    ↓
promotion / rejection
    ↓

FREEZE
    ↓
HELD-OUT FAST-ONLY
```

---

# 12. Introduction / Research Gap 反推

这轮广泛调研后，论文 Introduction 不建议从：

> “Agent 可以自己优化 Harness”

开始。

更强的逻辑是：

## Motivation 1 — Data readiness is utility-relative

自然时序即使没有显式 corruption：

> 不同 transformation 对具体 downstream Consumer 的 utility 差异依然非常大。

Salles 等的自然时序结果直接支持这一点。

链接：
https://doi.org/10.1016/j.knosys.2018.10.041

## Motivation 2 — Existing systems can search, but safe transfer is unsolved

TSPred / Learn2Clean / AutoAI-style：

> 可以在当前 dataset 上 search。

AegisTS：

> 可以利用 downstream reward 优化 cleaning pipeline。

但是这些还没有解决：

> 一个在 Source/Support 上有效的 Program，什么时候有资格在新的 entity / window / dataset 上获得执行权？

AegisTS 也没有 cross-domain persistent Skill + target-local execution-right separation。

## Motivation 3 — Pattern similarity is insufficient

FFORMS/FFORMA 证明 feature-based meta-selection 可行。

但本项目当前 evidence 表明：

> Pattern/Support safety 不能稳定预测 future harm。

因此不能简单：

\[
Pattern\rightarrow Program
\]

而应：

\[
Pattern\rightarrow prior
\]

再：

\[
TargetEvidence\rightarrow execution\ right
\]

## Motivation 4 — Experience reuse requires governance

Agent literature已经证明：

- memory 可以积累；
- skill 可以编译；
- harness 可以修改。

但在本场景：

> bad experience reuse 会真正伤害 downstream entity。

所以最关键问题变成：

> **How does an Agent earn, revise, and lose the right to execute a reusable data-readiness Skill?**

这可能是比“如何生成 Skill”更强的论文核心。

---

# 13. 最值得向老师汇报的最终判断

如果下次和老师讨论，我建议把本轮调研压成以下结论：

### 1

当前 v2 的安全直觉是对的：

> 不能因为 Pattern 相似就给新 entity 执行权。

### 2

但纯：

```text
Entity ID + positive count
```

容易被 reviewer 认为 ID memorization，而且 cold start 过度保守。

### 3

其他领域成熟做法不是二选一：

```text
global prior
vs
personal history
```

而是：

> **hierarchical partial pooling**。

### 4

所以方法升级成：

\[
Pattern/Task/Consumer
\rightarrow
population\ prior
\]

\[
entity\ feedback
\rightarrow
posterior
\]

\[
posterior\ LCB
\rightarrow
execution\ right
\]

### 5

同时保留当前 D5 推出的：

> training preparation 与 service model routing 两决策解耦。

### 6

source knowledge 跨域真正迁移的应该是：

```text
Program
response prior
failure signatures
probe policy
```

而不是：

```text
entity IDs
execution rights
```

### 7

A5 的关键实验不应该只问：

> Final score 是否比 A3 高。

还应该问：

> 相同 Target feedback budget 下，Source accumulated knowledge 是否让新 Target **更快、更安全、更少 Consumer fits** 地获得有效执行权？

### 8

如果多个 fresh Target 上 A5 对 A3 在：

```text
utility
safety
coverage
adaptation speed
```

都没有优势，就应诚实关闭 cross-domain self-evolution claim，而不是继续调整配置。

---

# 14. Core Literature Table

| # | Paper | Year / Venue / Status | Paper Type | Problem | Feedback / Supervision | Unit of Adaptation | Knowledge Representation | Safety Mechanism | Transfer Mechanism | Evaluation Unit | Relevance |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Salles et al., Nonstationary TS Transformation | 2019 KBS | TS prep | transform selection | validation forecast loss | series | transform choice | none | validation selection | series | closest natural readiness evidence |
| 2 | TSPred | 2022 Neurocomputing | pipeline | prep/model pipeline | rolling validation | series/dataset | explicit pipeline | none | pipeline reuse | series/dataset | strong static/search baseline |
| 3 | FFORMS | 2023 JF | meta-learning | feature→model | historical loss | series | feature-action mapping | none | cross-series | series | Pattern→Program prior analogue |
| 4 | FFORMA | 2020 IJF | meta-learning/MoE | soft model weighting | historical loss | series | feature→weights | soft mixture | cross-series | series | soft raw/program routing analogue |
| 5 | AutoAI-TS | 2021 SIGMOD | AutoML | TS prep/model/HPO | validation | dataset | pipeline graph | budgeted search | meta search | dataset | AutoML baseline |
| 6 | AegisTS | 2026 preprint | agentic cleaning | MTS cleaning pipeline | upstream + downstream | dataset/pipeline | hierarchical policy | reward/gates | learned policy | dataset/task | closest TS cleaning work |
| 7 | ActiveClean | 2016 | task-aware cleaning | impactful record cleaning | model loss | record | priority policy | partial guarantees | model-aware sampling | record/dataset | downstream-driven intervention |
| 8 | BoostClean | 2017 | cleaning ensemble | detector/repair selection | clean labels/accuracy | detector-repair | boosted ensemble | none | library selection | dataset | warning: clean-label dependence |
| 9 | Learn2Clean | WWW 2019 | RL prep | prep sequence | downstream metric | dataset/pipeline | Q-policy | none | learned pipeline policy | dataset | closest AutoML-style route |
| 10 | Raha | SIGMOD 2019 | error detection | detector choice | small target labels | tuple/value | detector feature rep | sample-efficient | historical configs | tuple/dataset | source prior + target calibration |
| 11 | Baran | PVLDB 2020 | data repair | contextual correction | small target labels | tuple/value | unified context | precision filtering | pretrained + target | tuple/dataset | partial-pooling style prior |
| 12 | HCOPE | ICML 2015 | safe policy improvement | deploy only safe candidate | offline trajectories | policy | confidence bound | explicit LCB acceptance | repeated improvement | policy | promotion analogue |
| 13 | SPIBB | ICML 2019 | safe offline RL | safe improvement from baseline | offline batch | state-action | baseline + learned | bootstrap uncertain actions | representation transfer | policy | closest raw-fallback analogue |
| 14 | Conservative Contextual Bandit | NeurIPS 2017 | safe online learning | explore under safety | sequential reward | context-action | confidence set | cumulative baseline constraint | contextual gen. | round | safe probe analogue |
| 15 | Delayed Anonymous Bandit | ICML 2018 | delayed learning | delayed aggregated reward | delayed reward | arm/round | delay structure | none | bandit estimation | round | attribution warning |
| 16 | Bootstrap Your Conversions | UAI 2024 | delayed contextual bandit | partially observed delayed reward | delayed/censored | context-action | posterior delay/reward | uncertainty | contextual | round | pending reward analogue |
| 17 | OPRA | NeurIPS 2021 | risk/OPE | estimate tail policy risk | logged bandits | policy | IS/DR CDF | CVaR | OPE | policy | evaluation only; assumptions matter |
| 18 | Robust Multi-Source Policy Learning | AISTATS 2025 | robust transfer | policy across heterogeneous sources | observational | source/context | DR + minimax | worst-source robustness | multi-source | source/policy | cross-domain prior |
| 19 | IntelligentPooling | ML 2021 | personalization | sparse per-user decisions | repeated rewards | user | Bayesian mixed effects | posterior uncertainty | partial pooling | user/round | strongest anti-ID analogue |
| 20 | Policy Learning with Observational Data | Econometrica 2021 | causal policy | who to treat | outcomes | individual | DR treatment value | constrained policy | heterogeneity | policy | useful but interference warning |
| 21 | Generalized Random Forests | AoS 2019 | heterogeneous effects | local effect estimation | outcomes | local neighborhood | adaptive forest | confidence intervals | feature locality | individual | alternative response model |
| 22 | SW-UCB / D-UCB | ALT 2011 | drift | nonstationary rewards | sequential | recent rounds | window/discount | recency | temporal | round | evidence decay |
| 23 | Selective Classification | NeurIPS 2017 | abstention | coverage-risk tradeoff | labeled validation | instance | confidence selector | reject option | calibration | instance | deployment abstention |
| 24 | Adaptive Conformal Inference | NeurIPS 2021 | temporal uncertainty | shift-aware calibration | sequential labels | time | adaptive calibration | long-run coverage | online update | time | drift risk analogue |
| 25 | Self-Harness | 2026 preprint | harness evolution | mine/propose/validate edits | traces + evaluator | Harness | versioned edits | held-out validation | persistent Harness | task/version | closest Harness lifecycle |
| 26 | TTHE | 2026 preprint | test-time Harness evolution | evolve harness from traces | proxy judge | Harness | executable harness | proxy commit | stream persistence | episode | mechanism/warning |
| 27 | WikiSkill | 2026 preprint | skill evolution | experience→knowledge→skill | task outcomes | skill | layered memory/skill | implicit | cross-model | task | closest representation idea |
| 28 | TimeClaw exploratory | 2026 preprint | agent experience | explore/compare/distill | execution metrics | task/routine | distilled experience | tool dropout | reuse | task | TS experience analogue |
| 29 | Learning to Defer | ICML 2020 | routing | act vs defer | supervised outcome | instance | predictor+rejector | explicit defer | classifier gen. | instance | serving route analogue |
| 30 | WILDS | ICML 2021 | OOD evaluation | realistic shift | held-out domain | dataset/domain | metadata | protocol | explicit OOD split | dataset/domain | evaluation blueprint |

---

# 15. Falsification Summary Table

| Hypothesis | Falsification evidence | Recommended stop/action |
|---|---|---|
| Hard entity eligibility improves safety usefully | coverage collapses; hierarchical dominates same safety frontier | demote hard ledger to baseline |
| Hierarchical partial pooling transfers | source prior harmful; leave-entity/dataset-out null; uncertainty uncalibrated | stop feature/model complexity, use safe probing |
| Route decoupling matters | fresh 2×2 shows little route-attributed harm or no better frontier | retain simpler coupled design |
| Per-channel Consumer solves route harm | harm decreases but gain collapses or pattern not reproducible | do not make per-channel default |
| A5 cross-domain knowledge helps | A5≈A3 on utility/safety/coverage/speed across fresh targets | close Track A; report target-local adaptation |
| LLM is useful Harness optimizer | equal-budget LLM≤det/random edits; no surviving revisions | LLM only proposal supplier |
| Pattern knowledge transfers | feature router≈fixed; Pattern prior offers no cold-start benefit | Pattern only descriptive/probe prior |
| Self-evolution occurs | no revise→validate→re-encounter chain | call result adaptation/memory update only |

---

# 16. Compact Bibliography by Requested Area

## A. Time-series data preparation / automatic transforms
- Salles et al. 2019: https://doi.org/10.1016/j.knosys.2018.10.041
- TSPred 2022: https://doi.org/10.1016/j.neucom.2021.09.067
- FFORMS 2023: https://doi.org/10.1002/for.2963
- FFORMA 2020: https://doi.org/10.1016/j.ijforecast.2019.02.011
- AutoAI-TS 2021: https://arxiv.org/abs/2102.12347
- AegisTS 2026 preprint: https://arxiv.org/abs/2605.04902

## B. Downstream-utility-driven cleaning
- ActiveClean: https://arxiv.org/abs/1601.03797
- BoostClean: https://arxiv.org/abs/1711.01299
- Learn2Clean: https://doi.org/10.1145/3308558.3313602
- Raha: https://doi.org/10.1145/3299869.3324956
- Baran: https://doi.org/10.14778/3407790.3407801
- HoloClean: https://arxiv.org/abs/1702.00820

## C. AutoML / algorithm selection / meta-learning
- FFORMS / FFORMA above
- Dataset2Vec: https://doi.org/10.1007/s10618-021-00737-9
- Task2Vec: https://openaccess.thecvf.com/content_ICCV_2019/html/Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.html

## D. Safe online learning / contextual bandits
- HCOPE: https://proceedings.mlr.press/v37/thomas15.html
- SPIBB: https://proceedings.mlr.press/v97/laroche19a.html
- Conservative contextual bandit: https://papers.nips.cc/paper_files/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html
- Delayed anonymous feedback: https://proceedings.mlr.press/v80/pike-burke18a.html
- Delayed conversions: https://proceedings.mlr.press/v244/gigli24a.html
- OPRA: https://proceedings.neurips.cc/paper/2021/hash/c7502c55f8db540625b59d9a42638520-Abstract.html

## E. Personalized decision / dynamic treatment
- IntelligentPooling: https://pmc.ncbi.nlm.nih.gov/articles/PMC8494236/
- Policy Learning with Observational Data: https://doi.org/10.3982/ECTA15732
- Generalized Random Forests: https://doi.org/10.1214/18-AOS1709

## F. Drift / temporal validity
- Garivier & Moulines 2011: https://doi.org/10.1007/978-3-642-24412-4_16
- Distributionally Robust Policy Learning under Concept Drifts: https://proceedings.mlr.press/v267/wang25cm.html

## G. Selective prediction / conformal risk
- Selective Classification: https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
- Adaptive Conformal Inference: https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html
- Risk-controlling prediction sets: https://doi.org/10.1145/3478535

## H. Agent memory / Skill / Harness evolution
- Reflexion: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
- ExpeL: https://ojs.aaai.org/index.php/AAAI/article/view/29936
- Voyager: https://arxiv.org/abs/2305.16291
- Self-Harness: https://arxiv.org/abs/2606.09498
- TTHE: https://arxiv.org/abs/2607.08124
- WikiSkill: https://arxiv.org/abs/2608.27454
- TimeClaw exploratory: https://arxiv.org/abs/2605.10038
- TimeClaw generalist: https://arxiv.org/abs/2606.05404

## I. Routing / MoE / multi-model deployment
- FFORMA: https://doi.org/10.1016/j.ijforecast.2019.02.011
- Learning to Defer: https://proceedings.mlr.press/v119/mozannar20b.html

## J. Cross-domain transfer / dataset-level evaluation
- WILDS: https://proceedings.mlr.press/v139/koh21a.html
- DomainBed: https://openreview.net/forum?id=lQdXeXDoWtI
- Dataset2Vec: https://doi.org/10.1007/s10618-021-00737-9
- Multi-source robust policy learning: https://proceedings.mlr.press/v258/carranza25a.html

---

# 17. Final Recommendation

本轮最推荐进入下一轮设计审议的候选名称可以暂时写成：

> **Hierarchical Evidence-Bounded Skill Deployment**

其核心不是：

> memorize which entity worked before

而是：

> **learn a transferable response prior, personalize it with target-local downstream evidence, and grant execution rights only when posterior evidence justifies deviation from the raw baseline.**

再加上：

> **Decoupled Training Preparation and Serving Routing**

构成下一版最小方法。

因此当前 Method v2 最适合的定位不是“冻结”，而是：

> **Architecture A：必须保留的最简单安全 baseline。**

真正值得下一步优先验证的是：

\[
\boxed{
Hard\ Entity\ Eligibility
\quad vs\quad
Hierarchical\ Partial\ Pooling
}
\]

在相同 exposed-development replay 中先比较：

- harm；
- coverage；
- recovered oracle headroom；
- cold-start；
- evidence efficiency。

它比现在继续扩 Pattern feature、继续调 entity `k`，或者继续增加 Skill 数量，更有研究价值。

