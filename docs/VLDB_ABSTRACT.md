# VLDB 题目与摘要草稿

更新：2026-09-24。依据 DEV-AUG-OFFLINE-SKILL、NAIVE-CONTROL 和 MAIN-COMPARISON 已完成结果撰写；本稿未提交会议系统。后续直接修订本文件，不另开多个版本。

## 工作题目

**Learning Reusable Data Augmentation Skills for Time-Series Forecasting**

中文：面向时序预测的可复用数据增强技能学习。

## English abstract

Preparing training data for time-series forecasting requires judging transformations by their downstream utility. This utility can conflict with data-plausibility heuristics: a language-model agent may reject strong augmentations that improve prediction. Evaluating candidate transformations through model training for each new case is expensive. We present a framework that learns reusable augmentation skills from offline task feedback. It links agent observations and decisions to executable data transformations and forecasting outcomes, incorporating both agent trials and fixed-program comparisons. A language model generates candidate workflows and principles from this evidence. The candidates are evaluated on separate development cases, and selected skills are frozen for deployment. On a new case, an agent inspects the available training history and constructs an augmentation program without downstream evaluation of candidate programs. We evaluate domain-specific and shared skills on 40 forecasting cases from electricity, traffic, and solar datasets, with disjoint development and test entities, later test periods, and a fixed MLP forecaster. The domain-specific skills reduce domain-averaged normalized mean squared error by 22.7% relative to an unguided agent and by 8.4% relative to task-knowledge guidance, while reducing deployment token usage by 58% relative to the unguided agent. The additional benefit over task-knowledge guidance is concentrated in solar forecasting, where offline evidence reverses a preference for mild augmentation.

## 中文对应稿

时序预测的训练数据准备需要根据下游效用判断数据变换，而这种效用可能与数据外观上的合理性相冲突：语言模型 Agent 可能拒绝实际上有利于预测的强增强。对每个新案例反复训练预测模型来评价候选方案，又会产生较高成本。我们提出一个从离线任务反馈中学习可复用增强技能的框架，将 Agent 的观察与决策、可执行的数据变换和实际预测效果对齐，并纳入固定程序的对照记录。语言模型据此生成工作流和原则，在独立开发案例上评价候选技能后选优冻结。面对新案例，Agent 只观察可用训练历史并构造增强程序，不获取候选程序的下游评价反馈。我们在电力、交通和太阳能三个数据集的 40 个预测案例上评价共享技能和域技能，开发与测试使用不同实体，测试时期更晚，下游模型固定为 MLP。域技能使三域平均归一化均方误差相对无指导 Agent 降低 22.7%，相对任务知识指导降低 8.4%，部署 token 相对无指导 Agent 减少 58%。相对任务知识指导的额外收益主要来自太阳能场景，在该场景中，离线证据改变了对温和增强的偏好。

## 写作依据与口径（不属于摘要正文）

2026-09-24 从主包 result.json 的 40 案例 × 9 臂 × 3 seed 复算，各臂汇总与存量值最大差 6.22e-15。此次为本上下文内数值核对，不是新的实验或独立复验。

| 摘要数字 | 原始读数和计算 |
|---|---|
| 相对无卡的 nMSE 下降 22.7% | (0.4357307385884397 − 0.33695956836120855) / 0.4357307385884397 = 22.6679% |
| 相对朴素卡的 nMSE 下降 8.4% | (0.3676763310591243 − 0.33695956836120855) / 0.3676763310591243 = 8.3543% |
| 部署 token 下降约 58% | (145723 − 61357) / 145723 = 57.8948%；每案例均值，包含重复请求上下文；不是总计算成本降幅 |
| 原有主要配对指标 | 域卡 − 无卡：17.66 pp of None，聚类 95% 区间 [11.46, 24.79]；域卡 − 朴素卡：6.44，[1.47, 11.05] |

nMSE 的百分比是“域内案例平均、再三域等权后的误差”之间的相对差，不是逐案例相对改善的平均。上表的 pp 区间不能直接标到 nMSE 百分比旁边。性能按三域等权，部署 token 沿用 40 条轨迹等权；正文分别定义。

主实验在形成卡片与选择卡片后才打开相应测试结果。朴素卡及 AutoDA 是已曝光 Test 上的后续补充比较；报告其时间顺序，不称全新终验。朴素卡有 2/40 条未完成，按预定规则计作 None；原报告已列明，不删除这两个案例。

朴素卡只生成一次、没有按下游效果选优，而完整学习卡经过选优。该比较支持完整离线学习流程相对一次性知识指导的增益，不独立隔离“结果证据”或“真实轨迹”的因果贡献。

NoMix 的 nMSE 为 0.32964869267686864，优于域卡的 0.33695956836120855；域卡对 NoMix 的配对差为 −1.01 pp，区间 [−3.98, 1.22]。这不是优效或等效证明。域卡对共享卡 +1.37 pp 的区间跨零，因此摘要不主张域级组织优于共享组织。

AutoDA 结果属于统一 MLP、训练预算、特征接口下的受控适配。它仍应出现在实验表和适配说明中；本摘要以无卡和朴素卡比较作为主要证据。官方模块调用一致不能排除适配或优化设置的影响；只增强输入、原样本保留率和概率变化都是机制观察，尚未分别验证为损失原因。

一次性学习 8.44M token、264 次拟合，以及复用的筛选阶段拟合须在正文成本表报告。摘要中的 58% 是部署 token 节约，不宣称当前 40 案例已经摊平全部学习成本。新增 Consumer 结果完成前不写跨模型结论。

## Introduction 接续逻辑

1. 数据增强的价值由下游任务效果决定，观察数据外观不足以判定处理收益。
2. 在每个新案例上反复训练评价候选代价高；通用知识指导能改善决策，但也会固化未经效用验证的偏好。
3. 离线保留“观察—处理—效果”的对应关系，将其组织为可在合法观察条件下执行的经验，并通过独立开发案例选择可复用技能。
4. 实验分别检验交付质量、部署成本、相对普通指导的收益和共享/域技能的差异；固定程序与外部增强方法提供实际参照。

这是当前论文的工作定位。方法部分应给出具体的数据准备机制，并与 Eval-Skill 的已有学习范式区分；本摘要不宣称离线技能学习或按域加载本身是新发明。

## 对应原始产物

- [主表及成对差](../_scratch/dev_aug_main_comparison/main_table.md)
- [逐案例逐 seed 结果](../_scratch/dev_aug_main_comparison/result.json)
- [方法核心](../_scratch/dev_aug_main_comparison/METHOD_CORE.md)
- [成本表](../_scratch/dev_aug_main_comparison/cost_table.md)
- [朴素卡报告](../_scratch/dev_aug_offline_skill/naive/REPORT_NAIVE.md)
- [AutoDA 适配说明](../_scratch/dev_aug_main_comparison/BASELINE_ADAPTATION.md)
