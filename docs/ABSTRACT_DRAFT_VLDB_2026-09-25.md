# VLDB 摘要草稿（2026-09-25）

按用户要求，暂不提 PatchTST。v2 已填入今晚两个补充实验（朴素上下文卡对照、新起点留出验证）的结果，改动见文末“v2 修改说明”。

## Working title

**Learning Consumer-Aware Data Preparation Skills for Time-Series Forecasting with an Agentic Harness**

## Abstract (≈250 words; v2 after the two supplementary experiments)

Preparing data for a time-series forecaster — augmenting its training set or assembling the history it reads at inference — is usually treated as a model-agnostic step. It is not: the same augmentation program reduces the error of a small MLP by 23 percentage points (pp) but that of DLinear by only 8 pp, and fine-tuning a pretrained foundation model (Time-MoE) with any of twelve augmentation programs leaves it 30–48 pp worse than using it zero-shot. We present an agentic harness that learns consumer-aware data-preparation skills offline and applies them without feedback. A Slow agent distils source-domain experience — real agent trajectories plus scripted outcomes of candidate preparations — into compact per-domain skill cards, which are selected on separate cases and frozen. At deployment, a Fast agent reads a card, inspects only data visible at that time, and commits a preparation plan: a training-augmentation program for trainable forecasters, or an inference-context policy (history length and normalization) for a frozen foundation model. No downstream score is ever returned to it. On 40 test cases from electricity, traffic and solar data, disjoint from the learning cases, domain skills improve the zero-feedback agent by 17.7 pp (95% CI [11.5, 24.8]) for an MLP and by 7.5 pp ([0.6, 14.2]) for DLinear, matching the best fixed programs selected on source data and exceeding AutoDA-Timeseries by 24.8 pp. For Time-MoE, where every fine-tuning variant loses to zero-shot, the same harness learns to prepare inference context instead: domain cards beat the no-card agent by 6.4 pp ([2.5, 12.9]) with 33% fewer tokens, and on 40 never-scored forecast origins the frozen cards still lead by 3.9 pp ([0.2, 8.2]). Experience, not general knowledge, drives these gains: a card written from general forecasting knowledge alone steers the agent toward short contexts and trails the learned cards by 20 pp on both test sets.

## 数字出处（全部可追溯）

| 摘要里的数字 | 出处 |
|---|---|
| MLP 上 23 pp，DLinear 上 8 pp（同一个 NoMix 程序） | Test 40 例上 NoMix 相对不增强：MLP +23.1（§5.11.5 主表）、DLinear +8.49（§5.11.10）。同一批 24 个 Source 案例上对应为 +12.95 / +7.98（DEV-AUG-DLINEAR-SOURCE）；如果要和 Time-MoE 的 Source 读数放在同一句，可改用 13 / 8。 |
| Time-MoE 微调比零样本差 30–48 pp | TSFM Source 零样本核对（24 例，12 个方案全部为负，−29.7 ~ −48.5）；这是 Source 读数 |
| MLP 上域卡 − 无卡 +17.7 [11.5, 24.8]，高于 AutoDA 24.8 pp | DEV-AUG-OFFLINE-SKILL、DEV-AUG-MAIN-COMPARISON（§5.11.3、§5.11.5） |
| DLinear 上 +7.5 [0.6, 14.2] | DEV-AUG-DLINEAR-OFFLINE-SKILL（§5.11.10） |
| “matching the best fixed programs” | MLP：域卡 − NoMix −1.01，− Source 固定方案 +0.71；DLinear：新卡 − NoMix +0.15，− Source 固定方案 +1.01（区间都跨零） |
| Time-MoE 域卡 − 无卡 +6.4 [2.5, 12.9]，− 统一上下文 +3.3 [1.1, 5.8]，token 少 33% | DEV-TSFM-CONTEXT-CARD（§5.11.11） |
| 新起点留出：域卡 − 无卡 +3.9 [0.2, 8.2]；域卡 − 朴素卡 +20.1 [10.8, 29.5] | DEV-TSFM-CONTEXT-CARD §6（新起点 40 例，冻结部署；`result_fresh.json`） |
| 朴素卡落后约 20 pp（两批都成立） | Test：+20.15 [11.8, 29.1]；新起点：+20.11 [10.8, 29.5] |

## 需要统一或核对的措辞

- **“test cases … disjoint from the learning cases”**：Source / Select / Test 三组实体互不相交，时间也分开。但 Test 已在多个开发包里用过，正文要如实写为开发期的复验；摘要可以写 disjoint，不宜写 “unseen” 或 “held-out benchmark”。
- **pp 的定义**：相对于未经准备的基线（不增强，或 192 点零样本）的误差降低百分点，三个域等权平均。
- **Time-MoE 的比较都在同一个 Consumer 内部**：长上下文的收益只和 Time-MoE 自己的 192 点零样本比，不与 MLP / DLinear 的数字放在同一张表里做绝对比较，因为输入长度不同。

## v2 修改说明（2026-09-25 01:2x）

- 删去“超过最好的统一上下文 3.3 pp”。这个优势在 Test 上成立（[1.1, 5.8]），但在新起点上只剩 +1.2 [−1.0, +3.1]，没能复现。正文可以写“与 Source 上选出的最佳固定上下文相当，并由零反馈 Agent 在部署时自行得出”，不写“超过”。
- 新增新起点留出结果（+3.9 [0.2, 8.2]）和朴素卡对照（两批都约 20 pp）。
- 首句去掉 “On the same development cases”：23 / 8 pp 是 Test 上的 NoMix 读数（MLP +23.1、DLinear +8.5），30–48 pp 是 Time-MoE 的 Source 读数，三者不是同一批案例。
