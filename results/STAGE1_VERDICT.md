# STAGE1_VERDICT — Pattern 条件化数据就绪路由：Stage 1 终局判决

日期：2026-07-05　依据：E-1.1(R2/v2.1) → F0(final) → E-3.2(dev, freeze `f1b6c1b75a4e975c`) →
**Confirmatory(seeds 20–39, freeze `d27acaf5f6c5bf7e`, router `90d8e0dfae87d1e0…`)**
详细数字：`results/E3_2/decision_e32.md`、`results/E3_2_confirmatory/decision_confirmatory.md`。
**协议权威=各 freeze.json + 本文件**；正文旧四方对比（Ours/Enumeration/Lookup/Ablation）已被
A-37/A-39/A-40/A-41 superseded，读者不应手工合并 A-1..A-41（快照见 §4）。

---

## 1. 判决（三层，按证据强度）

**T1 已确认（confirmatory，主 estimand=locked transfer，grouped full-refit B=1000）**
- 可观测 Pattern 特征在 degradation 之外携带**可实现**决策价值（判官口径）：dev 冻结的
  dp_abstain 原样迁移到 holdout，regret 0.274 vs global 0.455 / D-only 查表 0.432 / 连续-D
  0.452（C1–C3 点差 ≫ε 且 full-refit CI 上界全 <0）；学习程序 replication 亦复现（0.260）。
- **安全命题**：degradation-only 剂量路由的 season 结构伤害在 holdout 复现（−0.64/−0.90），
  dp_abstain 将其修复为**全 season 子群均值为正**、最差 full-refit q05 −0.039 > −δ_safe（C4）；
  trend 判官口径增益保留 113%（C5）。
- season 安全修复是**报告器稳健**的：dlinear_scratch 与 chronos 下 dp 在 S_season 全面优于
  两基线；d_lookup 的 season 伤害在 chronos 下同样成立。

**T2 明确未确立（confirmatory 直接证伪或未测）**
- **下游模型无关的全面增益（C6 FAIL，5/6）**：两独立报告器一致显示 trend 重剂量收益**不迁移**
  （dlinear 显著反向 +0.107/+0.083；chronos 方向失败 vs global）。判官（Ridge/FrozenProbe）
  标签合法，但其 trend 增益是**模型类条件化**的——F0"缺剂量维"须改写为"剂量维价值依赖
  下游模型类"。这是 Stage 1 最重要的新信息，直接由预冻结的判官↔报告器分离抓到。
- 全部结构子群全局非劣（overall worst q05 −0.41 = trend 重尾，非 season 式结构损伤）。
- 跨真实 domain 的 Pattern 决策价值（level-2 未做；LODO=留一结构≠跨域，S_both aliasing 已
  兑现为机制边界）。

**T3 附带确立的测量学结论**
- in-sample gate 不可用（A-30 winner's curse）；标签生成必须嵌入 policy outer fold（A-31/A-39
  P0）；便宜 CI 会翻转裁决（F0：单次-nested "显著" vs full-refit 跨 0）；算子身份/边界语义
  bug 能伪造路由价值（S0.7 v_stl/v_median 两案）。这些以 279 项测试+九守卫机器化。

## 2. What remains for the LLM/Agent?（D-3.2a/b：未答）

Stage 1 全部承重证据（E-1.1/F0/E-3.2/confirmatory）**零 LLM**：路由=GBDT/查表、供给=人工
剂量网格、安全=统计 abstain。F0 剂量维由人工网格+评审发现，不得包装为 LLM supply discovery。
LLM 要在本项目承重，必须至少在一向给出判决性证据（对应 [[project_core_critic]] 三脊柱）：

| 候选承重向 | 判决实验 | confirmatory 之后的可行性 |
|---|---|---|
| ①开放供给发现（超出人工网格的新算子/剂量） | Ours-vs-Enumeration（同预算，nested held-out Δ_supply 口径） | 需先修 C6 教训：供给评价必须多报告器，否则"发现"可能只是判官特异伪增益 |
| ②跨域 harness 迁移（level-2 真实 dataset/domain） | 真实域 holdout 上 frozen policy vs LLM 条件化迁移 | level-2 地基实验先行；平衡补样在真实数据不可用 → (vi) 类 residualization 承重 |
| ③漂移下的持续维护 | deploy_stream/reset-free（v4 D1 线） | 机制已有（[[project_refactor_v4]]），须接本冻结策略作 incumbent |
| ④同预算胜固定策略 | LLM proposer vs 冻结 dp_abstain，同 token/时间预算 | dp_abstain 已是强 incumbent（oracle-gap 覆盖度高）——LLM 的空间在 C6 揭示的模型类条件化轴 |

**若四向皆无判决性证据，项目声明收缩为 "pattern-conditioned safe TS preprocessing policy"
（无 agentic/self-evolving 前缀）。** C6 的模型类条件化发现同时打开一条诚实的新轴：
H=f(pattern, task) 需扩展为 f(pattern, task, **downstream model class**)——该扩展目前只是
观察，非声明。

## 3. 记录性降级与欠账

- **E-3.1（单字段可达性）记录性降级**：机制已从"编辑路径进化"转向"固定池策略路由"，可达性
  实验不再承重（16 轮共识③）。
- E-4.1 独立 reporter 欠账由 C6 panel 部分偿付——且其结果（FAIL）证明这笔账必须一直挂在
  每个后续供给/路由声明上。
- anomaly 任务的报告器分离仍未满足（report_target 注records）；classify 线证据在
  [[project_classify_c1]]，未进本 verdict。

## 4. 当前有效协议快照（2026-07-05）

- **语料**：合成四结构×退化网格（build_corpus）+ A-38/A38C 平衡补齐 N=40/槽；dev=seeds 0–19
  +A31e；confirmatory=seeds 20–39+A38C（**已消耗，永不复用**）。
- **动作池（冻结）**：pool_v2.1(7)+f0_median_w9/15/25=10；操作语义经 S0.7-6/8 修复。
- **策略（冻结）**：dp_abstain=8P+2D 特征、GBDT(100/2/0.1/0.7)、E=5 paired-advantage abstain
  κ=1、fallback=v_median；artifact=`frozen_arms.joblib`。
- **评估**：标签嵌入 policy outer fold（e32_nested）；判官=frozen_probe+Ridge、报告器
  panel={dlinear_scratch(S=5), chronos}（不相交）；主 CI=grouped full-refit；common-support
  主口径（排 S_ar）。
- **判据**：D-3.2e 七条（dev）+ C1–C6 布尔表（confirmatory）；ε=0.03、δ_safe=0.05。
- 全部修正案 A-1..A-41 见 `idea/Experiment_Plan_Decision_Experiments.md` 修正案表。

## 5. 下一站（Stage 2 入口）

1. **level-2 真实 dataset/domain holdout**（A-37⑤）：预期增益缩水、安全门必须保住；
   residualization 承重；**报告器 panel 从第一天进入判据**（C6 教训制度化）。
2. LLM 必要性判决实验（§2 表，四选一先做①或④——最便宜且直接回答项目身份）。
3. 剂量×模型类交互的机制刻画（为什么重 median 助 Ridge-head 伤 DLinear/Chronos）——
  只作解释性附录，不阻塞主线。
