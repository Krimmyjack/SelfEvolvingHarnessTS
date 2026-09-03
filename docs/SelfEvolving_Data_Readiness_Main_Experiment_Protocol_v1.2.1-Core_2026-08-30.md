Self-Evolving Data Readiness Harness  
论文主实验协议 v1.2.1-Core

方法论文优先 · Freshness-corrected · Budget-calibrated · Task-specific Experience Evolution

冻结候选版本：v1.2.1-Core · 2026-08-30 · 状态：FROZEN_P0B_CANDIDATE

# 0. 执行摘要

**最终定位：**本文是一份方法论文主实验协议，而不是大型 Benchmark 建设方案。核心目标是在 Forecasting、Classification、Anomaly Detection 三个任务上，用统一的 Harness 架构和统一的实验纪律，验证 Task/Consumer/Pattern 条件化的 Skill/Memory 是否能在各自任务内部持续积累、修订并带来真实下游收益。

**核心原则：**三任务同表评测，但具体 Experience/Skill 按 Task 隔离；Natural Track 承担能力主结论，Controlled Witness 只承担机制正控；A5 必须同时与 A3、K0、等预算搜索及 AegisTS-style Pipeline Optimization 比较。

- 不再建设 84-episode 的 12×7 大矩阵作为当前论文主线。

- 不要求跨 Task 动作级经验正迁移；跨 Task 只检验隔离、沉默和无负迁移。

- 不为削弱 baseline 或满足预设候选数量而新增 Operator、二步模板或 targeting 能力；Common DSL 只使用当前已有能力，实际覆盖率仅作描述性报告。

- 最终 Query/Test 一次性开封，Outcome 不回流本次 Harness。

- 所有随机方法使用 matched budget；不选择最好 seed/replica。

- DOCX 是唯一内容真源；Markdown 必须从冻结 DOCX 自动完整导出，不再人工维护摘要版。

# 1. 论文定位与中心假设

## 1.1 论文类型

论文定位为 Self-Evolving Data Readiness Harness 方法论文。统一 Benchmark 纪律用于证明方法，而不是将主要贡献扩展成一个新的大规模时序 Benchmark。

## 1.2 中心假设

待检验假设：在固定基础模型、Task-specific Consumer、Typed Workflow Space 和 Target Feedback Budget 下，Task/Consumer/Pattern 条件化 Scope、版本化 Skill，以及 Positive/Conflict/Negative 写回，应使 Harness 在后续未见 Target 上相对冷启动、固定经验、纯检索和等预算搜索获得正向下游效用增量，并保持受控的负迁移风险；最终能否成立完全由预注册实验结果决定。

## 1.3 三任务关系

- 共享：TaskSpec、ConsumerSpec、PatternCard、Typed Workflow、Verifier、Support-A/Support-B/Query、Skill Schema、Scope、Versioning、Write-back 规则。

- 隔离：Forecast Skill Store、Classification Skill Store、AD Skill Store。

- 不要求 Classification Skill 帮助 Forecast，也不要求 Forecast Skill 帮助 AD。

- 跨 Task 主实验只检查错误 Task 的 Skill 是否保持沉默、不会错误供应或部署。

# 2. 研究问题（RQ）与必要对比

| **RQ** | **研究问题**                                 | **关键对比**                               | **可支持主张**                                                            |
|--------|----------------------------------------------|--------------------------------------------|---------------------------------------------------------------------------|
| RQ1    | 完整 Harness 是否提升端到端 Data Readiness？ | A5 vs Identity / Fixed / AegisTS-style     | 完整系统提高下游效用                                                      |
| RQ2    | 历史经验是否优于每个 Target 从零开始？       | A5 vs A3                                   | Experience accumulation 有价值                                            |
| RQ3    | 在线 Gain/Harm 修订是否优于固定经验？        | A5 vs K0（独立 Evolution-mechanism claim） | 只有“真实修订→后续重检索→行为改变→材料改善”全链成立，才支持在线修订有增量 |
| RQ4    | Task/Consumer/Pattern 条件化是否必要？       | Full vs Consumer-blind / No-Scope          | Readiness 是条件化属性                                                    |
| RQ5    | 能否泛化到完全未见 Target？                  | Global Harness vs Target-adapted Harness   | Global transfer 与 target-time adaptation 的增量                          |

# 3. Harness 状态组织与任务隔离

逻辑上的 Global Harness：H_global = {H_shared-runtime, H_forecast, H_classification, H_AD}。每个 Task 内允许 Episode→Skill→Scope→Write-back→Revision；Task 之间只共享 schema、算法与运行时，不共享具体 Action-level Experience。

# 4. 数据集与三段式 Split

每个 Task 均采用 Evolution/Search → Validation → Natural Final。Search 用于积累/修订，Validation 用于冻结预算与消融，Final 只用于一次性能力评测。

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Task</strong></th>
<th><strong>Evolution/Search</strong></th>
<th><strong>Validation</strong></th>
<th><strong>Natural Final-1</strong></th>
<th><strong>Natural Final-2</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Forecasting</td>
<td>KDD Cup 2018 with missing values</td>
<td>KDD 未用于 Evolution 的 Cell / Origin Block</td>
<td>Traffic leftover</td>
<td>Solar-Energy</td>
</tr>
<tr class="even">
<td>Classification</td>
<td>已曝光 UCR Stream（PowerCons、GunPoint family 等）</td>
<td>Epilepsy2（EXPOSED，development / replay reference）+ 已曝光非 Evolution UCR</td>
<td>Adiac<br />
(P0 frozen)</td>
<td>ArrowHead<br />
(P0 frozen)</td>
</tr>
<tr class="odd">
<td>Anomaly Detection</td>
<td>Yahoo S5 已曝光 24 条</td>
<td>Yahoo dev subset 或 exposed NAB subset</td>
<td>Yahoo S5 sealed 41（same-dataset unseen-series）</td>
<td>FINAL_POOL_UNAVAILABLE<br />
(strict exposure audit)</td>
</tr>
</tbody>
</table>

## 4.1 Fresh Dataset 机械选择规则

- Classification Final-A/B：均从官方 UCR 的 fresh pool 机械选择；要求 univariate、length≥150、官方 TRAIN 行数×长度≤100,000、每类 TRAIN 样本≥4、项目内未运行/未看 TEST Outcome；按名称字典序依次冻结前两个结构合格项为 Adiac 与 ArrowHead。Epilepsy2 已开封，只能作 EXPOSED replay/reference，不再承担 Final。

- Classification 的 official TRAIN 按标签排序、类内保持官方行序做确定性 50/25/25 model-fit / Support-A / Support-B 切分；每个 surface 每类至少一行。Official TEST 在 Final 前保持关闭。

- AD Final-2：P0 严格暴露审计确认，官方剩余 NAB series 的 label windows 已被旧 LabelWall 全局加载，因此冻结为 FINAL_POOL_UNAVAILABLE；不得改用宽松 accessor 定义恢复 Final 资格。

- 只有下载、格式、Loader 或计算结构失败可以触发替换；科学结果不好不能替换。

## 4.2 Evolution Stream 规模

| **Episode 类型**               | **每 Task 数量** | **用途**                                         |
|--------------------------------|------------------|--------------------------------------------------|
| Natural heterogeneous          | 4                | 自然 Pattern、缺陷与可修复空间异质性             |
| Natural control / low-headroom | 2                | identity、abstain、负迁移控制                    |
| Controlled witness             | 2                | Headroom、Skill 生命周期、Conflict/Negative 正控 |

每个 Task 默认 8 个 Evolution Episodes；三个 Task 共 24 个。每个 Task 使用 3 个预注册 Replica：Forward / Reverse / Interleaved；禁止选择最好 Replica。Episode 数量不承担“保证 treatment 发生”的职责：凡要检验 A5−K0 的 Task，P0 必须先通过真实 Consumer 事件资格门（≥1 POSITIVE、≥1 CONFLICT/NEGATIVE、≥1 后续相似 Context re-encounter）。资格门不过则 RQ3 记 NOT_EXERCISED，不盲目增加 episode。

# 5. Natural Track 与 Controlled Witness Track

## 5.1 Natural Track：能力主结论

Final 主结论只来自自然数据：Traffic-Natural、Solar-Natural、P0 冻结后的 Fresh UCR-A/B、Yahoo-S5-sealed-41（same-dataset unseen-series）和 P0 冻结后的 Fresh NAB。Final 不为了创造 Headroom 再追加人工缺陷；任何 fresh 名单只有通过 P0 Exposure/Pool Audit 才能转为具体 roster。

预注册 NO_HEADROOM 分支：Final 方法冻结并完成 Query 后，允许使用不进入任何方法视野的冻结 Common-DSL menu-oracle 仅作诊断。如果 identity 与 oracle 的差异低于材料线，则该 Dataset 标记 NO_HEADROOM；此时正确结局是 identity/abstain、零 wrong-promotion、受控额外成本，而不强求清洗产生正 gain。该标签不得用于事前选靶或改变方法。

## 5.2 Controlled Witness：机制正控

仅在 Evolution/Validation 中使用，每个 Task 最多两个预冻结条件（Missing / Measurement Artifact）。用于验证 Program Headroom、Skill 形成、Scope 供应、Conflict/Negative 修订和 re-encounter；不进入 Natural Final 平均。

# 6. 统一 Target Episode 协议

1\. Support-A：候选发现、Failure analysis、Skill/Harness update proposal。

2\. Support-B：独立验证与 Promotion。

3\. Freeze Program / Target-local Harness。

4\. 处理完整 Training Dataset。

5\. 从头训练完全相同的固定 Consumer。

6\. Query/Test 只打开一次；结果不得回流本次 Skill/Memory/Harness。

| **Task**          | **Support-A/B**                              | **Query/Test**                 |
|-------------------|----------------------------------------------|--------------------------------|
| Forecasting       | 现有 40 held-in：20 Support-A + 20 Support-B | 20 held-out future，Horizon=48 |
| Classification    | Official TRAIN 内预冻结分层切分              | Official TEST                  |
| Anomaly Detection | Official / normal training 部分内切分        | Official test / event labels   |

# 7. Consumer 设置

| **Task**          | **Primary Consumer**      | **Primary Metric** |
|-------------------|---------------------------|--------------------|
| Forecasting       | pooled/shared Ridge       | sMASE ↓            |
| Classification    | 当前冻结 Ridge Consumer   | Macro-F1 ↑         |
| Anomaly Detection | 当前冻结 Isolation Forest | Event-F1 ↑         |

## 7.1 Consumer-conditioned 子实验

| **Task**          | **Secondary Consumer**                          |
|-------------------|-------------------------------------------------|
| Forecasting       | per-series / per-channel Ridge                  |
| Classification    | kNN                                             |
| Anomaly Detection | P0 从当前已接通 Consumer 中机械核定，不新建模型 |

Consumer-conditioned 子实验在 Validation Dataset + Final-1 Dataset 上重新执行完整 Target Adaptation；Consumer-blind 只隐藏 Observation 与 Skill Scope 中的 Consumer 信息，真实 Consumer 仍负责评分。

Classification 正式主指标冻结为 Macro-F1，Accuracy 为 secondary。此前 development 中 +0.269、+0.083 等数字采用 Accuracy 口径，只能作为历史机制证据，不能与本协议 Macro-F1 主结果直接数值对比。

# 8. Common Program Space

使用当前已有、已通过 Task Contract 的 Effect-distinct Workflow 空间；不为凑数量新增 Operator、二步模板、Targeting Mode 或执行平台。

- 只使用已有、已通过 Task Contract 的 Operator；不为主实验新增 Operator。

- Identity 永远可用；单步 Workflow 全部纳入。

- 已有且已接通的 Global / per-series targeting 可以按其真实语义使用；缺少某种 targeting 不构成补建要求。

- 已有且已在当前 Task 通过测试的二步 Template 可以使用；不得为扩大候选数新增 Template、放宽 Task Contract 或加入未验证执行语义。

- 所有 Common-DSL 方法共享完全相同的 Program Space。

- Effect identity 按 Operator family、parameter bucket、实际 targeting mode 与冻结 preflight panel 上的可观察执行效果判定；P0 不为此建设新的哈希、manifest 或冻结平台。

三个 Task 的 Primary operating budget 均固定为 B_main=4，包括 Anomaly Detection；Validation 额外报告 B∈{2,4,8}，其中 B=2 仅为曲线点。P0 描述性报告各 Task 当前 \|P_effect\| 与 B_main/\|P_effect\|，该比例不构成准入、通过或失败门。若当前 \|P_effect\|<B_main，所有 Common-DSL 方法最多评估全部可用 workflow，未使用预算如实记账；不得重复候选冒充预算消耗。

# 9. Baseline 与方法对照

## 9.1 End-to-End 主表

| **Method**               | **定义 / 归因**                                                            |
|--------------------------|----------------------------------------------------------------------------|
| Identity                 | 不处理；Raw baseline                                                       |
| Best Fixed Per-task      | Evolution 上选择一个固定 Pipeline，之后冻结                                |
| Fixed Heuristics         | Linear-impute / Hampel / Winsor / IQR 等固定规则                           |
| Parallel Best-of-N@B_main     | 相同 Consumer-evaluation 预算下独立候选搜索                            |
| Sequential Refinement@B_main  | 每轮读取上轮反馈修改 Program，但不修改 Harness                         |
| AegisTS-style Common-DSL | 高层选择 Family/Order，低层选择 Operator/Parameter；同 DSL/Consumer/Budget |
| Frozen H0                | 初始 Code Agent Harness，不做适应                                          |
| Ours A5                  | Scoped versioned Skill + Positive/Conflict/Negative write-back             |

Core cost roster 将 Fixed Heuristics 展开为四个独立方法：Linear-impute、Hampel、Winsor、IQR。与 Identity、Best Fixed、Parallel、Sequential、Frozen H0、Static、A3、K0、A5 合计 13 个方法；AegisTS-style 若判 STRUCTURALLY_INCOMPATIBLE、One-shot diagnostic 与 Tier-2 Extended 均不进入 Core cost matrix。

## 9.2 Harness / Memory 主表

| **Method**                             | **历史经验**                        | **在线修订**                               |
|----------------------------------------|-------------------------------------|--------------------------------------------|
| A3-reset                               | 无                                  | Target 内适应后清零                        |
| K0-fixed                               | 有，与 A5 同起点                    | 否                                         |
| \[Extended\] TimeClaw-style kNN Memory | Raw Episode / fingerprint retrieval | 否                                         |
| \[Extended\] Self-Harness-style        | Failure → bounded edit              | 是，但无本项目 TS-specific Scope evolution |
| A5-online                              | Scoped versioned Skill              | Positive / Conflict / Negative             |

## 9.3 Diagnostic Baseline

One-shot LLM：一次读取 Task/Consumer/Pattern 后直接输出 Workflow，不使用 Support Feedback，不修订；额外输出 predicted_gain∈{positive,neutral,negative}，用于计算方向一致率和“预测 positive、实际 material harm”的假阳性率。

Baseline 优先级：Tier-1（不阻塞主结果）= Identity、Best Fixed/Heuristics、Parallel Best-of-N、Sequential Refinement、Static/A3/K0/A5；AegisTS-style 在 P1 做限时 Adapter spike，通过则进入 Tier-1 主表，结构不兼容则记录 STRUCTURALLY_INCOMPATIBLE 并降为 related-work 附加说明，不阻塞其余 Core smoke。Tier-2 Extended = TimeClaw-style kNN Memory、Self-Harness-style bounded edit、Meta-Harness-style full code search；只有 Tier-1 主表完成且预算允许才运行，不得阻塞 Final。

# 10. 四个核心内部臂

| **Arm**   | **Target-local 适应** | **初始历史 Skill** | **单元间写回** | **归因**                   |
|-----------|-----------------------|--------------------|----------------|----------------------------|
| Static    | 否                    | 否                 | 否             | 无适应基线                 |
| A3-reset  | 是                    | 否                 | 否             | Target-local 冷启动价值    |
| K0-fixed  | 是                    | 与 A5 相同         | 否             | 固定历史经验价值           |
| A5-online | 是                    | 与 K0 相同         | 是             | 完整 Self-Evolving Harness |

- K0 与 A5 的初始 Store、Skill、Scope、Version、SHA 必须完全一致。

- A5 与 K0 的唯一系统性差异是是否允许写回和修订。

- A3 每个 Target 从公共 H0 开始。

- 所有适应臂使用相同 Target Feedback Budget。

- Historical Skill 只能供应待验证候选，不能直接执行；自主探索槽必须保留。

- 错误 Task Skill fail-closed。

# 11. Matched-Budget 协议

| **资源**                            | Primary B=4 Operating Point         |
|-------------------------------------|-------------------------------------|
| Full Support Consumer Evaluations   | 4                                   |
| Cheap Statistical / Verifier Probes | 12                                  |
| LLM Calls                           | 4                                   |
| LLM input+output Tokens             | 40,000                              |
| Accepted adaptive update            | 1 / Target（Ours = Skill revision） |
| Stochastic Arm Wall-clock           | 45 min / Episode                    |

Primary B=4 时，Support-A 最多 3 次完整 Consumer Evaluation，Support-B 最多 1 次独立 Promotion Evaluation；Query 统一额外评测一次，不计入 Adaptation Budget。所有 Slow 调用、Skill revision 验证和 Consumer Fit 都计费。

Budget Curve：Validation 报 B=2/4/8；Primary/Final operating point 固定 B=4。资源向量冻结为 B=2：(2 full eval, 6 cheap probe, 2 LLM calls, 20,000 tokens)；B=4：(4,12,4,40,000)；B=8：(8,24,6,60,000)。Final 不在看到 Validation 结果后切换 operating point。

Evolution、Validation 与 Final 均使用相同的三个预注册 Replica。成本以一次完整 task-native roster evaluation 作为 logical Consumer evaluation；Forecast/Classification 的 raw model fit 通常为 1，AD 的 raw model fit 等于当次 series 数，并在运行时单列。cache hit 的 raw fit 记 0，Query 每 method-cell 额外记一次 logical evaluation 且不计入 B。

# 12. 实验阶段与执行顺序

## 12.1 Phase 0：冻结与 Preflight

1\. Supersession Gate：生成旧协议取代台账，明确 Solar/Traffic/Yahoo/Epilepsy2 的新用途。

2\. Exposure Gate：逐 Dataset 记录是否下载、读数值、读标签/Outcome、参与过选参。

3\. Adapter Gate：只保留当前 Runtime 可低成本支持的 univariate 数据。

4\. Program-space Inventory（描述性，非 Gate）：按当前 DSL 枚举并去重各 Task 的 effect-distinct workflow，报告数量与 B_main/\|P_effect\|；不修改 Program Space，不影响 P0b 判词。

5\. Treatment Reachability + Empirical Event Gate：先确定性验证 Episode→Skill→Scope match→candidate supply→Support-A/B→revision→re-encounter 全链可达；凡要承担 A5−K0 / RQ3 的 Task，还必须在 development Consumer 上实测至少 1 个 POSITIVE、至少 1 个 CONFLICT/NEGATIVE、以及至少 1 个后续相似 Context re-encounter。只证明“代码能产卡”不算通过。

6\. Minimal Baseline Contract Smoke：在已曝光或 synthetic 微型 fixture 上验证 Identity、Fixed/Best-Fixed、Parallel、Sequential 与 Static/A3/K0/A5 的共同输入、结果结构、预算字段和 fail-closed 行为；不设效用/headroom/treatment 门，不读取 Natural Final Outcome。完整 Common-DSL Core smoke 属于 P1。

7\. Cost Accounting Freeze（描述性）：按实际 phase/cell 分项估算 Consumer fits/logical evaluations、cheap probes、LLM calls、tokens、accepted updates、Query extra evaluation 与 wall-clock。字段和计算不完整则 P0b 不完成；未预冻结总资源上限时不作 affordability PASS/FAIL。

## 12.2 Phase 1：Task-specific Evolution

每个通过 P0b 的 Task：默认 8 Evolution Episodes × 3 pre-registered replicas。A5 在本 Task 内写回；K0 冻结；A3 清零。若某 Task 的 Empirical Event Gate 不过，仍可运行 RQ1/RQ2 相关 end-to-end/accumulation 评测，但不得宣称该 Task 对在线 revision（RQ3）形成了有效 treatment。

## 12.3 Phase 2：Validation

- 每个 Task 运行 3 个 Validation Episodes。

- 不再修改 Global Harness，不换预算、阈值或 Dataset。

- 运行 B=2/4/8 Utility–Budget 曲线。

- 运行 Consumer-blind、No-Scope、No-Negative/Conflict Write-back、Support-only 等消融。

- 所有结构合法 Replica 全部进入 Final，禁止 best-seed / best-replica selection。

## 12.4 Phase 3：Final Natural Evaluation

- Global Transfer：直接使用 Task-specific Global Harness，不进行 Target-local 更新。

- Target-Time Adaptation：从相同 Global Harness 出发，Support-A 上做最多一次 local update，Support-B 验证后 Freeze，再打开 Query。

- Final Dataset 之间不共享 Target-local feedback；Final Outcome 不写回 Global Harness。

# 13. 指标与判分

| **Task**          | **Primary** | **Secondary**                 |
|-------------------|-------------|-------------------------------|
| Forecasting       | sMASE ↓     | Median / worst-series sMASE   |
| Classification    | Macro-F1 ↑  | Accuracy / worst-class recall |
| Anomaly Detection | Event-F1 ↑  | AUPRC / false-alarm / delay   |

统一效用方向定义为 U_forecasting=−sMASE、U_classification=Macro-F1、U_AD=Event-F1。每个 Dataset 报告 ΔU = U(method) − U(identity)，因此 ΔU>0 表示改善、ΔU<0 表示退化；原始表仍分别报告 sMASE↓、Macro-F1↑ 与 Event-F1↑，且不直接跨指标平均。

## 13.1 Confirmatory Contrasts

1\. Performance H1 — A5 − A3：历史经验积累相对冷启动是否提高 task-native downstream utility。

2\. Performance H2 — A5 − Parallel Best-of-N@B_main：完整经验系统是否超过等预算普通搜索。

Evolution Mechanism H3（独立判词，不与 H1/H2 同一 Holm family）— A5 − K0：只有真实 Gain/Harm 触发 revision、修订版被后续 Target 检索、改变 supply/probe/abstain/deployment，并带来效用或成本材料改善，才判 ONLINE_EVOLUTION_POSITIVE。

Performance family 只包含 H1/H2，并对这两条做 Holm 校正；H3 作为独立 Evolution-mechanism claim 单独判词，不连带惩罚 H1/H2。Supporting：A5 vs AegisTS-style、Target-adapted vs Global Transfer；TimeClaw-style/Self-Harness-style 仅在 Extended baseline 实际运行时报告。

## 13.2 Cross-task Summary

- Average Rank。

- Dataset-level Win/Tie/Loss。

- Task-equal macro relative gain。

- NRG / oracle-normalized regret 仅作为 Secondary；仅在 Program Space 和 oracle 定义一致、Headroom 足够时报告。

## 13.3 Safety

协议错误必须为 0：Query/Future leakage、Task-mismatch execution、Historical Skill 绕过双门直接部署、Wrong Promotion。效用安全报告 material harm rate、worst-series/worst-class degradation、correct abstention、cross-task supply leakage 和相对 comparator 的 harm 增量。

## 13.4 Cost

Consumer fits、LLM calls/tokens、probe、wall-clock、time-to-threshold、Adaptation AUC 均为 Secondary，不设为 Co-primary。

# 14. 统计分析

统计层级：Task → Dataset → Episode/Condition/Replica → Series；Dataset 是顶层独立单位。

- Hierarchical paired bootstrap：第一层必须重采样 Dataset。

- Dataset-level exact paired permutation / Wilcoxon 作为保守 Secondary。

- 报告 95% CI、Win/Tie/Loss、paired-difference CDF 和 per-dataset 完整结果。

- Condition、corruption seed、origin、series、LLM sampling seed 只能增加 Dataset 内精度，不能增加名义独立样本数。

- Final 的独立 Dataset 数量以 P0 fresh-pool audit 后冻结的 roster 为准；在小样本条件下不把 p\<0.05 作为唯一成功标准，并明确区分 task-level 结论与跨任务总体结论。

不设置“≥4/6”或“≥5/6 strictly-positive”之类粗粒度投票门。H1/H2 按 task-native paired effect、per-dataset 全表、hierarchical bootstrap CI 与方向一致性共同判断：若至少两个具 actionable headroom 的 Task 出现材料级正向平均效应，且其余 Task 不出现材料级平均负迁移（NO_HEADROOM Task 允许 SAFE_ABSTAIN），则可支持跨任务总体性能主张；否则按 Task 分别收账，不用一个合成投票门掩盖异质性。

ONLINE_EVOLUTION_POSITIVE 独立要求：真实 Skill revision 发生 → 修订版被后续 Target 检索 → 改变 supply/probe/abstain/deployment → A5 相对 K0 获得效用或成本材料改善。若前置事件资格门未形成则记 RQ3_NOT_EXERCISED；若链条形成但无材料改善则记 NO_ONLINE_EVOLUTION_ADVANTAGE。该结论不回溯修改 H1/H2。

Secondary Conditional-Benefit Analysis（Outcome-blind）：预注册分析 A5−A3 的收益是否随 missingness、Pattern complexity、effect-distinct candidate-space size、Support resolution 等部署前可见变量变化。连续变量优先；若画 low/medium/high 三档，阈值必须在 Final Outcome 打开前由 Evolution/Validation 冻结。该分析只解释“经验在什么条件下更有价值”，不改变 roster、预算或主判。

# 15. Validation-only 消融

| **Ablation**                    | **机制问题**                           |
|---------------------------------|----------------------------------------|
| Consumer-blind                  | Consumer conditioning 是否必要         |
| No-Scope                        | Pattern-conditioned retrieval 是否必要 |
| No Negative/Conflict Write-back | 失败经验是否驱动 Skill 修订            |
| Support-only                    | Support-B / delayed 独立验证是否必要   |
| K0-fixed                        | Online revision 的增量                 |
| A3-reset                        | Historical accumulation 的增量         |
| One-shot LLM                    | 真实 downstream feedback 的必要性      |

# 16. 论文结果组织

| **表/图** | **内容**                                                                                                   |
|-----------|------------------------------------------------------------------------------------------------------------|
| Table 1   | Dataset、Task、Consumer、Evolution/Validation/Final、Natural/Controlled、序列数与长度                      |
| Table 2   | Natural End-to-End：Identity / Fixed / Best-of-N / Sequential / AegisTS-style / Frozen H0 / A3 / K0 / A5   |
| Table 3   | Harness/Memory：A3 / K0 / TimeClaw-style / Self-Harness-style / A5                                         |
| Table 4   | Global Transfer vs Target-Time Adaptation（Natural Final roster；数量以 P0b fresh-pool audit 冻结结果为准） |
| Table 5   | Controlled Witness + Ablation                                                                              |
| Figure 1  | Utility–Budget：B=2/4/8                                                                                    |
| Figure 2  | Evolution stream cumulative utility/regret                                                                 |
| Figure 3  | Skill v0→v1→v2、Scope width 与 write-back 轨迹                                                             |
| Figure 4  | Task × Dataset Win-rate heatmap                                                                            |
| Figure 5  | Fixed Workflow × Task 的 ΔPerf 符号热图                                                                    |

# 17. 实际发车顺序

| **阶段** | **动作**                                                                                            | **通过后** |
|----------|-----------------------------------------------------------------------------------------------------|------------|
| P0b      | 旧协议取代 + Exposure/Fresh-pool + Adapter + Minimal Baseline Contract Smoke + Program-space 描述性 inventory + Treatment audit + Cost accounting | P1         |
| P1       | Common DSL contract check + 完整 Core Baseline Smoke                                                | P2         |
| P2       | Forecast 单流试跑，仅验证统一 Runner 与 Treatment 非空                                              | P3         |
| P3       | Classification / AD 的完整数据 roster 与纵向实验切片接入同一协议                                   | P4         |
| P4       | 三任务 Evolution × 3 Replica                                                                        | P5         |
| P5       | Validation + Budget Curve + Ablation                                                                | P6         |
| P6       | Freeze 方法、Store、Skill、Seed、预算                                                               | P7         |
| P7       | 一次性打开 P0 冻结的 Natural Final roster                                                           | P8         |
| P8       | 统计、主表与 Result-to-Claim 审计                                                                   | 论文整理   |

# 18. 止损与禁止事项

- 若 Treatment Reachability 不成立，停止，不进入大规模 Evolution；只修同一协议的接线问题。

- 若 Baseline Adapter 不支持某 Dataset，只按预冻结结构失败规则替换；不得按科学结果替换。

- Final Outcome 打开后禁止新增数据、Baseline、Program、Consumer、消融或阈值。

- 不允许为 A3/A5 的结果好坏临时增加 LLM/fit/probe 预算。

- 不把 Controlled Witness 的正结果写成 Natural Data Readiness 能力证据。

- 不把 Skill 被检索/供应等同于 Skill 产生收益；必须经过 Support-B 与 Query。

- 不以一次 sampling replicate 宣称复现。

- 若某 Task 的 Empirical Event Gate 未形成 POSITIVE + CONFLICT/NEGATIVE + re-encounter，则 RQ3 记 NOT_EXERCISED；禁止通过事后增加 episode、放宽 Scope 或降证据门来“制造” treatment。

- 若 Natural Final 被冻结 oracle 诊断为 NO_HEADROOM，则正确 abstention/identity 不计作能力失败；同时 Controlled Witness 的正结果仍不得替代 Natural capability。

# 19. 最终冻结主张

本协议检验以下条件式主张：若 H1（A5\>A3）与 H2（A5\>等预算 Best-of-N）通过预注册性能门，则支持“历史 Gain/Harm 的结构化积累能提高未见 Target 的下游 Data Readiness”；只有当 H3 的完整因果链同时成立时，才进一步支持“在线 Skill revision 本身带来额外增量”。所有结论必须以 Natural Final 的 task-native 指标为准，Controlled Witness 只提供机制证据。

# 20. 主要外部实验设计参考

- AegisTS: A Hierarchical Agent System with Reinforcement Learning for Multivariate Time Series Data Cleaning. arXiv:2605.04902.

- TimeClaw: Harnessing Generalist Agents for Contextualized Time Series. arXiv:2606.05404.

- Self-Harness: Harnesses That Improve Themselves. arXiv:2606.09498.

- Meta-Harness: End-to-End Optimization of Model Harnesses. arXiv:2603.28052.

- Rethinking the Evaluation of Harness Evolution for Agents. arXiv:2607.12227.

# 21. v1.1 → v1.2 冻结前修订记录

- DOCX 指定为唯一内容真源；Markdown 由 DOCX 自动完整导出。

- Epilepsy2 因已开封移出 Classification Final；Fresh UCR/NAB 改为 P0 pool audit 后冻结的 TBD roster。

- Primary operating budget 从 B=8 调整为 B=4，Validation 保留 B=2/4/8 曲线。

- v1.2 曾收紧 Program-space 候选数量门；该硬门已由下方 v1.2.1 修订撤销。

- A5−K0 从 performance confirmatory family 拆出，作为独立 online-evolution mechanism claim。

- Treatment Reachability 增加真实事件门：POSITIVE + CONFLICT/NEGATIVE + re-encounter。

- Final 增加 NO_HEADROOM/SAFE_ABSTAIN 预注册分支，删除“≥4/6 positive”粗门。

- accepted adaptive update 对 Ours 精确为 Skill revision；Classification 正式 metric = Macro-F1 primary / Accuracy secondary。

- 外部 Baseline 划分 Tier-1 Core 与 Tier-2 Extended，Extended 不阻塞主结果。

- 补回 Outcome-blind conditional-benefit secondary analysis，解释经验收益何时更大。

# 22. v1.2 → v1.2.1 P0b 修订记录

- 删除 6.25% / \|P_effect\|≥64 的 Program-space 硬门；三个 Task 的 B_main 均保持 4，只描述性报告实际覆盖率。

- 明确禁止为候选数量新增 Operator、二步模板、targeting 能力、候选 manifest、哈希链或冻结平台。

- P0b 只执行已曝光/synthetic fixture 上的 Minimal Baseline Contract Smoke；P1 执行完整 Common-DSL Core Baseline Smoke，消除 §12 与 §17 的阶段冲突。

- Parallel 与 Sequential 的命名统一为 @B_main；Primary/Final 显示 @4，Validation 分别显示 @2/@4/@8。

- Cost Gate 改为 Cost Accounting Freeze：要求字段与计算完整，但未冻结总资源上限时不判 affordability。

- Utility 统一为 U_forecasting=−sMASE、U_classification=Macro-F1、U_AD=Event-F1，确保所有 ΔU>0 都表示改善。

- Classification Final-A/B 冻结为 Adiac/ArrowHead 及 TRAIN 内确定性切分；严格暴露口径下 NAB Final-2 冻结为 FINAL_POOL_UNAVAILABLE。
