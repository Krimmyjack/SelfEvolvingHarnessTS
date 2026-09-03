# D1 任务书:路由伤害 0-LLM 诊断(HEC 路线图第 1 步)

日期:2026-09-03。地位:sol 已批准协议(2026-09-03),主线起草;执行方按本书实现**一个** audit 脚本、
产出**一份**工件,不新增 Gate/SHA/Manifest。预算:**0 LLM;Ridge fit 硬上限 ≤400**(账本超帽即停,
如实报未完成部分)。结论**只决定 HEC-2 是否冻结新 Risk 面**,不得回流修改 HEC-1 冻结面或任何已冻结合同。

## 0. 问题

Source-v3 三次独立重遇全败,尾部主要来自谓词新解析进来的序列(§10.11);sol 指出此前已有序列在
serving context **修改点为 0 时仍受严重伤害**——伤害来自"被路由到 program model",不是 context 被改。
本诊断回答两问:

1. 在部署期**不读 Outcome** 可算的量里,哪一个(如果有)能把 Scope 内的受害序列与非受害序列分开?
2. 同一数据、Program、Scope、指标下,把 Consumer 从 pooled Ridge 换成 per-channel Ridge,路由伤害是否消失?

## 1. 材料(全部已曝光 development,不新开任何窗口)

- 工件:`artifacts/main_protocol/p4w3b_source_line_v3_clean_post_fix_replicate_1.json`(Source-v3 干净跑)。
- 数据身份:`EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`;cohort `source readable[160:200]`,
  served = support_a 面 20 条(字典序:T260…T37),训练 roster 照 Source-v3 原样。
- **主窗口集(7 个)**:4 张 Draft 的 delayed 窗口(read_origin 1944 / 2424 / 2664 / 2904)+ 3 个独立重遇
  窗口(2136 / 2616 / 2856)。每窗口的 Program、`serving_scope` 谓词、解析集(Scope 内序列 S)与逐序列
  realized gain 均在工件 `delayed_gate` / `re_encounter_gate` 中(`per_series_gain` 为 served 字典序位置
  向量;非零位 = S,已与执行方治疗集逐条核符)。
- **次窗口集(可选,预算允许时)**:4 个 Support-A 探针窗口(origin 1896 / 2376 / 2616 / 2856,根 Scope
  `z_peak>=3`,`risk_refusals[].per_series_gain`)。先完成主窗口集,再决定是否做次集。
- 禁止:读取 `[80:120]` 任何 origin、任何 held-out origin、UCR TEST、Natural Final;禁止改任何阈值。

## 2. 定义

- **Scope 内序列 S_w**:窗口 w 上谓词解析集(= `per_series_gain` 非零位;若某位为 0 但谓词解析为内,以
  谓词重解析为准并记差异)。
- **标签**:`harmed_material` = gain < −0.005;`harmed_severe` = gain < −0.30(合同单条线,只作标签,不改动)。
- **归因**(§10.3):S_w 中每条标 `NEW_ENTRANT`(不在该 Draft 上一验证窗口的 S 中)或 `CONTINUING`。
- **证据区 E_w**:该 Draft 在窗口 w **之前**的验证窗口上、Scope 内且 gain ≥ −0.005 的序列,取其在**当时窗口**
  的部署可见特征向量(22 维,冻结分箱后的 bin 索引)。delayed 窗口的 E = Support-A 探针中修订后 Scope 成员的
  非受害者;重遇窗口的 E = 上者 ∪ delayed 非受害成员。

## 3. 四个信号(逐序列、逐窗口;全部 outcome-free、部署可见)

| 信号 | 定义 | 成本 |
| --- | --- | --- |
| **S1 预测分歧** `div` | `mean_h |program_prediction − raw_prediction| / metric_scale`,horizon 48;两份预测由 `scoped_serving_evaluator.scoped_evaluate` 在同一次调用内产出(`raw_prediction`、`program_prediction`),`metric_scale` = 该序列 `seasonal_scale(raw[:origin])` | 2 fits / 窗口(评估器现成) |
| **S2 证据区距离** `dist` | 该序列在 w 的分箱特征向量到 E_w 中最近向量的归一化 L1(bin 索引差之和 / 特征数);E_w 为空 → 记 `NaN` 并单列 | 0 fits |
| **S3 行为超覆盖** `beh_oor` | Program 作用于该序列 serving context 的行为向量(修改点比例 `moved/CONTEXT_LENGTH`、改动幅度 `sum|Δ|/(moved·metric_scale)`、最后 48 点内修改数)是否任一维落在 E_w 成员行为范围 [min,max] 之外;并输出连续量 `beh_dist`(超出量/范围宽) | 0 fits(仅 `_prepare`) |
| **S4 改动量(对照)** `mod` | 修改点比例与幅度本身(主线原 (b),按 sol 判为不足,作对照) | 0 fits |

分箱:沿用 Scope 归纳的冻结分箱(`observable_numeric_bin` / `s1._binned_contract_leaves` 同款);不新造分箱。

## 4. 反事实:pooled vs per-channel

- **相同**:窗口、served 序列、Program 步骤与参数、`serving_scope` 解析集、metric(sMASE,同 `metric_scale`)、
  Ridge 超参数、CONTEXT_LENGTH / HORIZON、anchor 列表。
- **不同**:仅 Consumer 结构。pooled = 现行(训练 roster 全部窗口拟合一个模型);per-channel = 每条 served 序列
  用**自身**历史上的 anchored 训练窗口各拟合一个模型(raw 与 program 各一),其余一律不变。
- 读数(逐窗口):per-channel 下 S_w 的逐序列 gain、`harmed_material` 数、`harmed_severe` 数、hf、msh、聚合;
  与 pooled 并列;按 `NEW_ENTRANT` / `CONTINUING` 分层。
- 成本:per-channel 每条 Scope 内序列 2 fits;主窗口集 |S| 合计 46 → ≈92 fits;加 pooled 14 → ≈106。
  次窗口集若做,≈4×(2+2×15)=128。合计 ≤ 240,留 160 余量;**超 400 即停**。

## 5. 预注册读数与判词词表

**分离度**(主窗口集合并,S 内序列 n≈46,`harmed_material` n≈10):每个信号对 `harmed_material` 的
AUC(秩统计)+ 按序列 bootstrap 的 95% CI;对 `harmed_severe`(n≈4)只报点估计与逐条列表。
预注册阈值:**AUC ≥ 0.75 且 CI 下界 > 0.5** 记 `SEPARATES`,否则 `DOES_NOT_SEPARATE`。

**路由伤害核验**:`harmed_severe` 序列中 `moved == 0`(或 ≤1 点)的比例;≥ 1/2 → `ROUTING_HARM_CONFIRMED`,
否则 `ROUTING_HARM_NOT_DOMINANT`。

**反事实**:主窗口集 3 个重遇窗口中,per-channel 相对 pooled 使 `NEW_ENTRANT` 受害数减半且 msh 不升的窗口数
≥ 2 → `PER_CHANNEL_LOWER_TAIL`;反向 → `PER_CHANNEL_HIGHER_TAIL`;其余 → `NO_CLEAR_DIFFERENCE`。

**总判词**(供 HEC-2 决策,不进 HEC-1):
- `RISK_FACE_CANDIDATE_IDENTIFIED`:至少一个 outcome-free 信号 `SEPARATES`;列出该信号与其单阈值命中表。
- `NO_OUTCOME_FREE_SEPARATOR`:无信号 `SEPARATES`;新进入者风险只能由 held-in 探针与证据有界 Scope 候选形式承担。
- `INCONCLUSIVE_SAMPLE`:受害 n<6 或 fit 超帽未完成主窗口集。

**不得**:据结果改 0.20/0.30/0.005、改 Scope 类、改 HEC-1 冻结面;不得把 AUC 写成"部署期可预测伤害"——
它只是已曝光窗口上的分离度。

## 6. 输出

`artifacts/main_protocol/p4ab_routing_harm_diagnostic.json` + `.md`(一份),字段:
`stage`、`data_version`、`sources`(工件路径)、`windows[]`(origin / read_origin / program / predicate /
S / E 大小)、`series_rows[]`(uid, window, attribution, gain, harmed_material, harmed_severe, div, dist,
beh_oor, beh_dist, mod_fraction, mod_magnitude, moved, gain_per_channel)、`auc{}`(信号 → 点估计、CI、
判词)、`routing_harm_check{}`、`counterfactual{}`(逐窗口 pooled vs per-channel 表 + 判词)、`verdict`、
`boundary{llm_calls:0, consumer_fits:≤400 实测, held_out_reads:0, thresholds_changed:0, operators_added:0,
artifacts_overwritten:0}`。

## 7. 执行方回报格式

(1) 逐窗口 S/E/受害表;(2) 四信号 AUC 表与 `harmed_severe` 逐条(含 `moved`);(3) pooled vs per-channel
逐窗口对照;(4) 判词三项;(5) 实际 fits;(6) 实现中任何偏离本书之处及理由;(7) 发现的规格矛盾。
