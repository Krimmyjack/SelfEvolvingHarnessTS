# R4A-b · 逐序列增益的持久性与方差分解（2026-09-07）

证据类别：**MECHANISM / NEGATIVE**（development 机制读数，不是能力证据，不做显著性主张）。
判词词表在任何数字计算之前冻结（见工件 `verdict rules`），本报告未调整。

- 脚本：`evaluation/main_protocol_p4/audit_r4a_persistence_and_variance.py`
  （`python -m evaluation.main_protocol_p4.audit_r4a_persistence_and_variance` 可复跑，0 fit）
- 工件：`artifacts/main_protocol/r4a_persistence_and_variance.json` / `.md`
- 前置：`docs/R4A_PATTERN_IDENTIFIABILITY_RESULT_2026-09-07.md`（R4A）
- 执行说明：脚本与两个工件由执行线写出并落盘；执行线在写本报告前被中止，本报告由主线
  按工件数字补写，所有数字与 `.md` 工件逐位相同。

## 0. 要回答的问题与一句话答案

R4A 说明：伤害既不能由 origin 前可见模式解释，也不能由序列自己的未来窗口解释。
那么**逐序列增益到底是不是一个稳定属性**？

> **不是。** 同一序列在 Support 面（origin）与 delayed 面（origin+48）的增益按秩几乎不相关
> （Spearman 中位数 0.034586 / −0.045113 / 0.142857）；方差分解里残差占 63–73%，
> 序列身份只解释 20–23%（与可交换噪声下的期望 18% 几乎相同），单元只解释 6–14%。
> 10/10 格判词 `MOSTLY_NOISE`。

## 1. 边界自检

| 项 | 值 |
| --- | --- |
| Consumer fits | 0 |
| LLM 调用 | 0 |
| held-out 读数 | 0（禁区 `[80:120]` × {4056, 4296, 4536, 4776, 5016}） |
| 读到的最大时间下标 | 3863（只读 origin 之前的 192 步 context；`horizon_reads = 0`，未用任何 ORACLE 量） |
| 原始序列读取次数 | 840 |
| 写入文件 | 两个工件；既有文件编辑 0；预测库未写；新增 SHA/Hash 0 |

数据与 loader 同 R4A：`preflight_natural_gap_variant.load_variant`，NaN=503712。
行构造同 R4A：4400 行，剔除 `W3_outlier_mad_then_pmc` 别名（42/42 条与 ANCESTOR 逐值全等，
max |diff| = 0）后 3560 行；80 个不同 uid；剔除 40 个 degenerate-uid 行、8 条 verifier 未过的
CAND 条目、33 条范围外条目。

## 2. 每 (program, face) 判词

| cell | 行数 | 跨面 Spearman 中位数 | 位置间份额 | uid 主效应份额（加权） | uid 噪声期望 | 残差份额 | 判词 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ANCESTOR / support | 420 | 0.034586 | 0.102973 | 0.201908 | 0.183457 | 0.705300 | `MOSTLY_NOISE` |
| ANCESTOR / delayed | 420 | 0.034586 | 0.105068 | 0.215059 | 0.183457 | 0.682188 | `MOSTLY_NOISE` |
| W1_hampel_filter / support | 420 | −0.045113 | 0.060364 | 0.195976 | 0.183457 | 0.733260 | `MOSTLY_NOISE` |
| W1_hampel_filter / delayed | 420 | −0.045113 | 0.144622 | 0.226196 | 0.183457 | 0.630583 | `MOSTLY_NOISE` |
| W2_pmc_then_outlier_mad / support | 420 | 0.142857 | 0.104049 | 0.200146 | 0.183457 | 0.710058 | `MOSTLY_NOISE` |
| W2_pmc_then_outlier_mad / delayed | 420 | 0.142857 | 0.131501 | 0.212886 | 0.183457 | 0.710000 | `MOSTLY_NOISE` |
| CAND_outlier_iqr>hampel_filter / support（参考） | 240 | 0.027820 | 0.084711 | 0.152354 | 0.242058 | 0.776320 | `MOSTLY_NOISE` |
| CAND_outlier_iqr>hampel_filter / delayed（参考） | 280 | 0.027820 | 0.143670 | 0.202260 | 0.206841 | 0.629343 | `MOSTLY_NOISE` |
| CAND_outlier_iqr>winsorize / support（参考） | 240 | 0.032331 | 0.168711 | 0.202186 | 0.242058 | 0.628635 | `MOSTLY_NOISE` |
| CAND_outlier_iqr>winsorize / delayed（参考） | 280 | 0.032331 | 0.091939 | 0.264730 | 0.206841 | 0.649080 | `MOSTLY_NOISE` |

三个判据没有一格接近另两条判词的门槛：uid 主效应份额最高 0.264730（< 0.30），位置间份额最高
0.168711（< 0.50），跨面 Spearman 中位数最高 0.142857（< 0.30）。

## 3. 五项读数

**3.1 跨面持久性。** 按位置各算一次 Spearman 再取分位：ANCESTOR q25/中位/q75 =
−0.103759 / 0.034586 / 0.133835；W1 = −0.294737 / −0.045113 / 0.172932；
W2 = −0.139850 / 0.142857 / 0.260150。pooled Spearman 0.065363 / 0.017177 / 0.108911；
pooled Pearson 0.363516 / 0.281393 / 0.286426（Pearson 高于 Spearman，说明仅有少数大幅值
序列在两面同时突出，秩序整体不保留）。`severe_harm` 的跨面一致性：Support 严重受损者在
delayed 仍严重受损的比例 = 0.107143 / 0.325203 / 0.283333，对应 delayed 基率
0.073810 / 0.235714 / 0.178571——只略高于基率。`helped` 的跨面 Cohen's kappa =
0.001664 / 0.072858 / 0.062870。

**3.2 方差分解。** 主表见 §2；按 cohort 拆开后，两个大 cohort（[0:40] 7 个位置、[40:80] 9 个
位置）的 uid 份额与噪声期望相当或更低（例如 ANCESTOR/support：0.126387 对 0.136691；
0.076123 对 0.106145），两个小 cohort（[80:120] 2 个位置、[120:160] 3 个位置）的 uid 份额虽高
（0.44–0.61）但噪声期望同样高（0.32–0.49），因为每个 uid 只有 2–3 个重复。位置间份额在
所有 cohort 都低于 0.23。

**3.3 z_peak 是作用强度还是作用方向。** `local_robust_z_peak` 与 |g| 的 Spearman 在
−0.230827 到 −0.013428 之间，与 g 的 Spearman 在 −0.008266 到 0.148804 之间，两者都很弱；
只有 W1/support 与 W2/support 两格出现 |ρ(|g|)| 明显大于 |ρ(g)|。对 severe_harm 的原始
pooled AUC 全部 < 0.5（0.260536–0.448430），方向一致：**z_peak 越高、严重受损越少**。
结论：z_peak 既不是稳定的方向信号，也算不上稳定的强度信号。

**3.4 单元级条件（in-sample 上界，不可部署）。** 把每个 (position, face) 的单元均值增益当作
cohort 层条件，它对逐序列 helped 的 AUC 为 0.600446–0.695842，对 severe_harm 为
0.591218–0.727869，与最佳可见序列级特征的 pooled AUC（helped 0.517443–0.599289，
severe_harm 0.560460–0.739464）处于同一量级。**即便把答案（单元均值）握在手里，
也预测不好该单元内哪条序列受益或受害。**

**3.5 一句话。** 逐序列增益在 48 步之外不可复现；条件既不在序列层也不在 cohort 层，
在本仪器（pooled Ridge、sMASE、horizon 48、KDD 含缺失变体）的分辨率下主要是噪声。

## 4. 正典 §10 五问

**Harness 行为改变了什么。** 没有。0 fit、0 LLM、不编辑任何现有文件；改变的是对"哪个粒度上
有可学信号"的判断依据。

**数据上观察到了什么。** (i) 同一序列相邻两个 origin 的处理效果秩相关约为 0；(ii) 序列身份
对增益方差的解释力与纯噪声期望相当；(iii) 单元均值作为 in-sample 上界也只到 0.60–0.73；
(iv) R4A 里 32–45% 的 Support→delayed 翻号率与这里的近零持久性是同一件事的两种读法。

**当前最大方法不确定性。** 这里的"噪声"是指序列身份与单元主效应都解释不了的方差，它也可能
是随时间快速变化（48 步内）的序列层条件；对部署决策而言两者等价，但对机制解释不等价。
本包无法区分，也无法判断该噪声来自 Consumer（pooled Ridge 的共享模型变化叠加逐序列 context
变化）还是评价侧（48 步 sMASE 本身的方差）。

**是否仍与目标一致。** 一致，但方向必须改写：项目前提"条件可从部署时可见的序列级模式读出"
在本数据 × Consumer × horizon 上不成立，且原因不是特征或表示，而是**逐序列目标本身没有
持久信号**。这一个数同时解释了 R4A、W54、W61、P4d、D1、HEC-1 transfer 全部负结果。
可学、可迁移的对象仍存在，但在**程序层**（R2：程序间排序跨单元大体稳定）与**单元级
Support 准入**这一粒度。

**下一项最小纵向切片。** 两条，先后有序：
(A) **风险线虚警率**（0 fit，只读同一预测库）：用置换估计一个真正中性的程序仅因逐序列噪声
撞上 `harmed_fraction ≤ 0.20` / `max_single_series_harm ≤ 0.30` 的概率；若虚警率高，权威门
应改为在更多序列或更多窗口上聚合后裁决，这是对系统设计冲击最大的一条。
(B) **持久性是否 Consumer 特有**（需授权，≈3000 fits，即 M5 Stage A / HEC-2 ①）：per-channel
Ridge 下逐序列效应是否变得持久。若是，Scope 一层在正确的 Consumer 下重新成立；若否，
噪声在评价侧，该改的是 horizon 与窗口数。

## 5. 这份结果不是什么

- **不是能力或泛化证据。** 全部来自 development 变体，held-out 对一个都没读。
- **不是显著性结论。** 80 个 uid 在 21 个位置上重复出现，折数与 cohort 数都很小；四个 cohort
  里有两个只有 2–3 个位置，其 uid 份额读数不可靠（噪声期望本身就高）。
- **不是"逐序列条件化在物理上不存在"的结论。** 只说明在本仪器的分辨率下测不到；换 Consumer、
  拉长 horizon 或增加窗口数后可能改变。
- **不是对程序层条件化的否定。** 程序间均值差异（ANCESTOR +0.245427 对 W1 −0.031355）与
  R2 的跨单元排序稳定性都成立；本包否定的只是序列层。
- **不改变任何已冻结的协议、roster、split、阈值或预算**，也没有新增 SHA、目录或平台层。

## 6. 追加更正（2026-09-07 夜，主线；追加式，不覆盖上文）

采认 `idea-stage/observation_mechanism/notes.md`（astra）§3 的三点收窄，上文以下表述撤回或收窄：

1. §4"这里的噪声……对部署决策而言两者等价"**撤回**。低重测相关只否定实体记忆策略
   （"该序列上次受益 → 下次继续用"，与 DEV-KNOW-1 的 `TRIED_BUT_NEVER_SELECTED` 和本包
   helped kappa ≈ 0 一致），**不否定**按 origin 时刻可见状态做决策：状态 X_t 独立同分布、
   g_t = X_t 时，相邻相关与 uid 主效应皆为 0 而决策可完美。`MOSTLY_NOISE` 应读作
   "uid/单元主效应与 R4A 已测的约 22 个特征都解释不了的方差"，不是不可约噪声。
2. §3.5"条件既不在序列层也不在 cohort 层"收窄为："在已测的序列级描述量与主效应分解下未识别到"。
   astra 补算显示 delayed 面 269/420 个实例服务窗口未被 MAD 修改、却承载 31 个严重受损中的 24 个，
   即多数实例的效果来自训练侧共享模型变化——本包与 R4A 的序列级特征均描述服务窗口，对这些实例
   观察对象本身不匹配。
3. §4"下一项最小纵向切片 (A) 风险线虚警率置换"**撤回**：Ridge 为确定性求解，"重测方差"提法不成立；
   对观测增益做置换也不能识别"真正中性程序"的虚警率；更改风险门的保护目标属协议决定。
   保留 (B) per-channel Consumer 对照，并建议在同一有界运行中采集三格作用分解
   （raw 模型/raw context、program 模型/raw context、program 模型/program context）。

本包数字不变；判词名称按冻结词表保留，解释按本节。
