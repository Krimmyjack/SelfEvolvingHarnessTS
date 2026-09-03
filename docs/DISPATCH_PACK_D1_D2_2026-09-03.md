# 分发包:D1 + D2(2026-09-03,主线发,用户转递)

主线当前会话无子代理分发能力,改由用户转递。两书**相互独立、可并行**,均 0 LLM。每块用
`======== 分发 X 开始 / 整段复制 ========` 包围,**整段复制**给接收 agent 即可,不必带另一块。
验收人 = 主线(Fable);方法争议呈 sol。账本已记「书已发」。

接收模型档(两块相同):**grok 4.6-xhigh**(一般难度;**不必上 opus**)。

---

======== 分发 A 开始 / 整段复制 ========

# 任务:D1 路由伤害 0-LLM 诊断

你是本任务的**唯一执行方**。只做这一本书。不要做 D2,不要改 HEC-1 冻结面,不要 git 提交。

**接收模型档**:grok 4.6-xhigh(不必上 opus)。
**工作目录**:`c:\Users\辉\Desktop\Agent\SelfEvolvingHarnessTS-deepseek-guidance-evolution`
(Windows,PowerShell)。Python 可用 `conda activate project` 或直接 `python`。这是可用环境,不是必须改当前
Shell 的要求。

## 必读顺序(按此顺序,读完再写代码)

1. 项目 `AGENTS.md` **§1、§6、§7、§8**(方向、单假设 / first-fault、反过度工程、数据与证据纪律;
   本任务必须携带这些约束;与任务书冲突时以任务书 + 用户当前指令为准,但不得违反 §7 的 SHA/平台禁令)。
2. `docs/D1_ROUTING_HARM_DIAGNOSTIC_2026-09-03.md`(**唯一规范**;字段、窗口、信号、判词词表、回报格式
   以它为准)。
3. `docs/HEC_EVOLUTION_MAINLINE_PLAN_2026-09-02.md` §10.11(背景:Source-v3 归因与 sol 的路由伤害纠正;
   改动量不是风险载体,模型路由才是)以及 §12 第 1 步(D1 结论只进 HEC-2)。
4. 相关脚本(只读,不改):
   - `evaluation/main_protocol_p4/scoped_serving_evaluator.py`(`scoped_evaluate` 同一次调用内部产出
     `raw_prediction` / `program_prediction`,各 1 fit,合计 2 fits / 窗口);
   - `evaluation/main_protocol_p4/scope_spec.py`;
   - `evaluation/main_protocol_p4/restricted_draft.py`;
   - `evaluation/main_protocol_p4/run_source_line_v3.py`(窗口几何、roster、Program 编译方式)。
5. 工件(只读):`artifacts/main_protocol/p4w3b_source_line_v3_clean_post_fix_replicate_1.json`。

## 要做什么(一句话)

在已曝光 development 上、0 LLM、Ridge fit ≤400,回答两问:(1) 部署期不读 Outcome 可算的四个信号里,
哪一个(如果有)能把 Scope 内受害序列与非受害序列分开;(2) 同一数据 / Program / Scope / 指标下,把
Consumer 从 pooled Ridge 换成 per-channel Ridge,路由伤害是否消失。产出**一个** audit 脚本 + **一份**
工件。结论只供 HEC-2 决策,不进 HEC-1。

## 硬约束

- **0 LLM** 网络调用。`boundary.llm_calls = 0`。
- Ridge fit **硬上限 ≤400**。自建计数器,与评估器返回的 `consumer_fits` 对齐;超帽即停,如实报未完成
  部分,总判词走 `INCONCLUSIVE_SAMPLE`(若主窗口集未完成)。
- 只新建**一个** audit 脚本(建议 `evaluation/main_protocol_p4/audit_routing_harm.py`)与**一份**工件
  `artifacts/main_protocol/p4ab_routing_harm_diagnostic.{json,md}`。
- **不修改任何现有文件**(含 `scoped_serving_evaluator.py`)。不 git 提交。不新增 SHA / manifest /
  Gate / 平台层。不改任何阈值(0.20 / 0.30 / 0.005 只作标签)。不新增算子。
- 不读 `[80:120]` 任何 origin、任何 held-out origin、UCR TEST、Natural Final。
- 材料全部已曝光:`EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`;cohort
  `source readable[160:200]`;served = support_a 面 20 条。

## 关键省钱点(必须遵守;预算花在反事实上)

**S1 预测分歧零额外 fit。** `scoped_evaluate` 在同一次调用内已经拟合 raw 与 program 两个 Ridge
(`fits += 1` 各一次),局部变量即 `raw_prediction` / `program_prediction`。S1 =

```text
mean_h |program_prediction − raw_prediction| / metric_scale
```

horizon 48;`metric_scale` = 该序列 `seasonal_scale(raw[:origin])`。主窗口 7 个 × 2 = **14 pooled
fits**,不要为 S1 再 fit。

**规格缝(执行时必遇,已预登记,按此处置、写入回报第 6/7 项):** 当前
`scoped_evaluate` 的返回 dict **不含**这两份预测数组(只含 smase / `consumer_fits` 等)。**禁止**为了
补字段去改 `scoped_serving_evaluator.py`。允许且只允许在 audit 脚本内捕获同一次调用的预测,例如:
对 `_serve` 做局部包装、或在 audit 内复制 `scoped_evaluate` 的 2-fit 路径并**额外返回**预测数组。
无论哪种,该窗口的 pooled 路径只走一遍、只计 2 fits。S2 / S3 / S4 均为 0 fit(S3 仅 `_prepare`)。

**预算全花在 per-channel 反事实上。** 管线**无**现成 per-channel serving 实现。只许在 audit 脚本内
最小函数实现:**同** Ridge 超参、同 `CONTEXT_LENGTH` / `HORIZON`、同 anchor 列表;唯一不同 = 每条
Scope 内序列用**自身**历史上的 anchored 训练窗口各拟合 raw / program 各一。不得改
`scoped_serving_evaluator.py`。成本:每条 Scope 内序列 2 fits;主窗口集 |S| 合计约 46 → ≈92;加
pooled 14 → **≈106**。次窗口集(可选,先完成主集再决定)≈128。合计 ≤240,留约 160 余量;**超 400 即停**。

## 窗口与信号(摘要;细节以任务书为准)

- **主窗口集(必须先完成)**:4 张 Draft 的 delayed(`read_origin` 1944 / 2424 / 2664 / 2904)+ 3 个独立
  重遇(2136 / 2616 / 2856)。Program、`serving_scope`、解析集 S、逐序列 `per_series_gain` 均在工件
  `delayed_gate` / `re_encounter_gate`。非零位 = S;若某位为 0 但谓词解析为内,以谓词重解析为准并记差异。
- **次窗口集(可选)**:4 个 Support-A 探针(origin 1896 / 2376 / 2616 / 2856),根 Scope `z_peak>=3`。
- 标签:`harmed_material` = gain < −0.005;`harmed_severe` = gain < −0.30(只作标签)。
- 归因:`NEW_ENTRANT` / `CONTINUING`(相对该 Draft 上一验证窗口的 S)。
- 四信号:S1 预测分歧 / S2 证据区归一化 L1 / S3 行为超覆盖 / S4 改动量对照。分箱沿用冻结
  `observable_numeric_bin`,不新造分箱。
- 反事实:相同窗口、served、Program、解析集、sMASE、Ridge、CONTEXT / HORIZON、anchors;只换
  Consumer 结构。读数按 `NEW_ENTRANT` / `CONTINUING` 分层,与 pooled 逐窗口并列。

## 预注册判词(只许从词表选;不得外推)

- 分离度(主窗口集合并):AUC ≥ 0.75 **且** 按序列 bootstrap 95% CI 下界 > 0.5 → `SEPARATES`,否则
  `DOES_NOT_SEPARATE`。`harmed_severe` 只报点估计 + 逐条列表(含 `moved`)。
- 路由伤害核验:`harmed_severe` 中 `moved == 0`(或 ≤1 点)比例 ≥ 1/2 → `ROUTING_HARM_CONFIRMED`,
  否则 `ROUTING_HARM_NOT_DOMINANT`。
- 反事实:3 个重遇窗口中,per-channel 相对 pooled 使 `NEW_ENTRANT` 受害数减半且 msh 不升的窗口数
  ≥ 2 → `PER_CHANNEL_LOWER_TAIL`;反向 → `PER_CHANNEL_HIGHER_TAIL`;其余 → `NO_CLEAR_DIFFERENCE`。
- 总判词三选一:`RISK_FACE_CANDIDATE_IDENTIFIED` / `NO_OUTCOME_FREE_SEPARATOR` /
  `INCONCLUSIVE_SAMPLE`(受害 n<6 或 fit 超帽未完成主窗口集)。
- **禁止**把 AUC 写成「部署期可预测伤害」——它只是已曝光窗口上的分离度。禁止据结果改阈值、改
  Scope 类、改 HEC-1 冻结面。

## 工件字段

见任务书 §6:`stage`、`data_version`、`sources`、`windows[]`、`series_rows[]`(含 `gain_per_channel`)、
`auc{}`、`routing_harm_check{}`、`counterfactual{}`、`verdict`、
`boundary{llm_calls:0, consumer_fits:≤400 实测, held_out_reads:0, thresholds_changed:0,
operators_added:0, artifacts_overwritten:0}`。

## 回报格式(任务书 §7,七项,缺一不可)

(1) 逐窗口 S / E / 受害表;
(2) 四信号 AUC 表与 `harmed_severe` 逐条(含 `moved`);
(3) pooled vs per-channel 逐窗口对照;
(4) 判词三项;
(5) 实际 fits(分 pooled / per-channel / 合计,对照 400 帽);
(6) 实现中任何偏离任务书之处及理由(含 S1 捕获方式);
(7) 发现的规格矛盾。

======== 分发 A 结束 ========

---

======== 分发 B 开始 / 整段复制 ========

# 任务:D2 HEC-1 课程供给扫描(0 LLM / 0 fit)

你是本任务的**唯一执行方**。只做这一本书。不要做 D1,不要冻结 HEC-1 合同,不要 git 提交。本书只
**盘点与提案**,冻结由主线在 `hec1_contract` 中完成。

**接收模型档**:grok 4.6-xhigh(不必上 opus)。
**工作目录**:`c:\Users\辉\Desktop\Agent\SelfEvolvingHarnessTS-deepseek-guidance-evolution`
(Windows,PowerShell)。Python 可用 `conda activate project` 或直接 `python`。这是可用环境,不是必须改当前
Shell 的要求。

## 必读顺序(按此顺序,读完再写代码)

1. 项目 `AGENTS.md` **§1、§6、§7、§8**(方向、单假设 / first-fault、反过度工程、数据与证据纪律;
   本任务必须携带这些约束。§7:禁止 SHA / manifest / 平台层。§8:development 可读 Context,但不得包装成
   fresh / held-out / Capability 正证据)。
2. `docs/D2_HEC1_COURSE_SUPPLY_SCAN_2026-09-03.md`(**唯一规范**;三张表、曝光交叉核对、切法结论、
   回报格式以它为准)。
3. `docs/HEC_EVOLUTION_MAINLINE_PLAN_2026-09-02.md` **§4.2**(课程与块分配裁定:HEC-1 只用 KDD 天然
   单元;Phase S 与 Phase T 不共块;不以注入数据补足;约 26–28 单元)与 **§12**(D2 是第 2 步的前半,
   产出进冻结件的**数据表**,本身不冻结)。
4. 相关脚本(只读,不改):
   - `evaluation/main_protocol_p4/audit_main_experiment_supply.py`(`READ_ORIGINS`、
     `CANDIDATE_ORIGINS`、`cohort()`);
   - `evaluation/main_protocol_p4/audit_candidate_cohort.py`(`evaluability`、`SWEPT_ORIGINS`);
   - `evaluation/main_protocol_p4/audit_exposure_ledger.py`;
   - `evaluation/main_protocol_p4/scope_initializer.py`(现行 Scope 初始化谓词 `z_peak>=3`);
   - 特征:`methods/ttha/public_tools.py::extract_public_features`(22 维,含 `local_robust_z_peak`、
     `missing_fraction`、`estimated_level_offset` 等)与
     `evaluation/main_protocol_p4/natural_structure_features.py::extract`;分箱
     `contracts/observables.py::observable_numeric_bin`。
5. 工件(只读):`artifacts/main_protocol/p4s_main_experiment_supply.json`(含 `readable_uids` 239 条、
   `why_it_is_not_outcome_selection`)、`p4t_exposure_ledger.json`、`p4u_main_experiment_contract.json`。

## 要做什么(一句话)

0 LLM / 0 Consumer fit / `outcome_values_read: 0`,给 HEC-1 冻结件准备三张表:(1) 六个 40 序列块 ×
候选 origin 的可用性(两口径)+ 曝光标签;(2) 每个候选单元的部署可见模式流行率(含同块相邻 origin 的
`z_peak>=3` 成员 Jaccard——这直接是 H1/H3 的落账原料);(3) Phase S / Phase T 候选单元清单与三种顺序
的生成规则提案。并做曝光交叉核对。只盘点,不冻结。

## 硬约束

- **0 LLM**。**0 Consumer fit**。**`outcome_values_read: 0`**:只读「指标是否可定义」与 origin **之前**
  192 点的部署可见特征。不读任何 gain / error / utility / 预测。
- 只新建**一个** audit 脚本(建议 `evaluation/main_protocol_p4/audit_hec1_course_supply.py`)与
  **一份**工件 `artifacts/main_protocol/p4ac_hec1_course_supply.{json,md}`。
- **不修改任何现有文件**。不 git 提交。不改阈值。不新增 SHA / manifest / Gate。
- **不得触碰** `[80:120]` × {4056, 4296, 4536, 4776, 5016} 的任何东西(p4u 已冻结 held-out;表一
  只列名、标 `HELD_OUT_FROZEN`)。
- 数据身份:`EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`。`readable_uids`(239 条)取自 p4s,
  **顺序不变**。
- 特征**只在 origin 之前的 192 点窗口上算**。
- 可用性沿用 p4s「只读是否可定义」口径,并在工件 `boundary` 里**照抄** p4s 的
  `why_it_is_not_outcome_selection` 声明(p4s 字段路径:
  `boundary_caveat_for_the_ruling.why_it_is_not_outcome_selection`)。

## 表一:六块 × origin 可用性(两口径 + 曝光标签)

块:`[0:40]` `[40:80]` `[80:120]` `[120:160]` `[160:200]` `[200:239]`。最后一块 **39 条**:面 A = 前 20,
面 B = 后 19;若管线要求两面等长,如实报告并给两种切法——**必须给出切法结论**(p4s 曾写
`[200:240]`「only 39 series remain; no cohort forms」,那是供给审计,不是本书结论)。

候选 origin 网格 = `audit_main_experiment_supply` 的 read origins ∪ `CANDIDATE_ORIGINS`,即
**1176 … 5256、步 240**:

```text
1176, 1416, 1656, 1896, 2136, 2376, 2616, 2856,
3096, 3336, 3576, 3816, 4056, 4296, 4536, 4776, 5016, 5256
```

(`READ_ORIGINS` = phase2 `ORIGINS`(1176, 2136, 2376, 2616, 2856) + (1416, 1656, 1896);
`CANDIDATE_ORIGINS` = 3096…5256 步 240。复用 `evaluability`;任务书还要求 raw serving context
非退化——若现成 `evaluability` 未查此项,只许在 audit 脚本内用 origin 前 192 点、0 fit 补查,并在
回报第 6 项声明。)

对每个 (块, origin):

- `usable`:两面全部 served 序列上 missing-aware sMASE 可定义(horizon 内有观测真值、
  `seasonal_scale` 在 `min_pairs=32` 下可算、raw serving context 非退化)。
- **两口径分别计数**:**全部可用**;以及 **≤ 3816**(全部早于首个 held-out origin 4056,保守口径)。
- 曝光标签(可多标,但 held-out 优先):
  - `SPENT_DEV`:`[0:40]`、`[40:80]` 的既读 origin;
  - `SOURCE_V1–V3_READ`:`[160:200]` 的 1896 / 2136 / 2376 / 2616 / 2856 及其 +48 / +240 读窗;
  - `TARGET_HELD_IN`:`[80:120]` × 1896…2856;
  - `HELD_OUT_FROZEN`:不得触碰,只列名;
  - `UNREAD`。

## 表二:逐单元部署可见模式流行率(H1/H3 落账原料)

对每个 usable (块, origin),在**面 A 的 20 条**上、origin 前 192 点:

- `n_z_peak_ge_3`(`local_robust_z_peak >= 3`——现行 Scope 初始化谓词);
- `n_missing_gt_0`、`missing_fraction` 的中位数与最大值、最长缺口长度中位数;
- `n_level_offset_material`(`estimated_level_offset` 绝对值超其分箱首级的序列数);
- 22 维分箱向量的**去重数**(窗口内异质性代理);
- **同块相邻 origin** 间 `z_peak>=3` 成员集的 Jaccard(时间重遇代理;**直接服务 H1/H3 落账设计**)。

不计算、不读取任何 Program 效果。

## 表三:Phase S / T 候选与三顺序生成规则(提案,不冻结)

- **Phase S 候选**:`[160:200]`(全部 usable origin,标注已读者)+ `[200:239]`(usable origin)。
- **Phase T 候选**:`[0:40]`、`[40:80]`、`[80:120]` held-in(1896…2856,+ 若有 ≤3816 的其他 usable)、
  `[120:160]`;按两种口径分别给单元总数。
- 三种顺序的**生成规则**提案(冻结件择一),并给出三份**具体单元序列**:
  - Forward = 块序 `[0:40]→[40:80]→[80:120]→[120:160]`、块内 origin 升序;
  - Reverse = 全序反转;
  - Interleaved = 块间轮转、块内升序。
- 构成三要素自检:重复模式族(同块跨 origin Jaccard 的分布)、族内异质(分箱去重数分布)、模式稀疏单元
  (`n_z_peak_ge_3 < 5` 的单元数)。**三者任一为空 → 如实报告,不调整数据。**

## 曝光交叉核对(硬条件)

列出 Phase S / Phase T 候选的全部 `(series, origin)` 读窗,含 **+48 / +144 / +240** 潜在验证 / 评价面,
与 `p4t_exposure_ledger.json` 的 held-out 对集合求交,**必须为空**。按表一标注每个候选窗口的既曝光状态。
评价面 +144 是否与任何 held-out 读窗重叠也要核(`[80:120]` × 2856 的 +144 = 3000,不重叠;逐个列)。

## 工件字段

见任务书 §6:`stage`、`data_version`、`blocks[]`(切法、面 A/B uid)、`usability[]`、`prevalence[]`、
`proposals{phase_s, phase_t_all, phase_t_le_3816, orderings{forward, reverse, interleaved}}`、
`composition_check{}`、`exposure_cross_check{held_out_intersection: [], per_window_labels}`、
`boundary{llm_calls:0, consumer_fits:0, outcome_values_read:0, held_out_reads:0,
thresholds_changed:0}`。

## 回报格式(任务书 §7,七项,缺一不可)

(1) 六块可用 origin 两口径表;
(2) `[200:239]` 切法结论;
(3) Phase S / T 候选单元数(两口径);
(4) 构成三要素自检结果;
(5) 曝光交集(应为空)与逐窗口标签;
(6) 任何偏离本书之处及理由;
(7) 规格矛盾。

======== 分发 B 结束 ========

---

## 主线验收要点(供用户参考;不复制给接收方)

- **D1**:`boundary.consumer_fits ≤ 400`、`llm_calls = 0`;AUC 表带 CI;`harmed_severe` 逐条含 `moved`;
  pooled / per-channel 逐窗口并列;判词只从预注册词表三选一。S1 不得出现「为分歧再 fit 一遍」。任何
  「部署期可预测伤害」之类的外推措辞退回。改了 `scoped_serving_evaluator.py` 的一律退回。
- **D2**:held-out 交集必须为空;两口径单元数齐;`[200:239]` 切法有结论;三要素自检如实(为空不调数据);
  `outcome_values_read: 0` 且 `consumer_fits: 0`;Jaccard 列必须在。触碰 `[80:120]` × held-out 五 origin
  的一律退回。
- 两书结论都**不进 HEC-1 冻结面**(D1 → HEC-2 决策;D2 → 冻结件的数据表)。两书相互独立,可并行。
