# D2 任务书:HEC-1 课程供给扫描(0 LLM / 0 fit)

日期:2026-09-03。地位:HEC 路线图第 2 步(§12)的前半;主线起草,sol 已裁课程范围(KDD 天然单元、
Phase S 与 Phase T 不共块、不以注入数据补足)。本书只**盘点与提案**,不冻结任何东西;冻结由主线在
`hec1_contract` 中完成。预算:**0 LLM;0 Consumer fit**;只读部署可见 Context 与"指标是否可定义",
不读任何 gain / error / utility。

## 0. 目的

给 HEC-1 冻结件提供三张表:(1) 六个 40 序列块 × 候选 origin 的可用性;(2) 每个候选单元的部署可见
模式流行率;(3) Phase S / Phase T 的候选单元清单与三种顺序的生成规则提案。并做曝光交叉核对。

## 1. 材料与约束

- 数据:`EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`(天然缺口版),`readable_uids`(239 条)取自
  `artifacts/main_protocol/p4s_main_experiment_supply.json`,顺序不变。
- 块:`[0:40]` `[40:80]` `[80:120]` `[120:160]` `[160:200]` `[200:239]`(最后一块 39 条:面 A = 前 20,
  面 B = 后 19;若管线要求两面等长,如实报告并给两种切法)。
- 复用:`evaluation/main_protocol_p4/audit_candidate_cohort.py::evaluability`、
  `audit_main_experiment_supply.py`(`CANDIDATE_ORIGINS`、`cohort()`);特征用 TTHA 公开特征
  `extract_public_features`(含 `local_robust_z_peak`、`missing_fraction`、`estimated_level_offset` 等
  22 维)与 `natural_structure_features.extract`,**只在 origin 之前的 192 点窗口上算**。
- 禁止:读取 `[80:120]` × {4056, 4296, 4536, 4776, 5016} 任何东西(p4u 冻结 held-out);读取任何 outcome
  值;改任何阈值;新增 SHA/manifest。

## 2. 表一:块 × origin 可用性

候选 origin 网格 = `audit_main_experiment_supply` 的 read origins ∪ `CANDIDATE_ORIGINS`
(1176 … 5256,步 240)。对每个 (块, origin):

- `usable`:两面全部 served 序列上 missing-aware sMASE 可定义(horizon 内有观测真值、`seasonal_scale`
  在 `min_pairs=32` 下可算、raw serving context 非退化)——沿用 p4s 的"只读是否可定义"口径,并在工件
  `boundary` 里照抄其 `why_it_is_not_outcome_selection` 声明。
- 两种口径分别计数:**全部可用**;**≤ 3816**(全部早于首个 held-out origin,保守口径)。
- 标注已曝光状态:`SPENT_DEV`(`[0:40]`、`[40:80]` 的既读 origin)、`SOURCE_V1–V3_READ`(`[160:200]` 的
  1896/2136/2376/2616/2856 及其 +48/+240 读窗)、`TARGET_HELD_IN`(`[80:120]` × 1896…2856)、
  `HELD_OUT_FROZEN`(不得触碰,只列名)、`UNREAD`。

## 3. 表二:候选单元的部署可见模式流行率

对每个 usable (块, origin),在面 A 的 20 条序列上、origin 前 192 点窗口:

- `n_z_peak_ge_3`(`local_robust_z_peak >= 3` 的序列数——现行 Scope 初始化谓词,`scope_initializer.py`);
- `n_missing_gt_0`、`missing_fraction` 的中位数与最大值、最长缺口长度中位数;
- `n_level_offset_material`(`estimated_level_offset` 绝对值超其分箱首级的序列数);
- 22 维分箱向量的**去重数**(窗口内异质性代理);
- 同块相邻 origin 间 `z_peak>=3` 成员集的 Jaccard(时间重遇代理;直接服务 H1/H3 的落账设计)。

不计算、不读取任何 Program 效果。

## 4. 表三:候选单元清单与顺序生成规则(提案,不冻结)

- **Phase S 候选**:`[160:200]`(全部 usable origin,标注已读者)+ `[200:239]`(usable origin)。
- **Phase T 候选**:`[0:40]`、`[40:80]`、`[80:120]` held-in(1896…2856,+ 若有 ≤3816 的其他 usable)、
  `[120:160]`;按两种口径分别给单元总数。
- 三种顺序的**生成规则**提案(冻结件择一):Forward = 块序 `[0:40]→[40:80]→[80:120]→[120:160]`、块内
  origin 升序;Reverse = 全序反转;Interleaved = 块间轮转、块内升序。给出三份具体单元序列。
- 构成三要素自检:重复模式族(同块跨 origin Jaccard 的分布)、族内异质(分箱去重数分布)、模式稀疏单元
  (`n_z_peak_ge_3 < 5` 的单元数)。三者任一为空 → 如实报告,不调整数据。

## 5. 曝光交叉核对

列出 Phase S / Phase T 候选的全部 (series, origin) 读窗(含 +48、+144、+240 潜在验证/评价面),与
`p4t_exposure_ledger.json` 的 held-out 对集合求交,**必须为空**;并按 §2 标注每个候选窗口的既曝光状态。
评价面 +144 是否与任何 held-out 读窗重叠也要核(`[80:120]` × 2856 的 +144 = 3000,不重叠;逐个列)。

## 6. 输出

`artifacts/main_protocol/p4ac_hec1_course_supply.json` + `.md`(一份):`stage`、`data_version`、
`blocks[]`(切法、面 A/B uid)、`usability[]`(块 × origin × usable × 曝光标签)、`prevalence[]`
(表二逐单元)、`proposals{phase_s, phase_t_all, phase_t_le_3816, orderings{forward, reverse, interleaved}}`、
`composition_check{}`、`exposure_cross_check{held_out_intersection: [], per_window_labels}`、
`boundary{llm_calls:0, consumer_fits:0, outcome_values_read:0, held_out_reads:0, thresholds_changed:0}`。

## 7. 回报格式

(1) 六块可用 origin 两口径表;(2) `[200:239]` 切法结论;(3) Phase S / T 候选单元数(两口径);
(4) 构成三要素自检结果;(5) 曝光交集(应为空)与逐窗口标签;(6) 任何偏离本书之处及理由;(7) 规格矛盾。
