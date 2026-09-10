# DEV-AUTO-1R — 统一比较器与接受框架离线对照

*2026-09-06 · development 级 · 0 次 LLM 调用 · 54 次新增 Consumer fits（上限 100）· 挂钟 ~4 分钟（上限 2 小时）*

机器可读结果：`artifacts/main_protocol/dev_auto1r_repair.json`
预检：`evaluation/main_protocol_p4/smoke_dev_auto1r.py`（10 项全过，0 LLM / 0 fits）

---

## 0. 一句话结论

**修正分母之后，DEV-AUTO-1 的核心结论翻转了。** 挡住修订的不是接受框架，
而是候选本身：在同一单元集合上，25 条提议里只有 3 条超过父版本 material，
且**没有一条**在两条风险线上不更差。把绝对历史屏换成相对参照屏（框架 B），
进入验证的数量仍然是 **0**。真正打开通路的是把历史 replay 降为排序信号
（框架 C）：10 个程序进入队列，5 个拿到 Support 准入，**2 个通过 delayed
权威门**——其中 `outlier_iqr({})` 在它被提出后的**第一个**窗口就通过了。

同时定位到一个此前未记录的接线缺陷：DEV-AUTO-1 的 Fast 侧跑在
`admission_policy.DEFAULT`（`strict_positive_only`，只要伤到任何一条序列就拒）
上，而所有生产 runner 装的都是 released bounded 策略。这一条比"接受规则"
更能解释 20/32 格选 identity。

---

## 1. 比较器修正前后的关键计数

### 1.1 错在哪里

DEV-AUTO-1 用**候选自己可读的格**求均值，用**父版本自己可读的格**求另一个均值，
然后相减。位置 16 上这是"7 格的均值"对"15 格的均值"。

修正后统一走 `dev_auto1_revision.compare_on_a_common_denominator`，
两侧声明同一单元集合、同一服务人群，并把三种状态分开：

| 状态 | 处理 | 依据 |
|---|---|---|
| `READ` | 用真实读数，保留完整服务人群分母 | — |
| `RAW_FALLBACK` | 记 0.0，**留在分母里**，treated = 0 | `run_main_baselines._one_pair`：非法 (program, scope) 对返回 Static-equal gains，"so the arm is scored on what it could actually deploy" |
| `UNKNOWN` | 不填零、不静默丢弃、单列并**扣留完整判词** | 本包新增 |

父版本的逐格读数 DEV-AUTO-1 没有存，本包从共享预测库的 ANCESTOR 条目重建
（`m_r0k._reading_from_entry`，0 fits），并**逐位核对**：13 个可核对步骤的
`mean / worst_single_series_harm / worst_harmed_fraction` 全部与 DEV-AUTO-1
记录的父版本摘要一致，0 处不符。重建可信后才使用。

### 1.2 u16 案例复算

`outlier_iqr({})>winsorize({})`，位置 16，同一 15 格口径：

| | DEV-AUTO-1（各自可读格） | 修正（共同 15 格） |
|---|---|---|
| 候选 | **0.268127**（7 格） | **0.125126**（7 格 READ + 8 格 RAW_FALLBACK） |
| 父版本 | 0.165630（15 格） | 0.165630 |
| 差 | **+0.102497**（超 material） | **−0.040504**（不超） |

候选在 15 格中 7 格 READ、8 格被窗口验证器或退化服务上下文拒绝。
那 8 格上策略实际交付的是 raw，记 0.0 是它真实交付的东西；
DEV-AUTO-1 把它们从分母里删掉，等于只在它能跑的地方给它打分。

### 1.3 重新统计"9 次超父、5 次风险不更差"

不再使用旧的派生布尔字段，全部从共同分母比较重算：

| | DEV-AUTO-1 报告 | 修正后 |
|---|---|---|
| 超过父版本 material | **9** | **3** |
| 且两条风险汇总不更差 | **5** | **0** |
| 且**逐单元**风险都不更差 | 未报告 | **0** |
| 未定（UNKNOWN 扣留判词） | 未报告 | 0 |

仍然超父的 3 条：`u11 outlier_mad>winsorize`、`u17 hampel_filter>winsorize`、
`u23 hampel_filter>winsorize`（后两条是同一程序）。三条都在
`single_series_harm` 的跨单元最坏值上比父版本更差。

"跨单元最坏值不更差"与"逐单元都不更差"分两列报告；本轮两列都是 0，
但比较器对两者的区分有独立测试覆盖（见 §4）。

### 1.4 父程序对照 ≠ 实际执行策略对照

两列分开算，绝不混用：

| 对照 | 单元集合 | 17 条 workflow 提议中超 material 的条数 |
|---|---|---|
| 父**程序** replay | 该步全部已处理格 | 3 |
| 实际**执行**策略 | 本课程真正跑过的格 | **13** |

差距的来源是：本课程里 Fast 绝大多数格服务的是 raw。到 u25，
实际执行策略在 12 个可配对单元上的累计均值是 **0.038351**，
而同期候选普遍在 0.12–0.19。这不是接受规则的问题，是执行层的问题。

⚠️ 这一列是**机会成本**，不是接受判据：候选那一栏是"若在这些单元全部部署"
的读数，逐单元风险线仍然适用。前五个位置属于入场状态、本课程没有跑过，
在这一列里是 UNKNOWN，被排除在分母外并已标明。

### 1.5 pos23 Support 缺测

审计把 `{"position": 23, "origin": null, "status": "NO_STORED_READING"}`
按 raw 回退记了 0.0。缺读数不是拒绝。本包尝试补测，**0 fits 即确定性复现**
`FaceNotEvaluable: evaluation context reached scale floor`，仍记 **UNKNOWN**。

把它从两侧分母中同时剔除后（Support 面 15 单元）：

| 程序 | 面 | 审计发布 vs 祖先 | 修正后 vs 祖先 | 判词 |
|---|---|---|---|---|
| `outlier_iqr>winsorize` | Support | +0.028772 | **+0.049957** | 扣留（1 UNKNOWN） |
| `outlier_iqr>winsorize` | delayed | +0.004215 | +0.004215 | 完整 |
| `outlier_iqr>hampel_filter` | Support | −0.016248 | **+0.001934**（变号） | 扣留（1 UNKNOWN） |
| `outlier_iqr>hampel_filter` | delayed | −0.031083 | −0.031083 | 完整 |

**delayed 面不受影响**，审计的 delayed 结论原样成立。Support 面两个数都要改，
且 `outlier_iqr>hampel_filter` 变号——它 Support **+0.0019**、delayed **−0.0311**，
两面反号，是泛化信号，不合并、不平均。

附带一条仪器观察（只记录、不处置）：祖先在 pos23 Support 面**可读**，
这两个程序不可读。`run_hec1` 对 `FaceNotEvaluable` 的说明是
"a property of the data, not of a policy, so it hits every arm on that unit
identically"——在这一格上不成立。

---

## 2. 三个接受框架

三框架共享合法性检查、Scope 校准和同口径读数；只改变历史 replay 在
"进入独立验证"中的作用。**框架 B 的参照在算任何数之前先定死**：
该步合法持有部署权的策略 = Skill 卡上的程序 + 当时的 serving scope，
即 DEV-AUTO-1 在给任何候选打分之前就记录的 `parent_policy`。
不换成 raw，不事后挑对候选更有利的一边。它是**程序**参照，
实际执行策略那一列单独报告（§1.4），不冒充。

| | A 现行历史绝对屏 | B 相对参照屏 | C 前向验证优先 |
|---|---|---|---|
| 25 条提议中进入验证 | **0** | **0** | **17** |
| 不同程序进入队列 | 0 | 0 | **10** |
| Support 准入至少一次 | — | — | **5** |
| delayed 验证通过 | — | — | **2** |
| 无窗口通过 / 尝试用尽 | — | — | 1 / 7 |
| UNKNOWN / 未评估 | 0 | 0 | 0（另有 2 个窗口记 NOT_EVALUATED，不计入尝试次数） |
| 生命周期允许晋升 | 0 | 0 | **0**（见下） |

**A 和 B 结果相同，且都是 0。** 这是本包最重要的单一发现：换掉绝对屏
不改变任何一条提议的命运，因为在同一分母上它们本来就没有"超父且风险不更差"。
DEV-AUTO-1 提议的"接受参照系单变量对照"实验，答案已经在这里了，且是负的。

**C 是唯一打开通路的改动**，代价是它不再用历史证据否决。

### 2.1 四个阶段严格分开

> 可进入验证 → Support 准入 → delayed 验证通过 → 生命周期允许晋升

- 10 个进入队列 ≠ 通过；
- 5 个拿到 Support 准入，其中 3 个 delayed 门失败：`winsorize` 与
  `outlier_mad>outlier_mad` 均只差 `coverage_floor`（treated < 5），
  `outlier_iqr>winsorize` 差 `harmed_fraction` + `single_series_harm`；
- 2 个 delayed 通过：`outlier_iqr({})`（u10 提出，u11 第一个窗口即通过）、
  `outlier_mad({})>winsorize({})`（u11 提出，u12 失败、u13 Support 被拒、
  u14 通过，用满 3 次尝试中的第 3 次）；
- **晋升仍为 0**：本线的晋升还要求 Fast 在通过的那个单元真的部署该程序。
  验证通过不等于已发生晋升，本包不替它补这一步。

Support 准入用生产规则（`admission_policy._budget_verdict`：harmed_fraction
≤ 0.20、单序列伤害 ≤ 0.30，加上线上循环在候选成为 winner 前施加的 material 线），
**没有新增 treated ≥ 5 的 Support 门**；delayed 用既有权威门（含 coverage floor）。
两处覆盖门的不对称正是上面 3 个"Support 过、delayed 挂在 coverage_floor"的来源。

### 2.2 复算的边界

固定提议轨迹的复算只能说明这些规则如何处理**已经出现过**的候选。
框架 C 一旦真的运行，就会改变状态（晋升、resupply、后续反馈），
后面那些原始提议不再是该新规则自然产生的完整课程。本表不能当成
"框架 C 跑一遍会得到的结果"。

---

## 3. 按真实提议时间计算的后续收益、伤害、覆盖与回退

每个候选只用它**实际被提出之后**的合法单元，按既有课程顺序取窗口，
不看未来成绩重挑。Support 面读数：

| 程序 | 提出于 | 窗口 | 均值收益 | 最坏 HF | 最坏单序列伤害 | 累计 treated | raw 回退窗口 | 结果 |
|---|---|---|---|---|---|---|---|---|
| `hampel_filter` | u6 | 3 | −0.086758 | 0.65 | 3.324 | 38 | 0 | 尝试用尽 |
| `winsorize` | u8 | 3 | +0.025251 | 0.70 | 1.163 | 36 | 0 | 尝试用尽 |
| `denoise_median` | u9 | 3 | 0.0 | 0.0 | 0.0 | 0 | **3** | 尝试用尽 |
| `outlier_iqr` | u10 | 1 | **+0.320571** | 0.05 | 0.103 | 12 | 0 | **通过** |
| `outlier_mad>winsorize` | u11 | 3 | +0.176741 | 0.45 | 0.466 | 47 | 0 | **通过** |
| `repair_level_shift>outlier_mad` | u12 | 3 | 0.0 | 0.0 | 0.0 | 0 | **3** | 尝试用尽 |
| `outlier_mad>outlier_mad` | u14 | 3 | +0.231613 | 0.25 | 0.275 | 38 | 0 | 尝试用尽 |
| `outlier_iqr>winsorize` | u16 | 3 | +0.185885 | 0.25 | 0.594 | 46 | 0 | 尝试用尽 |
| `hampel_filter>winsorize` | u17 | 3 | +0.236453 | 0.30 | 0.612 | 38 | 0 | 尝试用尽 |
| `outlier_iqr>hampel_filter` | u17 | 2 | +0.130672 | 0.40 | 0.731 | 32 | 0 | 无窗口通过 |

两个程序（`denoise_median`、`repair_level_shift>outlier_mad`）在**每一个**
后续窗口都被窗口验证器拒绝，全程回退 raw、覆盖 0——它们在历史屏上记的
`NOT_READABLE` 不是偶然，是这两个程序在这条线上根本跑不起来。

pos23 Support 面对两个候选不可评估，记 NOT_EVALUATED，**不消耗验证次数**：
一个没人读得出的窗口不是候选验证失败的窗口。

---

## 4. 两臂状态对称性与 Fast 预算接线

三处修复，均已改代码并有针对性测试；**没有重跑两臂完整课程**。

### 4.1 冻结臂只冻结 Skill

原配置给控制臂 `ArmSpec(..., write_back=False)`，在 `run_hec1` 里这一个标志同时关掉四件事：
每个单元重建方法并**丢弃整个 Draft ledger**、永不 `activate_approved`、
永不扩充 `bank` 与 `processed`；我又在其上加了每格 `_seed_arm` 全量重置。
那是"冻了五样"的控制臂，而本包只想比一样。

改为两臂都 `write_back=True`，删除每格重置，两臂只在**一处**不同：
`revision_step` 只对修订臂开放。冻结的不变量改为**检查**而非假设——
课程结束时断言控制臂的 `K0` 卡 body 只出现过一个版本（`frozen_arm_invariant`）。
仍然共享的：历史收纳、Draft 生命周期与验证事件、候选供给与 resupply、
Fast 侧对已批准 Skill 的激活。

### 4.2 Fast 预算退到包级

原来三处各自截断：`BudgetedAgentBackend(maximum_calls=5)`、
`BudgetGuard.per_unit_arm_cap=5`、以及 HEC-1 合同里的 `PER_UNIT_ARM_BUDGET["llm_calls"]=5`。
改为：`PER_ARM_LLM = MAX_LLM // 2 = 200` 对称预分配，
每格开头把 `guard.per_unit_arm_cap` 重写成**该臂剩余的包级额度**，
后端每单元重建时也拿这个数。**HEC-1 合同一字未动**（仍是 5），
只是这个 development 配置不再对内循环采用它，并在产物里写明。
请求超时、总 fit 上限、挂钟截止全部保留。
`_classify_cell_stop` 把预算截断 / 模型主动弃权 / 风险拒绝分三个计数记录。

这条修复有直接证据支持：run2 里 9 格抛出 `AgentCallBudgetExceeded`，
修订臂的 4 格恰好是 **u8、u16、u23、u24**——而这四个位置正是
**永远没有进入该臂 `processed` 列表**的四个单元。五次调用帽把格子打断，
格子被打断就不入历史，历史不入就不进反馈。到 u25 时提议器看到的是 17 格
而不是 21 格。这不是"Fast 少调了几次"，是反馈证据面被静默削掉了。

### 4.3 （新发现）Fast 跑在 strict 准入规则上

`run_hec1`、`run_source_line`(v1/v2/v3) 都装 `p4b_contract.BOUNDED_POLICY`；
`run_dev_auto1_skill_revision.py` **从未调用 `install_policy`**，
于是继承模块默认值 `strict_positive_only`：**只要伤到任何一条序列就拒**。

在 run2 自己的 39 条 probe 行上重判（0 fits）：

| | 数量 |
|---|---|
| 实际运行的 strict 规则准入 | **3 / 39** |
| released bounded 规则本会准入 | **9 / 39** |
| 两者判决不同的行 | 6（frozen u23、revising u5、frozen u16，各 2 行） |
| 全课程记录的 risk_refusal | **0** |

`risk_refusal` 恒为 0 的原因是机械的：strict 的拒绝理由是
`relation_not_positive`，它不在 `RISK_REFUSAL_REASONS` 里，
所以被拒的 probe 从不进入风险拒绝账本——而"程序有用、作用面太宽"
这类证据正是修订通道被设计来消费的东西。**通道的上游供料一直是空的。**

已修：DEV runner 现在装 released 策略（不新增任何常数，就是 P4 线已发布的
0.20 / 0.30）。**两臂未重跑**，run2 的数字仍带 strict 规则，报告中如实标注。

### 4.4 预检

`smoke_dev_auto1r.py`，10 项，0 LLM / 0 fits，全过：
三状态区分、共同分母（含"旧分母会翻转判词"的构造用例）、UNKNOWN 扣留判词、
逐单元风险 ≠ 跨单元最坏、控制臂只冻 Skill、Fast 预算包级化（含 guard 行为测试）、
released 准入规则已安装、三种停格分开计数、风险常数与 Scope 栅格未动、
评价面不可达。

---

## 5. 成本与未计量项

| 项 | 实测 | 上限 |
|---|---|---|
| LLM 调用 | **0** | 0 |
| 新增 Consumer fits | **54** | 100 |
| 挂钟 | ~4 分钟 | 2 小时 |

54 次由本包新增的 33 条预测库条目实测得出（`physical_fits` 求和）；
库中原有 230 条**逐位未变**，已核对。产物里另记 `reserved_this_run`
（下限保护用的事前预留）与实测值的区别。

**未计量项**（明确列出，不假装为零）：

1. pos23 Support 面对两个候选**至今未知**，不是 0.0。剔除后 Support 面
   比较基于 15 单元，判词已扣留。
2. 框架 C 的复算是**固定轨迹**复算，不等于框架 C 真跑一遍的课程。
3. 实际执行策略对照缺少课程未跑过的 5 个入场单元，记 UNKNOWN。
4. run2 的所有 Fast 侧数字都带 strict 准入规则；已修接线但未重跑。
5. 两臂对称性与预算接线的修复只做了针对性测试，其课程级后果未测。
6. 本包只读 Support 与 delayed 两个面；+144、密封材料、5 个
   TARGET_HELD_IN 单元全程未读。Consumer、DSL、风险阈值、风险分母、
   Scope 栅格一律未改；未提交代码；未覆盖任何历史收据。

---

## 6. 报告更正（四条）

**① "父版本不通过历史屏 ⇒ 所有子版本都不可能通过"——不成立，我此前这样推了。**
`outer_loop.screen` 是绝对门，父版本 16/16 步不过它，这两件事都是真的；
但从中推不出没有候选能过。事实上本包的复算显示，候选没能过屏与
父版本过不过屏无关：修正分母后它们连"超父且风险不更差"都做不到（3 / 0）。
框架 B 直接检验了这一点：把绝对屏换成相对参照屏，进入验证的数量仍是 0。
我此前把"接受层挡死"当成结论，实际是分母错位造出来的。

**② "Scope 八次失败都因为比例栅格"——错，0/8 归因于栅格。**
栅格不是一个元组：6 个特征是 (0,1,3,6)，4 个是 (0,.01,.05,.2)，
`period_change_score` 与 `period_reliability` 另有自己的边界。
Slow 实际命名的三个特征每个都试满 4 条边。逐条重判的结论是
**8/8 因风险线失败**（`harmed_fraction` 与 `single_series_harm`），
0/8 因栅格。其中 `period_reliability >=` 的四条边给出四个**完全不同**的
treated 数（182 / 42 / 23 / 13），四条全部失败——这是风险答案，不是栅格答案。
DEV-AUTO-1 建议的"修比例栅格"实验因此失去依据。
（另记一条现象：`level_excursion_score <=` 在 6.0 上 treated 反而**变大**
（120→123），`<=` 子句在超出数据范围的边上不收窄；这是独立观察，本包不处置。）

**③ "两次运行数字对调 ⇒ 两臂差异是纯噪声"——推不出来。**
run1 与 run2 的四个数字确实**逐位相同、仅两臂互换**。但完全相同的两个值
在臂之间对调，恰恰不是抽样噪声的样子；它说明在那套配置下两臂是**可交换的**——
修订通道从未触发，控制臂又被多冻了四样，两臂跑的其实是同一个策略，
谁拿到哪条轨迹只取决于 Fast 路径里的非确定顺序。正确的说法是
"该配置下不存在可测的臂对比"，而不是"测到的差异是噪声"。
n = 2 也估不出任何方差。

**④ "Fast 探到 +0.407 却选 identity ⇒ 漏选好候选"——不成立。**
那条 probe（frozen u15，`outlier_mad`）harmed_fraction = 0.25，
高于 0.20 的线；在 strict 与 released bounded 两条规则下**都会被拒**。
它不是被漏掉的好候选。20/32 选 identity 的真实原因见 §4.3：
运行在 strict 规则上，39 条 probe 只准入 3 条，而 released 规则会准入 9 条。

---

## 7. 下一次最值得实跑的一个实验

**在修好的接线上重跑两臂完整课程一次（16 个单元，两臂），
接受规则用框架 C，其余一切不变。**

依据：

1. 本包已经用离线复算排除了两个候选实验。接受参照系的单变量对照
   （绝对屏 vs 相对屏）答案是 0 vs 0，不值得实跑；Scope 栅格 0/8 归因，
   也不值得实跑。这两个正是 DEV-AUTO-1 排在前两位的建议。
2. 三个接线缺陷（控制臂多冻四样、五次调用帽削掉四个单元的历史、
   Fast 跑在 strict 准入上）都已修好且有测试，但**没有一个**在课程级别
   被观测过。run2 的所有臂间数字都产生于这三个缺陷之下。
3. 框架 C 是唯一在离线复算里打开通路的规则（10 进队列 / 5 Support 准入 /
   2 delayed 通过），而它未被验证的恰恰是**动态部分**：一旦候选真的晋升，
   后续提议、resupply 与反馈都会改变，固定轨迹复算说不了话。
4. 预期成本可控：LLM 侧包级 400 次上限已存在；fits 侧 run2 用了 293，
   预测库现已多出 33 条可复用条目。

**不要**在这次实验里同时改风险线、分母或 Scope 栅格；
也**不要**把 `outlier_iqr>winsorize` 喂给 Agent——它是研究者指定的程序对照
（见下），不是自主提议的标准答案。

### 附：研究者指定的程序对照

`outlier_iqr({})>winsorize({})` 保留为指定程序对照，与另一个被拒程序
`outlier_iqr({})>hampel_filter({})` 并列呈现：

| | Support（15 单元，判词扣留） | delayed（16 单元，判词完整） |
|---|---|---|
| `outlier_iqr>winsorize` vs 祖先 | +0.049957 | **+0.004215**（门 7/16 vs 4/16） |
| `outlier_iqr>hampel_filter` vs 祖先 | +0.001934 | **−0.031083**（门 2/16 vs 4/16） |

它不是新发现，也不是下一次自主提议的标准答案。未来做自主 live 时
不向 Agent 提供这个提示、不要求它重新提出；若直接供应该程序，
必须标注为指定程序对照。

---

## 8. 交付清单

| 文件 | 说明 |
|---|---|
| `evaluation/main_protocol_p4/dev_auto1_revision.py` | 追加共同比较器（`classify_reading` / `compare_on_a_common_denominator` / `frame_b_admits`），既有函数未改 |
| `evaluation/main_protocol_p4/run_dev_auto1_skill_revision.py` | 三处接线修复：控制臂只冻 Skill、Fast 预算包级化、装 released 准入策略 |
| `evaluation/main_protocol_p4/run_dev_auto1r_repair.py` | 本包入口：复算、pos23、三框架、前向验证、Scope 归因、Fast 归因、响应性 |
| `evaluation/main_protocol_p4/smoke_dev_auto1r.py` | 10 项针对性测试，0 LLM / 0 fits |
| `artifacts/main_protocol/dev_auto1r_repair.json` | 机器可读结果 |
| `_scratch/dev_auto1r_fit_ledger.json` | 本包 fit 账本（54 / 100） |
| `docs/DEV_AUTO1R_COMPARATOR_AND_ACCEPTANCE_FRAMES_2026-09-06.md` | 本文 |

历史收据 `dev_auto1_skill_revision__run{1,2}.json` 与
`dev_auto1_blocked_candidates__run2.json` 未改动；
`docs/DEV_AUTO1_AUTONOMOUS_SKILL_REVISION_2026-09-06.md` 保留原样，
其被推翻的结论由本文第 6 节逐条更正。

### 附：反馈响应性观察

不是因果学习的证据，也没有请任何 LLM 评审；只是"上次失败之后下一次改了什么"：

| 变化类型 | 次数 |
|---|---|
| 连续提议对 | 13 |
| 换算子 | 6 |
| 增删步骤 | 6 |
| 改 Scope 子句 | 10 |
| 切换提议种类 | 8 |
| 完全重复 | **1**（u23，父版本读数与 u17 逐位相同） |
| 弃权 | 2 |

一次调用常同时改多样，所以标签是集合而非单选；DEV-AUTO-1 把它压成单选，
才有了"10 次改 Scope"的说法。
