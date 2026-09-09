# DEV-SEQ-1：逐序列决策、相似失败分组、一次共享知识更新（开发级，2026-09-07）

**判词：`PER_SEQUENCE_EXECUTED__NO_CONDITION_TO_WRITE`**

逐序列观察、决策与执行**做到了**，两次独立运行各 30 次决策、共 60 次，交付数与决策数
**零不符**。分组与成功对照**做到了**。共享知识更新**两次都没有形成合格更新**：
第一次真实 Slow 弃权（`insufficient_public_evidence`），**这一次是我的证据包缺陷**——
成功对照被交给 Slow 时不带序列身份、不带程序、不带 delayed 面、不带可见特征，
可见 Pattern 展布只算了失败一侧；修好后第二次 Slow **真的提出了**一张指导卡并经真实
controller 编译入库，但被**预注册的验证**在 L2 拒绝：它写的条件只覆盖 6 条失败序列中的
**1 条**。原因随后被机械读出：**12 个部署可见特征里没有一个能把失败与成功分开**，
而且 6 条失败序列里有 **4 条同时出现在成功一侧**——同一条序列、同一个程序，在不同窗口
给出相反答案。因此后续段（两臂）未开：没有合格更新时两臂就是同一个库，比较失效。

**范围**：开发级机制验证。五个已曝光 development 单元、每单元 10 条序列、一门课程、两次运行。
不主张跨域泛化，不是 A5 结果，不做显著性检验。全程 0 次 held-out 读、0 字节密封数据、
未触碰 `TARGET_HELD_IN`（positions 18–22）与 +144 评价面，**未提交 git**，
**编辑的核心文件数 = 0**。

**读数分母**：一次**决策** = 一个 (单元, 臂, 序列)；一条逐序列增益 = **该序列自己**在
**它自己运行的那个程序**下的增益；一个单元的交付 = 该单元决策人群上的均值，每条序列
各按自己的程序计分，未处理序列贡献一个**实测的** 0.0。
**这些读数与 DEV-AUTO-1/2/3、DEV-KNOW-1 不可比**（那些是"一个程序广播到 20 条序列"的
组级分母），**不并表、不相减**。

源：`evaluation/main_protocol_p4/{per_sequence,dev_seq1_knowledge,run_dev_seq1,smoke_dev_seq1,audit_dev_seq1}.py`
工件：`artifacts/main_protocol/{dev_seq1_per_sequence__run1,dev_seq1_contrasts__run1,dev_seq1_per_sequence__run2,dev_seq1_contrasts__run2,dev_seq1_reference_readings__all}.json`
设计记载：项目 `AGENTS.md` §5.3；执行计划 `docs/NEXT_EXPERIMENT_EXECUTION_PLAN_2026-09-05.md` §0

---

## 一、先修正的设计记录（任务书 §二）

写入 `AGENTS.md` §5.3（追加式，历史数字全部保留）：

1. **决策单位是单条序列及其当前合法窗口**，不是 cohort 的代表序列。到 DEV-KNOW-1 为止，
   每个单元只构造一个 `PreparationRequest`，其 values 来自
   `cell.observation_block = values[support_a[0]][:origin]`——该 block 的**第一条 eval 序列**——
   一个 compiled 程序再由 `scoped_evaluate` 广播到 20 条被服务序列。本包 smoke
   `every_request_is_bound_to_its_own_sequence` **实跑**核实了这条事实（`observation_block`
   与 `values[support_a[0]][:origin]` 逐值相等，`equal_nan`；该 uid 确实是决策人群成员）。
2. **共享的是 Agent、工具与 Skill**，不是强制共享同一个处理程序。允许不同序列选中相同
   程序，也允许 identity，**程序多样性不作为成功条件**。
3. **Scope 保留**为适用条件（知识卡的 `observable_applicability`）、作用几何与执行边界，
   **不再替代逐序列程序生成**。
4. **旧组级实验与判词保留**，但测试范围明确为"cohort 广播式单程序 + 组级知识写回"。
5. **与 R4A 的关系**：R4A 测得逐序列增益不是序列的稳定属性（块内 ICC≈0）。这不改变决策
   单位，但约束能对逐序列**条件化**期待什么。本包 §五 的读数与该结论**独立地一致**。
6. **知识更新只发生在批次边界**；单条序列的 delayed 损失不在批次中途撤销共享知识。

同步：执行计划新增 §0 更正插入，指明其 M-W / Scope 线 / Workflow 线读数的主语是
cohort 级程序与 cohort 级 Scope。未改任何历史数字，未动 Fable 负责的报告。

---

## 二、实现：逐序列决策与正确执行（任务书 §三）

**优先复用，未重建。** `PreparationRequest`、Fast、`ScopeExecutor`、
`scoped_serving_evaluator`、`ReplayPredictionCache`、`online_loop.run_online_round`、
`open_delayed`、`group_fault`、`fault_cases`、`TTHASlowAgent`、`EditController` 全部原样使用。
**本包编辑的核心文件数 = 0**（smoke `this_package_edits_no_core_file` 把
`methods/ contracts/ runtime/ evaluation/minipipe/` 的工作树状态与包前基线逐行比对；
基线是 2026-09-05 就存在的 `online_loop.py` 与 `scope_executor.py` 两处改动）。

### 2.1 任务书五条要求逐条

1. **每条请求绑定自己的 UID、历史、Task/Consumer Context 和可见特征。**
   `per_sequence.sequence_request` 以 `dataclasses.replace` 只替换三个字段
   （`series_uid` / `values` / `observed_pattern_spec`）；TaskSpec、下游模型类、metric 与
   deployment constraints 逐字节沿用冻结线。特征由该序列 origin 前窗口经
   `extract_public_features` 取得；历史是该序列自己的 Episode 列表——**一条序列的决策
   不读另一条序列的生 Episode**。
2. **每条程序实际应用于对应数据。** 每条序列的 Fast 会话拿到一个
   `per_sequence.SequenceView`：执行边界固定为**这条序列**，`gain` 报成**这条序列自己的
   增益**。这一点是实质的而非形式的——若沿用 20 条分母，一条序列真实的 +0.10 会到达闸门
   时变成 +0.005，恰好压在物料线上，那是算术不是证据。`per_view_gain` 是配套的单元素列表，
   于是 `classify_relation` 与准入规则读到的是 **n=1 的人群**；该人群下 strict 与
   `bounded_risk_v1` 判定完全一致（单条序列不可能既聚合为正又被物料伤害），**阈值一个未改**。
3. **允许不同序列选择相同程序，也允许 identity。** 两者都实际发生（§四）。
4. **同批用同一冻结知识版本；Slow 只在批次边界跑。** `allow_fast_skill=False`、
   `store=None`、`allow_slow=False`、`allow_group_slow=False`（`run_dev_seq1.py:309`），
   因此逐序列一轮**结构上不可能**铸卡、PATCH 或撤销任何卡。一轮若"本会"撤销（部署的是卡
   且 delayed < −M），记 `revocation_pending_no_store` 作为边界证据，不动库。
5. **隔离，可并行。** 每条序列一个新 backend、新 core、新 `TTHAMethod`（同一冻结快照 +
   该序列自己的历史）。隔离是结构性的而非开关。本包**串行**执行，只为让成本账目留在一条
   时间线上；并行不改变任何读数。

### 2.2 Consumer 训练/服务语义：查明后复用，未重设计

`scoped_serving_evaluator.scoped_evaluate` 对一个程序拟合**两个**模型——raw 训练行上的
raw 模型与**已准备**训练行上的 program 模型——**两个都不依赖 Scope**；Scope 只进入一行
`prediction = where(in_scope, program_prediction, raw_prediction)`，而每条被服务序列的损失
只由它自己那一行预测算出。因此：

> 一条序列在程序 P 下的读数，**无论它单独被处理还是与另外十九条一起被处理，都是同一个数**。

这正是 `run_hec1.ReplayPredictionCache` 已写明并依赖的性质。所以逐序列的异质赋值
（uid → 程序）**不需要新建模**：按程序分组各读一次、各取自己那一行，逐位精确。
smoke 的 `different_programs_execute_on_their_own_data` 与
`the_reading_does_not_depend_on_co_treatment` **实跑**核实了这两点（后者还核实了同读
**0 额外 fit**）。

**必须说清的一点**：训练语料是**共享**的。选了程序 P 的序列，由"train 角色序列经 P 准备后
拟合出的模型"来服务；不同序列选不同程序，就由不同的已拟合模型服务。这不是本包引入的
新建模——它就是冻结的 serving-side 双管线，只是从 2 条管线变成 K+1 条——但它意味着一条
逐序列读数的准确名字是

> **该准备策略对该评价序列的影响**，不是"该序列自己处理自己的独立因果效应"。

train 与 eval roster 不相交，因此没有任何序列自己的程序碰到另一条序列的服务 context。
**未**改成逐序列独立建模，**未**改成混合准备后的共享训练，**未**建模型路由层。
原单序列接口（`contracts/method.PreparationRequest`，本就带 `series_uid` 与单条 `values`）
本来就承载得了逐序列程序；**没有出现"契约无法承载"的实质选择需要上报**。

---

## 三、必要检查（0 LLM，6 fits，全部实跑）

`smoke_dev_seq1.py`，9 项全过（运行前跑一次并嵌进工件；改证据包后又独立跑过两次，均全过）：

| 检查 | 结论 |
| --- | --- |
| `the_segments_are_registered_and_do_not_overlap` | 五个单元都是已曝光 development 单元；`TARGET_HELD_IN` 一个未碰；构造窗（1176/1416 + delayed 1224/1464）、留出窗（1656/1704）、后续窗（2136/2184、2376/2424）互不重叠 |
| `every_request_is_bound_to_its_own_sequence` | 10 条请求各带自己的 UID 与 values（逐值相等，`equal_nan`）；两两 values 不同；10 个各不相同的 observed pattern；10 张各不相同的公开特征卡；Task/Consumer 全同；并核实**旧形状确实只说了一条代表序列** |
| `different_programs_execute_on_their_own_data` | 异质赋值 (P1, P2, identity) 逐位等于各自单独读数；identity 恰为 0.0；两程序给出不同的数 |
| `the_reading_does_not_depend_on_co_treatment` | 同一序列单独读与与邻居同读**同一个数**，同读**0 额外 fit**；邻居的数确实变了 |
| `an_untreated_sequence_is_bit_identical_to_static` | 空 Scope 与 Static 的 `per_view_smase` 逐位相等，program 管线从未运行 ⇒ 未处理序列的 0.0 是**实测**的，不是填的 |
| `grouping_keeps_the_successes_and_leaves_singletons_alone` | 相似失败成组并**带匹配成功对照**；孤例留在 Episode 未被塞组；先过可比性；组内记症状与可见 Pattern 展布；错误类按机械证据屏蔽 |
| `a_knowledge_update_can_enter_the_next_fast` | 指导卡经**真实** controller 编译入库；匹配序列检索到、不匹配序列检索不到；父卡逐字段存活、库恰好 +1；它**不供给任何候选**；body 进到 prompt 面；三种越权提案（带冻结程序 / 声明候选供给权 / 带 serving scope）全部被 `guidance_preflight` 拒绝 |
| `the_transport_is_the_one_the_previous_package_used` | 与 DEV-KNOW-1 收据一致：`https://api.deepseek.com/v1` / `deepseek-chat`，来源 `M0_AGENT_*` |
| `this_package_edits_no_core_file` | 工作树与包前基线逐行相同 |

---

## 四、真实运行

### 4.1 逐序列决策确实发生了（两次运行一致）

| | run1 | run2 |
| --- | --- | --- |
| 决策数（3 单元 × 10 序列） | 30 | 30 |
| 每次决策绑定自己的序列 | 是 | 是 |
| 不同的公开特征卡 | 10 | 10 |
| **交付数 ≠ 决策数的次数** | **0 / 30** | **0 / 30** |
| 部署 / identity | 19 / 11 | 18 / 12 |
| 各单元内不同处理结果数 | 2, 2, 3 | 2, 2, 2 |
| 逐序列自身 Support 均值 | +0.174387 | +0.170284 |
| 逐序列自身 delayed 均值 | +0.061771 | +0.068199 |
| delayed 缺测（UNKNOWN） | 0 | 0 |
| Support 过线后 delayed 翻负 | 4 / 19（21.1%） | 3 / 18（16.7%） |

"交付数 ≠ 决策数"这一行是本包最强的执行检查：对每条序列，**Fast 决策时测到的那个数**
必须等于**被执行的整体策略交付给这条序列的那个数**。两次运行各 30 条，全部相等
（差 < 1e-12）。这排除了"程序被广播到它没被选中的序列上"和"某条序列按别人的程序计分"。

**异质性是真的。** run1 的 u7 有 `outlier_mad` 与 `impute_linear` 并存；run2 的 u7 有
`outlier_mad` 与 `impute_ema`；两次运行的 u8/u9 都有一批序列部署、一批 identity。

> **仪器更正与一条别名勘误（本包内发现、核实并修好）。**
> `programs_used` 原先以**只含算子名**的 label 为键，于是"同算子不同参数"的两个程序会
> 塌成一个键，per-unit 摘要少报一个程序。已改为 `program_label_with_params` 作键，
> 审计另加 `operator_labels_hiding_more_than_one_program` 显式报告。**读数不受影响**
> （逐序列增益按程序签名计算，60/60 决策全部对得上）。
>
> run2 的 u8 出现了唯一一处这样的键碰撞：五条序列用 `outlier_mad{}`，T147 用
> `outlier_mad{z_threshold: 3.5}`。**但按项目自己的纪律（`AGENTS.md` §5.1：程序计数必须
> 先按逐序列增益向量去重）核对后，这两个是别名**：T147 在两次运行里的读数逐位相同
> （Support +0.018029、delayed +0.513604），而 `outlier_mad` 的
> `public_parameter_schema` 是 `null`——该参数根本不是它声明的公开参数。
> 所以 **run2 u8 按效果去重后只有一个程序**，不是两个；本报告不把它算作参数层的
> 逐序列差异。

### 4.2 逐单元交付轨迹与两个参考读数

参考读数（`dev_seq1_reference_readings__all.json`，0 LLM / 20 fits，**不并入包账本**）：
`raw` = 什么都不准备（按构造即 Static，两面恰为 0.0）；`fixed` = 把**祖先程序
`outlier_mad`** 给决策人群里的每一条序列——即 cohort 级答案。

| 运行 | 单元 | 部署/identity | Support 交付 | delayed 交付 | 减 fixed（Support） | 减 fixed（delayed） | 单元权威门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run1 | u007 (1176) | 10 / 0 | +0.315397 | +0.072749 | **+0.032744** | +0.003681 | 否（single_series_harm 0.691） |
| run1 | u008 (1416) | 6 / 4 | +0.158156 | +0.127305 | **+0.090887** | −0.002928 | **是** |
| run1 | u009 (1656) | 3 / 7 | +0.049608 | −0.014742 | **+0.269299** | +0.012631 | 否（coverage_floor, aggregate） |
| run2 | u007 | 10 / 0 | +0.323780 | +0.067098 | **+0.041127** | −0.001970 | 否（single_series_harm） |
| run2 | u008 | 6 / 4 | +0.158156 | +0.127305 | **+0.090887** | −0.002928 | **是** |

（u008 两次运行的六条部署序列、程序与逐位读数完全相同——两次独立会话在这一格上做了
同一件事；u007/u009 则不同。这是稳定性的一处观察，不是复现性实验。）
| run2 | u009 | 2 / 8 | +0.028915 | +0.010194 | **+0.248606** | +0.037567 | 否（coverage_floor） |

均值（逐单元配对）：

| 对照 | run1 Support | run1 delayed | run2 Support | run2 delayed |
| --- | --- | --- | --- | --- |
| 逐序列 − raw | +0.174387 | +0.061771 | +0.170284 | +0.068199 |
| 逐序列 − **fixed（cohort 级答案）** | **+0.130977** | +0.004461 | **+0.126873** | +0.010890 |

**准确的读法**：逐序列决策在 **Support 面**（Fast 自己选择所依据的那一面）**六格全部**优于
把同一个祖先程序发给全体，两次运行同号同量级；在**权威的 delayed 面**上，差值是
+0.0045 / +0.0109，**量级上就是噪声**，不构成"逐序列条件化在后续窗口有效"的证据。
最大的一格是 u009：cohort 级答案在那里是**有害的**（Support −0.219691），而逐序列策略
靠对 7–8 条序列弃权把交付拉回微正（+0.0496 / +0.0289）。**这一格上"少做"就是收益的来源，
不是"挑对了程序"。**

覆盖随窗口下降（10 → 6 → 3 与 10 → 6 → 2）：越往后 Fast 越倾向 identity。
单元权威门（min_treated=5、aggregate、harmed_fraction、single_series_harm，**阈值一个未改**）
在两次运行里都**只在 u008 通过一次**；u007 都栽在单序列伤害，u009 都栽在覆盖下限。
该门在本包只作**读数**记录：本包唯一的知识写入路径在批次边界，没有任何单元门会激活任何东西。

---

## 五、批次边界：分组、真实 Slow、独立验证（任务书 §四、§五）

### 5.1 分组（两次运行结果相同）

8 条物料失败 Episode → **1 个组**：`g_outlier_mad_NEGATIVE`，7 名成员、跨 6 条序列
（T14, T141, T143, T144, T146, T15），**12 条匹配的成功对照**，1 条孤例失败
（`impute_ema` on T143，−0.516516）**留在 Episode 里没有被塞进组**。

**分组依据**（不是"都失败了"，也不是数据集名）：

- **先过可比性**：Task×Consumer×metric 键唯一（`forecast|ridge|sMASE`）、domain 唯一。
- **基键**：完整 typed workflow 指纹 × response sign（复用 `group_fault.group_first_faults`）。
- **随组记录**：症状分类、参数变体数、可见 Pattern 展布（12 个特征的 min/median/max），
  以及按**机械证据**屏蔽后的可选错误类——本组唯一可选类是 `SCOPE_MEMORY_RISK_ERROR`
  （"Support 正 + delayed 负"两侧都已实测），其余四类因无机械证据被屏蔽。
- **成功对照随组保留**：12 条同指纹的 POSITIVE Episode。

组内症状**混合**，报告不隐藏这一点：3 名成员是 `SUPPORT_POSITIVE_DELAYED_NEGATIVE`
（部署后在后续窗口翻负），4 名是 `SUPPORT_NEGATIVE` 且 delayed 记 **UNKNOWN**——
它们是**被探测但未被部署**的候选，按设计不开 delayed，所以 UNKNOWN 是真的未测，
**没有被填成 0**。这也是验证 L3 选择读 Support 面的原因（delayed 驱动的只有 3/7，不过半）。

编译/执行错误与真实效果差**分开记录**：verifier 拒绝写在 probe 行的
`kind: verifier_rejected`（不计合法 receipt），仪器不可读写在 `unreadable_series`，
两次运行都为空；真实负增益单独成行。

### 5.2 Slow：run1 弃权，而且那次是我的证据包有缺陷

run1 的真实 Slow 返回**弃权**，理由码 `insufficient_public_evidence`（schema 三个合法弃权
码之一），1 次调用。按本包的重试规则，**弃权成立、不重试**——只有协议/schema 故障才允许
一次换新会话重试，"提议内容被 Runtime 拒绝"和"Slow 弃权"都不重试，否则就是在钓正号。

但复核工件后发现：**那次弃权很可能是我给的证据包造成的。**
`group_fault.build_contrast_capsule` 返回的匹配成功案例形如
`{episode_id, provenance, origin, support_gain}`——**没有序列身份、没有程序、没有 delayed 面、
没有可见特征**；而可见 Pattern 展布我只算了**失败一侧**。也就是说：成功对照**被保留了**，
却被剥掉了正是"使它成为对照"的那部分。要写出一个**条件**，模型必须能把失败与成功放在
同一组特征上比；它当时没有这个东西，于是弃权是合理的。

**这是我的实现缺陷，不是机制读数。** 已修：`_enrich_contrast` 给每条对照补上序列 UID、
程序、两面增益、症状与公开特征卡；卡片改为**两侧同表**呈现可见 Pattern 展布，并附一句
如何读它（范围重叠的特征分不开任何东西，范围不相交的才是候选条件）。
然后**整包重跑一次**（run2）。两次运行都保留、都报告；重跑的理由在看到 run2 结果**之前**
就已写定，是"证据包缺陷"，不是"结果不好看"。

### 5.3 run2：Slow 真的提出了指导，验证把它拒了

**提议**（1 次调用，真实 `TTHASlowAgent.propose_edit`，schema `slow_edit_v1`）：
ADD 一张 capability 条目 `guidance_avoid_outlier_mad_high_missing`，body 是**散文指导**
（不含 `Frozen program steps:`，不声明候选供给权，不带 serving scope——三项都由
`guidance_preflight` 把关，越权会 fail closed）：

> "When a forecast series shows both substantial missingness (missing_fraction around 0.45
> or above) and a strong localized robust-z peak, be cautious before applying MAD-based
> outlier clipping to the whole flagged region … consider whether addressing missing data
> first would be more appropriate."

适用条件：`task_kind == forecast ∧ missing_fraction ≥ 0.45 ∧ local_robust_z_peak ≥ 5.0`。
预测行为改变：`retrieve_skill:guidance_avoid_outlier_mad_high_missing`。
它经**真实** `EditController.apply_to_fork`（cause `SKILL_LIBRARY_GAP`、
surface `skill_library.entries/{skill_id}`、前置 `ABSENT`）**编译成功**
（parent `98dea3b0…` → candidate `441d6c1b…`）。**没有激活。**

**预注册的三条验证线**（在**未参与构造**的留出单元 u009 上，0 LLM / 2 fits）：

| 线 | 判定 | 读数 |
| --- | --- | --- |
| L1 谓词是一个**条件**（匹配留出人群 ≥1 且 ≤ n−1） | **过** | 10 条中匹配 1 条（T14） |
| L2 它**覆盖自己引用的证据**（>半数失败序列） | **不过** | 6 条失败序列中覆盖 **1 条**（T14） |
| L3 诊断**未被反证**（组内程序在留出窗上未大多翻正） | **过** | 2 条可读，0 条物料翻正（T14 在 u009 上 −0.127428） |

**结论：`UPDATE_REFUSED_BY_VALIDATION`，失败线 L2。** 库**一个字节没变**，
K0 的父卡原样存活；**没有人工补卡，没有为凑正号追加运行**。

### 5.4 为什么写不出条件——机械读数，不是解释

审计的 `q3b_was_there_a_condition_to_write` 把该组的**失败一侧**与**成功一侧**放在同一组
部署可见特征上（两次运行结果相同）：

- 比较了 **12** 个数值特征；**能把两侧分开（范围不相交）的：0 个**；
- 其中 **4** 个在两侧恒为常数（`level_excursion_score`、`estimated_level_offset`、
  `level_region_fraction`、`level_region_end_fraction`）；
- **6 条失败序列里有 4 条（T141, T144, T146, T15）同时出现在成功一侧**——
  同一条序列、同一个程序，在不同窗口给出相反答案。

所以任何写得出的条件，要么覆盖几乎没有（run2 就是这样，被 L2 拒），要么覆盖所有人
（会被 L1 拒）。**这一条与 R4A 的"逐序列增益不是序列的稳定属性"独立地一致**，
而且是在**决策层**而不是相关系数层看到的。

### 5.5 旧 `handle_group_feedback` 没有被原样接回

本包**从未调用** `method.handle_group_feedback`：`run_dev_seq1.py:309` 传的是
`allow_group_slow=False`。那条路径要求"存在**一个**替代程序，在**全部**组员上都
`common_positive`"（`group_fault.find_common_headroom` + `unique_common_positive`），
即"一个程序改善整组"——正是本包要停止复辟的形状。本包验证的对象是任务书写明的那个：
**更新共享指导之后，Fast 为不同序列重新生成的处理方案**。这一次没走到那一步，因为没有
合格更新。

---

## 六、后续段：未开（任务书 §六.3）

没有合格更新 ⇒ 两臂将持有**同一个库** ⇒ 比较测不到任何关于知识的东西。按任务书 §七
"只在……比较失效……处停下受影响部分"，后续段**不跑**，记
`NO_UPDATE_TREATMENT`，`followup.ran = false`。

**没有做替代**：没有只跑一条臂冒充后续段，没有手工补一张"好 Skill"，没有换配置重开。
§四.2 的逐单元轨迹与两个参考读数是本包对"下一批 Fast 做了什么"能给出的全部内容，
它是一条**轨迹**，不是**对照**，不携带任何 treatment。

---

## 七、回答任务书 §八 的五个问题

**1. 是否真正逐序列观察、决策并执行？**
是。两次运行各 30 次决策：每次绑定自己的 UID 与 values（逐值核对）、自己的
observed pattern、自己的公开特征卡（10 张互不相同）、自己的历史、自己的 Fast 会话。
**交付数与决策数零不符（60/60）**。异质赋值真实发生（算子层；唯一一处"参数层差异"
经效果去重后是别名，见 §四.1 勘误）。
identity 是被真正执行的路径，其 0.0 经"空 Scope 与 Static 逐位相等"实测确认。

**2. 哪些失败被分到一起，依据是什么，保留了哪些成功对照？**
8 条物料失败 → 1 组 `g_outlier_mad_NEGATIVE`（7 名成员 / 6 条序列），依据是
**可比性（Task×Consumer×metric 与 domain 唯一）→ 完整 typed workflow 指纹 × response sign
→ 随组记症状、参数变体、可见 Pattern 展布与机械证据屏蔽后的错误类**
（唯一可选类 `SCOPE_MEMORY_RISK_ERROR`）。**保留了 12 条同指纹成功对照**；
1 条孤例失败留在 Episode 中未被塞进组。组内症状混合（3 条"Support 正 → delayed 负"、
4 条"Support 负、delayed 未测"），UNKNOWN 未被填零。

**3. Slow 实际修改了哪条知识，还是没有合格更新？**
**没有合格更新，两次都没有。** run1 是弃权（且我的证据包有缺陷，已定位并修复）；
run2 是**真的提出了**一张指导卡、编译入库、被预注册验证在 L2 拒绝（条件只覆盖 1/6 条
失败序列）。**没有任何卡被 PATCH 或撤销**，K0 父卡逐字段存活，库前后一致。

**4. 后续 Fast 的行为、效用、伤害、覆盖和成本怎样变化？**
**关于知识更新：未测**（比较失效，后续段未开）。能报的是形成段自身的轨迹：
覆盖随窗口下降（10→6→3 / 10→6→2），Support 交付随之从 +0.32 降到 +0.05 / +0.03，
delayed 交付在 +0.07…+0.13 区间、末窗接近 0；delayed 面物料受害序列 2 / 1 / 1 与 2 / 1 / 0；
单元权威门两次运行都只在 u008 通过一次。**对照 cohort 级固定答案，逐序列在 Support 面
六格全胜（均值 +0.131 / +0.127），在 delayed 面只有 +0.0045 / +0.0109，量级即噪声。**
成本：每单元每臂约 48–54 次 LLM、6–12 fits（详见 §八）。

**5. 剩余阻断究竟在观察、提议、执行、更新还是后续适用性？**
按 `AGENTS.md §6` 的阶梯机械判定，两次运行点亮同样两级，**最早的一级是**：

> **Support 成功而后续窗口不成功 → Scope / 过拟合 / 风险**

第二级是 **更新**（没有形成合格更新）。**选择层这次是干净的**：
"已探到更好的候选却没有部署"的决策数 = **0 / 30**（两次运行都是 0）——
这与 DEV-KNOW-1 / DEV-AUTO-1 把首因定位在选择层**不同**，因为那两包的决策单位是
cohort 代表序列。把决策单位改对之后，阻断**上移**到了 Support→delayed 的迁移，
而"更新"这一级的具体形状已在 §5.4 读出：**没有可写的条件**。

---

## 八、成本、传输与边界

| 项 | fits | LLM | 墙钟 |
| --- | ---: | ---: | ---: |
| run1（含内嵌 smoke 6 fits） | 30 | 155 | 290.8 s |
| run2（含内嵌 smoke 6 fits） | 40 | 152 | 273.5 s |
| 参考读数（全部 5 个单元） | 20 | 0 | 3.8 s |
| 前置试跑 5 次（含各自 smoke） | 82 | 140 | ≈384 s |
| 独立 smoke 3 次 + Slow 单点探针 | 18 | 2 | ≈60 s |
| **合计** | **190** | **449** | **≈19 分钟** |

单包上限 2000 fits / 6 小时；每次正式运行各只用到 24 / 34 fits（占包级上限 1.2% / 1.7%）与
155 / 152 次调用。**没有任何停止条件被触发**，`stopped_at` 两次都是 `null`。
逐序列会话的 LLM 帽 = 12，是**跑飞守卫**不是流程帽（一次正常流程是 3 个 stage 加至多一次
schema 重试）；实测每条序列 4–7 次。

传输：请求 `deepseek-chat` @ `https://api.deepseek.com/v1`（`M0_AGENT_*`），
**返回模型两次都是 `deepseek-v4-flash`**（与 DEV-KNOW-1 正文同样的别名路由，如实记录）。
用量：run1 692,075 prompt / 13,222 completion tokens；run2 682,919 / 13,463。

准入策略沿用 `bounded_risk_v1`（0.20 / 0.30）以与前几包连续；**n=1 人群下它与 strict
判定完全一致**，阈值改动数 = 0。SHA 新增数 = 0（未建任何新的哈希/manifest/ledger 平台）。
未提交 git。检查点逐单元落盘于 `_scratch/dev_seq1/{run1,run2}/`，未清理。

**不主张的东西**：不主张逐序列决策在后续窗口有效（delayed 面差值是噪声量级）；
不主张这一组失败没有条件可写（只主张**当前 12 维部署可见词表**里没有，且失败与成功
共享 4 条序列）；不主张 Slow 不会写知识（run2 它写了，是验证拒了）；
不与任何组级实验的数字并表或相减。

---

## 九、下一项最有信息量的实验（交裁定，未自行推进）

阶梯最早的一级是 **Support → delayed 的迁移**，而 §5.4 说明它**不是**"条件没找准"的问题：
在当前观测词表里失败与成功不可分，且同一条序列在不同窗口给出相反答案。因此：

- **不建议**再加卡片数量、再改保存方式、再做 series 级 Scope 收窄（R4A 已关闭该路线，
  本包在决策层独立复现了它的前提）。
- **值得打开的**是"证书粒度"：本包的决策与计分都在**单 origin 单序列**上，而这正是
  R4A 测得重测相关≈0 的那个量。同一套逐序列机制、把决策的**证据单位**换成
  跨窗口聚合（例如同序列多 origin 的合并读数），再看 Support→delayed 是否还翻号——
  这与 R4A 建议的 0-fit 证书粒度重打分是同一个方向，但本包提供了**带 Agent 的**版本。
- 若要继续测"知识更新是否有用"，需要先有一个**能过 L1+L2** 的条件；在当前词表下
  它不存在，所以下一步应该先解决**观测面**，而不是再跑一次同样的边界。
