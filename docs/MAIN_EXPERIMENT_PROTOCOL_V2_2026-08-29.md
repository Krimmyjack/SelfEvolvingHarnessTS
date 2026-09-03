# 论文级主实验协议 v2（待冻结）

日期：2026-08-29  
状态：`DRAFT_FOR_FREEZE`；本文是可执行协议稿，不是实验结果，也不自动覆盖项目 `AGENTS.md`。  
上游：`AGENTS.md`、`docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md`、
`docs/D4_DOWNLOAD_FREEZE_2026-08-29.md`、`docs/CLS_LINE_FINAL_REPORT_2026-08-28.md`、
`artifacts/functional/e2/s3_pilot_probe_policy.md`、
`artifacts/functional/e2/g1_kdd_trigger_census.md`。  
目的：把已定方法骨架转成论文级主实验协议；补齐主张、四臂、数据角色、Consumer 轴、
课程、重复、指标、判词、消融、密封验收和执行止损条件。

本轮设置收紧（2026-08-29）：

1. roster 先按 Outcome-blind 机械规则固定；难度不再参与选课，只作次级分层；
2. 主判只看固定 roster 的整体 ITT，分层结果不得改变协议；
3. 只借 TimeClaw 的确定性同底与 matched-memory 归因形态；
4. 只借 AEGIS-TS 的“适应成本与最终效用分开计量”，不搬其 reward/模型/窗口。

> 无结果声明：本文没有生成任何新实验结果。所有 `TBD` 项只能由真实运行工件填入。

---

## 0. 当前冻结状态与本协议边界

### 0.1 当前被测 Harness

- 主实验使用已冻结的 `DEFAULT/no_edit` 行为；已有 pilot 工件保留作版本冻结与
  回滚记录。
- 当前“自进化”的主实验对象是 Skill/Memory 的生成、写回、Scope/Risk 修订与
  后续复用。
- 当前方法中的 Task/Consumer/Pattern Context、Typed Workflow、Skill/Memory、
  Scope、权限阶梯、verifier、Support、delayed、harm 否决和 held-out 信息墙保持不变。
- 本协议不授权修改阈值、Scope 语义、权限阶梯、安全门、Operator 实现或 Slow 编辑面。

### 0.2 本协议允许的边际工程

只允许为主实验完成以下接线：

1. 课程 roster 与正反序参数化；
2. 现有 Forecast Consumer 的统一 adapter 参数化；
3. 将现有 imputation Workflow 合法接入 Forecast 菜单的兼容性检查；
4. 课程级预算、时间与机制漏斗记账；
5. 结果聚合器、checkpoint/resume 和聚焦测试。

不得借主实验之名扩建新平台、增加新 Operator、调材料阈值或修全仓历史测试债。
若 KDD 现有 Workflow 不可用，只能走本文预声明的单一 fallback。

---

## 1. 中心问题与统一方法原则

中心问题：

> 在不同 Task、Consumer 和 Pattern 下，同一 Harness 能否利用各任务自己的历史经验，
> 在相同 Target 反馈预算下更快、更安全地达到正确的数据就绪决策，并根据后续
> Gain/Harm 持续修订自己的 Skill。

统一机制：

```text
Task + Consumer + Pattern Context
→ 检索本 Task 内 Scope 匹配的历史 Skill
→ Skill 只供应一个待验证候选，不直接执行或部署
→ Agent 保留自主探索候选
→ 当前 Target 的 Support / delayed 反馈批准或拒绝
→ 形成 Target-local Skill / Episode
→ A5 在单元边界写回并修订 Skill
```

跨 Task 不要求动作级正迁移：

```text
Forecast 经验只服务 Forecast
Classification 经验只服务 Classification
AD 经验只服务 AD
跨 Task 只验证隔离、沉默和无负迁移
```

---

## 2. Claim–Evidence 矩阵

| Claim | Reviewer question | 必要证据 | 数据/任务 | 对照 | 主指标 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 Task/Consumer/Pattern 条件化 | Harness 是否真的因 Context 改变行为 | 同数据同菜单、只换 Consumer 的整机闭环；Consumer-blind 消融 | Forecast 双 Consumer；Classification 既有 ridge/kNN | 完整 A5 vs Consumer-blind | utility、regret、Workflow/Skill 差异、harm | planned |
| C2 Target-local 适应有效 | 下游反馈是否比不适应更好 | 同 Target、同预算 A3 vs Static | Forecast 主课程 | A3−Static | utility、regret、fits、harm | planned |
| C3 固定历史经验有价值 | 历史 Skill 是否减少重复搜索 | K0 与 A5 同起点；K0 不写回 | Forecast 主课程 | K0−A3 | utility、regret、time-to-threshold | planned；既有证据混合 |
| C4 在线 Skill 修订有增量 | Gain/Harm 写回是否优于固定 Skill | 实际修订触发、后续重遇、A5 与 K0 行为分叉 | Forecast natural-heterogeneous cells；Classification SA-1 | A5−K0 | regret、fits、版本链、重复错误 | planned；Classification 行为证据已有 |
| C5 完整经验系统有付酬 | A5 是否优于冷启动 A3 | 四臂同课程、同反馈预算、多个采样重复 | Forecast 主课程 | A5−A3 | 累计 regret、utility、适应成本、harm | 主承重，planned |
| C6 条件化与安全 | 不匹配/有害经验是否保持沉默或被否决 | task mismatch、Scope mismatch、harm/identity 场 | Forecast、AD、Solar-natural | A5 vs Static/A3 | wrong promotion、abstain、harm、zero supply | 部分已有；主实验复核 |
| C7 fresh 新域泛化 | 冻结 Harness 在未见 Outcome 上是否仍有效 | F1 同族新 Target + F2 新族密封 Target，一次开封 | traffic leftover、Solar | Static/A3/K0/A5 | utility、regret、harm、成本 | planned |

主实验的 primary claim 是 C5；C4 是在线 evolution 的严格增量主张；C1/C6 是方法成立所需的
条件化和安全证据。

---

## 3. 四个主实验臂与归因

| 臂 | 单元内 held-in 适应 | 课前历史 Skill | 单元间写回/修订 | 归因用途 |
| --- | ---: | ---: | ---: | --- |
| `Static` | 否 | 否 | 否 | identity/无适应基线 |
| `A3-reset` | 是 | 否 | 否 | Target-local 冷启动适应 |
| `K0-fixed` | 是 | 有，且与 A5 字节一致 | 否 | 固定历史知识 |
| `A5-online` | 是 | 与 K0 字节一致 | 是 | 完整 Self-Evolving Harness |

冻结比较：

```text
A3 − Static = Target-local 反馈适应价值
K0 − A3     = 固定历史经验价值
A5 − K0     = 在线写回与 Skill 修订价值
A5 − A3     = 完整累积经验系统价值
A5 − Static = 端到端完整系统价值
```

公平性硬约束：

1. K0-fixed 与 A5-online 的课前 Store、Skill 内容、Scope、版本和 SHA 必须一致。
2. 两者唯一行为差异是：A5 可以在单元边界写回/修订，K0 不可以。
3. A3 在每个新 Target 从公共 h0 开始，不携带历史 Experience/Skill。
4. 三个适应臂具有相同的 Target Support/delayed 预算。
5. A5 的 Slow 调用、版本化、额外 LLM 和 fit 成本全部计入，不得免费。
6. Source-derived Skill 最多供应一个候选；identity 始终可用；自主探索槽必须保留。
7. 分类/AD Skill 可以物理存在于 Store 中以检查隔离，但在 Forecast 中必须由 Task Scope
   fail-closed。

### 3.1 K0 构成规则

K0 只允许包含：

- 主实验 Target roster 之外、已曝光 development Source 上产生的合法 Forecast Skill；
- 机器可执行的 Task × Consumer × Pattern × Program geometry Scope；
- `supply-only` 或更低权限；
- 完整 provenance、内容 SHA 和版本；
- 不包含 raw Episode、main Target Outcome 或 sealed Target 信息。

每个 Forecast Consumer 至少需要一张 Consumer-compatible K0 Skill，才能对该 Consumer
检验 C3/C5。若不存在，不得复制另一个 Consumer 的卡；该 Consumer 的 C3/C5 记为
`SOURCE_K0_UNAVAILABLE`，仍可检验 C1/C2/C6。

### 3.2 必要的参考实验 setting（只借协议，不搬方法）

本协议只借鉴 TimeClaw 与 AEGIS-TS 中三项对公平性承重的设置，不新增它们的模型、
奖励、数据集或检索系统。

#### TimeClaw setting 1：确定性同底切分

- 固定 `MAIN_SPLIT_SEED = 20260829`；它只控制 Source/Target roster 切分，不冒充
  LLM sampling seed 或注入 seed。
- 同一 Task family 内的 Source/Target 名单在 live 运行前一次冻结。
- Static/A3/K0/A5 使用完全相同的 Target、顺序、Consumer、反馈预算和 held-out。
- 任何臂失败或弃权都不得改变其他臂的 Target roster。

这对应 TimeClaw 的固定 split seed 与同一 test records 上 `k=0/k=3` 配对比较；本项目
不复制其 raw trajectory prompt 注入。

#### TimeClaw setting 2：matched memory ablation

TimeClaw 的“无记忆 vs 有记忆”形态在本项目中由现有四臂更细地实现：

```text
A3 = 无历史经验
K0 = 固定历史经验
A5 = 相同初始经验 + 在线写回/修订
```

因此不新增 TimeClaw 检索器或 `k=3` 新臂；只借“同一测试集，只改变记忆状态”的归因
纪律。K0/A5 起点 SHA 不一致时，memory ablation 无效并立即停跑。

#### AEGIS-TS setting：适应成本与最终下游效用分开计量

沿用本项目正典而不复制 AEGIS-TS 的 proxy reward：

```text
held-in Support/delayed → 负责适应、产生 Gain/Harm 与成本
freeze
held-out final Consumer → 只负责一次性最终效用计分
```

最终 utility/regret 与 Consumer fits、fit wall-clock、time-to-threshold、LLM/probe
必须分栏报告；不得把数据缺陷率下降、成本与下游效用混成一个 reward 或总分。

明确不借：TimeClaw 的 ground-truth-reveal train mode、raw trajectory 注入、同 family
exemplar 保证；AEGIS-TS 的窗口/预测长度、DLinear/LSTM 配置、清洗率奖励、层级 RL 和
固定清洗策略。参考实现只读位置为 `../../TimeClaw/` 与 `../../a-evolve/AegisTS/`。

---

## 4. Task 与数据角色

### 4.1 Forecast：主承重任务

Forecast 主评测使用**固定异质 roster**，不以“易/中/难”作为选数据或保留数据的条件。
课程由三类预声明数据角色构成：

| 数据角色 | 固定来源 | 为什么纳入 | 主分析地位 |
| --- | --- | --- | --- |
| controlled development | 已曝光的 impulse/outlier cell，共 2 个 | 机制正控：合法 Workflow 是否能被发现和验证 | ITT，不能替代自然能力 |
| natural control | 已曝光的自然/clean dev cell，共 2 个 | identity/abstain 与不乱修 | ITT |
| natural heterogeneous | KDD with-missing 文件序 cell，共 4 个 | 自然缺失异质、CONFLICT/修订/re-encounter 可测性 | ITT，development only |

全部 8 个固定单元进入 primary ITT；不得因某个单元“太容易”“太难”、无 headroom、
A5 未获益或 oracle 不理想而删除、替换或降权。

#### 难度只作 Outcome-blind 次级分层

难度不参与 roster 选择，也不是主判分条件。若论文需要解释经验在什么条件下付酬，
只允许使用在 Outcome 打开前冻结的部署可见或协议级变量进行次级分层：

- 合法候选空间大小；
- 缺失率、缺口游程、跨序列异质性等公开 Pattern；
- Support/delayed 样本量与材料分辨率；
- controlled 条件的预声明注入强度；
- 由独立 development calibration 得到的冷发现调用数分布。

禁止用于定义难度或选课：

- held-out utility/headroom；
- oracle 最优增益；
- A5−A3、A5−K0 或任一臂运行结果；
- 运行后才调整的阈值；
- “哪个单元能让 A5 赢”的人工判断。

优先把难度变量作为连续量报告；如需 low/medium/high 三档，分档阈值必须由独立
development calibration 在主 roster 结果可见前冻结，然后机械套用到所有单元。
主结论仍基于全 roster ITT，分层只解释异质性。F1/F2 开封前不预判难度；开封后只能按
既有阈值作描述，不能改变协议或追加 Target。

#### KDD 固定角色

- 数据：`data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip`；md5 已核。
- development only，永不称 fresh。
- 文件序切分：`kdd_missing_00..03`，每 cell 60 条；spare 30 不凑格。
- G1 当前 `15.00/课` 只是缺陷几何代理；尚未由真实 Consumer 验证为
  POSITIVE/NEGATIVE/CONFLICT Episode。
- 发车前必须完成 §9.1 的现有 imputation Workflow 与真实 Consumer 可达性检查。

### 4.2 Classification：Skill 生命周期与修订腿

不重新进行大规模课程搜索，直接纳入已完成的 SA-1 两次固定协议结果：

- 单例 supply-only Skill 初始化；
- 正反馈追加证据；
- CONFLICT 后机械收窄 Scope；
- 再遇时避免一次重复拒绝；
- 两跑前四版内容 SHA 与顺序一致。

该证据支持“Specific Skill 可学习、可修订、可回滚”；当前只支持行为/成本改善，
不支持“Skill 修订稳定提高最终质量”。若最终代码冻结清单无法证明 DEFAULT 行为等价，
才允许一次固定协议 replay；否则不重跑。

### 4.3 AD：安全与正确弃权腿

AD 不承担跨 Task 正向动作迁移。它只验证：

- Forecast/Classification 卡在 AD 中保持沉默；
- 面对事件型异常时拒绝错误清洗；
- 无 headroom 时部署 identity/abstain；
- wrong promotion = 0；harm 不高于 Static。

最终可使用 Yahoo 剩余 sealed 41 条一次性验收；前 24 条只作 development。

---

## 5. Forecast Consumer 轴

至少使用两个 Consumer：

1. `pooled_ridge`：跨序列共享参数；
2. `per-series/per-channel ridge`：按序列/通道独立拟合。

P0 冻结其实际 contract ID；若仓内名称不同，以现有合同为准，不新造 Consumer。

两 Consumer 必须保持：

- 同一数据 roster 与 split；
- 同一窗口、horizon、metric 和 Workflow 菜单；
- 同一反馈预算；
- 相同 held-out evaluator；
- 只改变 Consumer 的归纳偏置与合法公开 Context。

成功不要求两个 Consumer 选择相同 Workflow。C1 的正确读法是：

> Harness 能针对每个 Consumer 形成安全且有用的决策；不同 Consumer 的 Gain/Harm
> 可以导致不同 Workflow、Skill 或 abstain。

### Consumer-blind 消融

真实 Consumer 仍负责评分，但 Agent Observation 与 Skill Scope 看不到 Consumer 信息。
它同时切断 Observation 可见性和 Scope 条件化，归因只到“整机 Consumer 条件化”，
不拆成两个子模块。

---

## 6. Forecast 连续多域评测序列（主课程）

### 6.1 roster 选择规则

正式 roster 在 P0 写入 `main_course_roster.json`，按以下机械规则选择：

1. 使用 `MAIN_SPLIT_SEED=20260829` 与预声明 family key 一次性确定 Source/Target 角色；
2. 排除 Solar、Yahoo sealed、Stage 3 直接触碰单元和任何 fresh held-out Outcome；
3. controlled dev pool 按数据源与 cell ID 字典序取前两个合格且互异的 cell；
4. natural-control dev pool 按数据源与 cell ID 字典序取前两个合格且互异的 cell；
5. KDD 四个 full cell 按文件序全部纳入，不挑选、不重排内容；
6. roster 选择不读取 oracle、held-out、A5/A3/K0 结果或难度标签；
7. 若任一预声明角色不足，停止并报告 `COURSE_ROSTER_INSUFFICIENT`，不得按结果追靶；
8. roster 对四臂、两个 Consumer 和全部重复完全一致，所有单元进入 ITT。

### 6.2 八单元模板

| 位置 | 冻结角色 | 数据槽 |
| ---: | --- | --- |
| U1 | controlled development A | `[CONTROLLED_A]` |
| U2 | natural control A | `[NATURAL_CONTROL_A]` |
| U3 | natural heterogeneous | `kdd_missing_00` |
| U4 | controlled development B | `[CONTROLLED_B]` |
| U5 | natural heterogeneous | `kdd_missing_01` |
| U6 | natural heterogeneous / possible re-encounter | `kdd_missing_02` |
| U7 | natural control/guard B | `[NATURAL_CONTROL_B]` |
| U8 | natural heterogeneous | `kdd_missing_03` |

Forward 使用 U1→U8；Reverse 必须使用精确逆序 U8→U1。任何具体 cell 名在首次 live
结果可见前填入并提交；结果可见后不得换课。难度标签不写入 runner 路由，也不改变
候选、预算、检索或计分。

### 6.3 单元内协议

所有臂的完整顺序：

```text
1. 构建公开 Observation
2. 按 Task × Consumer × Pattern 检索 Skill
3. Agent 生成自主候选
4. 历史 Skill 至多供应一个候选
5. verifier 检查合法性与修改帽
6. 在现行 SUPPORT_TRIAL_BUDGET=2 内 probe
7. Support 分类 POSITIVE / NEGATIVE / CONFLICT / ABSTAIN
8. held-in delayed 独立复核
9. 冻结 Target-local 部署
10. held-out Fast-only 一次执行
11. 外部 evaluator 在所有臂冻结后统一开分
12. 写 Episode；仅 A5 在单元边界编译/修订 Skill
```

不得改变当前 DEFAULT probe policy。Skill 候选没有额外反馈预算，未获 probe 是机制真实行为，
必须进入 ITT 结果与机制漏斗。

### 6.4 split 与材料线

沿用现役 Forecast cell 合同：

```text
CELL_WIDTH = 60
N_TRAIN = 40 = Support 20 + delayed 20
N_HELDOUT = 20
ORIGIN_HELDIN = 1104
ORIGIN_HELDOUT = 1800
HORIZON = 48
MATERIAL = max(0.005, 1 / n_half)
HARM_BAR = 0.005
```

若 Consumer adapter 的实际合同与此不兼容，只允许在 P0 报告并停止；不得运行中改 split。

---

## 7. 重复、顺序与统计

### 7.1 固定运行矩阵

| Consumer | 课程方向 | 采样重复数 | 角色 |
| --- | --- | ---: | --- |
| pooled ridge | Forward | 5 | primary |
| pooled ridge | Reverse | 3 | order robustness |
| per-series/per-channel ridge | Forward | 3 | Consumer generalization |

共 11 个课程运行。每个课程包含 Static/A3/K0/A5 四臂。

纪律：

- Source/Target 切分只使用冻结的 `MAIN_SPLIT_SEED=20260829`；四臂共享完全相同的
  Target records。这是 roster seed，不控制 LLM sampling。
- A3/K0/A5 的 matched-memory 对比只允许改变历史状态与写回权限，不得改变测试集、
  Consumer、Prompt、工具、预算或 held-out。
- 注入没有随机 seed，不得称“注入 seed”；重复是 LLM sampling replicate。
- 使用同一 returned model、Prompt、菜单、数据、阈值、预算和 Harness inventory。
- checkpoint/resume 属同一运行，不算新重复。
- `BACKEND_UNAVAILABLE`、墙钟中断或工件损坏不写科学判词。
- 第一条 primary course 若 Treatment 不可观测，按 §10 停止剩余重复。

### 7.2 统计口径

- primary comparison：`A5 − A3`；其余比较为 supporting。
- 报告每个 course replicate 的配对结果，不只报均值。
- 对 pooled primary 使用按 course replicate 聚类的 paired bootstrap CI；同时报告
  win/tie/loss 和中位数。
- Reverse 与第二 Consumer 主要作方向性 robustness；样本不足时不包装成显著性结论。
- 不因 p-value 不显著而忽略材料级差值，也不以一次正向 sampling replicate 宣称复现。

---

## 8. 指标与预注册判分

### 8.1 质量与安全

1. cumulative held-out regret（越低越好）；
2. held-out utility gain / sMASE gain（越高越好）；
3. worst-series gain；
4. material harm event 数与累计 harm；
5. wrong promotion；
6. correct abstention / identity；
7. task/Scope mismatch 下的错误 supply 数。

### 8.2 适应效率

1. 到首个安全有效 Skill 的 Consumer fits；
2. Support probe 数；
3. time-to-threshold；
4. Consumer fit wall-clock；
5. LLM 调用数；
6. A5 Slow/修订额外开销。

LLM、fit、probe、wall-clock 分项报告；禁止合成任意总分。

### 8.3 机制漏斗

每个单元至少记录：

```text
Episode written
→ Skill compiled/versioned
→ Scope retrieved/matched
→ candidate supplied
→ verifier passed
→ candidate probed
→ Support result
→ delayed result
→ Target-local deployment
→ later revision/reuse
```

### 8.4 材料判分

定义正向差值：

```text
Δutility(A,B) = utility_A − utility_B
Δregret_reduction(A,B) = regret_B − regret_A
```

质量非劣：

```text
mean per-unit Δutility >= -0.005
AND worst-series Δ >= -0.005
AND A 的 harm / wrong promotion 不高于 B
```

质量材料改善：

```text
mean per-unit Δutility >= +0.005
OR mean per-unit Δregret_reduction >= +0.005
```

效率材料改善不使用综合分。只有在质量与 harm 非劣的前提下，满足以下任一项才写
“降低适应成本”：

- 主要重复的中位数每 Target 至少节省 1 次 Consumer fit；
- time-to-threshold 在至少 4/5 pooled-forward primary replicates 中更短。

### 8.5 Claim 判词

| 判词 | 必要条件 | 可写主张 |
| --- | --- | --- |
| `ACCUMULATION_QUALITY_POSITIVE` | A5 vs A3 质量材料改善，harm 非劣 | 累积经验提高后续效用 |
| `ACCUMULATION_EFFICIENCY_POSITIVE` | A5 vs A3 质量非劣且效率材料改善 | 累积经验降低适应成本 |
| `FIXED_PRIOR_POSITIVE` | K0 vs A3 过质量或效率门 | 固定历史 Skill 有付酬 |
| `ONLINE_EVOLUTION_POSITIVE` | A5 vs K0 过质量或效率门，且真实修订影响后续单元 | 在线写回/修订有增量 |
| `C5B_UNEXERCISED` | 无实际修订触发或修订未进入后续行为 | 本课程没有考到在线修订 |
| `NO_ONLINE_EVOLUTION_ADVANTAGE` | 修订被实际考到但 A5 不优于 K0 | 在线修订在本协议下无增量 |
| `SAFE_NONINFERIOR` | A5 质量非劣、harm=0/非增，但无成本或效用优势 | 经验安全但未付酬 |
| `TREATMENT_EMPTY` | A5/K0 历史知识从未有合法机会影响行为 | 四臂效应不可解释，停止重复 |
| `NEGATIVE_TRANSFER` | A5 的历史知识实际影响行为并造成材料级质量下降/额外 harm | 可信负迁移 |

`ONLINE_EVOLUTION_POSITIVE` 必须同时满足：

1. 至少一次真实 Gain/Harm 触发 Skill 版本更新；
2. 更新后的版本在后续单元重新检索；
3. 它改变 supply/probe/abstain/deployment 中至少一项；
4. A5 相对 K0 过质量或效率材料门；
5. harm 不增加。

---

## 9. 发车前 P0 硬门

### 9.1 P0-A：KDD 现有 Workflow 可达性

仓内已有 `impute_linear/impute_ema/impute_ar/impute_fft/impute_ssm` 等 Operator；
P0 只检查它们是否能通过当前 Forecast Workflow/Binding/Consumer 合同，不新增 Operator。

对四个 KDD full cell × 两个 Consumer 全量报告：

```text
operator legal
→ verifier pass
→ Support
→ delayed
→ per-series harm/conflict
```

KDD 是 development，允许真实 Consumer fit。P0 必须全表报告，不能只留有利算子。

准入：至少一个现有 imputation Workflow 在不少于两个 cell 上具有合法可读的
POSITIVE/CONFLICT/NEGATIVE 反馈，并能产生后续 revision/re-encounter 考题。

若不满足，只允许切换到预声明 NOAA development fallback；若 NOAA 也不满足，
natural-heterogeneous 角色和 C4/C5b 记 `PROGRAM_HEADROOM_UNAVAILABLE`，不得新增
imputer 或修改阈值。

### 9.2 P0-B：真实修订触发率

`g1_kdd_trigger_census` 的 15.00/课是结构代理，不是 Episode。正式发车前须用真实
Consumer 反馈验证课程预计至少存在：

- 一次 POSITIVE；
- 一次 CONFLICT/NEGATIVE；
- 一次相似 Context 的后续 re-encounter。

若无法形成“反馈→修订→再遇”，C5b 不可测；不得把 proxy 写成实证。

### 9.3 P0-C：Consumer adapter

聚焦测试必须覆盖：

- 两 Consumer 同 roster/split/window/horizon/metric；
- Consumer ID 正确进入 TaskContext 和 Scope；
- 不存在 Runner 按 Consumer ID 指定 Workflow；
- held-out Outcome 在 freeze 前不可读；
- Consumer-blind 只隐藏 Context，不改变真实 evaluator。

### 9.4 P0-D：K0 与冻结清单

发车前落盘：

```text
code commit
Harness inventory SHA
h0 lock
K0/A5 store content SHA（必须相同）
K0 Skill IDs / versions / provenance / Scope
course roster and forward/reverse order
consumer contracts
data exposure ledger
menu / budgets / thresholds
model / backend route
result templates
```

只要求主路径聚焦测试全绿；全仓 pytest 的既有失败不得阻塞主实验，也不得在本书内修复。

---

## 10. 运行止损与自动分支

### 10.1 仪器停止

以下情况不写科学判词，修复同一协议后 resume/re-run：

- backend unavailable；
- checkpoint 损坏或记录缺项；
- K0/A5 起点不一致；
- held-out 泄漏；
- Consumer/split/menu 配置漂移；
- returned model 不一致。

### 10.2 Treatment 前置门

首个 pooled-forward primary course 后检查：

```text
K0/A5 Skill 是否检索可见
→ 是否有 Scope match
→ 是否 supply/probe 或明确被 default policy 跳过
→ A5 是否发生至少一次合法写回/修订
```

- 若历史 Skill 从未拥有合法影响机会：`TREATMENT_EMPTY`，停止其余 10 个课程运行。
- 若 Skill 有影响但没有收益：继续固定协议，形成可信效用结论。
- 不得根据第一跑的正负换课程、加预算、改 Scope 或阈值。

### 10.3 安全红灯

若出现以下任一项，停止 sealed 开封，先做 first-fault：

- historical Skill 绕过 Support/delayed 直接部署；
- task mismatch 卡进入执行；
- wrong promotion 或 harm 超出冻结门；
- held-out 反馈回流本次 Harness。

---

## 11. Development-only 基线与消融

四臂是主比较。以下只在预冻结的代表性三单元 dev 子集运行，不进入 sealed 数据：

### 11.1 次级基线

| 基线 | 定义 | 回答的问题 |
| --- | --- | --- |
| `OneShot-LLM` | 同 Observation/菜单，一次合法提案，不使用 Support/delayed，不写回 | 直接让 LLM 决策是否可靠 |
| `BudgetMatched-FixedSearch` | 固定候选次序、相同 probe/fit 预算、无 LLM/Memory | 收益是否只是菜单搜索 |
| `Oracle-menu` | 外部 evaluator 事后遍历合法菜单 | 只作 headroom/regret 上界，不是部署臂 |

### 11.2 消融

| 变体 | 改变 | 机制问题 | 边界 |
| --- | --- | --- | --- |
| Consumer-blind | 隐藏 Consumer Context | Consumer 条件化是否必要 | 归因到整机 |
| No-Scope/global retrieval | Task 内历史卡不做 Pattern Scope 匹配 | Scope 是否防错误复用 | dev only |
| Support-only | 移除 delayed 确认 | delayed 是否防短期误导 | evaluation-only，不进生产 |
| OneShot-LLM/no feedback | 移除全部下游反馈 | 真实 Consumer feedback 是否必要 | dev only |

`A3-reset` 已是 no-history 消融，`K0-fixed` 已是 no-online-writeback 消融，不再造重复臂。

---

## 12. Fresh 与密封验收

### 12.1 F1：traffic leftover

角色：同族新 Target、Outcome 未见；不是新 family。

- 使用冻结的 traffic leftover roster；
- Static/A3/K0/A5 四臂同场；
- 与 development 使用相同 Consumer、菜单、预算与阈值；
- 一次性 held-in adaptation→freeze→held-out open；
- 不重跑、不换卡、不根据结果追第二 Target。

主读数：同族新实例上的 A5 vs A3/K0、正确 identity、额外成本和 harm。

### 12.2 F2：Solar 10 Minutes

角色：Outcome 未见的新 family capstone；当前保持隔离。

在读取数值前冻结两个 disjoint 条件：

1. `Solar-Natural`：自然数据主考；
2. `Solar-Controlled`：预声明训练侧 impulse/outlier 注入，仅作机制正控。

建议机械分组：

```text
偶数序列索引 → Natural cell（前 40 held-in，其余 held-out）
奇数序列索引 → Controlled cell（前 40 held-in，其余 held-out）
```

全部 137 条纳入，不依据内容挑选；controlled corruption 的位置、强度、作用面必须由 dev
协议预先冻结，只污染 held-in 可准备数据，不污染最终 held-out Outcome。

Static/A3/K0/A5 一次性同场。所有臂冻结后统一打开 Outcome；开后不改、不重跑。

解释：

- Natural positive：支持自然新域能力；
- Natural neutral + Controlled positive：只支持新底物机制泛化与自然安全；
- Natural/Controlled 均 neutral 且正确 abstain：支持安全泛化，不支持经验付酬；
- Controlled positive 不能替代自然数据能力证据。

### 12.3 AD sealed safety

Yahoo 剩余 sealed 41 条只运行一次。四臂的主要正确答案可以是 identity/abstain；成功门是：

- cross-task Skill retrieval/supply = 0；
- wrong promotion = 0；
- harm 不高于 Static；
- 不因“必须有正收益”而新增清洗动作。

---

## 13. 结果表模板

### 13.1 Forecast 主表

Primary 表先报固定 roster 的整体 ITT，不按难度筛选：

| Consumer | Setting | Arm | Utility gain ↑ | Regret ↓ | Fits ↓ | Time-to-threshold ↓ | LLM ↓ | Harm ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled | all fixed cells (ITT) | Static | TBD | TBD | TBD | TBD | TBD | TBD |
| pooled | all fixed cells (ITT) | A3 | TBD | TBD | TBD | TBD | TBD | TBD |
| pooled | all fixed cells (ITT) | K0 | TBD | TBD | TBD | TBD | TBD | TBD |
| pooled | all fixed cells (ITT) | A5 | TBD | TBD | TBD | TBD | TBD | TBD |
| per-series | all fixed cells (ITT) | Static/A3/K0/A5 | TBD | TBD | TBD | TBD | TBD | TBD |

难度/复杂度只进入次级解释表，分层阈值须有预冻结 ID：

| Consumer | Outcome-blind stratum/proxy | n cells | Arm | Utility gain ↑ | Regret ↓ | Fits ↓ | Harm ↓ |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| pooled | low / continuous-bin-1 | TBD | Static/A3/K0/A5 | TBD | TBD | TBD | TBD |
| pooled | medium / continuous-bin-2 | TBD | Static/A3/K0/A5 | TBD | TBD | TBD | TBD |
| pooled | high / continuous-bin-3 | TBD | Static/A3/K0/A5 | TBD | TBD | TBD | TBD |

### 13.2 归因差值表

| 对比 | 含义 | ΔUtility ↑ | ΔRegret reduction ↑ | ΔFits ↑ | ΔTime ↑ | ΔHarm ↓ | 判词 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A3−Static | Target-local 适应 | TBD | TBD | TBD | TBD | TBD | TBD |
| K0−A3 | 固定历史经验 | TBD | TBD | TBD | TBD | TBD | TBD |
| A5−K0 | 在线进化 | TBD | TBD | TBD | TBD | TBD | TBD |
| A5−A3 | 完整经验价值 | TBD | TBD | TBD | TBD | TBD | TBD |
| A5−Static | 端到端系统 | TBD | TBD | TBD | TBD | TBD | TBD |

### 13.3 Skill 演化表

| Skill/version | Trigger | Scope/authority change | Later match | Behavior change | Target result |
| --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD |

### 13.4 Fresh/安全表

| Dataset | Freshness | Condition | Static | A3 | K0 | A5 | Harm | Claim grade |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| traffic leftover | F1 same-family | natural | TBD | TBD | TBD | TBD | TBD | TBD |
| Solar | F2 new-family | natural | TBD | TBD | TBD | TBD | TBD | TBD |
| Solar | F2 new-family | controlled | TBD | TBD | TBD | TBD | TBD | TBD |
| Yahoo | sealed safety | natural | TBD | TBD | TBD | TBD | TBD | TBD |

### 13.5 消融表

| Variant | Mechanism tested | Utility ↑ | Regret ↓ | Fits ↓ | Harm ↓ | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Full A5 | full method | TBD | TBD | TBD | TBD | TBD |
| Consumer-blind | consumer conditioning | TBD | TBD | TBD | TBD | TBD |
| No-Scope | scoped reuse | TBD | TBD | TBD | TBD | TBD |
| Support-only | delayed validation | TBD | TBD | TBD | TBD | TBD |
| OneShot-LLM | downstream feedback | TBD | TBD | TBD | TBD | TBD |

---

## 14. 执行优先级与交付物

| 优先级 | 工作 | Claim | 成本 | 依赖 | 停止条件 | 交付 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | KDD Workflow/Consumer 可达性、真实触发率、K0/课程冻结 | C3/C4/C5 | low–medium | 当前 v2 | P0-A/B/C/D 任一不过 | protocol freeze + preflight report |
| P1 | pooled-forward 首跑 + Treatment 门 | C5 | medium | P0 | TREATMENT_EMPTY/仪器红灯 | first-course report |
| P2 | 完成 pooled 5 forward + 3 reverse | C2–C5 | high | P1 non-empty | 预算/后端硬故障 | primary table |
| P3 | per-series forward ×3 | C1/C5 | high | P0 | Consumer adapter 漂移 | multi-consumer table |
| P4 | 三单元次级基线与消融 | C1/C6 | medium | P2/P3 | 主结果不可解释时优先 | ablation table |
| P5 | F1 traffic one-shot | C5/C7 | low–medium | P2 | sealed/instrument fault | F1 report |
| P6 | Yahoo sealed safety | C6 | low | P2 | safety red light | AD safety report |
| P7 | Solar F2 one-shot | C7 | medium | 全协议冻结 | 仅仪器故障可停 | capstone report |
| P8 | result-to-claim 审计与论文表格 | 全部 | low | P2–P7 | 数字/主张不一致 | claim matrix + paper tables |

建议顺序：

```text
P0 → P1 → P2/P3 → P4 → P5/P6 → P7 → P8
```

P0 到 development 主表预计是主要计算阶段；不得以赶时间为由裁掉 Static/A3/K0/A5 任一臂。
预算不足时裁剪顺序：次级消融 → 额外 robustness → 数据档数；不裁四臂，不动 F2 一次性纪律。

---

## 15. 论文主张边界

允许按结果支持的最强措辞：

- A5 vs A3 在固定 roster ITT 上过门：累积经验降低 regret 或适应成本；
- Outcome-blind 次级分层方向一致：可进一步说明经验收益随哪些部署可见条件变化；
- A5 vs K0 且修订链实际被考到：反馈驱动在线 Skill 修订带来增量；
- Consumer 轴与 blind 消融过门：数据就绪决策确实 Consumer-conditioned；
- Solar-Natural 过门：在未见新 family 上获得端到端自然能力证据；
- AD/Yahoo 安全门通过：Task-specific 历史知识在错误任务中保持隔离并安全弃权。

不得写：

- 分类经验正向帮助 Forecast 或 AD；本项目不作该主张；
- controlled injection 等于自然数据能力；
- 卡被检索/供应等于卡产生收益；
- A5 仅因一次 sampling replicate 胜出就已复现；
- KDD 结构代理等于真实 Consumer CONFLICT；
- Solar 在开封前具备已知 headroom。

---

## 16. 发车前仍待填写的冻结项

| 项 | 当前值 | 负责人/来源 | 发车前要求 |
| --- | --- | --- | --- |
| `MAIN_SPLIT_SEED` | `20260829` | 本协议 | 只控制 roster，不冒充 sampling seed |
| `[CONTROLLED_A]` / `[CONTROLLED_B]` | TBD | dev roster census | 按 §6.1 字典序冻结 |
| `[NATURAL_CONTROL_A]` / `[NATURAL_CONTROL_B]` | TBD | dev roster census | 按 §6.1 字典序冻结 |
| difficulty/complexity proxy thresholds | TBD | independent dev calibration | 仅次级分层，不参与 roster/runner |
| pooled Consumer contract ID | TBD | existing contracts | 聚焦测试后冻结 |
| per-series Consumer contract ID | TBD | existing contracts | 聚焦测试后冻结 |
| K0 Skill IDs/SHA | TBD | audited Forecast source skills | K0/A5 相同 |
| KDD legal imputation Workflow | TBD | P0-A | 不新增 Operator |
| KDD actual revision-trigger rate | TBD | P0-B | 不以 proxy 代替 |
| NOAA fallback roster | TBD | existing dev roster | 只在 KDD P0-A 失败时启用 |
| exact LLM/fit hard caps | TBD | production constants × 8 units | P0 算术冻结，不按结果调整 |
| Solar controlled corruption parameters | TBD | dev-only protocol | 开封前冻结 |

上述项未填完时，本文仍是 `DRAFT_FOR_FREEZE`，不得启动论文主实验 live run。
