# P4b 有限放宽 CONFLICT 门 · 前瞻性风险—效用策略实验（预注册，2026-08-31）

> **数据源勘误（2026-09-01 追加，不改动本文任何数字与判词）**
>
> 本文标注的数据集 `KDD Cup 2018 with missing values` **是错的**。实际使用的
> `data/kdd2018/series_cache.npz` 建自 `..._without_missing_values.tsf`，缓存内
> NaN 计数为 0；天然缺口（17.119%，270/270 条序列）已被上游填补消除。
>
> 逐值核验见 `artifacts/main_protocol/p4d_natural_gap_roster.json` 与
> `p4d_natural_gap_preflight.json`：两版本在 2,438,652 个观测位置上最大偏差
> `0.000e+00`。**本文数字在该（without）版本上测得正确，全部保留。**
>
> 但结论范围收窄为**无缺口的 outlier / level / denoise 场景**。identity 自身即
> `_linear_integrity` 线性插补，故全部 imputation 算子在本文数据上退化为恒等，
> 从未真正受考——本文任何负结论**不关闭 imputation 方向**。
>
> 含缺失版本记为独立数据身份 `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，
> 与本文结果平行、**不并表**。详见 `AGENTS.md` §8.1。


**状态**：`PREREGISTRATION_APPROVED_NARROWED` —— 已收缩为纯门实验（§0），
待真 backend 门排练（§7.3）通过后发车。
**实验标签**：`PROSPECTIVE_RISK_UTILITY_POLICY_EXPERIMENT`
**不覆盖**：`artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json`。
旧 P4 严格门下的 H1 = −0.01367 / H2 = −0.16506 / H3 = 0.0 按已收集状态成立，本实验
不改写、不重跑、不追认它们。

**上游依据**（三份已落盘的 0-LLM 诊断）：

- `docs/P4_CONFLICT_PER_SERIES_AUDIT_2026-08-31.md` §3：严格门在 11 个 Support 阶段
  单元上放弃了中位 +0.3040 的聚合增益，受害中位 4/20
- 同文 §5：受害序列集不稳定（20 条中 19 条至少受害一次）→ 按 series ID 的 Scope 修订不可行
- 同文 §8：部署可见特征分组 AUC 0.587，多个 origin 低于随机 → 提前预测受害序列不可行
- `artifacts/main_protocol/p4_bounded_risk_gate_calibration_20260831.json`：k=0 在所有
  m 下准入 0；Support-B 确认门使 realised 最小聚合增益恒 ≥ +0.0978，从不为负
- `artifacts/main_protocol/p4_source_treatment_empty_correction_20260831.json`：共享的
  审计 Source 卡在本研究任何 origin 上都不匹配 → 本实验收缩为纯门实验（见 §0）

既不能预测伤害、也不能永久放弃性能主张，剩下的动作是**给伤害设界**而不是禁止伤害。

---

## 0. 实验身份：这是门实验，不是积累实验

**发车前 preflight 查出的事实**（`artifacts/main_protocol/p4_source_treatment_empty_correction_20260831.json`）：

经审计的 Source 供给卡 `s2a_forecast_supply_v0` 的 `observable_applicability` 在本研究
的**任何 origin 上都不匹配**——held-in 0/8、held-out 0/8、旧 P4 origin 0/8。旧 P4 工件
逐单元核对一致：`A5-online` 与 `K0-fixed` 各 24 个单元**卡都在 store 里、被检索 0 次、
成为候选 0 次**。

卡本身没有错：它在不适用处保持沉默，这正是 Scope 应有的行为。但后果是
**跨域积累这个 treatment 在本批数据上是空的**，任何臂间对比都无法表达它。

因此本实验的身份收缩为：

> **只检验 bounded-risk 准入门对 Target-local 适应、Skill 形成与 held-out 性能的影响，
> 不检验跨域积累贡献。**

三条随之而来的约束：

1. **不换卡**。换一张能匹配的卡会把"Source 卡资格选择"混进"风险门实验"，重新制造
   本实验专门要避免的混淆，且需要一轮新的供给资格审计。
2. **不放宽现有卡的 Scope 来制造匹配**。那是伪造 treatment，不是发现 treatment。
3. **臂名沿用历史编码**。代码内仍写 `A5-strict` / `A5-bounded` 以与 P2–P4 对齐，
   **报告中一律按 Online-strict / Online-bounded 读**：两臂的历史卡在当前 Context 下
   均不适用，二者的唯一差别是准入门。

未来真正的 A5 积累实验必须先过 **Scope-match 非空门**：

> 至少一张预先审计的 Source Skill 在 **held-in 部署可见 Context** 上匹配 ≥ 1 次。

不过此门只能测"安全沉默"，不能测"积累收益"。该门已实装为
`run_forecast_p4b.preflight()` 的 `source_scope_census`（0 Consumer fit / 0 Outcome /
0 LLM），逐场落盘。

---

## 1. 冻结的 bounded-risk 准入规则

**准入（Target-local 部署权）** —— Support-A 与独立 Support-B **两个面都必须**满足
全部三条：

| 条件 | 阈值 |
| --- | --- |
| 聚合收益 | `aggregate_gain >= +0.005` |
| 受害比例 | `harmed_fraction <= 0.20`（受害 = per-series gain < −0.005） |
| 最大单序列损失 | `max_single_series_harm <= 0.30` |

**拒绝 / 撤销**：Support-B 聚合反号，或任一面风险越界。

**Scope 限制**：带局部冲突的正例**只取得 Target-local 部署权，不自动扩大 Shared
Skill Scope**。跨 Target 的推广仍需原有的独立证据阶梯。

### 1.1 参数口径（收紧项 1）

- 受害门写作**比例 ≤20%**，不写作"≤4 条"。当前 roster 为 20 条序列，20% 即 4 条；
  序列数变化时规则不失真。
- **`m = 0.30` 只对当前 Forecast Consumer（`pooled_ridge_a1`）与 sMASE 口径有效。**
  更换 Consumer、指标或标准化方式，`m` 必须重新校准，不得平移。
- 两个阈值来自 `p4_bounded_risk_gate_calibration_20260831.json` 的扫描：`m=0.30` 下
  realised 最坏单序列伤害 0.1343（为上限的 45%，留有余量）；`m` 放宽到 0.50 以上开始
  放进 0.40–0.79 的重损案例。校准显示**真正起保护作用的是 `m`，`k` 在有用的 `m` 值下
  接近冗余**——这一点写入预注册，避免事后误读为"放宽了受害条数"。

### 1.2 实现约束：不得改动 `classify_relation`

`experience_memory.classify_relation`（阈值 ±0.005）**保持原样**，理由有二：

1. `A5-strict` 臂必须与旧 P4 使用同一件仪器，否则 strict/bounded 对照不成立；
2. 严格门保留为安全基线，随时可回退。

bounded 门实现为**部署准入策略层**，读取同一份 per-series gain，独立判定部署权。
`classify_relation` 的 CONFLICT 判词继续照写进 Episode——证据记录不变，变的只是
执行权发放。

---

## 2. Support-B 的地位（收紧项 3）

Support-B 是 **held-in 的独立批准面**，不是部署性能。

批准后必须：**冻结 → 在未参与任何反馈的 held-out 窗口上一次性 Fast-only 计分**。
held-out 结果不回流、不触发修订、不用于重新选参。任何"Support-B 上 +0.4789"之类的
读数**不得**作为本实验的性能结论。

---

## 3. 机械 origin 选择（收紧项 4）

选择规则纯几何，**过程中不读取任何 Outcome**：

- 旧 P4 使用 `600 + 48k, k = 0..7` = 600…936，评估末端 **984**
- **origin 间距 = `CONTEXT_LENGTH + HORIZON` = 192 + 48 = 240**。旧 P4 用间距 48，
  相邻 origin 的 context 窗口重叠 75%（144/192），读数之间不独立；间距 240 使 origin
  `o` 的 context `[o−192, o)` 与前一 origin 的评估窗口 `[o−240, o−192)` 首尾相接而
  **不重叠**
### 3.1 固定等差块不可用——必须先做可行性筛选

原定的等差块（`3096 + 240k`）**会让实验中途中止**：KDD 是空气质量数据，部分序列的
context 窗口 `[origin−192, origin)` 近似恒定，forecast runtime 的 robust scale 塌到
下限后**拒绝该窗口**（`evaluation context reached scale floor`，
`run_e2_autonomous_natural_workflow_generation.py:723`）。实测 origin **3336 与 4536
不可评估**，且 **5016–6456 是一整段退化区**。

因此 origin 计划改为**由规则导出**，不写死数字：

> 在 **48 网格**上从起点向后贪心，取最早的**可评估**且与前一个至少相距
> `MIN_SPACING = 240` 的 origin，直到凑满 8 个。

**可行性筛选是结构性的、不读 Outcome**：它重算 runtime 会用的 centre/scale，输入只有
`[origin−192, origin)` 的观测历史——Fast Path 本就可见。不拟合 Consumer、不读评估
horizon、不读任何 Outcome，与"检查 roster 是否够长"同级。实现见
`evaluation/main_protocol_p4/p4b_viability.py`，`census()` 落盘
`outcome_reads / consumer_fits / llm_calls` 三个 0。

### 3.2 筛选后的实际计划

- **held-in**：`1176, 1416, 1656, 1896, 2136, 2376, 2616, 2856`，间距全部 240
  （8 个候选恰好全部可评估），评估末端 2904，context 起点 984 = 旧块末端
- **held-out**：`3096, 3576, 3816, 4056, 4296, 4584, 4824, 6600`，间距
  `480, 240, 240, 240, 288, 240, 1776`，评估末端 6648，context 起点 2904 = held-in 末端

间距不再等距，但**独立性依据的是"不重叠"而非"等距"**：所有间距 ≥ 240，任一 origin 的
context 与前一个的评估窗互不相交。

**需要注意的一点**：最后一个 held-out origin **6600 与其余相距 1776**，因为 5016–6456
整段退化。它仍是同一批序列的合法样本，但采到的是更晚的时段。若 Planner 认为该点不宜
与其余同表，替代方案是 held-out 只取 7 个（3096–4824），代价是 n 从 8 降到 7
（Wilcoxon 最小可达 p 由 0.0078 升至 0.0156）。**当前按 n = 8 冻结。**

roster 40 条序列长度均为 10898，几何上限 origin ≤ 10850，6648 < 10850，合法。

`frozen_contract.origins` 从 600..936 扩展到上述两块属**契约变更**，本预注册即为其
书面依据；旧 P4 的 origins 字段不改。

---

## 4. 同场对照臂

| 臂 | 门 | 写回 | 角色 |
| --- | --- | --- | --- |
| `A5-strict`（读作 Online-strict） | strict | 允许 | 旧策略同场重跑，strict/bounded 的直接对照 |
| `A5-bounded`（读作 Online-bounded） | bounded | 允许 | 主处理臂 |
| `Static` | — | — | 确定性参照（identity），0 LLM |
| `Parallel Best-of-N@8` | — | — | 确定性等预算搜索参照，0 LLM |

两臂**同起点、同写回、同状态携带，唯一差别是准入门**。

### 4.0 `K0-bounded` 与 `A3-bounded` 本轮不设（依 §0）

原计划以 `A5-bounded − A3-bounded` 作为"跨域积累的边际贡献"次指标，以 `K0-bounded`
分离"放宽门"与"在线写回"。§0 的事实使这两条都失去解释价值：

- `A5-bounded − A3-bounded`：两臂唯一差别是一条**永不触发的库存条目**，对照结构上为空。
- `K0-bounded`：其价值本在于与携卡臂的对照；卡不激活时，它退化为"不写回的 Online 臂"，
  而"写回效应"并非本实验的问题。

删去这两臂约省一半 held-in 成本。**代价如实记录：本实验因此不产出任何关于跨域积累
的读数**，该问题另案补做，见 §0 的 Scope-match 非空门。

### 4.0.1 Support-A / Support-B 是两组序列，不是两个时间面（2026-09-01 更正）

`support_b` 的 `delayed_token = origin + HORIZON` 是 dispatcher 的路由键，不是时间偏移。
两面读数都在同一 origin，差别是**序列组**：`run_forecast_p1.py:264-266` 将结构可读 UID
按字典序排序，`[:20]` 为 Support-A、`[20:40]` 为 Support-B。

因此本预注册中"独立 Support-B 确认"的独立性来自**不相交的序列组**，而非时间间隔；
凡把 B 面读作"延迟视界"的表述一律按此更正。切分平衡性复核见
`artifacts/main_protocol/p4c_split_and_headroom_check.json`。

### 4.1 held-out 上门不触发——臂的职责必须分开写

依 `run_e2_s2a_forecast_curriculum.py:502-560`，held-out 计分用的 `applied` 来自该臂
**held-in 已批准的 incumbent**，held-out 不再探测、不再取 Support。因此
**准入门在 held-out 根本不触发**，strict 与 bounded 在 held-out 的差异**完全来自
各自 held-in 学到了什么**。由此：

| 臂 | held-in 职责 | held-out 职责 |
| --- | --- | --- |
| `A5-strict` vs `A5-bounded` | 门效应 | **确证性主对照**（本实验唯一的主指标对比） |
| `Static` / `Parallel@8` | — | 确定性参照，次指标 |

**主指标的确证性对比是 2 臂。** `A5-bounded − Static` 与 `A5-bounded − Parallel@8`
按 AGENTS.md:63 一并报告，属次指标。

### 4.2 泄漏检查改为"写回必须经过门"

原计划以 `K0-bounded`"前后 store 逐字段相等"作零成本泄漏检查。该臂已删，且**本轮两臂
都写回**，"store 从未变动"不再是应有性质。检查改为等价强度的形式：

> **Skill store 的每一次变动都必须是被门支付过的**——该 cell 上存在 `admitted` 的
> probe，或产生了 approved Skill，或 incumbent 发生变更。

在没有任何准入的 cell 上出现 Skill 级变动，说明写回绕过了准入门，strict/bounded 对照
即失效。实装为 `p4b_heldin.gated_writeback_check`，逐字段结构化比对，
**不新增 SHA、不新增 manifest、不增加臂**。不满足即判 `LEAKAGE_SUSPECTED`。

---

## 5. 指标（收紧项 5）

### 5.0 统计单位 = 8 个 origin，不是 24 个读数

3 个 replica 跑的是**同一批 8 个 origin**，差别只在 LLM 采样，因此 24 个读数不是 24
个独立样本。聚合口径冻结为两步：

1. **先在每个 origin 内对 3 个 replica 取平均**（replica 方差按 origin 内噪声处理）
2. **再对 8 个 origin 做配对比较**

主统计 `n = 8`。检验预注册为 **配对 Wilcoxon 符号秩检验，双侧，α = 0.05**，并同时
报告 **按 origin 聚类的 BCa bootstrap 95% 置信区间**（10,000 次重抽，重抽单位 = origin）。

**功效如实声明**：n = 8 时 Wilcoxon 双侧可达的最小 p 为 `2/2^8 = 0.0078`，且
`p < 0.05` 要求 `W ≤ 3`——即 8 个 origin 中至多一个可以反向且幅度很小，实际上要求
接近一致的符号。**本实验没有能力检出小效应**；若结果落在 `BOUNDED_GATE_NEUTRAL`，
判词含义是"本设计分辨不出"，不是"效应为零"。

**主指标 —— 性能—风险，全部在 held-out 上一次性计分，对比为
`A5-bounded − A5-strict`，配对到 8 个 origin：**

1. held-out 平均 utility（Δ vs identity）
2. held-out 材料级 harm rate（单元聚合层）
3. held-out **最坏单序列损失**（逐序列层）

三项均由 `_score_heldout` 已有的 `heldout_gain` / `heldout_per_series_gain` /
`worst_series_gain` / `harm_event` 直接给出，无需新指标实现。

**次指标 —— 机制：**

4. `A5-bounded` 是否形成 Skill（条数、revision 链、版本号递增）
5. 该 Skill 是否在**后续 origin 被因果复用**——判定沿用
   `run_forecast_p4_evolution_revision.py:1235-1242` 的因果门：
   `OLD_SKILL_ID ∈ retrieved` ∧ `source_skill_of_candidate(chosen_id) == OLD_SKILL_ID`
   ∧ 候选程序步与卡一致 ∧ 探测已执行。"卡在池里"不算复用。
6. `A5-bounded` 相对 `Static` / `Parallel@8` 的 regret 或成本下降

**本实验不产出跨域积累的次指标**（§0 / §4.0）。工件中
`analysis.accumulation_contrast.reported = false`，并附不产出的理由。

**主指标优先。** 若主指标为负而次指标为正，判词写"机制成立、性能未兑现"，
**不得**把实验改述为 RQ3 证明。

---

## 6. 预算

- 每 cell 沿用 B=8：`full_support_evaluations 8`（Support-A ≤7 / Support-B ≤1）、
  `llm_call_max 8`、`token_max 60000`、`cheap_probe_max 24`、`accepted_update_max 1`
- held-in：2 个自适应臂 × 8 origin × 3 replica 顺序 = **48 cell（唯一消耗 LLM 的部分）**
- held-out：`_frozen_recall`（`run_e2_t6_cls_op_shared_harness.py:1406`）是纯确定性
  检索——`resolve_harness_view` + `_parse_frozen_steps` + 字典序取首，**不调 LLM、
  不读 Outcome、不写入**；退化路径是 incumbent，再退化是 identity。因此
  **held-out 计分 0 LLM**，48 个 held-out cell 不进 LLM 预算
- **全局 LLM 上限 = `48 × 8 = 384` 次**（上限，非预期开销）。旧 P4 三臂实际 339 次 /
  3h18m，按同等 cell 占用率外推本实验约 **225 次 / 2–3 小时**
- `Static` 与 `Parallel@8` 为 0 LLM
- 预算耗尽的行为沿用 `ABSTAIN_TO_IDENTITY_AND_CONTINUE`

### 6.1 `Parallel Best-of-N@8` 的选择必须只发生在 held-in

`Parallel@8` 在 **held-in** 上用 8 次完整评估搜索候选、选出最优程序，**冻结该程序**，
再在 held-out 上原样部署。

**绝不允许在 held-out 上搜索或选择最优候选。** held-out 只执行冻结程序、只计分一次。
违反此条即 held-out 被污染为选择面，实验作废。Runner 必须落盘
`parallel_selection_face = "held_in"` 与冻结的程序步，供事后核验。

---

## 7. 记录要求（本次必须补上）

旧 P4 终态工件**没有保存 probe 的 `per_view_gain`**，正是这一点使得逐序列诊断必须
事后重算。本实验的 Runner **必须**对每次 probe 落盘：

- `per_series_gain`（完整向量）
- `harmed_count` / `harmed_fraction` / `max_single_series_harm`
- 准入判定结果与被触发的具体条件
- 候选 `source_skill_id`（供因果复用判定）

这也是敏感性分析可行的前提：**k=1 / m=0.10 的保守敏感性分析必须能对同一批已收集
数据离线重读得出，不得为它再跑一轮。**（收紧项：不再开展调参轮次。）

### 7.1 实现面：bounded 门不是配置项，需要改共享代码

`ExplorationPolicy`（`methods/ttha/exploration_policy.py:40`）是冻结 dataclass，字段
经 `LEGAL_DOMAINS` 封闭校验；它管的是探测顺序与 winner 比较，**没有"部署准入"这一
维**。真正发放执行权的判断是 `online_loop.py:549` 的 `if str(ep.relation) ==
"POSITIVE"`，**硬编码**。

**不得**把准入规则加进 `ExplorationPolicy`：该模块 docstring 明写"harm 阈……不在
本面内"，且 `LEGAL_DOMAINS` 是 **Stage-3 Random-legal-edit 臂的采样空间**——把安全门
放进去，等于允许一次随机合法编辑翻转安全门。

因此实现形态冻结为**独立模块** `methods/ttha/admission_policy.py`（已实现）：

- `AdmissionPolicy(rule, max_harmed_fraction, max_single_series_harm)`，
  `DEFAULT = strict_positive_only`；`install_policy` / `reset_policy` / `decide`
- **刻意不进 `LEGAL_DOMAINS`**，随机编辑臂够不着（由
  `tests/functional/test_admission_policy.py` 断言）
- `bounded_risk_v1` **fail closed**：拿不到 per-series 读数时按 strict 拒绝，不盲准

`classify_relation` 与 Episode 的 CONFLICT 判词**不变**（§1.2）。

**共有四处内联 `relation == "POSITIVE"` 必须一起参数化**，少改任何一处实验都会失效
——前两处在回路层，后两处在方法层（持久化）：

| 层 | 位置 | 作用 | 少改的后果 |
| --- | --- | --- | --- |
| 回路 | `online_loop.py` winner 形成 | 发放本轮执行权 | 门没放宽 |
| 回路 | `online_loop.py` `_write_target_episode`（`accepted` / `LOCAL_DRAFT`） | Support 时保留 | 赢下该轮却只留 Episode |
| 回路 | `online_loop.py` `_update_delayed_status`（`LOCAL_ACTIVE` / `RESTRICTED`） | delayed 状态 | 预算内的 delayed CONFLICT 仍被撤销 |
| **方法** | `method.py` `handle_fast_winner`（`support_rejected`） | 形成 **pending** | **无法持久化为 Skill** |
| **方法** | `method.py` `handle_feedback_delayed`（`delayed_rejected`） | **批准 / snapshot 更新** | **pending 被丢弃，下一 origin 无 Skill 可复用** |

方法层两处是 §1 规则真正落地的地方：只改回路层会得到"候选获准 → 本轮执行 → 方法层
拒绝持久化 → 下一 origin 没有 Skill 可复用"，即**性能可能变化但 RQ3 依旧不可达**。
方法层读 `classify_relation` 的 facts 摘要（`aggregate_gain` / `series_read` /
`harmed_series_count` / `min_per_series_gain`），与回路层用同一套数字判定。

`handle_feedback_delayed` 同时就是 §1"独立 Support-B 确认"的落点：越界或反号仍然
拒绝并丢弃 pending，预算内的才批准。

### 7.1.1 非有限值一律 fail closed

`NaN` / `±Inf` 不是读数：聚合非有限 → `non_finite_aggregate_fail_closed`；任一逐序列
读数非有限 → 整份读数作废 `no_per_series_reading_fail_closed`（不对空洞取平均）。
两条判定入口（`decide` / `decide_from_facts`）行为一致，由
`tests/functional/test_admission_policy.py` 参数化断言。

### 7.1.2 本轮 `allow_slow = False`

Slow 的 Support 门（`handle_feedback_support`）**本轮不统一**，因为 P4b 不启用
Slow-generated Patch。Runner 必须以 `allow_slow=False` 运行，并在工件中落盘该字段。
若后续要启用 Slow，必须先把 Slow Support 门接到同一个 `admission_policy`，否则
bounded 与 strict 在 Slow 路径上不可比。

### 7.1.3 Runner 必须 try/finally 恢复 DEFAULT

`install_policy` 是进程级全局。Runner 每个臂必须

```python
admission_policy.install_policy(...)   # 仅 bounded 臂
try:
    ...run the arm...
finally:
    admission_policy.reset_policy()
```

否则 `A5-strict` 会串到上一个臂装的 bounded 规则上，strict/bounded 对照直接作废。

### 7.2 回归等价性 preflight（已执行，PASS）

`python -m evaluation.main_protocol_p4.preflight_strict_equivalence`
→ `artifacts/main_protocol/p4b_preflight_strict_equivalence.json`

比对的是**Support 时重算的 relation**，不是工件里 `episodes_written.relation` ——
后者是 delayed 之后的最终值（origin 696 的探测在 Support-A 上是 POSITIVE、取得部署权，
之后才被 delayed 改判为 CONFLICT/RESTRICTED），门读的是前者。

结果：16 个独特 (origin × operator) 单元，重算 Support-A 聚合与工件记录**全部吻合**；
strict 准入与原内联谓词**逐单元相等**；strict 仅在 **origin 696** 准入，而三个臂 ×
三个 replica 实际部署的 origin 恰好只有 696。判词 `STRICT_EQUIVALENCE_PASS`。

此 preflight 为发车前置条件，代码改动后必须重跑。（比对限于确定性部分；LLM 采样造成
的候选差异按已记录候选集回放，不重新调用 provider。）

### 7.3 真 backend 门排练只能在已曝光的旧 origin 上做

单元测试覆盖了准入规则本身与完整生命周期，但 scripted backend 只产 identity，
**门在 Runner 里从未被真实候选触达过**。发车前需要一次真 backend 的小规模排练，
其边界冻结如下：

- **只用旧 P4 origin（600–936）**。这批 origin 已在旧 P4 中曝光，排练不消耗
  held-in / held-out 任一冻结块的新鲜度。
- **不跑 held-out**：排练无终点面，`held_out_origins = []`，Parallel 选择跳过。
- **另写工件** `artifacts/main_protocol/p4b_live_gate_smoke.json`，
  `evidence_grade = SMOKE_NOT_EVIDENCE`、`frozen_blocks_touched = false`。
- 只验证这条链路：真实 Agent 提案 → bounded 准入 → pending → Support-B 批准 →
  Skill 持久化 → 下一 origin 可被检索。

命令：`python -m evaluation.main_protocol_p4.run_forecast_p4b --backend live
--old-origin-smoke --origins 2 --replicas 1`

**排练不构成任何科学结论**，其数值不进主表、不进论文。

**当前状态：`PASS`（2026-08-31）。** 工件 `p4b_live_gate_smoke.json`，
`status = COMPLETE`、`evidence_grade = SMOKE_NOT_EVIDENCE`、`frozen_blocks_touched = false`。

在 origin 648 上两臂拿到**逐序列增益向量完全相同**的程序（`support_per_view_gain`
20 维逐位相等，聚合同为 +0.261546），因此该点的分歧是**纯门效应**，不是候选采样差异：

| 臂 | 关系 | 门 | 结果 |
| --- | --- | --- | --- |
| `A5-strict` | CONFLICT | `strict_positive_only` 拒绝 | `EPISODE_ONLY`，无批准、store 不变 |
| `A5-bounded` | CONFLICT | `bounded_risk_v1` 准入（受害 4/20 = 0.20、最大单序列 0.2003，两项均在预算内） | winner → Support-B 延迟确认 +0.4200（同号、未反转）→ `LOCAL_ACTIVE` → Skill 批准并激活、incumbent 变更 |

链路已验证到 **Skill 持久化**为止。**"下一 origin 可被检索"这一环未验证**——批准发生在
排练的最后一个 origin（648），其后没有 origin 可供检索。正式发车时 8 个 origin 串联会
自然覆盖该环。

两处由此次排练发现并已修正的问题记于 §7.4。

**transport 变更**：本次及之后使用 `https://api.nowaterapi.xyz/v1` 的 `gpt-5.6-sol`；
旧 P4 用的是 `api.agicto.cn` 的 `gpt-5.6-luna`。Runner 现将 base_url / model 落盘。
P4b 内部两臂同场同模型，主对照不受影响；**与旧 P4 的跨场数值不并表**。

### 7.4 排练发现的两处问题（已修）

1. **判词阶梯会对空对照发出实质判词**。held-out 为空时 `primary_utility.origins = []`，
   阶梯一路落到 `BOUNDED_GATE_NEUTRAL`，把"从未做过的比较"报成"未检出差异"。
   已在风险门之后、显著性之前插入 `NO_ENDPOINT_DATA`（blocking）。
   排练工件的 analysis 已用修正后的阶梯**离线重算**（不重跑），
   `analysis_recomputed_offline` 记录原判词。
2. **逐序列记录的落点**：probe 记录里 strict 拒绝的候选 `series_count` 等为 null
   （strict 在 relation 处短路）。完整的 20 维 `support_per_view_gain` /
   `delayed_per_view_gain` 保存在 `episodes_written` 上，两臂、含被拒候选均有。
   §7 的离线敏感性重读因此可行——读 `episodes_written`，不读 `probes`。

---

## 8. 停止规则与冻结判词

**held-out 结果打开后，`k` 与 `m` 一律不得调整。** 以下判词在开盘前即冻结：

| 情形 | 判词 |
| --- | --- |
| 某 cell 上无任何准入却发生 Skill 级 store 变动（§4.2） | `LEAKAGE_SUSPECTED` —— **实验作废**，先查泄漏，不解释主指标 |
| `Parallel@8` 在 held-out 上发生过选择 | `HELDOUT_CONTAMINATED` —— **实验作废** |
| held-in 无任何候选获准入 | `BOUNDED_GATE_STILL_BLOCKING` —— 收口，不再放宽 |
| 有准入但无因果复用 | `NO_CAUSAL_REUSE` —— 次指标不成立，主指标照报 |
| held-out 出现风险越界（最坏单序列损失 > 0.30 或 harm rate 超预算） | `RISK_BUDGET_BREACHED` —— 收口，严格门恢复为默认 |
| 主指标 bounded − strict 显著为正且无越界 | `BOUNDED_GATE_POSITIVE` |
| 主指标不显著 | `BOUNDED_GATE_NEUTRAL` —— 据实报告，不追加轮次 |

任一收口判词出现后，**不继续在这批数据上追结果**。

---

## 9. 本实验不主张什么

- 不主张 held-out 之外的任何面上的性能
- 不主张跨 Consumer / 跨指标的 `m` 可迁移性
- 不主张 Shared Skill Scope 扩大
- 不主张旧 P4 负结果被修正或替代（仅就归因作一条事实更正，见 §0 与
  `p4_source_treatment_empty_correction_20260831.json`；旧数值原样保留）
- **不主张任何跨域积累收益**：本批数据上该 treatment 为空（§0）
- Natural Final、Query、UCR TEST、sealed AD 全程保持关闭，读取计数为 0
