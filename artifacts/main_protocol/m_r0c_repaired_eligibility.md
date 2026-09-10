# M-R0c · 修复后验收资格补表（as-run / repaired 分列 + 完整 Scope 后续覆盖 + 链时序）

**stage** `M_R0C_REPAIRED_ELIGIBILITY` · **evidence grade** `INSTRUMENT`
**机器可读工件** `artifacts/main_protocol/m_r0c_repaired_eligibility.json`
**复算** `python -m evaluation.main_protocol_p4.audit_m_r0c_repaired_eligibility`

**边界**：0 外部 LLM、0 新 Consumer fit、0 held-out 读、0 密封读、**0 次 +144 评价面参与选择或资格判定**、
0 生产代码/阈值/预算/候选空间/信息墙改动、0 既有工件覆盖、0 新增 SHA。未提交、未发车。

**承接**：M-R0b 的 as-run 结果与 3/3 历史影子复现**原样保留、未被取代**；本表只在其上加三列。

## 一、as-run 与 repaired 分列（全部 96 条，非只重算旧的 7 个）

| 量 | 值 |
| --- | --- |
| 步骤 × 谱系记录 | 96 |
| repaired 路径**实际发出**候选 | 26 |
| — 含**可提子句**分支 | 9 |
| — 含**拒绝**分支 | 18 |
| — **同时含两个分支** | **1** |
| 其中影子有可行子句 | 7 |
| 其中通过既有 narrowing 合法性检查 | 7 |
| **两种口径都真正搜索过**的记录 | 9 |
| 其中影子结果**确实不同** | **3** |

### 统计口径更正（互斥口径）

`26 = 9 + 18 − 1`。「含可提子句」与「含拒绝」**不互斥**，任何一方都不能写成另一方的「其余」。
本文件先前写的「17 条在 propose 阶段被拒」是用 `26 − 9` 相减得到的，属**非互斥口径，已作废**；
准确说法是：**含拒绝分支的记录 18 条**，其中 1 条同时含可提子句分支。

**重叠点唯一并已定位**：`reverse · A5-online · k5 · outlier_mad`。该步同一谱系并存两张 Draft ——
`resupplied_draft_1`（FLAGGED、3 次验证用尽、已关闭）与 `resupplied_draft_2`（REVISABLE、1 次验证）。
NARROW 对着已关闭的那张被拒（`LINEAGE_CLOSED`），仍开放的那张被作为 `REVISE` 提出，于是一条
记录同时带两个分支。该记录的可提子句分支实际停在 `NO_FEASIBLE_STUMP`。

阻断分布相应改为按**仍存活的那个分支**归类（先前的 elif 顺序会把它埋进拒绝桶）：
`LINEAGE_CLOSED` 由 14 改为 **13**，`NO_FEASIBLE_STUMP` 由 1 改为 **2**。

**这不影响首选入口 k1**：forward · A5-online · k1 当时只有一张 Draft、只发出一个 `REVISE` 分支、
无拒绝分支，不涉及本重叠。

> 96 条中另有 70 条 repaired propose 根本未发出候选，没有可搜对象；这不计作「影子差异」——
> 那是「没搜」，不是「搜出不同结果」。

**去重确实会把原本不可行的记录变成可行，也会改变选中的子句**。最明确的一例是
forward · A5-online · `outlier_mad` · k2：

| 口径 | 搜索行数 | 结论 | 子句 | 可行 stump 数 |
| --- | --- | --- | --- | --- |
| as-run | 220 | `BEST_STUMP` | `longest_missing_run_fraction >= 0.2` | **1** |
| repaired | 160 | `BEST_STUMP` | `local_robust_z_peak >= 6.0` | **4** |

差异原因：精确去重去掉 60 行重复观测后，同一序列不再被双重加权，可行 stump 从 1 增到 4，
最优子句随之易主。**不要求 repaired 复现旧子句**，这里也确实没有复现。

**阻断分布（96 条）**：

每条记录归入且仅归入一个桶；同时带两个分支的那一条按仍存活的分支归类。

| 阻断 | 数 |
| --- | --- |
| repaired propose 未发出候选 | 70 |
| propose 阶段拒绝：`LINEAGE_CLOSED` | 13 |
| propose 阶段拒绝：`REVISION_TARGET_FLAGGED` | 4 |
| 完整 Scope 下没有后续单元过覆盖线 | 3 |
| 影子无可行子句 `NO_FEASIBLE_STUMP` | 2 |
| 只剩一个后续单元，无链时序 | 1 |
| **完整链时序机会** | **3** |

合计 70+13+4+3+2+1+3 = 96。

## 二、修改后的**完整 Scope** 能否在后续匹配

完整 Scope = 候选自身的 base/current 谓词 **＋** 影子子句。覆盖用生产解析器
（`UnitContext.resolve`）在 Support origin 与 delayed origin（+48）上计算，**只用部署可见
Context**，取自已授权 development 数据。**没有**用「旧程序后来部署过且 treated≥5」当替代——
收窄后的谓词解析出的是另一组序列，必须自己过线。

候选由当时可见证据选出（影子只搜该步 bank）；后续 Context 只做资格检查，**未**据未来覆盖或
效用重挑子句。**+144 评价面全程未参与**。

### Support 侧没有覆盖门，本表也没有新增一道

覆盖资格**只**判 delayed 面：`MIN_TREATED = 5` 是 delayed 权威门里的 coverage floor
（`contract.RISK["min_treated"]`）。**Support admission（`admission_policy.decide`）根本没有
`treated >= N` 这个条件**——strict 看 relation 是否 POSITIVE，bounded 看 harmed_fraction 与
single_series_harm，两者都不看被处理序列的条数。

因此表中的 `treated_on_support` 只是**信息栏**，不是资格条件，工件里对应字段写明
`support_side_floor_applied: false`。像 p15 `[40:80]@3816` 的 `sup 3 / del 6`，其
**入选依据只有 del 6 ≥ 5**；`sup 3` 既不构成通过，也**不**构成不通过。Support 是否准入是
`support_admission_outcome: NOT_EVALUATED`，必须由真实评价给出。

## 三、一次验证机会 ≠ 完整链时序机会

| 量 | 值 |
| --- | --- |
| 有后续合法验证单元（完整 Scope 下过覆盖线） | **4** |
| 其中该验证之后**还有**另一个可重遇单元 | **3** |
| 去重谱系（有验证单元 / 有完整链） | 2 / 2 |
| 去重身份（有验证单元 / 有完整链） | 2 / 2 |

七个可行且合法的候选，逐条：

| 顺序 · 臂 · k | 程序 | 完整 Scope 下过线的后续单元 | 验证单元 | 其后可重遇 | 完整链 |
| --- | --- | --- | --- | --- | --- |
| forward · A5-online · k1 | `outlier_mad` | 7 / 21 | p06 | 6 个 | **是** |
| forward · A5-online · k2 | `outlier_mad` | 5 / 16 | p13 | 4 个 | **是** |
| reverse · A5-online · k2 | `winsorize` | 2 / 16 | p11 | 1 个 | **是** |
| reverse · A5-online · k3 | `winsorize` | 1 / 11 | p11 | 0 | 否（只剩一个后续单元） |
| reverse · A5-online · k4 | `winsorize` | 0 / 6 | — | — | 否（无单元过线） |
| reverse · A5-online · k5 | `winsorize` | 0 / 1 | — | — | 否 |
| reverse · A3-online · k5 | `winsorize` | 0 / 1 | — | — | 否 |

**覆盖资格不等于验证通过，也不等于 Support 准入。** 上表的「过线」只指 delayed 面的
coverage floor；真实 Slow 是否提出该子句、Support 是否准入、replay 屏、后续验证结果、重遇收益，
全部 `NOT_EVALUATED`；记录不足处标 `UNKNOWN`。

## 四、单列：forward · A5-online · `outlier_mad@local_robust_z_peak>=3`

### k1（边界 p04）—— 资格最完整的一条

| 项 | 值 |
| --- | --- |
| Active 祖先 | **有**（K0 卡，课程起始注入） |
| 当时证据 | bank 行 6 → 去重后 5 观测；unit_votes `{CONFLICT: 5}`；`adverse_units=5` |
| 当时 Draft | `resupplied_draft_1`：`REVISABLE`，`verification_attempts=1`，未关闭；`resupplied_draft_2` 尚未存在 |
| repaired 发出 | `REVISE`，作用于 `resupplied_draft_1`（不新壳、计数不变） |
| 影子（as-run / repaired） | 120 行 / 100 行，同得 `local_robust_z_peak>=6.0`，可行 8 |
| 完整 Scope | `z_peak>=3.0` **∧** `z_peak>=6.0` |
| 合法性检查 | `accepted=True`，`total_added_since_root=1`，四项 checks 全 True |
| 后续可评单元 | 21；完整 Scope 下过覆盖线 **7** |
| 过线单元（**入选只看 delayed**；Support 数仅供参考） | p06 `[0:40]@3576`(sup 20 / **del 20**)、p09 `[40:80]@1656`(7 / **13**)、p13 `[40:80]@2856`(7 / **5**)、p14 `[40:80]@3576`(20 / **20**)、p15 `[40:80]@3816`(3 / **6**)、p21 `[80:120]@2616`(2 / **5**)、p22 `[80:120]@2856`(6 / **7**) |
| 验证单元 | **p06**（delayed treated 20） |
| 其后可重遇 | **6 个**（p09、p13、p14、p15、p21、p22） |
| **完整链时序机会** | **是** |

### k2（边界 p09）—— 同一谱系的后续状态

同一条 `resupplied_draft_1`，仍 `REVISABLE`、`verification_attempts=1`、未关闭；证据长到
11 行 → 8 观测，`unit_votes {CONFLICT: 7, NEGATIVE: 1}`。影子在两种口径下分歧（见上表），
repaired 得 `z_peak>=6.0`（可行 4）。完整 Scope 相同；合法性通过；后续 16 个单元中 5 个过线，
验证单元 p13，其后 4 个可重遇 → 仍是完整链。

**注意**：k1 与 k2 是**同一条谱系的两个时点**，不是两次独立机会。去重后 forward 这条只算一条。

## 五、成本估算（估算，不构成授权）

三条完整链各自的最小真实验收成本，分列如下。常量取自 runner
（`FITS_PER_SCORED_FACE=3`、`CACHE_FITS_PER_CELL=2`、Support 每探针 2 fits、
`OUTER_LLM_PER_STEP=2`、Fast 每格上限 5）。

| 项 | forward k1 | forward k2 | reverse k2 |
| --- | --- | --- | --- |
| 外环 Slow 物理调用 | 每步上限 2，一次尝试 1 次调用 | 同 | 同 |
| — 一次**完整**提议的调用数 | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** |
| replay 屏 fits | 5 cell × 2 = **10** | 10 × 2 = **20** | 10 × 2 = **20** |
| 验证单元 fits | delayed 面 3；Support 2/探针 | 同 | 同 |
| — 该单元探针数 | `NOT_EVALUATED` | `NOT_EVALUATED` | `NOT_EVALUATED` |
| 重遇单元 | 可用 6 个，每面 3 fits | 可用 4 个 | 可用 1 个 |
| 祖先影子对照 fits | 7 过线单元 × 2 面 × 1 = **14** | 5 × 2 = **10** | 2 × 2 = **4** |
| 共享 baseline | 每单元 1 次评估，**计数不定价**（R3 未裁） | 同 | 同 |
| Fast 路径 LLM | 每格上限 5，涉及 7 格 | 5 格 | 2 格 |
| — 实际需要的调用数 | `NOT_EVALUATED` | `NOT_EVALUATED` | `NOT_EVALUATED` |

**无法确定的成本（明确列出）**：
1. 一次完整结构提议的物理调用数 —— 课程中从无提议到达 `CALIBRATED`（M-R0），从未测得；
2. 每个涉及单元会取几个 Support 探针 —— 取决于 Fast 在该单元提什么，本表不决定；
3. 任何一环是否通过 —— 不改变成本，但改变成本买到什么。

**缓存节省单列，不抵扣任何上限**：replay 屏的每格成本已是缓存价（2 而非 3），不再假设额外节省；
共享 baseline 一次评估服务该单元上所有臂。二者均**不**从审批上限中扣除。

## 六、新增产物

| 文件 | 性质 |
| --- | --- |
| `evaluation/main_protocol_p4/audit_m_r0c_repaired_eligibility.py` | 新增补充脚本，复用 M-R0 / M-R0b |
| `artifacts/main_protocol/m_r0c_repaired_eligibility.json` | 新增 |
| `artifacts/main_protocol/m_r0c_repaired_eligibility.md` | 本文件，新增 |

`m_r0_reachability.*` 与 `m_r0b_revision_opportunity.*` 未被覆盖。M-R0b 复用中新增了一个
`UnitContext` 缓存访问点（`_unit_ctx`），行为等价，其自身产物未重算。

## 七、本轮新发现的阻断（只报告，未扩大修复）

1. **完整 Scope 的覆盖是最紧的一环**：7 个合法候选里，3 个在收窄后的谓词下**没有任何**后续单元
   过覆盖线（reverse k4/k5、reverse A3 k5），1 个只剩一个（reverse k3）。收窄天然减覆盖，
   而 `MIN_TREATED=5` 是硬门；这在旧口径下被「旧程序后来 treated≥5」掩盖过。
2. **`LINEAGE_CLOSED`（13）已超过 `FLAGGED`（4）成为 propose 阶段的首位拒绝原因**：按当时状态，
   多数步骤上该谱系的 Draft 已因三次验证用尽而关闭。这与「课末 10/11 FLAGGED」是不同的读数。
   （按互斥口径 13；另有 1 条含拒绝分支的记录按其仍存活的分支归入 `NO_FEASIBLE_STUMP`，
   故「含拒绝分支的记录」总数为 18。）
3. 70 / 96 记录 repaired propose 根本不发候选 —— 主要是 `positive_units` / `adverse_units` 未达门，
   或该键不在当时 Active 集内。此项未细分，属既有门槛而非新缺陷。

补表完成，停止。
